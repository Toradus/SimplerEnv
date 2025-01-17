from collections import defaultdict
import functools
import os
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from transforms3d.euler import euler2axangle
import torch.nn as nn
import tensorflow_hub as hub
from transformers import CLIPTokenizer
from hydra import compose, initialize
import hydra
from safetensors.torch import load_file  # Add this import

# import Accelerator
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_model
# from agents.utils.ema import ExponentialMovingAverage
from flower.agents.utils.diffuser_ema import EMAModel
# from flower.agents.input_encoders.goal_encoders.language_encoders.clip_tokens import TokenLangClip
from flower.agents.lang_encoders.florence_tokens import TokenVLM
from flower.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower.dataset.utils.frequency_mapping import DATASET_FREQUENCY_MAP
from flower.agents.utils.action_index import ActionIndex
POLICY_SETUP_TO_DATASET_INDEX = {
    "widowx_bridge": 0,
    "google_robot": 7,
}

class UhaInference:
    def __init__(
        self,
        saved_model_base_dir: str = "/home/reuss/code/flower_vla_policy/logs/runs/2025-01-15/",
        saved_model_path: str = "15-23-08/checkpoint_5000",
        image_size: int = 224,
        pred_action_horizon: int = 10,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.lang_embed_model = TokenVLM("microsoft/Florence-2-large")
        
        self.image_size = image_size
        self.action_scale = action_scale
        # ------------------------- #
        model_path_split = saved_model_path.split("/")
        weights_path = saved_model_base_dir + model_path_split[0]
        checkpoint_path = os.path.join(weights_path, model_path_split[1])
        file_path = os.path.dirname(os.path.abspath(__file__))
        weights_path_relative = os.path.relpath(weights_path, file_path)
        with initialize(config_path=os.path.join(weights_path_relative, ".hydra")):
            cfg = compose(config_name="config")
            current_path = os.getcwd()
        ema_path = "random_states_0.pkl" # "best_test_loss_model_ema_state_dict.pth" # "7000_model_ema_state_dict.pth" # "ema_50000.pth" # "model_ema_state_dict.pth"
        cfg.batch_size = 1
        cfg.trainer.agent.agent.act_window_size = 10 # since we are doing single arm delta eef with 3 hz 
        cfg.trainer.agent.agent.multistep = 5 # since we are doing single arm delta eef with 3 hz
        cfg.trainer.agent.agent.num_sampling_steps = 10
        agent = hydra.utils.instantiate(cfg.trainer.agent, device=device, process_id=0)
         # In __init__:
        pre_load_hash = get_model_hash(agent)
        # Use in your UhaInference __init__:
        print("\nAnalyzing initial model state...")
        pre_load_details = check_model_weights(agent)
        # Initialize accelerator for loading
        accelerator = Accelerator()
        agent = accelerator.prepare(agent)

        model_path = os.path.join(checkpoint_path, "model.safetensors")
        missing, unexpected = load_model(agent, os.path.join(checkpoint_path, "model.safetensors"))
        print(missing)
        print(unexpected)
        # load weights from accelerator
       # Load weights
        '''model_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(model_path):
        
            # Convert device to string format for safetensors
            device_str = 'cuda' if device.type == 'cuda' else 'cpu'
            # Load with string device specification
            state_dict = load_file(model_path, device=device_str)
            
            # Remove any proprio encoder keys
            state_dict = {k: v for k, v in state_dict.items() 
                        if not k.startswith("agent.proprio_encoders")}
            
            # Load state dict with strict=False
            agent.load_state_dict(state_dict, strict=False)
            print("Loaded model weights successfully")

        missing, unexpected = [], []
        '''
        # Load EMA weights if they exist
        ema_helper = EMAModel(
            parameters=agent.parameters(),
            decay=cfg.decay,
            min_decay=0.0,
            update_after_step=0,
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=2/3,
            foreach=False,
            model_cls=type(agent),
            model_config=agent.config if hasattr(agent, 'config') else None
        )

        ema_path = os.path.join(checkpoint_path, ema_path)
        if os.path.exists(ema_path):
            ema_state = torch.load(ema_path, map_location=device)
            ema_helper.load_state_dict(ema_state)
            print("Loaded EMA weights successfully")
            ema_helper.copy_to(agent.parameters())

        # we cannot sue prioprio for isngl arm only meant for bimanual for now 
        agent.agent.use_proprio = False
        agent.to(dtype=torch.bfloat16)
        agent.eval()

        # ... load weights ...
        post_load_hash = get_model_hash(agent)

        if pre_load_hash == post_load_hash:
            print("WARNING: Model hash unchanged after loading weights!")
        else:
            print("Model parameters changed after loading weights")
                
        # Check if number of loaded parameters matches expected
        if len(missing) > 0:
            print(f"\nWARNING: {len(missing)} parameters were not loaded:")
            print("\n".join(missing))
        
        if len(unexpected) > 0:
            print(f"\nWARNING: {len(unexpected)} unexpected parameters were found:")
            print("\n".join(unexpected))

        print("\nAnalyzing model state after loading weights...")
        post_load_details = check_model_weights(agent)

        weights_loaded_successfully = compare_model_states(pre_load_details, post_load_details)

        if not weights_loaded_successfully:
            print("\nWARNING: Potential issues detected with weight loading!")
            print("Missing parameters:", missing)
            print("Unexpected parameters:", unexpected)


        pred_action_horizon = 5
        # ------------------------- #
        self.agent = agent
        self.pred_action_horizon = pred_action_horizon
        self.device = device
        self.observation = None
        self.task_description = None
        self.task_description_embedding = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.sticky_gripper_num_repeat = 15
            # use fractal20220817_data norm values
            self.max_values = torch.tensor([0.17824687153100965, 0.14938379630446405, 0.21842354819178575, 0.5892666035890578, 0.35272657424211445, 0.44796681255102094]) # p99
            self.min_values = torch.tensor([-0.22453527510166169, -0.14820013284683228, -0.231589707583189, -0.3517994859814644, -0.4193011274933815, -0.43643461108207704]) # p01
            self.format_instruction = functools.partial(
                generate_policy_prompt,
                robot_name="XARM",
                num_arms="1", 
                action_space="Delta End-Effector",
                prompt_style="minimal"
            )
        elif self.policy_setup == "widowx_bridge":
            self.sticky_gripper_num_repeat = 1
            # use bridge norm values
            self.max_values = torch.tensor([0.028122276067733765, 0.040630316659808145, 0.03994889184832546, 0.08121915772557152, 0.07724379181861864, 0.20214049845933896]) # p99 # 1.0
            self.min_values = torch.tensor([-0.028539552688598632, -0.041432044506073, -0.025977383628487588, -0.08020886614918708, -0.09213060349225997, -0.2054861941933632]) # p01 # 0.0
            self.format_instruction = functools.partial(
                generate_policy_prompt,
                robot_name="WindowX",
                num_arms="1",
                action_space="Delta End-Effector",
                prompt_style="minimal",
            )
        else:
            raise NotImplementedError()
        
        self.action_space_index = torch.tensor([get_action_space_index('EEF_POS', 1, 'velocity', return_tensor=False)])
        self.frequency = torch.tensor([DATASET_FREQUENCY_MAP[POLICY_SETUP_TO_DATASET_INDEX[self.policy_setup]]])
        
        self.action_index = ActionIndex()
    def rescale_to_range(self, tensor) -> torch.Tensor:
        max_values = self.max_values.cpu()
        min_values = self.min_values.cpu()
        # Scale the tensor to the new range [new_min, new_max]
        new_min = -torch.ones_like(tensor).cpu()
        new_max = torch.ones_like(tensor).cpu()
        rescaled_tensor = (tensor - new_min) / (new_max - new_min) * (max_values - min_values) + min_values
        return rescaled_tensor
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image
    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            print("task description: ", task_description)
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([self.task_description])
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)
    def reset(self, task_description: str) -> None:
        self._initialize_task_description(task_description)
        self.curr_horizon_index = 0
    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        task_description = self.format_instruction(task_description)
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)
        assert image.dtype == np.uint8
        image = torch.from_numpy(np.moveaxis(self._resize_image(image), -1, 0)).unsqueeze(0).unsqueeze(0).to(device=self.device)
        # image2 = torch.ones_like(image)
        # if self.curr_horizon_index % 5 == 0: # hardcode for now
        #     self.curr_horizon_index = 0
            # input_observation = {"image_primary": image, "image_wrist": image2}
        input_observation = {
            "image_primary": image,
            "pad_mask_dict": {"image_primary": torch.ones(1,1).bool().to(device=self.device)},
        }
        input_observation = {
            "observation": input_observation,
            "task": {
                "language_instruction": self.task_description_embedding,
                "frequency": self.frequency,
                "action_space_index": self.action_space_index,
            }
        }
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                unscaled_raw_actions = self.agent(input_observation).cpu() # (action_dim)
        unscaled_raw_actions = unscaled_raw_actions[:self.action_index.get_action_dim(self.action_space_index)]
        raw_actions = torch.cat([self.rescale_to_range(unscaled_raw_actions[..., :-1]), unscaled_raw_actions[...,-1:]], dim=-1).detach()
        #raw_actions = self.raw_actions[self.curr_horizon_index].numpy()
        # self.curr_horizon_index += 1
        assert raw_actions.shape == (7,)
        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(raw_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action
            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = relative_gripper_action
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action
    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str, process_index=0, wandb = None) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        if wandb is not None and wandb.run is not None and process_index is not None:
            name = "Simpler Env " + str(process_index) + ":"
            wandb.log({name: plt}, commit=False)
            plt.close()
        else:
            plt.savefig(save_path)


def check_model_weights(model):
    """Check if model weights appear randomly initialized or trained."""
    stats = {}
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Calculate basic statistics
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'num_zeros': (param.data == 0).sum().item()
            }
            
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
            
            # Random init usually has characteristic stats
            if abs(stats[name]['mean']) < 1e-8 and 0.001 < stats[name]['std'] < 1:
                print(f"WARNING: {name} looks potentially random - mean ≈ 0 and std in typical init range")
                
    # Check overall zero ratio
    zero_ratio = zero_params / total_params
    if zero_ratio > 0.5:
        print(f"WARNING: {zero_ratio:.1%} of parameters are exactly zero - unusually high")
        
    return stats


def check_model_outputs(model, device):
    """Check if model produces reasonable outputs."""
    model.eval()
    with torch.no_grad():
        # Create a simple fixed test input
        test_image = torch.ones(1, 3, 224, 224).to(device)
        test_input = {
            "observation": {
                "image_primary": test_image,
                "pad_mask_dict": {"image_primary": torch.ones(1,1).bool().to(device)}
            },
            "task": {
                "language_instruction": model.agent.lang_embed_model(["pick up the cube"]),
                "frequency": torch.tensor([10]),
                "action_space_index": torch.tensor([0])
            }
        }
        
        output = model(test_input)
        
        # Check if outputs are in expected ranges
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Output mean: {output.mean():.3f}")
        print(f"Output std: {output.std():.3f}")
        
        # Random weights often produce extreme values
        if output.abs().max() > 10:
            print("WARNING: Outputs seem unusually large")
            
        return output
    

def get_model_hash(model):
    """Get a hash of model parameters to detect changes."""
    state_dict = model.state_dict()
    param_str = ""
    for name, param in sorted(state_dict.items()):
        param_str += f"{name}:{param.data.sum().item():.4f}"
    return hash(param_str)


def check_model_weights(model):
    """Detailed check of model weights and architecture"""
    print("\n=== Detailed Model Analysis ===")
    
    # Track architecture details
    details = {
        'total_params': 0,
        'trainable_params': 0,
        'layer_stats': {},
        'suspicious_layers': [],
        'major_components': defaultdict(int)
    }

    # Analyze each parameter
    for name, param in model.named_parameters():
        # Get component name (first part of parameter path)
        component = name.split('.')[0]
        details['major_components'][component] += param.numel()
        
        if param.requires_grad:
            details['trainable_params'] += param.numel()
            
            # Calculate detailed statistics
            stats = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'zeros': (param.data == 0).float().mean().item() * 100,
                'size': param.numel()
            }
            details['layer_stats'][name] = stats
            
            # Check for suspicious patterns
            if abs(stats['mean']) < 1e-6 and 0.001 < stats['std'] < 1:
                details['suspicious_layers'].append({
                    'name': name,
                    'reason': 'Random-like statistics',
                    'stats': stats
                })
            if stats['zeros'] > 90:
                details['suspicious_layers'].append({
                    'name': name,
                    'reason': f"{stats['zeros']:.1f}% zeros",
                    'stats': stats
                })
            if stats['std'] < 1e-6:
                details['suspicious_layers'].append({
                    'name': name,
                    'reason': 'Near-zero variance',
                    'stats': stats
                })
                
        details['total_params'] += param.numel()

    # Print analysis
    print(f"\nModel Architecture Summary:")
    print(f"Total parameters: {details['total_params']:,}")
    print(f"Trainable parameters: {details['trainable_params']:,}")
    
    print("\nMajor Components:")
    for component, count in details['major_components'].items():
        print(f"{component}: {count:,} parameters ({count/details['total_params']*100:.1f}%)")
    
    if details['suspicious_layers']:
        print("\nSuspicious Layers Detected:")
        for layer in details['suspicious_layers']:
            print(f"\n- {layer['name']}")
            print(f"  Reason: {layer['reason']}")
            print(f"  Statistics:")
            for k, v in layer['stats'].items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6f}")
                else:
                    print(f"    {k}: {v}")
    return details
    
# Compare key statistics
def compare_model_states(pre, post):
    print("\n=== Model State Comparison ===")
    
    # Check if overall architecture changed
    arch_changed = pre['total_params'] != post['total_params']
    if arch_changed:
        print(f"WARNING: Total parameters changed: {pre['total_params']:,} -> {post['total_params']:,}")
    
    # Compare component sizes
    print("\nComponent Changes:")
    all_components = set(pre['major_components'].keys()) | set(post['major_components'].keys())
    for component in all_components:
        pre_count = pre['major_components'].get(component, 0)
        post_count = post['major_components'].get(component, 0)
        if pre_count != post_count:
            print(f"{component}: {pre_count:,} -> {post_count:,}")
    
    return not arch_changed and len(post['suspicious_layers']) == 0



if __name__ == "__main__":
    testing = UhaInference()