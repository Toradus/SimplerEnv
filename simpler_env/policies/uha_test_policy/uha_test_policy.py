from collections import defaultdict
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from transforms3d.euler import euler2axangle
import torch.nn as nn
import tensorflow_hub as hub
from transformers import CLIPTokenizer


class UhaInference:
    def __init__(
        self,
        agent: nn.Module,
        lang_embed: str = "openai/clip-vit-base-patch32", # "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        image_size: int = 224,
        pred_action_horizon: int = 10,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if lang_embed == "openai/clip-vit-base-patch32":
            self.lang_embed_model = CLIPTokenizer.from_pretrained(lang_embed)
        else:
            self.lang_embed_model = hub.load(lang_embed)
        
        self.image_size = image_size
        self.action_scale = action_scale
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
            self.max_values = torch.tensor([0.17824687153100965, 0.14938379630446405, 0.21842354819178575, 0.5892666035890578, 0.35272657424211445, 0.44796681255102094, 1.0]) # p99
            self.min_values = torch.tensor([-0.22453527510166169, -0.14820013284683228, -0.231589707583189, -0.3517994859814644, -0.4193011274933815, -0.43643461108207704, 0.0]) # p01
        elif self.policy_setup == "widowx_bridge":
            self.sticky_gripper_num_repeat = 1
            # use bridge norm values
            self.max_values = torch.tensor([0.02911195397377009, 0.04201051414012899, 0.04071581304073327, 0.08772125840187053, 0.08282401025295247, 0.16359195709228502, 1.0]) # p99
            self.min_values = torch.tensor([-0.029900161027908326, -0.04327958464622497, -0.02570973977446556, -0.0863340237736702, -0.09845495343208313, -0.1693541383743286, 0.0]) # p01
        else:
            raise NotImplementedError()

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
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([task_description], return_tensors = 'pt', padding = "max_length", truncation = True, max_length = 77)
            self.task_description_embedding["input_ids"] = self.task_description_embedding["input_ids"].unsqueeze(0).to(device=self.device)
            self.task_description_embedding["attention_mask"] = self.task_description_embedding["attention_mask"].unsqueeze(0).to(device=self.device)
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def reset(self, task_description: str) -> None:
        self._initialize_task_description(task_description)

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
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = torch.from_numpy(np.moveaxis(self._resize_image(image), -1, 0)).unsqueeze(0).unsqueeze(0).to(device=self.device)
        # image2 = torch.ones_like(image)

        # input_observation = {"image_primary": image, "image_wrist": image2}
        input_observation = {"image_primary": image}
        input_observation = {
            "observation": input_observation,
            "task": {"language_instruction": self.task_description_embedding}
        }
        unscaled_raw_actions = self.agent(input_observation)[0][:, :, :7].cpu()
        unscaled_raw_actions = unscaled_raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)

        raw_actions = self.rescale_to_range(unscaled_raw_actions)

        assert raw_actions.shape == (self.pred_action_horizon, 7)

        raw_action = {
            "world_vector": np.array(raw_actions[:, :3]),
            "rotation_delta": np.array(raw_actions[:, 3:6]),
            "open_gripper": np.array(raw_actions[:, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        action["rot_axangle"] = []
        for rotation in action_rotation_delta:
            roll, pitch, yaw = rotation
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"].append(action_rotation_axangle * self.action_scale)

        action["rot_axangle"] = np.asarray(action["rot_axangle"], dtype=np.float64)

        if self.policy_setup == "google_robot":
            action["gripper"] = []
            for current_gripper_action in raw_action["open_gripper"]:

                # This is one of the ways to implement gripper actions; we use an alternative implementation below for consistency with real
                # gripper_close_commanded = (current_gripper_action < 0.5)
                # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

                # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
                # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
                #     self.sticky_action_is_on = True
                #     self.sticky_gripper_action = relative_gripper_action

                # if self.sticky_action_is_on:
                #     self.gripper_action_repeat += 1
                #     relative_gripper_action = self.sticky_gripper_action

                # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
                #     self.sticky_action_is_on = False
                #     self.gripper_action_repeat = 0

                # action['gripper'] = np.array([relative_gripper_action])

                # alternative implementation
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

                action["gripper"].append(relative_gripper_action)
            
            action["gripper"] = np.asarray(action["gripper"])

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = []
            for current_gripper_action in raw_action["open_gripper"]:
                action["gripper"].append((
                    2.0 * (current_gripper_action > 0.5) - 1.0
                ))  # binarize gripper action to 1 (open) and -1 (close)
                # self.gripper_is_closed = (action['gripper'] < 0.0)
            action["gripper"] = np.asarray(action["gripper"])

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str, process_index, wandb = None) -> None:
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
                np.concatenate([a["world_vector"][i], a["rotation_delta"][i], a["open_gripper"][i]], axis=-1)
                for a in predicted_raw_actions
                for i in range(10)
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
        else:
            plt.savefig(save_path)
        plt.close()
