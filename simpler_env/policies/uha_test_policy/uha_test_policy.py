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
        policy_setup: str = "widowx_bridge",
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

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()

    @staticmethod
    def _rescale_action_with_bound(
        actions: np.ndarray | tf.Tensor,
        low: float,
        high: float,
        safety_margin: float = 0.0,
        post_scaling_max: float = 1.0,
        post_scaling_min: float = -1.0,
    ) -> np.ndarray:
        """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
        resc_actions = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min
        return np.clip(
            resc_actions,
            post_scaling_min + safety_margin,
            post_scaling_max - safety_margin,
        )

    def _unnormalize_action_widowx_bridge(self, action: dict[str, np.ndarray | tf.Tensor]) -> dict[str, np.ndarray]:
        action[:, :3] = self._rescale_action_with_bound(
            action[:, :3],
            low=-1.0,
            high=1.0,
            post_scaling_max=0.05,
            post_scaling_min=-0.05,
        )
        action[:, 3:6] = self._rescale_action_with_bound(
            action[:, 3:6],
            low=-1.0,
            high=1.0,
            post_scaling_max=0.25,
            post_scaling_min=-0.25,
        )
        return action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.float32).numpy()
        return image

    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([task_description], return_tensors = 'pt', padding = "max_length", truncation = True, max_length = 77)
            self.task_description_embedding["input_ids"] = self.task_description_embedding["input_ids"].unsqueeze(0)
            self.task_description_embedding["attention_mask"] = self.task_description_embedding["attention_mask"].unsqueeze(0)
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
        image2 = torch.ones_like(image)

        input_observation = {"image_primary": image, "image_wrist": image2}
        input_observation = {
            "observation": input_observation,
            "task": {"language_instruction": self.task_description_embedding}
        }
        unscaled_raw_actions = self.agent(input_observation)[0][:, :, :7].cpu()
        unscaled_raw_actions = unscaled_raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)

        raw_actions = self.unnormalize_action_fxn(unscaled_raw_actions)

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

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
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
        plt.savefig(save_path)
