from dataclasses import dataclass, field
from SimplerEnv.simpler_env.evaluation.argparse import parse_range_tuple
from simpler_env.main_inference import main
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat

from simpler_env.utils.io import DictAction

@dataclass
class FakeArgs:
    policy_model: str = "rt1"
    policy_setup: str = "google_robot"
    ckpt_path: str = None
    env_name: str = ""
    additional_env_save_tags: str = None
    scene_name: str = "google_pick_coke_can_1_v4"
    enable_raytracing: bool = False
    robot: str = "google_robot_static"
    obs_camera_name: str = None
    action_scale: float = 1.0
    control_freq: int = 3
    sim_freq: int = 513
    max_episode_steps: int = 80
    rgb_overlay_path: str = None
    robot_init_x_range: tuple = (0.35, 0.35, 1)
    robot_init_y_range: tuple = (0.20, 0.20, 1)
    robot_init_rot_quat_center: tuple = (1, 0, 0, 0)
    robot_init_rot_rpy_range: tuple = (0, 0, 1, 0, 0, 1, 0, 0, 1)
    obj_variation_mode: str = "xy"
    obj_episode_range: tuple = (0, 60)
    obj_init_x_range: tuple = (-0.35, -0.12, 5)
    obj_init_y_range: tuple = (-0.02, 0.42, 5)
    additional_env_build_kwargs: dict = field(default_factory=dict)
    logging_dir: str = "./results"
    tf_memory_limit: int = 3072
    octo_init_rng: int = 0


ckpt = "16-30-28/checkpoint_160000"
ckpt = '13-14-17/checkpoint_120000'
ckpt = '19-47-12/checkpoint_40000'

fake_args = {
    "medit_bridge.sh": [
        FakeArgs(
            policy_model="medit",
            ckpt_path=ckpt,
            robot="widowx",
            policy_setup="widowx_bridge",
            control_freq=5,
            sim_freq=500,
            max_episode_steps=60,
            env_name="PutCarrotOnPlateInScene-v0",
            scene_name="bridge_table_1_v1",
            rgb_overlay_path="/home/reuss/code/flower_vla_policy/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
            robot_init_x_range=(0.147, 0.147, 1),
            robot_init_y_range=(0.028, 0.028, 1),
            obj_variation_mode="episode",
            obj_episode_range=(0, 2),
            robot_init_rot_quat_center=(0, 0, 0, 1),
            robot_init_rot_rpy_range=(0, 0, 1, 0, 0, 1, 0, 0, 1),
        ),

        FakeArgs(
            policy_model="medit",
            ckpt_path=ckpt,
            robot="widowx",
            policy_setup="widowx_bridge",
            control_freq=5,
            sim_freq=500,
            max_episode_steps=60,
            env_name="PutSpoonOnTableClothInScene-v0",
            scene_name="bridge_table_1_v1",
            rgb_overlay_path="/home/reuss/code/flower_vla_policy/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
            robot_init_x_range=(0.147, 0.147, 1),
            robot_init_y_range=(0.028, 0.028, 1),
            obj_variation_mode="episode",
            obj_episode_range=(0, 2),
            robot_init_rot_quat_center=(0, 0, 0, 1),
            robot_init_rot_rpy_range=(0, 0, 1, 0, 0, 1, 0, 0, 1),
        ),
        
    ],

    "medit_pick_coke_can_variant_agg.sh": [

        # done
        FakeArgs(
            policy_model="medit",
            ckpt_path=ckpt,
            robot="google_robot_static",
            control_freq=3,
            sim_freq=513,
            max_episode_steps=80,
            env_name="GraspSingleOpenedCokeCanInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            robot_init_x_range=(0.35, 0.35, 1),
            robot_init_y_range=(0.20, 0.20, 1),
            obj_init_x_range=(-0.35, -0.12, 5),
            obj_init_y_range=(-0.02, 0.42, 5),
            obj_episode_range=(0, 2),
            robot_init_rot_quat_center=(0, 0, 0, 1),
            robot_init_rot_rpy_range=(0, 0, 1, 0, 0, 1, 0, 0, 1),
            additional_env_build_kwargs={"lr_switch": True},
        ),
    ]
}

for script, args_s in fake_args.items():
    for args in args_s:
        args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
        args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
        args.robot_init_quats = []
        for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
            for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
                for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                    args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
        # env args: object position
        if args.obj_variation_mode == "xy":
            args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
            args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
        # update logging info (args.additional_env_save_tags) if using a different camera from default
        if args.obs_camera_name is not None:
            if args.additional_env_save_tags is None:
                args.additional_env_save_tags = f"obs_camera_{args.obs_camera_name}"
            else:
                args.additional_env_save_tags = args.additional_env_save_tags + f"_obs_camera_{args.obs_camera_name}"


        main(args)