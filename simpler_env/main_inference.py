import os

import numpy as np
import tensorflow as tf
import torch

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.device_count())
    device = torch.device("cuda")  # Change this to "cpu" if needed
    torch.set_default_device(device)
    print(device)
    print(torch.cuda.is_available())

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    if model is None:
        # policy model creation; update this if you are using a new policy model
        if args.policy_model == "rt1":
            from simpler_env.policies.rt1.rt1_model import RT1Inference
            assert args.ckpt_path is not None
            model = RT1Inference(
                saved_model_path=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        elif "octo" in args.policy_model:
            if args.ckpt_path is None or args.ckpt_path == "None":
                args.ckpt_path = args.policy_model
            if "server" in args.policy_model:
                from simpler_env.policies.octo.octo_server_model import OctoServerInference
                model = OctoServerInference(
                    model_type=args.ckpt_path,
                    policy_setup=args.policy_setup,
                    action_scale=args.action_scale,
                )
            else:
                from simpler_env.policies.octo.octo_model import OctoInference
                model = OctoInference(
                    model_type=args.ckpt_path,
                    policy_setup=args.policy_setup,
                    init_rng=args.octo_init_rng,
                    action_scale=args.action_scale,
                )
        elif args.policy_model == "medit":
            from SimplerEnv.simpler_env.policies.uha_test_policy.flower_vla_test import UhaInference
            model = UhaInference(
                saved_model_path=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

if __name__ == "__main__":
    args = get_args()
    main(args)