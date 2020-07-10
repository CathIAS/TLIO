"""
IMU network training/testing/evaluation for displacement and covariance
Input: Nx6 IMU data
Output: 3x1 displacement, 3x1 covariance parameters
"""

import network
from utils.argparse_utils import add_bool_arg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # ------------------ directories -----------------
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--test_list", type=str, default=None)
    parser.add_argument(
        "--root_dir", type=str, default=None, help="Path to data directory"
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)

    # ------------------ architecture and training -----------------
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10000, help="max num epochs")
    parser.add_argument("--arch", type=str, default="resnet")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=3)

    # ------------------ commons -----------------
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument(
        "--imu_freq", type=float, default=200.0, help="imu_base_freq is a multiple"
    )
    parser.add_argument("--imu_base_freq", type=float, default=1000.0)

    # ----- perturbation -----
    add_bool_arg(parser, "do_bias_shift", default=True)
    parser.add_argument("--accel_bias_range", type=float, default=0.2)  # 5e-2
    parser.add_argument("--gyro_bias_range", type=float, default=0.05)  # 1e-3

    add_bool_arg(parser, "perturb_gravity", default=True)
    parser.add_argument(
        "--perturb_gravity_theta_range", type=float, default=5.0
    )  # degrees

    # ----- window size and inference freq -----
    parser.add_argument("--past_time", type=float, default=0.0)  # s
    parser.add_argument("--window_time", type=float, default=1.0)  # s
    parser.add_argument("--future_time", type=float, default=0.0)  # s

    # ----- for sampling in training / stepping in testing -----
    parser.add_argument("--sample_freq", type=float, default=20.0)  # hz

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "save_plot", default=False)
    parser.add_argument("--rpe_window", type=float, default="2.0")  # s

    args = parser.parse_args()

    ###########################################################
    # Main
    ###########################################################
    if args.mode == "train":
        network.net_train(args)
    elif args.mode == "test":
        network.net_test(args)
    elif args.mode == "eval":
        network.net_eval(args)
    else:
        raise ValueError("Undefined mode")
