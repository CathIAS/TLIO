"""
TLIO Stochastic Cloning Extended Kalman Filter
Input: IMU data
Measurement: window displacement estimates from networks
Filter states: position, velocity, rotation, IMU biases
"""

import argparse
import datetime
import json
import os

# silence NumbaPerformanceWarning
import warnings
from pprint import pprint

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from tracker.imu_tracker_runner import ImuTrackerRunner
from utils.argparse_utils import add_bool_arg
from utils.git_version import git_version
from utils.logging import logging
from utils.profile import profile


warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ----------------------- io params -----------------------
    io_groups = parser.add_argument_group("io")

    io_groups.add_argument(
        "--root_dir", type=str, default=None, help="Path to data directory"
    )
    io_groups.add_argument("--data_list", type=str, default=None)
    io_groups.add_argument("--dataset_number", type=int, default=None)
    io_groups.add_argument("--model_path", type=str, default=None)
    io_groups.add_argument("--model_param_path", type=str, default=None, required=True)
    io_groups.add_argument("--out_dir", type=str, default=".")
    io_groups.add_argument("--out_filename", type=str, default="not_vio_state.txt")
    io_groups.add_argument("--save_as_npy", action="store_true")
    io_groups.add_argument("--sim_data_path", type=str, default="imu-sim.txt")
    io_groups.add_argument(
        "--start_from_ts", type=int, default=None
    )  # dataloader loading data from timestamp (us)

    add_bool_arg(io_groups, "erase_old_log", default=False)

    # ----------------------- network params -----------------------
    net_groups = parser.add_argument_group("network")
    net_groups.add_argument("--cpu", action="store_true")

    # ----------------------- filter params -----------------------
    filter_group = parser.add_argument_group("filter tuning:")

    filter_group.add_argument("--update_freq", type=float, default=20.0)  # (Hz)

    filter_group.add_argument(
        "--sigma_na", type=float, default=np.sqrt(1e-3)
    )  # accel noise  m/s^2
    filter_group.add_argument(
        "--sigma_ng", type=float, default=np.sqrt(1e-4)
    )  # gyro noise  rad/s
    filter_group.add_argument(
        "--ita_ba", type=float, default=1e-4
    )  # accel bias noise  m/s^2/sqrt(s)
    filter_group.add_argument(
        "--ita_bg", type=float, default=1e-6
    )  # gyro bias noise  rad/s/sqrt(s)

    filter_group.add_argument(
        "--init_attitude_sigma", type=float, default=10.0 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument(
        "--init_yaw_sigma", type=float, default=0.1 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument("--init_vel_sigma", type=float, default=1.0)  # m/s
    filter_group.add_argument("--init_pos_sigma", type=float, default=0.001)  # m
    filter_group.add_argument(
        "--init_bg_sigma", type=float, default=0.0001
    )  # rad/s  0.001
    filter_group.add_argument("--init_ba_sigma", type=float, default=0.2)  # m/s^2  0.02
    filter_group.add_argument("--g_norm", type=float, default=9.81)

    filter_group.add_argument("--meascov_scale", type=float, default=1.0)

    add_bool_arg(
        filter_group, "initialize_with_vio", default=True
    )  # initialize state with gt state
    add_bool_arg(
        filter_group, "initialize_with_offline_calib", default=False
    )  # initialize bias state with offline calib or 0

    filter_group.add_argument(
        "--mahalanobis_fail_scale", type=float, default=0
    )  # if nonzero then mahalanobis gating test would scale the covariance by this scale if failed

    # ----------------------- debug params -----------------------
    debug_groups = parser.add_argument_group("debug")
    # covariance alternatives (note: if use_vio_meas is true, meas constant with default value 1e-4)
    add_bool_arg(debug_groups, "use_const_cov", default=False)
    debug_groups.add_argument(
        "--const_cov_val_x", type=float, default=np.power(0.1, 2.0)
    )
    debug_groups.add_argument(
        "--const_cov_val_y", type=float, default=np.power(0.1, 2.0)
    )
    debug_groups.add_argument(
        "--const_cov_val_z", type=float, default=np.power(0.1, 2.0)
    )

    # measurement alternatives (note: if use_vio_meas is false, add_sim_meas_noise must be false)
    add_bool_arg(
        debug_groups,
        "use_vio_meas",
        default=False,
        help='If using "vio" measurement for filter update instead of ouptut network',
    )
    add_bool_arg(debug_groups, "debug_using_vio_ba", default=False)
    add_bool_arg(
        debug_groups, "add_sim_meas_noise", default=False
    )  # adding noise on displacement measurement when using vio measurement
    debug_groups.add_argument(
        "--sim_meas_cov_val", type=float, default=np.power(0.01, 2.0)
    )
    debug_groups.add_argument(
        "--sim_meas_cov_val_z", type=float, default=np.power(0.01, 2.0)
    )
    add_bool_arg(debug_groups, "do_profile", default=False, help="Run the profiler")

    args = parser.parse_args()

    np.set_printoptions(linewidth=2000)

    logging.info("Program options:")
    logging.info(pprint(vars(args)))
    # run filter
    with open(args.data_list) as f:
        data_names = [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    param_dict = vars(args)
    param_dict["git_version"] = git_version()
    param_dict["date"] = str(datetime.datetime.now())
    with open(args.out_dir + "/parameters.json", "w") as parameters_file:
        parameters_file.write(json.dumps(param_dict, indent=4, sort_keys=True))

    # load offline calibration for IMU
    with profile(filename="./profile.prof", enabled=args.do_profile):
        if args.dataset_number is not None:
            logging.info("Running in one-shot mode")
            logging.info("Using dataset {}".format(data_names[args.dataset_number]))
            trackerRunner = ImuTrackerRunner(args, data_names[args.dataset_number])
            trackerRunner.run_tracker(args)
        else:
            logging.info("Running in batch mode")
            # add metadata for logging
            n_data = len(data_names)
            for i, name in enumerate(data_names):
                logging.info(f"Processing {i} / {n_data} dataset {name}")
                try:
                    trackerRunner = ImuTrackerRunner(args, name)
                    trackerRunner.run_tracker(args)
                except FileExistsError as e:
                    print(e)
                    continue
                except OSError as e:
                    print(e)
                    continue
