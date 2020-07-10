import argparse
import json
import os
import os.path as osp
import subprocess as sp
from pathlib import Path

from utils.logging import logging


homedir = Path.home()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ----------------------- io params -----------------------
    io_groups = parser.add_argument_group("io")
    io_groups.add_argument(
        "--root_dir",
        type=str,
        default=f"{homedir}/vo_output",
        help="Path to dataset directory",
    )
    io_groups.add_argument(
        "--data_list",
        type=str,
        default=f"{homedir}/vo_output/split/golden/golden_test.txt",
    )
    io_groups.add_argument(
        "--model_globbing",
        type=str,
        default="../models/200hz/1s-1s*/checkpoint_*.pt",
        help="Globbing expression for model selection",
    )
    io_groups.add_argument(
        "--out_dir",
        type=str,
        default=f"./all_output/",
        help="Path to dataset directory",
    )
    parser.add_argument("--sample_freq", type=float, default=5.0)
    parser.add_argument("--perturbation_analysis", action="store_true")

    args = parser.parse_args()

    all_models = list(Path.cwd().glob(args.model_globbing))
    logging.info(f"Found {len(all_models)} models")
    logging.info(f"Found {all_models}")

    for m in all_models:
        base_folder = Path(m).parent
        logging.info(base_folder)
        name_run = str(Path(m).parents[1].name) + "-" + str(Path(m).parents[0].name)
        if not osp.exists(f"./{args.out_dir}/{name_run}/"):
            os.mkdir(f"./{args.out_dir}/{name_run}/")
        with open(str(base_folder) + "/parameters.json", "r") as f:
            conf = json.load(f)
        print(conf)

        if args.perturbation_analysis:
            accel_bias_ptrb_range = [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            gyro_bias_ptrb_range = [
                0,
                0,
                0,
                0,
                0,
                0,
                0.025,
                0.05,
                0.075,
                0.1,
                0,
                0,
                0,
                0,
                0,
            ]
            grav_ptrb_range = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10]
        else:
            accel_bias_ptrb_range = [0]
            gyro_bias_ptrb_range = [0]
            grav_ptrb_range = [0]

        n_trials = len(grav_ptrb_range)
        for trial in range(n_trials):

            command = [
                "python3",
                "main_net.py",
                "--mode",
                "eval",
                "--test_list",
                f"{args.data_list}",
                "--root_dir",
                f"{args.root_dir}",
                "--model_path",
                f"{m}",
                "--out_dir",
                f"./{args.out_dir}/{name_run}/",
                "--imu_freq",
                f'{conf["imu_freq"]}',
                "--past_time",
                f'{conf["past_time"]}',
                "--window_time",
                f'{conf["window_time"]}',
                "--future_time",
                f'{conf["future_time"]}',
                "--sample_freq",
                f"{args.sample_freq}",
                "--do_bias_shift",
                "--accel_bias_range",
                f"{accel_bias_ptrb_range[trial]}",
                "--gyro_bias_range",
                f"{gyro_bias_ptrb_range[trial]}",
                "--perturb_gravity",
                "--perturb_gravity_theta_range",
                f"{grav_ptrb_range[trial]}",
            ]
            logging.info(" ".join(command))
            try:
                sp.run(command)
            except Exception as e:
                logging.error(e)
                continue
