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
    parser.add_argument("--arch", type=str, default="resnet")
    io_groups.add_argument("--save_plot", action="store_true")
    io_groups.add_argument("--concatenate", action="store_true")

    args = parser.parse_args()

    all_models = list(Path.cwd().glob(args.model_globbing))

    def save_plot_arg(yes):
        if yes:
            return f"--save_plot"
        else:
            return f"--no-save_plot"

    logging.info(f"Found {len(all_models)} models")
    logging.info(f"Found {all_models}")
    for m in all_models:
        base_folder = Path(m).parent
        logging.info(base_folder)
        name_run = str(Path(m).parents[1].name) + "-" + str(Path(m).parents[0].name)
        if not osp.exists(f"./{args.out_dir}/{name_run}/"):
            os.mkdir(f"./{args.out_dir}/{name_run}/")
        # read yaml
        with open(str(base_folder) + "/parameters.json", "r") as f:
            conf = json.load(f)
        print(conf)

        sample_freq = 20.0  # no concatenation
        if args.concatenate:
            sample_freq = 1.0 / conf["window_time"]  # concatenate

        command = [
            "python3",
            "main_net.py",
            "--mode",
            "test",
            "--root_dir",
            f"{args.root_dir}",
            "--arch",
            f"{args.arch}",
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
            f"{sample_freq}",
            f"{save_plot_arg(args.save_plot)}",
        ]
        logging.info(" ".join(command))
        try:
            sp.run(command)
        except Exception as e:
            logging.error(e)
            continue
