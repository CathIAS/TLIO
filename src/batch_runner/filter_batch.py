import argparse
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

    args = parser.parse_args()

    all_models = list(Path.cwd().glob(args.model_globbing))

    logging.info(f"Found {len(all_models)} models")
    logging.info(f"Found {all_models}")

    update_frequency_list = [20]
    for update_frequency in update_frequency_list:
        try:
            os.mkdir(f"./{args.out_dir}_uf{update_frequency}")
        except:
            pass

    for update_frequency in update_frequency_list:
        for m in all_models:
            base_folder = Path(m).parent
            logging.info(base_folder)
            name_run = str(Path(m).parents[1].name) + "-" + str(Path(m).parents[0].name)
            if not osp.exists(f"./{args.out_dir}_uf{update_frequency}/{name_run}/"):
                os.mkdir(f"./{args.out_dir}_uf{update_frequency}/{name_run}/")

            model_param_path = str(base_folder) + "/parameters.json"
            meascov_scale = 10
            command = [
                "python3",
                "main_filter.py",
                "--root_dir",
                f"{args.root_dir}",
                "--model_path",
                f"{m}",
                "--model_param_path",
                f"{model_param_path}",
                "--out_dir",
                f"./{args.out_dir}_uf{update_frequency}/{name_run}/",
                "--no-erase_old_log",
                "--save_as_npy",
                "--meascov_scale",
                f"{meascov_scale}",
                "--initialize_with_offline_calib",
                "--update_freq",
                f"{update_frequency}",
            ]
            logging.info(" ".join(command))
            try:
                sp.run(command)
            except Exception as e:
                logging.error(e)
                continue
