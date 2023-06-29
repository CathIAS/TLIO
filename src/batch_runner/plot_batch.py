import argparse
import json
import subprocess as sp
from pathlib import Path
from pprint import pprint

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
        "--runname_globbing",
        type=str,
        default="*hz-*s-*s*",
        help="Globbing expression for run to plot",
    )
    io_groups.add_argument("--filter_dir", type=str, help="")
    io_groups.add_argument(
        "--ronin_dir", type=str, help="folder names should match those in filter_dir"
    )
    io_groups.add_argument(
        "--window_time",
        type=float,
        default=1.0,
        help="window time when no ronin dir exists",
    )
    io_groups.add_argument("--no_make_plots", action="store_true")

    args = parser.parse_args()

    all_outputs_filter = list(
        Path.cwd().glob(args.filter_dir + "/" + args.runname_globbing)
    )
    logging.info(f"Found {len(all_outputs_filter)} runs")

    for m in all_outputs_filter:
        base_folder = Path(m)
        logging.info(base_folder)
        name_run = base_folder.name
        # read parameters
        with open(base_folder.joinpath("./parameters.json"), "r") as f:
            conf_filter = json.load(f)
        pprint(conf_filter)
        if "window_time" in conf_filter.keys():
            disp_time = float(conf_filter["window_time"])
        else:
            disp_time = args.window_time  # This is default

        command = [
            "python",
            "plot_filter_state.py",
            "--root_dir",
            f"{args.root_dir}",
            "--log_dir",
            f"{args.filter_dir + '/' + name_run}",
            "--save_fig",
            "--body_bias",
            "--displacement_time",
            f"{disp_time}",
        ]

        # look for corresponding ronin dir
        ronin_base_folder = Path(args.ronin_dir).joinpath(name_run)
        if ronin_base_folder.exists():
            ronin_dir = str(ronin_base_folder)
            command = command + ["--ronin_dir", f"{ronin_dir}"]
        if args.no_make_plots:
            command = command + ["--no-make_plots"]
        else:
            command = command + ["--make_plots"]
        logging.info(" ".join(command))
        sp.run(command)
