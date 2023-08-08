import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
from network.model_factory import get_model
from utils.logging import logging


def load_and_convert(args):
    past_data_size = int(args.past_time * args.imu_freq_net)
    disp_window_size = int(args.window_time * args.imu_freq_net)

    net_config = {"in_dim": (past_data_size + disp_window_size) // 32 + 1}

    net = get_model(args.arch, net_config, 6, 3)

    # load trained network model
    if not torch.cuda.is_available() or args.cpu:
        device = torch.device("cpu")
        checkpoint = torch.load(
            args.model_path, map_location=lambda storage, location: storage
        )
    else:
        device = torch.device("cuda:0")
        checkpoint = torch.load(args.model_path)

    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval().to(device)
    input_sequence_dim = past_data_size + disp_window_size

    traced_cell = torch.jit.trace(net, torch.zeros(1, 6, input_sequence_dim))
    traced_cell.save(args.out_dir + "/model_torchscript.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ----------------------- io params -----------------------
    io_groups = parser.add_argument_group("io")

    io_groups.add_argument(
        "--root_dir", type=str, default=None, help="Path to data directory"
    )
    io_groups.add_argument("--model_path", required=True, type=str, default=None)
    io_groups.add_argument("--model_param_path", required=True, type=str, default=None)
    io_groups.add_argument("--out_dir", type=str, default="./")

    parser.add_argument("--cpu", type=bool, default=True)
    args = parser.parse_args()

    # overwrite network model parameter from json file if specified
    if args.model_param_path is not None:
        with open(args.model_param_path) as json_file:
            data_json = json.load(json_file)
            args.imu_freq_net = data_json["imu_freq"]
            args.past_time = data_json["past_time"]
            args.window_time = data_json["window_time"]
            args.arch = data_json["arch"]

    np.set_printoptions(linewidth=2000)

    logging.info("Program options:")
    logging.info(pprint(vars(args)))
    # run filter
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    load_and_convert(args)
