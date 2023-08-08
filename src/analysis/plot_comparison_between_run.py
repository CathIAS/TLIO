#!/usr/bin/env python3

import glob
import json

# TODO remove copy pasta
import os
import re
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import plot_state
import progressbar
from math_utils import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from utils.logging import logging


def plot_position(args, log_folder, n, dataset):
    results_folder = log_folder + "/" + dataset
    filename = os.path.join(results_folder, args.log_filename + ".npy")
    try:
        logging.info(f"Loading {filename}")
        states = np.load(filename)
    except:
        logging.error(
            f"{filename}.npy was not found. Surely means filter did not finish"
        )
        raise FileNotFoundError

    # load all states
    R_init = states[0, :9].reshape(-1, 3, 3)
    Rs = states[:, :9].reshape(-1, 3, 3)
    rs = Rotation.from_matrix(Rs)
    ps = states[:, 12:15]
    ts = states[:, 27]
    sigma_p = np.sqrt(states[:, 34:37])

    if os.path.exists(os.path.join(results_folder, "vio_states.npy")):
        vio_states = np.load(os.path.join(results_folder, "vio_states.npy"))
        ref_p = vio_states[:, 3:6]
        ref_accelScaleInv_flat = vio_states[:, 24:33]
        ref_gyroScaleInv_flat = vio_states[:, 33:42]
    else:
        logging.error(
            "vio_states.npy was not found. you shoud create it with plot_state.py before... sorry :("
        )
        raise FileNotFoundError

    fig14 = plt.figure("position-2d " + dataset)
    plt.plot(ps[:, 0], ps[:, 1], label=n)
    plt.plot(
        ref_p[:, 0], ref_p[:, 1], label="vio", color=plot_state.color_vio, linestyle=":"
    )
    plt.legend(loc="upper center")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.title("position-2d")
    plt.gca().set_aspect("equal")

    fig1 = plot_state.plot_state_euclidean(
        "position " + dataset,
        n,
        ["p_x", "p_y", "p_z"],
        ts,
        ps,
        linestyle=":",
        color="black",
    )
    fig9 = plot_state.plot_error_euclidean(
        "position error " + dataset,
        n,
        ["p_x", "p_y", "p_z"],
        ts,
        ps - ref_p,
        sigma=sigma_p,
        linestyle=":",
        color="black",
    )


def plot_autocorellation(args, log_folder, n, dataset):
    results_folder = log_folder + "/" + dataset
    filename = os.path.join(results_folder, args.log_filename + ".npy")
    try:
        logging.info(f"Loading {filename}")
        states = np.load(filename)
    except:
        logging.error(
            f"{filename}.npy was not found. Surely means filter did not finish"
        )
        raise FileNotFoundError

    # load all states
    meas = states[:, 46:49]
    if os.path.exists(os.path.join(results_folder, "vio_states.npy")):
        vio_states = np.load(os.path.join(results_folder, "vio_states.npy"))
        ref_disp = vio_states[:, 15:18]
    else:
        logging.error(
            "vio_states.npy was not found. you shoud create it with plot_state.py before... sorry :("
        )
        raise FileNotFoundError

    fig = plt.figure("autocorellation " + dataset)
    meas_err = meas - ref_disp
    meas_err_update = meas_err[~np.isnan(meas_err).any(axis=1)]
    logging.warning("We assume update frequency at 20hz for autocorrelation")
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.acorr(meas_err_update[:, i], maxlags=100, lw=2, usevlines=False, label=n)
        locs, labels = plt.xticks()  # Get locations and labels
        for (l, t) in zip(locs, labels):
            t.set_text(str(l / 20.0) + "s")
        plt.xticks(locs, labels)  # Set locations and labels
        plt.xlim(left=0)
        plt.grid()
    plt.legend()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--root_dir_regexp", type=str, default="./output_*")
    parser.add_argument("--data_list", type=str, default="../data/filenames.txt")
    parser.add_argument("--dataset_number", type=int, default=None)
    parser.add_argument("--log_filename", type=str, default="not_vio_state.txt")

    # expose various subcommand to plot
    subparsers = parser.add_subparsers()
    parser_bias = subparsers.add_parser("position")
    parser_bias.set_defaults(func=plot_position)
    parser_bias = subparsers.add_parser("plot_autocorellation")
    parser_bias.set_defaults(func=plot_position)

    args = parser.parse_args()
    print(vars(args))

    with open(args.data_list) as f:
        data_names = [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]
    if args.dataset_number is not None:
        dataset = data_names[args.dataset_number]
        logging.info(f"Plotting dataset {dataset}")
        log_folders = glob.glob(args.root_dir_regexp)
        logging.info(f"{log_folders}")
        for f in log_folders:
            if f == "./output_b1":
                continue
            logging.info(f"Processing run {f}")
            n = f
            try:
                args.func(args, f, n, dataset)
            except Exception as e:
                print(e)
                logging.warning(f"Something went wrong with {dataset}")
                continue
        plt.show()
    else:
        for dataset_number in range(len(data_names)):
            dataset = data_names[dataset_number]
            logging.info(f"Plotting dataset {dataset}")
            log_folders = glob.glob(args.root_dir_regexp)
            logging.info(f"{log_folders}")
            for f in log_folders:
                if f == "./output_b1":
                    continue
                logging.info(f"Processing run {f}")
                n = f
                try:
                    args.func(args, f, n, dataset)
                except KeyboardInterrupt:
                    plt.show()
                except Exception as e:
                    print(e)
                    logging.warning(f"Something went wrong with {dataset}")
                    continue

