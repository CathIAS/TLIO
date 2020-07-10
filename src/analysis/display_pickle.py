#!/usr/bin/env python3

import glob
import inspect
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_pickle_to_dataframe(args, name_path):
    d = pd.read_pickle(name_path)
    d = d.sort_index()
    d["model"] = os.path.basename(os.path.dirname(name_path))
    d["name_run"] = os.path.basename(name_path)
    if args.perturbation:
        leg_acc = {
            "d-bias-0.0-0.0-grav-0.0.pkl": "0.0",
            "d-bias-0.1-0.0-grav-0.0.pkl": "0.1",
            "d-bias-0.2-0.0-grav-0.0.pkl": "0.2",
            "d-bias-0.3-0.0-grav-0.0.pkl": "0.3",
            "d-bias-0.4-0.0-grav-0.0.pkl": "0.4",
            "d-bias-0.5-0.0-grav-0.0.pkl": "0.5",
            "d-bias-0.0-0.025-grav-0.0.pkl": "0.025",
            "d-bias-0.0-0.05-grav-0.0.pkl": "0.05",
            "d-bias-0.0-0.075-grav-0.0.pkl": "0.075",
            "d-bias-0.0-0.1-grav-0.0.pkl": "0.1",
            "d-bias-0.0-0.0-grav-2.0.pkl": "2.0",
            "d-bias-0.0-0.0-grav-4.0.pkl": "4.0",
            "d-bias-0.0-0.0-grav-6.0.pkl": "6.0",
            "d-bias-0.0-0.0-grav-8.0.pkl": "8.0",
            "d-bias-0.0-0.0-grav-10.0.pkl": "10.0",
        }
        d["legend"] = leg_acc[os.path.basename(name_path)]

    return d


def load_folder_list(args, ndict):
    """
    Args:
        dict : "name_run" -> path
    """
    l = []
    for p in ndict:
        print("loading %s" % p)
        l.append(load_pickle_to_dataframe(args, p))
    d = pd.concat(l)
    d = d.sort_values("name_run")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Datasets:")
    print("=========")
    for n, name in zip(range(len(d.columns)), d.columns):
        print(f"{n} -> {name}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print()
    return d


def plot_var_boxplot(
    data, var, x="algo", hue="name_run", order=None, hue_order=None, showfliers=False
):
    if len(data["name_run"].unique()) == 1:
        p = sns.boxplot(x=x, y=var, data=data, whis=[0, 1])
    else:
        p = sns.boxplot(
            x=x,
            y=var,
            hue=hue,
            order=order,
            data=data,
            hue_order=hue_order,
            palette="Set1",
            whis=1.5,
            showfliers=showfliers,
            fliersize=2,
        )
        # ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
        # sns.swarmplot(x="algo", y=var, hue="name_run", data=data, palette="Set1", size=1, dodge=True)
    return p


def group_perturbation_data(d):
    d_acc = d.loc[
        d["name_run"].isin(
            [
                "d-bias-0.0-0.0-grav-0.0.pkl",
                "d-bias-0.1-0.0-grav-0.0.pkl",
                "d-bias-0.2-0.0-grav-0.0.pkl",
                "d-bias-0.3-0.0-grav-0.0.pkl",
                "d-bias-0.4-0.0-grav-0.0.pkl",
                "d-bias-0.5-0.0-grav-0.0.pkl",
            ]
        )
    ]
    d_gyr = d.loc[
        d["name_run"].isin(
            [
                "d-bias-0.0-0.0-grav-0.0.pkl",
                "d-bias-0.0-0.025-grav-0.0.pkl",
                "d-bias-0.0-0.05-grav-0.0.pkl",
                "d-bias-0.0-0.075-grav-0.0.pkl",
                "d-bias-0.0-0.1-grav-0.0.pkl",
            ]
        )
    ]
    d_grav = d.loc[
        d["name_run"].isin(
            [
                "d-bias-0.0-0.0-grav-0.0.pkl",
                "d-bias-0.0-0.0-grav-2.0.pkl",
                "d-bias-0.0-0.0-grav-4.0.pkl",
                "d-bias-0.0-0.0-grav-6.0.pkl",
                "d-bias-0.0-0.0-grav-8.0.pkl",
                "d-bias-0.0-0.0-grav-10.0.pkl",
            ]
        )
    ]
    return d_acc, d_gyr, d_grav


def plot_comparison(
    data, ticksize=16, fontsize=20, tickfont="Crimson Text", fontname="Crimson Text"
):
    # adding legends
    d_acc, d_gyr, d_grav = group_perturbation_data(data)

    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["mse_losses_x", "mse_losses_y", "mse_losses_z", "avg_mse_losses"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(
            data,
            func,
            x="legend",
            hue="model",
            order=None,
            hue_order=["noptrb", "bias", "bias-grav"],
        )
        plt.legend([])
    funcs = [
        "likelihood_losses_x",
        "likelihood_losses_y",
        "likelihood_losses_z",
        "avg_likelihood_losses",
    ]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot(
            data,
            func,
            x="legend",
            hue="model",
            order=None,
            hue_order=["noptrb", "bias", "bias-grav"],
        )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), dpi=90)
    for i in range(3):
        plt.sca(axs[i])
        if i == 2:
            plot_var_boxplot(
                d_grav,
                "avg_mse_losses",
                x="legend",
                hue="model",
                order=["0.0", "2.0", "4.0", "6.0", "8.0", "10.0"],
                hue_order=["noptrb", "bias", "bias-grav"],
            )
        elif i == 0:
            plot_var_boxplot(
                d_acc,
                "avg_mse_losses",
                x="legend",
                hue="model",
                order=None,
                hue_order=["noptrb", "bias", "bias-grav"],
            )
        elif i == 1:
            plot_var_boxplot(
                d_gyr,
                "avg_mse_losses",
                x="legend",
                hue="model",
                order=None,
                hue_order=["noptrb", "bias", "bias-grav"],
            )
        plt.setp(axs[i].get_xticklabels(), fontsize=ticksize, fontname=tickfont)
        plt.setp(axs[i].get_yticklabels(), fontsize=ticksize, fontname=tickfont)
    axs[0].set_xlabel("Accel Bias (m/s^2)", fontsize=fontsize, fontname=fontname)
    axs[0].set_ylabel("Avg MSE Loss", fontsize=fontsize, fontname=fontname)
    axs[1].set_xlabel("Gyro Bias (rad/s)", fontsize=fontsize, fontname=fontname)
    axs[1].set_ylabel(None)
    axs[2].set_xlabel("Gravity Direction (deg)", fontsize=fontsize, fontname=fontname)
    axs[2].set_ylabel(None)
    axs[0].legend([])
    axs[1].legend([])
    leg = axs[2].legend(loc="upper left")
    plt.setp(leg.texts, family=fontname, fontsize=fontsize)
    plt.tight_layout(pad=0.2)
    plt.show()


def plot_sigmas(
    data, ticksize=16, fontsize=20, tickfont="Crimson Text", fontname="Crimson Text"
):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), dpi=90)
    # gs = gridspec.GridSpec(1, 3)
    ## gs.update(wspace=0.3,hspace=0)
    # plt.subplot(gs[0])
    errs = ["errors_x", "errors_y", "errors_z"]
    sigs = ["sigmas_x", "sigmas_y", "sigmas_z"]
    for i in range(3):
        plt.sca(axs[i])
        plt.scatter(d[errs[i]], d[sigs[i]], s=0.3)
        l = mlines.Line2D([-3, 3], [-1, 1], color="r", linestyle="--", linewidth=0.7)
        axs[i].add_line(l)
        l = mlines.Line2D([3, -3], [-1, 1], color="r", linestyle="--", linewidth=0.7)
        axs[i].add_line(l)
        plt.xlim((-1, 1))
        plt.ylim((-0.01, 0.32))
        plt.setp(axs[i].get_xticklabels(), fontsize=ticksize, fontname=tickfont)
        plt.setp(axs[i].get_yticklabels(), fontsize=ticksize, fontname=tickfont)
        plt.grid(True)

    axs[0].set_ylabel("Sigmas", fontsize=fontsize, fontname=fontname)
    axs[0].set_xlabel("Error X", fontsize=fontsize, fontname=fontname)
    axs[1].set_xlabel("Error Y", fontsize=fontsize, fontname=fontname)
    axs[2].set_xlabel("Error Z", fontsize=fontsize, fontname=fontname)
    plt.tight_layout(pad=0.2)
    plt.show()


def get_percentage_outside_3sigma(d):
    mask_x = d["errors_x"] > 3 * d["sigmas_x"]
    mask_y = d["errors_y"] > 3 * d["sigmas_y"]
    mask_z = d["errors_z"] > 3 * d["sigmas_z"]
    total_n = d.shape[0]
    perc_x = sum(mask_x) / total_n
    perc_y = sum(mask_y) / total_n
    perc_z = sum(mask_z) / total_n
    return perc_x, perc_y, perc_z


def get_mahalanobis_stats(d):
    mean = np.mean(d["mahalanobis"])
    var = np.var(d["mahalanobis"])
    total_n = d.shape[0]
    perc99 = sum(d["mahalanobis"] > 11.345) / total_n
    return mean, var, perc99


def get_sigmas_from_errors(d):
    total_n = d.shape[0]
    ind_67 = int(0.67 * total_n)
    err_x = np.sort(np.abs(d["errors_x"].to_numpy()))
    err_y = np.sort(np.abs(d["errors_y"].to_numpy()))
    err_z = np.sort(np.abs(d["errors_z"].to_numpy()))
    x_67 = err_x[ind_67]
    y_67 = err_y[ind_67]
    z_67 = err_z[ind_67]
    return x_67, y_67, z_67


def getfunctions(module):
    l = []
    for key, value in module.__dict__.items():
        if inspect.isfunction(value):
            l.append(value)
    return l


list_of_plots = [o.__name__ for o in getfunctions(sys.modules[__name__])]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob_pickle",
        type=str,
        default="../results/*/*.pkl",
        help="globbing pattern for dataset pickle files",
    )
    parser.add_argument("--perturbation", action="store_true")
    args = parser.parse_args()

    dct_list = [p for p in glob.glob(args.glob_pickle)]
    print(dct_list)

    d = load_folder_list(args, dct_list)

    print("Dropping you in interactive shell : values are in DataFrame d")
    print("Function possibles are:")
    for n in list_of_plots:
        print("\t" + n)
    print("Suggest to start with plt.ion()")
    print()
    import IPython

    IPython.embed()
