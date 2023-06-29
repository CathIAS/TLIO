#!/usr/bin/env python3

import glob
import inspect
import json
import sys
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate


blacklisted_datasets = []
blacklisted_model = []
golden_shared_dataset = []


############## Loading data ###############


def remove_blacklist(d):
    # remove blacklisted datasets
    d = d[~d.dataset.isin(blacklisted_datasets)]
    return d


def load_folder_to_dataframe(name_run, split, folder):
    with open(folder + "/metrics.json", "r") as f:
        all_metrics = json.load(f)
        d = pd.DataFrame.from_dict(
            {
                (i, j): all_metrics[i][j]
                for i in all_metrics.keys()
                for j in all_metrics[i].keys()
            },
            orient="index",
        )

        d = d.sort_index()
        for format_table in ["presto"]:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(
                tabulate(
                    d, range(len(d.columns)), tablefmt=format_table, floatfmt=".2f"
                )
            )
            print("")
            print("Datasets:")
            print("=========")
            for n, name in zip(range(len(d.columns)), d.columns):
                print(f"{n} -> {name}")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print()
        d["type"] = list(d.index.get_level_values(0).str.split("_").str.get(0))
        d["hardware"] = [
            x
            for x in d.index.get_level_values(0)
            #.str.split("_")
            #.str.get(1)
            #.str.replace("hidacori", "")
        ]
        d["split"] = split
        d["name_run"] = osp.basename(name_run)

        d = d.reset_index()
        d = d.rename(columns={"level_0": "dataset", "level_1": "algo"})
        return d


def load_folder_dict(ndict, split):
    """
    Args:
        dict : "name_run" -> path
    """
    nmax_dataset = 0
    l = []
    for (k, v) in ndict.items():
        #try:
        l.append(load_folder_to_dataframe(k, split, v))
        #except:
        #    print("Could not read from ", v)
    dataset_length = [len(el["dataset"].unique()) for el in l]
    nmax_dataset = max(dataset_length)
    dataset_sets = [set(el["dataset"].unique()) for el in l]
    dataset_sets = set.intersection(*dataset_sets)
    if len(dataset_sets) < nmax_dataset:
        print("Some dataset were removed because no result for some run were found.")
        print(
            f"At least one run had {nmax_dataset}. While overlapping dataset vector size is {len(dataset_sets)}"
        )
        input("Press Enter to continue...")
    if len(dataset_sets) == 0:
        print("Could not find any common dataset!")
    for i in range(len(l)):
        l[i] = l[i][l[i].dataset.isin(dataset_sets)]

    d = pd.concat(l)
    d = d.sort_values("name_run")
    return d


############## Plotting functions ##############


def plot_var_boxplot(
    data,
    var,
    x="algo",
    hue="name_run",
    order=None,
    hue_order=None,
    palette="Set1",
    showfliers=False,
):
    p = sns.boxplot(
        x=x,
        y=var,
        hue=hue,
        order=order,
        data=data,
        hue_order=hue_order,
        palette=palette,
        whis=1.5,
        showfliers=showfliers,
        fliersize=2,
    )
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    # sns.swarmplot(x="algo", y=var, hue="name_run", data=data, palette="Set1", size=1, dodge=True)
    return p


def plot_var_cdf(data, var):
    linestyle_algo = {}
    linestyle_algo["filter"] = "-"
    linestyle_algo["ronin"] = ":"

    percentile = np.arange(0, 100.0 / 100, 0.1 / 100.0)
    color_rn = {}
    for i, nr in enumerate(data["name_run"].unique()):
        color_rn[nr] = "C" + str(i)

    for nr in data["name_run"].unique():
        drun = data[data.name_run == nr]
        for algo in drun["algo"].unique():
            d = drun[drun.algo == algo][var].quantile(percentile)
            plt.plot(
                d,
                percentile,
                linestyle=linestyle_algo[algo],
                color=color_rn[nr],
                label=f"{nr}-{algo}",
            )
    plt.xlim(left=0)
    plt.ylim([0, 1])
    plt.ylabel("cdf")
    plt.xlabel(var)
    plt.grid()


def plot_cdf_ax(data, var, ax, fontsize=10, fontname="Adobe Arabic"):
    linestyle_algo = {"filter": "-", "ronin": ":"}
    percentile = np.arange(0, 100.0 / 100, 0.1 / 100.0)
    color_rn = {}
    for i, nr in enumerate(data["name_run"].unique()):
        color_rn[nr] = "C" + str(i)
    for nr in data["name_run"].unique():
        drun = data[data.name_run == nr]
        for algo in drun["algo"].unique():
            d = drun[drun.algo == algo][var].quantile(percentile)
            ax.plot(
                d,
                percentile,
                linestyle=linestyle_algo[algo],
                color=color_rn[nr],
                label=f"{nr}-{algo}",
            )
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1])
    ax.set_xlabel(var, fontsize=fontsize, fontname=fontname)
    ax.grid()


############## Network result plots ##############

"""
Showing all metrics in bar plot for all models
"""


def plot_all_stats_net(d):
    # bar plot
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["rmse", "ate", "rmhe", "drift_pos (m/m)"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func, x="algo")
        plt.legend([])
    funcs = ["drift_yaw (deg/s)", "rpe", "rpe_z", "rpe_yaw"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot(d, func, x="algo")
        plt.legend([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 1))
    # plt.savefig('./netplot.svg')

    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["mse_loss_x", "mse_loss_y", "mse_loss_z", "mse_loss_avg"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func, x="algo")
        plt.legend([])
    funcs = [
        "likelihood_loss_x",
        "likelihood_loss_y",
        "likelihood_loss_z",
        "likelihood_loss_avg",
    ]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot(d, func, x="algo")
        plt.legend([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 1))
    # plt.savefig('./netlossplot.svg')
    plt.show()


############## Filter result plots ##############


def plot_rpe_stats(d):
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = [
        "relative_yaw_rmse_1000",
        "relative_yaw_rmse_10000",
        "relative_yaw_rmse_100000",
    ]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        bbox_transform=fig.transFigure,
    )

    funcs = [
        "relative_yaw_rmse_1000",
        "relative_yaw_rmse_10000",
        "relative_yaw_rmse_100000",
    ]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_cdf(d, func)
    plt.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
    )

    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["rpe_rmse_1000", "rpe_rmse_10000", "rpe_rmse_100000"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        bbox_transform=fig.transFigure,
    )

    funcs = ["rpe_rmse_1000", "rpe_rmse_10000", "rpe_rmse_100000"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_cdf(d, func)
    plt.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
    )
    plt.show()


def plot_all_stats(d, per="algo"):
    fig = plt.figure(figsize=(12, 8), dpi=90)
    funcs = ["ate", "rpe_rmse_1000", "drift_ratio"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func, x=per)
        plt.gca().legend().set_visible(False)
    funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot(d, func, x=per)
        plt.gca().legend().set_visible(False)
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        bbox_transform=fig.transFigure,
    )
    plt.show()

    # Plot CDF
    fig = plt.figure(figsize=(12, 8), dpi=90)
    funcs = ["ate", "rpe_rmse_1000", "drift_ratio"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_cdf(d, func)

    funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_cdf(d, func)
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        bbox_transform=fig.transFigure,
    )
    plt.show()


"""
For the paper
"""


def plot_sysperf_cdf(
    d, ticksize=16, fontsize=20, tickfont="Crimson Text", fontname="Crimson Text"
):
    d = d.rename(
        columns={
            "ate": "ATE (m)",
            "rpe_rmse_1000": "RPE-1s (m)",
            "drift_ratio": "Drift (%)",
            "mhe": "AYE (deg)",
            "relative_yaw_rmse_1000": "RYE-1s (deg)",
            "angular_drift_deg_hour": "Yaw-Drift (deg/hour)",
        }
    )
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), dpi=90)
    funcs = ["ATE (m)", "RPE-1s (m)", "Drift (%)"]
    for i, func in enumerate(funcs):
        plot_cdf_ax(d, func, axs[0, i], fontsize=fontsize, fontname=fontname)
    funcs = ["AYE (deg)", "RYE-1s (deg)", "Yaw-Drift (deg/hour)"]
    for i, func in enumerate(funcs):
        plot_cdf_ax(d, func, axs[1, i], fontsize=fontsize, fontname=fontname)

    for i in range(2):
        axs[i, 0].set_ylabel("CDF", fontsize=fontsize, fontname=fontname)
        for j in range(3):
            plt.setp(axs[i, j].get_xticklabels(), fontsize=ticksize, fontname=tickfont)
            plt.setp(axs[i, j].get_yticklabels(), fontsize=ticksize, fontname=tickfont)
    leg = plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        bbox_transform=fig.transFigure,
        fontsize=fontsize,
    )
    plt.setp(leg.texts, family=fontname)
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def plot_comparison_cdf(
    d, ticksize=16, fontsize=20, tickfont="Crimson Text", fontname="Crimson Text"
):
    d = d.rename(
        columns={
            "ate": "ATE (m)",
            "rpe_rmse_1000": "RPE-1s (m)",
            "drift_ratio": "Drift (%)",
            "mhe": "AYE (deg)",
            "relative_yaw_rmse_1000": "RYE-1s (deg)",
            "angular_drift_deg_hour": "Yaw-Drift (deg/hour)",
        }
    )
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), dpi=90)
    funcs = ["ATE (m)", "Drift (%)", "AYE (deg)"]
    for i, func in enumerate(funcs):
        plot_cdf_ax(d, func, axs[i], fontsize=fontsize, fontname=fontname)
    for i in range(3):
        axs[0].set_ylabel("CDF", fontsize=fontsize, fontname=fontname)
        plt.setp(axs[i].get_xticklabels(), fontsize=ticksize, fontname=tickfont)
        plt.setp(axs[i].get_yticklabels(), fontsize=ticksize, fontname=tickfont)
    leg = plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        bbox_transform=fig.transFigure,
        fontsize=fontsize - 3,
    )
    plt.setp(leg.texts, family=fontname)
    plt.subplots_adjust(hspace=0.4, bottom=0.4)
    plt.show()


def plot_net(
    d, ticksize=16, fontsize=20, tickfont="Crimson Text", fontname="Crimson Text"
):
    d = d.rename(
        columns={
            "rmse": "ATE (m)",
            "rpe": "RPE (m)",
            "drift_pos (m/m)": "DR (%)",
            "mse_loss_avg": "Avg MSE Loss",
        }
    )
    colorwheel = [
        "#326e3eff",
        "#469e5aff",
        "#6abe7eff",
        "#a0d6acff",
        "#204f8fff",
        "#3578d2ff",
        "#78a5e1ff",
        "#c75572ff",
        "#de9aacff",
    ]
    fig = plt.figure(figsize=(8, 3), dpi=90)
    funcs = ["ATE (m)", "Avg MSE Loss"]
    for i, func in enumerate(funcs):
        ax = plt.subplot2grid([1, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func, x="algo", palette=sns.color_palette(colorwheel))
        plt.legend([])
        ax.set_ylabel(func, fontsize=fontsize, fontname=fontname)
        plt.setp(ax.get_xticklabels(), fontsize=ticksize, fontname=tickfont)
        plt.setp(ax.get_yticklabels(), fontsize=ticksize, fontname=tickfont)
    fig.tight_layout()
    leg = plt.legend(
        loc="upper center", bbox_to_anchor=(-0.2, 0), fancybox=True, shadow=True, ncol=5
    )
    plt.setp(leg.texts, family=fontname, fontsize=fontsize)
    plt.show()


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
        "--glob_dataset",
        type=str,
        default="",
        help="globbing pattern for dataset directories",
    )
    args = parser.parse_args()

    dct = {osp.dirname(p): p for p in glob.glob(args.glob_dataset)}
    print("Found the following tested model", dct)

    d = load_folder_dict(dct, "test")

    print("Dropping you in interactive shell : values are in DataFrame d")
    print("Function possibles are:")
    for n in list_of_plots:
        print("\t" + n)
    print("Suggest to start with plt.ion()")
    print()
    import IPython

    IPython.embed()
