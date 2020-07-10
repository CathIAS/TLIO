#!/usr/bin/env python3

"""
usage: put under source folder, required files: evolving_state.txt, calib_state.txt, state.txt
After first run, integration_states.txt, vio_states.txt are generated and figures are saved in current dir
You can move the figures and state.txt, integration_states.txt, vio_states.txt into a folder
Rerun to generate graphs more efficiently by specifying the folder names that has the above three files
"""


import glob
import inspect
import json
import os
import sys
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate


blacklisted_datasets = []
blacklisted_datasets.append("loop_hidacori058_20180519_1525")
blacklisted_datasets.append("loop_hidacori058_20180524_1121")

blacklisted_model = []
blacklisted_model.append("200hz-retrain-05s-1s")
blacklisted_model.append("200hz-retrain-05s-2s")
blacklisted_model.append("200hz-retrain-05s-3s")
blacklisted_model.append("200hz-retrain-1s-2s")
blacklisted_model.append("200hz-retrain-1s-3s")
blacklisted_model.append("200hz-retrain-2s-3s")

golden_shared_dataset = [
    "loop_hidacori058_20180524_1121",
    "loop_hidacori0511_20180815_1152",
    "loop_hidacori059_20180522_1356",
    "loop_hidacori0511_20180808_1011",
    "loop_hidacori0512_20180731_1707",
]


def read_run_parameters(folder):
    with open(folder + "/parameters.json", "r") as f:
        parameters = json.load(f)
        return parameters


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
            .str.split("_")
            .str.get(1)
            .str.replace("hidacori", "")
        ]
        d["split"] = split
        d["name_run"] = osp.basename(name_run)

        d = d.reset_index()
        d = d.rename(
            columns={
                "level_0": "dataset",
                "level_1": "algo",
                "rmse": "ATE (m)",
                "rpe": "RPE (m)",
                "drift_pos (m/m)": "DR (%)",
                "mse_loss_avg": "avg MSE loss",
            }
        )

        return d


def load_folder_dict(ndict, split):
    """
    Args:
        dict : "name_run" -> path
    """
    nmax_dataset = 0
    l = []
    for (k, v) in ndict.items():
        try:
            l.append(load_folder_to_dataframe(k, split, v))
        except:
            print("Could not read from ", v)
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


def filter_results(d):
    d = d[~d.name_run.isin(blacklisted_model)]
    d = d[~d.dataset.isin(blacklisted_datasets)]
    return d


def filter_get_test_split_shared(d):
    d = d[d.dataset.isin(golden_shared_dataset)]
    return d


def plot_var_boxplot_per(data, var, per="algo"):
    if len(data["name_run"].unique()) == 1:
        sns.boxplot(x=per, y=var, data=data, whis=[0, 1])
        sns.swarmplot(
            x=per, y=var, data=data, color="black", edgecolor="black", dodge=True
        )
    else:
        ax = sns.boxplot(
            x=per,
            y=var,
            hue="name_run",
            data=data,
            palette="Set1",
            whis=1.5,
            showfliers=True,
            fliersize=2,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        # sns.swarmplot(x="algo", y=var, hue="name_run", data=data, palette="Set1", size=1, dodge=True)


def plot_var_boxplot_per_time_window(data, var):
    sns.boxplot(
        x="disp_window_s",
        y=var,
        hue="update_frequency",
        data=data,
        palette="Set1",
        whis=[0, 100],
    )


def plot_var_boxplot_per_update_frequency(data, var):
    sns.boxplot(
        x="update_frequency",
        y=var,
        hue="disp_window_s",
        data=data,
        palette="Set1",
        whis=[0, 100],
    )


def plot_var_boxplot_per_imu_frequency(data, var):
    sns.boxplot(
        x="update_frequency",
        y=var,
        hue="imu_freq_hz",
        data=data,
        palette="Set1",
        whis=[0, 100],
    )


def plot_var_cdf(data, var):
    from cycler import cycler

    ax = plt.gca()
    percentile = np.arange(0, 100.0 / 100, 0.1 / 100.0)
    if len(data["name_run"].unique()) == 1:
        linestyle_algo = {}
        linestyle_algo["filter"] = "-"
        linestyle_algo["ronin"] = ":"

        color_algo = {}
        for i, nr in enumerate(data["algo"].unique()):
            color_algo[nr] = "C" + str(i)

        for algo in data["algo"].unique():
            d = data[data.algo == algo][var].quantile(percentile)
            plt.plot(
                d,
                percentile,
                linestyle=linestyle_algo[algo],
                color=color_algo[algo],
                label=f"{algo}",
            )
        plt.xlim(left=0)
        plt.ylim([0, 1])
        plt.ylabel("cdf")
        plt.xlabel(var)
        plt.grid()
    else:
        linestyle_algo = {}
        linestyle_algo["filter"] = "-"
        linestyle_algo["ronin"] = ":"

        color_rn = {}
        for i, nr in enumerate(data["name_run"].unique()):
            color_rn[nr] = "C" + str(i)

        for nr in data["name_run"].unique():
            drun = data[data.name_run == nr]
            for algo in data["algo"].unique():
                d = drun[drun.algo == algo][var].quantile(percentile)
                plt.plot(
                    d,
                    percentile,
                    linestyle=linestyle_algo[algo],
                    color=color_rn[nr],
                    label=f"{nr}-{algo}",
                )
        plt.ylim([0, 1])
        plt.xlim(left=0)
        plt.ylabel("cdf")
        plt.xlabel(var)
        plt.grid()


def plot_rpe_stats(d):
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = [
        "relative_yaw_rmse_1000",
        "relative_yaw_rmse_10000",
        "relative_yaw_rmse_100000",
    ]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot_per(d, func)
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
        plot_var_boxplot_per(d, func)
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


def plot_all_stats(d, per="algo"):
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["ate", "rpe_rmse_1000", "drift_ratio"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot_per(d, func, per)
        plt.gca().legend().set_visible(False)
    funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot_per(d, func, per)
        plt.gca().legend().set_visible(False)
    plt.subplots_adjust(bottom=0.3)
    plt.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        bbox_transform=fig.transFigure,
    )
    # plt.savefig('./barplot.svg', bbox_inches='tight')
    plt.show()

    # Plot CDF
    fig = plt.figure(figsize=(16, 9), dpi=90)
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
    # plt.savefig('./cdfplot.svg', bbox_inches='tight')
    plt.show()


def plot_all_stats_ronin_vs_params(d):
    # copy d and rename name_run with configuration
    d = d.copy()
    print(
        "Assuming one model per tuple [imu_freq_hz,disp_window_s,data_window_s], otherwise, the comparison is doomed"
    )
    # remove roning duplicate, its results does not change with update_frequency as it is same model
    d["name_model"] = (
        d.imu_freq_hz.astype(str)
        + "hz-"
        + d.disp_window_s.astype(str)
        + "s-"
        + d.data_window_s.astype(str)
        + "s"
    )
    # drop duplicate ronin because
    d.loc[d.algo == "ronin", :].drop_duplicates(subset=["name_model"], inplace=True)

    dfilter = d.loc[d.algo == "filter", :]
    d.loc[d.algo == "filter", "name_run"] = (
        "upd_at_imu_freq_over_" + dfilter.update_decimator.astype(str) + "_hz"
    )
    dronin = d.loc[d.algo == "ronin", :]
    d.loc[d.algo == "ronin", "name_run"] = "ronin"

    d = d.sort_values(["algo", "update_frequency"])
    plot_all_stats(d, per="name_model")


def plot_all_stats_one_algo_only(d, algo="both", advanced_analysis=False):
    # copy d and rename name_run with configuration
    d = d.copy()
    d.loc[:, "name_run"] = (
        d.loc[:, "name_run"].imu_freq_hz.astype(str)
        + "hz-"
        + d.loc[:, "name_run"].disp_window_s.astype(str)
        + "s-"
        + d.loc[:, "name_run"].data_window_s.astype(str)
        + "s"
        + dfilter.update_frequency.astype(str)
        + "hz-filter"
    )
    d = d[d.algo == algo]
    d = d.sort_values("name_run")
    for sort_v in ["imu_freq_hz", "update_frequency", "disp_window_s"]:
        d = d.sort_values(sort_v)
        plot_all_stats(d)

    if advanced_analysis:
        for hz in d.imu_freq_hz.unique():
            fig = plt.figure(f"{hz}hz", figsize=(16, 9), dpi=90)
            funcs = ["ate", "drift_ratio", "rpe_rmse_1000", "rpe_rmse_z_1000"]
            for i, func in enumerate(funcs):
                plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
                plot_var_boxplot_per_time_window(d[d.imu_freq_hz == hz], func)
                plt.gca().legend().set_visible(False)
            funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
            for i, func in enumerate(funcs):
                plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
                plot_var_boxplot_per_time_window(d[d.imu_freq_hz == hz], func)
                plt.gca().legend().set_visible(False)
            plt.subplots_adjust(bottom=0.3)
            plt.legend(
                ncol=3,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.2),
                bbox_transform=fig.transFigure,
            )

        for hz in d.imu_freq_hz.unique():
            fig = plt.figure(f"{hz}hz-update", figsize=(16, 9), dpi=90)
            funcs = ["ate", "drift_ratio", "rpe_rmse_1000", "rpe_rmse_z_1000"]
            for i, func in enumerate(funcs):
                plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
                plot_var_boxplot_per_update_frequency(d[d.imu_freq_hz == hz], func)
                plt.gca().legend().set_visible(False)
            funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
            for i, func in enumerate(funcs):
                plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
                plot_var_boxplot_per_update_frequency(d[d.imu_freq_hz == hz], func)
                plt.gca().legend().set_visible(False)
            plt.subplots_adjust(bottom=0.3)
            plt.legend(
                ncol=3,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.2),
                bbox_transform=fig.transFigure,
            )

        for dw in d.disp_window_s.unique():
            fig = plt.figure(f"{dw}-disp_window_s", figsize=(16, 9), dpi=90)
            funcs = ["ate", "drift_ratio", "rpe_rmse_1000", "rpe_rmse_z_1000"]
            for i, func in enumerate(funcs):
                plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
                plot_var_boxplot_per_imu_frequency(d[d.disp_window_s == dw], func)
                plt.gca().legend().set_visible(False)
            funcs = ["mhe", "relative_yaw_rmse_1000", "angular_drift_deg_hour"]
            for i, func in enumerate(funcs):
                plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
                plot_var_boxplot_per_imu_frequency(d[d.disp_window_s == dw], func)
                plt.gca().legend().set_visible(False)
            plt.subplots_adjust(bottom=0.3)
            plt.legend(
                ncol=3,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.2),
                bbox_transform=fig.transFigure,
            )


def plot_comparison_fixed_non_fixed_cov(d):
    # select run that have a fixed non fixed version
    g = d.groupby(
        ["update_frequency", "update_frequency", "disp_window_s", "data_window_s"]
    )
    filtered_d = []
    for [k, df] in g:
        if len(df.fixed_cov.unique()) == 2:
            filtered_d.append(df)
    filtered_d = pd.concat(filtered_d)
    # filtered_d = filtered_d[filtered_d.algo=="filter"]
    plot_all_stats(filtered_d)


def plot_all_stats_net(d):
    # bar plot
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["rmse", "ate", "rmhe", "drift_pos (m/m)"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func)
        plt.legend([])
    funcs = ["drift_yaw (deg/s)", "rpe", "rpe_z", "rpe_yaw"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot_per(d, func, "algo")
        plt.legend([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 1))
    plt.savefig("./netplot.svg")

    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["mse_loss_x", "mse_loss_y", "mse_loss_z", "mse_loss_avg"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot_per(d, func, "algo")
        plt.legend([])
    funcs = [
        "likelihood_loss_x",
        "likelihood_loss_y",
        "likelihood_loss_z",
        "likelihood_loss_avg",
    ]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_boxplot_per(d, func, "algo")
        plt.legend([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 1))
    plt.savefig("./netlossplot.svg")

    # cdf plot
    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["rmse", "ate", "rmhe", "drift_pos (m/m)"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_cdf(d, func)
    funcs = ["drift_yaw (deg/s)", "rpe", "rpe_z", "rpe_yaw"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_cdf(d, func)
    plt.legend(loc="center left", bbox_to_anchor=(1, 1))
    plt.savefig("./netplotcdf.svg")

    fig = plt.figure(figsize=(16, 9), dpi=90)
    funcs = ["mse_loss_x", "mse_loss_y", "mse_loss_z", "mse_loss_avg"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [0, i], fig=fig)
        plot_var_cdf(d, func)
        plt.legend([])
    funcs = [
        "likelihood_loss_x",
        "likelihood_loss_y",
        "likelihood_loss_z",
        "likelihood_loss_avg",
    ]
    for i, func in enumerate(funcs):
        plt.subplot2grid([2, len(funcs)], [1, i], fig=fig)
        plot_var_cdf(d, func)
        plt.legend([])
    plt.legend(loc="center left", bbox_to_anchor=(1, 1))
    plt.savefig("./netlossplotcdf.svg")


def plot_var_boxplot(data, var):
    if len(data["name_run"].unique()) == 1:
        sns.boxplot(x="algo", y=var, data=data, whis=[0, 1])
        sns.swarmplot(
            x="algo", y=var, data=data, color="black", edgecolor="black", dodge=True
        )
    else:
        sns.boxplot(
            x="algo",
            y=var,
            hue="name_run",
            data=data,
            palette="Set1",
            whis=1.5,
            showfliers=False,
            fliersize=2,
        )
        # sns.swarmplot(x="algo", y=var, hue="name_run", data=data, palette="Set1", size=1, dodge=True)


def plot_net(d):
    # bar plot
    fig = plt.figure(figsize=(8, 3), dpi=90)
    funcs = ["ATE (m)", "avg MSE loss"]
    for i, func in enumerate(funcs):
        plt.subplot2grid([1, len(funcs)], [0, i], fig=fig)
        plot_var_boxplot(d, func)
        plt.legend([])
    fig.tight_layout()
    plt.legend(
        loc="upper center", bbox_to_anchor=(-0.2, 0), fancybox=True, shadow=True, ncol=5
    )
    plt.show()

    # fig = plt.figure(figsize=(16,9), dpi=90)
    # funcs = ['mse_loss_x', 'mse_loss_y', 'mse_loss_z', 'mse_loss_avg']
    # for i, func in enumerate(funcs):
    #    plt.subplot2grid([2,len(funcs)],[0,i], fig=fig)
    #    plot_var_boxplot(d, func)
    #    plt.legend([])
    # funcs = ['likelihood_loss_x', 'likelihood_loss_y', 'likelihood_loss_z', 'likelihood_loss_avg']
    # for i, func in enumerate(funcs):
    #    plt.subplot2grid([2,len(funcs)],[1,i], fig=fig)
    #    plot_var_boxplot(d, func)
    #    plt.legend([])
    # plt.legend(loc='center left', bbox_to_anchor=(1,1))
    # plt.show()


def plot_by_hardware(d):
    for n in [
        "ate_filter",
        "ate_ronin",
        "drift_filter_ratio",
        "drift_ronin_ratio",
        "mhe_filter",
        "mhe_ronin",
        "rpe_rmse_filter_1000",
        "rpe_rmse_ronin_1000",
    ]:
        d.boxplot(column=[n], by="hardware")


def getfunctions(module):
    import types

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
        default="../all_net_output_20hz/*/",
        help="globbing pattern for dataset directories",
    )
    parser.add_argument(
        "--is_net",
        type=bool,
        default=True,
        help="using network output (different labels)",
    )
    args = parser.parse_args()

    dct = {osp.dirname(p): p for p in glob.glob(args.glob_dataset)}
    print(dct)

    d = load_folder_dict(dct, "test")

    print("Dropping you in interactive shell : values are in DataFrame d")
    print("Function possibles are:")
    for n in list_of_plots:
        print("\t" + n)
    print("Suggest to start with plt.ion()")
    print()
    import IPython

    IPython.embed()
