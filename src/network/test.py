"""
This file includes the main libraries in the network testing module
"""

import json
import os
from os import path as osp

import matplotlib.pyplot as plt
import torch
from dataloader.dataset_fb import FbSequenceDataset
from network.losses import get_loss
from network.model_factory import get_model
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from utils.logging import logging
from utils.math_utils import *


def compute_rpe(rpe_ns, ps, ps_gt, yaw, yaw_gt):
    ns = ps_gt.shape[0]
    assert ns - rpe_ns > 100
    assert ps.shape == ps_gt.shape
    assert yaw.shape == yaw_gt.shape

    rpes = []
    relative_yaw_errors = []
    for i in range(0, ns - rpe_ns, 100):
        chunk = ps[i : i + rpe_ns, :]
        chunk_gt = ps_gt[i : i + rpe_ns, :]
        chunk_yaw = yaw[i : i + rpe_ns, :]
        chunk_yaw_gt = yaw_gt[i : i + rpe_ns, :]
        initial_error_yaw = wrap_rpy(chunk_yaw[0, :] - chunk_yaw_gt[0, :])
        final_error_p_relative = Rotation.from_euler(
            "z", initial_error_yaw, degrees=True
        ).as_matrix().dot((chunk[[-1], :] - chunk[[0], :]).T)[0, :, :].T - (
            chunk_gt[[-1], :] - chunk_gt[[0], :]
        )
        final_error_yaw = wrap_rpy(chunk_yaw[[-1], :] - chunk_yaw_gt[[-1], :])
        rpes.append(final_error_p_relative)
        relative_yaw_errors.append(wrap_rpy(final_error_yaw - initial_error_yaw))
    rpes = np.concatenate(rpes, axis=0)
    relative_yaw_errors = np.concatenate(relative_yaw_errors, axis=0)

    plt.figure("relative yaw error")
    plt.plot(relative_yaw_errors)
    plt.figure("rpes list")
    plt.plot(rpes)
    # compute statistics over z separately
    rpe_rmse = np.sqrt(np.mean(np.sum(rpes ** 2, axis=1)))
    rpe_rmse_z = np.sqrt(np.mean(rpes[:, 2] ** 2))
    relative_yaw_rmse = np.sqrt(np.mean(relative_yaw_errors ** 2))
    return rpe_rmse, rpe_rmse_z, relative_yaw_rmse, rpes


def pose_integrate(args, dataset, preds):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    dp_t = args.window_time
    pred_vels = preds / dp_t

    ind = np.array([i[1] for i in dataset.index_map], dtype=np.int)
    delta_int = int(
        args.window_time * args.imu_freq / 2.0
    )  # velocity as the middle of the segment
    if not (args.window_time * args.imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    ind_intg = ind + delta_int  # the indices of doing integral

    ts = dataset.ts[0]
    dts = np.mean(ts[ind_intg[1:]] - ts[ind_intg[:-1]])
    pos_intg = np.zeros([pred_vels.shape[0] + 1, args.output_dim])
    pos_intg[0] = dataset.gt_pos[0][ind_intg[0], :]
    pos_intg[1:] = np.cumsum(pred_vels[:, :] * dts, axis=0) + pos_intg[0]
    ts_intg = np.append(ts[ind_intg], ts[ind_intg[-1]] + dts)

    ts_in_range = ts[ind_intg[0] : ind_intg[-1]]  # s
    pos_pred = interp1d(ts_intg, pos_intg, axis=0)(ts_in_range)
    pos_gt = dataset.gt_pos[0][ind_intg[0] : ind_intg[-1], :]
    ori_pred = dataset.orientations[0][ind_intg[0] : ind_intg[-1], :]
    ori_gt = dataset.gt_ori[0][ind_intg[0] : ind_intg[-1], :]
    eul_pred = Rotation.from_quat(ori_pred).as_euler("xyz", degrees=True)
    eul_gt = Rotation.from_quat(ori_gt).as_euler("xyz", degrees=True)

    traj_attr_dict = {
        "ts": ts_in_range,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "eul_pred": eul_pred,
        "eul_gt": eul_gt,
    }

    return traj_attr_dict


def compute_metrics_and_plotting(args, net_attr_dict, traj_attr_dict):
    """
    Obtain trajectory and compute metrics.
    """

    """ ------------ Trajectory metrics ----------- """
    ts = traj_attr_dict["ts"]
    pos_pred = traj_attr_dict["pos_pred"]
    pos_gt = traj_attr_dict["pos_gt"]
    eul_pred = traj_attr_dict["eul_pred"]
    eul_gt = traj_attr_dict["eul_gt"]

    # get RMSE
    rmse = np.sqrt(np.mean(np.linalg.norm(pos_pred - pos_gt, axis=1) ** 2))
    # get ATE
    diff_pos = pos_pred - pos_gt
    ate = np.mean(np.linalg.norm(diff_pos, axis=1))
    # get RMHE (yaw)
    diff_eul = wrap_rpy(eul_pred - eul_gt)
    rmhe = np.sqrt(np.mean(diff_eul[:, 2] ** 2))
    # get position drift
    traj_lens = np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1))
    drift_pos = np.linalg.norm(pos_pred[-1, :] - pos_gt[-1, :])
    drift_ratio = drift_pos / traj_lens
    # get yaw drift
    duration = ts[-1] - ts[0]
    drift_ang = np.linalg.norm(
        diff_eul[-1, 2] - diff_eul[0, 2]
    )  # beginning not aligned
    drift_ang_ratio = drift_ang / duration
    # get RPE on position and yaw
    ns_rpe = int(args.rpe_window * args.imu_freq)
    rpe_rmse, rpe_rmse_z, relative_yaw_rmse, rpes = compute_rpe(
        ns_rpe, pos_pred, pos_gt, eul_pred[:, [2]], eul_gt[:, [2]]
    )

    metrics = {
        "ronin": {
            "rmse": rmse,
            "ate": ate,
            "rmhe": rmhe,
            "drift_pos (m/m)": drift_ratio,
            "drift_yaw (deg/s)": drift_ang_ratio,
            "rpe": rpe_rmse,
            "rpe_z": rpe_rmse_z,
            "rpe_yaw": relative_yaw_rmse,
        }
    }

    """ ------------ Network loss metrics ----------- """
    mse_loss = np.mean(
        (net_attr_dict["targets"] - net_attr_dict["preds"]) ** 2, axis=0
    )  # 3x1
    likelihood_loss = np.mean(net_attr_dict["losses"], axis=0)  # 3x1
    avg_mse_loss = np.mean(mse_loss)
    avg_likelihood_loss = np.mean(likelihood_loss)
    metrics["ronin"]["mse_loss_x"] = float(mse_loss[0])
    metrics["ronin"]["mse_loss_y"] = float(mse_loss[1])
    metrics["ronin"]["mse_loss_z"] = float(mse_loss[2])
    metrics["ronin"]["mse_loss_avg"] = float(avg_mse_loss)
    metrics["ronin"]["likelihood_loss_x"] = float(likelihood_loss[0])
    metrics["ronin"]["likelihood_loss_y"] = float(likelihood_loss[1])
    metrics["ronin"]["likelihood_loss_z"] = float(likelihood_loss[2])
    metrics["ronin"]["likelihood_loss_avg"] = float(avg_likelihood_loss)

    """ ------------ Data for plotting ----------- """
    total_pred = net_attr_dict["preds"].shape[0]
    pred_ts = (1.0 / args.sample_freq) * np.arange(total_pred)
    pred_sigmas = np.exp(net_attr_dict["preds_cov"])
    plot_dict = {
        "ts": ts,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "pred_ts": pred_ts,
        "preds": net_attr_dict["preds"],
        "targets": net_attr_dict["targets"],
        "pred_sigmas": pred_sigmas,
        "rmse": rmse,
        "rpe_rmse": rpe_rmse,
        "rpes": rpes,
    }

    return metrics, plot_dict


def plot_3d_2var(x, y1, y2, xlb, ylbs, lgs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1[:, i], label=lgs[0])
        plt.plot(x, y2[:, i], label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def plot_3d_1var(x, y, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        if x is not None:
            plt.plot(x, y[:, i])
        else:
            plt.plot(y[:, i])
        plt.ylabel(ylbs[i])
        plt.grid(True)
    if xlb is not None:
        plt.xlabel(xlb)
    return fig


def plot_3d_2var_with_sigma(
    x, y1, y2, sig, xlb, ylbs, lgs, num=None, dpi=None, figsize=None
):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    y1_plus_sig = y1 + 3 * sig
    y1_minus_sig = y1 - 3 * sig
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1_plus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(x, y1_minus_sig[:, i], "-g", linewidth=0.2)
        plt.fill_between(
            x, y1_plus_sig[:, i], y1_minus_sig[:, i], facecolor="green", alpha=0.5
        )
        plt.plot(x, y1[:, i], "-b", linewidth=0.5, label=lgs[0])
        plt.plot(x, y2[:, i], "-r", linewidth=0.5, label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def plot_3d_1var_with_sigma(x, y, sig, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    plus_sig = 3 * sig
    minus_sig = -3 * sig
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, plus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(x, minus_sig[:, i], "-g", linewidth=0.2)
        plt.fill_between(
            x, plus_sig[:, i], minus_sig[:, i], facecolor="green", alpha=0.5
        )
        plt.plot(x, y[:, i], "-b", linewidth=0.5)
        plt.ylabel(ylbs[i])
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def make_plots(args, plot_dict, outdir):
    ts = plot_dict["ts"]
    pos_pred = plot_dict["pos_pred"]
    pos_gt = plot_dict["pos_gt"]
    pred_ts = plot_dict["pred_ts"]
    preds = plot_dict["preds"]
    targets = plot_dict["targets"]
    pred_sigmas = plot_dict["pred_sigmas"]
    rmse = plot_dict["rmse"]
    rpe_rmse = plot_dict["rpe_rmse"]
    rpes = plot_dict["rpes"]

    dpi = 90
    figsize = (16, 9)

    fig1 = plt.figure(num="prediction vs gt", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1])
    plt.plot(pos_gt[:, 0], pos_gt[:, 1])
    plt.axis("equal")
    plt.legend(["Predicted", "Ground truth"])
    plt.title("2D trajectory and ATE error against time")
    plt.subplot2grid((3, 2), (2, 0))
    plt.plot(np.linalg.norm(pos_pred - pos_gt, axis=1))
    plt.legend(["RMSE:{:.3f}, RPE:{:.3f}".format(rmse, rpe_rmse)])
    for i in range(3):
        plt.subplot2grid((3, 2), (i, 1))
        plt.plot(preds[:, i])
        plt.plot(targets[:, i])
        plt.legend(["Predicted", "Ground truth"])
        plt.title("{}".format(targ_names[i]))
    plt.tight_layout()
    plt.grid(True)

    fig2 = plot_3d_2var(
        ts,
        pos_pred,
        pos_gt,
        xlb="t(s)",
        ylbs=["x(m)", "y(m)", "z(m)"],
        lgs=["RONIN", "Ground Truth"],
        num="Position",
        dpi=dpi,
        figsize=figsize,
    )
    fig3 = plot_3d_2var_with_sigma(
        pred_ts,
        preds,
        targets,
        pred_sigmas,
        xlb="t(s)",
        ylbs=["x(m)", "y(m)", "z(m)"],
        lgs=["imu", "vio"],
        num="Displacement",
        dpi=dpi,
        figsize=figsize,
    )
    fig4 = plot_3d_1var_with_sigma(
        pred_ts,
        preds - targets,
        pred_sigmas,
        xlb="t(s)",
        ylbs=["x(m)", "y(m)", "z(m)"],
        num="Displacement errors",
        dpi=dpi,
        figsize=figsize,
    )
    fig5 = plot_3d_1var(
        None,
        rpes,
        xlb=None,
        ylbs=["x(m)", "y(m)", "z(m)"],
        num=f"RTE error over {args.rpe_window}s",
        dpi=dpi,
        figsize=figsize,
    )

    pred_norm = np.linalg.norm(preds[:, 0:2], axis=1)
    targ_norm = np.linalg.norm(targets[:, 0:2], axis=1)
    pred_ang = np.arctan2(preds[:, 0], preds[:, 1])
    targ_ang = np.arctan2(targets[:, 0], targets[:, 1])
    ang_diff = targ_ang - pred_ang
    ang_diff = ang_diff + 2 * np.pi * (ang_diff <= -np.pi)
    ang_diff = ang_diff - 2 * np.pi * (ang_diff > np.pi)

    fig6 = plt.figure(num="2D Displacement norm and heading", dpi=dpi, figsize=(16, 9))
    plt.title("2D Displacement norm and heading")
    plt.subplot(411)
    plt.plot(pred_ts, pred_norm, "-b", linewidth=0.5, label="imu")
    plt.plot(pred_ts, targ_norm, "-r", linewidth=0.5, label="vio")
    plt.ylabel("distance (m)")
    plt.legend()
    plt.grid(True)
    plt.subplot(412)
    plt.plot(pred_ts, pred_norm - targ_norm, "-b", linewidth=0.5)
    plt.ylabel("distance (m)")
    plt.grid(True)
    plt.subplot(413)
    plt.plot(pred_ts, pred_ang, "-b", linewidth=0.5)
    plt.plot(pred_ts, targ_ang, "-r", linewidth=0.5)
    plt.ylabel("angle (rad)")
    plt.grid(True)
    plt.subplot(414)
    plt.plot(pred_ts, ang_diff, "-b", linewidth=0.5)
    plt.ylabel("angle (rad)")
    plt.xlabel("t")
    plt.grid(True)

    fig1.savefig(osp.join(outdir, "view.png"))
    fig2.savefig(osp.join(outdir, "pos.png"))
    fig3.savefig(osp.join(outdir, "pred.svg"))
    fig4.savefig(osp.join(outdir, "pred-err.svg"))
    fig5.savefig(osp.join(outdir, "rpe.svg"))
    fig6.savefig(osp.join(outdir, "norm_angle.svg"))

    plt.close("all")

    return


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(network, data_loader, device, epoch):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    network.eval()

    for bid, (feat, targ, _, _) in enumerate(data_loader):
        pred, pred_cov = network(feat.to(device))
        targ = targ.to(device)
        loss = get_loss(pred, pred_cov, targ, epoch)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        losses_all.append(torch_to_numpy(loss))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
    }
    return attr_dict


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dict(
        [
            ("past_data_size", int(args.past_time * args.imu_freq)),
            ("window_size", int(args.window_time * args.imu_freq)),
            ("future_data_size", int(args.future_time * args.imu_freq)),
            ("step_size", int(args.imu_freq / args.sample_freq)),
        ]
    )
    net_config = {
        "in_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 32
        + 1
    }

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Perturb on bias: %s" % args.do_bias_shift)
    logging.info("Perturb on gravity: %s" % args.perturb_gravity)
    logging.info("Sample frequency: %s" % args.sample_freq)
    return data_window_config, net_config


def net_test(args):
    """
    Main function for network testing
    Generate trajectories, plots, and metrics.json file
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.test_list is None:
            raise ValueError("test_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            logging.info(f"Testing output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    test_list = get_datalist(args.test_list)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {args.model_path} loaded to device {device}.")

    # initialize containers
    all_metrics = {}

    for data in test_list:
        logging.info(f"Processing {data}...")
        try:
            seq_dataset = FbSequenceDataset(
                args.root_dir, [data], args, data_window_config, mode="test"
            )
            seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
        except OSError as e:
            print(e)
            continue

        # Obtain trajectory
        net_attr_dict = get_inference(network, seq_loader, device, epoch=50)
        traj_attr_dict = pose_integrate(args, seq_dataset, net_attr_dict["preds"])
        outdir = osp.join(args.out_dir, data)
        if osp.exists(outdir) is False:
            os.mkdir(outdir)
        outfile = osp.join(outdir, "trajectory.txt")
        trajectory_data = np.concatenate(
            [
                traj_attr_dict["ts"].reshape(-1, 1),
                traj_attr_dict["pos_pred"],
                traj_attr_dict["pos_gt"],
            ],
            axis=1,
        )
        np.savetxt(outfile, trajectory_data, delimiter=",")

        # obtain metrics
        metrics, plot_dict = compute_metrics_and_plotting(
            args, net_attr_dict, traj_attr_dict
        )
        logging.info(metrics)
        all_metrics[data] = metrics

        outfile_net = osp.join(outdir, "net_outputs.txt")
        net_outputs_data = np.concatenate(
            [
                plot_dict["pred_ts"].reshape(-1, 1),
                plot_dict["preds"],
                plot_dict["targets"],
                plot_dict["pred_sigmas"],
            ],
            axis=1,
        )
        np.savetxt(outfile_net, net_outputs_data, delimiter=",")

        if args.save_plot:
            make_plots(args, plot_dict, outdir)

        try:
            with open(args.out_dir + "/metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=1)
        except ValueError as e:
            raise e
        except OSError as e:
            print(e)
            continue
        except Exception as e:
            raise e

    return
