#!/usr/bin/env python3

"""
usage: put under source folder, required files: evolving_state.txt, calib_state.txt, state.txt
After first run, integration_states.txt, vio_states.txt are generated and figures are saved in current dir
You can move the figures and state.txt, integration_states.txt, vio_states.txt into a folder
Rerun to generate graphs more efficiently by specifying the folder names that has the above three files
"""


import json
import os
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from utils.logging import logging
from utils.math_utils import *


color_vio = "C2"
color_ronin = "C3"
color_filter = "C0"


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def imu_integrate(gravity, last_R, last_v, last_p, accel, omega, dt):
    """
    Given compensated IMU data and corresponding dt, propagate the previous state in the world frame
    """
    dR = mat_exp(omega * dt)

    new_R = last_R.dot(dR)
    new_v = last_v + gravity * dt + last_R.dot(accel * dt)
    new_p = (
        last_p
        + last_v * dt
        + 0.5 * gravity * dt * dt
        + 0.5 * last_R.dot(accel * dt * dt)
    )

    return new_R, new_v, new_p


def load_aekf_rotation(attitude_filter_path):
    # load aekf rotation
    aekf_data = np.loadtxt(attitude_filter_path, delimiter=",", skiprows=3)
    # load atttitude filter rotation
    ts_aekf = aekf_data[:, 0] * 1e-6
    aekf_r = Rotation.from_quat(
        np.concatenate(
            [aekf_data[:, 2:5], np.expand_dims(aekf_data[:, 1], axis=1)], axis=1
        )
    )
    R_aekf = aekf_r.as_matrix()
    # aekf results with y pointing down, convert them
    R_wf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    R_aekf = Rotation.from_matrix(np.matmul(R_wf, R_aekf))

    return ts_aekf, R_aekf


def load_aekf_calibration(attitude_filter_path):
    with open(attitude_filter_path, "r") as f:
        line = f.readline()
        line = f.readline()
    init_calib = np.fromstring(line, sep=",")
    init_accelScaleInv = init_calib[1:10].reshape((3, 3))
    init_gyroScaleInv = init_calib[10:19].reshape((3, 3))
    init_gyroGSense = init_calib[19:28].reshape((3, 3))
    init_accelBias = init_calib[28:31].reshape((3, 1))
    init_gyroBias = init_calib[31:34].reshape((3, 1))
    return (
        init_gyroScaleInv,
        init_gyroBias,
        init_gyroGSense,
        init_accelScaleInv,
        init_accelBias,
    )


def plot_state_euclidean(
    name_plot, name_ref, components_name, ts, state, sigma=None, **kwargs
):
    assert state.shape[0] == ts.shape[0]
    assert state.shape[1] == len(components_name)
    assert sigma.shape == state.shape if sigma is not None else True

    fig = plt.figure(name_plot)
    for i, name in enumerate(components_name):
        plt.subplot(len(components_name), 1, i + 1, label=name_plot + " " + name)
        plt.plot(ts, state[:, i], label=name_ref, **kwargs)
        if sigma is not None:
            plt.plot(ts, state[:, i] + 3 * sigma[:, i], "-r", linewidth=0.5, **kwargs)
            plt.plot(ts, state[:, i] - 3 * sigma[:, i], "-r", linewidth=0.5, **kwargs)
        plt.ylabel(name)
        plt.grid(True)
        plt.title(name_plot)
    plt.xlabel("t (s)")
    plt.legend(loc="upper center")
    return fig


def plot_error_euclidean(
    name_plot, name_ref, components_name, ts, err, sigma, **kwargs
):
    assert err.shape[0] == ts.shape[0]
    assert err.shape[1] == len(components_name)
    assert sigma.shape == err.shape

    fig = plt.figure(name_plot)
    for i, name in enumerate(components_name):
        plt.subplot(len(components_name), 1, i + 1, label=name_plot + " " + name)
        plt.plot(ts, err[:, i], label=name_ref, **kwargs)
        plt.plot(ts, +3 * sigma[:, i], "-r", linewidth=0.5, **kwargs)
        plt.plot(ts, -3 * sigma[:, i], "-r", linewidth=0.5, **kwargs)
        plt.ylabel(name)
        plt.grid(True)
        plt.title(name_plot)
    plt.xlabel("t (s)")
    plt.legend(loc="upper center")
    return fig


# get RPE
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
    ## compute statistics over z separatly
    rpe_rmse = np.sqrt(np.mean(np.sum(rpes ** 2, axis=1)))
    rpe_rmse_z = np.sqrt(np.mean(rpes[:, 2] ** 2))
    relative_yaw_rmse = np.sqrt(np.mean(relative_yaw_errors ** 2))
    return rpe_rmse, rpe_rmse_z, relative_yaw_rmse


def run(args, dataset):
    plt.close("all")
    if args.dir is not None:
        results_folder = args.dir
    else:
        results_folder = os.path.join(args.log_dir, dataset)

    try:
        states = np.load(os.path.join(results_folder, args.log_filename + ".npy"))
        save_vio_states = True  # traj is done
    except:
        logging.warning(
            "Relying on .txt file because .npy was not found. Surely means filter did not finish"
        )
        states = np.loadtxt(
            os.path.join(results_folder, args.log_filename), delimiter=","
        )  # traj is still processing
        save_vio_states = False  # traj is done

    R_init = states[0, :9].reshape(-1, 3, 3)
    r_init = Rotation.from_matrix(R_init)
    Rs = states[:, :9].reshape(-1, 3, 3)
    rs = Rotation.from_matrix(Rs)
    euls = rs.as_euler("xyz", degrees=True)
    vs = states[:, 9:12]
    ps = states[:, 12:15]
    ps_dr = np.cumsum(
        states[:, 9:12] * np.diff(states[:, 27:28], prepend=states[0, 27], axis=0),
        axis=0,
    )
    ba = states[:, 15:18]
    bg = states[:, 18:21]
    accs = states[:, 21:24]  # offline calib compensated, scale+bias
    gyrs = states[:, 24:27]  # offline calib compensated, scale+bias
    ts = states[:, 27]
    sigma_r = np.sqrt(states[:, 28:31]) * 180.0 / np.pi
    sigma_v = np.sqrt(states[:, 31:34])
    sigma_p = np.sqrt(states[:, 34:37])
    sigma_bg = np.sqrt(states[:, 37:40])
    sigma_ba = np.sqrt(states[:, 40:43])
    innos = states[:, 43:46]
    meas = states[:, 46:49]
    pred = states[:, 49:52]
    meas_sigma = states[:, 52:55]
    inno_sigma = states[:, 55:58]
    nobs_sigma = states[:, 58 : 58 + 16]

    N = ts.shape[0]

    # get RoNIN concatenation results
    if args.ronin_dir is not None:
        try:
            ronin = np.loadtxt(
                osp.join(args.ronin_dir, dataset + ".txt"), delimiter=","
            )
            logging.info(
                f"Reading ronin data from {osp.join(args.ronin_dir, dataset)}.txt"
            )
        except:
            ronin = np.loadtxt(
                osp.join(args.ronin_dir, dataset, "trajectory.txt"), delimiter=","
            )
            logging.info(
                f"Reading ronin data from {osp.join(args.ronin_dir, dataset,  'trajectory.txt')}"
            )

        ronin_ts = ronin[:, 0]
        ronin_p = ronin[:, 1:4]
        if ronin_ts[0] > ts[0]:
            ronin_ts = np.insert(ronin_ts, 0, ts[0])
            ronin_p = np.concatenate([ronin_p[0].reshape(1, 3), ronin_p], axis=0)
        if ronin_ts[-1] < ts[-1]:
            ronin_ts = np.insert(ronin_ts, -1, ts[-1])
            ronin_p = np.concatenate([ronin_p, ronin_p[-1].reshape(1, 3)], axis=0)
        ronin_p = interp1d(ronin_ts, ronin_p, axis=0)(ts)

    # get vio states
    if not args.plot_sim:
        if os.path.exists(os.path.join(results_folder, "vio_states.npy")):
            vio_states = np.load(os.path.join(results_folder, "vio_states.npy"))
            vio_euls = vio_states[:, :3]
            vio_p = vio_states[:, 3:6]
            vio_v = vio_states[:, 6:9]
            ba = vio_states[:, 9:12]
            bg = vio_states[:, 12:15]
            vio_disp = vio_states[:, 15:18]
            ba_b = vio_states[:, 18:21]
            bg_b = vio_states[:, 21:24]
            accelScaleInv_flat = vio_states[:, 24:33]
            gyroScaleInv_flat = vio_states[:, 33:42]
            accelScaleInv = accelScaleInv_flat.reshape((-1, 3, 3))
            gyroScaleInv = gyroScaleInv_flat.reshape((-1, 3, 3))
        else:
            data = np.load(osp.join(args.root_dir, dataset, "imu0_resampled.npy"))
            vio_ts = data[:, 0] * 1e-6
            vio_rq = data[:,-10:-6]
            vio_p = data[:,-6:-3]
            vio_v = data[:,-3:]
            vio_r = Rotation.from_quat(vio_rq)
            vio_euls = vio_r.as_euler("xyz", degrees=True)
            
            vio_calib_ts = vio_ts

            with open(osp.join(args.root_dir, dataset, "calibration.json"), 'r') as f:
                calib_json = json.load(f)
                
            accelBias = np.array(calib_json["Accelerometer"]["Bias"]["Offset"])[None,:]
            gyroBias = np.array(calib_json["Gyroscope"]["Bias"]["Offset"])[None,:]
            accelScaleInv = np.array(
                calib_json["Accelerometer"]["Model"]["RectificationMatrix"]
            )[None,:,:]
            gyroScaleInv = np.array(
                calib_json["Gyroscope"]["Model"]["RectificationMatrix"]
            )[None,:,:]

            vio_pj_idx = np.searchsorted(vio_ts, ts) - 1
            vio_pi_idx = np.searchsorted(vio_ts, ts - args.displacement_time) - 1
            vio_pj = vio_p[vio_pj_idx, :]
            vio_pi = vio_p[vio_pi_idx, :]
            vio_disp = vio_pj - vio_pi

            vio_Ri = vio_r[vio_pi_idx].as_matrix()
            ri_z = Rotation.from_matrix(vio_Ri).as_euler("xyz")[:, 2]
            vio_Riz = Rotation.from_euler("z", ri_z).as_matrix()
            vio_Rizt = np.transpose(vio_Riz, (0, 2, 1))
            vio_disp = np.squeeze(
                np.matmul(vio_Rizt, np.expand_dims(vio_disp, axis=-1))
            )

            vio_uw_euls = unwrap_rpy(vio_euls)
            if vio_ts[0] > ts[0]:
                vio_ts = np.insert(vio_ts, 0, ts[0])
                vio_uw_euls = np.concatenate(
                    [vio_uw_euls[0].reshape(1, 3), vio_uw_euls], axis=0
                )
                vio_p = np.concatenate([vio_p[0].reshape(1, 3), vio_p], axis=0)
                vio_v = np.concatenate([vio_v[0].reshape(1, 3), vio_v], axis=0)
            if vio_ts[-1] < ts[-1]:
                vio_ts = np.insert(vio_ts, -1, ts[-1])
                vio_uw_euls = np.concatenate(
                    [vio_uw_euls, vio_uw_euls[-1].reshape(1, 3)], axis=0
                )
                vio_p = np.concatenate([vio_p, vio_p[-1].reshape(1, 3)], axis=0)
                vio_v = np.concatenate([vio_v, vio_v[-1].reshape(1, 3)], axis=0)
            vio_uw_euls_interp = interp1d(vio_ts, vio_uw_euls, axis=0)(ts)
            vio_euls = wrap_rpy(vio_uw_euls_interp)
            vio_p = interp1d(vio_ts, vio_p, axis=0)(ts)
            vio_v = interp1d(vio_ts, vio_v, axis=0)(ts)

            accelScaleInv_flat = accelScaleInv.reshape((1, 9))
            gyroScaleInv_flat = gyroScaleInv.reshape((1, 9))

            # We only have offline calib so just repeat it for all the timestamps
            ba = np.tile(accelBias, [ts.shape[0], 1])
            bg = np.tile(gyroBias, [ts.shape[0], 1])
            accelScaleInv_flat = np.tile(accelScaleInv_flat, [ts.shape[0], 1])
            gyroScaleInv_flat = np.tile(gyroScaleInv_flat, [ts.shape[0], 1])
            accelScaleInv = accelScaleInv_flat.reshape((-1, 3, 3))
            gyroScaleInv = gyroScaleInv_flat.reshape((-1, 3, 3))

            # This compute bias in non scaled sensor frame (I think)
            ba_temp = np.expand_dims(ba, axis=-1)
            bg_temp = np.expand_dims(bg, axis=-1)
            ba_b = np.squeeze(
                np.matmul(np.linalg.inv(accelScaleInv), ba_temp)
            )
            bg_b = np.squeeze(
                np.matmul(np.linalg.inv(gyroScaleInv), bg_temp)
            )

            if save_vio_states:
                vio_states = np.concatenate(
                    [
                        vio_euls,
                        vio_p,
                        vio_v,
                        ba,
                        bg,
                        vio_disp,
                        ba_b,
                        bg_b,
                        accelScaleInv_flat,
                        gyroScaleInv_flat,
                    ],
                    axis=1,
                )
                np.save(os.path.join(results_folder, "vio_states.npy"), vio_states)
            else:
                logging.warning(
                    "Not saving vio_states.npy because traj is still processing"
                )

    # get simulation states
    if args.plot_sim:
        sim_data = np.loadtxt(
            args.sim_data_path,
            delimiter=",",
            usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        )
        sim_ts = sim_data[:, 0]
        sim_p = sim_data[:, 1:4]
        sim_rq = sim_data[:, 4:8]
        sim_v = sim_data[:, 11:14]
        # acc_all = sim_data[:,8:11]
        # gyr_all = sim_data[:,14:17]
        sim_r = Rotation.from_quat(sim_rq)
        sim_euls = sim_r.as_euler("xyz", degrees=True)

        sim_pj_idx = np.searchsorted(sim_ts, ts) - 1
        sim_pi_idx = np.searchsorted(sim_ts, ts - args.displacement_time) - 1
        sim_pj = sim_p[sim_pj_idx, :]
        sim_pi = sim_p[sim_pi_idx, :]
        sim_disp = sim_pj - sim_pi

        ri_z = sim_r[sim_pi_idx].as_matrix().as_euler("xyz")[:, 2]
        sim_Riz = Rotation.from_euler("z", ri_z).as_matrix()
        sim_Rizt = np.transpose(sim_Riz, (0, 2, 1))
        sim_disp = np.squeeze(np.matmul(sim_Rizt, np.expand_dims(sim_disp, axis=-1)))

        sim_uw_euls = unwrap_rpy(sim_euls)
        if sim_ts[0] > ts[0]:
            sim_ts = np.insert(sim_ts, 0, ts[0])
            sim_uw_euls = np.concatenate(
                [sim_uw_euls[0].reshape(1, 3), sim_uw_euls], axis=0
            )
            sim_p = np.concatenate([sim_p[0].reshape(1, 3), sim_p], axis=0)
            sim_v = np.concatenate([sim_v[0].reshape(1, 3), sim_v], axis=0)
        if sim_ts[-1] < ts[-1]:
            sim_ts = np.insert(sim_ts, -1, ts[-1])
            sim_uw_euls = np.concatenate(
                [sim_uw_euls, sim_uw_euls[-1].reshape(1, 3)], axis=0
            )
            sim_p = np.concatenate([sim_p, sim_p[-1].reshape(1, 3)], axis=0)
            sim_v = np.concatenate([sim_v, sim_v[-1].reshape(1, 3)], axis=0)
        sim_uw_euls_interp = interp1d(sim_ts, sim_uw_euls, axis=0)(ts)
        sim_euls = wrap_rpy(sim_uw_euls_interp)
        sim_p = interp1d(sim_ts, sim_p, axis=0)(ts)
        sim_v = interp1d(sim_ts, sim_v, axis=0)(ts)
        sim_ba = np.zeros((ts.shape[0], 3)) + np.array([0.3, -0.2, 0.4])
        sim_bg = np.zeros((ts.shape[0], 3)) + np.array([0.0005, 0.002, -0.001])

    if args.plot_sim:
        ref_type = "sim"
        ref_p = sim_p
        ref_v = sim_v
        ref_bg = sim_bg
        ref_ba = sim_ba
        ref_euls = sim_euls
        ref_disp = sim_disp
    else:
        ref_type = "vio"
        ref_p = vio_p
        ref_v = vio_v
        ref_bg = bg
        ref_ba = ba
        ref_euls = vio_euls
        ref_disp = vio_disp

    # obtain biases in the body frame in the same unit
#    attitude_filter_path = osp.join(args.root_dir, dataset, "atttitude.txt")
#    (
#        init_gyroScaleInv,
#        init_gyroBias,
#        init_gyroGSense,
#        init_accelScaleInv,
#        init_accelBias,
#    ) = load_aekf_calibration(attitude_filter_path)
#
#    if args.body_bias:
#        ba_temp = np.expand_dims(ba, axis=-1)
#        bg_temp = np.expand_dims(bg, axis=-1)
#        ba_b = np.squeeze(np.matmul(np.linalg.inv(init_accelScaleInv), ba_temp))
#        bg_b = np.squeeze(np.matmul(np.linalg.inv(init_gyroScaleInv), bg_temp))
#        ba = ba_b
#        bg = bg_b

    # load aekf rotation
    #aekf_ts, aekf_R = load_aekf_rotation(attitude_filter_path)
    #aekf_euls = unwrap_rpy(aekf_R.as_euler("xyz", degrees=True))
    # Hack since we don't have aekf data
    aekf_ts = ts
    aekf_euls = vio_euls

    # plotting
    N = ts.shape[0]
    start_idx = 2000  # 2 s
    end_idx = N - 1
    start_ts = ts[start_idx]
    end_ts = ts[end_idx]

    # align diverse trajectory sources
    ts = ts - start_ts
    aekf_ts = aekf_ts - start_ts

    try:
        # align at start_ts
        aekf_euls[:, 2] -= (
            interp1d(aekf_ts, aekf_euls[:, 2])(ts[start_idx]) - ref_euls[start_idx, 2]
        )
    except:
        # if we can't align at first aekf timestamp
        aekf_euls[:, 2] -= aekf_euls[0, 2] - interp1d(ts, ref_euls[:, 2])(aekf_ts[0])

    aekf_euls_time_aligned = interp1d(aekf_ts, aekf_euls, axis=0, fill_value=np.nan)(ts)
    aekf_euls_time_aligned = wrap_rpy(aekf_euls_time_aligned)

    if args.ronin_dir is not None:
        ronin_p = ronin_p - (ronin_p[start_idx, :] - ref_p[start_idx, :])

    ps = ps - (ps[start_idx, :] - ref_p[start_idx, :])
    euls[:, 2] = euls[:, 2] - (euls[start_idx, 2] - ref_euls[start_idx, 2])
    euls = wrap_rpy(euls)

    ps_gt = ref_p[start_idx:end_idx, :]
    euls_gt = ref_euls[start_idx:end_idx, :]
    euls_aekf = aekf_euls_time_aligned[start_idx:end_idx, :]

    # metrics computation
    metric_map = {"filter": {}, "ronin": {}}

    # align on first error, WHY IS THAT?
    ps_filter = ps[start_idx:end_idx, :]
    euls_filter = euls[start_idx:end_idx, :]
    ps_filter = ps_filter - (ps_filter[0, :] - ps_gt[0, :])

    # get drift and ATE
    ps_diff = ps_gt[1:, :] - ps_gt[:-1, :]
    traj_length = np.sum(np.linalg.norm(ps_diff, axis=1))
    drift_filter = np.linalg.norm(ps_filter[-1, :] - ps_gt[-1, :])
    angular_drift_filter = np.linalg.norm(euls[-1, 2] - ref_euls[-1, 2])

    filter_heading_error = wrap_rpy(euls - ref_euls)[:, 2]
    ate_filter = np.sqrt(np.mean(np.linalg.norm(ps_filter - ps_gt, axis=1) ** 2))
    metric_map["filter"]["drift_ratio"] = drift_filter / traj_length
    metric_map["filter"]["ate"] = ate_filter
    metric_map["filter"]["mhe"] = np.sqrt(
        np.nansum(filter_heading_error ** 2)
        / np.count_nonzero(~(np.isnan(filter_heading_error)))
    )
    metric_map["filter"]["angular_drift_deg_hour"] = (
        angular_drift_filter / ts.max() * 3600
    )
    logging.info(f"drift of filter {metric_map['filter']['drift_ratio']}")
    logging.info(f"ATE of filter {metric_map['filter']['ate']}")
    logging.info(f"Mean Heading error of filter {metric_map['filter']['mhe']}")

    def compute_rpe_filter(ns_rpe):
        rpe_rmse, rpe_rmse_z, relative_yaw_rmse = compute_rpe(
            ns_rpe, ps_filter, ps_gt, euls_filter[:, [2]], euls_gt[:, [2]]
        )
        logging.info(f"RPE RMSE of filter over {ns_rpe}s: {rpe_rmse}")
        logging.info(f"RPE RMSE Z of filter over {ns_rpe}s: {rpe_rmse_z}")
        logging.info(f"RPE RMSE Yaw of filter over {ns_rpe}s: {relative_yaw_rmse}")
        metric_map["filter"]["rpe_rmse_" + str(ns_rpe)] = rpe_rmse
        metric_map["filter"]["rpe_rmse_z_" + str(ns_rpe)] = rpe_rmse_z
        metric_map["filter"]["relative_yaw_rmse_" + str(ns_rpe)] = relative_yaw_rmse

    if args.rpe_1 is True:
        compute_rpe_filter(1000)  # 1 s
    if args.rpe_10 is True:
        compute_rpe_filter(10000)  # 10 s
    if args.rpe_100 is True:
        compute_rpe_filter(100000)  # 100 s

    if args.ronin_dir is not None:
        ps_ronin = ronin_p[start_idx:end_idx, :]
        ps_ronin = ps_ronin - (ps_ronin[0, :] - ps_gt[0, :])
        drift_ronin = np.linalg.norm(ps_ronin[-1, :] - ps_gt[-1, :])
        ate_ronin = np.sqrt(np.mean(np.linalg.norm(ps_ronin - ps_gt, axis=1) ** 2))
        angular_drift_ronin = np.linalg.norm(
            aekf_euls_time_aligned[-1, 2] - ref_euls[-1, 2]
        )
        heading_error_ronin = wrap_rpy(aekf_euls_time_aligned - ref_euls)[:, 2]
        metric_map["ronin"]["drift_ratio"] = drift_ronin / traj_length
        metric_map["ronin"]["ate"] = ate_ronin

        metric_map["ronin"]["mhe"] = np.sqrt(
            np.nansum(heading_error_ronin ** 2)
            / np.count_nonzero(~(np.isnan(heading_error_ronin)))
        )
        metric_map["ronin"]["angular_drift_deg_hour"] = (
            angular_drift_ronin / ts.max() * 3600
        )
        logging.info(f"drift of ronin {metric_map['ronin']['drift_ratio']}")
        logging.info(f"ATE of ronin {metric_map['ronin']['ate']}")
        logging.info(f"Mean Heading error of ronin {metric_map['ronin']['mhe']}")

        def compute_rpe_ronin(ns_rpe):
            rpe_rmse, rpe_rmse_z, relative_yaw_rmse = compute_rpe(
                ns_rpe, ps_ronin, ps_gt, euls_aekf[:, [2]], euls_gt[:, [2]]
            )
            logging.info(f"RPE RMSE of ronin over 1s: {rpe_rmse}")
            logging.info(f"RPE RMSE Z of ronin over 1s: {rpe_rmse_z}")
            logging.info(f"RPE RMSE Yaw of ronin over 1s: {relative_yaw_rmse}")
            metric_map["ronin"]["rpe_rmse_" + str(ns_rpe)] = rpe_rmse
            metric_map["ronin"]["rpe_rmse_z_" + str(ns_rpe)] = rpe_rmse_z
            metric_map["ronin"]["relative_yaw_rmse_" + str(ns_rpe)] = relative_yaw_rmse

        if args.rpe_1 is True:
            compute_rpe_ronin(1000)
        if args.rpe_10 is True:
            compute_rpe_ronin(10000)
        if args.rpe_100 is True:
            compute_rpe_ronin(100000)

    if args.make_plots is False:
        return metric_map

    # Plot results
    idxs = slice(start_idx, end_idx)
    
#    if not args.plot_sim:
#        plt.figure("Calibration accelerometer vio vs init")
#        plt.plot(ts, vio_accelScaleInv[:, 0, 0], label="1")
#        plt.plot(ts, vio_accelScaleInv[:, 1, 1], label="2")
#        plt.plot(ts, vio_accelScaleInv[:, 2, 2], label="3")
#        plt.plot(ts, vio_accelScaleInv[:, 0, 1], label="12")
#        plt.plot(ts, vio_accelScaleInv[:, 0, 2], label="13")
#        plt.plot(ts, vio_accelScaleInv[:, 1, 2], label="23")
#        plt.plot(
#            ts,
#            np.ones_like(vio_accelScaleInv[:, 0, 0]) * init_accelScaleInv[0, 0],
#            label="1",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_accelScaleInv[:, 1, 1]) * init_accelScaleInv[1, 1],
#            label="2",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_accelScaleInv[:, 2, 2]) * init_accelScaleInv[2, 2],
#            label="2",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_accelScaleInv[:, 0, 1]) * init_accelScaleInv[0, 1],
#            label="12",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_accelScaleInv[:, 0, 2]) * init_accelScaleInv[0, 2],
#            label="13",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_accelScaleInv[:, 1, 2]) * init_accelScaleInv[1, 2],
#            label="23",
#        )
#        plt.grid(True)
#        plt.legend(loc="upper center")
#        plt.figure()
#        g = np.einsum("kip,p->ki", vio_accelScaleInv, [0, 9.81, 0])
#        plt.plot(ts, g[:, 1])
#        plt.figure("Calibration gyrometer vio vs init")
#        plt.plot(ts, vio_gyroScaleInv[:, 0, 0], label="1")
#        plt.plot(ts, vio_gyroScaleInv[:, 1, 1], label="2")
#        plt.plot(ts, vio_gyroScaleInv[:, 2, 2], label="3")
#        plt.plot(ts, vio_gyroScaleInv[:, 0, 1], label="12")
#        plt.plot(ts, vio_gyroScaleInv[:, 0, 2], label="13")
#        plt.plot(ts, vio_gyroScaleInv[:, 1, 2], label="23")
#        plt.plot(
#            ts,
#            np.ones_like(vio_gyroScaleInv[:, 0, 0]) * init_gyroScaleInv[0, 0],
#            label="1",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_gyroScaleInv[:, 1, 1]) * init_gyroScaleInv[1, 1],
#            label="2",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_gyroScaleInv[:, 2, 2]) * init_gyroScaleInv[2, 2],
#            label="2",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_gyroScaleInv[:, 0, 1]) * init_gyroScaleInv[0, 1],
#            label="12",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_gyroScaleInv[:, 0, 2]) * init_gyroScaleInv[0, 2],
#            label="13",
#        )
#        plt.plot(
#            ts,
#            np.ones_like(vio_gyroScaleInv[:, 1, 2]) * init_gyroScaleInv[1, 2],
#            label="23",
#        )
#        plt.grid(True)
#        plt.legend(loc="upper center")

    fig14 = plt.figure("position-2d")
    plt.plot(ps[idxs, 0], ps[idxs, 1], label="filter", color=color_filter)
    # plt.plot(ps_dr[idxs, 0], ps_dr[idxs, 1], label="filter_dr")
    plt.plot(ref_p[idxs, 0], ref_p[idxs, 1], label=ref_type, color=color_vio)
    if args.ronin_dir is not None:
        plt.plot(ronin_p[idxs, 0], ronin_p[idxs, 1], label="ronin", color=color_ronin)
    plt.legend(loc="upper center")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.title("position-2d")
    plt.gca().set_aspect("equal")

    fig15 = plt.figure("position-3d")
    ax = fig15.add_subplot(111, projection="3d")
    plt.plot(ps[idxs, 0], ps[idxs, 1], ps[idxs, 2], label="filter", color=color_filter)
    # plt.plot(ps_dr[idxs, 0], ps_dr[idxs, 1], ps_dr[idxs, 2], label="filter_dr")
    plt.plot(
        ref_p[idxs, 0], ref_p[idxs, 1], ref_p[idxs, 2], label="vio", color=color_vio
    )
    if args.ronin_dir is not None:
        plt.plot(
            ronin_p[idxs, 0],
            ronin_p[idxs, 1],
            ronin_p[idxs, 2],
            label="ronin",
            color=color_ronin,
        )
    set_axes_equal(ax)

    fig1 = plot_state_euclidean(
        "position",
        "filter",
        ["p_x", "p_y", "p_z"],
        ts[idxs],
        ps[idxs, :],
        sigma_p[idxs, :],
        color=color_filter,
    )
    fig2 = plot_state_euclidean(
        "velocity",
        "filter",
        ["v_x", "v_y", "v_z"],
        ts[idxs],
        vs[idxs, :],
        sigma_v[idxs, :],
        color=color_filter,
    )
    fig3 = plot_state_euclidean(
        "attitude",
        "filter",
        ["eul_x", "eul_y", "eul_z"],
        ts[idxs],
        euls[idxs, :],
        sigma_r[idxs, :],
        color=color_filter,
    )
    fig4 = plot_state_euclidean(
        "accel bias",
        "filter",
        ["ba_x", "ba_y", "ba_z"],
        ts[idxs],
        ba[idxs, :],
        sigma_ba[idxs, :],
        color=color_filter,
    )

    fig5 = plot_state_euclidean(
        "gyro bias",
        "filter",
        ["bg_x", "bg_y", "bg_z"],
        ts[idxs],
        bg[idxs, :],
        sigma_bg[idxs, :],
        color=color_filter,
    )
    # plot integrated speed output
    # plot_state_euclidean(
    #     "position",
    #     "filter_dr",
    #     ["p_x", "p_y", "p_z"],
    #     ts[idxs],
    #     ps_dr[idxs, :]
    # )
    # plot reference traj
    plot_state_euclidean(
        "position",
        ref_type,
        ["p_x", "p_y", "p_z"],
        ts[idxs],
        ref_p[idxs, :],
        color=color_vio,
    )
    plot_state_euclidean(
        "velocity",
        ref_type,
        ["v_x", "v_y", "v_z"],
        ts[idxs],
        ref_v[idxs, :],
        color=color_vio,
    )
    plot_state_euclidean(
        "attitude",
        ref_type,
        ["eul_x", "eul_y", "eul_z"],
        ts[idxs],
        ref_euls[idxs, :],
        color=color_vio,
    )
    plot_state_euclidean(
        "accel bias",
        ref_type,
        ["ba_x", "ba_y", "ba_z"],
        ts[idxs],
        ref_ba[idxs, :],
        color=color_vio,
    )
    plot_state_euclidean(
        "accel bias",
        ref_type + "b",
        ["ba_x", "ba_y", "ba_z"],
        ts[idxs],
        ba_b[idxs, :],
        color=color_vio,
    )
    plot_state_euclidean(
        "gyro bias",
        ref_type,
        ["bg_x", "bg_y", "bg_z"],
        ts[idxs],
        ref_bg[idxs, :],
        color=color_vio,
    )
    plot_state_euclidean(
        "gyro bias",
        ref_type + "b",
        ["bg_x", "bg_y", "bg_z"],
        ts[idxs],
        bg_b[idxs, :],
        color=color_vio,
    )
    # plot RONIN
    if args.ronin_dir is not None:
        plot_state_euclidean(
            "position",
            "ronin",
            ["p_x", "p_y", "p_z"],
            ts[idxs],
            ronin_p[idxs, :],
            color=color_ronin,
        )
        plot_state_euclidean(
            "attitude",
            "aekf",
            ["eul_x", "eul_y", "eul_z"],
            ts[idxs],
            aekf_euls_time_aligned[idxs, :],
            color=color_ronin,
        )

    # compute values at update rate
    idx_update = ~np.isnan(innos[idxs, :]).any(axis=1)
    ts_update = ts[idxs][idx_update]
    innos_update = innos[idxs, :][idx_update, :]
    innos_sigma_update = inno_sigma[idxs, :][idx_update, :]
    meas_update = meas[idxs, :][idx_update, :]
    meas_sigma_update = meas_sigma[idxs, :][idx_update, :]
    pred_update = pred[idxs, :][idx_update, :]
    ref_disp_update = ref_disp[idxs, :][idx_update, :]
    fig6 = plot_error_euclidean(
        "innovation (m)",
        "filter",
        ["x", "y", "z"],
        ts_update,
        innos_update,
        sigma=innos_sigma_update,
    )

    # plot measurement
    # compute norm for visualization
    meas_plus_nom = np.hstack(
        (meas_update, np.atleast_2d(np.linalg.norm(meas_update, axis=1)).T)
    )
    pred_plus_nom = np.hstack(
        (pred_update, np.atleast_2d(np.linalg.norm(pred_update, axis=1)).T)
    )
    ref_disp_plus_norm = np.hstack(
        (ref_disp_update, np.atleast_2d(np.linalg.norm(ref_disp_update, axis=1)).T)
    )
    meas_sigma_plus_norm = np.hstack(
        (meas_sigma_update, np.zeros((meas_update.shape[0], 1)) * np.nan)
    )

    fig7 = plot_state_euclidean(
        "displacement measurement vs target",
        "meas",
        ["x", "y", "z", "norm"],
        ts_update,
        meas_plus_nom,
        sigma=meas_sigma_plus_norm,
        color=color_ronin,
    )
    plot_state_euclidean(
        "displacement measurement vs target",
        "pred",
        ["x", "y", "z", "norm"],
        ts_update,
        pred_plus_nom,
        color=color_filter,
    )
    plot_state_euclidean(
        "displacement measurement vs target",
        ref_type,
        ["x", "y", "z", "norm"],
        ts_update,
        ref_disp_plus_norm,
        color=color_vio,
    )

    # Plot the error in covariance

    meas_err_update = meas_update - ref_disp_update
    fig8 = plot_error_euclidean(
        "displace measurement errors",
        "meas-err",
        ["x", "y", "z"],
        ts_update,
        meas_err_update,
        meas_sigma_update,
    )
    plot_error_euclidean(
        "displace measurement errors",
        "pred-err",
        ["x", "y", "z"],
        ts_update,
        pred_update - ref_disp_update,
        meas_sigma_update,
    )
    for ax in fig8.axes:
        ax.set_ylim([-0.5, 0.5])

    # display measurement error autocorellation (SLOW)
    fig8b = plt.figure("autocorellation of measurement error")
    logging.warning("We assume update frequency at 20hz for autocorrelation")
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.acorr(meas_err_update[:, i], maxlags=100, lw=2)
        locs, labels = plt.xticks()  # Get locations and labels
        for (l, t) in zip(locs, labels):
            t.set_text(str(l / 20.0) + "s")
        plt.xticks(locs, labels)  # Set locations and labels
        plt.xlim(left=0)
        plt.grid()

    fig9 = plot_error_euclidean(
        "position error",
        ref_type,
        ["p_x", "p_y", "p_z"],
        ts[idxs],
        ps[idxs, :] - ref_p[idxs, :],
        sigma_p[idxs, :],
    )
    fig10 = plot_error_euclidean(
        "velocity error",
        ref_type,
        ["v_x", "v_y", "v_z"],
        ts[idxs],
        vs[idxs, :] - ref_v[idxs, :],
        sigma_v[idxs, :],
    )

    euls_err = wrap_rpy(euls - ref_euls)
    fig11 = plot_error_euclidean(
        "attitude error",
        ref_type,
        ["eulx", "euly", "eulz"],
        ts[idxs],
        euls_err[idxs, :],
        sigma_r[idxs, :],
    )

    fig12 = plot_error_euclidean(
        "accel bias error",
        ref_type,
        ["ba_x", "ba_y", "ba_z"],
        ts[idxs],
        ba[idxs, :] - ref_ba[idxs, :],
        sigma_ba[idxs, :],
    )
    fig13 = plot_error_euclidean(
        "gyros bias error",
        ref_type,
        ["bg_x", "bg_y", "bg_z"],
        ts[idxs],
        bg[idxs, :] - ref_bg[idxs, :],
        sigma_bg[idxs, :],
    )

    if args.save_fig:
        if not save_vio_states:
            logging.warning("Not saving figure because, the filter did not finish")
        else:
            logging.info("Saving figures")
            for save_as in ["png", "svg"]:
                fig1.savefig(osp.join(results_folder, "position." + save_as))
                fig2.savefig(osp.join(results_folder, "velocity." + save_as))
                fig3.savefig(osp.join(results_folder, "rotation." + save_as))
                fig4.savefig(osp.join(results_folder, "acc-bias." + save_as))
                fig5.savefig(osp.join(results_folder, "gyr-bias." + save_as))
                fig6.savefig(osp.join(results_folder, "innovation." + save_as))
                fig7.savefig(osp.join(results_folder, "displacement." + save_as))
                fig8.savefig(osp.join(results_folder, "displacement-err." + save_as))
                fig9.savefig(osp.join(results_folder, "position-err." + save_as))
                fig10.savefig(osp.join(results_folder, "velocity-err." + save_as))
                fig11.savefig(osp.join(results_folder, "rotation-err." + save_as))
                fig12.savefig(osp.join(results_folder, "acc-bias-err." + save_as))
                fig13.savefig(osp.join(results_folder, "gyr-bias-err." + save_as))
                fig14.savefig(osp.join(results_folder, "position-2d." + save_as))
                fig15.savefig(osp.join(results_folder, "position-3d." + save_as))
            # # also save as pickle just in case (TOO LARGE)
            # import pickle as pl
            # pl.dump(fig1, open(osp.join(results_folder, "position.pickle"),"wb"))
            # pl.dump(fig2, open(osp.join(results_folder, "velocity.pickle"),"wb"))
            # pl.dump(fig3, open(osp.join(results_folder, "rotation.pickle"),"wb"))
            # pl.dump(fig4, open(osp.join(results_folder, "acc-bias.pickle"),"wb"))
            # pl.dump(fig5, open(osp.join(results_folder, "gyr-bias.pickle"),"wb"))
            # pl.dump(fig6, open(osp.join(results_folder, "innovation.pickle"),"wb"))
            # pl.dump(fig7, open(osp.join(results_folder, "displacement.pickle"),"wb"))
            # pl.dump(fig8, open(osp.join(results_folder, "displacement-err.pickle"),"wb"))
            # pl.dump(fig9, open(osp.join(results_folder, "position-err.pickle"),"wb"))
            # pl.dump(fig10, open(osp.join(results_folder, "velocity-err.pickle"),"wb"))
            # pl.dump(fig11, open(osp.join(results_folder, "rotation-err.pickle"),"wb"))
            # pl.dump(fig12, open(osp.join(results_folder, "acc-bias-err.pickle"),"wb"))
            # pl.dump(fig13, open(osp.join(results_folder, "gyr-bias-err.pickle"),"wb"))
            # pl.dump(fig14, open(osp.join(results_folder, "position-2d.pickle"),"wb"))
            # pl.dump(fig15, open(osp.join(results_folder, "position-3d.pickle"),"wb"))

    else:
        logging.warning("Not saving figure")

    if args.display_fig:
        fig_state = [fig1, fig2, fig3, fig4, fig5, fig14, fig15]
        fig_innovation = [fig6, fig7, fig8]
        fig_err = [fig9, fig10, fig11, fig12, fig13]

        # [plt.close(f) for f in fig_state]
        # [plt.close(f) for f in fig_innovation]
        # [plt.close(f) for f in fig_err]
        import matplotlib

        fig_button = plt.figure("continue")
        axcontinue = plt.axes([0, 0.0, 0.5, 0.5])
        bcontinue = matplotlib.widgets.Button(axcontinue, "Continue")
        bcontinue.on_clicked(lambda d: plt.close("all"))
        ax3d = plt.axes([0.5, 0.5, 0.5, 0.5])
        b3d = matplotlib.widgets.Button(ax3d, "state")
        b3d.on_clicked(lambda d: [f.show() for f in fig_state])
        axerror = plt.axes([0, 0.5, 0.5, 0.5])
        berror = matplotlib.widgets.Button(axerror, "error")
        berror.on_clicked(lambda d: [f.show() for f in fig_err])
        axinnov = plt.axes([0.5, 0, 0.5, 0.5])
        binnov = matplotlib.widgets.Button(axinnov, "innov")
        binnov.on_clicked(lambda d: [f.show() for f in fig_innovation])
        fig15.show()

        plt.show()

    return metric_map


def run_on_each_dataset_and_gather_metrics(args, data_names):
    all_metrics = {}
    # retrieve metric from old, logs, they would be erased if necessary
    # I use the presence of .svg or .png as the flag for something necessary
    if osp.exists(args.log_dir + "/metrics.json"):
        with open(args.log_dir + "/metrics.json", "r") as f:
            logging.info(f"Loading old metric file at {args.log_dir + '/metrics.json'}")
            all_metrics = json.load(f)

    for dataset in progressbar.progressbar(data_names, redirect_stdout=True):
        logging.info(f"Plotting dataset {dataset}")
        try:
            results_folder = os.path.join(args.log_dir, dataset)
            if osp.exists(osp.join(results_folder, "position-3d.png")):
                logging.info(f"Skipping {dataset} because alraedy processed")
                continue
            metric_map = run(args, dataset)
            all_metrics[dataset] = metric_map
            with open(args.log_dir + "/metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=1)
        except ValueError as e:
            raise e
        except OSError as e:
            print(e)
            continue
        except Exception as e:
            raise e


def compare_biases(args):
    with open(args.data_list) as f:
        data_names = [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]
    # get headset name
    headset_names = set()
    for dataset in data_names:
        headset_name = dataset.split("_")[1].strip("hidacori")
        headset_names.add(headset_name)
    logging.info(f"Got {len(headset_names)} different hw")
    # assign color to each
    headset_names_colors = {}
    for i, hw in enumerate(headset_names):
        headset_names_colors[hw] = "C" + str(i)

    for dataset in data_names:
        headset_name = dataset.split("_")[1].strip("hidacori")
        results_folder = os.path.join(args.log_dir, dataset)
        states = np.load(os.path.join(results_folder, args.log_filename + ".npy"))
        ts = states[:, 27]
        ts = ts - ts[0]
        ba = states[:, 15:18]
        bg = states[:, 18:21]
        sigma_bg = np.sqrt(states[:, 37:40])
        sigma_ba = np.sqrt(states[:, 40:43])

        plot_state_euclidean(
            "accel bias",
            dataset,
            ["ba_x", "ba_y", "ba_z"],
            ts,
            ba,
            sigma_ba,
            color=headset_names_colors[headset_name],
        )
        plt.legend([])
        plot_state_euclidean(
            "gyro bias",
            dataset,
            ["bg_x", "bg_y", "bg_z"],
            ts,
            np.rad2deg(bg) * 3600,
            np.rad2deg(sigma_bg) * 3600,
            color=headset_names_colors[headset_name],
        )
        plt.legend([])
    plt.show()


if __name__ == "__main__":

    from utils.argparse_utils import add_bool_arg
    import argparse

    parser = argparse.ArgumentParser()

    # interface
    parser.add_argument("--dir", type=str, default=None)  # overriding directory
    parser.add_argument(
        "--log_dir", type=str, default="."
    )  # dir for loading filter outputs and saving figures
    parser.add_argument("--log_filename", type=str, default="not_vio_state.txt")

    # dataset
    parser.add_argument("--root_dir", type=str, default="../data")
    parser.add_argument("--dataset_number", type=int, default=None)

    # sim data
    add_bool_arg(parser, "plot_sim", default=False)

    parser.add_argument("--sim_data_path", type=str, default="imu-sim.txt")

    # more data
    parser.add_argument("--ronin_dir", type=str, default=None)

    # variables
    parser.add_argument("--displacement_time", type=float, default=1.0)
    add_bool_arg(
        parser, "body_bias", default=False
    )  # plotting bias in the body frame, always false in sim

    # make plots / or just gathering metrics
    add_bool_arg(parser, "make_plots", default=True)

    # display selections
    add_bool_arg(parser, "save_fig", default=True)
    add_bool_arg(parser, "display_fig", default=False)

    parser.add_argument("--rpe_1", type=bool, default=True)
    parser.add_argument("--rpe_10", type=bool, default=True)
    parser.add_argument("--rpe_100", type=bool, default=True)

    subparsers = parser.add_subparsers()
    parser_bias = subparsers.add_parser("bias")
    parser_bias.set_defaults(func=compare_biases)

    args = parser.parse_args()
    print(vars(args))

    args.data_list = osp.join(args.root_dir, "test_list.txt")

    try:
        args.func(args)
    except AttributeError:
        pass

    # default plot
    if args.plot_sim is False:
        with open(args.data_list) as f:
            data_names = [
                s.strip().split("," or " ")[0]
                for s in f.readlines()
                if len(s) > 0 and s[0] != "#"
            ]
        if args.dataset_number is not None:
            logging.info(f"Plotting dataset {data_names[args.dataset_number]}")
            run(args, data_names[args.dataset_number])
        else:
            run_on_each_dataset_and_gather_metrics(args, data_names)
    else:
        pass
