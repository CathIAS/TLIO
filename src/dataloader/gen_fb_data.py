#!/usr/bin/env python3

"""
gen_fb_data.py

Input (FB dataset raw): my_timestamps_p.txt, calib_state.txt, imu_measurements.txt, evolving_state.txt, attitude.txt
Output: data.hdf5
    - ts
    - raw accel and gyro measurements
    - vio-calibrated accel and gyro measurements
    - ground truth (vio) states (R, p, v)
    - integration rotation (with offline calibration)
    - attitude filter rotation
    - offline calibration parameters

Note: the dataset has been processed in a particular way such that the time difference between ts is almost always 1ms. This is for the training and testing the network. 
Image frequency is known, and IMU data is interpolated evenly between the two images.
"""

import os
from os import path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from numba import jit
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from utils.math_utils import mat_exp, unwrap_rpy, wrap_rpy


def imu_integrate(gravity, last_state, imu_data, dt):
    """
    Given compensated IMU data and corresponding dt, propagate the previous state in the world frame
    """
    last_r = Rotation.from_rotvec(last_state[0:3])
    last_p = last_state[3:6]
    last_v = last_state[6:9]
    accel = imu_data[:3]
    omega = imu_data[3:]

    last_R = last_r.as_matrix()
    dR = mat_exp(omega * dt)

    new_R = last_R.dot(dR)
    new_v = last_v + gravity * dt + last_R.dot(accel * dt)
    new_p = (
        last_p
        + last_v * dt
        + 0.5 * gravity * dt * dt
        + 0.5 * last_R.dot(accel * dt * dt)
    )
    new_r = Rotation.from_matrix(new_R)

    return np.concatenate((new_r.as_rotvec(), new_p, new_v), axis=0)


def save_hdf5(args):

    # get list of data to process
    f = open(args.data_list, "r")
    name = [line.rstrip() for line in f]
    f.close()
    n_data = len(name)
    print(f"total {n_data} datasets")

    gravity = np.array([0, 0, -args.gravity])

    for i in progressbar.progressbar(range(n_data), redirect_stdout=True):
        n = name[i]

        # set flag for which data has been processed
        datapath = osp.join(args.data_dir, n)
        flag_file = os.path.join(args.data_dir, n, "hey.txt")
        if os.path.exists(flag_file):
            print(f"{i}th data {n} processed, skip")
            continue
        else:
            print(f"processing data {i} - {n}")
            # f = open(flag_file, 'w'); f.write('hey'); f.close()

        # start with the 20th image processed, bypass vio initialization
        image_ts = np.loadtxt(osp.join(datapath, "my_timestamps_p.txt"))
        imu_meas = np.loadtxt(osp.join(datapath, "imu_measurements.txt"), delimiter=",")
        vio_states = np.loadtxt(osp.join(datapath, "evolving_state.txt"), delimiter=",")
        calib_data = np.loadtxt(osp.join(datapath, "calib_state.txt"), delimiter=",")
        attitudes = np.loadtxt(
            osp.join(datapath, "atttitude.txt"), delimiter=",", skiprows=3
        )

        # find initial state, start from the 21st output from vio_state
        start_t = image_ts[20]
        imu_idx = np.searchsorted(imu_meas[:, 0], start_t)
        vio_idx = np.searchsorted(vio_states[:, 0], start_t)

        # get imu_data - raw and calibrated
        print("obtain raw and vio-calibrated IMU data")
        imu_data = imu_meas[imu_idx:, :]
        ts = imu_data[:, 0] * 1e-6  # s
        accel_raw = imu_data[:, 1:4]  # raw
        gyro_raw = imu_data[:, 7:10]
        accel = imu_data[:, 4:7]  # calibrated with vio calibration
        gyro = imu_data[:, 10:13]

        # get state_data (same timestamps as imu_data), integrated with vio-calibrated IMU
        print("obtain vio states by integrating vio-calibrated IMU")
        N = imu_data.shape[0]
        state_data = np.zeros((N, 9))

        r_init = Rotation.from_quat(
            [
                vio_states[vio_idx, 2],
                vio_states[vio_idx, 3],
                vio_states[vio_idx, 4],
                vio_states[vio_idx, 1],
            ]
        )
        p_init = [
            vio_states[vio_idx, 5],
            vio_states[vio_idx, 6],
            vio_states[vio_idx, 7],
        ]
        v_init = [
            vio_states[vio_idx, 8],
            vio_states[vio_idx, 9],
            vio_states[vio_idx, 10],
        ]
        state_init = np.concatenate((r_init.as_rotvec(), p_init, v_init), axis=0)
        state_data[0, :] = state_init

        for i in progressbar.progressbar(range(1, N)):
            # get calibrated imu data for integration
            imu_data_i = np.concatenate((imu_data[i, 4:7], imu_data[i, 10:13]), axis=0)
            curr_t = imu_data[i, 0]
            past_t = imu_data[i - 1, 0]
            dt = (curr_t - past_t) * 1e-6  # s

            last_state = state_data[i - 1, :]
            new_state = imu_integrate(gravity, last_state, imu_data_i, dt)
            state_data[i, :] = new_state

            # if this state has vio output, correct with vio
            has_vio = imu_data[i, 13]
            if has_vio == 1:
                vio_idx = np.searchsorted(vio_states[:, 0], imu_data[i, 0])
                r_vio = Rotation.from_quat(
                    [
                        vio_states[vio_idx, 2],
                        vio_states[vio_idx, 3],
                        vio_states[vio_idx, 4],
                        vio_states[vio_idx, 1],
                    ]
                )
                p_vio = [
                    vio_states[vio_idx, 5],
                    vio_states[vio_idx, 6],
                    vio_states[vio_idx, 7],
                ]
                v_vio = [
                    vio_states[vio_idx, 8],
                    vio_states[vio_idx, 9],
                    vio_states[vio_idx, 10],
                ]
                vio_state = np.concatenate((r_vio.as_rotvec(), p_vio, v_vio), axis=0)
                state_data[i, :] = vio_state
        # adding timestamps in state_data
        state_data = np.concatenate(
            (np.expand_dims(imu_data[:, 0], axis=1), state_data), axis=1
        )

        # vio data
        vio_rvec = state_data[:, 1:4]
        vio_p = state_data[:, 4:7]
        vio_v = state_data[:, 7:10]
        vio_r = Rotation.from_rotvec(vio_rvec)
        vio_q = vio_r.as_quat()
        vio_q_wxyz = np.concatenate(
            [np.expand_dims(vio_q[:, 3], axis=1), vio_q[:, 0:3]], axis=1
        )

        # get offline and factory calibration
        print("obtain offline and factory calibration")
        with open(osp.join(datapath, "atttitude.txt"), "r") as f:
            line = f.readline()
            line = f.readline()
        init_calib = np.fromstring(line, sep=",")
        # init_calib_fac = None
        accelScaleInv = init_calib[1:10].reshape((3, 3))
        gyroScaleInv = init_calib[10:19].reshape((3, 3))
        gyroGSense = init_calib[19:28].reshape((3, 3))
        accelBias = init_calib[28:31].reshape((3, 1))
        gyroBias = init_calib[31:34].reshape((3, 1))
        offline_calib = np.concatenate(
            (
                accelScaleInv.flatten(),
                gyroScaleInv.flatten(),
                gyroGSense.flatten(),
                accelBias.flatten(),
                gyroBias.flatten(),
            )
        )

        # calibrate raw IMU with fixed calibration
        print("calibrate IMU with fixed calibration")
        accel_calib = (np.dot(accelScaleInv, accel_raw.T) - accelBias).T
        gyro_calib = (
            np.dot(gyroScaleInv, gyro_raw.T)
            - np.dot(gyroGSense, accel_raw.T)
            - gyroBias
        ).T

        # integrate using fixed calibration data
        print("integrate R using fixed calibrated data")
        N = ts.shape[0]
        rvec_integration = np.zeros((N, 3))
        rvec_integration[0, :] = state_data[0, 1:4]

        for i in progressbar.progressbar(range(1, N)):
            dt = ts[i] - ts[i - 1]
            last_rvec = Rotation.from_rotvec(rvec_integration[i - 1, :])
            last_R = last_rvec.as_matrix()
            omega = gyro_calib[i, :]
            dR = mat_exp(omega * dt)
            next_R = last_R.dot(dR)
            next_r = Rotation.from_matrix(next_R)
            next_rvec = next_r.as_rotvec()
            rvec_integration[i, :] = next_rvec
        integration_r = Rotation.from_rotvec(rvec_integration)
        integration_q = integration_r.as_quat()
        integration_q_wxyz = np.concatenate(
            [np.expand_dims(integration_q[:, 3], axis=1), integration_q[:, 0:3]], axis=1
        )

        # get attitude filter data
        print("obtain attitude filter rotation")
        ts_filter = attitudes[:, 0] * 1e-6  # s
        ts_start_index = 0
        ts_end_index = -1
        if ts_filter[0] > ts[0]:
            print("dataset " + n + " has smaller attitude filter timerange")
            ts_start_index = np.where(ts > ts_filter[0])[0][0]
        if ts_filter[-1] < ts[-1]:
            print("dataset " + n + " has smaller attitude filter timerange")
            ts_end_index = np.where(ts < ts_filter[-1])[0][-1]

        filter_r = Rotation.from_quat(
            np.concatenate(
                [attitudes[:, 2:5], np.expand_dims(attitudes[:, 1], axis=1)], axis=1
            )
        )
        R_filter = filter_r.as_matrix()
        R_wf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        R_filter = np.matmul(R_wf, R_filter)
        filter_r = Rotation.from_matrix(R_filter)  # corrected rotation for filter
        filter_rpy = filter_r.as_euler("xyz", degrees=True)
        uw_filter_rpy = unwrap_rpy(filter_rpy)
        uw_filter_rpy_interp = interp1d(ts_filter, uw_filter_rpy, axis=0)(
            ts[ts_start_index:ts_end_index]
        )
        filter_rpy_interp = wrap_rpy(uw_filter_rpy_interp)
        filter_r_interp = Rotation.from_euler("xyz", filter_rpy_interp, degrees=True)
        filter_q_interp = filter_r_interp.as_quat()
        filter_q_wxyz_interp = np.concatenate(
            [np.expand_dims(filter_q_interp[:, 3], axis=1), filter_q_interp[:, 0:3]],
            axis=1,
        )

        # truncate to attitude filter range
        ts = ts[ts_start_index:ts_end_index]
        accel = accel[ts_start_index:ts_end_index, :]
        gyro = gyro[ts_start_index:ts_end_index, :]
        accel_raw = accel_raw[ts_start_index:ts_end_index, :]
        gyro_raw = gyro_raw[ts_start_index:ts_end_index, :]
        vio_q_wxyz = vio_q_wxyz[ts_start_index:ts_end_index, :]
        vio_p = vio_p[ts_start_index:ts_end_index, :]
        vio_v = vio_v[ts_start_index:ts_end_index, :]
        integration_q_wxyz = integration_q_wxyz[ts_start_index:ts_end_index, :]
        if ts.shape[0] != filter_q_wxyz_interp.shape[0]:
            print("data and attitude filter time does not match!")

        # output
        outdir = osp.join(args.output_dir, n)
        if not osp.isdir(outdir):
            os.makedirs(outdir)

        # everything under the same timestamp ts
        with h5py.File(osp.join(outdir, "data.hdf5"), "w") as f:
            ts_s = f.create_dataset("ts", data=ts)
            accel_dcalibrated_s = f.create_dataset("accel_dcalibrated", data=accel)
            gyro_dcalibrated_s = f.create_dataset("gyro_dcalibrated", data=gyro)
            accel_raw_s = f.create_dataset("accel_raw", data=accel_raw)
            gyro_raw_s = f.create_dataset("gyro_raw", data=gyro_raw)
            vio_q_wxyz_s = f.create_dataset("vio_q_wxyz", data=vio_q_wxyz)
            vio_p_s = f.create_dataset("vio_p", data=vio_p)
            vio_v_s = f.create_dataset("vio_v", data=vio_v)
            integration_q_wxyz_s = f.create_dataset(
                "integration_q_wxyz", data=integration_q_wxyz
            )
            filter_q_wxyz_s = f.create_dataset(
                "filter_q_wxyz", data=filter_q_wxyz_interp
            )
            offline_calib_s = f.create_dataset("offline_calib", data=offline_calib)
            print("File data.hdf5 written to " + outdir)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--output_dir", type=str, default="../../dataset_test_output")
    parser.add_argument(
        "--data_dir", type=str, default="/raid0/docker-raid/wenxin/data/Dataset"
    )
    parser.add_argument(
        "--data_list",
        type=str,
        default="/raid0/docker-raid/wenxin/data/Dataset/golden_test_small.txt",
    )
    args = parser.parse_args()

    save_hdf5(args)
