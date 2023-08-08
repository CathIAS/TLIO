#!/usr/bin/env python3

import json
from typing import Optional

import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from tracker.imu_buffer import ImuBuffer
from tracker.imu_calib import ImuCalib
from tracker.meas_source_torchscript import MeasSourceTorchScript
from tracker.scekf import ImuMSCKF
from utils.dotdict import dotdict
from utils.from_scipy import compute_euler_from_matrix
from utils.logging import logging
from utils.math_utils import mat_exp


class ImuTracker:
    """
    ImuTracker is responsible for feeding the EKF with the correct data
    It receives the imu measurement, fills the buffer, runs the network with imu data in buffer
    and drives the filter.
    """

    def __init__(
        self,
        model_path,
        model_param_path,
        update_freq,
        filter_tuning_cfg,
        imu_calib: Optional[ImuCalib] = None,
        force_cpu=False,
    ):

        config_from_network = dotdict({})
        with open(model_param_path) as json_file:
            data_json = json.load(json_file)
            config_from_network["imu_freq_net"] = data_json["imu_freq"]
            config_from_network["past_time"] = data_json["past_time"]
            config_from_network["window_time"] = data_json["window_time"]
            config_from_network["arch"] = data_json["arch"]

        # frequencies and sizes conversion
        if not (
            config_from_network.past_time * config_from_network.imu_freq_net
        ).is_integer():
            raise ValueError(
                "past_time cannot be represented by integer number of IMU data."
            )
        if not (
            config_from_network.window_time * config_from_network.imu_freq_net
        ).is_integer():
            raise ValueError(
                "window_time cannot be represented by integer number of IMU data."
            )
        self.imu_freq_net = (
            config_from_network.imu_freq_net
        )  # imu frequency as input to the network
        self.past_data_size = int(
            config_from_network.past_time * config_from_network.imu_freq_net
        )
        self.disp_window_size = int(
            config_from_network.window_time * config_from_network.imu_freq_net
        )
        self.net_input_size = self.disp_window_size + self.past_data_size

        # EXAMPLE :
        # if using 200 samples with step size 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        # if using 400 samples with 200 past data and clone_every_n_netimu_sample 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        if not (config_from_network.imu_freq_net / update_freq).is_integer():
            raise ValueError("update_freq must be divisible by imu_freq_net.")
        if not (config_from_network.window_time * update_freq).is_integer():
            raise ValueError(
                "window_time cannot be represented by integer number of updates."
            )
        self.update_freq = update_freq
        self.clone_every_n_netimu_sample = int(
            config_from_network.imu_freq_net / update_freq
        )  # network inference/filter update interval
        assert (
            config_from_network.imu_freq_net % update_freq == 0
        )  # imu frequency must be a multiple of update frequency
        self.update_distance_num_clone = int(
            config_from_network.window_time * update_freq
        )

        # time
        self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
        self.dt_update_us = int(
            1.0 / self.update_freq * 1e6
        )  # multiple of interpolation interval

        # logging
        logging.info(
            f"Network Input Time: {config_from_network.past_time + config_from_network.window_time} = {config_from_network.past_time} + {config_from_network.window_time} (s)"
        )
        logging.info(
            f"Network Input size: {self.net_input_size} = {self.past_data_size} + {self.disp_window_size} (samples)"
        )
        logging.info("IMU interpolation frequency: %s (Hz)" % self.imu_freq_net)
        logging.info("Measurement update frequency: %s (Hz)" % self.update_freq)
        logging.info(
            "Filter update stride state number: %i" % self.update_distance_num_clone
        )
        logging.info(
            f"Interpolating IMU measurement every {self.dt_interp_us}us for the network input"
        )

        # IMU initial calibration
        self.icalib = imu_calib
        self.filter_tuning_cfg = filter_tuning_cfg # Config
        # MSCKF
        self.filter = ImuMSCKF(filter_tuning_cfg)

        net_config = {"in_dim": (self.past_data_size + self.disp_window_size) // 32 + 1}
        self.meas_source = MeasSourceTorchScript(model_path, force_cpu)

        self.imu_buffer = ImuBuffer()

        #  This callback is called at first update if set
        self.callback_first_update = None
        # This callback can be use to bypass network use for measurement
        self.debug_callback_get_meas = None

        # keep track of past timestamp and measurement
        self.last_t_us = -1

        # keep track of the last measurement received before next interpolation time
        self.t_us_before_next_interpolation = -1
        self.last_acc_before_next_interp_time = None
        self.last_gyr_before_next_interp_time = None

        self.next_interp_t_us = None
        self.next_aug_t_us = None
        self.has_done_first_update = False

    @jit(forceobj=True, parallel=False, cache=False)
    def _get_imu_samples_for_network(self, t_begin_us, t_oldest_state_us, t_end_us):
        # extract corresponding network input data
        net_tus_begin = t_begin_us
        net_tus_end = t_end_us - self.dt_interp_us

        net_acc, net_gyr, net_tus = self.imu_buffer.get_data_from_to(
            net_tus_begin, net_tus_end
        )

        assert net_gyr.shape[0] == self.net_input_size
        assert net_acc.shape[0] == self.net_input_size
        # get data from filter
        R_oldest_state_wfb, _ = self.filter.get_past_state(t_oldest_state_us)  # 3 x 3
        # change the input of the network to be in local frame
        ri_z = compute_euler_from_matrix(R_oldest_state_wfb, "xyz", extrinsic=True)[
            0, 2
        ]
        Ri_z = np.array(
            [
                [np.cos(ri_z), -(np.sin(ri_z)), 0],
                [np.sin(ri_z), np.cos(ri_z), 0],
                [0, 0, 1],
            ]
        )
        R_oldest_state_wfb = Ri_z.T @ R_oldest_state_wfb

        bg = self.filter.state.s_bg
        # dynamic rotation integration using filter states
        # Rs_net will contains delta rotation since t_begin_us
        Rs_bofbi = np.zeros((net_tus.shape[0], 3, 3))  # N x 3 x 3
        Rs_bofbi[0, :, :] = np.eye(3)
        for j in range(1, net_tus.shape[0]):
            dt_us = net_tus[j] - net_tus[j - 1]
            dR = mat_exp((net_gyr[j, :].reshape((3, 1)) - bg) * dt_us * 1e-6)
            Rs_bofbi[j, :, :] = Rs_bofbi[j - 1, :, :].dot(dR)

        # find delta rotation index at time ts_oldest_state
        oldest_state_idx_in_net = np.where(net_tus == t_oldest_state_us)[0][0]

        # rotate all Rs_net so that (R_oldest_state_wfb @ (Rs_bofbi[idx].inv() @ Rs_bofbi[i])
        # so that Rs_net[idx] = R_oldest_state_wfb
        R_bofboldstate = (
            R_oldest_state_wfb @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
        )  # [3 x 3]
        Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)
        net_acc_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_acc)  # N x 3
        net_gyr_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)  # N x 3

        return net_gyr_w, net_acc_w

    def _compensate_measurement_with_initial_calibration(self, gyr_raw, acc_raw):
        if self.icalib:
            #logging.info("Using bias from initial calibration")
            init_ba = self.icalib.accelBias
            init_bg = self.icalib.gyroBias
            # calibrate raw imu data
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )  # removed offline bias and scaled
        else:
            #logging.info("Using zero bias")
            init_ba = np.zeros((3, 1))
            init_bg = np.zeros((3, 1))
            acc_biascpst, gyr_biascpst = acc_raw, gyr_raw
        return gyr_biascpst, acc_biascpst, init_bg, init_ba

    def _after_filter_init_member_setup(self, t_us, gyr_biascpst, acc_biascpst):
        self.next_interp_t_us = t_us
        self.next_aug_t_us = t_us
        self._add_interpolated_imu_to_buffer(acc_biascpst, gyr_biascpst, t_us)
        self.next_aug_t_us = t_us + self.dt_update_us

        self.last_t_us = t_us

        self.t_us_before_next_interpolation = t_us
        self.last_acc_before_next_interp_time = acc_biascpst
        self.last_gyr_before_next_interp_time = gyr_biascpst

    def init_with_state_at_time(self, t_us, R, v, p, gyr_raw, acc_raw):
        assert R.shape == (3, 3)
        assert v.shape == (3, 1)
        assert p.shape == (3, 1)

        logging.info(f"Initializing filter at time {t_us*1e-6}")
        (
            gyr_biascpst,
            acc_biascpst,
            init_bg,
            init_ba,
        ) = self._compensate_measurement_with_initial_calibration(gyr_raw, acc_raw)
        self.filter.initialize_with_state(t_us, R, v, p, init_ba, init_bg)
        self._after_filter_init_member_setup(t_us, gyr_biascpst, acc_biascpst)
        return False

    def _init_without_state_at_time(self, t_us, gyr_raw, acc_raw):
        assert isinstance(t_us, int)
        logging.info(f"Initializing filter at time {t_us*1e-6}")
        (
            gyr_biascpst,
            acc_biascpst,
            init_bg,
            init_ba,
        ) = self._compensate_measurement_with_initial_calibration(gyr_raw, acc_raw)
        self.filter.initialize(t_us, acc_biascpst, init_ba, init_bg)
        self._after_filter_init_member_setup(t_us, gyr_biascpst, acc_biascpst)

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw):
        assert isinstance(t_us, int)
        if t_us - self.last_t_us > 3e3:
            logging.warning(f"Big IMU gap : {t_us - self.last_t_us}us")

        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw)
        else:
            self._init_without_state_at_time(t_us, gyr_raw, acc_raw)
            return False

    def _on_imu_measurement_after_init(self, t_us, gyr_raw, acc_raw):
        """
        For new IMU measurement, after the filter has been initialized
        """
        assert isinstance(t_us, int)

        # Eventually calibrate
        if self.icalib:
            # calibrate raw imu data with offline calibation
            # this is used for network feeding
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )  # removed offline bias and scaled

            # calibrate raw imu data with offline calibation scale
            # this is used for the filter. By not applying offline bias
            # we expect the filter to estimate bias similar to the offline
            # calibrated one
            acc_raw, gyr_raw = self.icalib.scale_raw(
                acc_raw, gyr_raw
            )  # only offline scaled - into the filter
        else:
            acc_biascpst = acc_raw
            gyr_biascpst = gyr_raw

        # decide if we need to interpolate imu data or do update
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        do_augmentation_and_update = t_us >= self.next_aug_t_us

        # if augmenting the state, check that we compute interpolated measurement also
        assert (
            do_augmentation_and_update and do_interpolation_of_imu
        ) or not do_augmentation_and_update, (
            "Augmentation and interpolation does not match!"
        )

        # augmentation propagation / propagation
        # propagate at IMU input rate, augmentation propagation depends on t_augmentation_us
        t_augmentation_us = self.next_aug_t_us if do_augmentation_and_update else None

        # IMU interpolation and data saving for network (using compensated IMU)
        if do_interpolation_of_imu:
            self._add_interpolated_imu_to_buffer(acc_biascpst, gyr_biascpst, t_us)

        self.filter.propagate(
            acc_raw, gyr_raw, t_us, t_augmentation_us=t_augmentation_us
        )
        # filter update
        did_update = False
        if do_augmentation_and_update:
            did_update = self._process_update(t_us)
            # plan next update/augmentation of state
            self.next_aug_t_us += self.dt_update_us

        # set last value memory to the current one
        self.last_t_us = t_us

        if t_us < self.t_us_before_next_interpolation:
            self.t_us_before_next_interpolation = t_us
            self.last_acc_before_next_interp_time = acc_biascpst
            self.last_gyr_before_next_interp_time = gyr_biascpst

        return did_update

    def _process_update(self, t_us):
        logging.debug(f"Upd. @ {t_us * 1e-6} | Ns: {self.filter.state.N} ")
        # get update interval t_begin_us and t_end_us
        if self.filter.state.N <= self.update_distance_num_clone:
            return False
        t_oldest_state_us = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1
        ]
        t_begin_us = t_oldest_state_us - self.dt_interp_us * self.past_data_size
        t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        # If we do not have enough IMU data yet, just wait for next time
        if t_begin_us < self.imu_buffer.net_t_us[0]:
            return False
        # initialize with vio at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)
        assert t_begin_us <= t_oldest_state_us
        if self.debug_callback_get_meas:
            meas, meas_cov = self.debug_callback_get_meas(t_oldest_state_us, t_end_us)
        else:  # using network for measurements
            net_gyr_w, net_acc_w = self._get_imu_samples_for_network(
                t_begin_us, t_oldest_state_us, t_end_us
            )
            meas, meas_cov = self.meas_source.get_displacement_measurement(
                net_gyr_w, net_acc_w
            )
        # filter update
        self.filter.update(meas, meas_cov, t_oldest_state_us, t_end_us)
        self.has_done_first_update = True
        # marginalization of all past state with timestamp before or equal ts_oldest_state
        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        cut_idx = oldest_idx
        logging.debug(f"marginalize {cut_idx}")
        self.filter.marginalize(cut_idx)
        self.imu_buffer.throw_data_before(t_begin_us)
        return True

    def _add_interpolated_imu_to_buffer(self, acc_biascpst, gyr_biascpst, t_us):
        self.imu_buffer.add_data_interpolated(
            self.t_us_before_next_interpolation,
            t_us,
            self.last_gyr_before_next_interp_time,
            gyr_biascpst,
            self.last_acc_before_next_interp_time,
            acc_biascpst,
            self.next_interp_t_us,
        )
        self.next_interp_t_us += self.dt_interp_us
