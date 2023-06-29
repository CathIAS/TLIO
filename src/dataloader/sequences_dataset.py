"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import json
import torch
import shutil
import ctypes
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation

from utils.from_scipy import compute_euler_from_matrix
from utils.logging import get_logger
from .constants import *

log = get_logger(__name__)

class SequencesDataset:
    """
    A template class for sequences dataset in TLIO training.
    Each subclass is expected to load data in a different way, but from the same data format.
    """

    def __init__(
        self,
        data_path,
        split,
        genparams,
        only_n_sequence=-1,
        sequence_subset=None,
        normalize_sensor_data=True,
        verbose=False,
    ):
        self.data_path = data_path
        self.split = split
        self.genparams = genparams
        
        self.only_n_sequence = only_n_sequence
        self.sequence_subset = sequence_subset
        self.normalize_sensor_data = normalize_sensor_data
        self.verbose = verbose
        
        # The list of relevant sensor file names based on data_style
        self.sensor_file_basenames = self.get_sensor_file_basenames()
            
        # Index the mem-mapped files and open them (data is not read from disk here)
        self.load_list()
        if self.verbose:
            self.log_dataset_info()
        
    def get_base_sensor_name(self):
        return self.sensor_file_basenames[0]
    
    def load_list(self):
        assert torch.utils.data.get_worker_info() is None, "load_list() can only be called in main proc!"

        #list_info = np.loadtxt(
        #    os.path.join(self.data_path, f"{self.split}_list.txt"), 
        #    dtype=np.dtype(str),
        #)
        with open(os.path.join(self.data_path, f"{self.split}_list.txt")) as f:
            list_info = np.array([s.strip() for s in f.readlines() if len(s.strip()) > 0])
        
        # For picking exactly some particular sequences
        if self.sequence_subset is not None:
            to_keep = np.array([s in self.sequence_subset for s in list_info])
            assert np.count_nonzero(to_keep) == len(self.sequence_subset), \
                    f"Could not find some sequences from sequence_subset in data list"
            list_info = list_info[to_keep]

        if self.split == "train" and self.only_n_sequence > 0:
            list_info = list_info[:self.only_n_sequence]
            
        # Handle empty lists (i.e., if you don't want to do test or val or something)
        self.data_list = []
        if len(list_info) > 0:
            self.data_list = list_info
        
        # Load the descriptions of all the data (column info and num rows)
        self.data_descriptions = []
        seqs_to_remove = [] # The seqs, not the index
        for seq_id in self.data_list:
            seq_desc = {}
            valid = True
            for i, sensor_basename in enumerate(self.sensor_file_basenames):
                with open(os.path.join(self.data_path, seq_id, 
                        sensor_basename+"_description.json"), 'r') as f: 
                    d = json.load(f)
                    if i == 0 and d["num_rows"] < self.genparams.window_size:
                        valid = False
                        log.warning(f"Sequence {seq_id} being ignored since it is too short ({d['num_rows']} rows)")
                        break
                    seq_desc[sensor_basename] = d
            
            if valid:
                self.data_descriptions.append(seq_desc)
            else:
                seqs_to_remove.append(seq_id)

        # Remove too short sequences from list
        if len(seqs_to_remove) > 0:
            self.data_list = np.array([seq for seq in self.data_list if seq not in seqs_to_remove])

    def get_sensor_file_basenames(self):
        if self.genparams.data_style == "aligned":
            return [COMBINED_SENSOR_NAME]
        elif self.genparams.data_style == "resampled":
            return [s + "_resampled" for s in ALL_SENSORS_LIST if s in self.genparams.input_sensors]
        elif self.genparams.data_style == "raw":
            return [s for s in ALL_SENSORS_LIST if s in self.genparams.input_sensors]
        else:
            raise ValueError(f"Invalid data_style {self.genparams.data_style}")
    
    def log_dataset_info(self):
        cumulated_duration_hrs = 0
        self.max_num_rows = None
        self.min_num_rows = None
        for i, seq_id in enumerate(self.data_list):
            seq_fps = {}
            desc = self.data_descriptions[i]
            for j, sensor_basename in enumerate(self.sensor_file_basenames):
                sensor_desc = desc[sensor_basename]
                num_cols = sum([
                    int(c.split("(")[1].split(")")[0]) for c in sensor_desc["columns_name(width)"]
                ])
                cumulated_duration_hrs += 1e-6 * (sensor_desc["t_end_us"] - sensor_desc["t_start_us"]) / 60 / 60
                self.max_num_rows = (
                    sensor_desc["num_rows"] if self.max_num_rows is None
                    else max(sensor_desc["num_rows"], self.max_num_rows)
                )
                self.min_num_rows = (
                    sensor_desc["num_rows"] if self.min_num_rows is None
                    else min(sensor_desc["num_rows"], self.min_num_rows)
                )
    
        # log some statitstics
        #log.info(f"Using these sequences: {list(self.data_list)}")
        log.info(
            f"Cumulated {self.split} dataset duration is {cumulated_duration_hrs:.3f} hours"
        )
        log.info(
            f"Number of {self.split} sequences is {len(self.data_descriptions)}"
        )
        #log.info(
        #    f"Number of {self.split} samples is {self.length} "
        #    f"(decimated by {self.genparams.decimator}x)"
        #)
        log.info(f"Min/max sequences length={self.min_num_rows}, {self.max_num_rows}") 
    
    def poses_to_target(self, rot, pos):
        # Calculate relative info on the fly
        # targ is what we want to regress from these features
        R_W_0 = Rotation.from_quat(rot[0:1]).as_matrix()
        R_W_i = Rotation.from_quat(rot).as_matrix()

        # NOTE R_W_i @ R_W_0.transpose() looks strange, but it is the delta rotation between the two times
        # aligned with the world frame instead of body frame.
        targ_dR_World = R_W_i @ R_W_0.transpose([0,2,1])
        targ_dt_World = pos - pos[0:1] # Displacement in global frame
        return targ_dR_World, targ_dt_World

    def unpack_data_window(self, seq_data, seq_desc, row):
        feats = {}
        ts_us_base_sensor = None
        if self.genparams.data_style == "aligned":
            data_chunk = seq_data[COMBINED_SENSOR_NAME][row:row+self.genparams.window_size]
            # Make sure idx was valid with sufficient padding for window
            assert data_chunk.shape[0] == self.genparams.window_size
            
            #ts_us, gyr, acc, rot, pos, vel = np.hsplit(data_chunk, [1, 4, 7, 11, 14])
            ts_us = data_chunk[:,0:1]
            ts_us_base_sensor = np.copy(ts_us)

            # Check which sensor data we need to load
            # TODO make this less hard-coded for the columns somehow
            # Maybe the description json could be better for getting the columns
            #if "imu0" in self.genparams.input_sensors:
            if "imu0" in self.genparams.input_sensors:
                feats["imu0"] = data_chunk[:,1:7] # gyro0, accelerometer0
            if "imu1" in self.genparams.input_sensors:
                feats["imu1"] = data_chunk[:,7:13] # gyro1, accelerometer1
            if "mag0" in self.genparams.input_sensors:
                feats["mag0"] = data_chunk[:,13:16]
            if "barom0" in self.genparams.input_sensors:
                feats["barom0"] = data_chunk[:,16:18]
            
            # All sensors have the same timestamps in this data_style, just concat here
            #ts_normalized = 2 * (ts_us - ts_us[0]) / (ts_us[-1] - ts_us[0]) - 1
            for k, v in feats.items():
            #    feats[k] = np.concatenate([ts_normalized, v], axis=1).astype(np.float32).T
                feats[k] = v.astype(np.float32).T
            
            rot, pos, vel = data_chunk[:,-10:-6], data_chunk[:,-6:-3], data_chunk[:,-3:]

        elif self.genparams.data_style == "resampled":
            
            # With resampled data, the "approximate_frequency" in the json file is exact,
            # so we can quickly index the timestamps of sensors in different memmap files.
            base_sensor_freq = seq_desc[self.get_base_sensor_name()]["approximate_frequency_hz"]
            base_sensor_window_start_time = None
            base_sensor_window_end_time = None
            for i, sensor_name in enumerate(self.sensor_file_basenames):
                if i == 0:
                    sensor_row = row
                    window_size = self.genparams.window_size
                else:
                    # Index the row based on sensor frequency.
                    sensor_freq = seq_desc[sensor_name]["approximate_frequency_hz"]
                    sensor_seq_start_time = seq_desc[sensor_name]["t_start_us"]
                    # TODO off by one possible here from rounding/flooring
                    sensor_row = int(1e-6*(base_sensor_window_start_time - sensor_seq_start_time) * sensor_freq)
                    # TODO should calculate all the window sizes at startup so that we don't
                    # accidentally get an off-by-one window size error from float errors
                    window_size = int(self.genparams.window_size * sensor_freq / base_sensor_freq)
            
                data_chunk = seq_data[sensor_name][sensor_row:sensor_row+window_size]
                # Make sure idx was valid with sufficient padding for window
                assert data_chunk.shape[0] == window_size
                ts_us = data_chunk[:,0:1]

                if i == 0:
                    ts_us_base_sensor = np.copy(ts_us)
                    base_sensor_window_start_time = ts_us[0]
                    base_sensor_window_end_time = ts_us[-1]
                    # GT data comes from base sensor
                    rot, pos, vel = data_chunk[:,-10:-6], data_chunk[:,-6:-3], data_chunk[:,-3:]
                
                if "imu" in sensor_name:
                    feat = data_chunk[:,1:7] # gyro, accelerometer
                if "mag" in sensor_name:
                    feat = data_chunk[:,1:4]
                if "barom" in sensor_name:
                    feat = data_chunk[:,1:3]
                
                # All timestamps relative to base sensor
                #ts_normalized = 2 * (
                #    ts_us - ts_us_base_sensor[0]
                #) / (
                #    ts_us_base_sensor[-1] - ts_us_base_sensor[0]
                #) - 1
                
                #feats[sensor_name.split("_")[0]] = np.concatenate([ts_normalized, feat], axis=1).astype(np.float32).T
                feats[sensor_name.split("_")[0]] = feat.astype(np.float32).T

        else:
            raise NotImplementedError()
        
        gt_data = ts_us_base_sensor, rot.astype(np.float32), pos.astype(np.float32), vel.astype(np.float32)
        return feats, gt_data

    def data_chunk_from_seq_data(self, seq_data, seq_desc, row):
        feats, gt_data = self.unpack_data_window(seq_data, seq_desc, row)

        # Normalize the raw sensor data into something better for learning (sensor-dependent)
        if self.normalize_sensor_data:
            feats = self.normalize_feats(feats)
        
        ts_us, rot, pos, vel = gt_data
        targ_dR_World, targ_dt_World = self.poses_to_target(rot, pos)

        R_world_gla = np.eye(3)
        if self.genparams.express_in_t0_yaw_normalized_frame:
            assert False
            R_W_0 = Rotation.from_quat(rot[0:1]).as_matrix()
            angles_t0 = compute_euler_from_matrix(
                R_W_0, "xyz", extrinsic=True
            )
            ri_z = angles_t0[0,2]
            c = np.cos(ri_z)
            s = np.sin(ri_z)
            R_world_gla = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            targ_dt_World = np.einsum("ji,tj->ti", R_world_gla, targ_dt_World)

            # Only IMU and mag data need to be rotated (not barometer)
            for k, v in feats.items():
                if "imu" in k:
                    assert feats[k].shape[0] == 6
                    feats[k][:3] = np.einsum("ji,jt->it", R_world_gla, feats[k][:3])
                    feats[k][3:] = np.einsum("ji,jt->it", R_world_gla, feats[k][3:])
                elif "mag" in k:
                    assert feats[k].shape[0] == 3
                    feats[k] = np.einsum("ji,jt->it", R_world_gla, feats[k])
        
        # We may return multiple windows, so place them all in here for convenience.
        windows = {
            "main": {
                "ts_us": ts_us,
                "feats": feats,
                "targ_dR_World": targ_dR_World.astype(np.float32),
                "targ_dt_World": targ_dt_World.astype(np.float32),
                "vel_World": vel.astype(np.float32),
                "R_world_gla": R_world_gla,
            }
        }

        return seq_desc[self.get_base_sensor_name()], windows
    
    def normalize_feats(self, feats):
        """
        Normalize the sensor data from its raw form to some normalized form, typically in [-1,1] or [0,1].
        """
        
        new_feats = {}
        for sensor_name, feat in feats.items():
            # Note that all feat are [1+C,T] where C is channels in sensor data and T is tme dimension.
            # The 1+ is because the sensor data is concatenated with normalized time stamp.
            new_feat = np.copy(feat)
            # Check for nan/inf here (sometimes can pop up in the data)
            new_feat[~np.isfinite(new_feat)] = 0.0
            """  NOTE makes values too small, and disrupts bias perturbation logic
            if "imu" in sensor_name:
                assert new_feat.shape[0] == 6
                # See T74692750 for more info.
                # Out of the two IMUs, the one with the max range is at +/-8G and +/-1000 deg/sec.
                # Normalize by this one so that both IMU values have the same meaning, and are normalized in [-1,1]
                minmax_acc_range_g = 8 # In unit of Gs
                minmax_ang_vel_range_deg_per_sec = 1000
                # IMU values should be in [-1,1] after this
                new_feat[:3] = new_feat[:3] / (minmax_ang_vel_range_deg_per_sec / 180 * np.pi) # gyro
                new_feat[3:6] = new_feat[3:6] / (minmax_acc_range_g * 9.81) # accelerometer
            """
            if "mag" in sensor_name:
                assert new_feat.shape[0] == 3
                # Convert to Gauss, which is closer to 1 in magnitude (Earth's field is around .25-.65 Gauss, and 
                # can be negative here since the magnetomete returns a magnetic field vector instead of magnitude)
                GAUSS_IN_TESLA = 10_000
                new_feat[:3] = new_feat[:3] * GAUSS_IN_TESLA
            if "barom" in sensor_name:
                assert new_feat.shape[0] == 2
                # Pressure converted to bar and normalized heuristically to fit into [-1,1] better.
                # Setting -1,1 to be the min/max pressure/temp ever recorded leads to very small differences
                # in the values for normal situations, so just picked min/max based on some normal daily values on Earth.
                PA_IN_BAR = 100_000
                """
                avg_bar = 1.01325 # Average barometric pressure on earth
                max_bar_deviation = 0.01 # plus/minus avg is what we are considering
                min_bar, max_bar = avg_bar - max_bar_deviation, avg_bar + max_bar_deviation
                new_feat[0] = 2 * (new_feat[0] / PA_IN_BAR - min_bar) / (max_bar - min_bar) - 1
                min_temp = -100
                max_temp = 100
                new_feat[1] = 2 * (new_feat[1] - min_temp) / (max_temp - min_temp) - 1
                """
                new_feat[0] /= PA_IN_BAR # convert pa to bar

            new_feats[sensor_name] = new_feat
        
        return new_feats
    
    def load_data_chunk(self, seq_idx, row):
        raise NotImplementedError("Did not override load_data_chunk!!!")

    def load_and_preprocess_data_chunk(self, seq_idx, row_in_seq, num_rows_in_seq):
        # If training, randomize the row a bit so that we can get better coverage of the data
        # while still respecting the decimator and indexing.
        if self.split == "train":
            row_in_seq = min(num_rows_in_seq-1, row_in_seq + np.random.randint(self.genparams.decimator))
        meta_dict, windows = self.load_data_chunk(seq_idx, row_in_seq)

        ret = {
            "seq_id": self.data_list[seq_idx],
        }
        ret.update(windows["main"]) # Main target and GT data corresponding to seq_idx and row_in_seq
        return ret

    ##########################################################################
    # Functions needed by IntegrateRoninCallback
    ##########################################################################

    def get_ts_last_imu_us(self, seq_idx=0):
        raise NotImplementedError("Did not override get_ts_last_imu_us!!!")

    def get_gt_traj_center_window_times(self, seq_idx=0):
        raise NotImplementedError("Did not override get_gt_traj_center_window_times!!!")
