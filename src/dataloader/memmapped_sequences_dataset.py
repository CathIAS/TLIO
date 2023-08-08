"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
import torch
import shutil
import ctypes
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from utils.logging import get_logger
from .sequences_dataset import SequencesDataset
from .constants import *

log = get_logger(__name__)

class MemMappedSequencesDataset(Dataset, SequencesDataset):
    """
    This class contains a list of open numpy mem-mapped files,
    each containing the motion data from one sequence (compensated IMU, VIO gt)
    """

    def __init__(
        self,
        data_path,
        split,
        genparams,
        only_n_sequence=-1,
        sequence_subset=None,
        keep_all_memmap_open=True,
        use_index_map=True,
        verbose=False,
        store_in_ram=False,
    ):
        SequencesDataset.__init__(
            self,
            data_path=data_path,
            split=split,
            genparams=genparams,
            only_n_sequence=only_n_sequence,
            sequence_subset=sequence_subset,
            verbose=verbose,
        )

        self.use_index_map = use_index_map # If false, calculate indices on the fly to save mem
        self.keep_all_memmap_open = store_in_ram or keep_all_memmap_open # If true, keep all memmap files open the whole time
        self.store_in_ram = store_in_ram

        # Index the mem-mapped files (data is not read from disk here)
        if self.use_index_map:
            self.build_index_map()
        else:
            # We found that this can be faster and more mem-efficient with a large dataset.
            self.setup_on_the_fly_indexing()

        self.load_memmap_files()   
        
    def build_index_map(self):
        # The index map will map to the indices in the base sensor, which is the 
        # first one in the sensor_file_basenames list (IMU0 typically)
        # If we need to load data from other sensor files (i.e, if data_style != "combined")
        # we will perform a guided search for the timestamps in the other files,
        # with the help of the "approximate_frequency" attribute in the json files.
        
        base_sensor_name = self.get_base_sensor_name()

        self.index_map = np.empty((sum([
            d[base_sensor_name]["num_rows"]-self.genparams.window_size+1 for d in self.data_descriptions
        ]), 2), dtype=np.int32)
        curr_idx = 0

        for i, desc in enumerate(self.data_descriptions):
            rows = desc[base_sensor_name]["num_rows"]
            idx_rows = rows - self.genparams.window_size + 1
            # Sequence index
            self.index_map[curr_idx:curr_idx+idx_rows,0] = i
            # Index within the sequence
            self.index_map[curr_idx:curr_idx+idx_rows,1] = np.arange(idx_rows, dtype=np.int32)
            curr_idx += idx_rows
        assert curr_idx == self.index_map.shape[0]

        # Decimate the data to avoid redundant samples
        self.index_map = self.index_map[::self.genparams.decimator]
        self.length = len(self.index_map)

        """ Attempt more mem efficient index map (not just take ::step at the end, build it smaller at first)
        dec = self.genparams.decimator
        index_map2 = np.empty((sum([
            (d[base_sensor_name]["num_rows"]-self.genparams.window_size+1)//dec for d in self.data_descriptions
        ]), 2), dtype=np.int32)
        curr_idx = 0
        for i, desc in enumerate(self.data_descriptions):
            idx_rows = (desc[base_sensor_name]["num_rows"] - self.genparams.window_size + 1) // dec
            idx_within_seq = np.arange(0, idx_rows*dec, dec, dtype=np.int32)
            # Sequence index
            index_map2[curr_idx:curr_idx+len(idx_within_seq),0] = i
            index_map2[curr_idx:curr_idx+len(idx_within_seq),1] = idx_within_seq
            curr_idx += len(idx_within_seq)
        assert curr_idx == index_map2.shape[0] 

        assert len(index_map2) == len(self.index_map), f"{len(index_map2)} != {len(self.index_map)}"
        assert np.all(index_map2 == self.index_map), f"! {np.count_nonzero(index_map2 != self.index_map)}"
        """

    def setup_on_the_fly_indexing(self):
        # Instead of storing a flat index map mapping idx -> (seq_idx, timestamp_idx),
        # we store a list_cumsum, which is the length of the number of sequences,
        # and contains the cumultive sum at each idx of the number of timestamps
        # up to that sequence, including the sequence at idx, then perform a binary
        # search on this array to find the index in the sequence for the timestamp.
        # We found in preliminary tests that this was faster and more memory efficient
        # for scaled-up training.
        
        # For each sequence, the total data length up to that point
        data_dim = np.array([
            d[self.get_base_sensor_name()]["num_rows"] for d in self.data_descriptions
        ])
        self.list_cumsum = np.cumsum((
            data_dim - self.genparams.window_size #+ 1
        ) // self.genparams.decimator, dtype=np.int64)
        #self.length = (
        #    np.sum(self.data_dim) + len(self.data_list)*(1-self.genparams.window_size)
        #) // self.genparams.decimator
        self.length = self.list_cumsum[-1] if len(self.list_cumsum) > 0 else 0

    def load_memmap_files(self):
        # Memmap files should increase RAM alot over time as they fill up,
        # but based on our tests it does not. If it does, just open the memmpap
        # file at each getitem call instead (not too much overhead we found)
        # Let's see what happens....
        self.fps = [None] * len(self.data_list)
        self.memmap_filenames = [None] * len(self.data_list)
        cumulated_duration_hrs = 0
        self.max_num_rows = None
        self.min_num_rows = None
        for i, seq_id in enumerate(self.data_list):
            seq_fps = {}
            seq_memmap_filenames = {}
            desc = self.data_descriptions[i]
            for j, sensor_basename in enumerate(self.sensor_file_basenames):
                filename = os.path.join(self.data_path, seq_id, sensor_basename+".npy")
                seq_memmap_filenames[sensor_basename] = filename
                sensor_desc = desc[sensor_basename]
                num_cols = sum([
                    int(c.split("(")[1].split(")")[0]) for c in sensor_desc["columns_name(width)"]
                ])
                cumulated_duration_hrs += 1e-6 * (sensor_desc["t_end_us"] - sensor_desc["t_start_us"]) / 60 / 60
                self.max_num_rows = sensor_desc["num_rows"] if self.max_num_rows is None \
                        else max(sensor_desc["num_rows"], self.max_num_rows)
                self.min_num_rows = sensor_desc["num_rows"] if self.min_num_rows is None \
                        else min(sensor_desc["num_rows"], self.min_num_rows)
                if self.keep_all_memmap_open:
                    if self.store_in_ram:
                        fp = np.load(filename)
                    else:
                        fp = np.load(filename, mmap_mode='c')

                    seq_fps[sensor_basename] = fp
            self.fps[i] = seq_fps
            self.memmap_filenames[i] = seq_memmap_filenames 
    
    def map_index(self, idx):
        if self.use_index_map:
            return self.index_map[idx]
        else:
            return self.idx2tuple(idx)

    def idx2tuple(self, idx):
        """
        Give the index of the sequence and row in sequence without storing large flat index map
        """
        random_idx = False #self.split=="train" and len(self.list_cumsum) > 3000
        
        if not random_idx:
            # binsearch is super slow, try to speed it up with some bounds
            # Minimum sequence loc based on min num rows (if all sequences were min_num_rows long)
            #seq_idx_max = idx // self.min_num_rows
            #seq_idx_min = idx // self.max_num_rows
            #seq_idx = np.searchsorted(self.list_cumsum[seq_idx_min:seq_idx_max+1], idx) + seq_idx_min
            seq_idx = np.searchsorted(self.list_cumsum, idx)
            #print(idx, seq_idx_min, seq_idx_max, seq_idx, seq_idx_check)
            #assert seq_idx == seq_idx_check

            assert seq_idx < len(self.list_cumsum)
            row = (
                idx - (self.list_cumsum[seq_idx-1] if seq_idx>0 else 0)
            ) * self.genparams.decimator
            max_rows = self.data_descriptions[seq_idx][self.get_base_sensor_name()]["num_rows"]
            assert row < max_rows - self.genparams.window_size + 1 
        else:
            seq_idx = np.random.randint(len(self.list_cumsum))
            
            """
            # Each worker only responsible for chunk of dataset
            winfo = torch.utils.data.get_worker_info()
            seq_idx = np.random.randint(
                winfo.id*len(self.list_cumsum)//winfo.num_workers,
                (1+winfo.id)*len(self.list_cumsum)//winfo.num_workers,
            )"""

            row = np.random.randint(self.list_cumsum[seq_idx] - (self.list_cumsum[seq_idx-1] if seq_idx!=0 else 0))

        return seq_idx, row
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq_idx, row_in_seq = self.map_index(idx)
        ret = self.load_and_preprocess_data_chunk(
            seq_idx, row_in_seq, 
            self.data_descriptions[seq_idx][self.get_base_sensor_name()]["num_rows"]-self.genparams.window_size,
        )
        return ret

    def load_data_chunk(self, seq_idx, row):       
        if not hasattr(self, "madvise"):
            # https://github.com/numpy/numpy/issues/13172
            # Doesn't seem to help our speed
            self.madvise = ctypes.CDLL("libc.so.6").madvise
            self.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double]
            self.madvise.restype = ctypes.c_double

        seq_desc = self.data_descriptions[seq_idx]
        if self.keep_all_memmap_open:
            seq_data = self.fps[seq_idx]
        else:
            seq_data = {}
            for sensor_name, sensor_desc in seq_desc.items():
                filename = self.memmap_filenames[seq_idx][sensor_name]
                num_cols = sum([
                    int(c.split("(")[1].split(")")[0]) for c in sensor_desc["columns_name(width)"]
                ])
                seq_data[sensor_name] = np.load(filename, mmap_mode='c') #np.memmap(filename, dtype=np.float64, mode='c',
                #        shape=(sensor_desc["num_rows"],num_cols))
    
        return self.data_chunk_from_seq_data(seq_data, seq_desc, row)
    
    def get_ts_last_imu_us(self, seq_idx=0):
        """
        Get the last timestamp for all IMU windows in this sequence
        """
        # Each fp columns: ts_us, gyr (x,y,z), accel (x,y,z), q (x,y,z,w), t (x,y,z), vel (x,y,z)
        # This grabs index self.genparams.window_size-1 and every self.genparams.decimator'th
        # index thereafter in the sequence
        if self.keep_all_memmap_open:
            fp = self.fps[seq_idx][self.get_base_sensor_name()]
        else:
            desc = self.data_descriptions[seq_idx][self.get_base_sensor_name()]
            filename = self.memmap_filenames[seq_idx][self.get_base_sensor_name()]
            num_cols = sum([
                int(c.split("(")[1].split(")")[0]) for c in desc["columns_name(width)"]
            ])
            #fp = np.memmap(filename, dtype=np.float64, mode='c', shape=(desc["num_rows"],num_cols))
            fp = np.load(filename, mmap_mode='c')
        ts = fp[self.genparams.window_size-1::self.genparams.decimator,0]
        if not self.use_index_map: # off-by-one sometimes between these two options
            #new_len = (len(fp) - self.genparams.window_size + 1) // self.genparams.decimator
            new_len = self.list_cumsum[seq_idx] - (self.list_cumsum[seq_idx-1] if seq_idx!=0 else 0)
            assert abs(new_len - len(ts)) < 2, \
                    f"Expected off-by-one at most, but got {abs(new_len - len(ts))}"
            ts = ts[:new_len]
        return ts

    def get_gt_traj_center_window_times(self, seq_idx=0):
        """
        Get the GT orientatoin/position (in world frame) at the center 
        time for each IMU window in this sequence.
        Returned as stacked [N,4,4] SE3 matrices
        """
        # Each fp columns: ts_us, gyr (x,y,z), accel (x,y,z), q (x,y,z,w), t (x,y,z), vel (x,y,z)
        # This grabs index (self.genparams.window_size-1)//2 and every self.genparams.decimator'th
        # index thereafter in the sequence
        if self.keep_all_memmap_open:
            fp = self.fps[seq_idx][self.get_base_sensor_name()]
        else:
            desc = self.data_descriptions[seq_idx][self.get_base_sensor_name()]
            filename = self.memmap_filenames[seq_idx][self.get_base_sensor_name()]
            num_cols = sum([
                int(c.split("(")[1].split(")")[0]) for c in desc["columns_name(width)"]
            ])
            #fp = np.memmap(filename, dtype=np.float64, mode='c', shape=(desc["num_rows"],num_cols))
            fp = np.load(filename, mmap_mode='c')

        start_idx = (self.genparams.window_size - 1) // 2
        end = len(fp) - self.genparams.window_size//2
        traj = fp[start_idx:end:self.genparams.decimator,-10:-3] # [start:stop:step]
        if not self.use_index_map: # off-by-one sometimes between these two options
            #new_len = (len(fp) - self.genparams.window_size + 1) // self.genparams.decimator
            new_len = self.list_cumsum[seq_idx] - (self.list_cumsum[seq_idx-1] if seq_idx!=0 else 0)
            assert abs(new_len - len(traj)) < 2, \
                    f"Expected off-by-one at most, but got {abs(new_len - len(traj))}"
            traj = traj[:new_len]
        return Rotation.from_quat(traj[:,:4]), traj[:,4:]
