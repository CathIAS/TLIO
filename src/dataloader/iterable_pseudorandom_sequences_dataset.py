"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import torch
import ctypes
import numpy as np
from uuid import uuid4
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation

# Use thread for sequence prefetching.
# Not completely parallel, but prevents us from being hung up
# when we need to replenish a sequence in the cache.
# Note that we can't use multiprocessing since the dataloader
# workers are already daemonic processes, and are not allowed
# to have children.
from queue import Queue
from threading import Thread

from utils.logging import get_logger
from .sequences_dataset import SequencesDataset
from .constants import *

log = get_logger(__name__)

class IterablePseudoRandomSequencesDataset(IterableDataset, SequencesDataset):
    """
    Iterator-style dataset that loads data in pseudo-random manner.
    """

    def __init__(
        self,
        data_path,
        split,
        genparams,
        # How many seqs PER WORKER to cache in RAM. num_workers is multiplied by ngpu if doing DDP training.
        # Larger cache -> more randomness, but filling up more RAM of course.
        num_cache_seqs=150, 
        only_n_sequence=-1,
        sequence_subset=None,
        prefetch=True,
        verbose=False,
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
        
        self.num_cache_seqs = num_cache_seqs
        self.prefetch = prefetch

        # Keep orig list around in case we need to shuffle it before each epoch for multiple workers that
        # only look at a chunk of the dataset each.
        self.data_list_orig = self.data_list.copy()
        self.data_descriptions_orig = self.data_descriptions.copy()

        self.uuid = str(uuid4())
        self.current_epoch = 0

        # Purely for logging purposes (e.g., pytorch lightning progress bar)
        self.calculate_len()
        
        # Other indexing stuff is lazilly loaded in __iter__ (by that time, it's in the worker proc)
        self.seq_cache = None
        self.need_init = True
        if self.prefetch:
            self.prefetch_queue = Queue(maxsize=self.num_cache_seqs)
            self.prefetch_thread = None

    def calculate_len(self):
        # This is simply for logging purposes (so pytorch lightning can display progress bar)    

        base_sensor_name = self.get_base_sensor_name()

        self.length = sum([
            d[base_sensor_name]["num_rows"]-self.genparams.window_size+1 for d in self.data_descriptions
        ]) // self.genparams.decimator

    def reshuffle_data_list(self, local_rank, world_size, worker_id):
        """
        Re-shuffle the original data list (worker 0, rank 0) and send it to all the workers via a tmp text file.
        This will increase the randomness in training since the chunk for each worker will change after each epoch.
        We use a text file instead of torch multiprocessing communication since we may be training with DDP
        and that can be a hassle
        """
        assert self.split == "train"
        # split is set in main process in __init__, current_epoch is the same for all workers
        # since it's incremented after every StopIteration
        # TODO cleanup tmp files after done with them
        uuid = os.environ.get("TORCHELASTIC_RUN_ID")
        if uuid is None:
            assert world_size == 1, \
                    "Please launch training with 'torchrun --rdzv_id=`uuidgen`' for DDP training, missing env variables"
            # self.uuid is only valid if the main proc is the same for all GPUs (i.e., DataParallel or single GPU)
            # otherwise, in DDP, each GPU got a diffent uuid from __init__, so this would not work, hence the assert.
            uuid = self.uuid

        tmp_list_file = f"/tmp/shuffled_data_list_{self.split}_{uuid}_{self.current_epoch}.npy"
        if local_rank == 0 and worker_id == 0:
            # Write the new data list to file
            new_data_list_order = np.arange(len(self.data_list_orig))
            np.random.shuffle(new_data_list_order)
            np.save(tmp_list_file, new_data_list_order)
        else:
            found = False
            while not found:
                while not os.path.exists(tmp_list_file):
                    time.sleep(0.1)
                time.sleep(0.1)
                try:
                    new_data_list_order = np.load(tmp_list_file)
                    found = True
                except ValueError: # If for some reason file not done writing (never happened but just being robust)
                    continue

        self.data_list = self.data_list_orig[new_data_list_order]
        self.data_descriptions = [self.data_descriptions_orig[i] for i in new_data_list_order]
        assert len(self.data_list) == len(self.data_list_orig)
        assert len(self.data_descriptions) == len(self.data_list_orig)

    def maybe_reinit_and_reset_cache(self):
        """
        Split the datalist up per worker, this worker only gets a portion of the sequences to work on
        If in train mode, the portion is randomized each time (each worker gets a different portion than before)
        """

        winfo = torch.utils.data.get_worker_info() # None if on main proc (workers=0), but world_size may be >1
        node_rank = int(os.environ.get("NODE_RANK", 0))
        # TODO split by node rank too if we want training at that scale.
        assert node_rank == 0, f"Not yet setup for multi-node training. Found node_rank={node_rank}"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.need_init and self.sequence_subset is None and (winfo is not None or world_size > 1):
            if winfo is None:
                class _DefaultWorkerInfo:
                    id = 0
                    num_workers = 1
                winfo = _DefaultWorkerInfo()

            # Re-shuffle the original data list and give this worker a different portion than the last epoch
            if self.split == "train": # and self.current_epoch > 0:
                self.reshuffle_data_list(local_rank, world_size, winfo.id)

            # So far, in the main process (init functoin), we only loaded data_list, data_descriptions,
            # to calculate it (unique ID number for each user)
            orig_len = len(self.data_list)
            
            if world_size > 1:
                # Slice the data by GPU (if doing DDP, each GPU has it's own proc, and this worker is a subproc of it)
                per_gpu = int(round(len(self.data_list) / float(world_size)))
                # Give the last worker the remainder TODO distribute the remainder better
                if local_rank == world_size-1:
                    local_slice = slice(local_rank*per_gpu, None)
                else:
                    local_slice = slice(local_rank*per_gpu, (local_rank+1)*per_gpu)
                self.data_list = self.data_list[local_slice]
                self.data_descriptions = self.data_descriptions[local_slice]

            per_worker = int(round(len(self.data_list) / float(winfo.num_workers)))
            # Give the last worker the remainder TODO distribute the remainder better
            if winfo.id == winfo.num_workers-1:
                worker_slice = slice(winfo.id*per_worker, None)
            else:
                worker_slice = slice(winfo.id*per_worker, (winfo.id+1)*per_worker)
            self.data_list = self.data_list[worker_slice]
            self.data_descriptions = self.data_descriptions[worker_slice]

            #log.info(f"DATA WORKER {winfo.id+1}/{winfo.num_workers} Done (loading {len(self.data_list)}/{orig_len}.")
            assert len(self.data_list) > 0, "Not enough data for worker. Please make sure num sequences >= num_workers"

            # Make sure each worker has a different random seed since we just shuffle index_map
            # Training calls this every time, make sure to only shuffle the first time.
            if self.current_epoch == 0:
                np.random.seed(winfo.id)

            if self.split != "train":
                self.need_init = False

        self.num_cache_seqs = min(self.num_cache_seqs, len(self.data_list))
        
        # This function is called at the start of each epoch. No matter what we need to restart the seq caching.
        self.init_seq_cache()

    def launch_prefetching(self):
        
        # TODO persistent prefetch thread (how?)
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()

        seq_ids_in_load_order = [self.data_list[i] for i in self.avail_seq_inds]
        self.prefetch_thread = Thread(
            target=prefetch_worker_fn, 
            args=(
                self.prefetch_queue, 
                self.data_path, 
                seq_ids_in_load_order, 
                self.sensor_file_basenames
            ),
            daemon=True, # Exit when parent proc exits
        )
        self.prefetch_thread.start()

    def init_seq_cache(self):
        # Store the numpy arrays in this cache
        # Once a sequence has been read completely, we replace it with another.
        # If there are no more seqs available to replenish cache, we just pop the location in these lists.
        self.seq_cache = [None] * self.num_cache_seqs
        # Each index of this list is an array of indices that are available in this sequence
        self.iter_order_inds_in_each_seq = [None] * self.num_cache_seqs
        self.curr_ind_in_each_seq = [0] * self.num_cache_seqs # Keep as list so we can pop it once no more seqs to replenish
        self.cache_to_all_map = [None] * self.num_cache_seqs

        # Pick the initial sequences. We need to keep track of which sequences we already used too.
        self.next_seq_idx_to_be_cached = 0 # index of next seq to grab in avail_seq_inds
        self.avail_seq_inds = np.arange(len(self.data_list), dtype=np.int32)
        if self.split == "train":
            np.random.shuffle(self.avail_seq_inds)
        if self.prefetch:
            self.launch_prefetching()
        for i in range(self.num_cache_seqs):
            self.fill_cache_location_with_next_seq(i)
    
    def fill_cache_location_with_next_seq(self, idx_in_cache):
        # seq_idx is in terms of all seqs in data_list (total amount for this worker)
        seq_idx = self.avail_seq_inds[self.next_seq_idx_to_be_cached] # idx in data_list/data_descriptions
        self.next_seq_idx_to_be_cached += 1
        seq_id = self.data_list[seq_idx] # String name of sequence (directory name)
        desc = self.data_descriptions[seq_idx]
        if self.prefetch:
            # Get seq_data from prefetched
            seq_data = self.prefetch_queue.get()
        else:
            seq_data = {}
            for j, sensor_basename in enumerate(self.sensor_file_basenames):
                filename = os.path.join(self.data_path, seq_id, sensor_basename+".npy")
                sensor_desc = desc[sensor_basename]
                num_cols = sum([
                    int(c.split("(")[1].split(")")[0]) for c in sensor_desc["columns_name(width)"]
                ])
                seq_data[sensor_basename] = np.load(filename) #np.memmap(filename, dtype=np.float64, mode='c',
                #        shape=(sensor_desc["num_rows"],num_cols))[:].copy()

        self.seq_cache[idx_in_cache] = seq_data
        iter_order_inds_in_seq = np.arange(
            desc[self.get_base_sensor_name()]["num_rows"] - self.genparams.window_size,
            step=self.genparams.decimator, dtype=np.int32,
        ) # Ex: decimator=10, inds_in_seq=[0,10,20,...]
        # Be sure to make the order of accessing sequence data random if training.
        if self.split == "train":
            np.random.shuffle(iter_order_inds_in_seq)
        self.iter_order_inds_in_each_seq[idx_in_cache] = iter_order_inds_in_seq
        self.curr_ind_in_each_seq[idx_in_cache] = 0
        self.cache_to_all_map[idx_in_cache] = seq_idx

    def maybe_replenish_cached_seq(self, idx_in_cache):
        # Check if we used all the inds from this sequence
        if self.curr_ind_in_each_seq[idx_in_cache] == len(self.iter_order_inds_in_each_seq[idx_in_cache]):
            if self.next_seq_idx_to_be_cached < len(self.avail_seq_inds):
                self.fill_cache_location_with_next_seq(idx_in_cache)
            else:
                # Otherwise just pop that location off of the cache lists (del so we don't wait for gc to act on it)
                del self.seq_cache[idx_in_cache]
                del self.iter_order_inds_in_each_seq[idx_in_cache]
                del self.cache_to_all_map[idx_in_cache]
                del self.curr_ind_in_each_seq[idx_in_cache]

            if self.split != "train" and len(self.seq_cache) > 0:
                # Iterate in order
                self.curr_idx_in_cache = (1 + self.curr_idx_in_cache) % len(self.seq_cache)
    
    def idx_in_cache_to_seq_idx(self, idx_in_cache):
        return self.cache_to_all_map[idx_in_cache]

    def __len__(self):
        # Purely for pytorchg lightning progress bar logging
        return self.length

    def __iter__(self):
        self.current_epoch += 1
        if self.split != "train":
            # Iterate in order
            self.curr_idx_in_cache = 0
        self.maybe_reinit_and_reset_cache()
        return self

    def __next__(self):
        # We start popping off the seq cache once there are no more seqs to replenish it.
        # If the cache is empty, then we are done.
        if len(self.seq_cache) == 0:
            self.seq_cache = None
            raise StopIteration

        # Pick the seq_idx (in cache) and row in that seq
        if self.split == "train":
            #idx_in_cache = np.random.randint(len(self.seq_cache))

            # Make sure to weight the random choice as if we had a flat index map and had completely random access
            # This prevents longer sequences from hanging around more at the end and causing loss spikes.
            p = np.array([
                len(self.iter_order_inds_in_each_seq[i]) - self.curr_ind_in_each_seq[i] + 1
                for i in range(len(self.seq_cache))
            ]).astype(np.float)
            p /= np.sum(p)
            idx_in_cache = np.random.choice(len(self.seq_cache), p=p)
        else:
            idx_in_cache = self.curr_idx_in_cache

        row_in_seq = self.iter_order_inds_in_each_seq[idx_in_cache][
            self.curr_ind_in_each_seq[idx_in_cache] 
        ] # self.iter_order_inds_in_each_seq is shuffled already if training

        self.curr_ind_in_each_seq[idx_in_cache] += 1
        ret = self.load_and_preprocess_data_chunk(
            idx_in_cache, row_in_seq, len(self.iter_order_inds_in_each_seq[idx_in_cache]),
        )

        # Replenish if this sequence is exhausted
        self.maybe_replenish_cached_seq(idx_in_cache)
        return ret

    def load_data_chunk(self, idx_in_cache, row):
        seq_desc = self.data_descriptions[self.idx_in_cache_to_seq_idx(idx_in_cache)]
        seq_data = self.seq_cache[idx_in_cache]
        return self.data_chunk_from_seq_data(seq_data, seq_desc, row, 
                load_window_same_seq_if_not_in_cache=True) 

    def get_ts_last_imu_us(self, seq_idx=0):
        """
        Get the last timestamp for all IMU windows in this sequence
        """
        if self.seq_cache is None:
            self.reinit_and_reset_cache()
            
        assert len(self.data_list) == 1, "Only implemented for single trajectory loading"
        assert len(self.seq_cache) == 1 and self.seq_cache[0] is not None
        
        data = self.seq_cache[0][self.get_base_sensor_name()]
        ts = data[self.genparams.window_size-1::self.genparams.decimator,0]
        ts = ts[:len(self.iter_order_inds_in_each_seq[0])] # Sometimes off-by-one
        return ts

    def get_gt_traj_center_window_times(self, seq_idx=0):
        """
        Get the GT orientatoin/position (in world frame) at the center 
        time for each IMU window in this sequence.
        Returned as stacked [N,4,4] SE3 matrices
        """
        
        if self.seq_cache is None:
            self.reinit_and_reset_cache()
        
        assert len(self.data_list) == 1, "Only implemented for single trajectory loading"
        assert len(self.seq_cache) == 1 and self.seq_cache[0] is not None

        data = self.seq_cache[0][self.get_base_sensor_name()]
        start_idx = (self.genparams.window_size - 1) // 2
        end = len(data) - self.genparams.window_size//2
        traj = data[start_idx:end:self.genparams.decimator,-10:-3] # [start:stop:step]
        traj = traj[:len(self.iter_order_inds_in_each_seq[0])] # Sometimes off-by-one
        traj_SE3 = np.zeros((len(traj),4,4))
        traj_SE3[:,:3,:3] = Rotation.from_quat(traj[:,:4]).as_matrix()
        traj_SE3[:,:3,3] = traj[:,4:]
        traj_SE3[:,3,3] = 1
        return traj_SE3


def prefetch_worker_fn(
    queue,
    data_path,
    seq_ids_in_load_order,
    sensor_file_basenames,
):
    for i, seq_id in enumerate(seq_ids_in_load_order):
        seq_data = {}
        for j, sensor_basename in enumerate(sensor_file_basenames):
            filename = os.path.join(data_path, seq_id, sensor_basename+".npy")
            seq_data[sensor_basename] = np.load(filename)
        queue.put(seq_data) # Blocks if queue full
