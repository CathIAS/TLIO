"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

"""
This file includes the main libraries in the network training module.
"""

import os
import time
from itertools import repeat

#from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


from utils.logging import logging
from .constants import DatasetGenerationParams
from .memmapped_sequences_dataset import MemMappedSequencesDataset
from .iterable_pseudorandom_sequences_dataset import IterablePseudoRandomSequencesDataset
from .data_transform import TransformAddNoiseBias, TransformPerturbGravity, TransformInYawPlane

log = logging.getLogger(__name__)

class TlioData:
    def __init__(
        self,
        data_path,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        only_n_sequence=-1,
        task_subset=None,
        ignore_tasks=None,
        decimator=10,
        dataset_style="mmap", # "mmap", "ram", or "iter". "iter" is best for huge datasets but sacrifice true randomness, mmap can go a bit farther than "ram" which just stores all in memory
        data_window_config={
            "window_size": 200, # 200 window size @200 Hz for 1sec of input data
            "step_period_us": 5000, # NOTE: unused at this point
            "data_in_t0_yaw_normalized_frame": False,
            "input_sensors": ["imu0"],
            "data_style": "resampled",
        },
        augmentation_options={   
            "do_bias_shift": True,
            "bias_shift_options": {
                "accel_bias_range": 0.2,
                "gyro_bias_range": 0.05,
                "accel_noise_std": 0,
                "gyro_noise_std": 0,
                "mag_bias_range": 0.05, # In Gauss (.25-.65 Gauss is normal on earth)
                "barom_press_bias_range": 0.01, # In Pascals (always near 1.0)
                "barom_temp_bias_range": 1, # In deg celcius
            },
            "perturb_gravity": True,
            "perturb_gravity_theta_range": 5.0,
        },
    ):
        super().__init__()

        self.batch_size = batch_size
        self.data_path = data_path
        self.data_window_config = data_window_config
        self.augmentation_options = augmentation_options
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.only_n_sequence = only_n_sequence
        if task_subset == []:
            task_subset = None
        if ignore_tasks == []:
            ignore_tasks = None
        self.task_subset = task_subset
        self.ignore_tasks = ignore_tasks
        self.decimator = decimator

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.transform_done_in_dataloader = False
        self.dataset_style = dataset_style

    #def setup(self, stage=None):

    def prepare_data(self, testing=False):
        def setup_split(split):
            start_t = time.time()
            log.warning(
                f"{split}_dataloader : data_window_config is partially ignored here for now! "
                "(past and future data should be 0 for now)"
            )
            starting_point_time_us = 0  # TODO(dcaruso) positive if past imu data here
            prediction_times_us = 0  # TODO(dcaruso) negative if future imu data here
            genparams = DatasetGenerationParams(
                window_size=self.data_window_config["window_size"],
                step_period_us=self.data_window_config["step_period_us"],
                prediction_times_us=[prediction_times_us],
                starting_point_time_us=starting_point_time_us,
                generate_data_period_us=self.data_window_config["step_period_us"],
                decimator=self.decimator,
                express_in_t0_yaw_normalized_frame=self.data_window_config[
                    "data_in_t0_yaw_normalized_frame"
                ],
                input_sensors=self.data_window_config["input_sensors"],
                data_style=self.data_window_config["data_style"],
            )
            
            if self.dataset_style == "mmap":
                SequencesDataset = MemMappedSequencesDataset
            elif self.dataset_style == "ram":
                SequencesDataset = lambda *args, **kwargs: MemMappedSequencesDataset(*args, **kwargs, store_in_ram=True)
            elif self.dataset_style == "iter":
                SequencesDataset = IterablePseudoRandomSequencesDataset
            else:
                raise ValueError(f"Unknown dataset_style \"{self.dataset_style}\"")

            dataset = SequencesDataset(
                self.data_path,
                split,
                genparams,
                only_n_sequence=self.only_n_sequence,
                verbose=True,
            )   
            
            setattr(self, f"{split}_dataset", dataset)
            end_t = time.time()
            log.info(f"{split} set loaded. Loading time: {end_t - start_t:.3f}s")
            #log.info(f"Number of {split} samples: {len(dataset)}")
        
        if testing:
            setup_split("test")
        else:
            for split in ["val", "train"]:
                setup_split(split)


    def train_dataloader(self):
        """
        # Make train and val the same if doing quick dev run
        if self.only_n_sequence > 0:
            log.warning(
                f"\nSwapping train dataset for val dataset for fast dev run "
                f"with sequences {list(self.val_dataset.data_list)}\n"
            )
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=True,
            )
        else:
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle="iter" not in self.dataset_style,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        # If no test split was set (e.g., dev dataset), just return val split
        if len(self.test_dataset) > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                pin_memory=True,
            )
        else:
            log.warning("Test set has no data. Returning validation set for testing")
            return self.val_dataloader()

    def get_train_transforms(self):
        transforms = []
        if self.augmentation_options["do_bias_shift"]:
            transforms.append(
                TransformAddNoiseBias(self.data_window_config["input_sensors"],
                    **self.augmentation_options["bias_shift_options"])
            )

        if self.augmentation_options["perturb_gravity"]:
            transforms.append(
                TransformPerturbGravity(self.data_window_config["input_sensors"], 
                    self.augmentation_options["perturb_gravity_theta_range"])
            )

        transforms.append(TransformInYawPlane(self.data_window_config["input_sensors"]))
        return transforms

    def get_datalist(self, split="val"):
        dataset = getattr(self, f"{split}_dataset")
        assert dataset is not None, f"Tried to get {split} list but {split}_dataset is None"
        return dataset.data_list

