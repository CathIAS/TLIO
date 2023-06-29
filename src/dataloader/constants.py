"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import namedtuple

# TODO can we simplify this to dict or something to show defaults.
""" 
This type contains all the parameters needed for
data generation from a VIO sequence.

Note the data_style arg controls which type of data for multiple sensor input is loaded.
There are three options:

    1) data_style="aligned" loads sensors aligned (interpolated) to the 
       base sensor timestamps (typically IMU0), which may cause interpolation artifacts.
       However, this is the simplest input since the input is just a big matrix.

    2) data_style="resampled" loads sensors resampled (interpolated) to a constant
       frequency, which should be close to its actual frequency, or can be subsampled,
       like subsampling 1000Hz IMU to 200Hz. Still can cause interpolation artifacts,
       but is simpler to deal with than the actual raw data.

    3) data_style="raw" loads just the data at the original timestamps with no interpolation.
"""
DatasetGenerationParams = namedtuple(
    "DatasetGenerationParams",
    [
        # window size is counted in number imu samples
        "window_size",
        # step_period_us is the time between two imu samples
        "step_period_us",
        # for generate_data_period_us,
        # 1000, means one generate a pair of data every 1000 ms
        "generate_data_period_us",
        # prediction_times_us is the time after the last sample
        # that defines the ending point of relative pose output
        # can be either positive either negative
        "prediction_times_us",
        # starting_point_time_us is the time after the first sample
        # that defines the starting point of relative pose ourput
        # should be positive
        "starting_point_time_us",
        # decimate the dataset, by default one prediction for each frameset
        "decimator",
        # if true, this will express the displacement and delta Rotation
        # in a frame whose yaw angle is 0 instead of the world frame.
        # this allows to express everything in a imu centered frame
        "express_in_t0_yaw_normalized_frame",
        # A list of input sensors out of ["imu0", "imu1", "mag0", "barom0"]
        "input_sensors",
        # See comment above
        "data_style",
    ],
)

COMBINED_SENSOR_NAME = "imu0_imu1_mag0_barom0_aligned_with_resampled_imu0"
ALL_SENSORS_LIST = ["imu0", "imu1", "mag0", "barom0"]
