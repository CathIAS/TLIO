"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import math
import torch
#from pytorch3d.transforms import so3_exponential_map
from utils.torch_math_utils import so3_exp_map

# TODO augs for mag and barom
class TransformAddNoiseBias:
    def __init__(self, input_sensors, 
            accel_noise_std, gyro_noise_std, accel_bias_range, gyro_bias_range, 
            mag_bias_range, barom_press_bias_range, barom_temp_bias_range):
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_range = gyro_bias_range
        self.accel_bias_range = accel_bias_range
        self.mag_bias_range = mag_bias_range
        self.barom_press_bias_range = barom_press_bias_range
        self.barom_temp_bias_range = barom_temp_bias_range
        self.input_sensors = input_sensors
        
    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]
            
        for sensor, feat in feats_new.items():
            N, _InDim, T = feat.shape
            feat_aug = feat.clone()
            if "imu" in sensor:
                assert feat.shape[1] == 6
                # shift in the accel and gyro bias terms
                feat_aug[:, :3, :] += (
                    (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.gyro_bias_range / 0.5
                )
                feat_aug[:, 3:6, :] += (
                    (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.accel_bias_range / 0.5
                )

                # add gaussian noise
                feat_aug[:, :3, :] += (
                    torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std
                )
                feat_aug[:, 3:6, :] += (
                    torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std
                )

            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] += (
                    (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.mag_bias_range / 0.5
                )
                
                # TODO gaussian noise? 

            elif "barom" in sensor:
                assert feat.shape[1] == 2
                feat_aug[:, 0:1, :] += (
                    (torch.rand(N, 1, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.barom_press_bias_range / 0.5
                )
                feat_aug[:, 1:2, :] += (
                    (torch.rand(N, 1, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.barom_temp_bias_range / 0.5
                )

            else:
                assert False
            
            feats_new[sensor] = feat_aug
        
        sample_new = {k: v for k,v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        return sample_new

class TransformPerturbGravity:
    def __init__(self,  input_sensors, theta_range_deg):
        self.theta_range_deg = theta_range_deg
        self.input_sensors = input_sensors

    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]

        # get rotation vector of random horizontal direction
        angle_rand_rad = (
            torch.rand(sample["ts_us"].shape[0], device=sample["ts_us"].device, dtype=torch.float32) * math.pi * 2
        )
        theta_rand_rad = (
            torch.rand(sample["ts_us"].shape[0], device=sample["ts_us"].device, dtype=torch.float32)
            * math.pi
            * self.theta_range_deg
            / 180.0
        )
        c = torch.cos(angle_rand_rad)
        s = torch.sin(angle_rand_rad)
        zeros = torch.zeros_like(angle_rand_rad)
        vec_rand = torch.stack([c, s, zeros], dim=1)
        rvec = theta_rand_rad[:, None] * vec_rand  # N x 3
        R_mat = so3_exp_map(rvec)  # N x 3 x 3

        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                assert feat.shape[1] == 6
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
            elif "barom" in sensor:
                # Nothing to do here
                pass
            else:
                assert False
            feats_new[sensor] = feat_aug
        
        sample_new = {k: v for k,v in sample.items() if k != "feats"}

        sample_new["feats"] = feats_new
        return sample_new


class TransformInYawPlane:
    """this transform object:
        - rotate imu data in horizontal plane with a random planar rotation
        - rotate the target the same way
    this brings invariance in the data to planar rotation
    this can also prevent the network to learn cues specific to the IMU placement
    """

    def __init__(self, input_sensors, angle_half_range_rad=math.pi):
        """
        Random yaw angles will be in [-angle_half_range, angle_half_range]
        """
        self.input_sensors = input_sensors
        self.angle_half_range_rad = angle_half_range_rad

    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]
        # rotate in the yaw plane
        N = sample["ts_us"].shape[0]
        rand_unif = 2*torch.rand((N), device=sample["ts_us"].device, dtype=torch.float32) - 1 # in [-1,1]
        angle_rad = rand_unif * self.angle_half_range_rad
        c = torch.cos(angle_rad)  # N
        s = torch.sin(angle_rad)  # N
        ones = torch.ones_like(c)  # N
        zeros = torch.zeros_like(s)  # N
        R_newWorld_from_oldWorld_flat = torch.stack(
            (c, -s, zeros, s, c, zeros, zeros, zeros, ones), dim=1
        )  # N x 9
        R_newWorld_from_oldWorld = R_newWorld_from_oldWorld_flat.reshape((N, 3, 3))
        
        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                assert feat.shape[1] == 6
                feat_aug[:, :3, :] = torch.einsum(
                    "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                )
                feat_aug[:, 3:6, :] = torch.einsum(
                    "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, 3:6, :]
                )
            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] = torch.einsum(
                    "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                )
            elif "barom" in sensor:
                # Nothing to do here
                pass
            else:
                assert False

            feats_new[sensor] = feat_aug

        sample_new = {
            k: v for k,v in sample.items() 
            if k.split("second_")[-1].split("_same")[0] not in ["feats","targ_dt_World","vel_World"]
        }
        sample_new["feats"] = feats_new
        # Handle the target data. Only displacement and vel need rotating, not relative rotation (already relative).
        for k in "targ_dt_World", "vel_World":
            sample_new[k] = torch.einsum("nik,ntk->nti", R_newWorld_from_oldWorld, sample[k])

        return sample_new


"""
if __name__ == "__main__":
    # test
    def get_sample():
        acc = torch.tensor([[0], [1.0], [0]]).repeat(repeats=(1, 1, 200))
        gyr = torch.tensor([[0], [1.0], [0]]).repeat((1, 1, 200))
        Rarg = torch.tensor(torch.eye(3)).repeat(1, 1, 1)
        targ = torch.tensor([0, 1.0, 0]).repeat(1, 1)
        print(acc.shape)
        print(gyr.shape)
        print(Rarg)
        print(targ.shape)
        samples = (torch.cat((acc, gyr), dim=1), Rarg, targ)
        return samples

    transform = TransformInYawPlane("imu0")
    feat_aug, Rarg_aug, targ_aug = transform(
        get_sample(), angle_rad=torch.tensor(math.pi / 2)
    )

    print(feat_aug)
    print(Rarg_aug)
    print(targ_aug)
"""
