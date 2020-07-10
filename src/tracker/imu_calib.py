import os.path as osp

import numpy as np


class ImuCalib:
    def __init__(self):
        self.accelScaleInv = np.eye(3)
        self.gyroScaleInv = np.eye(3)
        self.gyroGSense = np.eye(3)
        self.accelBias = np.zeros((3, 1))
        self.gyroBias = np.zeros((3, 1))

    @classmethod
    def from_attitude_file(cls, dataset, args):
        ret = cls()
        attitude_filter_path = osp.join(args.root_dir, dataset, "atttitude.txt")
        with open(attitude_filter_path, "r") as f:
            line = f.readline()
            line = f.readline()
        init_calib = np.fromstring(line, sep=",")
        ret.accelScaleInv = init_calib[1:10].reshape((3, 3))
        ret.gyroScaleInv = init_calib[10:19].reshape((3, 3))
        ret.gyroGSense = init_calib[19:28].reshape((3, 3))
        ret.accelBias = init_calib[28:31].reshape((3, 1))
        ret.gyroBias = init_calib[31:34].reshape((3, 1))
        return ret

    def calibrate_raw(self, acc, gyr):
        acc_cal = np.dot(self.accelScaleInv, acc) - self.accelBias
        gyr_cal = (
            np.dot(self.gyroScaleInv, gyr)
            - np.dot(self.gyroGSense, acc)
            - self.gyroBias
        )
        return acc_cal, gyr_cal

    def scale_raw(self, acc, gyr):
        acc_cal = np.dot(self.accelScaleInv, acc)
        gyr_cal = np.dot(self.gyroScaleInv, gyr) - np.dot(self.gyroGSense, acc)
        return acc_cal, gyr_cal
