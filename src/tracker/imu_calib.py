import os.path as osp

import json
import numpy as np
from utils.logging import logging
from scipy.spatial.transform import Rotation


class ImuCalib:
    def __init__(self):
        self.accelScaleInv = np.eye(3)
        self.gyroScaleInv = np.eye(3)
        self.gyroGSense = np.zeros((3, 3))
        self.accelBias = np.zeros((3, 1))
        self.gyroBias = np.zeros((3, 1))
        self.T_Device_Imu = np.eye(4)

    def to_dict(self):
        return {
            k: np.squeeze(getattr(self, k)).tolist() for k in (
                "accelScaleInv", "gyroScaleInv", "gyroGSense", "accelBias", "gyroBias", "T_Device_Imu",
            )
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)
    
    def to_file(self, file_name):
        with open(file_name, 'w') as f:
            f.write(str(self))

    @classmethod
    def from_file(cls, file_name):
        ret = cls()
        with open(file_name, 'r') as f:
            d = json.load(f)
        for k, v in d.items():
            setattr(ret, k, np.array(v))
        return ret

    @classmethod
    def from_attitude_file(cls, dataset, args):
        raise NotImplementedError("This function is out of sync with the current codebase")
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

    @classmethod
    def from_unified_calib_dict_deprecated(cls, calib_dict):
        ret = cls()
        ret.accelScaleInv = np.linalg.inv(np.array(calib_dict["accel_scale"]).reshape((3, 3)))
        ret.gyroScaleInv = np.linalg.inv(np.array(calib_dict["gyro_scale"]).reshape((3, 3)))
        ret.gyroGSense = np.zeros((3, 3))
        ret.accelBias = np.array(calib_dict["accel_bias"]).reshape((3, 1))
        ret.gyroBias = np.array(calib_dict["gyro_bias"]).reshape((3, 1))
        q_Device_Imu = np.array(calib_dict["T_ir"]['QuaternionXYZW'])
        ret.T_Device_Imu[:3,:3] = Rotation.from_quat(q_Device_Imu).as_matrix()
        ret.T_Device_Imu[:3,3] = calib_dict["T_ir"]["Translation"]
        # NOTE we have to invert this legacy calib, as it's originally T_Imu_Device
        ret.T_Device_Imu = np.linalg.inv(ret.T_Device_Imu)
        return ret

    @classmethod
    def from_calib_json(cls, calib_json):
        ret = cls()
        ret.accelScaleInv = np.linalg.inv(np.array(calib_json["Accelerometer"]["Model"]["RectificationMatrix"]))
        ret.gyroScaleInv = np.linalg.inv(np.array(calib_json["Gyroscope"]["Model"]["RectificationMatrix"]))
        ret.gyroGSense = np.zeros((3, 3))
        ret.accelBias = np.array(calib_json["Accelerometer"]["Bias"]["Offset"]).reshape((3, 1))
        ret.gyroBias = np.array(calib_json["Gyroscope"]["Bias"]["Offset"]).reshape((3, 1))
        q_Device_Imu = calib_json["T_Device_Imu"]["UnitQuaternion"]
        # scipy expects xyzw quat
        q_Device_Imu = np.array([*q_Device_Imu[1], q_Device_Imu[0]])
        ret.T_Device_Imu[:3,:3] = Rotation.from_quat(q_Device_Imu).as_matrix()
        ret.T_Device_Imu[:3,3] = calib_json["T_Device_Imu"]["Translation"]
        return ret

    @classmethod
    def from_calib_struct_object(cls, obj):
        ret = cls()
        ret.accelScaleInv = np.linalg.inv(obj.accel.get_rectification_matrix()).reshape(
            (3, 3)
        )
        ret.gyroScaleInv = np.linalg.inv(obj.gyro.get_rectification_matrix()).reshape(
            (3, 3)
        )
        ret.gyroGSense = np.zeros((3, 3))
        ret.accelBias = obj.accel.get_constant_bias_vec_or_fail().reshape((3, 1))
        ret.gyroBias = obj.gyro.get_constant_bias_vec().reshape((3, 1))
        ret.T_Device_Imu = obj.T_Device_Imu
        return ret

    @classmethod
    def from_offline_calib(cls, dataset, args):
        ret = cls()
        logging.info(
            "loading offline calib from "
            + osp.join(args.root_dir, dataset, "calibration.json")
        )
        with open(osp.join(args.root_dir, dataset, "calibration.json"), 'r') as f:
            calib_json = json.load(f)
            
        ret.accelBias = np.array(calib_json["Accelerometer"]["Bias"]["Offset"])[:,None]
        ret.gyroBias = np.array(calib_json["Gyroscope"]["Bias"]["Offset"])[:,None]
        ret.accelScaleInv = np.linalg.inv(np.array(
            calib_json["Accelerometer"]["Model"]["RectificationMatrix"]
        ))
        ret.gyroScaleInv = np.linalg.inv(np.array(
            calib_json["Gyroscope"]["Model"]["RectificationMatrix"]
        ))
        return ret

    def calibrate_raw(self, acc, gyr):
        assert len(acc.shape) == 2
        N = acc.shape[1]
        assert acc.shape == (3, N)
        assert gyr.shape == (3, N)
        acc_cal = self.accelScaleInv @ acc - self.accelBias
        gyr_cal = (
            self.gyroScaleInv @ gyr
            - self.gyroGSense @ acc
            - self.gyroBias
        )
        assert acc_cal.shape == (3, N)
        assert gyr_cal.shape == (3, N)

        return acc_cal, gyr_cal

    def scale_raw(self, acc, gyr):
        acc_cal = np.dot(self.accelScaleInv, acc)
        gyr_cal = np.dot(self.gyroScaleInv, gyr) - np.dot(self.gyroGSense, acc)
        return acc_cal, gyr_cal
