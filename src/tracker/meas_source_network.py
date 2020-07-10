import numpy as np
import torch
from network.covariance_parametrization import DiagonalParam
from network.model_factory import get_model
from utils.logging import logging


class MeasSourceNetwork:
    def __init__(self, model_path, arch, net_config, force_cpu=False):
        # network
        self.net = get_model(arch, net_config, 6, 3)

        # load trained network model
        if not torch.cuda.is_available() or force_cpu:
            self.device = torch.device("cpu")
            checkpoint = torch.load(
                model_path, map_location=lambda storage, location: storage
            )
        else:
            self.device = torch.device("cuda:0")
            checkpoint = torch.load(model_path)

        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval().to(self.device)
        logging.info("Model {} loaded to device {}.".format(model_path, self.device))

    def get_displacement_measurement(self, net_gyr_w, net_acc_w):
        features = np.concatenate([net_gyr_w, net_acc_w], axis=1)  # N x 6
        features_t = torch.unsqueeze(
            torch.from_numpy(features.T).float().to(self.device), 0
        )  # 1 x 6 x N
        meas, meas_cov = self.net(features_t)
        meas = meas.cpu().detach().numpy()
        meas_cov[meas_cov < -4] = -4  # exp(-3) =~ 0.05
        meas_cov = DiagonalParam.vec2Cov(meas_cov).cpu().detach().numpy()[0, :, :]

        meas = meas.reshape((3, 1))
        return meas, meas_cov
