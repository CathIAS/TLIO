import numpy as np
import torch
from network.covariance_parametrization import DiagonalParam
from utils.logging import logging


class MeasSourceTorchScript:
    """ Loading a torchscript has the advantage that we do not need to reconstruct the original network class to
        load the weights, the network structure is contained into the torchscript file.
    """

    def __init__(self, model_path, force_cpu=False):
        # load trained network model
        if not torch.cuda.is_available() or force_cpu:
            self.device = torch.device("cpu")
            self.net = torch.jit.load(model_path, map_location="cpu")
        else:
            self.device = torch.device("cuda:0")
            self.net = torch.jit.load(model_path)

        self.net.to(self.device)
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
