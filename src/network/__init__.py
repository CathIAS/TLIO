"""
Three main functionality in network module
net_train: training a network
net_test: testing a network to find metrics of the concatenated trajectories
net_eval: testing a network to gather sample-level inference data
"""
from .covariance_parametrization import DiagonalParam
from .eval import net_eval
from .model_factory import get_model
from .test import net_test
from .train import net_train
