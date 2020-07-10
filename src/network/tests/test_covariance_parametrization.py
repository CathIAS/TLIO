import os
import sys
import unittest

import numpy as np
import numpy.testing as nt
import torch
import torch.testing as tt
from network.covariance_parametrization import *


class DiagonalBaseClass:
    def test_eye_cov(self):
        cov_pred = torch.zeros((1, self.covParamNumber))
        cov = self.vec2Cov(cov_pred)
        tt.assert_allclose(cov, torch.eye(3))

    def test_eye_cov_no_error(self):
        target = torch.FloatTensor([[0, 0, 0]])
        mean = torch.FloatTensor([[0, 0, 0]])
        cov_pred = torch.zeros((1, self.covParamNumber))

        err = self.toMahalanobisDistance(target, mean, cov_pred)
        nt.assert_equal(err[0], torch.FloatTensor([[0]]))

    def test_eye_cov_ones_error(self):
        target = torch.FloatTensor([[0, 0, 0]])
        mean = torch.FloatTensor([[1, 1, 1]])
        cov_pred = torch.zeros((1, self.covParamNumber))

        err = self.toMahalanobisDistance(target, mean, cov_pred)
        nt.assert_equal(err, torch.FloatTensor([[3]]))

    def test_eye_cov_ones_error_bis(self):
        target = torch.FloatTensor([[1, 1, 1]])
        mean = torch.FloatTensor([[0, 0, 0]])
        cov_pred = torch.zeros((1, self.covParamNumber))

        err = self.toMahalanobisDistance(target, mean, cov_pred)
        nt.assert_equal(err, torch.FloatTensor([[3]]))

    def test_2_cov_no_error(self):
        target = torch.FloatTensor([[0, 0, 0]])
        mean = torch.FloatTensor([[0, 0, 0]])
        cov_pred = torch.zeros((1, self.covParamNumber))

        err = self.toMahalanobisDistance(target, mean, cov_pred)
        nt.assert_equal(err, torch.FloatTensor([[0]]))

    def test_1_cov_no_error(self):
        target = torch.FloatTensor([[1, 1, 1]])
        mean = torch.FloatTensor([[0, 0, 0]])
        cov_pred = torch.zeros((1, self.covParamNumber))
        cov_pred[0, 0:3] = 1

        cov = self.vec2Cov(cov_pred)
        tt.assert_allclose(cov, np.exp(2) * torch.eye(3))

        expected_err = 3.0 / np.exp(2) + 3 * 2
        err = self.toMahalanobisDistance(target, mean, cov_pred)
        nt.assert_equal(err, torch.FloatTensor([[expected_err]]))


class DiagonalParamTest(unittest.TestCase, DiagonalBaseClass, DiagonalParam):
    """ Nothing to add here as test"""


class TestPearsonParamDiagonal(unittest.TestCase, DiagonalBaseClass, PearsonParam):
    """ Add test with non diagonal entries"""

    def test_nondiag_coef(self):
        cov_pred = torch.ones((1, self.covParamNumber))
        cov_pred[0, 3:6] = 1e9
        cov = self.vec2Cov(cov_pred)
        tt.assert_allclose(cov, np.exp(2), atol=1e-3, rtol=1e-3)


# class TestDiagRotParamDiagonal(unittest.TestCase, DiagonalBaseClass, DiagRotParam):
#     def test_nondiag_coef_2pi_angle_x(self):
#         cov_pred = torch.ones((1, self.covParamNumber))
#         cov_pred[0, 3:6] = 1e9
#         cov = self.vec2Cov(cov_pred)


if __name__ == "__main__":
    unittest.main()
