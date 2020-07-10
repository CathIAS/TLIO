import copy
import os
import sys
import unittest

import numpy as np
import numpy.testing as npt
import tracker.scekf as scekf


sys.path.insert(0, os.path.abspath("../.."))


class TestState(unittest.TestCase):
    g = np.array([[0], [0], [-9.81]])

    def get_zero_state(self):
        s = scekf.State()
        s.initialize_state(
            0,
            np.eye(3),
            np.array([[0], [0], [0]]),
            np.array([[0], [0], [0]]),
            np.array([[0], [0], [0]]),
            np.array([[0], [0], [0]]),
        )
        return s

    def test_construct(self):
        scekf.State()

    def test_initialize(self):
        _ = self.get_zero_state()

    def test_applycomputeCorrection_evolving(self):
        s = self.get_zero_state()
        s_init = copy.deepcopy(s)
        dX = np.random.randn(15, 1)
        s.apply_correction(dX)
        dX2 = s_init.compute_correction(s)
        npt.assert_almost_equal(dX, dX2)

    def test_propagation_jacobian(self):
        s = self.get_zero_state()
        dX = np.random.randn(15, 1)
        s.apply_correction(dX)
        s_init = copy.deepcopy(s)

        gyr = np.random.randn(3, 1)
        acc = np.random.randn(3, 1)
        # compute propagation
        R, v, p, A_computed = scekf.propagate_rvt_and_jac(
            s_init.s_R,
            s_init.s_v,
            s_init.s_p,
            s_init.s_bg,
            s_init.s_ba,
            gyr,
            acc,
            self.g,
            1,
        )
        s_npert = copy.deepcopy(s_init)  # non perturbated
        s_npert.s_R = R
        s_npert.s_v = v
        s_npert.s_p = p

        A_numerical = np.zeros_like(A_computed)
        eps = 1e-5
        for i in range(15):
            dXi_input = np.zeros((15, 1))
            dXi_input[i] = eps
            s = copy.deepcopy(s_init)
            s.apply_correction(dXi_input)
            R, v, p, _ = scekf.propagate_rvt_and_jac(
                s.s_R, s.s_v, s.s_p, s.s_bg, s.s_ba, gyr, acc, self.g, 1
            )
            s_pert = copy.deepcopy(s)  # non perturbated
            s_pert.s_R = R
            s_pert.s_v = v
            s_pert.s_p = p

            dXi_output = s_npert.compute_correction(s_pert)
            A_numerical[:, i : i + 1] = dXi_output / eps
        print(np.max(A_computed - A_numerical))
        import matplotlib.pyplot as plt

        plt.subplot(211)
        plt.imshow(A_computed)
        plt.xticks([1, 4, 7, 10, 13], ["R", "v", "p", "bg", "ba"])
        plt.yticks([1, 4, 7, 10, 13], ["R", "v", "p", "bg", "ba"])
        plt.subplot(212)
        plt.imshow(A_numerical)
        plt.xticks([1, 4, 7, 10, 13], ["R", "v", "p", "bg", "ba"])
        plt.yticks([1, 4, 7, 10, 13], ["R", "v", "p", "bg", "ba"])
        plt.figure()
        plt.imshow(np.log10(np.abs(A_numerical - A_computed)))
        plt.xticks([1, 4, 7, 10, 13], ["R", "v", "p", "bg", "ba"])
        plt.yticks([1, 4, 7, 10, 13], ["R", "v", "p", "bg", "ba"])
        plt.show()


if __name__ == "__main__":
    unittest.main()
