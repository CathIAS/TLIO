# Copyright 2004-present Facebook. All Rights Reserved.

import contextlib
import warnings

import numpy as np

from .from_scipy import compute_q_from_matrix
from .quiet_numba import jit


def get_rotation_from_gravity(acc):
    # take the first accel data to get gravity direction
    ig_w = np.array([0, 0, 1.0]).reshape((3, 1))
    return rot_2vec(acc, ig_w)


def inv_SE3(T):
    Tinv = np.eye(4)
    Tinv[:3,:3] = T[:3,:3].T
    Tinv[:3,3:4] = - T[:3,:3].T @ T[:3,3:4]
    return Tinv


def hat(v):
    v = np.squeeze(v)
    R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return R

def vee(w_x):
    return np.array([w_x[2,1], w_x[0,2], w_x[1,0]])

@jit(nopython=True, parallel=False, cache=True)
def rot_2vec(a, b):
    assert a.shape == (3, 1)
    assert b.shape == (3, 1)

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    a_hat = a / a_n
    b_hat = b / b_n
    omega = np.cross(a_hat.T, b_hat.T).T
    c = 1.0 / (1 + np.dot(a_hat.T, b_hat))
    R_ba = np.eye(3) + hat(omega) + c * hat(omega) @ hat(omega)
    return R_ba


@jit(nopython=True, parallel=False, cache=True)
def mat_exp(omega):
    if len(omega) != 3:
        raise ValueError("tangent vector must have length 3")

    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    angle = np.linalg.norm(omega)

    # Near phi==0, use first order Taylor expansion
    if angle < 1e-10:
        return np.identity(3) + hat(omega)

    axis = omega / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)


mat_exp_vec = np.vectorize(mat_exp, signature="(3)->(3,3)")


def mat_log(R):
    q = compute_q_from_matrix(R)
    w = q[3]
    vec = q[0:3]
    n = np.linalg.norm(vec)
    epsilon = 1e-7

    if n < epsilon:
        w2 = w * w
        n2 = n * n
        atn = 2.0 / w - (2.0 * n2) / (w * w2)
    else:
        if np.absolute(w) < epsilon:
            if w > 0:
                atn = np.pi / n
            else:
                atn = -np.pi / n
        else:
            #atn = 2.0 * np.arctan(n / w) / n
            atn = 2.0 * np.arctan2(n, w) / n
    tangent = atn * vec
    return tangent


def mat_log_vec(R):
    """
    Args:
        R [n x 3 x 3]
    """

    q = compute_q_from_matrix(R)
    w = q[:, 3]
    vec = q[:, 0:3]
    n = np.linalg.norm(vec, axis=1)
    epsilon = 1e-7

    mask = n < epsilon
    atn_small = 2.0 / w - (2.0 * n * n) / (w * w * w)

    mask2 = np.absolute(w) < epsilon
    atn_normal_small = np.sign(w) * np.pi / n
    #atn_normal_normal = 2.0 * np.arctan(n / w) / n
    atn_normal_normal = 2.0 * np.arctan2(n, w) / n

    atn = mask2 * atn_normal_small + (1 - mask2) * atn_normal_normal
    atn = mask * atn_small + (1 - mask2) * atn

    tangent = atn[0, np.newaxis] * vec
    return tangent


def hat_SE3(v):
    """
    Aligns with the Sophus convention of the 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    Ohat = np.zeros((4,4))
    Ohat[:3,:3] = hat(v[3:])
    Ohat[:3,3] = v[:3]
    return Ohat


def exp_SE3(v):
    """
    Aligns with the Sophus convention of the 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    Exp = np.eye(4)
    Exp[:3,:3] = mat_exp(v[3:])
    Exp[:3,3:4] = Jl_SO3(v[3:]) @ v[:3,None]
    return Exp


def log_SE3(T):
    """
    Aligns with the Sophus convention of the returned 6x1 v being
    in the block order: [log(translation) log(rotation)]
    """
    w = mat_log(T[:3,:3])
    V_inv = Jl_SO3_inv(w)
    v = V_inv @ T[:3,3:4]
    return np.concatenate([v[:,0], w], 0)


""" right jacobian for exp operation on SO(3) """


@jit(nopython=True, parallel=False, cache=True)
def Jr_exp(phi):
    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    theta = np.linalg.norm(phi)
    if theta < 1e-3:
        J = np.eye(3) - 0.5 * hat(phi) + 1.0 / 6.0 * (hat(phi) @ hat(phi))
    else:
        J = (
            np.eye(3)
            - (1 - np.cos(theta)) / np.power(theta, 2.0) * hat(phi)
            + (theta - np.sin(theta)) / np.power(theta, 3.0) * (hat(phi) @ hat(phi))
        )
    return J


def Jr_log(phi):
    """ right jacobian for log operation on SO(3) """
    theta = np.linalg.norm(phi)
    if theta < 1e-3:
        J = np.eye(3) + 0.5 * hat(phi)
    else:
        J = (
            np.eye(3)
            + 0.5 * hat(phi)
            + (
                1 / np.power(theta, 2.0)
                + (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
            )
            * hat(phi)
            * hat(phi)
        )
    return J


def Jr_SO3(phi):
    """ right Jacobian of SO(3) (Eq. 7.77a in Barfoot's "State Estimation in Robotics" book) """
    phi_norm = np.linalg.norm(phi)
    if phi_norm < 1e-5:
        return np.eye(3)
    else:
        a = phi / phi_norm
        a = a.reshape(3,1)
        sin_phi_div_phi = np.sin(phi_norm) / phi_norm
        return sin_phi_div_phi * np.eye(3) + (1-sin_phi_div_phi) * a @ a.T + (1-np.cos(phi_norm))/phi_norm * hat(a)


def Jl_SO3(phi):
    """ Left Jacobian of SO(3) """
    theta = np.linalg.norm(phi)
    Om = hat(phi)
    if theta < 1e-5:
        return np.eye(3) + 0.5 * Om
    else:
        theta2 = theta ** 2
        return np.eye(3) + (1-np.cos(theta))/theta2 * Om + (theta-np.sin(theta))/(theta2*theta) * Om @ Om


def Jl_SO3_inv(phi):
    """ Inverse of left Jacobian of SO(3) """

    theta = np.linalg.norm(phi)
    Om = hat(phi)
    if theta < 1e-5:
        return np.eye(3) - 0.5 * Om + 1.0/12 * Om @ Om
    else:
        theta2 = theta ** 2
        return np.eye(3) - 0.5 * Om + (1 - 0.5 * theta * np.cos(theta/2) / np.sin(theta/2)) / theta**2 * Om @ Om


def unwrap_rpy(rpys):
    diff = rpys[1:, :] - rpys[0:-1, :]
    uw_rpys = np.zeros(rpys.shape)
    uw_rpys[0, :] = rpys[0, :]
    diff[diff > 300] = diff[diff > 300] - 360
    diff[diff < -300] = diff[diff < -300] + 360
    uw_rpys[1:, :] = uw_rpys[0, :] + np.cumsum(diff, axis=0)
    return uw_rpys


def wrap_rpy(uw_rpys, radians=False):
    bound = np.pi if radians else 180
    rpys = uw_rpys
    while rpys.min() < -bound:
        rpys[rpys < -bound] = rpys[rpys < -bound] + 2*bound
    while rpys.max() >= bound:
        rpys[rpys >= bound] = rpys[rpys >= bound] - 2*bound
    return rpys
