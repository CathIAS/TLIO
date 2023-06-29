import numpy as np

def align_inertial_frames(T_est, T_gt):
    """
    Align two inertial frames using a single example, aligning by position and yaw
    """

    C = T_est[:3,:3] @ T_gt[:3,:3].T
    yaw = np.arctan2(C[0,1] - C[1,0], C[0,0] + C[1,1])
    R = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    p = T_gt[:3,3:4] - R @ T_est[:3,3:4]
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3:4] = p
    return T
