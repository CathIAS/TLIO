import torch

# NOTE taken from pytorch3d due to difficulty putting it in environment.yaml
# https://github.com/facebookresearch/pytorch3d/blob/7978ffd1e4819d24803b01a1147a2c33ad97c142/pytorch3d/transforms/so3.py#L148
def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.
    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.
    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`
    Raises:
        ValueError if `v` is of incorrect shape.
    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

def so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> torch.Tensor:
#) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R #, rot_angles, skews, skews_square


def normalize_angle(theta):
    """
    Normalize angles in [-pi, pi) smoothly
    """
    return torch.arctan2(torch.sin(theta), torch.cos(theta))


def sin_cos_pred_to_yaw(sin_cos, keepdim=False):
    """
    Convert a sin/cos prediction from network to yaw angles 
    (arbitrary batch dims supported).

    input:
        sin_cos: tensor(float) [b0,b1,...,2] where b0,b1,... ar arbitrary batch dimensions.
    output:
        yaw: tensor(float) [b0,b1,...] if keepdim is False, otherwise [b0,b1,...,1]
    """

    # No need to normalize sin/cos since arctan2 takes arbitrary x/y input
    # Normalizing sin/cos from network to be valid has a potential divide-by-zero error.
    # In reality, sin/cos doesn't even have to be in [-1,1] for this, but it can be.
    yaw = torch.arctan2(sin_cos[...,0], sin_cos[...,1])
    if keepdim:
        return yaw[...,None]
    else:
        return yaw


def yaw_to_rot2D(yaw):
    """
    Convert yaw to 2D rotation matrix (i.e., SO(2) matrix Exp map)
    (arbitrary batch dims supported).

    The 2D rotation matrix is defined by 
        
        | cos(yaw) -sin(yaw) |
        | sin(yaw)  cos(yaw) |
    
    as in https://github.com/strasdat/Sophus/blob/master/sophus/so2.hpp

    input:
        yaw: tensor(float) [b0,b1,...] where b0,b1,... ar arbitrary batch dimensions.
    output:
        R: tensor(float) [b0,b1,...,2,2] 2D rotation matrix for yaw angle
    """

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    R = torch.empty(list(yaw.shape) + [2,2], device=yaw.device)
    R[...,0,0], R[...,0,1] = cos_yaw, -sin_yaw
    R[...,1,0], R[...,1,1] = sin_yaw, cos_yaw
    return R


def rot2D_to_yaw(R):
    """
    Convert 2D rotation matrix (i.e., SO(2) matrix Log map) to yaw angle.
    (arbitrary batch dims supported).

    The 2D rotation matrix is defined by 
        
        | cos(yaw) -sin(yaw) |
        | sin(yaw)  cos(yaw) |
    
    as in https://github.com/strasdat/Sophus/blob/master/sophus/so2.hpp

    input:
        R: tensor(float) [b0,b1,...,2,2] 2D rotation matrix for yaw angle
    output:
        yaw: tensor(float) [b0,b1,...] where b0,b1,... ar arbitrary batch dimensions.
    """
    cos_yaw, sin_yaw = R[...,0,0], R[...,0,1]
    return torch.arctan2(sin_yaw, cos_yaw)
