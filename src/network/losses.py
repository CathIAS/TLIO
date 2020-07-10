import torch
from network.covariance_parametrization import DiagonalParam


"""
MSE loss between prediction and target, no covariance

input: 
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
output:
  loss: Nx3 vector of MSE loss on x,y,z
"""


def loss_mse(pred, targ):
    loss = (pred - targ).pow(2)
    return loss


"""
Log Likelihood loss, with covariance (only support diag cov)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_cov: Nx3 vector of log(sigma) on the diagonal entries
output:
  loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_cov meaning:
pred_cov:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""


def loss_distribution_diag(pred, pred_cov, targ):
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_cov)) + pred_cov
    return loss


"""
Log Likelihood loss, with covariance (support full cov)
(NOTE: output is Nx1)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_cov: Nxk covariance parametrization
output:
  loss: Nx1 vector of likelihood loss

resulting pred_cov meaning:
DiagonalParam:
pred_cov:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
PearsonParam:
pred_cov (Nx6): u = [log(sigma_x) log(sigma_y) log(sigma_z)
                     rho_xy, rho_xz, rho_yz] (Pearson correlation coeff)
FunStuff
"""


def criterion_distribution(pred, pred_cov, targ):
    loss = DiagonalParam.toMahalanobisDistance(
        targ, pred, pred_cov, clamp_covariance=False
    )


"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""


def get_loss(pred, pred_cov, targ, epoch):
    if epoch < 10:
        loss = loss_mse(pred, targ)
    # elif epoch < 50:
    #    loss = 0.5 * loss_mse(pred, targ) + 0.5 * loss_distribution_diag(
    #        pred, pred_cov, targ
    #    )
    else:
        loss = loss_distribution_diag(pred, pred_cov, targ)
    return loss
