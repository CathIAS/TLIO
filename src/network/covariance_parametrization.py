from abc import ABC, abstractmethod

import torch
from liegroups.torch import SO3


class BaseParams(ABC):
    @classmethod
    @abstractmethod
    def vec2Cov(cls, p):
        pass

    @classmethod
    def toMahalanobisDistance(cls, target, mean, pred_cov, clamp_covariance=False):
        """ Generic function that can be used once vec2Cov is implemented
        can be reimplemented if a better way to do it exists with one parametrization
        Args:
            mean [n x 3] : vx, vy, vz
            pred_cov [n x params] : xx, yy, zz
        Returns:
            err [n x 1] : mahalanobis distance square
        """
        cov_matrix = cls.vec2Cov(pred_cov)
        # compute the inverse of covariance matrices
        CovInv = torch.zeros_like(cov_matrix)
        N = target.shape[0]
        for i in range(N):
            u = torch.cholesky(cov_matrix[i, :, :])
            CovInv[i, :, :] = torch.cholesky_inverse(u)

        # compute the error
        err = mean - target
        loss_part1 = torch.einsum("ki,kij,kj->k", err, CovInv, err)
        if clamp_covariance:
            loss_part2 = torch.log(cov_matrix.det().clamp(min=1e-10))
        else:
            loss_part2 = torch.logdet(cov_matrix)

        loss = loss_part1 + loss_part2
        return loss.reshape((N, -1))


class DiagonalParam(BaseParams):
    """
    This is silly to do this that way as everything simplify with diagonal covariance
    But this allows for easy testing.
    """

    covParamNumber = 3

    @classmethod
    def vec2Cov(cls, p):
        """
        Args:
            pred_cov [n x 3] : xx, yy, zz (as log of sigmas)
        Returns:
            cov [n x 3 x 3] : full covariance (actually it is diagonal)
        """
        assert p.shape[1] == cls.covParamNumber
        N = p.shape[0]
        # activate rhos as in https://arxiv.org/pdf/1910.14215.pdf
        covf = torch.zeros((N, 9))
        # on diagonal terms
        covf[:, 0] = torch.exp(2 * p[:, 0])
        covf[:, 4] = torch.exp(2 * p[:, 1])
        covf[:, 8] = torch.exp(2 * p[:, 2])

        return covf.reshape((N, 3, 3))


class PearsonParam(BaseParams):
    """
    From Multivariate uncertainty in Deep Learning
    https://arxiv.org/pdf/1910.14215.pdf

    In this version the covariance matrix is computed with off-diagnoal coefficients :
    Cov_xy = pred_cov[xy] sqrt(Cov_x.Cov_y)

    """

    covParamNumber = 6

    @classmethod
    def vec2Cov(cls, p):
        """
        Args:
            pred_cov [n x 6] : xx, yy, zz, rho_xy, rho_xz, rho_yz
        Returns:
            cov [n x 3 x 3] : full covariance
        """
        assert p.shape[1] == cls.covParamNumber
        N = p.shape[0]
        # activate rhos as in https://arxiv.org/pdf/1910.14215.pdf
        alpha = 0.05
        eps = 1e-3  # "force the Pearson correlation coefficients to not get too close to 1"
        rho_xy = (1 - eps) * torch.tanh(alpha * p[:, 3])
        rho_xz = (1 - eps) * torch.tanh(alpha * p[:, 4])
        rho_yz = (1 - eps) * torch.tanh(alpha * p[:, 5])

        covf = torch.zeros((N, 9))
        # on diagonal terms
        covf[:, 0] = torch.exp(2 * p[:, 0])
        covf[:, 4] = torch.exp(2 * p[:, 1])
        covf[:, 8] = torch.exp(2 * p[:, 2])
        # off diagonal terms
        covf[:, 1] = rho_xy * torch.sqrt(covf[:, 0] * covf[:, 4])  # xy
        covf[:, 2] = rho_xz * torch.sqrt(covf[:, 0] * covf[:, 8])  # xz
        covf[:, 5] = rho_yz * torch.sqrt(covf[:, 4] * covf[:, 8])  # yz
        # symmetry
        covf[:, 3] = covf[:, 1]  # xy
        covf[:, 6] = covf[:, 2]  # xy
        covf[:, 7] = covf[:, 5]  # xy

        return covf.reshape((N, 3, 3))


class DiagRotParam(BaseParams):
    """
    In this version the covariance matrix is computed as :
    Cov_xy = R . diag . R^T

    The three first parameters are for the diag the three last the log of SO3 exponential
    """

    covParamNumber = 6

    @classmethod
    def vec2Cov(cls, p):
        """
        Args:
            pred_cov [n x 3] : xx, yy, zz,
        Returns:
            cov [n x 3 x 3] : full covariance (actually it is diagonal)
        """
        assert p.shape[1] == cls.covParamNumber
        N = p.shape[0]

        # I am not sure if it outpus R or RT wrt to Sophus library
        R = SO3.exp(p[:, 3:6]).mat

        covf = torch.zeros((N, 3, 3))
        # on diagonal terms
        covf[:, 0, 0] = torch.exp(2 * p[:, 0])
        covf[:, 1, 1] = torch.exp(2 * p[:, 1])
        covf[:, 2, 2] = torch.exp(2 * p[:, 2])
        output = torch.einsum("kip,kpl,kjl->kij", R, covf, R)  #  R.diag.R^T

        return output


class SinhParam(BaseParams):
    """
    In this version the covariance matrix is computed with off-diagnoal coefficients :
    Cov_xy = torch.sinh(pred_cov[xy])

    This is surely a bad idea but let's give it a try

    """

    covParamNumber = 6

    @classmethod
    def vec2Cov(cls, p):
        """
        Args:
            pred_cov [n x 6] : xx, yy, zz, xy, xz, yz

        Returns:
            cov [n x 3 x 3] : full covariance
        """
        assert p.shape[1] == cls.covParamNumber
        N = p.shape[0]
        covf = torch.zeros((N, 9))
        # on diagonal terms
        covf[:, 0] = torch.exp(2 * p[:, 0])
        covf[:, 4] = torch.exp(2 * p[:, 1])
        covf[:, 8] = torch.exp(2 * p[:, 2])
        # off diagonal terms
        covf[:, 1] = torch.sinh(p[:, 3])  # xy
        covf[:, 2] = torch.sinh(p[:, 4])  # xz
        covf[:, 5] = torch.sinh(p[:, 5])  # yz
        # symmetry
        covf[:, 3] = covf[:, 1]  # xy
        covf[:, 6] = covf[:, 2]  # xz
        covf[:, 7] = covf[:, 5]  # yz

        return covf.reshape((N, 3, 3))
