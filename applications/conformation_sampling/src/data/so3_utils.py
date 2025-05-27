import numpy as np
import torch
from scipy.spatial.transform import Rotation
import scipy.linalg

# hat map from vector space R^3 to Lie algebra so(3)
def hat(v):
    """
    v: [..., 3]
    hat_v: [..., 3, 3]
    """
    hat_v = torch.zeros([*v.shape[:-1], 3, 3])
    hat_v[..., 0, 1], hat_v[..., 0, 2], hat_v[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return hat_v + -hat_v.transpose(-1, -2)

# vee map from Lie algebra so(3) to the vector space R^3
def vee(A):
    assert torch.allclose(A, -A.transpose(-1, -2)), "Input A must be skew symmetric"
    vee_A = torch.stack([-A[..., 1, 2], A[..., 0, 2], -A[..., 0, 1]], dim=-1)
    return vee_A

# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
def Log(R):
    shape = list(R.shape[:-2])
    R_ = R.reshape([-1, 3, 3])
    Log_R_ = rotation_vector_from_matrix(R_)
    return Log_R_.reshape(shape + [3])

# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R): return hat(Log(R))

# Exponential map from so(3) to SO(3), this is the matrix exponential
def exp(A): return torch.linalg.matrix_exp(A)

# Exponential map from R^3 to SO(3)
def Exp(A): return exp(hat(A))

# Angle of rotation SO(3) to R^+, this is the norm in our chosen orthonormal basis
def Omega(R, eps=1e-4):
    # multiplying by (1-epsilon) prevents instability of arccos when provided with -1 or 1 as input.
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1) * (1-eps)
    return torch.arccos((trace-1)/2)

# exponential map from tangent space at R0 to SO(3)
def expmap(R0, tangent):
    skew_sym = torch.einsum('...ij,...ik->...jk', R0, tangent)
    assert torch.allclose(skew_sym, -skew_sym.transpose(-1, -2), atol=1e-4), "R0.T @ tangent must be skew symmetric"
    skew_sym = (skew_sym - torch.transpose(skew_sym, -2, -1))/2.
    exp_skew_sym = exp(skew_sym)
    return torch.einsum('...ij,...jk->...ik', R0, exp_skew_sym)

# Normal sample in tangent space at R0
def tangent_gaussian(R0): return torch.einsum('...ij,...jk->...ik', R0, hat(torch.randn(*R0.shape[:-2], 3)))

# Usual log density of normal distribution in Euclidean space
def normal_log_density(x, mean, var):
    return (-(1/2)*(x-mean)**2 / var - (1/2)*torch.log(2*torch.pi*var)).sum(dim=-1)

# log density of Gaussian in the tangent space
def tangent_gaussian_log_density(R, R_mean, var):
    Log_RmeanT_R = Log(torch.einsum('Nji,Njk->Nik', R_mean, R))
    return normal_log_density(Log_RmeanT_R, torch.zeros_like(Log_RmeanT_R), var)

# sample from uniform distribution on SO(3)
def sample_uniform(N, M=1000):
    omega_grid = np.linspace(0, np.pi, M)
    cdf = np.cumsum(np.pi**-1 * (1-np.cos(omega_grid)), 0)/(M/np.pi)
    omegas = np.interp(np.random.rand(N), cdf, omega_grid)
    axes = np.random.randn(N, 3)
    axes = omegas[..., None]* axes/np.linalg.norm(axes, axis=-1, keepdims=True)
    axes_ = axes.reshape([-1, 3])
    Rs = exp(hat(torch.tensor(axes_)))
    Rs = Rs.reshape([N, 3, 3])
    return Rs



### New Log map adapted from geomstats
def rotation_vector_from_matrix(rot_mat):
    """Convert rotation matrix (in 3D) to rotation vector (axis-angle).

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884

    Get the angle through the trace of the rotation matrix:
    The eigenvalues are:
    :math:`\{1, \cos(angle) + i \sin(angle), \cos(angle) - i \sin(angle)\}`
    so that:
    :math:`trace = 1 + 2 \cos(angle), \{-1 \leq trace \leq 3\}`
    The rotation vector is the vector associated to the skew-symmetric
    matrix
    :math:`S_r = \frac{angle}{(2 * \sin(angle) ) (R - R^T)}`

    For the edge case where the angle is close to pi,
    the rotation vector (up to sign) is derived by using the following
    equality (see the Axis-angle representation on Wikipedia):
    :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
    In nD, the rotation vector stores the :math:`n(n-1)/2` values
    of the skew-symmetric matrix representing the rotation.

    Parameters
    ----------
    rot_mat : array-like, shape=[..., n, n]
        Rotation matrix.

    Returns
    -------
    regularized_rot_vec : array-like, shape=[..., 3]
        Rotation vector.
    """
    angle = Omega(rot_mat)
    assert len(angle.shape)==1, "cannot handle vectorized Log map here"
    n_rot_mats = len(angle)
    rot_mat_transpose = torch.transpose(rot_mat, -2, -1)
    rot_vec_not_pi = vee(rot_mat - rot_mat_transpose)
    mask_0 = torch.isclose(angle, torch.tensor(0.0)).to(angle.dtype)
    mask_pi = torch.isclose(angle, torch.tensor(torch.pi), atol=1e-2).to(angle.dtype)
    mask_else = (1 - mask_0) * (1 - mask_pi)

    numerator = 0.5 * mask_0 + angle * mask_else
    denominator = (
        (1 - angle**2 / 6) * mask_0 + 2 * torch.sin(angle) * mask_else + mask_pi
    )

    rot_vec_not_pi = rot_vec_not_pi * numerator[..., None] / denominator[..., None]

    vector_outer = 0.5 * (torch.eye(3) + rot_mat)
    vector_outer = vector_outer + (torch.maximum(torch.tensor(0.0), vector_outer) - vector_outer)*torch.eye(3)
    squared_diag_comp = torch.diagonal(vector_outer, dim1=-2, dim2=-1)
    diag_comp = torch.sqrt(squared_diag_comp)
    norm_line = torch.linalg.norm(vector_outer, dim=-1)
    max_line_index = torch.argmax(norm_line, dim=-1)
    selected_line = vector_outer[range(n_rot_mats), max_line_index]
    # want
    signs = torch.sign(selected_line)
    rot_vec_pi = angle[..., None] * signs * diag_comp

    rot_vec = rot_vec_not_pi + mask_pi[..., None] * rot_vec_pi
    return regularize(rot_vec)

def regularize(point):
    """Regularize a point to be in accordance with convention.
    In 3D, regularize the norm of the rotation vector,
    to be between 0 and pi, following the axis-angle
    representation's convention.
    If the angle is between pi and 2pi,
    the function computes its complementary in 2pi and
    inverts the direction of the rotation axis.
    Parameters

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884
    ----------
    point : array-like, shape=[...,3]
        Point.
    Returns
    -------
    regularized_point : array-like, shape=[..., 3]
        Regularized point.
    """
    theta = torch.linalg.norm(point, axis=-1)
    k = torch.floor(theta / 2.0 / torch.pi)

    # angle in [0;2pi)
    angle = theta - 2 * k * torch.pi

    # this avoids dividing by 0
    theta_eps = torch.where(torch.isclose(theta, torch.tensor(0.0)), 1.0, theta)

    # angle in [0, pi]
    normalized_angle = torch.where(angle <= torch.pi, angle, 2 * torch.pi - angle)
    norm_ratio = torch.where(torch.isclose(theta, torch.tensor(0.0)), 1.0, normalized_angle / theta_eps)

    # reverse sign if angle was greater than pi
    norm_ratio = torch.where(angle > torch.pi, -norm_ratio, norm_ratio)
    return torch.einsum("...,...i->...i", norm_ratio, point)
