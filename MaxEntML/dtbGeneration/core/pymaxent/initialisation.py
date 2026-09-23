"""
initialisation.py
==================

PyTorch port of core/optimisation/initialisation.hpp -- batched.

Maxwellian-based initial guesses for the Newton dual solve, plus the
"cascade solver short circuit" that checks whether a plain Maxwellian already
reproduces the target moments to within tolerance.
"""

from __future__ import annotations

import math

import torch

from .config import as_batched, resolve_device, resolve_dtype
from .quadrature import Quadrature, build_monomial_basis
from .conversion import compute_primitive_moments, powers

DEFAULT_HIGHER_MOMENT_LAMBDA = -1.0e-8


# ============================================================================
# Initial guess from point (0, 2) in q*-r* space (used with standardization)
# ============================================================================

def maxwellian_init_q0_r2(n_mom: int, default_lambda: float = DEFAULT_HIGHER_MOMENT_LAMBDA,
                          varsigma: float = 1.0, device=None, dtype=None) -> torch.Tensor:
    """
    Initial guess (shape (n_mom,), shared by every sample) for the
    *standardized* dual system: the reduced Gaussian state (q* = 0, r* = 2).

    `varsigma` re-expresses it in the further-rescaled coordinates
    xi' = xi / varsigma: lambda_k -> varsigma^k * lambda_k.
    """
    assert n_mom >= 5
    dev, dt = resolve_device(device), resolve_dtype(dtype)

    lambda0_q0r2 = -1.24798252795480289223936261438
    lambda2_q0r2 = 0.27840661101339076
    lambda4_q0r2 = -0.19460165275334768

    head = torch.tensor([lambda0_q0r2, 0.0, lambda2_q0r2, 0.0, lambda4_q0r2], device=dev, dtype=dt)
    tail = torch.full((n_mom - 5,), default_lambda, device=dev, dtype=dt)
    lam0 = torch.cat([head, tail])

    if varsigma != 1.0:
        lam0 = lam0 * powers(torch.tensor(varsigma, device=dev, dtype=dt), n_mom)
    return lam0


# ============================================================================
# Maxwellian initial guess (physical / non-standardized moments)
# ============================================================================

def maxwellian_init(m_cell: torch.Tensor, default_lambda: float = DEFAULT_HIGHER_MOMENT_LAMBDA,
                    strict: bool = True) -> torch.Tensor:
    """Closed-form Gaussian dual multipliers, (B, n_mom) -> (B, n_mom)."""
    m = as_batched(m_cell, getattr(m_cell, "device", None), getattr(m_cell, "dtype", None))
    B, K = m.shape

    rho = m[:, 0]
    u = m[:, 1] / rho
    theta = (m[:, 2] - rho * u * u) / rho
    if strict and bool((theta <= 0).any()):
        raise RuntimeError("maxwellian_init: theta must be positive "
                           f"(offending sample {int((theta <= 0).nonzero()[0])}).")

    lam0 = torch.log(rho / torch.sqrt(2.0 * math.pi * theta)) - (0.5 * u * u) / theta
    lam1 = u / theta
    lam2 = -0.5 / theta
    head = torch.stack([lam0, lam1, lam2], dim=-1)
    tail = torch.full((B, K - 3), default_lambda, device=m.device, dtype=m.dtype)
    return torch.cat([head, tail], dim=-1)


# ============================================================================
# Preconditioned / reduced Maxwellian initial guess
# ============================================================================

def maxwellian_reduced_init(m_reduced: torch.Tensor,
                            default_lambda: float = DEFAULT_HIGHER_MOMENT_LAMBDA) -> torch.Tensor:
    """Reduced-frame Maxwellian (u_r = 0, theta_r = 1), (B, n_mom) -> (B, n_mom)."""
    m = as_batched(m_reduced, getattr(m_reduced, "device", None), getattr(m_reduced, "dtype", None))
    B, K = m.shape

    rho_r = m[:, 0]
    lam = torch.zeros(B, K, device=m.device, dtype=m.dtype)
    lam0 = torch.log(rho_r / math.sqrt(2.0 * math.pi))
    mask = torch.arange(K, device=m.device)
    lam = torch.where(mask >= 4, torch.full_like(lam, default_lambda), lam)
    lam = torch.where(mask == 0, lam0.unsqueeze(-1).expand_as(lam), lam)
    lam = torch.where(mask == 2, torch.full_like(lam, -0.5), lam)
    return lam


# ============================================================================
# Maxwellian fitness check (cascade solver short-circuit)
# ============================================================================

@torch.no_grad()
def check_maxwellian_fitness(m_cell: torch.Tensor, quad: Quadrature, entropy_policy,
                             maxwellian_tol: float = 1.0e-6):
    """
    Returns (is_fit (B,) bool, lam0 (B, n_mom)). `quad` may be shared (Q,)
    or per-sample (B, Q).
    """
    m = as_batched(m_cell, getattr(m_cell, "device", None), getattr(m_cell, "dtype", None))
    K = m.shape[-1]

    lam0 = maxwellian_init(m)

    Phi = build_monomial_basis(quad.nodes, K)                       # (K, Q) or (B, K, Q)
    s = torch.einsum("...kq,...k->...q", Phi, lam0)
    fp = entropy_policy.eta_dual_prime(s)
    m_maxwellian = torch.einsum("...kq,...q->...k", Phi, quad.weights * fp)

    prim_target = compute_primitive_moments(m)
    prim_maxwellian = compute_primitive_moments(m_maxwellian, strict=False)

    l2_diff = torch.sqrt(((prim_target - prim_maxwellian) ** 2).sum(dim=-1))
    return l2_diff < maxwellian_tol, lam0
