"""
realisability.py
=================

PyTorch port of moments_op/realisability.hpp -- batched.

Checks whether moment sets lie in the interior of the realizable moment
space (Hankel positive-definiteness) and, for the 5-moment case, whether they
sit on the "Junk line" (symmetric, super-Gaussian states with no admissible
MaxEnt solution).

All checks take primitive/central moments of shape (..., n_mom) and return
boolean tensors of shape (...,). They are discrete decisions, so they run
under ``torch.no_grad()`` (they are never part of a gradient path).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import as_tensor
from .conversion import compute_primitive_moments


@dataclass
class RealizabilityResult:
    is_realizable: torch.Tensor = None      # (...,) bool
    is_interior: torch.Tensor = None
    is_junk_line: torch.Tensor = None
    message: str = ""


def _eps(x: torch.Tensor) -> float:
    return float(torch.finfo(x.dtype).eps)


# =============================================================================
# 1. Hankel interior positivity (n_mom = 5)
# =============================================================================

@torch.no_grad()
def check_hankel_interior_5mom(c_mom: torch.Tensor, slack_factor: float = 10.0) -> torch.Tensor:
    eps = _eps(c_mom)
    rho, p, q, r = c_mom[..., 0], c_mom[..., 2], c_mom[..., 3], c_mom[..., 4]

    rho_tol = slack_factor * eps * torch.clamp(rho.abs(), min=1.0)
    p_tol = slack_factor * eps * torch.clamp(p.abs(), min=1.0)
    sign_ok = ~((rho <= -rho_tol) | (p <= -p_tol))

    # Dirac delta / cold gas limit
    cold = p <= 1e-14
    cold_ok = (q.abs() <= 1e-14) & (r.abs() <= 1e-14)

    p_safe = torch.where(cold, torch.ones_like(p), p)
    S = q / p_safe ** 1.5
    K = r / (p_safe * p_safe)
    delta = K - 1.0 - S * S
    delta_tol = slack_factor * eps * (1.0 + K.abs() + S * S) + 1e-15
    hankel_ok = delta >= -delta_tol

    return sign_ok & torch.where(cold, cold_ok, hankel_ok)


# =============================================================================
# 2. Hankel interior positivity (generic n_mom, LDL^T)
# =============================================================================

@torch.no_grad()
def check_hankel_interior_generic(c_mom: torch.Tensor, slack_factor: float = 10.0) -> torch.Tensor:
    n_mom = c_mom.shape[-1]
    batch = c_mom.shape[:-1]
    dev, dt = c_mom.device, c_mom.dtype
    eps = _eps(c_mom)

    if n_mom < 1:
        return torch.zeros(batch, dtype=torch.bool, device=dev)

    rho = c_mom[..., 0]
    rho_ok = rho > -(slack_factor * eps * torch.clamp(rho.abs(), min=1.0))
    if n_mom < 3:
        return rho_ok

    p = c_mom[..., 2]
    p_ok = p > -(slack_factor * eps * torch.clamp(p.abs(), min=1.0))

    cold = p <= 1e-14
    cold_ok = (c_mom[..., 3:].abs() <= 1e-14).all(dim=-1)

    M = (n_mom + 1) // 2
    p_safe = torch.where(cold, torch.ones_like(p), p)

    kk = torch.arange(2 * M - 1, device=dev, dtype=dt)
    C_hat = c_mom[..., : 2 * M - 1] / p_safe.unsqueeze(-1) ** (kk / 2.0)
    C_hat = C_hat.clone()
    C_hat[..., 0] = 1.0
    C_hat[..., 1] = 0.0

    D = torch.zeros(*batch, M, device=dev, dtype=dt)
    L = torch.zeros(*batch, M, M, device=dev, dtype=dt)
    ok = torch.ones(batch, dtype=torch.bool, device=dev)
    tol = slack_factor * eps

    for j in range(M):
        sum_d = (L[..., j, :j] ** 2 * D[..., :j]).sum(dim=-1)
        Dj = C_hat[..., 2 * j] - sum_d

        pivot_tol = tol * (1.0 + C_hat[..., 2 * j].abs()) + 1e-15
        ok = ok & ~(Dj < -pivot_tol)

        D[..., j] = Dj
        eff_D = torch.clamp(Dj, min=1e-15)
        L[..., j, j] = 1.0

        if j + 1 < M:
            sum_l = (L[..., j + 1:, :j] * L[..., j:j + 1, :j] * D[..., None, :j]).sum(dim=-1)
            L[..., j + 1:, j] = (C_hat[..., 2 * j + 1: M + j] - sum_l) / eff_D.unsqueeze(-1)

    return rho_ok & p_ok & torch.where(cold, cold_ok, ok)


# =============================================================================
# 3. Auto-dispatched Hankel check
# =============================================================================

@torch.no_grad()
def check_hankel_realizability(c_mom: torch.Tensor, slack_factor: float = 10.0) -> torch.Tensor:
    if c_mom.shape[-1] == 5:
        return check_hankel_interior_5mom(c_mom, slack_factor)
    return check_hankel_interior_generic(c_mom, slack_factor)


# =============================================================================
# 4. Junk line (n_mom = 5)
# =============================================================================

@torch.no_grad()
def check_junk_line_5mom(c_mom: torch.Tensor, slack_factor: float = 10.0) -> torch.Tensor:
    if c_mom.shape[-1] < 5:
        return torch.zeros(c_mom.shape[:-1], dtype=torch.bool, device=c_mom.device)

    eps = _eps(c_mom)
    p, q, r = c_mom[..., 2], c_mom[..., 3], c_mom[..., 4]

    cold = p <= 1e-14
    p_safe = torch.where(cold, torch.ones_like(p), p)
    S = q / p_safe ** 1.5
    K = r / (p_safe * p_safe)

    q_tol = slack_factor * eps
    mu_tol = slack_factor * eps * (1.0 + K.abs())

    flag_q_null = S.abs() <= q_tol
    flag_mu_positive = (K - 3.0) > mu_tol
    return ~cold & flag_q_null & flag_mu_positive


# =============================================================================
# 5. Unified checker
# =============================================================================

@torch.no_grad()
def check_realizability(mom_in: torch.Tensor,
                        is_primitive: bool = True,
                        slack_factor: float = 10.0,
                        throw_on_error: bool = False) -> RealizabilityResult:
    """
    Batched realizability check. Returns a `RealizabilityResult` whose boolean
    fields have shape (...,). If `throw_on_error` and *any* sample is not
    realizable, raises RuntimeError describing the first offender.
    """
    mom = as_tensor(mom_in, mom_in.device if isinstance(mom_in, torch.Tensor) else None,
                    mom_in.dtype if isinstance(mom_in, torch.Tensor) else None)
    c_mom = mom if is_primitive else compute_primitive_moments(mom)
    n_mom = c_mom.shape[-1]

    res = RealizabilityResult()
    res.is_interior = check_hankel_realizability(c_mom, slack_factor)
    if n_mom == 5:
        res.is_junk_line = check_junk_line_5mom(c_mom, slack_factor)
    else:
        res.is_junk_line = torch.zeros_like(res.is_interior)
    res.is_realizable = res.is_interior & ~res.is_junk_line

    if not bool(res.is_realizable.all()):
        flat = (~res.is_realizable).flatten().nonzero().flatten()
        first = int(flat[0])
        n_bad = int(flat.numel())
        cm = c_mom.reshape(-1, n_mom)[first]
        interior = bool(res.is_interior.flatten()[first])

        lines = [f"[{n_bad} non-realizable sample(s); showing sample index {first}]"]
        if not interior:
            lines.append("Realizability ERROR: Moment vector C is outside realizable moment space.")
            lines.append(f"  N_mom = {n_mom}")
            lines.append(f"  rho   = {float(cm[0])}")
            if n_mom >= 3:
                lines.append(f"  p     = {float(cm[2])}")
            if n_mom == 5:
                p, q, r = float(cm[2]), float(cm[3]), float(cm[4])
                if p > 1e-14:
                    q_star = q / p ** 1.5
                    r_star = r / (p * p)
                    lines.append(f"  q     = {q} (q* = {q_star})")
                    lines.append(f"  r     = {r} (r* = {r_star})")
                    lines.append(f"  Delta = r* - 1 - q*^2 = {r_star - 1.0 - q_star * q_star} (must be > 0)")
        else:
            lines.append("Bi-Gaussian / ME ERROR: Moment vector C lies on the Junk Line.")
            lines.append("  Symmetric state (q ~ 0) with super-Gaussian kurtosis (r - 3*p^2 > 0).")
            lines.append(f"  p  = {float(cm[2])}")
            lines.append(f"  q  = {float(cm[3])}")
            lines.append(f"  r  = {float(cm[4])}")
            lines.append(f"  mu = r - 3*p^2 = {float(cm[4]) - 3.0 * float(cm[2]) ** 2} > 0")

        res.message = "\n".join(lines)
        if throw_on_error:
            raise RuntimeError(res.message)

    return res
