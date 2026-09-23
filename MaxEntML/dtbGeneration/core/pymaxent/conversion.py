"""
conversion.py
=============

PyTorch port of moments_op/conversion.hpp -- batched over leading dims.

Three moment representations are used throughout the code:

* Raw moments        M_k = int v^k f(v) dv
* Primitive/central  P = (rho, u, p, q, r, ...), with
                        rho = M0
                        u   = M1 / M0
                        p   = (1/rho) int (v-u)^2 f dv   (variance)
                        q   = (1/rho) int (v-u)^3 f dv   (3rd central moment)
                        ...
* Standardized       S_k = C_k / p^(k/2) for k >= 2 (S2 == 1 by construction;
                      S3, S4 are the q*, r* skewness/kurtosis coordinates).

Every function takes tensors of shape (..., n_mom) and returns tensors with
the same leading dims. The per-order Python loops of the original are
replaced by one (n_mom x n_mom) binomial "shift matrix" per sample, so the
whole conversion is a single batched matmul and stays differentiable.

`strict=True` (default) keeps the original behaviour of raising RuntimeError
on non-positive rho / p. With `strict=False` nothing is raised and invalid
entries come back as NaN (useful for diagnostics on Newton iterates).
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

import torch

from .config import as_tensor


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _binom_cpu(n_mom: int) -> torch.Tensor:
    """(n_mom, n_mom) lower-triangular table of C(k, i) (float64, CPU)."""
    Bm = torch.zeros(n_mom, n_mom, dtype=torch.float64)
    for k in range(n_mom):
        for i in range(k + 1):
            Bm[k, i] = float(comb(k, i))
    return Bm


@lru_cache(maxsize=None)
def _exp_index_cpu(n_mom: int) -> torch.Tensor:
    """(n_mom, n_mom) table of max(k - i, 0) used to gather powers."""
    k = torch.arange(n_mom).unsqueeze(1)
    i = torch.arange(n_mom).unsqueeze(0)
    return torch.clamp(k - i, min=0)


def binomial(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return float(comb(n, k))


def powers(x: torch.Tensor, n: int) -> torch.Tensor:
    """[x^0, x^1, ..., x^(n-1)] stacked on a new last dim, via cumprod."""
    ones = torch.ones_like(x).unsqueeze(-1)
    if n == 1:
        return ones
    rest = x.unsqueeze(-1).expand(*x.shape, n - 1)
    return torch.cumprod(torch.cat([ones, rest], dim=-1), dim=-1)


def shift_matrix(shift: torch.Tensor, n_mom: int, scale: torch.Tensor | None = None) -> torch.Tensor:
    """
    T[..., k, i] = C(k, i) * (-shift)^(k-i) / scale^k   (i <= k, else 0).

    T @ M_raw gives the moments of (v - shift)/scale. This is exactly what
    `standardization_matrix` (shift = u, scale = v_th) and the raw ->
    central conversion (shift = u, scale = 1) need.
    """
    shift = as_tensor(shift, shift.device if isinstance(shift, torch.Tensor) else None,
                      shift.dtype if isinstance(shift, torch.Tensor) else None)
    dev, dt = shift.device, shift.dtype
    Bm = _binom_cpu(n_mom).to(device=dev, dtype=dt)
    E = _exp_index_cpu(n_mom).to(dev)

    pw = powers(-shift, n_mom)                      # (..., K)
    T = Bm * pw[..., E]                             # (..., K, K)
    if scale is not None:
        scale = as_tensor(scale, dev, dt)
        T = T * powers(1.0 / scale, n_mom).unsqueeze(-1)
    return T


def _check_positive(x: torch.Tensor, fn: str, name: str, strict: bool) -> None:
    if strict and bool((x <= 0.0).any()):
        idx = int((x <= 0.0).flatten().nonzero()[0])
        val = float(x.flatten()[idx])
        raise RuntimeError(f"{fn}: {name} must be positive (offending sample {idx}, {name} = {val}).")


# ----------------------------------------------------------------------------
# raw <-> primitive
# ----------------------------------------------------------------------------

def raw_to_primitive_moment(M: torch.Tensor, k: int, strict: bool = True) -> torch.Tensor:
    """Port of `raw_to_primitive_moment`: k-th central moment (rho, u for k = 0, 1)."""
    M = as_tensor(M, M.device if isinstance(M, torch.Tensor) else None,
                  M.dtype if isinstance(M, torch.Tensor) else None)
    rho = M[..., 0]
    _check_positive(rho, "raw_to_primitive_moment", "rho", strict)

    if k == 0:
        return rho
    u = M[..., 1] / rho
    if k == 1:
        return u

    Bk = _binom_cpu(k + 1)[k].to(device=M.device, dtype=M.dtype)        # C(k, i), i = 0..k
    pw = powers(-u, k + 1).flip(-1)                                      # (-u)^(k-i)
    return (Bk * M[..., : k + 1] * pw).sum(-1) / rho


def primitive_to_raw_moment(P: torch.Tensor, k: int) -> torch.Tensor:
    """Port of `primitive_to_raw_moment` (needs P[..., :k+1])."""
    return compute_raw_moments(P[..., : k + 1])[..., k]


def compute_primitive_moments(raw_mom: torch.Tensor, strict: bool = True) -> torch.Tensor:
    """Raw -> primitive/central moments, (..., K) -> (..., K)."""
    raw = as_tensor(raw_mom, raw_mom.device if isinstance(raw_mom, torch.Tensor) else None,
                    raw_mom.dtype if isinstance(raw_mom, torch.Tensor) else None)
    K = raw.shape[-1]

    rho = raw[..., 0]
    _check_positive(rho, "compute_primitive_moments", "rho", strict)
    if K == 1:
        return rho.unsqueeze(-1)

    u = raw[..., 1] / rho
    T = shift_matrix(u, K)                                               # (..., K, K)
    C = torch.einsum("...ki,...i->...k", T, raw) / rho.unsqueeze(-1)     # central moments
    return torch.cat([rho.unsqueeze(-1), u.unsqueeze(-1), C[..., 2:]], dim=-1)


def compute_raw_moments(primitive: torch.Tensor) -> torch.Tensor:
    """Primitive -> raw moments, (..., K) -> (..., K)."""
    P = as_tensor(primitive, primitive.device if isinstance(primitive, torch.Tensor) else None,
                  primitive.dtype if isinstance(primitive, torch.Tensor) else None)
    K = P.shape[-1]
    rho = P[..., 0]
    if K == 1:
        return rho.unsqueeze(-1)

    u = P[..., 1]
    # central-moment vector with (1, 0) in the first two slots:  M = rho * S(u) @ c
    c = torch.cat([torch.ones_like(rho).unsqueeze(-1), torch.zeros_like(rho).unsqueeze(-1), P[..., 2:]], dim=-1)
    S = shift_matrix(-u, K)                                              # C(k,i) u^(k-i)
    return rho.unsqueeze(-1) * torch.einsum("...ki,...i->...k", S, c)


def compute_normalized_moments(c_mom: torch.Tensor, strict: bool = True) -> torch.Tensor:
    """
    Port of `compute_normalized_moments`. S(0) = 1, S(1) = 0, S(2) = 1,
    S(k >= 3) = C_k / p^(k/2)  (S(3) = q*, S(4) = r*).
    """
    c = as_tensor(c_mom, c_mom.device if isinstance(c_mom, torch.Tensor) else None,
                  c_mom.dtype if isinstance(c_mom, torch.Tensor) else None)
    K = c.shape[-1]
    rho = c[..., 0]
    p = c[..., 2]
    _check_positive(rho, "compute_normalized_moments", "rho", strict)
    _check_positive(p, "compute_normalized_moments", "p", strict)

    k = torch.arange(K, device=c.device, dtype=c.dtype)
    S = c / p.unsqueeze(-1) ** (k / 2.0)

    fixed = torch.zeros(K, device=c.device, dtype=c.dtype)
    fixed[0] = 1.0
    fixed[2] = 1.0
    S = torch.where(torch.arange(K, device=c.device) < 3, fixed.expand_as(S), S)

    valid = (rho > 0.0) & (p > 0.0)
    return torch.where(valid.unsqueeze(-1), S, torch.full_like(S, float("nan")))


def raw_to_qr_star(raw_mom: torch.Tensor, strict: bool = True):
    """Raw moments -> (q*, r*) = (S3, S4). Requires n_mom >= 5. Returns two (...,) tensors."""
    c_mom = compute_primitive_moments(raw_mom, strict=strict)
    S = compute_normalized_moments(c_mom, strict=strict)
    return S[..., 3], S[..., 4]
