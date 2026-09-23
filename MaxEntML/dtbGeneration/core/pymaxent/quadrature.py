"""
quadrature.py
=============

PyTorch port of quadrature.hpp / quadrature_generator.hpp.

A `Quadrature` is a pair of tensors (nodes, weights) such that

    integral_R g(v) dv  ~=  sum_q weights[..., q] * g(nodes[..., q]),

with shape ``(Q,)`` (one rule shared by the whole batch) or ``(B, Q)`` (one
rule per batch element -- this is what the adaptive solver produces, since
each moment set gets its own domain).

Everything that *builds* a rule from tensor inputs (interval end points,
u / theta of a scaled Gauss-Hermite rule) is written with plain tensor ops,
so nodes and weights stay attached to the autograd graph of those inputs.

The only non-torch pieces are the *reference tables on [-1, 1]* (Gauss-
Legendre, Gauss-Hermite via numpy, Clenshaw-Curtis closed form). They depend
only on the integer number of points -- constants, never part of a gradient
path -- and are computed once in float64, cached, then cast to the requested
(device, dtype).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Tuple, Union

import numpy as np
import torch

from .config import as_tensor, resolve_device, resolve_dtype


@dataclass
class Quadrature:
    """Nodes + weights, shape (Q,) shared or (B, Q) per batch element."""
    nodes: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    weights: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def __len__(self) -> int:
        return int(self.nodes.shape[-1])

    def to(self, device=None, dtype=None) -> "Quadrature":
        return Quadrature(as_tensor(self.nodes, device, dtype), as_tensor(self.weights, device, dtype))

    def __getitem__(self, idx) -> "Quadrature":
        """Select batch element(s) of a per-sample rule (views, graph preserved)."""
        return Quadrature(self.nodes[idx], self.weights[idx])


def _empty(device=None, dtype=None) -> Quadrature:
    dev, dt = resolve_device(device), resolve_dtype(dtype)
    return Quadrature(torch.empty(0, device=dev, dtype=dt), torch.empty(0, device=dev, dtype=dt))


# ============================================================================
# Reference tables on [-1, 1] (constants; float64 numpy -> cached)
# ============================================================================

@lru_cache(maxsize=None)
def _gl_table(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return np.array(x), np.array(w)


@lru_cache(maxsize=None)
def _gh_table(n: int):
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.array(x), np.array(w)


@lru_cache(maxsize=None)
def _cc_table(n: int):
    """
    Clenshaw-Curtis nodes/weights on [-1, 1], n >= 2 points (Trefethen's
    ``clencurt``), returned in increasing node order.
    """
    N = n - 1
    theta = np.pi * np.arange(N + 1) / N
    x = np.cos(theta)

    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)

    if N % 2 == 0:
        w[0] = 1.0 / (N ** 2 - 1)
        w[N] = w[0]
        for k in range(1, N // 2):
            v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
        v -= np.cos(N * theta[ii]) / (N ** 2 - 1)
    else:
        w[0] = 1.0 / N ** 2
        w[N] = w[0]
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)

    w[ii] = 2.0 * v / N
    return x[::-1].copy(), w[::-1].copy()


def _to_pair(pair, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    x, w = pair
    dev, dt = resolve_device(device), resolve_dtype(dtype)
    return torch.as_tensor(x, dtype=dt, device=dev), torch.as_tensor(w, dtype=dt, device=dev)


def gauss_legendre_reference(n: int, device=None, dtype=None):
    """(x, w) of the n-point Gauss-Legendre rule on [-1, 1]."""
    return _to_pair(_gl_table(n), device, dtype)


def clenshaw_curtis_reference(n: int, device=None, dtype=None):
    """(x, w) of the n-point Clenshaw-Curtis rule on [-1, 1]."""
    if n == 1:
        return _to_pair((np.array([0.0]), np.array([2.0])), device, dtype)
    return _to_pair(_cc_table(n), device, dtype)


def _map_interval(x: torch.Tensor, w: torch.Tensor, a, b):
    """Affine map of a [-1, 1] rule to [a, b]; a, b may be scalars or tensors of shape (...)."""
    a = as_tensor(a, x.device, x.dtype)
    b = as_tensor(b, x.device, x.dtype)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    nodes = mid.unsqueeze(-1) + half.unsqueeze(-1) * x
    weights = half.unsqueeze(-1) * w
    return nodes, weights


# ============================================================================
# Gauss-Hermite quadrature
# ============================================================================

def apply_gaussian_adjustment(nodes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Port of `quad::apply_gaussian_adjustment`: w_adj = w * exp(x^2), so that
    the Gauss-Hermite rule integrates an *unweighted* function of v. The
    product is formed in float64 (exp(x^2) overflows float32 for |x| > 9.4);
    non-finite results are zeroed, as in the C++ overflow protection.
    """
    adjusted = weights.to(torch.float64) * torch.exp(nodes.to(torch.float64) ** 2)
    adjusted = adjusted.to(weights.dtype)
    return torch.where(torch.isfinite(adjusted), adjusted, torch.zeros_like(adjusted))


def gauss_hermite(n: int, adjusted: bool = False, device=None, dtype=None) -> Quadrature:
    """Port of `quad::gauss_hermite` (physicists' rule, weight exp(-x^2))."""
    if n <= 0:
        return _empty(device, dtype)
    nodes, weights = _to_pair(_gh_table(n), device, dtype)
    if adjusted:
        weights = apply_gaussian_adjustment(nodes, weights)
    return Quadrature(nodes, weights)


def gauss_hermite_scaled(n_quad: int, u, theta, device=None, dtype=None) -> Quadrature:
    """
    Port of `quad::gauss_hermite_scaled`: v = u + sqrt(2 theta) x with the
    *unadjusted* weights. `u`, `theta` may be scalars or (B,) tensors, in
    which case nodes have shape (B, n_quad) (and stay differentiable in u, theta).
    """
    base = gauss_hermite(n_quad, adjusted=False, device=device, dtype=dtype)
    u = as_tensor(u, base.nodes.device, base.nodes.dtype)
    theta = as_tensor(theta, base.nodes.device, base.nodes.dtype)
    scale = torch.sqrt(2.0 * theta)
    nodes = base.nodes * scale.unsqueeze(-1) + u.unsqueeze(-1)
    return Quadrature(nodes, base.weights.clone())


# ============================================================================
# Gauss-Legendre / Clenshaw-Curtis on [a, b]
# ============================================================================

def gauss_legendre(n: int, device=None, dtype=None) -> Quadrature:
    """Port of `quad::gauss_legendre`: standard rule on [-1, 1]."""
    if n <= 0:
        return _empty(device, dtype)
    x, w = gauss_legendre_reference(n, device, dtype)
    return Quadrature(x, w)


def gauss_legendre_interval(n: int, a, b, device=None, dtype=None) -> Quadrature:
    """Port of `quad::gauss_legendre_interval`; a, b scalars or (B,) tensors."""
    if n <= 0:
        return _empty(device, dtype)
    x, w = gauss_legendre_reference(n, device, dtype)
    return Quadrature(*_map_interval(x, w, a, b))


def clenshaw_curtis(n: int, device=None, dtype=None) -> Quadrature:
    if n <= 0:
        return _empty(device, dtype)
    x, w = clenshaw_curtis_reference(n, device, dtype)
    return Quadrature(x, w)


def clenshaw_curtis_interval(n: int, a, b, device=None, dtype=None) -> Quadrature:
    if n <= 0:
        return _empty(device, dtype)
    x, w = clenshaw_curtis_reference(n, device, dtype)
    return Quadrature(*_map_interval(x, w, a, b))


# ============================================================================
# Composite (piecewise) quadrature
# ============================================================================

#: name -> callable(n_pts, device, dtype) -> (x_ref, w_ref) on [-1, 1].
#: A custom rule can be passed to `composite_quadrature` directly as a
#: callable with that same signature.
COMPOSITE_RULES = {
    "gauss-legendre": gauss_legendre_reference,
    "gauss_legendre": gauss_legendre_reference,
    "gauss-legendre-interval": gauss_legendre_reference,
    "legendre": gauss_legendre_reference,
    "gl": gauss_legendre_reference,
    "clenshaw-curtis": clenshaw_curtis_reference,
    "clenshaw_curtis": clenshaw_curtis_reference,
    "clenshawcurtis": clenshaw_curtis_reference,
    "cc": clenshaw_curtis_reference,
}

DEFAULT_COMPOSITE_RULE = "gauss-legendre"


def _resolve_composite_rule(rule) -> Callable:
    if callable(rule):
        return rule
    try:
        return COMPOSITE_RULES[rule.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown composite quadrature rule {rule!r}; expected one of "
            f"{sorted(set(COMPOSITE_RULES))} or a callable(n_pts, device, dtype) -> (x_ref, w_ref)."
        ) from None


def composite_quadrature(a, b, n_cells: int, n_pts: int = 5,
                         rule=DEFAULT_COMPOSITE_RULE, device=None, dtype=None) -> Quadrature:
    """
    Piecewise quadrature on [a, b]: `n_cells` equal cells, an `n_pts`-point
    rule in each (Gauss-Legendre by default, Clenshaw-Curtis available).

    Fully vectorised: no loop over cells. `a`, `b` may be scalars (result
    shape (n_cells*n_pts,)) or (B,) tensors (result shape (B, n_cells*n_pts));
    the nodes/weights are differentiable functions of a and b. `n_cells` and
    `n_pts` are integers, i.e. they set tensor *shapes*, not values.
    """
    dev, dt = resolve_device(device), resolve_dtype(dtype)
    if n_cells < 1:
        return _empty(dev, dt)

    a = as_tensor(a, dev, dt)
    b = as_tensor(b, dev, dt)

    x_ref, w_ref = _resolve_composite_rule(rule)(n_pts, dev, dt)

    t = torch.linspace(0.0, 1.0, n_cells + 1, device=dev, dtype=dt)
    edges = a.unsqueeze(-1) + (b - a).unsqueeze(-1) * t          # (..., n_cells + 1)
    lo, hi = edges[..., :-1], edges[..., 1:]
    mid = 0.5 * (lo + hi)                                        # (..., n_cells)
    half = 0.5 * (hi - lo)

    nodes = mid.unsqueeze(-1) + half.unsqueeze(-1) * x_ref       # (..., n_cells, n_pts)
    weights = half.unsqueeze(-1) * w_ref
    return Quadrature(nodes.flatten(-2), weights.flatten(-2))


def composite_quadrature_centered(v0, half_width, n_cells: int, n_pts: int = 5,
                                  rule=DEFAULT_COMPOSITE_RULE, device=None, dtype=None) -> Quadrature:
    """Composite rule on [v0 - half_width, v0 + half_width] (tensor-valued v0, half_width allowed)."""
    v0 = as_tensor(v0, device, dtype)
    half_width = as_tensor(half_width, device, dtype)
    return composite_quadrature(v0 - half_width, v0 + half_width, n_cells, n_pts, rule=rule,
                                device=device, dtype=dtype)


# ---- backwards-compatible aliases ------------------------------------------

def composite_gauss_legendre(a, b, n_cells: int, n_pts: int = 5, device=None, dtype=None) -> Quadrature:
    return composite_quadrature(a, b, n_cells, n_pts, rule="gauss-legendre", device=device, dtype=dtype)


def composite_gauss_legendre_centered(v0, half_width, n_cells: int, n_pts: int = 5,
                                      device=None, dtype=None) -> Quadrature:
    return composite_quadrature_centered(v0, half_width, n_cells, n_pts, rule="gauss-legendre",
                                         device=device, dtype=dtype)


# ============================================================================
# Newton-Cotes quadrature
# ============================================================================

def uniform_trapezoidal(n: int, a, b, device=None, dtype=None) -> Quadrature:
    """Port of `quad::uniform_trapezoidal`; a, b scalars or (B,) tensors."""
    dev, dt = resolve_device(device), resolve_dtype(dtype)
    if n < 2:
        return _empty(dev, dt)
    a = as_tensor(a, dev, dt)
    b = as_tensor(b, dev, dt)

    t = torch.linspace(0.0, 1.0, n, device=dev, dtype=dt)
    nodes = a.unsqueeze(-1) + (b - a).unsqueeze(-1) * t
    h = (b - a) / (n - 1)

    shape = torch.ones(n, device=dev, dtype=dt)
    shape = torch.cat([shape.new_tensor([0.5]), shape[1:-1], shape.new_tensor([0.5])])
    weights = h.unsqueeze(-1) * shape
    return Quadrature(nodes, weights)


# ============================================================================
# Monomial basis
# ============================================================================

def build_monomial_basis(nodes: torch.Tensor, n_mom: int) -> torch.Tensor:
    """
    Port of `optim::build_monomial_basis`: Phi[..., k, q] = nodes[..., q]**k,
    k = 0..n_mom-1, for nodes of shape (Q,) -> (n_mom, Q) or (B, Q) ->
    (B, n_mom, Q). Built with a cumulative product (same recurrence as the
    original), which is differentiable, also at nodes exactly equal to 0.
    """
    ones = torch.ones_like(nodes).unsqueeze(-2)
    if n_mom == 1:
        return ones
    rest = nodes.unsqueeze(-2).expand(*nodes.shape[:-1], n_mom - 1, nodes.shape[-1])
    return torch.cumprod(torch.cat([ones, rest], dim=-2), dim=-2)


# ============================================================================
# Quadrature generator dispatcher (port of quadrature_generator.hpp)
# ============================================================================

def _dispatch(quad_type: int, n_quad: int, a, b, device=None, dtype=None) -> Quadrature:
    if quad_type == 1:
        return gauss_hermite(n_quad, adjusted=True, device=device, dtype=dtype)
    elif quad_type == 2:
        return gauss_legendre_interval(n_quad, a, b, device, dtype)
    elif quad_type == 3:
        return uniform_trapezoidal(n_quad, a, b, device, dtype)
    elif quad_type == 4:
        return clenshaw_curtis_interval(n_quad, a, b, device, dtype)
    else:
        return gauss_legendre_interval(n_quad, a, b, device, dtype)


def generate(closure_id: int, params, device=None, dtype=None) -> Quadrature:
    """Port of `quad::generate(closure_id, params)`."""
    if closure_id == 10:
        return gauss_hermite_scaled(params.n_quad, params.u_lin, params.theta_lin, device, dtype)
    return _dispatch(params.quad_type, params.n_quad, params.int_a, params.int_b, device, dtype)


def generate_from_center(closure_id: int, params, v0, deltaV, device=None, dtype=None) -> Quadrature:
    """Port of the `quad::generate(closure_id, params, v0, deltaV)` overload."""
    v0 = as_tensor(v0, device, dtype)
    deltaV = as_tensor(deltaV, device, dtype)
    if closure_id == 10:
        return gauss_hermite_scaled(params.n_quad, params.u_lin, params.theta_lin, device, dtype)
    return _dispatch(params.quad_type, params.n_quad, v0 - deltaV, v0 + deltaV, device, dtype)


def generate_with_limits(closure_id: int, params, a, b, device=None, dtype=None) -> Quadrature:
    """Port of `quad::generate_with_limits`."""
    if closure_id == 10:
        return gauss_hermite_scaled(params.n_quad, params.u_lin, params.theta_lin, device, dtype)
    return _dispatch(params.quad_type, params.n_quad, a, b, device, dtype)
