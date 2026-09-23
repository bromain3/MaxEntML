"""
optimisation.py
================

PyTorch port of core/optimisation/optimisation.hpp (+ strategy structs).

Everything is batched over B independent dual problems and built from
differentiable tensor ops only (no numpy, no ``.item()``, no ``detach`` on
the solution path), so the converged multipliers are differentiable with
respect to the target moments, the quadrature nodes/weights, ...

  * NewtonStrategy / HybridStrategy -- plain / Armijo-damped-then-full steps.
  * newton_direction  -- Cholesky first, Levenberg-Marquardt fallback,
                         chosen *per sample* with masks (no Python branching
                         on individual samples).
  * solve_dual        -- batched Newton with per-sample convergence masks,
                         optional Alldredge Adaptive Change of Basis (ACB),
                         and two differentiation modes (see below).

Differentiation modes (``grad_mode``)
-------------------------------------
"unroll"   (default) Every Newton iterate stays in the autograd graph.
           Exactly the computation you wrote, differentiated end to end.
           Memory grows with the number of iterations.
"implicit" The iterations run without recording; then ONE extra Newton step
           is taken at the converged point with the graph on. Because
           grad = 0 there, this reproduces the implicit-function-theorem
           gradient exactly, in O(1) memory. The output is still connected
           to the inputs.

Converged samples are frozen with ``torch.where`` masks so the batch keeps
static shapes; the loop stops when no sample is active any more.

`sys` (the "DualSystem") is a `maxent.MaxentDualSystem`-like object exposing
``evaluate(lam)``, ``evaluate_obj(lam)`` and (for ACB) a *functional*
``adapt_basis(L) -> new system``.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import as_tensor, floor_tol
from .conversion import shift_matrix


# ============================================================================
# Strategy parameter structs
# ============================================================================

@dataclass
class LineSearchParams:
    c1: float = 1.0e-4
    max_ls: int = 20
    rho: float = 0.5


@dataclass
class HybridParams:
    switch_tol: float = 1.0e-3
    ls_params: LineSearchParams = field(default_factory=LineSearchParams)


@dataclass
class NewtonStrategy:
    """Plain, full Newton step: lam += dlam."""
    pass


@dataclass
class HybridStrategy:
    """Armijo-damped Newton while ||grad|| > switch_tol, full Newton after."""
    params: HybridParams = field(default_factory=HybridParams)


def has_acb_support(sys) -> bool:
    return hasattr(sys, "adapt_basis")


# ============================================================================
# Armijo line search (vectorised over the batch, no graph recorded)
# ============================================================================

@torch.no_grad()
def armijo_step(sys, lam: torch.Tensor, dlam: torch.Tensor, grad: torch.Tensor,
                obj0: torch.Tensor, lsp: LineSearchParams) -> torch.Tensor:
    """
    Port of `optim::armijo_step`, one step length per sample: returns alpha of
    shape (B,). The search only makes discrete accept/reject decisions, so it
    runs without recording autograd history; the actual update
    ``lam + alpha * dlam`` (done by the caller) keeps the graph.
    """
    slope0 = (grad * dlam).sum(dim=-1)
    valid = (slope0 < 0.0) & torch.isfinite(slope0)
    slope = lsp.c1 * slope0
    obj0 = obj0.detach()

    alpha = torch.ones_like(slope0)
    done = ~valid                                   # invalid samples get alpha = 0 below

    for _ in range(lsp.max_ls):
        obj_try = sys.evaluate_obj(lam + alpha.unsqueeze(-1) * dlam)
        ok = torch.isfinite(obj_try) & (obj_try <= obj0 + alpha * slope)
        done = done | ok
        alpha = torch.where(done, alpha, alpha * lsp.rho)
        if bool(done.all()):
            break

    return torch.where(valid, alpha, torch.zeros_like(alpha))


def apply_step(sys, lam: torch.Tensor, dlam: torch.Tensor, grad: torch.Tensor,
               obj0: torch.Tensor, strat) -> torch.Tensor:
    """Port of the `optim::apply_step` overload set (per-sample switch for the hybrid strategy)."""
    if isinstance(strat, HybridStrategy):
        gn = torch.linalg.vector_norm(grad, dim=-1)
        use_ls = gn > strat.params.switch_tol
        if bool(use_ls.any()):
            alpha = armijo_step(sys, lam, dlam, grad, obj0, strat.params.ls_params)
            alpha = torch.where(use_ls, alpha, torch.ones_like(alpha))
            return lam + alpha.unsqueeze(-1) * dlam
        return lam + dlam
    return lam + dlam


# ============================================================================
# Newton direction
# ============================================================================

def newton_direction(grad: torch.Tensor, hess: torch.Tensor) -> torch.Tensor:
    """
    Port of `optim::newton_direction`, batched: grad (B, K), hess (B, K, K) -> (B, K).

    Per sample: Cholesky solve of hess @ d = -grad if the Hessian is SPD;
    otherwise Levenberg-Marquardt damping (up to 5 levels, x10 each) until a
    genuine descent direction is found; otherwise d = -grad. Samples are
    routed with masks, never with per-sample Python control flow.
    """
    K = grad.shape[-1]
    dev, dt = grad.device, grad.dtype
    rhs = (-grad).unsqueeze(-1)

    L, info = torch.linalg.cholesky_ex(hess)
    d_chol = torch.cholesky_solve(rhs, L).squeeze(-1)
    ok_chol = (info == 0) & torch.isfinite(d_chol).all(dim=-1)
    if bool(ok_chol.all()):
        return d_chol

    gnorm = torch.linalg.vector_norm(grad, dim=-1)
    diag_max = torch.diagonal(hess, dim1=-2, dim2=-1).amax(dim=-1)
    base = 1.0e-6 * diag_max
    base = torch.where(base > 0.0, base, torch.full_like(base, 1.0e-7))

    eye = torch.eye(K, device=dev, dtype=dt)
    d_fb = torch.zeros_like(grad)
    found = torch.zeros_like(ok_chol)
    damping = base

    for _ in range(5):
        reg = hess + damping.view(-1, 1, 1) * eye
        d_try, info_try = torch.linalg.solve_ex(reg, rhs)
        d_try = d_try.squeeze(-1)
        descent = (grad * d_try).sum(dim=-1) < -1.0e-12 * gnorm * torch.linalg.vector_norm(d_try, dim=-1)
        good = (info_try == 0) & torch.isfinite(d_try).all(dim=-1) & descent
        take = good & ~found
        d_fb = torch.where(take.unsqueeze(-1), d_try, d_fb)
        found = found | good
        damping = damping * 10.0

    # last resort (as in the original, whose fallback norm is 0): steepest descent
    d_fb = torch.where(found.unsqueeze(-1), d_fb, -grad)
    return torch.where(ok_chol.unsqueeze(-1), d_chol, d_fb)


def _clip_step(dlam: torch.Tensor, max_step: float) -> torch.Tensor:
    """Rescale each row of dlam to norm <= max_step (safe for zero rows)."""
    nrm = torch.linalg.vector_norm(dlam, dim=-1)
    big = nrm > max_step
    safe = torch.where(big, nrm, torch.ones_like(nrm))
    scale = torch.where(big, max_step / safe, torch.ones_like(nrm))
    return dlam * scale.unsqueeze(-1)


# ============================================================================
# Batched Newton solver with optional ACB
# ============================================================================

@dataclass
class SolveDiagnostics:
    """
    Per-sample diagnostics (all tensors are detached, they are not on the
    gradient path).

    converged        (B,) bool
    n_iter           (B,) long   -- number of dual evaluations the sample took part in
    grad_norm, obj   (B,)        -- at the returned multipliers
    lambda_history   list of (B, K) tensors, physical basis of the solved
                     system, [initial guess, after step 1, ...]; a sample's own
                     trajectory is ``[h[i] for h in hist[:n_iter[i]]]``.
    """
    converged: Optional[torch.Tensor] = None
    n_iter: Optional[torch.Tensor] = None
    grad_norm: Optional[torch.Tensor] = None
    obj: Optional[torch.Tensor] = None
    lambda_history: list = field(default_factory=list)
    obj_history: list = field(default_factory=list)
    grad_norm_history: list = field(default_factory=list)


def _phys(lam: torch.Tensor, T_cum: Optional[torch.Tensor]) -> torch.Tensor:
    """Rotated-basis lambda -> lambda in the basis of the original system: T_cum^T lam."""
    if T_cum is None:
        return lam
    return torch.einsum("bki,bk->bi", T_cum, lam)


def solve_dual(sys,
               K: int,
               strat=None,
               tol: float = 1e-10,
               maxit: int = 100,
               lam0: Optional[torch.Tensor] = None,
               use_adaptive: bool = False,
               verbose: bool = False,
               record_history: bool = False,
               grad_mode: str = "unroll",
               raise_on_fail: bool = True):
    """
    Port of `optim::solve_dual`, batched.

    Returns (lam (B, K), diagnostics). `lam` is in the basis of `sys` (any ACB
    rotation is undone). With ``raise_on_fail=True`` a RuntimeError is raised
    if any sample did not converge (original behaviour); with False the
    per-sample flags in ``diagnostics.converged`` tell which ones failed.

    The requested `tol` is floored to what the dtype can resolve
    (see `config.floor_tol`).
    """
    if strat is None:
        strat = NewtonStrategy()
    if grad_mode not in ("unroll", "implicit"):
        raise ValueError(f"grad_mode must be 'unroll' or 'implicit', got {grad_mode!r}")

    use_acb = bool(use_adaptive)
    if use_acb and not has_acb_support(sys):
        raise RuntimeError("solve_dual: use_adaptive = True was requested but this DualSystem "
                           "does not implement adapt_basis().")

    dev, dt = sys.targets.device, sys.targets.dtype
    B = sys.targets.shape[0]
    tol = floor_tol(tol, dt)

    lam = None
    if lam0 is not None:
        lam = as_tensor(lam0, dev, dt)
        if lam.shape[-1] != K:
            lam = None
        elif lam.ndim == 1:
            lam = lam.unsqueeze(0).expand(B, K)
    if lam is None:
        lam = torch.zeros(B, K, device=dev, dtype=dt)

    stall_patience = 5
    stall_rtol = 1e-2
    tol_floor_relaxed = max(tol * 1.0e3, 1.0e-6)
    max_step = 1.5

    eye = torch.eye(K, device=dev, dtype=dt).expand(B, K, K)
    cur = sys                                        # system in the current (possibly rotated) basis
    T_cum = eye.clone() if use_acb else None         # cumulative ACB rotation

    active = torch.ones(B, dtype=torch.bool, device=dev)
    converged = torch.zeros(B, dtype=torch.bool, device=dev)
    n_eval = torch.zeros(B, dtype=torch.long, device=dev)
    stall_count = torch.zeros(B, dtype=torch.long, device=dev)
    gn_prev = torch.full((B,), float("inf"), device=dev, dtype=dt)
    gn_last = torch.full((B,), float("inf"), device=dev, dtype=dt)
    obj_last = torch.full((B,), float("nan"), device=dev, dtype=dt)

    diag = SolveDiagnostics()

    ctx = torch.no_grad() if grad_mode == "implicit" else contextlib.nullcontext()
    with ctx:
        if record_history:
            diag.lambda_history.append(_phys(lam, T_cum).detach())

        for it in range(maxit):
            obj, grad, hess = cur.evaluate(lam)
            gn = torch.linalg.vector_norm(grad, dim=-1)
            n_eval = n_eval + active.to(n_eval.dtype)
            gn_last, obj_last = gn.detach(), obj.detach()

            if verbose:
                worst = float(torch.nan_to_num(gn.detach(), nan=float("inf")).max())
                print(f"iter={it:3d}  active={int(active.sum()):5d}/{B}  max|grad|={worst: .3e}")

            if record_history:
                diag.obj_history.append(obj.detach())
                diag.grad_norm_history.append(gn.detach())

            # ---- convergence / divergence bookkeeping --------------------
            bad = active & ~torch.isfinite(gn)
            hit = active & (gn < tol)
            converged = converged | hit
            active = active & ~hit & ~bad

            stalled = active & (gn > (1.0 - stall_rtol) * gn_prev)
            stall_count = torch.where(stalled, stall_count + 1, torch.zeros_like(stall_count))
            accept_stall = stalled & (stall_count >= stall_patience) & (gn < tol_floor_relaxed)
            converged = converged | accept_stall
            active = active & ~accept_stall
            gn_prev = torch.where(active, gn.detach(), gn_prev)

            if not bool(active.any()):
                break

            # ---- Alldredge Adaptive Change of Basis (functional) ---------
            if use_acb:
                L, info = torch.linalg.cholesky_ex(hess)
                rot = active & (info == 0) & torch.isfinite(L).flatten(-2).all(dim=-1)
                L = torch.where(rot.view(-1, 1, 1), L, eye)
                cur = cur.adapt_basis(L)
                T_cum = torch.linalg.solve_triangular(L, T_cum, upper=False)
                lam = torch.einsum("bji,bj->bi", L, lam)                       # L^T lam
                grad = torch.linalg.solve_triangular(L, grad.unsqueeze(-1), upper=False).squeeze(-1)
                hess = torch.where(rot.view(-1, 1, 1), eye, hess)

            # ---- Newton step ----------------------------------------------
            dlam = newton_direction(grad, hess)
            dlam = _clip_step(dlam, max_step)
            lam_new = apply_step(cur, lam, dlam, grad, obj, strat)
            lam = torch.where(active.unsqueeze(-1), lam_new, lam)

            if record_history:
                diag.lambda_history.append(_phys(lam, T_cum).detach())

    # ---- back to the basis of the original system --------------------------
    lam_sys = _phys(lam, T_cum)

    if grad_mode == "implicit":
        # One differentiable Newton step at the (detached) solution: exact
        # implicit-function-theorem gradient, since grad(lam*) = 0.
        _, grad_f, hess_f = sys.evaluate(lam_sys)
        lam_step = lam_sys + newton_direction(grad_f, hess_f)
        lam_sys = torch.where(converged.unsqueeze(-1), lam_step, lam_sys)

    diag.converged = converged
    diag.n_iter = n_eval
    diag.grad_norm = gn_last
    diag.obj = obj_last

    if raise_on_fail and not bool(converged.all()):
        n_bad = int((~converged).sum())
        raise RuntimeError(f"Newton-Raphson: Error! Optimization did not converge for {n_bad}/{B} "
                           f"sample(s) (worst final |grad| = {float(gn_last[~converged].max()):.3e}).")

    return lam_sys, diag


# ============================================================================
# Standardization (preconditioning) transform
# ============================================================================

def standardization_matrix(u: torch.Tensor, v_th: torch.Tensor, n_mom: int,
                           varsigma: float = 1.0) -> torch.Tensor:
    """
    Port of `maxent::standardization_matrix`, batched: u, v_th of shape (B,)
    -> T of shape (B, n_mom, n_mom), T[k, i] = C(k, i) (-u)^(k-i) / (v_th*varsigma)^k,
    i.e. the transform to the (v - u)/(v_th*varsigma) monomial basis.
    """
    return shift_matrix(u, n_mom, scale=v_th * varsigma)


# ============================================================================
# Further ("varsigma") preconditioning: last-order-moment rescaling
# ============================================================================

def varsigma_rescaling(n_mom: int) -> float:
    """
    Scalar varsigma = [(2M-1)!!]^(1/(2M)), M = n_mom - 1, obtained by
    matching the order-2M moment of the standard Gaussian to m_0 = 1.
    Pure Python float (depends only on the integer n_mom). Returns 1.0 for M <= 0.
    """
    M = n_mom - 1
    if M <= 0:
        return 1.0
    log_double_fact = sum(math.log(k) for k in range(1, 2 * M, 2))
    return math.exp(log_double_fact / (2 * M))
