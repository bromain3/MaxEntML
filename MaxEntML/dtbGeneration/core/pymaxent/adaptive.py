"""
adaptive.py
===========

PyTorch port of the adaptive quadrature strategy for the dual integrals
(Eq. 16): composite piecewise Gauss quadrature whose domain size and cell
resolution are adapted per moment set.

Two independent adaptation knobs, exactly as in the NumPy version:

1. Domain-size check (tail decay + leading-coefficient certificate). If the
   reconstructed VDF has not decayed at the edges, the half-width (and the
   cell count) is grown geometrically and the dual is re-solved.
2. Resolution check (moment re-verification on an independent, finer rule).
   If the re-computed moments miss the target, the cell count is increased.

Batching
--------
All B moment sets are solved together. Every sample keeps its OWN state
(half-width, number of cells, growth / refinement budgets), so results match
running the samples one by one. In each round the still-active samples are
grouped by their current cell count (the only quantity that changes tensor
shapes); each group is one batched solve on a per-sample domain
(nodes of shape (G, n_cells * n_pts)). Groups are few (cell counts follow a
geometric sequence), so the Python loop is over *groups*, never over samples.

Autograd
--------
The multipliers, the closing moment and the quadrature (a, b are functions of
the target moments) all stay attached to the graph; results of different
rounds/groups are written back with out-of-place ``index_put``. The tail /
moment diagnostics are discrete decisions and run under ``no_grad``.
Samples whose Newton solve never converges come back as NaN with
``converged = False`` (``raise_on_fail=True`` raises instead, like the
original); mask them out of any loss you backpropagate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .config import as_batched, floor_tol, tiny
from .quadrature import (
    Quadrature, composite_quadrature, build_monomial_basis, DEFAULT_COMPOSITE_RULE,
)
from .conversion import compute_primitive_moments
from .realisability import check_realizability
from .initialisation import maxwellian_init
from .optimisation import HybridStrategy, SolveDiagnostics, varsigma_rescaling
from . import maxent


# ============================================================================
# Parameters
# ============================================================================

@dataclass
class AdaptiveQuadratureParams:
    n_pts: int = 5                 # points per cell
    quad_rule: str = DEFAULT_COMPOSITE_RULE   # "gauss-legendre" | "clenshaw-curtis" | callable(n_pts, device, dtype)
    n_cells0: int = 40             # initial number of cells
    k_sigma0: float = 8.0          # initial half-width in units of sqrt(theta), around u

    # -- domain-size (tail decay) adaptation --
    tail_rtol: float = 1e-10
    domain_growth_factor: float = 1.5
    max_domain_growths: int = 8

    # -- resolution (moment re-verification) adaptation --
    verify_cell_factor: int = 4
    verify_n_pts: int = 8
    verify_quad_rule: Optional[str] = None
    moment_rtol: float = 1e-8      # (floored to what the dtype can resolve)
    moment_atol: float = 1e-10
    resolution_growth_factor: float = 2.0
    max_resolution_refinements: int = 6


@dataclass
class AdaptiveRound:
    """One (build quadrature -> solve -> diagnose) round of ONE sample, for the log."""
    round_index: int
    n_cells: int
    half_width: float
    domain: tuple
    converged: bool
    n_iter: int
    tail_ratio: float
    tail_ok: bool
    moment_rel_err: float
    moments_ok: bool
    action: str  # "accepted" | "grew_domain" | "refined_resolution" | "failed"

    # Newton trajectory of this sample (list of (K,) tensors, physical basis,
    # detached) and the sample's own quadrature for the round. Empty / None
    # for failed rounds or when record_history was False.
    lambda_history: list = field(default_factory=list)
    quad: "Quadrature | None" = None


@dataclass
class AdaptiveResult:
    lam: torch.Tensor                      # (B, K) physical-basis multipliers (NaN if not solved)
    closure: torch.Tensor                  # (B,)  closing moment M_K
    quads: List[Optional[Quadrature]]      # per-sample final quadrature, nodes/weights shape (Q_b,)
    diag: SolveDiagnostics                 # per-sample stats of the accepted round, tensors (B,)
    rounds: List[List[AdaptiveRound]]      # rounds[b] = adaptation log of sample b
    n_cells_final: torch.Tensor            # (B,) long
    half_width_final: torch.Tensor         # (B,)
    verified_moment_rel_err: torch.Tensor  # (B,) max abs moment mismatch on the finer rule
    tail_ratio: torch.Tensor               # (B,)
    converged: torch.Tensor                # (B,) bool


# ============================================================================
# Diagnostics (discrete decisions -> no_grad)
# ============================================================================

@torch.no_grad()
def _tail_ratio(entropy_policy, lam: torch.Tensor, quad: Quadrature) -> torch.Tensor:
    """
    (G,) ratio of the reconstructed VDF at the domain edges to its interior
    maximum; +inf if the maximum is not a finite positive number.
    """
    Phi = build_monomial_basis(quad.nodes, lam.shape[-1])
    f = entropy_policy.eta_dual_prime(torch.einsum("bkq,bk->bq", Phi, lam))

    f_max = f.amax(dim=-1)
    f_edge = torch.maximum(f[..., 0], f[..., -1])
    good = torch.isfinite(f_max) & (f_max > 0.0)
    ratio = f_edge / torch.where(good, f_max, torch.ones_like(f_max))
    return torch.where(good, ratio, torch.full_like(ratio, float("inf")))


def _leading_coeff_ok(lam: torch.Tensor, lambda_eps: float = 1.0e-15) -> torch.Tensor:
    """(G,) bool: highest-order multiplier strictly negative (VDF bounded on R)."""
    return lam[..., -1] < -lambda_eps


@torch.no_grad()
def _verify_moments(entropy_policy, lam: torch.Tensor, m_target: torch.Tensor,
                    a: torch.Tensor, b: torch.Tensor, n_cells_solve: int,
                    params: AdaptiveQuadratureParams):
    """
    Re-compute the moments implied by `lam` on an independent, finer rule
    (more cells AND more points per cell) spanning the same per-sample domain
    [a, b]; returns (m_recomputed (G, K), max abs error (G,), ok (G,) bool).
    """
    K = m_target.shape[-1]
    n_cells_verify = max(1, params.verify_cell_factor) * max(1, n_cells_solve)
    verify_rule = params.verify_quad_rule if params.verify_quad_rule is not None else params.quad_rule
    vq = composite_quadrature(a, b, n_cells=n_cells_verify, n_pts=params.verify_n_pts,
                              rule=verify_rule, device=lam.device, dtype=lam.dtype)

    Phi = build_monomial_basis(vq.nodes, K)
    f = entropy_policy.eta_dual_prime(torch.einsum("bkq,bk->bq", Phi, lam))
    m_re = torch.einsum("bkq,bq->bk", Phi, vq.weights * f)

    abs_err = (m_re - m_target).abs()
    atol = floor_tol(params.moment_atol, lam.dtype)
    rtol = floor_tol(params.moment_rtol, lam.dtype)
    ok = (abs_err <= atol + rtol * m_target.abs()).all(dim=-1)
    return m_re, abs_err.amax(dim=-1), ok


def _default_lam0_physical(m: torch.Tensor, use_initial_max: bool, use_precond: bool,
                           use_varsigma: bool) -> torch.Tensor:
    """The usual initial guess, expressed in the physical basis (fallback for warm starts)."""
    if use_initial_max:
        return maxwellian_init(m)
    varsigma = varsigma_rescaling(m.shape[-1]) if (use_precond and use_varsigma) else 1.0
    return maxent.physical_maxwellian_q0r2_init(m, varsigma=varsigma)


# ============================================================================
# Main adaptive driver
# ============================================================================

def solve_adaptive(entropy_policy, m_target: torch.Tensor,
                   params: AdaptiveQuadratureParams | None = None,
                   use_precond: bool = False, use_adapt_base: bool = False,
                   use_initial_max: bool = True, use_varsigma: bool = False,
                   strat=None, tol: float = 1e-10, maxit: int = 200,
                   record_history: bool = False, verbose: bool = False,
                   check_realizable: bool = True,
                   use_lambda_init: bool = False,
                   grad_mode: str = "unroll",
                   raise_on_fail: bool = True,
                   device=None, dtype=None) -> AdaptiveResult:
    """
    Batched port of `solve_adaptive`: solves the MaxEnt dual for every row of
    `m_target` (B, K) with per-sample adaptive composite quadrature.

    `use_lambda_init`: rounds after the first are warm-started from the
    sample's previous converged lambda (falls back to the usual guess for a
    sample whose previous round failed).
    `grad_mode` ("unroll" | "implicit") is forwarded to the Newton solver.
    `raise_on_fail`: raise RuntimeError if some sample exhausts its domain-
    growth budget without a converged solve (original behaviour); otherwise
    such samples are returned as NaN with `converged=False`.
    """
    if params is None:
        params = AdaptiveQuadratureParams()
    if strat is None:
        strat = HybridStrategy()

    m = as_batched(m_target,
                   device if device is not None else getattr(m_target, "device", None),
                   dtype if dtype is not None else getattr(m_target, "dtype", None))
    B, K = m.shape
    dev, dt = m.device, m.dtype

    # ---- initial per-sample domain from the target's own mean / variance ---
    c_mom = compute_primitive_moments(m)
    if check_realizable:
        check_realizability(c_mom, is_primitive=True, throw_on_error=True)
    u0 = c_mom[:, 1]
    sigma0 = torch.sqrt(torch.clamp(c_mom[:, 2], min=tiny(dt)))
    half_width = params.k_sigma0 * sigma0                                   # (B,) device

    # ---- per-sample control state (CPU: it only drives Python control flow)
    n_cells = torch.full((B,), int(params.n_cells0), dtype=torch.long)
    n_growths = torch.zeros(B, dtype=torch.long)
    n_refines = torch.zeros(B, dtype=torch.long)
    round_idx = torch.zeros(B, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    failed = torch.zeros(B, dtype=torch.bool)
    n_cells_final = torch.zeros(B, dtype=torch.long)
    rounds: List[List[AdaptiveRound]] = [[] for _ in range(B)]
    quads_out: List[Optional[Quadrature]] = [None] * B

    # ---- warm-start state and outputs (device) ------------------------------
    nan = float("nan")
    lam_prev = torch.zeros(B, K, device=dev, dtype=dt)
    has_prev = torch.zeros(B, dtype=torch.bool, device=dev)

    lam_out = torch.full((B, K), nan, device=dev, dtype=dt)
    closure_out = torch.full((B,), nan, device=dev, dtype=dt)
    niter_out = torch.zeros(B, dtype=torch.long, device=dev)
    gnorm_out = torch.full((B,), nan, device=dev, dtype=dt)
    obj_out = torch.full((B,), nan, device=dev, dtype=dt)
    tail_out = torch.full((B,), nan, device=dev, dtype=dt)
    merr_out = torch.full((B,), nan, device=dev, dtype=dt)
    hw_final_out = torch.full((B,), nan, device=dev, dtype=dt)

    while not bool(done.all()):
        active = (~done).nonzero().flatten()                 # CPU indices
        nc_snapshot = n_cells.clone()                        # group on the state at round start

        for nc in torch.unique(nc_snapshot[active]).tolist():
            grp = active[nc_snapshot[active] == nc]          # CPU indices of this group
            grp_d = grp.to(dev)
            G = int(grp.numel())

            m_g = m[grp_d]
            u0_g = u0[grp_d]
            hw_g = half_width[grp_d]
            a_g, b_g = u0_g - hw_g, u0_g + hw_g
            quad = composite_quadrature(a_g, b_g, n_cells=nc, n_pts=params.n_pts,
                                        rule=params.quad_rule, device=dev, dtype=dt)

            lam0_init = None
            if use_lambda_init:
                hp = has_prev[grp_d]
                if bool(hp.any()):
                    default = _default_lam0_physical(m_g, use_initial_max, use_precond, use_varsigma)
                    lam0_init = torch.where(hp.unsqueeze(-1), lam_prev[grp_d], default)

            closure_g, lam_g, _, diag_g = maxent.compute_closure(
                entropy_policy, m_g, quad,
                use_precond=use_precond, use_adapt_base=use_adapt_base,
                use_initial_max=use_initial_max, use_varsigma=use_varsigma,
                strat=strat, tol=tol, maxit=maxit,
                record_history=record_history, verbose=verbose,
                check_realizable=False, lam0_init=lam0_init,
                grad_mode=grad_mode, raise_on_fail=False,
            )

            # ---- diagnostics: a failed / non-finite solve counts as a domain failure
            conv_g = diag_g.converged & torch.isfinite(lam_g).all(dim=-1)

            with torch.no_grad():
                inf = torch.full((G,), float("inf"), device=dev, dtype=dt)
                tail_ratio = _tail_ratio(entropy_policy, lam_g, quad)
                tail_ok = tail_ratio <= params.tail_rtol
                lead_ok = _leading_coeff_ok(lam_g)           # exact divergence certificate
                tail_ratio = torch.where(lead_ok, tail_ratio, inf)
                tail_ok = tail_ok & lead_ok

                _, moment_err, moments_ok = _verify_moments(entropy_policy, lam_g.detach(), m_g.detach(),
                                                            a_g.detach(), b_g.detach(), nc, params)

                tail_ratio = torch.where(conv_g, tail_ratio, inf)
                tail_ok = tail_ok & conv_g
                moment_err = torch.where(conv_g, moment_err, inf)
                moments_ok = moments_ok & conv_g

            # ---- per-sample decisions (CPU) --------------------------------
            conv_c, tail_ok_c, mom_ok_c = conv_g.cpu(), tail_ok.cpu(), moments_ok.cpu()
            tail_c, merr_c = tail_ratio.cpu(), moment_err.cpu()
            niter_c = diag_g.n_iter.cpu()

            ng, nr = n_growths[grp], n_refines[grp]
            round_idx[grp] += 1

            fail = ~conv_c
            grow_fail = fail & (ng < params.max_domain_growths)
            exhaust = fail & ~grow_fail
            grow = conv_c & ~tail_ok_c & (ng < params.max_domain_growths)
            refine = conv_c & ~grow & ~mom_ok_c & (nr < params.max_resolution_refinements)
            accept = conv_c & ~grow & ~refine

            # ---- round log ---------------------------------------------------
            a_l, b_l = a_g.detach().cpu().tolist(), b_g.detach().cpu().tolist()
            hw_l = hw_g.detach().cpu().tolist()
            hist = None
            if record_history and diag_g.lambda_history:
                hist = torch.stack(diag_g.lambda_history, dim=0)             # (T, G, K), detached
            for j in range(G):
                i = int(grp[j])
                if grow_fail[j] or grow[j]:
                    action = "grew_domain"
                elif refine[j]:
                    action = "refined_resolution"
                elif accept[j]:
                    action = "accepted"
                else:
                    action = "failed"
                ok_j = bool(conv_c[j])
                n_it = int(niter_c[j]) if ok_j else 0
                hist_j = [hist[t, j] for t in range(n_it)] if (hist is not None and ok_j) else []
                rounds[i].append(AdaptiveRound(
                    int(round_idx[i]), int(nc), hw_l[j], (a_l[j], b_l[j]),
                    ok_j, n_it, float(tail_c[j]), bool(tail_ok_c[j]),
                    float(merr_c[j]), bool(mom_ok_c[j]), action,
                    lambda_history=hist_j,
                    quad=Quadrature(quad.nodes[j], quad.weights[j]) if ok_j else None,
                ))

            if verbose:
                print(f"[adaptive] group n_cells={nc:5d} size={G:4d}: accepted={int(accept.sum())} "
                      f"grow={int((grow | grow_fail).sum())} refine={int(refine.sum())} "
                      f"failed={int(exhaust.sum())}")

            # ---- state updates ------------------------------------------------
            gf = grow_fail | grow
            n_growths[grp[gf]] += 1
            half_width = half_width.index_put(
                (grp_d,), torch.where(gf.to(dev), hw_g * params.domain_growth_factor, hw_g))

            n_cells[grp[grow]] = torch.ceil(n_cells[grp[grow]].double() * params.domain_growth_factor).long()
            n_cells[grp[refine]] = torch.ceil(n_cells[grp[refine]].double() * params.resolution_growth_factor).long()
            n_refines[grp[refine]] += 1

            done[grp[accept | exhaust]] = True
            failed[grp[exhaust]] = True

            if use_lambda_init:
                has_prev = has_prev.index_put((grp_d,), conv_g)
                lam_prev = lam_prev.index_put((grp_d[conv_g],), lam_g[conv_g])

            # ---- write accepted samples into the outputs (out-of-place, graph kept)
            if bool(accept.any()):
                acc_d = accept.to(dev)
                sel = grp_d[acc_d]
                lam_out = lam_out.index_put((sel,), lam_g[acc_d])
                closure_out = closure_out.index_put((sel,), closure_g[acc_d])
                niter_out = niter_out.index_put((sel,), diag_g.n_iter[acc_d])
                gnorm_out = gnorm_out.index_put((sel,), diag_g.grad_norm[acc_d])
                obj_out = obj_out.index_put((sel,), diag_g.obj[acc_d])
                tail_out = tail_out.index_put((sel,), tail_ratio[acc_d])
                merr_out = merr_out.index_put((sel,), moment_err[acc_d])
                hw_final_out = hw_final_out.index_put((sel,), hw_g.detach()[acc_d])
                for j in accept.nonzero().flatten().tolist():
                    i = int(grp[j])
                    n_cells_final[i] = int(nc)
                    quads_out[i] = Quadrature(quad.nodes[j], quad.weights[j])

    if raise_on_fail and bool(failed.any()):
        bad = failed.nonzero().flatten().tolist()
        raise RuntimeError(
            "solve_adaptive: Newton solve failed to converge even after "
            f"{params.max_domain_growths} domain growths for sample(s) {bad}. "
            "These moment sets may lie too close to the Junk line / boundary of "
            "moment space for a MaxEnt solution to exist at finite lambda."
        )

    converged = ~failed.to(dev)
    diag = SolveDiagnostics(converged=converged, n_iter=niter_out, grad_norm=gnorm_out, obj=obj_out)

    return AdaptiveResult(
        lam=lam_out, closure=closure_out, quads=quads_out, diag=diag, rounds=rounds,
        n_cells_final=n_cells_final.to(dev), half_width_final=hw_final_out,
        verified_moment_rel_err=merr_out, tail_ratio=tail_out, converged=converged,
    )
