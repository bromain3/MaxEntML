"""
maxent.py
=========

PyTorch port of closures/maxent/maxent.hpp (minus `build_flux_term`).

`MaxentDualSystem` is the analogue of the C++ `MaxentDualSystem<EntropyPolicy>`:
it owns the monomial design matrix Phi, the quadrature rule and the target
moments, and evaluates the dual objective / gradient / Hessian -- for a whole
batch of B independent moment sets at once:

    Phi      (B, K, Q)      targets  (B, K)      lam  (B, K)
    quad.nodes / quad.weights : (Q,) shared or (B, Q) per sample

Differences from the NumPy version (all to keep the autograd graph intact):

* No in-place mutation of Phi / targets. The Adaptive Change of Basis returns
  a *new* system (`adapt_basis`), and standardization keeps both the solve
  basis (`Phi`, `targets`) and the physical basis (`Phi_phys`,
  `targets_phys`) side by side, so nothing needs to be "restored" afterwards
  (no T^{-1} round trip). `system.physical()` gives the physical-basis view
  that the NumPy code returned after `solve()`.
* The dual functional is evaluated with analytic gradient / Hessian (as
  before); differentiability w.r.t. the *inputs* comes from the tensor ops
  around it and from the (unrolled or implicit) Newton solve.

`EntropyPolicy` is duck-typed: `eta_dual`, `eta_dual_prime`,
`eta_dual_prime_prime` acting on tensors; an optional `eval_all(s)` returning
the three at once saves redundant exponentials.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch

from .config import as_batched, as_tensor
from .quadrature import Quadrature, build_monomial_basis
from .conversion import compute_primitive_moments, raw_to_primitive_moment
from .realisability import check_realizability
from .initialisation import maxwellian_init, maxwellian_init_q0_r2, DEFAULT_HIGHER_MOMENT_LAMBDA
from .optimisation import (
    NewtonStrategy, HybridStrategy, solve_dual, standardization_matrix,
    varsigma_rescaling,
)

tol_default = 1e-10
maxit_default = 100
maxwellian_tol = 1.0e-6


def _eval_entropy(policy, s: torch.Tensor):
    """(eta*, eta*', eta*'') at s, using policy.eval_all when available."""
    if hasattr(policy, "eval_all"):
        return policy.eval_all(s)
    return policy.eta_dual(s), policy.eta_dual_prime(s), policy.eta_dual_prime_prime(s)


def _apply_Tt(T: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """T^T @ lam for T (B, K, K), lam (B, K)."""
    return torch.einsum("bki,bk->bi", T, lam)


def _as_lam(lam, B: int, like: torch.Tensor) -> torch.Tensor:
    """Multiplier vector(s) -> (B, K) tensor on like's device/dtype."""
    lam = as_tensor(lam, like.device, like.dtype)
    if lam.ndim == 1:
        lam = lam.unsqueeze(0).expand(B, -1)
    return lam


@dataclass
class MaxentDualSystem:
    """Port of `maxent::MaxentDualSystem<EntropyPolicy>` (batched, functional)."""

    Phi: torch.Tensor                       # (B, K, Q) design matrix in the *solve* basis
    quad: Quadrature                        # nodes / weights, (Q,) or (B, Q)
    targets: torch.Tensor                   # (B, K)    target moments in the solve basis
    entropy_policy: type
    Phi_phys: Optional[torch.Tensor] = None       # (B, K, Q) plain monomials in v
    targets_phys: Optional[torch.Tensor] = None   # (B, K)    raw target moments
    T_precond: Optional[torch.Tensor] = None      # (B, K, K) standardization (lower-triangular) or None
    lambda_eps: float = 1.0e-15

    def __post_init__(self):
        if self.Phi_phys is None:
            self.Phi_phys = self.Phi
        if self.targets_phys is None:
            self.targets_phys = self.targets

    @property
    def precond_active(self) -> bool:
        return self.T_precond is not None

    def is_feasible(self, lam_opt: torch.Tensor) -> torch.Tensor:
        """(B,) bool: last multiplier must stay negative."""
        return lam_opt[..., -1] < -self.lambda_eps

    # ------------------------------------------------------------------
    # Adaptive Change of Basis (Alldredge ACB) -- functional
    # ------------------------------------------------------------------
    def adapt_basis(self, L: torch.Tensor) -> "MaxentDualSystem":
        """
        Rotate (Phi, targets) into the Cholesky-whitened basis defined by the
        lower-triangular factors L (B, K, K); returns a NEW system.
        """
        Phi = torch.linalg.solve_triangular(L, self.Phi, upper=False)
        targets = torch.linalg.solve_triangular(L, self.targets.unsqueeze(-1), upper=False).squeeze(-1)
        return dataclasses.replace(self, Phi=Phi, targets=targets)

    # ------------------------------------------------------------------
    # Objective only (fast path for line search)
    # ------------------------------------------------------------------
    def evaluate_obj(self, lam: torch.Tensor) -> torch.Tensor:
        s = torch.einsum("bkq,bk->bq", self.Phi, lam)
        f = self.entropy_policy.eta_dual(s)
        return (self.quad.weights * f).sum(dim=-1) - (lam * self.targets).sum(dim=-1)

    # ------------------------------------------------------------------
    # Full evaluation: objective (B,), gradient (B, K), Hessian (B, K, K)
    # ------------------------------------------------------------------
    def evaluate(self, lam: torch.Tensor):
        Phi, w = self.Phi, self.quad.weights
        s = torch.einsum("bkq,bk->bq", Phi, lam)
        f, fp, fpp = _eval_entropy(self.entropy_policy, s)

        obj = (w * f).sum(dim=-1) - (lam * self.targets).sum(dim=-1)
        grad = torch.einsum("bkq,bq->bk", Phi, w * fp) - self.targets

        hw = w * fpp
        hess = (Phi * hw.unsqueeze(-2)) @ Phi.transpose(-1, -2)
        hess = 0.5 * (hess + hess.transpose(-1, -2))
        return obj, grad, hess

    # ------------------------------------------------------------------
    # Newton solve (returns lam in the PHYSICAL basis)
    # ------------------------------------------------------------------
    def solve(self, strat=None, lam0: Optional[torch.Tensor] = None,
              tol: float = tol_default, maxit: int = maxit_default,
              use_adaptive: bool = True, record_history: bool = False,
              verbose: bool = False, grad_mode: str = "unroll", raise_on_fail: bool = True):
        if strat is None:
            strat = HybridStrategy()

        K = self.targets.shape[-1]
        lam, diag = solve_dual(self, K, strat, tol, maxit, lam0, use_adaptive,
                               verbose, record_history, grad_mode, raise_on_fail)

        if self.T_precond is not None:
            lam = _apply_Tt(self.T_precond, lam)
            if record_history and diag.lambda_history:
                diag.lambda_history = [_apply_Tt(self.T_precond, lh) for lh in diag.lambda_history]

        return lam, diag

    def physical(self) -> "MaxentDualSystem":
        """The same problem expressed in the plain monomial basis (no standardization)."""
        return dataclasses.replace(self, Phi=self.Phi_phys, targets=self.targets_phys, T_precond=None)

    # ------------------------------------------------------------------
    # Closing moment: int v^n_mom * f_ansatz(v) dv   (lam in physical basis)
    # ------------------------------------------------------------------
    def reconstruct(self, lam: torch.Tensor, n_mom: int) -> torch.Tensor:
        s = torch.einsum("bkq,bk->bq", self.Phi_phys, lam)
        fp = self.entropy_policy.eta_dual_prime(s)
        return (self.quad.weights * self.quad.nodes ** n_mom * fp).sum(dim=-1)

    def ansatz(self, lam: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return ansatz(self.entropy_policy, lam, v)


def ansatz(entropy_policy, lam: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Reconstructed kinetic ansatz f(v) = eta*'( sum_k lam_k v^k ) at arbitrary
    points. lam (B, K) or (K,); v (P,) shared or (B, P). Returns (B, P) / (P,).
    """
    lam = as_tensor(lam, getattr(lam, "device", None), getattr(lam, "dtype", None))
    v = as_tensor(v, lam.device, lam.dtype)
    Phi_v = build_monomial_basis(v, lam.shape[-1])
    s = torch.einsum("...kp,...k->...p", Phi_v, lam)
    return entropy_policy.eta_dual_prime(s)


# ============================================================================
# Factory function
# ============================================================================

def make_system(entropy_policy, m_orig: torch.Tensor, quad: Quadrature,
                use_precond: bool = False, use_varsigma: bool = False) -> MaxentDualSystem:
    """
    Port of `maxent::make_system<EntropyPolicy>`, batched: `m_orig` is (B, K)
    (or (K,), treated as B = 1). With `use_precond` the system is expressed in
    the (v - u)/v_th standardized basis (optionally further rescaled by
    `varsigma`, see `optimisation.varsigma_rescaling`); both bases are kept.
    """
    m = as_batched(m_orig, getattr(m_orig, "device", None), getattr(m_orig, "dtype", None))
    B, K = m.shape

    Phi_phys = build_monomial_basis(quad.nodes, K)
    if Phi_phys.ndim == 2:
        Phi_phys = Phi_phys.unsqueeze(0).expand(B, -1, -1)
    elif Phi_phys.shape[0] != B:
        Phi_phys = Phi_phys.expand(B, -1, -1)         # per-sample rule with leading dim 1

    T = None
    if use_precond:
        rho = m[:, 0]
        u = m[:, 1] / rho
        p = raw_to_primitive_moment(m, 2)
        v_th = torch.sqrt(rho * p)

        varsigma = varsigma_rescaling(K) if use_varsigma else 1.0
        T = standardization_matrix(u, v_th, K, varsigma)              # (B, K, K)
        Phi = T @ Phi_phys
        targets = torch.einsum("bkj,bj->bk", T, m)
    elif use_varsigma:
        raise ValueError(
            "make_system: use_varsigma=True requires use_precond=True -- "
            "the varsigma rescaling further scales the *standardized* "
            "velocity variable, so bulk-velocity/thermal-speed "
            "standardization must already be active."
        )
    else:
        Phi, targets = Phi_phys, m

    return MaxentDualSystem(Phi=Phi, quad=quad, targets=targets, entropy_policy=entropy_policy,
                            Phi_phys=Phi_phys, targets_phys=m, T_precond=T)


# ============================================================================
# Physical-basis lambda -> system-basis lambda
# ============================================================================

def to_system_basis(system: MaxentDualSystem, lam_phys: torch.Tensor) -> torch.Tensor:
    """
    Physical-basis multipliers -> the basis the system solves in:
    lam_sys = T^{-T} lam_phys (a triangular solve; identity without
    standardization).
    """
    B = system.targets.shape[0]
    lam_phys = _as_lam(lam_phys, B, system.targets)
    if system.T_precond is None:
        return lam_phys.clone()
    return torch.linalg.solve_triangular(system.T_precond.transpose(-1, -2),
                                         lam_phys.unsqueeze(-1), upper=True).squeeze(-1)


def physical_maxwellian_q0r2_init(m_orig: torch.Tensor,
                                  default_lambda: float = DEFAULT_HIGHER_MOMENT_LAMBDA,
                                  varsigma: float = 1.0) -> torch.Tensor:
    """
    Physical-basis equivalent of `maxwellian_init_q0_r2`: T^T @ lam0_std for
    each sample's own (u, v_th). Shape (B, K).
    """
    m = as_batched(m_orig, getattr(m_orig, "device", None), getattr(m_orig, "dtype", None))
    K = m.shape[-1]

    lam0_std = maxwellian_init_q0_r2(K, default_lambda=default_lambda, varsigma=varsigma,
                                     device=m.device, dtype=m.dtype)

    rho = m[:, 0]
    u = m[:, 1] / rho
    p = raw_to_primitive_moment(m, 2)
    v_th = torch.sqrt(rho * p)

    T = standardization_matrix(u, v_th, K, varsigma)
    return torch.einsum("bki,k->bi", T, lam0_std)


# ============================================================================
# Moment -> reconstruction parameters (i.e. solve for lambda)
# ============================================================================

def m_to_reconstruction_params(entropy_policy, m_cell: torch.Tensor, quad: Quadrature,
                               use_precond: bool = False, use_adapt_base: bool = False,
                               use_initial_max: bool = False, use_varsigma: bool = False,
                               strat=None, tol: float = tol_default, maxit: int = maxit_default,
                               record_history: bool = False, verbose: bool = False,
                               lam0_init: Optional[torch.Tensor] = None,
                               grad_mode: str = "unroll", raise_on_fail: bool = True):
    """
    Port of `maxent::m_to_reconstruction_params`, batched.

    Returns (lam (B, K), system, diagnostics); `lam` is in the physical
    monomial basis and `system` is the physical-basis view of the problem.

    Initial guess: `lam0_init` (physical basis, (B, K) or (K,)) if given, else
    the physical Maxwellian fit (`use_initial_max`), else the (q*=0, r*=2)
    standardized guess.

    `grad_mode` ("unroll" | "implicit") and `raise_on_fail` are forwarded to
    `optimisation.solve_dual`.
    """
    m = as_batched(m_cell, getattr(m_cell, "device", None), getattr(m_cell, "dtype", None))
    B, K = m.shape

    system = make_system(entropy_policy, m, quad, use_precond, use_varsigma)

    if lam0_init is not None:
        lam0 = to_system_basis(system, lam0_init)
    elif use_initial_max:
        lam0 = to_system_basis(system, maxwellian_init(m))
    elif use_precond:
        varsigma = varsigma_rescaling(K) if use_varsigma else 1.0
        lam0 = maxwellian_init_q0_r2(K, varsigma=varsigma, device=m.device, dtype=m.dtype)
        lam0 = lam0.unsqueeze(0).expand(B, K)
    else:
        lam0 = physical_maxwellian_q0r2_init(m)

    lam, diag = system.solve(strat, lam0, tol, maxit, use_adapt_base,
                             record_history=record_history, verbose=verbose,
                             grad_mode=grad_mode, raise_on_fail=raise_on_fail)

    return lam, system.physical(), diag


# ============================================================================
# Full closure computation (invert + reconstruct closing moment)
# ============================================================================

def compute_closure(entropy_policy, m_cell: torch.Tensor, quad: Quadrature,
                    use_precond: bool = False, use_adapt_base: bool = False,
                    use_initial_max: bool = False, use_varsigma: bool = False,
                    strat=None, tol: float = tol_default, maxit: int = maxit_default,
                    record_history: bool = False, verbose: bool = False,
                    check_realizable: bool = True,
                    lam0_init: Optional[torch.Tensor] = None,
                    grad_mode: str = "unroll", raise_on_fail: bool = True):
    """
    Port of `maxent::compute_closure<EntropyPolicy>`, batched.

    Returns (closure (B,), lam (B, K), system, diagnostics). `closure` and
    `lam` are differentiable w.r.t. `m_cell` (and the quadrature, if it was
    built from tensors that require grad).
    """
    m = as_batched(m_cell, getattr(m_cell, "device", None), getattr(m_cell, "dtype", None))
    K = m.shape[-1]

    if check_realizable:
        c_mom = compute_primitive_moments(m)
        check_realizability(c_mom, is_primitive=True, throw_on_error=True)

    lam, system, diag = m_to_reconstruction_params(
        entropy_policy, m, quad, use_precond, use_adapt_base, use_initial_max,
        use_varsigma, strat, tol, maxit, record_history, verbose,
        lam0_init=lam0_init, grad_mode=grad_mode, raise_on_fail=raise_on_fail,
    )

    closure = system.reconstruct(lam, K)
    return closure, lam, system, diag
