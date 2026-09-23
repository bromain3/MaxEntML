"""
levermore.py
============

PyTorch port of closures/levermore/levermore.hpp.

Boltzmann/Levermore entropy  eta(f) = f ln f - f  with Legendre dual

    eta*(s) = (eta*)'(s) = (eta*)''(s) = exp(s),

i.e. the classical MaxEnt closure whose reconstructed distribution is

    f(v) = exp( lambda_0 + lambda_1 v + ... + lambda_{n-1} v^{n-1} ).
"""

from __future__ import annotations

import torch

from . import maxent


class LevermoreEntropy:
    """Port of `struct LevermoreEntropy`; tensor-in / tensor-out static methods."""

    @staticmethod
    def eta_dual(s: torch.Tensor) -> torch.Tensor:
        return torch.exp(s)

    @staticmethod
    def eta_dual_prime(s: torch.Tensor) -> torch.Tensor:
        return torch.exp(s)

    @staticmethod
    def eta_dual_prime_prime(s: torch.Tensor) -> torch.Tensor:
        return torch.exp(s)

    @staticmethod
    def eval_all(s: torch.Tensor):
        """(eta*, eta*', eta*'') sharing a single exponential."""
        e = torch.exp(s)
        return e, e, e


def compute_closure(m_cell, quad, use_precond=False, use_adapt_base=False,
                    use_varsigma=False, strat=None, tol=1e-10, maxit=100,
                    record_history=False, verbose=False, check_realizable=True, **kwargs):
    """
    Port of `maxent::levermore::compute_closure` (batched). Extra keyword
    arguments (`use_initial_max`, `lam0_init`, `grad_mode`, `raise_on_fail`)
    are forwarded to `maxent.compute_closure`.
    """
    return maxent.compute_closure(LevermoreEntropy, m_cell, quad,
                                  use_precond=use_precond, use_adapt_base=use_adapt_base,
                                  use_varsigma=use_varsigma, strat=strat,
                                  tol=tol, maxit=maxit, record_history=record_history,
                                  verbose=verbose, check_realizable=check_realizable, **kwargs)


def m_to_reconstruction_params(m_cell, quad, use_precond=False, use_adapt_base=False,
                               use_varsigma=False, strat=None, tol=1e-10, maxit=100,
                               record_history=False, verbose=False, **kwargs):
    """Port of `maxent::levermore::m_to_reconstruction_params` (batched)."""
    return maxent.m_to_reconstruction_params(LevermoreEntropy, m_cell, quad,
                                             use_precond=use_precond, use_adapt_base=use_adapt_base,
                                             use_varsigma=use_varsigma, strat=strat,
                                             tol=tol, maxit=maxit, record_history=record_history,
                                             verbose=verbose, **kwargs)
