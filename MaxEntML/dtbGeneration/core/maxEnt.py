"""
maxent_lambdas.py
=================

Multiplicateurs de Lagrange (lambda_0 ... lambda_4) de la fermeture MaxEnt / Levermore
(entropie de Boltzmann, f(v) = exp(sum_k lambda_k v^k)) sur une grille de points (q*, r*).

Un point (q*, r*) est un état standardisé (rho, u, p, q, r) = (1, 0, 1, q*, r*),
c'est-à-dire de moments bruts m = (1, 0, 1, q*, r*). Les lambdas renvoyés sont ceux de
ce repère standardisé.

Usage
-----
    def uv_to_qr(u, v):        # u, v : tenseurs (Nu, Nv), valeurs dans [0, 1]
        q = ...                # tenseur (Nu, Nv)
        r = ...                # tenseur (Nu, Nv)
        return q, r

    lam = compute_lambdas(Nu, Nv, uv_to_qr, device="cuda", dtype=torch.float64)   # (Nu, Nv, 5)

Les paramètres du passage uv -> qr vivent dans `uv_to_qr` (closure, functools.partial, ...).
Les points non réalisables ou dont la résolution échoue sont renvoyés en NaN
(`torch.isnan(lam).any(-1)` donne le masque des échecs).
Tout est calculé sous torch.no_grad().
"""

import math

import numpy as np
import torch
import gc

# ----------------------------------------------------------------------------
# Constantes numériques (valeurs de l'implémentation d'origine)
# ----------------------------------------------------------------------------
NMOM = 5                       # moments (1, 0, 1, q*, r*)  ->  lambda_0 ... lambda_4

# Newton
TOL, MAXIT = 1e-10, 200
MAX_STEP = 1.5                 # norme maximale d'un pas de Newton
SWITCH_TOL = 1e-3              # |grad| > SWITCH_TOL : pas amorti (Armijo), sinon pas complet
ARMIJO_C1, ARMIJO_RHO, ARMIJO_MAX = 1e-4, 0.5, 20

# Quadrature adaptative : Gauss-Legendre composite sur [-hw, hw]
N_PTS, N_CELLS0, K_SIGMA0 = 5, 40, 8.0          # points/cellule, cellules initiales, hw initial
TAIL_RTOL = 1e-10                               # f(bord) / f(max) toléré
MOMENT_RTOL, MOMENT_ATOL = 1e-8, 1e-10          # vérification des moments (règle plus fine)
VERIFY_CELL_FACTOR, VERIFY_N_PTS = 4, 8
DOMAIN_GROWTH, MAX_GROWTHS = 1.5, 8
RESOLUTION_GROWTH, MAX_REFINES = 2.0, 6

# Initialisation : maxwellienne (rho = 1, u = 0, theta = 1)
LAM_INIT = (-0.5 * math.log(2.0 * math.pi), 0.0, -0.5, -1e-8, -1e-8)


# ----------------------------------------------------------------------------
# Quadrature et base monomiale
# ----------------------------------------------------------------------------

def _composite_gl(a, b, n_cells, n_pts):
    """Gauss-Legendre composite sur [a, b] (a, b : (G,)) -> nodes, weights : (G, n_cells * n_pts)."""
    x, w = np.polynomial.legendre.leggauss(n_pts)
    x = torch.as_tensor(x, device=a.device, dtype=a.dtype)
    w = torch.as_tensor(w, device=a.device, dtype=a.dtype)

    t = torch.linspace(0.0, 1.0, n_cells + 1, device=a.device, dtype=a.dtype)
    edges = a[:, None] + (b - a)[:, None] * t                    # (G, n_cells + 1)
    lo, hi = edges[:, :-1], edges[:, 1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)                 # (G, n_cells)

    nodes = (mid[..., None] + half[..., None] * x).flatten(-2)
    weights = (half[..., None] * w).flatten(-2)
    return nodes, weights


def _monomials(x):
    """x : (G, Q) -> Phi : (G, NMOM, Q), Phi[g, k, q] = x[g, q]**k."""
    return torch.stack([torch.ones_like(x)] + [x] * (NMOM - 1), dim=1).cumprod(dim=1)


# ----------------------------------------------------------------------------
# Problème dual : J(lam) = sum_q w_q exp(lam . Phi_q) - lam . m
# ----------------------------------------------------------------------------

def _objective(Phi, w, m, lam):
    f = torch.exp(torch.einsum("gkq,gk->gq", Phi, lam))
    return (w * f).sum(-1) - (lam * m).sum(-1)


def _evaluate(Phi, w, m, lam):
    """Objectif (G,), gradient (G, K), hessien (G, K, K)."""
    f = torch.exp(torch.einsum("gkq,gk->gq", Phi, lam))
    wf = w * f
    obj = wf.sum(-1) - (lam * m).sum(-1)
    grad = torch.einsum("gkq,gq->gk", Phi, wf) - m
    hess = (Phi * wf.unsqueeze(-2)) @ Phi.transpose(-1, -2)
    return obj, grad, 0.5 * (hess + hess.transpose(-1, -2))


def _newton_direction(grad, hess):
    """
    Direction de Newton par échantillon : Cholesky si le hessien est SPD, sinon
    Levenberg-Marquardt (5 niveaux, x10 à chaque fois) jusqu'à une vraie direction
    de descente, sinon -grad.
    """
    rhs = -grad.unsqueeze(-1)
    L, info = torch.linalg.cholesky_ex(hess)
    d_chol = torch.cholesky_solve(rhs, L).squeeze(-1)
    ok = (info == 0) & torch.isfinite(d_chol).all(dim=-1)
    if ok.all():
        return d_chol

    gnorm = torch.linalg.vector_norm(grad, dim=-1)
    damping = 1e-6 * torch.diagonal(hess, dim1=-2, dim2=-1).amax(dim=-1)
    damping = torch.where(damping > 0, damping, torch.full_like(damping, 1e-7))
    eye = torch.eye(NMOM, device=grad.device, dtype=grad.dtype)

    d_fb = -grad                                     # dernier recours : plus forte pente
    found = torch.zeros_like(ok)
    for _ in range(5):
        d, info_lm = torch.linalg.solve_ex(hess + damping.view(-1, 1, 1) * eye, rhs)
        d = d.squeeze(-1)
        descent = (grad * d).sum(-1) < -1e-12 * gnorm * torch.linalg.vector_norm(d, dim=-1)
        good = (info_lm == 0) & torch.isfinite(d).all(dim=-1) & descent & ~found
        d_fb = torch.where(good[:, None], d, d_fb)
        found = found | good
        damping = damping * 10.0

    return torch.where(ok[:, None], d_chol, d_fb)


def _hybrid_step(Phi, w, m, lam, d, grad, obj0):
    """lam + alpha * d : Armijo tant que |grad| > SWITCH_TOL, pas complet (alpha = 1) ensuite."""
    use_ls = torch.linalg.vector_norm(grad, dim=-1) > SWITCH_TOL
    if not use_ls.any():
        return lam + d

    slope0 = (grad * d).sum(-1)
    valid = (slope0 < 0) & torch.isfinite(slope0)
    alpha = torch.ones_like(slope0)
    done = ~valid
    for _ in range(ARMIJO_MAX):
        obj_try = _objective(Phi, w, m, lam + alpha[:, None] * d)
        ok = torch.isfinite(obj_try) & (obj_try <= obj0 + ARMIJO_C1 * alpha * slope0)
        done = done | ok
        alpha = torch.where(done, alpha, ARMIJO_RHO * alpha)
        if done.all():
            break

    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    alpha = torch.where(use_ls, alpha, torch.ones_like(alpha))
    return lam + alpha[:, None] * d


def _newton(Phi, w, m, lam):
    """Newton batché avec masques de convergence par échantillon. Renvoie (lam, converged)."""
    G, dev, dt = m.shape[0], m.device, m.dtype
    tol = max(TOL, 100.0 * torch.finfo(dt).eps)      # pas plus fin que ce que le dtype résout
    tol_relaxed = max(1e3 * tol, 1e-6)

    active = torch.ones(G, dtype=torch.bool, device=dev)
    converged = torch.zeros(G, dtype=torch.bool, device=dev)
    stall = torch.zeros(G, dtype=torch.long, device=dev)
    gn_prev = torch.full((G,), float("inf"), device=dev, dtype=dt)

    for _ in range(MAXIT):
        obj, grad, hess = _evaluate(Phi, w, m, lam)
        gn = torch.linalg.vector_norm(grad, dim=-1)

        hit = active & (gn < tol)
        bad = active & ~torch.isfinite(gn)
        converged |= hit
        active &= ~hit & ~bad

        # stagnation : gradient qui ne baisse plus mais déjà petit -> accepté
        stalled = active & (gn > 0.99 * gn_prev)
        stall = torch.where(stalled, stall + 1, torch.zeros_like(stall))
        accept = stalled & (stall >= 5) & (gn < tol_relaxed)
        converged |= accept
        active &= ~accept
        gn_prev = torch.where(active, gn, gn_prev)

        if not active.any():
            break

        d = _newton_direction(grad, hess)
        d = d * torch.clamp(MAX_STEP / torch.linalg.vector_norm(d, dim=-1, keepdim=True), max=1.0)
        lam_new = _hybrid_step(Phi, w, m, lam, d, grad, obj)
        lam = torch.where(active[:, None], lam_new, lam)

    return lam, converged


# ----------------------------------------------------------------------------
# Résolution adaptative pour un lot de points (q*, r*)
# ----------------------------------------------------------------------------

def _solve(q, r):
    torch.cuda.empty_cache()
    gc.collect()
    """lambdas (B, NMOM) pour B points (q*, r*) ; NaN si non réalisable ou non convergé."""
    dev, dt = q.device, q.dtype
    B = q.numel()
    eps = torch.finfo(dt).eps
    atol = max(MOMENT_ATOL, 100.0 * eps)
    rtol = max(MOMENT_RTOL, 100.0 * eps)

    m = torch.stack([torch.ones_like(q), torch.zeros_like(q), torch.ones_like(q), q, r], dim=-1)
    lam_out = torch.full((B, NMOM), float("nan"), device=dev, dtype=dt)

    # Réalisabilité : r* > 1 + q*^2 (intérieur de Hankel) et hors "Junk line" (q* = 0, r* > 3)
    junk = (q.abs() <= 10.0 * eps) & (r - 3.0 > 10.0 * eps * (1.0 + r.abs()))
    todo = ((r > 1.0 + q * q) & ~junk).cpu()

    # État de contrôle par échantillon (CPU : il ne sert qu'au flot de contrôle Python)
    n_cells = torch.full((B,), N_CELLS0, dtype=torch.long)
    half_width = torch.full((B,), K_SIGMA0, dtype=torch.float64)   # u = 0, sqrt(p) = 1
    n_grow = torch.zeros(B, dtype=torch.long)
    n_refine = torch.zeros(B, dtype=torch.long)

    while todo.any():
        pending = todo.nonzero().flatten()
        cells_at_start = n_cells.clone()             # groupes définis sur l'état de début de tour

        # un solve batché par nombre de cellules (seule quantité qui change les shapes)
        for nc in torch.unique(cells_at_start[pending]).tolist():
            idx = pending[cells_at_start[pending] == nc]
            idx_dev = idx.to(dev)
            m_g = m[idx_dev]
            hw = half_width[idx].to(device=dev, dtype=dt)

            # 1. dual sur la quadrature de ce tour
            nodes, weights = _composite_gl(-hw, hw, nc, N_PTS)
            Phi = _monomials(nodes)
            lam0 = torch.tensor(LAM_INIT, device=dev, dtype=dt).expand(len(idx), NMOM)
            lam, conv = _newton(Phi, weights, m_g, lam0)
            conv = conv & torch.isfinite(lam).all(dim=-1)

            # 2. domaine assez grand ? décroissance de f aux bords + lambda_4 < 0
            f = torch.exp(torch.einsum("gkq,gk->gq", Phi, lam))
            f_max = f.amax(dim=-1)
            f_edge = torch.maximum(f[:, 0], f[:, -1])
            good = torch.isfinite(f_max) & (f_max > 0.0)
            ratio = f_edge / torch.where(good, f_max, torch.ones_like(f_max))
            tail_ok = conv & good & (ratio <= TAIL_RTOL) & (lam[:, -1] < -1e-15)

            # 3. résolution suffisante ? moments recalculés sur une règle plus fine
            vn, vw = _composite_gl(-hw, hw, VERIFY_CELL_FACTOR * nc, VERIFY_N_PTS)
            Phi_v = _monomials(vn)
            f_v = torch.exp(torch.einsum("gkq,gk->gq", Phi_v, lam))
            m_re = torch.einsum("gkq,gq->gk", Phi_v, vw * f_v)
            moments_ok = conv & ((m_re - m_g).abs() <= atol + rtol * m_g.abs()).all(dim=-1)

            # 4. décisions par échantillon (CPU)
            conv, tail_ok, moments_ok = conv.cpu(), tail_ok.cpu(), moments_ok.cpu()
            can_grow = n_grow[idx] < MAX_GROWTHS
            can_refine = n_refine[idx] < MAX_REFINES

            grow_fail = ~conv & can_grow             # non convergé : on élargit le domaine
            exhausted = ~conv & ~can_grow            # non convergé, budget épuisé : abandon (NaN)
            grow = conv & ~tail_ok & can_grow        # queue non décrue : domaine + cellules
            refine = conv & ~grow & ~moments_ok & can_refine
            accept = conv & ~grow & ~refine

            widen = grow | grow_fail
            n_grow[idx[widen]] += 1
            half_width[idx[widen]] *= DOMAIN_GROWTH
            n_cells[idx[grow]] = torch.ceil(n_cells[idx[grow]].double() * DOMAIN_GROWTH).long()
            n_cells[idx[refine]] = torch.ceil(n_cells[idx[refine]].double() * RESOLUTION_GROWTH).long()
            n_refine[idx[refine]] += 1
            todo[idx[accept | exhausted]] = False

            acc = accept.to(dev)
            lam_out[idx_dev[acc]] = lam[acc]

    return lam_out


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

@torch.no_grad()
def compute_lambdas(q, r, batch_size=2048):
    """
    Lambdas MaxEnt sur une grille (Nu, Nv) de points (q*, r*).

    uv_to_qr(u, v) -> (q, r) : u, v sont les grilles régulières sur [0, 1] (indexation "ij",
    tenseurs (Nu, Nv) sur `device`/`dtype`) ; q, r doivent être des tenseurs (Nu, Nv).
    batch_size : nombre max de points résolus simultanément (borne la mémoire).

    Renvoie un tenseur (Nu, Nv, 5) : lambda_0 ... lambda_4 (NaN si le point est
    non réalisable ou si la résolution échoue).
    """
    #TODO: ajouter comme argument un compute device comme ça on peut stocker sur cpu et calculer sur gpu et une bare de progès tqdm pour afficher l'avancement
    lam = torch.cat([_solve(q[i:i + batch_size], r[i:i + batch_size])
                     for i in range(0, q.numel(), batch_size)])
    return lam