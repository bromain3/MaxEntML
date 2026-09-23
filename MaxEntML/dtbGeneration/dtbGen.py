from MaxEntML.dtbGeneration.core.maxEnt import compute_lambdas
from MaxEntML.dataRepresentation.serie import serie, agregateColumns

import math
import torch
from scipy.stats import qmc


def _q_of_u(u, q_max, beta_q):
    if beta_q == 0:
        return q_max * u
    return q_max * torch.sinh(beta_q * u) / math.sinh(beta_q)


def _r_of_qv(q, v, r_max, eps, gamma_r):
    r_low = 1.0 + q**2 + eps
    return r_low + (r_max - r_low) * v**gamma_r


def _check_qr_params(q_max, r_max, eps):
    if r_max <= 1.0 + q_max ** 2 + eps:
        raise ValueError(
            f"r_max={r_max} doit être > 1 + q_max^2 + eps = {1.0 + q_max ** 2 + eps}"
        )

def _label(i, d):
    cols = list(d.colName)
    return f"#{i} [{cols[0]}]" if len(cols) == 1 else f"#{i} [{cols[0]} ... {cols[-1]}]"


def _uv_to_qr(u, v, q_max, r_max, eps, beta_q, gamma_r):
    q = _q_of_u(u, q_max, beta_q)
    r = _r_of_qv(q, v, r_max, eps, gamma_r)
    return q.reshape(-1), r.reshape(-1)


def _make_qr_grid(q_max, r_max, eps, beta_q, gamma_r, Nu, Nv, dtype=None, device=None):
    _check_qr_params(q_max, r_max, eps)

    u = torch.linspace(-1.0, 1.0, Nu, dtype=dtype, device=device)  # Nu impair -> q* = 0 exact
    v = torch.linspace(0.0, 1.0, Nv, dtype=dtype, device=device)
    V, U = torch.meshgrid(v, u, indexing="ij")  # (Nv, Nu) : u varie le plus vite

    return _uv_to_qr(U, V, q_max, r_max, eps, beta_q, gamma_r)


def _halton_1d(n, base, start=1):
    i = torch.arange(start, start + n, dtype=torch.int64)
    f = 1.0
    out = torch.zeros(n, dtype=torch.float64)
    while bool((i > 0).any()):
        f /= base
        out += f * (i % base).to(torch.float64)
        i = i // base
    return out


def _make_qr_halton(q_max, r_max, eps, beta_q, gamma_r, Nsamples,
                    dtype=None, device=None, scramble=False, seed=None):
    _check_qr_params(q_max, r_max, eps)

    dtype = dtype if dtype is not None else torch.get_default_dtype()

    sampler = qmc.Halton(d=2, scramble=scramble, seed=seed)
    if not scramble:
        sampler.fast_forward(1)  # saute le point 0 = (0, 0), coin du domaine
    pts = sampler.random(Nsamples)  # (Nsamples, 2) float64 dans [0, 1)^2

    pts = torch.from_numpy(pts).to(dtype=dtype, device=device)
    u = 2.0 * pts[:, 0] - 1.0  # [0, 1] -> [-1, 1]
    v = pts[:, 1]

    return _uv_to_qr(u, v, q_max, r_max, eps, beta_q, gamma_r)


def _build_database(q, r, batch_size=2000, storingDtype=None, storingDevice=None, computeDtype=None, computeDevice=None):
    lambdas = compute_lambdas(q, r, batch_size=batch_size, storingDtype=storingDtype, storingDevice=storingDevice, compute_dtype=computeDtype, compute_device=computeDevice)
    lambdas = serie(lambdas, colName=[f"l{i}" for i in range(lambdas.shape[1])],
                    storingDtype=storingDtype, storingDevice=storingDevice)

    q = serie(q, colName=['q'], storingDtype=storingDtype, storingDevice=storingDevice)
    r = serie(r, colName=['r'], storingDtype=storingDtype, storingDevice=storingDevice)
    qr = agregateColumns([q, r])

    return qr, lambdas


def generateTargets(qr, batch_size=2000, computeDtype=None, computeDevice=None):
    lambdas = compute_lambdas(qr['q'].data, qr['r'].data, batch_size=batch_size, compute_dtype=computeDtype, compute_device=computeDevice)
    return serie(lambdas, colName=[f'l{i}' for i in range(lambdas.shape[1])],
                 storingDtype=qr.storingDtype, storingDevice=qr.storingDevice)


def generateTrainingDatabase(qr2xyParams, Nx, Ny, batch_size=2000, storingDtype=None, storingDevice=None, computeDtype=None, computeDevice=None):
    q_max, r_max, eps, beta_q, gamma_r = qr2xyParams.q_max, qr2xyParams.r_max, qr2xyParams.eps, qr2xyParams.beta_q, qr2xyParams.gamma_r
    q, r = _make_qr_grid(q_max, r_max, eps, beta_q, gamma_r, Nx, Ny, dtype=storingDtype, device=storingDevice)
    return _build_database(q, r, batch_size, storingDtype, storingDevice, computeDtype, computeDevice)


def generateValidationDatabase(qr2xyParams, Nsamples, batch_size=2000, storingDtype=None, storingDevice=None, computeDtype=None, computeDevice=None, scramble=True, seed=None):
    q_max, r_max, eps, beta_q, gamma_r = qr2xyParams.q_max, qr2xyParams.r_max, qr2xyParams.eps, qr2xyParams.beta_q, qr2xyParams.gamma_r
    q, r = _make_qr_halton(q_max, r_max, eps, beta_q, gamma_r, Nsamples, dtype=storingDtype, device=storingDevice, scramble=scramble, seed=seed)
    return _build_database(q, r, batch_size, storingDtype, storingDevice, computeDtype, computeDevice)


def clean(*datasets, checkOn=None, verbose=False):

    if len(datasets) == 0:
        raise ValueError("Au moins un dataset est requis.")

    nSamples = [d.data.shape[0] for d in datasets]
    if len(set(nSamples)) != 1:
        raise ValueError(f"Nombres d'échantillons incohérents : {nSamples}")
    N = nSamples[0]
    if N == 0:
        raise ValueError("Les datasets sont vides.")

    if checkOn is None:
        checkOn = list(range(len(datasets)))
    else:
        checkOn = list(checkOn)
        invalid = [i for i in checkOn if not 0 <= i < len(datasets)]
        if invalid:
            raise IndexError(f"checkOn contient des indices hors limites : {invalid} ({len(datasets)} datasets)")

    device = datasets[0].data.device
    keep = torch.ones(N, dtype=torch.bool, device=device)
    nBadPerDataset = {}

    for i in checkOn:
        # True si toutes les valeurs de la ligne sont finies (ni inf ni nan)
        finiteRows = torch.isfinite(datasets[i].data).reshape(N, -1).all(dim=1)
        if verbose:
            nBadPerDataset[i] = int((~finiteRows).sum())
        keep &= finiteRows.to(device)

    if verbose:
        nRemoved = int((~keep).sum())
        print(f"database cleaning : {nRemoved} / {N} échantillons retirés")
        for i, nBad in nBadPerDataset.items():
            print(f"    dataset {_label(i, datasets[i])} : {nBad} échantillons non finis")

    if not bool(keep.any()):
        raise ValueError("Tous les échantillons contiennent au moins une valeur non finie.")

    return tuple(
        serie(d.data[keep.to(d.data.device)], colName=d.colName,
              storingDtype=d.storingDtype, storingDevice=d.storingDevice)
        for d in datasets
    )
