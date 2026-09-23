#!/usr/bin/env python3
"""
main.py
=======
Streamlined MaxEnt closure module exposing the `compute_lambdas` API.
"""

from __future__ import annotations

import gc
import torch

from MaxEntML.dtbGeneration.core.pymaxent.quadrature import Quadrature, composite_quadrature, DEFAULT_COMPOSITE_RULE
from MaxEntML.dtbGeneration.core.pymaxent.levermore import LevermoreEntropy
from MaxEntML.dtbGeneration.core.pymaxent import maxent

NMOM = 5
N_QUAD = 80
N_PTS_FIXED = 5
# N_CELLS_FIXED = N_QUAD // N_PTS_FIXED
N_CELLS_FIXED = 203
FIXED_HALF_WIDTH = 64.090
FIXED_QUAD_RULE = DEFAULT_COMPOSITE_RULE


def build_quadrature():
    """Fixed composite quadrature rule on the standard domain."""
    return composite_quadrature(
        -FIXED_HALF_WIDTH, 
        FIXED_HALF_WIDTH, 
        n_cells=N_CELLS_FIXED, 
        n_pts=N_PTS_FIXED, 
        rule=FIXED_QUAD_RULE
    )


def _solve(q_batch, r_batch):
    """Helper to construct raw moments from standardized (q*, r*) and solve the MaxEnt dual system."""
    device = q_batch.device
    dtype = q_batch.dtype
    
    # Construct raw moments m in the standardized frame
    m = torch.stack([
        torch.ones_like(q_batch),
        torch.zeros_like(q_batch),
        torch.ones_like(q_batch),
        q_batch,
        r_batch
    ], dim=-1)
    
    # Build quadrature and move it to the batch device/dtype using its built-in .to() method
    quad = build_quadrature().to(device=device, dtype=dtype)
    
    lam, _, _ = maxent.m_to_reconstruction_params(
        LevermoreEntropy, m, quad,
        use_precond=True, 
        use_adapt_base=False, 
        use_initial_max=True,
        use_varsigma=False, 
        tol=1e-10, 
        maxit=100,
        grad_mode="unroll", 
        raise_on_fail=False
    )
    return lam


@torch.no_grad()
def compute_lambdas(q, r, batch_size=2000, compute_device=None, show_progress=True):
    """Lambdas MaxEnt sur une grille (Nu, Nv) de points (q*, r*) avec quadrature fixe."""
    orig_device = q.device
    if compute_device is not None:
        compute_device = torch.device(compute_device)
        q = q.to(compute_device)
        r = r.to(compute_device)
    else:
        compute_device = orig_device

    total_elements = q.numel()
    q_flat = q.reshape(-1)
    r_flat = r.reshape(-1)

    lam_list = []
    iterator = range(0, total_elements, batch_size)
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Calcul MaxEnt (quadrature fixe)")
        except ImportError:
            pass

    for i in iterator:
        q_batch = q_flat[i:i + batch_size]
        r_batch = r_flat[i:i + batch_size]
        
        # 1. Computation happens on the GPU (compute_device)
        lam_batch = _solve(q_batch, r_batch)
        
        # 2. Immediately offload the batch to CPU memory to free VRAM
        lam_list.append(lam_batch.cpu())
        
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    # 3. Concatenate results on CPU memory
    lam = torch.cat(lam_list).reshape(*q.shape, NMOM)
    
    # 4. Return to original device only if the user explicitly wants it back on GPU
    if orig_device != lam.device:
        lam = lam.to(orig_device)
        
    return lam