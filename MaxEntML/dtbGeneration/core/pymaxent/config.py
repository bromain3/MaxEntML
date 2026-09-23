"""
config.py
=========

Single place that decides *where* (device) and *in what precision* (dtype)
every tensor of the package lives.

Three ways to choose, lowest to highest priority:

1. Edit ``DEFAULT_DEVICE`` / ``DEFAULT_DTYPE`` below.
2. Call ``set_default(device=..., dtype=...)`` once at start-up
   (``main.py`` does it from its ``DEVICE`` / ``DTYPE`` constants and from
   the ``--device`` / ``--dtype`` CLI flags).
3. Pass ``device=`` / ``dtype=`` explicitly to an entry point that accepts
   them (``solve_adaptive``, ``composite_quadrature``, ...).

Anything that is already a tensor is moved/cast with ``Tensor.to``, which is
differentiable, so autograd history survives the conversion.

Everything in the package is *batched*: a set of moments is a tensor of shape
``(B, n_mom)``. A 1-D input ``(n_mom,)`` is accepted and treated as ``B = 1``
(outputs then carry a leading batch dimension of size 1).
"""

from __future__ import annotations

import argparse
from typing import Union

import torch

DeviceLike = Union[str, torch.device, None]
DTypeLike = Union[str, torch.dtype, None]

# ----------------------------------------------------------------------------
# THE FLAGS (edit here, or override with set_default / CLI)
# ----------------------------------------------------------------------------
DEFAULT_DEVICE: str = "cpu"        # "cpu", "cuda", "cuda:1", "mps", or "auto"
DEFAULT_DTYPE: str = "float64"     # "float64" or "float32"

_DTYPE_NAMES = {
    "float64": torch.float64, "double": torch.float64, "fp64": torch.float64, "f64": torch.float64,
    "float32": torch.float32, "float": torch.float32, "single": torch.float32,
    "fp32": torch.float32, "f32": torch.float32,
}

_STATE = {"device": torch.device("cpu"), "dtype": torch.float64}


# ----------------------------------------------------------------------------
# Resolution helpers
# ----------------------------------------------------------------------------

def resolve_dtype(dtype: DTypeLike = None) -> torch.dtype:
    """None -> current default; str / torch.dtype -> torch.dtype."""
    if dtype is None:
        return _STATE["dtype"]
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key not in _DTYPE_NAMES:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_NAMES)}.")
    return _DTYPE_NAMES[key]


def resolve_device(device: DeviceLike = None) -> torch.device:
    """None -> current default; 'auto' -> cuda if available else cpu."""
    if device is None:
        return _STATE["device"]
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_default(device: DeviceLike = None, dtype: DTypeLike = None):
    """
    Set the package-wide default device and/or dtype. Also sets torch's own
    default floating dtype so that any un-annotated float tensor created
    downstream agrees with it. Returns ``(device, dtype)``.
    """
    new_device = resolve_device(device) if device is not None else _STATE["device"]
    new_dtype = resolve_dtype(dtype) if dtype is not None else _STATE["dtype"]

    if new_device.type == "mps" and new_dtype == torch.float64:
        raise ValueError("The MPS backend has no float64 support; use dtype='float32'.")

    _STATE["device"] = new_device
    _STATE["dtype"] = new_dtype
    torch.set_default_dtype(new_dtype)
    return new_device, new_dtype


def get_device() -> torch.device:
    return _STATE["device"]


def get_dtype() -> torch.dtype:
    return _STATE["dtype"]


def eps(dtype: DTypeLike = None) -> float:
    return float(torch.finfo(resolve_dtype(dtype)).eps)


def tiny(dtype: DTypeLike = None) -> float:
    return float(torch.finfo(resolve_dtype(dtype)).tiny)


def floor_tol(tol: float, dtype: DTypeLike = None, factor: float = 100.0) -> float:
    """
    A tolerance can never be tighter than what the dtype can represent.
    ``factor * machine_eps`` is ~2e-12 in float64 (leaves 1e-10 untouched)
    and ~1.2e-5 in float32 (so a 1e-10 request does not loop forever).
    """
    return max(float(tol), factor * eps(dtype))


# ----------------------------------------------------------------------------
# Tensor conversion
# ----------------------------------------------------------------------------

def as_tensor(x, device: DeviceLike = None, dtype: DTypeLike = None) -> torch.Tensor:
    """Anything array-like -> tensor on (device, dtype). Tensors keep their graph."""
    dev = resolve_device(device)
    dt = resolve_dtype(dtype)
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=dt)
    return torch.as_tensor(x, dtype=dt, device=dev)


def as_batched(x, device: DeviceLike = None, dtype: DTypeLike = None) -> torch.Tensor:
    """Like ``as_tensor`` but guarantees shape (B, n): 1-D input becomes (1, n)."""
    x = as_tensor(x, device, dtype)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected a (n,) or (B, n) tensor, got shape {tuple(x.shape)}.")
    return x


# ----------------------------------------------------------------------------
# CLI glue
# ----------------------------------------------------------------------------

def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default=None,
                        help="Device the computation runs on: cpu, cuda, cuda:1, mps, auto "
                             "(default: the DEVICE constant of main.py).")
    parser.add_argument("--dtype", type=str, default=None, choices=sorted(_DTYPE_NAMES),
                        help="Floating dtype of the computation "
                             "(default: the DTYPE constant of main.py).")


def apply_cli_args(args, fallback_device: DeviceLike = None, fallback_dtype: DTypeLike = None):
    device = args.device if getattr(args, "device", None) is not None else fallback_device
    dtype = args.dtype if getattr(args, "dtype", None) is not None else fallback_dtype
    return set_default(device, dtype)


# Initialise from the constants above (no torch-global side effect at import).
_STATE["device"] = resolve_device(DEFAULT_DEVICE)
_STATE["dtype"] = resolve_dtype(DEFAULT_DTYPE)
