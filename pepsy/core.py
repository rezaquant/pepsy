"""Shared DMRG backend, optimizer, and fidelity helpers."""

import importlib.util
import warnings
from typing import Any

import numpy as np
import cotengra as ctg

__all__ = [
    "backend_torch",
    "backend_numpy",
    "backend_jax",
    "build_optimizer",
    "build_compressed_optimizer",
    "opt_",
    "copt_",
    "fidel_mps",
]


def backend_torch(device="cpu", dtype=None, requires_grad=False):
    """Return a converter that materializes arrays as torch tensors."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "backend_torch requires the optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch]"
        ) from exc

    if dtype is None:
        dtype = torch.float64

    def to_backend(x, device=device, dtype=dtype, requires_grad=requires_grad):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

    return to_backend


def backend_numpy(dtype=np.float64):
    """Return a converter that materializes arrays as NumPy arrays."""

    def to_backend(x, dtype=dtype):
        return np.array(x, dtype=dtype)

    return to_backend


def backend_jax(dtype=None, device=None):
    """Return a converter that places arrays onto a specific JAX device."""
    try:
        import jax  # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "backend_jax requires optional dependencies 'jax' and 'jaxlib'. "
            "Install them with: pip install pepsy[jax]"
        ) from exc

    if dtype is None:
        dtype = jnp.float64
    if device is None:
        device = jax.devices("cpu")[0]

    def to_backend(x, dtype=dtype, device=device):
        return jax.device_put(jnp.array(x, dtype=dtype), device)

    return to_backend


def build_optimizer(
    progbar=True,
    alpha=32,
    target_size=2**34,
    subtree_size=12,
    max_time="rate:1e8",
    max_repeats=2**6,
    parallel=True,
    optlib="cmaes",
    directory="cash/",
    hash_method="b",
):
    """Build and return a reusable cotengra contraction optimizer."""
    selected_optlib = optlib
    if selected_optlib == "cmaes" and importlib.util.find_spec("cmaes") is None:
        warnings.warn(
            "Package 'cmaes' not found. Falling back to optlib='random'.",
            RuntimeWarning,
        )
        selected_optlib = "random"
    opt = ctg.ReusableHyperOptimizer(
        minimize=f"combo-{alpha}",
        slicing_opts={"target_size": 2**40},
        slicing_reconf_opts={"target_size": target_size},
        reconf_opts={"subtree_size": subtree_size},
        max_repeats=max_repeats,
        parallel=parallel,
        optlib=selected_optlib,
        max_time=max_time,
        hash_method=hash_method,
        directory=directory,
        progbar=progbar,
    )
    return opt


def build_compressed_optimizer(
    progbar=True,
    chi=4,
    directory=None,
    max_repeats=2**8,
    max_time="rate:1e8",
):
    """Build and return a reusable cotengra compressed optimizer."""
    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats,
        minimize="combo-compressed",
        progbar=progbar,
        max_time=max_time,
        directory=directory,
    )
    return copt


# Backward-compatible aliases.
opt_ = build_optimizer
copt_ = build_compressed_optimizer


def fidel_mps(psi, psi_fix):
    """Compute normalized MPS fidelity |<psi|psi_fix>|^2/(||psi||^2||psi_fix||^2)."""
    opt: Any = build_optimizer(progbar=False)
    val_0 = abs((psi.H & psi).contract(all, optimize=opt))
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ref = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))

    val_1 = val_1**2
    fidelity = complex(val_1 / (val_0 * val_ref)).real
    return fidelity
