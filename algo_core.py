"""Compatibility shim for legacy ``algo_core`` imports.

Use ``dmrg_helpers`` directly. ``hyper_comp`` has been removed.
"""

from dmrg_helpers import backend_jax, backend_numpy, backend_torch, opt_

__all__ = ["backend_torch", "backend_numpy", "backend_jax", "opt_"]
