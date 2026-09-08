"""Shared dtype-aware truncation cutoff policy."""

from __future__ import annotations

import numpy as np


def dtype_auto_cutoff(dtype):
    """Return Pepsy's default truncation cutoff for ``dtype``.

    The policy is shared by the MPS, tree, PEPS, and Hamiltonian conversion
    APIs so ``cutoff="auto"`` has the same meaning everywhere:

    * ``float64``/``complex128``: ``1e-12``;
    * ``float32``/``complex64``: ``1e-6``;
    * 16-bit floating-point data: ``1e-3``.

    Backend dtype objects such as ``torch.float32`` are accepted in addition
    to NumPy dtypes. Their string form is sufficient for this precision
    classification when NumPy cannot construct a dtype directly.
    """

    try:
        dtype_name = np.dtype(dtype).name.lower()
    except (TypeError, ValueError):
        dtype_name = str(dtype).strip().lower()

    if "16" in dtype_name:
        return 1.0e-3
    if "32" in dtype_name or "complex64" in dtype_name:
        return 1.0e-6
    return 1.0e-12
