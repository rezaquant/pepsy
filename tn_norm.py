"""Deprecated compatibility wrapper for boundary norm evaluation."""

import warnings

from boundary_norm import BoundaryNormRunner


def tn_norm(p, **kwargs):
    """Deprecated wrapper. Use ``BoundaryNormRunner(...).run(p)`` instead."""
    warnings.warn(
        "tn_norm() is deprecated. Use BoundaryNormRunner(...).run(p) from boundary_norm.",
        DeprecationWarning,
        stacklevel=2,
    )
    runner = BoundaryNormRunner(**kwargs)
    return runner.run(p)
