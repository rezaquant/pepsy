"""Deprecated compatibility wrapper for boundary norm evaluation."""

import warnings

from .boundary_norm import ContractBoundary


def tn_norm(norm, mps_boundaries, **kwargs):
    """Deprecated wrapper. Use ``ContractBoundary(norm=..., mps_boundaries=...)`` instead."""
    warnings.warn(
        (
            "tn_norm() is deprecated. Use "
            "ContractBoundary(norm=..., mps_boundaries=...) from pepsy.boundary_norm."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return ContractBoundary(norm=norm, mps_boundaries=mps_boundaries, **kwargs)
