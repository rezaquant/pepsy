"""Compatibility wrapper for :mod:`pepsy.dmrg_fit`."""

from pepsy.dmrg_fit import (
    FIT,
    canonize_mps,
    energy_global,
    fidel_mps,
    gate_1d,
    internal_inds,
    opt_,
)

__all__ = ["FIT", "opt_", "fidel_mps", "energy_global", "gate_1d", "internal_inds", "canonize_mps"]
