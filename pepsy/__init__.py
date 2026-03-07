"""Pepsy boundary-contraction library package."""

from importlib import import_module
from typing import TYPE_CHECKING

from .version import __version__

if TYPE_CHECKING:
    from . import boundary_norm, boundary_states, boundary_sweeps
    from .boundary_norm import BoundaryContractResult, ContractBoundary, prepare_boundary_inputs
    from .boundary_states import BdyMPS, make_numpy_array_caster
    from .boundary_sweeps import CompBdy

__all__ = [
    "__version__",
    "BdyMPS",
    "CompBdy",
    "BoundaryContractResult",
    "ContractBoundary",
    "prepare_boundary_inputs",
    "make_numpy_array_caster",
    "boundary_norm",
    "boundary_states",
    "boundary_sweeps",
]


def __getattr__(name):
    """Lazily import public API symbols and common submodules."""
    if name in ("boundary_norm", "boundary_states", "boundary_sweeps"):
        return import_module(f".{name}", __name__)

    if name in ("ContractBoundary", "prepare_boundary_inputs", "BoundaryContractResult"):
        from .boundary_norm import (  # pylint: disable=import-outside-toplevel
            BoundaryContractResult,
            ContractBoundary,
            prepare_boundary_inputs,
        )

        return {
            "BoundaryContractResult": BoundaryContractResult,
            "ContractBoundary": ContractBoundary,
            "prepare_boundary_inputs": prepare_boundary_inputs,
        }[name]

    if name in ("BdyMPS", "make_numpy_array_caster"):
        from .boundary_states import (  # pylint: disable=import-outside-toplevel
            BdyMPS,
            make_numpy_array_caster,
        )

        return {
            "BdyMPS": BdyMPS,
            "make_numpy_array_caster": make_numpy_array_caster,
        }[name]

    if name == "CompBdy":
        from .boundary_sweeps import CompBdy  # pylint: disable=import-outside-toplevel

        return CompBdy

    raise AttributeError(f"module 'pepsy' has no attribute {name!r}")
