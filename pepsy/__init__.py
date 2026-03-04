"""Pepsy boundary-contraction library package."""

from typing import TYPE_CHECKING

from .version import __version__

if TYPE_CHECKING:
    from .boundary_norm import ContractBoundary, prepare_boundary_inputs
    from .boundary_states import BdyMPS, add_diagonalu_tags, make_numpy_array_caster
    from .boundary_sweeps import CompBdy

__all__ = [
    "__version__",
    "BdyMPS",
    "CompBdy",
    "ContractBoundary",
    "prepare_boundary_inputs",
    "add_diagonalu_tags",
    "make_numpy_array_caster",
]


def __getattr__(name):
    """Lazily import public API symbols to keep package import lightweight."""
    if name in ("ContractBoundary", "prepare_boundary_inputs"):
        from .boundary_norm import (  # pylint: disable=import-outside-toplevel
            ContractBoundary,
            prepare_boundary_inputs,
        )

        return {
            "ContractBoundary": ContractBoundary,
            "prepare_boundary_inputs": prepare_boundary_inputs,
        }[name]
    if name in ("BdyMPS", "add_diagonalu_tags", "make_numpy_array_caster"):
        from .boundary_states import (  # pylint: disable=import-outside-toplevel
            BdyMPS,
            add_diagonalu_tags,
            make_numpy_array_caster,
        )

        return {
            "BdyMPS": BdyMPS,
            "add_diagonalu_tags": add_diagonalu_tags,
            "make_numpy_array_caster": make_numpy_array_caster,
        }[name]
    if name == "CompBdy":
        from .boundary_sweeps import CompBdy  # pylint: disable=import-outside-toplevel

        return CompBdy
    raise AttributeError(f"module 'pepsy' has no attribute {name!r}")
