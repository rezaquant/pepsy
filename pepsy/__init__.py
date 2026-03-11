"""Pepsy boundary-contraction library package."""

from importlib import import_module
from typing import TYPE_CHECKING

from .version import __version__

if TYPE_CHECKING:
    from . import (
        boundary_norm,
        boundary_states,
        boundary_sweeps,
        core,
        dmrg_fit,
        optimize_sweep,
    )
    from .boundary_norm import (
        BoundaryContractResult,
        ContractBoundary,
        normalize,
        prepare_boundary_inputs,
    )
    from .boundary_states import BdyMPS, make_numpy_array_caster
    from .boundary_sweeps import CompBdy
    from .core import (
        get_default_array_backend,
        get_default_grad_backend,
        reset_default_backends,
        set_default_array_backend,
        set_default_grad_backend,
    )
    from .optimize_sweep import PEPSSweepOptimizer, SweepResult

__all__ = [
    "__version__",
    "BdyMPS",
    "CompBdy",
    "BoundaryContractResult",
    "ContractBoundary",
    "prepare_boundary_inputs",
    "normalize",
    "make_numpy_array_caster",
    "set_default_array_backend",
    "get_default_array_backend",
    "set_default_grad_backend",
    "get_default_grad_backend",
    "reset_default_backends",
    "PEPSSweepOptimizer",
    "SweepResult",
    "optimize_sweep",
    "boundary_norm",
    "boundary_states",
    "boundary_sweeps",
    "core",
    "dmrg_fit",
]

def __getattr__(name):
    """Lazily import public API symbols and common submodules."""
    if name in (
        "boundary_norm",
        "boundary_states",
        "boundary_sweeps",
        "optimize_sweep",
        "core",
        "dmrg_fit",
    ):
        return import_module(f".{name}", __name__)

    if name in ("ContractBoundary", "prepare_boundary_inputs", "BoundaryContractResult", "normalize"):
        from .boundary_norm import (  # pylint: disable=import-outside-toplevel
            BoundaryContractResult,
            ContractBoundary,
            normalize,
            prepare_boundary_inputs,
        )

        return {
            "BoundaryContractResult": BoundaryContractResult,
            "ContractBoundary": ContractBoundary,
            "prepare_boundary_inputs": prepare_boundary_inputs,
            "normalize": normalize,
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

    if name in (
        "set_default_array_backend",
        "get_default_array_backend",
        "set_default_grad_backend",
        "get_default_grad_backend",
        "reset_default_backends",
    ):
        from .core import (  # pylint: disable=import-outside-toplevel
            get_default_array_backend,
            get_default_grad_backend,
            reset_default_backends,
            set_default_array_backend,
            set_default_grad_backend,
        )

        return {
            "set_default_array_backend": set_default_array_backend,
            "get_default_array_backend": get_default_array_backend,
            "set_default_grad_backend": set_default_grad_backend,
            "get_default_grad_backend": get_default_grad_backend,
            "reset_default_backends": reset_default_backends,
        }[name]

    if name == "CompBdy":
        from .boundary_sweeps import CompBdy  # pylint: disable=import-outside-toplevel

        return CompBdy

    if name in ("PEPSSweepOptimizer", "SweepResult"):
        from .optimize_sweep import (  # pylint: disable=import-outside-toplevel
            PEPSSweepOptimizer,
            SweepResult,
        )

        return {
            "PEPSSweepOptimizer": PEPSSweepOptimizer,
            "SweepResult": SweepResult,
        }[name]

    raise AttributeError(f"module 'pepsy' has no attribute {name!r}")
