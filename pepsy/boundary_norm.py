"""Boundary-based tensor-network norm evaluation."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from .boundary_states import BdyMPS
from .boundary_sweeps import CompBdy
from .core import get_default_array_backend


_TAG_X = re.compile(r"^X\d+$")
_TAG_Y = re.compile(r"^Y\d+$")
_PHYS_OUTER = re.compile(r"^k\d+(?:,\d+)*$")

__all__ = [
    "prepare_boundary_inputs",
    "BoundaryContractResult",
    "ContractBoundary",
    "normalize",
]


@dataclass(frozen=True)
class BoundaryContractResult:
    """Structured output for :func:`ContractBoundary`."""

    cost: complex | float
    fidel: list[float]
    direction: str
    n_iter: int
    max_separation: int


def _validate_tensor_network_tags(p):
    """Ensure PEPS lattice tags are present for shape inference."""
    tags = set(getattr(p, "tags", ()))

    if not any(_TAG_X.match(tag) for tag in tags) or not any(_TAG_Y.match(tag) for tag in tags):
        raise ValueError("Input network must contain X* and Y* lattice tags.")


def _normalize_retag_for_direction(direction, re_tag):
    """Normalize ``re_tag`` flag for direction-specific calls."""
    _ = direction
    return bool(re_tag)


def _warn_nonstandard_physical_outer_inds(tn, role):
    """Warn when outer physical indices don't match ``k<int>[,<int>...]``."""
    bad = [
        idx
        for idx in tn.outer_inds()
        if not (isinstance(idx, str) and _PHYS_OUTER.fullmatch(idx))
    ]
    if bad:
        sample = ", ".join(sorted(bad)[:8])
        warnings.warn(
            f"{role} outer indices expected format k<int>[,<int>...]. "
            f"Found non-matching indices: {sample}",
            stacklevel=3,
        )



def prepare_boundary_inputs(
    ket=None,
    *,
    bra=None,
):
    """Prepare tagged ``ket``/``bra`` networks and build ``norm``.

    Parameters
    ----------
    ket : qtn.TensorNetwork | None
        Input ket network.
    bra : qtn.TensorNetwork | None
        Optional bra network. If ``None``, ``ket.copy().conj()`` is used.

    Returns
    -------
    tuple[qtn.TensorNetwork, qtn.TensorNetwork]
        ``(ket_tagged, norm_tagged)``

    Notes
    -----
    ``ket`` is tagged in-place (no copy). The returned ``ket_tagged`` is the same
    object as ``ket``.

    When ``bra`` is ``None``: shared internal ket/bra indices are renamed on the
    auto-generated bra side as ``<original>_*``.

    When ``bra`` is provided: no reindexing is performed; only bra/ket internal
    index names are required to be disjoint (outer-index overlap is allowed).
    """
    if ket is None:
        raise ValueError("Provide ket.")

    _validate_tensor_network_tags(ket)

    ket_tagged = ket
    auto_bra = bra is None
    bra_tagged = ket.copy().conj() if auto_bra else bra.copy().conj()

    if auto_bra:
        shared_inner = set(ket_tagged.inner_inds()) & set(bra_tagged.inner_inds())
        reindex_map = {idx: f"{idx}_*" for idx in shared_inner}
        final_collisions = set(reindex_map.values()) & (
            set(ket_tagged.ind_map) | (set(bra_tagged.ind_map) - shared_inner)
        )
        if final_collisions:
            sample = ", ".join(sorted(final_collisions)[:8])
            raise ValueError(
                "Automatic bra reindex idx -> idx_* collides with existing indices. "
                f"Collisions found: {sample}"
            )
        bra_tagged.reindex_(reindex_map)
    else:
        collisions = set(ket_tagged.inner_inds()) & set(bra_tagged.inner_inds())
        if collisions:
            sample = ", ".join(sorted(collisions)[:8])
            raise ValueError(
                "Provided bra must have internal index names disjoint from ket. "
                f"Internal collisions found: {sample}"
            )

    _warn_nonstandard_physical_outer_inds(ket_tagged, "ket")
    if not auto_bra:
        _warn_nonstandard_physical_outer_inds(bra_tagged, "bra")

    ket_tagged.add_tag("KET")
    bra_tagged.add_tag("BRA")
    norm_tagged = bra_tagged | ket_tagged
    return ket_tagged, norm_tagged


def ContractBoundary(
    *,
    norm,
    mps_boundaries,
    opt="auto-hq",
    flat=False,
    dmrg_run="eff",
    n_iter=2,
    re_tag=True,
    pbar=True,
    fidel_=False,
    visual_=False,
    re_update=True,
    max_separation=0,
    direction="y",
    eq_norms=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,invalid-name
    """Compute tensor-network norm via boundary sweeps.

    Parameters
    ----------
    norm : qtn.TensorNetwork
        Prebuilt double-layer network, usually from
        :func:`prepare_boundary_inputs`.
    mps_boundaries : dict[str, qtn.MatrixProductState]
        Boundary dictionary, usually from :class:`pepsy.boundary_states.BdyMPS`.
    opt : str | object, default="auto-hq"
        Contraction optimizer passed through to :class:`pepsy.boundary_sweeps.CompBdy`.
    flat : bool, default=False
        Forwarded to sweep backend.
    dmrg_run : {"eff", "global"}, default="eff"
        Fit backend mode.
    n_iter : int, default=2
        Number of local fit iterations per step.
    re_tag : bool, default=True
        Forwarded to fitting backend.
    pbar : bool, default=True
        Show progress bars.
    fidel_ : bool, default=False
        If ``True``, collect per-step fidelity values in ``result.fidel``.
    visual_ : bool, default=False
        Enable intermediate visualization in fitting backend.
    re_update : bool, default=True
        Whether to write fitted boundaries back into ``mps_boundaries``.
    max_separation : int, default=0
        Sweep separation mode.
    direction : str, default="y"
        Sweep selector.
    eq_norms : bool, default=False
        Forwarded normalization option for local fit outputs.

    Returns
    -------
    BoundaryContractResult
        Structured boundary contraction result including ``cost`` and
        fidelity history ``fidel``.
    """
    if norm is None:
        raise ValueError("norm must not be None.")
    if not isinstance(mps_boundaries, dict):
        raise TypeError("mps_boundaries must be a dictionary of boundary states.")

    re_tag = _normalize_retag_for_direction(direction, re_tag)
    norm_tagged = norm.copy()

    comp_bdy = CompBdy(
        norm_tagged,
        mps_boundaries,
        opt=opt,
        dmrg_run=dmrg_run,
    )

    cost = comp_bdy.run(
        n_iter=n_iter,
        re_tag=re_tag,
        pbar=pbar,
        fidel_=fidel_,
        visual_=visual_,
        flat=flat,
        re_update=re_update,
        max_separation=max_separation,
        direction=direction,
        eq_norms=eq_norms,
    )
 

    return BoundaryContractResult(
        cost=cost,
        fidel=list(comp_bdy.fidel),
        direction=direction,
        n_iter=n_iter,
        max_separation=max_separation,
    )


def normalize(
    p,
    *,
    chi=None,
    bdy=None,
    opt="auto-hq",
    n_iter=5,
    direction="y",
    max_separation=0,
    pbar=False,
    fidel_=False,
    dmrg_run="eff",
    single_layer=False,
    to_backend=None,
):
    """Normalize a PEPS state with boundary contraction.

    This performs:
    ``prepare_boundary_inputs -> BdyMPS -> ContractBoundary`` and returns a
    dictionary containing normalized state, cost, and boundary objects.

    Parameters
    ----------
    p : qtn.TensorNetwork
        Input PEPS state.
    chi : int | None, default=None
        Boundary MPS bond dimension used when ``bdy`` is not provided.
    bdy : pepsy.boundary_states.BdyMPS | None, default=None
        Pre-built boundary object. If provided, ``normalize`` reuses it and
        skips creating a new :class:`pepsy.boundary_states.BdyMPS`.
    opt : str | object, default="auto-hq"
        Contraction optimizer.
    n_iter : int, default=5
        Number of local fit iterations per boundary step.
    direction : str, default="y"
        Sweep direction passed to :func:`ContractBoundary`.
    max_separation : int, default=0
        Sweep separation mode.
    pbar : bool, default=False
        Show progress bar.
    fidel_ : bool, default=False
        Track fidelity history during boundary contraction.
    dmrg_run : {"eff", "global"}, default="eff"
        Boundary fitting backend mode.
    single_layer : bool, default=False
        Boundary initializer mode for :class:`pepsy.boundary_states.BdyMPS`.
    to_backend : callable | None, default=None
        Optional array caster applied to the double-layer norm network. If
        ``None``, uses :func:`pepsy.core.get_default_array_backend` when set.

    Returns
    -------
    dict[str, object]
        Dictionary with:

        - ``state``: normalized PEPS state
        - ``cost``: raw boundary contraction cost
        - ``cost_scalar``: scalar used for normalization
        - ``bdy``: constructed :class:`pepsy.boundary_states.BdyMPS`
        - ``contract_result``: :class:`BoundaryContractResult`
    """
    if p is None:
        raise ValueError("p must not be None.")

    ket_tagged, norm_tagged = prepare_boundary_inputs(ket=p, bra=None)
    backend = to_backend if to_backend is not None else get_default_array_backend()
    if backend is not None:
        norm_tagged.apply_to_arrays(backend)

    if bdy is None:
        if chi is None:
            raise ValueError("Provide chi when bdy is not supplied.")

        bdy = BdyMPS(
            tn_double=norm_tagged,
            chi=chi,
            single_layer=single_layer,
            to_backend=backend,
        )
    elif not hasattr(bdy, "mps_b"):
        raise TypeError("bdy must expose attribute 'mps_b'.")

    result = ContractBoundary(
        norm=norm_tagged,
        mps_boundaries=bdy.mps_b,
        opt=opt,
        dmrg_run=dmrg_run,
        n_iter=n_iter,
        pbar=pbar,
        direction=direction,
        max_separation=max_separation,
        fidel_=fidel_,
    )
    cost = result.cost
    try:
        cost_scalar = complex(cost)
    except TypeError:
        cost_scalar = cost

    if abs(cost_scalar) == 0:
        raise ZeroDivisionError("Boundary norm cost is zero; cannot normalize state.")
    return {
        "state": ket_tagged / (cost_scalar**0.5),
        "cost": cost,
        "cost_scalar": cost_scalar,
        "bdy": bdy,
        "contract_result": result,
    }
