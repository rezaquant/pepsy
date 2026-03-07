"""Boundary-based tensor-network norm evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .boundary_sweeps import CompBdy


_TAG_X = re.compile(r"^X\d+$")
_TAG_Y = re.compile(r"^Y\d+$")

__all__ = [
    "prepare_boundary_inputs",
    "BoundaryContractResult",
    "ContractBoundary",
]


@dataclass(frozen=True)
class BoundaryContractResult:
    """Structured boundary-contraction output with optional fidelity history."""

    cost: complex | float
    fidel: list
    direction: str
    n_iter: int
    max_separation: int


def _validate_tensor_network_tags(p):
    """Validate lattice tags."""
    tags = set(getattr(p, "tags", ()))

    if not any(_TAG_X.match(tag) for tag in tags) or not any(_TAG_Y.match(tag) for tag in tags):
        raise ValueError("Input network must contain X* and Y* lattice tags.")


def _normalize_retag_for_direction(direction, re_tag):
    """Normalize ``re_tag`` for sweep calls."""
    _ = direction
    return bool(re_tag)


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
    When ``bra`` is ``None``: shared internal ket/bra indices are renamed on the
    auto-generated bra side as ``<original>_*``.

    When ``bra`` is provided: no reindexing is performed; only bra/ket internal
    index names are required to be disjoint (outer-index overlap is allowed).
    """
    if ket is None:
        raise ValueError("Provide ket.")

    _validate_tensor_network_tags(ket)

    ket_tagged = ket.copy()
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
    return_info=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,invalid-name
    """Compute tensor-network norm via boundary sweeps.

    This function expects a prebuilt double-layer ``norm`` network and a
    boundary-state dictionary ``mps_boundaries``.

    Returns
    -------
    complex | float | BoundaryContractResult
        Boundary contraction cost scalar, or structured result when
        ``return_info=True``.
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
    if not return_info:
        return cost

    return BoundaryContractResult(
        cost=cost,
        fidel=list(comp_bdy.fidel),
        direction=direction,
        n_iter=n_iter,
        max_separation=max_separation,
    )
