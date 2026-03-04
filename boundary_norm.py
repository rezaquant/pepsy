"""Boundary-based tensor-network norm evaluation."""

from __future__ import annotations

import re
from uuid import uuid4
import warnings

from boundary_states import add_diagonalu_tags
from boundary_sweeps import CompBdy


_TAG_X = re.compile(r"^X\d+$")
_TAG_Y = re.compile(r"^Y\d+$")
_TAG_DU = re.compile(r"^Du\d+$")

__all__ = [
    "prepare_boundary_inputs",
    "ContractBoundary",
]


def _validate_tensor_network_tags(p, add_diag_to_p):
    """Validate lattice tags and update diagonal-tag behavior when needed."""
    tags = set(getattr(p, "tags", ()))

    if not any(_TAG_X.match(tag) for tag in tags) or not any(_TAG_Y.match(tag) for tag in tags):
        raise ValueError("Input network must contain X* and Y* lattice tags.")

    if not add_diag_to_p and not any(_TAG_DU.match(tag) for tag in tags):
        warnings.warn(
            "Input network has no Du* tags. Enabling diagonal tagging automatically.",
            RuntimeWarning,
        )
        return True
    return add_diag_to_p


def _normalize_retag_for_direction(direction, re_tag):
    """Normalize ``re_tag`` for direction-specific requirements."""
    direction_key = direction.lower() if isinstance(direction, str) else ""
    if direction_key.startswith("diag"):
        if not re_tag:
            warnings.warn(
                "direction='diag*' requires re_tag=True. Overriding re_tag to True.",
                RuntimeWarning,
            )
            re_tag = True

    return re_tag


def prepare_boundary_inputs(
    ket=None,
    *,
    bra=None,
    add_diag_to_p=True,
):
    """Prepare tagged ``ket``/``bra`` networks and build ``norm``.

    Parameters
    ----------
    ket : qtn.TensorNetwork | None
        Input ket network.
    bra : qtn.TensorNetwork | None
        Optional bra network. If ``None``, ``ket.copy().conj()`` is used.
    add_diag_to_p : bool
        Add diagonal ``Du*``/``du*`` tags to ``ket`` when constructing input.

    Returns
    -------
    tuple[qtn.TensorNetwork, qtn.TensorNetwork]
        ``(ket_tagged, norm_tagged)``
    """
    if ket is None:
        raise ValueError("Provide ket.")

    add_diag_to_p = _validate_tensor_network_tags(ket, add_diag_to_p)

    ket_tagged = ket.copy()
    bra_tagged = ket.copy().conj() if bra is None else bra.copy().conj()
    if add_diag_to_p:
        ket_tagged = add_diagonalu_tags(ket_tagged)
        bra_tagged = add_diagonalu_tags(bra_tagged)

    shared_inner = set(ket_tagged.inner_inds()) & set(bra_tagged.inner_inds())
    bra_tagged.reindex_({idx: f"r{uuid4().hex}" for idx in shared_inner})
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
    re_tag=False,
    pbar=True,
    fidel_=False,
    visual_=False,
    re_update=True,
    max_separation=0,
    max_seperation=0,
    direction="y",
    eq_norms=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,invalid-name
    """Compute tensor-network norm via boundary sweeps.

    This function expects a prebuilt double-layer ``norm`` network and a
    boundary-state dictionary ``mps_boundaries``.

    Returns
    -------
    complex | float
        Boundary contraction cost scalar.
    """
    if norm is None:
        raise ValueError("norm must not be None.")
    if not isinstance(mps_boundaries, dict):
        raise TypeError("mps_boundaries must be a dictionary of boundary states.")

    if max_seperation not in (0, max_separation):
        warnings.warn(
            "Both max_separation and max_seperation were set. Using max_seperation.",
            RuntimeWarning,
        )
    max_sep_value = max_seperation if max_seperation != 0 else max_separation

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
        max_separation=max_sep_value,
        direction=direction,
        eq_norms=eq_norms,
    )
    return cost
