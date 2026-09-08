"""Tree-structure search for :class:`TreeOptimizer` gate-stream replay.

The paper *Simulating quantum circuits using tree tensor networks*
(Seitz, Medina, Cruz, Huang, Mendl; Quantum 7, 964, 2023; arXiv:2206.01000)
first fixes a rooted tree adapted to the entanglement the circuit is expected
to generate, then applies gates to it.

:class:`TreeLayoutFinder` builds that structure from the two-qubit connectivity
of a bundled gate stream.  It reuses the interaction-graph and recursive
spectral-bisection machinery written for the MPS layout finder
(:mod:`pepsy.optimizers.mps.layout`); where the MPS finder *flattens* the
bisection recursion into a 1D order, the tree finder *keeps* the recursion as
the rooted tree structure.  Strongly coupled qubits end up as nearby leaves,
minimising the tree-path length that two-qubit gates must thread across.

The structure is not restricted to strictly-binary trees.  Internal nodes may
have any arity: ``max_arity`` gives flatter ``k``-ary trees (shallower
geodesics), while ``structure="adaptive"`` reads the gate-stream interaction
graph and lets each level branch into as many children as it has strongly
coupled communities.  The default is a binary tree below a three-virtual-leg
root, which keeps every tensor at rank three.  Pass an explicit ``top_arity``
or an iterable ``max_arity`` to request another geometry.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from numbers import Integral

import autoray as ar
import numpy as np

from ..mps.layout import (
    _gate_stream_adjacency,
    _gate_stream_event_weights,
    _gate_stream_pair_weights,
    _gate_stream_spectral_order,
    _normalize_layout_gate_queue,
    _normalize_layout_support,
    _operator_schmidt_rank_bound,
    _operator_schmidt_rank_info as _mps_operator_schmidt_rank_info,
    _normalize_weight_mode,
)
from .._layout_orders import normalize_fixed_order
from ..mps.optimizer import _control_event_parts as _mps_control_event_parts
from .._layout_visualization import (
    add_order_colorbar,
    coordinate_lattice_edge_keys,
    coordinate_lattice_edges,
    event_color,
    finish_schematic_axes,
    matplotlib_modules,
    resolve_site_coords,
    scale_color,
)
from ...tensors.maps import OneDMap

__all__ = ["TreePlan", "TreeLayoutFinder"]

_DEFAULT_MAX_ARITY = object()
_DEFAULT_TOP_ARITY = object()
_DEFAULT_CHI = object()
_DEFAULT_ORDER = object()
_DEFAULT_SEARCH_OPTION = object()
_DEFAULT_SCALE_MARKERS = ("o",)


def _looks_like_tree_tensor_network(value):
    """Identify a TTN input without importing ``ttn`` (avoids a cycle)."""
    return (
        getattr(value, "plan", None) is not None
        and getattr(value, "tensor_map", None) is not None
    )


def _normalize_hybrid_weights(weights):
    """Validate path / peak-load / total-load hybrid objective weights."""
    if weights is None:
        values = (1.0, 1.0, 0.25)
    elif isinstance(weights, Mapping):
        aliases = {
            "path": "path",
            "max_edge_load": "max_edge_load",
            "peak_load": "max_edge_load",
            "total_edge_load": "total_edge_load",
            "total_load": "total_edge_load",
        }
        normalized = {}
        for key, value in weights.items():
            name = aliases.get(str(key).replace("-", "_").strip().lower())
            if name is None:
                raise ValueError(
                    "hybrid_weights keys must be 'path', 'max_edge_load', "
                    "or 'total_edge_load'."
                )
            if name in normalized:
                raise ValueError(f"duplicate hybrid weight {name!r}.")
            normalized[name] = value
        values = tuple(
            normalized.get(name, 0.0)
            for name in ("path", "max_edge_load", "total_edge_load")
        )
    else:
        try:
            values = tuple(weights)
        except TypeError as exc:
            raise ValueError(
                "hybrid_weights must be a three-item sequence, mapping, or None."
            ) from exc
        if len(values) != 3:
            raise ValueError(
                "hybrid_weights must contain path, max-edge-load, and "
                "total-edge-load weights."
            )
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("hybrid_weights must be finite non-negative numbers.") from exc
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("hybrid_weights must be finite non-negative numbers.")
    if not any(values):
        raise ValueError("at least one hybrid weight must be positive.")
    return values


def _normalize_layout_refinement(refine):
    """Normalize an optional deterministic fixed-plan refinement mode."""
    if refine is None or refine is False:
        return None
    name = str(refine).replace("-", "_").strip().lower()
    aliases = {
        "adjacent": "greedy",
        "adjacent_swaps": "greedy",
        "local": "greedy",
    }
    name = aliases.get(name, name)
    if name != "greedy":
        raise ValueError("refine must be None or 'greedy'.")
    return name


def _normalize_topology_refinement(refine):
    """Normalize the optional joint topology refinement mode."""
    if refine is None or refine is False:
        return None
    name = str(refine).replace("-", "_").strip().lower()
    aliases = {
        "topology": "nni",
        "joint": "nni",
        "joint_greedy": "nni",
        "greedy_topology": "nni",
        "reconfigure": "subtree",
        "subtree_reconfigure": "subtree",
        "all_scales": "subtree",
    }
    name = aliases.get(name, name)
    if name not in {"nni", "subtree"}:
        raise ValueError("topology_refine must be None, 'nni', or 'subtree'.")
    return name


def _normalize_layout_search(search):
    """Normalize an optional offline fixed-plan search mode."""
    if search is None or search is False:
        return None
    name = str(search).replace("-", "_").strip().lower()
    aliases = {
        "ng": "nevergrad",
        "never_grad": "nevergrad",
        "simulated_annealing": "anneal",
        "subtree_anneal": "anneal",
        "annealing": "anneal",
        "quality": "hybrid",
        "combined": "hybrid",
        "full_tree": "hybrid",
    }
    name = aliases.get(name, name)
    if name not in {"nevergrad", "anneal", "hybrid"}:
        raise ValueError(
            "search must be None, 'nevergrad', 'anneal', or 'hybrid'."
        )
    return name


def _normalize_layout_order(order):
    """Normalize quality, geometric presets, or an explicit site order."""
    if order is None:
        return None
    if not isinstance(order, (str, bytes)):
        return tuple(order)
    name = str(order).replace("-", "_").strip().lower()
    aliases = {
        "auto": "quality",
        "best": "quality",
        "best_quality": "quality",
        "row": "row-major",
        "row_major": "row-major",
        "col": "col-major",
        "column": "col-major",
        "column_major": "col-major",
        "snake_col": "snake",
        "snake_column": "snake",
        "snake_col_major": "snake",
        "snake_row": "snake-row-major",
        "folded_snake_col": "folded-snake",
        "folded_snake_column": "folded-snake",
        "folded_snake_col_major": "folded-snake",
        "folded_snake_row": "folded-snake-row-major",
        "hilbert_curve": "hilbert",
        "hilbert_col": "hilbert",
        "hilbert_column": "hilbert",
        "hilbert_col_major": "hilbert",
        "hilbert_row": "hilbert-row-major",
        "alternate_x": "alternate-x",
        "alternate_y": "alternate-y",
        "alternate_z": "alternate-z",
        "coarse_row": "coarse-row-major",
        "coarse_row_major": "coarse-row-major",
        "coarse_col": "coarse-col-major",
        "coarse_column": "coarse-col-major",
        "coarse_col_major": "coarse-col-major",
        "coarse_snake": "coarse-snake",
        "coarse_snake_row": "coarse-snake-row-major",
        "coarse_snake_row_major": "coarse-snake-row-major",
        "coarse_snake_col": "coarse-snake",
        "coarse_snake_column": "coarse-snake",
        "coarse_snake_col_major": "coarse-snake",
        "coarse_alternate_x": "coarse-alternate-x",
        "coarse_alternate_y": "coarse-alternate-y",
        "coarse_alternate_z": "coarse-alternate-z",
        "coarse_folded_snake": "coarse-folded-snake",
        "coarse_folded_snake_row": "coarse-folded-snake-row-major",
        "coarse_folded_snake_row_major": "coarse-folded-snake-row-major",
        "coarse_hilbert": "coarse-hilbert",
        "coarse_hilbert_row": "coarse-hilbert-row-major",
        "coarse_hilbert_row_major": "coarse-hilbert-row-major",
    }
    name = aliases.get(name, name)
    if name == "quality":
        return name
    geometric = {
        "row_major": "row-major",
        "col_major": "col-major",
        "snake": "snake",
        "snake_row_major": "snake-row-major",
        "folded_snake": "folded-snake",
        "folded_snake_row_major": "folded-snake-row-major",
        "alternate_x": "alternate-x",
        "alternate_y": "alternate-y",
        "alternate_z": "alternate-z",
        "coarse-row-major": "coarse-row-major",
        "coarse-col-major": "coarse-col-major",
        "coarse-snake": "coarse-snake",
        "coarse-snake-row-major": "coarse-snake-row-major",
        "coarse-folded-snake": "coarse-folded-snake",
        "coarse-folded-snake-row-major": "coarse-folded-snake-row-major",
        "coarse-hilbert": "coarse-hilbert",
        "coarse-hilbert-row-major": "coarse-hilbert-row-major",
        "coarse-alternate-x": "coarse-alternate-x",
        "coarse-alternate-y": "coarse-alternate-y",
        "coarse-alternate-z": "coarse-alternate-z",
        "hilbert": "hilbert",
        "hilbert_row_major": "hilbert-row-major",
    }
    if name in geometric.values():
        return name
    if name in geometric:
        return geometric[name]
    raise ValueError(
        "order must be None, 'quality', a geometric lattice preset "
        "('row-major', 'col-major', 'snake', 'alternate-x', "
        "'alternate-y', 'alternate-z', 'folded-snake', 'hilbert', "
        "or a coarse-* variant), "
        "or an explicit site permutation."
    )


def _normalize_lattice_shape(shape):
    """Return a validated 2D or 3D ``(Lx, Ly[, Lz])`` lattice shape."""
    if shape is None:
        return None
    if isinstance(shape, (str, bytes)):
        raise TypeError(
            "lattice_shape must be a two- or three-item (Lx, Ly[, Lz]) "
            "sequence."
        )
    try:
        shape = tuple(shape)
    except TypeError as exc:
        raise TypeError(
            "lattice_shape must be a two- or three-item (Lx, Ly[, Lz]) "
            "sequence."
        ) from exc
    if len(shape) not in {2, 3}:
        raise ValueError(
            "lattice_shape must contain exactly (Lx, Ly) or (Lx, Ly, Lz)."
        )
    if any(isinstance(value, bool) for value in shape):
        raise ValueError("lattice_shape dimensions must be positive integers.")
    try:
        shape = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lattice_shape dimensions must be positive integers."
        ) from exc
    if any(value < 1 for value in shape):
        raise ValueError("lattice_shape dimensions must be positive integers.")
    return shape


_COARSE_LAYOUT_BASE_MODES = {
    "coarse-row-major": "row-major",
    "coarse-col-major": "col-major",
    "coarse-snake": "snake",
    "coarse-snake-row-major": "snake-row-major",
    "coarse-folded-snake": "folded-snake",
    "coarse-folded-snake-row-major": "folded-snake-row-major",
    "coarse-hilbert": "hilbert",
    "coarse-hilbert-row-major": "hilbert-row-major",
    "coarse-alternate-x": "alternate-x",
    "coarse-alternate-y": "alternate-y",
    "coarse-alternate-z": "alternate-z",
}


def _normalize_coarse_grain(grain, ndim=2):
    """Normalize a 2D/3D coarse block size, defaulting to two along x."""
    if ndim not in {2, 3}:
        raise ValueError("coarse_grain is only defined for 2D or 3D layouts.")
    default = (2,) + (1,) * (ndim - 1)
    axes = "gx, gy" if ndim == 2 else "gx, gy, gz"
    sequence_description = (
        "two-item" if ndim == 2 else "two- or three-item"
    )
    if grain is None:
        return default
    if isinstance(grain, bool):
        raise ValueError("coarse_grain must contain positive integers.")
    if isinstance(grain, Integral):
        grain = (int(grain),) + (1,) * (ndim - 1)
    elif isinstance(grain, (str, bytes)):
        raise TypeError(
            "coarse_grain must be a positive integer or a "
            f"{sequence_description} sequence."
        )
    else:
        try:
            grain = tuple(grain)
        except TypeError as exc:
            raise TypeError(
                "coarse_grain must be a positive integer or a "
                f"{sequence_description} sequence."
            ) from exc
    if len(grain) == 2 and ndim == 3:
        grain = (*grain, 1)
    if len(grain) != ndim:
        raise ValueError(f"coarse_grain must contain exactly ({axes}).")
    if any(isinstance(value, bool) for value in grain):
        raise ValueError("coarse_grain entries must be positive integers.")
    try:
        grain = tuple(int(value) for value in grain)
    except (TypeError, ValueError) as exc:
        raise ValueError("coarse_grain entries must be positive integers.") from exc
    if any(value < 1 for value in grain):
        raise ValueError("coarse_grain entries must be positive integers.")
    return grain


def _build_lattice_coordinates(shape, mode):
    """Build coordinates through the shared 2D/3D :class:`OneDMap` API."""
    if len(shape) == 2:
        one_d_to_lattice, _ = OneDMap.build(*shape, mode=mode)
    else:
        one_d_to_lattice, _ = OneDMap.build(
            shape[0], shape[1], Lz=shape[2], mode=mode
        )
    return tuple(one_d_to_lattice.values())


def _coarse_mirror_axes(mode, block_coord, block_shape):
    """Return axes to mirror for a 3D coarse block.

    The block traversal and its local traversal use the same path. Mirroring
    the local path at the corresponding block boundaries preserves the
    alternating direction through each coarse layer while still allowing
    partial edge blocks.
    """
    block_x, block_y, block_z = block_coord
    blocks_x, _blocks_y, _blocks_z = block_shape
    mirror = set()

    def toggle(axis):
        if axis in mirror:
            mirror.remove(axis)
        else:
            mirror.add(axis)

    if mode in {"coarse-alternate-x", "coarse-snake-row-major"}:
        if block_y % 2:
            toggle("x")
        if block_z % 2:
            toggle("x")
            toggle("y")
    elif mode in {"coarse-snake", "coarse-alternate-y"}:
        if block_x % 2:
            toggle("y")
        if block_z % 2:
            toggle("x")
            toggle("y")
    elif mode == "coarse-alternate-z":
        if block_y % 2:
            toggle("x")
        line = block_y * blocks_x
        if block_y % 2:
            line += blocks_x - 1 - block_x
        else:
            line += block_x
        if line % 2:
            toggle("z")
    return mirror


def _coarse_lattice_coordinates(Lx, Ly, mode, *, Lz=None, grain=(2, 1)):
    """Return fine coordinates in a 2D/3D block-traversal order."""
    shape = (Lx, Ly) if Lz is None else (Lx, Ly, Lz)
    ndim = len(shape)
    base_mode = _COARSE_LAYOUT_BASE_MODES[mode]
    grain = _normalize_coarse_grain(grain, ndim=ndim)
    block_shape = tuple(
        (length + block - 1) // block
        for length, block in zip(shape, grain)
    )
    block_order = _build_lattice_coordinates(block_shape, base_mode)
    coordinates = []
    for block_coord in block_order:
        block_extent = tuple(
            min(block, length - block_coord[axis] * block)
            for axis, (length, block) in enumerate(zip(shape, grain))
        )
        local_order = _build_lattice_coordinates(block_extent, base_mode)
        if ndim == 2:
            block_x, block_y = block_coord
            width, height = block_extent
            mirror_axis = None
            if (
                mode in {"coarse-alternate-x", "coarse-snake-row-major"}
                and block_y % 2
            ):
                mirror_axis = "x"
            elif (
                mode in {"coarse-snake", "coarse-alternate-y"}
                and block_x % 2
            ):
                mirror_axis = "y"
            for local_x, local_y in local_order:
                if mirror_axis == "x":
                    local_x = width - 1 - local_x
                elif mirror_axis == "y":
                    local_y = height - 1 - local_y
                coordinates.append(
                    (block_x * grain[0] + local_x,
                     block_y * grain[1] + local_y)
                )
            continue

        mirror_axes = _coarse_mirror_axes(mode, block_coord, block_shape)
        for local_coord in local_order:
            local_coord = list(local_coord)
            for axis, name in enumerate(("x", "y", "z")):
                if name in mirror_axes:
                    local_coord[axis] = block_extent[axis] - 1 - local_coord[axis]
            coordinates.append(tuple(
                block_coord[axis] * grain[axis] + local_coord[axis]
                for axis in range(3)
            ))
    return tuple(coordinates)


def _lattice_site_order(
    Lx, Ly, mode, *, Lz=None, site=None, grain=(2, 1)
):
    """Build a logical-qubit permutation from a regular 2D/3D mode."""
    shape = (Lx, Ly) if Lz is None else (Lx, Ly, Lz)
    if mode in _COARSE_LAYOUT_BASE_MODES:
        coordinates = _coarse_lattice_coordinates(
            *shape[:2], Lz=shape[2] if len(shape) == 3 else None,
            mode=mode, grain=grain,
        )
    else:
        coordinates = _build_lattice_coordinates(shape, mode)
    if site is None:
        if len(shape) == 2:
            # Match OneDMap's logical 2D labels: (x, y) -> x * Ly + y.
            site = lambda x, y: x * shape[1] + y
        else:
            # Match the natural x-major flattening of a 3D PEPS lattice.
            site = lambda x, y, z: (
                x * shape[1] * shape[2] + y * shape[2] + z
            )
    if not callable(site):
        raise TypeError("lattice_site must be callable or None.")
    order = tuple(int(site(*coord)) for coord in coordinates)
    size = int(np.prod(shape))
    return normalize_fixed_order(order, range(size), name="lattice order")


def _nevergrad_available():
    """Return whether the optional Nevergrad dependency can be imported."""
    try:
        import_module("nevergrad")
    except ImportError:
        return False
    return True


def _quality_search_mode():
    """Return the complete quality search, with a clear fallback."""
    return "hybrid" if _nevergrad_available() else "anneal"


def _validate_search_budget(value, name):
    """Validate a positive bounded layout-search evaluation budget."""
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _normalize_time_decay(value):
    """Validate an optional newest-event temporal decay factor."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_decay must be in (0, 1] or None.") from exc
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("time_decay must be in (0, 1] or None.")
    return value


def _normalize_time_window(value):
    """Validate an optional trailing event window."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("time_window must be a positive integer or None.")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_window must be a positive integer or None."
        ) from exc
    if value < 1:
        raise ValueError("time_window must be a positive integer or None.")
    return value


def _temporal_event_factors(num_events, *, time_decay=None, time_window=None):
    """Return one newest-event-normalized factor for each stream event."""
    if num_events < 1:
        return ()
    factors = np.ones(int(num_events), dtype=float)
    if time_window is not None and time_window < num_events:
        factors[: num_events - time_window] = 0.0
    if time_decay is not None and time_decay != 1.0:
        ages = np.arange(num_events - 1, -1, -1, dtype=float)
        factors *= np.power(time_decay, ages)
    return tuple(float(factor) for factor in factors)


def _safe_exp2(value):
    """Return ``2**value`` without emitting overflow warnings."""
    if value > np.log2(np.finfo(float).max):
        return float("inf")
    return float(np.exp2(value))


def _normalize_layout_objective(objective):
    """Normalize the tree-layout objective name."""
    name = str(objective).replace("-", "_").strip().lower()
    aliases = {
        "distance": "path",
        "path_length": "path",
        "edge": "congestion",
        "edge_load": "congestion",
        "bond": "congestion",
        "bond_load": "congestion",
        "combined": "hybrid",
        "compress": "compression",
        "accuracy": "compression",
        "bond_growth": "compression",
        "hyperedge": "hypergraph",
        "hyperedges": "hypergraph",
        "hypergraph_load": "hypergraph",
        "per_edge": "hypergraph",
        "per_edge_load": "hypergraph",
        "full": "full_tree",
        "tree": "full_tree",
        "cotengra": "full_tree",
        "all_scales": "full_tree",
    }
    name = aliases.get(name, name)
    if name not in {
        "path", "congestion", "hybrid", "compression", "hypergraph",
        "full_tree",
    }:
        raise ValueError(
            f"Unknown tree layout objective {objective!r}. "
            "Expected 'path', 'congestion', 'compression', 'hypergraph', "
            "'full_tree', or 'hybrid'."
        )
    return name


def _operator_schmidt_rank(payload, support, left_support):
    """Return an operator-Schmidt rank across a support bipartition."""
    support = tuple(support)
    left_support = tuple(left_support)
    left_set = set(left_support)
    if not left_set or left_set == set(support):
        return 1
    default_bound = _operator_schmidt_rank_bound(support, left_support)
    try:
        array = ar.to_numpy(payload)
    except Exception:
        return default_bound
    if array.size != 4 ** len(support):
        return default_bound
    try:
        array = array.reshape((2,) * (2 * len(support)))
        positions = {site: pos for pos, site in enumerate(support)}
        left_positions = [positions[site] for site in left_support]
        right_positions = [
            positions[site] for site in support if site not in left_set
        ]
        axes = (
            left_positions
            + [len(support) + pos for pos in left_positions]
            + right_positions
            + [len(support) + pos for pos in right_positions]
        )
        matrix = array.transpose(axes).reshape(
            4 ** len(left_positions),
            4 ** len(right_positions),
        )
        return max(1, int(np.linalg.matrix_rank(matrix)))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return default_bound


def _submpo_schmidt_rank_bound(payload, support, left_support):
    """Return an MPO-bond upper bound without calling ``to_dense``.

    An MPO's operator Schmidt rank across a site bipartition is bounded by the
    product of the virtual MPO bonds crossing that bipartition.  This is a
    conservative diagnostic, but unlike lowering an MPO to a dense matrix it
    remains cheap for wide supports.  ``None`` means that ``payload`` does not
    expose the Quimb MPO site interface.
    """
    gen_sites = getattr(payload, "gen_sites_present", None)
    site_tag = getattr(payload, "site_tag", None)
    tag_map = getattr(payload, "tag_map", None)
    tensor_map = getattr(payload, "tensor_map", None)
    if not all((callable(gen_sites), callable(site_tag),
                tag_map is not None, tensor_map is not None)):
        return None
    try:
        present = tuple(gen_sites())
        support = tuple(support)
        if set(present) != set(support):
            return None
        tensors = []
        for site in present:
            tids = tuple(tag_map[site_tag(site)])
            if len(tids) != 1:
                return None
            tensors.append(tensor_map[tids[0]])
        left = set(left_support)
        if not left or left == set(support):
            return 1
        rank = 1
        for left_site, right_site, left_tensor, right_tensor in zip(
            present, present[1:], tensors, tensors[1:]
        ):
            if (left_site in left) == (right_site in left):
                continue
            shared = set(left_tensor.inds).intersection(right_tensor.inds)
            if len(shared) != 1:
                return None
            rank *= int(payload.ind_size(next(iter(shared))))
        return max(1, int(rank))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _validate_chi(chi):
    """Coerce and validate an optional ``chi`` selection budget."""
    if chi is None:
        return None
    chi = int(chi)
    if chi < 1:
        raise ValueError("chi must be a positive integer.")
    return chi


def _normalize_arity_candidates(max_arity):
    """Return ``(representative_arity, candidates)`` from a ``max_arity`` arg.

    ``max_arity`` may be a single int (a fixed arity), ``None`` (unbounded), or
    an iterable of candidate arities to *search*. ``candidates`` is ``None``
    unless a search set was given;
    the representative single arity is what the legacy single-plan builders use
    and is the first concrete candidate.
    """
    if max_arity is None:
        return None, None
    if isinstance(max_arity, Integral):
        return int(max_arity), None
    if isinstance(max_arity, (str, bytes)):
        return int(max_arity), None
    if hasattr(max_arity, "__iter__"):
        cand = []
        for a in max_arity:
            key = None if a is None else int(a)
            if key is not None and key < 2:
                raise ValueError("arity candidates must be >= 2 or None.")
            if key not in cand:
                cand.append(key)
        if not cand:
            raise ValueError("max_arity iterable must be non-empty.")
        representative = next((a for a in cand if a is not None), None)
        return representative, tuple(cand)
    return int(max_arity), None



def _chi_cut_fields(plan, chi):
    """Return ``{max_bond_cut, chi_overflow, exact_at_chi}`` for ``plan``.

    ``max_bond_cut`` is the widest qubit bipartition any bond induces.  With a
    finite ``chi`` the structure can hold an arbitrary state exactly only when
    ``2 ** max_bond_cut <= chi``; ``chi_overflow`` is how many qubits the widest
    bond exceeds ``log2(chi)`` (0 when the structure is exact at ``chi``).
    """
    mbc = plan.max_bond_cut()
    fields = {"max_bond_cut": mbc}
    if chi is not None:
        log_chi = float(np.log2(chi))
        fields["chi_overflow"] = max(0.0, mbc - log_chi)
        fields["exact_at_chi"] = mbc <= log_chi
    return fields


def _tree_node_scales(plan):
    """Return hierarchical scales, with leaves at scale zero."""
    scales = {}

    def visit(node):
        if node in scales:
            return scales[node]
        children = tuple(plan.children.get(node, ()))
        if not children:
            scale = 0
        else:
            scale = 1 + max(visit(child) for child in children)
        scales[node] = scale
        return scale

    visit(plan.root)
    return scales


class TreePlan:
    """A rooted tree over ``n`` qubits (any internal-node arity).

    Nodes are integer ids. Leaves map one-to-one to qubits. Optionally, one
    additional qubit can be carried by the structural root via ``root_qubit``;
    this gives a binary top tensor two child bonds plus one open physical leg.
    Other internal nodes carry no physical qubit. The common default is binary
    below a ternary virtual root, but the structure
    supports arbitrary arity so a level can branch into as many subtrees as the
    gate stream suggests. The plan is a pure structure description: it carries
    no tensor data and is consumed by
    :class:`~pepsy.optimizers.tree.TreeOptimizer` to build the tree tensor
    network.
    """

    def __init__(
        self, root, children, parent, qubit_of_leaf, *, root_qubit=None,
        map_mode=None
    ):
        self.root = root
        self.children = dict(children)
        self.parent = dict(parent)
        self.qubit_of_leaf = dict(qubit_of_leaf)
        self.leaf_of_qubit = {q: nid for nid, q in self.qubit_of_leaf.items()}
        self.root_qubit = (
            None if root_qubit is None else int(root_qubit)
        )
        self._map_mode = (
            None
            if map_mode is None
            else _normalize_layout_order(map_mode)
        )
        if self.root_qubit is not None and self.root in self.qubit_of_leaf:
            raise ValueError(
                "the root cannot carry both a leaf qubit and root_qubit; "
                "insert a unary structural root above the leaf."
            )
        self.qubit_of_node = dict(self.qubit_of_leaf)
        if self.root_qubit is not None:
            self.qubit_of_node[self.root] = self.root_qubit
        self.node_of_qubit = {
            q: nid for nid, q in self.qubit_of_node.items()
        }
        self.n = len(self.node_of_qubit)
        self._path_cache = {}

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_order(cls, order, *, weights=None, structure="quality",
                   max_arity=2, community_frac=0.35, star_frac=0.75,
                   dense_max=512, root_qubit=None,
                   top_arity=_DEFAULT_TOP_ARITY, map_mode=None):
        """Build a rooted tree by recursive partition of ``order``.

        Parameters
        ----------
        order : sequence of int
            The qubit labels to place as leaves. When ``root_qubit`` is given,
            ``order`` contains every other qubit and the combined labels must
            still be ``0..n-1``.
        weights : mapping, optional
            Unordered ``(qi, qj) -> weight`` interaction weights.  Used to
            spectrally reorder each recursion level (``structure="quality"``)
            and to detect communities (``structure="adaptive"``).
        structure : {"quality", "balanced", "adaptive"}
            ``"quality"`` spectrally (Fiedler) reorders each level before
            splitting; ``"balanced"`` splits the given ``order`` directly;
            ``"adaptive"`` partitions each level into strongly coupled
            communities of the induced interaction graph so the arity of a node
            follows the gate connectivity, and collapses a densely coupled
            block (a near-clique) into a single flat *star* node.  All three
            respect ``max_arity``.
        max_arity : int or None
            Maximum number of children per internal node.  ``2`` (default)
            reproduces the strictly-binary tree; larger values give flatter
            ``k``-ary trees with shorter geodesics; ``None`` leaves the arity
            unbounded (``"adaptive"`` may then emit wide star nodes).
        community_frac : float
            For ``structure="adaptive"``: an induced edge is treated as a strong
            intra-community link when its weight is at least ``community_frac``
            times the largest induced edge weight at that level.
        star_frac : float
            For ``structure="adaptive"``: when a block is a single strong
            community whose fraction of present strong edges is at least
            ``star_frac`` (a near-clique), it becomes a flat star of leaves
            (all pairwise geodesics length two) instead of being bisected.
        dense_max : int
            Maximum subsystem size for dense spectral reordering.
        root_qubit : int, optional
            Qubit label carried by the top tensor rather than a leaf.
        map_mode : str, optional
            Canonical geometric label for the leaf order. Tree-native geometry
            uses ``coarse-*`` names; this is metadata only when an explicit
            qubit order is supplied.
        top_arity : int or None, optional
            Number of virtual child bonds on the structural root. By default,
            ``max_arity=2`` uses ``top_arity=3`` when there are at least three
            leaf qubits and no ``root_qubit``. Set ``top_arity=None`` or
            ``top_arity=2`` to use the ordinary binary root. A value greater
            than two is incompatible with ``root_qubit`` because that would
            make a rank-four root tensor.
        """
        order = list(order)
        if not order and root_qubit is None:
            raise ValueError("order must contain at least one qubit.")
        try:
            order = [int(q) for q in order]
        except (TypeError, ValueError) as exc:
            raise ValueError("order must contain integer qubit labels.") from exc
        if root_qubit is not None:
            try:
                root_qubit = int(root_qubit)
            except (TypeError, ValueError) as exc:
                raise ValueError("root_qubit must be an integer or None.") from exc
        if max_arity is not None:
            max_arity = int(max_arity)
            if max_arity < 2:
                raise ValueError("max_arity must be >= 2 (or None).")
        if top_arity is _DEFAULT_TOP_ARITY:
            top_arity = (
                3
                if root_qubit is None and max_arity == 2 and len(order) >= 3
                else None
            )
        if top_arity is not None:
            try:
                top_arity = int(top_arity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "top_arity must be an integer >= 2 or None."
                ) from exc
            if top_arity < 2:
                raise ValueError("top_arity must be >= 2 or None.")
            if top_arity > len(order):
                raise ValueError(
                    "top_arity cannot exceed the number of non-root qubits."
                )
            if root_qubit is not None and top_arity != 2:
                raise ValueError(
                    "top_arity > 2 cannot be combined with root_qubit: "
                    "the root would have a rank-four tensor."
                )
        all_qubits = order + ([] if root_qubit is None else [root_qubit])
        if sorted(all_qubits) != list(range(len(all_qubits))):
            raise ValueError(
                "leaf order plus root_qubit must be a permutation of "
                "qubit labels 0..n-1."
            )
        if structure not in {"quality", "balanced", "adaptive"}:
            raise ValueError(
                "structure must be 'quality', 'balanced', or 'adaptive'."
            )
        counter = [0]
        children = {}
        parent = {}
        qubit_of_leaf = {}

        def new_node():
            nid = counter[0]
            counter[0] += 1
            return nid

        def induced(nodes):
            if not weights:
                return {}
            node_set = set(nodes)
            return {
                edge: w
                for edge, w in weights.items()
                if edge[0] in node_set and edge[1] in node_set
            }

        def make_leaf(q):
            nid = new_node()
            children[nid] = ()
            qubit_of_leaf[nid] = q
            return nid

        def make_internal(child_ids):
            nid = new_node()
            children[nid] = tuple(child_ids)
            for c in child_ids:
                parent[c] = nid
            return nid

        def kary_split(qs, arity=None):
            """Split ``qs`` into up to ``arity`` contiguous balanced parts.

            Cut points use ``floor(i * L / k)`` so the two-way case reproduces
            the previous ``mid = len(qs) // 2`` bisection exactly.
            """
            length = len(qs)
            k = max_arity if arity is None else arity
            k = length if k is None else k
            k = min(k, length)
            if k <= 1:
                return [qs]
            cuts = [length * i // k for i in range(k + 1)]
            groups = [qs[cuts[i]:cuts[i + 1]] for i in range(k)]
            return [g for g in groups if g]

        def strong_adjacency(qs):
            """Return ``(adjacency, threshold)`` for the induced graph or None."""
            sub = induced(qs)
            if not sub:
                return None
            max_w = max(sub.values())
            if max_w <= 0.0:
                return None
            adj = _gate_stream_adjacency(qs, sub)
            return adj, float(community_frac) * max_w

        def communities(qs):
            """Return strongly coupled communities of ``qs`` or ``None``."""
            info = strong_adjacency(qs)
            if info is None:
                return None
            adj, thresh = info
            rank = {q: i for i, q in enumerate(qs)}
            seen = set()
            comps = []
            for start in qs:
                if start in seen:
                    continue
                stack = [start]
                seen.add(start)
                comp = []
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for nb, w in adj[cur].items():
                        if nb not in seen and w >= thresh:
                            seen.add(nb)
                            stack.append(nb)
                comps.append(sorted(comp, key=lambda x: rank[x]))
            comps.sort(key=lambda c: rank[c[0]])
            return comps

        def is_near_clique(qs):
            """Return ``True`` when strong edges nearly fully connect ``qs``."""
            info = strong_adjacency(qs)
            if info is None:
                return False
            adj, thresh = info
            m = len(qs)
            if m < 3:
                return False
            strong = 0
            for i, a in enumerate(qs):
                for b in qs[i + 1:]:
                    if adj[a].get(b, 0.0) >= thresh:
                        strong += 1
            total = m * (m - 1) // 2
            return total > 0 and strong / total >= float(star_frac)

        def split(qs, arity=None):
            """Return the child qubit-groups for the internal node over ``qs``."""
            arity_limit = max_arity if arity is None else arity
            groups = None
            if structure == "adaptive" and arity is None:
                comps = communities(qs)
                if comps is not None and len(comps) >= 2:
                    if arity_limit is None or len(comps) <= arity_limit:
                        groups = comps
                    # else: too many communities for the arity cap; fall back to
                    # a spectral k-ary split (deeper recursion still resolves
                    # communities inside each part).
                elif (arity_limit is None or len(qs) <= arity_limit) \
                        and is_near_clique(qs):
                    # A densely coupled block is flattest as a star of leaves.
                    groups = [[q] for q in qs]
            if groups is None:
                qs2 = qs
                if structure in ("quality", "adaptive"):
                    spectral = _gate_stream_spectral_order(
                        qs, induced(qs), dense_max=dense_max
                    )
                    if spectral:
                        qs2 = spectral
                groups = kary_split(qs2, arity_limit)
            return groups

        def build(qs, *, is_root=False):
            qs = list(qs)
            if len(qs) == 1:
                return make_leaf(qs[0])
            root_limit = top_arity if is_root else None
            groups = split(qs, root_limit)
            if len(groups) < 2:
                # Degenerate split (e.g. all mass in one part): force a split so
                # recursion always makes progress.
                mid = max(1, len(qs) // 2)
                groups = [qs[:mid], qs[mid:]]
            child_ids = [build(g) for g in groups]
            return make_internal(child_ids)

        if order:
            root = build(order, is_root=True)
            if root_qubit is not None and root in qubit_of_leaf:
                # With one non-root qubit, ``build`` returns that physical
                # leaf itself. The top qubit needs its own tensor, so insert a
                # unary structural root rather than putting two physical legs
                # on the same node.
                root = make_internal((root,))
        else:
            root = new_node()
            children[root] = ()
        return cls(
            root,
            children,
            parent,
            qubit_of_leaf,
            root_qubit=root_qubit,
            map_mode=map_mode,
        )

    @classmethod
    def from_children(
        cls, children, qubit_of_leaf, *, root=None, root_qubit=None,
        map_mode=None
    ):
        """Build and validate a :class:`TreePlan` from an explicit tree.

        This is the general entry point for arbitrary (non-binary) trees: a
        caller or a custom layout strategy supplies the ``children`` map and the
        leaf-to-qubit assignment, and this validates that they describe a single
        rooted tree covering qubits ``0..n-1`` exactly once.

        Parameters
        ----------
        children : mapping
            ``node_id -> tuple(child_ids)``.  Leaves map to an empty tuple.
        qubit_of_leaf : mapping
            ``leaf_id -> qubit`` for every leaf node.
        root : int, optional
            The root node id.  Inferred as the unique parent-less node when
            omitted.
        root_qubit : int, optional
            Qubit label carried by ``root`` rather than by a leaf.
        map_mode : str, optional
            Canonical geometric label retained when rebuilding a plan.
        """
        children = {int(k): tuple(int(c) for c in v)
                    for k, v in children.items()}
        qubit_of_leaf = {int(k): int(q) for k, q in qubit_of_leaf.items()}

        parent = {}
        for nid, ch in children.items():
            for c in ch:
                if c in parent:
                    raise ValueError(f"node {c} has more than one parent")
                if c not in children:
                    raise ValueError(
                        f"child {c} of node {nid} is not a declared node"
                    )
                parent[c] = nid

        roots = [nid for nid in children if nid not in parent]
        if root is None:
            if len(roots) != 1:
                raise ValueError(
                    f"expected exactly one root, found {sorted(roots)}"
                )
            root = roots[0]
        else:
            root = int(root)
            if root not in children or root in parent:
                raise ValueError(f"invalid root {root}")
        if root_qubit is not None:
            try:
                root_qubit = int(root_qubit)
            except (TypeError, ValueError) as exc:
                raise ValueError("root_qubit must be an integer or None.") from exc

        leaves = set()
        for nid, ch in children.items():
            if ch:
                if nid in qubit_of_leaf:
                    raise ValueError(
                        f"internal node {nid} must not have a qubit"
                    )
            else:
                leaves.add(nid)
                if (
                    nid not in qubit_of_leaf
                    and not (nid == root and root_qubit is not None)
                ):
                    raise ValueError(f"leaf node {nid} is missing a qubit")
        expected_leaf_nodes = (
            leaves - {root}
            if root_qubit is not None and not children[root]
            else leaves
        )
        if set(qubit_of_leaf) != expected_leaf_nodes:
            raise ValueError(
                "qubit_of_leaf must map exactly the leaf nodes"
            )
        if root_qubit is not None and root in qubit_of_leaf:
            raise ValueError("the root cannot carry both a leaf and root qubit")
        qs = sorted(
            [
                *qubit_of_leaf.values(),
                *([] if root_qubit is None else [root_qubit]),
            ]
        )
        if qs != list(range(len(qs))):
            raise ValueError(
                "leaf qubits plus root_qubit must be 0..n-1 without repeats"
            )

        seen = set()
        stack = [root]
        while stack:
            x = stack.pop()
            if x in seen:
                raise ValueError("cycle detected in tree")
            seen.add(x)
            stack.extend(children[x])
        if seen != set(children):
            unreached = set(children) - seen
            raise ValueError(
                f"nodes not reachable from root {root}: {sorted(unreached)}"
            )
        return cls(
            root,
            children,
            parent,
            qubit_of_leaf,
            root_qubit=root_qubit,
            map_mode=map_mode,
        )

    #: Fixed number of legs on the top tensor of a :meth:`build_layered` tree.
    LAYERED_ROOT_ARITY = 3

    @classmethod
    def build_layered(cls, order, *, block_size=4, root_qubit=None):
        """Build a fixed-structure layered tree with a ternary top tensor.

        The structure is fixed; only ``block_size`` is tunable:

        * **First layer** (leaf-parent "blocking" nodes): each node groups
          ``block_size`` consecutive qubits from ``order`` into one virtual
          bond.  This is the only choosable layer.
        * **Middle layers**: strictly binary (two bonds in, one out).
        * **Top tensor (root)**: always :attr:`LAYERED_ROOT_ARITY` (three)
          children when there are at least three blocks; fewer only in the
          degenerate small-``n`` case where three blocks do not exist.

        Parameters
        ----------
        order : sequence of int
            Leaf-qubit labels in the desired spatial order. Strongly coupled
            qubits should be consecutive so they land in the same block; use
            :meth:`TreeLayoutFinder.qubit_order` to obtain an
            entanglement-adapted ordering, or
            :meth:`TreeLayoutFinder.recommend_layered` to also search
            ``block_size``. Together with an optional ``root_qubit``, the
            labels must cover ``0..n-1``.
        block_size : int
            Number of physical qubits per leaf-parent node. Default 4.
        root_qubit : int, optional
            Qubit label carried by the top tensor rather than a leaf.
        """
        order = list(order)
        if not order and root_qubit is None:
            raise ValueError("order must be non-empty.")
        order = [int(q) for q in order]
        if root_qubit is not None:
            root_qubit = int(root_qubit)
        all_qubits = order + ([] if root_qubit is None else [root_qubit])
        n = len(order)
        if sorted(all_qubits) != list(range(len(all_qubits))):
            raise ValueError(
                "leaf order plus root_qubit must be a permutation of 0..n-1."
            )
        if not isinstance(block_size, Integral):
            raise ValueError("block_size must be an integer >= 1.")
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        counter = [0]
        children_map = {}
        qubit_of_leaf = {}

        def new_node():
            nid = counter[0]
            counter[0] += 1
            return nid

        # Leaves: one node per qubit in the given order.
        leaf_ids = []
        for q in order:
            nid = new_node()
            children_map[nid] = ()
            qubit_of_leaf[nid] = q
            leaf_ids.append(nid)

        if not leaf_ids:
            root_nid = new_node()
            children_map[root_nid] = ()
            return cls.from_children(
                children_map,
                qubit_of_leaf,
                root=root_nid,
                root_qubit=root_qubit,
            )

        # First layer: group block_size leaves into one blocking node.
        # A single-leaf chunk skips the parent and uses the leaf directly.
        block_nodes = []
        for start in range(0, n, block_size):
            chunk = leaf_ids[start: start + block_size]
            if len(chunk) == 1:
                block_nodes.append(chunk[0])
            else:
                nid = new_node()
                children_map[nid] = tuple(chunk)
                block_nodes.append(nid)

        # Middle layers: binary tree over block_nodes.
        def binary_subtree(nodes):
            if len(nodes) == 1:
                return nodes[0]
            mid = len(nodes) // 2
            left = binary_subtree(nodes[:mid])
            right = binary_subtree(nodes[mid:])
            nid = new_node()
            children_map[nid] = (left, right)
            return nid

        # Top tensor: fixed ternary root (or fewer only when < 3 blocks exist).
        num_blocks = len(block_nodes)
        if num_blocks == 1:
            # The blocking node is already a valid root.  In particular, do
            # not add a unary wrapper for n=1 or n <= block_size: it adds a
            # useless bond and makes the fixed layered family less efficient.
            # The exception is a physical root over one physical leaf: those
            # two qubits require distinct tensors joined by one bond.
            root_nid = block_nodes[0]
            if root_qubit is not None and not children_map[root_nid]:
                child = root_nid
                root_nid = new_node()
                children_map[root_nid] = (child,)
            return cls.from_children(
                children_map,
                qubit_of_leaf,
                root=root_nid,
                root_qubit=root_qubit,
            )
        root_arity = min(cls.LAYERED_ROOT_ARITY, num_blocks)
        if num_blocks <= root_arity:
            # Fewer blocks than the target arity: root takes them all directly.
            root_nid = new_node()
            children_map[root_nid] = tuple(block_nodes)
        else:
            # Split blocks into root_arity contiguous groups; each group is a
            # binary middle subtree whose root becomes a direct child of the
            # top tensor.
            root_children = []
            for i in range(root_arity):
                start = num_blocks * i // root_arity
                end = num_blocks * (i + 1) // root_arity
                root_children.append(binary_subtree(block_nodes[start:end]))
            root_nid = new_node()
            children_map[root_nid] = tuple(root_children)

        return cls.from_children(
            children_map,
            qubit_of_leaf,
            root=root_nid,
            root_qubit=root_qubit,
        )

    # -- queries --------------------------------------------------------------

    def nodes(self):
        """Return all node ids (leaves and internal nodes)."""
        return list(self.children.keys())

    def leaves(self):
        """Return the leaf node ids."""
        return list(self.qubit_of_leaf.keys())

    @property
    def map_mode(self):
        """Canonical geometric label for this tree's leaf layout, if known."""

        return self._map_mode

    def mpo_order(self, *, include_root=True):
        """Return the deterministic logical-site order for a tree MPO.

        The leaf positions are ordered by their structural node ids, matching
        the order used by :class:`TreeLayoutFinder` when it refines a plan.
        When ``root_qubit`` is present, that physical site is placed first by
        default, followed by the ordinary leaf positions. The result is a
        permutation of ``0 .. n - 1`` and is suitable for constructing a
        layout-aware chain MPO whose sites are subsequently routed over this
        tree.

        Parameters
        ----------
        include_root : bool, optional
            Include a physical site carried by the structural root. The
            default is ``True``; pass ``False`` to obtain only the leaf order.
        """
        order = tuple(
            self.qubit_of_leaf[nid]
            for nid in sorted(self.qubit_of_leaf)
        )
        if include_root and self.root_qubit is not None:
            return (self.root_qubit, *order)
        return order

    def build_tree_operator(self, hamiltonian, **kwargs):
        """Build the canonical :class:`TreeMPO` operator for this plan.

        The returned object is the native `TreeMPO`; its
        ``.tree_networks`` and ``.expectation`` expose the TreePlan-routed
        representation. A chain MPO, if needed, is built separately with the
        model's ``to_mpo`` method. With
        mixed native charges, one public ``TreeMPO`` contains one homogeneous
        network per charge. ``charge_sectors=True`` remains available when
        separate sector objects are specifically desired.
        """
        from .operators import build_tree_operator

        return build_tree_operator(self, hamiltonian, **kwargs)

    # Compatibility spelling retained while ``build_tree_operator`` becomes
    # the single plan-facing tree operator builder.
    to_tree_mpo = build_tree_operator

    def is_leaf(self, nid):
        return len(self.children.get(nid, ())) == 0

    def max_arity(self):
        """Return the largest number of children over all internal nodes."""
        return max((len(ch) for ch in self.children.values()), default=0)

    @property
    def top_arity(self):
        """Return the number of virtual child bonds on the structural root."""
        return len(self.children.get(self.root, ()))

    def virtual_degree(self, nid):
        """Return the number of virtual tree bonds incident on ``nid``."""
        if nid not in self.children:
            raise ValueError(f"node {nid!r} is not present in the tree")
        return len(self.children[nid]) + int(nid in self.parent)

    def max_virtual_degree(self):
        """Return the largest number of virtual bonds on any tensor."""
        return max(
            (self.virtual_degree(nid) for nid in self.children),
            default=0,
        )

    def max_tensor_rank(self):
        """Return the largest number of virtual/physical legs on a node."""
        return max(
            (
                self.virtual_degree(nid)
                + int(nid in self.qubit_of_node)
                for nid in self.children
            ),
            default=0,
        )

    def is_strictly_binary(self):
        """Return ``True`` when every internal node has exactly two children."""
        return all(len(ch) in (0, 2) for ch in self.children.values())

    def is_binary(self, *, allow_ternary_root=True):
        """Return whether the tree is binary below an optional ternary root.

        A conventional binary TTN has two child bonds entering every
        non-root internal tensor and one parent bond leaving it. Its top tensor
        has no parent, so it may carry three child bonds without increasing
        the maximum tensor rank. Pass ``allow_ternary_root=False`` to request
        the older strictly-binary predicate.
        """
        if not allow_ternary_root:
            return self.is_strictly_binary()
        for nid, children in self.children.items():
            if not children:
                continue
            allowed = (2, 3) if nid == self.root else (2,)
            if len(children) not in allowed:
                return False
        # A ternary virtual root is binary only when it has no additional
        # physical leg. This also keeps explicit hand-built rank-four roots
        # out of the binary predicate.
        return self.max_tensor_rank() <= 3

    def max_bond_cut(self):
        """Return the largest qubit bipartition induced by any tree bond.

        Every parent-child bond splits the qubits into the child's subtree
        (``k`` qubits) and the rest (``n - k``).  The Schmidt rank that bond can
        carry is bounded by ``2 ** min(k, n - k)``, so this maximum
        ``min(k, n - k)`` over all bonds is a purely structural, ``chi``-free
        accuracy ceiling: the tree can represent an *arbitrary* state exactly
        only when ``chi >= 2 ** max_bond_cut``.  A structure whose
        ``max_bond_cut`` exceeds ``log2(chi)`` must truncate at its widest bond
        regardless of the gate stream.
        """
        # One post-order pass to size every subtree, then reduce over bonds.
        visit = []
        stack = [self.root]
        while stack:
            x = stack.pop()
            visit.append(x)
            stack.extend(self.children[x])
        size = {}
        for x in reversed(visit):
            ch = self.children[x]
            local = 1 if x in self.qubit_of_node else 0
            size[x] = local + sum(size[c] for c in ch)
        best = 0
        for x, s in size.items():
            if x == self.root:
                continue
            best = max(best, min(s, self.n - s))
        return best

    def node_path(self, a, b):
        """Return the node id path from node ``a`` to node ``b`` (inclusive)."""
        if a not in self.children or b not in self.children:
            raise ValueError(f"nodes {a!r} and {b!r} must belong to the tree")
        cached = self._path_cache.get((a, b))
        if cached is not None:
            return list(cached)
        if a == b:
            result = [a]
            self._path_cache[(a, b)] = tuple(result)
            return result
        ancestors = []
        x = a
        while x is not None:
            ancestors.append(x)
            x = self.parent.get(x)
        depth = {v: i for i, v in enumerate(ancestors)}
        tail = []
        x = b
        while x not in depth:
            tail.append(x)
            x = self.parent.get(x)
            if x is None:
                raise ValueError("nodes are not in the same tree")
        lca = x
        result = ancestors[: depth[lca] + 1] + list(reversed(tail))
        self._path_cache[(a, b)] = tuple(result)
        return result

    def subtree_qubit_masks(self):
        """Return an integer bit mask of qubits below every node.

        Integer masks make repeated layout and preflight cut tests much cheaper
        than rebuilding Python ``set`` objects for every edge. Python integers
        remain exact for arbitrary qubit counts.
        """
        visit = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            visit.append(node)
            stack.extend(self.children[node])
        masks = {}
        for node in reversed(visit):
            mask = 0
            q = self.qubit_of_node.get(node)
            if q is not None:
                mask |= 1 << q
            for child in self.children[node]:
                mask |= masks[child]
            masks[node] = mask
        return masks

    def tree_distance(self, qa, qb):
        """Return the node-path length between physical qubits ``qa`` and ``qb``."""
        na = self.node_of_qubit[qa]
        nb = self.node_of_qubit[qb]
        return len(self.node_path(na, nb)) - 1

    def remove_qubit(self, q):
        """Return a plan with physical qubit ``q`` removed and labels compacted."""
        q = int(q)
        if q == self.root_qubit:
            if self.n <= 1:
                raise ValueError("cannot remove the only qubit from a tree.")
            qubit_of_leaf = {
                node: old_q - 1 if old_q > q else old_q
                for node, old_q in self.qubit_of_leaf.items()
            }
            return type(self).from_children(
                self.children,
                qubit_of_leaf,
                root=self.root,
                root_qubit=None,
            )
        return self.remove_leaf(q)

    def remove_leaf(self, q):
        """Return a plan with qubit ``q`` capped and its unary parent removed.

        The remaining logical labels are compacted in the same way as a
        one-dimensional MPS cap: labels above ``q`` shift down by one. The
        surviving parent node is retained when possible, which keeps tensor
        identities stable for callers holding live node references.
        """
        if q not in self.leaf_of_qubit:
            raise ValueError(f"qubit {q!r} is not present in the tree.")
        if self.n <= 1:
            raise ValueError("cannot remove the only qubit from a tree.")
        leaf = self.leaf_of_qubit[q]
        parent = self.parent.get(leaf)
        if parent is None:
            raise ValueError("cannot remove the root leaf from a multi-qubit tree.")

        children = {node: tuple(ch) for node, ch in self.children.items()}
        qubit_of_leaf = dict(self.qubit_of_leaf)
        children[parent] = tuple(c for c in children[parent] if c != leaf)
        del children[leaf]
        del qubit_of_leaf[leaf]

        # A virtual-only tree node may not become unary. A physical root is
        # different: its one child plus root physical leg is still a meaningful
        # rank-two top tensor, so retain that unary structural root.
        physical_root = (
            parent == self.root and self.root_qubit is not None
        )
        if len(children[parent]) == 1 and not physical_root:
            child = children[parent][0]
            children[parent] = children[child]
            del children[child]
            if child in qubit_of_leaf:
                qubit_of_leaf[parent] = qubit_of_leaf.pop(child)

        for node, old_q in tuple(qubit_of_leaf.items()):
            if old_q > q:
                qubit_of_leaf[node] = old_q - 1
        root_qubit = self.root_qubit
        if root_qubit is not None and root_qubit > q:
            root_qubit -= 1
        return type(self).from_children(
            children,
            qubit_of_leaf,
            root=self.root,
            root_qubit=root_qubit,
        )

    def __repr__(self):
        n_internal = sum(1 for nid in self.nodes() if not self.is_leaf(nid))
        root_site = (
            ""
            if self.root_qubit is None
            else f", root_qubit={self.root_qubit}"
        )
        return (
            f"TreePlan(n={self.n}, root={self.root}, "
            f"internal_nodes={n_internal}, "
            f"max_arity={self.max_arity()}, top_arity={self.top_arity}"
            f", map_mode={self.map_mode!r}"
            f"{root_site})"
        )


class TreeLayoutFinder:
    """Find a rooted tree structure adapted to a gate stream.

    Parameters
    ----------
    gates : bundled gate stream, optional
        ``[(gate, where), ...]`` entries.  Two-qubit ``where`` supports define
        the weighted interaction graph. This finder does not accept a tensor
        network state: pass that separately as ``state=`` to
        :class:`TreeOptimizer`. Ignored when ``supports`` is given.
    n : int, optional
        Number of qubits.  Inferred from the stream when omitted.
    root_qubit : int, optional
        Designated qubit carried by the top tensor instead of a leaf. It remains
        part of every path, Steiner-subtree, and congestion calculation.
    supports : sequence of sequences, optional
        Explicit interaction supports, used instead of extracting them from
        ``gates``.
    structure : {"quality", "balanced", "adaptive"}
        Partition strategy passed to :meth:`TreePlan.from_order`. ``"quality"``
        and ``"balanced"`` build binary trees below the optional ternary root
        when ``max_arity=2``;
        ``"adaptive"`` lets each level branch into its strongly coupled
        communities so the arity follows the gate connectivity.
    max_arity : int, None, or iterable of ints
        Maximum children per internal node.  A scalar builds one fixed tree
        (``2`` gives the binary tree; larger values or ``None`` give flatter /
        wider trees).  An iterable of candidate arities makes :meth:`run` *search*
        them and keep the objective-best plan. The default is the scalar
        ``2``.
    top_arity : int or None, optional
        Override the structural root's number of virtual child bonds. With
        the default ``max_arity=2``, omitted ``top_arity`` selects ``3`` when
        possible, so the top tensor has three virtual legs while all non-root
        internal tensors remain two-in/one-out. Set ``top_arity=None`` or
        ``top_arity=2`` to opt out. It cannot be greater than two with
        ``root_qubit``.
    chi : int, optional
        Bond-dimension budget used when an explicit iterable of arities is
        searched to prefer plans that stay exact at ``chi`` (see
        :meth:`recommend_arities`). The fixed default geometry does not
        allocate tensors or perform truncations.
        :class:`TreeOptimizer` forwards its own ``chi`` here automatically.
    community_frac : float
        Strong-edge fraction for ``structure="adaptive"`` (see
        :meth:`TreePlan.from_order`).
    star_frac : float
        Near-clique density threshold for ``structure="adaptive"`` star nodes
        (see :meth:`TreePlan.from_order`).
    dense_max : int
        Maximum subsystem size for dense spectral reordering.
    objective : {"path", "congestion", "compression", "hypergraph", "full_tree", "hybrid"}
        Layout objective. `"path"` preserves the co-occurrence/path-length
        heuristic; `"congestion"` selects among layout candidates using the
        predicted operator-Schmidt load on tree edges. `"hybrid"` combines
        normalized path, peak-edge-load, and total-edge-load costs using
        ``hybrid_weights``.
        `"compression"` adds a local tensor-size proxy to the edge-load
        objective. `"hypergraph"` is the direct multi-site mode: it ranks
        plans from the full support hyperedges and per-edge Schmidt loads,
        then applies bounded leaf and binary-topology refinement by default.
        `"full_tree"` evaluates dynamic bond pressure, predicted ``chi``
        overflow, tensor width, estimated work, write volume, and route length
        across every tree scale. It is
        the high-quality, Cotengra-inspired mode; ``order="quality"`` selects
        it automatically and enables its bounded search stages.
    order : {None, "quality", geometric preset} or sequence, optional
        Optional high-quality offline mode. `"quality"` means
        `objective="full_tree"` and enables bounded greedy leaf refinement,
        all-scale subtree topology refinement, and hybrid
        Nevergrad/annealing search. Named two- or three-dimensional lattice
        presets (`"row-major"`, `"snake"`, `"alternate-x"`,
        `"alternate-y"`, `"alternate-z"`, `"folded-snake"`, and
        `"hilbert"`, plus their supported `coarse-*` variants) require
        `lattice_shape=` and build an exact balanced tree over that traversal.
        Omitted keeps the fast deterministic objective selected by `objective`.
        An explicit site permutation builds a fixed tree without refinement.
    map_mode : str, optional
        Alias for a named geometric ``order``. For a tree tensor network use
        the canonical ``coarse-*`` spelling, for example
        ``map_mode="coarse-alternate-x"``. It cannot be combined with
        ``order``.
    lattice_shape : pair or triple of int, optional
        The `(Lx, Ly)` or `(Lx, Ly, Lz)` shape used by named geometric `order`
        presets. The product must equal `n`.
    lattice_site : callable, optional
        Optional `(x, y) -> qubit` or `(x, y, z) -> qubit` mapper for named
        geometric presets. The default is `x * Ly + y` in 2D and
        `x * Ly * Lz + y * Lz + z` in 3D.
    coarse_grain : int or pair/triple of int, optional
        Fine sites per coarse traversal block for `coarse-*` orders. In 2D a
        scalar `g` means `(g, 1)`; in 3D it means `(g, 1, 1)`. A 3D pair
        `(gx, gy)` is accepted as `(gx, gy, 1)`. The default groups two
        neighboring x sites. Edge blocks are allowed to be smaller. This
        changes only the leaf traversal order; it never merges tensors.
    hybrid_weights : mapping or sequence of three floats, optional
        Weights for the hybrid path, maximum edge load, and total edge load.
        The default is ``(1.0, 1.0, 0.25)``.
    refine : {None, "greedy"}
        Optional fixed-plan local search used by :meth:`run` and recommendation
        methods. `"greedy"` tries adjacent leaf-label swaps before simulation;
        it never changes a live :class:`TreeOptimizer` tree.
    topology_refine : {None, "nni", "subtree"}
        Optional joint topology refinement. `"nni"` tries bounded
        nearest-neighbor interchange moves on binary internal edges;
        `"subtree"` reconfigures descendant subtrees at all scales. Both
        retain only accepted candidates and never change a live
        :class:`TreeOptimizer` tree.
    refine_budget : int, optional
        Maximum greedy swap proposals per candidate plan. Defaults to at most
        64 proposals when refinement is enabled.
    topology_budget : int, optional
        Maximum topology proposals per candidate plan. Defaults to at most 64
        proposals when topology refinement is enabled. For ``"subtree"``,
        proposals are sampled across the available descendant scales.
    search : {None, "nevergrad", "anneal", "hybrid"}
        Optional offline derivative-free refinement. It is never run unless
        requested. `"nevergrad"` refines leaf order and requires the optional
        package; `"anneal"` performs bounded simulated annealing over subtree
        reconfigurations and has no additional dependency. `"hybrid"` splits
        the budget between subtree annealing and Nevergrad leaf refinement;
        it falls back to annealing if Nevergrad is unavailable.
    search_budget : int
        Number of offline search evaluations per candidate plan. For
        ``search="anneal"``, this is the number of subtree proposals; for
        ``search="hybrid"``, it is the shared total split between annealing
        and Nevergrad.
    seed : int
        Reproducible seed used by the optional Nevergrad stage.
    nevergrad_optimizer : str
        Nevergrad registry optimizer name, default ``"OnePlusOne"``.
    weight_mode : {"count", "auto", "angle", "operator_schmidt"}
        Event weighting used for the interaction graph. `"count"` is the
        backward-compatible default.
    time_decay : float, optional
        If supplied, multiply an event's weight by ``time_decay ** age`` where
        the newest event has age zero. Values must be in ``(0, 1]``.
    time_window : int, optional
        If supplied, only the final ``time_window`` stream events contribute to
        layout scoring and predicted edge load.
    """

    def __init__(self, gates=None, n=None, *, supports=None, structure="quality",
                 max_arity=2, top_arity=_DEFAULT_TOP_ARITY,
                 community_frac=0.35, star_frac=0.75,
                 dense_max=512, objective="path", weight_mode="count", chi=None,
                 max_operator_qubits=8, hybrid_weights=None, refine=None,
                 refine_budget=None, topology_refine=None, topology_budget=None,
                 search=None, search_budget=128, seed=0,
                 nevergrad_optimizer="OnePlusOne", order=None, map_mode=None,
                 root_qubit=None,
                 lattice_shape=None, lattice_site=None, coarse_grain=(2, 1),
                 time_decay=None, time_window=None):
        if map_mode is not None:
            if order is not None:
                raise TypeError("map_mode and order cannot both be supplied")
            order = map_mode
        if (
            _looks_like_tree_tensor_network(gates)
            or _looks_like_tree_tensor_network(supports)
        ):
            raise TypeError(
                "TreeLayoutFinder accepts a circuit gate stream or supports, "
                "not a TreeTensorNetwork. Build the layout from the circuit "
                "and pass the TTN separately as TreeOptimizer(state=...)."
            )
        if supports is None:
            payloads, wheres, event_types = self._events_from_gates(gates)
            supports = wheres
        else:
            supports = list(supports)
            payloads = [None] * len(supports)
            event_types = ["support"] * len(supports)
        supports = [tuple(_normalize_layout_support(s)) for s in supports]
        self.payloads = tuple(payloads)
        self.event_types = tuple(event_types)
        inferred = -1
        for support in supports:
            for site in support:
                if isinstance(site, Integral):
                    inferred = max(inferred, site)
        if root_qubit is not None:
            try:
                root_qubit = int(root_qubit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "root_qubit must be an integer or None."
                ) from exc
            inferred = max(inferred, root_qubit)
        if n is None:
            n = inferred + 1
        try:
            n = int(n)
        except (TypeError, ValueError) as exc:
            raise ValueError("n must be a positive integer.") from exc
        if n <= 0:
            raise ValueError(
                "Could not infer qubit count; pass n explicitly."
            )
        if root_qubit is not None:
            if not 0 <= root_qubit < n:
                raise ValueError(
                    f"root_qubit {root_qubit!r} is outside 0..{n - 1}."
                )
        normalized_supports = []
        for support in supports:
            if len(set(support)) != len(support):
                raise ValueError(
                    f"layout support contains duplicate qubits: {support!r}."
                )
            for site in support:
                if not isinstance(site, Integral):
                    raise ValueError(
                        "tree layout supports must contain integer qubits; "
                        f"got {site!r}."
                    )
                if not 0 <= int(site) < n:
                    raise ValueError(
                        f"layout support qubit {site!r} is outside 0..{n - 1}."
                    )
            normalized_supports.append(tuple(int(site) for site in support))
        self.n = n
        self.root_qubit = root_qubit
        self.leaf_qubits = tuple(
            q for q in range(self.n) if q != self.root_qubit
        )
        self.lattice_shape = _normalize_lattice_shape(lattice_shape)
        if self.lattice_shape is not None:
            lattice_size = int(np.prod(self.lattice_shape))
            if lattice_size != self.n:
                dims = " * ".join(str(dim) for dim in self.lattice_shape)
                raise ValueError(
                    "lattice_shape product must equal n; got "
                    f"{dims} != {self.n}."
                )
        if lattice_site is not None and not callable(lattice_site):
            raise TypeError("lattice_site must be callable or None.")
        if lattice_site is not None and self.lattice_shape is None:
            raise ValueError(
                "lattice_site requires lattice_shape=(Lx, Ly) or "
                "(Lx, Ly, Lz)."
            )
        self.lattice_site = lattice_site
        self.coarse_grain = _normalize_coarse_grain(
            coarse_grain,
            ndim=2 if self.lattice_shape is None else len(self.lattice_shape),
        )
        self.max_arity, self.arity_candidates = _normalize_arity_candidates(
            max_arity
        )
        if top_arity is _DEFAULT_TOP_ARITY:
            top_arity = (
                3
                if (
                    root_qubit is None
                    and self.max_arity == 2
                    and len(self.leaf_qubits) >= 3
                )
                else None
            )
        if top_arity is not None:
            try:
                top_arity = int(top_arity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "top_arity must be an integer >= 2 or None."
                ) from exc
            if top_arity < 2:
                raise ValueError("top_arity must be >= 2 or None.")
            if top_arity > len(self.leaf_qubits):
                raise ValueError(
                    "top_arity cannot exceed the number of non-root qubits."
                )
            if root_qubit is not None and top_arity != 2:
                raise ValueError(
                    "top_arity > 2 cannot be combined with root_qubit: "
                    "the root would have a rank-four tensor."
                )
        self.top_arity = top_arity
        self.supports = tuple(normalized_supports)
        self.structure = structure
        self.chi = _validate_chi(chi)
        self.community_frac = float(community_frac)
        self.star_frac = float(star_frac)
        self.dense_max = int(dense_max)
        self.objective = _normalize_layout_objective(objective)
        self.hybrid_weights = _normalize_hybrid_weights(hybrid_weights)
        self.weight_mode = _normalize_weight_mode(weight_mode)
        self.order = _normalize_layout_order(order)
        self.map_mode = (
            self.order
            if isinstance(self.order, str) and self.order.startswith("coarse-")
            else None
        )
        if self.order == "quality":
            # ``order="quality"`` is the explicit high-quality contract. It
            # is intentionally stronger than merely enabling a leaf swap: the
            # all-scale full-tree objective and its topology search are part
            # of the mode. Callers can still opt out of individual stages in
            # ``run`` with ``refine=None`` or ``search=None``.
            self.objective = "full_tree"
        self.refine = _normalize_layout_refinement(refine)
        if refine_budget is not None:
            refine_budget = _validate_search_budget(refine_budget, "refine_budget")
        self.refine_budget = refine_budget
        self.topology_refine = _normalize_topology_refinement(topology_refine)
        if topology_budget is not None:
            topology_budget = _validate_search_budget(
                topology_budget, "topology_budget"
            )
        self.topology_budget = topology_budget
        self.search = _normalize_layout_search(search)
        self.search_budget = _validate_search_budget(search_budget, "search_budget")
        try:
            self.seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer.") from exc
        self.nevergrad_optimizer = str(nevergrad_optimizer)
        if not self.nevergrad_optimizer:
            raise ValueError("nevergrad_optimizer must be a non-empty string.")
        if max_operator_qubits is not None:
            try:
                max_operator_qubits = int(max_operator_qubits)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_operator_qubits must be a positive integer or None."
                ) from exc
            if max_operator_qubits < 1:
                raise ValueError(
                    "max_operator_qubits must be a positive integer or None."
                )
        self.max_operator_qubits = max_operator_qubits
        self.time_decay = _normalize_time_decay(time_decay)
        self.time_window = _normalize_time_window(time_window)

        # Layout search asks for the same structural quantities several times
        # (once per candidate arity and once per diagnostic).  Keep these
        # caches local to this immutable stream description; callers still get
        # fresh dictionaries from the public diagnostic methods below.
        self._plan_cache = {}
        self._edge_load_cache = {}
        self._rank_diagnostics_cache = {}
        self._full_tree_profile_cache = {}
        self._full_tree_structure_cache = {}
        self._schmidt_rank_cache = {}
        self._similarity_cache = {}
        self._congestion_weights_cache = None
        self._balanced_plan_cache = None

        sites = list(range(self.n))
        base_event_weights = tuple(
            _gate_stream_event_weights(
                self.payloads,
                self.supports,
                self.event_types,
                weight_mode=self.weight_mode,
            )
        )
        self.temporal_factors = _temporal_event_factors(
            len(base_event_weights),
            time_decay=self.time_decay,
            time_window=self.time_window,
        )
        self.event_weights = tuple(
            0.0 if str(event_type).lower() in {
                "measure", "reset", "measure_reset", "cap"
            } else weight * temporal_factor
            for weight, temporal_factor, event_type in zip(
                base_event_weights, self.temporal_factors, self.event_types
            )
        )
        self.pair_weights = _gate_stream_pair_weights(
            supports, sites, self.event_weights
        )

    @classmethod
    def lattice_order(cls, Lx, Ly, *args, mode=None, Lz=None, site=None,
                      grain=(2, 1)):
        """Return a logical-qubit order from a 2D or 3D layout mode.

        This is the reusable order-only counterpart to passing a named
        geometric preset to :meth:`run`:

        ``TreeLayoutFinder.lattice_order(16, 16, "folded-snake")`` or
        ``TreeLayoutFinder.lattice_order(4, 4, 3, "alternate-z")``.

        The 2D calling convention keeps the mode as the third positional
        argument. For 3D, either pass ``Lz`` as a keyword or use
        ``(Lx, Ly, Lz, mode)`` positionally.

        Parameters
        ----------
        Lx, Ly : int
            First two lattice dimensions.
        Lz : int, optional
            Third lattice dimension. Supplying this selects 3D traversal.
        mode : str
            Any supported 2D lattice mode, or the 3D modes ``"row-major"``,
            ``"col-major"``, ``"snake"``, ``"snake-row-major"``,
            ``"alternate-x"``, ``"alternate-y"``, and ``"alternate-z"``.
            The corresponding supported `coarse-*` modes are also available.
        grain : int or pair/triple of int, optional
            Fine sites per coarse block. Used only by `coarse-*` modes;
            defaults to `(2, 1)` in 2D and `(2, 1, 1)` in 3D.
        site : callable, optional
            Optional ``(x, y) -> qubit`` or ``(x, y, z) -> qubit`` label
            mapper. The default is x-major flattening.
        """
        if len(args) > 2:
            raise TypeError(
                "lattice_order accepts at most Lz and mode as extra "
                "positional arguments."
            )
        if len(args) == 1:
            value = args[0]
            if isinstance(value, (str, bytes)):
                if mode is not None:
                    raise TypeError("mode was supplied both positionally and by keyword.")
                mode = value
            else:
                if Lz is not None:
                    raise TypeError("Lz was supplied both positionally and by keyword.")
                Lz = value
        elif len(args) == 2:
            if Lz is not None or mode is not None:
                raise TypeError(
                    "Lz and mode must not be repeated in lattice_order()."
                )
            Lz, mode = args
        if mode is None:
            mode = "row-major"
        normalized = _normalize_layout_order(mode)
        if normalized == "quality" or normalized is None:
            raise ValueError(
                "lattice_order mode must be a geometric OneDMap preset."
            )
        shape = _normalize_lattice_shape(
            (Lx, Ly) if Lz is None else (Lx, Ly, Lz)
        )
        return _lattice_site_order(
            shape[0], shape[1], normalized,
            Lz=shape[2] if len(shape) == 3 else None,
            site=site, grain=grain,
        )

    def _preset_order(self, mode):
        """Resolve a named geometric preset against this finder's lattice."""
        if self.lattice_shape is None:
            if isinstance(mode, str) and mode.startswith("coarse-"):
                # A one-dimensional TreePlan has no transverse lattice axes
                # to remap. Keep its natural order while retaining the
                # explicit coarse-* metadata and the same binary tree API.
                return tuple(self.leaf_qubits)
            raise ValueError(
                f"order={mode!r} requires lattice_shape=(Lx, Ly) or "
                "(Lx, Ly, Lz) "
                "when constructing TreeLayoutFinder."
            )
        order = self.lattice_order(
            *self.lattice_shape,
            mode=mode,
            site=self.lattice_site,
            grain=self.coarse_grain,
        )
        if self.root_qubit is not None:
            order = tuple(q for q in order if q != self.root_qubit)
        return order

    @staticmethod
    def _events_from_gates(gates):
        """Return payloads, supports, and event types for layout analysis."""
        if gates is None:
            return [], [], []

        # ``_normalize_gate_entries`` intentionally accepts concrete bundled
        # sequences, not one-shot iterators.  A gate stream is commonly a
        # generator, and the optimizer may pass the same stream to both its
        # queue normalizer and this finder, so materialize it exactly once at
        # the boundary.
        if hasattr(gates, "__next__"):
            gates = list(gates)

        # Control events carry support but are not gate entries.  Strip them
        # before delegating ordinary entries to the MPS layout normalizer, so
        # a mixed stream can still build an interaction-aware tree.
        control = _mps_control_event_parts(gates)
        if control is not None:
            return [None], [control[2]], [control[0]]
        if isinstance(gates, (tuple, list)):
            if not any(
                _mps_control_event_parts(entry) is not None
                for entry in gates
            ):
                return _normalize_layout_gate_queue(gates)
            payloads = []
            supports = []
            event_types = []
            for entry in gates:
                control = _mps_control_event_parts(entry)
                if control is not None:
                    payloads.append(None)
                    supports.append(control[2])
                    event_types.append(control[0])
                    continue
                one_payload, one_where, one_type = _normalize_layout_gate_queue(
                    (entry,)
                )
                payloads.extend(one_payload)
                supports.extend(one_where)
                event_types.extend(one_type)
            return payloads, supports, event_types

        payloads, supports, event_types = _normalize_layout_gate_queue(gates)
        return payloads, supports, event_types

    @staticmethod
    def _supports_from_gates(gates):
        """Return only multi-site supports for compatibility with old callers."""
        _payloads, supports, _event_types = TreeLayoutFinder._events_from_gates(gates)
        return [
            tuple(_normalize_layout_support(support))
            for support in supports
            if len(_normalize_layout_support(support)) >= 2
        ]

    def _resolve_search_settings(
        self,
        *,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        topology_refine=_DEFAULT_SEARCH_OPTION,
        topology_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
    ):
        """Resolve method overrides against finder-owned search defaults."""
        if refine is _DEFAULT_SEARCH_OPTION:
            refine = self.refine
            if self.objective == "hypergraph" and refine is None:
                # A direct hypergraph score is only useful as a layout search
                # objective when the candidate is allowed to move. Keep the
                # old objectives fast, but make the explicitly requested
                # hypergraph mode perform its bounded local search by default.
                refine = "greedy"
        else:
            refine = _normalize_layout_refinement(refine)
        if refine_budget is _DEFAULT_SEARCH_OPTION:
            refine_budget = self.refine_budget
        elif refine_budget is not None:
            refine_budget = _validate_search_budget(
                refine_budget, "refine_budget"
            )
        if refine is not None and refine_budget is None:
            refine_budget = max(1, min(len(self.leaf_qubits) - 1, 64))

        if topology_refine is _DEFAULT_SEARCH_OPTION:
            topology_refine = self.topology_refine
            if self.objective == "hypergraph" and topology_refine is None:
                topology_refine = "nni"
            elif self.objective == "full_tree" and topology_refine is None:
                topology_refine = "subtree"
        else:
            topology_refine = _normalize_topology_refinement(topology_refine)
        if topology_budget is _DEFAULT_SEARCH_OPTION:
            topology_budget = self.topology_budget
        elif topology_budget is not None:
            topology_budget = _validate_search_budget(
                topology_budget, "topology_budget"
            )
        if topology_refine is not None and topology_budget is None:
            topology_budget = max(1, min(max(1, len(self.leaf_qubits) - 2), 64))

        if search is _DEFAULT_SEARCH_OPTION:
            search = self.search
            if self.objective == "full_tree" and search is None:
                search = "anneal"
        else:
            search = _normalize_layout_search(search)
        if search_budget is _DEFAULT_SEARCH_OPTION:
            search_budget = self.search_budget
        else:
            search_budget = _validate_search_budget(search_budget, "search_budget")
        if seed is _DEFAULT_SEARCH_OPTION:
            seed = self.seed
        else:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("seed must be an integer.") from exc
        if nevergrad_optimizer is _DEFAULT_SEARCH_OPTION:
            nevergrad_optimizer = self.nevergrad_optimizer
        else:
            nevergrad_optimizer = str(nevergrad_optimizer)
            if not nevergrad_optimizer:
                raise ValueError("nevergrad_optimizer must be a non-empty string.")
        return {
            "refine": refine,
            "refine_budget": refine_budget,
            "topology_refine": topology_refine,
            "topology_budget": topology_budget,
            "search": search,
            "search_budget": search_budget,
            "seed": seed,
            "nevergrad_optimizer": nevergrad_optimizer,
        }

    @staticmethod
    def _leaf_nodes(plan):
        """Return the deterministic leaf-position order of a plan."""
        return tuple(sorted(plan.qubit_of_leaf))

    def _leaf_order(self, plan):
        """Return the qubit label assigned to each deterministic leaf position."""
        return tuple(plan.qubit_of_leaf[leaf] for leaf in self._leaf_nodes(plan))

    def _plan_with_leaf_order(self, plan, order):
        """Return ``plan``'s immutable topology with a new leaf assignment."""
        order = tuple(int(q) for q in order)
        if set(order) != set(self.leaf_qubits) or len(order) != len(
            self.leaf_qubits
        ):
            raise ValueError(
                "leaf order must contain every non-root qubit exactly once."
            )
        qubit_of_leaf = dict(plan.qubit_of_leaf)
        for leaf, qubit in zip(self._leaf_nodes(plan), order):
            qubit_of_leaf[leaf] = qubit
        return TreePlan.from_children(
            plan.children,
            qubit_of_leaf,
            root=plan.root,
            root_qubit=plan.root_qubit,
            map_mode=plan.map_mode,
        )

    def _plan_with_leaf_swap(self, plan, left_leaf, right_leaf):
        """Swap two labels while retaining the tree topology exactly."""
        qubit_of_leaf = dict(plan.qubit_of_leaf)
        qubit_of_leaf[left_leaf], qubit_of_leaf[right_leaf] = (
            qubit_of_leaf[right_leaf],
            qubit_of_leaf[left_leaf],
        )
        return TreePlan.from_children(
            plan.children,
            qubit_of_leaf,
            root=plan.root,
            root_qubit=plan.root_qubit,
            map_mode=plan.map_mode,
        )

    def _plan_with_nni(self, plan, parent, child, variant):
        """Return one binary nearest-neighbor interchange of ``parent-child``.

        The rooted local pattern is ``parent -> (child, sibling)`` and
        ``child -> (a, b)``. Each NNI variant keeps ``child`` below ``parent``
        while moving either ``a`` or ``b`` across the internal edge. Node ids
        and leaf labels are retained so this is a topology move, not a hidden
        relabeling.
        """
        parent_children = tuple(plan.children[parent])
        child_children = tuple(plan.children[child])
        if (
            len(parent_children) != 2
            or len(child_children) != 2
            or child not in parent_children
        ):
            raise ValueError("NNI requires a binary internal parent-child edge.")
        sibling = next(node for node in parent_children if node != child)
        if variant not in (0, 1):
            raise ValueError("NNI variant must be 0 or 1.")
        a, b = child_children
        moved = b if variant == 0 else a
        retained = a if variant == 0 else b
        children = {
            node: tuple(child_ids) for node, child_ids in plan.children.items()
        }
        children[parent] = (child, moved)
        children[child] = (retained, sibling)
        return TreePlan.from_children(
            children,
            plan.qubit_of_leaf,
            root=plan.root,
            root_qubit=plan.root_qubit,
            map_mode=plan.map_mode,
        )

    @staticmethod
    def _nni_edges(plan):
        """Return deterministic binary internal edges eligible for NNI."""
        return tuple(
            (parent, child)
            for parent, children in sorted(plan.children.items())
            if len(children) == 2
            for child in children
            if len(plan.children.get(child, ())) == 2
        )

    @staticmethod
    def _subtree_nodes(plan, root):
        """Return all node ids in the rooted subtree at ``root``."""
        nodes = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            stack.extend(plan.children[node])
        return nodes

    def _plan_with_subtree_reconfiguration(self, plan, subtree_root, rng):
        """Rebuild one descendant subtree while preserving its attachment."""
        old_nodes = self._subtree_nodes(plan, subtree_root)
        subtree_qubits = sorted(
            plan.qubit_of_leaf[node]
            for node in old_nodes
            if node in plan.qubit_of_leaf
        )
        if len(subtree_qubits) < 4:
            return None

        global_weights = self._similarity_weights(
            self._congestion_pair_weights()
            if self.objective == "full_tree" else None
        )
        local_index = {q: i for i, q in enumerate(subtree_qubits)}
        local_weights = {
            (local_index[qa], local_index[qb]): weight
            for (qa, qb), weight in global_weights.items()
            if qa in local_index and qb in local_index
        }
        local_order = list(range(len(subtree_qubits)))
        rng.shuffle(local_order)
        local_structure = (
            "adaptive" if self.structure == "adaptive" else "balanced"
        )
        local_root_qubit = (
            len(subtree_qubits)
            if subtree_root == plan.root and plan.root_qubit is not None
            else None
        )
        lower_arities = [
            len(children)
            for node, children in plan.children.items()
            if node != plan.root and children
        ]
        local_max_arity = max(lower_arities, default=2)
        local_top_arity = (
            plan.top_arity if subtree_root == plan.root else None
        )
        local_plan = TreePlan.from_order(
            local_order,
            weights=local_weights,
            structure=local_structure,
            max_arity=local_max_arity,
            community_frac=self.community_frac,
            star_frac=self.star_frac,
            dense_max=self.dense_max,
            root_qubit=local_root_qubit,
            top_arity=local_top_arity,
        )

        next_node = max(plan.children, default=-1) + 1
        local_to_global = {local_plan.root: subtree_root}
        for local_node in local_plan.children:
            if local_node != local_plan.root:
                local_to_global[local_node] = next_node
                next_node += 1

        children = {
            node: tuple(child_ids)
            for node, child_ids in plan.children.items()
            if node not in old_nodes
        }
        qubit_of_leaf = {
            node: qubit
            for node, qubit in plan.qubit_of_leaf.items()
            if node not in old_nodes
        }
        for local_node, local_children in local_plan.children.items():
            global_node = local_to_global[local_node]
            children[global_node] = tuple(
                local_to_global[child] for child in local_children
            )
        for local_node, local_qubit in local_plan.qubit_of_leaf.items():
            qubit_of_leaf[local_to_global[local_node]] = (
                subtree_qubits[local_qubit]
            )
        return TreePlan.from_children(
            children,
            qubit_of_leaf,
            root=plan.root,
            root_qubit=plan.root_qubit,
            map_mode=plan.map_mode,
        )

    def _path_score_and_max(self, plan):
        """Return the weighted interaction path sum and longest active path."""
        score = 0.0
        max_path = 0
        for (qa, qb), weight in self.pair_weights.items():
            distance = plan.tree_distance(qa, qb)
            score += float(weight) * distance
            max_path = max(max_path, distance)
        return float(score), int(max_path)

    def _path_score_after_leaf_swap(self, plan, left_leaf, right_leaf, score):
        """Return the exact path-score update for a two-label leaf swap."""
        qa = plan.qubit_of_leaf[left_leaf]
        qb = plan.qubit_of_leaf[right_leaf]
        change = 0.0
        for q in range(self.n):
            if q == qa or q == qb:
                continue
            weight_a = self.pair_weights.get(tuple(sorted((qa, q))), 0.0)
            weight_b = self.pair_weights.get(tuple(sorted((qb, q))), 0.0)
            if weight_a:
                change += float(weight_a) * (
                    plan.tree_distance(qb, q) - plan.tree_distance(qa, q)
                )
            if weight_b:
                change += float(weight_b) * (
                    plan.tree_distance(qa, q) - plan.tree_distance(qb, q)
                )
        return float(score + change)

    @staticmethod
    def _normalized_cost(value, reference):
        """Normalize a non-negative layout metric against a fixed baseline."""
        if reference > 0.0:
            return float(value / reference)
        return 0.0 if value == 0.0 else float(value)

    def _hybrid_key(self, plan):
        """Return normalized distance and rank-load cost for hybrid selection."""
        score, max_path = self._path_score_and_max(plan)
        loads = self.edge_loads(plan)
        max_load = max(loads.values(), default=0.0)
        total_load = sum(loads.values())

        balanced = self._balanced_plan()
        balanced_score, _ = self._path_score_and_max(balanced)
        balanced_loads = self.edge_loads(balanced)
        balanced_max_load = max(balanced_loads.values(), default=0.0)
        balanced_total_load = sum(balanced_loads.values())
        path_weight, max_load_weight, total_load_weight = self.hybrid_weights
        hybrid = (
            path_weight * self._normalized_cost(score, balanced_score)
            + max_load_weight * self._normalized_cost(max_load, balanced_max_load)
            + total_load_weight * self._normalized_cost(
                total_load, balanced_total_load
            )
        )
        return (
            float(hybrid),
            float(max_load),
            float(total_load),
            float(score),
            int(max_path),
        )

    def _tensor_cost_key(self, plan):
        """Return a chi-scaled proxy for local TTN tensor cost.

        A wider node reduces geodesic distance but increases the number of
        virtual legs on one tensor.  The exact contraction cost depends on
        the realized bond dimensions, so this uses the configured ``chi`` (or
        a conservative qubit bond of two) to rank structures without ever
        allocating tensors.
        """
        chi = max(2, int(self.chi or 2))
        log_chi = float(np.log2(chi))
        degrees = []
        log_sizes = []
        for node, children in plan.children.items():
            if not children:
                continue
            virtual_degree = plan.virtual_degree(node)
            physical_legs = 1 if node in plan.qubit_of_node else 0
            degrees.append(virtual_degree)
            log_sizes.append(virtual_degree * log_chi + physical_legs)
        if not degrees:
            return (0.0, 0.0, 0, 0)
        max_log_size = max(log_sizes)
        # log2(sum(2**log_size)) without overflowing for large chi/arity.
        shifted = np.asarray(log_sizes, dtype=float) - max_log_size
        total_log_size = max_log_size + float(np.log2(np.exp2(shifted).sum()))
        return (
            float(max_log_size),
            float(total_log_size),
            int(max(degrees)),
            int(sum(degrees)),
        )

    def _objective_key(self, plan):
        """Return the selected objective's deterministic comparison key."""
        if self.objective == "path":
            return self._path_score_and_max(plan)
        if self.objective == "congestion":
            return self._congestion_key(plan)
        if self.objective == "full_tree":
            profile = self.full_tree_profile(plan)
            return (
                # For a finite-chi optimizer, predicted cut overflow is the
                # first-order performance and accuracy failure. Prefer a
                # layout that stays within the cap, then distinguish the
                # remaining candidates by uncapped edge demand before
                # considering tensor work. With chi=None, overflow is zero
                # and this naturally reduces to uncapped demand ordering.
                profile["peak_overflow_log2"],
                profile["total_overflow_log2"],
                profile["peak_edge_demand_log2"],
                profile["total_edge_demand_log2"],
                profile["peak_tensor_log2"],
                profile["peak_work_log2"],
                profile["log_total_write"],
                profile["log_total_work"],
                profile["total_route_length"],
                self.score(plan),
            )
        if self.objective in {"compression", "hypergraph"}:
            loads = self.edge_loads(plan)
            values = tuple(loads.values())
            tensor_cost = self._tensor_cost_key(plan)
            return (
                max(values, default=0.0),
                sum(values),
                tensor_cost[0],
                tensor_cost[1],
                self.score(plan),
                max(
                    (plan.tree_distance(a, b) for a in range(self.n)
                     for b in range(a + 1, self.n)),
                    default=0,
                ),
            )
        return self._hybrid_key(plan)

    def _selection_key(self, plan, chi):
        """Return the objective key with the optional chi feasibility prefix."""
        key = self._objective_key(plan)
        if chi is not None:
            return (_chi_cut_fields(plan, chi)["chi_overflow"],) + key
        return key

    def _selection_loss(self, plan, chi):
        """Return a scalar surrogate for derivative-free layout search."""
        key = self._objective_key(plan)
        if self.objective == "path":
            value = key[0]
        elif self.objective == "full_tree":
            value = (
                key[0]
                + 0.50 * key[1]
                + 0.10 * key[2]
                + 0.10 * key[3]
                + 0.01 * key[4]
                + 0.001 * key[5]
                + 1.0e-6 * key[6]
            )
        elif self.objective in {"congestion", "compression", "hypergraph"}:
            value = key[0] + 1.0e-6 * key[1] + 1.0e-12 * key[2]
        else:
            value = key[0]
        if chi is not None:
            value += 1.0e6 * _chi_cut_fields(plan, chi)["chi_overflow"]
        return float(value)

    def _discard_plan_cache(self, plan):
        """Release diagnostics retained only for a rejected temporary plan."""
        cached = self._edge_load_cache.get(id(plan))
        if cached is not None and cached[0] is plan:
            del self._edge_load_cache[id(plan)]
        cached = self._rank_diagnostics_cache.get(id(plan))
        if cached is not None and cached[0] is plan:
            del self._rank_diagnostics_cache[id(plan)]
        cached = self._full_tree_profile_cache.get(id(plan))
        if cached is not None and cached[0] is plan:
            del self._full_tree_profile_cache[id(plan)]

    def _refine_plan_greedy(self, plan, *, chi, budget, progbar=False):
        """Greedily improve a fixed topology through adjacent leaf swaps."""
        initial_key = self._selection_key(plan, chi)
        leaf_nodes = self._leaf_nodes(plan)
        if len(leaf_nodes) < 2 or budget < 1:
            return plan, {
                "method": "greedy",
                "evaluations": 0,
                "accepted_moves": 0,
                "initial_key": initial_key,
                "final_key": initial_key,
            }

        current = plan
        current_key = initial_key
        current_path_score = self.score(current)
        evaluations = 0
        accepted_moves = 0
        position = 0
        progress = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress = tqdm(
                total=budget,
                desc="tree layout greedy",
                leave=False,
            )
        while position < len(leaf_nodes) - 1 and evaluations < budget:
            left_leaf = leaf_nodes[position]
            right_leaf = leaf_nodes[position + 1]
            evaluations += 1
            if progress is not None:
                progress.update()
            if self.objective == "path":
                candidate_path_score = self._path_score_after_leaf_swap(
                    current, left_leaf, right_leaf, current_path_score
                )
                if candidate_path_score >= current_path_score - 1.0e-12:
                    position += 1
                    continue
                candidate = self._plan_with_leaf_swap(
                    current, left_leaf, right_leaf
                )
                candidate_key = self._selection_key(candidate, chi)
            else:
                candidate = self._plan_with_leaf_swap(
                    current, left_leaf, right_leaf
                )
                candidate_key = self._selection_key(candidate, chi)
                candidate_path_score = None

            if candidate_key < current_key:
                self._discard_plan_cache(current)
                current = candidate
                current_key = candidate_key
                if candidate_path_score is None:
                    current_path_score = self.score(current)
                else:
                    current_path_score = candidate_path_score
                accepted_moves += 1
                position = max(0, position - 1)
            else:
                self._discard_plan_cache(candidate)
                position += 1
        if progress is not None:
            progress.close()
        return current, {
            "method": "greedy",
            "evaluations": evaluations,
            "accepted_moves": accepted_moves,
            "initial_key": initial_key,
            "final_key": current_key,
        }

    def _refine_plan_topology(self, plan, *, chi, budget, progbar=False):
        """Greedily improve binary topology through bounded NNI moves."""
        initial_key = self._selection_key(plan, chi)
        if budget < 1 or not plan.is_binary():
            return plan, {
                "method": "nni",
                "evaluations": 0,
                "accepted_moves": 0,
                "initial_key": initial_key,
                "final_key": initial_key,
            }

        current = plan
        current_key = initial_key
        evaluations = 0
        accepted_moves = 0
        progress = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress = tqdm(
                total=budget,
                desc="tree layout topology",
                leave=False,
            )

        while evaluations < budget:
            best = current
            best_key = current_key
            for parent, child in self._nni_edges(current):
                for variant in (0, 1):
                    if evaluations >= budget:
                        break
                    evaluations += 1
                    if progress is not None:
                        progress.update()
                    candidate = self._plan_with_nni(
                        current, parent, child, variant
                    )
                    candidate_key = self._selection_key(candidate, chi)
                    if candidate_key < best_key:
                        if best is not current:
                            self._discard_plan_cache(best)
                        best = candidate
                        best_key = candidate_key
                    else:
                        self._discard_plan_cache(candidate)
                if evaluations >= budget:
                    break
            if best is current:
                break
            current = best
            current_key = best_key
            accepted_moves += 1

        if progress is not None:
            progress.close()
        return current, {
            "method": "nni",
            "evaluations": evaluations,
            "accepted_moves": accepted_moves,
            "initial_key": initial_key,
            "final_key": current_key,
        }

    def _refine_plan_subtree(
        self, plan, *, chi, budget, seed, progbar=False
    ):
        """Greedily accept subtree replacements across all tree scales."""
        initial_key = self._selection_key(plan, chi)
        if budget < 1:
            return plan, {
                "method": "subtree",
                "search": "greedy",
                "evaluations": 0,
                "accepted_moves": 0,
                "initial_key": initial_key,
                "final_key": initial_key,
                "scales_visited": (),
            }
        rng = np.random.default_rng(seed)
        current = plan
        current_key = initial_key
        evaluations = 0
        accepted_moves = 0
        visited_scales = set()
        progress = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress = tqdm(
                total=budget,
                desc="tree layout subtree",
                leave=False,
            )
        while evaluations < budget:
            current_candidates = []
            for node, children in current.children.items():
                if not children:
                    continue
                subtree_size = sum(
                    1
                    for descendant in self._subtree_nodes(current, node)
                    if descendant in current.qubit_of_leaf
                )
                if subtree_size >= 4:
                    current_candidates.append((node, subtree_size))
            if not current_candidates:
                break
            evaluations += 1
            if progress is not None:
                progress.update()
            nodes = np.asarray(
                [node for node, _size in current_candidates], dtype=int
            )
            weights = np.asarray(
                [np.log2(size) for _node, size in current_candidates]
            )
            weights /= weights.sum()
            node = int(rng.choice(nodes, p=weights))
            visited_scales.add(_tree_node_scales(current)[node])
            candidate = self._plan_with_subtree_reconfiguration(
                current, node, rng
            )
            if candidate is None:
                continue
            candidate_key = self._selection_key(candidate, chi)
            if candidate_key < current_key:
                self._discard_plan_cache(current)
                current = candidate
                current_key = candidate_key
                accepted_moves += 1
            else:
                self._discard_plan_cache(candidate)
        if progress is not None:
            progress.close()
        return current, {
            "method": "subtree",
            "search": "greedy",
            "evaluations": evaluations,
            "accepted_moves": accepted_moves,
            "initial_key": initial_key,
            "final_key": current_key,
            "scales_visited": tuple(sorted(visited_scales)),
        }

    def _anneal_plan_subtree(
        self, plan, *, chi, budget, seed, progbar=False
    ):
        """Anneal subtree replacements across all available tree scales."""
        initial_key = self._selection_key(plan, chi)
        candidates = []
        for node, children in plan.children.items():
            if not children:
                continue
            subtree_size = sum(
                1
                for descendant in self._subtree_nodes(plan, node)
                if descendant in plan.qubit_of_leaf
            )
            if subtree_size >= 4:
                candidates.append((node, subtree_size))
        if budget < 1 or not candidates:
            return plan, {
                "method": "subtree",
                "search": "anneal",
                "evaluations": 0,
                "accepted_moves": 0,
                "initial_key": initial_key,
                "final_key": initial_key,
                "scales_visited": (),
            }

        rng = np.random.default_rng(seed)
        current = plan
        current_key = initial_key
        current_loss = self._selection_loss(current, chi)
        best = current
        best_key = current_key
        evaluations = 0
        accepted_moves = 0
        visited_scales = set()
        initial_temperature = max(1.0, abs(current_loss) * 0.05)
        progress = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress = tqdm(
                total=budget,
                desc="tree layout subtree anneal",
                leave=False,
            )

        while evaluations < budget:
            evaluations += 1
            if progress is not None:
                progress.update()
            current_candidates = []
            for current_node, current_children in current.children.items():
                if not current_children:
                    continue
                current_size = sum(
                    1
                    for descendant in self._subtree_nodes(current, current_node)
                    if descendant in current.qubit_of_leaf
                )
                if current_size >= 4:
                    current_candidates.append((current_node, current_size))
            if not current_candidates:
                break
            nodes = np.asarray(
                [node for node, _size in current_candidates], dtype=int
            )
            weights = np.asarray(
                [np.log2(size) for _node, size in current_candidates]
            )
            weights /= weights.sum()
            node = int(rng.choice(nodes, p=weights))
            visited_scales.add(_tree_node_scales(current)[node])
            candidate = self._plan_with_subtree_reconfiguration(
                current, node, rng
            )
            if candidate is None:
                continue
            candidate_loss = self._selection_loss(candidate, chi)
            delta = candidate_loss - current_loss
            fraction = evaluations / max(1, budget)
            temperature = initial_temperature * max(1.0e-3, 1.0 - fraction)
            accept = delta <= 0.0 or rng.random() < np.exp(
                -min(700.0, delta / temperature)
            )
            if accept:
                current = candidate
                current_loss = candidate_loss
                current_key = self._selection_key(current, chi)
                accepted_moves += 1
                if current_key < best_key:
                    self._discard_plan_cache(best)
                    best = current
                    best_key = current_key
            else:
                self._discard_plan_cache(candidate)

        if progress is not None:
            progress.close()
        if best is not current:
            self._discard_plan_cache(current)
        return best, {
            "method": "subtree",
            "search": "anneal",
            "evaluations": evaluations,
            "accepted_moves": accepted_moves,
            "initial_key": initial_key,
            "final_key": best_key,
            "scales_visited": tuple(sorted(visited_scales)),
        }

    def _refine_plan_hybrid(
        self, plan, *, chi, budget, seed, optimizer_name, progbar=False
    ):
        """Combine all-scale topology annealing with Nevergrad leaf search.

        The two search methods explore complementary spaces: annealing changes
        descendant subtree topology while Nevergrad changes the labels assigned
        to a fixed topology. ``budget`` is the total number of proposals and
        is split between the two stages, so hybrid quality mode remains bounded
        by the caller's requested offline budget.
        """
        initial_key = self._selection_key(plan, chi)
        if budget < 2:
            anneal_budget = budget
            nevergrad_budget = 0
        else:
            anneal_budget = max(1, budget // 2)
            nevergrad_budget = budget - anneal_budget

        current, anneal_info = self._anneal_plan_subtree(
            plan,
            chi=chi,
            budget=anneal_budget,
            seed=seed + 1,
            progbar=progbar,
        )
        nevergrad_info = None
        if nevergrad_budget:
            try:
                current, nevergrad_info = self._refine_plan_nevergrad(
                    current,
                    chi=chi,
                    budget=nevergrad_budget,
                    seed=seed + 2,
                    optimizer_name=optimizer_name,
                    progbar=progbar,
                )
            except ImportError:
                # Hybrid mode is the automatic quality path. Keep its
                # dependency-free topology result when the optional package
                # is absent; explicit search="nevergrad" still raises.
                nevergrad_info = {
                    "method": "nevergrad",
                    "optimizer": optimizer_name,
                    "budget": nevergrad_budget,
                    "evaluations": 0,
                    "seed": seed + 2,
                    "available": False,
                    "improved": False,
                }

        return current, {
            "method": "hybrid",
            "search": "hybrid",
            "budget": budget,
            "evaluations": int(anneal_info["evaluations"])
            + int((nevergrad_info or {}).get("evaluations", 0)),
            "accepted_moves": int(anneal_info["accepted_moves"])
            + int((nevergrad_info or {}).get("improved", False)),
            "initial_key": initial_key,
            "final_key": self._selection_key(current, chi),
            "anneal": anneal_info,
            "nevergrad": nevergrad_info,
        }

    def _refine_plan_nevergrad(
        self, plan, *, chi, budget, seed, optimizer_name, progbar=False
    ):
        """Use Nevergrad to refine a leaf assignment before simulation starts."""
        try:
            import nevergrad as ng
        except ImportError as exc:
            raise ImportError(
                "Nevergrad tree-layout search requires the optional dependency. "
                "Install it with `pip install pepsy[layout]`."
            ) from exc

        try:
            optimizer_class = ng.optimizers.registry[optimizer_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Nevergrad optimizer {optimizer_name!r}."
            ) from exc

        initial_plan = plan
        initial_key = self._selection_key(initial_plan, chi)
        initial_order = self._leaf_order(initial_plan)
        if len(initial_order) < 2 or budget < 1:
            return initial_plan, {
                "method": "nevergrad",
                "optimizer": optimizer_name,
                "budget": budget,
                "evaluations": 0,
                "seed": seed,
                "initial_key": initial_key,
                "final_key": initial_key,
                "improved": False,
            }
        leaf_qubits = tuple(initial_order)
        priorities = np.arange(len(leaf_qubits), dtype=float)
        parametrization = ng.p.Array(init=priorities)
        if hasattr(parametrization, "set_bounds"):
            parametrization.set_bounds(
                -float(len(leaf_qubits)),
                float(2 * len(leaf_qubits)),
            )
        optimizer = optimizer_class(parametrization=parametrization, budget=budget)
        random_state = getattr(optimizer.parametrization, "random_state", None)
        if random_state is not None:
            random_state.seed(seed)

        losses = {}
        progress = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress = tqdm(
                total=budget,
                desc="tree layout nevergrad",
                leave=False,
            )

        def loss(values):
            if progress is not None:
                progress.update()
            values = np.asarray(values)
            order = tuple(
                leaf_qubits[int(position)]
                for position in np.argsort(values, kind="stable")
            )
            cached = losses.get(order)
            if cached is not None:
                return cached
            candidate = self._plan_with_leaf_order(initial_plan, order)
            value = self._selection_loss(candidate, chi)
            losses[order] = value
            self._discard_plan_cache(candidate)
            return value

        try:
            recommendation = optimizer.minimize(loss)
        finally:
            if progress is not None:
                progress.close()
        final_order = tuple(
            leaf_qubits[int(position)]
            for position in np.argsort(
                np.asarray(recommendation.value), kind="stable"
            )
        )
        candidate = self._plan_with_leaf_order(initial_plan, final_order)
        candidate_key = self._selection_key(candidate, chi)
        if candidate_key < initial_key:
            final_plan = candidate
            improved = True
        else:
            self._discard_plan_cache(candidate)
            final_plan = initial_plan
            improved = False
        return final_plan, {
            "method": "nevergrad",
            "optimizer": optimizer_name,
            "budget": budget,
            "evaluations": len(losses),
            "seed": seed,
            "initial_key": initial_key,
            "final_key": self._selection_key(final_plan, chi),
            "improved": improved,
        }

    def _improve_plan(self, plan, *, chi, settings, progbar=False):
        """Run the requested pre-simulation plan refinements in sequence."""
        initial_order = self._leaf_order(plan)
        initial_key = self._selection_key(plan, chi)
        info = {
            "initial_order": initial_order,
            "initial_key": initial_key,
            "topology_refinement": None,
            "refinement": None,
            "search": None,
        }
        if settings["topology_refine"] == "nni":
            plan, info["topology_refinement"] = self._refine_plan_topology(
                plan,
                chi=chi,
                budget=settings["topology_budget"],
                progbar=progbar,
            )
        elif settings["topology_refine"] == "subtree":
            plan, info["topology_refinement"] = self._refine_plan_subtree(
                plan,
                chi=chi,
                budget=settings["topology_budget"],
                seed=settings["seed"],
                progbar=progbar,
            )
        if settings["refine"] == "greedy":
            plan, info["refinement"] = self._refine_plan_greedy(
                plan,
                chi=chi,
                budget=settings["refine_budget"],
                progbar=progbar,
            )
        if settings["search"] == "anneal":
            plan, info["search"] = self._anneal_plan_subtree(
                plan,
                chi=chi,
                budget=settings["search_budget"],
                seed=settings["seed"] + 1,
                progbar=progbar,
            )
        elif settings["search"] == "nevergrad":
            plan, info["search"] = self._refine_plan_nevergrad(
                plan,
                chi=chi,
                budget=settings["search_budget"],
                seed=settings["seed"],
                optimizer_name=settings["nevergrad_optimizer"],
                progbar=progbar,
            )
        elif settings["search"] == "hybrid":
            plan, info["search"] = self._refine_plan_hybrid(
                plan,
                chi=chi,
                budget=settings["search_budget"],
                seed=settings["seed"],
                optimizer_name=settings["nevergrad_optimizer"],
                progbar=progbar,
            )
        info["final_order"] = self._leaf_order(plan)
        info["final_key"] = self._selection_key(plan, chi)
        return plan, info

    def _build_plan(self, weights, *, structure=None,
                    max_arity=_DEFAULT_MAX_ARITY):
        """Build one deterministic candidate tree from pair weights."""
        if max_arity is _DEFAULT_MAX_ARITY:
            max_arity = self.max_arity
        structure = self.structure if structure is None else structure
        key = (structure, max_arity, id(weights))
        # ``weights`` is normally one of the finder-owned cached mappings.  The
        # identity component avoids returning a plan built from a different
        # caller-supplied weighting mapping with the same arity.
        cached = self._plan_cache.get(key)
        if cached is not None and cached[0] is weights:
            return cached[1]
        plan = TreePlan.from_order(
            self.leaf_qubits,
            weights=weights,
            structure=structure,
            max_arity=max_arity,
            community_frac=self.community_frac,
            star_frac=self.star_frac,
            dense_max=self.dense_max,
            root_qubit=self.root_qubit,
            top_arity=self.top_arity,
            map_mode=(
                self.order
                if isinstance(self.order, str) and self.order.startswith("coarse-")
                else None
            ),
        )
        self._plan_cache[key] = (weights, plan)
        return plan

    def _schmidt_rank(self, payload, support, left_support):
        """Return a cached numeric operator-Schmidt rank or bound."""
        return self._schmidt_rank_info(payload, support, left_support)["rank"]

    def _schmidt_rank_info(self, payload, support, left_support):
        """Return rank metadata used by compression diagnostics.

        ``exact=False`` is deliberate for opaque native arrays, MPO bond
        bounds, and supports larger than ``max_operator_qubits``.  The numeric
        rank is then a conservative operator-space bound, never an optimistic
        hard-coded rank-two fallback.
        """
        if (
            self.max_operator_qubits is not None
            and len(support) > self.max_operator_qubits
        ):
            return {
                "rank": _operator_schmidt_rank_bound(support, left_support),
                "exact": False,
                "reason": "max_operator_qubits",
            }
        # For an ordinary dense gate, its Schmidt rank depends on the operator
        # data and *wire positions* in ``support``, not on the global qubit
        # labels. Reusing the same CNOT/CZ/parameterized matrix across many
        # pairs should therefore reuse the small SVD. A structured MPO carries
        # explicit site labels, so retain its label-sensitive cache key.
        support = tuple(support)
        left_set = frozenset(left_support)
        is_structured_mpo = callable(getattr(payload, "gen_sites_present", None))
        if is_structured_mpo:
            partition_key = left_set
            support_key = support
        else:
            positions = {site: pos for pos, site in enumerate(support)}
            partition_key = tuple(
                positions[site] for site in support if site in left_set
            )
            support_key = len(support)
        key = (id(payload), support_key, partition_key)
        cached = self._schmidt_rank_cache.get(key)
        if cached is not None and cached[0] is payload:
            return cached[1]
        rank = _submpo_schmidt_rank_bound(payload, support, left_support)
        if rank is not None:
            info = {
                "rank": int(rank),
                "exact": False,
                "reason": "mpo_bond_bound",
            }
        else:
            info = _mps_operator_schmidt_rank_info(
                payload,
                support,
                left_support,
                max_operator_qubits=self.max_operator_qubits,
            )
        self._schmidt_rank_cache[key] = (payload, info)
        return info

    def _candidate_plans(self, max_arity):
        """Build deterministic seed plans for the selected objective.

        The ``hypergraph`` objective deliberately keeps several inexpensive
        pairwise-derived seeds, but all final ranking and its default local
        refinement use :meth:`edge_loads`, which scans each original
        multi-site support across the candidate tree. The pairwise seeds are
        therefore only an initialization strategy, not the objective itself.
        """
        interaction_plan = self._build_plan(self._similarity_weights())
        if max_arity != self.max_arity:
            interaction_plan = self._build_plan(
                self._similarity_weights(), max_arity=max_arity
            )
        if self.objective == "path":
            return {"interaction": interaction_plan}

        congestion_plan = self._build_plan(
            self._similarity_weights(self._congestion_pair_weights()),
            max_arity=max_arity,
        )
        balanced_plan = self._build_plan(
            self._similarity_weights(), structure="balanced",
            max_arity=max_arity,
        )
        return {
            "interaction": interaction_plan,
            "congestion": congestion_plan,
            "balanced": balanced_plan,
        }

    def _select_plan(self, max_arity):
        """Select one plan without changing the finder's stored diagnostics."""
        candidates = self._candidate_plans(max_arity)
        selected = min(
            candidates,
            key=lambda name: self._objective_key(candidates[name]),
        )
        return candidates[selected]

    def qubit_order(self):
        """Return a spectral qubit ordering adapted to the gate-stream interactions.

        The order is the global Fiedler spectral reordering of the leaf qubits
        under the similarity weights used internally by the layout finder.
        A configured ``root_qubit`` is fixed at the root and omitted from this
        returned leaf order. Strongly coupled leaf qubits end up consecutive,
        which is the ideal input for :meth:`TreePlan.build_layered` so that
        blocks group entangled qubits together.

        Returns
        -------
        list of int
            Every non-root qubit exactly once (all ``0..n-1`` qubits when
            ``root_qubit`` is ``None``).
        """
        weights = self._similarity_weights()
        order = _gate_stream_spectral_order(
            list(self.leaf_qubits), weights, dense_max=self.dense_max
        )
        return order if order else list(self.leaf_qubits)

    def layered(self, block_size=4, *, order=None):
        """Build a fixed layered tree for a chosen ``block_size`` (no search).

        This is the direct, single-``block_size`` counterpart to
        :meth:`recommend_layered`.  It orders the qubits with the spectral
        :meth:`qubit_order` (so strongly coupled qubits share a block) and
        returns the :class:`TreePlan` from :meth:`TreePlan.build_layered`
        straight away -- no candidate sweep, no wrapper dict.

        ``block_size`` is a cost/accuracy knob, not something to maximize
        blindly.  A blocking node fuses ``block_size`` physical qubits into one
        tensor, so every intra-block correlation is represented *exactly*, but
        that tensor has ``2 ** block_size`` physical dimension.  Larger blocks
        are therefore more accurate at fixed ``chi`` yet exponentially more
        expensive: pick the largest block that fits your memory budget rather
        than searching, since a congestion search cannot see the block tensor's
        exponential cost and simply trends toward the widest block.

        Parameters
        ----------
        block_size : int
            Number of physical qubits per leaf-parent (blocking) node.
        order : sequence of int, optional
            Qubit order fed to :meth:`TreePlan.build_layered`.  Defaults to the
            spectral :meth:`qubit_order`.

        Returns
        -------
        TreePlan
            The layered plan (blocking layer, binary middle, ternary top).
        """
        if order is None:
            order = self.qubit_order()
        else:
            order = [int(q) for q in order]
        return TreePlan.build_layered(
            order,
            block_size=block_size,
            root_qubit=self.root_qubit,
        )

    def recommend_layered(
        self,
        block_sizes=(2, 3, 4),
        *,
        order=None,
        chi=_DEFAULT_CHI,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        topology_refine=_DEFAULT_SEARCH_OPTION,
        topology_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
        progbar=False,
    ):
        """Optimize the fixed layered structure over ``block_size``.

        The structure family is fixed by :meth:`TreePlan.build_layered`
        (a ``block_size`` blocking layer, binary middle layers, and a ternary
        top tensor); only the blocking width is free.  This builds one layered
        plan per candidate ``block_size`` on the entanglement-adapted qubit
        order and returns the plan that minimizes the selected layout
        objective, mirroring :meth:`recommend_arities`.

        Parameters
        ----------
        block_sizes : iterable of int
            Candidate blocking widths (physical qubits per leaf-parent node).
        order : sequence of int, optional
            Qubit order fed to :meth:`TreePlan.build_layered`.  Defaults to the
            spectral :meth:`qubit_order` so strongly coupled qubits share a
            block.
        chi : int, optional
            Bond-dimension budget.  The path/congestion objectives are
            ``chi``-blind cost proxies that can favour a wider block whose
            widest bond overflows ``chi`` (see :meth:`TreePlan.max_bond_cut`).
            When ``chi`` is given the recommendation is made ``chi``-aware:
            candidates are ranked first by ``chi_overflow`` (how far the widest
            bond exceeds ``log2(chi)``), so a structure that is *exact* at
            ``chi`` is preferred, and the layout objective only breaks ties
            among equally-overflowing candidates.  Each candidate additionally
            reports ``max_bond_cut``, ``chi_overflow``, and ``exact_at_chi``.
            When omitted, uses the ``chi`` supplied to the finder; pass
            ``chi=None`` explicitly for a chi-blind comparison.
        refine : {None, "greedy"}, optional
            Override the finder refinement setting. `"greedy"` performs a
            bounded adjacent leaf-swap search on each candidate tree.
        topology_refine : {None, "nni", "subtree"}, optional
            Override the optional topology refinement. ``"nni"`` performs
            bounded nearest-neighbor interchange proposals and is a no-op for
            the non-binary layered structure; ``"subtree"`` rebuilds selected
            descendant subtrees and is useful for all-scale objectives.
        refine_budget : int, optional
            Maximum greedy proposals per candidate. When omitted, an enabled
            greedy search uses at most ``min(n - 1, 64)`` proposals.
        topology_budget : int, optional
            Maximum topology proposals per candidate. When omitted, an enabled
            search uses at most 64 proposals.
        search : {None, "nevergrad", "anneal", "hybrid"}, optional
            Override the finder offline search setting. Nevergrad optimizes
            only the returned fixed plan; annealing explores subtree
            replacements. ``"hybrid"`` combines both with one shared budget.
            Neither mutates a live TTN.
        search_budget, seed, nevergrad_optimizer
            Optional offline-search configuration for each candidate plan.
        progbar : bool, optional
            Display local-search progress for each candidate.

        Returns
        -------
        dict
            ``{"objective", "recommended_block_size", "order", "chi", "plan",
            "candidates"}``.  Each candidate carries its ``block_size``, the
            :class:`TreePlan`, ``max_bond_cut``, and the same structural/cost
            summary fields as :meth:`recommend_arities`.
        """
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        settings = self._resolve_search_settings(
            refine=refine,
            refine_budget=refine_budget,
            topology_refine=topology_refine,
            topology_budget=topology_budget,
            search=search,
            search_budget=search_budget,
            seed=seed,
            nevergrad_optimizer=nevergrad_optimizer,
        )
        options = []
        for bs in block_sizes:
            key = int(bs)
            if key < 1:
                raise ValueError("block_sizes must be >= 1.")
            if key not in options:
                options.append(key)
        if not options:
            raise ValueError("block_sizes must contain at least one option.")

        if order is None:
            order = self.qubit_order()
        else:
            order = [int(q) for q in order]

        # ``recommend_layered`` compares a fixed block-family. Keep the
        # all-scale objective, but do not silently replace that family with
        # arbitrary subtree topologies unless the caller explicitly asks for
        # those search stages.
        if self.objective == "full_tree":
            if topology_refine is _DEFAULT_SEARCH_OPTION:
                settings["topology_refine"] = None
                settings["topology_budget"] = None
            if search is _DEFAULT_SEARCH_OPTION:
                settings["search"] = None
                settings["search_budget"] = self.search_budget

        candidates = []
        for bs in options:
            plan = TreePlan.build_layered(
                order,
                block_size=bs,
                root_qubit=self.root_qubit,
            )
            plan, planning = self._improve_plan(
                plan,
                chi=chi,
                settings=settings,
                progbar=progbar,
            )
            report = self.report(
                plan, include_edge_loads=self.objective != "path"
            )
            arity_histogram = {}
            for node, children in plan.children.items():
                if not children:
                    continue
                arity_histogram[len(children)] = (
                    arity_histogram.get(len(children), 0) + 1
                )
            candidates.append({
                "block_size": bs,
                "actual_max_arity": plan.max_arity(),
                "root_arity": len(plan.children[plan.root]),
                "arity_histogram": arity_histogram,
                "score": report["score"],
                "max_path": report["max_path"],
                "max_edge_load": report["max_edge_load"],
                "peak_bond_growth": report["peak_bond_growth"],
                "full_tree_profile": report["full_tree"],
                "max_virtual_degree": report["max_virtual_degree"],
                "total_virtual_degree": report["total_virtual_degree"],
                "estimated_max_tensor_log2": report[
                    "estimated_max_tensor_log2"
                ],
                "estimated_total_tensor_log2": report[
                    "estimated_total_tensor_log2"
                ],
                **_chi_cut_fields(plan, chi),
                "order": self._leaf_order(plan),
                "planning": planning,
                "plan": plan,
            })

        def candidate_key(candidate):
            return self._selection_key(candidate["plan"], chi) + (
                candidate["block_size"],
            )

        recommended = min(candidates, key=candidate_key)
        return {
            "objective": self.objective,
            "recommended_block_size": recommended["block_size"],
            "initial_order": tuple(order),
            "order": recommended["order"],
            "chi": chi,
            "refine": settings["refine"],
            "topology_refine": settings["topology_refine"],
            "search": settings["search"],
            "plan": recommended["plan"],
            "candidates": candidates,
        }

    def run(
        self,
        *,
        order=_DEFAULT_ORDER,
        chi=_DEFAULT_CHI,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        topology_refine=_DEFAULT_SEARCH_OPTION,
        topology_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
        progbar=False,
    ):
        """Return a TreePlan for the selected layout objective.

        A scalar ``max_arity`` (the default ``2``) builds one fixed binary
        plan with the default ternary virtual root. When the finder is built
        with an iterable of candidate arities, this searches them with
        :meth:`recommend_arities` -- ``chi``-aware when the finder carries a
        ``chi`` -- and returns the objective-best plan.

        ``chi`` and the fixed-plan ``refine`` / ``search`` controls can be
        overridden for this call. Pass ``progbar=True`` to display greedy and
        offline search progress. Omitted values inherit the corresponding
        finder settings, so the original zero-argument behavior is unchanged.
        The explicit ``objective="hypergraph"`` mode is the one exception:
        when no refinement controls are supplied, it enables bounded greedy
        and binary-NNI stages so the full support hyperedges directly
        influence the returned layout.

        ``objective="full_tree"`` enables bounded all-scale subtree
        reconfiguration and annealing by default when no search controls are
        explicitly supplied. It evaluates every tree scale using dynamic
        bond-pressure and tensor-work proxies.

        ``order="quality"`` is the high-quality mode matching the MPS layout
        API: it upgrades the effective objective to ``"full_tree"``, enables
        greedy leaf refinement, all-scale subtree topology refinement, and
        hybrid topology annealing plus Nevergrad leaf search. When Nevergrad
        is unavailable, it selects dependency-free simulated annealing. Pass
        ``search=None`` or ``refine=None`` explicitly to disable either stage.

        An explicit site permutation can also be passed as ``order``. This
        returns the corresponding fixed tree immediately, without layout
        refinement or offline search.
        """
        if order is _DEFAULT_ORDER:
            order = self.order
        else:
            order = _normalize_layout_order(order)
        geometric_order = isinstance(order, str) and order != "quality"
        map_mode = order if isinstance(order, str) and order.startswith("coarse-") else None
        if geometric_order:
            order = self._preset_order(order)
        if not isinstance(order, str) and order is not None:
            if self.arity_candidates is not None:
                raise ValueError(
                    "an explicit site order requires scalar max_arity; "
                    "pass one fixed arity instead of arity candidates."
                )
            fixed_order = normalize_fixed_order(order, self.leaf_qubits)
            return TreePlan.from_order(
                fixed_order,
                # Named geometric modes are exact baselines: do not apply the
                # interaction-aware spectral reorder used by the default
                # quality structure. Explicit caller-provided permutations
                # retain the historical ``self.structure`` behavior.
                structure="balanced" if geometric_order else self.structure,
                max_arity=self.max_arity,
                root_qubit=self.root_qubit,
                top_arity=self.top_arity,
                map_mode=map_mode,
            )
        if order == "quality":
            if self.objective != "full_tree":
                self.objective = "full_tree"
                # Objective-dependent candidate caches may have been filled by
                # an earlier fast/path run on this finder. Quality mode must
                # not reuse those scores after upgrading to full-tree cost.
                self._plan_cache.clear()
                self._edge_load_cache.clear()
                self._rank_diagnostics_cache.clear()
                self._full_tree_profile_cache.clear()
                self._full_tree_structure_cache.clear()
                self._schmidt_rank_cache.clear()
                self._similarity_cache.clear()
                self._congestion_weights_cache = None
                self._balanced_plan_cache = None
            if refine is _DEFAULT_SEARCH_OPTION:
                refine = self.refine or "greedy"
            if topology_refine is _DEFAULT_SEARCH_OPTION:
                topology_refine = self.topology_refine or "subtree"
            if search is _DEFAULT_SEARCH_OPTION:
                search = self.search or _quality_search_mode()
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        settings = self._resolve_search_settings(
            refine=refine,
            refine_budget=refine_budget,
            topology_refine=topology_refine,
            topology_budget=topology_budget,
            search=search,
            search_budget=search_budget,
            seed=seed,
            nevergrad_optimizer=nevergrad_optimizer,
        )
        if self.arity_candidates is not None:
            rec = self.recommend_arities(
                self.arity_candidates,
                chi=chi,
                progbar=progbar,
                **settings,
            )
            self._last_arity_recommendation = rec
            self._selected_candidate = f"arity={rec['recommended_max_arity']}"
            self._last_candidate_scores = {
                f"arity={cand['max_arity']}": self._selection_key(
                    cand["plan"], rec["chi"]
                )
                for cand in rec["candidates"]
            }
            return rec["plan"]
        candidates = self._candidate_plans(self.max_arity)
        if (
            settings["topology_refine"] is not None
            or settings["refine"] is not None
            or settings["search"] is not None
        ):
            candidates = {
                name: self._improve_plan(
                    plan,
                    chi=chi,
                    settings=settings,
                    progbar=progbar,
                )[0]
                for name, plan in candidates.items()
            }
        selected = min(
            candidates,
            key=lambda name: self._selection_key(candidates[name], chi),
        )
        self._last_candidates = candidates
        self._last_candidate_scores = {
            name: self._selection_key(plan, chi)
            for name, plan in candidates.items()
        }
        self._selected_candidate = selected
        return candidates[selected]

    def candidate_plans(
        self,
        *,
        chi=_DEFAULT_CHI,
        include_quality=False,
        quality_refine_budget=None,
        quality_topology_budget=None,
        quality_search=_DEFAULT_SEARCH_OPTION,
        quality_search_budget=_DEFAULT_SEARCH_OPTION,
        quality_seed=_DEFAULT_SEARCH_OPTION,
    ):
        """Return immutable candidate plans for optional pilot replay.

        The normal :meth:`run` path remains static and cheap. This method
        exposes the interaction, congestion, balanced, and arity candidates
        that a state-aware pilot can compare without rebuilding the finder.
        Candidate names are stable strings such as
        ``"congestion:arity=2"``.

        Parameters
        ----------
        chi : int, optional
            Bond-dimension budget used in candidate ranking.
        include_quality : bool, optional
            Also add one ``"quality:arity=..."`` candidate per arity. These
            candidates start from the static objective candidates and apply
            bounded greedy leaf and topology refinement. For
            ``objective="full_tree"`` this means all-scale subtree search and
            the configured quality search; it is deliberately opt-in because
            it is more expensive than the static candidate list.
        quality_refine_budget, quality_topology_budget : int, optional
            Bounds for the quality candidate's leaf-swap and NNI proposals.
            Each defaults to the normal bounded quality-mode budget.
        quality_search : {None, "anneal", "nevergrad", "hybrid"}, optional
            Optional second-stage search for quality candidates. The default
            is no second stage for older objectives and ``"hybrid"`` (or
            dependency-free ``"anneal"``) for ``objective="full_tree"``.
        quality_search_budget, quality_seed : int, optional
            Budget and seed for ``quality_search``. These are planning-only
            controls; no tensor state is allocated or replayed here.
        """
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        arities = (
            tuple(self.arity_candidates)
            if self.arity_candidates is not None
            else (self.max_arity,)
        )
        result = {}
        quality_settings = None
        if include_quality:
            if quality_search is _DEFAULT_SEARCH_OPTION:
                quality_search = (
                    _quality_search_mode()
                    if self.objective == "full_tree" else None
                )
            if quality_search_budget is _DEFAULT_SEARCH_OPTION:
                quality_search_budget = self.search_budget
            elif quality_search_budget is not None:
                quality_search_budget = _validate_search_budget(
                    quality_search_budget, "quality_search_budget"
                )
            if quality_seed is _DEFAULT_SEARCH_OPTION:
                quality_seed = self.seed
            else:
                try:
                    quality_seed = int(quality_seed)
                except (TypeError, ValueError) as exc:
                    raise ValueError("quality_seed must be an integer.") from exc
            quality_topology_refine = (
                "subtree" if self.objective == "full_tree" else "nni"
            )
            quality_settings = self._resolve_search_settings(
                refine="greedy",
                refine_budget=quality_refine_budget,
                topology_refine=quality_topology_refine,
                topology_budget=quality_topology_budget,
                search=quality_search,
                search_budget=quality_search_budget,
                seed=quality_seed,
            )
        for arity in arities:
            plans = self._candidate_plans(arity)
            for name, plan in plans.items():
                key = f"{name}:arity={arity}"
                result[key] = {
                    "plan": plan,
                    "objective_key": self._selection_key(plan, chi),
                    "path_score": self.score(plan),
                    "tensor_cost": self._tensor_cost_key(plan),
                    "edge_loads": self.edge_loads(plan),
                }
            if quality_settings is not None:
                refined_candidates = []
                for name, plan in plans.items():
                    refined, planning = self._improve_plan(
                        plan,
                        chi=chi,
                        settings=quality_settings,
                    )
                    refined_candidates.append((
                        self._selection_key(refined, chi),
                        name,
                        refined,
                        planning,
                    ))
                _key, source_name, quality_plan, planning = min(
                    refined_candidates,
                    key=lambda item: (item[0], item[1]),
                )
                result[f"quality:arity={arity}"] = {
                    "plan": quality_plan,
                    "objective_key": self._selection_key(quality_plan, chi),
                    "path_score": self.score(quality_plan),
                    "tensor_cost": self._tensor_cost_key(quality_plan),
                    "edge_loads": self.edge_loads(quality_plan),
                    "selected_from": source_name,
                    "planning": planning,
                }
        return result

    def targeted_candidates(
        self,
        plan,
        edge_diagnostics,
        *,
        chi=_DEFAULT_CHI,
        budget=32,
        seed=0,
    ):
        """Propose static plans around replay-hot tree edges.

        This is the circuit-only feedback hook used by
        :meth:`TreeOptimizer.optimize_layout`. ``edge_diagnostics`` is the
        per-edge report produced by a short pilot replay. The method uses the
        measured hot edges only to choose where to explore; every proposed
        plan is still ranked by the configured static layout objective. It
        never constructs a tensor network, applies a gate, or truncates a
        state.

        The proposals include binary NNI moves where valid, local subtree
        reconfigurations at the hot edge, and leaf exchanges across the hot
        cut. Returned plans are immutable and deduplicated.
        """
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan.")
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        budget = _validate_search_budget(budget, "budget")
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer.") from exc
        if budget is None or budget < 1:
            return []
        if not hasattr(edge_diagnostics, "items"):
            raise TypeError("edge_diagnostics must be a mapping of edge metrics.")

        def hot_key(item):
            edge, metrics = item
            metrics = metrics if hasattr(metrics, "get") else {}
            return (
                -float(metrics.get("discarded_fraction", 0.0) or 0.0),
                -float(metrics.get("discarded_weight", 0.0) or 0.0),
                -int(metrics.get("truncated", 0) or 0),
                tuple(edge),
            )

        hot_edges = []
        for edge, metrics in sorted(edge_diagnostics.items(), key=hot_key):
            try:
                edge = tuple(int(x) for x in edge)
            except (TypeError, ValueError):
                continue
            if len(edge) != 2:
                continue
            parent, child = edge
            if plan.parent.get(child) != parent:
                continue
            hot_edges.append((edge, metrics))
        if not hot_edges:
            return []

        rng = np.random.default_rng(seed)
        leaf_nodes = self._leaf_nodes(plan)
        below = plan.subtree_qubit_masks()
        proposals = []
        seen = set()

        def signature(candidate):
            return (
                candidate.root,
                candidate.root_qubit,
                tuple(
                    sorted(
                        (node, tuple(children))
                        for node, children in candidate.children.items()
                    )
                ),
                tuple(sorted(candidate.qubit_of_leaf.items())),
            )

        original_signature = signature(plan)

        def add(candidate):
            if candidate is None:
                return
            key = signature(candidate)
            if key == original_signature or key in seen:
                return
            seen.add(key)
            proposals.append(candidate)

        for (parent, child), _metrics in hot_edges:
            if len(proposals) >= budget:
                break
            if (parent, child) in self._nni_edges(plan):
                for variant in (0, 1):
                    if len(proposals) >= budget:
                        break
                    add(self._plan_with_nni(plan, parent, child, variant))

            # Rebuild the smaller side first. A second draw at the parent
            # lets the search change the attachment itself when the hot edge
            # is a poor cut rather than merely a poor local ordering.
            for node in (child, parent):
                if len(proposals) >= budget:
                    break
                if node not in plan.children:
                    continue
                candidate = self._plan_with_subtree_reconfiguration(
                    plan, node, rng
                )
                add(candidate)

            hot_qubits = [
                leaf for leaf in leaf_nodes
                if below[child] & (1 << plan.qubit_of_leaf[leaf])
            ]
            cold_qubits = [leaf for leaf in leaf_nodes if leaf not in hot_qubits]
            if hot_qubits and cold_qubits:
                rng.shuffle(hot_qubits)
                rng.shuffle(cold_qubits)
                for left_leaf, right_leaf in zip(hot_qubits, cold_qubits):
                    if len(proposals) >= budget:
                        break
                    add(self._plan_with_leaf_swap(plan, left_leaf, right_leaf))

        # If the first cut generated fewer than the requested proposals, use
        # deterministic neighbouring swaps so a pilot round still has a
        # useful bounded exploration budget on shallow trees.
        if len(proposals) < budget:
            for left_leaf, right_leaf in zip(leaf_nodes, leaf_nodes[1:]):
                if len(proposals) >= budget:
                    break
                add(self._plan_with_leaf_swap(plan, left_leaf, right_leaf))
        proposals.sort(key=lambda candidate: self._selection_key(candidate, chi))
        return proposals[:budget]

    def recommend_arities(
        self,
        max_arities=(2, 3, 4),
        *,
        chi=_DEFAULT_CHI,
        refine=_DEFAULT_SEARCH_OPTION,
        refine_budget=_DEFAULT_SEARCH_OPTION,
        topology_refine=_DEFAULT_SEARCH_OPTION,
        topology_budget=_DEFAULT_SEARCH_OPTION,
        search=_DEFAULT_SEARCH_OPTION,
        search_budget=_DEFAULT_SEARCH_OPTION,
        seed=_DEFAULT_SEARCH_OPTION,
        nevergrad_optimizer=_DEFAULT_SEARCH_OPTION,
        progbar=False,
    ):
        """Compare binary and wider trees and return the best candidate.

        The returned mapping contains the recommended :class:`TreePlan` under
        ``"plan"`` and candidate plans alongside structural/cost summaries
        under ``"candidates"``.  Wider arities shorten paths but increase local
        tensor degree, so the recommendation uses the selected layout
        objective and reports both effects.

        Parameters
        ----------
        max_arities : iterable of int or None
            Candidate maximum arities (``2`` = binary; ``None`` = unbounded).
        chi : int, optional
            Bond-dimension budget.  When given, the recommendation is made
            ``chi``-aware exactly as in :meth:`recommend_layered`: candidates
            are ranked first by ``chi_overflow`` so a structure that stays
            exact at ``chi`` (widest bond ``<= log2(chi)``) is preferred, and
            the layout objective only breaks ties.  Each candidate reports
            ``max_bond_cut``, ``chi_overflow``, and ``exact_at_chi``.
            When omitted, uses the ``chi`` supplied to the finder; pass
            ``chi=None`` explicitly for a chi-blind comparison.
        refine : {None, "greedy"}, optional
            Override the finder refinement setting for each arity candidate.
        refine_budget : int, optional
            Maximum number of greedy proposals per arity candidate.
        topology_refine : {None, "nni", "subtree"}, optional
            Override the optional topology refinement for each candidate.
        topology_budget : int, optional
            Maximum number of topology proposals per candidate.
        search : {None, "nevergrad", "anneal", "hybrid"}, optional
            Override the optional offline search for each candidate.
        search_budget : int, optional
            Budget for the selected offline search.
        seed : int, optional
            Seed for deterministic candidate refinement and search.
        nevergrad_optimizer : str, optional
            Nevergrad optimizer name used by the optional search stage.
        progbar : bool, optional
            Display local-search progress for each candidate.
        """
        if chi is _DEFAULT_CHI:
            chi = self.chi
        else:
            chi = _validate_chi(chi)
        settings = self._resolve_search_settings(
            refine=refine,
            refine_budget=refine_budget,
            topology_refine=topology_refine,
            topology_budget=topology_budget,
            search=search,
            search_budget=search_budget,
            seed=seed,
            nevergrad_optimizer=nevergrad_optimizer,
        )
        options = []
        for arity in max_arities:
            if arity is None:
                key = None
            else:
                key = int(arity)
                if key < 2:
                    raise ValueError("max_arities must be >= 2 or None.")
            if key not in options:
                options.append(key)
        if not options:
            raise ValueError("max_arities must contain at least one option.")

        candidates = []
        for arity in options:
            plan = self._select_plan(arity)
            plan, planning = self._improve_plan(
                plan,
                chi=chi,
                settings=settings,
                progbar=progbar,
            )
            report = self.report(
                plan, include_edge_loads=self.objective != "path"
            )
            arity_histogram = {}
            for node, children in plan.children.items():
                if not children:
                    continue
                arity_histogram[len(children)] = (
                    arity_histogram.get(len(children), 0) + 1
                )
            candidates.append({
                "max_arity": arity,
                "actual_max_arity": plan.max_arity(),
                "is_binary": plan.is_binary(),
                "arity_histogram": arity_histogram,
                "max_virtual_degree": max(
                    (
                        len(children) + (1 if node in plan.parent else 0)
                        for node, children in plan.children.items()
                        if children
                    ),
                    default=0,
                ),
                "total_virtual_degree": sum(
                    len(children) + (1 if node in plan.parent else 0)
                    for node, children in plan.children.items()
                    if children
                ),
                "score": report["score"],
                "max_path": report["max_path"],
                "max_edge_load": report["max_edge_load"],
                "peak_bond_growth": report["peak_bond_growth"],
                "full_tree_profile": report["full_tree"],
                "estimated_max_tensor_log2": report[
                    "estimated_max_tensor_log2"
                ],
                "estimated_total_tensor_log2": report[
                    "estimated_total_tensor_log2"
                ],
                **_chi_cut_fields(plan, chi),
                "order": self._leaf_order(plan),
                "planning": planning,
                "plan": plan,
            })

        def candidate_key(candidate):
            return self._selection_key(candidate["plan"], chi) + (
                candidate["actual_max_arity"],
            )

        recommended = min(candidates, key=candidate_key)
        return {
            "objective": self.objective,
            "recommended_max_arity": recommended["max_arity"],
            "chi": chi,
            "refine": settings["refine"],
            "topology_refine": settings["topology_refine"],
            "search": settings["search"],
            "plan": recommended["plan"],
            "candidates": candidates,
        }

    def recommend_layout(self, max_arities=(2, 3, 4), **kwargs):
        """Alias for :meth:`recommend_arities` with a layout-oriented name."""
        return self.recommend_arities(max_arities=max_arities, **kwargs)

    def _congestion_pair_weights(self):
        """Return pair weights proportional to gate Schmidt load."""
        cached = getattr(self, "_congestion_weights_cache", None)
        if cached is not None:
            return cached
        event_weights = []
        for payload, support, event_type, temporal_factor in zip(
            self.payloads,
            self.supports,
            self.event_types,
            self.temporal_factors,
        ):
            if len(support) < 2 or str(event_type).lower() in {
                "measure", "reset", "measure_reset", "cap"
            }:
                event_weights.append(0.0)
                continue
            if payload is None:
                event_weights.append(float(temporal_factor))
                continue
            support = tuple(dict.fromkeys(support))
            logs = []
            if len(support) <= 8:
                for mask in range(1, (1 << len(support)) - 1):
                    left = tuple(
                        site for i, site in enumerate(support)
                        if mask & (1 << i)
                    )
                    rank = self._schmidt_rank(payload, support, left)
                    logs.append(float(np.log2(rank)))
            else:
                for site in support:
                    rank = self._schmidt_rank(payload, support, (site,))
                    logs.append(float(np.log2(rank)))
            event_weights.append(float(temporal_factor) * max(logs, default=1.0))
        self._congestion_weights_cache = _gate_stream_pair_weights(
            self.supports,
            range(self.n),
            event_weights,
        )
        return self._congestion_weights_cache

    def _subtree_qubits(self, plan):
        """Return the qubits below each node of plan."""
        return {
            node: frozenset(
                q for q in range(self.n) if mask & (1 << q)
            )
            for node, mask in plan.subtree_qubit_masks().items()
        }

    def edge_loads(self, plan=None):
        """Return predicted log-bond growth for every tree edge."""
        if plan is None:
            plan = self.run()
        cache_key = id(plan)
        cached = self._edge_load_cache.get(cache_key)
        if cached is not None and cached[0] is plan:
            return dict(cached[1])
        below = plan.subtree_qubit_masks()
        loads = {
            (parent, child): 0.0
            for parent, children in plan.children.items()
            for child in children
        }
        rank_diagnostics = {
            "exact_events": 0,
            "bounded_events": 0,
            "reasons": {},
        }
        for payload, support, event_type, temporal_factor in zip(
            self.payloads,
            self.supports,
            self.event_types,
            self.temporal_factors,
        ):
            support = tuple(dict.fromkeys(support))
            if (
                temporal_factor <= 0.0
                or len(support) < 2
                or str(event_type).lower() in {
                    "measure", "reset", "measure_reset", "cap"
                }
            ):
                continue
            support_mask = 0
            for site in support:
                support_mask |= 1 << site

            # An edge crosses the support iff it belongs to the minimal
            # subtree spanning the support nodes. Scanning every tree edge
            # is needlessly O(n) for each event; for the dominant two-qubit
            # case this reduces the work to the site-to-site geodesic.
            site_nodes = [plan.node_of_qubit[site] for site in support]
            if len(site_nodes) == 2:
                # This branch dominates ordinary circuit layout. Avoid sets
                # and an all-node parent scan: every path hop is one crossed
                # rooted tree edge.
                path = plan.node_path(site_nodes[0], site_nodes[1])
                crossed_edges = [
                    (u, v) if plan.parent.get(v) == u else (v, u)
                    for u, v in zip(path, path[1:])
                ]
            else:
                span_nodes = set()
                anchor = site_nodes[0]
                for site_node in site_nodes:
                    span_nodes.update(plan.node_path(anchor, site_node))
                crossed_edges = [
                    (parent, node)
                    for node in span_nodes
                    if (parent := plan.parent.get(node)) in span_nodes
                ]

            for edge in crossed_edges:
                _parent, child = edge
                left_mask = support_mask & below[child]
                if not left_mask or left_mask == support_mask:
                    continue
                left = tuple(
                    site for site in support if left_mask & (1 << site)
                )
                info = self._schmidt_rank_info(payload, support, left)
                rank = int(info["rank"])
                loads[edge] += float(temporal_factor) * float(
                    np.log2(max(1, rank))
                )
                if info["exact"]:
                    rank_diagnostics["exact_events"] += 1
                else:
                    rank_diagnostics["bounded_events"] += 1
                    reason = info["reason"]
                    rank_diagnostics["reasons"][reason] = (
                        rank_diagnostics["reasons"].get(reason, 0) + 1
                    )
        # Retain the plan alongside its id so a future id reuse cannot return
        # diagnostics for an unrelated short-lived plan.
        self._edge_load_cache[cache_key] = (plan, dict(loads))
        self._rank_diagnostics_cache[cache_key] = (
            plan,
            rank_diagnostics,
        )
        return dict(loads)

    @staticmethod
    def _log2_add(total, value):
        """Add two positive quantities represented by their log2 values."""
        if value == -np.inf:
            return float(total)
        if total == -np.inf:
            return float(value)
        return float(np.logaddexp2(total, value))

    def _support_span(self, plan, support):
        """Return the mask, nodes, and rooted edges spanned by ``support``."""
        support = tuple(dict.fromkeys(support))
        support_mask = 0
        for site in support:
            support_mask |= 1 << site
        site_nodes = [plan.node_of_qubit[site] for site in support]
        if len(site_nodes) == 2:
            path = plan.node_path(site_nodes[0], site_nodes[1])
            span_nodes = set(path)
            crossed_edges = [
                (u, v) if plan.parent.get(v) == u else (v, u)
                for u, v in zip(path, path[1:])
            ]
        else:
            span_nodes = set()
            anchor = site_nodes[0]
            for site_node in site_nodes:
                span_nodes.update(plan.node_path(anchor, site_node))
            crossed_edges = [
                (parent, node)
                for node in span_nodes
                if (parent := plan.parent.get(node)) in span_nodes
            ]
        return support_mask, span_nodes, crossed_edges

    def full_tree_profile(self, plan=None):
        """Return a dynamic, all-scale cost profile for ``plan``.

        The profile is a cheap layout proxy, not a replacement for replaying
        the circuit. It accumulates uncapped operator-Schmidt demand on every
        tree edge, tracks capped working-bond pressure and predicted ``chi``
        overflow at the configured ``chi``, estimates tensor widths and
        write/work volume for every touched node, and groups those quantities
        by hierarchical tree scale. It never allocates a TTN or performs a
        tensor truncation.
        """
        if plan is None:
            plan = self.run()
        cache_key = id(plan)
        cached = self._full_tree_profile_cache.get(cache_key)
        if cached is not None and cached[0] is plan:
            return cached[1]

        below = plan.subtree_qubit_masks()
        structure_key = (
            plan.root,
            plan.root_qubit,
            tuple(
                sorted(
                    (node, tuple(children))
                    for node, children in plan.children.items()
                )
            ),
            tuple(sorted(plan.qubit_of_node)),
        )
        cached_structure = self._full_tree_structure_cache.get(structure_key)
        if cached_structure is None:
            node_scales = _tree_node_scales(plan)
            edges = tuple(
                (parent, child)
                for parent, children in plan.children.items()
                for child in children
            )
            self._full_tree_structure_cache[structure_key] = (
                node_scales,
                edges,
            )
            structure_reused = False
        else:
            node_scales, edges = cached_structure
            structure_reused = True
        demand_log = {edge: 0.0 for edge in edges}
        bond_log = {edge: 0.0 for edge in edges}
        log_chi = (
            float(np.log2(self.chi)) if self.chi is not None else float("inf")
        )
        scales = {}
        for scale in sorted(set(node_scales.values())):
            scales[scale] = {
                "node_count": 0,
                "edge_count": 0,
                "peak_tensor_log2": 0.0,
                "log_total_tensor_size": -np.inf,
                "peak_edge_demand_log2": 0.0,
                "total_edge_demand_log2": 0.0,
                "peak_bond_log2": 0.0,
                "peak_overflow_log2": 0.0,
            }
        for node, scale in node_scales.items():
            scales[scale]["node_count"] += 1
        for parent, child in edges:
            scales[node_scales[child]]["edge_count"] += 1

        def node_log_size(node):
            incident = []
            if node in plan.parent:
                incident.append((plan.parent[node], node))
            incident.extend((node, child) for child in plan.children[node])
            return float(
                int(node in plan.qubit_of_node)
                + sum(bond_log[edge] for edge in incident)
            )

        peak_tensor_log2 = 0.0
        peak_work_log2 = 0.0
        log_total_write = -np.inf
        log_total_work = -np.inf
        total_route_length = 0
        event_count = 0
        exact_events = 0
        bounded_events = 0
        bound_reasons = {}
        support_span_cache = {}

        for payload, support, event_type, temporal_factor in zip(
            self.payloads,
            self.supports,
            self.event_types,
            self.temporal_factors,
        ):
            support = tuple(dict.fromkeys(support))
            if (
                temporal_factor <= 0.0
                or len(support) < 2
                or str(event_type).lower() in {
                    "measure", "reset", "measure_reset", "cap"
                }
            ):
                continue
            span_key = support
            cached_span = support_span_cache.get(span_key)
            if cached_span is None:
                cached_span = self._support_span(plan, support)
                support_span_cache[span_key] = cached_span
            support_mask, span_nodes, crossed_edges = cached_span
            if not crossed_edges:
                continue
            event_count += 1
            total_route_length += len(crossed_edges)
            for edge in crossed_edges:
                _parent, child = edge
                left_mask = support_mask & below[child]
                if not left_mask or left_mask == support_mask:
                    continue
                left = tuple(
                    site for site in support if left_mask & (1 << site)
                )
                info = self._schmidt_rank_info(payload, support, left)
                delta = float(temporal_factor) * float(
                    np.log2(max(1, int(info["rank"])))
                )
                demand_log[edge] += delta
                bond_log[edge] = min(log_chi, bond_log[edge] + delta)
                if info["exact"]:
                    exact_events += 1
                else:
                    bounded_events += 1
                    reason = info["reason"]
                    bound_reasons[reason] = bound_reasons.get(reason, 0) + 1

            tensor_logs = [node_log_size(node) for node in span_nodes]
            event_write_log = max(tensor_logs, default=0.0)
            event_work_logs = [
                log_size + 2.0 * len(support)
                for log_size in tensor_logs
            ]
            event_work_log = max(event_work_logs, default=0.0)
            peak_tensor_log2 = max(peak_tensor_log2, event_write_log)
            peak_work_log2 = max(peak_work_log2, event_work_log)
            log_total_write = self._log2_add(
                log_total_write, event_write_log
            )
            log_total_work = self._log2_add(
                log_total_work, event_work_log
            )
            for node, log_size in zip(span_nodes, tensor_logs):
                scale = scales[node_scales[node]]
                scale["peak_tensor_log2"] = max(
                    scale["peak_tensor_log2"], log_size
                )
                scale["log_total_tensor_size"] = self._log2_add(
                    scale["log_total_tensor_size"], log_size
                )
            for edge in crossed_edges:
                scale = scales[node_scales[edge[1]]]
                scale["peak_edge_demand_log2"] = max(
                    scale["peak_edge_demand_log2"], demand_log[edge]
                )
                scale["peak_bond_log2"] = max(
                    scale["peak_bond_log2"], bond_log[edge]
                )
                if np.isfinite(log_chi):
                    scale["peak_overflow_log2"] = max(
                        scale["peak_overflow_log2"],
                        max(0.0, demand_log[edge] - log_chi),
                    )

        for node in plan.children:
            log_size = node_log_size(node)
            scale = scales[node_scales[node]]
            scale["peak_tensor_log2"] = max(
                scale["peak_tensor_log2"], log_size
            )
            scale["log_total_tensor_size"] = self._log2_add(
                scale["log_total_tensor_size"], log_size
            )
        for edge, demand in demand_log.items():
            scale = scales[node_scales[edge[1]]]
            scale["peak_edge_demand_log2"] = max(
                scale["peak_edge_demand_log2"], demand
            )
            scale["peak_bond_log2"] = max(
                scale["peak_bond_log2"], bond_log[edge]
            )
            scale["total_edge_demand_log2"] += demand

        overflow_log = {
            edge: (
                max(0.0, demand - log_chi)
                if np.isfinite(log_chi) else 0.0
            )
            for edge, demand in demand_log.items()
        }

        profile = {
            "event_count": event_count,
            "peak_tensor_log2": float(peak_tensor_log2),
            "peak_work_log2": float(peak_work_log2),
            "log_total_write": float(
                0.0 if log_total_write == -np.inf else log_total_write
            ),
            "log_total_work": float(
                0.0 if log_total_work == -np.inf else log_total_work
            ),
            "peak_edge_demand_log2": float(max(demand_log.values(), default=0.0)),
            "total_edge_demand_log2": float(sum(demand_log.values())),
            "peak_bond_log2": float(max(bond_log.values(), default=0.0)),
            "peak_overflow_log2": float(max(overflow_log.values(), default=0.0)),
            "total_overflow_log2": float(sum(overflow_log.values())),
            "total_route_length": int(total_route_length),
            "exact_events": int(exact_events),
            "bounded_events": int(bounded_events),
            "bound_reasons": bound_reasons,
            "cache": {
                "structure_reused": bool(structure_reused),
                "unique_supports": len(support_span_cache),
            },
            "scales": scales,
        }
        self._full_tree_profile_cache[cache_key] = (plan, profile)
        return profile

    def _congestion_key(self, plan):
        """Return the lexicographic key used by the load-aware objective."""
        loads = self.edge_loads(plan)
        values = tuple(loads.values())
        return (
            max(values, default=0.0),
            sum(values),
            self.score(plan),
            max(
                (plan.tree_distance(a, b) for a in range(self.n)
                 for b in range(a + 1, self.n)),
                default=0,
            ),
        )

    def _similarity_weights(self, pair_weights=None):
        """Return the qubit-pair similarity of Seitz et al. (Eq. 1).

        ``s(qi, qj) = |G(qi) & G(qj)| + 1 / (|G(qi)| + |G(qj)|)`` where ``G(q)``
        is the set of multi-qubit events acting on qubit ``q``.  The integer
        co-occurrence term ``|G(qi) & G(qj)|`` is exactly the interaction weight
        already accumulated in :attr:`pair_weights`; the ``1/(deg_i + deg_j)``
        term is a tie-breaker that gently favours grouping qubits participating
        in fewer gates, biasing the recursive bisection towards balanced
        subtrees when co-occurrence counts tie.  Only the bisection uses this
        augmented similarity; :meth:`score` keeps the pure interaction weight.
        """
        if pair_weights is None:
            pair_weights = self.pair_weights
        cache_key = id(pair_weights)
        cached = self._similarity_cache.get(cache_key)
        if cached is not None and cached[0] is pair_weights:
            return cached[1]

        degree = {q: 0.0 for q in range(self.n)}
        for support in self.supports:
            for site in set(support):
                if isinstance(site, int) and 0 <= site < self.n:
                    degree[site] += 1.0
        sim = dict(pair_weights)
        for qi in range(self.n):
            for qj in range(qi + 1, self.n):
                deg = degree[qi] + degree[qj]
                if deg > 0.0:
                    sim[(qi, qj)] = sim.get((qi, qj), 0.0) + 1.0 / deg
        self._similarity_cache[cache_key] = (pair_weights, sim)
        return sim

    def score(self, plan):
        """Return the total interaction-weighted tree-path length of ``plan``.

        Lower is better: this is the quantity the tree structure minimises
        (short physical-node paths for strongly coupled qubits).
        """
        return self._path_score_and_max(plan)[0]

    def _balanced_plan(self):
        """Return the cached index-order balanced comparison plan."""
        if self._balanced_plan_cache is None:
            self._balanced_plan_cache = TreePlan.from_order(
                self.leaf_qubits,
                structure="balanced",
                root_qubit=self.root_qubit,
                top_arity=self.top_arity,
            )
        return self._balanced_plan_cache

    def report(self, plan=None, *, include_edge_loads=True):
        """Return layout-quality diagnostics for ``plan`` (or a fresh run).

        The dominant lever for tree-tensor-network accuracy at fixed ``chi`` is
        how well the tree keeps strongly coupled qubits as nearby nodes: a
        two-qubit gate threads its virtual bond along the whole site-to-site
        geodesic, and every crossed bond can grow.  This report summarises those
        geodesic lengths over the interaction graph and compares the chosen
        structure against a naive balanced index-order tree (lower ``score`` is
        better).
        """
        if plan is None:
            plan = self.run()
        dists = []
        total_weight = 0.0
        weighted_sum = 0.0
        for (qa, qb), weight in self.pair_weights.items():
            d = plan.tree_distance(qa, qb)
            dists.append(d)
            weighted_sum += float(weight) * d
            total_weight += float(weight)
        n_pairs = len(dists)
        balanced = self._balanced_plan()
        balanced_score = self.score(balanced)
        if include_edge_loads:
            loads = self.edge_loads(plan)
            balanced_loads = self.edge_loads(balanced)
            rank_info = self._rank_diagnostics_cache.get(id(plan), (plan, {}))[1]
        else:
            loads = None
            balanced_loads = None
            rank_info = {}
        max_load = max(loads.values(), default=0.0) if loads is not None else None
        total_load = sum(loads.values()) if loads is not None else None
        balanced_max_load = (
            max(balanced_loads.values(), default=0.0)
            if balanced_loads is not None else None
        )
        balanced_total_load = (
            sum(balanced_loads.values())
            if balanced_loads is not None else None
        )
        hybrid_cost = None
        if self.objective == "hybrid" and loads is not None:
            hybrid_cost = self._hybrid_key(plan)[0]
        full_tree = (
            self.full_tree_profile(plan)
            if self.objective == "full_tree" else None
        )
        arity_histogram = {}
        for node, children in plan.children.items():
            if children:
                arity_histogram[len(children)] = (
                    arity_histogram.get(len(children), 0) + 1
                )
        tensor_cost = self._tensor_cost_key(plan)
        objective_key = self._objective_key(plan)
        return {
            "n_qubits": self.n,
            "n_interacting_pairs": n_pairs,
            "objective": self.objective,
            "order": self.order,
            "map_mode": (
                self.order
                if isinstance(self.order, str) and self.order.startswith("coarse-")
                else None
            ),
            "lattice_shape": self.lattice_shape,
            "coarse_grain": self.coarse_grain,
            "weight_mode": self.weight_mode,
            "time_decay": self.time_decay,
            "time_window": self.time_window,
            "active_events": int(sum(factor > 0.0 for factor in self.temporal_factors)),
            "hybrid_weights": (
                self.hybrid_weights if self.objective == "hybrid" else None
            ),
            "hybrid_cost": hybrid_cost,
            "objective_key": objective_key,
            "path_score": float(weighted_sum),
            "compression_score": (
                float(objective_key[0] + objective_key[1])
                if self.objective == "compression" else None
            ),
            "hypergraph_score": (
                {
                    "max_edge_load": float(max_load),
                    "total_edge_load": float(total_load),
                }
                if self.objective == "hypergraph" and loads is not None else None
            ),
            "full_tree": full_tree,
            "root": plan.root,
            "root_qubit": plan.root_qubit,
            "top_arity": plan.top_arity,
            "is_binary": plan.is_binary(),
            "is_strictly_binary": plan.is_strictly_binary(),
            "max_arity": plan.max_arity(),
            "max_tensor_rank": plan.max_tensor_rank(),
            "arity_histogram": arity_histogram,
            "score": float(weighted_sum),
            "max_path": int(max(dists)) if dists else 0,
            "mean_path": float(sum(dists) / n_pairs) if n_pairs else 0.0,
            "weighted_mean_path": (
                float(weighted_sum / total_weight) if total_weight else 0.0
            ),
            "balanced_score": float(balanced_score),
            "score_ratio_vs_balanced": (
                float(weighted_sum / balanced_score) if balanced_score else 0.0
            ),
            "edge_loads": loads,
            "total_edge_load": float(total_load) if total_load is not None else None,
            "max_edge_load": float(max_load) if max_load is not None else None,
            "peak_bond_growth": (
                _safe_exp2(max_load) if max_load is not None else None
            ),
            "balanced_max_edge_load": (
                float(balanced_max_load)
                if balanced_max_load is not None else None
            ),
            "balanced_total_edge_load": (
                float(balanced_total_load)
                if balanced_total_load is not None else None
            ),
            "balanced_peak_bond_growth": (
                _safe_exp2(balanced_max_load)
                if balanced_max_load is not None else None
            ),
            "peak_bond_growth_log2": (
                float(max_load) if max_load is not None else None
            ),
            "balanced_peak_bond_growth_log2": (
                float(balanced_max_load)
                if balanced_max_load is not None else None
            ),
            "rank_exact_events": int(rank_info.get("exact_events", 0)),
            "rank_bounded_events": int(rank_info.get("bounded_events", 0)),
            "rank_bound_reasons": dict(rank_info.get("reasons", {})),
            "max_virtual_degree": tensor_cost[2],
            "total_virtual_degree": tensor_cost[3],
            "estimated_max_tensor_log2": tensor_cost[0],
            "estimated_total_tensor_log2": tensor_cost[1],
            "selected_candidate": getattr(self, "_selected_candidate", "interaction"),
            "candidate_scores": getattr(self, "_last_candidate_scores", {}),
        }

    def _plot_gate_routes(
        self,
        plan=None,
        *,
        site_coords=None,
        ax=None,
        figsize=(10, 8),
        cmap="turbo",
        color_by="gate",
        scale_cmap="viridis",
        scale_markers=_DEFAULT_SCALE_MARKERS,
        lattice=True,
        show_gate_connectivity=True,
        show_gate_paths=False,
        show_node_ids=False,
        show_site_labels=False,
        show_event_labels=False,
        colorbar=False,
        show_axes=False,
        show_title=False,
        rubberband=False,
        node_size=58,
        event_linewidth=2.0,
        event_alpha=0.5,
        tree_edge_alpha=0.38,
        gate_path_curvature=0.08,
    ):
        """Plot a tree plan over the physical lattice and gate connectivity.

        By default this draws only the explicit TTN geometry over the optional
        physical background. Pass ``show_gate_paths=True`` to add gate-stream
        route overlays as a separate diagnostic layer; those routes are not
        tensor legs. Pass ``rubberband=True`` for the
        physical-lattice rubberband view, where each non-root tree cluster is
        wrapped by a rounded translucent band. In either view,
        ``color_by="scale"`` uses colors independent of gate-stream length;
        the explicit tree view also uses circle markers by default (custom
        marker cycles can be supplied with ``scale_markers``). When enabled,
        gate-path edges are kept visually distinct: structural edges are
        straight grey segments, while colored gate routes are offset by
        small deterministic arcs (controlled by ``gate_path_curvature``).
        No stream-order colorbar or title is shown by default. Pass
        ``site_coords={qubit: (x, y)}`` to place the physical leaves on an
        existing lattice; internal tree nodes are then placed above the
        supplied leaves. Without coordinates, leaves use their deterministic
        tree order and the plot becomes a clean rooted-tree view. The default
        presentation is axis-free, following quimb's schematic drawing style;
        set ``show_axes=True`` to retain Matplotlib axes.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes, ready for further customization or saving.
        """
        plt, colormaps, ScalarMappable, Normalize, FancyArrowPatch = (
            matplotlib_modules()
        )
        if plan is None:
            plan = self.run()
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan returned by run().")
        color_by = str(color_by).replace("-", "_").strip().lower()
        color_by = {"stream": "gate", "event": "gate", "level": "scale"}.get(
            color_by, color_by
        )
        if color_by not in {"gate", "scale"}:
            raise ValueError("color_by must be 'gate' or 'scale'.")
        try:
            scale_markers = tuple(scale_markers)
        except TypeError as exc:
            raise TypeError("scale_markers must be a non-empty sequence.") from exc
        if not scale_markers:
            raise ValueError("scale_markers must be a non-empty sequence.")
        if rubberband:
            return self.plot_rubberband(
                plan,
                site_coords=site_coords,
                ax=ax,
                figsize=figsize,
                cmap=cmap,
                color_by=color_by,
                scale_cmap=scale_cmap,
                lattice=lattice,
                show_gate_connectivity=show_gate_connectivity,
                show_site_nodes=True,
                colorbar=colorbar,
                show_axes=show_axes,
                show_title=show_title,
                band_alpha=event_alpha,
                band_linewidth=event_linewidth,
                node_size=node_size,
            )
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=figsize)
            if not show_axes:
                ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig = ax.figure

        qubits = tuple(range(plan.n))
        supplied_coords = site_coords is not None
        logical_coords = resolve_site_coords(qubits, site_coords)
        leaf_order = tuple(sorted(plan.qubit_of_leaf))
        leaf_position = {node: index for index, node in enumerate(leaf_order)}
        positions = {}

        if supplied_coords:
            for node, qubit in plan.qubit_of_leaf.items():
                positions[node] = logical_coords[qubit]
            if plan.root_qubit is not None:
                # The root physical site shares the root node, so it is shown
                # at the root's eventual position below.
                positions[plan.root] = logical_coords[plan.root_qubit]
        else:
            for node, index in leaf_position.items():
                positions[node] = (float(index), 0.0)

        def place_internal(node):
            if node in positions:
                return positions[node]
            child_points = [place_internal(child) for child in plan.children[node]]
            x = sum(point[0] for point in child_points) / len(child_points)
            y = max(point[1] for point in child_points) + 1.0
            positions[node] = (x, y)
            return positions[node]

        place_internal(plan.root)
        node_scales = _tree_node_scales(plan)
        n_scales = max(node_scales.values(), default=0) + 1
        # If a root qubit was supplied, its physical coordinate should not
        # flatten the structural root into the lattice. Keep the tree center
        # while still recording the logical root-site label at that node.
        if plan.root_qubit is not None and not supplied_coords:
            positions[plan.root] = (
                positions[plan.root][0],
                positions[plan.root][1],
            )

        if lattice:
            for left, right in coordinate_lattice_edges(logical_coords):
                x0, y0 = logical_coords[left]
                x1, y1 = logical_coords[right]
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color="#d5d9de",
                    linewidth=1.0,
                    alpha=0.78,
                    zorder=1,
                )

        if show_gate_connectivity:
            lattice_pairs = (
                coordinate_lattice_edge_keys(logical_coords)
                if lattice
                else set()
            )
            for support in self.supports:
                unique = tuple(dict.fromkeys(support))
                for left, right in zip(unique, unique[1:]):
                    if frozenset((left, right)) in lattice_pairs:
                        continue
                    x0, y0 = logical_coords[left]
                    x1, y1 = logical_coords[right]
                    ax.plot(
                        (x0, x1),
                        (y0, y1),
                        color="#7e8995",
                        linewidth=0.72,
                        linestyle="-",
                        alpha=0.62,
                        zorder=1,
                    )

        # Draw the rooted tree underneath the gate ribbons.
        for parent, children in plan.children.items():
            for child in children:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color="#aeb6bf",
                    linewidth=1.05,
                    alpha=tree_edge_alpha,
                    zorder=2,
                )

        internal = [node for node in plan.nodes() if not plan.is_leaf(node)]
        leaves = list(plan.leaves())
        if color_by == "scale":
            def draw_scale_nodes(nodes, size):
                for scale in sorted({node_scales[node] for node in nodes}):
                    scale_nodes = [
                        node for node in nodes if node_scales[node] == scale
                    ]
                    marker = scale_markers[scale % len(scale_markers)]
                    ax.scatter(
                        [positions[node][0] for node in scale_nodes],
                        [positions[node][1] for node in scale_nodes],
                        s=size,
                        marker=marker,
                        color=scale_color(
                            colormaps, scale_cmap, scale, n_scales
                        ),
                        edgecolors="#41464c",
                        linewidths=0.7,
                        zorder=5,
                    )

            draw_scale_nodes(internal, node_size * 0.82)
            draw_scale_nodes(leaves, node_size)
        else:
            if internal:
                ax.scatter(
                    [positions[node][0] for node in internal],
                    [positions[node][1] for node in internal],
                    s=node_size * 0.82,
                    color="#7b8188",
                    edgecolors="#41464c",
                    linewidths=0.7,
                    zorder=5,
                )
            if leaves:
                ax.scatter(
                    [positions[node][0] for node in leaves],
                    [positions[node][1] for node in leaves],
                    s=node_size,
                    c=[plan.qubit_of_leaf[node] for node in leaves],
                    cmap=colormaps.get_cmap(cmap),
                    vmin=0,
                    vmax=max(1, plan.n - 1),
                    edgecolors="#41464c",
                    linewidths=0.7,
                    zorder=5,
                )

        n_events = len(self.supports)
        if show_gate_paths:
            event_weights = tuple(self.event_weights)
            max_weight = max(event_weights, default=1.0)
            for event_index, (support, weight) in enumerate(
                zip(self.supports, event_weights)
            ):
                support = tuple(dict.fromkeys(support))
                event_color_value = event_color(
                    colormaps, cmap, event_index, n_events
                )
                width = event_linewidth * (
                    0.75
                    + 0.75 * (float(weight) / max(max_weight, 1.0)) ** 0.5
                )
                if gate_path_curvature:
                    side = 1.0 if event_index % 2 == 0 else -1.0
                    magnitude = 1.0 + float((event_index // 2) % 3)
                    route_curvature = (
                        side * float(gate_path_curvature) * magnitude
                    )
                else:
                    route_curvature = 0.0
                paths = []
                for left, right in zip(support, support[1:]):
                    path = plan.node_path(
                        plan.node_of_qubit[left], plan.node_of_qubit[right]
                    )
                    paths.append(path)
                segments = set()
                for path in paths:
                    for left, right in zip(path, path[1:]):
                        edge = (left, right) if left < right else (right, left)
                        if edge in segments:
                            continue
                        segments.add(edge)
                        x0, y0 = positions[left]
                        x1, y1 = positions[right]
                        if color_by == "scale":
                            segment_color = scale_color(
                                colormaps,
                                scale_cmap,
                                max(node_scales[left], node_scales[right]),
                                n_scales,
                            )
                        else:
                            segment_color = event_color_value
                        ax.add_patch(
                            FancyArrowPatch(
                                (x0, y0),
                                (x1, y1),
                                arrowstyle="-",
                                connectionstyle=(
                                    f"arc3,rad={route_curvature:.4g}"
                                ),
                                linewidth=width,
                                color=segment_color,
                                alpha=event_alpha,
                                zorder=3,
                            )
                        )
                if color_by == "gate":
                    for qubit in support:
                        x, y = positions[plan.node_of_qubit[qubit]]
                        ax.scatter(
                            [x], [y], s=node_size * 1.25,
                            color=[event_color_value], alpha=event_alpha,
                            edgecolors="white", linewidths=0.5, zorder=7,
                        )
                if show_event_labels and support:
                    node = plan.node_of_qubit[support[0]]
                    x, y = positions[node]
                    ax.text(
                        x,
                        y,
                        str(event_index),
                        color=(
                            event_color_value
                            if color_by == "gate"
                            else "#59636e"
                        ),
                        fontsize=8,
                        ha="center",
                        va="center",
                        zorder=8,
                    )

        if show_site_labels:
            for qubit in qubits:
                node = plan.node_of_qubit[qubit]
                x, y = positions[node]
                ax.annotate(
                    f"q{qubit}",
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#374151",
                    zorder=9,
                )
        if show_node_ids:
            for node in plan.nodes():
                x, y = positions[node]
                ax.annotate(
                    f"n{node}",
                    (x, y),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#5b6168",
                    zorder=9,
                )

        colorbar_count = (
            n_events if color_by == "gate" and show_gate_paths else n_scales
        )
        if colorbar and colorbar_count:
            add_order_colorbar(
                fig,
                ax,
                colormaps,
                ScalarMappable,
                Normalize,
                cmap if color_by == "gate" else scale_cmap,
                colorbar_count,
                label=(
                    "gate stream order"
                    if color_by == "gate"
                    else "tree scale (leaf = 0)"
                ),
            )
        title = (
            "Tree layout finder — "
            + ("colored gate paths" if color_by == "gate" else "scale-colored tree")
        )
        if show_axes:
            if show_title:
                ax.set_title(title)
            ax.set_xlabel("layout x")
            ax.set_ylabel("layout y")
            ax.margins(0.14)
            ax.set_aspect("equal", adjustable="datalim")
        else:
            finish_schematic_axes(
                ax,
                title=title if show_title else None,
                margins=0.14,
            )
        return fig, ax

    def plot_tent(
        self,
        plan=None,
        *,
        site_coords=None,
        ax=None,
        figsize=(8, 8),
        cmap="turbo",
        edge_cmap="GnBu",
        node_cmap="YlOrRd",
        color_by="order",
        edge_color=None,
        leaf_edge_color=None,
        show_edge_arrows=False,
        arrow_size=8.0,
        order=True,
        lattice=True,
        show_gate_connectivity=False,
        show_node_ids=False,
        show_site_labels=False,
        show_leaf_nodes=False,
        show_lattice_markers=True,
        lattice_marker="+",
        lattice_marker_size=100,
        lattice_marker_color="#737e89",
        lattice_marker_alpha=0.95,
        lattice_skew=0.30,
        lattice_rise=0.18,
        colorbar=False,
        show_axes=False,
        show_title=False,
        node_size=24,
        edge_linewidth=1.35,
        edge_alpha=0.8,
        vertical_spacing=0.8,
    ):
        """Plot the hierarchy as a Cotengra-style tent over the raw graph.

        Physical sites and gate connectivity stay in the lower, grey raw
        graph. Internal TTN nodes are lifted above the mean position of their
        descendant sites, and each parent-child hierarchy edge uses one
        uniform solid color by default. Pass ``edge_color=None`` to match
        each incoming edge to the node it terminates at (so ``node_cmap``
        controls both). The default has no arrows, matching Cotengra's
        structural tent view; pass
        ``show_edge_arrows=True`` only when parent-to-child direction is
        needed. This is deliberately a structural
        visualization:
        gate-by-gate route overlays are not drawn. Set ``order=True`` to place
        hierarchy nodes by a deterministic post-order traversal, matching the
        ordering option in Cotengra's tent plots. Use ``color_by="order"`` if
        the same traversal should also control the colors.
        Pass ``show_leaf_nodes=False`` when the physical lattice already has
        its own site markers (for example gray ``+`` symbols) and only the
        internal tree nodes should be drawn over that backdrop.
        By default, supplied two-dimensional coordinates are projected into a
        shallow tent base using ``x' = x + lattice_skew * y`` and
        ``y' = lattice_rise * y``. Set ``lattice_skew=0`` and
        ``lattice_rise=1`` to preserve the supplied coordinates.
        The default order/turbo palette and matching hierarchy edges are
        intended to give a compact Cotengra-style structural view.
        Pass ``leaf_edge_color`` to highlight the first hierarchy layer,
        namely edges connecting physical leaf sites to their parent nodes.
        When omitted, those edges follow the same child-node palette as the
        other hierarchy edges.
        """
        plt, colormaps, ScalarMappable, Normalize, _FancyArrowPatch = (
            matplotlib_modules()
        )
        if plan is None:
            plan = self.run()
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan returned by run().")
        color_by = str(color_by).replace("-", "_").strip().lower()
        color_by = {"level": "scale", "size": "scale"}.get(
            color_by, color_by
        )
        if color_by not in {"scale", "order"}:
            raise ValueError("color_by must be 'scale' or 'order'.")
        try:
            arrow_size = float(arrow_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "arrow_size must be a positive real number."
            ) from exc
        if not np.isfinite(arrow_size) or arrow_size <= 0.0:
            raise ValueError("arrow_size must be a positive real number.")
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=figsize)
            if not show_axes:
                ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig = ax.figure

        qubits = tuple(range(plan.n))
        raw_coords = resolve_site_coords(qubits, site_coords)
        try:
            lattice_skew = float(lattice_skew)
            lattice_rise = float(lattice_rise)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "lattice_skew and lattice_rise must be real numbers."
            ) from exc
        if not np.isfinite(lattice_skew) or not np.isfinite(lattice_rise):
            raise ValueError("lattice_skew and lattice_rise must be finite.")
        coords = {
            qubit: (
                x + lattice_skew * y,
                lattice_rise * y,
            )
            for qubit, (x, y) in raw_coords.items()
        }
        node_scales = _tree_node_scales(plan)
        n_scales = max(node_scales.values(), default=0) + 1

        if lattice:
            for left, right in coordinate_lattice_edges(raw_coords):
                ax.plot(
                    (coords[left][0], coords[right][0]),
                    (coords[left][1], coords[right][1]),
                    color="#b7c0c9",
                    linewidth=1.05,
                    alpha=0.82,
                    zorder=1,
                )
            if show_lattice_markers:
                ax.scatter(
                    [coords[qubit][0] for qubit in qubits],
                    [coords[qubit][1] for qubit in qubits],
                    marker=lattice_marker,
                    s=lattice_marker_size,
                    color=lattice_marker_color,
                    alpha=lattice_marker_alpha,
                    linewidths=1.25,
                    zorder=1.5,
                )

        if show_gate_connectivity:
            lattice_pairs = (
                coordinate_lattice_edge_keys(raw_coords)
                if lattice
                else set()
            )
            for support in self.supports:
                unique = tuple(dict.fromkeys(support))
                for left, right in zip(unique, unique[1:]):
                    if frozenset((left, right)) in lattice_pairs:
                        continue
                    ax.plot(
                        (coords[left][0], coords[right][0]),
                        (coords[left][1], coords[right][1]),
                        color="#7e8995",
                        linewidth=0.72,
                        linestyle="-",
                        alpha=0.62,
                        zorder=1,
                    )

        subtree_qubits = {}

        def gather_qubits(node):
            if node in subtree_qubits:
                return subtree_qubits[node]
            result = []
            if node in plan.qubit_of_leaf:
                result.append(plan.qubit_of_leaf[node])
            if node == plan.root and plan.root_qubit is not None:
                result.append(plan.root_qubit)
            for child in plan.children.get(node, ()):
                result.extend(gather_qubits(child))
            subtree_qubits[node] = tuple(result)
            return subtree_qubits[node]

        for node in plan.nodes():
            gather_qubits(node)

        x_span = max(
            max(point[0] for point in coords.values())
            - min(point[0] for point in coords.values()),
            1.0,
        )
        y_max = max(point[1] for point in coords.values())
        if vertical_spacing is None:
            # Keep the tent compact for square 2-D lattices. The previous
            # spacing made a 6x6 lattice grow into a very tall strip even
            # though ``figsize`` only changes the canvas, not the geometry.
            vertical_spacing = max(0.55, 0.16 * x_span)
        vertical_spacing = float(vertical_spacing)
        if vertical_spacing <= 0.0:
            raise ValueError("vertical_spacing must be positive.")

        positions = {
            node: coords[qubit]
            for node, qubit in plan.qubit_of_leaf.items()
        }
        if plan.root_qubit is not None:
            positions[plan.root] = coords[plan.root_qubit]

        internal = [node for node in plan.nodes() if not plan.is_leaf(node)]
        if order or color_by == "order":
            postorder = []

            def visit(node):
                for child in plan.children.get(node, ()):
                    visit(child)
                postorder.append(node)

            visit(plan.root)
            order_values = {node: i for i, node in enumerate(postorder)}
            order_count = max(1, len(postorder))
            internal_order = {
                node: index
                for index, node in enumerate(
                    node for node in postorder if node in internal
                )
            }
            order_span = max(1, n_scales - 1)
            order_denominator = max(1, len(internal) - 1)
            for node in internal:
                sites = gather_qubits(node)
                x = sum(coords[qubit][0] for qubit in sites) / len(sites)
                if order:
                    # Preserve post-order relationships without giving every
                    # internal node a separate vertical layer. A separate
                    # layer for all nodes makes larger 2D circuits needlessly
                    # tall and narrow without conveying extra geometry.
                    height = 1.0 + order_span * (
                        internal_order[node] / order_denominator
                    )
                else:
                    height = 1.0 + order_values[node] / order_count
                y = y_max + vertical_spacing * height
                positions[node] = (x, y)
            n_colors = order_count if color_by == "order" else n_scales
        else:
            order_values = None
            for node in internal:
                sites = gather_qubits(node)
                x = sum(coords[qubit][0] for qubit in sites) / len(sites)
                y = y_max + vertical_spacing * (node_scales[node] + 1.0)
                positions[node] = (x, y)
            n_colors = n_scales

        def node_color(node):
            if color_by == "order":
                return event_color(
                    colormaps, cmap, order_values[node], n_colors
                )
            return scale_color(
                colormaps, node_cmap, node_scales[node], n_colors
            )

        def hierarchy_edge_color(parent, child):
            if edge_color is not None:
                return edge_color
            if leaf_edge_color is not None and plan.is_leaf(child):
                return leaf_edge_color
            # ``None`` means "follow the node palette": this is intentionally
            # the node color itself rather than a separate edge colormap, so
            # an incoming edge and its child are visually identical.
            return node_color(child)

        for parent, children in plan.children.items():
            for child in children:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                # The incoming edge is colored like the node it terminates at,
                # making each scale/order layer visually self-consistent.
                edge_color_value = hierarchy_edge_color(parent, child)
                ax.plot(
                    (x0, x1),
                    (y0, y1),
                    color=edge_color_value,
                    linewidth=edge_linewidth,
                    alpha=edge_alpha,
                    zorder=2,
                )
                if show_edge_arrows:
                    dx = x1 - x0
                    dy = y1 - y0
                    ax.add_patch(
                        _FancyArrowPatch(
                            (x0 + 0.42 * dx, y0 + 0.42 * dy),
                            (x0 + 0.62 * dx, y0 + 0.62 * dy),
                            arrowstyle="-|>",
                            mutation_scale=arrow_size,
                            linewidth=max(0.6, 0.75 * edge_linewidth),
                            color=edge_color_value,
                            shrinkA=0.0,
                            shrinkB=0.0,
                            zorder=3,
                        )
                    )

        for node in plan.nodes():
            if plan.is_leaf(node) and not show_leaf_nodes:
                continue
            x, y = positions[node]
            ax.scatter(
                [x],
                [y],
                s=node_size,
                marker="o",
                color=[node_color(node)],
                edgecolors="#41464c",
                linewidths=0.65,
                zorder=4,
            )

        if show_site_labels:
            for qubit in qubits:
                node = plan.node_of_qubit[qubit]
                x, y = positions[node]
                ax.annotate(
                    f"q{qubit}",
                    (x, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#374151",
                    zorder=5,
                )
        if show_node_ids:
            for node in plan.nodes():
                x, y = positions[node]
                ax.annotate(
                    f"n{node}",
                    (x, y),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#5b6168",
                    zorder=5,
                )

        if colorbar and n_colors:
            add_order_colorbar(
                fig,
                ax,
                colormaps,
                ScalarMappable,
                Normalize,
                cmap if color_by == "order" else node_cmap,
                n_colors,
                label=(
                    "tree order" if color_by == "order" else "tree scale"
                ),
            )

        title = "Tree tent"
        if show_axes:
            if show_title:
                ax.set_title(title)
            ax.set_xlabel("layout x")
            ax.set_ylabel("hierarchy height")
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.14)
        else:
            finish_schematic_axes(
                ax,
                title=title if show_title else None,
                margins=0.14,
            )
        return fig, ax

    # The public default is the structural tent view. Keep the older direct
    # route renderer private so the hierarchy cannot be mistaken for a set of
    # gate-stream legs.
    plot = plot_tent

    def plot_rubberband(
        self,
        plan=None,
        *,
        site_coords=None,
        ax=None,
        figsize=(10, 8),
        cmap="Spectral",
        color_by="gate",
        scale_cmap="viridis",
        lattice=True,
        show_gate_connectivity=True,
        show_site_nodes=True,
        colorbar=False,
        show_axes=False,
        show_title=False,
        band_alpha=0.68,
        band_linewidth=1.35,
        band_padding=0.12,
        node_size=58,
    ):
        """Plot hierarchical tree clusters as smooth rubberband regions.

        This is the physical-lattice counterpart to Quimb's contraction-tree
        ``plot_rubberband`` view: the lattice and gate connectivity remain
        grey, while each non-root tree cluster is wrapped by a rounded,
        translucent colored band. The default ``color_by="gate"`` uses a
        ``Spectral`` post-order progression, matching Cotengra's many-color
        rubberband view. ``color_by="scale"`` is available when one stable
        color is wanted for each tree scale measured from the leaves.

        The default presentation has no axes, site labels, or title. It
        returns a normal Matplotlib ``(fig, ax)`` pair for further styling.
        """
        plt, colormaps, ScalarMappable, Normalize, _FancyArrowPatch = (
            matplotlib_modules()
        )
        from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415

        if plan is None:
            plan = self.run()
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan returned by run().")
        color_by = str(color_by).replace("-", "_").strip().lower()
        color_by = {"stream": "gate", "event": "gate", "level": "scale"}.get(
            color_by, color_by
        )
        if color_by not in {"gate", "scale"}:
            raise ValueError("color_by must be 'gate' or 'scale'.")
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=figsize)
            if not show_axes:
                ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig = ax.figure

        qubits = tuple(range(plan.n))
        coords = resolve_site_coords(qubits, site_coords)
        node_scales = _tree_node_scales(plan)
        n_scales = max(node_scales.values(), default=0) + 1

        if lattice:
            for left, right in coordinate_lattice_edges(coords):
                ax.plot(
                    (coords[left][0], coords[right][0]),
                    (coords[left][1], coords[right][1]),
                    color="#c7cdd3",
                    linewidth=1.0,
                    alpha=0.62,
                    zorder=1,
                )

        if show_gate_connectivity:
            lattice_pairs = (
                coordinate_lattice_edge_keys(coords)
                if lattice
                else set()
            )
            for support in self.supports:
                unique = tuple(dict.fromkeys(support))
                for left, right in zip(unique, unique[1:]):
                    if frozenset((left, right)) in lattice_pairs:
                        continue
                    ax.plot(
                        (coords[left][0], coords[right][0]),
                        (coords[left][1], coords[right][1]),
                        color="#87919b",
                        linewidth=0.7,
                        linestyle="-",
                        alpha=0.42,
                        zorder=1,
                    )

        subtree_qubits = {}

        def gather_qubits(node):
            if node in subtree_qubits:
                return subtree_qubits[node]
            children = tuple(plan.children.get(node, ()))
            if not children:
                result = (plan.qubit_of_leaf[node],)
            else:
                result = tuple(
                    qubit
                    for child in children
                    for qubit in gather_qubits(child)
                )
                if node == plan.root and plan.root_qubit is not None:
                    result += (plan.root_qubit,)
            subtree_qubits[node] = result
            return result

        band_nodes = []

        def visit(node):
            for child in plan.children.get(node, ()):
                visit(child)
            if plan.children.get(node):
                band_nodes.append(node)

        visit(plan.root)
        n_bands = max(1, len(band_nodes))
        for band_index, node in enumerate(band_nodes):
            sites = tuple(dict.fromkeys(gather_qubits(node)))
            if len(sites) < 2:
                continue
            points = [coords[qubit] for qubit in sites]
            xmin = min(point[0] for point in points)
            xmax = max(point[0] for point in points)
            ymin = min(point[1] for point in points)
            ymax = max(point[1] for point in points)
            padding = band_padding + 0.012 * band_index
            width = max(xmax - xmin, 0.16) + 2.0 * padding
            height = max(ymax - ymin, 0.16) + 2.0 * padding
            rounding = min(0.28, 0.45 * min(width, height))
            if color_by == "scale":
                color = scale_color(
                    colormaps,
                    scale_cmap,
                    node_scales[node],
                    n_scales,
                )
            else:
                color = event_color(colormaps, cmap, band_index, n_bands)
            ax.add_patch(
                FancyBboxPatch(
                    (xmin - padding, ymin - padding),
                    width,
                    height,
                    boxstyle=f"round,pad=0,rounding_size={rounding}",
                    fill=False,
                    edgecolor=color,
                    linewidth=band_linewidth,
                    alpha=band_alpha,
                    # Draw inner/earlier contractions above outer/later
                    # bands, as in Cotengra, so overlapping bands remain
                    # individually legible.
                    zorder=3.0 + (n_bands - band_index) / n_bands,
                )
            )

        if show_site_nodes:
            ax.scatter(
                [coords[qubit][0] for qubit in qubits],
                [coords[qubit][1] for qubit in qubits],
                s=node_size,
                marker="o",
                color="#858b91",
                edgecolors="#3f454b",
                linewidths=0.75,
                zorder=5,
            )

        if colorbar and (n_bands if color_by == "gate" else n_scales):
            add_order_colorbar(
                fig,
                ax,
                colormaps,
                ScalarMappable,
                Normalize,
                cmap if color_by == "gate" else scale_cmap,
                n_bands if color_by == "gate" else n_scales,
                label=(
                    "rubberband order"
                    if color_by == "gate"
                    else "tree scale (leaf = 0)"
                ),
            )

        title = "Tree rubberband"
        if show_axes:
            if show_title:
                ax.set_title(title)
            ax.set_xlabel("logical site x")
            ax.set_ylabel("logical site y")
            ax.set_aspect("equal", adjustable="datalim")
            ax.margins(0.14)
        else:
            finish_schematic_axes(
                ax,
                title=title if show_title else None,
                margins=0.14,
            )
        return fig, ax

    plot_layout = plot
