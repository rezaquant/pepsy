"""Tree-tensor-network gate-stream simulator centered on :class:`TreeOptimizer`.

:class:`TreeOptimizer` implements the rooted tree-tensor-network circuit
simulator of *Simulating quantum circuits using tree tensor networks*
(Seitz, Medina, Cruz, Huang, Mendl; Quantum 7, 964, 2023; arXiv:2206.01000).

A quantum state is stored as a rooted tree tensor network whose leaves carry
physical qubit indices. One optional physical qubit may instead live on the
root tensor. Internal nodes may have any arity: the default structure is a
binary tree below a ternary virtual root, but flatter ``k``-ary trees
(``max_arity``) or gate-connectivity-driven communities
(``structure="adaptive"``) are supported unchanged.  A bundled gate stream
``[(gate, where), ...]`` is replayed:

* single-qubit gates are absorbed into their physical-site tensor (no bond growth); a
  unitary one-qubit gate preserves the tree canonical form regardless of where
  the orthogonality centre sits;
* ordinary gate-stream entries are lowered to a true :class:`TreeMPO`, whose
  operator bonds are routed losslessly through the active canonical Steiner
  region before the affected state bonds are compressed; the operator and state
  are never lowered to a chain MPO for this path;
* the explicit low-level two-qubit compatibility method retains the Seitz et
  al. two-factor SVD/QR path kernel: both gate factors are installed before a
  single canonical compression sweep truncates the touched bonds.

The orthogonality centre is tracked as a node id and moved *smartly* along the
tree geodesic with per-edge canonicalisation, mirroring the
``info_c["cur_orthog"]`` centre tracking of :class:`pepsy.MpsOptimizer`.  The
tree structure is chosen by :class:`TreeLayoutFinder` (entanglement-adapted
recursive partition) unless an explicit :class:`TreePlan` is supplied.

This is the tensor-network glue only: the heavy lifting (arbitrary-geometry
canonicalisation, bond compression, tensor splitting, tree path finding) uses
``quimb`` primitives.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
from collections.abc import Mapping
import heapq
import inspect
from numbers import Integral
import time
import warnings

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..._internal.cutoff import dtype_auto_cutoff
from ...fitting import TreeFIT
from ...fitting.tree import (
    _build_layered_operator_state_target,
    _layered_target_bond_sizes,
    _randomize_tree_guess,
)
from ...backends import (
    backend_infer,
    backend_signatures_compatible,
    infer_backend_converter_from_sample,
    infer_backend_signature,
    to_float,
)
from ...operators.gates import _normalize_gate_entries
from .._fidelity import (
    fidelity_from_log,
    infidelity_from_log,
    log_fidelity_from_norms,
)
from ..mps.optimizer import (
    _control_event_parts as _mps_control_event_parts,
    _resolve_conditional,
    normalize_submpo_where,
    submpo_event_parts,
)
from .layout import (
    _DEFAULT_TOP_ARITY,
    TreeLayoutFinder,
    TreePlan,
    _normalize_time_decay,
    _normalize_time_window,
    _submpo_schmidt_rank_bound,
)
from .ttn import (
    TreeTensorNetwork,
    _contract_two_tensors,
    _normalize_compression_mode,
)

__all__ = ["TreeOptimizer"]

try:  # threadpoolctl is a NumPy/SciPy transitive dependency; treat as optional.
    from threadpoolctl import ThreadpoolController as _ThreadpoolController

    # Built once (it scans the loaded BLAS/OpenMP libraries): reused per gate so
    # ``.limit(...)`` is a cheap set/restore rather than a fresh library scan.
    _THREAD_CONTROLLER = _ThreadpoolController()
except Exception:  # pragma: no cover - threadpoolctl missing
    _THREAD_CONTROLLER = None


def _normalize_where(where):
    """Return a tuple of int qubit labels for a gate support."""
    if isinstance(where, Integral):
        return (int(where),)
    return tuple(int(site) for site in where)


_TREE_MPO_EVENT_NAMES = frozenset({
    "subtreempo",
    "sub_treempo",
    "sub_tree_mpo",
    "subttno",
    "sub_ttno",
})


def _tree_mpo_event_parts(entry):
    """Return ``(TreeMPO, declared_support)`` for a TreeMPO event.

    TreeMPO events are deliberately a Tree-only stream extension.  The
    payload carries its ``TreePlan`` and the default support is every physical
    site of that plan; unlike an MPS sub-MPO marker, no chain interval is
    inferred or inserted.
    """
    if isinstance(entry, Mapping):
        kind = str(entry.get("kind", entry.get("type", ""))).strip().lower()
        if kind not in _TREE_MPO_EVENT_NAMES:
            return None
        payload = entry.get("treempo")
        if payload is None:
            payload = entry.get("tree_mpo")
        if payload is None:
            payload = entry.get("ttno", entry.get("operator"))
        where = entry.get("where")
    elif isinstance(entry, (tuple, list)) and entry:
        head = entry[0]
        if not isinstance(head, str) or head.strip().lower() not in _TREE_MPO_EVENT_NAMES:
            return None
        if len(entry) < 2 or len(entry) > 3:
            raise ValueError(
                "TreeMPO events must be ('subtreempo', operator[, where])."
            )
        payload = entry[1]
        where = entry[2] if len(entry) == 3 else None
    else:
        return None

    plan = getattr(payload, "plan", None)
    if plan is None or not hasattr(payload, "tree_networks"):
        raise TypeError(
            "TreeMPO events require a TreeMPO/TTNO payload with a TreePlan."
        )
    if where is None:
        where = tuple(sorted(plan.node_of_qubit))
    return payload, _normalize_where(where)


def _submpo_to_dense(submpo, where):
    """Materialize an explicit sub-MPO on its declared support.

    TTNs do not have a 1D sub-MPO absorption primitive. Their recursive
    subtree-operator path is the equivalent operation, so stream markers are
    lowered to the same dense support operator used by ``apply_gate``. The
    support-size guard is checked by the caller before this conversion.
    """
    to_dense = getattr(submpo, "to_dense", None)
    if not callable(to_dense):
        raise TypeError(
            "TreeOptimizer submpo events require an MPO-like payload with "
            "a to_dense() method."
        )
    try:
        dense = to_dense()
    except Exception as exc:
        raise ValueError("could not materialize the submpo event payload.") from exc
    expected = 4 ** len(where)
    size = int(np.prod(ar.shape(dense)))
    if size != expected:
        raise ValueError(
            f"submpo payload has {size} entries, but support {where!r} "
            f"requires {expected}."
        )
    return ar.do("reshape", dense, (2 ** len(where), 2 ** len(where)))


def _same_tree_plan(left, right):
    """Return whether two plans describe the same rooted tree geometry."""
    return (
        isinstance(left, TreePlan)
        and isinstance(right, TreePlan)
        and left.root == right.root
        and left.children == right.children
        and left.qubit_of_leaf == right.qubit_of_leaf
        and left.root_qubit == right.root_qubit
    )


def _is_product_tensor_network(state):
    """Return whether every virtual bond of ``state`` has dimension one."""
    max_bond = getattr(state, "max_bond", None)
    if not callable(max_bond):
        return False
    try:
        value = max_bond()
        # Quimb reports ``None`` for the vacuous bond maximum of a one-site
        # MPS. That state is still a valid product-state initializer for a
        # TreeOptimizer.
        if value is None:
            return int(getattr(state, "L", 0)) <= 1
        return int(value) <= 1
    except (TypeError, ValueError):
        return False


def _is_symmray_array(array):
    """Return whether ``array`` is a Symmray block-sparse / fermionic array.

    Symmray gate payloads (e.g. native Fermi-Hubbard hopping/onsite gates) are
    block-sparse arrays whose physical legs already carry the correct symmetry
    charges and, for fermions, the anticommutation signs. They must never be
    reshaped into base-2 sub-legs the way a dense qubit gate is; the raw
    symmetric array is handed straight to the Symmray-aware Quimb split /
    contract primitives.
    """
    try:
        return ar.infer_backend(array) == "symmray"
    except Exception:  # pragma: no cover - defensive backend inference
        return False


def _submpo_is_native(submpo):
    """Return whether an MPO visibly contains native Symmray tensors.

    An explicit ``pepsy_tree_native`` marker is honored for lightweight
    payloads, otherwise ordinary Quimb MPOs are inspected from their tensor
    payload. ``None`` means that the payload does not expose enough
    information to classify it without materialising it.
    """
    marker = getattr(submpo, "pepsy_tree_native", None)
    if marker is not None:
        return bool(marker)
    tensors = getattr(submpo, "tensors", None)
    if tensors is None:
        return None
    try:
        return any(_is_symmray_array(tensor.data) for tensor in tensors)
    except (AttributeError, TypeError):
        return None


def _array_backend_signature(array):
    """Return comparable backend / dtype / device metadata for an array."""
    return infer_backend_signature(array)


def _operator_schmidt_rank(op, where, left_where):
    """Return the operator-Schmidt rank across a support bipartition.

    ``where`` fixes the operator's qubit ordering.  The returned rank is the
    exact rank of the operator viewed as a matrix from the input/output legs on
    ``left_where`` to those on the complementary support.  For a two-qubit
    gate this is the ``k`` introduced by the gate SVD in the paper; for a
    higher-order operator it is the corresponding conservative generalisation.
    """
    where = tuple(where)
    left_where = tuple(left_where)
    left = {q: i for i, q in enumerate(where)}
    left_set = set(left_where)
    left_positions = [left[q] for q in left_where]
    right_positions = [
        i for i, q in enumerate(where) if q not in left_set
    ]
    k = len(where)
    try:
        array = ar.to_numpy(op).reshape((2,) * (2 * k))
    except Exception as exc:
        # Symmray arrays intentionally cannot be flattened through the generic
        # autoray ``to_numpy`` path: their blocks carry charge sectors and, for
        # fermions, graded signs.  Preflight only needs a safe upper bound, so
        # use the explicit operator leg dimensions without materialising the
        # structured array.  This is exact for the usual dense qubit shape and
        # conservative for a native/sparse backend.
        if not _is_symmray_array(op):
            raise ValueError(
                f"operator on support {where!r} must contain "
                f"{4 ** k} entries."
            ) from exc
        try:
            shape = tuple(int(dim) for dim in ar.shape(op))
        except (AttributeError, TypeError, ValueError) as shape_exc:
            raise ValueError(
                f"operator on support {where!r} must contain "
                f"{4 ** k} entries."
            ) from shape_exc
        if len(shape) == 2 and shape == (2 ** k, 2 ** k):
            shape = (2,) * (2 * k)
        if len(shape) != 2 * k or any(dim < 1 for dim in shape):
            raise ValueError(
                f"operator on support {where!r} must have {2 * k} "
                f"operator legs; got shape {shape}."
            ) from exc
        left_dim = int(np.prod([
            shape[i] * shape[k + i] for i in left_positions
        ], dtype=int))
        right_dim = int(np.prod([
            shape[i] * shape[k + i] for i in right_positions
        ], dtype=int))
        return max(1, min(left_dim, right_dim))

    axes = (
        left_positions
        + [k + i for i in left_positions]
        + right_positions
        + [k + i for i in right_positions]
    )
    matrix = array.transpose(axes).reshape(
        4 ** len(left_positions), 4 ** len(right_positions)
    )
    # A zero operator has Schmidt rank zero mathematically, but it cannot
    # reduce a pre-existing bond.  Treat it as rank one for this upper bound.
    return max(1, int(np.linalg.matrix_rank(matrix)))


_PAULI_1Q = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_RESET_FLIP_AXES = {"X": "Z", "Y": "X", "Z": "X"}
_DEFAULT_CUTOFF = "auto"
_DEFAULT_CUTOFF_MODE = "auto"
_DEFAULT_MAX_OPERATOR_QUBITS = 8
_DEFAULT_MAX_SUBTREE_NODES = 128


def _normalize_control_where(where):
    """Return a non-empty tuple of integer qubit labels for a control."""
    if isinstance(where, Integral):
        return (int(where),)
    if (
        isinstance(where, (tuple, list))
        and where
        and all(isinstance(site, Integral) for site in where)
    ):
        return tuple(int(site) for site in where)
    raise ValueError(
        "Tree control event where must be an int or non-empty sequence of ints."
    )


def _normalize_control_axes(axes, where, *, event):
    """Return one X/Y/Z axis per control-event site."""
    axes = tuple(char for char in str(axes).upper() if not char.isspace())
    if not axes or any(axis not in _PAULI_1Q for axis in axes):
        raise ValueError(f"{event} basis must use only X, Y, or Z axes.")
    if len(axes) == 1 and len(where) > 1:
        axes = axes * len(where)
    if len(axes) != len(where):
        raise ValueError(
            f"{event} basis has {len(axes)} axis/axes but where {where!r} "
            f"has {len(where)} site(s)."
        )
    return axes


def _normalize_measure_axes(pauli, where):
    """Return one X/Y/Z axis per measured site."""
    axes = tuple(char for char in str(pauli).upper() if not char.isspace())
    if not axes or any(axis not in _PAULI_1Q for axis in axes):
        raise ValueError("measure pauli must use only X, Y, or Z axes.")
    if len(axes) != len(where):
        raise ValueError(
            f"measure pauli {pauli!r} has {len(axes)} axis/axes but where "
            f"{where!r} has {len(where)} site(s)."
        )
    return axes


class TreeOptimizer:
    """Replay a bundled gate stream on a rooted tree tensor network.

    The constructor defaults are performance-oriented: ordinary gate entries
    are converted to a TreeMPO and routed through ``apply_subtreempo`` on
    their active canonical Steiner region, small tree contractions are capped
    to one BLAS/OpenMP thread (``threads=1``), and full singular-spectrum
    diagnostics are disabled
    (``track_truncation=False``). Enable those diagnostics explicitly when
    collecting truncation reports; the one-time warning in that case is
    intentional because it describes the additional SVD work.

    Parameters
    ----------
    gates : bundled gate stream, optional
        ``[(gate, where), ...]`` entries with ``where`` an int or a sequence
        of distinct qubits. Replayed eagerly on construction when given.
    n : int, optional
        Number of qubits.  Inferred from ``gates`` / ``tree`` when omitted.
    chi : int | None
        Maximum virtual bond dimension enforced during two-qubit threading.
        ``None`` leaves the bond uncapped; the singular-value ``cutoff`` still
        applies.
    cutoff : float | {"auto"}
        Singular-value cutoff for truncations, interpreted according to
        ``cutoff_mode``. ``"auto"`` selects a dtype-aware value: ``1e-6``
        for 32-bit data and ``1e-12`` for 64-bit data.
    cutoff_mode : str | None | {"auto"}
        Quimb singular-value cutoff mode. ``"auto"`` (and the compatibility
        spelling ``None``) selects Pepsy's relative discarded-squared-weight
        convention, ``"rsum2"``. Use ``"rel"`` for a relative
        largest-singular-value threshold.
    mode : {"auto", "direct", "dm", "sdc", "src", "zipup", "dmrg", "dmrg1", "dmrg2", "dmrg3", "tree_mpo_direct", "tree_mpo_dm", "mpo", "submpo"}
        Gate/operator route and state-compression method. Ordinary gate
        entries in ``"auto"``, ``"direct"``, ``"dm"``, ``"sdc"``,
        ``"src"``, and ``"mpo"`` are converted to a true :class:`TreeMPO`
        and applied with :meth:`apply_subtreempo` on the active Steiner
        subtree; the compression spelling selects the configured tree
        decomposition. ``"tree_mpo_direct"`` and ``"tree_mpo_dm"`` retain
        explicit route names and select direct SVD or density-matrix
        compression. ``"submpo"`` declares that the stream is already made
        of explicit chain-MPO entries. Explicit sub-MPO entries are accepted
        in the other legacy modes for compatibility.
        ``"sdc"`` and ``"src"`` are shorthands for automatic routing with
        deterministic or randomized successive tree-edge compression.
        ``"zipup"`` contracts one operator/state node with incoming child
        messages, then immediately truncates its outgoing message by SVD.
        Its intermediate cuts do not have a canonical right environment.
        ``"dmrg"`` and ``"dmrg1"``/``"dmrg2"``/``"dmrg3"`` select the
        tree-native :class:`pepsy.fitting.TreeFIT` engine. ``dmrg1`` and
        ``dmrg2`` use two-node warm-up blocks, ``dmrg3`` uses three-node
        warm-up blocks, and all named modes refine with one-node sweeps.
        Hyphenated spellings such as ``"tree-mpo-dm"`` are accepted, as are
        ``"tree_mpo_dem"`` and ``"tree_mpo"`` compatibility spellings.
    compression_mode : {"direct", "dm", "sdc", "src"}
        Decomposition used by ordinary tree compression and TreeFIT local
        splits. ``"direct"`` uses SVD; ``"dm"`` uses Quimb's
        density-matrix-equivalent ``svd:eig`` decomposition on the local
        canonical compression core; ``"sdc"`` uses the deterministic
        successive tree sweep and ``"src"`` uses randomized SVD per dense
        tree edge. These modes do not build a global dense state.
    structure : {"quality", "balanced", "adaptive"}
        Tree-structure strategy used when ``tree`` is not supplied.
    max_arity : int, None, or iterable of ints
        Maximum children per internal node for the auto-built structure.  A
        scalar builds one fixed tree (``2`` = binary; larger values or ``None``
        = flatter / wider). The default is the fixed binary tree and its
        ternary virtual root; an iterable can still be supplied explicitly to
        search candidate arities. Ignored when an explicit ``tree`` is
        supplied.
    top_arity : int or None, optional
        Number of virtual child bonds on the structural root when the layout
        is built automatically. Omitted with ``max_arity=2`` selects
        ``top_arity=3`` when possible, giving the conventional binary TTN
        with a three-leg top tensor. Pass ``top_arity=None`` or ``2`` to use a
        binary root. This keeps every tensor rank at most three.
    layout_objective : {"path", "congestion", "compression", "hypergraph", "full_tree", "hybrid"}
        Objective used when building an automatic tree.  ``"path"`` is the
        backward-compatible interaction-path heuristic; ``"congestion"``
        selects a candidate using predicted operator-Schmidt edge load;
        ``"compression"`` additionally penalizes peak/total load and the
        estimated local tensor cost at ``chi``; ``"hypergraph"`` directly
        scores every original multi-qubit support across every crossed tree
        edge and enables bounded leaf/NNI refinement by default;
        ``"full_tree"`` evaluates dynamic all-scale bond pressure, overflow,
        tensor width, work, write, and route costs;
        ``"hybrid"`` combines normalized path, peak-load, and total-load
        costs. Pass a configured :class:`TreeLayoutFinder` through ``layout=``
        to customize its hybrid weights or enable pre-simulation refinement.
    layout_weight_mode : {"count", "auto", "angle", "operator_schmidt"}
        Event weighting used by the automatic layout interaction graph.
    layout_time_decay : float, optional
        Optional newest-event decay passed to :class:`TreeLayoutFinder`.
        Values are in ``(0, 1]``; omitted means no temporal weighting.
    layout_time_window : int, optional
        Optional trailing gate-event window passed to the layout finder.
    layout : TreeLayoutFinder or TreePlan, optional
        A precomputed layout finder or its resulting plan.  This is an alias
        layer over ``tree=`` and is useful when the finder also provides
        arity/cost diagnostics.
    tree : TreePlan, optional
        Explicit tree structure (any arity).  When omitted a
        :class:`TreeLayoutFinder` builds one from the gate stream.
    root_qubit : int, optional
        Designated qubit carried by the top tensor instead of a leaf when the
        layout is built automatically. Gates, sub-MPOs, readout, capping, and
        layout scoring treat it as an ordinary physical site at the root. When
        ``tree`` or ``layout`` is supplied, this must match its
        ``TreePlan.root_qubit``.
    dtype : numpy dtype
        Data type of the initial product state (default ``complex128``).
    threads : int or None
        BLAS/OpenMP thread cap applied around gate application and the heavy
        contraction read-outs.  Tree tensors are small (rank ``<= 3``, bounded
        by ``chi``), so multi-threaded linear algebra is dominated by thread
        launch/synchronisation overhead: capping to ``1`` (the default) makes
        replay both markedly faster and stable in wall-clock time.  Pass
        ``None`` to leave the ambient thread count untouched (worthwhile only in
        a large-``chi`` regime where a single contraction is itself large).
    seed : int or None
        Seed for the internal random generator used by :meth:`measure` and
        :meth:`reset`.
    track_truncation : bool
        Whether to probe the full local singular spectrum before each
        truncating split/compression and record discarded-weight diagnostics.
        The extra spectrum probes are disabled by default.
    track_infidelity : bool
        Whether to record the cheap canonical-centre norm ledger and include
        its norm-based progress readout. This is enabled by default for
        compatibility with direct TreeOptimizer use, but can be disabled for
        non-unitary transfer-operator streams where norm changes are physical.
    compression_seed : int, optional
        Seed forwarded to randomized tree-edge compression when
        ``compression_mode='src'``. It is ignored by deterministic modes.
    fit_block_size : {1, 2, 3}, default=2
        Generic ``dmrg`` local block size. Named aliases select their own
        warm-up size; ``dmrg1`` uses two-node growth before one-node DMRG.
    fit_n_iter : int, default=2
        Maximum TreeFIT iterations per fitted gate window. Each iteration
        includes an inward and an outward pass in the configured order.
    fit_min_iter : int or None, default=2
        Minimum iterations before tolerance stopping; ``None`` permits
        stopping as soon as two comparable norm samples are available.
    fit_rtol : {"auto"}, float, or None, default="auto"
        Relative retained-center-norm stopping tolerance: ``1e-3`` for 16-bit
        data, ``1e-5`` for float32/complex64, and ``1e-9`` otherwise, matching
        MpsOptimizer. ``None`` disables tolerance stopping. Automatic stopping
        is disabled for non-unitary replay or ``track_norm=False`` updates.
    fit_patience : int, default=1
        Number of consecutive stable norm comparisons required to stop.
    fit_sweep_sequence : str, default="inward-outward"
        Order of the two passes: ``"inward-outward"`` or ``"outward-inward"``.
        The legacy ``"RL"``/``"LR"`` and ``"INOUT"``/``"OUTIN"`` aliases
        remain accepted with identical traversal order.
    fit_finite_check : bool, default=False
        Opt-in finite-value checks of active TreeFIT tensors after each sweep.
        Routine fitting uses the terminal canonical-centre norm for convergence
        and does not scan tensor entries or revalidate every tree isometry.
    fit_adaptive_sweeps : int, default=2
        Number of larger-block warm-up sweeps for generic ``dmrg``. Named
        ``dmrg1`` uses two warm-up sweeps, matching MpsOptimizer.
    max_intermediate_bond : int, optional
        Conservative preflight limit for the untruncated crossing-bond bound.
        When set, eager replay raises :class:`MemoryError` before tensor work if
        :meth:`estimate_bonds` predicts a larger intermediate bond.
    max_operator_qubits : int, optional
        Maximum support size allowed for dense multi-qubit operators.  The
        default is conservative protection against accidental ``4**k``
        allocations; pass ``None`` to disable it. Product-Pauli measurements
        use a factorized path and do not consume this dense budget.
    max_subtree_nodes : int, optional
        Maximum Steiner-subtree size allowed for multi-qubit application and
        preflight.  The default limits the number of recursive local messages;
        pass ``None`` to disable it.
    profile : bool
        Whether to collect opt-in kernel timing records in
        :meth:`profile_report`. Profiling is disabled by default and adds no
        synchronization or timing calls to the normal replay path.
    profile_sync : bool
        Whether profiled phase boundaries should synchronize the active device
        before taking timestamps. This is useful for asynchronous CuPy and
        CUDA backends, but adds a synchronization at every recorded phase and
        is therefore disabled by default.
    track_bond_diagnostics : bool
        Whether to record live and transient bond dimensions for each update.
        This is disabled by default because determining the live maximum scans
        the tree after QR hops. When enabled, :meth:`bond_diagnostic_report`
        distinguishes temporary QR/gate growth from the post-compression live
        state bonds.
    tn : TreeTensorNetwork or product MatrixProductState, optional
        Initial coefficient state. A tree state is copied and canonicalised if
        needed. Its plan must match ``tree``/``layout`` when it is entangled;
        a bond-dimension-one product TTN is instead remounted exactly on the
        requested plan (with a warning). A bond-dimension-one Quimb MPS is also
        accepted and mounted exactly. When omitted, the optimizer starts from
        the product state ``|0...0>``.
    state : TreeTensorNetwork or product MatrixProductState, optional
        Explicit alias for ``tn``. All initial-state tensors must share one
        backend, dtype, and device. User-provided gates/operators should use
        that same backend; stream payload mismatches raise before replay and
        must be prepared explicitly.
    run : bool
        Whether to replay ``gates`` immediately (default ``True``).

    Attributes
    ----------
    tn : TreeTensorNetwork
        The live tree tensor network (a geometry-owning ``quimb`` subclass).
    plan : TreePlan
        The tree structure.
    """

    @staticmethod
    def _normalize_mode(mode):
        """Validate and normalize the gate or sub-MPO replay mode."""
        mode = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "dem": "dm",
            "tree_mpo": "tree_mpo_direct",
            "treempo": "tree_mpo_direct",
            "treempo_direct": "tree_mpo_direct",
            "treempo_dm": "tree_mpo_dm",
            "treempo_dem": "tree_mpo_dm",
            "tree_mpo_dem": "tree_mpo_dm",
            "tree_mpo_svd": "tree_mpo_direct",
            "tree_mpo_eig": "tree_mpo_dm",
        }
        mode = aliases.get(mode, mode)
        if mode == "fit":
            mode = "dmrg"
        if mode in {"dmrg", "dmrg1", "dmrg2", "dmrg3"}:
            return mode
        if mode not in {
            "auto", "direct", "dm", "sdc", "src", "zipup", "mpo", "submpo",
            "tree_mpo_direct", "tree_mpo_dm",
        }:
            raise ValueError(
                "mode must be one of 'auto', 'direct', 'dm', 'sdc', 'src', 'zipup', 'mpo', "
                "'submpo', 'dmrg', 'dmrg1', 'dmrg2', 'dmrg3', "
                "'tree_mpo_direct', or 'tree_mpo_dm'."
            )
        return mode

    _PROGBAR_COLORS = {
        # Keep the two primary colors aligned with MpsOptimizer. The tree
        # implementation has additional compression spellings, but they are
        # still either a DMRG/FIT or MPO-style replay.
        "dmrg": "#1f77b4",
        "mpo": "#2ca02c",
    }

    def _progress_mode_name(self):
        """Return the active short mode name shown by a replay bar."""

        mode = str(self.mode).strip().lower().replace("-", "_")
        if mode == "dmrg":
            return self._dmrg_mode_alias or "dmrg"
        if mode in {"dmrg1", "dmrg2", "dmrg3"}:
            return mode
        if mode == "tree_mpo_dm":
            return "dm"
        if mode in {"auto", "direct", "mpo", "submpo", "tree_mpo_direct"}:
            compression_mode = str(self.compression_mode).strip().lower()
            if compression_mode in {"direct", "dm", "sdc", "src"}:
                return compression_mode
            return "direct"
        return mode

    def _gate_route(self, width):
        """Return the implementation route for an ordinary dense gate.

        Ordinary TreeOptimizer gates use the geometry-aware TreeMPO kernel.
        Keeping this decision here means construction-time and run-time mode
        overrides share one gate-to-TreeMPO path. Explicit ``submpo`` remains
        reserved for chain-MPO stream events.
        """
        if self.mode == "submpo":
            return "submpo"
        return "treempo"

    def _compression_for_mode(self, mode, compression_mode):
        """Resolve a mode's optional compression suffix.

        The combined TreeMPO names own their compression method. A conflicting
        explicit ``compression_mode`` is rejected rather than silently
        changing the meaning of a mode name.
        """
        if mode not in {"zipup", "tree_mpo_direct", "tree_mpo_dm"}:
            return compression_mode
        expected = "dm" if mode == "tree_mpo_dm" else "direct"
        if compression_mode not in {expected, "direct"}:
            raise ValueError(
                f"mode={mode!r} requires compression_mode={expected!r}."
            )
        return expected

    @staticmethod
    def _normalize_compression_mode(mode):
        """Validate the decomposition used for tree-state truncation."""

        return _normalize_compression_mode(mode)

    @staticmethod
    def _normalize_max_bond(max_bond):
        """Validate an optional per-update bond cap."""
        if max_bond is None:
            return None
        if isinstance(max_bond, bool):
            raise TypeError("max_bond must be a positive integer or None.")
        max_bond = int(max_bond)
        if max_bond < 1:
            raise ValueError("max_bond must be a positive integer or None.")
        return max_bond

    def _resolve_cutoff(self, value):
        """Return a validated truncation cutoff, including ``"auto"``."""
        if value == "auto":
            return dtype_auto_cutoff(self.backend_dtype)
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cutoff must be 'auto' or a non-negative number."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("cutoff must be 'auto' or a non-negative number.")
        return value

    @staticmethod
    def _resolve_cutoff_mode(value):
        """Resolve ``cutoff_mode='auto'`` to Pepsy's default convention."""
        if value is None or (
            isinstance(value, str) and value.strip().lower() == "auto"
        ):
            return "rsum2"
        return value

    def _resolve_fit_rtol(self, value):
        """Resolve the same dtype-aware stopping tolerance as MpsOptimizer."""
        if value == "auto":
            dtype = str(self.backend_dtype).lower()
            if "16" in dtype:
                return 1e-3
            if "32" in dtype or "complex64" in dtype:
                return 1e-5
            return 1e-9
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fit_rtol must be 'auto', a non-negative number, or None."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "fit_rtol must be 'auto', a non-negative number, or None."
            )
        return value

    def __init__(self, gates=None, n=None, *, chi=64,
                 cutoff=_DEFAULT_CUTOFF,
                 cutoff_mode=_DEFAULT_CUTOFF_MODE, mode="auto",
                 compression_mode="direct",
                 compression_seed=None,
                 fit_block_size=2, fit_n_iter=2, fit_adaptive_sweeps=2,
                 fit_min_iter=2, fit_rtol="auto", fit_patience=1,
                 fit_init_strategy="guess-src",
                 fit_init_rand_strength=0.0, fit_init_seed=0,
                 fit_sweep_sequence="inward-outward", fit_overlap_diagnostics=False,
                 fit_finite_check=False,
                 two_site_mode=None,
                 structure="quality", max_arity=2,
                 top_arity=_DEFAULT_TOP_ARITY,
                 community_frac=0.35,
                 star_frac=0.75, layout_objective="congestion",
                 layout_weight_mode="count", layout_time_decay=None,
                 layout_time_window=None, layout=None, tree=None,
                 map_mode=None,
                 root_qubit=None,
                 dtype=complex, threads=1, subtree_workers=1, seed=None,
                 run=True, tn=None,
                 state=None, track_truncation=False, track_infidelity=True,
                 max_intermediate_bond=None,
                 max_operator_qubits=_DEFAULT_MAX_OPERATOR_QUBITS,
                 max_subtree_nodes=_DEFAULT_MAX_SUBTREE_NODES,
                 record_history=True, profile=False, profile_sync=False,
                 track_bond_diagnostics=False):
        # Preserve one-shot streams for both queue normalization and automatic
        # layout discovery.  Materializing only inside
        # ``_normalize_gate_queue`` would leave the finder with an exhausted
        # iterator and silently degrade to an interaction-free layout.
        if hasattr(gates, "__next__"):
            gates = list(gates)
        if map_mode is not None and (layout is not None or tree is not None):
            raise TypeError(
                "map_mode cannot be combined with an explicit tree or layout"
            )
        layout_top_arity = (
            layout.top_arity if isinstance(layout, TreeLayoutFinder) else None
        )
        if layout is not None:
            if tree is not None:
                raise ValueError("pass either layout= or tree=, not both.")
            if isinstance(layout, TreeLayoutFinder):
                if n is not None and int(n) != layout.n:
                    raise ValueError("n does not match the supplied layout.")
                n = layout.n if n is None else n
                tree = layout.run()
            elif isinstance(layout, TreePlan):
                tree = layout
            else:
                raise TypeError(
                    "layout must be a TreeLayoutFinder or TreePlan; "
                    "pass an entangled TreeTensorNetwork as state= or tn=."
                )
        if state is not None:
            if tn is not None:
                raise ValueError("pass either state= or tn=, not both.")
            tn = state
        if isinstance(tree, TreeTensorNetwork):
            raise TypeError(
                "tree= expects a TreePlan, not a TreeTensorNetwork; "
                "pass the entangled state as state= or tn=."
            )
        if root_qubit is not None:
            try:
                root_qubit = int(root_qubit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "root_qubit must be an integer or None."
                ) from exc
        self.G, self.where, self.event_types = self._normalize_gate_queue(gates)
        self.layout_finder = layout if isinstance(layout, TreeLayoutFinder) else None

        product_state_source = None
        if tn is not None:
            if isinstance(tn, TreeTensorNetwork):
                tn.validate()
                source_n = tn.nqubits
                if tree is not None and tree is not tn.plan:
                    if _same_tree_plan(tree, tn.plan):
                        tree = tn.plan
                    elif _is_product_tensor_network(tn):
                        product_state_source = tn
                        warnings.warn(
                            "Requested tree/layout differs from a product "
                            "TreeTensorNetwork; rebuilding the product state "
                            "exactly on the selected tree plan.",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        raise ValueError(
                            "tree/layout geometry differs from the supplied "
                            "entangled TreeTensorNetwork. Refusing a potentially "
                            "lossy relayout; construct the state on the selected "
                            "TreePlan or explicitly convert it first."
                        )
                else:
                    tree = tn.plan
            elif isinstance(tn, qtn.MatrixProductState):
                if not _is_product_tensor_network(tn):
                    raise TypeError(
                        "TreeOptimizer accepts an MPS initial state only when "
                        "max_bond() == 1. Convert an entangled MPS explicitly "
                        "to a TreeTensorNetwork on the requested TreePlan."
                    )
                source_n = int(tn.nsites)
                sites = tuple(tn.sites)
                if tuple(sorted(sites)) != tuple(range(source_n)):
                    raise ValueError(
                        "Product MPS sites must be the compact labels 0..n-1 "
                        "to initialize TreeOptimizer."
                    )
                product_state_source = tn
            else:
                raise TypeError(
                    "tn/state must be a TreeTensorNetwork, or a product "
                    "quimb MatrixProductState with max_bond() == 1."
                )
            if n is not None and int(n) != source_n:
                raise ValueError("n does not match the supplied initial state.")
            n = source_n

        if n is None:
            if tree is not None:
                n = tree.n
            else:
                n = 1 + max(
                    (max(w) for w in self.where if len(w) > 0),
                    default=-1,
                )
                if root_qubit is not None:
                    n = max(n, root_qubit + 1)
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("Could not infer qubit count; pass n explicitly.")
        if root_qubit is not None and not 0 <= root_qubit < self.n:
            raise ValueError(
                f"root_qubit {root_qubit!r} is outside 0..{self.n - 1}."
            )
        if top_arity is _DEFAULT_TOP_ARITY:
            if layout_top_arity is not None:
                top_arity = layout_top_arity if layout_top_arity >= 2 else None
            elif isinstance(tree, TreePlan):
                tree_top_arity = tree.top_arity
                top_arity = tree_top_arity if tree_top_arity >= 2 else None
            elif (
                root_qubit is None
                and isinstance(max_arity, Integral)
                and int(max_arity) == 2
                and self.n >= 3
            ):
                top_arity = 3
            else:
                top_arity = None
        elif top_arity is None and layout_top_arity is not None:
            top_arity = layout_top_arity
        if top_arity is not None:
            try:
                top_arity = int(top_arity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "top_arity must be an integer >= 2 or None."
                ) from exc
            if top_arity < 2:
                raise ValueError("top_arity must be >= 2 or None.")
            if root_qubit is not None and top_arity != 2:
                raise ValueError(
                    "top_arity > 2 cannot be combined with root_qubit: "
                    "the root would have a rank-four tensor."
                )
        # The TTN itself always uses compact physical positions.  This facade
        # optionally preserves caller-facing logical labels across a cap while
        # keeping Quimb's internal site/index space contiguous.
        self._logical_qubits = list(range(self.n))
        self._logical_positions = {q: q for q in self._logical_qubits}

        compression_mode = self._normalize_compression_mode(compression_mode)
        raw_mode = self._normalize_mode(mode)
        if raw_mode in {"dm", "sdc", "src"}:
            if compression_mode not in {"direct", raw_mode}:
                raise ValueError(
                    f"mode={raw_mode!r} cannot be combined with a "
                    "different compression_mode."
                )
            compression_mode = raw_mode
            raw_mode = "auto"
        compression_mode = self._compression_for_mode(
            raw_mode, compression_mode
        )

        self.chi = self._normalize_max_bond(chi)
        self.cutoff = cutoff
        self.cutoff_mode = cutoff_mode
        self.mode = self._normalize_mode(raw_mode)
        self._dmrg_mode_alias = (
            self.mode if self.mode in {"dmrg1", "dmrg2", "dmrg3"} else None
        )
        if self.mode in {"dmrg1", "dmrg2", "dmrg3"}:
            self.mode = "dmrg"
        self.compression_mode = compression_mode
        if not isinstance(fit_block_size, Integral) or int(fit_block_size) not in {1, 2, 3}:
            raise ValueError("fit_block_size must be 1, 2, or 3.")
        self.fit_block_size = int(fit_block_size)
        if not isinstance(fit_n_iter, Integral) or int(fit_n_iter) < 1:
            raise ValueError("fit_n_iter must be a positive integer.")
        self.fit_n_iter = int(fit_n_iter)
        if (
            not isinstance(fit_adaptive_sweeps, Integral)
            or int(fit_adaptive_sweeps) < 1
        ):
            raise ValueError("fit_adaptive_sweeps must be a positive integer.")
        self.fit_adaptive_sweeps = int(fit_adaptive_sweeps)
        if fit_min_iter is not None and (
            isinstance(fit_min_iter, bool)
            or not isinstance(fit_min_iter, Integral) or fit_min_iter < 1
        ):
            raise ValueError("fit_min_iter must be a positive integer or None.")
        self.fit_min_iter = fit_min_iter
        self._fit_rtol_requested = fit_rtol
        if not isinstance(fit_patience, Integral) or int(fit_patience) < 1:
            raise ValueError("fit_patience must be a positive integer.")
        self.fit_patience = int(fit_patience)
        self.fit_init_strategy = str(fit_init_strategy).strip().lower().replace("-", "_")
        self.fit_init_rand_strength = float(fit_init_rand_strength)
        if (
            not np.isfinite(self.fit_init_rand_strength)
            or self.fit_init_rand_strength < 0.0
        ):
            raise ValueError(
                "fit_init_rand_strength must be finite and non-negative."
            )
        if isinstance(fit_init_seed, bool) or not isinstance(fit_init_seed, Integral):
            raise TypeError("fit_init_seed must be an integer.")
        self.fit_init_seed = int(fit_init_seed)
        self.fit_sweep_sequence = TreeFIT._normalize_sweep_sequence(fit_sweep_sequence)
        self.fit_overlap_diagnostics = bool(fit_overlap_diagnostics)
        self.fit_finite_check = bool(fit_finite_check)
        self.fit_diagnostics = []
        self._last_fit_diagnostics = None
        if compression_seed is not None:
            if isinstance(compression_seed, bool) or not isinstance(
                compression_seed, Integral
            ):
                raise TypeError("compression_seed must be an integer or None.")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError("compression_seed must be non-negative.")
        self.compression_seed = compression_seed
        if two_site_mode is not None:
            legacy_mode = self._normalize_mode(two_site_mode)
            if self.mode != "auto" and self.mode != legacy_mode:
                raise ValueError(
                    "pass either mode= or two_site_mode=, or give them "
                    "the same value."
                )
            warnings.warn(
                "two_site_mode= is deprecated; use mode= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.mode = legacy_mode
        self.structure = structure
        # ``max_arity`` may be a scalar (fixed tree), ``None`` (unbounded), or an
        # iterable of candidate arities to search; forward it to the finder,
        # which normalizes and (for a candidate set) searches it chi-aware.
        self.max_arity = max_arity
        self.top_arity = top_arity
        self.community_frac = float(community_frac)
        self.star_frac = float(star_frac)
        self.layout_objective = str(layout_objective)
        self.layout_weight_mode = str(layout_weight_mode)
        self.layout_time_decay = _normalize_time_decay(layout_time_decay)
        self.layout_time_window = _normalize_time_window(layout_time_window)
        self.dtype = dtype
        self.threads = None if threads is None else int(threads)
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be positive or None.")
        self.subtree_workers = self._positive_limit(
            subtree_workers, "subtree_workers"
        )
        self.rng = np.random.default_rng(seed)
        self.track_truncation = bool(track_truncation)
        self._track_warning_emitted = False
        self.track_infidelity = bool(track_infidelity)
        self.max_intermediate_bond = self._positive_limit(
            max_intermediate_bond, "max_intermediate_bond"
        )
        self.max_operator_qubits = self._positive_limit(
            max_operator_qubits, "max_operator_qubits"
        )
        self.max_subtree_nodes = self._positive_limit(
            max_subtree_nodes, "max_subtree_nodes"
        )
        self.record_history = bool(record_history)
        self.profile = bool(profile)
        self.profile_sync = bool(profile_sync)
        self.track_bond_diagnostics = bool(track_bond_diagnostics)
        self.profile_events = []
        self.measurements = []
        self.truncation_history = []
        self.update_history = []
        self.bond_history = []
        # ``infidelities`` remains the historical spectrum-based Tree trace.
        # The cheap canonical-centre norm ledger lives separately in
        # ``norm_events`` so enabling/disabling ``track_truncation`` cannot
        # change the meaning of the norm-based compression metric.
        self.infidelities = [0.0]
        self.infidelity_samples = []
        self.norm_events = []
        self.normalizations = []
        self.projection_diagnostics = []
        self._backend_conversion_warnings = set()
        # Gate payloads are treated as immutable during replay. Keeping a
        # small identity-keyed cache avoids repeating the same operator SVD or
        # MPO factorization for repeated circuit gates without hashing/copying
        # large backend arrays. Entries retain the payload object so Python id
        # reuse cannot return a stale factorization.
        self._gate_factor_cache = {}
        self._gate_factor_cache_limit = 64
        # Gate supports recur frequently in circuit streams. Cache only the
        # immutable geometry path; centre-dependent source orientation is
        # selected on every update below.
        self._two_site_path_cache = {}
        self._two_site_path_cache_limit = 256
        self._active_update = None
        self._update_counter = 0
        self._truncation_log_survival = 0.0
        self._norm_log_survival = 0.0
        self._norm_tracking_enabled = True

        if tree is None:
            self.layout_finder = TreeLayoutFinder(
                gates=self._layout_gate_stream(), n=self.n, structure=structure,
                max_arity=self.max_arity, community_frac=self.community_frac,
                star_frac=self.star_frac,
                top_arity=self.top_arity,
                objective=self.layout_objective,
                weight_mode=self.layout_weight_mode,
                time_decay=self.layout_time_decay,
                time_window=self.layout_time_window,
                chi=self.chi,
                max_operator_qubits=self.max_operator_qubits,
                root_qubit=root_qubit,
            )
            tree = self.layout_finder.run()
        if not isinstance(tree, TreePlan):
            raise TypeError("tree must be a TreePlan or None.")
        if tree.n != self.n:
            raise ValueError(
                f"tree contains {tree.n} qubits, but n={self.n} was requested."
            )
        if root_qubit is not None and tree.root_qubit != root_qubit:
            raise ValueError(
                "root_qubit does not match the supplied tree/layout plan."
            )
        if self.top_arity is not None and tree.top_arity != int(self.top_arity):
            raise ValueError(
                "top_arity does not match the supplied tree/layout plan."
            )
        if self.top_arity is None and tree.top_arity >= 2:
            # Preserve the root convention when an explicit plan is handed in,
            # so later candidate searches and plots keep the same topology.
            self.top_arity = tree.top_arity
        self.plan = tree

        if product_state_source is not None:
            self.tn = self._remount_product_state(product_state_source)
            self.center = self.plan.root
        elif tn is None:
            self.tn = self._build_product_state()
            # A freshly built product state has every virtual bond at dimension
            # 1, so it is already canonical at the root.
            self.center = self.plan.root
        else:
            self._install_tn(tn)
        self._attach_profile_sink()
        self._thread_ind = None
        self.backend_info()
        self._validate_gate_stream_backend(
            self.G,
            self.event_types,
            target_signature=_array_backend_signature(self._state_like()),
        )
        self.cutoff = self._resolve_cutoff(cutoff)
        self.cutoff_mode = self._resolve_cutoff_mode(cutoff_mode)
        self.fit_rtol = self._resolve_fit_rtol(fit_rtol)

        if run and self.G:
            if (
                self.max_intermediate_bond is not None
                or self.max_operator_qubits is not None
                or self.max_subtree_nodes is not None
            ):
                self.preflight(
                    max_bond=self.max_intermediate_bond,
                    max_operator_qubits=self.max_operator_qubits,
                    max_subtree_nodes=self.max_subtree_nodes,
                )
            self.run()

    # -- stream normalization -------------------------------------------------

    @staticmethod
    def _normalize_gate_queue(gates):
        if gates is None:
            return [], [], []

        # The low-level normalizer deliberately distinguishes a bundled
        # sequence from a single gate.  Materialize one-shot iterators here so
        # generator-based gate streams behave like the documented bundled
        # sequence and remain available for automatic layout discovery.
        if hasattr(gates, "__next__"):
            gates = list(gates)

        tree_mpo_parts = _tree_mpo_event_parts(gates)
        if tree_mpo_parts is not None:
            tree_mpo, where = tree_mpo_parts
            return [tree_mpo], [where], ["subtreempo"]

        submpo_parts = submpo_event_parts(gates)
        if submpo_parts is not None:
            submpo, where = submpo_parts
            return [submpo], [normalize_submpo_where(where)], ["submpo"]

        control_parts = _mps_control_event_parts(gates)
        if control_parts is not None:
            name, payload, where = control_parts
            return [payload], [where], [name]

        if isinstance(gates, (tuple, list)) and any(
            _tree_mpo_event_parts(entry) is not None
            or submpo_event_parts(entry) is not None
            or _mps_control_event_parts(entry) is not None
            for entry in gates
        ):
            payloads = []
            wheres = []
            event_types = []
            for entry in gates:
                tree_mpo_parts = _tree_mpo_event_parts(entry)
                if tree_mpo_parts is not None:
                    tree_mpo, where = tree_mpo_parts
                    payloads.append(tree_mpo)
                    wheres.append(where)
                    event_types.append("subtreempo")
                    continue
                submpo_parts = submpo_event_parts(entry)
                if submpo_parts is not None:
                    submpo, where = submpo_parts
                    payloads.append(submpo)
                    wheres.append(normalize_submpo_where(where))
                    event_types.append("submpo")
                    continue
                control_parts = _mps_control_event_parts(entry)
                if control_parts is not None:
                    name, payload, where = control_parts
                    payloads.append(payload)
                    wheres.append(where)
                    event_types.append(name)
                    continue
                gate_entries = _normalize_gate_entries(
                    (entry,), where=None, allow_empty=False
                )
                gate, where = gate_entries[0]
                payloads.append(gate)
                wheres.append(_normalize_where(where))
                event_types.append("gate")
            return payloads, wheres, event_types

        entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
        payloads = [g for g, _ in entries]
        wheres = [_normalize_where(w) for _, w in entries]
        return payloads, wheres, ["gate"] * len(payloads)

    def _layout_gate_stream(self):
        """Return a layout stream remapped through preceding cap events.

        The live TTN compacts physical positions after a cap, while callers may
        keep either compact or stable logical labels. Automatic layout is built
        once before replay, so it must translate every later support back to
        the original leaf labels before constructing the initial tree.
        """
        active_labels = list(range(self.n))
        original_labels = list(range(self.n))
        stream = []
        for payload, where, event_type in zip(
            self.G, self.where, self.event_types
        ):
            logical_where = _normalize_where(where)
            try:
                original_where = tuple(
                    original_labels[active_labels.index(q)]
                    for q in logical_where
                )
            except ValueError as exc:
                raise ValueError(
                    "cannot build a tree layout: event support references "
                    f"inactive labels {logical_where!r} after a cap."
                ) from exc

            if event_type == "gate":
                stream.append((payload, original_where))
            elif event_type == "subtreempo":
                stream.append(self.subtreempo_event(payload, original_where))
            elif event_type == "submpo":
                stream.append(self.submpo_event(payload, original_where))
            elif event_type == "measure":
                stream.append({
                    "kind": "measure",
                    "pauli": payload["pauli"],
                    "where": original_where,
                    "outcome": payload.get("outcome"),
                })
            elif event_type == "reset":
                stream.append({
                    "kind": "reset",
                    "where": original_where,
                    "basis": "".join(payload["axes"]),
                })
            elif event_type == "measure_reset":
                stream.append({
                    "kind": "measure_reset",
                    "where": original_where,
                    "basis": "".join(payload["axes"]),
                    "outcome": payload.get("outcomes"),
                })
            elif event_type == "cap":
                stream.append({
                    "kind": "cap",
                    "where": original_where[0],
                    "vec": payload["vec"],
                    "absorb": payload.get("absorb", "left"),
                    "compact_labels": payload.get("compact_labels", True),
                })
            elif event_type == "conditional":
                # Layout discovery only needs the action support. Use a
                # lightweight placeholder mapping on the original labels;
                # the live conditional action remains in ``self.G`` and is
                # replayed unchanged after the tree plan is selected. Avoid
                # materializing a dense identity for a wide conditional gate.
                stream.append({
                    "kind": "conditional",
                    "record": payload["record"],
                    "bit": payload["bit"],
                    "action": {"where": original_where},
                })
            else:  # pragma: no cover - normalized streams are exhaustive
                raise ValueError(f"unknown tree layout event {event_type!r}.")

            if event_type == "cap":
                capped = logical_where[0]
                position = active_labels.index(capped)
                active_labels.pop(position)
                original_labels.pop(position)
                if payload.get("compact_labels", True):
                    active_labels = [
                        label - 1 if label > capped else label
                        for label in active_labels
                    ]
        return stream

    def _validate_event_stream_for_run(self):
        """Validate stream labels, including optional stable-label caps."""
        active = list(self._logical_qubits)
        for step, (payload, where, event_type) in enumerate(
            zip(self.G, self.where, self.event_types), start=1
        ):
            support = _normalize_where(where)
            if event_type == "cap":
                if len(support) != 1:
                    raise ValueError(
                        f"cap event at step {step} must reference one qubit."
                    )
                q = support[0]
                if len(active) <= 1:
                    raise ValueError(
                        f"cap event at step {step} cannot remove the only qubit."
                    )
                if q not in active:
                    raise ValueError(
                        f"cap event at step {step} references qubit {q}, "
                        f"outside the current active labels {active!r}."
                    )
                active.remove(q)
                if payload.get("compact_labels", True):
                    active = [label - 1 if label > q else label for label in active]
                continue
            out_of_range = [q for q in support if q not in active]
            if out_of_range:
                raise ValueError(
                    f"event at step {step} references qubit(s) outside the "
                    f"current active labels {active!r}: {out_of_range!r}."
                )

    def _validate_mode_for_stream(self):
        """Validate mode-specific stream declarations before replay."""
        if not self.G:
            return
        if self.mode in {"tree_mpo_direct", "tree_mpo_dm"}:
            chain_events = [
                step for step, event_type in enumerate(self.event_types, 1)
                if event_type == "submpo"
            ]
            if chain_events:
                raise ValueError(
                    f"mode={self.mode!r} requires TreeMPO/TTNO events; "
                    "chain sub-MPO event(s) found at step(s) "
                    f"{chain_events!r}. Use mode='submpo' for chain MPOs or "
                    "TreeOptimizer.subtreempo_event(...) for TreeMPOs."
                )
            return
        if self.mode != "submpo":
            return
        # ``output_replay='submpo'`` still represents singleton supports as
        # ordinary one-site gates. They do not introduce a competing
        # multi-site lowering path and are therefore valid in this mode.
        ordinary = []
        for step, (where, event_type) in enumerate(
            zip(self.where, self.event_types), start=1
        ):
            if event_type != "gate":
                continue
            width = len(_normalize_where(where))
            if width > 1:
                ordinary.append((step, width))
        if ordinary:
            raise ValueError(
                "mode='submpo' requires explicit sub-MPO events for "
                "multi-site operations; ordinary multi-site gate event(s) "
                f"found at step/width {ordinary!r}. "
                "Use mode='direct', mode='dm', or a tree_mpo_* mode for "
                "dense gate streams."
            )
        if "submpo" not in self.event_types:
            raise ValueError(
                "mode='submpo' requires at least one explicit sub-MPO event."
            )

    # -- construction ---------------------------------------------------------

    @staticmethod
    def _positive_limit(value, name):
        """Validate an optional positive integer resource limit."""
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer or None.") from exc
        if value < 1:
            raise ValueError(f"{name} must be a positive integer or None.")
        return value

    @staticmethod
    def _format_progress_infidelity(value):
        """Format a norm-based progress infidelity compactly."""
        text = f"{float(value):#.0e}"
        if "e" not in text:
            return text
        mantissa, exponent = text.split("e", 1)
        sign = exponent[0] if exponent[:1] in "+-" else ""
        digits = exponent[1:] if sign else exponent
        digits = digits.lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"

    @staticmethod
    def _format_progress_scalar(value):
        """Format a fidelity value with the MPS progress-bar precision."""

        return f"{float(value):.6f}"

    def _cumulative_fidelity(self):
        """Return the cumulative retained-norm fidelity for display."""

        return float(fidelity_from_log(self._norm_log_survival))

    def _phys(self, q):
        return self.tn.site_ind(q)

    def _state_like(self):
        """Return a live state array for backend-coercing operator payloads."""
        return self.tn.node_tensor(self.plan.root).data

    def _invalidate_state_norm_cache(self):
        """Invalidate state-owned readout caches before direct tensor updates."""
        invalidate = getattr(self.tn, "_invalidate_norm_cache", None)
        if callable(invalidate):
            invalidate()

    @staticmethod
    def _state_backend_info(state):
        """Validate and describe the common backend of every state tensor."""
        return backend_infer(state)

    def backend_info(self):
        """Return the common backend, dtype, and device of the live TTN."""
        info = self._state_backend_info(self.tn)
        # Keep a state-derived public diagnostic in addition to the detailed
        # ``backend_info`` mapping.  It is refreshed on every query so direct
        # caller mutations cannot leave a stale optimizer backend label.
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        return info

    def _warn_backend_conversion(self, source_signature, target_signature):
        """Warn once for one explicit source/target backend conversion."""
        warning_key = (source_signature, target_signature)
        if (
            source_signature[0] != "builtins"
            and warning_key not in self._backend_conversion_warnings
        ):
            self._backend_conversion_warnings.add(warning_key)
            warnings.warn(
                "TreeOptimizer is converting a gate/operator payload from "
                f"backend/dtype/device {source_signature!r} to the TTN state "
                f"{target_signature!r}. Provide backend-compatible gate "
                "arrays to avoid this transfer or cast.",
                UserWarning,
                stacklevel=3,
            )

    def _warn_track_truncation_slow(self):
        """Warn once that complete-spectrum diagnostics add SVD work."""
        if self.mode == "zipup":
            # Zipup records bond sizes, not canonical discarded spectra.
            return
        if self.track_truncation and not self._track_warning_emitted:
            warnings.warn(
                "TreeOptimizer track_truncation=True enables complete "
                "singular-spectrum probes for edges that may truncate. "
                "Lossless zero-cutoff edges remain QR-only, but this "
                "diagnostic mode can substantially slow replay. Use "
                "track_truncation=False for performance runs.",
                UserWarning,
                stacklevel=3,
            )
            self._track_warning_emitted = True

    @staticmethod
    def _backend_converter(like):
        """Build one converter for a stream targeting ``like``."""
        converter = infer_backend_converter_from_sample(like)
        if converter is not None:
            return converter
        return lambda array: ar.do("array", array, like=like)

    def _as_state_backend(self, array, *, warn=True):
        """Return an operator payload compatible with the live TTN backend.

        Matching arrays pass through unchanged. A mismatched gate is converted
        for compatibility, but emits one warning per source/target signature so
        callers can keep data transfer and dtype promotion under their control.
        """
        like = self._state_like()
        state_info = self.backend_info()
        target_signature = _array_backend_signature(like)
        source_signature = _array_backend_signature(array)
        if source_signature == target_signature:
            return array
        # Python sequences/scalars are ordinary convenience inputs rather than
        # a selected numerical backend. Materialize those silently; explicit
        # array backends/dtypes still receive the transfer/cast warning.
        if warn:
            self._warn_backend_conversion(source_signature, target_signature)
        if state_info["backend"] == "symmray" and source_signature[0] != "symmray":
            raise TypeError(
                "Cannot convert a dense gate/operator payload into a native "
                "Symmray TTN without charge and fermionic metadata. Build the "
                "payload as a Symmray array on the target U1/U1U1 backend."
            )
        if state_info["backend"] == "numpy":
            return ar.to_numpy(array)
        return self._backend_converter(like)(array)

    def to_backend(self, array):
        """Prepare one user array or operator network on the live backend."""
        if hasattr(array, "tree_networks"):
            return self._coerce_tree_mpo_backend(array)
        if hasattr(array, "tensors") or hasattr(array, "tensor_map"):
            return self._coerce_tensor_network_backend(
                array.copy(), warn=False
            )
        return self._as_state_backend(array, warn=False)

    def _validate_gate_stream_backend(
        self, payloads, event_types, *, path_prefix="stream", paths=None,
        state=None, target_signature=None
    ):
        """Require every user gate/MPO payload to match the live TTN backend.

        This is a stream-boundary check. Replay receives the original payloads
        unchanged; internal operators created by the optimizer may still use
        ``_as_state_backend`` explicitly when they are constructed.
        """
        if not payloads:
            return
        if len(payloads) != len(event_types):
            raise ValueError(
                "TreeOptimizer backend validation requires payloads and event "
                "types to have the same length."
            )
        if paths is not None and len(paths) != len(payloads):
            raise ValueError(
                "TreeOptimizer backend validation requires one path per payload."
            )
        state = self.tn if state is None else state
        if target_signature is None:
            backend_infer(state)
            target_like = next(iter(state.tensor_map.values())).data
            target_signature = _array_backend_signature(target_like)
        mismatches = []

        def check_entry(items, types, prefix):
            for index, (payload, event_type) in enumerate(zip(items, types)):
                path = (
                    paths[index]
                    if paths is not None and prefix == path_prefix
                    else f"{prefix}[{index}]"
                )
                if event_type == "gate":
                    source = _array_backend_signature(payload)
                    if not backend_signatures_compatible(source, target_signature):
                        mismatches.append((path, "gate", source))
                elif event_type == "submpo":
                    for tensor_index, tensor in enumerate(
                        getattr(payload, "tensors", ())
                    ):
                        source = _array_backend_signature(tensor.data)
                        if not backend_signatures_compatible(
                            source, target_signature
                        ):
                            mismatches.append(
                                (f"{path}.tensor[{tensor_index}]", "sub-MPO", source)
                            )
                elif event_type == "subtreempo":
                    for network_index, network in enumerate(
                        getattr(payload, "tree_networks", ())
                    ):
                        for tensor_index, tensor in enumerate(network):
                            source = _array_backend_signature(tensor.data)
                            if not backend_signatures_compatible(
                                source, target_signature
                            ):
                                mismatches.append(
                                    (
                                        f"{path}.network[{network_index}]"
                                        f".tensor[{tensor_index}]",
                                        "TreeMPO",
                                        source,
                                    )
                                )
                elif event_type == "conditional":
                    action = payload.get("action")
                    action_payloads, _, action_types = self._normalize_gate_queue(
                        (action,)
                    )
                    check_entry(action_payloads, action_types, f"{path}.action")

        check_entry(payloads, event_types, path_prefix)
        if not mismatches:
            return
        details = "; ".join(
            f"{path} ({kind}) has {source!r}"
            for path, kind, source in mismatches[:8]
        )
        if len(mismatches) > 8:
            details += f"; ... and {len(mismatches) - 8} more"
        raise TypeError(
            "TreeOptimizer requires every gate and MPO payload to match the "
            f"TTN backend/device and required dtype {target_signature!r} "
            f"before use; {details}. "
            "Prepare each payload explicitly with the live backend converter "
            "before passing it to the gate stream."
        )

    def _coerce_tensor_network_backend(self, tn, *, warn=True):
        """Convert every tensor of an operator TN to the live state backend."""
        for tensor in tn.tensor_map.values():
            tensor.modify(data=self._as_state_backend(tensor.data, warn=warn))
        return tn

    def _coerce_tree_mpo_backend(self, tree_mpo):
        """Convert an internally generated TreeMPO without warning."""
        networks = tuple(getattr(tree_mpo, "tree_networks", ()))
        if not networks:
            return tree_mpo
        target_signature = _array_backend_signature(self._state_like())
        if all(
            backend_signatures_compatible(
                _array_backend_signature(tensor.data), target_signature
            )
            for network in networks
            for tensor in network
        ):
            return tree_mpo
        copied = tree_mpo.copy()
        converter = self._backend_converter(self._state_like())
        for network in copied.tree_networks:
            apply_to_arrays = getattr(network, "apply_to_arrays", None)
            if callable(apply_to_arrays):
                apply_to_arrays(converter)
            else:
                for tensor in network.tensor_map.values():
                    tensor.modify(data=self._as_state_backend(tensor.data, warn=False))
        return copied

    def _tag(self, nid):
        return self.tn.node_tag(nid)

    def _tid(self, nid):
        """Return the tensor id of node ``nid`` (self-healing cache on the TTN)."""
        return self.tn.node_tid(nid)

    def _thread_ctx(self):
        """Context manager capping BLAS/OpenMP threads for the small tree ops."""
        @contextlib.contextmanager
        def managed():
            depth = getattr(self, "_thread_ctx_depth", 0)
            if depth:
                self._thread_ctx_depth = depth + 1
                try:
                    yield
                finally:
                    self._thread_ctx_depth -= 1
                return

            self._thread_ctx_depth = 1
            try:
                if _THREAD_CONTROLLER is not None and self.threads is not None:
                    with _THREAD_CONTROLLER.limit(limits=self.threads):
                        yield
                else:
                    yield
            finally:
                self._thread_ctx_depth = 0

        return managed()

    def _bond_name(self, u, v):
        lo, hi = (u, v) if u < v else (v, u)
        return f"_tb{lo}_{hi}"

    def _neighbors(self, nid):
        """Return the adjacent node ids of ``nid`` (children plus parent)."""
        return self.tn.neighbors(nid)

    def _steiner_nodes(self, nodes):
        """Return the node set of the minimal subtree spanning ``nodes``."""
        return self.tn.steiner_nodes(nodes)

    def _build_product_state(self):
        return TreeTensorNetwork.from_plan(self.plan, dtype=self.dtype)

    @staticmethod
    def _product_site_vector(state, q):
        """Extract one qubit's vector from a bond-dimension-one TN site."""
        if isinstance(state, TreeTensorNetwork):
            tensor = state.node_tensor(state.node_of_qubit(q))
            physical_index = state.site_ind(q)
        else:
            site_tag = state.site_tag(q)
            tids = tuple(state.tag_map[site_tag])
            if len(tids) != 1:
                raise ValueError(
                    f"Product MPS site {q} must own exactly one tensor."
                )
            tensor = state.tensor_map[tids[0]]
            physical_index = state.site_ind(q)
        try:
            physical_axis = tensor.inds.index(physical_index)
        except ValueError as exc:
            raise ValueError(
                f"Initial product state is missing physical index {physical_index!r}."
            ) from exc
        moved = ar.do("moveaxis", tensor.data, physical_axis, 0)
        if int(np.prod(ar.shape(moved))) != 2:
            raise ValueError(
                "TreeOptimizer product-state remounting requires qubit "
                "physical dimension two and unit virtual bonds."
            )
        return ar.do("reshape", moved, (2,))

    def _remount_product_state(self, state):
        """Rebuild a product TTN exactly on ``self.plan`` using state vectors."""
        self._state_backend_info(state)
        if not _is_product_tensor_network(state):
            raise ValueError("only bond-dimension-one product states can be remounted.")
        target = TreeTensorNetwork.from_plan(self.plan, dtype=complex)
        sample = next(iter(state.tensor_map.values())).data
        for node in self.plan.nodes():
            tensor = target.node_tensor(node)
            q = self.plan.qubit_of_node.get(node)
            if q is not None:
                vector = self._product_site_vector(state, q)
                tensor.modify(data=ar.do("reshape", vector, ar.shape(tensor.data)))
            else:
                tensor.modify(
                    data=ar.do("ones", ar.shape(tensor.data), like=sample)
                )

        # In a product TTN, every internal tensor is a scalar because every
        # virtual bond has dimension one. Preserve any distributed global
        # normalization/phase by collecting those scalars on the new root.
        if isinstance(state, TreeTensorNetwork):
            factor = None
            for node in state.plan.nodes():
                if node in state.plan.qubit_of_node:
                    continue
                scalar = ar.do("reshape", state.node_tensor(node).data, ())
                factor = scalar if factor is None else factor * scalar
            if factor is not None:
                root_tensor = target.node_tensor(target.plan.root)
                root_tensor.modify(data=root_tensor.data * factor)

        # Quimb stores extracted global base-10 scale separately from tensor
        # data. Preserve it when remounting a geometry-neutral product state.
        if hasattr(state, "exponent"):
            target.exponent = state.exponent
        target._with_center(self.plan.root)
        target._set_isometry_metadata_from_region({self.plan.root}).validate()
        self._state_backend_info(target)
        return target

    def _install_tn(self, tn):
        """Install an independent, canonical copy of a supplied tree state."""
        self._state_backend_info(tn)
        tn.validate()
        self.tn = tn.copy()
        self.plan = self.tn.plan
        self._two_site_path_cache.clear()
        self.tn.validate()
        self.n = self.tn.nqubits
        self._logical_qubits = list(range(self.n))
        self._logical_positions = {q: q for q in self._logical_qubits}
        if (
            self.tn.canonical_region is None
            or not self.tn.is_subtree_canonical_form()
        ):
            self.tn.canonize_around_node_(self.plan.root)

    def _attach_profile_sink(self):
        """Attach the optimizer's optional timing sink to the live TTN."""
        self.tn._profile_sink = self.profile_events if self.profile else None

    def _profile_synchronize(self):
        """Synchronize an asynchronous backend for opt-in phase timing."""
        if not self.profile_sync:
            return
        backend = getattr(self, "backend", None)
        array_backend = getattr(self, "array_backend", backend)
        if array_backend == "cupy" or backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            cp.cuda.runtime.deviceSynchronize()
        elif backend == "torch":
            data = self._state_like()
            if bool(getattr(data, "is_cuda", False)):
                import torch  # pylint: disable=import-outside-toplevel

                torch.cuda.synchronize(data.device)

    def _profile_phase_start(self):
        """Return a phase timestamp, or ``None`` when profiling is disabled."""
        if not self.profile:
            return None
        self._profile_synchronize()
        return time.perf_counter()

    def _profile_phase_event(self, kind, started, **payload):
        """Append one opt-in phase event with optional device synchronization."""
        if started is None:
            return
        self._profile_synchronize()
        event = {
            "kind": str(kind),
            **payload,
            "seconds": time.perf_counter() - started,
        }
        self.profile_events.append(event)

    def set_tn(self, tn):
        """Replace the live tree state with a canonical independent copy."""
        if not isinstance(tn, TreeTensorNetwork):
            raise TypeError("tn must be a TreeTensorNetwork.")
        self._validate_gate_stream_backend(
            self.G, self.event_types, state=tn
        )
        self._install_tn(tn)
        self.truncation_history.clear()
        self.update_history.clear()
        self.bond_history.clear()
        self.infidelities[:] = [0.0]
        self.infidelity_samples.clear()
        self.norm_events.clear()
        self.normalizations.clear()
        self.projection_diagnostics.clear()
        self._truncation_log_survival = 0.0
        self._norm_log_survival = 0.0
        self._update_counter = 0
        self._attach_profile_sink()
        return self

    def set_p(self, tn):
        """MPS-compatible alias for :meth:`set_tn`.

        The name lets a shared coefficient-backend frontend install either an
        MPS or a tree coefficient state without branching on assignment APIs.
        """
        return self.set_tn(tn)

    def _validate_qubit(self, q):
        """Return the compact TTN position for a logical qubit label."""
        if not isinstance(q, Integral):
            raise ValueError(f"qubit label must be an integer; got {q!r}.")
        q = int(q)
        try:
            return self._logical_positions[q]
        except KeyError:
            raise ValueError(f"qubit {q} is outside the tree state.")

    def _validate_support(self, where, *, min_size=1, resolve=True):
        """Validate a logical support and optionally return TTN positions."""
        where = tuple(int(q) for q in where)
        if len(where) < min_size:
            raise ValueError(
                f"gate support must contain at least {min_size} qubit(s); "
                f"got {where}."
            )
        if resolve:
            return tuple(self._validate_qubit(q) for q in where)
        for q in where:
            if q not in self.plan.node_of_qubit:
                raise ValueError(f"tree position {q} is outside the state.")
        return where

    def _require_dense_qubit_state(self, operation):
        """Reject qubit-only helpers on native graded physical spaces."""
        native = bool(getattr(self.tn, "fermionic", False))
        if not native:
            try:
                native = any(
                    _is_symmray_array(tensor.data)
                    for tensor in self.tn.tensor_map.values()
                )
            except AttributeError:
                native = False
        if native:
            raise NotImplementedError(
                f"TreeOptimizer.{operation} is a dense two-level qubit API; "
                "native fermionic TTNs require a model-native Symmray "
                "observable or projector via TreeTensorNetwork.local_expectation."
            )

    def _check_operator_limits(self, where, *, dense=True):
        """Reject dense operators and oversized operator subtrees."""
        if (
            dense
            and
            self.max_operator_qubits is not None
            and len(where) > self.max_operator_qubits
        ):
            raise MemoryError(
                f"operator support of {len(where)} qubits exceeds the configured "
                f"max_operator_qubits={self.max_operator_qubits}."
            )
        if self.max_subtree_nodes is not None and len(where) > 1:
            site_nodes = [self.plan.node_of_qubit[q] for q in where]
            span = self._steiner_nodes(site_nodes)
            if len(span) > self.max_subtree_nodes:
                raise MemoryError(
                    f"operator Steiner subtree has {len(span)} nodes, exceeding "
                    f"max_subtree_nodes={self.max_subtree_nodes}."
                )

    def _projection_snapshot(self, where):
        """Return compact support/span/bond diagnostics for a projection."""
        site_nodes = [self.plan.node_of_qubit[q] for q in where]
        span = frozenset(self._steiner_nodes(site_nodes))
        bonds = {}
        for node in span:
            for neighbour in self._neighbors(node):
                if neighbour not in span or node > neighbour:
                    continue
                edge = (node, neighbour)
                bonds[edge] = int(
                    self.tn.ind_size(self.tn.bond(node, neighbour))
                )
        return {
            "support": tuple(self._logical_qubits[q] for q in where),
            "span": tuple(sorted(span)),
            "bonds": bonds,
            "max_bond": max(bonds.values(), default=1),
        }

    # -- canonical centre tracking -------------------------------------------

    @property
    def p(self):
        """The current :class:`TreeTensorNetwork` state.

        A convenience alias for :attr:`tn` that mirrors the ``MpsOptimizer.p``
        interface, so code written against either engine can read the state via
        ``engine.p``.
        """
        return self.tn

    @p.setter
    def p(self, value):
        self._install_tn(value)

    @property
    def center(self):
        """Node id of the current orthogonality centre (``None`` if unknown).

        A thin view onto the *single* centre tracked by the underlying
        :class:`TreeTensorNetwork` (:attr:`TreeTensorNetwork.orthogonality_center`),
        so the optimizer and the state can never disagree about the canonical
        form; it is carried across :meth:`copy` with the network.
        :attr:`orthogonality_center` is a name-parity alias.
        """
        return self.tn.orthogonality_center

    @center.setter
    def center(self, value):
        self.tn.orthogonality_center = value

    @property
    def orthogonality_center(self):
        """Alias of :attr:`center` matching :attr:`TreeTensorNetwork.orthogonality_center`."""
        return self.tn.orthogonality_center

    @orthogonality_center.setter
    def orthogonality_center(self, value):
        self.tn.orthogonality_center = value

    def _move_center(self, target):
        """Move the orthogonality centre to node ``target`` along the geodesic.

        Delegates to :meth:`TreeTensorNetwork.shift_orthogonality_center`: a
        no-op when already centred, an incremental per-edge QR walk along the
        tree geodesic from a known centre, or regional QR recovery followed by
        a path walk when a multi-node canonical region is known. Only an
        otherwise uncatalogued state requires a full O(N) canonicalisation.
        """
        started = self._profile_phase_start()
        previous = self.center
        try:
            return self.tn.shift_orthogonality_center(target)
        finally:
            self._profile_phase_event(
                "center_movement",
                started,
                source=previous,
                target=target,
            )

    def sync_canonicalization(self, center=None):
        """Rebuild the tracked tree canonical centre after external access.

        Tree canonical metadata is owned by the live
        :class:`TreeTensorNetwork`, unlike the separate ``info_c`` mapping used
        by :class:`MpsOptimizer`. Internal tree readout and gate paths update
        that state-owned metadata. This explicit recovery method is for code
        that directly modified or canonicalized :attr:`tn` through a lower-
        level API and now wants to resume optimizer evolution.

        Diagnostic readout should normally use :meth:`copy` so the live
        optimizer remains untouched.

        Parameters
        ----------
        center : int, optional
            Tree node at which to leave the single-node canonical centre.
            Defaults to the plan root.

        Returns
        -------
        int
            The synchronized tree node centre.
        """
        if center is None:
            center = self.plan.root
        center = int(center)
        # Clearing the state-owned region forces the next shift to use the
        # full canonicalization fallback instead of trusting possibly stale
        # lower-level metadata.
        self._invalidate_state_norm_cache()
        self.tn.orthogonality_center = None
        self.tn.shift_orthogonality_center(center)
        return self.center

    def _nearest_anchor(self, nodes):
        """Choose the closest node to the current centre or canonical region.

        Ties preserve the supplied order, keeping gate application
        deterministic. If no canonical location is known, the first node is
        used and the subsequent centre move establishes one.
        """
        nodes = tuple(nodes)
        if not nodes:
            raise ValueError("at least one anchor node is required.")
        center = self.center
        region = self.canonical_region
        if center is not None:
            return min(
                nodes,
                key=lambda node: len(self.plan.node_path(center, node)),
            )
        if region:
            return min(
                nodes,
                key=lambda node: min(
                    len(self.plan.node_path(r, node)) for r in region
                ),
            )
        return nodes[0]

    def _cached_two_site_path(self, qa, qb):
        """Return ``(leaf_a, leaf_b, path_a_to_b)`` for a gate support.

        The path depends only on the immutable tree plan, while the direction
        in which it is traversed depends on the current orthogonality centre.
        Cache the undirected support path and orient it for the caller so
        repeated gate supports avoid rebuilding the same geodesic without
        making a stale centre assumption.
        """
        key = (qa, qb) if qa < qb else (qb, qa)
        cached = self._two_site_path_cache.get(key)
        if cached is None:
            qlo, qhi = key
            leaf_lo = self.plan.node_of_qubit[qlo]
            leaf_hi = self.plan.node_of_qubit[qhi]
            path = tuple(self.plan.node_path(leaf_lo, leaf_hi))
            cached = (leaf_lo, leaf_hi, path)
            if len(self._two_site_path_cache) >= self._two_site_path_cache_limit:
                self._two_site_path_cache.pop(
                    next(iter(self._two_site_path_cache))
                )
            self._two_site_path_cache[key] = cached

        leaf_lo, leaf_hi, path = cached
        if qa < qb:
            return leaf_lo, leaf_hi, path
        return leaf_hi, leaf_lo, path[::-1]

    def shift_orthogonality_center(self, node):
        """Move the orthogonality centre to ``node`` along the tree geodesic.

        Public entry point to the same incremental per-edge canonicalisation the
        optimizer runs internally before gates and read-outs: the centre is
        walked to ``node`` with a lossless per-edge QR (a no-op when already
        centred, or a single O(N) canonicalisation when the centre is unknown),
        mirroring :meth:`TreeTensorNetwork.shift_orthogonality_center` and the
        MPS ``shift_orthogonality_center``.  Returns ``self`` so moves chain.
        """
        self._move_center(node)
        return self

    def is_canonical_form(self, center=None, *, tol=1e-9):
        """Whether the state is in canonical form about ``center``.

        ``center`` defaults to the tracked :attr:`center`.  Delegates to
        :meth:`TreeTensorNetwork.is_canonical_form`: every non-centre tensor must
        be an isometry pointing toward the centre.  A diagnostic / test aid.
        """
        return self.tn.is_canonical_form(center, tol=tol)

    def isometry_direction(self, node):
        """Neighbour toward which ``node`` has a proven local isometry."""
        return self.tn.isometry_direction(node)

    def isometry_map(self):
        """Return the live network-owned node-isometry orientation map."""
        return self.tn.isometry_map()

    def can_skip_canonize(self, a, b, *, absorb="right"):
        """Whether local metadata proves this edge QR is redundant."""
        return self.tn.can_skip_canonize(a, b, absorb=absorb)

    def validate_isometry_metadata(self, region=None):
        """Validate live tensor ``left_inds`` against the canonical region."""
        self.tn.validate_isometry_metadata(region)
        return self

    @property
    def canonical_region(self):
        """Frozenset of node ids forming the canonicalised subtree (``None`` if unknown).

        A thin view onto :attr:`TreeTensorNetwork.canonical_region`, the range /
        subtree generalisation of :attr:`center`: every tensor outside the region
        points inward toward it.  Assigning validates connectedness.
        """
        return self.tn.canonical_region

    @canonical_region.setter
    def canonical_region(self, value):
        self.tn.canonical_region = value

    def layout_report(self):
        """Return the stored layout diagnostics for the live tree, if available."""
        if self.layout_finder is None:
            return {
                "n_qubits": self.n,
                "root": self.plan.root,
                "root_qubit": self.plan.root_qubit,
                "map_mode": self.plan.map_mode,
                "top_arity": self.plan.top_arity,
                "is_binary": self.plan.is_binary(),
                "is_strictly_binary": self.plan.is_strictly_binary(),
                "max_arity": self.plan.max_arity(),
                "max_tensor_rank": self.plan.max_tensor_rank(),
            }
        return self.layout_finder.report(self.plan)

    def _layout_candidate_record(self, finder, plan, *, source=None):
        """Build the common static record for a pilot candidate."""
        return {
            "plan": plan,
            "objective_key": finder._selection_key(plan, self.chi),
            "path_score": finder.score(plan),
            "tensor_cost": finder._tensor_cost_key(plan),
            "edge_loads": finder.edge_loads(plan),
            **({"source": source} if source is not None else {}),
        }

    @staticmethod
    def _pilot_edge_diagnostics(events):
        """Aggregate replay losses by the immutable planned tree edge."""
        diagnostics = {}
        for event in events:
            edge = tuple(event["edge"])
            metric = diagnostics.setdefault(edge, {
                "events": 0,
                "truncated": 0,
                "discarded_weight": 0.0,
                "discarded_fraction": 0.0,
                "tracked": False,
                "max_before_bond": 0,
                "max_after_bond": 0,
            })
            metric["events"] += 1
            metric["truncated"] += int(bool(event.get("truncated", False)))
            metric["max_before_bond"] = max(
                metric["max_before_bond"], int(event.get("before_bond", 0))
            )
            metric["max_after_bond"] = max(
                metric["max_after_bond"], int(event.get("after_bond", 0))
            )
            discarded_weight = event.get("discarded_weight")
            discarded_fraction = event.get("discarded_fraction")
            if discarded_weight is not None:
                metric["tracked"] = True
                metric["discarded_weight"] += float(discarded_weight)
            if discarded_fraction is not None:
                metric["tracked"] = True
                metric["discarded_fraction"] = max(
                    metric["discarded_fraction"], float(discarded_fraction)
                )
        return diagnostics

    def _pilot_layout_candidate(
        self, plan, *, objective, pilot_steps=None, progbar=False
    ):
        """Replay one candidate and return state-aware diagnostics."""
        started = time.perf_counter()
        trial = type(self)(
            None,
            n=self.n,
            chi=self.chi,
            cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode,
            mode=self.mode,
            compression_mode=self.compression_mode,
            compression_seed=self.compression_seed,
            fit_block_size=self.fit_block_size,
            fit_n_iter=self.fit_n_iter,
            fit_adaptive_sweeps=self.fit_adaptive_sweeps,
            fit_min_iter=self.fit_min_iter,
            fit_rtol=("auto" if self._fit_rtol_requested == "auto" else self.fit_rtol),
            fit_patience=self.fit_patience,
            fit_init_strategy=self.fit_init_strategy,
            fit_init_rand_strength=self.fit_init_rand_strength,
            fit_init_seed=self.fit_init_seed,
            fit_sweep_sequence=self.fit_sweep_sequence,
            fit_overlap_diagnostics=self.fit_overlap_diagnostics,
            fit_finite_check=self.fit_finite_check,
            structure=self.structure,
            max_arity=self.max_arity,
            top_arity=self.top_arity,
            community_frac=self.community_frac,
            star_frac=self.star_frac,
            layout_objective=objective,
            tree=plan,
            dtype=self.dtype,
            threads=self.threads,
            subtree_workers=self.subtree_workers,
            track_truncation=True,
            track_infidelity=True,
            max_intermediate_bond=self.max_intermediate_bond,
            max_operator_qubits=self.max_operator_qubits,
            max_subtree_nodes=self.max_subtree_nodes,
            record_history=True,
            run=False,
            tn=self.tn,
        )
        trial.G = list(self.G)
        trial.where = list(self.where)
        trial.event_types = list(self.event_types)
        if pilot_steps is not None:
            trial.G = trial.G[:pilot_steps]
            trial.where = trial.where[:pilot_steps]
            trial.event_types = trial.event_types[:pilot_steps]
        # Pilots intentionally enable full diagnostics, but their parent
        # optimizer already owns the decision to pay that cost. Suppress the
        # user-facing warning for these internal comparison replays.
        trial._track_warning_emitted = True
        try:
            trial.run(progbar=progbar)
        except Exception as exc:  # pragma: no cover - backend-specific
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - started),
                "pilot_steps": len(trial.G),
            }

        elapsed = float(time.perf_counter() - started)
        events = deepcopy(trial.truncation_history)
        edge_diagnostics = self._pilot_edge_diagnostics(events)
        tracked = [
            event for event in events
            if event.get("discarded_weight") is not None
        ]
        total_discarded = float(
            sum(event["discarded_weight"] for event in tracked)
        ) if tracked else 0.0
        max_fraction = max(
            (float(event["discarded_fraction"]) for event in tracked),
            default=0.0,
        )
        update_runtime = float(sum(
            update.get("elapsed_seconds", 0.0)
            for update in trial.update_history
        ))
        return {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "update_runtime_seconds": update_runtime,
            "infidelity": float(trial.infidelities[-1]),
            "final_bond": int(trial.max_bond()),
            "truncated_edges": int(sum(
                event.get("truncated", False) for event in events
            )),
            "total_discarded_weight": total_discarded,
            "max_discarded_fraction": float(max_fraction),
            "pilot_steps": len(trial.G),
            "edge_diagnostics": edge_diagnostics,
            "updates": deepcopy(trial.update_history),
        }

    def optimize_layout(
        self,
        *,
        objective=None,
        pilot_candidates=4,
        candidate_budget=None,
        pilot_steps=None,
        pilot_workers=1,
        include_quality=True,
        rounds=2,
        topology_budget=None,
        refine_budget=None,
        search_budget=None,
        seed=0,
        install=False,
        progbar=False,
    ):
        """Optimize a tree layout with bounded pilot-guided feedback.

        ``TreeLayoutFinder`` first generates static candidates, including the
        all-scale ``objective="full_tree"`` search when requested. Each round
        replays a short list on independent copies of the current *product*
        state using the real tree kernels. The measured per-edge truncation
        losses then seed targeted NNI, subtree, and cut-crossing leaf proposals
        for the next round. The finder remains circuit-only: it does not
        allocate tensors, replay gates, or perform truncations.

        The original optimizer is unchanged unless ``install=True`` is passed.
        Installation is restricted to product states because an entangled TTN
        cannot be relaid out exactly without an explicit state conversion.
        ``objective="full_tree"`` is the recommended high-quality mode; its
        static score covers routing demand, tensor width, work, and every tree
        scale, while the pilot supplies the final state-aware choice.
        ``candidate_budget`` is an alias for ``pilot_candidates`` for callers
        that want to express the total candidate-pilot budget explicitly.
        """
        try:
            if candidate_budget is not None:
                pilot_candidates = candidate_budget
            pilot_candidates = int(pilot_candidates)
            rounds = int(rounds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pilot_candidates and rounds must be positive integers."
            ) from exc
        if pilot_candidates < 1 or rounds < 1:
            raise ValueError("pilot_candidates and rounds must be positive integers.")
        if pilot_steps is not None:
            try:
                pilot_steps = int(pilot_steps)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "pilot_steps must be a positive integer or None."
                ) from exc
            if pilot_steps < 1:
                raise ValueError("pilot_steps must be a positive integer or None.")
        try:
            pilot_workers = int(pilot_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("pilot_workers must be a positive integer.") from exc
        if pilot_workers < 1:
            raise ValueError("pilot_workers must be a positive integer.")
        if not _is_product_tensor_network(self.tn):
            raise ValueError(
                "Tree layout pilots require a product initial state when "
                "comparing different tree geometries. Convert the entangled "
                "state explicitly onto each candidate plan first."
            )
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer.") from exc

        objective = self.layout_objective if objective is None else objective
        previous_plan = None
        previous_edge_diagnostics = None
        round_reports = []
        final_finder = None
        final_candidates = None
        final_ranked = None
        final_selected_name = None

        for round_index in range(rounds):
            finder = TreeLayoutFinder(
                gates=self._layout_gate_stream(),
                n=self.n,
                structure=self.structure,
                max_arity=self.max_arity,
                community_frac=self.community_frac,
                star_frac=self.star_frac,
                objective=objective,
                weight_mode=self.layout_weight_mode,
                time_decay=self.layout_time_decay,
                time_window=self.layout_time_window,
                chi=self.chi,
                max_operator_qubits=self.max_operator_qubits,
                root_qubit=self.plan.root_qubit,
                top_arity=self.top_arity,
                seed=seed + round_index,
            )
            quality_kwargs = {
                "chi": self.chi,
                "include_quality": bool(include_quality),
            }
            if topology_budget is not None:
                quality_kwargs["quality_topology_budget"] = topology_budget
            if refine_budget is not None:
                quality_kwargs["quality_refine_budget"] = refine_budget
            if search_budget is not None:
                quality_kwargs["quality_search_budget"] = search_budget
            quality_kwargs["quality_seed"] = seed + round_index
            candidates = finder.candidate_plans(**quality_kwargs)

            if (
                previous_plan is not None
                and previous_edge_diagnostics
                and rounds > 1
            ):
                targeted = finder.targeted_candidates(
                    previous_plan,
                    previous_edge_diagnostics,
                    chi=self.chi,
                    budget=max(2 * pilot_candidates, 8),
                    seed=seed + round_index,
                )
                for proposal_index, plan in enumerate(targeted):
                    candidates[
                        f"pilot:round={round_index}:proposal={proposal_index}"
                    ] = self._layout_candidate_record(
                        finder,
                        plan,
                        source="pilot_feedback",
                    )

            ranked_static = sorted(
                candidates,
                key=lambda name: candidates[name]["objective_key"],
            )
            quality_names = [
                name for name in ranked_static if name.startswith("quality:")
            ]
            feedback_names = [
                name for name in ranked_static
                if name.startswith("pilot:")
            ]
            ordinary_names = [
                name for name in ranked_static
                if not name.startswith(("quality:", "pilot:"))
            ]
            if include_quality:
                ranked = quality_names[:1]
                remaining = pilot_candidates - len(ranked)
                ranked.extend(feedback_names[:remaining])
                remaining = pilot_candidates - len(ranked)
                ranked.extend(ordinary_names[:remaining])
                remaining = pilot_candidates - len(ranked)
                ranked.extend(
                    name for name in ranked_static
                    if name not in ranked
                )
                ranked = ranked[:pilot_candidates]
            else:
                ranked = ranked_static[:pilot_candidates]

            pilot_jobs = [
                (name, candidates[name]["plan"])
                for name in ranked
            ]

            def run_pilot(job):
                name, plan = job
                return name, self._pilot_layout_candidate(
                    plan,
                    objective=finder.objective,
                    pilot_steps=pilot_steps,
                    progbar=progbar,
                )

            if pilot_workers > 1 and len(pilot_jobs) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(
                    max_workers=min(pilot_workers, len(pilot_jobs)),
                    thread_name_prefix="pepsy-tree-pilot",
                ) as pool:
                    pilot_results = list(pool.map(run_pilot, pilot_jobs))
            else:
                pilot_results = [run_pilot(job) for job in pilot_jobs]

            reports = {}
            successful = []
            for name, report in pilot_results:
                reports[name] = report
                if report["status"] != "ok":
                    continue
                successful.append((
                    float(report["infidelity"]),
                    float(report["total_discarded_weight"]),
                    float(report["max_discarded_fraction"]),
                    int(report["truncated_edges"]),
                    float(report["elapsed_seconds"]),
                    int(report["final_bond"]),
                    name,
                ))
            if not successful:
                raise RuntimeError(
                    "All Tree layout pilot candidates failed. "
                    f"Diagnostics: {reports!r}"
                )
            selected_name = min(successful)[-1]
            selected_plan = candidates[selected_name]["plan"]
            selected_report = reports[selected_name]
            round_reports.append({
                "round": round_index,
                "objective": finder.objective,
                "pilot_candidates": tuple(ranked),
                "selected_candidate": selected_name,
                "reports": reports,
            })
            previous_plan = selected_plan
            previous_edge_diagnostics = selected_report.get(
                "edge_diagnostics", {}
            )
            final_finder = finder
            final_candidates = candidates
            final_ranked = ranked
            final_selected_name = selected_name

        selected_plan = final_candidates[final_selected_name]["plan"]
        if install:
            self.plan = selected_plan
            self._two_site_path_cache.clear()
            self.tn = self._remount_product_state(self.tn)
            self.center = self.plan.root
            self.layout_finder = final_finder
            self.layout_objective = final_finder.objective
        final_round = round_reports[-1]
        return {
            "plan": selected_plan,
            "selected_candidate": final_selected_name,
            "candidates": final_candidates,
            "pilot": {
                "objective": final_finder.objective,
                "include_quality": bool(include_quality),
                "pilot_candidates": tuple(final_ranked),
                "selected_candidate": final_selected_name,
                "reports": final_round["reports"],
                "rounds": round_reports,
                "n_rounds": rounds,
                "installed": bool(install),
            },
        }

    def select_layout_for_compression(
        self,
        *,
        pilot_candidates=4,
        candidate_budget=None,
        pilot_steps=None,
        pilot_workers=1,
        include_quality=True,
        rounds=1,
        topology_budget=None,
        refine_budget=None,
        search_budget=None,
        seed=0,
        install=False,
        progbar=False,
    ):
        """Backward-compatible one-round compression layout selection.

        For iterative state-aware optimization, use
        :meth:`optimize_layout`, for example with
        ``objective="full_tree"`` and ``rounds=2``. This wrapper retains the
        original compression objective and return shape.
        """
        return self.optimize_layout(
            objective="compression",
            pilot_candidates=pilot_candidates,
            candidate_budget=candidate_budget,
            pilot_steps=pilot_steps,
            pilot_workers=pilot_workers,
            include_quality=include_quality,
            rounds=rounds,
            topology_budget=topology_budget,
            refine_budget=refine_budget,
            search_budget=search_budget,
            seed=seed,
            install=install,
            progbar=progbar,
        )

    def plot_layout(self, plan=None, *, layout_kwargs=None, **plot_kwargs):
        """Plot the tree layout as a Cotengra-style tent.

        The method returns ``(fig, ax)`` and does not alter the live TTN. For
        an optimizer constructed with an explicit :class:`TreePlan`, the
        temporary finder needed for gate diagnostics is built from the queued
        stream without changing that plan.
        """
        finder = self.layout_finder
        if finder is None:
            finder = TreeLayoutFinder(
                gates=self._layout_gate_stream(),
                n=self.n,
                structure=self.structure,
                max_arity=self.max_arity,
                community_frac=self.community_frac,
                star_frac=self.star_frac,
                objective=self.layout_objective,
                weight_mode=self.layout_weight_mode,
                time_decay=self.layout_time_decay,
                time_window=self.layout_time_window,
                chi=self.chi,
                max_operator_qubits=self.max_operator_qubits,
                root_qubit=self.plan.root_qubit,
                top_arity=self.top_arity,
            )
        if plan is None:
            if layout_kwargs:
                plan = finder.run(**dict(layout_kwargs))
            else:
                plan = self.plan
        return finder.plot(plan, **plot_kwargs)

    def plot_rubberband(self, plan=None, *, layout_kwargs=None, **plot_kwargs):
        """Plot the tree's physical clusters as translucent rubberbands."""
        finder = self.layout_finder
        if finder is None:
            finder = TreeLayoutFinder(
                gates=self._layout_gate_stream(),
                n=self.n,
                structure=self.structure,
                max_arity=self.max_arity,
                community_frac=self.community_frac,
                star_frac=self.star_frac,
                objective=self.layout_objective,
                weight_mode=self.layout_weight_mode,
                time_decay=self.layout_time_decay,
                time_window=self.layout_time_window,
                chi=self.chi,
                max_operator_qubits=self.max_operator_qubits,
                root_qubit=self.plan.root_qubit,
                top_arity=self.top_arity,
            )
        if plan is None:
            if layout_kwargs:
                plan = finder.run(**dict(layout_kwargs))
            else:
                plan = self.plan
        return finder.plot_rubberband(plan, **plot_kwargs)

    def plot_tent(self, plan=None, *, layout_kwargs=None, **plot_kwargs):
        """Plot the selected tree as a Cotengra-style tent over the lattice."""
        finder = self.layout_finder
        if finder is None:
            finder = TreeLayoutFinder(
                gates=self._layout_gate_stream(),
                n=self.n,
                structure=self.structure,
                max_arity=self.max_arity,
                community_frac=self.community_frac,
                star_frac=self.star_frac,
                objective=self.layout_objective,
                weight_mode=self.layout_weight_mode,
                time_decay=self.layout_time_decay,
                time_window=self.layout_time_window,
                chi=self.chi,
                max_operator_qubits=self.max_operator_qubits,
                root_qubit=self.plan.root_qubit,
                top_arity=self.top_arity,
            )
        if plan is None:
            if layout_kwargs:
                plan = finder.run(**dict(layout_kwargs))
            else:
                plan = self.plan
        return finder.plot_tent(plan, **plot_kwargs)

    def canonize_subtree(self, nodes, *, span=False):
        """Canonicalise the state around the connected subtree ``nodes``.

        The range / subtree generalisation of :meth:`shift_orthogonality_center`:
        every tensor outside the subtree is gauged to point inward, so the whole
        state norm is carried by the subtree tensors.  Delegates to
        :meth:`TreeTensorNetwork.canonize_subtree_`; pass ``span=True`` to
        auto-expand to the minimal connected subtree spanning ``nodes``.  Returns
        ``self`` so calls chain.
        """
        self.tn.canonize_subtree_(nodes, span=span)
        return self

    def canonize_around_qubits(self, qubits):
        """Canonicalise around the minimal subtree spanning ``qubits``.

        The qubit-level "range canonicalisation" entry point: gauge every tensor
        outside the minimal connected subtree spanning the given qubits' nodes
        to point inward, so the reduced state on those qubits is captured by that
        subtree.  Delegates to :meth:`TreeTensorNetwork.canonize_around_qubits_`.
        Returns ``self``.
        """
        self.tn.canonize_around_qubits_(qubits)
        return self

    def is_subtree_canonical_form(self, nodes=None, *, span=False, tol=1e-9):
        """Whether the state is canonical about the subtree ``nodes``.

        ``nodes`` defaults to the tracked :attr:`canonical_region`.  Delegates to
        :meth:`TreeTensorNetwork.is_subtree_canonical_form`: every tensor outside
        the subtree must be an isometry pointing inward.  A diagnostic / test aid.
        """
        return self.tn.is_subtree_canonical_form(nodes, span=span, tol=tol)

    def canonize_mps(self, p, where, *, info=None):
        """Compatibility canonicalization entry point for shared frontends.

        The name is retained because ``MpsOptimizer`` exposes it publicly. For
        a tree, an integer or singleton support moves the centre to that
        qubit's leaf; a two-qubit support canonicalizes the minimal spanning
        subtree. ``info['cur_orthog']`` is updated when an info mapping is
        supplied, matching the MPS metadata convention.
        """
        if not isinstance(p, TreeTensorNetwork):
            raise TypeError("p must be a TreeTensorNetwork for tree canonicalization.")
        if isinstance(where, Integral):
            sites = (int(where),)
        else:
            try:
                sites = tuple(int(q) for q in where)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "where must be an int or a one-/two-qubit support."
                ) from exc
        if len(sites) not in {1, 2}:
            raise ValueError("where must be an int, (int,), or (int, int).")
        for q in sites:
            if q not in p.plan.node_of_qubit:
                raise ValueError(f"qubit {q} is outside the tree state.")
        if len(sites) == 1:
            p.shift_orthogonality_center(p.plan.node_of_qubit[sites[0]])
            target = (sites[0], sites[0])
        else:
            p.canonize_around_qubits_(sites)
            target = (min(sites), max(sites))
        if info is not None:
            info["cur_orthog"] = target
        return target

    # -- gate application -----------------------------------------------------

    def _begin_update(self, kind, where, *, track_norm=True):
        """Start aggregating diagnostics for one state update.

        ``track_norm`` is explicit because a subtree or MPO update is not
        necessarily unitary. The cheap norm ledger is meaningful for a
        unitary update, but not for a general Kraus/filter operator whose
        physical norm is expected to change.
        """
        if self._active_update is not None:
            return False
        live_before = (
            int(self.tn.max_bond())
            if self.track_bond_diagnostics else None
        )
        update_index = self._update_counter
        self._update_counter += 1
        self._active_update = {
            "kind": str(kind),
            "support": tuple(int(q) for q in where),
            "update": update_index,
            "edge_start": len(self.truncation_history),
            "started_at": time.perf_counter(),
            "live_max_bond_before": live_before,
            "transient_max_bond": live_before,
            "bond_trace": [],
            "track_norm": bool(track_norm),
        }
        # A Tree gate can touch several edges, so the norm event is recorded
        # once for the complete path update. This is the Tree analogue of one
        # MPS compression event. It deliberately does not inspect singular
        # spectra; that extra per-edge work remains behind track_truncation.
        if (
            self.track_infidelity
            and bool(track_norm)
            and self._norm_tracking_enabled
            and str(kind) in {"gate", "subtree", "submpo", "subtreempo"}
        ):
            self._active_update["norm_before"] = float(self.norm())
        return True

    def _record_transient_bond(self, dimension, *, phase, edge=None):
        """Record one potentially transient bond dimension for the update."""
        active = self._active_update
        if active is None or not self.track_bond_diagnostics:
            return
        dimension = int(dimension)
        previous = active["transient_max_bond"]
        active["transient_max_bond"] = (
            dimension if previous is None else max(previous, dimension)
        )
        if self.track_bond_diagnostics:
            active["bond_trace"].append({
                "phase": str(phase),
                "edge": None if edge is None else tuple(int(x) for x in edge),
                "bond": dimension,
            })

    def _abort_update(self):
        """Discard a partial aggregation after a failed state update."""
        if self._active_update is None:
            return
        start = self._active_update["edge_start"]
        del self.truncation_history[start:]
        self._active_update = None

    def _finish_norm_update(self, active):
        """Record one path-level canonical norm-survival event.

        The resulting metric is a retained-norm compression proxy. It is not
        a target-state overlap and it is intentionally independent of the
        optional edge-spectrum records collected by ``track_truncation``.
        """
        if not active.get("track_norm", True):
            return
        expected = active.get("norm_before")
        if expected is None:
            return
        observed = float(self.norm())
        log_local = log_fidelity_from_norms(observed, expected)
        raw_local = (
            None
            if (
                expected <= 0.0
                or not np.isfinite(expected)
                or not np.isfinite(observed)
            )
            else float((observed / expected) ** 2)
        )
        local_fidelity = fidelity_from_log(log_local)
        local_infidelity = infidelity_from_log(log_local)
        if self._norm_log_survival == -np.inf or log_local == -np.inf:
            self._norm_log_survival = -np.inf
        else:
            self._norm_log_survival += float(log_local)
        cumulative_fidelity = fidelity_from_log(self._norm_log_survival)
        cumulative_infidelity = infidelity_from_log(self._norm_log_survival)
        self.norm_events.append({
            "step": int(active["update"]),
            "kind": active["kind"],
            "where": tuple(active["support"]),
            "valid": True,
            "expected_norm": float(abs(expected)),
            "observed_norm": float(abs(observed)),
            "fidelity_raw": raw_local,
            "local_fidelity": local_fidelity,
            "local_infidelity": local_infidelity,
            "cumulative_fidelity": cumulative_fidelity,
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": cumulative_fidelity,
            "cumulative_compression_infidelity": cumulative_infidelity,
        })

    def _finish_update(self):
        """Commit one gate-level truncation aggregation."""
        active = self._active_update
        if active is None:
            return
        elapsed = time.perf_counter() - active["started_at"]
        live_after = (
            int(self.tn.max_bond())
            if self.track_bond_diagnostics else None
        )
        transient_max = active.get("transient_max_bond")
        if transient_max is None:
            transient_max = live_after
        update_index = active.get(
            "update",
            len(self.bond_history)
            if self.track_bond_diagnostics
            else len(self.update_history),
        )
        transient_over_chi = (
            None
            if self.chi is None or transient_max is None
            else bool(transient_max > self.chi)
        )
        bond_record = {
            "update": update_index,
            "kind": active["kind"],
            "support": active["support"],
            "live_max_bond_before": active.get("live_max_bond_before"),
            "transient_max_bond": transient_max,
            "live_max_bond_after": live_after,
            "transient_exceeds_chi": transient_over_chi,
            "bond_trace": deepcopy(active.get("bond_trace", [])),
        }
        self._finish_norm_update(active)
        if self.track_bond_diagnostics:
            self.bond_history.append(deepcopy(bond_record))
        if not self.record_history:
            if self.profile:
                self.profile_events.append({
                    "kind": "update",
                    "update": update_index,
                    "support": active["support"],
                    "seconds": elapsed,
                    "live_max_bond_before": active.get("live_max_bond_before"),
                    "transient_max_bond": transient_max,
                    "live_max_bond_after": live_after,
                })
            self._active_update = None
            return
        start = active["edge_start"]
        edge_events = self.truncation_history[start:]
        tracked = [
            event for event in edge_events
            if event["discarded_fraction"] is not None
        ]
        if tracked:
            edge_log_survival = 0.0
            for event in tracked:
                edge_survival = min(
                    1.0,
                    max(0.0, 1.0 - float(event["discarded_fraction"])),
                )
                if edge_survival <= 0.0:
                    edge_log_survival = -np.inf
                    break
                edge_log_survival += float(np.log(edge_survival))
            relative_loss = infidelity_from_log(edge_log_survival)
            absolute_loss = float(
                sum(event["discarded_weight"] for event in tracked)
            )
            max_edge_loss = float(
                max(event["discarded_weight"] for event in tracked)
            )
            max_edge_fraction = float(
                max(event["discarded_fraction"] for event in tracked)
            )
            if (
                np.isneginf(self._truncation_log_survival)
                or np.isneginf(edge_log_survival)
            ):
                self._truncation_log_survival = -np.inf
            else:
                self._truncation_log_survival += edge_log_survival
            cumulative_loss = infidelity_from_log(
                self._truncation_log_survival,
            )
        else:
            if self.track_truncation and self.mode != "zipup":
                relative_loss = 0.0
                absolute_loss = 0.0
                max_edge_loss = 0.0
                max_edge_fraction = 0.0
                cumulative_loss = infidelity_from_log(
                    self._truncation_log_survival,
                )
            else:
                relative_loss = None
                absolute_loss = None
                max_edge_loss = None
                max_edge_fraction = None
                cumulative_loss = None

        self.update_history.append({
            "update": update_index,
            "kind": active["kind"],
            "support": active["support"],
            "elapsed_seconds": float(elapsed),
            "edge_event_indices": list(range(start, len(self.truncation_history))),
            "edge_count": len(edge_events),
            "truncated_edges": sum(event["truncated"] for event in edge_events),
            "absolute_discarded_weight": absolute_loss,
            "relative_discarded_weight": relative_loss,
            "cumulative_relative_discarded_weight": cumulative_loss,
            "max_edge_discarded_weight": max_edge_loss,
            "max_edge_discarded_fraction": max_edge_fraction,
            "fit_diagnostics": deepcopy(active.get("fit_diagnostics")),
            **bond_record,
        })
        if self.profile:
            self.profile_events.append({
                "kind": "update",
                "update": update_index,
                "support": active["support"],
                "seconds": elapsed,
            })
        if tracked:
            local_infidelity = float(relative_loss)
            self.infidelities.append(float(cumulative_loss))
            self.infidelity_samples.append({
                "step": len(self.infidelity_samples) + 1,
                "where": active["support"],
                "edge_count": len(edge_events),
                "local_fidelity": fidelity_from_log(edge_log_survival),
                "local_infidelity": local_infidelity,
                "infidelity": float(cumulative_loss),
                "cumulative_infidelity": float(cumulative_loss),
                "method": "tree_edge_spectrum",
            })
        self._active_update = None

    def apply_gate(self, gate, where, *, renormalize=False, track_norm=True):
        """Apply a gate and aggregate its edge truncation diagnostics."""
        self._warn_track_truncation_slow()
        started = self._begin_update(
            "gate", _normalize_where(where), track_norm=track_norm
        )
        try:
            result = self._apply_gate_impl(
                gate, where, renormalize=renormalize, track_norm=track_norm
            )
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        return result

    def _apply_gate_tree_mpo_impl(
        self, gate, logical_where, *, renormalize=False, track_norm=True
    ):
        """Apply an ordinary gate through a true TreeMPO active span."""
        from .operators import TreeMPO

        logical_where = _normalize_where(logical_where)
        where = self._validate_support(logical_where)
        self._check_operator_limits(where)
        gate = self._as_state_backend(gate)
        factor_started = (
            self._profile_phase_start() if len(logical_where) > 1 else None
        )
        cache_source = gate
        cache_key = (
            "treempo_gate",
            id(cache_source),
            _array_backend_signature(cache_source),
            tuple(where),
            self.n,
            bool(getattr(self.tn, "fermionic", False)),
            getattr(self.tn, "symmetry", None),
        )
        cached = self._gate_factor_cache.get(cache_key)
        cache_hit = cached is not None and cached[0] is cache_source
        if cache_hit:
            tree_mpo = cached[1]
        else:
            tree_mpo = TreeMPO.from_gate(
                self.plan,
                gate,
                where,
                fermionic=bool(getattr(self.tn, "fermionic", False)),
                symmetry=getattr(self.tn, "symmetry", None),
                dtype=self.backend_dtype,
            )
            self._cache_gate_factorization(cache_key, cache_source, tree_mpo)
        self._profile_phase_event(
            "gate_factorization",
            factor_started,
            route="treempo",
            cache_hit=cache_hit,
            support=tuple(logical_where),
            operator_bond=tree_mpo.max_bond(),
        )
        self.apply_subtreempo(
            tree_mpo,
            tree_mpo.operator_support,
            max_bond=self.chi,
            cutoff=self.cutoff,
            track_norm=track_norm,
            _validate_backend=False,
        )
        if renormalize:
            self.normalize()
        return self

    def _fit_block_size(self):
        """Resolve a named DMRG mode to its requested warm-up block size."""

        if self._dmrg_mode_alias is not None:
            # DMRG1 is the MPS-compatible one-site algorithm with a bounded
            # two-site growth warm-up. DMRG2/3 retain their requested larger
            # local blocks before the common one-site refinement phase.
            return {"dmrg1": 2, "dmrg2": 2, "dmrg3": 3}[self._dmrg_mode_alias]
        return self.fit_block_size

    @staticmethod
    def _normalize_fit_init_strategy(strategy):
        """Normalize a TreeFIT disposable initial-guess policy."""

        strategy = str(strategy).strip().lower().replace("-", "_")
        if strategy == "auto":
            strategy = "guess_src"
        if strategy in {"direct", "random", "random_expand"}:
            return strategy
        if strategy.startswith("guess_") and strategy[6:] in {
            "direct", "dm", "sdc", "src", "zipup"
        }:
            return strategy
        raise ValueError(
            "fit_init_strategy must be one of 'auto', 'direct', 'random', "
            "'random_expand', or 'guess-<method>'."
        )

    def _tree_fit_initial_guess(self, target, region, *, operator=None):
        """Build a disposable TreeFIT guess without replacing the target.

        For a TreeMPO-backed gate, ``guess-src`` follows the requested
        ``sub_treempo @ tree`` route: the operator is applied to a private
        copy of the current tree and compressed with the tree-native SRC
        edge sweep. The exact target is kept separate and is never reused as
        the FIT initial state.
        """

        strategy = self._normalize_fit_init_strategy(self.fit_init_strategy)
        if strategy in {"random", "random_expand"}:
            guess, random_info = _randomize_tree_guess(
                self.tn,
                region,
                target=target,
                max_bond=self.chi,
                strength=self.fit_init_rand_strength,
                expand=strategy == "random_expand",
                seed=self.fit_init_seed,
            )
            return guess, strategy, random_info
        if strategy == "direct" or strategy == "guess_direct":
            return self.tn.copy(), strategy, None
        method = strategy[6:]
        if operator is not None:
            # This disposable replay needs the state and numerical policy,
            # not the parent's gate queue, diagnostics, histories, or RNG.
            # Preserve copy()'s one child-seed draw so later measurements
            # retain their existing seeded sequence.
            child_seed = int(self.rng.integers(0, 2**63, dtype=np.uint64))
            guess_optimizer = type(self)(
                None, state=self.tn, tree=self.plan,
                chi=self.chi, cutoff=self.cutoff, cutoff_mode=self.cutoff_mode,
                mode="zipup" if method == "zipup" else "auto",
                compression_mode="direct" if method == "zipup" else method,
                compression_seed=self.fit_init_seed, seed=child_seed,
                threads=self.threads, subtree_workers=self.subtree_workers,
                max_operator_qubits=self.max_operator_qubits,
                max_subtree_nodes=self.max_subtree_nodes,
                track_infidelity=False, track_truncation=False,
                record_history=False, run=False,
            )
            guess_optimizer._norm_tracking_enabled = False
            guess_optimizer.apply_subtreempo(
                operator,
                getattr(operator, "operator_support", None),
                max_bond=self.chi,
                cutoff=self.cutoff,
                track_norm=False,
                _validate_backend=False,
            )
            return guess_optimizer.tn, strategy, None, "tree_mpo"
        guess = target.copy()
        guess.compress(
            max_bond=self.chi,
            cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode,
            compression_mode=method,
            compression_seed=self.fit_init_seed,
        )
        return guess, strategy, None, "target_compress"

    def _build_tree_fit_target(self, gate, logical_where):
        """Build an exact layered operator--state TreeFIT target."""

        from .operators import TreeMPO

        gate = self._as_state_backend(gate)
        where = self._validate_support(logical_where)
        tree_mpo = TreeMPO.from_gate(
            self.plan,
            gate,
            where,
            fermionic=bool(getattr(self.tn, "fermionic", False)),
            symmetry=getattr(self.tn, "symmetry", None),
            dtype=self.backend_dtype,
        )

        return _build_layered_operator_state_target(self.tn, tree_mpo), tree_mpo

    def _run_tree_fit(self, target, region, support, *, operator=None, target_norm=None):
        """Fit one exact tree target and install it atomically."""

        guess_result = self._tree_fit_initial_guess(
            target,
            region,
            operator=operator,
        )
        if len(guess_result) == 3:
            guess, strategy, random_info = guess_result
            guess_backend = None
        else:
            guess, strategy, random_info, guess_backend = guess_result
        split_method = "direct" if self.compression_mode == "sdc" else self.compression_mode
        block_size = self._fit_block_size()
        fit = TreeFIT(
            target,
            guess,
            max_bond=self.chi,
            cutoffs=self.cutoff,
            cutoff_mode=self.cutoff_mode,
            split_method=split_method,
            split_seed=(
                self.compression_seed
                if self.compression_seed is not None else self.fit_init_seed
            ),
            inplace=True,
            copy_target=False,
            finite_check=self.fit_finite_check,
            target_norm=target_norm,
        )
        active_block_size = min(block_size, len(region))
        if (
            self._dmrg_mode_alias == "dmrg1"
            and block_size == 2
            and fit._active_bonds_at_rank_targets(region, state=self.tn)
        ):
            active_block_size = 1
        if (
            self._dmrg_mode_alias == "dmrg1"
            and active_block_size == 2
            and len(region) > 2
            and not fit._active_bonds_at_rank_targets(region, state=self.tn)
            and self.fit_n_iter < 3
        ):
            raise ValueError(
                "mode='dmrg1' requires fit_n_iter >= 3 for an under-capacity "
                "tree window: two block-growth sweeps and one-site refinement."
            )
        adaptive_sweeps = (
            2 if self._dmrg_mode_alias == "dmrg1"
            else self.fit_adaptive_sweeps
        )
        fit_rtol = (
            None if self._fit_rtol_requested == "auto" and (
                target_norm is None or not self._norm_tracking_enabled
            ) else self.fit_rtol
        )
        fit.run_gate(
            region,
            n_iter=self.fit_n_iter,
            block_size=active_block_size,
            sweep_sequence=self.fit_sweep_sequence,
            min_iter=self.fit_min_iter,
            rtol=fit_rtol,
            patience=self.fit_patience,
            adaptive_block_sweeps=adaptive_sweeps,
            adaptive_until_rank=(
                self._dmrg_mode_alias is None
                and not (
                    active_block_size in {2, 3}
                    and len(region) > active_block_size
                )
            ),
        )
        diagnostics = fit.fit_diagnostics(overlap=self.fit_overlap_diagnostics)
        diagnostics.update(
            {
                "backend": "tree_fit",
                "support": tuple(support),
                "region": tuple(sorted(region)),
                "fit_init_strategy": strategy,
                "fit_init_strategy_requested": self.fit_init_strategy,
                "fit_rtol": fit_rtol,
                "fit_rtol_requested": self._fit_rtol_requested,
                "guess_used": strategy != "direct",
                "guess_method": (
                    strategy[6:] if strategy.startswith("guess_") else strategy
                ),
                "random_initialization": bool(
                    random_info and random_info["enabled"]
                ),
                "random_initialization_info": random_info,
                "block_size": active_block_size,
                "requested_block_size": block_size,
                "adaptive_sweeps": fit.adaptive_sweeps_run,
                "one_site_refinement_sweeps": fit.one_site_sweeps_run,
                "block_size_trace": tuple(fit.block_size_trace),
                "guess_backend": guess_backend,
                "target_layout": fit.target_layout,
            }
        )
        self.tn = fit.p
        self._invalidate_state_norm_cache()
        self.plan = self.tn.plan
        self._last_fit_diagnostics = diagnostics
        self.fit_diagnostics.append(deepcopy(diagnostics))
        if self._active_update is not None:
            self._active_update["fit_diagnostics"] = deepcopy(diagnostics)
            target_edges = tuple(
                (node0, node1)
                for node0 in sorted(region)
                for node1 in self.tn.neighbors(node0)
                if node1 in region and node0 < node1
            )
            self._record_transient_bond(
                max(
                    _layered_target_bond_sizes(target, self.tn, target_edges)
                    .values(),
                    default=1,
                ),
                phase="fit.target",
            )
        return self

    def _apply_gate_dmrg_impl(
        self, gate, logical_where, *, renormalize=False, track_norm=True
    ):
        """Apply a gate through the shared TreeMPO-to-TreeFIT path."""

        return self._apply_gate_tree_mpo_impl(
            gate,
            logical_where,
            renormalize=renormalize,
            track_norm=track_norm,
        )

    def _apply_gate_impl(
        self, gate, where, *, renormalize=False, track_norm=True
    ):
        """Apply a gate without opening a nested diagnostic update."""
        logical_where = _normalize_where(where)
        if self.mode == "dmrg":
            return self._apply_gate_dmrg_impl(
                gate,
                logical_where,
                renormalize=renormalize,
                track_norm=track_norm,
            )
        where = self._validate_support(logical_where)
        self._check_operator_limits(where)
        if len(logical_where) == 2 and logical_where[0] == logical_where[1]:
            raise ValueError(
                "A two-qubit gate needs two distinct qubits; "
                f"got where={logical_where}."
            )
        if len(logical_where) > 2 and len(set(logical_where)) != len(logical_where):
            raise ValueError(
                "A multi-qubit gate needs distinct qubits; "
                f"got where={logical_where}."
            )
        route = self._gate_route(len(logical_where))
        if route == "treempo":
            return self._apply_gate_tree_mpo_impl(
                gate,
                logical_where,
                renormalize=renormalize,
                track_norm=track_norm,
            )
        if route == "submpo" and len(logical_where) > 1:
            raise ValueError(
                "mode='submpo' accepts explicit sub-MPO stream events, not "
                "ordinary dense gates; use mode='direct', mode='dm', or a "
                "tree_mpo_* mode."
            )
        with self._thread_ctx():
            if len(where) == 1:
                self.apply_1q(
                    gate,
                    logical_where[0],
                    renormalize=renormalize,
                    track_norm=track_norm,
                )
            elif len(where) == 2:
                if where[0] == where[1]:
                    raise ValueError(
                        "A two-qubit gate needs two distinct qubits; "
                        f"got where={where}."
                    )
                self.apply_2q(
                    gate,
                    logical_where[0],
                    logical_where[1],
                    track_norm=track_norm,
                )
                if renormalize:
                    self.normalize()
            else:
                if len(set(where)) != len(where):
                    raise ValueError(
                        "A multi-qubit gate needs distinct qubits; "
                        f"got where={where}."
                    )
                self.apply_subtree_operator(
                    gate,
                    logical_where,
                    renormalize=renormalize,
                    track_norm=track_norm,
                )
        return self

    def _trajectory_gate_stream(self):
        """Return the normalized queued stream in public replay form.

        ``self.G`` stores control events in split payload/support arrays so
        the live TTN replay loop can avoid reparsing them.  The trajectory and
        MPI runners consume the public bundled stream instead, so rebuild the
        entries without using ``_layout_gate_stream``: layout discovery is
        deliberately allowed to replace conditional actions with placeholders,
        whereas shot replay must preserve the actual action.
        """
        stream = []
        for payload, where, event_type in zip(
            self.G, self.where, self.event_types
        ):
            support = _normalize_where(where)
            if event_type == "gate":
                stream.append((payload, support))
            elif event_type == "subtreempo":
                stream.append(self.subtreempo_event(payload, support))
            elif event_type == "submpo":
                stream.append(self.submpo_event(payload, support))
            elif event_type == "measure":
                stream.append({
                    "kind": "measure",
                    "pauli": payload["pauli"],
                    "where": support,
                    "outcome": payload.get("outcome"),
                })
            elif event_type == "reset":
                stream.append({
                    "kind": "reset",
                    "where": support,
                    "basis": "".join(payload["axes"]),
                })
            elif event_type == "measure_reset":
                stream.append({
                    "kind": "measure_reset",
                    "where": support,
                    "basis": "".join(payload["axes"]),
                    "outcome": payload.get("outcomes"),
                })
            elif event_type == "cap":
                stream.append({
                    "kind": "cap",
                    "where": support[0],
                    "vec": payload["vec"],
                    "absorb": payload.get("absorb", "left"),
                    "compact_labels": payload.get("compact_labels", True),
                })
            elif event_type == "conditional":
                stream.append({
                    "kind": "conditional",
                    "record": payload["record"],
                    "bit": payload["bit"],
                    "action": payload["action"],
                })
            else:  # pragma: no cover - normalized streams are exhaustive
                raise ValueError(f"unknown tree trajectory event {event_type!r}.")
        return tuple(stream)

    def _run_shots(
        self,
        gates,
        shots,
        *,
        error_model=None,
        seed=None,
        run_kwargs=None,
        strategy="auto",
        max_branches=128,
        auto_max_expected_faults=0.1,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        retain="all",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=True,
        checkpoint_id=None,
    ):
        """Replay a tree stream through local or MPI shot orchestration."""
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
            raise ValueError("shots must be a nonnegative integer.")
        mpi_enabled = mpi is not None and mpi is not False
        if not mpi_enabled and any(
            value is not None for value in (observable, checkpoint_path)
        ):
            raise ValueError(
                "observable and checkpoint options require mpi=True or an "
                "MPI communicator."
            )
        if not mpi_enabled and (
            resume
            or checkpoint_keep != 2
            or checkpoint_sync is not True
            or collect_diagnostics is not True
            or checkpoint_id is not None
        ):
            raise ValueError(
                "MPI checkpoint options require mpi=True or an MPI communicator."
            )
        if workers in {None, "auto"} and parallel_workers != 1:
            workers = parallel_workers
        if mpi_enabled:
            from ..mpi import MPIShotRunner  # pylint: disable=import-outside-toplevel

            child_kwargs = dict(run_kwargs or {})
            if progress not in {False, "never"}:
                child_kwargs["progbar"] = False
            parent_rng_state = deepcopy(self.rng.bit_generator.state)
            template = self.copy()
            self.rng.bit_generator.state = parent_rng_state
            communicator = None if mpi is True else mpi
            runner = MPIShotRunner(
                lambda: template.copy(),
                gates,
                comm=communicator,
            )
            return runner.run(
                shots,
                seed=seed,
                error_model=error_model,
                run_kwargs=child_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=importance_sampling,
                auto_max_expected_faults=auto_max_expected_faults,
                retain=retain,
                local_workers=workers,
                local_backend="auto",
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
                progress=progress,
            )

        from ..mpi import (  # pylint: disable=import-outside-toplevel
            _make_progress_bar,
            _resolve_local_workers,
            _validate_progress,
        )
        from ..noise import (  # pylint: disable=import-outside-toplevel
            NoisyResult,
            _as_entries,
            run_noisy_shots,
            run_trajectory_shots,
        )

        workers = _resolve_local_workers(workers, shots=shots)
        gates = tuple(_as_entries(gates))
        progress_strategy = strategy
        if workers > 1 and strategy == "auto":
            from ..noise import _resolve_auto_parallel_strategy

            progress_strategy = _resolve_auto_parallel_strategy(
                gates,
                shots,
                error_model=error_model,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                auto_max_expected_faults=auto_max_expected_faults,
            )
        progress_mode = _validate_progress(progress)
        progress_bar = (
            _make_progress_bar(progress_mode, shots, desc="shots")
            if workers > 1 and progress_strategy == "independent"
            else None
        )
        child_kwargs = dict(run_kwargs or {})
        if workers > 1 and progress_mode != "never":
            child_kwargs["progbar"] = False
        parent_rng_state = deepcopy(self.rng.bit_generator.state)
        template = self.copy()
        self.rng.bit_generator.state = parent_rng_state
        factory = lambda: template.copy()
        common = {
            "seed": seed,
            "run_kwargs": child_kwargs,
            "strategy": strategy,
            "max_branches": max_branches,
            "importance_sampling": importance_sampling,
            "max_branch_factor": max_branch_factor,
            "parallel_workers": workers,
            "parallel_backend": parallel_backend,
            "retain": retain,
        }
        def update_progress(delta):
            if progress_bar is not None:
                progress_bar.update(int(delta))

        try:
            if error_model is None:
                raw = run_trajectory_shots(
                    factory,
                    gates,
                    shots,
                    _progress=(
                        update_progress if progress_bar is not None else None
                    ),
                    **common,
                )
            else:
                raw = run_noisy_shots(
                    factory,
                    gates,
                    error_model,
                    shots,
                    auto_max_expected_faults=auto_max_expected_faults,
                    _progress=(
                        update_progress if progress_bar is not None else None
                    ),
                    **common,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return NoisyResult(raw)

    def run(
        self,
        gates=None,
        *,
        progbar=False,
        mode=None,
        compression_mode=None,
        compression_seed=None,
        non_unitary=False,
        normalize_every=False,
        normalize_final=False,
        normalize_eps=1e-15,
        seed=None,
        track_infidelity=None,
        shots=1,
        error_model=None,
        strategy="auto",
        run_kwargs=None,
        max_branches=128,
        auto_max_expected_faults=0.1,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=True,
        checkpoint_id=None,
        retain="all",
    ):
        """Replay ``gates`` (or the construction stream) on the tree.

        Parameters
        ----------
        gates : bundled gate stream, optional
            Replacement stream to replay. If omitted, replay the queued stream.
        progbar : bool, default=False
            Show a tqdm progress bar with the active mode, two-qubit event
            count, cumulative retained-norm fidelity (``~F``), and live
            maximum bond. The bar uses the same core readout as
            :class:`MpsOptimizer`; tree-specific ``kq``, ``ctrl``, and
            explicit-operator ``mpo`` counters are added when present.
        mode : {"auto", "direct", "dm", "sdc", "src", "zipup", "dmrg", "dmrg1", "dmrg2", "dmrg3", "tree_mpo_direct", "tree_mpo_dm", "mpo", "submpo"} | {"tree", "ttn"} | None, default=None
            Optional persistent gate/sub-MPO replay selection: a supplied
            value updates :attr:`mode` before replay and remains active for
            future runs and copies. ``"submpo"`` validates an explicit chain
            MPO stream; ``"tree_mpo_direct"`` and ``"tree_mpo_dm"`` require
            TreeMPO routing. ``"tree"``/``"ttn"`` are deprecated no-op
            compatibility selectors for shared coefficient frontends.
            ``"dm"`` selects automatic gate routing with density-matrix
            compression. ``"sdc"`` and ``"src"`` select automatic routing
            with deterministic or randomized successive tree-edge
            compression, respectively. ``"dmrg"`` selects TreeFIT with the
            configured adaptive block schedule. ``"dmrg1"`` and ``"dmrg2"``
            use two-node warm-up blocks, while ``"dmrg3"`` uses three-node
            warm-up blocks; each named schedule then performs one-node
            refinement.
        compression_mode : {"direct", "dm", "sdc", "src"} | None, default=None
            Persistent decomposition used for TreeMPO state truncation and
            layered TreeFIT local splits.
        compression_seed : int | None, default=None
            Override the configured randomized-compression seed for this run.
        non_unitary : bool, default=False
            Mark the stream as non-unitary when using automatic working-scale
            control.
        normalize_every : bool, default=False
            Normalize the canonical working tensor after every replay event
            and accumulate the removed global scale in ``tn.exponent``.
            Requires ``non_unitary=True``.
        normalize_final : bool, default=False
            Apply working-scale control once after replay. Requires
            ``non_unitary=True``.
        normalize_eps : float, default=1e-15
            Zero-state threshold used by automatic normalization.
        seed : int | None, default=None
            Reseed measurement/reset sampling before replay.
        track_infidelity : bool | None, default=None
            Override :attr:`track_infidelity` for this replay. When disabled,
            the progress bar omits the norm-based infidelity field and avoids
            the per-event norm readout. Truncation-spectrum diagnostics remain
            controlled independently by :attr:`track_truncation`.

        Notes
        -----
        track_truncation remains False by default. Enabling it is a
        diagnostic mode: complete singular spectra require additional
        factorization work and can substantially slow replay.

        ``shots`` and ``mpi`` use the shared trajectory/MPI runner while
        keeping the parent optimizer unchanged.  The shot factory starts from
        the current tree state, so an already prepared tree can be sampled
        without replaying or consuming the caller's optimizer.
        """
        self._warn_track_truncation_slow()
        if mode is not None:
            requested_mode_raw = str(mode).strip().lower().replace("-", "_")
            if requested_mode_raw in {"tree", "ttn", "tree_tensor_network"}:
                warnings.warn(
                    "run(mode='tree'/'ttn') is a deprecated no-op; use "
                    "mode='auto', 'direct', 'dm', 'tree_mpo_direct', "
                    "'tree_mpo_dm', 'mpo', or 'submpo' to select "
                    "a gate/sub-MPO implementation.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                requested_mode = self._normalize_mode(requested_mode_raw)
                if requested_mode in {"dm", "sdc", "src"}:
                    self._dmrg_mode_alias = None
                    if compression_mode is not None:
                        requested_compression = self._normalize_compression_mode(
                            compression_mode
                        )
                        if requested_compression not in {"direct", requested_mode}:
                            raise ValueError(
                                f"mode={requested_mode!r} cannot be combined "
                                "with a different compression_mode."
                            )
                    self.mode = "auto"
                    self.compression_mode = requested_mode
                    # The shorthand owns the compression choice. The
                    # explicit ``direct`` value is only the neutral default
                    # accepted alongside it and must not overwrite it below.
                    compression_mode = None
                else:
                    requested_compression = (
                        None if compression_mode is None
                        else self._normalize_compression_mode(compression_mode)
                    )
                    if requested_mode in {
                        "zipup", "tree_mpo_direct", "tree_mpo_dm"
                    }:
                        self._dmrg_mode_alias = None
                        expected = (
                            "dm" if requested_mode == "tree_mpo_dm"
                            else "direct"
                        )
                        if (
                            requested_compression is not None
                            and requested_compression != expected
                        ):
                            raise ValueError(
                                f"mode={requested_mode!r} requires "
                                f"compression_mode={expected!r}."
                            )
                        self.mode = requested_mode
                        self.compression_mode = expected
                    else:
                        if requested_mode in {"dmrg1", "dmrg2", "dmrg3"}:
                            self._dmrg_mode_alias = requested_mode
                            self.mode = "dmrg"
                        else:
                            self._dmrg_mode_alias = None
                            self.mode = requested_mode
        if compression_mode is not None:
            if self.mode in {"zipup", "tree_mpo_direct", "tree_mpo_dm"}:
                normalized_compression = self._normalize_compression_mode(
                    compression_mode
                )
                expected = (
                    "dm" if self.mode == "tree_mpo_dm" else "direct"
                )
                if normalized_compression not in {expected, "direct"}:
                    raise ValueError(
                        f"mode={self.mode!r} requires "
                        f"compression_mode={expected!r}."
                    )
                self.compression_mode = expected
            else:
                self.compression_mode = self._normalize_compression_mode(
                    compression_mode
                )
        if compression_seed is not None:
            if isinstance(compression_seed, bool) or not isinstance(
                compression_seed, Integral
            ):
                raise TypeError("compression_seed must be an integer or None.")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError("compression_seed must be non-negative.")
            self.compression_seed = compression_seed
        non_unitary = bool(non_unitary)
        if track_infidelity is not None:
            self.track_infidelity = bool(track_infidelity)
        if not non_unitary and normalize_every not in (False, None):
            raise ValueError("normalize_every requires non_unitary=True.")
        if not non_unitary and normalize_final:
            raise ValueError("normalize_final requires non_unitary=True.")
        normalize_every = bool(normalize_every)
        normalize_eps = float(normalize_eps)
        if normalize_eps < 0.0:
            raise ValueError("normalize_eps must be non-negative.")

        shot_requested = bool(
            error_model is not None
            or isinstance(shots, bool)
            or not isinstance(shots, Integral)
            or int(shots) != 1
            or run_kwargs is not None
            or strategy != "auto"
            or max_branches != 128
            or auto_max_expected_faults != 0.1
            or importance_sampling is not None
            or max_branch_factor is not None
            or parallel_workers != 1
            or parallel_backend != "thread"
            or retain != "all"
            or (mpi is not None and mpi is not False)
            or (workers is not None and workers != "auto")
            or observable is not None
            or checkpoint_path is not None
        )
        if shot_requested:
            if run_kwargs is not None and not isinstance(run_kwargs, Mapping):
                raise TypeError("run_kwargs must be a mapping or None.")
            if non_unitary:
                child_kwargs = dict(run_kwargs or {})
                child_kwargs.setdefault("non_unitary", True)
                child_kwargs.setdefault("normalize_every", normalize_every)
                child_kwargs.setdefault("normalize_final", normalize_final)
                child_kwargs.setdefault("normalize_eps", normalize_eps)
            else:
                child_kwargs = dict(run_kwargs or {})
            if mode is not None:
                child_kwargs.setdefault("mode", mode)
            if compression_mode is not None:
                child_kwargs.setdefault("compression_mode", compression_mode)
            if compression_seed is not None:
                child_kwargs.setdefault("compression_seed", compression_seed)
            child_kwargs.setdefault("track_infidelity", track_infidelity)
            child_kwargs.setdefault("progbar", progbar)
            stream = self._trajectory_gate_stream() if gates is None else gates
            return self._run_shots(
                stream,
                shots,
                error_model=error_model,
                seed=seed,
                run_kwargs=child_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                auto_max_expected_faults=auto_max_expected_faults,
                importance_sampling=importance_sampling,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
                retain=retain,
                mpi=mpi,
                workers=workers,
                progress=progress,
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
            )
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if gates is not None:
            normalized = self._normalize_gate_queue(gates)
            self._validate_gate_stream_backend(normalized[0], normalized[2])
            self.G, self.where, self.event_types = normalized
            self._gate_factor_cache.clear()
        self._validate_event_stream_for_run()
        self._validate_mode_for_stream()
        # The complete stream was validated at installation. Replay therefore
        # uses the caller's payload objects without a second scan or cast.
        payloads = self.G
        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress_mode = self._progress_mode_name()
            pbar = tqdm(
                total=len(self.G),
                desc=progress_mode,
                leave=True,
                position=0,
                ascii=True,
                colour=(
                    self._PROGBAR_COLORS["dmrg"]
                    if progress_mode.startswith("dmrg")
                    else self._PROGBAR_COLORS["mpo"]
                ),
            )

        one_qubit_count = 0
        two_qubit_count = 0
        multi_qubit_count = 0
        control_count = 0
        submpo_count = 0
        previous_norm_tracking = self._norm_tracking_enabled
        # A non-unitary stream changes the physical norm for reasons other
        # than compression. Do not present that scale change as retained
        # compression fidelity; explicit Tree calls remain unitary by default
        # and can use the cheap ledger normally.
        self._norm_tracking_enabled = not non_unitary

        try:
            for step, (payload, where, event_type) in enumerate(zip(
                payloads, self.where, self.event_types
            ), start=1):
                logical_support = _normalize_where(where)
                support = logical_support
                if event_type == "gate":
                    if len(support) == 1:
                        one_qubit_count += 1
                    elif len(support) == 2:
                        two_qubit_count += 1
                    else:
                        multi_qubit_count += 1
                    self.apply_gate(payload, support)
                elif event_type == "subtreempo":
                    self.apply_subtreempo(
                        payload,
                        support,
                        max_bond=self.chi,
                        cutoff=self.cutoff,
                        _validate_backend=False,
                    )
                    submpo_count += 1
                elif event_type == "submpo":
                    # Reuse the public sub-MPO implementation so stream
                    # replay gets the two-site factor fast path as well as
                    # the native multi-site MPO router. Passing both forms
                    # of support is important after a stable-label cap:
                    # ``support`` addresses compact TTN sites, while
                    # ``logical_support`` addresses the MPO site tags.
                    support = self._validate_support(logical_support)
                    self._apply_submpo_resolved(
                        payload,
                        support,
                        logical_where=logical_support,
                        max_bond=self.chi,
                        cutoff=self.cutoff,
                    )
                    submpo_count += 1
                else:
                    control_count += 1
                    started = self._begin_update(event_type, support)
                    try:
                        self._apply_control_event(event_type, payload, support)
                    except Exception:
                        if started:
                            self._abort_update()
                        raise
                    if started:
                        self._finish_update()

                if normalize_every:
                    self._normalize_and_record_working_scale(
                        step=step,
                        support=support,
                        reason="step",
                        eps=normalize_eps,
                    )

                if pbar is not None:
                    postfix = {
                        "2q": two_qubit_count,
                        "~F": self._format_progress_scalar(
                            self._cumulative_fidelity()
                        ),
                        "bnd": self.tn.max_bond(),
                    }
                    if multi_qubit_count:
                        postfix["kq"] = multi_qubit_count
                    if control_count:
                        postfix["ctrl"] = control_count
                    if submpo_count:
                        postfix["mpo"] = submpo_count
                    pbar.set_postfix(postfix)
                    pbar.update(1)
        finally:
            self._norm_tracking_enabled = previous_norm_tracking
            if pbar is not None:
                pbar.close()
        if normalize_final and self.G:
            self._normalize_and_record_working_scale(
                step=len(self.G),
                support=(),
                reason="final",
                eps=normalize_eps,
            )
        return self

    def set_gates(self, gates):
        """Replace the queued gate stream and return ``self``."""
        normalized = self._normalize_gate_queue(gates)
        self._validate_gate_stream_backend(normalized[0], normalized[2])
        self.G, self.where, self.event_types = normalized
        self._gate_factor_cache.clear()
        return self

    def add_gates(self, gates):
        """Append gates to the queued stream and return ``self``."""
        G_new, where_new, event_types_new = self._normalize_gate_queue(gates)
        self._validate_gate_stream_backend(G_new, event_types_new)
        self.G.extend(G_new)
        self.where.extend(where_new)
        self.event_types.extend(event_types_new)
        self._gate_factor_cache.clear()
        return self

    @staticmethod
    def measure_event(pauli, where, outcome=None):
        """Return a Pauli-measurement event in the MPS-compatible format."""
        where = _normalize_control_where(where)
        _normalize_measure_axes(pauli, where)
        if outcome is None:
            return ("measure", str(pauli), where)
        if not isinstance(outcome, Integral) or int(outcome) not in (-1, 1):
            raise ValueError("measure event outcome must be +1 or -1.")
        return ("measure", str(pauli), where, int(outcome))

    @staticmethod
    def cap_event(where, vec, absorb="left", *, compact_labels=True):
        """Return an MPS-compatible physical-index cap event.

        A tree cap contracts a leaf into its parent; ``absorb`` is accepted for
        stream compatibility and must be ``"left"`` or ``"right"``.  Set
        ``compact_labels=False`` to preserve the caller-facing logical label
        gap; this form is represented as a mapping so the option survives
        stream normalization.
        """
        where = _normalize_control_where(where)
        if len(where) != 1:
            raise ValueError("cap event where must reference exactly one site.")
        if absorb not in {"left", "right"}:
            raise ValueError("cap absorb direction must be 'left' or 'right'.")
        try:
            vec = ar.do("reshape", vec, (-1,))
        except (TypeError, ValueError) as exc:
            raise ValueError("cap event vector must be array-like.") from exc
        if int(ar.shape(vec)[0]) != 2:
            raise ValueError("cap event vector must have exactly two entries.")
        if compact_labels:
            return ("cap", where[0], vec, absorb)
        return {
            "kind": "cap", "where": where[0], "vec": vec,
            "absorb": absorb, "compact_labels": False,
        }

    @staticmethod
    def reset_event(where, basis="Z"):
        """Return a mid-circuit reset event in the MPS-compatible format."""
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(basis, where, event="reset")
        if all(axis == "Z" for axis in axes):
            return ("reset", where)
        return ("reset", where, "".join(axes))

    @staticmethod
    def measure_reset_event(pauli, where, outcome=None):
        """Return a measure-then-reset event in the MPS-compatible format."""
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(pauli, where, event="measure_reset")
        if outcome is None:
            return ("measure_reset", "".join(axes), where)
        if isinstance(outcome, Integral):
            outcome = int(outcome)
            if outcome not in (-1, 1):
                raise ValueError("measure_reset outcome must be +1 or -1.")
            return ("measure_reset", "".join(axes), where, outcome)
        outcomes = tuple(int(value) for value in outcome)
        if len(outcomes) != len(where) or any(value not in (-1, 1) for value in outcomes):
            raise ValueError(
                "measure_reset outcomes must be +1/-1 and match where."
            )
        return ("measure_reset", "".join(axes), where, outcomes)

    @staticmethod
    def control_event_parts(entry):
        """Return ``(name, payload, where)`` for an MPS-style control event."""
        return _mps_control_event_parts(entry)

    @staticmethod
    def is_control_event(entry):
        """Return whether ``entry`` is an MPS-style control event."""
        return _mps_control_event_parts(entry) is not None

    @staticmethod
    def submpo_event(submpo, where):
        """Return an explicit sub-MPO stream marker for TTN replay."""
        return ("submpo", submpo, normalize_submpo_where(where))

    @staticmethod
    def subtreempo_event(tree_mpo, where=None):
        """Return a Tree-native TreeMPO/TTNO stream marker.

        The operator carries the complete TreePlan geometry. ``where`` is
        therefore only a declared logical support for stream/layout metadata;
        when omitted, all physical sites of the operator's plan are used.
        Application validates that the declared support is either the complete
        TreeMPO site set or the operator's explicit ``operator_support`` before
        routing its internal TTNO bonds.
        """
        parts = _tree_mpo_event_parts(("subtreempo", tree_mpo, where))
        return ("subtreempo", tree_mpo, parts[1])

    subttno_event = subtreempo_event
    sub_treempo_event = subtreempo_event
    sub_tree_mpo_event = subtreempo_event

    @staticmethod
    def subtreempo_event_parts(entry):
        """Return ``(TreeMPO, declared_support)`` for a TreeMPO marker."""
        return _tree_mpo_event_parts(entry)

    subttno_event_parts = subtreempo_event_parts
    sub_treempo_event_parts = subtreempo_event_parts

    @staticmethod
    def is_subtreempo_event(entry):
        """Return whether ``entry`` is a Tree-native TreeMPO marker."""
        return _tree_mpo_event_parts(entry) is not None

    is_subttno_event = is_subtreempo_event
    is_sub_treempo_event = is_subtreempo_event

    @staticmethod
    def submpo_event_parts(entry):
        """Return ``(submpo, where)`` for an explicit marker, else ``None``."""
        return submpo_event_parts(entry, normalize_where=True)

    @staticmethod
    def is_submpo_event(entry):
        """Return whether ``entry`` is an explicit sub-MPO stream marker."""
        return submpo_event_parts(entry) is not None

    def apply_1q(self, gate, q, *, renormalize=False, track_norm=True):
        """Absorb a one-qubit gate into the site tensor of qubit ``q``."""
        self._invalidate_state_norm_cache()
        started = self._begin_update(
            "gate", _normalize_where(q), track_norm=track_norm
        )
        try:
            with self._thread_ctx():
                if self.mode == "dmrg":
                    result = self._apply_gate_dmrg_impl(
                        gate,
                        (q,),
                        renormalize=renormalize,
                        track_norm=track_norm,
                    )
                elif self.mode in {"tree_mpo_direct", "tree_mpo_dm"}:
                    result = self._apply_gate_tree_mpo_impl(
                        gate, (q,), renormalize=renormalize,
                        track_norm=track_norm,
                    )
                else:
                    result = self._apply_1q_impl(gate, q, renormalize=renormalize)
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        return result

    def _apply_1q_impl(self, gate, q, *, renormalize=False):
        """Apply a one-qubit gate without opening another thread context."""
        q = self._validate_qubit(q)
        gate = self._as_state_backend(gate)
        d = int(self.tn.ind_size(self._phys(q)))
        if _is_symmray_array(gate):
            # A Symmray gate is already a (d, d) symmetric operator; reshaping
            # into base-2 sub-legs would destroy its block/charge structure. Its
            # unitarity cannot be cheaply certified here, so take the always-safe
            # non-unitary branch (move the centre onto the site node first).
            unitary = False
        else:
            if tuple(ar.shape(gate)) != (d, d):
                gate = ar.do("reshape", gate, (d, d))
            gate_np = ar.to_numpy(gate)
            unitary = np.allclose(
                gate_np.conj().T @ gate_np, np.eye(d, dtype=gate_np.dtype),
                rtol=1e-10, atol=1e-12,
            )
        if self._active_update is not None:
            # A direct one-site call can be non-unitary. Do not report its
            # physical scale change as retained compression loss.
            self._active_update["track_norm"] = (
                bool(self._active_update.get("track_norm", True)) and unitary
            )
        site_node = self.plan.node_of_qubit[q]
        if not unitary:
            self._move_center(site_node)
        region = self.tn.canonical_region
        left_inds = self.tn.node_tensor(site_node).left_inds
        absorb_started = self._profile_phase_start()
        try:
            self.tn.gate_inds_(gate, [self._phys(q)], contract=True)
        finally:
            self._profile_phase_event(
                "tensor_absorption",
                absorb_started,
                support=(q,),
                route="one_site",
            )
        if unitary:
            # A physical unitary preserves the isometric exterior, but the
            # state-owned gate mutator deliberately invalidates metadata for
            # direct callers. Restore both the local proof and known region
            # only for this proven canonical-preserving operation.
            self.tn.node_tensor(site_node).modify(left_inds=left_inds)
            self.tn.canonical_region = region
        if not unitary:
            self.center = site_node
            if renormalize:
                self.normalize()
        return self

    def apply_2q(self, gate, qa, qb, *, track_norm=True):
        """Apply a two-qubit gate to physical sites ``qa`` and ``qb``.

        This low-level method retains the specialized two-factor path kernel
        for compatibility with callers that explicitly use ``apply_2q``.
        Ordinary ``apply_gate`` calls and bundled gate streams use the shared
        ``TreeMPO -> apply_subtreempo`` route, which factorizes on the active
        canonical Steiner region. ``tree_mpo_direct`` and ``tree_mpo_dm`` also
        select that route here; ``submpo`` remains reserved for explicit chain
        MPO events.
        """
        self._invalidate_state_norm_cache()
        logical_where = _normalize_where((qa, qb))
        self._validate_support(logical_where)
        if logical_where[0] == logical_where[1]:
            raise ValueError("A two-qubit gate needs two distinct qubits.")
        started = self._begin_update(
            "gate", logical_where, track_norm=track_norm
        )
        try:
            with self._thread_ctx():
                if self.mode == "dmrg":
                    result = self._apply_gate_dmrg_impl(
                        gate,
                        logical_where,
                        track_norm=track_norm,
                    )
                elif self.mode in {"tree_mpo_direct", "tree_mpo_dm"}:
                    result = self._apply_gate_tree_mpo_impl(
                        gate, logical_where, track_norm=track_norm
                    )
                else:
                    result = self._apply_2q_impl(gate, *logical_where)
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        return result

    @staticmethod
    def _as_gate_tensor4(gate, da, db):
        """Return a two-site gate as a rank-4 (out_a, out_b, in_a, in_b) tensor.

        Dense qubit gates arrive as a ``(da*db, da*db)`` matrix (or an already
        rank-4 array) and are reshaped to the ``(da, db, da, db)`` convention.
        A Symmray gate is already a rank-4 symmetric array whose legs carry the
        charge/fermion-sign structure, so it is returned untouched -- reshaping
        it into base-2 sub-legs would corrupt the block structure.
        """
        if _is_symmray_array(gate):
            if len(ar.shape(gate)) != 4:
                raise ValueError(
                    "A Symmray two-site gate must be a rank-4 "
                    "(out_a, out_b, in_a, in_b) array; got shape "
                    f"{tuple(ar.shape(gate))}."
                )
            return gate
        if tuple(ar.shape(gate)) == (da, db, da, db):
            return gate
        return ar.do("reshape", gate, (da, db, da, db))

    def _apply_2q_impl(self, gate, qa, qb, *, max_bond=None, cutoff=None):
        """Apply a two-site gate using the legacy low-level route.

        ``mode='direct'`` splits the gate locally and ``mode='mpo'`` lets Quimb
        make the equivalent two-tensor MPO. Both immediately enter the same
        two-factor attach/QR-thread/compress kernel. ``'auto'`` and ``'dm'``
        select local direct factorization. Ordinary ``apply_gate`` calls do not
        use this compatibility helper: they are lowered to a true TreeMPO and
        passed to :meth:`apply_subtreempo`. ``'submpo'`` is reserved for
        explicit sub-MPO stream events and cannot be used with a dense gate.
        """
        if self.mode == "submpo":
            raise ValueError(
                "mode='submpo' accepts explicit sub-MPO stream events, not "
                "ordinary dense gates; use mode='direct' or mode='mpo'."
            )
        if self.mode in {"tree_mpo_direct", "tree_mpo_dm"}:
            logical_where = _normalize_where((qa, qb))
            return self._apply_gate_tree_mpo_impl(
                gate, logical_where, track_norm=True
            )
        gate = self._as_state_backend(gate)
        if self.mode in {"auto", "direct"}:
            return self._apply_2q_path_thread_impl(
                gate, qa, qb, max_bond=max_bond, cutoff=cutoff
            )
        return self._apply_2q_mpo_impl(
            gate, qa, qb, max_bond=max_bond, cutoff=cutoff
        )

    def _apply_2q_mpo_impl(
        self, gate, qa, qb, *, max_bond=None, cutoff=None,
    ):
        """Apply a two-site gate as a Quimb sub-MPO without nested updates."""

        qa = self._validate_qubit(qa)
        qb = self._validate_qubit(qb)
        if qa == qb:
            raise ValueError("A two-qubit gate needs two distinct qubits.")
        self._check_operator_limits((qa, qb))

        if _is_symmray_array(gate):
            # ``TensorNetwork.ind_size`` currently reports the size of a
            # selected Symmray block after a native one-site gate, rather than
            # the complete graded physical dimension. The native rank-four
            # gate's explicit axes are authoritative for ``from_dense``.
            gate_shape = tuple(int(dim) for dim in ar.shape(gate))
            if len(gate_shape) != 4:
                raise ValueError(
                    "A native two-site gate must have rank four; got shape "
                    f"{gate_shape}."
                )
            da, db = gate_shape[:2]
        else:
            da = int(self.tn.ind_size(self._phys(qa)))
            db = int(self.tn.ind_size(self._phys(qb)))
        cache_source = gate
        gate = self._as_gate_tensor4(gate, da, db)
        factor_started = self._profile_phase_start()

        # ``MatrixProductOperator.from_dense`` is backend-generic: for a
        # Symmray FermionicArray its block-aware SVD returns native fermionic
        # MPO tensors, including virtual-sector data and graded index
        # orientations. Its Tensor.split default has a nonzero cutoff, so make
        # the gate factorisation explicitly lossless; ``chi``/``self.cutoff``
        # are applied only after the complete gate reaches the TTN path.
        cache_key = (
            "mpo",
            id(cache_source),
            _array_backend_signature(cache_source),
            _array_backend_signature(self._state_like()),
            qa,
            qb,
            da,
            db,
            self.n,
        )
        cached = self._gate_factor_cache.get(cache_key)
        cache_hit = cached is not None and cached[0] is cache_source
        if cached is not None and cached[0] is cache_source:
            submpo = cached[1]
        else:
            submpo = qtn.MatrixProductOperator.from_dense(
                gate, dims=(da, db), sites=(qa, qb), L=self.n,
                max_bond=None, cutoff=0.0,
            )
            self._cache_gate_factorization(cache_key, cache_source, submpo)
        factors = self._two_site_mpo_factors(submpo, qa, qb)
        if factors is None:
            raise TypeError(
                "two-site gate could not be represented as a Quimb sub-MPO "
                "with one local factor per requested site."
            )
        self._profile_phase_event(
            "gate_factorization",
            factor_started,
            route="mpo",
            cache_hit=cache_hit,
            support=(qa, qb),
            input_shape=(da, db, da, db),
        )
        return self._apply_2q_factors_impl(
            *factors,
            qa,
            qb,
            max_bond=max_bond,
            cutoff=cutoff,
            preserve_subcap=True,
        )

    def _cache_gate_factorization(self, key, source, value):
        """Store one immutable gate factorization in the bounded cache."""
        if key in self._gate_factor_cache:
            self._gate_factor_cache[key] = (source, value)
            return
        if len(self._gate_factor_cache) >= self._gate_factor_cache_limit:
            self._gate_factor_cache.pop(next(iter(self._gate_factor_cache)))
        self._gate_factor_cache[key] = (source, value)

    def _two_site_mpo_factors(self, submpo, qa, qb, *, site_where=None):
        """Extract and normalize a two-site Quimb MPO into local factors.

        The returned factors have exactly the same interface as a direct
        gate-SVD factorization: each has one output leg, one state physical
        input leg, and their shared operator-Schmidt bond. Crucially, factors
        are found by their site tags rather than MPO tensor order, so a
        descending support ``(qb, qa)`` remains correct.
        """
        if site_where is None:
            site_where = (qa, qb)
        else:
            site_where = tuple(site_where)
        if len(site_where) != 2:
            raise ValueError("a two-site MPO support must contain two sites.")
        site_a, site_b = site_where
        gen_sites = getattr(submpo, "gen_sites_present", None)
        site_tag = getattr(submpo, "site_tag", None)
        upper_id = getattr(submpo, "upper_ind_id", None)
        lower_id = getattr(submpo, "lower_ind_id", None)
        tag_map = getattr(submpo, "tag_map", None)
        tensor_map = getattr(submpo, "tensor_map", None)
        if not all((gen_sites, callable(site_tag), upper_id, lower_id,
                    tag_map is not None, tensor_map is not None)):
            return None
        try:
            if set(gen_sites()) != {site_a, site_b}:
                return None
            raw_factors = {}
            for qubit, site in ((qa, site_a), (qb, site_b)):
                tids = tuple(tag_map[site_tag(site)])
                if len(tids) != 1:
                    return None
                factor = tensor_map[tids[0]].copy()
                # The two-site fast path consumes these private factors
                # directly, before the structured sub-MPO route gets a
                # chance to coerce its payload tensors. Keep the caller's
                # MPO untouched while matching each factor to the live TTN
                # backend, dtype, and device.
                factor.modify(data=self._as_state_backend(factor.data))
                upper = upper_id.format(site)
                lower = lower_id.format(site)
                if upper not in factor.inds or lower not in factor.inds:
                    return None
                raw_factors[qubit] = (factor, upper, lower)
            bonds = tuple(qtn.bonds(
                raw_factors[qa][0], raw_factors[qb][0]
            ))
        except (KeyError, TypeError, ValueError):
            return None
        if len(bonds) != 1:
            return None

        cache_key = (
            "mpo_factors",
            id(submpo),
            _array_backend_signature(self._state_like()),
            qa,
            qb,
            site_where,
            self.n,
        )
        cached = self._gate_factor_cache.get(cache_key)
        if cached is not None and cached[0] is submpo:
            raw_factors, shared_bond = cached[1]
        else:
            raw_factors = {
                qubit: (factor.copy(), upper, lower)
                for qubit, (factor, upper, lower) in raw_factors.items()
            }
            shared_bond = bonds[0]
            self._cache_gate_factorization(
                cache_key, submpo, (raw_factors, shared_bond)
            )

        thread_ind = self.tn._new_work_bond("mpo_thread", qa, qb)
        factors = {}
        outputs = {}
        for qubit, (factor_template, upper, lower) in raw_factors.items():
            factor = factor_template.copy()
            # These factors are private to this update and are consumed before
            # they enter the live tree. A stable, private output label avoids
            # two UUID allocations and one metadata reindex per local factor;
            # the shared routed bond remains fresh because it enters the live
            # tree during threading.
            output = f"_pepsy_mpo_out_{qubit}"
            factor.reindex_({
                upper: output,
                lower: self._phys(qubit),
                shared_bond: thread_ind,
            })
            factors[qubit] = factor
            outputs[qubit] = output
        return factors, outputs, thread_ind

    def _apply_2q_path_thread_impl(
        self, gate, qa, qb, *, max_bond=None, cutoff=None,
    ):
        """Apply a two-qubit gate without opening another thread context."""
        gate = self._as_state_backend(gate)
        cache_source = gate
        qa = self._validate_qubit(qa)
        qb = self._validate_qubit(qb)
        if qa == qb:
            raise ValueError("A two-qubit gate needs two distinct qubits.")
        self._check_operator_limits((qa, qb))
        pa, pb = self._phys(qa), self._phys(qb)
        da, db = int(self.tn.ind_size(pa)), int(self.tn.ind_size(pb))
        gate = self._as_gate_tensor4(gate, da, db)
        factors, outputs, thread_ind = self._cached_direct_gate_factors(
            gate, cache_source, qa, qb, pa, pb, da, db,
        )
        return self._apply_2q_factors_impl(
            factors,
            outputs,
            thread_ind,
            qa,
            qb,
            max_bond=max_bond,
            cutoff=cutoff,
        )

    def _cached_direct_gate_factors(
        self, gate, source, qa, qb, pa, pb, da, db,
    ):
        """Return fresh-index copies of a cached direct gate factorization."""
        factor_started = self._profile_phase_start()
        key = (
            "direct",
            id(source),
            _array_backend_signature(source),
            _array_backend_signature(self._state_like()),
            qa,
            qb,
            da,
            db,
        )
        cached = self._gate_factor_cache.get(key)
        cache_hit = cached is not None and cached[0] is source
        if cached is not None and cached[0] is source:
            left_template, right_template = cached[1]
        else:
            template_out_a = "_pepsy_gate_out_a"
            template_out_b = "_pepsy_gate_out_b"
            template_in_a = "_pepsy_gate_in_a"
            template_in_b = "_pepsy_gate_in_b"
            template_thread = "_pepsy_gate_thread"
            gate_tensor = qtn.Tensor(
                gate,
                inds=(
                    template_out_a,
                    template_out_b,
                    template_in_a,
                    template_in_b,
                ),
            )
            left_template, right_template = gate_tensor.split(
                left_inds=(template_out_a, template_in_a),
                method="svd",
                cutoff=0.0,
                absorb="both",
                get="tensors",
                bond_ind=template_thread,
            )
            self._cache_gate_factorization(
                key, source, (left_template, right_template)
            )

        self._profile_phase_event(
            "gate_factorization",
            factor_started,
            route="direct",
            cache_hit=cache_hit,
            support=(qa, qb),
            input_shape=(da, db, da, db),
        )

        thread_ind = self.tn._new_work_bond("gate_thread", qa, qb)
        left = left_template.copy()
        right = right_template.copy()
        output_a = "_pepsy_gate_out_a"
        output_b = "_pepsy_gate_out_b"
        left.reindex_({
            "_pepsy_gate_in_a": pa,
            "_pepsy_gate_thread": thread_ind,
        })
        right.reindex_({
            "_pepsy_gate_in_b": pb,
            "_pepsy_gate_thread": thread_ind,
        })
        return (
            {qa: left, qb: right},
            {qa: output_a, qb: output_b},
            thread_ind,
        )

    def _apply_2q_factors_impl(
        self, factors, outputs, thread_ind, qa, qb, *, max_bond=None,
        cutoff=None, preserve_subcap=False,
    ):
        """Apply two local operator factors with one shared path-thread kernel.

        ``direct`` supplies factors from its gate SVD and ``mpo`` supplies the
        two factors Quimb made for its sub-MPO. From here onward they have
        identical handling: choose the closest source leaf, attach its factor,
        QR-thread the shared operator bond, attach the other factor, then make
        one canonical compression sweep.
        """
        plan = self.plan
        path_started = self._profile_phase_start()
        la, lb, path = self._cached_two_site_path(qa, qb)
        parent = plan.parent.get(la)
        if (
            plan.is_leaf(la)
            and plan.is_leaf(lb)
            and parent is not None
            and plan.parent.get(lb) == parent
        ):
            self._profile_phase_event(
                "metadata_path",
                path_started,
                support=(qa, qb),
                route="sibling",
                path_length=len(path),
            )
            return self._apply_2q_sibling_factors(
                factors, outputs, qa, qb, la, lb, parent,
                max_bond=max_bond,
                cutoff=cutoff,
                preserve_subcap=preserve_subcap,
            )

        source, destination = (
            (qa, qb)
            if self._nearest_anchor((la, lb)) == la
            else (qb, qa)
        )
        source_node = plan.node_of_qubit[source]
        destination_node = plan.node_of_qubit[destination]
        if source_node != path[0]:
            path = path[::-1]
        self._profile_phase_event(
            "metadata_path",
            path_started,
            support=(qa, qb),
            route="threaded",
            path_length=len(path),
            source=source_node,
            destination=destination_node,
        )
        self._move_center(source_node)
        self._thread_ind = thread_ind
        try:
            source_tensor = self.tn.tensor_map[self._tid(source_node)]
            absorb_started = self._profile_phase_start()
            try:
                merged_source = _contract_two_tensors(
                    source_tensor,
                    factors[source],
                    shared_ind=self._phys(source),
                ).reindex_({outputs[source]: self._phys(source)})
            finally:
                self._profile_phase_event(
                    "tensor_absorption",
                    absorb_started,
                    support=(source,),
                    route="threaded_source",
                )
            source_tensor.modify(
                data=merged_source.data, inds=merged_source.inds,
            )

            for u, v in zip(path, path[1:]):
                self._thread_hop(u, v)

            destination_tensor = self.tn.tensor_map[self._tid(destination_node)]
            absorb_started = self._profile_phase_start()
            try:
                merged_destination = _contract_two_tensors(
                    factors[destination],
                    destination_tensor,
                    shared_ind=self._phys(destination),
                ).reindex_({outputs[destination]: self._phys(destination)})
            finally:
                self._profile_phase_event(
                    "tensor_absorption",
                    absorb_started,
                    support=(destination,),
                    route="threaded_destination",
                )
            destination_tensor.modify(
                data=merged_destination.data, inds=merged_destination.inds,
            )
            self.center = destination_node
            self._compress_path(
                path,
                max_bond=max_bond,
                cutoff=cutoff,
                preserve_subcap=preserve_subcap,
            )
        finally:
            self._thread_ind = None
        return self

    def _apply_2q_sibling_factors(
        self, factors, outputs, qa, qb, la, lb, parent, *, max_bond=None,
        cutoff=None, preserve_subcap=False,
    ):
        """Apply two local factors to sibling leaves through their parent.

        There is no intermediate tree edge on which to thread the shared
        operator bond. Absorb both factors, contract the two leaves and parent
        into one blob (which consumes that bond), then make the usual two
        truncating splits. This is the sibling specialization of
        :meth:`_apply_2q_factors_impl`, shared by direct and MPO factors.
        """
        pa, pb = self._phys(qa), self._phys(qb)
        self._move_center(parent)
        tla = self.tn.tensor_map[self._tid(la)]
        tp = self.tn.tensor_map[self._tid(parent)]
        tlb = self.tn.tensor_map[self._tid(lb)]
        e_la = self._bond_name(la, parent)
        e_lb = self._bond_name(lb, parent)

        absorb_started = self._profile_phase_start()
        try:
            merged_a = _contract_two_tensors(
                tla, factors[qa], shared_ind=pa,
            ).reindex_(
                {outputs[qa]: pa}
            )
            merged_b = _contract_two_tensors(
                tlb, factors[qb], shared_ind=pb,
            ).reindex_(
                {outputs[qb]: pb}
            )
            blob = qtn.tensor_contract(merged_a, tp, merged_b)
        finally:
            self._profile_phase_event(
                "tensor_absorption",
                absorb_started,
                support=(qa, qb),
                route="sibling_blob",
            )

        # Split off leaf a (isometric), then leaf b, leaving the centre at the
        # parent; both new bonds keep their canonical tree-edge names.
        la_t, rem = self._split_with_diagnostics(
            blob, [pa], edge=(la, parent), bond_ind=e_la,
            max_bond=self.chi if max_bond is None else max_bond,
            cutoff=self.cutoff if cutoff is None else cutoff,
            preserve_subcap=preserve_subcap,
        )
        lb_t, p_t = self._split_with_diagnostics(
            rem, [pb], edge=(lb, parent), bond_ind=e_lb,
            max_bond=self.chi if max_bond is None else max_bond,
            cutoff=self.cutoff if cutoff is None else cutoff,
            preserve_subcap=preserve_subcap,
        )
        tla.modify(
            data=la_t.data,
            inds=la_t.inds,
            left_inds=la_t.left_inds,
        )
        tlb.modify(
            data=lb_t.data,
            inds=lb_t.inds,
            left_inds=lb_t.left_inds,
        )
        tp.modify(data=p_t.data, inds=p_t.inds, left_inds=None)
        self.center = parent
        return self

    def _thread_hop(self, u, v):
        """Thread the virtual bond exactly from node ``u`` to adjacent ``v``.

        The bond is moved by an *economical QR* factorisation (Seitz et al.,
        Fig. 6): the intermediate node keeps its isometric ``Q`` factor and the
        upper-triangular ``R`` carries the virtual bond forward to ``v``, moving
        the orthogonality centre with it.  QR is lossless and cheaper than the
        SVD used for the final truncating sweep, so a single gate grows a
        crossed bond by at most its operator-Schmidt rank (bounded by
        ``min(d_a**2, d_b**2)`` for local dimensions ``d_a`` and ``d_b``)
        above its pre-gate size, and that growth is undone by
        :meth:`_compress_path`.
        """
        profile_started = time.perf_counter() if self.profile else None
        before_bond = self.tn.bond(u, v)
        before_dim = int(self.tn.ind_size(before_bond))
        try:
            if self.tn.fermionic:
                result = self._fermionic_thread_hop(u, v)
            else:
                result = self._dense_thread_hop(u, v)
        finally:
            after_bond = self.tn.bond(u, v)
            after_dim = int(self.tn.ind_size(after_bond))
            self._record_transient_bond(
                after_dim, phase="thread_hop", edge=(u, v),
            )
            if self.profile:
                try:
                    thread_dim = int(self.tn.ind_size(self._thread_ind))
                except (KeyError, TypeError, ValueError):
                    thread_dim = None
                self.profile_events.append({
                    "kind": "thread_hop",
                    "update": (
                        None if self._active_update is None
                        else self._active_update.get("update")
                    ),
                    "edge": (u, v),
                    "before_bond": before_dim,
                    "after_bond": after_dim,
                    "thread_bond": thread_dim,
                    "native": bool(self.tn.fermionic),
                    "seconds": time.perf_counter() - profile_started,
                })
        return result

    def _dense_thread_hop(self, u, v):
        """Perform one dense QR thread hop without timing/diagnostic wrapping."""

        tu = self.tn.tensor_map[self._tid(u)]
        tv = self.tn.tensor_map[self._tid(v)]
        edge = next(iter(qtn.bonds(tu, tv)))
        left_inds = [ix for ix in tu.inds if ix not in (edge, self._thread_ind)]
        keep, carry = tu.split(
            left_inds=left_inds,
            method="qr",
            cutoff=0.0,
            absorb="right",
            get="tensors",
        )
        merged_v = _contract_two_tensors(carry, tv, shared_ind=edge)
        # ``keep`` is the exact Q factor pointing toward ``v``. Preserve that
        # isometry metadata so a later canonical walk can recognize the tensor
        # without repeating the same QR decomposition.
        tu.modify(
            data=keep.data,
            inds=keep.inds,
            left_inds=keep.left_inds,
        )
        tv.modify(
            data=merged_v.data,
            inds=merged_v.inds,
            left_inds=None,
        )

    def _fermionic_thread_hop(self, u, v):
        """QR-route the operator bond without leaving native graded arrays."""
        tu = self.tn.tensor_map[self._tid(u)]
        tv = self.tn.tensor_map[self._tid(v)]
        edge = self.tn.bond(u, v)
        left_inds = [
            index
            for index in tu.inds
            if index not in (edge, self._thread_ind)
        ]
        right_inds = [
            index
            for index in tu.inds
            if index not in left_inds
        ]
        keep, carry = self.tn._native_qr_split(
            tu,
            left_inds=left_inds,
            right_inds=right_inds,
            absorb="right",
            cutoff=0.0,
            get="tensors",
            bond_ind=self.tn._new_work_bond("thread_hop", u, v),
        )
        merged_v = _contract_two_tensors(carry, tv, shared_ind=edge)
        tu.modify(
            data=keep.data,
            inds=keep.inds,
            left_inds=keep.left_inds,
        )
        tv.modify(
            data=merged_v.data,
            inds=merged_v.inds,
            left_inds=None,
        )

    @staticmethod
    def _tensor_ind_size(tensor, ind):
        """Return the dimension of ``ind`` on ``tensor``."""
        return int(tensor.shape[tensor.inds.index(ind)])

    @staticmethod
    def _split_rank_bound(tensor, left_inds):
        """Return the matrix-rank upper bound for a tensor split."""
        left_inds = set(left_inds)
        left_dim = int(np.prod([
            TreeOptimizer._tensor_ind_size(tensor, ind) for ind in left_inds
        ], dtype=int))
        right_dim = int(np.prod([
            TreeOptimizer._tensor_ind_size(tensor, ind)
            for ind in tensor.inds if ind not in left_inds
        ], dtype=int))
        return min(left_dim, right_dim)

    @staticmethod
    def _spectrum_to_numpy(values):
        """Flatten a dense or block-sparse singular spectrum by magnitude."""
        if hasattr(values, "blocks"):
            parts = [
                np.abs(np.asarray(ar.to_numpy(block))).ravel()
                for block in values.blocks.values()
            ]
            spectrum = (
                np.concatenate(parts)
                if parts
                else np.empty(0, dtype=float)
            )
        else:
            spectrum = np.abs(np.asarray(ar.to_numpy(values))).ravel()
        # Keep a deterministic ordering for dense diagnostics and rank display.
        return np.sort(spectrum.astype(float, copy=False))[::-1]

    @staticmethod
    def _spectrum_payload(values, kept_values=None):
        """Package full/kept spectra for exact native truncation diagnostics."""
        payload = {"values": TreeOptimizer._spectrum_to_numpy(values)}
        if hasattr(values, "blocks") and kept_values is not None:
            payload["kept_values"] = TreeOptimizer._spectrum_to_numpy(
                kept_values
            )
        return payload

    @staticmethod
    def _probe_bond_spectrum(
        ta, tb, *, max_bond=None, cutoff=0.0, cutoff_mode="rel",
    ):
        """Return full/kept singular spectra for a two-tensor bond."""
        info = {}
        qtn.tensor_compress_bond(
            ta.copy(), tb.copy(), max_bond=None, cutoff=0.0,
            cutoff_mode=cutoff_mode, absorb=None, info=info,
        )
        values = info.get("singular_values")
        if values is None:
            return {"values": np.empty(0, dtype=float)}
        kept_values = None
        if hasattr(values, "blocks") and (
            max_bond is not None or float(cutoff) != 0.0
        ):
            kept_info = {}
            qtn.tensor_compress_bond(
                ta.copy(), tb.copy(), max_bond=max_bond,
                cutoff=cutoff, cutoff_mode=cutoff_mode, absorb=None,
                info=kept_info,
            )
            kept_values = kept_info.get("singular_values")
        return TreeOptimizer._spectrum_payload(values, kept_values)

    @staticmethod
    def _probe_split_spectrum(
        tensor, left_inds, *, max_bond=None, cutoff=0.0,
        cutoff_mode="rel",
    ):
        """Return full/kept singular spectra for a tensor split."""
        _, values, _ = tensor.split(
            left_inds=left_inds,
            method="svd",
            cutoff=0.0,
            max_bond=None,
            cutoff_mode=cutoff_mode,
            absorb=None,
            get="arrays",
        )
        kept_values = None
        if hasattr(values, "blocks") and (
            max_bond is not None or float(cutoff) != 0.0
        ):
            _, kept_values, _ = tensor.split(
                left_inds=left_inds,
                method="svd",
                cutoff=cutoff,
                max_bond=max_bond,
                cutoff_mode=cutoff_mode,
                absorb=None,
                get="arrays",
            )
        return TreeOptimizer._spectrum_payload(values, kept_values)

    def _record_truncation(
        self, *, kind, edge, before_bond, after_bond, bond_ind,
        full_spectrum=None, max_bond=None, cutoff=None,
    ):
        """Record one edge split/compression and optional discarded weight."""
        self._record_transient_bond(
            before_bond, phase=str(kind), edge=edge,
        )
        if not self.record_history:
            return
        discarded_weight = None
        discarded_fraction = None
        spectrum_norm_sq = None
        spectrum_rank = None
        if full_spectrum is not None:
            if isinstance(full_spectrum, dict):
                spectrum = np.asarray(
                    full_spectrum["values"], dtype=float,
                ).ravel()
                kept = full_spectrum.get("kept_values")
            else:
                spectrum = np.asarray(full_spectrum, dtype=float).ravel()
                kept = None
            spectrum_rank = int(spectrum.size)
            spectrum_norm_sq = float(np.sum(spectrum * spectrum))
            if kept is not None:
                kept = np.asarray(kept, dtype=float).ravel()
                kept_norm_sq = float(np.sum(kept * kept))
                discarded_weight = max(0.0, spectrum_norm_sq - kept_norm_sq)
            else:
                discarded = spectrum[int(after_bond):]
                discarded_weight = float(np.sum(discarded * discarded))
            if spectrum_norm_sq > 0.0:
                discarded_fraction = float(discarded_weight / spectrum_norm_sq)
            else:
                discarded_fraction = 0.0

        self.truncation_history.append({
            "kind": str(kind),
            "edge": tuple(int(x) for x in edge),
            "bond": bond_ind,
            "before_bond": int(before_bond),
            "after_bond": int(after_bond),
            "truncated": bool(after_bond < before_bond),
            "spectrum_rank": spectrum_rank,
            "spectrum_norm_sq": spectrum_norm_sq,
            "discarded_weight": discarded_weight,
            "discarded_fraction": discarded_fraction,
            # ``None`` is meaningful: native MPO routing uses an uncapped,
            # lossless split before the final ``chi``-limited path sweep.
            "max_bond": None if max_bond is None else int(max_bond),
            "cutoff": float(self.cutoff if cutoff is None else cutoff),
            "cutoff_mode": self.cutoff_mode,
        })

    def _split_with_diagnostics(
        self, tensor, left_inds, *, edge, bond_ind, max_bond, cutoff,
        preserve_subcap=False,
    ):
        """Split ``tensor`` and record the resulting virtual edge."""
        before_bond = self._split_rank_bound(tensor, left_inds)
        if preserve_subcap:
            cutoff = self._subtree_cutoff_for_size(
                before_bond, max_bond=max_bond, cutoff=cutoff,
            )
        # A zero-cutoff split whose rank bound is already within the requested
        # cap cannot truncate. Use QR in that case, including capped splits:
        # the previous uncapped-only condition needlessly sent sibling leaf
        # updates through a full SVD even when the cap could not bind.
        lossless = (
            float(cutoff) == 0.0
            and (max_bond is None or before_bond <= max_bond)
        )
        full_spectrum = (
            self._probe_split_spectrum(
                tensor,
                left_inds,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=self.cutoff_mode,
            )
            if self.track_truncation and not lossless else None
        )
        if lossless:
            left, right = self.tn._native_qr_split(
                tensor,
                left_inds=left_inds,
                cutoff=0.0,
                absorb="right",
                get="tensors",
                bond_ind=bond_ind,
            )
        else:
            left, right = tensor.split(
                left_inds=left_inds, method="svd", max_bond=max_bond,
                cutoff=cutoff, cutoff_mode=self.cutoff_mode,
                absorb="right", get="tensors", bond_ind=bond_ind,
            )
        after_bond = self._tensor_ind_size(left, bond_ind)
        self._record_truncation(
            kind="split", edge=edge, before_bond=before_bond,
            after_bond=after_bond, bond_ind=bond_ind,
            full_spectrum=full_spectrum,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        return left, right

    def _subtree_cutoff_for_size(self, before_bond, *, max_bond, cutoff):
        """Return the cutoff for a Tree/MPO update at ``before_bond`` size."""
        requested_cutoff = self.cutoff if cutoff is None else float(cutoff)
        effective_max_bond = (
            self.chi
            if max_bond is None
            else self._normalize_max_bond(max_bond)
        )
        if (
            effective_max_bond is not None
            and int(before_bond) <= effective_max_bond
        ):
            return 0.0
        return requested_cutoff

    def _compress_edge_with_diagnostics(
        self, u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False, compression_mode=None, compression_seed=None,
    ):
        """Compress one live tree edge and record its truncation diagnostics."""
        profile_started = time.perf_counter() if self.profile else None
        max_bond = (
            self.chi if max_bond is None else self._normalize_max_bond(max_bond)
        )
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        compression_mode = self._normalize_compression_mode(
            self.compression_mode if compression_mode is None else compression_mode
        )
        if compression_seed is None:
            compression_seed = self.compression_seed
        bond_before = self.tn.bond(u, v)
        before_bond = int(self.tn.ind_size(bond_before))
        lossless = (
            cutoff == 0.0
            and (max_bond is None or before_bond <= max_bond)
        )
        if lossless:
            # No singular value can be removed in this case. A lossless QR
            # still moves the centre across the edge, but avoids both the
            # diagnostic spectrum probe and the compression SVD.
            self.tn.canonize_edge_(u, v, absorb="right")
            bond_after = self.tn.bond(u, v)
            self._record_truncation(
                kind="canonize", edge=(u, v), before_bond=before_bond,
                after_bond=int(self.tn.ind_size(bond_after)),
                bond_ind=bond_after, max_bond=max_bond, cutoff=cutoff,
            )
            if profile_started is not None:
                self.profile_events.append({
                    "kind": "edge_canonize",
                    "update": (
                        None if self._active_update is None
                        else self._active_update.get("update")
                    ),
                    "edge": (u, v),
                    "before_bond": before_bond,
                    "after_bond": int(self.tn.ind_size(bond_after)),
                    "seconds": time.perf_counter() - profile_started,
                })
            return
        full_spectrum = None
        if self.track_truncation:
            self._warn_track_truncation_slow()
            ta = self.tn.tensor_map[self._tid(u)]
            tb = self.tn.tensor_map[self._tid(v)]
            full_spectrum = self._probe_bond_spectrum(
                ta,
                tb,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=self.cutoff_mode,
            )

        # Keep the live canonical-region metadata in one place: the TTN edge
        # wrapper performs the compression and advances its tracked centre.
        self.tn.compress_edge_(
            u, v, max_bond=max_bond, cutoff=cutoff, absorb="right",
            cutoff_mode=self.cutoff_mode, reduced=reduced,
            compression_mode=compression_mode,
            compression_seed=compression_seed,
            _reduction_proven=reduction_proven,
        )
        bond_after = self.tn.bond(u, v)
        after_bond = int(self.tn.ind_size(bond_after))
        self._record_truncation(
            kind="compress", edge=(u, v), before_bond=before_bond,
            after_bond=after_bond, bond_ind=bond_after,
            full_spectrum=full_spectrum, max_bond=max_bond, cutoff=cutoff,
        )
        if profile_started is not None:
            self.profile_events.append({
                "kind": "edge_compress",
                "update": (
                    None if self._active_update is None
                    else self._active_update.get("update")
                ),
                "edge": (u, v),
                "before_bond": before_bond,
                "after_bond": after_bond,
                "reduced": reduced,
                "native": bool(self.tn.fermionic),
                "seconds": time.perf_counter() - profile_started,
            })

    def _compress_edge_compat(
        self, u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False,
    ):
        """Call the compression hook while supporting older overrides.

        Tree stabilizer and diagnostic integrations can wrap the private
        compression hook. Keep wrappers with the older signature working
        while the built-in hook receives the proof flag.
        """
        method = self._compress_edge_with_diagnostics
        try:
            parameters = inspect.signature(method).parameters.values()
            supports_proof = any(
                parameter.name == "reduction_proven"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            supports_compression_mode = any(
                parameter.name == "compression_mode"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            supports_compression_seed = any(
                parameter.name == "compression_seed"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            supports_proof = True
            supports_compression_mode = True
            supports_compression_seed = True
        kwargs = {
            "max_bond": max_bond,
            "cutoff": cutoff,
            "reduced": reduced,
        }
        if supports_proof:
            kwargs["reduction_proven"] = reduction_proven
        if supports_compression_mode:
            kwargs["compression_mode"] = self.compression_mode
        if supports_compression_seed:
            kwargs["compression_seed"] = self.compression_seed
        return method(u, v, **kwargs)

    def _metadata_aware_reduction(self, u, v):
        """Choose one-sided compression when ``v`` is proven isometric.

        Every caller compresses ``u -> v`` with ``absorb="right"``. If the
        live ``left_inds`` on ``v`` prove that it is already isometric toward
        ``u``, the compression kernel can SVD only ``u`` and reuse ``v``
        directly. Native graded metadata is accepted only after the
        TreeTensorNetwork charge alignment guard; missing or malformed
        metadata falls back to the usual two-sided reduced compression.
        """
        if self.tn.can_skip_canonize(u, v, absorb="left"):
            return "left"
        return True

    def _edge_reduction(self, u, v, *, max_bond, cutoff):
        """Return ``(reduction, proof)`` for one compression edge.

        A lossless edge only needs a QR move, so asking the native tensor to
        validate a truncating reduction hint would be wasted work. For a
        truncating native edge, the proof is passed through to the low-level
        compressor so the same expensive charge-map check is not repeated.
        """
        requested_cutoff = self.cutoff if cutoff is None else float(cutoff)
        effective_max_bond = (
            self.chi
            if max_bond is None
            else self._normalize_max_bond(max_bond)
        )
        if requested_cutoff == 0.0 and (
            effective_max_bond is None
            or int(self.tn.ind_size(self.tn.bond(u, v))) <= effective_max_bond
        ):
            # The low-level edge routine will take its lossless QR branch and
            # never consume this hint. Keep the one-sided marker for
            # diagnostics/observers while avoiding the proof lookup entirely.
            return "left", False
        reduced = self._metadata_aware_reduction(u, v)
        return reduced, reduced == "left"

    def _compress_path(
        self, path, *, max_bond=None, cutoff=None, preserve_subcap=False,
    ):
        """Canonically compress every bond along ``path`` down to ``chi``.

        The orthogonality centre sits at ``path[-1]`` on entry; sweeping back to
        ``path[0]`` with ``absorb="right"`` truncates each bond with an isometric
        environment (``canonize_distance=0`` local reduced compression) and
        leaves the centre at ``path[0]``.  This is the re-orthonormalisation
        sweep of Seitz et al. (Fig. 6) applied along the gate geodesic.
        """
        # Every node before the destination was produced by the lossless QR
        # threading sweep.  Its ``left_inds`` therefore prove that it is
        # isometric toward the destination side of the next compression edge.
        # Carry this proof through the reverse sweep instead of asking every
        # native edge to revalidate the same charge maps.  The proof is local
        # to this update and is not used by public arbitrary edge callers.
        for v, u in zip(path[::-1], path[-2::-1]):
            edge_cutoff = cutoff
            if preserve_subcap:
                edge_cutoff = self._subtree_cutoff_for_size(
                    self.tn.ind_size(self.tn.bond(v, u)),
                    max_bond=max_bond,
                    cutoff=cutoff,
                )
            reduced, reduction_proven = "left", True
            self._compress_edge_compat(
                v, u, max_bond=max_bond, cutoff=edge_cutoff,
                reduced=reduced, reduction_proven=reduction_proven,
            )
        self.center = path[0]

    def _compress_subtree(
        self, snodes, hub, *, max_bond=None, cutoff=None,
        preserve_subcap=True,
    ):
        """Canonically compress every edge of a connected updated subtree.

        Starting at ``hub``, descend each branch. Compressing ``node -> child``
        moves the centre onto the child; a lossless QR move returns it before
        the next branch. Thus every actual SVD sees the completed operator
        update with an isometric environment, while every affected edge is
        compressed exactly once.
        """
        snodes = frozenset(snodes)
        self._move_center(hub)

        def edge_cutoff(node, child):
            """Keep existing sub-cap bonds lossless during subtree replay.

            The routed subtree already contains the complete state and MPO
            update. Reapplying a positive cutoff to a bond that is still
            within the active bond cap can repeatedly remove tiny
            *pre-existing* state components on every MPO event. Keep this
            Tree/MPO route stable by using the configured Quimb cutoff mode on
            over-cap bonds, and using a lossless QR on bonds that remain within
            the cap.
            """
            if not preserve_subcap:
                return self.cutoff if cutoff is None else float(cutoff)
            return self._subtree_cutoff_for_size(
                self.tn.ind_size(self.tn.bond(node, child)),
                max_bond=max_bond, cutoff=cutoff,
            )

        def descend(node, parent):
            children = sorted(
                neighbor
                for neighbor in self._neighbors(node)
                if neighbor in snodes and neighbor != parent
            )
            for child in children:
                child_cutoff = edge_cutoff(node, child)
                reduced, reduction_proven = self._edge_reduction(
                    node, child, max_bond=max_bond, cutoff=child_cutoff,
                )
                self._compress_edge_compat(
                    node,
                    child,
                    max_bond=max_bond,
                    cutoff=child_cutoff,
                    reduced=reduced,
                    reduction_proven=reduction_proven,
                )
                descend(child, node)
                canonize_bond = int(
                    self.tn.ind_size(self.tn.bond(child, node))
                )
                canonize_started = self._profile_phase_start()
                try:
                    self.tn.canonize_edge_(child, node, absorb="right")
                finally:
                    self._profile_phase_event(
                        "edge_canonize",
                        canonize_started,
                        edge=(child, node),
                        before_bond=canonize_bond,
                        after_bond=int(
                            self.tn.ind_size(self.tn.bond(child, node))
                        ),
                    )

        descend(hub, None)
        self.center = hub

    def _qr_route_message(self, tensor, left_inds, *, bond_ind):
        """Split one subtree message losslessly while carrying operator legs."""
        return self.tn._native_qr_split(
            tensor,
            left_inds=left_inds,
            absorb="right",
            get="tensors",
            bond_ind=bond_ind,
        )

    def _route_subtree_messages(
        self, local, state_inds, operator_inds, order, *, token,
        workers=None,
    ):
        """QR-route open MPO bonds from subtree leaves to their common hub."""
        if workers is None:
            workers = self.subtree_workers
        workers = self._positive_limit(workers, "subtree_workers")
        # Native graded contractions remain deliberately serial. The QR
        # factorization is algebraically independent on a peel wave, but
        # Symmray's global index/phase bookkeeping has not been proven
        # thread-safe and correctness takes precedence over throughput there.
        if self.tn.fermionic:
            workers = 1

        pending = list(order)
        pool = None
        if workers > 1 and not self.tn.fermionic and len(order) > 1:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(
                max_workers=min(workers, len(order)),
                thread_name_prefix="pepsy-ttn-qr",
            )

        try:
            while pending:
                pending_destinations = {v for _, v in pending}
                ready = [
                    (index, u, v)
                    for index, (u, v) in enumerate(pending)
                    if u not in pending_destinations
                ]
                if not ready:
                    raise RuntimeError(
                        "subtree peel order contains a cyclic message dependency."
                    )
                if workers == 1:
                    ready = ready[:1]

                def split_message(item):
                    index, u, v = item
                    state_bond = self.tn.bond(u, v)
                    left_inds = [
                        ix for ix in local[u].inds
                        if ix != state_bond and ix not in operator_inds[u]
                    ]
                    new_bond = f"_ttn_mpo_route_{token}_{u}_{v}"
                    hop_started = self._profile_phase_start()
                    try:
                        kept, message = self._qr_route_message(
                            local[u], left_inds, bond_ind=new_bond,
                        )
                    finally:
                        self._profile_phase_event(
                            "thread_hop",
                            hop_started,
                            edge=(u, v),
                            route="subtreempo",
                        )
                    return index, u, v, state_bond, new_bond, kept, message

                if pool is not None and len(ready) > 1:
                    results = list(pool.map(split_message, ready))
                else:
                    results = [split_message(item) for item in ready]

                # Keep worker results deterministic, then merge all messages
                # landing on the same destination in one contraction. The
                # old serial loop contracted a busy hub once per child, which
                # repeatedly rebuilt the same destination tensor and paid the
                # contraction dispatch/reindex bookkeeping for every message.
                # Messages in one ready wave have disjoint source edges and
                # are therefore independent until this grouped merge.
                results = sorted(results)
                by_destination = {}
                for result in results:
                    by_destination.setdefault(result[2], []).append(result)

                for destination, destination_results in by_destination.items():
                    messages = [result[-1] for result in destination_results]
                    merge_started = self._profile_phase_start()
                    try:
                        if len(messages) == 1:
                            local[destination] = qtn.tensor_contract(
                                local[destination], messages[0]
                            )
                        else:
                            local[destination] = qtn.tensor_contract(
                                local[destination], *messages
                            )
                    finally:
                        self._profile_phase_event(
                            "subtree_hub_merge",
                            merge_started,
                            destination=destination,
                            message_count=len(messages),
                        )
                    for (
                        _, source, _, state_bond, new_bond, kept, _
                    ) in destination_results:
                        # The source tensors are private QR outputs and can
                        # be installed independently of the destination
                        # contraction.
                        local[source] = kept
                        state_inds[destination].discard(state_bond)
                        state_inds[destination].add(new_bond)
                    operator_inds[destination] = (
                        set(local[destination].inds)
                        - state_inds[destination]
                    )
                removed = {index for index, *_ in results}
                pending = [
                    edge for index, edge in enumerate(pending)
                    if index not in removed
                ]
        finally:
            if pool is not None:
                pool.shutdown()

    def _zipup_subtree_messages(self, local, state_inds, order, hub, *, max_bond, cutoff):
        """Contract and truncate one layered tree node at a time toward a hub.

        Each outgoing message has one retained state leg capped by max_bond
        and the original state/operator legs toward its unvisited parent.
        Unlike direct compression, this truncation precedes contraction of
        the complete operator. Its right environment is not canonical, so
        local discarded weights are not global fidelity errors.
        """
        physical_map = {self._phys(q) + "*": self._phys(q) for q in range(self.n)}
        for u, v in order:
            tensor = qtn.tensor_contract(*local[u]).reindex(physical_map)
            state_bond = self.tn.bond(u, v)
            left_inds = tuple(ix for ix in tensor.inds
                              if ix in state_inds[u] and ix != state_bond)
            new_bond = qtn.rand_uuid()
            before_bond = self._split_rank_bound(tensor, left_inds)
            if cutoff == 0.0 and (max_bond is None or before_bond <= max_bond):
                kept, message = self._qr_route_message(
                    tensor, left_inds, bond_ind=new_bond,
                )
            else:
                kept, message = tensor.split(
                    left_inds=left_inds, method="svd", absorb="right",
                    max_bond=max_bond, cutoff=cutoff, cutoff_mode=self.cutoff_mode,
                    get="tensors", bond_ind=new_bond,
                )
            local[u] = kept
            local[v].append(message)
            state_inds[v].discard(state_bond)
            state_inds[v].add(new_bond)
            self._record_truncation(
                kind="zipup", edge=(u, v), before_bond=before_bond,
                after_bond=kept.ind_size(new_bond), bond_ind=new_bond,
                max_bond=max_bond, cutoff=cutoff,
            )
        local[hub] = qtn.tensor_contract(*local[hub]).reindex(physical_map)
        if self.tn.fermionic and any(not tensor.data.blocks for tensor in local.values()):
            # Early independent branch cuts can remove every combination
            # compatible with the hub charge. Do not install an empty native
            # array, whose absent blocks also erase its backend/dtype evidence.
            raise ValueError(
                "zipup left no compatible charge blocks; increase chi or use "
                "mode='direct' to truncate with the complete operator environment"
            )

    def _install_routed_subtree(self, local, snodes, hub):
        """Install routed tensors and recover their proven hub centre.

        Dense routing already QR-isometrizes every peeled non-hub tensor toward
        ``hub``. Retaining each Q factor's ``left_inds`` lets Quimb's canonical
        recovery walk short-circuit those decompositions while still advancing
        the canonical-region state machine honestly. Native graded routing
        retains the same metadata when Symmray supplied it; the
        :class:`TreeTensorNetwork` predicate validates the charge maps and
        falls back to explicit graded QR otherwise.
        """
        for nid in snodes:
            routed = local[nid]
            modify_opts = {
                "data": routed.data,
                "inds": routed.inds,
            }
            if nid == hub:
                # The accumulated operator and state norm live here.
                modify_opts["left_inds"] = None
            elif routed.left_inds is not None:
                # Both dense and native QR return a Q factor with a live
                # isometry proof. The network-level predicate performs the
                # stricter native charge-map check when this is installed.
                modify_opts["left_inds"] = routed.left_inds
            elif not self.tn.fermionic:
                raise RuntimeError(
                    "dense subtree routing lost QR isometry metadata "
                    f"for non-hub node {nid}."
                )
            self.tn.tensor_map[self._tid(nid)].modify(**modify_opts)

        self.tn.canonical_region = frozenset(snodes)
        self._move_center(hub)

    # -- general multi-qubit / sub-MPO application ----------------------------

    def apply_subtree_operator(
        self, op, where, *, max_bond=None, cutoff=None,
        renormalize=False, track_norm=True,
    ):
        """Apply a subtree operator and aggregate its edge truncations.

        Set ``track_norm=False`` for a known non-unitary/Kraus operator so its
        physical norm change is not reported as compression loss.
        """
        self._warn_track_truncation_slow()
        self._invalidate_state_norm_cache()
        started = self._begin_update(
            "subtree", _normalize_where(where), track_norm=track_norm
        )
        try:
            if self.mode in {"dmrg", "zipup"}:
                from .operators import TreeMPO

                logical_where = _normalize_where(where)
                compact_where = self._validate_support(logical_where)
                self._check_operator_limits(compact_where)
                if len(set(compact_where)) != len(compact_where):
                    raise ValueError("apply_subtree_operator needs distinct qubits")
                operator = TreeMPO.from_gate(
                    self.plan, op, compact_where,
                    fermionic=self.tn.fermionic,
                    symmetry=self.tn.symmetry, dtype=self.backend_dtype,
                )
                result = self.apply_subtreempo(
                    operator, compact_where, max_bond=max_bond, cutoff=cutoff,
                    track_norm=track_norm, _validate_backend=False,
                )
                if renormalize:
                    self.normalize()
                if started:
                    self._finish_update()
                return result
            result = self._apply_subtree_operator_impl(
                op, where, max_bond=max_bond, cutoff=cutoff,
                renormalize=renormalize,
            )
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        return result

    def apply_subtreempo(
        self,
        tree_mpo,
        where=None,
        *,
        max_bond=None,
        cutoff=None,
        track_norm=True,
        _validate_backend=True,
    ):
        """Apply a complete TreeMPO/TTNO without lowering it to a chain MPO.

        ``tree_mpo`` must use the same :class:`TreePlan` as this state and
        contain one primary TTNO network.  The operator's virtual bonds are
        contracted on the Tree geometry itself: each state/operator site is
        absorbed locally, open operator bonds are QR-routed from the leaves to
        a common hub, and the affected state bonds are compressed once after
        the complete TreeMPO has arrived.  This is the Tree-native analogue of
        applying a sub-MPO, not a call into an MPS backend.
        """
        self._warn_track_truncation_slow()
        self._invalidate_state_norm_cache()
        # Direct TreeMPO calls obey the same no-implicit-transfer boundary as
        # stream replay. Generated internal operators use explicit coercion at
        # their construction sites instead.
        if _validate_backend:
            self._validate_gate_stream_backend([tree_mpo], ["subtreempo"])
        else:
            tree_mpo = self._coerce_tree_mpo_backend(tree_mpo)
        plan = getattr(tree_mpo, "plan", None)
        networks = getattr(tree_mpo, "tree_networks", None)
        if plan is None or networks is None:
            raise TypeError(
                "apply_subtreempo requires a TreeMPO/TTNO payload with a TreePlan."
            )
        if not _same_tree_plan(self.plan, plan):
            raise ValueError("TreeMPO and state use different TreePlans.")
        networks = tuple(networks)
        if len(networks) != 1:
            raise NotImplementedError(
                "apply_subtreempo currently requires one TreeMPO network; "
                "multi-sector TreeMPO expectation remains supported separately."
            )
        sites = tuple(sorted(self.plan.node_of_qubit))
        declared = sites if where is None else _normalize_where(where)
        if len(set(declared)) != len(declared):
            raise ValueError(
                f"TreeMPO application support repeats a site: {declared!r}."
            )
        operator_support = getattr(tree_mpo, "operator_support", None)
        if operator_support is not None:
            operator_support = tuple(sorted(_normalize_where(operator_support)))
            if any(site not in sites for site in operator_support):
                raise ValueError(
                    "TreeMPO operator_support contains sites outside its "
                    f"TreePlan: {operator_support!r}."
                )
        if tuple(sorted(declared)) == sites:
            active_support = sites if operator_support is None else operator_support
        elif operator_support is not None and set(declared) == set(operator_support):
            # A complete TreeMPO may be declared by its non-identity support.
            # Identity legs outside that support are validated and stripped
            # below before the minimal Steiner route is constructed.
            active_support = operator_support
        else:
            raise ValueError(
                "a TreeMPO application must declare every physical site of its "
                "TreePlan, or exactly its known operator_support; got "
                f"{declared!r} for sites {sites!r}."
            )
        if bool(getattr(tree_mpo, "fermionic", False)) != bool(
            getattr(self.tn, "fermionic", False)
        ):
            raise TypeError(
                "TreeMPO and TreeTensorNetwork must agree on the fermionic "
                "backend when applying a TreeMPO."
            )
        if bool(getattr(self.tn, "fermionic", False)) and (
            getattr(tree_mpo, "symmetry", None)
            != getattr(self.tn, "symmetry", None)
        ):
            raise TypeError(
                "native TreeMPO and TreeTensorNetwork must use the same "
                f"symmetry, got operator={getattr(tree_mpo, 'symmetry', None)!r} "
                f"and state={getattr(self.tn, 'symmetry', None)!r}."
            )
        if hasattr(tree_mpo, "validate"):
            tree_mpo.validate()

        max_bond = self.chi if max_bond is None else self._normalize_max_bond(max_bond)
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")
        started = self._begin_update(
            "subtreempo", declared, track_norm=track_norm
        )
        try:
            if self.mode == "dmrg":
                # Keep the exact DMRG target in operator--state form. The
                # TreeMPO is still the source of the complete
                # ``sub_treempo @ tree_state`` action, but its virtual layer
                # is not fused into the fitted target.
                active_nodes = tuple(
                    self.plan.node_of_qubit[site] for site in active_support
                )
                target = _build_layered_operator_state_target(
                    self.tn, tree_mpo
                )
                region = frozenset(self.tn.steiner_nodes(active_nodes))
                target_norm = None
                if track_norm and self._norm_tracking_enabled:
                    if self.center is None:
                        self._move_center(min(region))
                    target_norm = TreeFIT._center_norm_stripped(self.tn)[:2]
                self._run_tree_fit(
                    target,
                    region,
                    active_support,
                    operator=tree_mpo,
                    target_norm=target_norm,
                )
                if started:
                    self._finish_update()
                return self
            with self._thread_ctx():
                active_nodes = tuple(
                    self.plan.node_of_qubit[site] for site in active_support
                )
                snodes = (
                    frozenset(self.plan.nodes())
                    if tuple(sorted(active_support)) == sites
                    else self._steiner_nodes(active_nodes)
                )
                order, hub = self._peel_order(snodes)
                if order:
                    path_started = self._profile_phase_start()
                    self._profile_phase_event(
                        "metadata_path",
                        path_started,
                        support=tuple(active_support),
                        route="subtreempo",
                        subtree_nodes=len(snodes),
                        message_edges=len(order),
                        hub=hub,
                    )
                self._move_center(hub)
                local = {}
                state_inds = {}
                operator_inds = {}
                for nid in snodes:
                    state_t = self.tn.tensor_map[self._tid(nid)].copy()
                    state_inds[nid] = set(state_t.inds)
                    op_t = tree_mpo.node_tensor(nid).copy()
                    # The state plan is the validated routing authority. Use
                    # its adjacency here so compatible TreeMPO views need only
                    # expose node tensors, not duplicate the neighbor API.
                    for neighbor in self._neighbors(nid):
                        if neighbor in snodes:
                            continue
                        shared = qtn.bonds(
                            op_t, tree_mpo.node_tensor(neighbor),
                        )
                        if len(shared) != 1:
                            raise ValueError(
                                "TreeMPO boundary must have one virtual bond "
                                f"on edge {(nid, neighbor)!r}."
                            )
                        edge = next(iter(shared))
                        if int(op_t.ind_size(edge)) != 1:
                            raise ValueError(
                                "TreeMPO operator_support omits a nontrivial "
                                f"boundary bond on edge {(nid, neighbor)!r}."
                            )
                        op_t = op_t.isel({edge: 0})
                    qubit = self.plan.qubit_of_node.get(nid)
                    absorb_started = self._profile_phase_start()
                    if qubit is not None:
                        upper = tree_mpo.upper_ind(qubit)
                        lower = tree_mpo.lower_ind(qubit)
                        physical = self._phys(qubit)
                        if upper not in op_t.inds or lower not in op_t.inds:
                            raise ValueError(
                                f"TreeMPO is missing physical site {qubit!r}."
                            )
                        op_t.reindex_({
                            lower: physical,
                            upper: physical + "*",
                        })
                        if self.mode == "zipup":
                            local[nid] = [state_t, op_t]
                            continue
                        try:
                            local[nid] = _contract_two_tensors(
                                state_t, op_t, shared_ind=physical,
                            ).reindex_({physical + "*": physical})
                        finally:
                            self._profile_phase_event(
                                "tensor_absorption",
                                absorb_started,
                                support=(qubit,),
                                route="subtreempo",
                            )
                    else:
                        if self.mode == "zipup":
                            local[nid] = [state_t, op_t]
                            continue
                        try:
                            local[nid] = qtn.tensor_contract(state_t, op_t)
                        finally:
                            self._profile_phase_event(
                                "tensor_absorption",
                                absorb_started,
                                support=(),
                                route="subtreempo",
                            )
                    operator_inds[nid] = (
                        set(local[nid].inds) - state_inds[nid]
                    )

                if self.mode == "zipup":
                    self._zipup_subtree_messages(
                        local, state_inds, order, hub,
                        max_bond=max_bond, cutoff=cutoff,
                    )
                    self._install_routed_subtree(local, snodes, hub)
                    if started:
                        self._finish_update()
                    return self
                self._route_subtree_messages(
                    local,
                    state_inds,
                    operator_inds,
                    order,
                    token=qtn.rand_uuid(),
                    workers=self.subtree_workers,
                )
                if operator_inds[hub]:
                    raise ValueError(
                        "TreeMPO application left open operator bonds at its hub."
                    )
                self._install_routed_subtree(local, snodes, hub)
                self._compress_subtree(
                    snodes,
                    hub,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    preserve_subcap=False,
                )
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        return self

    apply_sub_tree_mpo = apply_subtreempo
    apply_sub_treempo = apply_subtreempo
    apply_subttno = apply_subtreempo

    def apply_submpo(
        self, submpo, where, *, max_bond=None, cutoff=None, track_norm=True
    ):
        """Apply an explicit MPO on ``where`` using the native tree path.

        This is the backend-neutral coefficient-state entry point used by
        stabilizer and ordinary operator-sum frontends. Two-site MPOs reuse
        the factorized gate path; larger MPOs exposing Quimb's site interface
        stay structured and are QR-routed through the Steiner subtree before
        one compression sweep. Opaque MPO-like payloads fall back to dense
        :meth:`apply_subtree_operator` lowering.
        Set ``track_norm=False`` when the MPO is a known non-unitary map.
        """
        self._warn_track_truncation_slow()
        self._invalidate_state_norm_cache()
        logical_where = _normalize_where(where)
        where = self._validate_support(logical_where)
        payload_native = _submpo_is_native(submpo)
        state_native = bool(getattr(self.tn, "fermionic", False))
        if payload_native is not None and payload_native != state_native:
            if state_native:
                raise TypeError(
                    "native fermionic TreeTensorNetwork requires a native "
                    "Symmray MPO. Build it with build_tree_operator(...) or "
                    "supply a model-native MPO."
                )
            raise TypeError(
                "a native Symmray MPO cannot be applied to an ordinary dense "
                "TreeTensorNetwork. Use fermionic=False for the explicit "
                "Jordan--Wigner compatibility MPO."
            )
        return self._apply_submpo_resolved(
            submpo, where, logical_where=logical_where,
            max_bond=max_bond, cutoff=cutoff, track_norm=track_norm,
        )

    def expectation_mpo(
        self, submpo, where, *, max_bond=None, cutoff=0.0,
        normalized=True, optimize="auto", warn_on_truncation=True,
        return_diagnostics=False,
    ):
        """Evaluate ``<psi|MPO|psi>`` through one structured tree-MPO pass.

        The live state is not modified.  A private branch routes an ordinary
        chain MPO with :meth:`apply_submpo`, while a complete :class:`TreeMPO`
        is routed with :meth:`apply_subtreempo` so its internal TTNO bonds are
        contracted through the TreePlan rather than left open.  In both cases
        no ``to_dense`` conversion is needed.  ``max_bond``
        defaults to this optimizer's ``chi``; pass a larger cap when the
        operator application must retain more of the exact MPO-transformed
        state.  ``cutoff=0.0`` is the default because this is a measurement,
        not a variational update. A finite ``max_bond`` can still truncate if
        the transformed ket exceeds that cap. Such truncation emits a
        ``UserWarning`` by default; set ``warn_on_truncation=False`` only when
        that approximation is intentional. Set ``return_diagnostics=True`` to
        receive ``(value, diagnostics)`` with the compression events from this
        expectation only.
        """
        if not isinstance(warn_on_truncation, bool):
            raise TypeError("warn_on_truncation must be a bool.")
        if not isinstance(return_diagnostics, bool):
            raise TypeError("return_diagnostics must be a bool.")
        logical_where = _normalize_where(where)
        effective_max_bond = (
            self.chi
            if max_bond is None
            else self._normalize_max_bond(max_bond)
        )
        effective_cutoff = (
            self.cutoff if cutoff is None else float(cutoff)
        )
        if effective_cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")
        event_start = len(self.profile_events)
        work = self.copy()
        history_start = len(work.truncation_history)
        # A TreeMPO is already a complete tree operator.  It cannot be
        # treated as an ordinary site-labelled MPO: doing so extracts only
        # its physical legs and strands the TTNO virtual bonds.  Keep this
        # check structural instead of importing TreeMPO here, which avoids a
        # module cycle and also permits TreeMPO-compatible subclasses.
        is_tree_mpo = (
            getattr(submpo, "plan", None) is not None
            and hasattr(submpo, "tree_networks")
        )
        if is_tree_mpo:
            work.apply_subtreempo(
                submpo,
                logical_where,
                max_bond=max_bond,
                cutoff=effective_cutoff,
                track_norm=False,
            )
        else:
            work.apply_submpo(
                submpo,
                logical_where,
                max_bond=max_bond,
                cutoff=effective_cutoff,
                track_norm=False,
            )
        compression_events = work.truncation_history[history_start:]
        truncated_events = [
            event for event in compression_events
            if event.get("truncated", False)
        ]
        if truncated_events and warn_on_truncation:
            warnings.warn(
                "expectation_mpo compressed its private transformed ket on "
                f"{len(truncated_events)} edge(s) with max_bond="
                f"{effective_max_bond!r}; the expectation is approximate. "
                "Increase max_bond or inspect return_diagnostics=True if "
                "an untruncated measurement is required.",
                UserWarning,
                stacklevel=2,
            )

        # The bra and ket are separate TTNs. Keep their physical indices shared
        # for the inner product, but rename the ket's virtual bonds so each
        # layer remains an ordinary tree and no bond is accidentally merged
        # across the bra/ket boundary.
        ket = work.tn.copy()
        outer = set(ket.outer_inds())
        virtual = {
            index
            for tensor in ket.tensors
            for index in tensor.inds
            if index not in outer
        }
        ket.reindex_({index: qtn.rand_uuid() for index in virtual})
        numerator = (self.tn.H | ket).contract(all, optimize=optimize)
        if not normalized:
            result = numerator
        elif self.tn.fermionic:
            result = numerator / self.tn._fermionic_norm_squared()
        else:
            result = numerator / (self.tn.norm() ** 2)

        if self.profile and len(work.profile_events) > event_start:
            self.profile_events.extend(
                deepcopy(work.profile_events[event_start:])
            )
        if not return_diagnostics:
            return result
        diagnostics = {
            "support": tuple(logical_where),
            "max_bond": effective_max_bond,
            "cutoff": effective_cutoff,
            "n_events": len(compression_events),
            "n_truncated": len(truncated_events),
            "truncated": bool(truncated_events),
            "events": deepcopy(compression_events),
        }
        return result, diagnostics

    def expectation_mpo_exact(
        self, submpo, where, *, normalized=True, optimize="auto",
    ):
        """Evaluate an MPO by exact separate-network contraction.

        Unlike :meth:`expectation_mpo`, this method never applies the MPO to a
        copied tree and never compresses a state bond. It delegates to
        :meth:`TreeTensorNetwork.expectation_mpo_exact`, which connects the
        MPO input legs to a private ket view and its output legs to the bra in
        one complete doubled contraction. Native fermionic MPOs therefore
        retain Symmray's graded contraction rules.
        """
        return self.tn.expectation_mpo_exact(
            submpo,
            where,
            normalized=normalized,
            optimize=optimize,
        )

    def _apply_submpo_resolved(
        self, submpo, where, *, max_bond=None, cutoff=None,
        logical_where=None, track_norm=True,
    ):
        """Apply a sub-MPO whose support is already in compact TTN positions."""
        where = tuple(where)
        if logical_where is None:
            logical_where = where
        else:
            logical_where = tuple(logical_where)
        if len(logical_where) != len(where):
            raise ValueError(
                "logical and compact sub-MPO supports must have equal length."
            )
        self._check_operator_limits(where, dense=False)
        max_bond = (
            self.chi if max_bond is None else self._normalize_max_bond(max_bond)
        )
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        if cutoff < 0.0:
            raise ValueError(
                "max_bond must be positive or None and cutoff non-negative."
            )
        started = self._begin_update(
            "submpo", where, track_norm=track_norm
        )
        try:
            with self._thread_ctx():
                applied = None
                if len(where) == 2:
                    factor_started = self._profile_phase_start()
                    try:
                        factors = self._two_site_mpo_factors(
                            submpo, where[0], where[1],
                            site_where=(logical_where[0], logical_where[1]),
                        )
                    finally:
                        self._profile_phase_event(
                            "gate_factorization",
                            factor_started,
                            route="submpo",
                            cache_hit=False,
                            support=tuple(logical_where),
                        )
                    if factors is not None:
                        self._apply_2q_factors_impl(
                            *factors, where[0], where[1],
                            max_bond=max_bond,
                            cutoff=cutoff,
                            preserve_subcap=True,
                        )
                        applied = True
                if applied is None:
                    applied = self._try_apply_native_submpo(
                        submpo, where, payload_where=logical_where,
                        max_bond=max_bond, cutoff=cutoff,
                    )
                if applied is None:
                    self._check_operator_limits(where)
                    self._apply_subtree_operator_impl(
                        _submpo_to_dense(submpo, logical_where), logical_where,
                        max_bond=max_bond, cutoff=cutoff,
                    )
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        return self

    def apply_pauli_rotation(
        self, theta, pauli, where, *, sign=1.0, _force_tree_mpo=False
    ):
        """Apply ``exp(-i theta * sign * P / 2)`` on a Pauli support.

        The operator is represented as a compact TreeMPO on the true support
        Steiner subtree, so this remains efficient when ``where`` is sparse or
        long. Explicit ``mode='submpo'`` retains the chain-MPO compatibility
        route. This method is deliberately frame-neutral: callers such as a
        stabilizer wrapper may pass a tableau-conjugated Pauli here.
        """
        self._require_dense_qubit_state("apply_pauli_rotation")
        from ..stabilizer_tn.operators import pauli_combo_submpo

        logical_where = _normalize_where(where)
        where = self._validate_support(logical_where)
        axes = _normalize_measure_axes(pauli, logical_where)
        sign = float(sign)
        if sign not in (-1.0, 1.0):
            raise ValueError("Pauli rotation sign must be +1 or -1.")
        terms = dict(zip(where, axes))
        c = np.cos(float(theta) / 2.0)
        coef = -1j * sign * np.sin(float(theta) / 2.0)
        if self.mode == "submpo" and not _force_tree_mpo:
            mpo, mpo_where = pauli_combo_submpo(
                c, coef, terms, self.n, dtype=self.dtype,
                compact_support=True,
            )
            self._coerce_tensor_network_backend(mpo, warn=False)
            return self._apply_submpo_resolved(mpo, mpo_where)

        from .operators import TreeMPO

        tree_mpo = TreeMPO.from_pauli_sum(
            self.plan,
            [(c, {}), (coef, terms)],
            dtype=self.dtype,
        )
        return self.apply_subtreempo(
            tree_mpo,
            tree_mpo.operator_support,
            track_norm=True,
            _validate_backend=False,
        )

    def apply_pauli_sum(
        self, weighted_terms, *, max_bond=None, cutoff=None, track_norm=True,
        _force_tree_mpo=False,
    ):
        """Apply a weighted sum of Pauli products as one native TreeMPO.

        ``weighted_terms`` contains ``(coefficient, mapping)`` pairs, where
        each mapping is ``{qubit: 'X'|'Y'|'Z'}``. The exact TTNO bond is
        bounded by the number of branches, its exterior legs remain bond one,
        and the resulting operator is absorbed through the native TreeMPO
        QR-routing and compression path. ``mode='submpo'`` retains the
        explicit MPS-style compatibility implementation.
        """
        self._require_dense_qubit_state("apply_pauli_sum")

        terms = tuple(weighted_terms)
        if not terms:
            raise ValueError("weighted_terms must contain at least one term.")
        resolved_terms = []
        for weight, term in terms:
            resolved_terms.append((
                weight,
                {
                    self._validate_qubit(q): axis
                    for q, axis in term.items()
                },
            ))
        if self.mode == "submpo" and not _force_tree_mpo:
            from ..stabilizer_tn.operators import pauli_sum_submpo

            mpo, where = pauli_sum_submpo(
                resolved_terms, self.n, dtype=self.dtype,
                compact_support=True,
            )
            self._coerce_tensor_network_backend(mpo, warn=False)
            return self._apply_submpo_resolved(
                mpo,
                where,
                max_bond=max_bond,
                cutoff=cutoff,
                track_norm=track_norm,
            )

        from .operators import TreeMPO

        tree_mpo = TreeMPO.from_pauli_sum(
            self.plan,
            resolved_terms,
            dtype=self.dtype,
        )
        return self.apply_subtreempo(
            tree_mpo,
            tree_mpo.operator_support,
            max_bond=max_bond,
            cutoff=cutoff,
            track_norm=track_norm,
            _validate_backend=False,
        )

    def expectation_pauli(self, pauli, where, *, sign=1.0):
        """Return the normalized expectation of a product Pauli operator."""
        self._require_dense_qubit_state("expectation_pauli")
        where = self._validate_support(_normalize_where(where))
        axes = _normalize_measure_axes(pauli, where)
        sign = float(sign)
        if sign not in (-1.0, 1.0):
            raise ValueError("Pauli expectation sign must be +1 or -1.")
        return sign * to_float(
            self._product_pauli_expectation(axes, where), real=True
        )

    def project_pauli(self, pauli, where, outcome, *, sign=1.0,
                      renormalize=True, normalize=None,
                      return_diagnostics=False, _force_tree_mpo=False):
        """Project onto a product-Pauli eigenvalue.

        By default the post-projection state is normalized.  Set
        ``renormalize=False`` (or the compatibility alias ``normalize=False``)
        to retain the physical projection norm, which is useful for composing
        Kraus branches and for survival-probability accounting.  With
        ``return_diagnostics=True`` a dictionary containing the norm change and
        support/tree/bond snapshots is returned instead of ``self``.
        """
        self._require_dense_qubit_state("project_pauli")
        logical_where = _normalize_where(where)
        where = self._validate_support(logical_where)
        if not isinstance(outcome, Integral) or int(outcome) not in (-1, 1):
            raise ValueError("Pauli projection outcome must be +1 or -1.")
        sign = float(sign)
        if sign not in (-1.0, 1.0):
            raise ValueError("Pauli projection sign must be +1 or -1.")
        if normalize is not None:
            renormalize = bool(normalize)
        axes = _normalize_measure_axes(pauli, logical_where)
        diagnostics = self._apply_product_pauli_projector(
            axes, where, int(outcome) * int(sign),
            renormalize=bool(renormalize),
            return_diagnostics=return_diagnostics,
            logical_support=logical_where,
            _force_tree_mpo=_force_tree_mpo,
        )
        return diagnostics if return_diagnostics else self

    def _apply_subtree_operator_impl(self, op, where, *, max_bond=None,
                               cutoff=None, renormalize=False):
        """Apply a general multi-qubit operator over its minimal subtree.

        The one-shot generalisation of :meth:`apply_2q` to an operator on
        ``k >= 1`` qubits -- a ``k``-qubit gate (e.g. a Toffoli), a multi-site
        non-unitary / Kraus operator, or a whole Trotter block -- applied as a
        single object rather than decomposed into one- and two-qubit gates.

        The operator is first factorized into an exact tree-MPO on the
        *minimal connected subtree* (Steiner subtree) spanning the target
        physical nodes. It is then applied recursively from the subtree leaves toward
        a hub: each local state/operator message is QR-split losslessly on one
        edge and immediately absorbed by its parent. Thus no dense state tensor
        for the whole Steiner subtree is formed. This is the tree analogue of a
        sub-MPO applied over a covering range and then compressed (cf.
        ``quimb``'s ``MatrixProductState.gate_with_submpo``, which exists for
        the 1D chain only). Once the complete operator has reached the hub, a
        canonical subtree sweep truncates every affected edge once against an
        isometric environment.

        ``track_norm`` controls the cheap retained-norm ledger. ``op`` acts on
        ``len(where)`` qubits: an array reshaped to ``(2,) * 2k``
        with output indices first, ``op[o_0..o_{k-1}, i_0..i_{k-1}]`` (a
        ``(2**k, 2**k)`` matrix is accepted and reshaped).  It need **not** be
        unitary; pass ``renormalize=True`` to renormalise the state afterwards
        (e.g. after a Kraus/projection operator).  ``max_bond`` / ``cutoff``
        default to the optimizer's ``chi`` / ``cutoff``.  Returns ``self``.
        """
        logical_where = _normalize_where(where)
        k = len(logical_where)
        if k < 1:
            raise ValueError("apply_subtree_operator needs at least one qubit.")
        where = self._validate_support(logical_where)
        self._check_operator_limits(where)
        if len(set(where)) != k:
            raise ValueError(
                f"apply_subtree_operator needs distinct qubits; got {where}."
            )
        max_bond = (
            self.chi if max_bond is None else self._normalize_max_bond(max_bond)
        )
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")

        if _is_symmray_array(op):
            op_shape = tuple(int(dim) for dim in ar.shape(op))
            if len(op_shape) != 2 * k:
                raise ValueError(
                    "native subtree operators must have one output and one "
                    f"input leg per site; got shape {op_shape}."
                )
            if tuple(op_shape[:k]) != tuple(op_shape[k:]):
                raise ValueError(
                    "native subtree operator output and input dimensions "
                    f"must match; got shape {op_shape}."
                )
            with self._thread_ctx():
                if k == 1:
                    self._apply_1q_impl(op, logical_where[0])
                elif k == 2:
                    # Keep the public logical labels here. The two-site kernel
                    # resolves them exactly once, which is essential after a
                    # stable-label cap.
                    self._apply_2q_impl(
                        op, logical_where[0], logical_where[1],
                        max_bond=max_bond, cutoff=cutoff,
                    )
                else:
                    # Native multi-site arrays are lowered to a native MPO;
                    # unlike the dense path this preserves charge sectors and
                    # graded signs. The generated MPO uses compact site tags,
                    # while the outer update retains the caller's labels.
                    submpo = qtn.MatrixProductOperator.from_dense(
                        self._as_state_backend(op),
                        dims=op_shape[:k],
                        sites=where,
                        L=self.n,
                        max_bond=None,
                        cutoff=0.0,
                    )
                    self._apply_submpo_resolved(
                        submpo,
                        where,
                        logical_where=where,
                        max_bond=max_bond,
                        cutoff=cutoff,
                    )
            if renormalize:
                self.normalize()
            return self

        phys = [self._phys(q) for q in where]
        op_arr = ar.do(
            "reshape", self._as_state_backend(op), [2] * (2 * k)
        )

        with self._thread_ctx():
            if k == 1:
                # Single-site operator (possibly non-unitary): centre on its node
                # so it holds the (rescaled) norm, then absorb the operator.
                site_node = self.plan.node_of_qubit[where[0]]
                self._move_center(site_node)
                self.tn.gate_inds_(op_arr, [phys[0]], contract=True)
                self.center = site_node
                if renormalize:
                    self.normalize()
                return self

            site_nodes = [self.plan.node_of_qubit[q] for q in where]
            snodes = self._steiner_nodes(site_nodes)
            # Centre on a target physical node so the whole exterior is isometric toward
            # the subtree. Operator bonds are routed losslessly first; the
            # final subtree sweep then measures true state error against that
            # complete operator update.
            anchor = self._nearest_anchor(site_nodes)
            if self.center != anchor:
                self._move_center(anchor)

            # Factor the operator into an exact tree-MPO and apply it by
            # leaf-to-hub message passing. This is the tree generalisation of
            # the paper's two-qubit SVD + thread + compress update. In
            # particular, no state tensor in the Steiner subtree is contracted
            # with any other state tensor: each edge creates one local message,
            # which is immediately absorbed by its parent and split again.
            order, hub = self._peel_order(snodes)
            factor_started = self._profile_phase_start()
            op_factors, op_bonds = self._decompose_tree_operator(
                op_arr, where, snodes, order, hub,
            )
            self._profile_phase_event(
                "gate_factorization",
                factor_started,
                route="tree_operator",
                support=tuple(logical_where),
                subtree_nodes=len(snodes),
            )
            self._apply_factorized_subtree_operator_impl(
                op_factors, op_bonds, where, snodes, order, hub,
                max_bond=max_bond, cutoff=cutoff,
            )

            if renormalize:
                self.normalize()
        return self

    def _try_apply_native_submpo(
        self, submpo, where, *, max_bond, cutoff, payload_where=None,
    ):
        """Apply an MPO payload without materialising ``to_dense()``.

        A MatrixProductOperator is itself a tensor network. Its open operator
        bonds are QR-routed through the same leaf-to-hub sweep as the TTN state
        bonds: each peeled state tensor retains all currently open MPO bonds,
        and they contract when their partner reaches the same accumulated
        subtree. Only after the operator is complete is the affected subtree
        canonically SVD-compressed. This works for arbitrary MPO bond
        dimensions and does not require the MPO chain to match the TTN geometry.

        ``None`` is returned when the payload does not expose the Quimb MPO
        site interface; callers then use the legacy dense fallback.
        """
        where = tuple(where)
        if payload_where is None:
            payload_where = where
        else:
            payload_where = tuple(payload_where)
        if len(payload_where) != len(where):
            raise ValueError(
                "logical and compact sub-MPO supports must have equal length."
            )
        payload_for_compact = dict(zip(where, payload_where))
        gen_sites = getattr(submpo, "gen_sites_present", None)
        site_tag = getattr(submpo, "site_tag", None)
        upper_id = getattr(submpo, "upper_ind_id", None)
        lower_id = getattr(submpo, "lower_ind_id", None)
        tag_map = getattr(submpo, "tag_map", None)
        tensor_map = getattr(submpo, "tensor_map", None)
        if not all((gen_sites, callable(site_tag), upper_id, lower_id,
                    tag_map is not None, tensor_map is not None)):
            return None
        try:
            present = tuple(gen_sites())
        except Exception:
            return None
        if set(present) != set(payload_where):
            return None

        site_nodes = [self.plan.node_of_qubit[q] for q in where]
        snodes = self._steiner_nodes(site_nodes)
        self._move_center(self._nearest_anchor(site_nodes))
        path_started = self._profile_phase_start()
        order, hub = self._peel_order(snodes)
        self._profile_phase_event(
            "metadata_path",
            path_started,
            support=tuple(payload_where),
            route="submpo_subtree",
            subtree_nodes=len(snodes),
            message_edges=len(order),
            hub=hub,
        )
        local = {}
        state_inds = {}
        operator_inds = {}
        for nid in snodes:
            state_t = self.tn.tensor_map[self._tid(nid)].copy()
            state_inds[nid] = set(state_t.inds)
            q = self.plan.qubit_of_node.get(nid)
            # A physical root can be an internal Steiner node without being
            # acted on by the MPO. Keep its state tensor untouched and route
            # it as ordinary state data. Only target physical nodes need an
            # MPO site tensor and the associated site-tag lookup.
            if q is None or q not in payload_for_compact:
                local[nid] = state_t
                operator_inds[nid] = set()
                continue
            try:
                payload_q = payload_for_compact[q]
                tids = tuple(tag_map[site_tag(payload_q)])
                if len(tids) != 1:
                    return None
                op_t = tensor_map[tids[0]].copy()
                op_t.modify(data=self._as_state_backend(op_t.data))
                upper = upper_id.format(payload_q)
                lower = lower_id.format(payload_q)
                if upper not in op_t.inds or lower not in op_t.inds:
                    return None
                op_t.reindex_({upper: self._phys(q) + "*",
                               lower: self._phys(q)})
                operator_inds[nid] = set(op_t.inds) - {
                    self._phys(q) + "*", self._phys(q)
                }
                absorb_started = self._profile_phase_start()
                try:
                    local[nid] = _contract_two_tensors(
                        state_t, op_t, shared_ind=self._phys(q),
                    ).reindex_(
                        {self._phys(q) + "*": self._phys(q)}
                    )
                finally:
                    self._profile_phase_event(
                        "tensor_absorption",
                        absorb_started,
                        support=(q,),
                        route="submpo_site",
                    )
            except (KeyError, TypeError, ValueError):
                return None

        self._route_subtree_messages(
            local, state_inds, operator_inds, order, token=qtn.rand_uuid(),
            workers=self.subtree_workers,
        )

        if operator_inds[hub]:
            raise ValueError(
                "native sub-MPO application left open operator bonds; "
                "use an MPO with a closed tensor-network contraction."
            )
        # The exterior remained isometric toward the updated Steiner subtree.
        # Dense routed Q tensors retain their isometry metadata, so recovering
        # the hub centre is metadata-only; native graded trees keep their
        # explicit QR recovery. Truncate only after every MPO bond has arrived.
        self._install_routed_subtree(local, snodes, hub)
        self._compress_subtree(
            snodes, hub, max_bond=max_bond, cutoff=cutoff,
        )
        return True

    def _apply_factorized_subtree_operator_impl(
        self, op_factors, op_bonds, where, snodes, order, hub, *,
        max_bond, cutoff,
    ):
        """Apply a factorized tree-MPO by QR routing then one final sweep."""
        local = {}
        state_inds = {}
        operator_inds = {}
        for nid in snodes:
            state_t = self.tn.tensor_map[self._tid(nid)].copy()
            state_inds[nid] = set(state_t.inds)
            op_t = op_factors[nid]
            q = self.plan.qubit_of_node.get(nid)
            if q is not None and q in where:
                # Operator sites are packed into one dimension-four leg. Split
                # that leg only at physical sites, then contract its input leg
                # with the live state physical index.
                absorb_started = self._profile_phase_start()
                try:
                    op_t = self._expand_tree_operator_leaf(
                        op_t,
                        op_bonds["physical"][q],
                        self._phys(q),
                    )
                    local[nid] = _contract_two_tensors(
                        state_t, op_t, shared_ind=self._phys(q),
                    )
                finally:
                    self._profile_phase_event(
                        "tensor_absorption",
                        absorb_started,
                        support=(q,),
                        route="tree_operator_site",
                    )
            else:
                absorb_started = self._profile_phase_start()
                try:
                    local[nid] = qtn.tensor_contract(state_t, op_t)
                finally:
                    self._profile_phase_event(
                        "tensor_absorption",
                        absorb_started,
                        support=(),
                        route="tree_operator_internal",
                    )
            if q is not None and q in where:
                local[nid].reindex_({f"{self._phys(q)}*": self._phys(q)})
            operator_inds[nid] = set(local[nid].inds) - state_inds[nid]

        self._route_subtree_messages(
            local, state_inds, operator_inds, order, token=qtn.rand_uuid(),
            workers=self.subtree_workers,
        )
        if operator_inds[hub]:
            raise ValueError(
                "factorized subtree operator left open operator bonds at its hub."
            )

        self._install_routed_subtree(local, snodes, hub)
        self._compress_subtree(
            snodes, hub, max_bond=max_bond, cutoff=cutoff,
        )

    def _decompose_tree_operator(self, op_arr, where, snodes, order, hub):
        """Factor a dense operator into tensors on a Steiner tree.

        The operator is first viewed as a tensor with one dimension-four leg
        per target qubit, where that leg packs (output, input). Repeated
        leaf-to-hub SVDs then produce the exact hierarchical/tree-MPO factors.
        Unlike the old state contraction path, this decomposition never sees a
        state tensor and is performed once on the operator payload itself.

        Returns
        -------
        factors : dict[int, qtn.Tensor]
            One operator factor for each Steiner node.
        bonds : dict
            (child, parent) -> operator bond plus
            "physical" -> qubit -> packed physical index.
        """
        where = tuple(where)
        op_axes = [f"_ttn_op_phys_{qtn.rand_uuid()}_{q}" for q in where]
        # op_arr has all output axes followed by all input axes. Interleave
        # them before packing each pair into one dimension-four operator leg.
        interleaved = ar.do(
            "transpose",
            op_arr,
            [i for q in range(len(where)) for i in (q, len(where) + q)],
        )
        interleaved = ar.do("reshape", interleaved, (4,) * len(where))
        blob = qtn.Tensor(interleaved, inds=op_axes)

        node_for_q = {
            self.plan.node_of_qubit[q]: q for q in where
        }
        owned = {nid: set() for nid in snodes}
        for node, q in node_for_q.items():
            owned[node].add(op_axes[where.index(q)])

        factors = {}
        op_bonds = {"physical": dict(zip(where, op_axes))}
        for u, v in order:
            left_inds = [ix for ix in blob.inds if ix in owned[u]]
            if not left_inds:
                raise RuntimeError(
                    "tree-MPO decomposition encountered an empty child block "
                    f"at node {u} on edge {(u, v)}."
                )
            bond = f"_ttn_op_bond_{qtn.rand_uuid()}_{u}_{v}"
            left, blob = blob.split(
                left_inds=left_inds,
                method="svd",
                max_bond=None,
                cutoff=0.0,
                absorb="right",
                bond_ind=bond,
            )
            factors[u] = left
            op_bonds[(u, v)] = bond
            owned[v].add(bond)

        factors[hub] = blob
        return factors, op_bonds

    def _apply_product_pauli_projector_impl(
        self, axes, where, snodes, order, hub, outcome, *, max_bond, cutoff,
    ):
        """Apply ``(I + outcome P) / 2`` with a dimension-two branch index."""
        target_axes = dict(zip(where, axes))
        local = {}
        branch_index = {}
        for nid in snodes:
            state_t = self.tn.tensor_map[self._tid(nid)].copy()
            q = self.plan.qubit_of_node.get(nid)
            if q in target_axes:
                p = self._phys(q)
                branch = f"_ttn_pauli_branch_{qtn.rand_uuid()}"
                operators = np.stack(
                    (
                        np.eye(2, dtype=complex),
                        outcome * _PAULI_1Q[target_axes[q]],
                    ),
                    axis=-1,
                )
                op_t = qtn.Tensor(
                    self._as_state_backend(operators, warn=False),
                    inds=(p + "*", p, branch),
                )
                local[nid] = _contract_two_tensors(
                    state_t, op_t, shared_ind=p,
                ).reindex_(
                    {p + "*": p}
                )
                branch_index[nid] = branch
            else:
                local[nid] = state_t

        # Carry one dimension-two branch through the tree. When multiple child
        # messages meet, a three-leg copy tensor enforces the same branch while
        # keeping the representation linear in the node degree (rather than
        # forming a high-rank 2**degree copy tensor at a wide hub).
        update_token = qtn.rand_uuid()
        for u, v in order:
            state_bond = self.tn.bond(u, v)
            branch = branch_index[u]
            left_inds = [
                ix for ix in local[u].inds
                if ix not in {state_bond, branch}
            ]
            new_bond = f"_ttn_pauli_apply_{update_token}_{u}_{v}"
            tu, message = self._split_with_diagnostics(
                local[u], left_inds, edge=(u, v), bond_ind=new_bond,
                max_bond=max_bond, cutoff=cutoff,
            )
            local[u] = tu

            parent_branch = branch_index.get(v)
            if parent_branch is None:
                local[v] = qtn.tensor_contract(local[v], message)
                branch_index[v] = branch
            else:
                branch_out = f"_ttn_pauli_branch_{qtn.rand_uuid()}"
                copy_tensor = np.zeros((2, 2, 2), dtype=complex)
                copy_tensor[0, 0, 0] = 1.0
                copy_tensor[1, 1, 1] = 1.0
                copy_tensor = qtn.Tensor(
                    self._as_state_backend(copy_tensor, warn=False),
                    inds=(parent_branch, branch, branch_out),
                )
                local[v] = qtn.tensor_contract(
                    local[v], message, copy_tensor
                )
                branch_index[v] = branch_out

        root_branch = branch_index[hub]
        local[hub] = qtn.tensor_contract(
            local[hub],
            qtn.Tensor(
                self._as_state_backend(
                    np.array([0.5, 0.5], dtype=complex), warn=False
                ),
                inds=(root_branch,),
            ),
        )
        for nid in snodes:
            node_t = self.tn.tensor_map[self._tid(nid)]
            node_t.modify(
                data=local[nid].data,
                inds=local[nid].inds,
                left_inds=None if nid == hub else local[nid].left_inds,
            )
        self.center = hub

    def _apply_product_pauli_projector(
        self, axes, where, outcome, *, renormalize=True,
        return_diagnostics=False, logical_support=None, probability=None,
        _force_tree_mpo=False,
    ):
        """Apply a product-Pauli parity projector without dense materialization."""
        # Callers of this private helper have already resolved logical labels
        # to compact TTN positions.
        where = self._validate_support(where, resolve=False)
        self._check_operator_limits(where, dense=False)
        if len(set(where)) != len(where):
            raise ValueError(
                f"product-Pauli measurement needs distinct qubits; got {where}."
            )
        before_norm = self.norm()
        before = self._projection_snapshot(where)
        started = self._begin_update("measure", where)
        try:
            # Build every projector, including the one-site case, as the
            # same two-branch TreeMPO. ``where`` is in compact Tree positions;
            # map it back to logical qubit labels before constructing the
            # operator so custom/snake layouts remain correct.
            logical_where = tuple(self._logical_qubits[q] for q in where)
            self.apply_pauli_sum(
                [
                    (0.5, {}),
                    (0.5 * outcome, dict(zip(logical_where, axes))),
                ],
                max_bond=self.chi,
                cutoff=self.cutoff,
                track_norm=False,
                _force_tree_mpo=_force_tree_mpo,
            )
            if renormalize:
                self.normalize()
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        after_norm = self.norm()
        after = self._projection_snapshot(where)
        diagnostics = {
            "support": tuple(
                before["support"] if logical_support is None
                else tuple(logical_support)
            ),
            "span_before": before["span"],
            "span_after": after["span"],
            "bonds_before": before["bonds"],
            "bonds_after": after["bonds"],
            "max_bond_before": before["max_bond"],
            "max_bond_after": after["max_bond"],
            "norm_before": float(before_norm),
            "norm_after": float(after_norm),
            "norm_ratio": float(after_norm / before_norm)
            if before_norm > 0.0 else 0.0,
            "renormalized": bool(renormalize),
            "outcome": int(outcome),
        }
        if probability is not None:
            diagnostics["probability"] = float(probability)
        self.projection_diagnostics.append(diagnostics)
        if started and self.record_history and self.update_history:
            self.update_history[-1]["projection"] = deepcopy(diagnostics)
        return diagnostics if return_diagnostics else None

    def _product_pauli_expectation(self, axes, where):
        """Evaluate a product-Pauli expectation using one-site insertions."""
        site_nodes = [self.plan.node_of_qubit[q] for q in where]
        snodes = self._steiner_nodes(site_nodes)
        if self.center not in snodes:
            self._move_center(site_nodes[0])

        internal = set()
        for nid in snodes:
            for nb in self._neighbors(nid):
                if nb in snodes:
                    internal.add(self.tn.bond(nid, nb))
        ket = qtn.TensorNetwork([
            self.tn.tensor_map[self._tid(nid)].copy() for nid in snodes
        ])
        phys = [self._phys(q) for q in where]
        with self._thread_ctx():
            internal_map = {ix: qtn.rand_uuid() for ix in internal}
            bra_num = ket.H.reindex({
                **internal_map,
                **{p: p + "*" for p in phys},
            })
            numerator = bra_num & ket
            for axis, p in zip(axes, phys):
                numerator = numerator & qtn.Tensor(
                    self._as_state_backend(_PAULI_1Q[axis], warn=False),
                    inds=(p + "*", p),
                )
            num = numerator.contract(output_inds=[])
            den = (ket.H.reindex(internal_map) & ket).contract(output_inds=[])
        return num / den

    @staticmethod
    def _expand_tree_operator_leaf(op_tensor, packed_ind, physical_ind):
        """Unpack one dimension-four tree-MPO leg into output and input legs."""
        axis = op_tensor.inds.index(packed_ind)
        data = ar.do("moveaxis", op_tensor.data, axis, 0)
        data = ar.do("reshape", data, (2, 2) + tuple(data.shape[1:]))
        rest_inds = [ind for ind in op_tensor.inds if ind != packed_ind]
        return qtn.Tensor(
            data,
            inds=[f"{physical_ind}*", physical_ind] + rest_inds,
        )

    def _peel_order(self, snodes):
        """Return ``(peels, hub)`` for recursive leaf-to-hub application.

        ``peels`` is a list of ``(u, v)`` edges: repeatedly a current
        subtree-leaf ``u`` (with a single remaining subtree neighbour ``v``) is
        peeled off toward ``v`` until a single ``hub`` node remains -- the node
        the orthogonality centre ends on. Deterministic (smallest id first).
        """
        remaining = set(snodes)
        adj = {
            u: tuple(w for w in self._neighbors(u) if w in remaining)
            for u in remaining
        }
        degree = {
            u: sum(w in remaining for w in neighbours)
            for u, neighbours in adj.items()
        }
        leaves = [u for u, degree_u in degree.items() if degree_u == 1]
        heapq.heapify(leaves)
        peels = []
        while len(remaining) > 1:
            while leaves and leaves[0] not in remaining:
                heapq.heappop(leaves)
            if not leaves:
                raise ValueError("subtree peel order requires a connected tree")
            leaf = heapq.heappop(leaves)
            v = next(w for w in adj[leaf] if w in remaining)
            peels.append((leaf, v))
            remaining.discard(leaf)
            degree[leaf] = 0
            degree[v] -= 1
            if degree[v] == 1:
                heapq.heappush(leaves, v)
        hub = min(remaining)
        return peels, hub

    # -- readout --------------------------------------------------------------

    def max_bond(self):
        """Return the largest virtual bond dimension in the tree."""
        return self.tn.max_bond()

    def show(self, *, bond_dims=True, node_ids=False, color=True):
        """Print a top-down ASCII drawing of the tree with current bond dims.

        Delegates to :meth:`TreeTensorNetwork.show`: the root sits at the top,
        structural leaves at the bottom, physical nodes are labelled with their
        qubits, and each edge carries its virtual-bond dimension -- the tree
        analogue of a ``quimb`` MPS ``show``. Markers are coloured by tree layer
        by default; pass ``color=False`` for plain text.
        """
        self.tn.show(bond_dims=bond_dims, node_ids=node_ids, color=color)

    def bond_report(self):
        """Return a summary of the current virtual bond dimensions.

        A quick health check over the tree edges (inner indices): the maximum
        and mean bond dimension, the number of bonds and tensors, and the
        requested ``chi``.  Bonds pinned at ``chi`` mean the truncation is
        active (raise ``chi`` for more accuracy); bonds well below ``chi`` mean
        ``chi`` is not the accuracy bottleneck.
        """
        bonds = self.tn.inner_inds()
        bond_dims = [int(self.tn.ind_size(ix)) for ix in bonds]
        return {
            "max_bond": max(bond_dims) if bond_dims else 1,
            "mean_bond": (
                float(sum(bond_dims) / len(bond_dims)) if bond_dims else 1.0
            ),
            "n_bonds": len(bond_dims),
            "n_tensors": self.tn.num_tensors,
            "chi": self.chi,
        }

    def estimate_bonds(self, gates=None, *, max_operator_qubits=None):
        """Estimate untruncated bond growth from the fixed tree and gate stream.

        This is the paper's conservative dry-run bound (Eq. 4), not a
        contraction of the live state.  Each gate contributes its
        operator-Schmidt rank to every tree edge that separates its support;
        the reported edge dimensions are the product of those ranks over the
        stream.  Single-qubit gates and gates wholly contained on one side of
        an edge contribute one.  Because cancellations and state-specific rank
        deficiencies are intentionally ignored, the estimate is safe but can
        overestimate the dimensions reached by an actual replay.

        Parameters
        ----------
        gates : bundled gate stream, optional
            Stream to estimate.  When omitted, the optimizer's queued stream is
            used.  Control events are included in the event trace but do not
            change the bound.
        max_operator_qubits : int, optional
            If supplied, skip dense rank calculation for larger operators and
            mark those events with ``rank_skipped=True``.  This lets
            :meth:`preflight` reject oversized operators before allocating a
            dense ``4**k`` array.

        Returns
        -------
        dict
            ``edge_bonds`` maps undirected tree edges to their estimated bond
            dimensions; ``max_bond`` is their maximum (or ``1`` for a one-leaf
            tree); ``requires_truncation`` compares that bound to ``chi``; and
            ``events`` records the crossing edges and ranks contributed by each
            stream item.
        """
        max_operator_qubits = self._positive_limit(
            max_operator_qubits, "max_operator_qubits"
        )
        if gates is None:
            payloads = self.G
            wheres = self.where
            event_types = self.event_types
        else:
            payloads, wheres, event_types = self._normalize_gate_queue(gates)

        def edge_key(a, b):
            return (a, b) if a < b else (b, a)

        sim_plan = self.plan
        active = list(self._logical_qubits)
        edge_bonds = {}

        def plan_steiner(plan, support):
            site_nodes = [plan.node_of_qubit[q] for q in support]
            if len(site_nodes) == 1:
                return {site_nodes[0]}
            nodes = set()
            anchor = site_nodes[0]
            for site_node in site_nodes[1:]:
                nodes.update(plan.node_path(anchor, site_node))
            return nodes

        def plan_edges(plan):
            return {
                edge_key(parent, child): child
                for parent, children in plan.children.items()
                for child in children
            }

        events = []
        for index, (payload, where, event_type) in enumerate(
            zip(payloads, wheres, event_types)
        ):
            logical_support = _normalize_where(where)
            try:
                support = tuple(active.index(q) for q in logical_support)
            except ValueError as exc:
                raise ValueError(
                    f"event at step {index + 1} references qubit(s) outside "
                    f"the current active labels {active!r}: {logical_support!r}."
                ) from exc
            if len(support) > 1 and len(set(support)) != len(support):
                raise ValueError(
                    f"gate support must contain distinct qubits; got {support}."
                )
            span_nodes = len(plan_steiner(sim_plan, support))
            subtree_masks = sim_plan.subtree_qubit_masks()
            edge_sides = {
                edge: subtree_masks[child]
                for edge, child in plan_edges(sim_plan).items()
            }
            current_edges = set(edge_sides)
            edge_bonds = {
                edge: edge_bonds.get(edge, 1) for edge in current_edges
            }
            crossing = {}
            rank_skipped = bool(
                max_operator_qubits is not None
                and event_type in {"gate", "submpo"}
                and len(logical_support) > max_operator_qubits
            )
            rank_payload = payload
            if (
                event_type in {"gate", "submpo"}
                and len(logical_support) > 1
                and not rank_skipped
            ):
                support_mask = 0
                for site in support:
                    support_mask |= 1 << site
                for edge, side in edge_sides.items():
                    left_mask = support_mask & side
                    if not left_mask or left_mask == support_mask:
                        continue
                    left_positions = tuple(
                        q for q in support if left_mask & (1 << q)
                    )
                    left_where = tuple(
                        logical_support[support.index(q)] for q in left_positions
                    )
                    if event_type == "submpo":
                        rank = _submpo_schmidt_rank_bound(
                            rank_payload, logical_support, left_where
                        )
                        if rank is None:
                            rank = _operator_schmidt_rank(
                                _submpo_to_dense(
                                    rank_payload, logical_support
                                ),
                                logical_support,
                                left_where,
                            )
                    else:
                        rank = _operator_schmidt_rank(
                            rank_payload, logical_support, left_where
                        )
                    crossing[edge] = rank
                    edge_bonds[edge] *= rank
            events.append({
                "index": index,
                "kind": event_type,
                "support": support,
                "span_nodes": span_nodes,
                "rank_skipped": rank_skipped,
                "crossing_edges": dict(crossing),
                "edge_bonds": dict(edge_bonds),
            })

            if event_type == "cap":
                if len(logical_support) != 1:
                    raise ValueError(
                        f"cap event at step {index + 1} must reference one site."
                    )
                if len(active) <= 1:
                    raise ValueError(
                        f"cap event at step {index + 1} cannot remove the only site."
                    )
                sim_plan = sim_plan.remove_qubit(support[0])
                capped = logical_support[0]
                active.remove(capped)
                if payload.get("compact_labels", True):
                    active = [label - 1 if label > capped else label
                              for label in active]

        max_bond = max(edge_bonds.values(), default=1)
        return {
            "edge_bonds": edge_bonds,
            "max_bond": int(max_bond),
            "chi": self.chi,
            "requires_truncation": (
                False if self.chi is None else bool(max_bond > self.chi)
            ),
            "events": events,
        }

    def preflight(self, gates=None, *, max_bond=None,
                  max_operator_qubits=None, max_subtree_nodes=None,
                  raise_on_error=True):
        """Check conservative resource limits before replaying a stream.

        The bond limit uses :meth:`estimate_bonds` and is deliberately an
        upper bound: it ignores cancellations and state-specific rank loss. The
        operator and subtree limits protect dense operator allocation and bound
        the number of recursive Steiner-tree messages.
        With ``raise_on_error=False`` the method returns a report containing the
        violations instead of raising :class:`MemoryError`.

        The optimizer invokes this automatically before eager replay when any
        of its ``max_*`` constructor limits are set.  Direct calls to
        :meth:`apply_gate` and :meth:`apply_subtree_operator` still enforce the
        operator/subtree limits immediately.
        """
        max_bond = self._positive_limit(max_bond, "max_bond")
        max_operator_qubits = self._positive_limit(
            max_operator_qubits, "max_operator_qubits"
        )
        max_subtree_nodes = self._positive_limit(
            max_subtree_nodes, "max_subtree_nodes"
        )
        report = self.estimate_bonds(
            gates, max_operator_qubits=max_operator_qubits
        )
        violations = []
        if max_bond is not None and report["max_bond"] > max_bond:
            violations.append(
                f"estimated max bond {report['max_bond']} exceeds "
                f"max_bond={max_bond}"
            )
        for event in report["events"]:
            if (
                max_operator_qubits is not None
                and event["kind"] in {"gate", "submpo"}
                and len(event["support"]) > max_operator_qubits
            ):
                violations.append(
                    f"event {event['index']} has {len(event['support'])} "
                    f"operator qubits, exceeding max_operator_qubits="
                    f"{max_operator_qubits}"
                )
            if (
                max_subtree_nodes is not None
                and event["span_nodes"] > max_subtree_nodes
            ):
                violations.append(
                    f"event {event['index']} spans {event['span_nodes']} "
                    f"tree nodes, exceeding max_subtree_nodes="
                    f"{max_subtree_nodes}"
                )
        result = dict(report)
        result["limits"] = {
            "max_bond": max_bond,
            "max_operator_qubits": max_operator_qubits,
            "max_subtree_nodes": max_subtree_nodes,
        }
        result["violations"] = violations
        result["ok"] = not violations
        if violations and raise_on_error:
            raise MemoryError(
                "TreeOptimizer preflight failed: " + "; ".join(violations)
            )
        return result

    def truncation_report(self):
        """Return per-edge truncation diagnostics collected during replay.

        The ``events`` list contains both ordinary edge compressions and local
        SVD splits from sibling or subtree updates.  Each event reports the
        bond dimension before and after the update.  When
        ``track_truncation=True`` was requested, it also reports the local
        singular-spectrum norm, absolute discarded weight, and relative
        discarded fraction.  Spectrum-based fields are ``None`` otherwise so
        callers can distinguish an untracked run from a zero-loss truncation.
        """
        events = deepcopy(self.truncation_history)
        tracked = [
            event for event in events
            if event["discarded_weight"] is not None
        ]
        if tracked:
            total_discarded = float(
                sum(event["discarded_weight"] for event in tracked)
            )
            max_discarded = float(
                max(event["discarded_weight"] for event in tracked)
            )
            max_fraction = float(
                max(event["discarded_fraction"] for event in tracked)
            )
        else:
            total_discarded = None
            max_discarded = None
            max_fraction = None
        return {
            "track_truncation": self.track_truncation,
            "n_events": len(events),
            "n_truncated": sum(event["truncated"] for event in events),
            "n_tracked": len(tracked),
            "total_discarded_weight": total_discarded,
            "max_discarded_weight": max_discarded,
            "max_discarded_fraction": max_fraction,
            "events": events,
            "updates": deepcopy(self.update_history),
        }

    def to_dense(self, logical_order=True):
        """Return the dense statevector in logical qubit order.

        ``logical_order`` is accepted for MPS interface compatibility. Tree
        storage has no separate physical permutation, so both values produce
        the same ``k0, k1, ..., k(n-1)`` ordering.
        """
        _ = logical_order
        with self._thread_ctx():
            return self.tn.to_statevector(range(self.n))

    @property
    def qubits(self):
        """Return the active caller-facing qubit labels in logical order.

        With the default compact cap behavior this is ``range(self.n)``.  A
        stable-label cap can leave gaps here while the underlying TTN remains
        compact.
        """
        return list(self._logical_qubits)

    @property
    def logical_order(self):
        """Return active logical labels ordered by their compact TTN position."""
        return list(self._logical_qubits)

    def logical_site(self, position):
        """Return the logical qubit at compact tree position ``position``."""
        position = int(position)
        if not 0 <= position < self.n:
            raise IndexError(
                f"tree position {position} is outside the range [0, {self.n})."
            )
        return self._logical_qubits[position]

    def position(self, site):
        """Return the tree position of logical qubit ``site``."""
        return self._validate_qubit(site)

    def remap_sample(self, config):
        """Return a sample in active logical order."""
        if isinstance(config, dict):
            return dict(config)
        config = np.asarray(config)
        if config.ndim == 0 or config.shape[-1] != self.n:
            raise ValueError(
                "sample configuration must have tree size as its final "
                f"dimension, got shape {config.shape}."
            )
        return config.copy()

    def restore_qubit_order(self):
        """Return the state; tree storage is always already in logical order."""
        return self.tn

    def norm(self):
        """Return the state norm ``<psi|psi>**0.5``.

        Native fermionic trees use a graded one-tensor center contraction when
        a canonical center is known, with the complete doubled-network
        contraction as the unknown-gauge fallback. Dense/nonfermionic trees
        use the ordinary single-centre contraction when known and otherwise
        the full doubled-tree path. The fast one-tensor paths explicitly
        restore Quimb's extracted base-10 ``tn.exponent``; full contractions
        already include it.
        """
        center = self.center
        if self.tn.fermionic:
            with self._thread_ctx():
                val = self.tn._fermionic_center_norm_squared()
            nrm = float(np.sqrt(abs(to_float(val, real=True))))
            if center is not None:
                nrm *= self._represented_scale()
            return nrm
        if center is not None:
            t = self.tn.tensor_map[self._tid(center)]
            val = qtn.tensor_contract(t.H, t, output_inds=[])
            nrm = float(np.sqrt(abs(to_float(val, real=True))))
            return nrm * self._represented_scale()
        with self._thread_ctx():
            val = (self.tn.H & self.tn).contract(output_inds=[])
        return float(np.sqrt(abs(to_float(val, real=True))))

    def _represented_scale(self):
        """Return Quimb's extracted global base-10 state scale."""
        exponent = float(getattr(self.tn, "exponent", 0.0))
        try:
            return float(10.0 ** exponent)
        except OverflowError:
            return np.inf

    def _working_norm(self):
        """Return the raw canonical-centre norm without ``tn.exponent``.

        Non-unitary replay uses this to keep tensor entries numerically scaled
        while preserving the removed global factor in Quimb's exponent.
        Establishing a centre here is lossless and only needed for an
        unmanaged/unknown canonical gauge.
        """
        center = self.center
        if center is None:
            region = self.tn.canonical_region
            center = min(region) if region else self.plan.root
            self._move_center(center)
        if self.tn.fermionic:
            with self._thread_ctx():
                val = self.tn._fermionic_center_norm_squared(center)
        else:
            tensor = self.tn.tensor_map[self._tid(center)]
            val = qtn.tensor_contract(tensor.H, tensor, output_inds=[])
        return float(np.sqrt(abs(to_float(val, real=True))))

    def _normalize_working_scale(self, eps=1e-15):
        """Normalize canonical working data and preserve represented scale."""
        working_norm = self._working_norm()
        if working_norm > float(eps) and np.isfinite(working_norm):
            self._invalidate_state_norm_cache()
            tensor = self.tn.tensor_map[self._tid(self.center)]
            tensor.modify(data=tensor.data / working_norm)
            self.tn.exponent = (
                float(getattr(self.tn, "exponent", 0.0))
                + float(np.log10(working_norm))
            )
        return working_norm

    def _normalize_and_record_working_scale(
        self, *, step, support, reason, eps,
    ):
        """Apply one non-unitary scale-control event and record its factor."""
        old_norm = self._normalize_working_scale(eps=eps)
        log10_scale = (
            float(np.log10(old_norm)) if old_norm > 0.0 else -np.inf
        )
        support = tuple(support)
        event = {
            "step": int(step),
            "old_norm": float(old_norm * old_norm),
            "span": support,
            "insert": self.center,
            "sites": support,
            "scales": (float(old_norm),),
            "log10_scale": log10_scale,
            "log10_scales": (log10_scale,),
            "reason": str(reason),
            "method": "canonical_center",
            "exponent": float(getattr(self.tn, "exponent", 0.0)),
        }
        self.normalizations.append(event)
        return event

    def _leaf_canonical_norm(self):
        """Return a cheap dense canonical norm or the exact fermionic norm.

        Dense/nonfermionic trees move the orthogonality centre onto a
        first-layer leaf and read that tensor's Frobenius norm. Native
        fermionic trees must retain the complete graded exterior, so they
        dispatch to :meth:`norm` instead of using the one-tensor shortcut.
        """
        if self.tn.fermionic:
            return self.norm()
        site_node = self.plan.node_of_qubit[min(self.plan.node_of_qubit)]
        self._move_center(site_node)
        t = self.tn.tensor_map[self._tid(site_node)]
        val = qtn.tensor_contract(t.H, t, output_inds=[])
        return float(np.sqrt(abs(to_float(val, real=True))))

    def normalize(self, eps=1e-15, insert=None):
        """Normalise the state in place and return the previous norm.

        ``eps`` and ``insert`` are accepted for compatibility with
        :meth:`MpsOptimizer.normalize`; a tree has no chain insertion site, so
        ``insert`` is intentionally ignored. Unlike the non-unitary replay
        scale-control path, this is a physical renormalization: any accumulated
        Quimb ``tn.exponent`` is cleared so the represented state has unit norm.
        """
        eps = float(eps)
        if eps < 0.0:
            raise ValueError("eps must be non-negative.")
        nrm = self.norm()
        working_norm = self._working_norm()
        if working_norm > eps and np.isfinite(working_norm):
            self._invalidate_state_norm_cache()
            tid = self._tid(self.center)
            t = self.tn.tensor_map[tid]
            t.modify(data=t.data / working_norm)
            if hasattr(self.tn, "exponent"):
                self.tn.exponent = 0.0
        return nrm

    def _measure_pauli(self, pauli, where, outcome=None, *, renormalize=True,
                       return_diagnostics=False):
        """Measure a Pauli product, collapse, and return ``(outcome, prob)``."""
        self._require_dense_qubit_state("measure_pauli")
        logical_where = _normalize_control_where(where)
        where = self._validate_support(logical_where)
        axes = _normalize_measure_axes(pauli, logical_where)
        expectation = to_float(
            self._product_pauli_expectation(axes, where), real=True
        )
        p_plus = min(max(0.5 * (1.0 + expectation), 0.0), 1.0)
        if outcome is None:
            outcome = 1 if self.rng.random() < p_plus else -1
        elif not isinstance(outcome, Integral) or int(outcome) not in (-1, 1):
            raise ValueError("measure event outcome must be +1 or -1.")
        else:
            outcome = int(outcome)
        probability = p_plus if outcome > 0 else 1.0 - p_plus
        if probability <= 1e-12:
            raise ValueError(
                f"forced measure outcome {outcome} has ~0 probability "
                f"({probability:.2e})."
            )
        diagnostics = self._apply_product_pauli_projector(
            axes, where, outcome, renormalize=renormalize,
            return_diagnostics=return_diagnostics,
            logical_support=logical_where, probability=probability,
        )
        if return_diagnostics:
            return int(outcome), float(probability), diagnostics
        return int(outcome), float(probability)

    def measure_pauli(self, pauli, where, outcome=None, *, renormalize=True,
                      return_diagnostics=False):
        """Measure a product Pauli and return ``(outcome, probability)``.

        ``pauli`` supplies one ``X``, ``Y``, or ``Z`` axis per site in
        ``where``.  An outcome is sampled unless forced to ``+1`` or ``-1``.
        The state is normalized after collapse by default; set
        ``renormalize=False`` to retain the branch norm.  Requesting
        diagnostics adds a third return value containing projection norm,
        support/span, and bond-dimension before/after information.
        """
        return self._measure_pauli(
            pauli, where, outcome, renormalize=renormalize,
            return_diagnostics=return_diagnostics,
        )

    def _reset_pauli(self, q, axis):
        """Reset one qubit to the ``+1`` eigenstate of ``axis``."""
        outcome, _ = self._measure_pauli(axis, (q,))
        if outcome < 0:
            flip = _RESET_FLIP_AXES[axis]
            self.apply_1q(
                self._as_state_backend(_PAULI_1Q[flip], warn=False), q
            )

    def _apply_control_event(self, name, payload, where):
        """Apply one normalized measure/reset event from the queued stream."""
        with self._thread_ctx():
            return self._apply_control_event_impl(name, payload, where)

    def _apply_control_event_impl(self, name, payload, where):
        """Apply a control event without opening another thread context."""
        if name == "conditional":
            record_index, expected = _resolve_conditional(
                payload, len(self.measurements)
            )
            record = self.measurements[record_index]
            outcome = (
                int(record.outcome)
                if hasattr(record, "outcome")
                else int(record[2])
            )
            if int(outcome < 0) != expected:
                return self
            action_payloads, action_wheres, action_types = self._normalize_gate_queue(
                (payload["action"],)
            )
            if len(action_payloads) != 1:
                raise ValueError(
                    "conditional action must normalize to exactly one stream entry."
                )
            action_payload = action_payloads[0]
            action_where = action_wheres[0]
            action_type = action_types[0]
            if action_type == "gate":
                self.apply_gate(action_payload, action_where)
            elif action_type == "submpo":
                action_support = self._validate_support(action_where)
                self._apply_submpo_resolved(
                    action_payload,
                    action_support,
                    logical_where=action_where,
                    max_bond=self.chi,
                    cutoff=self.cutoff,
                )
            else:
                self._apply_control_event_impl(
                    action_type, action_payload, action_where
                )
            return self
        if name == "cap":
            self.cap(
                where[0], payload["vec"],
                absorb=payload.get("absorb", "left"),
                compact_labels=payload.get("compact_labels", True),
            )
            return self
        if name == "measure":
            outcome, probability = self._measure_pauli(
                payload["pauli"], where, payload.get("outcome")
            )
            self.measurements.append(
                (str(payload["pauli"]), tuple(where), outcome, probability)
            )
            return self
        if name == "reset":
            for q, axis in zip(where, payload["axes"]):
                self._reset_pauli(q, axis)
            return self
        if name == "measure_reset":
            for q, axis, forced in zip(
                where, payload["axes"], payload["outcomes"]
            ):
                outcome, probability = self._measure_pauli(axis, (q,), forced)
                self.measurements.append(
                    (axis, (q,), outcome, probability)
                )
                if outcome < 0:
                    self.apply_1q(
                        self._as_state_backend(
                            _PAULI_1Q[_RESET_FLIP_AXES[axis]], warn=False
                        ),
                        q,
                    )
            return self
        raise ValueError(f"Unknown TreeOptimizer control event {name!r}.")

    def measure(self, q, outcome=None):
        """Projectively measure qubit ``q`` in the computational basis.

        Moves the orthogonality centre onto the site node, reads the Born
        probabilities from that one canonical tensor, samples (or forces via
        ``outcome``) a result, projects the site, and renormalises.
        Returns the outcome bit. Because the centre sits on the site node the
        probabilities are exact regardless of the global state norm.
        """
        self._require_dense_qubit_state("measure")
        with self._thread_ctx():
            q = self._validate_qubit(q)
            site_node = self.plan.node_of_qubit[q]
            self._move_center(site_node)
            t = self.tn.tensor_map[self._tid(site_node)]
            p = self._phys(q)
            ax = t.inds.index(p)
            arr = ar.do("reshape", ar.do("moveaxis", t.data, ax, 0), (2, -1))
            w = ar.do("sum", arr * ar.do("conj", arr), axis=1)
            w = np.real(ar.to_numpy(w))
            total = float(w.sum())
            if total <= 0:
                raise ValueError("Cannot measure a zero-norm state.")
            probs = np.clip(w, 0.0, None)
            probs = probs / probs.sum()
            if outcome is None:
                outcome = int(self.rng.choice(2, p=probs))
            else:
                if not isinstance(outcome, Integral) or int(outcome) not in (0, 1):
                    raise ValueError("measurement outcome must be 0 or 1.")
                outcome = int(outcome)
                if probs[outcome] <= 1e-12:
                    raise ValueError(
                        f"forced measure outcome {outcome} has ~0 probability."
                    )
            proj = np.zeros((2, 2), dtype=complex)
            proj[outcome, outcome] = 1.0
            self.apply_1q(self._as_state_backend(proj, warn=False), q)
            self.normalize()
        return outcome

    def reset(self, q):
        """Reset qubit ``q`` to ``|0>`` (measure, then flip if it was ``|1>``)."""
        if self.measure(q) == 1:
            x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
            self.apply_1q(self._as_state_backend(x, warn=False), q)
        return 0

    def cap(self, q, vec, *, absorb="left", compact_labels=True,
            stable_labels=None):
        """Contract and remove qubit ``q`` from the tree state.

        This is the tree analogue of an MPS physical-index cap. Trees have no
        left/right chain neighbour at a leaf, so ``absorb`` is accepted for
        stream compatibility but the unique parent is used.  By default the
        remaining caller-facing labels are compacted above ``q``.  Set
        ``compact_labels=False`` (or ``stable_labels=True``) to retain the
        original logical labels while the internal TTN stays compact.
        """
        if stable_labels is not None:
            compact_labels = not bool(stable_labels)
        if not isinstance(compact_labels, (bool, np.bool_)):
            raise ValueError("compact_labels must be boolean.")
        self._invalidate_state_norm_cache()
        logical_q = int(q)
        q = self._validate_qubit(logical_q)
        if absorb not in {"left", "right"}:
            raise ValueError("cap absorb direction must be 'left' or 'right'.")
        started = self._begin_update("cap", (q,))
        try:
            self.tn.cap_qubit_(q, self._as_state_backend(vec))
            self.plan = self.tn.plan
            self._two_site_path_cache.clear()
            self.n = self.tn.nqubits
            remaining = [label for label in self._logical_qubits if label != logical_q]
            if compact_labels:
                remaining = [
                    label - 1 if label > logical_q else label
                    for label in remaining
                ]
            self._logical_qubits = remaining
            self._logical_positions = {
                label: position for position, label in enumerate(remaining)
            }
            self.layout_finder = None
        except Exception:
            if started:
                self._abort_update()
            raise
        if started:
            self._finish_update()
        return self

    def copy(self):
        """Return an independent optimizer at the current tree state.

        The returned optimizer owns its own copy of the live tensor network --
        which carries the tracked orthogonality centre with it -- so it can be
        evolved separately, useful for branching experiments or trial gate
        sequences.  The immutable :class:`TreePlan` is shared; the gate queue is
        retained (gate payloads are not copied) but not replayed.
        """
        child_seed = int(self.rng.integers(0, 2**63, dtype=np.uint64))
        other = type(self)(
            None,
            n=self.n,
            chi=self.chi,
            cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode,
            mode=self.mode,
            compression_mode=self.compression_mode,
            compression_seed=self.compression_seed,
            fit_block_size=self.fit_block_size,
            fit_n_iter=self.fit_n_iter,
            fit_adaptive_sweeps=self.fit_adaptive_sweeps,
            fit_min_iter=self.fit_min_iter,
            fit_rtol=("auto" if self._fit_rtol_requested == "auto" else self.fit_rtol),
            fit_patience=self.fit_patience,
            fit_init_strategy=self.fit_init_strategy,
            fit_init_rand_strength=self.fit_init_rand_strength,
            fit_init_seed=self.fit_init_seed,
            fit_sweep_sequence=self.fit_sweep_sequence,
            fit_overlap_diagnostics=self.fit_overlap_diagnostics,
            fit_finite_check=self.fit_finite_check,
            structure=self.structure,
            max_arity=self.max_arity,
            top_arity=self.top_arity,
            community_frac=self.community_frac,
            star_frac=self.star_frac,
            tree=self.plan,
            dtype=self.dtype,
            threads=self.threads,
            subtree_workers=self.subtree_workers,
            layout_objective=self.layout_objective,
            layout_weight_mode=self.layout_weight_mode,
            layout_time_decay=self.layout_time_decay,
            layout_time_window=self.layout_time_window,
            track_truncation=self.track_truncation,
            track_infidelity=self.track_infidelity,
            max_intermediate_bond=self.max_intermediate_bond,
            max_operator_qubits=self.max_operator_qubits,
            max_subtree_nodes=self.max_subtree_nodes,
            record_history=self.record_history,
            profile=self.profile,
            profile_sync=self.profile_sync,
            track_bond_diagnostics=self.track_bond_diagnostics,
            seed=child_seed,
            run=False,
            tn=self.tn,
        )
        other.G = list(self.G)
        other.where = list(self.where)
        other.event_types = list(self.event_types)
        other.layout_finder = self.layout_finder
        other.measurements = deepcopy(self.measurements)
        other.truncation_history = deepcopy(self.truncation_history)
        other.update_history = deepcopy(self.update_history)
        other.bond_history = deepcopy(self.bond_history)
        other.infidelities = list(self.infidelities)
        other.infidelity_samples = deepcopy(self.infidelity_samples)
        other.norm_events = deepcopy(self.norm_events)
        other.normalizations = deepcopy(self.normalizations)
        other.projection_diagnostics = deepcopy(self.projection_diagnostics)
        other._backend_conversion_warnings = set(
            self._backend_conversion_warnings
        )
        other._track_warning_emitted = self._track_warning_emitted
        other._logical_qubits = list(self._logical_qubits)
        other._logical_positions = dict(self._logical_positions)
        other._update_counter = self._update_counter
        other._truncation_log_survival = self._truncation_log_survival
        other._norm_log_survival = self._norm_log_survival
        other._norm_tracking_enabled = self._norm_tracking_enabled
        other.profile_events = deepcopy(self.profile_events)
        other._dmrg_mode_alias = self._dmrg_mode_alias
        other.fit_diagnostics = deepcopy(self.fit_diagnostics)
        other._last_fit_diagnostics = deepcopy(self._last_fit_diagnostics)
        other._attach_profile_sink()
        return other

    def get_infidelities(self):
        """Return cumulative tracked tree-truncation infidelities.

        The first value is ``0.0``. Additional values are recorded after
        updates for which ``track_truncation=True`` supplied singular spectra.
        Unlike a chain MPS, a tree has several compression edges per update,
        so each value aggregates the retained weight over all touched edges.
        """
        return self.infidelities

    def get_infidelity_samples(self):
        """Return detailed cumulative tree-truncation sample records."""
        return self.infidelity_samples

    def bond_diagnostic_report(self):
        """Return live-versus-transient bond diagnostics collected per update.

        The transient maximum includes the bond dimensions presented to the
        final compression SVD and the dimensions observed after exact QR
        thread hops. Consequently it can exceed ``chi`` even when every
        ``live_max_bond_after`` is capped by ``chi``. The report is populated
        only when ``track_bond_diagnostics=True``; otherwise the update list
        remains available but its live/transient fields are ``None``.
        """
        updates = deepcopy(
            self.bond_history if self.track_bond_diagnostics else self.update_history
        )
        measured = [
            update for update in updates
            if update.get("transient_max_bond") is not None
        ]
        live_after = [
            update["live_max_bond_after"]
            for update in measured
            if update.get("live_max_bond_after") is not None
        ]
        transient = [
            update["transient_max_bond"]
            for update in measured
            if update.get("transient_max_bond") is not None
        ]
        return {
            "enabled": self.track_bond_diagnostics,
            "chi": self.chi,
            "updates": updates,
            "max_live_bond_after": max(live_after) if live_after else None,
            "max_transient_bond": max(transient) if transient else None,
            "n_transient_exceeds_chi": sum(
                bool(update.get("transient_exceeds_chi"))
                for update in measured
            ),
        }

    def profile_report(self):
        """Return opt-in tree kernel timings grouped by operation kind.

        Construct the optimizer with ``profile=True`` to collect records.
        Timing is deliberately kept separate from truncation history so the
        normal replay and diagnostic APIs remain unchanged. In addition to
        update and compression events, two-site direct routing reports each
        exact QR ``thread_hop`` separately. Native compression events also
        identify whether the reduced graded QR/SVD route or the conservative
        full two-node SVD was selected. The returned ``events`` list is a deep
        copy and can safely be serialized alongside a benchmark result.
        """
        events = deepcopy(self.profile_events)
        grouped = {}
        native_routes = {}
        for event in events:
            kind = str(event.get("kind", "unknown"))
            summary = grouped.setdefault(
                kind, {"count": 0, "seconds": 0.0}
            )
            summary["count"] += 1
            summary["seconds"] += float(event.get("seconds", 0.0))
            if kind == "native_compression_route":
                route = str(event.get("route", "unknown"))
                native_routes[route] = native_routes.get(route, 0) + 1
        update_seconds = float(grouped.get("update", {}).get("seconds", 0.0))
        return {
            "enabled": self.profile,
            "events": events,
            "by_kind": grouped,
            "native_compression_routes": native_routes,
            "update_seconds": update_seconds,
            "timing_semantics": {
                "wall_envelope": "update",
                "nested_event_kinds": [
                    "gate_factorization",
                    "center_movement",
                    "metadata_path",
                    "thread_hop",
                    "tensor_absorption",
                    "edge_canonize",
                    "edge_compress",
                    "subtree_hub_merge",
                    "native_compression_route",
                ],
                "total_seconds_is_sum_of_events_not_wall_time": True,
            },
            "total_seconds": float(
                sum(float(event.get("seconds", 0.0)) for event in events)
            ),
        }

    def get_normalizations(self):
        """Return automatic normalization records.

        Non-unitary replay records each raw canonical-centre scale removed
        from tensor data and accumulated into ``tn.exponent``. Thus the
        working tensors remain normalized without changing the represented
        state norm.
        """
        return self.normalizations

    def get_norm_events(self):
        """Return path-level retained-norm compression events.

        These events are collected independently of ``track_truncation``.
        Each event represents the complete gate/subtree path update, whereas
        :meth:`truncation_report` contains optional per-edge spectrum data.
        """
        return deepcopy(self.norm_events)

    def norm_diagnostics(self):
        """Return canonical norm-based compression diagnostics.

        ``cumulative_fidelity`` is the log-accumulated product of the
        path-level fidelities measured from retained norms. It is a
        compression proxy, not a directional overlap with a target state.
        Tree target-overlap checks, when a caller has an exact reference, must
        be reported separately.
        ``track_truncation`` is intentionally exposed as an independent flag:
        it controls expensive per-edge singular-spectrum probes only.

        ``state_norm`` and ``norm`` are the live represented Tree norm.
        ``cumulative_norm`` is instead the square root of
        ``cumulative_fidelity`` and is only a retained-compression proxy.
        """
        valid = [event for event in self.norm_events if event.get("valid")]
        current = valid[-1] if valid else None
        cumulative_fidelity = (
            None
            if not valid
            else fidelity_from_log(self._norm_log_survival)
        )
        cumulative_infidelity = (
            None
            if cumulative_fidelity is None
            else infidelity_from_log(self._norm_log_survival)
        )
        state_norm = float(self.norm())
        return {
            "tracking": bool(self.track_infidelity),
            "norm_tracking": bool(self.track_infidelity),
            "truncation_tracking": bool(self.track_truncation),
            "events": len(self.norm_events),
            "completed_events": len(valid),
            "current_valid": current is not None,
            "current_fidelity": (
                None if current is None else current["local_fidelity"]
            ),
            "current_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "current_segment_norm": (
                None
                if current is None
                else float(current["local_fidelity"] ** 0.5)
            ),
            "current_segment_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "local_fidelity": (
                None if current is None else current["local_fidelity"]
            ),
            "local_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "local_norm": (
                None
                if current is None
                else float(current["local_fidelity"] ** 0.5)
            ),
            "cumulative_fidelity": cumulative_fidelity,
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": cumulative_fidelity,
            "cumulative_compression_infidelity": cumulative_infidelity,
            # ``norm_survival`` records the norm-derived provenance; it is not
            # the live ``norm`` returned below. ``fidelity`` and ``infidelity``
            # are cumulative convenience
            # aliases; the explicit names above distinguish local from total.
            "norm_survival": cumulative_fidelity,
            "fidelity": cumulative_fidelity,
            "infidelity": cumulative_infidelity,
            "state_norm": state_norm,
            "cumulative_norm": (
                None
                if cumulative_fidelity is None
                else float(cumulative_fidelity ** 0.5)
            ),
            # ``norm`` historically means the represented Tree norm. Keep it
            # as a compatibility alias; ``cumulative_norm`` is the retained
            # norm proxy shared with the MPS diagnostics.
            "norm": state_norm,
            "norm_events": self.get_norm_events(),
        }

    def get_projection_diagnostics(self):
        """Return projection norm/support/span/bond diagnostics in order."""
        return self.projection_diagnostics

    def get_fit_diagnostics(self):
        """Return the latest TreeFIT diagnostic record, if available."""

        return None if self._last_fit_diagnostics is None else deepcopy(
            self._last_fit_diagnostics
        )

    @classmethod
    def find_tree_layout(cls, gates, n=None, *, structure="quality",
                         max_arity=2, community_frac=0.35,
                         star_frac=0.75, layout_objective="congestion",
                         layout_weight_mode="count",
                         layout_time_decay=None, layout_time_window=None,
                         root_qubit=None, top_arity=_DEFAULT_TOP_ARITY,
                         max_operator_qubits=_DEFAULT_MAX_OPERATOR_QUBITS,
                         lattice_shape=None, lattice_site=None,
                         coarse_grain=(2, 1),
                         order=None, map_mode=None):
        """Return the :class:`TreePlan` a :class:`TreeLayoutFinder` would use."""
        return TreeLayoutFinder(
            gates=gates, n=n, structure=structure,
            max_arity=max_arity, community_frac=community_frac,
            star_frac=star_frac, objective=layout_objective,
            top_arity=top_arity,
            weight_mode=layout_weight_mode,
            time_decay=layout_time_decay,
            time_window=layout_time_window,
            root_qubit=root_qubit,
            max_operator_qubits=max_operator_qubits,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
            coarse_grain=coarse_grain,
            order=order,
            map_mode=map_mode,
        ).run()

    @classmethod
    def convergence_sweep(cls, gates, n=None, chi_values=(2, 4, 8, 16, 32), *,
                          ops=None, structure="quality", max_arity=2,
                          community_frac=0.35, star_frac=0.75, tree=None,
                          root_qubit=None, top_arity=_DEFAULT_TOP_ARITY,
                          dense_cap=1 << 14):
        """Replay ``gates`` at several ``chi`` and report convergence.

        The tree structure is built once and reused for every ``chi`` so the
        comparison isolates the truncation effect from the layout choice.  For
        each ``chi`` the achieved ``max_bond``, the state ``norm``, and the
        expectation of every ``(operator, where)`` in ``ops`` are recorded.
        When the Hilbert space is small (``2**n <= dense_cap``) the fidelity
        against the untruncated statevector is also reported; the observable
        drift between consecutive ``chi`` values is always reported as a
        reference-free convergence signal.

        Parameters
        ----------
        ops : sequence of ``(operator, where)``, optional
            Observables tracked across ``chi``, labelled ``op{i}`` in the output.
        chi_values : sequence of int
            Bond-dimension caps to sweep (sorted ascending internally).
        tree : TreePlan, optional
            Fixed structure to reuse; inferred once from ``gates`` when omitted.
        dense_cap : int
            Skip the exact-fidelity reference when ``2**n`` exceeds this.

        Returns
        -------
        list of dict
            One record per ``chi`` (ascending) with ``chi``, ``max_bond``,
            ``norm``, ``expectations``, ``fidelity`` (or ``None``), and
            ``max_drift`` (max ``|Delta<op>|`` from the previous ``chi``, or
            ``None`` for the first).
        """
        if hasattr(gates, "__next__"):
            gates = list(gates)
        chi_values = sorted(int(c) for c in chi_values)
        if tree is None:
            probe = cls(gates, n=n, structure=structure, max_arity=max_arity,
                        top_arity=top_arity, community_frac=community_frac,
                        star_frac=star_frac, root_qubit=root_qubit, run=False)
            tree = probe.plan
            n = probe.n
        elif n is None:
            n = tree.n

        exact_state = None
        if (1 << n) <= dense_cap:
            ref = cls(gates, n=n, tree=tree, chi=(1 << n), run=True)
            exact_state = ref.to_dense()
            nrm = np.linalg.norm(exact_state)
            if nrm > 0:
                exact_state = exact_state / nrm

        ops = list(ops) if ops else []
        records = []
        prev_vals = None
        for chi in chi_values:
            opt = cls(gates, n=n, tree=tree, chi=chi, run=True)
            expectations = {}
            vals = []
            for i, (op, where) in enumerate(ops):
                # Observable evaluation belongs to the state object. Keep the
                # replay optimizer focused on state evolution rather than a
                # qubit-only local-expectation implementation.
                val = complex(opt.tn.local_expectation(
                    op, _normalize_where(where), max_bond=None,
                    optimize="auto",
                ))
                expectations[f"op{i}"] = val
                vals.append(val)
            fidelity = None
            if exact_state is not None:
                psi = opt.to_dense()
                nrm = np.linalg.norm(psi)
                if nrm > 0:
                    psi = psi / nrm
                    fidelity = float(abs(np.vdot(exact_state, psi)) ** 2)
            max_drift = None
            if prev_vals is not None and vals:
                max_drift = float(
                    max(abs(a - b) for a, b in zip(vals, prev_vals))
                )
            if vals:
                prev_vals = vals
            records.append({
                "chi": chi,
                "max_bond": opt.max_bond(),
                "norm": opt.norm(),
                "expectations": expectations,
                "fidelity": fidelity,
                "max_drift": max_drift,
            })
        return records

    def __repr__(self):
        return (
            f"TreeOptimizer(n={self.n}, chi={self.chi}, "
            f"mode={self.mode!r}, "
            f"compression_mode={self.compression_mode!r}, "
            f"max_bond={self.max_bond()}, gates={len(self.G)})"
        )
