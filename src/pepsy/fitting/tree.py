"""Tree-native local variational fitting.

This module contains the tree counterpart of :class:`pepsy.fitting.FIT`.
Unlike the chain FIT implementation, a tree fit has no distinguished left and
right boundary.  It therefore caches one overlap environment for every
directed tree edge and moves an explicit orthogonality centre along the unique
tree geodesic between local update blocks.

The fitted network is expected to expose the small geometry interface supplied
by ``TreeTensorNetwork`` and ``TreePeps``: ``plan``, ``node_tensor``, ``bond``,
``neighbors``, ``site_ind``, canonicalization methods, and ``copy``. The target
can be one of those tree objects or a plain Quimb tensor network whose layer
tensors are tagged by the fitted structural node tags. Keeping this interface
duck-typed lets the same kernel serve both tree optimizers without importing
either optimizer here.
"""

from __future__ import annotations

from itertools import combinations
from functools import lru_cache
import inspect
import math
import warnings
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from .._internal.random import backend_random_array

__all__ = ["TreeFIT"]


def _native_blockwise_tensordot(a, b, axes):
    """Use Symmray's graded contraction without fusing its charge blocks."""
    import symmray as sr

    return sr.tensordot(a, b, axes=axes, mode="blockwise")


@lru_cache(maxsize=1)
def _native_environment_implementation():
    """Capability-gate the optional, per-contraction upstream implementation."""
    import cotengra as ctg
    import symmray as sr

    if (
        "mode" not in inspect.signature(sr.AbelianArray.tensordot).parameters
        or "implementation" not in inspect.signature(
            ctg.ContractionTree.get_contractor
        ).parameters
    ):
        raise NotImplementedError(
            "native-blockwise requires Symmray tensordot(mode=...) and "
            "Cotengra's per-contraction implementation option"
        )
    return sr.einsum, _native_blockwise_tensordot


def _randomize_tree_guess(
    state,
    region,
    *,
    target=None,
    max_bond=None,
    strength=0.0,
    expand=False,
    seed=0,
):
    """Build a deterministic dense randomized warm-start for ``TreeFIT``.

    ``random`` perturbs the existing active tensors. ``random_expand`` also
    grows active tree bonds towards the exact target rank (capped by
    ``max_bond``), filling only the new directions with seeded noise. Native
    Symmray/fermionic data is left untouched because dense noise cannot
    preserve its charge sectors and graded index metadata.
    """

    region = frozenset(region)
    info = {
        "enabled": False,
        "rand_strength": float(strength),
        "expanded": bool(expand),
        "bonds": [],
        "sites": [],
        "reason": None,
    }
    guess = state.copy()
    if float(strength) == 0.0:
        info["reason"] = "disabled"
        return guess, info

    tensors = [_tensor_of(guess, node) for node in _nodes_of(guess)]
    if any(
        ar.infer_backend(tensor.data) == "symmray"
        or bool(getattr(tensor.data, "fermionic", False))
        for tensor in tensors
    ):
        info["reason"] = "native_sector_growth"
        return guess, info

    rng = np.random.default_rng(int(seed))
    if expand and target is not None:
        tree_edges = tuple(
            (node0, node1)
            for node0 in _nodes_of(guess)
            for node1 in _neighbors_of(guess, node0)
            if node0 < node1
        )
        target_bond_sizes = None
        if not hasattr(target, "bond"):
            target_bond_sizes = _layered_target_bond_sizes(
                target,
                guess,
                tree_edges,
            )
        planned = []
        for node0, node1 in tree_edges:
            if node0 not in region or node1 not in region:
                continue
            fitted_bond = guess.bond(node0, node1)
            current = int(guess.ind_size(fitted_bond))
            if target_bond_sizes is None:
                target_bond = target.bond(node0, node1)
                target_rank = int(target.ind_size(target_bond))
            else:
                target_rank = int(target_bond_sizes[(node0, node1)])
            if max_bond is not None:
                target_rank = min(target_rank, int(max_bond))
            if target_rank > current:
                planned.append((node0, node1, current, target_rank, fitted_bond))

        # Quimb expands all requested indices to the same minimum size, so
        # process bonds in rank groups and then add noise to only their new
        # slices. This is valid for arbitrary tree degree, not just paths.
        for target_rank in sorted({item[3] for item in planned}):
            group = [item for item in planned if item[3] == target_rank]
            guess.expand_bond_dimension(
                target_rank,
                mode="zeros",
                inds_to_expand=[item[4] for item in group],
                inplace=True,
            )
            for node0, node1, current, _, fitted_bond in group:
                for node in (node0, node1):
                    tensor = _tensor_of(guess, node)
                    axis = tensor.inds.index(fitted_bond)
                    old_slices = [slice(None)] * tensor.ndim
                    old_slices[axis] = slice(0, current)
                    new_shape = list(tensor.shape)
                    new_shape[axis] = target_rank - current
                    random_data = backend_random_array(
                        new_shape,
                        like=tensor.data,
                        dtype=getattr(tensor.data, "dtype", None),
                        scale=float(strength),
                        rng=rng,
                    )
                    old_data = tensor.data[tuple(old_slices)]
                    tensor.modify(data=ar.do(
                        "concatenate", (old_data, random_data), axis=axis
                    ))
                info["bonds"].append({
                    "bond": tuple(sorted((node0, node1))),
                    "current_rank": current,
                    "target_rank": target_rank,
                    "new_rank": int(guess.ind_size(fitted_bond)),
                })

    for node in sorted(region):
        tensor = _tensor_of(guess, node)
        random_data = backend_random_array(
            tensor.shape,
            like=tensor.data,
            dtype=getattr(tensor.data, "dtype", None),
            scale=float(strength),
            rng=rng,
        )
        tensor.modify(data=ar.do("add", tensor.data, random_data))
        info["sites"].append(node)

    invalidate = getattr(guess, "invalidate_canonical_form", None)
    if callable(invalidate):
        invalidate()
    guess.canonize_subtree_(region)
    info["enabled"] = True
    return guess, info


def _nodes_of(state):
    """Return all structural nodes in deterministic order."""

    plan = state.plan
    nodes = getattr(plan, "nodes", None)
    if callable(nodes):
        return tuple(sorted(nodes()))
    return tuple(sorted(state.sites))


def _neighbors_of(state, node):
    """Return structural neighbours for either tree state class."""

    neighbors = getattr(state, "neighbors", None)
    if callable(neighbors):
        return tuple(neighbors(node))
    return tuple(state.plan.neighbors(node))


def _path_of(state, node0, node1):
    """Return the unique structural path between two nodes."""

    path = getattr(state, "node_path", None)
    if callable(path):
        return tuple(path(node0, node1))
    return tuple(state.plan.path(node0, node1))


def _is_connected(state, nodes):
    """Check connectivity for both tree-plan implementations."""

    nodes = frozenset(nodes)
    if len(nodes) <= 1:
        return True
    checker = getattr(state.plan, "is_connected", None)
    if callable(checker):
        return bool(checker(nodes))
    start = next(iter(nodes))
    reached = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in _neighbors_of(state, node):
            if neighbor in nodes and neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    return reached == nodes


def _component_of(state, start, blocked):
    """Return the component containing ``start`` after cutting an edge."""

    component = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in _neighbors_of(state, node):
            if node == start and neighbor == blocked:
                continue
            if neighbor not in component:
                component.add(neighbor)
                stack.append(neighbor)
    return frozenset(component)


def _physical_ind(state, node):
    """Return the physical index on ``node``, or ``None`` for virtual nodes."""

    plan = state.plan
    qubit_of_node = getattr(plan, "qubit_of_node", None)
    if qubit_of_node is not None:
        qubit = qubit_of_node.get(node)
        if qubit is None:
            return None
        return state.site_ind(qubit)

    # TreePepsPlan has one physical site per structural node.
    try:
        return state.site_ind(node)
    except (KeyError, ValueError, IndexError):
        return None


def _tensor_of(state, node):
    """Get one structural tensor from either supported tree state."""

    return state.node_tensor(node)


def _exponent_value(network):
    """Return Quimb's represented base-ten exponent as a float."""

    value = getattr(network, "exponent", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _scalar_value(value):
    """Convert a scalar backend value into a Python complex number."""

    if hasattr(value, "data"):
        value = value.data
    try:
        value = ar.to_numpy(value)
    except (AttributeError, ImportError, TypeError, ValueError):
        value = np.asarray(value)
    return np.asarray(value).reshape(()).item()


def _scale_stripped(mantissa, exponent):
    """Reconstruct a scalar from Quimb's mantissa/base-ten exponent pair."""

    try:
        return mantissa * (10.0 ** float(exponent))
    except (OverflowError, FloatingPointError):
        return np.inf if float(exponent) >= 0.0 else 0.0


def _build_layered_operator_state_target(state, operator):
    """Build a layered operator--state target without fusing tree bonds.

    The state and operator retain independent virtual tree layers. At a
    physical node, only the operator input and state physical leg are joined
    through a fresh internal index; the operator output is renamed to the
    state's physical index. This is the tree equivalent of the ordinary
    two-layer MPS operator application network and is accepted by ``TreeFIT``
    as a correctly tagged layered target.
    """

    if getattr(operator, "tree_networks", None) is not None and len(
        operator.tree_networks
    ) != 1:
        raise ValueError(
            "layered TreeFIT targets require one operator tensor network"
        )

    state_tensors = []
    operator_tensors = []
    state_node_tag = getattr(state, "node_tag", None)
    if not callable(state_node_tag):
        state_node_tag = lambda node: f"N{node}"

    plan = state.plan
    for node in _nodes_of(state):
        state_tensor = _tensor_of(state, node).copy()
        operator_tensor = operator.node_tensor(node).copy()
        qubit_of_node = getattr(plan, "qubit_of_node", None)
        if qubit_of_node is None:
            qubit = node
        else:
            qubit = qubit_of_node.get(node)
        if qubit is not None:
            physical = state.site_ind(qubit)
            intermediate = f"_pepsy_fit_input_{qtn.rand_uuid()}"
            if callable(getattr(operator, "input_ind", None)):
                operator_input = operator.input_ind(qubit)
                operator_output = operator.output_ind(qubit)
            else:
                # TreeMPO follows the generalized-operator convention where
                # ``lower`` is the ket input and ``upper`` is the output.
                operator_input = operator.lower_ind(qubit)
                operator_output = operator.upper_ind(qubit)
            state_tensor.reindex_({physical: intermediate})
            operator_tensor.reindex_({
                operator_input: intermediate,
                operator_output: physical,
            })
        operator_tensor.modify(
            tags=set(operator_tensor.tags) | {state_node_tag(node)}
        )
        state_tensors.append(state_tensor)
        operator_tensors.append(operator_tensor)

    # These tensor wrappers are already private. Transfer them through the
    # temporary layers without copying their metadata again.
    state_layer = qtn.TensorNetwork(state_tensors, virtual=True)
    operator_layer = qtn.TensorNetwork(operator_tensors, virtual=True)
    state_layer.reindex_({
        index: qtn.rand_uuid() for index in state_layer.inner_inds()
    })
    operator_layer.reindex_({
        index: qtn.rand_uuid() for index in operator_layer.inner_inds()
    })
    target = qtn.TensorNetwork([
        *state_layer.tensors,
        *operator_layer.tensors,
    ], virtual=True)
    target.exponent = (
        _exponent_value(state) + _exponent_value(operator)
    )
    return target


def _layered_target_bond_sizes(target, state, edges):
    """Return product dimensions of layered target bonds across tree edges."""

    tag_map = getattr(target, "tag_map", {})
    state_node_tag = getattr(state, "node_tag", None)
    if not callable(state_node_tag):
        state_node_tag = lambda node: f"N{node}"
    groups = {
        node: tuple(
            target.tensor_map[tid]
            for tid in tag_map.get(state_node_tag(node), ())
        )
        for node in _nodes_of(state)
    }
    result = {}
    for edge in edges:
        node0, node1 = edge
        shared = set()
        for tensor0 in groups.get(node0, ()):
            for tensor1 in groups.get(node1, ()):
                shared.update(qtn.bonds(tensor0, tensor1))
        dimension = 1
        for index in shared:
            tensor = next(
                tensor
                for tensor in groups[node0] + groups[node1]
                if index in tensor.inds
            )
            dimension *= int(tensor.ind_size(index))
        result[tuple(edge)] = dimension
    return result


class TreeFIT:
    """Locally fit a bounded-bond tree tensor network to a target tree.

    The implementation mirrors the responsibilities of the chain ``FIT``
    class while replacing chain environments with cached directed tree
    messages.  For a directed edge ``u -> v``, the cached message is the
    contraction of the target and conjugated fitted-state branches on the
    component containing ``u``.  It has the target bond legs crossing that
    tree edge and one fitted bond leg.  A local block contracts only its target
    tensors with the messages on its boundary, so repeated one-, two-, and
    three-node updates do not recontract the untouched branches. A target node
    group can contain several layer tensors and several target bonds can cross
    one fitted tree edge.

    Parameters
    ----------
    tn : tree-compatible tensor network
        Fused or correctly tagged layered target network. Every target tensor
        must carry exactly one structural node tag, and target bonds between
        different node groups must follow the fitted tree topology. Local
        layer bonds are allowed inside a node group. The target is copied by
        default and its virtual indices are privately reindexed, while
        physical indices remain shared with ``p``.
    p : tree tensor network
        Initial bounded-bond tree to optimize.
    max_bond : int or None, default=None
        Maximum dimension of internal update bonds. ``None`` keeps the current
        dimensions unless a cutoff removes directions.
    cutoffs : float, default=1e-12
        Singular-value cutoff used when splitting two- or three-node blocks.
    cutoff_mode : str, default="rsum2"
        Quimb singular-value cutoff convention.
    contraction_opt : object, default="auto-hq"
        Contraction optimizer forwarded to local environment contractions.
    traversal : {"depth", "depth-first"}, default="depth"
        Legacy depth ordering or branch-grouped depth-first local updates.
        Both visit the same blocks; truncated results can depend on order.
    environment_strategy : {"default", "native-blockwise"}, default="default"
        Optional per-contraction Symmray blockwise implementation for messages
        and effective tensors. Requires native target/state arrays and public
        upstream support; never changes global backend dispatch.
    split_method : {"direct", "dm", "src"}, default="direct"
        Local decomposition used to split fitted blocks. ``sdc`` is accepted
        as an alias for the deterministic direct local split.
    retag : bool, default=False
        Align the copied target's structural node tags with ``p``. This is
        useful when two tree objects use different ``node_tag_id`` formats;
        physical/site tags and tensor order are preserved.
    info : dict, optional
        Caller-owned diagnostics mapping. FIT-style live metadata is written
        here without replacing the supplied object.
    warning : bool, default=False
        Reserved for compatibility with FIT diagnostics and fallback warnings.
    inplace : bool, default=False
        Whether to optimize the supplied ``p`` object directly.
    copy_target : bool, default=True
        Copy ``tn`` before private virtual-index reindexing. Set to ``False``
        only when the target is disposable and ownership is transferred.
    target_norm : float or (float, float), optional
        Known exact target norm, or its (mantissa, base-ten exponent) pair.
        Enables normalized local fidelity for a lazy layered target without
        contracting it. Omit when unknown; the retained norm remains available.
    finite_check : bool, default=False
        Check active tensor entries for finite values once per sweep. The
        default does not scan arrays or numerically revalidate every isometry.
    """

    def __init__(
        self,
        tn,
        p,
        *,
        max_bond=None,
        cutoffs=1e-12,
        cutoff_mode="rsum2",
        contraction_opt="auto-hq",
        traversal="depth",
        environment_strategy="default",
        split_method="direct",
        split_seed=0,
        inplace=False,
        retag=False,
        info=None,
        warning=False,
        copy_target=True,
        target_norm=None,
        finite_check=False,
    ):
        self._validate_geometry(tn, p)
        self.traversal = self._normalize_traversal(traversal)
        self.environment_strategy = self._normalize_environment_strategy(
            environment_strategy
        )
        self._environment_contract_opts = {}
        if self.environment_strategy == "native-blockwise":
            if any(
                ar.infer_backend(t.data) != "symmray"
                for net in (tn, p) for t in net.tensors
            ):
                raise TypeError(
                    "native-blockwise requires native Symmray target and state tensors"
                )
            self._environment_contract_opts["implementation"] = (
                _native_environment_implementation()
            )
        if any(
            tensor.isfermionic() and tensor.data.parity
            for network in (tn, p) for tensor in network.tensors
        ):
            raise NotImplementedError(
                "TreeFIT does not support odd-parity fermionic tensors; "
                "use TreeOptimizer mode='direct' or mode='zipup'"
            )
        if max_bond is not None:
            if isinstance(max_bond, bool) or not isinstance(max_bond, Integral):
                raise TypeError("max_bond must be an integer or None")
            max_bond = int(max_bond)
            if max_bond < 1:
                raise ValueError("max_bond must be positive")
        cutoffs = float(cutoffs)
        if cutoffs < 0.0:
            raise ValueError("cutoffs must be non-negative")
        split_method = str(split_method).strip().lower().replace("-", "_")
        split_method = {"svd": "direct", "eigh": "dm", "sdc": "direct"}.get(
            split_method, split_method
        )
        if split_method not in {"direct", "dm", "src"}:
            raise ValueError("split_method must be 'direct', 'dm', or 'src'")
        if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
            raise TypeError("split_seed must be an integer")
        if int(split_seed) < 0:
            raise ValueError("split_seed must be non-negative")

        self.p = p if inplace else p.copy()
        self.finite_check = bool(finite_check)
        self._finite_check_warning_handled = False
        if target_norm is not None:
            target_norm = ((float(target_norm), 0.0) if np.isscalar(target_norm)
                           else tuple(float(x) for x in target_norm))
            if (len(target_norm) != 2 or target_norm[0] < 0
                    or not all(np.isfinite(x) for x in target_norm)):
                raise ValueError("target_norm must be a non-negative norm or (mantissa, exponent)")
        self._known_target_norm = target_norm
        self.tn = tn.copy() if copy_target else tn
        self.max_bond = max_bond
        self.cutoffs = cutoffs
        self.cutoff_mode = cutoff_mode
        self.contraction_opt = contraction_opt
        self.split_method = split_method
        self.split_seed = int(split_seed)
        self.nodes = _nodes_of(self.p)
        self._node_set = frozenset(self.nodes)
        self.retag = bool(retag)
        self.warning = bool(warning)
        self.info = info if info is not None else {}
        self._target_tensors = self._collect_target_groups()
        if retag:
            self._retag_target()
        self._neighbors = {
            node: tuple(sorted(_neighbors_of(self.p, node))) for node in self.nodes
        }
        self._target_physical = {
            node: _physical_ind(self.p, node) for node in self.nodes
        }
        self._validate_target_groups()
        self.target_layout = (
            "fused"
            if len(self.tn.tensors) == len(self.nodes)
            and all(len(tensors) == 1 for tensors in self._target_tensors.values())
            else "layered"
        )
        self._target_bonds = {}
        self._prepare_private_target_indices()

        self._messages = {}
        self._effective_cache = {}
        self.environment_cache_hits = 0
        self.environment_cache_misses = 0
        self.iterations_run = 0
        self.converged = False
        self.convergence_reason = None
        self.last_relative_change = None
        self.last_norm = None
        self.last_overlap = None
        self.final_center_site = None
        self.sweep_sequence = None
        self.local_norm_trace = []
        self.local_norm_stripped_trace = []
        self.sweep_norm_trace = []
        self._target_norm_stripped = None
        self.adaptive_sweeps_run = 0
        self.one_site_sweeps_run = 0
        self.block_size_trace = []
        self.timing_records = []
        self._split_counter = 0

    @staticmethod
    def _validate_geometry(target, state):
        """Validate the common tree-state interface and geometry."""

        state_required = ("plan", "node_tensor", "bond", "copy")
        if not all(hasattr(state, name) for name in state_required):
            raise TypeError("p must be a TreeTensorNetwork or TreePeps state")
        if not all(hasattr(target, name) for name in ("copy", "tensors")):
            raise TypeError(
                "tn must be a tree-compatible tensor network target"
            )
        state_nodes = set(_nodes_of(state))
        has_target_geometry = all(
            hasattr(target, name) for name in ("plan", "node_tensor", "bond")
        )
        target_nodes = set(_nodes_of(target)) if has_target_geometry else state_nodes
        if target_nodes != state_nodes:
            raise ValueError("target and fitted tree must contain the same nodes")
        structural = tuple(_tensor_of(state, node) for node in state_nodes)
        if len({id(tensor) for tensor in structural}) != len(state_nodes):
            raise ValueError("fitted tree must expose one tensor per structural node")
        tensors = getattr(state, "tensors", None)
        if tensors is not None and len(tensors) != len(state_nodes):
            raise ValueError("fitted tree must contain one tensor per structural node")
        state_edges = {
            frozenset((node, neighbor))
            for node in state_nodes
            for neighbor in _neighbors_of(state, node)
            if node != neighbor
        }
        if has_target_geometry:
            target_structural = tuple(
                _tensor_of(target, node) for node in target_nodes
            )
            if len({id(tensor) for tensor in target_structural}) != len(target_nodes):
                raise ValueError(
                    "target structural node tags must identify one backbone "
                    "tensor per tree node; additional layer tensors are allowed"
                )
            target_edges = {
                frozenset((node, neighbor))
                for node in target_nodes
                for neighbor in _neighbors_of(target, node)
                if node != neighbor
            }
            if target_edges != state_edges:
                raise ValueError(
                    "target and fitted tree must use the same tree topology"
                )
            for node in state_nodes:
                if _physical_ind(target, node) != _physical_ind(state, node):
                    raise ValueError(
                        "target and fitted tree must use matching physical indices"
                    )
                for neighbor in _neighbors_of(state, node):
                    if neighbor not in target_nodes:
                        raise ValueError(
                            "target and fitted tree use different tree plans"
                        )
                    if len(qtn.bonds(
                        _tensor_of(target, node), _tensor_of(target, neighbor)
                    )) != 1:
                        raise ValueError(
                            "target structural backbone edges must have exactly one bond"
                        )
        for node in state_nodes:
            for neighbor in _neighbors_of(state, node):
                if len(qtn.bonds(_tensor_of(state, node), _tensor_of(state, neighbor))) != 1:
                    raise ValueError("fitted tree edges must have exactly one bond")

    def _target_node_tags(self, node):
        """Return candidate structural tags for a target node."""

        tags = []
        target_node_tag = getattr(self.tn, "node_tag", None)
        if callable(target_node_tag):
            tags.append(target_node_tag(node))
        state_node_tag = getattr(self.p, "node_tag", None)
        if callable(state_node_tag):
            tags.append(state_node_tag(node))
        target_tag_id = getattr(self.tn, "_node_tag_id", None)
        if target_tag_id is not None:
            tags.append(str(target_tag_id).format(node))
        state_tag_id = getattr(self.p, "_node_tag_id", None)
        if state_tag_id is not None:
            tags.append(str(state_tag_id).format(node))
        return tuple(dict.fromkeys(tags))

    def _collect_target_groups(self):
        """Group every target tensor by one and only one node tag."""

        tensor_map = getattr(self.tn, "tensor_map", None)
        tag_map = getattr(self.tn, "tag_map", None)
        if tensor_map is None or tag_map is None:
            raise TypeError(
                "tn must expose tensor_map and tag_map so layered target "
                "tensors can be assigned to structural nodes"
            )
        tensor_order = {tid: i for i, tid in enumerate(tensor_map)}
        groups = {}
        owners = {}
        for node in self.nodes:
            tids = []
            for tag in self._target_node_tags(node):
                tids.extend(tag_map.get(tag, ()))
            tids = tuple(dict.fromkeys(tids))
            if not tids:
                raise ValueError(
                    f"target tensors for structural node {node!r} are not "
                    "tagged with its node tag"
                )
            tensors = tuple(
                tensor_map[tid]
                for tid in sorted(tids, key=tensor_order.__getitem__)
            )
            groups[node] = tensors
            for tensor in tensors:
                tensor_id = id(tensor)
                previous = owners.setdefault(tensor_id, node)
                if previous != node:
                    raise ValueError(
                        "each target tensor must carry exactly one structural "
                        "node tag; a tensor is tagged for multiple tree nodes"
                    )

        if len(owners) != len(self.tn.tensors):
            raise ValueError(
                "every target tensor must carry exactly one structural node "
                "tag; untagged or ambiguously tagged layer tensors cannot be "
                "assigned to TreeFIT"
            )
        return groups

    def _retag_target(self):
        """Align every target layer tensor with its fitted node tag."""

        target_node_tag_id = getattr(self.tn, "_node_tag_id", None)
        target_node_tag = getattr(self.tn, "node_tag", None)
        state_node_tag_id = getattr(self.p, "_node_tag_id", None)
        state_node_tag = getattr(self.p, "node_tag", None)
        if state_node_tag_id is None or not callable(state_node_tag):
            raise TypeError(
                "retag=True requires tree objects with structural node tags"
            )

        for node, tensors in self._target_tensors.items():
            target_tags = set(self._target_node_tags(node))
            target_tags.discard(state_node_tag(node))
            backbone = (
                _tensor_of(self.tn, node)
                if callable(target_node_tag)
                else None
            )
            for tensor in tensors:
                tags = set(tensor.tags)
                if tensor is not backbone:
                    tags.difference_update(target_tags)
                tags.add(state_node_tag(node))
                tensor.modify(tags=tags)

        # Keep the target's own node lookup API coherent after changing the
        # tags. A tree object reserves its native node tag for the unique
        # structural backbone tensor. Layer tensors therefore retain the
        # fitted tag while the backbone keeps the target's native tag, so the
        # tree object's node lookup still selects the backbone. Fused targets
        # can safely switch the native format.
        layered = any(len(tensors) > 1 for tensors in self._target_tensors.values())
        if hasattr(self.tn, "_node_tag_id") and not layered:
            self.tn._node_tag_id = state_node_tag_id
        if layered and target_node_tag_id is not None:
            self.info["target_node_tag_id"] = target_node_tag_id
        for cache_name in ("_node_tid_cache", "_tree_peps_tid_cache"):
            self.tn.__dict__.pop(cache_name, None)
        self.info["retagged"] = True

    def _validate_target_groups(self):
        """Check target outputs and inter-node bonds after layer grouping."""

        target_outer = set(self.tn.outer_inds())
        state_outer = set(self.p.outer_inds())
        if target_outer != state_outer:
            raise ValueError(
                "target and fitted tree must use matching physical outer indices"
            )
        for node, physical in self._target_physical.items():
            if physical is None:
                continue
            matches = [
                tensor for tensor in self._target_tensors[node]
                if physical in tensor.inds
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"target physical index {physical!r} must occur exactly "
                    f"once in structural node group {node!r}"
                )

        node_of_tensor = {
            id(tensor): node
            for node, tensors in self._target_tensors.items()
            for tensor in tensors
        }
        state_edges = {
            frozenset((node, neighbor))
            for node in self.nodes
            for neighbor in _neighbors_of(self.p, node)
            if node != neighbor
        }
        for index in self.tn.inner_inds():
            owners = set()
            for tensor_id in self.tn.ind_map.get(index, ()):
                tensor = self.tn.tensor_map[tensor_id]
                node = node_of_tensor.get(id(tensor))
                if node is not None:
                    owners.add(node)
            if len(owners) > 1:
                if len(owners) != 2 or frozenset(owners) not in state_edges:
                    raise ValueError(
                        "target inter-node virtual bonds must follow the fitted "
                        "tree topology"
                    )

    def _component(self, start, blocked):
        """Return the component containing ``start`` after cutting an edge."""

        return _component_of(self.p, start, blocked)

    def _prepare_private_target_indices(self):
        """Reindex target virtual bonds so target and state never cross-connect."""

        physical = {index for index in self._target_physical.values() if index is not None}
        mapping = {}
        all_target_tensors = [
            tensor
            for node in self.nodes
            for tensor in self._target_tensors[node]
        ]
        for tensor in all_target_tensors:
            for index in tensor.inds:
                if index not in physical and index not in mapping:
                    mapping[index] = qtn.rand_uuid()
        for tensor in all_target_tensors:
            tensor.reindex_(mapping)

        def target_bonds(node0, node1):
            bonds = []
            seen = set()
            for tensor0 in self._target_tensors[node0]:
                for tensor1 in self._target_tensors[node1]:
                    for index in qtn.bonds(tensor0, tensor1):
                        if index not in seen:
                            seen.add(index)
                            bonds.append(index)
            return tuple(bonds)

        for node in self.nodes:
            for neighbor in _neighbors_of(self.p, node):
                if (node, neighbor) in self._target_bonds:
                    continue
                target_bond = target_bonds(node, neighbor)
                if not target_bond:
                    raise ValueError(
                        f"target node groups {node!r} and {neighbor!r} must "
                        "share at least one virtual bond"
                    )
                self._target_bonds[(node, neighbor)] = target_bond
                self._target_bonds[(neighbor, node)] = target_bond

    def clear_environment_cache(self):
        """Discard cached branch entanglement environments."""

        self._messages.clear()
        self._effective_cache.clear()
        return self

    @staticmethod
    def _normalize_traversal(value):
        value = str(value).strip().lower().replace("_", "-")
        if value not in {"depth", "depth-first"}:
            raise ValueError("traversal must be 'depth' or 'depth-first'")
        return value

    @staticmethod
    def _normalize_environment_strategy(value):
        value = str(value).strip().lower().replace("_", "-")
        if value not in {"default", "native-blockwise"}:
            raise ValueError("environment_strategy must be 'default' or 'native-blockwise'")
        return value

    def environment_cache_info(self):
        """Return cache size and hit/miss counters."""

        return {
            "messages": len(self._messages),
            "effective_blocks": len(self._effective_cache),
            "hits": int(self.environment_cache_hits),
            "misses": int(self.environment_cache_misses),
        }

    def _message(self, outside, inside):
        """Assemble a directed environment from local tensors and messages.

        The explicit postorder stack avoids Python recursion on deep trees.
        Each cache miss contracts one node and its incoming environments;
        no full branch tensor network or per-edge component set is built.
        """

        key = (outside, inside)
        cached = self._messages.get(key)
        if cached is not None:
            self.environment_cache_hits += 1
            return cached
        pending = [(outside, inside, False)]
        while pending:
            node, destination, ready = pending.pop()
            edge = (node, destination)
            if edge in self._messages:
                self.environment_cache_hits += 1
                continue
            incoming = tuple(
                (neighbor, node) for neighbor in self._neighbors[node]
                if neighbor != destination
            )
            if not ready:
                pending.append((node, destination, True))
                pending.extend((u, v, False) for u, v in reversed(incoming))
                continue
            tensors = [
                *self._target_tensors[node],
                _tensor_of(self.p, node).H,
                *(self._messages[edge] for edge in incoming),
            ]
            # Canonicalization can rename fitted bonds; only the target's
            # private virtual indices remain fixed throughout the fit.
            output_inds = (
                *self._target_bonds[edge], self.p.bond(node, destination),
            )
            self._messages[edge] = qtn.tensor_contract(
                *tensors, output_inds=output_inds,
                optimize=self.contraction_opt, preserve_tensor=True, drop_tags=True,
                **self._environment_contract_opts,
            )
            self.environment_cache_misses += 1
        return self._messages[key]

    def _boundary_edges(self, block):
        """Return sorted edges crossing from ``block`` to its exterior."""

        block = frozenset(block)
        return tuple(sorted(
            (node, neighbor)
            for node in block
            for neighbor in _neighbors_of(self.p, node)
            if neighbor not in block
        ))

    def _effective_block(self, block):
        """Build the projected target tensor for a connected local block."""

        block = frozenset(block)
        key = tuple(sorted(block))
        cached = self._effective_cache.get(key)
        if cached is not None:
            self.environment_cache_hits += 1
            return cached.copy()
        self.environment_cache_misses += 1
        tensors = [
            tensor
            for node in block
            for tensor in self._target_tensors[node]
        ]
        boundary = self._boundary_edges(block)
        for inside, outside in boundary:
            tensors.append(self._message(outside, inside))
        output_inds = tuple(
            self._target_physical[node]
            for node in sorted(block)
            if self._target_physical[node] is not None
        ) + tuple(
            self.p.bond(inside, outside) for inside, outside in boundary
        )
        effective = qtn.tensor_contract(
            *tensors,
            output_inds=output_inds,
            optimize=self.contraction_opt,
            preserve_tensor=True,
            drop_tags=True,
            **self._environment_contract_opts,
        )
        if bool(getattr(effective.data, "fermionic", False)):
            # Open overlap-environment legs carry the graded bra metric.
            # Convert that covector to the fitted ket basis before writeback:
            # each dual boundary leg contributes its odd-sector parity phase.
            # Internal target legs and true physical outputs are not boundaries.
            boundary_inds = {self.p.bond(u, v) for u, v in boundary}
            axes = tuple(
                axis for axis, ind in enumerate(effective.inds)
                if ind in boundary_inds and effective.data.indices[axis].dual
            )
            if axes:
                effective.modify(data=effective.data.phase_flip(*axes))
        self._effective_cache[key] = effective.copy()
        return effective

    def _invalidate_for_block(self, block):
        """Invalidate only messages whose component contains an updated node."""

        block = frozenset(block)
        if not block:
            return
        pending = [
            (node, neighbor) for node in block for neighbor in self._neighbors[node]
        ]
        while pending:
            node, destination = pending.pop()
            if self._messages.pop((node, destination), None) is None:
                # Every cached message retains its dependencies. If this
                # input is absent, no downstream cached message can use it.
                continue
            pending.extend(
                (destination, neighbor) for neighbor in self._neighbors[destination]
                if neighbor != node
            )
        # Every effective tensor depends on the fitted exterior or on the
        # indices of its own block, so any state update invalidates it.
        self._effective_cache.clear()

    def _canonicalize_for_block(self, block, center):
        """Prepare isometric exterior branches and move the centre."""

        region = frozenset(block)
        current_region = getattr(self.p, "canonical_region", None)
        current_center = getattr(self.p, "orthogonality_center", None)
        if current_center is not None:
            # Only the exterior must be isometric towards the active block.
            # Its interior is replaced by the projected target, so QR moves
            # within that block do no useful work and invalidate more messages.
            if current_center in region:
                return
            path = _path_of(self.p, current_center, center)
            entry = next(i for i, node in enumerate(path) if node in region)
            changed_path = path[:entry + 1]
            self._invalidate_for_block(changed_path)
            self.p.shift_orthogonality_center(changed_path[-1], _skip_validate=True)
            return

        is_canonical = getattr(self.p, "is_subtree_canonical_form", None)
        if current_region != region or not (
            callable(is_canonical) and is_canonical(region)
        ):
            # With no single tracked centre there is no safe incremental path
            # to identify. Establish the block gauge once and discard the
            # basis-dependent messages conservatively.
            self.clear_environment_cache()
            self.p.canonize_subtree_(region)
        self.p.shift_orthogonality_center(center)

    def _split_method(self):
        return {
            "direct": "svd",
            "dm": "svd:eig",
            "src": "svd:rand",
        }[self.split_method]

    def _block_center(self, block, preferred=None):
        """Choose a tree-medial centre for a connected local block."""

        block = tuple(block)
        if preferred in block:
            return preferred
        return min(
            block,
            key=lambda node: (
                max(len(_path_of(self.p, node, other)) for other in block),
                sum(len(_path_of(self.p, node, other)) for other in block),
                node,
            ),
        )

    def _local_legs(self, node, block):
        """Return physical and exterior fitted-bond legs owned by ``node``."""

        inds = []
        physical = _physical_ind(self.p, node)
        if physical is not None:
            inds.append(physical)
        for neighbor in sorted(_neighbors_of(self.p, node)):
            if neighbor not in block:
                inds.append(self.p.bond(node, neighbor))
        return tuple(inds)

    def _factor_block(self, effective, block, center):
        """Factor a projected block back onto its original tree edges."""

        block = frozenset(block)
        if len(block) == 1:
            node = next(iter(block))
            return {node: effective}

        parent = {center: None}
        queue = [center]
        while queue:
            node = queue.pop(0)
            for neighbor in sorted(_neighbors_of(self.p, node)):
                if neighbor in block and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        leaves = sorted(
            node for node in block if node != center and not any(
                child in block and parent.get(child) == node for child in block
            )
        )
        remaining = effective
        factors = {}
        while leaves:
            leaf = leaves.pop(0)
            local_inds = self._local_legs(leaf, block) + tuple(
                self.p.bond(leaf, child) for child in sorted(factors)
                if parent[child] == leaf
            )
            right_inds = tuple(ind for ind in remaining.inds if ind not in local_inds)
            if not local_inds or not right_inds:
                raise ValueError(
                    "TreeFIT could not identify a non-empty local block split"
                )
            edge = (leaf, parent[leaf])
            bond_ind = self.p.bond(*edge)
            split_kwargs = {
                "method": self._split_method(),
                "absorb": "right",
                "max_bond": self.max_bond,
                "cutoff": self.cutoffs,
                "cutoff_mode": self.cutoff_mode,
                "get": "tensors",
                "bond_ind": bond_ind,
            }
            if self.split_method == "src":
                split_kwargs["seed"] = self.split_seed + self._split_counter
                self._split_counter += 1
            left, remaining = qtn.tensor_split(
                remaining,
                local_inds,
                right_inds=right_inds,
                **split_kwargs,
            )
            left.modify(tags=_tensor_of(self.p, leaf).tags)
            factors[leaf] = left
            destination = parent[leaf]
            if destination != center and all(
                child in factors for child in block if parent.get(child) == destination
            ):
                leaves.append(destination)
                leaves.sort()
        remaining.modify(tags=_tensor_of(self.p, center).tags)
        factors[center] = remaining
        return factors

    def _install_block(self, factors, block, center, *, validate=True):
        """Install fitted block tensors and restore canonical metadata."""

        # Center movement has already prepared every untouched branch in the
        # required gauge. Clearing/rebuilding ``left_inds`` for the complete
        # tree here would turn each local update into an O(N) operation and
        # would also discard the very proofs used to skip later QR moves.
        invalidate_norm = getattr(self.p, "_invalidate_norm_cache", None)
        if callable(invalidate_norm):
            invalidate_norm()
        for node in block:
            fitted = factors[node]
            live = _tensor_of(self.p, node)
            left_inds = None if node == center else fitted.left_inds
            live.modify(
                data=fitted.data,
                inds=fitted.inds,
                tags=live.tags,
                left_inds=left_inds,
            )
        # The effective block projects the target's raw tensors onto an
        # isometric exterior. Its represented scale therefore belongs to the
        # target, independently of the initial guess's extracted exponent.
        self.p.exponent = _exponent_value(self.tn)
        self.p._canonical_region = frozenset({center})
        if validate:
            self.p.validate(check_canonical=True)

    def _global_overlap(self):
        """Contract the current fitted state against the target."""

        mantissa, exponent = self._global_overlap_stripped()
        return _scale_stripped(mantissa, exponent)

    def _global_overlap_stripped(self):
        """Return the target overlap as a mantissa and base-ten exponent."""

        network = qtn.TensorNetwork([
            *[
                tensor.copy()
                for node in self.nodes
                for tensor in self._target_tensors[node]
            ],
            *[_tensor_of(self.p, node).H for node in self.nodes],
        ])
        value, exponent = network.contract(
            all,
            optimize=self.contraction_opt,
            strip_exponent=True,
        )
        return (
            _scalar_value(value),
            float(exponent)
            + _exponent_value(self.tn)
            + _exponent_value(self.p),
        )

    @staticmethod
    def _center_norm_stripped(network, center=None):
        """Read one canonical centre norm as ``(mantissa, exponent)``.

        This is the tree equivalent of FIT's terminal MPS tensor readout.  A
        canonical exterior cancels, so only the centre tensor is contracted;
        the network exponent is kept separately to avoid materialising a
        large represented norm.  Native fermionic trees provide a graded
        one-tensor contraction which must be used instead of ``Tensor.H``.
        """

        if center is None:
            center = getattr(network, "orthogonality_center", None)
        if center is None:
            region = getattr(network, "canonical_region", None)
            if region is not None and len(region) == 1:
                center = next(iter(region))
        if center is None:
            raise RuntimeError(
                "TreeFIT requires a tracked single canonical centre for "
                "local norm diagnostics."
            )

        fermionic_center_norm = getattr(
            network, "_fermionic_center_norm_squared", None
        )
        if bool(getattr(network, "fermionic", False)) and callable(
            fermionic_center_norm
        ):
            squared = _scalar_value(fermionic_center_norm(center))
            value = float(np.sqrt(max(0.0, float(np.real(squared)))))
        else:
            tensor = _tensor_of(network, center)
            squared = qtn.tensor_contract(tensor.H, tensor, output_inds=[])
            squared = _scalar_value(squared)
            value = float(np.sqrt(max(0.0, float(np.real(squared)))))
        return value, _exponent_value(network), center

    @staticmethod
    def _log_norm_pair(norm_pair):
        """Return ``log(norm)`` from a stripped norm pair."""

        mantissa, exponent = norm_pair
        mantissa = abs(float(mantissa))
        if mantissa == 0.0:
            return -np.inf
        return float(np.log(mantissa) + float(exponent) * np.log(10.0))

    @staticmethod
    def _relative_log_change(log_current, log_previous):
        """Return a scale-safe relative change between two positive norms."""

        if log_current == log_previous:
            return 0.0
        if not np.isfinite(log_current) or not np.isfinite(log_previous):
            return 1.0
        # This is |a-b| / max(a, b), evaluated without constructing a or b.
        return float(-np.expm1(-abs(float(log_current - log_previous))))

    def _target_norm_stripped_for_center(self, center):
        """Return the target norm without contracting its full overlap path.

        Fused native tree targets can be moved to ``center`` and read from one
        tensor just like the fitted state. A plain or layered target requires
        a supplied norm or the cached explicit canonical-QR diagnostic.
        """

        if self._target_norm_stripped is not None:
            return self._target_norm_stripped
        if self._known_target_norm is not None:
            return self._known_target_norm

        target = self.tn
        if (
            self.target_layout == "fused"
            and callable(getattr(target, "shift_orthogonality_center", None))
            and callable(getattr(target, "node_tensor", None))
        ):
            work = target.copy()
            try:
                work.shift_orthogonality_center(center, _skip_validate=True)
            except TypeError:
                work.shift_orthogonality_center(center)
            local_mantissa, _, _ = self._center_norm_stripped(work, center)
            self._target_norm_stripped = (
                local_mantissa,
                _exponent_value(work),
            )
            return self._target_norm_stripped

        # An opaque layered target has no known canonical norm. Do not turn
        # routine diagnostics into a doubled target contraction.
        raise ValueError("layered target norm is unknown; supply target_norm")

    def _canonical_target_norm(self, center):
        """Explicit diagnostic fallback: lossless leaf QR, then one hub norm.

        Never form <target|target>. Only requested exact-overlap diagnostics
        may do this extra pass; normal FIT retains its lazy target throughout.
        """
        parent = {center: None}
        order = [center]
        for node in order:
            for neighbor in _neighbors_of(self.p, node):
                if neighbor not in parent:
                    parent[neighbor] = node
                    order.append(neighbor)
        groups = {node: list(ts) for node, ts in self._target_tensors.items()}
        for node in reversed(order[1:]):
            destination = parent[node]
            tensor = qtn.tensor_contract(*groups[node])
            right_inds = self._target_bonds[(node, destination)]
            left_inds = tuple(ix for ix in tensor.inds if ix not in right_inds)
            splitter = getattr(self.p, "_native_qr_split", None)
            if splitter is None:
                if ar.infer_backend(tensor.data) == "symmray":
                    raise NotImplementedError("native target QR requires the tree QR policy")
                _, message = tensor.split(left_inds, method="qr", get="tensors")
            else:
                _, message = splitter(tensor, left_inds=left_inds, get="tensors")
            groups[destination].append(message)
        hub = qtn.tensor_contract(*groups[center])
        if ar.infer_backend(hub.data) == "symmray":
            network = qtn.TensorNetwork([hub])
            squared = (network.H | network).contract(all)
        else:
            squared = qtn.tensor_contract(hub.H, hub, output_inds=[])
        self._target_norm_stripped = (
            float(np.sqrt(max(0.0, float(np.real(_scalar_value(squared)))))),
            _exponent_value(self.tn),
        )

    @staticmethod
    def _check_state_finite(state, region):
        """Optional backend-native finite checks on the updated tensors."""
        flags = []
        owners = []
        for node in region:
            data = _tensor_of(state, node).data
            arrays = data.blocks.values() if ar.infer_backend(data) == "symmray" else (data,)
            for array in arrays:
                flags.append(ar.do("all", ar.do("isfinite", array)))
                owners.append(node)
        if flags:
            finite = np.asarray(ar.to_numpy(ar.do("stack", flags)), dtype=bool)
            bad = np.flatnonzero(~finite)
            if bad.size:
                raise FloatingPointError(f"non-finite TreeFIT tensor at node {owners[bad[0]]}")

    def _check_finite(self, region):
        self._check_state_finite(self.p, region)

    def _local_norm_fidelity(self):
        """Return MPS-style retained-centre-norm fidelity for the latest sweep."""

        if self.local_norm_stripped_trace:
            local_pair = self.local_norm_stripped_trace[-1]
            center = self.final_center_site
        else:
            try:
                local_mantissa, local_exponent, center = (
                    self._center_norm_stripped(self.p)
                )
            except (RuntimeError, ValueError, KeyError):
                return None
            local_pair = (local_mantissa, local_exponent)
        if center is None:
            return None
        try:
            target_pair = self._target_norm_stripped_for_center(center)
        except (RuntimeError, TypeError, ValueError, KeyError):
            return None
        log_local = self._log_norm_pair(local_pair)
        log_target = self._log_norm_pair(target_pair)
        if log_local == -np.inf and log_target == -np.inf:
            return 1.0
        if not np.isfinite(log_local) or not np.isfinite(log_target):
            return 0.0
        log_fidelity = 2.0 * (log_local - log_target)
        if log_fidelity >= 0.0:
            return 1.0
        if log_fidelity < np.log(np.finfo(float).tiny):
            return 0.0
        return float(np.exp(log_fidelity))

    def _record_local_norm(self):
        """Record one terminal canonical-centre norm for the latest sweep."""

        mantissa, exponent, center = self._center_norm_stripped(
            self.p, self.final_center_site
        )
        pair = (mantissa, exponent)
        represented = _scale_stripped(mantissa, exponent)
        self.final_center_site = center
        self.local_norm_stripped_trace.append(pair)
        self.local_norm_trace.append(represented)
        self.sweep_norm_trace.append(represented)
        self.last_norm = represented
        return pair

    @staticmethod
    def _log_stripped_norm(network):
        """Return a canonical tree log norm without a doubled network."""
        return TreeFIT._log_norm_pair(TreeFIT._center_norm_stripped(network)[:2])

    def _network_norm(self, network):
        """Read a represented tree norm without changing its gauge."""

        mantissa, exponent, _ = self._center_norm_stripped(network)
        return float(abs(_scale_stripped(mantissa, exponent)))

    def _normalized_overlap_fidelity(self, overlap_pair=None):
        """Return normalized target overlap using stripped exponents."""

        overlap, overlap_exponent = (
            self._global_overlap_stripped() if overlap_pair is None else overlap_pair
        )
        log_overlap = -np.inf if abs(overlap) == 0.0 else float(
            np.log(abs(overlap)) + overlap_exponent * np.log(10.0)
        )
        log_target = self._log_norm_pair(
            self._target_norm_stripped_for_center(self.final_center_site)
        )
        log_fitted = self._log_norm_pair(self._center_norm_stripped(self.p)[:2])
        log_fidelity = 2.0 * (log_overlap - log_target - log_fitted)
        if not np.isfinite(log_fidelity):
            return 0.0 if log_fidelity < 0.0 else 1.0
        return float(min(1.0, max(0.0, np.exp(log_fidelity))))

    def fit_block(self, block, *, center=None, validate=True):
        """Perform one cached local variational update on a connected block."""

        block = frozenset(block)
        if not block:
            raise ValueError("fit block must be non-empty")
        if not block.issubset(self._node_set):
            raise ValueError("fit block contains an unknown tree node")
        if len(block) > 3:
            raise ValueError("TreeFIT supports one-, two-, and three-node blocks")
        if not _is_connected(self.p, block):
            raise ValueError("fit block must be a connected subtree")
        center = self._block_center(block, preferred=center)
        self._canonicalize_for_block(block, center)
        effective = self._effective_block(block)
        factors = self._factor_block(effective, block, center)
        self._install_block(factors, block, center, validate=validate)
        self._invalidate_for_block(block)
        return {
            "block": tuple(sorted(block)),
            "center": center,
            "block_size": len(block),
            "cache": self.environment_cache_info(),
        }

    def _connected_edges(self, region):
        """Return deterministic local two-node blocks in a region."""

        region = frozenset(region)
        center = self._block_center(region)
        return sorted(
            ((node, neighbor) for node in region for neighbor in _neighbors_of(self.p, node)
             if neighbor in region and node < neighbor),
            key=lambda edge: (
                min(len(_path_of(self.p, center, edge[0])), len(_path_of(self.p, center, edge[1]))),
                edge,
            ),
        )

    def _connected_triples(self, region):
        """Return every connected three-node block in a region once."""

        region = frozenset(region)
        triples = set()
        for center in sorted(region):
            neighbors = sorted(
                neighbor for neighbor in _neighbors_of(self.p, center)
                if neighbor in region
            )
            for left, right in combinations(neighbors, 2):
                triples.add(tuple(sorted((left, center, right))))
        return sorted(triples)

    def _sweep_blocks(self, region, block_size, direction):
        """Return one inward or outward sequence of local update blocks."""

        region = frozenset(region)
        if self.traversal == "depth-first":
            return self._depth_first_sweep_blocks(region, block_size, direction)
        if block_size == 1:
            center = self._block_center(region)
            order = sorted(
                region,
                key=lambda node: (
                    len(_path_of(self.p, node, center)),
                    node,
                ),
                reverse=direction == "in",
            )
            return [(node,) for node in order]
        if block_size == 2:
            blocks = self._connected_edges(region)
        else:
            blocks = self._connected_triples(region)
            if not blocks:
                blocks = self._connected_edges(region)
        if not blocks:
            # A one-site gate still has a valid one-site FIT update when the
            # requested DMRG block size is two or three.
            blocks = [(node,) for node in sorted(region)]
        center = self._block_center(region)
        return sorted(
            blocks,
            key=lambda block: (
                min(len(_path_of(self.p, node, center)) for node in block),
                block,
            ),
            reverse=direction == "in",
        )

    def _depth_first_sweep_blocks(self, region, block_size, direction):
        """Group updates by branch using one iterative walk of the region.

        A block is anchored at its node nearest the medial hub. Reversing the
        same order gives the inward pass and preserves the exact block set.
        """
        hub = self._block_center(region)
        order = {}
        depth = {}
        stack = [(hub, None, 0)]
        while stack:
            node, parent, level = stack.pop()
            order[node] = len(order)
            depth[node] = level
            stack.extend(
                (neighbor, node, level + 1)
                for neighbor in sorted(self._neighbors[node], reverse=True)
                if neighbor != parent and neighbor in region
            )
        if block_size == 1:
            blocks = [(node,) for node in order]
        else:
            blocks = self._connected_triples(region) if block_size == 3 else []
            if not blocks:
                blocks = [
                    (node, neighbor)
                    for node in region for neighbor in self._neighbors[node]
                    if neighbor in region and node < neighbor
                ]
            if not blocks:
                blocks = [(node,) for node in order]
            blocks.sort(key=lambda block: (
                order[min(block, key=depth.__getitem__)], block,
            ))
        return blocks[::-1] if direction == "in" else blocks

    def _active_edge_rank_targets(self, region, *, state=None):
        """Return physical rank ceilings for tree edges inside ``region``.

        The adaptive DMRG phase follows FIT's chain rule: it is governed by
        the physical Hilbert-space capacity available on either side of an
        edge, including the live state bonds where the active region meets an
        untouched exterior. It must not use the raw bond dimension of a
        factorized operator-state target, which can be larger than the actual
        state rank (for example, a CNOT acting on ``|00>``).
        """

        if self.max_bond is None:
            return None
        targets = []
        region = frozenset(region)
        state = self.p if state is None else state
        for node in sorted(region):
            for neighbor in _neighbors_of(state, node):
                if neighbor not in region or node >= neighbor:
                    continue
                sides = (
                    region.intersection(_component_of(state, node, neighbor)),
                    region.intersection(_component_of(state, neighbor, node)),
                )
                side_caps = []
                for side in sides:
                    capacity = 1
                    for member in side:
                        physical = _physical_ind(state, member)
                        if physical is not None:
                            capacity *= int(state.ind_size(physical))
                        for outside in _neighbors_of(state, member):
                            if outside not in region:
                                capacity *= int(
                                    state.ind_size(state.bond(member, outside))
                                )
                    side_caps.append(capacity)
                targets.append(((node, neighbor), min(
                    int(self.max_bond), *side_caps
                )))
        return tuple(targets)

    def _active_bonds_at_rank_targets(self, region, *, state=None):
        """Return whether every active tree edge is at its target ceiling."""

        state = self.p if state is None else state
        targets = self._active_edge_rank_targets(region, state=state)
        if targets is None:
            return False
        return all(
            int(state.ind_size(state.bond(*edge))) >= int(target)
            for edge, target in targets
        )

    @staticmethod
    def _normalize_sweep_sequence(sequence):
        """Return a tree-oriented name while accepting legacy chain aliases."""
        key = str(sequence).strip().lower().replace("-", "").replace("_", "")
        if key in {"inwardoutward", "inout", "rl"}:
            return "inward-outward"
        if key in {"outwardinward", "outin", "lr"}:
            return "outward-inward"
        raise ValueError(
            "sweep_sequence must be 'inward-outward' or 'outward-inward' "
            "(legacy 'RL', 'LR', 'INOUT', and 'OUTIN' are also accepted)"
        )

    def run_gate(
        self,
        region,
        n_iter=6,
        verbose=False,
        *,
        block_size=2,
        sweep_sequence="inward-outward",
        min_iter=None,
        rtol=None,
        patience=1,
        adaptive_block_sweeps=None,
        adaptive_until_rank=False,
        two_site_transition_sweeps=0,
        final_one_site_sweeps=0,
        single_node_fast_path=True,
    ):
        """Run cached tree FIT sweeps over a connected active region.

        ``block_size`` is the number of connected structural tree nodes in a
        local update. ``sweep_sequence`` selects ``"inward-outward"`` or
        ``"outward-inward"``; legacy ``"RL"``/``"LR"`` remain aliases.
        One iteration includes both directional passes. These directions are
        measured relative to the active region's medial node, not necessarily
        the structural root of the whole tree.
        The target remains fixed and the fitted state is updated in place.
        ``adaptive_block_sweeps`` enables the MPS-compatible larger-block
        warm-up followed by one-site refinement. ``adaptive_until_rank``
        extends that warm-up until the active physical rank ceilings are
        reached, and ``final_one_site_sweeps`` adds optional one-site polish.
        ``verbose=True`` records one retained-centre-norm fidelity per
        completed sweep, matching the chain FIT diagnostic behavior. When
        ``adaptive_block_sweeps`` is supplied, the first requested number of
        sweeps use ``block_size`` and the remaining sweeps use one-site
        refinement. ``adaptive_until_rank=True`` keeps the larger block until
        the active tree edges reach their target/max-bond ceilings, subject to
        the minimum warm-up. ``final_one_site_sweeps`` adds fixed-rank polish
        sweeps after the requested iterations.
        ``two_site_transition_sweeps`` inserts two-node iterations between
        three-node warm-up and one-node refinement within the same budget.
        Its default zero preserves standalone fixed-block schedules.

        ``single_node_fast_path=True`` solves a one-node region with a single
        exact local projection, regardless of tolerance or iteration budget.
        This is exact for the fixed exterior, not necessarily for an arbitrary
        target outside that region. Disable it to repeat the fixed sweeps.
        Complete-tree ``run``/``run_eff`` keep this shortcut off by default.

        The per-sweep ``local_norm_trace`` is read from the final canonical
        centre tensor, matching MPS FIT. It is not a full-tree contraction;
        the corresponding stripped ``(mantissa, exponent)`` values are kept
        in ``local_norm_stripped_trace``. ``fidelity_trace`` records the
        retained-centre-norm fidelity. A genuine full target overlap is an
        optional :meth:`fit_diagnostics` diagnostic only.
        """

        if isinstance(region, Integral):
            region = (region,)
        region = frozenset(region)
        if not region or not region.issubset(self._node_set):
            raise ValueError("region must contain known tree nodes")
        if not _is_connected(self.p, region):
            raise ValueError("region must be a connected subtree")
        if not isinstance(n_iter, Integral) or int(n_iter) < 1:
            raise ValueError("n_iter must be a positive integer")
        if int(block_size) not in {1, 2, 3}:
            raise ValueError("block_size must be 1, 2, or 3")
        # A short tree window follows FIT's active-span behavior: a requested
        # three-node update on a two-node region is an ordinary two-node
        # update, and a one-node region is necessarily one-site.
        block_size = min(int(block_size), len(region))
        sequence = self._normalize_sweep_sequence(sweep_sequence)
        directions = {
            "inward-outward": ("in", "out"),
            "outward-inward": ("out", "in"),
        }[sequence]
        self.sweep_sequence = sequence
        if min_iter is None:
            min_iter = 1
        if int(min_iter) < 1:
            raise ValueError("min_iter must be positive")
        if not isinstance(patience, Integral) or int(patience) < 1:
            raise ValueError("patience must be positive")
        if rtol is not None:
            rtol = float(rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError("rtol must be a finite non-negative number")

        adaptive_schedule = adaptive_block_sweeps is not None
        if adaptive_schedule:
            if (
                not isinstance(adaptive_block_sweeps, Integral)
                or int(adaptive_block_sweeps) < 1
            ):
                raise ValueError(
                    "adaptive_block_sweeps must be a positive integer or None"
                )
            adaptive_block_sweeps = min(int(adaptive_block_sweeps), int(n_iter))
        else:
            adaptive_block_sweeps = int(n_iter)
        if not isinstance(final_one_site_sweeps, Integral) or int(
            final_one_site_sweeps
        ) < 0:
            raise ValueError("final_one_site_sweeps must be a non-negative integer")
        final_one_site_sweeps = int(final_one_site_sweeps)
        adaptive_until_rank = bool(adaptive_until_rank)
        if (
            not isinstance(two_site_transition_sweeps, Integral)
            or int(two_site_transition_sweeps) < 0
        ):
            raise ValueError("two_site_transition_sweeps must be a non-negative integer")
        if self.finite_check and not self._finite_check_warning_handled:
            warnings.warn(
                "TreeFIT finite_check is enabled: optional tensor scans can "
                "synchronize devices; leave it disabled for normal optimization.",
                RuntimeWarning,
                stacklevel=2,
            )

        previous = None
        stable = 0
        self.iterations_run = 0
        self.fidelity_trace = []
        self.local_norm_trace = []
        self.local_norm_stripped_trace = []
        self.sweep_norm_trace = []
        self.last_relative_change = None
        self.last_norm = None
        self.last_overlap = None
        self.final_center_site = None
        self._target_norm_stripped = None
        self.converged = False
        self.convergence_reason = None
        self.adaptive_sweeps_run = 0
        self.one_site_sweeps_run = 0
        self.block_size_trace = []
        if single_node_fast_path and len(region) == 1:
            # The exterior is fixed: one canonical local projection solves
            # this region exactly, independent of the initial centre tensor.
            self.fit_block(region, validate=False)
            self.iterations_run = 1
            self.one_site_sweeps_run = 1
            self.block_size_trace = [1]
            self.final_direction = directions[-1]
            self.final_center_site = getattr(self.p, "orthogonality_center", None)
            if self.finite_check:
                self._check_finite(region)
            self._record_local_norm()
            if verbose:
                self.fidelity_trace.append(self._local_norm_fidelity())
            self.converged = True
            self.convergence_reason = "single_node_exact"
            return self
        active_edges = tuple(
            (node, neighbor)
            for node in sorted(region)
            for neighbor in _neighbors_of(self.p, node)
            if neighbor in region and node < neighbor
        )
        adaptive_phase_done = not (
            adaptive_until_rank
            and block_size in {2, 3}
            and bool(active_edges)
        )
        if adaptive_until_rank and not adaptive_phase_done:
            adaptive_phase_done = self._active_bonds_at_rank_targets(region)
        rank_targets = (
            self._active_edge_rank_targets(region)
            if adaptive_until_rank and not adaptive_phase_done
            else None
        )

        transition_start = 1 if adaptive_phase_done else None
        block_orders = {}

        def sweep_blocks(size, direction):
            # The region and topology are fixed for this run; only tensors
            # change. Keep the exact traversal order without recomputing its
            # tree-medial node and pairwise paths on every iteration.
            key = (size, direction)
            if key not in block_orders:
                block_orders[key] = tuple(self._sweep_blocks(region, size, direction))
            return block_orders[key]

        def block_size_for_sweep(sweep_number):
            if block_size not in {2, 3}:
                return 1
            if adaptive_until_rank:
                use_block = not adaptive_phase_done
            else:
                use_block = sweep_number <= adaptive_block_sweeps
            if use_block:
                return block_size
            start = transition_start if adaptive_until_rank else adaptive_block_sweeps + 1
            if block_size == 3 and sweep_number < start + two_site_transition_sweeps:
                return 2
            return 1

        for iteration in range(1, int(n_iter) + 1):
            active_block_size = block_size_for_sweep(iteration)
            previous_block_size = (
                None if not self.block_size_trace else self.block_size_trace[-1]
            )
            if previous_block_size is not None and active_block_size != previous_block_size:
                # A block-to-one-site transition starts a new convergence
                # phase, just as in FIT.run_gate.
                previous = None
                stable = 0
                self.last_relative_change = None
            self.block_size_trace.append(active_block_size)
            if active_block_size == 1:
                self.one_site_sweeps_run += 1
            else:
                self.adaptive_sweeps_run += 1
            for direction in directions:
                for block in sweep_blocks(active_block_size, direction):
                    self.fit_block(block, validate=False)
            self.iterations_run = iteration
            self.final_direction = directions[-1]
            self.final_center_site = getattr(self.p, "orthogonality_center", None)
            if self.finite_check:
                self._check_finite(region)
            local_pair = self._record_local_norm()
            local_norm_log = self._log_norm_pair(local_pair)
            if verbose:
                self.fidelity_trace.append(self._local_norm_fidelity())
            if rtol is not None:
                warmup_incomplete = False
                warmup_finished_with_refinement = False
                adaptive_rank_incomplete = False
                if previous is not None:
                    relative_change = self._relative_log_change(
                        local_norm_log, previous
                    )
                    self.last_relative_change = float(relative_change)
                    warmup_incomplete = (
                        adaptive_schedule
                        and active_block_size > 1
                        and iteration < adaptive_block_sweeps
                    )
                    warmup_finished_with_refinement = (
                        adaptive_schedule
                        and active_block_size > 1
                        and iteration == adaptive_block_sweeps
                        and iteration < int(n_iter)
                    )
                    adaptive_rank_incomplete = (
                        adaptive_until_rank
                        and not adaptive_phase_done
                        and active_block_size > 1
                    )
                    if (
                        iteration >= int(min_iter)
                        and relative_change <= float(rtol)
                        and not (
                            warmup_incomplete
                            or warmup_finished_with_refinement
                            or adaptive_rank_incomplete
                            or (
                                active_block_size > 1
                                and iteration < int(n_iter)
                                and (
                                    block_size_for_sweep(iteration + 1) != active_block_size
                                    or (block_size == 3 and active_block_size == 2)
                                )
                            )
                        )
                    ):
                        stable += 1
                    else:
                        stable = 0
                    if stable >= int(patience):
                        self.converged = True
                        self.convergence_reason = "rtol"
                        break
                if warmup_finished_with_refinement or adaptive_rank_incomplete:
                    previous = None
                    stable = 0
                    self.last_relative_change = None
                else:
                    previous = local_norm_log
            else:
                previous = local_norm_log

            if (
                adaptive_until_rank
                and not adaptive_phase_done
                and active_block_size > 1
                and iteration >= adaptive_block_sweeps
                and rank_targets is not None
                and all(
                    int(self.p.ind_size(self.p.bond(*edge))) >= int(target)
                    for edge, target in rank_targets
                )
            ):
                adaptive_phase_done = True
                transition_start = iteration + 1
                previous = None
                stable = 0
                self.last_relative_change = None
        else:
            self.convergence_reason = "max_iter"

        if (
            not self.converged
            and final_one_site_sweeps > 0
            and block_size in {2, 3}
            and len(region) >= 3
        ):
            for _ in range(final_one_site_sweeps):
                self.block_size_trace.append(1)
                self.one_site_sweeps_run += 1
                for direction in directions:
                    for block in sweep_blocks(1, direction):
                        self.fit_block(block, validate=False)
                self.iterations_run += 1
                self.final_direction = directions[-1]
                self.final_center_site = getattr(
                    self.p, "orthogonality_center", None
                )
                if self.finite_check:
                    self._check_finite(region)
                self._record_local_norm()
                if verbose:
                    self.fidelity_trace.append(self._local_norm_fidelity())
        if self.converged is False and self.convergence_reason is None:
            self.convergence_reason = "max_iter"
        return self

    def run_eff(
        self,
        n_iter=6,
        verbose=False,
        *,
        block_size=1,
        sweep_sequence="inward-outward",
        min_iter=None,
        rtol=None,
        patience=1,
        adaptive_block_sweeps=None,
        adaptive_until_rank=False,
        two_site_transition_sweeps=0,
        final_one_site_sweeps=0,
        single_node_fast_path=False,
    ):
        """Fit the complete tree using the cached local environment engine."""

        return self.run_gate(
            self.nodes,
            n_iter=n_iter,
            block_size=block_size,
            sweep_sequence=sweep_sequence,
            min_iter=min_iter,
            rtol=rtol,
            patience=patience,
            verbose=verbose,
            adaptive_block_sweeps=adaptive_block_sweeps,
            adaptive_until_rank=adaptive_until_rank,
            two_site_transition_sweeps=two_site_transition_sweeps,
            final_one_site_sweeps=final_one_site_sweeps,
            single_node_fast_path=single_node_fast_path,
        )

    def run(
        self,
        n_iter=6,
        verbose=False,
        *,
        block_size=1,
        sweep_sequence="inward-outward",
        min_iter=None,
        rtol=None,
        patience=1,
        adaptive_block_sweeps=None,
        adaptive_until_rank=False,
        two_site_transition_sweeps=0,
        final_one_site_sweeps=0,
        single_node_fast_path=False,
    ):
        """Run a complete-tree FIT sweep with the chain-compatible API.

        Unlike the chain implementation there is no useful tree analogue of
        a left-to-right full-contraction reference path: the directed-message
        engine is the natural full-tree update. ``run`` therefore intentionally
        delegates to :meth:`run_eff`, while keeping FIT's positional
        ``n_iter``/``verbose`` call shape.
        """

        return self.run_eff(
            n_iter=n_iter,
            block_size=block_size,
            sweep_sequence=sweep_sequence,
            min_iter=min_iter,
            rtol=rtol,
            patience=patience,
            verbose=verbose,
            adaptive_block_sweeps=adaptive_block_sweeps,
            adaptive_until_rank=adaptive_until_rank,
            two_site_transition_sweeps=two_site_transition_sweeps,
            final_one_site_sweeps=final_one_site_sweeps,
            single_node_fast_path=single_node_fast_path,
        )

    def fit_diagnostics(self, *, overlap=False):
        """Return a copy-safe summary of the latest tree FIT run.

        ``local_fidelity`` is the MPS-compatible retained-centre-norm
        fidelity when both the retained canonical norm and target norm are
        known. It is None for opaque layered targets without ``target_norm``.
        ``overlap=True`` additionally requests the expensive genuine overlap
        with the full target; that value is reported as ``target_fidelity``
        and never replaces the local norm diagnostic.
        """

        local_pair = (
            self.local_norm_stripped_trace[-1]
            if self.local_norm_stripped_trace
            else None
        )
        if local_pair is None:
            try:
                local_mantissa, local_exponent, center = (
                    self._center_norm_stripped(self.p)
                )
                local_pair = (local_mantissa, local_exponent)
                if self.final_center_site is None:
                    self.final_center_site = center
            except (RuntimeError, ValueError, KeyError):
                local_pair = None
        local_norm = (
            None if local_pair is None
            else _scale_stripped(*local_pair)
        )
        local_fidelity = self._local_norm_fidelity()
        result = {
            "iterations": int(self.iterations_run),
            "converged": bool(self.converged),
            "convergence_reason": self.convergence_reason,
            "relative_change": self.last_relative_change,
            "final_norm": self.last_norm if self.last_norm is not None else local_norm,
            "final_norm_mantissa": (
                None if local_pair is None else float(local_pair[0])
            ),
            "final_norm_exponent": (
                None if local_pair is None else float(local_pair[1])
            ),
            "local_norm": self.last_norm if self.last_norm is not None else local_norm,
            "local_norm_trace": tuple(self.local_norm_trace),
            "local_norm_stripped_trace": tuple(self.local_norm_stripped_trace),
            "sweep_norm_trace": tuple(self.sweep_norm_trace),
            "local_fidelity": local_fidelity,
            "local_infidelity": (
                None if local_fidelity is None
                else float(max(0.0, 1.0 - local_fidelity))
            ),
            "adaptive_sweeps": int(self.adaptive_sweeps_run),
            "one_site_refinement_sweeps": int(self.one_site_sweeps_run),
            "block_size_trace": tuple(self.block_size_trace),
            "sweep_sequence": self.sweep_sequence,
            "traversal": self.traversal,
            "environment_strategy": self.environment_strategy,
            "target_layout": self.target_layout,
            "cache": self.environment_cache_info(),
        }
        if overlap:
            try:
                try:
                    self._target_norm_stripped_for_center(self.final_center_site)
                except ValueError:
                    self._canonical_target_norm(self.final_center_site)
                local_fidelity = self._local_norm_fidelity()
                result.update({
                    "local_fidelity": local_fidelity,
                    "local_infidelity": (
                        None if local_fidelity is None
                        else float(max(0.0, 1.0 - local_fidelity))
                    ),
                })
                overlap_mantissa, overlap_exponent = (
                    self._global_overlap_stripped()
                )
                overlap_value = _scale_stripped(
                    overlap_mantissa, overlap_exponent
                )
                target_fidelity = self._normalized_overlap_fidelity(
                    (overlap_mantissa, overlap_exponent)
                )
                result.update({
                    "overlap": overlap_value,
                    "target_fidelity": float(target_fidelity),
                    "target_infidelity": float(
                        max(0.0, 1.0 - target_fidelity)
                    ),
                    # Keep the MpsOptimizer naming available for callers
                    # which request the optional exact FIT overlap.
                    "fit_overlap_fidelity": float(target_fidelity),
                    "fit_overlap_infidelity": float(
                        max(0.0, 1.0 - target_fidelity)
                    ),
                    "fit_overlap_error": None,
                    "overlap_mantissa": overlap_mantissa,
                    "overlap_exponent": overlap_exponent,
                })
            except Exception as exc:  # diagnostics must not invalidate a fit
                result.update({
                    "overlap": None,
                    "target_fidelity": None,
                    "target_infidelity": None,
                    "fit_overlap_fidelity": None,
                    "fit_overlap_infidelity": None,
                    "fit_overlap_error": str(exc),
                    "overlap_error": str(exc),
                })
        return result
