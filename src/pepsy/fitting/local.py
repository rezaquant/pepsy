"""DMRG local fitting utilities for MPS/MPO tensor networks.

This module provides local and environment-based sweep routines used by
boundary contraction code. The focus is to keep tensor-index handling explicit
and fail early when input structure is inconsistent.
"""

import functools
import logging
import math
import time
import warnings
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..tensors.core import tn_fidelity

logger = logging.getLogger(__name__)

__all__ = [
    "FIT",
    "internal_inds",
]


class _SweepEnvironmentCache:
    """Boundary tensors and compatibility metadata from one completed sweep.

    ``boundaries`` is retained by reference: constructing this object copies
    neither its dictionary nor any backend tensor. Sweep kernels still receive
    that plain dictionary and use direct ``dict.get`` calls in their hot loops.

    A cache is directional. It can supply fixed environments only to an
    opposite-direction sweep. Equal block sizes have matching minimal
    boundary coverage. For a smaller reversed block, the producer extends the
    mapping by only the missing terminal boundaries: one for 3-to-2 or
    2-to-1, and two for 3-to-1.
    """

    __slots__ = (
        "boundaries", "block_size", "direction", "one_site_ready", "two_site_ready"
    )

    def __init__(
        self,
        boundaries,
        *,
        direction,
        block_size,
        one_site_ready=None,
        two_site_ready=False,
    ):
        self.boundaries = boundaries
        self.direction = direction
        self.block_size = int(block_size)
        self.two_site_ready = bool(two_site_ready)
        self.one_site_ready = (
            self.block_size == 1
            if one_site_ready is None
            else bool(one_site_ready)
        )

    def fixed_for(self, *, direction, block_size):
        """Return the zero-copy fixed-boundary mapping, or ``None``."""
        if direction == self.direction:
            return None
        block_size = int(block_size)
        if self.block_size == block_size or (
            block_size == 1 and self.one_site_ready
        ) or (
            block_size == 2 and self.block_size == 3
            and (self.two_site_ready or self.one_site_ready)
        ):
            return self.boundaries
        return None


# ---------------------------------------------------------------------------
# Low-level backend and tensor-network helpers
# ---------------------------------------------------------------------------

def internal_inds(psi):
    """Return all internal (non-open) indices of ``psi``."""
    open_inds = psi.outer_inds()
    inner = []
    for tensor in psi:
        for ind in tensor.inds:
            if ind not in open_inds:
                inner.append(ind)
    return inner


def _iter_backend_arrays(value):
    """Yield dense backend leaves from tensors, networks, or containers."""
    if value is None:
        return

    if isinstance(value, qtn.Tensor):
        yield from _iter_backend_arrays(value.data)
        return

    if isinstance(value, qtn.TensorNetwork):
        for tensor in value.tensors:
            yield from _iter_backend_arrays(tensor.data)
        return

    blocks = getattr(value, "blocks", None)
    if blocks is not None:
        values = blocks.values() if hasattr(blocks, "values") else blocks
        for block in values:
            yield from _iter_backend_arrays(block)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_backend_arrays(item)
        return

    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_backend_arrays(item)
        return

    yield value


class _BackendSynchronizer:
    """A cached accelerator barrier selected once for a timing session."""

    __slots__ = ("backend", "device")

    def __init__(self, backend, device=None):
        self.backend = backend
        self.device = device

    @classmethod
    def from_value(cls, value):
        """Return a synchronizer for the first supported accelerator leaf."""
        for data in _iter_backend_arrays(value):
            module = type(data).__module__.split(".", 1)[0]
            if module == "torch":
                device = getattr(data, "device", None)
                if getattr(device, "type", None) == "cuda":
                    return cls("torch", device=device)
            elif module == "cupy":
                return cls("cupy")
            elif module in {"jax", "jaxlib"}:
                return cls("jax")
        return None

    def synchronize(self, value, *, fallback=None):
        """Wait for work represented by ``value`` without host conversion."""
        if self.backend == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            torch.cuda.synchronize(self.device)
            return

        if self.backend == "cupy":
            import cupy  # pylint: disable=import-outside-toplevel

            cupy.cuda.get_current_stream().synchronize()
            return

        synchronized = False
        for data in _iter_backend_arrays(value):
            module = type(data).__module__.split(".", 1)[0]
            if module not in {"jax", "jaxlib"}:
                continue
            block_until_ready = getattr(data, "block_until_ready", None)
            if block_until_ready is not None:
                block_until_ready()
                synchronized = True

        if not synchronized and fallback is not None and fallback is not value:
            self.synchronize(fallback)


def _synchronize_tensor_network(tn):
    """Synchronize supported accelerator work represented by ``tn`` once."""
    synchronizer = _BackendSynchronizer.from_value(tn)
    if synchronizer is not None:
        synchronizer.synchronize(tn)


def _native_fermionic_bra_fit(method):
    """Run a FIT method with a native fermionic conjugated working MPS.

    Symmray's graded variational derivative naturally updates the conjugated
    fitting network. Keeping that representation for the complete sweep
    sequence lets left- and right-moving environments share one consistent
    dual-leg convention. The physical ket is restored even when a sweep
    raises, and no array is converted or flattened.
    """

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        psi = self.p
        if (
            psi is None
            or int(psi.L) <= 1
            or not psi.isfermionic()
            or self._fermionic_bra_working
        ):
            return method(self, *args, **kwargs)

        psi.conj_()
        self._fermionic_bra_working = True
        self._fermionic_physical_outer_inds = frozenset(psi.outer_inds())
        try:
            return method(self, *args, **kwargs)
        finally:
            psi.conj_()
            self._fermionic_bra_working = False
            self._fermionic_physical_outer_inds = frozenset()
            self._fermionic_left_exterior_environment = None
            self._fermionic_right_exterior_environment = None

    return wrapped


def _native_fermionic_bra_block_fit(method):
    """Apply the native bra gauge only to opt-in block ``run_eff`` fits."""

    wrapped = _native_fermionic_bra_fit(method)

    @functools.wraps(method)
    def block_wrapped(self, *args, **kwargs):
        if int(kwargs.get("block_size", 1)) not in {2, 3}:
            return method(self, *args, **kwargs)
        return wrapped(self, *args, **kwargs)

    return block_wrapped


class FIT:  # pylint: disable=too-many-instance-attributes
    """Variationally fit an open-boundary MPS or MPO to a target network.

    ``FIT`` is the shared local-compression kernel used by MPS, MPO, PEPS
    boundary, and sampling workflows.  The public sweep methods deliberately
    represent three different workloads:

    ``run``
        Small, full-contraction reference solver.  It is useful for simple
        compatibility paths and debugging, but it does not reuse environments.
    ``run_eff``
        Cached full-chain solver for an MPS/MPO.  It defaults to the
        historical one-site boundary/sampling path and optionally supports
        native two- and three-site block updates with fixed-sweep semantics.
    ``run_gate``
        Cached active-window solver for circuit compression.  It only updates
        ``range_int`` and optionally performs one-, two-, or three-site updates;
        the block updates use native SVDs with adaptive convergence and timing
        diagnostics.

    The implementation is organized in four stages: input ownership and
    tagging, effective-environment construction, one-, two-, or three-site
    updates, and optional diagnostics. Keeping those responsibilities explicit
    is important because the same class must support dense arrays, Torch/CuPy
    arrays, and native Symmray tensors without converting or densifying them.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Target tensor network to fit.
    p : qtn.MatrixProductState | qtn.MatrixProductOperator
        Initial open-boundary state or operator to optimize.
    cutoffs : float, default=1e-12
        Numerical cutoff used by local decompositions/truncations.
    backend : str | None, default=None
        Compatibility metadata retained on the solver. Array execution is
        inferred from tensor data; choose the local contraction route with
        ``environment_strategy``.
    site_tag_id : str, default="I{}"
        Site-tag format used by ``p`` and local environment builders.
    contraction_opt : str | object, default="auto-hq"
        Contraction optimizer used for effective-environment contractions.
    range_int : sequence[int] | None, default=None
        Optional active interval ``(start, stop)`` used by :meth:`run_gate`.
    retag : bool, default=False
        If ``True``, regenerate tags on ``tn`` from ``p`` site connectivity.
    info : dict[str, Any] | None, default=None
        Optional scratch dictionary used by callers to store metadata.
    warning : bool, default=False
        Enable warning logs for fallback and retagging edge-cases.
    inplace : bool, default=False
        If ``True``, optimize ``p`` in place; otherwise operate on ``p.copy()``.
    environment_strategy : {"auto", "mps-direct", "symmray-native", "generic"}, default="auto"
        Local environment contraction route. ``"mps-direct"`` avoids
        temporary TensorNetwork construction when both the target and fitted
        state are ordinary dense MPS/MPO networks. ``"generic"`` is the
        native-safe general/Symmray route. Fermionic Symmray inputs use
        Quimb's graph-planned native tensor contraction within the resolved
        ``symmray-native`` strategy so contraction order, dummy modes, and
        graded phases remain authoritative without building a temporary
        TensorNetwork.
    copy_target : bool, default=True
        Copy ``tn`` before its internal indices are randomized. Optimizer
        integrations that construct a disposable target can pass ``False``
        and transfer ownership, avoiding one complete target-network copy.
    Attributes
    ----------
    tn : qtn.TensorNetwork
        Owned target network. Its internal indices are randomized during
        construction so it can safely share physical outer indices with ``p``.
    p : qtn.MatrixProductState | qtn.MatrixProductOperator
        Current fitted network. This is a copy unless ``inplace=True``.
    range_int : list[int]
        Inclusive active interval represented as ``[start, stop]`` for
        :meth:`run_gate`; an empty list means no gate window was requested.
    environment_strategy : {"mps-direct", "symmray-native", "generic"}
        Resolved effective-environment implementation selected during
        construction. Non-fermionic Symmray inputs use the native blockwise
        chain route; fermionic inputs retain the resolved native strategy but
        use graph-planned Symmray contraction for dummy-mode safety.
    timing_records : list[dict]
        Copy-safe per-sweep timing records collected by ``run_gate(timing=True)``.
    info : dict
        Caller-owned diagnostics channel. Two- and three-site split metadata
        is appended here when ``collect_split_diagnostics=True``.
    """

    # ------------------------------------------------------------------
    # Construction and configuration
    # ------------------------------------------------------------------

    def __init__(
        self,
        tn: qtn.TensorNetwork,
        p: Optional[qtn.TensorNetwork] = None,
        cutoffs: float = 1e-12,
        backend: Optional[str] = None,
        site_tag_id: str = "I{}",
        contraction_opt: str = "auto-hq",
        range_int: Optional[Sequence[int]] = None,
        retag: bool = False,
        info: Optional[Dict[str, Any]] = None,
        warning: bool = False,
        inplace: bool = False,
        *,
        environment_strategy: str = "auto",
        copy_target: bool = True,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        # Validate the fitted network before taking ownership of either input.
        if p is None:
            raise ValueError("Initial MPS/MPO `p` must be provided for FIT.")
        if not isinstance(p, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
            raise TypeError(
                "Initial `p` must be MatrixProductState or MatrixProductOperator."
            )
        if p.cyclic:
            raise ValueError("FIT currently supports open-boundary MPS/MPO guesses only.")
        if not isinstance(site_tag_id, str) or "{}" not in site_tag_id:
            raise ValueError("site_tag_id must be a format string containing '{}'.")

        self.L = int(p.L)

        # FIT owns the working state and, by default, its target.  The
        # ``copy_target=False`` escape hatch is reserved for integrations that
        # have already created a disposable target network.
        self.p = p if inplace else p.copy()
        self.tn = tn.copy() if copy_target else tn
        if site_tag_id:
            if isinstance(self.p, qtn.MatrixProductOperator):
                # Preserve both MPO physical-index families. Re-viewing every
                # guess as an MPS leaves ``site_ind_id=None`` and breaks normal
                # MPO operations such as ``to_dense()`` after fitting.
                self.p.view_as_(
                    qtn.MatrixProductOperator,
                    L=self.L,
                    site_tag_id=site_tag_id,
                    upper_ind_id=self.p.upper_ind_id,
                    lower_ind_id=self.p.lower_ind_id,
                    cyclic=False,
                )
            else:
                self.p.view_as_(
                    qtn.MatrixProductState,
                    L=self.L,
                    site_tag_id=site_tag_id,
                    site_ind_id=self.p.site_ind_id,
                    cyclic=False,
                )

        self.site_tag_id = site_tag_id

        # Store numerical and contraction controls before building caches.
        self.contraction_opt = contraction_opt
        self.cutoffs = cutoffs
        self.backend = backend

        environment_strategy = str(environment_strategy).strip().lower()
        if environment_strategy not in {
            "auto",
            "mps-direct",
            "symmray-native",
            "generic",
        }:
            raise ValueError(
                "environment_strategy must be 'auto', 'mps-direct', "
                "'symmray-native', or 'generic'."
            )

        # Initialize user-facing diagnostics. These values describe the most
        # recent run and are reset by ``run_gate`` before a new sweep sequence.
        self.warning = warning
        self.timing_records: List[Dict[str, Any]] = []
        self.fidelity_trace: List[float] = []
        self.local_norm_trace: List[Any] = []
        self.sweep_norm_trace: List[float] = []
        self.iterations_run = 0
        self.converged = False
        self.convergence_reason = "not_run"
        self.last_relative_change: Optional[float] = None
        self.final_center_site: Optional[int] = None
        self.final_direction: Optional[str] = None
        self.final_norm = None
        self.adaptive_sweeps_run = 0
        self.one_site_sweeps_run = 0
        self._fermionic_bra_working = False
        self._fermionic_physical_outer_inds = frozenset()
        self._fermionic_left_exterior_environment = None
        self._fermionic_right_exterior_environment = None
        self._timing_sync_device = False
        self._timing_synchronizer = None
        # An owning optimizer may already have warned for the whole replay.
        # Standalone FIT calls emit their own diagnostic-cost warning.
        self._finite_check_warning_handled = False
        # Preserve an explicitly supplied empty dictionary: callers may use
        # ``info`` as a live diagnostics channel during and after a run.
        self.info: Dict[str, Any] = info if info is not None else {}
        self.range_int: List[int] = list(range_int) if range_int is not None else []
        if self.range_int:
            if len(self.range_int) != 2:
                raise ValueError("range_int must be a sequence of two integers: (start, stop).")
            if not all(isinstance(site, Integral) for site in self.range_int):
                raise TypeError("range_int entries must be integers.")
            self.range_int = [int(site) for site in self.range_int]
            start, stop = self.range_int
            if start >= stop:
                raise ValueError("range_int must satisfy start < stop.")

        # Randomize only internal target indices. Physical outer indices must
        # remain aligned with the fitted network for the overlap objective.
        self.tn.reindex_({idx: qtn.rand_uuid() for idx in self.tn.inner_inds()})

        if set(self.tn.outer_inds()) != set(self.p.outer_inds()):
            raise ValueError("tn and p have different outer indices.")

        # Optional retagging makes layered or otherwise untagged targets
        # compatible with the fitted network's site tags.
        if retag:
            self._re_tag()

        # Resolve the local contraction route only after reindexing and
        # retagging. In particular, ``retag=True`` can turn an initially
        # untagged dense target into a valid one-tensor-per-site cache.
        self._target_site_tensors = self._build_target_site_cache()
        self._target_tensor_order = (
            {tensor_id: order for order, tensor_id in enumerate(self.tn.tensor_map)}
            if self._target_site_tensors is None else {}
        )
        # A gate fit visits only its active site tags. Build those selections
        # lazily rather than duplicating the complete target's tag map.
        # Global run/run_eff populate the same cache as they visit each site.
        self._target_tag_tensor_ids = {}
        # Layered targets can carry several tensors per site, so their chain
        # bond is not available through ``TensorNetwork.bond``. The target
        # graph is immutable during FIT: resolve each boundary locally once
        # and retain only its index name, never tensor data.
        self._target_bond_cache = {}
        # One metadata pass supplies all routing decisions, including mixed
        # dense/native inputs. No tensor values or device scalars are read.
        array_kinds = {
            type(tensor.data).__module__.split(".", 1)[0] == "symmray"
            for network in (self.tn, self.p)
            for tensor in network.tensor_map.values()
        }
        has_symmray = True in array_kinds
        symmray_native_available = array_kinds == {True}
        direct_available = (
            self._target_site_tensors is not None
            and not self.tn.isfermionic()
            and not self.p.isfermionic()
            and not has_symmray
        )
        if environment_strategy == "mps-direct" and not direct_available:
            raise ValueError(
                "environment_strategy='mps-direct' requires an ordinary dense "
                "MPS/MPO target and fitted state with exactly one target tensor "
                "per site."
            )
        if environment_strategy == "symmray-native" and not symmray_native_available:
            raise ValueError(
                "environment_strategy='symmray-native' requires Symmray-backed "
                "target and fitted tensors."
            )
        self.environment_strategy = (
            "mps-direct"
            if environment_strategy == "mps-direct"
            or (environment_strategy == "auto" and direct_available)
            else (
                "symmray-native"
                if environment_strategy == "symmray-native"
                or (environment_strategy == "auto" and symmray_native_available)
                else "generic"
            )
        )

        # Dense arrays, the audited conjugated native-fermion gauge, and the
        # native bosonic Symmray route can reuse partial environments across
        # an immediate direction reversal. Keep mixed-backend or explicitly
        # generic bosonic Symmray fits conservative: only the native route has
        # the zero-copy ownership/fusion-metadata contract exercised here.
        native_fermionic_pair = (
            self.tn.isfermionic() and self.p.isfermionic()
        )
        native_bosonic_symmray_pair = (
            symmray_native_available
            and not native_fermionic_pair
            and self.environment_strategy == "symmray-native"
        )
        self._allow_sweep_environment_reuse = (
            native_fermionic_pair
            or native_bosonic_symmray_pair
            or not has_symmray
        )
        self._sweep_environment_reuse_count = 0
    # ------------------------------------------------------------------
    # Target cache, public inspection, and visualization
    # ------------------------------------------------------------------

    def _build_target_site_cache(self):
        """Return one target tensor per site, or ``None`` for a layered TN.

        This is a structural optimization, not a dense conversion. The cache
        stores views of the already-copied target and never changes array
        backends, devices, symmetry sectors, or fermionic metadata.
        """
        # Lazy gate layers add tensors. Reject that case before walking the
        # untouched prefix of a long chain looking for the first layered site.
        if len(self.tn.tensor_map) != self.L:
            return None
        tensors = []
        for site in range(self.L):
            tag = self.site_tag_id.format(site)
            tensor_ids = self.tn.tag_map.get(tag, ())
            if len(tensor_ids) != 1:
                return None
            (tensor_id,) = tuple(tensor_ids)
            tensors.append(self.tn.tensor_map[tensor_id])
        # A single tensor carrying several site tags is not an MPS site cache:
        # storing it once per tag would contract the same tensor repeatedly.
        # Likewise, extra untagged/layer tensors make the direct site-by-site
        # route incomplete. Require an exact one-to-one site/tensor mapping.
        if len({id(tensor) for tensor in tensors}) != self.L:
            return None
        return tuple(tensors)

    def _resolve_cutoff(self, cutoff):
        """Resolve an optional dtype-aware truncation cutoff."""
        if cutoff != "auto":
            return float(cutoff)

        dtype_names = []
        for network in (self.p, self.tn):
            for tensor in network.tensors:
                dtype = getattr(tensor.data, "dtype", None)
                if dtype is not None:
                    dtype_names.append(str(dtype).lower())
        if any("16" in dtype for dtype in dtype_names):
            resolved = 1.0e-3
        elif any("32" in dtype or "complex64" in dtype for dtype in dtype_names):
            resolved = 1.0e-6
        else:
            resolved = 1.0e-12
        self.info["cutoff_requested"] = "auto"
        self.info["cutoff_resolved"] = resolved
        return resolved

    def visual(
        self,
        figsize=(14, 14),
        layout="neato",
        show_tags=False,
        tags_: Optional[Sequence[str]] = None,
        show_inds=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Visualize the combined target network and current fitted state."""
        tag_list = tags_ if tags_ is not None else []
        tags = [self.site_tag_id.format(i) for i in range(self.L)] + tag_list
        return (self.tn & self.p).draw(
            tags,
            legend=False,
            show_inds=show_inds,
            show_tags=show_tags,
            figsize=figsize,
            node_outline_darkness=0.1,
            node_outline_size=None,
            highlight_inds_color="darkred",
            edge_scale=2.0,
            layout=layout,
            refine_layout="auto",
            highlight_inds=self.p.outer_inds(),
        )

    # ------------------------------------------------------------------
    # Target tagging and structural preparation
    # ------------------------------------------------------------------
    def _deep_tag(self, seed_sites):
        """Partition target tensors into ordered, local site regions.

        ``seed_sites`` maps target tensor ids to the MPS sites whose physical
        indices touch those tensors. A multi-source shortest-path traversal
        then assigns every reachable unseeded tensor to the closest site. The
        site number breaks equal-distance ties, making the result independent
        of the order in which Quimb stores or exposes tensor neighbors. An
        unseeded tensor on a region boundary receives both neighboring site
        tags, so selecting either local region retains the bridge tensor.

        This is deliberately a graph-only operation. It does not reorder
        tensors, rename indices, or move any tensor data.
        """
        tn = self.tn
        tensor_ids = tuple(tn.tensor_map)
        tensor_order = {tid: order for order, tid in enumerate(tensor_ids)}
        site_tags = [self.site_tag_id.format(i) for i in range(self.p.L)]

        # Build the target graph once. The existing index map is the source of
        # truth for connectivity; this temporary adjacency view avoids
        # repeatedly rediscovering the same neighbors while partitioning.
        adjacency = {}
        for tid in tensor_ids:
            neighbors = set()
            for index in tn.tensor_map[tid].inds:
                neighbors.update(tn.ind_map.get(index, ()))
            neighbors.discard(tid)
            adjacency[tid] = tuple(
                sorted(neighbors, key=tensor_order.__getitem__)
            )

        # ``best`` stores the winning ``(distance, site)`` for each tensor. A
        # multi-source BFS is linear in the target graph size; seeding the
        # queue by MPS site number makes equal-distance ownership deterministic.
        best = {}
        pending = deque()
        for tid, sites in sorted(
            seed_sites.items(),
            key=lambda item: (min(item[1]), tensor_order[item[0]]),
        ):
            if tid not in tensor_order or not sites:
                continue
            site = min(sites)
            best[tid] = (0, site)
            pending.append((tid, 0, site))

        while pending:
            tid, distance, site = pending.popleft()

            for neighbor in adjacency[tid]:
                if neighbor in best:
                    continue
                best[neighbor] = (distance + 1, site)
                pending.append((neighbor, distance + 1, site))

        # Start with all direct and trusted metadata associations. Seed
        # tensors retain every site they explicitly represent, including a
        # legitimate multi-site bridge tensor.
        region_sites = {
            tid: set(sites) for tid, sites in seed_sites.items()
        }

        for tid, (_distance, site) in best.items():
            if tid in seed_sites:
                continue

            # Normally an unseeded tensor belongs to one nearest region. If
            # it touches another region's frontier, include the neighboring
            # site tag too. This creates a one-tensor overlap at boundaries,
            # preserving local contractions without broadening whole regions.
            sites = {site}
            for neighbor in adjacency[tid]:
                neighbor_sites = seed_sites.get(neighbor)
                if neighbor_sites is None:
                    neighbor_sites = {best[neighbor][1]}
                sites.update(
                    neighbor_site
                    for neighbor_site in neighbor_sites
                    if neighbor_site != site
                )
            region_sites[tid] = sites

        for tid, sites in region_sites.items():
            tensor = tn.tensor_map[tid]
            for site in sorted(sites):
                tensor.add_tag(site_tags[site])

    def _re_tag(self):
        """Assign target tags from physical connectivity to the fitted MPS.

        Every target tensor attached to a site's physical index is seeded with
        that site's tag. Remaining tensors are assigned to the nearest seeded
        site in the target tensor graph. This keeps layered target ownership
        local without changing tensor order, index names, or tensor data.
        """
        p = self.p
        tn = self.tn
        site_tags = tuple(self.site_tag_id.format(i) for i in range(p.L))
        site_for_tag = {tag: site for site, tag in enumerate(site_tags)}

        # Existing canonical site tags are meaningful layout metadata for a
        # layered target: gate application can move a physical leg onto a
        # gate tensor while the original MPS tensor still carries its site
        # tag. Dropping those hints would make the MPS backbone look closer to
        # the wrong neighboring site. Keep only tags from the current MPS site
        # scheme and discard unrelated presentation tags below.
        seed_sites = {}
        for tid, tensor in tn.tensor_map.items():
            hinted_sites = {
                site_for_tag[tag]
                for tag in tensor.tags
                if tag in site_for_tag
            }
            if hinted_sites:
                seed_sites[tid] = hinted_sites

        tn.drop_tags()

        outer_inds = frozenset(p.outer_inds())

        # Seed from all shared physical indices, not just the first target
        # tensor returned by ``ind_map``. This is authoritative for physical
        # ownership; existing canonical tags provide additional backbone
        # hints for layers whose physical legs have moved to gate tensors.
        for site in range(p.L):
            site_tensor = p[site_tags[site]]
            for index in site_tensor.inds:
                if index not in outer_inds:
                    continue
                for tid in tn.ind_map.get(index, ()):
                    seed_sites.setdefault(tid, set()).add(site)

        self._deep_tag(seed_sites)

        untagged_tensors = [tensor for tensor in tn if not tensor.tags]
        if untagged_tensors and self.warning:
            logger.warning(
                "%d target tensors are disconnected from all MPS physical "
                "sites and remain untagged after nearest-site retagging.",
                len(untagged_tensors),
            )

    def run(self, n_iter=6, verbose=False):
        """Run the simple full-contraction reference sweeps.

        This is intentionally the least-specialized solver. It updates every
        site left-to-right and contracts the complete local objective at each
        update. Use :meth:`run_eff` for cached full-chain fitting or
        :meth:`run_gate` for an active circuit window.

        Parameters
        ----------
        n_iter : int
            Number of complete sweeps.
        verbose : bool
            If ``True``, append per-sweep fidelity values to ``self.fidelity_trace``.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")
        if not isinstance(n_iter, Integral) or int(n_iter) < 1:
            raise ValueError("n_iter must be a positive integer.")
        n_iter = int(n_iter)
        self._reset_run_traces()

        psi = self.p
        L = self.L
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        for _ in range(n_iter):
            for site in range(L):
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1

                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                psi_h = psi.H.select([site_tag_id.format(site)], "!any")
                tn_ = psi_h | self.tn

                # Contract and normalize
                f = tn_.contract(all, optimize=contraction_opt)
                f = f.transpose(*psi[site].inds)

                # norm_f is never applied (f.data used as-is); keep only for diagnostics if needed
                # norm_f = (f.H & f).contract(all) ** 0.5
                # self.local_norm_trace.append(complex(norm_f).real)

                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = tn_fidelity(
                    self.tn,
                    psi,
                    contraction_opt=contraction_opt,
                )
                self.fidelity_trace.append(ar.do("real", fidelity))

    # ------------------------------------------------------------------
    # Legacy full-chain solvers
    # ------------------------------------------------------------------

    def _build_env_right(self, psi, env_right):
        """Build inclusive right environments for all sites.

        Populates ``env_right[site_tag]`` for each site, where each entry is
        the contraction of the current site block and everything to its right.
        """
        L = self.L
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        # iterate from rightmost to leftmost
        for i in reversed(range(L)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block

            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=contraction_opt)
            else:
                # tie to previously computed right environment
                t |= env_right[site_tag_id.format(i + 1)]
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=contraction_opt)

    def _update_env_left(self, psi, site: int, env_left):
        """Update left environment incrementally for current site."""

        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block

        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=contraction_opt)
        else:
            t |= env_left[site_tag_id.format(site - 1)]
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=contraction_opt)

    def _build_env_left(self, psi, env_left):
        """Build inclusive left environments for all sites."""
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        for i in range(self.L):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block

            if i == 0:
                env_left[site_tag_id.format(i)] = t.contract(
                    all,
                    optimize=contraction_opt,
                )
            else:
                t |= env_left[site_tag_id.format(i - 1)]
                env_left[site_tag_id.format(i)] = t.contract(
                    all,
                    optimize=contraction_opt,
                )

    def _update_env_right(self, psi, site: int, env_right):
        """Update right environment incrementally for current site."""
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        contraction_opt = self.contraction_opt
        site_tag_id = self.site_tag_id

        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block

        if site == self.L - 1:
            env_right[site_tag_id.format(site)] = t.contract(
                all,
                optimize=contraction_opt,
            )
        else:
            t |= env_right[site_tag_id.format(site + 1)]
            env_right[site_tag_id.format(site)] = t.contract(
                all,
                optimize=contraction_opt,
            )

    @_native_fermionic_bra_block_fit
    def run_eff(
        self,
        n_iter=6,
        verbose=False,
        *,
        block_size=1,
        sweep_sequence="RL",
        max_bond=None,
        cutoff=None,
        cutoff_mode="rsum2",
        collect_split_diagnostics=True,
        adaptive_block_sweeps=None,
        min_iter=None,
        rtol=None,
        patience=1,
    ):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Run full-chain fitting sweeps with cached left/right environments.

        This method avoids rebuilding full contractions at each site by
        incrementally reusing left and right environments. It is the
        full-chain counterpart of :meth:`run_gate`: ``run_eff`` deliberately
        visits every site and does not use ``range_int`` to restrict the fit.

        ``block_size=1`` retains the fixed-rank one-site compatibility update. The
        opt-in ``block_size=2`` and ``block_size=3`` paths use the same cached
        full-chain environments and native Quimb/Symmray SVD splits as
        :meth:`run_gate`; only bonds reached by those splits can grow, up to
        ``max_bond``. These block updates keep fixed-sweep semantics unless
        ``rtol`` is enabled: every requested sweep in ``sweep_sequence`` is
        performed when tolerance stopping is disabled.

        The DMRG circuit path uses ``run_gate`` instead. A gate target differs
        from the current MPS only on its active interval, so fitting the whole
        chain here would do unnecessary work and would change the algorithm
        from local gate compression into a global variational refit.

        Parameters
        ----------
        n_iter : int
            Number of complete full-chain sweeps.
        verbose : bool
            If ``True``, append one full-chain fidelity value per sweep.
        block_size : {1, 2, 3}, default=1
            Number of neighboring tensors optimized by the cached update.
            The default one-site path is retained for boundary and sampling
            compatibility. Two- and three-site paths use native SVD splits.
        sweep_sequence : str, default="RL"
            Fixed sequence of sweep directions for all block sizes.
            ``"R"`` is left-to-right, ``"L"`` is right-to-left, and
            sequences such as ``"RL"`` alternate directions.
        max_bond : int | None, default=None
            Maximum bond dimension passed to native block SVD splits. This
            applies only to block sizes 2 and 3.
        cutoff : float | None, default=None
            Output SVD cutoff for block sizes 2 and 3. ``None`` uses
            ``self.cutoffs``.
        cutoff_mode : str, default="rsum2"
            Quimb SVD cutoff mode for block sizes 2 and 3.
        collect_split_diagnostics : bool, default=True
            Store native SVD metadata in ``self.info`` for block sizes 2 and 3.
        adaptive_block_sweeps : int | None, default=None
            If set for ``block_size=2`` or ``block_size=3``, use the selected
            block update for this many initial sweeps and then use one-site
            refinement for the remaining sweeps. ``None`` preserves the
            historical fixed-block behavior.
        min_iter : int | None, default=None
            Minimum number of completed sweeps before ``rtol`` can stop the
            run. Defaults to ``2`` when ``rtol`` is enabled and to ``n_iter``
            otherwise. Adaptive stopping always requires two completed sweeps
            so it has a comparable pair of retained norms. A block-to-one-site
            transition always completes its requested block warm-up before
            convergence can stop the run.
        rtol : float | None, default=None
            Relative change tolerance for the terminal retained center norm.
            ``None`` performs exactly ``n_iter`` sweeps and does not add a
            backend-to-host diagnostic transfer.
        patience : int, default=1
            Number of stable retained-norm samples required when ``rtol`` is
            enabled. A phase transition resets this window.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")
        if not isinstance(n_iter, Integral) or int(n_iter) < 1:
            raise ValueError("n_iter must be a positive integer.")
        n_iter = int(n_iter)
        self._reset_run_traces()

        if not isinstance(block_size, Integral) or int(block_size) not in {1, 2, 3}:
            raise ValueError("block_size must be 1, 2, or 3.")
        block_size = int(block_size)
        sweep_sequence = self._validate_sweep_sequence(sweep_sequence)
        if max_bond is not None:
            if not isinstance(max_bond, Integral) or int(max_bond) < 1:
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        if cutoff is None:
            cutoff = self.cutoffs
        cutoff = self._resolve_cutoff(cutoff)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be a finite non-negative number.")
        collect_split_diagnostics = bool(collect_split_diagnostics)
        if adaptive_block_sweeps is not None:
            if block_size not in {2, 3}:
                raise ValueError(
                    "adaptive_block_sweeps is only configurable for block_size=2 or 3."
                )
            if (
                not isinstance(adaptive_block_sweeps, Integral)
                or int(adaptive_block_sweeps) < 1
            ):
                raise ValueError(
                    "adaptive_block_sweeps must be a positive integer or None."
                )
            adaptive_block_sweeps = min(int(adaptive_block_sweeps), n_iter)
        adaptive_schedule = adaptive_block_sweeps is not None
        if min_iter is None:
            min_iter = 2 if rtol is not None else n_iter
        if not isinstance(min_iter, Integral) or int(min_iter) < 1:
            raise ValueError("min_iter must be a positive integer or None.")
        min_iter = min(int(min_iter), n_iter)
        if rtol is not None:
            if n_iter < 2:
                raise ValueError(
                    "run_eff with rtol requires n_iter >= 2 for a comparable "
                    "pair of sweep norms."
                )
            if min_iter < 2:
                raise ValueError("run_eff with rtol requires min_iter >= 2.")
            rtol = float(rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError("rtol must be a finite non-negative number or None.")
        if not isinstance(patience, Integral) or int(patience) < 1:
            raise ValueError("patience must be a positive integer.")
        patience = int(patience)

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        contraction_opt = self.contraction_opt

        if L == 1:
            if self.warning:
                logger.warning("run_eff called for L=1; falling back to run().")
            self.run(n_iter=n_iter, verbose=verbose)
            return

        if block_size == 3 and L < 3:
            raise ValueError("block_size=3 requires a full chain of at least three sites.")

        if block_size in {2, 3}:
            if self._fermionic_bra_working:
                self._prepare_fermionic_active_fit(
                    psi,
                    0,
                    L - 1,
                    sweep_sequence[0],
                )
            sweep_cache = None
            self._sweep_environment_reuse_count = 0
            previous_sweep_norm = None
            stable_sweeps = 0
            for sweep in range(n_iter):
                direction = sweep_sequence[sweep % len(sweep_sequence)]
                previous_direction = (
                    None if sweep_cache is None else sweep_cache.direction
                )
                previous_block_size = (
                    None if sweep_cache is None else sweep_cache.block_size
                )
                active_block_size = (
                    block_size
                    if not adaptive_schedule or sweep < adaptive_block_sweeps
                    else 1
                )
                if (
                    previous_block_size is not None
                    and active_block_size != previous_block_size
                ):
                    previous_sweep_norm = None
                    stable_sweeps = 0
                    self.last_relative_change = None
                reuse_canonical_form = (
                    self._fermionic_bra_working
                    and previous_direction is None
                ) or (
                    previous_direction is not None
                    and previous_direction != direction
                )
                fixed_environments = None
                if self._allow_sweep_environment_reuse and sweep_cache is not None:
                    fixed_environments = sweep_cache.fixed_for(
                        direction=direction,
                        block_size=active_block_size,
                    )
                if fixed_environments is not None:
                    self._sweep_environment_reuse_count += 1
                self.iterations_run = sweep + 1
                if active_block_size == 1:
                    self.one_site_sweeps_run += 1
                    boundaries = self._run_gate_one_site_sweep(
                        psi,
                        0,
                        L - 1,
                        direction=direction,
                        timing_record=None,
                        reuse_canonical_form=reuse_canonical_form,
                        fixed_environments=fixed_environments,
                    )
                elif active_block_size == 2:
                    self.adaptive_sweeps_run += 1
                    boundaries = self._run_gate_two_site_sweep(
                        psi,
                        0,
                        L - 1,
                        direction=direction,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        timing_record=None,
                        collect_split_diagnostics=collect_split_diagnostics,
                        reuse_canonical_form=reuse_canonical_form,
                        fixed_environments=fixed_environments,
                    )
                else:
                    self.adaptive_sweeps_run += 1
                    boundaries = self._run_gate_three_site_sweep(
                        psi,
                        0,
                        L - 1,
                        direction=direction,
                        max_bond=max_bond,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        timing_record=None,
                        collect_split_diagnostics=collect_split_diagnostics,
                        reuse_canonical_form=reuse_canonical_form,
                        fixed_environments=fixed_environments,
                    )
                self.final_direction = direction
                self.final_center_site = L - 1 if direction == "R" else 0
                self.final_norm = self.local_norm_trace[-1]
                sweep_cache = _SweepEnvironmentCache(
                    boundaries,
                    direction=direction,
                    block_size=active_block_size,
                )
                if verbose:
                    fidelity = tn_fidelity(
                        self.tn,
                        self._physical_working_state(psi),
                        contraction_opt=contraction_opt,
                    )
                    self.fidelity_trace.append(ar.do("real", fidelity))

                should_stop = False
                reset_tolerance = False
                if rtol is not None:
                    _, sweep_norm = self._sweep_diagnostics_to_host(
                        psi,
                        0,
                        L - 1,
                        self.final_norm,
                        check_finite=False,
                        read_norm=True,
                    )
                    self.sweep_norm_trace.append(sweep_norm)
                    if previous_sweep_norm is not None:
                        scale = max(
                            abs(sweep_norm),
                            abs(previous_sweep_norm),
                            float.fromhex("0x1.0p-1022"),
                        )
                        relative_change = abs(
                            sweep_norm - previous_sweep_norm
                        ) / scale
                        self.last_relative_change = relative_change
                        if relative_change <= rtol:
                            stable_sweeps += 1
                        else:
                            stable_sweeps = 0
                        required_stable_changes = max(1, patience - 1)
                        if (
                            sweep + 1 >= min_iter
                            and stable_sweeps >= required_stable_changes
                        ):
                            warmup_incomplete = (
                                adaptive_schedule
                                and sweep + 1 < adaptive_block_sweeps
                            )
                            warmup_finished_with_refinement = (
                                adaptive_schedule
                                and sweep + 1 == adaptive_block_sweeps
                                and sweep + 1 < n_iter
                            )
                            if not (
                                warmup_incomplete
                                or warmup_finished_with_refinement
                            ):
                                self.converged = True
                                self.convergence_reason = "relative_tolerance"
                                should_stop = True
                            elif warmup_finished_with_refinement:
                                reset_tolerance = True
                                stable_sweeps = 0
                                self.last_relative_change = None
                    previous_sweep_norm = None if reset_tolerance else sweep_norm

                next_sweep = sweep + 1
                next_block_size = (
                    block_size
                    if not adaptive_schedule or next_sweep < adaptive_block_sweeps
                    else 1
                )
                if (
                    self._allow_sweep_environment_reuse
                    and not should_stop
                    and adaptive_schedule
                    and active_block_size in {2, 3}
                    and next_sweep <= n_iter - 1
                    and next_block_size == 1
                    and sweep_sequence[next_sweep % len(sweep_sequence)] != direction
                ):
                    self._extend_block_cache_for_smaller_block(
                        psi,
                        boundaries,
                        0,
                        L - 1,
                        direction,
                        block_size=active_block_size,
                    )
                    sweep_cache = _SweepEnvironmentCache(
                        boundaries,
                        direction=direction,
                        block_size=active_block_size,
                        one_site_ready=True,
                    )
                if should_stop:
                    break
            return

        if rtol is not None or not (self.p.isfermionic() or self.tn.isfermionic()):
            # Dense and non-fermionic native fits use the same cached one-site
            # kernel as run_gate. Fixed-sweep run_eff keeps the compatibility
            # update and sweep order, while reusing the completed opposite-side
            # environment instead of rebuilding it at every direction change.
            # Fermionic fixed-sweep compatibility remains on the legacy route;
            # the native bra wrapper is intentionally limited to block fits.
            sweep_cache = None
            self._sweep_environment_reuse_count = 0
            previous_sweep_norm = None
            stable_sweeps = 0
            for sweep in range(n_iter):
                direction = sweep_sequence[sweep % len(sweep_sequence)]
                previous_direction = (
                    None if sweep_cache is None else sweep_cache.direction
                )
                fixed_environments = None
                if self._allow_sweep_environment_reuse and sweep_cache is not None:
                    fixed_environments = sweep_cache.fixed_for(
                        direction=direction,
                        block_size=1,
                    )
                if fixed_environments is not None:
                    self._sweep_environment_reuse_count += 1
                boundaries = self._run_gate_one_site_sweep(
                    psi,
                    0,
                    L - 1,
                    direction=direction,
                    timing_record=None,
                    reuse_canonical_form=(
                        previous_direction is not None
                        and previous_direction != direction
                    ),
                    fixed_environments=fixed_environments,
                )
                self.iterations_run = sweep + 1
                self.one_site_sweeps_run += 1
                self.final_direction = direction
                self.final_center_site = L - 1 if direction == "R" else 0
                self.final_norm = self.local_norm_trace[-1]
                sweep_cache = _SweepEnvironmentCache(
                    boundaries,
                    direction=direction,
                    block_size=1,
                )
                if verbose:
                    fidelity = tn_fidelity(
                        self.tn,
                        self._physical_working_state(psi),
                        contraction_opt=contraction_opt,
                    )
                    self.fidelity_trace.append(ar.do("real", fidelity))
                if rtol is not None:
                    _, sweep_norm = self._sweep_diagnostics_to_host(
                        psi,
                        0,
                        L - 1,
                        self.final_norm,
                        check_finite=False,
                        read_norm=True,
                    )
                    self.sweep_norm_trace.append(sweep_norm)
                    if previous_sweep_norm is not None:
                        scale = max(
                            abs(sweep_norm),
                            abs(previous_sweep_norm),
                            float.fromhex("0x1.0p-1022"),
                        )
                        relative_change = abs(
                            sweep_norm - previous_sweep_norm
                        ) / scale
                        self.last_relative_change = relative_change
                        if relative_change <= rtol:
                            stable_sweeps += 1
                        else:
                            stable_sweeps = 0
                        if (
                            sweep + 1 >= min_iter
                            and stable_sweeps >= max(1, patience - 1)
                        ):
                            self.converged = True
                            self.convergence_reason = "relative_tolerance"
                            break
                    previous_sweep_norm = sweep_norm
            return

        env_left = {site_tag_id.format(i): None for i in range(psi.L)}
        env_right = {site_tag_id.format(i): None for i in range(psi.L)}

        for sweep in range(n_iter):
            direction = sweep_sequence[sweep % len(sweep_sequence)]
            self.iterations_run = sweep + 1
            self.one_site_sweeps_run += 1
            self.final_direction = direction
            self.final_center_site = L - 1 if direction == "R" else 0

            if direction == "R":
                sites = range(L)
            else:
                sites = range(L - 1, -1, -1)

            for site in sites:
                # Determine orthogonalization reference
                if direction == "R":
                    ortho_arg = "calc" if site == 0 else site - 1
                else:
                    ortho_arg = "calc" if site == L - 1 else site + 1
                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                if direction == "R" and site == 0:
                    self._build_env_right(psi, env_right)
                elif direction == "R":
                    self._update_env_left(psi, site - 1, env_left)
                elif site == L - 1:
                    self._build_env_left(psi, env_left)
                else:
                    self._update_env_right(psi, site + 1, env_right)

                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                tn = None
                if site == 0:
                    if tn_site is not None:
                        tn = tn_site | env_right[site_tag_id.format(site + 1)]
                    else:
                        tn = env_right[site_tag_id.format(site + 1)]

                if 0 < site < L - 1:
                    if tn_site is not None:
                        tn = (
                            tn_site
                            | env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )
                    else:
                        tn = (
                            env_right[site_tag_id.format(site + 1)]
                            | env_left[site_tag_id.format(site - 1)]
                        )

                if site == L - 1:
                    if tn_site is not None:
                        tn = tn_site | env_left[site_tag_id.format(site - 1)]
                    else:
                        tn = env_left[site_tag_id.format(site - 1)]

                if tn is None:
                    raise ValueError("Failed to build effective tensor for current site.")

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=contraction_opt).transpose(
                        *psi[site_tag_id.format(site)].inds
                    )
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                else:
                    raise TypeError("Unexpected effective tensor type during run_eff.")

                # norm_f is never applied (f.data used as-is); keep only for diagnostics if needed
                # norm_f = (f.H & f).contract(all) ** 0.5
                # self.local_norm_trace.append(complex(norm_f).real)

                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = tn_fidelity(
                    self.tn,
                    psi,
                    contraction_opt=contraction_opt,
                )
                self.fidelity_trace.append(ar.do("real", fidelity))

    # ------------------------------------------------------------------
    # Effective-environment and target assembly helpers
    # ------------------------------------------------------------------

    def _target_components(self, sites, *, reindex=None):
        """Return target tensors for ``sites`` without changing the target.

        Dense MPS/MPO targets use the per-site cache. General layered targets
        use pre-indexed tag-to-tensor maps. A boundary reindex is always
        applied to a copy because cached tensors are shared across sweeps.
        """
        sites = tuple(int(site) for site in sites)
        if self._target_site_tensors is not None:
            components = [self._target_site_tensors[site] for site in sites]
        else:
            tensor_ids = set()
            for site in sites:
                tag = self.site_tag_id.format(site)
                if tag not in self._target_tag_tensor_ids:
                    self._target_tag_tensor_ids[tag] = tuple(
                        self.tn.tag_map.get(tag, ())
                    )
                tensor_ids.update(self._target_tag_tensor_ids[tag])
            components = [
                self.tn.tensor_map[tensor_id]
                for tensor_id in sorted(
                    tensor_ids,
                    key=self._target_tensor_order.__getitem__,
                )
            ]

        if reindex:
            components = [
                tensor.reindex(reindex, inplace=False)
                if any(index in reindex for index in tensor.inds)
                else tensor
                for tensor in components
            ]
        return components

    def _active_boundary_reindex(self, psi, sites, start, stop):
        """Map target boundary bonds onto the fixed outside MPS bonds.

        FIT deliberately does not contract sites outside the active gate
        interval. Identifying each target boundary bond with the corresponding
        fitted-state bond is therefore the exact identity environment supplied
        by those untouched sites.
        """
        if self._fermionic_bra_working:
            # Native fermions retain the actual outside overlap environments.
            # Replacing those graded contractions by identity reindexing loses
            # the bond-space Koszul gauge on a right-moving sweep.
            return {}

        first, last = min(sites), max(sites)
        mapping = {}
        if first == start and start > 0:
            mapping[self._target_bond(start - 1, start)] = psi.bond(
                start - 1, start
            )
        if last == stop and stop < self.L - 1:
            mapping[self._target_bond(stop, stop + 1)] = psi.bond(
                stop, stop + 1
            )
        return mapping

    def _target_bond(self, left_site, right_site):
        """Return the unique target index crossing two neighboring site tags.

        A gate target can contain multiple tensors per site, so the MPS/MPO
        ``bond(i, j)`` convenience method is not always defined. The cut itself
        must still have one chain index for an active-window identity boundary.
        """
        if self._target_site_tensors is not None:
            return self.tn.bond(left_site, right_site)

        key = (int(left_site), int(right_site))
        try:
            return self._target_bond_cache[key]
        except KeyError:
            pass

        left_tids = set(
            self.tn.tag_map[self.site_tag_id.format(left_site)]
        )
        right_tids = set(
            self.tn.tag_map[self.site_tag_id.format(right_site)]
        )
        # Search only indices attached to the left site's tensors. The old
        # whole-network ``ind_map.items()`` scan made a local gate fit scale
        # with the complete MPS length.
        candidates = []
        seen = set()
        for tensor_id in sorted(
            left_tids,
            key=self._target_tensor_order.__getitem__,
        ):
            for index in self.tn.tensor_map[tensor_id].inds:
                if index in seen:
                    continue
                seen.add(index)
                if right_tids.intersection(self.tn.ind_map[index]):
                    candidates.append(index)
        if len(candidates) != 1:
            raise ValueError(
                "A gate-window FIT target must have one chain index across "
                f"sites {left_site} and {right_site}; found {candidates}."
            )
        self._target_bond_cache[key] = candidates[0]
        return self._target_bond_cache[key]

    def _contract_components(self, components, *, output_inds=None):
        """Contract local components through a safe or specialized route.

        The direct route avoids constructing a temporary TensorNetwork for an
        ordinary dense MPS target. Fermionic Symmray tensors intentionally use
        Quimb's graph-planned tensor contraction so native block contraction,
        dual-leg handling, dummy modes, and graded phases remain authoritative.
        """
        components = tuple(components)
        if not components:
            raise ValueError("Cannot contract an empty FIT environment.")

        if self.environment_strategy == "symmray-native" and all(
            type(component.data).__module__.split(".", 1)[0] == "symmray"
            for component in components
        ):
            # Fermionic Symmray arrays carry dummy-mode and graded-phase
            # metadata that cannot be safely propagated by the small
            # left-to-right product below. Keep those contractions on
            # Quimb's graph-planned Symmray route, even though the selected
            # strategy remains ``symmray-native`` (the tensors are never
            # densified and no temporary TensorNetwork is constructed).
            if self.tn.isfermionic() or self.p.isfermionic():
                return self._contract_generic_components(
                    components,
                    output_inds=output_inds,
                )
            try:
                return self._contract_symmray_components(
                    components,
                    output_inds=output_inds,
                )
            except ValueError as exc:
                # Symmray's fermionic dummy-mode validation is stricter than
                # the ordinary blockwise product used by this chain-special
                # case.  A moving environment can temporarily carry repeated
                # dummy-mode metadata after a native split.  Retry that one
                # contraction through Quimb's graph-planned fermionic route,
                # which owns the dummy-mode and graded-phase bookkeeping. Do
                # not hide unrelated native contraction errors.
                if "dummy_modes" not in str(exc):
                    raise
                return self._contract_generic_components(
                    components,
                    output_inds=output_inds,
                )

        if self.environment_strategy == "mps-direct" and all(
            isinstance(component, qtn.Tensor) for component in components
        ):
            if len(components) == 1:
                result = components[0]
                if output_inds is not None:
                    result = result.transpose(*output_inds)
                return result
            return self._contract_dense_chain_components(
                components,
                output_inds=output_inds,
            )

        # Layered dense targets contain several ordinary tensors per site, so
        # they cannot use the one-tensor-per-site ``mps-direct`` cache above.
        # They can nevertheless use Quimb's direct tensor contraction without
        # first allocating a temporary TensorNetwork. Keep fermionic and
        # Symmray data on the graph-planned native route below, where their
        # metadata is part of the contraction semantics.
        if (
            all(isinstance(component, qtn.Tensor) for component in components)
            and not self.tn.isfermionic()
            and not self.p.isfermionic()
            and not any(
                type(component.data).__module__.split(".", 1)[0] == "symmray"
                for component in components
            )
        ):
            return qtn.tensor_contract(
                *components,
                optimize=self.contraction_opt,
                output_inds=output_inds,
            )

        return self._contract_generic_components(
            components,
            output_inds=output_inds,
        )

    @staticmethod
    def _contract_dense_chain_components(components, *, output_inds=None):
        """Contract ordinary MPS components without rebuilding a TN graph.

        The direct MPS path presents components in chain order: neighboring
        site tensors share one virtual index, while overlap environments attach
        at the two ends. A left-to-right backend ``tensordot`` therefore avoids
        Quimb's contraction-planner and temporary ``TensorNetwork`` overhead
        while retaining one final ``qtn.Tensor`` for FIT's split API.
        """
        result = components[0]
        result_data = result.data
        result_inds = list(result.inds)
        result_tags = result.tags

        for right in components[1:]:
            right_inds = list(right.inds)
            shared = tuple(index for index in result_inds if index in right_inds)
            if not shared:
                raise ValueError(
                    "Dense FIT chain components must share at least one "
                    "index at each contraction step."
                )
            left_axes = tuple(result_inds.index(index) for index in shared)
            right_axes = tuple(right_inds.index(index) for index in shared)
            result_data = ar.do(
                "tensordot",
                result_data,
                right.data,
                axes=(left_axes, right_axes),
            )
            result_inds = [
                index for index in result_inds if index not in shared
            ] + [index for index in right_inds if index not in shared]
            result_tags = result_tags | right.tags

        result_inds = tuple(result_inds)
        if output_inds is not None:
            output_inds = tuple(output_inds)
            if len(result_inds) != len(output_inds) or set(result_inds) != set(
                output_inds
            ):
                raise ValueError(
                    "'output_inds' must be a permutation of the current "
                    f"tensor indices, but {set(result_inds)} != {set(output_inds)}"
                )
            axes = tuple(result_inds.index(index) for index in output_inds)
            if axes != tuple(range(len(axes))):
                result_data = ar.do("transpose", result_data, axes=axes)
            result_inds = output_inds

        return qtn.Tensor(
            data=result_data,
            inds=result_inds,
            tags=result_tags,
        )

    def _contract_generic_components(self, components, *, output_inds=None):
        """Contract components through Quimb's graph-planned native route.

        This remains the fallback for fermionic, Symmray, and otherwise
        general tensor data. In particular, it is the safety route for a
        native Symmray environment whose blockwise metadata needs a contraction
        order that annihilates conjugate dummy modes before like-dual copies
        meet. ``qtn.tensor_contract`` plans directly from the component index
        graph and dispatches on their native arrays; it neither constructs an
        intermediate TensorNetwork nor converts block-sparse data to dense.
        """
        if len(components) == 1:
            result = components[0]
            if output_inds is not None:
                result = result.transpose(*output_inds)
            return result

        return qtn.tensor_contract(
            *components,
            optimize=self.contraction_opt,
            output_inds=output_inds,
        )

    @staticmethod
    def _contract_symmray_components(components, *, output_inds=None):
        """Contract Symmray components through native blockwise products.

        FIT environments are chain-shaped: a site overlap is appended to one
        already-contracted boundary environment at a time.  Quimb's generic
        network contraction is robust, but rebuilding a contraction tree for
        every such step adds avoidable Python overhead and can obscure
        Symmray's native block contraction.  This route mirrors SymDMRG2's
        pair contraction while retaining the exact Quimb index ordering and
        tensor tags expected by the active-window solver.
        """
        result = components[0]
        for right in components[1:]:
            shared = tuple(ind for ind in result.inds if ind in right.inds)
            left_axes = tuple(result.inds.index(ind) for ind in shared)
            right_axes = tuple(right.inds.index(ind) for ind in shared)
            data = result.data.tensordot(
                right.data,
                axes=(left_axes, right_axes),
                mode="blockwise",
                preserve_array=True,
            )
            inds = (
                tuple(ind for ind in result.inds if ind not in shared)
                + tuple(ind for ind in right.inds if ind not in shared)
            )
            result = qtn.Tensor(
                data=data,
                inds=inds,
                tags=result.tags | right.tags,
            )
        if output_inds is not None:
            result = result.transpose(*output_inds)
        return result

    def _overlap_environment_site(self, psi, site, start, stop, prior=None):
        """Contract one ``<psi|target>`` site into a cached environment."""
        mapping = self._active_boundary_reindex(
            psi,
            (site,),
            start,
            stop,
        )
        components = [
            psi[site] if self._fermionic_bra_working else psi[site].H
        ]
        components.extend(
            self._target_components((site,), reindex=mapping)
        )
        if prior is not None:
            components.append(prior)
        return self._contract_components(components)

    def _build_active_environments(
        self,
        psi,
        start,
        stop,
        direction,
        *,
        block_size,
    ):
        """Build only fixed environments reachable by this block sweep."""
        environments = {}
        if direction == "R":
            prior = self._fermionic_right_exterior_environment
            if prior is not None:
                environments[stop + 1] = prior
            # The first block spans ``start:start + block_size`` and therefore
            # needs a fixed right environment beginning at ``start +
            # block_size``. Environments closer to the left edge can never be
            # queried by this sweep.
            for site in range(stop, start + block_size - 1, -1):
                prior = self._overlap_environment_site(
                    psi,
                    site,
                    start,
                    stop,
                    prior=prior,
                )
                environments[site] = prior
        else:
            prior = self._fermionic_left_exterior_environment
            if prior is not None:
                environments[start - 1] = prior
            # Mirror the right-moving rule: the first block needs fixed data
            # only through ``stop - block_size``.
            for site in range(start, stop - block_size + 1):
                prior = self._overlap_environment_site(
                    psi,
                    site,
                    start,
                    stop,
                    prior=prior,
                )
                environments[site] = prior
        return environments

    def _extend_block_cache_for_smaller_block(
        self,
        psi,
        boundaries,
        start,
        stop,
        direction,
        *,
        block_size,
        next_block_size=1,
        timing_record=None,
    ):
        """Complete only the boundaries needed by a smaller reversed block.

        A minimal reversed block cache stops ``block_size - 1`` sites before
        the terminal center. Those tensors are already canonical after the
        final block split. Extend through ``block_size - next_block_size``
        of them: one for 3-to-2 or 2-to-1 and two for 3-to-1. This avoids
        rebuilding the complete fixed side without retaining unused terminal
        environments between equal-size block sweeps.
        """
        if block_size not in {2, 3} or not 1 <= next_block_size < block_size:
            return

        started = self._timing_mark() if timing_record is not None else None
        if direction == "R":
            sites = range(stop - block_size + 1, stop - next_block_size + 1)
            for site in sites:
                boundaries[site] = self._overlap_environment_site(
                    psi,
                    site,
                    start,
                    stop,
                    prior=boundaries.get(site - 1),
                )
        else:
            sites = range(start + block_size - 1, start + next_block_size - 1, -1)
            for site in sites:
                boundaries[site] = self._overlap_environment_site(
                    psi,
                    site,
                    start,
                    stop,
                    prior=boundaries.get(site + 1),
                )

        if timing_record is not None:
            elapsed = float(self._timing_mark(boundaries) - started)
            # The extension is terminal moving-environment work associated
            # with the final block update, not an additional update/site.
            final_update = timing_record["site_timings"][-1]
            for key in (
                "environment_seconds",
                "moving_environment_seconds",
                "elapsed_seconds",
            ):
                final_update[key] += elapsed

    def _prepare_fermionic_active_fit(self, psi, start, stop, direction):
        """Canonicalize and cache exact outside graded environments once.

        ``psi`` is the conjugated fitting MPS while this helper runs. A full
        canonicalization around the first sweep center makes the outside
        norm metric an identity in the ordinary sense, while explicitly
        contracting those outside tensors retains the fermionic bond gauge
        and dummy-mode ordering that an index-only identity would discard.
        """
        center = start if direction == "R" else stop
        psi.canonize_around_(self.site_tag_id.format(center))

        prior = None
        for site in range(start):
            prior = self._overlap_environment_site(
                psi,
                site,
                start,
                stop,
                prior=prior,
            )
        self._fermionic_left_exterior_environment = prior

        prior = None
        for site in range(self.L - 1, stop, -1):
            prior = self._overlap_environment_site(
                psi,
                site,
                start,
                stop,
                prior=prior,
            )
        self._fermionic_right_exterior_environment = prior

    def _prepare_fermionic_effective_tensor(
        self,
        tensor,
        left_tensor,
        right_tensor,
        left_environment,
        right_environment,
    ):
        """Convert a native effective ket derivative into the working bra.

        This is the local graded derivative convention used by Quimb's native
        two-site FIT: synchronize the dual environment legs, phase true
        physical outer legs, then conjugate before decomposition/writeback.
        Active-window virtual boundaries are represented by the cached real
        outside environments and therefore never masquerade as physical legs.
        """
        if not self._fermionic_bra_working:
            return tensor

        left_environment_ind = None
        if left_environment is not None:
            (left_environment_ind,) = left_tensor.bonds(left_environment)
        right_environment_ind = None
        if right_environment is not None:
            (right_environment_ind,) = right_tensor.bonds(right_environment)

        data = tensor.data
        left_axis = (
            tensor.inds.index(left_environment_ind)
            if left_environment_ind is not None
            else None
        )
        right_axis = (
            tensor.inds.index(right_environment_ind)
            if right_environment_ind is not None
            else None
        )
        if left_axis is not None and right_axis is not None:
            data.phase_flip(
                left_axis if data.duals[left_axis] else right_axis,
                inplace=True,
            )
        elif left_axis is not None and data.duals[left_axis]:
            data.phase_flip(left_axis, inplace=True)
        elif right_axis is not None and data.duals[right_axis]:
            data.phase_flip(right_axis, inplace=True)

        dual_physical_axes = tuple(
            axis
            for axis, index in enumerate(tensor.inds)
            if index in self._fermionic_physical_outer_inds
            and data.indices[axis].dual
        )
        if dual_physical_axes:
            data.phase_flip(*dual_physical_axes, inplace=True)

        tensor.conj_()
        return tensor

    def _require_nonempty_fermionic_effective_tensor(self, tensor, sites):
        """Raise a direct sector-support error for an empty native update."""
        if not self._fermionic_bra_working:
            return
        if getattr(tensor.data, "num_blocks", 1) != 0:
            return
        sites = tuple(int(site) for site in sites)
        raise ValueError(
            "Native fermionic FIT produced an empty effective tensor at "
            f"sites {sites}; the target and initial MPS have disconnected "
            "charge-sector support. Provide a sector-compatible current MPS "
            "or use a native block update that can open the needed charge "
            "sectors."
        )

    def _resolve_fermionic_writeback_phase(self, *tensors):
        """Resolve odd dummy-mode global signs after native writeback."""
        if not self._fermionic_bra_working:
            return
        for tensor in tensors:
            if sum(mode.parity for mode in tensor.data.dummy_modes) % 2:
                tensor.data.phase_global(inplace=True)

    def _physical_working_state(self, psi):
        """Return a physical-ket view for optional user diagnostics."""
        return psi.H if self._fermionic_bra_working else psi

    def _reset_run_traces(self):
        """Clear diagnostics that describe one invocation rather than lifetime."""
        self.fidelity_trace = []
        self.local_norm_trace = []
        self.sweep_norm_trace = []
        self.info.pop("two_site_splits", None)
        self.info.pop("three_site_splits", None)

    @staticmethod
    def _sweep_diagnostics_to_host(
        psi,
        start,
        stop,
        final_norm,
        *,
        check_finite,
        read_norm,
    ):
        """Transfer one compact backend-native diagnostic vector per sweep.

        ``finite_check=True`` reduces every active tensor (including native
        Symmray blocks) to backend boolean scalars before the transfer. If rtol
        also needs the retained norm, that scalar shares the same transfer.
        """
        if not check_finite:
            if not read_norm:
                return True, None
            # A scalar-only convergence read needs no stack/vector allocation.
            # The caller retains the same scalar finite check and stop policy.
            norm = np.asarray(ar.to_numpy(ar.do("real", final_norm))).item()
            return True, float(norm)
        scalars = []
        finite_count = 0
        if check_finite:
            for site in range(start, stop + 1):
                for array in _iter_backend_arrays(psi[site]):
                    scalars.append(ar.do("all", ar.do("isfinite", array)))
                    finite_count += 1
            scalars.append(ar.do("isfinite", final_norm))
            finite_count += 1
        if read_norm:
            scalars.append(ar.do("real", final_norm))

        if not scalars:
            return True, None
        try:
            host_values = np.asarray(
                ar.to_numpy(ar.do("stack", scalars))
            ).reshape(-1)
        except Exception:
            # Supported dense and Symmray backends take the single-transfer
            # route. Retain a conservative fallback for custom autoray leaves.
            host_values = np.asarray(
                [np.asarray(ar.to_numpy(value)).item() for value in scalars]
            ).reshape(-1)

        finite = bool(np.all(host_values[:finite_count])) if check_finite else True
        norm = float(host_values[-1]) if read_norm else None
        return finite, norm

    @staticmethod
    def _active_bond_rank_targets(psi, start, stop, max_bond):
        """Return physical-rank ceilings for bonds inside an active window.

        The ceiling includes the current virtual dimensions at the window
        boundaries. This gives the largest rank the local two-/three-site
        updates can reach without padding or changing the outside MPS. For a
        full dense chain this produces the usual ``2, 4, 8, ...`` profile,
        capped by ``max_bond``. The adaptive block phase does not use a
        rank-stability shortcut: if a target cannot reach this ceiling, it
        remains in the adaptive phase until ``n_iter`` is exhausted.
        """
        if max_bond is None or stop <= start:
            return None
        try:
            if hasattr(psi, "upper_ind") and hasattr(psi, "lower_ind"):
                # An operator site has two physical legs. Its vectorized
                # Hilbert--Schmidt space has dimension d_upper * d_lower.
                physical_dims = [
                    int(psi.ind_size(psi.upper_ind(site)))
                    * int(psi.ind_size(psi.lower_ind(site)))
                    for site in range(start, stop + 1)
                ]
            else:
                physical_dims = [int(psi.phys_dim(site)) for site in range(start, stop + 1)]
            left_rank = (
                int(psi.bond_size(start - 1, start)) if start > 0 else 1
            )
            right_rank = (
                int(psi.bond_size(stop, stop + 1))
                if stop + 1 < int(psi.L)
                else 1
            )
        except (AttributeError, TypeError, ValueError):
            return None

        left_caps = []
        rank = left_rank
        for dim in physical_dims[:-1]:
            rank = min(int(max_bond), rank * dim)
            left_caps.append(rank)

        right_caps = [right_rank] * (len(physical_dims) - 1)
        rank = right_rank
        for index in range(len(physical_dims) - 1, 0, -1):
            rank = min(int(max_bond), rank * physical_dims[index])
            right_caps[index - 1] = rank

        return tuple(
            min(int(max_bond), left, right)
            for left, right in zip(left_caps, right_caps)
        )

    @classmethod
    def _active_bonds_at_rank_targets(cls, psi, start, stop, max_bond):
        """Return whether every active bond is already at its rank ceiling."""
        targets = cls._active_bond_rank_targets(
            psi,
            start,
            stop,
            max_bond,
        )
        if targets is None:
            return False
        try:
            current = tuple(
                int(psi.bond_size(site, site + 1))
                for site in range(start, stop)
            )
        except (AttributeError, TypeError, ValueError):
            return False
        return all(rank >= target for rank, target in zip(current, targets))

    def _effective_tensor(
        self,
        psi,
        sites,
        start,
        stop,
        *,
        left_environment=None,
        right_environment=None,
        output_inds,
    ):
        """Form the active one-, two-, or three-site variational target."""
        mapping = self._active_boundary_reindex(psi, sites, start, stop)
        components = self._target_components(sites, reindex=mapping)
        if left_environment is not None:
            components.append(left_environment)
        if right_environment is not None:
            components.append(right_environment)
        return self._contract_components(components, output_inds=output_inds)

    @staticmethod
    def _validate_sweep_sequence(sweep_sequence):
        """Return a non-empty Quimb-compatible ``R``/``L`` sweep sequence."""
        sequence = str(sweep_sequence).strip().upper()
        if not sequence or any(direction not in {"R", "L"} for direction in sequence):
            raise ValueError("sweep_sequence must contain only 'R' and 'L'.")
        return sequence

    def _prepare_gate_sweep_environments(
        self,
        psi,
        start,
        stop,
        direction,
        *,
        block_size,
        timing_record=None,
        reuse_canonical_form=False,
        fixed_environments=None,
    ):
        """Prepare the gauge and fixed environments for one gate sweep.

        The preparation work is deliberately timed separately from the
        per-site records. In particular, fixed-environment construction takes
        place before the first active update and was previously hidden inside
        ``non_site_elapsed_seconds``. The canonicalization calls are Quimb's
        QR/gauge preparation route for dense MPS data (or the corresponding
        native route for other tensor types).
        """
        canonicalization_started = (
            self._timing_mark() if timing_record is not None else None
        )
        if direction == "R":
            if not reuse_canonical_form:
                for site in range(stop, start, -1):
                    psi.right_canonize_site(site, bra=None)
        else:
            if not reuse_canonical_form:
                for site in range(start, stop):
                    psi.left_canonize_site(site, bra=None)
        canonicalization_finished = (
            self._timing_mark(psi) if timing_record is not None else None
        )

        fixed_environment_started = (
            self._timing_mark() if timing_record is not None else None
        )
        if fixed_environments is None:
            fixed_environments = self._build_active_environments(
                psi,
                start,
                stop,
                direction,
                block_size=block_size,
            )
        fixed_environment_finished = (
            self._timing_mark(fixed_environments)
            if timing_record is not None
            else None
        )

        if timing_record is not None:
            preparation_canonicalization_seconds = float(
                canonicalization_finished - canonicalization_started
            )
            timing_record["canonicalization_seconds"] += (
                preparation_canonicalization_seconds
            )
            timing_record["sweep_preparation_canonicalization_seconds"] += (
                preparation_canonicalization_seconds
            )
            timing_record["fixed_environment_seconds"] += float(
                fixed_environment_finished - fixed_environment_started
            )

        return fixed_environments

    # ------------------------------------------------------------------
    # Active gate-window solver
    # ------------------------------------------------------------------

    @staticmethod
    def _isometrize_before_one_site_overwrite(psi, site, direction):
        """QR the optimized site when the next effective tensor replaces its neighbor.

        Cached environments exclude that neighbor from its own next update.
        Thus absorbing R into it is dead work, provided the bond size stays
        unchanged. This is deliberately not a general canonical-center move:
        the intermediate network does not represent the previous state.
        Native arrays and shape-changing QR keep Quimb's complete gauge move.
        """
        neighbor = site + (1 if direction == "R" else -1)
        tensor = psi[site]
        shared = tuple(ind for ind in tensor.inds if ind in psi[neighbor].inds)
        if len(shared) == 1 and ar.infer_backend(tensor.data) in {
            "numpy", "torch", "jax"
        }:
            bond, = shared
            left_inds = tuple(ind for ind in tensor.inds if ind != bond)
            # Reduced QR must not leave mismatched index sizes in the network.
            # Numerical rank deficiency alone does not shrink a dense QR.
            rows = math.prod(tensor.ind_size(ind) for ind in left_inds)
            if rows >= tensor.ind_size(bond):
                q, _ = tensor.split(
                    left_inds=left_inds, right_inds=(bond,), method="qr",
                    absorb="right", get="tensors",
                )
                q.transpose_like_(tensor)
                tensor.modify(data=q.data, left_inds=left_inds)
                return
        if direction == "R":
            psi.left_canonize_site(site, bra=None)
        else:
            psi.right_canonize_site(site, bra=None)

    def _run_gate_one_site_sweep(
        self,
        psi,
        start,
        stop,
        *,
        direction,
        timing_record,
        reuse_canonical_form=False,
        fixed_environments=None,
    ):
        """Perform one cached one-site FIT sweep in ``direction``."""
        # An R sweep leaves every site strictly to the left of ``stop``
        # left-canonical, and an L sweep leaves every site strictly to the
        # right of ``start`` right-canonical. When the next sweep reverses
        # direction, that is exactly the gauge preparation it needs. Reusing
        # it avoids repeating a full boundary canonicalization pass.
        if direction == "R":
            fixed_environments = self._prepare_gate_sweep_environments(
                psi,
                start,
                stop,
                "R",
                block_size=1,
                timing_record=timing_record,
                reuse_canonical_form=reuse_canonical_form,
                fixed_environments=fixed_environments,
            )
            moving_environments = {}
            if self._fermionic_left_exterior_environment is not None:
                moving_environments[start - 1] = (
                    self._fermionic_left_exterior_environment
                )
            sites = range(start, stop + 1)
        else:
            fixed_environments = self._prepare_gate_sweep_environments(
                psi,
                start,
                stop,
                "L",
                block_size=1,
                timing_record=timing_record,
                reuse_canonical_form=reuse_canonical_form,
                fixed_environments=fixed_environments,
            )
            moving_environments = {}
            if self._fermionic_right_exterior_environment is not None:
                moving_environments[stop + 1] = (
                    self._fermionic_right_exterior_environment
                )
            sites = range(stop, start - 1, -1)

        for site in sites:
            site_started = self._timing_mark() if timing_record is not None else None
            left_environment = (
                moving_environments.get(site - 1)
                if direction == "R"
                else fixed_environments.get(site - 1)
            )
            right_environment = (
                fixed_environments.get(site + 1)
                if direction == "R"
                else moving_environments.get(site + 1)
            )
            f = self._effective_tensor(
                psi,
                (site,),
                start,
                stop,
                left_environment=left_environment,
                right_environment=right_environment,
                output_inds=psi[site].inds,
            )
            f = self._prepare_fermionic_effective_tensor(
                f,
                psi[site],
                psi[site],
                left_environment,
                right_environment,
            )
            self._require_nonempty_fermionic_effective_tensor(
                f,
                (site,),
            )
            # Only the final update's sweep-facing tensor is the retained
            # canonical center returned to the caller. Its norm A determines
            # fidelity through (A / T)**2, or true infidelity 1 - A**2 for a
            # normalized target. Earlier local norms are neither used nor
            # authoritative after subsequent updates, so do not reduce them.
            terminal_update = site == (stop if direction == "R" else start)
            norm_f = f.norm() if terminal_update else None
            effective_finished = (
                self._timing_mark(f, norm_f)
                if terminal_update and timing_record is not None
                else self._timing_mark(f)
                if timing_record is not None
                else None
            )
            if terminal_update:
                self.local_norm_trace.append(ar.do("real", norm_f))
            psi[site].modify(data=f.data)
            self._resolve_fermionic_writeback_phase(psi[site])
            writeback_finished = (
                self._timing_mark(psi[site], norm_f)
                if terminal_update and timing_record is not None
                else self._timing_mark(psi[site])
                if timing_record is not None
                else None
            )

            moving_environment_started = writeback_finished
            moving_environment_updated = False
            moving_canonicalization_seconds = 0.0
            if direction == "R" and site < stop:
                moving_canonicalization_started = (
                    self._timing_mark() if timing_record is not None else None
                )
                self._isometrize_before_one_site_overwrite(psi, site, "R")
                moving_canonicalization_finished = (
                    self._timing_mark(psi[site])
                    if timing_record is not None
                    else None
                )
                if timing_record is not None:
                    moving_canonicalization_seconds = float(
                        moving_canonicalization_finished
                        - moving_canonicalization_started
                    )
                    timing_record["canonicalization_seconds"] += (
                        moving_canonicalization_seconds
                    )
                    moving_environment_started = moving_canonicalization_finished
                moving_environments[site] = self._overlap_environment_site(
                    psi,
                    site,
                    start,
                    stop,
                    prior=moving_environments.get(site - 1),
                )
                moving_environment_updated = True
            elif direction == "L" and site > start:
                moving_canonicalization_started = (
                    self._timing_mark() if timing_record is not None else None
                )
                self._isometrize_before_one_site_overwrite(psi, site, "L")
                moving_canonicalization_finished = (
                    self._timing_mark(psi[site])
                    if timing_record is not None
                    else None
                )
                if timing_record is not None:
                    moving_canonicalization_seconds = float(
                        moving_canonicalization_finished
                        - moving_canonicalization_started
                    )
                    timing_record["canonicalization_seconds"] += (
                        moving_canonicalization_seconds
                    )
                    moving_environment_started = moving_canonicalization_finished
                moving_environments[site] = self._overlap_environment_site(
                    psi,
                    site,
                    start,
                    stop,
                    prior=moving_environments.get(site + 1),
                )
                moving_environment_updated = True

            if timing_record is not None:
                environment_finished = self._timing_mark(moving_environments)
                moving_environment_seconds = (
                    float(environment_finished - moving_environment_started)
                    if moving_environment_updated
                    else 0.0
                )
                environment_seconds = float(
                    environment_finished - writeback_finished
                )
                timing_record["site_timings"].append(
                    {
                        "site": int(site),
                        "sites": (int(site),),
                        "block_size": 1,
                        "effective_seconds": float(
                            effective_finished - site_started
                        ),
                        "svd_seconds": 0.0,
                        "writeback_seconds": float(
                            writeback_finished - effective_finished
                        ),
                        "canonicalization_seconds": moving_canonicalization_seconds,
                        "environment_seconds": environment_seconds,
                        "moving_environment_seconds": moving_environment_seconds,
                        "elapsed_seconds": float(
                            environment_finished - site_started
                        ),
                    }
                )

        return moving_environments

    def _run_gate_two_site_sweep(
        self,
        psi,
        start,
        stop,
        *,
        direction,
        max_bond,
        cutoff,
        cutoff_mode,
        timing_record,
        collect_split_diagnostics,
        reuse_canonical_form=False,
        fixed_environments=None,
    ):
        """Optimize dense two-site wavefunctions and split their middle bond.

        ``Tensor.split`` is essential here: it dispatches to the registered
        NumPy/Torch/CuPy SVD for dense arrays and to Symmray's native block SVD
        for U1, U1xU1, and fermionic arrays. Calling ``numpy.linalg.svd`` would
        silently destroy charge sectors, dual-leg metadata, and graded signs.
        """
        # The previous opposite-direction sweep already produced the
        # canonical gauge required here. Same-direction sweeps, and the first
        # sweep of a run, still perform the explicit preparation pass.
        if direction == "R":
            fixed_environments = self._prepare_gate_sweep_environments(
                psi,
                start,
                stop,
                "R",
                block_size=2,
                timing_record=timing_record,
                reuse_canonical_form=reuse_canonical_form,
                fixed_environments=fixed_environments,
            )
            moving_environments = {}
            if self._fermionic_left_exterior_environment is not None:
                moving_environments[start - 1] = (
                    self._fermionic_left_exterior_environment
                )
            pairs = ((site, site + 1) for site in range(start, stop))
        else:
            fixed_environments = self._prepare_gate_sweep_environments(
                psi,
                start,
                stop,
                "L",
                block_size=2,
                timing_record=timing_record,
                reuse_canonical_form=reuse_canonical_form,
                fixed_environments=fixed_environments,
            )
            moving_environments = {}
            if self._fermionic_right_exterior_environment is not None:
                moving_environments[stop + 1] = (
                    self._fermionic_right_exterior_environment
                )
            pairs = ((site, site + 1) for site in range(stop - 1, start - 1, -1))

        for left_site, right_site in pairs:
            site_started = self._timing_mark() if timing_record is not None else None
            left_tensor = psi[left_site]
            right_tensor = psi[right_site]
            (bond,) = left_tensor.bonds(right_tensor)
            left_inds = tuple(index for index in left_tensor.inds if index != bond)
            right_inds = tuple(index for index in right_tensor.inds if index != bond)
            left_environment = (
                moving_environments.get(left_site - 1)
                if direction == "R"
                else fixed_environments.get(left_site - 1)
            )
            right_environment = (
                fixed_environments.get(right_site + 1)
                if direction == "R"
                else moving_environments.get(right_site + 1)
            )
            theta = self._effective_tensor(
                psi,
                (left_site, right_site),
                start,
                stop,
                left_environment=left_environment,
                right_environment=right_environment,
                output_inds=left_inds + right_inds,
            )
            theta = self._prepare_fermionic_effective_tensor(
                theta,
                left_tensor,
                right_tensor,
                left_environment,
                right_environment,
            )
            self._require_nonempty_fermionic_effective_tensor(
                theta,
                (left_site, right_site),
            )
            effective_finished = (
                self._timing_mark(theta)
                if timing_record is not None
                else None
            )

            split_info = {} if collect_split_diagnostics else None
            new_left, new_right = theta.split(
                left_inds=left_inds,
                right_inds=right_inds,
                method="svd",
                absorb="right" if direction == "R" else "left",
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                bond_ind=bond,
                ltags=left_tensor.tags,
                rtags=right_tensor.tags,
                get="tensors",
                info=split_info,
            )
            split_finished = (
                self._timing_mark(new_left, new_right)
                if timing_record is not None
                else None
            )
            new_left.transpose_like_(left_tensor)
            new_right.transpose_like_(right_tensor)
            left_tensor.modify(
                data=new_left.data,
                left_inds=new_left.left_inds,
            )
            right_tensor.modify(
                data=new_right.data,
                left_inds=new_right.left_inds,
            )
            self._resolve_fermionic_writeback_phase(
                left_tensor,
                right_tensor,
            )

            # The terminal SVD absorbs the retained singular values into the
            # final canonical center. Its norm A obeys the FIT projection
            # identity fidelity = (A / T)**2; for normalized ``p_target``, true
            # infidelity = 1 - A**2. Intermediate center norms are superseded
            # by later updates and would add a redundant backend reduction.
            center = right_tensor if direction == "R" else left_tensor
            terminal_update = (
                right_site == stop if direction == "R" else left_site == start
            )
            retained_norm = center.norm() if terminal_update else None
            if terminal_update:
                self.local_norm_trace.append(ar.do("real", retained_norm))
            if collect_split_diagnostics:
                self.info.setdefault("two_site_splits", []).append(
                    {
                        "sites": (int(left_site), int(right_site)),
                        "direction": direction,
                        "bond": bond,
                        "bond_dim": int(psi.bond_size(left_site, right_site)),
                        "max_bond": None if max_bond is None else int(max_bond),
                        "cutoff": float(cutoff),
                        "cutoff_mode": str(cutoff_mode),
                        "truncation_error": split_info.get("error"),
                    }
                )
            writeback_finished = (
                self._timing_mark(center, retained_norm)
                if terminal_update and timing_record is not None
                else self._timing_mark(center)
                if timing_record is not None
                else None
            )

            needs_next_update = (
                (direction == "R" and right_site < stop)
                or (direction == "L" and left_site > start)
            )
            moving_environment_updated = False
            if direction == "R" and needs_next_update:
                moving_environments[left_site] = self._overlap_environment_site(
                    psi,
                    left_site,
                    start,
                    stop,
                    prior=moving_environments.get(left_site - 1),
                )
                moving_environment_updated = True
            elif direction == "L" and needs_next_update:
                moving_environments[right_site] = self._overlap_environment_site(
                    psi,
                    right_site,
                    start,
                    stop,
                    prior=moving_environments.get(right_site + 1),
                )
                moving_environment_updated = True

            if timing_record is not None:
                environment_finished = self._timing_mark(moving_environments)
                moving_environment_seconds = (
                    float(environment_finished - writeback_finished)
                    if moving_environment_updated
                    else 0.0
                )
                timing_record["site_timings"].append(
                    {
                        "site": int(left_site),
                        "sites": (int(left_site), int(right_site)),
                        "block_size": 2,
                        "effective_seconds": float(
                            effective_finished - site_started
                        ),
                        "svd_seconds": float(
                            split_finished - effective_finished
                        ),
                        "writeback_seconds": float(
                            writeback_finished - split_finished
                        ),
                        "canonicalization_seconds": 0.0,
                        "environment_seconds": moving_environment_seconds,
                        "moving_environment_seconds": moving_environment_seconds,
                        "elapsed_seconds": float(
                            environment_finished - site_started
                        ),
                    }
                )

        return moving_environments

    def _run_gate_three_site_sweep(
        self,
        psi,
        start,
        stop,
        *,
        direction,
        max_bond,
        cutoff,
        cutoff_mode,
        timing_record,
        collect_split_diagnostics,
        reuse_canonical_form=False,
        fixed_environments=None,
    ):
        """Optimize three-site wavefunctions using two native SVD splits.

        The effective tensor is formed exactly like the two-site update, but
        its three physical site groups are split sequentially. In an ``R``
        sweep the first split makes the left site left-canonical and the
        second split makes the middle site left-canonical. In an ``L`` sweep
        the order is reversed, leaving the right site(s) right-canonical and
        the leftmost site as the retained center. No dense MPS conversion is
        used, so the same path remains available to Torch, CuPy, Symmray, and
        fermionic tensor backends.
        """
        # As with the one- and two-site paths, an opposite-direction sweep
        # can consume the canonical form produced by the previous sweep.
        if direction == "R":
            fixed_environments = self._prepare_gate_sweep_environments(
                psi,
                start,
                stop,
                "R",
                block_size=3,
                timing_record=timing_record,
                reuse_canonical_form=reuse_canonical_form,
                fixed_environments=fixed_environments,
            )
            moving_environments = {}
            if self._fermionic_left_exterior_environment is not None:
                moving_environments[start - 1] = (
                    self._fermionic_left_exterior_environment
                )
            blocks = (
                (site, site + 1, site + 2)
                for site in range(start, stop - 1)
            )
        else:
            fixed_environments = self._prepare_gate_sweep_environments(
                psi,
                start,
                stop,
                "L",
                block_size=3,
                timing_record=timing_record,
                reuse_canonical_form=reuse_canonical_form,
                fixed_environments=fixed_environments,
            )
            moving_environments = {}
            if self._fermionic_right_exterior_environment is not None:
                moving_environments[stop + 1] = (
                    self._fermionic_right_exterior_environment
                )
            blocks = (
                (site, site + 1, site + 2)
                for site in range(stop - 2, start - 1, -1)
            )

        for left_site, middle_site, right_site in blocks:
            site_started = self._timing_mark() if timing_record is not None else None
            left_tensor = psi[left_site]
            middle_tensor = psi[middle_site]
            right_tensor = psi[right_site]
            (left_bond,) = left_tensor.bonds(middle_tensor)
            (right_bond,) = middle_tensor.bonds(right_tensor)
            left_inds = tuple(
                index for index in left_tensor.inds if index != left_bond
            )
            middle_inds = tuple(
                index
                for index in middle_tensor.inds
                if index not in (left_bond, right_bond)
            )
            right_inds = tuple(
                index for index in right_tensor.inds if index != right_bond
            )
            left_environment = (
                moving_environments.get(left_site - 1)
                if direction == "R"
                else fixed_environments.get(left_site - 1)
            )
            right_environment = (
                fixed_environments.get(right_site + 1)
                if direction == "R"
                else moving_environments.get(right_site + 1)
            )
            theta = self._effective_tensor(
                psi,
                (left_site, middle_site, right_site),
                start,
                stop,
                left_environment=left_environment,
                right_environment=right_environment,
                output_inds=left_inds + middle_inds + right_inds,
            )
            theta = self._prepare_fermionic_effective_tensor(
                theta,
                left_tensor,
                right_tensor,
                left_environment,
                right_environment,
            )
            self._require_nonempty_fermionic_effective_tensor(
                theta,
                (left_site, middle_site, right_site),
            )
            effective_finished = (
                self._timing_mark(theta)
                if timing_record is not None
                else None
            )

            split_info_left = {} if collect_split_diagnostics else None
            split_info_right = {} if collect_split_diagnostics else None
            if direction == "R":
                new_left, middle_right = theta.split(
                    left_inds=left_inds,
                    right_inds=middle_inds + right_inds,
                    method="svd",
                    absorb="right",
                    max_bond=max_bond,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    bond_ind=left_bond,
                    ltags=left_tensor.tags,
                    rtags=middle_tensor.tags | right_tensor.tags,
                    get="tensors",
                    info=split_info_left,
                )
                middle_left_inds = tuple(
                    index
                    for index in middle_right.inds
                    if index in (set(middle_inds) | {left_bond})
                )
                new_middle, new_right = middle_right.split(
                    left_inds=middle_left_inds,
                    right_inds=right_inds,
                    method="svd",
                    absorb="right",
                    max_bond=max_bond,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    bond_ind=right_bond,
                    ltags=middle_tensor.tags,
                    rtags=right_tensor.tags,
                    get="tensors",
                    info=split_info_right,
                )
            else:
                left_middle, new_right = theta.split(
                    left_inds=left_inds + middle_inds,
                    right_inds=right_inds,
                    method="svd",
                    absorb="left",
                    max_bond=max_bond,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    bond_ind=right_bond,
                    ltags=left_tensor.tags | middle_tensor.tags,
                    rtags=right_tensor.tags,
                    get="tensors",
                    info=split_info_right,
                )
                middle_right_inds = tuple(
                    index
                    for index in left_middle.inds
                    if index in (set(middle_inds) | {right_bond})
                )
                new_left, new_middle = left_middle.split(
                    left_inds=left_inds,
                    right_inds=middle_right_inds,
                    method="svd",
                    absorb="left",
                    max_bond=max_bond,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    bond_ind=left_bond,
                    ltags=left_tensor.tags,
                    rtags=middle_tensor.tags,
                    get="tensors",
                    info=split_info_left,
                )
            split_finished = (
                self._timing_mark(new_left, new_middle, new_right)
                if timing_record is not None
                else None
            )

            new_left.transpose_like_(left_tensor)
            new_middle.transpose_like_(middle_tensor)
            new_right.transpose_like_(right_tensor)
            left_tensor.modify(
                data=new_left.data,
                left_inds=new_left.left_inds,
            )
            middle_tensor.modify(
                data=new_middle.data,
                left_inds=new_middle.left_inds,
            )
            right_tensor.modify(
                data=new_right.data,
                left_inds=new_right.left_inds,
            )
            self._resolve_fermionic_writeback_phase(
                left_tensor,
                middle_tensor,
                right_tensor,
            )

            # The terminal sweep-facing tensor carries the retained center norm
            # A after both orthogonal truncations. As in the two-site path,
            # fidelity = (A / ||p_target||)**2, and a normalized target has true
            # infidelity 1 - A**2. Earlier centers do not survive the sweep.
            center = right_tensor if direction == "R" else left_tensor
            terminal_update = (
                right_site == stop if direction == "R" else left_site == start
            )
            retained_norm = center.norm() if terminal_update else None
            if terminal_update:
                self.local_norm_trace.append(ar.do("real", retained_norm))
            if collect_split_diagnostics:
                self.info.setdefault("three_site_splits", []).append(
                    {
                        "sites": (
                            int(left_site),
                            int(middle_site),
                            int(right_site),
                        ),
                        "direction": direction,
                        "bonds": (left_bond, right_bond),
                        "bond_dims": (
                            int(psi.bond_size(left_site, middle_site)),
                            int(psi.bond_size(middle_site, right_site)),
                        ),
                        "max_bond": (
                            None if max_bond is None else int(max_bond)
                        ),
                        "cutoff": float(cutoff),
                        "cutoff_mode": str(cutoff_mode),
                        "truncation_errors": (
                            split_info_left.get("error"),
                            split_info_right.get("error"),
                        ),
                    }
                )
            writeback_finished = (
                self._timing_mark(center, retained_norm)
                if terminal_update and timing_record is not None
                else self._timing_mark(center)
                if timing_record is not None
                else None
            )

            needs_next_update = (
                (direction == "R" and right_site < stop)
                or (direction == "L" and left_site > start)
            )
            moving_environment_updated = False
            if direction == "R" and needs_next_update:
                moving_environments[left_site] = self._overlap_environment_site(
                    psi,
                    left_site,
                    start,
                    stop,
                    prior=moving_environments.get(left_site - 1),
                )
                moving_environment_updated = True
            elif direction == "L" and needs_next_update:
                moving_environments[right_site] = self._overlap_environment_site(
                    psi,
                    right_site,
                    start,
                    stop,
                    prior=moving_environments.get(right_site + 1),
                )
                moving_environment_updated = True

            if timing_record is not None:
                environment_finished = self._timing_mark(moving_environments)
                moving_environment_seconds = (
                    float(environment_finished - writeback_finished)
                    if moving_environment_updated
                    else 0.0
                )
                timing_record["site_timings"].append(
                    {
                        "site": int(left_site),
                        "sites": (
                            int(left_site),
                            int(middle_site),
                            int(right_site),
                        ),
                        "block_size": 3,
                        "effective_seconds": float(
                            effective_finished - site_started
                        ),
                        "svd_seconds": float(
                            split_finished - effective_finished
                        ),
                        "writeback_seconds": float(
                            writeback_finished - split_finished
                        ),
                        "canonicalization_seconds": 0.0,
                        "environment_seconds": moving_environment_seconds,
                        "moving_environment_seconds": moving_environment_seconds,
                        "elapsed_seconds": float(
                            environment_finished - site_started
                        ),
                    }
                )

        return moving_environments

    @_native_fermionic_bra_fit
    def run_gate(
        self,
        n_iter=8,
        verbose=False,
        *,
        block_size=2,
        sweep_sequence="RL",
        max_bond=None,
        cutoff=None,
        cutoff_mode="rsum2",
        min_iter=2,
        rtol="auto",
        patience=2,
        finite_check=False,
        timing=False,
        timing_sync_device=False,
        single_pair_fast_path=False,
        three_site_sweeps=1,
        adaptive_block_sweeps=2,
        adaptive_until_rank=False,
        two_site_transition_sweeps=1,
        final_one_site_sweeps=0,
        collect_split_diagnostics=False,
    ):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Run fitting restricted to ``range_int`` with gate-style sweeps.

        This is the gate-restricted, cached-environment form of the paper's
        DMRG/FIT update. ``block_size=1`` updates one tensor with fixed bond
        dimensions. ``block_size=2`` forms a wavefunction tensor with the two
        outer virtual legs and both sites' physical legs, then performs a
        native SVD across the middle bond. ``block_size=3`` forms the analogous
        three-site tensor and performs two direction-aware native SVDs. The
        block updates can discover and grow useful bond subspaces up to
        ``max_bond`` without globally padding the MPS.

        ``run_eff`` implements the same environment-reuse idea for a complete
        MPS. Keeping this method separate is intentional: MpsOptimizer uses
        it for local gate compression, and must not refit unrelated sites.

        ``sweep_sequence`` follows Quimb's convention: ``"R"`` sweeps from
        left to right and ``"L"`` sweeps right to left; sequences such as
        ``"RL"`` alternate directions. By default, exactly ``n_iter`` sweeps
        are performed. Supplying ``rtol``
        enables early stopping after ``min_iter`` sweeps once the final local
        norm changes by at most ``rtol`` across a ``patience``-sample window.
        Thus ``patience=2`` means one stable comparison between two same-phase
        sweep norms; ``patience=1`` retains the same minimum comparable pair.
        ``finite_check=False`` is the default: these optional diagnostics are
        not required for normal optimization. Enabling them emits a performance
        warning unless the owning optimizer has already warned for this replay.
        ``finite_check=True`` reduces all active tensor blocks to native
        finite-status scalars and transfers one tiny vector per sweep. The
        terminal retained norm used by ``rtol`` shares that transfer.
        With ``finite_check=False``,
        the non-finite norm guard is skipped while ``rtol`` still reads the
        scalar and compares convergence. A callable
        retains the general state-check callback behavior. ``timing=True``
        records one wall-clock entry per sweep and per active-site update.
        Accelerator timings become kernel-complete when
        ``timing_sync_device=True``.
        ``three_site_sweeps`` controls how many initial sweeps use the
        three-site update when ``adaptive_block_sweeps`` is not supplied.
        ``adaptive_block_sweeps`` requests a common adaptive warm-up for
        two- or three-site updates; remaining sweeps use one-site refinement,
        which preserves the bond space opened by the larger block.
        ``adaptive_until_rank=True`` interprets that value as the minimum
        number of adaptive sweeps and keeps the larger block until every
        active bond reaches its physical ``max_bond`` ceiling. There is no
        rank-stability early exit: if a target remains rank-deficient, the
        larger block is retained until ``n_iter`` is exhausted. This is the
        optional rank-adaptive schedule; named MPS DMRG modes use fixed phases.
        Three-site fits insert ``two_site_transition_sweeps`` two-site sweeps
        after the block phase, within the same ``n_iter`` budget, before
        one-site refinement. Set this to zero for a direct three-to-one handoff.
        The default gate fit uses eight alternating RL sweeps, two initial
        two-site sweeps, and dtype-aware ``rtol="auto"`` convergence with
        ``min_iter=2`` and ``patience=2``. Explicit ``rtol=None`` uses fixed
        sweeps. Split diagnostics are disabled by default.
        For ordinary dense arrays, an opposite-direction sweep reuses the
        compatible partial environments produced by the preceding sweep. A
        smaller reversed block extends that cache only through the missing
        terminal tensors, including three-to-two and two-/three-to-one changes.
        ``single_pair_fast_path=True`` stops a two-site interval after its one
        exact variational update; additional sweeps cannot change that local
        optimum. ``final_one_site_sweeps`` optionally adds fixed-rank one-site
        polish sweeps after two- or three-site FIT on windows spanning at least
        three sites; it is ignored for a two-site window.
        ``collect_split_diagnostics=False`` avoids allocating SVD metadata when
        the caller only needs the fitted state.
        The supplied ``p`` is always the live variational initial state. If
        active bonds need a larger dense initialization, callers should
        expand and seed that MPS before constructing FIT; FIT itself never
        installs a target copy as ``p``.
        """
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")
        if not isinstance(block_size, Integral) or int(block_size) not in {1, 2, 3}:
            raise ValueError("block_size must be 1, 2, or 3.")
        block_size = int(block_size)
        sweep_sequence = self._validate_sweep_sequence(sweep_sequence)
        if max_bond is not None:
            if not isinstance(max_bond, Integral) or int(max_bond) < 1:
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        if cutoff is None:
            cutoff = self.cutoffs
        cutoff = self._resolve_cutoff(cutoff)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be a finite non-negative number.")
        if not isinstance(n_iter, Integral) or int(n_iter) < 1:
            raise ValueError("n_iter must be a positive integer.")
        n_iter = int(n_iter)
        if not isinstance(three_site_sweeps, Integral) or int(three_site_sweeps) < 1:
            raise ValueError("three_site_sweeps must be a positive integer.")
        three_site_sweeps = min(int(three_site_sweeps), n_iter)
        if block_size != 3 and three_site_sweeps != 1:
            raise ValueError(
                "three_site_sweeps is only configurable when block_size=3."
            )
        adaptive_schedule = adaptive_block_sweeps is not None
        if adaptive_schedule:
            if (
                not isinstance(adaptive_block_sweeps, Integral)
                or int(adaptive_block_sweeps) < 1
            ):
                raise ValueError(
                    "adaptive_block_sweeps must be a positive integer or None."
                )
            adaptive_block_sweeps = min(int(adaptive_block_sweeps), n_iter)
        else:
            adaptive_block_sweeps = (
                three_site_sweeps if block_size == 3 else n_iter
            )
        adaptive_until_rank = bool(adaptive_until_rank)
        if (
            not isinstance(two_site_transition_sweeps, Integral)
            or int(two_site_transition_sweeps) < 0
        ):
            raise ValueError("two_site_transition_sweeps must be a non-negative integer.")
        two_site_transition_sweeps = int(two_site_transition_sweeps)
        if (
            not isinstance(final_one_site_sweeps, Integral)
            or int(final_one_site_sweeps) < 0
        ):
            raise ValueError("final_one_site_sweeps must be a non-negative integer.")
        final_one_site_sweeps = int(final_one_site_sweeps)
        if rtol == "auto":
            dtype_names = [str(t.data.dtype).lower() for t in self.p.tensors]
            rtol = (
                1e-3 if any("16" in d for d in dtype_names)
                else 1e-5 if any("32" in d or "complex64" in d for d in dtype_names)
                else 1e-9
            )
        if min_iter is None:
            min_iter = n_iter if rtol is None else 1
        if not isinstance(min_iter, Integral) or int(min_iter) < 1:
            raise ValueError("min_iter must be a positive integer or None.")
        min_iter = min(int(min_iter), n_iter)
        if rtol is not None:
            rtol = float(rtol)
            if not math.isfinite(rtol) or rtol < 0.0:
                raise ValueError("rtol must be a finite non-negative number or None.")
        if not isinstance(patience, Integral) or int(patience) < 1:
            raise ValueError("patience must be a positive integer.")
        patience = int(patience)
        if finite_check not in (None, False, True) and not callable(finite_check):
            raise TypeError("finite_check must be bool, callable, or None.")
        # Validation is diagnostic only; normal fitting does not need it.
        # Warn once at the owning replay boundary, or here for standalone FIT.
        if (
            (finite_check is True or callable(finite_check))
            and not self._finite_check_warning_handled
        ):
            warnings.warn(
                "FIT finite_check is enabled: this optional diagnostic is "
                "off by default and is not required for normal optimization. "
                "It adds validation work and can synchronize accelerator "
                "devices; use finite_check=False to avoid this overhead.",
                RuntimeWarning,
                stacklevel=2,
            )
        timing = bool(timing)
        timing_sync_device = bool(timing_sync_device)
        single_pair_fast_path = bool(single_pair_fast_path)
        collect_split_diagnostics = bool(collect_split_diagnostics)
        self._timing_sync_device = timing and timing_sync_device
        self._timing_synchronizer = (
            self._make_backend_synchronizer(self.p)
            if self._timing_sync_device
            else None
        )
        if timing:
            self.timing_records = []

        self._reset_run_traces()
        self.iterations_run = 0
        self.converged = False
        self.convergence_reason = "max_sweeps"
        self.last_relative_change = None
        self.final_center_site = None
        self.final_direction = None
        self.final_norm = None
        self.adaptive_sweeps_run = 0
        self.one_site_sweeps_run = 0

        psi = self.p
        L = self.L

        if L == 1:
            if self.warning:
                logger.warning("run_gate called for L=1; falling back to run().")
            self.run(n_iter=n_iter, verbose=verbose)
            return

        if len(self.range_int) != 2:
            raise ValueError("range_int must be set to (start, stop) before calling run_gate.")
        start, stop = self.range_int
        if start < 0 or stop >= L or start > stop:
            raise ValueError(f"range_int={self.range_int} is out of bounds for L={L}.")
        if stop == start:
            raise ValueError("run_gate requires range_int spanning at least two sites.")
        if block_size == 3 and stop - start + 1 < 3:
            raise ValueError(
                "block_size=3 requires range_int to span at least three sites."
            )
        if (
            adaptive_schedule
            and block_size in {2, 3}
            and stop - start >= 2
            and n_iter < 2
        ):
            raise ValueError(
                "adaptive two-/three-site FIT requires n_iter >= 2 for an "
                "active window spanning at least three sites."
            )

        if self._fermionic_bra_working:
            self._prepare_fermionic_active_fit(
                psi,
                start,
                stop,
                sweep_sequence[0],
            )
            self.info["fermionic_sweep_sequence"] = {
                "requested": sweep_sequence,
                "used": sweep_sequence,
                "reason": "native_conjugated_fit_gauge",
            }

        previous_sweep_norm = None
        stable_sweeps = 0
        sweep_cache = None
        self._sweep_environment_reuse_count = 0
        adaptive_phase_done = not (
            adaptive_schedule
            and adaptive_until_rank
            and block_size in {2, 3}
            and stop - start >= 2
        )
        rank_targets = (
            self._active_bond_rank_targets(psi, start, stop, max_bond)
            if not adaptive_phase_done
            else None
        )

        block_phase_end = None if not adaptive_phase_done else adaptive_block_sweeps

        def block_size_for_sweep(sweep_number):
            """Resolve the active block after any live rank-phase update."""
            if block_size not in {2, 3}:
                return 1
            if adaptive_until_rank:
                use_block = not adaptive_phase_done or (
                    stop == start + 1 and block_size == 2
                )
            else:
                use_block = sweep_number <= adaptive_block_sweeps
            if use_block:
                return block_size
            if (
                block_size == 3
                and block_phase_end is not None
                and sweep_number <= block_phase_end + two_site_transition_sweeps
            ):
                return 2
            return 1

        for sweep in range(1, n_iter + 1):
            direction = sweep_sequence[(sweep - 1) % len(sweep_sequence)]
            previous_direction = (
                None if sweep_cache is None else sweep_cache.direction
            )
            previous_block_size = (
                None if sweep_cache is None else sweep_cache.block_size
            )
            active_block_size = block_size_for_sweep(sweep)
            if previous_block_size is not None and active_block_size != previous_block_size:
                # A block-to-one-site transition changes the optimization
                # regime. Do not compare the first fixed-rank refinement
                # sweep against the last adaptive split when applying rtol.
                previous_sweep_norm = None
                stable_sweeps = 0
                self.last_relative_change = None
            reuse_canonical_form = (
                self._fermionic_bra_working
                and previous_direction is None
            ) or (
                previous_direction is not None
                and previous_direction != direction
            )
            sweep_timing = self._start_timing_record(
                sweep,
                timing,
                direction=direction,
                block_size=active_block_size,
            )
            self.iterations_run = sweep
            if active_block_size == 1:
                self.one_site_sweeps_run += 1
            else:
                self.adaptive_sweeps_run += 1
            sweep_norm_start = len(self.local_norm_trace)
            fixed_environments = None
            if self._allow_sweep_environment_reuse and sweep_cache is not None:
                fixed_environments = sweep_cache.fixed_for(
                    direction=direction,
                    block_size=active_block_size,
                )
            if fixed_environments is not None:
                self._sweep_environment_reuse_count += 1
            one_site_ready = active_block_size == 1
            two_site_ready = False
            try:
                if active_block_size == 1:
                    boundaries = self._run_gate_one_site_sweep(
                        psi,
                        start,
                        stop,
                        direction=direction,
                        timing_record=sweep_timing,
                        reuse_canonical_form=reuse_canonical_form,
                        fixed_environments=fixed_environments,
                    )
                else:
                    if active_block_size == 2:
                        boundaries = self._run_gate_two_site_sweep(
                            psi,
                            start,
                            stop,
                            direction=direction,
                            max_bond=max_bond,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            timing_record=sweep_timing,
                            collect_split_diagnostics=collect_split_diagnostics,
                            reuse_canonical_form=reuse_canonical_form,
                            fixed_environments=fixed_environments,
                        )
                    else:
                        boundaries = self._run_gate_three_site_sweep(
                            psi,
                            start,
                            stop,
                            direction=direction,
                            max_bond=max_bond,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            timing_record=sweep_timing,
                            collect_split_diagnostics=collect_split_diagnostics,
                            reuse_canonical_form=reuse_canonical_form,
                            fixed_environments=fixed_environments,
                        )

                self.final_direction = direction
                self.final_center_site = stop if direction == "R" else start
                if len(self.local_norm_trace) != sweep_norm_start + 1:
                    raise RuntimeError(
                        "FIT sweep did not produce exactly one terminal center norm."
                    )
                self.final_norm = self.local_norm_trace[-1]

                if verbose:
                    fidelity = tn_fidelity(
                        self.tn,
                        self._physical_working_state(psi),
                        contraction_opt=self.contraction_opt,
                    )
                    self.fidelity_trace.append(ar.do("real", fidelity))

                if callable(finite_check) and not bool(
                    finite_check(self._physical_working_state(psi))
                ):
                    error = FloatingPointError(
                        f"FIT gate sweep {sweep} produced non-finite tensor data."
                    )
                    error.fit_iteration = sweep
                    raise error

                sweep_norm = None
                if finite_check is True or rtol is not None:
                    finite, sweep_norm = self._sweep_diagnostics_to_host(
                        psi,
                        start,
                        stop,
                        self.final_norm,
                        check_finite=finite_check is True,
                        read_norm=rtol is not None,
                    )
                    if not finite:
                        error = FloatingPointError(
                            f"FIT gate sweep {sweep} produced non-finite tensor data."
                        )
                        error.fit_iteration = sweep
                        raise error

                should_stop = False
                if (
                    single_pair_fast_path
                    and active_block_size == 2
                    and stop == start + 1
                ):
                    # There is only one variational pair. Its effective tensor
                    # and native SVD solve the complete active problem in one
                    # update, so rebuilding identical environments is wasted.
                    self.converged = True
                    self.convergence_reason = "single_pair_exact"
                    self.last_relative_change = 0.0
                    should_stop = True
                if rtol is not None:
                    self.sweep_norm_trace.append(sweep_norm)
                    # Convergence still needs the norm when diagnostics are
                    # off; detecting non-finite values is a separate opt-in.
                    if (
                        (finite_check is True or callable(finite_check))
                        and not math.isfinite(sweep_norm)
                    ):
                        error = FloatingPointError(
                            f"FIT gate sweep {sweep} produced a non-finite local norm."
                        )
                        error.fit_iteration = sweep
                        raise error
                    reset_tolerance = False
                    if previous_sweep_norm is not None:
                        # For a fixed target norm, A is proportional to the
                        # square root of the true fidelity. Thus this tests
                        # convergence of fidelity through changes in A; it is
                        # not an absolute threshold on 1 - A**2.
                        scale = max(
                            abs(sweep_norm),
                            abs(previous_sweep_norm),
                            float.fromhex("0x1.0p-1022"),
                        )
                        relative_change = abs(
                            sweep_norm - previous_sweep_norm
                        ) / scale
                        self.last_relative_change = relative_change
                        if relative_change <= rtol:
                            stable_sweeps += 1
                        else:
                            stable_sweeps = 0
                        # ``patience`` counts norm samples in the tolerance
                        # window, not additional comparisons after its first
                        # sample. Thus the public default of two stops after
                        # one stable comparison between two same-phase sweeps.
                        required_stable_changes = max(1, patience - 1)
                        if (
                            sweep >= min_iter
                            and stable_sweeps >= required_stable_changes
                        ):
                            # An explicitly requested adaptive schedule must
                            # complete its block warm-up before convergence can
                            # stop it. If refinement remains, transition to
                            # one-site sweeps and start a fresh tolerance run.
                            has_block_warmup = (
                                adaptive_schedule and block_size in {2, 3}
                            )
                            warmup_incomplete = (
                                has_block_warmup
                                and sweep < adaptive_block_sweeps
                            )
                            warmup_finished_with_refinement = (
                                has_block_warmup
                                and sweep == adaptive_block_sweeps
                                and sweep < n_iter
                            )
                            adaptive_rank_incomplete = (
                                adaptive_until_rank
                                and not adaptive_phase_done
                                and active_block_size in {2, 3}
                            )
                            if not (
                                warmup_incomplete
                                or warmup_finished_with_refinement
                                or adaptive_rank_incomplete
                                or (
                                    block_size == 3
                                    and two_site_transition_sweeps > 0
                                    and active_block_size != 1
                                    and sweep < n_iter
                                )
                            ):
                                self.converged = True
                                self.convergence_reason = "relative_tolerance"
                                should_stop = True
                            elif (
                                warmup_finished_with_refinement
                                or adaptive_rank_incomplete
                            ):
                                reset_tolerance = True
                                stable_sweeps = 0
                                self.last_relative_change = None
                    previous_sweep_norm = None if reset_tolerance else sweep_norm

                if (
                    adaptive_until_rank
                    and not adaptive_phase_done
                    and active_block_size in {2, 3}
                ):
                    current_ranks = tuple(
                        int(psi.bond_size(site, site + 1))
                        for site in range(start, stop)
                    )
                    rank_ready = (
                        sweep >= adaptive_block_sweeps
                        and rank_targets is not None
                        and all(
                            current >= target
                            for current, target in zip(current_ranks, rank_targets)
                        )
                    )
                    if sweep >= adaptive_block_sweeps and rank_ready:
                        adaptive_phase_done = True
                        block_phase_end = sweep
                        # The first one-site sweep is a new numerical phase;
                        # do not compare its norm with the last SVD sweep.
                        previous_sweep_norm = None
                        stable_sweeps = 0
                        self.last_relative_change = None

                next_sweep = None
                next_block_size = None
                if not should_stop and sweep < n_iter:
                    next_sweep = sweep + 1
                    next_block_size = block_size_for_sweep(next_sweep)
                elif (
                    block_size in {2, 3}
                    and stop - start + 1 >= 3
                    and final_one_site_sweeps > 0
                ):
                    next_sweep = sweep + 1
                    next_block_size = 1

                if (
                    self._allow_sweep_environment_reuse
                    and active_block_size in {2, 3}
                    and next_block_size is not None
                    and next_block_size < active_block_size
                    and next_sweep is not None
                    and sweep_sequence[(next_sweep - 1) % len(sweep_sequence)]
                    != direction
                ):
                    self._extend_block_cache_for_smaller_block(
                        psi,
                        boundaries,
                        start,
                        stop,
                        direction,
                        block_size=active_block_size,
                        next_block_size=next_block_size,
                        timing_record=sweep_timing,
                    )
                    one_site_ready = next_block_size == 1
                    two_site_ready = next_block_size <= 2
            except BaseException as error:
                self.convergence_reason = "failed"
                if sweep_timing is not None:
                    sweep_timing["error"] = f"{type(error).__name__}: {error}"
                    self._finish_timing_record(sweep_timing, status="failed")
                raise
            else:
                sweep_cache = _SweepEnvironmentCache(
                    boundaries,
                    direction=direction,
                    block_size=active_block_size,
                    one_site_ready=one_site_ready,
                    two_site_ready=two_site_ready,
                )
                if sweep_timing is not None:
                    self._finish_timing_record(sweep_timing, status="complete")
                if should_stop:
                    break

        if (
            block_size in {2, 3}
            and stop - start + 1 >= 3
            and final_one_site_sweeps > 0
        ):
            polish_start = self.iterations_run + 1
            for polish_index in range(final_one_site_sweeps):
                sweep = polish_start + polish_index
                direction = sweep_sequence[(sweep - 1) % len(sweep_sequence)]
                previous_direction = (
                    None if sweep_cache is None else sweep_cache.direction
                )
                reuse_canonical_form = (
                    previous_direction is not None
                    and previous_direction != direction
                )
                sweep_timing = self._start_timing_record(
                    sweep,
                    timing,
                    direction=direction,
                    block_size=1,
                )
                self.iterations_run = sweep
                self.one_site_sweeps_run += 1
                sweep_norm_start = len(self.local_norm_trace)
                fixed_environments = None
                if self._allow_sweep_environment_reuse and sweep_cache is not None:
                    fixed_environments = sweep_cache.fixed_for(
                        direction=direction,
                        block_size=1,
                    )
                if fixed_environments is not None:
                    self._sweep_environment_reuse_count += 1
                try:
                    boundaries = self._run_gate_one_site_sweep(
                        psi,
                        start,
                        stop,
                        direction=direction,
                        timing_record=sweep_timing,
                        reuse_canonical_form=reuse_canonical_form,
                        fixed_environments=fixed_environments,
                    )
                    self.final_direction = direction
                    self.final_center_site = stop if direction == "R" else start
                    if len(self.local_norm_trace) != sweep_norm_start + 1:
                        raise RuntimeError(
                            "FIT polish sweep did not produce exactly one terminal "
                            "center norm."
                        )
                    self.final_norm = self.local_norm_trace[-1]

                    if verbose:
                        fidelity = tn_fidelity(
                            self.tn,
                            self._physical_working_state(psi),
                            contraction_opt=self.contraction_opt,
                        )
                        self.fidelity_trace.append(ar.do("real", fidelity))

                    if callable(finite_check) and not bool(
                        finite_check(self._physical_working_state(psi))
                    ):
                        error = FloatingPointError(
                            f"FIT gate sweep {sweep} produced non-finite tensor data."
                        )
                        error.fit_iteration = sweep
                        raise error

                    if finite_check is True:
                        finite, _ = self._sweep_diagnostics_to_host(
                            psi,
                            start,
                            stop,
                            self.final_norm,
                            check_finite=True,
                            read_norm=False,
                        )
                        if not finite:
                            error = FloatingPointError(
                                f"FIT gate sweep {sweep} produced non-finite tensor data."
                            )
                            error.fit_iteration = sweep
                            raise error
                except BaseException as error:
                    self.convergence_reason = "failed"
                    if sweep_timing is not None:
                        sweep_timing["error"] = f"{type(error).__name__}: {error}"
                        self._finish_timing_record(sweep_timing, status="failed")
                    raise
                else:
                    sweep_cache = _SweepEnvironmentCache(
                        boundaries,
                        direction=direction,
                        block_size=1,
                    )
                    if sweep_timing is not None:
                        self._finish_timing_record(sweep_timing, status="complete")

    # ------------------------------------------------------------------
    # Timing and diagnostics
    # ------------------------------------------------------------------

    def _start_timing_record(
        self,
        sweep,
        enabled,
        *,
        direction="R",
        block_size=1,
    ):
        """Start one opt-in FIT sweep timing record."""
        if not enabled:
            return None
        started = self._timing_mark()
        return {
            "timing_schema": 3,
            "sweep": int(sweep),
            "range_int": tuple(self.range_int),
            "active_site_count": int(self.range_int[1] - self.range_int[0] + 1),
            "direction": str(direction),
            "block_size": int(block_size),
            "environment_strategy": self.environment_strategy,
            "timing_sync_device": bool(self._timing_sync_device),
            "canonicalization_seconds": 0.0,
            "sweep_preparation_canonicalization_seconds": 0.0,
            "fixed_environment_seconds": 0.0,
            "site_timings": [],
            "_started": started,
        }

    def _finish_timing_record(self, record, *, status):
        """Finalize and retain one FIT sweep timing record."""
        started = record.pop("_started")
        record["status"] = status
        record["elapsed_seconds"] = float(self._timing_mark() - started)
        record["site_count"] = len(record["site_timings"])
        record["site_elapsed_seconds"] = float(
            sum(site["elapsed_seconds"] for site in record["site_timings"])
        )
        record["update_count"] = record["site_count"]
        record["moving_environment_seconds"] = float(
            sum(
                site.get(
                    "moving_environment_seconds",
                    site.get("environment_seconds", 0.0),
                )
                for site in record["site_timings"]
            )
        )
        record["moving_canonicalization_seconds"] = float(
            sum(
                site.get("canonicalization_seconds", 0.0)
                for site in record["site_timings"]
            )
        )
        for stage in (
            "effective_seconds",
            "svd_seconds",
            "writeback_seconds",
            "environment_seconds",
        ):
            record[stage] = float(
                sum(site.get(stage, 0.0) for site in record["site_timings"])
            )
        record["non_site_elapsed_seconds"] = max(
            0.0,
            record["elapsed_seconds"] - record["site_elapsed_seconds"],
        )
        record["sweep_overhead_seconds"] = max(
            0.0,
            record["non_site_elapsed_seconds"]
            - record["sweep_preparation_canonicalization_seconds"]
            - record["fixed_environment_seconds"],
        )
        record["converged"] = bool(self.converged)
        record["convergence_reason"] = self.convergence_reason
        record["relative_change"] = self.last_relative_change
        self.timing_records.append(record)

    def _timing_mark(self, *values):
        """Return a wall-clock mark after an optional device barrier."""
        if self._timing_synchronizer is not None:
            value = self.p if not values else values[0] if len(values) == 1 else values
            self._timing_synchronizer.synchronize(value, fallback=self.p)
        return time.perf_counter()

    @staticmethod
    def _make_backend_synchronizer(value):
        """Resolve one reusable accelerator barrier for ``value``."""
        return _BackendSynchronizer.from_value(value)

    @staticmethod
    def synchronize_backend(tn):
        """Synchronize accelerator work for explicit external profiling."""
        _synchronize_tensor_network(tn)

    def _take_timing_records(self):
        """Transfer ownership of timing records to an internal consumer."""
        records = self.timing_records
        self.timing_records = []
        return records

    def get_timing(self):
        """Return a copy of the most recent opt-in per-sweep timing records."""
        return deepcopy(self.timing_records)
