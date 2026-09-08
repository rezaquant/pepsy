"""Tree-embedded PEPS operator replay and compression."""

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping
from numbers import Integral
from time import perf_counter

import autoray as ar
import numpy as np

from ..._internal.cutoff import dtype_auto_cutoff
from ...backends import (
    backend_infer,
    backend_signatures_compatible,
    infer_backend_signature,
)
from ...fitting import TreeFIT
from ...fitting.tree import (
    _build_layered_operator_state_target,
    _layered_target_bond_sizes,
    _randomize_tree_guess,
)
from .._fidelity import (
    fidelity_from_log,
    infidelity_from_log,
    log_fidelity_from_norms,
)
from .operators import (
    TreePepo,
    TreeSubPepo,
    _normalize_compression_layout,
    plan_signature,
)
from .plan import TreePepsPlan
from .state import TreePeps, _normalize_compression_mode

__all__ = ["TreePepsOptimizer"]

_UNSET = object()


class TreePepsOptimizer:
    """Apply gates and tree-structured PEPO fragments to a ``TreePeps``.

    Two update modes are supported:

    ``"direct"``
        Build a small dense operator on the supplied support, factorize it on
        the support's unique tree span, and compress only that span.  For a
        two-site gate the span is exactly the tree geodesic between its sites.

    ``"sub_treepepo"``
        Apply an already factorized :class:`TreeSubPepo`.  The complete
        operator span is compressed either as a Quimb-compatible two-layer
        path network or through the fused tree-state fallback.

    ``compression_layout="auto"`` selects the two-layer path network for
    Quimb's multi-tensor methods and keeps the fused representation for other
    ordinary compression cases.  The DMRG/TreeFIT route always builds a
    layered operator--state target. Use ``"fused"`` or ``"two_layer"`` to
    select the ordinary compression layout explicitly. The optimizer owns an
    independent state copy by default and mutates that live state when
    :meth:`apply` or :meth:`apply_gate` is called. ``mode="dmrg"``
    and its ``dmrg1``/``dmrg2``/``dmrg3`` aliases select the cached
    tree-native :class:`pepsy.fitting.TreeFIT` engine. ``dmrg1`` and
    ``dmrg2`` use two-node warm-up blocks, ``dmrg3`` uses three-node warm-up
    blocks, and all named modes refine with one-node sweeps.
    """

    _MODE_ALIASES = {
        "subtreepepo": "sub_treepepo",
        "sub_tree_pepo": "sub_treepepo",
        "subtree_pepo": "sub_treepepo",
        "subtree": "sub_treepepo",
        # MPS-style compatibility spelling.  The canonical PEPS name is
        # ``sub_treepepo`` because the operator is a tree PEPO, not a chain
        # MPO.
        "sub_treepepsmpo": "sub_treepepo",
        "sub_tree_peps_mpo": "sub_treepepo",
        "subtreepepsmpo": "sub_treepepo",
        "subtree_peps_mpo": "sub_treepepo",
    }
    _DMRG_MODE_ALIASES = {"dmrg1": 1, "dmrg2": 2, "dmrg3": 3}
    _PROGBAR_COLORS = {
        # Match MpsOptimizer's DMRG and MPO-family colors.
        "dmrg": "#1f77b4",
        "mpo": "#2ca02c",
    }
    _STREAM_EVENT_ALIASES = {
        "gate": "gate",
        "dense": "gate",
        "dense_gate": "gate",
        "pepo": "tree_pepo",
        "tree_pepo": "tree_pepo",
        "treepepo": "tree_pepo",
        "sub_treepepo": "sub_treepepo",
        "sub_tree_pepo": "sub_treepepo",
        "subtree_pepo": "sub_treepepo",
        "subtreepepo": "sub_treepepo",
        "sub_treepepsmpo": "sub_treepepo",
        "sub_tree_peps_mpo": "sub_treepepo",
        "subtreepepsmpo": "sub_treepepo",
        "subtree_peps_mpo": "sub_treepepo",
        # The normal/full PEPS operator is ``tree_pepo``.  These aliases make
        # the MPS-style naming available at the stream boundary only.
        "tree_pepsmpo": "tree_pepo",
        "tree_peps_mpo": "tree_pepo",
        "treepepsmpo": "tree_pepo",
        "treepeps_mpo": "tree_pepo",
    }

    def __init__(
        self,
        state: TreePeps | None = None,
        *,
        tn=None,
        plan: TreePepsPlan | None = None,
        layout=None,
        mode="direct",
        compression_mode="direct",
        compression_seed=None,
        compression_layout="auto",
        fit_block_size=2,
        fit_n_iter=2,
        fit_adaptive_sweeps=2,
        fit_min_iter=None,
        fit_rtol=None,
        fit_patience=1,
        fit_init_strategy="guess-src",
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_sweep_sequence="RL",
        fit_overlap_diagnostics=False,
        chi=64,
        max_bond=None,
        cutoff="auto",
        cutoff_mode="rsum2",
        reduced=True,
        inplace=False,
        info_c=None,
        max_operator_sites=_UNSET,
        max_operator_qubits=_UNSET,
        max_subtree_nodes=None,
        gates=None,
        run=True,
        record_history=True,
        track_truncation=False,
        track_infidelity=True,
        max_intermediate_bond=None,
        profile=False,
        profile_sync=False,
        track_bond_diagnostics=False,
    ):
        if state is not None and tn is not None:
            raise ValueError("pass either state= or tn=, not both")
        if state is None:
            state = tn
        if not isinstance(state, TreePeps):
            raise TypeError("state must be a TreePeps")
        if layout is not None:
            if plan is not None:
                raise TypeError("pass either plan= or layout=, not both")
            plan = layout.plan if hasattr(layout, "plan") else layout
        if plan is not None:
            if not isinstance(plan, TreePepsPlan):
                raise TypeError("plan must be a TreePepsPlan")
            if plan_signature(plan) != plan_signature(state.plan):
                raise ValueError("plan and state must use the same tree plan")
        compression_mode = _normalize_compression_mode(compression_mode)
        raw_mode = str(mode).strip().lower().replace("-", "_")
        if raw_mode in {"dm", "sdc", "src", "zipup"}:
            if compression_mode not in {"direct", raw_mode}:
                raise ValueError(
                    f"mode={raw_mode!r} cannot be combined with a different "
                    "compression_mode."
                )
            compression_mode = raw_mode
            raw_mode = "direct"
        self._dmrg_mode_alias = (
            raw_mode if raw_mode in self._DMRG_MODE_ALIASES else None
        )
        if raw_mode == "fit" or raw_mode in self._DMRG_MODE_ALIASES:
            raw_mode = "dmrg"
        self.mode = self._normalize_mode(raw_mode)
        self.compression_mode = compression_mode
        self.compression_layout = _normalize_compression_layout(compression_layout)
        if (
            not isinstance(fit_block_size, Integral)
            or int(fit_block_size) not in {1, 2, 3}
        ):
            raise ValueError("fit_block_size must be 1, 2, or 3")
        self.fit_block_size = int(fit_block_size)
        if not isinstance(fit_n_iter, Integral) or int(fit_n_iter) < 1:
            raise ValueError("fit_n_iter must be a positive integer")
        self.fit_n_iter = int(fit_n_iter)
        if (
            not isinstance(fit_adaptive_sweeps, Integral)
            or int(fit_adaptive_sweeps) < 1
        ):
            raise ValueError("fit_adaptive_sweeps must be a positive integer")
        self.fit_adaptive_sweeps = int(fit_adaptive_sweeps)
        self.fit_min_iter = fit_min_iter
        self.fit_rtol = fit_rtol
        if not isinstance(fit_patience, Integral) or int(fit_patience) < 1:
            raise ValueError("fit_patience must be a positive integer")
        self.fit_patience = int(fit_patience)
        self.fit_init_strategy = (
            str(fit_init_strategy).strip().lower().replace("-", "_")
        )
        self.fit_init_rand_strength = float(fit_init_rand_strength)
        if (
            not np.isfinite(self.fit_init_rand_strength)
            or self.fit_init_rand_strength < 0.0
        ):
            raise ValueError(
                "fit_init_rand_strength must be finite and non-negative."
            )
        if isinstance(fit_init_seed, bool) or not isinstance(fit_init_seed, Integral):
            raise TypeError("fit_init_seed must be an integer")
        self.fit_init_seed = int(fit_init_seed)
        self.fit_sweep_sequence = fit_sweep_sequence
        self.fit_overlap_diagnostics = bool(fit_overlap_diagnostics)
        if compression_seed is not None:
            if isinstance(compression_seed, bool) or not isinstance(
                compression_seed, Integral
            ):
                raise TypeError("compression_seed must be an integer or None")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError("compression_seed must be non-negative")
        self.compression_seed = compression_seed
        if max_bond is not None:
            chi = max_bond
        self.chi = self._normalize_max_bond(chi)
        self.cutoff = cutoff
        self.cutoff_mode = cutoff_mode
        self.reduced = bool(reduced)
        self.info_c = info_c
        if max_operator_sites is _UNSET and max_operator_qubits is _UNSET:
            max_operator_sites = 12
        elif max_operator_sites is _UNSET:
            max_operator_sites = max_operator_qubits
        elif max_operator_qubits is not _UNSET:
            if max_operator_sites != max_operator_qubits:
                raise ValueError(
                    "max_operator_sites and max_operator_qubits must agree"
                )
        self.max_operator_sites = self._normalize_limit(max_operator_sites, "max_operator_sites")
        self.max_operator_qubits = self.max_operator_sites
        self.max_subtree_nodes = self._normalize_limit(max_subtree_nodes, "max_subtree_nodes")
        self.max_intermediate_bond = self._normalize_limit(
            max_intermediate_bond, "max_intermediate_bond"
        )
        self.record_history = bool(record_history)
        self.track_truncation = bool(track_truncation)
        self.track_infidelity = bool(track_infidelity)
        self.profile = bool(profile)
        self.profile_sync = bool(profile_sync)
        self.track_bond_diagnostics = bool(track_bond_diagnostics)
        self.inplace = bool(inplace)
        self.history = []
        self.infidelities = [0.0]
        self.norm_events = []
        self._norm_log_survival = 0.0
        self._last_local_fidelity = None
        self._last_local_infidelity = None
        self.normalizations = []
        self.profile_events = []
        self.fit_diagnostics = []
        self._last_fit_diagnostics = None
        self.state = state if self.inplace else state.copy()
        self.state.validate()
        self.backend_info()
        self.cutoff = self._normalize_cutoff(self.cutoff)
        self._sync_info()
        self._gate_stream = ()

        if gates is not None:
            self.set_gates(gates)
            if run:
                self.run()

    @classmethod
    def _normalize_mode(cls, mode):
        mode = str(mode).strip().lower()
        mode = cls._MODE_ALIASES.get(mode, mode)
        if mode == "fit" or mode in cls._DMRG_MODE_ALIASES:
            return "dmrg"
        if mode in {"dm", "sdc", "src", "zipup"}:
            return "direct"
        if mode not in {"direct", "sub_treepepo", "dmrg", "auto"}:
            raise ValueError(
                "mode must be 'direct', 'sub_treepepo', 'dmrg', or 'auto'"
            )
        return mode

    @staticmethod
    def _resolve_modes(mode, compression_mode):
        """Resolve operator routing and compression modes independently."""

        raw_mode = str(mode).strip().lower().replace("-", "_")
        compression_mode = _normalize_compression_mode(compression_mode)
        if raw_mode == "fit" or raw_mode in TreePepsOptimizer._DMRG_MODE_ALIASES:
            return "dmrg", compression_mode
        if raw_mode == "dm":
            if compression_mode not in {"direct", "dm"}:
                raise ValueError(
                    "mode='dm' cannot be combined with a different "
                    "compression_mode."
                )
            return "direct", "dm"
        if raw_mode in {"sdc", "src", "zipup"}:
            if compression_mode not in {"direct", raw_mode}:
                raise ValueError(
                    f"mode={raw_mode!r} cannot be combined with a different "
                    "compression_mode."
                )
            return "direct", raw_mode
        return (
            TreePepsOptimizer._normalize_mode(raw_mode),
            compression_mode,
        )

    def _progress_mode_name(self, mode=None, compression_mode=None):
        """Return the active short mode name shown by a replay bar."""

        explicit_mode = mode is not None
        raw_mode = self.mode if mode is None else str(mode)
        raw_mode = raw_mode.strip().lower().replace("-", "_")
        raw_mode = self._MODE_ALIASES.get(raw_mode, raw_mode)
        if raw_mode == "fit":
            raw_mode = "dmrg"
        if raw_mode in self._DMRG_MODE_ALIASES:
            return raw_mode
        if raw_mode == "dmrg":
            return self._dmrg_mode_alias or "dmrg"
        if raw_mode in {"dm", "sdc", "src", "zipup"}:
            return raw_mode
        if raw_mode in {"auto", "direct", "sub_treepepo"}:
            if compression_mode is not None:
                selected = _normalize_compression_mode(compression_mode)
            elif explicit_mode:
                selected = "direct"
            else:
                selected = self.compression_mode
            return selected
        return raw_mode

    @staticmethod
    def _normalize_max_bond(max_bond):
        if max_bond is None:
            return None
        if isinstance(max_bond, bool) or not isinstance(max_bond, Integral):
            raise TypeError("chi/max_bond must be a positive integer or None")
        max_bond = int(max_bond)
        if max_bond < 1:
            raise ValueError("chi/max_bond must be a positive integer or None")
        return max_bond

    @staticmethod
    def _normalize_limit(value, name):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer or None")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer or None")
        return value

    def _normalize_cutoff(self, cutoff):
        if cutoff is None:
            return 1e-10
        if isinstance(cutoff, str):
            if cutoff.strip().lower() == "auto":
                return dtype_auto_cutoff(self.backend_dtype)
            raise ValueError("cutoff must be a non-negative number or 'auto'")
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be a non-negative number or 'auto'")
        return cutoff

    @property
    def tn(self):
        """Compatibility alias for the live ``TreePeps`` state."""

        return self.state

    @tn.setter
    def tn(self, state):
        self.set_state(state)

    @property
    def p(self):
        """MPS/TTN name-parity alias for the live TreePeps state."""

        return self.state

    @p.setter
    def p(self, state):
        self.set_state(state)

    def set_state(self, state):
        """Replace the live state after validating its replay contract.

        The replacement must use the same tree plan as this optimizer and
        must be backend/device compatible with every queued payload.  Unless
        the optimizer was created with ``inplace=True``, the replacement is
        copied before it becomes live.  The queued stream is retained, while
        state-dependent reports are reset.
        """

        if not isinstance(state, TreePeps):
            raise TypeError("state must be a TreePeps")
        if plan_signature(state.plan) != plan_signature(self.plan):
            raise ValueError("state and optimizer must use the same tree plan")
        state.validate()
        self._validate_stream_backend(state=state)
        self.state = state if self.inplace else state.copy()
        self.history = []
        self.infidelities = [0.0]
        self.norm_events = []
        self._norm_log_survival = 0.0
        self._last_local_fidelity = None
        self._last_local_infidelity = None
        self.normalizations = []
        self.profile_events = []
        self.backend_info()
        self._sync_info()
        return self

    set_tn = set_state
    set_p = set_state

    @property
    def plan(self):
        return self.state.plan

    @property
    def n(self):
        """Number of physical sites, matching ``TreeOptimizer.n``."""

        return self.plan.size

    @property
    def nsites(self):
        """Number of physical lattice sites."""

        return self.plan.size

    @property
    def nqubits(self):
        """TTN name-parity alias for the number of physical sites."""

        return self.plan.size

    @property
    def shape(self):
        return self.plan.shape

    @property
    def ndim(self):
        return self.plan.ndim

    @property
    def root(self):
        return self.plan.root

    @property
    def top_arity(self):
        return self.state.top_arity

    @property
    def max_virtual_degree(self):
        return self.state.max_virtual_degree

    @property
    def max_tensor_rank(self):
        return self.state.max_tensor_rank

    def is_binary(self, *, allow_ternary_root=True):
        return self.state.is_binary(allow_ternary_root=allow_ternary_root)

    @property
    def center(self):
        return self.state.orthogonality_center

    @center.setter
    def center(self, site):
        self.state.orthogonality_center = site
        self._sync_info()

    @property
    def orthogonality_center(self):
        """Alias for :attr:`center` matching the TTN optimizer API."""

        return self.center

    @orthogonality_center.setter
    def orthogonality_center(self, site):
        self.center = site

    @property
    def canonical_region(self):
        return self.state.canonical_region

    @canonical_region.setter
    def canonical_region(self, region):
        self.state.canonical_region = region
        self._sync_info()

    def canonicalize(self, center=None, *, info_c=None, **canonize_opts):
        """Canonicalize the live state and return this optimizer.

        This is the optimizer-level spelling of ``TreePeps.canonicalize``.
        The optimizer's state and ``info_c`` remain the single source of truth;
        an explicitly supplied mapping is additionally updated for callers
        sharing metadata with another optimizer frontend.
        """

        metadata = self.info_c if info_c is None else info_c
        self.state.canonicalize(
            center,
            inplace=True,
            info_c=metadata,
            **canonize_opts,
        )
        self._sync_info()
        return self

    def canonicalize_(self, center=None, *, info_c=None, **canonize_opts):
        """In-place alias for :meth:`canonicalize`."""

        return self.canonicalize(
            center=center,
            info_c=info_c,
            **canonize_opts,
        )

    canonize = canonicalize_

    def _sync_info(self):
        if self.info_c is not None:
            self.state._sync_info_c(self.info_c)

    def validate(self, *, check_canonical=False, tol=1e-9):
        self.state.validate(check_canonical=check_canonical, tol=tol)
        return self

    def is_canonical_form(self, center=None, *, tol=1e-9):
        return self.state.is_canonical_form(center, tol=tol)

    def is_subtree_canonical_form(self, sites=None, *, span=False, tol=1e-9):
        return self.state.is_subtree_canonical_form(sites, span=span, tol=tol)

    def sync_canonicalization(self, center=None):
        """Rebuild canonical metadata after direct state-level mutation."""

        if center is None:
            center = self.plan.root
        self.state.invalidate_canonical_form()
        self.state.canonize_to(center, inplace=True)
        self._sync_info()
        return self.center

    def shift_orthogonality_center(self, site, *, absorb="right", **canonize_opts):
        """Move the state center along the unique tree path."""

        self.state.shift_orthogonality_center(
            site,
            absorb=absorb,
            info_c=self.info_c,
            **canonize_opts,
        )
        return self

    def canonize_subtree(self, sites, *, span=False, absorb="right", **canonize_opts):
        """Canonicalize around a connected TreePeps region in place."""

        self.state.canonize_subtree(
            sites,
            span=span,
            absorb=absorb,
            inplace=True,
            info_c=self.info_c,
            **canonize_opts,
        )
        return self

    def canonize_around_qubits(self, sites, **canonize_opts):
        """Canonicalize around the minimal tree span of physical sites."""

        return self.canonize_subtree(sites, span=True, **canonize_opts)

    def canonize_mps(self, state, where, *, info=None, **canonize_opts):
        """Compatibility canonicalization entry point for shared frontends.

        The method name follows ``TreeOptimizer``/``MpsOptimizer``. The
        supplied state must be this optimizer's live TreePeps; singleton
        supports leave a one-site center and multi-site supports leave their
        connected canonical region.
        """

        if state is not self.state:
            raise ValueError("state must be the optimizer's live TreePeps")
        if isinstance(where, Integral):
            sites = (int(where),)
        else:
            try:
                sites = tuple(where)
            except TypeError as exc:
                raise ValueError("where must be a site or iterable of sites") from exc
        if not sites:
            raise ValueError("where cannot be empty")
        sites = tuple(self.plan.resolve_site(site) for site in sites)
        if len(sites) == 1:
            self.state.canonize_to(
                sites[0], inplace=True, info_c=self.info_c, **canonize_opts
            )
            target = (sites[0], sites[0])
        else:
            self.canonize_subtree(sites, span=True, **canonize_opts)
            target = (min(sites), max(sites))
        if info is not None:
            if not hasattr(info, "__setitem__"):
                raise TypeError("info must be a mutable mapping when supplied")
            info["cur_orthog"] = (
                (self.center, self.center) if self.center is not None else target
            )
        return target

    def isometry_direction(self, site):
        """Return the live ``left_inds`` isometry direction at ``site``."""

        return self.state.isometry_direction(site)

    def isometry_map(self, region=None):
        """Return the state-owned outward-to-region isometry map."""

        return self.state.isometry_map(region)

    def can_skip_canonize(self, site0, site1, *, absorb="right"):
        """Whether the state metadata proves an edge QR is redundant."""

        return self.state.can_skip_canonize(site0, site1, absorb=absorb)

    def validate_isometry_metadata(self, region=None):
        """Validate live ``left_inds`` metadata against the state region."""

        self.state.validate_isometry_metadata(region)
        return self

    def layout_report(self):
        """Return immutable geometry and live canonical-layout diagnostics."""

        return {
            "shape": self.plan.shape,
            "ndim": self.plan.ndim,
            "n_sites": self.plan.size,
            "root": self.plan.root,
            "root_coordinate": self.plan.coordinate(self.plan.root),
            "order": self.plan.order,
            "tree_order": self.plan.tree_order,
            "map_mode": self.plan.map_mode,
            "coarse_grain": self.plan.coarse_grain,
            "topology": self.plan.topology,
            "tree_edges": self.plan.tree_edges,
            "max_virtual_degree": self.plan.max_virtual_degree,
            "max_degree": self.plan.max_degree,
            "max_tensor_rank": self.plan.max_tensor_rank,
            "center": self.center,
            "canonical_region": self.canonical_region,
        }

    @classmethod
    def find_tree_layout(
        cls,
        geometry,
        interactions=None,
        *,
        supports=None,
        gates=None,
        terms=None,
        max_virtual_degree=None,
        objective="hybrid",
        seed=0,
        max_iter=64,
        refine=True,
        order=None,
        map_mode=None,
        tree_order=None,
        seed_modes=None,
        tree_orders=None,
        root=None,
        topology=None,
        coarse_grain=None,
    ):
        """Return a workload-aware :class:`TreePepsPlan`.

        This convenience constructor mirrors ``TreeOptimizer.find_tree_layout``
        while retaining the lattice and rank-four branching TreePeps
        constraint. Pass ``topology="path"`` explicitly for an MPS-compatible
        control geometry. The returned plan can be passed directly as
        ``plan=`` or ``layout=`` to
        the state, TreePePO constructors, and this optimizer.
        """

        from .layout import TreePepsLayoutFinder

        if isinstance(geometry, TreePeps):
            geometry = geometry.plan
        finder = TreePepsLayoutFinder(
            geometry,
            interactions,
            supports=supports,
            gates=gates,
            terms=terms,
            max_virtual_degree=max_virtual_degree,
            objective=objective,
            seed=seed,
            max_iter=max_iter,
            order=order,
            map_mode=map_mode,
            tree_order=tree_order,
            seed_modes=seed_modes,
            tree_orders=tree_orders,
            root=root,
            topology=topology,
            coarse_grain=coarse_grain,
        )
        return finder.run(refine=refine)

    def optimize_layout(
        self,
        *,
        interactions=None,
        finder=None,
        objective="hybrid",
        max_iter=64,
        refine=True,
        seed=0,
        install=False,
    ):
        """Select a workload-aware layout without replaying the state.

        Layout search is non-mutating by default. A live, generally entangled
        TreePeps cannot be moved to a different spanning tree by changing
        metadata, so ``install=True`` is accepted only when the selected plan
        is structurally identical to the current one; callers must explicitly
        remount tensor data for a different plan.
        """

        from .layout import TreePepsLayoutFinder

        if finder is not None and interactions is not None:
            raise TypeError("pass either finder= or interactions=, not both")
        if finder is None:
            if interactions is None:
                interactions = tuple(
                    (entry[1], entry[2])
                    if entry[0] == "gate"
                    else entry[1]
                    for entry in self._gate_stream
                )
            finder = TreePepsLayoutFinder(
                self.plan,
                interactions=interactions,
                objective=objective,
                max_iter=max_iter,
                seed=seed,
            )
        if not isinstance(finder, TreePepsLayoutFinder):
            raise TypeError("finder must be a TreePepsLayoutFinder")
        selected = finder.run(refine=refine)
        same_plan = plan_signature(selected) == plan_signature(self.plan)
        if install and not same_plan:
            raise ValueError(
                "cannot install a different TreePeps layout into a live state; "
                "explicitly remount the tensors on the selected plan"
            )
        return {
            "plan": selected,
            "finder": finder,
            "report": finder.report,
            "installed": bool(install and same_plan),
        }

    def select_layout_for_compression(self, **kwargs):
        """Compatibility wrapper selecting a span-minimizing layout."""

        kwargs.setdefault("objective", "span")
        return self.optimize_layout(**kwargs)

    @staticmethod
    def _state_backend_like_for(state):
        """Return representative raw tensor data from a tree state."""

        tensor_map = getattr(state, "tensor_map", None)
        if tensor_map:
            return next(iter(tensor_map.values())).data
        return None

    @classmethod
    def _state_backend_signature_for(cls, state):
        like = cls._state_backend_like_for(state)
        if like is None:
            return None
        return infer_backend_signature(like)

    @staticmethod
    def _state_backend_info_for(state):
        return backend_infer(state)

    @staticmethod
    def _payload_backend_values(payload):
        """Return raw array values held by an operator or tensor payload."""

        tensor_map = getattr(payload, "tensor_map", None)
        if tensor_map is not None:
            return tuple(
                tensor.data
                for tensor in tensor_map.values()
                if getattr(tensor, "data", None) is not None
            )
        data = getattr(payload, "data", None)
        if data is not None and hasattr(data, "shape"):
            return (data,)
        return ()

    @classmethod
    def _payload_backend_signature(cls, payload):
        values = cls._payload_backend_values(payload)
        if not values:
            return None
        backend_infer(payload)
        return infer_backend_signature(values[0])

    @staticmethod
    def _gate_backend_signature(gate):
        candidate = gate.to_dense() if hasattr(gate, "to_dense") else gate
        candidate = getattr(candidate, "data", candidate)
        if not hasattr(candidate, "shape") or not hasattr(candidate, "dtype"):
            return None
        return infer_backend_signature(candidate)

    @staticmethod
    def _backend_mismatch_hint(target_signature):
        if target_signature[0] == "symmray":
            return (
                "Native Symmray states require native Symmray operator payloads "
                "with matching charge metadata."
            )
        return (
            "Convert the payload to the state's backend/device before queuing "
            "it; implicit backend transfers are not performed."
        )

    @classmethod
    def _backend_payload_compatible(cls, source_signature, target_signature):
        if source_signature is None:
            # Python lists and tuples are convenience dense inputs. They are
            # materialized by the current TreePepo factory as NumPy arrays.
            return target_signature[0] == "numpy"
        return backend_signatures_compatible(source_signature, target_signature)

    @staticmethod
    def _backend_signatures_compatible(source_signature, target_signature):
        return backend_signatures_compatible(source_signature, target_signature)

    def _validate_backend_payload(self, payload, *, state=None, path="payload"):
        """Validate one dense/operator payload against a candidate state."""

        target_signature = self._state_backend_signature_for(
            self.state if state is None else state
        )
        if target_signature is None:
            return
        if isinstance(payload, TreePepo):
            source_signature = self._payload_backend_signature(payload)
        else:
            source_signature = self._gate_backend_signature(payload)
        if self._backend_payload_compatible(source_signature, target_signature):
            return
        raise TypeError(
            "TreePepsOptimizer requires every gate and TreePepo payload to "
            "match the TreePeps backend/device and required dtype "
            f"{target_signature!r}; {path} has {source_signature!r}. "
            f"{self._backend_mismatch_hint(target_signature)}"
        )

    def backend_info(self):
        """Return backend, dtype, and device metadata for the live state."""

        info = self._state_backend_info_for(self.state)
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        return info

    def to_backend(self, payload):
        """Convert a dense payload or TreePeps operator to the live backend.

        Stream insertion remains strict and rejects mismatches. This explicit
        helper mirrors ``TreeOptimizer.to_backend`` so callers can prepare a
        gate or operator deliberately before queuing it.
        """

        like = self._state_backend_like_for(self.state)
        if like is None:
            return payload

        def convert(array):
            if not hasattr(array, "shape") or not hasattr(array, "dtype"):
                array = np.asarray(array)
            source_signature = infer_backend_signature(array)
            target_signature = infer_backend_signature(like)
            if backend_signatures_compatible(source_signature, target_signature):
                return array
            if target_signature[0] == "numpy":
                return np.asarray(array)
            return ar.do("array", array, like=like)

        if isinstance(payload, TreeSubPepo):
            operator = self.to_backend(payload.operator)
            return TreeSubPepo(operator, payload.support, span=payload.span)
        if isinstance(payload, TreePepo):
            operator = payload.copy()
            for tensor in operator.tensor_map.values():
                tensor.modify(data=convert(tensor.data))
            return operator
        return convert(payload)

    @staticmethod
    def _normalize_event_name(name):
        return str(name).strip().lower().replace("-", "_")

    @staticmethod
    def _looks_like_site_selector(value):
        if isinstance(value, Integral):
            return True
        if isinstance(value, (str, bytes)):
            return False
        try:
            values = tuple(value)
        except TypeError:
            return False
        return bool(values) and all(isinstance(site, Integral) for site in values)

    @classmethod
    def _looks_like_single_stream_entry(cls, gates):
        if not isinstance(gates, (tuple, list)) or not gates:
            return False
        if isinstance(gates[0], str):
            return True
        return len(gates) == 2 and cls._looks_like_site_selector(gates[1])

    @classmethod
    def _as_stream_entries(cls, gates):
        """Materialize a stream while preserving a single raw gate entry."""

        if gates is None:
            return []
        if isinstance(gates, (TreePepo, TreeSubPepo, Mapping)):
            return [gates]
        if cls._looks_like_single_stream_entry(gates):
            return [gates]
        if isinstance(gates, (str, bytes)):
            raise TypeError("a gate stream must contain structured entries")
        try:
            return list(gates)
        except TypeError as exc:
            raise TypeError(
                "gates must be a gate entry or an iterable of entries"
            ) from exc

    @staticmethod
    def _event_support(where):
        if isinstance(where, Integral):
            return int(where)
        try:
            return tuple(where)
        except TypeError as exc:
            raise TypeError("where must be a site or iterable of sites") from exc

    @classmethod
    def _normalize_stream_event_name(cls, name):
        normalized = cls._normalize_event_name(name)
        try:
            return cls._STREAM_EVENT_ALIASES[normalized]
        except KeyError as exc:
            raise ValueError(
                "unknown TreePepsOptimizer stream event "
                f"{name!r}; expected 'gate', 'tree_pepo', or 'sub_treepepo'"
            ) from exc

    def _normalize_stream_entry(self, entry):
        """Normalize one public stream entry to an immutable event tuple."""

        if isinstance(entry, TreeSubPepo):
            if entry.plan_signature != plan_signature(self.plan):
                raise ValueError("operator and optimizer must use the same tree plan")
            return ("sub_treepepo", entry)
        if isinstance(entry, TreePepo):
            if plan_signature(entry.plan) != plan_signature(self.plan):
                raise ValueError("operator and optimizer must use the same tree plan")
            return ("tree_pepo", entry)

        if isinstance(entry, Mapping):
            name = entry.get("kind", entry.get("type", entry.get("event")))
            if name is None:
                raise ValueError("mapping gate entries require a 'kind' field")
            name = self._normalize_stream_event_name(name)
            if name == "gate":
                if "gate" not in entry or "where" not in entry:
                    raise ValueError("a 'gate' event requires 'gate' and 'where'")
                return (
                    "gate",
                    entry["gate"],
                    self._normalize_support(entry["where"]),
                )
            operator = entry.get("operator", entry.get("pepo"))
            if name == "tree_pepo" and isinstance(operator, TreePepo):
                return self._normalize_stream_entry(operator)
            if name == "sub_treepepo" and isinstance(operator, TreeSubPepo):
                return self._normalize_stream_entry(operator)
            raise TypeError(
                f"{name!r} events require a matching TreePepo or TreeSubPepo"
            )

        if isinstance(entry, (tuple, list)) and entry:
            if isinstance(entry[0], str):
                name = self._normalize_stream_event_name(entry[0])
                if name == "gate":
                    if len(entry) != 3:
                        raise ValueError("a 'gate' event must be ('gate', gate, where)")
                    return (
                        "gate",
                        entry[1],
                        self._normalize_support(entry[2]),
                    )
                if len(entry) != 2:
                    raise ValueError(
                        f"a {name!r} event must contain exactly one operator"
                    )
                return self._normalize_stream_entry_for_kind(name, entry[1])
            if len(entry) == 2:
                return (
                    "gate",
                    entry[0],
                    self._normalize_support(entry[1]),
                )

        raise ValueError(
            "gate entries must be (gate, support), ('gate', gate, support), "
            "TreePepo, or TreeSubPepo"
        )

    def _normalize_stream_entry_for_kind(self, name, operator):
        if name == "tree_pepo" and isinstance(operator, TreePepo):
            return self._normalize_stream_entry(operator)
        if name == "sub_treepepo" and isinstance(operator, TreeSubPepo):
            return self._normalize_stream_entry(operator)
        raise TypeError(
            f"{name!r} events require a matching TreePepo or TreeSubPepo"
        )

    def _normalize_stream(self, gates):
        return tuple(
            self._normalize_stream_entry(entry)
            for entry in self._as_stream_entries(gates)
        )

    def _validate_stream_backend(self, *, state=None, stream=None):
        stream = self._gate_stream if stream is None else stream
        for index, entry in enumerate(stream):
            name = entry[0]
            if name == "gate":
                self._validate_backend_payload(
                    entry[1], state=state, path=f"stream[{index}].gate"
                )
            else:
                self._validate_backend_payload(
                    entry[1].operator if name == "sub_treepepo" else entry[1],
                    state=state,
                    path=f"stream[{index}].operator",
                )

    @staticmethod
    def gate_event(gate, where):
        """Build a tagged dense-gate stream entry."""

        return ("gate", gate, TreePepsOptimizer._event_support(where))

    @staticmethod
    def tree_pepo_event(operator):
        """Build a tagged complete :class:`TreePepo` stream entry."""

        if not isinstance(operator, TreePepo):
            raise TypeError("tree_pepo_event requires a TreePepo")
        return ("tree_pepo", operator)

    @staticmethod
    def sub_treepepo_event(operator):
        """Build a tagged complete :class:`TreeSubPepo` stream entry."""

        if not isinstance(operator, TreeSubPepo):
            raise TypeError("sub_treepepo_event requires a TreeSubPepo")
        return ("sub_treepepo", operator)

    pepo_event = tree_pepo_event
    tree_pepsmpo_event = tree_pepo_event
    tree_peps_mpo_event = tree_pepo_event
    treepepsmpo_event = tree_pepo_event
    subtree_pepo_event = sub_treepepo_event
    sub_tree_pepo_event = sub_treepepo_event
    sub_treepepsmpo_event = sub_treepepo_event
    sub_tree_peps_mpo_event = sub_treepepo_event
    subtreepepsmpo_event = sub_treepepo_event

    @property
    def gate_stream(self):
        """Return the immutable stream currently owned by the optimizer."""

        return self._gate_stream

    @property
    def gates(self):
        """Compatibility alias for :attr:`gate_stream`."""

        return self._gate_stream

    def set_gates(self, gates):
        """Replace the queued gate/operator stream without executing it."""

        stream = self._normalize_stream(gates)
        self._validate_stream_backend(stream=stream)
        self._gate_stream = stream
        return self

    def add_gates(self, gates):
        """Append entries to the queued gate/operator stream."""

        additions = self._normalize_stream(gates)
        stream = self._gate_stream + additions
        self._validate_stream_backend(stream=stream)
        self._gate_stream = stream
        return self

    def _normalize_support(self, support):
        if isinstance(support, Integral):
            support = (support,)
        try:
            support = tuple(self.plan.resolve_site(site) for site in support)
        except TypeError as exc:
            raise TypeError("support must be a site or iterable of sites") from exc
        if not support:
            raise ValueError("support cannot be empty")
        if len(set(support)) != len(support):
            raise ValueError("support must contain distinct sites")
        if self.max_operator_sites is not None and len(support) > self.max_operator_sites:
            raise ValueError(
                f"operator support has {len(support)} sites, exceeding "
                f"max_operator_sites={self.max_operator_sites}"
            )
        return support

    def _normalize_span(self, span):
        span = frozenset(self.plan.resolve_site(site) for site in span)
        if not span or not self.plan.is_connected(span):
            raise ValueError("operator span must be a non-empty connected subtree")
        if self.max_subtree_nodes is not None and len(span) > self.max_subtree_nodes:
            raise ValueError(
                f"operator span has {len(span)} nodes, exceeding "
                f"max_subtree_nodes={self.max_subtree_nodes}"
            )
        return span

    def _physical_dims(self):
        return {
            q: int(self.state.node_tensor(q).ind_size(self.state.site_ind_1d(q)))
            for q in self.state.sites
        }

    def _state_dtype(self):
        data = self.state.node_tensor(self.plan.root).data
        try:
            return np.asarray(data).dtype
        except (TypeError, ValueError):
            return np.dtype(ar.get_dtype_name(data))

    def _gate_dtype(self, gate):
        candidate = gate.to_dense() if hasattr(gate, "to_dense") else gate
        candidate = getattr(candidate, "data", candidate)
        try:
            gate_dtype = np.asarray(candidate).dtype
        except (TypeError, ValueError):
            gate_dtype = np.dtype(ar.get_dtype_name(candidate))
        return np.result_type(self._state_dtype(), gate_dtype)

    def _region_center(self, region, preferred=None):
        region = frozenset(region)
        if preferred in region:
            return preferred
        return min(
            region,
            key=lambda q: (
                max(len(self.plan.path(q, other)) for other in region),
                sum(len(self.plan.path(q, other)) for other in region),
                q,
            ),
        )

    def _region_edges(self, region):
        region = frozenset(region)
        return tuple(
            edge for edge in self.plan.tree_edges if edge[0] in region and edge[1] in region
        )

    @staticmethod
    def _bond_sizes(state, edges):
        return {
            tuple(edge): int(state.node_tensor(edge[0]).ind_size(state.bond(*edge)))
            for edge in edges
        }

    def _prepare_span(self, span):
        span = frozenset(span)
        if self.state.canonical_region != span or not self.state.is_subtree_canonical_form(span):
            self.state.canonize_subtree(span, inplace=True, info_c=self.info_c)

    @staticmethod
    def _format_progress_scalar(value):
        """Format a fidelity value for the replay progress bar."""

        if value is None:
            return "-"
        return f"{float(value):.6f}"

    def _cumulative_fidelity(self):
        """Return cumulative fidelity measured from retained norms."""

        if self._norm_log_survival == -math.inf:
            return 0.0
        return float(math.exp(self._norm_log_survival))

    def _cumulative_infidelity(self):
        """Return cumulative infidelity using stable ``expm1``."""

        if self._norm_log_survival == -math.inf:
            return 1.0
        return float(-math.expm1(self._norm_log_survival))

    def _record_norm_fidelity(
        self,
        norm_before,
        norm_after,
        *,
        track_norm,
    ):
        """Record one update's local and cumulative retained fidelity."""

        if not track_norm or norm_before is None or norm_after is None:
            self._last_local_fidelity = None
            self._last_local_infidelity = None
            return {
                "valid": False,
                "expected_norm": None,
                "observed_norm": None,
                "fidelity_raw": None,
                "local_fidelity": None,
                "local_infidelity": None,
                "cumulative_fidelity": None,
                "cumulative_infidelity": None,
                "cumulative_compression_fidelity": None,
                "cumulative_compression_infidelity": None,
            }

        observed_norm = float(abs(norm_after))
        expected_norm = float(abs(norm_before))
        if (
            expected_norm <= 0.0
            or not np.isfinite(expected_norm)
            or not np.isfinite(observed_norm)
        ):
            self._last_local_fidelity = None
            self._last_local_infidelity = None
            return {
                "valid": False,
                "expected_norm": None,
                "observed_norm": None,
                "fidelity_raw": None,
                "local_fidelity": None,
                "local_infidelity": None,
                "cumulative_fidelity": None,
                "cumulative_infidelity": None,
                "cumulative_compression_fidelity": None,
                "cumulative_compression_infidelity": None,
            }

        raw = (observed_norm / expected_norm) ** 2
        log_local = log_fidelity_from_norms(observed_norm, expected_norm)
        local_fidelity = fidelity_from_log(log_local)
        local_infidelity = infidelity_from_log(log_local)
        if local_fidelity == 0.0:
            self._norm_log_survival = -math.inf
        elif math.isfinite(self._norm_log_survival):
            self._norm_log_survival += math.log(local_fidelity)
        cumulative_fidelity = self._cumulative_fidelity()
        cumulative_infidelity = self._cumulative_infidelity()
        self._last_local_fidelity = float(local_fidelity)
        self._last_local_infidelity = local_infidelity
        return {
            "valid": True,
            "expected_norm": expected_norm,
            "observed_norm": observed_norm,
            "fidelity_raw": float(raw),
            "local_fidelity": float(local_fidelity),
            "local_infidelity": local_infidelity,
            "cumulative_fidelity": cumulative_fidelity,
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": cumulative_fidelity,
            "cumulative_compression_infidelity": cumulative_infidelity,
        }

    @staticmethod
    def _normalize_fit_init_strategy(strategy):
        """Normalize a TreeFIT disposable initial-guess policy."""

        strategy = str(strategy).strip().lower().replace("-", "_")
        if strategy == "auto":
            strategy = "guess_src"
        if strategy in {"direct", "random", "random_expand"}:
            return strategy
        if strategy.startswith("guess_"):
            method = strategy[6:]
            if method in {"direct", "dm", "sdc", "src", "zipup"}:
                return strategy
        raise ValueError(
            "fit_init_strategy must be one of 'auto', 'direct', 'random', "
            "'random_expand', or 'guess-<method>'"
        )

    def _fit_block_size(self):
        """Resolve a named DMRG mode to its requested warm-up block size."""

        if self._dmrg_mode_alias is not None:
            # Match MpsOptimizer: ``dmrg1`` is one-site DMRG with a bounded
            # two-site growth warm-up, while dmrg2 and dmrg3 select the
            # corresponding larger warm-up block.
            return {
                "dmrg1": 2,
                "dmrg2": 2,
                "dmrg3": 3,
            }[self._dmrg_mode_alias]
        return self.fit_block_size

    def _tree_fit_initial_guess(
        self,
        operator,
        span,
        *,
        target,
        max_bond,
        cutoff,
        cutoff_mode,
        compression_layout,
    ):
        """Build a disposable TreeFIT guess without changing the live state."""

        strategy = self._normalize_fit_init_strategy(self.fit_init_strategy)
        if strategy in {"random", "random_expand"}:
            guess, random_info = _randomize_tree_guess(
                self.state,
                span,
                target=target,
                max_bond=max_bond,
                strength=self.fit_init_rand_strength,
                expand=strategy == "random_expand",
                seed=self.fit_init_seed,
            )
            return guess, strategy, random_info
        if strategy == "direct":
            return self.state.copy(), strategy, None
        method = strategy[6:]
        if method == "direct":
            return self.state.copy(), strategy, None
        if method == "zipup" and not self.plan.is_mps_topology:
            raise NotImplementedError(
                "fit_init_strategy='guess-zipup' requires a path TreePeps."
            )
        guess = operator.apply_to(
            self.state,
            compress=True,
            center=self._region_center(span),
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            reduced=self.reduced,
            compression_mode=method,
            compression_seed=self.fit_init_seed,
            compression_layout=compression_layout,
            _active_sites=span,
        )
        return guess, strategy, None

    def _apply_operator_fit(
        self,
        operator,
        span,
        *,
        max_bond,
        cutoff,
        cutoff_mode,
        compression_mode,
        compression_layout,
    ):
        """Fit an exact disposable PEPO target with the tree-native FIT kernel."""

        target = _build_layered_operator_state_target(self.state, operator)
        guess, strategy, random_info = self._tree_fit_initial_guess(
            operator,
            span,
            target=target,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression_layout=compression_layout,
        )
        split_method = compression_mode
        if split_method == "zipup":
            split_method = "direct"
        block_size = self._fit_block_size()
        fit = TreeFIT(
            target,
            guess,
            max_bond=max_bond,
            cutoffs=cutoff,
            cutoff_mode=cutoff_mode,
            split_method=split_method,
            split_seed=(
                self.compression_seed
                if self.compression_seed is not None else self.fit_init_seed
            ),
            inplace=True,
        )
        active_block_size = min(block_size, len(span))
        if (
            self._dmrg_mode_alias == "dmrg1"
            and block_size == 2
            and fit._active_bonds_at_rank_targets(span, state=self.state)
        ):
            active_block_size = 1
        if (
            self._dmrg_mode_alias == "dmrg1"
            and active_block_size == 2
            and len(span) > 2
            and not fit._active_bonds_at_rank_targets(span, state=self.state)
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
        fit.run_gate(
            span,
            n_iter=self.fit_n_iter,
            block_size=active_block_size,
            sweep_sequence=self.fit_sweep_sequence,
            min_iter=self.fit_min_iter,
            rtol=self.fit_rtol,
            patience=self.fit_patience,
            adaptive_block_sweeps=adaptive_sweeps,
            adaptive_until_rank=(
                self._dmrg_mode_alias is None
                and not (
                    active_block_size in {2, 3}
                    and len(span) > active_block_size
                )
            ),
        )
        diagnostics = fit.fit_diagnostics(
            overlap=self.fit_overlap_diagnostics,
        )
        diagnostics.update(
            {
                "backend": "tree_fit",
                "fit_init_strategy": strategy,
                "fit_init_strategy_requested": self.fit_init_strategy,
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
                "guess_backend": "tree_pepo" if strategy.startswith("guess_") else None,
                "target_layout": fit.target_layout,
            }
        )
        self._last_fit_diagnostics = deepcopy(diagnostics)
        self.fit_diagnostics.append(deepcopy(diagnostics))
        return fit.p, target, diagnostics

    def _apply_operator(
        self,
        operator,
        support,
        span,
        *,
        mode,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
        compression_layout=None,
        renormalize=False,
        track_norm=True,
    ):
        if not isinstance(operator, TreePepo):
            raise TypeError("operator must be a TreePepo")
        if plan_signature(operator.plan) != plan_signature(self.plan):
            raise ValueError("operator and optimizer must use the same tree plan")
        self._validate_backend_payload(operator, path="operator")
        support = self._normalize_support(support)
        span = self._normalize_span(span)
        if operator.operator_span is not None and not operator.operator_span.issubset(span):
            span = self._normalize_span(operator.operator_span)
        operator.validate()

        max_bond = self.chi if max_bond is _UNSET else self._normalize_max_bond(max_bond)
        cutoff = self.cutoff if cutoff is _UNSET else self._normalize_cutoff(cutoff)
        if cutoff_mode is None:
            cutoff_mode = self.cutoff_mode
        if compression_mode is None:
            compression_mode = self.compression_mode
        compression_mode = _normalize_compression_mode(compression_mode)
        if compression_layout is None:
            compression_layout = self.compression_layout
        compression_layout = _normalize_compression_layout(compression_layout)
        started = perf_counter() if self.profile else None
        center_before = self.center
        canonical_region_before = self.state.canonical_region
        norm_before = self.norm() if track_norm else None
        edges = self._region_edges(span)
        before_bonds = self._bond_sizes(self.state, edges)
        fit_diagnostics = None
        fit_target = None

        # The state is canonical around the active region before the complete
        # PEPO is applied.  This is a fast metadata-aware move when possible.
        self._prepare_span(span)
        use_tree_fit = bool(compress and mode == "dmrg")
        if use_tree_fit:
            result, fit_target, fit_diagnostics = self._apply_operator_fit(
                operator,
                span,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                compression_layout=compression_layout,
            )
            uncompressed_bonds = _layered_target_bond_sizes(
                fit_target,
                self.state,
                edges,
            )
            transient_max_bond = max(uncompressed_bonds.values(), default=1)
            use_two_layer = False
        else:
            use_two_layer = (
                compress
                and compression_layout != "fused"
                and self.plan.is_mps_topology
                and compression_mode in {"direct", "dm", "sdc", "src", "zipup"}
            )
        if compression_layout == "two_layer" and not use_two_layer and not use_tree_fit:
            if not self.plan.is_mps_topology:
                raise NotImplementedError(
                    "compression_layout='two_layer' requires a path "
                    "TreePeps topology."
                )
            raise ValueError(
                "compression_layout='two_layer' requires compress=True and "
                "a supported compression_mode."
            )
        if use_tree_fit:
            # TreeFIT already produced the bounded-bond result from the exact
            # disposable target above.
            pass
        elif use_two_layer:
            # Fusing the same operator and state tensors would produce the
            # transient dimensions used by the diagnostics.  Compute them
            # without materializing that second network, then let Quimb
            # compress the original two-layer path directly.
            uncompressed_bonds = {
                tuple(edge): self.state.node_tensor(edge[0]).ind_size(
                    self.state.bond(*edge)
                ) * operator.node_tensor(edge[0]).ind_size(
                    operator.bond(*edge)
                )
                for edge in edges
            }
            result = operator.apply_to(
                self.state,
                compress=True,
                center=self._region_center(span, preferred=center),
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                reduced=self.reduced,
                compression_mode=compression_mode,
                compression_seed=self.compression_seed,
                compression_layout="two_layer",
                info_c=self.info_c,
                _active_sites=span,
            )
        else:
            result = operator.apply_to(
                self.state,
                compress=False,
                _active_sites=span,
            )
            result._canonical_region = frozenset(span)
            result._set_isometry_metadata_from_region(span)
            result.validate(check_canonical=True)
            uncompressed_bonds = self._bond_sizes(result, edges)
            transient_max_bond = max(uncompressed_bonds.values(), default=1)

        if compress and not use_two_layer and not use_tree_fit:
            center = self._region_center(span, preferred=center)
            result.compress_subtree(
                span,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                compression_seed=self.compression_seed,
                reduced=self.reduced,
                inplace=True,
                info_c=self.info_c,
            )
        norm_after_unscaled = (
            float(abs(np.asarray(result.norm()))) if track_norm else None
        )
        if renormalize:
            norm_before_normalize = norm_after_unscaled
            result.normalize()
        else:
            norm_before_normalize = None
        self.state = result
        self._sync_info()
        if renormalize:
            self.normalizations.append(
                {
                    "step": len(self.normalizations) + 1,
                    "norm_before": norm_before_normalize,
                    "norm_after": self.norm(),
                    "source": "operator_update",
                }
            )

        norm_after = self.norm() if track_norm else None
        fidelity_report = self._record_norm_fidelity(
            norm_before,
            norm_after_unscaled,
            track_norm=track_norm,
        )

        after_bonds = self._bond_sizes(self.state, edges)
        truncated = bool(
            compress
            and any(
                after_bonds[edge] < uncompressed_bonds[edge]
                for edge in before_bonds
            )
        )
        report = {
            "step": len(self.history) + 1,
            "mode": mode,
            "compression_mode": compression_mode,
            "compression_layout": compression_layout,
            "support": tuple(support),
            "span": tuple(sorted(span)),
            "path": (self.plan.path(support[0], support[1]) if len(support) == 2 else None),
            "center_before": center_before,
            "center_after": self.center,
            "before_bonds": before_bonds,
            "after_bonds": after_bonds,
            "uncompressed_bonds": uncompressed_bonds,
            "max_bond": max_bond,
            "cutoff": cutoff,
            "compressed": bool(compress),
            "truncated": truncated,
            "compression_scope": (
                "fit" if use_tree_fit else "span" if compress else "none"
            ),
            "backend": "tree_fit" if use_tree_fit else "compression",
            "touched_edges": tuple(edges),
            "canonical_region_before": canonical_region_before,
            "canonical_region_after": self.state.canonical_region,
            "norm_before": norm_before,
            "norm_after": norm_after,
            "norm_after_unscaled": norm_after_unscaled,
            "norm_ratio": (
                None
                if norm_before in (None, 0.0) or norm_after_unscaled is None
                else norm_after_unscaled / norm_before
            ),
            "renormalized": bool(renormalize),
            "norm_before_normalize": norm_before_normalize,
            "track_norm": bool(track_norm),
            "track_truncation": bool(self.track_truncation),
            **fidelity_report,
        }
        if fit_diagnostics is not None:
            report["fit_diagnostics"] = fit_diagnostics
        if self.track_bond_diagnostics:
            report.update(
                {
                    "live_max_bond_after": self.state.max_bond(),
                    "transient_max_bond": transient_max_bond,
                    "transient_exceeds_chi": bool(
                        self.chi is not None and transient_max_bond > self.chi
                    ),
                }
            )
        if started is not None:
            self.profile_events.append(
                {
                    "kind": "update",
                    "mode": mode,
                    "support": tuple(support),
                    "span": tuple(sorted(span)),
                    "seconds": perf_counter() - started,
                }
            )
        if track_norm:
            self.norm_events.append(
                {
                    "kind": "compression",
                    "mode": mode,
                    "support": tuple(support),
                    "span": tuple(sorted(span)),
                    "path": report["path"],
                    "norm_before": norm_before,
                    "norm_after": norm_after,
                    "norm_after_unscaled": norm_after_unscaled,
                    "norm_ratio": report["norm_ratio"],
                    "compressed": bool(compress),
                    "compression_scope": report["compression_scope"],
                    "touched_edges": tuple(edges),
                    "canonical_region_before": canonical_region_before,
                    "canonical_region_after": self.state.canonical_region,
                    "renormalized": bool(renormalize),
                    "norm_before_normalize": norm_before_normalize,
                    "track_norm": bool(track_norm),
                    **fidelity_report,
                }
            )
        if self.record_history:
            self.history.append(report)
        return self

    def apply_gate(
        self,
        gate,
        where,
        *,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
        compression_layout=None,
        renormalize=False,
        track_norm=True,
        _mode=None,
    ):
        """Apply a dense one- or multi-site gate in direct tree mode."""

        if _mode is None:
            route_mode = self.mode
        else:
            route_mode, shorthand_compression = self._resolve_modes(
                _mode,
                self.compression_mode
                if compression_mode is None
                else compression_mode,
            )
            if compression_mode is None:
                compression_mode = shorthand_compression
        if route_mode == "sub_treepepo":
            raise ValueError("mode='sub_treepepo' requires a TreeSubPepo operator")
        support = self._normalize_support(where)
        self._validate_backend_payload(gate, path="gate")
        operator = TreePepo.from_operator(
            self.plan,
            ar.to_numpy(gate),
            support,
            dims=self._physical_dims(),
            dtype=self._gate_dtype(gate),
            max_operator_sites=self.max_operator_sites,
        )
        # Dense TreePepo factorization currently uses host-side NumPy
        # decompositions. Convert the resulting operator back to the live
        # state's backend before the strict operator validation boundary.
        operator = self.to_backend(operator)
        suboperator = TreeSubPepo(operator, support)
        return self.apply_sub_treepepo(
            suboperator,
            _mode=route_mode,
            _report_mode=route_mode,
            compress=compress,
            center=center,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression_mode=(
                self.compression_mode
                if compression_mode is None else compression_mode
            ),
            compression_layout=(
                self.compression_layout
                if compression_layout is None else compression_layout
            ),
            renormalize=renormalize,
            track_norm=track_norm,
        )

    def apply_sub_treepepo(
        self,
        operator,
        *,
        _mode=None,
        _report_mode=None,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
        compression_layout=None,
        renormalize=False,
        track_norm=True,
    ):
        """Apply a complete ``TreeSubPepo`` without intermediate truncation."""

        if not isinstance(operator, TreeSubPepo):
            raise TypeError("apply_sub_treepepo requires a TreeSubPepo")
        if operator.plan_signature != plan_signature(self.plan):
            raise ValueError("operator and optimizer must use the same tree plan")
        route_mode = self.mode if _mode is None else _mode
        report_mode = (
            "dmrg" if route_mode == "dmrg"
            else "sub_treepepo" if _report_mode is None else _report_mode
        )
        return self._apply_operator(
            operator.operator,
            operator.support,
            operator.span,
            mode=report_mode,
            compress=compress,
            center=center,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression_mode=(
                self.compression_mode
                if compression_mode is None else compression_mode
            ),
            compression_layout=(
                self.compression_layout
                if compression_layout is None else compression_layout
            ),
            renormalize=renormalize,
            track_norm=track_norm,
        )

    apply_subtree_pepo = apply_sub_treepepo
    apply_sub_tree_pepo = apply_sub_treepepo
    # MPS-style compatibility spellings.  These all use the same
    # support/span-aware TreeSubPepo -> TreePepo application path.
    apply_sub_treepepsmpo = apply_sub_treepepo
    apply_sub_tree_peps_mpo = apply_sub_treepepo
    apply_subtreepepsmpo = apply_sub_treepepo

    def apply_subtree_operator(self, operator, where=None, **kwargs):
        """Apply a complete or support-declared TreePePO operator.

        ``TreeOptimizer`` uses this name for its subtree operator primitive.
        TreePeps accepts both an explicit ``TreeSubPepo`` and a complete
        ``TreePepo`` plus an optional support selector.
        """

        if isinstance(operator, TreeSubPepo):
            if where is not None:
                raise TypeError("where cannot be supplied with a TreeSubPepo")
            return self.apply_sub_treepepo(operator, **kwargs)
        if isinstance(operator, TreePepo):
            return self.apply(operator, where=where, **kwargs)
        raise TypeError(
            "apply_subtree_operator requires a TreePepo or TreeSubPepo"
        )

    def apply_1q(self, gate, site, **kwargs):
        """Apply a dense one-site gate using the common optimizer spelling."""

        return self.apply_gate(gate, (site,), **kwargs)

    def apply_2q(self, gate, site0, site1, **kwargs):
        """Apply a dense two-site gate along the tree geodesic."""

        return self.apply_gate(gate, (site0, site1), **kwargs)

    def apply_multi_site(self, gate, where, *more_sites, **kwargs):
        """Apply a dense gate on an arbitrary supported site tuple."""

        if more_sites:
            where = (where, *more_sites)
        return self.apply_gate(gate, where, **kwargs)

    apply_nq = apply_multi_site
    apply_multisite = apply_multi_site

    def apply_pepo(self, operator, **kwargs):
        """Apply a complete :class:`TreePepo` or :class:`TreeSubPepo`."""

        if not isinstance(operator, (TreePepo, TreeSubPepo)):
            raise TypeError("apply_pepo requires a TreePepo or TreeSubPepo")
        return self.apply(operator, **kwargs)

    def apply(
        self,
        operator,
        where=None,
        *,
        mode=None,
        compress=True,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
        compression_layout=None,
        renormalize=False,
        track_norm=True,
    ):
        """Dispatch a raw dense gate or explicit tree PEPO fragment."""

        selected_mode, selected_compression = self._resolve_modes(
            self.mode if mode is None else mode,
            self.compression_mode if compression_mode is None else compression_mode,
        )
        if isinstance(operator, TreeSubPepo):
            if where is not None:
                raise TypeError("where cannot be supplied with a TreeSubPepo")
            return self.apply_sub_treepepo(
                operator,
                _mode=selected_mode,
                compress=compress,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=selected_compression,
                compression_layout=(
                    self.compression_layout
                    if compression_layout is None else compression_layout
                ),
                renormalize=renormalize,
                track_norm=track_norm,
            )
        if isinstance(operator, TreePepo):
            if where is None:
                where = operator.operator_support
                if not where:
                    where = operator.sites
            return self._apply_operator(
                operator,
                where,
                operator.operator_span or operator.sites,
                mode=selected_mode,
                compress=compress,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=selected_compression,
                compression_layout=(
                    self.compression_layout
                    if compression_layout is None else compression_layout
                ),
                renormalize=renormalize,
                track_norm=track_norm,
            )
        if selected_mode == "sub_treepepo":
            raise TypeError("mode='sub_treepepo' requires a TreeSubPepo operator")
        if where is None:
            raise TypeError("where is required for a dense direct gate")
        return self.apply_gate(
            operator,
            where,
            compress=compress,
            center=center,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            compression_mode=selected_compression,
            compression_layout=(
                self.compression_layout
                if compression_layout is None else compression_layout
            ),
            renormalize=renormalize,
            track_norm=track_norm,
            _mode=selected_mode,
        )

    def _apply_stream_entry(
        self,
        entry,
        *,
        mode=None,
        compression_mode=None,
        compression_layout=None,
        renormalize=False,
        track_norm=True,
    ):
        name = entry[0]
        if name == "gate":
            self.apply_gate(
                entry[1],
                entry[2],
                _mode=mode,
                compression_mode=compression_mode,
                compression_layout=compression_layout,
                renormalize=renormalize,
                track_norm=track_norm,
            )
        elif name == "tree_pepo":
            self.apply(
                entry[1],
                mode=mode,
                compression_mode=compression_mode,
                compression_layout=compression_layout,
                renormalize=renormalize,
                track_norm=track_norm,
            )
        else:
            self.apply_sub_treepepo(
                entry[1],
                _mode=(
                    None if mode is None else self._resolve_modes(
                        mode,
                        self.compression_mode
                        if compression_mode is None else compression_mode,
                    )[0]
                ),
                compression_mode=compression_mode,
                compression_layout=compression_layout,
                renormalize=renormalize,
                track_norm=track_norm,
            )

    def run(
        self,
        gates=None,
        *,
        progbar=False,
        mode=None,
        compression_mode=None,
        compression_seed=None,
        compression_layout=None,
        non_unitary=False,
        normalize_every=False,
        normalize_final=False,
        normalize_eps=1e-15,
        track_infidelity=None,
    ):
        """Replay the queued stream, optionally replacing it first.

        The stream is persistent: calling ``run()`` again replays the same
        entries from the optimizer's current state.  Passing ``gates`` keeps
        the historical one-shot spelling and first installs that stream.

        ``normalize_every`` may be ``True`` (after every event) or a positive
        integer interval. ``non_unitary`` disables norm-ledger collection;
        explicit ``normalize_final`` still normalizes the represented state.
        ``mode`` and ``compression_mode`` overrides are normalized, stored on
        the optimizer, and used consistently for every queued event;
        ``dm``/``sdc``/``src``/``zipup`` are direct-routing compression
        shorthands.
        ``progbar=True`` shows the active event count, latest local fidelity,
        cumulative retained fidelity, and live maximum bond. The progress bar
        intentionally does not display the live state norm.
        """

        if gates is not None:
            self.set_gates(gates)
        raw_mode = self.mode if mode is None else str(mode)
        raw_mode = raw_mode.strip().lower().replace("-", "_")
        raw_mode = self._MODE_ALIASES.get(raw_mode, raw_mode)
        # Bare compression names own their compression choice, just as they
        # do at construction time. In particular, ``run(mode="sdc")`` must
        # replace a previously selected ``src``/``dm`` mode unless the caller
        # explicitly supplies a conflicting compression_mode (which is
        # rejected by _resolve_modes).
        if mode is not None and raw_mode in {"dm", "sdc", "src", "zipup"}:
            mode_compression = (
                "direct" if compression_mode is None else compression_mode
            )
        else:
            mode_compression = (
                self.compression_mode
                if compression_mode is None else compression_mode
            )
        selected_mode, selected_compression = self._resolve_modes(
            raw_mode,
            mode_compression,
        )
        if mode is not None:
            self._dmrg_mode_alias = (
                raw_mode if raw_mode in self._DMRG_MODE_ALIASES else None
            )
        self.mode = selected_mode
        self.compression_mode = selected_compression
        # Pass canonical, resolved values into each event below. This is
        # important for explicit TreeSubPepo entries: passing the shorthand
        # alone would normalize the route to ``direct`` and otherwise leave
        # the old self.compression_mode in effect.
        mode = selected_mode
        compression_mode = selected_compression
        if compression_seed is not None:
            if isinstance(compression_seed, bool) or not isinstance(
                compression_seed, Integral
            ):
                raise TypeError("compression_seed must be an integer or None")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError("compression_seed must be non-negative")
            self.compression_seed = compression_seed
        if compression_layout is not None:
            self.compression_layout = _normalize_compression_layout(
                compression_layout
            )
        if self.max_intermediate_bond is not None:
            self.preflight(max_intermediate_bond=self.max_intermediate_bond)
        if isinstance(normalize_every, bool):
            interval = 1 if normalize_every else None
        elif isinstance(normalize_every, Integral) and normalize_every > 0:
            interval = int(normalize_every)
        else:
            raise ValueError("normalize_every must be False, True, or a positive integer")
        if track_infidelity is None:
            track_norm = self.track_infidelity and not non_unitary
        else:
            track_norm = bool(track_infidelity) and not non_unitary
        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress_mode = self._progress_mode_name(
                mode=mode,
                compression_mode=compression_mode,
            )
            pbar = tqdm(
                total=len(self._gate_stream),
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

        two_qubit_count = 0
        multi_site_count = 0
        pepo_count = 0
        try:
            for index, entry in enumerate(self._gate_stream, start=1):
                renormalize = interval is not None and index % interval == 0
                self._apply_stream_entry(
                    entry,
                    mode=mode,
                    compression_mode=compression_mode,
                    compression_layout=compression_layout,
                    renormalize=renormalize,
                    track_norm=track_norm,
                )
                if pbar is not None:
                    kind = entry[0]
                    if kind == "gate":
                        support = tuple(entry[2])
                    elif kind == "sub_treepepo":
                        support = tuple(entry[1].support)
                    else:
                        operator = entry[1]
                        support = tuple(
                            operator.operator_support or operator.sites
                        )
                    if kind == "gate":
                        if len(support) == 2:
                            two_qubit_count += 1
                        elif len(support) > 2:
                            multi_site_count += 1
                    else:
                        pepo_count += 1

                    postfix = {
                        "2q": two_qubit_count,
                        "bnd": self.max_bond(),
                    }
                    if track_norm:
                        postfix["~F"] = self._format_progress_scalar(
                            self._cumulative_fidelity()
                        )
                    if multi_site_count:
                        postfix["kq"] = multi_site_count
                    if pepo_count:
                        postfix["pepo"] = pepo_count
                    pbar.set_postfix(postfix)
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()
        if normalize_final:
            self.normalize(eps=normalize_eps)
        return self

    def max_bond(self):
        """Return the largest live virtual bond dimension."""

        return self.state.max_bond()

    def show(
        self,
        *,
        bond_dims=True,
        node_ids=False,
        color=False,
        show_lower=False,
        show_upper=False,
    ):
        """Print the current state using the TreePeps coordinate schematic."""

        self.state.show(
            bond_dims=bond_dims,
            node_ids=node_ids,
            color=color,
            show_lower=show_lower,
            show_upper=show_upper,
        )

    def compress(
        self,
        sites=None,
        *,
        span=False,
        form=None,
        center=None,
        max_bond=_UNSET,
        cutoff=_UNSET,
        cutoff_mode=None,
        compression_mode=None,
        compression_seed=None,
        order="rank",
    ):
        """Compress the whole tree or only a selected gate-like span.

        Passing ``sites`` performs the same localized leaf-to-center sweep
        used after a gate update. It canonicalizes only the minimal requested
        span when ``span=True`` and leaves exterior virtual bonds untouched.
        With no sites, the complete tree is compressed toward ``center``.
        ``order="rank"`` uses live tree dimensions to choose the next branch;
        ``order="depth"`` retains the deterministic farthest-first schedule.
        """

        max_bond = self.chi if max_bond is _UNSET else self._normalize_max_bond(max_bond)
        cutoff = self.cutoff if cutoff is _UNSET else self._normalize_cutoff(cutoff)
        cutoff_mode = self.cutoff_mode if cutoff_mode is None else cutoff_mode
        compression_mode = (
            self.compression_mode
            if compression_mode is None
            else _normalize_compression_mode(compression_mode)
        )
        if sites is None:
            if form is not None and center is not None:
                raise TypeError("specify either form or center, not both")
            self.state.compress(
                form=form,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                reduced=self.reduced,
                compression_mode=compression_mode,
                compression_seed=(
                    self.compression_seed
                    if compression_seed is None else compression_seed
                ),
                order=order,
                info_c=self.info_c,
            )
        else:
            if form is not None:
                raise TypeError("form is only valid when compressing the whole tree")
            self.state.compress_subtree(
                sites,
                span=span,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                reduced=self.reduced,
                compression_mode=compression_mode,
                compression_seed=(
                    self.compression_seed
                    if compression_seed is None else compression_seed
                ),
                order=order,
                inplace=True,
                info_c=self.info_c,
            )
        self._sync_info()
        return self

    def local_expectation(self, operator, where, **kwargs):
        """Evaluate a local observable through the live TreePeps state."""

        return self.state.local_expectation(operator, where, **kwargs)

    def local_expectations(self, terms, **kwargs):
        """Evaluate several local observables in mapping iteration order."""

        return self.state.local_expectations(terms, **kwargs)

    def expectation(self, operator, *, normalized=True, **kwargs):
        """Evaluate ``<state|operator|state>`` for a TreePEPO fragment."""

        if not isinstance(operator, (TreePepo, TreeSubPepo)):
            raise TypeError("expectation requires a TreePepo or TreeSubPepo")
        return operator.expectation(self.state, normalized=normalized, **kwargs)

    def bond_report(self):
        """Return current bond, tensor, degree, and optimizer-cap diagnostics."""

        report = self.state.bond_report()
        report.update({"chi": self.chi, "queued_events": len(self._gate_stream)})
        return report

    def estimate_bonds(self, gates=None):
        """Estimate conservative bond growth for the queued operator stream.

        Each event contributes its exact factorized TreePepo bond dimension on
        every retained tree edge. The estimate intentionally ignores
        cancellations and state-specific rank deficiencies, matching the
        resource-preflight meaning of ``TreeOptimizer.estimate_bonds``.
        """

        stream = self._gate_stream if gates is None else self._normalize_stream(gates)
        self._validate_stream_backend(stream=stream)
        edge_bonds = self.state.bond_sizes()
        events = []
        for index, entry in enumerate(stream):
            kind = entry[0]
            if kind == "gate":
                support = tuple(entry[2])
                operator = TreePepo.from_operator(
                    self.plan,
                    entry[1],
                    support,
                    dims=self._physical_dims(),
                    dtype=self._gate_dtype(entry[1]),
                    max_operator_sites=self.max_operator_sites,
                )
            elif kind == "sub_treepepo":
                operator = entry[1].operator
                support = tuple(entry[1].support)
            else:
                operator = entry[1]
                support = tuple(operator.operator_support or ())
            operator.validate()
            operator_bonds = operator.bond_sizes()
            for edge in tuple(edge_bonds):
                edge_bonds[edge] *= int(operator_bonds.get(edge, 1))
            span = operator.operator_span or frozenset()
            events.append(
                {
                    "index": index,
                    "kind": kind,
                    "support": support,
                    "span": tuple(sorted(span)),
                    "span_nodes": len(span),
                    "operator_bonds": dict(operator_bonds),
                    "edge_bonds": dict(edge_bonds),
                }
            )
        max_bond = max(edge_bonds.values(), default=1)
        return {
            "edge_bonds": edge_bonds,
            "initial_edge_bonds": self.state.bond_sizes(),
            "max_bond": int(max_bond),
            "chi": self.chi,
            "requires_truncation": (
                False if self.chi is None else bool(max_bond > self.chi)
            ),
            "events": events,
        }

    @staticmethod
    def _positive_limit(value, name):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer or None")
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer or None")
        return value

    def preflight(
        self,
        gates=None,
        *,
        max_bond=None,
        max_operator_qubits=None,
        max_subtree_nodes=None,
        max_intermediate_bond=None,
        raise_on_error=True,
    ):
        """Check conservative operator, subtree, and bond limits before replay."""

        max_bond = self._positive_limit(max_bond, "max_bond")
        max_operator_qubits = self._positive_limit(
            max_operator_qubits, "max_operator_qubits"
        )
        max_subtree_nodes = self._positive_limit(
            max_subtree_nodes, "max_subtree_nodes"
        )
        if max_intermediate_bond is None:
            max_intermediate_bond = self.max_intermediate_bond
        max_intermediate_bond = self._positive_limit(
            max_intermediate_bond, "max_intermediate_bond"
        )
        report = self.estimate_bonds(gates)
        violations = []
        if max_bond is not None and report["max_bond"] > max_bond:
            violations.append(
                f"estimated max bond {report['max_bond']} exceeds max_bond={max_bond}"
            )
        if (
            max_intermediate_bond is not None
            and report["max_bond"] > max_intermediate_bond
        ):
            violations.append(
                "estimated intermediate max bond "
                f"{report['max_bond']} exceeds "
                f"max_intermediate_bond={max_intermediate_bond}"
            )
        for event in report["events"]:
            if (
                max_operator_qubits is not None
                and len(event["support"]) > max_operator_qubits
            ):
                violations.append(
                    f"event {event['index']} has {len(event['support'])} operator sites, "
                    f"exceeding max_operator_qubits={max_operator_qubits}"
                )
            if (
                max_subtree_nodes is not None
                and event["span_nodes"] > max_subtree_nodes
            ):
                violations.append(
                    f"event {event['index']} spans {event['span_nodes']} tree nodes, "
                    f"exceeding max_subtree_nodes={max_subtree_nodes}"
                )
        result = dict(report)
        result["limits"] = {
            "max_bond": max_bond,
            "max_operator_qubits": max_operator_qubits,
            "max_subtree_nodes": max_subtree_nodes,
            "max_intermediate_bond": max_intermediate_bond,
        }
        result["violations"] = violations
        result["ok"] = not violations
        if violations and raise_on_error:
            raise MemoryError(
                "TreePepsOptimizer preflight failed: " + "; ".join(violations)
            )
        return result

    def truncation_report(self):
        """Return per-update bond changes collected during replay."""

        events = deepcopy(self.history)
        return {
            "track_truncation": bool(self.track_truncation),
            "n_events": len(events),
            "n_truncated": sum(bool(event.get("truncated")) for event in events),
            "n_tracked": 0,
            "total_discarded_weight": None,
            "max_discarded_weight": None,
            "max_discarded_fraction": None,
            "events": events,
            "updates": deepcopy(events),
        }

    @classmethod
    def convergence_sweep(
        cls,
        gates=None,
        state=None,
        *,
        tn=None,
        chi_values=(2, 4, 8, 16, 32),
        ops=None,
        plan=None,
        cutoff=0.0,
        cutoff_mode="rsum2",
        compression_mode="direct",
        dense_cap=1 << 14,
        **optimizer_opts,
    ):
        """Replay one gate stream at several bond caps.

        ``TreePeps`` has no implicit product-state constructor because its
        physical dimensions belong to the supplied tensors. The initial state
        is therefore required, either as ``state=``/``tn=`` or as the first
        positional argument. For small states an uncapped replay supplies an
        exact dense reference fidelity; larger states still report bond,
        norm, and observable convergence without materializing a reference.
        """

        if isinstance(gates, TreePeps):
            if state is None and tn is None:
                state, gates = gates, state
            elif isinstance(state, TreePeps):
                raise TypeError("two TreePeps states were supplied")
            else:
                gates, state = state, gates
        if state is None:
            state = tn
        if not isinstance(state, TreePeps):
            raise TypeError(
                "convergence_sweep requires an initial TreePeps state via "
                "state= or tn="
            )
        if plan is None:
            plan = state.plan
        if not isinstance(plan, TreePepsPlan):
            raise TypeError("plan must be a TreePepsPlan")
        if plan_signature(plan) != plan_signature(state.plan):
            raise ValueError("plan and state must use the same tree plan")

        stream = tuple(cls._as_stream_entries(gates))
        chi_values = tuple(sorted(set(chi_values)))
        if not chi_values:
            raise ValueError("chi_values cannot be empty")
        for chi in chi_values:
            if isinstance(chi, bool) or not isinstance(chi, Integral) or chi < 1:
                raise ValueError("chi_values must contain positive integers")

        physical_size = 1
        for dimension in cls(state, plan=plan, run=False)._physical_dims().values():
            physical_size *= dimension

        common = dict(optimizer_opts)
        for name in (
            "state",
            "tn",
            "plan",
            "layout",
            "gates",
            "run",
            "chi",
            "cutoff",
            "cutoff_mode",
            "compression_mode",
        ):
            if name in common:
                raise TypeError(f"{name} is controlled by convergence_sweep")

        exact_vector = None
        if physical_size <= int(dense_cap):
            exact = cls(
                state,
                plan=plan,
                chi=None,
                cutoff=0.0,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                run=False,
                **common,
            )
            exact.set_gates(stream).run()
            exact_vector = np.asarray(exact.to_dense(), dtype=complex).reshape(-1)
            exact_norm = np.linalg.norm(exact_vector)
            if exact_norm > 0.0:
                exact_vector = exact_vector / exact_norm
            else:
                exact_vector = None

        if ops is None:
            ops = ()
        elif isinstance(ops, Mapping):
            ops = tuple((operator, where) for where, operator in ops.items())
        else:
            ops = tuple(ops)

        records = []
        previous_values = None
        for chi in chi_values:
            optimizer = cls(
                state,
                plan=plan,
                chi=int(chi),
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                run=False,
                **common,
            )
            optimizer.set_gates(stream).run()
            expectations = {}
            values = []
            for index, item in enumerate(ops):
                try:
                    operator, where = item
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "ops must contain (operator, support) pairs"
                    ) from exc
                if isinstance(operator, (TreePepo, TreeSubPepo)):
                    value = optimizer.expectation(operator)
                else:
                    value = optimizer.local_expectation(
                        operator,
                        where,
                        max_bond=None,
                        optimize="auto",
                    )
                value = complex(np.asarray(ar.to_numpy(value)))
                expectations[f"op{index}"] = value
                values.append(value)

            fidelity = None
            if exact_vector is not None:
                vector = np.asarray(optimizer.to_dense(), dtype=complex).reshape(-1)
                vector_norm = np.linalg.norm(vector)
                if vector_norm > 0.0:
                    vector = vector / vector_norm
                    fidelity = float(abs(np.vdot(exact_vector, vector)) ** 2)

            max_drift = None
            if previous_values is not None and values:
                max_drift = float(
                    max(abs(old - new) for old, new in zip(previous_values, values))
                )
            if values:
                previous_values = values
            records.append(
                {
                    "chi": int(chi),
                    "max_bond": optimizer.max_bond(),
                    "norm": optimizer.norm(),
                    "expectations": expectations,
                    "fidelity": fidelity,
                    "max_drift": max_drift,
                }
            )
        return records

    def get_infidelities(self):
        """Return conservative compression-loss markers for replay updates.

        TreePeps records exact before/after bond dimensions rather than
        performing an additional dense reference contraction. ``None`` marks
        an update for which a scalar fidelity cannot be inferred from those
        dimensions alone; callers needing spectra should inspect the live
        ``bond_report`` or perform an explicit small-system comparison.
        """

        return [0.0] + [
            None if event.get("truncated") else 0.0 for event in self.history
        ]

    def get_infidelity_samples(self):
        """Return update-level truncation markers with their support metadata."""

        return [
            {
                "step": index,
                "support": event.get("support"),
                "span": event.get("span"),
                "truncated": bool(event.get("truncated")),
                "infidelity": (
                    None if event.get("truncated") else 0.0
                ),
            }
            for index, event in enumerate(self.history, start=1)
        ]

    def bond_diagnostic_report(self):
        """Return live bond changes and transient span diagnostics."""

        updates = deepcopy(self.history)
        if not self.track_bond_diagnostics:
            updates = [
                {
                    **update,
                    "live_max_bond_after": None,
                    "transient_max_bond": None,
                    "transient_exceeds_chi": None,
                }
                for update in updates
            ]
        measured = [
            update
            for update in updates
            if update.get("transient_max_bond") is not None
        ]
        return {
            "enabled": bool(self.track_bond_diagnostics),
            "chi": self.chi,
            "updates": updates,
            "max_live_bond_after": (
                max(
                    update["live_max_bond_after"]
                    for update in measured
                    if update.get("live_max_bond_after") is not None
                )
                if measured
                else None
            ),
            "max_transient_bond": (
                max(update["transient_max_bond"] for update in measured)
                if measured
                else None
            ),
            "n_transient_exceeds_chi": sum(
                bool(update.get("transient_exceeds_chi")) for update in measured
            ),
        }

    def profile_report(self):
        """Return opt-in replay timings grouped by update kind.

        TreePeps routing is intentionally a single span-local kernel, so the
        report records update envelopes rather than pretending to expose the
        internal QR/SVD timings of Quimb. Profiling is disabled by default and
        adds no timer calls to normal replay.
        """

        events = deepcopy(self.profile_events)
        grouped = {}
        for event in events:
            kind = str(event.get("kind", "unknown"))
            summary = grouped.setdefault(kind, {"count": 0, "seconds": 0.0})
            summary["count"] += 1
            summary["seconds"] += float(event.get("seconds", 0.0))
        return {
            "enabled": bool(self.profile),
            "profile_sync": bool(self.profile_sync),
            "events": events,
            "by_kind": grouped,
            "update_seconds": float(
                grouped.get("update", {}).get("seconds", 0.0)
            ),
            "total_seconds": float(
                sum(float(event.get("seconds", 0.0)) for event in events)
            ),
            "timing_semantics": {
                "wall_envelope": "update",
                "total_seconds_is_sum_of_events_not_wall_time": True,
            },
        }

    def get_norm_events(self):
        """Return retained-norm fidelity records for replay updates."""

        return deepcopy(self.norm_events)

    def get_normalizations(self):
        """Return explicit physical normalizations performed by the optimizer."""

        return deepcopy(self.normalizations)

    def get_fit_diagnostics(self):
        """Return the latest TreeFIT diagnostic record, if available."""

        return None if self._last_fit_diagnostics is None else deepcopy(
            self._last_fit_diagnostics
        )

    def norm_diagnostics(self):
        """Return local/cumulative compression fidelity diagnostics.

        ``local_fidelity`` and ``cumulative_fidelity`` are retained-norm
        proxies, not directional overlaps with an independently supplied
        target state. The live represented state norm remains available under
        ``norm``/``state_norm`` but is deliberately separate from the
        compression-fidelity values.
        """

        valid = [event for event in self.norm_events if event.get("valid")]
        current = valid[-1] if valid else None
        cumulative_fidelity = (
            None if not valid else self._cumulative_fidelity()
        )
        cumulative_infidelity = (
            None
            if cumulative_fidelity is None
            else self._cumulative_infidelity()
        )
        state_norm = self.norm()
        event_fidelities = [
            float(event["local_fidelity"]) for event in valid
        ]
        event_infidelities = [
            float(event["local_infidelity"]) for event in valid
        ]
        if event_fidelities and any(value <= 0.0 for value in event_fidelities):
            geometric_fidelity = 0.0
        elif event_fidelities:
            geometric_fidelity = float(
                math.exp(
                    sum(math.log(value) for value in event_fidelities)
                    / len(event_fidelities)
                )
            )
        else:
            geometric_fidelity = None
        return {
            "tracking": bool(self.track_infidelity),
            "norm_tracking": bool(self.track_infidelity),
            "truncation_tracking": bool(self.track_truncation),
            "current_valid": current is not None,
            "events": len(self.norm_events),
            "completed_events": len(valid),
            "completed_segments": len(valid),
            "segments_including_current": len(valid),
            "completed_segment_norms": [
                float(max(0.0, value) ** 0.5) for value in event_fidelities
            ],
            "completed_segment_infidelities": event_infidelities,
            "current_event": None if current is None else deepcopy(current),
            "current_fidelity": (
                None if current is None else current["local_fidelity"]
            ),
            "current_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "local_fidelity": (
                None if current is None else current["local_fidelity"]
            ),
            "local_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "cumulative_fidelity": cumulative_fidelity,
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": cumulative_fidelity,
            "cumulative_compression_infidelity": cumulative_infidelity,
            "norm_survival": cumulative_fidelity,
            "fidelity": cumulative_fidelity,
            "infidelity": cumulative_infidelity,
            "norm": state_norm,
            "state_norm": state_norm,
            "cumulative_norm": (
                None
                if cumulative_fidelity is None
                else float(cumulative_fidelity**0.5)
            ),
            "total_survival_proxy": cumulative_fidelity,
            "total_infidelity_proxy": cumulative_infidelity,
            "total_norm_proxy": (
                None
                if cumulative_fidelity is None
                else float(cumulative_fidelity**0.5)
            ),
            "geometric_mean_survival": geometric_fidelity,
            "geometric_mean_norm": (
                None
                if geometric_fidelity is None
                else float(geometric_fidelity**0.5)
            ),
            "mean_segment_infidelity": (
                None
                if not event_infidelities
                else float(sum(event_infidelities) / len(event_infidelities))
            ),
            "max_segment_infidelity": (
                None
                if not valid
                else max(event["local_infidelity"] for event in valid)
            ),
            "segment_infidelities": event_infidelities,
            "current_event_kind": None if current is None else current["kind"],
            "current_segment_norm": (
                None
                if current is None
                else float(max(0.0, current["local_fidelity"]) ** 0.5)
            ),
            "current_segment_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "norm_events": self.get_norm_events(),
            "normalizations": self.get_normalizations(),
        }

    def to_dense(self, logical_order=True):
        """Return the dense statevector in logical site order."""

        _ = logical_order
        return self.state.to_statevector()

    @property
    def qubits(self):
        """Return logical site ids in their stable one-dimensional order."""

        return list(self.state.sites)

    @property
    def logical_order(self):
        """Return the stable logical site order."""

        return list(self.state.sites)

    def logical_site(self, position):
        """Return the logical site at a compact position."""

        position = int(position)
        if not 0 <= position < self.plan.size:
            raise IndexError(f"site position {position} is outside the state")
        return position

    def position(self, site):
        """Return the compact logical position of ``site``."""

        return self.plan.resolve_site(site)

    def remap_sample(self, config):
        """Return a sample in the unchanged logical site order."""

        if isinstance(config, dict):
            return dict(config)
        array = np.asarray(config)
        if array.ndim == 0 or array.shape[-1] != self.plan.size:
            raise ValueError(
                "sample configuration must have TreePeps size as its final "
                f"dimension, got shape {array.shape}"
            )
        return array.copy()

    def restore_qubit_order(self):
        """Return the live state, which is always stored in logical order."""

        return self.state

    def norm(self):
        """Return the represented state norm as a real scalar."""

        return float(abs(np.asarray(self.state.norm())))

    def normalize(self, eps=1e-15, insert=None):
        """Normalize the live state and return its previous norm."""

        old_norm = self.state.normalize(eps=eps, insert=insert)
        self._sync_info()
        old_norm = float(abs(np.asarray(old_norm)))
        self.normalizations.append(
            {
                "step": len(self.normalizations) + 1,
                "norm_before": old_norm,
                "norm_after": self.norm(),
                "source": "optimizer.normalize",
            }
        )
        return old_norm

    @property
    def last_report(self):
        return None if not self.history else self.history[-1]

    def copy(self):
        """Copy the optimizer and its live state without replaying gates."""

        copied = type(self)(
            self.state,
            tn=None,
            mode=self.mode,
            compression_mode=self.compression_mode,
            compression_seed=self.compression_seed,
            compression_layout=self.compression_layout,
            fit_block_size=self.fit_block_size,
            fit_n_iter=self.fit_n_iter,
            fit_adaptive_sweeps=self.fit_adaptive_sweeps,
            fit_min_iter=self.fit_min_iter,
            fit_rtol=self.fit_rtol,
            fit_patience=self.fit_patience,
            fit_init_strategy=self.fit_init_strategy,
            fit_init_rand_strength=self.fit_init_rand_strength,
            fit_init_seed=self.fit_init_seed,
            fit_sweep_sequence=self.fit_sweep_sequence,
            fit_overlap_diagnostics=self.fit_overlap_diagnostics,
            chi=self.chi,
            cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode,
            reduced=self.reduced,
            inplace=False,
            info_c=None,
            max_operator_sites=self.max_operator_sites,
            max_operator_qubits=self.max_operator_qubits,
            max_subtree_nodes=self.max_subtree_nodes,
            max_intermediate_bond=self.max_intermediate_bond,
            run=False,
            record_history=self.record_history,
            track_truncation=self.track_truncation,
            track_infidelity=self.track_infidelity,
            profile=self.profile,
            profile_sync=self.profile_sync,
            track_bond_diagnostics=self.track_bond_diagnostics,
        )
        copied.history = [dict(report) for report in self.history]
        copied.infidelities = list(self.infidelities)
        copied.norm_events = deepcopy(self.norm_events)
        copied._norm_log_survival = self._norm_log_survival
        copied._last_local_fidelity = self._last_local_fidelity
        copied._last_local_infidelity = self._last_local_infidelity
        copied.normalizations = deepcopy(self.normalizations)
        copied.profile_events = deepcopy(self.profile_events)
        copied._gate_stream = tuple(self._gate_stream)
        copied._dmrg_mode_alias = self._dmrg_mode_alias
        copied.fit_diagnostics = deepcopy(self.fit_diagnostics)
        copied._last_fit_diagnostics = deepcopy(self._last_fit_diagnostics)
        if self.info_c is not None:
            copied.info_c = deepcopy(self.info_c)
            copied._sync_info()
        return copied

    def __repr__(self):
        return (
            f"TreePepsOptimizer(mode={self.mode!r}, "
            f"compression_mode={self.compression_mode!r}, "
            f"compression_layout={self.compression_layout!r}, "
            f"shape={self.plan.shape!r}, "
            f"center={self.center!r}, chi={self.chi!r})"
        )
