"""Workload-aware spanning-tree layouts for :class:`TreePeps`."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
import random

import numpy as np

from ...tensors.maps import OneDMap
from .operators import TreePepo, TreeSubPepo
from .plan import (
    TreePepsPlan,
    _normalize_tree_peps_coarse_grain,
    _normalize_span_mode,
    _normalize_topology,
)

__all__ = ["TreePepsLayoutFinder"]


@dataclass(frozen=True)
class _Workload:
    """One interaction used to score candidate virtual trees."""

    support: tuple[int, ...]
    weight: float = 1.0
    schmidt_rank: int = 1

    @property
    def demand(self):
        return self.weight * self.schmidt_rank


class TreePepsLayoutFinder:
    """Find a bounded-degree lattice tree adapted to operator supports.

    The finder never changes tensor data. It searches legal spanning trees of
    the physical lattice and scores each interaction by the number of tree
    edges in its minimal connected span. Weighted edge load, including an
    optional operator-Schmidt rank estimate, breaks ties between layouts with
    similar total span. The result of :meth:`run` is an ordinary
    :class:`TreePepsPlan`, ready for ``TreePeps``, ``TreePepo``, and
    ``TreePepsOptimizer``.

    ``topology="tree"`` is the default and requires a rank-four branching
    site. Use ``topology="path"`` explicitly for one-dimensional or other
    MPS-compatible control geometries.

    ``interactions`` accepts dense stream entries ``(gate, where)``, explicit
    supports, ``TreePepo``/``TreeSubPepo`` objects, or mappings containing
    ``where``/``support`` and optional ``weight`` and ``schmidt_rank`` fields.
    ``supports``, ``gates``, and ``terms`` are compatibility spellings for the
    same workload input.
    """

    _OBJECTIVE_ALIASES = {
        "span": "span",
        "gate_span": "span",
        "path": "span",
        "load": "load",
        "congestion": "load",
        "hybrid": "hybrid",
    }

    @staticmethod
    def _normalize_seed_modes(seed_modes):
        """Normalize fixed traversal and stochastic seed spellings."""
        if seed_modes is None:
            seed_modes = ("span-up", "span-down", "span-middle", "span-out")
        elif isinstance(seed_modes, (str, bytes)):
            seed_modes = (seed_modes,)
        else:
            try:
                seed_modes = tuple(seed_modes)
            except TypeError as exc:
                raise TypeError("seed_modes must be a mode or iterable of modes") from exc
        if not seed_modes:
            raise ValueError("seed_modes cannot be empty")

        normalized = []
        for mode in seed_modes:
            mode = str(mode).strip().lower().replace("_", "-")
            span_mode = _normalize_span_mode(mode)
            if span_mode is not None and mode.startswith("span-"):
                mode = span_mode
            elif mode.startswith("coarse-"):
                from ..tree.layout import _normalize_layout_order

                mode = _normalize_layout_order(mode)
            else:
                # Keep legacy aliases in reports while TreePepsPlan itself
                # canonicalizes them when constructing the candidate.
                mode = OneDMap._normalize_mode(mode)
            if mode in {"source", "current", "weighted"}:
                mode = {"current": "source"}.get(mode, mode)
            elif mode in {"span-up", "span-down", "span-out", "span-middle"}:
                pass
            elif mode.startswith("coarse-"):
                pass
            elif mode not in OneDMap._KNOWN_MODES or mode == "finder":
                supported = ", ".join(OneDMap._KNOWN_MODES[:-1])
                raise ValueError(
                    f"unknown TreePeps seed mode {mode!r}; supported modes: "
                    f"source, weighted, {supported}"
                )
            if mode not in normalized:
                normalized.append(mode)
        return tuple(normalized)

    @staticmethod
    def _resolve_root(plan, root):
        if isinstance(root, str):
            if root.strip().lower().replace("_", "-") != "center":
                raise ValueError("root string must be 'center'")
            center = tuple((extent - 1) // 2 for extent in plan.shape)
            return plan.coord_to_one_d[center]
        return plan.resolve_site(root)

    def __init__(
        self,
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
        refine_budget=None,
        order=None,
        map_mode=None,
        tree_order=None,
        seed_modes=None,
        tree_orders=None,
        root=None,
        topology=None,
        coarse_grain=None,
    ):
        if supports is not None:
            if interactions is not None:
                raise TypeError("pass either interactions or supports, not both")
            interactions = supports
        for alias_name, alias_value in (("gates", gates), ("terms", terms)):
            if alias_value is None:
                continue
            if interactions is not None:
                raise TypeError(
                    f"pass only one of interactions, supports, gates, or {alias_name}"
                )
            interactions = alias_value

        if map_mode is not None and any(
            value is not None for value in (tree_order, seed_modes, tree_orders)
        ):
            raise TypeError(
                "map_mode cannot be combined with tree_order, seed_modes, "
                "or tree_orders"
            )
        if sum(value is not None for value in (tree_order, seed_modes, tree_orders)) > 1:
            raise TypeError(
                "pass only one of tree_order, seed_modes, or tree_orders"
            )
        if map_mode is not None:
            span_mode = _normalize_span_mode(map_mode)
            if span_mode is not None:
                map_mode = span_mode
            else:
                # Compatibility for the old PEPS map_mode contract.
                map_mode = str(map_mode).strip().lower().replace("_", "-")
            tree_order = map_mode
        if tree_order is not None:
            seed_modes = (tree_order,)
        elif tree_orders is not None:
            seed_modes = tree_orders
        explicit_seed_modes = seed_modes is not None
        self.seed_modes = self._normalize_seed_modes(seed_modes)
        auto_span_middle_degree = (
            max_virtual_degree is None
            and "span-middle" in self.seed_modes
        )
        self.map_mode = (
            None
            if map_mode is None
            else (_normalize_span_mode(map_mode) or map_mode)
        )

        if isinstance(geometry, TreePepsPlan):
            source_geometry = geometry
            if coarse_grain is None:
                coarse_grain = geometry.coarse_grain
            else:
                coarse_grain = _normalize_tree_peps_coarse_grain(
                    coarse_grain,
                    geometry.ndim,
                )
            if topology is None:
                topology = geometry.topology
            elif _normalize_topology(topology) != geometry.topology:
                raise ValueError("topology cannot override a TreePepsPlan topology")
            if order is not None and str(order) != geometry.order:
                raise ValueError("order cannot override the order of a TreePepsPlan")
            if max_virtual_degree is None:
                max_virtual_degree = (
                    4 if auto_span_middle_degree else geometry.max_virtual_degree
                )
            if root is not None and self._resolve_root(geometry, root) != geometry.root:
                source_geometry = TreePepsPlan(
                    geometry.shape,
                    one_d_to_coord=geometry.one_d_to_coord,
                    tree_edges=geometry.tree_edges,
                    lattice_edges=geometry.lattice_edges,
                    coord_to_one_d=geometry.coord_to_one_d,
                    root=self._resolve_root(geometry, root),
                    max_virtual_degree=geometry.max_virtual_degree,
                    order=geometry.order,
                    tree_order=geometry.tree_order,
                    map_mode=geometry.map_mode,
                    topology=geometry.topology,
                    boundary=geometry.boundary,
                    coarse_grain=coarse_grain,
                )
        else:
            if max_virtual_degree is None:
                # ``span-middle`` deliberately gives central-row/plane sites
                # four virtual bonds: two along the backbone and one chain in
                # each axial direction.
                max_virtual_degree = (
                    4 if "span-middle" in self.seed_modes else 3
                )
            topology = "tree" if topology is None else _normalize_topology(topology)
            source_root = root
            if source_root is None and explicit_seed_modes and (
                "inside-out" in self.seed_modes or "span-out" in self.seed_modes
            ):
                source_root = "center"
            source_geometry = TreePepsPlan.from_shape(
                geometry,
                order="snake" if order is None else order,
                map_mode=map_mode,
                max_virtual_degree=max_virtual_degree,
                root=source_root,
                topology=topology,
                coarse_grain=coarse_grain,
            )

        if not isinstance(max_virtual_degree, Integral) or isinstance(
            max_virtual_degree, bool
        ):
            raise TypeError("max_virtual_degree must be an integer from 1 to 4")
        max_virtual_degree = int(max_virtual_degree)
        if not 1 <= max_virtual_degree <= 4:
            raise ValueError("TreePeps virtual degree must be between 1 and 4")

        objective = str(objective).strip().lower().replace("-", "_")
        try:
            objective = self._OBJECTIVE_ALIASES[objective]
        except KeyError as exc:
            raise ValueError("objective must be 'span', 'load', or 'hybrid'") from exc

        if refine_budget is not None:
            max_iter = refine_budget
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
            raise TypeError("max_iter/refine_budget must be a non-negative integer")
        if int(max_iter) < 0:
            raise ValueError("max_iter/refine_budget must be a non-negative integer")
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise TypeError("seed must be an integer") from exc

        if (
            source_geometry.max_virtual_degree < max_virtual_degree
            and not auto_span_middle_degree
        ):
            # A geometry with a stricter cap is still a valid source graph; its
            # cap is the physical plan contract unless the caller asks for a
            # smaller value.
            max_virtual_degree = source_geometry.max_virtual_degree
        if topology == "path":
            max_virtual_degree = min(max_virtual_degree, 2)
        self.geometry = source_geometry
        self.coarse_grain = source_geometry.coarse_grain
        self.topology = topology
        self.max_virtual_degree = max_virtual_degree
        self.objective = objective
        self.seed = seed
        self.max_iter = int(max_iter)
        self.workload = self._normalize_workload(interactions)
        self._recommended = None
        self._last_report = None

    @staticmethod
    def _is_site_selector(value):
        if isinstance(value, Integral):
            return True
        if isinstance(value, (str, bytes)):
            return False
        try:
            values = tuple(value)
        except TypeError:
            return False
        if not values:
            return False
        for site in values:
            if isinstance(site, Integral):
                continue
            if isinstance(site, (str, bytes)):
                return False
            try:
                coordinate = tuple(site)
            except TypeError:
                return False
            if not coordinate or not all(isinstance(axis, Integral) for axis in coordinate):
                return False
        return True

    @staticmethod
    def _is_gate_like(value):
        if hasattr(value, "shape") or hasattr(value, "to_dense"):
            return True
        if isinstance(value, (list, tuple)):
            try:
                return np.asarray(value).ndim >= 2
            except (TypeError, ValueError):
                return False
        return hasattr(value, "data")

    @classmethod
    def _is_single_entry(cls, value):
        if isinstance(value, (TreePepo, TreeSubPepo, Mapping)):
            return True
        if not isinstance(value, (tuple, list)) or not value:
            return False
        if len(value) == 2 and cls._is_gate_like(value[0]):
            return cls._is_site_selector(value[1])
        return all(isinstance(site, Integral) for site in value)

    @classmethod
    def _as_entries(cls, value):
        if value is None:
            return []
        if cls._is_single_entry(value):
            return [value]
        if isinstance(value, (str, bytes)):
            raise TypeError("layout interactions must be structured entries")
        try:
            return list(value)
        except TypeError as exc:
            raise TypeError(
                "interactions must be a support, gate entry, or iterable"
            ) from exc

    def _normalize_support(self, support):
        if isinstance(support, Integral):
            support = (support,)
        try:
            support = tuple(self.geometry.resolve_site(site) for site in support)
        except TypeError as exc:
            raise TypeError("interaction support must be a site or iterable") from exc
        if not support:
            raise ValueError("interaction support cannot be empty")
        if len(set(support)) != len(support):
            raise ValueError("interaction support must contain distinct sites")
        return support

    @staticmethod
    def _operator_schmidt_rank(operator):
        if isinstance(operator, TreeSubPepo):
            return max(operator.operator_bond_dims.values(), default=1)
        if isinstance(operator, TreePepo):
            try:
                sizes = operator.bond_sizes()
            except (AttributeError, ValueError):
                return 1
            return max((int(size) for size in sizes.values()), default=1)
        return 1

    @staticmethod
    def _dense_gate_schmidt_rank(gate, support):
        """Estimate a two-site operator-Schmidt rank when dimensions permit."""

        try:
            support_size = len(support)
        except TypeError:
            support_size = 1
        if support_size != 2:
            return 1
        try:
            data = gate.to_dense() if hasattr(gate, "to_dense") else gate
            data = getattr(data, "data", data)
            data = np.asarray(data)
        except (TypeError, ValueError):
            return 1
        if data.ndim == 2:
            dimension = int(round(np.sqrt(data.shape[0])))
            if data.shape != (dimension * dimension, dimension * dimension):
                return 1
            data = data.reshape(dimension, dimension, dimension, dimension)
        if data.ndim != 4 or data.shape[0] != data.shape[2] or data.shape[1] != data.shape[3]:
            return 1
        matrix = data.transpose(0, 2, 1, 3).reshape(
            data.shape[0] * data.shape[0], data.shape[1] * data.shape[1]
        )
        return max(1, int(np.linalg.matrix_rank(matrix)))

    def _normalize_workload_entry(self, entry):
        weight = 1.0
        schmidt_rank = None
        if isinstance(entry, TreeSubPepo):
            support = entry.support
            schmidt_rank = self._operator_schmidt_rank(entry)
        elif isinstance(entry, TreePepo):
            support = entry.operator_support or entry.sites
            schmidt_rank = self._operator_schmidt_rank(entry)
        elif isinstance(entry, Mapping):
            support = entry.get("where", entry.get("support"))
            if support is None:
                raise ValueError("mapping interactions require 'where' or 'support'")
            weight = entry.get("weight", 1.0)
            schmidt_rank = entry.get(
                "schmidt_rank",
                entry.get("operator_schmidt_rank", entry.get("rank")),
            )
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and self._is_gate_like(entry[0])
            and self._is_site_selector(entry[1])
        ):
            gate, support = entry
            schmidt_rank = self._dense_gate_schmidt_rank(gate, support)
        else:
            support = entry

        support = self._normalize_support(support)
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("interaction weights must be finite non-negative numbers") from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("interaction weights must be finite non-negative numbers")
        if schmidt_rank is None:
            schmidt_rank = 1
        if isinstance(schmidt_rank, bool) or not isinstance(schmidt_rank, Integral):
            raise TypeError("schmidt_rank must be a positive integer")
        if int(schmidt_rank) < 1:
            raise ValueError("schmidt_rank must be a positive integer")
        return _Workload(support, weight, int(schmidt_rank))

    def _normalize_workload(self, interactions):
        return tuple(
            self._normalize_workload_entry(entry)
            for entry in self._as_entries(interactions)
        )

    def _build_plan(self, tree_edges, *, tree_order=None):
        tree_order_use = (
            self.geometry.tree_order if tree_order is None else tree_order
        )
        span_tree_mode = _normalize_span_mode(tree_order_use)
        map_mode = (
            span_tree_mode
            if span_tree_mode is not None
            else self.geometry.map_mode if tree_order is None else None
        )
        return TreePepsPlan(
            self.geometry.shape,
            one_d_to_coord=self.geometry.one_d_to_coord,
            tree_edges=tree_edges,
            lattice_edges=self.geometry.lattice_edges,
            coord_to_one_d=self.geometry.coord_to_one_d,
            root=self.geometry.root,
            max_virtual_degree=self.max_virtual_degree,
            order=self.geometry.order,
            tree_order=tree_order_use,
            map_mode=map_mode,
            topology=self.topology,
            boundary=self.geometry.boundary,
            coarse_grain=self.coarse_grain,
        )

    @staticmethod
    def _edge_key(edge):
        return tuple(sorted(edge))

    def _span_edges(self, plan, span):
        return tuple(
            edge
            for edge in plan.tree_edges
            if edge[0] in span and edge[1] in span
        )

    def _score_details(self, plan):
        edge_loads = {edge: 0.0 for edge in plan.tree_edges}
        span_cost = 0.0
        path_cost = 0.0
        interactions = []
        for workload in self.workload:
            span = plan.subtree_span(workload.support)
            span_edges = self._span_edges(plan, span)
            span_cost += workload.weight * len(span_edges)
            if len(workload.support) == 2:
                path_cost += workload.weight * (
                    len(plan.path(*workload.support)) - 1
                )
            for edge in span_edges:
                edge_loads[edge] += workload.demand
            interactions.append(
                {
                    "support": workload.support,
                    "span": tuple(sorted(span)),
                    "span_size": len(span_edges),
                    "path_length": (
                        len(plan.path(*workload.support)) - 1
                        if len(workload.support) == 2
                        else None
                    ),
                    "weight": workload.weight,
                    "schmidt_rank": workload.schmidt_rank,
                }
            )
        total_load = sum(edge_loads.values())
        peak_load = max(edge_loads.values(), default=0.0)
        if self.objective == "span":
            score = span_cost
        elif self.objective == "load":
            score = peak_load + 0.25 * total_load
        else:
            score = span_cost + peak_load + 0.25 * total_load
        return {
            "score": float(score),
            "span_cost": float(span_cost),
            "path_cost": float(path_cost),
            "edge_loads": edge_loads,
            "total_edge_load": float(total_load),
            "max_edge_load": float(peak_load),
            "interactions": tuple(interactions),
        }

    def score(self, plan):
        """Return the workload score of a compatible candidate plan."""

        if not isinstance(plan, TreePepsPlan):
            raise TypeError("plan must be a TreePepsPlan")
        if plan.shape != self.geometry.shape or plan.coordinates != self.geometry.coordinates:
            raise ValueError("plan must use the finder's lattice geometry")
        return self._score_details(plan)["score"]

    def _lattice_adjacency(self):
        adjacency = {q: [] for q in range(self.geometry.size)}
        for q0, q1 in self.geometry.lattice_edges:
            adjacency[q0].append(q1)
            adjacency[q1].append(q0)
        return {q: tuple(sorted(neighbors)) for q, neighbors in adjacency.items()}

    def _lattice_path(self, source, target, adjacency):
        if source == target:
            return (source,)
        previous = {source: None}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in previous:
                    continue
                previous[neighbor] = current
                if neighbor == target:
                    queue.clear()
                    break
                queue.append(neighbor)
        if target not in previous:
            return ()
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = previous[current]
        return tuple(reversed(path))

    def _edge_affinity(self):
        adjacency = self._lattice_adjacency()
        affinity = {edge: 0.0 for edge in self.geometry.lattice_edges}
        for workload in self.workload:
            for index, source in enumerate(workload.support):
                for target in workload.support[index + 1 :]:
                    path = self._lattice_path(source, target, adjacency)
                    for q0, q1 in zip(path, path[1:]):
                        edge = self._edge_key((q0, q1))
                        affinity[edge] += workload.demand
        return affinity

    def _weighted_seed(self, *, rng):
        """Build a degree-bounded Prim-style lattice tree."""

        affinity = self._edge_affinity()
        adjacency = self._lattice_adjacency()
        root = self.geometry.root
        visited = {root}
        degree = {q: 0 for q in range(self.geometry.size)}
        edges = set()
        while len(visited) < self.geometry.size:
            candidates = []
            for q0 in sorted(visited):
                for q1 in adjacency[q0]:
                    if q1 in visited or degree[q0] >= self.max_virtual_degree:
                        continue
                    if degree[q1] >= self.max_virtual_degree:
                        continue
                    candidates.append((self._edge_key((q0, q1)), q0, q1))
            if not candidates:
                return None
            rng.shuffle(candidates)
            edge, q0, q1 = max(
                candidates,
                key=lambda item: (affinity[item[0]], -item[0][0], -item[0][1]),
            )
            edges.add(edge)
            degree[q0] += 1
            degree[q1] += 1
            visited.add(q0)
            visited.add(q1)
        return tuple(sorted(edges))

    def _fixed_seed(self, mode):
        """Build a tree from a shared :class:`OneDMap` traversal."""
        if mode == "source":
            return self.geometry.tree_edges
        if mode == "weighted":
            return None
        candidate = TreePepsPlan.from_shape(
            self.geometry.shape,
            order=self.geometry.order,
            tree_order=mode,
            root=self.geometry.root,
            max_virtual_degree=self.max_virtual_degree,
            boundary=self.geometry.boundary,
            topology=self.topology,
            coarse_grain=self.coarse_grain,
        )
        # Constructing through TreePepsPlan keeps the shared OneDMap growth
        # ordering and all degree/fallback rules in one place.
        return candidate.tree_edges

    def _candidate_plans(self):
        try:
            base = self._build_plan(self.geometry.tree_edges)
        except ValueError:
            # The caller may ask for a stricter cap than the source geometry's
            # current tree. Rebuild the deterministic lattice path as a legal
            # seed, then let workload refinement improve it.
            base = TreePepsPlan.from_shape(
                self.geometry.shape,
                order=self.geometry.order,
                root=self.geometry.root,
                max_virtual_degree=self.max_virtual_degree,
                boundary=self.geometry.boundary,
                topology=self.topology,
                coarse_grain=self.coarse_grain,
            )
        candidates = [base]
        candidate_modes = {base.tree_edges: "source"}
        seen = {plan.tree_edges for plan in candidates}
        for mode in self.seed_modes:
            if mode in {"source", "weighted"}:
                continue
            try:
                edges = self._fixed_seed(mode)
                candidate = self._build_plan(edges, tree_order=mode)
            except (NotImplementedError, ValueError):
                continue
            if candidate.tree_edges not in seen:
                candidates.append(candidate)
                candidate_modes[candidate.tree_edges] = mode
                seen.add(candidate.tree_edges)
        for offset in range(4):
            rng = random.Random(self.seed + offset)
            edges = self._weighted_seed(rng=rng)
            if edges is None:
                continue
            candidate = self._build_plan(edges)
            if candidate.tree_edges not in seen:
                candidates.append(candidate)
                candidate_modes[candidate.tree_edges] = "weighted"
                seen.add(candidate.tree_edges)
        self._candidate_modes = candidate_modes
        return candidates

    def _refine(self, plan):
        current = plan
        for _ in range(self.max_iter):
            best = current
            best_score = self.score(current)
            current_edges = set(current.tree_edges)
            for added in self.geometry.lattice_edges:
                if added in current_edges:
                    continue
                cycle = current.path(*added)
                cycle_edges = tuple(
                    self._edge_key((q0, q1)) for q0, q1 in zip(cycle, cycle[1:])
                )
                for removed in cycle_edges:
                    edges = (current_edges - {removed}) | {added}
                    try:
                        candidate = self._build_plan(tuple(sorted(edges)))
                    except ValueError:
                        continue
                    candidate_score = self.score(candidate)
                    if candidate_score < best_score - 1e-12:
                        best = candidate
                        best_score = candidate_score
            if best is current:
                break
            current = best
        return current

    def run(self, *, refine=True):
        """Return the best deterministic bounded-degree ``TreePepsPlan``."""

        if not isinstance(refine, bool):
            raise TypeError("refine must be a boolean")
        candidates = self._candidate_plans()
        if refine and self.max_iter:
            candidates = [self._refine(candidate) for candidate in candidates]
        selected = min(candidates, key=self.score)
        self._recommended = selected
        details = self._score_details(selected)
        self._last_report = {
            "objective": self.objective,
            "seed": self.seed,
            "max_virtual_degree": self.max_virtual_degree,
            "max_degree": selected.max_degree,
            "max_tensor_rank": selected.max_tensor_rank,
            "tree_edges": selected.tree_edges,
            "score": details["score"],
            "span_cost": details["span_cost"],
            "path_cost": details["path_cost"],
            "edge_loads": dict(details["edge_loads"]),
            "total_edge_load": details["total_edge_load"],
            "max_edge_load": details["max_edge_load"],
            "interactions": details["interactions"],
            "seed_modes": self.seed_modes,
            "coarse_grain": self.coarse_grain,
            "map_mode": selected.map_mode,
            "topology": selected.topology,
            "selected_seed": self._candidate_modes.get(
                selected.tree_edges, "refined"
            ),
            "n_candidates": len(candidates),
        }
        return selected

    recommend = run

    @property
    def plan(self):
        """Return the cached recommendation, computing it if necessary."""

        return self._recommended if self._recommended is not None else self.run()

    @property
    def report(self):
        """Return diagnostics for the last recommendation."""

        if self._last_report is None:
            self.run()
        return dict(self._last_report)

    def __repr__(self):
        return (
            f"TreePepsLayoutFinder(shape={self.geometry.shape!r}, "
            f"interactions={len(self.workload)}, objective={self.objective!r}, "
            f"max_virtual_degree={self.max_virtual_degree})"
        )
