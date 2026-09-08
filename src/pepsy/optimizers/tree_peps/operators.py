"""Tree-native PEPO operators and support-aware sub-operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ..._internal.quimb import quimb_1d_compression_function
from ...operators._structural_compression import _structural_compress_tree
from ..tree._display import ascii_tree
from ._compression import (
    iter_tree_compression_order,
    normalize_tree_compression_order,
)
from .plan import TreePepsPlan

__all__ = ["TreePEPO", "TreeSubPEPO", "TreePepo", "TreeSubPepo"]


_PATH_TWO_LAYER_COMPRESSION_MODES = frozenset(
    {"direct", "dm", "sdc", "src", "zipup"}
)
_AUTO_TWO_LAYER_COMPRESSION_MODES = frozenset({"sdc", "src", "zipup"})


def _normalize_compression_layout(layout):
    """Normalize how an operator and state are presented to compression."""

    if layout is None:
        return "auto"
    layout = str(layout).strip().lower().replace("-", "_")
    aliases = {
        "2layer": "two_layer",
        "two_layers": "two_layer",
        "two_layered": "two_layer",
        "fused_layer": "fused",
    }
    layout = aliases.get(layout, layout)
    if layout not in {"auto", "fused", "two_layer"}:
        raise ValueError(
            "compression_layout must be 'auto', 'fused', or 'two_layer'."
        )
    return layout


class TreePepo(qtn.TensorNetworkGenOperator):
    """A PEPO-like operator whose virtual graph is a ``TreePepsPlan``.

    The operator has one tensor per lattice site, an input and output
    physical leg at every site, and one virtual operator bond per retained
    tree edge.  It deliberately subclasses Quimb's generic operator network
    rather than ``quimb.PEPO``: the latter assumes a rectangular 2D lattice
    with the usual four-neighbour bond pattern.

    Dense operators are accepted in output-then-input axis order.  A matrix
    is reshaped according to ``support`` and then factorized exactly over its
    minimal tree span.  Sites outside that span receive bond-one identity
    tensors, so the result remains a complete operator network and can be
    applied to :class:`TreePeps` without creating a second tensor graph.
    """

    _EXTRA_PROPS = qtn.TensorNetworkGenOperator._EXTRA_PROPS + (
        "_tree_peps_plan",
        "_coord_site_tag_id",
        "_logical_site_tag_id",
        "_node_tag_id",
        "_operator_bond_id",
        "_input_ind_id",
        "_output_ind_id",
        "_physical_dims",
        "_operator_support",
        "_operator_span",
        "_canonical_region",
        "_operator_terms",
        "_layout_finder",
    )

    def __init__(
        self,
        ts=(),
        *,
        plan: TreePepsPlan | None = None,
        coord_site_tag_id: str | None = None,
        logical_site_tag_id: str = "I{}",
        node_tag_id: str = "N{}",
        operator_bond_id: str = "_tppo{}_{}",
        input_ind_id: str | None = None,
        output_ind_id: str | None = None,
        physical_dims=None,
        operator_support=None,
        operator_span=None,
        canonical_region=None,
        operator_terms=None,
        layout_finder=None,
        **tn_opts,
    ) -> None:
        if isinstance(ts, TreePepo):
            if plan is not None and plan_signature(plan) != ts.plan_signature:
                raise ValueError("copied TreePepo and plan describe different trees")
            if layout_finder is None:
                layout_finder = ts.layout_finder
            tn_opts.pop("virtual", None)
            super().__init__(ts, virtual=False, **tn_opts)
            self.layout_finder = layout_finder
            return

        if plan is None:
            raise TypeError("TreePepo requires a TreePepsPlan")
        if not isinstance(plan, TreePepsPlan):
            raise TypeError("plan must be a TreePepsPlan")
        if coord_site_tag_id is None:
            coord_site_tag_id = _format_for_plan("I", plan)
        if input_ind_id is None:
            input_ind_id = _format_for_plan("k", plan)
        if output_ind_id is None:
            output_ind_id = _format_for_plan("b", plan)

        tn_opts.pop("virtual", None)
        super().__init__(ts, virtual=False, **tn_opts)
        self._tree_peps_plan = plan
        self._coord_site_tag_id = str(coord_site_tag_id)
        self._site_tag_id = self._coord_site_tag_id
        self._logical_site_tag_id = str(logical_site_tag_id)
        self._node_tag_id = str(node_tag_id)
        self._operator_bond_id = str(operator_bond_id)
        self._input_ind_id = str(input_ind_id)
        self._output_ind_id = str(output_ind_id)
        self._upper_ind_id = self._input_ind_id
        self._lower_ind_id = self._output_ind_id
        self._physical_dims = None if physical_dims is None else dict(physical_dims)
        self._sites = tuple(range(plan.size))
        self._operator_support = (
            None
            if operator_support is None
            else frozenset(plan.resolve_site(site) for site in operator_support)
        )
        self._operator_span = (
            None
            if operator_span is None
            else frozenset(plan.resolve_site(site) for site in operator_span)
        )
        self._canonical_region = (
            None
            if canonical_region is None
            else frozenset(plan.resolve_site(site) for site in canonical_region)
        )
        self._operator_terms = operator_terms
        self.layout_finder = layout_finder

    @classmethod
    def identity(
        cls,
        plan: TreePepsPlan,
        *,
        phys_dim: int | Sequence[int] | Mapping = 2,
        dtype=complex,
        **operator_opts,
    ) -> "TreePepo":
        """Construct a product identity operator on every plan site."""

        return cls.from_product(
            plan,
            operators=None,
            phys_dim=phys_dim,
            dtype=dtype,
            **operator_opts,
        )

    @classmethod
    def from_product(
        cls,
        plan: TreePepsPlan,
        operators=None,
        *,
        phys_dim: int | Sequence[int] | Mapping = 2,
        dtype=complex,
        **operator_opts,
    ) -> "TreePepo":
        """Construct a bond-one product operator.

        ``operators`` may be a mapping from logical ids or coordinates to
        square local matrices.  Unspecified sites receive identities.
        """

        if not isinstance(plan, TreePepsPlan):
            raise TypeError("plan must be a TreePepsPlan")
        dims = _site_dimensions(plan, phys_dim, name="phys_dim")
        local = {}
        if operators is not None:
            if isinstance(operators, Mapping):
                entries = operators.items()
            else:
                values = tuple(operators)
                if len(values) != plan.size:
                    raise ValueError("operator sequence must contain one entry per site")
                entries = enumerate(values)
            for site, operator in entries:
                q = plan.resolve_site(site)
                array = np.asarray(operator)
                if array.shape != (dims[q], dims[q]):
                    raise ValueError(
                        f"operator at site {q} has shape {array.shape}, "
                        f"expected {(dims[q], dims[q])}"
                    )
                local[q] = np.asarray(array, dtype=dtype)

        tensors = []
        support = frozenset(local)
        for q in range(plan.size):
            dim = dims[q]
            array = local.get(q, np.eye(dim, dtype=dtype))
            data = np.zeros((dim, dim) + (1,) * len(plan.neighbors(q)), dtype=dtype)
            data[(slice(None), slice(None)) + (0,) * len(plan.neighbors(q))] = array
            inds = [
                _output_ind(plan, q, operator_opts.get("output_ind_id")),
                _input_ind(plan, q, operator_opts.get("input_ind_id")),
            ]
            inds.extend(
                _edge_name(plan, q, neighbor, operator_opts.get("operator_bond_id"))
                for neighbor in plan.neighbors(q)
            )
            tensors.append(
                qtn.Tensor(
                    data=data,
                    inds=inds,
                    tags=_site_tags(plan, q, operator_opts.get("coord_site_tag_id")),
                )
            )

        state = cls(
            tensors,
            plan=plan,
            operator_support=support,
            operator_span=plan.subtree_span(support) if support else frozenset(),
            physical_dims=dims,
            **operator_opts,
        )
        state.validate()
        return state

    @classmethod
    def from_operator(
        cls,
        plan: TreePepsPlan,
        operator,
        support,
        *,
        dims: int | Sequence[int] | Mapping = 2,
        dtype=None,
        canonicalize=False,
        center=None,
        max_operator_sites=12,
        **operator_opts,
    ) -> "TreePepo":
        """Build an exact tree operator from a dense local operator.

        ``support`` gives the logical sites, in the same order as the dense
        operator axes.  The operator is supplied as a matrix or as a tensor
        with output axes followed by input axes.  The factorization is exact
        up to machine-precision null singular values and is performed only on
        the minimal connected tree span of ``support``.
        """

        if not isinstance(plan, TreePepsPlan):
            raise TypeError("plan must be a TreePepsPlan")
        support = _normalize_sites(plan, support, name="support")
        if max_operator_sites is not None and len(support) > int(max_operator_sites):
            raise ValueError(
                "dense TreePepo construction is limited to "
                f"{int(max_operator_sites)} support sites; use a structured "
                "operator constructor for larger operators"
            )
        all_dims = _operator_dimensions(plan, support, dims)
        data, site_dims = _normalize_operator(
            operator,
            support,
            dims=tuple(all_dims[q] for q in support),
            dtype=dtype,
        )
        network = _build_dense_operator_network(
            plan,
            data,
            support,
            site_dims,
            coord_site_tag_id=operator_opts.get("coord_site_tag_id"),
            input_ind_id=operator_opts.get("input_ind_id"),
            output_ind_id=operator_opts.get("output_ind_id"),
            operator_bond_id=operator_opts.get("operator_bond_id"),
            physical_dims=all_dims,
        )
        result = cls(
            network.tensor_map.values(),
            plan=plan,
            operator_support=frozenset(support),
            operator_span=plan.subtree_span(support),
            physical_dims=all_dims,
            **operator_opts,
        )
        result.validate()
        if canonicalize:
            result.canonicalize(center=center, inplace=True)
        return result

    @classmethod
    def from_dense(
        cls,
        plan: TreePepsPlan,
        array,
        *,
        dims: int | Sequence[int] | Mapping = 2,
        dtype=None,
        canonicalize=False,
        center=None,
        max_operator_sites=12,
        **operator_opts,
    ) -> "TreePepo":
        """Build an operator over every site from its dense matrix."""

        return cls.from_operator(
            plan,
            array,
            tuple(range(plan.size)),
            dims=dims,
            dtype=dtype,
            canonicalize=canonicalize,
            center=center,
            max_operator_sites=max_operator_sites,
            **operator_opts,
        )

    from_gate = from_operator

    @classmethod
    def from_terms(
        cls,
        plan: TreePepsPlan,
        terms,
        *,
        dims: int | Sequence[int] | Mapping = 2,
        dtype=None,
        max_terms=None,
        **operator_opts,
    ) -> "TreePepo":
        """Build a sum of local dense terms without a full dense lattice array.

        Each term is factorized on its own tree span and the networks are
        combined with Quimb's arbitrary-geometry direct-sum operation.  The
        resulting operator bonds therefore grow with the routed term channels,
        rather than with the full Hilbert-space dimension.
        """

        if not isinstance(terms, Mapping) or not terms:
            raise TypeError("terms must be a non-empty mapping of support to operators")
        if max_terms is not None and len(terms) > int(max_terms):
            raise ValueError("terms exceeds the requested max_terms limit")
        result = None
        support_union = set()
        for support, operator in terms.items():
            term_support = _normalize_sites(plan, support, name="term support")
            term = cls.from_operator(
                plan,
                operator,
                term_support,
                dims=dims,
                dtype=dtype,
                **operator_opts,
            )
            support_union.update(term_support)
            if result is None:
                result = term
            else:
                result = qtn.tensor_network_ag_sum(
                    result,
                    term,
                    site_tags=result.site_tags,
                    inplace=True,
                )
                result._canonical_region = None
                for q in result.sites:
                    result.node_tensor(q).modify(left_inds=None)
        result._operator_support = frozenset(support_union)
        result._operator_span = plan.subtree_span(support_union)
        result._operator_terms = dict(terms)
        result.validate()
        return result

    @property
    def plan(self) -> TreePepsPlan:
        """The lattice and spanning-tree plan for this operator."""

        return self._tree_peps_plan

    @property
    def layout_finder(self):
        """The optional workload-aware layout metadata for this operator."""

        return self._layout_finder

    @layout_finder.setter
    def layout_finder(self, finder):
        """Attach a finder describing the same physical TreePeps layout."""

        if finder is not None:
            # Import lazily because ``TreePepsLayoutFinder`` accepts TreePepo
            # workloads and therefore imports this module itself.
            from .layout import TreePepsLayoutFinder

            if not isinstance(finder, TreePepsLayoutFinder):
                raise TypeError(
                    "layout_finder must be a TreePepsLayoutFinder or None."
                )
            if plan_signature(finder.geometry) != self.plan_signature:
                raise ValueError(
                    "layout_finder and TreePepo must describe the same "
                    "TreePepsPlan."
                )
        self._layout_finder = finder

    @property
    def map_mode(self) -> str | None:
        """Canonical lattice spanning-tree mode, if the plan has one."""

        return self.plan.map_mode

    @property
    def plan_signature(self):
        """Immutable geometry signature used when applying the operator."""

        return plan_signature(self.plan)

    @property
    def shape(self):
        return self.plan.shape

    @property
    def ndim(self):
        return self.plan.ndim

    @property
    def site_tags(self):
        return tuple(self.site_tag(q) for q in self.sites)

    def site_tag(self, site, *rest):
        """Return a coordinate-style site tag."""

        q = self.plan.resolve_site(site, *rest)
        return self._coord_site_tag_id.format(*self.plan.coordinate(q))

    def logical_site_tag(self, site) -> str:
        q = self.plan.resolve_site(site)
        return self._logical_site_tag_id.format(q)

    def node_tag(self, site) -> str:
        q = self.plan.resolve_site(site)
        return self._node_tag_id.format(q)

    def input_ind(self, site, *rest) -> str:
        q = self.plan.resolve_site(site, *rest)
        return self._input_ind_id.format(*self.plan.coordinate(q))

    def output_ind(self, site, *rest) -> str:
        q = self.plan.resolve_site(site, *rest)
        return self._output_ind_id.format(*self.plan.coordinate(q))

    def upper_ind(self, site):
        """Quimb-compatible alias for the operator input leg."""

        return self.input_ind(site)

    def lower_ind(self, site):
        """Quimb-compatible alias for the operator output leg."""

        return self.output_ind(site)

    @property
    def input_ind_id(self):
        return self._input_ind_id

    @property
    def output_ind_id(self):
        return self._output_ind_id

    def coordinate(self, site, *rest):
        return self.plan.coordinate(self.plan.resolve_site(site, *rest))

    def axis_tags(self, site):
        coordinate = self.coordinate(site)
        tags = [f"X{coordinate[0]}", f"Y{coordinate[1]}"]
        if self.ndim == 3:
            tags.append(f"Z{coordinate[2]}")
        return tuple(tags)

    def node_tid(self, site):
        q = self.plan.resolve_site(site)
        cache = self.__dict__.get("_tree_pepo_tid_cache")
        if cache is None:
            cache = self.__dict__["_tree_pepo_tid_cache"] = {}
        tid = cache.get(q)
        if tid is not None and tid in self.tensor_map:
            return tid
        tid = next(iter(self.tag_map[self.node_tag(q)]))
        cache[q] = tid
        return tid

    def node_tensor(self, site):
        return self.tensor_map[self.node_tid(site)]

    def operator_bond_ind(self, site0, site1) -> str:
        q0, q1 = self._edge_sites(site0, site1)
        return _edge_name(self.plan, q0, q1, self._operator_bond_id)

    def _edge_sites(self, site0, site1):
        q0 = self.plan.resolve_site(site0)
        q1 = self.plan.resolve_site(site1)
        if q1 not in self.plan.neighbors(q0):
            raise ValueError(f"sites {q0} and {q1} are not adjacent in the tree")
        return q0, q1

    def bond(self, site0, site1) -> str:
        q0, q1 = self._edge_sites(site0, site1)
        shared = qtn.bonds(self.node_tensor(q0), self.node_tensor(q1))
        if len(shared) != 1:
            raise ValueError(f"sites {q0} and {q1} must share one operator bond")
        return next(iter(shared))

    def neighbors(self, site):
        return self.plan.neighbors(site)

    @property
    def operator_support(self):
        if self._operator_support is None:
            return None
        return tuple(sorted(self._operator_support))

    @property
    def operator_span(self):
        if self._operator_span is None:
            return None
        return frozenset(self._operator_span)

    @property
    def canonical_region(self):
        return self._canonical_region

    @canonical_region.setter
    def canonical_region(self, region):
        if region is None:
            self._canonical_region = None
            return
        region = frozenset(self.plan.resolve_site(site) for site in region)
        if not region or not self.plan.is_connected(region):
            raise ValueError("canonical_region must be a non-empty connected subtree")
        self._canonical_region = region

    @property
    def orthogonality_center(self):
        if self.canonical_region is not None and len(self.canonical_region) == 1:
            return next(iter(self.canonical_region))
        return None

    def max_bond(self):
        return max(
            (self.node_tensor(q).ind_size(self.bond(q, n)) for q, n in self.plan.tree_edges),
            default=1,
        )

    def bond_sizes(self):
        return {
            tuple(sorted((q, n))): self.node_tensor(q).ind_size(self.bond(q, n))
            for q, n in self.plan.tree_edges
        }

    def validate(self, *, check_canonical=False, tol=1e-9):
        """Validate physical legs, operator bonds, tags, and tree topology."""

        virtual_counts = {}
        expected_outer = set()
        for q in self.sites:
            tensor = self.node_tensor(q)
            input_ind = self.input_ind(q)
            output_ind = self.output_ind(q)
            expected_outer.update((input_ind, output_ind))
            required_tags = {
                self.site_tag(q),
                self.logical_site_tag(q),
                self.node_tag(q),
                *self.axis_tags(q),
            }
            if not required_tags.issubset(tensor.tags):
                raise ValueError(f"operator tensor at site {q} is missing TreePepo tags")
            if input_ind == output_ind or input_ind not in tensor.inds:
                raise ValueError(f"operator tensor at site {q} has an invalid input leg")
            if output_ind not in tensor.inds:
                raise ValueError(f"operator tensor at site {q} has an invalid output leg")
            for ind in tensor.inds:
                if ind not in {input_ind, output_ind}:
                    virtual_counts[ind] = virtual_counts.get(ind, 0) + 1

        if set(self.outer_inds()) != expected_outer:
            raise ValueError("TreePepo has unexpected outer physical indices")
        if any(count != 2 for count in virtual_counts.values()):
            raise ValueError("every TreePepo virtual index must connect two tensors")

        for q0, q1 in self.plan.tree_edges:
            if len(qtn.bonds(self.node_tensor(q0), self.node_tensor(q1))) != 1:
                raise ValueError(f"tree edge ({q0}, {q1}) is not one live operator bond")
        for q0 in self.sites:
            for q1 in range(q0 + 1, self.plan.size):
                if q1 not in self.plan.neighbors(q0) and qtn.bonds(
                    self.node_tensor(q0), self.node_tensor(q1)
                ):
                    raise ValueError(f"non-tree sites ({q0}, {q1}) share an operator bond")

        if self.canonical_region is not None:
            if not self.plan.is_connected(self.canonical_region):
                raise ValueError("canonical_region must be a connected subtree")
            if check_canonical:
                self.validate_isometry_metadata()
                if not self.is_subtree_canonical_form(tol=tol):
                    raise ValueError("tracked operator canonical_region failed its isometry check")
        return True

    def isometry_direction(self, site):
        q = self.plan.resolve_site(site)
        tensor = self.node_tensor(q)
        if tensor.left_inds is None:
            return None
        left_inds = set(tensor.left_inds)
        for neighbor in self.plan.neighbors(q):
            bond = self.bond(q, neighbor)
            if left_inds == set(tensor.inds) - {bond}:
                return neighbor
        return None

    def isometry_map(self, region=None):
        if region is None:
            region = self.canonical_region
        region = frozenset() if region is None else frozenset(region)
        return {q: None if q in region else self.isometry_direction(q) for q in self.sites}

    def _toward_region(self, site, region):
        q = self.plan.resolve_site(site)
        if q in region:
            return None
        target = min(region, key=lambda candidate: len(self.plan.path(q, candidate)))
        return self.plan.path(q, target)[1]

    def _set_isometry_metadata_from_region(self, region):
        region = frozenset(self.plan.resolve_site(site) for site in region)
        if not region or not self.plan.is_connected(region):
            raise ValueError("region must be a non-empty connected subtree")
        for q in self.sites:
            tensor = self.node_tensor(q)
            if q in region:
                tensor.modify(left_inds=None)
            else:
                bond = self.bond(q, self._toward_region(q, region))
                tensor.modify(left_inds=tuple(ind for ind in tensor.inds if ind != bond))
        return self

    def validate_isometry_metadata(self, region=None):
        if region is None:
            region = self.canonical_region
        else:
            region = frozenset(self.plan.resolve_site(site) for site in region)
        for q in self.sites:
            direction = self.isometry_direction(q)
            if self.node_tensor(q).left_inds is not None and direction is None:
                raise ValueError(f"operator site {q} has malformed left_inds")
            if region is not None and q not in region:
                expected = self._toward_region(q, region)
                if direction != expected:
                    raise ValueError(f"operator site {q} is not isometric toward {expected}")
        return self

    def _isometry_toward(self, site, toward, tol=1e-9):
        tensor = self.node_tensor(site)
        bond = self.bond(site, toward)
        other_axes = [axis for axis, ind in enumerate(tensor.inds) if ind != bond]
        bond_axis = tensor.inds.index(bond)
        data = np.asarray(ar.to_numpy(tensor.data))
        matrix = data.transpose(*other_axes, bond_axis).reshape(-1, data.shape[bond_axis])
        gram = matrix.conj().T @ matrix
        real_dtype = np.asarray(data).real.dtype
        if np.issubdtype(real_dtype, np.inexact):
            tol = max(float(tol), 64.0 * np.finfo(real_dtype).eps)
        return np.allclose(
            gram,
            np.eye(data.shape[bond_axis], dtype=gram.dtype),
            atol=tol,
            rtol=tol,
        )

    def is_subtree_canonical_form(self, sites=None, *, span=False, tol=1e-9):
        if sites is None:
            region = self.canonical_region
            if region is None:
                return False
        else:
            region = (
                self.plan.subtree_span(sites)
                if span
                else frozenset(self.plan.resolve_site(site) for site in sites)
            )
        if not region or not self.plan.is_connected(region):
            return False
        return all(
            self._isometry_toward(q, self._toward_region(q, region), tol=tol)
            for q in self.sites
            if q not in region
        )

    def is_canonical_form(self, center=None, *, tol=1e-9):
        if center is None:
            center = self.orthogonality_center
        if center is None:
            return False
        return self.is_subtree_canonical_form({center}, tol=tol)

    def invalidate_canonical_form(self):
        self._canonical_region = None
        for q in self.sites:
            self.node_tensor(q).modify(left_inds=None)
        return self

    def _sync_info_c(self, info_c):
        if info_c is None:
            return None
        if not hasattr(info_c, "__setitem__"):
            raise TypeError("info_c must be a mutable mapping")
        center = self.orthogonality_center
        info_c["operator_center"] = center
        info_c["operator_region"] = self.canonical_region
        info_c["operator_isometry_map"] = self.isometry_map()
        info_c["operator_left_inds"] = {
            q: None
            if self.node_tensor(q).left_inds is None
            else tuple(self.node_tensor(q).left_inds)
            for q in self.sites
        }
        return info_c

    def canonize_between(self, *args, **kwargs):
        qtn.TensorNetworkGenOperator.canonize_between(self, *args, **kwargs)
        self.invalidate_canonical_form()
        return self

    def compress_between(self, *args, **kwargs):
        qtn.TensorNetworkGenOperator.compress_between(self, *args, **kwargs)
        self.invalidate_canonical_form()
        return self

    def canonize_edge_(
        self,
        site0,
        site1,
        absorb="right",
        *,
        info_c=None,
        _isometry_proven=False,
        **canonize_opts,
    ):
        q0, q1 = self._edge_sites(site0, site1)
        if absorb not in {"left", "right", "both"}:
            raise ValueError("absorb must be 'left', 'right', or 'both'")
        previous = self.orthogonality_center
        if _isometry_proven or (
            absorb in {"left", "right"} and self.can_skip_canonize(q0, q1, absorb=absorb)
        ):
            self._track_edge_center(q0, q1, absorb, previous=previous)
            self._sync_info_c(info_c)
            return self
        opts = {"method": "qr", "cutoff": 0.0}
        opts.update(canonize_opts)
        qtn.TensorNetworkGenOperator.canonize_between(
            self,
            self.node_tag(q0),
            self.node_tag(q1),
            absorb=absorb,
            **opts,
        )
        if absorb == "right" and previous in {q0, q1}:
            self._canonical_region = frozenset({q1})
        elif absorb == "left" and previous in {q0, q1}:
            self._canonical_region = frozenset({q0})
        else:
            self._canonical_region = None
        if absorb == "right":
            bond = self.bond(q0, q1)
            self.node_tensor(q0).modify(
                left_inds=tuple(ind for ind in self.node_tensor(q0).inds if ind != bond)
            )
            if self.orthogonality_center == q1:
                self.node_tensor(q1).modify(left_inds=None)
        elif absorb == "left":
            bond = self.bond(q0, q1)
            self.node_tensor(q1).modify(
                left_inds=tuple(ind for ind in self.node_tensor(q1).inds if ind != bond)
            )
            if self.orthogonality_center == q0:
                self.node_tensor(q0).modify(left_inds=None)
        self._sync_info_c(info_c)
        return self

    def can_skip_canonize(self, site0, site1, *, absorb="right"):
        if absorb not in {"left", "right"}:
            raise ValueError("absorb must be 'left' or 'right'")
        q0, q1 = self._edge_sites(site0, site1)
        q = q0 if absorb == "right" else q1
        tensor = self.node_tensor(q)
        if tensor.left_inds is None:
            return False
        return set(tensor.left_inds) == set(tensor.inds) - {self.bond(q0, q1)}

    def _track_edge_center(self, q0, q1, absorb, *, previous):
        if absorb == "right" and previous in {q0, q1}:
            self._canonical_region = frozenset({q1})
        elif absorb == "left" and previous in {q0, q1}:
            self._canonical_region = frozenset({q0})
        else:
            self._canonical_region = None

    def canonize_to(
        self,
        site,
        *,
        absorb="right",
        inplace=False,
        info_c=None,
        _force_full=False,
        _validate=True,
        **canonize_opts,
    ):
        q = self.plan.resolve_site(site)
        work = self if inplace else self.copy()
        if work.canonical_region is not None and not canonize_opts and not _force_full:
            return work.shift_orthogonality_center(q, absorb=absorb, info_c=info_c)
        opts = {"method": "qr", "cutoff": 0.0}
        opts.update(canonize_opts)
        qtn.TensorNetworkGenOperator.canonize_around_(
            work,
            [work.node_tag(q)],
            which="any",
            absorb=absorb,
            **opts,
        )
        work._canonical_region = frozenset({q})
        work._set_isometry_metadata_from_region({q})
        if _validate:
            work.validate(check_canonical=True)
        work._sync_info_c(info_c)
        return work

    def canonicalize(self, center=None, *, inplace=True, info_c=None, **canonize_opts):
        target = self if inplace else self.copy()
        if center is None:
            center = target.orthogonality_center
            if center is None:
                center = target.plan.root
        return target.canonize_to(
            center,
            inplace=True,
            info_c=info_c,
            **canonize_opts,
        )

    def canonicalize_(self, center=None, *, info_c=None, **canonize_opts):
        return self.canonicalize(
            center=center,
            inplace=True,
            info_c=info_c,
            **canonize_opts,
        )

    canonize = canonicalize_

    def _canonicalize_region_fast(self, region, *, absorb="right", **canonize_opts):
        if absorb not in {"left", "right"}:
            raise ValueError("canonical region movement requires absorb='left' or 'right'")
        region = frozenset(region)
        distances = {
            q: min(len(self.plan.path(q, target)) for target in region)
            for q in self.sites
            if q not in region
        }
        order = sorted(distances, key=lambda q: (-distances[q], q))
        for q in order:
            toward = self._toward_region(q, region)
            source, target = (q, toward) if absorb == "right" else (toward, q)
            if not self.can_skip_canonize(source, target, absorb=absorb):
                self.canonize_edge_(source, target, absorb=absorb, **canonize_opts)
        self._canonical_region = region
        self._set_isometry_metadata_from_region(region)
        return self

    def _recover_center_from_region(self, region, target, *, absorb="right"):
        region = set(region)
        target = self.plan.resolve_site(target)
        if target not in region:
            raise ValueError("target center must lie inside canonical_region")
        remaining = set(region)
        while len(remaining) > 1:
            leaves = [
                q
                for q in remaining
                if q != target and sum(n in remaining for n in self.plan.neighbors(q)) == 1
            ]
            if not leaves:
                raise ValueError("canonical_region is not a connected tree")
            q = min(leaves)
            neighbor = next(n for n in self.plan.neighbors(q) if n in remaining)
            source, destination = (q, neighbor) if absorb == "right" else (neighbor, q)
            self.canonize_edge_(
                source,
                destination,
                absorb=absorb,
                _isometry_proven=self.can_skip_canonize(source, destination, absorb=absorb),
            )
            remaining.remove(q)
        self._canonical_region = frozenset({target})
        self._set_isometry_metadata_from_region({target})
        return self

    def shift_orthogonality_center(self, site, *, absorb="right", info_c=None, **canonize_opts):
        q = self.plan.resolve_site(site)
        current = self.orthogonality_center
        if current == q:
            self._sync_info_c(info_c)
            return self
        if current is None:
            region = self.canonical_region
            if region:
                if q in region:
                    self._recover_center_from_region(region, q, absorb=absorb)
                    self._sync_info_c(info_c)
                    return self
                entry = min(region, key=lambda candidate: len(self.plan.path(candidate, q)))
                self._recover_center_from_region(region, entry, absorb=absorb)
                current = entry
            else:
                return self.canonize_to(
                    q,
                    absorb=absorb,
                    inplace=True,
                    info_c=info_c,
                    _force_full=True,
                    **canonize_opts,
                )
        path = self.plan.path(current, q)
        for site0, site1 in zip(path, path[1:]):
            if absorb == "right":
                source, target = site0, site1
            elif absorb == "left":
                source, target = site1, site0
            else:
                raise ValueError("absorb must be 'left' or 'right'")
            self.canonize_edge_(source, target, absorb=absorb, **canonize_opts)
        self._canonical_region = frozenset({q})
        self._sync_info_c(info_c)
        return self

    def canonize_subtree(
        self,
        sites,
        *,
        span=False,
        absorb="right",
        inplace=False,
        info_c=None,
        **canonize_opts,
    ):
        sites = _normalize_sites(self.plan, sites, name="sites")
        region = self.plan.subtree_span(sites) if span else frozenset(sites)
        if not region or not self.plan.is_connected(region):
            raise ValueError("sites must form a connected subtree, or pass span=True")
        work = self if inplace else self.copy()
        opts = {"method": "qr", "cutoff": 0.0}
        opts.update(canonize_opts)
        work._canonicalize_region_fast(region, absorb=absorb, **opts)
        work._canonical_region = frozenset(region)
        work._set_isometry_metadata_from_region(region)
        work.validate(check_canonical=True)
        work._sync_info_c(info_c)
        return work

    def canonize_subtree_(self, sites, *, span=False, absorb="right", info_c=None, **opts):
        return self.canonize_subtree(
            sites,
            span=span,
            absorb=absorb,
            inplace=True,
            info_c=info_c,
            **opts,
        )

    def compress_edge(
        self,
        site0,
        site1,
        *,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        absorb="right",
        reduced=True,
        inplace=False,
        info_c=None,
        **compress_opts,
    ):
        work = self if inplace else self.copy()
        work.compress_edge_(
            site0,
            site1,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            absorb=absorb,
            reduced=reduced,
            info_c=info_c,
            **compress_opts,
        )
        return work

    def compress_edge_(
        self,
        site0,
        site1,
        *,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        absorb="right",
        reduced=True,
        info_c=None,
        _validate=True,
        **compress_opts,
    ):
        q0, q1 = self._edge_sites(site0, site1)
        if cutoff is None:
            cutoff = 1e-10
        cutoff = float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if max_bond is not None and int(max_bond) < 1:
            raise ValueError("max_bond must be at least one")
        before_bond = self.node_tensor(q0).ind_size(self.bond(q0, q1))
        if cutoff == 0.0 and (max_bond is None or before_bond <= int(max_bond)):
            return self.canonize_edge_(q0, q1, absorb=absorb, info_c=info_c)
        previous = self.orthogonality_center
        qtn.TensorNetworkGenOperator.compress_between(
            self,
            self.node_tag(q0),
            self.node_tag(q1),
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            absorb=absorb,
            reduced=reduced,
            **compress_opts,
        )
        self._track_edge_center(q0, q1, absorb, previous=previous)
        self._sync_info_c(info_c)
        if _validate:
            self.validate()
        return self

    def compress(
        self,
        form=None,
        *,
        center=None,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        reduced=True,
        order="rank",
        info_c=None,
    ):
        """Compress operator bonds inward toward ``center``.

        ``order="rank"`` greedily removes the currently cheapest leaf
        branch using live tensor dimensions, while preserving the fixed
        ``TreePepsPlan`` topology. ``order="depth"`` keeps the simple
        farthest-first schedule for reproducibility and comparison.
        """

        if form is not None:
            if center is not None:
                raise TypeError("specify either form or center, not both")
            if isinstance(form, Integral):
                center = form
            elif form in {"right", "left"}:
                center = self.plan.root
            else:
                raise ValueError("TreePepo form must be None, 'right', 'left', or a site id")
        if center is None:
            center = self.orthogonality_center
            if center is None:
                center = self.plan.root
        center = self.plan.resolve_site(center)
        order = normalize_tree_compression_order(order)
        # TreePEPO direct sums can contain repeated boundary vectors even
        # when no numerical bond cap is requested. Remove those exact dense
        # dependencies before the existing native edge SVD sweep. Non-NumPy
        # data (including native symmetric tensors) is left untouched.
        _structural_compress_tree(
            self,
            root=self.plan.root,
            parent=self.plan.parent,
            children=self.plan.children,
            nodes=self.sites,
            tensor_getter=self.node_tensor,
            bond_getter=self.bond,
            method="auto",
        )
        self.shift_orthogonality_center(
            center,
            info_c=info_c,
            _validate=False,
        )
        # Structural reduction can change live dimensions before the final
        # SVDs. The iterator also recomputes the rank choice after every
        # subsequent edge reduction.
        edge_order = iter_tree_compression_order(
            self.plan,
            center=center,
            nodes=self.sites,
            order=order,
            tensor_getter=self.node_tensor,
            bond_getter=self.bond,
        )
        for q, toward in edge_order:
            self.compress_edge_(
                q,
                toward,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb="right",
                reduced=reduced,
                _validate=False,
            )
        self._canonical_region = frozenset({center})
        self._set_isometry_metadata_from_region({center})
        self.validate(check_canonical=True)
        self._sync_info_c(info_c)
        return self

    def add_operator(
        self,
        other,
        *,
        inplace=False,
        negate=False,
        compress=False,
        _validate=True,
        **compress_opts,
    ):
        """Add another matching ``TreePepo`` exactly by direct sum.

        Compression is opt-in and runs only after the exact operator sum has
        been assembled.  The support and tree-span metadata are recomputed so
        the result remains safe for ``TreeSubPepo`` routing.
        """
        if not isinstance(other, TreePepo):
            raise TypeError("other must be a TreePepo.")
        if self.plan_signature != other.plan_signature:
            raise ValueError("TreePepo operators must use the same TreePepsPlan.")
        if (
            self.sites != other.sites
            or self.input_ind_id != other.input_ind_id
            or self.output_ind_id != other.output_ind_id
            or self._node_tag_id != other._node_tag_id
            or self._operator_bond_id != other._operator_bond_id
        ):
            raise ValueError(
                "TreePepo operators must use matching site, physical-index, "
                "node-tag, and bond layouts."
            )
        network = qtn.tensor_network_ag_sum(
            self,
            other,
            site_tags=self.site_tags,
            negate=negate,
            compress=False,
            inplace=False,
        )
        layout_finder = self.layout_finder or other.layout_finder
        if not isinstance(network, TreePepo):
            network = type(self)(
                network,
                plan=self.plan,
                coord_site_tag_id=self._coord_site_tag_id,
                logical_site_tag_id=self._logical_site_tag_id,
                node_tag_id=self._node_tag_id,
                operator_bond_id=self._operator_bond_id,
                input_ind_id=self._input_ind_id,
                output_ind_id=self._output_ind_id,
                physical_dims=self._physical_dims,
                layout_finder=layout_finder,
            )
        else:
            network.layout_finder = layout_finder
        left_support = self.operator_support
        right_support = other.operator_support
        support = (
            None
            if left_support is None or right_support is None
            else frozenset(left_support) | frozenset(right_support)
        )
        network._operator_support = support
        network._operator_span = (
            None
            if support is None
            else self.plan.subtree_span(support) if support else frozenset()
        )
        network._operator_terms = None
        network._canonical_region = None
        for site in network.sites:
            network.node_tensor(site).modify(left_inds=None)
        if _validate:
            network.validate()
        if compress:
            network.compress(**compress_opts)
        if inplace:
            self.__dict__.clear()
            self.__dict__.update(network.__dict__)
            return self
        return network

    def scale(self, factor, *, inplace=False):
        """Multiply this tree-PEPO operator by a scalar."""
        if not np.isscalar(factor):
            raise TypeError("TreePepo.scale requires a scalar factor.")
        target = self if inplace else self.copy(deep=True)
        tensor = target.node_tensor(target.sites[0])
        tensor.modify(data=tensor.data * factor, left_inds=tensor.left_inds)
        if target._operator_terms is not None:
            target._operator_terms = {
                support: value * factor
                for support, value in target._operator_terms.items()
            }
        target.invalidate_canonical_form()
        return target

    def compose(
        self,
        other,
        *,
        inplace=False,
        compress=False,
        max_bond=None,
        cutoff=1e-10,
        order="rank",
    ):
        """Compose two dense tree-PEPO operators without densifying them.

        The result represents ``self @ other``: ``other`` acts first.  The
        operator-state ``two_layer`` application path remains separate and is
        selected by :meth:`apply_to`, not by this operator product.
        """
        if not isinstance(other, TreePepo):
            raise TypeError("other must be a TreePepo.")
        if self.plan_signature != other.plan_signature:
            raise ValueError("TreePepo operators must use the same TreePepsPlan.")
        if (
            self.sites != other.sites
            or self.input_ind_id != other.input_ind_id
            or self.output_ind_id != other.output_ind_id
            or self._node_tag_id != other._node_tag_id
        ):
            raise ValueError(
                "TreePepo operators must use matching site, physical-index, "
                "and node-tag layouts."
            )
        from ..tree.operators import _compose_tree_operator_network, _network_bond

        network = _compose_tree_operator_network(
            self,
            other,
            nodes=tuple(self.sites),
            edges=tuple(self.plan.tree_edges),
            node_tag=lambda node: self.node_tag(node),
            site_of_node=lambda node: node,
            neighbors=lambda node: self.neighbors(node),
            output_ind=lambda site: self.output_ind(site),
            input_ind=lambda site: self.input_ind(site),
            bond=lambda operator_network, node, neighbor: (
                _network_bond(
                    operator_network,
                    self.node_tag(node),
                    self.node_tag(neighbor),
                )
            ),
        )
        left_support = self.operator_support
        right_support = other.operator_support
        support = (
            None
            if left_support is None or right_support is None
            else frozenset(left_support) | frozenset(right_support)
        )
        result = type(self)(
            network,
            plan=self.plan,
            coord_site_tag_id=self._coord_site_tag_id,
            logical_site_tag_id=self._logical_site_tag_id,
            node_tag_id=self._node_tag_id,
            operator_bond_id=self._operator_bond_id,
            input_ind_id=self._input_ind_id,
            output_ind_id=self._output_ind_id,
            physical_dims=self._physical_dims,
            operator_support=support,
            operator_span=(
                None
                if support is None
                else self.plan.subtree_span(support) if support else frozenset()
            ),
            layout_finder=self.layout_finder or other.layout_finder,
        )
        if compress:
            result.compress(max_bond=max_bond, cutoff=cutoff, order=order)
        if inplace:
            self.__dict__.clear()
            self.__dict__.update(result.__dict__)
            return self
        return result

    def __add__(self, other):
        return self.add_operator(other)

    def __sub__(self, other):
        return self.add_operator(other, negate=True)

    def __neg__(self):
        return self.scale(-1)

    def __mul__(self, factor):
        if not np.isscalar(factor):
            return NotImplemented
        return self.scale(factor)

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __matmul__(self, other):
        return self.compose(other)

    def to_dense(self, *inds_seq, to_qarray=False, **contract_opts):
        if not inds_seq:
            inds_seq = (
                tuple(self.output_ind(q) for q in self.sites),
                tuple(self.input_ind(q) for q in self.sites),
            )
        return qtn.TensorNetwork.to_dense(
            self,
            *inds_seq,
            to_qarray=to_qarray,
            **contract_opts,
        )

    def apply_to(
        self,
        state,
        *,
        inplace=False,
        compress=False,
        center=None,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        reduced=True,
        compression_mode="direct",
        compression_seed=None,
        compression_layout="auto",
        order="rank",
        info_c=None,
        _active_sites=None,
    ):
        """Apply this operator and return a valid ``TreePeps`` state.

        The input physical leg is contracted site-by-site.  At every tree
        edge, the state and operator bonds are then fused into one state bond,
        which prevents the result from becoming a two-graph tensor network.
        Compression is opt-in.  ``compression_layout="fused"`` retains this
        fused application path.  ``"two_layer"`` passes a path-shaped
        operator-state network directly to Quimb's 1D compressor, and
        ``"auto"`` selects that path for Quimb's multi-tensor methods while
        retaining the fused fallback elsewhere.
        """

        from .state import TreePeps
        from .state import _normalize_compression_mode

        if not isinstance(state, TreePeps):
            raise TypeError("TreePepo.apply_to requires a TreePeps state")
        if plan_signature(state.plan) != self.plan_signature:
            raise ValueError("TreePepo and TreePeps must use the same tree plan")
        self.validate()
        state.validate()
        if _active_sites is None:
            active_sites = frozenset(state.sites)
        else:
            active_sites = frozenset(
                _normalize_sites(state.plan, _active_sites, name="active_sites")
            )
            if self.operator_span is None:
                active_sites = frozenset(state.sites)
            elif not self.operator_span.issubset(active_sites):
                raise ValueError("active_sites must contain the complete operator span")
        compression_mode = _normalize_compression_mode(compression_mode)
        compression_layout = _normalize_compression_layout(compression_layout)
        order = normalize_tree_compression_order(order)
        if compression_layout == "two_layer":
            if not state.plan.is_mps_topology:
                raise NotImplementedError(
                    "compression_layout='two_layer' requires a path "
                    "TreePeps topology."
                )
            if compression_mode not in _PATH_TWO_LAYER_COMPRESSION_MODES:
                raise ValueError(
                    "two-layer path compression requires compression_mode in "
                    "{'direct', 'dm', 'sdc', 'src', 'zipup'}"
                )
        use_two_layer = (
            (compress or max_bond is not None)
            and state.plan.is_mps_topology
            and len(active_sites) > 1
            and (
                compression_layout == "two_layer"
                or (
                    compression_layout == "auto"
                    and compression_mode in _AUTO_TWO_LAYER_COMPRESSION_MODES
                )
            )
        )
        if use_two_layer:
            return self._apply_to_path_1d(
                state,
                active_sites=active_sites,
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                compression_seed=compression_seed,
                inplace=inplace,
                info_c=info_c,
            )
        tensors = []
        for q in state.sites:
            if q not in active_sites:
                tensors.append(state.node_tensor(q).copy())
                continue
            state_tensor = state.node_tensor(q)
            operator_tensor = self.node_tensor(q)
            state_phys = state.site_ind(q)
            input_ind = self.input_ind(q)
            output_ind = self.output_ind(q)
            if operator_tensor.ind_size(input_ind) != state_tensor.ind_size(state_phys):
                raise ValueError(f"operator input dimension at site {q} does not match state")
            if operator_tensor.ind_size(output_ind) != state_tensor.ind_size(state_phys):
                raise ValueError(f"operator output dimension at site {q} does not match state")
            state_bonds = tuple(state.bond(q, neighbor) for neighbor in state.neighbors(q))
            operator_bonds = tuple(self.bond(q, neighbor) for neighbor in self.neighbors(q))
            if input_ind != state_phys:
                if output_ind == state_phys or output_ind in state_tensor.inds:
                    raise ValueError("operator physical indices collide with state indices")
                operator_tensor = operator_tensor.reindex({input_ind: state_phys})
                input_ind = state_phys
            if output_ind in state_tensor.inds or set(operator_bonds) & set(state_tensor.inds):
                raise ValueError("operator and state virtual indices collide")
            raw_inds = (output_ind, *state_bonds, *operator_bonds)
            joined = qtn.tensor_contract(
                state_tensor,
                operator_tensor,
                output_inds=raw_inds,
            ).transpose(*raw_inds)
            # The contraction leaves all state bonds followed by all
            # operator bonds.  Interleave the corresponding pair before
            # flattening it into one fused tree bond; a plain reshape here
            # would pair the wrong axes whenever the site has degree > 1.
            fused_inds = (output_ind,) + tuple(
                ind
                for state_bond, operator_bond in zip(state_bonds, operator_bonds)
                for ind in (state_bond, operator_bond)
            )
            joined = joined.transpose(*fused_inds)
            new_bonds = tuple(state.tree_bond_ind(q, neighbor) for neighbor in state.neighbors(q))
            new_shape = (
                joined.ind_size(output_ind),
                *(
                    joined.ind_size(sb) * joined.ind_size(ob)
                    for sb, ob in zip(state_bonds, operator_bonds)
                ),
            )
            data = ar.do("reshape", joined.data, new_shape)
            tensors.append(
                qtn.Tensor(
                    data=data,
                    inds=(state_phys, *new_bonds),
                    tags=state_tensor.tags,
                )
            )

        result = TreePeps(tensors, plan=state.plan)
        result.validate()
        # ``cutoff`` is only meaningful when a compression sweep is requested.
        # In particular, the non-zero default must not silently truncate an
        # exact operator application.
        if compress or max_bond is not None:
            if center is None:
                center = state.orthogonality_center
                if center is None:
                    center = state.plan.root
            result.compress(
                center=center,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                reduced=reduced,
                compression_mode=compression_mode,
                compression_seed=compression_seed,
                order=order,
                info_c=info_c,
            )
        if inplace:
            return _replace_tree_peps(state, result)
        return result

    def _apply_to_path_1d(
        self,
        state,
        *,
        active_sites,
        center,
        max_bond,
        cutoff,
        cutoff_mode,
        compression_mode,
        compression_seed=None,
        inplace=False,
        info_c=None,
    ):
        """Apply and compress an operator-state path as one Quimb 1D TN.

        The state and operator tensors retain separate virtual layers.  Their
        input physical legs are joined with private indices and the operator
        output legs are renamed to the state's physical indices.  Quimb then
        sees the standard MPO-MPS layout: multiple tensors grouped by one site
        tag, with only the state output legs and state boundary bonds open.
        """

        from .state import TreePeps

        if not isinstance(state, TreePeps):
            raise TypeError("state must be a TreePeps")
        if not state.plan.is_mps_topology:
            raise NotImplementedError(
                "two-layer Quimb compression requires a path TreePeps topology"
            )
        if compression_mode not in _PATH_TWO_LAYER_COMPRESSION_MODES:
            raise ValueError(
                "two-layer path compression requires compression_mode in "
                "{'direct', 'dm', 'sdc', 'src', 'zipup'}"
            )
        if cutoff is None:
            cutoff = 1e-10
        cutoff = float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if max_bond is not None:
            max_bond = int(max_bond)
            if max_bond < 1:
                raise ValueError("max_bond must be at least one")
        if compression_mode in {"sdc", "src"} and max_bond is None:
            if not (compression_mode == "sdc" and cutoff == 0.0):
                raise ValueError(
                    f"compression_mode={compression_mode!r} requires a finite "
                    "max_bond/chi"
                )

        active_sites = frozenset(
            state.plan.resolve_site(site) for site in active_sites
        )
        if not active_sites or not state.plan.is_connected(active_sites):
            raise ValueError("active_sites must form a connected path region")
        endpoints = sorted(
            site
            for site in active_sites
            if sum(
                neighbor in active_sites
                for neighbor in state.plan.neighbors(site)
            ) <= 1
        )
        if len(endpoints) != 2:
            raise ValueError(
                "two-layer path compression requires a non-trivial path region"
            )
        order = state.plan.path(endpoints[0], endpoints[1])
        if set(order) != set(active_sites):
            raise ValueError("active_sites must be a connected path region")

        tensors = []
        for site in order:
            state_tensor = state.node_tensor(site).copy()
            operator_tensor = self.node_tensor(site).copy()
            state_physical = state.site_ind(site)
            input_link = qtn.rand_uuid()
            # A custom TreePepo may use a different coordinate-tag format.
            # Add the state's canonical site tag explicitly so both layers
            # are grouped by the same Quimb 1D site selector.
            operator_tensor.add_tag(state.site_tag(site))
            state_tensor.reindex_({state_physical: input_link})
            operator_tensor.reindex_(
                {
                    self.input_ind(site): input_link,
                    self.output_ind(site): state_physical,
                }
            )

            # The selected TreePepo span may have identity operator bonds at
            # its boundary.  They are not part of the result state and must be
            # sliced away before the temporary 1D network is compressed.
            for neighbor in state.plan.neighbors(site):
                if neighbor in active_sites:
                    continue
                operator_bond = self.bond(site, neighbor)
                if operator_tensor.ind_size(operator_bond) != 1:
                    raise ValueError(
                        "a two-layer path application requires operator bonds "
                        "outside the active span to have dimension one"
                    )
                operator_tensor = operator_tensor.isel({operator_bond: 0})
            tensors.extend((state_tensor, operator_tensor))

        temporary = qtn.TensorNetwork(tensors)
        temporary.exponent = (
            float(getattr(state, "exponent", 0.0))
            + float(getattr(self, "exponent", 0.0))
        )
        compressor = quimb_1d_compression_function(compression_mode)
        if not callable(compressor):
            raise NotImplementedError(
                f"Quimb compression method {compression_mode!r} is not "
                "available in the installed build"
            )
        options = {
            "max_bond": max_bond,
            "cutoff": cutoff,
            "site_tags": [state.site_tag(site) for site in order],
            "permute_arrays": False,
            "canonize": True,
            "inplace": False,
        }
        if compression_mode in {"direct", "dm", "sdc", "zipup"}:
            options["cutoff_mode"] = cutoff_mode
        if compression_mode == "src" and compression_seed is not None:
            options["seed"] = int(compression_seed)
        result = compressor(temporary, **options)

        work = state if inplace else state.copy()
        for site in order:
            tag = state.site_tag(site)
            tids = tuple(result.tag_map[tag])
            if len(tids) != 1:
                raise ValueError(
                    f"path compressor did not preserve one tensor for site {site}"
                )
            compressed = result.tensor_map[tids[0]]
            work.node_tensor(site).modify(
                data=compressed.data,
                inds=compressed.inds,
                tags=work.node_tensor(site).tags,
                left_inds=compressed.left_inds,
            )
        work.exponent = getattr(result, "exponent", temporary.exponent)
        work._canonical_region = None
        if center is None:
            center = state.orthogonality_center
            if center is None:
                center = state.plan.root
        center = state.plan.resolve_site(center)
        work.canonize_to(
            center,
            inplace=True,
            info_c=info_c,
            _force_full=True,
        )
        work.validate(check_canonical=True)
        if inplace:
            return work
        return work

    def expectation(self, state, *, normalized=True, **apply_opts):
        """Evaluate ``<state|self|state>`` through the application boundary."""

        from .state import TreePeps

        if not isinstance(state, TreePeps):
            raise TypeError("TreePepo.expectation requires a TreePeps state")
        result = self.apply_to(state, compress=False, **apply_opts)
        # ``TreePeps.H`` already carries the same physical labels as the
        # ket.  Reindexing those legs would leave the bra and ket disconnected
        # and turn the requested overlap into a product of open tensors.
        bra = state.H
        numerator = (bra & result).contract(all, output_inds=[])
        if not normalized:
            return numerator
        return numerator / state.norm(squared=True)

    def ascii_lattice(self, *, bond_dims=True, node_ids=False):
        """Return a PEPO-style coordinate drawing of the retained bonds.

        The layout follows Quimb's ``PEPO.show`` convention: lattice sites are
        laid out in rows, operator physical legs are shown with Unicode
        diagonals, and only bonds present in the retained tree are drawn.
        """

        def label(q):
            return f"●N{q}" if node_ids else "●"

        def edge_dim(site0, site1):
            try:
                bond = self.bond(site0, site1)
            except ValueError:
                return None
            return self.node_tensor(site0).ind_size(bond)

        if self.ndim != 2:
            return "\n".join(
                f"{self.coordinate(q)} {label(q)}"
                for q in self.sites
            )
        coord_to_q = {self.plan.coordinate(q): q for q in self.sites}
        if node_ids:
            label_width = max(len(label(q)) for q in self.sites)
        else:
            label_width = 1

        # Keep every site and every edge on a fixed character grid.  The
        # previous implementation concatenated labels and edge fragments of
        # different widths, which made a missing tree bond shift everything to
        # its right.  This mirrors Quimb's ``show_2d`` spacing, while allowing
        # a tree to omit arbitrary lattice neighbours.
        dim_width = 3
        if bond_dims:
            dimensions = []
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    q = coord_to_q[(x, y)]
                    if y + 1 < self.shape[1]:
                        dim = edge_dim(q, coord_to_q[(x, y + 1)])
                        if dim is not None:
                            dimensions.append(len(str(dim)))
                    if x + 1 < self.shape[0]:
                        dim = edge_dim(q, coord_to_q[(x + 1, y)])
                        if dim is not None:
                            dimensions.append(len(str(dim)))
            dim_width = max(dim_width, max(dimensions, default=3))
        step = max(label_width, 1) + dim_width + 1
        site_x = lambda y: 1 + y * step
        # The lower physical leg is one character to the right of the final
        # site's virtual-bond column, as in Quimb's PEPO schematic.
        line_width = site_x(self.shape[1] - 1) + max(label_width, 2)

        def make_line():
            return [" "] * line_width

        def put(line, position, value):
            for offset, char in enumerate(str(value)):
                if position + offset < len(line):
                    line[position + offset] = char

        def put_dim(line, position, dim):
            if not bond_dims or dim is None:
                return
            put(line, position, f"{dim:^{dim_width}}")

        def render_upper(x):
            """Render upper operator legs and horizontal bond dimensions."""
            line = make_line()
            for y in range(self.shape[1]):
                q = coord_to_q[(x, y)]
                put(line, site_x(y), "╱")
                if y + 1 < self.shape[1]:
                    right = coord_to_q[(x, y + 1)]
                    put_dim(line, site_x(y) + 1, edge_dim(q, right))
            return "".join(line).rstrip()

        def render_sites(x):
            """Render sites and only the retained horizontal tree bonds."""
            line = make_line()
            for y in range(self.shape[1]):
                q = coord_to_q[(x, y)]
                put(line, site_x(y), label(q))
                if y + 1 < self.shape[1]:
                    right = coord_to_q[(x, y + 1)]
                    if edge_dim(q, right) is not None:
                        put(line, site_x(y) + label_width, "━" * (step - label_width))
            return "".join(line).rstrip()

        def render_vertical(x):
            """Render lower physical legs and vertical tree bonds."""
            line = make_line()
            for y in range(self.shape[1]):
                q = coord_to_q[(x, y)]
                put(line, site_x(y) - 1, "╱")
                if x + 1 < self.shape[0]:
                    down = coord_to_q[(x + 1, y)]
                    dim = edge_dim(q, down)
                    if dim is not None:
                        put(line, site_x(y), "┃")
                        put_dim(line, site_x(y) + 1, dim)
            return "".join(line).rstrip()

        def render_lower(x):
            """Render lower-row upper legs and horizontal dimensions."""
            line = make_line()
            for y in range(self.shape[1]):
                q = coord_to_q[(x + 1, y)]
                down_x = site_x(y)
                if edge_dim(coord_to_q[(x, y)], q) is not None:
                    put(line, down_x, "┃")
                put(line, down_x + 1, "╱")
                if y + 1 < self.shape[1]:
                    right = coord_to_q[(x + 1, y + 1)]
                    put_dim(line, down_x + 2, edge_dim(q, right))
            return "".join(line).rstrip()

        rows = []
        for x in range(self.shape[0]):
            rows.extend((render_upper(x), render_sites(x)))
            if x + 1 < self.shape[0]:
                rows.extend((render_vertical(x), render_lower(x)))
        final = make_line()
        for y in range(self.shape[1]):
            put(final, site_x(y) - 1, "╱")
        rows.append("".join(final).rstrip())
        return "\n".join(rows)

    def show(
        self,
        *,
        bond_dims=True,
        node_ids=False,
        layout="lattice",
        color=False,
        **_,
    ):
        """Print the PEPO-style lattice view or the native tree view."""
        layout = str(layout).strip().lower().replace("-", "_")
        if layout in {"tree", "plan"}:
            drawing = self.ascii_tree(
                bond_dims=bond_dims,
                node_ids=node_ids,
                color=color,
            )
        elif layout in {"lattice", "grid", "coordinates"}:
            drawing = self.ascii_lattice(
                bond_dims=bond_dims,
                node_ids=node_ids,
            )
        else:
            raise ValueError("layout must be 'tree' or 'lattice'.")
        print(drawing)

    def ascii_tree(self, *, bond_dims=True, node_ids=False, color=False):
        """Return a compact tree-topology drawing of the operator."""
        return ascii_tree(
            self.plan,
            lambda node, child: self.bond_size(node, child),
            bond_dims=bond_dims,
            node_ids=node_ids,
            color=color,
            marker="●",
            leaf_marker="◆",
            label_site=lambda site: f"q{site}",
        )

    def __repr__(self):
        return (
            f"TreePepo(shape={self.shape!r}, sites={self.plan.size}, "
            f"tree_edges={len(self.plan.tree_edges)})"
        )


class TreeSubPepo:
    """A support/span-aware operator fragment for a ``TreePeps`` update."""

    def __init__(self, operator: TreePepo, support, *, span=None):
        if not isinstance(operator, TreePepo):
            raise TypeError("TreeSubPepo requires a TreePepo operator")
        support = _normalize_sites(operator.plan, support, name="support")
        if span is None:
            span = operator.plan.subtree_span(support)
        else:
            span = frozenset(operator.plan.resolve_site(site) for site in span)
        if not span or not set(support).issubset(span) or not operator.plan.is_connected(span):
            raise ValueError("span must be a connected tree region containing support")
        if operator.operator_support is not None and not set(support).issubset(
            operator.operator_support
        ):
            raise ValueError("operator does not contain all requested support sites")
        self._operator = operator
        self._support = tuple(support)
        self._span = frozenset(span)

    @classmethod
    def from_operator(cls, plan_or_operator, operator=None, support=None, **operator_opts):
        """Wrap a ``TreePepo`` or build one for a local dense operator.

        The plan-first form is ``from_operator(plan, array, support)``.  An
        already-built operator can be wrapped more tersely as
        ``from_operator(operator, support=support)``.
        """

        if isinstance(plan_or_operator, TreePepo):
            if operator is not None and support is None:
                # Also accept ``from_operator(operator, support)``.
                support = operator
            elif operator is not None:
                raise TypeError("support was supplied twice")
            if support is None:
                raise TypeError("support is required")
            tree_operator = plan_or_operator.copy()
        else:
            plan = plan_or_operator
            if operator is None:
                raise TypeError("operator is required")
            if support is None:
                raise TypeError("support is required")
            tree_operator = TreePepo.from_operator(
                plan,
                operator,
                support,
                **operator_opts,
            )
        return cls(tree_operator, support)

    from_dense = from_operator
    from_gate = from_operator

    @classmethod
    def from_product(cls, plan, operators, support=None, **operator_opts):
        tree_operator = TreePepo.from_product(plan, operators, **operator_opts)
        if support is None:
            support = tree_operator.operator_support
        if support is None or not tuple(support):
            raise ValueError("TreeSubPepo.from_product requires non-empty support")
        return cls(tree_operator, support)

    @property
    def operator(self):
        return self._operator

    @property
    def plan(self):
        return self._operator.plan

    @property
    def map_mode(self):
        return self.plan.map_mode

    @property
    def layout_finder(self):
        """The layout finder carried by the wrapped tree operator."""

        return self.operator.layout_finder

    @property
    def plan_signature(self):
        return self._operator.plan_signature

    @property
    def support(self):
        return self._support

    @property
    def span(self):
        return self._span

    @property
    def boundary_edges(self):
        return tuple(
            (inside, outside)
            for inside in sorted(self._span)
            for outside in self.plan.neighbors(inside)
            if outside not in self._span
        )

    @property
    def attachment_map(self):
        return {
            inside: tuple(outside for left, outside in self.boundary_edges if left == inside)
            for inside in sorted(self._span)
            if any(left == inside for left, _ in self.boundary_edges)
        }

    @property
    def operator_bond_dims(self):
        return {
            edge: self.operator.bond_sizes()[edge]
            for edge in self.operator.bond_sizes()
            if edge[0] in self._span and edge[1] in self._span
        }

    def to_dense(self, *args, **kwargs):
        return self.operator.to_dense(*args, **kwargs)

    def validate(self, **kwargs):
        self.operator.validate(**kwargs)
        return self

    def apply_to(self, state, **kwargs):
        return self.operator.apply_to(state, **kwargs)

    def expectation(self, state, **kwargs):
        return self.operator.expectation(state, **kwargs)

    def copy(self):
        return type(self)(self.operator.copy(), self.support, span=self.span)

    def __repr__(self):
        return (
            f"TreeSubPepo(support={self.support!r}, span={tuple(sorted(self.span))!r}, "
            f"plan={self.plan.shape!r})"
        )


# Acronym-preserving spellings are the canonical public names.  Keep the
# original mixed-case names as aliases because they are already part of the
# Pepsy API and appear in existing tree-PEPS optimizer streams.
TreePEPO = TreePepo
TreeSubPEPO = TreeSubPepo


def plan_signature(plan):
    """Return the structural identity used for state/operator compatibility."""

    if not isinstance(plan, TreePepsPlan):
        raise TypeError("plan must be a TreePepsPlan")
    return (
        plan.shape,
        plan.coordinates,
        plan.tree_edges,
        plan.root,
        plan.max_virtual_degree,
        plan.order,
        plan.tree_order,
        plan.topology,
        plan.boundary,
    )


def _format_for_plan(prefix, plan):
    return prefix + ",".join("{}" for _ in range(plan.ndim))


def _normalize_sites(plan, sites, *, name):
    if isinstance(sites, Integral):
        sites = (sites,)
    try:
        sites = tuple(sites)
    except TypeError as exc:
        raise TypeError(f"{name} must be a site or iterable of sites") from exc
    if not sites:
        raise ValueError(f"{name} cannot be empty")
    result = tuple(plan.resolve_site(site) for site in sites)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain distinct sites")
    return result


def _site_dimensions(plan, dimensions, *, name):
    if isinstance(dimensions, Integral):
        if int(dimensions) < 1:
            raise ValueError(f"{name} must be positive")
        return {q: int(dimensions) for q in range(plan.size)}
    if isinstance(dimensions, Mapping):
        result = {}
        for site, dimension in dimensions.items():
            q = plan.resolve_site(site)
            if not isinstance(dimension, Integral) or int(dimension) < 1:
                raise ValueError(f"{name} entries must be positive integers")
            result[q] = int(dimension)
        if set(result) != set(range(plan.size)):
            raise ValueError(f"{name} mapping must specify every site")
        return result
    dimensions = tuple(dimensions)
    if len(dimensions) != plan.size or any(
        not isinstance(dimension, Integral) or int(dimension) < 1 for dimension in dimensions
    ):
        raise ValueError(f"{name} must be one integer or one positive integer per site")
    return {q: int(dimension) for q, dimension in enumerate(dimensions)}


def _operator_dimensions(plan, support, dimensions):
    """Resolve support dimensions and use dimension two for passive sites."""

    if isinstance(dimensions, Integral):
        if int(dimensions) < 1:
            raise ValueError("dims must be positive")
        return {q: int(dimensions) for q in range(plan.size)}
    if isinstance(dimensions, Mapping):
        result = {q: 2 for q in range(plan.size)}
        for site, dimension in dimensions.items():
            q = plan.resolve_site(site)
            if not isinstance(dimension, Integral) or int(dimension) < 1:
                raise ValueError("dims entries must be positive integers")
            result[q] = int(dimension)
        if any(q not in result for q in support):
            raise ValueError("dims must specify every support site")
        return result
    values = tuple(dimensions)
    if len(values) == plan.size:
        if any(not isinstance(dim, Integral) or int(dim) < 1 for dim in values):
            raise ValueError("dims entries must be positive integers")
        return {q: int(dim) for q, dim in enumerate(values)}
    if len(values) != len(support) or any(
        not isinstance(dim, Integral) or int(dim) < 1 for dim in values
    ):
        raise ValueError("dims must be one integer, one value per site, or one per support site")
    result = {q: 2 for q in range(plan.size)}
    result.update(zip(support, (int(dim) for dim in values)))
    return result


def _normalize_operator(operator, support, *, dims, dtype):
    if hasattr(operator, "to_dense") and not isinstance(operator, np.ndarray):
        operator = operator.to_dense()
    data = np.asarray(operator, dtype=dtype)
    rank = len(support)
    if data.ndim == 2:
        if isinstance(dims, Integral):
            site_dims = (int(dims),) * rank
        elif isinstance(dims, Mapping):
            site_dims = tuple(int(dims[site]) for site in support)
        else:
            site_dims = tuple(int(dim) for dim in dims)
        expected = int(np.prod(site_dims, dtype=int))
        if len(site_dims) != rank or data.shape != (expected, expected):
            raise ValueError(
                f"operator shape {data.shape} does not match support dimensions {site_dims}"
            )
        data = data.reshape((*site_dims, *site_dims))
    elif data.ndim == 2 * rank:
        output_dims = tuple(int(dim) for dim in data.shape[:rank])
        input_dims = tuple(int(dim) for dim in data.shape[rank:])
        if output_dims != input_dims:
            raise ValueError("TreePepo currently requires matching input/output dimensions")
        site_dims = output_dims
    else:
        raise ValueError(f"a {rank}-site operator must have rank {2 * rank} or be a matrix")
    if not np.issubdtype(data.dtype, np.inexact):
        data = data.astype(float)
    return data, site_dims


def _edge_name(plan, q0, q1, formatter):
    if formatter is None:
        formatter = "_tppo{}_{}"
    return formatter.format(*sorted((q0, q1)))


def _input_ind(plan, q, formatter=None):
    return (formatter or _format_for_plan("k", plan)).format(*plan.coordinate(q))


def _output_ind(plan, q, formatter=None):
    return (formatter or _format_for_plan("b", plan)).format(*plan.coordinate(q))


def _site_tags(plan, q, formatter=None):
    coordinate = plan.coordinate(q)
    tags = [
        (formatter or _format_for_plan("I", plan)).format(*coordinate),
        f"I{q}",
        f"X{coordinate[0]}",
        f"Y{coordinate[1]}",
        f"N{q}",
    ]
    if plan.ndim == 3:
        tags.insert(4, f"Z{coordinate[2]}")
    return tuple(tags)


def _build_dense_operator_network(
    plan,
    data,
    support,
    site_dims,
    physical_dims=None,
    *,
    coord_site_tag_id=None,
    input_ind_id=None,
    output_ind_id=None,
    operator_bond_id=None,
):
    support = tuple(support)
    span = plan.subtree_span(support)
    packed_inds = tuple(f"_tppo_pack_{qtn.rand_uuid()}_{q}" for q in support)
    interleaved = data.transpose(
        [axis for site in range(len(support)) for axis in (site, len(support) + site)]
    ).reshape(tuple(dim * dim for dim in site_dims))
    blob = qtn.Tensor(interleaved, inds=packed_inds)

    remaining = set(span)
    peel_order = []
    while len(remaining) > 1:
        leaves = [
            q
            for q in remaining
            if sum(neighbor in remaining for neighbor in plan.neighbors(q)) == 1
        ]
        if not leaves:
            raise ValueError("operator span is not a connected tree")
        leaf = min(leaves)
        neighbor = next(n for n in plan.neighbors(leaf) if n in remaining)
        peel_order.append((leaf, neighbor))
        remaining.remove(leaf)
    hub = next(iter(remaining))

    owned = {q: set() for q in span}
    for q, packed in zip(support, packed_inds):
        owned[q].add(packed)
    factors = {}
    raw_bonds = {}
    real_dtype = np.asarray(data).real.dtype
    if not np.issubdtype(real_dtype, np.inexact):
        real_dtype = np.dtype(float)
    structural_cutoff = 64.0 * np.finfo(real_dtype).eps
    for leaf, neighbor in peel_order:
        left_inds = tuple(ind for ind in blob.inds if ind in owned[leaf])
        if not left_inds:
            raise RuntimeError(f"operator decomposition lost payload at site {leaf}")
        raw_bond = f"_tppo_raw_{qtn.rand_uuid()}"
        left, blob = blob.split(
            left_inds=left_inds,
            method="svd",
            absorb="right",
            cutoff=structural_cutoff,
            get="tensors",
            bond_ind=raw_bond,
        )
        factors[leaf] = left
        raw_bonds[(leaf, neighbor)] = raw_bond
        owned[neighbor].add(raw_bond)
    factors[hub] = blob

    def edge_name(q0, q1):
        return _edge_name(plan, q0, q1, operator_bond_id)

    raw_to_live = {raw: edge_name(q0, q1) for (q0, q1), raw in raw_bonds.items()}
    support_dims = dict(zip(support, site_dims))
    if physical_dims is None:
        physical_dims = support_dims
    tensors = []
    for q in range(plan.size):
        neighbors = plan.neighbors(q)
        if q in factors:
            factor = factors[q]
            tensor_data = np.asarray(factor.data)
            inds = [raw_to_live.get(ind, ind) for ind in factor.inds]
            if q in support_dims:
                packed = packed_inds[support.index(q)]
                axis = inds.index(packed)
                dim = support_dims[q]
                shape = list(tensor_data.shape)
                shape[axis : axis + 1] = [dim, dim]
                tensor_data = tensor_data.reshape(shape)
                inds[axis : axis + 1] = [
                    _output_ind(plan, q, output_ind_id),
                    _input_ind(plan, q, input_ind_id),
                ]
            else:
                dim = physical_dims[q]
                tensor_data = np.einsum(
                    "ab,...->ab...", np.eye(dim, dtype=tensor_data.dtype), tensor_data
                )
                inds = [
                    _output_ind(plan, q, output_ind_id),
                    _input_ind(plan, q, input_ind_id),
                    *inds,
                ]
            existing = set(inds)
            for neighbor in neighbors:
                edge = edge_name(q, neighbor)
                if edge not in existing:
                    tensor_data = np.expand_dims(tensor_data, axis=-1)
                    inds.append(edge)
                    existing.add(edge)
        else:
            dim = physical_dims[q]
            tensor_data = np.zeros((dim, dim) + (1,) * len(neighbors), dtype=data.dtype)
            tensor_data[(slice(None), slice(None)) + (0,) * len(neighbors)] = np.eye(
                dim, dtype=data.dtype
            )
            inds = [
                _output_ind(plan, q, output_ind_id),
                _input_ind(plan, q, input_ind_id),
                *(edge_name(q, neighbor) for neighbor in neighbors),
            ]
        desired = (
            _output_ind(plan, q, output_ind_id),
            _input_ind(plan, q, input_ind_id),
            *(edge_name(q, neighbor) for neighbor in neighbors),
        )
        tensors.append(
            qtn.Tensor(
                tensor_data,
                inds=inds,
                tags=_site_tags(plan, q, coord_site_tag_id),
            ).transpose(*desired)
        )
    return qtn.TensorNetwork(tensors)


def _replace_tree_peps(target, source):
    """Replace a ``TreePeps`` tensor set while preserving object identity."""

    target.tensor_map = {}
    target.tag_map = {}
    target.ind_map = {}
    target._inner_inds = qtn.oset()
    target._outer_inds = qtn.oset()
    target._tid_counter = 0
    target.exponent = source.exponent
    target._canonical_region = source._canonical_region
    target.__dict__.pop("_tree_peps_tid_cache", None)
    for tensor in source.tensor_map.values():
        target.add_tensor(tensor.copy(), virtual=False)
    return target
