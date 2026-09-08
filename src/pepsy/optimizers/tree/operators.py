"""Tree-plan-aware native fermionic operator construction.

``SymHamiltonian.to_mpo`` remains the explicit model-level conversion to an
ordinary chain MPO. A chain's Jordan--Wigner wire, however, is not an
embedding of a branched fermionic tree. This module constructs only the native
tree operator, without applying it to the state. Exact tree readout contracts
that operator between a private bra and ket copy.

Two routes are provided:

* general neutral native term sums are decomposed term-by-term on the
  TreePlan Steiner subtrees and amalgamated into one direct-sum TTNO;
* a rank-one pair correlator with separable coefficients is compiled into a
  four-state endpoint automaton.  This is the compact route for the full
  staggered eta-pair observable and keeps its tree bond independent of the
  lattice size.

``TreeMPO`` is a Quimb ``TensorNetworkGenOperator`` over the TreePlan geometry,
analogous to ``TreeTensorNetwork`` being a ``TensorNetworkGenVector``. It has
one representation: the native TreePlan operator network.
"""

from __future__ import annotations

import heapq
from numbers import Integral

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...operators._structural_compression import _structural_compress_tree
from .layout import TreeLayoutFinder, TreePlan
from ._display import ascii_lattice, ascii_tree

__all__ = ["TreeMPO", "build_tree_operator"]


def _as_numpy(data, *, dtype=None):
    """Convert a dense backend array to host NumPy construction data."""
    if hasattr(data, "to_dense"):
        data = data.to_dense()
    return np.asarray(ar.to_numpy(data), dtype=dtype)


def _tree_plan_signature(plan):
    """Return a stable structural signature for a tree-MPO annotation."""
    return (
        int(plan.root),
        tuple(
            (int(node), tuple(int(child) for child in children))
            for node, children in sorted(plan.children.items())
        ),
        tuple(
            (int(node), int(qubit))
            for node, qubit in sorted(plan.qubit_of_leaf.items())
        ),
        None if plan.root_qubit is None else int(plan.root_qubit),
    )


def _tree_node_selector(plan, selector, node_tag_id="N{}"):
    """Resolve one public tree-node selector to a structural node id."""
    if isinstance(selector, str):
        for node in plan.nodes():
            if selector == node_tag_id.format(node):
                return node
        if selector.startswith("N"):
            selector = selector[1:]
        else:
            raise ValueError(
                "TreeMPO geometry selectors must be node ids or the configured "
                "structural node tag."
            )
    try:
        node = int(selector)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"invalid TreeMPO node selector {selector!r}.") from exc
    if node not in plan.children:
        raise ValueError(f"{node!r} is not a TreePlan node.")
    return node


def _tree_region_selector(plan, selector, node_tag_id="N{}"):
    """Resolve one or more node selectors to a connected TreePlan region."""
    if isinstance(selector, (tuple, list, set, frozenset)):
        nodes = tuple(
            _tree_node_selector(plan, node, node_tag_id)
            for node in selector
        )
    else:
        nodes = (_tree_node_selector(plan, selector, node_tag_id),)
    if not nodes:
        raise ValueError("TreeMPO canonical regions cannot be empty.")
    region = set(nodes)
    for node in nodes[1:]:
        region.update(plan.node_path(nodes[0], node))
    return frozenset(region)


def _tree_subtree_span(plan, nodes):
    """Return the minimal connected node set spanning ``nodes``."""
    nodes = tuple(nodes)
    if not nodes:
        raise ValueError("need at least one tree node to span a subtree.")
    region = {nodes[0]}
    for node in nodes[1:]:
        region.update(plan.node_path(nodes[0], node))
    return frozenset(region)


class TreeMPO(qtn.TensorNetworkGenOperator):
    """TreePlan-aware operator with dense and native Symmray backends.

    ``TreeMPO`` is the operator-level API for measurements on a
    :class:`TreeTensorNetwork`. It subclasses Quimb's generalized operator
    network, so common methods such as ``sites``, ``site_tag``, ``upper_ind``,
    ``lower_ind``, ``to_dense``, ``H``, and ``copy`` operate on its native
    TreePlan networks. It does not store or create a second chain
    representation; use the model-level ``to_mpo(...)`` methods when a chain
    MPO is explicitly required.

    ``tree_networks``
        One or more operator tensor networks whose physical indices are
        labelled by the logical qubits in ``plan``. General native terms are
        combined into one Symmray TTNO whose graded source/target channels
        preserve the fermionic contraction rules.

    General neutral native sums use one direct-sum TTNO. Each term is first
    factorized on its native graded TreePlan subtree, then all term channels
    are amalgamated on common charge-aware virtual bonds. The resulting
    operator can be canonicalized and compressed with native graded QR/SVD.
    Structured observables such as the eta-pair table may use a smaller
    compact network instead.
    """

    # Match Quimb's generalized operator API while retaining the additional
    # TreePlan-native metadata owned by this class.
    _EXTRA_PROPS = qtn.TensorNetworkGenOperator._EXTRA_PROPS + (
        "_plan",
        "_node_tag_id",
        "_operator_support",
        "_pepsy_backend",
        "tree_networks",
        "_canonical_region",
        "terms",
        "fermionic",
        "symmetry",
        "cutoff",
        "compressed",
        "_layout_finder",
    )

    def __init__(
        self,
        plan=None,
        tree_networks=None,
        *,
        terms=None,
        backend="dense",
        fermionic=False,
        symmetry=None,
        cutoff=1e-12,
        compressed=False,
        sites=None,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
        operator_support=None,
        layout_finder=None,
        virtual=True,
        deep=False,
    ):
        if isinstance(plan, TreeMPO) and tree_networks is None:
            source = plan
            plan = source.plan
            networks = tuple(
                network.copy(virtual=virtual, deep=deep)
                for network in source.tree_networks
            )
            terms = source.terms
            backend = source.backend
            fermionic = source.fermionic
            symmetry = source.symmetry
            cutoff = source.cutoff
            compressed = source.compressed
            sites = source.sites
            site_tag_id = source.site_tag_id
            upper_ind_id = source.upper_ind_id
            lower_ind_id = source.lower_ind_id
            node_tag_id = source.node_tag_id
            operator_support = source.operator_support
            layout_finder = source.layout_finder
        elif tree_networks is None:
            raise TypeError("TreeMPO requires a tree operator network.")
        else:
            networks = (
                tuple(tree_networks)
                if isinstance(tree_networks, (tuple, list))
                else (tree_networks,)
            )

        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan.")
        if not networks or any(network is None for network in networks):
            raise ValueError("TreeMPO requires at least one tree operator network.")
        # The first network is the primary generalized tree operator. Use a
        # virtual Quimb view so inherited operator methods such as
        # ``sites``, ``upper_ind``, ``lower_ind``, ``to_dense``, ``bond``, and
        # ``H`` operate on the same tensors as ``tree_networks[0]``.
        super().__init__(networks[0], virtual=True)
        self._plan = plan
        self.tree_networks = networks
        self.terms = None if terms is None else dict(terms)
        self._pepsy_backend = str(backend)
        self.fermionic = bool(fermionic)
        self.symmetry = symmetry
        self.cutoff = float(cutoff)
        self.compressed = bool(compressed)
        self._sites = (
            tuple(sorted(plan.node_of_qubit)) if sites is None else tuple(sites)
        )
        self._site_tag_id = site_tag_id
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._node_tag_id = node_tag_id
        self._canonical_region = None
        if operator_support is None:
            self._operator_support = None
        else:
            normalized_support = frozenset(
                int(site) for site in operator_support
            )
            if not normalized_support:
                raise ValueError("operator_support cannot be empty.")
            if not normalized_support.issubset(plan.node_of_qubit):
                raise ValueError(
                    "operator_support must contain only TreePlan physical sites."
                )
            self._operator_support = normalized_support
        self.pepsy_tree_plan_signature = _tree_plan_signature(plan)
        self.layout_finder = layout_finder
        for network in networks:
            # Native QR/SVD helpers need the structural tag format even when
            # callers choose a non-default ``node_tag_id``. Keep it on each
            # stored network so copies and backend conversions retain the
            # geometry contract without hard-coding ``N{node}``.
            network.pepsy_tree_node_tag_id = node_tag_id
    @property
    def backend(self):
        """Return the logical Pepsy backend label for this operator."""
        return self._pepsy_backend

    @property
    def pepsy_backend(self):
        """Compatibility view used by Quimb's structured-network copier."""
        return self._pepsy_backend

    @property
    def plan(self):
        """The :class:`TreePlan` describing the operator geometry."""
        return self._plan

    @property
    def layout_finder(self):
        """The optional layout finder carrying lattice and term metadata."""

        return self._layout_finder

    @layout_finder.setter
    def layout_finder(self, finder):
        if finder is not None:
            if not isinstance(finder, TreeLayoutFinder):
                raise TypeError("layout_finder must be a TreeLayoutFinder or None.")
            if finder.n != self.plan.n:
                raise ValueError(
                    "layout_finder and TreeMPO must describe the same number "
                    "of physical sites."
                )
            if (
                finder.root_qubit is not None
                and finder.root_qubit != self.plan.root_qubit
            ):
                raise ValueError(
                    "layout_finder and TreeMPO must use the same root_qubit."
                )
            shape = finder.lattice_shape
            if shape is not None and int(np.prod(shape, dtype=int)) != self.plan.n:
                raise ValueError(
                    "layout_finder lattice_shape must cover every TreeMPO site."
                )
        self._layout_finder = finder

    @property
    def map_mode(self):
        """Canonical geometric label for the operator's tree layout."""

        return self.plan.map_mode

    @property
    def node_tag_id(self):
        """Format string for structural tree-node tags."""
        return self._node_tag_id

    @property
    def site_ind_id(self):
        """Alias for the operator's upper physical-index format."""
        return self.upper_ind_id

    @site_ind_id.setter
    def site_ind_id(self, value):
        self.upper_ind_id = value

    def site_ind(self, site):
        """Return the ket-like physical index for ``site``."""
        return self.upper_ind(site)

    @property
    def root(self):
        """The structural root node id."""
        return self.plan.root

    @property
    def canonical_region(self):
        """The currently canonicalized connected operator region."""
        return self._canonical_region

    @property
    def operator_support(self):
        """Logical sites with known non-identity operator support.

        This is an application optimization hint. The stored TreeMPO remains
        complete and still carries explicit identity legs outside this set.
        ``None`` means that a conservative full-tree route is required.
        """
        if self._operator_support is None:
            return None
        return tuple(sorted(self._operator_support))

    @property
    def orthogonality_center(self):
        """The single canonical node, or ``None`` for a larger region."""
        region = self.canonical_region
        return next(iter(region)) if region is not None and len(region) == 1 else None

    @property
    def fermionic(self):
        """Whether the operator stores native fermionic arrays."""
        return self._fermionic

    @fermionic.setter
    def fermionic(self, value):
        self._fermionic = bool(value)

    @property
    def symmetry(self):
        """Native Symmray symmetry label, if present."""
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value):
        self._symmetry = value

    @classmethod
    def from_hamiltonian(
        cls,
        plan,
        hamiltonian,
        *,
        cutoff=1e-12,
        max_bond=None,
        compress=True,
        dtype=None,
        fermionic=True,
        layout_finder=None,
    ):
        """Construct a ``TreeMPO`` from a ``SymHamiltonian``."""
        from ...tensors.symmetric import SymHamiltonian

        if not isinstance(hamiltonian, SymHamiltonian):
            raise TypeError("hamiltonian must be a SymHamiltonian instance.")
        networks = _build_tree_operator(
            plan,
            hamiltonian,
            cutoff=cutoff,
            max_bond=max_bond,
            compress=compress,
            dtype=dtype,
            fermionic=fermionic,
        )
        if isinstance(networks, (tuple, list)):
            native_networks = tuple(networks)
        else:
            native_networks = (networks,)
        backend = "symmray" if fermionic else "dense"
        operator = cls(
            plan,
            native_networks,
            terms=hamiltonian.terms,
            backend=backend,
            fermionic=fermionic,
            symmetry=hamiltonian.symmetry,
            cutoff=cutoff,
            compressed=compress,
            layout_finder=layout_finder,
        )
        if compress:
            operator.compress(max_bond=max_bond, cutoff=cutoff)
        return operator

    @classmethod
    def from_terms(
        cls,
        plan,
        terms,
        *,
        cutoff=1e-12,
        dtype=None,
        max_bond=None,
        compress=True,
        layout_finder=None,
    ):
        """Construct one ordinary dense TTNO from a term mapping.

        ``terms`` maps an integer site or support tuple to a dense operator
        array. The dense route is useful for non-fermionic trees and for
        callers that already have Jordan--Wigner-compatible local matrices.
        Supports of any rank are accepted. Higher-order terms are factored
        exactly over their minimal TreePlan Steiner subtrees and combined by
        a virtual direct sum, so the result remains one tensor per TreePlan
        node and can use the normal TTNO canonicalization/compression API.
        """
        if not hasattr(terms, "items"):
            raise TypeError("terms must be a mapping of supports to operators.")
        normalized_terms = {
            _term_support(where): term
            for where, term in terms.items()
        }
        if any(len(support) > 2 for support in normalized_terms):
            # Dense higher-order terms cannot be represented by the compact
            # one-/two-site channel automaton. Factor every term over its
            # minimal Steiner subtree, then direct-sum the resulting TTNOs.
            network = _direct_sum_dense_tnno(
                tuple(
                    _dense_tree_term_tnno(
                        plan,
                        term,
                        support,
                        dtype=dtype,
                    )
                    for support, term in normalized_terms.items()
                ),
                plan,
                dtype=dtype,
            )
        else:
            network = _combined_tree_operator(
                plan,
                normalized_terms,
                symmetry=None,
                cutoff=cutoff,
                dtype=dtype,
                fermionic=False,
            )
        operator = cls(
            plan,
            network,
            terms=terms,
            backend="dense",
            fermionic=False,
            cutoff=cutoff,
            compressed=compress,
            layout_finder=layout_finder,
        )
        if compress:
            operator.compress(max_bond=max_bond, cutoff=cutoff)
        return operator

    @classmethod
    def from_gate(
        cls,
        plan,
        gate,
        where,
        *,
        dims=2,
        fermionic=False,
        symmetry=None,
        dtype=None,
        cutoff=0.0,
        max_bond=None,
        compress=False,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
        layout_finder=None,
    ):
        """Build a complete TreeMPO for a local gate support.

        ``where`` contains logical qubit labels, not positions in a chain
        layout. The gate is factorized only over the minimal TreePlan Steiner
        subtree joining those sites; bond-one identity tensors are installed
        on the remaining TreePlan nodes. Thus the result can be passed
        directly to :meth:`TreeOptimizer.apply_subtreempo` without building a
        ``2**n`` operator or introducing a fictitious contiguous MPS window.

        Dense gates may be supplied as a square matrix or as a tensor with
        output legs followed by input legs. Native Symmray gates require
        ``fermionic=True`` and preserve their charge-aware factorization.
        Gate factorization is lossless by default; ``compress=True`` applies
        the optional operator-side ``max_bond``/``cutoff`` sweep.
        """
        support = _term_support(where)
        if any(site not in plan.node_of_qubit for site in support):
            raise ValueError(
                f"gate support {support!r} is outside the supplied TreePlan."
            )
        if fermionic:
            if symmetry is None:
                symmetry = getattr(gate, "symmetry", None)
            if symmetry is None:
                raise TypeError(
                    "native TreeMPO.from_gate requires symmetry= or a gate "
                    "with a Symmray symmetry attribute."
                )
            network = _native_tree_term_network(
                plan,
                gate,
                support,
                symmetry=symmetry,
                cutoff=0.0,
                dtype=dtype,
            )
            backend = "symmray"
        else:
            data = _dense_operator_array(gate, dtype=dtype)
            if data.ndim == 2:
                if isinstance(dims, Integral):
                    gate_dims = (int(dims),) * len(support)
                else:
                    gate_dims = tuple(int(dim) for dim in dims)
                if len(gate_dims) != len(support):
                    raise ValueError(
                        "dims must contain one physical dimension per gate "
                        "support site."
                    )
                matrix_dim = int(np.prod(gate_dims, dtype=int))
                if data.shape != (matrix_dim, matrix_dim):
                    raise ValueError(
                        "gate matrix shape does not match the supplied dims: "
                        f"got {data.shape}, expected {(matrix_dim, matrix_dim)}."
                    )
                data = data.reshape((*gate_dims, *gate_dims))
            network = _dense_tree_term_tnno(
                plan,
                data,
                support,
                dtype=dtype,
            )
            backend = "dense"
        _relabel_tree_operator_network(
            network,
            plan,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )

        operator = cls(
            plan,
            network,
            backend=backend,
            fermionic=bool(fermionic),
            symmetry=symmetry,
            cutoff=cutoff,
            compressed=False,
            sites=tuple(sorted(plan.node_of_qubit)),
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
            operator_support=support,
            layout_finder=layout_finder,
        )
        if compress:
            operator.compress(max_bond=max_bond, cutoff=cutoff)
        return operator

    from_operator = from_gate

    @classmethod
    def from_pauli_sum(
        cls,
        plan,
        weighted_terms,
        *,
        dtype=complex,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
        layout_finder=None,
    ):
        """Build a compact TreeMPO for a weighted Pauli-product sum.

        ``weighted_terms`` contains ``(coefficient, {site: axis})`` pairs.
        One virtual channel is used per retained branch, but channels are
        installed only on the union of the branches' TreePlan Steiner
        subtrees. Exterior tensors remain explicit bond-one identities, so
        the result can use the support-aware :meth:`TreeOptimizer` route
        without constructing a ``2**n`` dense matrix or a chain MPO.
        """
        network, support = _pauli_sum_tree_operator(
            plan,
            weighted_terms,
            dtype=dtype,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )
        return cls(
            plan,
            network,
            backend="dense",
            fermionic=False,
            sites=tuple(sorted(plan.node_of_qubit)),
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
            operator_support=support,
            layout_finder=layout_finder,
        )

    @classmethod
    def from_dense(
        cls,
        plan,
        array=None,
        dims=2,
        *,
        tree=None,
        sites=None,
        tags=None,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
        layout_finder=None,
        **split_opts,
    ):
        """Build an exact tree operator from a dense matrix.

        This is the tree analogue of ``MatrixProductOperator.from_dense``.
        The matrix is decomposed over the supplied ``TreePlan`` with lossless
        leaf-to-root SVDs. Only the physical site ordering differs from the
        chain constructor: ``sites`` labels the plan's logical qubits.
        """
        if not isinstance(plan, TreePlan):
            if tree is None:
                raise TypeError("pass a TreePlan with `tree=` or as the first argument.")
            if array is not None:
                raise TypeError("dense array was supplied more than once.")
            array = plan
            plan = tree
        elif tree is not None and tree is not plan:
            raise ValueError("plan and tree specify different TreePlans.")
        if array is None:
            raise TypeError("TreeMPO.from_dense requires a dense matrix.")
        if not isinstance(plan, TreePlan):
            raise TypeError("plan must be a TreePlan.")
        if sites is None:
            sites = tuple(sorted(plan.node_of_qubit))
        else:
            sites = tuple(int(site) for site in sites)
        if sites != tuple(sorted(sites)):
            raise ValueError("TreeMPO.from_dense requires sorted site labels.")
        if set(sites) != set(plan.node_of_qubit):
            raise ValueError(
                "TreeMPO.from_dense currently requires one matrix site per tree site."
            )
        if isinstance(dims, Integral):
            dims = (int(dims),) * len(sites)
        else:
            dims = tuple(int(dim) for dim in dims)
        if len(dims) != len(sites):
            raise ValueError("dims must have one entry per TreePlan site.")
        if np.prod(dims, dtype=int) ** 2 != np.size(array):
            raise ValueError("array size does not match the supplied physical dims.")
        network = _tree_operator_from_dense(
            plan,
            array,
            sites=sites,
            dims=dims,
            split_opts=split_opts,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )
        if tags is not None:
            network.add_tag(tags)
        return cls(
            plan,
            network,
            backend="dense",
            fermionic=False,
            sites=sites,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
            layout_finder=layout_finder,
        )

    @classmethod
    def from_fill_fn(
        cls,
        fill_fn,
        plan,
        bond_dim,
        *,
        phys_dim=2,
        dtype=float,
        sites=None,
        tags=None,
        site_tag_id="I{}",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        node_tag_id="N{}",
        layout_finder=None,
    ):
        """Build a tree operator from a tensor filling function.

        ``fill_fn`` is called as ``fill_fn(shape)`` for each plan node, where
        ``shape`` is ordered as physical upper/lower legs followed by the
        node's tree bonds. A scalar ``bond_dim`` or one value per edge is
        accepted through the same uniform tree convention.
        """
        if sites is None:
            sites = tuple(sorted(plan.node_of_qubit))
        else:
            sites = tuple(sites)
        network = _tree_operator_from_fill_fn(
            plan,
            fill_fn,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            dtype=dtype,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
        )
        if tags is not None:
            network.add_tag(tags)
        return cls(
            plan,
            network,
            backend="dense",
            fermionic=False,
            sites=sites,
            site_tag_id=site_tag_id,
            upper_ind_id=upper_ind_id,
            lower_ind_id=lower_ind_id,
            node_tag_id=node_tag_id,
            layout_finder=layout_finder,
        )

    @classmethod
    def rand(
        cls,
        plan,
        bond_dim,
        *,
        phys_dim=2,
        dtype=complex,
        seed=None,
        **operator_opts,
    ):
        """Build a random dense TreeMPO with uniform virtual bond size."""
        rng = np.random.default_rng(seed)

        def fill(shape):
            if np.issubdtype(np.dtype(dtype), np.complexfloating):
                return (
                    rng.standard_normal(shape)
                    + 1j * rng.standard_normal(shape)
                ).astype(dtype)
            return rng.standard_normal(shape).astype(dtype)

        return cls.from_fill_fn(
            fill,
            plan,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            dtype=dtype,
            **operator_opts,
        )

    @property
    def tree_network(self):
        """Return the sole tree network, or raise for a term sum."""
        if len(self.tree_networks) != 1:
            raise AttributeError(
                "this TreeMPO contains multiple internal networks; use "
                "tree_networks or expectation()"
            )
        return self.tree_networks[0]

    @property
    def nqubits(self):
        """Number of logical physical sites in the TreePlan."""
        return self.plan.n

    @property
    def top_arity(self):
        """Number of virtual child bonds at the structural root."""
        return self.plan.top_arity

    @property
    def max_virtual_degree(self):
        """Largest number of virtual tree bonds on one operator tensor."""
        return self.plan.max_virtual_degree()

    @property
    def max_tensor_rank(self):
        """Largest virtual/physical leg count on one operator tensor."""
        return self.plan.max_tensor_rank()

    def node_tag(self, node):
        """Return the structural tag for a TreePlan node."""
        return self._node_tag_id.format(int(node))

    def node_tensor(self, node):
        """Return a primary TTNO tensor by TreePlan node id."""
        return self.tree_networks[0][self.node_tag(node)]

    def _select_tids(self, tids, virtual=True, with_exponent=False):
        """Select a structured view while keeping its primary network live."""
        selected = super()._select_tids(
            tids,
            virtual=virtual,
            with_exponent=with_exponent,
        )
        # Quimb's generic ``new(like=...)`` copies the extra properties from
        # the source, including ``tree_networks``. Replace that source tuple
        # with the selected view so inherited selection methods never mutate
        # or inspect the original operator by accident.
        selected.tree_networks = (qtn.TensorNetwork(selected, virtual=True),)
        return selected

    def neighbors(self, node):
        """Return the TreePlan neighbors of a structural node."""
        node = int(node)
        if node not in self.plan.children:
            raise ValueError(f"{node!r} is not a TreePlan node.")
        return tuple(self.plan.children[node]) + (
            (self.plan.parent[node],)
            if self.plan.parent.get(node) is not None
            else ()
        )

    def is_leaf(self, node):
        """Whether ``node`` is a structural leaf."""
        return self.plan.is_leaf(int(node))

    def parent(self, node):
        """Return the parent structural node, or ``None`` at the root."""
        node = int(node)
        if node not in self.plan.children:
            raise ValueError(f"{node!r} is not a TreePlan node.")
        return self.plan.parent.get(node)

    def children(self, node):
        """Return the structural children of ``node``."""
        node = int(node)
        if node not in self.plan.children:
            raise ValueError(f"{node!r} is not a TreePlan node.")
        return self.plan.children[node]

    def node_path(self, node1, node2):
        """Return the inclusive structural path between two nodes."""
        return self.plan.node_path(int(node1), int(node2))

    def leaf_of_qubit(self, qubit):
        """Return the structural leaf carrying ``qubit``."""
        return self.plan.leaf_of_qubit[int(qubit)]

    def qubit_of_leaf(self, node):
        """Return the qubit carried by a structural leaf."""
        return self.plan.qubit_of_leaf[int(node)]

    def qubit_of_node(self, node):
        """Return the qubit carried by a node, or ``None`` if virtual."""
        return self.plan.qubit_of_node.get(int(node))

    def node_of_qubit(self, qubit):
        """Return the structural node carrying ``qubit``."""
        return self.plan.node_of_qubit[int(qubit)]

    def tree_distance(self, qubit1, qubit2):
        """Return the structural distance between two physical sites."""
        return self.plan.tree_distance(int(qubit1), int(qubit2))

    def steiner_nodes(self, nodes):
        """Return the minimal connected subtree spanning ``nodes``."""
        return self.plan.steiner_nodes(tuple(int(node) for node in nodes))

    def subtree_span(self, nodes):
        """Return the minimal connected subtree spanning arbitrary nodes."""
        return _tree_subtree_span(
            self.plan, tuple(int(node) for node in nodes),
        )

    def is_binary(self, *, allow_ternary_root=True):
        """Whether this operator's tree is binary below its root."""
        return self.plan.is_binary(allow_ternary_root=allow_ternary_root)

    def bond(self, node, neighbor):
        """Return the live operator bond between adjacent TreePlan nodes."""
        node = int(node)
        neighbor = int(neighbor)
        if neighbor not in self.neighbors(node):
            raise ValueError(
                f"nodes {node} and {neighbor} are not adjacent in the tree."
            )
        shared = qtn.bonds(
            self.node_tensor(node), self.node_tensor(neighbor),
        )
        if len(shared) != 1:
            raise ValueError(
                f"nodes {node} and {neighbor} must share exactly one bond; "
                f"found {sorted(shared)}."
            )
        return next(iter(shared))

    def validate(self, *, check_canonical=False):
        """Validate every stored TTNO against its TreePlan geometry.

        Virtual bond names are intentionally not part of the contract: an
        externally assembled TTNO may use arbitrary labels as long as each
        label belongs to exactly one TreePlan edge. ``check_canonical=True``
        additionally checks the tracked ``left_inds`` orientation for every
        stored network.
        """
        expected_outer = {
            self.upper_ind(site) for site in self.sites
        } | {
            self.lower_ind(site) for site in self.sites
        }
        region = self.canonical_region
        if region is not None:
            region = frozenset(region)
            if not region.issubset(set(self.plan.nodes())):
                raise ValueError("TreeMPO canonical region contains unknown nodes.")
            if _tree_subtree_span(self.plan, region) != region:
                raise ValueError("TreeMPO canonical region is not connected.")

        for network in self.tree_networks:
            node_tids = {}
            for node in self.plan.nodes():
                tids = tuple(network.tag_map.get(self.node_tag(node), ()))
                if len(tids) != 1:
                    raise ValueError(
                        f"TreeMPO node {node} must have exactly one tensor "
                        f"tagged {self.node_tag(node)!r}; found {len(tids)}."
                    )
                node_tids[node] = tids[0]
            if set(node_tids.values()) != set(network.tensor_map):
                raise ValueError(
                    "TreeMPO tensor set disagrees with its TreePlan nodes."
                )

            edge_bonds = {}
            for parent, children in self.plan.children.items():
                for child in children:
                    shared = qtn.bonds(
                        network.tensor_map[node_tids[parent]],
                        network.tensor_map[node_tids[child]],
                    )
                    if len(shared) != 1:
                        raise ValueError(
                            f"TreeMPO edge ({parent}, {child}) must have "
                            f"exactly one live bond; found {sorted(shared)!r}."
                        )
                    edge_bonds[frozenset((parent, child))] = next(iter(shared))

            for node, tid in node_tids.items():
                tensor = network.tensor_map[tid]
                expected_inds = {
                    edge_bonds[frozenset((node, neighbor))]
                    for neighbor in self.neighbors(node)
                }
                physical = self.plan.qubit_of_node.get(node)
                if physical is not None:
                    expected_inds.update((
                        self.upper_ind(physical),
                        self.lower_ind(physical),
                    ))
                if set(tensor.inds) != expected_inds:
                    raise ValueError(
                        f"TreeMPO node {node} has unexpected indices: "
                        f"{tensor.inds!r}."
                    )
            if set(network.outer_inds()) != expected_outer:
                raise ValueError("TreeMPO has unexpected outer physical indices.")

            if check_canonical and region is not None:
                for node, tid in node_tids.items():
                    tensor = network.tensor_map[tid]
                    if node in region:
                        continue
                    left_inds = tensor.left_inds
                    if left_inds is None:
                        raise ValueError(
                            f"TreeMPO node {node} is missing canonical left_inds."
                        )
                    right_inds = set(tensor.inds) - set(left_inds)
                    if len(right_inds) != 1:
                        raise ValueError(
                            f"TreeMPO node {node} has malformed canonical "
                            f"left_inds {left_inds!r}."
                        )
                    expected = min(
                        (
                            self.plan.node_path(node, target)
                            for target in region
                        ),
                        key=len,
                    )[1]
                    if next(iter(right_inds)) != edge_bonds[
                        frozenset((node, expected))
                    ]:
                        raise ValueError(
                            f"TreeMPO node {node} is not canonical toward "
                            f"node {expected}."
                        )
        return self

    def max_bond(self):
        """Return the largest virtual bond among the tree networks."""
        bonds = []
        for network in self.tree_networks:
            for index in network.inner_inds():
                bonds.append(network.ind_size(index))
        return max(bonds, default=1)

    def bond_size(self, node, neighbor):
        """Return the dimension of one live operator tree bond."""
        return self.node_tensor(node).ind_size(self.bond(node, neighbor))

    def bond_sizes(self):
        """Return operator bond dimensions in deterministic tree-edge order."""
        return tuple(
            self.bond_size(node, child)
            for node in self.plan.nodes()
            for child in self.plan.children[node]
        )

    def edge_nodes(self):
        """Return all directed parent-child tree edges."""
        return tuple(
            (node, child)
            for node in self.plan.nodes()
            for child in self.plan.children[node]
        )

    @property
    def L(self):
        """Number of logical physical sites, as in a chain MPO."""
        return self.nsites

    @property
    def cyclic(self):
        """TreeMPOs are open tree networks, never cyclic chains."""
        return False

    def to_dense(self, *inds_seq, to_qarray=False, **contract_opts):
        """Contract the complete operator, summing internal term networks."""
        if len(self.tree_networks) == 1 and not self.fermionic:
            return qtn.TensorNetworkGenOperator.to_dense(
                self,
                *inds_seq,
                to_qarray=to_qarray,
                **contract_opts,
            )
        if not inds_seq:
            inds_seq = (self.upper_inds_present, self.lower_inds_present)
        values = []
        for network in self.tree_networks:
            if self.fermionic:
                # Symmray's block-sparse contraction assumes a neutral scalar
                # when it closes all internal legs. A charged operator has
                # nonzero open physical charge, so densify each local block
                # first and contract the ordinary tree network. This is only
                # the explicit ``to_dense`` escape hatch; native expectation
                # and compression remain graded and factorized.
                dense_tensors = []
                for tensor in network:
                    data = tensor.data
                    if hasattr(data, "to_dense"):
                        data = data.to_dense()
                    dense_tensors.append(qtn.Tensor(
                        data,
                        inds=tensor.inds,
                        tags=tensor.tags,
                    ))
                network = qtn.TensorNetwork(dense_tensors)
            view = qtn.TensorNetworkGenOperator(
                network,
                virtual=True,
            )
            view._sites = self.sites
            view._site_tag_id = self.site_tag_id
            view._upper_ind_id = self.upper_ind_id
            view._lower_ind_id = self.lower_ind_id
            values.append(view.to_dense(*inds_seq, **contract_opts))
        result = values[0]
        for value in values[1:]:
            result = result + value
        if to_qarray:
            import quimb as qu

            return qu.qarray(result)
        return result

    def identity(self, *, phys_dim=None, dtype=None):
        """Return the exact bond-one identity TreeMPO on this plan."""
        if self.fermionic:
            if phys_dim is not None:
                raise ValueError(
                    "phys_dim is inferred from a native TreeMPO's physical "
                    "charge maps; do not override it."
                )
            if dtype is None:
                dtype = self.dtype
            network = _native_identity_tree_operator(
                self.plan,
                self.tree_networks[0],
                symmetry=self.symmetry,
                upper_ind_id=self.upper_ind_id,
                lower_ind_id=self.lower_ind_id,
                site_tag_id=self.site_tag_id,
                node_tag_id=self.node_tag_id,
                dtype=dtype,
            )
            return type(self)(
                self.plan,
                network,
                backend=self.backend,
                fermionic=True,
                symmetry=self.symmetry,
                cutoff=self.cutoff,
                sites=self.sites,
                site_tag_id=self.site_tag_id,
                upper_ind_id=self.upper_ind_id,
                lower_ind_id=self.lower_ind_id,
                node_tag_id=self.node_tag_id,
            )
        if phys_dim is None:
            phys_dim = tuple(self.phys_dim(site) for site in self.sites)
        if dtype is None:
            dtype = self.dtype
        network = _identity_tree_operator(
            self.plan,
            phys_dim=phys_dim,
            dtype=dtype,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
        )
        return type(self)(
            self.plan,
            network,
            backend="dense",
            fermionic=False,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
            layout_finder=self.layout_finder,
        )

    def add_TreeMPO(
        self,
        other,
        inplace=False,
        negate=False,
        compress=False,
        **compress_opts,
    ):
        """Add another matching native ``TreeMPO`` by TTNO direct sum."""
        if not isinstance(other, TreeMPO):
            raise TypeError("other must be a TreeMPO.")
        if self.pepsy_tree_plan_signature != other.pepsy_tree_plan_signature:
            raise ValueError("TreeMPOs must use the same TreePlan.")
        if self.fermionic != other.fermionic:
            raise TypeError("cannot add dense and native TreeMPOs.")
        if (
            self.sites != other.sites
            or self.upper_ind_id != other.upper_ind_id
            or self.lower_ind_id != other.lower_ind_id
            or self.node_tag_id != other.node_tag_id
        ):
            raise ValueError(
                "TreeMPOs must use matching site, physical-index, and "
                "node-tag layouts before they can be added."
            )
        if self.fermionic and self.symmetry != other.symmetry:
            raise TypeError(
                "native TreeMPOs must use the same symmetry, got "
                f"{self.symmetry!r} and {other.symmetry!r}."
            )

        if compress:
            unsupported = set(compress_opts).difference(
                {"max_bond", "cutoff", "order"},
            )
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise TypeError(
                    "TreeMPO addition compression supports only max_bond and "
                    f"cutoff, got {names}."
                )

        if self.fermionic:
            # Symmray arrays cannot be padded by Quimb's generic direct-sum
            # helper when an axis contains multiple charge sectors. Group
            # complete TTNO networks by their open operator charge and build
            # each group with the TreePlan-aware native block direct sum.
            grouped = {}
            for source, sign in ((self, 1), (other, -1 if negate else 1)):
                for network in source.tree_networks:
                    charge = _tree_operator_charge(network, self.plan)
                    grouped.setdefault(charge, []).append((network, sign))
            networks = tuple(
                _native_tree_operator_sum(
                    tuple(network for network, _ in components),
                    self.plan,
                    symmetry=self.symmetry,
                    signs=tuple(sign for _, sign in components),
                    dtype=self.dtype,
                    upper_ind_id=self.upper_ind_id,
                    lower_ind_id=self.lower_ind_id,
                    site_tag_id=self.site_tag_id,
                    node_tag_id=self.node_tag_id,
                )
                for components in grouped.values()
            )
        else:
            if len(self.tree_networks) != len(other.tree_networks):
                raise ValueError("TreeMPO term-network counts must match.")
            networks = tuple(
                qtn.tensor_network_ag_sum(
                    left,
                    right,
                    site_tags=tuple(
                        self.node_tag(node) for node in self.plan.nodes()
                    ),
                    negate=negate,
                    # ``TensorNetwork`` has no generic ``compress`` method in
                    # Quimb. Addition first forms the direct-sum TTNO here,
                    # then uses the native tree SVD sweep below when requested.
                    compress=False,
                )
                for left, right in zip(self.tree_networks, other.tree_networks)
            )
        terms = None
        if self.terms is not None and other.terms is not None:
            terms = dict(self.terms)
            for support, value in other.terms.items():
                if support in terms:
                    terms[support] = terms[support] + ((-1) if negate else 1) * value
                else:
                    terms[support] = ((-1) if negate else 1) * value
        result = type(self)(
            self.plan,
            tuple(networks),
            terms=terms,
            backend=self.backend,
            fermionic=self.fermionic,
            symmetry=self.symmetry,
            cutoff=self.cutoff,
            compressed=compress,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
            # A direct-sum/addition can carry non-trivial virtual channels
            # through identity-only exterior nodes. Do not claim that the
            # union of the operand supports is a safe minimal route unless a
            # future rank-aware contraction proves those boundary channels
            # are bond one.
            operator_support=None,
            layout_finder=self.layout_finder or other.layout_finder,
        )
        if compress:
            result.compress(
                max_bond=compress_opts.get("max_bond"),
                cutoff=compress_opts.get("cutoff"),
                order=compress_opts.get("order", "rank"),
            )
        if inplace:
            self.__dict__.clear()
            self.__dict__.update(result.__dict__)
            return self
        return result

    def add_MPO(self, other, **kwargs):
        """Compatibility wrapper for :meth:`add_TreeMPO`."""
        return self.add_TreeMPO(other, **kwargs)

    def add_operator(
        self,
        other,
        *,
        inplace=False,
        negate=False,
        compress=False,
        **compress_opts,
    ):
        """Add another matching tree operator.

        Addition is exact by default.  Set ``compress=True`` to perform an
        explicit native tree SVD after forming the direct-sum operator.
        """
        return self.add_TreeMPO(
            other,
            inplace=inplace,
            negate=negate,
            compress=compress,
            **compress_opts,
        )

    def scale(self, factor, *, inplace=False):
        """Multiply this tree operator by a scalar."""
        if not np.isscalar(factor):
            raise TypeError("TreeMPO.scale requires a scalar factor.")
        target = self if inplace else self.copy(deep=True)
        if not target.tree_networks:
            raise ValueError("cannot scale a TreeMPO without stored networks.")
        for network in target.tree_networks:
            tensor = next(iter(network))
            tensor.modify(data=tensor.data * factor, left_inds=tensor.left_inds)
        if target.terms is not None:
            target.terms = {
                support: value * factor
                for support, value in target.terms.items()
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
        cutoff=None,
        order="rank",
    ):
        """Compose two dense tree operators without densifying them.

        The result represents ``self @ other``: ``other`` acts first.  The
        local physical legs are contracted and each pair of operator bonds is
        fused on the same TreePlan edge.  Compression is explicit because
        composition can increase every virtual bond.  Charge-aware native
        Symmray composition needs a graded Kronecker/fusion kernel and is
        rejected until that kernel is available.
        """
        if not isinstance(other, TreeMPO):
            raise TypeError("other must be a TreeMPO.")
        if self.pepsy_tree_plan_signature != other.pepsy_tree_plan_signature:
            raise ValueError("TreeMPOs must use the same TreePlan.")
        if self.fermionic or other.fermionic:
            raise NotImplementedError(
                "TreeMPO composition for native fermionic operators requires "
                "a graded fused-bond product; use addition or apply operators "
                "sequentially for now."
            )
        if self.sites != other.sites:
            raise ValueError("TreeMPOs must use matching logical site layouts.")
        if (
            self.upper_ind_id != other.upper_ind_id
            or self.lower_ind_id != other.lower_ind_id
            or self.node_tag_id != other.node_tag_id
        ):
            raise ValueError(
                "TreeMPOs must use matching physical-index and node-tag layouts."
            )
        if len(self.tree_networks) != 1 or len(other.tree_networks) != 1:
            raise NotImplementedError(
                "TreeMPO composition currently requires one dense network per operator."
            )

        network = _compose_tree_operator_network(
            self.tree_networks[0],
            other.tree_networks[0],
            nodes=tuple(self.plan.nodes()),
            edges=tuple(
                (node, child)
                for node, children in self.plan.children.items()
                for child in children
            ),
            node_tag=lambda node: self.node_tag(node),
            site_of_node=lambda node: self.plan.qubit_of_node.get(node),
            neighbors=lambda node: self.neighbors(node),
            output_ind=lambda site: self.upper_ind(site),
            input_ind=lambda site: self.lower_ind(site),
            bond=lambda operator_network, node, neighbor: _network_bond(
                operator_network,
                self.node_tag(node),
                self.node_tag(neighbor),
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
            self.plan,
            network,
            backend="dense",
            fermionic=False,
            cutoff=self.cutoff,
            compressed=False,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
            operator_support=support,
            layout_finder=self.layout_finder,
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

    add_TreeMPO_ = lambda self, other, **kwargs: self.add_TreeMPO(  # noqa: E731
        other, inplace=True, **kwargs,
    )
    add_MPO_ = lambda self, other, **kwargs: self.add_TreeMPO(  # noqa: E731
        other, inplace=True, **kwargs,
    )

    def matrix_element(self, bra, ket=None):
        """Return ``<bra|TreeMPO|ket>`` for computational-basis strings."""
        if ket is None:
            ket = bra
        bra = tuple(int(value) for value in bra)
        ket = tuple(int(value) for value in ket)
        if len(bra) != self.nsites or len(ket) != self.nsites:
            raise ValueError("basis configurations must match TreeMPO.nsites.")
        selector = {}
        for site, bra_value, ket_value in zip(self.sites, bra, ket):
            selector[self.upper_ind(site)] = bra_value
            selector[self.lower_ind(site)] = ket_value
        value = 0.0
        for network in self.tree_networks:
            value = value + network.isel(selector).contract(all)
        return value

    def amplitude(self, configuration):
        """Return the diagonal computational-basis matrix element."""
        return self.matrix_element(configuration)

    def singular_values(self, node, neighbor=None, *, method="svd"):
        """Return singular values across one tree operator edge."""
        if neighbor is None:
            try:
                node, neighbor = node
            except (TypeError, ValueError) as exc:
                raise TypeError("singular_values needs an operator edge.") from exc
        node = _tree_node_selector(self.plan, node, self.node_tag_id)
        neighbor = _tree_node_selector(
            self.plan, neighbor, self.node_tag_id,
        )
        if neighbor not in self.neighbors(node):
            raise ValueError("singular_values requires adjacent tree nodes.")
        work = self.copy()
        work.canonicalize(center=neighbor)
        tensor = work.node_tensor(node)
        bond = work.bond(node, neighbor)
        return tensor.singular_values(
            tuple(ind for ind in tensor.inds if ind != bond),
            method=method,
        )

    def rand_state(self, bond_dim, **state_opts):
        """Return a random :class:`TreeTensorNetwork` on the same plan."""
        from .ttn import TreeTensorNetwork

        return TreeTensorNetwork.rand(self.plan, D=bond_dim, **state_opts)

    def _layout_site_coords(self):
        """Return physical-site coordinates carried by the layout finder."""
        finder = self.layout_finder
        if finder is None or finder.lattice_shape is None:
            raise ValueError(
                "TreeMPO has no 2D/3D layout metadata; pass a TreeLayoutFinder "
                "with lattice_shape= when constructing it."
            )
        shape = finder.lattice_shape
        site = finder.lattice_site
        if site is None:
            if len(shape) == 2:
                site = lambda x, y: x * shape[1] + y
            else:
                site = lambda x, y, z: x * shape[1] * shape[2] + y * shape[2] + z
        if len(shape) == 2:
            coords = {
                int(site(x, y)): (x, y)
                for x in range(shape[0])
                for y in range(shape[1])
            }
        else:
            coords = {
                int(site(x, y, z)): (x, y, z)
                for x in range(shape[0])
                for y in range(shape[1])
                for z in range(shape[2])
            }
        if set(coords) != set(self.plan.node_of_qubit):
            raise ValueError(
                "layout_finder site coordinates do not match the TreeMPO plan."
            )
        return coords

    def ascii_lattice(self, *, node_ids=False):
        """Return the physical lattice view supplied by ``layout_finder``."""
        finder = self.layout_finder
        if finder is None:
            raise ValueError(
                "TreeMPO.ascii_lattice requires a TreeLayoutFinder; construct "
                "the operator with layout_finder=."
            )
        return ascii_lattice(
            self.plan,
            finder.lattice_shape,
            self._layout_site_coords(),
            terms=(
                self.terms
                if self.terms is not None
                else {support: None for support in finder.supports}
            ),
            node_ids=node_ids,
        )

    def plot_layout(self, **plot_opts):
        """Plot the retained tree over the physical lattice and term graph."""
        finder = self.layout_finder
        if finder is None:
            raise ValueError(
                "TreeMPO.plot_layout requires a TreeLayoutFinder with lattice "
                "metadata."
            )
        plot_opts.setdefault("show_site_labels", True)
        plot_opts.setdefault("show_node_ids", False)
        plot_opts.setdefault("show_gate_connectivity", bool(self.terms))
        plot_opts.setdefault("site_coords", self._layout_site_coords())
        return finder.plot(self.plan, **plot_opts)

    def ascii_tree(self, *, bond_dims=True, node_ids=False, color=False):
        """Return a compact Quimb-style drawing of the operator tree."""
        return ascii_tree(
            self.plan,
            lambda node, child: self.bond_size(node, child),
            bond_dims=bond_dims,
            node_ids=node_ids,
            color=color,
            label_site=lambda site: f"q{site}",
        )

    def show(
        self,
        *,
        bond_dims=True,
        node_ids=False,
        color=False,
        layout="tree",
    ):
        """Show the clean native ASCII tree by default.

        ``layout="lattice"`` or ``layout="both"`` opt into the physical
        coordinate view supplied by a ``TreeLayoutFinder``. ``layout="auto"``
        remains an explicit convenience alias that selects both views when
        lattice metadata is available. ``layout="plot"`` returns and displays
        the Matplotlib layout figure; use :meth:`plot_layout` directly when
        further customization is needed.
        """
        layout = str(layout).strip().lower().replace("-", "_")
        if layout == "auto":
            layout = (
                "both"
                if (
                    self.layout_finder is not None
                    and self.layout_finder.lattice_shape is not None
                )
                else "tree"
            )
        if layout in {"tree", "plan"}:
            print(self.ascii_tree(
                bond_dims=bond_dims,
                node_ids=node_ids,
                color=color,
            ))
            return None
        if layout in {"lattice", "grid", "coordinates"}:
            print(self.ascii_lattice(node_ids=node_ids))
            return None
        if layout in {"both", "all"}:
            print(self.ascii_lattice(node_ids=node_ids))
            print()
            print(self.ascii_tree(
                bond_dims=bond_dims,
                node_ids=node_ids,
                color=color,
            ))
            return None
        if layout in {"plot", "figure", "tent"}:
            figure, _axes = self.plot_layout(
                show_node_ids=node_ids,
            )
            figure.show()
            return figure
        raise ValueError(
            "layout must be 'tree', 'auto', 'lattice', 'both', or 'plot'."
        )

    def canonicalize(self, center=None, *, inplace=True, info_c=None):
        """Canonicalize every stored TTNO around one TreePlan node.

        This is the tree equivalent of an MPO mixed-canonical gauge. The
        default is inplace, matching Quimb's MPO canonicalization methods;
        pass ``inplace=False`` to obtain an independent operator. If
        ``info_c`` is supplied, it is updated with the MPS-compatible
        ``"cur_orthog"`` pair and the tree-native ``"canonical_region"``,
        ``"isometry_map"``, and immutable ``"left_inds"`` snapshots.
        The live tensor ``left_inds`` and :attr:`canonical_region` remain the
        source of truth for the tree gauge; ``info_c`` is only a synchronization
        view for optimizer code.
        """
        if info_c is not None and not hasattr(info_c, "__setitem__"):
            raise TypeError("info_c must be a mutable mapping when supplied.")
        if center is None:
            center = self.plan.root
        center = _tree_node_selector(self.plan, center, self.node_tag_id)
        target = self if inplace else self.copy()
        for network in target.tree_networks:
            _canonicalize_tree_operator(network, target.plan, center)
        target._canonical_region = frozenset({center})
        if info_c is not None:
            info_c["cur_orthog"] = (center, center)
            info_c["canonical_region"] = target.canonical_region
            info_c["isometry_map"] = target.isometry_map()
            info_c["left_inds"] = tuple(
                {
                    node: (
                        None
                        if (tensor := network[target.node_tag(node)]).left_inds
                        is None
                        else tuple(tensor.left_inds)
                    )
                    for node in target.plan.nodes()
                }
                for network in target.tree_networks
            )
        return target

    def canonicalize_(self, center=None, *, info_c=None):
        """Inplace alias for :meth:`canonicalize`."""
        return self.canonicalize(
            center=center, inplace=True, info_c=info_c,
        )

    canonize = canonicalize_

    def invalidate_canonical_form(self):
        """Forget operator gauge metadata after an unmanaged tensor edit."""
        self._canonical_region = None
        return self

    def isometry_direction(self, node):
        """Return the neighbour receiving a node's canonical QR factor."""
        node = _tree_node_selector(self.plan, node, self.node_tag_id)
        tensor = self.node_tensor(node)
        if tensor.left_inds is None:
            return None
        right_inds = [ind for ind in tensor.inds if ind not in tensor.left_inds]
        if len(right_inds) != 1:
            return None
        for neighbor in self.neighbors(node):
            if right_inds[0] == self.bond(node, neighbor):
                return neighbor
        return None

    def isometry_map(self):
        """Return the live QR orientation map for all TreePlan nodes."""
        return {
            node: self.isometry_direction(node)
            for node in self.plan.nodes()
        }

    def is_subtree_canonical_form(self, nodes=None, *, span=False):
        """Check the lossless QR metadata around a connected operator region."""
        if nodes is None:
            region = self.canonical_region
            if region is None:
                return False
        else:
            region = (
                _tree_region_selector(self.plan, nodes, self.node_tag_id)
                if span
                else frozenset(
                    _tree_node_selector(self.plan, node, self.node_tag_id)
                    for node in nodes
                )
            )
            if _tree_subtree_span(self.plan, region) != region:
                return False
        for node in self.plan.nodes():
            if node in region:
                continue
            path = min(
                (
                    self.plan.node_path(node, target)
                    for target in region
                ),
                key=len,
            )
            if self.isometry_direction(node) != path[1]:
                return False
        return True

    def is_canonical_form(self, center=None):
        """Check whether the operator has a one-node canonical region."""
        if center is None:
            center = self.orthogonality_center
        if center is None:
            return False
        return self.is_subtree_canonical_form((center,))

    def shift_orthogonality_center(self, current, new):
        """Move the operator QR centre to another TreePlan node."""
        del current
        return self.canonicalize(center=new, inplace=True)

    def calc_current_orthog_center(self):
        """Return the current operator canonical region bounds."""
        region = self.canonical_region
        if not region:
            return None
        ordered = sorted(region)
        return ordered[0], ordered[-1]

    def left_canonicalize(self, *, center=None, inplace=False, **kwargs):
        """MPO-compatible alias for a root-oriented tree QR sweep."""
        del kwargs
        return self.canonicalize(
            center=self.plan.root if center is None else center,
            inplace=inplace,
        )

    left_canonicalize_ = lambda self, **kwargs: self.left_canonicalize(  # noqa: E731
        inplace=True, **kwargs,
    )
    left_canonize = left_canonicalize_

    def right_canonicalize(self, *, center=None, inplace=False, **kwargs):
        """MPO-compatible alias for a root-oriented tree QR sweep."""
        del kwargs
        return self.canonicalize(
            center=self.plan.root if center is None else center,
            inplace=inplace,
        )

    right_canonicalize_ = lambda self, **kwargs: self.right_canonicalize(  # noqa: E731
        inplace=True, **kwargs,
    )
    right_canonize = right_canonicalize_

    def compress_site(
        self, node, *, max_bond=None, cutoff=None, order="rank", **kwargs,
    ):
        """Compress all tree bonds consistently around ``node``."""
        del kwargs
        self.canonicalize(center=node)
        return self.compress(
            max_bond=max_bond,
            cutoff=cutoff,
            order=order,
        )

    def left_compress(
        self, *, max_bond=None, cutoff=None, order="rank", **kwargs,
    ):
        """Tree analogue of a rooted, rank-aware compression sweep."""
        del kwargs
        return self.compress(
            max_bond=max_bond,
            cutoff=cutoff,
            order=order,
        )

    def right_compress(
        self, *, max_bond=None, cutoff=None, order="rank", **kwargs,
    ):
        """Tree analogue of a rooted, rank-aware compression sweep."""
        del kwargs
        return self.compress(
            max_bond=max_bond,
            cutoff=cutoff,
            order=order,
        )

    def canonize_around(
        self, tags, which="all", *, inplace=False, **canonize_opts,
    ):
        """Quimb-style alias for TreePlan-centered TTNO canonicalization.

        Tree operators have a rooted geometry rather than a one-dimensional
        tag interval, so the supported target is one TreePlan node. The
        additional Quimb options are accepted for API familiarity and are
        intentionally ignored after validating that the target is a node.
        """
        del which, canonize_opts
        if isinstance(tags, (tuple, list, set, frozenset)):
            if len(tags) == 0:
                raise ValueError("TreeMPO.canonize_around needs one node.")
            if len(tags) != 1:
                target = self if inplace else self.copy()
                region = _tree_region_selector(
                    target.plan, tags, target.node_tag_id,
                )
                for network in target.tree_networks:
                    _canonicalize_tree_operator_region(
                        network, target.plan, region,
                    )
                target._canonical_region = region
                return target
        target = self if inplace else self.copy()
        return target.canonicalize(center=tags, inplace=True)

    def canonize_around_(self, tags, **kwargs):
        """In-place Quimb-style alias for :meth:`canonize_around`."""
        kwargs["inplace"] = True
        return self.canonize_around(tags, **kwargs)

    def canonize_between(
        self, tags1, tags2, *, inplace=False, absorb="right", **canonize_opts,
    ):
        """Canonicalize the operator exterior to a TreePlan path.

        A path is the tree analogue of the mixed-canonical interval used by
        an MPS. ``absorb`` and other Quimb gauge options are accepted for API
        compatibility; the lossless native QR policy controls the operation.
        """
        del absorb, canonize_opts
        node1 = _tree_node_selector(self.plan, tags1, self.node_tag_id)
        node2 = _tree_node_selector(self.plan, tags2, self.node_tag_id)
        region = frozenset(self.plan.node_path(node1, node2))
        target = self if inplace else self.copy()
        for network in target.tree_networks:
            _canonicalize_tree_operator_region(network, target.plan, region)
        target._canonical_region = region
        return target

    def canonize_between_(self, tags1, tags2, **kwargs):
        """In-place alias for :meth:`canonize_between`."""
        kwargs["inplace"] = True
        return self.canonize_between(tags1, tags2, **kwargs)

    def compress_between(
        self, tags1, tags2, max_bond=None, cutoff=1e-10, **compress_opts,
    ):
        """Quimb-style compression entry point for a tree operator.

        A TreeMPO compression is a global leaf-to-root sweep so every edge
        sees the complete operator sum. ``tags1`` and ``tags2`` identify an
        adjacent TreePlan edge for validation; the configured sweep then
        compresses all TreePlan bonds consistently.
        """
        inplace = compress_opts.pop("inplace", True)
        order = compress_opts.pop("order", "rank")
        del compress_opts
        node1 = _tree_node_selector(self.plan, tags1, self.node_tag_id)
        node2 = _tree_node_selector(self.plan, tags2, self.node_tag_id)
        if node2 not in self.neighbors(node1):
            raise ValueError(
                "TreeMPO.compress_between requires adjacent TreePlan nodes."
            )
        target = self if inplace else self.copy()
        return target.compress(
            max_bond=max_bond,
            cutoff=cutoff,
            order=order,
        )

    def compress_between_(self, tags1, tags2, **kwargs):
        """In-place alias for :meth:`compress_between`."""
        kwargs["inplace"] = True
        return self.compress_between(tags1, tags2, **kwargs)

    def compress(self, *, max_bond=None, cutoff=None, order="rank"):
        """Compress every TreePlan edge with a native SVD sweep.

        ``order="rank"`` uses a deterministic greedy leaf-elimination order.
        At every step it estimates the next edge rank from the live tensor
        dimensions and current operator bond, then removes the cheapest leaf.
        This keeps small-rank branches out of the active parent tensor first,
        reducing intermediate bond growth for large Pauli/direct-sum TTNOs.
        The ordering is a heuristic on a fixed tree, not a global search over
        alternate TreePlan geometries. ``order="depth"`` retains the simple
        deterministic depth-first order for reproducibility comparisons.
        Native graded Symmray TTNOs report the requested rank order but safely
        use the charge-preserving depth order until a graded permutation kernel
        can prove arbitrary sibling reordering phase-correct.
        """
        if cutoff is None:
            cutoff = self.cutoff
        cutoff = float(cutoff)
        reports = []
        for network in self.tree_networks:
            raw_bond = max(
                (network.ind_size(index) for index in network.inner_inds()),
                default=1,
            )
            structural_report = _structural_compress_tree(
                network,
                root=self.plan.root,
                parent=self.plan.parent,
                children=self.plan.children,
                nodes=self.plan.nodes(),
                tensor_getter=lambda node: _tree_operator_tensor(network, node),
                bond_getter=lambda node, neighbor: _tree_operator_bond(
                    network, self.plan, node, neighbor,
                ),
                method="auto",
            )
            report = _compress_tree_operator(
                network,
                self.plan,
                max_bond=max_bond,
                cutoff=cutoff,
                order=order,
            )
            report["structural"] = structural_report
            # Report the pre-structural dimension as the raw dimension, so
            # rank_reduced includes the exact pass as well as numerical SVD.
            report["raw_max_bond"] = raw_bond
            report["rank_reduced"] = report["final_max_bond"] < raw_bond
            reports.append(report)
        self.cutoff = cutoff
        self.compressed = True
        self.pepsy_compression_report = reports[0] if len(reports) == 1 else reports
        self._canonical_region = None
        return self

    def copy(self, virtual=False, deep=False, *, conj=False, transpose=False):
        """Copy the native tree operator.

        The signature follows Quimb's tensor-network ``copy`` API. ``virtual``
        keeps tensor data shared while copying the network structure; ``deep``
        requests independent numeric data as in the underlying Quimb views.
        ``conj`` and ``transpose`` are accepted as convenient MPO-compatible
        view operations.
        """
        copied = type(self)(
            self.plan,
            tuple(
                network.copy(virtual=virtual, deep=deep)
                for network in self.tree_networks
            ),
            terms=self.terms,
            backend=self.backend,
            fermionic=self.fermionic,
            symmetry=self.symmetry,
            cutoff=self.cutoff,
            compressed=self.compressed,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
            operator_support=self.operator_support,
            layout_finder=self.layout_finder,
        )
        if hasattr(self, "pepsy_compression_report"):
            copied.pepsy_compression_report = self.pepsy_compression_report
        copied._canonical_region = self.canonical_region
        if transpose:
            copied._transpose_operator_inplace()
        if conj:
            copied.conj(inplace=True)
        return copied

    def _transpose_operator_inplace(self):
        """Transpose every local upper/lower physical pair in place."""
        for network in self.tree_networks:
            for tensor in network:
                physical_axes = []
                for site in self.sites:
                    upper = self.upper_ind(site)
                    lower = self.lower_ind(site)
                    if upper in tensor.inds and lower in tensor.inds:
                        physical_axes.append((
                            tensor.inds.index(upper),
                            tensor.inds.index(lower),
                        ))
                if not physical_axes:
                    continue
                permutation = list(range(tensor.ndim))
                for upper_axis, lower_axis in physical_axes:
                    permutation[upper_axis], permutation[lower_axis] = (
                        permutation[lower_axis], permutation[upper_axis]
                    )
                tensor.modify(
                    data=ar.do("transpose", tensor.data, permutation),
                    # Transposition changes the data attached to the named
                    # physical indices, not the QR ownership of those named
                    # indices. Preserve the tree canonical gauge metadata.
                    left_inds=tensor.left_inds,
                )
        return self

    def conj(
        self,
        mangle_inner=False,
        output_inds=None,
        phase_dual=True,
        inplace=False,
    ):
        """Conjugate every stored tree operator like a Quimb operator view."""
        if inplace:
            for network in self.tree_networks:
                network.conj(
                    mangle_inner=mangle_inner,
                    output_inds=output_inds,
                    phase_dual=phase_dual,
                    inplace=True,
                )
            return self

        networks = tuple(
            network.conj(
                mangle_inner=mangle_inner,
                output_inds=output_inds,
                phase_dual=phase_dual,
            )
            for network in self.tree_networks
        )
        result = type(self)(
            self.plan,
            networks,
            terms=self.terms,
            backend=self.backend,
            fermionic=self.fermionic,
            symmetry=self.symmetry,
            cutoff=self.cutoff,
            compressed=self.compressed,
            sites=self.sites,
            site_tag_id=self.site_tag_id,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            node_tag_id=self.node_tag_id,
            operator_support=self.operator_support,
            layout_finder=self.layout_finder,
        )
        if not mangle_inner and output_inds is None:
            result._canonical_region = self.canonical_region
        return result

    def expectation(self, state, *, normalized=True, optimize="auto"):
        """Evaluate ``<state|TreeMPO|state>`` in one public operation."""
        import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

        tree = getattr(state, "tn", state)
        if getattr(tree, "plan", None) is None:
            raise TypeError("state must be a TreeTensorNetwork or TreeOptimizer.")
        if _tree_plan_signature(tree.plan) != self.pepsy_tree_plan_signature:
            raise ValueError("TreeMPO and state use different TreePlans.")
        if self.fermionic and not getattr(tree, "fermionic", False):
            raise TypeError("native TreeMPO requires a native fermionic tree state.")
        if self.fermionic is False and getattr(tree, "fermionic", False):
            raise TypeError("dense TreeMPO cannot be contracted with a native fermionic tree.")
        if self.fermionic and self.symmetry != getattr(tree, "symmetry", None):
            raise TypeError(
                "native TreeMPO and TreeTensorNetwork must use the same "
                f"symmetry, got operator={self.symmetry!r} and "
                f"state={getattr(tree, 'symmetry', None)!r}."
            )

        sites = tuple(sorted(tree.plan.node_of_qubit))
        numerator = 0.0
        for operator in self.tree_networks:
            ket = tree.copy()
            operator_work = operator.copy()
            ket_reindex = {}
            operator_reindex = {}
            for site in sites:
                physical = tree.site_ind(site)
                upper = self.upper_ind(site)
                lower = self.lower_ind(site)
                if upper not in operator_work.ind_map or lower not in operator_work.ind_map:
                    raise ValueError(f"TreeMPO is missing physical site {site!r}.")
                fresh = qtn.rand_uuid()
                ket_reindex[physical] = fresh
                operator_reindex[upper] = physical
                operator_reindex[lower] = fresh
            ket.reindex_(ket_reindex)
            operator_work.reindex_(operator_reindex)
            numerator = numerator + (tree.H | operator_work | ket).contract(
                all,
                optimize=optimize,
            )
        if not normalized:
            return numerator
        denominator = (tree.H | tree).contract(all, optimize=optimize)
        return numerator / denominator

    def __repr__(self):
        return (
            f"TreeMPO(nsite={self.plan.n}, backend={self.backend!r}, "
            f"networks={len(self.tree_networks)}, max_bond={self.max_bond()})"
        )

    def __getattr__(self, name):
        """Preserve the old metadata attributes on compact tree operators."""
        if name.startswith("pepsy_tree_operator_"):
            networks = self.__dict__.get("tree_networks", ())
            if len(networks) == 1:
                return getattr(networks[0], name)
        raise AttributeError(name)


def _term_support(where):
    """Normalize one Hamiltonian key to an integer support tuple."""
    if isinstance(where, Integral):
        support = (int(where),)
    else:
        try:
            support = tuple(int(site) for site in where)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "tree MPO Hamiltonian term locations must be integer sites "
                "or tuples of integer sites."
            ) from exc
    if not support:
        raise ValueError("tree MPO Hamiltonian term supports cannot be empty.")
    if len(set(support)) != len(support):
        raise ValueError(
            f"tree MPO Hamiltonian term support {support!r} repeats a site."
        )
    return support


def _network_bond(network, tag0, tag1):
    """Return the unique bond joining two tagged operator tensors."""
    tids0 = tuple(network.tag_map.get(tag0, ()))
    tids1 = tuple(network.tag_map.get(tag1, ()))
    if len(tids0) != 1 or len(tids1) != 1:
        raise ValueError(
            f"operator tags {tag0!r} and {tag1!r} must identify one tensor each."
        )
    shared = qtn.bonds(network.tensor_map[tids0[0]], network.tensor_map[tids1[0]])
    if len(shared) != 1:
        raise ValueError(
            f"operator tensors {tag0!r} and {tag1!r} must share one bond."
        )
    return next(iter(shared))


def _compose_tree_operator_network(
    left,
    right,
    *,
    nodes,
    edges,
    node_tag,
    site_of_node,
    neighbors,
    output_ind,
    input_ind,
    bond,
):
    """Compose two dense operator networks over the same tree geometry."""
    right_work = right.copy()
    right_bond_map = {
        bond(right, node, neighbor): qtn.rand_uuid()
        for node, neighbor in edges
    }
    right_physical_map = {}
    right_input_temps = {}
    for node in nodes:
        site = site_of_node(node)
        if site is None:
            continue
        right_output = output_ind(site)
        right_input = input_ind(site)
        left_input = input_ind(site)
        right_input_temp = qtn.rand_uuid()
        right_physical_map[right_output] = left_input
        right_physical_map[right_input] = right_input_temp
        right_input_temps[node] = right_input_temp
    right_work.reindex_({**right_bond_map, **right_physical_map})
    fused_bonds = {
        frozenset((node, neighbor)): qtn.rand_uuid()
        for node, neighbor in edges
    }

    tensors = []
    for node in nodes:
        tag = node_tag(node)
        left_tids = tuple(left.tag_map.get(tag, ()))
        right_tids = tuple(right_work.tag_map.get(tag, ()))
        if len(left_tids) != 1 or len(right_tids) != 1:
            raise ValueError(f"operator node tag {tag!r} must identify one tensor.")
        left_tensor = left.tensor_map[left_tids[0]]
        right_tensor = right_work.tensor_map[right_tids[0]]
        neighbors_node = tuple(neighbors(node))
        left_bonds = tuple(bond(left, node, neighbor) for neighbor in neighbors_node)
        right_bonds = tuple(
            right_bond_map[bond(right, node, neighbor)]
            for neighbor in neighbors_node
        )
        site = site_of_node(node)
        physical_left_output = None if site is None else output_ind(site)
        physical_right_input = None if site is None else right_input_temps[node]
        raw_inds = tuple(
            ind
            for ind in (physical_left_output, physical_right_input)
            if ind is not None
        ) + left_bonds + right_bonds
        joined = qtn.tensor_contract(
            left_tensor,
            right_tensor,
            output_inds=raw_inds,
        ).transpose(*raw_inds)
        interleaved_inds = tuple(
            ind
            for left_bond, right_bond in zip(left_bonds, right_bonds)
            for ind in (left_bond, right_bond)
        )
        joined = joined.transpose(
            *tuple(
                ind
                for ind in (physical_left_output, physical_right_input)
                if ind is not None
            ),
            *interleaved_inds,
        )
        fused_edge_names = tuple(
            fused_bonds[frozenset((node, neighbor))]
            for neighbor in neighbors_node
        )
        new_shape = tuple(
            joined.ind_size(ind)
            for ind in (physical_left_output, physical_right_input)
            if ind is not None
        ) + tuple(
            joined.ind_size(left_bond) * joined.ind_size(right_bond)
            for left_bond, right_bond in zip(left_bonds, right_bonds)
        )
        data = ar.do("reshape", joined.data, new_shape)
        output_inds = tuple(
            ind
            for ind in (
                physical_left_output,
                None if site is None else input_ind(site),
            )
            if ind is not None
        ) + fused_edge_names
        tensors.append(
            qtn.Tensor(
                data=data,
                inds=output_inds,
                tags=left_tensor.tags,
            )
        )
    return qtn.TensorNetwork(tensors)


def _expanded_index_charges(index):
    """Expand a native block index into its dense charge ordering."""
    chargemap = getattr(index, "chargemap", None)
    if chargemap is None:
        raise TypeError("native operator factors must expose block charges.")
    return [charge for charge, size in chargemap.items() for _ in range(size)]


def _operator_charge_neg(charge):
    """Negate one Abelian charge used by a native operator channel."""
    if isinstance(charge, tuple):
        return tuple(-value for value in charge)
    return -charge


def _operator_charge_sub(left, right):
    """Subtract two expanded physical charges componentwise."""
    if isinstance(left, tuple):
        return tuple(a - b for a, b in zip(left, right))
    return left - right


def _operator_charge_from_matrix(data, physical_map, *, tol=1e-10):
    """Infer the homogeneous local operator charge from a dense matrix."""
    values = {
        _operator_charge_sub(physical_map[int(out)], physical_map[int(inp)])
        for out, inp in np.argwhere(np.abs(data) > tol)
    }
    if len(values) != 1:
        raise ValueError(
            "operator-Schmidt factors must have one homogeneous physical "
            f"charge, got {sorted(values, key=repr)!r}."
        )
    return values.pop()


def _operator_native_channels(
    term, support, *, symmetry, dtype=None, cutoff=1e-12,
):
    """Split one native two-site term into charged local operator channels."""
    original_support = tuple(int(site) for site in support)
    support = tuple(sorted(original_support))
    if len(support) != 2:
        raise ValueError("native operator channels require two sites.")
    if tuple(_term_support(support)) != support:
        raise ValueError("operator support must contain distinct sites.")
    ordered_term = term
    if original_support != support:
        ordered_term = term.transpose((1, 0, 3, 2))

    # ``cutoff=0`` retains structural zero sectors as separate channels. The
    # small fixed threshold removes only those exact numerical zeros; the
    # user-facing TreeMPO cutoff is applied later to the combined TTNO.
    structural_cutoff = 64.0 * np.finfo(float).eps
    fused = ordered_term.fuse((0, 2), (1, 3))
    left, _, right = fused.svd(
        absorb="right", cutoff=structural_cutoff,
    )
    if left is None or right is None:
        raise ValueError("could not split a native two-site operator.")
    left = left.unfuse(0).transpose((2, 0, 1))
    right = right.unfuse(1)
    left_data = _as_numpy(left.to_dense(), dtype=dtype)
    right_data = _as_numpy(right.to_dense(), dtype=dtype)
    physical_map = _expanded_index_charges(left.indices[1])
    if _expanded_index_charges(left.indices[2]) != physical_map:
        raise ValueError("native operator factors have mismatched physical maps.")
    channels = []
    for channel in range(left_data.shape[0]):
        source = left_data[channel]
        target = right_data[channel]
        if not np.any(np.abs(source) > 1e-10):
            continue
        if not np.any(np.abs(target) > 1e-10):
            raise ValueError("native operator SVD produced an empty channel.")
        source_charge = _operator_charge_from_matrix(source, physical_map)
        target_charge = _operator_charge_from_matrix(target, physical_map)
        if target_charge != _operator_charge_neg(source_charge):
            raise ValueError(
                "native operator channel charges do not cancel: "
                f"{source_charge!r} and {target_charge!r}."
            )
        channels.append((source, target, source_charge))
    if not channels:
        raise ValueError("native two-site operator has no nonzero channels.")
    return channels, physical_map


def _operator_dense_channels(operator, support, *, dtype=None, cutoff=1e-12):
    """Split one ordinary dense two-site term into local channels."""
    support = tuple(sorted(int(site) for site in support))
    data = _dense_operator_array(operator, dtype=dtype)
    if data.ndim != 4 or data.shape[0] != data.shape[1] or data.shape[0] != data.shape[2]:
        raise ValueError("dense two-site operators must have shape (d, d, d, d).")
    if data.shape[2] != data.shape[3]:
        raise ValueError("dense two-site operators must have matching input legs.")
    dim = data.shape[0]
    matrix = data.transpose(0, 2, 1, 3).reshape(dim * dim, dim * dim)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    structural_cutoff = max(64.0 * np.finfo(float).eps, float(cutoff))
    channels = []
    for channel, value in enumerate(singular):
        if float(value) <= structural_cutoff:
            continue
        scale = np.sqrt(value)
        channels.append((
            (left[:, channel] * scale).reshape(dim, dim),
            (scale * right[channel, :]).reshape(dim, dim),
            0,
        ))
    if not channels:
        raise ValueError("dense two-site operator has no nonzero channels.")
    return channels, [0] * dim


def _operator_valid_child_states(nchildren, nstate, nchannel, done):
    """Generate only the valid sparse automaton child configurations."""
    if not nchildren:
        return [()]
    states = {(0,) * nchildren}
    active_states = tuple(range(1, done))
    for child in range(nchildren):
        for state in active_states:
            values = [0] * nchildren
            values[child] = state
            states.add(tuple(values))
        values = [0] * nchildren
        values[child] = done
        states.add(tuple(values))
    nchannel = int(nchannel)
    source = lambda channel: 1 + channel
    target = lambda channel: 1 + nchannel + channel
    for left in range(nchildren):
        for right in range(left + 1, nchildren):
            for channel in range(nchannel):
                for first, second in (
                    (source(channel), target(channel)),
                    (target(channel), source(channel)),
                ):
                    values = [0] * nchildren
                    values[left] = first
                    values[right] = second
                    states.add(tuple(values))
    return tuple(sorted(states))


def _combined_tree_operator(
    plan, terms, *, symmetry=None, cutoff=1e-12, dtype=None, fermionic=True,
):
    """Build one TreePlan TTNO for a neutral one-/two-site term mapping.

    Each two-site operator-Schmidt channel becomes a source/target charge
    channel. At a branching node, the channel automaton can collect one source
    and one target from different child subtrees before closing into the
    neutral ``done`` sector. This is the tree analogue of the start/channel/
    done construction used by a native chain MPO, but the channels follow the
    selected TreePlan rather than a Jordan--Wigner wire.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    if not hasattr(terms, "items") or not terms:
        raise ValueError("At least one operator term is required.")

    channels = []
    one_site = {}
    physical_map = None
    for where, term in terms.items():
        support = _term_support(where)
        if any(site not in plan.node_of_qubit for site in support):
            raise ValueError(f"operator support {support!r} is outside the TreePlan.")
        if len(support) == 1:
            data = (
                _dense_operator_array(term, dtype=dtype)
                if not fermionic else _as_numpy(term.to_dense(), dtype=dtype)
            )
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError("one-site operators must be square matrices.")
            if physical_map is None:
                physical_map = (
                    _expanded_index_charges(term.indices[0])
                    if fermionic else [0] * data.shape[0]
                )
            if fermionic:
                charge = _operator_charge_from_matrix(data, physical_map)
                zero = (
                    tuple(0 for _ in charge) if isinstance(charge, tuple) else 0
                )
                if charge != zero:
                    raise ValueError(
                        "combined native TreeMPO currently requires neutral "
                        "one-site terms."
                    )
            site = support[0]
            one_site[site] = one_site.get(site, 0) + data
            continue
        if len(support) != 2:
            raise NotImplementedError(
                "combined TreeMPO currently supports one- and two-site terms; "
                "use a precompiled structured TTNO for higher-rank terms."
            )
        if fermionic:
            term_channels, term_map = _operator_native_channels(
                term, support, symmetry=symmetry, dtype=dtype, cutoff=cutoff,
            )
        else:
            term_channels, term_map = _operator_dense_channels(
                term, support, dtype=dtype, cutoff=cutoff,
            )
        if physical_map is None:
            physical_map = list(term_map)
        elif list(term_map) != list(physical_map):
            raise ValueError("all TreeMPO terms must share one physical map.")
        for source, target, charge in term_channels:
            channels.append({
                "source": source,
                "target": target,
                "charge": charge if fermionic else 0,
                "sites": tuple(sorted(support)),
            })

    if physical_map is None:
        raise ValueError("At least one operator term is required.")
    if fermionic:
        first_charge = physical_map[0]
        zero = tuple(0 for _ in first_charge) if isinstance(first_charge, tuple) else 0
    else:
        zero = 0
    nchannel = len(channels)
    if not nchannel and not one_site:
        raise ValueError("operator terms produced no nonzero channels.")

    source_id = lambda channel: 1 + channel
    target_id = lambda channel: 1 + nchannel + channel
    done = 1 + 2 * nchannel
    state_map = [zero]
    state_map.extend(
        _operator_charge_neg(channel["charge"]) for channel in channels
    )
    state_map.extend(channel["charge"] for channel in channels)
    state_map.append(zero)
    physical_dim = len(physical_map)
    tensors = []

    for node in plan.nodes():
        children = tuple(plan.children[node])
        parent = plan.parent.get(node)
        has_parent = parent is not None
        neighbors = list(children) + ([parent] if has_parent else [])
        maps = [state_map] * len(neighbors)
        duals = [True] * len(children) + ([False] if has_parent else [])
        inds = [
            f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
            for neighbor in neighbors
        ]
        qubit = plan.qubit_of_node.get(node)
        if qubit is not None:
            # Native tree leaves conventionally expose physical legs before
            # their single virtual parent.  This ordering is not cosmetic:
            # Symmray's graded contraction phase depends on the ordered leg
            # exterior.  Keep the TTNO leaf in the same convention as the
            # native state and direct local-observable route.
            maps = [physical_map, physical_map] + maps
            duals = [False, True] + duals
            inds = [f"k{qubit}", f"b{qubit}"] + inds
        shape = [len(index_map) for index_map in maps]
        data = np.zeros(shape, dtype=dtype or complex)
        identity = (
            np.eye(physical_dim, dtype=data.dtype)
            if qubit is not None else 1.0
        )
        endpoint = {}
        for channel, info in enumerate(channels):
            if qubit == info["sites"][0]:
                endpoint.setdefault("source", []).append(channel)
            if qubit == info["sites"][1]:
                endpoint.setdefault("target", []).append(channel)

        valid_children = _operator_valid_child_states(
            len(children), len(state_map), nchannel, done,
        )
        for child_states in valid_children:
            active = []
            completed = False
            invalid = False
            for state in child_states:
                if state == 0:
                    continue
                if state == done:
                    if completed or active:
                        invalid = True
                    completed = True
                    continue
                if state < source_id(nchannel):
                    channel = state - 1
                    flag = "source"
                else:
                    channel = state - target_id(0)
                    flag = "target"
                if completed or any(
                    old_channel == channel and old_flag == flag
                    for old_channel, old_flag in active
                ):
                    invalid = True
                active.append((channel, flag))
            if invalid or len({channel for channel, _ in active}) > 1:
                continue
            if completed:
                base = done
            elif active:
                flags = {flag for _, flag in active}
                channel = active[0][0]
                base = (
                    done
                    if flags == {"source", "target"}
                    else source_id(channel)
                    if "source" in flags
                    else target_id(channel)
                )
            else:
                base = 0

            options = [(base, identity)]
            if base == 0 and qubit in one_site:
                options.append((done, one_site[qubit]))
            if base != done:
                for channel in endpoint.get("source", ()):
                    local = channels[channel]["source"]
                    if base == 0:
                        options.append((source_id(channel), local))
                    elif base == target_id(channel):
                        options.append((done, local))
                for channel in endpoint.get("target", ()):
                    local = channels[channel]["target"]
                    if base == 0:
                        options.append((target_id(channel), local))
                    elif base == source_id(channel):
                        options.append((done, local))

            for output, local in options:
                if not has_parent and output != done:
                    continue
                index = child_states + (output,) if has_parent else child_states
                if qubit is not None:
                    data[(slice(None), slice(None)) + index] += local
                else:
                    data[index] += local

        if fermionic:
            native = _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=maps,
                duals=duals,
                charge=zero,
                label=None,
            )
        else:
            native = data
        tags = [f"N{node}"]
        if qubit is not None:
            tags.append(f"I{qubit}")
        tensors.append(qtn.Tensor(native, inds=inds, tags=tags))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = (
        "native_tree_tnno" if fermionic else "dense_tree_tnno"
    )
    network.pepsy_tree_operator_bond = len(state_map)
    network.pepsy_tree_operator_raw_bond = len(state_map)
    network.pepsy_tree_operator_is_ttno = True
    return network


def _tree_plan_neighbors(plan, node):
    """Return a plan node's children followed by its optional parent."""
    return tuple(plan.children[node]) + (
        (plan.parent[node],) if plan.parent.get(node) is not None else ()
    )


def _tree_operator_peel_order(plan, nodes):
    """Return a deterministic leaf-to-hub order for a connected node set."""
    remaining = set(nodes)
    adjacency = {
        node: tuple(
            neighbor for neighbor in _tree_plan_neighbors(plan, node)
            if neighbor in remaining
        )
        for node in remaining
    }
    degree = {
        node: sum(neighbor in remaining for neighbor in neighbors)
        for node, neighbors in adjacency.items()
    }
    leaves = [node for node, value in degree.items() if value == 1]
    heapq.heapify(leaves)
    order = []
    while len(remaining) > 1:
        while leaves and leaves[0] not in remaining:
            heapq.heappop(leaves)
        if not leaves:
            raise ValueError("operator decomposition requires a connected tree")
        leaf = heapq.heappop(leaves)
        neighbor = next(
            node for node in adjacency[leaf] if node in remaining
        )
        order.append((leaf, neighbor))
        remaining.remove(leaf)
        degree[leaf] = 0
        degree[neighbor] -= 1
        if degree[neighbor] == 1:
            heapq.heappush(leaves, neighbor)
    return tuple(order), next(iter(remaining))


def _tree_operator_external_dim(network, tensor, bond):
    """Return the matrix-side dimension of ``tensor`` away from ``bond``."""
    dimension = 1
    for index in tensor.inds:
        if index != bond:
            dimension *= int(network.ind_size(index))
    return max(1, dimension)


def _tree_operator_compression_order(network, plan, order):
    """Return a deterministic leaf-to-root TTNO compression order.

    The rank-aware mode is intentionally a cheap greedy policy. It does not
    perform a second SVD merely to plan the first SVD: the current edge bond
    and the two matrix-side dimensions give a safe local rank bound. The
    parent tensors are updated after every edge, so the next candidate sees
    the reduced live dimensions.
    """
    if order in {None, "auto", "rank", "rank-aware"}:
        rank_aware = True
    elif order in {"depth", "tree", "deterministic"}:
        rank_aware = False
    else:
        raise ValueError(
            "TreeMPO compression order must be 'rank' or 'depth', got "
            f"{order!r}."
        )

    if not rank_aware:
        return tuple(
            (
                node,
                plan.node_path(node, plan.root)[1],
            )
            for node in sorted(
                (node for node in plan.nodes() if node != plan.root),
                key=lambda node: (
                    -len(plan.node_path(node, plan.root)),
                    int(node),
                ),
            )
        )

    remaining = set(plan.nodes())
    compression_order = []
    while len(remaining) > 1:
        leaves = [
            node
            for node in remaining
            if node != plan.root
            and sum(
                neighbor in remaining
                for neighbor in _tree_plan_neighbors(plan, node)
            ) == 1
        ]
        if not leaves:
            raise ValueError(
                "TreeMPO compression requires a connected TreePlan."
            )

        scored = []
        for node in leaves:
            neighbor = next(
                neighbor
                for neighbor in _tree_plan_neighbors(plan, node)
                if neighbor in remaining
            )
            tensor = _tree_operator_tensor(network, node)
            target = _tree_operator_tensor(network, neighbor)
            bond = _tree_operator_bond(network, plan, node, neighbor)
            left_dim = _tree_operator_external_dim(network, tensor, bond)
            right_dim = _tree_operator_external_dim(network, target, bond)
            rank_bound = min(left_dim, right_dim)
            bond_dim = int(network.ind_size(bond))
            # Prefer a small possible retained rank, then a small existing
            # direct-sum channel, then the cheaper local matrix shape. The
            # node id makes all ties reproducible.
            scored.append((
                rank_bound,
                bond_dim,
                left_dim * right_dim,
                int(node),
                node,
                neighbor,
            ))
        _, _, _, _, node, neighbor = min(scored)
        compression_order.append((node, neighbor))
        remaining.remove(node)

    return tuple(compression_order)


def _tree_operator_from_dense(
    plan,
    array,
    *,
    sites,
    dims,
    split_opts,
    site_tag_id,
    upper_ind_id,
    lower_ind_id,
    node_tag_id,
):
    """Decompose a dense matrix exactly across a TreePlan."""
    data = ar.do("reshape", array, tuple(dims) + tuple(dims))
    upper = tuple(upper_ind_id.format(site) for site in sites)
    lower = tuple(lower_ind_id.format(site) for site in sites)
    blob = qtn.Tensor(data, inds=upper + lower)

    owned = {node: [] for node in plan.nodes()}
    for site in sites:
        owned[plan.node_of_qubit[site]].extend((
            upper_ind_id.format(site),
            lower_ind_id.format(site),
        ))
    factors = {}
    peel_order, hub = _tree_operator_peel_order(plan, set(plan.nodes()))
    opts = dict(split_opts)
    opts.setdefault("method", "svd")
    opts.setdefault("absorb", "right")
    opts.setdefault("cutoff", 0.0)
    opts.setdefault("get", "tensors")

    for node, neighbor in peel_order:
        bond_ind = f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
        left, blob = blob.split(
            left_inds=tuple(owned[node]),
            right_inds=tuple(ind for ind in blob.inds if ind not in owned[node]),
            bond_ind=bond_ind,
            **opts,
        )
        factors[node] = left
        owned[neighbor].append(bond_ind)
    factors[hub] = blob

    tensors = []
    for node in plan.nodes():
        tensor = factors[node]
        qubit = plan.qubit_of_node.get(node)
        neighbors = tuple(plan.children[node]) + (
            (plan.parent[node],) if plan.parent.get(node) is not None else ()
        )
        desired = [
            *( (
                upper_ind_id.format(qubit),
                lower_ind_id.format(qubit),
            ) if qubit is not None else () ),
            *(
                f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
                for neighbor in neighbors
            ),
        ]
        tensor = tensor.transpose(*desired)
        tensor.add_tag(node_tag_id.format(node))
        if qubit is not None:
            tensor.add_tag(site_tag_id.format(qubit))
        tensors.append(tensor)

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_tnno"
    network.pepsy_tree_operator_is_ttno = True
    network.pepsy_tree_operator_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_raw_bond = network.pepsy_tree_operator_bond
    return network


def _tree_operator_from_fill_fn(
    plan,
    fill_fn,
    *,
    bond_dim,
    phys_dim,
    dtype,
    site_tag_id,
    upper_ind_id,
    lower_ind_id,
    node_tag_id,
):
    """Build a regular dense TTNO from local filled tensors."""
    if isinstance(bond_dim, Integral):
        edge_dims = {
            (min(node, child), max(node, child)): int(bond_dim)
            for node in plan.nodes()
            for child in plan.children[node]
        }
    else:
        edge_values = tuple(int(value) for value in bond_dim)
        edges = tuple(
            (node, child)
            for node in plan.nodes()
            for child in plan.children[node]
        )
        if len(edge_values) != len(edges):
            raise ValueError("bond_dim must be one value per TreePlan edge.")
        edge_dims = {
            (min(node, child), max(node, child)): value
            for (node, child), value in zip(edges, edge_values)
        }

    if isinstance(phys_dim, Integral):
        physical_dims = {site: int(phys_dim) for site in plan.node_of_qubit}
    else:
        values = tuple(int(value) for value in phys_dim)
        sites = tuple(sorted(plan.node_of_qubit))
        if len(values) != len(sites):
            raise ValueError("phys_dim must have one value per tree site.")
        physical_dims = dict(zip(sites, values))

    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = tuple(plan.children[node]) + (
            (plan.parent[node],) if plan.parent.get(node) is not None else ()
        )
        inds = [
            *((
                upper_ind_id.format(qubit),
                lower_ind_id.format(qubit),
            ) if qubit is not None else ()),
            *(
                f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
                for neighbor in neighbors
            ),
        ]
        shape = [
            *( (physical_dims[qubit], physical_dims[qubit])
               if qubit is not None else () ),
            *(
                edge_dims[(min(node, neighbor), max(node, neighbor))]
                for neighbor in neighbors
            ),
        ]
        try:
            data = fill_fn(tuple(shape))
        except TypeError:
            data = fill_fn(node, tuple(shape))
        tensor = qtn.Tensor(np.asarray(data, dtype=dtype), inds=inds)
        tensor.add_tag(node_tag_id.format(node))
        if qubit is not None:
            tensor.add_tag(site_tag_id.format(qubit))
        tensors.append(tensor)

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_tnno"
    network.pepsy_tree_operator_is_ttno = True
    return network


def _identity_tree_operator(
    plan,
    *,
    phys_dim=2,
    dtype=complex,
    site_tag_id="I{}",
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    node_tag_id="N{}",
):
    """Build an exact bond-one identity TTNO."""
    if isinstance(phys_dim, Integral):
        physical_dims = {site: int(phys_dim) for site in plan.node_of_qubit}
    else:
        sites = tuple(sorted(plan.node_of_qubit))
        values = tuple(int(value) for value in phys_dim)
        if len(values) != len(sites):
            raise ValueError("phys_dim must have one value per tree site.")
        physical_dims = dict(zip(sites, values))

    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = tuple(plan.children[node]) + (
            (plan.parent[node],) if plan.parent.get(node) is not None else ()
        )
        inds = [
            *((
                upper_ind_id.format(qubit),
                lower_ind_id.format(qubit),
            ) if qubit is not None else ()),
            *(
                f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
                for neighbor in neighbors
            ),
        ]
        shape = [
            *((physical_dims[qubit], physical_dims[qubit])
              if qubit is not None else ()),
            *(1 for _ in neighbors),
        ]
        data = np.zeros(shape, dtype=dtype)
        if qubit is None:
            data[...] = 1
        else:
            data[(slice(None), slice(None)) + (0,) * len(neighbors)] = np.eye(
                physical_dims[qubit], dtype=dtype,
            )
        tensor = qtn.Tensor(data, inds=inds, tags=[node_tag_id.format(node)])
        if qubit is not None:
            tensor.add_tag(site_tag_id.format(qubit))
        tensors.append(tensor)
    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_identity"
    network.pepsy_tree_operator_is_ttno = True
    return network


def _relabel_tree_operator_network(
    network,
    plan,
    *,
    site_tag_id="I{}",
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    node_tag_id="N{}",
):
    """Relabel a generated default TTNO to the public TreeMPO layout.

    The native and dense local-term builders intentionally share compact
    internal labels.  Apply the public labels in one place so ``from_gate``
    has the same layout contract as ``from_dense`` and ``from_fill_fn``.
    """
    index_map = {}
    tag_map = {}
    for node in plan.nodes():
        tag_map[f"N{node}"] = node_tag_id.format(node)
        qubit = plan.qubit_of_node.get(node)
        if qubit is not None:
            index_map[f"k{qubit}"] = upper_ind_id.format(qubit)
            index_map[f"b{qubit}"] = lower_ind_id.format(qubit)
            tag_map[f"I{qubit}"] = site_tag_id.format(qubit)
    if index_map:
        network.reindex_(index_map)
    if tag_map:
        network.retag_(tag_map)
    network.pepsy_tree_node_tag_id = node_tag_id
    return network


def _pauli_sum_tree_operator(
    plan,
    weighted_terms,
    *,
    dtype=complex,
    site_tag_id="I{}",
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    node_tag_id="N{}",
):
    """Construct a compact dense TTNO for a sparse Pauli-product sum."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    from ..stabilizer_tn.operators import pauli_matrix

    if not hasattr(weighted_terms, "__iter__"):
        raise TypeError("weighted_terms must be an iterable of Pauli branches.")
    terms = []
    support = set()
    for item in weighted_terms:
        try:
            weight, mapping = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Pauli branches must be (coefficient, {site: axis}) pairs."
            ) from exc
        if not hasattr(mapping, "items"):
            raise TypeError("each Pauli branch mapping must provide items().")
        clean = {}
        for site, axis in mapping.items():
            site = int(site)
            if site not in plan.node_of_qubit:
                raise ValueError(
                    f"Pauli branch site {site} is outside the TreePlan."
                )
            axis = str(axis).upper()
            if axis == "I":
                continue
            if axis not in {"X", "Y", "Z"}:
                raise ValueError(f"invalid Pauli axis {axis!r}.")
            clean[site] = axis
        terms.append((complex(weight), clean))
        support.update(clean)
    if not terms:
        raise ValueError("at least one Pauli branch is required.")

    if support:
        active_nodes = _tree_subtree_span(
            plan,
            tuple(plan.node_of_qubit[site] for site in sorted(support)),
        )
        route_support = tuple(sorted(support))
    else:
        # A pure scalar identity has no non-identity support. Anchor its
        # scalar on one physical site so the normal TreeMPO route still has a
        # concrete, bond-one local support and never needs a global scalar
        # special case.
        anchor_qubit = min(plan.node_of_qubit)
        active_nodes = frozenset((plan.node_of_qubit[anchor_qubit],))
        route_support = (anchor_qubit,)
    anchor = min(active_nodes)
    active_edges = {
        frozenset((node, neighbor))
        for node in active_nodes
        for neighbor in _tree_plan_neighbors(plan, node)
        if neighbor in active_nodes
    }
    rank = len(terms)
    dtype = np.dtype(dtype or complex)
    identity = np.eye(2, dtype=dtype)
    paulis = {
        axis: np.asarray(pauli_matrix(axis), dtype=dtype)
        for axis in ("X", "Y", "Z")
    }

    def edge_name(node, neighbor):
        return f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"

    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = _tree_plan_neighbors(plan, node)
        inds = [
            *((f"k{qubit}", f"b{qubit}") if qubit is not None else ()),
            *(edge_name(node, neighbor) for neighbor in neighbors),
        ]
        if node not in active_nodes:
            shape = list((2, 2) if qubit is not None else ())
            shape.extend(1 for _ in neighbors)
            data = np.zeros(tuple(shape), dtype=dtype)
            if qubit is None:
                data[...] = 1
            else:
                data[(slice(None), slice(None)) + (0,) * len(neighbors)] = identity
        else:
            shape = list((2, 2) if qubit is not None else ())
            shape.extend(
                rank if frozenset((node, neighbor)) in active_edges else 1
                for neighbor in neighbors
            )
            data = np.zeros(tuple(shape), dtype=dtype)
            for branch, (weight, mapping) in enumerate(terms):
                local = 1.0 if qubit is None else paulis.get(
                    mapping.get(qubit), identity,
                )
                if node == anchor:
                    local = weight * local
                edge_indices = tuple(
                    branch
                    if frozenset((node, neighbor)) in active_edges else 0
                    for neighbor in neighbors
                )
                if qubit is None:
                    data[edge_indices] += local
                else:
                    data[(slice(None), slice(None)) + edge_indices] += local
        tensor = qtn.Tensor(
            data,
            inds=inds,
            tags=[f"N{node}"] + ([f"I{qubit}"] if qubit is not None else []),
        )
        tensors.append(tensor)

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_pauli_sum_tnno"
    network.pepsy_tree_operator_is_ttno = True
    network.pepsy_tree_operator_bond = rank
    network.pepsy_tree_operator_raw_bond = rank
    _relabel_tree_operator_network(
        network,
        plan,
        site_tag_id=site_tag_id,
        upper_ind_id=upper_ind_id,
        lower_ind_id=lower_ind_id,
        node_tag_id=node_tag_id,
    )
    return network, route_support


def _native_tree_term_network(
    plan, term, support, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Decompose one native term into a graded TTNO on the selected tree.

    The decomposition is performed on the native operator tensor itself, not
    on a dense Jordan--Wigner matrix.  The physical upper/lower pair at every
    supported site is fused into one packed leg and the resulting tensor is
    peeled across the TreePlan Steiner subtree with native Symmray SVDs. This
    is the important fermionic distinction from factorizing ordinary dense
    local matrices: the native fuse/SVD retains the graded phases at every
    branch of the tree.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    support = _term_support(support)
    if any(site not in plan.node_of_qubit for site in support):
        raise ValueError(f"term support {support!r} is outside the TreePlan.")

    physical_map = _expanded_index_charges(term.indices[0])
    zero = (
        tuple(0 for _ in physical_map[0])
        if physical_map and isinstance(physical_map[0], tuple)
        else 0
    )

    endpoint_nodes = tuple(plan.node_of_qubit[site] for site in support)
    if len(support) == 1:
        factors = {
            endpoint_nodes[0]: qtn.Tensor(
                term,
                inds=(f"k{support[0]}", f"b{support[0]}"),
            )
        }
        active_nodes = {endpoint_nodes[0]}
    else:
        ordered_support = tuple(sorted(support))
        if support == ordered_support:
            ordered_term = term
        else:
            rank = len(support)
            order = tuple(sorted(range(rank), key=support.__getitem__))
            ordered_term = term.transpose(
                (*order, *(axis + rank for axis in order))
            )
        rank = len(ordered_support)
        fused = ordered_term.fuse(*(
            (axis, axis + rank) for axis in range(rank)
        ))
        packed_inds = tuple(f"_pepsy_op_packed_{site}" for site in ordered_support)
        blob = qtn.Tensor(fused, inds=packed_inds)
        ordered_nodes = tuple(plan.node_of_qubit[site] for site in ordered_support)
        active_nodes = {ordered_nodes[0]}
        for target in ordered_nodes[1:]:
            anchor = min(
                active_nodes,
                key=lambda node: len(plan.node_path(node, target)),
            )
            active_nodes.update(plan.node_path(anchor, target))
        peel_order, hub = _tree_operator_peel_order(plan, active_nodes)
        owned = {node: set() for node in active_nodes}
        for node, site in zip(ordered_nodes, ordered_support):
            owned[node].add(f"_pepsy_op_packed_{site}")
        factors = {}
        for node, neighbor in peel_order:
            left_inds = tuple(
                index for index in blob.inds if index in owned[node]
            )
            if not left_inds:
                raise RuntimeError(
                    f"operator decomposition lost subtree payload at {node}."
                )
            left, right = blob.split(
                left_inds=left_inds,
                method="svd",
                absorb="right",
                cutoff=max(64.0 * np.finfo(float).eps, float(cutoff)),
                get="tensors",
                bond_ind=f"_pepsy_op_bond_{node}_{neighbor}",
            )
            factors[node] = left
            blob = right
            owned[neighbor].add(f"_pepsy_op_bond_{node}_{neighbor}")
        factors[hub] = blob

    def edge_name(node, neighbor):
        return f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"

    def rebuild_with_axis(data, maps, duals, dense):
        return _native_from_dense(
            dense,
            symmetry=symmetry,
            index_maps=maps,
            duals=duals,
            charge=getattr(data, "charge", zero),
        )

    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = _tree_plan_neighbors(plan, node)
        if node in factors:
            factor = factors[node]
            data = factor.data
            inds = list(factor.inds)
            if qubit in support:
                packed = next(
                    (
                        index for index in inds
                        if index.startswith("_pepsy_op_packed_")
                    ),
                    None,
                )
                if packed is not None:
                    axis = inds.index(packed)
                    data = data.unfuse(axis)
                    inds[axis:axis + 1] = [f"k{qubit}", f"b{qubit}"]
            elif qubit is not None:
                # A physical TreePlan root can lie on the active Steiner
                # subtree without being an endpoint. Its operator action is
                # the identity, so add that even physical pair explicitly.
                dense = _as_numpy(data.to_dense(), dtype=dtype or complex)
                dense = np.einsum(
                    "ab,...->ab...",
                    np.eye(len(physical_map), dtype=dense.dtype),
                    dense,
                )
                maps = [physical_map, physical_map] + [
                    _expanded_index_charges(index) for index in data.indices
                ]
                duals = [False, True] + [
                    index.dual for index in data.indices
                ]
                data = rebuild_with_axis(data, maps, duals, dense)
                inds = [f"k{qubit}", f"b{qubit}"] + inds

            # Rename the native decomposition's temporary bond labels before
            # adding the trivial exterior bonds.
            inds = [
                edge_name(*map(int, index.removeprefix("_pepsy_op_bond_").split("_")))
                if index.startswith("_pepsy_op_bond_") else index
                for index in inds
            ]
            existing = set(inds)
            for neighbor in neighbors:
                index = edge_name(node, neighbor)
                if index in existing:
                    continue
                dense = np.expand_dims(
                    _as_numpy(data.to_dense(), dtype=dtype or complex),
                    axis=-1,
                )
                maps = [
                    _expanded_index_charges(axis)
                    for axis in data.indices
                ] + [[zero]]
                duals = [axis.dual for axis in data.indices] + [
                    neighbor in plan.children[node]
                ]
                data = rebuild_with_axis(data, maps, duals, dense)
                inds.append(index)
                existing.add(index)

            desired = [
                *((f"k{qubit}", f"b{qubit}") if qubit is not None else ()),
                *(edge_name(node, neighbor) for neighbor in neighbors),
            ]
            tensor = qtn.Tensor(data, inds=inds).transpose(*desired)
            tensor.add_tag(f"N{node}")
            if qubit is not None:
                tensor.add_tag(f"I{qubit}")
            tensors.append(tensor)
            continue

        # Nodes outside the term's Steiner subtree carry a neutral identity.
        maps = []
        duals = []
        inds = []
        if qubit is not None:
            maps.extend((physical_map, physical_map))
            duals.extend((False, True))
            inds.extend((f"k{qubit}", f"b{qubit}"))
        for neighbor in neighbors:
            maps.append([zero])
            duals.append(neighbor in plan.children[node])
            inds.append(edge_name(node, neighbor))
        shape = tuple(len(index_map) for index_map in maps)
        data = np.zeros(shape, dtype=dtype or complex)
        if qubit is None:
            data[(0,) * len(neighbors)] = 1.0
        else:
            data[(slice(None), slice(None)) + (0,) * len(neighbors)] = np.eye(
                len(physical_map), dtype=data.dtype
            )
        tensors.append(qtn.Tensor(
            _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=maps,
                duals=duals,
                charge=zero,
            ),
            inds=inds,
            tags=[f"N{node}"] + ([f"I{qubit}"] if qubit is not None else []),
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "native_tree_term_tnno"
    network.pepsy_tree_operator_charge = getattr(term, "charge", zero)
    network.pepsy_tree_operator_is_ttno = True
    return network


def _normalize_native_term_edge_orientation(network, plan, *, symmetry, dtype=None):
    """Normalize native term-network virtual duals to the TreePlan orientation."""
    if not hasattr(symmetry, "parity"):
        from symmray import get_symmetry  # pylint: disable=import-outside-toplevel

        symmetry = get_symmetry(symmetry)
    for node in plan.nodes():
        tensor = network[f"N{node}"]
        for neighbor in _tree_plan_neighbors(plan, node):
            edge = f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
            axis = tensor.inds.index(edge)
            index = tensor.data.indices[axis]
            desired_dual = neighbor in plan.children[node]
            if index.dual == desired_dual:
                continue
            old_charges = _expanded_index_charges(index)
            relabelled = [_operator_charge_neg(charge) for charge in old_charges]
            zero = (
                tuple(0 for _ in relabelled[0])
                if relabelled and isinstance(relabelled[0], tuple)
                else 0
            )
            probe = _native_from_dense(
                np.zeros((len(relabelled),), dtype=dtype or complex),
                symmetry=symmetry,
                index_maps=[relabelled],
                duals=[desired_dual],
                charge=zero,
            )
            new_charges = _expanded_index_charges(probe.indices[0])
            old_positions = {}
            for position, charge in enumerate(old_charges):
                old_positions.setdefault(charge, []).append(position)
            used = {charge: 0 for charge in old_positions}
            permutation = []
            for charge in new_charges:
                old_charge = _operator_charge_neg(charge)
                position = used[old_charge]
                permutation.append(old_positions[old_charge][position])
                used[old_charge] = position + 1
            dense = np.take(
                _as_numpy(tensor.data.to_dense(), dtype=dtype or complex),
                permutation,
                axis=axis,
            )
            # Reversing a fermionic virtual edge is a graded dualization, not
            # just a charge-label permutation.  The plan-parent endpoint
            # carries the parity gauge associated with that reversal.  Apply
            # it once per edge (the child endpoint gets the dual charge map,
            # but not a second parity phase).
            if desired_dual:
                parity = np.asarray(
                    [
                        -1 if symmetry.parity(charge) else 1
                        for charge in new_charges
                    ],
                    dtype=dense.dtype,
                )
                phase_shape = [1] * dense.ndim
                phase_shape[axis] = len(parity)
                dense = dense * parity.reshape(phase_shape)
            maps = [
                new_charges if current_axis == axis else
                _expanded_index_charges(current)
                for current_axis, current in enumerate(tensor.data.indices)
            ]
            duals = [
                desired_dual if current_axis == axis else current.dual
                for current_axis, current in enumerate(tensor.data.indices)
            ]
            rebuilt = _native_from_dense(
                dense,
                symmetry=symmetry,
                index_maps=maps,
                duals=duals,
                charge=getattr(tensor.data, "charge", 0),
            )
            tensor.modify(data=rebuilt)
    return network


def _tree_operator_charge(network, plan):
    """Return the homogeneous open charge carried by one native TTNO."""
    if hasattr(network, "pepsy_tree_operator_charge"):
        return network.pepsy_tree_operator_charge
    node_tag_id = getattr(network, "pepsy_tree_node_tag_id", "N{}")
    tensor = network[node_tag_id.format(int(plan.root))]
    return getattr(tensor.data, "charge", 0)


def _native_tree_operator_sum(
    networks,
    plan,
    *,
    symmetry,
    signs=None,
    dtype=None,
    upper_ind_id="k{}",
    lower_ind_id="b{}",
    site_tag_id="I{}",
    node_tag_id="N{}",
):
    """Direct-sum native TTNOs without Quimb's dense-axis padding.

    Quimb's generic ``tensor_network_ag_sum`` is correct for ordinary dense
    arrays but its padding hook cannot construct a multi-sector Symmray axis.
    This routine builds each virtual index from its expanded charge order,
    embeds every component into the corresponding charge-aware positions, and
    reconstructs each local tensor with ``symmray.from_dense``. It is the
    operator analogue of :func:`_native_term_sum_tree_operator`, but accepts
    already-factorized complete TTNOs as its inputs.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    networks = tuple(networks)
    if not networks:
        raise ValueError("native TTNO direct sum requires at least one network.")
    if signs is None:
        signs = (1,) * len(networks)
    signs = tuple(signs)
    if len(signs) != len(networks):
        raise ValueError("native TTNO direct-sum signs must match networks.")

    def edge_name(node, neighbor):
        return f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"

    def node_tensor(network, node):
        network_node_tag_id = getattr(
            network, "pepsy_tree_node_tag_id", node_tag_id,
        )
        return network[network_node_tag_id.format(int(node))]

    reference = node_tensor(networks[0], plan.root)
    operator_charge = getattr(reference.data, "charge", 0)
    for network in networks[1:]:
        charge = getattr(node_tensor(network, plan.root).data, "charge", 0)
        if charge != operator_charge:
            raise ValueError(
                "native TTNO direct-sum inputs must have the same operator "
                f"charge, got {operator_charge!r} and {charge!r}."
            )

    physical_maps = {}
    for site in sorted(plan.node_of_qubit):
        upper = upper_ind_id.format(site)
        lower = lower_ind_id.format(site)
        for network in networks:
            tensor = node_tensor(network, plan.node_of_qubit[site])
            upper_axis = tensor.inds.index(upper)
            lower_axis = tensor.inds.index(lower)
            upper_map = _expanded_index_charges(tensor.data.indices[upper_axis])
            lower_map = _expanded_index_charges(tensor.data.indices[lower_axis])
            if upper_map != lower_map:
                raise ValueError(
                    f"native TTNO physical maps disagree at site {site}."
                )
            if site in physical_maps and physical_maps[site] != upper_map:
                raise ValueError(
                    f"native TTNO physical maps disagree at site {site}."
                )
            physical_maps[site] = upper_map

    sample_charge = next(iter(physical_maps.values()))[0]
    zero = (
        tuple(0 for _ in sample_charge)
        if isinstance(sample_charge, tuple) else 0
    )
    edge_maps = {}
    edge_positions = [dict() for _ in networks]
    edge_duals = {}
    edge_endpoints = {}
    for parent in plan.nodes():
        for child in plan.children[parent]:
            edge = edge_name(parent, child)
            edge_endpoints[edge] = (parent, child)
            charges = []
            for network in networks:
                child_tensor = node_tensor(network, child)
                child_axis = child_tensor.inds.index(edge)
                child_index = child_tensor.data.indices[child_axis]
                charges.extend(_expanded_index_charges(child_index))
                parent_tensor = node_tensor(network, parent)
                parent_axis = parent_tensor.inds.index(edge)
                parent_index = parent_tensor.data.indices[parent_axis]
                endpoint_duals = (child_index.dual, parent_index.dual)
                if edge in edge_duals and edge_duals[edge] != endpoint_duals:
                    raise ValueError(
                        f"native TTNO edge {edge!r} has inconsistent duals."
                    )
                edge_duals[edge] = endpoint_duals

            probe = _native_from_dense(
                np.zeros((len(charges),), dtype=dtype or complex),
                symmetry=symmetry,
                index_maps=[charges],
                duals=[False],
                charge=zero,
            )
            global_charges = _expanded_index_charges(probe.indices[0])
            edge_maps[edge] = global_charges
            positions_by_charge = {}
            for position, charge in enumerate(global_charges):
                positions_by_charge.setdefault(charge, []).append(position)
            used_by_charge = {charge: 0 for charge in positions_by_charge}
            for network_index, network in enumerate(networks):
                child_tensor = node_tensor(network, child)
                child_axis = child_tensor.inds.index(edge)
                local_charges = _expanded_index_charges(
                    child_tensor.data.indices[child_axis]
                )
                positions = []
                for charge in local_charges:
                    offset = used_by_charge[charge]
                    positions.append(positions_by_charge[charge][offset])
                    used_by_charge[charge] = offset + 1
                edge_positions[network_index][edge] = tuple(positions)

    tensors = []
    for node in plan.nodes():
        neighbors = _tree_plan_neighbors(plan, node)
        qubit = plan.qubit_of_node.get(node)
        desired = [
            *(
                (upper_ind_id.format(qubit), lower_ind_id.format(qubit))
                if qubit is not None else ()
            ),
            *(edge_name(node, neighbor) for neighbor in neighbors),
        ]
        global_maps = []
        global_duals = []
        if qubit is not None:
            global_maps.extend((physical_maps[qubit], physical_maps[qubit]))
            global_duals.extend((False, True))
        for neighbor in neighbors:
            edge = edge_name(node, neighbor)
            global_maps.append(edge_maps[edge])
            child, parent = edge_duals[edge]
            _, child_node = edge_endpoints[edge]
            global_duals.append(
                child if node == child_node else parent
            )

        shape = tuple(len(index_map) for index_map in global_maps)
        data = np.zeros(shape, dtype=dtype or complex)
        for network_index, (network, sign) in enumerate(zip(networks, signs)):
            tensor = node_tensor(network, node).transpose(*desired)
            local = _as_numpy(tensor.data.to_dense(), dtype=data.dtype)
            if sign != 1:
                local = sign * local
            selections = []
            if qubit is not None:
                selections.extend((slice(None), slice(None)))
            for neighbor in neighbors:
                selections.append(
                    edge_positions[network_index][edge_name(node, neighbor)]
                )
            data[np.ix_(*[
                np.arange(local.shape[axis])
                if isinstance(selection, slice)
                else np.asarray(selection)
                for axis, selection in enumerate(selections)
            ])] += local

        tensors.append(qtn.Tensor(
            _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=global_maps,
                duals=global_duals,
                charge=operator_charge,
            ),
            inds=desired,
            tags=[node_tag_id.format(node)] + (
                [site_tag_id.format(qubit)] if qubit is not None else []
            ),
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_node_tag_id = node_tag_id
    network.pepsy_tree_operator_kind = "native_tree_tnno"
    network.pepsy_tree_operator_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_raw_bond = network.pepsy_tree_operator_bond
    network.pepsy_tree_operator_is_ttno = True
    return network


def _native_term_sum_tree_operator(
    plan, terms, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Direct-sum exact native term TTNOs into one operator network."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    term_networks = []
    for where, term in terms.items():
        network = _native_tree_term_network(
            plan,
            term,
            _term_support(where),
            symmetry=symmetry,
            cutoff=cutoff,
            dtype=dtype,
        )
        term_networks.append(_normalize_native_term_edge_orientation(
            network, plan, symmetry=symmetry, dtype=dtype,
        ))
    term_networks = tuple(term_networks)
    if not term_networks:
        raise ValueError("At least one operator term is required.")

    def edge_name(node, neighbor):
        return f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"

    physical_map = None
    for term in terms.values():
        physical_map = _expanded_index_charges(term.indices[0])
        break
    zero = (
        tuple(0 for _ in physical_map[0])
        if physical_map and isinstance(physical_map[0], tuple)
        else 0
    )
    edge_maps = {}
    edge_positions = {index: {} for index in range(len(term_networks))}
    for node in plan.nodes():
        for neighbor in plan.children[node]:
            edge = (node, neighbor)
            edge_index = edge_name(*edge)
            charges = []
            for index, network in enumerate(term_networks):
                tensor = network[f"N{neighbor}"]
                local = tensor.data.indices[tensor.inds.index(edge_index)]
                local_charges = _expanded_index_charges(local)
                charges.extend(local_charges)
            # Symmray groups an index's sectors by charge, so a raw
            # concatenation of per-term charge lists is not a set of
            # contiguous direct-sum slices when two terms share a charge.
            # Build the actual expanded order through the same native index
            # constructor and allocate duplicate charge sectors term by term.
            probe = _native_from_dense(
                np.zeros((len(charges),), dtype=dtype or complex),
                symmetry=symmetry,
                index_maps=[charges],
                duals=[False],
                charge=zero,
            )
            global_charges = _expanded_index_charges(probe.indices[0])
            edge_maps[edge_index] = global_charges
            positions_by_charge = {}
            for position, charge in enumerate(global_charges):
                positions_by_charge.setdefault(charge, []).append(position)
            used_by_charge = {charge: 0 for charge in positions_by_charge}
            for index, network in enumerate(term_networks):
                tensor = network[f"N{neighbor}"]
                local = tensor.data.indices[tensor.inds.index(edge_index)]
                local_charges = _expanded_index_charges(local)
                positions = []
                for charge in local_charges:
                    offset = used_by_charge[charge]
                    positions.append(positions_by_charge[charge][offset])
                    used_by_charge[charge] = offset + 1
                edge_positions[index][edge_index] = tuple(positions)

    tensors = []
    for node in plan.nodes():
        neighbors = _tree_plan_neighbors(plan, node)
        qubit = plan.qubit_of_node.get(node)
        desired = [
            *((f"k{qubit}", f"b{qubit}") if qubit is not None else ()),
            *(edge_name(node, neighbor) for neighbor in neighbors),
        ]
        global_maps = []
        global_duals = []
        if qubit is not None:
            global_maps.extend((physical_map, physical_map))
            global_duals.extend((False, True))
        for neighbor in neighbors:
            index = edge_name(node, neighbor)
            global_maps.append(edge_maps[index])
            # Native tree decomposition can orient an odd operator bond
            # differently at a hub than the ordinary state-tree convention.
            # The two endpoint dual flags are part of the graded operator
            # data; recomputing them from the plan would change valid local
            # sectors and drop them during ``from_dense``.
            reference = term_networks[0][f"N{node}"]
            reference_axis = reference.data.indices[
                reference.inds.index(index)
            ]
            global_duals.append(reference_axis.dual)
        shape = tuple(len(index_map) for index_map in global_maps)
        data = np.zeros(shape, dtype=dtype or complex)
        for term_index, network in enumerate(term_networks):
            tensor = network[f"N{node}"].transpose(*desired)
            local = _as_numpy(tensor.data.to_dense(), dtype=data.dtype)
            slices = []
            if qubit is not None:
                slices.extend((slice(None), slice(None)))
            for neighbor in neighbors:
                slices.append(edge_positions[term_index][edge_name(node, neighbor)])
            # ``np.ix_`` is needed here because charge-grouped duplicate
            # sectors are generally interleaved across the direct-sum axis.
            data[np.ix_(*[
                np.arange(local.shape[axis]) if isinstance(selection, slice)
                else np.asarray(selection)
                for axis, selection in enumerate(slices)
            ])] += local
        tensors.append(qtn.Tensor(
            _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=global_maps,
                duals=global_duals,
                charge=zero,
            ),
            inds=desired,
            tags=[f"N{node}"] + ([f"I{qubit}"] if qubit is not None else []),
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "native_tree_tnno"
    network.pepsy_tree_operator_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_raw_bond = network.pepsy_tree_operator_bond
    network.pepsy_tree_operator_is_ttno = True
    return network


def _tree_operator_tensor(network, node):
    """Fetch one operator tensor by its stable TreePlan node tag."""
    node_tag_id = getattr(network, "pepsy_tree_node_tag_id", "N{}")
    return network[node_tag_id.format(int(node))]


def _tree_operator_bond(network, plan, node, neighbor):
    """Find the unique live operator bond for one TreePlan edge."""
    left = _tree_operator_tensor(network, node)
    right = _tree_operator_tensor(network, neighbor)
    shared = tuple(set(left.inds).intersection(right.inds))
    if len(shared) != 1:
        raise ValueError(
            f"operator TTNO edge {(node, neighbor)!r} has {len(shared)} bonds."
        )
    return shared[0]


def _tree_operator_qr(tensor, *, left_inds, bond_ind):
    """Run the lossless dense/native QR policy for one operator tensor."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel
    from .ttn import _native_qr_split_tensor  # pylint: disable=import-outside-toplevel

    split_bond = qtn.rand_uuid()
    options = {
        "left_inds": tuple(left_inds),
        "right_inds": (bond_ind,),
        "method": "qr",
        "absorb": "right",
        "cutoff": 0.0,
        "get": "tensors",
        # Keep the stable TreePlan edge label. Replacing it with Quimb's
        # random split label would make a valid TTNO fail ``TreeMPO.validate``
        # after canonicalization/compression. The temporary split bond is
        # renamed by the caller after the old edge has been contracted.
        "bond_ind": split_bond,
    }
    kept, carry = _native_qr_split_tensor(tensor, **options)
    return kept, carry, split_bond


def _canonicalize_tree_operator(network, plan, center):
    """Canonicalize a tree operator by lossless QR from leaves to center."""
    return _canonicalize_tree_operator_region(network, plan, {center})


def _canonicalize_tree_operator_region(network, plan, region):
    """Canonicalize the complement of a connected tree region inwards."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    region = frozenset(region)
    if not region or not region.issubset(plan.children):
        raise ValueError("operator canonicalization region is invalid.")
    if _tree_subtree_span(plan, region) != region:
        raise ValueError("operator canonicalization region must be connected.")

    def distance_to_region(node):
        return min(len(plan.node_path(node, target)) for target in region)

    order = sorted(
        (node for node in plan.nodes() if node not in region),
        key=distance_to_region,
        reverse=True,
    )
    for node in order:
        path = min(
            (
                plan.node_path(node, target)
                for target in region
                if target != node
            ),
            key=len,
        )
        neighbor = path[1]
        tensor = _tree_operator_tensor(network, node)
        target = _tree_operator_tensor(network, neighbor)
        bond = _tree_operator_bond(network, plan, node, neighbor)
        kept, carry, split_bond = _tree_operator_qr(
            tensor,
            left_inds=tuple(index for index in tensor.inds if index != bond),
            bond_ind=bond,
        )
        merged = qtn.tensor_contract(carry, target)
        kept.reindex_({split_bond: bond})
        merged.reindex_({split_bond: bond})
        tensor.modify(
            data=kept.data,
            inds=kept.inds,
            left_inds=kept.left_inds,
        )
        target.modify(
            data=merged.data,
            inds=merged.inds,
            left_inds=None,
        )
    network.pepsy_tree_operator_center = (
        next(iter(region)) if len(region) == 1 else None
    )
    network.pepsy_tree_operator_canonical = True
    return network


def _compress_tree_operator(network, plan, *, max_bond, cutoff, order="rank"):
    """Compress one combined TTNO with a native rank-aware edge sweep."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    requested_order = order
    native_graded = str(
        getattr(network, "pepsy_tree_operator_kind", "")
    ).startswith("native")
    if native_graded and order in {None, "auto", "rank", "rank-aware"}:
        # Symmray's graded contraction convention makes sibling elimination
        # order part of the stored tensor-leg phase convention. Until a
        # charge-aware permutation kernel is available, retain the safe
        # deterministic order for native graded networks rather than risking
        # an apparently harmless but algebraically wrong sign change.
        order = "depth"

    raw_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    if cutoff == 0.0 and max_bond is None:
        _canonicalize_tree_operator(network, plan, plan.root)
        final_bond = raw_bond
        return {
            "compressed": False,
            "cutoff": cutoff,
            "requested_max_bond": None,
            "raw_max_bond": raw_bond,
            "final_max_bond": final_bond,
            "rank_reduced": False,
            "order": requested_order,
            "effective_order": order,
            "edge_order": (),
        }

    edge_order = _tree_operator_compression_order(network, plan, order)
    for node, neighbor in edge_order:
        tensor = _tree_operator_tensor(network, node)
        target = _tree_operator_tensor(network, neighbor)
        bond = _tree_operator_bond(network, plan, node, neighbor)
        left_inds = tuple(index for index in tensor.inds if index != bond)
        right_inds = tuple(index for index in target.inds if index != bond)
        # Fix the open-leg order before a graded contraction. Symmray's phase
        # convention is sensitive to an implicit contraction output order;
        # explicitly separating the eliminated leaf legs from the surviving
        # parent legs makes the dense and native paths deterministic.
        combined = qtn.tensor_contract(
            tensor,
            target,
            output_inds=left_inds + right_inds,
        )
        options = {
            "left_inds": left_inds,
            "right_inds": right_inds,
            "method": "svd",
            "absorb": "right",
            "cutoff": cutoff,
            "cutoff_mode": "rsum2",
            "get": "tensors",
            "bond_ind": bond,
        }
        if max_bond is not None:
            options["max_bond"] = int(max_bond)
        left, right = combined.split(**options)
        tensor.modify(
            data=left.data,
            inds=left.inds,
            left_inds=left.left_inds,
        )
        target.modify(
            data=right.data,
            inds=right.inds,
            left_inds=right.left_inds,
        )
    final_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_bond = final_bond
    network.pepsy_tree_operator_canonical = False
    return {
        "compressed": True,
        "cutoff": cutoff,
        "requested_max_bond": None if max_bond is None else int(max_bond),
        "raw_max_bond": raw_bond,
        "final_max_bond": final_bond,
        "rank_reduced": final_bond < raw_bond,
        "max_bond_exceeded": (
            max_bond is not None and final_bond > int(max_bond)
        ),
        "order": requested_order,
        "effective_order": order,
        "edge_order": edge_order,
    }


def _native_from_dense(
    data, *, symmetry, index_maps, duals, charge, label=None,
):
    """Create one native Symmray tensor lazily."""
    from symmray import utils as sr_utils  # pylint: disable=import-outside-toplevel

    return sr_utils.from_dense(
        data,
        symmetry=symmetry,
        index_maps=index_maps,
        duals=duals,
        fermionic=True,
        charge=charge,
        label=label,
    )


def _native_identity_tree_operator(
    plan,
    reference,
    *,
    symmetry,
    upper_ind_id,
    lower_ind_id,
    site_tag_id,
    node_tag_id,
    dtype=None,
):
    """Build a bond-one native identity using a reference TTNO's charges."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    def reference_tensor(node):
        tag_id = getattr(reference, "pepsy_tree_node_tag_id", node_tag_id)
        return reference[tag_id.format(node)]

    def index_charges(index):
        return _expanded_index_charges(index)

    physical_maps = {}
    physical_duals = {}
    for site, node in plan.node_of_qubit.items():
        tensor = reference_tensor(node)
        upper = upper_ind_id.format(site)
        lower = lower_ind_id.format(site)
        try:
            upper_axis = tensor.data.indices[tensor.inds.index(upper)]
            lower_axis = tensor.data.indices[tensor.inds.index(lower)]
        except (ValueError, KeyError, AttributeError) as exc:
            raise ValueError(
                f"native TreeMPO identity is missing physical site {site!r}."
            ) from exc
        physical_maps[site] = (
            index_charges(upper_axis), index_charges(lower_axis),
        )
        physical_duals[site] = (upper_axis.dual, lower_axis.dual)

    sample_charge = next(iter(physical_maps.values()))[0][0]
    zero = (
        tuple(0 for _ in sample_charge)
        if isinstance(sample_charge, tuple) else 0
    )
    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = _tree_plan_neighbors(plan, node)
        inds = []
        maps = []
        duals = []
        if qubit is not None:
            upper_map, lower_map = physical_maps[qubit]
            if upper_map != lower_map:
                raise ValueError(
                    "native TreeMPO identity requires matching upper/lower "
                    f"physical charge maps at site {qubit!r}."
                )
            inds.extend((
                upper_ind_id.format(qubit), lower_ind_id.format(qubit),
            ))
            maps.extend((upper_map, lower_map))
            duals.extend(physical_duals[qubit])

        for neighbor in neighbors:
            edge = f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
            reference_edge = reference_tensor(node)
            reference_neighbor = reference_tensor(neighbor)
            shared = qtn.bonds(reference_edge, reference_neighbor)
            if len(shared) != 1:
                raise ValueError(
                    "native TreeMPO identity requires one bond per TreePlan "
                    f"edge, got {len(shared)} for {(node, neighbor)!r}."
                )
            reference_bond = next(iter(shared))
            axis = reference_edge.data.indices[
                reference_edge.inds.index(reference_bond)
            ]
            inds.append(edge)
            maps.append([zero])
            duals.append(axis.dual)

        shape = tuple(len(index_map) for index_map in maps)
        data = np.zeros(shape, dtype=dtype or complex)
        if qubit is None:
            data[...] = 1.0
        else:
            data[(slice(None), slice(None)) + (0,) * len(neighbors)] = np.eye(
                len(physical_maps[qubit][0]), dtype=data.dtype,
            )
        tensors.append(qtn.Tensor(
            _native_from_dense(
                data,
                symmetry=symmetry,
                index_maps=maps,
                duals=duals,
                charge=zero,
            ),
            inds=inds,
            tags=[node_tag_id.format(node)] + (
                [site_tag_id.format(qubit)] if qubit is not None else []
            ),
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "native_tree_identity"
    network.pepsy_tree_operator_is_ttno = True
    network.pepsy_tree_operator_bond = 1
    network.pepsy_tree_operator_raw_bond = 1
    return network


def _pair_coefficient_factors(terms, nsite):
    """Factor an off-diagonal symmetric coefficient table, if possible."""
    first_support, first_term = next(iter(terms.items()))
    first_matrix = _as_numpy(first_term.to_dense()).reshape(
        (first_term.shape[0] * first_term.shape[2],) * 2
    )
    table = np.zeros((nsite, nsite), dtype=complex)
    for where, term in terms.items():
        support = _term_support(where)
        if len(support) != 2 or support[0] >= support[1]:
            return None
        matrix = _as_numpy(term.to_dense()).reshape(
            (term.shape[0] * term.shape[2],) * 2
        )
        denominator = np.vdot(first_matrix, first_matrix)
        ratio = np.vdot(first_matrix, matrix) / denominator
        if not np.allclose(matrix, ratio * first_matrix, rtol=1e-10, atol=1e-12):
            return None
        table[support] = ratio / 2.0
        table[support[::-1]] = ratio / 2.0

    nonzero = np.argwhere(np.abs(table) > 1e-14)
    if len(nonzero) < 3:
        return None
    i0, j0 = map(int, nonzero[0])
    candidates = [
        index for index in range(nsite)
        if index not in {i0, j0} and abs(table[i0, index]) > 1e-14
    ]
    if not candidates or abs(table[i0, j0]) <= 1e-14:
        return None
    k = candidates[0]
    a = np.zeros(nsite, dtype=complex)
    b = np.zeros(nsite, dtype=complex)
    a[i0] = 1.0
    b[j0] = table[i0, j0]
    b[k] = table[i0, k]
    a[k] = table[k, j0] / b[j0]
    if abs(a[k]) <= 1e-14 or abs(b[k]) <= 1e-14:
        return None
    a[j0] = table[j0, k] / b[k]
    b[i0] = table[k, i0] / a[k]
    for index in range(nsite):
        if index != j0 and abs(b[j0]) > 1e-14:
            a[index] = table[index, j0] / b[j0]
        if index != i0:
            b[index] = table[i0, index] / a[i0]

    for i in range(nsite):
        for j in range(nsite):
            if i == j:
                continue
            if not np.allclose(
                a[i] * b[j], table[i, j], rtol=1e-9, atol=1e-11,
            ):
                return None
    return first_term, first_support, a, b


def _pair_endpoint_automaton(
    plan, terms, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Compile a separable pair correlator into a four-state tree operator."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    factored = _pair_coefficient_factors(terms, plan.n)
    if factored is None:
        return None
    first_term, first_support, source_weights, target_weights = factored
    fused = first_term.fuse((0, 2), (1, 3))
    left, _, right = fused.svd(absorb="right", cutoff=cutoff)
    if left is None or right is None:
        return None
    left = left.unfuse(0).transpose((2, 0, 1))
    right = right.unfuse(1)
    left_data = _as_numpy(left.to_dense(), dtype=dtype or complex)[0]
    right_data = _as_numpy(right.to_dense(), dtype=dtype or complex)[0]
    physical_map = _expanded_index_charges(left.indices[1])
    physical_dim = len(physical_map)
    zero = 0 if symmetry in {"U1", "Z2"} else (0, 0)
    pair_charge = getattr(first_term, "pair_charge", None)
    if pair_charge is None:
        # The local factor's first physical charge is sufficient to infer the
        # endpoint channel charge for the standard pair observable.
        pair_charge = physical_map[-1]
    opposite_pair = (
        tuple(-value for value in pair_charge)
        if isinstance(pair_charge, tuple) else -pair_charge
    )
    state_map = [zero, opposite_pair, pair_charge, zero]
    tensors = []

    for node in plan.nodes():
        children = tuple(plan.children[node])
        parent = plan.parent.get(node)
        qubit = plan.qubit_of_node.get(node)
        has_parent = parent is not None
        edges = list(children) + ([parent] if has_parent else [])
        shape = [4] * len(edges)
        maps = [state_map] * len(edges)
        duals = [True] * len(children) + ([False] if has_parent else [])
        inds = [
            f"_to{min(node, neighbor)}_{max(node, neighbor)}"
            for neighbor in edges
        ]
        if qubit is not None:
            shape.extend((physical_dim, physical_dim))
            maps.extend((physical_map, physical_map))
            duals.extend((False, True))
            inds.extend((f"k{qubit}", f"b{qubit}"))
        data = np.zeros(shape, dtype=dtype or complex)

        if qubit is not None:
            source = source_weights[qubit] * left_data
            target = target_weights[qubit] * right_data
            identity = np.eye(physical_dim, dtype=data.dtype)

        for child_states in (
            np.ndindex(*(4 for _ in children)) if children else [()]
        ):
            source_count = sum(state & 1 for state in child_states)
            target_count = sum((state >> 1) & 1 for state in child_states)
            if source_count > 1 or target_count > 1:
                continue
            base = source_count | (target_count << 1)
            options = [(base, identity if qubit is not None else 1.0)]
            if qubit is not None:
                if not source_count:
                    options.append((1 | (target_count << 1), source))
                if not target_count:
                    options.append((source_count | 2, target))
            for output, local_operator in options:
                if has_parent:
                    index = child_states + (output,)
                else:
                    if output != 3:
                        continue
                    index = child_states
                data[index] += local_operator

        array = _native_from_dense(
            data,
            symmetry=symmetry,
            index_maps=maps,
            duals=duals,
            charge=zero,
        )
        tags = [f"N{node}"]
        if qubit is not None:
            tags.append(f"I{qubit}")
        tensors.append(qtn.Tensor(array, inds=inds, tags=tags))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "pair_endpoint_automaton"
    network.pepsy_tree_operator_bond = 4
    return network


def _tree_tensor_network_for_term(
    plan, term, support, *, symmetry, cutoff=1e-12, dtype=None,
):
    """Build an exact native fallback network for one higher-rank term.

    Higher-rank terms can still be kept as complete graded operator tensors
    for callers that explicitly need this compatibility fallback. The normal
    Hamiltonian path uses :func:`_native_tree_term_network` and amalgamates
    the resulting factors into one canonicalizable TTNO rather than a list of
    hyperedges.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    support = tuple(int(site) for site in support)
    if not support:
        raise ValueError("native tree term support cannot be empty.")
    if any(site not in plan.node_of_qubit for site in support):
        raise ValueError(f"term support {support!r} is outside the TreePlan.")
    expected_rank = 2 * len(support)
    if len(term.indices) != expected_rank:
        raise TypeError(
            f"a {len(support)}-site native term must have rank "
            f"{expected_rank}, got {len(term.indices)}."
        )

    physical_map = _expanded_index_charges(term.indices[0])
    physical_dim = len(physical_map)
    zero = getattr(term, "zero_charge", None)
    if zero is None:
        zero = 0 if symmetry in {"U1", "Z2"} else (0, 0)
    operator_dtype = np.dtype(dtype or _as_numpy(term.to_dense()).dtype)

    # The term's native indices are ordered as all upper physical legs,
    # followed by all lower physical legs. Keep that ordering intact while
    # assigning the logical tree-site labels.
    term_inds = [f"k{site}" for site in support]
    term_inds.extend(f"b{site}" for site in support)
    term_tags = [f"N{plan.node_of_qubit[site]}" for site in support]
    term_tags.extend(f"I{site}" for site in support)
    tensors = [qtn.Tensor(term, inds=term_inds, tags=term_tags)]

    support_set = set(support)
    for site in sorted(plan.node_of_qubit):
        if site in support_set:
            continue
        identity = _native_from_dense(
            np.eye(physical_dim, dtype=operator_dtype),
            symmetry=symmetry,
            index_maps=[physical_map, physical_map],
            duals=[False, True],
            charge=zero,
        )
        node = plan.node_of_qubit[site]
        tensors.append(qtn.Tensor(
            identity,
            inds=[f"k{site}", f"b{site}"],
            tags=[f"N{node}", f"I{site}"],
        ))

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "native_term_hyperedge"
    span = {plan.node_of_qubit[support[0]]}
    for site in support[1:]:
        anchor = next(iter(span))
        span.update(plan.node_path(anchor, plan.node_of_qubit[site]))
    network.pepsy_tree_operator_path = tuple(sorted(span))
    return network


def _dense_operator_array(operator, *, dtype=None):
    """Extract one ordinary dense operator array."""
    if hasattr(operator, "to_dense"):
        operator = operator.to_dense()
    elif hasattr(operator, "data") and not hasattr(operator, "shape"):
        # Backend arrays expose ``.data`` too, but for CuPy that is the raw
        # MemoryPointer rather than an array that Autoray can convert.
        operator = operator.data
    return _as_numpy(operator, dtype=dtype)


def _dense_tree_tensor_network_for_term(plan, operator, support, *, dtype=None):
    """Build one exact ordinary dense tree operator hyperedge."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    support = tuple(int(site) for site in support)
    data = _dense_operator_array(operator, dtype=dtype)
    expected_rank = 2 * len(support)
    if data.ndim != expected_rank:
        raise ValueError(
            f"a {len(support)}-site dense term must have rank {expected_rank}, "
            f"got {data.ndim}."
        )
    if any(site not in plan.node_of_qubit for site in support):
        raise ValueError(f"term support {support!r} is outside the TreePlan.")
    physical_dim = data.shape[0]
    if any(size != physical_dim for size in data.shape):
        raise ValueError("dense tree terms must have one physical dimension.")
    tensors = [qtn.Tensor(
        data,
        inds=[f"k{site}" for site in support]
        + [f"b{site}" for site in support],
        tags=[f"N{plan.node_of_qubit[site]}" for site in support]
        + [f"I{site}" for site in support],
    )]
    support_set = set(support)
    identity = np.eye(physical_dim, dtype=data.dtype)
    for site in sorted(plan.node_of_qubit):
        if site in support_set:
            continue
        node = plan.node_of_qubit[site]
        tensors.append(qtn.Tensor(
            identity,
            inds=[f"k{site}", f"b{site}"],
            tags=[f"N{node}", f"I{site}"],
        ))
    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_term_hyperedge"
    span = {plan.node_of_qubit[support[0]]}
    for site in support[1:]:
        anchor = next(iter(span))
        span.update(plan.node_path(anchor, plan.node_of_qubit[site]))
    network.pepsy_tree_operator_path = tuple(sorted(span))
    return network


def _dense_tree_term_tnno(plan, operator, support, *, dtype=None):
    """Build one exact dense term as a valid TreePlan TTNO.

    Unlike ``_dense_tree_tensor_network_for_term`` (the historical hyperedge
    fallback), this factors the operator only over the term's minimal Steiner
    subtree and adds bond-one identities on the exterior.  The result has one
    tensor per TreePlan node and can therefore be merged, canonicalized,
    compressed, and applied by the native TreeMPO router.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    raw_support = _term_support(support)
    ordered_support = tuple(sorted(raw_support))
    data = _dense_operator_array(operator, dtype=dtype)
    rank = len(ordered_support)
    if data.ndim != 2 * rank:
        raise ValueError(
            f"a {rank}-site dense term must have rank {2 * rank}, "
            f"got {data.ndim}."
        )
    if raw_support != ordered_support:
        order = tuple(sorted(range(rank), key=raw_support.__getitem__))
        data = data.transpose((*order, *(axis + rank for axis in order)))
    output_dims = tuple(int(size) for size in data.shape[:rank])
    input_dims = tuple(int(size) for size in data.shape[rank:])
    if output_dims != input_dims:
        raise ValueError("dense TreeMPO terms must have matching input/output dimensions.")
    if any(site not in plan.node_of_qubit for site in ordered_support):
        raise ValueError(
            f"term support {ordered_support!r} is outside the TreePlan."
        )

    packed_inds = tuple(
        f"_pepsy_dense_term_phys_{qtn.rand_uuid()}_{site}"
        for site in ordered_support
    )
    real_dtype = np.empty((), dtype=data.dtype).real.dtype
    if not np.issubdtype(real_dtype, np.inexact):
        real_dtype = np.dtype(float)
    # ``cutoff=0`` in Quimb deliberately retains numerical zero singular
    # values. For a local gate that can inflate a rank-two operator such as
    # CNOT to the full packed physical dimension, even though no physical
    # information is retained. Remove only machine-precision null sectors;
    # user-requested TreeMPO compression remains a separate later sweep.
    structural_cutoff = 64.0 * np.finfo(real_dtype).eps
    interleaved = data.transpose(
        [axis for site in range(rank) for axis in (site, rank + site)]
    ).reshape(tuple(dim * dim for dim in output_dims))
    blob = qtn.Tensor(interleaved, inds=packed_inds)
    site_nodes = tuple(plan.node_of_qubit[site] for site in ordered_support)
    active_nodes = {site_nodes[0]}
    for target in site_nodes[1:]:
        anchor = min(
            active_nodes,
            key=lambda node: len(plan.node_path(node, target)),
        )
        active_nodes.update(plan.node_path(anchor, target))
    peel_order, hub = _tree_operator_peel_order(plan, active_nodes)
    owned = {node: set() for node in active_nodes}
    for node, packed in zip(site_nodes, packed_inds):
        owned[node].add(packed)
    factors = {}
    edge_bonds = {}
    for node, neighbor in peel_order:
        left_inds = tuple(index for index in blob.inds if index in owned[node])
        if not left_inds:
            raise RuntimeError(
                f"dense TreeMPO term decomposition lost payload at {node}."
            )
        bond = f"_pepsy_dense_term_bond_{qtn.rand_uuid()}"
        left, blob = blob.split(
            left_inds=left_inds,
            method="svd",
            absorb="right",
            cutoff=structural_cutoff,
            get="tensors",
            bond_ind=bond,
        )
        factors[node] = left
        edge_bonds[(node, neighbor)] = bond
        owned[neighbor].add(bond)
    factors[hub] = blob

    def edge_name(node, neighbor):
        return f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"

    bond_names = {}
    for (node, neighbor), bond in edge_bonds.items():
        bond_names[bond] = edge_name(node, neighbor)

    physical_dims = dict(zip(ordered_support, output_dims))
    tensors = []
    for node in plan.nodes():
        qubit = plan.qubit_of_node.get(node)
        neighbors = _tree_plan_neighbors(plan, node)
        if node in factors:
            factor = factors[node]
            tensor_data = np.asarray(ar.to_numpy(factor.data), dtype=dtype)
            inds = [bond_names.get(index, index) for index in factor.inds]
            if qubit in physical_dims:
                packed = packed_inds[ordered_support.index(qubit)]
                axis = inds.index(packed)
                dim = physical_dims[qubit]
                shape = list(tensor_data.shape)
                shape[axis:axis + 1] = [dim, dim]
                tensor_data = tensor_data.reshape(shape)
                inds[axis:axis + 1] = [f"k{qubit}", f"b{qubit}"]
            elif qubit is not None:
                dim = output_dims[0] if output_dims else 2
                tensor_data = np.einsum(
                    "ab,...->ab...",
                    np.eye(dim, dtype=tensor_data.dtype),
                    tensor_data,
                )
                inds = [f"k{qubit}", f"b{qubit}"] + inds
            existing = set(inds)
            for neighbor in neighbors:
                edge = edge_name(node, neighbor)
                if edge not in existing:
                    tensor_data = np.expand_dims(tensor_data, axis=-1)
                    inds.append(edge)
                    existing.add(edge)
            desired = [
                *((f"k{qubit}", f"b{qubit}") if qubit is not None else ()),
                *(edge_name(node, neighbor) for neighbor in neighbors),
            ]
            tensor = qtn.Tensor(tensor_data, inds=inds).transpose(*desired)
        else:
            dims = []
            inds = []
            if qubit is not None:
                dim = physical_dims.get(qubit, output_dims[0] if output_dims else 2)
                dims.extend((dim, dim))
                inds.extend((f"k{qubit}", f"b{qubit}"))
            for neighbor in neighbors:
                dims.append(1)
                inds.append(edge_name(node, neighbor))
            tensor_data = np.zeros(tuple(dims), dtype=dtype or data.dtype)
            if qubit is None:
                tensor_data[...] = 1.0
            else:
                tensor_data[(slice(None), slice(None)) + (0,) * len(neighbors)] = (
                    np.eye(dims[0], dtype=tensor_data.dtype)
                )
            tensor = qtn.Tensor(tensor_data, inds=inds)
        tensor.add_tag(f"N{node}")
        if qubit is not None:
            tensor.add_tag(f"I{qubit}")
        tensors.append(tensor)

    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_term_tnno"
    network.pepsy_tree_operator_is_ttno = True
    network.pepsy_tree_operator_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_raw_bond = network.pepsy_tree_operator_bond
    return network


def _direct_sum_dense_tnno(networks, plan, *, dtype=None):
    """Direct-sum dense term TTNOs into one valid TreeMPO network."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    networks = tuple(networks)
    if not networks:
        raise ValueError("at least one dense TreeMPO term is required.")
    edge_names = {
        (min(node, neighbor), max(node, neighbor)):
        f"_pepsy_tnno_{min(node, neighbor)}_{max(node, neighbor)}"
        for node in plan.nodes()
        for neighbor in plan.children[node]
    }

    def edge_name(node, neighbor):
        return edge_names[(min(node, neighbor), max(node, neighbor))]

    edge_sizes = {
        index: sum(network.ind_size(index) for network in networks)
        for index in edge_names.values()
    }
    offsets = {edge: 0 for edge in edge_names.values()}
    tensors = []
    for node in plan.nodes():
        neighbors = _tree_plan_neighbors(plan, node)
        qubit = plan.qubit_of_node.get(node)
        reference = networks[0][f"N{node}"]
        desired = [
            *((f"k{qubit}", f"b{qubit}") if qubit is not None else ()),
            *(edge_name(node, neighbor) for neighbor in neighbors),
        ]
        shape = []
        if qubit is not None:
            shape.extend((reference.ind_size(f"k{qubit}"), reference.ind_size(f"b{qubit}")))
        shape.extend(edge_sizes[edge_name(node, neighbor)] for neighbor in neighbors)
        data = np.zeros(tuple(shape), dtype=dtype or np.asarray(ar.to_numpy(reference.data)).dtype)
        for network in networks:
            tensor = network[f"N{node}"].transpose(*desired)
            local = np.asarray(ar.to_numpy(tensor.data), dtype=data.dtype)
            slices = []
            if qubit is not None:
                slices.extend((slice(None), slice(None)))
            for neighbor in neighbors:
                edge = edge_name(node, neighbor)
                start = offsets[edge]
                stop = start + network.ind_size(edge)
                slices.append(slice(start, stop))
            data[tuple(slices)] += local
            for neighbor in neighbors:
                edge = edge_name(node, neighbor)
                offsets[edge] += network.ind_size(edge)
        # The per-edge offsets must restart for each node; use cumulative
        # offsets only inside this tensor construction.
        for edge in offsets:
            offsets[edge] = 0
        tensors.append(qtn.Tensor(data, inds=desired, tags=[f"N{node}"] + (
            [f"I{qubit}"] if qubit is not None else []
        )))
    network = qtn.TensorNetwork(tensors)
    network.pepsy_tree_operator_kind = "dense_tree_tnno"
    network.pepsy_tree_operator_is_ttno = True
    network.pepsy_tree_operator_bond = max(
        (network.ind_size(index) for index in network.inner_inds()),
        default=1,
    )
    network.pepsy_tree_operator_raw_bond = network.pepsy_tree_operator_bond
    return network


def _terms_by_operator_charge(terms):
    """Group native terms without mixing their Symmray operator charges."""
    grouped = {}
    for where, term in terms.items():
        grouped.setdefault(getattr(term, "charge", 0), {})[where] = term
    return grouped


def _charge_is_zero(charge):
    """Return whether an Abelian scalar or tuple charge is neutral."""
    if isinstance(charge, tuple):
        return all(value == 0 for value in charge)
    return charge == 0


def _build_mixed_charge_tree_operator(
    plan,
    hamiltonian,
    *,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    dtype=None,
    fermionic=True,
    to_backend=None,
):
    """Build one public ``TreeMPO`` from separate native charge networks."""
    from ...tensors.symmetric import (
        SymHamiltonian,
        _apply_to_tensor_network_arrays,
    )

    networks = []
    for charge, sector_terms in _terms_by_operator_charge(
        hamiltonian.terms
    ).items():
        if not _charge_is_zero(charge):
            # A nonzero-charge TTNO cannot be amalgamated into a neutral
            # direct-sum tensor: the charge belongs to one open operator
            # boundary tensor. Keep each charged term as its own homogeneous
            # network and let the public TreeMPO sum them.
            for where, term in sector_terms.items():
                network = _native_tree_term_network(
                    plan,
                    term,
                    _term_support(where),
                    symmetry=hamiltonian.symmetry,
                    cutoff=cutoff,
                    dtype=dtype,
                )
                networks.append(_normalize_native_term_edge_orientation(
                    network,
                    plan,
                    symmetry=hamiltonian.symmetry,
                    dtype=dtype,
                ))
            continue
        sector_hamiltonian = SymHamiltonian.from_terms(
            hamiltonian.model,
            hamiltonian.symmetry,
            sector_terms,
            parameters=hamiltonian.parameters,
        )
        sector_operator = _build_tree_operator(
            plan,
            sector_hamiltonian,
            cutoff=cutoff,
            max_bond=max_bond,
            compress=False,
            dtype=dtype,
            fermionic=fermionic,
        )
        if isinstance(sector_operator, TreeMPO):
            networks.extend(sector_operator.tree_networks)
        else:
            networks.append(sector_operator)

    operator = TreeMPO(
        plan,
        tuple(networks),
        terms=hamiltonian.terms,
        backend="symmray" if fermionic else "dense",
        fermionic=fermionic,
        symmetry=hamiltonian.symmetry,
        compressed=False,
    )
    if compress:
        operator.compress(max_bond=max_bond, cutoff=cutoff)
    if to_backend is not None:
        for network in operator.tree_networks:
            _apply_to_tensor_network_arrays(network, to_backend)
    return operator


def _build_tree_operator(
    plan,
    hamiltonian,
    *,
    cutoff=1e-12,
    max_bond=None,
    compress=True,
    dtype=None,
    fermionic=True,
):
    """Build the backend-specific tree representation used by ``TreeMPO``."""
    symmetry = hamiltonian.symmetry
    terms = hamiltonian.terms

    if any(
        not _charge_is_zero(charge)
        for charge in _terms_by_operator_charge(terms)
    ):
        # A charged TTNO carries its operator charge on an open boundary
        # tensor. Keep charged terms as separate homogeneous networks rather
        # than forcing them into the neutral direct-sum construction below.
        return tuple(
            _normalize_native_term_edge_orientation(
                _native_tree_term_network(
                    plan,
                    term,
                    _term_support(where),
                    symmetry=symmetry,
                    cutoff=cutoff,
                    dtype=dtype,
                ),
                plan,
                symmetry=symmetry,
                dtype=dtype,
            )
            for where, term in terms.items()
        )

    # The full staggered eta correlator is a symmetric rank-one pair table.
    # Compile it before falling back to one actual tree contraction per term;
    # this keeps p_eta_stag2 at a four-state tree bond for arbitrary N.
    if fermionic:
        # The factorization helper assumes every term is a two-site operator.
        # In particular, an onsite-only Hamiltonian is a valid generic TTNO
        # input but must go directly to the combined automaton.
        is_pair_table = (
            len(terms) >= 3
            and all(len(_term_support(where)) == 2 for where in terms)
        )
        pair_network = _pair_endpoint_automaton(
            plan, terms, symmetry=symmetry, cutoff=cutoff, dtype=dtype,
        ) if is_pair_table else None
        if pair_network is not None:
            return pair_network
        return _native_term_sum_tree_operator(
            plan,
            terms,
            symmetry=symmetry,
            cutoff=cutoff,
            dtype=dtype,
        )

    return _combined_tree_operator(
        plan,
        terms,
        symmetry=symmetry,
        cutoff=cutoff,
        dtype=dtype,
        fermionic=False,
    )


def build_tree_operator(
    plan,
    hamiltonian,
    *,
    max_bond=None,
    cutoff=1e-12,
    compress=True,
    dtype=None,
    fermionic=True,
    charge_sectors=False,
    to_backend=None,
):
    """Build the canonical native :class:`TreeMPO` for a ``TreePlan``.

    This function returns only the native TreePlan operator. Mixed native
    charges are combined into one ``TreeMPO`` with one homogeneous Symmray
    network per charge. With ``charge_sectors=True`` it instead returns
    ``{charge: TreeMPO}`` for callers that need separate sector objects.
    """
    from ...tensors.symmetric import SymHamiltonian

    if fermionic and not charge_sectors:
        if not isinstance(hamiltonian, SymHamiltonian):
            raise TypeError("hamiltonian must be a SymHamiltonian instance.")
        if any(
            not _charge_is_zero(charge)
            for charge in _terms_by_operator_charge(hamiltonian.terms)
        ):
            return _build_mixed_charge_tree_operator(
                plan,
                hamiltonian,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compress,
                dtype=dtype,
                fermionic=fermionic,
                to_backend=to_backend,
            )
    if not isinstance(hamiltonian, SymHamiltonian):
        raise TypeError("hamiltonian must be a SymHamiltonian instance.")
    if charge_sectors:
        sectors = {}
        for charge, sector_terms in _terms_by_operator_charge(
            hamiltonian.terms
        ).items():
            sector_hamiltonian = SymHamiltonian.from_terms(
                hamiltonian.model,
                hamiltonian.symmetry,
                sector_terms,
                parameters=hamiltonian.parameters,
            )
            sectors[charge] = build_tree_operator(
                plan,
                sector_hamiltonian,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compress,
                dtype=dtype,
                fermionic=fermionic,
                charge_sectors=False,
                to_backend=to_backend,
            )
        return sectors
    from ...tensors.symmetric import _apply_to_tensor_network_arrays

    networks = _build_tree_operator(
        plan,
        hamiltonian,
        cutoff=cutoff,
        max_bond=max_bond,
        compress=False,
        dtype=dtype,
        fermionic=fermionic,
    )
    if isinstance(networks, (tuple, list)):
        native_networks = tuple(networks)
    else:
        native_networks = (networks,)
    operator = TreeMPO(
        plan,
        native_networks,
        terms=hamiltonian.terms,
        backend="symmray" if fermionic else "dense",
        fermionic=fermionic,
        symmetry=hamiltonian.symmetry,
        cutoff=cutoff,
        compressed=False,
    )
    if compress:
        operator.compress(max_bond=max_bond, cutoff=cutoff)
    if to_backend is not None:
        for network in operator.tree_networks:
            _apply_to_tensor_network_arrays(network, to_backend)
    return operator
