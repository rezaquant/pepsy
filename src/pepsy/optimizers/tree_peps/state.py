"""A PEPS-like tensor state whose virtual graph is a spanning tree."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import quimb.tensor as qtn
import autoray as ar

from ..._internal.quimb import (
    quimb_1d_compression_function,
    require_quimb_1d_compression_method,
)
from ._compression import (
    iter_tree_compression_order,
    normalize_tree_compression_order,
    tree_edge_rank_key,
)
from .plan import TreePepsPlan

__all__ = ["TreePeps"]


def _normalize_compression_mode(mode):
    """Normalize the local bond-compression decomposition mode."""

    mode = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "svd": "direct",
        "eigh": "dm",
        "density_matrix": "dm",
        "densitymatrix": "dm",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"direct", "dm", "sdc", "src", "zipup"}:
        raise ValueError(
            "compression_mode must be 'direct', 'dm', 'sdc', 'src', or "
            "'zipup'."
        )
    return mode


def _compression_method(mode):
    """Return the Quimb decomposition used by a compression mode."""

    mode = _normalize_compression_mode(mode)
    if mode == "dm":
        return "svd:eig"
    if mode == "src":
        # Branching trees cannot use Quimb's chain-only SRC environment
        # sweep, but each local dense edge split can use randomized SVD.
        return "svd:rand"
    if mode == "zipup":
        raise NotImplementedError(
            "compression_mode='zipup' is only available for a complete "
            "path compression or operator-state two-layer application."
        )
    # ``sdc`` is the deterministic successive edge sweep for a tree. The
    # actual Quimb SDC kernel is selected separately for path topologies.
    return "svd"


class TreePeps(qtn.TensorNetworkGenVector):
    """A lattice-aware tensor network with a tree of virtual bonds.

    Every lattice site owns one tensor and one physical index.  The retained
    virtual bonds are described by :class:`TreePepsPlan` and form a spanning
    tree of the underlying 2D or 3D lattice.  Tags intentionally expose both
    coordinate and logical identities, for example ``I1,2``, ``I7``, and
    ``N7``.  The physical index is a single coordinate-style ``k1,2`` index;
    :meth:`site_ind_1d` is an alias for looking it up by logical id.

    This initial implementation uses Quimb's generic tensor-network engine,
    which keeps the representation compatible with PEPS-style tags without
    imposing the rectangular PEPS bond pattern on a tree state.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_tree_peps_plan",
        "_coord_site_tag_id",
        "_coord_site_ind_id",
        "_logical_site_tag_id",
        "_node_tag_id",
        "_tree_bond_id",
        "_canonical_region",
    )

    def __init__(
        self,
        ts=(),
        *,
        plan: TreePepsPlan | None = None,
        coord_site_tag_id: str | None = None,
        coord_site_ind_id: str | None = None,
        logical_site_tag_id: str = "I{}",
        node_tag_id: str = "N{}",
        tree_bond_id: str = "_tpb{}_{}",
        canonical_region=None,
        **tn_opts,
    ) -> None:
        if isinstance(ts, TreePeps):
            if plan is None:
                plan = ts.plan
            if coord_site_tag_id is None:
                coord_site_tag_id = ts._coord_site_tag_id
            if coord_site_ind_id is None:
                coord_site_ind_id = ts._coord_site_ind_id
            logical_site_tag_id = ts._logical_site_tag_id
            node_tag_id = ts._node_tag_id
            tree_bond_id = ts._tree_bond_id
            if canonical_region is None:
                canonical_region = ts._canonical_region
        if plan is None:
            raise TypeError("TreePeps requires a TreePepsPlan")
        if not isinstance(plan, TreePepsPlan):
            raise TypeError("plan must be a TreePepsPlan")

        if coord_site_tag_id is None:
            coord_site_tag_id = _default_format("I", plan.ndim)
        if coord_site_ind_id is None:
            coord_site_ind_id = _default_format("k", plan.ndim)

        # Quimb passes ``virtual`` back to the constructor when copying a
        # TensorNetwork.  TreePeps always owns physical site indices, so keep
        # the generic vector in non-virtual mode and consume that copy hint.
        tn_opts.pop("virtual", None)
        super().__init__(ts, virtual=False, **tn_opts)
        self._tree_peps_plan = plan
        # Quimb's inherited ``nsites`` property and related helpers read this
        # storage-level site list directly. Keep it with the copied metadata.
        self._sites = tuple(range(plan.size))
        self._coord_site_tag_id = str(coord_site_tag_id)
        self._coord_site_ind_id = str(coord_site_ind_id)
        self._logical_site_tag_id = str(logical_site_tag_id)
        self._node_tag_id = str(node_tag_id)
        self._tree_bond_id = str(tree_bond_id)
        self._canonical_region = (
            None
            if canonical_region is None
            else frozenset(plan.resolve_site(site) for site in canonical_region)
        )

    @classmethod
    def from_plan(
        cls,
        plan: TreePepsPlan,
        *,
        phys_dim: int | Sequence[int] | Mapping = 2,
        dtype=complex,
        **tn_opts,
    ) -> "TreePeps":
        """Create a product-state tree with all virtual bond dimensions one."""

        dims = _site_dimensions(plan, phys_dim, name="phys_dim")
        tensors = []
        for q in range(plan.size):
            shape = (dims[q],) + (1,) * len(plan.neighbors(q))
            data = np.zeros(shape, dtype=dtype)
            data[(0,) * len(shape)] = 1
            tensors.append(
                qtn.Tensor(
                    data=data,
                    inds=(cls._site_ind_for_plan(plan, q),)
                    + tuple(cls._bond_ind_for_plan(plan, q, n) for n in plan.neighbors(q)),
                    tags=cls._tags_for_plan(plan, q),
                )
            )
        state = cls(tensors, plan=plan, **tn_opts)
        state._canonical_region = frozenset({plan.root})
        state._set_isometry_metadata_from_region({plan.root})
        state.validate()
        return state

    @classmethod
    def rand(
        cls,
        plan: TreePepsPlan,
        *,
        bond_dim: int = 2,
        phys_dim: int | Sequence[int] | Mapping = 2,
        dtype=complex,
        seed=None,
        canonicalize: bool = False,
        **tn_opts,
    ) -> "TreePeps":
        """Create a random tree state with uniform virtual bond dimensions."""

        if not isinstance(bond_dim, Integral) or int(bond_dim) < 1:
            raise ValueError("bond_dim must be a positive integer")
        bond_dim = int(bond_dim)
        dims = _site_dimensions(plan, phys_dim, name="phys_dim")
        dtype = np.dtype(dtype)
        rng = np.random.default_rng(seed)
        tensors = []
        for q in range(plan.size):
            shape = (dims[q],) + (bond_dim,) * len(plan.neighbors(q))
            data = rng.standard_normal(shape)
            if np.issubdtype(dtype, np.complexfloating):
                data = data + 1j * rng.standard_normal(shape)
            data = data.astype(dtype, copy=False)
            tensors.append(
                qtn.Tensor(
                    data=data,
                    inds=(cls._site_ind_for_plan(plan, q),)
                    + tuple(cls._bond_ind_for_plan(plan, q, n) for n in plan.neighbors(q)),
                    tags=cls._tags_for_plan(plan, q),
                )
            )
        state = cls(tensors, plan=plan, **tn_opts)
        if canonicalize:
            state.canonize_to(plan.root, inplace=True)
        else:
            state.validate()
        return state

    @property
    def plan(self) -> TreePepsPlan:
        """The lattice and spanning-tree plan for this state."""

        return self._tree_peps_plan

    @property
    def map_mode(self) -> str | None:
        """Canonical lattice spanning-tree mode, if the plan has one."""

        return self.plan.map_mode

    @property
    def plan_signature(self):
        """Immutable geometry signature used by state/operator adapters."""

        return (
            self.plan.shape,
            self.plan.coordinates,
            self.plan.tree_edges,
            self.plan.root,
            self.plan.max_virtual_degree,
            self.plan.order,
            self.plan.tree_order,
            self.plan.topology,
            self.plan.boundary,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.plan.shape

    @property
    def ndim(self) -> int:
        return self.plan.ndim

    @property
    def nqubits(self) -> int:
        """Number of physical lattice sites (TTN name-parity alias)."""

        return self.plan.size

    @property
    def nsites(self) -> int:
        """Number of physical lattice sites."""

        return self.plan.size

    @property
    def root(self) -> int:
        """Logical site used as the root of the retained virtual tree."""

        return self.plan.root

    @property
    def top_arity(self) -> int:
        """Number of virtual child bonds entering the rooted site."""

        return len(self.plan.children[self.plan.root])

    @property
    def max_virtual_degree(self) -> int:
        """The largest number of retained virtual bonds at one site."""

        return self.plan.max_degree

    @property
    def topology(self) -> str:
        """The virtual topology contract, ``'tree'`` or explicit ``'path'``."""

        return self.plan.topology

    @property
    def is_branching(self) -> bool:
        """Whether this state has a site with at least three virtual bonds."""

        return self.plan.is_branching

    @property
    def is_mps_topology(self) -> bool:
        """Whether the retained virtual graph is MPS-like."""

        return self.plan.is_mps_topology

    @property
    def max_rank(self) -> int:
        """The largest local tensor rank, including its one physical leg."""

        return self.plan.max_tensor_rank

    @property
    def max_tensor_rank(self) -> int:
        """TTN name-parity alias for :attr:`max_rank`."""

        return self.max_rank

    def is_binary(self, *, allow_ternary_root=True) -> bool:
        """Whether the rooted TreePeps has at most two children per site.

        A degree-three root is allowed by default, matching the binary-tree
        convention used by :class:`TreeTensorNetwork`. ``span-middle`` may
        additionally use degree-four backbone sites.
        """

        root = self.plan.root
        if not allow_ternary_root and len(self.plan.children[root]) > 2:
            return False
        return all(
            len(children) <= 2
            for site, children in self.plan.children.items()
            if site != root
        )

    @property
    def rank(self) -> int:
        """Compatibility spelling for the maximum local tensor rank."""

        return self.max_rank

    def tensor_rank(self, site) -> int:
        """Return the local tensor rank at a logical site or coordinate."""

        return self.plan.tensor_rank(site)

    @property
    def sites(self) -> tuple[int, ...]:
        """Logical site ids in stable one-dimensional order."""

        return tuple(range(self.plan.size))

    @property
    def coordinates(self) -> tuple[tuple[int, ...], ...]:
        return self.plan.coordinates

    @property
    def canonical_region(self):
        """The region most recently used as a canonical center, if known."""

        return self._canonical_region

    @canonical_region.setter
    def canonical_region(self, region):
        if region is None:
            self._canonical_region = None
            return
        if isinstance(region, Integral):
            region = (region,)
        region = frozenset(self.plan.resolve_site(site) for site in region)
        if not region or not self.plan.is_connected(region):
            raise ValueError("canonical_region must be a non-empty connected subtree")
        self._canonical_region = region

    @property
    def orthogonality_center(self):
        """The unique canonical center site, or ``None`` for a region."""

        if self._canonical_region is not None and len(self._canonical_region) == 1:
            return next(iter(self._canonical_region))
        return None

    @orthogonality_center.setter
    def orthogonality_center(self, site):
        if site is None:
            self._canonical_region = None
        else:
            self._canonical_region = frozenset({self.plan.resolve_site(site)})

    def coordinate(self, site, *rest) -> tuple[int, ...]:
        return self.plan.coordinate(self.plan.resolve_site(site, *rest))

    def logical_site(self, *coordinate) -> int:
        if len(coordinate) == 1 and isinstance(coordinate[0], (tuple, list)):
            coordinate = tuple(coordinate[0])
        return self.plan.logical_site(tuple(coordinate))

    def site_tag(self, site, *rest) -> str:
        """Return the Quimb-style coordinate site tag ``I{x},{y[,z]}``."""

        coordinate = self.coordinate(site, *rest)
        return self._coord_site_tag_id.format(*coordinate)

    @property
    def site_tags(self) -> tuple[str, ...]:
        """Coordinate site tags in logical one-dimensional order."""

        return tuple(self.site_tag(q) for q in self.sites)

    @property
    def x_tag_id(self) -> str:
        return "X{}"

    @property
    def y_tag_id(self) -> str:
        return "Y{}"

    @property
    def z_tag_id(self) -> str:
        return "Z{}"

    def x_tag(self, x: int) -> str:
        """Return the Quimb-style x-axis tag for a lattice coordinate."""

        return self.x_tag_id.format(int(x))

    def y_tag(self, y: int) -> str:
        """Return the Quimb-style y-axis tag for a lattice coordinate."""

        return self.y_tag_id.format(int(y))

    def z_tag(self, z: int) -> str:
        """Return the z-axis tag for a 3D lattice coordinate."""

        if self.ndim != 3:
            raise ValueError("z_tag is only available for a 3D TreePeps")
        return self.z_tag_id.format(int(z))

    def axis_tags(self, site) -> tuple[str, ...]:
        """Return coordinate-axis tags carried by the site's tensor."""

        coordinate = self.coordinate(site)
        tags = [self.x_tag(coordinate[0]), self.y_tag(coordinate[1])]
        if self.ndim == 3:
            tags.append(self.z_tag(coordinate[2]))
        return tuple(tags)

    def logical_site_tag(self, site) -> str:
        """Return the stable one-dimensional logical site tag ``I{q}``."""

        q = self.plan.resolve_site(site)
        return self._logical_site_tag_id.format(q)

    def node_tag(self, site) -> str:
        """Return the structural tree tag ``N{q}``."""

        q = self.plan.resolve_site(site)
        return self._node_tag_id.format(q)

    def site_ind(self, site, *rest) -> str:
        """Return the single physical index for a coordinate or logical site."""

        coordinate = self.coordinate(site, *rest)
        return self._coord_site_ind_id.format(*coordinate)

    @property
    def site_inds(self) -> tuple[str, ...]:
        """Physical indices in logical one-dimensional order."""

        return tuple(self.site_ind_1d(q) for q in self.sites)

    def site_ind_1d(self, site) -> str:
        """Return the same physical index addressed by logical id ``q``."""

        return self.site_ind(self.plan.resolve_site(site))

    def tree_bond_ind(self, site0, site1) -> str:
        """Return the initial virtual index name for a tree edge."""

        q0 = self.plan.resolve_site(site0)
        q1 = self.plan.resolve_site(site1)
        if q1 not in self.plan.neighbors(q0):
            raise ValueError(f"sites {q0} and {q1} are not adjacent in the tree")
        return self._tree_bond_id.format(*sorted((q0, q1)))

    def node_tid(self, site) -> int:
        """Return the live tensor id for a logical or coordinate site."""

        q = self.plan.resolve_site(site)
        cache = self.__dict__.get("_tree_peps_tid_cache")
        if cache is None:
            cache = self.__dict__["_tree_peps_tid_cache"] = {}
        tid = cache.get(q)
        if tid is not None and tid in self.tensor_map:
            return tid
        tid = next(iter(self.tag_map[self.node_tag(q)]))
        cache[q] = tid
        return tid

    def node_tensor(self, site):
        """Return the live tensor at a logical or coordinate site."""

        return self.tensor_map[self.node_tid(site)]

    def bond(self, site0, site1) -> str:
        """Return the live shared virtual index for an adjacent tree edge."""

        q0 = self.plan.resolve_site(site0)
        q1 = self.plan.resolve_site(site1)
        if q1 not in self.plan.neighbors(q0):
            raise ValueError(f"sites {q0} and {q1} are not adjacent in the tree")
        shared = qtn.bonds(self.node_tensor(q0), self.node_tensor(q1))
        if len(shared) != 1:
            raise ValueError(f"sites {q0} and {q1} must share exactly one bond; found {shared}")
        return next(iter(shared))

    def neighbors(self, site):
        return self.plan.neighbors(site)

    def path(self, site0, site1):
        return self.plan.path(site0, site1)

    def node_path(self, site0, site1):
        """TTN name-parity alias for the unique site path."""

        return self.path(site0, site1)

    def tree_distance(self, site0, site1) -> int:
        """Return the number of retained tree bonds between two sites."""

        return len(self.path(site0, site1)) - 1

    def is_leaf(self, site) -> bool:
        """Whether ``site`` is a leaf in the tree rooted at ``root``."""

        q = self.plan.resolve_site(site)
        return not self.plan.children[q]

    def parent(self, site):
        """Return the parent site in the rooted retained tree."""

        q = self.plan.resolve_site(site)
        return self.plan.parent[q]

    def children(self, site):
        """Return the child sites in the rooted retained tree."""

        q = self.plan.resolve_site(site)
        return self.plan.children[q]

    def steiner_nodes(self, sites):
        """Return the minimal connected tree span of ``sites``."""

        return self.plan.subtree_span(sites)

    def subtree_span(self, sites):
        """Return the minimal connected tree span of ``sites``."""

        return self.plan.subtree_span(sites)

    def max_bond(self) -> int:
        """Return the largest live virtual bond dimension."""

        return max(self.bond_sizes().values(), default=1)

    def bond_sizes(self) -> dict[tuple[int, int], int]:
        """Return live dimensions keyed by undirected tree edge."""

        return {
            tuple(sorted((q0, q1))): int(self.ind_size(self.bond(q0, q1)))
            for q0, q1 in self.plan.tree_edges
        }

    def bond_report(self) -> dict[str, object]:
        """Return a compact health report for the retained virtual bonds."""

        bond_sizes = self.bond_sizes()
        dimensions = tuple(bond_sizes.values())
        return {
            "max_bond": max(dimensions, default=1),
            "mean_bond": (
                float(sum(dimensions) / len(dimensions))
                if dimensions else 1.0
            ),
            "n_bonds": len(dimensions),
            "n_tensors": self.num_tensors,
            "topology": self.topology,
            "is_branching": self.is_branching,
            "max_virtual_degree": self.max_virtual_degree,
            "max_tensor_rank": self.max_tensor_rank,
            "bond_sizes": bond_sizes,
        }

    def validate(self, *, check_canonical=False, tol=1e-9):
        """Validate tags, physical legs, and the live virtual tree graph."""

        if self.plan.max_virtual_degree > 4 or self.plan.max_degree > 4:
            raise ValueError("TreePeps tensors may have at most four virtual bonds")

        physical_counts = Counter()
        virtual_counts = Counter()
        for q in self.sites:
            tensor = self.node_tensor(q)
            required_tags = {
                self.site_tag(q),
                self.logical_site_tag(q),
                self.node_tag(q),
                *self.axis_tags(q),
            }
            if not required_tags.issubset(tensor.tags):
                raise ValueError(f"tensor at site {q} is missing TreePeps tags")
            if len(tensor.inds) > 5:
                raise ValueError(
                    f"tensor at site {q} exceeds TreePeps rank five "
                    "(one physical leg plus four virtual bonds)"
                )
            physical = self.site_ind_1d(q)
            if physical not in tensor.inds:
                raise ValueError(f"tensor at site {q} is missing physical index {physical}")
            physical_counts[physical] += 1
            for index in tensor.inds:
                if index != physical:
                    virtual_counts[index] += 1

        if (
            any(count != 1 for count in physical_counts.values())
            or len(physical_counts) != self.plan.size
        ):
            raise ValueError("each TreePeps site must have exactly one physical index")
        if any(count != 2 for count in virtual_counts.values()):
            raise ValueError("every TreePeps virtual index must connect exactly two tensors")

        for q0, q1 in self.plan.tree_edges:
            if len(qtn.bonds(self.node_tensor(q0), self.node_tensor(q1))) != 1:
                raise ValueError(f"tree edge ({q0}, {q1}) is not a single live bond")
        for q0 in self.sites:
            for q1 in range(q0 + 1, self.plan.size):
                shared = qtn.bonds(self.node_tensor(q0), self.node_tensor(q1))
                if q1 not in self.plan.neighbors(q0) and shared:
                    raise ValueError(f"non-tree sites ({q0}, {q1}) share a virtual bond")
        if self.canonical_region is not None:
            if not self.canonical_region or not self.plan.is_connected(self.canonical_region):
                raise ValueError("canonical_region must be a connected subtree")
            if check_canonical:
                self.validate_isometry_metadata()
                if not self.is_subtree_canonical_form(tol=tol):
                    raise ValueError(
                        "tracked canonical_region does not satisfy its isometry checks"
                    )
        return True

    def isometry_direction(self, site):
        """Return the neighbor toward which a site is currently isometric."""

        q = self.plan.resolve_site(site)
        tensor = self.node_tensor(q)
        left_inds = tensor.left_inds
        if left_inds is None:
            return None
        left_inds = set(left_inds)
        for neighbor in self.plan.neighbors(q):
            bond = self.bond(q, neighbor)
            if bond not in left_inds and left_inds == set(tensor.inds) - {bond}:
                return neighbor
        return None

    def isometry_map(self, region=None):
        """Return the live outward-to-region isometry directions."""

        if region is None:
            region = self.canonical_region
        if region is None:
            region = frozenset()
        else:
            region = frozenset(self.plan.resolve_site(site) for site in region)
        result = {}
        for q in self.sites:
            result[q] = None if q in region else self.isometry_direction(q)
        return result

    def _set_isometry_metadata_from_region(self, region):
        """Record ``left_inds`` after a completed canonicalization sweep."""

        region = frozenset(self.plan.resolve_site(site) for site in region)
        if not region or not self.plan.is_connected(region):
            raise ValueError("canonical region must be a non-empty connected subtree")
        for q in self.sites:
            tensor = self.node_tensor(q)
            if q in region:
                tensor.modify(left_inds=None)
                continue
            toward = self._toward_region(q, region)
            bond = self.bond(q, toward)
            tensor.modify(left_inds=tuple(ind for ind in tensor.inds if ind != bond))
        return self

    def validate_isometry_metadata(self, region=None):
        """Validate the local ``left_inds`` proofs against a canonical region."""

        if region is None:
            region = self.canonical_region
        else:
            region = frozenset(self.plan.resolve_site(site) for site in region)
            if not region or not self.plan.is_connected(region):
                raise ValueError("region must be a non-empty connected subtree")

        for q in self.sites:
            tensor = self.node_tensor(q)
            direction = self.isometry_direction(q)
            if tensor.left_inds is not None and direction is None:
                raise ValueError(f"site {q} has left_inds that do not identify one tree direction")
            if region is not None and q not in region:
                expected = self._toward_region(q, region)
                if direction != expected:
                    raise ValueError(
                        f"site {q} must be isometric toward {expected}, "
                        f"but left_inds point toward {direction}"
                    )
        return self

    def can_skip_canonize(self, site0, site1, *, absorb="right"):
        """Whether ``left_inds`` proves that an edge QR can be skipped."""

        if absorb not in {"left", "right"}:
            raise ValueError("absorb must be 'left' or 'right'")
        q0 = self.plan.resolve_site(site0)
        q1 = self.plan.resolve_site(site1)
        bond = self.bond(q0, q1)
        q = q0 if absorb == "right" else q1
        tensor = self.node_tensor(q)
        if tensor.left_inds is None:
            return False
        return set(tensor.left_inds) == set(tensor.inds) - {bond}

    def _toward_region(self, site, region):
        q = self.plan.resolve_site(site)
        region = frozenset(region)
        if q in region:
            return None
        target = min(region, key=lambda candidate: len(self.plan.path(q, candidate)))
        return self.plan.path(q, target)[1]

    def _isometry_toward(self, site, toward, tol=1e-9):
        """Check ``T.conj().T`` on all non-toward legs numerically."""

        tensor = self.node_tensor(site)
        bond = self.bond(site, toward)
        other_inds = [ind for ind in tensor.inds if ind != bond]
        bond_axis = tensor.inds.index(bond)
        other_axes = [tensor.inds.index(ind) for ind in other_inds]
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
        """Check the defining isometry condition outside a canonical region."""

        if sites is None:
            region = self.canonical_region
            if region is None:
                return False
        else:
            if isinstance(sites, Integral):
                sites = (sites,)
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
        """Check whether the tree is in one-site canonical form."""

        if center is None:
            center = self.orthogonality_center
        if center is None:
            return False
        return self.is_subtree_canonical_form({self.plan.resolve_site(center)}, tol=tol)

    def invalidate_canonical_form(self):
        """Forget canonical metadata after direct tensor mutation."""

        self._canonical_region = None
        for q in self.sites:
            self.node_tensor(q).modify(left_inds=None)
        return self

    def _sync_info_c(self, info_c):
        if info_c is None:
            return None
        if not hasattr(info_c, "__setitem__"):
            raise TypeError("info_c must be a mutable mapping when supplied")
        center = self.orthogonality_center
        info_c["cur_orthog"] = None if center is None else (center, center)
        info_c["canonical_region"] = self.canonical_region
        info_c["isometry_map"] = self.isometry_map()
        info_c["left_inds"] = {
            q: None
            if self.node_tensor(q).left_inds is None
            else tuple(self.node_tensor(q).left_inds)
            for q in self.sites
        }
        return info_c

    def gate_inds_(self, *args, **kwargs):
        """Apply a Quimb gate and invalidate canonical metadata."""

        result = qtn.TensorNetworkGenVector.gate_inds_(self, *args, **kwargs)
        self.invalidate_canonical_form()
        return self if result is None else result

    def canonize_between(self, *args, **kwargs):
        """Canonicalize one live edge and invalidate tracked metadata."""

        qtn.TensorNetworkGenVector.canonize_between(self, *args, **kwargs)
        self.invalidate_canonical_form()
        return self

    def compress_between(self, *args, **kwargs):
        """Compress through Quimb and invalidate tracked metadata."""

        qtn.TensorNetworkGenVector.compress_between(self, *args, **kwargs)
        self.invalidate_canonical_form()
        return self

    def canonize_around_(self, *args, **kwargs):
        """Canonicalize around tags through Quimb and invalidate metadata."""

        qtn.TensorNetworkGenVector.canonize_around_(self, *args, **kwargs)
        self.invalidate_canonical_form()
        return self

    def canonize_to(
        self,
        site,
        *,
        absorb="right",
        inplace=False,
        info_c=None,
        _force_full=False,
        **canonize_opts,
    ):
        """Canonicalize the full tree around one site."""

        q = self.plan.resolve_site(site)
        work = self if inplace else self.copy()
        if work.canonical_region is not None and not canonize_opts and not _force_full:
            if absorb not in {"left", "right"}:
                raise ValueError("canonical movement requires absorb='left' or 'right'")
            return work.shift_orthogonality_center(
                q,
                absorb=absorb,
                info_c=info_c,
            )
        opts = {"method": "qr", "cutoff": 0.0}
        opts.update(canonize_opts)
        work.canonize_around_([work.node_tag(q)], which="any", absorb=absorb, **opts)
        work._canonical_region = frozenset({q})
        work._set_isometry_metadata_from_region({q})
        work.validate()
        work._sync_info_c(info_c)
        return work

    def canonicalize(self, center=None, *, inplace=True, info_c=None, **canonize_opts):
        """Canonicalize the tree around ``center`` like a Quimb MPS."""

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
        """In-place alias for :meth:`canonicalize`."""

        return self.canonicalize(
            center=center,
            inplace=True,
            info_c=info_c,
            **canonize_opts,
        )

    canonize = canonicalize_

    def _canonicalize_region_fast(self, region, *, absorb="right", **canonize_opts):
        """Canonicalize only the branches outside ``region``.

        The tree is peeled from the farthest leaves inward.  A tensor whose
        live ``left_inds`` already prove the required isometry is left
        untouched, so moving between compatible canonical regions can avoid
        all QR work on their common exterior.
        """

        if absorb not in {"left", "right"}:
            raise ValueError("canonical region movement requires absorb='left' or 'right'")
        region = frozenset(region)
        order = sorted(
            (q for q in self.sites if q not in region),
            key=lambda q: (
                -len(self.plan.path(q, min(region, key=lambda r: len(self.plan.path(q, r))))),
                q,
            ),
        )
        for q in order:
            toward = self._toward_region(q, region)
            if absorb == "right":
                source, target = q, toward
            else:
                source, target = toward, q
            if self.can_skip_canonize(source, target, absorb=absorb):
                continue
            self.canonize_edge_(
                source,
                target,
                absorb=absorb,
                **canonize_opts,
            )
        self._canonical_region = region
        self._set_isometry_metadata_from_region(region)
        return self

    def _recover_center_from_region(self, region, target, *, absorb="right"):
        """Peel a tracked canonical region down to one site."""

        region = set(region)
        target = self.plan.resolve_site(target)
        if target not in region:
            raise ValueError("target center must lie inside canonical_region")
        if absorb not in {"left", "right"}:
            raise ValueError("canonical movement requires absorb='left' or 'right'")

        remaining = set(region)
        while len(remaining) > 1:
            leaves = [
                q
                for q in remaining
                if q != target
                and sum(neighbor in remaining for neighbor in self.plan.neighbors(q)) == 1
            ]
            if not leaves:
                raise ValueError("canonical_region is not a connected tree")
            q = min(leaves)
            neighbor = next(
                neighbor for neighbor in self.plan.neighbors(q) if neighbor in remaining
            )
            if absorb == "right":
                source, destination = q, neighbor
            else:
                source, destination = neighbor, q
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
        """Canonicalize around a connected tree region."""

        if isinstance(sites, Integral):
            sites = (sites,)
        sites = tuple(self.plan.resolve_site(site) for site in sites)
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

    def canonize_subtree_(self, sites, *, span=False, absorb="right", info_c=None, **canonize_opts):
        """In-place alias for :meth:`canonize_subtree`."""

        return self.canonize_subtree(
            sites,
            span=span,
            absorb=absorb,
            inplace=True,
            info_c=info_c,
            **canonize_opts,
        )

    def canonize_around_qubits(
        self,
        sites,
        *,
        absorb="right",
        inplace=False,
        info_c=None,
        **canonize_opts,
    ):
        """Canonicalize around the minimal tree span of physical sites."""

        return self.canonize_subtree(
            sites,
            span=True,
            absorb=absorb,
            inplace=inplace,
            info_c=info_c,
            **canonize_opts,
        )

    def canonize_around_qubits_(
        self,
        sites,
        *,
        absorb="right",
        info_c=None,
        **canonize_opts,
    ):
        """In-place alias for :meth:`canonize_around_qubits`."""

        return self.canonize_around_qubits(
            sites,
            absorb=absorb,
            inplace=True,
            info_c=info_c,
            **canonize_opts,
        )

    def shift_orthogonality_center(
        self,
        site,
        *,
        absorb="right",
        info_c=None,
        _skip_validate=False,
        **canonize_opts,
    ):
        """Move a known one-site canonical center along the tree."""

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
                entry = min(
                    region,
                    key=lambda candidate: len(self.plan.path(candidate, q)),
                )
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
                raise ValueError("absorb must be 'right' or 'left'")
            self.canonize_edge_(source, target, absorb=absorb, **canonize_opts)
        self._canonical_region = frozenset({q})
        if not _skip_validate:
            self.validate()
        self._sync_info_c(info_c)
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
        """Canonicalize one tree edge in place and move a known center."""

        q0 = self.plan.resolve_site(site0)
        q1 = self.plan.resolve_site(site1)
        if q1 not in self.plan.neighbors(q0):
            raise ValueError(f"sites {q0} and {q1} are not adjacent in the tree")
        if absorb not in {"left", "right", "both"}:
            raise ValueError("absorb must be 'left', 'right', or 'both'")
        previous = self.orthogonality_center
        if _isometry_proven or self.can_skip_canonize(q0, q1, absorb=absorb):
            if absorb == "right" and previous in {q0, q1}:
                self._canonical_region = frozenset({q1})
            elif absorb == "left" and previous in {q0, q1}:
                self._canonical_region = frozenset({q0})
            else:
                self._canonical_region = None
            self._sync_info_c(info_c)
            return self
        opts = {"method": "qr", "cutoff": 0.0}
        opts.update(canonize_opts)
        qtn.TensorNetworkGenVector.canonize_between(
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
        bond = self.bond(q0, q1)
        if absorb == "right":
            self.node_tensor(q0).modify(
                left_inds=tuple(ind for ind in self.node_tensor(q0).inds if ind != bond)
            )
            if self.orthogonality_center == q1:
                self.node_tensor(q1).modify(left_inds=None)
        elif absorb == "left":
            self.node_tensor(q1).modify(
                left_inds=tuple(ind for ind in self.node_tensor(q1).inds if ind != bond)
            )
            if self.orthogonality_center == q0:
                self.node_tensor(q0).modify(left_inds=None)
        self._sync_info_c(info_c)
        return self

    def _compress_path_region_1d(
        self,
        region,
        *,
        max_bond,
        cutoff,
        cutoff_mode,
        compression_mode,
        compression_seed=None,
    ):
        """Apply Quimb's environment compressor to a path-shaped region.

        ``TreePeps`` deliberately wraps a plain temporary ``TensorNetwork``
        here. Quimb's 1D compressors reconstruct their input class, whereas
        this class needs its plan and coordinate metadata retained explicitly.
        The temporary network keeps the original site tags and boundary bonds;
        only the resulting tensors in ``region`` are installed back.
        """
        compression_mode = _normalize_compression_mode(compression_mode)
        if compression_mode not in {"sdc", "src", "zipup"}:
            return False
        if not self.plan.is_mps_topology or len(region) <= 1:
            return self.plan.is_mps_topology
        if cutoff is None:
            cutoff = 1e-10
        cutoff = float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if max_bond is None:
            if compression_mode in {"sdc", "zipup"} and float(cutoff) == 0.0:
                return False
            if compression_mode == "src":
                raise ValueError(
                    "compression_mode='src' requires a finite max_bond/chi."
                )
        if compression_mode == "sdc":
            require_quimb_1d_compression_method("sdc")
        compressor = quimb_1d_compression_function(compression_mode)
        if not callable(compressor):
            raise NotImplementedError(
                f"Quimb compression method {compression_mode!r} is not "
                "available in the installed Quimb build."
            )

        region = frozenset(self.plan.resolve_site(site) for site in region)
        endpoints = sorted(
            site for site in region
            if sum(neighbor in region for neighbor in self.plan.neighbors(site)) <= 1
        )
        if len(endpoints) != 2:
            raise ValueError("a non-trivial path compression region needs two endpoints")
        order = self.plan.path(endpoints[0], endpoints[1])
        if set(order) != set(region):
            raise ValueError("path compression region must be connected")

        temporary = qtn.TensorNetwork(
            [self.node_tensor(site).copy() for site in order]
        )
        if hasattr(self, "exponent"):
            temporary.exponent = self.exponent
        options = {
            "max_bond": None if max_bond is None else int(max_bond),
            "cutoff": float(cutoff),
            "site_tags": [self.site_tag(site) for site in order],
            "permute_arrays": False,
            "canonize": True,
            "inplace": False,
        }
        if compression_mode in {"sdc", "zipup"}:
            options["cutoff_mode"] = cutoff_mode
        if compression_mode == "src" and compression_seed is not None:
            options["seed"] = int(compression_seed)
        result = compressor(temporary, **options)

        for site in order:
            tag = self.site_tag(site)
            tids = tuple(result.tag_map[tag])
            if len(tids) != 1:
                raise ValueError(
                    f"path compressor did not preserve a unique tensor for site {site}"
                )
            compressed = result.tensor_map[tids[0]]
            self.node_tensor(site).modify(
                data=compressed.data,
                inds=compressed.inds,
                tags=compressed.tags,
                left_inds=compressed.left_inds,
            )
        if hasattr(result, "exponent"):
            self.exponent = result.exponent
        self._canonical_region = None
        return True

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
        compression_mode="direct",
        compression_seed=None,
        inplace=False,
        info_c=None,
        **compress_opts,
    ):
        """Compress one tree edge using Quimb's generic tree contraction."""

        q0 = self.plan.resolve_site(site0)
        q1 = self.plan.resolve_site(site1)
        if q1 not in self.plan.neighbors(q0):
            raise ValueError(f"sites {q0} and {q1} are not adjacent in the tree")
        work = self if inplace else self.copy()
        work.compress_edge_(
            q0,
            q1,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            absorb=absorb,
            reduced=reduced,
            compression_mode=compression_mode,
            compression_seed=compression_seed,
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
        compression_mode="direct",
        compression_seed=None,
        info_c=None,
        _validate=True,
        **compress_opts,
    ):
        """In-place compression of one edge with canonical-center tracking.

        ``compression_mode="direct"`` uses SVD, ``"dm"`` uses the
        density-matrix-equivalent local ``svd:eig`` decomposition, and
        ``"src"`` uses randomized SVD on the local edge split. ``"sdc"``
        selects the deterministic successive sweep; ``"zipup"`` is
        available for path operator-state compression. On a path, the
        higher-level whole/subtree methods use Quimb's environment
        compressor for these multi-tensor methods.
        """

        return self._compress_edge_inplace(
            site0,
            site1,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            absorb=absorb,
            reduced=reduced,
            compression_mode=compression_mode,
            compression_seed=compression_seed,
            info_c=info_c,
            _validate=_validate,
            **compress_opts,
        )

    def _compress_edge_inplace(
        self,
        site0,
        site1,
        *,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        absorb="right",
        reduced=True,
        compression_mode="direct",
        compression_seed=None,
        info_c=None,
        _validate=True,
        **compress_opts,
    ):
        q0 = self.plan.resolve_site(site0)
        q1 = self.plan.resolve_site(site1)
        if q1 not in self.plan.neighbors(q0):
            raise ValueError(f"sites {q0} and {q1} are not adjacent in the tree")
        if absorb not in {"left", "right", "both"}:
            raise ValueError("absorb must be 'left', 'right', or 'both'")
        if cutoff is None:
            cutoff = 1e-10
        cutoff = float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if max_bond is not None:
            max_bond = int(max_bond)
            if max_bond < 1:
                raise ValueError("max_bond must be at least one")

        compression_mode = _normalize_compression_mode(compression_mode)
        if compression_mode == "src" and max_bond is None:
            raise ValueError(
                "compression_mode='src' requires a finite max_bond/chi."
            )

        bond = self.bond(q0, q1)
        before_bond = int(self.ind_size(bond))
        if cutoff == 0.0 and (max_bond is None or before_bond <= max_bond):
            return self.canonize_edge_(
                q0,
                q1,
                absorb=absorb,
                info_c=info_c,
                **compress_opts,
            )

        compress_opts.setdefault("method", _compression_method(compression_mode))
        if compression_mode == "src" and compression_seed is not None:
            compress_opts.setdefault("seed", int(compression_seed))
        previous = self.orthogonality_center
        qtn.TensorNetworkGenVector.compress_between(
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

    def _track_edge_center(self, q0, q1, absorb, *, previous):
        if absorb == "right" and previous in {q0, q1}:
            self._canonical_region = frozenset({q1})
        elif absorb == "left" and previous in {q0, q1}:
            self._canonical_region = frozenset({q0})
        else:
            self._canonical_region = None

    def compress(
        self,
        form=None,
        *,
        center=None,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        reduced=True,
        compression_mode="direct",
        compression_seed=None,
        order="rank",
        info_c=None,
    ):
        """Compress the tree inward toward a selected canonical center.

        ``order="rank"`` greedily removes the currently cheapest leaf
        branch from the live tree, using physical and virtual dimensions
        after each reduction. ``order="depth"`` retains the simple
        farthest-first schedule. Both policies preserve the selected
        ``TreePepsPlan`` topology.
        """

        if form is not None:
            if center is not None:
                raise TypeError("specify either form or center, not both")
            if isinstance(form, Integral):
                center = form
            elif form in {"right", "left"}:
                center = self.plan.root
            else:
                raise ValueError("TreePeps form must be None, 'right', 'left', or a site id")
        if center is None:
            center = self.orthogonality_center
            if center is None:
                center = self.plan.root
        center = self.plan.resolve_site(center)
        order = normalize_tree_compression_order(order)
        compression_mode = _normalize_compression_mode(compression_mode)
        if compression_mode in {"sdc", "src", "zipup"} and self.plan.is_mps_topology:
            self._compress_path_region_1d(
                self.sites,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                compression_seed=compression_seed,
            )
            self._canonical_region = None
            self.canonize_to(center, inplace=True, info_c=info_c, _force_full=True)
            return self
        self.shift_orthogonality_center(
            center,
            info_c=info_c,
            _skip_validate=True,
        )

        edge_order = iter_tree_compression_order(
            self.plan,
            center=center,
            nodes=self.sites,
            order=order,
            tensor_getter=self.node_tensor,
            bond_getter=self.bond,
        )
        for q, toward in edge_order:
            self._compress_edge_inplace(
                q,
                toward,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb="right",
                reduced=reduced,
                compression_mode=compression_mode,
                compression_seed=compression_seed,
                _validate=False,
            )
        # The inward sweep has already established the defining isometries.
        # Record the final center directly instead of running a second full
        # QR sweep over the tree.
        self._canonical_region = frozenset({center})
        self._set_isometry_metadata_from_region({center})
        self.validate(check_canonical=True)
        self._sync_info_c(info_c)
        return self

    def compress_subtree(
        self,
        sites,
        *,
        span=False,
        center=None,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        reduced=True,
        compression_mode="direct",
        compression_seed=None,
        order="rank",
        inplace=False,
        info_c=None,
    ):
        """Compress only the connected region spanned by ``sites``.

        Tensors outside the region are first made isometric towards it, using
        the existing ``left_inds`` proofs whenever possible.  Internal region
        edges are then compressed once in a leaf-to-center sweep.  Exterior
        branches and their boundary bonds are not touched.
        """

        if isinstance(sites, Integral):
            sites = (sites,)
        sites = tuple(self.plan.resolve_site(site) for site in sites)
        region = self.plan.subtree_span(sites) if span else frozenset(sites)
        if not region or not self.plan.is_connected(region):
            raise ValueError("sites must form a connected subtree, or pass span=True")

        work = self if inplace else self.copy()
        if center is None:
            center = min(
                region,
                key=lambda q: (
                    max(len(work.plan.path(q, other)) for other in region),
                    sum(len(work.plan.path(q, other)) for other in region),
                    q,
                ),
            )
        center = work.plan.resolve_site(center)
        if center not in region:
            raise ValueError("center must lie inside the compressed subtree")
        order = normalize_tree_compression_order(order)

        compression_mode = _normalize_compression_mode(compression_mode)
        if compression_mode in {"sdc", "src", "zipup"} and work.plan.is_mps_topology:
            work._canonicalize_region_fast(region)
            work._compress_path_region_1d(
                region,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                compression_mode=compression_mode,
                compression_seed=compression_seed,
            )
            work._canonical_region = None
            work.canonize_to(
                center,
                inplace=True,
                info_c=info_c,
                _force_full=True,
            )
            return work
        if work.canonical_region != frozenset(region) or not work.is_subtree_canonical_form(region):
            work._canonicalize_region_fast(region)
            work._canonical_region = frozenset(region)
            work._set_isometry_metadata_from_region(region)

        if len(region) > 1:
            # Recover a single hub first, then process each branch as a
            # complete canonical sweep.  Compressing all edges in a flat
            # leaf-to-hub ordering is not generally optimal: after the first
            # edge of one branch is truncated, the neighboring tensor is no
            # longer the canonical boundary needed by the next edge.  The
            # recursive descent below mirrors TreeOptimizer's subtree sweep:
            # compress the parent-child edge, finish the child branch, then
            # QR-canonize that branch back into the parent before moving to
            # the next sibling.
            work._recover_center_from_region(region, center)

            def edge_cutoff(node, child):
                """Avoid re-cutting a bond that is already within ``max_bond``."""

                if max_bond is not None and work.ind_size(
                    work.bond(node, child)
                ) <= max_bond:
                    return 0.0
                return cutoff

            def descend(node, parent):
                pending = {
                    neighbor
                    for neighbor in work.plan.neighbors(node)
                    if neighbor in region and neighbor != parent
                }
                while pending:
                    if order == "rank":
                        # Re-score after each completed branch: its
                        # compression can reduce a bond on ``node`` and
                        # change the cost of the remaining siblings.
                        child = min(
                            pending,
                            key=lambda candidate: (
                                *tree_edge_rank_key(
                                    work.node_tensor(node),
                                    work.node_tensor(candidate),
                                    work.bond(node, candidate),
                                ),
                                int(candidate),
                            ),
                        )
                    else:
                        child = min(pending)
                    pending.remove(child)
                    work._compress_edge_inplace(
                        node,
                        child,
                        max_bond=max_bond,
                        cutoff=edge_cutoff(node, child),
                        cutoff_mode=cutoff_mode,
                        absorb="right",
                        reduced=reduced,
                        compression_mode=compression_mode,
                        compression_seed=compression_seed,
                        _validate=False,
                    )
                    descend(child, node)
                    work.canonize_edge_(child, node, absorb="right")

            descend(center, None)

        work._canonical_region = frozenset({center})
        work._set_isometry_metadata_from_region({center})
        work.validate(check_canonical=True)
        work._sync_info_c(info_c)
        return work

    def compress_subtree_(
        self,
        sites,
        *,
        span=False,
        center=None,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        reduced=True,
        compression_mode="direct",
        compression_seed=None,
        order="rank",
        info_c=None,
    ):
        """In-place alias for :meth:`compress_subtree`."""

        return self.compress_subtree(
            sites,
            span=span,
            center=center,
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            reduced=reduced,
            compression_mode=compression_mode,
            compression_seed=compression_seed,
            order=order,
            inplace=True,
            info_c=info_c,
        )

    def _bond_dim(self, site0, site1):
        """Return a live tree-bond dimension for display."""

        try:
            return int(self.ind_size(self.bond(site0, site1)))
        except ValueError:
            return 1

    def ascii_lattice(self, *, bond_dims=True, node_ids=False):
        """Return a PEPS-style coordinate drawing of the retained tree bonds."""

        def label(q):
            return f"N{q}" if node_ids else "●"

        def draw_layer(z=None):
            coords = (
                (x, y) if z is None else (x, y, z)
                for x in range(self.shape[0])
                for y in range(self.shape[1])
            )
            qs = {coordinate: self.plan.logical_site(coordinate) for coordinate in coords}
            width = max(len(label(q)) for q in qs.values())
            horizontal_width = max(7, width + 4)
            lines = []
            for x in range(self.shape[0]):
                row = []
                for y in range(self.shape[1]):
                    q = qs[(x, y) if z is None else (x, y, z)]
                    row.append(label(q).center(width))
                    if y + 1 < self.shape[1]:
                        q1 = qs[(x, y + 1) if z is None else (x, y + 1, z)]
                        segment = " " * horizontal_width
                        if q1 in self.plan.neighbors(q):
                            edge = self._bond_dim(q, q1) if bond_dims else ""
                            segment = f"--{edge}--".center(horizontal_width, "─")
                        row.append(segment)
                lines.append("".join(row).rstrip())

                if x + 1 < self.shape[0]:
                    row = []
                    for y in range(self.shape[1]):
                        q = qs[(x, y) if z is None else (x, y, z)]
                        q1 = qs[(x + 1, y) if z is None else (x + 1, y, z)]
                        segment = " " * width
                        if q1 in self.plan.neighbors(q):
                            edge = self._bond_dim(q, q1) if bond_dims else ""
                            segment = f"│{edge}│".center(width, "│")
                        row.append(segment)
                        if y + 1 < self.shape[1]:
                            row.append(" " * horizontal_width)
                    lines.append("".join(row).rstrip())
            return lines

        if self.ndim == 2:
            return "\n".join(draw_layer())

        lines = []
        for z in range(self.shape[2]):
            if lines:
                lines.append("")
            lines.append(f"z={z}")
            lines.extend(draw_layer(z))
            if z + 1 < self.shape[2]:
                z_edges = []
                for x in range(self.shape[0]):
                    for y in range(self.shape[1]):
                        q = self.plan.logical_site((x, y, z))
                        q1 = self.plan.logical_site((x, y, z + 1))
                        if q1 in self.plan.neighbors(q):
                            dim = self._bond_dim(q, q1) if bond_dims else ""
                            z_edges.append(f"({x},{y},{z})│{dim}│({x},{y},{z + 1})")
                if z_edges:
                    lines.append("z-bonds: " + ", ".join(z_edges))
        return "\n".join(lines)

    def _quimb_ascii_2d(
        self,
        *,
        bond_dims=True,
        node_ids=False,
        show_lower=False,
        show_upper=False,
    ):
        """Render a 2D tree using Quimb ``PEPS.show`` spacing conventions."""

        def label(q):
            return f"N{q}" if node_ids else "●"

        node_width = max(3, max(len(label(q)) for q in self.sites))
        connector_width = max(4, node_width + 1)

        def horizontal_dimension(row):
            pieces = []
            for y in range(self.shape[1]):
                q = self.plan.logical_site((row, y))
                pieces.append(" " * node_width)
                if y + 1 < self.shape[1]:
                    q1 = self.plan.logical_site((row, y + 1))
                    retained = q1 in self.plan.neighbors(q)
                    dimension = self._bond_dim(q, q1) if retained and bond_dims else ""
                    prefix = "╱" if show_upper and retained else " "
                    pieces.append(
                        (prefix + str(dimension)).center(connector_width)
                        if retained
                        else " " * connector_width
                    )
            return "".join(pieces).rstrip()

        def node_row(row):
            pieces = []
            for y in range(self.shape[1]):
                q = self.plan.logical_site((row, y))
                pieces.append(label(q).center(node_width))
                if y + 1 < self.shape[1]:
                    q1 = self.plan.logical_site((row, y + 1))
                    retained = q1 in self.plan.neighbors(q)
                    pieces.append(
                        "━" * connector_width
                        if retained
                        else " " * connector_width
                    )
            return "".join(pieces).rstrip()

        def vertical_row(row):
            pieces = []
            for y in range(self.shape[1]):
                q = self.plan.logical_site((row, y))
                q1 = self.plan.logical_site((row + 1, y))
                retained = q1 in self.plan.neighbors(q)
                dimension = self._bond_dim(q, q1) if retained and bond_dims else ""
                text = ("╱" if show_lower and retained else " ")
                text += "┃" if retained else " "
                text += str(dimension) if retained else ""
                pieces.append(text.center(node_width))
                if y + 1 < self.shape[1]:
                    pieces.append(" " * connector_width)
            return "".join(pieces).rstrip()

        lines = [horizontal_dimension(0)]
        for row in range(self.shape[0]):
            lines.append(node_row(row))
            if row + 1 < self.shape[0]:
                lines.append(vertical_row(row))
                lines.append(horizontal_dimension(row + 1))
        return "\n".join(lines)

    def show(
        self,
        *,
        bond_dims=True,
        node_ids=False,
        color=False,
        show_lower=False,
        show_upper=False,
    ):
        """Print a PEPS-style coordinate schematic of this tree state.

        ``show_lower`` and ``show_upper`` are accepted for compatibility with
        :meth:`quimb.tensor.PEPS.show`; for 2D trees they add the same diagonal
        visual markers, while the tree-specific drawing omits non-retained
        lattice edges.
        """

        del color
        if self.ndim == 2:
            drawing = self._quimb_ascii_2d(
                bond_dims=bond_dims,
                node_ids=node_ids,
                show_lower=show_lower,
                show_upper=show_upper,
            )
        else:
            drawing = self.ascii_lattice(bond_dims=bond_dims, node_ids=node_ids)
        print(drawing)

    def norm(self, output_inds=None, squared=False, strip_exponent=False, **contract_opts):
        """Return the exact Frobenius norm using Quimb's vector semantics.

        When a one-site canonical centre is known, the centre tensor contains
        the complete represented norm because every other tree tensor is an
        isometry toward it.  Use that contraction for the optimizer's hot
        diagnostic path, while retaining Quimb's full contraction for custom
        output-index, exponent, or contraction-option requests.
        """

        if (
            output_inds is None
            and not strip_exponent
            and not contract_opts
            and getattr(self, "exponent", 0.0) == 0.0
            and self.orthogonality_center is not None
        ):
            center = self.node_tensor(self.orthogonality_center)
            value = qtn.tensor_contract(center.H, center, output_inds=[])
            value = float(abs(np.asarray(ar.to_numpy(value))))
            return value if squared else value**0.5

        return super().norm(
            output_inds=output_inds,
            squared=squared,
            strip_exponent=strip_exponent,
            **contract_opts,
        )

    def to_dense(self, *inds_seq, **contract_opts):
        """Contract to a dense tensor in logical one-dimensional site order."""

        if inds_seq:
            output_inds = tuple(inds_seq)
        else:
            output_inds = tuple(self.site_ind_1d(q) for q in self.sites)
        return self.contract(all, output_inds=output_inds, **contract_opts)

    def to_statevector(self, order=None):
        """Return a host NumPy statevector in the requested site order."""

        if order is None:
            order = self.sites
        order = tuple(self.plan.resolve_site(site) for site in order)
        if len(order) != self.plan.size or set(order) != set(self.sites):
            raise ValueError("order must contain every TreePeps site exactly once")
        dense = self.to_dense(*(self.site_ind_1d(site) for site in order))
        return ar.to_numpy(dense.data).reshape(-1)

    def local_expectation(
        self,
        operator,
        where,
        *,
        normalized=True,
        max_bond=None,
        optimize=None,
        **contract_opts,
    ):
        """Evaluate an exact one- or multi-site observable."""

        # These names are accepted for parity with TreeTensorNetwork and
        # MPS readout. TreePeps uses an exact tree contraction here, so there
        # is no approximate max-bond path to select.
        del max_bond, optimize

        if isinstance(where, Integral):
            where = (int(where),)
        else:
            where = tuple(where)
        sites = tuple(self.plan.resolve_site(site) for site in where)
        if not sites or len(set(sites)) != len(sites):
            raise ValueError("where must contain distinct TreePeps sites")
        physical = [self.site_ind_1d(q) for q in sites]
        dims = [
            self.node_tensor(q).shape[self.node_tensor(q).inds.index(ind)]
            for q, ind in zip(sites, physical)
        ]
        operator = _reshape_operator(operator, dims)
        gate = qtn.Tensor(
            operator,
            inds=tuple(ind + "*" for ind in physical) + tuple(physical),
        )
        bra = self.H.reindex({ind: ind + "*" for ind in physical})
        numerator = (bra & gate & self).contract(all, output_inds=[], **contract_opts)
        if not normalized:
            return numerator
        denominator = (self.H & self).contract(all, output_inds=[], **contract_opts)
        return numerator / denominator

    def local_expectations(
        self,
        terms,
        *,
        normalized=True,
        max_bond=None,
        optimize=None,
        **contract_opts,
    ):
        """Evaluate a mapping of local observables in iteration order.

        This mirrors ``TreeTensorNetwork.local_expectations`` while retaining
        the TreePeps coordinate/logical site selectors accepted by
        :meth:`local_expectation`.
        """

        if not hasattr(terms, "items"):
            raise TypeError("terms must be a mapping of support to operators")
        return {
            where: self.local_expectation(
                operator,
                where,
                normalized=normalized,
                max_bond=max_bond,
                optimize=optimize,
                **contract_opts,
            )
            for where, operator in terms.items()
        }

    def normalize(self, eps=1e-15, insert=None):
        """Normalize the state in place and return its previous norm.

        ``insert`` is accepted for MPS/TTN API compatibility. TreePeps has no
        distinguished insertion site, so normalization is applied to the
        current canonical center or to the root after one lossless
        canonicalization.
        """

        del insert
        eps = float(eps)
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        old_norm = self.norm()
        magnitude = float(abs(np.asarray(ar.to_numpy(old_norm))))
        if magnitude <= eps or not np.isfinite(magnitude):
            return old_norm
        center = self.orthogonality_center
        if center is None:
            self.canonize_to(self.plan.root, inplace=True)
            center = self.plan.root
        tensor = self.node_tensor(center)
        tensor.modify(data=tensor.data / magnitude)
        return old_norm

    @staticmethod
    def _site_ind_for_plan(plan: TreePepsPlan, q: int) -> str:
        return _default_format("k", plan.ndim).format(*plan.coordinate(q))

    @staticmethod
    def _bond_ind_for_plan(plan: TreePepsPlan, q0: int, q1: int) -> str:
        return f"_tpb{min(q0, q1)}_{max(q0, q1)}"

    @staticmethod
    def _tags_for_plan(plan: TreePepsPlan, q: int) -> tuple[str, ...]:
        coordinate = plan.coordinate(q)
        tags = [
            _default_format("I", plan.ndim).format(*coordinate),
            f"X{coordinate[0]}",
            f"Y{coordinate[1]}",
        ]
        if plan.ndim == 3:
            tags.append(f"Z{coordinate[2]}")
        tags.extend((f"I{q}", f"N{q}"))
        return tuple(tags)

    def __repr__(self) -> str:
        return (
            f"TreePeps(shape={self.shape!r}, sites={self.plan.size}, "
            f"tree_edges={len(self.plan.tree_edges)}, "
            f"topology={self.topology!r})"
        )


def _default_format(prefix: str, ndim: int) -> str:
    return prefix + ",".join("{}" for _ in range(ndim))


def _site_dimensions(plan: TreePepsPlan, dimensions, *, name: str) -> dict[int, int]:
    if isinstance(dimensions, Integral):
        dimensions = int(dimensions)
        if dimensions < 1:
            raise ValueError(f"{name} must be positive")
        return {q: dimensions for q in range(plan.size)}
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


def _reshape_operator(operator, dims):
    expected = tuple(dims) + tuple(dims)
    shape = tuple(getattr(operator, "shape", np.shape(operator)))
    if shape == expected:
        return operator
    matrix_shape = (int(np.prod(dims)),) * 2
    if shape != matrix_shape:
        raise ValueError(f"operator shape {shape} is incompatible with site dimensions {dims}")
    return np.reshape(operator, expected)
