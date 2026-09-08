"""Explicit changes of state geometry, without a dense statevector handoff.

Both conversion paths construct isometries from the physical leaves toward
the root. The exact path passes untruncated QR remainders up the tree. The
capped path projects the original network into a succession of smaller
subspaces, retaining the residual network until the root is formed.

In particular, an MPS virtual bond and a TTN virtual bond need not describe
the same bipartition. ``chi`` therefore bounds the *output* tree bonds;
comparing it with the source MPS bond dimension does not establish exactness.
"""

from math import prod
from numbers import Integral

import autoray as ar
import quimb.tensor as qtn

from ..backends import infer_backend_signature

__all__ = ["mps_to_ttn"]


def _check_size(size, limit, operation):
    if limit is not None and size > limit:
        raise MemoryError(
            f"mps_to_ttn {operation} needs {size} tensor elements, exceeding "
            f"max_intermediate_elements={limit}. Choose a different tree or "
            "explicitly increase the limit; conversion has not changed chi."
        )


def _contract(network, output_inds, optimize, limit):
    """Check the actual contraction path before allocating its intermediates."""
    _check_size(prod(network.ind_size(ix) for ix in output_inds), limit, "output")
    path = network.contraction_tree(output_inds=output_inds, optimize=optimize)
    _check_size(path.max_size(), limit, "contraction")
    return network.contract(all, output_inds=output_inds, optimize=path, preserve_tensor=True)


def _postorder(plan):
    """Iterative traversal also handles long, unbalanced trees."""
    if plan.root not in plan.children or plan.parent.get(plan.root) is not None:
        raise ValueError("tree must have a known root with no parent.")
    stack = [(plan.root, False)]
    visited = set()
    while stack:
        node, ready = stack.pop()
        if ready:
            yield node
            continue
        if node in visited:
            raise ValueError("tree must be a rooted tree without repeated nodes.")
        visited.add(node)
        stack.append((node, True))
        for child in reversed(plan.children[node]):
            if child not in plan.children or plan.parent.get(child) != node:
                raise ValueError("tree has inconsistent parent/child links.")
            stack.append((child, False))
    if visited != set(plan.nodes()):
        raise ValueError("tree has disconnected nodes.")


def _projector(remainder, left_inds, chi, sample, optimize, limit):
    """Leading Schmidt subspace of the *current* residual state.

    Contract the environment as well as the local tensors. A local SVD on
    arbitrary MPS virtual legs would depend on the input gauge. Earlier
    projections remain explicit in this network, so no exact TTN is formed
    before imposing the requested cap. If the current coefficients are
    Psi[x, e], where x combines ``left_inds`` and e labels the environment,
    the desired subspace is given by the leading eigenvectors of
    rho[x, x'] = sum_e Psi[x, e] conj(Psi[x', e]).
    """
    shape = tuple(remainder.ind_size(ix) for ix in left_inds)
    dimension = prod(shape)
    _check_size(dimension * dimension, limit, "local density matrix / identity")
    if dimension <= chi:
        # No compression is necessary. Namespace creation inherits the
        # sample's dtype AND device, including a nondefault CUDA device.
        basis = ar.get_namespace(like=sample).eye(dimension)
    else:
        # The eigenvectors are independent of positive global scale. Scale
        # private factors by their largest entry before doubling the network
        # to avoid squaring a very large/small input amplitude. The live
        # residual and the returned state's scale remain untouched.
        scaled = remainder.copy()
        for tensor in scaled:
            scale = ar.do("max", ar.do("abs", tensor.data))
            safe = ar.do("where", scale > 0, scale, ar.do("ones_like", scale))
            tensor.modify(data=tensor.data / safe)
        bra_inds = tuple(qtn.rand_uuid() for _ in left_inds)
        # Bra-internal bonds must be independent of ket-internal bonds.
        # The selected x legs stay open as distinct x' legs; all remaining
        # outer legs deliberately keep their names to trace the environment.
        rename = {ix: qtn.rand_uuid() for ix in scaled.inner_inds()}
        rename.update(zip(left_inds, bra_inds))
        bra = scaled.conj().reindex(rename)
        density = _contract(scaled & bra, (*left_inds, *bra_inds), optimize, limit)
        matrix = ar.do("reshape", density.data, (dimension, dimension))
        # Remove the roundoff-level anti-Hermitian part before diagonalizing.
        matrix = (matrix + ar.do("conj", ar.do("transpose", matrix))) * 0.5
        _, vectors = ar.do("linalg.eigh", matrix)
        # eigh orders eigenvalues increasingly. Keep orthonormal columns U
        # only: their eigenvalues are NOT absorbed into these tree tensors.
        # The residual network below carries the state amplitudes instead.
        basis = vectors[:, -chi:]
    return ar.do("reshape", basis, (*shape, min(dimension, chi)))


def mps_to_ttn(
    mps,
    *,
    tree=None,
    chi=None,
    optimize="greedy",
    max_intermediate_elements=2**26,
    node_tag_id="N{}",
):
    """Rebuild a dense-array MPS on a chosen :class:`TreePlan`.

    ``chi=None`` is lossless (up to floating-point roundoff): bottom-up QR
    factorizations retain all directions without a singular-value cutoff.
    With a positive ``chi``, sequential reduced-density-matrix projections
    cap every output tree bond. This is an approximation unless all required
    tree Schmidt ranks fit; it is not a globally optimal variational fit.

    Parameters
    ----------
    mps : quimb.tensor.MatrixProductState
        One tensor per site, with site labels ``0 .. L-1``. Dense NumPy,
        Torch, CuPy and JAX arrays are supported through Quimb/Autoray.
        Symmray/fermionic tensors require a separate graded conversion and
        are explicitly rejected. The input is never modified.
    tree : TreePlan, optional
        Geometry with the same physical site labels. May have arbitrary
        leaf order, arity, and an optional physical root site. Omitted means
        a balanced tree in MPS site order.
    chi : positive int or None
        Output TTN bond cap, independent of the MPS bond dimension. ``None``
        never truncates, even when the required TTN bonds grow substantially.
        No automatic normalization or magnitude cutoff is applied.
    optimize : contraction optimizer, optional
        Quimb/Cotengra path optimizer for exact intermediate contractions.
    max_intermediate_elements : positive int or None
        Guard on each planned contraction intermediate and local density
        matrix, default ``2**26`` elements. This is not a total-memory or
        eigensolver-workspace limit. ``None`` disables the guard. Exceeding
        it raises ``MemoryError``; it never enables truncation implicitly.
    node_tag_id : str
        Structural tree node tag format. Physical index/site tag formats and
        source tensor tags are preserved.

    Returns
    -------
    TreeTensorNetwork
        State canonical toward the root, on the source backend/device/dtype.
        The input's extracted exponent and global phase are preserved;
        finite-chi projection can reduce its norm. No full statevector is
        constructed, but unfavorable partitions can still be expensive.
    """
    from ..optimizers.tree import TreePlan, TreeTensorNetwork

    if not isinstance(mps, qtn.MatrixProductState):
        raise TypeError("mps must be a quimb MatrixProductState.")
    for name, value in (("chi", chi), ("max_intermediate_elements", max_intermediate_elements)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None.")
    if chi is not None and not callable(getattr(ar, "get_namespace", None)):
        raise RuntimeError(
            "Finite-chi mps_to_ttn requires Autoray's get_namespace(like=...) "
            "API for dtype/device-preserving array creation."
        )
    sites = tuple(mps.sites)
    if not sites or set(sites) != set(range(mps.L)) or mps.num_tensors != mps.L:
        raise ValueError("mps must have one tensor per site with labels 0..L-1.")
    signatures = {infer_backend_signature(t.data) for t in mps.tensors}
    if any(sig[0] == "symmray" for sig in signatures) or getattr(mps, "fermionic", False):
        raise TypeError("mps_to_ttn does not yet support Symmray/fermionic MPS.")
    if len(signatures) != 1:
        raise TypeError("MPS tensors must share one backend, dtype, and device.")
    dtype = next(iter(signatures))[1]
    if not any(kind in dtype for kind in ("float", "complex")):
        raise TypeError("MPS arrays must have floating-point or complex dtype.")
    if tree is None:
        tree = TreePlan.from_order(sites, structure="balanced")
    if not isinstance(tree, TreePlan):
        raise TypeError("tree must be a TreePlan.")
    if set(tree.node_of_qubit) != set(sites):
        raise ValueError("tree and mps must have the same physical site labels.")
    order = tuple(_postorder(tree))
    physical = {q: mps.site_ind(q) for q in sites}
    if set(mps.outer_inds()) != set(physical.values()):
        raise ValueError("MPS outer indices must be exactly its physical indices.")
    node_tags = {node_tag_id.format(node) for node in order}
    if len(node_tags) != len(order) or node_tags.intersection(mps.tags):
        raise ValueError("node_tag_id must produce unique tags absent from the MPS.")
    # Private copies isolate metadata and contractions from the source state.
    sources = {q: mps[q].copy(deep=True) for q in sites}
    bonds = {node: qtn.rand_uuid() for node in order if node != tree.root}
    outputs = []
    messages = {}
    frontiers = {}
    remainder = qtn.TensorNetwork(tuple(sources.values())) if chi is not None else None
    sample = next(iter(sources.values())).data

    for node in order:
        children = tree.children[node]
        q = tree.qubit_of_node.get(node)
        left = tuple(bonds[child] for child in children)
        tags = [node_tag_id.format(node)]
        if q is not None:
            left += (physical[q],)
            tags.extend(sources[q].tags)
        # Every original virtual bond crossing this partition contributes
        # its dimension to an exact upper bound on the tree Schmidt rank.
        # Earlier projections act wholly on one side of this cut and cannot
        # increase that rank. Avoid padding a small-rank MPS up to chi.
        frontier = set()
        for child in children:
            frontier.symmetric_difference_update(frontiers.pop(child))
        if q is not None:
            frontier.symmetric_difference_update(ix for ix in sources[q].inds if ix != physical[q])
        frontiers[node] = frontier

        if chi is None:
            # Contract only the messages within this subtree. ``left`` is
            # the physical/child-tree side; ``boundary`` comprises original
            # MPS bonds still connecting this subtree to its complement.
            local = [messages.pop(child) for child in children]
            if q is not None:
                local.append(sources[q])
            network = qtn.TensorNetwork(local)
            boundary = tuple(ix for ix in network.outer_inds() if ix not in left)
            tensor = _contract(network, (*left, *boundary), optimize, max_intermediate_elements)
            if node != tree.root:
                # M[left, boundary] = Q[left, new_bond] R[new_bond, boundary].
                # Reduced QR removes only a structural dimension excess,
                # never directions selected by a numerical rank tolerance.
                # Q is an inward isometry; R retains scale and phase.
                tensor, messages[node] = tensor.split(
                    left,
                    right_inds=boundary,
                    method="qr",
                    stabilized=False,
                    get="tensors",
                    bond_ind=bonds[node],
                )
        elif node == tree.root:
            # All selected subspaces have been installed. These are the
            # remaining coefficients in their product basis, with the
            # original working scale and phase, not a normalized root.
            tensor = _contract(remainder, left, optimize, max_intermediate_elements)
        else:
            rank_cap = min(int(chi), prod(mps.ind_size(ix) for ix in frontier))
            data = _projector(
                remainder, left, rank_cap, sample, optimize, max_intermediate_elements
            )
            tensor = qtn.Tensor(data, inds=(*left, bonds[node]), left_inds=left)
            # Store U as an output isometry and replace x in the residual
            # by U^dagger Psi. This inserts the orthogonal projection U U^dagger
            # into the eventual state. H conjugates the entries; named-index
            # contraction supplies the matrix-transpose part of the adjoint.
            remainder.add_tensor(tensor.H)

        tensor.modify(tags=tags)
        outputs.append(tensor)

    result = TreeTensorNetwork(
        outputs,
        plan=tree,
        site_ind_id=mps.site_ind_id,
        site_tag_id=mps.site_tag_id,
        node_tag_id=node_tag_id,
    )
    # The source tensors omit Quimb's extracted factor 10**exponent. Carry
    # it once on the completed network; including it in each subtree or in
    # the density matrix would respectively double-count it or risk overflow.
    result.exponent = mps.exponent
    # Every nonroot tensor was constructed as an inward isometry. Recording
    # the center needs no additional (potentially expensive) canonical sweep.
    result.orthogonality_center = tree.root
    return result.validate()
