"""Exact low-cost structural compression for dense tensor-network operators.

The generic MPO construction in :mod:`pepsy.operators.mpo_automaton` is
already a finite-state construction.  Its states can nevertheless carry
parallel, or more generally linearly dependent, boundary vectors after
different terms have been summed.  This module removes those dependencies
before a numerical SVD is attempted.

The implementation is deliberately private and conservative:

* only NumPy-backed tensors are touched;
* proportional columns are detected exactly;
* the optional linear-dependence pass only accepts a reconstruction whose
  residual is at floating-point roundoff;
* tensor-network indices are replaced in place, so the operation preserves
  the represented operator without introducing a public compression API.

This is the dense/operator-network part of the deparallelization and
delinearization ideas from arXiv:1611.02498.  Symmray, Torch, and other
backends continue through their existing metadata-preserving SVD paths.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import quimb.tensor as qtn


def _is_dense_numpy(data):
    """Whether ``data`` is safe to mutate with NumPy-only operations."""

    return isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number)


def _parallel_factor(matrix):
    """Factor exact proportional columns as ``matrix = basis @ transfer``.

    The representative column for each proportionality class is retained as
    the basis.  Ratios are checked by exact equality after they are obtained
    from the first non-zero entry, which avoids silently turning a close but
    genuinely independent pair into the same state.
    """

    _, n_cols = matrix.shape
    transfer_dtype = (
        matrix.dtype
        if np.issubdtype(matrix.dtype, np.inexact)
        else np.result_type(matrix.dtype, np.float64)
    )
    if n_cols <= 1:
        return matrix, np.eye(n_cols, dtype=transfer_dtype), False

    representatives = []
    zero_columns = []
    transfer = np.zeros((n_cols, n_cols), dtype=transfer_dtype)
    for column_index in range(n_cols):
        column = matrix[:, column_index]
        nonzero = np.flatnonzero(column != 0)
        if nonzero.size == 0:
            zero_columns.append(column_index)
            continue

        matched = False
        pivot = int(nonzero[0])
        for basis_index, representative_index in enumerate(representatives):
            representative = matrix[:, representative_index]
            representative_nonzero = np.flatnonzero(representative != 0)
            if representative_nonzero.size == 0:
                continue
            if pivot >= representative.size or representative[pivot] == 0:
                continue
            ratio = column[pivot] / representative[pivot]
            if np.array_equal(column, representative * ratio):
                transfer[basis_index, column_index] = ratio
                matched = True
                break

        if not matched:
            basis_index = len(representatives)
            representatives.append(column_index)
            transfer[basis_index, column_index] = 1

    if not representatives:
        # Keep one zero basis vector so the tensor bond remains valid.
        return (
            matrix[:, :1],
            np.zeros((1, n_cols), dtype=transfer_dtype),
            n_cols > 1,
        )
    # All-zero columns need no coefficient once a non-zero basis exists.
    # Their transfer entries are already zero; this explicit loop documents
    # that they are intentionally discarded rather than represented by a
    # separate zero state.
    for column_index in zero_columns:
        transfer[:, column_index] = 0
    basis = matrix[:, representatives]
    transfer = transfer[: len(representatives)]
    changed = len(representatives) < n_cols
    return basis, transfer, changed


def _linear_factor(matrix):
    """Find a roundoff-safe independent-column factorization, if available."""

    if matrix.shape[1] <= 1:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    if not np.issubdtype(matrix.dtype, np.inexact):
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False

    try:
        from scipy.linalg import qr as scipy_qr
    except ImportError:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False

    scale = float(np.max(np.abs(matrix), initial=0.0))
    if scale == 0.0:
        return matrix[:, :1], np.zeros((1, matrix.shape[1]), dtype=matrix.dtype), (
            matrix.shape[1] > 1
        )

    try:
        _q, r, piv = scipy_qr(matrix, mode="economic", pivoting=True)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        rank = 1
    else:
        eps = np.finfo(matrix.real.dtype).eps
        tolerance = 64.0 * eps * max(matrix.shape) * scale
        rank = int(np.count_nonzero(diagonal > tolerance))
        rank = max(1, min(rank, matrix.shape[1]))
    if rank >= matrix.shape[1]:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False

    basis = matrix[:, np.asarray(piv[:rank], dtype=int)]
    try:
        transfer, *_ = np.linalg.lstsq(basis, matrix, rcond=None)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    reconstructed = basis @ transfer
    eps = np.finfo(matrix.real.dtype).eps
    residual = float(np.max(np.abs(reconstructed - matrix), initial=0.0))
    allowed = 256.0 * eps * max(matrix.shape) * scale
    if not np.isfinite(residual) or residual > allowed:
        return matrix, np.eye(matrix.shape[1], dtype=matrix.dtype), False
    return basis, transfer, True


def _factor_columns(matrix, *, method="auto"):
    """Return a conservative low-rank column factorization."""

    basis, transfer, changed = _parallel_factor(matrix)
    if method in {"deparallelize", "parallel", "sparse"}:
        return basis, transfer, changed
    if method not in {"auto", "delinearize", "linear"}:
        raise ValueError(
            "structural compression method must be 'auto', "
            "'deparallelize', or 'delinearize'."
        )

    linear_basis, linear_transfer, linear_changed = _linear_factor(basis)
    if linear_changed:
        combined_transfer = linear_transfer @ transfer
        reconstructed = linear_basis @ combined_transfer
        scale = float(np.max(np.abs(matrix), initial=0.0))
        eps = np.finfo(matrix.real.dtype).eps
        allowed = 256.0 * eps * max(matrix.shape) * scale
        residual = float(np.max(np.abs(reconstructed - matrix), initial=0.0))
        # A well-conditioned local reduction should remain accurate after
        # the transfer is composed with any preceding parallel factor. This
        # guard is important when a tiny floating-point pivot would otherwise
        # amplify a harmless intermediate QR residual.
        if np.isfinite(residual) and residual <= allowed:
            return linear_basis, combined_transfer, True
    return basis, transfer, changed


def _replace_bond(tensor, old_bond, new_bond, data):
    """Replace one tensor bond while dropping stale canonical metadata."""

    inds = list(tensor.inds)
    inds[inds.index(old_bond)] = new_bond
    tensor.modify(data=data, inds=tuple(inds), left_inds=None)


def _transform_axis(data, matrix, axis):
    """Apply ``matrix`` to one tensor axis, preserving the axis position."""

    moved = np.moveaxis(data, axis, 0)
    transformed = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(transformed, 0, axis)


def _factor_edge_from_child(
    child_tensor,
    parent_tensor,
    bond,
    *,
    method,
    reduce_rows=False,
):
    """Reduce one edge and absorb its exact transfer into the parent tensor."""

    if not (
        _is_dense_numpy(child_tensor.data)
        and _is_dense_numpy(parent_tensor.data)
    ):
        return False, None

    child_axis = child_tensor.inds.index(bond)
    child_data = np.moveaxis(child_tensor.data, child_axis, -1)
    old_dim = child_data.shape[-1]
    matrix = child_data.reshape(-1, old_dim)
    if reduce_rows:
        basis, transfer, changed = _factor_columns(matrix.T, method=method)
        # ``matrix.T`` has one column per non-bond configuration, so its
        # column rank can be smaller without reducing the actual virtual
        # bond. Do not perform a pointless gauge change in that case.
        if not changed or basis.shape[1] >= old_dim:
            return False, None
        parent_transform = basis.T
        child_data = transfer.reshape(
            basis.shape[1], *child_data.shape[:-1]
        )
        child_data = np.moveaxis(child_data, 0, -1)
        child_data = np.moveaxis(child_data, -1, child_axis)
    else:
        basis, transfer, changed = _factor_columns(matrix, method=method)
        if not changed:
            return False, None
        parent_transform = transfer
        child_data = basis.reshape(*child_data.shape[:-1], basis.shape[1])
        child_data = np.moveaxis(child_data, -1, child_axis)

    parent_axis = parent_tensor.inds.index(bond)
    parent_data = _transform_axis(parent_tensor.data, parent_transform, parent_axis)
    new_bond = qtn.rand_uuid()
    _replace_bond(child_tensor, bond, new_bond, child_data)
    _replace_bond(parent_tensor, bond, new_bond, parent_data)
    return True, (old_dim, int(child_tensor.ind_size(new_bond)))


def _tree_depth_order(root, parent, children):
    """Return tree nodes grouped in deterministic depth order."""

    depths = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in children.get(node, ()):
            depths[child] = depths[node] + 1
            queue.append(child)
    return tuple(sorted(depths, key=lambda node: (depths[node], node)))


def _structural_compress_mpo(mpo, *, method="auto"):
    """Reduce exact dense MPO boundary dependencies in two short sweeps."""

    tensors = tuple(mpo)
    if not tensors or any(not _is_dense_numpy(tensor.data) for tensor in tensors):
        return {"changed": False, "reductions": (), "method": method}

    reductions = []
    for _direction in range(2):
        indices = range(len(tensors) - 1) if _direction == 0 else range(
            len(tensors) - 2, -1, -1
        )
        for index in indices:
            left = tensors[index]
            right = tensors[index + 1]
            bond = next(iter(qtn.bonds(left, right)))
            if _direction == 0:
                changed, detail = _factor_edge_from_child(
                    right,
                    left,
                    bond,
                    method=method,
                )
            else:
                changed, detail = _factor_edge_from_child(
                    left,
                    right,
                    bond,
                    method=method,
                    reduce_rows=True,
                )
            if changed:
                reductions.append((index, detail[0], detail[1], _direction))

    return {
        "changed": bool(reductions),
        "reductions": tuple(reductions),
        "method": method,
    }


def _structural_compress_tree(
    network,
    *,
    root,
    parent: Mapping,
    children: Mapping,
    nodes: Iterable,
    tensor_getter: Callable,
    bond_getter: Callable,
    method="auto",
):
    """Reduce exact dense dependencies on every edge of a rooted tree."""

    node_list = tuple(nodes)
    tensors = tuple(tensor_getter(node) for node in node_list)
    if not tensors or any(not _is_dense_numpy(tensor.data) for tensor in tensors):
        return {"changed": False, "reductions": (), "method": method}

    order = _tree_depth_order(root, parent, children)
    reductions = []
    # Leaf-to-root is the tree analogue of a deparallelization sweep: all
    # child boundary vectors are reduced before their parent is processed.
    for child in reversed(order):
        if child == root:
            continue
        ancestor = parent[child]
        bond = bond_getter(child, ancestor)
        changed, detail = _factor_edge_from_child(
            tensor_getter(child),
            tensor_getter(ancestor),
            bond,
            method=method,
        )
        if changed:
            reductions.append((child, detail[0], detail[1], "up"))

    # A second orientation catches dependencies that are visible on the
    # parent-facing rows after transfers have been accumulated upward.
    for child in order:
        if child == root:
            continue
        ancestor = parent[child]
        bond = bond_getter(child, ancestor)
        changed, detail = _factor_edge_from_child(
            tensor_getter(child),
            tensor_getter(ancestor),
            bond,
            method=method,
            reduce_rows=True,
        )
        if changed:
            reductions.append((child, detail[0], detail[1], "down"))

    # TreeMPO stores a live maximum-bond diagnostic on its underlying
    # network. Keep that cache synchronized even when the builder requested
    # no numerical SVD and therefore never enters TreeMPO.compress().
    metadata_network = network
    stored_networks = getattr(network, "tree_networks", None)
    if stored_networks:
        metadata_network = stored_networks[0]
    try:
        final_bond = max(
            (metadata_network.ind_size(index)
             for index in metadata_network.inner_inds()),
            default=1,
        )
    except AttributeError:
        final_bond = None
    if (
        final_bond is not None
        and hasattr(metadata_network, "pepsy_tree_operator_bond")
    ):
        metadata_network.pepsy_tree_operator_bond = final_bond

    return {
        "changed": bool(reductions),
        "reductions": tuple(reductions),
        "method": method,
        "final_max_bond": final_bond,
    }
