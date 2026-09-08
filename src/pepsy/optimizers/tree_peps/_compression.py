"""Geometry-aware compression schedules shared by TreePeps and TreePEPO."""

from __future__ import annotations


def normalize_tree_compression_order(order):
    """Normalize the native tree compression scheduling policy."""

    if order is None:
        return "rank"
    order = str(order).strip().lower().replace("-", "_")
    aliases = {
        "auto": "rank",
        "rank_aware": "rank",
        "tree": "depth",
        "deterministic": "depth",
    }
    order = aliases.get(order, order)
    if order not in {"rank", "depth"}:
        raise ValueError(
            "tree compression order must be 'rank' or 'depth', "
            f"got {order!r}."
        )
    return order


def tree_compression_order(
    plan,
    *,
    center,
    nodes,
    order="rank",
    tensor_getter,
    bond_getter,
):
    """Return a snapshot of a safe leaf-to-center edge schedule.

    For compression code that performs an SVD between choices, use
    :func:`iter_tree_compression_order` so ``order="rank"`` can inspect the
    dimensions after each reduction. This tuple helper remains useful for
    diagnostics and callers that need a fixed, non-mutating schedule.
    """

    return tuple(
        iter_tree_compression_order(
            plan,
            center=center,
            nodes=nodes,
            order=order,
            tensor_getter=tensor_getter,
            bond_getter=bond_getter,
        )
    )


def iter_tree_compression_order(
    plan,
    *,
    center,
    nodes,
    order="rank",
    tensor_getter,
    bond_getter,
):
    """Yield the next safe edge using the dimensions currently in memory.

    The caller must perform the yielded compression before requesting the
    next edge.  This is deliberately an iterator rather than a precomputed
    tuple: an SVD can reduce the shared bond and change the cost of every
    remaining edge incident on the same target tensor.
    """

    order = normalize_tree_compression_order(order)
    center = plan.resolve_site(center)
    nodes = frozenset(plan.resolve_site(node) for node in nodes)
    if not nodes or center not in nodes or not plan.is_connected(nodes):
        raise ValueError(
            "tree compression requires a connected node set containing center"
        )

    remaining = set(nodes)
    while len(remaining) > 1:
        leaves = [
            node
            for node in remaining
            if node != center
            and sum(neighbor in remaining for neighbor in plan.neighbors(node)) == 1
        ]
        if not leaves:
            raise ValueError("tree compression requires a connected node set")

        candidates = []
        for node in leaves:
            neighbor = next(
                neighbor
                for neighbor in plan.neighbors(node)
                if neighbor in remaining
            )
            if order == "rank":
                tensor = tensor_getter(node)
                target = tensor_getter(neighbor)
                bond = bond_getter(node, neighbor)
                key = tree_edge_rank_key(tensor, target, bond)
            else:
                # Farthest-first is the deterministic compatibility policy.
                key = (-len(plan.path(node, center)),)
            candidates.append((*key, int(node), node, neighbor))

        *_, node, neighbor = min(candidates)
        yield node, neighbor
        remaining.remove(node)


def tree_edge_rank_key(tensor, target, bond):
    """Return the live local rank score for one candidate tree edge."""

    left_dim = _external_dim(tensor, bond)
    right_dim = _external_dim(target, bond)
    rank_bound = min(left_dim, right_dim)
    bond_dim = int(tensor.ind_size(bond))
    return rank_bound, bond_dim, left_dim * right_dim


def _external_dim(tensor, bond):
    """Return the product of all live tensor dimensions except ``bond``."""

    dimension = 1
    for index in tensor.inds:
        if index != bond:
            dimension *= int(tensor.ind_size(index))
    return max(1, dimension)
