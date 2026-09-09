"""Successive compression of layered, acyclic tensor networks.

SRC contracts product-noise sketches of the complementary branches. SDC
contracts deterministic low-rank factors of those branches instead. A second
pass constructs nested output isometries and projects the *original* target
onto them. Neither algorithm truncates an already materialized target tree.
"""

import math
import warnings
from functools import lru_cache
from types import MappingProxyType

import autoray as ar
import quimb.tensor as qtn

from ..._internal.random import backend_random_array


@lru_cache(maxsize=128)
def _successive_environment_plan(order, hub):
    """Cache immutable geometry only, never tensors, dimensions, or RNG state.

    Order and hub jointly specify both the active tree and sweep direction.
    Numerical messages and their mutable last-use counters belong to one
    compression invocation; sharing a plan cannot share either of those.
    """
    nodes = {hub, *(u for edge in order for u in edge)}
    remaining = set(nodes)
    neighbors = {u: [] for u in nodes}
    for u, v in order:
        if u == v or u not in remaining or v not in remaining:
            raise ValueError("order must peel each tree node once toward the hub")
        neighbors[u].append(v)
        neighbors[v].append(u)
        remaining.remove(u)
    if remaining != {hub}:
        raise ValueError("order must describe one connected tree ending at the hub")
    required = set()
    todo = [(v, u) for u, v in order]
    while todo:
        u, v = todo.pop()
        if (u, v) in required:
            continue
        required.add((u, v))
        todo.extend((w, u) for w in neighbors[u] if w != v)
    schedule = [edge for edge in order if edge in required]
    schedule.extend((v, u) for u, v in reversed(order) if (v, u) in required)
    uses = {edge: 0 for edge in required}
    for u, v in schedule:
        for w in neighbors[u]:
            if w != v:
                uses[w, u] += 1
    for u, v in order:
        uses[v, u] += 1
    return (
        MappingProxyType({u: tuple(vs) for u, vs in neighbors.items()}),
        tuple(schedule),
        MappingProxyType(uses),
    )


def successive_tree_compress(local, order, hub, *, method, max_bond,
                             cutoff=0.0, cutoff_mode="rsum2", seed=None):
    """Return one tensor per node and edge records, without mutating inputs.

    ``local`` groups target layers by node; ``order`` peels children toward
    ``hub``. Exterior state bonds count as local output indices, so callers
    must first make the exterior isometric toward the active region.
    On a path these are the SRC/SDC low-rank-environment sweeps. Branching
    uses a directed environment for each cut and nested child projections.
    """
    if method not in {"src", "sdc"}:
        raise ValueError("successive compression requires 'src' or 'sdc'")
    if any(ar.infer_backend(t.data) not in {"numpy", "torch", "jax", "cupy"}
           for ts in local.values() for t in ts):
        raise NotImplementedError(
            f"tree {method} environments support dense tree tensors only; "
            "use direct or zipup for native symmetry tensors"
        )
    if method == "src" and cutoff != 0.0:
        warnings.warn("cutoff is ignored for tree SRC; use max_bond instead",
                      UserWarning, stacklevel=2)
    order = tuple(order)
    neighbors, schedule, consumers = _successive_environment_plan(order, hub)
    if neighbors.keys() != local.keys():
        raise ValueError("local tensors must exactly cover the planned tree nodes")
    indices = {u: {ix for t in ts for ix in t.inds} for u, ts in local.items()}
    sizes = {ix: t.ind_size(ix) for ts in local.values() for t in ts for ix in t.inds}
    bonds = {}
    for u, v in order:
        bonds[u, v] = bonds[v, u] = tuple(sorted(indices[u] & indices[v]))
    # Every invocation owns fresh counters and numerical caches, including
    # repeated targets and failed retries. Array edits, new ranks, changed
    # seeds/backends, and gates therefore cannot reuse stale environments.
    uses = dict(consumers)
    outer = {}
    for u, ts in local.items():
        inner = {ix for v in neighbors[u] for ix in bonds[u, v]}
        counts = {}
        for t in ts:
            for ix in t.inds:
                counts[ix] = counts.get(ix, 0) + 1
        outer[u] = tuple(ix for ix, count in counts.items() if count == 1 and ix not in inner)
    if max_bond is None:
        # Uncapped SRC samples enough columns to span every original cut.
        rank = max((math.prod(sizes[ix] for ix in bix) for bix in bonds.values()), default=1)
    else:
        rank = int(max_bond)
        if rank < 1:
            raise ValueError("max_bond must be positive")
    batch = qtn.rand_uuid()
    noise = {}
    if method == "src":
        rng = (
            ar.get_namespace(like=next(iter(local.values()))[0].data)
            .random.default_rng(seed)
            if seed is not None else None
        )
        for u in dict.fromkeys(u for u, _ in schedule):
            if outer[u]:
                data = backend_random_array(
                    (rank, *(sizes[ix] for ix in outer[u])),
                    like=local[u][0].data,
                    dtype=ar.get_dtype_name(local[u][0].data),
                    rng=rng,
                )
                noise[u] = qtn.Tensor(data, inds=(batch, *outer[u]))
    environments = {}
    latent = {}

    def release(key):
        uses[key] -= 1
        if uses[key] == 0:
            del environments[key]
            del latent[key]

    def environment(u, v):
        key = u, v
        if key in environments:
            return environments[key]
        ts = list(local[u])
        left = list(outer[u])
        for w in neighbors[u]:
            if w != v:
                ts.append(environments[w, u])
                left.extend(latent[w, u])
        if method == "src":
            if u in noise:
                ts.append(noise[u])
            if not any(batch in t.inds for t in ts):
                # A physically empty complementary component is a scalar
                # boundary vector, replicated across the sample index.
                ts.append(qtn.Tensor(ar.do("ones", (rank,), like=ts[0].data,
                                          dtype=ar.get_dtype_name(ts[0].data)),
                                     inds=(batch,)))
            message = qtn.tensor_contract(*ts, output_inds=(batch, *bonds[key]),
                                          drop_tags=True)
            # Uniform rescaling leaves the sampled range unchanged and avoids
            # exponential scale drift along large complementary components.
            scale = ar.do("max", ar.do("abs", message.data))
            message.modify(data=message.data / ar.do("where", scale > 0, scale, 1.0))
            latent[key] = (batch,)
        else:
            tensor = qtn.tensor_contract(*ts, output_inds=(*left, *bonds[key]),
                                         drop_tags=True)
            _, message = tensor.split(
                # Deterministic truncated SVD forms the low-rank environment
                # directly. It avoids Gram-matrix conditioning and the
                # installed NumPy complex64 svd:eig JIT failure.
                left_inds=left, right_inds=bonds[key], method="svd",
                absorb="right", max_bond=max_bond, cutoff=cutoff,
                cutoff_mode=cutoff_mode, get="tensors",
            )
            latent[key] = tuple(ix for ix in message.inds if ix not in bonds[key])
        environments[key] = message
        for w in neighbors[u]:
            if w != v:
                release((w, u))
        return message

    # Build fixed complementary environments before making any projections.
    for u, v in schedule:
        environment(u, v)
    noise.clear()
    pending = {u: list(ts) for u, ts in local.items()}
    result = {}
    records = []
    for u, v in order:
        env = environments[v, u]
        if method == "src":
            # More columns than the original cut dimension can only add
            # arbitrary null-space Q columns, needlessly growing exact bonds.
            cut_rank = math.prod(sizes[ix] for ix in bonds[u, v])
            if cut_rank < rank:
                env = env.isel({batch: slice(0, cut_rank)})
        sample = qtn.tensor_contract(*pending[u], env, drop_tags=True)
        left = tuple(ix for ix in sample.inds if ix not in latent[v, u])
        q, _ = sample.split(left_inds=left, right_inds=latent[v, u],
                            method="qr", absorb="lorthog", get="tensors")
        release((v, u))
        new_bond, = (ix for ix in q.inds if ix not in sample.inds)
        message = qtn.tensor_contract(*pending.pop(u), q.H,
                                      output_inds=(new_bond, *bonds[u, v]),
                                      drop_tags=True)
        result[u] = q
        pending[v].append(message)
        records.append((u, v, math.prod(sizes[ix] for ix in bonds[u, v]),
                        q.ind_size(new_bond), new_bond))
    result[hub] = qtn.tensor_contract(*pending[hub], drop_tags=True)
    # As in Quimb SRC, structural tags belong to the final local tensor,
    # never to sketches accumulating an entire complementary component.
    for u, tensor in result.items():
        tensor.modify(tags=tuple(dict.fromkeys(tag for t in local[u] for tag in t.tags)))
    return result, records
