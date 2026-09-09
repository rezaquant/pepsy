"""Successive compression of layered, acyclic tensor networks.

SRC contracts product-noise sketches of the complementary branches. SDC
contracts deterministic low-rank factors of those branches instead. A second
pass constructs nested output isometries and projects the *original* target
onto them. Neither algorithm truncates an already materialized target tree.
"""

import math
import warnings

import autoray as ar
import quimb.tensor as qtn

from ..._internal.random import backend_random_array


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
    neighbors = {u: [] for u in local}
    indices = {u: {ix for t in ts for ix in t.inds} for u, ts in local.items()}
    sizes = {ix: t.ind_size(ix) for ts in local.values() for t in ts for ix in t.inds}
    bonds = {}
    for u, v in order:
        neighbors[u].append(v)
        neighbors[v].append(u)
        bonds[u, v] = bonds[v, u] = tuple(sorted(indices[u] & indices[v]))
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
        for offset, u in enumerate(sorted(local)):
            if outer[u]:
                data = backend_random_array(
                    (rank, *(sizes[ix] for ix in outer[u])),
                    like=local[u][0].data,
                    dtype=ar.get_dtype_name(local[u][0].data),
                    rng=None if seed is None else int(seed) + offset,
                )
                noise[u] = qtn.Tensor(data, inds=(batch, *outer[u]))
    environments = {}
    latent = {}

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
            message = qtn.tensor_contract(*ts, output_inds=(batch, *bonds[key]))
            # Uniform rescaling leaves the sampled range unchanged and avoids
            # exponential scale drift along large complementary components.
            scale = ar.do("max", ar.do("abs", message.data))
            message.modify(data=message.data / ar.do("where", scale > 0, scale, 1.0))
            latent[key] = (batch,)
        else:
            tensor = qtn.tensor_contract(*ts, output_inds=(*left, *bonds[key]))
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
        return message

    # Build fixed complementary environments before making any projections.
    for u, v in order:
        environment(u, v)
    for u, v in reversed(order):
        environment(v, u)
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
        sample = qtn.tensor_contract(*pending[u], env)
        left = tuple(ix for ix in sample.inds if ix not in latent[v, u])
        q, _ = sample.split(left_inds=left, right_inds=latent[v, u],
                            method="qr", absorb="right", get="tensors")
        new_bond, = (ix for ix in q.inds if ix not in sample.inds)
        message = qtn.tensor_contract(*pending[u], q.H,
                                      output_inds=(new_bond, *bonds[u, v]))
        result[u] = q
        pending[v].append(message)
        records.append((u, v, math.prod(sizes[ix] for ix in bonds[u, v]),
                        q.ind_size(new_bond), new_bond))
    result[hub] = qtn.tensor_contract(*pending[hub])
    return result, records
