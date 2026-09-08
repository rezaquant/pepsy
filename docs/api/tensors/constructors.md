# Tensor constructors

## Explicit MPS-to-TTN conversion

`pepsy.tensors.mps_to_ttn` (also `pepsy.mps_to_ttn`) converts an existing
entangled MPS onto a `TreePlan`. Unlike `ps_to_ttn`, it preserves the input
state rather than creating a new product state.

```python
import pepsy as py

plan = py.TreePlan.from_order([0, 2, 4, 1, 3, 5], structure="balanced")
exact_tree = py.mps_to_ttn(p0, tree=plan, chi=None)
small_tree = py.mps_to_ttn(p0, tree=plan, chi=2)

engine = py.TreeOptimizer(
    gates, tree=plan, state=exact_tree, chi=60,
    mode="dmrg1", fit_n_iter=8, run=False,
)
engine.run()
```

Here `p0` is a six-site Quimb MPS and gate site labels retain their original
meaning. The leaf order chooses the tree geometry; it does not permute the
physical state. A plan with an optional physical root and arbitrary arity
is also supported. Omit `tree` for a balanced tree in MPS site order.

| Argument | Meaning |
| --- | --- |
| `chi=None` | Lossless QR conversion, up to floating-point roundoff. No cutoff, implicit normalization, or truncation. |
| `chi=k` | Cap every TTN bond at positive integer `k`, using sequential environment-aware density-matrix projections. |
| `optimize="greedy"` | Contraction-path policy. A Quimb/Cotengra optimizer can be supplied explicitly. |
| `max_intermediate_elements=2**26` | Raise `MemoryError` before a planned contraction intermediate or density matrix exceeds this many elements. `None` disables this guard. |
| `node_tag_id="N{}"` | Structural node tags; must not collide with the source tensor tags. |

The required rank of a tree bond is the Schmidt rank across that tree's
physical-site partition. Consequently, `chi < p0.max_bond()` can still be
exact, while `chi == p0.max_bond()` need not be. Exact TTN bonds can be
larger than the source MPS bonds. The finite-cap algorithm does not promise
the globally optimal TTN approximation or a monotonic fidelity curve as
`chi` changes.

Both paths leave `p0` untouched and preserve its physical index/site-tag
formats, source tensor tags, backend, dtype, device, extracted exponent,
and global phase convention. The result is canonical toward the root.
Finite-cap projections can reduce the norm; no renormalization conceals
that loss. Open and cyclic MPS are accepted, with one tensor per physical
site and labels `0 .. L-1`. Dense NumPy, Torch, CuPy, and JAX arrays are
supported. Symmray and fermionic MPS require a graded converter and are
explicitly rejected by this API.

There is no separate dense statevector handoff and the finite-cap path
does not construct the full exact TTN first. Unfavorable partitions or
large tree arity can nevertheless require large tensors or expensive
contractions. The resource guard limits individual planned arrays, not
total memory or QR/eigensolver workspace; it never silently changes `chi`.

For MPS/tree replay comparisons, perform conversion before starting the
replay timer. Check the initial conversion fidelity separately when using
a finite cap, so its approximation error is not attributed to replay.
