# `pepsy.operators.hamiltonians`

`ham_tn.to_mpo` accepts explicit local operators or compact Pauli terms and
returns a standard Quimb MPO. By default it adds and compresses one term at a
time, which keeps intermediate bonds bounded; the workload-aware automatic
construction remains available explicitly:

```python
builder = pepsy.ham_tn(shape=20)
mpo = builder.to_mpo(
    [
        ((pepsy.x,), (10,), 0.5),
        ((pepsy.z, pepsy.z), (10, 11), 1.2),
    ],
    max_bond=64,
    cutoff="auto",
    cutoff_mode="auto",
)
```

`shape` is the geometry spelling shared with `exp_mpo`: use an integer for a
1D chain, `(Lx, Ly)` for a 2D lattice, or `(Lx, Ly, Lz)` for a 3D lattice.
The legacy `Lx=..., Ly=..., Lz=...` spelling remains supported, but cannot
conflict with `shape` when both are supplied.

`ham_tn.build_mpo(...)` remains a deprecated compatibility wrapper; new code
should use `to_mpo(...)`.

Compact Pauli terms can use either requested spelling:

```python
terms = [
    ((10,), "X", 0.5),
    (("ZZ", 1.2), (10, 11)),
]
mpo = builder.to_mpo(terms, max_bond=64, cutoff="auto", cutoff_mode="auto")
```

Set `to_backend` on the builder to convert the finished MPO tensors to a
selected array backend. The initial local operators are created using
`data_type`, then the MPO additions and truncations are carried out on the
selected backend:

```python
import torch

to_backend = pepsy.backend_torch(dtype=torch.complex128)
builder = pepsy.ham_tn(Lx=20, Ly=1, to_backend=to_backend)
mpo = builder.to_mpo(
    terms,
    chi=64,
    form="left",
    cutoff="auto",
    cutoff_mode="auto",
    method="svd",
)
assert all(isinstance(tensor.data, torch.Tensor) for tensor in mpo)
```

When `data_type` is omitted, the builder infers it from the converter's target
dtype. Without `to_backend`, it defaults to `float64`. `chi` is an alias for
`max_bond`; `form="left"` is forwarded to Quimb's compressor. An omitted
`max_bond` inherits the builder cap, while `max_bond=None` or `False` disables
numerical compression. Additional Quimb compression keywords can be supplied
through `compress_opts`.

Use `compress="term"` (the default) to add and compress each term
sequentially. Use `compress="automaton"` to canonicalize equivalent terms,
compile the complete list with Pepsy's shared finite-state MPO automaton, and
then apply one final numerical compression to `chi` when a bond cap is active.
`compress=True` and `compress="auto"` explicitly select the workload-aware
automatic policy: they use the automaton when its estimated structural width
is reasonable, and otherwise use a sequential term sum. `compress=False`,
`max_bond=None`, and `max_bond=False` disable numerical compression while
preserving the selected exact construction. The older separate `mode=`
keyword remains accepted for compatibility.

The automaton preparation combines duplicate product terms, folds all one-site
terms acting on the same site, and removes identity factors from two-site
terms. These are exact algebraic simplifications; `chi` and `cutoff` control
the numerical compression sweep(s) selected by the construction strategy.

For a 2D builder, locations can be lattice coordinates and are mapped through
`OneDMap`; a one-site coordinate can be written as `((x, y),)`:

```python
mapper = pepsy.OneDMap(Lx=4, Ly=4, mode="snake")
builder = pepsy.ham_tn(shape=(4, 4), mapper=mapper)
mpo = builder.to_mpo(
    [
        (((0, 0),), "X", 0.5),
        (("ZZ", 1.2), ((0, 0), (1, 0))),
    ],
    max_bond=64,
    cutoff="auto",
    cutoff_mode="auto",
)
```

`cutoff="auto"` selects ``1e-3`` for 16-bit data, ``1e-6`` for
32-bit/complex64 data, and ``1e-12`` otherwise. `cutoff_mode="auto"` selects
Pepsy's usual ``rsum2`` policy. For local-term construction,
`compress="term"` is the default after-each-term policy. `compress=True` or
`compress="auto"` explicitly selects the workload-aware automatic route,
while `compress="automaton"` forces the finite-state route. The old `mode=`
and `compress_each=` spellings remain available for compatibility.

The builder is the shared configuration point for these common conversion
defaults. A conversion can override its traversal or compression locally
without changing the builder or the other representations:

```python
builder = pepsy.ham_tn(
    shape=(4, 4),
    map_mode="snake",
    max_bond=64,
    cutoff="auto",
    cutoff_mode="auto",
)

mpo = builder.to_mpo(terms, map_mode="row-major", form="left")
tree_mpo = builder.to_tree_mpo(
    terms,
    map_mode="coarse-alternate-x",
    compress_opts={"order": "rank"},
)
tree_pepo = builder.to_tree_pepo(
    terms,
    map_mode="span-middle",
    form="left",
)
```

`to_tree_mpo` builds a workload-aware `TreePlan` automatically when no plan or
mapping is passed; explicit `map_mode` for this route uses the canonical
`coarse-*` geometric vocabulary, for example `"coarse-alternate-x"`, and the
resulting `TreeMPO.map_mode` is available on the operator and its plan.
`to_tree_pepo` similarly builds a workload-aware `TreePepsPlan`, but its
canonical `map_mode` is one of `"span-up"`,
`"span-down"`, `"span-out"`, or `"span-middle"`. That mode controls the
retained physical spanning tree; the builder's ordinary map remains the
logical site order. Both resulting state/operator families report the same
mode through `.map_mode` when they share a plan. Historical generic PEPO map
spellings remain accepted for compatibility. Both native methods accept
`compress="term"` by default and add/compress one term at a time.
`compress=True` or `compress="auto"` explicitly chooses the workload-aware
native state-diagram route; it uses one final compression for automaton
assembly and per-term compression for term assembly. With no explicit `plan`,
`map_mode`, or `tree_order`, the automatic route chooses a
TreePlan/TreePepsPlan from the term-support graph; an explicit plan or mapping
always wins. `compress="automaton"` forces full native assembly. These
numerical compressions require an effective bond cap. The native tree
compression options are
`order="rank"` or
`order="depth"` for both native `TreeMPO` and `TreePEPO` compression, and
`form`, `center`, and `reduced` are also accepted by `TreePEPO`; common
`max_bond`, `cutoff`, and `cutoff_mode` values come from
`ham_tn` unless overridden. An omitted `max_bond` inherits that builder cap;
`max_bond=None` or `False` disables numerical compression. For a consistent
conversion API, `compress=` is
the single canonical strategy control accepted by all four `to_*` methods:
use `True`, `False`, or the explicit strategy strings
`"term"`/`"automaton"`/`"auto"`. Passing `to_backend=None` explicitly
disables a builder-level backend converter for one conversion.

Automatic tree mapping is a bounded workload search over legal arities and
spanning-tree seeds; it is deterministic for a fixed term list, but it is not
a claim of a globally optimal tree (that problem is combinatorial). The
selected operator retains its layout finder for inspection and later reuse.

The builder also accepts a `Fermion` model for symmetry-aware MPO construction:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
builder = pepsy.ham_tn(Lx=3, Ly=1)
mpo = builder.to_mpo(
    fermion=fermion,
    edges=[(0, 1), (1, 2)],
    t=1.0,
    U=2.0,
    mu=0.1,
)
```

The native model-facing shorthand is
`fermion.build_mpo(edges, L=3, t=..., U=..., mu=...)`. Couplings remain
explicit; they are not stored on the `Fermion` object. Pass `fermionic=False`
to the same builder when the Jordan-Wigner-compatible MPO convention is
wanted.

`Fermion.build_mpo(...)` and `SymHamiltonian.to_mpo(..., fermionic=True)` return
native graded `FermionicArray` MPO tensors. Explicit mappings can contain
arbitrary homogeneous-charge multi-site terms; non-contiguous supports are
represented by charged virtual channels, and the open boundary carries the
operator charge when it is nonzero. `ham_tn.to_mpo(..., fermionic=True)`
selects the same native path. `Fermion.to_mpo(...)` remains a compatibility
alias. Pass `to_backend=...` to map the stored Symmray blocks to a selected
array backend.

For a mixed-charge operator, request an explicit charge-sector decomposition:

```python
sectors = fermion.build_mpo(
    mixed_terms,
    L=4,
    fermionic=True,
    charge_sectors=True,
)
# sectors[charge] is one homogeneous native MPO.
```

The same `charge_sectors=True` option is available on
`SymHamiltonian.to_mpo`, `Fermion.to_pepo`, `SymHamiltonian.to_pepo`,
`ham_tn.to_mpo`, and `ham_tn.to_pepo`; those methods return
`{charge: MPO}` or `{charge: PEPO}`. This keeps each block-sparse tensor
network within one charge sector while preserving the exact sum decomposition.

The corresponding 2D entry points all use the same `OneDMap` ordering:

```python
mapper = pepsy.OneDMap(3, 2, mode="snake-row-major")

pepo = hamiltonian.to_pepo(
    Lx=3,
    Ly=2,
    mapper=mapper,
    fermionic=True,
)
pepo = fermion.build_pepo(
    {(left, right): native_term},
    Lx=3,
    Ly=2,
    mapper=mapper,
    fermionic=True,
)
pepo = builder.to_pepo(
    {(left, right): native_term},
    fermion=fermion,
    mapper=mapper,
    fermionic=True,
)
```

Use a coordinate-keyed mapping for native terms, with one-site support written
as `((x, y),)`. PEPO embedding currently requires `snake` or
`snake-row-major` ordering; transverse lattice bonds are rank one unless
periodic PEPO bonds are requested with `cycle_peps=True`.

Native MPO assembly, replay, and exact energy measurement are supported. The
native energy path applies the MPO sitewise as a factorized graded MPO-MPS
contraction, so it does not materialize the global physical operator. Its cost
is controlled by the MPS and MPO bond dimensions.

## Native tree operators

Use `to_tree_mpo` or `to_tree_pepo` when the target geometry is already a
`TreePlan` or `TreePepsPlan`:

```python
tree_mpo = builder.to_tree_mpo(tree_plan, terms, max_bond=64)
tree_pepo = builder.to_tree_pepo(tree_peps_plan, terms, max_bond=64)
```

These conversions factor terms directly over native tree spans. They do not
construct an intermediate chain MPO, and the returned `TreeMPO` or `TreePEPO`
can be sent directly to the corresponding tree optimizer. The short aliases
`to_treempo` and `to_treepepsmpo` are retained for compatibility. Dense tree
operators support exact `+`, `-`, scalar multiplication, and `@`; call
`compress(...)` explicitly when a compressed sum or product is wanted.
The returned operators retain their plan and the workload-aware
`layout_finder` metadata, so that context can be reused later. Native
fermionic `TreeMPO @ TreeMPO` composition remains guarded until a graded
fused-bond composition kernel is available.

Both native operator classes expose `ascii_tree()` for a returned drawing and
`.show()` for printing it. `TreeMPO.show()` uses the root-first tree view.
`TreePEPO.show()` follows Quimb's PEPO-style coordinate view by default: it
draws the retained bonds and their current dimensions while leaving removed
lattice edges as gaps. Use `tree_pepo.show(layout="tree")` or
`ascii_tree()` for the native topology view.

> API details are maintained as handwritten Markdown in this page.
