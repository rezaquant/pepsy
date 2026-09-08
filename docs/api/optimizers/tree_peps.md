# `pepsy.optimizers.tree_peps`

`TreePeps` is the first PEPS-like state class for tree-embedded tensor
networks. Every lattice site keeps one physical tensor, while the retained
virtual bonds are a validated spanning tree of an open 2D or 3D lattice.

The state exposes both coordinate and logical identities:

```python
from pepsy.optimizers import TreePEPO, TreePeps, TreePepsPlan

plan = TreePepsPlan.from_shape((3, 4), order="snake")
state = TreePeps.rand(plan, bond_dim=4, phys_dim=2, seed=7)

state.site_tag(1, 2)       # "I1,2"
state.logical_site_tag(7)  # "I7"
state.site_ind(1, 2)       # "k1,2"
state.site_ind_1d(7)       # "k1,2" (the same physical leg)
```

The retained tree degree is capped at four virtual bonds per site. Thus a
state tensor has at most rank five: one physical leg plus at most four
virtual legs. A normal `topology="tree"` plan also requires at least one
three-virtual-bond site (a rank-four tensor), so it cannot silently degenerate
to an MPS. `state.max_virtual_degree`, `state.tensor_rank(site)`,
`state.max_rank`, `state.topology`, and `state.is_branching` expose these
diagnostics.

The physical index is intentionally present only once. The logical 1D
address is represented by an additional tag, not by a second physical leg.
Each tensor also carries a structural `N{q}` tag, making it straightforward
to select either lattice sites or tree regions with Quimb operations.

For a workload-adapted tree, score the physical lattice supports before
constructing the state. The finder returns a regular `TreePepsPlan`, so the
same result can be passed to the state, PEPO constructors, and optimizer:

```python
from pepsy.optimizers import TreePepsLayoutFinder

layout = TreePepsLayoutFinder(
    plan,
    interactions=[(dense_gate, (0, 7)), (dense_gate, (2, 5))],
    objective="hybrid",
    seed=0,
).recommend()

state = TreePeps.rand(layout, bond_dim=4, seed=7)
operator = TreePEPO.from_operator(layout, dense_gate, support=(0, 7))
optimizer = TreePepsOptimizer(state, plan=layout)
```

When a `TreePEPO` is built from a workload, it can also retain the
`TreePepsLayoutFinder` itself as `operator.layout_finder`; this preserves the
coordinate/workload context for later layout-aware diagnostics or plotting.

`objective="span"` minimizes the weighted number of virtual edges in each
gate’s minimal tree span. `"load"` emphasizes peak routed edge demand, and
`"hybrid"` combines both with total edge load. The bounded refinement is
deterministic for a fixed `seed`; inspect `finder.report` for spans, edge
loads, degree, and rank diagnostics.

The finder compares the source tree with deterministic spanning-tree seeds and
workload-weighted growth. The canonical PEPS names are `span-up`, `span-down`,
`span-out`, and `span-middle`; they can be supplied directly as `map_mode=`, or
as one of `seed_modes` (with `tree_orders` and singular `tree_order` retained
as aliases). `span-up` and `span-down` use a boundary plane with axial teeth,
`span-out` grows from the geometric centre, and `span-middle` keeps a central
horizontal line/plane and attaches one vertical/axial chain above and below
each backbone site. Interior backbone tensors therefore have four virtual
bonds, while interior off-backbone tensors have two. All four
work for 2D and 3D open lattices. The historical `row-major`, `col-major`,
`hilbert`, `inside-out`, `diag`, and `snake` seed spellings remain accepted;
`inside-out`, `center-out`, and `outward` now canonicalize to `span-out` when
they describe a retained tree. The selected seed, map mode, candidate count,
and seed modes are recorded in `finder.report`.

For native tree operators, `.ascii_tree()` returns a top-down tree drawing and
`.show()` prints it with Unicode branches and bond dimensions. `TreePEPO.show()`
follows Quimb's PEPO-style coordinate view by default: it draws retained tree
bonds and their dimensions while leaving removed lattice edges visible as
gaps. Use `show(layout="tree")` or `ascii_tree()` for the native topology
view. Three-dimensional states use the same coordinate schematic layer by
layer.

`TreePepsPlan.from_shape` uses a branching spanning tree by default. For the
canonical API, pass one simple `map_mode` string for the retained physical
tree:

```python
plan = TreePepsPlan.from_shape((4, 4), map_mode="span-middle")
state = TreePeps.from_plan(plan)
operator = TreePEPO.identity(plan)
assert state.map_mode == operator.map_mode == "span-middle"
```

The logical site order (`order`) is still independent when needed. `span-up`
and `span-down` use a boundary row/plane as a backbone with nearest-neighbour
teeth through the lattice. `span-out` starts at the geometric centre and grows
outward by Manhattan distance. `span-middle` keeps a straight central
row/plane and extends one vertical/axial chain in each direction from every
backbone site; its interior backbone sites have four virtual bonds and its
off-backbone chain sites have two. The same
definitions extend from 2D to 3D by replacing each backbone row with a
nearest-neighbour snake through the transverse plane.

The old `tree_order` and generic map spellings remain compatibility inputs.
In particular, `inside-out`, `center-out`, and `outward` are aliases for
`span-out`. Generic `row-major`, `col-major`, `snake`, `hilbert`, and
`coarse-*` spellings still work as legacy traversal-growth modes, but new
PEPS code should use `span-*` so it cannot be confused with TreeMPO's
coarsening vocabulary. Legacy coarse modes accept the same
`coarse_grain=(gx, gy[, gz])` control on `TreePepsPlan` and
`TreePepsLayoutFinder`.

One-dimensional and geometrically non-branching lattices must opt into their
MPS-compatible path topology explicitly:

```python
path_plan = TreePepsPlan.from_shape((1, 16), topology="path")
assert path_plan.is_mps_topology
```

```python
plan = TreePepsPlan.from_shape(
    (4, 4), order="row-major", tree_order="inside-out"
)
plan.coordinate(plan.root)  # (1, 1)
```

Custom tree edges can be supplied as logical-id pairs or coordinate pairs,
for example:

```python
plan = TreePepsPlan.from_shape(
    (2, 2, 2),
    tree_edges=[
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 0, 1)),
        ((1, 0, 1), (0, 0, 1)),
    ],
    max_virtual_degree=3,
    topology="path",
)
```

This particular custom edge list is a Hamiltonian path, so it is marked
explicitly as `topology="path"`. A custom branching edge list can keep the
default `topology="tree"`.

The state API includes exact `norm`, `to_dense`, and local observable readout,
together with `show`, `canonicalize`, `canonize_subtree`, and `compress`.
Canonical operations track `canonical_region` and `orthogonality_center`, and
store each proven outward isometry in the tensor's Quimb-compatible
`left_inds`. Moving a known center uses only the unique tree path and skips QR
when those local proofs already establish the required edge gauge. A
multi-site canonical region can be reduced to a center before the path move,
and the center-oriented compression sweep performs the inward edge reductions
without a redundant full-tree QR. Native tree compression uses the same kind
of live rank-aware scheduling as `TreeMPO`: at each step it chooses among the
current leaves using the physical and virtual dimensions, then removes that
branch toward the selected center. Pass `order="depth"` to retain the
deterministic farthest-first schedule. The plan and lattice layout are never
changed by either policy. Callers that already use Quimb-style
optimizer state can pass a mutable `info_c` mapping to synchronize
`cur_orthog`, `canonical_region`, `isometry_map`, and `left_inds` snapshots.

The first operator layer is now available through the canonical `TreePEPO` and
`TreeSubPEPO` names (with `TreePepo` and `TreeSubPepo` retained as aliases):

```python
from pepsy.optimizers import TreePEPO, TreeSubPEPO

gate = TreeSubPEPO.from_operator(plan, dense_gate, support=(0, 5))
updated = gate.apply_to(state, compress=True, max_bond=8)
value = gate.expectation(state)
```

`TreePEPO` is the normal/full tree-PEPS operator, while `TreeSubPEPO` is its
support/span-aware sub-operator—the PEPS analogue of an MPS sub-MPO. The
canonical method and event names are `apply_sub_treepepo` and
`sub_treepepo_event`; MPS-style compatibility spellings
`apply_sub_treepepsmpo` and `sub_treepepsmpo_event` are accepted and resolve to
the same implementation. The full-operator event also accepts the matching
`tree_pepsmpo_event` spelling. No second MPO representation is created.

Dense `TreePEPO` term sums also receive an exact structural edge sweep before
their center-oriented SVD compression. The sweep is restricted to NumPy
arrays, removes proportional and roundoff-safe linearly dependent boundary
channels, and leaves the retained tree geometry unchanged. Backend tensors
that carry autodiff or native symmetry metadata continue through their
existing compression paths.

`ham_tn.to_tree_pepo(...)` defaults to `compress="term"`, so it adds and
compresses one term at a time. Pass `compress=True` or `compress="auto"` for
the workload-aware route, or `compress="automaton"` to force full native
assembly. Pass `progbar=True` for the matching MPS-style construction bar.
Term mode advances once per added term and reports the current `chi` against
the requested cap; the progress-bar default is `False`.

For an explicit path-method selection, the same call can retain both layers
until Quimb compresses them:

```python
updated = gate.apply_to(
    state,
    compress=True,
    compression_mode="zipup",
    compression_layout="two_layer",
    max_bond=8,
)
```

`TreePEPO` is a generic tree operator with separate input/output physical
legs. `TreeSubPEPO` records the physical support and its connected tree span.
The default `compression_layout="auto"` preserves the fused operator/state
application for ordinary and branching updates, while path updates using
Quimb's multi-tensor methods can retain the separate operator and state
layers until compression. Use `compression_layout="fused"` to force the
original fused path, or `"two_layer"` to require the path-only MPO-MPS-style
path. The full design and the future
`TreePepsStabOptimizer` interface are documented in the development plan.

`TreePepsOptimizer` owns a state copy by default and supports the two update
paths:

```python
from pepsy.optimizers import TreePepsOptimizer

direct = TreePepsOptimizer(state, mode="direct", chi=16)
direct.apply_gate(dense_gate, where=(0, 5))

subtree = TreePepsOptimizer(state, mode="sub_treepepo", chi=16)
subtree.apply(subop)
```

Its default `cutoff="auto"` follows the shared MPS policy: `1e-12` for
`float64`/`complex128`, `1e-6` for `float32`/`complex64`, and `1e-3` for
16-bit floating-point data. Pass a numeric cutoff to override it explicitly;
`cutoff=None` retains the legacy `1e-10` compatibility value.

Direct gates are first built as normal `TreePEPO` operators and wrapped in a
`TreeSubPEPO`, then routed over the unique connected tree span. This mirrors
MPS `sub_mpo`: the complete span is injected before one localized
leaf-to-center compression sweep. Full `TreePEPO` inputs remain the normal
operator path. Both paths keep intermediate routing lossless and use
`left_inds`-aware canonical movement. For compatibility with MPS code,
`sub_treepepsmpo` is accepted as an alias for `sub_treepepo`; the normal/full
operator remains `TreePEPO` (or a `tree_pepo` stream event).

The optimizer also owns a persistent, replayable stream. Install or extend
it without executing the state, then call `run()` when ready:

```python
streamed = TreePepsOptimizer(state, run=False, chi=16)
streamed.set_gates([
    (dense_gate, (0, 5)),
    TreePepsOptimizer.sub_treepepo_event(subop),
])
streamed.add_gates([
    TreePepsOptimizer.gate_event(dense_gate, 2),
])
streamed.run()
```

The accepted event forms are `(gate, where)`, tagged
`("gate", gate, where)`, a `TreePEPO`, or a `TreeSubPEPO`; mapping forms with
`kind`, `gate`/`where`, or `operator` keys are also accepted. `run()` without
arguments replays the currently queued stream, while `run(gates)` preserves
the older one-shot spelling by replacing the queue first. `run(mode=...,
compression_mode=...)` uses the same persistent selection model: the route
and compression mode are normalized and stored before replay, and the
resolved pair is passed to every queued gate, full `TreePEPO`, and
`TreeSubPEPO` event. Thus `run(mode="sdc")` persists
`mode="direct", compression_mode="sdc"` and applies that compression to
explicit sub-operator entries as well. The `sub_treepepsmpo` spelling is
normalized to the canonical `sub_treepepo` route. The normalized stream is
available as `gate_stream`. Convenience methods `apply_1q`,
`apply_2q`, `apply_multi_site`, and `apply_pepo` match the corresponding
optimizer vocabulary.

Use `set_state(new_state)` (or assign `optimizer.tn`) to replace the live
state. The new state must have the same tree plan, and all queued operator
payloads must match its backend, device, and required dtype contract before
the replacement is installed. By default the replacement is copied; use
`inplace=True` at construction when state identity should be retained.
`backend_info()` reports the live state metadata.

Compression is selected independently from the operator route with
`compression_mode="direct"` (the default SVD decomposition) or
`compression_mode="dm"` (Quimb's density-matrix-equivalent `svd:eig`
decomposition of the local fused compression core). `compression_mode="zipup"`
is also available for path operator-state compression. In fused mode the
state is canonicalized around the active span first, the PEPO is fused locally
with the state, and only then are the combined tree bonds truncated. In
two-layer mode, the state and PEPO tensors are grouped by the same site tags
and passed to Quimb's 1D compressor as an MPO-MPS-like network. No global
dense lattice state is formed. For convenience, `mode="dm"` is accepted as a
shorthand for direct TreePepo routing with `compression_mode="dm"`.

For branching updates, `TreePepsOptimizer` applies the complete operator to
the active connected span before truncating it. Its native leaf-to-center
sweep re-scores the remaining legal branches after every SVD/QR reduction, so
the next choice sees the current live bond dimensions; exterior branches are
not compressed. This is a geometry-aware local SVD policy, not a global search
over all possible tree gauges or layouts.

`mode="sdc"`, `mode="src"`, and `mode="zipup"` are also accepted shorthands
for direct TreePepo routing with the corresponding compression mode. On an
explicit path topology, `compression_layout="auto"` uses Quimb's actual 1D
SDC/SRC/ZipUp kernels with the separate operator and state layers, then
restores the TreePeps plan, tags, exponent, and canonical metadata. On a
branching topology, `sdc` uses the tree's deterministic successive edge sweep
and `src` uses randomized SVD per edge; neither silently invokes a chain-only
environment algorithm, while `zipup` is path-only. Truncating path `sdc`/`src`
requires finite `chi`/`max_bond` (`sdc` with zero cutoff may still be used as a
lossless canonicalization), and `compression_seed=...` makes randomized
results reproducible. The paper's full projected Cholesky (CBC) tree
compressor is not represented by these aliases and remains a separate future
method.

### TreePEPS FIT / DMRG

`TreePepsOptimizer` also exposes the tree-native `TreeFIT` engine through
`mode="dmrg"` and the `dmrg1`/`dmrg2`/`dmrg3` aliases. The exact layered
operator-state target is built from disposable tensor copies, then fitted on
the active connected tree span with cached directed branch environments:

```python
optimizer = TreePepsOptimizer(
    state,
    mode="dmrg2",
    chi=16,
    fit_n_iter=3,
    fit_init_strategy="guess-src",
)
optimizer.apply_gate(gate, where=(0, 5))
report = optimizer.get_fit_diagnostics()
```

Generic `dmrg` uses `fit_block_size=2` and its configured adaptive warm-up.
`dmrg1` and `dmrg2` use two-node warm-up blocks, while `dmrg3` uses
three-node warm-up blocks; all named modes then refine with one-node sweeps.
The remaining controls are `fit_adaptive_sweeps`, `fit_min_iter`, `fit_rtol`,
`fit_patience`, `fit_sweep_sequence`,
`fit_init_rand_strength`, and `fit_init_seed`. Initial guesses may be
`"direct"`, `"guess-src"`, `"guess-sdc"`, `"guess-dm"`, `"random"`, or
`"random_expand"`; random policies are disposable, seeded, and active-span
only. `get_fit_diagnostics()` reports the cache hit/miss counts, block size,
convergence, and the MPS-compatible retained-centre-norm `local_fidelity`.
TreeFIT records one terminal centre norm per sweep in
`local_norm_trace` (with stripped mantissa/exponent pairs in
`local_norm_stripped_trace`). A genuine normalized target overlap is optional
and is reported separately as `target_fidelity`/`target_infidelity` (with
`fit_overlap_fidelity`/`fit_overlap_infidelity` aliases) when
`fit_overlap_diagnostics=True`.

This DMRG path builds the exact TreePEPO target as a correctly tagged layered
operator--state network: state and TreePEPO virtual bonds remain separate,
and only the physical input/output legs are joined. TreeFIT also accepts
fused targets and other correctly tagged layered targets when each layer
tensor belongs to exactly one structural node group; local layer bonds remain
inside that group and inter-group bonds must follow the tree.
The separate two-layer operator-state path remains available for the direct
`sdc`/`src`/`zipup` compressors above when that Quimb path is desired.

## TreeTensorNetwork API parity

The dense TreePeps state and optimizer expose the reusable, geometry-neutral
parts of the existing `TreeTensorNetwork`/`TreeOptimizer` surface. The state
provides `nsites`/`nqubits`, `root`, `top_arity`, `is_binary`, rooted
`parent`/`children`/`is_leaf` helpers, `tree_distance` and `subtree_span`,
`max_bond`/`bond_sizes`/`bond_report`, batched `local_expectations`,
`to_statevector`, and in-place `normalize`. The optimizer provides the `p` and
`tn` state aliases, `center`/`orthogonality_center`, center movement and
`left_inds` validation delegates, `show`, `to_dense`, `norm`/`normalize`,
`bond_report`, conservative `estimate_bonds`/`preflight`, and a
`truncation_report` over its replay history. Logical site order is fixed by
the plan, so the optimizer's `qubits`, `logical_order`, `position`, and
`remap_sample` helpers are identity mappings.

The optimizer-level `canonicalize`/`canonize`, `canonize_subtree`,
`canonize_around_qubits`, and `compress` methods delegate to the live state and
refresh `info_c` after every center or region change. `compress(sites, span=True)`
uses the same minimal-span, leaf-to-center sweep as a gate update; its history
record reports `span`, `touched_edges`, `uncompressed_bonds`, and
`compression_scope="span"`, so exterior tree bonds are not silently included.
`run` also supports `non_unitary`, `normalize_every`, `normalize_final`, and
`track_infidelity` controls. `profile=True` enables update-envelope timings,
`track_bond_diagnostics=True` records transient versus live bond pressure, and
`max_intermediate_bond` can reject a queued stream during preflight before
replay begins. `TreePepsOptimizer.find_tree_layout(...)` and
`convergence_sweep(...)` provide the corresponding layout and bond-cap
convenience entry points.

Pass `progbar=True` to `run()` for a replay bar matching the MPS optimizer's
compression readout. It reports the active mode, exact two-qubit gate count
`2q`, cumulative retained fidelity as `~F`, and the live maximum bond as `bnd`;
it does not display the live state norm. `~F` is a retained-norm compression
proxy, not a directional overlap with a target state. The latest local
fidelity remains available after replay as
`norm_diagnostics()["local_fidelity"]`, alongside
`norm_diagnostics()["cumulative_fidelity"]` and the matching infidelity fields.
Tree-specific `kq` and `pepo` counters are included when multi-site gates or
explicit PEPO events occur.

As with the state, TreePeps truncation history records exact bond dimensions;
it does not claim a scalar discarded-weight fidelity unless a caller performs
an explicit reference comparison (the convergence sweep does so for small
enough dense states).

The intentionally absent TTN-only paths are qubit measurement/reset/capping,
`TreeMPO` expectation/application, and native Symmray/fermionic support.
TreePeps sites can have general physical dimensions and its lattice plan is
fixed during compression; these operations need dedicated physical-space and
topology contracts rather than a compatibility alias. Stabilizer replay and
structured TreePePO backends remain separate roadmap phases.
