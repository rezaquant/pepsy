# `pepsy.optimizers.tree`

`TreeOptimizer` simulates a quantum circuit by replaying a canonical bundled
gate stream `[(gate, where), ...]` on a **rooted tree tensor network**, after
*Simulating quantum circuits using tree tensor networks* (Seitz, Medina, Cruz,
Huang, Mendl; Quantum 7, 964, 2023; [arXiv:2206.01000](https://arxiv.org/abs/2206.01000)).

By default the state is stored with one leaf tensor per qubit. A plan may
instead designate one `root_qubit`, placing that physical index directly on
the top tensor while every other qubit remains a leaf. Internal nodes may have
**any arity** -- the default is a binary tree below a three-virtual-leg root,
but a fixed binary root, flatter `k`-ary trees, or gate-connectivity-driven
communities (see *Tree structure*) all work through the same machinery.

For example, this constructs a binary detector tree whose logical qubit is the
open top index. The root tensor has two virtual child bonds and physical index
`k4`:

```python
from pepsy.optimizers.tree import TreeLayoutFinder, TreeOptimizer

finder = TreeLayoutFinder(gates, n=5, root_qubit=4, max_arity=2)
plan = finder.run(
    order="quality",
    refine="greedy",
    refine_budget=64,
    search="nevergrad",
    search_budget=128,
    progbar=True,
)
opt = TreeOptimizer(
    gates,
    tree=plan,
    chi=64,
    cutoff="auto",
    cutoff_mode="auto",
)
assert opt.plan.node_of_qubit[4] == opt.plan.root
assert set(opt.tn.node_tensor(opt.plan.root).inds) >= {"k4"}
```

`root_qubit` is first-class rather than an unregistered outer leg:
`to_dense()` retains it in normal qubit order, `cap(root_qubit, vec)` contracts
only that physical leg, and direct gates, dense subtree operators, and
structured sub-MPOs may include it in their support. `TreeLayoutFinder` keeps
the site fixed at the root while its path, Steiner, congestion, greedy, and
Nevergrad objectives permute only the remaining leaf sites.

## Layout-aware native MPOs

After selecting a plan, the canonical tree-native operator is built with
`plan.build_tree_operator(...)` or `Fermion.build_tree_operator(...)`. The
native path includes `U1FermionicArray` and `U1U1FermionicArray` tensors:

```python
from pepsy.tensors import Fermion
from pepsy.optimizers.tree import TreeLayoutFinder

finder = TreeLayoutFinder(gates, n=8, max_arity=2)
plan = finder.run(order="quality")
fermion = Fermion(spinful=True, symmetry="U1U1")
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)

tree_operator = plan.build_tree_operator(hamiltonian)
energy = tree_operator.expectation(opt.tn)
```

For a model-facing operator build, use `Fermion.build_tree_operator(...)`.
`Fermion.to_tree_mpo(...)` and `TreePlan.to_tree_mpo(...)` remain native
compatibility aliases. `TreeMPO` stores only the native tree representation:

```python
from pepsy.tensors import Fermion

tree_operator = fermion.build_tree_operator(
    hamiltonian=hamiltonian,
    tree=plan,
    compress=True,
)
energy = tree_operator.expectation(opt.tn)      # TreePlan-native readout
```

When a chain MPO is specifically required, call the model-level
`Fermion.to_mpo(...)` or `SymHamiltonian.to_mpo(...)` explicitly. Tree builders
do not create or attach chain MPOs.

`TreeMPO.from_terms(plan, dense_terms)` provides the corresponding ordinary
dense backend. Native fermionic `TreeMPO` objects retain Symmray arrays for
U1, U1U1, and other supported symmetries; dense and native operators are not
silently mixed with incompatible tree states.

For an exact native readout that keeps the chain MPO separate, use
`expectation_mpo_exact` as above. The general `expectation_mpo` API remains
available for an explicitly approximate structured-MPO application: it uses a
private transformed copy of the TTN. Routing itself is lossless, but the final
subtree sweep may compress that copy when an MPO increases a bond beyond
`max_bond`; `cutoff=0.0` does not disable a finite bond cap. The default
`warn_on_truncation=True` reports this approximation, and the diagnostic form
makes it easy to check a benchmark:

```python
energy, report = opt.tn.expectation_mpo(
    mpo,
    range(plan.n),
    max_bond=64,
    cutoff=0.0,
    return_diagnostics=True,
)
assert not report["truncated"]
```

Use a larger `max_bond` when an untruncated measurement is required. Native
fermionic trees also reject ordinary dense MPOs (and dense trees reject native
Symmray MPOs) instead of silently changing the fermionic interpretation.

For an exact readout that does not move the MPO into the tree at all, use
`expectation_mpo_exact`:

```python
energy = opt.tn.expectation_mpo_exact(
    mpo,
    range(plan.n),
)
```

The same method is available directly as `opt.expectation_mpo_exact(...)`.

This keeps the bra, ket, and structured MPO as separate networks. Fresh ket
physical indices connect to the MPO input legs, the MPO output legs connect to
the bra, and Quimb contracts the complete doubled network. No state-bond
compression or `to_dense()` lowering occurs. Native Symmray MPOs retain their
graded contraction and fermionic sign rules.

`TreePlan.mpo_order()` remains available as the structural leaf-position order
chosen by the plan; a physical `root_qubit`, when present, is placed first.
It is useful when a caller deliberately chooses a one-dimensional model-level
MPO ordering, but Tree builders do not construct a chain MPO from it.

`TreeMPO` is the tree-routed operator class. `tree_networks` contains the
TreePlan-labelled operator networks used by `expectation`; no second chain
representation is stored on the object. Native neutral
terms are factorized directly from their native Symmray operator tensor over
the term's TreePlan Steiner subtree, then amalgamated into one charge-aware
direct-sum TTNO. This applies to one-, two-, and higher-site native terms; it
does not create a hyperedge for the normal Hamiltonian path. The resulting
TTNO can be canonicalized and compressed with
`tree_operator.canonicalize()` and
`tree_operator.compress(cutoff=..., max_bond=...)`; no Jordan--Wigner
conversion is used. Nonzero or mixed operator charges remain separate
homogeneous native networks inside the same public `TreeMPO`, so callers do
not need `charge_sectors=True` just to construct one operator object;
`charge_sectors=True` remains available when separate objects are preferred.
Structured observables can use a smaller compact TTNO.
Pass `fermionic=False` only for dense ordinary/Jordan--Wigner-compatible terms.
Native `TreeMPO.identity()` preserves the operator's Symmray symmetry and
returns a bond-one TTNO that can be applied directly to a native
`TreeTensorNetwork`; dense identities remain ordinary dense TTNOs.
`OneDMap` is the shared source of truth for regular 2D/3D coordinate layouts;
the tree and MPS geometric layout finders consume its row/column, snake,
alternate-x/y/z, folded-snake, and generalized Hilbert traversals directly.
For a tree-native layout, the short canonical spelling is
`map_mode="coarse-alternate-x"` (or another `coarse-*` preset). It describes
the lattice coarsening/traversal used to label the leaves and is available as
`map_mode` on the resulting `TreePlan`, `TreeTensorNetwork`, and `TreeMPO`.
Pass the same plan through state, operator, and optimizer construction so all
three components share one binary-tree geometry.

`TreeMPO` subclasses Quimb's `TensorNetworkGenOperator`, in the same way that
`TreeTensorNetwork` subclasses `TensorNetworkGenVector`. It is the tree twin
of Quimb's `MatrixProductOperator`: the common operator surface includes
`sites`, `nsites`, `site_tag`, `upper_ind`, `lower_ind`, `to_dense`, `H`,
`copy`, `identity`, `from_dense`, `add_TreeMPO`, `singular_values`, and
`amplitude`. Tree-specific geometry is exposed through `plan`, `node_tensor`,
`neighbors`, and `bond`; `canonicalize`/`compress` perform the corresponding
tree-wide QR/SVD sweeps. It cannot inherit Quimb's chain-only
`MatrixProductOperator` implementation because a branched tree has no single
left/right ordering. A chain MPO for a chain workflow is constructed
separately with the model-level `to_mpo(...)`; it is not stored on `TreeMPO`.
`validate()`
checks every stored TTNO network against the
TreePlan; `validate(check_canonical=True)` also checks the tracked operator
`left_inds` directions when a canonical region is known.

Dense `TreeMPO` builder outputs run the same exact structural boundary sweep
over the rooted tree before the native tree SVD. It reduces proportional and
roundoff-safe linearly dependent edge channels in a leaf-to-root and reverse
pass, so direct-sum term assembly does not carry redundant states into every
parent tensor. Native Symmray/fermionic networks are left on their existing
charge-preserving QR/SVD route.

`ham_tn.to_tree_mpo(...)` defaults to `compress="term"`, so it adds and
compresses one term at a time. Pass `compress=True` or `compress="auto"` for
the workload-aware route, or `compress="automaton"` to force full native
assembly. Pass `progbar=True` for the same MPS-style construction bar. Term
mode advances once per added term and reports the current `chi` against the
requested cap; the progress-bar default is `False`.

`add_TreeMPO(..., compress=True)` first builds the tree direct sum and then runs
the same native tree SVD sweep; its compression options are `max_bond`,
`cutoff`, and `order`. The default `order="rank"` is a deterministic greedy
leaf-elimination policy that uses live edge dimensions to reduce small-rank
branches before they enlarge parent tensors. Use `order="depth"` for the
simple fixed depth-first sweep when reproducing older benchmarks. This is
rank-aware ordering on a fixed TreePlan, not a global search over alternate
tree geometries. Native graded Symmray TTNOs safely retain the charge-preserving
depth order and report that effective fallback, because arbitrary sibling
reordering requires an explicit graded permutation proof. Ordinary
`copy(transpose=True)` and `conj()` views preserve the
canonical gauge metadata when they keep the TreePlan index layout unchanged.
For native Symmray operators, addition uses a charge-aware TreePlan direct
sum; it does not use Quimb's dense-axis padding. A chain MPO, when needed, is
constructed separately with the model-level `to_mpo(...)`.
`add_MPO(...)` remains a compatibility alias for `add_TreeMPO(...)`.

`TreeMPO.ascii_tree()` returns a compact Quimb-inspired native drawing, and
`TreeMPO.show()` prints only this clean native tree by default, with one bond
dimension per branch. When a
`TreeLayoutFinder` with `lattice_shape=` is attached, the operator also keeps
the physical coordinate map and term supports for display:

```python
print(tree_operator.ascii_tree())
tree_operator.show(bond_dims=True)
tree_operator.show(layout="both")  # physical lattice above native tree
```

Hamiltonian builders attach this layout metadata automatically when their
lattice shape is known. For a manually constructed operator, pass the finder
explicitly:

```python
finder = TreeLayoutFinder(
    supports=terms,
    n=16,
    lattice_shape=(4, 4),
)
tree_operator = TreeMPO.from_terms(
    plan,
    terms,
    layout_finder=finder,
)
tree_operator.show(layout="lattice")
```

`layout="tree"` (the default) keeps the output to the native ASCII tree.
`layout="auto"` remains an explicit convenience option that shows both
sections when this metadata is available. `ascii_lattice()` shows
the physical site array and a compact support list; `plot_layout()` or
`show(layout="plot")` gives the Matplotlib tent view with the physical lattice,
tree hierarchy, and term connectivity.

Dense `TreeMPO` objects also provide exact `+`, `-`, scalar multiplication,
and operator composition with `@` while retaining the native TreePlan
network. Composition does not lower either operand to a chain MPO; use
`compress(...)` explicitly to truncate the resulting tree bonds. Composition
of native graded fermionic operators is intentionally guarded until a graded
fused-bond kernel is available.

`canonicalize(center=..., info_c=...)` performs lossless tree QR
canonicalization. The tree-native source of truth is `canonical_region` plus
each tensor's `left_inds`; when `info_c` is supplied, Pepsy mirrors the
single-node center as `info_c["cur_orthog"] = (center, center)` and stores the
connected region in `info_c["canonical_region"]`. It also records
`info_c["isometry_map"]` and immutable per-network `info_c["left_inds"]`
snapshots for diagnostics and optimizer synchronization.

`TreeMPO.from_terms(plan, terms)` accepts dense one-site, two-site, and
higher-order terms. A higher-order term is first factorized exactly on the
minimal Steiner subtree joining its physical sites, then combined with the
other terms by a TTNO virtual direct sum. Thus every resulting network still
has one operator tensor per `TreePlan` node; it is not a hyperedge that must be
lowered to a chain. The same Tree-native factorization is used by the dense
term path and by the native Hamiltonian builders (with graded Symmray tensors
on the fermionic path).

For a local gate, use `TreeMPO.from_gate(plan, gate, where)` instead of
materializing a full-system matrix. `where` always contains logical qubit
labels, independent of whether the plan was created from row-major, snake,
folded-snake, or Hilbert order. The constructor factors only the gate over
the minimal TreePlan Steiner subtree and adds bond-one identity legs elsewhere,
so the result can be sent directly to `TreeOptimizer.apply_subtreempo` or a
`subtreempo_event`. Dense gate factorization removes only machine-precision
null operator-Schmidt sectors; configured TreeMPO compression remains explicit.
`TreeMPO.from_pauli_sum(plan, weighted_terms)` provides the analogous compact
TTNO for a weighted sum of product-Pauli branches. It uses one virtual branch
channel per retained term only on the union of the active Steiner subtrees,
with bond-one identity legs outside; it never constructs a full-system dense
matrix or a chain MPO.

The conventional binary TTN with a three-leg top tensor is the default when
there are at least three leaves and no `root_qubit`. Pass
`max_arity=2, top_arity=3` explicitly to `TreePlan.from_order`,
`TreeLayoutFinder`, or `TreeTensorNetwork.from_order` for the same geometry.
The structural root then has three **virtual**
child bonds and no parent bond; every non-root internal tensor has two child
bonds and one parent bond. Thus the root is still in the rank-three binary
class, rather than being a genuinely wider tensor. `top_arity=3` is not
combined with `root_qubit`, because adding a physical root leg would make a
rank-four tensor. `TreePlan.is_binary()` accepts this ternary-root convention,
while `TreePlan.is_strictly_binary()` requests two children at every internal
node.

Gates are absorbed into the tree according to the selected optimizer mode:

- **ordinary `apply_gate` entries** in `auto`, `direct`, `dm`, `sdc`, `src`,
  and `mpo` are lowered to a true `TreeMPO` with `TreeMPO.from_gate`,
  factorized on the minimal TreePlan Steiner subtree, and applied with
  `apply_subtreempo`. The compression mode controls the tree-edge state
  compression, and gate width no longer causes a dense-route cliff.
- **`tree_mpo_direct` / `tree_mpo_dm`** are explicit names for the same
  TreeMPO path, with direct SVD or density-matrix compression selected by the
  suffix. This route never constructs a chain `sub_mpo`.
- **`submpo`** remains the explicit chain-MPO stream mode. The low-level
  `apply_1q`/`apply_2q` compatibility methods retain their specialized direct
  or chain-MPO kernels when called explicitly; this does not change the
  ordinary `apply_gate` contract.
- **stream events** -- MPS-compatible `measure`, `cap`, `reset`, and
  `measure_reset` entries can be mixed into the stream. Measurements use Pauli
  eigenvalue outcomes (`+1`/`-1`) and are appended to `measurements`;
  explicit `submpo` markers use the same native QR-routing and final subtree
  compression when the payload exposes Quimb's MPO site interface, so
  `to_dense()` is not required; opaque MPO-like payloads fall back to the dense
  recursive subtree-operator path.
  `cap` contracts and removes one physical site, compacts the remaining qubit
  labels above it, and keeps the live tree canonical.

The orthogonality centre is a single node id tracked on the
`TreeTensorNetwork` itself (`orthogonality_center`), so the state -- not any one
driver -- owns the canonical form; it survives `.copy()` and is what
`TreeOptimizer.center` reads. It is moved with
`TreeTensorNetwork.shift_orthogonality_center(node)`, the tree analogue of
Quimb's MPS `shift_orthogonality_center`: the centre is walked to the target
along the unique tree geodesic with a per-edge lossless QR (Quimb
`canonize_between`), touching only the tensors on that path (an O(path length)
move, not O(N)). The move is idempotent when already centred; when the centre is
unknown it is established once with Quimb `canonize_around`. This mirrors the
`info_c["cur_orthog"]` centre tracking of `MpsOptimizer`.
`TreeTensorNetwork.is_canonical_form(center)` verifies the property directly
(every non-centre tensor is an isometry toward the centre) as a diagnostic/test
aid. `TreeOptimizer` mirrors this public surface: `TreeOptimizer.center` (with
the `orthogonality_center` name-parity alias), `shift_orthogonality_center(node)`
and `is_canonical_form(center)` delegate to the state, so the optimizer and its
`TreeTensorNetwork` speak the same canonicalisation vocabulary.
`TreeOptimizer.sync_canonicalization(center=None)` is the explicit recovery
path after lower-level code directly mutates or canonicalizes `opt.tn`; it
rebuilds the state-owned centre before replay resumes. Post-run diagnostics
should normally use `opt.copy()` so the gate-evolution state is not touched.

Direct state users can call `TreeTensorNetwork.compress(max_bond=..., cutoff=...,
center=...)`. This performs one native leaf-to-centre SVD sweep over every
TreePlan edge and leaves the requested node as the validated canonical centre.
The canonicalization phase is QR-only; `max_bond` and `cutoff` control only the
compression phase.

Local isometry orientation also has one owner: each live Quimb tensor carries
its proven `left_inds`, while `TreeTensorNetwork.isometry_direction(node)` and
`isometry_map()` derive read-only node-to-neighbour views from those tensors.
`can_skip_canonize(a, b)` exposes the exact local condition used to avoid an
already-proven QR, and `validate_isometry_metadata()` checks the local
orientations against the tracked canonical region. `TreeOptimizer` delegates
the same four methods without maintaining another mutable map. Native
fermionic edges use this shortcut only when Symmray reports a fermionic array
with aligned charge maps and a complete `left_inds` proof; otherwise they
retain the explicit graded QR path.

`TreeTensorNetwork.validate()` checks the live tensor set, physical legs, tree
edges, and bond ownership against the `TreePlan`; pass
`check_canonical=True` when the metadata alignment and more expensive numerical
isometry check are also desired.
Direct Quimb mutations such as `gate_inds_`, `canonize_between`,
`compress_between`, and `canonize_around_` invalidate the tracked canonical
region. Call `invalidate_canonical_form()` after mutating tensor data directly;
it also invalidates the native fermionic norm cache. The optimizer's
state-aware wrappers do both automatically and restore the centre only for
operations that prove canonicality is preserved. `TreeOptimizer` also exposes
`sync_canonicalization(center=None)` to explicitly rebuild a single tracked
centre before replay continues.

Native fermionic trees use a separate graded edge path. Centre moves explicitly
QR-split the Symmray tensor and absorb the native carry into the next node;
edge compression uses a reduced graded core whenever the destination endpoint
is already proven isometric: the active endpoint is QR-split first, and only
its `R` factor is sent to the truncating native block SVD. If that proof is
absent, both endpoints are QR-reduced and their contracted core is SVD'd;
only an unrecognised reduction hint forms the complete two-node tensor.
Dense and nonfermionic trees continue to use Quimb's generic
`canonize_between` / `compress_between` wrappers. A graded exterior is not
assumed to be an ordinary Frobenius identity for readout: a known native
fermionic centre uses a one-tensor `TensorNetwork.H` contraction (which applies
the required outer-leg phase flips), while an unknown centre falls back to an
exact complete doubled-network contraction.

## Native fermionic QR stability

Native Symmray tree routes use Pepsy's internal
`TreeTensorNetwork._native_qr_split` policy for every lossless QR gauge move,
including two-qubit path threading, edge canonicalization, lossless path
splits, and sub-MPO message routing. The corresponding network-level subtree
canonicalization uses the same policy through `_native_qr_options()`.

For native block-sparse tensors, the policy passes `stabilized=False` to
Quimb's QR split. Symmray's stabilized QR phase-normalizes each diagonal of
`R`; symmetry can make a diagonal an exact structural zero, so the phase
`0 / |0|` can produce a NaN in `complex64`. Plain QR avoids that undefined
phase while preserving the exact factorization (`Q @ R`) and the tensor's
`left_inds` isometry metadata. This is a gauge choice, not a truncation or a
change to the represented state, and native `complex64` trees therefore do not
need to be promoted to `complex128` as a workaround for this issue.

The safeguard is tensor-aware: dense TTNs retain Quimb's ordinary stabilized
QR convention. It is internal to the tree implementation, so callers do not
need to pass a QR flag. Native truncating compression continues to use the
graded block SVD and the configured `chi`, `cutoff`, and `cutoff_mode`. This
policy is specific to `TreeTensorNetwork` / `TreeOptimizer`; the separate MPS
optimizer implementation is unchanged.

## Native central-edge compression and profiling

For a compression from active endpoint `A` to destination endpoint `B`, the
one-sided native path is valid only when `B` is structurally proven isometric
toward `A` (`can_skip_canonize(A, B, absorb="left")`). It uses

```text
A = Q_A R_A
R_A = U S V†
new_A = Q_A U
new_B = (S V†) B
```

The first QR is lossless and uses `_native_qr_split`; the second factorization
is the only truncating SVD. This avoids SVD'ing the fully fused `A B` tensor,
which can be thousands by thousands at moderate `chi` even when the live
edge is small. The implementation keeps the reduction hint separate from the
destination tensor, uses fresh intermediate bond labels while the old edge is
still present in `R_A`, and restores the original live edge label after the
factors are contracted. `reduced=True` uses the analogous two-sided QR/core
reduction. A positive cutoff never turns this into metadata-only compression.

On the calibrated 6x6 χ=64 complex64 Torch-CPU run (12 threads, 48 gates),
Tree evolution improved from 127.77 s to 6.21 s; MPS took 2.85 s in the same
post-fix run. Thus this fix removes the pathological Tree kernel, but Tree is
still about 2.18x slower for this prefix. The saved profile showed 276 edge
compression events totaling 2.40 s inside 48 update envelopes totaling 6.18 s;
it identified the gate update/threading/contraction path, especially the
central edges, rather than route-length tuning, as the next target.

The update path carries the isometry proof produced by the lossless threading
sweep into the reverse compression sweep, so native edges do not revalidate
the same `left_inds`/charge-map proof at every central edge. Two-site factors
use state-owned unique work labels rather than per-factor UUID allocation;
live routed bonds remain collision-safe across copied states without random
label setup in the hot loop. Native Torch-CPU one-edge contractions use
Symmray's blockwise mode, while CUDA and other backends retain the fused mode.
In multi-site Tree/MPO updates, independent QR messages
landing at the same node are contracted as one batch; dense waves reuse their
worker pool, while native fermionic routing remains serial for Symmray safety.
These changes preserve the complete-gate-before-truncation rule and the
configured χ/cutoff semantics. On repeated warm-cache runs of the same
harness, Tree evolution was 4.90--5.15 s; absolute timings vary with BLAS
thread state, so profile envelopes remain the authoritative comparison.

The remaining two-site update bookkeeping is also shared across arbitrary
gate streams: immutable qubit-support geodesics are cached and re-oriented
against the live centre for each gate, while ordinary one-edge tensor merges
use a direct backend ``tensordot``. Symmray dispatches through its graded
fermionic contraction implementation; unusual hyperedges still use Quimb's
general contraction path. This removes repeated path construction and
contraction-expression setup without changing the routed QR/SVD sequence.

Native complex64 QR also applies a reversible power-of-two scale separately
to small Symmray charge blocks. This avoids a Torch QR failure on finite,
rank-deficient blocks around ``1e-9`` without changing ``Q @ R`` or promoting
the replay to complex128. Native ``reduced="right"`` now has the matching
one-sided endpoint-SVD path; unproven ``right``, ``False``, and ``lazy`` modes
continue to use the conservative complete SVD.

The exact 6x6 ``nsteps=0`` stream (468 gates, χ=64, complex64 Torch CPU,
12 threads, ``track_truncation=False``) subsequently completed without the
previous gate-235 NaN. In one profiled run, MPS evolution took 20.39 s and
Tree evolution 137.11 s; Tree layout planning was a separate 25.79 s, giving
an evolution ratio of 6.72x. This confirms stability, not parity: the remaining
Tree cost is concentrated in the per-gate update envelopes and needs further
backend/profile-guided reduction. The same run's normalized Tree/MPS state
fidelity was 0.590; at χ=64 this measures different truncation histories on
the two geometries, not a QR gauge error. Compare observables or increase χ
when using this number as an accuracy diagnostic.

### QR/hop and bond-growth diagnostics

Construct `TreeOptimizer(..., profile=True)` to split the update envelope into
timed `thread_hop`, `edge_canonize`, and `edge_compress` events. The profile
also records `gate_factorization`, `tensor_absorption`, `center_movement`,
`metadata_path`, and `subtree_hub_merge` phases when those routes are used. The
`thread_hop` events are the exact, lossless QR carry moves; `edge_compress`
events are the truncating SVD work. These timings are nested inside the
per-update envelope and should not be added as independent wall-clock totals;
use `profile_report()["update_seconds"]` as the envelope total. The
`timing_semantics` field records this relationship explicitly. For asynchronous
CuPy or CUDA work, `profile_sync=True` synchronizes the active device at each
phase boundary so phase durations represent device execution; this is a
diagnostic mode and adds synchronization overhead.
For native Symmray compression, `profile_report()` also returns
`native_compression_routes`: counts of `one_sided_left`, `one_sided_right`, and
`two_sided_reduced` show that the graded reduced-core paths are active, while
`full_svd_fallback` identifies a conservative complete two-node SVD. Route
records have zero duration; use the enclosing `edge_compress` event for timing.

For a dimension-level report, also pass
`track_bond_diagnostics=True`. `bond_diagnostic_report()` then records
the per-update `live_max_bond_before`, `transient_max_bond` during
routing/factorization, and `live_max_bond_after` after the compression sweep.
The former may exceed `chi` by the gate's
operator-Schmidt rank; the latter is the enforced live-state cap. The extra
live maximum scans are opt-in so ordinary replay retains its default cost.

For deterministic small-system fidelity checks, use `norm()` for the
canonical local norm and compare `to_dense()` with an independently replayed
NumPy statevector using a fixed gate stream. This avoids making a restricted
Cotengra overlap path the correctness oracle. A Tree/MPS overlap remains a
useful comparative accuracy diagnostic, but it includes both layouts'
different truncation histories.

## Range / subtree canonicalisation

The single orthogonality centre generalises to a connected **canonical region**
-- the tree analogue of an MPS mixed-canonical range. `canonical_region` is a
frozenset of node ids tracked on the `TreeTensorNetwork` alongside (in fact,
underlying) `orthogonality_center`, which is simply the one-node special case:
when the region spans more than one node `orthogonality_center` honestly reads
`None`. `TreeTensorNetwork.canonize_subtree_(nodes)` gauges every tensor
*outside* a connected subtree to point inward (Quimb `canonize_around` with
`which="any"`), so the whole state norm is carried by the region tensors --
contracting just the region against its graded conjugate reproduces the squared norm,
exactly as the single centre tensor does for a one-node region. Disconnected
`nodes` raise unless `span=True` auto-expands to the minimal connected subtree
that spans them (`subtree_span`). `canonize_around_qubits_(qubits)` is the
qubit-level entry point: it canonicalises around the minimal subtree spanning
those qubits' physical nodes, so the reduced state on a set of qubits is captured by one
subtree. `is_subtree_canonical_form(nodes)` verifies the outside-is-isometric
property directly; `is_canonical_form` is its one-node case. `TreeOptimizer`
mirrors this too: `canonical_region`, `canonize_subtree(nodes, span=...)`,
`canonize_around_qubits(qubits)`, and `is_subtree_canonical_form(nodes)` all
delegate to the state.

## Multi-qubit / sub-MPO application

`apply_subtree_operator(op, where, *, max_bond=None, cutoff=None,
renormalize=False, track_norm=True)`
applies a general operator on `k >= 1` qubits as a single object, the one-shot
generalisation of the two-qubit gate: a `k`-qubit gate, a multi-site
**non-unitary / Kraus** operator, or a whole **Trotter block**. It is the tree
analogue of a sub-MPO applied over the covering range and then compressed (cf.
Quimb's `MatrixProductState.gate_with_submpo`, which exists for the 1D chain
only). The dense operator is first factorized into an exact tree-MPO on the
**minimal connected subtree** (Steiner subtree) spanning the target physical
nodes.
Application then proceeds recursively from subtree leaves to a hub: each local
state/operator message is losslessly QR-split on one edge and absorbed by its
parent, carrying every still-open operator virtual leg. No dense state tensor
for the whole Steiner subtree is formed. Each routed Q tensor, including native
Symmray graded Q factors, retains its `left_inds` isometry metadata when
available, so canonical recovery recognizes that it already points toward the
hub instead of repeating the same QR. The native predicate additionally
validates charge-map alignment before skipping. Once all MPO factors have
arrived, every
touched edge is compressed once. A bond that remains within its configured
`max_bond` uses a lossless QR, avoiding repeated cutoff loss of tiny state
components across successive sub-MPO events; when the MPO expands an edge
past its cap, the configured Quimb `cutoff` and `cutoff_mode` are applied to
the truncating SVD. Thus every actual truncation sees the complete operator in
an isometric environment.

`op` acts on `len(where)` qubits: an array reshaped to `(2,) * 2k` with output
indices first, `op[o_0..o_{k-1}, i_0..i_{k-1}]` (a `(2**k, 2**k)` matrix is
accepted). It need **not** be unitary; pass `renormalize=True` to renormalise
afterwards (e.g. after a Kraus/projection operator). `max_bond` / `cutoff`
default to the optimizer's `chi` / `cutoff`. The native TreeMPO route scales
with the operator's spread and factor ranks, using recursive edge messages
rather than one dense state tensor for the whole spanning subtree.

`track_norm=True` records the cheap path-level retained-norm ledger. Set it to
`False` for a known non-unitary operator: its physical norm change is then not
misreported as compression loss. The same keyword is accepted by
`apply_gate`, `apply_1q`, `apply_2q`, `apply_submpo`, and `apply_pauli_sum`.

An explicit MPS-style sub-MPO marker, `("submpo", mpo, where)` (or the
equivalent mapping form), is accepted in a TreeOptimizer stream. Quimb MPOs are
applied natively by carrying their virtual operator bonds through the same
lossless leaf-to-hub QR sweep followed by one subtree compression sweep; bond
estimates use MPO bond dimensions as a conservative Schmidt-rank bound.
Payloads without the required site interface fall back to `mpo.to_dense()`,
which must produce an operator on the declared support.
`TreeOptimizer.submpo_event(...)` builds the tuple form.

For a complete operator already represented as a `TreeMPO`, use
`apply_subtreempo(tree_operator, where=None, ...)` or the aliases
`apply_sub_tree_mpo` / `apply_subttno`. This contracts the operator's internal
TTNO bonds directly on the TreePlan, routes the resulting messages by the
TreePlan geometry, and performs one final configured Tree compression sweep.
It never extracts a contiguous chain MPO. The operator must use the same plan,
contain one primary network, and either declare all physical sites or declare
exactly its `operator_support` metadata when that known non-identity support is
available. Bond-one identity factors outside a term's active support are still
part of the complete TTNO; the shorter declaration only selects the minimal
Steiner route. Any omitted boundary operator bond must be bond one, otherwise
the application raises instead of silently discarding operator information.
The stream constructor `TreeOptimizer.subtreempo_event(tree_operator)` and
the matching `TreeStabOptimizer.subtreempo_event(...)` provide the same
native route; `subttno_event` is an accepted spelling alias. Set
`track_norm=False` for a general non-unitary TreeMPO so its physical norm
change is not recorded as compression loss.

The internal Pauli rotation and Pauli-sum constructors have a separate compact
support form for Tree evolution. A sparse operator on qubits such as
`(q0, q7)` is represented by MPO tensors only at `q0` and `q7`; identity-only
sites between them are not inserted into a fictitious chain window. The native
Tree MPO router therefore receives the true active support and computes the
minimal Steiner subtree/geodesic before QR routing and compression. The shared
constructors retain their contiguous-window default for the one-dimensional
MPS backend, whose compression domain is a chain interval.

The two explicit two-site families preserve native Symmray gates and their
block-sparse fermionic grading. Ordinary `apply_gate` entries in
`auto`/`direct`/`dm`/`sdc`/`src`/`mpo` are lowered to a true TreeMPO and
contracted through `apply_subtreempo` on the active canonical Steiner region.
`tree_mpo_direct` and `tree_mpo_dm` are explicit names for that same route,
selecting direct SVD or density-matrix compression. The `submpo` mode remains
the explicit chain-MPO stream mode. The low-level `apply_1q`/`apply_2q`
compatibility methods retain their specialized kernels when called directly.

For an ordinary gate stream, any of `mode="direct"`, `mode="dm"`,
`mode="sdc"`, `mode="src"`, or `mode="mpo"` now uses the TreeMPO path;
`mode="tree_mpo"` is an alias for `tree_mpo_direct`, hyphenated names are
accepted, and `tree_mpo_dem` is kept as a compatibility spelling for
`tree_mpo_dm`. The combined `tree_mpo_*` names own their compression suffix,
so a conflicting `compression_mode` is rejected.

`run(mode=...)` has the same persistent semantics as `MpsOptimizer`: it updates
the optimizer's selected gate route and compression method for that run, later
runs, and copies.
The old `run(mode="tree")`/`"ttn"` selector is a deprecated no-op retained only
for shared frontends.

For the ordinary TreeMPO route, the decomposition used for state truncation
can be selected independently with `compression_mode="direct"` (SVD) or
`compression_mode="dm"` (Quimb's density-matrix-equivalent `svd:eig`
decomposition on the local fused state/operator compression core).
The latter does not form a global dense state and is currently restricted to
dense tree tensors; native fermionic trees retain their graded direct
compression path. `mode="dm"` is the automatic-routing shorthand for
`mode="auto", compression_mode="dm"`.
The combined `tree_mpo_direct` and `tree_mpo_dm` names select both the true
TreeMPO route and its compression method, so a conflicting explicit
`compression_mode` is rejected.

`mode="zipup"` uses the same `apply_subtreempo` entry point, but contracts
one layered operator/state node only when its incoming child messages are
ready. An SVD immediately caps the outgoing state leg at `chi` before the
message reaches its parent. This avoids constructing the fully enlarged
tree before truncation. The unvisited environment is not canonical, so
zipup's intermediate discarded weights are not global error estimates; its
accuracy can differ from `direct` at the same `chi`. It uses direct SVD,
rejects a conflicting `compression_mode`, and leaves a canonical hub.
Dense NumPy/Torch and native fermionic arrays retain their backend and dtype.
Provably lossless zero-cutoff messages use the shared QR policy instead of
SVD. Bond-size history is recorded, but zipup does not report intermediate
discarded spectra as canonical truncation errors, even with
`track_truncation=True`; those error fields remain unavailable.
On a native fermionic tree, an overly small cap can remove every compatible
charge path. Zipup raises before installing such an empty state; increase
`chi` or choose `direct`, whose cuts see the complete operator environment.

`compression_mode="src"` contracts product-noise sketches of complementary
branches, caching a directed environment on each tree edge. A second sweep
forms QR projectors using those environments and the original layered target,
then propagates the projected target toward the hub. `compression_seed=...`
makes the sketches reproducible. As in Quimb SRC, the sample count is set by
`chi`; nonzero `cutoff` is ignored with a warning. With `chi=None`, the sample
count uses the largest original cut dimension to retain the full range.

`compression_mode="sdc"` uses the same successive projection structure with
deterministic low-rank complementary environments. Their factors are computed
using direct truncated SVD, avoiding squared conditioning and a NumPy
complex64 JIT failure in the installed Quimb eigendecomposition driver.
`cutoff`, `cutoff_mode`, and `chi` control those environment factors.

These are distinct environment algorithms, not randomized local SVD or aliases
of `direct`. On a path they reproduce Quimb's SRC/SDC sweeps (SRC comparisons
use identical sketches). On a branching tree, each retained node incorporates
its already projected children and a cached complementary environment.
They never materialize the complete operator-applied tree before compression.
The final hub is canonical and retains the projected target norm.

Both environment methods currently require dense arrays. Native symmetry
trees reject them explicitly; use native `direct` or `zipup`. Edge records
report dimensions, but do not invent discarded spectra or global error bounds
from these approximate environments. TreeFIT guesses use these algorithms;
its subsequent local refinement uses direct SVD. A Cholesky-based compressor
is not implemented by either mode.

### Tree-native FIT / DMRG

`pepsy.fitting.TreeFIT` is the cached local variational fitting kernel for a
`TreeTensorNetwork`. It has the same separation of target, disposable initial
guess, bounded local updates, ownership controls, and diagnostics as
`pepsy.FIT`, but replaces the chain's left/right environments with one cached
directed overlap message for each tree edge:

```python
from pepsy.fitting import TreeFIT

fit = TreeFIT(target, guess, max_bond=32, cutoffs=1e-12)
fit.run_gate(
    active_nodes,
    n_iter=4,
    block_size=2,       # 1, 2, or 3 connected tree nodes
    sweep_sequence="inward-outward",
)
updated = fit.p
report = fit.fit_diagnostics(overlap=True)
```

Before each local solve, the kernel moves the orthogonality centre along the
unique tree path. Only messages whose branch intersects a changed local block
or centre path are invalidated, so untouched branches retain their cached
entanglement environments. Each missing directed message contracts only its
node's target tensors, fitted bra tensor, and incoming neighbor messages.
An iterative postorder traversal avoids recursive calls on deep trees.
Invalidation follows those dependencies outward; there is no table of full
branch-node sets. Temporary messages carry indices and data without
accumulating branch-wide tags. `dmrg`, `dmrg1`, `dmrg2`, and `dmrg3` select this
engine in `TreeOptimizer`; `TreePepsOptimizer` accepts the same names. Generic
`dmrg` uses `fit_block_size` (two by default) and its configured adaptive
warm-up. `dmrg1` and `dmrg2` use two-node warm-up blocks, while `dmrg3` uses
three-node warm-up blocks followed by a two-node transition and one-node
refinement. The default `fit_n_iter=4` permits eight directional passes and
reaches refinement: `(2, 2, 1, 1)` for `dmrg2`, `(3, 3, 2, 1)` for `dmrg3`.
Tree warm-up still counts complete iterations, so this is not an identical
sequence of local updates to eight MPS sweeps. Explicit smaller budgets
remain honored. `fit_two_site_transition_sweeps=1` controls the `dmrg3`
transition within that budget; zero restores its previous three-to-one schedule.
Every block-size change resets the tolerance history, and tolerance stopping
cannot skip a pending transition or refinement phase.

`fit_sweep_sequence="inward-outward"` is the TreeOptimizer default. Use
`"outward-inward"` to reverse the order. Each iteration includes both passes,
ordered relative to the active region's medial node, which need not be the
whole tree's root. `RL`/`INOUT` and `LR`/`OUTIN` remain compatible aliases
for the two orders. Standalone TreeFIT uses the same names and default;
diagnostics report the normalized `sweep_sequence`.

FIT reuses each traversal order within a run. Before a block update, it moves
the canonical center only as far as needed to make the exterior isometric;
an existing center inside the block needs no preparatory QR. Local
factorization still establishes the requested final center, including explicit
endpoint centers for three-node `TreeFIT.fit_block` updates.

`fit_traversal="depth-first"` groups updates by branch to reduce canonical
center travel and environment invalidation. It visits the same connected
blocks as the default `"depth"` ordering; inward reverses outward. The medial
node anchors an iterative depth-first walk, and a multi-node block is ordered
by its node nearest that hub. This is opt-in because update order can change
finite-bond fidelity and convergence. Standalone TreeFIT calls this option
`traversal`.

For native Symmray states, `fit_environment_strategy="native-blockwise"`
uses graded blockwise contractions for FIT messages and effective tensors,
avoiding repeated charge-block fusion/unfusion. It retains Quimb/Cotengra
contraction planning, backend/device, and fermionic phases. The default is
`"default"`; the alternative requires native target/state arrays and checks
actual upstream API support. No global dispatch is changed. Standalone
TreeFIT calls this option `environment_strategy`. Performance depends on the
sector sizes and backend; blockwise is not universally faster.

`fit_single_node_fast_path=True` is automatic for a truly one-node active
region. It skips the compressed/random guess and performs one local
projection, preserving the parent RNG sequence. Diagnostics report one
iteration, `block_size_trace=(1,)`, `guess_used=False`, and
`convergence_reason="single_node_exact"`. This remains exact for a local
nonunitary gate and records its resulting norm; it does not depend on
`fit_rtol` or `fit_min_iter`. Set the flag to `False` to restore repeated
sweeps. A multi-node region using one-node updates still needs iteration.
For standalone `TreeFIT.run_gate`, `single_node_fast_path=True` solves the
local least-squares problem with its exterior held fixed; it does not promise
global fidelity one for an arbitrary target. Complete-tree `run`/`run_eff`
retain fixed iteration defaults with `single_node_fast_path=False`.

Small CPU measurements and compatibility probes are recorded in the
[FIT execution review](../../development/notes/tree_fit_execution.md).

The optimizer options `fit_n_iter`, `fit_adaptive_sweeps`,
`fit_two_site_transition_sweeps`, `fit_min_iter`, `fit_rtol`,
`fit_patience`, `fit_sweep_sequence`, `fit_init_strategy`,
`fit_init_rand_strength`, and `fit_init_seed` are forwarded to TreeFIT.
`fit_init_strategy="auto"` is the default: it selects `guess-src` for dense
trees and `guess-direct` for native fermionic trees, whose randomized SRC
compression is unsupported. This preserves the previous dense numerical
policy. Explicit unsupported native `guess-src`/`guess-dm` requests still
raise; `auto` does not change the requested output compression method.
`fit_init_strategy="direct"` keeps the current state as the initial guess;
`"guess-direct"`, `"guess-src"`, `"guess-sdc"`, `"guess-zipup"`, and `"guess-dm"` use a disposable compressed
warm start; `"random"` perturbs only active tensors and `"random_expand"`
also grows active bonds towards the exact target rank, capped by `chi`.
Randomized guesses remain deterministic for a fixed seed. Dense TreeFIT
updates preserve canonical metadata and the represented exponent; no implicit
normalization is performed on a non-unitary target.
`guess-direct` applies the operator to a private copy and compresses it;
it is distinct from the unchanged current-state `direct` initialization.
Native even-parity fermionic projections apply the graded metric correction
on dual open environment legs before local factorization. This preserves an
already representable state at each local update, including one-node refinement.
Odd-parity fermionic FIT remains explicitly unsupported.

Disposable compressed guesses copy the state once through the ordinary state
handoff. They do not clone the parent optimizer's queued gates, replay history,
or diagnostic records. They retain the previous single child-seed draw from
the parent RNG, preserving later measurement sequences; randomized
compression still uses `fit_init_seed`. Public `TreeOptimizer.copy()` preserves independent
histories, replay configuration, and a derived child RNG; it also preserves the
configured `fit_adaptive_sweeps`.

TreeOptimizer defaults to `fit_rtol="auto"` and `fit_min_iter=2`:

| State precision | `cutoff="auto"` | `fit_rtol="auto"` |
| --- | --- | --- |
| 16-bit | `1e-3` | `1e-3` |
| float32 / complex64 | `1e-6` | `1e-5` |
| float64 / complex128 | `1e-12` | `1e-9` |

The tolerances resolve from the installed state's dtype at construction,
matching MpsOptimizer's numerical policy. `cutoff_mode="auto"` retains
`"rsum2"`. Explicit numbers remain unchanged, and `fit_rtol=None` disables
tolerance stopping. `fit_patience=1` means one stable comparison of two
same-phase norm samples; the counter resets when the block size changes.
`fit_n_iter` remains the iteration budget, and rank-growth/warm-up conditions
still gate early stopping. This criterion measures relative change in the
retained canonical-center norm, not a bound on the global state error.

As with MpsOptimizer's non-unitary policy, automatic tolerance stopping is
disabled during `run(non_unitary=True)`. It is also disabled for updates with
`track_norm=False`, whose target norm is not assumed known. Explicit numeric
`fit_rtol` remains honored in those cases. Optimizer FIT diagnostics report
both `fit_rtol_requested` and the effective `fit_rtol` for each update.
Standalone TreeFIT retains `rtol=None` and its fixed-block defaults. All three
entry points accept `two_site_transition_sweeps=0`; set this to one with
`block_size=3, adaptive_block_sweeps=2` to request the new transition explicitly.

TreeFIT accepts `retag=True` for structural node-tag alignment and
`copy_target=False` for an explicitly disposable target. Its target may be a
fused tree network or a correctly tagged layered tree network. Every target
tensor must belong to exactly one structural node group; local layer bonds
stay inside a group, and one or more inter-group bonds must follow the fitted
tree edges. Ambiguous or untagged layer tensors are rejected rather than
dropped. The separate two-layer path compressor remains the `TreePeps`
`sdc`/`src`/`zipup` route when that direct Quimb path is desired.

`TreeOptimizer`'s DMRG target is built as a layered operator--state network:
the state and TreeMPO virtual bonds are not fused, and only corresponding
physical input/output legs are connected. This keeps the DMRG target aligned
with the layered FIT representation while leaving the direct TreeMPO route
unchanged.

All ordinary DMRG gate entries now use `apply_subtreempo` too. The optimizer
transfers ownership of its disposable layered target to TreeFIT with
`copy_target=False`. `fit_finite_check=False` is the default; enable it to
check active tensor entries once per iteration. `run(finite_check=True)` also
checks active FIT tensors and scans the final state in every replay mode,
including empty streams. This setting is scoped to that replay and inherited
by shot workers unless `run_kwargs` overrides it. Enabled replay warns once
about the optional diagnostic cost; nested FIT calls share that warning.
Standalone TreeFIT uses `finite_check=False` and warns when scans are enabled.
Scalar convergence, scale bookkeeping, and explicit overlap diagnostics remain
independent; the existing scale/zero guards retain their semantics.
Trusted local updates no longer recompute every
outside isometry after each sweep; explicit `state.validate(check_canonical=True)`
remains available. The existing `track_infidelity` default is unchanged.

TreeFIT's `local_fidelity` is the clipped squared ratio of the retained
terminal canonical-centre norm to the fixed target norm, matching MPS FIT's
local norm diagnostic. `local_norm_trace` contains one such centre readout per
completed sweep and `local_norm_stripped_trace` preserves its mantissa and
base-ten exponent. It is distinct from the optimizer's `norm_diagnostics()`
local and cumulative retained-norm proxy, which is computed from
canonical-centre norm ratios and stored in logarithmic form to avoid
underflow. For lazy targets, pass the known exact `target_norm` (a norm or
mantissa/exponent pair) to obtain this normalized ratio without additional
target work. TreeOptimizer supplies the pre-update canonical norm when
`track_norm=True`, the unitary-update contract. For non-unitary updates use
`track_norm=False`; an unknown lazy target norm yields `local_fidelity=None`
while the retained norm and convergence trace remain available.
Routine fitting never contracts `target.norm()` or a doubled target network.
`fit_diagnostics(overlap=True)` separately requests a genuine directional
overlap. If the target norm is unknown, this explicit diagnostic uses a
lossless leaf-to-hub QR pass and one hub norm, never `<target|target>`.
Its directional overlap is reported as `target_fidelity`,
with MPS-compatible `fit_overlap_fidelity` aliases, not as `local_fidelity`.
Thus an MPO/TreeMPO identity normalization scale is not silently treated as
compression fidelity.
Each completed local fit also carries the target's stored exponent into the
fitted state, preserving its represented scale when the guess has a different
exponent.
Optional norm or overlap diagnostic failures are reported in
`fit_overlap_error`; they do not invalidate or prevent installation of a
successful fit.
TreeFIT rejects odd-parity fermionic tensors: its local projection does not
yet preserve their graded signs. Use the native `direct` or `zipup` routes
for those states. This restriction applies to DMRG modes as well.

`TreeOptimizer` accepts Quimb's `cutoff_mode` conventions for every truncating
Tree-edge SVD. Its defaults, `cutoff="auto"` and `cutoff_mode="auto"`, resolve
once at construction from the installed state dtype, using the shared MPS
policy: `1e-3` for 16-bit data, `1e-6` for float32/complex64, and `1e-12` for
float64/complex128. An explicit numeric cutoff, including zero, is preserved.

`cutoff_mode="auto"` (or the compatibility spelling `None`) selects `"rsum2"`
for every tree split, including `mode="dm"`, `mode="tree_mpo_dm"`, and
`compression_mode="dm"`. This matches MPS DM's **numerical criterion**:
tree DM's `svd:eig` kernel truncates singular values `s`, while MPS MPO DM
truncates density-matrix eigenvalues `s**2` with its native `"rsum1"` mode.
Both automatic rules bound the relative discarded squared weight
`sum(s_discarded**2) / sum(s**2)` when the bond cap does not force more loss.
Explicit modes pass through unchanged: tree `"rsum1"` instead bounds
`sum(s_discarded) / sum(s)`, and `"rel"` uses a relative largest-singular-value
threshold. Copies preserve the resolved cutoff and mode.

The lower-level `TreeTensorNetwork.compress_edge_` API retains explicit
numeric defaults (`1e-10` and `"rsum2"`); pass its cutoff controls directly
when using that lower-level interface.

### Performance-oriented defaults and warnings

The default replay configuration is intended for production evolution:
`mode="auto"` uses the true TreeMPO route on every ordinary gate support,
while `threads=1` avoids
oversubscribing the small tree contractions, `subtree_workers=1` keeps the
serial path allocation-free, `profile=False` avoids timing overhead, and
`track_truncation=False` avoids full-spectrum diagnostic SVDs, while
`track_bond_diagnostics=False` avoids live-bond scans. `record_history` and
`track_infidelity` retain the established API defaults; the latter enables the
cheap canonical-centre norm ledger and its progress-bar readout. It does not
enable spectrum probes.

Warnings are reserved for an actionable behavior change: enabling
`track_truncation=True` emits one diagnostic-performance warning, while
legacy mode selectors emit deprecation warnings. Explicit user stream
backend/device mismatches and incompatible dtypes are errors, not implicit
conversions. Dense and
native paths share the direct one-edge contraction, path-cache, routing, and
proof-reuse optimizations. Only the QR phase safeguard and graded reduced-core
SVD are native Symmray specializations; dense arrays continue through Quimb's
ordinary QR/SVD with the same cutoff, path, and truncation semantics.

`TreeOptimizer.apply_submpo(..., track_norm=True)` is the public form for an explicit MPO of
arbitrary support. It losslessly QR-routes its virtual bonds, then uses its
supplied (or configured) `max_bond` / `cutoff` in one final canonical sweep over
the affected subtree. Existing bonds at or below `max_bond` take a lossless
QR; only bonds expanded past the cap invoke the configured cutoff mode.
The tree backend also exposes numerical Pauli primitives used by a future
stabilizer frontend: `apply_pauli_rotation(...)`, `apply_pauli_sum(...)`,
`expectation_pauli(...)`, `measure_pauli(...)`, and `project_pauli(...)`. These
operate on dense two-level qubit coefficient states and do not require tableau
metadata; they intentionally reject native fermionic TTNs.
`measure_pauli` returns `(outcome, probability)` and accepts an optional
`return_diagnostics=True` flag. `project_pauli` normalizes by default; pass
`renormalize=False` to retain the branch norm. Both APIs can report projection
diagnostics containing the norm ratio and support, spanning-tree, and bond
snapshots before and after the update. The records are also available through
`projection_diagnostics` and `get_projection_diagnostics()`.

## Coefficient-backend feature boundary

`TreeOptimizer` covers the state operations shared with `MpsOptimizer`:
ordinary one-/two-/multi-qubit gates, structured sub-MPOs, Pauli expectation
and projection, measurement, reset, measure-reset, cap, normalization,
copying, canonicalization, layout construction, dense readout, and truncation
diagnostics for dense two-level qubit TTNs. A cap's `absorb` argument is
accepted for stream compatibility. A leaf site absorbs into its unique parent;
a root site is contracted directly without changing the tree edges.
`cap(q, vec)` compacts labels by default; use
`stable_labels=True` (or `compact_labels=False`) to preserve caller-facing
logical IDs across the cap while the internal TTN stays compact.
`TreeOptimizer.qubits`, `logical_order`, `position`, and `logical_site` expose
that mapping. Native fermionic TTNs support native Symmray gates and MPOs, but
the qubit Pauli/measurement/reset helpers intentionally reject them; use the
fermion model's native observable/projector with
`TreeTensorNetwork.local_expectation` instead of silently treating a graded
local space as a qubit.

The shared trajectory runner supports dense-qubit `TreeOptimizer` instances as
well. Independent trajectories can sample Pauli mixtures, depolarizing
channels, and state-dependent Kraus channels; branch probabilities are
evaluated from copied TTNs and selected branches are normalized before replay
continues. Coalesced trajectory replay supports exact branching of mid-circuit
measurement, reset, and measure-reset events through `expectation_pauli`. Use
matrix-valued gate payloads in tree streams (for example `pepsy.h()`), since
textual MPS gate aliases are not normalized by the tree gate parser. Native
fermionic trajectories may use native gates/MPOs, but Pauli/control events
require a model-native observable or projector.

MPS execution modes such as `svd`, `mpo`, `swap`, `perm`, `su`, and `mix` are
chain algorithms and are intentionally not copied into `TreeOptimizer`.
Tree-native `dmrg`/`dmrg1`/`dmrg2`/`dmrg3` are provided by `TreeFIT` instead.
Tree layout is part of the TTN geometry and is selected with
`tree=`/`layout=` at construction. `TreeStabOptimizer` uses
`tree_mpo_direct` or `tree_mpo_dm` for its numerical coefficient updates,
delegating the active-span TreeMPO contraction to this class while keeping
tableau state and stabilizer-specific bookkeeping above it. See the
[TreeStabOptimizer API](tree_stabilizer.md) for the supported fixed-basis,
basis-updating, immediate, and deferred magic-injection
Clifford/rotation/measurement paths, bounded dense matrix dispatch, and the
safe MPS naming-compatibility surface.

`TreeOptimizer.run` also exposes the shared shot and MPI entry point. It
creates independent copies from the current tree state, so the parent state
and queued stream remain unchanged:

```python
result = optimizer.run(
    shots=1_000_000,
    seed=7,
    mpi=True,
    workers="auto",
    progress="auto",
    retain="none",
)
```

The `strategy="independent"` and `strategy="coalesced"` options follow the
shared runner semantics. `TreeStabOptimizer.run` provides the corresponding
API and intentionally supports independent shot distribution only.

## Tree state class

`TreeTensorNetwork` is the tree analogue of Quimb's `MatrixProductState`: a
geometry-owning subclass of Quimb's arbitrary-geometry vector class
`quimb.tensor.TensorNetworkGenVector`. It *is* a Quimb tensor network, so all of
Quimb's arbitrary-geometry methods (`canonize_around`, `canonize_between`,
`compress_between`, `gate_inds`, `to_dense`, `copy`, ...) apply directly; the
class adds the naming and geometry glue on top of a
`TreePlan`:

- every node (leaf **and** internal) is one tensor tagged with the structural
  node tag `node_tag_id.format(nid)` (default `"N{}"`);
- leaf tensors additionally carry the Quimb site tag `site_tag_id.format(q)`
  (default `"I{}"`) and physical index `site_ind_id.format(q)` (default `"k{}"`)
  for qubit `q`; when `plan.root_qubit` is set, the root tensor carries that
  qubit's site tag and physical index as well;
- adjacent nodes share one live virtual bond. Newly constructed edges use the
  deterministic `_tb{lo}_{hi}` name, but Quimb may replace it with a UUID during
  threading or canonicalisation; `TreeTensorNetwork.bond(a, b)` resolves the
  current live index.

Because the geometry (`plan`) and naming live in `_EXTRA_PROPS`, they survive
`.copy()` and every Quimb view, exactly like `site_ind_id` does for an MPS.
Build one with `TreeTensorNetwork.from_plan(plan)` (product `|0...0>`),
`TreeTensorNetwork.from_order(order, structure=...)` (build the plan and the
product state in one step; its default exposes the ternary virtual root), or
`TreeTensorNetwork.rand(plan, D=..., seed=...)`
(a random state, canonicalised around the root by default). `TreeOptimizer`
builds and evolves its state on this class, delegating all node/qubit naming and
geometry queries to it.

`TreeTensorNetwork.local_expectation(op, where, max_bond=None)` has two
backend-specific exact paths. Dense/nonfermionic TTNs move the centre to the
target physical node/subtree, cancel the ordinary isometric exterior, and contract only
the minimal Steiner subtree. Native fermionic TTNs insert the Symmray operator
without densifying it and contract the complete doubled tree, preserving every
graded boundary phase. For native fermionic states, `max_bond` is accepted for
API compatibility but cannot truncate this exact doubled-network contraction.
Observable readout deliberately belongs to the state, not to `TreeOptimizer`;
use `optimizer.tn.local_expectation(...)`.

Readout is gauge-preserving: a dense expectation restores the previously
tracked canonical centre/region, while an unknown dense gauge is evaluated on a
temporary state copy. Native fermionic expectations do not move the gauge.
The repeated normalized native-readout denominator is cached and invalidated by
state mutation, copying, caps, and canonical/gate updates.

`TreeTensorNetwork.local_expectations(terms, optimize=..., normalized=True)`
evaluates many observables at once, where `terms` maps each `where` (an int
site or a tuple of sites) to its operator. It delegates each term to
`local_expectation` with a *shared* `optimize` handle, so a reusable
`pepsy.build_optimizer(...)` caches one contraction path per topology, and it
reuses the memoized graded norm across the batch. Each returned value matches
the corresponding single-term call exactly. For a Hamiltonian-level energy
readout or variational energy optimization, `pepsy.TreeEnergyOptimizer` wraps
this batch path, returns an `EnergyEstimate` mirroring `MpsEnergyOptimizer`,
and exposes the corresponding `make_tn_optimizer` / `optimize` methods.
Optimization updates the tree tensor parameters with Quimb's autodiff
`TNOptimizer` while retaining the exact tree local-expectation objective. For
ordinary readout the native graded norm is memoized. The optimization loss
uses a fresh full doubled-tree denominator because Quimb's direct parameter
injection cannot invalidate that cache; the post-optimization state is marked
non-canonical rather than recanonicalized around an arbitrary centre.

For the package-level product-state constructor, matching `ps_to_mps`, use
`pepsy.ps_to_ttn(n, theta=..., tree=...)`. It builds the requested tree,
initialises every physical site with `[cos(theta), sin(theta)]`, and optionally
expands the virtual bonds with `chi`. Pass `root_qubit=q` to build the plan
directly, or supply a matching root-site `TreePlan` through `tree=`.

For a native Symmray fermionic state, pass a `Fermion` model and occupations:
`pepsy.ps_to_ttn(n, tree=plan, fermion=fermion, occupations=..., chi=1)`.
Physical sites then carry the model's charge/parity sectors, virtual-only
internal nodes are neutral, and every tree edge uses conjugate Symmray virtual
indices.
The constructor selects a definite local Fock basis vector, not a random vector
inside a degenerate charge sector. For spinful `U1`/`Z2`, a scalar occupation
`1` selects the checkerboard `|up>, |down>, ...` representative; pass
`(n_up, n_down)` occupations to choose each spin explicitly. The completed
graded product tree is normalized by an exact graded norm contraction, so its
represented norm is one rather than an arbitrary constructor scalar.
`pepsy.hrs_to_ttn(..., chi=...)` creates the corresponding random symmetric
tree with the requested charge-sector bond dimension and accepts the same
`root_qubit=`, `max_arity=`, and `top_arity=` options. These constructors keep
the Symmray arrays native; they do not materialize dense tensor data.

`pepsy.TreeSampler(state)` samples every registered physical site, including
the optional root site. Its cached canonical arrays use parent, physical, then
child axes, so probabilities and amplitudes retain normal `q0..q(n-1)` order.

`TreeTensorNetwork.show()` prints a top-down ASCII drawing of the tree -- the
tree analogue of a quimb MPS `show()` -- with the root at the top, structural
leaves at the bottom, physical sites labelled by qubit, and every branch
annotated with its current virtual bond dimension
(`ascii_tree()` returns the same drawing as a string).
`TreeOptimizer.show()` delegates to it.

## Tree structure

The tree structure is chosen by `TreeLayoutFinder`, which builds a weighted
interaction graph from the two-qubit supports of the gate stream and applies
recursive spectral (Fiedler) partition, keeping the recursion as the rooted
tree (`structure="quality"`). This reuses the interaction-graph and spectral
machinery of `pepsy.optimizers.mps.layout`; where the MPS finder flattens the
recursion into a 1D order, the tree finder keeps the tree. Strongly coupled
qubits become nearby physical nodes, minimising the tree-path length that
two-qubit gates thread across. With `root_qubit=q`, that physical node stays
fixed at the top while the finder searches over the remaining leaf sites.
`structure="balanced"` splits the leaf-qubit order in half at each level.
`TreeLayoutFinder.score(plan)` returns the total interaction-weighted tree-path
length that the structure minimises.

For circuits with gates of different operator-Schmidt ranks, use
`TreeLayoutFinder(..., objective="congestion")` or
`TreeOptimizer(..., layout_objective="congestion")`. This evaluates interaction,
congestion-aware, and balanced candidates using the predicted log bond growth
on every edge. A gate crossing an edge contributes `log2(k)`, where `k` is its
operator-Schmidt rank across that edge; the maximum edge load therefore
predicts the worst-case multiplicative bond growth. `TreeOptimizer` uses
`layout_objective="congestion"` by default because it is a better
execution-oriented choice at finite `chi`; a bare `TreeLayoutFinder` retains
`objective="path"` as its fast, backward-compatible default. The path
objective remains the co-occurrence/path-length heuristic.
`objective="hybrid"` is useful when both replay cost and bond pressure matter:
it combines normalized path score, maximum edge load, and total edge load with
`hybrid_weights=(path, max_edge_load, total_edge_load)`. The
`weight_mode` / `layout_weight_mode` option accepts `count`, `auto`, `angle`, or
`operator_schmidt` for interaction-graph weighting.

For a genuinely multi-site layout objective, use
`objective="hypergraph"` (or `layout_objective="hypergraph"`). Each original
gate support is kept as one hyperedge, and the finder scores its actual
operator-Schmidt load on every crossed tree edge rather than selecting from a
pairwise proxy alone. This mode starts from inexpensive pairwise-derived seed
trees, then automatically performs bounded direct greedy leaf swaps and binary
NNI topology moves using the full hyperedge score. Pass
`refine=None, topology_refine=None` to inspect the unrefined direct score, or
set explicit budgets for a larger search. Dense operators wider than
`max_operator_qubits` still use the documented conservative rank bound.

For a whole-tree optimization, use `objective="full_tree"` (also accepted as
`"tree"` or `"cotengra"`). This evaluates dynamic operator-Schmidt demand,
cap overflow, working tensor width, estimated work/write volume, and route
length across every hierarchical scale, not only the root cut. Finite-`chi`
overflow and edge demand are ranked before tensor-work proxies, since avoiding
unnecessary truncation is the primary execution concern. It enables bounded
subtree reconfiguration and simulated annealing by default; override these
with `topology_refine="subtree"`, `topology_budget=`, `search="anneal"`, and
`search_budget=`. The result is still a cheap layout proxy rather than a real
TTN replay, so the state-aware pilot remains the final accuracy check. The
default `chi=None` leaves this as a static, chi-blind objective; supplying
`chi` only adds cap-aware ranking and does not change the no-tensor nature of
layout discovery.

For the 6×6 periodic square-lattice calibration stream (Hadamards followed by
periodic controlled-phase gates), predicted total overflow ranked binary,
ternary, and four-way candidates in the same order as actual capped replay
pressure and truncation counts. This validates the profile as a layout-ranking
proxy; use the state-aware pilot when the circuit has strong cancellations or
state-dependent rank loss.

Use `order="quality"` with `finder.run()` (or set it on the finder) for the
MPS-style high-quality offline search. Quality mode now means
`objective="full_tree"`: it evaluates every hierarchy scale, enables bounded
greedy leaf refinement and all-scale subtree topology refinement, and runs a
hybrid topology-annealing/Nevergrad search when Nevergrad is available. It
falls back explicitly to dependency-free simulated annealing otherwise. A
finder without `order="quality"` keeps the fast deterministic zero-argument
`run()` path.
Disable stages explicitly with `refine=None`, `topology_refine=None`, or
`search=None`, or bound them with `refine_budget=`, `topology_budget=`, and
`search_budget=`.

For a stream whose locality changes over time, pass `time_decay=` and/or
`time_window=` to `TreeLayoutFinder`. A decay in `(0, 1]` weights an event by
`time_decay ** age` (the newest event has age zero), while a window keeps only
the final events. The same factors are used for interaction paths, congestion
candidate construction, and per-edge operator-Schmidt load estimates, so the
diagnostics and selected plan use one consistent time model. The defaults are
unchanged. `TreeOptimizer` exposes these as `layout_time_decay=` and
`layout_time_window=`.

For compression-first selection, use `objective="compression"` (or
`layout_objective="compression"`). It prioritizes peak and total predicted
operator-Schmidt load, then penalizes the estimated local tensor size at the
configured `chi`, and only then uses path length. This differs from the
default fast `"path"` objective: path optimizes routing locality, while
compression also accounts for bond pressure and wider-node cost.

`weight_mode="operator_schmidt"` is a cheap **two-qubit entangling-strength
proxy** used to form the spectral qubit order; it is not itself the exact
operator-Schmidt rank. Use `objective="congestion"` when selecting a tree: its
edge-load calculation uses the actual rank across each candidate tree cut (or
an MPO bond bound), which is the quantity that predicts TTN bond growth.

Rank diagnostics are explicit. Small dense qubit operators use exact
operator-Schmidt ranks; opaque native arrays, MPO bond products, and supports
larger than `max_operator_qubits` use conservative operator-space bounds and
are counted in `rank_bounded_events` with a reason in `rank_bound_reasons`.
They are not silently assigned rank two.

The structure is **not restricted to binary trees**. Internal nodes may have
any arity, controlled by two knobs on `TreeLayoutFinder` / `TreePlan.from_order`
/ `TreeOptimizer`:

- `max_arity` caps the children per internal node. It accepts a scalar (a
  single fixed tree: `2` reproduces the strictly-binary tree exactly, larger
  values give flatter `k`-ary trees with shorter geodesics, `None` leaves the
  arity unbounded) or an iterable of candidate arities to **search**. The
  default `2` selects the fixed binary tree; pass an iterable such as
  `(2, 3, 4)` to search candidate arities explicitly.
- `structure="adaptive"` reads the gate-stream interaction graph and lets each
  level branch into as many children as it has strongly coupled communities
  (edges above `community_frac` times the level's strongest edge). A densely
  coupled block -- a near-clique with a present-strong-edge fraction of at
  least `star_frac` -- is collapsed into a single flat **star** node, so all
  its pairwise geodesics are length two instead of the up-to-`log2 m` of a
  bisection. Binary trees remain a valid special case (`max_arity=2`).

A caller may bypass the finder entirely by passing an explicit `TreePlan` via
`TreeOptimizer(..., tree=plan)`. `TreePlan` is exported from both `pepsy` and
`pepsy.optimizers.tree`. Build one with
`TreePlan.from_order(order, weights=..., structure=..., max_arity=..., top_arity=...)`, or -- for
a fully hand-specified arbitrary-arity tree -- with
`TreePlan.from_children(children, qubit_of_leaf)`, which validates that the
children map and leaf assignment describe a single rooted tree covering qubits
`0..n-1` exactly once. Set `top_arity=3` with `max_arity=2` for the
three-virtual-bond root convention described above. `TreePlan.max_arity()` and
`TreePlan.is_binary()` report the shape; `TreePlan.is_strictly_binary()` is the
strict two-child-at-every-internal-node predicate.

`TreeLayoutFinder` also provides the same regular-lattice baseline vocabulary
as `OneDMap`. Pass `lattice_shape=(Lx, Ly)` or
`lattice_shape=(Lx, Ly, Lz)` once, then use named `order` presets for exact
tree coarsenings of the corresponding leaf traversal:

```python
finder = TreeLayoutFinder(
    gates,
    n=36,
    lattice_shape=(6, 6),
    max_arity=2,
    top_arity=3,
)

tree_row = finder.run(order="row-major")
tree_snake = finder.run(order="snake")
tree_folded = finder.run(order="folded-snake")
tree_hilbert = finder.run(order="hilbert")
tree_coarse = finder.run(order="coarse-alternate-x")
tree_quality = finder.run(order="quality")

# Equivalent one-string spelling for the tree geometry:
tree_coarse = TreeLayoutFinder(
    gates,
    n=36,
    lattice_shape=(6, 6),
    map_mode="coarse-alternate-x",
).run()
```

The supported 2D geometric presets include `"row-major"`, `"snake"`,
`"alternate-x"`, `"alternate-y"`, `"folded-snake"`, and `"hilbert"`, plus
their `coarse-*` variants. In 3D, use `"row-major"`, `"col-major"`,
`"snake"`, `"snake-row-major"`, `"alternate-x"`, `"alternate-y"`, or
`"alternate-z"`, together with their supported `coarse-*` variants.
`alternate-x` and `alternate-y` snake within each xy layer and reverse the
layer direction along z; `alternate-z` makes z the alternating inner line.
The 3D folded-snake and Hilbert presets remain intentionally 2D-only because
`OneDMap` does not define a 3D version of those paths.

A coarse preset first partitions the lattice into blocks, follows the selected
base traversal on the coarse grid, and then expands each block back to its
physical qubits. The default `coarse_grain=(2, 1)` groups two neighboring x
sites in 2D and `(2, 1, 1)` does the same in 3D; pass `(1, 2)` or `(1, 2, 1)`
for y-oriented blocks, or `(1, 1, 2)` for z-oriented blocks. Edge blocks may
be smaller. Coarse modes change only the leaf traversal order and never merge
physical tensors. Preset orders use a balanced recursive tree so the leaf
sequence and every higher tree layer are preserved as contiguous intervals of
that traversal; `"quality"` remains the independent interaction-aware
TreeLayoutFinder search. The default logical label is
`x * Ly + y` in 2D and `x * Ly * Lz + y * Lz + z` in 3D; pass a
`lattice_site` callable when the gate stream uses a different convention.

For example, a 3D Tree layout can use the same handoff through
`TreeOptimizer`:

```python
finder3d = TreeLayoutFinder(
    gates,
    n=4 * 4 * 3,
    lattice_shape=(4, 4, 3),
    coarse_grain=(2, 2, 1),
)
tree3d = finder3d.run(order="coarse-alternate-z")
opt3d = TreeOptimizer(gates, tree=tree3d, chi=32)
```

The lower-level helper is useful when constructing a `TreePlan` directly:

```python
from pepsy.optimizers.tree import TreeLayoutFinder, TreePlan

zigzag = TreeLayoutFinder.lattice_order(6, 6, "coarse-alternate-x")
tree_plan = TreePlan.from_order(
    zigzag,
    structure="balanced",
    max_arity=2,
    top_arity=3,
    map_mode="coarse-alternate-x",
)

zigzag3d = TreeLayoutFinder.lattice_order(
    4, 4, 3, "coarse-alternate-z", grain=(1, 1, 2)
)
tree_plan3d = TreePlan.from_order(zigzag3d, structure="balanced")
```

Both layout finders still accept an explicit site permutation as `order` for a
custom fixed baseline. For example, this builds the same binary/ternary-root
geometry using a square-lattice snake order without refinement:

```python
zigzag = py.square_lattice_zigzag(6, 6)
tree_plan = TreeLayoutFinder(
    gates,
    n=36,
    max_arity=2,
    top_arity=3,
    lattice_shape=(6, 6),
    coarse_grain=(2, 1),
).run(order="coarse-alternate-x")
```

The explicit order must cover every site exactly once and cannot be combined
with iterable `max_arity` candidate search.

For an automatic arity search, call `finder.recommend_arities((2, 3, 4))`
explicitly. The default `TreeOptimizer(gate_stream, n=n, chi=chi)` uses the
fixed binary/ternary-root geometry; it does not allocate tensors or perform
truncations while finding the layout.
The result contains the recommended
`TreePlan` plus per-candidate path, edge-load, peak-bond-growth, and local
virtual-degree summaries. An explicit handoff looks like:

```python
finder = TreeLayoutFinder(gate_stream, n=n, objective="congestion")
choice = finder.recommend_arities((2, 3, 4))
opt = TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

The `path` and `congestion` objectives are `chi`-blind cost proxies: they score
geodesic length and additive edge load, so they can favour a wider block or
arity whose widest bond induces a qubit bipartition too large to fit `chi`.
Every bond splitting `k` of the `n` qubits from the rest can carry a Schmidt
rank up to `2 ** min(k, n - k)`, so `TreePlan.max_bond_cut()` is a purely
structural accuracy ceiling: the tree can hold an arbitrary state exactly only
when `chi >= 2 ** max_bond_cut`. Pass `chi=` to `recommend_layered` or
`recommend_arities` to make the search `chi`-aware -- candidates are ranked
first by `chi_overflow` (how far the widest bond exceeds `log2(chi)`), so a
structure that stays exact at `chi` is preferred and the layout objective only
breaks ties. Each candidate then also reports `max_bond_cut`, `chi_overflow`,
and `exact_at_chi`:

```python
choice = finder.recommend_layered((2, 3, 4), chi=chi)   # prefers a chi-exact block
opt = TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

For a fixed layered family, prefer an explicit recommendation to a hard-coded
block size:

```python
finder = TreeLayoutFinder(
    gates,
    n=L,
    objective="congestion",
    weight_mode="operator_schmidt",
    chi=chi,
)
choice = finder.recommend_layered(block_sizes=(2, 3, 4))
tree_plan = choice["plan"]
```

`layered(block_size=4)` remains the right API when the block size is an
intentional experimental control: it spectral-orders the qubits and builds
exactly that fixed structure, but it does not score alternatives or use `chi`.
`recommend_layered()` inherits the finder's `chi` when its own `chi` argument
is omitted; pass `chi=None` explicitly for a chi-blind comparison. Inspect its
per-candidate `max_edge_load`, `peak_bond_growth`, `max_bond_cut`, and
`chi_overflow` rather than relying only on the chosen block size.

### Fixed-plan refinement and Nevergrad search

The tree topology is fixed before `TreeOptimizer` begins replay. Moving the
canonical centre and threading a gate along a path are tensor operations, not
layout rewrites. For a stronger *pre-simulation* layout search,
`recommend_layered` and `recommend_arities` can refine each candidate through
adjacent leaf-label swaps. This preserves every parent/child edge in the plan:
only which qubit label occupies each leaf changes.

```python
finder = TreeLayoutFinder(
    gate_stream,
    n=n,
    objective="hybrid",
    hybrid_weights=(1.0, 1.0, 0.25),
    chi=chi,
)
choice = finder.recommend_layered(
    block_sizes=(2, 3, 4),
    refine="greedy",
    refine_budget=64,
)
tree_plan = choice["plan"]
```

`refine="greedy"` is deterministic and bounded; it is opt-in for the existing
fast/default objectives. The explicit `objective="hypergraph"` mode enables
greedy and NNI refinement by default because otherwise its full-support score
would only rank a few pairwise-derived seed trees. A balanced TTN turns a
well-aligned physical span `r` into a path with `O(log r)` tree hops, so the
hybrid score uses path length as a replay-cost proxy while edge loads estimate
the accuracy/bond-dimension cost.

For offline quality searches, `search="nevergrad"` starts from the
spectral/greedy plan, proposes leaf orders, and keeps its result only when it
improves the same objective. `search="anneal"` explores subtree replacements
at multiple scales without optional dependencies. For the highest-quality
`objective="full_tree"` search, use `search="hybrid"`: it splits the bounded
budget between topology annealing and Nevergrad leaf refinement. Quality mode
selects this hybrid automatically when Nevergrad is installed and falls back
to annealing otherwise. None of these stages allocates or replays a TTN.
Install the optional dependency with `pip install pepsy[layout]`:

```python
choice = finder.recommend_layered(
    block_sizes=(2, 3, 4),
    refine="greedy",
    search="nevergrad",
    search_budget=128,
    seed=0,
)
```

The same fixed-plan quality controls can be supplied directly to `run()`,
which gives finder-based frontends the same define-then-search shape as the MPS
layout API:

```python
tree_plan = finder.run(
    order="quality",
    topology_refine="nni",
    topology_budget=64,
    refine="greedy",
    refine_budget=64,
    search="hybrid",
    search_budget=128,
    seed=0,
    nevergrad_optimizer="OnePlusOne",
    progbar=True,
)
```

Omitted `run()` options inherit the finder configuration, preserving the
zero-argument API. Structure, arity candidates, objective, and event weighting
remain finder-construction options because they define the Tree search space
and scoring model rather than one refinement pass.

Nevergrad evaluates every candidate plan, so reserve it for offline circuit
studies rather than routine short simulations. Candidate records expose their
initial/final leaf order and the greedy/Nevergrad diagnostics under
`candidate["planning"]`.

When an explicit iterable of arity candidates is supplied, a `chi` biases the
search toward `chi`-exact structures. The default fixed binary/ternary-root
geometry is independent of `chi`; layout finding still does not allocate
tensors or perform truncations. A bare finder with no `chi` is likewise
static unless candidate search is explicitly requested.
Set `max_operator_qubits` to bound dense rank diagnostics and operator
allocation; wider native MPO events can still replay without dense
materialization. `TreeLayoutFinder(..., max_operator_qubits=...)` uses a
conservative rank bound above that width and reports the bounded events. The
public `score` remains the path score for compatibility; inspect
`objective_key`, `max_edge_load`, `peak_bond_growth`, and the tensor-cost
fields for compression decisions. `report(plan, include_edge_loads=False)`
skips the event-by-edge calculation for path-only diagnostics; when loads are
included, `peak_bond_growth_log2` remains finite even when the human-readable
`peak_bond_growth` would overflow floating point.

For a state-aware choice between static candidates, call:

```python
choice = opt.select_layout_for_compression(
    pilot_candidates=4,
    pilot_steps=64,
)
opt = py.TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

The pilot replays candidates on independent copies with the real tree update
kernels and returns measured infidelity, final bond, truncation count, and
runtime under `choice["pilot"]`. By default one bounded `order="quality"`
candidate (greedy leaf refinement plus topology refinement; for
`full_tree`, this also includes bounded subtree/hybrid search) is
reserved a pilot slot, so it cannot be rejected before state-aware replay. Use
`include_quality=False` for the static-only candidate set. The original
optimizer is unchanged unless `install=True` is passed. Installation is
restricted to product initial states; an entangled TTN cannot generally be
relaid out exactly.

For the recommended closed-loop choice, use the high-level optimizer helper:

```python
choice = opt.optimize_layout(
    objective="full_tree",
    rounds=2,
    pilot_candidates=4,
    pilot_steps=64,
    pilot_workers=2,
    topology_budget=32,
    search_budget=64,
)
```

This keeps the finder tensor-free, pilots the selected candidates with the
actual tree replay kernels, and uses each round's per-edge truncation loss,
discarded weight, and update runtime to seed bounded NNI, subtree, and
cross-cut leaf proposals for the next round. `choice["pilot"]["rounds"]`
contains every round and `report["edge_diagnostics"]` identifies hot tree
edges. `objective="full_tree"` combines all-scale static work/bond estimates
with this short state-aware replay. `install=True` remounts the product state
on the final plan; it remains rejected for an entangled state.

Independent product-state pilots can be evaluated concurrently with
pilot_workers greater than one; the default is one for minimal overhead and
deterministic resource use. Candidate order and tie-breaking remain
deterministic.

Both helpers are also available from the package-level API:

```python
import pepsy as py

finder = py.TreeLayoutFinder(gate_stream, n=n, objective="congestion")
opt = py.TreeOptimizer(gate_stream, layout=finder, chi=chi)
```

To evolve a non-product or entangled initial state, pass it explicitly as
`state=` (or the backward-compatible `tn=`):

```python
opt = TreeOptimizer(gate_stream, layout=finder, state=initial_ttn, chi=chi)
```

`tree=` / `layout=` accept only a `TreePlan` or `TreeLayoutFinder`; passing a
`TreeTensorNetwork` there raises an error so an entangled state cannot be
silently replaced by the default `|0...0⟩` product state.

### Initial-state layout handoff and array backends

`TreeLayoutFinder` is deliberately circuit-only: it consumes gate supports and
weights to choose a plan, never an already-entangled coefficient state. Pass
the resulting finder/plan *and* a state separately to `TreeOptimizer`.
An entangled `TreeTensorNetwork` must already own that same plan. Supplying a
different `tree=` or `layout=` raises an error before any tensor is changed:
there is no generally exact, cheap relayout of an entangled TTN, and silently
compressing it would hide a fidelity loss.

Product states are the safe exception. A `TreeTensorNetwork` with
`max_bond() == 1` is rebuilt exactly on the requested plan (and emits a warning
that the requested layout replaced its old geometry). A bond-one Quimb
`MatrixProductState` is likewise accepted and mounted exactly on the selected
tree, so a caller may choose the tree layout after preparing an MPS product
state. Entangled MPS inputs are rejected rather than implicitly converted.

The live TTN has one array contract: every tensor must have the same backend,
dtype, and device. `backend_info()` reports that contract. Construct the full
initial state and every user-supplied gate/operator with the same converter.
Every user gate and every tensor in every queued sub-MPO/TreeMPO must match the
state backend and device. Non-NumPy payloads must also match dtype, while
NumPy-to-NumPy dtype promotion is compatible. A mismatch raises `TypeError` at construction,
`set_gates`, `add_gates`, or replacement-state installation, before replay or
tensor work. Prepare a payload explicitly with `opt.to_backend(payload)` (or
the same converter used to build the state). Internal Pauli/projector tensors
follow the state backend automatically.
Mixed-backend initial states fail immediately because there is no unambiguous
safe execution backend.

The same diagnostic is reflected by the state-derived `backend`,
`backend_dtype`, `backend_device`, and `array_backend` attributes. The
complete gate stream is checked once at its boundary, including every tensor in
each sub-MPO; replay uses the accepted payload objects without a second scan or
implicit transfer. Native Symmray states report `backend="symmray"` plus
the underlying NumPy, Torch, or CuPy `array_backend`, preserving U1/U1U1 charge
and fermionic metadata.

```python
import pepsy as py
import torch

to_backend = py.backend_torch(device="cuda", dtype=torch.complex128)
finder = py.TreeLayoutFinder(gates, n=L, weight_mode="operator_schmidt")
plan = finder.layered(block_size=4)

state = py.TreeTensorNetwork.from_plan(plan)
state.apply_to_arrays(to_backend)  # backend-only conversion preserves left_inds

# Convert user-provided gate arrays once, at their source.
native_gates = [(to_backend(gate), where) for gate, where in gates]
opt = py.TreeOptimizer(native_gates, layout=plan, state=state, chi=chi)
assert opt.backend_info()["backend"] == "torch"
```

The same rule applies to CuPy; choose `py.backend_cupy(...)` and convert every
state tensor and payload with that converter. `to_dense()` intentionally returns
a host NumPy vector for interoperability; the live state remains on its native
backend.

## Diagnostics

The dominant lever for accuracy at fixed `chi` is the tree structure, so the
finder and optimizer expose diagnostics to choose it:

The same diagnostics are available as a Cotengra-style tent plot.
`TreeLayoutFinder.plot(plan)` is the default tent view (also available as
`plot_tent(plan)`): it keeps the raw graph at the bottom and lifts
the selected hierarchy above its descendant sites: the raw lattice and
optional gate connectivity are gray, while internal tree nodes use a stable order-based
`turbo` palette by default. Incoming edges match their child nodes by default;
pass an explicit `edge_color` for a uniform structural color. Arrows are
disabled by default, matching Cotengra's structural tent view; pass
`show_edge_arrows=True` only when parent-to-child direction is needed.
When the physical background already has its own markers, pass
`show_leaf_nodes=False` to hide the tree's physical leaf circles while keeping
internal tree nodes and hierarchy edges visible. This is useful for a gray
`+`-marked lattice backdrop.
When `site_coords` are supplied, the default tent presentation projects them
with `lattice_skew=0.30` and `lattice_rise=0.18`, and draws gray `+` markers at
the physical sites. Override those values to use a different base projection.
Nearest-neighbor gate edges are not duplicated over the lattice. Supplying
`site_coords={qubit: (x, y)}` places the physical sites on an existing lattice.
It returns `(fig, ax)` and does not mutate the plan or live TTN:

```python
finder = py.TreeLayoutFinder(gates, n=n, objective="congestion")
plan = finder.run()
fig, ax = finder.plot(
    plan,
    site_coords=logical_lattice_coords,
    color_by="scale",
    edge_color=None,
    edge_cmap="GnBu",
    node_cmap="YlOrRd",
    order=True,
    show_edge_arrows=False,
)

# For a live optimizer, the same plot is available without changing its state.
fig, ax = opt.plot_layout(site_coords=logical_lattice_coords)
```

To make the first hierarchy layer easier to see, give the leaf-to-parent
edges a contrasting color while leaving higher layers scale-colored:

```python
fig, ax = finder.plot_tent(
    plan,
    site_coords=logical_lattice_coords,
    color_by="scale",
    leaf_edge_color="#2563eb",
)
```

The default plot is therefore the hierarchy that `TreeLayoutFinder` selected:
one hierarchy edge per parent-child connection, drawn over the physical lattice
and gate connectivity. For a background-free binary check, hide the physical
background with:

```python
fig, ax = finder.plot(
    plan,
    lattice=False,
    show_gate_connectivity=False,
)
```

This leaves only the selected hierarchy. The public tent plot intentionally
does not draw gate-by-gate route overlays.

Use `finder.plot_rubberband(...)` for the same hierarchy in physical-lattice
rubberband form. The optional `viz`
profile provides Matplotlib. The plot uses the same visual
idea as Cotengra's circuit/rubberband views: the source interaction structure
remains visible underneath, and band color can encode either tree scale or
post-order. The default styling is axis-free, following Quimb's schematic
drawings, and the
background lattice is not numbered. Pass `show_axes=True` or
`show_site_labels=True` when those annotations are wanted. A stream-order
colorbar is hidden by default; pass `colorbar=True` when that diagnostic is
wanted.

For a scale-invariant tree view, use `color_by="scale"`:

```python
fig, ax = finder.plot(
    plan,
    site_coords=logical_lattice_coords,
    color_by="scale",
    edge_color=None,
    edge_cmap="GnBu",
    node_cmap="YlOrRd",
)
```

Here leaves are scale zero and nodes use stable colors for their hierarchical
scale. With `edge_color=None`, each incoming hierarchy edge uses the exact
same node-palette color as the node it terminates at, so scale layers remain
easy to follow. Set a literal `edge_color` when a uniform structural edge
color is preferred. The scale colorbar, when enabled, follows `node_cmap`.
Midpoint arrows can show the direction from each parent to its children when
`show_edge_arrows=True`. The mapping is independent of the number or order of gates;
`colorbar=True` then labels tree scale rather than gate-stream order. The plot
has no title by default; pass `show_title=True` if a title is wanted.

For a physical-lattice view closer to Quimb's rubberband drawing, use:

```python
fig, ax = finder.plot_rubberband(
    plan,
    site_coords=logical_lattice_coords,
    color_by="gate",
)

# The live optimizer exposes the same non-mutating view.
fig, ax = opt.plot_rubberband(site_coords=logical_lattice_coords)
```

This keeps the lattice sites and gate connectivity grey and wraps each
non-root tree cluster in a rounded, translucent band. The default is a
Cotengra-style `Spectral` post-order progression, giving each nested band a
distinct color. Use `color_by="scale"` for one stable band color per tree
scale.

- `TreeLayoutFinder.report(plan=None)` summarises the physical-node geodesic
  lengths over the interaction graph (`score`, `max_path`, `mean_path`,
  `weighted_mean_path`) and compares against a balanced index tree
  (`balanced_score`, `score_ratio_vs_balanced`). It also reports
  `edge_loads`, `max_edge_load`, and `peak_bond_growth` for the rank-aware
  congestion estimate.
- `TreeOptimizer.bond_report()` reports the current `max_bond`, `mean_bond`,
  and tensor/bond counts -- bonds pinned at `chi` mean truncation is active.
- `TreeOptimizer.estimate_bonds()` performs the paper's non-mutating dry run:
  it multiplies the operator-Schmidt ranks of gates crossing each tree edge,
  returning the conservative Eq. (4) bound before replay. This is useful for
  choosing `chi`; it deliberately ignores cancellations and can overestimate
  the live dimensions.
- `TreeOptimizer.preflight(...)` turns that bound into explicit resource
  protection. `max_bond`, `max_operator_qubits`, and `max_subtree_nodes` can
  reject a replay with `MemoryError`; the same limits can be passed to the
  constructor for automatic checking before eager replay. The constructor
  defaults to `max_operator_qubits=8` and `max_subtree_nodes=128`; pass `None`
  to disable either guard. Product-Pauli measurements use a factorized parity
  projector and do not materialize a `4**k` dense operator.
- `TreeOptimizer.truncation_report()` exposes the per-edge compression and
  SVD-split history, including before/after bond dimensions. Pass
  `track_truncation=True` to also collect each local full singular spectrum's
  absolute discarded weight and relative discarded fraction. Dense states use
  the global spectrum; native Symmray states compare the full and actually
  retained charge-block spectra using the same sector-aware truncation rule as
  the live update. Spectrum probes are opt-in because they add local SVD work
  per truncation edge. Enabling it emits a one-time warning because the
  diagnostic spectrum probes can add substantial SVD work. It remains
  disabled by default. Lossless zero-cutoff edges that are already within
  their bond cap use QR and do not probe a spectrum even when tracking is on.
  Per-edge retained survival and cumulative infidelity are accumulated in log
  space and exponentiated only for readout, avoiding product underflow on long
  streams.
  The report also contains gate-level `updates`, grouping
  edge events by support and reporting the cumulative relative loss.
- `TreeOptimizer.get_norm_events()` and `TreeOptimizer.norm_diagnostics()`
  expose the separate cheap path-level norm ledger. A Tree event groups the
  complete QR-thread/compression path for one gate or subtree update and
  reports `local_fidelity` plus the log-accumulated `cumulative_fidelity`.
  These are compression fidelities measured from retained norms, not target-
  state overlaps. They are collected independently of
  `track_truncation`; use the latter only when per-edge discarded weight and
  singular-spectrum attribution are required.
  In `norm_diagnostics()`, `norm`/`state_norm` are the live represented Tree
  norm, while `cumulative_norm` is the square-root retained-compression
  proxy. `norm_survival` is an explicit provenance alias for
  `cumulative_fidelity`.
- `TreeOptimizer.convergence_sweep(gates, n, chi_values, ops=...)` replays the
  stream at several `chi` on one fixed tree and returns per-`chi` `max_bond`,
  `norm`, observable `expectations`, `fidelity` against the untruncated state
  (when `2**n <= dense_cap`), and observable `max_drift` between consecutive
  `chi` -- a reference-free convergence signal for large systems. Optional
  observables are evaluated by the underlying `TreeTensorNetwork`; they are
  not a `TreeOptimizer` readout API.

## Readout

`to_dense()` returns the dense statevector in index order `k0, k1, ..., k(n-1)`.
`run(progbar=True)` shows a tqdm replay bar matching the MPS optimizer's core
compression readout: the active mode, exact two-qubit event count `2q`,
cumulative retained-norm fidelity `~F`, and current bond usage `bnd`.
Tree-specific `kq`, `ctrl`, and explicit-operator `mpo` counters are added only
when those event types occur. The bar is display-only and does not replace the
path-level norm ledger or recorded per-edge truncation history. The norm ledger
is the canonical `local_fidelity` / `cumulative_fidelity` diagnostic;
`track_truncation=True` is the more expensive spectrum-attribution diagnostic.
For dense two-level qubit TTNs, `measure(q, outcome=None)` projectively
measures a qubit in the computational basis and returns a bit; `reset(q)`
returns a qubit to `|0>`. Native fermionic TTNs deliberately do not expose
these qubit readouts.
For stream control events, `TreeOptimizer.measure_event`,
`cap_event`, `reset_event`, and `measure_reset_event` build the same tuple forms as
`MpsOptimizer`, including Pauli-basis measurement and reset. Their recorded
results are `(pauli, where, outcome, probability)` in `measurements`.
`cap(q, vec)` contracts and removes one physical site, shifting the remaining labels
above `q` down by one unless stable labels are requested.
For a non-unitary run, `normalize_every=True` (or `normalize_final=True`) keeps
the canonical working tensor numerically normalized and accumulates each
removed base-10 scale in `tn.exponent`; `norm()`, `to_dense()`, copies, and
full contractions continue to represent the original physical scale.
The normalization records expose both the per-event raw scale and the
accumulated exponent. The public `normalize()` method remains a physical
renormalization: it clears that exponent and rescales the represented state to
unit norm. `max_bond()` reports the largest virtual bond. Truncation details are
available through `truncation_report()`, `get_infidelities()`, and
`get_infidelity_samples()` when spectrum tracking is enabled.

## Performance and stability

- **Lossless QR fast paths.** Zero-cutoff splits and edge updates whose
  rank is already within the active bond cap use QR rather than SVD. This
  includes the sibling-leaf split and remains valid for native graded QR.
  A positive cutoff retains the existing rank-revealing compression semantics.
- **Repeated-gate cache.** Direct gate SVDs and MPO factorizations are cached
  by immutable payload identity, backend signature, support, and local
  dimensions. The bounded cache returns fresh-index tensor copies, so it does
  not share mutable network indices with the live state.
- **Subtree and pilot parallelism.** Set `subtree_workers>1` to evaluate
  independent dense leaf-to-hub QR messages in a wave, with deterministic
  merging. Native fermionic routing stays serial because graded Symmray phase
  bookkeeping has not been established as thread-safe. Set `pilot_workers>1`
  for independent layout pilot replays; both options default to one.
- **Sibling fast path.** A two-qubit gate on two leaves that share a parent is
  applied as a single two-site update: the two leaves and their parent are
  contracted into one blob, the gate is applied, and the blob is re-split by
  two truncating SVDs against the (isometric) surrounding tree. This avoids the
  QR bond-threading and double-bond fusion of the general geodesic route and is
  the common case in a locality-aware layout.

- **Thread cap.** Tree tensors are moderate-rank (set by local arity and the
  optional root physical leg, with dimensions bounded by `chi`), so
  multi-threaded BLAS/OpenMP linear algebra is dominated by thread launch and
  synchronisation overhead. `TreeOptimizer` caps threads to `1` around gate
  application and the heavy read-outs by default (`threads=1`), which makes
  replay both markedly faster and stable in wall-clock time; pass
  `threads=None` to leave the ambient thread count untouched (worthwhile only
  in a large-`chi` regime where a single contraction is itself large). Thread
  limiting uses `threadpoolctl` when available and is a no-op otherwise.
- **Lazy canonical centre.** A freshly built product state has every virtual
  bond at dimension 1, so it is already canonical with the root as
  orthogonality centre; `from_plan` records that centre on the network rather
  than recomputing it on the first gate. Native fermionic product trees are
  additionally normalized by their exact graded norm readout.
- **Routed isometry reuse.** Dense geodesic and subtree QR routing retains each
  Q tensor's `left_inds`, allowing later canonical recovery to reuse the proven
  isometry without repeating the decomposition or entering Quimb's dense
  canonicalization kernel. Final path and subtree compression also consults
  that live proof: when the destination-side tensor is already isometric,
  Quimb uses one-sided `reduced="left"` compression and avoids its redundant
  reduction QR; otherwise it falls back to the full two-sided reduction. The
  network derives orientation diagnostics from those tensors; the optimizer
  does not keep a duplicate map. Native fermionic trees keep their separate
  explicit graded QR/SVD path.
- **State-owned centre.** The orthogonality centre lives on the
  `TreeTensorNetwork` (`orthogonality_center`, an `_EXTRA_PROPS` field), so the
  optimizer and the state cannot disagree and the centre is carried by
  `.copy()`. Incremental moves (`shift_orthogonality_center`) touch only the
  geodesic between old and new centre.
- **Self-healing tid cache.** Node-to-tensor lookups are cached and validated
  against the live tensor map, so the hot path avoids re-scanning tags while
  staying correct when a gate rebuilds a tensor.
- **Resource guards.** `max_intermediate_bond`, `max_operator_qubits`, and
  `max_subtree_nodes` provide preflight and direct-application limits; the
  latter two default to conservative finite values and accept `None` to opt
  out. A dense `k`-qubit operator still has `4**k` payload values, while
  product-Pauli measurement uses a factorized parity projector and recursive
  edge messages.
- **`copy()`.** Returns an independent optimizer that shares the immutable
  `TreePlan` but owns its own tensor network (which carries the tracked
  orthogonality centre), for branching experiments or trial gate sequences.


> API details are maintained as handwritten Markdown in this page.
