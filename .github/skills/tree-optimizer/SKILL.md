---
name: tree-optimizer
description: 'Run, review, debug, benchmark, or extend pepsy.TreeOptimizer, the rooted TTN gate-stream circuit simulator. Use for tree layouts/plans, exact gate threading and canonical compression, product-state handoff, backend compatibility, TreeTensorNetwork local expectations, measurement/reset, noisy trajectories, diagnostics, sampling, and stream-wired control events. Not for MPS, MERA, stabilizer tensor networks, or BP.'
---

# Tree Optimizer in pepsy

Use this skill for `pepsy.TreeOptimizer` (also `pepsy.optimizers.TreeOptimizer`)
and its implementation under `src/pepsy/optimizers/tree/`:

- [`optimizer.py`](../../../src/pepsy/optimizers/tree/optimizer.py) -- `TreeOptimizer` (state + gate replay + readout).
- [`layout.py`](../../../src/pepsy/optimizers/tree/layout.py) -- `TreePlan` (pure rooted-tree structure, any arity) and `TreeLayoutFinder` (entanglement-adapted structure search).
- [`__init__.py`](../../../src/pepsy/optimizers/tree/__init__.py) -- subpackage exports.
- Docs: [`docs/api/optimizers/tree.md`](../../../docs/api/optimizers/tree.md).
- Tests: [`tests/test_optimize_tree.py`](../../../tests/test_optimize_tree.py).

Read the docs page and the closest tests before editing. It is a thin
tensor-network glue layer: the heavy lifting (canonicalisation, compression,
tensor splitting, tree path finding) uses `quimb` primitives.

## What it implements

The rooted tree-tensor-network circuit simulator of *Simulating quantum
circuits using tree tensor networks* (Seitz, Medina, Cruz, Huang, Mendl;
Quantum 7, 964, 2023; arXiv:2206.01000). The state is a rooted TTN (internal
nodes of **any arity**; binary is the default, see *Non-binary trees* below)
whose leaves carry physical qubit indices. An optional ``root_qubit`` is instead
carried by the top tensor; all other physical sites remain leaves. A bundled gate stream
`[(gate, where), ...]` is replayed. `where` is an `int` (1q) or a pair of `int`
(2q); supports with `len(where) >= 3` route through
`apply_subtree_operator` (see *Multi-qubit / sub-MPO application*).

Preferred public handoff:

```python
finder = TreeLayoutFinder(gate_stream, n=n, objective="congestion")
choice = finder.recommend_arities((2, 3, 4))
optimizer = TreeOptimizer(gate_stream, tree=choice["plan"], chi=chi)
```

Alternatively pass the finder itself with `layout=finder`. An initial
non-product or entangled state must be passed explicitly as `state=` (alias
`tn=`). `tree=` and `layout=` accept only `TreePlan` / `TreeLayoutFinder`; a
`TreeTensorNetwork` passed there must raise a clear error rather than being
silently replaced by the default `|0...0>` state.

### State/layout handoff and backend contract

`TreeLayoutFinder` is **circuit-only**: it accepts a gate stream or explicit
supports, never a `TreeTensorNetwork`. Passing a TTN to it must raise a clear
`TypeError`; construct the plan from the circuit, then pass the state separately
to `TreeOptimizer`.

- An entangled `TreeTensorNetwork` (`max_bond() > 1`) can be installed only on
  its matching `TreePlan`. A different `tree=` / `layout=` must raise before
  tensor work: implicit relayout would be lossy and hide a fidelity decision.
- A product `TreeTensorNetwork` (`max_bond() == 1`) may be remounted **exactly**
  on a requested plan; warn that its geometry changed. Preserve the product
  vectors and any distributed global scalar/phase.
- A bond-one Quimb `MatrixProductState` is also a geometry-neutral product input
  and may be mounted exactly on the selected tree. Reject an entangled MPS;
  converting it to a tree is an explicit caller-controlled operation.
- All live state tensors must use one backend, dtype, and device.
  `backend_info()` reports this. Reject a mixed state at construction or
  `set_tn`, rather than choosing an arbitrary execution backend.
- Callers must prepare every gate/operator/sub-MPO/TreeMPO payload with the
  same backend and device as the state. Non-NumPy payloads must also use the
  state dtype for direct contractions; dense NumPy-to-NumPy dtype promotion is
  compatible. The complete user gate stream is checked
  once at construction, `set_gates`, `add_gates`, or state replacement; every
  gate and every operator tensor is checked, and a mismatch raises a
  location-specific `TypeError` before replay. Replay does not rescan or cast
  accepted payloads. Use `TreeOptimizer.to_backend(...)` for explicit
  preparation. Internal Pauli, projector, reset, and generated TreeMPO helper
  tensors may still convert explicitly without warning.
- Backend-aware contractions must stay in Autoray/Quimb. Convert only scalar
  readouts intentionally (`to_float`); `to_dense()` intentionally returns a
  host NumPy vector for interoperability while the live TTN remains native.

## Tree state class (`TreeTensorNetwork`)

`pepsy.TreeTensorNetwork` (also `pepsy.optimizers.TreeTensorNetwork`, source
`src/pepsy/optimizers/tree/ttn.py`) is the tree analogue of quimb's
`MatrixProductState`: a geometry-owning subclass of
`quimb.tensor.TensorNetworkGenVector` (import from `quimb.tensor`, **not** the
deprecated `tensor_arbgeom`). It owns a `TreePlan` plus the node/site/index
naming, so all inherited quimb methods (`canonize_around`, `canonize_between`,
  `compress_between`, `gate_inds`, `to_dense`, `copy`) work directly. The
  state overrides `local_expectation` with a canonical Steiner-subtree
  contraction that also supports native Symmray observables.

- `_EXTRA_PROPS = ("_sites", "_site_tag_id", "_site_ind_id", "_plan",
  "_node_tag_id", "_canonical_region", "_symmetry", "_fermionic",
  "_physical_sectors")` -- these are copied on `.copy()`/
  every quimb view. The `__init__` copy-branch guard
  `if isinstance(ts, TensorNetwork): super().__init__(ts, **o); return` lets
  the base copy the extra props without the fresh-construction defaults
  clobbering `_plan`.
- Each physical-site tensor carries **both** the structural node tag `N{nid}`
  and the quimb site tag `I{q}` plus physical index `k{q}`. These tensors are
  structural leaves by default. When
  `plan.root_qubit` is set, the root carries that site tag/index too; other
  internal nodes carry only `N{nid}`. So quimb sees the leaf sites and optional
  root site as `nsites`, with remaining internal nodes as ancillary bond carriers --
  `ttn.local_expectation(G, where=[q], max_bond=None, optimize="auto")` uses
  the tree's canonical contraction for dense states and an exact complete
  doubled-tree contraction for native fermionic states.
- `node_tid(nid)` is a self-healing tid cache kept in `__dict__` (not
  `_EXTRA_PROPS`) so a copy starts with a fresh, independent cache.
- Builders: `from_plan(plan)` (product `|0...0>`), `from_order(order,
  structure=...)` (plan + product in one call), `rand(plan, D=, seed=,
  canonicalize=True)` (random state, canonicalised around the root).
- `show()` prints a top-down ASCII tree (root on top, structural leaves at the
  bottom, physical nodes labelled `q{q}`, each branch annotated with its bond dim);
  `ascii_tree()` returns that string. `TreeOptimizer.show()` delegates to it.
- `TreeOptimizer.tn` **is** a `TreeTensorNetwork`; the optimizer delegates
  `_phys->tn.site_ind`, `_tag->tn.node_tag`, `_tid->tn.node_tid`,
  `_neighbors->tn.neighbors`, `_steiner_nodes->tn.steiner_nodes`, and
  `_build_product_state->TreeTensorNetwork.from_plan`. Keep these names/values
  identical (`k{q}`, `N{nid}`, `_tb{lo}_{hi}`) so behaviour is unchanged.

## Conventions (must stay consistent across optimizer + layout)

- Node ids are ints from `TreePlan`. Tensor tag = `N{nid}` (`TreeTensorNetwork.
  node_tag`, via optimizer `_tag`).
- Physical index of qubit `q` = `k{q}` (`TreeTensorNetwork.site_ind`, via
  optimizer `_phys`) -- ket-leg convention. Physical nodes also carry site tag
  `I{q}`. Resolve their geometry with `plan.node_of_qubit[q]`; use
  `leaf_of_qubit` only when a true structural leaf is required.
- Newly created virtual bonds between adjacent nodes `u,v` use `_tb{lo}_{hi}`
  with `lo<hi` (`optimizer._bond_name`), but Quimb may mint UUIDs during
  threading or canonicalisation. `TreeTensorNetwork.bond(u, v)` resolves the
  live shared index; use it for diagnostics and readout.
- `plan.node_path(a, b)` is the inclusive node-id geodesic (unique in a tree);
  `plan.tree_distance(qa, qb)` is the physical-node path length.

## Canonical-centre contract (core invariant)

The orthogonality centre is a single node id **owned by the `TreeTensorNetwork`**,
and it is the one-node case of the more general **canonical region** (a connected
node set) tracked in `ttn._canonical_region` (declared in `_EXTRA_PROPS` so it
survives `.copy()` and quimb views). `ttn.orthogonality_center` is *derived* from
that region: the sole node when the region has size 1, else `None` (honest "no
single centre"). `TreeOptimizer.center` is a thin property view onto it, so the
optimizer and the state can never disagree. It is **algorithm state**, not
cosmetic: every tensor outside the region must be isometric pointing inward so it
telescopes to identity between bra and ket.

- `TreeTensorNetwork.shift_orthogonality_center(node)` is the primitive: it walks
  the geodesic from the current centre to `node` with `canonize_between(absorb=
  "right")` per edge (lossless QR), touching only the path tensors (O(path
  length), not O(N)); it is idempotent when already centred and falls back once
  to `canonize_around_node_` (O(N)) only when the centre and canonical region
  are both unknown. This is the tree analogue of quimb's MPS
  `shift_orthogonality_center` /
  `MpsOptimizer.info_c["cur_orthog"]`.
- `TreeOptimizer._move_center(target)` simply delegates to it.
- When the centre is unknown but `_canonical_region` contains multiple nodes,
  `shift_orthogonality_center` first peels that region with lossless QR and
  then walks only the remaining path. Do not regress this regional recovery to
  an unconditional O(N) recanonicalisation.
- Local isometry proofs live only on each tensor's ``left_inds``.
  `TreeTensorNetwork.isometry_direction` / `isometry_map` derive read-only
  orientations, `can_skip_canonize` recognizes an already-proven dense edge or
  a native Symmray edge with aligned charge maps, and
  `validate_isometry_metadata` checks alignment with the canonical region.
  `TreeOptimizer` delegates these methods; never add a second mutable
  optimizer-owned orientation map. Native fermionic edges fall back to
  explicit graded QR when the proof is absent or malformed; positive-cutoff or
  over-cap native compression still uses the explicit graded SVD.
- `ttn.is_canonical_form(center)` verifies the invariant directly (every
  non-centre tensor is an isometry toward the centre) — use it in tests/diagnostics.
- A freshly built product state is **already canonical at the root** (all
  virtual bonds are dim 1, so every tensor is trivially isometric). `from_plan`
  records `orthogonality_center = root`, so the first gate skips an O(N)
  canonicalisation. Do not reset this to `None`.
- `canonize_edge_` / `compress_edge_` advance the tracked centre by one hop when
  it starts on the isometric side, else set it to `None` (honest: a lone edge
  move cannot leave a global centre). The optimizer's hot paths call quimb
  `canonize_between` / `compress_between` **directly** and set `self.center`
  explicitly afterward — keep that.
- Unitary 1q gates preserve canonical form regardless of centre (absorbed into
  the leaf, no bond growth, no centre move). Non-unitary 1q operators
  (projectors in `measure`) keep the centre on that leaf but require a
  subsequent `normalize()`.
- `norm()` uses the single centre tensor for dense/nonfermionic states when
  `center is not None`; only their unknown-centre fallback contracts the full
  doubled tree. Native fermionic states use a one-tensor
  `TensorNetwork.H` contraction when a centre is known, so Symmray applies the
  graded outer-leg phase flips; unknown-centre fermionic states use the exact
  complete doubled-network contraction. Known-centre fast paths multiply the
  raw centre norm by Quimb's extracted `10 ** tn.exponent`; full contractions
  already apply it. During non-unitary replay, `normalize_every` /
  `normalize_final` normalize only the raw working centre and accumulate its
  removed scale in that exponent, preserving the represented state. Public
  `normalize()` is physical renormalization and clears the exponent. Keep the
  backend dispatch separate.
- Any operation that moves/rebuilds the centre must update the tracked centre
  (via `self.center = ...`, i.e. `ttn.orthogonality_center`).

### Range / subtree canonicalisation

The centre generalises to a connected subtree — the tree analogue of an MPS
mixed-canonical range. Do **not** reintroduce a separate `_orthog_center` field:
`_canonical_region` is the single source of truth and the single centre is its
one-node case.

- `ttn.canonize_subtree_(nodes, span=False, absorb="right")` gauges every tensor
  outside a connected subtree inward via quimb `canonize_around_(tags,
  which="any")` (**`which="any"`** selects the union of region tags — `"all"`
  would intersect to empty). The whole norm concentrates on the region:
  `(region.H | region) ^ all` equals the full squared norm. Sets
  `_canonical_region`. `canonize_around_node_({nid})` is the one-node delegate.
- Disconnected `nodes` raise unless `span=True`, which expands to the minimal
  connected subtree via `ttn.subtree_span(nodes)` (union of tree paths from
  `nodes[0]`; generalises `steiner_nodes` to arbitrary internal nodes).
- `ttn.canonize_around_qubits_(qubits)` is the qubit-level "range" entry point =
  `canonize_subtree_(nodes_of(qubits), span=True)`.
- `ttn.is_subtree_canonical_form(nodes=None, span=False)` verifies every outside
  tensor is an inward isometry (defaults to the tracked region);
  `is_canonical_form` is its one-node case and delegates to it.
- `TreeOptimizer` mirrors all of this: `canonical_region` property,
  `canonize_subtree(nodes, span=...)`, `canonize_around_qubits(qubits)`,
  `is_subtree_canonical_form(nodes)` — all thin delegates to the state.
## Ordinary gate routing and local FIT

Ordinary gate streams in `auto`, `direct`, `dm`, `sdc`, `src`, `zipup`,
`mpo`, and DMRG modes build a TreeMPO and use `apply_subtreempo`. Explicit
`submpo` still declares chain-MPO entries. Keep the target and disposable
FIT guess separate; `guess-zipup` is an opt-in tree warm start.
TreeFIT rejects odd-parity fermionic tensors because their graded local
projection is unsupported; use native `direct` or `zipup` for those states.

- `direct` QR-routes the complete operator before canonical SVD compression.
  `dm` changes the local split to `svd:eig`. Dense `src` contracts product-noise
  complementary environments; `sdc` builds deterministic low-rank environments.
  Both construct nested QR projectors from the original layered target. Never
  replace these algorithms with local randomized SVD or a `direct` alias.
- `zipup` contracts node layers with arriving child messages and truncates
  each outgoing message immediately. Do not add a full-target materialization
  or canonical precompression. Its intermediate cuts have noncanonical
  unvisited environments, so discarded weights are not global error bounds.
- TreeFIT uses incremental neighboring messages and `inward-outward` passes.
  TreeOptimizer's `fit_rtol="auto"` follows the state dtype; `None` disables
  tolerance stopping. Keep finite scans and exact overlap diagnostics opt-in.
  Read [`references/fit-environments.md`](references/fit-environments.md)
  before changing environment reuse, guess ownership, or stopping policy.

## Low-level two-qubit gate = exact threading + one compression sweep

This is the paper's accuracy point (Figs. 3-6) -- do not regress it.

1. SVD-split the gate into left/right factors joined by a virtual bond
   (`cutoff=0.0`, exact rank `k <= 4`).
2. Move the centre to physical node `a`, absorb the left factor into `a`.
3. Thread the virtual bond **exactly** along the geodesic to physical node `b` via
   `_thread_hop` (economical **QR**, lossless, `absorb="right"`); the crossed
   bond grows transiently by at most `k <= 4`.
4. Absorb the right factor into physical node `b`.
5. Only now run `_compress_path` -- a single canonical compression sweep back
   along the geodesic, truncating every touched bond to `chi`.

Because both gate factors are present before any truncation, each SVD sees the
complete gate -- markedly more accurate at finite `chi` than truncating each hop
while threading. `compress_between` kwargs: `(tags1, tags2, max_bond, cutoff,
absorb, canonize_distance=0, ...)`; **`canonize=` is NOT a valid kwarg** (it is
forwarded to the SVD and raises `TypeError`). Unique `rand_uuid()` bonds avoid
"index appears more than twice" errors during threading.

Native fermionic trees take an isolated version of this kernel:
`_fermionic_thread_hop` explicitly calls the native Symmray QR and carries its
graded factor. `TreeTensorNetwork._fermionic_compress_edge_` uses a reduced
graded core: when the destination endpoint is proven isometric, it QR-splits
the active endpoint and SVDs only its `R` factor; otherwise it QR-reduces both
endpoints and SVDs their contracted core. Only an unrecognised reduction hint
falls back to the complete two-node SVD. The reduction hint must remain
separate from the destination tensor object, and the one-sided split must use
fresh intermediate bond names before restoring the live edge label. See the
performance reference for the algebra and profiling evidence. Dense/nonfermionic
trees retain the generic Quimb edge wrappers.

### Sibling-leaf fast path (`_apply_2q_sibling_factors`)

When `plan.parent[la] == plan.parent[lb]` the two leaves meet at a shared
parent, so no threading is needed. Both direct-SVD and Quimb-MPO factors are
absorbed into their leaves, then the two leaves and parent are contracted into
one blob and re-split by two truncating SVDs (`absorb="right"` -> the two leaf
factors are isometric, the parent is the new centre). Both new bonds keep their
canonical `_tb...` names via `bond_ind=`. This is the common case in a good
layout and avoids QR hops and double-bond fusion. Leaves are never directly
bonded (both bond only to the parent), so the correlation flows through the
parent blob -- this is exact up to the truncation.

## Multi-qubit / sub-MPO application (`apply_subtree_operator`)

`apply_subtree_operator(op, where, *, max_bond=None, cutoff=None,
renormalize=False)` applies a general operator on `k >= 1` qubits in one shot --
a `k`-qubit gate, a multi-site **non-unitary / Kraus** operator, or a whole
**Trotter block**. It extends the two-factor path-thread kernel to the whole
spanning subtree: the tree analogue of a sub-MPO applied over a
covering range then compressed (quimb's `gate_with_submpo` is `MatrixProductState`
-only; the tree base `TensorNetworkGenVector` has no such method).

1. `snodes = _steiner_nodes(site_nodes)` -- minimal connected subtree spanning
   the target physical nodes.
2. Move the centre onto a target physical node
   (`_move_center(site_nodes[0])`, incremental)
   so the **whole exterior is isometric toward the subtree**.
3. Factor `op` into an exact tree-MPO on the same Steiner tree by packing each
   `(output,input)` physical pair into a dimension-four leg and applying
   leaf-to-hub SVDs.
4. Absorb the tree-MPO into copied local state tensors. For each
   `_peel_order(snodes)` edge, QR-split the child message while retaining all
   physical and exterior state legs, then contract its new state bond into the
   parent together with the old state/operator bonds. No dense state tensor for
   the whole Steiner subtree is formed; the last node is the hub.
5. Install every routed Q factor with its ``left_inds`` isometry metadata.
   Dense trees and charge-aligned native Symmray trees can then recover the
   hub centre through the normal canonical state machine without repeating
   those QRs; missing or malformed native proofs use explicit graded QR.
   Finally make one depth-first canonical SVD sweep: every affected tree edge
   is truncated once, after the complete operator has arrived. Dense path and
   subtree sweeps select one-sided ``reduced="left"`` compression only when
   the destination tensor's live ``left_inds`` proves the required isometry;
   native graded compression keeps its explicit block-SVD semantics.
   `renormalize=True` renormalises afterwards (for Kraus/projection).

State bonds are always read from the live tensors because gate application can
rename them. New state message bonds are fresh per-update names, while operator
bonds are private to the temporary tree-MPO. `apply_gate` routes
`len(where) >= 3` here; `k == 1`/`k == 2` still take the optimised
leaf-absorb / threading paths (but `k == 1` non-unitary and `k == 2` Kraus
can be sent here explicitly).

### Native streamed sub-MPOs

An explicit `("submpo", mpo, where)` stream event first attempts the native
leaf-to-hub QR-routing sweep. Quimb MPO payloads expose their active site tags,
tensor map, and operator bond indices, so their virtual bonds can be carried
through the TTN without calling `mpo.to_dense()`, then compressed once over the
affected subtree. `estimate_bonds()` uses the product of MPO bond dimensions
crossing a cut as a conservative operator-Schmidt bound. Payloads without that
interface use the dense `to_dense()` fallback and remain subject to
`max_operator_qubits`.

## Readout

- `TreeTensorNetwork.local_expectation(op, where)`: dense/nonfermionic
  single-site readout contracts the centre tensor; dense multi-site readout
  contracts the minimal Steiner subtree. Native fermionic readout instead
  inserts the Symmray operator with `contract=False` and contracts the complete
  doubled tree. This preserves graded boundary phases and deliberately avoids
  the ordinary isometric-exterior shortcut. Its `max_bond` argument is
  compatibility-only: the exact native contraction is not truncated. Dense
  readout restores a known canonical centre/region and uses a temporary copy
  when the gauge is unknown; native readout leaves the gauge untouched.
  Normalized native readout reuses a state-versioned norm denominator until a
  mutation invalidates it.
- `measure(q, outcome=None)`: move centre to the physical node, read exact Born
  probabilities from that one tensor (`w_i = sum_bond |t[i,bond]|^2`,
  normalise), sample via `self.rng.choice` or force `outcome`, project with a
  one-hot `apply_1q`, then `normalize()`. Returns the outcome bit. `reset(q)` =
  `measure` then conditional `X`. `seed` in `__init__` sets `self.rng`; `copy()`
  derives a deterministic child seed for a fresh independent RNG.
- Stream control events follow the MPS tuple/mapping contract for `measure`,
  `cap`, `reset`, and `measure_reset`. Pauli measurements support product observables
  on distinct qubits, use `+1`/`-1` eigenvalue outcomes, and append
  `(pauli, where, outcome, probability)` to `measurements`; reset measurements
  are internal and are not recorded. A cap contracts and removes one physical
  site, compacts labels above it by default, and absorbs a leaf into its unique
  tree parent; a physical root leg is contracted without removing tree edges;
  `stable_labels=True` / `compact_labels=False` preserves caller-facing labels
  while storage remains compact. `measure_pauli` returns outcome and Born
  probability directly; `project_pauli(..., renormalize=False)` preserves the
  branch norm, and both can return support/span/bond/norm diagnostics.
- `to_dense()` returns a host NumPy statevector in `k0, k1, ..., k(n-1)` order;
  it is a readout boundary, not evidence that a Torch/CuPy live state moved.
- `ps_to_ttn`, `hrs_to_ttn`, and `TreeSampler` resolve physical sites through
  `node_of_qubit`, so an optional root site is constructed and sampled in the
  same `q0..q(n-1)` order as leaf sites.
- `run(progbar=True)` shows a tqdm replay bar with one-/two-/multi-qubit
  counts, current bond usage, norm, and a norm-based truncation proxy. Dense and
  native fermionic replay use the same `1 - (norm / reference_norm)^2` proxy;
  the reference resets after control or explicitly non-unitary events. This is
  display-only, not a substitute for truncation history.
- `bond_report()` / `estimate_bonds()` / `max_bond()` /
  `convergence_sweep(...)` are diagnostics. `estimate_bonds()` is the
  non-mutating Eq. (4) dry run: it multiplies operator-Schmidt ranks on each
  crossed edge and can conservatively flag a `chi` that will truncate.
- `TreeTensorNetwork.validate()` checks the live tensor/bond structure against
  its `TreePlan`; use `check_canonical=True` for the more expensive isometry
  check. `TreeOptimizer.preflight(...)` adds `max_bond`,
  `max_operator_qubits`, and `max_subtree_nodes` resource limits before replay;
  the constructor defaults to finite dense/operator-subtree guards, and
  `None` disables either guard. Product-Pauli measurement uses a factorized
  parity projector rather than a dense `4**k` matrix.
  `convergence_sweep` builds the tree once and reuses it across `chi` so the
  comparison isolates truncation from layout; it reports `fidelity` (only when
  `2**n <= dense_cap`) and reference-free `max_drift`.
- `record_history=False` disables retained per-edge and per-update history for
  long replays. `TreeTensorNetwork` invalidates its canonical-region metadata
  and native norm cache after direct Quimb mutators; use
  `invalidate_canonical_form()` after raw tensor edits. The optimizer's
  state-aware wrappers restore a known centre only for operations proven to
  preserve canonicality.
- `truncation_report()` returns the per-edge compression / split history with
  before/after dimensions. `track_truncation=True` additionally probes the
  untruncated local singular spectrum and records absolute discarded weight and
  relative discarded fraction; native Symmray reports use the actually kept
  charge-block spectrum. Leave it false on performance runs: the spectrum
  probe adds local SVD work per truncation edge. The report's gate-level
  `updates` group edge events by support and include cumulative relative loss,
  analogous to the MPS infidelity trace.

### Tree operators and noisy trajectories

Use `TreeMPO` rather than a chain MPO for exact operators on a
`TreeTensorNetwork`; never densify or install its chain compatibility view as a
state update. Tree trajectory replay samples Kraus branches on copied TTNs and
normalizes only the selected live branch. Read
[`references/operators-trajectories.md`](references/operators-trajectories.md)
before changing TreeMPO construction, native charge sectors, operator
canonicalization/compression, or independent/coalesced trajectory replay.

## Performance and layout

The public performance defaults are ``mode="auto"`` (TreeMPO routing),
``threads=1`` and ``subtree_workers=1`` (small-TTN operations avoid
oversubscription), ``profile=False``, and ``track_truncation=False`` (no
diagnostic spectrum SVDs). The low-level
``TreeTensorNetwork.compress_edge_`` default is the same ``cutoff_mode="rsum2"``
used by ``TreeOptimizer``. A ``track_truncation=True`` warning is intentional:
it identifies the extra diagnostic work; explicit stream backend/device
mismatches and incompatible non-NumPy dtypes are errors, and legacy mode
selectors are the other actionable warning class.

Dense and native trees share the direct one-edge contraction, immutable path
cache, routed-isometry reuse, and proof-forwarding optimizations. Keep the
thread cap, self-healing tensor-id cache, copy semantics, and
TreeLayoutFinder objective plumbing intact. The detailed performance and
non-binary layout contract is in
[`references/performance-layout.md`](references/performance-layout.md); read
it before changing those paths.

## Gotchas / teaching notes
- `convergence_sweep` observable `max_drift` can show a **false plateau**: a
  garbage low-`chi` state can have small drift while fidelity is ~0.01. Trust
  fidelity (small `n`) or push `chi`; drift alone lies.
- Do not truncate while threading in the direct route -- it breaks the
  "each truncation sees the whole gate" accuracy property. The explicit
  `zipup` route deliberately makes its approximation during message routing.
- Do not pass `canonize=` to `compress_between`.
- 2q gate reshape order is `(out_a, out_b, in_a, in_b)`.

## Roadmap / not yet implemented
- Chain-only MPS execution modes (`svd`, `swap`, `perm`, `su`, and `mix`)
  are not exposed on arbitrary tree geometry. `dmrg` uses native TreeFIT;
  `mpo` and `zipup` have tree-specific routes described above. Native structured
  sub-MPO payloads are routed through the Quimb MPO site interface; only
  payloads that do not expose that interface use the guarded dense fallback.
  Pauli and computational-basis measurement helpers are dense two-level qubit
  APIs and intentionally reject native fermionic TTNs, whose local observables
  must be supplied through the fermion/Symmray model layer.

## Validation

Activate the shared environment and use temp caches:

```bash
source ~/envs/py312/bin/activate
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
  PYTHONPYCACHEPREFIX=/tmp pytest -q tests/test_optimize_tree.py
```

For public-API/layout changes also run:

```bash
pytest -q tests/test_public_api.py tests/test_package_layout.py
python -m pyflakes src/pepsy/optimizers/tree tests/test_optimize_tree.py
```

Documentation is plain Markdown under `docs/`; no local documentation build is
required.

The safety-net tests are `test_tree_matches_statevector` (untruncated fidelity
must stay exactly 1.0), the multi-site / sibling / measurement regressions, and
the state-handoff/backend cases: exact product TTN/MPS mounting, rejected
entangled relayouts, native Torch controls/readout, and mixed-backend rejection.
Add a regression test for every new behaviour and prefer `structure="balanced"`
plans when a test needs deterministic sibling relationships.

For noisy trajectory changes, also run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_trajectory_noise.py -k 'not benchmark'
```
