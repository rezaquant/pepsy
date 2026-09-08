# Tree optimizer performance and layout contract

This reference contains the lower-frequency details moved out of the main
Tree Optimizer skill so the upload-facing `SKILL.md` stays concise.

## Performance and stability

- FIT offers opt-in `fit_traversal="depth-first"` for branch-grouped updates
  and native `fit_environment_strategy="native-blockwise"` for local
  contractions without charge-block fusion. Both retain legacy defaults;
  the traversal can change truncated results and blockwise speed depends on
  sector structure. A truly one-node FIT region automatically uses one exact
  local projection. See `fit-environments.md` for the execution contract and
  `docs/development/notes/tree_fit_execution.md` for measured tradeoffs.

- **BLAS thread cap is the biggest performance lever.** Tree tensors are
  moderate-rank (set by local arity and an optional root physical leg, with
  dimensions bounded by `chi`), so multi-threaded BLAS/OpenMP is dominated by
  thread launch/sync overhead. `threads=1` is the default; gate
  application and heavy readouts run inside `self._thread_ctx()` using
  `threadpoolctl` when available. Only raise `threads` in a large-`chi` regime.
- The self-healing tid cache (`_nid_to_tid`, `_tid`) validates cached tensor
  ids against `self.tn.tensor_map`; a stale entry is recomputed safely.
- **Native central-edge compression.** A native compression call receives a
  reduction hint separately from its destination tensor. For a proven
  one-sided reduction (`reduced="left"`), let `A` be the active endpoint and
  `B` the destination endpoint, with `B†B = I` on the non-shared legs. The
  implementation computes the lossless graded factorization
  `A = Q_A R_A`, then SVDs only `R_A = U S V†`. It installs
  `Q_A U` on `A` and absorbs `S V†` into `B`. Thus the expensive SVD scales
  with the active endpoint's QR carry and the live bond, rather than the
  full fused two-node tensor. The proof is structural:
  `can_skip_canonize(A, B, absorb="left")` must accept the destination's
  `left_inds` and aligned Symmray charge maps.
- If that one-sided proof is absent but `reduced=True`, both endpoints are
  QR-reduced and only the contracted `R_A L_B` core is SVD'd. Unknown hints
  retain the complete two-node graded SVD as a compatibility fallback. The
  truncating step always remains the explicit native block SVD with the
  configured `max_bond`, `cutoff`, and `cutoff_mode`; the native
  `stabilized=False` policy applies only to lossless QR.
- The one-sided path uses fresh intermediate QR/SVD bond names because the
  original live edge label is still present in `R_A` while that factor is
  decomposed. After both contractions, the new compressed bond is reindexed
  to the original live edge. Reusing the old label during the SVD creates a
  repeated index and can route the contraction into an unsupported Symmray
  hyperedge.
- A 6x6, 48-gate, chi=64, complex64 Torch-CPU calibration with 12 Torch/tree
  threads reduced Tree evolution from 127.77 s to 6.21 s. The same run's MPS
  evolution was 2.85 s, so the remaining Tree/MPS ratio was 2.18x. The
  pre-fix central SVDs were 4096x4096; after the QR reduction they were
  384x384. Layout planning (~26.6 s in that run) is setup cost and is not
  included in the evolution comparison. The remaining replay gap is mainly
  gate-update/threading/contraction work; use `profile=True` to inspect
  `update` envelopes and nested `edge_compress` events before changing path
  geometry or observable contractions.
- Dense path and subtree routing preserve each QR-produced Q tensor's
  `left_inds`. Canonical recovery therefore recognizes an already-isometric
  routed branch without repeating its decomposition or entering Quimb's dense
  canonicalization kernel. Path and subtree compression also reads that proof
  before selecting one-sided `reduced="left"` compression, avoiding the
  redundant reduction QR only when the destination tensor is proven
  isometric. Missing proofs fall back to two-sided reduction. Native Symmray
  routing preserves the same proof when its charge maps are aligned; native
  canonical recovery skips only that proven lossless QR, while truncating
  native compression remains an explicit graded SVD. The network derives
  orientation views directly from live tensors; do not cache a duplicate map
  in the optimizer.
- **Update-path bookkeeping.** A compression sweep now computes the live
  isometry proof once per truncating edge and passes that proof into the native
  compressor; lossless QR edges do not perform a reduction-proof lookup at
  all. Private two-site gate-factor output labels are stable within the local
  update, avoiding per-factor UUID/reindex work while the routed operator bond
  remains fresh. Subtree QR messages are merged by destination, so sibling
  messages landing at one hub use one multi-tensor contraction instead of
  rebuilding that hub once per child. Dense message waves reuse one worker
  pool; native fermionic waves remain serial, but use the same grouped merge
  without changing graded block semantics.
- **Two-site routing prefactors.** The immutable geodesic for a repeated
  qubit support is cached and reversed when the current centre chooses the
  opposite endpoint as the source, so the cache never assumes a gauge
  location. Ordinary adjacent two-tensor merges in gate threading and local
  operator absorption use a direct backend ``tensordot``; Symmray retains its
  graded fermionic contraction semantics and unsupported hyperedges fall back
  to Quimb's general contraction path.
- **Public performance defaults.** Ordinary ``mode="auto"`` gates use the
  TreeMPO active-subtree route and deterministic compression. The dedicated
  two-site kernel remains available through the lower-level explicit APIs.
  ``threads=1`` and ``subtree_workers=1`` avoid oversubscription
  on small tree tensors, ``profile=False`` avoids timing overhead, and
  ``track_truncation=False`` avoids diagnostic spectrum probes. The low-level
  ``TreeTensorNetwork.compress_edge_`` API uses the same ``rsum2`` cutoff-mode
  default as ``TreeOptimizer``. Dense and native trees share these routing,
  contraction, path-cache, and proof-reuse optimizations; native-only code is
  limited to graded QR/SVD semantics and the complex64 zero-sector safeguard.
- **Native complex64 QR stability.** Torch can return NaNs for a finite,
  rank-deficient Symmray charge block whose norm has decayed into the
  ``1e-9`` range. Native QR therefore uses a block-local power-of-two scale
  for small complex64 blocks and divides the triangular factor by the same
  scale. This leaves ``Q @ R`` unchanged, keeps zero-charge sectors finite,
  and avoids promoting the complete replay to complex128. The native
  ``reduced="right"`` branch now mirrors Quimb's one-sided path: it SVDs only
  the active endpoint and contracts that factor into the already-left-
  isometric endpoint; unproven ``right``, ``False``, and ``lazy`` hints retain
  the conservative full-SVD fallback.
- On the same 6x6 χ=64 complex64 Torch-CPU harness, the post-compression-fix
  Tree evolution was 6.21 s in the saved baseline; the update-path pass ran
  in 4.90--5.15 s on repeated warm-cache runs. Threaded BLAS and cache warmth
  affect absolute wall time, so use the profile envelopes for comparisons;
  the optimization preserves the 48-gate state and χ/cutoff semantics.
- The full 468-gate ``6x6_nsteps=0`` replay then completed with
  ``track_truncation=False`` and profiling enabled: one run measured 20.39 s
  for MPS evolution, 137.11 s for Tree evolution, and 25.79 s for Tree layout
  planning (Tree/MPS evolution ratio 6.72x). This is the stability baseline;
  layout planning is reported separately and the remaining update envelopes
  are the next prefactor target. The normalized Tree/MPS state fidelity in the
  same χ=64 run was 0.590; because the two geometries truncate different
  bonds, this is an accuracy diagnostic rather than an exact-gauge check.
- `copy()` shares the immutable `TreePlan`, owns `self.tn.copy()`, resets the
  tid cache, and derives a deterministic child seed for an independent RNG.
- FIT prepares only the exterior of each active block: retain a center already
  inside the block, otherwise move it only to the first node on the entering
  geodesic. Factorization establishes the requested final block center. Do not
  add interior QR moves before replacing all active tensors.
- Reuse immutable FIT block traversal orders within a run. Preserve the order
  and scope the cache to its fixed region; geometry-based reordering changes
  the variational schedule and needs an explicit numerical comparison.
- Compression-hook signature capabilities are cached for the current hook
  function, with one entry per optimizer. Replacing a legacy/custom hook must
  trigger fresh inspection; never retain bound-method owners in a global cache.
- CPU profiles and mode/accuracy comparisons are recorded in
  `docs/development/notes/tree_modes_review.md`. At small chi, FIT environment
  contractions and center movement dominate local SVDs; direct routes spend
  substantial time in QR/compression and TreeMPO construction. Do not infer
  GPU or high-chi performance from those small CPU measurements.

## Layout (`TreeLayoutFinder` / `TreePlan`)

`TreeLayoutFinder` reuses the MPS interaction-graph plus recursive spectral
(Fiedler) partition machinery, but keeps the recursion as a rooted tree.
Strongly coupled qubits become nearby leaves, reducing the geodesic a two-qubit
gate must thread.

- Partition uses `_similarity_weights()` from Seitz Eq. 1:
  `s(qi,qj) = |G(qi) & G(qj)| + 1/(deg_i + deg_j)`.
- `score(plan)` uses pure `pair_weights` (weighted geodesic sum, lower is
  better); keep it separate from augmented partition similarity.
- `report(plan)` compares against an index-order `structure="balanced"` tree.
  Path-quality layouts should be no worse than balanced; congestion mode is
  selected by edge-load cost.
- `TreePlan.from_order(..., structure=...)` supports `quality` (spectral),
  `balanced` (direct order, useful for deterministic sibling tests), and
  `adaptive` (community/clique-driven arity).

### Non-binary trees

The data structures and algorithms are arity-agnostic. `max_arity=2` forces a
binary tree; `(2, 3, 4)` is the default candidate set for recommendations.
`structure="adaptive"` can make variable-arity communities or near-clique
stars. `TreePlan.from_children(...)` accepts validated hand-built trees.

`recommend_arities(...)` compares path or rank-aware congestion candidates and
reports virtual degree, edge load, and peak bond growth. `objective` supports
`path`, `congestion`, and `hybrid`; `weight_mode` supports `count`, `auto`,
`angle`, and `operator_schmidt`. `recommend_layered` and
`recommend_arities` may opt into bounded `refine="greedy"` or the separate,
optional seeded `search="nevergrad"` path. Defaults remain fast and
reproducible.

Layout hot paths must traverse only a support's Steiner subtree, and dense
gate Schmidt-rank caches must key on wire positions rather than global labels.
The same controls are exposed through `TreeLayoutFinder`, `TreeOptimizer`,
`find_tree_layout`, `convergence_sweep`, and `TreeTensorNetwork.from_order`.
