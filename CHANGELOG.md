# Changelog

All notable PePsY changes are documented here.

PePsY follows [Semantic Versioning](https://semver.org/):

- **MAJOR** versions may contain incompatible public API changes.
- **MINOR** versions add backwards-compatible public functionality.
- **PATCH** versions contain backwards-compatible fixes and documentation updates.

## [Unreleased]

Changes for the next release should be added here before the version is bumped.

### Added

- Made runtime non-finite detection opt-in across all MpsOptimizer modes,
  including FIT convergence norms and mixed-mode commit checks. Enabling
  `finite_check=True` warns once per replay and validates final tensor data
  in every mode. Code comments, API docs, and warnings clarify that this is
  an optional diagnostic, disabled by default and unnecessary for normal
  optimization; nested FIT calls do not repeat the warning.
  Mixed sticky non-finite handling now defaults to False. Convergence/norm
  calculations, input validation, and explicit diagnostics remain available.

- Reduced gate FIT overhead by reusing environments across 3-to-2 sweeps,
  skipping unused rank checks for SRC guesses, caching copy capabilities only
  within a replay, and reading scalar convergence norms without stacked
  vectors. Preserved SRC initialization and sweep budgets.

- Updated gate FIT defaults to eight alternating RL sweeps with two-site
  blocks, two warm-up sweeps, dtype-aware convergence tolerance, and disabled
  split diagnostics. Named `dmrg3` now uses two three-site sweeps, one
  two-site transition sweep, then one-site refinement within the same sweep
  budget. `dmrg2` retains its two-site-to-one-site schedule and adjacent-pair
  shortcut. SRC remains the default optimizer initialization.

- Added opt-in `MpsOptimizer.run(finite_check=True)` validation for DMRG,
  mixed-mode FIT, and measurement/shot FIT. Per-sweep active-array scans are
  disabled by default; enabling them emits a performance warning. Scalar
  convergence/norm calculations and periodic quality checks remain independent.
- Reduced dense DMRG rollback and SRC-guess copy traffic by isolating array
  data only in the active window, with independent tensor metadata and
  read-only exterior sharing. Preserved canonical `left_inds`, Torch gradient
  connections, and conservative full copies for native symmetry arrays.

- Added `pepsy.tensors.mps_to_ttn` (also `pepsy.mps_to_ttn`) for explicit
  conversion of an entangled MPS onto a `TreePlan`. `chi=None` uses lossless
  QR without truncation; finite `chi` imposes a TTN bond cap through
  sequential environment-aware density-matrix projections without building
  the full exact TTN first. The converter preserves dense-array backend,
  device, dtype, physical labels, and represented scale, and returns a tree
  canonical at its root. A contraction-size guard raises instead of silently
  approximating. Native Symmray/fermionic input is explicitly unsupported.

- Added a geometry-aware rank scheduling policy to native `TreePeps` and
  `TreePEPO` compression. The default `order="rank"` removes the currently
  cheapest legal leaf branch using live physical/virtual dimensions, while
  `order="depth"` preserves the previous farthest-first schedule. TreePeps
  full sweeps now re-score after each completed reduction and batch their
  expensive whole-network validation to one final canonicality check;
  localized `TreePepsOptimizer` sweeps re-score sibling branches in the same
  way. Standalone edge operations retain validation by default.

- Canonicalized tree layout names across the public handoff: TreeMPO,
  TreeTensorNetwork, and TreeOptimizer accept `map_mode="coarse-*"` for
  lattice coarsening/traversal, while TreePEPO, TreePeps, and
  TreePepsOptimizer accept `map_mode="span-up"`, `"span-down"`,
  `"span-out"`, or `"span-middle"` for bounded-degree physical spanning
  trees. The selected mode is exposed on the shared plan, state, and operator;
  historical generic and `inside-out` spellings remain compatibility aliases.
  TreePEPS legacy `coarse-*` modes also accept the shared `coarse_grain`
  control through the plan and layout finder.

- Corrected `TreePeps` `span-middle` to use one central horizontal
  line/plane with an axial chain above and below every backbone site. Central
  interior sites therefore have four virtual bonds and off-backbone interior
  sites have two; TreePeps now permits rank-five site tensors. `TreeMPO` also
  retains optional `TreeLayoutFinder` metadata so `show(layout="both")` can
  print the physical lattice and term supports above its native tree view.

- Unified the Hamiltonian `to_*` conversion surface around the single
  strategy-bearing `compress=` control. The public `to_mpo`, `to_pepo`,
  `to_tree_mpo`, and `to_tree_pepo` builders now default to
  `compress="term"`, adding and compressing one term at a time.
  `compress=True`/`"auto"` explicitly select a workload-aware construction:
  automaton assembly gets one final compression, while the term route
  compresses after every term. `compress="automaton"` forces
  shared/state-diagram assembly. `compress=False`, `max_bond=None`, and
  `max_bond=False` disable numerical compression; `mode=` and
  `compress_each=` remain compatibility spellings. Automatic native tree
  conversions also choose a layout from the interaction supports when no plan
  or mapping is supplied.

- Batched internal `TreePEPO` validation across each full compression sweep.
  Standalone edge compression still validates by default, while term-by-term
  and one-shot builder paths avoid repeating the quadratic whole-network
  topology check after every compressed edge.
  `TreeMPO.show()` now keeps the clean native ASCII tree as its default, and
  `TreePEPO` retains its `TreePepsLayoutFinder` metadata through later
  operations.

- Unified `cutoff="auto"` across MPS, tree, TreePEPS, and Hamiltonian
  conversion paths: it resolves to `1e-12` for `float64`/`complex128`, `1e-6`
  for `float32`/`complex64`, and `1e-3` for 16-bit floating-point data.

- Added canonical `ham_tn.to_mpo`, `ham_tn.to_pepo`, `ham_tn.to_tree_mpo`, and
  `ham_tn.to_tree_pepo` conversions. Native tree conversions preserve the
  supplied tree geometry and avoid a chain-MPO round trip; `TreePEPO` and
  `TreeSubPEPO` are now the canonical acronym spellings, with legacy names and
  `build_*` methods retained as compatibility aliases.

- Made `mode="direct"` the explicit GibbsMps default while retaining
  `mode="mpo"` as a compatibility alias for the same direct Quimb replay.
  Added coverage for inferred map ordering, triangular coordinate graphs, and
  connected one-site term fusion.

- Added `pepsy.fitting.TreeFIT`, a tree-native cached variational fitting
  engine with directed branch environments, canonical-centre path movement,
  one-/two-/three-node local blocks, seeded randomized warm starts, and
  normalized target-overlap diagnostics. Its `run`, `run_eff`, and `run_gate`
  controls follow the chain FIT calling convention; structural retagging and
  disposable-target ownership are explicit. Correctly tagged layered targets
  are grouped by structural node, retaining local layer bonds and multiple
  inter-node bonds; ambiguous or untagged tensors are rejected. The dedicated
  path two-layer compressor remains available. `TreeOptimizer` and
  `TreePepsOptimizer` expose it through `dmrg`, `dmrg1`, `dmrg2`, and `dmrg3`.
- Aligned tree DMRG scheduling with MPS FIT: generic DMRG supports an adaptive
  larger-block warm-up followed by one-site refinement, named `dmrg1`/`dmrg2`
  use two-node growth before one-site updates, and `dmrg3` uses three-node
  growth. `guess-src` now builds a disposable TreeMPO/TreePEPO-applied tree
  guess and feeds that guess to TreeFIT while retaining the exact target
  separately; diagnostics expose the schedule and warm-start backend.

- Added `sdc` and `src` compression modes to `TreeOptimizer` and
  `TreePepsOptimizer`. Path-shaped `TreePeps` states delegate to Quimb's
  environment compressors; branching trees use a deterministic successive
  edge sweep or dense randomized-SVD edge splits, with `compression_seed` for
  reproducibility. The modes preserve TreePeps plan/index metadata and reject
  charge-unsafe randomized compression for native fermionic TTNs.

- Added path `TreePeps` two-layer operator-state compression. With
  `compression_layout="auto"`, Quimb's multi-tensor `sdc`, `src`, and `zipup`
  kernels can compress the separate PEPO and state layers directly; the
  original fused application remains available with
  `compression_layout="fused"`, while `"two_layer"` requires an explicit
  path topology.

- Made the MPS SDC surface explicit and regression-tested: bare
  `mode="sdc"` / `mode="sdc-oversample"` aliases normalize to Quimb's
  successive deterministic compressors, and the same methods are available
  as `fit_init_strategy="guess-sdc"` / `"guess-sdc-oversample"`, with strict
  Quimb version gating and no silent fallback.

- Aligned `MpoOptimizer` with the MPS DMRG scheduling and timing APIs. Generic
  DMRG now uses adaptive block warm-up followed by one-site refinement,
  `dmrg2` retains the adjacent-pair fast path, timing exposes the MPS-shaped
  stage/FIT schema, and MPO norm diagnostics retain physical `2**L` scale via
  explicit squared-norm fields and scale-safe mantissa/exponent event pairs.
  Dense direct two-site MPO replay also uses Quimb's dagger-aware auto-swap
  sandwich when available, while local fidelity events now cover compressed
  multi-site MPO paths. `MpoOptimizer.to_backend(...)` now mirrors the
  state-derived MPS conversion helper for backend, dtype, and device routing.

- Extended `exp_mpo_cluster`, `exp_mpo_cluster_product`, and the reusable MPO
  cluster expansion with native bosonic Abelian block-sparse output through
  `MPOPhysicalSpace` / `symmetry` metadata. Direct assembly now retains sparse
  virtual blocks and compiles them through the existing Symmray boundary.
  Streaming assembly also supports directional adaptive TT-SVD through
  `assembly_cutoff`, `assembly_cutoff_mode`, and `assembly_form`, while
  preserving fixed-rank backend-autodiff streaming when no cutoff is given.

- Extended `GibbsMps` with the reusable `bell_to_mps` Quimb constructor and
  configurable tag-wise trace contraction options, while preserving the
  backend and autodiff paths.

- Added exponent-aware Gibbs-MPS trace bookkeeping and a natural-log
  `log_partition_function()` readout for stable large-scale partition
  functions.

- Kept Gibbs-MPS readout on Quimb's native partial-trace path (including
  Pepsy-rescaled states), and stopped resolving the same Trotter graph
  ordering twice; randomized ordering metadata now matches the executable
  replay schedule.

- Updated `GibbsMps` to use Quimb's graph-aware `LocalHamGen` Trotter scheduler
  for first-, second-, and fourth-order product formulas. Connected one-site
  terms are combined into incident edges without leaving the selected backend;
  isolated one-site terms use exact one-site gates. Trotter layer metadata and
  ordering/fusion controls are now exposed on the Gibbs-MPS object.

- Added `GibbsMps`, a first finite-temperature purification API. It builds an
  interleaved physical/ancilla MPS from Bell pairs, applies backend-aware
  second-order imaginary-time Trotter gates through `MpsOptimizer`, supports
  `MPOBasis` one-dimensional and `OneDMap` lattice terms, and traces ancillas
  back to a thermal MPO.

- Added bounded graph-cluster assembly controls to `exp_mpo_cluster` and
  `MPOBasis.compile_graph_cluster_expansion`. The default
  `graph_assembly="auto"` uses a cutwidth-aware frontier planner before
  materializing collections, while `graph_assembly="exact"` and
  `graph_assembly="bounded"` make the exact versus controlled-approximation
  choice explicit and report the selected strategy.

- Added opt-in streaming graph-path assembly to `exp_mpo_cluster` and
  `MPOBasis.compile_graph_cluster_expansion`. `assembly="streaming"`
  inserts local graph-path cores directly into the accumulator in bounded
  batches, applies a semantic fixed-rank SVD after each batch through
  `assembly_chi`, and avoids temporary path or batch MPOs while reporting the
  working compression diagnostics.

- Added the product-named one-shot `exp_mpo_cluster_product(factors, step, ...)`
  facade for ordered `exp(A) @ exp(B) @ ...` MPO cluster expansions. It shares
  the term parsing, graph/cyclic, streaming, backend, report, and final
  compression controls of `exp_mpo_cluster` while making the factor list
  explicit.

- Fixed exact Torch export/compile log-amplitude evaluation by tracing the
  scalar contraction directly and making `backend="eager"` use the stable
  exported/vmapped graph. This prevents PyTorch 2.6 FakeTensor leakage into
  Metropolis acceptance while preserving real compiler backends.

- Added an opt-in Torch PEPS export pipeline matching the GPU VMC
  `torch.export -> torch.vmap -> torch.compile` flow. Exact PEPS models can
  compile a fixed walker batch, including stable log amplitudes; Metropolis
 proposal evaluations pad changing subsets to that batch and discard the
 auxiliary rows, while existing eager/vmap/serial paths remain unchanged.

- Added opt-in compiled boundary-MPS reuse for finite rectangular Torch PEPS.
  Boundary environments and fixed one-/two-row or one-/two-column geometry
 classes can be exported, vmapped, and compiled; connected local-energy
 targets are grouped across parent walkers and retain the eager fallback for
 unsupported Quimb or Symmray contraction paths.

- Added the backend-neutral `MPOBlock` / `MPOBlockPlan` structural inspection
  layer. First-degree automata and persistent higher-order block-sparse MPOs
  now expose virtual-state transitions, stored block counts, recipes, and
  charge metadata without retaining numerical backend arrays.

- Added charge-aware block validation and sector-wise compression diagnostics
  for higher-order MPOs. `MPOBlockPlan.validate_charges()` checks virtual
  charge labels before materialization, `FirstDegreeMPO.validate_charge_flow()`
  invokes the native Symmray local flow check, and
  `sector_aware="auto"` records native sector dimensions and block counts
  around final Quimb compression without densifying symmetric tensors.

- Extended `ham_tn.build_mpo` with compact Pauli term spellings, integer chain
  locations, and dtype-aware `cutoff="auto"` / `cutoff_mode="auto"` options;
  existing explicit local-operator terms remain supported. Builders now also
  accept `to_backend=...` and perform generic MPO accumulation and compression
  on the selected backend, infer `data_type` from that converter, accept
  `chi`/Quimb compression options, and expose a shared automaton mode that
  canonicalizes duplicate and identity-containing terms before compilation;
  `mode="auto"` selects it only when its structural width is reasonable.

- Extended higher-order `MPOBasis` and `exp_mpo` term input with the same
  compact Pauli tuple spellings as `ham_tn`, and added final-`chi` compression
  controls including `cutoff="auto"`, `cutoff_mode="auto"`, `form`, and
  `compress_opts`.

- Added the shared `shape=` geometry alias to `ham_tn`, supporting 1D, 2D,
  and 3D layouts while retaining the legacy `Lx`/`Ly`/`Lz` spelling.

- Added `to_backend=` to `exp_mpo` and `MPOBasis` term compilation. Operator
  blocks and coefficient assembly now use the requested backend before
  higher-order contractions, and ordinary final MPO output is rechecked with
  `apply_to_arrays` after optional `chi` compression.

- Added opt-in `progress=True` diagnostics to higher-order MPO exponentials.
  The color-coded bar is labeled `exp(order=N)`, reports history and
  analytical Algorithm 1--4 stages, and distinguishes Algorithm 4 analytical
  compression from final numerical `chi` compression. Timing data and the
  separate `analytical_compression` / `numerical_compression` metadata are
  retained on the returned MPO.

- Added canonical higher-order exponential modes: `exact` for Algorithm 3,
  `folded` for Algorithm 4, `hybrid` for Algorithms 3 and 4, and `auto` for
  the order-aware exact/folded policy. Historical `algorithm4`, `optimal`,
  and `approximate` spellings remain compatible aliases.

- Updated Quimb compatibility handling to defer the Symmray `safe_inverse`
  shim to older builds, accept Quimb's native long-range simple-update path,
  capability-check generalized-loop options, and adapt loop-series resummation
  to Quimb's newer `num_tensors` argument.

- Fixed tree-energy Torch policy initialization and protected native complex64
  QR from stale process-global Autoray registrations.

- Added opt-in Quimb `sdc` and `sdc-oversample` MPS compression mode names
  with execution-time capability checks; existing compression defaults and
  modes are unchanged.
- Updated randomized MPS compression to use Quimb's explicit `seed` support
  when available, while retaining a compatibility fallback for older builds.
- Added opt-in fourth-order Suzuki-Yoshida gate streams to the symmetric
  Hamiltonian and fermion helpers. Existing first- and second-order streams
  retain their public behavior, with overlapping hopping terms now arranged in
  a symmetric edge-colored half-step schedule before the fourth-order lift.
- Added a narrow Symmray compatibility path for older Quimb projector
  compression builds whose `safe_inverse` incorrectly passes an axis to a
  one-dimensional block vector.

- Added the initial `TreePepsPlan` and `TreePeps` state API for PEPS-like
  tensor networks with 2D/3D coordinate tags, stable 1D logical tags, and
  validated spanning-tree virtual bonds, including PEPS-style `show`, tree
  canonical-center movement, `info_c` synchronization, and compression hooks.
- Added a hard three-virtual-bond TreePeps rank invariant, explicit local and
  maximum tensor-rank diagnostics, a workload-aware `TreePepsLayoutFinder`,
  and Quimb-style 2D Unicode state schematics that show retained tree bonds.
- Added `left_inds`-backed isometry metadata to `TreePeps`, with canonical
  region recovery, path-only center movement, QR-free canonical edge moves,
  and a center-oriented compression sweep that avoids a redundant full QR.
- Added tree-native `TreePepo` and `TreeSubPepo` operators with separate
  input/output legs, support/span metadata, exact dense-factorized gates,
  term sums, tree-bond fusion on application, expectation values, and
  optional canonical compression.
- Added `TreePepsOptimizer` with direct tree-geodesic gate replay and
  `sub_treepepo` span replay, lossless routing, `left_inds`-aware canonical
  preparation, localized compression, per-update bond reports, persistent
  `set_gates`/`add_gates` streams, common one-/two-/multi-site aliases, and
  validated state replacement.
- Added TreePeps parity helpers for rooted topology traversal, bond-growth
  estimates and preflight, batched local expectations, dense state-vector
  conversion, normalization, optimizer state aliases, and truncation reports.
- Completed TreePeps optimizer parity for state canonicalization aliases,
  span-local explicit compression, normalization controls, intermediate-bond
  preflight, profile and transient-bond diagnostics, layout convenience, and
  chi convergence sweeps with optional dense-reference fidelity.
- Added `compression_mode="dm"` to the tree, TreePeps, and TreeStab optimizer
  families. It applies Quimb's density-matrix-equivalent local `svd:eig`
  decomposition after the complete state/operator network is fused; the
  existing direct SVD mode remains the default and native fermionic trees keep
  their graded direct-compression path.
- Added explicit `TreeOptimizer` `tree_mpo_direct` and `tree_mpo_dm` modes.
  These build and route a true TreeMPO over the active Steiner subtree rather
  than lowering gates to a chain sub-MPO; `auto` now promotes gates wider than
  four qubits to the TreeMPO route, while bounded dense direct gates fail with
  an actionable mode recommendation.
- Updated `TreeStabOptimizer` to use true TreeMPO active-span routing for
  coefficient-frame gates, Pauli sums, projections, localizers, and exact
  cooling. Its canonical modes are `tree_mpo_direct` and `tree_mpo_dm`, with
  `tree-mpo-dm` and `tree_mpo_dem` accepted as aliases.
- Added `OneDMap` center-out/inside-out traversal aliases and independent
  `TreePepsPlan.tree_order` seeds. Row-major, Hilbert, diagonal, and
  center-out orders can now guide legal degree-bounded virtual trees, and
  `TreePepsLayoutFinder` compares those deterministic seeds with weighted
  growth and reports the selected seed.
- Centralized execution-time Quimb capability checks for optional MPS
  compressors and gate transforms. `gate` and `gate_simple` now forward
  `dagger`/`transpose` to the user gate while leaving routing SWAPs unchanged.
- Added `gloop_opts` to the scalar and 2-norm loop-cluster entry points so
  newer Quimb generalized-loop generator controls can be used without
  changing existing defaults.
- Added capability-gated BP constructor/run forwarding, Autoray-native random
  FIT initialization, Quimb `LatticeBondMap` periodic-bond naming, and an
  explicit opt-in MPO auto-swap wrapper. These integrations preserve existing
  defaults and fail locally when an optional Quimb capability is unavailable.

### Fixed

- Optional MPS FIT overlap diagnostics now report non-finite contraction
  results as diagnostic errors instead of clipping NaN/infinity to a valid
  fidelity. Default replay performs no additional checks or contractions.

- Fixed TreeMPO DMRG update finalization so explicit `apply_subtreempo` calls
  close their active norm ledger and record diagnostics. Fixed
  `TreePepsOptimizer.run(mode=...)` persistence for all route/compression
  aliases and propagated shorthand compression modes to explicit `TreeSubPepo`
  stream events.

- Fixed backend preservation through higher-order sparse history materialization.
  Empty sparse virtual tensors now retain a backend reference, backend-native
  zero/equality operations no longer silently fall back to NumPy, and the
  `to_backend=` contract is covered through Torch/JAX semantic and final MPO
  boundaries for all canonical history modes.

- Fixed the native Symmray MPO dense-conversion boundary. Pepsy now supplies
  the original physical basis-to-charge maps when Quimb contracts fused native
  sectors, so `to_mpo().to_dense()` and the result of native sector-aware
  compression use computational-basis order rather than Symmray's packed
  sector order. The compiled MPO remains block-sparse until dense conversion
  is explicitly requested.
- Fixed native DMRG product results to retain Pepsy's MPO dense-conversion
  boundary after FIT returns a base Quimb MPO. `compress_mpo_product` now
  records that DMRG refinement uses `FIT.run_eff` and preserves the physical
  charge maps needed for computational-basis output.

### Changed

- Fixed the higher-order MPO convenience API so `extension_budget` is
  available consistently on batched and compatibility evolution entry points,
  while compiled evaluator policy remains fixed at compile time. The
  compatibility `history_storage="blocks"` alias now shares its compiled cache
  with the canonical `"block_sparse"` spelling.

- Extended `compress_mpo_product` with `guess_method` and `guess_seed` for
  DMRG/FIT warm starts. The default deterministic SDC guess is retained;
  dense-only `src` and `src-oversample` guesses can now initialize the exact
  lazy target before `FIT.run_eff`. Native Symmray SRC warm starts fail with a
  clear sector-awareness error rather than attempting unsupported randomized
  backend operations.

## [0.4.1] - 2026-08-27

This patch release consolidates the recent sampling, optimizer, operator,
backend, and documentation improvements developed on `develop`.

### Added

- `MpsStabSampler` now supports shared-prefix branch sampling, direct
  tableau/coefficient-MPS construction, basis-absorbing measurements, explicit
  physical/MPS qubit orders, probability queries, and branch diagnostics while
  preserving the coefficient-MPS backend for batched outputs.
- `PepsSampler` now provides exact, Quimb-MPS, and DMRG/FIT boundary proposal
  engines with conditioned ket boundaries, future marginal environments,
  prefix-grouped batches, and compact-row transfer caching.
- MPI shot-ensemble execution now includes checkpoint-aware orchestration,
  progress reporting, robust reductions, and native MPS/tree entry points.
- The documentation build now includes a generated API reference site, and
  the Guppy gate-stream adapter is available through the public interoperability
  API.
- MPO cluster expansions now expose a reusable compiled topology for ordered
  `exp(A) @ exp(B) @ exp(C)` products, explicit local `max_bond` control, and
  stabilized Torch/JAX autodiff factorization paths.
- Graph-aware MPO cluster expansions now reuse `ClusterLattice` connectivity,
  support long-range two-site clusters on 2D coordinate graphs, preserve
  ordered local exponential factors, expose trace and rank diagnostics, and
  map noncontiguous graph residuals into controlled MPO paths.
- PEPO cluster products now support ordered `exp(A) @ exp(B) @ ...` factors,
  direct physical traces, optional intermediate PEPO compression, and
  Torch/JAX-safe factor and step autodiff.
- `MPOBasis.from_square_lattice(...)` compiles coordinate-based Pauli terms
  through a reusable `OneDMap`, aligns reversed location/Pauli descriptions,
  and preserves backend autodiff coefficients while sharing MPO channels.
- `exp_mpo(...)` provides a term-centric operator/location/coefficient entry
  point that infers 1D/2D/3D layouts, accepts custom `OneDMap` orderings,
  accepts Pepsy-style Pauli-keyed mappings such as `{"XX": ((2, 3), J)}`,
  shares common MPO paths, and returns a compiled Quimb MPO by default.
- Higher-order MPO symmetry metadata now accepts case-insensitive compact
  symmetry names and charge-to-multiplicity mappings for degenerate physical
  sectors, while retaining the per-basis-state charge sequence form.
- Core package facades now resolve implementation modules lazily, and the test
  suite exposes explicit `core`, `optional`, and responsibility-based domain
  markers with a scheduled full-suite workflow.
- The top-level `pepsy` namespace is documented and guarded as a frozen
  compatibility facade; new advanced APIs should live in their owning domain
  or under `pepsy.experimental`.
- Accelerated contraction search is now optional through the `contraction`
  extra. Without it, reusable contraction optimizers fall back to Cotengra's
  built-in `sbplx` search and native Python pathfinders.
- General `MPOLocalOperatorTerm` inputs compile arbitrary dense multi-site
  operators through an exact operator-Schmidt MPO decomposition while keeping
  coefficient slots differentiable.
- `MPOPhysicalSpace` and `MPOBraiding` make local dimensions, Abelian sectors,
  grading, and odd-factor exchange signs explicit MPO construction metadata.
- `history_storage="reduced"` streams reachable products directly into the
  Algorithms 1--2 reduced history space without materializing raw virtual
  tensors, including the Algorithm 3 and 4 policies.
- MPS FIT convergence controls now use mode-neutral `fit_min_iter`,
  `fit_rtol`, and `fit_patience` names, with deprecated `mix_fit_*` aliases,
  and `stabilize_unitary` now covers DMRG, mixed MPO warm-up/fallback, and the
  standalone MPO/swap/permutation/SVD compression modes.
- PEPS boundary contractions expose typed per-fit convergence diagnostics,
  opt-in detailed timing, `return_info=True` on scalar norm helpers, and an
  information-preserving `peps_fidelity(..., return_info=True)` path.
- Dense PEPS DMRG boundaries now support cached two-site FIT sweeps with
  native SVD rank growth, independent norm/overlap bond caps, configurable
  sweep and truncation policy, and optional adaptive stopping across the
  boundary metrics, `SweepOptimizer`, and `PepsOptimizer` APIs.
- `SimulatorPlanner` and `recommend_simulator` provide non-executing,
  chi-aware rankings across MPS, tree, MPS-stabilizer, and tree-stabilizer
  circuit strategies using physical and dressed-frame support geometry.
- `TreeOptimizer` and `TreeTensorNetwork.compress_edge_` now accept
  `cutoff_mode`, allowing Tree truncations to use the same Quimb
  singular-value cutoff conventions as MPS truncations.

### Changed

- MPS and tree optimizer truncation defaults now use dtype-aware automatic
  cutoffs and explicit automatic cutoff-mode resolution. MPS DMRG/FIT defaults
  now use adaptive schedules, tighter automatic fit tolerances, and a
  two-site pair policy that can be overridden explicitly.
- MPS quality checks are opt-in, and optimizer seed handling no longer leaks
  MPS sampling seeds into contraction options.
- Torch linear-algebra defaults now select the exact SVD path, with the
  configured backend policy preserved across optimizer workflows.
- MPS, tree, and stabilizer optimizer streams now enforce backend, device, and
  applicable dtype compatibility at their stream boundaries; callers must use
  an explicit backend converter for intentional cross-backend payloads.

### Deprecated

- Backend helpers imported from `pepsy.tensors` now warn and direct callers to
  their canonical `pepsy.backends` namespace. `pepsy.experimental.mera` now
  directs callers to `pepsy.experimental.qmera`; the equivalent
  `pepsy.optimizers.mera` compatibility namespace directs callers to
  `pepsy.optimizers.qmera`. The legacy tensor constructor spellings
  `build_contraction`, `SpinfulFermionHubbard`, and `hrps_to_*`, the generic
  boundary spellings `normalize` and `infidelity`, the qMERA optimizer alias
  `QMeraParametricEnergyOptimizer`, and the stabilizer alias
  `MpsStabOptimizer` now also warn and identify their canonical names.

### Fixed

- Graph MPO cluster assembly now carries singleton backgrounds through skipped
  chain sites and retains products of disjoint long-range clusters with
  crossing or nested MPO spans; ordered MPO-basis products also reject
  mismatched chain geometry. Ordered PEPO products now use the same joint
  local-residual construction instead of multiplying independent factor
  PEPOs. MPO
  fixed-rank SVD dispatch now remains compatible with custom JAX registrations
  and switches stabilized Torch mode to match real or complex inputs.
- Term-centric MPO parsing now accepts integer coefficients without confusing
  them with lattice sites, rejects fractional shapes and coordinates instead
  of truncating them, and reports when semantic history cannot survive Quimb
  compression.
- MPO and Pauli supports now preserve site/operator pairing while sorting,
  multiply repeated-site factors in supplied local order, retain the Pauli
  phase, and reject Boolean or fractional site labels instead of coercing
  them to integers.
- Stabilizer planner diagnostics now explicitly describe when a cap changes
  the logical MPS width, preserving the warning contract for unavailable
  static-frame candidates.
- Native Symmray MPS compression now measures non-unitary target norms from a
  sector-preserving canonical active-span overlap instead of constructing a
  routed target copy, and native bosonic FIT reuses audited reversed-sweep
  environments. Infidelity samples identify the target-norm source route.
- MPS/FIT diagnostics now rebase unitary norm tracking after state replacement,
  manual normalization, and layout changes; profile DMRG target-norm work; keep
  unclipped norm-ratio diagnostics with an overshoot guard; and compute one
  terminal canonical-center norm per FIT sweep. Reused FIT objects reset
  per-run traces and split metadata, and full-chain entry points reject invalid
  sweep counts consistently. Canonical modes reject cyclic MPS inputs, local
  normalization reuses FIT's singleton center without an extra QR sweep, and
  mixed in-place commits preserve Quimb isometry metadata.
- Two-site PEPS boundary warm starts retain their requested future bond caps
  instead of treating the current rank as the cap; target replacement,
  per-call FIT overrides, and lowering `chi` now preserve explicit policy.
- `TreeOptimizer` non-unitary scale control now preserves removed normalization
  in the TTN exponent, and fast centre-based norm reads include that exponent,
  so `normalize_every=True` no longer changes the represented state.
- Stabilizer optimizer and sampler branch state handling now keeps temporary
  conditional branches isolated from the live optimizer and separates Born
  probabilities from compression-fidelity diagnostics.
- CuPy-backed tree gate application and recent stabilizer optimizer routes now
  preserve their backend and consistency contracts.

## [0.4.0] - 2026-07-27

This release removes obsolete package-layout compatibility layers and keeps
advanced-domain discovery under the single lazy `pepsy.experimental` namespace.

### Removed

- Old flat modules such as `pepsy.core`, `pepsy.gates`, and `pepsy.optimize_mps`.
- The duplicate `pepsy.extensions` namespace and unused re-export leaf modules.
- The in-package benchmark directory and its orphaned benchmark test.

### Changed

- Repository agent guidance is concise and delegates domain invariants to the
  relevant skills.
- Active documentation now points to public simulation and sampling APIs rather
  than deleted benchmark scripts.

## [0.3.0] - 2026-07-24

This release consolidates the tensor-network API refresh and the new native
TreeOptimizer and symmetric-tensor workflows.

### Added

- Native fermionic TreeTensorNetwork evolution, observables, measurements, and
  state-versioned norm caching with explicit mutation invalidation.
- TreeOptimizer support for direct and MPO execution paths, including native
  subtree and multi-site operator routing.
- Backend-aware symmetric-tensor and Symmray sweep support with regression
  coverage for Torch-backed block arrays.
- Public trajectory, stabilizer tensor-network, fermionic, and VMC workflow
  APIs with corresponding documentation and examples.

### Changed

- TreeOptimizer execution modes are now limited to `auto`, `direct`, and
  `mpo`; unsupported legacy mode names fail clearly.
- Dense and native TreeOptimizer measurements use consistent gauge and norm
  diagnostics semantics.
- Progress reporting uses a common norm-infidelity proxy, and Symmray
  truncation diagnostics use the actual retained block spectra.
- Public imports and package documentation are organized around the current
  `pepsy.backends`, `pepsy.boundary`, `pepsy.operators`, `pepsy.optimizers`,
  `pepsy.sampling`, `pepsy.solvers`, and `pepsy.tensors` namespaces.

### Fixed

- Fermionic local expectations and norm calculations now agree with complete
  graded-network reference contractions, including nonzero hopping terms.
- Norm-cache invalidation covers public optimizer mutation and normalization
  paths, including constructor normalization.
- Symmray backend conversion, soft MPO bond caps, and blockwise discarded
  weight reporting are now handled without dense global-spectrum assumptions.

### Removed

- Stale benchmark and example artifacts that no longer represent the current
  public API.

## [0.2.0] - Baseline

`0.2.0` is the package metadata baseline that preceded this changelog. Earlier
changes were not recorded in a versioned changelog, so historical entries are
intentionally not reconstructed here.
