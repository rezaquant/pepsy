# Tree FIT environments and ownership

Read this reference before changing `pepsy.fitting.TreeFIT` or its
`TreeOptimizer` integration. The shared kernel also serves `TreePeps`.

- A directed message `u -> v` contracts only node `u`'s target layer(s), its
  fitted bra tensor, and cached messages `w -> u` for neighbors other than
  `v`. Use an iterative postorder traversal for missing messages. Do not
  reconstruct whole branches or store their node sets per edge.
- A cached message retains its incoming dependencies. Invalidation starts
  from changed nodes and propagates through cached dependent messages;
  opposite, untouched branches retain their tensors. Center movement must
  invalidate the full changed path before native QR can rename its bonds.
  Stop that path at the first active-block node. A center already inside the
  block proves its exterior is isometric; no interior QR is needed before
  replacing the block. Factorization sets the final requested center.
- Target bonds are fixed private indices; fitted bonds must be resolved
  from the live state. Temporary messages need no tensor tags.
  Native fermionic effective tensors need parity flips on their dual open
  environment legs before factorization. Cache the corrected tensor once;
  do not phase physical outputs or contracted target bonds as boundaries.
  Verify individual exact-state updates, since phase errors can cancel over
  a complete pass and evade a terminal fidelity check.
- `environment_strategy="native-blockwise"` is a capability-gated opt-in.
  Keep Quimb/Cotengra planning and pass the local implementation pair only to
  messages/effective tensors. Use public Symmray graded tensordot with
  `mode="blockwise"`; never patch global dispatch or bypass fermionic phases.
  The optimizer forwards `fit_environment_strategy`; default is `"default"`.
- Use pure Quimb contractions on existing target/message tensors. Keep the
  protective effective-tensor copy because factorization changes its tags.
  Owned target-layer tensor wrappers may be transferred with `virtual=True`;
  input state/operator wrappers must remain independent.
- TreeFIT accepts `copy_target=False` only for a transferred disposable
  target. The exact lazy target and the FIT guess are separate networks.
  Compressed guesses copy the state without copying parent replay history,
  queued gates, or diagnostics. Keep copy()'s single child-seed draw from the
  parent RNG so later measurements retain their seeded sequence; randomized
  compression still uses `fit_init_seed`.
  Public optimizer copies still retain independent histories and a derived
  child RNG, and must preserve the complete FIT schedule.
  Default `fit_init_strategy="auto"` selects dense `guess-src` or native
  fermionic `guess-direct`. Explicit unsupported native methods must still
  raise. `guess-direct` applies/compresses the operator on a private state;
  `direct` alone keeps the current state as the initial guess.
- TreeOptimizer defaults to automatic cutoff, `rsum2` cutoff mode,
  `fit_rtol="auto"`, and `fit_min_iter=2`. FIT tolerance is `1e-3` for
  16-bit data, `1e-5` for float32/complex64, and `1e-9` otherwise. Explicit
  `None` disables tolerance stopping. Automatic stopping is disabled for
  declared non-unitary replay and `track_norm=False` updates; explicit
  numeric tolerances remain honored. Standalone TreeFIT defaults to no rtol.
- `inward-outward` and `outward-inward` name the two passes of each iteration,
  relative to the active region's medial node. Preserve `RL`/`INOUT` and
  `LR`/`OUTIN` as aliases with unchanged traversal order. The default four
  optimizer iterations permit eight directional passes and reach one-site
  refinement. Named `dmrg3` uses `(3, 3, 2, 1)`, with its two-node transition
  controlled by `fit_two_site_transition_sweeps` (one by default). Explicit
  budgets remain authoritative. Standalone TreeFIT keeps its fixed defaults
  and offers `two_site_transition_sweeps=0` on each run entry point.
- `traversal="depth"` preserves depth ordering. Opt-in `"depth-first"` groups
  connected blocks by branch of the region's medial hub using iterative DFS.
  Preserve the block set and reverse the order for the inward pass. Test
  real center travel and finite-bond fidelity before changing the default.
  The optimizer forwards `fit_traversal` and copies preserve both policies.
- A one-node active region uses one exact projection by default in
  `run_gate(single_node_fast_path=True)` / TreeOptimizer. Skip guess replay,
  preserve the compressed guess's parent child-seed draw, record one norm
  and optional finite scan, and report `single_node_exact`. This solves the
  fixed-exterior problem, not an arbitrary full target. Do not shortcut a
  multi-node region merely because its block size is one. `run`/`run_eff`
  retain `single_node_fast_path=False` for fixed-sweep compatibility.
- Convergence reads only the terminal canonical-center norm. Patience counts
  stable same-phase comparisons and resets when the block size changes.
  Do not interpret norm stagnation as a global fidelity bound.
  Cache traversal orders only within each run's fixed region. Three-node
  factorization must also support an explicit endpoint center by peeling the
  newly exposed middle node and retaining the already factored child bond.
- Routine diagnostics never contract a doubled target. Use a known
  `target_norm` for normalized local fidelity, or report it unavailable.
  Explicit overlap diagnostics may use lossless QR to obtain the target norm
  from one hub tensor, and failures must remain diagnostic errors.
- `fit_finite_check=False` keeps scans opt-in. Trusted local updates do not
  revalidate every outside isometry. Keep `track_infidelity` independent.
  `TreeOptimizer.run(finite_check=True)` additionally scans final tensor data
  in all modes. Scope this setting to replay, forward it to shots, and warn
  once per replay; owned TreeFIT instances suppress duplicate warnings only.
- Odd-parity fermionic FIT remains unsupported and must fail before state
  installation. Validate even-parity native FIT independently of dense
  NumPy/Torch paths when changing message contractions.

Regressions: `tests/test_tree_fit_priorities.py`, `tests/test_tree_fit_messages.py`,
`tests/test_optimize_tree.py`,
`tests/test_tree_zipup.py`, and `tests/test_tree_peps_optimizer.py`.
