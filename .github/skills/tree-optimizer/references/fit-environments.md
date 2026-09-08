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
- Target bonds are fixed private indices; fitted bonds must be resolved
  from the live state. Temporary messages need no tensor tags.
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
- TreeOptimizer defaults to automatic cutoff, `rsum2` cutoff mode,
  `fit_rtol="auto"`, and `fit_min_iter=2`. FIT tolerance is `1e-3` for
  16-bit data, `1e-5` for float32/complex64, and `1e-9` otherwise. Explicit
  `None` disables tolerance stopping. Automatic stopping is disabled for
  declared non-unitary replay and `track_norm=False` updates; explicit
  numeric tolerances remain honored. Standalone TreeFIT defaults to no rtol.
- `inward-outward` and `outward-inward` name the two passes of each iteration,
  relative to the active region's medial node. Preserve `RL`/`INOUT` and
  `LR`/`OUTIN` as aliases with unchanged traversal order. The warm-up consumes
  the iteration budget: two default block iterations do not guarantee a
  subsequent one-site refinement iteration.
- Convergence reads only the terminal canonical-center norm. Patience counts
  stable same-phase comparisons and resets when the block size changes.
  Do not interpret norm stagnation as a global fidelity bound.
- Routine diagnostics never contract a doubled target. Use a known
  `target_norm` for normalized local fidelity, or report it unavailable.
  Explicit overlap diagnostics may use lossless QR to obtain the target norm
  from one hub tensor, and failures must remain diagnostic errors.
- `fit_finite_check=False` keeps scans opt-in. Trusted local updates do not
  revalidate every outside isometry. Keep `track_infidelity` independent.
- Odd-parity fermionic FIT remains unsupported and must fail before state
  installation. Validate even-parity native FIT independently of dense
  NumPy/Torch paths when changing message contractions.

Regressions: `tests/test_tree_fit_messages.py`, `tests/test_optimize_tree.py`,
`tests/test_tree_zipup.py`, and `tests/test_tree_peps_optimizer.py`.
