---
name: mps-optimizer
description: 'Run, review, debug, or extend pepsy.MpsOptimizer for MPS gate-stream replay, MPO/SVD/DMRG/swap compression, persistent graph layouts, canonical-center diagnostics, normalization, control events, and exact-mode transitions.'
---

# MPS Optimizer in pepsy

Use this skill for `pepsy.MpsOptimizer` and its implementation in
`src/pepsy/optimizers/mps/optimizer.py`. Read the public API documentation at
[`docs/api/optimizers/mps.md`](../../../docs/api/optimizers/mps.md) and the closest
tests before editing.

## Decision guide

Make decisions in this order; each choice owns a different invariant:

1. **State contract.** Use `exact` for a dense reference or cyclic input,
   `su` for gate-only simple-update evolution, and a canonical open-boundary
   MPS mode for local norms, controls, and DMRG. Do not make a cyclic MPS look
   canonical by scanning it—the missing loop environment makes a one-center
   norm invalid.
2. **Compression route.** Use `dmrg2` for the normal variational production
   path, `dmrg1` when the fixed two-site-growth/one-site-refinement schedule is
   required, and `dmrg3` when a three-site warm-up is useful. Use
   `quimb-<method>` (or its bare method alias) when benchmarking a specific
   Quimb compressor, `svd` for a transparent local split reference, and
   `swap`/`perm` when endpoint movement is the intended representation.
3. **Target versus guess.** `fit_target_strategy` decides how the exact
   operator-applied target is represented (`layered` dense factors versus
   materialized/native MPS). `fit_init_strategy` decides only the disposable
   starting point for FIT (`direct`, random expansion, or `guess-*`). Never
   replace the exact target with the guess, and do not disable an explicit
   `guess-*` merely because the active bonds already reached `chi`.
4. **Cutoff and seed.** `target_cutoff` controls optional target
   materialization, while `cutoff` controls output compression. SRC/SRCMPS
   choose rank through `max_bond` and ignore singular-value cutoff. Forward an
   explicit `compression_seed`/`fit_init_seed` only to the randomized policy;
   never leak it into a contraction-option dictionary.
5. **Control boundaries.** Flush the preceding gate segment before
   `measure`, `reset`, `measure_reset`, `cap`, or feed-forward dispatch. A
   measurement probability belongs to the pre-collapse physical state; a
   post-collapse norm is a separate compression/branch diagnostic. Preserve
   logical site labels in records even when a persistent layout executes on
   physical positions.

When reviewing a change, ask which decision above it changes. If it changes
more than one, keep the conversion or bookkeeping boundary explicit rather
than hiding policy in a mode-specific helper.

## Execution modes

- `fit` / `dmrg` / `dmrg1` / `dmrg2` / `dmrg3`: local variational compression;
  `dmrg1` uses at most two two-site growth sweeps followed by one-site
  refinement, then latches one-site updates after all full-chain attainable
  bond ceilings are reached. `dmrg2` uses its required two-site warm-up (two
  sweeps by default) followed by one-site refinement, and `dmrg3` uses the
  same fixed warm-up policy with three-site updates, then one two-site
  transition sweep before one-site refinement. Generic `dmrg` remains
  rank-adaptive until its active-bond ceilings are reached; rank stagnation is
  not an early exit there. A `dmrg1` window already at its attainable ceilings
  starts directly with one-site FIT. An under-capacity non-adjacent `dmrg1`
  window requires `n_iter >= 3`: two block sweeps plus at least one refinement
  sweep. Its default `fit_patience=2` is a two-sample same-phase norm window,
  i.e. one stable comparison. A two-site window is a structural special case for
  `dmrg2` by default, or other schedules with `fit_single_pair_fast_path=True`:
  perform exactly one two-site update, no
  one-site refinement, then advance to the next gate without consuming the
  remaining `n_iter` budget. Compose with
  [`tensor-fitting`](../tensor-fitting/SKILL.md) for FIT kernel, target, rank
  growth, symmetry, stability, or profiling changes.
- `quimb-<method>`: native Quimb non-local gate/MPO replay with the selected
  compression method; `quimb-direct` is explicit and `quimb` is its direct
  alias. The legacy `mpo-<method>` and `mpo` spellings remain supported.
  Supported methods include `direct`, `dm`, `zipup`, the zipup/SDC/SRC/SRCMPS
  oversampling variants, `fit`, `fit-zipup`, `fit-projector`, and
  `fit-oversample`.
- `svd`: local SVD compression; `swap`: swap-and-split with swap-back.
- `perm`: swap-and-split with lazy logical-to-physical tracking.
- `mix`: transactional direct/MPO warm-up followed by one-site FIT by default,
  with per-step MPO fallback. Explicit `fit_block_size=2` or `3` opts into
  mixed block-FIT transactions.
- `exact`: fully contracted TensorNetwork replay, without MPS canonical metadata.

Keep `exact` separate from MPS code. When switching from exact to an MPS mode,
rebuild an MPS from the explicit physical indices and canonicalize it. Do not
switch a persistent-layout optimizer into exact mode because that would lose
the physical/logical layout contract.

## Persistent layouts

For repeated evolution, use:

```python
opt.apply_layout("quality")
opt.run()
opt.run()  # reuses the same physical order
```

`logical_order[position]` is the logical site stored at a physical MPS
position. Use `logical_site(position)`, `position(site)`, `remap_sample`, and
`to_dense()` for readout. A persistent layout never swaps the MPS back.

The reorder is free exactly when `p.max_bond() == 1`: rebuild the product MPS
with tensors in the new order and do not call an SVD swap. If the initial MPS
is entangled, raise by default. Only `allow_lossy_reorder=True` may enable the
one-time SVD reorder, and it must use the caller's `cutoff` rather than a
hardcoded exact cutoff. `cap` control events are rejected because they change
the MPS length; measure/reset events keep recording logical labels.

Do not recommend `run(use_layout_finder=True)` for iterated evolution. It is a
deprecated compatibility path that temporarily reorders, runs, and swaps back.

## Canonical-center contract

`optimizer.info_c["cur_orthog"]` is algorithm state, not cosmetic metadata.

Canonical MPS modes require an open-boundary MPS. Reject cyclic states before
constructor, `set_p()`, or mode-switch mutations because a periodic loop has no
exact one-tensor mixed-canonical norm. Exact mode may contract a cyclic input
before rebuilding an open MPS.

- `_current_orthog` may scan only when the state metadata is unknown.
- Pass the tracked range through Quimb `info` arguments and canonicalization.
- Local Pauli expectations should use `local_expectation_canonical` when
  available and move the center from the tracked range to the support.
- Norm diagnostics should canonicalize to one center and use its tensor norm;
  do not replace this with a global doubled-network contraction.
- Local non-unitary scale control reuses an authoritative singleton center
  inside the active span, collapsing only a genuinely broad center, and adds
  the removed base-10 scale to `p.exponent`.
- Every rebuilt/replaced live MPS needs its known canonical span recorded.
- `set_p()` starts a new fidelity interval. Manual normalization and layout
  changes preserve cumulative fidelity but must invalidate/rebase raw unitary
  norm scalars; normalization must also restore a valid one-site center at its
  insertion site.
- Temporary target copies (`p.copy()`) must use isolated metadata and must not
  overwrite the live `info_c` dictionary.
- Dense SRC/random FIT guesses own their active data, so rollback may retain
  the untouched original MPS. If the actual guess aliases the live state,
  isolate rollback before FIT; native warm-starts still copy before mutation.
- FIT may consume an optimizer-owned guess/target to remove redundant copies,
  but mixed mode must still isolate the complete trial before mutation.
- Mixed in-place trial copies and commits must preserve each tensor's
  `left_inds` isometry metadata as well as its native array data.
- After FIT, reuse its final center and norm for stabilization and canonical
  metadata instead of scanning or canonicalizing the active interval again.
- Unitary one-site gates preserve the existing center. Do not replace the
  cache with the gate site. MPO/SVD/swap compression should reuse the
  one-site center left by the backend and read that tensor norm directly;
  collapse a span only when the backend genuinely leaves a broad center.
- Single-gate DMRG target norms should use the canonical dense/native local
  expectation before contracting the already-materialized FIT target. Batch
  fallback work belongs in the `infidelity.target_norm` timing stage.
- Preserve unclipped norm-ratio diagnostics. A significant retained-norm
  overshoot is a broken orthogonal-projection invariant and must raise when
  diagnostic validation is enabled with `finite_check=True`.

After a mutation that invalidates canonicality, either canonicalize explicitly
or invalidate the cache before any canonical-only operation. Unitary one-site
gates preserve the isometry structure; non-unitary one-site gates need explicit
recanonicalization.

Do not transfer diagnostic semantics from `MpsOptimizer` to
`MpsStabOptimizer`. The stabilizer coefficient-MPS simulator has its own sparse
normalized-unitary norm-loss contract; read
`.github/skills/stabilizer-tensor-networks/SKILL.md` before changing it.

`TrajectoryEvent` Kraus branches are normalized physical-state updates. The
trajectory runner applies the selected branch with `non_unitary=True`, calls
`normalize()`, and clears `p.exponent` so the next gate sees a truly normalized
MPS. This does **not** turn ordinary-MPS `track_infidelity` (a local normalized
overlap diagnostic) into the STN norm-loss proxy: keep the contracts separate.

## Backend rules

Use Quimb and Autoray APIs. Preserve Symmray arrays and route symmetric target
gates through backend-compatible split/auto-swap paths; never force a dense
NumPy identity into a Symmray contraction. Keep optional Symmray coverage
guarded with `pytest.importorskip("symmray")`.

The MPS gate-stream backend boundary is explicit and single-pass:

- `MpsOptimizer` requires user-supplied gates and sub-MPO tensor data to use
  the same array backend and device as the state. Non-NumPy payloads must also
  match the state dtype because their contractions reject mixed dtypes;
  dense NumPy-to-NumPy dtype promotion remains compatible. Backend/device
  mismatches, and incompatible dtypes, raise a location-specific `TypeError`.
- Constructor, `set_gates`, and `add_gates` normalize a stream once, validate
  that normalized queue once, and install that same queue. `set_p()` validates
  the existing normalized queue against the replacement state before mutation.
  Do not normalize or validate the whole stream again for each replay segment.
- `_execute_mode` receives the already validated payloads and must dispatch
  directly. Do not add a per-segment conversion or `gate_stream.prepare` stage.
  Internal control operators remain the optimizer's responsibility, while
  library-generated trajectory outcomes are converted by the shot runner before
  calling `set_gates`.
- `MpsOptimizer.to_backend(...)` is an explicit caller helper for preparing a
  payload before installing it. Native Symmray gates must retain their charge
  and fermionic metadata; a dense gate cannot be generically promoted.

Native fermionic FIT environments must use graph-planned contraction directly
on the Symmray arrays so dummy-mode conjugate pairs and graded phases determine
the contraction order; do not replace this with an arbitrary pairwise loop or
a temporary `TensorNetwork`. Gate-window FIT keeps a conjugated native working
MPS, contracts real outside overlap environments, applies dual-leg corrections
before local writeback, and resolves odd dummy-mode global phases afterward.
This convention must honor `R`, `L`, and `RL` directly and may reuse compatible
environments across a direction reversal. A non-adjacent fermionic DMRG target
is warm-started by the same chi-capped native auto-swap replay, analogous to
deterministic sector enrichment, then follows the selected block-to-one-site
schedule normally.
`target_cutoff=0.0` may prune only structural zeros using the smallest positive
absolute cutoff; every representable nonzero singular value must remain.
Native FIT never replaces the current MPS with a target copy. Its graded local
SVD and chi-capped auto-swap algebra are responsible for opening compatible
charge sectors; if the current sectors and target are disconnected, raise a
clear disconnected-sector error rather than hiding the failure behind a dense
conversion or global warm start. Dense two-/three-site growth windows select
their disposable FIT `p` with `fit_init_strategy`: direct current MPS,
fixed-rank deterministic random perturbation, active-bond random expansion,
or isolated `guess-<method>` replay (default `guess-src`). The underscore
spelling remains accepted for compatibility. In `auto`, only active bonds below their attainable
physical/`chi` rank are expanded, and the exact gate target remains separate as
`p_g`. Native Symmray/fermionic paths keep their graded sector-growth route and
do not use dense random padding. If `run()` omits `cutoff_mode`, ordinary paths
use `rsum2` while MPO `dm` preserves Quimb's native `rsum1` default; an
explicit mode overrides it. Interior oversampled zipup and `fit-*` replay keep
the local sub-MPO partition and disable nested full-chain array permutation.

## Profiling contract

`finite_check=False` is the default in every mode. Tensor scans and scalar
non-finite/norm-consistency guards are optional diagnostics, not required
for normal optimization. Enabling them warns once per replay about their
cost; owned FIT calls share that warning. Convergence calculations and
explicit quality/overlap diagnostics remain independent.

Cache physical rank ceilings only within replay and invalidate after state,
cap, layout, mode, or external canonical-resynchronization changes; chi/length
changes also require new ceilings. Never cache changing active ranks as if
they were physical ceilings. Mixed replay prepares a FIT window once and
reuses its validated final maximum until a mutation or quality repair. Array
classification uses weak network identity and explicit inheritance through
backend-preserving copies, not first-array-type equivalence between networks.

`run(timing=False)` performs no replay profiling clock reads, accelerator
barriers, or timing-record allocation; mixed summaries leave elapsed time
unset on that path. Enabled timing must remain observational: do not turn on
FIT split diagnostics or another numerical/metadata path merely because clocks
are active. Move optimizer-owned FIT records into the run collector and keep
the defensive deep copy at `get_run_timing()` rather than copying every nested
site record during replay. Stage and FIT totals are hierarchical and contain
documented compatibility overlaps, so never present every field as one
additive total. Resolve synchronized accelerator timing once per session, and
for JAX wait on each actual stage result rather than an older MPS leaf.
For MPO/SVD/swap diagnostics, separate non-unitary target-norm work, the
retained one-center norm read, and scalar log-fidelity bookkeeping in timing
records. In SVD mode measure the non-unitary target before the routed gate
split so both that cutoff and the final chi compression contribute to reported
loss.

## Implementation review checklist

Before changing `optimizer.py`, verify the following ownership boundaries:

- `_normalize_mode` may canonicalize public aliases, but the DMRG alias must
  remain available to select its schedule.
- `_execute_mode` receives backend-prepared payloads and dispatches only gate
  or sub-MPO events. `_run_segmented` owns control-event boundaries.
- DMRG builds an isolated exact target and selects an owned guess or a direct
  live-state guess. Rollback retains the original only for an isolated dense
  guess; direct/native paths require a copy. A fallback must restore both
  tensor data and `info_c` before direct MPO replay.
- `_apply_measure_event` computes the Born probability before any frame
  localizer and records compression survival separately. A dense multi-site
  projector should remain a bond-two sub-MPO; native graded states retain
  their metadata-safe route.
- `info_c["cur_orthog"]`, `p.exponent`, and raw unitary norm baselines are
  updated together after every replacement, normalization, layout reorder, or
  control event.
- `where` has one meaning per layer: logical stream locations at the public
  API, mapped physical locations during execution, and logical locations again
  in user-facing measurement/feed-forward records.

## Validation

Activate the shared environment first:

```bash
source /Users/rezah/envs/genpy/bin/activate
```

Focused optimizer validation:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/test_optimize_mps.py \
  tests/test_optimize_mpo.py \
  tests/test_symmetric_tensors.py
```

Also run `tests/test_public_api.py`, `tests/test_package_layout.py`, and
`tests/test_sampler.py` for public API/readout changes. Add regression tests
for layout persistence, logical sample/dense remapping, entangled reorder
errors, exact-mode transitions, no-rescan center reuse, exponent accounting,
and control-event bookkeeping.
