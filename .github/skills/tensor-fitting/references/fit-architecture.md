# FIT architecture and review checklist

## File map

- `src/pepsy/fitting/local.py`: `FIT`, cached environments, one-/two-site
  active-window sweeps, native split, convergence, and timing.
- `src/pepsy/optimizers/mps/optimizer.py`: public FIT controls, target/layer
  construction, transaction fallback, infidelity, stabilization, and profiling.
- `src/pepsy/optimizers/mpo/optimizer.py`: MPO use of the shared FIT class.
- `src/pepsy/sampling/samplers.py`: full-chain `run_eff` consumer.
- `docs/api/optimizers/mps.md`: public behavior and algorithm choices.
- `tests/test_optimize_mps.py`: dense correctness, timing, growth, and
  complex64 regression coverage.
- `tests/test_symmetric_tensors.py`: native Z2/U1/U1U1 and fermionic coverage.

## Public controls

- `fit_block_size=2`: recommended two-site local wavefunction and native SVD.
- `fit_block_size=1`: fixed-rank compatibility update.
- `fit_sweep_sequence="RL"`: alternating left-to-right/right-to-left sweeps.
- `fit_min_iter`, `fit_rtol`, `fit_patience`: mode-neutral adaptive stopping
  controls for DMRG and mixed DMRG. Patience counts same-phase norm samples,
  so the default value 2 represents one stable comparison; legacy
  `mix_fit_*` names are deprecated.
- `fit_layer_size=N`: number of circuit gates in one target block; compatibility
  alias `k_2q_batch` remains.
- `target_cutoff=0.0`: target construction cutoff.
- `fit_target_strategy={"auto", "layered", "mps"}`: lazy exact gate layers
  for ordinary arrays versus a materialized/native-routed target MPS.
- `fit_init_strategy={"auto", "direct", "random", "random_expand", "guess_<method>"}`:
  disposable FIT initial guess policy, defaulting to `guess_src`;
  `fit_init_seed` makes random policies reproducible without a global backend
  RNG.
- `fit_init_rand_strength`: scale of deterministic random initialization.
- `fit_single_pair_fast_path=True`: opt-in one update for an adjacent active pair;
  named `dmrg2` enables this automatically.
- `cutoff`, `cutoff_mode`, `chi`: output split/truncation controls.
- Quimb guess methods follow the native 1D registry, including `direct`, `dm`,
  `zipup`, SDC/SRC/SRCMPS and their oversampling variants, `fit`,
  `fit-zipup`, `fit-projector`, and `fit-oversample`. Interior nested methods
  preserve local sub-MPO tags by disabling the inner full-chain permutation.
- `stabilize_unitary=True`: restore raw norm after recording compression loss
  for DMRG/FIT, mixed MPO compression, and standalone MPO/swap/perm/SVD modes,
  preventing deep complex64 underflow. Sampling and stabilization are
  independent controls. `fit_stabilize_unitary` remains a deprecated alias.
- `environment_strategy={"auto", "mps-direct", "symmray-native", "generic"}`
  on `FIT`: dense MPS specialization, native Symmray chain contraction, or the
  general conservative route.
- `timing_sync_device=True`: opt-in accelerator barriers for kernel-complete
  profiling; two-site timings include effective/SVD/writeback/environment.
  Resolve the accelerator once, wait on actual JAX stage results, and keep
  timing independent of `collect_split_diagnostics`.
- `local_norm_trace` stores one terminal retained-center scalar per completed
  sweep. With `finite_check=True`, active backend leaves are reduced natively
  and transferred with the optional rtol norm in one compact vector.

## Algorithm map

For local circuit compression, `MpsOptimizer` builds an exact target,
constructs `FIT(target, p=current, range_int=[xmin, xmax], inplace=True,
copy_target=False)`, then calls `run_gate`. Ordinary dense targets default to
small spatially split gate tensors layered over one owned MPS copy. The
optimizer selects a disposable FIT initial state: direct current MPS,
fixed-rank deterministic random perturbation, active-bond random expansion,
or isolated `guess_<method>` replay (default `guess_src`). `auto` also selects
SRC, including on saturated bonds; the exact target remains separate.
FIT never installs the exact target as `fit.p` and never performs a global
target warm start. This avoids intermediate target-MPS rank growth in the
normal objective path. Symmray
uses its native auto-swap MPS target until graded layered targets have an
independently validated tag/phase contract. Left/right overlap environments
project the target onto the fixed
outside MPS. A two-site update contracts both target site tensors with those
environments, yielding the two physical groups and two outer virtual legs.
`Tensor.split` truncates only the middle bond and absorbs singular values in
the sweep direction.

During local expansion, the optimizer first prepares the selected disposable
FIT guess, while the exact target remains unchanged. FIT then performs the
ordinary variational effective tensor and SVD from that guess. Native FIT uses
its graded local sector rules and chi-capped auto-swap algebra for compatible
sector growth; it never receives dense random padding.

Fresh gate sweeps build fixed environments only beyond the first active
block. Completed block sweeps retain the minimal cumulative boundaries needed
by an equal-size reversed sweep. Immediately before a reversed one-site
transition, FIT extends that cache through one terminal tensor for a two-site
producer or two terminal tensors for a three-site producer; it never rebuilds
the complete fixed side. A 3-to-2 transition similarly extends through one
terminal tensor and marks the cache ready for a reversed two-site sweep.
A terminal single-pair fast path needs no
active-window environment. Layered targets
cache boundary index names discovered from neighboring site tensors rather
than scanning the global target index map; this cache owns no tensor data.
The private `_SweepEnvironmentCache` keeps each completed sweep's boundary
mapping, direction, and block size together. It retains the mapping by
reference and performs compatibility checks once per sweep; update kernels
continue to use direct dictionary lookups with no wrapper in their hot loops.

An active interval containing one adjacent pair reaches its complete local
optimum after that split. Named `dmrg2` enables the single-pair fast path even
when tolerance stopping is disabled. Other modes require
`fit_single_pair_fast_path=True` for this shortcut.
After any final sweep, FIT's retained norm and center tensor are authoritative
for infidelity and unitary stabilization; recanonicalizing the interval is
redundant. Non-unitary scale control likewise normalizes that singleton center
in place when it remains inside the active interval; it must not move a valid
left endpoint to the right endpoint merely to extract the same norm.

For `dmrg1`, inspect the active and full-chain attainable rank targets before
starting FIT. An already-capped window starts with one-site updates. An
under-capacity non-adjacent window requires at least three requested sweeps:
two two-site growth sweeps followed by at least one one-site refinement sweep.
The two-site phase is bounded at two sweeps; it does not extend because of
rank stagnation. Once every full-chain bond reaches its physical/``chi``
ceiling, the optimizer latches one-site updates for later windows in the same
replay. Named `dmrg2` and `dmrg3` are fixed warm-up schedules: they perform
exactly `fit_adaptive_sweeps` two- or three-site sweeps (two by default).
`dmrg3` then performs one two-site transition sweep. Both spend the remaining
`n_iter` budget on one-site refinement subject to
`fit_rtol`. Generic `dmrg` remains available for rank-adaptive block
scheduling.

Standalone `FIT.run_gate` defaults are eight sweeps, block size two, RL
directions, two adaptive block sweeps, automatic dtype-aware tolerance,
minimum two sweeps, patience two, and disabled split diagnostics/finite scans.
With `finite_check=False`, scalar non-finite convergence checks are disabled
too; norm calculations and convergence comparisons still run. MpsOptimizer
scopes tensor and scalar-norm validation to its opt-in `finite_check` flag
across every mode, including mixed commit checks. These are optional
diagnostics, not required for normal optimization. MpsOptimizer warns once
per replay when enabled; its owned FIT calls suppress only the duplicate
warning, while standalone FIT still warns. Explicit quality and
overlap diagnostics remain separate opt-ins.
`two_site_transition_sweeps=1` applies to three-site fits; MpsOptimizer enables
it only for named `dmrg3`. All block-size changes reset convergence history.

`run_eff` is a separate global full-chain fit used by boundary/sampling code.
Do not substitute it for the gate-window solver.
`run` and `run_eff` retain fixed-sweep behavior; PEPS boundary diagnostics
describe them as `fixed_sweeps` and use only coarse opt-in elapsed timing.

FIT timing records contain both compatibility totals and their named subsets.
`canonicalization_seconds` includes preparation and moving canonicalization;
legacy `environment_seconds` includes the complete post-writeback phase. Do
not sum every timing field. MpsOptimizer owns its temporary FIT instances, so
it moves their records into the replay collector and copies only at the public
getter boundary.

## Native tensor rule

Quimb and Symmray own contraction order, dual indices, fusion metadata, dummy
modes, graded signs, and block SVD. Do not convert native arrays with
`np.asarray`, `ar.to_numpy`, or `.to_dense()` in the solver. Host conversion is
allowed only for bounded diagnostics after native scalar reduction.

Validate at least one spinful `U1U1FermionicArray` result against a native MPO
reference, not merely for finite values.

## Literature boundary

- Stable FIT basis: Ayral et al., PRX Quantum 4, 020304 (2023),
  <https://doi.org/10.1103/PRXQuantum.4.020304>.

## Focused validation commands

Run with `source /Users/rezah/envs/genpy/bin/activate` first.

```bash
python -m pytest -q -m '' tests/test_optimize_mps.py
python -m pytest -q -m '' tests/test_symmetric_tensors.py -k 'mps_optimizer and (dmrg or two_site_fit)'
python -m pytest -q -m '' tests/test_optimize_mpo.py
python -m ruff check src/pepsy/fitting src/pepsy/optimizers/mps tests/test_optimize_mps.py tests/test_symmetric_tensors.py
```

For performance work, compare `environment_strategy="mps-direct"` with
`"generic"` on identical inputs and verify numerical equivalence before
claiming speedup. Compare `fit_target_strategy="layered"` and `"mps"` by exact
dense equality on small ordinary problems. Use `run(timing=True)` and inspect
`dmrg.target`, `dmrg.fit`, `dmrg.stabilize`, FIT call/record indices,
directions, pair phases, and failed partial sweeps. On an asynchronous GPU,
repeat with `timing_sync_device=True` before attributing time to a stage.
