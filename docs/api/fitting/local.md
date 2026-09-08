# `pepsy.fitting.local`

`FIT.run_gate(finite_check=False)` skips per-sweep active-array finite scans
and scalar non-finite detection by default. These optional diagnostics are
not required for normal optimization. Enabling `finite_check=True` (or a
custom checking callback) emits a warning explaining the extra validation
work and possible accelerator synchronization. `MpsOptimizer` warns once per
replay and suppresses duplicate warnings from its owned FIT calls.
The optional scan is independent of timing and the
terminal scalar norm used for convergence. `MpsOptimizer.run(finite_check=...)`
forwards this policy to its FIT calls, including measurements and shot replay.

`FIT(target, p=guess, ...)` variationally fits an open-boundary MPS or MPO
guess to a target tensor network. There are three sweep entry points:

- `run()` is the simple full-contraction reference.
- `run_eff()` is the cached full-chain solver used by the default
  boundary/sampling path; it defaults to one-site updates and also supports
  opt-in native two- and three-site updates.
- `run_gate()` is the cached active-window solver used by MPS/MPO circuit FIT
  and by `fit_mode="two-site"` boundary contraction over the full interval.

Choose the entry point by the scope of the fit:

| Entry point | Scope | Typical consumer |
| --- | --- | --- |
| `run()` | Full chain, simple reference contractions | Debugging and compatibility checks |
| `run_eff()` | Full chain, cached one-, two-, or three-site environments | Boundary and sampling workflows |
| `run_gate()` | `range_int` only, one-, two-, or three-site updates | MPS/MPO circuit compression |

The distinction is deliberate: `run_eff()` must not replace `run_gate()` for a
local gate target, because refitting sites outside the active interval changes
the circuit-compression algorithm. The FIT implementation follows the same
high-level order in each path: own and validate the target/state, prepare
effective environments, update the requested sites, then record optional
fidelity or timing diagnostics.

Tree-shaped states use the companion `pepsy.fitting.TreeFIT` class. It keeps
the same target/guess ownership model and `run`/`run_eff`/`run_gate` vocabulary,
but caches a directed overlap environment for every tree edge and moves the
canonical centre along tree geodesics. Connected one-, two-, and three-node
local blocks are supported; untouched branch messages survive local updates.
The tree optimizer wrappers select it with `mode="dmrg"`, `"dmrg1"`, `"dmrg2"`,
or `"dmrg3"`.

```python
from pepsy.fitting import TreeFIT

fit = TreeFIT(target_tree, guess_tree, max_bond=chi, cutoffs=1e-12)
fit.run_gate(
    active_span,
    n_iter=4,
    block_size=2,
    adaptive_block_sweeps=2,
    sweep_sequence="RL",
)
updated = fit.p
diagnostics = fit.fit_diagnostics(overlap=True)
```

TreeFIT keeps the target fixed and does not normalize it. After each sweep it
reads one norm from the terminal canonical-centre tensor, just as MPS FIT does;
`local_norm_trace` and `sweep_norm_trace` contain those values, while
`local_norm_stripped_trace` preserves each `(mantissa, exponent)` pair. Its
`local_fidelity` is the clipped squared ratio of retained centre norm to the
target norm, and `local_infidelity` is the matching `1 - local_fidelity` value.
This is a local norm-survival diagnostic, not a directional target overlap.
`fit_diagnostics(overlap=True)` additionally performs the expensive full
target contraction and reports its genuine normalized value as
`target_fidelity`/`target_infidelity` (also available as the MPS-compatible
`fit_overlap_fidelity`/`fit_overlap_infidelity`); it never replaces
`local_fidelity`.
Optimizer-level cumulative compression fidelity remains the canonical
norm-survival ledger, accumulated in log form so large MPO/TreeMPO scale
factors do not overflow or get mistaken for discarded weight. `TreeFIT.run()`
is the FIT-compatible full-tree convenience call and delegates to the cached
`TreeFIT.run_eff()` engine;
`run_gate(region, ...)` remains the active-span entry point used by the
optimizers. TreeFIT accepts fused targets and correctly tagged layered targets.
Every target tensor must belong to exactly one structural node group; local
layer bonds stay within a group, while one or more bonds between groups must
follow the fitted tree edges. Ambiguous or untagged layer tensors are
rejected. The separate operator-state two-layer target can still use the
path-only `TreePeps` `sdc`/`src`/`zipup` compressor.

The tree optimizers construct their DMRG/DMRG-alias targets in this layered
form: the state and operator tree bonds remain separate, with only the
operator input and output physical legs joined. Fused targets remain accepted
for callers that already have one, and the direct path compressor retains its
own two-layer route.

`adaptive_block_sweeps=2` gives the same warm-up/refinement schedule as the
MPS FIT engine: two- or three-node updates for the warm-up, followed by
one-node sweeps. `adaptive_until_rank=True` keeps the larger block until the
active physical rank ceilings are reached. Tree optimizer `dmrg1` and `dmrg2`
use two-node warm-up blocks, while `dmrg3` uses three-node blocks; each named
mode then refines with one-node updates.

`retag=True` aligns structural node tags on the copied target with the fitted
tree while preserving tensor order and physical/site tags. Layered targets use
the same structural tags to assign every layer tensor to its node group.
`copy_target=False`
is available when an optimizer has created a disposable target and can transfer
ownership safely.

For circuit compression, set `range_int=(xmin, xmax)` and use:

```python
fit = pepsy.FIT(
    target,
    p=current,
    range_int=(xmin, xmax),
    cutoffs=1e-12,
    environment_strategy="auto",
    # Keep the safe default for caller-owned targets. MpsOptimizer transfers
    # its fresh disposable target with copy_target=False.
    copy_target=True,
)
fit.run_gate(
    n_iter=4,
    block_size=2,
    sweep_sequence="RL",
    max_bond=chi,
    cutoff=1e-12,
    cutoff_mode="rsum2",
    single_pair_fast_path=True,
    collect_split_diagnostics=False,
)
compressed = fit.p
```

`block_size=2` is recommended for the usual DMRG compression. Each update
forms the two-site wavefunction with two outer virtual legs and both physical
groups, then calls Quimb `Tensor.split` across the middle bond. This permits
active rank growth and dispatches natively for dense NumPy/Torch/CuPy and
Symmray U1/U1xU1 fermionic arrays. `block_size=3` forms a three-site
wavefunction and performs two direction-aware native SVD splits, while
`block_size=1` retains the fixed-rank compatibility update. Three-site FIT is
useful when a larger local window is worth the extra SVD cost; it is not a
dense `from_dense` conversion.

For two- or three-site FIT on an active window spanning at least three sites,
`final_one_site_sweeps=1` adds a fixed-rank one-site polish pass after the
block sweeps. The pass reuses the canonical window and never touches sites
outside `range_int`; it is skipped for a two-site window. This is an explicit
direct-`FIT.run_gate` control. `MpsOptimizer` uses its separate adaptive
`fit_adaptive_sweeps`/rank-ceiling schedule and does not add this legacy polish
pass automatically.

Direct `FIT.run_gate()` defaults to `n_iter=8`, `block_size=2`,
`sweep_sequence="RL"`, `adaptive_block_sweeps=2`, `min_iter=2`,
`rtol="auto"`, and `patience=2`. Automatic tolerance is `1e-3` for 16-bit,
`1e-5` for float32/complex64, and `1e-9` for higher precision.
Split diagnostics default to `False`; callers that need them must enable
`collect_split_diagnostics=True`. `rtol=None` requests fixed sweeps.
Standalone FIT does not classify the gate as unitary or non-unitary; unlike
MpsOptimizer's non-unitary replay policy, it always resolves `rtol="auto"`
to a numeric tolerance.
With `block_size=3`, `two_site_transition_sweeps=1` inserts one two-site sweep
after the three-site phase, before one-site refinement. Set it to zero for
the previous direct handoff. The transition consumes the same `n_iter` budget.
The legacy `three_site_sweeps` control applies when `adaptive_block_sweeps=None`.
With
`adaptive_until_rank=True`, the block phase continues until all active bonds
reach their physical ceilings; rank stagnation is deliberately not an early
exit. Remaining requested sweeps use one-site FIT. One-site refinement
preserves the bond dimensions opened by the larger block and is cheaper than
repeating the larger SVD block.

The MPS optimizer passes `adaptive_block_sweeps=fit_adaptive_sweeps` and enables
`adaptive_until_rank` only for eligible generic `dmrg` windows.
Before constructing a `dmrg1` fit, the optimizer checks the active
attainable bond ceilings: an already-capped window starts with one-site FIT,
while an under-capacity non-adjacent window requires `n_iter >= 3` for two
two-site growth sweeps and at least one one-site refinement sweep. `dmrg2`
and `dmrg3` use exactly the configured two- or three-site block warm-up (two
sweeps by default); `dmrg3` adds one two-site transition sweep. Both then
refine with one-site FIT. The direct FIT
diagnostics `adaptive_sweeps_run` and `one_site_sweeps_run` count both
scheduled block sweeps and any explicit `final_one_site_sweeps` polish passes.
`adaptive_sweeps_run` includes the two-site transition: a completed default
DMRG3 warm-up therefore contributes three block sweeps in total. Timing
records expose each sweep's individual block size when timing is enabled.

For tolerance-controlled `run_gate`, `patience` counts same-phase retained-norm
samples, not norm differences. Thus `patience=2` needs two comparable
one-site samples and stops after their first stable relative change, subject
to `min_iter`. A phase change from block growth to one-site refinement resets
the convergence window. Each completed sweep reduces only its terminal
canonical-center norm; intermediate update norms are superseded by later
updates and are not computed. Consequently `local_norm_trace` contains exactly
one backend scalar per completed sweep, and `sweep_norm_trace` contains the
host values actually used by tolerance stopping.

`finite_check=True` reduces every dense tensor or native Symmray block in the
active interval to backend finite-status scalars. Those flags and the optional
rtol norm are transferred as one compact vector per sweep. A callable keeps the
general user-defined state-check behavior. Reusing one `FIT` instance clears
per-run norm/fidelity traces and split diagnostics before the next invocation;
`run`, `run_eff`, and `run_gate` all require a positive integer `n_iter`.
With finite scans disabled, tolerance stopping transfers just the terminal
norm scalar without allocating a stacked diagnostic vector. It retains the
same convergence calculations, but non-finite scalar detection is also
disabled by default. `finite_check=True` enables both tensor and scalar
checks; reads required for convergence are not deferred during warm-up.

Ordinary dense arrays and native bosonic or fermionic Symmray arrays reuse the
compatible partial overlap environments produced by the preceding
opposite-direction sweep. Fermionic FIT keeps the working state conjugated
across the complete sweep sequence, so the reused environments retain one
dual-leg convention.
A block sweep retains only the boundaries needed by another reversed sweep of
the same size. If the next reversed sweep changes to one-site refinement, FIT
extends that cache through exactly one terminal tensor after a two-site sweep,
or two terminal tensors after a three-site sweep. Both 2-to-1 and 3-to-1
transitions therefore avoid a complete fixed-side rebuild without constructing
unused terminal environments during block warm-up. The 3-to-2 transition
extends through exactly one terminal tensor and reuses the resulting cache.
Fresh sweeps construct only
the fixed boundaries that their active block can query. Explicitly generic or
mixed-backend bosonic Symmray fits retain the conservative rebuild policy;
automatic/native Symmray fits use the audited zero-copy cache.

The same native block updates are available for the full-chain path:

```python
fit.run_eff(
    n_iter=4,
    block_size=3,
    sweep_sequence="RL",
    max_bond=chi,
    cutoff=1e-12,
)
```

`run_eff(block_size=1)` remains the default compatibility path. All block
sizes honor the requested sweep sequence, which defaults to `RL` (left to
right, then right to left). Dense and native bosonic one-site sweeps reuse the
same opposite-direction environment cache as the block paths; conservative
mixed-backend and fermionic fixed-sweep compatibility routes retain their
audited rebuild behavior. The block-2 and block-3 variants visit the complete
chain, reuse cached environments, and grow only bonds reached by their native
SVD splits. To use DMRG-style block growth
followed by cheaper one-site refinement, set `adaptive_block_sweeps=2` (or
another positive count):

```python
fit.run_eff(
    n_iter=6,
    block_size=2,
    adaptive_block_sweeps=2,
    sweep_sequence="RL",
    max_bond=chi,
    cutoff=1e-12,
)
```

The first two sweeps use two-site SVD growth and the remaining sweeps use
one-site FIT. `block_size=3` behaves the same way with three-site updates.
The block-to-one-site transition extends only the terminal cached boundaries
needed by the reversed one-site sweep. Optional `rtol`, `min_iter`, and
`patience` controls use the terminal retained-center norm; `rtol=None` keeps
fixed-sweep behavior and adds no diagnostic transfer. When `rtol` is enabled,
`run_eff` requires at least two completed sweeps and defaults `min_iter` to 2,
so the first stopping comparison is always between two retained norms.
Detailed per-site timing remains a `run_gate(timing=True)` feature.

For an interval containing exactly one neighboring pair,
`single_pair_fast_path=True` marks structural convergence after one update:
the effective tensor and its SVD solve the entire active problem, so another
sweep only rebuilds the same environments. That terminal update constructs no
active-window environments; native fermionic outside-window environments stay
intact. The default is `False` on direct
`FIT.run_gate` calls and in `MpsOptimizer`, preserving fixed-sweep
compatibility. Set it to `True` to make named `dmrg1`, `dmrg2`, and `dmrg3`
windows of two sites perform one two-site update and advance to the next gate
without one-site refinement; `n_iter` and tolerance controls cannot add a
second sweep while the fast path is enabled. `collect_split_diagnostics=False`
omits per-SVD truncation dictionaries when only the fitted state and retained
norm are needed. `MpsOptimizer` additionally keeps the named `dmrg2`
nearest-neighbor schedule at one update by default.

`sweep_sequence` uses Quimb direction names: `"R"` is left-to-right, `"L"` is
right-to-left, and `"RL"` alternates. Native fermionic `run_gate` executes the
requested sequence exactly and records the conjugated fitting convention in
`info["fermionic_sweep_sequence"]`. It canonicalizes once around the first
sweep center, contracts the real outside overlap environments rather than
substituting graded boundary identities, applies Symmray's dual-leg phase
correction before each local writeback, and resolves odd dummy-mode global
phases afterward. The physical ket is restored on both success and failure.
The same convention supports one-, two-, and three-site native `run_eff`
sweeps.

FIT never replaces its live `p` with a target copy. If dense MPS gate replay
needs rank growth, the caller can prepare a disposable copy of the current MPS
with small random entries on the active bonds below their attainable rank and
pass that copy as `p`. FIT then uses that state as its ordinary variational
initialization while the target remains the exact variational objective.
Native Symmray fits retain their graded local SVD/auto-swap sector rules and
should use native sector growth rather than dense random padding.

`environment_strategy="auto"` selects
`"mps-direct"` for an ordinary dense one-tensor-per-site target,
`"symmray-native"` when all target and fitted tensors are Symmray-backed, and
otherwise uses the general `"generic"` route. Non-fermionic Symmray inputs
use the native blockwise chain product; fermionic Symmray inputs stay on the
resolved native strategy but use Quimb's graph-planned direct tensor
contraction so contraction order, dummy modes, and graded phases remain
authoritative. It dispatches directly on the Symmray arrays and does not build
a temporary TensorNetwork. Neither route densifies the tensor arrays. The
explicit settings are mainly useful for profiling and regression comparison.
For a layered target, FIT resolves each active-window boundary bond by
inspecting tensors on the two neighboring site tags and caches the resulting
index name. It does not rescan the complete target index map during local
environment updates; no tensor data or backend array is copied by this cache.

`cutoff="auto"` chooses `1e-3` for 16-bit data, `1e-6` for 32-bit/complex64
data, and `1e-12` for 64-bit data. Numeric cutoffs retain their explicit
behavior.

When consecutive sweeps reverse direction, `run_gate()` reuses the canonical
form produced by the preceding block update instead of repeating the boundary
canonicalization pass. The first sweep and consecutive same-direction sweeps
still prepare their required gauge explicitly.

With `timing=True`, `get_timing()` returns completed and failed partial sweep
records, including a `timing_schema` version, direction, block size, active
window size, update count, environment strategy, block/site times, and
convergence status. Timing schema 3 adds sweep-level
`canonicalization_seconds`, `sweep_preparation_canonicalization_seconds`,
`fixed_environment_seconds`, `moving_canonicalization_seconds`,
`moving_environment_seconds`, and `sweep_overhead_seconds`. Every update also
reports `effective_seconds`, `svd_seconds`, `writeback_seconds`,
`canonicalization_seconds`, and `moving_environment_seconds`; the legacy
`environment_seconds` remains the complete post-writeback phase, and one-site
updates report `svd_seconds=0.0`. The sweep record aggregates each stage as
well as `elapsed_seconds`, making one-, two-, and three-site runs directly
comparable in benchmark output. Block records break out effective contraction,
SVD, writeback/terminal-norm, canonicalization, and moving-environment time. These
fields intentionally include named subtotals and compatibility overlaps:
`canonicalization_seconds` equals preparation plus moving canonicalization,
and `environment_seconds` includes moving canonicalization as well as the
moving environment. Do not sum every reported field. The non-overlapping
partition uses preparation canonicalization, fixed environments, effective
contraction, SVD, writeback, moving canonicalization, moving environments,
and sweep overhead once each.

Timing is independent of `collect_split_diagnostics`: clock recording does not
turn on native SVD truncation metadata. Add
`timing_sync_device=True` for device-complete Torch CUDA, CuPy, or JAX timing;
the backend is detected once, CPU data becomes a no-op, and JAX waits on the
new effective/split/writeback results associated with each timing boundary.
Normal runs never pay for clocks, timing records, or synchronization barriers.
FIT also exposes
`final_center_site`, `final_norm`, `final_direction`, and
`convergence_reason`, allowing an optimizer to reuse the known canonical
center without a redundant sweep.

Those adaptive stopping and detailed timing fields belong to `run_gate()`. The
`run()` and `run_eff()` solvers remain fixed-sweep numerical paths and are not
silently changed by the gate-window controls. PEPS boundary results describe
them as `convergence_reason="fixed_sweeps"` and can collect one coarse elapsed
time per boundary fit without altering either solver's update sequence.
