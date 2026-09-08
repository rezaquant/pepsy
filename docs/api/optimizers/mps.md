# `pepsy.optimizers.mps.optimizer`

`MpsOptimizer` defaults to `mode="direct"`: Quimb's direct compression to
the requested `chi`. Prefer this algorithm name in new code. `mode="mpo"`
remains a silent compatibility alias for the same path; an MPO is an operator
representation, not a distinct replay algorithm.

```python
opt = pepsy.MpsOptimizer(state, gates, chi=64)  # mode="direct"
opt.run()
```

The previous constructor default was `"dmrg"`. Specify `mode="dmrg"` (or a
named `dmrg1`/`dmrg2`/`dmrg3` schedule) to retain variational FIT replay.

For finite-temperature preparation from Hamiltonian terms, see the dedicated
[`GibbsMps` guide](gibbs_mps.md). It uses an interleaved physical/ancilla MPS
and delegates non-unitary gate replay to this optimizer.

## Torch SVD policy

Torch/Autoray SVD dispatch is process-global, so configure it once at
application startup or use a scoped policy for an experiment:

```python
import pepsy

svd_policy = pepsy.TorchLinalgConfig(
    mode="complex",
    stabilized=False,       # native Torch forward and backward
    svd_driver="gesvd",    # CUDA: Pepsy's exact fast default
    cpu_svd="torch",       # CPU: Torch's native LAPACK path
    svd_fallback="auto",   # no fallback for native mode
)
svd_policy.register()
print(svd_policy.describe())
```

`svd_driver` applies only to CUDA and accepts `"auto"`, `"gesvdj"`,
`"gesvda"`, or `"gesvd"`. Pepsy defaults to the exact `"gesvd"` route;
`"auto"` and `"gesvdj"` remain explicit alternatives. `gesvda` is approximate and requires
`allow_approximate=True`. `cpu_svd` accepts `"torch"`, `"scipy_gesdd"`, or
`"scipy_gesvd"`; the SciPy choices are intended for explicit forward-only
CPU experiments when `stabilized=False`, or for stabilized autodiff when
`stabilized=True`. `svd_fallback="auto"` means no fallback for native mode
and SciPy `gesvd` for stabilized mode.

The non-approximate choices are CUDA `gesvdj` and `gesvd`, plus CPU Torch,
SciPy `gesdd`, and SciPy `gesvd`. For `complex64`, benchmark CUDA
`gesvdj` against the default `gesvd` route on your hardware; on CPU,
benchmark `scipy_gesdd` against the native Torch path. The approximate CUDA `gesvda`
driver is never selected unless `allow_approximate=True` is passed, and the
policy exposes this decision as `policy.exact` and `policy.approximate`.

For example, an exact complex64-oriented CPU experiment is:

```python
pepsy.TorchLinalgConfig(
    mode="complex",
    stabilized=False,
    cpu_svd="scipy_gesdd",
).register()
```

On CUDA, select the exact Jacobi driver explicitly with
`svd_driver="gesvdj"` when it is faster on your hardware. These settings do
not change the tensor dtype; they change only the underlying SVD implementation.

For ordinary MPS simulation, native mode is the recommended default. The
regularized mode exists for finite SVD gradients and difficult autodiff
inputs, not as a faster forward SVD. A temporary policy restores the previous
one when the block exits:

```python
with pepsy.TorchLinalgConfig(
    stabilized=True,
    svd_fallback="scipy_gesvd",
).activated():
    run_differentiable_workflow()
```

Use `pepsy.get_torch_linalg_config()` to inspect the last Pepsy-installed
policy. `pepsy.reset_linalg_registrations(backend="torch")` restores native
Torch and Quimb split registrations.

`MpsOptimizer` consumes canonical bundled gate streams of the form
`[(gate, where), ...]`. It also accepts stabilizer-style symbolic entries
`("H", site)`, `("CNOT", control, target)`, and
`("rzz", angle, site_a, site_b)`, along with the matching one- and two-qubit
rotation forms. Symbolic names are resolved through Pepsy's standard gate
constructors before replay, so uppercase names are accepted. Pass
`to_backend=...` to convert those internally generated matrices before the
strict stream/backend check; if omitted, the converter is inferred from the
initial MPS. For example:

```python
import torch

backend = pepsy.backend_torch(dtype=torch.complex64, device="cuda")
state.apply_to_arrays(backend)
opt = pepsy.MpsOptimizer(
    state,
    [("H", 0), ("rzz", 0.19, 0, 1)],
    chi=64,
    mode="dmrg3",
    to_backend=backend,
)
```

Numeric matrix gates and sub-MPO payloads retain the explicit-preparation
contract described below. Bare Quimb compression names such as `mode="src"`,
`mode="zipup"`, and `mode="direct"` are accepted; they normalize internally
to `quimb-<method>`. The qualified `mode="quimb-<method>"` forms, direct alias
`mode="quimb"`, and legacy `mode="mpo-<method>"` / `mode="mpo"` spellings
remain supported. The bare name `fit` remains the DMRG alias, so Quimb's
`fit` compression method is selected as `mode="quimb-fit"`. Quimb's newer
successive deterministic compressors are available explicitly as
`mode="quimb-sdc"` and `mode="quimb-sdc-oversample"` when the installed Quimb
build provides them. These modes are opt-in and do not change existing
defaults.
These events represent already-factorized nonlocal operators:

```python
event = ("submpo", mpo, where)
# or
event = {"kind": "submpo", "mpo": mpo, "where": where}
```

`where` is a non-empty tuple/list of unique 1D MPS sites. The convenience
helper `MpsOptimizer.submpo_event(mpo, where)` builds the tuple form. In the
Quimb compression family these events are applied with `gate_with_submpo_` and
compressed to `chi`. DMRG also accepts multi-site sub-MPO events: it
canonicalizes the active region, aligns the MPO site tags, and keeps the
operator as a layered FIT target while using the DMRG SRC warm-up guess.
`svd`, `swap`, `perm`, `su`, and `exact` reject sub-MPO stream events; `mix`
retains its existing gate-oriented unitary path.

Modes that use canonical MPS metadata require an open-boundary MPS. A cyclic
MPS has a nontrivial loop environment, so no single tensor norm can equal its
global norm under the open-chain mixed-canonical identity. Such inputs are
rejected before optimizer state is mutated. `mode="exact"` does not consume
canonical metadata and can first contract a cyclic input; switching its
contracted result back to an MPS mode rebuilds an open MPS.

`MpsOptimizer.backend_info()` reports the backend, dtype, and device inferred
from every live MPS tensor; the same values are also available as the
state-derived `backend`, `backend_dtype`, and `backend_device` attributes.
Every numeric gate and every tensor in a sub-MPO is checked for matching backend and
device at construction, `set_gates`, `add_gates`, and `set_p`; non-NumPy
payloads must also match dtype, while NumPy-to-NumPy dtype promotion is
compatible. Symbolic gates are generated and converted internally as described
above. A
mismatch raises a `TypeError` with the stream location and preparation guidance;
`MpsOptimizer` does not silently copy or cast user payloads. Use the same
explicit converter used to build the state, for example
`gate = to_backend(gate)`, before passing the gate stream to the optimizer.
Library-generated trajectory outcomes are prepared by the shot runner before
they are installed. Native Symmray MPS data reports `backend="symmray"` and
includes `array_backend` for the underlying NumPy, Torch, or CuPy charge-sector
blocks. Dense payloads cannot be promoted to native Symmray gates because that
would lose charge and fermionic metadata; construct those gates with the
matching Symmray convention instead.

Canonical metadata and observable readout are deliberately separate. Internal
mid-circuit `measure`, `reset`, and Kraus paths pass the live `info_c` mapping
through Quimb's canonical routines, so moving the centre during state
evolution remains tracked. For post-run diagnostics, evaluate observables on
`opt.p.copy()` rather than on the live `opt.p`: Quimb's
`local_expectation_canonical` moves an MPS centre in place. If lower-level code
has intentionally touched the live state, call
`opt.sync_canonicalization()` before resuming a canonical-mode replay; it
re-discovers the centre, canonicalizes to one site, and refreshes
`info_c["cur_orthog"]`.

Streams may also include control events. `("measure", pauli, where[, outcome])`
collapses onto a Pauli eigenvalue and records `(pauli, where, outcome, prob)`.
For dense MPS states, a multi-site Pauli measurement uses a bond-two windowed
sub-MPO for the projector `(I + outcome * P) / 2`, so the measurement does not
form a dense `2**k`-by-`2**k` operator. In DMRG modes, that sub-MPO becomes an
exact lazy FIT target on the endpoint span and the normal `guess-src` warm-start
is reused. Native Symmray and fermionic states retain their metadata-safe dense
projector path.
`("reset", where[, basis])` resets each target to the `+1` eigenstate of
`basis` (`"Z"` by default, so the legacy form resets to `|0>`); the internal
measurement is not recorded. `("measure_reset", basis, where[, outcome])`
measures each target in `basis`, records the result, then resets it to the
`+1` eigenstate. The aliases `("mrx", where[, outcome])`, `("mry", ...)`, and
`("mrz", ...)` are accepted. `("cap", where, vec[, absorb])` contracts one
physical leg with `vec`, absorbs it into the selected neighbour, and shortens
the MPS by one site. Later events use the shortened chain's labels: sites
above the removed site shift down by one. Caps preserve the raw contracted
norm rather than normalizing or performing a partial trace. They invalidate
the next unitary segment's norm baseline without changing accumulated
compression fidelity; `cap_history` records the structural operation.

Classical feed-forward is represented by `("if", record, bit, action)` (the
aliases `"conditional"`, `"condition"`, `"feed_forward"`, and
`"feedforward"` are accepted). `record=-1` refers to the latest measurement;
nonnegative records are absolute measurement indices. Measurement eigenvalue
`+1` is computational bit `0`, and `-1` is bit `1`. The action is one ordinary
bundled gate, control event, or sub-MPO event, for example:

```python
stream = [
    (qu.hadamard(), 0),
    ("measure", "Z", 0),
    ("if", -1, 1, (qu.pauli("X"), 1)),
]
```

Mapping form accepts `kind`/`type`/`event`, `record`, `bit` (or `value`), and
`then` (or `action`). The same event is evaluated per noisy trajectory and per
coalesced leaf, so the selected action follows that shot's measurement record.
Conditional gates inherit the ordinary segment's validated replay options,
including `n_iter`, the named DMRG block schedule, FIT guess, sweep directions,
cutoffs, normalization, and diagnostic controls. Selected conditional controls
follow the same cap/reset/measurement handling as unconditional events.

### Shot-aware replay

`MpsOptimizer` also owns shot replay. The constructor snapshots the initial MPS
and raw stream, including `TrajectoryEvent` objects and stochastic entries such
as `("x_error", p, q)`. Ordinary `run()` calls retain the single-state return
value. A noisy stream is automatically sent through trajectory replay, and an
explicit `shots > 1` requests an ensemble for any stream:

```python
simulator = pepsy.MpsOptimizer(
    initial_mps,
    [("h", 0), ("x_error", 1e-3, 0), ("measure", "Z", 0)],
    chi=64,
    mode="direct",
)
result = simulator.run(shots=10_000, strategy="auto", seed=7)
```

Each trajectory starts from the constructor state, so repeated shot runs are
independent. `strategy="independent"` stores one optimizer per shot, while
`strategy="coalesced"` or `"auto"` shares deterministic prefixes and preserves
branch multiplicities in the returned `NoisyResult`. Use `run_kwargs={...}` for
ordinary single-trajectory replay options such as `progbar=False`.

Use `retain="all"` (the default) for final states plus replay metadata,
`retain="final"` for final states without concrete streams and records, or
`retain="none"` when only the shot count/side effects matter. The latter keeps
no optimizer states in the result and therefore cannot be used to evaluate
observables afterward. `dmrg2` is the normal variational production backend;
`direct` is the default compression path for explicit sub-MPO events, while
DMRG schedules retain multi-site sub-MPOs as layered FIT targets. `svd`, `swap`,
and the other DMRG schedules use the same
trajectory contract and should be benchmarked for the workload. `mix` and `su`
remain gate-oriented/unitary modes, while `exact` also supports state-dependent
Kraus branches by evaluating copied dense TensorNetwork leaves. Shot
replay uses a frozen persistent-layout template when one is installed, but
still requires a fresh identity-order optimizer for an already-permuted `perm`
state.

For MPI, use the same optimizer-level API by passing `mpi=True` (or an
explicit communicator). `workers="auto"` divides the available CPU allowance
across local MPI ranks, while `workers=1` forces serial execution inside each
rank. `progress="auto"` shows one aggregate rank-zero shot bar only in an
interactive terminal; child optimizer bars are suppressed during distributed
execution:

```python
result = simulator.run(
    shots=1_000_000,
    mpi=True,
    workers="auto",
    progress="auto",
    seed=7,
    retain="none",
)
```

Launch the program with `mpiexec -n 4 ...`; `mpi=True` uses the already-launched
communicator and does not create MPI processes itself. Each shot is initialized
from the optimizer's constructor snapshot, so repeated seeded ensembles are
stable and do not mutate the template optimizer.

The practical shot-mode matrix is:

| mode | trajectory status |
| --- | --- |
| `direct` (default) | Quimb direct compression; `mpo` and `quimb` are compatibility aliases |
| bare `<method>` / `quimb-<method>` | explicitly selected Quimb 1D compressor |
| `svd`, `swap` | supported ordinary replay paths; benchmark truncation cost |
| `dmrg`, `dmrg1/2/3` | opt-in variational compressed replay with the selected FIT schedule |
| `mix` | unitary FIT plus an explicit MPO fallback for Kraus gates; no controls/leakage |
| `su` | gate-only simple-update; selected Kraus gates are normalized after replay |
| `exact` | exact unitary, mixture, control, and state-dependent Kraus replay |
| `perm` | fresh identity-order shots only; persistent layouts use the normal MPS modes |

Bare Quimb method names and their `quimb-<method>` qualified forms are passed
to Quimb's native 1D compression dispatcher. The legacy `mpo-<method>` names
remain accepted as aliases. The bare `fit` name is reserved for DMRG; use
`quimb-fit` when selecting Quimb's one-site FIT compressor.
This includes bare `mode="sdc"` and `mode="sdc-oversample"`, which normalize to
the corresponding `quimb-*` modes. They are version-gated through Quimb's
compressor registry and never silently fall back to another method.
Oversampled methods retain Quimb's two-stage structure: an intermediate larger
bond followed by a direct sweep to `chi`. `fit-projector` disables only the
optional simple-update pre-gauge, which is singular on exact product-state
bonds; its projector guess and variational FIT remain native. If `run()`
receives `cutoff_mode="auto"` (now the default), ordinary Pepsy paths use
`rsum2` while Quimb compressors keep their native defaults. In particular,
`dm` keeps `rsum1`. Passing a concrete string explicitly overrides that
method default. `None` remains a compatibility alias for `"auto"`.

For dense MPS, `mode="quimb-src"` applies each gate with Quimb's Successive
Randomized Compression, while `fit_init_strategy="guess-src"` uses SRC to
build the disposable DMRG/FIT initial guess. For native Symmray/fermionic MPS,
the default and an explicit `fit_init_strategy="guess-src"` use Symmray's
sector-preserving randomized SVD (`svd:rand`) instead, so the guess remains
native and never enters dense SRC. The equivalent
`fit_init_strategy="guess_src"` spelling is accepted as a compatibility alias
and is normalized internally to `guess_src`. Set `compression_seed` for reproducible randomized
MPO replay; `fit_init_seed` controls randomized disposable FIT guesses.

`mode="fit"` is a clear alias for the historical `mode="dmrg"`. The
convenience modes share the DMRG backend but have distinct schedules:
`"dmrg1"` uses at most two two-site growth sweeps and then fixed-rank one-site
FIT. If every active bond is already at its attainable ceiling before the fit,
`"dmrg1"` starts directly with one-site FIT. Once every full-chain bond
reaches its physical/`chi` ceiling, the optimizer latches one-site updates for
later windows in the same replay. `"dmrg2"` uses two-site FIT for the required
warm-up (two sweeps by default) and then one-site FIT; `"dmrg3"` follows the
same fixed warm-up schedule with three-site FIT, then one two-site transition
sweep, then one-site FIT. All phases share the `n_iter` budget: with eight
fixed sweeps the schedules are `2,2,1,1,1,1,1,1` and `3,3,2,1,1,1,1,1`.
Tolerance comparisons reset whenever the block size changes. The adjacent
two-site `dmrg2` exact-update exception remains active.
For dense DMRG windows, FIT starts from a disposable compressed guess. The
default is `fit_init_strategy="guess-src"`; the exact FIT target and named
schedules are unchanged. The strategy can be `direct`, `random`,
`random_expand`, or `guess-<method>`, where `<method>` is one of the Quimb
compression methods listed below. The underscore spelling remains accepted for
compatibility. `auto` selects `guess-src` in both the rank-expansion and
reached-chi phases. Native Symmray and fermionic states use the native
sector-preserving randomized guess by default as well. The legacy
`fit_mpo_guess=False` switch still disables the
default named-mode guess. Both the target and the guess remain separate from
the live MPS. The fixed expansion handoff remains two two-site sweeps followed
by one one-site sweep; a window already at its attainable `chi` ceiling uses
one-site FIT directly.
`mode="dmrg"` remains the generic spelling and keeps the adaptive two-site
schedule for local windows. For a long-range window that is wider than the
selected FIT block, it uses the corresponding fixed block handoff so the
terminal canonical center remains authoritative for unitary norm tracking;
the randomized FIT initialization is unchanged. `mode="mix"` is the
transactional unitary variant and defaults to one-site DMRG/FIT after a
direct/MPO warm-up of under-capacity active bonds.
With `fit_block_size=2`, FIT grows only bonds visited by the gate interval, up
to `chi`, through the middle-bond SVD; it does not pad the whole MPS and does
not need an MPO rank warm-up. `fit_block_size=3` uses a three-site effective
wavefunction and two direction-aware native SVD splits, and is useful when a
larger local window is worth the extra decomposition cost. An adjacent
two-site gate span automatically falls back to `fit_block_size=2`. Both block
sizes preserve native dense and Symmray backends. For block sizes 2 and 3, the
optimizer's `fit_init_strategy` chooses whether a disposable FIT guess is
direct, randomly perturbed at fixed rank, randomly expanded on active bonds,
or `guess-<method>` compressed by Quimb. For native Symmray/fermionic states,
`guess-src` is implemented by native randomized SVD while the other
Quimb-specific guess methods retain their native direct fallback. The available methods are
`direct`, `dm`, `zipup`, `zipup-first`, `zipup-oversample`, `src`,
`src-first`, `src-oversample`, `srcmps`,
`srcmps-first`, `srcmps-oversample`, `fit`, `fit-zipup`, and
`fit-projector`, `fit-oversample`, `sdc`, and `sdc-oversample`. The latter two
require a Quimb build containing the corresponding successive deterministic
compressor. They are also valid FIT warm-start policies as
`fit_init_strategy="guess-sdc"` and `fit_init_strategy="guess-sdc-oversample"`.
`auto` selects `guess-src` in both phases;
the current MPS is used directly only when the caller explicitly requests
`direct` (or a native Symmray/fermionic route requires its native warm-start).
Native Symmray and fermionic paths use their graded sector-growth route without
dense random padding. `fit_block_size=1` selects the fixed-rank compatibility
algorithm. In mixed mode, it first applies eligible gates through the
direct/MPO path while active bonds are under capacity, then hands later
eligible gates to one-site DMRG/FIT through a transactional commit. Mixed
two-site and three-site FIT transactions remain available explicitly with
`fit_block_size=2` and `3`, respectively.
Standalone one-site gates use the exact direct/MPO
path; ordinary DMRG target blocks can absorb intervening one-site gates before
the block's shared compression. Generic `mode="dmrg"` remains rank-adaptive
on local windows and uses the fixed canonical handoff for long-range windows,
while named `"dmrg1"` bounds its two-site warm-up at two sweeps and then uses
one-site FIT for the remaining requested sweeps. The named mode does not
extend the two-site phase because of rank stagnation. Once all full-chain
ceilings are reached, it latches one-site updates for later gate windows.
This keeps the bond spaces opened by the SVD warm-up while avoiding repeated
GPU SVD truncations. The dense open-chain ceilings are
`2, 4, 8, ..., chi, ..., 8, 4, 2` (also limited by the current outside-window
bonds); FIT never pads a bond merely to make it equal to `chi`. Set
`fit_adaptive_sweeps` to configure the named `dmrg2`/`dmrg3` warm-up; `dmrg1`
keeps its two-sweep policy. A generic `mode="dmrg"` with `fit_block_size=1`
remains the fixed-rank one-site compatibility path. `fit_layer_size` is the
clear name for
`k_2q_batch`; it counts two-site gates in a contiguous paper-style target
block. For `fit_block_size=2`, an active window spanning at least three sites
uses the generic adaptive schedule for local windows and the fixed canonical
handoff described above for long-range windows; an ordinary two-site gate window
has a complete local variational problem, but by default it honors the
requested FIT sweeps and convergence controls. With
`fit_single_pair_fast_path=True`, `dmrg1`, `dmrg2`, and `dmrg3` immediately
advance to the next gate after one exact update instead of repeating their
warm-up or entering one-site refinement.
The named `dmrg2` schedule is an exception for an adjacent two-site gate: it
uses one exact update by default, regardless of the general fast-path default.
`fit_three_site_sweeps` remains a deprecated alias for
`fit_adaptive_sweeps`.
`fit_max_span="auto"` also limits the spatial width of a batched
target, splitting disjoint gates before they create an unnecessarily wide FIT
window. Set `fit_max_span=None` to restore unrestricted gate-count batching.
If a DMRG/FIT batch
raises, produces non-finite data, or exceeds `chi`, the optimizer restores the
complete pre-batch state (including canonical metadata) and
replays the batch through MPO. Interrupts restore the trial state and are
re-raised.

For ordinary DMRG and mixed DMRG, `n_iter` is a maximum rather than an
unconditional sweep count. `fit_min_iter`, `fit_rtol`, and `fit_patience`
control adaptive stopping from FIT's final retained-center norm change. The
public `MpsOptimizer.run` defaults are `n_iter=8` and `fit_rtol="auto"`.
The automatic tolerance selects `1e-3`, `1e-5`, or `1e-9` for 16-,
32-/complex64-, or higher-precision data. Pass an explicit numeric tolerance
to choose another threshold, or `fit_rtol=None` for fixed iterations.
`fit_patience` counts same-phase sweep-norm samples in the
convergence window, so the default `fit_patience=2` stops after one stable
comparison between two one-site samples; `fit_min_iter` still sets the
minimum completed-sweep count. The old
`mix_fit_min_iter`, `mix_fit_rtol`, and `mix_fit_patience` spellings remain as
deprecated aliases. A legacy value replaces the canonical default for old
call sites; a conflicting non-default canonical value fails instead of
silently choosing a policy. FIT computes only the terminal canonical-center
norm once per sweep. Its native finite checks reduce active tensor blocks and
transfer those flags together with the optional tolerance norm as one compact
vector. Adaptive rank-growing windows require `n_iter >= 2`; a
shorter request raises before fitting, except for the adjacent two-site exact
fast path. An under-capacity, non-adjacent `mode="dmrg1"` window requires
`n_iter >= 3`, reserving its first two sweeps for two-site rank growth and at
least one later sweep for one-site refinement. An already-capped `dmrg1`
window has no growth reservation and uses all requested sweeps as one-site
updates.
At least two adaptive block sweeps are required whenever the active window
needs rank growth, regardless of `fit_rtol`; an adjacent two-site interval is
a structural special case whose only pair is
the complete variational problem. The default
`fit_single_pair_fast_path=False` honors `n_iter` and `fit_rtol`; set it to
`True` to stop after one effective-tensor SVD, even when `fit_rtol=None`.
It does not allocate or scan a second MPS. Ordinary DMRG raises on a detected
non-finite sweep; for compatibility, non-unitary DMRG retains fixed sweeps
when `fit_rtol="auto"`, while an explicit numeric tolerance enables
adaptive stopping there too. With `finite_check=True`, mixed DMRG and
direct/MPO warm-up transactions validate the retained canonical-center norm
and represented exponent before commit. Default replay skips that validation; enable
`quality_check_every=N` when periodic full finite-data and canonical-gauge
checks are needed. Transactional MPO fallbacks are norm-checked only when
`finite_check=True`. Torch and CuPy quality checks process one tensor at a time, combine
scalar results on the device, and transfer one Boolean to the host.

`run(finite_check=False)` disables runtime non-finite detection by default
in every MpsOptimizer mode, including DMRG1/2/3, mixed, MPO/direct/SRC/SDC,
swap/permutation/SVD, SU, and exact replay. This is an optional diagnostic
feature, not a requirement for normal optimization. Leave it off to avoid
extra validation work and possible accelerator synchronization.
FIT array scans, scalar non-finite
convergence checks, mixed commit validation, and unitary norm-consistency
validation are opt-in. `finite_check=True` enables these checks and a final
tensor-data scan in every mode, and emits one performance warning per replay.
Nested FIT calls share that warning instead of warning again for every gate.
Standalone FIT calls still warn when their checks are enabled. Shot workers
inherit the flag unless `run_kwargs["finite_check"]` overrides it.
Convergence and norm accounting still calculate/read the required scalars,
which can synchronize an accelerator. Input validation, zero-divisor guards,
and explicitly requested quality/overlap diagnostics retain their own policies.
Backend linear algebra can still raise its own numerical errors.

Dense DMRG SRC guesses copy tensor metadata across the chain but allocate
independent array data only inside the active endpoint span. When the actual
guess owns its active arrays, rollback retains the untouched original MPS
instead of copying it again. Direct guesses that alias the live state still
receive an isolated rollback copy before FIT. Exterior arrays are read-only
shared inputs; Quimb canonicalization
replaces arrays in the private copy. Active copies retain `left_inds`, backend,
dtype/device, and Torch autograd connections. Native Symmray/fermionic and
unrecognized array backends retain full deep copies. The public standalone
`guess(..., inplace=False)` contract is unchanged.
During a replay, the copy helper classifies each network/array type once and
reuses that decision across backend-preserving updates. The cache is cleared
on replay exit (including failure), invalidated by `set_p`, and never reused
by standalone helper calls. SRC selection also skips rank-ceiling checks
that only affect random initialization policies.

Physical rank ceilings are also cached within a replay; actual changing bond
dimensions remain independent of that cache. State replacement, cap events,
layout changes, mode changes, and explicit canonical resynchronization clear
cached metadata, and a changed `chi` or chain length forces new ceilings.
Mixed replay prepares each FIT window once and reuses its validated final
maximum bond for history and the next transaction. Quality checks invalidate
that maximum because repair may change dimensions. DMRG1 skips rank checks
used solely to validate a sweep budget when `n_iter >= 3` already suffices.

Backend/symmetry classification uses weak references to actual networks,
propagated only through owned backend-preserving copies. It does not retain
discarded MPS states or assume that two different networks have the same
array kind. Standalone helpers inspect their current inputs afresh.

The expensive direct FIT-target overlap contraction is opt-in through
`fit_overlap_diagnostics=True`. Its result is reported in
`opt.get_fit_diagnostics()` as `fit_overlap_fidelity` and
`fit_overlap_infidelity`; the default `False` leaves those fields as `None`
while retaining the ordinary FIT convergence metadata. If enabled, the
contraction is performed after each successful DMRG FIT update, including
DMRG1/2/3 schedules. Mixed-mode transactions retain the existing behavior of
omitting this target-overlap calculation.
If the optional contraction fails or returns NaN/infinity, both overlap values
remain `None` and `fit_overlap_error` explains the failure. This does not reject
the FIT update. The scalar check applies only to the requested overlap result;
it does not enable per-sweep `finite_check` scans.

The DMRG/FIT update follows the variational update described in
the [Ayral *et al.* PRX Quantum paper](https://doi.org/10.1103/PRXQuantum.4.020304):
the effective tensor is built from cached contractions on the left and right,
then the MPS is swept repeatedly. Recommended `fit_block_size=2` forms a
local wavefunction with the two outer virtual legs and both sites' physical
groups, then splits its middle bond with `Tensor.split`. `fit_block_size=3`
forms the analogous three-site tensor and splits it twice, absorbing singular
values toward the sweep direction. Both dispatch to configured dense SVD
drivers and, crucially, Symmray's native block SVD for U1, U1xU1, and fermionic
tensors. `fit_sweep_sequence="RL"` alternates canonical directions; `"R"`
preserves a one-way sweep for dense and native fermionic arrays. Fermionic FIT
keeps a conjugated native working MPS across the sweep sequence, includes the
actual outside graded overlap environments, applies dual-leg phase corrections
before each split, and restores the physical ket afterward. Thus `R`, `L`, and
`RL` are honored exactly without dense conversion or Jordan-Wigner
bosonization.

The named `dmrg1`, `dmrg2`, and `dmrg3` schedules are backend-independent:
native U1, U1xU1, and Z2 fermionic states use the same schedules as ordinary
arrays. `dmrg1` uses its bounded two-sweep warm-up and sticky one-site phase,
while `dmrg2` and `dmrg3` perform their fixed block warm-up before one-site
refinement, with one intervening two-site sweep for `dmrg3`.
Ordinary dense MPS replay keeps the exact gate target `p_g` separate
from FIT's initial state. For a two- or three-site growth window,
`fit_init_strategy` selects the disposable guess: `direct` uses the current
MPS, `random` adds deterministic small noise without changing ranks,
`random_expand` adds noise while opening only under-capacity active bonds, and
`guess-<method>` uses an isolated Quimb replay with the selected compression
method. `fit_init_rand_strength` controls the random scale (default `0.0`),
and `fit_init_seed` controls reproducibility, including randomized Quimb
methods selected through `guess-<method>`. Because the default is
`fit_init_strategy="guess-src"`, no random perturbation or random bond
expansion is used unless a random strategy is selected explicitly.
FIT never copies the target into `fit.p` and never uses a target warm start.
Native nonlocal gates retain their graded auto-swap/sector-growth preparation;
the native `guess-src` path adds only sector-preserving randomized SVD on a
disposable copy and never uses dense random padding.

In this optimizer the fit is intentionally
restricted to the interval `[xmin, xmax]` touched by the current two-site gate
or batch. This is implemented by `FIT.run_gate`, the gate-window version of
`FIT.run_eff`; `run_eff` remains the default one-site full-chain boundary
solver but also has opt-in native block-2/3 updates, while PEPS
`fit_mode="two-site"` uses `run_gate` over the full boundary.
Using `run_eff` for each gate would refit unrelated sites and would no longer
be local DMRG compression. `fit_layer_size=N` explicitly forms the paper's
multi-gate/layer target before each restricted fit. With the default
`fit_target_strategy="auto"`, ordinary NumPy/Torch/CuPy gates remain as exact
spatially split operator layers: FIT contracts them lazily instead of growing,
copying, and repeatedly decomposing an intermediate target MPS. The gate SVD
has only the operator-Schmidt rank and does not apply the output `chi` limit.
`fit_target_strategy="mps"` selects the traditional materialized target;
`"auto"` also chooses that native routed representation for Symmray U1/U1xU1
and fermionic data. `target_cutoff=0.0` keeps either representation exact while
ordinary `cutoff` controls only the two-site output split, so target-
construction loss is not reported as FIT loss.

For a non-adjacent native fermionic gate, MPS DMRG uses Quimb's chi-capped
graded auto-swap algebra on the current native MPS to open charge sectors
before alternating least squares can project them out. This is a native
sector-support preparation, not a copy of the exact target into `fit.p`.
After that preparation, fermionic FIT follows the selected DMRG schedule:
`dmrg2`, for example, can switch from its two block warm-up sweeps to native
one-site refinement. The uncapped target remains separate. At
`target_cutoff=0.0`, routed target splits use the smallest
representable positive absolute cutoff, which removes structural zero singular
directions while retaining every representable nonzero value. This prevents
invalid duplicate dummy modes without introducing target truncation.

All unitary compressed modes (`dmrg*`, `mix`, `direct`, `swap`, `perm`, and
`svd`) default to `stabilize_unitary=False`. The retained approximation scale
therefore remains in the raw working MPS, making norm decay visible by default.
Canonicalization and QR only move that scale to the tracked orthogonality
center; they do not normalize the whole MPS or change the represented state.
Set `stabilize_unitary=True` when numerical scale control is more important
than observing raw norm decay. In that opt-in mode Pepsy restores the raw
working MPS to its pre-compression norm without accumulating the approximation
scale in `p.exponent`. Pass `non_unitary=True` for filters/Kraus/sub-MPO
streams. The old `fit_stabilize_unitary` spelling remains a deprecated alias.

Norm-survival bookkeeping is automatic; there is no
`track_infidelity` constructor or run flag for `MpsOptimizer`. Every retained
unitary compression records an event in `opt.get_norm_events()`. Its
`local_fidelity` is the clipped squared ratio of retained canonical-centre norm
to the expected pre-compression norm, while `fidelity_raw` preserves the
unclipped ratio. `opt.norm_diagnostics()` exposes the latest local value as
`local_fidelity` and the log-accumulated product as `cumulative_fidelity`, with
matching `*_infidelity` fields. These are compression fidelities measured from
norms, not directional target-state fidelities.

Measurement and reset Born weights refer to the state before collapse. Very
small weights are contracted through the branch projector to avoid cancellation
in `1 - <P>`; positive rare forced outcomes remain valid. Measurement norm
accounting includes `p.exponent` on both sides, including the raw norm returned
by DMRG FIT, so represented scale does not appear as compression loss.
Canonical controls reuse tracked center norms; caps keep their raw contraction
scale and establish a fresh baseline for subsequent unitary evolution.

In `opt.norm_diagnostics()`, `norm` and `state_norm` are the actual represented
live-MPS norm. `cumulative_norm` is different: it is the square root of the
accumulated retained-norm survival proxy. Thus the local and cumulative
fidelity fields describe compression survival, while `state_norm` describes
the current tensor-network state scale.

DMRG additionally reports `fit_overlap_fidelity` and
`fit_overlap_infidelity` in `opt.get_fit_diagnostics()` only when
`fit_overlap_diagnostics=True`. Those values contract the final fitted MPS
against the disposable exact FIT target and are genuine target-overlap
diagnostics. They are specific to DMRG and must not be substituted for the
norm ledger used by the other modes. If a backend cannot perform that optional
contraction, the values are `None` and `fit_overlap_error` explains why; the
FIT update itself is not rejected.
Measurement, reset, and state-dependent Kraus events are also recorded,
including `branch_probability`, `physical_boundary`, and `renormalized`. Their
expected norm includes the Born probability, so a normal physical branch has
zero compression infidelity. Renormalization closes the current raw-norm
baseline but does not erase the cumulative compression ledger. The same norm
contract is used by DMRG1/2/3 and the MPO, SVD, swap/perm, and mixed backends.

Unitary compression also validates that the retained canonical-center norm
does not materially exceed its pre-compression norm. The raw overshoot remains
visible in `fidelity_raw`; only small dtype-scaled roundoff is accepted
for low-precision data (for example, `complex64` uses a bounded multiple of
float32 machine epsilon). A larger overshoot still raises because it indicates
broken canonical projection metadata rather than ordinary truncation loss.

`MpsOptimizer.run()` now defaults to `cutoff="auto", cutoff_mode="auto"`.
The cutoff selects `1e-3`
for 16-bit data, `1e-6` for 32-bit/complex64 data, and `1e-12` for 64-bit
data. Explicit numeric cutoffs are unchanged.
Set `quality_check_every=N` to record finite-data and canonical-gauge health in
`opt.get_quality_checks()`. Its default is `False`, so checks are disabled by
default; when enabled,
`quality_check_repair=True` re-canonicalizes if canonical coverage is lost.

Mixed-mode DMRG trials isolate only the active FIT window and the canonicalization
path leading to it. Untouched MPS tensors are shared until a successful trial is
committed, avoiding a full deep copy for every transaction while preserving
rollback safety for the active update. After a non-finite DMRG result,
`mix_sticky_nonfinite=True` (default `False`) keeps the remainder
of the current `run()` call on MPO rather than retrying an unhealthy FIT for
every gate. An ordinary exception still falls back only for its transaction.
The initial MPS must satisfy `p.max_bond() <= chi`. The mixed replay history is
stored in `opt.mix_history` and summarized in `opt.last_mix_summary`; entries
include logical `where`, execution `execution_where`, FIT iterations and
convergence, target bond, fallback sweep, and sticky-disable diagnostics. With
`progbar=True`, the progress bar shows the current backend, cumulative
MPO/DMRG/fallback counts, `~F` (the cumulative compression fidelity), and
`bond=current/chi`. `~F` is the cumulative compression fidelity measured from
retained norms, converted from the log-survival ledger only for display;
accumulation remains logarithmic and numerically stable.
The progress-bar descriptor shows only the active mode: `src`, `zipup`, and
other Quimb compression modes are displayed without the internal `quimb-` or
legacy `mpo-` prefix, while the `quimb` and `mpo` aliases display as `direct`.
Named DMRG schedules display as `dmrg1`, `dmrg2`, or `dmrg3`; generic `dmrg`
and its `fit` alias display as `dmrg`.

Replay timing is opt-in and does not print by itself:

```python
opt.run(timing=True)
print(opt.get_run_timing())
```

The copy-safe record contains replay wall time, event count, final bond,
backend signature, the normalized execution `mode`, the original named DMRG
`mode_alias` when applicable, and the latest `fit_diagnostics` snapshot.
When using `mode="mix"`, it also contains a copy of `last_mix_summary`,
including its elapsed time and backend decision counts.
Mixed runs leave `last_mix_summary["elapsed_seconds"]` as `None` when replay
timing is disabled, so the normal mixed path performs no profiling clock
reads. The measured replay interval begins after argument validation and any
temporary layout setup; it ends before temporary layout restoration and
before `get_run_timing()` makes its defensive result copy.
It also contains inclusive `stages` totals for the active compression-method
replay, such as `direct.replay` or `src.replay`, together with
`canonicalize`, `gate.apply`, `dmrg.target`, `dmrg.fit`,
`normalization`, `control.<event>`, and (when enabled)
`<method>.stabilize`. The internal MPO implementation is therefore not
exposed as the timing label. Stage totals can overlap with the method replay
total; use
them to identify the dominant work, not to add into a second total. DMRG and
mixed-mode timing also
expose
`fit_steps`: one record per completed or failed FIT sweep, including its FIT
call index, global record index, direction, block size, active interval, sweep
time, per-site/block update times, and phase-level sweep overhead. Timing
schema 3 reports `canonicalization_seconds` (including sweep gauge/QR
preparation), `fixed_environment_seconds`, `effective_seconds`,
`svd_seconds`, `writeback_seconds`, `moving_environment_seconds`, and
`sweep_overhead_seconds`. Per-site records additionally expose
`canonicalization_seconds` and `moving_environment_seconds`; the legacy
`environment_seconds` field remains the complete post-writeback environment
phase. `sweep_preparation_canonicalization_seconds` separates the preparation
part of the canonicalization total, while
`moving_canonicalization_seconds` identifies one-site gauge moves inside a
sweep. `MpsOptimizer.get_run_timing()["fit_totals"]` provides the same phase
totals across all FIT calls in the replay, while `fit_steps` retains the
per-sweep and per-site records. FIT phase fields are not one flat additive
list: `canonicalization_seconds` contains both sweep preparation and moving
canonicalization, while legacy `environment_seconds` contains the complete
post-writeback phase. An additive decomposition uses preparation
canonicalization, fixed environments, effective contraction, SVD, writeback,
moving canonicalization, moving environments, and sweep overhead exactly
once. Timing also remains independent of `collect_split_diagnostics`;
profiling an MPS run does not allocate per-SVD truncation dictionaries.

The diagnostic accessors are copy-safe: `get_quality_checks()`,
`get_normalizations()`, `get_norm_events()`, and `get_fit_diagnostics()` return
independent snapshots, so editing a returned list or record cannot corrupt
optimizer-owned state. `get_fit_diagnostics()` returns a copy of the last
DMRG/FIT record or `None` before a FIT update and for modes that do not use
FIT. The record
includes the iteration count, convergence reason, relative change, active block
size, adaptive and one-site sweep counts, and the DMRG1 one-site lock state
when applicable. The record also includes `fit_overlap_diagnostics` so callers
can distinguish a disabled overlap calculation from a backend failure.

Ordinary runs retain no per-gate timer or timing-record overhead. Enabled
profiling moves its internally owned FIT records into the replay result and
copies them only when `get_run_timing()` is called. These are host wall-clock
measurements by default. Use
`run(timing=True, timing_sync_device=True)` for kernel-complete Torch CUDA,
CuPy, or JAX timings; the added barriers intentionally make profiling slower
and are recorded as `timing_sync_device=True` in both replay and FIT records.
The accelerator backend is detected once per timing session, so CPU timing
does not repeatedly scan the MPS. JAX barriers wait on each newly returned
stage result rather than an unrelated previously ready MPS leaf.

`mode="su"` uses simple-update evolution for imaginary-time or other
non-unitary gate streams. It keeps `opt.p` as the simple-update core and
stores the external bond factors in `opt.gauges`. After every run,
`opt.p_ungauged` is refreshed as a physical copy with those gauges inserted.
If the supplied dictionary
does not contain the current bond gauges, the optimizer initializes it with
`opt.p.gauge_all_simple_(gauges=opt.gauges, progbar=False)`, then applies each
gate through `pepsy.gate_simple(..., renorm=True)`. This mode does not
canonicalize the MPS or expose canonical diagnostics. Use
`opt.p_ungauged` for the physical state and `opt.p` for continued SU updates.
If an independent physical copy is needed, use:

```python
physical = opt.p_ungauged.copy()
```

For Symmray block-sparse MPS data, `gate_simple` automatically uses Quimb's
full two-site `split` path so symmetry and fermionic fusion metadata are
preserved. Dense MPS data keeps the faster `reduce-split` path by default.

`mode="swap"` applies non-local two-site gates through a swap-and-split path
and swaps the sites back after each gate. `mode="perm"` uses the same
swap-and-split path but leaves the swaps in place, tracking the current
physical-site-to-logical-site ordering in `opt.qubits`. This is useful for
streams with little expected locality. The returned `opt.p` remains an MPS in
physical order; call `opt.restore_qubit_order()` when a conventional logical
site order is needed.

For repeated evolution, use `opt.apply_layout("quality")` once. This installs
the selected position-to-logical mapping in `opt.logical_order` and keeps the
MPS in that order across subsequent `run()` calls. A bond-one initial MPS is
relabelled without SVD swaps; an initially entangled MPS raises by default, or
can pay one explicit lossy reorder with `allow_lossy_reorder=True` and a caller
provided `cutoff`. The old `run(use_layout_finder=True)` path is retained only
for compatibility and performs the deprecated temporary reorder and swap-back.
Use `opt.logical_site(position)`, `opt.position(site)`,
`opt.remap_sample(config)`, and `opt.to_dense()` for logical readout.

Pauli control events use Quimb's `local_expectation_canonical` when available.
The optimizer passes its `info_c` dictionary, so canonicalization starts from
the tracked `info_c["cur_orthog"]` range, moves only as needed around the
observable support, and records the new range. A concrete tracked range avoids
an orthogonality-center scan. Older Quimb versions without the local evaluator
use a compatibility overlap contraction instead.

Normalization uses the same canonical-center contract as gate application. For a non-unitary run with `normalize_every` enabled, the optimizer reuses an authoritative one-site center inside the active span, normalizes that tensor, and stores the removed scale in `p.exponent`. Only a genuinely broad tracked center is collapsed to one site. Thus `p.norm()` restores the represented norm, while a copy with `exponent=0` exposes the normalized working data. For DMRG, a multi-gate batch is one replay step for this purpose.\n\nUse `get_normalizations()` for scale events, `get_quality_checks()` for optional finite/canonical health records, and `get_fit_diagnostics()` for the latest DMRG/FIT convergence record. `mode="exact"` and `mode="su"` deliberately skip canonical metadata; switching back to an MPS mode rebuilds and canonicalizes the contracted state.\n\n
For a logical gate stream whose site order has not been chosen yet,
`MpsOptimizer.LayoutFinder(gates, L=...)` or
`MpsOptimizer.gate_stream_layout(gates, L=...)` returns a 1D layout plan with
the optimized site order, old-to-new site map, internal mapped locations, and
span statistics. The finder implementation lives in
`pepsy.optimizers.mps.layout.MpsGateStreamLayoutFinder`, while
`MpsOptimizer.LayoutFinder` keeps the attached optimizer-facing API. The finder
builds a weighted interaction graph from gate and
sub-MPO supports, scores layouts with a Tensy-like scalar objective
`weighted_total_span + weighted_cut_congestion_l2 + tail_span_penalty`, and
uses degree/BFS/spectral/recursive candidates, periodic folded-block candidates,
and adjacent-swap refinement. Folded-block candidates are useful for periodic
grid-like streams because they remove the long wrap-around tail from an
ordinary row/column scan. If available, the refinement uses numba;
`order="quality"` also tries optional nevergrad candidates, and optional
KaHyPar recursive bisection when a config is provided with
`kahypar_config_path=...` or `PEPSY_KAHYPAR_CONFIG`. Event weights default to
`weight_mode="auto"`: angle metadata when present, otherwise a cheap
operator-Schmidt proxy for small dense two-site gates, falling back to count
weights. Pass `weight_fn(payload, support, event_type)` for explicit weights.

By default, the quality search keeps the original site order as an explicit
baseline and may use it to initialize the optional Nevergrad search. To test
the graph search independently of that baseline, pass `from_scratch=True`:

```python
scratch_plan = finder.run(
    order="quality",
    from_scratch=True,
    nevergrad_seed=0,
)
```

This still uses the gate-support interaction graph and deterministic
graph-derived candidates; periodic folded-block candidates may also use the
declared site labels. `input_stats` reports the omitted original order for
comparison. On symmetric graphs, different label-equivalent layouts can have
the same score, so a different order is not necessarily a better one.

For a prescribed baseline rather than a searched order, pass an explicit site
permutation as `order`. The returned plan is marked `selected_order="fixed"`
and keeps the original gate stream unchanged:

```python
zigzag = py.square_lattice_zigzag(6, 6)
fixed_plan = finder.run(order=zigzag)
```

For regular two-dimensional lattices, the MPS finder also accepts the shared
`OneDMap` geometric presets. Set `lattice_shape=(Lx, Ly)` on the finder (or
pass it through `MpsOptimizer.gate_stream_layout`) and choose
`"row-major"`, `"col-major"`, `"snake"`, `"snake-row-major"`,
`"folded-snake"`, `"folded-snake-row-major"`, `"hilbert"`, or
`"hilbert-row-major"`. The default logical label is `x * Ly + y`; pass
`lattice_site=lambda x, y: ...` for another site-label convention.
`"folded-snake"` is the periodic-boundary baseline, while the Hilbert modes
use the classical traversal on power-of-two squares and a complete
generalized rectangular Hilbert traversal elsewhere. Tree geometric presets
use the same `OneDMap` order, so an MPS/tree handoff does not silently change
the lattice traversal.

```python
finder = py.MpsOptimizer.LayoutFinder(
    gates,
    L=36,
    lattice_shape=(6, 6),
)
hilbert_plan = finder.run(order="hilbert")
```

`square_lattice_zigzag` scans x across each row and reverses direction on
successive rows. It is a deterministic comparison layout; it performs no
refinement or tensor work.

For compression-oriented selection, pass `objective="compression"`. This
uses operator-Schmidt load over every MPS cut crossed by each support, with
support span retained as a replay-cost tie-breaker. Exact small dense ranks
are used when available; opaque, native, and wide operators use a conservative
operator-space rank bound and are marked in `rank_bound_reasons` rather than
silently being treated as rank two. The default `objective="locality"` keeps
the faster span/congestion heuristic for backwards compatibility.

The layout score depends on gate supports and optional gate/event weights, not
on the initial MPS tensor values. The plan does not rewrite the gate stream. To
use a layout during replay, call `opt.run(use_layout_finder=True)` or pass a
layout order such as `opt.run(use_layout_finder="quality")`; the optimizer
temporarily permutes the working MPS and restores the returned MPS to the
original site order. Layout-aware replay prints a concise report by default;
pass `layout_report=False` to silence it.

When the current state matters, use the explicit pilot selector:

```python
plan = opt.select_layout_for_compression(
    pilot_candidates=4,
    pilot_steps=64,
)
opt.apply_layout(plan, layout_report=False)
```

The selector replays the best static candidates on independent copies using
the real MPS mode, `chi`, cutoff, backend, and dtype. It chooses by final bond
dimension and elapsed time, and returns per-candidate records under
`plan["pilot"]`. The original state, queue, and
layout are unchanged. Perform this before installing a persistent layout;
reordering an already-entangled MPS remains explicitly guarded because the
reorder itself can be lossy or expensive.

The layout can be inspected graphically without changing the optimizer. The
finder returns a Matplotlib `(fig, ax)` pair. The original lattice and gate
connectivity remain a light grey background, while the colored arrow chain
shows the selected MPS permutation directly. The default plot is axis-free and
does not number the background lattice; use `show_site_labels=True` and
`show_axes=True` when those annotations are useful:

```python
finder = opt.layout_finder()
plan = finder.run(order="quality")
fig, ax = finder.plot(
    plan,
    site_coords={q: (q % 4, q // 4) for q in range(opt.p.nsites)},
)
```

`opt.plot_layout(plan, site_coords=...)` is the equivalent convenience wrapper.
Coordinates are optional; tuple-valued site labels are interpreted as `(x, y)`
automatically, and ordinary labels fall back to a 1D line. Install the
optional `viz` profile to enable plotting. A stream-order colorbar is not shown
by default; pass `colorbar=True` only when the MPS-position scale is useful.
The default plot contains visible `0` through `last` order labels but no title,
chain sentence, or other text. The styling follows Quimb's axis-free schematic
drawings while retaining Pepsy's ordinary `(fig, ax)` return value.
Pass `show_order_labels=False` to hide the position labels, or use
`show_chain_label=True` and `show_title=True` for additional annotations.


> API details are maintained as handwritten Markdown in this page.
