# `pepsy.optimizers.noise`

Pepsy's native noise design is **stream-local**: put stochastic entries directly
where the hardware schedule says the channel acts, then choose trajectory
sampling settings (`shots`, `seed`, independent/coalesced replay, and
`run_kwargs`) at the runner.

```python
stream = [
    ("h", 0),
    ("x_error", 1e-4, 0),
    ("cnot", 0, 1),
    ("depolarize2", 1e-3, 0, 1),
    ("t", 0),
    ("pauli_channel1", {"z": 2e-4}, 0),
]

result = pepsy.run_coalesced_trajectory_shots(
    lambda: pepsy.MpsStabOptimizer(2, chi=64),
    stream,
    shots=10_000,
    seed=7,
)
```

Equivalently, select the sampling strategy on the trajectory runner:

```python
result = pepsy.run_trajectory_shots(
    lambda: pepsy.MpsStabOptimizer(2, chi=64),
    stream,
    shots=10_000,
    seed=7,
    strategy="coalesced",   # or "independent" / "auto"
    max_branches=256,
)
```

Supported first-cut stochastic entries are:

- `("x_error", p, q)`, `("y_error", p, q)`, `("z_error", p, q)`
- `("depolarize1", p, q)`, `("depolarize2", p, q0, q1)`
- `("pauli_channel1", probs, q)`, where `probs` is `(p_x, p_y, p_z)` or a mapping
- `("pauli_channel2", probs, q0, q1)`, using Stim's 15 non-identity two-qubit Pauli labels
- `("amplitude_damping", gamma, q)`, sampled with the state-dependent trajectory runner

Stateful leakage entries are also Pepsy-native trajectory events:

- `("leakage", p, q)` or `("leak", p, q)` marks `q` leaked with probability `p`
- `("leakage_return", p, q)`, `("seepage", p, q)`, or `("unleak", p, q)`
  returns an already leaked qubit to a random computational-basis branch with
  probability `p`
- `("measure_leaked", q)` records a ternary PECOS/Selene-style result:
  `0` or `1` for a normal computational-basis measurement, and `2` when the
  trajectory state knows the qubit is leaked
- `("leak2depolar", enabled)` makes later `("leakage", p, q)` events use a
  full one-qubit depolarizing replacement instead of marking leakage
- `("leakage_depolarize", p, q)` applies that depolarizing replacement for a
  single event regardless of the current `leak2depolar` mode

Leakage state is carried per shot or coalesced leaf outside the qubit MPS. While
a qubit is leaked, ordinary gates touching it are suppressed. `reset` and
`measure_reset` clear the leakage flag; `measure_reset` first records the
leaked-qubit measurement as bit `1`. The sampled diagnostics live in
`TrajectoryShotResult.leakage_records` or
`CoalescedTrajectoryLeaf.leakage_records` as `LeakageRecord` objects. Leakage
streams support independent and coalesced replay; coalescing branches only when
the classical leakage outcome changes the represented state.

An MPS `cap` is a structural boundary in both strategies. It always removes
the selected site, including a leaked site's placeholder, then removes that
site's leakage flag and shifts higher logical labels down by one. Subsequent
gate suppression, reset, and measurement use those updated labels. Conditional
caps apply this update only to the selected branches; persistent layouts still
reject caps, while `perm` maintains its shortened logical mapping.

`PauliErrorModel` remains a convenience macro for clean deterministic streams.
It samples independent **physical Pauli trajectories**, not a density matrix.
Each non-identity X/Y/Z fault is inserted into a concrete gate stream after every
target of an ordinary gate. The resulting stream can be replayed by either
`MpsOptimizer` or `MpsStabOptimizer`; for STN, every sampled fault is a Clifford
that is absorbed by the Stim tableau. Do not mix this macro with stream-local
stochastic entries; use `run_trajectory_shots(...)` or
`run_coalesced_trajectory_shots(...)` when the stream already contains noise.

```python
import pepsy

noise = pepsy.PauliErrorModel.depolarizing(1e-3)
result = pepsy.run_noisy_shots(
    lambda: pepsy.MpsStabOptimizer(6, chi=32),
    gates,
    noise,
    shots=1_000,
    seed=7,
)

# Each trajectory can be measured/read independently.
samples = [sim.sample_bits(100, seed=shot) for shot, sim in enumerate(result.optimizers)]
```

For ordinary MPS replay, use the same function with a factory that constructs a
fresh state for every shot:

```python
result = pepsy.run_noisy_shots(
    lambda: pepsy.MpsOptimizer(initial_mps, chi=64, mode="mpo"),
    gates,
    pepsy.PauliErrorModel.bit_flip(0.01),
    shots=100,
    seed=7,
)
```

`result.gate_streams` holds the sampled, replayable streams and `result.faults`
holds concise `(gate_index, site, pauli)` records. Use
`sample_noisy_gate_stream(...)` or `sample_noisy_gate_streams(...)` when only
stream construction is needed.

## Shot-aware `MpsOptimizer` API

`MpsOptimizer` now owns the ordinary and noisy replay APIs. Pass the initial
MPS and gate stream once; `run()` keeps the existing single-state behavior for
ordinary streams, while a stream-local noisy event or `shots > 1` dispatches to
the trajectory runner:

```python
simulator = pepsy.MpsOptimizer(
    initial_mps,
    gate_stream,
    chi=64,
    mode="mpo",
)
result = simulator.run(
    shots=10_000,
    strategy="auto",
    seed=7,
)
```

With `error_model=None`, stream-local stochastic entries use trajectory replay.
For the legacy clean-stream Pauli model, pass
`error_model=pepsy.PauliErrorModel.depolarizing(1e-3)` to `run`. The result is a
`NoisyResult` with `.optimizers`, `.counts`, `.gate_streams`, `.weights`,
`.shots`, and `.branches`; `.coalesced` identifies count-coalesced storage and
`.raw` retains the original runner result. `strategy="auto"` shares exact
prefixes under the `max_branches` safety cap. It performs a conservative
branch-cap preflight when possible, avoiding a partial coalesced replay that is
guaranteed to restart; the exact branch cap remains a hard safety limit.

The `retain` option is available on `MpsOptimizer`, `TreeNoisy`,
and the low-level shot runners. `retain="all"` keeps states and replay
metadata, `retain="final"` keeps only final states and weights, and
`retain="none"` keeps no final optimizer states. Use the last form for runs
whose outputs are consumed during execution rather than inspected afterward.

For repeated custom-runner use, `compile_trajectory_stream(gates)` returns a
backend-neutral `TrajectoryStreamPlan`. It parses stochastic entries once and
records ordinary-segment boundaries; live optimizers still perform their own
device/backend conversion.

The low-level factory-based functions remain available for custom optimizer
classes or non-MPS backends.

`TreeNoisy` exposes the same API for `TreeOptimizer` and accepts either an
entangled `TreeTensorNetwork` or a product MPS as its initial state. Its
`tree_settings` mapping contains `TreeOptimizer` options such as `chi`,
`layout`, `tree`, and `max_arity`; the gate stream is used when constructing an
automatic tree layout:

```python
simulator = pepsy.TreeNoisy(
    initial_state,
    gate_stream,
    tree_settings={"chi": 64, "layout_objective": "congestion"},
)
result = simulator.run(shots=10_000, strategy="auto", seed=7)
```

Tree and MPS optimizers share the logical feed-forward form
`("if", record, bit, action)`. Tree replay resolves the measurement record
before applying the action, including in count-coalesced branches. `NoisyResult`
is the generic result-facade name for all noisy replay backends.

## Exact coalesced ensembles for rare noise

When the total fault rate is small, avoid replaying the same no-error prefix
once per shot. `run_coalesced_noisy_shots(...)` holds one optimizer state per
distinct sampled branch and its number of represented shots. It runs an ideal
prefix once, samples exact multinomial branch counts at each Pauli channel,
and copies an MPS only when two nonempty branches genuinely diverge:

```python
result = pepsy.run_coalesced_noisy_shots(
    lambda: pepsy.MpsStabOptimizer(6, chi=32),
    gates,
    pepsy.PauliErrorModel.depolarizing(1e-3),
    shots=100_000,
    seed=7,
)

assert sum(result.counts) == 100_000
for leaf in result.leaves:
    print(leaf.count, leaf.faults)
```

The represented samples are still independent draws; only their identical
state evolution is shared. `run_coalesced_trajectory_shots(...)` provides the
same exact tree for `TrajectoryEvent` mixtures, state-dependent Kraus channels,
leakage, and mid-circuit controls. It branches `measure`, `reset`, and
`measure_reset` with exact binomial counts when a hidden measurement outcome
can change the pure state. Product-state resets use a one-leaf fast path; leaf
`measurements` records selected projective outcomes.

Every trajectory result exposes a lightweight `diagnostics` summary:

```python
print(result.diagnostics.max_kraus_probability_residual)
print(result.diagnostics.used_kraus_copy_fallback)
```

`max_kraus_probability_residual` is the largest deviation of the raw Kraus
branch probabilities from one before the sampler normalizes them. A small
residual is expected from finite-bond truncation; a large residual indicates
that the channel, local contraction, or compression settings should be
checked.

This is normally more useful than `torch.vmap` for rare faults: after a fault
or collapse, states have different tensor data and often different bond
profiles, while a coalesced no-error group stays one ordinary MPS/STN replay.
For terminal readout, call `result.sample_bits(...)` (or
`sample_coalesced_bits(result, ...)`). It invokes one batched `MpsSampler`
call per ordinary-MPS leaf and the STN tree sampler per STN leaf, returning
only terminal rows plus the source `leaf_indices`—never one optimizer per row:

```python
samples = result.sample_bits(seed=8)
assert samples.shots == result.shots
# samples.configs: (shots, n) computational-basis rows
# samples.leaf_indices: source coalesced leaf for each row
```

Conditional caps can leave different register lengths across leaves. In that
case `samples.configs` has width equal to the longest surviving register and
pads shorter rows on the right with `-1`. `samples.lengths[row]` gives the valid
prefix length, so use `samples.configs[row, :samples.lengths[row]]` for measured
bits. Columns use each leaf's current logical numbering after its caps, not
the original register labels. Lengths, leaf indices, and probabilities remain
aligned when rows are shuffled. Uniform-length batches keep their previous
configuration values and shape, and also expose `lengths`.

### Conservative automatic strategy

`run_noisy_shots(...)` keeps its backward-compatible independent replay by
default. For ordinary Pauli gate streams, `strategy="auto"` selects exact
count coalescing only when the expected per-shot number of non-identity faults
is small:

```python
result = pepsy.run_noisy_shots(
    factory,
    gates,
    pepsy.PauliErrorModel.depolarizing(1e-3),
    shots=512,
    strategy="auto",
    max_branches=128,
)
```

The automatic threshold is `lambda = (# noisy gate targets) *
(p_x + p_y + p_z) <= 0.1`. This is deliberately conservative: coalescing is
strongest when most shots take the no-fault path. An unforced `measure`,
`reset`, or `measure_reset` makes the policy choose independent trajectories,
because its physical collapse branches can dominate even when noise is rare.

The live-leaf cap is exact safety control, not truncation. If a selected
coalesced run would retain more than `max_branches`, automatic mode discards
the partial tree and restarts the whole ensemble independently. Pass
`strategy="coalesced"` to request coalescing explicitly; its same cap raises
instead of silently changing strategy. `auto_max_expected_faults` can tune the
default `0.1` threshold when profiling a different workload.

## Rare-event importance sampling

For a logical event much rarer than the physical noise rate, bias the proposal
distribution toward the relevant branches and retain an unbiased likelihood
ratio. The physical probabilities remain the `TrajectoryChannel` or
`PauliErrorModel` probabilities; only the sampling proposal changes:

```python
proposal = pepsy.ImportanceSamplingPolicy({
    12: {"I": 0.5, "X": 0.5},  # event 12: proposal, not physical probability
})
result = pepsy.run_trajectory_shots(
    factory,
    noisy_stream,
    shots=100_000,
    seed=7,
    importance_sampling=proposal,
    max_branches=256,
    max_branch_factor=4,
)
logical_error = [is_logical_error(sim) for sim in result.optimizers]
estimate = result.estimate(logical_error)
print(estimate, result.effective_sample_size)
```

The policy mapping can be label-based for every event, event-index keyed, or a
callable `(event_index, labels, target_probabilities, optimizer)`. Every target
branch must have nonzero proposal probability. `TrajectoryRecord` exposes both
`probability` (physical) and `proposal_probability`, plus `likelihood_ratio`.
Coalesced leaves carry the product ratio in `leaf.weight`, and
`CoalescedTrajectoryResult.estimate(...)` includes leaf multiplicities. For the
Pauli convenience API, pass a proposal `PauliErrorModel` as
`importance_sampling` to `run_noisy_shots(...)` or
`run_coalesced_noisy_shots(...)`.

`max_branches` bounds live coalesced states and `max_branch_factor` bounds the
number of nonempty children created by any one stochastic event. These are hard
safety budgets: a bounded coalesced run raises (or `strategy="auto"` restarts
independently) rather than pruning probability mass.

## Deterministic parallel trajectories

Use `parallel_workers` directly on `run_trajectory_shots(...)` or
`run_noisy_shots(...)`, or call the explicit
`run_parallel_trajectory_shots(...)` / `run_parallel_noisy_shots(...)` helpers.
Independent shots receive their channel and optimizer child seeds before worker
dispatch, so changing the worker count preserves shot order and outcomes.
Coalesced execution keeps one deterministic branch-splitting stream and runs
independent live leaves concurrently:

```python
result = pepsy.run_trajectory_shots(
    factory,
    noisy_stream,
    shots=100_000,
    seed=7,
    strategy="coalesced",
    parallel_workers=8,
    parallel_backend="thread",
)
```

`parallel_backend="gpu"` also uses threads, intentionally keeping
Torch/CuPy/JAX objects in one process. It is a scheduling hint, not a device
selector: place the initial MPS on the desired device once (for example with
`mps.apply_to_arrays(py.build_backend(device="cuda"))`). `MpsOptimizer` then
lazily coerces ordinary dense gates to the live MPS backend, and converts
sub-MPO tensor payloads through `apply_to_arrays`; pre-converting gates is still
preferable when avoiding per-shot conversion overhead matters. This is
concurrent trajectory execution, not an unsafe shared mutable optimizer or an
automatic choice of Torch versus CuPy, CUDA device, or dtype.
The high-level `run_noisy_shots` and `run_trajectory_shots` helpers resolve
`strategy="auto"` consistently before local parallel dispatch. The lower-level
`run_parallel_noisy_shots` and `run_parallel_trajectory_shots` helpers still
expect an explicit `"independent"` or `"coalesced"` strategy.

## MPI shot ensembles

Use `MPIShotRunner` when the shot ensemble should be distributed across MPI
processes. It is an orchestration layer rather than another optimizer, so the
same factory works for `MpsOptimizer`, `MpsStabOptimizer`, `TreeOptimizer`,
and `TreeStabOptimizer`:

```python
import pepsy

runner = pepsy.MPIShotRunner(
    lambda: pepsy.MpsStabOptimizer(32, chi=64),
    noisy_stream,
)
result = runner.run(
    shots=1_000_000,
    seed=7,
    retain="final",
)
```

For a single ensemble, `run_mpi_shots` is the concise equivalent:

```python
result = pepsy.run_mpi_shots(
    lambda: pepsy.MpsStabOptimizer(32, chi=64),
    noisy_stream,
    shots=1_000_000,
    seed=7,
    retain="final",
)
```

Launch the program with `mpiexec` or `mpirun` after installing the optional
MPI profile (`pip install -e ".[mpi]"`). Every rank must construct and call
the runner collectively. Each rank owns complete local optimizer states; MPI
distributes global shot IDs, not pieces of an MPS or tree tensor network.
All ranks perform a synchronized preflight for runner arguments before
entering the shot collectives. Invalid input therefore raises an
`MPIShotError` on every rank; callers must still provide the same valid run
configuration on every rank.

MPI supports independent and rank-local coalesced execution. With
`strategy="independent"`, the global shot ID is part of the trajectory seed,
so changing the number of ranks does not change a shot's stochastic stream.
With `strategy="coalesced"`, each rank coalesces only its local batch; this is
useful for rare faults but is not rank-count invariant. Use `retain="none"`
when no post-run state is needed, or `retain="final"`/`"all"` before reducing
an observable:

Independent MPI execution supports all four optimizer families. Coalesced
execution additionally requires the backend's trajectory-copy contract; the
current coalesced backends are `MpsOptimizer`, `MpsStabOptimizer`, and
`TreeOptimizer`. Use independent MPI execution for `TreeStabOptimizer`.

The same orchestration is available directly from `MpsOptimizer.run`,
`MpsStabOptimizer.run`, `TreeOptimizer.run`, and `TreeStabOptimizer.run` by
passing `shots=...` and `mpi=...`. Direct calls create fresh per-shot copies
from the current optimizer state and leave the caller's state and queued
stream unchanged. Use `MPIShotRunner` when the factory/stream needs to be
shared across optimizer types or when constructing a reusable runner.

```python
def observable(optimizer):
    # Define this for the optimizer backend you are using.
    return measure_observable(optimizer)

estimate = result.reduce_mean(observable)
```

`reduce_mean` requires retained final states (`"final"` or `"all"`). With
`retain="all"`, `result.gather_records(root=0)` gathers trajectory records in
global shot order for independent runs; optimizer states are never gathered
automatically. For million-shot runs, evaluate an observable in bounded
chunks instead:

```python
streamed = runner.run(
    shots=1_000_000,
    seed=7,
    observable=measure_observable,
    chunk_size=2_048,
)
estimate = streamed.reduce_mean()
```

The callback is evaluated on each temporary optimizer and those states are
released after each chunk. `retain="none"` is required in this mode.
The callback may return a scalar or a numeric array, provided its shape is
consistent across ranks. `result.reduce_mean(...)` uses the same shot-count
denominator as the underlying result estimator while combining rank-local
multiplicities and importance weights. `result.reduce_sum(value)` combines
already-computed local scalars or arrays. The runner materializes the gate
stream once, so it can be reused for multiple collective runs. MPI is the
outer process-level parallelism; `local_workers` can optionally enable
the existing thread/GPU runner inside each rank. The direct runner defaults to
one local worker to avoid oversubscription; pass `local_workers="auto"` to
divide the host CPU allowance among ranks. `progress=True` reports one
rank-zero aggregate bar for independent ordinary, streaming, and checkpointed
runs; coalesced runs intentionally suppress shot-level progress because their
work is branch-based rather than one optimizer per shot.

For rank-scaling measurements, use the repository benchmark script and vary
only the MPI process count between runs:

```bash
mpiexec --oversubscribe -n 4 python benchmarks/mpi_shots.py \
  --shots 10000 --qubits 16 --depth 8
```

The script defaults to `--workers auto` and reports the slowest-rank wall time
and global shots per second. Pass `--workers 1` for a process-only baseline.
Compare independent and local coalesced execution separately; coalescing is
not rank-count invariant.

For a multi-node Slurm allocation, the repository includes a launcher smoke
template:

```bash
sbatch benchmarks/mpi_slurm.sh
```

It uses `srun` and the cluster's configured PMI/PMIx transport. Set
`PEPSY_MPI_SHOTS`, `PEPSY_MPI_QUBITS`, `PEPSY_MPI_DEPTH`,
`PEPSY_MPI_WORKERS`, or `PEPSY_MPI_STRATEGY` in the batch environment to adjust
the workload; the
script assumes Pepsy and its MPI-enabled Python environment are already
available on every node.

### Resuming a streaming run

For long bounded-memory observable runs, pass a checkpoint prefix. Each rank
atomically writes its own progress file after every completed chunk:

```python
checkpoint = "/scratch/pepsy/shots"
result = runner.run(
    shots=1_000_000,
    seed=7,
    observable=measure_observable,
    chunk_size=2_048,
    checkpoint_path=checkpoint,
)
```

If a rank fails, rerun the same collective call with `resume=True` and the
same checkpoint prefix, seed, shot count, strategy, chunk size, retention mode,
and MPI process count:

```python
result = runner.run(
    shots=1_000_000,
    seed=7,
    observable=measure_observable,
    chunk_size=2_048,
    checkpoint_path=checkpoint,
    resume=True,
)
estimate = result.reduce_mean()
```

`checkpoint_keep` controls how many historical per-rank snapshots are retained
in addition to the atomically updated latest file; the default is `2`. If the
latest file is unreadable, resume searches the retained snapshots from newest
to oldest. An existing checkpoint is never overwritten by a fresh run; use
`resume=True` or choose a new prefix.

Checkpointing also supports independent optimizer-state runs when the result
must retain states:

```python
retained = runner.run(
    shots=100_000,
    seed=7,
    retain="final",
    chunk_size=2_048,
    checkpoint_path="/scratch/pepsy/retained",
    checkpoint_keep=3,
)
```

This mode serializes each completed raw shot-result chunk plus a small index in
trusted per-rank checkpoint files, then merges the chunks when resuming. It requires
`strategy="independent"`, `retain="final"` or
`"all"`, and pickle-compatible optimizer states. Coalesced optimizer-state
checkpoints are intentionally rejected until branch identity and count merges
have a durable protocol. Checkpoint files must live on a reliable shared
filesystem, or on rank-local storage with the same path visible to each rank.
When a custom optimizer factory or observable callback changes independently of
the gate stream, pass the same stable `checkpoint_id` on every run to bind the
checkpoint to the application-level semantics.
A resumed result exposes `resumed=True`, keeps the prefix in
`result.checkpoint_path`, and publishes one `MPIRankDiagnostics` record per
rank through `result.rank_diagnostics` with shot ownership and elapsed time.
Set `checkpoint_sync=False` only when an external filesystem policy provides
the required durability; set `collect_diagnostics=False` to skip the final
diagnostics gather on very large communicators.
After a successful run, call `result.cleanup_checkpoints()` collectively on
all ranks when the files are no longer needed.

## User-defined quantum trajectories

`TrajectoryEvent` is the general independent noise-simulation interface. Put
one directly inside an ordinary gate stream and run independently sampled shots
with `MpsOptimizer`, `TreeOptimizer`, or `MpsStabOptimizer`. It does not require
Stim or a density matrix.

Use a `mixture` for a user-defined random-unitary channel. Its outcomes have
explicit probabilities, so `sample_trajectory_stream(...)` can make a concrete
noisy stream without an optimizer:

```python
import numpy as np
import pepsy

x = np.array([[0, 1], [1, 0]], dtype=complex)
bit_flip = pepsy.TrajectoryChannel.mixture([
    ("I", 0.99, np.eye(2)),
    ("X", 0.01, x),
])
stream = [
    (pepsy.h(), 0),
    pepsy.TrajectoryEvent(bit_flip, 0),
]
sample = pepsy.sample_trajectory_stream(stream, seed=7)
```

Use `kraus` when the branch probability must be computed from the evolving
state. Each selected branch is normalized before the later stream entries run;
this supports non-Pauli channels such as amplitude damping:

```python
stream = [
    (pepsy.x(), 0),
    pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.02), 0),
    (pepsy.h(), 0),
]
result = pepsy.run_trajectory_shots(
    lambda: pepsy.MpsStabOptimizer(1, chi=32),
    stream,
    shots=10_000,
    seed=7,
)

# One named result per noise event in each shot.
print(result.records[0])
```

`TrajectoryChannel.kraus([("no_jump", K0), ("jump", K1)])` accepts any
complete local qubit channel (`sum(K.conj().T @ K) == I`) on the corresponding
one- or multi-qubit `TrajectoryEvent` support. For ordinary MPS or TTN replay,
replace the factory above with a fresh `MpsOptimizer(initial_mps, ...)` or
`TreeOptimizer(...)` and pass its usual options through `run_kwargs`.

For ordinary `MpsOptimizer`, Kraus normalization is tracked automatically in
the optimizer's norm-survival ledger. The selected branch event retains its
Born `branch_probability` and is marked as a `physical_boundary`; the expected
norm includes that probability, so physical renormalization is not reported as
compression infidelity. Inspect `optimizer.norm_diagnostics()` and
`optimizer.get_norm_events()` after independent or coalesced replay. Fidelity
tracking is automatic for `MpsStabOptimizer`; no tracking flag is needed.

For `MpsStabOptimizer`, a selected Kraus outcome is a
normalized trajectory boundary, just like a measurement/reset: its Born weight
is retained in the trajectory record but is not treated as compression loss.
`sim.norm_diagnostics()["norm"]` is the square root of the product of all
completed/current segment survivals, so it remains meaningful after state
renormalization. The older `total_norm_proxy` key remains as a compatibility
alias.

## Reading a Stim circuit

`compile_stim_circuit(...)` accepts `stim.Circuit` or Stim source text. It
compiles one- and two-qubit Clifford gates, Pauli measurements/resets, and
**every native Stim stochastic error instruction**, then reuses that plan for
all shots:

```python
circuit = """
H 0
CX 0 1
PAULI_CHANNEL_2(0,0,0, 0,0.01,0,0, 0,0,0,0, 0,0,0,0) 0 1
HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0.02) 0
"""

result = pepsy.run_stim_shots(
    lambda: pepsy.MpsStabOptimizer(2), circuit, shots=10_000, seed=7,
)
print(result.faults[0])
print(result.heralds[0])
```

`run_coalesced_stim_shots(...)` has the same output shape as the coalesced
ordinary trajectory runner and supports the complete compiled native Stim
noise set, including two-qubit, heralded, and `E`/`ELSE_CORRELATED_ERROR`
chains. It shares all ideal segments and records per-leaf Pauli faults and
herald bits. `TrajectoryShotResult.measurements` and
`StimShotResult.measurements` expose structured Pauli outcomes with event
metadata. Detector and logical-observable annotations are compiled into the
plan and resolved as `result.syndromes` and `result.observables`; coalesced
Stim results expose the same records once per leaf, alongside each leaf's
count.

Measurement-record feed-forward is also supported: `CX/CY/CZ rec[k] q` is
lowered to `("if", k, bit, action)`, and the ordinary MPS/STN stream form is
available directly. In a hand-written stream, `("if", record, bit, action)`
uses computational bits (`+1 -> 0`, `-1 -> 1`), with negative records counting
back from the latest measurement. Independent and coalesced trajectory replay
resolve the predicate separately for every shot/leaf before applying `action`.
Selected actions inherit the configured replay/FIT settings. Trajectory
runners retain the concrete executed action rather than both an action and its
conditional wrapper; nested conditional controls use normal measurement,
reset, and cap handling, including coalesced branching and leakage updates.
The `PauliErrorModel` convenience macro treats the conditional wrapper as a
control event and automatically samples ordinary named or matrix gate actions
only on the branch where the predicate is true. Conditional measurements,
resets, nested controls, and sub-MPO actions are left unchanged; use explicit
stream-local noise entries when those operations need a particular noisy
channel. `k=-1` means the latest measurement; general record-to-record
arithmetic is intentionally not lowered.

Supported Stim error channels are `X_ERROR`, `Y_ERROR`, `Z_ERROR`,
`DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`,
`CORRELATED_ERROR`/`E`, `ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`,
`HERALDED_PAULI_CHANNEL_1`, `I_ERROR`, and `II_ERROR`. Stim itself only
represents Pauli noise, so amplitude damping is not a missing Stim channel.

## Coherent crosstalk and truncation studies

`CoherentCrosstalkModel` inserts coherent nearest-neighbour `ZZ` rotations
after selected two-qubit gates. The emitted `rzz` angle follows Pepsy's
`exp(-i theta P / 2)` convention; `sign_mode="random_sign"` provides a
reproducible random-sign comparison when a seed is supplied:

```python
model = pepsy.CoherentCrosstalkModel(
    theta=0.01,
    adjacency={0: (1,), 1: (0, 2), 2: (1,)},
    sign_mode="random_sign",
)
noisy_stream = model.transform(gates, seed=7)
```

For coherent-noise and QEC studies, both STN frontends provide
`MpsStabOptimizer.truncation_convergence(...)` and
`TreeStabOptimizer.truncation_convergence(...)`. They replay the same stream
at several `chi` values and report peak bond, norm diagnostics, and an
optional observable. `chi=None` is the lossless reference up to the configured
cutoff.

For an end-to-end repeated-check validation, use the public
`compile_stim_circuit`, `run_stim_shots`, and
`run_stabilizer_tree_stream` APIs directly. Keep performance experiments in
the external benchmark workspace so the package remains focused on reusable
simulation APIs.


> API details are maintained as handwritten Markdown in this page.
