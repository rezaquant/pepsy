# `pepsy.optimizers.mpo.optimizer`

`MpoOptimizer` accepts ordinary Quimb MPOs and Symmray block-sparse MPOs. For
a native graded fermion workflow, use the canonical
`Fermion.build_mpo(...)` entry point and replay the matching native gate
stream:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
edges = [(0, 1), (1, 2)]
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)
mpo = fermion.build_mpo(hamiltonian=hamiltonian, L=3)

opt = pepsy.MpoOptimizer(
    mpo,
    hamiltonian.trotter_gates(0.01),
    chi=16,
    mode="svd",
)
mpo = opt.run(progbar=False)
```

For dense MPOs, `mode="mpo"` accepts one- or multi-site dense gates, including
non-contiguous supports such as `(0, 1, 3)`. The gate matrix has dimension
`2**len(where)` by `2**len(where)` for qubit sites. Bare/default two-site gates
use Quimb's native `gate_sandwich_with_auto_swap(..., dagger=True)` route,
including automatic swaps for non-local supports. Explicit ket/bra pairs and
multi-site gates retain the dedicated layer-compression path. `mode="svd"`
and `mode="dmrg"` retain their one-/two-site replay contracts. Native Symmray
`mode="mpo"` uses the symmetry-aware SVD route and therefore follows that
one-/two-site restriction.

The dense MPO compression backend exposes the same Quimb compressor family as
`MpsOptimizer`. `mode="src"` is equivalent to `mode="quimb-src"`, and the
legacy-qualified `mode="mpo-src"` spelling is equivalent as well. The same
aliases are available for `direct`, `dm`, `zipup`, `zipup-first`,
`zipup-oversample`, `src`, `src-first`, `src-oversample`, `srcmps`,
`srcmps-first`, `srcmps-oversample`, `sdc`, `sdc-oversample`, `fit-zipup`,
`fit-projector`, and `fit-oversample`. Use the qualified `quimb-fit` or
`mpo-fit` spelling for Quimb's FIT compressor because bare `mode="fit"`
remains the historical DMRG alias. These modes compress the ket and bra
physical layers independently through `gate_nonlocal_opt` except for the
native bare/default two-site `mode="mpo"` route described above; they do not
turn MPO evolution into an MPS-only `mix`, `su`, `perm`, or `swap` algorithm.
On a native Symmray MPO, all of these aliases intentionally select the
existing block-aware SVD path so no dense auxiliary MPO is constructed.

For dense local-term construction, `ham_tn.to_mpo(...)` performs a small
structural deparallelization/delinearization sweep before the requested Quimb
compression. In the shared automaton route this follows exact
prefix/future-channel sharing; in `compress="term"` mode it runs before every
per-term SVD, preventing each direct-sum addition from carrying redundant
channels into the next compression. The pass is exact to floating-point
roundoff, runs only for NumPy-backed operator tensors, and is skipped for
Torch/autodiff and native Symmray data so their backend metadata and gradients
remain intact. The same builder policy also applies when the result is
converted with `to_pepo(...)`.

The public `ham_tn.to_*` builders default to `compress="term"`: terms are
accumulated and compressed one at a time. Pass `compress=True` or
`compress="auto"` to request the workload-aware route, or
`compress="automaton"` to force finite-state assembly.

Pass `progbar=True` to `ham_tn.to_mpo(...)` (or `to_pepo(...)`) for an
MPS-style construction bar. In a term-by-term build, it advances once per
term and displays `chi=current/cap`; `peak` reports a temporarily larger
direct-sum bond before that term's compression. The default is `False`, so
existing output remains quiet.

For an explicit neutral term collection, arbitrary one- or multi-site support
is accepted:

```python
term = fermion.operator_term(
    [(1.0, ((0, "create_up"), (2, "annihilate_up")))],
    sites=(0, 2),
    add_hc=True,
)
mpo = fermion.build_mpo({(0, 2): term}, L=3)
```

The Jordan-Wigner compatibility path remains available by passing
`fermionic=False` to the same builder, together with the matching
`SymHamiltonian.jw_trotter_gates(...)` stream:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
edges = [(0, 1), (1, 2)]
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)
mpo = fermion.build_mpo(
    edges, L=3, t=1.0, U=2.0, mu=0.1, fermionic=False
)

opt = pepsy.MpoOptimizer(
    mpo,
    hamiltonian.jw_trotter_gates(0.01),
    chi=16,
    mode="svd",
)
mpo = opt.run(progbar=False)
```

Symmray gates keep their charge and dual metadata and are not coerced to dense
arrays. Native graded gates from `Fermion.strang_gate_stream(...)` are accepted
as well; even gates are adapted explicitly to the MPO's Jordan-Wigner
convention. `mode="svd"` and native `mode="mpo"` use symmetry-aware direct
compression, while native `mode="dmrg"` and its `dmrg1`/`dmrg2`/`dmrg3`
schedule aliases use block-aware FIT. All three avoid
generic dense auxiliary MPO compression and bond padding, which do not support
multi-sector Symmray bonds reliably.

Native MPO tensors retain their Symmray graded metadata throughout replay and
compression. `mode="svd"` and `mode="mpo"` use the block-aware direct SVD path
for native Symmray MPOs, while `mode="dmrg"` and the named DMRG schedules use
native block-aware FIT; the
optimizer does not require a dense conversion of the input MPO.

For dense Quimb gate payloads, the public MPO convention is output/input
ordering. Internally the effective ket operator is therefore `G.T`. The exact
dense mappings are:

```text
G                 -> G.T @ O @ G.conj()
(G, None)         -> G.T @ O
(None, B)         -> O @ B.conj()
(G, B)            -> G.T @ O @ B.conj()
```

Thus the bare entry is the API shorthand for unitary conjugation, but it is
not the same raw-array convention as `MpsOptimizer`. To apply a standard
linear-algebra operator `A` as `A @ O @ A†`, pass `A.T` as the MPO gate. To
apply `A† @ O @ A`, pass `A.conj()`. Native Symmray gates carry their own
graded metadata and follow the native block-aware path rather than this dense
transpose rule. `MpoChannelEvent` uses the explicit standard mapping
`sum_a w_a K_a @ O @ K_a†`.

Backend conversion is explicit and state-derived, matching the MPS helper:

```python
gate = opt.to_backend(gate)
```

The converter follows the live MPO's backend, dtype, and device and returns an
already-compatible payload by identity. Dense payloads cannot be generically
cast into a native Symmray MPO because charge and fermionic metadata would be
lost; construct those gates natively instead.

For MPO DMRG, `fit_block_size=2` is the default native two-site FIT update.
`fit_block_size=3` enables the corresponding three-site effective tensor and
two direction-aware SVD splits; an interval containing only two sites
automatically uses the two-site update. The optimizer forwards `cutoff`,
`cutoff_mode`, `chi`, and `fit_sweep_sequence` to every output FIT split.
The first one or two sweeps can use the three-site block through
`fit_three_site_sweeps`; remaining sweeps use one-site refinement. Batched
targets accept `fit_max_span="auto"` to split disjoint gates before a wide
active window is formed. `cutoff="auto"` selects a dtype-aware cutoff.
`target_cutoff` controls only construction of the disposable target MPO, so
target construction and output compression remain separate choices. Use
`fit_block_size=1` to retain the legacy fixed-rank one-site path.

The named modes are schedule aliases over this same MPO/FIT implementation:

| mode | warm-up block | warm-up policy | following sweeps |
| --- | --- | --- | --- |
| `dmrg1` | two-site | exactly two sweeps, unless the active window or full MPO is already at its attainable rank ceiling | one-site FIT; phase latches after full-chain saturation |
| `dmrg2` | two-site | `fit_adaptive_sweeps` (default two) | one-site FIT |
| `dmrg3` | three-site | `fit_adaptive_sweeps` (default two) | one-site FIT |

The aliases normalize the backend to `opt.mode == "dmrg"` while retaining the
requested schedule in `opt._dmrg_mode_alias`, matching `MpsOptimizer`'s API
shape. Passing `fit_block_size` does not override a named mode; the mode's
block size is authoritative. `fit_single_pair_fast_path=False` is the default,
matching `MpsOptimizer`; `dmrg2` still enables the automatic shortcut for an
adjacent two-site window, while setting the option to `True` enables it for
every DMRG schedule. The generic `mode="dmrg"` path now follows the MPS
schedule as well: two- or three-site FIT uses `fit_adaptive_sweeps` as an
adaptive block warm-up and hands the remaining `n_iter` budget to one-site
refinement. For a long-range window, the generic schedule uses the fixed
canonical handoff after the block phase, matching the MPS behavior. As in
`MpsOptimizer`, an under-capacity `dmrg1` window spanning at least three sites
requires `n_iter >= 3` so its two growth sweeps have room for one-site
refinement.

For dense DMRG windows, FIT uses an isolated compressed MPO warm start by
default. The policy is controlled by `fit_init_strategy`: `"direct"` uses the
current MPO, `"guess-src"` (also written `"guess_src"`) uses Quimb's seeded
source compressor, `"guess-<method>"` accepts any of the MPO compressor names,
and `"random"` / `"random_expand"` provide deterministic dense random starts.
The exact FIT target is always built separately from the unmodified current
MPO. `fit_init_seed` controls the guess seed and `fit_init_rand_strength`
controls the explicit random strategies. `fit_mpo_guess=False` remains a
compatibility switch that disables the source guess for named
`dmrg1`/`dmrg2`/`dmrg3` schedules; use the explicit random or direct strategy
when that legacy switch is disabled.
The replay order is controlled by `fit_mpo_guess_order`, which defaults to
`"lower_upper"` (bra then ket); `"upper_lower"` (ket then bra) is also
available. In this API the lower MPO layer is bra and the upper layer is ket.
Native Symmray MPOs retain their block-aware FIT warm start rather than being
converted to dense random or Quimb source guesses.

The selected and requested initialization policies, compressor name, and
randomization details are available in `get_fit_diagnostics()` and
`get_fit_history()` under `fit_init_strategy`, `guess_method`,
`random_initialization`, and the compatibility field `mpo_fit_guess_used`.

As with MPS DMRG, `n_iter` counts FIT sweeps. `mode="mpo"` applies each gate
with one direct MPO compression step and does not perform variational sweeps;
the two modes therefore have different one-iteration behavior for a
non-local gate even when they use the same `chi` and SVD cutoff.

## Norm and run diagnostics

Every compressed multi-site segment records an automatic norm-survival event. The
event compares the observed canonical-center norm after compression with the
expected norm of the uncompressed gate target. For a provably unitary default
`U O U†` gate, the expected value is read from the live canonical center, so no
extra target contraction is needed. Explicit ket/bra pairs and non-unitary
gates use a disposable exact target, keeping physical norm changes separate
from truncation loss.

```python
diagnostics = opt.norm_diagnostics()
events = opt.get_norm_events()

print(diagnostics["fidelity"])
print(diagnostics["infidelity"])
```

`diagnostics["cumulative_fidelity"]` is the stable cumulative product of the
per-event `local_fidelity` values, evaluated in log space. The matching
`cumulative_infidelity` uses `expm1` for small losses. These are compression
fidelities measured from retained norms, not target-state overlaps. The live
represented MPO norm is reported separately as `norm`/`state_norm`, while
`cumulative_norm` is only the square-root retained-compression proxy. The
progress bar's `~F` field uses the cumulative compression fidelity.
`get_fidelities()` remains the legacy normalized-MPO-norm history and is not
the compression ledger. Norms are reported at their physical scale: for an
identity MPO on `L` qubits, `norm_sq`/`state_norm_sq` are `2**L` and
`norm`/`state_norm` are `sqrt(2**L)`. Event fields
`expected_norm_sq`, `observed_norm_sq`, and `target_norm_sq` expose the same
unscaled squared norms; their ratios, rather than unit-normalized values, form
the local and cumulative fidelity metrics. Every compressed multi-site
`mode="mpo"` event contributes an event with the same local-fidelity fields.
The event also retains the scaled norm measurements as
`*_norm_mantissa`/`*_norm_exponent`; local ratios are formed from those pairs in
log space before any display-scale reconstruction.

If lower-level code accesses `opt.p` and moves its canonical centre directly,
call `opt.sync_canonicalization()` before resuming replay so `opt.info_c` is
rebased to the live MPO. With no compression events yet, the cumulative
fidelity fields are `None`; they are not a claim that a compression has been
measured.

DMRG FIT controls are exposed directly through `run`: `fit_min_iter`,
`fit_rtol`, `fit_patience`, `fit_finite_check`, `timing`,
`timing_sync_device`, and `fit_collect_split_diagnostics`. Timing is opt-in;
`timing=False` does not read the profiling clock or construct a device
synchronizer. With `timing=True`, `get_run_timing()` follows the MPS schema
with `stages`, per-sweep `fit_steps`, aggregate `fit_totals`, backend metadata,
and the latest `fit_diagnostics` record. The compact legacy `fit_calls` and
`fallback` fields remain available. The latest local FIT record is available
from `get_fit_diagnostics()`, and the complete replay history from
`get_fit_history()`.

`atomic=True` restores the optimizer state when replay fails. Set
`fit_fallback="svd"` or `fit_fallback="mpo"` to restore the pre-run state and
replay the complete stream through a direct backend if DMRG FIT raises. The
fallback is automatically routed through the native block-aware SVD path for
Symmray MPOs. With `inplace=True`, the optimizer state is restored, but an
external reference to the original MPO may already have observed in-place
updates.

`transactional_steps=True` adds a smaller rollback boundary around every DMRG
gate or batch. Thus `atomic=False` can retain completed earlier updates while
restoring only the failed local FIT trial. A configured `fit_fallback` is also
attempted at that local boundary; `get_fit_history()` records the failed FIT
and `channel_diagnostics()` reports each fallback. This is the useful setting
for long streams where one ill-conditioned window should not discard the
whole replay.

For dense MPOs, `fit_target_strategy="auto"` uses a lazy layered target: gate
layers are split onto their physical endpoints and FIT contracts the resulting
network without repeatedly forming a chi-capped intermediate MPO. Set
`fit_target_strategy="mpo"` to materialize the disposable target back to one
MPO tensor per site, or use `"layered"` explicitly. Native Symmray MPOs use
the block-aware `"mpo"` representation; requesting `"layered"` for native
data raises rather than dropping symmetry metadata. `target_cutoff` affects
only disposable target construction, while the output `cutoff` remains the
compression policy.

Gate streams are compiled once at construction and queue updates. The
immutable plan is inspectable with `opt.compile_gate_stream()` and includes
event type, arity, support span, and prepared-payload cache size. Dense gate
transposes are cached by stream source identity and can be released with
`opt.clear_gate_cache()`; state-dependent native gate adaptation remains
uncached so charge and layout metadata are always taken from the live MPO.

Long-range streams can install a physical order before replay:

```python
opt = pepsy.MpoOptimizer(mpo, gates, chi=32, mode="dmrg")
plan = opt.gate_stream_layout(L=mpo.L, order="quality")
opt.apply_layout(plan, cutoff=1e-12)
out = opt.run(progbar=False)
```

`apply_layout` performs the required logical SWAP conjugations, records their
norm survival, and remaps later gate supports through `site_map`. Reordering
an already entangled MPO is rejected unless
`allow_lossy_reorder=True`, because the layout installation itself may need
compression. The layout is persistent for that optimizer; construct a fresh
optimizer to compare another order.

## Deterministic operator channels

`MpoChannelEvent` provides the operator-channel API for MPO replay. It applies

\[
  O \mapsto \sum_a w_a K_a O K_a^\dagger
\]

as one deterministic event. Build it from the shared noise channel objects or
directly from Kraus matrices:

```python
channel = pepsy.TrajectoryChannel.amplitude_damping(0.25)
event = pepsy.MpoOptimizer.channel_event(channel, where=0)

opt = pepsy.MpoOptimizer(mpo, gates=[event], chi=32, mode="dmrg2")
out = opt.run(progbar=False)
```

`semantics="sum"` is the only executable MPO meaning. Passing
`semantics="sample"` raises a clear error: sampled branches are an MPS
trajectory operation and require branch probabilities, branch normalization,
and per-trajectory state. The MPO path never silently selects one Kraus
branch. `MpoChannelEvent.from_channel` uses unit weights for a state-dependent
Kraus channel and declared mixture probabilities for a classical mixture.

Channel trace preservation is deliberately separate from Hilbert-Schmidt norm
survival. `opt.get_norm_events()` and `opt.norm_diagnostics()` report
compression retention, while `opt.get_trace_events()` and
`opt.channel_diagnostics()` report the completeness residual
\(\|\sum_a w_a K_a^\dagger K_a-I\|\), input/target/retained traces, and any
trace change caused by output compression. A non-trace-preserving operator
therefore remains physically visible instead of being misreported as
compression infidelity.

`ham_tn.build_mpo(..., fermionic=True)` is also routed to the native
`Fermion.build_mpo(...)` entry point. `Fermion.to_mpo(...)` remains a
compatibility alias. Use `to_backend=...` on the model-facing builder when
the stored blocks must be moved to Torch or another supported backend.

Native MPO assembly/replay is also measurable with a native fermionic MPS.
`MpsEnergyOptimizer` applies the native MPO sitewise as a factorized graded
MPO-MPS network, preserving Symmray ordering while retaining MPO bond scaling.
Repeated evaluations reuse the optimizer's contraction paths. Optional
controlled truncation is available through
``native_mpo_compression={"max_bond": ..., "cutoff": ..., "method": "svd"}``;
the default remains exact and uncompressed.

To compress an existing MPO without replaying gates, use
`MpoOptimizer(mpo, gates=[], chi=...).compress()`. Symmray may retain a small
sector-multiplicity overshoot above the requested numeric bond cap.

## Parameterized higher-order MPO construction

`MPOBasis` is the reusable API for a Hamiltonian whose couplings change during
an optimization:

```python
basis = MPOBasis.from_pauli_terms(
    L,
    [((i, i + 1), "ZZ", MPOParameter("J")) for i in range(L - 1)]
    + [((i,), "X", MPOParameter("hx")) for i in range(L)],
)
U = basis.exp(
    -1j * dt,
    {"J": J, "hx": hx},
    order=2,
    mode="exact",
)
# Fast analytical approximation: Algorithms 1, 2, and 4.
U_fast = basis.exp(
    -1j * dt,
    {"J": J, "hx": hx},
    order=2,
    mode="folded",
)
```

`mode="base"` applies Algorithms 1--2. `mode="exact"` applies Algorithms 1--3,
where Algorithm 3 adds selected order-`N + 1` terms without increasing the
analytical history bond dimension. `mode="folded"` applies Algorithms 1, 2,
and 4 without the selected next-order replay. `mode="hybrid"` additionally
enables Algorithm 4 after the exact extension. `mode="auto"` selects `exact`
through order 2 and `folded` above order 2. The historical spellings
`algorithm4`, `optimal`, and `approximate` remain accepted as aliases. These modes, together
with `order`, `max_bond`, `on_exceed`, `cache_history`, and `history_storage`,
are shared by `basis.exp()` and `basis.compile_exp()`. The names
`basis.time_evolution()` and `basis.evolution_mpo()` remain compatibility
aliases. `MPOProductTerm` also accepts arbitrary
one-dimensional supports, for example `(0, 1, 3)` with operators `"XYZ"`.

The first build compiles the first-degree MPO topology and symbolic history
plans. Later calls reuse level indices, reachability information, local
gather/index metadata, and merge/insertion plans; local tensors are rebuilt
from the supplied numerical parameters. This separation keeps Torch and JAX
autodiff graphs correct. The plan state is visible through
`basis.cache_info["history"]`, including `compression_plan_orders`,
`tensor_plan_orders`, `extension_plan_orders`, and `extension_plan_batches`.
`cache_history=False` avoids retaining a new raw
history topology for one-off large builds; the streaming compatibility mode
also keeps that topology ephemeral during generation before assembling the
current MPO needed by Algorithms 1--3. Use `basis.clear_history_cache()` to
release cached history orders while retaining the compiled coefficient basis.

For compiled numerical kernels, `MPOBasis.exp_arrays()` returns backend-native
tensor tuples without crossing the Quimb wrapper boundary. Use
`MPOBasis.exp_batch()` for coefficient arrays with shape
`(batch, number_of_terms)`; current JAX and Torch backends use native `vmap`
when available and retain an autodiff-safe fallback loop otherwise. These
interfaces share structural caches but never cache parameter-dependent tensor
values.

> API details are maintained as handwritten Markdown in this page.
