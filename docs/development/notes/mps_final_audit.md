# MpsOptimizer correctness, performance, and API audit

Date: 2026-09-08. Reviewed commit: `4f88acb` on `develop`.
The original review findings below refer to that commit. The subsequent
authorized fixes are recorded in "Implementation follow-up" at the end;
the original reproductions are retained as historical evidence.

## Original assessment

The existing regression suite passes, including the earlier reset/leakage,
global branch-budget, cap, scaled DMRG measurement, and rare Z-outcome fixes.
Additional probes found uncovered correctness issues. The implementation
should not yet be described as correct for every supported control or as
performance-optimal. Fix the first two findings before relying on unusual
postselection or retrying rejected transient-layout calls.

## Original confirmed findings, in priority order

### 1. High: rare X/Y outcomes still suffer cancellation

Location: `src/pepsy/optimizers/mps/optimizer.py:6767` and `:6800`.
`_measurement_probabilities` recomputes a rare branch using its projector,
but `_state_operator_expectation` evaluates that projector against a reduced
density matrix. For non-diagonal projectors, this still subtracts large,
nearly equal terms. The earlier Z-only regression does not cover this case.

Reproduction in NumPy complex128:

```python
plus = np.array([1., 1.]) / np.sqrt(2)
minus = np.array([1., -1.]) / np.sqrt(2)
psi = np.sqrt(1 - 1e-18) * plus + np.sqrt(1e-18) * minus
opt = MpsOptimizer(
    qtn.MatrixProductState.from_dense(psi, [2]),
    [("measure", "X", 0, -1)], chi=2,
)
opt.run(cutoff=0.)
```

The dense amplitude reference gives probability `1.000000054e-18`;
the recorded probability is `2.775557562e-17`. The compression ledger then
reports approximately `0.963971199` loss for an untruncated projection.
Y-basis reproduction gives the same discrepancy. At probability `1e-13`,
the X example already reports roughly `5.88e-4` false compression loss.

Recommended correction: measure the norm of the projected local amplitude,
with scaling as needed, instead of contracting `Tr(rho @ projector)` for
rare outcomes. Test X/Y and multi-Pauli projectors, coalesced probabilities,
and the resulting compression ledger. Do not reinstate a positive-probability
rejection threshold.

### 2. High: invalid transient-layout calls can change the logical state

Location: `optimizer.py:5391` through the later run-option validation;
the restoration `finally` blocks begin only after that validation.

Starting from `|100>`, queue `[("h", 0)]`, request an explicit reverse layout
with `site_order=(2, 1, 0)` and `site_map={2: 0, 1: 1, 0: 2}`, and pass
`normalize_final=True` without `non_unitary=True`. The call raises the
expected validation error, but the live readout has become `|001>` while
`logical_order` still reports `[0, 1, 2]`. No queued H gate was executed.

Recommended correction: resolve and validate all run options before reordering,
or extend the restoration transaction around every operation after reordering.
Check invalid FIT options and compressor overrides through the same boundary.
Persistent layouts intentionally remain installed and need a separate contract.

### 3. Medium: represented scale can overflow during measurement

Location: `optimizer.py:6816`, the DMRG measurement exponent conversion, and
the instance norm wrapper near `:7594`.

A finite raw `|++>` MPS with `p.exponent=400` raises `OverflowError` during
an ordinary forced Z measurement. Its normalized Born probabilities and
normalized projected state are well-defined; materializing `10**exponent`
is unnecessary for those quantities. This matters for long nonunitary runs
that intentionally keep raw tensor values bounded with an exponent.

Recommended correction: retain mantissa/exponent or logarithmic norm pairs
through control/compression ratios, reconstructing display values only at
the presentation boundary. Validate large positive and negative exponents.

### 4. Medium API pitfall: shot replay silently ignores top-level solver options

Location: `optimizer.py:5216` and `_run_shots`; documentation does prescribe
`run_kwargs`, so this is a hazardous interface asymmetry rather than an
undocumented solver policy.

Observed actual FIT calls for
`run(shots=2, strategy="independent", n_iter=3, cutoff=0., fit_rtol=None)`:
both shots used `n_iter=8`, `cutoff=1e-12`, and `rtol=1e-9`.
Putting those three controls in `run_kwargs` correctly used `3`, `0.0`, and
`None`. The parent mode and finite-check policy have separate forwarding.

Recommended correction: provide one resolved replay-options path shared by
ordinary and shot execution, with explicit precedence for nested overrides;
alternatively reject conflicting/ignored top-level options. In current code,
place per-trajectory solver settings inside `run_kwargs`.

### 5. Low: FIT diagnostic lifecycle contradicts the getter contract

Location: `optimizer.py:3528` and `:5838`.
After a successful DMRG2 run, switching to direct, queuing a one-site H gate,
and running again leaves `get_fit_diagnostics()` populated with the previous
FIT record. The getter promises `None` for modes that do not use FIT.

Clear the record at the appropriate run/mode boundary or explicitly expose
it as historical data with run provenance. Separately, `get_run_timing()`
also retains a previous timed run after an untimed run, but its docstring
explicitly says "most recent opt-in" record: this is expected historical
behavior, not an additional correctness bug.

## Performance findings

1. **Exponential measurement preparation is confirmed.** `_state_expectation`
   builds a dense `2**k` by `2**k` Pauli operator before the bond-two collapse
   MPO is used. An eight-site Z measurement allocated a `(256, 256)`
   complex128 operator (1 MiB) in the probe. The same single-array formula
   is 256 MiB at 12 sites and 64 GiB at 16 sites; reduced density matrices
   and temporaries add memory. These larger cases were not allocated.
   Use factorized Pauli/product or MPO expectation contraction, paired with
   a stable projected-amplitude norm for rare outcomes.
2. **Always-retained support histories scale with events times active sites.**
   `_record_effective_event` sorts/copies the entire active support into a
   tuple for every event. Ordinary replays of one H per site followed by
   1,000 H gates on site zero retained 571,240 bytes of distinct support
   tuples at length 64 and 2,361,448 bytes at length 256. These counts exclude
   dictionary/list overhead and tensor data, and count shared tuple references
   only once. Prefer deltas, shared unchanged support snapshots, or configurable
   history retention. Cumulative summaries can remain available.
3. **Coalesced branch copies still scan and copy history.** Each
   `_copy_for_trajectory_branch()` invokes the constructor/canonicalization
   path and then a second explicit orthogonality-center discovery. Spies
   counted two center scans per copy at lengths 16 and 64. It also deep-copies
   several growing histories. Investigate a trusted internal clone that owns
   mutable tensors and preserves proven isometries and immutable history
   prefixes. Do not merely copy center labels onto recanonicalized tensors.
4. **Public norm diagnostics can be expensive when polled frequently.**
   `norm_diagnostics()` scans histories and calls the represented full-network
   `p.norm()`; spies confirmed one class norm call per invocation. This differs
   from the optimized local control/Kraus paths. Reuse the tracked center and
   incremental summary statistics when their invariants are valid.
5. **FIT/SVD cost remains workload-dependent.** Larger chi, long gate spans,
   three-site blocks, guess compression, and overlap diagnostics add work.
   Passing correctness tests does not establish an optimal compressor.
   Benchmark direct versus DMRG2/3 at equal output accuracy, on the intended
   device and gate distribution. Existing no-clock/no-finite-scan regressions
   pass; retain those optimizations.

No GPU wall-time benchmark was performed. The measurements above are allocation
sizes and operation counts, not a general throughput claim.

## API and architecture assessment

`run` has 71 parameters excluding `self`, covering replay, legacy FIT aliases,
layouts, controls, shots, workers, MPI, checkpointing, and retention. The shot
forwarding asymmetry demonstrates the maintenance cost of this surface.
Introduce structured replay/shot/checkpoint options behind a compatibility
adapter rather than adding more independent forwarding dictionaries.

The direct default and `mpo` alias are consistent. Bare `fit` continues to
mean Pepsy DMRG; `quimb-fit` selects Quimb's compressor. `normalize()` preserves
represented scale through `p.exponent`; it is not physical state normalization.
Compression-survival diagnostics are not a global exact-state overlap. These
distinctions are documented and should remain explicit in the public API.

## Verification and scope

- MPS optimizer, dynamic controls, trajectory noise, FIT copy/performance,
  replay metadata, nonfinite policy, FIT schedules/hot paths, sampler, and
  native symmetric tensors: **1,135 passed, 7 skipped**, including the slow
  cases in those files (191.64 seconds).
- Public API, package layout, MPI unit tests, Quimb compatibility, and
  contraction-dependency checks: **113 passed** (2.18 seconds).
- The uncovered failures above were reproduced independently of that suite.
  Passing existing tests therefore does not invalidate these findings.
- Repository Ruff passed. No implementation, dependency, or skill was changed.
- This is not a full-repository proof or a live multi-rank MPI/CUDA/CuPy test.
  Native coverage is limited to the installed backends and tested sectors.

## Dependency audit

Installed versions: Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev6+g08fe1a3a1`, and Symmray
`0.3.2.dev6+ga17699db6`. Inspected installed canonical expectation,
canonicalization, shallow/deep copy, and nonlocal gating signatures, plus
NumPy/Torch/JAX copy, SVD, QR, and finite-check dispatch.

Reviewed [Quimb's changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray](https://github.com/jcmgray/symmray). The requested Symmray
Abelian-array documentation URL again failed to load. Public development docs
are ahead of some installed checkouts; the tested installed APIs are the
authority for this report.

**Adopt/retain:** current public canonicalization and backend dispatch APIs.
**Compatibility shim/prototype:** none introduced.
**Defer:** new upstream compressors and automatic dependency-driven behavior
changes. The confirmed failures are in Pepsy's numerical formulation and
policy/lifecycle boundaries; no upstream upgrade was attempted as a substitute.

## Implementation follow-up

All five confirmed findings and the concrete avoidable performance costs
identified above have been addressed in the working tree:

- Dense Pauli probabilities use normalized projected amplitudes. A disposable,
  untruncated Clifford parity circuit handles multi-site products with only
  one-/two-qubit operators. Ordinary small-support expectations retain Quimb's
  canonical evaluator; larger dense Pauli expectations use the parity path.
  The physical projection still uses the selected bond-two MPO/FIT compressor.
  Single-site dense Kraus probabilities also use amplitudes to avoid a small
  Gram-form weight suffering the same cancellation.
- Measurement, reset, and Kraus ratios cancel the represented exponent before
  division. Events retain mantissa/exponent provenance. Display overflow or
  underflow yields infinity/zero without changing physical probabilities or
  compression ratios. Independent and coalesced amplitude damping at exponents
  +/-400 both preserve `[0.7, 0.3]` probabilities and zero compression loss.
- Transient reordering occurs after numerical-option validation and inside
  the existing restoration boundary. Invalid normalization, FIT patience,
  and compressor requests leave the logical state unchanged.
- A single shot replay-option resolver inherits ordinary top-level controls,
  then applies explicit `run_kwargs` overrides. Identity-based deprecated
  sentinels are omitted before serialization. The broad public signature is
  retained for compatibility; no breaking API redesign was introduced.
- Mode changes clear FIT records, and direct-mode getters return `None`.
  Timing continues to expose the most recent opt-in timing record, as documented.
- Unchanged support sets reuse immutable tuples. In the original microprobe,
  distinct support-tuple storage fell from 571,240 to 19,240 bytes at length 64
  and from 2,361,448 to 273,448 bytes at length 256. This measures tuple storage,
  not total process memory. Complete history output still has an unavoidable
  cost proportional to the data requested.
- Trusted internal dense branch clones own arrays and preserve tensor
  isometries without constructing/recanonicalizing a new optimizer. Spies
  now report zero center-discovery scans, versus two previously. Internal
  append-only history prefixes share records; public coalesced leaves detach
  those records once so users can mutate leaf histories independently.
- Norm diagnostics use the tracked center. The norm spy now reports zero
  global norm calls. `include_history=False` processes only newly appended
  records and returns scalar summaries; state replacement and mixed rollback
  invalidate this cache. Full historical output stays compatible by default.

Added 28 regressions in `tests/test_mps_audit_fixes.py`. These cover rare X/Y
outcomes down to `1e-18`, NumPy/Torch/JAX paths, coalesced rare weights,
exponents +/-400, a 24-site measurement with dense operator construction
forbidden, invalid layout calls, shot-option precedence, branch ownership and
isometries, published-history independence, support snapshot sharing, compact
diagnostic parity, and FIT diagnostic lifecycle. Existing measurement route
tests now distinguish lossless probability-preparation gates from the selected
physical projector compressor.

Rechecked the upstream sources above during implementation; installed versions
are unchanged. Additionally inspected `Tensor.gate(G, ind, preserve_inds=True,
transpose=False, inplace=False, transposed=None)`, `Tensor.norm(squared=False,
**contract_opts)`, and `Tensor.modify(**kwargs)`. Adopted these existing APIs;
no shim, dependency edit, or new upstream compressor was introduced.

Remaining limits: native/unknown arrays retain conservative copy and graded
projector routes. Floating-point conditioning still limits the relative
accuracy of arbitrarily small probabilities. FIT/SVD cost remains dependent on
chi, gate span, backend, and requested accuracy; it cannot be eliminated by an
API cleanup. No universal optimum or GPU throughput claim is made.

### Final validation of the fixes

- Combined MPS/MPO, trajectory/control, FIT schedule/performance/metadata,
  sampler, native symmetric tensor, and MPI-unit suites: **1,346 passed,
  7 skipped**, including the selected slow tests (190.42 seconds).
- Final focused regressions after the Pauli-input validation/docstring
  cleanup: **28 passed**.
- Default smoke: **128 passed**. Repository Ruff, MPS skill validation,
  skill catalog validation, and whitespace checks passed.
- Counts overlap. No live multi-rank MPI or GPU performance benchmark was run.
  Validation describes the implementation accompanying this report.
