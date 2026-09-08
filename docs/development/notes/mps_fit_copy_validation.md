# DMRG finite checks and active-window copies — 2026-09-08

## Scope and behavior

`MpsOptimizer.run(finite_check=False)` now leaves per-sweep FIT array scans off.
The option is forwarded through DMRG, mixed-mode FIT, measurement FIT, and shot
workers; explicit shot `run_kwargs` overrides the outer value. Direct
`FIT.run_gate` also defaults to False. Enabling the scan or a direct FIT checking
callback emits a RuntimeWarning about overhead and accelerator synchronization.
Scalar convergence norms, unitary norm-accounting guards, and explicit periodic
quality checks retain their separate policies.

Private dense guess builders and DMRG rollback snapshots now own tensor metadata
for the whole chain and independent arrays only in the active endpoint span.
Exterior data is shared read-only. Two-site and sub-MPO gates, batches, and
measurement FIT use the same helper. Native symmetry/fermionic arrays and
unknown backends retain full deep copies. Public `guess(inplace=False)` and
mixed-mode transaction copies retain their existing ownership contracts.

Quimb's `Tensor.modify(data=...)` clears `left_inds` unless explicitly supplied;
the helper preserves it. Autoray's Torch copy detaches gradients, so the helper
uses Torch's differentiable `Tensor.clone()` instead. Other audited dense
backends use their registered array-copy operation.

## Upstream compatibility audit

Installed versions: Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev6+g08fe1a3a1`, and Symmray
`0.3.2.dev6+ga17699db6`, inspected in the activated genpy environment.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and the
[Symmray repository](https://github.com/jcmgray/symmray). The requested
[Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
could not be retrieved by the browser; native behavior was checked against the
installed package and native regression suite instead.

- **Adopt:** Quimb's public shallow network copy with independent Tensor
  metadata, plus explicit active-array copies. Inspected installed
  `Tensor.copy(deep=False, virtual=False)` and
  `TensorNetwork.copy(virtual=False, deep=False)` implementations.
- **Adopt:** the existing local compression boundary. Installed
  `gate_nonlocal(G, where, dims=None, method='direct', transpose=False,
  info=None, inplace=False, **compress_opts)` delegates to
  `gate_with_submpo(..., inplace_mpo=False, **compress_opts)`. Its implementation
  canonicalizes around the support, partitions the endpoint span, compresses
  that region, and rejoins it. Canonicalization replaces tensor arrays; it does
  not write through shared exterior array storage.
- **Compatibility shim:** local Torch clone selection instead of Autoray's
  registered `copy`, whose installed body is `x.detach().clone()`. Queried
  `get_lib_fn(backend, 'copy')` and `'isfinite'` for NumPy, Torch, and JAX.
  NumPy uses `numpy.copy`; JAX uses `jax.numpy.copy`. No global registrations
  or installed dependency files were changed.
- **Defer:** new Quimb compression algorithms/random generation changes;
  this task preserves SRC selection, seeds, target construction, and sweeps.
- **Defer:** broader Cotengra and Symmray changes; contraction planning and
  native split/metadata policies are unchanged. Native arrays retain full
  deep-copy protection rather than adopting unverified sharing.

## Validation and measurement

New `tests/test_mps_fit_performance.py` regressions cover disabled scans with
timing enabled, enabled scans and warnings, non-finite detection, measurement
and shot forwarding, NumPy/Torch/JAX storage isolation, canonical metadata, Torch
non-leaf gradients, dense full-copy equivalence, and injected in-place FIT
failure followed by an uncorrupted MPO fallback.

A 32-site NumPy complex128 MPS with bond cap 8 and active sites 14–17 copied
8,192 array bytes instead of 55,936 per helper invocation: an 85.4% reduction
in copied array payload for this fixture. Metadata copies and live shared data
remain; this is not a total-memory or wall-time speedup claim. A full-chain
active window receives no array-copy volume reduction.

Final validation:

- New performance/ownership regression file: 9 passed.
- Full `test_optimize_mps.py`, `test_quimb_compat.py`, and
  `test_contraction_dependencies.py`: 643 passed.
- Selected native MPS FIT/DMRG tests in `test_symmetric_tensors.py`: 11 passed.
- Default smoke suite: 128 passed.
- `python -m ruff check src tests` and `git diff --check`: passed.

These suites overlap; their counts are not a unique-test total. No full
repository extended suite or GPU runtime benchmark was run.

## Follow-up FIT diagnostics review

Ordinary FIT diagnostics read existing convergence and initialization metadata.
The public accessor defensively copies that metadata without copying MPS arrays.
Optional target-overlap contractions remain disabled with timing both off and
on across DMRG, DMRG1/2/3, and mixed replay. Mixed replay omits these contractions.

Fixed optional overlap reporting so NaN or infinity produces `None` values and
`fit_overlap_error`, rather than being clipped to a valid fidelity. This scalar
guard runs only after an explicitly requested overlap contraction and does not
enable per-sweep finite scans.

Follow-up validation after this fix:

- Focused overlap, diagnostics, disabled-clock, and norm-survival tests: 23 passed.
- Accessor isolation, timing records, layout/readout, and target-overlap tests:
  5 passed.
- Performance regressions, public API, package layout, and sampler suites:
  152 passed, 6 skipped.
- Ruff and `git diff --check`: passed.

## Gate schedule and default alignment (2026-09-08)

Named `dmrg3` now forwards one two-site transition sweep through the shared
MPS `_run_fit_gate` boundary, including gate batches and measurement FIT.
The default fixed eight-sweep schedule is `3,3,2,1,1,1,1,1`; `dmrg2` retains
`2,2,1,1,1,1,1,1` and its adjacent-pair exact-update exception. Generic DMRG
keeps its existing handoff. The transition consumes the total budget and
cannot be skipped by tolerance convergence when refinement budget remains.
Every block-size transition resets convergence history. Existing cache
compatibility checks conservatively rebuild environments for 3-to-2; the
2-to-1 transition reuses the existing terminal-boundary extension.

Standalone gate FIT defaults now match the optimizer's sweep count, two-site
block size, RL directions, two warm-up sweeps, min_iter=2, patience=2, and
automatic dtype-aware relative tolerance. Split diagnostics are opt-in.
Tests of legacy fixed one-site or direct three-to-one behavior now request
those controls explicitly. `run` and `run_eff` defaults are unchanged.

The upstream audit was repeated against the Quimb changelog, Autoray
repository, Cotengra docs/changelog, and Symmray repository listed above;
the Symmray Abelian-array documentation endpoint again returned an error.
Installed versions remain Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev6+g08fe1a3a1`, and Symmray
`0.3.2.dev6+ga17699db6`. Inspected installed `Tensor.split`,
`tensor_contract`, and `MatrixProductState.canonicalize` signatures and
NumPy/Torch/JAX SVD/QR dispatch. Classification: retain/adopt the existing
public split and contraction APIs; defer upstream compressor, layout, and
backend changes. No new compatibility shim or installed-package edit.

Validation for the schedule/default change:

- Complete MPS optimizer suite: 643 passed (within the combined replay run).
- MPO suite plus new schedule/default tests after explicit MPO policy forwarding:
  132 passed.
- Schedule/default, copy/finite-policy, public API, package layout, and sampler
  tests after the final changes: 156 passed, 6 skipped.
- Selected native MPS DMRG/FIT tests: 11 passed.
- Default smoke suite: 128 passed.
- Ruff, skill/catalog validation, and `git diff --check`: passed.

The first combined replay run exposed legacy expectations and MPO calls that
relied on omitted FIT defaults. Tests of those explicit legacy controls now
specify them; MPO forwards its existing schedule explicitly. The final
affected-suite reruns passed. Counts overlap. No full repository extended
suite or GPU timing benchmark was run; validation focused on both consumers
of the changed gate solver, native paths, and the public API.

## Four hot-path improvements (2026-09-08)

- Extended the reversed-sweep cache through only the missing terminal site
  for 3-to-2 transitions. The cache explicitly records two-site readiness;
  same-direction and unsupported transitions retain the rebuild path.
  The common extension helper continues to support 2-to-1 and 3-to-1.
- Dense ordinary/sub-MPO guess selection computes attainable-rank checks only
  for random initialization policies that depend on them. SRC, including
  `auto`, keeps its existing guess, seed, and compression controls.
- Copy capability classification is lazy and scoped to one replay, keyed by
  network and array type. Backend-preserving state replacements reuse it;
  `set_p` invalidates it, copies do not inherit it, and all replay exits
  restore the previous scope. Standalone copy-helper calls recheck their
  input. Native/unknown backends still receive full deep copies.
- With finite scans disabled, convergence transfers the terminal norm scalar
  directly without constructing a stacked vector. Per-sweep reads and the
  caller's non-finite norm guard remain active. Enabled finite scans retain
  their combined vector transfer.

The upstream audit was repeated using the same six source URLs recorded
above. Installed versions are unchanged. Probed public `TensorNetwork.copy`,
`Tensor.modify`, `tensor_contract`, `MatrixProductState.canonicalize`,
`autoray.to_numpy`, and `infer_backend` signatures, plus NumPy/Torch/JAX
`to_numpy`, `real`, `stack`, and `copy` dispatch. Classification: adopt/retain
these public APIs and the audited native contractions; defer upstream
algorithm changes. No new shim or installed-package edits were needed.
The Symmray Abelian-array documentation URL remained unavailable.

A NumPy complex128 24-site MPS (seed 714, chi 8, CNOT on sites 4 and 19)
used 130 site-overlap contractions with the new cache versus 143 with the
previous transition reconstructed through temporary monkeypatches. Tensor
data matched exactly, as did center metadata and FIT diagnostics. This is a
9.1% reduction in that contraction count, not a measured runtime speedup.

Validation:

- Combined MPS/MPO, schedule, copy-policy, and hot-path suites: 794 passed.
- Final hot-path regressions after state-replacement invalidation and an
  additional scalar non-finite test: 11 passed. These cover NumPy, Torch,
  JAX, U1U1 fermions, both sweep directions, full-rebuild equivalence, SRC
  selection, cache lifetime/failure cleanup, and allocation-free scalar reads.
- Selected native MPS FIT/DMRG suite: 11 passed.
- Default smoke suite: 128 passed.
- Ruff, skill/catalog checks, and `git diff --check`: passed.

Counts overlap. No full repository extended suite or GPU wall-time benchmark
was run; the shared FIT consumers and native/backend paths were checked.

## All-mode opt-in non-finite policy (2026-09-08)

At the user's request, `MpsOptimizer.run(finite_check=False)` now disables
runtime non-finite detection in every replay mode. This supersedes the
earlier sections' always-on scalar validation policy. Gate FIT skips both
active-array scans and scalar non-finite convergence guards; mixed replay
skips commit norm validation; normalization/stabilization and norm accounting
skip non-finite and unitary norm-consistency validation. Scalar calculations
needed by convergence and the norm ledger still execute. Input validation,
zero-divisor guards, and explicitly requested quality/overlap diagnostics
retain their own policies. Unchecked NaN ratios propagate as NaN instead of
being clipped into an apparently valid zero fidelity.

`finite_check=True` emits a performance warning, enables the runtime checks,
and scans final tensor data for every mode, including empty replays. The
check runs inside the timing scope, so an invalid final result records a
failed run. The validation flag is restored on all exits and shot workers
inherit the public flag unless their run kwargs override it. Mixed sticky
non-finite handling now also defaults to False; explicitly enabling it can
recognize non-finite exceptions raised by the backend. Backend linear algebra
may still raise its own numerical errors when checks are disabled.

This change gates existing validation code; it adds no upstream numerical
API or dispatch changes relative to the preceding audit.

Validation:

- Combined MPS/MPO, FIT schedule/hot-path, and non-finite-policy suites:
  812 passed.
- Final policy/hot-path regressions, including empty replay: 29 passed.
  Coverage includes dmrg/dmrg1/dmrg2/dmrg3, mix, mpo/direct/src/sdc,
  swap/perm/svd, su, and exact, plus scalar norm policy and failure cleanup.
- Selected native MPS FIT/DMRG tests: 11 passed.
- Public API, package layout, and sampler suites: 143 passed, 6 skipped.
- Default smoke suite: 128 passed.
- Ruff, skill/catalog validation, and `git diff --check`: passed.

Counts overlap; no full repository extended suite or GPU benchmark was run.

## Diagnostic warning clarification (2026-09-08)

Code comments, docstrings, API documentation, and the enabled warning now
explicitly describe `finite_check` as optional diagnostics: it is off by
default in every MpsOptimizer mode and is not required for normal
optimization. The warning explains the extra validation work and possible
device synchronization, and points to `finite_check=False` to avoid it.

MpsOptimizer emits one warning per enabled replay. Its disposable FIT
instances record that the owning replay has already warned, suppressing only
duplicate warnings while retaining all requested checks. Standalone FIT calls
still emit their own warning. No global warning filters or numerical backend
operations changed.

Validation: 39 focused copy/finite-policy, all-mode detection, and FIT hot-path
tests passed. Coverage includes multiple FIT calls per replay, timing enabled
and disabled, standalone FIT warnings, and enabled rejection of invalid data.
Ruff, skill/catalog checks, and `git diff --check` passed. The numerical suites
recorded above were not repeated for this warning/comment clarification.

## Replay metadata and FIT setup improvements (2026-09-08)

Implemented the next four reviewed improvements without changing circuit
initialization, sweep budgets, convergence tolerances, or diagnostic defaults:

1. Cache physical rank ceilings within each replay and prepare mixed FIT
   windows once. State replacement, caps, layout changes, mode changes, and
   canonical resynchronization invalidate metadata; changed chi or length
   also forces new ceilings. Actual active bond dimensions remain live.
2. Skip DMRG1's rank-dependent minimum-budget validation when at least three
   sweeps already suffice. Ordinary DMRG no longer computes an unused startup
   maximum. Mixed replay reuses the maximum measured for its chi guard through
   commit, history, and the next transaction; quality checks discard that
   cached maximum because repair may alter dimensions. Segment boundaries
   also start fresh. Required chi enforcement remains enabled.
3. Build layered FIT tag selections only when visited, retain target tensor
   ordering, and reject extra target tensors before checking site-by-site
   shortcut eligibility. Dense SRC/random guesses with independent active
   arrays retain the untouched source for rollback. A guess that aliases the
   source still snapshots before FIT; native warm-starts retain their early
   full copy. Full target-index separation remains intact.
4. Reuse backend/symmetry classification by weak network identity during
   replay and inherit it only through owned backend-preserving copies.
   Standalone calls inspect their inputs afresh, discarded networks remain
   collectable, and optimizer copies do not inherit replay caches. FIT setup
   combines its repeated native/dense routing scans into one metadata pass.

The cache scope restores prior state on all exits, including failures. These
changes do not enable non-finite diagnostics, timing, profiling, or extra
target-overlap contractions.

### Upstream audit

Repeated the audit of the Quimb changelog, Autoray repository, Cotengra
documentation/changelog, and Symmray repository linked above. The Symmray
Abelian-array URL again could not be retrieved. Installed versions remained
Quimb `1.15.1.dev39+g369d09b9d`, Autoray `0.11.1.dev1+gc56f64427`, Cotengra
`0.8.3.dev6+g08fe1a3a1`, and Symmray `0.3.2.dev6+ga17699db6`.

Inspected installed public `TensorNetwork.copy`, `reindex`, `select`,
`select_tensors`, `add_tensor_network`, `Tensor.modify`,
`MatrixProductState.canonicalize`, `max_bond`, `bond_size`, `phys_dim`,
`tensor_contract`, `autoray.infer_backend`, and `to_numpy` signatures.
Checked copy/reindex/selection implementations, weak-key support on actual
Quimb networks, and NumPy/Torch/JAX copy/conjugate/real/host-transfer dispatch.
Classification: **adopt/retain** these public APIs and the existing
differentiable Torch clone policy; **defer** new upstream compression,
contraction-planning, random-generation, and native algorithm changes.
No new compatibility shim or installed-package edit was needed.

### Boundary compatibility found during validation

The existing typed boundary convergence regression failed because the
previous gate-FIT default change also gave boundary two-site fitting a
block-to-one-site warm-up schedule. Loading the committed pre-change FIT
implementation reproduced the same four-sweep result instead of the
documented two-site convergence behavior. `CompBdy` now explicitly forwards
`adaptive_block_sweeps=None`, retaining its two-site updates until stopping.
The original numerical regression passes without weakening its expectation;
the wrapper keyword-forwarding test now checks this explicit policy.

### Measurement

Compared the changed optimizer/FIT against their committed pre-change source
in the same process, using NumPy complex128, 256 sites, initial bond and chi
8, and twelve gates spanning five sites each. State seed was 714; gate seeds
were 820 through 831. Timing and finite diagnostics stayed off. BLAS used one
thread; optimizer construction and garbage collection were outside the timed
region. Seven before/after pairs alternated order; the first pair was warm-up
and the table reports medians of the remaining six samples.

| Mode | Before | After | Time reduction |
| --- | ---: | ---: | ---: |
| DMRG2 | 123.62 ms | 114.41 ms | 7.4% |
| DMRG3 | 129.62 ms | 119.49 ms | 7.8% |
| Mixed | 137.84 ms | 128.05 ms | 7.1% |

Tensor arrays agreed to `1e-12`; canonical-center metadata, full FIT
diagnostics, and mixed history matched. Separate call profiles found mixed
rank-ceiling builds reduced from 26 to 1, maximum-bond scans from 38 to 14,
window preparation calls from 24 to 12, and full array-kind scans from 61 to
1. Ordinary DMRG2/3 active-window copy calls fell from 24 to 12. These results
describe this small CPU workload, not a general or GPU speedup guarantee.
Temporary measurement scripts and output remain outside the repository.

### Validation

- New replay metadata regressions: 10 passed, covering invalidation,
  reference equivalence, weak ownership, quality repair, lazy FIT selections,
  and injected partial FIT writes for gates, batches, sub-MPOs, and measurement.
- Final combined MPS/MPO, symmetric tensors, boundary preparation, sampler,
  FIT hot-path/schedule, copy/non-finite policy, replay metadata, fermionic
  boundary, public API, and package-layout suites: 1,254 passed, 7 skipped.
- Quimb compatibility and contraction-dependency suites: 12 passed.
- Default smoke suite: 128 passed.
- Ruff, skill catalog, both affected skill validators, and `git diff --check`:
  passed.

Counts overlap. No full repository extended suite or GPU runtime benchmark
was run. Native Symmray and dense NumPy/Torch/JAX paths were covered by the
affected numerical and ownership suites.
