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
