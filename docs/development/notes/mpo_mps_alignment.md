# MPO alignment with MPS replay improvements

Audit date: 2026-09-08. Scope: MPO replay controls, FIT scheduling and rank
limits, canonical metadata, and transactional ownership.

## Shared policies

- `direct` is the constructor default; `mpo` and `quimb` are silent aliases.
- Replay uses eight FIT sweeps, dtype-aware cutoff/tolerance defaults,
  minimum two sweeps, and patience two. Explicit `fit_rtol=None` retains a
  fixed budget. DMRG3 includes a two-site transition before one-site refinement.
- Public `finite_check` scopes optional validation to one replay and warns
  once. The legacy `fit_finite_check` spelling remains accepted, with
  conflicting values rejected. Empty streams also run requested validation.
- Physical rank targets count both operator legs. Shared FIT previously used
  `MPO.phys_dim`, which reports only one leg and underestimated rank capacity.
  Replay caches immutable site geometry while reading exterior bond ranks live.
- Temporary canonicalization no longer reads or overwrites the live center.
  Nonunitary one-site multiplication moves the center before changing an
  isometry, retaining the absolute operator scale.
- Dense transactional snapshots own mutable active-window arrays; full-run
  snapshots own all arrays. Compressed guesses also own active arrays, so
  independent-guess failures can retain the original live state. Torch uses
  differentiable clones. Native/unknown arrays retain deep-copy isolation.
  Append-only diagnostic histories snapshot list containers instead of
  repeatedly deep-copying every old record. Public accessors retain defensive
  copies. Replay-local caches are discarded on success and failure.

## Deliberate operator differences

MPO evolution keeps independent upper/lower physical legs. Raw dense gate
payloads retain `G.T @ O @ B.conj()` orientation; ordinary channel Kraus
matrices use `K @ O @ K.conj().T`. Channels sum all branches deterministically.
There are no trajectory sampling, measurement-collapse, or state-renormalization
semantics here. Absolute Frobenius norms and the network exponent remain
separate from compression-survival ratios; channel trace diagnostics remain
separate from both. Native direct/Quimb aliases use block-aware SVD, while
native DMRG still uses native FIT.

## Upstream audit

Installed development versions inspected in the `genpy` environment:

| Package | Version |
| --- | --- |
| Quimb | `1.15.1.dev39+g369d09b9d` |
| Autoray | `0.11.1.dev1+gc56f64427` |
| Cotengra | `0.8.3.dev6+g08fe1a3a1` |
| Symmray | `0.3.2.dev6+ga17699db6` |

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The requested
Symmray Abelian-array documentation URL did not load; installed APIs and
repository sources were used instead.

Inspected installed `MPO.gate_sandwich_with_auto_swap`, canonicalization,
`TensorNetwork.copy`, `Tensor.modify`, and `FIT.run_gate` signatures, plus
NumPy/Torch/JAX copy, finite-check, SVD, and QR dispatch. Confirmed that
`MPO.phys_dim` reports one physical leg.

**Adopt:** existing Quimb gating/copy/metadata APIs and Autoray dispatch;
correct Pepsy's operator rank accounting. **Compatibility shim:** none needed.
**Prototype:** none introduced. **Defer:** replacing guarded native norm
contractions with center-only shortcuts, or choosing new upstream compression
algorithms automatically. Those require independent graded-norm and backend
evidence. No installed package was modified.

## Validation

Focused MPO/FIT checks cover complex non-symmetric dense gate orientation,
direct aliases, deterministic channel sums and absolute scale, rank ceilings,
temporary/live canonical centers, off-center nonunitary updates, empty-run
validation, in-place mutation recovery, Torch clone gradients, and actual
MPS/MPO DMRG2/3 sweep schedules. Existing MPO tests cover native symmetry,
backends, batching, fallback, trace diagnostics, and disabled timing.

The focused MPO/FIT suites passed (175 tests across the focused runs).
The broader non-slow MPS, FIT performance/metadata/finite-policy, fermionic
boundary, boundary preparation, and MPO Trotter checks passed (736 tests;
39 slow tests deselected). All 128 default smoke tests, repository Ruff, and
`git diff --check` passed. No accelerator performance claim is made from these
correctness checks.
