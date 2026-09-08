# Tree zipup and lazy FIT audit — 2026-09-08

Affected paths: `optimizers/tree/optimizer.py`, `fitting/tree.py`, and their
shared TreePeps FIT tests. No installed dependency or sibling repository edits.

## Upstream audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The required [Abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
failed to load twice; the installed Symmray callable probes and native
regressions provide the numerical compatibility evidence instead.

Installed in the shared Python 3.12 environment:

| Package | Version |
| --- | --- |
| Quimb | 1.15.1.dev39+g369d09b9d |
| Autoray | 0.11.1.dev1+gc56f64427 |
| Cotengra | 0.8.3.dev7+g1d7fd333f |
| Cotengrust | 0.2.1 |
| Symmray | 0.3.2.dev7+gd63bb4e3f |
| Torch | 2.6.0+cu124 |

Actual callable/dispatch probes:

- `Tensor.split(T, left_inds, *, method, absorb, max_bond, cutoff,
  cutoff_mode, renorm, get, bond_ind, right_inds, ..., **kwargs)` accepts
  `svd`, `svd:eig`, `svd:rand`, and `qr` through Quimb's split registry.
  These dispatch to `svd_truncated`, `svd_via_eig_truncated`,
  `svd_rand_truncated`, and `qr_stabilized`, respectively.
- `tensor_contract(*tensors, output_inds=None, optimize=None, get=None,
  backend=None, preserve_tensor=False, ..., strip_exponent=False, ...)`
  supports the per-node layered contraction and directional FIT messages.
- `TensorNetwork.norm(..., squared=False, strip_exponent=False, ...)`
  exists but is deliberately not called for a lazy target.
- `cotengra.array_contract(arrays, inputs, output=None, optimize='auto',
  strip_exponent=False, cache_expression=True, backend=None, **kwargs)`
  remains the underlying contraction interface.
- Autoray `get_lib_fn(backend, 'linalg.svd'/'linalg.qr')` resolves NumPy's
  SVD/QR, Torch's SVD wrapper/`linalg_qr`, and Symmray's native SVD/QR.
  Symmray `linalg.svd(x, *args, **kwargs)` and `linalg.qr(x, *args, **kwargs)`
  retain block-array inputs. Native lossless QR stays behind Pepsy's shared
  structural-zero-safe policy.

Decisions:

- **Adopt:** public Quimb per-node contraction and native tensor SVD for the
  opt-in tree zipup route. No version-string dispatch and no vendored code.
- **Adopt:** existing layered TreeMPO targets and directed TreeFIT overlap
  messages; transfer the optimizer-owned target with `copy_target=False`.
- **Defer:** Quimb's chain SDC/SRC environment algorithms and newer gate
  transforms are not replacements for an arbitrary branching tree algorithm.
  Existing tree `sdc` currently uses the same deterministic SVD sweep as
  `direct`; tree `src` uses randomized local edge SVD.
- **Compatibility shim:** none added. Preserve native QR policy and backend
  preparation requirements.

## Numerical contract

Direct application QR-routes the full operator and then canonically truncates
the tree. Zipup keeps node layers separate until incoming messages arrive,
SVD-truncates the outgoing message immediately, and installs the retained
isometries around the final hub. Its cuts have unvisited, noncanonical
operator environments; a local discarded weight is not a global error bound.
The route supports branching as well as path-shaped active regions.
Native capped tests exposed an empty-charge-path case at chi=2: independent
early branch cuts can leave no combination allowed at the hub. The upstream
audit was repeated; installed Symmray `svd_truncated(x, *args, **kwargs)`
retains its documented block-SVD cap/cutoff policy. This is a zipup
approximation limitation, not an upstream dispatch fix. Reject empty native
blocks before installing the candidate; preserve the prior physical state
and recommend a larger cap or an explicit direct route. Native capped chi=4
and uncapped comparisons are tested separately.

Finite-check dispatch was also probed: NumPy `isfinite`, `all(a, axis=None,
out=None, keepdims=..., *, where=...)`, and `stack(arrays, axis=0, out=None,
*, dtype=None, casting='same_kind')` resolve through Autoray. Torch resolves
its corresponding builtins (their Python signatures are not introspectable).
Checks reduce on the backend and transfer one compact boolean vector per
sweep, not the state arrays.

TreeFIT uses the terminal canonical-center norm for convergence, retains its
lazy exact target independently from its disposable guess, and does not
revalidate every untouched isometry per sweep. Optional finite checks scan
only active tensor data, natively for dense arrays and Symmray blocks.
`target_norm` may be a represented norm or `(mantissa, base10_exponent)` pair.
An unknown layered target yields `local_fidelity=None`, not a guessed
normalization. Explicit overlap diagnostics may perform a separate lossless
QR norm pass, but never form a doubled target. The actual target–fit overlap
is contracted only when requested and reused in the normalized diagnostic.

## Validation

Focused coverage: `tests/test_tree_zipup.py`, `tests/test_optimize_tree.py`,
`tests/test_tree_peps_optimizer.py`, `tests/test_quimb_compat.py`, and
`tests/test_contraction_dependencies.py`. Zipup checks include dense
NumPy/Torch complex64/complex128, native U1U1 fermions, exact-rank agreement,
finite bond caps, canonical metadata, run/copy policy, and a guard against
doubled-target norms. The shared tree tests check unchanged FIT schedules
and existing dense/native contraction routes. Full-suite and Ruff outcomes
are recorded below.

- Final focused tree/TreeFIT/TreePeps and contraction run: **481 passed**.
- Full suite: **3406 passed, 1 skipped, 5 failed** in 815.85 seconds. The
  five failures were JAX CPU/GPU context mismatches (qMERA, stabilizer, and
  fermion pairing tests), a native GPU QR fallback test, and a Torch VMC
  compiled-environment count assertion. Repeating those exact five tests
  together in a fresh process: **5 passed**. This supports test-order/shared
  backend-state interference; the full run is not recorded as clean.
- Ruff (`src tests`), skill validation, catalog validation, and
  `git diff --check`: passed. No dependency changes or global backend patches.

## Completion review — 2026-09-08

Rechecked all six upstream URLs above and repeated installed-version,
callable-signature, and SVD/QR dispatch probes in the shared Python 3.12
environment. Versions and dispatch match the original audit; the Abelian-array
documentation URL still returns an error. The adopt/defer decisions remain
unchanged, and no compatibility shim is needed.

The review found that the new explicit target-norm diagnostic ran outside
the existing overlap-error boundary. Moved it inside that boundary so a
failed optional QR diagnostic cannot abort a successful fit. Added a
regression for the failure report and unchanged fitted state, plus dense
non-unitary target checks with a stored base-ten exponent of 30, both with
and without a supplied norm. The supplied-norm case also forbids the extra
QR pass; both cases forbid doubled-target norm calls.

Those scale checks exposed a second issue: local FIT updates copied the
target's raw tensors but retained the guess's exponent. Each installed block
now carries the target exponent, since its exterior is isometric and its
effective tensor contains the target's raw scale. The regression compares
the fitted represented state against the exact vector, not only normalized
fidelity. This is an internal scale-preservation fix, with no upstream shim.

Native FIT probes also exposed stale fitted-bond names in directional
messages after native center movement. Messages now resolve the live fitted
bond while retaining fixed private target bonds. A deterministic bond-rename
regression covers that invariant.

Removing that exception exposed an unsupported graded projection: a lossless
two-site FIT replay of a hopping gate on four singly occupied U1U1 sites
returned fidelity 0.9226188356698385 against exact direct replay. The layered
target itself had unit overlap with the exact state. **Defer** extending FIT
to odd-parity tensors; reject those inputs explicitly before installation
instead of returning an incorrect state. A regression verifies the error and
preserved input state. Native direct/zipup support is unchanged.
The Quimb changelog likewise flags odd-parity fermionic FIT as unreliable.
Installed `TensorNetwork.conj(mangle_inner=False, output_inds=None,
phase_dual=True, inplace=False)` and the native `parity` property were
inspected; network-level branch conjugation alone did not fix the projection.
No speculative phase correction or global dispatch change is retained.

Final focused validation after all review fixes: **486 passed**, covering
every test module that uses TreeFIT as well as native zipup and upstream
contraction regressions. Ruff, both skill validators, and `git diff --check`
passed. The tree skill is below the 500-line policy limit.

The full-suite review run completed with **3416 passed, 1 skipped, 5 failed**
in 956.29 seconds. It was started before the final live-bond and odd-parity
guard fixes; the 486-test focused run above covers the final implementation.
The five failures match the prior handoff exactly:

- `test_qmera_compiled_parametric_loss_jax_jit_smoke`
- `test_native_complex64_gpu_qr_retries_same_device_double_precision[torch]`
- `test_optional_array_backends_match_numpy_for_stn_paths[jax-_jax_backend-jax]`
- `test_fermion_fields_and_pairing_preserve_jax_backend`
- `test_compiled_boundary_reuse_batches_connected_local_energy`

Rerunning those exact cases together in a fresh process on the final code:
**5 passed** in 14.75 seconds. This repeats the evidence for shared backend
state or test-order interference; the full suite is not a clean pass. No
unrelated backend/test-isolation changes were made. Review logs are under
`/tmp/tree-review-final-tests.log`, `/tmp/tree-review-full-suite.log`, and
`/tmp/tree-review-failure-rerun.log`.
