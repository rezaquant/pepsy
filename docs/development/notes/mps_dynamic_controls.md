# MPS dynamic controls and noisy trajectories — 2026-09-08

## Reproduced problems

- Ordinary conditional gate dispatch forced one sweep and omitted the caller's
  FIT options. Requesting eight sweeps and `guess-sdc` could raise in DMRG1 or
  silently take the MPO fallback in DMRG2/3. Independent and coalesced shot
  execution used different paths and therefore different solver policies.
- A leaked site retained its old logical label after a cap. Leaking site 2,
  capping site 0, then measuring leakage at the new site 1 returned 0 instead
  of the leakage marker 2 in both trajectory strategies.
- A cap could change the physical norm without rebasing the unitary ledger.
  Projecting one Bell-pair leg and applying a subsequent lossless CNOT reported
  compression infidelity 0.5 despite an exact state update.
- Conditional caps could leave valid leaves of lengths 3 and 2, but terminal
  sampling failed while concatenating their rectangular bit arrays.

## Implementation

Conditional MPS gates now forward the same validated `mode_kwargs` as ordinary
segments. This preserves the caller's sweep count, named DMRG block schedule,
guess, cutoff, direction, normalization, quality, and finite/timing policies.
The FIT kernel and its defaults have not changed.
Physical action sites are resolved only after a true predicate, so a false
conditional in `perm` mode can safely mention a site removed by an earlier cap.

Both trajectory strategies treat caps as immediate structural boundaries:
commit earlier work, execute the cap, then remove its leakage flag and shift
higher logical labels down by one. Capping a leaked site's placeholder is
allowed. The remap uses logical labels, so `perm` remains consistent when its
physical order differs. Persistent layouts still reject caps. The helper
changes classical labels only after successful cap execution.

Conditional trajectory actions are resolved against committed measurement
records. Selected controls use normal control dispatch, including nested caps,
reset/leakage cleanup, and measurement branching. Concrete executed actions
are retained once in the replay stream. Conditional measurement branches obey
the configured branch-count limit. Additional review found a missing RNG
argument when coalesced measurement splits normal and leaked leaves; that
recursive call now forwards the existing RNG.

Successful caps invalidate only the raw unitary norm baseline. The next
unitary segment reads its baseline from the shorter state through the existing
canonical-center path. Accumulated compression survival and the cap's raw
contraction scale are preserved. `cap_history` records the structural event;
there is no new cap norm contraction, tensor scan, or artificial truncation
event. The existing rank/backend cache invalidation remains active.

`CoalescedSampleResult` now exposes per-row `lengths`. Uniform batches retain
their previous values and shape. Different-length batches allocate one output
array of maximum width with `-1` right padding; valid bits occupy each row's
prefix. Lengths, leaf indices, and probabilities use the same shuffle. Columns
refer to each leaf's current logical numbering, not original pre-cap labels.
The existing three-argument result constructor supplies uniform lengths, and
empty ensembles return an empty lengths array.

## Upstream compatibility audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and the
[Symmray repository](https://github.com/jcmgray/symmray). The
[Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
endpoint could not be retrieved; native behavior is covered using the
installed development package.

Installed versions in the activated genpy environment:

- Quimb `1.15.1.dev39+g369d09b9d`
- Autoray `0.11.1.dev1+gc56f64427`
- Cotengra `0.8.3.dev6+g08fe1a3a1`
- Symmray `0.3.2.dev6+ga17699db6`

Inspected installed public `TensorNetwork.copy`, `reindex`, `retag`, `view_as`,
`norm`, `Tensor.modify`, `tensor_contract`, `MatrixProductState.canonicalize`,
`gate_nonlocal`, `MpsSampler.sample_batch`, and `autoray.to_numpy` signatures.
Probed NumPy/Torch/JAX copy, absolute-value, square-root, and host-transfer
dispatch. **Adopt/retain:** existing public canonicalization, contraction,
copy, and batched sampling APIs. **Defer:** new upstream compressor variants,
gating flags, operator-layout changes, and native algorithm changes. No new
compatibility shim or installed-package edit was needed; the fixes are in
Pepsy control dispatch, bookkeeping, and terminal result assembly.

## Validation

The new `tests/test_mps_dynamic_controls.py` contains 19 passing regressions.
They cover explicit solver settings against unconditional references,
independent/coalesced shots, NumPy complex128 and Torch/JAX complex64,
permuted leakage labels, nested conditional caps, removal of leaked sites,
cap norm accounting with prior compression loss, mixed-length output and
shuffle alignment, empty/uniform result compatibility, conditional measurement
branch budgets, and mixed normal/leaked measurements.

The original temporary reproductions now show eight FIT sweeps with SDC for
DMRG1/2/3, correct leakage marker 2 after renumbering, and successful terminal
samples from lengths 3 and 2. Existing trajectory tests plus the first 16 new
regressions passed together (115 tests). Quimb compatibility and contraction
dependency tests passed (12 tests). These counts overlap later combined runs.

Final validation:

- Combined MPS/MPO, trajectory, dynamic-control, replay metadata, finite/copy
  policy, FIT schedule/hot-path, sampler, native symmetric tensor, MPI unit,
  public API, and package-layout suites: 1,342 passed, 7 skipped.
- After the final false-predicate site-resolution fix, the complete new file
  plus affected existing control/cap/layout tests: 83 passed.
- Default smoke suite: 128 passed.
- Ruff, MPS skill validation, skill catalog, and `git diff --check`: passed.

Counts overlap. The full repository extended suite was not run: validation
covered the changed MPS control layer and shared trajectory consumers, while
unrelated solver and VMC kernels were unchanged. No GPU runtime benchmark was
performed. No changes were made to per-sweep finite-check defaults or scalar
channel-probability validation.
