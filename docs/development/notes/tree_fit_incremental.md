# Incremental TreeFIT environments and automatic tolerances — 2026-09-08

Affected paths: `src/pepsy/fitting/tree.py`,
`src/pepsy/optimizers/tree/optimizer.py`, tree API documentation, and the
tree-optimizer skill. No sibling repositories or installed packages changed.

## Upstream audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The required [Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
still returns an error; actual installed callable probes and native
regressions provide the compatibility evidence for this task.

Installed versions in the shared Python 3.12 environment:

| Package | Version |
| --- | --- |
| Quimb | `1.15.1.dev39+g369d09b9d` |
| Autoray | `0.11.1.dev1+gc56f64427` |
| Cotengra | `0.8.3.dev7+g1d7fd333f` |
| Cotengrust | `0.2.1` |
| Symmray | `0.3.2.dev7+gd63bb4e3f` |
| Torch | `2.6.0+cu124` |

Actual API/dispatch probes:

- `Tensor.copy(self, deep=False, virtual=False)` and
  `TensorNetwork.copy(self, virtual=False, deep=False)` distinguish tensor
  wrapper ownership from underlying array copies.
- `TensorNetwork(ts=(), *, virtual=False, check_collisions=True)` supports
  transferring already-private tensor wrappers through temporary layers.
- `tensor_contract(*tensors, output_inds=None, optimize=None, get=None,
  backend=None, preserve_tensor=False, drop_tags=False, strip_exponent=False,
  exponent=None, **contract_opts)` supports pure local message contractions
  and dropping unnecessary intermediate tags.
- `Tensor.split(T, left_inds, *, method='auto', absorb='auto', max_bond=None,
  cutoff=1e-10, cutoff_mode='rel', ..., **kwargs)` dispatches `svd`,
  `svd:eig`, `svd:rand`, and `qr` to `svd_truncated`,
  `svd_via_eig_truncated`, `svd_rand_truncated`, and `qr_stabilized`.
- `cotengra.array_contract(arrays, inputs, output=None, optimize='auto',
  strip_exponent=False, cache_expression=True, backend=None, **kwargs)`
  remains the contraction implementation.
- Autoray `get_lib_fn(backend, 'linalg.svd'/'linalg.qr')` resolves NumPy,
  Torch, and native Symmray functions. Symmray SVD/QR retain their block-array
  arguments. No dispatch registrations or split drivers changed.

Decisions:

- **Adopt:** public Quimb local contraction with `preserve_tensor=True` and
  `drop_tags=True`; owned layer assembly with `virtual=True`.
- **Adopt:** MpsOptimizer's existing tolerance values for TreeOptimizer's
  new automatic FIT stopping default. The chain optimizer is unchanged.
- **Defer:** Quimb chain SRC/SDC algorithms remain distinct from tree-local
  randomized/deterministic splits. This change does not relabel algorithms.
- **Defer:** odd-parity fermionic FIT remains explicitly unsupported. Even
  parity U1U1 fitting and dense NumPy/Torch messages are tested separately.
- **Compatibility shim:** none. The existing native QR policy remains intact.

## Algorithm and ownership

A message `u -> v` combines only `u`'s target layer(s), its fitted bra tensor,
and incoming messages `w -> u` with `w != v`. An iterative postorder stack
constructs each missing dependency once. Invalidating an updated node
propagates through cached dependent messages, preserving opposite untouched
branches. The invariant that every cached message retains its dependencies
makes it safe to stop invalidation at a missing entry. No per-edge full
component sets are stored, and messages do not accumulate branch-wide tags.

Pure contractions borrow target/message wrappers. The effective tensor is
still copied at its cache boundary because block factorization changes tags.
Newly owned state/operator wrappers are transferred through target layers;
caller-owned state/operator wrappers remain independent. A disposable FIT
guess uses a fresh optimizer with one state copy and the numerical routing
policy, without copying accumulated histories, diagnostics, or queued gates.
It preserves the previous single child-seed draw from the parent RNG, so
later measurements keep their seeded sequence; randomized compression still
uses `fit_init_seed`. Public `copy()`
retains independent history and a derived child RNG, and now also preserves
`fit_adaptive_sweeps`, which was previously omitted.

TreeOptimizer's automatic cutoff policy is unchanged. Its new defaults are
`fit_rtol='auto'`, `fit_min_iter=2`, and
`fit_sweep_sequence='inward-outward'`. FIT tolerance resolves to `1e-3` for
16-bit, `1e-5` for float32/complex64, and `1e-9` for higher precision.
`fit_rtol=None` preserves fixed-iteration behavior. Automatic tolerance
stopping is disabled for declared non-unitary replay or `track_norm=False`;
explicit numeric tolerances remain honored. Non-unitary replay also avoids
claiming that the pre-update norm is the target's exact norm.

One iteration includes both inward and outward passes relative to the active
region's medial node. `RL`/`INOUT` and `LR`/`OUTIN` remain aliases with
unchanged traversal order. Standalone TreeFIT retains `rtol=None`. The
warm-up/refinement schedule and total iteration budget are unchanged.

## Validation

New regressions in `tests/test_tree_fit_messages.py` cover every directed
message against an independent complete-branch contraction, local input
counts, cache retention/invalidation, fused/layered NumPy and Torch targets,
a physical root on a branching tree, complete FIT sweep equivalence, even
native U1U1 fitting, automatic/explicit tolerances, non-unitary replay,
sweep-name aliases, copy policy, parent-history isolation, and RNG ownership.

The final focused run includes the new tests, tree zipup, TreeOptimizer, shared
TreePeps FIT, Quimb/Cotengra regressions, and public API/package layout tests:
**558 passed**, including the final preservation of the parent's child-seed
draw sequence. The non-unitary replay checks also passed in a fresh
**26 passed** run of the new module. Ruff (`src tests`), the individual
skill validator, catalog validator, and `git diff --check` passed.

Full-suite results and controlled timing measurements are recorded below.

### Controlled FIT timing

Temporary harness: `/tmp/tree_incremental_benchmark.py`; no performance
harness was added to the package. Compared the final engine with a reference
subclass restoring whole-branch messages and component-set invalidation.
Both use the same effective-block contractions and SVD policy. Balanced
trees, NumPy complex128, target bond dimension 3, guess/output dimension 2,
zero cutoff, seeds 7/8, and two full iterations; BLAS threads capped at one.
Times include FIT construction and are medians of three runs.

| Physical sites | Whole-branch messages | Incremental messages | Ratio |
| --- | --- | --- | --- |
| 15 | 0.1740 s | 0.1186 s | 1.47x |
| 31 | 0.9234 s | 0.2738 s | 3.37x |

Normalized state fidelities are within `2e-15` of one, and relative retained
norm differences are below `2e-15`. These are small controlled FIT cases,
not a guarantee of the same improvement for whole circuit replay or GPUs.
The contraction-count regression independently verifies one local
contraction per missing directed edge, with inputs bounded by the node's
degree and number of target layers.

### Full-suite readout

With `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`, the full review run completed in 785.90 seconds:
**3444 passed, 1 skipped, 5 failed**. It was launched before the final
child-seed draw compatibility adjustment; the final 558-test run above
covers that change as well. Failures match the preceding review exactly:

- `test_qmera_compiled_parametric_loss_jax_jit_smoke`
- `test_native_complex64_gpu_qr_retries_same_device_double_precision[torch]`
- `test_optional_array_backends_match_numpy_for_stn_paths[jax-_jax_backend-jax]`
- `test_fermion_fields_and_pairing_preserve_jax_backend`
- `test_compiled_boundary_reuse_batches_connected_local_energy`

Rerunning all five together in a fresh process on the final code, with the
same thread caps: **5 passed** in 13.81 seconds. Three failures were JAX
CPU/GPU context mismatches, one a native GPU QR fallback, and one a Torch
compiled-environment count assertion. This repeats the evidence for shared
backend state or test-order interference; the full suite is not a clean
pass. No unrelated backend-isolation changes were made.

Logs: `/tmp/tree-incremental-focused.log`,
`/tmp/tree-incremental-full.log`,
`/tmp/tree-incremental-failure-rerun.log`, and
`/tmp/tree-incremental-benchmark.log`.
