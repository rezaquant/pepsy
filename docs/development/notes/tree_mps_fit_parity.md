# Tree equivalents of the MPS FIT maintenance — 2026-09-08

Reviewed upstream Pepsy commits `ddcda99`, `4a4c287`, and `dc2e103` after
rebasing the earlier tree work onto `origin/develop`. Changes are confined to
TreeOptimizer, its shared TreeFIT kernel, tests, and documentation.

## Review decisions

| MPS change | Tree decision |
| --- | --- |
| Eight alternating sweeps by default | **Adopt with tree semantics:** four complete inward/outward iterations, instead of two. Keep two complete warm-up iterations and explicit budgets. |
| DMRG3 three/two/one schedule | **Adopt:** `(3, 3, 2, 1)` by default; `fit_two_site_transition_sweeps=0` restores the previous direct three-to-one handoff. |
| Reuse environments across 3-to-2 transitions | **Already present:** directed neighbor messages depend on updated tensors, not the block size. No transition-wide reset or chain boundary cache is needed. |
| Skip unused SRC rank scans | **Already present:** tree compressed guesses do not inspect attainable ranks. DMRG1's rank-dependent schedule retains its necessary checks. |
| Cache active-window copy capabilities | **Not needed:** tree FIT uses independent tensor wrappers and replacement-based updates, with no whole-state deep copy or repeated backend-capability classifier in the guess path. Public history copies stay defensive. |
| Scalar-only convergence transfer | **Already present:** TreeFIT reads a scalar canonical-center contraction; vector stacking occurs only in enabled finite diagnostics. |
| Opt-in finite scans across all replay modes | **Adopt:** `run(finite_check=True)` enables FIT scans and a final-state scan, including empty replay; restore flags on all exits and propagate to shot workers. |
| One warning per enabled replay | **Adopt:** optimizer-owned FIT instances suppress only their duplicate warning. Standalone FIT warns independently. |
| Remove scalar non-finite and mixed-commit guards | **Defer:** TreeOptimizer has no MPS mixed transaction path; retain its existing scale/logarithm guards. They do not scan tensors or add device transfers. |
| Adjacent chain-pair shortcut | **Defer:** two tree leaves generally span at least three structural nodes. Do not infer a chain two-site optimum from two physical support labels. |
| New MPS-to-TTN converter | **Already available from the pull:** explicit `pepsy.tensors.mps_to_ttn` conversion remains caller-controlled. No implicit entangled-state relayout. |

The new iteration budget matches the number of directional passes, not the
exact chain sequence of local updates. Tree block sizes count connected
structural nodes, including internal tensors. More iterations can change
runtime and the approximated state at finite chi. Explicit iteration settings
are preserved. Standalone TreeFIT and TreePeps retain their existing defaults;
all TreeFIT run entry points offer an explicit `two_site_transition_sweeps`.

Every change of block size resets tolerance history. Stagnation during
warm-up or a two-node transition cannot skip pending refinement. The tree
patience value remains one stable comparison, equivalent to two MPS samples.

## Native projection correction found during validation

Extending the existing even-parity U1U1 hopping regression to one-node
refinement exposed fidelity `0.9226188356698383` for an already exact initial
guess. Individual two-/three-node projections also showed errors that could
cancel over a complete sweep, hiding the problem from the previous terminal
fidelity test.

The effective tensor is a covector on open overlap-environment legs. Native
fermionic writeback needs the graded metric: flip odd-sector phases on each
dual boundary leg before factorization. Use native `phase_flip`, without
densification or changing physical outputs, contracted bonds, or target
arrays. Cache the corrected effective tensor once. This works for one-, two-,
and three-node blocks and is independent of sweep direction.

The regression now checks fidelity after every individual projection through
the complete three/two/one schedule, as well as final native target overlap.
Odd-parity FIT remains explicitly unsupported; the existing QR safeguard and
split drivers are unchanged.

## Upstream compatibility audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The required [Abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an error again. The installed source/signatures and numerical probes
were inspected, including Symmray's conjugation convention when diagnosing
the native projection failure.

Installed shared Python 3.12 environment:

| Package | Version |
| --- | --- |
| Quimb | `1.15.1.dev39+g369d09b9d` |
| Autoray | `0.11.1.dev1+gc56f64427` |
| Cotengra | `0.8.3.dev7+g1d7fd333f` |
| Cotengrust | `0.2.1` |
| Symmray | `0.3.2.dev7+gd63bb4e3f` |
| Torch | `2.6.0+cu124` |

Actual callable and dispatch probes:

- `Tensor.copy(deep=False, virtual=False)` and
  `TensorNetwork.copy(virtual=False, deep=False)` retain separate metadata.
- `tensor_contract` supports `output_inds`, `optimize`, `preserve_tensor`,
  and `drop_tags`; `Tensor.split` supports native method/cutoff dispatch.
- `cotengra.array_contract(arrays, inputs, output=None, optimize='auto',
  strip_exponent=False, cache_expression=True, backend=None, **kwargs)`.
- Quimb's registry resolves `svd`, `svd:eig`, `svd:rand`, and `qr` to the
  expected truncated-SVD and stabilized-QR drivers.
- Autoray resolves NumPy/Torch `all`, `isfinite`, `stack`, QR, and SVD;
  Symmray QR/SVD remain native. Symmray has no generic `stack` registration:
  the finite helper reduces its dense blocks first, then stacks backend
  boolean scalars, exactly as the existing TreeFIT helper did.
- `FermionicArray.phase_flip(self, *axs, inplace=False)` returns a native
  copy. `conj(phase_permutation=True, phase_dual=False, inplace=False)` does
  not automatically supply the open dual-leg metric. No global dispatch,
  installed package, or dependency changes were made.

**Adopt:** native `phase_flip` for the tree effective-tensor metric and the
existing backend reductions for optional finite scans. **Defer:** upstream
chain-only compressor additions and odd-parity FIT. **Compatibility shim:**
none. **Prototype:** none.

## Validation

Focused checks cover default/explicit budgets, convergence phase resets,
copy configuration, optional scans and warning scope, empty replay, shot
forwarding, dense NumPy/Torch state preservation, native per-update fidelity,
and existing TreePeps, Quimb contraction, and public API regressions.

- **576 passed**: tree parity, incremental messages, zipup, TreeOptimizer,
  TreePepsOptimizer, Quimb compatibility, contraction dependencies, public API,
  and package layout (`/tmp/tree-mps-parity-focused.log`).
- **181 passed** in the final follow-up: incremental messages (including a
  new complete-branch reference comparison across 3-to-2-to-1), tree parity,
  MPS FIT schedules/hotpaths/non-finite policy, and sampler regressions
  (`/tmp/tree-mps-parity-final.log`). These runs overlap.
- Repository Ruff, tree skill validation, catalog validation, and
  `git diff --check` passed.
- The full repository suite was not rerun: this change is scoped to the tree
  kernel and replay, covered by the complete closest domain suites above.
  No performance speedup is claimed from changing the sweep budget.
