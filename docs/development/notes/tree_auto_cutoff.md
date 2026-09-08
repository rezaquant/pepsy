# Tree automatic cutoff parity with MPS

Audit date: 2026-09-08.

`MpsOptimizer.run()` resolves `cutoff="auto"` through the shared
`dtype_auto_cutoff` helper. TreeOptimizer already uses that helper after
installing its state at construction. The shared values are `1e-3` for
16-bit data, `1e-6` for float32/complex64, and `1e-12` for float64/complex128.
Both optimizers preserve nonnegative explicit numbers, including zero, and
reject negative or nonfinite cutoffs.

The automatic cutoff-mode strings differ at the underlying DM boundary:

| Path | Spectrum being truncated | Automatic mode |
| --- | --- | --- |
| MPS ordinary SVD | singular values `s` | `rsum2` |
| MPS MPO DM | density-matrix eigenvalues `s**2` | native `rsum1` |
| Tree direct or DM (`svd:eig`) | singular values `s` | `rsum2` |

MPS preserves its MPO compressor's native default by omitting `cutoff_mode`
when the user selects `auto` or `None`. Tree cannot copy that omission:
Quimb's generic `tensor_split` defaults to `rel`. Nor should tree DM copy
the literal `rsum1` string: `svd:eig` takes square roots of the Gram spectrum
before applying its cutoff. Tree's existing `rsum2` gives the same relative
discarded probability as MPS DM. Explicit user-supplied modes retain their
literal Quimb meaning. No numerical behavior change is required.

## Upstream compatibility audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The required [Abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an access error; installed-source and dispatch probes supplemented
the repository review.

The documented `~/envs/py312` environment was absent. Validation used the
available Python 3.12 environment at
`/home/zeus/miniconda3/envs/cloudspace` with temporary Numba/Matplotlib caches.
Installed versions: Quimb `1.15.1.dev46+g0ad529894`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust
`0.2.1`, Symmray `0.3.2.dev7+gd63bb4e3f`, NumPy `2.4.4`, Torch `2.11.0`.

Actual callable/source probes confirmed:

- `tensor_network_1d_compress_dm(..., cutoff_mode="rsum1", ...)` sets
  `method="eigh"` on its positive density matrix. Its installed docstring
  explicitly identifies equivalence to direct `rsum2`.
- `tensor_split(..., method="auto", cutoff=1e-10, cutoff_mode="rel", ...)`
  accepts a literal cutoff mode; `tensor_compress_bond(..., **kwargs)`
  forwards split options.
- `svd_via_eig_truncated(x, cutoff=-1.0, cutoff_mode=4, max_bond=-1,
  absorb=0, renorm=0, info=None)` uses singular-value truncation. Code `4`
  means `rsum2`. Both the generic and NumPy implementations obtain `s`
  before trimming. Quimb's `_SPLIT_FNS["svd:eig"]` selects that driver.
- Autoray `linalg.eigh` resolves to NumPy's `eigh`, Torch's `linalg_eigh`,
  and Symmray's `eigh` respectively. Native fermionic tree DM remains
  unsupported; its graded direct compression is unchanged.

Classification: **adopt** the verified spectrum-aware explanation and
numerical regression coverage. **Defer** unrelated upstream compression,
random-array, ordering, and contraction changes; no compatibility shim or
new algorithm is needed for this task.

## Regression coverage

`tests/test_optimize_tree.py` compares actual MPS/tree DM outputs for a
two-qubit state with Schmidt coefficients `sqrt(0.99)` and `0.1`. At cutoff
`0.05`, both automatic policies retain rank one and squared norm `0.99`;
explicit tree `rsum1` retains rank two. The test exercises DM shorthand,
TreeMPO DM, the compression-mode alias, omitted/None/auto cutoff modes, and
copy preservation.

A second comparison uses a `1e-7`-weight branch: automatic single-precision
compression discards it, while double precision retains it. It supplies a
TTN explicitly so the installed state's dtype, rather than the constructor's
default dtype, controls the cutoff. Explicit zero preserves the branch in
both precisions.

Validation: the focused tree cutoff selection passed all eight tests, and
the MPS native-DM cutoff selection passed both tests. Repository-wide
`python -m ruff check src tests` passed with Ruff installed under `/tmp`.
The full numerical suite was not rerun because the implementation changes
are explanatory docstrings; the existing truncation policy is unchanged.
