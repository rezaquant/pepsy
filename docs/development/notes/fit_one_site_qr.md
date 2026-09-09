# One-site FIT QR absorption audit

Audit date: 2026-09-08.

## Change and safety boundary

`FIT._run_gate_one_site_sweep` computes an effective tensor from the fixed
target and cached left/right environments. The next site's old data does
not participate in its own effective update. Its indices supply output
ordering, and the moving environment reads only the site just optimized.
Consequently, the QR factor normally absorbed into that next tensor is
discarded by the next writeback. The new private helper retains Q with the
same index order, tags, backend, dtype, and `left_inds`, skipping absorption.
Both directions use Quimb's existing QR convention, including right-moving
and left-moving tensor index ordering; no custom QR implementation is added.

The shortcut is limited to NumPy, Torch, and JAX arrays sharing exactly one
bond, with QR row dimension at least the original bond dimension. Reduced
QR otherwise changes the bond size, so those cases use the full Quimb move.
Native Symmray arrays and other backends also retain the full move. Exact
numerical rank deficiency does not change dense reduced-QR dimensions.

This helper is specific to an immediately overwriting variational update:
it is not a lossless operation on the intermediate full state. No full-state
observable is evaluated between the shortcut and the next effective update.
The final sweep site retains the unmodified effective tensor as the center;
final norm and convergence semantics remain unchanged. Failed partial sweeps
are not successful fitted states. No new public switch or mode is introduced.

## Upstream audit

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned a retrieval error; installed native regression tests cover the
retained fallback.

Installed versions:

- Quimb `1.15.1.dev39+g369d09b9d`.
- Autoray `0.11.1.dev1+gc56f64427`.
- Cotengra `0.8.3.dev6+g08fe1a3a1`.
- Symmray `0.3.2.dev6+ga17699db6`.

Inspected installed `Tensor.split(T, left_inds, *, method='auto',
absorb='auto', ..., get=None, bond_ind=None, right_inds=None, ...)`,
`left_canonize_site(i, bra=None, create_bond=False)`,
`right_canonize_site(i, bra=None, create_bond=False)`, and their
`tensor_canonize_bond` implementation. Quimb QR uses
`qr_stabilized(x, absorb=1, stabilized=True, **kwargs)`; Autoray resolves
`linalg.qr` to NumPy QR, Torch `linalg_qr`, and JAX QR in this environment.

Classification: **adopt** the existing public tensor QR split for the
overwrite-only shortcut. **Defer** native shortcuts and unrelated upstream
compression algorithms; no dependency upgrade, dispatch registration,
compatibility shim, or installed-package modification is needed.

## Validation

Focused regressions compare the optimized path with the original complete
gauge move for MPS and MPO fits, complex64/complex128, NumPy/Torch/JAX,
both directions, repeated directions, larger-block transitions, and final
one-site polishing. They check final dense tensors, center/norm, actual
isometries, and tensor metadata. Separate checks exercise shape-reducing QR,
structural-zero columns, skipped neighbor writes, and Torch target gradients.
Existing native cache-transition tests exercise both sweep directions.

Completed checks (counts overlap):

- FIT hot-path regressions: 26 passed.
- FIT schedules/performance, MPS, MPO, and symmetry suites: 1,016 passed,
  1 skipped (before the final structural-zero test was added).
- Sampler, boundary-input, and fermionic-boundary suites: 181 passed,
  6 skipped.
- Default smoke suite: 128 passed.
- Repository Ruff and `git diff --check`: passed.

The full repository suite and GPU hardware were not exercised.

A small Torch CPU experiment used a 20-site complex128 MPS, guess bond 16,
target bond 24, and six block-size-two sweeps with one-site refinement.
Tensor multiplications fell from 95 to 19 (76 removed). Six alternating
post-warmup samples gave medians of 0.06519 s for the full move and 0.06490 s
for the shortcut. Those timings overlapped a test run and do not establish
a reliable speedup. The operation reduction is deterministic; larger CPU
and synchronized GPU performance measurements remain workload-dependent.
