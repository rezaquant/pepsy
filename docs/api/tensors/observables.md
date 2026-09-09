# Tensor observables

## MPS transfer spectra and correlation lengths

```python
from pepsy.tensors import mps_correlation_length, mps_transfer_spectrum

# Copy an open MPS, left-canonicalize it, and measure an interior bulk site.
xi = mps_correlation_length(psi, i)
spectrum = mps_transfer_spectrum(psi, i, k=6, canonicalize="left")
print(spectrum.normalized_eigenvalues[1:])
print(spectrum.correlation_lengths[1:])  # up to five subleading mode lengths
print(spectrum.gaps[1:], spectrum.momenta[1:])

# Or canonicalize from the opposite side and select a two-site bulk cell.
window = mps_transfer_spectrum(psi, i, cell_size=2, k=6, canonicalize="right")

# Explicit repeating arrays use (left bond, right bond, physical) axes.
xi = mps_correlation_length([A, B])
```

These functions extract transfer eigenvalues of a repeated cell. For an open
MPS, a selected bulk window supplies an **approximate repeating cell** after
canonicalizing a copy. Explicit arrays supply an **infinitely repeated unit
cell** directly. A cyclic Quimb `MatrixProductState` with at least three sites is
also accepted: its full tensor sequence becomes the cell, using its named
physical and bond indices. Its finite-ring environment and global `exponent`
are not used. Supply explicit arrays for one- and two-site cells.

### Choosing a bulk site and canonical form

The positional call resembles Quimb's `psi.entropy(i)`, but **i here is a
zero-based tensor site**, rather than a bipartition index. Open MPS input
requires an explicit site. `cell_size` selects consecutive tensors starting
there. Open-chain windows must stay in the interior; cyclic windows can wrap.
With no site argument, a cyclic MPS uses its full chain as the cell. To choose
the cut of a complete cyclic cell, use `i` with `cell_size=psi.L`; this does
not require `allow_local` and preserves the nonzero cell spectrum.

`canonicalize="auto"` (the default) makes an independent copy of an open MPS,
left-canonicalizes the whole copy with lossless Quimb QR, and selects the
requested tensors. The orthogonality center ends at the right edge.
`canonicalize="right"` places it at the left edge; `canonicalize="left"`
explicitly selects the default direction. No normalization or truncation is
requested. Stale isometry metadata on the copied tensors is cleared before
the sweep. The original tensors, data buffers, tags, exponent, and canonical
metadata are preserved. Torch buffers are detached before the diagnostic
sweep, and no global linalg registration is changed.

This **bulk estimate** assumes the selected tensors approximately repeat in
compatible bond bases: the transfer product through the bulk is approximated
by powers of the selected cell. Compare a few interior sites and, when useful,
both canonical directions to assess this assumption. Canonicalization alone
does not enforce translation invariance or remove all residual unitary gauge
freedom, so arbitrary nonuniform states need not give the same estimate at
different sites or in both directions. For exact finite-chain correlations,
measure connected correlators at the requested separations.

With `canonicalize=None, allow_local=True`, a local window is instead repeated
in its supplied gauge. Partial cyclic windows also require `allow_local=True`;
auto does not canonicalize a ring. Explicit left/right canonicalization is
accepted only for an open Quimb MPS. Explicit array cells and complete cyclic
cells retain their supplied repeating-cell representation under auto or None.

Neighboring virtual spaces and the two cell boundaries must match. Bond
dimensions may vary *inside* a window, so `cell_size=2` can work when a
single selected tensor has unequal left/right dimensions. A rectangular
transfer action is rejected; no padding or implicit basis map is introduced.
Symmray additionally requires matching boundary charges, dimensions, and duals.
Boundary compatibility is checked **after** canonicalization, since exact QR
can reduce oversized bond dimensions. The result records `sites`,
`interpretation="bulk_estimate"`, and `canonical_form="left"` or `"right"`
for canonicalized open-MPS input. Supplied-gauge windows have
`interpretation="local_repeated_cell"` and `canonical_form=None`.
Explicit cells and full cyclic cells have `interpretation="unit_cell"`.
The functions do not update optimizer state.

For a cell of `p` sites and transfer eigenvalues ordered by magnitude,

```text
xi = -p / log(abs(lambda_1 / lambda_0))
```

The action applies `sum_s A_s @ X @ A_s.conj().T` successively from the
rightmost cell tensor to the leftmost. It never constructs the full
`D**2 × D**2` transfer matrix. The general transfer operator is non-Hermitian,
so the iterative solver uses Arnoldi. Eigenvalue magnitudes determine decay;
their phases retain information about oscillations. An observable can decay
faster if it does not couple to the leading available mode.

### Array backends and solver controls

`mps_transfer_spectrum(unit_cell, i=None, *, cell_size=1, canonicalize="auto", allow_local=False,
k=2, sector=None, solver="auto", ncv=None, max_ncv=None, tol=None, maxiter=None, seed=0,
projected_solver="auto", degeneracy_tol=None)` returns a
`MpsTransferSpectrum` with:

- `eigenvalues`: up to `k` complex eigenvalues in decreasing magnitude order.
- `residuals`: `||T v - lambda v|| / (rho * ||v||)`, where `rho` is the
  largest returned magnitude. Zero spectra use one as the scale.
- `reference_eigenvalue`: the nonnegative neutral Perron root, including for
  a charged-sector query; phase normalization always uses this positive root.
- `reference_residual`: the relative residual of that reference eigenpair.
- `normalized_eigenvalues`: eigenvalues divided by that reference.
- `gaps`: `-log(abs(normalized_eigenvalues)) / unit_cell_size`, the transfer
  decay rates in inverse lattice sites. These are not Hamiltonian energies.
- `correlation_lengths`: inverse gaps, with infinite length for zero gaps.
- `momenta`: `angle(normalized_eigenvalues) / unit_cell_size`; zero modes have
  undefined phases and return NaN.
- `peripheral_mask`: candidate unit-magnitude modes within `degeneracy_tol`;
  this clustering does not alter gaps or lengths.
- `numerically_peripheral_mask`: magnitudes within ten machine epsilons of
  one, with combined residual screening at twenty epsilons. These have zero
  gap and infinite length, as a numerical classification of peripheral modes.
- `unresolved_mask`: small gaps that fail roundoff/residual screening;
  their gaps and lengths are NaN.
- `leading_degeneracy`: number of returned eigenvalues equal to the positive
  normalization root within `degeneracy_tol`. This can be a lower bound if
  `k` does not include the full degenerate space. Nonzero-phase peripheral
  modes are not counted as degenerate eigenvalues.
- `unit_cell_size`, `sites`, `interpretation`, `canonical_form`, `sector`, `is_neutral`,
  `solver`, and `projected_backend`: interpretation and solver information.
- `krylov_dimension`: final Arnoldi basis size, or configured SciPy basis size.

All per-mode arrays use the same indices and **include the leading mode**.
Ordering within tied magnitudes is unspecified. A normalization eigenvalue
has gap zero and length infinity. For ordinary dense/neutral input, the
scalar helper selects mode 1 by default, after the first leading mode.

NumPy, Torch, and CuPy are supported, including CUDA arrays. All cell arrays
or Symmray blocks must use the same backend, dtype, and device. For canonical
bulk estimates this is checked across the whole MPS before QR, preventing
implicit promotion during the sweep. Nonfinite input is rejected before QR
or the eigensolver. Float32 and
float64 inputs use complex64 and complex128 spectral work respectively;
complex inputs retain their precision. Eigenvalues and residual arrays return
on the original backend/device. Torch input is detached for these forward-only
diagnostics; the input itself retains its gradient metadata.

The **measurement API supports multiple backends; SciPy ARPACK runs on CPU**.
ARPACK accepts a matrix-free operator, but that does not make its Krylov
storage automatically compatible with Torch or CuPy. In this implementation:

| Input arrays / Symmray blocks | Default solver |
| --- | --- |
| NumPy | Quimb → SciPy ARPACK, on CPU; tiny sectors use Pepsy Arnoldi. |
| Torch CPU or CUDA | Pepsy Arnoldi with Torch contractions, vectors, and projected eigensolves. |
| CuPy | Pepsy Arnoldi with CuPy contractions and vectors; projected eigensolver follows the runtime capability described below. |

Bosonic sparse Symmray contracts natively and packs only the requested sector;
its numeric block backend selects the solver. `solver="scipy"` rejects
Torch/CuPy input instead of silently converting it to NumPy. Pepsy's backend
Arnoldi is a separate, newer implementation, not a GPU port of ARPACK.

| Option | Behavior |
| --- | --- |
| `solver="auto"` | Quimb/SciPy for NumPy blocks; Pepsy Arnoldi for Torch/CuPy. |
| `solver="scipy"` | NumPy blocks only; tiny sectors use Arnoldi when ARPACK cannot request `k`. |
| `solver="arnoldi"` | Ritz-restarted block Krylov iteration with two-pass orthogonalization. |
| `ncv` | Initial basis size, capped by sector dimension; defaults to `max(32, 4*k + 8)`. An explicit value fixes the basis unless `max_ncv` permits growth. |
| `max_ncv` | Arnoldi memory cap, at most the sector dimension. Defaults to `max(256, ncv)` for automatic `ncv`, otherwise `ncv`. Every three unsuccessful cycles the basis grows by 50% up to this cap. SciPy keeps a fixed `ncv`. |
| `tol` | Relative residual threshold: `1e-5` for single precision, `1e-10` for double precision by default. |
| `maxiter` | `None` defaults to 20 restart cycles for Pepsy Arnoldi and `10 * sector_dimension` update iterations for ARPACK. Explicit integers set the selected solver's budget. |
| `seed` | Nonnegative local random seed; global random state is preserved. ARPACK's starting vector uses the working complex precision. Its additional RNG is also seeded when the installed SciPy signature supports it. |
| `degeneracy_tol` | Clustering tolerance on normalized eigenvalues and magnitudes near one. Defaults to `10*machine_eps`. Does not close a resolved finite gap. |

The Arnoldi basis uses up to `O(sector_dimension * max_ncv)` storage. Its
projected matrix is at most `max_ncv × max_ncv`. Small sectors may fit entirely
in the basis. If CuPy's general eigensolver is unavailable
on the installed CUDA runtime, `projected_solver="auto"` emits a warning and
diagonalizes **only that projected matrix** with NumPy. The transfer action,
basis vectors, and returned results stay on the GPU. Set
`projected_solver="native"` to forbid this fallback. Scalar convergence
checks synchronize accelerator work; this is not a fully asynchronous solver.

Every accepted eigenpair is checked by applying the original transfer
operator. Pepsy Arnoldi additionally rechecks candidate convergence with
fresh transfer applications after its projected residual check. ARPACK
nonconvergence raises a diagnostic error with the converged count, basis
size, and iteration budget; partial spectra are never returned. Other ARPACK
errors also raise, with no automatic solver or backend fallback. ARPACK's
largest-magnitude non-Hermitian route follows its public
[SciPy interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html).

These checks establish backward accuracy, not a guarantee of accurate
eigenvalues for arbitrarily ill-conditioned nonnormal operators. Small gaps
and very long lengths should be checked with double-precision input and
tighter tolerances. The GPU solver has focused and reference-based coverage,
but does not have ARPACK's long production history.

`mps_correlation_length(unit_cell, i=None, *, cell_size=1, canonicalize="auto", allow_local=False, mode=None,
sector=None, **solver_options)` accepts the same solver controls except `k`
and returns a Python float. `mode` indexes the magnitude-ordered spectrum;
it defaults to 1 for dense/neutral input and 0 for charged sectors. An
explicit out-of-range mode raises. A bond-one product state's default length
is zero because there is no second mode.

Normalized magnitudes within ten machine epsilons of zero are treated as
unresolved zeros: their gap is infinite and length zero. Numerical peripheral
modes have infinite length. Other gaps comparable to the eigenpair and neutral
reference residuals plus roundoff are flagged unresolved and give NaN, rather
than being reported as degeneracy. Residuals measure backward accuracy; for a
nonnormal transfer operator they are not certified eigenvalue error bounds.
Use tighter solves and higher-precision input to check sensitive small gaps.
A zero normalization eigenvalue raises. Failed eigensolver
convergence also raises; increase `ncv`/`maxiter` or adjust `tol` rather than
treating an unconverged estimate as a measured length.

### GHZ and several transfer excitations

```python
import numpy as np

ghz_cell = np.zeros((2, 2, 2))
ghz_cell[0, 0, 0] = ghz_cell[1, 1, 1] = 1
s = mps_transfer_spectrum(ghz_cell, k=4)
# normalized_eigenvalues ~= [1, 1, 0, 0]
# gaps                   = [0, 0, inf, inf]
# correlation_lengths    = [inf, inf, 0, 0]
assert s.leading_degeneracy == 2
xi = mps_correlation_length(ghz_cell)  # inf
```

Degenerate leading modes are retained, so requesting the default length does
not silently skip GHZ's second normalization mode. To inspect additional
decay channels, request more eigenvalues with `k`, or select an explicit
`mode` in the scalar helper. A state with a degenerate leading space can
also have finite-length subleading modes; these remain in the same result.

For the standard GHZ state, `⟨Z_i⟩ = 0` and `⟨Z_i Z_j⟩ = 1` at any distinct
sites, so its connected Z correlation does not decay. Its connected X
correlation is zero for distinct sites when the chain has more than two sites.
An observable can miss a transfer mode; that does not remove the mode from
the spectrum. Infinite chain length alone also does not make a nonuniform
MPS uniform: bulk site independence requires additional structure, such as a
uniform or fixed repeating cell.

### Bosonic Symmray sectors

Sparse bosonic Symmray cells retain their native charges, duals, and block
backend, including Torch/CuPy blocks. The solver packs only allowed boundary
environment blocks in one charge sector. Open-MPS canonicalization also uses
native Symmray QR and preserves the block backend; it does not densify tensors.
The selected cell boundaries must still have matching charge spaces after QR.

```python
# For a U(1) cell with the indicated sectors present:
xi_neutral = mps_correlation_length([A_sym])
xi_charged = mps_correlation_length([A_sym], sector=1)
charged_modes = mps_transfer_spectrum([A_sym], sector=1, k=2)
```

The default sector is neutral. Its default correlation length uses its two
largest eigenvalue magnitudes. A nonneutral spectrum query also solves for
the neutral normalization eigenvalue so that every reported gap and length
uses the same physical reference. Sector labels use the
charge of the right-boundary environment with indices
`(right_bond.conj(), right_bond)`. No automatic search over all charge sectors
is performed; a charged sector can have a longer length than the neutral one.

Fermionic parity channels and flat Symmray storage are explicitly unsupported
in this first pass. Finite-MPS correlation fitting and differentiable
correlation-length losses are also outside this API's current scope.
