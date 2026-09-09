# MPS transfer diagnostics and upstream audit

Audit date: 2026-09-08. Scope: `pepsy.tensors.mps_transfer` and its lazy tensor
namespace exports. No optimizer, split-driver, or global
dispatch defaults change.

## Installed environment and API probes

The shared Python 3.12 environment contains:

| Package | Installed version |
| --- | --- |
| Quimb | `1.15.1.dev39+g369d09b9d` |
| Autoray | `0.11.1.dev1+gc56f64427` |
| Cotengra | `0.8.3.dev7+g1d7fd333f` |
| Cotengrust | `0.2.1` |
| Symmray | `0.3.2.dev7+gd63bb4e3f` |
| Torch | `2.6.0+cu124` |
| CuPy (`cupy-cuda12x`) | `14.1.1` |
| SciPy | `1.17.1` |

Torch CUDA and one CuPy CUDA device were available. CuPy reported runtime
`12090`; `cupyx.cusolver.check_availability("geev")` returned `False`.

Inspected installed signatures and dispatch, rather than relying on versions:

- `quimb.eig(A, *, isherm=False, k=-1, sort=True, return_vecs=True, **kwargs)`;
  partial solve forwards `ncv`, `tol`, `maxiter`, and `v0` to the SciPy route.
  Quimb has no top-level `eigs`. Its solver table contains NumPy, SciPy,
  PRIMME, LOBPCG, and SLEPc variants, not Torch/CuPy Arnoldi backends.
- `TensorNetwork.aslinearoperator(self, left_inds, right_inds, ldims=None,
  rdims=None, backend=None, optimize=None)` returns a SciPy `LinearOperator`;
  `_matvec` reshapes a dense input and calls `ravel` on its output. This is
  not a native Symmray sector-vector adapter.
- `MatrixProductState.bond(self, i, j)` and
  `Tensor.transpose(self, *output_inds, inplace=False)` support named-bond
  cell extraction without changing tensor metadata.
- `autoray.do(fn, *args, like=None, **kwargs)`: queried `conj`, `tensordot`,
  `linalg.eig`, and `random.array` dispatch for NumPy, Torch, CuPy, and
  Symmray. Contractions resolve to each owning array implementation;
  dense eig dispatch resolves to NumPy/Torch/CuPy. Symmray has no general
  `linalg.eig` dispatch. Random creation uses the existing Pepsy
  `backend_random_array` adapter for Autoray's `random.array` operation;
  there is no importable `autoray.random` module in this checkout.
- `cotengra.array_contract(arrays, inputs, output=None, optimize="auto",
  strip_exponent=False, cache_expression=True, backend=None, **kwargs)` was
  inspected. The three-tensor transfer step has a fixed two-contraction
  order, so no planner or compressed contraction is needed here.
- `Symmray.U1Array(indices, charge=None, blocks=(), symmetry=None, label=None)`,
  `copy_with(indices=None, charge=None, blocks=None)`,
  `tensordot(other, axes=2, mode="auto", preserve_array=False)`, and
  `conj(inplace=False)` were inspected. `BlockIndex.matches(other)` checks
  opposite-dual compatible spaces. `symmetry.combine`/`sign` enumerate the
  allowed two-leg environment sectors. There is no dense `ravel` or general
  `eig` method on these sparse arrays.
- CuPy exposes `linalg.eig(a)`, but calling it fails with
  `RuntimeError("geev is not available")` on this runtime. The public
  `cupyx.cusolver.check_availability(name)` probe is used before iteration.
  The compiled probe has no Python inspect signature; its docstring and
  a direct `"geev"` call establish the callable contract.

## Source review and dispositions

All required upstream sources were checked at task start and again after the
CuPy numerical test exposed the runtime capability mismatch:

| Source / opportunity | Disposition |
| --- | --- |
| [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html): partial eigensolver and named tensor indices | **adopt** public APIs for NumPy and cyclic cell extraction. |
| Quimb's new SDC compression, random split, and fermionic FIT options | **defer**: no compression or optimization occurs in this measurement. |
| [Autoray repository](https://github.com/jcmgray/autoray): backend dispatch and random arrays | **adopt** native arithmetic and the existing seeded random adapter. |
| [Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html): new planners, slicing and scale controls | **defer**: the transfer step has a fixed contraction order; existing planner defaults are untouched. The changelog page lags the installed development checkout. |
| [Symmray repository](https://github.com/jcmgray/symmray): sparse arrays over backend-native blocks | **adopt** public sparse constructors, contractions, and index matching. |
| [Symmray Abelian-array docs](https://symmray.readthedocs.io/en/latest/abelian_arrays.html) | The requested page could not be retrieved; repository material and installed source/signatures supplied the usable evidence. |
| GPU and symmetry-sector iterative eigenvectors | **prototype**: opt-in measurement API with Ritz-restarted block Krylov iteration, independent initial vectors, two-pass orthogonalization, and explicit residual checks. No installed package changes or vendored solver code. |
| [CuPy general eigensolver](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.eig.html) runtime availability | **compatibility shim**: only the small projected matrix goes to NumPy when unavailable, with a warning. The returned eigenpairs go back to the same device. `projected_solver="native"` disables fallback. |
| Fermionic parity channels, flat Symmray, gauge-invariant finite open-MPS bulk extraction, autodiff | **defer** with explicit capability errors where applicable. Local site selection is supported with the supplied-gauge interpretation below. |

## Numerical interpretation and validation

The cell action is `T0(T1(...Tp-1(X)))`. A cyclic MPS supplies one repeating
cell, not its finite-ring environment. Cell closure is explicit; internal
gauge transforms telescope around that closure. MPS exponents cancel from
eigenvalue ratios and are ignored. The array backend, precision, device,
native block structure, tensor tags, and canonical metadata are preserved.

The neutral sector contains the normalization mode. Nonneutral queries need
one leading eigenvalue from the requested sector and one from the neutral
sector. No automatic cross-sector maximum is reported. Peripheral degeneracy
or an unresolved gap within the requested tolerance gives infinite length.
Residual convergence is necessary but is not a rigorous eigenvalue-error
bound for a strongly nonnormal transfer operator; sensitive applications
should check stability with increased basis size and precision.

Focused validation lives in `tests/test_mps_transfer.py`: analytic channels,
cell-size and scale invariance, cyclic gauge and metadata preservation,
complex modes with restart convergence, product/zero/degenerate cases,
NumPy/Torch CPU/Torch CUDA/CuPy, forbidden full-array host conversion,
CuPy strict fallback rejection, sparse neutral/charged Symmray sectors on
NumPy/Torch CUDA/CuPy, structural zeros, and seeded random-state isolation.
`tests/test_public_api.py` covers owning namespace exports. Neighboring checks
are `test_package_layout.py`, `test_import_boundaries.py`,
`test_quimb_compat.py`, `test_contraction_dependencies.py`,
`test_tensor_constructors.py`, and `test_sampler.py`, plus the repository Ruff
gate. No new dependencies or optimizer defaults are introduced.

Initial-pass validation: 188 focused and neighboring tests passed, including actual
Torch CPU/CUDA and CuPy execution; `python -m ruff check src tests` and
`git diff --check` passed. The CuPy fallback warnings were expected. The full
repository suite was not run because production changes are confined to a new
tensor measurement module and lazy exports; existing optimizers are untouched.

## Site selection and mode-resolved API follow-up

Re-audited the same six upstream sources on 2026-09-08 before extending the
API. Installed versions and solver/Autoray dispatch entries were unchanged.
The Symmray Abelian-array page remained unavailable. Also inspected the actual
Quimb signatures and implementations:

- `MatrixProductState.entropy(self, i, info=None, method="svd")` uses a
  bipartition index and obtains squared Schmidt values through
  `schmidt_values`/`singular_values`.
- `canonicalize(self, where, cur_orthog="calc", info=None, bra=None,
  create_bond=False, inplace=False)` can move gauges on a copy. It does not
  fix the independent boundary-basis identification needed to repeat a
  nonuniform local tensor. No canonicalization route was added.
- `copy(self, virtual=False, deep=False)` was inspected, but site selection
  needs only the already-probed nonmutating named-index `transpose`.
- Autoray `where`, `angle`, `log`, `sum`, and native array arithmetic resolve
  on NumPy, Torch, and CuPy. New mode-derived arrays stay on these backends.

Dispositions: **adopt** Quimb's positional-selector ergonomics and named
tensor access. **prototype** the explicit local repeated-cell diagnostic for
nonuniform MPS. **defer** automatic canonicalization or a claim of a unique
finite-chain correlation length. Cotengra planning and Symmray fermionic/flat
opportunities remain deferred; the existing CuPy projection shim is unchanged.

`mps_transfer_spectrum(psi, i, cell_size=1, k=...)` now records the selected
sites. For an open MPS, selections must stay in the interior; cyclic selections
can wrap. Equal boundary dimensions are required, and Symmray boundaries must
also match their native charge spaces and duals. A multi-site cell can bridge
unequal intermediate dimensions. The supplied tensor gauge defines the
boundary identification. This is deliberately not asserted to be invariant
under independent boundary gauges, even for a canonically gauged finite MPS.

The result adds normalized eigenvalues, transfer gaps (inverse lengths),
momenta, mode lengths, and observed leading degeneracy. All arrays include
every requested mode; degenerate leading modes are never removed. The
`mode` argument of the scalar helper chooses a zero-based entry, defaulting
to 1 for dense/neutral input and 0 for a charged sector. Only the default
bond-one query maps an absent second mode to zero length; explicitly missing
modes raise.

`degeneracy_tol` controls gap closure separately from the residual tolerance.
Normalized magnitudes within ten machine epsilons of zero have infinite gap,
zero length, and undefined phase. The positive neutral spectral radius is
the phase reference, even when Arnoldi returns a negative or complex
peripheral eigenvalue first. A charged spectrum now obtains that same neutral
reference, so its derived mode data does not normalize against its own sector.

Follow-up validation: **195 tests passed** (20 transfer diagnostics, 12 closest
Quimb/contraction regressions, and 163 public API/layout/import/constructor/
sampler checks). Added actual finite GHZ site extraction with normalized
`[1, 1, 0, 0]`, a degenerate leading space with finite subleading decay modes
under both solvers, nonuniform site dependence, two-site windows, cyclic
wrapping, rectangular-cell rejection, separate degeneracy tolerance, and
periodic phases. Existing Torch CPU/CUDA, CuPy, and native bosonic Symmray
tests now also exercise site selection and derived mode arrays. Ruff and
`git diff --check` passed. The full repository suite was not run; changes
remain confined to tensor measurements and their documentation.

## Gauge, gap resolution, and restart correction — 2026-09-08

Rechecked the required Quimb changelog, Autoray repository, Cotengra docs and
changelog, and Symmray repository at the start of this correction. The
Symmray Abelian documentation URL remained unavailable. Installed versions
and the production signatures/dispatch above were unchanged. Also inspected
Quimb's public `copy`, `left_canonize`, and `right_canonize` for the canonical
gauge experiment. No production canonicalization or new upstream dispatch
route was introduced. The small tolerance regression during this correction
was traced to our classification of accumulated roundoff, not a changed
upstream API or package.

Dispositions: **adopt** the existing public NumPy/SciPy eigensolver and native
array operations unchanged; **compatibility shim** retains the capability
checked CuPy projected solve; **prototype** bounded adaptive Ritz restarts and
the explicitly opted-in local repeated-cell proxy; **defer** a new local
singular attenuation API, physical correlator fitting, and any automatic
claim that canonicalization produces a unique local eigenvalue spectrum.
The preceding sections describe earlier iterations; the behavior below
supersedes their local-selection and degeneracy-tolerance defaults.

### Canonicalization does not identify independent boundary bases

For finite left-canonical tensors, the allowed change
`A_i[s] -> U_(i-1).conj().T @ A_i[s] @ U_i` preserves each isometry. It also
preserves the state when the neighboring factors cancel. Independent `U`s
do not produce a similarity transformation of the one-tensor transfer map.

The deterministic GHZ regression places a Hadamard unitary and its inverse
on the bond between sites 3 and 4. Both tensors remain left isometries, and
the finite state changes by less than `2e-16` in norm. Nevertheless, the
site-3 repeated-tensor spectrum changes from `[1, 1, 0, 0]` to `[1, 0, 0, 0]`
in exact arithmetic. The latter map is defective at zero, so numerical
eigenvalues can deviate from zero by roughly the square root of roundoff.
Repeating public left/right canonicalization does not repair the ambiguity.

Partial MPS selection now requires `allow_local=True` on both public helpers;
the result remains labeled `local_repeated_cell`. It is a gauge-dependent
proxy. Explicit array cells and complete cyclic MPS cells keep the physical
repeating-cell interpretation. A full cyclic cell can be selected at any cut
with `i` and `cell_size=psi.L`; the nonzero spectrum is unchanged.

The same regression evaluates physical connected correlators: GHZ has
`<Z_i Z_j> - <Z_i><Z_j> = 1` for arbitrary distinct sites, while the connected
X correlation vanishes for more than two sites. Infinite length refers to a
nondecaying available transfer mode, not every observable. Increasing chain
length alone does not guarantee a uniform bulk or eliminate site dependence.

### Finite, numerical peripheral, and unresolved modes

`degeneracy_tol` now controls candidate clustering only, with a default of
ten machine epsilons independent of solver tolerance. It never rounds a
resolved finite gap to zero. `numerically_peripheral_mask` requires a
magnitude within ten epsilons of one and combined eigenpair/reference
residual screening within twenty epsilons. This is a numerical convention,
not a proof of degeneracy. Other gaps comparable to roundoff and the
normalized residuals are flagged by `unresolved_mask` and yield NaN gaps and
lengths. Neutral-reference residuals are included for charged Symmray modes.
Residual screening does not certify forward eigenvalue accuracy for
nonnormal transfer maps.

For the weakly mixed GHZ channel with mixing `1e-6`, the previous complex64
default incorrectly returned infinity because it used the residual tolerance
`1e-5` as a degeneracy tolerance. It now returns a finite estimate around
`578524`; complex128 returns `499999.500124`, agreeing with the analytic
`499999.500013`. Single precision still has limited accuracy for such a
small gap. Tests distinguish a finite estimate, loose clustering, and
insufficient residual accuracy rather than treating all three as degeneracy.

### Bounded adaptive restarts

Arnoldi retains `k + max(2, k//2)` Ritz directions (subject to basis capacity)
to leave room for new Krylov directions. Every three unsuccessful cycles it
grows the basis by 50% up to `max_ncv`. With automatic `ncv`, the default cap
is `min(sector_dimension, max(256, ncv))`; an explicit `ncv` fixes the basis
unless `max_ncv` separately permits growth. Returned `krylov_dimension`
records the final Arnoldi basis size or configured SciPy size. Failure still
raises with residual and basis information.

The seeded bond-dimension-10, physical-dimension-3, six-mode case that
previously stagnated now converges with default controls on NumPy,
Torch/CUDA, and CuPy. Larger exploratory NumPy checks at bond dimensions
16, 24, and 32 also converged, with maximum residuals around `1e-12` or
better. Fixed memory budgets and explicit failure remain tested. The
algorithm is matrix-free, but a small enough sector can fit entirely in its
Krylov basis; memory remains bounded by the documented cap.

Validation: **200 tests passed**: 25 transfer diagnostics, 12 nearest
Quimb/contraction regressions, and 163 public API/layout/import/constructor/
sampler checks. Backend coverage includes Torch CPU/CUDA, CuPy with the
documented projected solve warning, and bosonic Symmray NumPy/Torch/CuPy
blocks without densification. Ruff and `git diff --check` passed. The full
repository suite was not run; no optimizer or global backend configuration
changed. Fermionic transfer channels and flat Symmray storage remain deferred.

## Canonical bulk estimator — 2026-09-08

The user clarified the intended approximation: the bulk transfer operators
approximately repeat, so one interior tensor or short window should estimate
the decay through the bulk. At their request, open-MPS site calls now
canonicalize a private copy before extracting the cell. This supersedes the
earlier requirement to opt in to every open-MPS site selection.

`canonicalize="auto"` defaults to left canonical form for open MPS, with the
center at the right edge. Explicit `"right"` puts the center at the left
edge. Results record `interpretation="bulk_estimate"` and `canonical_form`.
All requested transfer modes and their lengths remain available, including
degenerate peripheral modes; slicing `[1:]` skips just the first normalization
mode, without discarding other degenerate modes. Supplied-gauge local proxies
remain accessible with `canonicalize=None, allow_local=True`.

Auto preserves explicit array cells and cyclic MPS without a sweep. Explicit
left/right sweeps reject these inputs because an open-chain canonical norm
cannot be assigned to a ring. Canonicalization supplies a practical convention
for the approximately repeating bulk assumption. It does not guarantee
arbitrary independent bond gauges or arbitrary nonuniform left/right forms
give identical local spectra. Tests retain the canonical GHZ counterexample
and independently verify an analytic approximately repeating bulk case.

All six required upstream source URLs were checked again for this maintenance
task. Versions remain those in the environment table above; the Symmray
Abelian-array documentation URL remains unavailable. New installed probes:

- `MatrixProductState.copy(self, virtual=False, deep=False)` copies network
  metadata but shares numeric data by default. The diagnostic explicitly
  copies every numeric buffer; Torch buffers are detached before copying.
- `left_canonize` and `right_canonize` accept
  `(self, stop=None, start=None, normalize=False, bra=None, create_bond=False,
  *, inplace=True)`. Both operate on the private copy with `normalize=False`.
- The public `left_canonize_site(self, i, bra=None, create_bond=False)` and
  `right_canonize_site` route through `tensor_canonize_bond`. Inspection of
  the latter showed that `left_inds` can skip a QR; copied tensors therefore
  clear this metadata before the requested full sweep.
- Autoray `copy` resolves to NumPy/CuPy copy or Autoray's Torch copy wrapper.
  `linalg.qr` and `qr_stabilized` resolve to the installed NumPy, Torch, CuPy,
  and Symmray implementations. No QR/SVD registry entry is changed.
- Native Symmray `copy_with` retains charge/index metadata while replacing
  blocks with private buffers. Probed `MPS_abelian_rand(symmetry, L, bond_dim,
  phys_dim=2, cyclic=False, seed=None, dtype="float64", ..., fermionic=False,
  flat=False, site_charge=None, subsizes="maximal", duals="reversed", **kwargs)`
  and both canonical sweeps on native U(1)/Z2 arrays. Exact QR can reduce
  oversized bond spaces, so selected-cell compatibility is checked afterward.

Dispositions: **adopt** public Quimb canonicalization and Autoray copies;
**prototype** the documented canonical bulk estimator; retain the CuPy
projected-eigensolver **compatibility shim**; **defer** unrelated new upstream
compression/planning algorithms, automatic bulk uniformity detection, and
fermionic/flat transfer channels. No upstream internals are vendored and no
global/backend/optimizer configuration is mutated.

Validation: **209 tests passed**: 34 transfer diagnostics, 12 nearest
Quimb/contraction regressions, and 163 public API/layout/import/constructor/
sampler checks. New tests recover the analytic Pauli-channel lengths after
nonunitary positive bond gauges, in both canonical directions, including
two-site windows. They preserve original array identity and values, index
order, tags, deliberately stale `left_inds`, exponent, dtype/device, and
Torch gradient metadata. Native U(1) canonical/charged spectra run without
densification on NumPy, Torch/CUDA, and CuPy. GHZ remains infinite under both
canonical directions for the ordinary bulk representation. Ruff and
`git diff --check` passed. The full suite was not run; no commit or push was
requested.

## ARPACK and backend reliability review — 2026-09-08

Rechecked all six required stack sources plus the public
[SciPy eigs API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html)
and [ARPACK guide](https://docs.scipy.org/doc/scipy/tutorial/arpack.html).
Installed versions remain unchanged. The Symmray Abelian documentation URL
is still unavailable. This review does not replace the backend-native
Arnoldi implementation with ARPACK: SciPy's ARPACK vectors live on CPU;
matrix-free reverse communication does not make them Torch/CuPy-native.

New installed probes and findings:

- Quimb `eig(A, *, isherm=False, k=-1, sort=True, return_vecs=True, **kwargs)`
  delegates partial solves to `eigensystem_partial(..., backend=None,
  fallback_to_scipy=False, **backend_opts)`. Its explicit `SCIPY` route calls
  `scipy.sparse.linalg.eigs` for `isherm=False`. It does not silently accept
  partial eigenpairs or choose another backend on ARPACK failure.
- Installed SciPy `eigs(A, k=6, M=None, sigma=None, which="LM", v0=None,
  ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None,
  OPinv=None, OPpart=None, rng=None)` accepts the additional `rng` argument.
  Capability inspection gates that argument for older installations.
  `ArpackNoConvergence(msg, eigenvalues, eigenvectors)` exposes the converged
  partial data, which Pepsy deliberately does not return as a spectrum.
- Autoray's NumPy/Torch/CuPy `linalg.eig` dispatch still resolves to those
  array libraries. The existing random helper forwards `dtype=None` if not
  explicitly supplied. A regression found that ARPACK's starting vector was
  therefore float64 even for a complex64 action; the transfer caller now
  supplies `dtype=action.dtype`, as the native Arnoldi caller already did.
- A mixed-precision open MPS, with only its first tensor in float32 and the
  rest float64, was silently promoted during QR and accepted at a float64
  bulk site. A shared validation boundary now checks all original tensor
  blocks before canonicalization, as well as explicit cells before solving.
  It rejects mixed backend/dtype/device and nonfinite data before QR or
  ARPACK can encounter it. Finiteness is reduced on the block backend.

Changes and dispositions:

- **Adopt** ARPACK's normal iteration budget: `maxiter=None` means
  `10 * sector_dimension` for SciPy and retains 20 restart cycles for native
  Arnoldi. Explicit budgets remain honored. This removes the old shared
  20-iteration CPU default without changing the memory budget.
- **Adopt**, capability-gated: seed the public SciPy `rng` parameter when
  available, alongside the seeded, correctly typed starting vector.
- **Adopt** explicit error reporting: ARPACK nonconvergence reports the
  converged count, `ncv`, and `maxiter`, preserving the original exception
  as the cause. Other ARPACK errors also raise. No partial spectrum or
  automatic solver/backend fallback is allowed.
- **Prototype**, hardened: native Arnoldi rechecks candidate convergence
  using fresh applications of the original transfer operator, beyond the
  residual calculated from stored projected images. Both eigensolvers thus
  report residuals measured against the original operator.
- Retain the warned, capability-gated CuPy projected eigensolver
  **compatibility shim**. **Defer** unrelated upstream compression/planner
  algorithms, fermionic/flat channels, and claims that the newer backend
  Arnoldi has ARPACK's production history.

Validation: **215 tests passed**: 40 transfer diagnostics, 12 closest
Quimb/contraction regressions, and 163 API/layout/import/constructor/sampler
checks. New regressions exercise actual ARPACK nonconvergence with no partial
result, complex64 ARPACK vector precision, default iteration budget, seeded
SciPy repeatability, a triangular nonnormal channel with an analytic spectrum,
and pre-QR rejection of mixed precision/nonfinite data. The canonical Torch
and CuPy tests now forbid host conversion except the documented small CuPy
projected matrix. Native U(1) NumPy blocks also explicitly exercise ARPACK,
in addition to the existing native Arnoldi coverage.

Additional bounded reference checks passed in **45 cases**: 36 NumPy solves
across random and clustered channels at bond dimensions 4, 10, and 16, three
seeds, and both solvers; nine GPU solves of clustered channels using Torch
complex64/complex128 and CuPy complex128. These compare requested leading
magnitudes with explicit dense reference matrices. Stronger nonnormal
experiments confirmed the documented limitation: tiny residuals alone do
not certify small forward eigenvalue error for ill-conditioned operators.
Very long correlation lengths should be checked in double precision with
tighter controls.

Ruff and `git diff --check` passed. The full repository suite was not run.
All changes remain in Pepsy, uncommitted on `develop`; no global registration
or installed package was changed. The supported diagnostics are covered by
focused and reference-based tests, but arbitrary nonnormal problems and the
newer GPU Arnoldi path should not be described as unconditionally mature.
