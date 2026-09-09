"""Transfer spectra of repeating MPS cells and local tensor windows.

The transfer action is matrix-free. Dense and bosonic block-sparse arrays
share the eigensolver through a bond-environment packing boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any
import warnings

import autoray as ar
import numpy as np

from .._internal.random import backend_random_array

__all__ = ["MpsTransferSpectrum", "mps_transfer_spectrum", "mps_correlation_length"]


@dataclass(frozen=True)
class MpsTransferSpectrum:
    """Leading transfer eigenvalues, sorted by decreasing magnitude.

    ``eigenvalues`` and ``residuals`` stay on the input array backend/device.
    Residuals are ``||T v - lambda v|| / (rho * ||v||)``, where ``rho`` is
    the largest returned eigenvalue magnitude. ``sector=None`` denotes the
    full dense space; a Symmray result records the selected charge sector.
    These are forward-only diagnostics, without an autodiff contract.
    """

    eigenvalues: Any
    residuals: Any
    unit_cell_size: int
    sector: Any
    solver: str
    projected_backend: str
    reference_eigenvalue: Any
    reference_residual: Any
    krylov_dimension: int
    sites: tuple[int, ...] | None
    interpretation: str
    canonical_form: str | None
    is_neutral: bool
    degeneracy_tol: float

    @property
    def normalized_eigenvalues(self):
        """Eigenvalues divided by the neutral normalization eigenvalue."""
        return self.eigenvalues / self.reference_eigenvalue

    @property
    def _zero_mask(self):
        return ar.do("abs", self.normalized_eigenvalues) <= 10 * self._eps

    @property
    def _eps(self):
        dtype = ar.get_dtype_name(self.eigenvalues)
        return np.finfo("float32" if dtype == "complex64" else "float64").eps

    @property
    def _gap_uncertainty(self):
        magnitudes = ar.do("abs", self.normalized_eigenvalues)
        uncertainty = self.residuals * magnitudes[0] + self.reference_residual * magnitudes
        if self.is_neutral:
            # The first magnitude is divided by itself: its normalization gap
            # is identically zero, even when the absolute Perron root is noisy.
            uncertainty = ar.do("concatenate", [uncertainty[:1] * 0, uncertainty[1:]])
        return uncertainty

    @property
    def peripheral_mask(self):
        """Candidate peripheral modes within the requested clustering tolerance.

        This clustering does not round finite gaps to zero.
        """
        return ar.do("abs", ar.do("abs", self.normalized_eigenvalues) - 1) <= self.degeneracy_tol

    @property
    def numerically_peripheral_mask(self):
        """Unit magnitudes at roundoff with residuals also near roundoff.

        This is a numerical classification, not a proof of exact degeneracy.
        """
        delta = ar.do("abs", ar.do("abs", self.normalized_eigenvalues) - 1)
        return (delta <= 10 * self._eps) & (self._gap_uncertainty <= 20 * self._eps)

    @property
    def unresolved_mask(self):
        """Modes whose small gap is unresolved by roundoff/residual screening.

        Residuals are not eigenvalue error bounds for a nonnormal operator;
        passing this screen does not certify the forward accuracy of a gap.
        """
        magnitudes = ar.do("abs", self.normalized_eigenvalues)
        delta = ar.do("abs", magnitudes - 1)
        return ~self.numerically_peripheral_mask & (
            (delta <= 10 * self._eps + self._gap_uncertainty) | (magnitudes > 1)
        )

    @property
    def leading_degeneracy(self):
        """Number of returned eigenvalues equal to the normalization mode.

        This is a lower bound if k does not cover its full multiplicity.
        Unit-magnitude modes with nonzero phase are counted only by
        ``peripheral_mask``, not as degenerate eigenvalues here.
        """
        return int(ar.do("sum", ar.do("abs", self.normalized_eigenvalues - 1) <= self.degeneracy_tol))

    @property
    def gaps(self):
        """Inverse correlation lengths in inverse lattice sites, for every mode.

        These are transfer decay rates, not Hamiltonian excitation energies.
        Numerically peripheral modes have zero gap; unresolved small gaps are
        NaN. Magnitudes within ten machine epsilons of zero are treated as
        unresolved zeros. Clustering with degeneracy_tol does not close gaps.
        """
        magnitudes = ar.do("abs", self.normalized_eigenvalues)
        mask = self._zero_mask | self.numerically_peripheral_mask | self.unresolved_mask
        safe = ar.do("where", mask, magnitudes * 0 + 1, magnitudes)
        gaps = -ar.do("log", safe) / self.unit_cell_size
        gaps = ar.do("where", self._zero_mask, ar.do("ones_like", gaps) * math.inf, gaps)
        return ar.do("where", self.unresolved_mask, ar.do("ones_like", gaps) * math.nan, gaps)

    @property
    def correlation_lengths(self):
        """Length for every returned mode, including the normalization mode."""
        gaps = self.gaps
        zero = gaps == 0
        safe = ar.do("where", zero, ar.do("ones_like", gaps), gaps)
        lengths = 1 / safe
        return ar.do("where", zero, ar.do("ones_like", lengths) * math.inf, lengths)

    @property
    def momenta(self):
        """Transfer phases per lattice site; undefined (NaN) for zero modes."""
        phases = ar.do("angle", self.normalized_eigenvalues) / self.unit_cell_size
        return ar.do("where", self._zero_mask, phases * 0 + math.nan, phases)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _canonical_copy(mps, direction):
    """Own diagnostic buffers and metadata before a lossless Quimb QR sweep."""
    native, _, _, _ = _validate_arrays(tuple(t.data for t in mps.tensors))
    work = mps.copy()

    def copy_block(a):
        if ar.infer_backend(a) == "torch":
            a = a.detach()
        return ar.do("copy", a)

    for tensor in work.tensors:
        a = tensor.data
        if native:
            data = a.copy_with(blocks={q: copy_block(b) for q, b in a.blocks.items()})
        else:
            data = copy_block(a)
        # Do not let possibly stale user isometry metadata skip the sweep.
        tensor.modify(data=data, left_inds=None)
    getattr(work, f"{direction}_canonize")(normalize=False)
    return work


def _extract_cell(unit_cell, i=None, cell_size=1, allow_local=False, canonicalize="auto"):
    """Select a cell, optionally canonicalizing a private open-MPS copy."""
    from quimb.tensor import MatrixProductState

    cell_size = _positive_int(cell_size, "cell_size")
    if not isinstance(allow_local, bool):
        raise TypeError("allow_local must be a bool.")
    if canonicalize not in ("auto", "left", "right", None):
        raise ValueError("canonicalize must be 'auto', 'left', 'right', or None.")
    sites = None
    interpretation = "unit_cell"
    canonical_form = None
    if isinstance(unit_cell, MatrixProductState):
        if i is None and not unit_cell.cyclic:
            raise ValueError(
                "An open MPS requires an interior site i for a bulk correlation-length estimate."
            )
        if canonicalize in ("left", "right") and unit_cell.cyclic:
            raise ValueError("Left/right canonicalization requires an open MPS.")
        if not unit_cell.cyclic:
            canonical_form = "left" if canonicalize == "auto" else canonicalize
        if unit_cell.cyclic and unit_cell.L < 3:
            raise ValueError(
                "For cells shorter than three sites, pass explicit "
                "(left, right, physical) arrays to disambiguate loop bonds."
            )
        if i is None:
            if cell_size != 1:
                raise ValueError("cell_size requires an explicit starting site i.")
            sites = tuple(range(unit_cell.L))
        else:
            if isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < unit_cell.L:
                raise ValueError("i must be a site index with 0 <= i < MPS length.")
            if cell_size > unit_cell.L:
                raise ValueError("cell_size cannot exceed the MPS length.")
            if unit_cell.cyclic:
                sites = tuple((int(i) + offset) % unit_cell.L for offset in range(cell_size))
            else:
                if i == 0 or i + cell_size >= unit_cell.L:
                    raise ValueError("A local open-MPS cell must contain only interior sites.")
                sites = tuple(range(int(i), int(i) + cell_size))
            if not unit_cell.cyclic or cell_size != unit_cell.L:
                interpretation = "bulk_estimate" if canonical_form else "local_repeated_cell"
                if not canonical_form and not allow_local:
                    raise ValueError(
                        "Repeating a local MPS window is gauge dependent, even in "
                        "left/right canonical form. Use canonicalize='left' or 'right' "
                        "on an open MPS for a bulk estimate, allow_local=True for "
                        "a supplied-gauge proxy, or a complete repeating unit cell."
                    )
        if canonical_form:
            unit_cell = _canonical_copy(unit_cell, canonical_form)
        arrays = tuple(
            unit_cell[site].transpose(
                unit_cell.bond((site - 1) % unit_cell.L, site),
                unit_cell.bond(site, (site + 1) % unit_cell.L),
                unit_cell.site_ind(site),
            ).data
            for site in sites
        )
    else:
        if canonicalize in ("left", "right"):
            raise ValueError("canonicalize='left'/'right' requires an open Quimb MPS.")
        if i is not None or cell_size != 1:
            raise ValueError("i and cell_size select tensors from a Quimb MPS, not an array cell.")
        if getattr(unit_cell, "ndim", None) == 3:
            arrays = (unit_cell,)
        else:
            try:
                arrays = tuple(unit_cell)
            except TypeError as exc:
                raise TypeError("unit_cell must be a Quimb MPS or a sequence of arrays.") from exc
    if not arrays:
        raise ValueError("unit_cell must not be empty.")
    for i, a in enumerate(arrays):
        if getattr(a, "ndim", None) != 3 or any(d < 1 for d in a.shape):
            raise ValueError(f"Cell tensor {i} must have shape (left, right, physical).")
    for i, a in enumerate(arrays):
        if a.shape[1] != arrays[(i + 1) % len(arrays)].shape[0]:
            raise ValueError(
                f"Cell bond after tensor {i} does not close consistently. "
                "Local extraction requires equal left and right boundary dimensions; "
                "choose a different site or cell_size."
            )
    return arrays, sites, interpretation, canonical_form


def _device(a, backend):
    if backend == "torch":
        return str(a.device)
    if backend == "cupy":
        return a.device.id
    return None


def _working_array(a, dtype):
    # Diagnostics must not retain the input's Torch autograd graph.
    if ar.infer_backend(a) == "torch":
        a = a.detach()
    return ar.do("astype", a, dtype)


def _validate_arrays(arrays):
    """Validate before QR/contractions can promote mixed input or encounter NaN."""
    native = ar.infer_backend(arrays[0]) == "symmray"
    blocks = []
    for a in arrays:
        if (ar.infer_backend(a) == "symmray") != native:
            raise TypeError("Do not mix dense and Symmray tensors in a unit cell.")
        if native:
            if getattr(a, "fermionic", False):
                raise NotImplementedError("Fermionic transfer parity channels are not supported yet.")
            if not isinstance(a.blocks, Mapping):
                raise NotImplementedError("Only sparse bosonic Symmray storage is supported.")
            if not a.blocks:
                raise ValueError("Cell tensors must contain at least one array block.")
            blocks.extend(a.blocks.values())
        else:
            blocks.append(a)

    backend = ar.infer_backend(blocks[0])
    if backend not in {"numpy", "torch", "cupy"}:
        raise NotImplementedError("Supported block backends are NumPy, Torch, and CuPy.")
    dtype = ar.get_dtype_name(blocks[0])
    if dtype not in {"float32", "float64", "complex64", "complex128"}:
        raise TypeError("Transfer spectra require float32/64 or complex64/128 arrays.")
    device = _device(blocks[0], backend)
    if any(
        ar.infer_backend(b) != backend
        or ar.get_dtype_name(b) != dtype
        or _device(b, backend) != device
        for b in blocks
    ):
        raise TypeError("All cell blocks must share an array backend, dtype, and device.")
    # Keep this reduction on the block backend, with only one scalar check.
    finite = ar.do("stack", [ar.do("all", ar.do("isfinite", b)) for b in blocks])
    if not bool(ar.do("all", finite)):
        raise ValueError("Transfer input arrays must be finite.")
    return native, blocks, backend, dtype


class _TransferAction:
    """Right-to-left cell action on a dense or packed sector environment."""

    def __init__(self, unit_cell, sector):
        arrays, _, _, _ = _extract_cell(unit_cell)
        native, blocks, self.backend, dtype = _validate_arrays(arrays)
        self.dtype = "complex64" if dtype in {"float32", "complex64"} else "complex128"
        self.eps = np.finfo("float32" if self.dtype == "complex64" else "float64").eps
        self.like = _working_array(blocks[0], self.dtype)
        self.arrays = tuple(
            a.copy_with(blocks={q: _working_array(b, self.dtype) for q, b in a.blocks.items()})
            if native else _working_array(a, self.dtype)
            for a in arrays
        )
        self.bras = tuple(ar.do("conj", a) for a in self.arrays)
        self.native = native
        self.cell_size = len(arrays)
        self.sector = None
        if not native:
            if sector is not None:
                raise ValueError("sector is only meaningful for Symmray unit cells.")
            self.bond_dim = arrays[-1].shape[1]
            self.size = self.bond_dim**2
            return

        a = self.arrays[-1]
        for i, current in enumerate(self.arrays):
            following = self.arrays[(i + 1) % len(arrays)]
            if (
                type(current) is not type(a)
                or current.symmetry != a.symmetry
                or not current.indices[1].matches(following.indices[0])
            ):
                raise ValueError("Symmray cell bonds must match in charges, dimensions, and duals.")
        self.sector = a.symmetry.combine() if sector is None else sector
        if not a.symmetry.valid(self.sector):
            raise ValueError(f"Invalid symmetry sector {self.sector!r}.")
        right = a.indices[1]
        indices = (right.conj(), right)
        self.layout = []
        offset = 0
        for q0, d0 in indices[0].chargemap.items():
            for q1, d1 in indices[1].chargemap.items():
                charge = a.symmetry.combine(
                    a.symmetry.sign(q0, indices[0].dual),
                    a.symmetry.sign(q1, indices[1].dual),
                )
                if charge == self.sector:
                    self.layout.append(((q0, q1), (d0, d1), slice(offset, offset + d0 * d1)))
                    offset += d0 * d1
        if not offset:
            raise ValueError(f"Sector {self.sector!r} is absent from the cell boundary.")
        self.size = offset
        self.template = type(a)(indices=indices, charge=self.sector, blocks={}, symmetry=a.symmetry)

    def __call__(self, vector):
        if self.native:
            x = self.template.copy_with(blocks={
                q: ar.do("reshape", vector[sl], shape) for q, shape, sl in self.layout
            })
        else:
            x = ar.do("reshape", vector, (self.bond_dim, self.bond_dim))
        # At no point form A x conj(A) as a four-leg transfer tensor.
        for a, bra in zip(reversed(self.arrays), reversed(self.bras)):
            x = ar.do("tensordot", a, x, axes=((1,), (0,)))
            x = ar.do("tensordot", x, bra, axes=((1, 2), (2, 1)))
        if not self.native:
            return ar.do("reshape", x, (-1,))
        return ar.do("concatenate", [
            ar.do("reshape", x.blocks[q], (-1,)) if q in x.blocks
            else vector[sl] * 0
            for q, _, sl in self.layout
        ])


def _norm(a):
    return ar.do("linalg.norm", a)


def _adjoint(a):
    return ar.do("transpose", ar.do("conj", a))


def _leading(values, vectors, k):
    # Negation works on Torch too, where negative-stride slices do not.
    order = ar.do("argsort", -ar.do("abs", values))[:k]
    return values[order], vectors[:, order]


def _residuals(action, values, vectors, images=None):
    if images is None:
        images = ar.do("stack", [action(vectors[:, j]) for j in range(len(values))], axis=1)
    rho = float(ar.do("max", ar.do("abs", values)))
    if not math.isfinite(rho):
        raise ValueError("Transfer eigenvalues are non-finite; check cell tensor scales.")
    scale = rho if rho > 0 else 1.0
    return ar.do("stack", [
        _norm(images[:, j] - values[j] * vectors[:, j]) / (_norm(vectors[:, j]) * scale)
        for j in range(len(values))
    ])


def _projection_eigensolver(action, projected_solver):
    if action.backend == "cupy":
        import cupy as cp
        from cupyx.cusolver import check_availability

        if not hasattr(cp.linalg, "eig") or not check_availability("geev"):
            if projected_solver == "native":
                raise NotImplementedError(
                    "CuPy's general eigensolver is unavailable on this CUDA runtime; "
                    "projected_solver='auto' permits a small projected-matrix CPU solve."
                )
            warnings.warn(
                "CuPy's general eigensolver is unavailable; only the small Krylov "
                "projected matrix will be diagonalized with NumPy. "
                "Transfer contractions and Krylov vectors remain on the input device.",
                RuntimeWarning,
                stacklevel=3,
            )

            def eig_projected(h):
                values, vectors = np.linalg.eig(cp.asnumpy(h))
                with cp.cuda.Device(h.device.id):
                    return cp.asarray(values), cp.asarray(vectors)

            return eig_projected, "numpy"
    return lambda h: ar.do("linalg.eig", h), action.backend


def _arnoldi(action, k, ncv, max_ncv, tol, maxiter, seed, projected_solver):
    """Ritz-restarted block Krylov iteration with two-pass orthogonalization.

    All basis vectors and transfer applications use the block backend.
    Scalar norm checks synchronize accelerator execution.
    Independent initial vectors and breakdown completion retain multiplicities.
    """
    eig_projected, projected_backend = _projection_eigensolver(action, projected_solver)
    draws = 0

    def random_vector():
        nonlocal draws
        v = backend_random_array(
            (action.size,), like=action.like, dtype=action.dtype, rng=seed + draws
        )
        draws += 1
        return v

    def append_orthogonal(basis, v):
        initial_norm = float(_norm(v))
        if not math.isfinite(initial_norm):
            raise ValueError("Non-finite transfer action; check cell tensor scales.")
        if basis:
            q = ar.do("stack", basis, axis=1)
            for _ in range(2):
                v = v - q @ (_adjoint(q) @ v)
        length = float(_norm(v))
        if length <= 10 * action.eps * initial_norm or length == 0:
            return False
        basis.append(v / length)
        return True

    seeds = [random_vector() for _ in range(k)]
    for cycle in range(maxiter):
        # A short restarted block Krylov sequence can stagnate even on benign
        # channels. Grow the search space within the caller's memory budget.
        if cycle and cycle % 3 == 0:
            ncv = min(max_ncv, ncv + max(1, ncv // 2))
        basis = []
        for v in seeds:
            append_orthogonal(basis, v)
        images = []
        cursor = 0
        while cursor < ncv:
            if cursor == len(basis):
                for _ in range(10):
                    if append_orthogonal(basis, random_vector()):
                        break
                else:
                    raise RuntimeError("Could not complete the Krylov basis.")
            av = action(basis[cursor])
            images.append(av)
            if len(basis) < ncv:
                append_orthogonal(basis, av)
            cursor += 1
        q = ar.do("stack", basis, axis=1)
        aq = ar.do("stack", images, axis=1)
        values, coeffs = eig_projected(_adjoint(q) @ aq)
        values, coeffs = _leading(values, coeffs, ncv)
        vectors = q @ coeffs[:, :k]
        residuals = _residuals(action, values[:k], vectors, aq @ coeffs[:, :k])
        if bool(ar.do("all", ar.do("isfinite", residuals))) and float(ar.do("max", residuals)) <= tol:
            # Reapply the original operator: projected residuals alone can
            # hide cancellation/roundoff in the stored Krylov images.
            residuals = _residuals(action, values[:k], vectors)
            if bool(ar.do("all", ar.do("isfinite", residuals))) and float(ar.do("max", residuals)) <= tol:
                return values[:k], residuals, projected_backend, ncv
        # Leave enough room for new Krylov directions after every restart.
        retain = min(k + max(2, k // 2), ncv - 1)
        restart = q @ coeffs[:, :retain]
        seeds = [restart[:, j] for j in range(retain)]
    raise RuntimeError(
        f"Transfer Arnoldi did not converge in {maxiter} restart cycles "
        f"with ncv={ncv} (largest relative residual "
        f"{float(ar.do('max', residuals)):.3g}). "
        "Increase ncv, max_ncv, or maxiter, or choose a tolerance appropriate to the dtype."
    )


def _scipy_eigenpairs(action, k, ncv, tol, maxiter, seed):
    """CPU ARPACK through Quimb, with no partial-result or backend fallback."""
    from inspect import signature

    from quimb import eig
    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, eigs

    if ncv <= k + 1:
        raise ValueError("SciPy's complex Arnoldi solver requires ncv > k + 1.")
    op = LinearOperator((action.size, action.size), matvec=action, dtype=action.dtype)
    v0 = backend_random_array((action.size,), like=action.like, dtype=action.dtype, rng=seed)
    # New SciPy releases also expose the RNG used during ARPACK breakdown
    # completion. Capability gate it for older supported environments.
    options = {"rng": np.random.default_rng(seed)} if "rng" in signature(eigs).parameters else {}
    try:
        values, vectors = eig(
            op, k=k, isherm=False, which="LM", backend="SCIPY", sort=False,
            ncv=ncv, tol=tol, maxiter=maxiter, v0=v0, fallback_to_scipy=False, **options,
        )
    except ArpackNoConvergence as exc:
        raise RuntimeError(
            f"Transfer ARPACK did not converge ({len(exc.eigenvalues)}/{k} eigenpairs, "
            f"ncv={ncv}, maxiter={maxiter}). No partial spectrum is returned. "
            "Increase ncv or maxiter, or choose solver='arnoldi'."
        ) from exc
    except ArpackError as exc:
        raise RuntimeError(
            f"Transfer ARPACK failed with ncv={ncv}: {exc}. "
            "Increase ncv or choose solver='arnoldi'."
        ) from exc
    values, vectors = _leading(values, vectors, k)
    # Quimb may wrap vectors as qarray; keep standard backend arrays here.
    values, vectors = np.asarray(values), np.asarray(vectors)
    residuals = _residuals(action, values, vectors)
    if not np.all(np.isfinite(residuals)) or np.max(residuals) > tol:
        raise RuntimeError("Transfer eigenpairs failed the requested residual tolerance.")
    return values, residuals


def mps_transfer_spectrum(
    unit_cell, i=None, *, cell_size=1, canonicalize="auto", allow_local=False, k=2, sector=None,
    solver="auto", ncv=None, max_ncv=None, tol=None, maxiter=None, seed=0,
    projected_solver="auto", degeneracy_tol=None,
):
    """Compute a few transfer modes of an MPS cell selected by site.

    Parameters
    ----------
    unit_cell : quimb.MatrixProductState or sequence of rank-three arrays
        Explicit repeating cell. Arrays use ``(left, right, physical)`` order,
        including one- and two-site cells. Neighboring bonds and the cell
        boundary must match in the same gauge. Without i, a cyclic MPS is
        interpreted as a cell repeated infinitely; its finite ring environment
        and ``exponent`` are not used. Open MPS input requires i and defaults
        to a bulk estimate from a left-canonicalized copy.
    i : int, optional
        Zero-based starting site of a local cell, including for a nonuniform
        open MPS. This is a site, not entropy's bipartition index. A local
        estimate assumes approximately repeating bulk tensors in compatible
        bond bases. It is not an exact finite-chain correlation length.
    cell_size : int, default=1
        Number of consecutive selected sites. Open-MPS selections must stay
        in the interior; cyclic selections may wrap. Cell boundary dimensions
        (and Symmray charge spaces) must match after canonicalization.
    canonicalize : {'auto', 'left', 'right', None}
        Auto left-canonicalizes a private copy of an open MPS, placing the
        center at the right edge. Right places it at the left edge. Both use
        lossless Quimb QR, without normalization or truncation. Auto leaves
        explicit array cells and cyclic MPS unchanged; explicit left/right
        require an open MPS. None uses supplied tensors. Canonicalization
        provides a bulk-estimation convention, not a unique gauge invariant
        definition for arbitrary nonuniform states.
    allow_local : bool, default=False
        With canonicalize=None, opt in to repeating a partial MPS window in
        its supplied gauge. Also needed for partial cyclic cells in auto.
        Such a diagnostic depends on independent boundary gauges, including
        residual unitary gauges in left/right canonical form. A complete
        cyclic cell is accepted without this opt-in, including at a chosen i.
    k : int, default=2
        Number of largest-magnitude eigenvalues, capped by the sector dimension.
    sector : charge, optional
        Bosonic sparse Symmray charge sector; defaults to the neutral sector.
        Dense input has no sector selector. Fermionic and flat Symmray storage
        are not yet supported.
    solver : {'auto', 'scipy', 'arnoldi'}
        Auto uses Quimb/SciPy for NumPy blocks and Pepsy's Ritz-restarted Arnoldi
        for Torch/CuPy blocks. SciPy is rejected for non-NumPy blocks. Tiny
        NumPy problems use the native Arnoldi path when ARPACK cannot request k.
    ncv : int, optional
        Initial Krylov basis size, default ``min(dimension, max(32, 4*k + 8))``.
        Must exceed k unless the full sector fits. An explicit ncv fixes the
        basis size unless max_ncv also permits growth.
    max_ncv : int, optional
        Arnoldi basis cap. Defaults to min(dimension, max(256, ncv)) when
        ncv is omitted, otherwise ncv. After three unsuccessful cycles Arnoldi
        grows the basis by 50%, up to this cap. Memory is O(dimension*max_ncv).
        Does not change SciPy's basis size.
    tol : float, optional
        Maximum relative eigenpair residual. Defaults to 1e-5 for single
        precision and 1e-10 for double precision.
    maxiter : int, optional
        Maximum restart cycles for Arnoldi (default 20), or ARPACK iterations
        for SciPy (default 10 times the transfer sector dimension).
    seed : nonnegative int, default=0
        Local random seed, without changing global random state.
    projected_solver : {'auto', 'native'}
        Auto warns and uses NumPy for only the small projected matrix if
        CuPy's general eigensolver is unavailable at runtime. Native forbids
        that fallback. Basis vectors and transfer actions stay on the device.
    degeneracy_tol : float, optional
        Absolute tolerance on normalized eigenvalue differences and on
        magnitude differences from one. Defaults to 10*machine_eps. Controls
        clustering only, never rounds a finite gap to zero. Numerical
        peripheral modes at roundoff have infinite length; unresolved small
        gaps have NaN length. See the result's corresponding masks.

    Returns
    -------
    MpsTransferSpectrum
        Complex eigenvalues, residuals, normalized eigenvalues, transfer gaps,
        momenta, and mode correlation lengths on the input backend/device.
        All mode arrays include the leading mode. ``sites`` and
        ``interpretation`` and ``canonical_form`` record local extraction.
        Canonicalized open-MPS input is labeled ``bulk_estimate``.
        Nonneutral sectors also
        solve for the neutral reference eigenvalue used to normalize modes.
        Real inputs use the corresponding complex precision for spectral work.
        Input arrays, tags, gauges, and canonical metadata are not modified.
        Failure to converge raises instead of returning an unchecked length.
    """
    k = _positive_int(k, "k")
    if maxiter is not None:
        maxiter = _positive_int(maxiter, "maxiter")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if solver not in {"auto", "scipy", "arnoldi"}:
        raise ValueError("solver must be 'auto', 'scipy', or 'arnoldi'.")
    if projected_solver not in {"auto", "native"}:
        raise ValueError("projected_solver must be 'auto' or 'native'.")
    requested_solver, requested_ncv = solver, ncv
    arrays, sites, interpretation, canonical_form = _extract_cell(
        unit_cell, i, cell_size, allow_local, canonicalize,
    )
    action = _TransferAction(arrays, sector)
    k = min(k, action.size)
    if tol is None:
        tol = 1e-5 if action.dtype == "complex64" else 1e-10
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    if degeneracy_tol is None:
        degeneracy_tol = 10 * action.eps
    if not math.isfinite(degeneracy_tol) or not 0 <= degeneracy_tol < 1:
        raise ValueError("degeneracy_tol must be finite and satisfy 0 <= degeneracy_tol < 1.")
    ncv = min(action.size, max(32, 4 * k + 8) if ncv is None else _positive_int(ncv, "ncv"))
    if ncv < k or (ncv == k and k < action.size):
        raise ValueError("ncv must exceed k unless it spans the full transfer sector.")
    requested_max_ncv = max_ncv
    if max_ncv is None:
        max_ncv = max(256, ncv) if requested_ncv is None else ncv
    max_ncv = min(action.size, _positive_int(max_ncv, "max_ncv"))
    if max_ncv < ncv:
        raise ValueError("max_ncv must be at least ncv (after capping to the sector dimension).")
    if solver == "auto":
        solver = "scipy" if action.backend == "numpy" else "arnoldi"
    if solver == "scipy" and action.backend != "numpy":
        raise ValueError("The SciPy solver requires NumPy blocks; use solver='arnoldi'.")
    if solver == "scipy" and k >= action.size - 1:
        solver = "arnoldi"
    if solver == "arnoldi":
        values, residuals, projected_backend, ncv = _arnoldi(
            action, k, ncv, max_ncv, tol, 20 if maxiter is None else maxiter, int(seed), projected_solver
        )
    else:
        values, residuals = _scipy_eigenpairs(
            action, k, ncv, tol, 10 * action.size if maxiter is None else maxiter, int(seed),
        )
        projected_backend = "numpy"
    is_neutral = not action.native or action.sector == action.arrays[0].symmetry.combine()
    if is_neutral:
        # A completely positive transfer map has a nonnegative Perron root.
        # A periodic map can return -rho or another peripheral phase first;
        # normalize phases against +rho, never that arbitrary tied mode.
        reference = ar.do("abs", values[0])
        reference_residual = residuals[0]
    else:
        neutral = mps_transfer_spectrum(
            arrays, k=1, solver=requested_solver, ncv=requested_ncv, tol=tol,
            maxiter=maxiter, seed=seed, projected_solver=projected_solver,
            degeneracy_tol=degeneracy_tol, max_ncv=requested_max_ncv,
        )
        reference = neutral.reference_eigenvalue
        reference_residual = neutral.reference_residual
    leading = float(ar.do("abs", reference))
    if not math.isfinite(leading) or leading <= 0:
        raise ValueError("The unit cell has no nonzero normalization eigenvalue.")
    ratio = float(ar.do("max", ar.do("abs", values))) / leading
    screening_tol = 4 * action.eps + ratio * float(ar.do("max", residuals)) + float(reference_residual)
    if ratio > 1 + screening_tol:
        raise RuntimeError("Sector eigenvalue exceeds the neutral normalization eigenvalue.")
    return MpsTransferSpectrum(
        eigenvalues=values, residuals=residuals, unit_cell_size=action.cell_size,
        sector=action.sector, solver=solver, projected_backend=projected_backend,
        reference_eigenvalue=reference, sites=sites, interpretation=interpretation,
        canonical_form=canonical_form,
        reference_residual=reference_residual, krylov_dimension=ncv,
        is_neutral=is_neutral, degeneracy_tol=degeneracy_tol,
    )


def mps_correlation_length(
    unit_cell, i=None, *, cell_size=1, canonicalize="auto", allow_local=False, mode=None, sector=None,
    **solver_options,
):
    """Return a transfer correlation length in lattice sites as a Python float.

    For dense or neutral-sector input, use the two largest magnitudes and
    ``-cell_size / log(abs(lambda1 / lambda0))``. For a nonneutral Symmray
    sector, compare its leading mode with the neutral normalization mode.
    This is a spectral length; an observable must couple to the mode to show
    that decay. Only the selected sector is searched, not all symmetry sectors.

    ``i`` selects the starting site in an MPS and ``cell_size`` selects the
    window. Open MPS input is left-canonicalized on a private copy by default;
    canonicalize='right' selects the opposite direction. This is a bulk
    estimate assuming approximately repeating tensors, as in the spectrum API.
    Supplied-gauge partial cells require canonicalize=None, allow_local=True.
    ``mode`` is the zero-based index in that sector's magnitude-ordered
    spectrum. It defaults to 1 for neutral/dense input and 0 for charged input.
    Degenerate modes are retained: a GHZ mode 1 has infinite length.

    Product states and unresolved zero subleading eigenvalues give zero;
    numerically peripheral modes give infinity, unresolved small gaps NaN. Use
    ``mps_transfer_spectrum(..., k=...)`` for several modes, their gaps,
    momenta, and lengths. ``solver_options`` accepts spectrum controls except
    k. This is a forward-only diagnostic returning a Python float.
    """
    if "k" in solver_options:
        raise TypeError("mps_correlation_length chooses k; use mps_transfer_spectrum for k.")
    # Materialize a generator once because charged queries also solve neutral.
    arrays, _, _, _ = _extract_cell(unit_cell, i, cell_size, allow_local, canonicalize)
    neutral = None
    if ar.infer_backend(arrays[0]) == "symmray":
        neutral = arrays[0].symmetry.combine()
    default_mode = mode is None
    if default_mode:
        mode = 0 if sector is not None and sector != neutral else 1
    elif isinstance(mode, bool) or not isinstance(mode, Integral) or mode < 0:
        raise ValueError("mode must be a nonnegative integer.")
    result = mps_transfer_spectrum(arrays, k=mode + 1, sector=sector, **solver_options)
    if mode >= len(result.eigenvalues):
        if default_mode and result.is_neutral and len(result.eigenvalues) == 1:
            return 0.0
        raise ValueError(f"mode={mode} is outside the transfer sector dimension.")
    return float(result.correlation_lengths[mode])
