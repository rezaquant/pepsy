"""Transfer spectra against small explicit channels and backend invariants."""

import math
from dataclasses import replace
from functools import wraps

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.tensors import mps_correlation_length, mps_transfer_spectrum

pytestmark = [pytest.mark.mps, pytest.mark.tensors]


def _pauli_cell():
    paulis = np.array([
        [[1, 0], [0, 1]], [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]], [[1, 0], [0, -1]],
    ])
    return np.transpose(np.sqrt([0.55, 0.2, 0.15, 0.1])[:, None, None] * paulis, (1, 2, 0))


def _dense_transfer(arrays):
    result = np.eye(arrays[0].shape[0] ** 2, dtype=complex)
    for a in arrays:
        result = result @ sum(np.kron(a[:, :, s], a[:, :, s].conj()) for s in range(a.shape[2]))
    return result


@pytest.mark.smoke
def test_known_channel_length_and_unit_cell_scale():
    a = _pauli_cell()
    spectrum = mps_transfer_spectrum([a], k=4)
    np.testing.assert_allclose(spectrum.eigenvalues, [1, 0.5, 0.4, 0.3], atol=1e-13)
    assert max(spectrum.residuals) < 1e-12
    np.testing.assert_allclose(spectrum.normalized_eigenvalues, [1, 0.5, 0.4, 0.3], atol=1e-13)
    np.testing.assert_allclose(spectrum.gaps, -np.log([1, 0.5, 0.4, 0.3]), atol=1e-13)
    assert spectrum.leading_degeneracy == 1
    expected = -1 / np.log(0.5)
    assert mps_correlation_length(a) == pytest.approx(expected)
    assert mps_correlation_length([a, a]) == pytest.approx(expected)
    assert mps_correlation_length([7 * a, 0.3 * a]) == pytest.approx(expected)
    assert mps_correlation_length(a, mode=2) == pytest.approx(-1 / np.log(0.4))


def test_cyclic_mps_gauge_and_metadata_are_preserved():
    rng = np.random.default_rng(72)
    arrays = [rng.normal(size=(3, 3, 2)) + 1j * rng.normal(size=(3, 3, 2)) for _ in range(3)]
    p = qtn.MatrixProductState(arrays, shape="lrp", site_ind_id="phys{}", site_tag_id="SITE{}")
    p.exponent = 42
    p[1].modify(left_inds=(p.site_ind(1),))
    p[2].transpose_(*reversed(p[2].inds))
    before = [(t.data, t.inds, tuple(t.tags), t.left_inds) for t in p.tensors]
    expected = np.linalg.eigvals(_dense_transfer(arrays))
    expected = expected[np.argsort(-abs(expected))][:2]
    original = mps_transfer_spectrum(p)
    np.testing.assert_allclose(original.eigenvalues, expected, rtol=1e-9, atol=1e-9)
    assert original.solver == "scipy"
    assert p.exponent == 42
    for t, (data, inds, tags, left_inds) in zip(p.tensors, before):
        assert t.data is data
        assert (t.inds, tuple(t.tags), t.left_inds) == (inds, tags, left_inds)
    gauges = [np.eye(3) + 0.1 * rng.normal(size=(3, 3)) for _ in arrays]
    transformed = [
        np.einsum("ij,jks,kl->ils", np.linalg.inv(gauges[i - 1]), a, gauges[i])
        for i, a in enumerate(arrays)
    ]
    np.testing.assert_allclose(
        mps_transfer_spectrum(transformed).eigenvalues, expected, rtol=1e-9, atol=1e-9
    )
    full_wrapped = mps_transfer_spectrum(p, 2, cell_size=3)
    assert full_wrapped.sites == (2, 0, 1)
    assert full_wrapped.interpretation == "unit_cell"
    np.testing.assert_allclose(abs(full_wrapped.eigenvalues), abs(original.eigenvalues))
    wrapped = mps_transfer_spectrum(p, 2, cell_size=2, allow_local=True)
    assert wrapped.sites == (2, 0)
    assert wrapped.interpretation == "local_repeated_cell"
    expected = np.linalg.eigvals(_dense_transfer([arrays[2], arrays[0]]))
    np.testing.assert_allclose(abs(wrapped.eigenvalues), np.sort(abs(expected))[-2:][::-1])


def test_nonhermitian_complex_modes_and_arnoldi_restarts():
    rng = np.random.default_rng(11)
    a = (rng.normal(size=(6, 6, 3)) + 1j * rng.normal(size=(6, 6, 3))) / 6
    expected = np.linalg.eigvals(_dense_transfer([a]))
    expected = expected[np.argsort(-abs(expected))]
    result = mps_transfer_spectrum(a, k=4, solver="arnoldi", ncv=18, maxiter=100, tol=1e-9)
    # Conjugate pairs can be returned in either order when their moduli agree.
    for value in result.eigenvalues:
        assert min(abs(value - expected)) < 1e-8
    np.testing.assert_allclose(abs(result.eigenvalues), abs(expected[:4]), rtol=1e-8)
    assert np.max(result.residuals) < 1e-9
    assert result.krylov_dimension == 18
    assert np.max(abs(result.eigenvalues.imag)) > 0.01
    np.testing.assert_allclose(result.momenta, np.angle(result.normalized_eigenvalues))
    with pytest.raises(RuntimeError, match="did not converge"):
        mps_transfer_spectrum(a, solver="arnoldi", ncv=4, maxiter=1, tol=1e-13)


def test_arpack_uses_cpu_precision_iteration_budget_and_local_rng(monkeypatch):
    import scipy.sparse.linalg as spla

    original = spla.eigs
    calls = []

    @wraps(original)
    def observed(operator, **kwargs):
        calls.append(kwargs)
        assert operator.dtype == np.dtype("complex64")
        return original(operator, **kwargs)

    monkeypatch.setattr(spla, "eigs", observed)
    result = mps_transfer_spectrum(_pauli_cell().astype(np.complex64), k=2, seed=17)
    assert result.solver == "scipy"
    assert result.eigenvalues.dtype == np.dtype("complex64")
    assert calls[0]["maxiter"] == 40
    assert calls[0]["which"] == "LM"
    assert calls[0]["v0"].dtype == np.dtype("complex64")
    # rng was added to public SciPy in 1.17; old installations use seeded v0.
    from inspect import signature
    if "rng" in signature(original).parameters:
        assert isinstance(calls[0]["rng"], np.random.Generator)


def test_arpack_failure_does_not_return_partial_spectrum():
    from scipy.sparse.linalg import ArpackNoConvergence

    rng = np.random.default_rng(51)
    a = (rng.normal(size=(10, 10, 3)) + 1j * rng.normal(size=(10, 10, 3))) / 10
    with pytest.raises(RuntimeError, match="ARPACK did not converge.*No partial spectrum") as caught:
        mps_transfer_spectrum(a, k=6, solver="scipy", ncv=8, maxiter=1, tol=1e-13)
    assert isinstance(caught.value.__cause__, ArpackNoConvergence)


@pytest.mark.parametrize("solver", ["scipy", "arnoldi"])
def test_nonnormal_transfer_matches_analytic_spectrum(solver):
    diagonal = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5])
    a = (np.diag(diagonal) + 0.3 * np.diag(np.ones(5), 1)).astype(complex)[:, :, None]
    # Triangular Kraus matrix: exact eigenvalues are all pairwise products.
    expected = np.sort(np.outer(diagonal, diagonal).ravel())[::-1][:6]
    result = mps_transfer_spectrum(a, k=6, solver=solver)
    np.testing.assert_allclose(np.sort(abs(result.eigenvalues))[::-1], expected, atol=2e-8)
    assert max(result.residuals) <= 1e-10


def test_product_zero_and_degenerate_peripheral_modes():
    product = np.array([[[1.0, 0.0]]])
    assert mps_correlation_length(product) == 0
    assert len(mps_transfer_spectrum(product).eigenvalues) == 1
    with pytest.raises(ValueError, match="normalization"):
        mps_correlation_length(product * 0)
    ghz = np.zeros((2, 2, 2))
    ghz[0, 0, 0] = ghz[1, 1, 1] = 1
    assert math.isinf(mps_correlation_length(ghz, solver="arnoldi"))
    identity_channel = np.eye(3)[:, :, None].astype(complex)
    result = mps_transfer_spectrum(identity_channel, k=3, solver="arnoldi")
    np.testing.assert_allclose(result.eigenvalues, 1, atol=1e-13)
    assert math.isinf(mps_correlation_length(identity_channel))


@pytest.mark.smoke
def test_ghz_site_extraction_keeps_degenerate_modes_and_infinite_length():
    psi = qtn.MPS_ghz_state(8)
    result = mps_transfer_spectrum(psi, 3, k=4)
    assert result.sites == (3,)
    assert result.interpretation == "bulk_estimate"
    assert result.canonical_form == "left"
    np.testing.assert_allclose(result.normalized_eigenvalues, [1, 1, 0, 0], atol=1e-13)
    assert result.leading_degeneracy == 2
    np.testing.assert_array_equal(result.peripheral_mask, [True, True, False, False])
    np.testing.assert_array_equal(result.gaps, [0, 0, np.inf, np.inf])
    np.testing.assert_array_equal(result.correlation_lengths, [np.inf, np.inf, 0, 0])
    assert np.all(np.isnan(result.momenta[2:]))
    assert math.isinf(mps_correlation_length(psi, 3))
    assert math.isinf(mps_correlation_length(psi, 3, canonicalize="right"))
    assert math.isinf(mps_correlation_length(psi, 2, cell_size=2))
    assert mps_correlation_length(psi, 3, mode=2) == 0


def test_canonical_gauge_freedom_and_physical_ghz_correlations():
    psi = qtn.MPS_ghz_state(8)
    psi.left_canonize()
    gauged = psi.copy()
    u = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for site in (3, 4):
        inds = (gauged.bond(site - 1, site), gauged.bond(site, site + 1), gauged.site_ind(site))
        a = gauged[site].transpose(*inds).data
        b = np.einsum("lrs,rt->lts", a, u) if site == 3 else np.einsum("tl,lrs->trs", u.T, a)
        gauged[site].modify(
            data=b.transpose([inds.index(ix) for ix in gauged[site].inds]), left_inds=None,
        )
        # The bond unitary preserves the left-canonical isometry condition.
        np.testing.assert_allclose(np.einsum("lrs,lts->rt", b.conj(), b), np.eye(2), atol=1e-14)
    v = np.asarray(psi.to_dense()).ravel()
    w = np.asarray(gauged.to_dense()).ravel()
    np.testing.assert_allclose(v, w, atol=1e-14)
    bits = np.arange(v.size)
    zi, zj = 1 - 2 * ((bits >> 6) & 1), 1 - 2 * ((bits >> 1) & 1)
    for state in (v, w):
        probabilities = abs(state) ** 2
        connected_z = probabilities @ (zi * zj) - (probabilities @ zi) * (probabilities @ zj)
        assert connected_z == pytest.approx(1)
        connected_x = np.vdot(state, state[bits ^ (1 << 6) ^ (1 << 1)])
        connected_x -= np.vdot(state, state[bits ^ (1 << 6)]) * np.vdot(state, state[bits ^ (1 << 1)])
        assert connected_x == pytest.approx(0)
    for state in (psi, gauged):
        with pytest.raises(ValueError, match="gauge dependent"):
            mps_transfer_spectrum(state, 3, canonicalize=None)
        with pytest.raises(ValueError, match="gauge dependent"):
            mps_correlation_length(state, 3, canonicalize=None)
    original = mps_transfer_spectrum(psi, 3, k=4, allow_local=True, canonicalize=None)
    changed = mps_transfer_spectrum(gauged, 3, k=4, allow_local=True, canonicalize=None)
    np.testing.assert_allclose(original.normalized_eigenvalues, [1, 1, 0, 0], atol=1e-13)
    np.testing.assert_allclose(changed.normalized_eigenvalues, [1, 0, 0, 0], atol=2e-8)


@pytest.mark.parametrize("solver", ["scipy", "arnoldi"])
def test_degenerate_leading_space_and_several_finite_decay_modes(solver):
    # Two GHZ-like sectors, each with a nontrivial Pauli transfer channel.
    a = np.zeros((4, 4, 8), dtype=complex)
    a[:2, :2, :4] = _pauli_cell()
    a[2:, 2:, 4:] = _pauli_cell()
    result = mps_transfer_spectrum(a, k=6, solver=solver)
    np.testing.assert_allclose(result.normalized_eigenvalues, [1, 1, 0.5, 0.5, 0.4, 0.4], atol=1e-12)
    assert result.leading_degeneracy == 2
    assert math.isinf(mps_correlation_length(a, solver=solver))
    assert mps_correlation_length(a, mode=2, solver=solver) == pytest.approx(-1 / np.log(0.5))
    np.testing.assert_allclose(result.correlation_lengths[2:], -1 / np.log([0.5, 0.5, 0.4, 0.4]))


def test_nonuniform_open_mps_sites_and_window_follow_supplied_tensors():
    a = _pauli_cell()
    b = a * np.sqrt(np.array([0.85, 0.05, 0.05, 0.05]) / [0.55, 0.2, 0.15, 0.1])
    psi = qtn.MatrixProductState(
        [np.ones((2, 4)), a, b, a, np.ones((2, 4))],
        shape="lrp", site_ind_id="phys{}", site_tag_id="SITE{}",
    )
    psi.exponent = 17
    psi[2].transpose_(*reversed(psi[2].inds))
    psi[2].modify(left_inds=(psi.site_ind(2),))
    before = [(t.data, t.inds, tuple(t.tags), t.left_inds) for t in psi.tensors]
    raw = {"allow_local": True, "canonicalize": None}
    assert mps_correlation_length(psi, 1, **raw) == pytest.approx(-1 / np.log(0.5))
    assert mps_correlation_length(psi, i=2, **raw) == pytest.approx(-1 / np.log(0.8))
    window = mps_transfer_spectrum(psi, 1, cell_size=2, k=4, **raw)
    assert window.sites == (1, 2)
    assert window.unit_cell_size == 2
    np.testing.assert_allclose(window.normalized_eigenvalues, [1, 0.4, 0.32, 0.24], atol=1e-13)
    assert mps_correlation_length(psi, 1, cell_size=2, **raw) == pytest.approx(-2 / np.log(0.4))
    for t, (data, inds, tags, left_inds) in zip(psi.tensors, before):
        assert t.data is data
        assert (t.inds, tuple(t.tags), t.left_inds) == (inds, tags, left_inds)
    assert psi.exponent == 17


def test_rectangular_site_requires_window_with_matching_boundaries():
    rng = np.random.default_rng(14)
    a, b = rng.normal(size=(2, 3, 2)), rng.normal(size=(3, 2, 2))
    psi = qtn.MatrixProductState([np.ones((2, 2)), a, b, np.ones((2, 2))], shape="lrp")
    with pytest.raises(ValueError, match="boundary dimensions"):
        mps_transfer_spectrum(psi, 1, allow_local=True, canonicalize=None)
    result = mps_transfer_spectrum(psi, 1, cell_size=2, k=4, allow_local=True, canonicalize=None)
    expected = np.linalg.eigvals(_dense_transfer([a, b]))
    np.testing.assert_allclose(abs(result.eigenvalues), np.sort(abs(expected))[::-1], atol=1e-12)
    for options in ({"i": 0}, {"i": 3}, {"i": -1}, {"i": 1.5}, {"i": 1, "cell_size": 4}):
        with pytest.raises(ValueError):
            mps_transfer_spectrum(psi, allow_local=True, **options)
    with pytest.raises(ValueError, match="not an array cell"):
        mps_transfer_spectrum([a, b], i=0)
    with pytest.raises(ValueError, match="outside"):
        mps_correlation_length(_pauli_cell(), mode=4)


def test_degeneracy_tolerance_is_separate_from_residual_tolerance():
    # Slight mixing lifts the GHZ normalization degeneracy without removing it
    # from the returned spectrum. Clustering must not turn a finite length infinite.
    a = np.zeros((2, 2, 4), dtype=complex)
    a[0, 0, 0] = a[1, 1, 1] = np.sqrt(1 - 1e-5)
    a[0, 1, 2] = a[1, 0, 3] = np.sqrt(1e-5)
    loose = mps_transfer_spectrum(a, k=4, degeneracy_tol=1e-4)
    tight = mps_transfer_spectrum(a, k=4, degeneracy_tol=1e-8)
    assert loose.leading_degeneracy == 2
    assert tight.leading_degeneracy == 1
    assert loose.correlation_lengths[1] == pytest.approx(tight.correlation_lengths[1])
    assert tight.correlation_lengths[1] == pytest.approx(-1 / np.log(1 - 2e-5), rel=1e-9)


def test_single_precision_small_gap_is_finite_and_poor_resolution_is_explicit():
    a = np.zeros((2, 2, 4), dtype=np.complex64)
    a[0, 0, 0] = a[1, 1, 1] = np.sqrt(1 - 1e-6)
    a[0, 1, 2] = a[1, 0, 3] = np.sqrt(1e-6)
    result = mps_transfer_spectrum(a, k=4)
    # Use the actual rounded input channel as the high-precision reference.
    reference = np.linalg.eigvals(_dense_transfer([a.astype(complex)]))
    reference = np.sort(abs(reference))[::-1]
    expected = -1 / np.log(reference[1] / reference[0])
    assert np.isfinite(result.correlation_lengths[1])
    assert result.correlation_lengths[1] == pytest.approx(expected, rel=0.2)
    assert not result.unresolved_mask[1]
    # An eigenpair can pass a loose residual tolerance without resolving its gap.
    uncertain = replace(result, residuals=np.full(4, 1e-4), reference_residual=1e-4)
    assert np.isinf(uncertain.correlation_lengths[0])
    assert uncertain.unresolved_mask[1]
    assert np.isnan(uncertain.gaps[1])
    assert np.isnan(uncertain.correlation_lengths[1])


def test_periodic_modes_keep_their_phase_relative_to_positive_perron_root():
    a = np.zeros((2, 2, 2), dtype=complex)
    a[0, 1, 0] = a[1, 0, 1] = 1
    result = mps_transfer_spectrum(a, k=4)
    assert float(result.reference_eigenvalue) == pytest.approx(1)
    assert result.leading_degeneracy == 1
    assert np.sum(result.peripheral_mask) == 2
    for value, momentum, length in zip(
        result.normalized_eigenvalues[:2], result.momenta[:2], result.correlation_lengths[:2]
    ):
        assert abs(momentum) == pytest.approx(0 if value.real > 0 else np.pi, abs=1e-12)
        assert math.isinf(length)


def _backend_array(a, backend):
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        device = "cuda" if backend == "torch_cuda" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        return torch.tensor(a, device=device, dtype=torch.complex64, requires_grad=True)
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CuPy CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy CUDA unavailable")
    return cp.asarray(a)


@pytest.mark.parametrize("direction,backend", [
    ("left", None), ("right", None), ("left", "torch"),
    ("right", "torch_cuda"), ("left", "cupy"),
])
def test_canonical_bulk_modes_and_input_metadata_are_preserved(monkeypatch, direction, backend):
    a = _pauli_cell()
    arrays = [np.eye(2, dtype=complex), *[a.copy() for _ in range(6)], np.eye(2, dtype=complex)]
    # Nonunitary positive gauges change the supplied tensors. A full QR sweep
    # recovers this analytic channel in either direction.
    gauges = [np.diag([1 + j / 10, 0.8 + j / 20]) for j in range(7)]
    arrays[0] = gauges[0].T @ arrays[0]
    for j in range(1, 7):
        arrays[j] = np.einsum("ab,bcs,cd->ads", np.linalg.inv(gauges[j - 1]), arrays[j], gauges[j])
    arrays[-1] = np.linalg.inv(gauges[-1]) @ arrays[-1]
    if backend:
        arrays = [_backend_array(a, backend) for a in arrays]
    psi = qtn.MatrixProductState(arrays, shape="lrp", site_ind_id="phys{}", site_tag_id="SITE{}")
    psi.exponent = 17
    psi[3].transpose_(*reversed(psi[3].inds))
    # Deliberately stale metadata must not prevent the diagnostic sweep.
    psi[3].modify(left_inds=tuple(ix for ix in psi[3].inds if ix != psi.bond(3, 4)))
    before = [(t.data, ar.do("copy", t.data), t.inds, tuple(t.tags), t.left_inds) for t in psi.tensors]

    def forbidden(*args, **kwargs):
        raise AssertionError("Canonicalization must not move data to the host")

    monkeypatch.setattr(ar, "to_numpy", forbidden)
    if backend and backend.startswith("torch"):
        import torch
        monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
        monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    elif backend == "cupy":
        import cupy as cp
        asnumpy = cp.asnumpy

        def projected_only(h):
            assert h.shape == (4, 4)
            return asnumpy(h)

        monkeypatch.setattr(cp, "asnumpy", projected_only)
    result = mps_transfer_spectrum(psi, 3, k=4, canonicalize=direction)
    assert result.interpretation == "bulk_estimate"
    assert result.canonical_form == direction
    assert result.sites == (3,)
    for value, expected in zip(result.normalized_eigenvalues, [1, 0.5, 0.4, 0.3]):
        assert complex(value) == pytest.approx(expected, abs=2e-6)
    assert mps_correlation_length(psi, 3, canonicalize=direction, mode=2) == pytest.approx(-1 / np.log(0.4), rel=2e-5)
    window = mps_transfer_spectrum(psi, 3, cell_size=2, k=4, canonicalize=direction)
    for length, expected in zip(window.correlation_lengths[1:], -1 / np.log([0.5, 0.4, 0.3])):
        assert float(length) == pytest.approx(expected, rel=2e-5)
    for tensor, (data, values, inds, tags, left_inds) in zip(psi.tensors, before):
        assert tensor.data is data
        assert bool(ar.do("all", tensor.data == values))
        assert (tensor.inds, tuple(tensor.tags), tensor.left_inds) == (inds, tags, left_inds)
    assert psi.exponent == 17
    if backend:
        assert result.eigenvalues.device == arrays[0].device
        assert result.eigenvalues.dtype == arrays[0].dtype
    if backend and backend.startswith("torch"):
        assert all(t.data.requires_grad for t in psi.tensors)
        assert not result.eigenvalues.requires_grad


def test_canonicalization_options_validate_input_contract():
    a = _pauli_cell()
    cyclic = qtn.MatrixProductState([a, a, a], shape="lrp")
    for value in ("left", "right"):
        with pytest.raises(ValueError, match="open MPS"):
            mps_transfer_spectrum(cyclic, 1, canonicalize=value)
        with pytest.raises(ValueError, match="open Quimb MPS"):
            mps_transfer_spectrum(a, canonicalize=value)
    with pytest.raises(ValueError, match="canonicalize"):
        mps_transfer_spectrum(a, canonicalize="middle")
    assert mps_transfer_spectrum(cyclic).canonical_form is None


def test_canonicalization_rejects_mixed_precision_and_nonfinite_input_before_qr(monkeypatch):
    psi = qtn.MPS_rand_state(8, 4, dtype="float64", seed=3)
    psi[0].modify(data=psi[0].data.astype("float32"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid input must be rejected before canonicalization")

    monkeypatch.setattr(type(psi), "left_canonize", forbidden)
    with pytest.raises(TypeError, match="dtype"):
        mps_transfer_spectrum(psi, 3)
    psi[0].modify(data=psi[0].data.astype("float64") * np.nan)
    with pytest.raises(ValueError, match="finite"):
        mps_transfer_spectrum(psi, 3)
    for solver in ("scipy", "arnoldi"):
        nonfinite = _pauli_cell()
        nonfinite[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            mps_transfer_spectrum(nonfinite, solver=solver)


@pytest.mark.parametrize("backend", [None, "torch_cuda", "cupy"])
def test_arnoldi_grows_within_budget_on_previously_stagnating_channel(backend):
    rng = np.random.default_rng(51)
    a = (rng.normal(size=(10, 10, 3)) + 1j * rng.normal(size=(10, 10, 3))) / 10
    expected = np.sort(abs(np.linalg.eigvals(_dense_transfer([a]))))[::-1][:6]
    if backend:
        a = _backend_array(a, backend)
    result = mps_transfer_spectrum(a, k=6, solver="arnoldi")
    assert 32 < result.krylov_dimension <= 100
    actual = np.array([abs(complex(value)) for value in result.eigenvalues])
    np.testing.assert_allclose(actual, expected, rtol=2e-5)
    assert float(ar.do("max", result.residuals)) <= (1e-5 if backend == "torch_cuda" else 1e-10)
    if backend is None:
        with pytest.raises(RuntimeError, match="ncv=32"):
            mps_transfer_spectrum(a, k=6, solver="arnoldi", ncv=32)
        with pytest.raises(ValueError, match="max_ncv"):
            mps_transfer_spectrum(a, k=6, ncv=32, max_ncv=16)


@pytest.mark.parametrize("backend", ["torch", "torch_cuda", "cupy"])
def test_device_backend_precision_and_no_host_transfer(monkeypatch, backend):
    a = _backend_array(_pauli_cell(), backend)
    before = a.clone() if backend.startswith("torch") else a.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected conversion to NumPy")

    # Spectral work must not feed whole arrays through a host bridge.
    monkeypatch.setattr(ar, "to_numpy", forbidden)
    if backend.startswith("torch"):
        import torch
        monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
        monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    else:
        import cupy as cp
        from cupyx import cusolver
        asnumpy = cp.asnumpy

        def projected_only(h):
            assert h.shape == (4, 4)
            return asnumpy(h)

        # Exercise the compatibility path regardless of the machine's CUDA version.
        monkeypatch.setattr(cusolver, "check_availability", lambda name: False)
        monkeypatch.setattr(cp, "asnumpy", projected_only)
    psi = qtn.MatrixProductState([a, a, a], shape="lrp")
    result = mps_transfer_spectrum(psi, 1, k=4, allow_local=True)
    assert result.sites == (1,)
    assert result.solver == "arnoldi"
    assert ar.infer_backend(result.eigenvalues) == ar.infer_backend(a)
    assert result.eigenvalues.dtype == a.dtype
    assert result.eigenvalues.device == a.device
    assert result.gaps.device == a.device
    assert result.correlation_lengths.device == a.device
    assert result.momenta.device == a.device
    assert math.isinf(float(result.correlation_lengths[0]))
    for value, expected in zip(result.eigenvalues, [1, 0.5, 0.4, 0.3]):
        assert complex(value) == pytest.approx(expected, abs=2e-6)
    assert mps_correlation_length(a) == pytest.approx(-1 / np.log(0.5), rel=2e-5)
    assert bool(ar.do("all", a == before))
    if backend.startswith("torch"):
        assert a.requires_grad
        assert not result.eigenvalues.requires_grad
        assert result.projected_backend == "torch"
    else:
        assert result.projected_backend == "numpy"
        with pytest.raises(NotImplementedError, match="general eigensolver"):
            mps_transfer_spectrum(a, projected_solver="native")
    with pytest.raises(ValueError, match="requires NumPy"):
        mps_transfer_spectrum(a, solver="scipy")


def _symmetric_cell(backend=None):
    sr = pytest.importorskip("symmray")
    bond = sr.BlockIndex({0: 1, 1: 1})
    phys = sr.BlockIndex({-1: 1, 0: 2, 1: 1})
    blocks = {
        (0, 0, 0): np.array([np.sqrt(0.75), np.sqrt(0.05)]).reshape(1, 1, 2),
        (1, 1, 0): np.array([np.sqrt(0.75), -np.sqrt(0.05)]).reshape(1, 1, 2),
        (1, 0, -1): np.full((1, 1, 1), np.sqrt(0.2)),
        (0, 1, 1): np.full((1, 1, 1), np.sqrt(0.2)),
    }
    if backend:
        blocks = {q: _backend_array(b, backend) for q, b in blocks.items()}
    return sr.U1Array(indices=(bond, bond.conj(), phys), charge=0, blocks=blocks)


@pytest.mark.parametrize("direction,backend", [("left", None), ("right", "torch_cuda"), ("left", "cupy")])
def test_symmray_canonical_bulk_spectrum_stays_native(monkeypatch, direction, backend):
    a = _symmetric_cell(backend)
    left = type(a)(indices=a.indices[1:], charge=0, blocks={
        (qr, qp): ar.do("reshape", b, b.shape[1:])
        for (ql, qr, qp), b in a.blocks.items() if ql == 0
    })
    right = type(a)(indices=(a.indices[0], a.indices[2]), charge=0, blocks={
        (ql, qp): ar.do("reshape", b, (b.shape[0], b.shape[2]))
        for (ql, qr, qp), b in a.blocks.items() if qr == 0
    })
    psi = qtn.MatrixProductState([left, *[a for _ in range(6)], right], shape="lrp")
    before = [(t.data, t.data.blocks.copy(), t.inds, t.left_inds) for t in psi.tensors]

    def forbidden(*args, **kwargs):
        raise AssertionError("Canonical bulk diagnostics must stay native")

    monkeypatch.setattr(type(a), "to_dense", forbidden)
    result = mps_transfer_spectrum(psi, 3, k=2, canonicalize=direction)
    assert result.canonical_form == direction
    assert result.interpretation == "bulk_estimate"
    assert result.sector == 0
    assert bool(ar.do("all", ar.do("isfinite", result.eigenvalues)))
    assert float(ar.do("max", result.residuals)) < 1e-5
    charged = mps_transfer_spectrum(psi, 3, sector=1, canonicalize=direction)
    assert charged.canonical_form == direction
    assert float(charged.reference_eigenvalue) == pytest.approx(float(result.reference_eigenvalue), rel=2e-5)
    for tensor, (data, blocks, inds, left_inds) in zip(psi.tensors, before):
        assert tensor.data is data
        assert tensor.inds == inds
        assert tensor.left_inds == left_inds
        for q, block in blocks.items():
            assert tensor.data.blocks[q] is block
    block = next(iter(a.blocks.values()))
    assert ar.infer_backend(result.eigenvalues) == ar.infer_backend(block)
    if backend:
        assert result.eigenvalues.device == block.device
    if backend == "torch_cuda":
        assert not result.eigenvalues.requires_grad


@pytest.mark.parametrize("backend", [None, "torch_cuda", "cupy"])
def test_symmray_sector_lengths_without_densification(monkeypatch, backend):
    a = _symmetric_cell(backend)
    before = a.blocks.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Symmray transfer must not densify")

    monkeypatch.setattr(type(a), "to_dense", forbidden)
    psi = qtn.MatrixProductState([a, a, a], shape="lrp")
    neutral = mps_transfer_spectrum(psi, 1, allow_local=True)
    assert neutral.sites == (1,)
    assert neutral.sector == 0
    for value, expected in zip(neutral.eigenvalues, [1, 0.6]):
        assert complex(value) == pytest.approx(expected, abs=2e-6)
    for sector in (-1, 1):
        charged = mps_transfer_spectrum(a, sector=sector)
        assert complex(charged.eigenvalues[0]) == pytest.approx(0.7, abs=2e-6)
        assert complex(charged.reference_eigenvalue) == pytest.approx(1, abs=2e-6)
        assert float(charged.gaps[0]) == pytest.approx(-np.log(0.7), abs=2e-6)
        assert mps_correlation_length([a, a], sector=sector) == pytest.approx(-1 / np.log(0.7), rel=2e-5)
    assert mps_correlation_length(a) == pytest.approx(-1 / np.log(0.6), rel=2e-5)
    assert all(a.blocks[q] is b for q, b in before.items())
    if backend:
        block = next(iter(a.blocks.values()))
        assert neutral.eigenvalues.device == block.device
        dtype = ar.get_dtype_name(block)
        expected_dtype = "complex64" if dtype in {"float32", "complex64"} else "complex128"
        assert ar.get_dtype_name(neutral.eigenvalues) == expected_dtype
    with pytest.raises(ValueError, match="absent"):
        mps_transfer_spectrum(a, sector=3)
    # Missing output blocks must be restored as structural zeros when packed.
    sparse = a.copy_with(blocks={(0, 0, 0): a.blocks[(0, 0, 0)]})
    assert float(ar.do("abs", mps_transfer_spectrum(sparse).eigenvalues[0])) == pytest.approx(0.8, abs=2e-6)


def test_invalid_cells_and_explicit_capability_limits():
    a = _pauli_cell()
    with pytest.raises(ValueError, match="open MPS"):
        mps_correlation_length(qtn.MPS_rand_state(4, 2))
    with pytest.raises(ValueError, match="empty"):
        mps_transfer_spectrum([])
    with pytest.raises(ValueError, match="does not close"):
        mps_transfer_spectrum([np.ones((2, 3, 2))])
    with pytest.raises(TypeError, match="dtype"):
        mps_transfer_spectrum([a, a.astype("complex64")])
    with pytest.raises(ValueError, match="sector"):
        mps_transfer_spectrum(a, sector=0)
    for options in ({"k": 0}, {"ncv": 2}, {"tol": -1}, {"maxiter": 0}, {"seed": -1}):
        with pytest.raises(ValueError):
            mps_transfer_spectrum(a, **options)
    sr = pytest.importorskip("symmray")
    boson = _symmetric_cell()
    fermion = sr.U1FermionicArray(indices=boson.indices, charge=0, blocks=boson.blocks)
    with pytest.raises(NotImplementedError, match="Fermionic"):
        mps_transfer_spectrum(fermion)


@pytest.mark.parametrize("solver", ["scipy", "arnoldi"])
def test_seeded_solver_does_not_change_numpy_global_rng(solver):
    before = np.random.get_state()
    first = mps_transfer_spectrum(_pauli_cell(), solver=solver, seed=17)
    second = mps_transfer_spectrum(_pauli_cell(), solver=solver, seed=17)
    after = np.random.get_state()
    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_symmray_multidimensional_blocks_match_dense_sector_reference():
    sr = pytest.importorskip("symmray")
    bond = sr.BlockIndex({0: 2, 1: 1})
    physical = sr.BlockIndex({-1: 1, 0: 2, 1: 1})
    a = sr.utils.get_rand("U1", (bond, bond.conj(), physical), charge=0, seed=13)
    dense = _dense_transfer([a.to_dense()])
    charges = [0, 0, 1]
    for sector in (0, 1):
        positions = [i * 3 + j for i in range(3) for j in range(3) if charges[i] - charges[j] == sector]
        expected = np.linalg.eigvals(dense[np.ix_(positions, positions)])
        expected = expected[np.argsort(-abs(expected))]
        result = mps_transfer_spectrum(a, sector=sector, k=len(positions))
        np.testing.assert_allclose(abs(result.eigenvalues), abs(expected), rtol=1e-10)
        for value in result.eigenvalues:
            assert min(abs(value - expected)) < 1e-10
        if sector == 0:
            partial = mps_transfer_spectrum(a, k=2, solver="scipy")
            assert partial.solver == "scipy"
            np.testing.assert_allclose(abs(partial.eigenvalues), abs(expected[:2]), rtol=1e-10)
