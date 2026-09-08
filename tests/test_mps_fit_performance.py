"""FIT validation policy and active-window ownership regressions."""

import warnings

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py
import pepsy.fitting.local as fitting


def _state():
    return qtn.MPS_rand_state(8, 2, dtype="complex128", seed=714)


@pytest.mark.parametrize("mode", ["dmrg2", "mix"])
def test_finite_scan_is_opt_in_even_with_timing(monkeypatch, mode):
    """Timing must not turn array scans on; enabling them warns and scans."""
    original = py.FIT._sweep_diagnostics_to_host
    checks = []

    def observe(*args, **kwargs):
        checks.append(kwargs["check_finite"])
        return original(*args, **kwargs)

    monkeypatch.setattr(py.FIT, "_sweep_diagnostics_to_host", staticmethod(observe))
    p = _state()
    gates = [(qu.CNOT(), (2, 4))]
    default = py.MpsOptimizer(p, gates, chi=2, mode=mode)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        default.run(n_iter=3, timing=True)
    assert checks and not any(checks)
    assert not any("finite_check" in str(w.message) for w in recorded)

    checks.clear()
    checked = py.MpsOptimizer(p, gates, chi=2, mode=mode)
    with pytest.warns(RuntimeWarning, match="finite_check is enabled"):
        checked.run(n_iter=3, finite_check=True)
    assert checks and all(checks)
    np.testing.assert_allclose(default.p.to_dense(), checked.p.to_dense(), atol=1e-11)
    assert default.info_c == checked.info_c


def test_measurement_and_shots_forward_finite_policy(monkeypatch):
    """Multi-site projector FIT and shot workers honor the same switch."""
    original = py.FIT.run_gate
    policies = []

    def observe(self, *args, **kwargs):
        policies.append(kwargs["finite_check"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", observe)
    opt = py.MpsOptimizer(
        _state(), [("measure", "ZZ", (2, 4), 1)], chi=2, mode="dmrg2"
    )
    opt.run(n_iter=3)
    assert policies and not any(policies)
    policies.clear()
    with pytest.warns(RuntimeWarning, match="finite_check is enabled"):
        opt.run(shots=2, workers=1, strategy="independent", finite_check=True,
                run_kwargs={"n_iter": 3}, seed=4)
    assert policies and all(policies)


def test_fit_finite_scan_detects_bad_array_and_warns(monkeypatch):
    """The explicitly requested scan must still reject invalid tensor data."""
    p = _state()
    fit = py.FIT(p.copy(), p=p, range_int=[2, 4])
    monkeypatch.setattr(fitting, "_iter_backend_arrays", lambda _t: (np.array([np.nan]),))
    with pytest.warns(RuntimeWarning, match="finite_check is enabled"):
        with pytest.raises(FloatingPointError, match="non-finite tensor data"):
            fit.run_gate(n_iter=1, finite_check=True)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_fit_window_copy_isolates_active_data_and_preserves_metadata(backend):
    """Copies allocate only active arrays, preserve gradients, and isolate edits."""
    p = _state()
    leaf = None
    if backend == "torch":
        torch = pytest.importorskip("torch")
        p.apply_to_arrays(lambda a: torch.as_tensor(a))
        leaf = p[3].data.clone().requires_grad_()
        p[3].modify(data=leaf * 1.0)  # non-leaf data must remain differentiable
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        p.apply_to_arrays(lambda a: jax.numpy.asarray(a, dtype="complex64"))
    opt = py.MpsOptimizer(p, [], chi=2, mode="dmrg2")
    p = opt.p
    copied = opt._copy_fit_window_state(p, (2, 4))
    for site in range(p.L):
        assert copied[site] is not p[site]
        assert copied[site].left_inds == p[site].left_inds
        assert copied[site].inds == p[site].inds
        if 2 <= site <= 4:
            assert copied[site].data is not p[site].data
        else:
            assert copied[site].data is p[site].data
    if backend == "torch":
        copied[3].data.abs().square().sum().backward()
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
        with torch.no_grad():
            copied[3].data.fill_(0)
        assert torch.any(p[3].data != 0)
    elif backend == "jax":
        copied[3].modify(data=jax.numpy.zeros_like(copied[3].data))
        assert np.any(np.asarray(p[3].data) != 0)
    else:
        copied[3].data[...] = 0
        assert np.any(p[3].data != 0)


def test_window_copies_match_full_copies_for_batched_and_submpo_fit(monkeypatch):
    """Gate batches, explicit MPOs and measurement preserve numerical results."""
    p = _state()
    before = p.to_dense().copy()
    mpo = qtn.MPO_product_operator([qu.pauli("X"), qu.pauli("Z")],
                                   sites=(2, 4), L=8)
    gates = [(qu.CNOT(), (2, 4)), (qu.CNOT(), (3, 5)),
             ("submpo", mpo, (2, 4)), ("measure", "ZZ", (2, 4), 1)]
    opt = py.MpsOptimizer(p, gates, chi=2, mode="dmrg2")
    opt.run(n_iter=3, k_2q_batch=2)
    monkeypatch.setattr(py.MpsOptimizer, "_copy_fit_window_state",
                        lambda self, state, where: state.copy(deep=True))
    baseline = py.MpsOptimizer(p, gates, chi=2, mode="dmrg2")
    baseline.run(n_iter=3, k_2q_batch=2)
    np.testing.assert_allclose(opt.p.to_dense(), baseline.p.to_dense(), atol=1e-11)
    np.testing.assert_allclose(p.to_dense(), before, atol=1e-13)
    assert opt.info_c == baseline.info_c
    for actual, expected in zip(opt.measurements, baseline.measurements):
        assert actual[:3] == expected[:3]
        np.testing.assert_allclose(actual[3], expected[3], atol=1e-13)


def test_failed_inplace_fit_restores_uncorrupted_mpo_fallback(monkeypatch):
    """An active-array write before failure must not contaminate rollback."""
    p = _state()
    before = p.to_dense().copy()
    gates = [(qu.CNOT(), (2, 4))]
    reference = py.MpsOptimizer(p, gates, chi=2, mode="direct")
    reference.run(cutoff=0.0)

    def fail(self, **kwargs):
        self.p[3].data[...] = np.nan
        raise RuntimeError("injected active-window failure")

    monkeypatch.setattr(py.FIT, "run_gate", fail)
    opt = py.MpsOptimizer(p, gates, chi=2, mode="dmrg2")
    opt.run(cutoff=0.0, fit_init_strategy="direct")
    np.testing.assert_allclose(opt.p.to_dense(), reference.p.to_dense(), atol=1e-11)
    np.testing.assert_allclose(p.to_dense(), before, atol=1e-13)
    assert opt.get_fit_diagnostics()["fallback"]
