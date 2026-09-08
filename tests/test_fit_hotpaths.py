"""Cache, initialization, and scalar-read regressions for gate FIT."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py
import pepsy.fitting.local as local


@pytest.mark.parametrize("backend,direction", [
    ("numpy", "RL"), ("numpy", "LR"), ("torch", "RL"), ("jax", "LR"),
    ("fermionic", "RL"), ("fermionic", "LR"),
])
def test_three_to_two_cache_matches_full_rebuild(monkeypatch, backend, direction):
    if backend == "fermionic":
        pytest.importorskip("symmray")
        state = py.hrs_to_mps(
            6, fermion=py.Fermion(spinful=True, symmetry="U1U1", dtype="complex128"),
            occupations=((1, 0), (0, 1)) * 3, chi=2,
            random_rounds=2, seed=61, dtype="complex128",
        )
        target = state.copy(deep=True)
    else:
        state = qtn.MPS_rand_state(6, 2, dtype="complex128", seed=714)
        target = state.gate_nonlocal(qu.CNOT(), (1, 4), max_bond=4, cutoff=0.0)
        if backend == "torch":
            torch = pytest.importorskip("torch")
            for p in (state, target):
                p.apply_to_arrays(lambda a: torch.as_tensor(a, dtype=torch.complex128))
        elif backend == "jax":
            jax = pytest.importorskip("jax")
            for p in (state, target):
                p.apply_to_arrays(lambda a: jax.numpy.asarray(a, dtype="complex64"))

    cached = py.FIT(target, p=state, range_int=[1, 4])
    rebuilt = py.FIT(target, p=state, range_int=[1, 4])
    rebuilt._allow_sweep_environment_reuse = False
    options = dict(n_iter=5, block_size=3, sweep_sequence=direction,
                   max_bond=4, cutoff=1e-12, rtol=None)
    with monkeypatch.context() as patcher:
        if backend == "fermionic":
            def no_dense(*args, **kwargs):
                raise AssertionError("native cache transition must stay sparse")
            for cls in {type(t.data) for t in state.tensors}:
                patcher.setattr(cls, "to_dense", no_dense)
        cached.run_gate(**options)
        rebuilt.run_gate(**options)
    assert cached._sweep_environment_reuse_count == 4
    assert rebuilt._sweep_environment_reuse_count == 0
    assert cached.final_center_site == rebuilt.final_center_site
    if backend == "fermionic":
        assert float(np.real(py.tn_fidelity(cached.p, rebuilt.p,
                                           contraction_opt="greedy"))) == pytest.approx(1, abs=1e-10)
    else:
        np.testing.assert_allclose(local.ar.to_numpy(cached.p.to_dense()),
                                   local.ar.to_numpy(rebuilt.p.to_dense()), atol=3e-6)


def test_src_skips_rank_scan_and_reuses_copy_capability(monkeypatch):
    p = qtn.MPS_rand_state(8, 2, dtype="complex128", seed=714)
    mpo = qtn.MPO_product_operator([qu.pauli("X"), qu.pauli("Z")],
                                   sites=(2, 4), L=8)
    gates = [(qu.CNOT(), (2, 4)), ("submpo", mpo, (2, 4)),
             ("measure", "ZZ", (2, 4), 1)]
    opt = py.MpsOptimizer(p, gates, chi=2, mode="dmrg3")
    calls = []
    original = opt._fit_window_copy_supported

    def classify(state):
        calls.append(state)
        return original(state)

    def no_rank_scan(*args, **kwargs):
        raise AssertionError("SRC selection does not depend on rank growth")

    monkeypatch.setattr(opt, "_fit_window_copy_supported", classify)
    monkeypatch.setattr(py.FIT, "_active_bonds_at_rank_targets", no_rank_scan)
    opt.run(n_iter=5, progbar=False)
    assert opt.get_fit_diagnostics()["guess_method"] == "src"
    assert len(calls) == 1
    assert opt._fit_copy_policy_cache is None
    opt._copy_fit_window_state(opt.p, (2, 4))
    assert len(calls) == 2  # Calls outside replay recheck their actual input.

    def fail(*args, **kwargs):
        opt._copy_fit_window_state(opt.p, (2, 4))
        opt.set_p(opt.p.copy())
        opt._copy_fit_window_state(opt.p, (2, 4))
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(opt, "_run_segmented", fail)
    with pytest.raises(RuntimeError, match="injected replay failure"):
        opt.run(progbar=False)
    assert len(calls) == 4  # Explicit state replacement invalidates the cache.
    assert opt._fit_copy_policy_cache is None


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_scalar_norm_read_avoids_stack_and_preserves_nonfinite(monkeypatch, backend):
    values = [np.array(x) for x in (0.5, float("nan"), float("inf"))]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        values = [torch.as_tensor(x).requires_grad_() for x in values]
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        values = [jax.numpy.asarray(x) for x in values]
    original = local.ar.do

    def no_stack(fn, *args, **kwargs):
        if fn == "stack":
            raise AssertionError("scalar-only read should allocate no vector")
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(local.ar, "do", no_stack)
    result = [py.FIT._sweep_diagnostics_to_host(None, 0, 0, x,
              check_finite=False, read_norm=True)[1] for x in values]
    assert result[0] == 0.5
    assert np.isnan(result[1])
    assert np.isinf(result[2])


@pytest.mark.parametrize("finite_check", [False, True])
def test_scalar_norm_detection_is_opt_in(monkeypatch, finite_check):
    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=53)
    fit = py.FIT(state.copy(), p=state, range_int=[0, 3])
    original = fit._run_gate_two_site_sweep

    def bad_norm(*args, **kwargs):
        boundaries = original(*args, **kwargs)
        fit.local_norm_trace[-1] = np.array(float("nan"))
        return boundaries

    monkeypatch.setattr(fit, "_run_gate_two_site_sweep", bad_norm)
    if finite_check:
        with pytest.warns(RuntimeWarning, match="finite_check is enabled"):
            with pytest.raises(FloatingPointError, match="non-finite"):
                fit.run_gate(n_iter=2, finite_check=True)
    else:
        fit.run_gate(n_iter=2)
        assert fit.iterations_run == 2
        assert not fit.converged
        assert np.isnan(fit.final_norm)
