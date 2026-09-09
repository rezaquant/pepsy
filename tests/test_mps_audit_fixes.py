"""Regressions for the final MPS correctness and scaling audit."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy import FIT, MpsOptimizer, TrajectoryChannel


@pytest.mark.parametrize("axis", ["X", "Y"])
@pytest.mark.parametrize("probability", [1e-13, 1e-18])
def test_rare_rotated_measurement_uses_amplitudes(axis, probability):
    phase = 1. if axis == "X" else 1j
    plus = np.array([1., phase]) / np.sqrt(2.)
    minus = np.array([1., -phase]) / np.sqrt(2.)
    vector = np.sqrt(1 - probability) * plus + np.sqrt(probability) * minus
    expected = abs(np.vdot(minus, vector)) ** 2 / np.vdot(vector, vector).real
    opt = MpsOptimizer(qtn.MatrixProductState.from_dense(vector, [2]),
                       [("measure", axis, 0, -1)], chi=2)
    opt.run(cutoff=0.)
    assert opt.measurements[-1][3] == pytest.approx(expected, rel=3e-7, abs=0.)
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0., abs=3e-7)
    assert abs(np.vdot(minus, opt.to_dense().ravel())) == pytest.approx(1.)


@pytest.mark.parametrize("mode", ["direct", "dmrg2", "exact"])
@pytest.mark.parametrize("exponent", [-400., 400.])
def test_measurement_cancels_unrepresentable_common_scale(mode, exponent):
    opt = MpsOptimizer(qtn.MPS_computational_state("+++", dtype="complex128"),
                       [("measure", "ZZ", (0, 2), 1)], chi=4, mode=mode)
    opt.p.exponent = exponent
    opt.run(cutoff=0.)
    assert opt.measurements[-1][3] == pytest.approx(.5)
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0., abs=1e-12)
    assert np.linalg.norm(opt.to_dense()) == pytest.approx(1.)
    assert opt.get_norm_events()[-1]["expected_norm_exponent"] == exponent


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
@pytest.mark.parametrize("exponent", [-400., 400.])
def test_kraus_replay_cancels_common_scale(strategy, exponent):
    from pepsy.optimizers.noise import _kraus_probabilities
    state = qtn.MPS_computational_state("11", dtype="complex128")
    state.exponent = exponent
    opt = MpsOptimizer(state, [("amplitude_damping", .3, 0)], chi=2)
    np.testing.assert_allclose(
        _kraus_probabilities(opt, TrajectoryChannel.amplitude_damping(.3), (0,)), [.7, .3])
    result = opt.run(shots=16, strategy=strategy, seed=9)
    for leaf in result.optimizers:
        assert leaf.norm_diagnostics()["infidelity"] == pytest.approx(0., abs=1e-12)
        assert np.linalg.norm(leaf.to_dense()) == pytest.approx(1.)


def test_coalesced_rare_x_measurement_retains_accurate_branch_weight():
    probability = 1e-13
    plus, minus = np.array([1., 1.]) / np.sqrt(2.), np.array([1., -1.]) / np.sqrt(2.)
    state = np.sqrt(1 - probability) * plus + np.sqrt(probability) * minus
    opt = MpsOptimizer(qtn.MatrixProductState.from_dense(state, [2]),
                       [("measure", "X", 0)], chi=2)
    result = opt.run(shots=10**15, strategy="coalesced", seed=8)
    assert result.branches == 2
    rare = next(leaf for leaf in result.optimizers if leaf.measurements[-1][2] == -1)
    assert rare.measurements[-1][3] == pytest.approx(probability, rel=1e-6, abs=0.)
    assert rare.norm_diagnostics()["infidelity"] == pytest.approx(0., abs=1e-6)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_rotated_amplitude_probability_preserves_backend(backend):
    probability = 1e-7 if backend == "jax" else 1e-18
    vector = (np.sqrt(1 - probability) * np.array([1., 1.])
              + np.sqrt(probability) * np.array([1., -1.])) / np.sqrt(2.)
    state = qtn.MatrixProductState.from_dense(vector.astype(complex), [2])
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
    else:
        jnp = pytest.importorskip("jax.numpy")
        state.apply_to_arrays(lambda x: jnp.asarray(x, dtype=jnp.complex64))
    opt = MpsOptimizer(state, [("measure", "X", 0, -1)], chi=2)
    opt.run(cutoff=0.)
    assert opt.backend == backend
    assert opt.measurements[-1][3] == pytest.approx(probability, rel=1e-3, abs=0.)
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0., abs=1e-3)


def test_large_pauli_measurement_never_builds_dense_operator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dense Pauli operator allocation")
    opt = MpsOptimizer(qtn.MPS_computational_state("+" * 24, dtype="complex128"),
                       [("measure", "Z" * 24, tuple(range(24)), 1)], chi=2)
    monkeypatch.setattr(opt, "_pauli_operator", forbidden)
    opt.run(cutoff=0.)
    assert opt.measurements[-1][3] == pytest.approx(.5)
    assert opt.p.max_bond() <= 2
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("invalid", [{"normalize_final": True}, {"fit_patience": 0},
                                    {"submpo_method": "invalid"}])
def test_invalid_layout_replay_preserves_logical_state(invalid):
    opt = MpsOptimizer(qtn.MPS_computational_state("100"), [("h", 0)], chi=4, mode="dmrg2")
    before = opt.to_dense().copy()
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
        opt.run(layout=opt._explicit_layout_plan((2, 1, 0)), layout_report=False, **invalid)
    np.testing.assert_allclose(opt.to_dense(), before)
    assert opt.logical_order == [0, 1, 2]


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_shots_forward_solver_options_with_explicit_child_override(monkeypatch, strategy):
    original = FIT.run_gate
    calls = []
    def observe(self, *args, **kwargs):
        calls.append((kwargs["n_iter"], kwargs["cutoff"], kwargs["rtol"]))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(FIT, "run_gate", observe)
    opt = MpsOptimizer(qtn.MPS_rand_state(4, 2, seed=19),
                       [(qu.CNOT(), (0, 3))], chi=4, mode="dmrg2")
    opt.run(shots=2, strategy=strategy, n_iter=3, cutoff=0., fit_rtol=None)
    assert calls and all(call == (3, 0., None) for call in calls)
    calls.clear()
    opt.run(shots=2, strategy=strategy, n_iter=3, cutoff=0., fit_rtol=None,
            run_kwargs={"n_iter": 4})
    assert calls and all(call == (4, 0., None) for call in calls)


def test_internal_branch_copy_preserves_isometries_without_scanning(monkeypatch):
    opt = MpsOptimizer(qtn.MPS_rand_state(5, 3, seed=23), chi=4)
    opt.canonize_mps(opt.p, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Branch clone rediscovered a known center")
    monkeypatch.setattr(qtn.MatrixProductState, "calc_current_orthog_center", forbidden)
    branch = opt._copy_for_trajectory_branch()
    assert branch.info_c == opt.info_c
    for site in range(opt.p.L):
        assert branch.p[site].left_inds == opt.p[site].left_inds
        assert not np.shares_memory(branch.p[site].data, opt.p[site].data)
    branch.set_gates([("measure", "Z", 4, 1)])
    branch.run()
    np.testing.assert_allclose(opt.to_dense(), opt._initial_p.to_dense(), atol=1e-12)


def test_published_branches_have_independent_histories():
    opt = MpsOptimizer(qtn.MPS_computational_state("00"),
                       [("h", 0), ("cnot", 0, 1), ("measure", "Z", 0)], chi=4)
    result = opt.run(shots=128, strategy="coalesced", seed=18)
    first, second = result.optimizers
    original = second.norm_events[0]["kind"]
    first.norm_events[0]["kind"] = "mutated"
    assert second.norm_events[0]["kind"] == original


def test_support_history_shares_unchanged_immutable_snapshots():
    opt = MpsOptimizer(qtn.MPS_computational_state("0" * 64),
                       [("h", site) for site in range(64)] + [("h", 0)] * 1000, chi=2)
    opt.run()
    last = opt._effective_site_history[-1]
    assert all(snapshot is last for snapshot in opt._effective_site_history[-1000:])


def test_compact_diagnostics_match_full_without_global_norm(monkeypatch):
    opt = MpsOptimizer(qtn.MPS_rand_state(4, 2, seed=7), [(qu.CNOT(), (0, 3))], chi=2)
    opt.run()
    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostics contracted a global norm")
    monkeypatch.setattr(qtn.MatrixProductState, "norm", forbidden)
    for _ in range(2):
        full = opt.norm_diagnostics()
        compact = opt.norm_diagnostics(include_history=False)
        for key, value in compact.items():
            assert value == pytest.approx(full[key]) if isinstance(value, float) else value == full[key]
        opt.run()


def test_direct_mode_drops_old_fit_diagnostics():
    opt = MpsOptimizer(qtn.MPS_rand_state(4, 2, seed=7), [(qu.CNOT(), (0, 3))], chi=4, mode="dmrg2")
    opt.run()
    assert opt.get_fit_diagnostics() is not None
    opt.set_mode("direct")
    opt.set_gates([("h", 0)])
    opt.run()
    assert opt.get_fit_diagnostics() is None
