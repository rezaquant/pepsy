"""Gate FIT defaults and phased DMRG sweep schedules."""

import inspect

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py


@pytest.mark.parametrize("mode,expected", [
    ("dmrg2", [2, 2, 1, 1, 1, 1, 1, 1]),
    ("dmrg3", [3, 3, 2, 1, 1, 1, 1, 1]),
])
@pytest.mark.parametrize("operator", [False, True])
def test_named_schedule_and_canonical_handoff(monkeypatch, mode, expected, operator):
    state = (qtn.MPO_rand(6, 2, dtype="complex128", seed=714) if operator
             else qtn.MPS_rand_state(6, 2, dtype="complex128", seed=714))
    blocks = []
    original = py.FIT._start_timing_record

    def observe(self, *args, **kwargs):
        blocks.append(kwargs["block_size"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "_start_timing_record", observe)
    optimizer = py.MpoOptimizer if operator else py.MpsOptimizer
    opt = optimizer(state, [(qu.CNOT(), (1, 4))], chi=4, mode=mode)
    opt.run(progbar=False, fit_rtol=None)
    assert blocks == expected
    d = opt.get_fit_diagnostics()
    assert d["guess_method"] == "src"
    assert not d.get("fallback", False)
    assert d["adaptive_sweeps"] == sum(b > 1 for b in expected)
    center = d["center_site"]
    np.testing.assert_allclose(
        opt.p[center].norm(), np.linalg.norm(opt.p.to_dense()), rtol=1e-10
    )
    assert opt.p.max_bond() <= 4


def test_three_site_transition_survives_loose_tolerance(monkeypatch):
    state = qtn.MPS_rand_state(6, 2, dtype="complex128", seed=12)
    blocks = []
    original = py.FIT._start_timing_record

    def observe(self, *args, **kwargs):
        blocks.append(kwargs["block_size"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "_start_timing_record", observe)
    fit = py.FIT(state.copy(), p=state.copy(), range_int=[1, 4])
    fit.run_gate(block_size=3, rtol=1.0, max_bond=2)
    assert blocks == [3, 3, 2, 1, 1]
    assert fit.convergence_reason == "relative_tolerance"


def test_gate_defaults():
    defaults = inspect.signature(py.FIT.run_gate).parameters
    for name, expected in {
        "n_iter": 8, "block_size": 2, "sweep_sequence": "RL",
        "min_iter": 2, "rtol": "auto", "patience": 2,
        "adaptive_block_sweeps": 2, "finite_check": False,
        "timing": False,
        "collect_split_diagnostics": False,
    }.items():
        assert defaults[name].default == expected

    state = qtn.MPS_rand_state(6, 2, dtype="complex64", seed=34)
    fit = py.FIT(state.copy(), p=state.copy(), range_int=[1, 4])
    fit.run_gate(max_bond=2)
    assert fit.iterations_run == 4
    assert fit.adaptive_sweeps_run == 2
    assert fit.one_site_sweeps_run == 2
    assert fit.convergence_reason == "relative_tolerance"
    assert "two_site_splits" not in fit.info
    np.testing.assert_allclose(fit.p.to_dense(), state.to_dense(), atol=2e-6)
