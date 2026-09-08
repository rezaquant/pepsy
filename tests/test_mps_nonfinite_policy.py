"""All MPS replay modes opt in to runtime non-finite detection."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py


@pytest.mark.parametrize("mode", [
    "dmrg", "dmrg1", "dmrg2", "dmrg3", "mix", "mpo", "direct",
    "src", "sdc", "swap", "perm", "svd", "su", "exact",
])
def test_runtime_nonfinite_detection_is_opt_in(monkeypatch, mode):
    original = py.MpsOptimizer._execute_mode

    def poison_after_replay(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        tensor = self.p.tensors[0]
        tensor.modify(data=np.full_like(tensor.data, np.nan))
        return result

    monkeypatch.setattr(py.MpsOptimizer, "_execute_mode", poison_after_replay)
    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=714)
    unchecked = py.MpsOptimizer(state, [(qu.CNOT(), (0, 3))], chi=2, mode=mode)
    unchecked.run(n_iter=3, progbar=False, timing=True)
    assert np.isnan(unchecked.p.tensors[0].data).all()
    assert unchecked.get_run_timing()["status"] == "complete"
    assert unchecked._finite_check_enabled is False

    checked = py.MpsOptimizer(state, [(qu.CNOT(), (0, 3))], chi=2, mode=mode)
    with pytest.warns(RuntimeWarning, match="finite_check is enabled"):
        with pytest.raises(FloatingPointError, match="non-finite MPS data"):
            checked.run(n_iter=3, progbar=False, finite_check=True, timing=True)
    assert checked.get_run_timing()["status"] == "failed"
    assert checked._finite_check_enabled is False


def test_default_mix_skips_commit_norm_validation(monkeypatch):
    opt = py.MpsOptimizer(qtn.MPS_computational_state("000"), [], chi=2, mode="mix")

    def no_read(*args, **kwargs):
        raise AssertionError("disabled validation must not read a norm")

    monkeypatch.setattr(opt, "_retained_center_norm", no_read)
    assert opt._validate_mix_norm((0, 2), operation="test") is None


def test_unchecked_norm_diagnostics_propagate_nan():
    raw, clipped = py.MpsOptimizer._fidelity_ratio_from_norms(float("nan"), 1)
    assert np.isnan(raw) and np.isnan(clipped)
    assert py.MpsOptimizer._fidelity_ratio_from_norms(
        float("nan"), 1, finite_check=True
    ) == (None, None)


def test_empty_replay_obeys_finite_policy():
    opt = py.MpsOptimizer(qtn.MPS_computational_state("00"), [], chi=2, mode="mpo")
    opt.p[0].modify(data=np.full_like(opt.p[0].data, np.nan))
    opt.run(progbar=False)
    with pytest.warns(RuntimeWarning, match="finite_check is enabled"):
        with pytest.raises(FloatingPointError, match="non-finite MPS data"):
            opt.run(progbar=False, finite_check=True)
    assert opt._finite_check_enabled is False
