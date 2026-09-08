"""Tree equivalents of the September MPS FIT maintenance changes."""

import warnings

import autoray as ar
import numpy as np
import pytest

from pepsy.fitting import TreeFIT
from pepsy.optimizers.tree import TreeOptimizer, TreePlan, TreeTensorNetwork


@pytest.mark.parametrize("mode,trace", [
    ("dmrg1", (2, 2, 1, 1)),
    ("dmrg2", (2, 2, 1, 1)),
    ("dmrg3", (3, 3, 2, 1)),
])
def test_default_tree_budget_reaches_refinement(mode, trace):
    optimizer = TreeOptimizer(None, n=5, mode=mode, chi=2, cutoff=0.,
                              fit_init_strategy="direct", run=False)
    optimizer.apply_gate(np.eye(8, dtype=complex), (0, 2, 4))
    assert optimizer.fit_n_iter == 4
    assert optimizer.get_fit_diagnostics()["block_size_trace"] == trace
    expected = np.zeros(32, dtype=complex)
    expected[0] = 1
    np.testing.assert_allclose(optimizer.to_dense(), expected, atol=1e-12)
    optimizer.tn.validate(check_canonical=True)


@pytest.mark.parametrize("transition,trace", [
    (0, (3, 3, 1, 1)), (1, (3, 3, 2, 1, 1)),
    (2, (3, 3, 2, 2, 1, 1)),
])
def test_dmrg3_transition_resets_stopping_and_survives_copy(transition, trace):
    optimizer = TreeOptimizer(None, n=5, mode="dmrg3", chi=2, cutoff=0.,
                              fit_init_strategy="direct", fit_n_iter=8,
                              fit_two_site_transition_sweeps=transition, run=False).copy()
    optimizer.apply_gate(np.eye(8, dtype=complex), (0, 2, 4))
    diagnostic = optimizer.get_fit_diagnostics()
    assert diagnostic["block_size_trace"] == trace
    assert diagnostic["convergence_reason"] == "rtol"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_transition_preserves_exact_state_and_backend(backend):
    plan = TreePlan.from_order(range(6), structure="balanced", top_arity=3)
    state = TreeTensorNetwork.rand(plan, D=2, seed=218)
    center = state.node_tensor(plan.root)
    center.modify(data=center.data / np.linalg.norm(state.to_dense()))
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex64))
    fit = TreeFIT(state, state, max_bond=2, cutoffs=0.)
    fit.run(5, block_size=3, adaptive_block_sweeps=2,
            two_site_transition_sweeps=1)
    assert fit.block_size_trace == [3, 3, 2, 1, 1]
    assert all(ar.infer_backend(t.data) == backend for t in fit.p.tensors)
    np.testing.assert_allclose(fit.p.to_dense(), state.to_dense(), atol=2e-6)


@pytest.mark.parametrize("mode", ["direct", "src", "zipup", "dmrg2", "dmrg3"])
def test_replay_final_finite_scan_is_optional_and_scoped(monkeypatch, mode):
    original = TreeOptimizer.apply_gate

    def poison(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        tensor = self.tn.node_tensor(self.plan.root)
        tensor.modify(data=np.full_like(tensor.data, np.nan))
        return result

    monkeypatch.setattr(TreeOptimizer, "apply_gate", poison)
    optimizer = TreeOptimizer([(np.eye(2), 0)], n=3, mode=mode, run=False)
    with warnings.catch_warnings(record=True) as caught:
        optimizer.run()
    assert not any("finite_check" in str(w.message) for w in caught)
    assert np.isnan(optimizer.tn.node_tensor(optimizer.plan.root).data).all()
    with pytest.warns(RuntimeWarning, match="TreeOptimizer finite_check") as caught:
        with pytest.raises(FloatingPointError, match="non-finite"):
            optimizer.run(gates=[], finite_check=True)
    assert len(caught) == 1
    assert not optimizer._finite_check_enabled
    assert not optimizer._finite_check_warning_handled


def test_finite_warning_shared_across_fit_calls_and_checks_still_run(monkeypatch):
    optimizer = TreeOptimizer([(np.eye(8, dtype=complex), (0, 2, 4))] * 2,
                              n=5, mode="dmrg3", run=False, fit_init_strategy="direct")
    original = TreeFIT._check_state_finite
    calls = []

    def check(state, region):
        calls.append(tuple(region))
        return original(state, region)

    monkeypatch.setattr(TreeFIT, "_check_state_finite", staticmethod(check))
    with pytest.warns(RuntimeWarning, match="TreeOptimizer finite_check") as caught:
        optimizer.run(finite_check=True)
    assert len(caught) == 1
    assert len(calls) == 9  # Four iterations per gate, then the final state.
    assert not optimizer._finite_check_enabled
    optimizer.run()
    assert len(calls) == 9


def test_shots_inherit_finite_policy_with_explicit_override(monkeypatch):
    optimizer = TreeOptimizer(None, n=3, run=False)
    options = []
    monkeypatch.setattr(optimizer, "_run_shots", lambda *a, **kw: options.append(kw))
    optimizer.run(shots=2, finite_check=True)
    optimizer.run(shots=2, finite_check=True, run_kwargs={"finite_check": False})
    assert options[0]["run_kwargs"]["finite_check"] is True
    assert options[1]["run_kwargs"]["finite_check"] is False


@pytest.mark.parametrize("value", [-1, 1.5, None])
def test_invalid_transition_budget(value):
    with pytest.raises(ValueError, match="transition_sweeps"):
        TreeOptimizer(None, n=3, fit_two_site_transition_sweeps=value, run=False)
    state = TreeTensorNetwork.from_order(range(3))
    fit = TreeFIT(state, state)
    with pytest.raises(ValueError, match="transition_sweeps"):
        fit.run(two_site_transition_sweeps=value)
