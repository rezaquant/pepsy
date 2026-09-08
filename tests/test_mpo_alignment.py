"""MPS/MPO shared policies must preserve operator-specific invariants."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy import MpoOptimizer, TrajectoryChannel
from pepsy.fitting import FIT


@pytest.mark.parametrize("mode", [None, "direct", "mpo", "quimb"])
@pytest.mark.parametrize("sides", ["both", "ket", "bra", "pair"])
def test_direct_aliases_preserve_raw_operator_convention(mode, sides):
    rng = np.random.default_rng(927)
    operator, gate, bra = [
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        for _ in range(3)
    ]
    payload = {"both": gate, "ket": (gate, None), "bra": (None, bra),
               "pair": (gate, bra)}[sides]
    expected = operator
    if sides != "bra":
        expected = gate.T @ expected
    if sides != "ket":
        expected = expected @ (gate if sides == "both" else bra).conj()
    kwargs = {} if mode is None else {"mode": mode}
    opt = MpoOptimizer(qtn.MatrixProductOperator.from_dense(operator, dims=[2, 2]),
                       gates=[(payload, (0, 1))], chi=16, **kwargs)
    out = opt.run(cutoff=0.0)
    assert opt.mode == "quimb-direct"
    np.testing.assert_allclose(out.to_dense(), expected, atol=1e-10)
    assert opt.norm_diagnostics()["norm"] == pytest.approx(np.linalg.norm(expected))


def test_operator_rank_ceiling_counts_both_physical_legs():
    mpo = qtn.MPO_identity(4)
    mps = qtn.MPS_computational_state("0000")
    assert FIT._active_bond_rank_targets(mpo, 0, 3, 32) == (4, 16, 4)
    assert FIT._active_bond_rank_targets(mps, 0, 3, 32) == (2, 4, 2)
    opt = MpoOptimizer(mpo, gates=[], chi=32, mode="dmrg1")
    assert opt._rank_targets(opt.p, 0, 3) == (4, 16, 4)
    opt.p.expand_bond_dimension(2, inplace=True)
    assert not opt._dmrg1_all_bonds_at_rank_targets()


@pytest.mark.parametrize("mode", ["direct", "dmrg", "svd"])
def test_channel_sum_retains_absolute_operator_scale(mode):
    gamma = 0.3
    event = MpoOptimizer.channel_event(TrajectoryChannel.amplitude_damping(gamma), 0)
    opt = MpoOptimizer(qtn.MPO_identity(2, dtype="complex128"),
                       gates=[event], chi=4, mode=mode)
    out = opt.run(cutoff=0.0)
    expected = np.kron(np.diag([1 + gamma, 1 - gamma]), np.eye(2))
    np.testing.assert_allclose(out.to_dense(), expected, atol=1e-10)
    assert np.trace(out.to_dense()) == pytest.approx(4)
    assert opt.norm_diagnostics()["norm"] == pytest.approx(np.linalg.norm(expected))
    assert opt.channel_diagnostics()["trace_preserving"] is True


def test_temporary_canonicalization_does_not_change_live_center():
    opt = MpoOptimizer(qtn.MPO_rand(4, 3, seed=93), gates=[], chi=8)
    center = opt.info_c.copy()
    temporary = opt.p.copy(deep=True)
    opt.canonize_mpo(temporary, (0,))
    assert opt._current_orthog(temporary) == (0, 0)
    assert opt.info_c == center


@pytest.mark.parametrize("mode", ["direct", "svd", "dmrg"])
def test_off_center_one_sided_nonunitary_preserves_norm_tracking(mode):
    initial = qtn.MPO_rand(4, 3, dtype="complex128", seed=92)
    gate = np.array([[0.3, 0.7j], [0.1, 1.4]], dtype=complex)
    full = np.kron(gate.T, np.eye(8))
    expected = full @ initial.to_dense()
    opt = MpoOptimizer(initial, gates=[((gate, None), (0,))], chi=16, mode=mode)
    out = opt.run(cutoff=0.0)
    np.testing.assert_allclose(out.to_dense(), expected, atol=1e-10)
    assert opt.norm_diagnostics()["norm"] == pytest.approx(np.linalg.norm(expected))
    assert opt.info_c["cur_orthog"] == (0, 0)


@pytest.mark.parametrize("keyword", ["finite_check", "fit_finite_check"])
def test_empty_replay_checks_and_cleans_up_policy(keyword):
    opt = MpoOptimizer(qtn.MPO_identity(3), gates=[], chi=4)
    with pytest.warns(RuntimeWarning, match="finite_check"), pytest.raises(
        FloatingPointError, match="non-finite"
    ):
        opt.run(**{keyword: lambda _: False}, timing=True)
    assert opt.last_run_status == "failed"
    assert opt.get_run_timing()["status"] == "failed"
    assert not opt._finite_check_enabled
    assert opt._replay_rank_cache is None
    assert opt._replay_array_kinds is None
    opt.run()
    assert opt.last_run_status == "complete"


def test_finite_check_conflicting_aliases_rejected():
    opt = MpoOptimizer(qtn.MPO_identity(3), gates=[], chi=4)
    with pytest.raises(ValueError, match="disagree"):
        opt.run(finite_check=False, fit_finite_check=True)


def test_transaction_restores_inplace_array_mutation(monkeypatch):
    opt = MpoOptimizer(qtn.MPO_rand(4, 2, seed=52),
                       gates=[(qu.CNOT(), (1, 2))], chi=4, mode="dmrg")
    expected = opt.p.to_dense().copy()

    def corrupt_and_fail(fit, **kwargs):
        fit.p[1].data[...] = 0.0
        raise RuntimeError("injected array mutation")

    monkeypatch.setattr(opt, "_run_fit_gate", corrupt_and_fail)
    with pytest.raises(RuntimeError, match="injected array mutation"):
        opt.run(atomic=False, transactional_steps=True, fit_init_strategy="direct")
    np.testing.assert_allclose(opt.p.to_dense(), expected, atol=1e-10)


def test_window_copy_preserves_torch_gradients_and_isometry_metadata():
    torch = pytest.importorskip("torch")
    opt = MpoOptimizer(qtn.MPO_identity(4), gates=[], chi=4)
    for tensor in opt.p:
        tensor.modify(data=torch.tensor(tensor.data, requires_grad=True),
                      left_inds=tensor.left_inds)
    copied = opt._copy_working_state(opt.p, (1, 2))
    assert copied[0].data is opt.p[0].data
    for site in (1, 2):
        assert copied[site].data.data_ptr() != opt.p[site].data.data_ptr()
        assert copied[site].left_inds == opt.p[site].left_inds
    copied[1].data.sum().backward()
    assert opt.p[1].data.grad is not None
