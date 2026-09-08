"""Replay metadata reuse, invalidation, and disposable FIT ownership."""

import gc
import weakref

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py


def _state(length=16, bond=4):
    return qtn.MPS_rand_state(length, bond, dtype="complex128", seed=714)


def _scope(opt, fn):
    return opt._run_with_fit_copy_policy(fn, enabled=False, event_count=0)


def test_mixed_metadata_reuse_matches_uncached_replay(monkeypatch):
    state = _state()
    gates = [(qu.rand_uni(4, seed=820 + i), (4 + i % 2, 8 + i % 2))
             for i in range(4)]
    opt = py.MpsOptimizer(state, gates, chi=4, mode="mix")
    counts = dict(ranks=0, arrays=0, prepare=0, maximum=0)
    for name, key in (("_compute_mix_target_bond_dimensions", "ranks"),
                      ("_has_symmray_data", "arrays"),
                      ("_prepare_fit_window", "prepare")):
        original = getattr(opt, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(opt, name, counted)
    original_maximum = qtn.MatrixProductState.max_bond

    def maximum(p):
        counts["maximum"] += 1
        return original_maximum(p)

    with monkeypatch.context() as patcher:
        patcher.setattr(qtn.MatrixProductState, "max_bond", maximum)
        opt.run()
    assert counts == dict(ranks=1, arrays=1, prepare=4, maximum=6)
    assert all(event["backend"] == "dmrg" for event in opt.mix_history)

    reference = py.MpsOptimizer(state, gates, chi=4, mode="mix")
    monkeypatch.setattr(reference, "_mix_target_bond_dimensions",
                        reference._compute_mix_target_bond_dimensions)
    monkeypatch.setattr(reference, "_replay_has_symmray_data", reference._has_symmray_data)
    reference.run()
    np.testing.assert_allclose(opt.p.to_dense(), reference.p.to_dense(), atol=1e-11)
    assert opt.info_c == reference.info_c
    assert opt.mix_history == reference.mix_history
    assert opt.get_fit_diagnostics() == reference.get_fit_diagnostics()


def test_rank_cache_invalidates_for_state_chi_layout_and_cap():
    def product(dims):
        return qtn.MPS_product_state([np.ones(d, dtype=complex) / np.sqrt(d) for d in dims])

    opt = py.MpsOptimizer(product([2, 3, 4, 2]), [], chi=8, mode="dmrg2")

    def execute():
        first = opt._mix_target_bond_dimensions()
        assert first is opt._mix_target_bond_dimensions()
        assert first == [2, 6, 2]
        opt.set_p(product([3, 2, 2, 4]))
        assert not opt._replay_rank_cache
        assert opt._mix_target_bond_dimensions() == [3, 6, 4]
        opt.chi = 2
        assert opt._mix_target_bond_dimensions() == [2, 2, 2]
        opt.chi = 8
        opt._relabel_product_mps((3, 1, 2, 0), current_order=(0, 1, 2, 3))
        assert not opt._replay_rank_cache
        assert opt._mix_target_bond_dimensions() == [4, 6, 3]
        opt._apply_cap_event((1,), [1, 0], "right")
        assert not opt._replay_rank_cache
        assert opt._mix_target_bond_dimensions() == [4, 3]
        copied = opt.copy()
        assert copied._replay_rank_cache is None and copied._replay_array_cache is None
        raise RuntimeError("injected scope exit")

    with pytest.raises(RuntimeError, match="injected scope exit"):
        _scope(opt, execute)
    assert opt._replay_rank_cache is None
    assert opt._replay_array_cache is None
    assert opt._fit_copy_policy_cache is None


def test_array_classification_uses_actual_networks_and_weak_ownership():
    class WithBlocks(np.ndarray):
        blocks = {}
        indices = ()

    opt = py.MpsOptimizer(_state(4, 2), [], chi=2, mode="dmrg2")
    dense = qtn.TensorNetwork([qtn.Tensor(np.ones(2), inds=("a",))])
    mixed = dense.copy()
    mixed.add_tensor(qtn.Tensor(np.ones(2).view(WithBlocks), inds=("b",)))

    def execute():
        assert not opt._replay_has_symmray_data(dense)
        assert opt._replay_has_symmray_data(mixed)
        copy = opt._inherit_replay_array_kind(dense.copy(), dense)
        ref = weakref.ref(copy)
        assert not opt._replay_has_symmray_data(copy)
        del copy
        gc.collect()
        assert ref() is None
        assert len(opt._replay_array_cache) == 2

    _scope(opt, execute)
    # Unscoped helpers must inspect external mutations afresh.
    assert not opt._replay_has_symmray_data(dense)
    dense.add_tensor(qtn.Tensor(np.ones(2).view(WithBlocks), inds=("c",)))
    assert opt._replay_has_symmray_data(dense)


def test_dmrg1_sufficient_budget_needs_no_rank_check(monkeypatch):
    opt = py.MpsOptimizer(_state(4, 2), [], chi=4, mode="dmrg1")

    def unexpected(*args, **kwargs):
        raise AssertionError("sufficient sweep budgets need no rank check")

    monkeypatch.setattr(py.FIT, "_active_bonds_at_rank_targets", unexpected)
    opt._validate_dmrg1_iteration_budget(opt.p, (0, 3), n_iter=8, block_size=2)


def test_mixed_maximum_is_refreshed_after_quality_repair(monkeypatch):
    gates = [(qu.CNOT(), (2, 5))] * 2
    opt = py.MpsOptimizer(_state(8, 4), gates, chi=4, mode="mix")
    repaired = False

    def repair(step, where, every, *, repair):
        nonlocal repaired
        # Repair at the outer transaction boundary, after its end-bond value
        # was recorded, rather than during the nested DMRG executor.
        if not repaired and len(opt.mix_history) == 1:
            opt.p.compress(max_bond=1, cutoff=0.0)
            opt.sync_canonicalization()
            repaired = True
            return {"repaired": True}
        return None

    monkeypatch.setattr(opt, "_maybe_run_quality_check", repair)
    opt.run()
    assert opt.mix_history[0]["end_bond"] == 4
    assert opt.mix_history[1]["start_bond"] == 1
    assert opt.mix_history[1]["backend"] == "mpo"


@pytest.mark.parametrize("kind", ["gate", "batch", "submpo", "measurement"])
def test_disposable_src_guess_keeps_rollback_state_intact(monkeypatch, kind):
    state = _state(8, 2)
    before = state.to_dense().copy()
    gates = [(qu.CNOT(), (2, 5))]
    options = {}
    if kind == "batch":
        gates.append((qu.CNOT(), (3, 5)))
        options["k_2q_batch"] = 2
    elif kind == "submpo":
        mpo = qtn.MPO_product_operator([qu.pauli("X"), qu.pauli("Z")], sites=(2, 5), L=8)
        gates = [("submpo", mpo, (2, 5))]
    elif kind == "measurement":
        gates = [("measure", "ZZ", (2, 5), 1)]
    reference = py.MpsOptimizer(state, gates, chi=2, mode="direct")
    reference.run(cutoff=0.0)

    def fail(fit, **kwargs):
        fit.p[3].data[...] = np.nan
        raise RuntimeError("partial SRC-guess FIT write")

    monkeypatch.setattr(py.FIT, "run_gate", fail)
    opt = py.MpsOptimizer(state, gates, chi=2, mode="dmrg2")
    opt.run(cutoff=0.0, **options)
    np.testing.assert_allclose(opt.p.to_dense(), reference.p.to_dense(), atol=1e-11)
    np.testing.assert_allclose(state.to_dense(), before, atol=1e-13)
    assert opt.get_fit_diagnostics()["fallback"]


def test_layered_fit_builds_only_visited_tag_selections_and_can_run_globally():
    opt = py.MpsOptimizer(_state(12, 2), [], chi=2, mode="dmrg2")
    opt.canonize_mps(opt.p, (4, 7))
    target = opt._build_norm_target(opt.p, qu.CNOT(), (4, 7), 0.0,
                                    target_strategy="layered")
    lazy = py.FIT(target, p=opt.p, range_int=(4, 7))
    eager = py.FIT(target, p=opt.p, range_int=(4, 7))
    eager._target_tag_tensor_ids = {tag: tuple(ids) for tag, ids in eager.tn.tag_map.items()}
    assert lazy._target_site_tensors is None
    assert not lazy._target_tag_tensor_ids
    for fit in (lazy, eager):
        fit.run_gate(n_iter=4, max_bond=2, rtol=None)
    assert set(lazy._target_tag_tensor_ids) == {f"I{i}" for i in range(4, 8)}
    np.testing.assert_allclose(lazy.p.to_dense(), eager.p.to_dense(), atol=1e-11)
    for fit in (lazy, eager):
        fit.run_eff(n_iter=1)
    assert set(lazy._target_tag_tensor_ids) == {f"I{i}" for i in range(12)}
    np.testing.assert_allclose(lazy.p.to_dense(), eager.p.to_dense(), atol=1e-11)
