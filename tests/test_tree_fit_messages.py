"""Tree FIT environment reuse, automatic tolerances, and private ownership."""

from copy import deepcopy
from types import MethodType

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn
import pepsy

from pepsy.fitting import TreeFIT
from pepsy.optimizers.tree import TreeOptimizer, TreePlan, TreeTensorNetwork


def _full_branch_message(fit, outside, inside):
    """Independent reference: contract the complete branch in one network."""
    component = fit._component(outside, inside)
    tensors = [t for node in component for t in fit._target_tensors[node]]
    tensors.extend(fit.p.node_tensor(node).H for node in component)
    return qtn.TensorNetwork(tensors).contract(
        output_inds=(*fit._target_bonds[(outside, inside)], fit.p.bond(outside, inside)),
        optimize=fit.contraction_opt,
    )


@pytest.mark.parametrize("backend,dtype", [("numpy", "complex128"), ("torch", "complex64")])
@pytest.mark.parametrize("layered", [False, True])
def test_incremental_messages_match_complete_branches(backend, dtype, layered, monkeypatch):
    plan = TreePlan.from_order(range(1, 7), structure="balanced", max_arity=3,
                               top_arity=2, root_qubit=0)
    state = TreeTensorNetwork.rand(plan, D=2, seed=51)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda x: torch.as_tensor(x, dtype=getattr(torch, dtype))
    else:
        convert = lambda x: np.asarray(x, dtype=dtype)
    state.apply_to_arrays(convert)
    optimizer = TreeOptimizer(None, state=state, run=False)
    if layered:
        gate = convert(np.diag(np.arange(1, 9) / 8).astype(complex))
        target, _ = optimizer._build_tree_fit_target(gate, (0, 2, 6))
        target.apply_to_arrays(convert)
    else:
        target = state.copy()
    fit = TreeFIT(target, optimizer.tn, max_bond=2, cutoffs=0.)
    target_before = {tid: tensor.inds for tid, tensor in fit.tn.tensor_map.items()}
    input_counts = []
    contract = qtn.tensor_contract

    def counted(*tensors, **kwargs):
        input_counts.append(len(tensors))
        return contract(*tensors, **kwargs)

    edges = [(u, v) for u in fit.nodes for v in fit.p.neighbors(u)]
    with monkeypatch.context() as patch:
        patch.setattr(qtn, "tensor_contract", counted)
        for edge in edges:
            fit._message(*edge)
    assert len(input_counts) == len(edges)
    layers = 2 if layered else 1
    assert max(input_counts) <= layers + max(len(fit.p.neighbors(n)) for n in fit.nodes)
    assert not hasattr(fit, "_components")  # no quadratic component-set table
    assert all(not tensor.tags for tensor in fit._messages.values())
    tolerance = 2e-5 if dtype == "complex64" else 1e-11
    for edge in edges:
        expected = _full_branch_message(fit, *edge)
        actual = fit._message(*edge).transpose(*expected.inds)
        np.testing.assert_allclose(ar.to_numpy(actual.data), ar.to_numpy(expected.data),
                                   atol=tolerance, rtol=tolerance)
    assert target_before == {tid: t.inds for tid, t in fit.tn.tensor_map.items()}

    changed = plan.node_of_qubit[2]
    survivors = {edge: fit._messages[edge] for edge in edges
                 if changed not in fit._component(*edge)}
    fit.p.node_tensor(changed).modify(data=fit.p.node_tensor(changed).data * 1.125)
    fit._invalidate_for_block((changed,))
    assert set(fit._messages) == set(survivors)
    assert all(fit._messages[edge] is tensor for edge, tensor in survivors.items())
    for edge in edges:
        expected = _full_branch_message(fit, *edge)
        actual = fit._message(*edge).transpose(*expected.inds)
        np.testing.assert_allclose(ar.to_numpy(actual.data), ar.to_numpy(expected.data),
                                   atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("schedule", [
    dict(n_iter=3, block_size=2),
    dict(n_iter=5, block_size=3, adaptive_block_sweeps=2,
         two_site_transition_sweeps=1),
])
def test_incremental_fit_matches_full_branch_sweeps(schedule):
    plan = TreePlan.from_order(range(6), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=3, seed=43)
    guess = TreeTensorNetwork.rand(plan, D=2, seed=44)
    fast = TreeFIT(target, guess, max_bond=2, cutoffs=0.)
    reference = TreeFIT(target, guess, max_bond=2, cutoffs=0.)
    reference._message = MethodType(_full_branch_message, reference)
    fast.run_eff(**schedule)
    reference.run_eff(**schedule)
    np.testing.assert_allclose(fast.p.to_dense(), reference.p.to_dense(), atol=1e-10)
    np.testing.assert_allclose(fast.local_norm_trace, reference.local_norm_trace, atol=1e-10)
    fast.p.validate(check_canonical=True)


def test_incremental_fit_preserves_even_native_fermionic_state(monkeypatch):
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=((1, 1), (0, 0), (1, 1), (0, 0)))
    optimizer = TreeOptimizer(None, state=state, chi=16, cutoff=0., run=False)
    target, operator = optimizer._build_tree_fit_target(
        fermion.hopping_gate(.1, t=1., imaginary=False), (0, 3),
    )
    optimizer.apply_subtreempo(operator)
    fit = TreeFIT(target, optimizer.tn, max_bond=16, cutoffs=0., finite_check=True)
    update = fit.fit_block

    def checked_update(*args, **kwargs):
        result = update(*args, **kwargs)
        # Opposite phase errors can cancel over a full pass. Every individual
        # projection must preserve an already exact, representable state.
        assert float(pepsy.tensors.tn_fidelity(fit.p, optimizer.tn)) > 1 - 1e-10
        return result

    monkeypatch.setattr(fit, "fit_block", checked_update)
    with pytest.warns(RuntimeWarning, match="TreeFIT finite_check"):
        fit.run_eff(4, block_size=3, adaptive_block_sweeps=2,
                    two_site_transition_sweeps=1)
    assert fit.block_size_trace == [3, 3, 2, 1]
    assert float(pepsy.tensors.tn_fidelity(fit.p, optimizer.tn)) > 1 - 1e-10
    assert fit.fit_diagnostics(overlap=True)["target_fidelity"] > 1 - 1e-10
    assert all(ar.infer_backend(t.data) == "symmray" for t in fit.p.tensors)


@pytest.mark.parametrize("dtype,cutoff,rtol", [
    ("float16", 1e-3, 1e-3), ("float32", 1e-6, 1e-5),
    ("complex64", 1e-6, 1e-5), ("float64", 1e-12, 1e-9),
    ("complex128", 1e-12, 1e-9),
])
def test_tree_auto_tolerances_follow_state_dtype(dtype, cutoff, rtol):
    optimizer = TreeOptimizer(None, n=3, dtype=dtype, run=False)
    assert optimizer.cutoff == cutoff
    assert optimizer.cutoff_mode == "rsum2"
    assert optimizer.fit_rtol == rtol
    assert optimizer.fit_min_iter == 2
    assert optimizer.fit_sweep_sequence == "inward-outward"


@pytest.mark.parametrize("rtol", [None, 0., 1e-7])
def test_tree_explicit_tolerances_and_copy_preserve_policy(rtol):
    optimizer = TreeOptimizer(None, n=3, mode="dmrg3", run=False,
                              cutoff=1e-4, cutoff_mode="rel", fit_rtol=rtol,
                              fit_adaptive_sweeps=3, fit_sweep_sequence="LR")
    clone = optimizer.copy()
    assert clone.cutoff == 1e-4
    assert clone.cutoff_mode == "rel"
    assert clone.fit_rtol == rtol
    assert clone.fit_adaptive_sweeps == 3
    assert clone.fit_sweep_sequence == "outward-inward"
    assert clone._dmrg_mode_alias == "dmrg3"


@pytest.mark.parametrize("rtol", [-1, float("nan"), float("inf"), "invalid"])
def test_tree_invalid_fit_tolerance_fails_at_construction(rtol):
    with pytest.raises(ValueError, match="fit_rtol"):
        TreeOptimizer(None, n=2, run=False, fit_rtol=rtol)


@pytest.mark.parametrize("sequence,canonical", [
    ("RL", "inward-outward"), ("INOUT", "inward-outward"),
    ("inward-outward", "inward-outward"), ("LR", "outward-inward"),
    ("OUTIN", "outward-inward"), ("outward-inward", "outward-inward"),
])
def test_tree_sweep_names_preserve_direction(sequence, canonical):
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=19)
    fit = TreeFIT(state, state, max_bond=2, cutoffs=0.)
    fit.run_eff(1, block_size=2, sweep_sequence=sequence)
    assert fit.fit_diagnostics()["sweep_sequence"] == canonical
    assert fit.final_direction == ("out" if canonical == "inward-outward" else "in")
    np.testing.assert_allclose(fit.p.to_dense(), state.to_dense(), atol=1e-10)


def test_auto_stopping_and_nonunitary_fixed_iterations():
    gate = np.eye(8, dtype=complex)
    options = dict(n=5, mode="dmrg2", chi=2, cutoff=0., fit_n_iter=8,
                   fit_init_strategy="direct", run=False)
    unitary = TreeOptimizer(None, **options)
    unitary.apply_gate(gate, (0, 2, 4))
    assert unitary.get_fit_diagnostics()["fit_rtol"] == 1e-9
    assert unitary.get_fit_diagnostics()["convergence_reason"] == "rtol"
    assert unitary.get_fit_diagnostics()["block_size_trace"] == (2, 2, 1, 1)
    for optimizer in (TreeOptimizer(None, **options), TreeOptimizer(None, **options).copy()):
        optimizer.apply_gate(0.5 * gate, (0, 2, 4), track_norm=False)
        assert optimizer.get_fit_diagnostics()["fit_rtol"] is None
        assert optimizer.get_fit_diagnostics()["iterations"] == 8
        assert optimizer.get_fit_diagnostics()["convergence_reason"] == "max_iter"
    fixed = TreeOptimizer(None, **options, fit_rtol=None)
    fixed.apply_gate(gate, (0, 2, 4))
    assert fixed.get_fit_diagnostics()["iterations"] == 8
    nonunitary = TreeOptimizer([(0.5 * gate, (0, 2, 4))], **options)
    nonunitary.run(non_unitary=True)
    diagnostic = nonunitary.get_fit_diagnostics()
    assert diagnostic["fit_rtol"] is None
    assert diagnostic["local_fidelity"] is None
    assert diagnostic["iterations"] == 8
    explicit = TreeOptimizer(None, **options, fit_rtol=1e-8)
    explicit.apply_gate(0.5 * gate, (0, 2, 4), track_norm=False)
    assert explicit.get_fit_diagnostics()["fit_rtol"] == 1e-8
    assert explicit.get_fit_diagnostics()["convergence_reason"] == "rtol"


def test_fit_guess_skips_histories_and_preserves_child_seed_sequence(monkeypatch):
    optimizer = TreeOptimizer(None, n=5, mode="dmrg2", chi=2, cutoff=0.,
                              seed=13, run=False)
    target, operator = optimizer._build_tree_fit_target(np.eye(8, dtype=complex), (0, 2, 4))
    region = optimizer.tn.steiner_nodes([optimizer.plan.node_of_qubit[q] for q in (0, 2, 4)])
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = deepcopy(optimizer.rng.bit_generator.state)
    expected_rng.integers(0, 2**63, dtype=np.uint64)
    state_before = optimizer.to_dense().copy()
    operator_before = [(t.inds, t.data.copy()) for t in operator.tensors]

    class NoHistoryCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("FIT warm start copied parent history")

    optimizer.update_history.append(NoHistoryCopy())

    def no_optimizer_copy(*args, **kwargs):
        raise AssertionError("FIT warm start cloned the full optimizer")

    monkeypatch.setattr(optimizer, "copy", no_optimizer_copy)
    guess, *_ = optimizer._tree_fit_initial_guess(target, region, operator=operator)
    assert optimizer.rng.bit_generator.state == expected_rng.bit_generator.state
    np.testing.assert_array_equal(optimizer.to_dense(), state_before)
    for tensor, (inds, data) in zip(operator.tensors, operator_before):
        assert tensor.inds == inds
        np.testing.assert_array_equal(tensor.data, data)
    guess.node_tensor(guess.plan.root).modify(data=guess.node_tensor(guess.plan.root).data * 2)
    np.testing.assert_array_equal(optimizer.to_dense(), state_before)


@pytest.mark.parametrize("backend,block_size", [("numpy", 2), ("numpy", 3), ("torch", 3)])
def test_fit_skips_interior_qr_and_reuses_sweep_order(backend, block_size, monkeypatch):
    plan = TreePlan.from_order(range(6), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=3, seed=716)
    guess = TreeTensorNetwork.rand(plan, D=2, seed=717)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        for state in (target, guess):
            state.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    fast = TreeFIT(target, guess, max_bond=2, cutoffs=0.)
    reference = TreeFIT(target, guess, max_bond=2, cutoffs=0.)

    def prepare_at_midpoint(block, center):
        current = reference.p.orthogonality_center
        assert current is not None
        reference._invalidate_for_block(plan.node_path(current, center))
        reference.p.shift_orthogonality_center(center, _skip_validate=True)

    monkeypatch.setattr(reference, "_canonicalize_for_block", prepare_at_midpoint)
    hops = [[], []]
    for index, fit in enumerate((fast, reference)):
        shift = fit.p.shift_orthogonality_center

        def count(node, *, _index=index, _fit=fit, _shift=shift, **kwargs):
            hops[_index].append(len(plan.node_path(_fit.p.orthogonality_center, node)) - 1)
            return _shift(node, **kwargs)

        monkeypatch.setattr(fit.p, "shift_orthogonality_center", count)
    orders = []
    sweep_blocks = fast._sweep_blocks

    def count_orders(region, size, direction):
        orders.append((size, direction))
        return sweep_blocks(region, size, direction)

    monkeypatch.setattr(fast, "_sweep_blocks", count_orders)
    fast.run_eff(3, block_size=block_size)
    reference.run_eff(3, block_size=block_size)
    assert len(orders) == 2
    assert sum(hops[0]) < sum(hops[1])
    np.testing.assert_allclose(fast.p.to_dense(), reference.p.to_dense(), atol=1e-9)
    np.testing.assert_allclose(fast.local_norm_trace, reference.local_norm_trace, atol=1e-9)
    fast.p.validate(check_canonical=True)


def test_three_node_fit_accepts_an_endpoint_center():
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=123)
    fit = TreeFIT(state, state, max_bond=4, cutoffs=0.)
    block = fit._connected_triples(fit.nodes)[0]
    endpoint = next(n for n in block if sum(v in block for v in state.neighbors(n)) == 1)
    fit.fit_block(block, center=endpoint)
    assert fit.p.orthogonality_center == endpoint
    np.testing.assert_allclose(fit.p.to_dense(), state.to_dense(), atol=1e-10)
    fit.p.validate(check_canonical=True)


def test_guess_direct_applies_operator_without_mutating_source():
    optimizer = TreeOptimizer(None, n=3, chi=4, cutoff=0., run=False,
                              fit_init_strategy="guess-direct")
    gate = np.eye(4, dtype=complex)[[3, 2, 1, 0]]
    before = optimizer.to_dense().copy()
    target, operator = optimizer._build_tree_fit_target(gate, (0, 2))
    region = optimizer.tn.steiner_nodes([optimizer.plan.node_of_qubit[q] for q in (0, 2)])
    guess, strategy, *_ = optimizer._tree_fit_initial_guess(target, region, operator=operator)
    assert strategy == "guess_direct"
    np.testing.assert_array_equal(optimizer.to_dense(), before)
    expected = np.zeros(8, dtype=complex)
    expected[5] = 1
    np.testing.assert_allclose(guess.to_dense().reshape(-1), expected, atol=1e-12)


@pytest.mark.parametrize("mode", ["dmrg2", "dmrg3"])
def test_native_auto_guess_uses_graded_direct_replay(mode):
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                           occupations=((1, 1), (0, 0), (1, 1), (0, 0)))
    gates = [(fermion.hopping_gate(.17, t=1., imaginary=False), (0, 3))]
    exact = TreeOptimizer(gates, state=state, mode="direct", chi=16, cutoff=0., run=False)
    exact.run()
    fitted = TreeOptimizer(gates, state=state, mode=mode, chi=16, cutoff=0., run=False).copy()
    fitted.run()
    assert fitted.get_fit_diagnostics()["fit_init_strategy_requested"] == "auto"
    assert fitted.get_fit_diagnostics()["fit_init_strategy"] == "guess_direct"
    assert float(pepsy.tensors.tn_fidelity(fitted.tn, exact.tn)) > 1 - 1e-10
    unsupported = TreeOptimizer(gates, state=state, mode=mode, chi=2, cutoff=0.,
                                fit_init_strategy="guess-src", run=False)
    with pytest.raises(NotImplementedError, match="dense tree tensors only"):
        unsupported.run()
