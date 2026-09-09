"""Exact local solves and opt-in tree FIT execution policies."""

import autoray as ar
import numpy as np
import pytest

import pepsy
from pepsy.fitting import TreeFIT
from pepsy.optimizers.tree import TreeOptimizer, TreePlan, TreeTensorNetwork


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
def test_single_node_gate_exact_without_guess_replay(mode, monkeypatch):
    plan = TreePlan.from_order(range(1, 6), root_qubit=0, structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=41)
    state.exponent = 2.0
    gate = np.array([[.3, .2j], [.1, .8]], dtype=complex)
    options = dict(state=state, chi=2, cutoff=0., mode=mode,
                   fit_rtol=None, seed=89, run=False)
    fast = TreeOptimizer(None, **options)
    reference = TreeOptimizer(None, **options, fit_single_node_fast_path=False)
    reference.apply_gate(gate, (0,), track_norm=False)
    updates = []
    update = TreeFIT.fit_block

    def counted(fit, block, **kwargs):
        updates.append(tuple(block))
        return update(fit, block, **kwargs)

    apply = TreeOptimizer.apply_subtreempo

    def no_guess(optimizer, *args, **kwargs):
        assert optimizer is fast, "one-node FIT must not replay a compressed guess"
        return apply(optimizer, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TreeFIT, "fit_block", counted)
        patch.setattr(TreeOptimizer, "apply_subtreempo", no_guess)
        fast.apply_gate(gate, (0,), track_norm=False)
    expected = (gate @ state.to_dense().reshape(2, -1)).reshape(-1)
    np.testing.assert_allclose(fast.to_dense().reshape(-1), expected, atol=1e-10)
    np.testing.assert_allclose(fast.to_dense(), reference.to_dense(), atol=1e-10)
    assert len(updates) == 1
    diagnostic = fast.get_fit_diagnostics()
    assert diagnostic["iterations"] == 1
    assert diagnostic["convergence_reason"] == "single_node_exact"
    assert diagnostic["guess_used"] is False
    assert diagnostic["guess_backend"] == "single_node"
    assert diagnostic["block_size_trace"] == (1,)
    assert fast.rng.bit_generator.state == reference.rng.bit_generator.state
    fast.tn.validate(check_canonical=True)


def test_single_node_projection_is_exact_for_fixed_exterior(monkeypatch):
    plan = TreePlan.from_order(range(5), structure="balanced")
    target = TreeTensorNetwork.rand(plan, D=3, seed=42)
    guess = TreeTensorNetwork.rand(plan, D=2, seed=43)
    region = (plan.node_of_qubit[0],)
    fast = TreeFIT(target, guess, finite_check=True)
    reference = TreeFIT(target, guess)
    scans = []
    scan = fast._check_finite
    monkeypatch.setattr(fast, "_check_finite", lambda r: (scans.append(r), scan(r)))
    with pytest.warns(RuntimeWarning, match="finite_check"):
        fast.run_gate(region, n_iter=6, verbose=True)
    reference.run_gate(region, n_iter=6, single_node_fast_path=False)
    np.testing.assert_allclose(fast.p.to_dense(), reference.p.to_dense(), atol=1e-10)
    assert len(scans) == len(fast.fidelity_trace) == len(fast.local_norm_trace) == 1
    assert fast.last_relative_change is None
    assert fast.fit_diagnostics(overlap=True)["target_fidelity"] < 1 - 1e-5
    fast.p.validate(check_canonical=True)
    # A block-size-one sweep over several nodes is still an iterative solve.
    reference.run_gate(reference.nodes, n_iter=3, block_size=1)
    assert reference.iterations_run == 3
    assert reference.convergence_reason == "max_iter"


def test_complete_one_node_tree_preserves_standalone_iteration_default():
    state = pepsy.ps_to_ttn(1)
    fit = TreeFIT(state, state)
    fit.run(3)
    assert fit.iterations_run == 3
    fit.run_eff(3, single_node_fast_path=True)
    assert fit.iterations_run == 1
    assert fit.convergence_reason == "single_node_exact"


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_depth_first_preserves_blocks_and_reduces_tree_travel(block_size):
    plan = TreePlan.from_order(range(1, 16), structure="balanced", root_qubit=0)
    state = pepsy.ps_to_ttn(16, tree=plan)
    legacy = TreeFIT(state, state)
    branch = TreeFIT(state, state, traversal="depth-first")
    region = branch.nodes
    old = legacy._sweep_blocks(region, block_size, "out")
    new = branch._sweep_blocks(frozenset(region), block_size, "out")
    assert set(new) == set(old)
    assert len(new) == len(set(new))
    assert new[::-1] == branch._sweep_blocks(frozenset(region), block_size, "in")
    # Count real canonical edge moves, including overlaps between blocks.
    def travel(fit):
        edges = []
        move = fit.p.shift_orthogonality_center

        def counted(node, *args, **kwargs):
            start = fit.p.orthogonality_center
            if start is not None:
                edges.extend(plan.node_path(start, node)[1:])
            return move(node, *args, **kwargs)

        fit.p.shift_orthogonality_center = counted
        fit.run_eff(1, block_size=block_size)
        fit.p.validate(check_canonical=True)
        return len(edges)

    assert travel(branch) < travel(legacy)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_native_blockwise_fit_matches_default_without_global_dispatch(backend):
    pytest.importorskip("symmray")
    convert = None
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda x: torch.as_tensor(x, dtype=torch.complex128)
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128",
                           to_backend=convert)
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion, to_backend=convert,
                            occupations=((1, 1), (0, 0), (1, 1), (0, 0)))
    gate = fermion.hopping_gate(.17, t=1., imaginary=False)
    optimizer = TreeOptimizer(None, state=state, chi=8, cutoff=0., run=False)
    target, operator = optimizer._build_tree_fit_target(gate, (0, 3))
    if convert is not None:
        for network in (target, *operator.tree_networks):
            for tensor in network.tensors:
                data = tensor.data.copy()
                data.apply_to_arrays(convert)
                tensor.modify(data=data)
    optimizer.apply_subtreempo(operator)
    default_dispatch = ar.get_lib_fn("symmray", "tensordot")
    old = TreeFIT(target, optimizer.tn, max_bond=8, cutoffs=0.)
    new = TreeFIT(target, optimizer.tn, max_bond=8, cutoffs=0.,
                  environment_strategy="native-blockwise")
    # Full schedule checks message invalidation, effective tensors, graded
    # boundary phases, and splits with the alternative contractor together.
    for fit in (old, new):
        fit.run_eff(4, block_size=3, adaptive_block_sweeps=2,
                    two_site_transition_sweeps=1)
        fit.p.validate(check_canonical=True)
        assert all(ar.infer_backend(block) == backend
                   for t in fit.p.tensors for block in t.data.blocks.values())
    assert ar.get_lib_fn("symmray", "tensordot") is default_dispatch
    np.testing.assert_allclose(ar.to_numpy(new.p.to_dense().to_dense()),
                               ar.to_numpy(old.p.to_dense().to_dense()), atol=1e-10)
    np.testing.assert_allclose(new.local_norm_trace, old.local_norm_trace, atol=1e-10)
    assert new.fit_diagnostics(overlap=True)["target_fidelity"] == pytest.approx(
        old.fit_diagnostics(overlap=True)["target_fidelity"], abs=1e-10,
    )


@pytest.mark.parametrize("traversal,expected", [
    ("depth_first", "depth-first"), ("depth", "depth"),
])
def test_fit_execution_options_validate_and_survive_optimizer_copy(traversal, expected):
    state = pepsy.ps_to_ttn(3)
    with pytest.raises(TypeError, match="native Symmray"):
        TreeFIT(state, state, environment_strategy="native-blockwise")
    with pytest.raises(ValueError, match="traversal"):
        TreeFIT(state, state, traversal="chain")
    with pytest.raises(ValueError, match="environment_strategy"):
        TreeFIT(state, state, environment_strategy="invalid")
    optimizer = TreeOptimizer(None, state=state, run=False,
                              fit_traversal=traversal,
                              fit_single_node_fast_path=False)
    copied = optimizer.copy()
    assert copied.fit_traversal == expected
    assert copied.fit_environment_strategy == "default"
    assert copied.fit_single_node_fast_path is False


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
def test_default_depth_first_src_lossless_gate_replay(mode):
    plan = TreePlan.from_order(range(1, 6), root_qubit=0, structure="balanced",
                               max_arity=3)
    state = TreeTensorNetwork.rand(plan, D=2, seed=10)
    matrix = np.random.default_rng(10).normal(size=(8, 8))
    gate = np.linalg.qr(matrix)[0].astype(complex)
    stream = [(gate, (0, 3, 5)), (gate, (1, 2, 4))]
    exact = TreeOptimizer(stream, state=state, chi=64, cutoff=0., mode="direct")
    fitted = TreeOptimizer(stream, state=state, chi=64, cutoff=0., mode=mode)
    np.testing.assert_allclose(fitted.to_dense(), exact.to_dense(), atol=1e-10)
    assert fitted.get_fit_diagnostics()["traversal"] == "depth-first"
    assert fitted.get_fit_diagnostics()["fit_init_strategy_requested"] == "auto"
    assert fitted.get_fit_diagnostics()["fit_init_strategy"] == "guess_src"
    assert fitted.get_fit_diagnostics()["guess_used"] is True
    assert fitted.copy().fit_traversal == "depth-first"
    fitted.tn.validate(check_canonical=True)


@pytest.mark.parametrize("environment", ["default", "native-blockwise"])
def test_native_single_node_gate_exact(environment):
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    state = pepsy.ps_to_ttn(4, fermion=fermion,
                            occupations=((1, 1), (0, 0), (1, 1), (0, 0)))
    prepared = TreeOptimizer([(fermion.hopping_gate(.17, t=1., imaginary=False),
                               (0, 3))], state=state, chi=16, cutoff=0.)
    gate = fermion.onsite_gate(.2, site=3, U=1., mu=.1, imaginary=True)
    stream = [(gate, (3,))]
    exact = TreeOptimizer(stream, state=prepared.tn, chi=16, cutoff=0., mode="direct")
    fitted = TreeOptimizer(stream, state=prepared.tn, chi=16, cutoff=0., mode="dmrg3",
                           fit_environment_strategy=environment)
    assert float(pepsy.tensors.tn_fidelity(fitted.tn, exact.tn)) > 1 - 1e-10
    assert fitted.tn.norm() == pytest.approx(exact.tn.norm(), abs=1e-10)
    assert fitted.get_fit_diagnostics()["convergence_reason"] == "single_node_exact"
    fitted.tn.validate(check_canonical=True)


def test_native_contractor_checks_capability(monkeypatch):
    import cotengra as ctg
    from pepsy.fitting.tree import _native_environment_implementation

    pytest.importorskip("symmray")
    _native_environment_implementation.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(ctg.ContractionTree, "get_contractor", lambda self: None)
        with pytest.raises(NotImplementedError, match="per-contraction"):
            _native_environment_implementation()
    # A failed capability check must not poison subsequent valid fits.
    assert len(_native_environment_implementation()) == 2
