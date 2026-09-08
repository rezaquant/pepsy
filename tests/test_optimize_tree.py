"""Tests for the tree-tensor-network gate simulator (:class:`TreeOptimizer`)."""

import inspect
import sys
import types

import numpy as np
import pytest
import quimb.tensor as qtn
import pepsy

from pepsy.optimizers.tree import (
    TreeLayoutFinder,
    TreeMPO,
    TreeOptimizer,
    TreePlan,
    TreeTensorNetwork,
)
from pepsy.optimizers.tree.optimizer import _contract_two_tensors
from pepsy.optimizers.tree.ttn import _native_qr_block_scaled
from pepsy.fitting import TreeFIT


def test_tree_map_mode_is_shared_by_plan_state_and_native_operator():
    plan = TreePlan.from_order(
        range(8),
        map_mode="coarse-alternate-x",
    )
    state = TreeTensorNetwork.from_plan(plan)
    operator = TreeMPO.from_terms(
        plan,
        {(0,): np.diag([1.0, -1.0])},
        compress=False,
    )

    assert plan.map_mode == "coarse-alternate-x"
    assert state.map_mode == "coarse-alternate-x"
    assert operator.map_mode == "coarse-alternate-x"


def test_tree_coarse_map_mode_is_available_without_a_lattice_shape():
    plan = TreeLayoutFinder(
        [],
        n=8,
        map_mode="coarse-alternate-x",
    ).run(refine=None, search=None)

    assert plan.map_mode == "coarse-alternate-x"
    assert plan.mpo_order() == tuple(range(8))


def test_tree_mpo_higher_order_term_routes_and_replays_natively():
    """A dense higher-order TreeMPO applies without chain-MPO lowering."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    rng = np.random.default_rng(42)
    term = rng.normal(size=(2,) * 8) + 1j * rng.normal(size=(2,) * 8)
    operator = TreeMPO.from_terms(
        plan,
        {(3, 0, 2, 1): term},
        compress=True,
        cutoff=0.0,
    )

    assert operator.max_bond() > 1
    assert operator.tree_network.pepsy_tree_operator_is_ttno is True
    support = (3, 0, 2, 1)
    order = tuple(sorted(range(4), key=support.__getitem__))
    reference = term.transpose(order + tuple(axis + 4 for axis in order))
    np.testing.assert_allclose(operator.to_dense(), reference.reshape(16, 16))
    operator.canonicalize(center=plan.root)
    assert operator.is_canonical_form()
    assert operator.validate(check_canonical=True) is operator
    operator.compress(
        max_bond=None,
        cutoff=0.0,
    )
    assert operator.validate() is operator
    np.testing.assert_allclose(operator.to_dense(), reference.reshape(16, 16))

    initial = np.zeros(16, dtype=complex)
    initial[0] = 1.0
    optimizer = TreeOptimizer(
        None,
        n=4,
        tree=plan,
        chi=None,
        cutoff=0.0,
        run=False,
    )
    optimizer.apply_subtreempo(operator, track_norm=False)

    np.testing.assert_allclose(
        optimizer.to_dense(),
        operator.to_dense() @ initial,
        rtol=1e-11,
        atol=1e-11,
    )
    assert optimizer.tn.is_canonical_form()
    assert optimizer.tn.validate_isometry_metadata() is optimizer.tn

    streamed = TreeOptimizer(
        [TreeOptimizer.subtreempo_event(operator)],
        n=4,
        tree=plan,
        chi=None,
        cutoff=0.0,
        run=False,
    )
    streamed.run()
    np.testing.assert_allclose(
        streamed.to_dense(),
        optimizer.to_dense(),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("compression_mode", ("sdc", "src"))
def test_tree_successive_compression_modes_are_reproducible(compression_mode):
    """Tree compression modes preserve the tree sweep and randomized seed API."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    kwargs = dict(
        n=5,
        tree=plan,
        mode=compression_mode,
        chi=1,
        cutoff=0.0,
        compression_seed=23,
        track_infidelity=False,
        run=False,
    )
    first = TreeOptimizer(None, **kwargs)
    second = TreeOptimizer(None, **kwargs)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    first.apply_gate(cnot, (0, 4), track_norm=False)
    second.apply_gate(cnot, (0, 4), track_norm=False)

    assert first.mode == "auto"
    assert first.compression_mode == compression_mode
    assert first.tn.max_bond() <= 1
    assert first.tn.validate(check_canonical=True) is first.tn
    np.testing.assert_allclose(first.to_dense(), second.to_dense())


@pytest.mark.parametrize(
    ("mode", "fit_n_iter", "expected_block_size"),
    (("dmrg1", 3, 2), ("dmrg2", 1, 2), ("dmrg3", 1, 3)),
)
def test_tree_optimizer_dmrg_uses_tree_fit_engine(
    mode, fit_n_iter, expected_block_size
):
    """TreeOptimizer DMRG aliases use cached tree-local FIT updates."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode=mode,
        chi=2,
        cutoff=0.0,
        fit_n_iter=fit_n_iter,
        fit_init_strategy="guess-src",
        fit_overlap_diagnostics=True,
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_gate(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
        track_norm=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["backend"] == "tree_fit"
    assert diagnostics["block_size"] == expected_block_size
    assert diagnostics["requested_block_size"] == expected_block_size
    if mode == "dmrg1":
        assert diagnostics["block_size_trace"] == (2, 2, 1)
        assert diagnostics["adaptive_sweeps"] == 2
        assert diagnostics["one_site_refinement_sweeps"] == 1
    assert diagnostics["guess_backend"] == "tree_mpo"
    assert diagnostics["target_layout"] == "layered"
    assert diagnostics["cache"]["hits"] > 0
    assert diagnostics["local_fidelity"] > 1.0 - 1.0e-10
    assert optimizer._active_update is None
    assert optimizer.tn.validate(check_canonical=True) is optimizer.tn


def test_tree_optimizer_generic_dmrg_warmup_then_refinement():
    """Generic tree DMRG uses the MPS-style two-site handoff."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=4,
        fit_adaptive_sweeps=2,
        fit_init_strategy="guess-src",
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_gate(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
        track_norm=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["block_size_trace"] == (2, 2, 1, 1)
    assert diagnostics["adaptive_sweeps"] == 2
    assert diagnostics["one_site_refinement_sweeps"] == 2


def test_tree_optimizer_explicit_dmrg_subtreempo_finishes_norm_event():
    """Explicit TreeMPO DMRG updates close the optimizer norm transaction."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    gate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    operator = TreeMPO.from_gate(plan, gate, (0, 4), cutoff=0.0)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=1,
        fit_init_strategy="direct",
        track_infidelity=True,
        run=False,
    )

    optimizer.apply_subtreempo(operator, (0, 4))

    assert optimizer._active_update is None
    assert len(optimizer.get_norm_events()) == 1
    assert optimizer.get_norm_events()[0]["kind"] == "subtreempo"


def test_tree_optimizer_guess_src_uses_fit_init_seed(monkeypatch):
    """TreeMPO disposable guesses use the FIT-specific randomized seed."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        compression_seed=11,
        fit_init_seed=37,
        fit_init_strategy="guess-src",
        track_infidelity=False,
        run=False,
    )
    target, operator = optimizer._build_tree_fit_target(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
    )
    nodes = [plan.node_of_qubit[0], plan.node_of_qubit[4]]
    region = frozenset(optimizer.tn.steiner_nodes(nodes))
    observed = []
    original = TreeOptimizer.apply_subtreempo

    def capture_seed(self, *args, **kwargs):
        observed.append(self.compression_seed)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TreeOptimizer, "apply_subtreempo", capture_seed)
    optimizer._tree_fit_initial_guess(target, region, operator=operator)

    assert observed == [37]


def test_tree_optimizer_dmrg_target_keeps_operator_and_state_layers():
    """Tree DMRG builds a non-fused operator--state target for TreeFIT."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        track_infidelity=False,
        run=False,
    )
    target, _operator = optimizer._build_tree_fit_target(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
    )

    assert len(target.tensors) == 2 * len(optimizer.tn.tensors)
    fit = TreeFIT(target, optimizer.tn.copy(), max_bond=2, cutoffs=0.0)
    assert fit.target_layout == "layered"
    assert all(len(group) == 2 for group in fit._target_tensors.values())


@pytest.mark.parametrize("strategy", ("random", "random_expand"))
def test_tree_optimizer_dmrg_random_fit_guess_is_seeded(strategy):
    """Tree DMRG exposes the same disposable randomized-guess policy as MPS."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    kwargs = dict(
        n=5,
        tree=plan,
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=1,
        fit_init_strategy=strategy,
        fit_init_rand_strength=1.0e-4,
        fit_init_seed=17,
        track_infidelity=False,
        run=False,
    )
    first = TreeOptimizer(None, **kwargs)
    second = TreeOptimizer(None, **kwargs)
    gate = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    first.apply_gate(gate, (0, 4), track_norm=False)
    second.apply_gate(gate, (0, 4), track_norm=False)

    diagnostics = first.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy"] == strategy
    assert diagnostics["random_initialization"] is True
    assert diagnostics["random_initialization_info"]["enabled"] is True
    np.testing.assert_allclose(first.to_dense(), second.to_dense())
    assert first.tn.validate(check_canonical=True) is first.tn


def test_tree_fit_environment_cache_reuses_untouched_branches():
    """TreeFIT caches directed branch overlaps between local updates."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    initial = TreeTensorNetwork.from_plan(plan)
    target_optimizer = TreeOptimizer(
        None,
        n=5,
        tree=plan,
        chi=None,
        cutoff=0.0,
        run=False,
        tn=initial,
    )
    target_optimizer.apply_gate(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        (0, 4),
        track_norm=False,
    )

    fit = TreeFIT(target_optimizer.tn, initial, max_bond=2, cutoffs=0.0)
    region = initial.steiner_nodes(
        [plan.node_of_qubit[0], plan.node_of_qubit[4]]
    )
    fit.run_gate(region, n_iter=1, block_size=2)
    diagnostics = fit.fit_diagnostics(overlap=True)

    assert diagnostics["cache"]["messages"] > 0
    assert diagnostics["cache"]["hits"] > 0
    assert diagnostics["local_fidelity"] > 1.0 - 1.0e-10


def test_tree_fit_invalidates_effective_cache_through_branch_dependencies():
    """A disjoint effective block is invalidated when its exterior message changes."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=2, seed=51, canonicalize=True)
    state = TreeTensorNetwork.rand(plan, D=2, seed=52, canonicalize=True)
    fit = TreeFIT(target, state, max_bond=2, cutoffs=0.0)
    changed = plan.node_of_qubit[0]
    disjoint = plan.node_of_qubit[4]

    fit._canonicalize_for_block((disjoint,), disjoint)
    fit._effective_block((disjoint,))
    assert (disjoint,) in fit._effective_cache

    fit.fit_block((changed,))

    assert (disjoint,) not in fit._effective_cache


def test_tree_fit_overlap_respects_represented_exponents():
    """Fidelity divides represented target/state scale before reporting."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = state.copy()
    state.exponent = 3.0
    target.exponent = 3.0

    diagnostics = TreeFIT(
        target,
        state,
        max_bond=1,
        cutoffs=0.0,
    ).fit_diagnostics(overlap=True)

    assert diagnostics["local_fidelity"] == pytest.approx(1.0)


def test_tree_fit_run_api_matches_fit_positional_controls_and_verbose_trace():
    """TreeFIT keeps FIT's run/run_eff positional controls and trace behavior."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=2, seed=33)
    fit = TreeFIT(target, target.copy(), max_bond=2, cutoffs=0.0)

    fit.run(1, True)

    assert len(fit.fidelity_trace) == 1
    diagnostics = fit.fit_diagnostics(overlap=True)
    assert diagnostics["local_fidelity"] > 1.0 - 1.0e-10
    assert diagnostics["target_fidelity"] > 1.0 - 1.0e-10
    assert diagnostics["fit_overlap_fidelity"] == pytest.approx(
        diagnostics["target_fidelity"]
    )
    fit.run_eff(1, True)
    assert len(fit.fidelity_trace) == 1


def test_tree_fit_local_norm_diagnostics_do_not_contract_full_state(monkeypatch):
    """TreeFIT local fidelity uses one terminal centre tensor per sweep."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    target = TreeTensorNetwork.rand(plan, D=2, seed=330, canonicalize=True)
    fit = TreeFIT(target, target.copy(), max_bond=2, cutoffs=0.0)

    def fail_full_path(*args, **kwargs):
        raise AssertionError("routine TreeFIT diagnostics used a full contraction")

    monkeypatch.setattr(fit, "_network_norm", fail_full_path)
    monkeypatch.setattr(fit, "_global_overlap", fail_full_path)
    monkeypatch.setattr(fit.tn, "norm", fail_full_path)

    fit.run_eff(n_iter=2, verbose=True, block_size=1, rtol=0.0)
    diagnostics = fit.fit_diagnostics()

    assert len(fit.local_norm_trace) == 2
    assert len(fit.local_norm_stripped_trace) == 2
    assert len(fit.sweep_norm_trace) == 2
    assert len(fit.fidelity_trace) == 2
    assert diagnostics["local_fidelity"] == pytest.approx(1.0)
    assert "target_fidelity" not in diagnostics


@pytest.mark.parametrize("block_size", (2, 3))
def test_tree_fit_adaptive_block_warmup_then_one_site_refinement(block_size):
    """TreeFIT mirrors FIT's larger-block warm-up schedule."""

    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    initial = TreeTensorNetwork.from_plan(plan)
    target = TreeTensorNetwork.rand(plan, D=2, seed=34)
    fit = TreeFIT(target, initial, max_bond=2, cutoffs=0.0)

    fit.run_eff(
        n_iter=4,
        block_size=block_size,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
    )

    assert fit.iterations_run == 4
    assert fit.adaptive_sweeps_run == 2
    assert fit.one_site_sweeps_run == 2
    assert fit.block_size_trace == [block_size, block_size, 1, 1]
    diagnostics = fit.fit_diagnostics()
    assert diagnostics["adaptive_sweeps"] == 2
    assert diagnostics["one_site_refinement_sweeps"] == 2


def test_tree_fit_retag_aligns_structural_tags_without_mutating_target():
    """FIT-style retagging is private, ordered, and preserves physical tags."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan, node_tag_id="N{}")
    target = TreeTensorNetwork.from_plan(plan, node_tag_id="T{}")
    original_tags = tuple(tuple(tensor.tags) for tensor in target.tensors)
    info = {}

    fit = TreeFIT(target, state, retag=True, info=info)

    assert info["retagged"] is True
    assert tuple(tuple(tensor.tags) for tensor in target.tensors) == original_tags
    assert all(
        fit.tn.node_tag(node) == state.node_tag(node)
        for node in plan.nodes()
    )
    assert fit.tn.validate() is fit.tn


def test_tree_fit_rejects_different_tree_topology():
    """Same node labels are insufficient when target and state edges differ."""

    state_plan = TreePlan(
        0,
        {0: (1, 2), 1: (3,), 2: (), 3: ()},
        {0: None, 1: 0, 2: 0, 3: 1},
        {2: 0, 3: 1},
    )
    target_plan = TreePlan(
        0,
        {0: (1, 3), 1: (2,), 2: (), 3: ()},
        {0: None, 1: 0, 2: 1, 3: 0},
        {2: 0, 3: 1},
    )
    state = TreeTensorNetwork.from_plan(state_plan)
    target = TreeTensorNetwork.from_plan(target_plan)

    with pytest.raises(ValueError, match="same tree topology"):
        TreeFIT(target, state)


def test_tree_fit_rejects_disconnected_public_regions():
    """TreePlan regions use the same connectivity contract as TreePepsPlan."""

    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    fit = TreeFIT(state.copy(), state)
    nodes = tuple(plan.nodes())
    disconnected = (nodes[0], nodes[-1])

    if fit.p.node_path(disconnected[0], disconnected[1]) == disconnected:
        pytest.skip("selected nodes happen to be adjacent in this plan")
    with pytest.raises(ValueError, match="connected"):
        fit.fit_block(disconnected)
    with pytest.raises(ValueError, match="connected"):
        fit.run_gate(disconnected, n_iter=1)


def test_tree_fit_accepts_correctly_tagged_layered_target():
    """TreeFIT contracts local target layers without dropping their tensors."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = state.copy()
    node = plan.node_of_qubit[0]
    backbone = target.node_tensor(node)
    layer_ind = "target_layer_bond"
    backbone.modify(
        data=np.expand_dims(backbone.data, -1),
        inds=backbone.inds + (layer_ind,),
    )
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=(layer_ind,),
        tags=backbone.tags,
    )
    parent, child = next(
        (parent, child)
        for parent, children in plan.children.items()
        for child in children
    )
    inter_node_layer = "target_inter_node_layer"
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=(inter_node_layer,),
        tags=target.node_tensor(parent).tags,
    )
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=(inter_node_layer,),
        tags=target.node_tensor(child).tags,
    )

    fit = TreeFIT(target, state, max_bond=1, cutoffs=0.0)

    assert fit.target_layout == "layered"
    assert len(fit._target_tensors[node]) == 2
    assert len(fit._target_bonds[(parent, child)]) == 2
    fit.run_eff(1)
    assert fit.fit_diagnostics(overlap=True)["local_fidelity"] == pytest.approx(1.0)


def test_tree_fit_rejects_untagged_layered_target():
    """Layer tensors without structural ownership are rejected clearly."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = state.copy()
    target |= qtn.Tensor(np.ones((1,)), inds=("unused_layer",))

    with pytest.raises(ValueError, match="exactly one structural node tag"):
        TreeFIT(target, state)


def test_tree_fit_accepts_plain_tagged_layered_tensor_network_target():
    """A tagged Quimb target can use the fitted tree as its geometry source."""

    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    target = qtn.TensorNetwork([tensor.copy() for tensor in state.tensors])
    node = plan.node_of_qubit[0]
    layer_tags = state.node_tensor(node).tags
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=("plain_layer_bond",),
        tags=layer_tags,
    )
    target |= qtn.Tensor(
        np.ones((1,)),
        inds=("plain_layer_bond",),
        tags=layer_tags,
    )

    fit = TreeFIT(target, state, max_bond=1, cutoffs=0.0)

    assert fit.target_layout == "layered"
    fit.run_eff(1)
    assert fit.fit_diagnostics(overlap=True)["local_fidelity"] == pytest.approx(1.0)
# -- exact statevector reference ----------------------------------------------


def _sv_apply_1q(psi, g, q, n):
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q).reshape(-1)


def _sv_apply_2q(psi, g, a, b, n):
    g = g.reshape(2, 2, 2, 2)
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=([2, 3], [a, b]))
    return np.moveaxis(psi, [0, 1], [a, b]).reshape(-1)


def _sv_apply_kq(psi, g, where, n):
    """Exact statevector application of a ``k``-qubit operator on ``where``."""
    k = len(where)
    g = np.asarray(g).reshape([2] * (2 * k))
    psi = psi.reshape([2] * n)
    psi = np.tensordot(g, psi, axes=(list(range(k, 2 * k)), list(where)))
    return np.moveaxis(psi, range(k), where).reshape(-1)


def _rand_unitary(k, rng):
    m = rng.standard_normal((2**k, 2**k)) + 1j * rng.standard_normal((2**k, 2**k))
    q, _ = np.linalg.qr(m)
    return q


def _random_stream(n, ngates, rng, two_qubit_frac=0.5):
    stream = []
    for _ in range(ngates):
        if n >= 2 and rng.random() < two_qubit_frac:
            a, b = rng.choice(n, size=2, replace=False)
            stream.append((_rand_unitary(2, rng), (int(a), int(b))))
        else:
            stream.append((_rand_unitary(1, rng), int(rng.integers(n))))
    return stream


def _two_branch_flip_submpo(*, L, sites, targets, w0=0.7, w1=0.3):
    """Return ``w0 * I + w1 * prod(X_targets)`` as a sparse-site MPO."""
    eye = np.eye(2, dtype=complex)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sites = tuple(sites)
    targets = set(targets)
    branch0 = [eye.copy() for _site in sites]
    branch1 = [flip.copy() if site in targets else eye.copy() for site in sites]
    branch0[0] *= w0
    branch1[0] *= w1
    mpo0 = qtn.MPO_product_operator(
        branch0,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    mpo1 = qtn.MPO_product_operator(
        branch1,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    return mpo0.add_MPO(mpo1)


def _exact_state(stream, n):
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    for g, where in stream:
        if isinstance(where, int):
            psi = _sv_apply_1q(psi, g, where, n)
        else:
            psi = _sv_apply_2q(psi, g, where[0], where[1], n)
    return psi


def _sv_expect(psi, op, where, n):
    """Exact ``<psi|op|psi>`` for a (multi-site) operator from the dense state."""
    psi = psi.reshape([2] * n)
    k = len(where)
    op = np.asarray(op).reshape([2] * (2 * k))
    o = np.tensordot(op, psi, axes=(list(range(k, 2 * k)), list(where)))
    o = np.moveaxis(o, range(k), where)
    return np.vdot(psi.reshape(-1), o.reshape(-1))


def _fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2 / (
        np.vdot(a, a).real * np.vdot(b, b).real
    )


# -- tests --------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_matches_statevector(n, seed):
    """Untruncated tree replay reproduces the exact statevector."""
    rng = np.random.default_rng(seed)
    stream = _random_stream(n, 8 * n, rng)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_tree_two_site_direct_and_mpo_modes_agree():
    """Dense direct threading and gate-to-MPO routing are equivalent."""
    rng = np.random.default_rng(918)
    n = 6
    stream = _random_stream(n, 24, rng)
    exact = _exact_state(stream, n)
    direct = TreeOptimizer(
        stream, n=n, chi=128, cutoff=0.0, mode="direct",
    )
    mpo = TreeOptimizer(
        stream, n=n, chi=128, cutoff=0.0, mode="mpo",
    )

    assert _fidelity(direct.to_dense(), mpo.to_dense()) > 1 - 1e-10
    assert _fidelity(direct.to_dense(), exact) > 1 - 1e-9
    assert _fidelity(mpo.to_dense(), exact) > 1 - 1e-9


def test_tree_cutoff_mode_controls_edge_truncation_and_copy():
    """Tree truncations honor the configured Quimb cutoff convention."""
    small = 0.1
    large = np.sqrt(1.0 - small**2)
    gate = np.array(
        [
            [large, 0.0, 0.0, -small],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [small, 0.0, 0.0, large],
        ],
        dtype=complex,
    )

    relative = TreeOptimizer(
        [(gate, (0, 1))],
        n=2,
        chi=2,
        cutoff=0.05,
        cutoff_mode="rel",
        track_truncation=True,
    )
    relative_sum2 = TreeOptimizer(
        [(gate, (0, 1))],
        n=2,
        chi=2,
        cutoff=0.05,
        cutoff_mode="rsum2",
        track_truncation=True,
    )

    assert relative.max_bond() == 2
    assert relative_sum2.max_bond() == 1
    assert all(
        event["cutoff_mode"] == "rsum2"
        for event in relative_sum2.truncation_history
    )
    assert relative_sum2.copy().cutoff_mode == "rsum2"


def test_tree_cutoff_defaults_are_dtype_aware():
    """TreeOptimizer defaults resolve from the default complex128 state."""
    opt = TreeOptimizer(None, n=2, run=False)

    assert opt.cutoff == pytest.approx(1e-12)
    assert opt.cutoff_mode == "rsum2"
    assert opt.copy().cutoff == pytest.approx(1e-12)
    assert opt.copy().cutoff_mode == "rsum2"


def test_tree_dm_compression_uses_the_fused_operator_state_network():
    """DM mode changes the local truncating decomposition, not routing."""
    plan = TreePlan.from_order(range(2), structure="balanced", top_arity=2)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    stream = [(hadamard, 0), (cnot, (0, 1))]

    direct = TreeOptimizer(
        stream,
        n=2,
        tree=plan,
        chi=1,
        cutoff=0.0,
        compression_mode="direct",
    )
    dm = TreeOptimizer(
        stream,
        n=2,
        tree=plan,
        chi=1,
        cutoff=0.0,
        compression_mode="dm",
    )

    np.testing.assert_allclose(direct.to_dense(), dm.to_dense(), atol=1e-10)
    assert dm.compression_mode == "dm"
    assert dm.copy().compression_mode == "dm"
    assert dm.tn.validate(check_canonical=True) is dm.tn


def test_tree_dm_mode_is_a_shorthand_for_direct_routing():
    opt = TreeOptimizer(None, n=2, mode="dm", run=False)

    assert opt.mode == "auto"
    assert opt.compression_mode == "dm"


@pytest.mark.parametrize(
    ("mode", "normalized", "compression"),
    [
        ("tree_mpo_direct", "tree_mpo_direct", "direct"),
        ("tree-mpo-dm", "tree_mpo_dm", "dm"),
        ("tree_mpo_dem", "tree_mpo_dm", "dm"),
        ("tree_mpo", "tree_mpo_direct", "direct"),
    ],
)
def test_tree_mpo_modes_normalize_to_explicit_tree_routes(
    mode, normalized, compression,
):
    """TreeMPO mode names retain both their route and compression contract."""
    opt = TreeOptimizer(None, n=2, mode=mode, run=False)

    assert opt.mode == normalized
    assert opt.compression_mode == compression


def test_tree_mpo_gate_modes_use_tree_mpo_not_chain_submpo(monkeypatch):
    """Named TreeMPO modes use the active TreePlan span, never a chain MPO."""
    rng = np.random.default_rng(921)
    support = (0, 3, 7)
    gate = _rand_unitary(len(support), rng)
    opt = TreeOptimizer(
        None,
        n=8,
        chi=64,
        cutoff=0.0,
        mode="tree_mpo_direct",
        profile=True,
        run=False,
    )
    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    def no_chain_mpo(*args, **kwargs):
        raise AssertionError("TreeMPO gate route used a chain sub-MPO")

    monkeypatch.setattr(opt, "apply_subtreempo", traced_apply_subtreempo)
    monkeypatch.setattr(qtn.MatrixProductOperator, "from_dense", no_chain_mpo)
    opt.apply_gate(gate, support)

    assert routed
    assert isinstance(routed[0], TreeMPO)
    path_events = [
        event.get("kind") == "gate_factorization"
        and event.get("route") == "treempo"
        for event in opt.profile_events
    ]
    assert any(path_events)
    metadata = [
        event for event in opt.profile_events
        if event.get("kind") == "metadata_path"
        and event.get("route") == "subtreempo"
    ]
    assert metadata
    assert metadata[-1]["support"] == support
    assert metadata[-1]["subtree_nodes"] == len(
        opt._steiner_nodes([opt.plan.node_of_qubit[q] for q in support])
    )
    assert opt.tn.validate(check_canonical=True) is opt.tn


@pytest.mark.parametrize("mode", ("auto", "direct", "dm", "sdc", "src", "zipup", "mpo", "dmrg2"))
def test_tree_ordinary_gate_modes_all_lower_to_subtreempo(monkeypatch, mode):
    """Every ordinary gate mode shares the TreeMPO active-region kernel."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        None,
        n=4,
        chi=8,
        cutoff=0.0,
        mode=mode,
        run=False,
    )
    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_subtreempo", traced_apply_subtreempo)
    opt.apply_gate(cnot, (0, 3))

    assert routed
    assert isinstance(routed[0], TreeMPO)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_gate_mode_uses_subtreempo_for_four_qubits(monkeypatch):
    """Ordinary dense gates enter the TreeMPO route, including four sites."""
    rng = np.random.default_rng(922)
    opt = TreeOptimizer(None, n=6, chi=64, cutoff=0.0, mode="direct", run=False)

    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_subtreempo", traced_apply_subtreempo)
    opt.apply_gate(_rand_unitary(4, rng), (0, 2, 4, 5))

    assert routed
    assert isinstance(routed[0], TreeMPO)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_gate_mode_routes_wider_dense_gate_through_tree_mpo(monkeypatch):
    """Wide ordinary gates use the TreeMPO route without a width cliff."""
    opt = TreeOptimizer(None, n=5, chi=8, cutoff=0.0, mode="direct", run=False)
    routed = []
    apply_subtreempo = opt.apply_subtreempo

    def traced_apply_subtreempo(tree_mpo, *args, **kwargs):
        routed.append(tree_mpo)
        return apply_subtreempo(tree_mpo, *args, **kwargs)

    monkeypatch.setattr(opt, "apply_subtreempo", traced_apply_subtreempo)
    opt.apply_gate(np.eye(32, dtype=complex), tuple(range(5)))

    assert routed
    assert isinstance(routed[0], TreeMPO)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_mpo_run_mode_updates_route_and_compression():
    """run(mode=...) persists the combined TreeMPO mode contract."""
    opt = TreeOptimizer(None, n=3, mode="direct", run=False)

    opt.run(mode="tree-mpo-dm")
    assert opt.mode == "tree_mpo_dm"
    assert opt.compression_mode == "dm"

    opt.run(mode="tree_mpo_direct")
    assert opt.mode == "tree_mpo_direct"
    assert opt.compression_mode == "direct"


def test_tree_mpo_dm_uses_density_matrix_compression_after_tree_routing():
    """The TreeMPO density-matrix mode reaches the tree compression core."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(hadamard, 0), (cnot, (0, 1))],
        n=2,
        chi=1,
        cutoff=0.0,
        mode="tree_mpo_dm",
        track_truncation=True,
    )

    assert opt.compression_mode == "dm"
    assert any(event["kind"] == "compress" for event in opt.truncation_history)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_path_thread_preserves_qr_isometry_metadata(monkeypatch):
    """Every dense path-thread Q keeps its toward-destination isometry."""
    rng = np.random.default_rng(919)
    opt = TreeOptimizer(None, n=8, chi=16, run=False)
    checked = []
    compress_path = opt._compress_path

    def check_then_compress(path, **kwargs):
        for node, toward_destination in zip(path, path[1:]):
            tensor = opt.tn.node_tensor(node)
            bond = opt.tn.bond(node, toward_destination)
            assert tensor.left_inds is not None
            assert set(tensor.left_inds) == set(tensor.inds) - {bond}
            checked.append(node)
        return compress_path(path, **kwargs)

    monkeypatch.setattr(opt, "_compress_path", check_then_compress)
    opt.apply_2q(_rand_unitary(2, rng), 0, 7)

    assert checked
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_thread_hops_use_explicitly_lossless_qr(monkeypatch):
    """Geodesic threading never inherits Quimb's truncating cutoff default."""
    import quimb.tensor.tensor_core as qtc

    rng = np.random.default_rng(9191)
    calls = []
    tensor_split = qtc.tensor_split

    def traced_tensor_split(*args, **kwargs):
        if kwargs.get("method") == "qr":
            calls.append(dict(kwargs))
        return tensor_split(*args, **kwargs)

    monkeypatch.setattr(qtc, "tensor_split", traced_tensor_split)
    opt = TreeOptimizer(
        None, n=8, chi=64, cutoff=1e-10, mode="direct", run=False,
    )
    opt.apply_2q(_rand_unitary(2, rng), 0, 7)

    assert calls
    assert all(call["cutoff"] == 0.0 for call in calls)


@pytest.mark.parametrize("mode", ("direct", "mpo", "submpo"))
def test_two_site_modes_reuse_path_isometries_for_compression(
    mode, monkeypatch,
):
    """All two-site routes skip the QR already proven by ``left_inds``."""
    rng = np.random.default_rng(920)
    n = 8
    where = (0, 7)
    gate = _rand_unitary(2, rng)
    opt = TreeOptimizer(
        None, n=n, chi=16, cutoff=1e-12, mode=mode, run=False,
    )

    reductions = []
    compress_edge = opt._compress_edge_with_diagnostics

    def traced_compress_edge(
        u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False,
    ):
        assert opt.tn.can_skip_canonize(u, v, absorb="left")
        reductions.append(reduced)
        return compress_edge(
            u,
            v,
            max_bond=max_bond,
            cutoff=cutoff,
            reduced=reduced,
            reduction_proven=reduction_proven,
        )

    monkeypatch.setattr(
        opt, "_compress_edge_with_diagnostics", traced_compress_edge,
    )
    if mode == "submpo":
        submpo = qtn.MatrixProductOperator.from_dense(
            gate.reshape((2,) * 4),
            dims=(2, 2),
            sites=where,
            L=n,
            max_bond=None,
            cutoff=0.0,
        )
        opt.apply_submpo(submpo, where)
    else:
        opt.apply_2q(gate, *where)

    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    expected = _sv_apply_kq(expected, gate, where, n)
    assert reductions
    assert all(reduced == "left" for reduced in reductions)
    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_lossless_path_skips_reduction_proof_lookup(monkeypatch):
    """A QR-only path does not query truncating-compression metadata."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(
        None, n=6, chi=64, cutoff=0.0, mode="direct", run=False,
    )

    def fail_reduction_lookup(*args, **kwargs):
        raise AssertionError("lossless path queried truncation metadata")

    monkeypatch.setattr(
        opt, "_metadata_aware_reduction", fail_reduction_lookup,
    )
    opt.apply_2q(np.kron(x, x), 0, 5)

    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_path_one_sided_compression_matches_full_reduction(monkeypatch):
    """Reusing routed Q tensors is exact even when the path truncates."""
    rng = np.random.default_rng(921)
    n = 8
    plan = TreePlan.from_order(range(n), structure="balanced")
    seed = TreeTensorNetwork.rand(plan, D=2, seed=921)
    optimized = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        mode="direct",
        run=False,
    )
    reference = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        mode="direct",
        run=False,
    )
    monkeypatch.setattr(
        reference, "_metadata_aware_reduction", lambda _u, _v: True,
    )

    for where in ((0, 7), (1, 6), (2, 5), (0, 4)):
        gate = _rand_unitary(2, rng)
        optimized.apply_2q(gate, *where)
        reference.apply_2q(gate, *where)

    assert optimized.max_bond() <= 2
    assert reference.max_bond() <= 2
    assert _fidelity(optimized.to_dense(), reference.to_dense()) > 1 - 1e-10
    assert optimized.tn.validate(check_canonical=True) is optimized.tn
    assert reference.tn.validate(check_canonical=True) is reference.tn


def test_compression_reduction_falls_back_without_local_isometry_proof():
    """One-sided compression is selected only from live ``left_inds``."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, tree=plan, run=False)
    child = plan.leaf_of_qubit[0]
    parent = plan.parent[child]

    assert opt._metadata_aware_reduction(parent, child) == "left"
    opt.tn.node_tensor(child).modify(left_inds=None)
    assert opt._metadata_aware_reduction(parent, child) is True


def test_tree_mpo_mode_keeps_small_operator_schmidt_components():
    """MPO lowering must not apply Quimb's default gate-SVD cutoff."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    theta = 1.0e-7
    gate = (
        np.cos(theta) * np.eye(4, dtype=complex)
        - 1.0j * np.sin(theta) * np.kron(x, x)
    )
    direct = TreeOptimizer(
        [(gate, (0, 1))], n=2, chi=4, cutoff=0.0, mode="direct",
    )
    mpo = TreeOptimizer(
        [(gate, (0, 1))], n=2, chi=4, cutoff=0.0, mode="mpo",
    )
    expected = np.array([np.cos(theta), 0.0, 0.0, -1.0j * np.sin(theta)])

    np.testing.assert_allclose(direct.to_dense(), expected, atol=1e-13)
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=1e-13)
    np.testing.assert_allclose(mpo.to_dense(), direct.to_dense(), atol=1e-13)


def test_tree_mpo_and_direct_share_path_compression_diagnostics():
    """Two-site MPO mode uses the same routed-factor kernel as direct mode."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0],
         [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex,
    )
    direct = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=1, cutoff=0.0, mode="direct",
        track_truncation=True,
    )
    mpo = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=1, cutoff=0.0, mode="mpo",
        track_truncation=True,
    )

    fields = ("kind", "edge", "before_bond", "after_bond", "max_bond", "cutoff")
    direct_trace = [
        tuple(event[field] for field in fields)
        for event in direct.truncation_history
    ]
    mpo_trace = [
        tuple(event[field] for field in fields)
        for event in mpo.truncation_history
    ]
    assert mpo_trace == direct_trace
    assert all(event["max_bond"] == 1 for event in mpo.truncation_history)


def test_tree_multisite_submpo_qr_routes_before_one_subtree_sweep():
    """A 3-site MPO transports its virtual legs without routing SVDs."""
    gate = _rand_unitary(3, np.random.default_rng(51))
    mpo = qtn.MatrixProductOperator.from_dense(
        gate.reshape((2,) * 6),
        dims=(2, 2, 2),
        sites=(0, 2, 4),
        L=5,
        max_bond=None,
        cutoff=0.0,
    )
    opt = TreeOptimizer(
        None, n=5, chi=1, cutoff=0.0, track_truncation=True, run=False,
    )
    opt.apply_submpo(mpo, (0, 2, 4))

    assert opt.truncation_history
    assert all(event["kind"] != "split" for event in opt.truncation_history)
    assert all(event["max_bond"] == 1 for event in opt.truncation_history)


def test_tree_submpo_does_not_retruncate_existing_within_cap_bonds():
    """A native MPO replay keeps tiny pre-existing state components."""
    eps = 1e-8
    large = np.sqrt(1.0 - eps**2)
    operator = np.zeros((4, 4), dtype=complex)
    operator[:, 0] = (large, 0.0, 0.0, eps)
    plan = TreePlan.from_order(range(3), structure="balanced")

    seed = TreeOptimizer(
        None, n=3, tree=plan, chi=4, cutoff=0.0, mode="direct", run=False,
    )
    seed.apply_2q(operator, 0, 1)
    expected = seed.to_dense()

    identity = qtn.MatrixProductOperator.from_dense(
        np.eye(8, dtype=complex).reshape((2,) * 6),
        dims=(2, 2, 2),
        sites=(0, 1, 2),
        L=3,
        max_bond=None,
        cutoff=0.0,
    )
    replay = TreeOptimizer(
        None,
        n=3,
        tree=plan,
        state=seed.tn,
        chi=4,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        mode="submpo",
        run=False,
    )
    replay.apply_submpo(identity, (0, 1, 2))

    np.testing.assert_allclose(expected, replay.to_dense(), atol=1e-14)
    assert replay.to_dense()[6] == pytest.approx(eps)

    mpo_replay = TreeOptimizer(
        None,
        n=3,
        tree=plan,
        state=seed.tn,
        chi=4,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        mode="mpo",
        run=False,
    )
    mpo_replay.apply_2q(np.eye(4, dtype=complex), 0, 1)
    np.testing.assert_allclose(expected, mpo_replay.to_dense(), atol=1e-14)


def test_dense_subtree_hub_recovery_reuses_routed_q_metadata(monkeypatch):
    """Dense routed Q tensors recover the hub without another numerical QR."""
    import quimb.tensor.tensor_core as qtc

    rng = np.random.default_rng(52)
    n = 8
    where = (0, 3, 7)
    gate = _rand_unitary(3, rng)
    opt = TreeOptimizer(None, n=n, chi=16, run=False)
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    expected = _sv_apply_kq(expected, gate, where, n)

    qr_calls = []
    tensor_split = qtc.tensor_split
    canonize_calls = []
    canonize_between = opt.tn.canonize_between

    def traced_tensor_split(*args, **kwargs):
        if kwargs.get("method") == "qr":
            qr_calls.append(args[0])
        return tensor_split(*args, **kwargs)

    def traced_canonize_between(*args, **kwargs):
        canonize_calls.append(args)
        return canonize_between(*args, **kwargs)

    recoveries = []
    move_center = opt._move_center

    def traced_move_center(target):
        region = opt.canonical_region
        if region is not None and len(region) > 1 and target in region:
            for nid in region:
                tensor = opt.tn.node_tensor(nid)
                if nid == target:
                    assert tensor.left_inds is None
                    continue
                toward_hub = opt.plan.node_path(nid, target)[1]
                bond = opt.tn.bond(nid, toward_hub)
                assert tensor.left_inds is not None
                assert set(tensor.left_inds) == set(tensor.inds) - {bond}
            before = (len(qr_calls), len(canonize_calls))
            result = move_center(target)
            recoveries.append(
                (
                    len(qr_calls) - before[0],
                    len(canonize_calls) - before[1],
                )
            )
            return result
        return move_center(target)

    monkeypatch.setattr(qtc, "tensor_split", traced_tensor_split)
    monkeypatch.setattr(opt.tn, "canonize_between", traced_canonize_between)
    monkeypatch.setattr(opt, "_move_center", traced_move_center)
    opt.apply_subtree_operator(gate, where)

    assert recoveries == [(0, 0)]
    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_dense_subtree_uses_proven_one_sided_compression(monkeypatch):
    """A routed gate ladder matches full reduction while skipping child QRs."""
    rng = np.random.default_rng(53)
    n = 8
    plan = TreePlan.from_order(range(n), structure="balanced")
    seed = TreeTensorNetwork.rand(plan, D=2, seed=53)
    optimized = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        run=False,
    )
    reference = TreeOptimizer(
        None,
        tree=plan,
        state=seed.copy(),
        chi=2,
        cutoff=1e-12,
        run=False,
    )

    left_reductions = []
    compress_edge = TreeTensorNetwork.compress_edge_

    def traced_compress_edge(tn, a, b, **kwargs):
        if tn is optimized.tn:
            reduced = kwargs.get("reduced", True)
            if reduced == "left":
                child = tn.node_tensor(b)
                bond = tn.bond(a, b)
                assert child.left_inds is not None
                assert set(child.left_inds) == set(child.inds) - {bond}
                left_reductions.append((a, b))
        return compress_edge(tn, a, b, **kwargs)

    full_compress = reference._compress_edge_with_diagnostics

    def force_full_reduction(
        u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False,
    ):
        del reduced
        return full_compress(
            u,
            v,
            max_bond=max_bond,
            cutoff=cutoff,
            reduced=True,
            reduction_proven=reduction_proven,
        )

    monkeypatch.setattr(
        TreeTensorNetwork, "compress_edge_", traced_compress_edge,
    )
    monkeypatch.setattr(
        reference, "_compress_edge_with_diagnostics", force_full_reduction,
    )

    exact = seed.to_statevector()
    ladder_supports = (
        (0, 2, 4),
        (1, 3, 5),
        (2, 4, 6),
        (3, 5, 7),
    )
    for where in ladder_supports:
        gate = _rand_unitary(3, rng)
        optimized.apply_subtree_operator(gate, where)
        reference.apply_subtree_operator(gate, where)
        exact = _sv_apply_kq(exact, gate, where, n)

    assert left_reductions
    assert _fidelity(optimized.to_dense(), reference.to_dense()) > 1 - 1e-10
    assert any(event["truncated"] for event in optimized.truncation_history)
    assert _fidelity(optimized.to_dense(), exact) == pytest.approx(
        _fidelity(reference.to_dense(), exact),
        rel=1e-10,
        abs=1e-12,
    )
    assert optimized.tn.validate(check_canonical=True) is optimized.tn


def test_tree_mode_is_construction_and_run_override():
    """Tree gate implementation mode follows the MPS construction/run API."""
    opt = TreeOptimizer(None, n=2, mode="direct", run=False)
    assert opt.mode == "direct"
    opt.run(mode="mpo")
    assert opt.mode == "mpo"
    # Existing shared-front-end spelling remains a deprecated no-op.
    with pytest.warns(DeprecationWarning, match="deprecated no-op"):
        opt.run(mode="tree")
    assert opt.mode == "mpo"
    with pytest.warns(DeprecationWarning, match="two_site_mode"):
        legacy = TreeOptimizer(None, n=2, two_site_mode="direct", run=False)
    assert legacy.mode == "direct"


def test_single_qubit_stream():
    """A one-qubit tree replays single-qubit gates correctly."""
    rng = np.random.default_rng(3)
    stream = [(_rand_unitary(1, rng), 0) for _ in range(5)]
    opt = TreeOptimizer(stream, n=1)
    psi = _exact_state(stream, 1)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-10


def test_tree_optimizer_has_no_local_expectation_api():
    """Observable contraction is owned by the TTN state, not its optimizer."""
    assert not hasattr(TreeOptimizer(None, n=1, run=False), "local_expectation")


def test_chi_truncation_caps_bond():
    """The maximum bond never exceeds the requested chi."""
    rng = np.random.default_rng(5)
    n = 8
    stream = _random_stream(n, 80, rng)
    chi = 4
    opt = TreeOptimizer(stream, n=n, chi=chi)
    assert opt.max_bond() <= chi


def test_tree_truncation_infidelity_compatibility_trace():
    """Tracked tree compression exposes the MPS-style diagnostic readout."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 1))],
        n=4,
        chi=1,
        track_truncation=True,
    )

    assert opt.get_infidelities()[0] == 0.0
    assert len(opt.get_infidelities()) >= 2
    assert len(opt.get_infidelity_samples()) == len(opt.get_infidelities()) - 1
    assert 0.0 <= opt.get_infidelities()[-1] <= 1.0
    assert opt.get_normalizations() == []


def test_tree_norm_ledger_is_independent_of_spectrum_tracking():
    """Tree norm fidelity remains available without per-edge SVD probes."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 3))],
        n=4,
        chi=1,
        track_truncation=False,
    )

    diagnostics = opt.norm_diagnostics()
    assert diagnostics["norm_tracking"] is True
    assert diagnostics["truncation_tracking"] is False
    removed_metric_names = {
        "norm_fidelity_raw",
        "norm_fidelity",
        "norm_infidelity",
        "local_norm_fidelity",
        "local_norm_infidelity",
        "cumulative_norm_fidelity",
        "cumulative_norm_infidelity",
    }
    assert removed_metric_names.isdisjoint(diagnostics)
    assert removed_metric_names.isdisjoint(opt.get_norm_events()[0])
    assert diagnostics["cumulative_fidelity"] == pytest.approx(0.5)
    assert diagnostics["cumulative_infidelity"] == pytest.approx(0.5)
    assert len(opt.get_norm_events()) == 2
    assert opt.get_infidelity_samples() == []


def test_tree_norm_ledger_can_exclude_known_nonunitary_updates():
    """Physical filter scale must not be labeled retained compression loss."""
    filter_gate = np.diag([1.0, 0.25]).astype(complex)
    opt = TreeOptimizer(None, n=2, chi=1, track_truncation=False, run=False)

    opt.apply_subtree_operator(filter_gate, (0,), track_norm=False)

    assert opt.get_norm_events() == []
    assert opt.norm_diagnostics()["cumulative_fidelity"] is None

    opt.apply_1q(np.eye(2, dtype=complex), 0)
    assert len(opt.get_norm_events()) == 1
    assert opt.get_norm_events()[0]["local_fidelity"] == pytest.approx(1.0)


def test_tree_truncation_survival_accumulates_in_log_space():
    """Many local survival factors remain stable in the cumulative trace."""
    opt = TreeOptimizer(None, n=2, chi=2, track_truncation=True, run=False)
    opt._active_update = {
        "kind": "gate",
        "support": (0, 1),
        "edge_start": 0,
        "started_at": 0.0,
    }
    local_survival = 0.999999999999
    count = 1000
    opt.truncation_history = [
        {
            "discarded_fraction": 1.0 - local_survival,
            "discarded_weight": 1.0 - local_survival,
            "truncated": True,
        }
        for _ in range(count)
    ]

    opt._finish_update()

    expected_log = count * np.log(local_survival)
    expected_infidelity = -np.expm1(expected_log)
    assert opt._truncation_log_survival == pytest.approx(expected_log)
    assert opt.get_infidelities()[-1] == pytest.approx(expected_infidelity)


def test_tree_run_supports_shared_non_unitary_normalization_controls():
    """Tree replay accepts the shared non-unitary normalization contract."""
    half = 0.5 * np.eye(2, dtype=complex)
    opt = TreeOptimizer([(half, 0), (half, 1)], n=3, run=False)
    opt.run(non_unitary=True, normalize_every=True)
    assert opt.norm() == pytest.approx(0.25)
    assert np.linalg.norm(opt.to_dense()) == pytest.approx(0.25)
    assert opt.tn.exponent == pytest.approx(np.log10(0.25))
    assert len(opt.get_normalizations()) == 2
    assert [event["old_norm"] for event in opt.get_normalizations()] == pytest.approx(
        [0.25, 0.25]
    )
    with pytest.raises(ValueError, match="non_unitary"):
        opt.run(normalize_every=True)


def test_tree_nonunitary_scale_control_preserves_represented_state():
    """Per-step scale control changes only the TTN working-data gauge."""
    twice = 2.0 * np.eye(2, dtype=complex)
    gates = [(twice, 0), (twice, 1)]
    raw = TreeOptimizer(gates, n=3)
    controlled = TreeOptimizer(gates, n=3, run=False)

    controlled.run(non_unitary=True, normalize_every=True)

    assert np.allclose(controlled.to_dense(), raw.to_dense())
    assert controlled.norm() == pytest.approx(raw.norm())
    assert controlled.tn.exponent == pytest.approx(np.log10(4.0))
    center = controlled.tn.node_tensor(controlled.center)
    assert np.linalg.norm(np.asarray(center.data)) == pytest.approx(1.0)
    assert [event["exponent"] for event in controlled.get_normalizations()] == (
        pytest.approx([np.log10(2.0), np.log10(4.0)])
    )


def test_tree_physical_normalize_clears_accumulated_scale():
    """Public normalize still makes the represented state unit norm."""
    twice = 2.0 * np.eye(2, dtype=complex)
    opt = TreeOptimizer([(twice, 0)], n=2, run=False)
    opt.run(non_unitary=True, normalize_every=True)
    assert opt.norm() == pytest.approx(2.0)

    old_norm = opt.normalize()

    assert old_norm == pytest.approx(2.0)
    assert opt.tn.exponent == pytest.approx(0.0)
    assert opt.norm() == pytest.approx(1.0)
    assert np.linalg.norm(opt.to_dense()) == pytest.approx(1.0)


def test_tree_logical_position_helpers_are_identity_mps_compatibility():
    """Tree backends expose identity logical/physical mapping helpers."""
    opt = TreeOptimizer(None, n=4, run=False)
    assert opt.qubits == [0, 1, 2, 3]
    assert opt.logical_order == [0, 1, 2, 3]
    assert [opt.logical_site(i) for i in range(4)] == [0, 1, 2, 3]
    assert [opt.position(i) for i in range(4)] == [0, 1, 2, 3]
    sample = np.array([0, 1, 0, 1])
    assert np.array_equal(opt.remap_sample(sample), sample)
    assert opt.restore_qubit_order() is opt.tn


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_truncated_fidelity_improves_with_chi(seed):
    """Threading the whole gate before truncating yields high truncated fidelity.

    The gate is threaded *exactly* along the tree geodesic and only compressed
    once both factors are present (Seitz et al., Figs. 3-6).  Every bond
    truncation therefore sees the complete gate, so fidelity rises
    monotonically with ``chi`` and reaches good accuracy at moderate ``chi`` --
    unlike truncating each hop before the far gate factor has been absorbed.
    """
    rng = np.random.default_rng(seed)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    psi = _exact_state(stream, n)

    fids = [
        _fidelity(psi, TreeOptimizer(stream, n=n, chi=chi).to_dense())
        for chi in (2, 4, 8)
    ]
    # monotone non-decreasing in chi (allowing tiny numerical slack)
    assert fids[1] >= fids[0] - 1e-9
    assert fids[2] >= fids[1] - 1e-9
    # moderate chi already recovers most of the state
    assert fids[2] > 0.4


def test_normalize_sets_unit_norm():
    """normalize() rescales the represented state to unit norm."""
    rng = np.random.default_rng(6)
    n = 5
    stream = _random_stream(n, 30, rng)
    opt = TreeOptimizer(stream, n=n, chi=8)  # truncated -> norm < 1
    opt.normalize()
    assert abs(opt.norm() - 1.0) < 1e-9


def test_user_supplied_plan_runs():
    """A caller-provided TreePlan is honoured."""
    rng = np.random.default_rng(7)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    assert isinstance(plan, TreePlan)
    stream = _random_stream(n, 20, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_root_physical_qubit_is_first_class_tree_site():
    """A binary top tensor can own one physical qubit alongside two bonds."""
    plan = TreePlan.from_order(
        range(4), structure="balanced", root_qubit=4,
    )
    state = TreeTensorNetwork.from_plan(plan)
    root = state.node_tensor(plan.root)

    assert plan.n == 5
    assert plan.root_qubit == 4
    assert plan.node_of_qubit[4] == plan.root
    assert 4 not in plan.leaf_of_qubit
    assert set(root.inds) == {
        state.site_ind(4),
        *(state.bond(plan.root, child) for child in plan.children[plan.root]),
    }
    assert root.ndim == 3
    assert set(state.outer_inds()) == {
        state.site_ind(q) for q in range(plan.n)
    }
    expected = np.zeros(2**plan.n)
    expected[0] = 1.0
    assert np.array_equal(state.to_statevector(), expected)
    assert state.validate(check_canonical=True) is state


@pytest.mark.parametrize("root_qubit", [0, 1])
def test_two_qubit_tree_uses_distinct_root_and_leaf_sites(root_qubit):
    """The smallest root-site tree uses a unary root over one physical leaf."""
    leaf_qubit = 1 - root_qubit
    plan = TreePlan.from_order(
        [leaf_qubit], structure="balanced", root_qubit=root_qubit,
    )
    layered = TreePlan.build_layered(
        [leaf_qubit], block_size=2, root_qubit=root_qubit,
    )
    found = TreeLayoutFinder(
        [], n=2, root_qubit=root_qubit, max_arity=2,
    ).run()
    automatic = TreeOptimizer(
        None, n=2, root_qubit=root_qubit, max_arity=2, run=False,
    )

    for candidate in (plan, layered, found):
        assert candidate.n == 2
        assert candidate.root_qubit == root_qubit
        assert candidate.node_of_qubit[root_qubit] == candidate.root
        assert candidate.node_of_qubit[leaf_qubit] != candidate.root
        assert len(candidate.children[candidate.root]) == 1
    assert automatic.plan.node_of_qubit[root_qubit] == automatic.plan.root
    assert automatic.tn.validate(check_canonical=True) is automatic.tn

    stream = [
        (pepsy.h(), root_qubit),
        (pepsy.cnot(), (root_qubit, leaf_qubit)),
    ]
    opt = TreeOptimizer(stream, tree=plan, chi=8)
    assert _fidelity(_exact_state(stream, 2), opt.to_dense()) > 1 - 1e-12
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_root_physical_qubit_gate_and_submpo_replay_are_exact():
    """Direct gates and a structured sub-MPO can target the top physical leg."""
    plan = TreePlan.from_order(
        range(4), structure="balanced", root_qubit=4,
    )
    direct_stream = [(pepsy.h(), 0), (pepsy.cnot(), (0, 4))]
    direct = TreeOptimizer(
        direct_stream, tree=plan, chi=32, cutoff=0.0,
    )
    assert _fidelity(
        _exact_state(direct_stream, plan.n), direct.to_dense()
    ) > 1 - 1e-10
    assert direct.tn.validate(check_canonical=True) is direct.tn

    where = (1, 3, 4)
    mpo = _two_branch_flip_submpo(
        L=plan.n,
        sites=where,
        targets=where,
        w0=0.0,
        w1=1.0,
    )
    submpo = TreeOptimizer(
        None, tree=plan, chi=32, cutoff=0.0, run=False,
    )
    submpo.apply_submpo(mpo, where)
    expected = np.zeros(2**plan.n, dtype=complex)
    expected[int("01011", 2)] = 1.0
    assert np.allclose(submpo.to_dense(), expected)
    assert submpo.tn.validate(check_canonical=True) is submpo.tn


def test_root_physical_qubit_layout_and_cap_are_root_aware():
    """Layout scoring reaches the root site and capping removes only its leg."""
    finder = TreeLayoutFinder(
        supports=[(0, 4), (0, 4), (1, 2)],
        n=5,
        root_qubit=4,
        structure="balanced",
        max_arity=2,
    )
    plan = finder.run(refine="greedy", refine_budget=16)
    root_path = plan.node_path(plan.node_of_qubit[0], plan.root)
    loads = finder.edge_loads(plan)
    path_edges = {
        (u, v) if plan.parent.get(v) == u else (v, u)
        for u, v in zip(root_path, root_path[1:])
    }

    assert plan.root_qubit == 4
    assert plan.node_of_qubit[4] == plan.root
    assert all(loads[edge] > 0.0 for edge in path_edges)
    assert finder.report(plan)["root_qubit"] == 4

    automatic = TreeOptimizer(
        None, root_qubit=4, max_arity=2, chi=16, run=False,
    )
    assert automatic.n == 5
    assert automatic.plan.root_qubit == 4

    opt = TreeOptimizer(None, tree=plan, chi=16, run=False)
    assert opt.layout_report()["root_qubit"] == 4
    opt.apply_1q(pepsy.h(), 4)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert opt.tn.local_expectation(x, 4) == pytest.approx(1.0)
    opt.cap(4, [1.0, 0.0])
    assert opt.plan.root_qubit is None
    assert opt.n == 4
    assert opt.to_dense().shape == (2**4,)
    assert opt.norm() == pytest.approx(1 / np.sqrt(2))
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_explicit_tree_rejects_mismatched_n():
    """Explicit plans enforce the same qubit-count invariant as finders."""
    plan = TreePlan.from_order(
        range(4), structure="balanced", root_qubit=4,
    )
    with pytest.raises(
        ValueError, match=r"tree contains 5 qubits, but n=4"
    ):
        TreeOptimizer(None, n=4, tree=plan, run=False)
    with pytest.raises(
        ValueError, match=r"tree contains 5 qubits, but n=4"
    ):
        TreeOptimizer(None, n=4, layout=plan, run=False)


def test_layout_finder_builds_strict_binary_tree_when_requested():
    """Explicit top_arity=2 opts out of the ternary virtual root."""
    rng = np.random.default_rng(8)
    n = 8
    stream = _random_stream(n, 60, rng)
    plan = TreeLayoutFinder(stream, n=n, max_arity=2, top_arity=2).run()
    assert plan.n == n
    assert set(plan.leaf_of_qubit) == set(range(n))
    # every internal node has exactly two children
    for nid in plan.nodes():
        assert len(plan.children[nid]) in (0, 2)
    # tree distances are well-defined for all pairs
    for a in range(n):
        for b in range(a + 1, n):
            assert plan.tree_distance(a, b) >= 1


def test_tree_layout_accepts_explicit_fixed_site_order():
    """An explicit order builds the requested binary/ternary-root tree."""
    order = pepsy.square_lattice_zigzag(2, 2)
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 1)), (pepsy.cnot(), (2, 3))],
        n=4,
        max_arity=2,
        top_arity=3,
    )
    plan = finder.run(order=order)

    assert tuple(plan.qubit_of_leaf.values()) == order
    assert plan.top_arity == 3
    assert plan.is_binary()


@pytest.mark.parametrize(
    "mode",
    [
        "row-major",
        "col-major",
        "snake",
        "snake-row-major",
        "folded-snake",
        "folded-snake-row-major",
        "hilbert",
        "hilbert-row-major",
    ],
)
def test_tree_layout_finder_supports_onedmap_lattice_presets(mode):
    """Named Tree presets preserve the corresponding OneDMap leaf traversal."""
    Lx = Ly = 4
    finder = TreeLayoutFinder(
        [],
        n=Lx * Ly,
        structure="quality",
        max_arity=2,
        top_arity=2,
        lattice_shape=(Lx, Ly),
    )
    one_d_to_lattice, _ = pepsy.OneDMap.build(Lx, Ly, mode=mode)
    expected = tuple(x * Ly + y for x, y in one_d_to_lattice.values())

    plan = finder.run(order=mode)

    assert plan.mpo_order() == expected


def test_tree_layout_aliases_match_onedmap_hilbert_orientation():
    """Tree aliases preserve both rectangular generalized-Hilbert orientations."""
    finder = TreeLayoutFinder(
        [],
        n=15,
        max_arity=2,
        top_arity=2,
        lattice_shape=(3, 5),
    )
    expected = {
        mode: tuple(x * 5 + y for x, y in pepsy.OneDMap.build(
            3, 5, mode="hilbert-row-major" if "row" in mode else "hilbert",
        )[0].values())
        for mode in ("hilbert-row", "hilbert-row-major", "hilbert")
    }

    assert finder.run(order="hilbert-row").mpo_order() == expected["hilbert-row"]
    assert finder.run(order="hilbert-row-major").mpo_order() == expected[
        "hilbert-row-major"
    ]
    assert finder.run(order="hilbert").mpo_order() == expected["hilbert"]
    assert expected["hilbert-row"] != expected["hilbert"]


@pytest.mark.parametrize("mode", [
    "row-major",
    "col-major",
    "snake",
    "snake-row-major",
    "folded-snake",
    "folded-snake-row-major",
    "hilbert",
    "hilbert-row-major",
])
@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (1, 7), (2, 5)])
def test_tree_geometric_order_is_preserved_at_every_hierarchy_layer(
    mode, shape,
):
    """Geometric trees coarsen adjacent OneDMap intervals at every level."""
    Lx, Ly = shape
    n = Lx * Ly
    finder = TreeLayoutFinder(
        [],
        n=n,
        max_arity=2,
        top_arity=3,
        lattice_shape=shape,
    )
    expected = tuple(
        x * Ly + y
        for x, y in pepsy.OneDMap.build(Lx, Ly, mode=mode)[0].values()
    )
    plan = finder.run(order=mode)

    assert plan.mpo_order() == expected
    masks = plan.subtree_qubit_masks()
    positions = {qubit: i for i, qubit in enumerate(expected)}

    # Every internal node must partition one contiguous path interval into
    # consecutive child intervals. This includes the top layer, rather than
    # checking only the leaf MPO order.
    for node, children in plan.children.items():
        if len(children) < 2:
            continue
        intervals = []
        for child in children:
            child_positions = sorted(
                positions[q]
                for q in range(n)
                if masks[child] & (1 << q)
            )
            intervals.append((child_positions[0], child_positions[-1]))
            assert child_positions == list(
                range(child_positions[0], child_positions[-1] + 1)
            )
        assert intervals[0][0] == min(interval[0] for interval in intervals)
        for previous, current in zip(intervals, intervals[1:]):
            assert previous[1] + 1 == current[0]

    # These cases are large enough to exercise the root plus multiple middle
    # layers; the top children must themselves be non-leaf subtrees.
    assert len(plan.children[plan.root]) == 3
    assert all(plan.children[child] for child in plan.children[plan.root])


def test_tree_lattice_order_helper_and_missing_shape_error():
    """Tree geometric orders are reusable and require an explicit shape."""
    expected = (0, 2, 4, 1, 3, 5)
    assert TreeLayoutFinder.lattice_order(
        2, 3, "row-major", site=lambda x, y: y * 2 + x,
    ) == expected

    finder = TreeLayoutFinder([], n=4)
    with pytest.raises(ValueError, match="lattice_shape"):
        finder.run(order="hilbert")


def test_tree_alternating_lattice_orders_and_coarse_blocks():
    """Coarse presets traverse complete blocks before moving to the next."""
    shape = (4, 3)
    row_alternate = TreeLayoutFinder.lattice_order(
        *shape, "alternate-x"
    )
    col_alternate = TreeLayoutFinder.lattice_order(
        *shape, "alternate-y"
    )
    assert row_alternate == (0, 3, 6, 9, 10, 7, 4, 1, 2, 5, 8, 11)
    assert col_alternate == (0, 1, 2, 5, 4, 3, 6, 7, 8, 11, 10, 9)

    coarse_row = TreeLayoutFinder.lattice_order(
        *shape, "coarse-row-major", grain=(2, 1)
    )
    assert coarse_row == (0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11)

    coarse_y = TreeLayoutFinder.lattice_order(
        *shape, "coarse-alternate-y", grain=(1, 2)
    )
    assert coarse_y == col_alternate

    for mode in (
        "coarse-row-major",
        "coarse-col-major",
        "coarse-snake",
        "coarse-snake-row-major",
        "coarse-folded-snake",
        "coarse-folded-snake-row-major",
        "coarse-hilbert",
        "coarse-hilbert-row-major",
    ):
        order = TreeLayoutFinder.lattice_order(
            5, 4, mode, grain=(2, 2)
        )
        assert len(order) == 20
        assert set(order) == set(range(20))

        coords = [(q // 4, q % 4) for q in order]
        blocks = [(x // 2, y // 2) for x, y in coords]
        block_runs = []
        start = 0
        while start < len(blocks):
            block = blocks[start]
            end = start + 1
            while end < len(blocks) and blocks[end] == block:
                end += 1
            block_runs.append(block)
            assert all(current == block for current in blocks[start:end])
            start = end
        assert len(block_runs) == 6
        assert len(set(block_runs)) == 6


def test_tree_coarse_order_handoff_and_validation():
    """TreeOptimizer and finder diagnostics retain coarse-layout settings."""
    finder = TreeLayoutFinder(
        [],
        n=12,
        max_arity=2,
        top_arity=2,
        lattice_shape=(4, 3),
        order="coarse-alternate-x",
        coarse_grain=(2, 1),
    )
    plan = finder.run()
    assert plan.mpo_order() == (0, 3, 6, 9, 10, 7, 4, 1, 2, 5, 8, 11)
    assert finder.report(plan)["coarse_grain"] == (2, 1)

    forwarded = TreeOptimizer.find_tree_layout(
        [],
        n=12,
        max_arity=2,
        top_arity=2,
        lattice_shape=(4, 3),
        order="coarse-row-major",
        coarse_grain=2,
    )
    assert forwarded.mpo_order() == (0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11)

    with pytest.raises(ValueError, match="coarse_grain"):
        TreeLayoutFinder.lattice_order(
            4, 3, "coarse-snake", grain=(0, 1)
        )


def test_tree_3d_lattice_orders_support_axis_alternation():
    """Tree presets preserve the shared 3D OneDMap traversal vocabulary."""
    shape = (3, 2, 2)
    size = np.prod(shape)
    coords = {
        q: (q // (shape[1] * shape[2]),
            (q // shape[2]) % shape[1],
            q % shape[2])
        for q in range(size)
    }
    for mode in (
        "row-major",
        "col-major",
        "snake",
        "snake-row-major",
        "alternate-x",
        "alternate-y",
        "alternate-z",
    ):
        expected = tuple(
            x * shape[1] * shape[2] + y * shape[2] + z
            for x, y, z in pepsy.OneDMap.build(
                *shape[:2], Lz=shape[2], mode=mode,
            )[0].values()
        )
        order = TreeLayoutFinder.lattice_order(*shape, mode=mode)
        assert order == expected
        assert set(order) == set(range(size))

    # Each non-coarse alternating path is a nearest-neighbor 3D traversal.
    for mode in ("alternate-x", "alternate-y", "alternate-z"):
        order = TreeLayoutFinder.lattice_order(*shape, mode=mode)
        assert all(
            sum(abs(coords[left][axis] - coords[right][axis]) for axis in range(3))
            == 1
            for left, right in zip(order[:-1], order[1:])
        )


@pytest.mark.parametrize(
    ("mode", "grain"),
    [
        ("coarse-alternate-x", (2, 1, 1)),
        ("coarse-alternate-y", (1, 2, 1)),
        ("coarse-alternate-z", (1, 1, 2)),
    ],
)
def test_tree_3d_coarse_alternating_axes_keep_blocks_and_paths(mode, grain):
    """3D coarse alternation keeps each block contiguous and path-connected."""
    shape = (4, 3, 3)
    size = int(np.prod(shape))
    order = TreeLayoutFinder.lattice_order(*shape, mode=mode, grain=grain)
    assert len(order) == size
    assert set(order) == set(range(size))

    coords = {
        q: (q // (shape[1] * shape[2]),
            (q // shape[2]) % shape[1],
            q % shape[2])
        for q in range(size)
    }
    block_ids = [
        tuple(coords[q][axis] // grain[axis] for axis in range(3))
        for q in order
    ]
    runs = []
    start = 0
    while start < len(block_ids):
        block = block_ids[start]
        end = start + 1
        while end < len(block_ids) and block_ids[end] == block:
            end += 1
        assert all(block_id == block for block_id in block_ids[start:end])
        runs.append(block)
        start = end
    expected_blocks = int(np.prod([
        (length + block - 1) // block
        for length, block in zip(shape, grain)
    ]))
    assert len(runs) == len(set(runs)) == expected_blocks
    assert all(
        sum(abs(coords[left][axis] - coords[right][axis]) for axis in range(3))
        == 1
        for left, right in zip(order[:-1], order[1:])
    )


def test_tree_3d_coarse_order_handoff_and_site_mapper():
    """Finder and TreeOptimizer retain 3D coarse-layout metadata."""
    shape = (3, 2, 2)
    finder = TreeLayoutFinder(
        [],
        n=int(np.prod(shape)),
        max_arity=2,
        top_arity=2,
        lattice_shape=shape,
        order="coarse-alternate-z",
        coarse_grain=(1, 1, 2),
    )
    plan = finder.run()
    expected = tuple(
        x * shape[1] * shape[2] + y * shape[2] + z
        for x, y, z in pepsy.OneDMap.build(
            *shape[:2], Lz=shape[2], mode="alternate-z",
        )[0].values()
    )
    assert plan.mpo_order() == expected
    report = finder.report(plan)
    assert report["lattice_shape"] == shape
    assert report["coarse_grain"] == (1, 1, 2)

    custom = TreeLayoutFinder.lattice_order(
        3, 2, 2, "alternate-z",
        site=lambda x, y, z: z * 6 + y * 3 + x,
    )
    assert set(custom) == set(range(12))

    forwarded = TreeOptimizer.find_tree_layout(
        [],
        n=12,
        max_arity=2,
        top_arity=2,
        lattice_shape=shape,
        order="coarse-alternate-x",
        coarse_grain=2,
    )
    assert set(forwarded.mpo_order()) == set(range(12))


def test_quality_layout_not_worse_than_balanced():
    """Entanglement-adapted structure scores no worse than balanced order."""
    rng = np.random.default_rng(9)
    n = 8
    # locally clustered interactions: quality bisection should exploit them
    stream = []
    for _ in range(80):
        a = int(rng.integers(n - 1))
        b = a + 1 if rng.random() < 0.85 else int(rng.integers(n))
        if a == b:
            b = (b + 1) % n
        stream.append((_rand_unitary(2, rng), (a, b)))
    finder = TreeLayoutFinder(stream, n=n, structure="quality")
    quality = finder.run()
    balanced = TreePlan.from_order(range(n), structure="balanced")
    assert finder.score(quality) <= finder.score(balanced)


def test_congestion_layout_uses_operator_schmidt_edge_load():
    """The load-aware diagnostic predicts the product of crossed ranks."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    rng = np.random.default_rng(101)
    generic = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    finder = TreeLayoutFinder(
        [(cnot, (0, 3)), (generic, (0, 3))],
        n=4,
        objective="congestion",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    loads = finder.edge_loads(plan)
    path = plan.node_path(plan.leaf_of_qubit[0], plan.leaf_of_qubit[3])
    path_edges = {
        (u, v) if plan.parent.get(v) == u else (v, u)
        for u, v in zip(path, path[1:])
    }

    assert path_edges
    assert all(loads[edge] == pytest.approx(3.0) for edge in path_edges)
    assert max(loads.values()) == pytest.approx(3.0)
    report = finder.report(plan)
    assert report["objective"] == "congestion"
    assert report["peak_bond_growth"] == pytest.approx(8.0)


def test_compression_layout_reports_rank_bounds_and_tensor_cost():
    """Compression selection is explicit and honest for wide operators."""
    gate = np.eye(8, dtype=complex)
    finder = TreeLayoutFinder(
        [(gate, (0, 1, 2))],
        n=3,
        objective="compression",
        max_arity=(2, 3),
        max_operator_qubits=2,
        chi=2,
    )
    plan = finder.run()
    report = finder.report(plan)

    assert report["objective"] == "compression"
    assert report["rank_bounded_events"] > 0
    assert report["rank_bound_reasons"]["max_operator_qubits"] > 0
    assert report["estimated_max_tensor_log2"] >= 0.0
    assert len(report["objective_key"]) >= 5


def test_hypergraph_layout_scores_full_multisite_supports():
    """Direct mode ranks original hyperedges on every crossed tree cut."""
    rng = np.random.default_rng(104)
    gate = _rand_unitary(3, rng)
    supports = ((0, 1, 2), (2, 3, 4))
    finder = TreeLayoutFinder(
        [(gate, supports[0]), (gate, supports[1])],
        n=5,
        max_arity=2,
        objective="hypergraph",
    )
    plan = TreePlan.from_order(range(5), structure="balanced", max_arity=2)

    loads = finder.edge_loads(plan)
    below = plan.subtree_qubit_masks()
    expected = {edge: 0.0 for edge in loads}
    for payload, support in zip(finder.payloads, finder.supports):
        support_mask = sum(1 << q for q in support)
        for edge in expected:
            _parent, child = edge
            left_mask = support_mask & below[child]
            if not left_mask or left_mask == support_mask:
                continue
            left = tuple(q for q in support if left_mask & (1 << q))
            expected[edge] += np.log2(finder._schmidt_rank(payload, support, left))

    assert loads == pytest.approx(expected)
    report = finder.report(plan)
    assert report["objective"] == "hypergraph"
    assert report["hypergraph_score"] == {
        "max_edge_load": max(loads.values()),
        "total_edge_load": sum(loads.values()),
    }

    recommendation = finder.recommend_arities((2,), chi=None)
    assert recommendation["refine"] == "greedy"
    assert recommendation["topology_refine"] == "nni"
    assert recommendation["candidates"][0]["planning"][
        "topology_refinement"
    ]["method"] == "nni"


def test_tree_compression_layout_pilot_is_non_mutating():
    """Tree pilot selection compares copied product states only."""
    gates = [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))]
    opt = TreeOptimizer(gates, n=4, chi=2, run=False)
    original_plan = opt.plan

    selected = opt.select_layout_for_compression(
        pilot_candidates=1,
        pilot_steps=1,
    )

    assert selected["selected_candidate"]
    assert selected["pilot"]["reports"]
    assert any(
        name.startswith("quality:")
        for name in selected["pilot"]["pilot_candidates"]
    )
    assert opt.plan is original_plan
    assert opt.max_bond() == 1


def test_tree_layout_pilot_feedback_is_iterative_and_edge_aware():
    """Full-tree pilots feed bounded hot-edge proposals into the next round."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    gates = [(h, 0), (pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))]
    opt = TreeOptimizer(gates, n=4, max_arity=2, chi=1, run=False)
    original_plan = opt.plan

    selected = opt.optimize_layout(
        objective="full_tree",
        pilot_candidates=1,
        pilot_steps=3,
        rounds=2,
        topology_budget=1,
        refine_budget=1,
        search_budget=2,
    )

    assert selected["pilot"]["objective"] == "full_tree"
    assert selected["pilot"]["n_rounds"] == 2
    assert len(selected["pilot"]["rounds"]) == 2
    assert opt.plan is original_plan
    report = selected["pilot"]["reports"][selected["selected_candidate"]]
    assert report["status"] == "ok"
    assert report["update_runtime_seconds"] >= 0.0
    assert isinstance(report["edge_diagnostics"], dict)
    assert opt.max_bond() == 1


def test_tree_layout_pilots_can_run_in_parallel():
    """Independent layout pilots preserve the normal report contract."""
    gates = [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))]
    opt = TreeOptimizer(gates, n=4, chi=1, run=False)

    selected = opt.optimize_layout(
        objective="full_tree",
        pilot_candidates=2,
        pilot_workers=2,
        pilot_steps=2,
        rounds=1,
        topology_budget=1,
        refine_budget=1,
        search_budget=2,
    )

    assert selected["pilot"]["selected_candidate"]
    assert len(selected["pilot"]["reports"]) == 2
    assert all(
        report["status"] == "ok"
        for report in selected["pilot"]["reports"].values()
    )


def test_tree_layout_targeted_candidates_are_static_and_bounded():
    """Hot-edge proposal generation never allocates or mutates a TTN."""
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))],
        n=4,
        max_arity=2,
        objective="full_tree",
        chi=None,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", max_arity=2)
    edge = next(
        (parent, child)
        for parent, children in plan.children.items()
        for child in children
    )
    proposals = finder.targeted_candidates(
        plan,
        {edge: {"truncated": 1, "discarded_fraction": 0.5}},
        budget=3,
        seed=3,
    )

    assert len(proposals) <= 3
    assert all(candidate.is_binary() for candidate in proposals)
    unchanged = TreePlan.from_order(range(4), structure="balanced", max_arity=2)
    assert plan.children == unchanged.children
    assert plan.qubit_of_leaf == unchanged.qubit_of_leaf


def test_tree_candidate_plans_include_quality_for_state_aware_pilots():
    """Quality refinement is exposed as an explicit pilot candidate."""
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))],
        n=4,
        max_arity=2,
        objective="compression",
    )

    candidates = finder.candidate_plans(
        chi=2,
        include_quality=True,
        quality_refine_budget=2,
        quality_topology_budget=2,
    )

    quality = candidates["quality:arity=2"]
    assert quality["planning"]["topology_refinement"]["method"] == "nni"
    assert quality["planning"]["refinement"]["method"] == "greedy"
    assert quality["plan"].is_binary()
    assert not any(
        name.startswith("quality:")
        for name in finder.candidate_plans(chi=2)
    )


def test_full_tree_profile_reports_dynamic_cost_at_all_scales():
    """Whole-tree mode exposes width, work, demand, and scale diagnostics."""
    gates = [
        (pepsy.cnot(), (0, 1)),
        (pepsy.cnot(), (1, 2)),
        (pepsy.cnot(), (3, 4)),
        (pepsy.cnot(), (2, 4)),
    ]
    finder = TreeLayoutFinder(
        gates, n=5, max_arity=2, objective="full_tree", chi=4,
    )
    plan = TreePlan.from_order(range(5), structure="balanced", max_arity=2)
    profile = finder.full_tree_profile(plan)

    assert profile["event_count"] == len(gates)
    assert profile["peak_tensor_log2"] >= 1.0
    assert profile["peak_work_log2"] >= profile["peak_tensor_log2"]
    assert profile["total_route_length"] > 0
    assert profile["scales"]
    assert all(
        {
            "node_count",
            "edge_count",
            "peak_tensor_log2",
            "peak_edge_demand_log2",
        } <= set(scale_info)
        for scale_info in profile["scales"].values()
    )

    report = finder.report(plan)
    assert report["objective"] == "full_tree"
    assert report["full_tree"] == profile
    assert len(report["objective_key"]) == 10
    assert report["objective_key"][:4] == (
        profile["peak_overflow_log2"],
        profile["total_overflow_log2"],
        profile["peak_edge_demand_log2"],
        profile["total_edge_demand_log2"],
    )


def test_full_tree_6x6_pbc_calibrates_against_actual_replay():
    """All-scale overflow ranking tracks real capped-tree replay pressure."""
    Lx = Ly = 6
    n = Lx * Ly

    def site(x, y):
        return x * Ly + y

    edges = []
    for x in range(Lx):
        for y in range(Ly):
            for dx, dy in ((1, 0), (0, 1)):
                edge = tuple(sorted((
                    site(x, y),
                    site((x + dx) % Lx, (y + dy) % Ly),
                )))
                if edge[0] != edge[1] and edge not in edges:
                    edges.append(edge)

    gates = (
        [(pepsy.h(), q) for q in range(n)]
        + [(pepsy.cphase(np.pi / 4), edge) for edge in edges]
    )
    finder = TreeLayoutFinder(
        gates,
        n=n,
        max_arity=(2, 3, 4),
        top_arity=3,
        objective="full_tree",
        chi=4,
        seed=11,
    )
    recommendation = finder.recommend_arities(
        (2, 3, 4),
        refine=None,
        topology_refine=None,
        search=None,
    )

    calibrated = []
    for candidate in recommendation["candidates"]:
        optimizer = TreeOptimizer(
            gates,
            n=n,
            tree=candidate["plan"],
            chi=4,
            cutoff=0.0,
            track_truncation=False,
            run=False,
        )
        optimizer.run()
        history = optimizer.truncation_history
        calibrated.append({
            "arity": candidate["max_arity"],
            "predicted_total_overflow": candidate["full_tree_profile"][
                "total_overflow_log2"
            ],
            "actual_total_excess": sum(
                max(0, event["before_bond"] - 4) for event in history
            ),
            "actual_truncations": sum(
                bool(event["truncated"]) for event in history
            ),
        })

    by_arity = {item["arity"]: item for item in calibrated}
    assert recommendation["recommended_max_arity"] == 4
    assert (
        by_arity[4]["predicted_total_overflow"]
        < by_arity[3]["predicted_total_overflow"]
        < by_arity[2]["predicted_total_overflow"]
    )
    assert (
        by_arity[4]["actual_total_excess"]
        < by_arity[3]["actual_total_excess"]
        < by_arity[2]["actual_total_excess"]
    )
    assert (
        by_arity[4]["actual_truncations"]
        < by_arity[3]["actual_truncations"]
        < by_arity[2]["actual_truncations"]
    )


def test_full_tree_anneals_subtrees_without_changing_binary_contract():
    """All-scale subtree search preserves the requested binary tree shape."""
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1)),
         (pepsy.cnot(), (2, 5)), (pepsy.cnot(), (4, 5))],
        n=6,
        max_arity=2,
        top_arity=3,
        objective="full_tree",
        chi=4,
        seed=7,
    )
    recommendation = finder.recommend_arities(
        (2,),
        topology_budget=3,
        search_budget=4,
        refine=None,
    )
    candidate = recommendation["candidates"][0]
    planning = candidate["planning"]
    plan = recommendation["plan"]

    assert plan.top_arity == 3
    assert plan.is_binary()
    assert planning["topology_refinement"]["method"] == "subtree"
    assert planning["search"]["method"] == "subtree"
    assert planning["search"]["search"] == "anneal"
    assert candidate["full_tree_profile"]["scales"]


def test_full_tree_hybrid_quality_search_is_static_and_budgeted():
    """Full-tree quality combines topology and leaf search without a TTN."""
    pytest.importorskip("nevergrad")
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1)),
         (pepsy.cnot(), (2, 5)), (pepsy.cnot(), (4, 5))],
        n=6,
        max_arity=(2,),
        top_arity=3,
        objective="full_tree",
        seed=7,
    )
    plan = finder.run(
        order="quality",
        topology_budget=2,
        refine_budget=2,
        search_budget=6,
    )

    planning = finder._last_arity_recommendation["candidates"][0]["planning"]
    search = planning["search"]
    assert finder.chi is None
    assert plan.is_binary()
    assert search["method"] == "hybrid"
    assert search["anneal"]["search"] == "anneal"
    assert search["nevergrad"]["method"] == "nevergrad"
    assert search["evaluations"] <= 6


def test_tree_edge_loads_match_full_edge_reference():
    """Steiner-only edge scanning preserves the full congestion calculation."""
    rng = np.random.default_rng(109)
    n = 9
    stream = _random_stream(n, 25, rng, two_qubit_frac=0.8)
    finder = TreeLayoutFinder(stream, n=n, objective="congestion")
    plan = TreePlan.from_order(range(n), structure="balanced")

    got = finder.edge_loads(plan)
    below = plan.subtree_qubit_masks()
    expected = {edge: 0.0 for edge in got}
    for payload, support, event_type in zip(
        finder.payloads, finder.supports, finder.event_types
    ):
        support = tuple(dict.fromkeys(support))
        if len(support) < 2 or event_type in {
            "measure", "reset", "measure_reset", "cap",
        }:
            continue
        support_mask = sum(1 << q for q in support)
        for edge in expected:
            _parent, child = edge
            left_mask = support_mask & below[child]
            if not left_mask or left_mask == support_mask:
                continue
            left = tuple(q for q in support if left_mask & (1 << q))
            rank = finder._schmidt_rank(payload, support, left)
            expected[edge] += np.log2(rank)

    assert got == pytest.approx(expected)


def test_tree_layout_reuses_dense_gate_schmidt_rank_across_labels():
    """One gate matrix needs one rank calculation for each wire partition."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder(
        [(cnot, (0, 1)), (cnot, (2, 3))], n=4,
        objective="congestion",
    )

    assert finder._schmidt_rank(cnot, (0, 1), (0,)) == 2
    assert finder._schmidt_rank(cnot, (2, 3), (2,)) == 2
    assert len(finder._schmidt_rank_cache) == 1


def test_optimizer_exposes_congestion_layout_objective():
    """TreeOptimizer can request the rank-aware automatic layout."""
    rng = np.random.default_rng(102)
    stream = _random_stream(6, 20, rng, two_qubit_frac=0.8)
    opt = TreeOptimizer(
        stream,
        n=6,
        max_arity=2,
        layout_objective="congestion",
        layout_weight_mode="operator_schmidt",
        run=False,
    )

    assert opt.layout_objective == "congestion"
    assert opt.plan.n == 6
    assert opt.plan.is_binary()


def test_optimizer_defaults_to_congestion_layout_for_replay_performance():
    """Automatic optimizer layouts default to finite-chi edge pressure."""
    opt = TreeOptimizer(None, n=6, run=False)

    assert opt.layout_objective == "congestion"
    assert opt.layout_finder.objective == "congestion"
    assert opt.mode == "auto"
    assert opt.threads == 1
    assert opt.subtree_workers == 1
    assert opt.track_truncation is False
    assert opt.track_infidelity is True
    assert inspect.signature(TreeOptimizer).parameters["cutoff"].default == "auto"
    assert inspect.signature(TreeOptimizer).parameters["cutoff_mode"].default == "auto"
    assert opt.cutoff == pytest.approx(1e-12)
    assert opt.cutoff_mode == "rsum2"
    assert opt.profile is False


def test_tree_optimizer_auto_cutoff_tracks_state_dtype():
    """Automatic tree cutoffs use the live TTN precision."""
    opt = TreeOptimizer(
        None,
        n=3,
        dtype="complex64",
        cutoff="auto",
        cutoff_mode="auto",
        run=False,
    )

    assert opt.backend_info()["dtype"] == "complex64"
    assert opt.cutoff == pytest.approx(1e-6)
    assert opt.cutoff_mode == "rsum2"


def test_tree_state_compression_default_matches_optimizer():
    """The low-level TTN edge API uses the same cutoff convention."""
    parameter = inspect.signature(
        TreeTensorNetwork.compress_edge_
    ).parameters["cutoff_mode"]

    assert parameter.default == "rsum2"


def test_layout_recommends_arity_and_reports_tree_shape():
    """The finder compares binary/wider candidates and exposes their costs."""
    rng = np.random.default_rng(103)
    stream = _random_stream(8, 30, rng, two_qubit_frac=0.8)
    finder = TreeLayoutFinder(stream, n=8, objective="congestion")

    recommendation = finder.recommend_arities((2, 3))
    assert recommendation["recommended_max_arity"] in (2, 3)
    assert len(recommendation["candidates"]) == 2
    assert all("max_virtual_degree" in item
               for item in recommendation["candidates"])
    report = finder.report(recommendation["plan"])
    assert report["arity_histogram"]
    assert report["max_arity"] in (2, 3)


def test_layout_finder_layered_direct_block_size():
    """`layered` builds a valid fixed layered tree for a chosen block_size."""
    rng = np.random.default_rng(107)
    stream = _random_stream(12, 40, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=12, weight_mode="operator_schmidt")

    plan = finder.layered(block_size=4)
    assert isinstance(plan, TreePlan)
    assert plan.n == 12
    # Top tensor is fixed ternary once there are at least three blocks.
    assert len(plan.children[plan.root]) == 3
    # The blocking layer groups block_size leaves per blocking node.
    assert 4 in {len(ch) for ch in plan.children.values() if ch}
    # Direct build matches recommend_layered's plan for the same block_size.
    recommended = finder.recommend_layered(block_sizes=(4,))
    assert plan.children == recommended["plan"].children


def test_layered_greedy_refinement_improves_without_changing_tree_shape():
    """Planning swaps leaf labels but preserves the immutable TTN topology."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder([(cnot, (0, 2))], n=4, max_arity=2)
    initial = TreePlan.build_layered(range(4), block_size=2)

    choice = finder.recommend_layered(
        block_sizes=(2,),
        order=range(4),
        refine="greedy",
        refine_budget=3,
    )
    candidate = choice["candidates"][0]
    refined = choice["plan"]

    assert choice["refine"] == "greedy"
    assert refined.children == initial.children
    assert finder.score(refined) < finder.score(initial)
    assert refined.tree_distance(0, 2) == 2
    assert candidate["planning"]["refinement"]["accepted_moves"] >= 1


def test_hybrid_layout_objective_reports_normalized_combined_cost():
    """Hybrid selection combines path and operator-Schmidt edge-load costs."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder(
        [(cnot, (0, 3)), (cnot, (1, 2))],
        n=4,
        objective="hybrid",
        hybrid_weights={"path": 1.0, "max_edge_load": 2.0},
    )
    plan = finder.recommend_arities((2, 3))["plan"]
    report = finder.report(plan)

    assert report["objective"] == "hybrid"
    assert report["hybrid_weights"] == (1.0, 2.0, 0.0)
    assert np.isfinite(report["hybrid_cost"])
    assert report["total_edge_load"] is not None


def test_layered_nevergrad_search_is_optional_and_pre_simulation(monkeypatch):
    """Nevergrad refines only a returned plan and is optional at import time."""
    class FakeArray:
        def __init__(self, *, init):
            self.value = np.asarray(init)

        def set_bounds(self, _lower, _upper):
            return self

    class FakeOptimizer:
        def __init__(self, *, parametrization, budget):
            self.parametrization = parametrization
            self.budget = budget

        def minimize(self, loss):
            loss(self.parametrization.value)
            proposal = np.array([0.0, 2.0, 1.0, 3.0])
            loss(proposal)
            return types.SimpleNamespace(value=proposal)

    fake_nevergrad = types.SimpleNamespace(
        p=types.SimpleNamespace(Array=FakeArray),
        optimizers=types.SimpleNamespace(registry={"OnePlusOne": FakeOptimizer}),
    )
    monkeypatch.setitem(sys.modules, "nevergrad", fake_nevergrad)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    finder = TreeLayoutFinder([(cnot, (0, 2))], n=4, max_arity=2)
    initial = TreePlan.build_layered(range(4), block_size=2)

    choice = finder.recommend_layered(
        block_sizes=(2,),
        order=range(4),
        search="nevergrad",
        search_budget=4,
        seed=17,
    )
    search_info = choice["candidates"][0]["planning"]["search"]

    assert choice["search"] == "nevergrad"
    assert choice["plan"].children == initial.children
    assert finder.score(choice["plan"]) <= finder.score(initial)
    assert search_info["method"] == "nevergrad"
    assert search_info["evaluations"] == 2


def test_nevergrad_layout_search_explains_missing_optional_dependency(monkeypatch):
    """The optional Nevergrad dependency fails with an actionable message."""
    monkeypatch.setitem(sys.modules, "nevergrad", None)
    finder = TreeLayoutFinder([], n=4, max_arity=2)

    with pytest.raises(ImportError, match=r"pepsy\[layout\]"):
        finder.recommend_layered(
            block_sizes=(2,), order=range(4), search="nevergrad"
        )


def test_tree_qubit_order_honors_configured_dense_limit(monkeypatch):
    """The layered spectral order uses the finder's dense-size policy."""
    import pepsy.optimizers.tree.layout as tree_layout

    seen = {}

    def ordered(sites, _weights, *, dense_max):
        seen["dense_max"] = dense_max
        return list(sites)

    monkeypatch.setattr(tree_layout, "_gate_stream_spectral_order", ordered)
    finder = TreeLayoutFinder([], n=6, dense_max=17)

    assert finder.qubit_order() == list(range(6))
    assert seen["dense_max"] == 17


def test_treeplan_max_bond_cut_is_structural():
    """`max_bond_cut` is the widest qubit bipartition, set by shape alone."""
    order = list(range(16))
    # block_size=3 groups into an even 2+2+2 ternary top -> widest cut is 6.
    assert TreePlan.build_layered(order, block_size=3).max_bond_cut() == 6
    # block_size=4 forces a {1,1,2}-block top -> one 8-vs-8 cut.
    assert TreePlan.build_layered(order, block_size=4).max_bond_cut() == 8
    # A single leaf has no bonds.
    assert TreePlan.build_layered([0]).max_bond_cut() == 0


def test_recommend_layered_chi_aware_prefers_exact_structure():
    """With ``chi`` set, the layered recommendation avoids chi-overflow bonds."""
    rng = np.random.default_rng(211)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, weight_mode="operator_schmidt")

    # chi-blind candidates always carry ``max_bond_cut`` but no chi fields.
    blind = finder.recommend_layered(block_sizes=(3, 4))
    assert "chi" in blind and blind["chi"] is None
    for c in blind["candidates"]:
        assert "max_bond_cut" in c
        assert "chi_overflow" not in c and "exact_at_chi" not in c

    # chi=64 fits a 6-qubit cut exactly (2**6) but not an 8-qubit cut (2**8):
    # block_size=4 (cut 8) overflows, block_size=3 (cut 6) is exact.
    aware = finder.recommend_layered(block_sizes=(3, 4), chi=64)
    assert aware["chi"] == 64
    assert aware["recommended_block_size"] == 3
    assert aware["plan"].max_bond_cut() <= 6
    by_bs = {c["block_size"]: c for c in aware["candidates"]}
    assert by_bs[3]["exact_at_chi"] and by_bs[3]["chi_overflow"] == 0.0
    assert not by_bs[4]["exact_at_chi"] and by_bs[4]["chi_overflow"] == 2.0
    # The recommended candidate always has the minimum chi_overflow.
    overflows = [c["chi_overflow"] for c in aware["candidates"]]
    assert by_bs[aware["recommended_block_size"]]["chi_overflow"] == min(overflows)


def test_recommend_layered_inherits_finder_chi_unless_overridden():
    """The direct layered recommendation follows the finder's chi policy."""
    rng = np.random.default_rng(216)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(
        stream, n=16, chi=64, weight_mode="operator_schmidt"
    )

    inherited = finder.recommend_layered(block_sizes=(3, 4))
    assert inherited["chi"] == 64
    assert inherited["recommended_block_size"] == 3

    blind = finder.recommend_layered(block_sizes=(3, 4), chi=None)
    assert blind["chi"] is None


def test_recommend_arities_chi_aware_minimizes_overflow():
    """``chi``-aware arity search prefers a structure exact at ``chi``."""
    rng = np.random.default_rng(212)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, objective="congestion")

    rec = finder.recommend_arities((2, 3, 4), chi=64)
    assert rec["chi"] == 64
    for c in rec["candidates"]:
        assert {"max_bond_cut", "chi_overflow", "exact_at_chi"} <= set(c)
    recommended = next(
        c for c in rec["candidates"]
        if c["max_arity"] == rec["recommended_max_arity"]
    )
    overflows = [c["chi_overflow"] for c in rec["candidates"]]
    assert recommended["chi_overflow"] == min(overflows)
    # recommend_layout forwards chi unchanged.
    assert finder.recommend_layout((2, 3, 4), chi=64)["chi"] == 64


def test_recommend_layered_rejects_bad_chi():
    """A non-positive ``chi`` budget is rejected."""
    finder = TreeLayoutFinder([], n=4, structure="balanced")
    with pytest.raises(ValueError, match="chi must be a positive integer"):
        finder.recommend_layered(block_sizes=(2,), chi=0)


def test_layout_finder_uses_binary_ternary_root_by_default():
    """The finder default is fixed binary below a ternary virtual root."""
    rng = np.random.default_rng(213)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, weight_mode="operator_schmidt")

    assert finder.arity_candidates is None
    assert finder.chi is None
    searched = finder.run()
    assert searched.top_arity == 3
    assert searched.is_binary()

    # Candidate arity search remains available explicitly.
    blind = finder.recommend_arities((2, 3, 4))
    assert blind["chi"] is None

    # A scalar max_arity opts back into a single fixed binary tree.
    fixed = TreeLayoutFinder(stream, n=16, max_arity=2,
                             weight_mode="operator_schmidt")
    assert fixed.arity_candidates is None
    assert fixed.run().is_binary()


def test_layout_finder_run_accepts_search_overrides(monkeypatch):
    """Tree ``run`` mirrors MPS by accepting per-run quality-search controls."""
    finder = TreeLayoutFinder([], n=4, max_arity=(2, 3), chi=8)
    captured = {}
    recommend_arities = finder.recommend_arities

    def capture(max_arities, **kwargs):
        captured.update(kwargs)
        return recommend_arities(max_arities, **kwargs)

    monkeypatch.setattr(finder, "recommend_arities", capture)
    plan = finder.run(
        chi=None,
        refine="greedy",
        refine_budget=2,
        search=None,
        search_budget=7,
        seed=11,
        nevergrad_optimizer="OnePlusOne",
        progbar=True,
    )

    assert isinstance(plan, TreePlan)
    assert captured == {
        "chi": None,
        "refine": "greedy",
        "refine_budget": 2,
        "topology_refine": None,
        "topology_budget": None,
        "search": None,
        "search_budget": 7,
        "seed": 11,
        "nevergrad_optimizer": "OnePlusOne",
        "progbar": True,
    }
    assert finder._last_arity_recommendation["refine"] == "greedy"
    assert finder._last_arity_recommendation["chi"] is None


def test_layout_finder_explicit_arity_search_is_chi_aware():
    """An explicit candidate search remains ``chi``-aware."""
    rng = np.random.default_rng(214)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    finder = TreeLayoutFinder(stream, n=16, chi=64,
                              weight_mode="operator_schmidt")

    assert finder.chi == 64
    searched = finder.run()
    aware = finder.recommend_arities((2, 3, 4), chi=64)
    assert searched.top_arity == 3
    assert searched.is_binary()
    # The chi-aware search never overflows chi by more than the binary tree.
    by_arity = {c["max_arity"]: c for c in aware["candidates"]}
    chosen = by_arity[aware["recommended_max_arity"]]
    assert chosen["chi_overflow"] <= by_arity[2]["chi_overflow"]


def test_optimizer_uses_binary_ternary_root_by_default():
    """TreeOptimizer shares the fixed binary/ternary-root default."""
    rng = np.random.default_rng(215)
    stream = _random_stream(16, 60, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=16, chi=64,
                        layout_weight_mode="operator_schmidt", run=False)

    finder = TreeLayoutFinder(stream, n=16, max_arity=2, top_arity=3, chi=64,
                              weight_mode="operator_schmidt")
    assert opt.plan.children == finder.run().children
    assert opt.plan.top_arity == 3

    # A scalar max_arity=2 forces a fixed binary tree through the optimizer.
    fixed = TreeOptimizer(stream, n=16, chi=64, max_arity=2,
                          layout_weight_mode="operator_schmidt", run=False)
    assert fixed.plan.is_binary()


def test_layout_and_entangled_state_handoff_is_explicit():
    """A layout finder and an entangled TTN can be handed off safely."""
    plan_finder = TreeLayoutFinder([], n=4, structure="balanced")
    state_plan = plan_finder.run()
    state = TreeTensorNetwork.rand(state_plan, D=2, seed=104)
    before = state.to_statevector()

    opt = TreeOptimizer(
        None,
        layout=plan_finder,
        state=state,
        run=False,
    )

    assert _fidelity(before, opt.to_dense()) > 1 - 1e-10
    assert opt.layout_report()["is_binary"]
    with pytest.raises(TypeError, match="TreePlan"):
        TreeOptimizer(None, tree=state, n=4, run=False)
    with pytest.raises(TypeError, match="not a TreeTensorNetwork"):
        TreeLayoutFinder(state)


def test_product_ttn_is_remounted_exactly_on_a_requested_new_layout():
    """A product TTN can safely move to a different tree geometry."""
    source_plan = TreePlan.from_order(range(4), structure="balanced")
    target_plan = TreePlan.from_order((0, 2, 1, 3), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    source = TreeOptimizer([(h, 1)], tree=source_plan, chi=8).tn.copy()
    source.exponent = np.log10(3.0)

    with pytest.warns(UserWarning, match="product TreeTensorNetwork"):
        opt = TreeOptimizer(None, state=source, tree=target_plan, run=False)

    assert opt.plan is target_plan
    assert opt.max_bond() == 1
    assert opt.tn.exponent == pytest.approx(source.exponent)
    assert np.allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(source.to_dense()).reshape(-1),
    )


def test_entangled_ttn_relayout_is_rejected_before_any_lossy_conversion():
    """Changing an entangled TTN's geometry requires an explicit conversion."""
    source_plan = TreePlan.from_order(range(4), structure="balanced")
    target_plan = TreePlan.from_order((0, 2, 1, 3), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    source = TreeOptimizer(
        [(h, 0), (cnot, (0, 1))], tree=source_plan, chi=8
    ).tn.copy()
    assert source.max_bond() > 1

    with pytest.raises(ValueError, match="potentially lossy relayout"):
        TreeOptimizer(None, state=source, tree=target_plan, run=False)


def test_product_mps_is_accepted_and_exactly_mounted_on_the_tree():
    """A bond-one MPS is a geometry-neutral product-state input."""
    mps = qtn.MPS_computational_state("010", dtype="complex128")
    plan = TreePlan.from_order((2, 0, 1), structure="balanced")

    opt = TreeOptimizer(None, state=mps, tree=plan, run=False)

    assert opt.plan is plan
    assert opt.max_bond() == 1
    assert np.allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(mps.to_dense()).reshape(-1),
    )


def test_entangled_mps_initial_state_is_rejected():
    """An MPS with a nontrivial virtual bond cannot be silently tree-remapped."""
    mps = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=45
    )
    assert mps.max_bond() > 1

    with pytest.raises(TypeError, match=r"max_bond\(\) == 1"):
        TreeOptimizer(None, state=mps, run=False)


def test_public_api_exports_tree_optimizer():
    """TreeOptimizer is exposed through the public namespaces."""
    import pepsy

    assert pepsy.TreeOptimizer is TreeOptimizer
    from pepsy.optimizers import TreeOptimizer as FromOptimizers

    assert FromOptimizers is TreeOptimizer


# -- diagnostics --------------------------------------------------------------


def test_layout_report_summarizes_quality():
    """TreeLayoutFinder.report exposes geodesic + score diagnostics."""
    rng = np.random.default_rng(11)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    finder = TreeLayoutFinder(stream, n=n)
    rep = finder.report()
    assert rep["n_qubits"] == n
    assert rep["n_interacting_pairs"] >= 1
    assert rep["max_path"] >= 1
    assert rep["weighted_mean_path"] > 0.0
    # the chosen quality structure is no worse than a balanced index tree
    assert rep["score"] <= rep["balanced_score"] + 1e-9


def test_tree_layout_finder_plot_defaults_to_tent():
    """The public plot shows the structural tent, not gate-route overlays."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2, top_arity=2)
    plan = finder.run()
    assert plan.is_binary()
    assert len(plan.children[plan.root]) == 2
    fig, ax = finder.plot(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert fig is ax.figure
    assert ax.get_title() == ""
    assert not ax.patches
    assert len(fig.axes) == 1
    assert not ax.axison  # schematic-style presentation by default
    assert not ax.texts
    assert len(ax.collections) == 1 + sum(
        not plan.is_leaf(node) for node in plan.nodes()
    )
    plt.close(fig)


def test_tree_layout_finder_can_hide_gate_paths_for_structural_view():
    """The structural view makes the binary TTN edges unambiguous."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot(
        plan,
        lattice=False,
        show_gate_connectivity=False,
        show_edge_arrows=False,
    )

    assert len(ax.lines) == len(plan.nodes()) - 1
    assert not ax.patches
    assert not ax.texts
    assert not ax.axison
    plt.close(fig)


def test_tree_layout_finder_plot_tent_draws_hierarchy_over_raw_graph():
    """Tent plotting separates the binary hierarchy from raw connectivity."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert plan.is_binary()
    assert fig is ax.figure
    assert not ax.patches
    assert not ax.texts
    assert not ax.axison
    assert len(ax.lines) >= len(plan.nodes()) - 1
    assert len(ax.collections) == 1 + sum(
        not plan.is_leaf(node) for node in plan.nodes()
    )
    plt.close(fig)


def test_tree_layout_tent_can_hide_physical_leaf_nodes():
    """Physical plus-mark backdrops can replace tree leaf circles."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (3, 1))],
        n=4,
        max_arity=2,
    )
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        lattice=False,
        show_gate_connectivity=False,
        show_leaf_nodes=False,
    )

    assert len(ax.collections) == sum(
        not plan.is_leaf(node) for node in plan.nodes()
    )
    assert len(ax.lines) == len(plan.nodes()) - 1
    plt.close(fig)


def test_tree_layout_scale_colors_do_not_depend_on_gate_stream_length():
    """Scale coloring remains fixed when the gate stream changes."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plan = TreePlan.from_order(range(8), structure="balanced", max_arity=2)
    streams = [
        [(pepsy.cnot(), (0, 7))],
        [
            (pepsy.cnot(), (0, 7)),
            (pepsy.cnot(), (1, 6)),
            (pepsy.cnot(), (2, 5)),
            (pepsy.cnot(), (3, 4)),
        ],
    ]
    structural_colors = []
    scale_node_colors = []
    for gates in streams:
        finder = TreeLayoutFinder(gates, n=8, max_arity=2)
        fig, ax = finder.plot_tent(
            plan,
            color_by="scale",
            edge_color=None,
            show_edge_arrows=False,
        )
        # Gate-connectivity overlays are disabled by default, so only the
        # one-dimensional physical lattice precedes the hierarchy edges.
        background_lines = len(plan.leaves()) - 1
        structural_colors.append(
            tuple(
                tuple(line.get_color())
                for line in ax.lines[background_lines:]
            )
        )
        scale_node_colors.append(
            tuple(
                tuple(collection.get_facecolors()[0])
                for collection in ax.collections
            )
        )
        assert len(fig.axes) == 1
        assert len(ax.collections) == 1 + sum(
            not plan.is_leaf(node) for node in plan.nodes()
        )
        plt.close(fig)

    assert structural_colors[0] == structural_colors[1]
    assert scale_node_colors[0] == scale_node_colors[1]
    assert len(set(structural_colors[0])) > 1
    assert len(set(scale_node_colors[0])) > 1


def test_tree_layout_tent_edges_match_order_colors_by_default():
    """Tent hierarchy edges follow the default order color palette."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot_tent(plan, color_by="scale")

    background_lines = len(plan.leaves()) - 1
    hierarchy_colors = {
        line.get_color() for line in ax.lines[background_lines:]
    }
    assert len(hierarchy_colors) > 1
    assert not ax.patches
    plt.close(fig)


def test_tree_layout_tent_can_highlight_leaf_edges():
    """The physical-to-first-parent layer can use a contrasting color."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
        top_arity=2,
    )
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        lattice=False,
        show_gate_connectivity=False,
        leaf_edge_color="#2563eb",
    )

    line_index = 0
    for parent, children in plan.children.items():
        for child in children:
            if plan.is_leaf(child):
                assert ax.lines[line_index].get_color() == "#2563eb"
            line_index += 1
    assert line_index == len(plan.nodes()) - 1
    plt.close(fig)


def test_tree_layout_tent_colored_edges_match_child_nodes():
    """Colored incoming edges use the same scale color as their child node."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
    )
    plan = finder.run()
    fig, ax = finder.plot_tent(
        plan,
        color_by="scale",
        edge_color=None,
        show_edge_arrows=False,
    )

    background_lines = plan.n - 1
    internal_nodes = [node for node in plan.nodes() if not plan.is_leaf(node)]
    node_colors = {
        node: tuple(collection.get_facecolors()[0])
        for node, collection in zip(internal_nodes, ax.collections[1:])
    }
    hierarchy_lines = ax.lines[background_lines:]
    line_index = 0
    for parent, children in plan.children.items():
        for child in children:
            if not plan.is_leaf(child):
                assert tuple(
                    hierarchy_lines[line_index].get_color()
                ) == pytest.approx(node_colors[child])
            line_index += 1
    assert line_index == len(hierarchy_lines)
    plt.close(fig)


def test_tree_layout_tent_validates_arrow_size():
    """Arrow marker sizing rejects values Matplotlib cannot render usefully."""
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 1))], n=2, max_arity=2
    )
    plan = finder.run()

    with pytest.raises(ValueError, match="arrow_size"):
        finder.plot_tent(plan, arrow_size=0.0)


def test_tree_layout_finder_plot_rubberband_is_axis_free_and_unlabeled():
    """Rubberband plots show clusters without plot text or axes."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (pepsy.cnot(), (0, 3)),
        (pepsy.cnot(), (3, 1)),
        (pepsy.cnot(), (1, 2)),
    ]
    finder = TreeLayoutFinder(gates, n=4, max_arity=2)
    plan = finder.run()
    fig, ax = finder.plot_rubberband(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert fig is ax.figure
    assert ax.get_title() == ""
    assert not ax.axison
    assert not ax.texts
    assert len(ax.patches) >= 1
    plt.close(fig)


def test_tree_layout_quality_order_enables_bounded_refinement(monkeypatch):
    """Tree order='quality' mirrors the MPS high-quality mode."""
    monkeypatch.setitem(sys.modules, "nevergrad", None)
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
        order="quality",
    )
    captured = {}

    def fake_improve(plan, *, chi, settings, progbar=False):
        captured.update(settings)
        return plan, {"method": "test"}

    monkeypatch.setattr(finder, "_improve_plan", fake_improve)
    plan = finder.run()

    assert plan.n == 4
    assert finder.objective == "full_tree"
    assert captured["refine"] == "greedy"
    assert captured["topology_refine"] == "subtree"
    assert captured["search"] == "anneal"
    assert captured["search_budget"] == finder.search_budget


def test_tree_layout_quality_run_upgrades_a_fast_finder(monkeypatch):
    """The explicit quality run is the full-tree mode even after construction."""
    monkeypatch.setitem(sys.modules, "nevergrad", None)
    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
    )
    captured = {}

    def fake_improve(plan, *, chi, settings, progbar=False):
        captured.update(settings)
        return plan, {"method": "test"}

    monkeypatch.setattr(finder, "_improve_plan", fake_improve)
    plan = finder.run(order="quality")

    assert plan.n == 4
    assert finder.objective == "full_tree"
    assert captured["topology_refine"] == "subtree"
    assert captured["search"] == "anneal"


def test_tree_layout_nni_refinement_changes_binary_topology():
    """Quality refinement can move a correlated subtree, not only labels."""
    cnot = pepsy.cnot()
    finder = TreeLayoutFinder(
        [(cnot, (0, 2)), (cnot, (0, 3))],
        n=4,
        max_arity=2,
        top_arity=2,
        objective="path",
    )
    initial = TreePlan.from_order(
        range(4), structure="balanced", max_arity=2, top_arity=2,
    )

    refined, planning = finder._refine_plan_topology(
        initial,
        chi=None,
        budget=4,
    )

    assert refined.is_binary()
    assert refined.children != initial.children
    assert finder.score(refined) < finder.score(initial)
    assert planning["accepted_moves"] >= 1


def test_tree_layout_temporal_weights_apply_to_paths_and_edge_loads():
    """Recent-event weighting affects both locality and Schmidt-load scoring."""
    cnot = pepsy.cnot()
    gates = [(cnot, (0, 1)), (cnot, (2, 3))]
    full = TreeLayoutFinder(gates, n=4, max_arity=2, objective="congestion")
    recent = TreeLayoutFinder(
        gates,
        n=4,
        max_arity=2,
        objective="congestion",
        time_decay=0.5,
        time_window=1,
    )

    assert full.temporal_factors == (1.0, 1.0)
    assert recent.temporal_factors == (0.0, 1.0)
    assert sum(recent.event_weights) == pytest.approx(1.0)
    assert sum(recent.edge_loads(recent.run()).values()) < sum(
        full.edge_loads(full.run()).values()
    )
    report = recent.report()
    assert report["time_decay"] == pytest.approx(0.5)
    assert report["time_window"] == 1
    assert report["active_events"] == 1

    opt = TreeOptimizer(
        gates,
        n=4,
        max_arity=2,
        layout_time_decay=0.5,
        layout_time_window=1,
        run=False,
    )
    assert opt.layout_finder.time_window == 1
    assert opt.layout_finder.time_decay == pytest.approx(0.5)


def test_tree_layout_order_rejects_non_quality_modes():
    """Tree layouts expose quality mode rather than 1-D order names."""
    finder = TreeLayoutFinder([], n=4, max_arity=2)
    with pytest.raises(ValueError, match="order"):
        finder.run(order="input")


def test_tree_layout_rubberband_defaults_to_cotengra_ordered_colors():
    """Default rubberbands use distinct post-order Spectral colors."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finder = TreeLayoutFinder(
        [(pepsy.cnot(), (0, 3)), (pepsy.cnot(), (1, 2))],
        n=4,
        max_arity=2,
    )
    fig, ax = finder.plot_rubberband(
        finder.run(),
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    expected = matplotlib.colormaps["Spectral"]
    assert np.allclose(
        ax.patches[0].get_edgecolor()[:3], expected(0.0)[:3]
    )
    assert np.allclose(
        ax.patches[-1].get_edgecolor()[:3], expected(1.0)[:3]
    )
    assert ax.patches[0].get_zorder() > ax.patches[-1].get_zorder()
    plt.close(fig)


def test_tree_optimizer_plot_layout_with_explicit_plan_is_non_mutating():
    """The tree optimizer wrapper plots an explicit plan without replay."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [(pepsy.cnot(), (0, 3))]
    plan = TreeLayoutFinder(gates, n=4, max_arity=2).run()
    opt = TreeOptimizer(gates, tree=plan, run=False)
    before = opt.to_dense().copy()
    fig, _ = opt.plot_layout(site_coords={q: (q, 0) for q in range(4)})

    assert np.allclose(opt.to_dense(), before)
    plt.close(fig)


def test_bond_report_reflects_chi():
    """bond_report caps at chi and counts the tree tensors."""
    rng = np.random.default_rng(12)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=4)
    rep = opt.bond_report()
    assert rep["chi"] == 4
    assert rep["max_bond"] <= 4
    assert rep["mean_bond"] <= rep["max_bond"]
    assert rep["n_tensors"] == len(opt.plan.nodes())


def test_estimate_bonds_uses_crossing_operator_schmidt_ranks():
    """The paper dry-run multiplies ranks only on edges crossed by a gate."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    rng = np.random.default_rng(0)
    generic = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    opt = TreeOptimizer(None, n=4, tree=plan, chi=4, run=False)
    before = opt.to_dense()

    report = opt.estimate_bonds([(cnot, (0, 3)), (generic, (0, 3))])
    path_edges = {
        tuple(sorted(edge))
        for edge in zip(
            plan.node_path(plan.leaf_of_qubit[0], plan.leaf_of_qubit[3]),
            plan.node_path(plan.leaf_of_qubit[0], plan.leaf_of_qubit[3])[1:],
        )
    }

    assert report["max_bond"] == 8  # rank(CNOT)=2, rank(generic)=4
    assert report["requires_truncation"]
    assert set(report["edge_bonds"]) == {
        tuple(sorted((parent, child)))
        for parent, children in plan.children.items()
        for child in children
    }
    assert all(report["edge_bonds"][edge] == 8 for edge in path_edges)
    assert any(report["edge_bonds"][edge] == 1 for edge in report["edge_bonds"]
               if edge not in path_edges)
    assert report["events"][0]["crossing_edges"]
    assert set(report["events"][0]["crossing_edges"].values()) == {2}
    assert set(report["events"][1]["crossing_edges"].values()) == {4}
    assert np.allclose(before, opt.to_dense())  # diagnostic is non-mutating


def test_estimate_bonds_ignores_single_site_and_control_events():
    """One-site operations and measurements do not grow the dry-run bound."""
    z = np.diag([1.0, -1.0]).astype(complex)
    opt = TreeOptimizer(
        [(z, 0), ("measure", "Z", 1, +1)], n=3, chi=2, run=False
    )
    report = opt.estimate_bonds()
    assert report["max_bond"] == 1
    assert all(not event["crossing_edges"] for event in report["events"])


def test_preflight_reports_and_rejects_resource_limits():
    """Preflight protects replay without changing the live state."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(None, n=4, tree=plan, chi=4, run=False)
    report = opt.preflight(
        [(cnot, (0, 3))], max_bond=1, raise_on_error=False
    )
    assert report["ok"] is False
    assert report["violations"]
    with pytest.raises(MemoryError, match="max_bond"):
        opt.preflight([(cnot, (0, 3))], max_bond=1)
    with pytest.raises(MemoryError, match="estimated max bond"):
        TreeOptimizer(
            [(cnot, (0, 3))], n=4, tree=plan,
            max_intermediate_bond=1,
        )

    with pytest.raises(MemoryError, match="max_operator_qubits"):
        TreeOptimizer(None, n=3, max_operator_qubits=2).apply_gate(
            _rand_unitary(3, np.random.default_rng(1)), (0, 1, 2)
        )


def test_truncation_report_tracks_per_edge_discarded_weight():
    """Tracked runs expose local spectra and discarded weights per edge."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 3))],
        n=4,
        tree=plan,
        chi=1,
        track_truncation=True,
    )

    report = opt.truncation_report()
    assert report["track_truncation"] is True
    assert report["n_events"] == len(opt.truncation_history) > 0
    assert report["n_tracked"] == report["n_events"]
    assert report["max_discarded_fraction"] == pytest.approx(0.5)
    assert any(event["kind"] == "compress" for event in report["events"])
    updates = report["updates"]
    assert len(updates) == 2
    assert updates[0]["support"] == (0,)
    assert updates[0]["edge_count"] == 0
    assert updates[0]["relative_discarded_weight"] == pytest.approx(0.0)
    assert updates[1]["support"] == (0, 3)
    assert updates[1]["edge_count"] == 4
    assert updates[1]["relative_discarded_weight"] == pytest.approx(0.5)
    assert updates[1]["cumulative_relative_discarded_weight"] == pytest.approx(0.5)
    for event in report["events"]:
        assert event["after_bond"] <= event["before_bond"]
        assert event["spectrum_rank"] is not None
        assert event["discarded_weight"] >= 0.0
        assert event["discarded_fraction"] >= 0.0


def test_truncation_history_keeps_fast_untracked_path_cheap():
    """The default path records dimensions without probing full spectra."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 3))], n=4, chi=1, track_truncation=False
    )

    report = opt.truncation_report()
    assert report["track_truncation"] is False
    assert report["n_events"] > 0
    assert report["n_tracked"] == 0
    assert report["total_discarded_weight"] is None
    assert all(event["discarded_weight"] is None for event in report["events"])


def test_track_truncation_warns_and_skips_impossible_spectra():
    """Tracking warns, while lossless within-cap edges still use QR only."""
    with pytest.warns(UserWarning, match="track_truncation=True"):
        opt = TreeOptimizer(
            [(pepsy.cnot(), (0, 3))],
            n=4,
            chi=16,
            cutoff=0.0,
            track_truncation=True,
        )

    assert opt.track_truncation is True
    assert opt.truncation_history
    assert all(event["spectrum_rank"] is None for event in opt.truncation_history)


def test_repeated_direct_gate_reuses_factorization(monkeypatch):
    """Repeated immutable gate objects do not repeat their operator SVD."""
    original_split = qtn.Tensor.split
    svd_calls = []

    def traced_split(tensor, *args, **kwargs):
        if kwargs.get("method") == "svd":
            svd_calls.append(tensor)
        return original_split(tensor, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "split", traced_split)
    gate = pepsy.cnot()
    opt = TreeOptimizer(None, n=4, chi=16, cutoff=0.0, run=False)
    opt.apply_2q(gate, 0, 3)
    first_count = len(svd_calls)
    opt.apply_2q(gate, 0, 3)

    assert len(svd_calls) == first_count
    assert sum(key[0] == "direct" for key in opt._gate_factor_cache) == 1


def test_repeated_two_site_support_reuses_only_immutable_path():
    """Path caching never assumes the current centre or traversal direction."""
    opt = TreeOptimizer(None, n=8, chi=16, cutoff=0.0, run=False)

    leaf_a, leaf_b, path = opt._cached_two_site_path(0, 7)
    reverse_a, reverse_b, reverse_path = opt._cached_two_site_path(7, 0)
    cached_a, cached_b, cached_path = opt._cached_two_site_path(0, 7)

    assert (leaf_a, leaf_b) == (cached_a, cached_b)
    assert (reverse_a, reverse_b) == (leaf_b, leaf_a)
    assert reverse_path == path[::-1]
    assert cached_path is path


def test_adjacent_two_tensor_contract_matches_quimb():
    """The direct one-edge backend contraction preserves Quimb ordering."""
    rng = np.random.default_rng(912)
    left = qtn.Tensor(
        rng.standard_normal((2, 4, 3)), inds=("a", "edge", "b"),
    )
    right = qtn.Tensor(
        rng.standard_normal((4, 5, 2)), inds=("edge", "c", "d"),
    )

    fast = _contract_two_tensors(left, right, shared_ind="edge")
    reference = qtn.tensor_contract(left, right)

    assert fast.inds == reference.inds
    assert np.allclose(fast.data, reference.data)


def test_adjacent_dense_contract_avoids_generic_quimb_dispatch(monkeypatch):
    """The shared dense hot path does not rebuild a generic contraction."""
    left = qtn.Tensor(
        np.arange(12.0).reshape(3, 4), inds=("a", "edge"),
    )
    right = qtn.Tensor(
        np.arange(20.0).reshape(4, 5), inds=("edge", "b"),
    )

    def unexpected_generic_contract(*args, **kwargs):
        raise AssertionError("dense one-edge contraction used generic Quimb")

    monkeypatch.setattr(qtn, "tensor_contract", unexpected_generic_contract)
    fast = _contract_two_tensors(left, right, shared_ind="edge")

    assert fast.inds == ("a", "b")
    np.testing.assert_allclose(
        fast.data, np.asarray(left.data) @ np.asarray(right.data)
    )


def test_parallel_subtree_messages_match_serial():
    """Independent dense QR message waves preserve the serial result."""
    rng = np.random.default_rng(818)
    operator, _ = np.linalg.qr(
        rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    )
    serial = TreeOptimizer(
        None, n=8, chi=16, cutoff=0.0, subtree_workers=1, run=False,
    )
    parallel = TreeOptimizer(
        None, n=8, chi=16, cutoff=0.0, subtree_workers=3, run=False,
    )

    serial.apply_subtree_operator(operator, (0, 2, 5))
    parallel.apply_subtree_operator(operator, (0, 2, 5))

    assert _fidelity(serial.to_dense(), parallel.to_dense()) > 1 - 1e-12


def test_convergence_sweep_reports_rising_fidelity():
    """convergence_sweep reuses one tree and reports monotone fidelity."""
    rng = np.random.default_rng(13)
    n = 8
    stream = _random_stream(n, 60, rng, two_qubit_frac=0.6)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    recs = TreeOptimizer.convergence_sweep(
        stream, n=n, chi_values=(8, 2, 4, 64), ops=[(z, 0), (z, n - 1)]
    )
    # sorted ascending internally
    assert [r["chi"] for r in recs] == [2, 4, 8, 64]
    fids = [r["fidelity"] for r in recs]
    assert all(f is not None for f in fids)
    for a, b in zip(fids, fids[1:]):
        assert b >= a - 1e-9
    assert fids[-1] > 1 - 1e-6
    assert recs[0]["max_drift"] is None
    assert all(r["max_drift"] is not None for r in recs[1:])
    assert all(len(r["expectations"]) == 2 for r in recs)
    assert all(r["max_bond"] <= r["chi"] for r in recs)


def test_convergence_sweep_skips_fidelity_when_large():
    """The dense fidelity reference is skipped past dense_cap."""
    rng = np.random.default_rng(14)
    n = 6
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.6)
    recs = TreeOptimizer.convergence_sweep(
        stream, n=n, chi_values=(2, 4), dense_cap=8
    )
    assert all(r["fidelity"] is None for r in recs)


def test_convergence_sweep_reuses_generator_stream():
    """A one-shot gate iterator produces the same sweep as a list."""
    rng = np.random.default_rng(141)
    stream = _random_stream(4, 8, rng, two_qubit_frac=0.6)
    kwargs = dict(n=4, chi_values=(1, 2, 4), dense_cap=0)
    from_list = TreeOptimizer.convergence_sweep(stream, **kwargs)
    from_generator = TreeOptimizer.convergence_sweep(
        (entry for entry in stream), **kwargs
    )
    assert [
        (rec["chi"], rec["max_bond"], rec["norm"])
        for rec in from_generator
    ] == pytest.approx([
        (rec["chi"], rec["max_bond"], rec["norm"])
        for rec in from_list
    ])


# -- stability / speed hardening ----------------------------------------------


def test_fresh_state_is_canonical_at_root():
    """A newly built product state is canonical with the root as centre.

    Every virtual bond starts at dimension 1, so each tensor is trivially
    isometric: the tree is already normalised with the root as orthogonality
    centre, and no canonicalisation is needed before the first gate.
    """
    opt = TreeOptimizer(None, n=8, chi=16)
    assert opt.center == opt.plan.root
    # one-site canonical norm (uses the tracked centre) is exactly 1
    assert abs(opt.norm() - 1.0) < 1e-12
    # ...and it agrees with the full doubled-tree contraction
    opt.center = None
    assert abs(opt.norm() - 1.0) < 1e-12


def test_tree_tensor_network_has_native_whole_tree_compress():
    """Direct TTN compression leaves one validated tree canonical centre."""
    plan = TreePlan.from_order(range(6), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=31, canonicalize=False)

    assert state.compress(
        max_bond=None,
        cutoff=0.0,
        center=plan.root,
    ) is state
    assert state.orthogonality_center == plan.root
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True) is state


def test_tree_mpo_rank_aware_compression_records_edge_order():
    """TreeMPO compression uses and reports the native rank-aware order."""
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    rng = np.random.default_rng(32)
    dense = rng.normal(size=(2**4, 2**4))
    operator = TreeMPO.from_dense(plan, dense)

    operator.compress(max_bond=4, cutoff=1e-12)
    report = operator.pepsy_compression_report
    assert report["order"] == "rank"
    assert len(report["edge_order"]) == len(plan.nodes()) - 1
    assert operator.validate() is operator
    assert operator.max_bond() <= 4


def test_two_qubit_gate_rejects_repeated_qubit():
    """A two-qubit gate on a single qubit is rejected loudly."""
    rng = np.random.default_rng(15)
    opt = TreeOptimizer(None, n=4, chi=8)
    with pytest.raises(ValueError, match="two distinct qubits"):
        opt.apply_gate(_rand_unitary(2, rng), (2, 2))


@pytest.mark.parametrize("where", [(0, 3, 5), (1, 2, 6), (0, 1, 2)])
def test_apply_subtree_operator_3q_matches_dense(where):
    """A three-qubit gate over its spanning subtree matches the dense state."""
    rng = np.random.default_rng(20)
    n = 7
    stream = _random_stream(n, 24, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=n, chi=64)
    psi = _exact_state(stream, n)

    g3 = _rand_unitary(3, rng)
    opt.apply_subtree_operator(g3, where)
    psi = _sv_apply_kq(psi, g3, where, n)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-9
    # unitary gate preserves the norm and leaves a valid canonical form
    assert abs(opt.norm() - np.linalg.norm(psi)) < 1e-9
    assert opt.is_canonical_form()


def test_apply_gate_routes_three_qubit_gate_to_subtree():
    """apply_gate on three qubits routes to apply_subtree_operator (no error)."""
    rng = np.random.default_rng(21)
    n = 6
    stream = _random_stream(n, 18, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, chi=64)
    psi = _exact_state(stream, n)

    g3 = _rand_unitary(3, rng)
    opt.apply_gate(g3, (0, 2, 4))
    psi = _sv_apply_kq(psi, g3, (0, 2, 4), n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-9


def test_subtree_operator_uses_recursive_pairwise_messages(monkeypatch):
    """The tree-MPO path never contracts the whole state subtree at once."""
    rng = np.random.default_rng(211)
    opt = TreeOptimizer(None, n=7, chi=64)
    calls = []
    tensor_contract = qtn.tensor_contract

    def traced_contract(*tensors, **kwargs):
        calls.append(len(tensors))
        return tensor_contract(*tensors, **kwargs)

    monkeypatch.setattr(qtn, "tensor_contract", traced_contract)
    opt.apply_subtree_operator(_rand_unitary(3, rng), (0, 3, 5))

    assert calls
    assert max(calls) <= 2
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_apply_gate_four_qubit_trotter_block_matches_dense():
    """A four-qubit block applied in one shot matches the dense reference."""
    rng = np.random.default_rng(22)
    n = 8
    stream = _random_stream(n, 30, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=n, chi=128)
    psi = _exact_state(stream, n)

    block = _rand_unitary(4, rng)
    where = (1, 3, 5, 7)
    opt.apply_gate(block, where)
    psi = _sv_apply_kq(psi, block, where, n)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8
    assert opt.is_canonical_form()


def test_apply_gate_rejects_repeated_qubit_multi():
    """A multi-qubit gate with a repeated qubit is rejected loudly."""
    rng = np.random.default_rng(23)
    opt = TreeOptimizer(None, n=6, chi=8)
    with pytest.raises(ValueError, match="distinct qubits"):
        opt.apply_gate(_rand_unitary(3, rng), (1, 3, 1))


def test_apply_subtree_operator_nonunitary_renormalizes():
    """A non-unitary (Kraus) operator with renormalize keeps a unit-norm state."""
    rng = np.random.default_rng(24)
    n = 7
    stream = _random_stream(n, 24, rng, two_qubit_frac=0.7)
    opt = TreeOptimizer(stream, n=n, chi=64)
    psi = _exact_state(stream, n)

    kraus = 0.3 * (
        rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    )
    where = (2, 5, 6)
    opt.apply_subtree_operator(kraus, where, renormalize=True)
    psi = _sv_apply_kq(psi, kraus, where, n)
    psi = psi / np.linalg.norm(psi)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-9
    assert abs(opt.norm() - 1.0) < 1e-9
    assert opt.is_canonical_form()


def test_apply_subtree_operator_single_qubit_nonunitary():
    """A single-qubit non-unitary operator centres on the leaf and applies exactly."""
    rng = np.random.default_rng(25)
    n = 5
    stream = _random_stream(n, 16, rng)
    opt = TreeOptimizer(stream, n=n, chi=32)
    psi = _exact_state(stream, n)

    op = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    opt.apply_subtree_operator(op, 3)
    psi = _sv_apply_kq(psi, op, (3,), n)

    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-10
    assert opt.center == opt.plan.leaf_of_qubit[3]
    assert opt.is_canonical_form()


def test_apply_subtree_operator_respects_max_bond():
    """The re-split truncation honours an explicit max_bond override."""
    rng = np.random.default_rng(26)
    n = 8
    stream = _random_stream(n, 36, rng, two_qubit_frac=0.8)
    opt = TreeOptimizer(stream, n=n, chi=64)

    opt.apply_subtree_operator(_rand_unitary(4, rng), (0, 2, 5, 7), max_bond=6)
    assert opt.max_bond() <= 6
    assert opt.is_canonical_form()


def test_tid_cache_self_heals_after_leaf_replacement():
    """The node->tid cache stays valid after gates replace leaf tensors."""
    rng = np.random.default_rng(16)
    n = 6
    opt = TreeOptimizer(None, n=n, chi=16)
    # warm the cache
    for nid in opt.plan.nodes():
        assert opt._tid(nid) in opt.tn.tensor_map
    # single-qubit gates rebuild leaf tensors (new tids)
    for q in range(n):
        opt.apply_1q(_rand_unitary(1, rng), q)
    # cache still resolves every node to a live tensor id
    for nid in opt.plan.nodes():
        assert opt._tid(nid) in opt.tn.tensor_map


def test_copy_is_independent():
    """copy() yields an optimizer that evolves without touching the original."""
    rng = np.random.default_rng(17)
    n = 5
    stream = _random_stream(n, 20, rng)
    base = TreeOptimizer(stream, n=n, chi=16)
    before = base.to_dense()

    clone = base.copy()
    assert clone.plan is base.plan
    assert clone.chi == base.chi and clone.threads == base.threads
    assert _fidelity(before, clone.to_dense()) > 1 - 1e-12

    clone.apply_gate(_rand_unitary(2, rng), (0, 1))
    # the original is unchanged; the clone has diverged
    assert np.allclose(base.to_dense(), before)
    assert not np.allclose(base.to_dense(), clone.to_dense())


def test_copy_preserves_layout_and_history_configuration():
    """copy() keeps the parameters that determine future layout/replay."""
    base = TreeOptimizer(
        None, n=5, max_arity=3, community_frac=0.11, star_frac=0.22,
        record_history=False, run=False,
    )
    clone = base.copy()
    assert clone.max_arity == 3
    assert clone.community_frac == pytest.approx(0.11)
    assert clone.star_frac == pytest.approx(0.22)
    assert clone.record_history is False


def test_record_history_can_be_disabled():
    """Large replays can omit retained per-edge/update history."""
    opt = TreeOptimizer(
        [(_rand_unitary(2, np.random.default_rng(142)), (0, 3))],
        n=4, chi=1, record_history=False,
    )
    report = opt.truncation_report()
    assert report["n_events"] == 0
    assert report["updates"] == []


def test_copy_rng_is_deterministic_but_independent():
    """Copies derive reproducible but distinct random streams."""
    first = TreeOptimizer(None, n=2, seed=91, run=False)
    second = TreeOptimizer(None, n=2, seed=91, run=False)
    first_draws = [first.copy().rng.random() for _ in range(3)]
    second_draws = [second.copy().rng.random() for _ in range(3)]

    assert np.allclose(first_draws, second_draws)
    assert not np.isclose(first_draws[0], first_draws[1])


def test_thread_index_clears_after_failed_two_qubit_update(monkeypatch):
    """A failed threaded gate cannot leave its temporary bond index live."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, n=4, tree=plan, run=False)

    def fail_thread(*_args):
        raise RuntimeError("synthetic thread failure")

    monkeypatch.setattr(opt, "_thread_hop", fail_thread)
    with pytest.raises(RuntimeError, match="synthetic thread failure"):
        opt.apply_2q(np.eye(4, dtype=complex), 0, 3)
    assert opt._thread_ind is None


def test_tree_run_progbar_reports_infidelity(monkeypatch):
    """Tree replay exposes MPS-style progress fields without SVD probes."""
    progress_instances = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.desc = kwargs["desc"]
            self.n = 0
            self.postfix_calls = []
            progress_instances.append(self)

        def set_postfix(self, postfix):
            self.postfix_calls.append(dict(postfix))

        def update(self, amount):
            self.n += amount

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm)
    )

    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, n=4, tree=plan, chi=1, run=False)
    opt.set_gates([(h, 0), (cnot, (0, 3))])
    opt.run(progbar=True)

    progress = progress_instances[-1]
    assert progress.total == 2
    assert progress.n == 2
    assert progress.desc == "direct"
    last = progress.postfix_calls[-1]
    assert {"2q", "~F", "bnd"} <= set(last)
    assert "infidelity" not in last
    # kq is only shown when multi-qubit (>2) gates are present.
    assert "kq" not in last
    assert last["2q"] == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("direct", "direct"),
        ("dm", "dm"),
        ("sdc", "sdc"),
        ("src", "src"),
        ("mpo", "direct"),
        ("tree_mpo_dm", "dm"),
        ("dmrg", "dmrg"),
        ("dmrg1", "dmrg1"),
        ("dmrg2", "dmrg2"),
        ("dmrg3", "dmrg3"),
    ],
)
def test_tree_progress_bar_uses_mps_mode_names(monkeypatch, mode, expected):
    """Tree replay bars expose the same active mode names as MPS bars."""

    descriptors = []

    class _FakeTqdm:
        def __init__(self, **kwargs):
            descriptors.append(kwargs["desc"])

        def set_postfix(self, _postfix):
            pass

        def update(self, _count):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "tqdm", types.SimpleNamespace(tqdm=_FakeTqdm)
    )
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    optimizer = TreeOptimizer(None, n=2, mode=mode, run=False)
    optimizer.set_gates([(h, 0)])
    # The mode label is selected when the bar is created; avoid making this
    # label-only regression test depend on each compression backend.
    monkeypatch.setattr(optimizer, "apply_gate", lambda *_args, **_kwargs: optimizer)

    optimizer.run(progbar=True)

    assert descriptors == [expected]


def test_threads_setting_preserves_result():
    """The thread cap is a performance knob only; results are identical."""
    rng = np.random.default_rng(18)
    n = 7
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    a = TreeOptimizer(stream, n=n, chi=8, threads=1).to_dense()
    b = TreeOptimizer(stream, n=n, chi=8, threads=None).to_dense()
    assert np.allclose(a, b)


# -- sibling fast path / measurement / multi-site expectation -----------------


def test_sibling_fast_path_matches_statevector():
    """Two-qubit gates on sibling leaves reproduce the exact statevector.

    A balanced plan over ``range(4)`` makes qubits ``(0, 1)`` and ``(2, 3)``
    siblings, so every two-qubit gate here takes the parent-blob fast path.
    """
    rng = np.random.default_rng(20)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    stream = [
        (_rand_unitary(2, rng), (0, 1) if rng.random() < 0.5 else (2, 3))
        for _ in range(30)
    ]
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_mixed_paths_match_statevector():
    """A stream mixing sibling and non-sibling two-qubit gates stays exact."""
    rng = np.random.default_rng(21)
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    stream = _random_stream(n, 40, rng, two_qubit_frac=0.6)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=64)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_measure_born_statistics_and_collapse():
    """Measurement samples the Born rule and collapses to a unit-norm state."""
    theta = 0.7
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=complex)
    base = TreeOptimizer([(ry, 0)], n=3, chi=4, seed=0)

    n_shots = 3000
    ones = sum(base.copy().measure(0) for _ in range(n_shots))
    assert abs(ones / n_shots - s**2) < 0.03  # p(1) = sin^2(theta/2)

    forced = base.copy()
    assert forced.measure(0, outcome=0) == 0
    assert abs(forced.norm() - 1.0) < 1e-9


def test_reset_forces_ground_state():
    """reset() returns a qubit to |0> regardless of its prior value."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer([(x, 1)], n=3, chi=4, seed=1)  # qubit 1 in |1>
    opt.reset(1)
    assert _fidelity(opt.to_dense(), np.array([1.0] + [0.0] * 7)) > 1 - 1e-9
    assert abs(opt.norm() - 1.0) < 1e-9


def test_measure_is_seed_reproducible():
    """Two optimizers with the same seed measure the same outcome."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    a = TreeOptimizer([(h, 0)], n=2, seed=42).measure(0)
    b = TreeOptimizer([(h, 0)], n=2, seed=42).measure(0)
    assert a == b


def test_tree_stream_measure_and_reset_match_mps_event_contract():
    """TTN streams accept Pauli measurement/reset events and record outcomes."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(
        [(x, 0), ("measure", "Z", 0, -1), ("reset", 0)],
        n=2,
        chi=4,
        seed=12,
    )

    assert len(opt.measurements) == 1
    pauli, where, outcome, probability = opt.measurements[0]
    assert (pauli, where, outcome) == ("Z", (0,), -1)
    assert probability == pytest.approx(1.0)
    assert _fidelity(opt.to_dense(), np.array([1.0, 0.0, 0.0, 0.0])) > 1 - 1e-12
    assert opt.event_types == ["gate", "measure", "reset"]


def test_tree_stream_measure_reset_records_then_prepares_pauli_state():
    """measure_reset records the result and leaves the + Pauli eigenstate."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    opt = TreeOptimizer(
        [(h, 0), TreeOptimizer.measure_reset_event("X", 0, +1)],
        n=1,
        chi=4,
    )

    assert opt.measurements == [("X", (0,), +1, pytest.approx(1.0))]
    assert _fidelity(
        opt.to_dense(), np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    ) > 1 - 1e-12
    assert opt.norm() == pytest.approx(1.0)


def test_tree_stream_multisite_pauli_measurement():
    """A product-Pauli event can collapse a multi-qubit tree subtree."""
    opt = TreeOptimizer([("measure", "ZZ", (0, 1), +1)], n=2, chi=4)

    assert opt.measurements[0][:3] == ("ZZ", (0, 1), +1)
    assert opt.measurements[0][3] == pytest.approx(1.0)
    assert opt.norm() == pytest.approx(1.0)


def test_multisite_pauli_measurement_preserves_parity_sector_coherence():
    """Parity projection must not collapse the individual Pauli outcomes."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=2, chi=8)

    opt._measure_pauli("ZZ", (0, 1), +1)
    assert _fidelity(
        opt.to_dense(), np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    ) > 1 - 1e-12


def test_wide_pauli_measurement_avoids_dense_projector():
    """A wide product-Pauli event remains factorized past the dense limit."""
    n = 9
    opt = TreeOptimizer(
        [("measure", "Z" * n, tuple(range(n)), +1)], n=n, chi=4
    )
    assert opt.measurements[0][2] == +1
    assert opt.norm() == pytest.approx(1.0)


def test_default_dense_operator_guard():
    """General dense operators have a finite default support limit."""
    opt = TreeOptimizer(None, n=9, run=False)
    assert opt.max_operator_qubits == 8
    with pytest.raises(MemoryError, match="max_operator_qubits"):
        opt.apply_gate(np.eye(2**9, dtype=complex), tuple(range(9)))


def test_tree_stream_control_mapping_and_cap_event():
    """Mapping controls and MPS-compatible cap events work on a tree."""
    opt = TreeOptimizer(
        [{"kind": "measure", "pauli": "Z", "where": 0, "outcome": +1}],
        n=1,
    )
    assert opt.measurements[0][:3] == ("Z", (0,), +1)

    capped = TreeOptimizer(
        [TreeOptimizer.cap_event(0, [1.0, 0.0])], n=2, chi=4
    )
    assert capped.n == capped.plan.n == capped.tn.nqubits == 1
    assert np.allclose(capped.to_dense(), [1.0, 0.0])

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    shifted = TreeOptimizer(
        [TreeOptimizer.cap_event(1, [1.0, 0.0]), (x, 1)],
        n=3,
        chi=4,
    )
    assert shifted.n == 2
    assert np.argmax(np.abs(shifted.to_dense())) == 1


def test_tree_cap_matches_dense_contraction_and_compacts_plan():
    """Capping an entangled leaf matches dense contraction and keeps a valid TTN."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(h, 0), (cnot, (0, 1))],
        n=4,
        tree=TreePlan.from_order(range(4), structure="balanced"),
        chi=8,
    )
    before = opt.to_dense().reshape((2,) * 4)
    vec = np.array([1.0, 2.0], dtype=complex)
    expected = np.tensordot(vec, before, axes=(0, 1)).reshape(-1)

    opt.cap(1, vec)

    assert opt.n == opt.plan.n == opt.tn.nqubits == 3
    assert np.allclose(opt.to_dense(), expected)
    assert opt.tn.validate(check_canonical=True) is opt.tn
    assert set(opt.plan.leaf_of_qubit) == {0, 1, 2}


def test_tree_public_submpo_and_pauli_backend_operations():
    """Public tree operator primitives cover native MPO and Pauli paths."""
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    mpo = _two_branch_flip_submpo(L=n, sites=(0, 3), targets=(0, 3))
    opt = TreeOptimizer(None, n=n, tree=plan, chi=16, run=False)

    opt.apply_submpo(mpo, (0, 3))
    assert np.allclose(
        opt.to_dense(),
        0.7 * np.eye(16, dtype=complex)[:, 0]
        + 0.3 * _sv_apply_kq(
            np.eye(16, dtype=complex)[:, 0],
            np.kron(np.array([[0, 1], [1, 0]], dtype=complex),
                    np.array([[0, 1], [1, 0]], dtype=complex)),
            (0, 3),
            n,
        ),
    )

    rotated = TreeOptimizer(None, n=n, tree=plan, chi=16, run=False)
    theta = 0.37
    rotated.apply_pauli_rotation(theta, "XZ", (0, 3))
    pauli = np.kron(
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    )
    expected = (
        np.cos(theta / 2.0) * np.eye(4, dtype=complex)
        - 1j * np.sin(theta / 2.0) * pauli
    )
    assert np.allclose(
        rotated.to_dense(),
        _sv_apply_kq(np.eye(16, dtype=complex)[:, 0], expected, (0, 3), n),
    )

    summed = TreeOptimizer(None, n=n, tree=plan, chi=16, run=False)
    summed.apply_pauli_sum([(1.0, {0: "X", 3: "X"})])
    assert np.allclose(
        summed.to_dense(),
        _sv_apply_kq(
            np.eye(16, dtype=complex)[:, 0],
            np.kron(
                np.array([[0, 1], [1, 0]], dtype=complex),
                np.array([[0, 1], [1, 0]], dtype=complex),
            ),
            (0, 3),
            n,
        ),
    )


def test_tree_pauli_sum_routes_only_active_support_over_steiner_subtree():
    """Sparse Pauli TreeMPOs must not turn support into a chain window."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    active = (0, 2, 7)
    opt = TreeOptimizer(
        None, n=8, tree=plan, chi=8, cutoff=0.0, profile=True, run=False
    )

    opt.apply_pauli_sum([
        (0.7, {0: "X", 2: "Y", 7: "Z"}),
        (0.2, {0: "Z", 7: "X"}),
    ])

    assert opt.update_history[-1]["support"] == active
    routes = [
        event for event in opt.profile_events
        if event.get("route") == "subtreempo"
        and event.get("kind") == "metadata_path"
    ]
    assert routes
    assert routes[-1]["support"] == active
    assert routes[-1]["subtree_nodes"] == len(
        opt._steiner_nodes([opt.plan.node_of_qubit[q] for q in active])
    )
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_two_site_numpy_mpo_is_coerced_to_cupy_state_backend():
    """Two-site MPO factors follow a CuPy TTN without mutating the MPO."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(
        lambda array: cupy.asarray(array, dtype=cupy.complex64)
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        np.kron(x, x), dims=(2, 2), sites=(0, 1), L=n,
    )
    before = [tensor.data.copy() for tensor in mpo.tensors]
    opt = TreeOptimizer(
        None, state=state, tree=plan, chi=8, cutoff=0.0, run=False,
    )

    with pytest.warns(UserWarning, match="converting a gate/operator payload"):
        opt.apply_submpo(mpo, (0, 1))

    assert opt.backend_info() == {
        "backend": "cupy",
        "dtype": "complex64",
        "device": str(cupy.cuda.Device()),
    }
    expected = np.zeros(2**n, dtype=np.complex64)
    expected[12] = 1.0  # X_0 X_1 |0000> = |1100>
    np.testing.assert_allclose(opt.to_dense(), expected, atol=1e-5)
    assert opt.tn.validate() is opt.tn
    for tensor, original in zip(mpo.tensors, before):
        np.testing.assert_array_equal(tensor.data, original)


def test_tree_two_site_cupy_gate_does_not_unwrap_memory_pointer():
    """A CuPy dense gate reaches TreeMPO factorization as an array."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a CUDA device.")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime unavailable: {exc}")

    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(
        lambda array: cupy.asarray(array, dtype=cupy.complex128)
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    gate = cupy.asarray(np.kron(x, x), dtype=cupy.complex128)
    opt = TreeOptimizer(
        [(gate, (0, 1))],
        n=n,
        tree=plan,
        state=state,
        chi=8,
        cutoff=0.0,
        run=False,
    )

    opt.run(progbar=False)

    expected = np.zeros(2**n, dtype=np.complex128)
    expected[12] = 1.0
    np.testing.assert_allclose(opt.to_dense(), expected, atol=1e-12)


def test_tree_expectation_mpo_is_batched_and_non_mutating():
    """A structured MPO expectation uses one tree pass and preserves state."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    zz = np.diag([1.0, -1.0, -1.0, 1.0])
    mpo = qtn.MatrixProductOperator.from_dense(
        zz, dims=(2, 2), sites=(0, 1), L=4,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=4, chi=16)
    before = opt.to_dense().copy()

    value = opt.expectation_mpo(mpo, (0, 1), max_bond=16)

    assert value == pytest.approx(1.0)
    assert np.allclose(opt.to_dense(), before)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_expectation_mpo_reports_private_ket_truncation():
    """Expectation diagnostics expose accidental finite-cap truncation."""
    identity = np.eye(4, dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        identity, dims=(2, 2), sites=(0, 1), L=4,
    )
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=4, chi=16)

    with pytest.warns(UserWarning, match="private transformed ket"):
        value, diagnostics = opt.expectation_mpo(
            mpo, (0, 1), max_bond=1, return_diagnostics=True,
        )

    assert diagnostics["truncated"] is True
    assert diagnostics["n_truncated"] >= 1
    assert diagnostics["max_bond"] == 1
    assert value != pytest.approx(1.0)

    exact_value, exact_diagnostics = opt.expectation_mpo(
        mpo, (0, 1), max_bond=16, return_diagnostics=True,
    )
    assert exact_value == pytest.approx(1.0)
    assert exact_diagnostics["truncated"] is False


def test_tree_exact_mpo_expectation_keeps_mpo_separate(monkeypatch):
    """Exact MPO readout does not lower or compress the tree state."""
    identity = np.eye(4, dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        identity, dims=(2, 2), sites=(0, 1), L=4,
    )
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=4, chi=1)
    before = opt.to_dense().copy()
    before_bond = opt.tn.max_bond()

    def forbid_dense(*args, **kwargs):
        raise AssertionError("exact MPO readout must not call MPO.to_dense()")

    monkeypatch.setattr(qtn.MatrixProductOperator, "to_dense", forbid_dense)
    value = opt.expectation_mpo_exact(mpo, (0, 1))

    assert value == pytest.approx(1.0)
    assert opt.tn.max_bond() == before_bond
    assert np.allclose(opt.to_dense(), before)


def test_tree_rejects_native_mpo_on_dense_state():
    """Native and ordinary tensor backends cannot be mixed silently."""
    class NativeMarker:
        pepsy_tree_native = True

    opt = TreeOptimizer(None, n=2, run=False)
    with pytest.raises(TypeError, match="native Symmray MPO"):
        opt.apply_submpo(NativeMarker(), (0, 1))


def test_tree_subtree_route_batches_sibling_messages(monkeypatch):
    """Independent leaf messages landing at one node use one contraction."""
    n = 4
    plan = TreePlan.from_order(range(n), structure="balanced")
    opt = TreeOptimizer(
        None, n=n, tree=plan, chi=16, cutoff=0.0,
        subtree_workers=2, run=False,
    )
    identity = np.eye(2**n, dtype=complex)
    before = opt.to_dense().copy()
    original_contract = qtn.tensor_contract
    grouped_calls = []

    def traced_contract(*tensors, **kwargs):
        if len(tensors) >= 3:
            grouped_calls.append(len(tensors))
        return original_contract(*tensors, **kwargs)

    monkeypatch.setattr(qtn, "tensor_contract", traced_contract)
    opt.apply_subtree_operator(identity, tuple(range(n)))

    assert grouped_calls
    assert np.allclose(opt.to_dense(), before)
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_profile_report_is_opt_in():
    """Kernel timings are empty by default and available when requested."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    quiet = TreeOptimizer([(x, 0)], n=4, chi=4)
    quiet_report = quiet.profile_report()
    assert quiet_report["enabled"] is False
    assert quiet_report["events"] == []
    assert quiet_report["by_kind"] == {}
    assert quiet_report["native_compression_routes"] == {}
    assert quiet_report["update_seconds"] == 0.0
    assert quiet_report["total_seconds"] == 0.0
    assert quiet_report["timing_semantics"][
        "total_seconds_is_sum_of_events_not_wall_time"
    ] is True

    profiled = TreeOptimizer(
        [(x, 0), (cnot, (0, 3))], n=4, chi=4, profile=True,
    )
    report = profiled.profile_report()
    assert report["enabled"] is True
    assert report["events"]
    assert report["by_kind"]["update"]["count"] == 2
    assert report["native_compression_routes"] == {}
    assert report["by_kind"]["gate_factorization"]["count"] == 1
    assert report["by_kind"]["tensor_absorption"]["count"] >= 2
    assert report["by_kind"]["metadata_path"]["count"] == 1
    assert report["by_kind"]["center_movement"]["count"] >= 1
    assert report["update_seconds"] == report["by_kind"]["update"]["seconds"]
    assert report["total_seconds"] > 0.0


def test_tree_profile_separates_qr_thread_hops_from_compression():
    """Profiled direct routing exposes exact QR hops as separate events."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=2, cutoff=0.0,
        profile=True, track_bond_diagnostics=True,
    )

    report = opt.profile_report()
    hops = [event for event in report["events"] if event["kind"] == "thread_hop"]
    assert hops
    assert all(event["seconds"] >= 0.0 for event in hops)
    assert report["by_kind"]["thread_hop"]["count"] == len(hops)
    assert report["by_kind"]["edge_canonize"]["count"] >= 1


def test_tree_bond_diagnostics_distinguish_transient_qr_growth():
    """Temporary gate/QR growth may exceed chi while live bonds do not."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(
        [(cnot, (0, 3))], n=4, chi=1, cutoff=0.0,
        record_history=False, track_bond_diagnostics=True,
    )

    report = opt.bond_diagnostic_report()
    assert report["enabled"] is True
    assert report["max_transient_bond"] >= 2
    assert report["max_live_bond_after"] <= 1
    assert report["n_transient_exceeds_chi"] >= 1
    update = report["updates"][0]
    assert update["transient_max_bond"] > update["live_max_bond_after"]
    assert update["transient_exceeds_chi"] is True
    assert update["bond_trace"]


def test_tree_norm_and_fidelity_check_is_deterministic_without_network_fidelity():
    """Small exact replay uses local norm plus a deterministic statevector oracle."""
    rng = np.random.default_rng(417)
    stream = _random_stream(4, 10, rng, two_qubit_frac=0.8)
    exact = _exact_state(stream, 4)

    first = TreeOptimizer(
        stream, n=4, chi=64, cutoff=0.0, threads=1,
    )
    second = TreeOptimizer(
        stream, n=4, chi=64, cutoff=0.0, threads=2,
    )
    first_dense = first.to_dense()
    second_dense = second.to_dense()

    assert first.norm() == pytest.approx(np.linalg.norm(first_dense))
    assert second.norm() == pytest.approx(np.linalg.norm(second_dense))
    assert _fidelity(exact, first_dense) > 1.0 - 1e-10
    assert _fidelity(exact, second_dense) > 1.0 - 1e-10
    assert _fidelity(first_dense, second_dense) > 1.0 - 1e-12


def test_tree_pauli_expectation_and_projection_are_public():
    """Pauli expectation/projection share the measurement backend semantics."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], n=2, chi=8)
    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)
    assert opt.expectation_pauli("XX", (0, 1)) == pytest.approx(1.0)
    opt.project_pauli("ZZ", (0, 1), +1)
    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)


def test_tree_sync_canonicalization_rebuilds_state_owned_center():
    """Tree recovery clears stale lower-level canonical-region metadata."""
    opt = TreeOptimizer([], n=4, chi=8, run=False)
    opt.tn.canonize_around_qubits_((0, 3))

    center = opt.sync_canonicalization()

    assert center == opt.plan.root
    assert opt.center == opt.tn.orthogonality_center == center
    assert opt.is_canonical_form(center)


def test_tree_public_pauli_measurement_returns_probability_and_diagnostics():
    """The public Pauli measurement API exposes Born probability diagnostics."""
    theta = 0.8
    ry = np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0), np.cos(theta / 2.0)],
    ], dtype=complex)
    opt = TreeOptimizer([(ry, 0)], n=5, chi=8)

    outcome, probability, diagnostics = opt.measure_pauli(
        "Z", 0, outcome=+1, return_diagnostics=True
    )

    assert outcome == +1
    assert probability == pytest.approx(np.cos(theta / 2.0) ** 2)
    assert diagnostics["probability"] == pytest.approx(probability)
    assert diagnostics["norm_before"] == pytest.approx(1.0)
    assert diagnostics["norm_after"] == pytest.approx(1.0)
    assert diagnostics["support"] == (0,)
    assert diagnostics["span_before"] == diagnostics["span_after"]
    assert diagnostics["bonds_before"] == diagnostics["bonds_after"]
    assert opt.expectation_pauli("Z", 0) == pytest.approx(1.0)


def test_tree_pauli_projection_can_preserve_branch_norm():
    """A non-normalizing Pauli projection retains its physical survival norm."""
    theta = 0.9
    ry = np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0), np.cos(theta / 2.0)],
    ], dtype=complex)
    opt = TreeOptimizer([(ry, 0)], n=4, chi=8)

    diagnostics = opt.project_pauli(
        "Z", 0, +1, renormalize=False, return_diagnostics=True
    )
    expected_probability = np.cos(theta / 2.0) ** 2

    assert diagnostics["renormalized"] is False
    assert diagnostics["norm_after"] == pytest.approx(
        np.sqrt(expected_probability)
    )
    assert diagnostics["norm_ratio"] == pytest.approx(
        np.sqrt(expected_probability)
    )
    assert opt.expectation_pauli("Z", 0) == pytest.approx(1.0)
    assert opt.get_projection_diagnostics()[-1] is diagnostics


def test_tree_sparse_long_pauli_avoids_dense_operator_limit():
    """A long sparse Pauli measurement uses the factorized tree path."""
    n = 17
    where = (0, 3, 7, 11, 16)
    opt = TreeOptimizer(None, n=n, chi=8, max_operator_qubits=2, run=False)

    outcome, probability, diagnostics = opt.measure_pauli(
        "ZZZZZ", where, outcome=+1, return_diagnostics=True
    )

    assert outcome == +1
    assert probability == pytest.approx(1.0)
    assert diagnostics["support"] == where
    assert diagnostics["max_bond_after"] <= 8
    assert opt.norm() == pytest.approx(1.0)


def test_tree_cap_can_preserve_stable_logical_labels():
    """Stable-label caps compact storage but preserve caller-facing IDs."""
    opt = TreeOptimizer(None, n=4, chi=8, run=False)

    opt.cap(1, [1.0, 0.0], stable_labels=True)

    assert opt.n == 3
    assert opt.qubits == [0, 2, 3]
    assert opt.logical_order == [0, 2, 3]
    assert opt.position(2) == 1
    assert opt.logical_site(1) == 2

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt.apply_1q(x, 3)
    assert np.argmax(np.abs(opt.to_dense())) == 1


def test_tree_stable_labels_work_for_pauli_projection_paths():
    """Pauli projection resolves stable labels once, then uses compact sites."""
    opt = TreeOptimizer(None, n=4, chi=8, run=False)
    opt.cap(1, [1.0, 0.0], stable_labels=True)

    diagnostics = opt.project_pauli(
        "Z", 2, +1, renormalize=False, return_diagnostics=True
    )

    assert diagnostics["support"] == (2,)
    assert diagnostics["norm_after"] == pytest.approx(1.0)
    assert opt.expectation_pauli("Z", 2) == pytest.approx(1.0)


def test_tree_stable_cap_event_supports_later_logical_events():
    """A stable-label cap event keeps later stream labels addressable."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(
        [
            TreeOptimizer.cap_event(
                1, [1.0, 0.0], compact_labels=False
            ),
            (x, 3),
        ],
        n=4,
        chi=8,
    )

    assert opt.qubits == [0, 2, 3]
    assert np.argmax(np.abs(opt.to_dense())) == 1


def test_tree_reset_reuses_an_ancilla_without_disturbing_data():
    """Reset and repeated use of an ancilla leave the data Bell pair intact."""
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer([(h, 0), (cnot, (0, 1)), (x, 2)], n=3, chi=8)

    opt.reset(2)
    assert opt.expectation_pauli("Z", 2) == pytest.approx(1.0)
    opt.apply_1q(x, 2)
    opt.reset(2)

    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)
    assert opt.expectation_pauli("XX", (0, 1)) == pytest.approx(1.0)
    assert opt.expectation_pauli("Z", 2) == pytest.approx(1.0)


def test_tree_stream_submpo_markers_use_recursive_operator_path():
    """Tuple and mapping sub-MPO markers match their dense support operator."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    dense = np.asarray(mpo.to_dense())

    event = TreeOptimizer.submpo_event(mpo, (0, 3))
    tuple_opt = TreeOptimizer([event], n=4, chi=8)
    mapping_opt = TreeOptimizer(
        [{"kind": "submpo", "mpo": mpo, "where": [0, 3]}],
        n=4,
        chi=8,
    )
    expected = _sv_apply_kq(np.eye(16, dtype=complex)[:, 0], dense, (0, 3), 4)

    assert tuple_opt.event_types == ["submpo"]
    assert TreeOptimizer.is_submpo_event(event)
    assert TreeOptimizer.submpo_event_parts(event)[1] == (0, 3)
    assert tuple_opt.update_history[0]["kind"] == "submpo"
    assert _fidelity(expected, tuple_opt.to_dense()) > 1 - 1e-10
    assert _fidelity(tuple_opt.to_dense(), mapping_opt.to_dense()) > 1 - 1e-10


def test_tree_stream_submpo_does_not_require_dense_materialization(monkeypatch):
    """Native MPO replay and bond estimation avoid ``to_dense`` allocation."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    expected = np.asarray(mpo.to_dense())

    def fail_to_dense():
        raise AssertionError("sub-MPO was unexpectedly materialized")

    monkeypatch.setattr(mpo, "to_dense", fail_to_dense)
    event = TreeOptimizer.submpo_event(mpo, (0, 3))
    opt = TreeOptimizer([event], n=4, chi=8)
    report = opt.estimate_bonds([event])

    reference = _sv_apply_kq(
        np.eye(16, dtype=complex)[:, 0], expected, (0, 3), 4
    )
    assert _fidelity(reference, opt.to_dense()) > 1 - 1e-10
    assert report["events"][0]["crossing_edges"]


def test_tree_native_submpo_keeps_unacted_physical_root_structured(monkeypatch):
    """A physical root on the Steiner subtree need not be an MPO site."""
    where = (1, 2, 3)
    plan = TreePlan.from_order(
        range(1, 5), structure="balanced", root_qubit=0
    )
    gate = _rand_unitary(len(where), np.random.default_rng(53))
    mpo = qtn.MatrixProductOperator.from_dense(
        gate.reshape((2,) * (2 * len(where))),
        dims=(2,) * len(where),
        sites=where,
        L=5,
        max_bond=None,
        cutoff=0.0,
    )
    expected = _sv_apply_kq(
        np.eye(2**5, dtype=complex)[:, 0], gate, where, 5
    )

    def fail_to_dense():
        raise AssertionError("sub-MPO was unexpectedly materialized")

    monkeypatch.setattr(mpo, "to_dense", fail_to_dense)
    opt = TreeOptimizer(
        None, n=5, tree=plan, chi=64, cutoff=0.0, run=False
    )
    opt.apply_submpo(mpo, where)

    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10


def test_tree_submpo_mode_declares_and_validates_mpo_streams():
    """The explicit sub-MPO mode accepts MPO events and rejects dense gates."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    opt = TreeOptimizer(None, n=4, chi=8, mode="submpo", run=False)
    opt.run([TreeOptimizer.submpo_event(mpo, (0, 3))])
    assert opt.mode == "submpo"

    dense = np.asarray(mpo.to_dense())
    ordinary = TreeOptimizer(None, n=4, chi=8, mode="submpo", run=False)
    with pytest.raises(ValueError, match="requires explicit sub-MPO"):
        ordinary.run([(dense, (0, 3))])


def test_tree_estimate_bonds_includes_submpo_operator_schmidt_rank():
    """Sub-MPO markers participate in the same conservative bond estimate."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    plan = TreePlan.from_order(range(4), structure="balanced")
    opt = TreeOptimizer(None, n=4, tree=plan, chi=1, run=False)

    report = opt.estimate_bonds([("submpo", mpo, (0, 3))])

    assert report["events"][0]["kind"] == "submpo"
    assert set(report["events"][0]["crossing_edges"].values()) == {2}
    assert report["max_bond"] == 2
    assert report["requires_truncation"]
    with pytest.raises(MemoryError, match="max_operator_qubits"):
        opt.preflight(
            [("submpo", mpo, (0, 3))],
            max_operator_qubits=1,
        )


def test_measurement_enforces_max_subtree_nodes_on_streamed_product_pauli():
    """Control-event execution applies the same subtree guard as dense gates."""
    opt = TreeOptimizer(None, n=8, max_subtree_nodes=1, run=False)
    with pytest.raises(MemoryError, match="max_subtree_nodes"):
        opt.run([{
            "kind": "measure",
            "pauli": "ZZ",
            "where": [0, 7],
            "outcome": +1,
        }])


# -- TreeTensorNetwork class --------------------------------------------------


def test_ttn_from_plan_is_product_state():
    """from_plan builds |0...0> with the expected tags, indices, and sites."""
    plan = TreePlan.from_order(range(5), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    assert isinstance(ttn, TreeTensorNetwork)
    assert ttn.nqubits == 5 == ttn.nsites
    assert tuple(ttn.sites) == (0, 1, 2, 3, 4)
    # site index / tag / node tag conventions
    assert ttn.site_ind(2) == "k2"
    assert ttn.site_tag(2) == "I2"
    assert ttn.node_tag(plan.root) == f"N{plan.root}"
    # dense state is exactly |0...0>
    sv = ttn.to_statevector()
    assert sv.shape == (2**5,)
    assert abs(sv[0] - 1.0) < 1e-12
    assert np.linalg.norm(sv[1:]) < 1e-12


def test_binary_tree_supports_a_three_virtual_leg_top_tensor():
    """A ternary virtual root keeps every tensor in the binary rank class."""
    plan = TreePlan.from_order(
        range(9), structure="balanced", max_arity=2, top_arity=3,
    )
    assert plan.top_arity == 3
    assert plan.is_binary()
    assert not plan.is_strictly_binary()
    assert all(
        len(children) in (0, 2)
        for node, children in plan.children.items()
        if node != plan.root
    )

    ttn = TreeTensorNetwork.from_plan(plan)
    assert len(ttn.node_tensor(plan.root).inds) == 3
    assert ttn.max_virtual_degree == 3
    assert ttn.max_tensor_rank == 3
    assert ttn.validate(check_canonical=True) is ttn

    ordered = TreeTensorNetwork.from_order(
        range(9), max_arity=2, top_arity=3,
    )
    assert ordered.top_arity == 3
    assert len(ordered.node_tensor(ordered.plan.root).inds) == 3

    finder = TreeLayoutFinder([], n=9, max_arity=2, top_arity=3)
    found = finder.run()
    report = finder.report(found)
    assert found.top_arity == 3
    assert found.is_binary()
    assert report["top_arity"] == 3
    assert report["max_tensor_rank"] == 3

    automatic = TreeOptimizer(
        [], n=9, max_arity=2, top_arity=3, run=False,
    )
    assert automatic.plan.top_arity == 3
    assert automatic.tn.max_tensor_rank == 3

    layout = TreeLayoutFinder([], n=9, max_arity=2, top_arity=3)
    from_layout = TreeOptimizer([], layout=layout, run=False)
    assert from_layout.top_arity == 3
    assert from_layout.plan.top_arity == 3

    product = pepsy.ps_to_ttn(9, max_arity=2, top_arity=3)
    assert product.top_arity == 3
    assert product.max_tensor_rank == 3

    random = pepsy.hrs_to_ttn(9, max_arity=2, top_arity=3, seed=11)
    assert random.top_arity == 3
    assert random.max_tensor_rank == 3


def test_binary_tree_with_ternary_root_is_the_shared_default():
    """All high-level tree builders share the rank-three root convention."""
    plan = TreePlan.from_order(range(9), structure="balanced")
    assert plan.top_arity == 3
    assert plan.is_binary()
    assert not plan.is_strictly_binary()

    ordered = TreeTensorNetwork.from_order(range(9))
    assert ordered.top_arity == 3
    assert len(ordered.node_tensor(ordered.plan.root).inds) == 3

    finder = TreeLayoutFinder([], n=9)
    found = finder.run()
    assert found.top_arity == 3
    assert found.is_binary()

    optimizer = TreeOptimizer([], n=9, run=False)
    assert optimizer.plan.top_arity == 3
    assert optimizer.tn.max_tensor_rank == 3

    product = pepsy.ps_to_ttn(9)
    random = pepsy.hrs_to_ttn(9, seed=11)
    assert product.top_arity == random.top_arity == 3
    assert product.max_tensor_rank == random.max_tensor_rank == 3

    # A physical root cannot also use three incoming virtual bonds, and small
    # systems naturally fall back to the ordinary binary root.
    rooted = TreePlan.from_order(range(8), root_qubit=8)
    assert rooted.top_arity == 2
    assert TreePlan.from_order(range(2)).top_arity == 2


def test_ps_to_ttn_matches_product_state_constructor_api():
    """The high-level TTN constructor mirrors ``ps_to_mps`` amplitudes."""
    theta = 0.31
    expected = np.array([1.0], dtype="complex128")
    local = np.array([np.cos(theta), np.sin(theta)], dtype="complex128")
    for _ in range(4):
        expected = np.kron(expected, local)

    state = pepsy.ps_to_ttn(4, theta=theta)
    assert isinstance(state, TreeTensorNetwork)
    assert state.max_bond() == 1
    assert np.allclose(state.to_statevector(), expected)
    assert state.is_canonical_form(state.root)

    expanded = pepsy.ps_to_ttn(4, chi=2, rand_strength=0.0)
    assert expanded.max_bond() == 2
    assert np.allclose(expanded.to_statevector(), [1.0] + [0.0] * 15)

    plan = TreePlan.from_order(range(4), structure="balanced")
    explicit = pepsy.ps_to_ttn(4, tree=plan)
    assert explicit.plan is plan


def test_product_ttn_constructors_support_a_physical_root_site():
    """Product/random public constructors resolve every physical node."""
    theta = 0.23
    local = np.array([np.cos(theta), np.sin(theta)], dtype="complex128")
    expected = local
    for _ in range(4):
        expected = np.kron(expected, local)

    plan = TreePlan.from_order(
        [0, 1, 3, 4], structure="balanced", root_qubit=2,
    )
    explicit = pepsy.ps_to_ttn(5, tree=plan, theta=theta)
    automatic = pepsy.ps_to_ttn(5, root_qubit=2, theta=theta)
    smallest = pepsy.ps_to_ttn(2, root_qubit=1, theta=theta)
    random = pepsy.hrs_to_ttn(5, root_qubit=2, seed=11)

    assert explicit.plan is plan
    assert automatic.plan.root_qubit == 2
    assert smallest.plan.n == 2
    assert smallest.plan.root_qubit == 1
    assert random.plan.root_qubit == 2
    assert np.allclose(explicit.to_statevector(), expected)
    assert np.allclose(automatic.to_statevector(), expected)
    assert random.to_statevector().shape == (2**5,)
    assert explicit.validate(check_canonical=True) is explicit

    with pytest.raises(ValueError, match="root_qubit does not match"):
        pepsy.ps_to_ttn(5, tree=plan, root_qubit=3)


def test_ttn_copy_preserves_geometry_and_type():
    """copy() keeps the plan, ids, and class, with an independent tid cache."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    other = ttn.copy()
    assert type(other) is TreeTensorNetwork
    assert other.plan is ttn.plan
    assert other.site_ind_id == ttn.site_ind_id
    assert other.node_tag_id == ttn.node_tag_id
    # tid cache is rebuilt lazily on the copy (fresh tensor identities)
    assert other.node_tid(2) in other.tensor_map


def test_plain_tensor_network_cast_requires_explicit_plan():
    """A generic Quimb network cannot silently become geometry-owning."""
    plain = qtn.TensorNetwork([
        qtn.Tensor(np.ones(2), inds=("k0",)),
    ])
    with pytest.raises(TypeError, match="explicit TreePlan"):
        TreeTensorNetwork(plain)


def test_layout_rejects_out_of_range_supports_early():
    """Bad interaction supports fail during layout construction, not replay."""
    with pytest.raises(ValueError, match="outside"):
        TreeLayoutFinder(supports=[(0, 3)], n=2)


def test_three_qubit_torch_operator_is_backend_coerced():
    """Torch operators work with the default NumPy-backed TTN state."""
    torch = pytest.importorskip("torch")
    opt = TreeOptimizer(None, n=3, run=False)
    with pytest.warns(UserWarning, match="backend-compatible gate"):
        opt.apply_subtree_operator(
            torch.eye(8, dtype=torch.complex128), (0, 1, 2)
        )
    assert np.allclose(opt.to_dense(), [1.0] + [0.0] * 7)


def test_tree_torch_state_stays_native_across_public_operations():
    """Tree controls, Pauli helpers, and readout preserve a Torch TTN."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(3), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    assert state.validate_isometry_metadata() is state
    h = to_backend(
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    )
    cnot = to_backend(np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    ))

    opt = TreeOptimizer([(h, 0), (cnot, (0, 1))], state=state, chi=8)
    assert opt.backend_info() == {
        "backend": "torch", "dtype": "complex128", "device": "cpu",
    }
    assert opt.expectation_pauli("ZZ", (0, 1)) == pytest.approx(1.0)
    opt.apply_subtree_operator(to_backend(np.eye(8, dtype=complex)), (0, 1, 2))
    assert opt.tn.validate(check_canonical=True) is opt.tn
    opt.project_pauli("ZZ", (0, 1), +1)
    assert opt.measure(2, outcome=0) == 0
    assert opt.reset(2) == 0
    opt.cap(2, to_backend(np.array([1.0, 0.0], dtype=complex)))
    opt.apply_pauli_rotation(0.2, "XZ", (0, 1))
    opt.apply_pauli_sum([(1.0, {0: "X", 1: "Z"})])

    assert all(torch.is_tensor(tensor.data) for tensor in opt.tn.tensor_map.values())


def test_tree_canonical_check_uses_backend_scalar_reduction(monkeypatch):
    """Canonical diagnostics must not convert device tensors with NumPy."""
    torch = pytest.importorskip("torch")
    import importlib

    ttn_module = importlib.import_module("pepsy.optimizers.tree.ttn")
    state = TreeTensorNetwork.from_plan(TreePlan.from_order(range(3)))
    state.apply_to_arrays(
        pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    )

    def fail_to_numpy(_value):
        raise AssertionError("canonical checks must stay on the live backend")

    monkeypatch.setattr(ttn_module.ar, "to_numpy", fail_to_numpy)
    assert state.is_canonical_form()


def test_tree_warns_once_when_a_gate_does_not_match_the_state_backend():
    """User payload mismatches are explicit while compatibility is preserved."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    opt = TreeOptimizer(None, state=state, run=False)

    with pytest.warns(UserWarning, match="backend-compatible gate"):
        opt.apply_1q(np.eye(2, dtype=complex), 0)
    opt.apply_1q(np.eye(2, dtype=complex), 1)
    assert opt.backend_info()["backend"] == "torch"


def test_tree_gate_stream_backend_requires_explicit_preparation():
    """Every stream gate must already match the live TTN backend."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    gates = [
        np.eye(2, dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    ]
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    assert all(isinstance(gate, np.ndarray) for gate in gates)

    matching = [to_backend(gate) for gate in gates]
    opt.set_gates([(matching[0], 0), (matching[1], 1)])
    with pytest.raises(TypeError, match=r"stream\[1\].*gate"):
        opt.set_gates([(matching[0], 0), (gates[1], 1)])


def test_tree_gate_stream_backend_checks_late_payloads():
    """A matching first gate cannot hide a later backend mismatch."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    matching = to_backend(np.eye(2, dtype=complex))
    foreign = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    with pytest.raises(TypeError, match=r"stream\[1\].*gate"):
        opt._validate_gate_stream_backend(
            [matching, foreign], ["gate", "gate"]
        )


def test_tree_optimizer_reports_symmray_block_backend():
    """Native fermionic TTNs report their underlying block backend."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")

    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(3), structure="balanced")
    state = pepsy.ps_to_ttn(
        3,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        dtype="complex128",
    )
    state.apply_to_arrays(
        pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    )
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    assert opt.backend_info() == {
        "backend": "symmray",
        "dtype": "complex128",
        "device": "cpu",
        "array_backend": "torch",
    }


def test_tree_submpo_stream_backend_requires_explicit_preparation():
    """Every stream sub-MPO tensor must match the live TTN backend."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    submpo = _two_branch_flip_submpo(L=2, sites=(0, 1), targets=(0, 1))
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    with pytest.raises(TypeError, match=r"stream\[0\].*sub-MPO"):
        opt.set_gates([opt.submpo_event(submpo, (0, 1))])
    assert all(isinstance(tensor.data, np.ndarray) for tensor in submpo.tensors)

    prepared = opt.to_backend(submpo)
    opt.set_gates([opt.submpo_event(prepared, (0, 1))])


def test_tree_treemppo_stream_backend_requires_explicit_preparation():
    """Every TreeMPO tensor must match the live TTN backend."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    state.apply_to_arrays(to_backend)
    identity = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
    tree_mpo = TreeMPO.from_terms(
        plan,
        {(0, 1): identity},
        compress=False,
    )
    opt = TreeOptimizer(None, state=state, tree=plan, run=False)

    with pytest.raises(TypeError, match=r"stream\[0\].*TreeMPO"):
        opt.set_gates([opt.subtreempo_event(tree_mpo, (0, 1))])

    prepared = opt.to_backend(tree_mpo)
    assert all(
        torch.is_tensor(tensor.data)
        for network in prepared.tree_networks
        for tensor in network
    )
    assert all(
        isinstance(tensor.data, np.ndarray)
        for network in tree_mpo.tree_networks
        for tensor in network
    )
    opt.set_gates([opt.subtreempo_event(prepared, (0, 1))])
    opt.run()
    np.testing.assert_allclose(opt.to_dense(), [1.0, 0.0, 0.0, 0.0])
    assert opt.tn.validate(check_canonical=True) is opt.tn


def test_tree_rejects_a_mixed_backend_initial_state():
    """A TTN must use one backend, dtype, and device across all tensors."""
    torch = pytest.importorskip("torch")
    plan = TreePlan.from_order(range(2), structure="balanced")
    state = TreeTensorNetwork.from_plan(plan)
    leaf = state.leaf_of_qubit(0)
    state.node_tensor(leaf).modify(
        data=torch.as_tensor(state.node_tensor(leaf).data, dtype=torch.complex128)
    )

    with pytest.raises(TypeError, match="one compatible backend"):
        TreeOptimizer(None, state=state, run=False)


def test_ttn_geometry_helpers_match_plan():
    """Geometry delegators agree with the underlying TreePlan."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    root = ttn.root
    for child in ttn.children(root):
        assert ttn.parent(child) == root
        assert root in ttn.neighbors(child)
        # deterministic, symmetric bond name
        assert ttn.bond(child, root) == ttn.bond(root, child)
    leaf = ttn.leaf_of_qubit(0)
    assert ttn.qubit_of_leaf(leaf) == 0
    assert ttn.is_leaf(leaf)
    # steiner subtree of two leaves == their node path
    la, lb = ttn.leaf_of_qubit(0), ttn.leaf_of_qubit(5)
    assert ttn.steiner_nodes([la, lb]) == set(ttn.node_path(la, lb))
    with pytest.raises(ValueError):
        ttn.bond(la, lb)  # non-adjacent


def test_ttn_validate_checks_structure_and_canonicality():
    """TreeTensorNetwork.validate catches malformed physical legs."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    assert ttn.validate(check_canonical=True) is ttn

    broken = ttn.copy()
    broken.reindex_({broken.site_ind(0): "broken-physical"})
    with pytest.raises(ValueError, match="missing physical index"):
        broken.validate()


def test_tree_canonize_mps_compatibility_entry_point():
    """Shared coefficient frontends can use the MPS canonicalization name."""
    opt = TreeOptimizer(None, n=4, run=False)
    info = {}
    assert opt.canonize_mps(opt.p, (0, 3), info=info) == (0, 3)
    assert info["cur_orthog"] == (0, 3)
    assert opt.is_subtree_canonical_form()
    assert opt.canonize_mps(opt.p, 2, info=info) == (2, 2)
    assert opt.is_canonical_form(opt.plan.leaf_of_qubit[2])


def test_ttn_rand_is_canonical_around_root():
    """rand(canonicalize=True) leaves the root tensor as the orthogonality centre."""
    import quimb.tensor as qtn

    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.rand(plan, D=4, seed=0)
    root_t = ttn.node_tensor(ttn.root)
    canon_norm = float(
        np.sqrt(np.abs(qtn.tensor_contract(root_t.H, root_t, output_inds=[])))
    )
    full_norm = float(np.sqrt(np.abs((ttn.H & ttn).contract(output_inds=[]))))
    assert np.isclose(canon_norm, full_norm)


@pytest.mark.filterwarnings(
    "ignore:The contraction tree is not a compressed one"
)
def test_ttn_gate_and_local_expectation():
    """Inherited gate/canonicalisation/expectation work on the tree."""
    plan = TreePlan.from_order(range(5), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    ttn.gate_inds_(x, [ttn.site_ind(2)], contract=True)
    ttn.canonize_around_node_(ttn.leaf_of_qubit(2))
    val = ttn.local_expectation(z, [2], max_bond=None, optimize="auto")
    assert abs(val + 1.0) < 1e-9  # <Z> = -1 after X


def test_ttn_multisite_local_expectation_contracts_only_canonical_subtree():
    """The custom Steiner-subtree readout matches a full dense-TTN overlap."""
    rng = np.random.default_rng(41)
    ttn = TreeTensorNetwork.rand(TreePlan.from_order(range(5)), D=3, seed=7)
    matrix = rng.standard_normal((4, 4)) + 1.0j * rng.standard_normal((4, 4))
    operator = matrix + matrix.conj().T
    where = (1, 4)
    operated = qtn.tensor_network_gate_inds(
        ttn,
        operator,
        [ttn.site_ind(site) for site in where],
        contract=False,
        inplace=False,
        tags=[],
    )
    expected = (ttn.H | operated).contract(all, optimize="auto")
    expected /= (ttn.H | ttn).contract(all, optimize="auto")

    actual = ttn.local_expectation(operator, where, max_bond=None, optimize="auto")
    assert actual == pytest.approx(expected)


def test_optimizer_state_is_a_tree_tensor_network():
    """TreeOptimizer builds its state on the TreeTensorNetwork class."""
    opt = TreeOptimizer(None, n=4)
    assert isinstance(opt.tn, TreeTensorNetwork)
    assert opt.tn.plan is opt.plan


def test_ttn_show_ascii_tree(capsys):
    """show() prints a top-down tree with leaf/qubit labels and bond dims."""
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    text = ttn.ascii_tree()
    # root marker on top, qubit leaves labelled at the bottom
    assert text.splitlines()[0].strip() == "\u25cf"
    for q in range(4):
        assert f"q{q}" in text
    assert "\u25c6" in text  # leaf markers drawn
    # box-drawing connectors are used
    assert "\u2534" in text and "\u250c" in text

    def dim_rows(drawing):
        # bond-dim annotation rows contain only digits and whitespace
        return [
            ln for ln in drawing.splitlines()
            if ln.strip() and all(c.isdigit() or c.isspace() for c in ln)
        ]

    # product state: every annotated bond dimension is 1
    rows = dim_rows(text)
    assert rows and all(set(ln.split()) <= {"1"} for ln in rows)
    # dropping bond dims removes the annotation rows but keeps the structure
    assert not dim_rows(ttn.ascii_tree(bond_dims=False))
    # the coloured drawing embeds ANSI escapes but strips back to the plain one
    colored = ttn.ascii_tree(color=True)
    assert "\x1b[" in colored
    import re as _re
    assert _re.sub(r"\x1b\[[0-9;]*m", "", colored) == text
    # show() prints the coloured drawing by default (+ trailing newline)
    ttn.show()
    assert capsys.readouterr().out.rstrip("\n") == colored
    # ...and the plain drawing when colour is disabled
    ttn.show(color=False)
    assert capsys.readouterr().out.rstrip("\n") == text
    # optimizer delegates to the state's drawing
    TreeOptimizer(None, n=4).show(color=False)
    assert capsys.readouterr().out.rstrip("\n") == text


# -- non-binary / arbitrary-arity trees ---------------------------------------


def _nonbinary_plan():
    """Two arity-3 star nodes under a binary root over qubits 0..5."""
    children = {
        0: (), 1: (), 2: (), 3: (), 4: (), 5: (),
        6: (0, 1, 2), 7: (3, 4, 5), 8: (6, 7),
    }
    qubit_of_leaf = {i: i for i in range(6)}
    return TreePlan.from_children(children, qubit_of_leaf)


def test_from_children_builds_and_validates():
    """from_children builds an arbitrary-arity tree and validates its shape."""
    plan = _nonbinary_plan()
    assert plan.n == 6
    assert plan.root == 8
    assert plan.max_arity() == 3
    assert not plan.is_binary()
    assert plan.parent[6] == 8 and plan.parent[0] == 6
    # star geodesics inside a clique are length two (vs up to three when split)
    assert plan.tree_distance(0, 1) == 2
    assert plan.tree_distance(0, 2) == 2


def test_from_children_rejects_invalid_trees():
    """from_children raises on malformed children / leaf maps."""
    # a node with two parents
    with pytest.raises(ValueError):
        TreePlan.from_children(
            {0: (), 1: (), 2: (0, 1), 3: (0,)}, {0: 0, 1: 1}
        )
    # a leaf missing its qubit assignment
    with pytest.raises(ValueError):
        TreePlan.from_children({0: (), 1: (), 2: (0, 1)}, {0: 0})
    # leaf qubits must be 0..n-1 without gaps
    with pytest.raises(ValueError):
        TreePlan.from_children(
            {0: (), 1: (), 2: (0, 1)}, {0: 0, 1: 2}
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nonbinary_tree_matches_statevector(seed):
    """Untruncated replay on a hand-built non-binary tree is exact."""
    rng = np.random.default_rng(seed)
    n = 6
    plan = _nonbinary_plan()
    stream = _random_stream(n, 8 * n, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=256)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


@pytest.mark.parametrize("max_arity", [3, 4])
def test_kary_layout_flatter_and_exact(max_arity):
    """k-ary layouts raise the arity and still replay exactly at large chi."""
    rng = np.random.default_rng(11)
    n = 8
    plan = TreePlan.from_order(range(n), structure="balanced",
                               max_arity=max_arity)
    assert plan.max_arity() <= max_arity
    assert plan.max_arity() > 2  # genuinely non-binary
    stream = _random_stream(n, 40, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=1 << n)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_binary_defaults_unchanged():
    """max_arity=2 keeps the strictly-binary tree for every structure."""
    for structure in ("quality", "balanced"):
        plan = TreePlan.from_order(range(9), structure=structure)
        assert plan.is_binary()
    # a scalar max_arity=2 opts back into a binary layout-finder tree
    rng = np.random.default_rng(2)
    stream = _random_stream(8, 60, rng)
    assert TreeLayoutFinder(stream, n=8, max_arity=2).run().is_binary()


def test_adaptive_layout_emits_star_for_cliques():
    """Adaptive layout collapses mutually coupled cliques into flat stars."""
    stream = []
    for _ in range(20):
        for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
            stream.append((np.eye(4, dtype=complex), (a, b)))
    stream.append((np.eye(4, dtype=complex), (2, 3)))  # weak cross link
    finder = TreeLayoutFinder(stream, n=6, structure="adaptive",
                              max_arity=None)
    plan = finder.run()
    assert plan.max_arity() == 3  # each clique becomes an arity-3 star
    # every intra-clique geodesic is the star length two
    for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        assert plan.tree_distance(a, b) == 2
    # and it is a better structure than the binary layout for these weights
    binary = TreePlan.from_order(range(6), weights=finder._similarity_weights(),
                                 structure="quality", max_arity=2)
    assert finder.score(plan) < finder.score(binary)


def test_adaptive_layout_replays_exactly():
    """A star-containing adaptive tree replays a random circuit exactly."""
    rng = np.random.default_rng(7)
    n = 6
    # build an adaptive plan from a clustered stream, then replay a fresh one
    layout_stream = []
    for _ in range(15):
        for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
            layout_stream.append((_rand_unitary(2, rng), (a, b)))
    plan = TreeLayoutFinder(layout_stream, n=n, structure="adaptive",
                            max_arity=None).run()
    assert not plan.is_binary()
    stream = _random_stream(n, 40, rng)
    opt = TreeOptimizer(stream, n=n, tree=plan, chi=1 << n)
    psi = _exact_state(stream, n)
    assert _fidelity(psi, opt.to_dense()) > 1 - 1e-8


def test_nonbinary_ascii_tree_renders_arity():
    """ascii_tree draws an internal node with more than two children."""
    plan = _nonbinary_plan()
    ttn = TreeTensorNetwork.from_plan(plan)
    text = ttn.ascii_tree()
    for q in range(6):
        assert f"q{q}" in text
    # an arity-3 star centres the middle child under the parent stem ('┼')
    assert "\u253c" in text


# -- orthogonality-centre movement --------------------------------------------


def _entangled_ttn(seed=0, n=6, D=3, structure="balanced"):
    """A canonical-at-root random tree state for centre-movement tests."""
    plan = TreePlan.from_order(range(n), structure=structure)
    return TreeTensorNetwork.rand(plan, D=D, seed=seed)


def test_isometry_metadata_api_has_one_network_owned_orientation_map():
    """Product construction and optimizer delegates expose one live map."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.from_plan(plan)
    directions = ttn.isometry_map()

    assert directions[plan.root] is None
    for nid in plan.nodes():
        if nid == plan.root:
            continue
        assert directions[nid] == plan.parent[nid]
        assert ttn.can_skip_canonize(nid, plan.parent[nid])
        assert ttn.can_skip_canonize(
            plan.parent[nid], nid, absorb="left",
        )
    assert ttn.validate_isometry_metadata() is ttn
    assert ttn.validate(check_canonical=True) is ttn

    opt = TreeOptimizer(None, tree=plan, state=ttn, run=False)
    assert opt.isometry_map() == directions
    leaf = plan.leaf_of_qubit[0]
    assert opt.isometry_direction(leaf) == plan.parent[leaf]
    assert opt.can_skip_canonize(leaf, plan.parent[leaf])
    assert opt.validate_isometry_metadata() is opt


def test_isometry_metadata_validation_detects_cleared_local_proof():
    """A live canonical-region claim cannot outlast cleared ``left_inds``."""
    ttn = _entangled_ttn(seed=43)
    leaf = ttn.leaf_of_qubit(0)
    tensor = ttn.node_tensor(leaf)
    assert ttn.isometry_direction(leaf) == ttn.parent(leaf)

    # Quimb correctly clears ``left_inds`` whenever tensor data changes.
    tensor.modify(data=np.array(tensor.data))
    assert ttn.isometry_direction(leaf) is None
    with pytest.raises(ValueError, match="must be isometric"):
        ttn.validate_isometry_metadata()
    with pytest.raises(ValueError, match="must be isometric"):
        ttn.validate(check_canonical=True)

    ttn.invalidate_canonical_form()
    assert ttn.validate_isometry_metadata() is ttn


def test_shift_center_lossless_and_recanonical():
    """Shifting the centre preserves the state exactly and re-canonicalises."""
    ttn = _entangled_ttn(seed=1)
    assert ttn.orthogonality_center == ttn.root
    assert ttn.is_canonical_form()  # about the tracked centre
    sv0 = ttn.to_statevector()
    for target in (
        ttn.leaf_of_qubit(5), ttn.leaf_of_qubit(0), ttn.root,
        ttn.leaf_of_qubit(3),
    ):
        ttn.shift_orthogonality_center(target)
        assert ttn.orthogonality_center == target
        assert ttn.is_canonical_form(target)
        assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_shift_center_idempotent_touches_nothing():
    """Shifting to the current centre is a no-op that mutates no tensor."""
    ttn = _entangled_ttn(seed=2)
    snap = {nid: np.array(ttn.node_tensor(nid).data) for nid in ttn.plan.nodes()}
    ttn.shift_orthogonality_center(ttn.orthogonality_center)
    for nid in ttn.plan.nodes():
        assert np.array_equal(ttn.node_tensor(nid).data, snap[nid])


def test_dense_edge_canonization_skips_proven_isometry(monkeypatch):
    """A direct lossless edge move reuses a live dense ``left_inds`` proof."""
    ttn = _entangled_ttn(seed=40)
    leaf = ttn.leaf_of_qubit(0)
    parent = ttn.parent(leaf)
    ttn.shift_orthogonality_center(parent)
    assert ttn.orthogonality_center == parent
    calls = []
    canonize_between = ttn.canonize_between

    def traced_canonize_between(*args, **kwargs):
        calls.append((args, kwargs))
        return canonize_between(*args, **kwargs)

    monkeypatch.setattr(ttn, "canonize_between", traced_canonize_between)
    before = {
        nid: np.array(ttn.node_tensor(nid).data)
        for nid in ttn.plan.nodes()
    }

    assert ttn.can_skip_canonize(leaf, parent)
    ttn.canonize_edge_(leaf, parent)

    assert not calls
    assert all(
        np.array_equal(ttn.node_tensor(nid).data, data)
        for nid, data in before.items()
    )


def test_shift_center_from_unknown_canonicalises_once():
    """An unknown centre falls back to a full canonicalisation about the target."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=3, canonicalize=False)
    assert ttn.orthogonality_center is None
    assert not ttn.is_canonical_form()
    leaf = ttn.leaf_of_qubit(4)
    ttn.shift_orthogonality_center(leaf)
    assert ttn.orthogonality_center == leaf
    assert ttn.is_canonical_form(leaf)


def test_center_move_only_touches_geodesic():
    """A centre move is O(path length): off-geodesic tensors are untouched."""
    ttn = _entangled_ttn(seed=4)
    src = ttn.orthogonality_center
    dst = ttn.leaf_of_qubit(5)
    path = set(ttn.node_path(src, dst))
    off = [nid for nid in ttn.plan.nodes() if nid not in path]
    assert off  # the geodesic does not span the whole tree
    snap = {nid: np.array(ttn.node_tensor(nid).data) for nid in off}
    ttn.shift_orthogonality_center(dst)
    for nid in off:
        assert np.array_equal(ttn.node_tensor(nid).data, snap[nid])


def test_center_moves_use_lossless_qr_in_quimb(monkeypatch):
    """Known-centre moves explicitly select Quimb's non-truncating QR path."""
    ttn = _entangled_ttn(seed=41)
    calls = []
    canonize_between = ttn.canonize_between

    def traced_canonize_between(*args, **kwargs):
        calls.append(dict(kwargs))
        return canonize_between(*args, **kwargs)

    monkeypatch.setattr(ttn, "canonize_between", traced_canonize_between)
    ttn.shift_orthogonality_center(ttn.leaf_of_qubit(5))

    assert calls
    assert all(call["method"] == "qr" for call in calls)
    assert all(call["cutoff"] == 0.0 for call in calls)
    assert ttn.is_canonical_form()


def test_shift_center_recovers_from_multinode_region_locally():
    """A tracked canonical region is reduced by QR without touching its exterior."""
    ttn = _entangled_ttn(seed=42)
    region = {ttn.root, *ttn.children(ttn.root)}
    ttn.canonize_subtree_(region)
    assert ttn.orthogonality_center is None

    outside = [nid for nid in ttn.plan.nodes() if nid not in region]
    snapshot = {
        nid: np.array(ttn.node_tensor(nid).data)
        for nid in outside
    }
    target = sorted(region - {ttn.root})[0]

    ttn.shift_orthogonality_center(target)

    assert ttn.orthogonality_center == target
    assert ttn.is_canonical_form(target)
    for nid in outside:
        assert np.array_equal(ttn.node_tensor(nid).data, snapshot[nid])


def test_two_qubit_anchor_uses_nearest_endpoint(monkeypatch):
    """A non-sibling gate starts from the endpoint nearest the centre."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    opt = TreeOptimizer(None, n=8, tree=plan, run=False)
    near = plan.leaf_of_qubit[7]
    opt.shift_orthogonality_center(near)

    moves = []
    move_center = opt._move_center

    def traced_move_center(target):
        moves.append(target)
        return move_center(target)

    monkeypatch.setattr(opt, "_move_center", traced_move_center)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    opt.apply_2q(np.kron(x, x), 0, 7)

    assert moves == [near]
    assert opt.center == near


def test_shift_center_validates_node():
    """Shifting to a non-node raises loudly."""
    ttn = _entangled_ttn(seed=5)
    with pytest.raises(ValueError):
        ttn.shift_orthogonality_center(9999)


def test_canonize_edge_tracks_centre_honestly():
    """A lone edge move advances the centre by one hop or marks it unknown."""
    ttn = _entangled_ttn(seed=6)  # centre at root
    root = ttn.root
    c0, c1 = ttn.children(root)[:2]
    ttn.canonize_edge_(root, c0, absorb="right")  # centre root -> c0
    assert ttn.orthogonality_center == c0
    # an edge move not starting at the centre cannot leave a global centre
    ttn.canonize_edge_(root, c1, absorb="right")
    assert ttn.orthogonality_center is None


def test_center_survives_copy():
    """The tracked centre rides along with a network / optimizer copy."""
    opt = TreeOptimizer(None, n=6, chi=8)
    opt._move_center(opt.plan.leaf_of_qubit[4])
    ttn2 = opt.tn.copy()
    assert ttn2.orthogonality_center == opt.tn.orthogonality_center
    other = opt.copy()
    assert other.center == opt.center
    assert other.tn.orthogonality_center == opt.tn.orthogonality_center


def test_optimizer_center_is_network_view():
    """optimizer.center is a single value shared with the network; moves stay canonical."""
    rng = np.random.default_rng(11)
    n = 6
    opt = TreeOptimizer(_random_stream(n, 20, rng), n=n, chi=1 << n)  # exact
    assert opt.center == opt.tn.orthogonality_center
    for q in (0, 5, 2):
        leaf = opt.plan.leaf_of_qubit[q]
        opt._move_center(leaf)
        assert opt.center == leaf == opt.tn.orthogonality_center
        assert opt.tn.is_canonical_form(leaf)


def test_optimizer_public_canonicalisation_api():
    """TreeOptimizer exposes the same public canonicalisation surface as its state."""
    rng = np.random.default_rng(21)
    n = 6
    opt = TreeOptimizer(_random_stream(n, 18, rng), n=n, chi=1 << n)  # exact
    # name-parity alias reads the single shared centre
    assert opt.orthogonality_center == opt.center == opt.tn.orthogonality_center
    # public shift returns self and moves the shared centre, staying canonical
    leaf = opt.plan.leaf_of_qubit[4]
    assert opt.shift_orthogonality_center(leaf) is opt
    assert opt.center == leaf
    assert opt.is_canonical_form()  # about the tracked centre
    assert opt.is_canonical_form(leaf)
    # the alias setter writes straight through to the network
    opt.orthogonality_center = opt.plan.root
    assert opt.tn.orthogonality_center == opt.plan.root


def test_nonbinary_center_movement_is_canonical():
    """Centre movement is exact and canonical on a non-binary tree, incl. internal nodes."""
    plan = _nonbinary_plan()
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=7)
    sv0 = ttn.to_statevector()
    for target in (ttn.leaf_of_qubit(0), 6, 7, ttn.leaf_of_qubit(5), ttn.root):
        ttn.shift_orthogonality_center(target)
        assert ttn.orthogonality_center == target
        assert ttn.is_canonical_form(target)
        assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_subtree_canonicalisation_lossless_and_isometric():
    """Canonicalising around a connected subtree is lossless and gauges outside inward."""
    ttn = _entangled_ttn(seed=1)
    region = {ttn.root, *ttn.children(ttn.root)}
    assert len(region) > 1
    sv0 = ttn.to_statevector()
    ttn.canonize_subtree_(region)
    assert ttn.canonical_region == frozenset(region)
    # a multi-node region has no single orthogonality centre
    assert ttn.orthogonality_center is None
    assert ttn.is_subtree_canonical_form()          # tracked region
    assert ttn.is_subtree_canonical_form(region)    # explicit region
    assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_subtree_norm_concentrates_on_region():
    """After subtree canonicalisation the whole squared norm is carried by the region."""
    import quimb.tensor as qtn
    ttn = _entangled_ttn(seed=2)
    region = {ttn.root, *ttn.children(ttn.root)}
    ttn.canonize_subtree_(region)
    full = float(abs((ttn.H | ttn) ^ all))
    reg = qtn.TensorNetwork([ttn.node_tensor(n).copy() for n in region])
    assert np.isclose(float(abs((reg.H | reg) ^ all)), full)


def test_single_node_subtree_is_orthogonality_center():
    """A one-node subtree is exactly an orthogonality centre."""
    ttn = _entangled_ttn(seed=3)
    leaf = ttn.leaf_of_qubit(4)
    ttn.canonize_subtree_({leaf})
    assert ttn.canonical_region == frozenset({leaf})
    assert ttn.orthogonality_center == leaf
    assert ttn.is_canonical_form()
    assert ttn.is_subtree_canonical_form({leaf})


def test_subtree_span_and_connectivity_validation():
    """subtree_span links arbitrary nodes; a disconnected region needs span=True."""
    ttn = _entangled_ttn(seed=4)
    la, lb = ttn.leaf_of_qubit(0), ttn.leaf_of_qubit(5)
    span = ttn.subtree_span({la, lb})
    assert set(ttn.node_path(la, lb)) == span
    # a disconnected node set raises unless auto-spanned
    with pytest.raises(ValueError):
        ttn.canonize_subtree_({la, lb})
    ttn.canonize_subtree_({la, lb}, span=True)
    assert ttn.canonical_region == frozenset(span)
    assert ttn.is_subtree_canonical_form()


def test_canonize_around_qubits_range():
    """Qubit-level range canonicalisation spans the right subtree and stays canonical."""
    ttn = _entangled_ttn(seed=5)
    sv0 = ttn.to_statevector()
    ttn.canonize_around_qubits_([1, 2, 3])
    leaves = [ttn.leaf_of_qubit(q) for q in (1, 2, 3)]
    assert ttn.canonical_region == frozenset(ttn.subtree_span(leaves))
    assert ttn.is_subtree_canonical_form()
    assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10
    # a single qubit collapses to a one-leaf orthogonality centre
    ttn.canonize_around_qubits_([2])
    assert ttn.orthogonality_center == ttn.leaf_of_qubit(2)


def test_subtree_region_survives_copy():
    """A multi-node canonical region rides along with a copy."""
    ttn = _entangled_ttn(seed=6)
    region = {ttn.root, *ttn.children(ttn.root)}
    ttn.canonize_subtree_(region)
    clone = ttn.copy()
    assert clone.canonical_region == ttn.canonical_region
    assert clone.orthogonality_center is None
    assert clone.is_subtree_canonical_form()


def test_nonbinary_subtree_canonicalisation():
    """Subtree canonicalisation works around an internal star node on a non-binary tree."""
    plan = _nonbinary_plan()
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=7)
    sv0 = ttn.to_statevector()
    region = {8, 6, 7}  # root plus both arity-3 star nodes
    ttn.canonize_subtree_(region)
    assert ttn.canonical_region == frozenset(region)
    assert ttn.is_subtree_canonical_form()
    assert _fidelity(sv0, ttn.to_statevector()) > 1 - 1e-10


def test_optimizer_subtree_canonicalisation_api():
    """TreeOptimizer mirrors the state's public subtree-canonicalisation surface."""
    rng = np.random.default_rng(31)
    n = 6
    opt = TreeOptimizer(_random_stream(n, 16, rng), n=n, chi=1 << n)  # exact
    region = {opt.plan.root, *opt.plan.children[opt.plan.root]}
    # public canonize_subtree returns self and installs the shared region view
    assert opt.canonize_subtree(region) is opt
    assert opt.canonical_region == opt.tn.canonical_region == frozenset(region)
    assert opt.is_subtree_canonical_form()
    # qubit-level range entry point
    assert opt.canonize_around_qubits([0, 5]) is opt
    leaves = [opt.plan.leaf_of_qubit[q] for q in (0, 5)]
    assert opt.canonical_region == frozenset(opt.tn.subtree_span(leaves))
    assert opt.is_subtree_canonical_form()
    # the region setter writes straight through to the network
    opt.canonical_region = {opt.plan.root}
    assert opt.tn.canonical_region == frozenset({opt.plan.root})
    assert opt.orthogonality_center == opt.plan.root


def test_live_bond_dimensions_survive_general_threading():
    """Tree edge diagnostics use live bonds after a non-sibling gate."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    opt = TreeOptimizer(None, n=8, tree=plan, chi=8)
    opt.apply_gate(h, 0)
    opt.apply_gate(cnot, (0, 6))

    for node in plan.nodes():
        for child in plan.children[node]:
            ix = opt.tn.bond(node, child)
            assert ix in opt.tn.ind_map
            assert opt.tn._bond_dim(node, child) == opt.tn.ind_size(ix)


def test_nonunitary_one_qubit_gate_recenters_state():
    """A non-unitary one-qubit gate cannot leave a stale canonical centre."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    projector = np.diag([1.0, 0.0]).astype(complex)
    opt = TreeOptimizer([(h, 0), (cnot, (0, 7))], n=8, tree=plan)
    opt.apply_gate(projector, 7)

    assert opt.is_canonical_form()
    assert np.isclose(opt.norm(), np.linalg.norm(opt.to_dense()))


def test_forced_measurement_validates_outcome_probability():
    """Invalid or impossible forced outcomes raise before collapsing the state."""
    opt = TreeOptimizer(None, n=2)
    with pytest.raises(ValueError, match="outcome must be 0 or 1"):
        opt.measure(0, outcome=2)
    with pytest.raises(ValueError, match="~0 probability"):
        opt.measure(0, outcome=1)
    assert np.isclose(opt.norm(), 1.0)


def test_multinode_region_normalization_preserves_canonicality():
    """Normalization scales inside a multi-node canonical region."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    opt = TreeOptimizer(None, n=8, tree=plan)
    opt.tn = TreeTensorNetwork.rand(plan, D=2, seed=31)
    leaf = plan.leaf_of_qubit[0]
    region = {leaf, plan.parent[leaf]}
    opt.tn.canonize_subtree_(region)
    opt.tn.node_tensor(leaf).modify(data=3.0 * opt.tn.node_tensor(leaf).data)

    opt.normalize()

    assert np.isclose(opt.norm(), 1.0)
    assert opt.is_subtree_canonical_form()


def test_shift_center_supports_left_absorption_orientation():
    """The optional left-absorption orientation still centres the target."""
    plan = TreePlan.from_order(range(8), structure="balanced")
    ttn = TreeTensorNetwork.rand(plan, D=3, seed=32)
    leaf = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(leaf, absorb="left")
    assert ttn.orthogonality_center == leaf
    assert ttn.is_canonical_form()


def test_tree_plan_rejects_malformed_orders():
    """TreePlan.from_order enforces the same qubit-label contract as from_children."""
    with pytest.raises(ValueError, match="at least one"):
        TreePlan.from_order([])
    with pytest.raises(ValueError, match="permutation"):
        TreePlan.from_order([0, 0])
    with pytest.raises(ValueError, match="structure"):
        TreePlan.from_order(range(2), structure="unknown")


def test_tree_gate_queue_set_and_add():
    """TreeOptimizer exposes queue replacement and extension like MpsOptimizer."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    opt = TreeOptimizer(None, n=2)
    assert opt.set_gates([(x, 0)]) is opt
    assert opt.add_gates([(x, 1)]) is opt
    assert len(opt.G) == 2
    opt.run()
    assert _fidelity(opt.to_dense(), np.array([0.0, 0.0, 0.0, 1.0])) > 1 - 1e-12


def test_optimizer_accepts_and_copies_initial_ttn():
    """TreeOptimizer can evolve an arbitrary supplied tree state independently."""
    plan = TreePlan.from_order(range(6), structure="balanced")
    state = TreeTensorNetwork.rand(plan, D=2, seed=33)
    before = state.to_statevector()
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    opt = TreeOptimizer([(x, 0)], n=6, tn=state, chi=8)

    expected = _sv_apply_1q(before, x, 0, 6)
    assert _fidelity(expected, opt.to_dense()) > 1 - 1e-10
    assert _fidelity(before, state.to_statevector()) > 1 - 1e-12
    assert opt.set_tn(state) is opt


def test_tree_native_fermionic_gate_stream_matches_mps():
    """Native (dim-4 Symmray) Fermi-Hubbard gates evolve correctly on a tree.

    The tree gate engine must apply block-sparse fermionic gates without
    reshaping them into base-2 sub-legs. At a bond dimension large enough to be
    exact, the tree real-time evolution must reproduce the MPS reference
    (identical seed) to numerical precision.
    """
    pytest.importorskip("symmray")
    tensors = pepsy.tensors

    Lx, Ly, L = 2, 2, 4
    t, U, dt = 1.0, 8.0, 0.05
    state_dtype = "complex128"

    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype=state_dtype)
    setup = fermion.lattice_half_filling(Lx, Ly, pattern="checkerboard", cyclic=True)
    mapper = tensors.OneDMap(Lx, Ly, mode="snake")
    _, coo2idx = mapper.build()
    edges_1d = [
        tuple(sorted((coo2idx[a], coo2idx[b]))) for a, b in setup.edges
    ]
    sites = tuple(range(L))
    occ_1d = {coo2idx[coo]: c for coo, c in setup.occupations.items()}
    occupations = tuple(occ_1d[p] for p in range(L))

    def build_stream():
        half = dt / 2
        u_hop = fermion.hopping_gate(half, t=t, imaginary=False)
        onsite = [
            (fermion.onsite_gate(half, site=s, U=U, mu=0.0, imaginary=False), s)
            for s in sites
        ]
        layers = fermion.edge_coloring_layers(edges_1d)
        fwd = [(u_hop, e) for layer in layers for e in layer]
        rev = [(u_hop, e) for layer in reversed(layers) for e in reversed(layer)]
        return onsite + fwd + rev + onsite

    gates = build_stream() * 3
    native_hopping = fermion.hopping_gate(dt / 2, t=t, imaginary=False)
    native_submpo = qtn.MatrixProductOperator.from_dense(
        native_hopping, dims=(4, 4), sites=(0, 2), L=L,
    )
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in native_submpo.tensors
    )

    seed_mps = pepsy.ps_to_mps(
        L, fermion=fermion, occupations=occupations,
        seed=1234, dtype=state_dtype, cyclic=False,
    )
    plan = TreeLayoutFinder(
        [(fermion.hopping_gate(0.1, t=t, imaginary=False), e) for e in edges_1d],
        n=L, chi=8, objective="hybrid",
    ).recommend_arities((2, 3, 4), seed=0)["plan"]
    seed_ttn = pepsy.ps_to_ttn(
        L, tree=plan, fermion=fermion, occupations=occupations, dtype=state_dtype
    )

    # The tree and MPS seeds must represent the identical fermionic state.
    assert float(tensors.tn_fidelity(seed_mps, seed_ttn)) > 1 - 1e-10

    # Large-chi references (no truncation for L=4: exact bond is 16).
    mps_exact = pepsy.MpsOptimizer(
        seed_mps.copy(), gates=gates, chi=256, mode="mpo", inplace=False,
    )
    mps_exact.run(cutoff=0.0)
    engine = TreeOptimizer(
        gates, n=L, tree=plan, state=seed_ttn.copy(), chi=256, cutoff=0.0,
        mode="mpo", run=False,
    )
    engine.run()

    # The two public modes are algebraically the same gate SVD: both defer
    # truncation until the whole path has been updated. They can only differ by
    # floating-point roundoff from the extra MPO factorisation/QR gauges.
    direct = TreeOptimizer(
        None, n=L, tree=plan, state=seed_ttn.copy(), chi=256, cutoff=0.0,
        mode="direct", run=False,
    )
    direct.run(gates)

    auto = TreeOptimizer(
        None, n=L, tree=plan, state=seed_ttn.copy(), chi=256, cutoff=0.0,
        mode="auto", run=False,
    )
    auto.run(gates)

    assert float(tensors.tn_fidelity(engine.p, direct.p)) > 1 - 1e-8
    assert float(tensors.tn_fidelity(auto.p, direct.p)) > 1 - 1e-10
    assert float(tensors.tn_fidelity(direct.p, mps_exact.p)) > 1 - 1e-9
    assert float(tensors.tn_fidelity(engine.p, mps_exact.p)) > 1 - 1e-8


@pytest.mark.filterwarnings(
    "ignore:TreeOptimizer is converting a gate/operator payload"
)
def test_native_complex64_threading_disables_zero_phase_stabilization(monkeypatch):
    """Native path threading keeps structural-zero QR sectors finite."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex64",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    state = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex64",
    )
    gate = fermion.hopping_gate(0.05, t=1.0, imaginary=False)

    threaded = False
    qr_stabilized = []
    original_hop = TreeOptimizer._fermionic_thread_hop
    original_split = qtn.Tensor.split

    def traced_hop(self, u, v):
        nonlocal threaded
        threaded = True
        try:
            return original_hop(self, u, v)
        finally:
            threaded = False

    def traced_split(self, *args, **kwargs):
        if threaded and kwargs.get("method") == "qr":
            qr_stabilized.append(kwargs.get("stabilized"))
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(TreeOptimizer, "_fermionic_thread_hop", traced_hop)
    monkeypatch.setattr(qtn.Tensor, "split", traced_split)

    optimizer = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=state,
        chi=16,
        cutoff=0.0,
        mode="direct",
        run=False,
    )
    optimizer.apply_2q(gate, 0, 2)

    assert qr_stabilized
    assert qr_stabilized == [False] * len(qr_stabilized)
    assert all(
        np.isfinite(np.asarray(block)).all()
        for tensor in optimizer.tn.tensors
        for block in tensor.data.blocks.values()
    )


@pytest.mark.filterwarnings(
    "ignore:TreeOptimizer is converting a gate/operator payload"
)
def test_native_complex64_all_lossless_qr_routes_skip_zero_phase(monkeypatch):
    """All native tree QR routes avoid phase division on zero sectors."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex64",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    state = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex64",
    )
    hopping = fermion.hopping_gate(0.05, t=1.0, imaginary=False)
    routed_ops = [
        fermion.onsite_gate(
            0.01, site=site, U=8.0, mu=0.0, imaginary=False,
        )
        for site in (0, 1, 2)
    ]
    submpo = qtn.MPO_product_operator(
        routed_ops,
        sites=(0, 1, 2),
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )

    observed = []
    original_split = qtn.Tensor.split

    def traced_split(self, *args, **kwargs):
        if kwargs.get("method") == "qr" and hasattr(self.data, "blocks"):
            observed.append(kwargs.get("stabilized"))
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "split", traced_split)

    for operation in (
        lambda: TreeOptimizer(
            None, n=L, tree=plan, state=state.copy(), chi=16,
            cutoff=0.0, mode="direct", run=False,
        ).apply_2q(hopping, 0, 1),
        lambda: TreeOptimizer(
            None, n=L, tree=plan, state=state.copy(), chi=16,
            cutoff=0.0, mode="direct", run=False,
        ).apply_2q(hopping, 0, 2),
        lambda: TreeOptimizer(
            None, n=L, tree=plan, state=state.copy(), chi=16,
            cutoff=0.0, mode="submpo", run=False,
        ).apply_submpo(submpo, (0, 1, 2)),
    ):
        operation()

    assert observed
    assert observed == [False] * len(observed)


@pytest.mark.parametrize(
    ("symmetry", "occupations"),
    [
        ("U1", (1, 1, 1, 1)),
        ("U1U1", ((1, 0), (0, 1), (1, 0), (0, 1))),
    ],
)
def test_native_tree_direct_mpo_and_submpo_match_without_global_fidelity(
    symmetry, occupations,
):
    """Native gate kernels agree without Cotengra's process-based fidelity."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    seed = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    hopping = fermion.hopping_gate(0.05, t=1.0, imaginary=False)
    onsite = fermion.onsite_gate(
        0.03, site=1, U=8.0, mu=0.0, imaginary=False,
    )
    submpo = qtn.MatrixProductOperator.from_dense(
        hopping, dims=(4, 4), sites=(0, 2), L=L,
    )

    def dense_vector(opt):
        tensor = opt.tn.contract(all, optimize="greedy").transpose(
            *(opt.tn.site_ind(q) for q in range(L))
        )
        return np.asarray(tensor.data.to_dense()).reshape(-1)

    outputs = []
    for mode in ("direct", "mpo"):
        opt = TreeOptimizer(
            None,
            n=L,
            tree=plan,
            state=seed.copy(),
            chi=64,
            cutoff=0.0,
            mode=mode,
            run=False,
        )
        opt.apply_1q(onsite, 1)
        opt.apply_2q(hopping, 0, 2)
        outputs.append(dense_vector(opt))
        assert opt.tn.validate(check_canonical=True) is opt.tn

    submpo_opt = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=seed.copy(),
        chi=64,
        cutoff=0.0,
        mode="submpo",
        run=False,
    )
    submpo_opt.apply_1q(onsite, 1)
    submpo_opt.apply_submpo(submpo, (0, 2))
    outputs.append(dense_vector(submpo_opt))
    assert submpo_opt.tn.validate(check_canonical=True) is submpo_opt.tn

    for output in outputs[1:]:
        assert _fidelity(outputs[0], output) > 1 - 1e-10

    before = dense_vector(submpo_opt)
    mpo_value = submpo_opt.expectation_mpo(
        submpo, (0, 2), max_bond=64,
    )
    direct_value = submpo_opt.tn.local_expectation(hopping, (0, 2))
    assert complex(mpo_value) == pytest.approx(complex(direct_value), abs=1e-5)
    assert np.allclose(dense_vector(submpo_opt), before)


def test_native_fermionic_submpo_keeps_graded_hub_recovery(monkeypatch):
    """Native routed Q metadata skips only already-proven graded QR."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    occupations = ((1, 0), (0, 1), (1, 0), (0, 1))
    plan = TreePlan.from_order(range(L), structure="balanced")
    seed = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    sites = (0, 1, 2)
    local_ops = [
        fermion.onsite_gate(
            0.01, site=site, U=8.0, mu=0.0, imaginary=False
        )
        for site in sites
    ]
    submpo = qtn.MPO_product_operator(
        local_ops,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    candidate = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=seed.copy(),
        chi=64,
        cutoff=0.0,
        run=False,
    )
    reference = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=seed.copy(),
        chi=64,
        cutoff=0.0,
        run=False,
    )
    installs = []
    install_routed = candidate._install_routed_subtree
    compressions = []
    compress_edge = candidate._compress_edge_with_diagnostics

    def traced_install(local, snodes, hub):
        installs.append((frozenset(snodes), hub))
        assert candidate.tn.fermionic
        return install_routed(local, snodes, hub)

    def traced_compress_edge(
        u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False,
    ):
        compressions.append(reduced)
        return compress_edge(
            u,
            v,
            max_bond=max_bond,
            cutoff=cutoff,
            reduced=reduced,
            reduction_proven=reduction_proven,
        )

    monkeypatch.setattr(candidate, "_install_routed_subtree", traced_install)
    monkeypatch.setattr(
        candidate, "_compress_edge_with_diagnostics", traced_compress_edge,
    )
    candidate.apply_submpo(submpo, sites)
    for site, op in zip(sites, local_ops):
        reference.apply_1q(op, site)

    def dense_vector(opt):
        tensor = opt.tn.contract(all, optimize="greedy").transpose(
            *(opt.tn.site_ind(q) for q in range(L))
        )
        return np.asarray(tensor.data.to_dense()).reshape(-1)

    assert installs
    assert compressions
    assert all(reduced in {True, "left"} for reduced in compressions)
    assert any(reduced == "left" for reduced in compressions)
    assert (
        _fidelity(dense_vector(candidate), dense_vector(reference))
        > 1 - 1e-10
    )
    assert candidate.validate_isometry_metadata() is candidate
    for nid, toward in candidate.isometry_map().items():
        if toward is not None:
            assert candidate.can_skip_canonize(nid, toward)
    assert candidate.tn.validate(check_canonical=True) is candidate.tn


@pytest.mark.parametrize(
    ("symmetry", "spinful", "occupations"),
    [
        ("U1", False, (1, 0, 1, 0)),
        ("U1U1", True, ((1, 0), (0, 1), (1, 0), (0, 1))),
    ],
)
def test_native_fermionic_left_inds_skips_lossless_qr(
    symmetry, spinful, occupations, monkeypatch,
):
    """Symmray U1 variants reuse native QR isometry metadata safely."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    ttn = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    target = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(target)

    source = next(
        nid for nid, toward in ttn.isometry_map().items()
        if toward == target and ttn.can_skip_canonize(nid, toward)
    )
    before = {
        nid: np.asarray(ttn.node_tensor(nid).data.to_dense()).copy()
        for nid in plan.nodes()
    }
    calls = []
    graded_qr = ttn._fermionic_canonize_edge_

    def traced_qr(*args, **kwargs):
        calls.append((args, kwargs))
        return graded_qr(*args, **kwargs)

    monkeypatch.setattr(ttn, "_fermionic_canonize_edge_", traced_qr)
    ttn.canonize_edge_(source, target)

    assert not calls
    assert ttn.orthogonality_center == target
    assert ttn.is_canonical_form(target)
    assert ttn.validate(check_canonical=True) is ttn
    for nid in plan.nodes():
        np.testing.assert_array_equal(
            np.asarray(ttn.node_tensor(nid).data.to_dense()), before[nid]
        )


def test_native_truncating_compression_keeps_explicit_svd(monkeypatch):
    """A positive cutoff never turns a native truncation into metadata-only."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex128",
    )
    target = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(target)
    source = next(
        nid for nid, toward in ttn.isometry_map().items()
        if toward == target and ttn.can_skip_canonize(nid, toward)
    )

    calls = []
    compress = ttn._fermionic_compress_edge_

    def traced_compress(*args, **kwargs):
        calls.append((args, kwargs))
        return compress(*args, **kwargs)

    monkeypatch.setattr(ttn, "_fermionic_compress_edge_", traced_compress)
    ttn.compress_edge_(
        source, target, max_bond=64, cutoff=1e-10, cutoff_mode="rel",
    )

    assert calls


def test_native_one_sided_compression_qr_reduces_before_svd(monkeypatch):
    """A proven native isometry sends only its reduced core to the SVD."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex128",
    )
    target = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(target)
    source = next(
        nid for nid, toward in ttn.isometry_map().items()
        if toward == target and ttn.can_skip_canonize(nid, toward)
    )

    split_methods = []
    original_split = qtn.Tensor.split

    def traced_split(self, *args, **kwargs):
        method = kwargs.get("method")
        if method in {"qr", "svd"} and hasattr(self.data, "blocks"):
            split_methods.append((method, tuple(self.shape)))
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "split", traced_split)
    ttn._fermionic_compress_edge_(
        target,
        source,
        max_bond=64,
        cutoff=1e-10,
        cutoff_mode="rel",
        absorb="right",
        reduced="left",
    )

    assert [method for method, _shape in split_methods] == ["qr", "svd"]
    assert ttn.validate(check_canonical=True) is ttn


def test_native_profile_reports_reduced_compression_routes():
    """Native profiling exposes reduced compression and no hidden fallback."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex128",
    )
    hopping = fermion.hopping_gate(0.05, t=1.0, imaginary=False)
    optimizer = TreeOptimizer(
        None,
        n=4,
        tree=plan,
        state=state,
        chi=1,
        cutoff=0.0,
        mode="direct",
        profile=True,
        run=False,
    )

    optimizer.apply_2q(hopping, 0, 2)
    report = optimizer.profile_report()
    routes = report["native_compression_routes"]

    assert routes
    assert routes.get("full_svd_fallback", 0) == 0
    assert sum(
        count for route, count in routes.items()
        if route != "full_svd_fallback"
    ) == report["by_kind"]["native_compression_route"]["count"]
    assert optimizer.tn.validate(check_canonical=True) is optimizer.tn


@pytest.mark.parametrize(
    ("symmetry", "spinful", "occupations"),
    [
        ("U1", False, (1, 0, 1, 0)),
        ("U1U1", True, ((1, 0), (0, 1), (1, 0), (0, 1))),
    ],
)
@pytest.mark.parametrize("state_dtype", ["complex64", "complex128"])
def test_native_one_sided_and_two_sided_compression_fidelity(
    symmetry, spinful, occupations, state_dtype,
):
    """Native left/right/two-sided reductions preserve the state and gauge."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype=state_dtype,
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    base = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype=state_dtype,
    )
    target = plan.leaf_of_qubit[0]
    base.shift_orthogonality_center(target)
    source = next(
        nid for nid, toward in base.isometry_map().items()
        if toward == target and base.can_skip_canonize(nid, target)
    )
    cutoff = 1e-10 if state_dtype == "complex128" else 1e-7
    fidelity_floor = 1 - (1e-10 if state_dtype == "complex128" else 1e-5)

    for a, b, reduced in (
        (source, target, "right"),
        (target, source, "left"),
        (target, source, True),
    ):
        candidate = base.copy()
        candidate._fermionic_compress_edge_(
            a,
            b,
            max_bond=64,
            cutoff=cutoff,
            cutoff_mode="rel",
            absorb="right",
            reduced=reduced,
        )
        assert float(pepsy.tn_fidelity(base, candidate)) > fidelity_floor
        assert candidate.validate(check_canonical=True) is candidate


def test_native_complex64_qr_scales_low_norm_rank_deficient_block():
    """Native QR keeps tiny complex64 charge blocks finite and exact."""
    torch = pytest.importorskip("torch")
    block = torch.zeros((16, 18), dtype=torch.complex64)
    generator = torch.Generator().manual_seed(17)
    left = torch.randn((16, 12), generator=generator) * 1e-9
    right = torch.randn((12, 18), generator=generator)
    block = (left @ right).to(torch.complex64)

    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert torch.isfinite(q).all()
    assert torch.isfinite(r).all()
    torch.testing.assert_close(
        q @ r,
        block,
        rtol=2e-4,
        atol=1e-12,
    )


def test_native_complex64_qr_leaves_healthy_block_on_native_path(monkeypatch):
    """Healthy complex64 blocks use one unmodified native Torch QR call."""
    torch = pytest.importorskip("torch")

    generator = torch.Generator().manual_seed(4108)
    block = torch.randn((6, 4), generator=generator).to(torch.complex64)
    qr_calls = []
    original_qr = torch.linalg.qr

    def record_qr_call(x, *args, **kwargs):
        qr_calls.append(x.detach().clone())
        return original_qr(x, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", record_qr_call)
    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )
    expected_q, expected_r = original_qr(block)

    assert len(qr_calls) == 1
    torch.testing.assert_close(qr_calls[0], block)
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(r, expected_r)


def test_native_complex64_qr_scales_dynamic_range_block(monkeypatch):
    """Native QR scales moderate-norm blocks with tiny charge entries."""
    torch = pytest.importorskip("torch")

    block = torch.zeros((4, 10), dtype=torch.complex64)
    for index, magnitude in enumerate((8.9e-3, 8.9e-11, 8.9e-25, 8.9e-41)):
        block[index, index] = complex(magnitude, magnitude)

    qr_input_maxes = []
    original_qr = torch.linalg.qr

    def record_qr_input(x, *args, **kwargs):
        qr_input_maxes.append(float(x.abs().amax().item()))
        return original_qr(x, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", record_qr_input)
    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert len(qr_input_maxes) == 2
    assert qr_input_maxes[0] == pytest.approx(8.9e-3 * 2**0.5, rel=1e-6)
    assert 0.5 <= qr_input_maxes[1] < 1.0
    assert torch.isfinite(q).all()
    assert torch.isfinite(r).all()
    torch.testing.assert_close(
        q @ r,
        block,
        rtol=2e-4,
        atol=1e-12,
    )


def test_native_complex64_qr_handles_structurally_rank_deficient_block():
    """Native QR keeps structural rank-deficient blocks in complex64."""
    torch = pytest.importorskip("torch")
    block = torch.zeros((4, 10), dtype=torch.complex64)
    for index, magnitude in enumerate((2e-2, 2e-10, 2e-24, 2e-40)):
        block[index, index] = complex(magnitude, magnitude)

    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert torch.isfinite(q).all()
    assert torch.isfinite(r).all()
    assert q.dtype == torch.complex64
    assert r.dtype == torch.complex64
    torch.testing.assert_close(
        q @ r,
        block,
        rtol=2e-4,
        atol=1e-12,
    )


@pytest.mark.parametrize("backend_name", ["torch", "cupy"])
def test_native_complex64_gpu_qr_retries_same_device_double_precision(
    monkeypatch, backend_name,
):
    """Failed GPU QR uses an optimized same-device complex128 retry."""
    if backend_name == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        backend_module = torch
        block = torch.tensor(
            [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
            dtype=torch.complex64,
            device="cuda",
        )
        dtype32 = torch.complex64
    else:
        cupy = pytest.importorskip("cupy")
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA is unavailable")
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA is unavailable: {exc}")
        backend_module = cupy
        block = cupy.asarray(
            [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
            dtype=cupy.complex64,
        )
        dtype32 = cupy.complex64

    import pepsy.optimizers.tree.ttn as ttn_module

    original_do = ttn_module.ar.do
    qr_dtypes = []

    def fail_complex64_qr(fn, value, *args, **kwargs):
        result = original_do(fn, value, *args, **kwargs)
        if fn == "linalg.qr":
            qr_dtypes.append(value.dtype)
            if value.dtype == dtype32:
                result = tuple(
                    backend_module.full_like(
                        factor, complex(float("nan"), float("nan")),
                    )
                    for factor in result
                )
        return result

    monkeypatch.setattr(ttn_module.ar, "do", fail_complex64_qr)
    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert qr_dtypes == [dtype32, backend_module.complex128]
    assert q.dtype == dtype32
    assert r.dtype == dtype32
    assert bool(backend_module.isfinite(q).all())
    assert bool(backend_module.isfinite(r).all())
    assert float(backend_module.max(backend_module.abs(q @ r - block))) < 1e-5


def test_tree_stable_labels_route_submpo_by_payload_sites(monkeypatch):
    """Stable logical labels do not disable native structured MPO routing."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        np.kron(x, x), dims=(2, 2), sites=(2, 3), L=4
    )
    opt = TreeOptimizer(None, n=4, chi=8, run=False)
    opt.cap(1, [1.0, 0.0], compact_labels=False)

    monkeypatch.setattr(
        mpo,
        "to_dense",
        lambda: (_ for _ in ()).throw(AssertionError("dense MPO fallback")),
    )
    opt.apply_submpo(mpo, (2, 3))
    assert opt.qubits == [0, 2, 3]
    assert opt.norm() == pytest.approx(1.0)


def test_tree_stream_stable_labels_route_submpo_natively(monkeypatch):
    """Stream replay preserves native MPO routing after a stable-label cap."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    mpo = qtn.MatrixProductOperator.from_dense(
        np.kron(x, x), dims=(2, 2), sites=(2, 3), L=4
    )
    events = [
        TreeOptimizer.cap_event(1, [1.0, 0.0], compact_labels=False),
        TreeOptimizer.submpo_event(mpo, (2, 3)),
    ]
    opt = TreeOptimizer(None, n=4, chi=8, run=False)

    monkeypatch.setattr(
        mpo,
        "to_dense",
        lambda: (_ for _ in ()).throw(AssertionError("dense MPO fallback")),
    )
    opt.run(events)
    assert opt.qubits == [0, 2, 3]
    assert opt.norm() == pytest.approx(1.0)


def test_tree_estimate_bonds_tracks_compact_plan_after_cap():
    """Bond preflight follows the live logical mapping across a cap event."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    events = [
        TreeOptimizer.cap_event(1, [1.0, 0.0], compact_labels=False),
        (cnot, (2, 3)),
    ]
    opt = TreeOptimizer(
        None,
        n=4,
        tree=TreePlan.from_order(range(4), structure="balanced"),
        run=False,
    )
    report = opt.estimate_bonds(events)

    assert report["events"][0]["kind"] == "cap"
    assert report["events"][1]["support"] == (1, 2)
    assert report["events"][1]["crossing_edges"]


def test_tree_auto_layout_remaps_supports_after_compact_cap():
    """Automatic layout uses original leaves for post-cap compact labels."""
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )
    events = [
        TreeOptimizer.cap_event(1, [1.0, 0.0]),
        (cnot, (1, 2)),
    ]
    opt = TreeOptimizer(events, n=4, run=False)

    assert (2, 3) in opt.layout_finder.supports
