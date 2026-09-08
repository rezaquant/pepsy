"""Focused tests for the tree-embedded PEPS optimizer."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy.fitting import TreeFIT
from pepsy.optimizers import (
    MpsOptimizer,
    TreePeps,
    TreePepsOptimizer,
    TreePepsPlan,
    TreePepo,
    TreeSubPepo,
)

pytestmark = pytest.mark.smoke


def _cnot():
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )


def _path_plan(shape, **kwargs):
    """Make the explicit MPS-compatible geometry used by chain tests."""

    kwargs.setdefault("topology", "path")
    return TreePepsPlan.from_shape(shape, **kwargs)


def test_chain_compression_matches_mps_svd_when_cap_is_sufficient():
    """A TreePeps path must use the same optimal sweep as an MPS."""

    n = 6
    plan = _path_plan((1, n), order="row-major", tree_order="snake")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 5)),
        (qu.hadamard(), (1,)),
        (qu.CNOT(), (1, 4)),
        (qu.CNOT(), (0, 3)),
    ]

    exact = TreePepsOptimizer(
        TreePeps.from_plan(plan), gates=gates, chi=None, cutoff=0.0,
        track_infidelity=False,
    )
    tree = TreePepsOptimizer(
        TreePeps.from_plan(plan), gates=gates, chi=4, cutoff=0.0,
        track_infidelity=False,
    )
    mps = MpsOptimizer(
        qtn.MPS_computational_state("0" * n, dtype="complex128"),
        gates=gates,
        chi=4,
        mode="svd",
    )
    mps.run(progbar=False, cutoff=0.0)

    exact_vector = np.asarray(exact.state.to_statevector()).reshape(-1)
    tree_vector = np.asarray(tree.state.to_statevector()).reshape(-1)
    mps_vector = np.asarray(mps.to_dense()).reshape(-1)
    np.testing.assert_allclose(tree_vector, exact_vector, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(tree_vector, mps_vector, atol=1e-10, rtol=1e-10)
    assert tree.last_report["truncated"]
    assert tree.validate(check_canonical=True) is tree


@pytest.mark.parametrize("compression_mode", ("sdc", "src", "zipup"))
def test_path_compression_modes_use_seeded_quimb_kernels(compression_mode):
    """Path TreePeps exposes Quimb's multi-tensor methods safely."""

    plan = _path_plan((1, 6))
    state = TreePeps.rand(plan, bond_dim=3, seed=19)
    first = TreePepsOptimizer(
        state,
        mode=compression_mode,
        chi=1,
        cutoff=0.0,
        compression_seed=23,
        track_infidelity=False,
        run=False,
    )
    second = first.copy()
    first.apply_gate(_cnot(), (0, 5))
    second.apply_gate(_cnot(), (0, 5))

    assert first.compression_mode == compression_mode
    assert first.state.max_bond() <= 1
    assert first.validate(check_canonical=True) is first
    np.testing.assert_allclose(
        first.state.to_statevector(), second.state.to_statevector()
    )


@pytest.mark.parametrize("compression_mode", ("sdc", "src", "zipup"))
def test_path_two_layer_and_fused_operator_application_agree(compression_mode):
    """Path compression can retain either the MPO-MPS or fused application."""

    plan = _path_plan((1, 5))
    state = TreePeps.rand(plan, bond_dim=2, seed=31)
    operator = TreeSubPepo.from_operator(_path_plan((1, 5)), _cnot(), support=(1, 4))
    expected = np.asarray(operator.to_dense().data).reshape(32, 32) @ state.to_statevector()

    two_layer = TreePepsOptimizer(
        state,
        chi=4,
        cutoff=0.0,
        compression_mode=compression_mode,
        compression_seed=29,
        compression_layout="two_layer",
        run=False,
        track_infidelity=False,
    )
    fused = two_layer.copy()
    fused.compression_layout = "fused"

    two_layer.apply(operator)
    fused.apply(operator)

    np.testing.assert_allclose(two_layer.to_dense(), expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(fused.to_dense(), expected, atol=1e-10, rtol=1e-10)
    assert two_layer.last_report["compression_layout"] == "two_layer"
    assert fused.last_report["compression_layout"] == "fused"
    assert two_layer.validate(check_canonical=True) is two_layer
    assert fused.validate(check_canonical=True) is fused


def test_path_two_layer_layout_rejects_branching_tree():
    """The two-layer Quimb path adapter never silently linearizes a tree."""

    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.from_plan(plan)
    operator = TreePepo.from_operator(plan, _cnot(), support=(0, 4))
    with pytest.raises(NotImplementedError, match="path TreePeps"):
        operator.apply_to(
            state,
            compress=True,
            max_bond=2,
            cutoff=0.0,
            compression_mode="sdc",
            compression_layout="two_layer",
        )


@pytest.mark.parametrize(
    ("mode", "fit_n_iter", "expected_block_size"),
    (("dmrg1", 3, 2), ("dmrg2", 1, 2), ("dmrg3", 1, 3)),
)
def test_tree_peps_dmrg_uses_tree_fit_engine(
    mode, fit_n_iter, expected_block_size
):
    """TreePEPS DMRG modes fit an exact PEPO target with cached environments."""

    plan = TreePepsPlan.from_shape((2, 3))
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan),
        mode=mode,
        chi=2,
        cutoff=0.0,
        fit_n_iter=fit_n_iter,
        fit_init_strategy="direct",
        fit_overlap_diagnostics=True,
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_gate(_cnot(), (0, 4))

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["backend"] == "tree_fit"
    assert diagnostics["block_size"] == expected_block_size
    assert diagnostics["requested_block_size"] == expected_block_size
    if mode == "dmrg1":
        assert diagnostics["block_size_trace"] == (2, 2, 1)
        assert diagnostics["adaptive_sweeps"] == 2
        assert diagnostics["one_site_refinement_sweeps"] == 1
    assert diagnostics["target_layout"] == "layered"
    assert diagnostics["cache"]["hits"] > 0
    assert diagnostics["local_fidelity"] > 1.0 - 1.0e-10
    assert optimizer.validate(check_canonical=True) is optimizer


@pytest.mark.parametrize("strategy", ("random", "random_expand"))
def test_tree_peps_dmrg_random_fit_guess_is_seeded(strategy):
    """TreePEPS DMRG random guesses remain disposable and reproducible."""

    plan = TreePepsPlan.from_shape((2, 3))
    kwargs = dict(
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
    first = TreePepsOptimizer(TreePeps.from_plan(plan), **kwargs)
    second = TreePepsOptimizer(TreePeps.from_plan(plan), **kwargs)
    first.apply_gate(_cnot(), (0, 4))
    second.apply_gate(_cnot(), (0, 4))

    diagnostics = first.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy"] == strategy
    assert diagnostics["random_initialization"] is True
    assert diagnostics["random_initialization_info"]["enabled"] is True
    np.testing.assert_allclose(first.to_dense(), second.to_dense())
    assert first.validate(check_canonical=True) is first


def test_tree_peps_dmrg_guess_src_uses_tree_pepo_warm_start():
    """TreePEPS ``guess-src`` applies the disposable operator-state path."""

    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(TreePepsPlan.from_shape((2, 3))),
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=1,
        fit_init_strategy="guess-src",
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_gate(_cnot(), (0, 4))

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy"] == "guess_src"
    assert diagnostics["guess_backend"] == "tree_pepo"
    assert diagnostics["guess_used"] is True
    assert optimizer.validate(check_canonical=True) is optimizer


def test_tree_peps_generic_dmrg_warmup_then_refinement():
    """Generic TreePEPS DMRG uses the MPS-style two-site handoff."""

    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(TreePepsPlan.from_shape((2, 3))),
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=4,
        fit_adaptive_sweeps=2,
        fit_init_strategy="guess-src",
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_gate(_cnot(), (0, 4))

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["block_size_trace"] == (2, 2, 1, 1)
    assert diagnostics["adaptive_sweeps"] == 2
    assert diagnostics["one_site_refinement_sweeps"] == 2


def test_tree_peps_dmrg_fits_explicit_sub_treepepo():
    """The DMRG backend also handles an already-factorized TreeSubPepo."""

    plan = TreePepsPlan.from_shape((2, 3))
    operator = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 4))
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan),
        mode="dmrg",
        chi=2,
        cutoff=0.0,
        fit_n_iter=1,
        fit_init_strategy="direct",
        fit_overlap_diagnostics=True,
        track_infidelity=False,
        run=False,
    )

    optimizer.apply_sub_treepepo(operator)

    assert optimizer.get_fit_diagnostics()["backend"] == "tree_fit"
    assert optimizer.get_fit_diagnostics()["local_fidelity"] > 1.0 - 1.0e-10
    assert optimizer.validate(check_canonical=True) is optimizer


def test_tree_fit_center_motion_avoids_full_canonical_validation():
    """Trusted local updates do not recheck every outside isometry per sweep."""

    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.from_plan(plan)
    fit = TreeFIT(state.copy(), state, max_bond=2, cutoffs=0.0)
    original_validate = fit.p.validate
    calls = []

    def count_validate(*args, **kwargs):
        calls.append(kwargs.get("check_canonical", False))
        return original_validate(*args, **kwargs)

    fit.p.validate = count_validate
    fit.run_eff(n_iter=2, block_size=2)

    assert calls == []
    original_validate(check_canonical=True)


def test_direct_optimizer_routes_over_the_tree_geodesic_exactly():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.rand(plan, bond_dim=2, seed=12)
    state_dense = np.asarray(state.to_dense().data).reshape(-1)
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 4))

    optimizer = TreePepsOptimizer(state, mode="direct", chi=None, cutoff=0.0)
    assert optimizer.apply_gate(_cnot(), (0, 4)) is optimizer

    expected = np.asarray(subop.to_dense().data).reshape(64, 64) @ state_dense
    actual = np.asarray(optimizer.state.to_dense().data).reshape(-1)
    assert np.allclose(actual, expected)
    assert optimizer.last_report["path"] == plan.path(0, 4)
    assert optimizer.last_report["span"] == tuple(sorted(subop.span))
    assert optimizer.validate(check_canonical=True) is optimizer


def test_sub_treepepo_optimizer_fuses_then_compresses_only_the_span():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.rand(plan, bond_dim=2, seed=13)
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 4))
    info_c = {}

    optimizer = TreePepsOptimizer(
        state,
        mode="sub_treepepo",
        chi=1,
        info_c=info_c,
    )
    optimizer.apply(subop)

    assert optimizer.last_report["mode"] == "sub_treepepo"
    assert optimizer.last_report["truncated"]
    assert optimizer.validate(check_canonical=True) is optimizer
    assert optimizer.center in subop.span
    assert info_c["cur_orthog"] == (optimizer.center, optimizer.center)

    for edge in plan.tree_edges:
        if edge[0] in subop.span and edge[1] in subop.span:
            assert optimizer.state.node_tensor(edge[0]).ind_size(optimizer.state.bond(*edge)) <= 1


def test_optimizer_modes_and_plan_validation_are_explicit():
    plan = _path_plan((2, 2))
    state = TreePeps.from_plan(plan)
    subop = TreeSubPepo.from_operator(plan, np.eye(4), support=(0, 3))

    with pytest.raises(TypeError, match="requires a TreeSubPepo"):
        TreePepsOptimizer(state, mode="sub_treepepo").apply(np.eye(4), (0, 3))

    other = TreePeps.from_plan(_path_plan((2, 2), order="row-major"))
    with pytest.raises(ValueError, match="same tree plan"):
        TreePepsOptimizer(other, plan=plan)

    optimizer = TreePepsOptimizer(state, mode="auto", chi=None, cutoff=0.0)
    optimizer.apply(subop)
    assert optimizer.last_report["mode"] == "sub_treepepo"

    identity_optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    identity_optimizer.apply(TreePepo.identity(plan))
    assert identity_optimizer.validate(check_canonical=True) is identity_optimizer


def test_direct_optimizer_promotes_real_state_for_complex_gates():
    plan = _path_plan((1, 2))
    state = TreePeps.from_plan(plan, dtype=float)
    optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    optimizer.apply_gate(np.diag([1.0, 1.0j]), 0)

    assert np.issubdtype(optimizer.state.node_tensor(0).data.dtype, np.complexfloating)


def test_direct_optimizer_converts_factorized_operator_to_torch_backend():
    torch = pytest.importorskip("torch")
    from pepsy.backends import backend_torch

    plan = TreePepsPlan.from_shape((3, 3), order="row-major", tree_order="row-major")
    to_torch = backend_torch(dtype=torch.complex64, device="cpu")
    state = TreePeps.from_plan(plan, dtype=np.complex64)
    state.apply_to_arrays(to_torch)
    gate = to_torch(np.diag([1.0, 1.0j]).astype(np.complex64))

    optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0, run=False)
    optimizer.apply_gate(gate, 0)

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in optimizer.state.tensors)
    assert optimizer.validate(check_canonical=True) is optimizer


def test_dm_compression_uses_the_fused_tree_pepo_state_network():
    plan = _path_plan((2, 2))
    state = TreePeps.rand(plan, bond_dim=2, seed=21)
    gate = _cnot()

    direct = TreePepsOptimizer(
        state, chi=1, cutoff=0.0, compression_mode="direct"
    )
    dm = TreePepsOptimizer(
        state, chi=1, cutoff=0.0, compression_mode="dm"
    )
    direct.apply_gate(gate, (0, 3))
    dm.apply_gate(gate, (0, 3))

    np.testing.assert_allclose(
        np.asarray(direct.state.to_dense().data).reshape(-1),
        np.asarray(dm.state.to_dense().data).reshape(-1),
        atol=1e-10,
        rtol=1e-10,
    )
    assert dm.last_report["compression_mode"] == "dm"
    assert dm.validate(check_canonical=True) is dm


def test_dm_mode_is_a_shorthand_for_direct_tree_pepo_routing():
    plan = _path_plan((2, 2))
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan), mode="dm", chi=1, cutoff=0.0
    )

    assert optimizer.mode == "direct"
    assert optimizer.compression_mode == "dm"


def test_dm_compression_is_used_for_tree_sub_treepepo_updates():
    plan = _path_plan((2, 2))
    operator = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 3))
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=22),
        mode="sub_treepepo",
        compression_mode="dm",
        chi=1,
        cutoff=0.0,
    )

    optimizer.apply_sub_treepepo(operator)

    assert optimizer.last_report["mode"] == "sub_treepepo"
    assert optimizer.last_report["compression_mode"] == "dm"
    assert optimizer.validate(check_canonical=True) is optimizer


def test_optimizer_owns_a_persistent_stream_and_supports_replacement():
    plan = _path_plan((1, 2))
    state = TreePeps.from_plan(plan)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)

    optimizer = TreePepsOptimizer(
        state,
        gates=[(x, 0)],
        run=False,
        chi=None,
        cutoff=0.0,
    )
    assert len(optimizer.gate_stream) == 1
    assert optimizer.history == []

    optimizer.add_gates([TreePepsOptimizer.gate_event(z, 1)])
    assert len(optimizer.gates) == 2
    optimizer.run()

    reference = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    reference.run([(x, 0), (z, 1)])
    np.testing.assert_allclose(
        np.asarray(optimizer.state.to_dense().data),
        np.asarray(reference.state.to_dense().data),
    )

    replacement = TreePeps.from_plan(plan)
    optimizer.set_state(replacement)
    assert optimizer.state is not replacement
    assert len(optimizer.gate_stream) == 2
    assert optimizer.history == []

    with pytest.raises(ValueError, match="same tree plan"):
        optimizer.set_state(TreePeps.from_plan(_path_plan((1, 3))))


@pytest.mark.parametrize(
    ("requested_mode", "expected_mode", "expected_compression"),
    [
        ("direct", "direct", "direct"),
        ("auto", "auto", "direct"),
        ("dm", "direct", "dm"),
        ("sdc", "direct", "sdc"),
        ("src", "direct", "src"),
        ("zipup", "direct", "zipup"),
        ("sub_treepepsmpo", "sub_treepepo", "direct"),
        ("dmrg", "dmrg", "direct"),
    ],
)
def test_run_persists_all_tree_peps_modes(
    requested_mode, expected_mode, expected_compression
):
    """run(mode=...) stores the canonical route and compression selection."""

    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(_path_plan((1, 2))),
        mode="direct",
        compression_mode="direct",
        run=False,
    )

    optimizer.run(mode=requested_mode)

    assert optimizer.mode == expected_mode
    assert optimizer.compression_mode == expected_compression


def test_run_persists_an_explicit_compression_override():
    """A compression-only run override remains active for later replays."""

    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(_path_plan((1, 2))),
        mode="direct",
        run=False,
    )

    optimizer.run(compression_mode="sdc")

    assert optimizer.mode == "direct"
    assert optimizer.compression_mode == "sdc"


def test_run_mode_shorthand_applies_to_explicit_sub_treepepo():
    """Shorthand compression reaches explicit PEPO stream events."""

    plan = _path_plan((1, 3))
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 2))
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan),
        gates=[TreePepsOptimizer.sub_treepepo_event(subop)],
        mode="direct",
        compression_mode="direct",
        chi=1,
        cutoff=0.0,
        track_infidelity=False,
        run=False,
    )

    optimizer.run(mode="sdc")

    assert optimizer.mode == "direct"
    assert optimizer.compression_mode == "sdc"
    assert optimizer.last_report["compression_mode"] == "sdc"


def test_optimizer_stream_event_forms_and_common_aliases():
    plan = _path_plan((1, 3))
    state = TreePeps.from_plan(plan)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = TreePepo.identity(plan)
    subop = TreeSubPepo.from_operator(plan, np.eye(4), support=(0, 2))

    optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0)
    optimizer.apply_1q(x, 0)
    optimizer.apply_2q(np.eye(4), 0, 2)
    optimizer.apply_multi_site(np.eye(8), 0, 1, 2)
    optimizer.apply_pepo(identity)
    assert optimizer.validate(check_canonical=True) is optimizer

    queued = TreePepsOptimizer(
        state,
        gates=[
            {"kind": "gate", "gate": x, "where": 0},
            TreePepsOptimizer.tree_pepo_event(identity),
            TreePepsOptimizer.sub_treepepo_event(subop),
        ],
        run=False,
        chi=None,
        cutoff=0.0,
    )
    assert [entry[0] for entry in queued.gate_stream] == [
        "gate",
        "tree_pepo",
        "sub_treepepo",
    ]
    queued.run()
    assert queued.validate(check_canonical=True) is queued
    copied = queued.copy()
    assert len(copied.gate_stream) == len(queued.gate_stream)
    assert copied.gate_stream[0][1] is queued.gate_stream[0][1]


def test_peps_mps_style_sub_treepepsmpo_aliases_share_pepo_route():
    """PEPS MPO-style names resolve to the canonical TreeSubPepo API."""
    plan = _path_plan((1, 3))
    state = TreePeps.from_plan(plan)
    subop = TreeSubPepo.from_operator(plan, _cnot(), support=(0, 2))
    optimizer = TreePepsOptimizer(state, chi=None, cutoff=0.0, run=False)

    assert optimizer.apply_sub_treepepsmpo(subop) is optimizer
    assert optimizer.last_report["mode"] == "sub_treepepo"

    queued = TreePepsOptimizer(
        state,
        gates=[
            ("sub_treepepsmpo", subop),
            ("tree_pepsmpo", TreePepo.identity(plan)),
        ],
        chi=None,
        cutoff=0.0,
        run=False,
    )
    assert [entry[0] for entry in queued.gate_stream] == [
        "sub_treepepo",
        "tree_pepo",
    ]
    queued.run()
    assert queued.validate(check_canonical=True) is queued


def test_optimizer_rejects_queued_backend_mismatches_atomically():
    torch = pytest.importorskip("torch")
    plan = _path_plan((1, 2))
    optimizer = TreePepsOptimizer(TreePeps.from_plan(plan), run=False)

    with pytest.raises(TypeError, match="backend/device"):
        optimizer.set_gates([(torch.eye(2), 0)])
    assert optimizer.gate_stream == ()


def test_optimizer_matches_ttn_state_aliases_and_readout_helpers():
    plan = _path_plan((1, 3))
    optimizer = TreePepsOptimizer(TreePeps.rand(plan, seed=31), run=False)

    assert optimizer.p is optimizer.state
    assert optimizer.tn is optimizer.state
    assert optimizer.orthogonality_center == optimizer.center
    assert optimizer.qubits == [0, 1, 2]
    assert optimizer.logical_order == optimizer.qubits
    assert optimizer.logical_site(1) == 1
    assert optimizer.position((0, 2)) == 2
    assert optimizer.to_dense().shape == (8,)
    assert np.allclose(optimizer.norm(), optimizer.state.norm())
    assert optimizer.bond_report()["n_bonds"] == 2

    optimizer.shift_orthogonality_center(2)
    assert optimizer.is_canonical_form()
    assert optimizer.validate_isometry_metadata() is optimizer
    assert optimizer.sync_canonicalization(1) == 1
    assert optimizer.center == 1


def test_optimizer_estimate_and_preflight_report_conservative_tree_bonds():
    plan = _path_plan((1, 3))
    optimizer = TreePepsOptimizer(TreePeps.from_plan(plan), chi=1, run=False)

    estimate = optimizer.estimate_bonds([(_cnot(), (0, 2))])
    assert estimate["max_bond"] >= 2
    assert estimate["requires_truncation"]
    report = optimizer.preflight(
        [(_cnot(), (0, 2))],
        max_bond=1,
        raise_on_error=False,
    )
    assert not report["ok"]
    assert report["violations"]


def test_optimizer_truncation_report_and_normalize_are_available():
    plan = _path_plan((1, 2))
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=37),
        chi=1,
        cutoff=0.0,
    )
    optimizer.apply_gate(_cnot(), (0, 1))

    report = optimizer.truncation_report()
    assert report["n_events"] == 1
    assert report["n_truncated"] == 1
    old_norm = optimizer.normalize()
    assert old_norm > 0.0
    assert np.allclose(optimizer.norm(), 1.0)


def test_optimizer_canonicalization_and_info_c_are_state_owned():
    plan = _path_plan((2, 2), tree_order="row-major")
    info_c = {}
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=41),
        info_c=info_c,
        run=False,
    )

    assert optimizer.canonicalize(0) is optimizer
    assert info_c["cur_orthog"] == (0, 0)
    optimizer.center = 3
    assert info_c["cur_orthog"] == (3, 3)
    optimizer.canonize_subtree((0, 3), span=True)
    assert optimizer.is_subtree_canonical_form((0, 3), span=True)
    assert info_c["canonical_region"] == optimizer.canonical_region
    optimizer.canonicalize_(1)
    assert optimizer.center == 1


def test_optimizer_compresses_only_the_requested_span_and_reports_scope():
    plan = TreePepsPlan.from_shape((3, 3), tree_order="row-major")
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=43),
        chi=1,
        cutoff=0.0,
        track_bond_diagnostics=True,
        run=False,
    )
    support = (0, 8)
    span = plan.subtree_span(support)
    before = optimizer.state.bond_sizes()
    optimizer.compress(support, span=True)
    after = optimizer.state.bond_sizes()

    for edge in plan.tree_edges:
        if not (edge[0] in span and edge[1] in span):
            assert after[edge] == before[edge]
    assert optimizer.canonical_region == frozenset({optimizer.center})

    optimizer.apply_gate(_cnot(), (0, 8))
    report = optimizer.bond_diagnostic_report()
    assert report["enabled"]
    assert report["max_transient_bond"] is not None
    assert optimizer.last_report["compression_scope"] == "span"
    assert optimizer.last_report["touched_edges"]


def test_tree_peps_optimizer_batches_validation_across_span_edges(monkeypatch):
    """Localized optimizer compression validates after, not during, its sweep."""
    plan = TreePepsPlan.from_shape((2, 3), tree_order="row-major")
    events = []
    original_validate = TreePeps.validate
    original_edge = TreePeps._compress_edge_inplace

    def capture_validate(self, *args, **kwargs):
        events.append(("validate", kwargs.get("check_canonical", False)))
        return original_validate(self, *args, **kwargs)

    def capture_edge(self, *args, **kwargs):
        events.append(("edge", kwargs.get("_validate", True)))
        return original_edge(self, *args, **kwargs)

    monkeypatch.setattr(TreePeps, "validate", capture_validate)
    monkeypatch.setattr(TreePeps, "_compress_edge_inplace", capture_edge)
    optimizer = TreePepsOptimizer(
        TreePeps.rand(plan, bond_dim=2, seed=47),
        chi=1,
        cutoff=0.0,
        track_infidelity=False,
        run=False,
    )
    events.clear()
    optimizer.apply_gate(_cnot(), (0, 4))

    edge_positions = [i for i, event in enumerate(events) if event[0] == "edge"]
    assert edge_positions
    assert all(events[i][1] is False for i in edge_positions)
    for first, second in zip(edge_positions, edge_positions[1:]):
        assert not any(
            event[0] == "validate" for event in events[first + 1 : second]
        )
    assert any(event == ("validate", True) for event in events)
    assert optimizer.validate(check_canonical=True) is optimizer


def test_optimizer_run_supports_norm_controls_and_profile_report():
    plan = _path_plan((1, 2))
    scale = np.diag([2.0, 1.0])
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan),
        gates=[(scale, 0)],
        run=False,
        chi=None,
        cutoff=0.0,
        profile=True,
    )
    optimizer.run(normalize_every=True)
    assert np.allclose(optimizer.norm(), 1.0)
    assert optimizer.get_normalizations()
    profile = optimizer.profile_report()
    assert profile["enabled"]
    assert profile["by_kind"]["update"]["count"] == 1


def test_optimizer_progress_bar_reports_fidelities_not_live_norm(monkeypatch):
    import tqdm as tqdm_module

    plan = _path_plan((1, 3))
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    postfixes = []
    descriptors = []

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            del args
            descriptors.append(kwargs["desc"])

        def set_postfix(self, postfix):
            postfixes.append(dict(postfix))

        def update(self, _count):
            pass

        def close(self):
            pass

    monkeypatch.setattr(tqdm_module, "tqdm", FakeProgress)
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(plan),
        gates=[(x, 0), (_cnot(), (0, 2))],
        run=False,
        chi=1,
        cutoff=0.0,
    )

    optimizer.run(progbar=True)

    assert descriptors == ["direct"]
    assert len(postfixes) == 2
    assert all("norm" not in postfix for postfix in postfixes)
    assert all("~F" in postfix and "F" not in postfix for postfix in postfixes)
    assert all("bnd" in postfix for postfix in postfixes)
    assert postfixes[-1]["2q"] == 1
    diagnostics = optimizer.norm_diagnostics()
    assert diagnostics["local_fidelity"] is not None
    assert diagnostics["cumulative_fidelity"] is not None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("direct", "direct"),
        ("dm", "dm"),
        ("sdc", "sdc"),
        ("src", "src"),
        ("zipup", "zipup"),
        ("dmrg", "dmrg"),
        ("dmrg1", "dmrg1"),
        ("dmrg2", "dmrg2"),
        ("dmrg3", "dmrg3"),
    ],
)
def test_optimizer_progress_bar_uses_mps_mode_names(monkeypatch, mode, expected):
    """TreePEPS replay bars expose the active MPS-compatible mode name."""

    import tqdm as tqdm_module

    descriptors = []

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            del args
            descriptors.append(kwargs["desc"])

        def set_postfix(self, _postfix):
            pass

        def update(self, _count):
            pass

        def close(self):
            pass

    monkeypatch.setattr(tqdm_module, "tqdm", FakeProgress)
    optimizer = TreePepsOptimizer(
        TreePeps.from_plan(_path_plan((1, 2))),
        gates=[(np.eye(2, dtype=complex), 0)],
        run=False,
        mode="direct",
        chi=2,
        cutoff=0.0,
    )
    optimizer.set_gates([(np.eye(2, dtype=complex), 0)])
    # Test the label selection independently of the compression implementation.
    monkeypatch.setattr(optimizer, "apply_gate", lambda *_args, **_kwargs: optimizer)

    optimizer.run(progbar=True, mode=mode)

    assert descriptors == [expected]


def test_optimizer_layout_preflight_and_convergence_helpers():
    plan = _path_plan((2, 2))
    layout = TreePepsOptimizer.find_tree_layout(
        plan,
        interactions=[(np.eye(4), (0, 3))],
        max_iter=0,
    )
    assert isinstance(layout, TreePepsPlan)
    optimizer = TreePepsOptimizer(TreePeps.from_plan(plan), run=False)
    preflight = optimizer.preflight(
        [(_cnot(), (0, 3))],
        max_intermediate_bond=1,
        raise_on_error=False,
    )
    assert not preflight["ok"]
    records = TreePepsOptimizer.convergence_sweep(
        [(_cnot(), (0, 3))],
        state=TreePeps.from_plan(plan),
        chi_values=(1, 2),
    )
    assert [record["chi"] for record in records] == [1, 2]
    assert records[-1]["fidelity"] is not None
