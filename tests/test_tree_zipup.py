"""Incremental tree zipup and lazy TreeFIT diagnostic contracts."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.fitting import TreeFIT
from pepsy.optimizers.tree import TreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
import pepsy


@pytest.mark.parametrize("backend", ("numpy", "torch"))
@pytest.mark.parametrize("dtype", ("complex64", "complex128"))
@pytest.mark.parametrize("chi", (2, None))
def test_zipup_incremental_tree_operator(backend, dtype, chi, monkeypatch):
    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=14)
    preparation = TreeOptimizer(None, state=state, run=False)
    preparation.normalize()
    state = preparation.tn
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda x: torch.as_tensor(x, dtype=getattr(torch, dtype))
    else:
        convert = lambda x: np.asarray(x, dtype=dtype)
    state.apply_to_arrays(convert)
    rng = np.random.default_rng(9)
    gate, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    gate = convert(gate)
    operator = TreeMPO.from_gate(plan, gate, (0, 2, 4))
    for network in operator.tree_networks:
        network.apply_to_arrays(convert)
    reference = TreeOptimizer(None, state=state.copy(), chi=None, cutoff=0., run=False)
    reference.apply_subtreempo(operator, track_norm=False)
    optimizer = TreeOptimizer(None, state=state.copy(), chi=chi, cutoff=0.,
                              mode="zipup", run=False)

    def no_final_compression(*args, **kwargs):
        raise AssertionError("zipup must truncate messages before the full target exists")

    monkeypatch.setattr(optimizer, "_compress_subtree", no_final_compression)
    optimizer.apply_subtreempo(operator, track_norm=False)
    tolerance = 2e-5 if dtype == "complex64" else 1e-10
    assert optimizer.tn.is_canonical_form(optimizer.center, tol=tolerance)
    if chi is None:
        np.testing.assert_allclose(optimizer.to_dense(), reference.to_dense(),
                                   atol=tolerance, rtol=tolerance)
    else:
        assert optimizer.tn.max_bond() <= chi
    assert optimizer.copy().mode == "zipup"


def test_tree_fit_lazy_diagnostics_never_double_target(monkeypatch):
    optimizer = TreeOptimizer(None, n=5, chi=2, mode="dmrg2", run=False,
                              track_infidelity=False)
    gate = np.diag([1., 1., 1., -1.]).astype(complex)
    target, _ = optimizer._build_tree_fit_target(gate, (0, 4))
    fit = TreeFIT(target, optimizer.tn, max_bond=2, cutoffs=0., copy_target=False)

    def no_norm(*args, **kwargs):
        raise AssertionError("layered target norm contraction")

    monkeypatch.setattr(qtn.TensorNetwork, "norm", no_norm)
    fit.run_eff(2, block_size=2)
    diagnostics = fit.fit_diagnostics()
    assert diagnostics["local_fidelity"] is None  # no invented target normalization
    assert diagnostics["local_norm"] == pytest.approx(1.)
    assert fit.fit_diagnostics(overlap=True)["local_fidelity"] == pytest.approx(1.)
    optimizer.apply_gate(gate, (0, 4))
    assert optimizer.get_fit_diagnostics()["local_fidelity"] == pytest.approx(1.)


@pytest.mark.parametrize("known_norm", (False, True))
def test_tree_fit_scaled_nonunitary_target_diagnostics(known_norm, monkeypatch):
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=31)
    optimizer = TreeOptimizer(None, state=state, run=False)
    gate = np.diag([0.2, 0.7, 1.3, 2.1]).astype(complex)
    target, _ = optimizer._build_tree_fit_target(gate, (0, 2))
    target.exponent = 30.
    exact = target.to_dense(tuple(f"k{q}" for q in range(3))).reshape(-1)
    norm = np.linalg.norm(exact)
    fit = TreeFIT(
        target, optimizer.tn, max_bond=2, cutoffs=0.,
        target_norm=(norm / 1e30, 30.) if known_norm else None,
    )
    fit.run_eff(2, block_size=2)

    def no_norm(*args, **kwargs):
        raise AssertionError("diagnostics contracted a doubled target")

    monkeypatch.setattr(qtn.TensorNetwork, "norm", no_norm)
    if known_norm:
        monkeypatch.setattr(fit, "_canonical_target_norm", no_norm)
    diagnostics = fit.fit_diagnostics(overlap=True)
    fitted = fit.p.to_dense().reshape(-1)
    np.testing.assert_allclose(fitted / norm, exact / norm, atol=1e-10)
    expected = abs(np.vdot(exact / norm, fitted / np.linalg.norm(fitted))) ** 2
    assert diagnostics["fit_overlap_error"] is None
    assert diagnostics["target_fidelity"] == pytest.approx(expected)
    assert diagnostics["local_fidelity"] == pytest.approx(1.)


def test_tree_fit_optional_norm_failure_preserves_fit(monkeypatch):
    optimizer = TreeOptimizer(None, n=3, chi=2, run=False)
    target, _ = optimizer._build_tree_fit_target(np.eye(4, dtype=complex), (0, 2))
    fit = TreeFIT(target, optimizer.tn, max_bond=2, cutoffs=0.)
    fit.run_eff(1, block_size=2)
    before = fit.p.to_dense().copy()

    def failed_norm(center):
        raise RuntimeError("target QR diagnostic failed")

    monkeypatch.setattr(fit, "_canonical_target_norm", failed_norm)
    diagnostics = fit.fit_diagnostics(overlap=True)
    assert diagnostics["fit_overlap_error"] == "target QR diagnostic failed"
    assert diagnostics["target_fidelity"] is None
    assert diagnostics["local_fidelity"] is None
    assert diagnostics["local_norm"] == pytest.approx(1.)
    np.testing.assert_array_equal(fit.p.to_dense(), before)


def test_tree_fit_finite_check_is_opt_in(monkeypatch):
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.from_plan(plan)
    fit = TreeFIT(state.copy(), state)
    calls = []
    monkeypatch.setattr(fit, "_check_finite", lambda region: calls.append(region))
    fit.run_eff(1)
    assert not calls
    fit.finite_check = True
    fit.run_eff(1)
    assert len(calls) == 1
    fit.p.node_tensor(fit.final_center_site).modify(
        data=fit.p.node_tensor(fit.final_center_site).data * np.nan,
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        TreeFIT._check_finite(fit, fit.nodes)


def test_zipup_run_override():
    optimizer = TreeOptimizer(None, n=2, run=False, fit_finite_check=True)
    optimizer.run(mode="zipup")
    assert optimizer.mode == "zipup"
    assert optimizer.copy().fit_finite_check
    with pytest.raises(ValueError, match="requires compression_mode"):
        optimizer.run(mode="zipup", compression_mode="dm")


def test_tree_fit_messages_resolve_live_fitted_bonds():
    plan = TreePlan.from_order(range(3), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=2, seed=43)
    fit = TreeFIT(state, state, max_bond=2, cutoffs=0.)
    # Native center movement can rename bonds after FIT prepares its target.
    fit.p.reindex_({ix: f"renamed_{ix}" for ix in fit.p.inner_inds()})
    fit.run_eff(1, block_size=2)
    np.testing.assert_allclose(fit.p.to_dense(), state.to_dense(), atol=1e-10)


@pytest.mark.parametrize("mode", ("dmrg1", "dmrg2", "dmrg3", "zipup"))
def test_subtree_operator_uses_lazy_tree_mpo(mode, monkeypatch):
    optimizer = TreeOptimizer(None, n=5, chi=2, mode=mode, run=False,
                              fit_n_iter=3, fit_init_strategy="guess-zipup")

    def no_materialized_target(*args, **kwargs):
        raise AssertionError("subtree application materialized the exact target")

    monkeypatch.setattr(TreeOptimizer, "_apply_subtree_operator_impl", no_materialized_target)
    monkeypatch.setattr(qtn.TensorNetwork, "norm", no_materialized_target)
    optimizer.apply_subtree_operator(np.eye(8, dtype=complex), (0, 2, 4))
    assert optimizer._active_update is None
    assert optimizer.tn.is_canonical_form(optimizer.center)
    if mode != "zipup":
        assert optimizer.get_fit_diagnostics()["target_layout"] == "layered"


@pytest.mark.parametrize("dtype", ("complex64", "complex128"))
@pytest.mark.parametrize("chi", (4, None))
def test_zipup_native_fermionic_matches_direct(dtype, chi):
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype=dtype)
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=((1, 0), (0, 1), (1, 0), (0, 1)), dtype=dtype)
    gate = fermion.hopping_gate(0.1, t=1., imaginary=False)
    stream = [(gate, (0, 3)), (gate, (0, 1)), (gate, (1, 2))]
    direct = TreeOptimizer(stream, state=state.copy(), chi=None, cutoff=0.)
    zipup = TreeOptimizer(stream, state=state.copy(), chi=chi, cutoff=0., mode="zipup")
    tolerance = 2e-5 if dtype == "complex64" else 1e-10
    if chi is None:
        assert float(pepsy.tensors.tn_fidelity(zipup.tn, direct.tn)) > 1 - tolerance
        assert zipup.norm() == pytest.approx(direct.norm(), rel=tolerance)
    else:
        assert zipup.max_bond() <= chi
        assert 0 < zipup.norm() <= direct.norm() + tolerance
    assert zipup.tn.is_canonical_form(zipup.center, tol=tolerance)


def test_zipup_does_not_install_empty_native_charge_paths():
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=((1, 0), (0, 1), (1, 0), (0, 1)))
    optimizer = TreeOptimizer(None, state=state.copy(), chi=2, cutoff=0.,
                              mode="zipup", run=False)
    with pytest.raises(ValueError, match="no compatible charge blocks"):
        optimizer.apply_gate(fermion.hopping_gate(.1, t=1., imaginary=False), (0, 3))
    assert float(pepsy.tensors.tn_fidelity(optimizer.tn, state)) > 1 - 1e-10
    assert optimizer._active_update is None
    assert all(t.data.blocks for t in optimizer.tn.tensors)


def test_tree_fit_rejects_unsupported_odd_parity_without_installing_state():
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(
        4, tree=plan, fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
    )
    optimizer = TreeOptimizer(None, state=state.copy(), chi=16, cutoff=0.,
                              mode="dmrg2", run=False)
    with pytest.raises(NotImplementedError, match="odd-parity fermionic"):
        optimizer.apply_gate(fermion.hopping_gate(.1, t=1., imaginary=False), (0, 3))
    assert float(pepsy.tensors.tn_fidelity(optimizer.tn, state)) > 1 - 1e-10
    assert optimizer._active_update is None
