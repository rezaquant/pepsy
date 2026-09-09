"""Algorithm-level tests for complementary-environment tree compression."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import TreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.optimizers.tree.compression import successive_tree_compress


@pytest.mark.parametrize("method", ["src", "sdc"])
def test_path_reduces_to_quimb_successive_algorithm(monkeypatch, method):
    import quimb.tensor.tn1d.compress as qc
    import pepsy.optimizers.tree.compression as tc

    state = qtn.MPS_rand_state(5, 4, dtype="complex128", seed=812)
    local = {i: [state[i]] for i in range(5)}
    noise = []
    random_array = tc.backend_random_array

    def record(shape, **kwargs):
        data = random_array(shape, **kwargs)
        noise.append(data)
        return data

    monkeypatch.setattr(tc, "backend_random_array", record)
    tensors, _ = successive_tree_compress(
        local, [(i, i - 1) for i in range(4, 0, -1)], 0,
        method=method, max_bond=2, seed=71, cutoff=0,
    )
    if method == "src":
        counter = iter(noise)

        def same_noise(tn, *, Bix, inds, **kwargs):
            return [qtn.Tensor(next(counter), inds=(Bix, *inds))]

        monkeypatch.setattr(qc, "_src_get_local_noise_tensors", same_noise)
    reference = getattr(qc, f"tensor_network_1d_compress_{method}")(
        state, max_bond=2, cutoff=0,
    )
    actual = qtn.TensorNetwork(tensors.values()).to_dense(
        [f"k{i}" for i in range(5)]
    ).reshape(-1)
    np.testing.assert_allclose(actual, reference.to_dense().reshape(-1), atol=1e-10)


@pytest.mark.parametrize("method,backend,chi", [
    ("src", "numpy", 2), ("sdc", "numpy", 2),
    ("src", "torch", 16), ("sdc", "torch", 16),
    ("src", "jax", 2), ("sdc", "jax", 2),
    ("zipup", "numpy", 16), ("zipup", "torch", 2),
])
def test_layered_tree_compression_uses_real_algorithm(monkeypatch, method, backend, chi):
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2,
                               root_qubit=4)
    state = TreeTensorNetwork.rand(plan, D=3, seed=51, dtype="complex128")
    rng = np.random.default_rng(81)
    gate, _ = np.linalg.qr(rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32)))
    operator = TreeMPO.from_gate(plan, gate, tuple(range(5)))
    reference = operator.to_dense() @ state.to_dense().reshape(-1)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        convert = lambda a: torch.as_tensor(a)
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        convert = lambda a: jax.numpy.asarray(a, dtype="complex64")
    else:
        convert = np.asarray
    state.apply_to_arrays(convert)
    operator.apply_to_arrays(convert)
    opt = TreeOptimizer(None, state=state, tree=plan, mode=method, chi=chi,
                         cutoff=0, compression_seed=7, run=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("materialized-target routing/compression is not this algorithm")

    monkeypatch.setattr(opt, "_route_subtree_messages", forbidden)
    monkeypatch.setattr(opt, "_compress_subtree", forbidden)
    monkeypatch.setattr(opt.tn, "compress_edge_", forbidden)
    opt.apply_subtreempo(operator, track_norm=False)
    actual = opt.to_dense().reshape(-1)
    tol = 2e-4 if backend == "jax" else 1e-10
    assert opt.tn.is_canonical_form(tol=tol)
    assert opt.tn.max_bond() <= chi
    assert opt.tn.validate_isometry_metadata() is opt.tn
    # The returned hub is the orthogonal projection of the original target
    # onto the nested retained branch bases, also for a truncated result.
    np.testing.assert_allclose(np.vdot(actual, reference), np.vdot(actual, actual),
                               atol=tol, rtol=tol)
    if chi == 16:
        np.testing.assert_allclose(actual, reference, atol=tol)
    assert all(event["kind"] == method for event in opt.truncation_history)


@pytest.mark.parametrize("method", ["src", "sdc"])
def test_state_compress_uses_environment_algorithm(monkeypatch, method):
    plan = TreePlan.from_order(range(5), structure="balanced", top_arity=2)
    state = TreeTensorNetwork.rand(plan, D=3, seed=62, dtype="complex128")
    before = state.to_dense().reshape(-1)
    original = qtn.Tensor.split

    def no_randomized_local_svd(self, *args, **kwargs):
        assert kwargs.get("method") != "svd:rand"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "split", no_randomized_local_svd)
    state.compress(max_bond=16, cutoff=0, compression_mode=method, compression_seed=12)
    np.testing.assert_allclose(state.to_dense().reshape(-1), before, atol=1e-10)
    assert state.is_canonical_form()


@pytest.mark.parametrize("method", ["src", "sdc", "zipup"])
def test_partial_span_preserves_exterior_and_weak_branch(method):
    plan = TreePlan.from_order(range(7), structure="balanced", top_arity=3)
    state = TreeTensorNetwork.rand(plan, D=2, seed=11, dtype="complex64")
    opt = TreeOptimizer(None, state=state, tree=plan, mode=method,
                         chi=16, cutoff=0, compression_seed=19, run=False)
    theta = 1e-4
    gate = np.diag(np.exp(1j * theta * np.array([1, -1, -1, 1]))).astype("complex64")
    operator = TreeMPO.from_gate(plan, gate, (0, 1))
    before = opt.to_dense().reshape(-1)
    expected = operator.to_dense() @ before
    opt.apply_subtreempo(operator, track_norm=False)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, rtol=2e-5, atol=2e-5)
    assert opt.tn.is_canonical_form(tol=2e-5)


@pytest.mark.parametrize("method", ["src", "sdc"])
def test_zero_target_and_low_level_edge_are_finite(method):
    state = TreeTensorNetwork.from_order(range(4), dtype="complex128")
    a = next(u for u in state.plan.nodes() if u != state.plan.root)
    b = state.plan.node_path(a, state.plan.root)[1]
    state.compress_edge_(a, b, compression_mode=method, max_bond=2, cutoff=0)
    assert state.is_canonical_form()
    state.node_tensor(state.orthogonality_center).modify(
        data=state.node_tensor(state.orthogonality_center).data * 0
    )
    state.compress(max_bond=2, cutoff=0, compression_mode=method)
    assert np.isfinite(state.to_dense()).all()
    np.testing.assert_allclose(state.to_dense(), 0)


@pytest.mark.parametrize("method", ["src", "sdc"])
def test_native_successive_request_is_explicitly_rejected(method):
    import pepsy

    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(4, tree=plan, fermion=fermion,
                            occupations=((1, 1), (0, 0), (1, 1), (0, 0)))
    arrays = [t.data for t in state.tensors]
    center = state.orthogonality_center
    with pytest.raises(NotImplementedError, match="dense tensors"):
        state.compress(compression_mode=method, max_bond=2)
    assert state.orthogonality_center == center
    assert all(t.data is array for t, array in zip(state.tensors, arrays))
