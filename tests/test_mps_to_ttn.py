"""State-preserving and explicitly capped MPS-to-tree conversion."""

import numpy as np
import pytest
import quimb.tensor as qtn
import autoray as ar

import pepsy as py
from pepsy.tensors import mps_to_ttn


def _vector(tn):
    return np.asarray(tn.to_dense([tn.site_ind(q) for q in range(tn.nsites)])).ravel()


@pytest.mark.parametrize("layout", ["chain_order", "interleaved", "physical_root"])
def test_exact_conversion_preserves_state_and_source(layout, monkeypatch):
    mps = qtn.MPS_rand_state(6, 3, seed=21, dtype="complex128")
    mps[0].modify(data=mps[0].data * (2 - 3j))
    mps.exponent = 1.5
    mps.site_ind_id = "physical_{}"
    mps.site_tag_id = "SITE{}"
    mps.add_tag("SOURCE")
    before = _vector(mps)
    arrays = [t.data.copy() for t in mps.tensors]
    metadata = [(t.inds, tuple(t.tags), t.left_inds) for t in mps.tensors]
    if layout == "physical_root":
        plan = py.TreePlan.from_order([0, 4, 1, 5, 3], root_qubit=2, structure="balanced")
    else:
        order = range(6) if layout == "chain_order" else [0, 2, 4, 1, 3, 5]
        plan = py.TreePlan.from_order(order, structure="balanced", top_arity=2)
    # Dense readout is a test oracle only, never a converter implementation.
    with monkeypatch.context() as patch:
        patch.setattr(qtn.MatrixProductState, "to_dense", lambda *a, **kw: pytest.fail("densified"))
        ttn = mps_to_ttn(mps, tree=plan)
    np.testing.assert_allclose(_vector(ttn), before, atol=2e-12, rtol=2e-12)
    assert ttn.exponent == mps.exponent
    assert ttn.orthogonality_center == plan.root
    assert ttn.validate(check_canonical=True) is ttn
    assert "SOURCE" in ttn.tags
    if layout == "interleaved":
        assert ttn.max_bond() > mps.max_bond()
    for tensor, array, meta in zip(mps.tensors, arrays, metadata):
        np.testing.assert_array_equal(tensor.data, array)
        assert (tensor.inds, tuple(tensor.tags), tensor.left_inds) == meta


def test_smaller_tree_cap_can_still_be_exact():
    # Two crossing Bell pairs have chain Schmidt rank four, but tree rank two
    # when each pair sits below its own parent.
    state = np.zeros((2,) * 4, dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            state[a, b, b, a] = 0.5
    mps = qtn.MatrixProductState.from_dense(state, dims=[2] * 4, cutoff=0.0)
    plan = py.TreePlan.from_order([0, 3, 1, 2], structure="balanced", top_arity=2)
    assert mps.max_bond() == 4
    ttn = mps_to_ttn(mps, tree=plan, chi=2)
    assert ttn.max_bond() <= 2
    np.testing.assert_allclose(_vector(ttn), state.ravel(), atol=1e-12)
    assert ttn.validate(check_canonical=True) is ttn


def test_exact_keeps_tiny_schmidt_value():
    mps = qtn.MatrixProductState([np.eye(2), np.diag([1.0, 1e-14])])
    ttn = mps_to_ttn(mps)
    np.testing.assert_allclose(_vector(ttn)[3], 1e-14, atol=1e-28, rtol=1e-13)


@pytest.mark.parametrize("scale", [1e-170, 1e170])
def test_capped_conversion_preserves_large_and_small_global_scale(scale):
    mps = qtn.MPS_rand_state(4, 2, seed=81, dtype="complex128")
    expected = _vector(mps_to_ttn(mps, chi=1))
    mps[0].modify(data=mps[0].data * scale)
    actual = _vector(mps_to_ttn(mps, chi=1)) / scale
    np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_finite_cap_does_not_pad_chain_aligned_tree_ranks():
    mps = qtn.MPS_rand_state(8, 2, seed=29, dtype="complex128")
    plan = py.TreePlan.from_order(range(8), structure="balanced", top_arity=2)
    ttn = mps_to_ttn(mps, tree=plan, chi=32)
    assert ttn.max_bond() <= 4
    np.testing.assert_allclose(_vector(ttn), _vector(mps), atol=2e-12)


@pytest.mark.parametrize("chi", [None, 2])
def test_nonbinary_tree_and_nonqubit_physical_dimensions(chi):
    # The converter operates on physical indices, not hard-coded qubit axes.
    # Check a ternary internal node and an interleaved tree partition as well.
    dims = (2, 3, 2, 3, 2, 2)
    rng = np.random.default_rng(80)
    state = rng.normal(size=dims)
    mps = qtn.MatrixProductState.from_dense(state, dims=dims, cutoff=0.0)
    plan = py.TreePlan.from_order(
        [0, 2, 4, 1, 3, 5], structure="balanced", max_arity=3, top_arity=2
    )
    ttn = mps_to_ttn(mps, tree=plan, chi=chi)
    assert any(len(children) == 3 for children in plan.children.values())
    assert tuple(ttn.ind_size(ttn.site_ind(q)) for q in range(6)) == dims
    assert all(t.data.dtype == np.float64 for t in ttn)
    if chi is None:
        np.testing.assert_allclose(_vector(ttn), state.ravel(), atol=3e-13)
    else:
        assert ttn.max_bond() <= chi
    ttn.validate(check_canonical=True)


@pytest.mark.parametrize("chi", [None, 1])
def test_result_does_not_share_single_site_input_storage(chi):
    mps = qtn.MPS_computational_state("1", dtype="complex128")
    ttn = mps_to_ttn(mps, chi=chi)
    ttn.tensors[0].data[...] = 0
    np.testing.assert_array_equal(_vector(mps), [0, 1])


@pytest.mark.parametrize("chi", [None, 2])
def test_cyclic_mps(chi):
    mps = qtn.MPS_rand_state(4, 2, cyclic=True, seed=14, dtype="complex128")
    ttn = mps_to_ttn(mps, chi=chi)
    if chi is None:
        np.testing.assert_allclose(_vector(ttn), _vector(mps), atol=2e-12)
    else:
        assert ttn.max_bond() <= chi
    ttn.validate(check_canonical=True)


def test_capped_projection_is_gauge_invariant_and_does_not_renormalize():
    mps = qtn.MPS_rand_state(6, 3, seed=92, dtype="complex128")
    mps[0].modify(data=mps[0].data * (2 + 1j))
    mps.exponent = -0.5
    gauged = mps.copy(deep=True)
    bond = gauged.bond(2, 3)
    gauge = np.array([[2, 0.3j, 0.2], [0.1, 0.5, 0.1j], [0.1j, -0.2, 1.4]])
    gauged[2].gate_(gauge, bond)
    gauged[3].gate_(np.linalg.inv(gauge).T, bond)
    np.testing.assert_allclose(_vector(mps), _vector(gauged), atol=1e-12)
    plan = py.TreePlan.from_order([0, 2, 4, 1, 3, 5], structure="balanced")
    a = mps_to_ttn(mps, tree=plan, chi=2)
    b = mps_to_ttn(gauged, tree=plan, chi=2)
    original, approximate = _vector(mps), _vector(a)
    assert a.max_bond() <= 2
    np.testing.assert_allclose(approximate, _vector(b), atol=2e-11)
    assert np.linalg.norm(original - approximate) > 1e-4
    assert np.linalg.norm(approximate) < np.linalg.norm(original)
    # Orthogonal projections onto nested tree subspaces leave this residual
    # orthogonal to the final approximation, including its original scale.
    np.testing.assert_allclose(np.vdot(approximate, original - approximate), 0, atol=1e-11)
    a.validate(check_canonical=True)


@pytest.mark.parametrize("chi", [None, 1])
def test_zero_state_and_single_site(chi):
    for n in (1, 4):
        mps = qtn.MPS_computational_state("0" * n, dtype="complex128")
        mps[0].modify(data=mps[0].data * 0)
        ttn = mps_to_ttn(mps, chi=chi)
        np.testing.assert_array_equal(_vector(ttn), np.zeros(2**n))
        ttn.validate(check_canonical=True)


@pytest.mark.parametrize("chi", [None, 2])
def test_torch_dtype_device_and_replay_handoff(chi):
    torch = pytest.importorskip("torch")
    mps = qtn.MPS_rand_state(5, 3, seed=4, dtype="complex64")
    mps.apply_to_arrays(lambda a: torch.as_tensor(a))
    plan = py.TreePlan.from_order([0, 2, 4, 1, 3], structure="balanced")
    ttn = py.mps_to_ttn(mps, tree=plan, chi=chi)
    assert all(t.data.dtype == torch.complex64 and t.data.device.type == "cpu" for t in ttn)
    gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    engine = py.TreeOptimizer([(gate, (0,))], tree=plan, state=ttn, chi=8, run=False)
    before = engine.to_dense().reshape((2,) * 5)
    engine.run()
    np.testing.assert_allclose(engine.to_dense().reshape((2,) * 5), before[::-1], atol=2e-6)


@pytest.mark.parametrize("chi", [0, -1, True, 1.5])
def test_invalid_cap(chi):
    with pytest.raises(ValueError, match="chi must"):
        mps_to_ttn(qtn.MPS_computational_state("00"), chi=chi)


def test_resource_guard_and_geometry_validation():
    mps = qtn.MPS_rand_state(6, 3, seed=8, dtype="complex128")
    with pytest.raises(MemoryError, match="has not changed chi"):
        mps_to_ttn(mps, max_intermediate_elements=1)
    with pytest.raises(ValueError, match="same physical site"):
        mps_to_ttn(mps, tree=py.TreePlan.from_order(range(5)))
    with pytest.raises(TypeError, match="TreePlan"):
        mps_to_ttn(mps, tree="balanced")
    with pytest.raises(TypeError, match="MatrixProductState"):
        mps_to_ttn(np.zeros(8))
    mps[1].modify(data=mps[1].data.astype("complex64"))
    with pytest.raises(TypeError, match="one backend, dtype, and device"):
        mps_to_ttn(mps)


def test_malformed_tree_fails_before_tensor_operations():
    mps = qtn.MPS_computational_state("0000")
    plan = py.TreePlan.from_order(range(4))
    plan.parent[plan.root] = plan.root
    with pytest.raises(ValueError, match="root with no parent"):
        mps_to_ttn(mps, tree=plan)

    plan = py.TreePlan.from_order(range(4))
    missing = max(plan.nodes()) + 1
    plan.children[plan.root] = (missing,)
    plan.parent[missing] = plan.root
    with pytest.raises(ValueError, match="parent/child"):
        mps_to_ttn(mps, tree=plan)


def test_finite_conversion_checks_namespace_capability(monkeypatch):
    mps = qtn.MPS_computational_state("00")
    monkeypatch.delattr(ar, "get_namespace")
    with pytest.raises(RuntimeError, match="get_namespace"):
        mps_to_ttn(mps, chi=1)


@pytest.mark.parametrize("chi", [None, 2])
def test_cuda_conversion_stays_on_device(chi):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    mps = qtn.MPS_rand_state(5, 2, seed=3, dtype="complex64")
    expected = _vector(mps)
    mps.apply_to_arrays(lambda a: torch.as_tensor(a, device=device))
    plan = py.TreePlan.from_order([0, 2, 4, 1, 3], structure="balanced")
    ttn = mps_to_ttn(mps, tree=plan, chi=chi)
    assert all(t.data.device == device and t.data.dtype == torch.complex64 for t in ttn)
    if chi is None:
        np.testing.assert_allclose(ttn.to_statevector(), expected, atol=2e-6)


@pytest.mark.parametrize("backend", ["cupy", "jax"])
@pytest.mark.parametrize("chi", [None, 2])
def test_other_array_backends(backend, chi):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
        convert = cp.asarray
    else:
        jax = pytest.importorskip("jax")
        convert = lambda a: jax.device_put(a, jax.devices("cpu")[0])
    mps = qtn.MPS_rand_state(5, 2, seed=18, dtype="complex64")
    plan = py.TreePlan.from_order([0, 2, 4, 1, 3], structure="balanced")
    expected = _vector(mps_to_ttn(mps, tree=plan, chi=chi))
    mps.apply_to_arrays(convert)
    ttn = mps_to_ttn(mps, tree=plan, chi=chi)
    assert all(ar.infer_backend(t.data) == backend for t in ttn)
    assert all(ar.get_dtype_name(t.data) == "complex64" for t in ttn)
    np.testing.assert_allclose(ttn.to_statevector(), expected, atol=3e-6)
