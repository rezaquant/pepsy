"""Algorithm-level tests for complementary-environment tree compression."""

import weakref

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy import TreeMPO, TreeOptimizer, TreePlan, TreeTensorNetwork
from pepsy.optimizers.tree.compression import successive_tree_compress


def test_environment_plan_is_immutable_bounded_and_directional():
    from pepsy.optimizers.tree.compression import _successive_environment_plan as plan

    plan.cache_clear()
    order = ((2, 1), (1, 0))
    first = plan(order, 0)
    assert plan(order, 0) is first
    assert plan.cache_info().hits == 1
    assert first[1] == ((0, 1), (1, 2))
    assert plan(((0, 1), (1, 2)), 2)[1] == ((2, 1), (1, 0))
    with pytest.raises(TypeError):
        first[0][0] = (99,)
    with pytest.raises(TypeError):
        first[2][0, 1] = 999
    for i in range(150):
        plan(((i, i + 1),), i + 1)
    assert plan.cache_info().currsize == 128
    for invalid in (((0, 1), (0, 2)), ((0, 1), (2, 0)), ((0, 0),)):
        with pytest.raises(ValueError, match="order must"):
            plan(invalid, 2)


def test_src_environment_objects_are_reused_then_released(monkeypatch):
    import pepsy.optimizers.tree.compression as tc

    local = {i: [qtn.Tensor(np.eye(2, dtype=complex), inds=(f"k{i}", f"b{i}"))]
             for i in range(3)}
    local[3] = [qtn.Tensor(np.arange(8).reshape(2, 2, 2).astype(complex),
                           inds=("b0", "b1", "b2"))]
    order = ((0, 3), (1, 3), (2, 3))
    original_contract = qtn.tensor_contract
    original_split = qtn.Tensor.split
    original_modify = qtn.Tensor.modify
    built = []
    references = {}
    reads = {}
    projecting = False

    def contract(*ts, **kwargs):
        for tensor in ts:
            identity = id(tensor)
            if identity in references and references[identity]() is tensor.data:
                reads[identity] += 1
        result = original_contract(*ts, **kwargs)
        if not projecting and kwargs.get("output_inds") is not None:
            identity = id(result)
            references[identity] = weakref.ref(result.data)
            reads[identity] = 0
            built.append(identity)
        return result

    def modify(self, **kwargs):
        tracked = (id(self) in references
                   and references[id(self)]() is self.data)
        result = original_modify(self, **kwargs)
        if tracked:
            references[id(self)] = weakref.ref(self.data)
        return result

    def split(self, *args, **kwargs):
        nonlocal projecting
        projecting = True
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(qtn, "tensor_contract", contract)
    monkeypatch.setattr(qtn.Tensor, "split", split)
    monkeypatch.setattr(qtn.Tensor, "modify", modify)
    tc.successive_tree_compress(local, order, 3, method="src", max_bond=1, seed=51)
    # Each directed sketch is contracted once, with the same cached object
    # reused at branching consumers. No numerical environment survives return.
    assert len(built) == len(tc._successive_environment_plan(order, 3)[1]) == 6
    assert max(reads.values()) >= 2
    assert all(reference() is None for reference in references.values())


def test_cached_plan_cannot_reuse_stale_values_shapes_or_failed_work(monkeypatch):
    import pepsy.optimizers.tree.compression as tc

    tc._successive_environment_plan.cache_clear()
    order = ((3, 2), (2, 1), (1, 0))

    def run(state, seed, rank):
        tensors, _ = successive_tree_compress(
            {i: [state[i]] for i in range(4)}, order, 0,
            method="src", max_bond=rank, seed=seed,
        )
        return qtn.TensorNetwork(tensors.values()).to_dense([f"k{i}" for i in range(4)])

    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=12)
    first = run(state, 1, 1)
    # Same tensor and array identities: in-place edits must still rebuild all
    # numerical messages. Object identity is not a state version.
    state[0].data[...] *= 3
    np.testing.assert_allclose(run(state, 1, 1), 3 * first, atol=1e-11)
    changed = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=71)
    actual = run(changed, 93, 2)
    assert tc._successive_environment_plan.cache_info().misses == 1
    original = qtn.Tensor.split

    def fail(*args, **kwargs):
        raise RuntimeError("injected QR failure after environment construction")

    with monkeypatch.context() as patch:
        patch.setattr(qtn.Tensor, "split", fail)
        with pytest.raises(RuntimeError, match="injected QR failure"):
            run(state, 6, 2)
    assert qtn.Tensor.split is original
    np.testing.assert_allclose(run(changed, 93, 2), actual, atol=1e-11)
    tc._successive_environment_plan.cache_clear()
    np.testing.assert_allclose(run(changed, 93, 2), actual, atol=1e-11)


def test_branched_src_matches_dense_khatri_rao_qb_reference(monkeypatch):
    import pepsy.optimizers.tree.compression as tc

    rng = np.random.default_rng(51)
    target = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    local = {i: [qtn.Tensor(np.eye(2, dtype=complex), inds=(f"k{i}", f"b{i}"))]
             for i in range(3)}
    local[3] = [qtn.Tensor(target, inds=("b0", "b1", "b2"))]
    sketches = []
    original = tc.backend_random_array

    def record(shape, **kwargs):
        array = original(shape, **kwargs)
        sketches.append(array[0])
        return array

    monkeypatch.setattr(tc, "backend_random_array", record)
    result, _ = successive_tree_compress(
        local, [(0, 3), (1, 3), (2, 3)], 3, method="src", max_bond=1, seed=91,
    )
    # Independent dense QB calculation: contract the complementary physical
    # axes with their Khatri-Rao columns, then orthonormalize each leaf range.
    bases = []
    for site in range(3):
        others = [i for i in range(3) if i != site]
        matrix = np.moveaxis(target, site, 0).reshape(2, 4)
        omega = np.kron(sketches[others[0]], sketches[others[1]])
        q, _ = np.linalg.qr((matrix @ omega).reshape(2, 1))
        bases.append(q[:, 0])
    basis = np.kron(np.kron(bases[0], bases[1]), bases[2])
    expected = basis * np.vdot(basis, target.reshape(-1))
    actual = qtn.TensorNetwork(result.values()).to_dense(["k0", "k1", "k2"])
    np.testing.assert_allclose(actual.reshape(-1), expected, atol=1e-11)


@pytest.mark.parametrize("backend,reverse", [
    ("numpy", False), ("numpy", True), ("torch", False), ("jax", False),
])
def test_layered_src_matches_unmodified_quimb_seed_and_work(monkeypatch, backend, reverse):
    import quimb.tensor.tn1d.compress as qc
    import pepsy.optimizers.tree.compression as tc

    state = qtn.MPS_rand_state(6, 4, dtype="complex128", seed=12)
    operator = qtn.MPO_rand(6, 3, dtype="complex128", seed=13)
    target = operator.apply(state, contract=False)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        target.apply_to_arrays(lambda a: torch.as_tensor(a))
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        target.apply_to_arrays(lambda a: jax.numpy.asarray(a, dtype="complex64"))
    local = {i: list(target.select(target.site_tag(i)).tensors) for i in range(6)}
    sequence = list(range(6)) if reverse else list(reversed(range(6)))
    order = list(zip(sequence, sequence[1:]))
    original_contract = qtn.tensor_contract
    original_split = qtn.Tensor.split
    environment_calls = []
    qr_calls = []

    def contract(*ts, **kwargs):
        assert kwargs.get("drop_tags") is True
        if not qr_calls:
            environment_calls.append(kwargs.get("output_inds"))
        return original_contract(*ts, **kwargs)

    def split(self, *args, **kwargs):
        assert kwargs["method"] == "qr"
        assert kwargs["absorb"] == "lorthog"
        qr_calls.append(self.shape)
        return original_split(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(qtn, "tensor_contract", contract)
        patch.setattr(qtn.Tensor, "split", split)
        tensors, _ = successive_tree_compress(
            local, order, sequence[-1], method="src", max_bond=2, seed=43,
        )
    # Five fixed environments, then the first local sample before its QR.
    assert len(environment_calls) == 6
    assert len(qr_calls) == 5
    for site, tensor in tensors.items():
        assert set(tensor.tags) == {tag for t in local[site] for tag in t.tags}
    reference = qc.tensor_network_1d_compress_src(
        target, max_bond=2, cutoff=0, seed=43, sweep_reverse=reverse,
    )
    actual = qtn.TensorNetwork(tensors.values()).to_dense([f"k{i}" for i in range(6)])
    tolerance = 2e-4 if backend == "jax" else 1e-10
    np.testing.assert_allclose(tc.ar.to_numpy(actual).reshape(-1),
                               tc.ar.to_numpy(reference.to_dense()).reshape(-1),
                               rtol=tolerance, atol=tolerance)


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
    if method in {"src", "sdc"}:
        original_install = opt._install_routed_subtree
        original_move = opt._move_center
        installing = False

        def install(*args, **kwargs):
            nonlocal installing
            installing = True
            return original_install(*args, **kwargs)

        def move(*args, **kwargs):
            # Installation recovers the center using the newly proven Qs;
            # moving through old active tensors before projection is waste.
            assert installing
            return original_move(*args, **kwargs)

        monkeypatch.setattr(opt, "_install_routed_subtree", install)
        monkeypatch.setattr(opt, "_move_center", move)
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
