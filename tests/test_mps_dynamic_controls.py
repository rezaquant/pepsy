"""Conditional solver settings and physical/classical cap boundaries."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py


def _product(bits):
    return qtn.MPS_computational_state(bits, dtype="complex128")


def test_coalesced_reset_uses_logical_site_after_permutation():
    opt = py.MpsOptimizer(_product("000"), [
        ("h", 0), ("cnot", 0, 2), ("reset", 2),
    ], chi=4, mode="perm")
    result = opt.run(shots=128, strategy="coalesced", seed=714)
    assert result.branches == 2
    assert sum(result.counts) == 128
    assert {np.argmax(abs(leaf.to_dense().ravel())) for leaf in result.optimizers} == {0, 4}


def test_coalesced_reset_does_not_discard_rare_entangled_outcome():
    probability = 2e-8  # Previously classified as a product by purity tolerance.
    state = qtn.MatrixProductState.from_dense(
        [np.sqrt(1 - probability), 0., 0., np.sqrt(probability)], [2, 2], cutoff=0.,
    )
    opt = py.MpsOptimizer(state, [("reset", 0)], chi=2, mode="direct")
    result = opt.run(shots=1_000_000_000, strategy="coalesced", seed=714)
    assert result.branches == 2
    assert sum(result.counts) == 1_000_000_000
    assert min(result.counts) < 100
    assert {np.argmax(abs(leaf.to_dense().ravel())) for leaf in result.optimizers} == {0, 1}


def test_small_positive_measurement_probability_is_not_impossible():
    probability = 1e-13
    state = qtn.MatrixProductState.from_dense(
        np.array([np.sqrt(1 - probability), np.sqrt(probability)]), [2],
    )
    opt = py.MpsOptimizer(state, [("measure", "Z", 0, -1)], chi=2, mode="direct")
    opt.run(cutoff=0.)
    assert 0 < opt.measurements[-1][3] < 1e-12
    assert opt.measurements[-1][3] == pytest.approx(probability, rel=1e-12, abs=0.)
    np.testing.assert_allclose(opt.to_dense().ravel(), [0., 1.], atol=1e-12)
    assert opt.norm_diagnostics()["cumulative_compression_infidelity"] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("stream", [
    [("h", 0), ("h", 1), ("measure", "Z", 0), ("measure", "Z", 1)],
    [("x", 0), ("x", 1), ("amplitude_damping", .5, 0), ("amplitude_damping", .5, 1)],
    [("leakage", .5, 0), ("leakage", .5, 1)],
    [("h", 0), ("h", 1), ("measure_leaked", 0), ("measure_leaked", 1)],
])
def test_coalesced_branch_cap_counts_all_parents(stream):
    opt = py.MpsOptimizer(_product("00"), stream, chi=4, mode="direct")
    with pytest.raises(RuntimeError, match="branch cap"):
        opt.run(shots=128, strategy="coalesced", seed=714, max_branches=2)


def test_coalesced_leakage_preserves_entangled_remote_branches():
    opt = py.MpsOptimizer(_product("00"), [
        ("h", 0), ("cnot", 0, 1), ("leakage", 1., 0),
        ("measure_leaked", 0),
    ], chi=4, mode="direct")
    result = opt.run(shots=128, strategy="coalesced", seed=714)
    assert result.branches == 2
    assert sum(result.counts) == 128
    assert {np.argmax(abs(leaf.to_dense().ravel())) for leaf in result.optimizers} == {0, 1}
    assert all(leaf.leakage_records[-1].measurement == 2 for leaf in result.leaves)
    with pytest.raises(RuntimeError, match="branch cap"):
        opt.run(shots=128, strategy="coalesced", seed=714, max_branches=1)


def test_leakage_hidden_reset_counts_toward_per_event_budget():
    opt = py.MpsOptimizer(_product("00"), [
        ("h", 0), ("cnot", 0, 1), ("leakage", .5, 0),
    ], chi=4, mode="direct")
    with pytest.raises(RuntimeError, match="per-event branch budget"):
        opt.run(shots=128, strategy="coalesced", seed=714, max_branch_factor=2)
    result = opt.run(shots=128, strategy="coalesced", seed=714, max_branch_factor=3)
    assert result.branches == 3 and sum(result.counts) == 128


def _dense_gate(vector, matrix, where):
    length = int(round(np.log2(vector.size)))
    order = tuple(where) + tuple(i for i in range(length) if i not in where)
    block = vector.reshape((2,) * length).transpose(order).reshape(2 ** len(where), -1)
    return (matrix @ block).reshape((2,) * length).transpose(np.argsort(order)).ravel()


def _assert_tracked_center(opt):
    if opt.mode == "exact":
        return
    lo, hi = opt.info_c["cur_orthog"]
    assert 0 <= lo <= hi < opt.p.L
    # Check the actual isometries, independently of the cached center label.
    for site in range(opt.p.L):
        if lo <= site <= hi:
            continue
        neighbor = site + 1 if site < lo else site - 1
        bond = opt.p.bond(site, neighbor)
        tensor = opt.p[site]
        array = np.asarray(tensor.transpose(
            *(ind for ind in tensor.inds if ind != bond), bond,
        ).data).reshape(-1, tensor.ind_size(bond))
        np.testing.assert_allclose(array.conj().T @ array, np.eye(array.shape[1]), atol=2e-11)


@pytest.mark.parametrize("mode", ["direct", "dmrg2", "perm", "exact"])
def test_cap_measure_reset_sequence_matches_dense_and_tracks_center(mode):
    vector = np.asarray(qu.rand_ket(32, seed=817)).ravel() * 2.3
    opt = py.MpsOptimizer(qtn.MatrixProductState.from_dense(vector, [2] * 5), chi=16, mode=mode)
    opt.normalize()  # Keep the represented scale while normalizing working data.
    gate = qu.rand_uni(4, seed=818)
    steps = [
        (gate, (0, 4)), ("measure", "XY", (4, 1), -1),
        ("cap", 2, [1, .3j], "right"), ("measure_reset", "Y", 3, -1),
        ("cap", 0, [1, 1], "left"), ("measure", "ZZ", (0, 2), 1),
        ("cap", 2, [1, 0], "right"), ("cap", 0, [1, 1], "left"),
        ("measure_reset", "X", 0, 1),
    ]
    for entry in steps:
        name = entry[0] if isinstance(entry[0], str) else "gate"
        probability = None
        if name == "gate":
            vector = _dense_gate(vector, entry[0], entry[1])
        elif name == "cap":
            length = int(round(np.log2(vector.size)))
            vector = np.tensordot(vector.reshape((2,) * length), entry[2], axes=(entry[1], 0)).ravel()
        else:
            axes, where, outcome = entry[1:4]
            where = (where,) if isinstance(where, int) else where
            matrix = np.asarray(qu.pauli(axes[0]))
            for axis in axes[1:]:
                matrix = np.kron(matrix, qu.pauli(axis))
            projected = .5 * (vector + outcome * _dense_gate(vector, matrix, where))
            probability = np.vdot(projected, projected).real / np.vdot(vector, vector).real
            vector = projected / np.linalg.norm(projected)
            if name == "measure_reset" and outcome < 0:
                vector = _dense_gate(vector, qu.pauli("Z" if axes == "X" else "X"), where)
        opt.set_gates([entry])
        opt.run(cutoff=0., fit_init_strategy="guess-direct")
        actual = np.asarray(opt.to_dense()).ravel()
        # Global phase has no physical effect, including Pauli reset flips.
        phase = np.vdot(vector, actual)
        np.testing.assert_allclose(actual, vector * phase / abs(phase), atol=2e-10)
        if probability is not None:
            assert opt.measurements[-1][3] == pytest.approx(probability, abs=2e-11)
        _assert_tracked_center(opt)
        if mode != "exact":
            assert opt.norm_diagnostics()["cumulative_compression_infidelity"] == pytest.approx(0., abs=2e-10)
    before = opt.to_dense().copy()
    opt.set_gates([("cap", 0, [1, 0])])
    with pytest.raises(ValueError, match="length-1"):
        opt.run()
    np.testing.assert_allclose(opt.to_dense(), before)


@pytest.mark.parametrize("mode", ["direct", "dmrg2"])
def test_scaled_multisite_measurement_does_not_report_false_loss(mode):
    state = qtn.MPS_computational_state("+++", dtype="complex128") * 10
    opt = py.MpsOptimizer(state, [("measure", "ZZ", (0, 2), 1)], chi=4, mode=mode)
    opt.normalize()
    opt.run(cutoff=0.)
    assert opt.norm_diagnostics()["cumulative_compression_infidelity"] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_kraus_probabilities_use_tracked_center_without_global_norms(backend, monkeypatch):
    from pepsy.optimizers.noise import _kraus_probabilities

    state = qtn.MPS_rand_state(4, bond_dim=3, dtype="complex128", seed=819)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(lambda x: torch.as_tensor(x.copy(), dtype=torch.complex128))
    elif backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        state.apply_to_arrays(lambda x: jnp.asarray(x, dtype=jnp.complex64))
    opt = py.MpsOptimizer(state, chi=4, mode="direct")
    vector = np.asarray(opt.to_dense()).ravel()
    excited = np.sum(abs(vector.reshape(2, 2, 2, 2)[:, :, 1, :]) ** 2) / np.vdot(vector, vector).real

    def forbidden(*args, **kwargs):
        raise AssertionError("canonical channel probabilities must not rebuild global environments or scan centers")

    with monkeypatch.context() as patch:
        patch.setattr(qtn.MatrixProductState, "norm", forbidden)
        patch.setattr(qtn.MatrixProductState, "compute_local_expectation", forbidden)
        patch.setattr(qtn.MatrixProductState, "calc_current_orthog_center", forbidden)
        probabilities = _kraus_probabilities(opt, py.TrajectoryChannel.amplitude_damping(.3), (2,))
    np.testing.assert_allclose(probabilities, [1 - .3 * excited, .3 * excited], atol=2e-7)
    assert opt.info_c["cur_orthog"] == (2, 2)
    np.testing.assert_allclose(opt.to_dense().ravel(), vector, atol=2e-7)


@pytest.mark.parametrize("mode", ["dmrg1", "dmrg2", "dmrg3"])
def test_conditional_gate_preserves_fit_policy_and_state(mode):
    gate = (qu.rand_uni(4, seed=17), (1, 6))
    measure = ("measure", "Z", 0, 1)
    options = dict(n_iter=8, fit_rtol=None, fit_init_strategy="guess-direct",
                   fit_sweep_sequence="LR", cutoff=0.0)
    direct = py.MpsOptimizer(_product("0000000"), [measure, gate], chi=8, mode=mode)
    conditional = py.MpsOptimizer(
        _product("0000000"), [measure, ("if", -1, 0, gate)], chi=8, mode=mode
    )
    for opt in (direct, conditional):
        opt.run(**options)
    diagnostics = conditional.get_fit_diagnostics()
    assert not diagnostics.get("fallback", False)
    assert diagnostics["iterations"] == 8
    assert diagnostics["guess_method"] == "direct"
    assert diagnostics["block_size"] == (3 if mode == "dmrg3" else 2)
    assert diagnostics == direct.get_fit_diagnostics()
    assert conditional.info_c == direct.info_c
    np.testing.assert_allclose(conditional.to_dense(), direct.to_dense(), atol=1e-12)


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_conditional_shots_preserve_dmrg_policy(strategy):
    gate = (qu.rand_uni(4, seed=17), (1, 6))
    opt = py.MpsOptimizer(
        _product("0000000"), [("measure", "Z", 0, 1), ("if", -1, 0, gate)],
        chi=8, mode="dmrg1",
    )
    result = opt.run(shots=2, strategy=strategy, seed=714, run_kwargs={
        "n_iter": 8, "fit_rtol": None, "fit_init_strategy": "guess-direct",
    })
    for leaf in result.optimizers:
        diagnostics = leaf.get_fit_diagnostics()
        assert diagnostics["iterations"] == 8
        assert diagnostics["guess_method"] == "direct"
        assert not diagnostics.get("fallback", False)


def test_false_conditional_does_not_resolve_a_capped_permutation_site():
    opt = py.MpsOptimizer(_product("000"), [
        ("cap", 2, [1, 0]), ("measure", "Z", 0, 1),
        ("if", -1, 1, ("x", 2)),
    ], chi=4, mode="perm")
    opt.run()
    assert opt.p.L == 2
    np.testing.assert_allclose(opt.to_dense().reshape(-1), [1, 0, 0, 0])


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_conditional_fit_and_cap_preserve_dense_backend(backend):
    if backend == "torch":
        library = pytest.importorskip("torch")

        def convert(array):
            return library.as_tensor(np.asarray(array).copy(), dtype=library.complex64)
    else:
        library = pytest.importorskip("jax.numpy")

        def convert(array):
            return library.asarray(array, dtype=library.complex64)

    state = _product("0000000")
    state.apply_to_arrays(convert)
    gate = (convert(qu.rand_uni(4, seed=17)), (1, 6))
    options = dict(n_iter=8, fit_rtol=None, fit_init_strategy="guess-direct")
    prefix = [("measure", "Z", 0, 1)]
    tail = [("cap", 0, [1, 0])]
    reference = py.MpsOptimizer(state, prefix + [gate] + tail, chi=8, mode="dmrg3")
    actual = py.MpsOptimizer(state, prefix + [("if", -1, 0, gate)] + tail,
                            chi=8, mode="dmrg3")
    for opt in (reference, actual):
        opt.run(**options)
    assert actual.backend == backend
    assert actual.p.L == 6 and actual.info_c == reference.info_c
    np.testing.assert_allclose(actual.to_dense(), reference.to_dense(), atol=2e-5)


@pytest.mark.parametrize(("strategy", "mode"), [
    ("independent", "direct"), ("coalesced", "direct"), ("coalesced", "perm"),
])
def test_caps_remap_leakage_and_remove_leaked_sites(strategy, mode):
    opt = py.MpsOptimizer(_product("0000"), [
        ("cnot", 0, 3),  # Perm mode moves a site before either structural cap.
        ("leakage", 1.0, 2), ("cap", 0, [1, 0]),
        ("x", 1), ("measure_leaked", 1),
        ("cap", 1, [1, 0]), ("x", 1), ("measure_leaked", 1),
    ], chi=4, mode=mode)
    result = opt.run(shots=2, strategy=strategy, seed=714)
    records = (result.raw.leakage_records if strategy == "independent"
               else tuple(leaf.leakage_records for leaf in result.raw.leaves))
    assert all([r.measurement for r in shot if r.kind == "measure_leaked"] == [2, 1]
               for shot in records)
    for leaf in result.optimizers:
        assert leaf.p.L == 2
        assert leaf.mps_length_diagnostics()["length_history"] == (4, 3, 2)
        np.testing.assert_allclose(leaf.to_dense().reshape(-1), [0, 1, 0, 0])


@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_nested_conditional_cap_remaps_only_selected_leakage_branch(strategy):
    opt = py.MpsOptimizer(_product("0000"), [
        ("h", 0), ("measure", "Z", 0), ("leakage", 1.0, 3),
        ("if", -1, 1, ("if", -1, 1, ("cap", 1, [1, 0]))),
        ("measure_leaked", 2),
    ], chi=4, mode="direct")
    result = opt.run(shots=16, strategy=strategy, seed=714)
    records = (result.raw.leakage_records if strategy == "independent"
               else tuple(leaf.leakage_records for leaf in result.raw.leaves))
    assert {leaf.p.L for leaf in result.optimizers} == {3, 4}
    for leaf, shot in zip(result.optimizers, records):
        assert shot[-1].measurement == (2 if leaf.p.L == 3 else 0)
        assert len(leaf.cap_history) == (1 if leaf.p.L == 3 else 0)


@pytest.mark.parametrize("mode", ["direct", "dmrg2"])
def test_cap_physical_norm_change_is_not_compression_loss(mode):
    opt = py.MpsOptimizer(_product("0000"), [
        ("h", 0), ("cnot", 0, 1), ("cap", 0, [1, 0]), ("cnot", 0, 2),
    ], chi=8, mode=mode)
    opt.run(cutoff=0.0, stabilize_unitary=False)
    np.testing.assert_allclose(np.linalg.norm(opt.to_dense()), np.sqrt(0.5), atol=1e-12)
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0.0, abs=1e-12)
    assert len(opt.cap_history) == 1
    assert all(event["local_infidelity"] == pytest.approx(0.0, abs=1e-12)
               for event in opt.get_norm_events())


def test_cap_rebases_norm_without_erasing_prior_compression_loss():
    opt = py.MpsOptimizer(_product("00000"), [
        ("h", 0), ("cnot", 0, 3),
    ], chi=1, mode="direct")
    opt.run(cutoff=0.0, stabilize_unitary=False)
    before = opt.norm_diagnostics()["infidelity"]
    assert before == pytest.approx(0.5, abs=1e-12)
    opt.set_gates([("cap", 4, [2, 0]), ("cnot", 0, 2)])
    with pytest.warns(RuntimeWarning, match="finite_check"):
        opt.run(cutoff=0.0, stabilize_unitary=False, finite_check=True)
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(before, abs=1e-12)


def test_variable_length_terminal_sampling_preserves_rows_and_probabilities():
    opt = py.MpsOptimizer(_product("000"), [
        ("h", 0), ("measure", "Z", 0), ("if", -1, 1, ("cap", 2, [1, 0])),
    ], chi=4, mode="direct")
    result = opt.run(shots=32, strategy="coalesced", seed=714)
    assert {o.p.L for o in result.optimizers} == {2, 3}
    plain = result.sample_bits(seed=715, shuffle=False)
    shuffled = result.sample_bits(seed=715)
    assert plain.configs.shape == (32, 3)
    for row, length, leaf_index in zip(plain.configs, plain.lengths, plain.leaf_indices):
        leaf = result.optimizers[leaf_index]
        bit = int(leaf.measurements[0][2] < 0)
        assert length == leaf.p.L
        np.testing.assert_array_equal(row[:length], [bit] + [0] * (length - 1))
        assert np.all(row[length:] == -1)
    np.testing.assert_allclose(plain.probs, 1.0)
    permutation = np.random.default_rng(715).permutation(32)
    for attribute in ("configs", "lengths", "leaf_indices", "probs"):
        np.testing.assert_array_equal(getattr(shuffled, attribute),
                                      getattr(plain, attribute)[permutation])
    # Uniform old-style construction still supplies useful lengths.
    uniform = py.CoalescedSampleResult(np.zeros((2, 3), dtype=np.int8), np.zeros(2))
    np.testing.assert_array_equal(uniform.lengths, [3, 3])
    empty = py.CoalescedTrajectoryResult(())
    samples = py.sample_coalesced_bits(empty)
    assert samples.configs.shape == (0, 0) and samples.lengths.shape == (0,)


def test_conditional_measurement_splits_coalesced_leaves_and_keeps_budget():
    opt = py.MpsOptimizer(_product("00"), [
        ("h", 0), ("h", 1), ("measure", "Z", 0),
        ("if", -1, 0, ("measure", "Z", 1)),
    ], chi=4, mode="direct")
    result = opt.run(shots=128, strategy="coalesced", seed=714, max_branches=3)
    assert result.branches == 3 and sum(result.counts) == 128
    with pytest.raises(RuntimeError, match="branch cap"):
        opt.run(shots=128, strategy="coalesced", seed=714, max_branches=2)


def test_measurement_handles_mixed_leakage_labels_after_conditional_cap():
    opt = py.MpsOptimizer(_product("0000"), [
        ("h", 0), ("measure", "Z", 0), ("leakage", 1.0, 3),
        ("if", -1, 1, ("cap", 1, [1, 0])), ("measure", "Z", 2),
    ], chi=4, mode="direct")
    result = opt.run(shots=16, strategy="coalesced", seed=714)
    assert {leaf.p.L for leaf in result.optimizers} == {3, 4}
    for leaf in result.optimizers:
        assert leaf.measurements[-1][2] == (-1 if leaf.p.L == 3 else 1)
