"""Tests for :mod:`pepsy.optimizers.mpo.optimizer`."""

import types

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py
import pepsy.optimizers.mpo.optimizer as mpo_optimizer_module


def test_mpo_optimizer_exported():
    """Public API should expose MpoOptimizer and the optimizers namespace."""
    assert "MpoOptimizer" in py.__all__
    assert "optimizers" in py.__all__
    assert py.MpoOptimizer is not None
    assert py.optimizers.mpo is not None


def test_mpo_backend_info_only_relaxes_same_backend_dtype_mixes():
    """MPO metadata must not hide a genuine mixed-backend state."""
    torch = pytest.importorskip("torch")
    mixed_backend = qtn.MPO_identity(3, dtype="complex128")
    mixed_backend[0].modify(
        data=torch.as_tensor(mixed_backend[0].data, dtype=torch.complex128)
    )

    with pytest.raises(TypeError, match="one compatible backend"):
        py.MpoOptimizer._backend_info_for(mixed_backend)  # pylint: disable=protected-access

    mixed_dtype = qtn.MPO_identity(3, dtype="complex128")
    mixed_dtype[0].modify(
        data=np.asarray(mixed_dtype[0].data, dtype=np.complex64)
    )
    assert py.MpoOptimizer._backend_info_for(mixed_dtype)["backend"] == "numpy"  # pylint: disable=protected-access


@pytest.mark.parametrize("where", [(0.9,), (True,), (0, 0)])
def test_mpo_channel_event_rejects_invalid_support_sites(where):
    """Channel replay must not truncate or alias malformed support indices."""
    error = TypeError if where != (0, 0) else ValueError
    with pytest.raises(error):
        py.MpoChannelEvent((np.eye(2),), where)


def test_mpo_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpoOptimizer`` mode validation."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


@pytest.mark.parametrize(
    ("mode", "expected_method"),
    [
        ("src", "src"),
        ("quimb-src", "src"),
        ("mpo-src", "src"),
        ("zipup", "zipup"),
        ("zipup-first", "zipup-first"),
        ("fit-zipup", "fit-zipup"),
        ("fit-projector", "fit-projector"),
        ("quimb-fit", "fit"),
    ],
)
def test_mpo_optimizer_accepts_mps_quimb_compression_mode_aliases(
    monkeypatch, mode, expected_method
):
    """MPO mode aliases dispatch through the selected Quimb compressor."""
    calls = []
    original = py.optimizers.mpo.optimizer.gate_nonlocal_opt

    def recording_gate_nonlocal_opt(*args, **kwargs):
        calls.append(kwargs["method"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        py.optimizers.mpo.optimizer,
        "gate_nonlocal_opt",
        recording_gate_nonlocal_opt,
    )
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=4,
        mode=mode,
    )

    out = opt.run(
        n_iter=1,
        cutoff=0.0,
        fidelity_samples=0,
        compression_seed=17,
    )

    assert out.max_bond() <= 4
    assert calls == [expected_method, expected_method]


def test_mpo_optimizer_submpo_method_overrides_mpo_mode():
    """The MPS-compatible sub-MPO method override works for MPO replay."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=4,
        mode="mpo",
    )

    out = opt.run(
        cutoff=0.0,
        fidelity_samples=0,
        submpo_method="src",
        compression_seed=31,
    )

    assert out.max_bond() <= 4


def test_mpo_optimizer_quimb_mode_preserves_dense_torch_backend():
    """Quimb MPO modes operate on already-prepared non-NumPy gate arrays."""
    torch = pytest.importorskip("torch")
    backend = py.backend_torch(dtype=torch.complex128, device="cpu")
    mpo = qtn.MPO_identity(5, dtype="complex128")
    mpo.apply_to_arrays(backend)
    gate = backend(qu.CNOT())

    out = py.MpoOptimizer(
        mpo,
        gates=[(gate, (0, 4))],
        chi=4,
        mode="src",
    ).run(cutoff=0.0, fidelity_samples=0, compression_seed=37)

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in out)
    assert out.max_bond() <= 4


def test_mpo_optimizer_to_backend_matches_live_mpo_backend():
    """The public MPO converter follows backend, dtype, and device metadata."""
    torch = pytest.importorskip("torch")
    backend = py.backend_torch(dtype=torch.complex64, device="cpu")
    mpo = qtn.MPO_identity(3, dtype="complex128")
    mpo.apply_to_arrays(backend)
    opt = py.MpoOptimizer(mpo, gates=[], chi=4, mode="svd")

    converted = opt.to_backend(np.eye(4, dtype=np.complex128))

    assert isinstance(converted, torch.Tensor)
    assert converted.dtype == torch.complex64
    assert str(converted.device) == "cpu"
    assert opt.to_backend(converted) is converted


def test_mpo_norm_events_compare_scaled_pairs_without_reconstructing_norms():
    """Extreme physical norms still produce a valid local fidelity ratio."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(3, dtype="complex128"), gates=[], chi=4, mode="svd"
    )

    event = opt._record_norm_event(
        "scaled_test",
        expected_norm=(1.0, 400.0),
        observed_norm=(1.0, 400.0),
        target_norm=(1.0, 400.0),
    )

    assert event["valid"] is True
    assert event["local_fidelity"] == pytest.approx(1.0)
    assert event["expected_norm_mantissa"] == pytest.approx(1.0)
    assert event["expected_norm_exponent"] == pytest.approx(400.0)
    assert np.isinf(event["expected_norm"])


def test_mpo_norm_events_keep_zero_observed_norm_as_zero_fidelity():
    """A zero retained norm is valid loss, not an invalid measurement."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(3, dtype="complex128"), gates=[], chi=4, mode="svd"
    )

    event = opt._record_norm_event(
        "zero_test",
        expected_norm=(1.0, 4.0),
        observed_norm=(0.0, 0.0),
    )

    assert event["valid"] is True
    assert event["local_fidelity"] == pytest.approx(0.0)
    assert event["local_infidelity"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("strategy", "expected_method", "expected_used"),
    [
        ("direct", None, False),
        ("guess-src", "src", True),
        ("guess_zipup", "zipup", True),
    ],
)
def test_mpo_dmrg_fit_initial_guess_strategies(
    strategy, expected_method, expected_used
):
    """MPO DMRG exposes the MPS-style disposable FIT guess policies."""
    opt = py.MpoOptimizer(
        qtn.MPO_rand(5, bond_dim=1, phys_dim=2, dtype="complex128", seed=812),
        gates=[(qu.CNOT(), (0, 4))],
        chi=4,
        mode="dmrg2",
    )

    out = opt.run(
        n_iter=2,
        cutoff=0.0,
        fidelity_samples=0,
        fit_init_strategy=strategy,
        fit_init_rand_strength=0.01,
        fit_init_seed=19,
    )

    diagnostics = opt.get_fit_diagnostics()
    assert out.max_bond() <= 4
    assert diagnostics["fit_init_strategy"] == (
        "guess_src" if strategy == "guess-src" else strategy
    )
    assert diagnostics["guess_method"] == expected_method
    assert diagnostics["guess_used"] is expected_used
    assert diagnostics["mpo_fit_guess_used"] is expected_used


def test_mpo_dmrg_random_expand_fit_initial_guess_is_seeded():
    """Random-expanded MPO FIT guesses grow only the disposable copy."""
    initial = qtn.MPO_rand(
        5, bond_dim=1, phys_dim=2, dtype="complex128", seed=813
    )
    opt = py.MpoOptimizer(
        initial,
        gates=[(qu.CNOT(), (0, 4))],
        chi=4,
        mode="dmrg2",
    )

    out = opt.run(
        n_iter=2,
        cutoff=0.0,
        fidelity_samples=0,
        fit_init_strategy="random_expand",
        fit_init_rand_strength=0.01,
        fit_init_seed=23,
    )

    diagnostics = opt.get_fit_diagnostics()
    assert out.max_bond() <= 4
    assert diagnostics["fit_init_strategy"] == "random_expand"
    assert diagnostics["guess_used"] is False
    assert diagnostics["random_initialization"]["enabled"] is True
    assert diagnostics["random_initialization"]["bonds"]
    assert initial.max_bond() == 1


@pytest.mark.parametrize(
    ("mode", "expected_block", "expected_warmup", "requested_warmup"),
    [
        ("dmrg1", 2, 2, 5),
        ("dmrg2", 2, 1, 1),
        ("dmrg3", 3, 2, 2),
    ],
)
def test_mpo_optimizer_dmrg_mode_aliases_select_fit_schedule(
    monkeypatch,
    mode,
    expected_block,
    expected_warmup,
    requested_warmup,
):
    """Named MPO DMRG modes select their native block schedule."""
    calls = []
    original_run_gate = py.FIT.run_gate

    def recording_run_gate(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", recording_run_gate)

    opt = py.MpoOptimizer(
        qtn.MPO_identity(5, dtype="complex128"),
        gates=[((qu.CNOT(), None), (0, 4))],
        chi=2,
        mode=mode,
    )
    out = opt.run(
        n_iter=3,
        progbar=False,
        cutoff=1.0e-12,
        fit_adaptive_sweeps=requested_warmup,
    )

    assert opt.mode == "dmrg"
    assert opt._dmrg_mode_alias == mode
    assert out.max_bond() <= 2
    assert len(calls) == 1
    assert calls[0]["block_size"] == expected_block
    assert calls[0]["adaptive_block_sweeps"] == expected_warmup
    assert calls[0]["adaptive_until_rank"] is False
    # The optional fast path is disabled by default, just as in
    # MpsOptimizer. The fixture is non-adjacent, so dmrg2's automatic
    # adjacent-pair exception does not apply.
    assert calls[0]["single_pair_fast_path"] is False


def test_mpo_dmrg2_enables_mps_compatible_adjacent_pair_fast_path(monkeypatch):
    """Named dmrg2 keeps the MPS adjacent-pair shortcut."""
    calls = []
    original_run_gate = py.FIT.run_gate

    def recording_run_gate(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", recording_run_gate)
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (1, 2))],
        chi=2,
        mode="dmrg2",
    )
    opt.run(n_iter=2, progbar=False, cutoff=0.0, fidelity_samples=0)

    assert calls
    assert calls[0]["single_pair_fast_path"] is True


def test_mpo_dmrg1_latches_one_site_phase_after_rank_saturation():
    """DMRG1 keeps later adjacent windows in the one-site phase."""
    rng = np.random.default_rng(20260902)
    first, _ = np.linalg.qr(
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )
    second, _ = np.linalg.qr(
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )
    opt = py.MpoOptimizer(
        qtn.MPO_rand(3, bond_dim=1, phys_dim=2, dtype="complex128", seed=8),
        gates=[(first, (0, 2)), (second, (0, 1))],
        chi=2,
        mode="dmrg1",
    )
    opt.run(
        n_iter=3,
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"] for record in opt.get_run_timing()["fit_steps"]
    ] == [2, 2, 1, 1, 1, 1]
    assert opt.get_fit_diagnostics()["dmrg1_one_site_locked"] is True


@pytest.mark.parametrize("n_iter", [1, 2])
def test_mpo_dmrg1_growth_requires_room_for_one_site_refinement(n_iter):
    """DMRG1 rejects under-capacity long-range windows without refinement."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(5, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 4))],
        chi=2,
        mode="dmrg1",
    )

    with pytest.raises(ValueError, match="n_iter >= 3"):
        opt.run(
            n_iter=n_iter,
            progbar=False,
            cutoff=0.0,
            fidelity_samples=0,
            fit_rtol=None,
        )


def test_mpo_optimizer_dmrg_mode_alias_set_mode_tracks_schedule():
    """Switching modes updates the named DMRG schedule metadata."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(3, dtype="complex128"),
        gates=[],
        chi=4,
        mode="dmrg1",
    )

    assert opt.mode == "dmrg"
    assert opt._dmrg_mode_alias == "dmrg1"

    opt.set_mode("dmrg3")
    assert opt.mode == "dmrg"
    assert opt._dmrg_mode_alias == "dmrg3"

    opt.set_mode("svd")
    assert opt.mode == "svd"
    assert opt._dmrg_mode_alias is None


def test_fit_two_site_preserves_mpo_view_and_dense_readout():
    """Direct FIT on an MPO must return a functional MPO, not an MPS view."""
    guess = qtn.MPO_rand(
        3, bond_dim=1, phys_dim=2, dtype="complex128", seed=212
    )
    target = qtn.MPO_rand(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=213
    )
    fit = py.FIT(target, p=guess, range_int=[0, 2])

    fit.run_gate(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
    )

    assert isinstance(fit.p, qtn.MatrixProductOperator)
    assert fit.p.upper_ind_id == guess.upper_ind_id
    assert fit.p.lower_ind_id == guess.lower_ind_id
    assert fit.p.to_dense().shape == (8, 8)
    assert fit.p.max_bond() <= 2


def test_mpo_prepare_gate_pair_uses_matrix_transpose_for_2q_quimb_gate():
    """For rank-2 two-site gates, use direct matrix transpose on ket only."""
    gate = qu.CNOT()
    g_k, g_b = py.MpoOptimizer._prepare_gate_pair(gate, n_sites=2)

    # Simplified branch: same rank and direct transpose.
    assert g_k.shape == gate.shape
    assert (g_k == gate.T).all()
    assert g_b is None

    # Equivalent to the previous reshape -> (2,3,0,1) mapping.
    old_style = gate.reshape(2, 2, 2, 2).transpose(2, 3, 0, 1).reshape(4, 4)
    assert (g_k == old_style).all()


def test_mpo_optimizer_single_gate_defaults_to_unitary_conjugation():
    """A bare ``G`` entry should apply ``G`` on ket and ``G†`` on bra."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    gates = [(gate, (0,))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_apply_gate_pair_separates_ket_and_bra_indices(monkeypatch):
    """Ket updates should use k indices and bra updates should use b indices."""
    calls = []

    def _fake_apply_gate(tn, gate, where, **kwargs):
        calls.append((where, kwargs.copy()))
        return tn

    monkeypatch.setattr("pepsy.optimizers.mpo.optimizer.apply_gate", _fake_apply_gate)

    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    opt = py.MpoOptimizer(
        mpo0.copy(),
        gates=[],
        chi=4,
        mode="svd",
        ind_id_k="k{}",
        ind_id_b="b{}",
    )

    opt._apply_gate_pair(
        opt.p,
        gate,
        (1,),
        bra_gate=gate,
        cutoff=1.0e-12,
        contract=True,
    )

    assert len(calls) == 2
    assert [kwargs["ind_id"] for _, kwargs in calls] == ["k{}", "b{}"]
    assert all(where == (1,) for where, _ in calls)


def test_mpo_optimizer_explicit_ket_only_pair_changes_identity():
    """Explicit ``(G, None)`` should keep ket-only update semantics."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    gates = [((gate, None), (0,))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_accepts_singleton_ket_only_shorthand():
    """A ``(G,)`` entry should be treated as explicit ket-only shorthand."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    gate = qu.phase_gate(0.37)
    gates = [((gate,), (0,))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_dmrg_smoke():
    """MpoOptimizer should apply mixed 1q/2q gates without error."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12)

    assert out.L == 4
    assert out.max_bond() >= 1


def test_mpo_optimizer_accepts_bundled_gate_stream():
    """Construction should accept bundled ``[(gate, where), ...]`` entries."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [((qu.hadamard(), None), (1,)), ((qu.CNOT(), None), (0, 3))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)

    assert out.L == 4
    assert opt.where == [(1,), (0, 3)]


def test_mpo_optimizer_compiles_and_caches_gate_stream_payloads():
    """Ordinary gate replay should reuse the compiled transpose payload."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1)), (qu.CNOT(), (0, 1))],
        chi=4,
        mode="svd",
    )

    plan = opt.compile_gate_stream()
    assert plan["length"] == 2
    assert plan["event_types"] == ("gate", "gate")
    assert plan["arities"] == (2, 2)
    assert plan["prepared_cache_size"] == 0
    assert isinstance(opt.gate_stream, tuple)

    opt.run(progbar=False, cutoff=1.0e-12, fidelity_samples=0)
    assert opt.compile_gate_stream()["prepared_cache_size"] == 1

    opt.clear_gate_cache()
    assert opt.compile_gate_stream()["prepared_cache_size"] == 0


def test_mpo_optimizer_layered_and_materialized_targets_are_distinct():
    """Dense FIT targets expose lazy-layered and materialized MPO policies."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[],
        chi=4,
        mode="dmrg",
    )
    layered = opt._build_dmrg_target(
        opt.p,
        qu.CNOT(),
        (0, 3),
        qu.CNOT(),
        cutoff=0.0,
        target_cutoff=0.0,
        target_strategy="layered",
    )
    materialized = opt._build_dmrg_target(
        opt.p,
        qu.CNOT(),
        (0, 3),
        qu.CNOT(),
        cutoff=0.0,
        target_cutoff=0.0,
        target_strategy="mpo",
    )

    assert len(layered.tensors) > len(materialized.tensors)
    assert len(materialized.tensors) == opt.p.L


def test_mpo_optimizer_layout_reduces_long_range_execution_span():
    """A persistent explicit layout should swap the MPO and remap supports."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=4,
        mode="svd",
    )

    opt.apply_layout((0, 3, 1, 2))
    assert opt.logical_order == [0, 3, 1, 2]
    assert opt.last_layout_plan["site_map"][3] == 1
    assert opt._execution_stream()[1] == [(0, 1)]

    out = opt.run(progbar=False, cutoff=1.0e-12, fidelity_samples=0)
    assert out.max_bond() <= 4
    assert any(event["kind"] == "layout_swap" for event in opt.get_norm_events())


def test_mpo_optimizer_channel_sum_reports_trace_preservation_separately():
    """A deterministic Kraus sum has a separate trace-preservation ledger."""
    channel = py.TrajectoryChannel.amplitude_damping(0.25)
    event = py.MpoOptimizer.channel_event(channel, 0)
    opt = py.MpoOptimizer(
        qtn.MPO_identity(2, dtype="complex128"),
        gates=[event],
        chi=4,
        mode="svd",
    )

    out = opt.run(progbar=False, cutoff=1.0e-12, fidelity_samples=0)
    diagnostics = opt.channel_diagnostics()
    trace_event = diagnostics["trace_events"][0]

    assert out.L == 2
    assert opt.compile_gate_stream()["event_types"] == ("channel_sum",)
    assert diagnostics["trace_preserving"] is True
    assert trace_event["channel_completeness_residual"] < 1.0e-10
    assert trace_event["trace_preservation_residual"] < 1.0e-10
    assert opt.norm_diagnostics()["events"] == 1


def test_mpo_channel_event_uses_standard_kraus_orientation():
    """Kraus payloads represent the documented ``K O K.H`` action."""
    operator = np.array(
        [[1.0 + 0.2j, 0.3 - 0.1j], [-0.4 + 0.5j, 0.7 + 0.6j]],
        dtype=np.complex128,
    )
    initial = qtn.MPO_identity(2, dtype="complex128")
    event = py.MpoOptimizer.kraus_event((operator,), 0)
    out = py.MpoOptimizer(
        initial,
        gates=[event],
        chi=4,
        mode="svd",
    ).run(progbar=False, cutoff=0.0, fidelity_samples=0)

    full_operator = np.kron(operator, np.eye(2))
    expected = full_operator @ initial.to_dense() @ full_operator.conj().T
    np.testing.assert_allclose(out.to_dense(), expected, atol=1.0e-12)


def test_mpo_optimizer_dmrg_handles_nonlocal_channel_sum():
    """A two-site Kraus sum remains executable through the DMRG backend."""
    identity = np.eye(4, dtype=np.complex128)
    phase = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    event = py.MpoOptimizer.kraus_event(
        (identity, phase),
        (0, 3),
        weights=(0.5, 0.5),
    )
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[event],
        chi=4,
        mode="dmrg",
    )

    out = opt.run(n_iter=1, progbar=False, cutoff=1.0e-12, fidelity_samples=0)
    assert out.max_bond() <= 4
    assert opt.channel_diagnostics()["channel_events"][0]["backend"] == "dmrg"
    assert opt.channel_diagnostics()["trace_preserving"] is True


def test_mpo_optimizer_non_trace_preserving_channel_is_reported():
    """Non-TP operator sums must not be conflated with compression loss."""
    event = py.MpoOptimizer.kraus_event((0.5 * np.eye(2),), 0)
    opt = py.MpoOptimizer(
        qtn.MPO_identity(2, dtype="complex128"),
        gates=[event],
        chi=4,
        mode="svd",
    )

    opt.run(progbar=False, cutoff=1.0e-12, fidelity_samples=0)
    trace_event = opt.get_trace_events()[0]
    assert trace_event["channel_trace_preserving"] is False
    assert trace_event["channel_completeness_residual"] == pytest.approx(0.75)
    assert trace_event["target_trace_residual"] == pytest.approx(0.75)


def test_mpo_optimizer_rejects_sampled_channel_semantics():
    """MPO replay must not silently turn a channel sum into one trajectory."""
    channel = py.TrajectoryChannel.amplitude_damping(0.25)
    event = py.MpoOptimizer.channel_event(channel, 0, semantics="sample")
    opt = py.MpoOptimizer(
        qtn.MPO_identity(2, dtype="complex128"),
        gates=[event],
        chi=4,
        mode="svd",
    )

    with pytest.raises(ValueError, match="semantics='sum'.*sampled branches"):
        opt.run(progbar=False, fidelity_samples=0)


def test_mpo_optimizer_transactional_fit_failure_preserves_completed_steps(monkeypatch):
    """Per-update rollback keeps earlier gates and records the failed FIT."""
    original_run_gate = py.FIT.run_gate
    calls = {"count": 0}

    def fail_second(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("forced local FIT failure")
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", fail_second)
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1)), (qu.CNOT(), (2, 3))],
        chi=2,
        mode="dmrg",
    )

    with pytest.raises(RuntimeError, match="forced local FIT failure"):
        opt.run(
            n_iter=1,
            progbar=False,
            fidelity_samples=0,
            atomic=False,
            transactional_steps=True,
        )

    assert len(opt.get_norm_events()) == 1
    assert opt.get_fit_history()[-1]["convergence_reason"] == "failed"


def test_mpo_optimizer_per_gate_fit_fallback_continues_stream(monkeypatch):
    """A local FIT failure can fall back without discarding later gates."""
    original_run_gate = py.FIT.run_gate
    calls = {"count": 0}

    def fail_first(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("forced local FIT failure")
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", fail_first)
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1)), (qu.CNOT(), (2, 3))],
        chi=2,
        mode="dmrg",
    )

    out = opt.run(
        n_iter=1,
        progbar=False,
        fidelity_samples=0,
        atomic=False,
        fit_fallback="svd",
    )

    assert out.max_bond() <= 2
    assert opt.last_run_status == "fallback"
    assert opt.last_run_fallback == "svd"
    assert len(opt.fallback_events) == 1
    assert len(opt.get_norm_events()) == 2


def test_mpo_optimizer_default_inplace_false_keeps_input_unchanged():
    """Default construction should work on a copy and keep input MPO intact."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    mpo0_ref = mpo0.copy()
    gate = qu.phase_gate(0.37)
    gates = [((gate, None), (0,))]

    opt = py.MpoOptimizer(mpo0, gates=gates, chi=4, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert opt.p is not mpo0
    assert np.allclose(mpo0.to_dense(), mpo0_ref.to_dense())
    assert not np.allclose(out.to_dense(), mpo0_ref.to_dense())


def test_mpo_optimizer_inplace_true_updates_input_mpo():
    """inplace=True should optimize the original input MPO object."""
    mpo0 = qtn.MPO_identity(2, dtype="complex128")
    mpo0_ref = mpo0.copy()
    gate = qu.phase_gate(0.37)
    gates = [((gate, None), (0,))]

    opt = py.MpoOptimizer(mpo0, gates=gates, chi=4, mode="svd", inplace=True)
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert opt.p is mpo0
    assert out is mpo0
    assert not np.allclose(mpo0.to_dense(), mpo0_ref.to_dense())


def test_mpo_optimizer_rejects_noncanonical_bundled_gate_aliases():
    """Bundled gate input should require the canonical list/tuple shapes."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")

    opt = py.MpoOptimizer(
        mpo0.copy(),
        gates=(((qu.hadamard(), None), (1,)),),
        chi=8,
        mode="svd",
    )
    assert opt.where == [(1,)]

    with pytest.raises(ValueError, match="exact shape"):
        py.MpoOptimizer(
            mpo0.copy(),
            gates=[((qu.hadamard(), None), (1,)), (qu.CNOT(), None)],
            chi=8,
            mode="svd",
        )


def test_mpo_optimizer_canonicalization_state_initialized():
    """Construction should initialize canonicalization metadata."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")

    assert isinstance(opt.info_c, dict)
    assert "cur_orthog" in opt.info_c
    cur = opt.info_c["cur_orthog"]
    assert isinstance(cur, tuple)
    assert len(cur) == 2
    expected_m, expected_e = py.tensors.tn_norm(
        mpo0, contraction_opt=opt.contraction_opt, strip_exponent=True
    )
    got_m, got_e = opt.norm_mpo
    assert np.isclose(got_m, expected_m)
    assert np.isclose(got_e, expected_e)


def test_mpo_optimizer_current_orthog_normalizes_supported_shapes():
    """Cached orthogonality metadata should accept 1-site and 2-site forms."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")

    opt.info_c["cur_orthog"] = 2
    assert opt._current_orthog() == (2, 2)

    opt.info_c["cur_orthog"] = (3,)
    assert opt._current_orthog() == (3, 3)

    opt.info_c["cur_orthog"] = (4, 1)
    assert opt._current_orthog() == (1, 4)

    opt.info_c["cur_orthog"] = (1, 2, 3)
    with pytest.raises(ValueError, match="cur_orthog must be"):
        opt._current_orthog()


def test_mpo_optimizer_sync_canonicalization_repairs_external_readout():
    """External MPO canonicalization can be rebound to ``info_c``."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")
    opt.canonize_mpo(opt.p, 0)
    assert opt.info_c["cur_orthog"] == (0, 0)

    # Model a low-level caller bypassing the optimizer's metadata owner.
    opt.p.canonize([4], cur_orthog=(0, 0))
    assert opt.info_c["cur_orthog"] == (0, 0)
    assert tuple(opt.p.calc_current_orthog_center()) == (4, 4)

    assert opt.sync_canonicalization() == (4, 4)
    assert opt.info_c["cur_orthog"] == (4, 4)


def test_mpo_optimizer_empty_norm_diagnostics_use_none_for_compression():
    """No compression events should not masquerade as unit fidelity."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"), gates=[], chi=4, mode="svd"
    )
    diagnostics = opt.norm_diagnostics()

    assert diagnostics["cumulative_fidelity"] is None
    assert diagnostics["cumulative_infidelity"] is None
    assert diagnostics["cumulative_norm"] is None
    assert diagnostics["norm"] == pytest.approx(diagnostics["state_norm"])
    assert diagnostics["norm_sq"] == pytest.approx(2**4)


def test_mpo_optimizer_canonize_mpo_accepts_supported_where_shapes():
    """canonize_mpo should accept int, singleton, and pair site selectors."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=6, mode="dmrg")

    opt.canonize_mpo(opt.p, 2)
    assert opt.info_c["cur_orthog"] == (2, 2)

    opt.canonize_mpo(opt.p, (3,))
    assert opt.info_c["cur_orthog"] == (3, 3)

    opt.canonize_mpo(opt.p, (4, 1))
    assert opt.info_c["cur_orthog"] == (1, 4)

    with pytest.raises(ValueError, match="where must be"):
        opt.canonize_mpo(opt.p, (1, 2, 3))


def test_mpo_optimizer_prepare_dmrg_state_expands_to_chi():
    """DMRG preparation should expand low-bond MPOs up to ``chi``."""
    mpo0 = qtn.MPO_identity(6, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=7, mode="dmrg")

    # Force a low-bond starting point then check expansion.
    opt.p = qtn.MPO_identity(6, dtype="complex128")
    assert opt.p.max_bond() == 1

    opt._prepare_dmrg_state()
    assert opt.p.max_bond() >= 7


@pytest.mark.parametrize("fit_block_size", (2, 3))
def test_mpo_optimizer_dmrg_forwards_native_fit_controls(
    monkeypatch,
    fit_block_size,
):
    """MPO DMRG must pass block and SVD policy into the FIT kernel."""
    calls = []
    original_run_gate = py.FIT.run_gate

    def recording_run_gate(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", recording_run_gate)

    opt = py.MpoOptimizer(
        qtn.MPO_identity(5, dtype="complex128"),
        gates=[((qu.CNOT(), None), (0, 4))],
        chi=2,
        mode="dmrg",
    )
    out = opt.run(
        n_iter=1,
        progbar=False,
        cutoff=2.0e-2,
        cutoff_mode="rel",
        fit_block_size=fit_block_size,
        fit_sweep_sequence="L",
        target_cutoff=0.0,
    )

    assert out.max_bond() <= 2
    assert len(calls) == 1
    assert calls[0]["block_size"] == fit_block_size
    assert calls[0]["sweep_sequence"] == "L"
    assert calls[0]["max_bond"] == 2
    assert calls[0]["cutoff"] == pytest.approx(2.0e-2)
    assert calls[0]["cutoff_mode"] == "rel"


def test_mpo_optimizer_norm_ledger_separates_gate_norm_from_compression():
    """Compression survival should be tracked independently of gate norm changes."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    nonunitary = np.diag([2.0, 1.0, 1.0, 1.0]).astype(np.complex128)
    gates = [
        (qu.CNOT(), (0, 1)),
        ((nonunitary, None), (2, 3)),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1.0e-12, fidelity_samples=0)

    events = opt.get_norm_events()
    diagnostics = opt.norm_diagnostics()
    assert len(events) == 2
    assert all(event["valid"] for event in events)
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
    assert removed_metric_names.isdisjoint(events[0])
    assert events[0]["local_fidelity"] == pytest.approx(1.0, abs=1.0e-10)
    assert events[1]["expected_norm"] != pytest.approx(events[0]["expected_norm"])
    assert diagnostics["events"] == 2
    assert diagnostics["completed_events"] == 2
    assert diagnostics["norm_survival"] == pytest.approx(
        np.prod([event["local_fidelity"] for event in events])
    )
    assert diagnostics["cumulative_fidelity"] == pytest.approx(
        diagnostics["norm_survival"]
    )
    assert diagnostics["norm"] == pytest.approx(diagnostics["state_norm"])
    assert diagnostics["cumulative_norm"] == pytest.approx(
        diagnostics["cumulative_fidelity"] ** 0.5
    )
    assert diagnostics["norm"] != pytest.approx(diagnostics["cumulative_norm"])


def test_mpo_optimizer_set_mpo_resets_lifecycle_diagnostics():
    """Replacing the live MPO must start a fresh norm and run ledger."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=4,
        mode="svd",
    )
    opt.run(progbar=False, cutoff=1.0e-12, fidelity_samples=1)
    assert opt.get_norm_events()
    assert len(opt.get_fidelities()) > 1

    opt.set_mpo(qtn.MPO_identity(4, dtype="complex128"))
    assert opt.get_norm_events() == []
    assert opt.get_fidelities() == [1.0]
    assert opt.norm_diagnostics()["events"] == 0
    assert opt.last_run_status == "not_run"


def test_mpo_optimizer_reports_fit_controls_and_timing():
    """DMRG should expose FIT convergence and opt-in timing diagnostics."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )
    opt.run(
        n_iter=3,
        progbar=False,
        cutoff=1.0e-12,
        fidelity_samples=0,
        fit_min_iter=1,
        fit_rtol=1.0e-8,
        fit_patience=1,
        fit_finite_check=True,
        timing=True,
        fit_collect_split_diagnostics=True,
        fit_overlap_diagnostics=True,
    )

    fit_diagnostics = opt.get_fit_diagnostics()
    timing = opt.get_run_timing()
    assert opt.last_run_status == "complete"
    assert fit_diagnostics["iterations"] >= 1
    assert fit_diagnostics["convergence_reason"] is not None
    assert fit_diagnostics["final_norm"] is not None
    assert fit_diagnostics["fit_overlap_diagnostics"] is True
    assert fit_diagnostics["fit_overlap_fidelity"] is not None
    assert fit_diagnostics["timing"]
    assert timing["status"] == "complete"
    assert timing["fit_calls"] == 1


def test_mpo_generic_dmrg_uses_mps_block_then_one_site_schedule():
    """Generic MPO DMRG follows the MPS adaptive warm-up handoff."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(5, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 4))],
        chi=2,
        mode="dmrg",
    )
    opt.run(
        n_iter=4,
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        fit_min_iter=1,
        fit_rtol=None,
        fit_adaptive_sweeps=2,
    )

    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["adaptive_sweeps"] == 2
    assert diagnostics["one_site_refinement_sweeps"] == 2


def test_mpo_timing_is_opt_in_and_mps_shaped(monkeypatch):
    """MPO timing has the MPS schema and no untimed clock side effects."""
    untimed = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg",
    )

    def fail_clock():
        raise AssertionError("timing=False must not read the profiling clock")

    def fail_synchronizer(*_args, **_kwargs):
        raise AssertionError(
            "timing=False must not construct a device synchronizer"
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            mpo_optimizer_module,
            "time",
            types.SimpleNamespace(perf_counter=fail_clock),
        )
        patch.setattr(
            mpo_optimizer_module.FIT,
            "_make_backend_synchronizer",
            fail_synchronizer,
        )
        untimed.run(
            progbar=False,
            cutoff=0.0,
            fidelity_samples=0,
            timing=False,
            timing_sync_device=True,
        )

    assert untimed.get_run_timing() is None

    timed = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg",
    )
    timed.run(
        n_iter=2,
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        timing=True,
    )
    timing = timed.get_run_timing()
    assert timing["event_count"] == 1
    assert timing["stages"]["dmrg.prepare"]["calls"] == 1
    assert timing["stages"]["dmrg.target"]["calls"] == 1
    assert timing["stages"]["dmrg.fit"]["calls"] == 1
    assert timing["fit_totals"]["calls"] == 1
    assert timing["fit_steps"]
    assert timing["backend"] == "numpy"


@pytest.mark.parametrize(("mode", "block_size"), [("dmrg1", 2), ("dmrg3", 3)])
@pytest.mark.parametrize("fit_mpo_guess", [True, False])
def test_mpo_named_dmrg_optional_mpo_fit_guess(mode, block_size, fit_mpo_guess):
    """Named MPO DMRG modes can toggle their direct MPO FIT guess."""
    initial = qtn.MPO_rand(
        5, bond_dim=1, phys_dim=2, dtype="complex128", seed=20260819
    )
    opt = py.MpoOptimizer(
        initial,
        gates=[(qu.CNOT(), (0, 4))],
        chi=2,
        mode=mode,
    )

    out = opt.run(
        n_iter=3,
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        fit_rtol=None,
        fit_mpo_guess=fit_mpo_guess,
    )

    diagnostics = opt.get_fit_diagnostics()
    assert out.max_bond() <= 2
    assert diagnostics["block_size"] == block_size
    assert diagnostics["mpo_fit_guess_used"] is fit_mpo_guess
    assert diagnostics["mpo_fit_guess_order"] == (
        "lower_upper" if fit_mpo_guess else None
    )


def test_mpo_named_dmrg_mpo_fit_guess_defaults_to_enabled():
    """The MPO FIT guess is enabled by default for named DMRG growth."""
    opt = py.MpoOptimizer(
        qtn.MPO_rand(5, bond_dim=1, phys_dim=2, dtype="complex128", seed=20260820),
        gates=[(qu.CNOT(), (0, 4))],
        chi=2,
        mode="dmrg1",
    )

    opt.run(n_iter=3, progbar=False, cutoff=0.0, fidelity_samples=0)

    assert opt.get_fit_diagnostics()["mpo_fit_guess_used"] is True


@pytest.mark.parametrize(
    ("order", "expected_layers"),
    [
        ("lower_upper", ["lower", "upper"]),
        ("upper_lower", ["upper", "lower"]),
    ],
)
def test_mpo_fit_guess_layer_order(monkeypatch, order, expected_layers):
    """MPO FIT guesses apply bra/lower and ket/upper in the requested order."""
    calls = []
    original = py.optimizers.mpo.optimizer.gate_nonlocal_opt

    def recording_gate_nonlocal_opt(*args, **kwargs):
        if kwargs.get("max_bond") == 2:
            calls.append(kwargs["which"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        py.optimizers.mpo.optimizer,
        "gate_nonlocal_opt",
        recording_gate_nonlocal_opt,
    )
    opt = py.MpoOptimizer(
        qtn.MPO_rand(5, bond_dim=1, phys_dim=2, dtype="complex128", seed=20260821),
        gates=[(qu.CNOT(), (0, 4))],
        chi=2,
        mode="dmrg1",
    )

    opt.run(
        n_iter=3,
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
        fit_mpo_guess_order=order,
    )

    assert calls[:2] == expected_layers


def test_mpo_optimizer_atomic_failure_restores_state():
    """A failed FIT replay should raise and leave the pre-run MPO intact."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    opt = py.MpoOptimizer(
        mpo0.copy(),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        opt.run(
            n_iter=2,
            progbar=False,
            fit_finite_check=lambda _state: False,
            fidelity_samples=0,
        )

    assert opt.last_run_status == "failed"
    assert opt.get_norm_events() == []
    assert np.allclose(opt.p.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_fit_fallback_replays_from_atomic_snapshot(monkeypatch):
    """A configured direct fallback should replay from the original MPO."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )

    def fail_fit(*_args, **_kwargs):
        raise RuntimeError("forced FIT failure")

    monkeypatch.setattr(opt, "_run_dmrg", fail_fit)
    out = opt.run(
        n_iter=2,
        progbar=False,
        fidelity_samples=0,
        fit_fallback="svd",
    )

    assert out.max_bond() <= 2
    assert opt.last_run_status == "fallback"
    assert opt.last_run_fallback == "svd"
    assert len(opt.get_norm_events()) == 1


def test_mpo_optimizer_tracks_fidelity_proxy_for_two_site_fit():
    """Two-site DMRG updates should append to the local-fidelity proxy trace."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [qu.CNOT(), qu.hadamard(), qu.CNOT()]
    where = [(0, 1), (2,), (2, 3)]
    gates = list(zip(G, where))

    chi = 6
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=chi, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12)

    assert len(opt.get_fidelities()) >= 3
    assert out.max_bond() <= chi


def test_mpo_optimizer_dmrg_accepts_k_2q_batch():
    """DMRG MPO mode should accept batching sequential two-site gates."""
    mpo0 = qtn.MPO_identity(5, dtype="complex128")
    G = [qu.CNOT(), qu.hadamard(), qu.CNOT(), qu.CNOT()]
    where = [(0, 1), (2,), (2, 3), (3, 4)]
    gates = list(zip(G, where))

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="dmrg")
    out = opt.run(n_iter=2, progbar=False, cutoff=1e-12, k_2q_batch=2)

    assert out.L == 5
    assert out.max_bond() <= 8
    assert len(opt.get_fidelities()) >= 2


def test_mpo_optimizer_dmrg_rejects_invalid_k_2q_batch():
    """DMRG MPO batching count should fail clearly when invalid."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [(qu.CNOT(), (0, 1))]
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="dmrg")

    with pytest.raises(ValueError, match="k_2q_batch must be >= 1"):
        opt.run(n_iter=2, progbar=False, cutoff=1e-12, k_2q_batch=0)


def test_mpo_optimizer_rejects_unknown_mode():
    """Unknown modes should fail with a clear supported-modes message."""
    mpo0 = qtn.MPO_identity(3, dtype="complex128")
    opt = py.MpoOptimizer(mpo0.copy(), gates=[], chi=4, mode="dmrg")

    with pytest.raises(ValueError, match="Supported modes:"):
        opt.set_mode("invalid_mode")


def test_mpo_optimizer_svd_smoke():
    """SVD mode should apply mixed 1q/2q gates without errors."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=2)

    assert out.L == 4
    assert out.max_bond() <= 8


def test_mpo_optimizer_normalized_norm_trace_stays_one_for_unitary_identity_evolution():
    """Two-sided unitary MPO evolution should preserve the normalized norm trace."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (1, 2)),
        (qu.phase_gate(0.37), (3,)),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=len(gates))

    assert np.allclose(out.to_dense(), mpo0.to_dense())
    assert all(np.isclose(val, 1.0, atol=1e-10) for val in opt.get_fidelities())


def test_mpo_optimizer_svd_rejects_negative_fidelity_samples():
    """Negative fidelity_samples should fail clearly in SVD mode."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [qu.CNOT()]
    where = [(0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")

    with pytest.raises(ValueError, match="fidelity_samples must be >= 0"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=-1)


def test_mpo_prepare_gate_pair_uses_explicit_bra_when_provided():
    """When B is provided, use B† on bra indices."""
    g = qu.CNOT()
    b = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    g_k, g_b = py.MpoOptimizer._prepare_gate_pair(g, n_sites=2, bra_gate=b)

    assert (g_k == g.T).all()
    assert (g_b == b.conj().T).all()


def test_mpo_optimizer_accepts_three_tuple_gate_spec():
    """Each ``G`` entry may be ``(G, B)`` with explicit bra-side operator B†."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(qu.hadamard(), qu.hadamard()), (qu.CNOT(), qu.swap())]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    assert out.L == 4


def test_mpo_optimizer_accepts_ket_only_gate_pair():
    """A ``(G, None)`` entry should apply ket side only."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(qu.hadamard(), None), (qu.CNOT(), None)]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    assert out.L == 4


def test_mpo_optimizer_accepts_bra_only_gate_pair():
    """A ``(None, B)`` entry should apply bra side only."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(None, qu.hadamard()), (None, qu.CNOT())]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)
    assert out.L == 4


def test_mpo_optimizer_rejects_empty_gate_pair():
    """A ``(None, None)`` gate entry should fail clearly."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    G = [(None, None)]
    where = [(1,)]
    gates = list(zip(G, where))
    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="svd")

    with pytest.raises(ValueError, match="at least one of G or B"):
        opt.run(progbar=False, cutoff=1e-12, fidelity_samples=1)


def test_mpo_optimizer_mpo_mode_smoke():
    """MPO mode should apply mixed 1q/2q gates via gate_nonlocal_opt without errors."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        (qu.hadamard(), (1,)),
        (qu.CNOT(), (0, 2)),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=2)

    assert out.L == 4
    assert out.max_bond() <= 8


def test_mpo_optimizer_mpo_mode_supports_three_site_gate():
    """MPO mode should accept a non-contiguous three-site gate support."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    z = np.diag([1.0, -1.0])
    local_xyz = np.kron(np.kron(x, y), z)
    gate = scipy_linalg.expm(-1j * 0.03 * local_xyz)
    mpo0 = qtn.MPO_identity(4, dtype="complex128")

    opt = py.MpoOptimizer(
        mpo0.copy(),
        gates=[((gate.T, None), (0, 1, 3))],
        chi=16,
        mode="mpo",
    )
    out = opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)

    assert out.L == 4
    assert out.max_bond() <= 16
    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def _embed_two_site_operator(operator, length, where):
    """Embed a two-site operator while preserving the MPO site ordering."""
    where = tuple(where)
    rest = [site for site in range(length) if site not in where]
    embedded = np.zeros((2**length, 2**length), dtype=complex)
    for row in range(2**length):
        row_bits = [(row >> (length - 1 - site)) & 1 for site in range(length)]
        local_row = sum(
            row_bits[site] << (len(where) - 1 - offset)
            for offset, site in enumerate(where)
        )
        for col in range(2**length):
            col_bits = [(col >> (length - 1 - site)) & 1 for site in range(length)]
            if any(row_bits[site] != col_bits[site] for site in rest):
                continue
            local_col = sum(
                col_bits[site] << (len(where) - 1 - offset)
                for offset, site in enumerate(where)
            )
            embedded[row, col] = operator[local_row, local_col]
    return embedded


@pytest.mark.parametrize("where", [(0, 1), (0, 5)])
def test_mpo_mode_complex_gate_matches_dmrg2(where):
    """MPO mode must match DMRG2 for nonsymmetric complex two-site gates."""
    length = 6
    initial = qtn.MPO_rand(
        length, bond_dim=2, phys_dim=2, dtype="complex128", seed=2718
    )
    rng = np.random.default_rng(20260817)
    random_matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gate = np.linalg.qr(random_matrix)[0]

    effective_gate = _embed_two_site_operator(gate.T, length, where)
    expected = (
        effective_gate
        @ initial.to_dense()
        @ effective_gate.conj().T
    )

    mpo_out = py.MpoOptimizer(
        initial.copy(), gates=[(gate, where)], chi=64, mode="mpo"
    ).run(progbar=False, cutoff=1e-13, fidelity_samples=1)
    dmrg_out = py.MpoOptimizer(
        initial.copy(), gates=[(gate, where)], chi=64, mode="dmrg2"
    ).run(
        n_iter=6,
        progbar=False,
        cutoff=1e-13,
        fidelity_samples=1,
        fit_adaptive_sweeps=2,
    )

    np.testing.assert_allclose(mpo_out.to_dense(), expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dmrg_out.to_dense(), expected, atol=1e-10, rtol=1e-10)


def test_mpo_mode_complex_gate_pair_sides_match_dense_action():
    """MPO mode preserves explicit ket-only and bra-only gate semantics."""
    length = 6
    where = (0, 5)
    initial = qtn.MPO_rand(
        length, bond_dim=2, phys_dim=2, dtype="complex128", seed=2719
    )
    rng = np.random.default_rng(20260818)
    random_matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gate = np.linalg.qr(random_matrix)[0]
    effective_gate = _embed_two_site_operator(gate.T, length, where)
    initial_dense = initial.to_dense()

    ket_out = py.MpoOptimizer(
        initial.copy(), gates=[((gate, None), where)], chi=64, mode="mpo"
    ).run(progbar=False, cutoff=1e-13, fidelity_samples=1)
    bra_out = py.MpoOptimizer(
        initial.copy(), gates=[((None, gate), where)], chi=64, mode="mpo"
    ).run(progbar=False, cutoff=1e-13, fidelity_samples=1)

    np.testing.assert_allclose(
        ket_out.to_dense(), effective_gate @ initial_dense, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        bra_out.to_dense(), initial_dense @ effective_gate.conj().T,
        atol=1e-10,
        rtol=1e-10,
    )


def test_mpo_mode_bare_two_site_gate_uses_native_dagger_sandwich(monkeypatch):
    """The direct MPO path uses Quimb's dagger-aware auto-swap method."""
    calls = []
    original = qtn.MatrixProductOperator.gate_sandwich_with_auto_swap

    def recording_gate_sandwich(self, *args, **kwargs):
        calls.append((args, dict(kwargs)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductOperator,
        "gate_sandwich_with_auto_swap",
        recording_gate_sandwich,
    )
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="mpo",
    )
    opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)

    assert len(calls) == 1
    assert calls[0][0][1] == (0, 3)
    assert calls[0][1]["dagger"] is True
    assert calls[0][1]["swap_back"] is True


def test_mpo_mode_multisite_compression_records_local_fidelity():
    """Multi-site MPO compression contributes to the norm-fidelity ledger."""
    rng = np.random.default_rng(20260902)
    gate, _ = np.linalg.qr(
        rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    )
    opt = py.MpoOptimizer(
        qtn.MPO_identity(4, dtype="complex128"),
        gates=[(gate, (0, 1, 3))],
        chi=64,
        mode="mpo",
    )
    opt.run(progbar=False, cutoff=0.0, fidelity_samples=0)

    events = opt.get_norm_events()
    assert len(events) == 1
    assert events[0]["where"] == (0, 1, 3)
    assert events[0]["expected_norm_sq"] == pytest.approx(2**4)
    assert events[0]["observed_norm_sq"] == pytest.approx(2**4)
    assert events[0]["local_fidelity"] == pytest.approx(1.0, abs=1e-10)


def test_mpo_unitary_norm_overshoot_guard_rejects_inconsistent_metadata():
    """A dense unitary compression event cannot report a real norm overshoot."""
    opt = py.MpoOptimizer(
        qtn.MPO_identity(3, dtype="complex128"), gates=[], chi=4, mode="mpo"
    )
    opt._finite_check_enabled = True
    with pytest.raises(FloatingPointError, match="exceeds its expected norm"):
        opt._record_norm_event(
            "mpo_compression",
            expected_norm=1.0,
            observed_norm=np.sqrt(1.01),
            unitary=True,
        )
    assert opt.get_norm_events() == []


def test_mpo_optimizer_mpo_mode_unitary_evolution_preserves_norm():
    """Two-sided unitary evolution in MPO mode should preserve the normalized norm."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (1, 2)),
        (qu.phase_gate(0.37), (3,)),
    ]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=len(gates))

    assert np.allclose(out.to_dense(), mpo0.to_dense(), atol=1e-10)
    assert all(np.isclose(val, 1.0, atol=1e-8) for val in opt.get_fidelities())


def test_mpo_optimizer_mpo_mode_ket_only_gate():
    """MPO mode with ket-only gate should not crash and should change the MPO."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [((qu.CNOT(), None), (0, 2))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert out.L == 4
    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def test_mpo_optimizer_mpo_mode_bra_only_gate():
    """MPO mode with bra-only gate should not crash and should change the MPO."""
    mpo0 = qtn.MPO_identity(4, dtype="complex128")
    gates = [((None, qu.CNOT()), (0, 2))]

    opt = py.MpoOptimizer(mpo0.copy(), gates=gates, chi=8, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-12, fidelity_samples=0)

    assert out.L == 4
    assert not np.allclose(out.to_dense(), mpo0.to_dense())


def _native_u1u1_identity_mpo(L=3):
    """Make a small native graded MPO fixture without testing construction."""
    pytest.importorskip("symmray")
    import symmray.utils as sr_utils

    phys_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
    arrays = []
    for site in range(L):
        if site == 0:
            data = np.zeros((1, 4, 4), dtype="complex128")
            data[0] = np.eye(4)
            index_maps = [[(0, 0)], phys_map, phys_map]
            duals = (False, False, True)
        elif site == L - 1:
            data = np.zeros((1, 4, 4), dtype="complex128")
            data[0] = np.eye(4)
            index_maps = [[(0, 0)], phys_map, phys_map]
            duals = (True, False, True)
        else:
            data = np.zeros((1, 1, 4, 4), dtype="complex128")
            data[0, 0] = np.eye(4)
            index_maps = [[(0, 0)], [(0, 0)], phys_map, phys_map]
            duals = (True, False, False, True)
        arrays.append(
            sr_utils.from_dense(
                data,
                symmetry="U1U1",
                index_maps=index_maps,
                duals=duals,
                fermionic=True,
                charge=(0, 0),
            )
        )
    return qtn.MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
    )


@pytest.mark.parametrize("mode", ["svd", "mpo", "dmrg"])
def test_mpo_optimizer_replays_native_graded_mpo_without_dense_fallback(mode):
    """Native graded MPO inputs remain FermionicArray-backed through replay."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    gates = fermion.strang_gate_stream(
        [(0, 1), (1, 2)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    out = py.MpoOptimizer(
        _native_u1u1_identity_mpo(),
        gates=gates,
        chi=8,
        mode=mode,
    ).run(progbar=False, cutoff=1e-10, fidelity_samples=0, n_iter=1)

    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


def test_mpo_optimizer_native_dmrg_uses_fit_controls(monkeypatch):
    """Native Symmray MPO DMRG must use block-aware FIT, not direct SVD."""
    calls = []
    original_run_gate = py.FIT.run_gate

    def recording_run_gate(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_run_gate(self, *args, **kwargs)

    monkeypatch.setattr(py.FIT, "run_gate", recording_run_gate)

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    gates = fermion.strang_gate_stream(
        [(0, 2)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    out = py.MpoOptimizer(
        _native_u1u1_identity_mpo(),
        gates=gates,
        chi=8,
        mode="dmrg",
    ).run(
        progbar=False,
        cutoff=2.0e-2,
        cutoff_mode="rel",
        target_cutoff=0.0,
        fit_block_size=3,
        fit_sweep_sequence="L",
        fidelity_samples=0,
        n_iter=1,
    )

    assert calls
    assert all(call["block_size"] == 3 for call in calls)
    assert all(call["sweep_sequence"] == "L" for call in calls)
    assert all(call["max_bond"] == 8 for call in calls)
    assert all(call["cutoff"] == pytest.approx(2.0e-2) for call in calls)
    assert all(call["cutoff_mode"] == "rel" for call in calls)
    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


@pytest.mark.parametrize(
    ("spinful", "symmetry", "params", "array_name"),
    [
        (False, "U1", {"V": 0.4, "mu": 0.1}, "U1FermionicArray"),
        (False, "Z2", {"V": 0.4, "mu": 0.1}, "Z2FermionicArray"),
        (True, "U1", {"U": 2.0, "mu": 0.1}, "U1FermionicArray"),
        (True, "U1U1", {"U": 2.0, "mu": 0.1}, "U1U1FermionicArray"),
        (True, "Z2", {"U": 2.0, "mu": 0.1}, "Z2FermionicArray"),
    ],
)
def test_mpo_optimizer_native_fermion_symmetries_use_dmrg_fit(
    spinful, symmetry, params, array_name,
):
    """Native fermion U1/U1U1/Z2 MPOs remain block-sparse under FIT."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=spinful, symmetry=symmetry)
    edges = [(0, 1), (1, 2), (2, 3)]
    hamiltonian = fermion.hamiltonian(edges, t=1.0, **params)
    mpo = fermion.to_mpo(hamiltonian=hamiltonian, L=4, compress=False)
    gates = fermion.gate_stream(edges, 0.002, t=1.0, **params)

    out = py.MpoOptimizer(
        mpo,
        gates=gates,
        chi=8,
        mode="dmrg",
    ).run(
        progbar=False,
        cutoff=1.0e-12,
        fidelity_samples=0,
        n_iter=1,
        fit_block_size=3,
    )

    assert out.max_bond() <= 8
    assert all(type(tensor.data).__name__ == array_name for tensor in out)


@pytest.mark.parametrize("mode", ["svd", "mpo"])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "params", "array_name"),
    [
        (False, "U1", {"V": 0.4, "mu": 0.1}, "U1FermionicArray"),
        (False, "Z2", {"V": 0.4, "mu": 0.1}, "Z2FermionicArray"),
        (True, "U1", {"U": 2.0, "mu": 0.1}, "U1FermionicArray"),
        (True, "U1U1", {"U": 2.0, "mu": 0.1}, "U1U1FermionicArray"),
        (True, "Z2", {"U": 2.0, "mu": 0.1}, "Z2FermionicArray"),
    ],
)
def test_mpo_optimizer_native_fermion_symmetries_use_direct_modes(
    mode, spinful, symmetry, params, array_name,
):
    """Native fermion MPOs survive direct SVD and MPO-mode replay."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=spinful, symmetry=symmetry)
    edges = [(0, 3)]
    hamiltonian = fermion.hamiltonian(edges, t=1.0, **params)
    mpo = fermion.to_mpo(hamiltonian=hamiltonian, L=4, compress=False)
    gates = fermion.strang_gate_stream(edges, 0.002, t=1.0, **params)

    out = py.MpoOptimizer(
        mpo,
        gates=gates,
        chi=8,
        mode=mode,
    ).run(
        progbar=False,
        cutoff=2.0e-2,
        cutoff_mode="rel",
        fidelity_samples=0,
    )

    assert out.max_bond() <= 8
    assert all(type(tensor.data).__name__ == array_name for tensor in out)


def test_mpo_optimizer_materializes_native_long_range_split_gates():
    """Long-range native split gates are canonicalizable after replay."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    gates = fermion.strang_gate_stream(
        [(0, 3)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    out = py.MpoOptimizer(
        _native_u1u1_identity_mpo(L=4),
        gates=gates,
        chi=8,
        mode="svd",
    ).run(progbar=False, cutoff=1e-10, fidelity_samples=0)

    assert out.L == 4
    assert all(len(out.tag_map[f"I{site}"]) == 1 for site in range(4))
    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


def test_mpo_optimizer_adapts_long_range_native_gate_to_jw_symmray_mpo():
    """The current JW MPO path also handles long-range native even gates."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    mpo = fermion.build_mpo(
        [(0, 3)],
        L=4,
        t=1.0,
        U=2.0,
        mu=0.1,
        fermionic=False,
    )
    gates = fermion.strang_gate_stream(
        [(0, 3)],
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    out = py.MpoOptimizer(mpo, gates=gates, chi=8, mode="svd").run(
        progbar=False,
        cutoff=1e-10,
        fidelity_samples=0,
    )

    assert out.L == 4
    assert all(type(tensor.data).__name__ == "U1U1Array" for tensor in out)


@pytest.mark.parametrize("mode", ["svd", "mpo", "dmrg"])
def test_mpo_optimizer_handles_fermion_symmray_mpo_and_native_gate_stream(mode):
    """The optimizer adapts native gates onto the current U1U1 MPO path."""
    pytest.importorskip("symmray")

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    edges = [(0, 1), (1, 2)]
    mpo = fermion.build_mpo(
        edges,
        L=3,
        t=1.0,
        U=2.0,
        mu=0.1,
        fermionic=False,
        max_bond=16,
        cutoff=1e-12,
    )
    gates = fermion.strang_gate_stream(
        edges,
        dt=0.01,
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    optimizer = py.MpoOptimizer(mpo, gates=gates, chi=8, mode=mode)
    out = optimizer.run(progbar=False, cutoff=1e-10, fidelity_samples=0, n_iter=1)

    assert out.L == 3
    assert out.max_bond() <= 8
    assert all(type(tensor.data).__name__ == "U1U1Array" for tensor in out)


def test_fermion_build_mpo_and_ham_tn_adapter_preserve_symmetry():
    """Both public MPO builders preserve the model's U1U1 symmetry."""
    pytest.importorskip("symmray")

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    edges = [(0, 1), (1, 2)]
    builder = py.ham_tn(Lx=3, Ly=1, data_type="complex128")

    direct = fermion.build_mpo(
        edges, L=3, t=1.0, U=0.0, mu=0.0, fermionic=True,
    )
    adapted = builder.build_mpo(
        fermion=fermion,
        edges=edges,
        phys_dim=4,
        t=1.0,
        U=0.0,
        mu=0.0,
        fermionic=True,
    )
    positional = builder.build_mpo(
        fermion,
        edges=edges,
        t=1.0,
        U=0.0,
        mu=0.0,
        fermionic=True,
    )

    assert direct.L == adapted.L == positional.L == 3
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in direct
    )
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in adapted
    )
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in positional
    )


def test_build_mpo_defaults_to_native_and_to_mpo_is_its_alias():
    """The model-facing builder has one native default and one alias."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=True, symmetry="U1U1")

    native = fermion.build_mpo(
        [(0, 1)],
        L=2,
        t=1.0,
        U=0.0,
        mu=0.0,
        compress=False,
    )
    direct = fermion.to_mpo(
        [(0, 1)],
        L=2,
        t=1.0,
        U=0.0,
        mu=0.0,
        compress=False,
    )

    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in native
    )
    assert type(fermion).to_mpo is type(fermion).build_mpo
    assert native.to_dense().allclose(direct.to_dense())


def test_mpo_optimizer_explicit_compress_handles_empty_symmray_stream():
    """The optimizer compresses symmetry-preserving Symmray MPOs directly."""
    pytest.importorskip("symmray")

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    mpo = fermion.build_mpo(
        [(0, 1), (1, 2)],
        L=3,
        t=1.0,
        U=0.0,
        mu=0.0,
        fermionic=False,
        compress=False,
    )
    raw_bond = mpo.max_bond()
    optimizer = py.MpoOptimizer(mpo, gates=[], chi=2, mode="svd")
    out = optimizer.compress(cutoff=1e-10)

    # Symmray can retain a small sector-multiplicity overshoot for a requested
    # cap, but the compression must still reduce the raw MPO bond.
    assert out.max_bond() < raw_bond
    assert all(type(tensor.data).__name__ == "U1U1Array" for tensor in out)


def test_fermion_to_mpo_builds_native_mpo_for_optimizer_replay():
    """The native Fermion.to_mpo path feeds the MPO optimizer directly."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    hopping = fermion.hopping_operator()
    two_site_mpo = fermion.to_mpo(
        {(0, 1): hopping},
        L=2,
        compress=False,
    )
    assert two_site_mpo.to_dense().allclose(hopping.fuse((0, 1), (2, 3)))

    hamiltonian = fermion.hamiltonian(
        [(0, 1), (1, 2)],
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    mpo = fermion.to_mpo(
        hamiltonian=hamiltonian,
        L=3,
        max_bond=16,
        cutoff=1e-12,
    )

    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in mpo)

    out = py.MpoOptimizer(
        mpo,
        gates=hamiltonian.trotter_gates(0.01),
        chi=8,
        mode="svd",
    ).run(progbar=False, cutoff=1e-10, fidelity_samples=0)
    assert all(type(tensor.data).__name__ == "U1U1FermionicArray" for tensor in out)


def test_fermion_to_mpo_preserves_configured_backend():
    """Native MPO conversion applies the Fermion backend to every block."""
    torch = pytest.importorskip("torch")
    backend = py.backend_torch(dtype=torch.complex128, device="cpu")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        to_backend=backend,
    )

    mpo = fermion.to_mpo(
        [(0, 1)],
        L=2,
        t=1.0,
        U=2.0,
        mu=0.1,
        compress=False,
    )

    for tensor in mpo:
        assert tensor.data.backend == "torch"
        assert all(isinstance(block, torch.Tensor) for block in tensor.data.blocks.values())


def test_fermion_to_mpo_accepts_arbitrary_neutral_term_support():
    """Native MPO conversion supports non-contiguous multi-site terms."""
    fermion = py.Fermion(spinful=False, symmetry="U1")
    term = fermion.operator_term(
        [(1.0, ((2, "create"), (0, "number"), (3, "annihilate")))],
        sites=(2, 0, 3),
    )
    hamiltonian = fermion.hamiltonian({(2, 0, 3): term})
    mpo = fermion.to_mpo(
        hamiltonian=hamiltonian,
        L=4,
        compress=False,
    )

    assert mpo.L == 4
    assert all(type(tensor.data).__name__ == "U1FermionicArray" for tensor in mpo)
    embedded_term = fermion.operator_term(
        [(1.0, ((2, "create"), (0, "number"), (3, "annihilate")))],
        sites=(0, 1, 2, 3),
    )
    assert mpo.to_dense().allclose(
        embedded_term.fuse((0, 1, 2, 3), (4, 5, 6, 7))
    )


def test_fermion_to_mpo_handles_one_site_native_term():
    """Native MPO construction also handles the no-virtual-bond case."""
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    term = fermion.interaction_operator()
    mpo = fermion.to_mpo({(0,): term}, L=1, compress=False)

    assert mpo.to_dense().allclose(term)
    assert mpo.pepsy_compression_report["raw_max_bond"] == 1
