"""Tests for :mod:`pepsy.optimizers.mps.optimizer`."""

import inspect
import types

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy as py
import pepsy.fitting.local as fitting_local_module
import pepsy.optimizers.mps.optimizer as mps_optimizer_module


def _non_unitary_entangling_gate():
    """Return a small two-site filter that creates entanglement from |++>."""
    return np.diag([1.0, 0.5, 0.5, 2.0]).astype(complex)


def _two_branch_flip_submpo(*, L, sites, targets, w0=0.7, w1=0.3):
    """Return ``w0 * I + w1 * prod(X_targets)`` as a sparse-site MPO."""
    eye = np.eye(2, dtype=np.complex128)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
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


def _mps_data_norm(mps):
    """Return the MPS norm without its stored global exponent."""
    mps_data = mps.copy()
    mps_data.exponent = 0.0
    return mps_data.norm()


def _perm_mps_to_logical_dense(opt):
    """Return a permuted-mode MPS dense state in logical site order."""
    physical = opt.p.to_dense().reshape((2,) * opt.p.L)
    logical_axes = [opt.qubits.index(site) for site in range(opt.p.L)]
    return np.transpose(physical, logical_axes).reshape(-1)


def _tensor_data_norm(mps, site):
    """Return the Frobenius norm of one MPS tensor's data."""
    return float(np.linalg.norm(np.asarray(mps[site].data)))


def test_mps_optimizer_effective_length_is_gate_active_and_cap_reduced():
    """L_eff is a lightweight stream-support ledger, not a rank probe."""
    product = qtn.MPS_computational_state("000", dtype="complex128")

    empty = py.MpsOptimizer(product, gates=[], chi=8, mode="mpo")
    assert empty.L_eff == 0
    empty.run(cutoff=0.0, progbar=False)
    assert empty.L_eff == 0

    one_site = py.MpsOptimizer(
        product,
        gates=[(qu.hadamard(), (1,))],
        chi=8,
        mode="mpo",
    )
    one_site.run(cutoff=0.0, progbar=False)
    assert one_site.L_eff == 1
    assert one_site.mps_length_diagnostics()["L_eff_history"] == (0, 1)

    nonlocal_gate = py.MpsOptimizer(
        product,
        gates=[(qu.CNOT(), (0, 2))],
        chi=8,
        mode="mpo",
    )
    nonlocal_gate.run(cutoff=0.0, progbar=False)
    assert nonlocal_gate.L_eff == 3

    capped = py.MpsOptimizer(
        product,
        gates=[
            (qu.hadamard(), (1,)),
            ("cap", 1, np.array([1.0, 1.0]), "left"),
        ],
        chi=8,
        mode="mpo",
    )
    capped.run(cutoff=0.0, progbar=False)
    assert capped.L_eff == 0
    assert capped.mps_length_diagnostics()["L_eff_history"] == (0, 1, 0)


def _nonuniform_product_mps():
    """Return a non-translationally-invariant complex product state."""
    return qtn.MPS_product_state(
        [
            np.array([1.0, 0.0], dtype=complex),
            np.array([0.0, 1.0], dtype=complex),
            np.array([np.cos(0.3), np.sin(0.3)], dtype=complex),
            np.array([np.cos(0.5), 1j * np.sin(0.5)], dtype=complex),
        ]
    )


def _three_site_ghz_target():
    """Return a four-site GHZ-like target embedded in a five-site MPS."""
    state = qtn.MPS_computational_state("00000", dtype="complex128")
    target = state.copy()
    target.gate_(qu.hadamard(), 0, contract=True)
    for where in ((0, 1), (1, 2), (2, 3)):
        target.gate_(
            qu.CNOT(),
            where,
            contract="split",
            max_bond=2,
            cutoff=0.0,
        )
    return state, target


def _assert_event_sites_locally_normalized(mps, event):
    """Check that every tensor rescaled by an event has local norm one."""
    for site in event["sites"]:
        assert _tensor_data_norm(mps, site) == pytest.approx(1.0)


def test_mps_optimizer_has_automatic_norm_tracking_without_legacy_api():
    """Norm-survival diagnostics are automatic, without an opt-in flag."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    opt = py.MpsOptimizer(state, gates=[], chi=2, mode="svd")

    assert callable(opt.norm_diagnostics)
    assert opt.norm_diagnostics()["tracking"] is True
    assert not hasattr(opt, "get_fidelities")
    assert not hasattr(opt, "get_true_infidelities")
    assert not hasattr(opt, "get_norm_infidelity_samples")
    assert not hasattr(opt, "track_infidelity")
    assert not hasattr(opt, "get_infidelities")
    assert not hasattr(opt, "get_infidelity_samples")
    assert not hasattr(opt, "reset_infidelity_tracking")
    with pytest.raises(TypeError, match="track_infidelity"):
        py.MpsOptimizer(state, gates=[], chi=2, mode="svd", track_infidelity=False)
    with pytest.raises(TypeError, match="fidelity_samples"):
        opt.run(progbar=False, fidelity_samples=0)


def test_mps_optimizer_run_rejects_removed_infidelity_option():
    """The removed keyword is rejected instead of silently doing extra work."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="svd",
    )
    with pytest.raises(TypeError, match="track_infidelity"):
        opt.run(progbar=False, track_infidelity=False)


def test_mps_optimizer_accepts_svd_mode():
    """SVD mode should be accepted by ``MpsOptimizer`` mode validation."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    assert opt.mode == "svd"


def test_mps_optimizer_simple_update_routes_torch_u1u1_long_range_gate():
    """SU routed SWAPs should stay on the live Torch Symmray backend."""
    torch = pytest.importorskip("torch")

    backend = py.backend_torch(dtype=torch.float64, device="cpu")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="float64",
    )
    state = py.hrs_to_mps(
        4,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        chi=4,
        seed=1,
        dtype="float64",
        cyclic=False,
    )
    state.apply_to_arrays(backend)
    fermion.to_backend = backend
    hopping = fermion.hopping_gate(0.001, t=1.0, imaginary=True)

    optimizer = py.MpsOptimizer(
        state,
        gates=[(hopping, (0, 3))],
        chi=4,
        mode="su",
        inplace=True,
    )
    out = optimizer.run(progbar=False, cutoff=1.0e-10, non_unitary=True)

    assert type(out[0].data).__name__ == "U1U1FermionicArray"
    assert out.max_bond() <= 4
    assert len(optimizer.gauges) == out.L - 1


@pytest.mark.parametrize(
    "mode", ["dmrg", "mpo", "svd", "swap", "perm", "mix", "su", "exact"]
)
def test_mps_optimizer_rejects_mismatched_gate_stream_backend(mode):
    """Mismatched user gates must be prepared before optimizer construction."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    gate = qu.CNOT()  # ordinary NumPy gate stream
    with pytest.raises(TypeError, match="requires every gate"):
        py.MpsOptimizer(
            state,
            gates=[(gate, (0, 1))],
            chi=2,
            mode=mode,
            inplace=True,
        )


def test_mps_optimizer_rejects_mismatched_submpo_stream_backend():
    """Mismatched sub-MPO tensors must be prepared before replay."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    submpo = _two_branch_flip_submpo(L=2, sites=(0, 1), targets=(0, 1))
    optimizer = py.MpsOptimizer(
        state,
        gates=[],
        chi=2,
        mode="mpo",
        inplace=True,
    )

    with pytest.raises(TypeError, match="sub-MPO"):
        optimizer.set_gates([py.MpsOptimizer.submpo_event(submpo, (0, 1))])
    assert all(isinstance(tensor.data, np.ndarray) for tensor in submpo.tensors)
    assert all(isinstance(tensor.data, torch.Tensor) for tensor in optimizer.p.tensors)


def test_mps_optimizer_backend_diagnostics_check_every_gate():
    """Backend checks inspect every gate, not only the first stream payload."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    to_backend = py.backend_torch(dtype=torch.complex128, device="cpu")
    state.apply_to_arrays(to_backend)
    matching = to_backend(np.eye(2, dtype=complex))
    foreign = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    optimizer = py.MpsOptimizer(
        state,
        gates=[],
        chi=2,
        mode="mpo",
        inplace=True,
    )

    assert optimizer.backend_info() == {
        "backend": "torch",
        "dtype": "complex128",
        "device": "cpu",
    }
    assert optimizer.backend == "torch"
    with pytest.raises(TypeError, match=r"stream\[1\]"):
        optimizer._validate_gate_stream_backend(
            [matching, foreign], ["gate", "gate"]
        )


def test_mps_optimizer_accepts_explicitly_prepared_gate():
    """Callers can prepare a payload explicitly before installing it."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="svd")

    gate = optimizer.to_backend(qu.CNOT())
    optimizer.set_gates([(gate, (0, 1))])
    optimizer.run(progbar=False)

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in optimizer.p.tensors)


def test_mps_optimizer_rejects_mixed_state_backends():
    """All live MPS tensors must agree on backend, dtype, and device."""
    torch = pytest.importorskip("torch")

    state = qtn.MPS_computational_state("00", dtype="complex128")
    state[0].modify(data=torch.as_tensor(state[0].data, dtype=torch.complex128))
    with pytest.raises(TypeError, match="one compatible backend"):
        py.MpsOptimizer(state, gates=[], chi=2, mode="mpo")


def test_mps_optimizer_reports_symmray_block_backend():
    """Symmray diagnostics retain the underlying Torch block backend."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")

    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=1,
        dtype="complex128",
    )
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="mpo")

    assert optimizer.backend_info() == {
        "backend": "symmray",
        "dtype": "complex128",
        "device": "cpu",
        "array_backend": "torch",
    }


def test_mps_optimizer_rejects_dense_symmray_submpo_blocks():
    """Dense sub-MPO blocks cannot be promoted into a native Symmray MPS."""
    pytest.importorskip("symmray")
    torch = pytest.importorskip("torch")

    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=1,
        dtype="complex128",
    )
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    submpo = qtn.MatrixProductOperator.from_dense(
        fermion.hopping_gate(0.01, t=1.0),
        dims=(4, 4),
        sites=(0, 1),
        L=3,
    )
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="mpo")

    with pytest.raises(TypeError, match="sub-MPO"):
        optimizer._validate_gate_stream_backend([submpo], ["submpo"])
    assert all(tensor.data.backend == "numpy" for tensor in submpo.tensors)


def test_mps_optimizer_accepts_perm_mode():
    """Perm mode should expose an identity logical-to-physical ordering initially."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="perm")

    assert opt.mode == "perm"
    assert opt.qubits == [0, 1, 2, 3]


def test_mps_optimizer_simple_update_initializes_and_keeps_gauges_separate():
    """Simple-update mode keeps its core and external bond gauges separate."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gauges = {}
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 3))],
        chi=2,
        mode="su",
        gauges=gauges,
    )

    out = opt.run(progbar=False, cutoff=1e-12)

    assert out is opt.p
    assert opt.gauges is gauges
    assert len(gauges) == out.L - 1
    assert opt.info_c == {}
    assert opt.p_ungauged is not None

    physical = out.copy()
    physical.gauge_simple_insert(gauges)
    assert physical.norm() == pytest.approx(1.0, rel=1e-10, abs=1e-10)
    assert opt.p_ungauged.norm() == pytest.approx(physical.norm())
    assert np.allclose(opt.p_ungauged.to_dense(), physical.to_dense())


def test_mps_optimizer_simple_update_forwards_gate_simple_options(monkeypatch):
    """SU mode should use ``gate_simple`` with fixed renormalization."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gauges = {}
    p0.gauge_all_simple_(gauges=gauges, progbar=False)
    calls = []

    def fake_gate_simple(tn, gate, where, **kwargs):
        calls.append((tn, gate, where, kwargs))
        return tn

    monkeypatch.setattr(mps_optimizer_module, "apply_gate_simple", fake_gate_simple)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 3))],
        chi=3,
        mode="su",
        gauges=gauges,
    )
    opt.run(progbar=False, cutoff=1e-9, cutoff_mode="rel")

    assert len(calls) == 1
    _, _, where, kwargs = calls[0]
    assert where == (0, 3)
    assert kwargs["gauges"] is gauges
    assert kwargs["renorm"] is True
    assert kwargs["max_bond"] == 3
    assert kwargs["cutoff"] == pytest.approx(1e-9)
    assert kwargs["cutoff_mode"] == "rel"


def test_mps_optimizer_perm_tracks_lazy_order_and_logical_state():
    """Perm mode should leave swaps in place while preserving logical evolution."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (0, 2)),
    ]
    perm = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="perm")
    reference = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="swap")

    perm.run(progbar=False, cutoff=1e-12)
    reference.run(progbar=False, cutoff=1e-12)

    assert perm.qubits == [0, 2, 3, 1]
    assert np.allclose(_perm_mps_to_logical_dense(perm), reference.p.to_dense().reshape(-1))

    perm.restore_qubit_order()
    assert perm.qubits == [0, 1, 2, 3]
    assert np.allclose(perm.p.to_dense().reshape(-1), reference.p.to_dense().reshape(-1))


def test_mps_optimizer_perm_maps_control_events_to_logical_sites():
    """Controls after lazy swaps should still address their logical site labels."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        [
            (qu.hadamard(), (0,)),
            (qu.CNOT(), (0, 3)),
            ("measure", "Z", 3, +1),
        ],
        chi=8,
        mode="perm",
    )

    opt.run(progbar=False)

    assert opt.qubits == [0, 3, 1, 2]
    assert opt.measurements[0][1] == (3,)
    assert opt.measurements[0][2] == 1


def test_mps_optimizer_svd_smoke():
    """SVD mode should apply mixed 1q/2q gates without errors."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.p.max_bond() <= 8


def test_mps_optimizer_opt_in_run_timing_reports_replay_metrics():
    """Timing is opt-in and returns a copy-safe replay record."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="direct",
    )

    assert opt.get_run_timing() is None
    opt.run(progbar=False, timing=True)

    timing = opt.get_run_timing()
    assert timing["status"] == "complete"
    assert timing["mode"] == "quimb-direct"
    assert timing["mode_alias"] is None
    assert timing["event_count"] == 1
    assert timing["elapsed_seconds"] >= 0.0
    assert timing["final_bond"] <= 2
    assert timing["stages"]["direct.replay"]["calls"] == 1
    assert timing["stages"]["canonicalize"]["calls"] >= 1
    assert not any(name.startswith("infidelity.") for name in timing["stages"])
    timing["mode"] = "changed"
    assert opt.last_run_timing["mode"] == "quimb-direct"


def test_mps_optimizer_diagnostic_accessors_are_copy_safe():
    """Public diagnostic snapshots cannot mutate optimizer-owned state."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 1)), (scale, (0,))],
        chi=1,
        mode="dmrg",
    )

    opt.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        non_unitary=True,
        normalize_every=True,
        quality_check_every=True,
    )

    quality_checks = opt.get_quality_checks()
    normalizations = opt.get_normalizations()
    fit_diagnostics = opt.get_fit_diagnostics()

    assert quality_checks
    assert normalizations
    assert fit_diagnostics is not None

    quality_checks[0]["step"] = -1
    normalizations[0]["step"] = -1
    fit_diagnostics["iterations"] = -1

    assert opt.quality_checks[0]["step"] != -1
    assert opt.normalizations[0]["step"] != -1
    assert opt._last_dmrg_fit_diagnostics["iterations"] != -1


def test_mps_optimizer_fit_diagnostics_is_none_before_fit():
    """The public FIT diagnostic accessor is explicit before any FIT run."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="mpo",
    )

    assert opt.get_fit_diagnostics() is None


def test_mps_optimizer_fit_overlap_diagnostic_failure_is_nonfatal(monkeypatch):
    """Optional FIT overlap failures stay diagnostics, not replay failures."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="dmrg",
    )

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(mps_optimizer_module, "tn_fidelity", fail)
    result = opt._fit_overlap_diagnostics(opt.p, opt.p)

    assert result["fit_overlap_fidelity"] is None
    assert result["fit_overlap_infidelity"] is None
    assert "diagnostic unavailable" in result["fit_overlap_error"]


@pytest.mark.parametrize("overlap", [float("nan"), float("inf"), -float("inf")])
def test_mps_optimizer_fit_overlap_nonfinite_is_reported(monkeypatch, overlap):
    """Non-finite optional overlaps must not become valid clipped fidelities."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="dmrg",
    )
    monkeypatch.setattr(mps_optimizer_module, "tn_fidelity", lambda *a, **k: overlap)

    result = opt._fit_overlap_diagnostics(opt.p, opt.p)

    assert result["fit_overlap_fidelity"] is None
    assert result["fit_overlap_infidelity"] is None
    assert "non-finite" in result["fit_overlap_error"]


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3", "mix"])
@pytest.mark.parametrize("timing", [False, True])
def test_mps_optimizer_fit_overlap_diagnostics_are_opt_in(monkeypatch, mode, timing):
    """The expensive FIT-target overlap contraction is disabled by default."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(8, 2, dtype="complex128", seed=714),
        gates=[(qu.CNOT(), (2, 4))],
        chi=2,
        mode=mode,
    )

    def fail(*args, **kwargs):
        raise AssertionError("FIT overlap diagnostics should be opt-in")

    monkeypatch.setattr(mps_optimizer_module, "tn_fidelity", fail)
    optimizer.run(progbar=False, n_iter=2, fit_rtol=None, timing=timing)

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_overlap_diagnostics"] is False
    assert diagnostics["fit_overlap_fidelity"] is None
    assert diagnostics["fit_overlap_infidelity"] is None
    assert diagnostics["fit_overlap_error"] is None


@pytest.mark.parametrize("mode", ["mpo", "swap", "svd", "dmrg", "mix"])
def test_mps_optimizer_modes_skip_clocks_when_timing_disabled(
    monkeypatch,
    mode,
):
    """Untimed replay must not touch the profiling clock in any mode."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    def fail_clock():
        raise AssertionError("timing=False must not read the profiling clock")

    monkeypatch.setattr(
        mps_optimizer_module,
        "time",
        types.SimpleNamespace(perf_counter=fail_clock),
    )

    def fail_synchronizer(*_args, **_kwargs):
        raise AssertionError(
            "timing=False must not construct a device synchronizer"
        )

    monkeypatch.setattr(
        mps_optimizer_module.FIT,
        "_make_backend_synchronizer",
        fail_synchronizer,
    )
    optimizer.run(progbar=False, timing=False, timing_sync_device=True)

    assert optimizer.get_run_timing() is None


def test_mps_optimizer_timing_record_identifies_named_dmrg_mode():
    """Timed DMRG aliases retain their schedule identity and fit summary."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=213),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(progbar=False, n_iter=2, fit_rtol=None, timing=True)

    timing = optimizer.get_run_timing()
    assert timing["mode"] == "dmrg"
    assert timing["mode_alias"] == "dmrg2"
    assert timing["fit_diagnostics"] == optimizer.get_fit_diagnostics()
    assert timing["fit_diagnostics"]["block_size"] == 2


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
def test_mps_optimizer_long_range_dmrg_seeds_disposable_fit_guess(mode):
    """DMRG keeps the target exact and seeds only the disposable FIT guess."""
    stream = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 7)),
    ]
    reference = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        stream,
        chi=4,
        mode="mpo",
    ).run(progbar=False)

    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        stream,
        chi=4,
        mode=mode,
    )
    out = optimizer.run(progbar=False, n_iter=4, fit_rtol=None)

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    assert out.max_bond() == 2
    assert optimizer.norm_diagnostics()["infidelity"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["backend"] == "fit"
    assert diagnostics["fallback"] is False
    initialization = diagnostics["random_initialization"]
    assert diagnostics["mpo_fit_guess_used"] is True
    assert diagnostics["guess_used"] is True
    assert diagnostics["guess_method"] == "src"
    assert initialization["enabled"] is False
    assert initialization["reason"] == "guess_src"


def test_randomized_fit_guess_is_disposable_and_active_only():
    """Random initialization expands a copy, never the live current MPS."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="dmrg2")
    guess, initialization = optimizer._build_randomized_fit_guess(
        state,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-4,
    )
    assert guess is not state
    assert [state.bond_size(i, i + 1) for i in range(3)] == [1, 1, 1]
    assert [guess.bond_size(i, i + 1) for i in range(3)] == [2, 2, 2]
    assert initialization["enabled"] is True
    assert [record["new_rank"] for record in initialization["bonds"]] == [2, 2, 2]


def test_random_fit_guess_preserves_existing_bond_dimensions_and_is_deterministic():
    """The random-only guess perturbs p without changing its bond ranks."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="dmrg2")
    guess_a, info_a = optimizer._build_randomized_fit_guess(
        state,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-2,
        expand=False,
        seed=17,
    )
    guess_b, info_b = optimizer._build_randomized_fit_guess(
        state,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-2,
        expand=False,
        seed=17,
    )

    assert info_a["enabled"] is True
    assert info_a["expanded"] is False
    assert info_a["sites"] == [0, 1, 2, 3]
    assert info_b == info_a
    assert [guess_a.bond_size(i, i + 1) for i in range(3)] == [1, 1, 1]
    assert all(
        np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(guess_a.tensors, guess_b.tensors)
    )
    assert any(
        not np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(state.tensors, guess_a.tensors)
    )


@pytest.mark.parametrize(
    "strategy",
    [
        "direct",
        "random",
        "random_expand",
        "guess_direct",
        "guess_zipup",
        "svd_guess",
    ],
)
def test_mps_optimizer_fit_initial_guess_strategy_is_diagnostic(strategy):
    """Each initial-guess strategy remains separate from FIT's target."""
    stream = [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 3))]
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        stream,
        chi=2,
        mode="dmrg2",
    )
    optimizer.run(
        progbar=False,
        n_iter=2,
        cutoff=1.0e-12,
        fit_rtol=None,
        fit_init_strategy=strategy,
        fit_init_rand_strength=1.0e-2,
        stabilize_unitary=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy_requested"] == strategy
    assert diagnostics["fit_init_strategy"] == strategy
    assert diagnostics["backend"] == "fit"
    if strategy == "random":
        assert diagnostics["random_initialization"]["expanded"] is False
        assert diagnostics["random_initialization"]["enabled"] is True
    elif strategy == "random_expand":
        assert diagnostics["random_initialization"]["expanded"] is True
        assert diagnostics["random_initialization"]["enabled"] is True
    elif strategy.startswith("guess_") or strategy == "svd_guess":
        assert diagnostics["svd_guess_used"] is True
        assert diagnostics["guess_used"] is True
    else:
        assert diagnostics["mpo_fit_guess_used"] is False


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    ],
)
def test_mps_optimizer_mpo_method_modes(method):
    """Every Quimb method is selectable as ``quimb-<method>``."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=f"quimb-{method}",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert out.max_bond() <= 2
    assert optimizer.mode == f"quimb-{method}"


@pytest.mark.parametrize("method", ["sdc", "sdc-oversample"])
def test_mps_optimizer_sdc_modes_are_opt_in_and_version_gated(method):
    """New SDC modes never silently fall back on older Quimb installations."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=f"quimb-{method}",
    )
    supported = mps_optimizer_module._quimb_compression_method_available(method)

    if not supported:
        with pytest.raises(NotImplementedError, match="sdc compressor"):
            optimizer.run(progbar=False, cutoff=1.0e-12)
        return

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )
    assert out.max_bond() <= 2
    assert optimizer.mode == f"quimb-{method}"


def test_mps_optimizer_bare_sdc_mode_normalizes_to_quimb_sdc():
    """The bare SDC spelling is a first-class MPS compression mode."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="sdc",
    )
    assert optimizer.mode == "quimb-sdc"

    if not mps_optimizer_module._quimb_compression_method_available("sdc"):
        with pytest.raises(NotImplementedError, match="sdc compressor"):
            optimizer.run(progbar=False, cutoff=1.0e-12)
        return

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )
    assert out.max_bond() <= 2


@pytest.mark.parametrize("method", ["sdc", "sdc-oversample"])
def test_mps_optimizer_sdc_fit_init_strategies_are_version_gated(method):
    """SDC is available both as a mode and as a FIT warm-start method."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg2",
    )
    supported = mps_optimizer_module._quimb_compression_method_available(method)

    if not supported:
        with pytest.raises(NotImplementedError, match="sdc compressor"):
            optimizer.run(
                progbar=False,
                n_iter=2,
                fit_rtol=None,
                fit_init_strategy=f"guess-{method}",
                stabilize_unitary=False,
            )
        return

    out = optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        fit_init_strategy=f"guess-{method}",
        stabilize_unitary=False,
    )
    diagnostics = optimizer.get_fit_diagnostics()
    assert out.max_bond() <= 2
    assert diagnostics["fit_init_strategy"] == f"guess_{method}"
    assert diagnostics["guess_method"] == method
    assert diagnostics["guess_used"] is True


@pytest.mark.parametrize(
    "mode, method",
    [
        ("quimb", "direct"),
        ("quimb-direct", "direct"),
        ("quimb-src", "src"),
        ("mpo", "direct"),
        ("mpo-direct", "direct"),
        ("mpo-src", "src"),
    ],
)
def test_mps_optimizer_quimb_mode_aliases(mode, method):
    """Quimb names are canonical while the old MPO names remain valid."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert out.max_bond() <= 2
    assert optimizer._mode_mpo_method(optimizer.mode) == method


@pytest.mark.parametrize(
    ("mode", "canonical", "method"),
    [
        ("direct", "quimb-direct", "direct"),
        ("dm", "quimb-dm", "dm"),
        ("zipup", "quimb-zipup", "zipup"),
        ("src", "quimb-src", "src"),
        ("srcmps", "quimb-srcmps", "srcmps"),
        ("fit-projector", "quimb-fit-projector", "fit-projector"),
    ],
)
def test_mps_optimizer_accepts_bare_quimb_method_modes(mode, canonical, method):
    """Bare Quimb method names normalize to the qualified backend mode."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[],
        chi=2,
        mode=mode,
    )

    assert optimizer.mode == canonical
    assert optimizer._is_mpo_mode(mode)
    assert optimizer._mode_mpo_method(mode) == method


def test_mps_optimizer_defaults_to_direct_and_preserves_legacy_aliases():
    """Omitting mode selects direct compression, including sub-MPO replay."""
    assert inspect.signature(py.MpsOptimizer).parameters["mode"].default == "direct"
    state = qtn.MPS_rand_state(4, bond_dim=3, dtype="complex128", seed=917)
    stream = [(qu.rand_uni(4, seed=918), (0, 3))]
    stream.append(("submpo", qtn.MPO_identity(2, dtype="complex128"), (1, 2)))
    reference = None
    for kwargs in ({}, {"mode": "direct"}, {"mode": "mpo"}, {"mode": "quimb"}):
        optimizer = py.MpsOptimizer(state, stream, chi=2, **kwargs)
        optimizer.run(cutoff=0., timing=True)
        assert optimizer.mode == "quimb-direct"
        assert optimizer._progress_mode_name(optimizer.mode) == "direct"
        assert optimizer.get_run_timing()["mode"] == "quimb-direct"
        assert optimizer.get_fit_diagnostics() is None
        result = optimizer.to_dense()
        if reference is None:
            reference = result
        else:
            np.testing.assert_allclose(result, reference, atol=1e-12)


def test_mps_optimizer_fit_name_remains_dmrg_alias():
    """Bare ``fit`` remains DMRG while Quimb FIT stays qualified."""
    dmrg = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="fit",
    )
    quimb_fit = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="quimb-fit",
    )

    assert dmrg.mode == "dmrg"
    assert quimb_fit.mode == "quimb-fit"


@pytest.mark.parametrize("mode", ["src", "zipup"])
def test_mps_optimizer_runs_bare_quimb_method_mode(mode):
    """Bare SRC and zip-up names select the corresponding replay backend."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert optimizer.mode == f"quimb-{mode}"
    assert out.max_bond() <= 2


@pytest.mark.parametrize(
    "mode, timing_name",
    [("mpo", "direct"), ("quimb-src", "src")],
)
def test_mps_optimizer_timing_names_follow_quimb_method(mode, timing_name):
    """MPO-family timing stages expose the selected compressor name."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
        timing=True,
    )

    stages = optimizer.get_run_timing()["stages"]
    assert stages[f"{timing_name}.replay"]["calls"] == 1
    assert stages[f"{timing_name}.stabilize"]["calls"] == 1


@pytest.mark.parametrize(
    "mode, expected_desc",
    [
        ("quimb", "direct"),
        ("quimb-dm", "dm"),
        ("quimb-src", "src"),
        ("mpo", "direct"),
        ("mpo-zipup", "zipup"),
        ("mpo-src", "src"),
    ],
)
def test_mps_optimizer_progress_bar_uses_mode_name(
    monkeypatch,
    mode,
    expected_desc,
):
    """MPO-family progress bars show only the selected replay mode."""
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
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=True,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert descriptors == [expected_desc]


@pytest.mark.parametrize(
    "mode, expected_desc",
    [
        ("fit", "dmrg"),
        ("dmrg", "dmrg"),
        ("dmrg1", "dmrg1"),
        ("dmrg2", "dmrg2"),
        ("dmrg3", "dmrg3"),
    ],
)
def test_mps_optimizer_dmrg_progress_bar_uses_schedule_name(
    monkeypatch,
    mode,
    expected_desc,
):
    """Named DMRG schedules identify themselves in the progress bar."""
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
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        [(qu.CNOT(), (0, 1))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=True,
        n_iter=2,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    assert descriptors == [expected_desc]


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    ],
)
def test_mps_optimizer_guess_method_strategies(method):
    """Every Quimb method is available as a ``guess-<method>`` policy."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        fit_init_strategy=f"guess-{method}",
        stabilize_unitary=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy"] == f"guess_{method}"
    assert diagnostics["guess_method"] == method
    assert diagnostics["guess_used"] is True


def test_mps_optimizer_hyphenated_guess_strategy_alias():
    """The canonical ``guess-<method>`` spelling normalizes cleanly."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        fit_init_strategy="guess-src",
        stabilize_unitary=False,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy_requested"] == "guess_src"
    assert diagnostics["fit_init_strategy"] == "guess_src"
    assert diagnostics["guess_method"] == "src"


def test_mps_optimizer_default_src_guess_reaches_one_site_fit():
    """The default DMRG warm-start remains SRC after reaching ``chi``."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=20260823,
        ),
        [(np.eye(4, dtype=np.complex128), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=1,
        fit_rtol=None,
        fit_block_size=1,
        timing=True,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["block_size"] == 1
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["guess_used"] is True


def test_mps_optimizer_src_compression_seed_is_reproducible():
    """Explicit seeds control both MPO replay and disposable SRC guesses."""
    state = qtn.MPS_rand_state(
        6,
        bond_dim=2,
        phys_dim=2,
        seed=23,
        dtype="complex128",
    )
    gate = qu.CNOT()

    guess_optimizer = py.MpsOptimizer(state, gates=[], chi=3, mode="dmrg2")
    guess_a = guess_optimizer._build_compression_fit_guess(
        state,
        gate,
        (0, 5),
        method="src",
        cutoff=1.0e-12,
        cutoff_mode="rsum2",
        seed=17,
    )
    guess_b = guess_optimizer._build_compression_fit_guess(
        state,
        gate,
        (0, 5),
        method="src",
        cutoff=1.0e-12,
        cutoff_mode="rsum2",
        seed=17,
    )
    assert all(
        np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(guess_a.tensors, guess_b.tensors)
    )

    replay_a = py.MpsOptimizer(
        state.copy(), [(gate, (0, 5))], chi=3, mode="quimb-src"
    ).run(
        progbar=False,
        compression_seed=17,
        stabilize_unitary=False,
    )
    replay_b = py.MpsOptimizer(
        state.copy(), [(gate, (0, 5))], chi=3, mode="quimb-src"
    ).run(
        progbar=False,
        compression_seed=17,
        stabilize_unitary=False,
    )
    assert all(
        np.array_equal(tensor_a.data, tensor_b.data)
        for tensor_a, tensor_b in zip(replay_a.tensors, replay_b.tensors)
    )


@pytest.mark.parametrize("method", ["srcmps", "fit", "fit-oversample"])
def test_mps_optimizer_compression_seed_is_not_forwarded_as_quimb_option(method):
    """Seeded Quimb methods use Quimb's RNG without leaking ``seed`` kwargs."""
    state = qtn.MPS_rand_state(
        6,
        bond_dim=2,
        phys_dim=2,
        seed=23,
        dtype="complex128",
    )
    gate = qu.rand_uni(4, seed=11)
    outputs = []
    for _ in range(2):
        optimizer = py.MpsOptimizer(
            state.copy(deep=True),
            [(gate, (0, 5))],
            chi=3,
            mode=f"quimb-{method}",
        )
        out = optimizer.run(
            progbar=False,
            cutoff=1.0e-12,
            compression_seed=17,
            stabilize_unitary=False,
        )
        outputs.append([np.array(tensor.data, copy=True) for tensor in out.tensors])

    assert all(
        np.array_equal(tensor_a, tensor_b)
        for tensor_a, tensor_b in zip(outputs[0], outputs[1])
    )


def test_mps_optimizer_submpo_fit_projector_product_state_is_finite():
    """Explicit endpoint sub-MPOs disable the singular projector pre-gauge."""
    mpo = qtn.MatrixProductOperator.from_dense(
        qu.CNOT(),
        dims=(2, 2),
        sites=(0, 3),
        L=4,
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [py.MpsOptimizer.submpo_event(mpo, (0, 3))],
        chi=2,
        mode="quimb-fit-projector",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        non_unitary=True,
        stabilize_unitary=False,
    )

    assert all(np.isfinite(np.asarray(tensor.data)).all() for tensor in out.tensors)


@pytest.mark.parametrize("method", ["zipup-first", "fit-zipup", "fit-projector"])
def test_mps_optimizer_interior_oversampled_mpo_methods(method):
    """Nested Quimb methods also work on an interior sub-MPO span."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            8,
            bond_dim=2,
            phys_dim=2,
            seed=3,
            dtype="complex128",
        ),
        [(qu.CNOT(), (1, 6))],
        chi=3,
        mode=f"quimb-{method}",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
    )

    assert out.max_bond() <= 3
    assert all(np.isfinite(np.asarray(tensor.data)).all() for tensor in out.tensors)


@pytest.mark.parametrize("method", ["zipup-first", "fit-zipup"])
def test_mps_optimizer_interior_submpo_oversampled_methods(method):
    """Explicit interior sub-MPO events use the same safe local path."""
    mpo = _two_branch_flip_submpo(
        L=8,
        sites=(1, 6),
        targets=(1, 6),
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            8,
            bond_dim=2,
            phys_dim=2,
            seed=3,
            dtype="complex128",
        ),
        [py.MpsOptimizer.submpo_event(mpo, (1, 6))],
        chi=3,
        mode=f"quimb-{method}",
    )

    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        stabilize_unitary=False,
        non_unitary=True,
    )

    assert out.max_bond() <= 3
    assert all(np.isfinite(np.asarray(tensor.data)).all() for tensor in out.tensors)


@pytest.mark.parametrize("cutoff_mode", [None, "auto"])
def test_mps_optimizer_dm_uses_native_cutoff_mode_by_default(
    monkeypatch,
    cutoff_mode,
):
    """The MPO density-matrix method keeps Quimb's native rsum1 default."""
    calls = []

    def fake_gate_nonlocal_(self, gate, where, **kwargs):
        calls.append(kwargs)
        kwargs["info"]["cur_orthog"] = (min(where), min(where))
        return self

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_nonlocal_",
        fake_gate_nonlocal_,
    )

    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        [(qu.CNOT(), (0, 3))],
        chi=2,
        mode="quimb-dm",
    )
    optimizer.run(
        progbar=False,
        stabilize_unitary=False,
        cutoff_mode=cutoff_mode,
    )
    assert "cutoff_mode" not in calls[-1]

    optimizer.run(
        progbar=False,
        cutoff_mode="rsum2",
        stabilize_unitary=False,
    )
    assert calls[-1]["cutoff_mode"] == "rsum2"


@pytest.mark.parametrize("mode", ["mpo", "swap", "svd"])
def test_mps_optimizer_compression_timing_separates_norm_stages(mode):
    """Enabled timing records stabilization work without diagnostic stages."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(_non_unitary_entangling_gate(), (0, 3))],
        chi=1,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        non_unitary=True,
        normalize_final=False,
        timing=True,
    )

    stages = optimizer.get_run_timing()["stages"]
    assert not any(name.startswith("infidelity.") for name in stages)
    assert not any(name.endswith(".stabilize") for name in stages)


def test_mps_optimizer_empty_run_retains_timing_sync_request():
    """An empty replay reports the same synchronization option it received."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(timing=True, timing_sync_device=True)

    timing = optimizer.get_run_timing()
    assert timing["event_count"] == 0
    assert timing["timing_sync_device"] is True


def test_mps_optimizer_mix_skips_elapsed_clock_when_timing_is_disabled():
    """Mixed-mode summaries avoid clock reads outside opt-in profiling."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="mix",
    )

    optimizer.run(progbar=False, timing=False)

    assert optimizer.last_mix_summary["elapsed_seconds"] is None


def test_mps_optimizer_timing_reports_fit_sweeps_and_sites():
    """Opt-in DMRG timing exposes FIT sweep and active-site records."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=31),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    opt.run(progbar=False, n_iter=3, timing=True)

    timing = opt.get_run_timing()
    fit_steps = timing["fit_steps"]
    assert timing["stages"]["dmrg.target"]["calls"] == 1
    assert len(fit_steps) == 3
    assert [record["sweep"] for record in fit_steps] == [1, 2, 3]
    assert all(record["status"] == "complete" for record in fit_steps)
    assert all(record["range_int"] == (0, 2) for record in fit_steps)
    assert [record["site_count"] for record in fit_steps] == [2, 2, 3]
    assert [record["direction"] for record in fit_steps] == ["R", "L", "R"]
    assert [record["block_size"] for record in fit_steps] == [2, 2, 1]
    assert [record["fit_index"] for record in fit_steps] == [0, 0, 0]
    assert [record["record_index"] for record in fit_steps] == [0, 1, 2]
    assert [len(record["site_timings"]) for record in fit_steps] == [2, 2, 3]
    assert all(
        record["elapsed_seconds"] >= 0.0
        and all(site["elapsed_seconds"] >= 0.0 for site in record["site_timings"])
        for record in fit_steps
    )


def test_mps_optimizer_timing_distinguishes_fit_calls_from_sweeps():
    """All sweeps from one target fit share a stable FIT call identifier."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3, bond_dim=2, phys_dim=2, dtype="complex128", seed=209
        ),
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (0, 2))],
        chi=2,
        mode="fit",
    )

    optimizer.run(progbar=False, n_iter=3, timing=True)

    records = optimizer.get_run_timing()["fit_steps"]
    assert [record["fit_index"] for record in records] == [0, 0, 0, 1, 1, 1]
    assert [record["record_index"] for record in records] == [0, 1, 2, 3, 4, 5]
    assert [record["sweep"] for record in records] == [1, 2, 3, 1, 2, 3]


def test_mps_optimizer_timing_does_not_enable_split_diagnostics(monkeypatch):
    """Profiling records clocks without changing FIT's SVD metadata path."""
    observed = []
    original_run_gate = mps_optimizer_module.FIT.run_gate

    def inspect_run_gate(self, *args, **kwargs):
        observed.append(kwargs.get("collect_split_diagnostics"))
        result = original_run_gate(self, *args, **kwargs)
        assert "two_site_splits" not in self.info
        return result

    monkeypatch.setattr(
        mps_optimizer_module.FIT,
        "run_gate",
        inspect_run_gate,
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=210,
        ),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(progbar=False, n_iter=3, fit_rtol=None, timing=True)

    assert observed == [False]
    assert optimizer.get_run_timing()["fit_steps"]


def test_mps_optimizer_timing_transfers_records_without_internal_copies(
    monkeypatch,
):
    """Detailed FIT records move into the run result and copy only on read."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=211,
        ),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    def fail_internal_copy(_value):
        raise AssertionError("timing collection must transfer owned records")

    with monkeypatch.context() as context:
        context.setattr(mps_optimizer_module, "deepcopy", fail_internal_copy)
        context.setattr(fitting_local_module, "deepcopy", fail_internal_copy)
        optimizer.run(progbar=False, n_iter=3, fit_rtol=None, timing=True)

    timing = optimizer.get_run_timing()
    assert timing["fit_steps"]
    timing["fit_steps"].clear()
    assert optimizer.last_run_timing["fit_steps"]


def test_mps_optimizer_dmrg_uses_gate_window_fit(monkeypatch):
    """DMRG keeps FIT restricted to the gate window, not the full MPS."""
    called_ranges = []
    original_run_gate = mps_optimizer_module.FIT.run_gate

    def record_run_gate(self, *args, **kwargs):
        called_ranges.append(tuple(self.range_int))
        return original_run_gate(self, *args, **kwargs)

    def fail_full_chain_fit(*args, **kwargs):
        raise AssertionError("MpsOptimizer DMRG must not call FIT.run_eff")

    monkeypatch.setattr(mps_optimizer_module.FIT, "run_gate", record_run_gate)
    monkeypatch.setattr(mps_optimizer_module.FIT, "run_eff", fail_full_chain_fit)

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(5, bond_dim=2, phys_dim=2, dtype="complex128", seed=32),
        gates=[(qu.CNOT(), (1, 3))],
        chi=2,
        mode="dmrg",
    )
    opt.run(progbar=False, n_iter=2)

    assert called_ranges == [(1, 3)]


def test_mps_optimizer_mix_timing_includes_mix_summary():
    """Mixed timing records expose the existing backend decision summary."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 1))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, timing=True)

    timing = opt.get_run_timing()
    assert timing["mix_summary"] == opt.last_mix_summary
    assert timing["mix_summary"]["elapsed_seconds"] >= 0.0


def test_mps_optimizer_mix_warms_up_with_mpo_then_uses_dmrg():
    """Mix mode should use MPO until the active bonds reach their targets."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 1)),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (1, 2)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=3,
        fit_block_size=1,
    )

    assert out.max_bond() <= 2
    assert [event["backend"] for event in opt.mix_history][:3] == [
        "mpo",
        "mpo",
        "mpo",
    ]
    assert opt.mix_history[-1]["backend"] == "dmrg"
    assert opt.mix_history[-1]["reason"] == "bond_at_target"
    assert opt.last_mix_summary["mpo_steps"] == 3
    assert opt.last_mix_summary["dmrg_steps"] == 1
    assert opt.last_mix_summary["fallback_steps"] == 0


def test_mps_optimizer_mix_mpo_warmup_hands_off_to_dmrg1():
    """Direct warm-up should feed the default mixed one-site DMRG schedule."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 1)),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (0, 2)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=3,
        fit_rtol=None,
        timing=True,
    )

    assert out.max_bond() <= 2
    assert [event["backend"] for event in opt.mix_history] == [
        "mpo",
        "mpo",
        "mpo",
        "dmrg",
    ]
    fit_steps = opt.get_run_timing()["fit_steps"]
    assert [record["block_size"] for record in fit_steps] == [1, 1, 1]
    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["block_size"] == 1
    assert diagnostics["one_site_refinement_sweeps"] == 3
    assert opt.last_mix_summary["mpo_steps"] == 3
    assert opt.last_mix_summary["dmrg_steps"] == 1


def test_mps_optimizer_mix_one_site_fast_path_keeps_input_identity():
    """Mixed mode should apply one-site gates without a DMRG trial copy."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=17)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.hadamard(), (1,))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.mix_history[0]["backend"] == "mpo"
    assert opt.mix_history[0]["reason"] == "one_site_exact"
    assert opt.p is p0
    assert out is p0


def test_mps_optimizer_mix_targets_attainable_edge_bonds():
    """Mixed warm-up should cap edge targets by their physical ranks."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[],
        chi=8,
        mode="mix",
    )

    assert opt._mix_target_bond_dimensions() == [2, 4, 2]


def test_mps_optimizer_mix_keeps_short_active_bonds_on_mpo():
    """Mixed mode must not pad a previously short active bond for DMRG."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.CNOT(), (0, 1)),
        (qu.CNOT(), (1, 2)),
        (qu.CNOT(), (2, 3)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=8,
        fit_block_size=1,
    )

    expected = np.zeros(16, dtype=complex)
    expected[[0, 15]] = 1.0 / np.sqrt(2.0)
    assert np.allclose(out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1), expected)
    assert opt.mix_history[2]["backend"] == "mpo"
    assert opt.mix_history[2]["reason"] == "active_bond_below_target"
    assert opt.mix_history[3]["backend"] == "mpo"
    assert opt.mix_history[3]["reason"] == "active_bond_below_target"


def test_mps_optimizer_mix_history_accumulates_control_segments():
    """Mixed-mode diagnostics should cover all gate segments in one run."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00"),
        gates=[
            (qu.hadamard(), (0,)),
            ("measure", "Z", 0, +1),
            (qu.CNOT(), (0, 1)),
        ],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, seed=7)

    assert [event["step"] for event in opt.mix_history] == [1, 2]
    assert [event["backend"] for event in opt.mix_history] == ["mpo", "mpo"]
    assert opt.last_mix_summary["mpo_steps"] == 2
    assert opt.last_mix_summary["dmrg_steps"] == 0


def test_mps_optimizer_mix_one_site_is_exact_at_target_bond():
    """One-site gates stay on the exact fast path at the target bond."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=17)
    assert p0.max_bond() == 2
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.mix_history[0]["backend"] == "mpo"
    assert opt.mix_history[0]["reason"] == "one_site_exact"


def test_mps_optimizer_mix_falls_back_to_mpo_on_nonfinite_dmrg(monkeypatch):
    """Mix mode should restore and use MPO if DMRG leaves non-finite data."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=19)
    p0_ref = p0.copy()
    gates = [(qu.CNOT(), (0, 2))]
    original_run_dmrg = py.MpsOptimizer._run_dmrg

    def nonfinite_dmrg(self, *args, **kwargs):
        original_run_dmrg(self, *args, **kwargs)
        center = self._current_orthog(self.p)[0]
        data = np.asarray(self.p[center].data)
        self.p[center].modify(data=np.full_like(data, np.nan))

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", nonfinite_dmrg)
    opt = py.MpsOptimizer(p0, gates=gates, chi=2, mode="mix", inplace=True)

    opt.run(progbar=False, cutoff=1e-12, finite_check=True)

    assert opt.mix_history[0]["backend"] == "mpo"
    assert opt.mix_history[0]["reason"] == "dmrg_fallback"
    assert "non-finite" in opt.mix_history[0]["fallback_error"]
    assert opt.last_mix_summary["fallback_steps"] == 1
    assert opt.p is p0
    assert py.MpsOptimizer._mps_data_is_finite(opt.p)
    reference = py.MpsOptimizer(p0_ref, gates=gates, chi=2, mode="mpo")
    reference.run(progbar=False, cutoff=1e-12)
    assert np.allclose(opt.p.to_dense(), reference.p.to_dense())


def test_mps_optimizer_finite_check_combines_backend_scalars_once(monkeypatch):
    """A whole-MPS health check should perform one backend-to-host conversion."""
    torch = pytest.importorskip("torch")
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=20
    )
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex128, device="cpu"))
    original_to_numpy = mps_optimizer_module.ar.to_numpy
    conversions = []

    def counted_to_numpy(value):
        conversions.append(value)
        return original_to_numpy(value)

    monkeypatch.setattr(mps_optimizer_module.ar, "to_numpy", counted_to_numpy)

    assert py.MpsOptimizer._mps_data_is_finite(state)
    assert len(conversions) == 1


def test_fit_gate_cheap_finite_check_transfers_one_vector_per_sweep(monkeypatch):
    """Cheap FIT health checks should transfer one tiny vector per sweep."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=201
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])
    original_to_numpy = mps_optimizer_module.ar.to_numpy
    conversions = []

    def counted_to_numpy(value):
        conversions.append(value)
        return original_to_numpy(value)

    monkeypatch.setattr(mps_optimizer_module.ar, "to_numpy", counted_to_numpy)

    fit.run_gate(rtol=None, n_iter=3, finite_check=True)

    assert fit.iterations_run == 3
    assert len(conversions) == 3
    # One native finite flag per active tensor plus the terminal center norm's
    # finite flag are transferred together; local norms are reduced only once
    # per completed sweep.
    assert all(np.asarray(original_to_numpy(value)).size == 4 for value in conversions)
    assert len(fit.local_norm_trace) == 3


def test_fit_gate_timing_records_sweep_and_site_steps():
    """FIT timing is opt-in and reports the active interval."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=202
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(block_size=1, n_iter=2, timing=True)

    records = fit.get_timing()
    assert [record["sweep"] for record in records] == [1, 2]
    assert all(record["status"] == "complete" for record in records)
    assert all(record["range_int"] == (0, 2) for record in records)
    assert all(record["site_count"] == 3 for record in records)
    assert all(record["timing_schema"] == 3 for record in records)
    assert all(record["active_site_count"] == 3 for record in records)
    assert all(record["update_count"] == 3 for record in records)
    assert all(record["svd_seconds"] == 0.0 for record in records)
    assert all(
        {
            "effective_seconds",
            "svd_seconds",
            "writeback_seconds",
            "environment_seconds",
            "canonicalization_seconds",
            "moving_environment_seconds",
        }.issubset(site_timing)
        for record in records
        for site_timing in record["site_timings"]
    )
    assert all(
        {
            "canonicalization_seconds",
            "sweep_preparation_canonicalization_seconds",
            "fixed_environment_seconds",
            "moving_environment_seconds",
            "moving_canonicalization_seconds",
            "sweep_overhead_seconds",
        }.issubset(record)
        for record in records
    )


def test_fit_gate_disabled_timing_never_reads_a_clock():
    """The normal FIT path bypasses timing marks and record allocation."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=212,
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    def fail_timing_mark(*_values):
        raise AssertionError("timing=False must not read the profiling clock")

    fit._timing_mark = fail_timing_mark
    fit.run_gate(n_iter=2, timing=False)

    assert fit.get_timing() == []


def test_fit_gate_timing_separates_fixed_and_moving_environments():
    """Two-site FIT timing exposes environment and decomposition phases."""
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=203
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 3])

    fit.run_gate(n_iter=2, block_size=2, timing=True)

    records = fit.get_timing()
    assert all(record["block_size"] == 2 for record in records)
    assert all(record["fixed_environment_seconds"] >= 0.0 for record in records)
    assert all(record["canonicalization_seconds"] >= 0.0 for record in records)
    assert all(record["moving_environment_seconds"] >= 0.0 for record in records)
    assert all(record["sweep_overhead_seconds"] >= 0.0 for record in records)
    for record in records:
        assert record["moving_environment_seconds"] == pytest.approx(
            sum(
                site["moving_environment_seconds"]
                for site in record["site_timings"]
            )
        )
        assert record["canonicalization_seconds"] >= (
            record["sweep_preparation_canonicalization_seconds"]
        )


def test_fit_gate_two_site_grows_only_active_bonds():
    """Two-site FIT should discover rank without globally padding the MPS."""
    initial = qtn.MPS_computational_state("0000", dtype="complex128")
    initial.gate_(qu.hadamard(), 0, contract=True)
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 2),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    fit = py.FIT(target, p=initial, range_int=[0, 2], cutoffs=0.0)

    fit.run_gate(
        n_iter=4,
        block_size=2,
        sweep_sequence="RL",
        max_bond=4,
        cutoff=0.0,
    )

    assert fit.p.bond_size(0, 1) > 1
    assert fit.p.bond_size(1, 2) > 1
    assert fit.p.bond_size(2, 3) == 1
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(
        1.0,
        abs=1.0e-10,
    )
    assert [record["direction"] for record in fit.get_timing()] == []


def test_fit_gate_randomized_guess_handles_cutoff_from_product_state():
    """A seeded disposable guess opens remote-gate sectors before the cutoff."""
    initial = qtn.MPS_computational_state("0000", dtype="complex128")
    initial.gate_(qu.hadamard(), 0, contract=True)
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 3),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    optimizer = py.MpsOptimizer(initial, gates=[], chi=2, mode="dmrg2")
    guess, initialization = optimizer._build_randomized_fit_guess(
        initial,
        (0, 3),
        block_size=2,
        rand_strength=1.0e-4,
    )
    fit = py.FIT(
        target,
        p=guess,
        range_int=[0, 3],
        cutoffs=1.0e-12,
        inplace=True,
    )

    fit.run_gate(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=1.0e-12,
    )

    assert fit.p is guess
    assert initialization["enabled"] is True
    assert [initial.bond_size(i, i + 1) for i in range(3)] == [1, 1, 1]
    assert [fit.p.bond_size(i, i + 1) for i in range(3)] == [2, 2, 2]
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)


@pytest.mark.parametrize("block_size", (2, 3))
def test_fit_run_eff_native_blocks_grow_full_chain(block_size):
    """Full-chain block FIT should grow only bonds supported by the target."""
    initial = qtn.MPS_computational_state("00000", dtype="complex128")
    target = initial.copy()
    target.gate_(qu.hadamard(), 2, contract=True)
    target.gate_(
        qu.CNOT(),
        (2, 3),
        contract="split",
        max_bond=2,
        cutoff=0.0,
    )
    fit = py.FIT(
        target,
        p=initial,
        cutoffs=1.0e-12,
        contraction_opt="greedy",
    )

    fit.run_eff(
        n_iter=2,
        verbose=True,
        block_size=block_size,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=1.0e-12,
    )

    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)
    assert [fit.p.bond_size(site, site + 1) for site in range(4)] == [1, 1, 2, 1]
    assert len(fit.fidelity_trace) == 2
    split_key = "two_site_splits" if block_size == 2 else "three_site_splits"
    assert fit.info[split_key]


@pytest.mark.parametrize("block_size", (2, 3))
def test_fit_run_eff_adaptive_block_warmup_then_one_site_refinement(block_size):
    """Full-chain run_eff can switch from block growth to one-site updates."""
    initial = qtn.MPS_computational_state("00000", dtype="complex128")
    target = initial.copy()
    target.gate_(qu.hadamard(), 2, contract=True)
    target.gate_(
        qu.CNOT(),
        (2, 3),
        contract="split",
        max_bond=2,
        cutoff=0.0,
    )
    fit = py.FIT(target, p=initial, cutoffs=0.0)

    fit.run_eff(
        n_iter=4,
        block_size=block_size,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
    )

    assert fit.iterations_run == 4
    assert fit.adaptive_sweeps_run == 2
    assert fit.one_site_sweeps_run == 2
    assert fit._sweep_environment_reuse_count == 3
    split_key = "two_site_splits" if block_size == 2 else "three_site_splits"
    assert fit.info[split_key]
    assert fit.p.bond_size(2, 3) == 2


@pytest.mark.parametrize("block_size", (2, 3))
@pytest.mark.parametrize("sweep_sequence", ("RL", "LR"))
def test_fit_run_eff_transition_cache_matches_rebuild(
    block_size,
    sweep_sequence,
):
    """Block-to-one-site cache extensions preserve the rebuilt result."""
    initial = qtn.MPS_rand_state(
        5,
        bond_dim=1,
        phys_dim=2,
        dtype="complex128",
        seed=610,
    )
    target = qtn.MPS_rand_state(
        5,
        bond_dim=3,
        phys_dim=2,
        dtype="complex128",
        seed=611,
    )
    options = {
        "n_iter": 3,
        "block_size": block_size,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 3,
        "cutoff": 1.0e-12,
    }
    cached = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt._allow_sweep_environment_reuse = False

    cached.run_eff(**options)
    rebuilt.run_eff(**options)

    assert cached._sweep_environment_reuse_count == 2
    assert rebuilt._sweep_environment_reuse_count == 0
    assert np.allclose(
        cached.p.to_dense(),
        rebuilt.p.to_dense(),
        atol=1.0e-12,
    )


def test_fit_run_eff_adaptive_rtol_waits_for_one_site_phase():
    """Adaptive run_eff resets tolerance at the block-to-one-site boundary."""
    initial = qtn.MPS_computational_state("00000", dtype="complex128")
    target = initial.copy()
    target.gate_(qu.hadamard(), 2, contract=True)
    target.gate_(
        qu.CNOT(),
        (2, 3),
        contract="split",
        max_bond=2,
        cutoff=0.0,
    )
    fit = py.FIT(target, p=initial, cutoffs=0.0)

    fit.run_eff(
        n_iter=5,
        block_size=2,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        rtol=1.0,
        patience=2,
    )

    assert fit.iterations_run >= 3
    assert fit.adaptive_sweeps_run == 2
    assert fit.one_site_sweeps_run >= 1
    assert len(fit.sweep_norm_trace) == fit.iterations_run
    assert fit.convergence_reason in {"relative_tolerance", "max_sweeps"}


def test_fit_run_eff_default_keeps_fixed_rank_one_site_compatibility():
    """The default full-chain path remains a fixed-rank one-site solver."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, cutoffs=0.0)

    fit.run_eff(n_iter=2)

    assert fit.p.max_bond() == 1
    assert "two_site_splits" not in fit.info
    assert "three_site_splits" not in fit.info


@pytest.mark.parametrize("sweep_sequence", ("RL", "LR"))
def test_fit_run_eff_fixed_one_site_reuses_opposite_sweep_cache(
    sweep_sequence,
):
    """Fixed-sweep one-site run_eff reuses compatible dense environments."""
    initial = qtn.MPS_rand_state(
        5,
        bond_dim=1,
        phys_dim=2,
        dtype="complex128",
        seed=612,
    )
    target = qtn.MPS_rand_state(
        5,
        bond_dim=3,
        phys_dim=2,
        dtype="complex128",
        seed=613,
    )
    options = {
        "n_iter": 4,
        "sweep_sequence": sweep_sequence,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt = py.FIT(target, p=initial, cutoffs=1.0e-12)
    rebuilt._allow_sweep_environment_reuse = False

    cached.run_eff(**options)
    rebuilt.run_eff(**options)

    assert cached._sweep_environment_reuse_count == 3
    assert rebuilt._sweep_environment_reuse_count == 0
    assert len(cached.local_norm_trace) == 4
    assert np.allclose(
        cached.p.to_dense(),
        rebuilt.p.to_dense(),
        atol=1.0e-12,
    )


def test_fit_run_eff_one_site_default_alternates_directions():
    """Default run_eff sweeps left-to-right and then right-to-left."""
    initial = qtn.MPS_computational_state("000", dtype="complex128")
    fit = py.FIT(initial.copy(), p=initial, cutoffs=0.0)

    fit.run_eff(n_iter=2)

    assert fit.iterations_run == 2
    assert fit.final_direction == "L"
    assert fit.final_center_site == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_iter": 1, "rtol": 1.0e-8}, "n_iter >= 2"),
        ({"n_iter": 2, "rtol": 1.0e-8, "min_iter": 1}, "min_iter >= 2"),
    ],
)
def test_fit_run_eff_rtol_requires_two_sweeps(kwargs, match):
    """Adaptive run_eff must have two retained norms to compare."""
    initial = qtn.MPS_computational_state("000", dtype="complex128")
    fit = py.FIT(initial.copy(), p=initial, cutoffs=0.0)

    with pytest.raises(ValueError, match=match):
        fit.run_eff(**kwargs)


def test_fit_run_eff_three_site_requires_three_sites():
    """A three-site full-chain update needs a sufficiently long chain."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    fit = py.FIT(state.copy(), p=state)

    with pytest.raises(ValueError, match="at least three sites"):
        fit.run_eff(block_size=3)


def test_fit_gate_three_site_native_splits_and_keeps_outside_bonds():
    """Three-site FIT should use two native splits within the active window."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        collect_split_diagnostics=True,
        n_iter=2,
        block_size=3,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        three_site_sweeps=2,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)
    assert fit.p.bond_size(3, 4) == 1
    assert len(fit.info["three_site_splits"]) == 4
    assert all(
        len(record["truncation_errors"]) == 2
        for record in fit.info["three_site_splits"]
    )
    timing = fit.get_timing()
    assert [record["direction"] for record in timing] == ["R", "L"]
    assert all(record["block_size"] == 3 for record in timing)
    assert all(
        len(site_timing["sites"]) == 3
        for record in timing
        for site_timing in record["site_timings"]
    )


def test_fit_gate_three_site_warmup_then_one_site_refinement():
    """Three-site warm-up should switch to one-site polishing sweeps."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        adaptive_block_sweeps=None, two_site_transition_sweeps=0,
        collect_split_diagnostics=True,
        n_iter=3,
        block_size=3,
        three_site_sweeps=1,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        timing=True,
    )

    assert [record["block_size"] for record in fit.get_timing()] == [3, 1, 1]
    assert len(fit.info["three_site_splits"]) == 2
    # The 3->1 transition extends two terminal boundaries instead of
    # rebuilding the fixed side; the following 1->1 sweep reuses normally.
    assert fit._sweep_environment_reuse_count == 2


def test_fit_gate_polish_sweeps_update_iteration_diagnostics():
    """Explicit one-site polish sweeps count in FIT diagnostics."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        adaptive_block_sweeps=None, two_site_transition_sweeps=0, rtol=None,
        n_iter=1,
        block_size=3,
        three_site_sweeps=1,
        final_one_site_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        timing=True,
    )

    assert fit.iterations_run == 3
    assert fit.adaptive_sweeps_run == 1
    assert fit.one_site_sweeps_run == 2
    assert [record["sweep"] for record in fit.get_timing()] == [1, 2, 3]


def test_fit_gate_two_site_warmup_then_one_site_refinement():
    """Two-site warm-up should switch to fixed-rank one-site sweeps."""
    initial, target = _three_site_ghz_target()
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        collect_split_diagnostics=True,
        n_iter=4,
        block_size=2,
        adaptive_block_sweeps=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
        timing=True,
    )

    timing = fit.get_timing()
    assert [record["block_size"] for record in timing] == [2, 2, 1, 1]
    assert len(fit.info["two_site_splits"]) == 6
    assert fit._sweep_environment_reuse_count == 3
    assert all(
        record["svd_seconds"] == 0.0
        for record in timing[2:]
    )


def test_fit_adaptive_rank_targets_follow_open_chain_capacity():
    """Adaptive FIT should use attainable 2, 4, 8, ... bond ceilings."""
    state = qtn.MPS_computational_state("00000000", dtype="complex128")

    assert py.FIT._active_bond_rank_targets(  # pylint: disable=protected-access
        state,
        0,
        7,
        16,
    ) == (2, 4, 8, 16, 8, 4, 2)

    optimizer = py.MpsOptimizer(state, gates=[], chi=16, mode="dmrg1")
    assert optimizer._mix_target_bond_dimensions() == [2, 4, 8, 16, 8, 4, 2]


def test_dmrg1_leaves_adaptive_phase_after_two_sweeps_on_rank_stagnation():
    """DMRG1 does not extend its two-site phase when rank growth stalls."""
    state = qtn.MPS_computational_state("000", dtype="complex128")
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=6,
        cutoff=1.0e-12,
        fit_adaptive_sweeps=6,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1, 1, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 4
    assert optimizer._last_dmrg_fit_diagnostics["dmrg1_one_site_locked"] is False


@pytest.mark.parametrize("n_iter", [1, 2])
def test_dmrg1_growth_requires_room_for_one_site_refinement(n_iter):
    """DMRG1 growth needs two block sweeps plus one refinement sweep."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    with pytest.raises(ValueError, match="n_iter >= 3"):
        optimizer.run(progbar=False, n_iter=n_iter, fit_rtol=None)


def test_dmrg1_under_capacity_grows_twice_then_refines():
    """DMRG1 grows an under-capacity window twice before refinement."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    bell_gate = cnot @ np.kron(hadamard, np.eye(2))
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(bell_gate, (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 1


@pytest.mark.parametrize("fit_mpo_guess", [True, False])
def test_dmrg1_optional_svd_guess(fit_mpo_guess):
    """DMRG1 can toggle the legacy switch for the direct-SVD guess."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    gate = cnot @ np.kron(hadamard, np.eye(2))
    state = qtn.MPS_computational_state("000", dtype="complex128")
    stream = [(gate, (0, 2))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0, stabilize_unitary=False)
    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="dmrg1",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        stabilize_unitary=False,
        fit_mpo_guess=fit_mpo_guess,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    assert (
        optimizer.get_fit_diagnostics()["mpo_fit_guess_used"]
        is fit_mpo_guess
    )
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1]


@pytest.mark.parametrize("fit_mpo_guess", [True, False])
def test_dmrg3_optional_svd_guess(fit_mpo_guess):
    """DMRG3 can toggle the legacy switch for the direct-SVD guess."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    gate = cnot @ np.kron(hadamard, np.eye(2))
    state = qtn.MPS_computational_state("000", dtype="complex128")
    stream = [(gate, (0, 2))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0, stabilize_unitary=False)
    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="dmrg3",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        stabilize_unitary=False,
        fit_mpo_guess=fit_mpo_guess,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    assert (
        optimizer.get_fit_diagnostics()["mpo_fit_guess_used"]
        is fit_mpo_guess
    )
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [3, 3, 2]


def test_dmrg1_latches_one_site_phase_after_full_chain_saturation():
    """After filling all bonds, later DMRG1 windows stay one-site."""
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    cnot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    bell_gate = cnot @ np.kron(hadamard, np.eye(2))
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[
            (bell_gate, (0, 2)),
            (np.eye(4), (0, 2)),
        ],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=0.0,
        fit_rtol=None,
        timing=True,
    )

    records = optimizer.get_run_timing()["fit_steps"]
    assert [record["block_size"] for record in records] == [2, 2, 1, 1, 1, 1]
    assert [record["fit_index"] for record in records] == [0, 0, 0, 1, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["dmrg1_one_site_locked"] is True


def test_dmrg1_already_at_ceiling_starts_with_one_site_sweeps():
    """A full-rank DMRG1 window should not repeat two-site warm-up."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=123,
    )
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4, dtype=np.complex128), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=1.0e-12,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [1, 1, 1]
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["adaptive_sweeps"] == 0
    assert diagnostics["one_site_refinement_sweeps"] == 3
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["guess_used"] is True
    assert optimizer._last_dmrg_fit_diagnostics["dmrg1_one_site_locked"] is True


def test_dmrg1_reopens_block_warmup_for_rank_preserving_nonlocal_target():
    """DMRG1 must rotate saturated subspaces for a nonlocal gate."""
    state = (
        qtn.MPS_computational_state("00000000", dtype="complex128")
        + qtn.MPS_computational_state("11111111", dtype="complex128")
    ) / np.sqrt(2.0)
    controlled_phase = np.diag([1.0, 1.0, 1.0, -1.0]).astype("complex128")
    stream = [(controlled_phase, (0, 7))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="mpo",
    ).run(progbar=False, cutoff=1.0e-12, stabilize_unitary=False)
    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=2,
        mode="dmrg1",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=6,
        cutoff=1.0e-12,
        target_cutoff=1.0e-12,
        fit_rtol=None,
        stabilize_unitary=False,
        timing=True,
    )

    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-12)
    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["random_initialization"]["enabled"] is False
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [1, 1, 1, 1, 1, 1]


def test_dmrg1_default_ftol_window_uses_two_one_site_samples():
    """The default window of two stops after two stable one-site norms."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=124,
    )
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4, dtype=np.complex128), (0, 2))],
        chi=2,
        mode="dmrg1",
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=1.0e9,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["iterations"] == 2
    assert (
        optimizer._last_dmrg_fit_diagnostics["convergence_reason"]
        == "relative_tolerance"
    )


def test_dmrg2_switches_after_required_two_site_warmup():
    """DMRG2 uses two sites twice, then one-site refinement."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=123,
    )
    optimizer = py.MpsOptimizer(
        state,
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        cutoff=1.0e-12,
        fit_rtol=None,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 1


def test_dmrg2_rtol_can_stop_after_two_site_warmup():
    """DMRG2 tolerance stopping starts only after its two-site phase."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=1.0e9,
        fit_patience=1,
        cutoff=1.0e-12,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2


def test_dmrg3_rtol_can_stop_after_three_site_warmup():
    """DMRG3 tolerance stopping starts only after its three-site phase."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(np.eye(4), (0, 2))],
        chi=2,
        mode="dmrg3",
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=1.0e9,
        fit_patience=1,
        cutoff=1.0e-12,
        timing=True,
    )

    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [3, 3, 2, 1, 1]
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["adaptive_sweeps"] == 3
    assert diagnostics["one_site_refinement_sweeps"] == 2
    assert diagnostics["convergence_reason"] == "relative_tolerance"


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_fit_gate_large_window_block_sizes_compare_with_timing(block_size):
    """Large active windows compare rank growth and expose benchmark stages."""
    length = 14
    max_bond = 8
    initial = qtn.MPS_rand_state(
        length,
        bond_dim=1,
        phys_dim=2,
        dtype="complex128",
        seed=921,
    )
    target = qtn.MPS_rand_state(
        length,
        bond_dim=max_bond,
        phys_dim=2,
        dtype="complex128",
        seed=922,
    )
    fit = py.FIT(target, p=initial, range_int=[0, length - 1], cutoffs=1.0e-10)

    fit.run_gate(
        adaptive_block_sweeps=None, two_site_transition_sweeps=0,
        n_iter=2,
        block_size=block_size,
        sweep_sequence="RL",
        max_bond=max_bond,
        cutoff=1.0e-10,
        timing=True,
    )

    fidelity = float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    )
    records = fit.get_timing()
    assert [record["direction"] for record in records] == ["R", "L"]
    assert all(record["timing_schema"] == 3 for record in records)
    assert all(record["active_site_count"] == length for record in records)
    expected_blocks = [block_size, block_size]
    if block_size == 3:
        expected_blocks = [3, 1]
    assert [record["block_size"] for record in records] == expected_blocks
    assert [record["update_count"] for record in records] == [
        length - active_block_size + 1
        for active_block_size in expected_blocks
    ]
    assert all(
        all(
            site_timing[stage] >= 0.0
            for site_timing in record["site_timings"]
            for stage in (
                "effective_seconds",
                "svd_seconds",
                "writeback_seconds",
                "environment_seconds",
            )
        )
        for record in records
    )
    assert all(
        record["svd_seconds"]
        == pytest.approx(
            sum(site_timing["svd_seconds"] for site_timing in record["site_timings"])
        )
        for record in records
    )

    if block_size == 1:
        assert fit.p.max_bond() == 1
        assert fidelity < 0.1
    elif block_size == 2:
        assert 1 < fit.p.max_bond() < max_bond
        assert 0.4 < fidelity < 0.9
    else:
        # The default three-site schedule uses one warm-up sweep followed by
        # one-site refinement, so the refinement sweep cannot open additional
        # bonds to reach the old two-three-site-sweep rank.
        assert 1 < fit.p.max_bond() < max_bond
        assert fidelity > 0.5


def test_fit_gate_three_site_direct_and_generic_routes_match():
    """Three-site dense direct environments must match the generic route."""
    initial, target = _three_site_ghz_target()
    options = {
        "n_iter": 2,
        "block_size": 3,
        "sweep_sequence": "RL",
        "max_bond": 2,
        "cutoff": 0.0,
    }
    direct = py.FIT(
        target,
        p=initial,
        range_int=[0, 3],
        environment_strategy="mps-direct",
    )
    generic = py.FIT(
        target,
        p=initial,
        range_int=[0, 3],
        environment_strategy="generic",
    )

    direct.run_gate(**options)
    generic.run_gate(**options)

    assert np.allclose(
        direct.p.to_dense(),
        generic.p.to_dense(),
        atol=1.0e-10,
    )


def test_fit_dense_direct_environment_matches_generic_route():
    """The dense MPS specialization must preserve the generic FIT result."""
    initial = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=208
    )
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 2),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    direct = py.FIT(
        target,
        p=initial,
        range_int=[0, 2],
        environment_strategy="mps-direct",
    )
    generic = py.FIT(
        target,
        p=initial,
        range_int=[0, 2],
        environment_strategy="generic",
    )

    options = {
        "n_iter": 2,
        "block_size": 2,
        "sweep_sequence": "RL",
        "max_bond": 3,
        "cutoff": 1.0e-12,
    }
    direct.run_gate(**options)
    generic.run_gate(**options)

    assert direct.environment_strategy == "mps-direct"
    assert generic.environment_strategy == "generic"
    assert np.allclose(direct.p.to_dense(), generic.p.to_dense(), atol=1.0e-11)


def test_fit_gate_reuses_dense_opposite_sweep_environments():
    """Dense R/L sweeps reuse only compatible cached boundary environments."""
    initial = qtn.MPS_rand_state(
        10, bond_dim=1, phys_dim=2, dtype="complex128", seed=601
    )
    target = qtn.MPS_rand_state(
        10, bond_dim=3, phys_dim=2, dtype="complex128", seed=602
    )
    options = {
        "n_iter": 4,
        "block_size": 2,
        "sweep_sequence": "RL",
        "max_bond": 3,
        "cutoff": 1.0e-12,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, range_int=[0, 9])
    uncached = py.FIT(target, p=initial, range_int=[0, 9])
    uncached._allow_sweep_environment_reuse = False

    cached.run_gate(**options)
    uncached.run_gate(**options)

    assert cached._sweep_environment_reuse_count == 3
    assert uncached._sweep_environment_reuse_count == 0
    cached_dense = np.asarray(cached.p.to_dense()).reshape(-1)
    uncached_dense = np.asarray(uncached.p.to_dense()).reshape(-1)
    overlap = np.vdot(cached_dense, uncached_dense)
    assert abs(overlap) ** 2 == pytest.approx(
        np.vdot(cached_dense, cached_dense).real
        * np.vdot(uncached_dense, uncached_dense).real,
        rel=1.0e-9,
        abs=1.0e-12,
    )


@pytest.mark.parametrize("direction", ["R", "L"])
@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_fit_gate_builds_only_fixed_environments_reachable_by_block(
    monkeypatch,
    direction,
    block_size,
):
    """Fresh sweeps contract only boundaries reachable by their blocks."""
    initial = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=603
    )
    target = qtn.MPS_rand_state(
        4, bond_dim=3, phys_dim=2, dtype="complex128", seed=604
    )
    fit = py.FIT(target, p=initial, range_int=[0, 3])
    overlap_calls = 0
    original = fit._overlap_environment_site

    def count_overlap(*args, **kwargs):
        nonlocal overlap_calls
        overlap_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "_overlap_environment_site", count_overlap)
    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        block_size=block_size,
        sweep_sequence=direction,
        max_bond=3,
        cutoff=1.0e-12,
    )

    window_size = 4
    fixed_count = window_size - block_size
    moving_count = (
        window_size - 1
        if block_size == 1
        else window_size - block_size
    )
    assert overlap_calls == fixed_count + moving_count


@pytest.mark.parametrize("direction", ["R", "L"])
def test_fit_single_pair_fast_path_builds_no_active_environments(
    monkeypatch,
    direction,
):
    """A terminal update covering the full window needs no active cache."""
    initial = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=605
    )
    fit = py.FIT(initial.copy(), p=initial, range_int=[1, 2])
    overlap_calls = 0
    original = fit._overlap_environment_site

    def count_overlap(*args, **kwargs):
        nonlocal overlap_calls
        overlap_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "_overlap_environment_site", count_overlap)
    fit.run_gate(
        n_iter=7,
        block_size=2,
        sweep_sequence=direction,
        max_bond=2,
        rtol=None,
        single_pair_fast_path=True,
    )

    assert overlap_calls == 0


@pytest.mark.parametrize("sweep_sequence", ["RL", "LR"])
def test_fit_reuses_reversed_two_site_cache_for_one_site_refinement(
    sweep_sequence,
):
    """The final two-site boundaries exactly serve reversed one-site FIT."""
    initial = qtn.MPS_rand_state(
        5, bond_dim=1, phys_dim=2, dtype="complex128", seed=606
    )
    target = qtn.MPS_rand_state(
        5, bond_dim=3, phys_dim=2, dtype="complex128", seed=607
    )
    options = {
        "n_iter": 3,
        "block_size": 2,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 3,
        "cutoff": 1.0e-12,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, range_int=[0, 4])
    conservative = py.FIT(target, p=initial, range_int=[0, 4])
    overlap_calls = {"cached": 0, "conservative": 0}
    cached_overlap = cached._overlap_environment_site
    conservative_overlap = conservative._overlap_environment_site

    def count_cached(*args, **kwargs):
        overlap_calls["cached"] += 1
        return cached_overlap(*args, **kwargs)

    def count_conservative(*args, **kwargs):
        overlap_calls["conservative"] += 1
        return conservative_overlap(*args, **kwargs)

    cached._overlap_environment_site = count_cached
    conservative._overlap_environment_site = count_conservative
    conservative._allow_sweep_environment_reuse = False

    cached.run_gate(**options)
    conservative.run_gate(**options)

    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert overlap_calls == {"cached": 14, "conservative": 20}
    assert np.allclose(
        cached.p.to_dense(),
        conservative.p.to_dense(),
        atol=1.0e-12,
    )


@pytest.mark.parametrize("sweep_sequence", ["RL", "LR"])
def test_fit_reuses_reversed_three_site_cache_for_one_site_refinement(
    sweep_sequence,
):
    """Three-site FIT extends only two terminal boundaries before 1-site."""
    initial = qtn.MPS_rand_state(
        5, bond_dim=1, phys_dim=2, dtype="complex128", seed=608
    )
    target = qtn.MPS_rand_state(
        5, bond_dim=3, phys_dim=2, dtype="complex128", seed=609
    )
    options = {
        "two_site_transition_sweeps": 0,
        "n_iter": 3,
        "block_size": 3,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 3,
        "cutoff": 1.0e-12,
        "rtol": None,
    }
    cached = py.FIT(target, p=initial, range_int=[0, 4])
    conservative = py.FIT(target, p=initial, range_int=[0, 4])
    overlap_calls = {"cached": 0, "conservative": 0}
    cached_overlap = cached._overlap_environment_site
    conservative_overlap = conservative._overlap_environment_site

    def count_cached(*args, **kwargs):
        overlap_calls["cached"] += 1
        return cached_overlap(*args, **kwargs)

    def count_conservative(*args, **kwargs):
        overlap_calls["conservative"] += 1
        return conservative_overlap(*args, **kwargs)

    cached._overlap_environment_site = count_cached
    conservative._overlap_environment_site = count_conservative
    conservative._allow_sweep_environment_reuse = False

    cached.run_gate(**options)
    conservative.run_gate(**options)

    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert overlap_calls == {"cached": 12, "conservative": 16}
    assert np.allclose(
        cached.p.to_dense(),
        conservative.p.to_dense(),
        atol=1.0e-12,
    )


def test_fit_auto_cutoff_is_dtype_aware():
    """FIT's automatic cutoff follows the fitted tensor precision."""
    initial = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex64", seed=209
    )
    target = initial.copy(deep=True)
    target.gate_nonlocal_(qu.CNOT(), (0, 2), max_bond=None, cutoff=0.0)
    fit = py.FIT(target, p=initial, range_int=[0, 2])

    fit.run_gate(adaptive_block_sweeps=None, n_iter=1, block_size=2, cutoff="auto")

    assert fit.info["cutoff_requested"] == "auto"
    assert fit.info["cutoff_resolved"] == pytest.approx(1.0e-6)


@pytest.mark.parametrize(
    ("dtype", "expected_cutoff"),
    [("complex64", 1.0e-6), ("complex128", 1.0e-12)],
)
def test_mps_optimizer_default_cutoff_policy_is_dtype_aware(
    monkeypatch,
    dtype,
    expected_cutoff,
):
    """An omitted MPS run cutoff resolves from the live tensor dtype."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0", dtype=dtype),
        gates=[(qu.hadamard(), (0,))],
        chi=2,
        mode="svd",
    )
    calls = {}

    def record_execute(*args, **kwargs):
        calls.update(kwargs)
        return optimizer.p

    monkeypatch.setattr(optimizer, "_execute_mode", record_execute)
    optimizer.run(progbar=False)

    assert calls["cutoff"] == pytest.approx(expected_cutoff)
    assert calls["cutoff_mode"] == "rsum2"
    assert calls["mpo_cutoff_mode"] == "rsum2"


def test_fit_retag_resolves_environment_and_preserves_info_object():
    """Retagging precedes route selection and caller diagnostics stay live."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=210
    )
    target = state.copy()
    target.drop_tags()
    info = {}

    fit = py.FIT(
        target,
        p=state,
        range_int=[0, 2],
        retag=True,
        info=info,
        environment_strategy="mps-direct",
    )
    fit.run_gate(adaptive_block_sweeps=None, collect_split_diagnostics=True, n_iter=1, block_size=2, max_bond=2)

    assert fit.environment_strategy == "mps-direct"
    assert fit.info is info
    assert info["two_site_splits"]


def test_fit_retag_keeps_layered_tensor_ownership_near_mps_sites():
    """Retagging uses nearest graph sites without reordering the target."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    bonds = [f"bond{i}" for i in range(3)]
    tensors = []

    for site in range(4):
        inds = []
        shape = []
        if site:
            inds.append(bonds[site - 1])
            shape.append(1)
        if site < 3:
            inds.append(bonds[site])
            shape.append(1)
        inds.append(state.site_ind(site))
        shape.append(2)
        if site == 1:
            inds.append("path-left")
            shape.append(1)
        if site == 2:
            inds.append("path-right")
            shape.append(1)
        tensors.append(qtn.Tensor(np.ones(shape), inds=inds))

    # These tensors have no physical legs. Their graph distances to sites 1
    # and 2 are deliberately different, so propagation must not simply walk
    # in the first tag's direction.
    tensors.extend(
        [
            qtn.Tensor(
                np.ones((1, 1)),
                inds=("path-left", "path-middle"),
            ),
            qtn.Tensor(
                np.ones((1, 1)),
                inds=("path-middle", "path-right"),
            ),
        ]
    )
    target = qtn.TensorNetwork(tensors)
    original_order = tuple(target.tensor_map)

    fit = py.FIT(target, p=state, retag=True)

    assert tuple(fit.tn.tensor_map) == original_order

    left_path = fit.tn.tensor_map[original_order[4]]
    right_path = fit.tn.tensor_map[original_order[5]]
    assert set(left_path.tags) == {"I1", "I2"}
    assert set(right_path.tags) == {"I1", "I2"}


def test_fit_retag_preserves_layered_mps_backbone_regions():
    """Canonical layered tags keep base MPS tensors on their original sites."""
    state = qtn.MPS_computational_state("0000", dtype="complex128")
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    target = optimizer._build_norm_target(
        state,
        qu.CNOT(),
        (1, 2),
        0.0,
        target_strategy="layered",
    )

    fit = py.FIT(target, p=state, retag=True)

    assert [set(tensor.tags) for tensor in fit.tn] == [
        {"I0"},
        {"I1"},
        {"I2"},
        {"I3"},
        {"I1"},
        {"I2"},
    ]


def test_fit_direct_environment_requires_unique_tensor_per_site():
    """A tensor carrying every site tag must not be cached multiple times."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=211
    )
    collapsed_target = qtn.TensorNetwork([state.contract(all)])

    automatic = py.FIT(
        collapsed_target,
        p=state,
        range_int=[0, 2],
    )
    assert automatic.environment_strategy == "generic"

    with pytest.raises(ValueError, match="exactly one target tensor per site"):
        py.FIT(
            collapsed_target,
            p=state,
            range_int=[0, 2],
            environment_strategy="mps-direct",
        )


def test_new_fit_configuration_is_keyword_only():
    """New policy controls must not extend the legacy positional API."""
    fit_parameters = inspect.signature(py.FIT).parameters
    run_parameters = inspect.signature(py.MpsOptimizer.run).parameters

    assert fit_parameters["environment_strategy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["n_iter"].default == 8
    assert run_parameters["cutoff"].default == "auto"
    assert run_parameters["cutoff_mode"].default == "auto"
    assert run_parameters["fit_rtol"].default == "auto"
    assert run_parameters["quality_check_every"].default is False
    assert run_parameters["fit_overlap_diagnostics"].default is False
    assert run_parameters["fit_init_rand_strength"].default == 0.0
    for name in (
        "fit_min_iter",
        "fit_rtol",
        "fit_patience",
        "fit_block_size",
        "fit_adaptive_sweeps",
        "fit_sweep_sequence",
        "fit_layer_size",
        "fit_max_span",
        "fit_three_site_sweeps",
        "target_cutoff",
        "fit_target_strategy",
        "fit_single_pair_fast_path",
        "fit_overlap_diagnostics",
        "stabilize_unitary",
        "fit_stabilize_unitary",
        "timing",
        "timing_sync_device",
        "quality_check_every",
        "quality_check_repair",
    ):
        assert run_parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_mps_optimizer_fit_defaults_are_adaptive_and_fixed_pair_sweeps():
    """The public DMRG defaults retain requested adjacent-pair sweeps."""
    fit_rtol = inspect.signature(py.MpsOptimizer.run).parameters["fit_rtol"]
    fast_path = inspect.signature(py.MpsOptimizer.run).parameters[
        "fit_single_pair_fast_path"
    ]

    assert fit_rtol.default == "auto"
    assert fast_path.default is False


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [("complex64", 1.0e-5), ("complex128", 1.0e-9)],
)
def test_mps_optimizer_auto_fit_rtol_tracks_state_dtype(dtype, expected):
    """The automatic FIT tolerance follows the live MPS precision."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype=dtype),
        gates=[],
        chi=2,
        mode="dmrg",
    )

    assert optimizer._resolve_fit_rtol("auto") == pytest.approx(expected)


def test_mps_optimizer_default_runs_requested_adjacent_pair_sweeps():
    """The default MPS optimizer path does not stop after one pair update."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(progbar=False, n_iter=2, fit_rtol=None, timing=True)

    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["iterations"] == 2
    assert diagnostics["convergence_reason"] != "single_pair_exact"


def test_mps_optimizer_dmrg2_adjacent_pair_defaults_to_one_update():
    """Named DMRG2 keeps its one-update schedule for neighboring gates."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg2",
    )

    optimizer.run(progbar=False, n_iter=8, fit_rtol=None, timing=True)

    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["iterations"] == 1
    assert diagnostics["convergence_reason"] == "single_pair_exact"


def test_fit_two_site_single_pair_fast_path_is_structurally_converged():
    """One variational pair needs one effective tensor and one native SVD."""
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=215
    )
    fit = py.FIT(state.copy(), p=state, range_int=[1, 2])

    fit.run_gate(
        n_iter=7,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        rtol=None,
        single_pair_fast_path=True,
    )

    assert fit.iterations_run == 1
    assert fit.converged is True
    assert fit.convergence_reason == "single_pair_exact"
    assert fit.last_relative_change == 0.0
    assert fit.final_center_site == 2
    assert fit.final_direction == "R"


@pytest.mark.parametrize("mode", ["dmrg", "dmrg1", "dmrg2", "dmrg3"])
def test_dmrg_modes_advance_after_one_update_per_two_site_window(mode):
    """A two-site window gets one exact update, independent of n_iter."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[
            (qu.hadamard(), (0,)),
            (qu.CNOT(), (0, 1)),
            (qu.CNOT(), (1, 2)),
        ],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        n_iter=8,
        fit_rtol=None,
        fit_single_pair_fast_path=True,
        timing=True,
    )

    records = optimizer.get_run_timing()["fit_steps"]
    assert [record["fit_index"] for record in records] == [0, 1]
    assert [record["sweep"] for record in records] == [1, 1]
    assert [record["block_size"] for record in records] == [2, 2]
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    assert diagnostics["iterations"] == 1
    assert diagnostics["block_size"] == 2
    assert diagnostics["adaptive_sweeps"] == 1
    assert diagnostics["one_site_refinement_sweeps"] == 0
    assert diagnostics["convergence_reason"] == "single_pair_exact"
    assert diagnostics["center_site"] == 2


def test_dense_layered_fit_target_matches_materialized_mps_target():
    """Lazy gate layers must preserve the exact uncompressed target state."""
    state = qtn.MPS_rand_state(
        5, bond_dim=3, phys_dim=2, dtype="complex128", seed=216
    )
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    gate = qu.CNOT()

    materialized = optimizer._build_norm_target(
        state,
        gate,
        (0, 4),
        0.0,
        target_strategy="mps",
    )
    layered = optimizer._build_norm_target(
        state,
        gate,
        (0, 4),
        0.0,
        target_strategy="layered",
    )

    assert layered.num_tensors == state.L + 2
    assert np.allclose(layered.to_dense(), materialized.to_dense(), atol=1.0e-12)


def test_layered_fit_resolves_boundary_bonds_locally_and_caches_them():
    """Layered boundary discovery must not scan the global target index map."""
    state = qtn.MPS_rand_state(
        6, bond_dim=2, phys_dim=2, dtype="complex128", seed=608
    )
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    target = optimizer._build_norm_target(
        state,
        qu.CNOT(),
        (2, 3),
        0.0,
        target_strategy="layered",
    )
    fit = py.FIT(
        target,
        p=state,
        range_int=[2, 3],
        copy_target=False,
    )

    class LocalOnlyIndexMap(dict):
        def items(self):
            raise AssertionError("layered FIT scanned the global index map")

    fit.tn.ind_map = LocalOnlyIndexMap(fit.tn.ind_map)
    fit.run_gate(
        n_iter=1,
        block_size=2,
        max_bond=4,
        cutoff=0.0,
        single_pair_fast_path=True,
    )

    assert set(fit._target_bond_cache) == {(1, 2), (3, 4)}
    assert fit._target_bond(1, 2) == fit._target_bond_cache[(1, 2)]
    assert fit._target_bond(3, 4) == fit._target_bond_cache[(3, 4)]


def test_layered_dmrg_batch_target_avoids_intermediate_mps_rank_growth():
    """A target block should add two small tensors per gate, not copy/split MPSs."""
    state = qtn.MPS_rand_state(
        5, bond_dim=2, phys_dim=2, dtype="complex128", seed=217
    )
    optimizer = py.MpsOptimizer(state.copy(), gates=[], chi=4, mode="dmrg")
    gates = [qu.CNOT(), qu.CNOT()]
    locations = [(0, 4), (1, 3)]

    layered = optimizer._build_dmrg_batch_target(
        state,
        gates,
        locations,
        0.0,
        target_strategy="layered",
    )
    materialized = optimizer._build_dmrg_batch_target(
        state,
        gates,
        locations,
        0.0,
        target_strategy="mps",
    )

    assert layered.num_tensors == state.L + 2 * len(gates)
    assert np.allclose(layered.to_dense(), materialized.to_dense(), atol=1.0e-12)


def test_unitary_fit_stabilization_reuses_known_center_norm(monkeypatch):
    """Stabilization should not canonicalize a FIT result a second time."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    optimizer = py.MpsOptimizer(state, gates=[], chi=2, mode="dmrg")
    state[1].modify(data=state[1].data * 0.5)

    def unexpected_canonicalization(*_args, **_kwargs):
        raise AssertionError("known FIT center should bypass canonicalization")

    monkeypatch.setattr(
        optimizer,
        "_canonical_span_norm",
        unexpected_canonicalization,
    )
    optimizer._stabilize_unitary_fit_state(
        state,
        (0, 1),
        1.0,
        current_norm=0.5,
        center_site=1,
    )

    assert float(np.real(state.norm())) == pytest.approx(1.0, abs=1.0e-12)
    assert optimizer._current_orthog(state) == (1, 1)


def test_fit_gate_two_site_timing_reports_pairs_and_directions():
    """Alternating two-site timing should identify every optimized pair."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=204
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        timing=True,
    )

    records = fit.get_timing()
    assert [record["direction"] for record in records] == ["R", "L"]
    assert all(record["block_size"] == 2 for record in records)
    assert all(record["site_count"] == 2 for record in records)
    assert all(
        len(site_timing["sites"]) == 2
        for record in records
        for site_timing in record["site_timings"]
    )
    assert all(
        {
            "effective_seconds",
            "svd_seconds",
            "writeback_seconds",
            "environment_seconds",
        }.issubset(site_timing)
        for record in records
        for site_timing in record["site_timings"]
    )


def test_fit_gate_two_site_final_polish_only_spans_large_windows():
    """Two-site FIT can polish a large window without touching a pair window."""
    state = qtn.MPS_rand_state(
        4, bond_dim=2, phys_dim=2, dtype="complex128", seed=205
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])
    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        final_one_site_sweeps=1,
        timing=True,
    )

    records = fit.get_timing()
    assert [record["block_size"] for record in records] == [2, 1]
    assert [record["direction"] for record in records] == ["R", "L"]
    assert [record["site_count"] for record in records] == [2, 3]
    assert fit._sweep_environment_reuse_count == 1

    pair = py.FIT(state.copy(), p=state, range_int=[1, 2])
    pair.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        block_size=2,
        final_one_site_sweeps=1,
        timing=True,
    )
    assert [record["block_size"] for record in pair.get_timing()] == [2]


def test_fit_alternating_sweeps_reuse_opposite_canonical_form(monkeypatch):
    """An R/L pair should not repeat the canonicalization boundary pass."""
    initial = qtn.MPS_rand_state(
        5, bond_dim=2, phys_dim=2, dtype="complex128", seed=220
    )
    target = initial.copy()
    target.gate_nonlocal_(
        qu.CNOT(),
        (0, 3),
        max_bond=None,
        method="direct",
        cutoff=0.0,
    )
    options = {
        "block_size": 2,
        "max_bond": 3,
        "cutoff": 1.0e-12,
    }

    # Two separate calls intentionally force the old, conservative
    # preparation pass before the L sweep and provide a numerical reference.
    reference = py.FIT(target, p=initial.copy(), range_int=[0, 3])
    reference.run_gate(adaptive_block_sweeps=None, n_iter=1, sweep_sequence="R", **options)
    reference.run_gate(adaptive_block_sweeps=None, n_iter=1, sweep_sequence="L", **options)

    counts = {"left": 0, "right": 0}
    original_left = qtn.MatrixProductState.left_canonize_site
    original_right = qtn.MatrixProductState.right_canonize_site

    def count_left(state, *args, **kwargs):
        counts["left"] += 1
        return original_left(state, *args, **kwargs)

    def count_right(state, *args, **kwargs):
        counts["right"] += 1
        return original_right(state, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "left_canonize_site", count_left)
    monkeypatch.setattr(qtn.MatrixProductState, "right_canonize_site", count_right)

    optimized = py.FIT(target, p=initial.copy(), range_int=[0, 3])
    optimized.run_gate(n_iter=2, sweep_sequence="RL", **options)

    # The first R sweep prepares sites 3, 2, and 1. The following L sweep
    # consumes the canonical form produced by the R sweep's SVDs directly.
    assert counts == {"left": 0, "right": 3}
    assert np.allclose(
        optimized.p.to_dense(),
        reference.p.to_dense(),
        atol=1.0e-10,
    )


def test_timing_synchronizer_waits_on_jax_stage_outputs():
    """JAX barriers follow new stage results rather than an old MPS leaf."""

    class FakeJaxArray:
        __module__ = "jaxlib._jax"

        def __init__(self):
            self.waits = 0

        def block_until_ready(self):
            self.waits += 1

    source = FakeJaxArray()
    result_a = FakeJaxArray()
    result_b = FakeJaxArray()
    synchronizer = fitting_local_module._BackendSynchronizer.from_value(source)

    assert synchronizer.backend == "jax"
    synchronizer.synchronize((result_a, result_b), fallback=source)
    assert result_a.waits == 1
    assert result_b.waits == 1
    assert source.waits == 0
    assert (
        fitting_local_module._BackendSynchronizer.from_value(np.ones(1))
        is None
    )


def test_dmrg_synchronized_timing_marks_device_complete_stages(monkeypatch):
    """Opt-in synchronized profiling should be visible in every timing layer."""
    synchronizations = []

    class RecordingSynchronizer:
        def synchronize(self, value, *, fallback=None):
            synchronizations.append((value, fallback))

    monkeypatch.setattr(
        py.FIT,
        "_make_backend_synchronizer",
        staticmethod(lambda _state: RecordingSynchronizer()),
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        timing=True,
        timing_sync_device=True,
    )

    timing = optimizer.get_run_timing()
    assert synchronizations
    assert any(value is not optimizer.p for value, _fallback in synchronizations)
    assert timing["timing_sync_device"] is True
    assert timing["fit_steps"][0]["timing_sync_device"] is True
    assert timing["stages"]["dmrg.stabilize"]["calls"] == 1


def test_fit_gate_timing_retains_failed_partial_sweep():
    """Profiling must keep work completed before a failed FIT validation."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=206
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    with pytest.raises(FloatingPointError, match="non-finite tensor data"):
        fit.run_gate(
            n_iter=2,
            block_size=2,
            max_bond=2,
            finite_check=lambda _state: False,
            timing=True,
        )

    records = fit.get_timing()
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["site_count"] == 2
    assert records[0]["error"].startswith("FloatingPointError:")


def test_dmrg_complex64_deep_unitary_stream_keeps_working_norm_stable():
    """Unitary FIT stabilization should prevent complex64 norm underflow."""
    state = qtn.MPS_computational_state("00", dtype="complex64")
    gates = [(qu.hadamard(dtype="complex64"), (0,)), (qu.CNOT(dtype="complex64"), (0, 1))] * 180
    optimizer = py.MpsOptimizer(
        state,
        gates=gates,
        chi=1,
        mode="dmrg",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=1,
        cutoff=0.0,
        target_cutoff=0.0,
        stabilize_unitary=True,
    )

    raw = out.copy()
    raw.exponent = 0.0
    assert py.MpsOptimizer._mps_data_is_finite(out)
    assert float(np.real(raw.norm())) == pytest.approx(1.0, abs=2.0e-5)


def test_unitary_norm_overshoot_tolerance_is_dtype_aware():
    """Float32 roundoff is tolerated without hiding larger overshoots."""
    complex64 = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex64"),
        gates=[],
        chi=2,
        mode="mpo",
    )
    complex128 = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=2,
        mode="mpo",
    )

    assert complex64._unitary_norm_overshoot_tolerance() == pytest.approx(
        128.0 * np.finfo(np.float32).eps
    )
    assert complex128._unitary_norm_overshoot_tolerance() == pytest.approx(
        1.0e-6
    )
    complex64._finite_check_enabled = True
    complex128._finite_check_enabled = True

    event = complex64._record_norm_event(
        "unitary_compression",
        expected_norm=1.0,
        observed_norm=np.sqrt(1.0 + 1.0e-5),
        where=(0, 1),
    )
    assert event["fidelity_raw"] == pytest.approx(1.0 + 1.0e-5)
    assert event["local_fidelity"] == pytest.approx(1.0)

    with pytest.raises(FloatingPointError, match="squared ratio"):
        complex64._record_norm_event(
            "unitary_compression",
            expected_norm=1.0,
            observed_norm=np.sqrt(1.0 + 2.0e-5),
            where=(0, 1),
        )

    with pytest.raises(FloatingPointError, match="squared ratio"):
        complex128._record_norm_event(
            "unitary_compression",
            expected_norm=1.0,
            observed_norm=np.sqrt(1.0 + 2.0e-6),
            where=(0, 1),
        )


def test_dmrg_torch_complex64_two_site_fit_grows_native_dense_bond():
    """Torch complex64 FIT should retain dtype while using two-site SVD."""
    torch = pytest.importorskip("torch")
    state = qtn.MPS_computational_state("00", dtype="complex64")
    state.apply_to_arrays(py.backend_torch(dtype=torch.complex64, device="cpu"))
    gates = [
        (
            torch.as_tensor(
                np.array(qu.hadamard(), copy=True), dtype=torch.complex64
            ),
            (0,),
        ),
        (
            torch.as_tensor(np.array(qu.CNOT(), copy=True), dtype=torch.complex64),
            (0, 1),
        ),
    ]
    optimizer = py.MpsOptimizer(
        state,
        gates=gates,
        chi=2,
        mode="fit",
    )

    out = optimizer.run(
        progbar=False,
        n_iter=2,
        cutoff=0.0,
        target_cutoff=0.0,
    )

    expected = np.zeros(4, dtype=np.complex64)
    expected[[0, 3]] = 1.0 / np.sqrt(2.0)
    assert out.max_bond() == 2
    assert all(tensor.data.dtype == torch.complex64 for tensor in out.tensors)
    assert np.allclose(out.to_dense().cpu().numpy().reshape(-1), expected, atol=2.0e-6)


def test_fit_gate_three_site_torch_complex64_uses_native_splits():
    """Torch complex64 three-site FIT should preserve the backend dtype."""
    torch = pytest.importorskip("torch")
    initial, target = _three_site_ghz_target()
    initial.apply_to_arrays(py.backend_torch(dtype=torch.complex64, device="cpu"))
    target.apply_to_arrays(py.backend_torch(dtype=torch.complex64, device="cpu"))
    fit = py.FIT(target, p=initial, range_int=[0, 3])

    fit.run_gate(
        n_iter=2,
        block_size=3,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=0.0,
    )

    assert all(tensor.data.dtype == torch.complex64 for tensor in fit.p.tensors)
    assert np.allclose(
        fit.p.to_dense().cpu().numpy(),
        target.to_dense().cpu().numpy(),
        atol=2.0e-5,
    )


def test_fit_gate_three_site_symmray_uses_native_block_splits():
    """Symmray three-site FIT should preserve charge and fermionic blocks."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=1,
        dtype="complex128",
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(
        adaptive_block_sweeps=None, collect_split_diagnostics=True,
        n_iter=1,
        block_size=3,
        sweep_sequence="R",
        max_bond=4,
        cutoff=0.0,
    )

    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in fit.p.tensors
    )
    assert fit.info["three_site_splits"][0]["truncation_errors"] == (
        0.0,
        0.0,
    )


def test_fit_symmray_native_environment_matches_generic_route():
    """Native Symmray environments preserve the generic FIT result."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0)),
        seed=2,
        dtype="complex128",
    )
    target = state.copy(deep=True)
    target[1].modify(data=target[1].data * 0.75)

    native = py.FIT(
        target,
        p=state.copy(deep=True),
        range_int=[0, 2],
        environment_strategy="symmray-native",
    )
    generic = py.FIT(
        target.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[0, 2],
        environment_strategy="generic",
    )
    native.run_gate(n_iter=2, block_size=2, sweep_sequence="RL", cutoff=0.0)
    generic.run_gate(n_iter=2, block_size=2, sweep_sequence="RL", cutoff=0.0)

    assert native.environment_strategy == "symmray-native"
    native_dense = np.asarray(native.p.to_dense().to_dense()).reshape(-1)
    generic_dense = np.asarray(generic.p.to_dense().to_dense()).reshape(-1)
    overlap = np.vdot(native_dense, generic_dense)
    assert abs(overlap) ** 2 == pytest.approx(
        np.vdot(native_dense, native_dense).real
        * np.vdot(generic_dense, generic_dense).real,
        rel=1.0e-9,
        abs=1.0e-14,
    )


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (0, 1, 0, 1, 0, 1, 0, 1)),
        (
            True,
            "U1U1",
            ((0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)),
        ),
    ],
)
def test_fit_symmray_native_environment_preserves_dummy_modes(
    block_size, spinful, symmetry, occupations
):
    """Long native sweeps preserve dummy modes without tensor densification."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        8,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=2,
        seed=7,
        dtype="complex128",
    )

    native = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[0, 7],
        environment_strategy="symmray-native",
    )
    generic = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[0, 7],
        environment_strategy="generic",
    )

    native.run_gate(
        n_iter=4,
        min_iter=1,
        rtol=None,
        patience=99,
        block_size=block_size,
        max_bond=4,
        cutoff=1.0e-12,
        sweep_sequence="R",
    )
    generic.run_gate(
        n_iter=4,
        min_iter=1,
        rtol=None,
        patience=99,
        block_size=block_size,
        max_bond=4,
        cutoff=1.0e-12,
        sweep_sequence="R",
    )

    native_dense = np.asarray(native.p.to_dense().to_dense()).reshape(-1)
    generic_dense = np.asarray(generic.p.to_dense().to_dense()).reshape(-1)
    overlap = np.vdot(native_dense, generic_dense)
    assert abs(overlap) ** 2 == pytest.approx(
        np.vdot(native_dense, native_dense).real
        * np.vdot(generic_dense, generic_dense).real,
        rel=1.0e-9,
        abs=1.0e-14,
    )
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        for tensor in native.p.tensors
    )


def test_fit_auto_selects_native_symmray_environment():
    """Automatic FIT routing uses the native Symmray environment path."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=False, symmetry="U1", dtype="complex128")
    state = py.ps_to_mps(
        3,
        fermion=fermion,
        occupations=(1, 0, 1),
        seed=3,
        dtype="complex128",
    )
    fit = py.FIT(state.copy(deep=True), p=state, range_int=[0, 2])

    assert fit.environment_strategy == "symmray-native"


@pytest.mark.parametrize("block_size", [1, 2, 3])
@pytest.mark.parametrize("sweep_sequence", ["R", "RL"])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (1, 0, 1, 0, 1, 0)),
        (
            True,
            "U1U1",
            ((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        ),
        (False, "Z2", (1, 0, 1, 0, 1, 0)),
    ],
)
def test_fit_fermionic_native_writeback_gauge_is_phase_safe(
    block_size,
    sweep_sequence,
    spinful,
    symmetry,
    occupations,
):
    """Every native FIT block size preserves an exact graded MPS target."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=2,
        seed=17,
        dtype="complex128",
    )
    fit = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )
    run_options = {
        "n_iter": 2,
        "min_iter": 1,
        "rtol": None,
        "block_size": block_size,
        "sweep_sequence": sweep_sequence,
        "max_bond": 8,
        "cutoff": 1.0e-12,
    }
    if block_size > 1:
        run_options["adaptive_block_sweeps"] = 2
    fit.run_gate(**run_options)

    assert fit.info["fermionic_sweep_sequence"] == {
        "requested": sweep_sequence,
        "used": sweep_sequence,
        "reason": "native_conjugated_fit_gauge",
    }
    assert fit._sweep_environment_reuse_count == (1 if sweep_sequence == "RL" else 0)
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, state, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("block_size", [2, 3])
def test_fit_run_eff_fermionic_blocks_alternate_natively(block_size):
    """Full-chain native blocks reuse the conjugated RL environment gauge."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        chi=2,
        random_rounds=2,
        seed=31,
        dtype="complex128",
    )
    fit = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        environment_strategy="symmray-native",
    )
    fit.run_eff(
        n_iter=2,
        block_size=block_size,
        sweep_sequence="RL",
        max_bond=8,
        cutoff=1.0e-12,
    )

    assert fit._sweep_environment_reuse_count == 1
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, state, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize("sweep_sequence", ["RL", "LR"])
def test_fit_fermionic_block_cache_feeds_one_site_refinement(
    monkeypatch,
    sweep_sequence,
    block_size,
):
    """Native U1U1 fermions reuse reversed block-to-1 caches natively."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        chi=2,
        random_rounds=2,
        seed=61,
        dtype="complex128",
    )
    cached = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )
    conservative = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )
    conservative._allow_sweep_environment_reuse = False
    options = {
        "n_iter": 3,
        "block_size": block_size,
        "adaptive_block_sweeps": 2,
        "sweep_sequence": sweep_sequence,
        "max_bond": 8,
        "cutoff": 1.0e-12,
        "rtol": None,
    }

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native cache reuse must not call to_dense")

    array_types = {type(tensor.data) for tensor in state}
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        cached.run_gate(two_site_transition_sweeps=0, **options)
        conservative.run_gate(two_site_transition_sweeps=0, **options)

    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in cached.p
    )
    assert float(
        np.real(
            py.tn_fidelity(
                cached.p,
                conservative.p,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize(
    ("symmetry", "phys_dim", "occupations", "block_size", "sweep_sequence"),
    [
        ("U1", {0: 1, 1: 1}, [1, 0, 1, 0, 1, 0], 2, "RL"),
        (
            "U1U1",
            {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1},
            [(1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)],
            3,
            "LR",
        ),
        ("Z2", {0: 1, 1: 1}, [0, 0, 0, 0, 0, 0], 1, "RL"),
    ],
)
def test_fit_bosonic_symmray_reuses_native_reversed_environments(
    monkeypatch,
    symmetry,
    phys_dim,
    occupations,
    block_size,
    sweep_sequence,
):
    """Native bosonic caches match conservative rebuilds without densifying."""
    pytest.importorskip("symmray")
    common = {
        "L": 6,
        "symmetry": symmetry,
        "phys_dim": phys_dim,
        "site_charge": py.site_charge_from_occupations(occupations),
        "bond_dim": 3,
        "dtype": "complex128",
    }
    guess = py.SymMPS.random(seed=71, **common).tn
    target = py.SymMPS.random(seed=72, **common).tn
    cached = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
    )
    conservative = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
        environment_strategy="symmray-native",
    )
    generic = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
        environment_strategy="generic",
    )
    conservative._allow_sweep_environment_reuse = False
    options = {
        "n_iter": 3,
        "min_iter": 1,
        "rtol": None,
        "block_size": block_size,
        "sweep_sequence": sweep_sequence,
        "max_bond": 4,
        "cutoff": 1.0e-12,
    }
    if block_size > 1:
        options["adaptive_block_sweeps"] = 2

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native bosonic environment reuse must stay sparse")

    array_types = {type(tensor.data) for tensor in (*guess.tensors, *target.tensors)}
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        cached.run_gate(two_site_transition_sweeps=0, **options)
        conservative.run_gate(two_site_transition_sweeps=0, **options)

    assert cached.environment_strategy == "symmray-native"
    assert cached._allow_sweep_environment_reuse is True
    assert generic._allow_sweep_environment_reuse is False
    assert cached._sweep_environment_reuse_count == 2
    assert conservative._sweep_environment_reuse_count == 0
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and not type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in cached.p
    )
    assert float(
        np.real(
            py.tn_fidelity(
                cached.p,
                conservative.p,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=1.0e-10)


def test_fit_fermionic_failure_restores_physical_ket(monkeypatch):
    """A failed native sweep cannot leak FIT's conjugated working gauge."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=False,
        symmetry="U1",
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        5,
        fermion=fermion,
        occupations=(1, 0, 1, 0, 1),
        chi=2,
        random_rounds=2,
        seed=41,
        dtype="complex128",
    )
    fit = py.FIT(
        state.copy(deep=True),
        p=state.copy(deep=True),
        range_int=[1, 3],
        environment_strategy="symmray-native",
    )

    def fail_sweep(*_args, **_kwargs):
        raise RuntimeError("injected native sweep failure")

    monkeypatch.setattr(fit, "_run_gate_two_site_sweep", fail_sweep)
    with pytest.raises(RuntimeError, match="injected native sweep failure"):
        fit.run_gate(
            adaptive_block_sweeps=None,
            n_iter=1,
            block_size=2,
            sweep_sequence="R",
            max_bond=8,
            cutoff=1.0e-12,
        )

    assert fit._fermionic_bra_working is False
    assert fit._fermionic_left_exterior_environment is None
    assert fit._fermionic_right_exterior_environment is None
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, state, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("block_size", [1, 2, 3])
@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (1, 0, 1, 0, 1, 0)),
        (
            True,
            "U1U1",
            ((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        ),
        (False, "Z2", (1, 0, 1, 0, 1, 0)),
    ],
)
def test_fit_fermionic_arbitrary_target_keeps_native_guess_separate(
    block_size,
    spinful,
    symmetry,
    occupations,
    monkeypatch,
):
    """Native FIT does not replace its current state with the target."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    target = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=37,
        dtype="complex128",
    )
    guess = target.copy(deep=True)
    fit = py.FIT(
        target.copy(deep=True),
        p=guess.copy(deep=True),
        range_int=[0, 5],
        environment_strategy="symmray-native",
    )

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native sector initialization must not call to_dense")

    def fail_network_contract(*_args, **_kwargs):
        raise AssertionError(
            "native sector initialization must not build a temporary "
            "TensorNetwork"
        )

    run_options = {
        "n_iter": 2,
        "min_iter": 1,
        "rtol": None,
        "block_size": block_size,
        "sweep_sequence": "RL",
        "max_bond": 2,
        "cutoff": 1.0e-12,
    }
    if block_size > 1:
        run_options["adaptive_block_sweeps"] = 2
    array_types = {type(tensor.data) for tensor in (*guess, *target)}
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        patcher.setattr(qtn.TensorNetwork, "contract", fail_network_contract)
        fit.run_gate(**run_options)

    assert "native_sector_initialization" not in fit.info
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in fit.p
    )
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
@pytest.mark.parametrize("fit_init_strategy", [None, "guess_src"])
def test_mps_optimizer_native_guess_src_uses_sector_preserving_randomized_guess(
    symmetry,
    fit_init_strategy,
    monkeypatch,
):
    """Native default and explicit ``guess_src`` supply a randomized guess."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.ps_to_mps(
        4,
        fermion=fermion,
        occupations=fermion.half_filled_occupations(4),
        seed=17,
        dtype="complex128",
    )
    gate_stream = [(fermion.hopping_gate(0.2, t=1.0), (0, 3))]

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native guess_src must not call dense compression")

    def fail_quimb_guess(*_args, **_kwargs):
        raise AssertionError("native guess_src must not call Quimb guess()")

    monkeypatch.setattr(
        mps_optimizer_module,
        "_apply_dense_gate_with_method",
        fail_dense,
    )
    monkeypatch.setattr(mps_optimizer_module, "guess", fail_quimb_guess)

    optimizer = py.MpsOptimizer(state, gate_stream, chi=8, mode="dmrg2")
    run_kwargs = {}
    if fit_init_strategy is not None:
        run_kwargs["fit_init_strategy"] = fit_init_strategy
    out = optimizer.run(
        progbar=False,
        n_iter=2,
        fit_rtol=None,
        cutoff=1.0e-12,
        fit_init_seed=23,
        stabilize_unitary=False,
        **run_kwargs,
    )

    diagnostics = optimizer.get_fit_diagnostics()
    assert diagnostics["fit_init_strategy_requested"] == "guess_src"
    assert diagnostics["fit_init_strategy"] == "guess_src"
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["guess_used"] is True
    assert diagnostics["svd_guess_used"] is True
    assert diagnostics["guess_backend"] == "symmray-svd:rand"
    assert diagnostics["native_randomized_guess_used"] is True
    assert diagnostics["random_initialization"]["reason"] == "native_src"
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in out.tensors
    )


def test_fit_run_eff_fermionic_keeps_current_state_as_initial_guess():
    """Native block run_eff does not replace its current state."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    occupations = (
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    )
    target = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=37,
        dtype="complex128",
    )
    guess = target.copy(deep=True)
    fit = py.FIT(
        target,
        p=guess,
        environment_strategy="symmray-native",
    )
    fit.run_eff(
        n_iter=2,
        block_size=2,
        sweep_sequence="RL",
        max_bond=2,
        cutoff=1.0e-12,
    )

    assert "native_sector_initialization" not in fit.info
    assert float(
        np.real(py.tn_fidelity(fit.p, target, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-10)


def test_fit_fermionic_partial_window_reports_disconnected_target_sectors():
    """A disconnected partial FIT reports its fixed-boundary sector issue."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    occupations = (
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    )
    guess = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=29,
        dtype="complex128",
    )
    target = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=3,
        seed=37,
        dtype="complex128",
    )
    fit = py.FIT(
        target,
        p=guess,
        range_int=[1, 4],
        environment_strategy="symmray-native",
    )

    with pytest.raises(
        ValueError,
        match="disconnected charge-sector support",
    ):
        fit.run_gate(
            n_iter=2,
            block_size=2,
            sweep_sequence="RL",
            max_bond=2,
            cutoff=1.0e-12,
        )


@pytest.mark.parametrize(
    ("spinful", "symmetry", "occupations"),
    [
        (False, "U1", (1, 0, 1, 0, 1, 0)),
        (
            True,
            "U1U1",
            ((1, 0), (0, 1), (1, 0), (0, 1), (1, 0), (0, 1)),
        ),
        (False, "Z2", (1, 0, 1, 0, 1, 0)),
    ],
)
@pytest.mark.parametrize("fit_sweep_sequence", ["R", "RL"])
@pytest.mark.parametrize("mode", ["dmrg1", "dmrg2", "dmrg3"])
def test_mps_optimizer_named_dmrg_long_range_fermions_stay_native_and_exact(
    spinful,
    symmetry,
    occupations,
    fit_sweep_sequence,
    mode,
    monkeypatch,
):
    """Every named DMRG mode keeps U1/U1U1/Z2 grading end to end."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.hrs_to_mps(
        6,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=2,
        seed=23,
        dtype="complex128",
    )
    gate = fermion.hopping_gate(0.02, t=1.0)
    stream = [(gate, (0, 3))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=16,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0)
    raw_zero_cutoff_target = state.copy(deep=True)
    raw_zero_cutoff_target.gate_with_auto_swap_(
        gate,
        (0, 3),
        info={},
        swap_back=True,
        cutoff=0.0,
        cutoff_mode="rsum2",
    )
    assert float(
        np.real(
            py.tn_fidelity(
                reference,
                raw_zero_cutoff_target,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=1.0e-12)

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("native fermionic DMRG must not call to_dense")

    def fail_network_contract(*_args, **_kwargs):
        raise AssertionError(
            "native fermionic FIT must not build a temporary TensorNetwork"
        )

    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=16,
        mode=mode,
    )
    array_types = {type(tensor.data) for tensor in (*state, *reference)}
    array_types.add(type(gate))
    with monkeypatch.context() as patcher:
        for array_type in array_types:
            patcher.setattr(array_type, "to_dense", fail_dense)
        patcher.setattr(qtn.TensorNetwork, "contract", fail_network_contract)
        out = optimizer.run(
            progbar=False,
            n_iter=3,
            fit_rtol=None,
            cutoff=1.0e-12,
            target_cutoff=0.0,
            fit_sweep_sequence=fit_sweep_sequence,
            stabilize_unitary=False,
        )

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in out
    )
    assert optimizer._last_dmrg_fit_diagnostics[
        "native_fermionic_warm_start"
    ] is True
    diagnostics = optimizer._last_dmrg_fit_diagnostics
    if mode == "dmrg1" and diagnostics["block_size"] == 1:
        # The native warm start can already fill every active rank ceiling.
        # DMRG1 then correctly spends the complete budget on one-site FIT.
        assert diagnostics["adaptive_sweeps"] == 0
        assert diagnostics["one_site_refinement_sweeps"] == 3
    else:
        assert diagnostics["block_size"] == (3 if mode == "dmrg3" else 2)
        assert diagnostics["adaptive_sweeps"] >= 2
    assert (
        diagnostics["adaptive_sweeps"]
        + diagnostics["one_site_refinement_sweeps"]
        == 3
    )
    if mode in {"dmrg2", "dmrg3"}:
        assert diagnostics["adaptive_sweeps"] == (3 if mode == "dmrg3" else 2)
        assert diagnostics["one_site_refinement_sweeps"] == (0 if mode == "dmrg3" else 1)
    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-9)


@pytest.mark.parametrize(
    "symmetry",
    ["U1", "U1U1"],
)
@pytest.mark.parametrize(
    "mode",
    [
        "fit",
        "dmrg",
        "dmrg1",
        "dmrg2",
        "dmrg3",
        "mpo",
        "svd",
        "swap",
        "perm",
        "mix",
        "su",
        "exact",
    ],
)
def test_mps_optimizer_spinful_fermion_symmetry_mode_matrix_matches_mpo(
    symmetry,
    mode,
):
    """All supported MPS modes preserve native spinful U1 charge sectors."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = py.ps_to_mps(
        4,
        fermion=fermion,
        occupations=fermion.half_filled_occupations(4),
        seed=41,
        dtype="complex128",
    )
    gate_stream = [(fermion.hopping_gate(0.02, t=1.0), (0, 3))]
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        gate_stream,
        chi=16,
        mode="mpo",
    ).run(
        progbar=False,
        cutoff=0.0,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    optimizer = py.MpsOptimizer(
        state.copy(deep=True),
        gate_stream,
        chi=16,
        mode=mode,
    )
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    if mode == "perm":
        optimizer.restore_qubit_order()
        compared = optimizer.p
    elif mode == "su":
        # SU evolves an ungauged core and exposes the physical state through
        # the separately stored bond gauges.
        compared = optimizer.p_ungauged
        assert compared is not None
    else:
        compared = out

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in compared.tensors
    )
    assert float(
        np.real(py.tn_fidelity(compared, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-9)


@pytest.mark.parametrize(
    ("spinful", "symmetry"),
    [
        (False, "U1"),
        (True, "U1"),
        (True, "U1U1"),
        (False, "Z2"),
        (True, "Z2"),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "fit",
        "dmrg",
        "dmrg1",
        "dmrg2",
        "dmrg3",
        "mpo",
        "svd",
        "swap",
        "perm",
        "mix",
        "su",
        "exact",
    ],
)
def test_mps_optimizer_fermion_gate_stream_ps_to_mps_all_modes_native(
    spinful,
    symmetry,
    mode,
):
    """Fermion gate streams remain native and exact across all MPS modes."""
    pytest.importorskip("symmray")

    def build_case():
        fermion = py.Fermion(
            spinful=spinful,
            symmetry=symmetry,
            dtype="complex128",
        )
        params = {"t": 0.7, "mu": 0.13}
        if spinful:
            params["U"] = 1.2
        else:
            params["V"] = 0.2
        stream = list(
            fermion.gate_stream(
                ((0, 1), (1, 2), (2, 3)),
                0.01,
                sites=range(4),
                order=1,
                **params,
            )
        )
        state = py.ps_to_mps(
            4,
            fermion=fermion,
            occupations=fermion.half_filled_occupations(4),
            seed=7,
            dtype="complex128",
        )
        return state, stream

    reference_state, reference_stream = build_case()
    reference = py.MpsOptimizer(
        reference_state,
        reference_stream,
        chi=16,
        mode="exact",
    ).run(
        progbar=False,
        cutoff=0.0,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )

    state, stream = build_case()
    assert all(
        type(gate).__module__.split(".", 1)[0] == "symmray"
        and type(gate).__name__.endswith("FermionicArray")
        for gate, _ in stream
    )
    optimizer = py.MpsOptimizer(
        state,
        stream,
        chi=16,
        mode=mode,
    )
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-12,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    if mode == "perm":
        optimizer.restore_qubit_order()
        compared = optimizer.p
    elif mode == "su":
        compared = optimizer.p_ungauged
        assert compared is not None
    else:
        compared = out

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in compared.tensors
    )
    assert float(
        np.real(py.tn_fidelity(compared, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=1.0e-9)


@pytest.mark.parametrize(
    ("spinful", "symmetry"),
    [
        (False, "U1"),
        (False, "Z2"),
        (True, "U1"),
        (True, "U1U1"),
        (True, "Z2"),
    ],
)
@pytest.mark.parametrize("occupation_kind", ["vacuum", "full"])
def test_mps_optimizer_complex64_native_fit_short_sector_edges(
    spinful,
    symmetry,
    occupation_kind,
):
    """FIT handles short native sectors at complex64 precision."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex64",
    )
    if spinful:
        occupation = (0, 0) if occupation_kind == "vacuum" else (1, 1)
    else:
        occupation = 0 if occupation_kind == "vacuum" else 1
    occupations = (occupation, occupation)
    params = {"t": 0.3, "mu": 0.1, "V": 0.2}
    if spinful:
        params["U"] = 0.8
    stream = list(
        fermion.gate_stream(
            ((0, 1),),
            0.01,
            sites=(0, 1),
            order=2,
            **params,
        )
    )
    state = py.ps_to_mps(
        2,
        fermion=fermion,
        occupations=occupations,
        seed=13,
        dtype="complex64",
    )
    reference = py.MpsOptimizer(
        state.copy(deep=True),
        stream,
        chi=4,
        mode="exact",
    ).run(
        progbar=False,
        cutoff=0.0,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )
    optimizer = py.MpsOptimizer(state, stream, chi=4, mode="fit")
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-6,
        target_cutoff=0.0,
        n_iter=2,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    assert py.MpsOptimizer._mps_data_is_finite(out)
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        and all(
            np.dtype(block.dtype) == np.dtype("complex64")
            for block in tensor.data.blocks.values()
        )
        for tensor in out.tensors
    )
    assert float(
        np.real(py.tn_fidelity(out, reference, contraction_opt="greedy"))
    ) == pytest.approx(1.0, abs=2.0e-5)


@pytest.mark.slow
@pytest.mark.parametrize("symmetry", ["U1", "U1U1", "Z2"])
@pytest.mark.parametrize(
    "mode",
    [
        "fit",
        "dmrg",
        "dmrg1",
        "dmrg2",
        "dmrg3",
        "mpo",
        "svd",
        "swap",
        "perm",
        "mix",
        "su",
        "exact",
    ],
)
def test_mps_optimizer_3x4_pbc_hubbard_long_range_modes_native(
    symmetry,
    mode,
):
    """Stress native Hubbard modes on a periodic 3x4 lattice."""
    pytest.importorskip("symmray")
    Lx, Ly = 3, 4
    mapper = py.OneDMap(Lx, Ly, mode="snake")
    idx2coo, coo2idx = mapper.build()
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    setup = fermion.lattice_half_filling(Lx, Ly, cyclic=True)
    occupations = tuple(
        setup.occupations[idx2coo[index]] for index in range(Lx * Ly)
    )
    mapped_edges = tuple(
        tuple(coo2idx[site] for site in edge) for edge in setup.edges
    )
    stream = list(
        fermion.gate_stream(
            mapped_edges,
            0.002,
            sites=range(Lx * Ly),
            order=1,
            t=0.8,
            U=2.0,
            mu=0.1,
        )
    )
    long_range_gate = fermion.hopping_gate(0.003, t=0.4)
    stream.extend(
        [
            (
                long_range_gate,
                (coo2idx[(0, 0)], coo2idx[(2, 2)]),
            ),
            (
                long_range_gate,
                (coo2idx[(0, 3)], coo2idx[(2, 0)]),
            ),
        ]
    )
    assert len(set(setup.edges)) == 24
    assert len(stream) == 38
    assert all(
        type(gate).__module__.split(".", 1)[0] == "symmray"
        and type(gate).__name__.endswith("FermionicArray")
        for gate, _ in stream
    )

    state = py.ps_to_mps(
        Lx * Ly,
        fermion=fermion,
        occupations=occupations,
        seed=12,
        dtype="complex128",
    )
    optimizer = py.MpsOptimizer(
        state,
        stream,
        chi=8 if mode == "exact" else 16,
        mode=mode,
    )
    out = optimizer.run(
        progbar=False,
        cutoff=1.0e-8 if mode == "exact" else 1.0e-10,
        target_cutoff=0.0,
        n_iter=3,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    compared = out
    if mode == "perm":
        optimizer.restore_qubit_order()
        compared = optimizer.p
    elif mode == "su":
        compared = optimizer.p_ungauged
        assert compared is not None

    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in compared.tensors
    )
    if mode == "exact":
        assert len(compared.tensors) == 1
        assert float(np.real(py.to_float(compared.norm()))) == pytest.approx(
            1.0,
            abs=1.0e-8,
        )
    else:
        reference_state = py.ps_to_mps(
            Lx * Ly,
            fermion=py.Fermion(
                spinful=True,
                symmetry=symmetry,
                dtype="complex128",
            ),
            occupations=occupations,
            seed=12,
            dtype="complex128",
        )
        reference = py.MpsOptimizer(
            reference_state,
            stream,
            chi=16,
            mode="mpo",
        ).run(
            progbar=False,
            cutoff=0.0,
            target_cutoff=0.0,
            stabilize_unitary=False,
        )
        assert float(
            np.real(
                py.tn_fidelity(
                    compared,
                    reference,
                    contraction_opt="greedy",
                )
            )
        ) == pytest.approx(1.0, abs=5.0e-5)


@pytest.mark.slow
@pytest.mark.parametrize("symmetry", ["U1", "U1U1", "Z2"])
def test_mps_optimizer_complex64_3x4_pbc_perm_stays_finite_native(symmetry):
    """Native complex64 lazy swaps remain finite on the hard lattice case."""
    pytest.importorskip("symmray")
    Lx, Ly = 3, 4
    mapper = py.OneDMap(Lx, Ly, mode="snake")
    idx2coo, coo2idx = mapper.build()
    fermion = py.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex64",
    )
    setup = fermion.lattice_half_filling(Lx, Ly, cyclic=True)
    occupations = tuple(
        setup.occupations[idx2coo[index]] for index in range(Lx * Ly)
    )
    mapped_edges = tuple(
        tuple(coo2idx[site] for site in edge) for edge in setup.edges
    )
    stream = list(
        fermion.gate_stream(
            mapped_edges,
            0.002,
            sites=range(Lx * Ly),
            order=1,
            t=0.8,
            U=2.0,
            mu=0.1,
        )
    )
    long_range_gate = fermion.hopping_gate(0.003, t=0.4)
    stream.extend(
        [
            (
                long_range_gate,
                (coo2idx[(0, 0)], coo2idx[(2, 2)]),
            ),
            (
                long_range_gate,
                (coo2idx[(0, 3)], coo2idx[(2, 0)]),
            ),
        ]
    )
    assert {
        np.dtype(block.dtype)
        for gate, _where in stream
        for block in gate.blocks.values()
    } == {np.dtype("complex64")}
    state = py.ps_to_mps(
        Lx * Ly,
        fermion=fermion,
        occupations=occupations,
        seed=12,
        dtype="complex64",
    )
    reference_state = py.ps_to_mps(
        Lx * Ly,
        fermion=py.Fermion(
            spinful=True,
            symmetry=symmetry,
            dtype="complex64",
        ),
        occupations=occupations,
        seed=12,
        dtype="complex64",
    )
    reference = py.MpsOptimizer(
        reference_state,
        stream,
        chi=16,
        mode="mpo",
    ).run(
        progbar=False,
        cutoff=1.0e-6,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )
    optimizer = py.MpsOptimizer(state, stream, chi=16, mode="perm")
    optimizer.run(
        progbar=False,
        cutoff=1.0e-6,
        target_cutoff=0.0,
        stabilize_unitary=False,
    )
    optimizer.restore_qubit_order()

    compared = optimizer.p
    assert py.MpsOptimizer._mps_data_is_finite(compared)
    assert all(
        type(tensor.data).__module__.split(".", 1)[0] == "symmray"
        and type(tensor.data).__name__.endswith("FermionicArray")
        and all(
            np.dtype(block.dtype) == np.dtype("complex64")
            for block in tensor.data.blocks.values()
        )
        for tensor in compared.tensors
    )
    assert float(
        np.real(
            py.tn_fidelity(
                compared,
                reference,
                contraction_opt="greedy",
            )
        )
    ) == pytest.approx(1.0, abs=3.0e-4)


def test_mps_optimizer_three_site_fit_uses_window_and_falls_back_short():
    """MpsOptimizer should use three-site FIT and shorten adjacent windows."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 3))],
        chi=2,
        mode="dmrg",
    )
    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=3,
        timing=True,
    )
    assert optimizer._last_dmrg_fit_diagnostics["block_size"] == 3
    assert optimizer._last_dmrg_fit_diagnostics["adaptive_sweeps"] == 2
    assert optimizer._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 1
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [3, 3, 1]
    assert [
        record["site_count"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == [2, 2, 4]

    adjacent = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="dmrg",
    )
    adjacent.run(
        progbar=False,
        n_iter=1,
        fit_rtol=None,
        fit_block_size=3,
        timing=True,
    )
    assert adjacent._last_dmrg_fit_diagnostics["block_size"] == 2
    assert adjacent._last_dmrg_fit_diagnostics["one_site_refinement_sweeps"] == 0
    assert [
        record["block_size"]
        for record in adjacent.get_run_timing()["fit_steps"]
    ] == [2]


@pytest.mark.parametrize(
    ("where", "block_size"),
    [((0, 2), 2), ((0, 3), 3)],
)
def test_mps_optimizer_boundary_long_range_uses_fixed_handoff(
    where,
    block_size,
    monkeypatch,
):
    """A window one site wider than its FIT block is not rank-adaptive."""
    adaptive_rank_flags = []
    original_run_fit_gate = py.MpsOptimizer._run_fit_gate

    def record_schedule(self, fit, **kwargs):
        adaptive_rank_flags.append(bool(kwargs["adaptive_until_rank"]))
        return original_run_fit_gate(self, fit, **kwargs)

    monkeypatch.setattr(py.MpsOptimizer, "_run_fit_gate", record_schedule)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state(
            "0" * (max(where) + 1),
            dtype="complex128",
        ),
        gates=[(qu.CNOT(), where)],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=block_size,
    )

    assert adaptive_rank_flags == [False]


def test_mps_optimizer_batched_boundary_long_range_uses_fixed_handoff(
    monkeypatch,
):
    """The batched DMRG path applies the same inclusive-span rule."""
    adaptive_rank_flags = []
    original_run_fit_gate = py.MpsOptimizer._run_fit_gate

    def record_schedule(self, fit, **kwargs):
        adaptive_rank_flags.append(bool(kwargs["adaptive_until_rank"]))
        return original_run_fit_gate(self, fit, **kwargs)

    monkeypatch.setattr(py.MpsOptimizer, "_run_fit_gate", record_schedule)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 1)), (qu.CNOT(), (1, 2))],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=2,
        fit_layer_size=2,
    )

    assert adaptive_rank_flags == [False]


@pytest.mark.parametrize("mode", ("dmrg", "mix"))
@pytest.mark.parametrize("block_size", (2, 3))
def test_mps_optimizer_adaptive_blocks_do_not_preexpand_bonds(
    mode,
    block_size,
    monkeypatch,
):
    """Adaptive FIT must grow only bonds reached by its native SVD splits."""
    length = 8
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * length, dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 4))],
        chi=2,
        mode=mode,
    )

    def fail_rank_warmup(*args, **kwargs):
        raise AssertionError(
            "adaptive FIT must not pre-expand MPS bonds before fitting"
        )

    monkeypatch.setattr(
        optimizer,
        "_prepare_mix_dmrg_state",
        fail_rank_warmup,
    )
    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        fit_block_size=block_size,
        cutoff=0.0,
        target_cutoff=0.0,
        fit_single_pair_fast_path=False,
        timing=True,
    )

    assert [
        optimizer.p.bond_size(site, site + 1) for site in range(length - 1)
    ] == [2, 2, 2, 2, 1, 1, 1]
    assert optimizer._last_dmrg_fit_diagnostics["block_size"] == block_size
    if mode == "mix":
        assert optimizer.mix_history[-1]["backend"] == "dmrg"


def test_unitary_fit_stabilization_preserves_working_norm():
    """Unitary FIT stabilization keeps the live state at its working norm."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[(qu.hadamard(), (0,)), (qu.CNOT(), (0, 1))],
        chi=1,
        mode="fit",
    )

    out = optimizer.run(progbar=False, n_iter=2, stabilize_unitary=True)
    raw = out.copy()
    raw.exponent = 0.0

    assert float(np.real(raw.norm())) == pytest.approx(1.0, abs=1.0e-12)


def test_dmrg_fit_layer_size_and_target_cutoff_are_independent(monkeypatch):
    """Paper-style gate blocks must not reuse the output truncation cutoff."""
    calls = []
    original = py.MpsOptimizer._build_dmrg_batch_target

    def record_target(
        self,
        p,
        gates,
        where,
        target_cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="auto",
    ):
        calls.append(
            (len(gates), float(target_cutoff), cutoff_mode, target_strategy)
        )
        return original(
            self,
            p,
            gates,
            where,
            target_cutoff,
            cutoff_mode,
            target_strategy=target_strategy,
        )

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_build_dmrg_batch_target",
        record_target,
    )
    optimizer = py.MpsOptimizer(
        qtn.MPS_rand_state(
            4, bond_dim=2, phys_dim=2, dtype="complex128", seed=207
        ),
        gates=[
            (qu.CNOT(), (0, 2)),
            (qu.CNOT(), (1, 3)),
            (qu.CNOT(), (0, 3)),
        ],
        chi=2,
        mode="dmrg",
    )

    optimizer.run(
        progbar=False,
        n_iter=2,
        cutoff=1.0e-3,
        target_cutoff=0.0,
        fit_layer_size=2,
    )

    assert calls == [
        (2, 0.0, "rsum2", "layered"),
        (1, 0.0, "rsum2", "layered"),
    ]


def test_mps_optimizer_fit_mode_is_clear_dmrg_alias():
    """The public FIT spelling should select the maintained DMRG kernel."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00"),
        gates=[(qu.CNOT(), (0, 1))],
        chi=2,
        mode="fit",
    )

    out = optimizer.run(progbar=False, n_iter=1)

    assert optimizer.mode == "dmrg"
    assert out.L == 2


@pytest.mark.parametrize(
    ("mode", "expected_blocks"),
    [
        ("dmrg1", [2, 2, 1]),
        ("dmrg2", [2, 2, 1]),
        ("dmrg3", [3, 3, 2]),
    ],
)
def test_mps_optimizer_dmrg_mode_aliases_select_block_size(mode, expected_blocks):
    """Named DMRG modes select the corresponding FIT block size."""
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode=mode,
    )

    optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        timing=True,
    )

    assert optimizer.mode == "dmrg"
    assert [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ] == expected_blocks


def test_mps_optimizer_mix_uses_norm_guard_without_full_scan(monkeypatch):
    """Successful mixed FIT should avoid a full post-update data scan."""
    original_check = py.MpsOptimizer._mps_data_is_finite
    checks = 0

    def counted_check(candidate):
        nonlocal checks
        checks += 1
        return original_check(candidate)

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_mps_data_is_finite",
        staticmethod(counted_check),
    )
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(
            3, bond_dim=2, phys_dim=2, dtype="complex128", seed=202
        ),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, n_iter=3, fit_rtol=None)

    assert opt.mix_history[0]["backend"] == "dmrg"
    assert checks == 0


def test_mps_optimizer_mix_uses_norm_guard_after_mpo_warmup(monkeypatch):
    """MPO warm-up should avoid a full post-update data scan."""
    original_check = py.MpsOptimizer._mps_data_is_finite
    checks = 0

    def counted_check(candidate):
        nonlocal checks
        checks += 1
        return original_check(candidate)

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_mps_data_is_finite",
        staticmethod(counted_check),
    )
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=[
            (qu.hadamard(), (0,)),
            (qu.CNOT(), (0, 1)),
            (qu.CNOT(), (1, 2)),
        ],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, fit_block_size=1)

    assert [entry["backend"] for entry in opt.mix_history] == ["mpo"] * 3
    assert checks == 0


def test_mps_optimizer_mix_stops_fit_adaptively():
    """Mixed n_iter is a maximum when the FIT norm has converged."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=21
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(
        progbar=False,
        n_iter=8,
        fit_min_iter=2,
        fit_rtol=1e9,
        fit_patience=1,
    )

    event = opt.mix_history[0]
    assert event["backend"] == "dmrg"
    # Mixed mode's default is one-site FIT, so there is no separate block
    # warm-up phase before tolerance stopping.
    assert event["fit_iterations"] == 2
    assert event["fit_converged"] is True
    assert event["fit_relative_change"] <= 1e9


def test_mps_optimizer_dmrg_stops_fit_adaptively():
    """Ordinary DMRG should use the same adaptive FIT stopping controls."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=212
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    opt.run(
        progbar=False,
        n_iter=8,
        fit_min_iter=2,
        fit_rtol=1e9,
        fit_patience=1,
    )

    assert opt._last_dmrg_fit_diagnostics["iterations"] == 4


def test_mps_optimizer_mix_can_keep_fixed_fit_iterations():
    """Disabling mixed FIT tolerance should preserve exact n_iter behavior."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=211
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, n_iter=3, fit_rtol=None)

    event = opt.mix_history[0]
    assert event["fit_iterations"] == 3
    assert event["fit_converged"] is False
    assert event["fit_relative_change"] is None


def test_mps_optimizer_deprecated_fit_controls_warn_and_remain_functional():
    """Legacy mixed FIT names should delegate to the canonical controls."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=217
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    with pytest.warns(DeprecationWarning, match="mix_fit_rtol"):
        opt.run(
            progbar=False,
            n_iter=4,
            mix_fit_rtol=None,
        )

    assert opt._last_dmrg_fit_diagnostics["iterations"] == 4


def test_mps_optimizer_rejects_conflicting_legacy_fit_controls():
    """Mixed old/new convergence policies must never be resolved silently."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"),
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="dmrg",
    )

    with pytest.warns(DeprecationWarning, match="mix_fit_rtol"):
        with pytest.raises(ValueError, match="different values"):
            opt.run(
                progbar=False,
                fit_rtol=1.0e-7,
                mix_fit_rtol=None,
            )


def test_mix_unitary_stabilization_covers_mpo_rank_warmup():
    """One-site mixed MPO warm-up should retain the unitary working norm."""
    gates = []
    for depth in range(4):
        start = depth % 2
        for site in range(start, 7, 2):
            gates.append((qu.rand_uni(4, seed=100 + len(gates)), (site, site + 1)))

    stabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        gates=gates,
        chi=4,
        mode="mix",
    )
    stabilized.run(
        progbar=False,
        n_iter=1,
        fit_block_size=1,
        fit_rtol=None,
        stabilize_unitary=True,
    )

    unstabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("0" * 8, dtype="complex128"),
        gates=gates,
        chi=4,
        mode="mix",
    )
    unstabilized.run(
        progbar=False,
        n_iter=1,
        fit_block_size=1,
        fit_rtol=None,
        stabilize_unitary=False,
    )

    assert _mps_data_norm(stabilized.p) == pytest.approx(1.0, abs=1.0e-12)
    assert _mps_data_norm(unstabilized.p) < 0.999


def test_mps_optimizer_mix_nonfinite_sweep_disables_later_dmrg(monkeypatch):
    """A non-finite FIT sweep should fall back once and latch to MPO."""
    state = qtn.MPS_rand_state(
        3, bond_dim=2, phys_dim=2, dtype="complex128", seed=22
    )
    def failed_fit_sweep(self, *_args, **_kwargs):
        self.iterations_run = 1
        self.converged = False
        self.last_relative_change = None
        raise np.linalg.LinAlgError(
            "Array must not contain infs or NaNs."
        )

    monkeypatch.setattr(
        mps_optimizer_module.FIT,
        "run_gate",
        failed_fit_sweep,
    )
    opt = py.MpsOptimizer(
        state,
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
    )

    opt.run(progbar=False, n_iter=8, finite_check=True, mix_sticky_nonfinite=True)

    first, second = opt.mix_history
    assert first["reason"] == "dmrg_fallback"
    assert first["fit_iterations"] == 1
    assert first["failed_sweep"] == 1
    assert second["reason"] == "dmrg_disabled_nonfinite"
    assert opt.last_mix_summary["dmrg_disabled"] is True
    assert opt.last_mix_summary["failed_sweep"] == 1


def test_mps_optimizer_mix_inplace_success_two_site_keeps_input_identity():
    """Successful two-site mixed DMRG updates must preserve inplace semantics."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=23)
    before = p0.to_dense().copy()
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=3)

    assert out is p0
    assert opt.p is p0
    assert opt.mix_history[0]["backend"] == "dmrg"
    assert not np.allclose(p0.to_dense(), before)


def test_mps_optimizer_mix_inplace_short_active_bond_uses_mpo():
    """In-place mixed replay should warm a short active bond through MPO."""
    dense = np.zeros((2, 2, 2, 2), dtype=complex)
    dense[0, 0, 0, 0] = 1.0 / np.sqrt(2.0)
    dense[1, 1, 0, 0] = 1.0 / np.sqrt(2.0)
    p0 = qtn.MatrixProductState.from_dense(dense)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.hadamard(), (2,)), (qu.CNOT(), (2, 3))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    out = opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=4,
        fit_block_size=1,
    )

    assert out is p0
    assert opt.p is p0
    assert opt.mix_history[1]["backend"] == "mpo"
    assert opt.mix_history[1]["reason"] == "active_bond_below_target"
    assert p0.bond_size(2, 3) == 2


def test_mps_optimizer_mix_trial_copies_only_active_canonical_path():
    """Mixed transactions isolate the active window without copying the chain."""
    p0 = qtn.MPS_rand_state(5, bond_dim=2, phys_dim=2, dtype="complex128", seed=28)
    opt = py.MpsOptimizer(p0, gates=[], chi=2, mode="mix", inplace=True)
    left_inds = tuple(tensor.left_inds for tensor in opt.p)

    trial, sites = opt._copy_mix_trial(opt.p, (0, 2), {"cur_orthog": (0, 0)})

    assert sites == (0, 1, 2)
    assert tuple(tensor.left_inds for tensor in trial) == left_inds
    assert trial[0].data is not opt.p[0].data
    assert trial[2].data is not opt.p[2].data
    assert trial[3].data is opt.p[3].data
    before = np.array(opt.p[0].data, copy=True)
    trial[0].modify(data=np.zeros_like(trial[0].data))
    assert np.allclose(opt.p[0].data, before)


def test_mps_optimizer_mix_inplace_commit_preserves_left_inds():
    """Committing a trial must retain Quimb's canonical-isometry metadata."""
    p0 = qtn.MPS_rand_state(
        5,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=281,
    )
    opt = py.MpsOptimizer(p0, gates=[], chi=2, mode="mix", inplace=True)
    committed = opt.p
    trial, sites = opt._copy_mix_trial(
        committed,
        (0, 4),
        opt.info_c,
    )
    trial.canonize([4], cur_orthog=opt.info_c["cur_orthog"])
    expected_left_inds = tuple(tensor.left_inds for tensor in trial)

    opt._commit_mix_trial(committed, trial, sites=sites)

    assert opt.p is committed
    assert tuple(tensor.left_inds for tensor in committed) == expected_left_inds


def test_mps_optimizer_mix_fallback_restores_unitary_norm_tracking(monkeypatch):
    """A failed DMRG trial must restore state before MPO fallback."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=29)
    gates = [(qu.CNOT(), (0, 2))]
    original_run_dmrg = py.MpsOptimizer._run_dmrg

    def failed_dmrg(self, *args, **kwargs):
        original_run_dmrg(self, *args, **kwargs)
        self._unitary_previous_norm = 123.456
        raise RuntimeError("forced DMRG failure")

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", failed_dmrg)
    opt = py.MpsOptimizer(
        p0,
        gates=gates,
        chi=2,
        mode="mix",
        inplace=True,
    )
    opt.run(progbar=False, cutoff=1e-12)
    assert py.MpsOptimizer._mps_data_is_finite(opt.p)
    assert opt.mix_history[-1]["backend"] == "mpo"


def test_mps_optimizer_mix_strict_restores_then_reraises(monkeypatch):
    """Strict mixed mode should expose DMRG errors without corrupting state."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=30)
    before = p0.to_dense().copy()

    def failed_dmrg(self, *args, **kwargs):
        raise RuntimeError("strict DMRG failure")

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", failed_dmrg)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    with pytest.raises(RuntimeError, match="strict DMRG failure"):
        opt.run(progbar=False, mix_strict=True)
    assert opt.p is p0
    assert np.allclose(p0.to_dense(), before)
    assert opt.mix_history == []


def test_mps_optimizer_mix_interrupt_restores_trial_state(monkeypatch):
    """Interrupting a DMRG trial must leave the committed MPS usable."""
    p0 = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=31)
    before = p0.to_dense().copy()

    def interrupt(self, *args, **kwargs):
        data = np.asarray(self.p[0].data)
        self.p[0].modify(data=np.full_like(data, np.nan))
        raise KeyboardInterrupt

    monkeypatch.setattr(py.MpsOptimizer, "_run_dmrg", interrupt)
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 2))],
        chi=2,
        mode="mix",
        inplace=True,
    )

    with pytest.raises(KeyboardInterrupt):
        opt.run(progbar=False)
    assert opt.p is p0
    assert py.MpsOptimizer._mps_data_is_finite(p0)
    assert np.allclose(p0.to_dense(), before)
    assert opt.mix_history == []


def test_mps_optimizer_mix_batches_two_site_transactions():
    """Mixed mode should support transactional DMRG batches."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=37),
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (1, 3))],
        chi=2,
        mode="mix",
    )

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=3, k_2q_batch=2)

    assert out.max_bond() <= 2
    assert [entry["backend"] for entry in opt.mix_history] == ["dmrg", "dmrg"]
    assert opt.mix_history[0]["reason"] == "bond_at_target"
    assert opt.mix_history[1]["reason"] == "dmrg_batch"


def test_mps_optimizer_batch_collection_respects_spatial_span():
    """Span-aware batching splits disjoint gates before a wide FIT window."""
    gates = [qu.CNOT(), qu.CNOT(), qu.CNOT()]
    where = [(0, 1), (2, 3), (10, 11)]

    batch_G, batch_where, count, next_idx = py.MpsOptimizer._collect_dmrg_batch(
        gates,
        where,
        0,
        3,
        max_span=5,
    )

    assert batch_G == gates[:2]
    assert batch_where == where[:2]
    assert count == 2
    assert next_idx == 2


def test_mps_optimizer_quality_checks_report_finite_canonical_state():
    """Periodic quality checks expose finite data and gauge coverage."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 2)), (qu.CNOT(), (1, 3))],
        chi=2,
        mode="dmrg",
    )

    opt.run(progbar=False, n_iter=2, quality_check_every=True)

    assert len(opt.get_quality_checks()) == 2
    assert all(record["finite"] for record in opt.get_quality_checks())
    assert all(record["canonical_ok"] for record in opt.get_quality_checks())


def test_mps_optimizer_mix_rejects_initial_bond_above_chi():
    """Mixed mode must not silently violate its configured bond limit."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=4, phys_dim=2, dtype="complex128", seed=41),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode="mix",
    )

    with pytest.raises(ValueError, match="initial MPS max bond"):
        opt.run(progbar=False)


def test_mps_optimizer_mix_history_keeps_logical_layout_sites():
    """Mixed history should expose logical and execution gate locations."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=2,
        mode="mix",
    )
    opt.apply_layout((3, 2, 1, 0), layout_report=False)
    opt.run(progbar=False)

    assert opt.mix_history[0]["where"] == (0, 3)
    assert opt.mix_history[0]["execution_where"] == (3, 0)


def test_mps_optimizer_mix_rejects_non_unitary_stream_controls():
    """Mix mode is intentionally restricted to unitary streams."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    gates = [(qu.CNOT(), (0, 1))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="mix")

    with pytest.raises(ValueError, match="only for unitary"):
        opt.run(non_unitary=True, normalize_every=True)


def test_mps_optimizer_accepts_bundled_gate_stream():
    """Construction should accept ``[(gate, where), ...]`` with ``where=None``."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12)

    assert opt.p.L == 4
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_accepts_stabilizer_style_symbolic_gate_stream():
    """Named entries should resolve to the same matrices as Pepsy primitives."""
    theta = 0.19
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[("H", 0), ("rzz", theta, 0, 1)],
        chi=4,
        mode="svd",
    )

    assert opt.where == [(0,), (0, 1)]
    assert opt.G[0].shape == (2, 2)
    assert opt.G[1].shape == (2, 2, 2, 2)

    expected = p0.copy()
    expected.gate_(py.h(), 0, contract=True)
    expected.gate_(py.rzz(theta), (0, 1), contract="split", cutoff=0.0)
    opt.run(progbar=False, cutoff=1.0e-12)

    np.testing.assert_allclose(
        np.asarray(opt.p.to_dense()).reshape(-1),
        np.asarray(expected.to_dense()).reshape(-1),
        atol=1.0e-10,
    )


def test_mps_optimizer_symbolic_gate_stream_uses_explicit_to_backend():
    """Named gates should be converted before strict backend validation."""
    torch = pytest.importorskip("torch")

    backend = py.backend_torch(dtype=torch.complex64, device="cpu")
    p0 = qtn.MPS_computational_state("000", dtype="complex64")
    p0.apply_to_arrays(backend)
    converted = []

    def to_backend(array):
        converted.append(array)
        return backend(np.array(array, copy=True))

    opt = py.MpsOptimizer(
        p0,
        gates=[("H", 0), ("rzz", 0.19, 0, 1)],
        chi=4,
        mode="svd",
        to_backend=to_backend,
    )

    assert len(converted) == 2
    assert all(isinstance(gate, torch.Tensor) for gate in opt.G)
    assert all(gate.dtype == torch.complex64 for gate in opt.G)
    opt.run(progbar=False, cutoff=1.0e-7)
    assert opt.backend_info()["backend"] == "torch"


def test_mps_optimizer_symbolic_set_and_add_gates_resolve():
    """Queue mutation should use the same symbolic resolver as construction."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        gates=[],
        chi=4,
        mode="svd",
    )

    opt.set_gates([("H", 0)])
    opt.add_gates([("rzz", 0.2, 0, 1)])

    assert len(opt.G) == 2
    assert all(isinstance(gate, np.ndarray) for gate in opt.G)
    opt.run(progbar=False, cutoff=1.0e-12)


def test_mps_optimizer_compiles_gate_stream_once(monkeypatch):
    """Repeated replay reuses compiled stream metadata."""
    calls = []
    original = mps_optimizer_module._normalize_gate_queue

    def count_normalization(gates):
        calls.append(gates)
        return original(gates)

    monkeypatch.setattr(
        mps_optimizer_module,
        "_normalize_gate_queue",
        count_normalization,
    )
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.hadamard(), (1,)), (qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )

    assert len(calls) == 1
    opt.run(progbar=False, cutoff=1e-12)
    opt.run(progbar=False, cutoff=1e-12)
    assert len(calls) == 1


def test_mps_optimizer_forwards_custom_ind_id_to_gate_application():
    """Optimizer gate application should honor non-default physical indices."""
    p0 = qtn.MPS_computational_state("000", dtype=np.complex128)
    p0.reindex_({f"k{i}": f"b{i}" for i in range(3)})
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[(x_gate, (1,))],
        chi=4,
        mode="svd",
        ind_id="b{}",
    )
    out = opt.run(progbar=False, cutoff=1e-12)

    assert out is opt.p
    assert set(out.outer_inds()) == {"b0", "b1", "b2"}


def test_mps_optimizer_set_and_add_gates_accept_bundled_gate_stream():
    """set_gates/add_gates should accept bundled gate-stream entries."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), chi=8, mode="svd")

    opt.set_gates([(qu.hadamard(), (1,))])
    opt.add_gates([(qu.CNOT(), (0, 3))])

    assert len(opt.G) == 2
    assert opt.where == [(1,), (0, 3)]


def test_mps_optimizer_layout_finder_api_is_separate_module():
    """Layout finder should live outside the optimizer implementation file."""
    from pepsy.optimizers.mps import MpsGateStreamLayoutFinder
    from pepsy.optimizers.mps.layout import MpsGateStreamLayoutFinder as LayoutFinder

    assert MpsGateStreamLayoutFinder is LayoutFinder
    assert py.MpsOptimizer.LayoutFinder is LayoutFinder


def test_mps_optimizer_gate_stream_layout_remaps_long_range_path():
    """Gate-stream layout should find a short order without changing the stream."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(gates, L=4)

    assert set(plan["site_order"]) == {0, 1, 2, 3}
    assert plan["stats"]["max_span"] == 1
    assert plan["stats"]["long_range_events"] == 0
    assert plan["input_stats"]["long_range_events"] == 2
    assert plan["stats"]["loss"] <= plan["input_stats"]["loss"]
    assert plan["score"] == plan["stats"]["loss"]
    assert plan["layout"] == plan["site_map"]
    assert "recursive_refined" in plan["candidate_scores"]
    assert "gate_stream" not in plan
    assert "gates" not in plan
    assert plan["where"] == tuple(where for _gate, where in gates)
    assert set(plan["inverse_site_map"]) == {0, 1, 2, 3}
    assert all(
        abs(where[0] - where[1]) == 1
        for where in plan["mapped_where"]
    )


def test_mps_quality_layout_includes_periodic_folded_candidate():
    """Quality search should remove the long wrap tail of a periodic grid."""
    Lx = Ly = 8

    def site(x, y):
        return (x % Lx) * Ly + (y % Ly)

    edge_builder = getattr(qtn, "edges_square", qtn.edges_2d_square)
    gates = tuple(
        (qu.CNOT(), (site(*left), site(*right)))
        for left, right in edge_builder(Lx, Ly, cyclic=True)
    )
    plan = py.MpsOptimizer.LayoutFinder(gates, L=Lx * Ly).run(
        order="quality",
        nevergrad_budget=0,
    )

    assert "folded_8" in plan["candidate_losses"]
    assert plan["selected_order"] == "folded_8"
    assert plan["stats"]["max_span"] < plan["input_stats"]["max_span"]
    assert plan["stats"]["loss"] < plan["input_stats"]["loss"]


def test_mps_quality_layout_can_exclude_input_candidate():
    """From-scratch search keeps the original order as diagnostics only."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    plan = py.MpsOptimizer.LayoutFinder(gates, L=4).run(
        order="quality",
        from_scratch=True,
        nevergrad_budget=0,
    )

    assert plan["from_scratch"] is True
    assert "input" not in plan["candidate_plans"]
    assert plan["input_stats"]["long_range_events"] == 2
    assert plan["stats"]["loss"] < plan["input_stats"]["loss"]


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
def test_mps_layout_geometric_presets_match_onedmap(mode):
    """MPS geometric presets must use the shared OneDMap traversal exactly."""
    Lx, Ly = 3, 5
    mapping, _ = py.OneDMap.build(Lx, Ly, mode=mode)
    expected = tuple(x * Ly + y for x, y in mapping.values())

    finder = py.MpsOptimizer.LayoutFinder(
        [],
        L=Lx * Ly,
        lattice_shape=(Lx, Ly),
    )
    plan = finder.run(order=mode)

    assert plan["selected_order"] == mode
    assert plan["site_order"] == expected
    assert plan["mapped_where"] == ()


def test_mps_hilbert_layout_requires_lattice_shape():
    """Named MPS lattice orders reject ambiguous unshaped site sets."""
    finder = py.MpsOptimizer.LayoutFinder([], L=6)
    with pytest.raises(ValueError, match="lattice_shape"):
        finder.run(order="hilbert")


def test_mps_layout_accepts_explicit_fixed_site_order():
    """An explicit site permutation bypasses search and is preserved."""
    gates = [(qu.CNOT(), (0, 3)), (qu.CNOT(), (1, 2))]
    order = (2, 0, 3, 1)
    plan = py.MpsOptimizer.LayoutFinder(gates, L=4).run(order=order)

    assert plan["selected_order"] == "fixed"
    assert plan["site_order"] == order
    assert plan["mapped_where"] == ((1, 2), (3, 0))


def test_mps_compression_layout_reports_operator_cut_load():
    """Compression objective exposes cut-load diagnostics and rank bounds."""
    gate = np.eye(8, dtype=complex)
    plan = py.MpsOptimizer.gate_stream_layout(
        [(gate, (0, 1, 2))],
        L=3,
        objective="compression",
        max_operator_qubits=2,
    )

    assert plan["objective"] == "compression"
    assert plan["stats"]["compression_score"] == plan["score"]
    assert plan["rank_bounded_events"] > 0
    assert plan["rank_bound_reasons"]["max_operator_qubits"] > 0
    assert plan["candidate_plans"]

    exact = py.MpsOptimizer.gate_stream_layout(
        [(qu.CNOT(), (0, 1))],
        L=2,
        objective="compression",
    )
    assert exact["stats"]["rank_exact_events"] == 1
    assert exact["stats"]["total_operator_cut_load"] == pytest.approx(1.0)


def test_mps_compression_layout_pilot_is_non_mutating():
    """Pilot selection uses copied state and does not install a layout."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 3)), (qu.CNOT(), (3, 1))]
    opt = py.MpsOptimizer(
        p0, gates=gates, chi=2, mode="svd"
    )
    before = opt.to_dense()

    selected = opt.select_layout_for_compression(
        pilot_candidates=1,
        pilot_steps=1,
    )

    assert selected["pilot"]["selected_order"]
    assert selected["pilot"]["reports"]
    assert opt._persistent_layout_plan is None
    assert np.allclose(opt.to_dense(), before)


def test_mps_layout_finder_plot_draws_lattice_and_gate_order():
    """The MPS plot exposes the lattice, gate graph, and colored chain."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    finder = py.MpsOptimizer.LayoutFinder(gates, L=4)
    plan = finder.run(order="input")
    fig, ax = finder.plot(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert fig is ax.figure
    assert ax.get_title() == ""
    assert len(ax.patches) == len(plan["site_order"]) - 1
    assert len(fig.axes) == 1  # no stream-order colorbar by default
    assert not ax.axison  # schematic-style presentation by default
    assert any(text.get_text() == "0" for text in ax.texts)
    assert any(text.get_text() == "3" for text in ax.texts)
    assert any(collection.get_offsets().shape[0] for collection in ax.collections)
    plt.close(fig)


def test_mps_optimizer_plot_layout_is_non_mutating():
    """The optimizer plotting wrapper does not install or alter a layout."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    before = tuple(opt.logical_order)
    fig, _ = opt.plot_layout(
        layout_kwargs={"order": "input"},
        site_coords={q: (q, 0) for q in range(4)},
    )

    assert tuple(opt.logical_order) == before
    assert opt._persistent_layout_plan is None
    plt.close(fig)


def test_mps_optimizer_gate_stream_layout_accepts_weight_fn():
    """User event weights should feed the weighted graph and report."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (1, 2)),
    ]

    def weight_fn(_payload, support, _event_type):
        return 10.0 if tuple(support) == (0, 3) else 1.0

    plan = py.MpsOptimizer.gate_stream_layout(
        gates,
        L=4,
        order="input",
        weight_fn=weight_fn,
    )

    assert plan["event_weights"] == (10.0, 1.0)
    assert plan["input_stats"]["total_edge_weight"] == pytest.approx(11.0)
    assert plan["input_stats"]["weighted_long_range_events"] == pytest.approx(10.0)


def test_mps_optimizer_gate_stream_layout_can_use_nevergrad():
    """Optional nevergrad candidate should be usable without touching streams."""
    pytest.importorskip("nevergrad")
    gates = [
        (qu.CNOT(), (0, 4)),
        (qu.CNOT(), (4, 1)),
        (qu.CNOT(), (1, 3)),
        (qu.CNOT(), (3, 2)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(
        gates,
        L=5,
        order="nevergrad",
        nevergrad_budget=8,
        refine_passes=1,
    )

    assert plan["selected_order"] == "nevergrad"
    assert "nevergrad" in plan["candidate_scores"]
    assert set(plan["site_order"]) == set(range(5))
    assert plan["where"] == tuple(where for _gate, where in gates)


def test_mps_optimizer_gate_stream_layout_kahypar_requires_config(monkeypatch):
    """Explicit KaHyPar layouts need a user-supplied config path."""
    monkeypatch.delenv("PEPSY_KAHYPAR_CONFIG", raising=False)
    gates = [(qu.CNOT(), (0, 3)), (qu.CNOT(), (3, 1))]

    with pytest.raises(ValueError, match="kahypar_config_path"):
        py.MpsOptimizer.gate_stream_layout(gates, L=4, order="kahypar")


def test_mps_optimizer_current_gate_stream_layout_uses_state_length():
    """Instance helper should include untouched MPS sites via ``p.L``."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[(qu.CNOT(), (0, 2))],
        chi=8,
        mode="svd",
    )

    plan = opt.current_gate_stream_layout(order="input")

    assert plan["site_order"] == (0, 1, 2, 3)
    assert plan["where"] == ((0, 2),)
    assert plan["mapped_where"] == ((0, 2),)


def test_mps_optimizer_gate_stream_layout_preserves_submpo_events():
    """Layout planning should not rewrite explicit sub-MPO stream events."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [
        py.MpsOptimizer.submpo_event(mpo, (0, 3)),
        (qu.CNOT(), (3, 1)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(stream, L=4)

    assert plan["event_types"] == ("submpo", "gate")
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)
    assert plan["where"][0] == (0, 3)
    assert plan["mapped_where"][0] != plan["where"][0]


def test_mps_optimizer_layout_run_restores_original_mps_order_and_stream():
    """Layout-aware replay should be internal and return original site labels."""
    p0 = qtn.MPS_computational_state("0101", dtype="complex128")
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    ref = py.MpsOptimizer(
        p0.copy(),
        gates=gates,
        chi=16,
        mode="svd",
    ).run(progbar=False, cutoff=1e-12)

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="svd")
    out = opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
    )

    inds = ["k0", "k1", "k2", "k3"]
    assert np.allclose(out.to_dense(inds), ref.to_dense(inds))
    assert out.outer_inds() == tuple(inds)
    assert opt.where == [(0, 3), (3, 1), (1, 2)]
    assert all(
        actual is expected
        for actual, (expected, _where) in zip(opt.G, gates)
    )
    assert opt.last_layout_plan is not None


def test_mps_optimizer_apply_layout_relabels_product_state_without_swaps(monkeypatch):
    """A nonuniform bond-one state should relabel without any SVD swaps."""
    calls = []
    original = qtn.MatrixProductState.swap_site_to_

    def fail_swap(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "swap_site_to_", fail_swap)

    opt = py.MpsOptimizer(
        _nonuniform_product_mps(),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    opt._start_unitary_norm_tracking(opt.p)  # pylint: disable=protected-access
    assert opt._unitary_previous_norm is not None  # pylint: disable=protected-access
    opt.apply_layout((0, 2, 3, 1), layout_report=False)

    assert calls == []
    assert opt._unitary_previous_norm is None  # pylint: disable=protected-access
    assert opt.logical_order == [0, 2, 3, 1]
    assert opt.p.max_bond() == 1
    assert [opt.logical_site(pos) for pos in range(4)] == [0, 2, 3, 1]
    assert [opt.position(site) for site in range(4)] == [0, 3, 1, 2]


def test_mps_optimizer_persistent_layout_reuses_order_and_remaps_readout():
    """Persistent layout replay should agree with identity replay over repeats."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
    ]
    reference = py.MpsOptimizer(
        _nonuniform_product_mps(), gates=gates, chi=8, mode="svd"
    )
    reference.run(progbar=False, cutoff=1e-12)
    reference.run(progbar=False, cutoff=1e-12)

    laid_out = py.MpsOptimizer(
        _nonuniform_product_mps(), gates=gates, chi=8, mode="svd"
    )
    laid_out.apply_layout((0, 2, 3, 1), layout_report=False)
    laid_out.run(progbar=False, cutoff=1e-12)
    laid_out.run(progbar=False, cutoff=1e-12)

    reference_dense = np.asarray(reference.to_dense()).reshape(-1)
    laid_out_dense = np.asarray(laid_out.to_dense()).reshape(-1)
    overlap = np.vdot(reference_dense, laid_out_dense)
    assert abs(overlap) == pytest.approx(1.0, abs=1e-10)
    assert laid_out.logical_order == [0, 2, 3, 1]
    assert laid_out.p.max_bond() <= laid_out.chi
    assert laid_out.layout_plan is laid_out.last_layout_plan

    physical_configs = py.MpsSampler(laid_out.p, backend="quimb").sample(
        n_samples=24, seed=19
    ).configs_1d
    physical_dense = np.asarray(laid_out.p.to_dense()).reshape(-1)
    for physical_config in physical_configs:
        logical_config = laid_out.remap_sample(physical_config).tolist()
        physical_index = int("".join(map(str, physical_config)), 2)
        logical_index = int("".join(map(str, logical_config)), 2)
        assert abs(physical_dense[physical_index]) ** 2 == pytest.approx(
            abs(reference_dense[logical_index]) ** 2,
            abs=1e-10,
        )


def test_mps_optimizer_persistent_layout_rejects_entangled_state_by_default():
    """Entangled initialization needs explicit permission for one-time loss."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=23),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    before = np.asarray(opt.p.to_dense()).copy()

    with pytest.raises(ValueError, match="initially product MPS"):
        opt.apply_layout((0, 2, 3, 1), layout_report=False)

    assert opt.logical_order == [0, 1, 2, 3]
    assert np.allclose(np.asarray(opt.p.to_dense()), before)


def test_mps_optimizer_persistent_layout_entangled_reorder_uses_cutoff(monkeypatch):
    """Lossy persistent initialization should use the caller's cutoff once."""
    calls = []
    original = qtn.MatrixProductState.swap_site_to_

    def counting(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "swap_site_to_", counting)
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=23),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    opt.apply_layout(
        (0, 2, 3, 1),
        cutoff=1e-7,
        allow_lossy_reorder=True,
        layout_report=False,
    )

    assert opt.logical_order == [0, 2, 3, 1]
    assert calls
    assert all(call["cutoff"] == pytest.approx(1e-7) for call in calls)


def test_mps_optimizer_persistent_layout_controls_keep_logical_labels():
    """Persistent layout control events execute physically but record logically."""
    opt = py.MpsOptimizer(
        _nonuniform_product_mps(),
        gates=[
            (qu.hadamard(), (3,)),
            (qu.CNOT(), (0, 3)),
            ("measure", "Z", 3, +1),
        ],
        chi=8,
        mode="mpo",
    )
    opt.apply_layout((0, 2, 3, 1), layout_report=False)
    opt.run(progbar=False)

    assert opt.measurements[0][0:3] == ("Z", (3,), 1)
    assert np.isclose(
        py.MpsOptimizer._real_float(
            opt._state_expectation("Z", (opt.position(3),))
        ),
        1.0,
    )


def test_mps_optimizer_persistent_layout_remaps_submpo_without_mutating_stream():
    """Persistent layout should copy/remap each sub-MPO on every replay."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [py.MpsOptimizer.submpo_event(mpo, (0, 3))]
    reference = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    reference.run(progbar=False, cutoff=1e-12)
    reference.run(progbar=False, cutoff=1e-12)

    opt = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    opt.apply_layout((0, 2, 3, 1), layout_report=False)
    opt.run(progbar=False, cutoff=1e-12)
    opt.run(progbar=False, cutoff=1e-12)

    assert np.allclose(
        np.abs(np.asarray(opt.to_dense()).reshape(-1)),
        np.abs(np.asarray(reference.to_dense()).reshape(-1)),
    )
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)


def test_mps_optimizer_persistent_layout_rejects_cap_events():
    """Persistent layout cannot survive a stream that changes MPS length."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        gates=[("cap", 1, [1.0, 1.0])],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="cap control events"):
        opt.apply_layout((0, 2, 3, 1), layout_report=False)


def test_mps_optimizer_layout_run_reports_score_reduction(capsys):
    """Layout-aware replay should print a concise before/after report."""
    p0 = qtn.MPS_computational_state("0101", dtype="complex128")
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="svd")

    opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
        layout_report=True,
    )

    report = capsys.readouterr().out
    assert "MpsOptimizer layout finder:" in report
    assert "long-range events:" in report
    assert "score:" in report
    assert "graph span:" in report


def test_mps_optimizer_layout_run_copies_submpo_payloads():
    """Layout replay should remap sub-MPO copies without mutating the stream."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [py.MpsOptimizer.submpo_event(mpo, (0, 3))]
    ref = py.MpsOptimizer(
        p0.copy(),
        gates=stream,
        chi=16,
        mode="mpo",
    ).run(progbar=False, cutoff=1e-12)

    opt = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    out = opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
    )

    inds = ["k0", "k1", "k2", "k3"]
    assert np.allclose(out.to_dense(inds), ref.to_dense(inds))
    assert out.outer_inds() == tuple(inds)
    assert out.site_inds == tuple(inds)
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)
    assert opt.where == [(0, 3)]


def test_mps_optimizer_mpo_mode_applies_submpo_stream_event():
    """MPO mode should apply explicit sparse sub-MPO stream events."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    )

    out = opt.run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
        normalize_final=False,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert opt.event_types == ["submpo"]
    assert opt.where == [(1, 3)]
    assert np.allclose(vec, expected)
    assert out.max_bond() <= 8


def test_mps_optimizer_dmrg_mode_applies_submpo_as_layered_fit_target():
    """DMRG keeps an explicit sub-MPO lazy and fits its tagged target layer."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=2,
        mode="dmrg2",
    )

    out = opt.run(
        progbar=False,
        cutoff=1.0e-12,
        non_unitary=True,
        normalize_final=False,
        fit_overlap_diagnostics=True,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert np.allclose(vec, expected)
    diagnostics = opt.get_fit_diagnostics()
    assert diagnostics["target_strategy"] == "layered"
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["fit_overlap_fidelity"] == pytest.approx(1.0)


def test_mps_optimizer_mpo_mode_accepts_submpo_mapping_event():
    """Mapping events should provide a readable public sub-MPO stream API."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 2), targets=(0, 2))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[{"kind": "submpo", "mpo": mpo, "where": [0, 2]}],
        chi=8,
        mode="mpo",
    )

    out = opt.run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
        normalize_final=False,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[10] = 0.3

    assert opt.event_types == ["submpo"]
    assert opt.where == [(0, 2)]
    assert np.allclose(vec, expected)


def test_mps_optimizer_public_submpo_event_helpers():
    """Public helpers should own the sub-MPO stream event contract."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 2), targets=(0, 2))
    tuple_event = py.MpsOptimizer.submpo_event(mpo, [0, 2])
    mapping_event = {"kind": "submpo", "mpo": mpo, "where": [0, 2]}
    gate_event = (np.eye(2), (0,))

    assert py.MpsOptimizer.is_submpo_event(tuple_event)
    assert py.MpsOptimizer.is_submpo_event(mapping_event)
    assert not py.MpsOptimizer.is_submpo_event(gate_event)

    assert py.MpsOptimizer.submpo_event_parts(tuple_event) == (mpo, (0, 2))
    assert py.MpsOptimizer.submpo_event_parts(
        mapping_event,
        normalize_where=True,
    ) == (mpo, (0, 2))
    assert py.MpsOptimizer.submpo_event_parts(gate_event) is None
    assert py.optimizers.mps.is_submpo_event(mapping_event)
    assert py.optimizers.mps.normalize_submpo_where([0, 2]) == (0, 2)

    bad_mapping = {"kind": "submpo", "mpo": mpo}
    with pytest.raises(ValueError, match="mpo.*where"):
        py.MpsOptimizer.submpo_event_parts(bad_mapping)


def test_mps_optimizer_submpo_diagnostics_do_not_consume_event_mpo():
    """Applying a reusable event MPO should not mutate its payload."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(1, 3), targets=(1, 3))

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    )
    out = opt.run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
    )
    vec = out.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 0.7
    expected[5] = 0.3

    assert np.allclose(vec, expected)

    reuse = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (1, 3))],
        chi=8,
        mode="mpo",
    ).run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
        normalize_final=False,
    )
    reuse_vec = reuse.to_dense(["k0", "k1", "k2", "k3"]).reshape(-1)
    assert np.allclose(reuse_vec, expected)


def test_mps_optimizer_submpo_method_and_optimize_are_forwarded(monkeypatch):
    """Sub-MPO replay should expose compression method and optimizer choice."""
    p0 = qtn.MPS_computational_state("000000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=6, sites=(0, 5), targets=(0, 5))
    calls = []
    optimize = object()

    def fake_gate_with_submpo_(
        self,
        submpo,
        *,
        where=None,
        method="direct",
        info=None,
        optimize=None,
        **_kwargs,
    ):
        calls.append((submpo, tuple(where), method, optimize))
        if info is not None:
            info["cur_orthog"] = (min(where), min(where))
        return self

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        fake_gate_with_submpo_,
    )

    direct = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (0, 5))],
        chi=8,
        mode="mpo",
        contraction_opt=optimize,
    )
    direct.run(
        progbar=False,
        cutoff=0.0,
        submpo_method="direct",
    )

    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(mpo, (0, 5))],
        chi=8,
        mode="mpo",
        contraction_opt=optimize,
    )
    opt.run(
        progbar=False,
        cutoff=0.0,
        submpo_method="fit-zipup",
    )

    assert calls == [
        (mpo, (0, 5), "direct", None),
        (mpo, (0, 5), "fit-zipup", optimize),
    ]


def test_mps_optimizer_submpo_method_validation(monkeypatch):
    """Unknown sub-MPO methods should be rejected clearly."""
    p0 = qtn.MPS_computational_state("000000", dtype="complex128")
    short_mpo = _two_branch_flip_submpo(L=6, sites=(0, 3), targets=(0, 3))

    def fake_gate_with_submpo_(
        self,
        _submpo,
        *,
        where=None,
        method="direct",
        info=None,
        **_kwargs,
    ):
        if info is not None:
            info["cur_orthog"] = (min(where), min(where))
        return self

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        fake_gate_with_submpo_,
    )

    bad = py.MpsOptimizer(
        p0.copy(),
        gates=[py.MpsOptimizer.submpo_event(short_mpo, (0, 3))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="Unknown subMPO method"):
        bad.run(progbar=False, submpo_method="bad")


def test_mps_optimizer_submpo_stream_events_require_mpo_or_dmrg_mode():
    """SVD/swap/exact modes still reject sub-MPO stream events clearly."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=3, sites=(0, 2), targets=(0, 2))
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 2))],
        chi=8,
        mode="svd",
    )

    with pytest.raises(ValueError, match="require an MPO or DMRG mode"):
        opt.run(progbar=False)


def test_mps_optimizer_submpo_stream_validates_support_sites():
    """Sub-MPO support should be a unique in-range set of 1D sites."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=3, sites=(0, 2), targets=(0, 2))

    repeated = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 0))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="repeated site"):
        repeated.run(progbar=False)

    out_of_range = py.MpsOptimizer(
        p0.copy(),
        gates=[("submpo", mpo, (0, 3))],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="outside the MPS range"):
        out_of_range.run(progbar=False)


def test_mps_optimizer_default_inplace_false_keeps_input_unchanged():
    """Default construction should work on a copy and keep input state intact."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    p0_ref = p0.copy()
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0, gates=gates, chi=8, mode="svd")
    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.p is not p0
    assert np.allclose(p0.to_dense(), p0_ref.to_dense())
    assert not np.allclose(out.to_dense(), p0_ref.to_dense())


def test_mps_optimizer_inplace_true_updates_input_state():
    """inplace=True should optimize the original input state object."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    p0_ref = p0.copy()
    gates = [(qu.hadamard(), (1,))]

    opt = py.MpsOptimizer(p0, gates=gates, chi=8, mode="svd", inplace=True)
    out = opt.run(progbar=False, cutoff=1e-12)

    assert opt.p is p0
    assert out is p0
    assert not np.allclose(p0.to_dense(), p0_ref.to_dense())


def test_mps_optimizer_rejects_noncanonical_bundled_gate_aliases():
    """Bundled gate input should require the canonical list/tuple shapes."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")

    opt = py.MpsOptimizer(p0.copy(), gates=((qu.hadamard(), (1,)),), chi=8, mode="svd")
    assert opt.where == [(1,)]

    with pytest.raises(ValueError, match="exact shape"):
        py.MpsOptimizer(p0.copy(), gates=[(qu.hadamard(), (1,)), qu.CNOT()], chi=8, mode="svd")


def test_mps_optimizer_run_returns_state_for_empty_queue():
    """run() should return the managed MPS even when there are no gates."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")

    out = opt.run(progbar=False, cutoff=1e-12)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "perm", "svd", "exact"])
def test_mps_optimizer_run_returns_state_after_updates(mode):
    """run() should return the updated MPS for every execution mode."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    G = [qu.hadamard(), qu.CNOT()]
    where = [(1,), (0, 3)]
    gates = list(zip(G, where))
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)

    out = opt.run(progbar=False, cutoff=1e-12, n_iter=2)

    assert out is opt.p


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "perm", "svd"])
def test_mps_optimizer_one_site_unitary_preserves_cached_center_and_norm(mode):
    """A one-site unitary must not invent a new orthogonality center."""
    state = qtn.MPS_rand_state(
        6,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=213,
    )
    state.multiply_(2.0, spread_over="all")
    optimizer = py.MpsOptimizer(
        state,
        gates=[(qu.hadamard(), (0,))],
        chi=16,
        mode=mode,
    )

    optimizer.run(progbar=False, cutoff=0.0, n_iter=2)

    assert optimizer.info_c["cur_orthog"] == tuple(
        optimizer.p.calc_current_orthog_center()
    )

    optimizer.set_gates([(qu.CNOT(), (0, 1))])
    optimizer.run(progbar=False, cutoff=0.0, n_iter=2)

    assert optimizer.info_c["cur_orthog"] == tuple(
        optimizer.p.calc_current_orthog_center()
    )


def test_mps_optimizer_svd_forwards_cutoff_mode_to_final_compression(monkeypatch):
    """SVD mode should honor explicit cutoff_mode in its chi compression pass."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 3))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=2, mode="svd")
    calls = []
    original_left_compress = opt.p.left_compress

    def _recording_left_compress(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_left_compress(*args, **kwargs)

    monkeypatch.setattr(opt.p, "left_compress", _recording_left_compress)

    opt.run(progbar=False, cutoff=1.0e-9, cutoff_mode="rsum2")

    assert calls
    assert calls[-1]["cutoff"] == pytest.approx(1.0e-9)
    assert calls[-1]["cutoff_mode"] == "rsum2"


def test_mps_optimizer_non_unitary_flag_normalizes_one_site_gate():
    """non_unitary=True should normalize at run end for short streams."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)

    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (1,))], chi=8, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
    )

    events = opt.get_normalizations()
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)
    assert opt.p.exponent == pytest.approx(np.log10(2.0))
    assert len(events) == 1
    assert events[0]["step"] == 1
    assert events[0]["old_norm"] == pytest.approx(4.0)
    assert events[0]["span"] == (1, 1)
    assert events[0]["insert"] == 1
    assert events[0]["exponent"] == pytest.approx(np.log10(2.0))
    assert opt.info_c["cur_orthog"] == (1, 1)


def test_mps_optimizer_manual_normalize_accumulates_exponent():
    """Manual normalization should preserve represented norm via ``p.exponent``."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    p0[0].modify(data=2.0 * p0[0].data)

    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")
    old_norm = opt.normalize(insert=0)

    assert old_norm == pytest.approx(4.0)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)
    assert opt.p.exponent == pytest.approx(np.log10(2.0))


def test_mps_optimizer_manual_normalize_reuses_singleton_center(monkeypatch):
    """Default manual normalization should avoid a full norm and QR sweep."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    center = opt.info_c["cur_orthog"][0]
    opt.p[center].modify(data=3.0 * opt.p[center].data)

    def fail_full_normalize(*_args, **_kwargs):
        raise AssertionError("open-MPS normalization must use its tracked center")

    def fail_canonize(*_args, **_kwargs):
        raise AssertionError("a singleton center must not be moved")

    monkeypatch.setattr(qtn.MatrixProductState, "normalize", fail_full_normalize)
    monkeypatch.setattr(qtn.MatrixProductState, "canonize", fail_canonize)

    old_norm = opt.normalize()

    assert old_norm == pytest.approx(9.0)
    assert opt.info_c["cur_orthog"] == (center, center)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(3.0)
    assert opt.p.exponent == pytest.approx(np.log10(3.0))


def test_mps_optimizer_manual_normalize_rejects_zero_center_transactionally():
    """An undefined normalization must not alter scale or center metadata."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    opt = py.MpsOptimizer(p0, gates=[], chi=8, mode="svd")
    center = opt.info_c["cur_orthog"][0]
    opt.p[center].modify(data=np.zeros_like(opt.p[center].data))
    exponent = opt.p.exponent

    with pytest.raises(FloatingPointError, match="zero or non-finite"):
        opt.normalize()

    assert opt.info_c["cur_orthog"] == (center, center)
    assert opt.p.exponent == exponent


def test_mps_optimizer_non_unitary_default_does_not_normalize():
    """The non-unitary flag should not enable scale control by default."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
        (qu.hadamard(), (2,)),
        (_non_unitary_entangling_gate(), (2, 3)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt_none = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_final=False)
    opt_none.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=None,
        normalize_final=False,
    )

    assert opt.get_normalizations() == []
    assert opt_none.get_normalizations() == []
    assert opt.p.exponent == pytest.approx(0.0)
    assert opt_none.p.exponent == pytest.approx(0.0)


def test_mps_optimizer_non_unitary_scale_control_preserves_normalization():
    """Fast non-unitary scale control preserves normalization bookkeeping."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode="svd")
    opt.run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=True,
        normalize_every=True,
        normalize_final=True,
    )
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    events = opt.get_normalizations()
    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert opt.p.exponent == pytest.approx(sum(event["log10_scale"] for event in events))
    assert [event["step"] for event in events] == [1, 2, 3]
    assert [event["reason"] for event in events] == ["step", "step", "compression"]
    assert all(event["sites"] == (event["insert"],) for event in events)
    _assert_event_sites_locally_normalized(opt.p, events[-1])


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "svd"])
def test_mps_optimizer_non_unitary_compression_works_without_diagnostics(mode):
    """Non-unitary compression remains usable without diagnostic flags."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [
        (qu.hadamard(), (0,)),
        (qu.hadamard(), (1,)),
        (_non_unitary_entangling_gate(), (0, 1)),
    ]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")

    opt.run(
        progbar=False,
        cutoff=1e-12,
        n_iter=4,
        non_unitary=True,
        normalize_every=True,
    )

    assert opt.p.norm() > 0.0
    assert opt.get_normalizations()


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "perm", "svd"])
def test_mps_optimizer_normalization_insert_site_stays_inside_span(mode):
    """Normalization events should insert factors inside the canonical span."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 1)), (qu.CNOT(), (2, 3))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode=mode)
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")

    opt.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=1)

    assert opt.p.norm() == pytest.approx(1.0)
    assert [event["span"] for event in opt.get_normalizations()] == [(0, 1), (2, 3)]
    assert all(
        event["span"][0] <= event["insert"] <= event["span"][1]
        for event in opt.get_normalizations()
    )


def test_mps_optimizer_normalize_every_normalizes_every_step():
    """Enabled normalize_every should scale each replay step at one center."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, non_unitary=True, normalize_final=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    events = opt.get_normalizations()
    assert opt.p.norm() == pytest.approx(ref.p.norm())
    assert [event["step"] for event in events] == [1, 2, 3]
    assert [event["reason"] for event in events] == ["step", "compression", "step"]
    assert all(event["sites"] == (event["insert"],) for event in events)
    assert opt.p.exponent == pytest.approx(sum(event["log10_scale"] for event in events))
    _assert_event_sites_locally_normalized(opt.p, events[-1])


def test_mps_optimizer_normalize_final_can_be_disabled():
    """Per-step normalization should not depend on normalize_final."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    gates = [(scale, (0,)), (_non_unitary_entangling_gate(), (0, 1)), (scale, (0,))]

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    ref = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")
    opt.run(progbar=False, cutoff=1e-12, normalize_every=2, normalize_final=False, non_unitary=True)
    ref.run(progbar=False, cutoff=1e-12, non_unitary=True, normalize_every=False)

    assert opt.p.norm() == pytest.approx(ref.p.norm())
    events = opt.get_normalizations()
    assert [event["step"] for event in events] == [1, 2, 3]
    assert events[1]["reason"] == "compression"
    assert all(event["sites"] == (event["insert"],) for event in events)


def test_mps_optimizer_automatic_normalization_rejects_exact_mode():
    """Exact mode has no MPS canonicalization range for automatic normalization."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.array([[2.0, 0.0], [0.0, 0.5]], dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="exact")

    with pytest.raises(ValueError, match="not available in exact mode"):
        opt.run(progbar=False, non_unitary=True, normalize_every=True)


def test_mps_optimizer_exact_mode_keeps_canonical_metadata_separate():
    """Switching through exact mode rebuilds an MPS before canonical use."""
    p0 = qtn.MPS_computational_state("000", dtype="complex128")
    gates = [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 2))]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=8, mode="svd")

    opt.set_mode("exact")
    opt.run(progbar=False)
    assert opt.info_c == {}

    exact_dense = opt.to_dense()
    opt.set_gates([])
    opt.set_mode("svd")

    assert isinstance(opt.p, qtn.MatrixProductState)
    assert opt.info_c["cur_orthog"] not in (None, "calc")
    assert np.allclose(opt.p.to_dense().reshape(-1), exact_dense)


def test_mps_optimizer_canonical_modes_reject_cyclic_mps():
    """A periodic MPS has no exact one-tensor mixed-canonical norm."""
    cyclic = qtn.MPS_rand_state(
        4,
        bond_dim=2,
        phys_dim=2,
        cyclic=True,
        dtype="complex128",
        seed=92,
    )

    with pytest.raises(ValueError, match="open-boundary MPS"):
        py.MpsOptimizer(cyclic, gates=[], chi=4, mode="mpo")


def test_mps_optimizer_cyclic_rejection_is_transactional():
    """Rejected replacement and mode switches must preserve optimizer state."""
    cyclic = qtn.MPS_rand_state(
        4,
        bond_dim=2,
        phys_dim=2,
        cyclic=True,
        dtype="complex128",
        seed=93,
    )
    open_opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[],
        chi=4,
        mode="mpo",
        inplace=True,
    )
    original_state = open_opt.p
    original_info = dict(open_opt.info_c)
    assert not getattr(cyclic, "_pepsy_norm_includes_exponent", False)

    with pytest.raises(ValueError, match="open-boundary MPS"):
        open_opt.set_p(cyclic)

    assert open_opt.p is original_state
    assert open_opt.info_c == original_info
    assert not getattr(cyclic, "_pepsy_norm_includes_exponent", False)

    exact_opt = py.MpsOptimizer(
        cyclic,
        gates=[],
        chi=4,
        mode="exact",
    )
    with pytest.raises(ValueError, match="open-boundary MPS"):
        exact_opt.set_mode("svd")

    assert exact_opt.mode == "exact"
    assert exact_opt.info_c == {}


def test_mps_optimizer_persistent_layout_rejects_exact_mode_switch():
    """Exact mode cannot silently discard a persistent logical-site map."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"), gates=[], chi=4, mode="svd"
    )
    opt.apply_layout((2, 0, 1), layout_report=False)

    with pytest.raises(ValueError, match="persistent-layout"):
        opt.set_mode("exact")


def test_mps_optimizer_rejects_invalid_normalize_every():
    """normalize_every should fail clearly for non-positive intervals."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.eye(2, dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="svd")

    with pytest.raises(ValueError, match="normalize_every must be >= 1"):
        opt.run(progbar=False, normalize_every=0, non_unitary=True)


def test_mps_optimizer_normalization_options_require_non_unitary():
    """Normalization controls should not act as aliases for ``non_unitary=True``."""
    p0 = qtn.MPS_computational_state("00", dtype="complex128")
    scale = np.eye(2, dtype=complex)
    opt = py.MpsOptimizer(p0.copy(), gates=[(scale, (0,))], chi=8, mode="svd")

    with pytest.raises(ValueError, match="normalize_every requires non_unitary=True"):
        opt.run(progbar=False, normalize_every=1)

    with pytest.raises(ValueError, match="normalize_final requires non_unitary=True"):
        opt.run(progbar=False, normalize_final=True)


@pytest.mark.parametrize("where", [(1, 2), (0, 3)])
def test_mps_optimizer_canonical_span_norm_matches_full_target_norm(where):
    """Canonical span norm should match full norm for split-gate targets."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128")
    gate = _non_unitary_entangling_gate()
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")

    xmin, xmax = sorted(where)
    opt.canonize_mps(opt.p, (xmin, xmax))
    target = opt._build_norm_target(  # pylint: disable=protected-access
        opt.p,
        gate,
        where,
        cutoff=1e-12,
        cutoff_mode="rel",
    )

    local_norm = opt._canonical_span_norm(target, (xmin, xmax))  # pylint: disable=protected-access
    assert local_norm == pytest.approx(target.norm())


def test_mps_optimizer_target_norm_does_not_mutate_live_canonical_metadata():
    """Temporary norm targets must not overwrite the live MPS center cache."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=8),
        gates=[],
        chi=8,
        mode="svd",
    )
    opt.canonize_mps(opt.p, (0, 1))
    before = dict(opt.info_c)
    target = opt._build_norm_target(  # pylint: disable=protected-access
        opt.p,
        _non_unitary_entangling_gate(),
        (0, 3),
        cutoff=1e-12,
    )

    measured = opt._canonical_span_norm(  # pylint: disable=protected-access
        target, (0, 3)
    )

    assert measured == pytest.approx(target.norm())
    assert opt.info_c == before


def test_mps_optimizer_local_normalization_reuses_tracked_center(monkeypatch):
    """Local scale control should not rescan a live canonical MPS."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=9),
        gates=[],
        chi=8,
        mode="svd",
    )
    opt.canonize_mps(opt.p, (0, 2))

    def fail_scan(*args, **kwargs):
        raise AssertionError("normalization should reuse the tracked centre")

    monkeypatch.setattr(qtn.MatrixProductState, "calc_current_orthog_center", fail_scan)
    event = opt._normalize_orthog_tensors(  # pylint: disable=protected-access
        opt.p,
        (0, 2),
        step=1,
        reason="test",
        canonicalize=False,
    )

    assert event is not None
    assert event["insert"] == 2
    assert event["sites"] == (2,)
    assert opt.info_c["cur_orthog"] == (2, 2)


def test_mps_optimizer_local_normalization_keeps_singleton_center(monkeypatch):
    """Scale control should reuse FIT's endpoint instead of sweeping the span."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(
            4,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=91,
        ),
        gates=[],
        chi=8,
        mode="dmrg2",
    )
    opt.canonize_mps(opt.p, 0)
    opt.p[0].modify(data=2.0 * opt.p[0].data)

    def fail_canonize(*_args, **_kwargs):
        raise AssertionError("normalization moved an authoritative FIT center")

    monkeypatch.setattr(qtn.MatrixProductState, "canonize", fail_canonize)
    event = opt._normalize_orthog_tensors(
        opt.p,
        (0, 3),
        step=1,
        reason="test",
        canonicalize=False,
    )

    assert event["span"] == (0, 3)
    assert event["insert"] == 0
    assert event["sites"] == (0,)
    assert opt.info_c["cur_orthog"] == (0, 0)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0)
    assert opt.p.norm() == pytest.approx(2.0)


def test_mps_optimizer_canonical_span_norm_ignores_stored_exponent():
    """Internal normalization should measure raw data, not represented scale."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128")
    opt = py.MpsOptimizer(p0.copy(), gates=[], chi=8, mode="svd")
    opt.p.exponent = 3.0

    raw = opt.p.copy()
    raw.exponent = 0.0
    measured = opt._canonical_span_norm(opt.p, (0, 3))  # pylint: disable=protected-access

    assert measured == pytest.approx(raw.norm())
    assert opt.p.exponent == pytest.approx(3.0)


def test_mps_optimizer_represented_norm_capability_check_is_cached(monkeypatch):
    """Repeated FIT results should not re-contract the full MPS norm."""
    p0 = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=13)
    cache = mps_optimizer_module._NORM_INCLUDES_EXPONENT_CACHE  # pylint: disable=protected-access
    cache.pop(type(p0), None)
    original = py.MpsOptimizer._class_norm_includes_exponent
    calls = []

    def count_checks(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(
        py.MpsOptimizer,
        "_class_norm_includes_exponent",
        staticmethod(count_checks),
    )
    py.MpsOptimizer._install_represented_norm(p0.copy())  # pylint: disable=protected-access
    py.MpsOptimizer._install_represented_norm(p0.copy())  # pylint: disable=protected-access

    assert len(calls) == 1


def test_mps_optimizer_dmrg_non_unitary_matches_mpo_accuracy():
    """DMRG should match MPO accuracy for normalized non-unitary MPS updates."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    h_gate = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    filter_gate = _non_unitary_entangling_gate()
    gates = [
        (h_gate, (0,)),
        (h_gate, (1,)),
        (h_gate, (2,)),
        (h_gate, (3,)),
        (filter_gate, (0, 3)),
        (filter_gate, (1, 2)),
        (filter_gate, (0, 1)),
    ]

    target = p0.copy()
    for gate, where in gates:
        py.gate(target, gate, where, contract=True, cutoff=1e-12, inplace=True)

    results = {}
    for mode in ("dmrg", "mpo"):
        opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=1, mode=mode)
        opt.run(
            progbar=False,
            cutoff=1e-12,
            n_iter=20,
            # This legacy accuracy comparison intentionally requests the old
            # fixed-sweep behavior; the public default is adaptive FIT.
            fit_rtol=None,
            non_unitary=True,
            normalize_every=1,
        )

        fidelity = float(np.real(py.tn_fidelity(opt.p, target, contraction_opt="auto-hq")))
        results[mode] = {
            "fidelity": fidelity,
            "represented_norm": float(np.real(opt.p.norm())),
        }

        events = opt.get_normalizations()
        assert len(events) == len(gates)
        assert [event["step"] for event in events] == list(range(1, len(gates) + 1))
        assert [event["reason"] for event in events] == [
            "step",
            "step",
            "step",
            "step",
            "compression",
            "compression",
            "compression",
        ]
        assert all(event["sites"] == (event["insert"],) for event in events)
        _assert_event_sites_locally_normalized(opt.p, events[-1])
        assert fidelity > 0.92

    assert results["dmrg"]["fidelity"] == pytest.approx(
        results["mpo"]["fidelity"],
        abs=5e-10,
    )
    assert results["dmrg"]["represented_norm"] == pytest.approx(
        results["mpo"]["represented_norm"],
        abs=5e-10,
    )
# --------------------------------------------------------------------------- #
# Control events: measure / cap / reset
# --------------------------------------------------------------------------- #
_PAULI_1Q_TEST = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _dense_pauli_expectation(mps, pauli, where):
    """Return ``<psi|P|psi> / <psi|psi>`` from the dense statevector."""
    psi = mps.to_dense().reshape(-1)
    ops = [np.eye(2, dtype=complex) for _ in range(mps.L)]
    for axis, site in zip(pauli, where):
        ops[site] = _PAULI_1Q_TEST[axis]
    operator = ops[0]
    for op in ops[1:]:
        operator = np.kron(operator, op)
    return complex(psi.conj() @ (operator @ psi) / (psi.conj() @ psi)).real


def _full_network_pauli_expectation(mps, pauli, where, optimize="auto-hq"):
    """Return a Pauli expectation from an explicit full MPS overlap."""
    op = _PAULI_1Q_TEST[pauli[0]]
    for axis in pauli[1:]:
        op = np.kron(op, _PAULI_1Q_TEST[axis])

    acted = mps.copy()
    acted.gate_nonlocal_(
        op,
        tuple(int(site) for site in where),
        max_bond=None,
        info={},
        method="direct",
        cutoff=0.0,
        cutoff_mode="abs",
    )
    numerator = (mps.H & acted).contract(all, output_inds=(), optimize=optimize)
    denominator = (mps.H & mps).contract(all, output_inds=(), optimize=optimize)
    return float(np.real(complex(numerator / denominator)))


def test_mps_optimizer_measure_forced_outcome_collapses_and_records():
    """A forced measurement should collapse the state and record the result."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    opt = py.MpsOptimizer(m.copy(), [("measure", "Z", 2, +1)], chi=8, mode="mpo")
    opt.run(progbar=False)

    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (2,)), 1.0)
    assert np.isclose(float(abs(opt.p.norm())), 1.0)
    assert len(opt.measurements) == 1
    pauli, where, outcome, prob = opt.measurements[0]
    assert pauli == "Z"
    assert where == (2,)
    assert outcome == 1
    assert 0.0 <= prob <= 1.0
    event = opt.get_norm_events()[0]
    assert event["kind"] == "measure"
    assert event["branch_probability"] == pytest.approx(prob)
    assert event["physical_boundary"] is True
    assert event["renormalized"] is True
    assert event["local_infidelity"] == pytest.approx(0.0, abs=1e-10)


def test_mps_optimizer_measure_multisite_pauli():
    """Multi-qubit Pauli measurements should collapse onto the eigenspace."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    opt = py.MpsOptimizer(m.copy(), [("measure", "ZZ", (1, 3), -1)], chi=8, mode="mpo")
    opt.run(progbar=False)

    assert np.isclose(_dense_pauli_expectation(opt.p, "ZZ", (1, 3)), -1.0)
    assert opt.measurements[0][:3] == ("ZZ", (1, 3), -1)


@pytest.mark.parametrize(
    ("mode", "expected_method"),
    [("mpo", "direct"), ("quimb-src", "src")],
)
def test_mps_optimizer_multisite_measurement_uses_bond_two_submpo(
    monkeypatch, mode, expected_method
):
    """Dense MPS measurements should use the low-bond sub-MPO compressor."""
    calls = []
    original = qtn.MatrixProductState.gate_with_submpo_

    def recording(self, submpo, *args, **kwargs):
        calls.append(
            (
                kwargs.get("method"),
                tuple(kwargs.get("where", ())),
                submpo.max_bond(),
                kwargs.get("max_bond"),
            )
        )
        return original(self, submpo, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        recording,
    )

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 2, seed=2, dtype="complex128"),
        [("measure", "XZY", (1, 3, 5), +1)],
        chi=8,
        mode=mode,
    )
    opt.run(progbar=False, cutoff=0.0)

    # Probability preparation is lossless; only the final physical projection
    # consumes the requested compressor and chi.
    assert calls == [
        ("direct", (5, 3), 2, None), ("direct", (3, 1), 2, None),
        (expected_method, (1, 2, 3, 4, 5), 2, 8),
    ]
    assert _dense_pauli_expectation(opt.p, "XZY", (1, 3, 5)) == pytest.approx(
        1.0
    )


def test_mps_optimizer_dmrg_measurement_uses_lazy_submpo_and_src_guess(monkeypatch):
    """DMRG measurements should use a lazy target and the normal SRC guess."""
    methods = []
    original = qtn.MatrixProductState.gate_with_submpo_

    def recording(self, submpo, *args, **kwargs):
        methods.append(kwargs.get("method"))
        return original(self, submpo, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "gate_with_submpo_",
        recording,
    )

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 2, seed=2, dtype="complex128"),
        [("measure", "XZY", (1, 3, 5), +1)],
        chi=8,
        mode="dmrg2",
    )
    opt.run(
        progbar=False,
        n_iter=3,
        fit_min_iter=1,
        fit_patience=1,
        cutoff=0.0,
    )

    diagnostics = opt.get_fit_diagnostics()
    assert methods == ["direct", "direct", "lazy", "src"]
    assert diagnostics["target_representation"] == "lazy_submpo"
    assert diagnostics["guess_method"] == "src"
    assert diagnostics["fallback"] is False
    assert _dense_pauli_expectation(opt.p, "XZY", (1, 3, 5)) == pytest.approx(
        1.0
    )


def test_mps_optimizer_expectation_uses_local_canonical_path(monkeypatch):
    """MPS expectations should use Quimb's local canonical evaluator."""
    calls = []
    original = qtn.MatrixProductState.local_expectation_canonical

    def counting(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "local_expectation_canonical", counting)

    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=7), gates=[], chi=8, mode="mpo"
    )
    observed = opt._state_expectation("ZZ", (1, 4))  # pylint: disable=protected-access
    expected = _dense_pauli_expectation(opt.p, "ZZ", (1, 4))

    assert observed == pytest.approx(expected)
    assert len(calls) == 1
    assert calls[0]["normalized"] is True
    assert calls[0]["info"] is opt.info_c


def test_mps_optimizer_expectation_converts_operator_to_state_backend(monkeypatch):
    """The local expectation operator should pass through backend conversion."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, 2, seed=8), gates=[], chi=4, mode="mpo"
    )
    converted = []
    original = opt._to_state_backend

    def recording(array):
        converted.append(array)
        return original(array)

    monkeypatch.setattr(opt, "_to_state_backend", recording)
    opt._state_expectation("Z", (1,))  # pylint: disable=protected-access

    assert len(converted) == 1
    assert converted[0].shape == (2, 2)


def test_mps_optimizer_local_expectation_uses_torch_state_backend(monkeypatch):
    """Torch-backed control expectations should stay on the Torch backend."""
    torch = pytest.importorskip("torch")
    vector = torch.tensor([1.0, 1.0], dtype=torch.complex128)
    vector = vector / torch.linalg.vector_norm(vector)
    opt = py.MpsOptimizer(
        qtn.MPS_product_state([vector.clone() for _ in range(3)]),
        gates=[],
        chi=4,
        mode="mpo",
    )
    observed_operators = []
    original = qtn.MatrixProductState.local_expectation_canonical

    def recording(self, operator, *args, **kwargs):
        observed_operators.append(operator)
        return original(self, operator, *args, **kwargs)

    monkeypatch.setattr(
        qtn.MatrixProductState,
        "local_expectation_canonical",
        recording,
    )
    assert opt._state_expectation("Z", (1,)) == pytest.approx(0.0)

    assert isinstance(observed_operators[0], torch.Tensor)
    assert all(isinstance(tensor.data, torch.Tensor) for tensor in opt.p.tensors)


@pytest.mark.parametrize(
    ("pauli", "where"),
    [("X", (4,)), ("YZ", (1, 4))],
)
def test_mps_optimizer_expectation_reuses_tracked_center_without_rescan(
    monkeypatch, pauli, where
):
    """Local Pauli expectations should move a known centre without rescanning."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=11), gates=[], chi=8, mode="mpo"
    )
    opt.canonize_mps(opt.p, 0)
    assert opt.info_c["cur_orthog"] == (0, 0)

    def fail_scan(*args, **kwargs):
        raise AssertionError("expectation should reuse the tracked canonical centre")

    monkeypatch.setattr(qtn.MatrixProductState, "calc_current_orthog_center", fail_scan)

    observed = opt._state_expectation(pauli, where)  # pylint: disable=protected-access
    expected = _dense_pauli_expectation(opt.p, pauli, where)

    assert observed == pytest.approx(expected)
    center = opt.info_c["cur_orthog"]
    assert center[0] == center[1]
    assert min(where) <= center[0] <= max(where)


@pytest.mark.parametrize(
    ("pauli", "where"),
    [("X", (4,)), ("YZ", (1, 4))],
)
def test_mps_optimizer_local_expectation_matches_full_network(pauli, where):
    """Local canonical and full-network Pauli expectations should agree."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=13), gates=[], chi=8, mode="mpo"
    )

    local = opt._state_expectation(pauli, where)  # pylint: disable=protected-access
    full = _full_network_pauli_expectation(opt.p, pauli, where)

    assert local == pytest.approx(full, abs=1e-10)


def test_mps_optimizer_sync_canonicalization_repairs_external_readout():
    """External Quimb readout can be explicitly rebound to ``info_c``."""
    z = np.diag([1.0, -1.0]).astype(complex)
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=17), gates=[], chi=8, mode="dmrg2"
    )
    opt.canonize_mps(opt.p, 0)
    assert opt.info_c["cur_orthog"] == (0, 0)

    # This deliberately models a lower-level caller bypassing Pepsy's
    # tracked expectation helper.
    opt.p.local_expectation_canonical(z, (5,), normalized=True)
    assert opt.info_c["cur_orthog"] == (0, 0)
    assert tuple(opt.p.calc_current_orthog_center()) == (5, 5)

    assert opt.sync_canonicalization() == (5, 5)
    assert opt.info_c["cur_orthog"] == (5, 5)

    opt.set_gates([(np.eye(4, dtype=complex), (0, 1))])
    opt.run(progbar=False, n_iter=1, cutoff=0.0)
    assert tuple(opt.p.calc_current_orthog_center()) == opt.info_c["cur_orthog"]


def test_mps_optimizer_measure_born_statistics():
    """Sampled outcomes should follow the Born rule for a biased qubit."""
    theta = np.pi / 3
    ry = np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=complex,
    )
    n_shots = 800
    plus = 0
    for shot in range(n_shots):
        opt = py.MpsOptimizer(
            qtn.MPS_computational_state("0"),
            [(ry, (0,)), ("measure", "Z", 0)],
            chi=2,
            mode="mpo",
        )
        opt.run(progbar=False, seed=shot)
        if opt.measurements[0][2] == 1:
            plus += 1
    expected = np.cos(theta / 2) ** 2
    assert abs(plus / n_shots - expected) < 0.05


def test_mps_optimizer_measure_forced_zero_probability_raises():
    """Forcing an impossible outcome should fail clearly."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0"),
        [("measure", "Z", 0, -1)],
        chi=2,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="probability"):
        opt.run(progbar=False)


def test_mps_optimizer_cap_matches_dense_projection_and_shortens():
    """A cap event should shorten the MPS and match the dense contraction."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    vec = np.array([1.0, 1.0])
    dense = m.to_dense().reshape([2] * 6)
    expected = np.tensordot(dense, vec, axes=([2], [0]))

    for absorb in ("left", "right"):
        opt = py.MpsOptimizer(m.copy(), [("cap", 2, vec, absorb)], chi=8, mode="mpo")
        opt.run(progbar=False)
        assert isinstance(opt.p, qtn.MatrixProductState)
        assert opt.p.L == 5
        got = opt.p.to_dense().reshape([2] * 5)
        assert np.allclose(got, expected)


def test_mps_optimizer_cap_boundary_sites():
    """Capping the first or last site should stay a valid shorter MPS."""
    m = qtn.MPS_rand_state(5, 3, seed=7)
    dense = m.to_dense().reshape([2] * 5)

    first = py.MpsOptimizer(m.copy(), [("cap", 0, [1.0, 0.0])], chi=8, mode="svd")
    first.run(progbar=False)
    assert first.p.L == 4
    assert np.allclose(
        first.p.to_dense().reshape([2] * 4),
        np.tensordot([1.0, 0.0], dense, axes=([0], [0])),
    )

    last = py.MpsOptimizer(m.copy(), [("cap", 4, [0.0, 1.0])], chi=8, mode="svd")
    last.run(progbar=False)
    assert last.p.L == 4
    assert np.allclose(
        last.p.to_dense().reshape([2] * 4),
        np.tensordot(dense, [0.0, 1.0], axes=([4], [0])),
    )


def test_mps_optimizer_cap_length_one_raises():
    """Capping the single site of a length-1 MPS should fail clearly."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0"),
        [("cap", 0, [1.0, 1.0])],
        chi=2,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="length-1"):
        opt.run(progbar=False)


def test_mps_optimizer_reset_returns_qubit_to_zero():
    """Reset should leave the target qubit in |0> without changing length."""
    hadamard = qu.hadamard()
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"),
        [(hadamard, (1,)), ("reset", 1)],
        chi=4,
        mode="mpo",
    )
    opt.run(progbar=False, seed=0)

    assert opt.p.L == 3
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (1,)), 1.0)
    assert opt.measurements == []
    assert [event["kind"] for event in opt.get_norm_events()] == ["reset"]
    assert opt.norm_diagnostics()["infidelity"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_mps_optimizer_reset_supports_pauli_bases(axis):
    """Reset should return the target to the +1 eigenstate of X/Y/Z."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0"),
        [(qu.hadamard(), (0,)), ("reset", 0, axis)],
        chi=4,
        mode="mpo",
    )
    opt.run(progbar=False, seed=7)

    assert opt.p.L == 1
    assert np.isclose(_dense_pauli_expectation(opt.p, axis, (0,)), 1.0)
    assert opt.measurements == []


@pytest.mark.parametrize(
    ("axis", "bits", "outcome"),
    [("Z", "1", -1), ("X", "0", -1), ("Y", "0", -1)],
)
def test_mps_optimizer_measure_reset_records_then_resets(axis, bits, outcome):
    """MR should record the measured eigenvalue and leave the + basis state."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state(bits),
        [("measure_reset", axis, 0, outcome)],
        chi=4,
        mode="mpo",
    )
    opt.run(progbar=False)

    assert opt.measurements[0][:3] == (axis, (0,), outcome)
    assert np.isclose(_dense_pauli_expectation(opt.p, axis, (0,)), 1.0)


@pytest.mark.parametrize("mode", ["dmrg", "mpo", "mix", "swap", "perm", "svd", "exact"])
def test_mps_optimizer_control_events_all_modes(mode):
    """measure/cap/reset should work in every run mode."""
    m = qtn.MPS_rand_state(6, 4, seed=2)
    opt = py.MpsOptimizer(
        m.copy(),
        [("measure", "Z", 2, +1), ("reset", 0), ("cap", 4, [1.0, 1.0])],
        chi=8,
        mode=mode,
    )
    if mode == "swap" and not hasattr(opt.p, "gate_with_auto_swap_"):
        pytest.skip("swap mode requires gate_with_auto_swap_ in this quimb version.")
    opt.run(progbar=False, seed=3)

    assert opt.p.L == 5
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (2,)), 1.0)
    assert len(opt.measurements) == 1


def test_mps_optimizer_gates_and_control_interleaved():
    """Gates and control events should interleave and stay consistent."""
    hadamard = qu.hadamard()
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        [
            (hadamard, (0,)),
            (cnot, (0, 1)),
            ("measure", "Z", 0, +1),
            ("cap", 3, [1.0, 1.0]),
        ],
        chi=8,
        mode="mpo",
    )
    opt.run(progbar=False)

    # H then CNOT builds a Bell pair on (0, 1); forcing Z_0 = +1 puts both in |0>.
    assert opt.p.L == 3
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (0,)), 1.0)
    assert np.isclose(_dense_pauli_expectation(opt.p, "Z", (1,)), 1.0)
    assert opt.measurements[0][:3] == ("Z", (0,), 1)


def test_mps_optimizer_control_event_seed_is_reproducible():
    """The same seed should reproduce sampled measurement outcomes."""
    hadamard = qu.hadamard()
    stream = [(hadamard, (0,)), ("measure", "Z", 0)]
    first = py.MpsOptimizer(qtn.MPS_computational_state("0"), stream, chi=2, mode="mpo")
    first.run(progbar=False, seed=123)
    second = py.MpsOptimizer(qtn.MPS_computational_state("0"), stream, chi=2, mode="mpo")
    second.run(progbar=False, seed=123)
    assert first.measurements[0][2] == second.measurements[0][2]


def test_mps_optimizer_control_event_mapping_forms():
    """Mapping-form control events should parse into the same queue metadata."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000"),
        [
            {"kind": "measure", "pauli": "Z", "where": 1, "outcome": +1},
            {"kind": "cap", "where": 2, "vec": [1.0, 1.0]},
        ],
        chi=4,
        mode="mpo",
    )
    assert opt.event_types == ["measure", "cap"]
    assert opt.where == [(1,), (2,)]
    opt.run(progbar=False)
    assert opt.p.L == 2
    assert opt.measurements[0][:3] == ("Z", (1,), 1)


def test_mps_optimizer_control_event_public_helpers():
    """Public event builders and detectors should own the control contract."""
    measure = py.MpsOptimizer.measure_event("Z", 2, +1)
    cap = py.MpsOptimizer.cap_event(1, [1, 1], absorb="right")
    reset = py.MpsOptimizer.reset_event([0, 3])
    reset_x = py.MpsOptimizer.reset_event(0, basis="X")
    measure_reset = py.MpsOptimizer.measure_reset_event("Y", 1, -1)

    assert measure == ("measure", "Z", (2,), 1)
    assert cap[0] == "cap" and cap[1] == 1 and cap[3] == "right"
    assert reset == ("reset", (0, 3))
    assert reset_x == ("reset", (0,), "X")
    assert measure_reset == ("measure_reset", "Y", (1,), -1)

    assert py.MpsOptimizer.is_control_event(measure)
    assert py.MpsOptimizer.is_control_event(cap)
    assert py.MpsOptimizer.is_control_event(measure_reset)
    assert py.MpsOptimizer.is_control_event(("mrx", 0, -1))
    assert not py.MpsOptimizer.is_control_event((np.eye(2), (0,)))

    name, payload, where = py.MpsOptimizer.control_event_parts(measure)
    assert name == "measure"
    assert payload["pauli"] == "Z"
    assert payload["outcome"] == 1
    assert where == (2,)


@pytest.mark.parametrize(
    "conditional",
    [
        ("if", -1, 1, (np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), 1)),
        {
            "kind": "feed_forward",
            "record": -1,
            "value": 1,
            "then": (np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), 1),
        },
    ],
)
def test_mps_optimizer_conditional_gate_follows_measurement_bit(conditional):
    """MPS feed-forward applies only the matching classical branch."""
    hadamard = np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=complex
    ) / np.sqrt(2.0)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    for outcome, expected_index in ((+1, 0), (-1, 3)):
        entry = conditional.copy() if isinstance(conditional, dict) else conditional
        if isinstance(entry, dict):
            entry["then"] = (flip, 1)
        else:
            entry = (entry[0], entry[1], entry[2], (flip, 1))
        optimizer = py.MpsOptimizer(
            qtn.MPS_computational_state("00", dtype="complex128"),
            [(hadamard, 0), ("measure", "Z", 0, outcome), entry],
            chi=4,
            mode="mpo",
        )
        optimizer.run(progbar=False)
        state = optimizer.to_dense().reshape(-1)
        expected = np.zeros(4, dtype=complex)
        expected[expected_index] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)


def test_mps_optimizer_measure_reset_support_layout_finder():
    """measure/reset should replay correctly under the layout finder."""
    su4 = qu.rand_uni(4, seed=5)
    hadamard = qu.hadamard()
    stream = [
        (su4, (0, 7)),
        (su4, (1, 6)),
        (hadamard, (3,)),
        ("measure", "Z", 3, +1),
        (su4, (2, 5)),
        ("reset", 0),
        ("measure", "ZZ", (1, 6), +1),
    ]
    init = qtn.MPS_computational_state("0" * 8, dtype="complex128")

    ref = py.MpsOptimizer(init.copy(), list(stream), chi=32, mode="mpo")
    ref.run(progbar=False, seed=7)

    lay = py.MpsOptimizer(init.copy(), list(stream), chi=32, mode="mpo")
    lay.run(progbar=False, seed=7, use_layout_finder=True, layout_report=False)

    inds = [f"k{i}" for i in range(8)]
    assert isinstance(lay.p, qtn.MatrixProductState)
    assert lay.p.site_inds == tuple(inds)
    # Recorded sites use logical labels, not layout-order labels.
    assert [rec[:2] for rec in lay.measurements] == [("Z", (3,)), ("ZZ", (1, 6))]
    assert [rec[:2] for rec in ref.measurements] == [("Z", (3,)), ("ZZ", (1, 6))]
    assert np.allclose(np.abs(lay.p.to_dense(inds)), np.abs(ref.p.to_dense(inds)))


def test_mps_optimizer_cap_events_reject_layout_finder():
    """cap events change the MPS length, so the layout finder is rejected."""
    su4 = qu.rand_uni(4, seed=1)
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        [(su4, (0, 3)), ("cap", 1, [1.0, 1.0])],
        chi=8,
        mode="mpo",
    )
    with pytest.raises(ValueError, match="cap control"):
        opt.run(progbar=False, use_layout_finder=True)


def test_mps_optimizer_control_events_track_canonical_center(monkeypatch):
    """Control events move the orthogonality centre explicitly (never a rescan)."""
    calls = {"n": 0}
    original = qtn.MatrixProductState.calc_current_orthog_center

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    su4 = qu.rand_uni(4, seed=5)
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(6, 4, seed=2),
        [
            (su4, (0, 3)),
            ("measure", "Z", 2, +1),
            ("reset", 0),
            ("measure", "ZZ", (1, 4), +1),
            ("cap", 4, [1.0, 1.0]),
        ],
        chi=16,
        mode="mpo",
    )
    # Prime the queued gate segment (which legitimately locates the centre once),
    # then assert no rescans happen while the control events run.
    monkeypatch.setattr(
        qtn.MatrixProductState, "calc_current_orthog_center", counting
    )
    opt.run(progbar=False, seed=1)

    assert calls["n"] == 0
    center = opt.info_c.get("cur_orthog")
    assert isinstance(center, tuple) and len(center) == 2
    assert center not in ("calc", None)
    # The tracked centre is a genuine orthogonality centre of the final MPS.
    canonical = opt.p.copy()
    canonical.canonize(list(center))
    assert np.allclose(
        np.abs(canonical.to_dense()), np.abs(opt.p.to_dense())
    )


def test_mps_optimizer_set_p_rebases_unitary_stabilization_norm():
    """A replacement state must not inherit the prior raw-norm baseline."""
    identity = np.eye(4, dtype=np.complex128)
    optimizer = py.MpsOptimizer(
        qtn.MPS_computational_state("00", dtype="complex128"),
        [(identity, (0, 1))],
        chi=2,
        mode="dmrg2",
    )
    optimizer.run(progbar=False, n_iter=1)

    replacement = qtn.MPS_computational_state("00", dtype="complex128")
    replacement[0].modify(data=3.0 * replacement[0].data)
    optimizer.set_p(replacement)

    assert optimizer._unitary_previous_norm is None  # pylint: disable=protected-access
    optimizer.run(progbar=False, n_iter=1)

    assert _mps_data_norm(optimizer.p) == pytest.approx(3.0)


def test_mps_optimizer_normalize_rebases_raw_unitary_stabilization_norm():
    """Manual normalization preserves represented scale without restoring it twice."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    state[0].modify(data=3.0 * state[0].data)
    optimizer = py.MpsOptimizer(
        state,
        [(np.eye(4, dtype=np.complex128), (0, 1))],
        chi=2,
        mode="dmrg2",
    )
    optimizer.run(progbar=False, n_iter=1)

    optimizer.normalize(insert=0)
    optimizer.run(progbar=False, n_iter=1)

    assert _mps_data_norm(optimizer.p) == pytest.approx(1.0)
    assert optimizer.p.norm() == pytest.approx(3.0)


@pytest.mark.parametrize("mode", ["mpo", "swap", "perm", "svd"])
def test_standalone_compression_modes_honor_unitary_stabilization(mode):
    """Every compressed mode can restore norm after a lossy unitary split."""
    gates = [
        (qu.hadamard(dtype="complex64"), (0,)),
        (qu.CNOT(dtype="complex64"), (0, 2)),
    ]
    stabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex64"),
        gates,
        chi=1,
        mode=mode,
    )
    stabilized.run(
        progbar=False,
        cutoff=0.0,
        stabilize_unitary=True,
        timing=True,
    )
    unstabilized = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex64"),
        gates,
        chi=1,
        mode=mode,
    )
    unstabilized.run(
        progbar=False,
        cutoff=0.0,
        stabilize_unitary=False,
    )

    assert _mps_data_norm(stabilized.p) == pytest.approx(1.0, abs=2.0e-5)
    assert _mps_data_norm(unstabilized.p) < 0.999
    assert stabilized.norm_diagnostics()["infidelity"] == pytest.approx(
        0.5, abs=2.0e-5
    )
    assert unstabilized.norm_diagnostics()["infidelity"] == pytest.approx(
        0.5, abs=2.0e-5
    )
    assert stabilized.get_norm_events()[0]["kind"] == "unitary_compression"
    timing_name = "direct" if mode == "mpo" else mode
    assert stabilized.get_run_timing()["stages"][f"{timing_name}.stabilize"]["calls"] == 1


@pytest.mark.parametrize("mode", ["dmrg1", "dmrg2", "dmrg3"])
def test_dmrg_schedules_record_automatic_norm_survival(mode):
    """All named DMRG schedules use the same automatic norm ledger."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 2))],
        chi=1,
        mode=mode,
    )
    opt.run(progbar=False, cutoff=0.0, n_iter=3, stabilize_unitary=True)

    diagnostics = opt.norm_diagnostics()
    assert diagnostics["events"] == 1
    assert diagnostics["infidelity"] == pytest.approx(0.5, abs=2.0e-5)
    assert _mps_data_norm(opt.p) == pytest.approx(1.0, abs=2.0e-5)


def test_mps_norm_names_and_dmrg_target_overlap_are_distinct():
    """DMRG exposes target overlap separately from retained norm fidelity."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        [(qu.hadamard(), (0,)), (qu.CNOT(), (0, 2))],
        chi=1,
        mode="dmrg",
    )
    opt.run(
        progbar=False,
        cutoff=0.0,
        n_iter=3,
        stabilize_unitary=True,
        fit_overlap_diagnostics=True,
    )

    diagnostics = opt.norm_diagnostics()
    fit = opt.get_fit_diagnostics()
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
    assert diagnostics["local_fidelity"] == pytest.approx(0.5, abs=2e-5)
    assert diagnostics["cumulative_fidelity"] == pytest.approx(0.5, abs=2e-5)
    assert diagnostics["norm"] == pytest.approx(1.0, abs=2e-5)
    assert diagnostics["state_norm"] == pytest.approx(1.0, abs=2e-5)
    assert diagnostics["cumulative_norm"] == pytest.approx(
        np.sqrt(0.5), abs=2e-5,
    )
    assert fit["fit_overlap_fidelity"] == pytest.approx(0.5, abs=2e-5)
    assert fit["fit_overlap_infidelity"] == pytest.approx(0.5, abs=2e-5)


def test_fit_run_gate_reuse_resets_per_run_traces_and_split_diagnostics():
    """Reusing FIT should report only the latest invocation's sweep work."""
    state = qtn.MPS_rand_state(
        3,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=250,
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 2])

    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=2,
        verbose=True,
        block_size=2,
        max_bond=2,
        collect_split_diagnostics=True,
    )
    assert len(fit.local_norm_trace) == 2
    assert len(fit.info["two_site_splits"]) == 4

    fit.run_gate(
        adaptive_block_sweeps=None,
        n_iter=1,
        verbose=True,
        block_size=2,
        max_bond=2,
        collect_split_diagnostics=True,
    )

    assert len(fit.local_norm_trace) == 1
    assert len(fit.fidelity_trace) == 1
    assert len(fit.info["two_site_splits"]) == 2
    assert "three_site_splits" not in fit.info


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_fit_reduces_exactly_one_tensor_norm_per_sweep(monkeypatch, block_size):
    """Intermediate local updates must not pay for unused norm reductions."""
    state = qtn.MPS_rand_state(
        4,
        bond_dim=2,
        phys_dim=2,
        dtype="complex128",
        seed=251,
    )
    fit = py.FIT(state.copy(), p=state, range_int=[0, 3])
    original_norm = qtn.Tensor.norm
    norm_calls = []

    def count_norm(tensor, *args, **kwargs):
        norm_calls.append(tensor)
        return original_norm(tensor, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "norm", count_norm)
    fit.run_gate(
        n_iter=2,
        block_size=block_size,
        three_site_sweeps=2 if block_size == 3 else 1,
        max_bond=2,
        rtol=None,
    )

    assert len(norm_calls) == 2
    assert len(fit.local_norm_trace) == 2


@pytest.mark.parametrize("method_name", ["run", "run_eff"])
@pytest.mark.parametrize("n_iter", [0, -1, 1.5])
def test_fit_full_chain_runs_validate_iteration_count(method_name, n_iter):
    """All FIT entry points reject empty, negative, and fractional sweeps."""
    state = qtn.MPS_computational_state("00", dtype="complex128")
    fit = py.FIT(state.copy(), p=state)

    with pytest.raises(ValueError, match="n_iter must be a positive integer"):
        getattr(fit, method_name)(n_iter=n_iter)
