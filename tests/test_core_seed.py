"""Tests for seed forwarding in optimizer/fidelity helpers."""

import pepsy.core as core
import quimb.tensor as qtn


def test_build_optimizer_forwards_seed(monkeypatch):
    """build_optimizer should pass seed through to cotengra optimizer."""
    captured = {}

    class DummyOpt:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(core.ctg, "ReusableHyperOptimizer", DummyOpt)

    out = core.build_optimizer(
        progbar=False,
        parallel=False,
        optlib="random",
        directory=None,
        max_repeats=1,
        max_time="rate:1e2",
        seed=123,
    )

    assert isinstance(out, DummyOpt)
    assert captured["seed"] == 123


def test_build_compressed_optimizer_forwards_seed(monkeypatch):
    """build_compressed_optimizer should pass seed to cotengra."""
    captured = {}

    class DummyCOpt:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)

    monkeypatch.setattr(core.ctg, "ReusableHyperCompressedOptimizer", DummyCOpt)

    out = core.build_compressed_optimizer(
        progbar=False,
        chi=4,
        directory=None,
        max_repeats=1,
        max_time="rate:1e2",
        seed=77,
    )

    assert isinstance(out, DummyCOpt)
    assert captured["seed"] == 77


def test_fidel_mps_forwards_seed(monkeypatch):
    """fidel_mps should call build_optimizer with provided seed."""
    captured = {}

    def fake_build_optimizer(**kwargs):
        captured.update(kwargs)
        return "auto-hq"

    monkeypatch.setattr(core, "build_optimizer", fake_build_optimizer)

    psi = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=5)
    psi_fix = psi.copy()

    fidelity = core.fidel_mps(psi, psi_fix, seed=9)

    assert abs(fidelity - 1.0) < 1e-12
    assert captured["seed"] == 9
    assert captured["progbar"] is False


def test_default_backend_setters_roundtrip():
    """Default backend setters/getters should round-trip callables."""
    array_backend = lambda x: x  # noqa: E731
    grad_backend = lambda x: x  # noqa: E731

    core.reset_default_backends()
    try:
        core.set_default_array_backend(array_backend)
        core.set_default_grad_backend(grad_backend)
        assert core.get_default_array_backend() is array_backend
        assert core.get_default_grad_backend() is grad_backend
    finally:
        core.reset_default_backends()


def test_bdymps_uses_default_array_backend_when_not_provided():
    """BdyMPS should pick global default array backend if to_backend is omitted."""
    import pepsy

    array_backend = lambda x: x  # noqa: E731
    core.reset_default_backends()
    try:
        core.set_default_array_backend(array_backend)
        ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=101, dtype="complex128")
        _, norm = pepsy.prepare_boundary_inputs(ket=ket)
        bdy = pepsy.BdyMPS(
            tn_double=norm,
            chi=4,
            single_layer=False,
        )
        assert bdy.to_backend is array_backend
    finally:
        core.reset_default_backends()
