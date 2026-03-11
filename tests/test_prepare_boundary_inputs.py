"""Behavior tests for ``prepare_boundary_inputs``."""

import pepsy
import pytest
import quimb.tensor as qtn


def test_prepare_boundary_inputs_uses_readable_bra_reindex_suffix():
    """Bra internal indices should map to readable ``<ket>_*`` names."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=7, dtype="complex128")

    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    ket_inner = set(ket_tagged.inner_inds())
    bra_inds = set(norm_tagged.select("BRA").ind_map)

    assert ket_inner
    for idx in ket_inner:
        assert idx not in bra_inds
        assert f"{idx}_*" in bra_inds


def test_prepare_boundary_inputs_accepts_user_bra_with_disjoint_indices():
    """User-provided bra should pass when inner indices are disjoint."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=11, dtype="complex128")
    bra = ket.copy().conj()
    bra.reindex_({idx: f"{idx}_br" for idx in bra.inner_inds()})

    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(
        ket=ket,
        bra=bra,
    )
    bra_inds = set(norm_tagged.select("BRA").ind_map)
    ket_outer = set(ket_tagged.outer_inds())
    bra_outer = set(norm_tagged.select("BRA").outer_inds())

    assert set(ket_tagged.inner_inds()).isdisjoint(set(norm_tagged.select("BRA").inner_inds()))
    assert ket_outer & bra_outer
    assert any(ind.endswith("_br") for ind in bra_inds)


def test_prepare_boundary_inputs_rejects_user_bra_with_colliding_indices():
    """User-provided bra should fail fast when inner index names collide."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=13, dtype="complex128")
    bra = ket.copy().conj()

    with pytest.raises(ValueError, match="internal index names disjoint"):
        pepsy.prepare_boundary_inputs(ket=ket, bra=bra)


def test_bdymps_initializes_xy_boundaries():
    """BdyMPS should initialize only X/Y boundary keys."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=17, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    keys = bdy.available_boundary_keys()
    assert keys
    assert all(key[0] in {"X", "Y"} for key in keys)


def test_bdymps_flat_overrides_single_layer_with_warning():
    """flat=True should warn and force single_layer=False."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=18, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)

    with pytest.warns(UserWarning, match="flat=True is incompatible"):
        bdy = pepsy.BdyMPS(
            tn_flat=ket_tagged,
            tn_double=norm_tagged,
            chi=8,
            single_layer=True,
            flat=True,
        )

    assert bdy.flat is True
    assert bdy.mps_b


def test_bdymps_norm_matches_manual_mean():
    """norm should equal manual mean over all boundary MPS norms."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=19, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )

    values = list(bdy.mps_b.values())
    manual_avg = sum(mps.norm() for mps in values) / len(values)
    assert bdy.norm == manual_avg


def test_bdymps_chi_reports_largest_boundary_bond():
    """chi should report largest boundary bond dimension."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=29, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )

    bdy.expand_bnd(12)
    assert bdy.chi == max(int(mps.max_bond()) for mps in bdy.mps_b.values())


def test_bdymps_expand_bnd_updates_all_boundaries():
    """expand_bnd should update chi and enforce max-bond cap."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=37, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )

    out = bdy.expand_bnd(4)
    assert out is bdy
    assert all(mps.max_bond() <= 4 for mps in bdy.mps_b.values())
    assert bdy.chi == max(int(mps.max_bond()) for mps in bdy.mps_b.values())
    assert bdy.chi <= 4

    bdy.expand_bnd(6)
    assert all(mps.max_bond() <= 6 for mps in bdy.mps_b.values())
    assert bdy.chi == max(int(mps.max_bond()) for mps in bdy.mps_b.values())
    assert bdy.chi <= 6


def test_bdymps_expand_bnd_expands_when_target_larger():
    """expand_bnd should actually increase at least one bond when chi grows."""
    ket = qtn.PEPS.rand(Lx=4, Ly=4, bond_dim=2, seed=57, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=10,
        single_layer=False,
    )

    before = {key: int(mps.max_bond()) for key, mps in bdy.mps_b.items()}
    target_chi = max(before.values()) + 6
    bdy.expand_bnd(target_chi)
    after = {key: int(mps.max_bond()) for key, mps in bdy.mps_b.items()}

    assert all(value <= target_chi for value in after.values())
    assert any(value == target_chi for value in after.values())
    assert any(after[key] > before[key] for key in before)
    assert bdy.chi == max(after.values())


def test_bdymps_expand_bnd_inplace_flag():
    """expand_bnd should support in-place and copied modes."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=41, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )

    ids_before = {key: id(mps) for key, mps in bdy.mps_b.items()}
    bdy.expand_bnd(4, inplace=True)
    ids_after_inplace = {key: id(mps) for key, mps in bdy.mps_b.items()}
    assert ids_after_inplace == ids_before

    bdy.expand_bnd(6, inplace=False)
    ids_after_copy = {key: id(mps) for key, mps in bdy.mps_b.items()}
    assert any(ids_after_copy[key] != ids_after_inplace[key] for key in ids_after_copy)


def test_bdymps_normalize_normalizes_all_boundaries_inplace():
    """normalize should update all boundary MPS objects in place."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=73, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )

    ids_before = {key: id(mps) for key, mps in bdy.mps_b.items()}
    for mps in bdy.mps_b.values():
        mps[0].modify(data=2.0 * mps[0].data)
    assert any(abs(complex(mps.norm()) - 1.0) > 1e-6 for mps in bdy.mps_b.values())

    out = bdy.normalize()
    assert out is bdy

    ids_after = {key: id(mps) for key, mps in bdy.mps_b.items()}
    assert ids_after == ids_before
    assert all(abs(complex(mps.norm()) - 1.0) < 1e-9 for mps in bdy.mps_b.values())


def test_compbdy_fidelity_history_resets_each_run(monkeypatch):
    """CompBdy.run should rebuild self.fidel per run when fidel_=True."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=23, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    monkeypatch.setattr(pepsy.boundary_sweeps, "fidel_mps", lambda _tn, _p: 0.5)
    monkeypatch.setattr(
        pepsy.boundary_sweeps.CompBdy,
        "_run_fit_solver",
        lambda self, fit, boundary_mps: None,
    )

    comp.run(direction="y", fidel_=True, pbar=False, n_iter=1, max_separation=0)
    assert comp.fidel == [0.5, 0.5]

    comp.fidel.append(9.0)
    comp.run(direction="y", fidel_=True, pbar=False, n_iter=1, max_separation=0)
    assert comp.fidel == [0.5, 0.5]


def test_compbdy_run_eff_does_not_use_fit_verbose_fidelity(monkeypatch):
    """CompBdy should keep FIT.run_eff verbose=False even when fidel_=True."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=29, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b, dmrg_run="eff")

    run_eff_verbose_args = []

    def fake_run_eff(self, n_iter=6, verbose=False):
        run_eff_verbose_args.append(verbose)

    monkeypatch.setattr(pepsy.boundary_sweeps.FIT, "run_eff", fake_run_eff)
    monkeypatch.setattr(pepsy.boundary_sweeps, "fidel_mps", lambda _tn, _p: 0.5)

    comp.run(direction="y", fidel_=True, pbar=False, n_iter=1, max_separation=0)

    assert run_eff_verbose_args
    assert all(arg is False for arg in run_eff_verbose_args)
    assert comp.fidel == [0.5, 0.5]


class _DummyNorm:
    """Minimal norm-like object with copy() for ContractBoundary tests."""

    def copy(self):
        return self


def test_contract_boundary_default_returns_structured_result(monkeypatch):
    """ContractBoundary should always return BoundaryContractResult."""

    class _FakeCompBdy:
        def __init__(self, *_args, **_kwargs):
            self.fidel = [0.3, 0.4]

        def run(self, **_kwargs):
            return 12.5

    monkeypatch.setattr(pepsy.boundary_norm, "CompBdy", _FakeCompBdy)
    out = pepsy.ContractBoundary(norm=_DummyNorm(), mps_boundaries={})
    assert isinstance(out, pepsy.BoundaryContractResult)
    assert out.cost == 12.5
    assert out.fidel == [0.3, 0.4]


def test_contract_boundary_includes_requested_metadata(monkeypatch):
    """ContractBoundary structured return should include direction and run metadata."""

    class _FakeCompBdy:
        def __init__(self, *_args, **_kwargs):
            self.fidel = [0.3, 0.4]

        def run(self, **_kwargs):
            return 12.5

    monkeypatch.setattr(pepsy.boundary_norm, "CompBdy", _FakeCompBdy)
    out = pepsy.ContractBoundary(
        norm=_DummyNorm(),
        mps_boundaries={},
        direction="x",
        n_iter=3,
        max_separation=1,
    )

    assert isinstance(out, pepsy.BoundaryContractResult)
    assert out.cost == 12.5
    assert out.fidel == [0.3, 0.4]
    assert out.direction == "x"
    assert out.n_iter == 3
    assert out.max_separation == 1


def test_contract_boundary_fidelity_flag_keeps_structured_result(monkeypatch):
    """fidel_=True should still return structured result with fidelity history."""

    class _FakeCompBdy:
        def __init__(self, *_args, **_kwargs):
            self.fidel = [0.9, 0.8]

        def run(self, **_kwargs):
            return 7.5

    monkeypatch.setattr(pepsy.boundary_norm, "CompBdy", _FakeCompBdy)
    out = pepsy.ContractBoundary(
        norm=_DummyNorm(),
        mps_boundaries={},
        fidel_=True,
    )

    assert isinstance(out, pepsy.BoundaryContractResult)
    assert out.cost == 7.5
    assert out.fidel == [0.9, 0.8]


def test_contract_boundary_empty_fidelity_list_returns_structured_result(monkeypatch):
    """Structured return should work even when fidelity history is empty."""

    class _FakeCompBdy:
        def __init__(self, *_args, **_kwargs):
            self.fidel = []

        def run(self, **_kwargs):
            return 3.25

    monkeypatch.setattr(pepsy.boundary_norm, "CompBdy", _FakeCompBdy)
    out = pepsy.ContractBoundary(
        norm=_DummyNorm(),
        mps_boundaries={},
        fidel_=True,
    )

    assert isinstance(out, pepsy.BoundaryContractResult)
    assert out.cost == 3.25
    assert out.fidel == []


def test_compbdy_move_step_resets_and_updates_fidelity(monkeypatch):
    """move_step_bdy should rebuild fidelity history from scratch each call."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=31, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.prepare_boundary_inputs(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    def fake_fit_one_step(self, side, step_, cut_tag_id, site_tag_id):
        _ = (step_, cut_tag_id, site_tag_id)
        if self.fidel_:
            self.fidel.append(0.77 if side == "left" else 0.66)
        return None

    monkeypatch.setattr(pepsy.boundary_sweeps.CompBdy, "_fit_one_step", fake_fit_one_step)

    comp.move_step_bdy(pos=0, direction="y_left", fidel_=True)
    assert comp.fidel == [0.77]
    assert comp.fidelity == [0.77]


class _DummyNormWithTags:
    """Minimal norm-like object exposing X/Y tags for CompBdy init."""

    tags = {"X0", "X1", "X2", "Y0", "Y1", "Y2"}


class _DummyFinalTN:
    """Minimal contractable object for patched CompBdy.run tests."""

    @staticmethod
    def contract(*_args, **_kwargs):
        return 1.0, 0


class _FakeTqdm:
    """Capture tqdm totals/updates without terminal output."""

    instances = []

    def __init__(self, *args, **kwargs):
        _ = args
        self.total = kwargs.get("total")
        self.n = 0
        self.postfix_calls = []
        _FakeTqdm.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return False

    def set_postfix(self, *args, **kwargs):
        if args:
            self.postfix_calls.append(dict(args[0]))
        elif kwargs:
            self.postfix_calls.append(dict(kwargs))
        return None

    def refresh(self):
        return None

    def update(self, n=1):
        self.n += n


def _fake_fit_one_side_with_flat_skip(self, _side, steps, progress_bar, _cut_tag_id, _site_tag_id):
    """Mimic update-count behavior of _fit_one_side for progress accounting tests."""
    for step_idx in range(steps):
        if step_idx == 0 and self.flat:
            continue
        if progress_bar is not None:
            progress_bar.update(1)
    return None


def test_compbdy_run_progress_total_accounts_for_flat_skip(monkeypatch):
    """run() tqdm total should match actual fit updates when flat=True."""
    comp = pepsy.CompBdy(_DummyNormWithTags(), {})
    _FakeTqdm.instances = []

    monkeypatch.setattr(pepsy.boundary_sweeps, "tqdm", _FakeTqdm)
    monkeypatch.setattr(
        pepsy.boundary_sweeps.CompBdy,
        "_fit_one_side",
        _fake_fit_one_side_with_flat_skip,
    )
    monkeypatch.setattr(
        pepsy.boundary_sweeps.CompBdy,
        "_build_final_boundary_network",
        lambda self, spec, p_previous_l, p_previous_r: _DummyFinalTN(),
    )

    _ = comp.run(direction="y", flat=True, pbar=True, max_separation=0)

    # Ly=3 -> y_left=1, y_right=2 at max_separation=0 => effective updates: 0 + 1 = 1.
    assert _FakeTqdm.instances
    pbar = _FakeTqdm.instances[-1]
    assert pbar.total == 1
    assert pbar.n == 1


def test_compbdy_move_bdy_progress_total_accounts_for_flat_skip(monkeypatch):
    """move_bdy() tqdm total should match actual fit updates when flat=True."""
    comp = pepsy.CompBdy(_DummyNormWithTags(), {})
    _FakeTqdm.instances = []

    monkeypatch.setattr(pepsy.boundary_sweeps, "tqdm", _FakeTqdm)
    monkeypatch.setattr(
        pepsy.boundary_sweeps.CompBdy,
        "_fit_one_side",
        _fake_fit_one_side_with_flat_skip,
    )

    comp.move_bdy(direction="y_left", flat=True, pbar=True)

    # Ly=3 -> n_steps=Ly-1=2, flat skip removes first fit update => total 1.
    assert _FakeTqdm.instances
    pbar = _FakeTqdm.instances[-1]
    assert pbar.total == 1
    assert pbar.n == 1


def test_compbdy_move_step_pbar_shows_current_fidelity(monkeypatch):
    """move_step_bdy should expose current step fidelity and chi in tqdm postfix."""
    comp = pepsy.CompBdy(_DummyNormWithTags(), {})
    _FakeTqdm.instances = []

    class _DummyMPS:
        @staticmethod
        def max_bond():
            return 13

    def fake_fit_one_step(self, side, step_, cut_tag_id, site_tag_id):
        _ = (side, step_, cut_tag_id, site_tag_id)
        if self.fidel_:
            self.fidel.append(0.77)
        return _DummyMPS()

    monkeypatch.setattr(pepsy.boundary_sweeps, "tqdm", _FakeTqdm)
    monkeypatch.setattr(pepsy.boundary_sweeps.CompBdy, "_fit_one_step", fake_fit_one_step)

    comp.move_step_bdy(pos=1, direction="y_left", fidel_=True, pbar=True)

    assert _FakeTqdm.instances
    pbar = _FakeTqdm.instances[-1]
    assert pbar.total == 1
    assert pbar.n == 1
    assert pbar.postfix_calls
    last = pbar.postfix_calls[-1]
    assert last["pos"] == 1
    assert last["chi"] == 13
    assert abs(last["F"] - 0.77) < 1e-12


def test_normalize_returns_dict_with_boundary_and_contract_result(monkeypatch):
    """normalize should return state/cost and include built boundary object."""
    captured = {}

    class _FakeBdy:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    def fake_bdymps(**kwargs):
        captured["bdy_kwargs"] = kwargs
        return _FakeBdy()

    def fake_contract_boundary(**kwargs):
        captured["contract_kwargs"] = kwargs
        return pepsy.BoundaryContractResult(
            cost=4.0,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary_states, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary_norm, "ContractBoundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=131, dtype="complex128")
    out = pepsy.normalize(
        ket,
        chi=8,
        n_iter=1,
        pbar=False,
    )

    assert set(out) == {"state", "cost", "cost_scalar", "bdy", "contract_result"}
    assert out["cost"] == 4.0
    assert out["cost_scalar"] == 4.0 + 0.0j
    assert out["bdy"].mps_b is captured["contract_kwargs"]["mps_boundaries"]
    assert isinstance(out["contract_result"], pepsy.BoundaryContractResult)
    assert "tn_double" in captured["bdy_kwargs"]


def test_normalize_uses_provided_bdy_without_constructing_new_one(monkeypatch):
    """normalize should reuse provided bdy and skip BdyMPS construction."""
    class _ProvidedBdy:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    provided = _ProvidedBdy()
    captured = {}

    def fail_bdymps(**kwargs):
        _ = kwargs
        raise AssertionError("BdyMPS constructor should not be called when bdy is provided.")

    def fake_contract_boundary(**kwargs):
        captured["contract_kwargs"] = kwargs
        return pepsy.BoundaryContractResult(
            cost=9.0,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary_states, "BdyMPS", fail_bdymps)
    monkeypatch.setattr(pepsy.boundary_norm, "ContractBoundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=137, dtype="complex128")
    out = pepsy.normalize(
        ket,
        bdy=provided,
        n_iter=1,
        pbar=False,
    )

    assert out["bdy"] is provided
    assert captured["contract_kwargs"]["mps_boundaries"] is provided.mps_b
    assert out["cost"] == 9.0


def test_normalize_requires_chi_when_bdy_not_provided():
    """normalize should require chi if caller does not pass bdy."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=139, dtype="complex128")
    with pytest.raises(ValueError, match="Provide chi when bdy is not supplied."):
        pepsy.normalize(ket)
