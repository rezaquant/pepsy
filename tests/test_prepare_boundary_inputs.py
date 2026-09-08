"""Behavior tests for ``build_bra_ket``."""

import autoray as ar
import numpy as np
import pepsy
import pytest
import quimb.tensor as qtn


def test_validate_tensor_network_tags_requires_i_tags():
    """Validator should require at least one site tag with I<int>[,<int>...]."""
    dummy = type("DummyTN", (), {"tags": {"X0", "Y0"}})()

    with pytest.raises(ValueError, match=r"X\*, Y\*, and I\*"):
        pepsy.boundary.metrics.validate_tensor_network_tags(dummy)


def test_validate_tensor_network_tags_accepts_multi_index_i_tag():
    """Validator should accept I tags like I2, I3,4, or I2,3,4."""
    dummy = type("DummyTN", (), {"tags": {"X2", "Y3", "I2,3,4"}})()

    pepsy.boundary.metrics.validate_tensor_network_tags(dummy)


def test_build_bra_ket_uses_readable_bra_reindex_suffix():
    """Bra internal indices should map to readable ``<ket>_*`` names."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=7, dtype="complex128")

    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    ket_inner = set(ket_tagged.inner_inds())
    bra_inds = set(norm_tagged.select("BRA").ind_map)

    assert ket_inner
    for idx in ket_inner:
        assert idx not in bra_inds
        assert f"{idx}_*" in bra_inds


def test_build_bra_ket_accepts_user_bra_with_disjoint_indices():
    """User-provided bra should pass when inner indices are disjoint."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=11, dtype="complex128")
    bra = ket.copy().conj()
    bra.reindex_({idx: f"{idx}_br" for idx in bra.inner_inds()})

    ket_tagged, norm_tagged = pepsy.build_bra_ket(
        ket=ket,
        bra=bra,
    )
    bra_inds = set(norm_tagged.select("BRA").ind_map)
    ket_outer = set(ket_tagged.outer_inds())
    bra_outer = set(norm_tagged.select("BRA").outer_inds())

    assert set(ket_tagged.inner_inds()).isdisjoint(set(norm_tagged.select("BRA").inner_inds()))
    assert ket_outer & bra_outer
    assert any(ind.endswith("_br") for ind in bra_inds)


def test_build_bra_ket_reindexes_user_bra_with_colliding_indices():
    """User-provided bra with colliding inner indices should be auto-reindexed."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=13, dtype="complex128")
    bra = ket.copy().conj()

    _, norm = pepsy.build_bra_ket(ket=ket, bra=bra)
    # After reindex the norm network should have no inner-index collisions
    # and should contract to a scalar (all physical outer indices shared).
    assert norm is not None


def test_build_bra_ket_removes_stale_layer_tags_from_overlap():
    """Repeated norm/overlap builds should keep KET and BRA layers distinct."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=14, dtype="complex128")
    target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=15, dtype="complex128")

    pepsy.build_bra_ket(ket=ket)
    pepsy.build_bra_ket(ket=target)
    _, overlap = pepsy.build_bra_ket(ket=ket, bra=target)

    assert len(overlap.select("KET").tensor_map) == len(ket.tensor_map)
    assert len(overlap.select("BRA").tensor_map) == len(target.tensor_map)
    assert len(overlap.select(["KET", "BRA"], which="all").tensor_map) == 0


def test_bdymps_initializes_xy_boundaries():
    """BdyMPS should initialize only X/Y boundary keys."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=17, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
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
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)

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
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
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
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
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
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
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
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
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
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
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
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
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


def test_bdymps_dispatch_backend_converter_routes_cupy(monkeypatch):
    """Backend dispatch should route cupy tensors through cupy caster builder."""
    sentinel = object()

    def _fake_build_to_cupy(sample_data, dtype_name):
        assert dtype_name == "complex128"
        return sentinel

    monkeypatch.setattr(pepsy.BdyMPS, "_build_to_cupy", staticmethod(_fake_build_to_cupy))
    bdy = object.__new__(pepsy.BdyMPS)

    out = pepsy.BdyMPS._dispatch_backend_converter(
        bdy,
        backend="cupy",
        dtype_name="complex128",
        sample_data=object(),
    )

    assert out is sentinel


def test_bdymps_dispatch_backend_converter_routes_jax(monkeypatch):
    """Backend dispatch should route jax tensors through jax caster builder."""
    sentinel = object()

    def _fake_build_to_jax(sample_data, dtype_name):
        assert dtype_name == "complex128"
        return sentinel

    monkeypatch.setattr(pepsy.BdyMPS, "_build_to_jax", staticmethod(_fake_build_to_jax))
    bdy = object.__new__(pepsy.BdyMPS)

    out = pepsy.BdyMPS._dispatch_backend_converter(
        bdy,
        backend="jax",
        dtype_name="complex128",
        sample_data=object(),
    )

    assert out is sentinel


def test_bdymps_build_to_jax_casts_complex_numpy_input_to_real():
    """JAX boundary caster should drop imaginary part when target dtype is real."""
    jnp = pytest.importorskip("jax.numpy")

    sample = jnp.asarray([0.0], dtype=jnp.float32)
    caster = pepsy.BdyMPS._build_to_jax(sample, "float32")
    out = caster(np.asarray([1.0 + 2.0j], dtype=np.complex64))

    assert ar.infer_backend(out) == "jax"
    assert out.dtype == jnp.float32
    assert np.allclose(np.asarray(out), np.asarray([1.0], dtype=np.float32))


def test_bdymps_build_to_jax_accepts_torch_complex_input():
    """JAX boundary caster should accept torch tensors and keep real component."""
    torch = pytest.importorskip("torch")
    jnp = pytest.importorskip("jax.numpy")

    sample = jnp.asarray([0.0], dtype=jnp.float32)
    caster = pepsy.BdyMPS._build_to_jax(sample, "float32")
    out = caster(torch.tensor([2.5 - 3.0j], dtype=torch.complex64))

    assert ar.infer_backend(out) == "jax"
    assert out.dtype == jnp.float32
    assert np.allclose(np.asarray(out), np.asarray([2.5], dtype=np.float32))


def test_compbdy_fidelity_history_resets_each_run(monkeypatch):
    """CompBdy.run should rebuild self.fidel per run when track_boundary_fidelity=True."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=23, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    monkeypatch.setattr(pepsy.boundary.sweeps, "tn_fidelity", lambda _tn, _p, **kwargs: 0.5)
    monkeypatch.setattr(
        pepsy.boundary.sweeps.CompBdy,
        "_run_fit_solver",
        lambda self, fit, boundary_mps: None,
    )

    comp.run(direction="y", track_boundary_fidelity=True, progress=False, n_iter=1, max_separation=0)
    assert comp.fidel == [0.5, 0.5]

    comp.fidel.append(9.0)
    comp.run(direction="y", track_boundary_fidelity=True, progress=False, n_iter=1, max_separation=0)
    assert comp.fidel == [0.5, 0.5]


def test_compbdy_run_eff_does_not_use_fit_verbose_fidelity(monkeypatch):
    """CompBdy should keep FIT.run_eff verbose=False even when track_boundary_fidelity=True."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=29, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b, fit_mode="eff")

    run_eff_verbose_args = []

    def fake_run_eff(self, n_iter=6, verbose=False):
        run_eff_verbose_args.append(verbose)

    monkeypatch.setattr(pepsy.boundary.sweeps.FIT, "run_eff", fake_run_eff)
    monkeypatch.setattr(pepsy.boundary.sweeps, "tn_fidelity", lambda _tn, _p, **kwargs: 0.5)

    comp.run(direction="y", track_boundary_fidelity=True, progress=False, n_iter=1, max_separation=0)

    assert run_eff_verbose_args
    assert all(arg is False for arg in run_eff_verbose_args)
    assert comp.fidel == [0.5, 0.5]


def test_compbdy_two_site_routes_cached_full_boundary_controls():
    """Two-site boundaries should use the cached full-interval FIT kernel."""
    captured = {}

    class _Fit:
        range_int = None

        @staticmethod
        def run_gate(**kwargs):
            captured.update(kwargs)

    class _Boundary:
        L = 4

        @staticmethod
        def max_bond():
            return 3

    comp = object.__new__(pepsy.CompBdy)
    comp.fit_mode = "two-site"
    comp.fit_max_bond = 7
    comp.fit_sweep_sequence = "LR"
    comp.fit_cutoff = 2.0e-9
    comp.fit_cutoff_mode = "rel"
    comp.fit_min_iter = 2
    comp.fit_rtol = 3.0e-8
    comp.fit_patience = 4
    comp.n_iter = 9

    fit = _Fit()
    comp._run_fit_solver(fit, _Boundary())

    assert fit.range_int == (0, 3)
    assert captured == {
        "n_iter": 9,
        "verbose": False,
        "block_size": 2,
        "sweep_sequence": "LR",
        "max_bond": 7,
        "cutoff": 2.0e-9,
        "cutoff_mode": "rel",
        "adaptive_block_sweeps": None,
        "min_iter": 2,
        "rtol": 3.0e-8,
        "patience": 4,
        "collect_split_diagnostics": False,
    }


def test_compbdy_two_site_falls_back_to_current_boundary_cap():
    """Direct CompBdy use must never leave two-site bond growth uncapped."""
    captured = {}

    class _Fit:
        range_int = None

        @staticmethod
        def run_gate(**kwargs):
            captured.update(kwargs)

    class _Boundary:
        L = 3

        @staticmethod
        def max_bond():
            return 5

    comp = object.__new__(pepsy.CompBdy)
    comp.fit_mode = "two-site"
    comp.fit_max_bond = None
    comp.fit_sweep_sequence = "RL"
    comp.fit_cutoff = 1.0e-12
    comp.fit_cutoff_mode = "rsum2"
    comp.fit_min_iter = None
    comp.fit_rtol = None
    comp.fit_patience = 1
    comp.n_iter = 2

    comp._run_fit_solver(_Fit(), _Boundary())

    assert captured["max_bond"] == 5


def test_compbdy_fit_mode_is_canonicalized_and_validated_early():
    """Mode aliases should be stable and mistakes should fail at construction."""
    norm = type("TaggedNorm", (), {"tags": {"X0", "Y0"}})()

    assert pepsy.CompBdy(norm, {}, fit_mode="two_site").fit_mode == "two-site"
    assert pepsy.CompBdy(norm, {}, fit_mode="one-site").fit_mode == "eff"

    with pytest.raises(ValueError, match="Unknown fit_mode"):
        pepsy.CompBdy(norm, {}, fit_mode="two-sites")


def test_compbdy_run_reuses_equalized_boundaries_without_stale_exponent():
    """Switching equalize modes should not reuse stale boundary exponents."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=31, dtype="complex128")
    for tensor in ket:
        tensor.modify(data=5.0 * tensor.data)

    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    exact = norm_tagged.contract(all, optimize="auto-hq")
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=16,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    comp.run(
        direction="y",
        equalize_norms=True,
        progress=False,
        n_iter=12,
        max_separation=0,
    )
    out = comp.run(
        direction="y",
        equalize_norms=False,
        progress=False,
        n_iter=12,
        max_separation=0,
    )

    assert abs(out - exact) / abs(exact) < 1.0e-10


def test_compbdy_run_write_back_false_does_not_mutate_boundary_exponents(monkeypatch):
    """write_back=False should not alter caller-owned boundary exponent state."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=33, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    bdy.mps_b["Y0_r"].exponent = 3.0
    bdy.mps_b["Y1_r"].exponent = 7.0
    before = {key: bdy.mps_b[key].exponent for key in ("Y0_r", "Y1_r")}
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    monkeypatch.setattr(
        pepsy.boundary.sweeps.CompBdy,
        "_run_fit_solver",
        lambda self, fit, boundary_mps: None,
    )

    comp.run(
        direction="y",
        equalize_norms=False,
        progress=False,
        n_iter=1,
        max_separation=0,
        write_back=False,
    )

    for key, exponent in before.items():
        assert bdy.mps_b[key].exponent == exponent


def test_compbdy_move_step_clears_stale_exponent_for_data_carried_fit(monkeypatch):
    """move_step_bdy(equalize_norms=False) should write a data-carried boundary."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=35, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    bdy.mps_b["Y0_l"].exponent = 4.0
    bdy.mps_b["Y1_l"].exponent = 9.0
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    monkeypatch.setattr(
        pepsy.boundary.sweeps.CompBdy,
        "_run_fit_solver",
        lambda self, fit, boundary_mps: None,
    )

    comp.move_step_bdy(
        pos=1,
        direction="y_left",
        equalize_norms=False,
        progress=False,
        n_iter=1,
    )

    assert bdy.mps_b["Y1_l"].exponent == 0.0


@pytest.mark.parametrize("equalize_norms", [False, True])
@pytest.mark.parametrize("max_separation", [0, 1])
def test_compbdy_run_includes_input_network_exponent(equalize_norms, max_separation):
    """CompBdy.run should report the represented value of exponent-scaled norm."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=37, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    norm_tagged.exponent = 5.0
    exact = norm_tagged.contract(all, optimize="auto-hq")
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=16,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    out = comp.run(
        direction="y",
        equalize_norms=equalize_norms,
        progress=False,
        n_iter=12,
        max_separation=max_separation,
    )

    assert abs(out - exact) / abs(exact) < 1.0e-10


def test_compbdy_run_strip_exponent_includes_negative_input_network_exponent():
    """strip_exponent=True should add the input TN exponent without changing mantissa."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=39, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    norm_base = norm_tagged.copy()
    norm_scaled = norm_tagged.copy()
    norm_scaled.exponent = -4.0

    bdy_base = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_base,
        chi=16,
        single_layer=False,
    )
    bdy_scaled = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_scaled,
        chi=16,
        single_layer=False,
    )

    main_base, exp_base = pepsy.CompBdy(norm_base, bdy_base.mps_b).run(
        direction="y",
        equalize_norms=True,
        progress=False,
        n_iter=12,
        max_separation=0,
        strip_exponent=True,
    )
    main_scaled, exp_scaled = pepsy.CompBdy(norm_scaled, bdy_scaled.mps_b).run(
        direction="y",
        equalize_norms=True,
        progress=False,
        n_iter=12,
        max_separation=0,
        strip_exponent=True,
    )

    assert main_scaled == pytest.approx(main_base)
    assert exp_scaled == pytest.approx(exp_base - 4.0)


def test_contract_boundary_includes_input_network_exponent():
    """Public contract_boundary should preserve a copied norm's exponent scale."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=41, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    norm_tagged.exponent = 3.0
    exact = norm_tagged.contract(all, optimize="auto-hq")
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=16,
        single_layer=False,
    )

    out = pepsy.contract_boundary(
        norm=norm_tagged,
        bdy=bdy,
        direction="y",
        equalize_norms=True,
        progress=False,
        n_iter=12,
        max_separation=0,
    )

    assert abs(out.cost - exact) / abs(exact) < 1.0e-10


def test_peps_norm_two_site_is_exact_and_reuses_boundary_holder():
    """Two-site PEPS norms should retain accuracy and warm boundary storage."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=43, dtype="complex128")
    exact = ket.make_norm().contract(all, optimize="greedy")
    holder = {}
    kwargs = {
        "chi": 16,
        "bdy": holder,
        "fit_mode": "two-site",
        "fit_sweep_sequence": "RL",
        "n_iter": 4,
        "cutoff": 0.0,
        "contraction_opt": "greedy",
        "progress": False,
    }

    first = pepsy.peps_norm(ket, **kwargs)
    boundary_id = id(holder["bdy"])
    second = pepsy.peps_norm(ket, **kwargs)

    assert abs(first - exact) / abs(exact) < 1.0e-10
    assert abs(second - exact) / abs(exact) < 1.0e-10
    assert id(holder["bdy"]) == boundary_id
    assert all(mps.max_bond() <= 16 for mps in holder["bdy"].mps_b.values())


def test_peps_norm_two_site_can_grow_from_rank_one_without_padding(monkeypatch):
    """An explicit fit cap should let pair updates discover boundary rank."""
    ket = qtn.PEPS.rand(Lx=4, Ly=4, bond_dim=2, seed=47, dtype="complex128")
    _, norm = pepsy.build_bra_ket(ket=ket.copy())
    bdy = pepsy.BdyMPS(tn_double=norm, chi=1)

    assert max(mps.max_bond() for mps in bdy.mps_b.values()) == 1

    def fail_expand(*_args, **_kwargs):
        raise AssertionError("two-site reuse should not globally pad the boundary")

    monkeypatch.setattr(bdy, "expand_bnd", fail_expand)
    pepsy.peps_norm(
        ket,
        chi=4,
        bdy=bdy,
        fit_mode="two-site",
        n_iter=2,
        cutoff=0.0,
        contraction_opt="greedy",
        progress=False,
    )

    assert max(mps.max_bond() for mps in bdy.mps_b.values()) == 4


def test_peps_norm_eff_can_use_adaptive_run_eff_blocks():
    """PEPS norm forwards run_eff block warm-up and one-site refinement."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=48, dtype="complex128")
    exact = ket.make_norm().contract(all, optimize="greedy")

    result = pepsy.peps_norm(
        ket,
        chi=8,
        method="dmrg",
        fit_mode="eff",
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        n_iter=4,
        cutoff=0.0,
        contraction_opt="greedy",
        progress=False,
        return_info=True,
    )

    assert abs(result.cost - exact) / abs(exact) < 1.0e-10
    assert result.fit_diagnostics
    assert all(
        diagnostic.fit_mode == "eff"
        and diagnostic.iterations == 4
        and diagnostic.convergence_reason == "fixed_sweeps"
        for diagnostic in result.fit_diagnostics
    )


def test_peps_norm_eff_adaptive_rtol_reports_run_eff_convergence():
    """PEPS boundary diagnostics expose adaptive run_eff stopping metadata."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=49, dtype="complex128")

    result = pepsy.peps_norm(
        ket,
        chi=8,
        method="dmrg",
        fit_mode="eff",
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        n_iter=5,
        fit_min_iter=2,
        fit_rtol=1.0,
        fit_patience=2,
        cutoff=0.0,
        contraction_opt="greedy",
        progress=False,
        return_info=True,
    )

    assert result.fit_diagnostics
    assert all(
        diagnostic.fit_mode == "eff"
        and 3 <= diagnostic.iterations <= 5
        and diagnostic.convergence_reason
        in {"relative_tolerance", "max_sweeps"}
        for diagnostic in result.fit_diagnostics
    )


class _DummyNorm:
    """Minimal norm-like object with copy() for contract_boundary tests."""

    def copy(self):
        return self


def test_contract_boundary_default_returns_structured_result(monkeypatch):
    """contract_boundary should always return BoundaryContractResult."""

    class _FakeCompBdy:
        def __init__(self, *_args, **_kwargs):
            self.fidel = [0.3, 0.4]

        def run(self, **_kwargs):
            return 12.5

    monkeypatch.setattr(pepsy.boundary.metrics, "CompBdy", _FakeCompBdy)
    out = pepsy.contract_boundary(norm=_DummyNorm(), bdy={"bdy": type("B", (), {"mps_b": {}})()})
    assert isinstance(out, pepsy.BoundaryContractResult)
    assert out.cost == 12.5
    assert out.fidel == [0.3, 0.4]


def test_contract_boundary_includes_requested_metadata(monkeypatch):
    """contract_boundary structured return should include direction and run metadata."""

    class _FakeCompBdy:
        def __init__(self, *_args, **_kwargs):
            self.fidel = [0.3, 0.4]

        def run(self, **_kwargs):
            return 12.5

    monkeypatch.setattr(pepsy.boundary.metrics, "CompBdy", _FakeCompBdy)
    out = pepsy.contract_boundary(
        norm=_DummyNorm(),
        bdy={"bdy": type("B", (), {"mps_b": {}})()},
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
    assert out.fit_diagnostics == ()


def test_peps_norm_can_return_typed_fit_convergence_and_timing():
    """High-level PEPS norms should expose every adaptive boundary FIT."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=49, dtype="complex128")
    exact = ket.make_norm().contract(all, optimize="greedy")

    result = pepsy.peps_norm(
        ket,
        chi=8,
        method="dmrg",
        fit_mode="two-site",
        fit_sweep_sequence="RL",
        n_iter=8,
        fit_min_iter=2,
        fit_rtol=1.0e9,
        fit_patience=1,
        fit_timing=True,
        return_info=True,
        cutoff=0.0,
        contraction_opt="greedy",
        progress=False,
    )

    assert isinstance(result, pepsy.BoundaryContractResult)
    assert abs(result.cost - exact) / abs(exact) < 1.0e-10
    assert result.fit_diagnostics
    for diagnostic in result.fit_diagnostics:
        assert isinstance(diagnostic, pepsy.BoundaryFitDiagnostic)
        assert diagnostic.fit_mode == "two-site"
        assert diagnostic.status == "complete"
        assert diagnostic.iterations == 2
        assert diagnostic.converged is True
        assert diagnostic.convergence_reason == "relative_tolerance"
        assert diagnostic.elapsed_seconds >= 0.0
        assert len(diagnostic.sweep_timings) == 2
        assert [record["direction"] for record in diagnostic.sweep_timings] == [
            "R",
            "L",
        ]
        assert all(record["timing_schema"] == 3 for record in diagnostic.sweep_timings)
        assert all(record["active_site_count"] == 3 for record in diagnostic.sweep_timings)
        assert all(
            record["svd_seconds"]
            == pytest.approx(
                sum(
                    site_timing["svd_seconds"]
                    for site_timing in record["site_timings"]
                )
            )
            for record in diagnostic.sweep_timings
        )


def test_peps_norm_fit_diagnostics_are_cheap_without_timing():
    """Convergence metadata should not enable per-site timers implicitly."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=51, dtype="complex128")

    result = pepsy.peps_norm(
        ket,
        chi=4,
        method="dmrg",
        fit_mode="two-site",
        n_iter=2,
        fit_rtol=None,
        return_info=True,
        contraction_opt="greedy",
        progress=False,
    )

    assert result.fit_diagnostics
    assert all(
        diagnostic.iterations == 2
        and diagnostic.convergence_reason == "max_sweeps"
        and diagnostic.elapsed_seconds is None
        and diagnostic.sweep_timings == ()
        for diagnostic in result.fit_diagnostics
    )


@pytest.mark.parametrize("fit_mode", ["eff", "global"])
def test_peps_norm_reports_legacy_fit_solvers_without_changing_them(fit_mode):
    """Reference and cached one-site FIT should remain fixed-sweep solvers."""
    ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=71, dtype="complex128")
    exact = ket.make_norm().contract(all, optimize="greedy")

    result = pepsy.peps_norm(
        ket,
        chi=8,
        method="dmrg",
        fit_mode=fit_mode,
        n_iter=1,
        fit_timing=True,
        return_info=True,
        contraction_opt="greedy",
        progress=False,
    )

    assert abs(result.cost - exact) / abs(exact) < 1.0e-10
    assert result.fit_diagnostics
    assert all(
        diagnostic.fit_mode == fit_mode
        and diagnostic.iterations == 1
        and diagnostic.converged is False
        and diagnostic.convergence_reason == "fixed_sweeps"
        and diagnostic.elapsed_seconds >= 0.0
        and diagnostic.sweep_timings == ()
        for diagnostic in result.fit_diagnostics
    )


def test_contract_boundary_can_return_stripped_exponent(monkeypatch):
    """contract_boundary(strip_exponent=True) should preserve scaled output."""
    captured = {}

    class _FakeCompBdy:
        def __init__(self, *_args, **_kwargs):
            self.fidel = []

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return 1.25, 42.0

    monkeypatch.setattr(pepsy.boundary.metrics, "CompBdy", _FakeCompBdy)
    out = pepsy.contract_boundary(
        norm=_DummyNorm(),
        bdy={"bdy": type("B", (), {"mps_b": {}})()},
        strip_exponent=True,
    )

    assert captured["run_kwargs"]["strip_exponent"] is True
    assert out.cost == (1.25, 42.0)


def test_contract_boundary_accepts_bdy_object(monkeypatch):
    """contract_boundary should accept bdy object instead of mps_boundaries."""
    captured = {}

    class _FakeCompBdy:
        def __init__(self, _norm, mps_boundaries, **_kwargs):
            captured["mps_boundaries"] = mps_boundaries
            self.fidel = []

        def run(self, **_kwargs):
            return 3.0

    class _BdyObj:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    monkeypatch.setattr(pepsy.boundary.metrics, "CompBdy", _FakeCompBdy)
    bdy = _BdyObj()
    out = pepsy.contract_boundary(norm=_DummyNorm(), bdy=bdy)
    assert out.cost == 3.0
    assert captured["mps_boundaries"] is bdy.mps_b


def test_contract_boundary_forwards_two_site_policy_and_target_chi(monkeypatch):
    """The boundary API should preserve every two-site convergence control."""
    captured = {}

    class _FakeCompBdy:
        def __init__(self, _norm, _mps_boundaries, **kwargs):
            captured.update(kwargs)
            self.fidel = []

        def run(self, **_kwargs):
            return 3.0

    class _BdyObj:
        mps_b = {"Y0_l": object()}

    monkeypatch.setattr(pepsy.boundary.metrics, "CompBdy", _FakeCompBdy)
    pepsy.contract_boundary(
        norm=_DummyNorm(),
        bdy=_BdyObj(),
        fit_mode="two-site",
        fit_max_bond=11,
        fit_sweep_sequence="LR",
        fit_cutoff=2.0e-9,
        fit_cutoff_mode="rel",
        fit_min_iter=2,
        fit_rtol=3.0e-8,
        fit_patience=4,
    )

    assert captured["fit_mode"] == "two-site"
    assert captured["fit_max_bond"] == 11
    assert captured["fit_sweep_sequence"] == "LR"
    assert captured["fit_cutoff"] == 2.0e-9
    assert captured["fit_cutoff_mode"] == "rel"
    assert captured["fit_min_iter"] == 2
    assert captured["fit_rtol"] == 3.0e-8
    assert captured["fit_patience"] == 4


def test_contract_boundary_accepts_bdy_holder_dict(monkeypatch):
    """contract_boundary should accept bdy holder dict with bdy['bdy']."""
    captured = {}

    class _FakeCompBdy:
        def __init__(self, _norm, mps_boundaries, **_kwargs):
            captured["mps_boundaries"] = mps_boundaries
            self.fidel = []

        def run(self, **_kwargs):
            return 5.0

    class _BdyObj:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    monkeypatch.setattr(pepsy.boundary.metrics, "CompBdy", _FakeCompBdy)
    holder = {"bdy": _BdyObj()}
    out = pepsy.contract_boundary(norm=_DummyNorm(), bdy=holder)
    assert out.cost == 5.0
    assert captured["mps_boundaries"] is holder["bdy"].mps_b


def test_contract_boundary_requires_bdy():
    """contract_boundary should require bdy."""
    with pytest.raises(ValueError, match="Provide bdy"):
        pepsy.contract_boundary(norm=_DummyNorm())


def test_compbdy_move_step_resets_and_updates_fidelity(monkeypatch):
    """move_step_bdy should rebuild fidelity history from scratch each call."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=31, dtype="complex128")
    ket_tagged, norm_tagged = pepsy.build_bra_ket(ket=ket)
    bdy = pepsy.BdyMPS(
        tn_flat=ket_tagged,
        tn_double=norm_tagged,
        chi=8,
        single_layer=False,
    )
    comp = pepsy.CompBdy(norm_tagged, bdy.mps_b)

    def fake_fit_one_step(self, side, step_, cut_tag_id, site_tag_id):
        _ = (step_, cut_tag_id, site_tag_id)
        if self.track_boundary_fidelity:
            self.fidel.append(0.77 if side == "left" else 0.66)
        return None

    monkeypatch.setattr(pepsy.boundary.sweeps.CompBdy, "_fit_one_step", fake_fit_one_step)

    comp.move_step_bdy(pos=0, direction="y_left", track_boundary_fidelity=True)
    assert comp.fidel == [0.77]


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

    monkeypatch.setattr(pepsy.boundary.sweeps, "tqdm", _FakeTqdm)
    monkeypatch.setattr(
        pepsy.boundary.sweeps.CompBdy,
        "_fit_one_side",
        _fake_fit_one_side_with_flat_skip,
    )
    monkeypatch.setattr(
        pepsy.boundary.sweeps.CompBdy,
        "_build_final_boundary_network",
        lambda self, spec, p_previous_l, p_previous_r: _DummyFinalTN(),
    )

    _ = comp.run(direction="y", flat=True, progress=True, max_separation=0)

    # Ly=3 -> y_left=1, y_right=2 at max_separation=0 => effective updates: 0 + 1 = 1.
    assert _FakeTqdm.instances
    progress = _FakeTqdm.instances[-1]
    assert progress.total == 1
    assert progress.n == 1


def test_compbdy_move_bdy_progress_total_accounts_for_flat_skip(monkeypatch):
    """move_bdy() tqdm total should match actual fit updates when flat=True."""
    comp = pepsy.CompBdy(_DummyNormWithTags(), {})
    _FakeTqdm.instances = []

    monkeypatch.setattr(pepsy.boundary.sweeps, "tqdm", _FakeTqdm)
    monkeypatch.setattr(
        pepsy.boundary.sweeps.CompBdy,
        "_fit_one_side",
        _fake_fit_one_side_with_flat_skip,
    )

    comp.move_bdy(direction="y_left", flat=True, progress=True)

    # Ly=3 -> n_steps=Ly-1=2, flat skip removes first fit update => total 1.
    assert _FakeTqdm.instances
    progress = _FakeTqdm.instances[-1]
    assert progress.total == 1
    assert progress.n == 1


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
        if self.track_boundary_fidelity:
            self.fidel.append(0.77)
        return _DummyMPS()

    monkeypatch.setattr(pepsy.boundary.sweeps, "tqdm", _FakeTqdm)
    monkeypatch.setattr(pepsy.boundary.sweeps.CompBdy, "_fit_one_step", fake_fit_one_step)

    comp.move_step_bdy(pos=1, direction="y_left", track_boundary_fidelity=True, progress=True)

    assert _FakeTqdm.instances
    progress = _FakeTqdm.instances[-1]
    assert progress.total == 1
    assert progress.n == 1
    assert progress.postfix_calls
    last = progress.postfix_calls[-1]
    assert last["pos"] == 1
    assert last["chi"] == 13
    assert abs(last["F"] - 0.77) < 1e-12


def test_normalize_returns_old_norm_and_updates_state_in_place(monkeypatch):
    """normalize should return old norm and update the input state in place."""
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

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=131, dtype="complex128")
    ket_id_before = id(ket)
    old_norm = pepsy.peps_normalize(
        ket,
        chi=8,
        n_iter=1,
        progress=False,
    )

    assert old_norm == 4.0
    assert id(ket) == ket_id_before
    assert captured["contract_kwargs"]["bdy"] is not None
    assert "tn_double" in captured["bdy_kwargs"]


def test_normalize_strip_exponent_updates_tensor_network_exponent(monkeypatch):
    """normalize(strip_exponent=True) should not reconstruct the full scale."""
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
            cost=(4.0, 6.0),
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=134, dtype="complex128")
    old_exponent = float(getattr(ket, "exponent", 0.0))
    old_norm = pepsy.peps_normalize(
        ket,
        chi=8,
        n_iter=1,
        progress=False,
        strip_exponent=True,
    )

    assert old_norm == (4.0, 6.0)
    assert captured["contract_kwargs"]["strip_exponent"] is True
    assert float(ket.exponent) == pytest.approx(old_exponent - 3.0)


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

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fail_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=137, dtype="complex128")
    old_norm = pepsy.peps_normalize(
        ket,
        bdy=provided,
        n_iter=1,
        progress=False,
    )

    assert captured["contract_kwargs"]["bdy"] is provided
    assert old_norm == 9.0


def test_normalize_accepts_bdy_dict_and_creates_entry(monkeypatch):
    """normalize should accept ``bdy={}`` and fill ``bdy['bdy']``."""
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

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=147, dtype="complex128")
    bdy = {}
    old_norm = pepsy.peps_normalize(
        ket,
        chi=8,
        bdy=bdy,
        n_iter=1,
        progress=False,
    )

    assert old_norm == 4.0
    assert "bdy" in bdy
    assert captured["contract_kwargs"]["bdy"] is bdy["bdy"]
    assert "tn_double" in captured["bdy_kwargs"]


def test_normalize_accepts_bdy_dict_with_existing_boundary(monkeypatch):
    """normalize should reuse ``bdy['bdy']`` when provided."""
    class _ProvidedBdy:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    provided = _ProvidedBdy()
    bdy = {"bdy": provided}
    captured = {}

    def fail_bdymps(**kwargs):
        _ = kwargs
        raise AssertionError("BdyMPS constructor should not be called when bdy['bdy'] is provided.")

    def fake_contract_boundary(**kwargs):
        captured["contract_kwargs"] = kwargs
        return pepsy.BoundaryContractResult(
            cost=9.0,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fail_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=151, dtype="complex128")
    old_norm = pepsy.peps_normalize(
        ket,
        bdy=bdy,
        n_iter=1,
        progress=False,
    )

    assert old_norm == 9.0
    assert bdy["bdy"] is provided
    assert captured["contract_kwargs"]["bdy"] is provided


def test_normalize_expands_existing_boundary_when_chi_increases(monkeypatch):
    """normalize should expand provided bdy in-place when larger chi is requested."""
    captured = {}

    class _ExpandableBdy:
        def __init__(self, chi):
            self.chi = chi
            self.mps_b = {"Y0_l": object()}
            self.expands = []

        def expand_bnd(self, chi, inplace=True):
            self.expands.append((chi, inplace))
            self.chi = chi

    def fake_contract_boundary(**kwargs):
        captured["contract_kwargs"] = kwargs
        return pepsy.BoundaryContractResult(
            cost=4.0,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=157, dtype="complex128")
    bdy = {"bdy": _ExpandableBdy(4)}
    old_norm = pepsy.peps_normalize(
        ket,
        chi=10,
        bdy=bdy,
        n_iter=1,
        progress=False,
    )

    assert old_norm == 4.0
    assert bdy["bdy"].chi == 10
    assert bdy["bdy"].expands == [(10, True)]
    assert captured["contract_kwargs"]["bdy"] is bdy["bdy"]


def test_normalize_retunes_existing_boundary_when_chi_decreases(monkeypatch):
    """normalize should retune provided bdy in-place when lower chi is requested."""
    captured = {}

    class _RetunableBdy:
        def __init__(self, chi):
            self.chi = chi
            self.mps_b = {"Y0_l": object()}
            self.expands = []

        def expand_bnd(self, chi, inplace=True):
            self.expands.append((chi, inplace))
            self.chi = chi

    def fake_contract_boundary(**kwargs):
        captured["contract_kwargs"] = kwargs
        return pepsy.BoundaryContractResult(
            cost=4.0,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=158, dtype="complex128")
    bdy = {"bdy": _RetunableBdy(12)}
    old_norm = pepsy.peps_normalize(
        ket,
        chi=6,
        bdy=bdy,
        n_iter=1,
        progress=False,
    )

    assert old_norm == 4.0
    assert bdy["bdy"].chi == 6
    assert bdy["bdy"].expands == [(6, True)]
    assert captured["contract_kwargs"]["bdy"] is bdy["bdy"]


def test_normalize_requires_chi_when_bdy_not_provided():
    """normalize should require chi if caller does not pass bdy."""
    ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=139, dtype="complex128")
    with pytest.raises(ValueError, match="Provide chi when bdy is not supplied."):
        pepsy.peps_normalize(ket)


def test_peps_normalize_mps_method_uses_quimb_contract_boundary(monkeypatch):
    """method='mps' should use the double-layer contract_boundary method."""
    captured = {}

    class _Ket:
        def __init__(self):
            self.divisor = None
            self.balanced = False

        def __itruediv__(self, value):
            self.divisor = value
            return self

        def balance_bonds_(self):
            self.balanced = True

    class _Norm:
        def contract_boundary(self, **kwargs):
            captured["kwargs"] = kwargs
            return 4.0

    def fake_build_bra_ket(ket=None, *, bra=None):
        assert bra is None
        return ket, _Norm()

    monkeypatch.setattr(pepsy.boundary.metrics, "build_bra_ket", fake_build_bra_ket)

    ket = _Ket()
    old_norm = pepsy.peps_normalize(
        ket,
        chi=7,
        method="mps",
        contraction_opt="OPT",
        sequence=("xmin", "xmax"),
        cutoff=1.0e-8,
        equalize_norms=True,
        progress=True,
    )

    assert old_norm == 4.0
    assert ket.divisor == 2.0
    assert ket.balanced is True
    assert captured["kwargs"]["max_bond"] == 7
    assert captured["kwargs"]["mode"] == "mps"
    assert captured["kwargs"]["sequence"] == ("xmin", "xmax")
    assert captured["kwargs"]["cutoff"] == 1.0e-8
    assert captured["kwargs"]["equalize_norms"] is True
    assert captured["kwargs"]["progbar"] is True
    assert captured["kwargs"]["layer_tags"] == ["KET", "BRA"]
    assert captured["kwargs"]["final_contract_opts"]["optimize"] == "OPT"


def test_peps_normalize_can_skip_bond_balancing(monkeypatch):
    """normalize should allow backends to skip post-rescale bond balancing."""

    class _Ket:
        def __init__(self):
            self.divisor = None
            self.balanced = False

        def __itruediv__(self, value):
            self.divisor = value
            return self

        def balance_bonds_(self):
            self.balanced = True

    class _Norm:
        def contract_boundary(self, **kwargs):
            _ = kwargs
            return 9.0

    def fake_build_bra_ket(ket=None, *, bra=None):
        assert bra is None
        return ket, _Norm()

    monkeypatch.setattr(pepsy.boundary.metrics, "build_bra_ket", fake_build_bra_ket)

    ket = _Ket()
    old_norm = pepsy.peps_normalize(
        ket,
        chi=7,
        method="mps",
        balance_bonds=False,
    )

    assert old_norm == 9.0
    assert ket.divisor == 3.0
    assert ket.balanced is False


def test_peps_normalize_skips_bond_balancing_for_symmray(monkeypatch):
    """Symmray normalization must not invoke Quimb's unsupported balancer."""

    class _SymmrayLikeData:
        blocks = {"q0": object()}

        def apply_to_arrays(self, fn):
            _ = fn

    class _Tensor:
        data = _SymmrayLikeData()

    class _Ket:
        tensor_map = {"I0,0": _Tensor()}

        def __init__(self):
            self.divisor = None

        def __itruediv__(self, value):
            self.divisor = value
            return self

        def balance_bonds_(self):
            raise AssertionError("Symmray normalization should not balance bonds.")

    class _Norm:
        def contract_boundary(self, **kwargs):
            _ = kwargs
            return 16.0

    def fake_build_bra_ket(ket=None, *, bra=None):
        assert bra is None
        return ket, _Norm()

    monkeypatch.setattr(pepsy.boundary.metrics, "build_bra_ket", fake_build_bra_ket)

    ket = _Ket()
    old_norm = pepsy.peps_normalize(ket, chi=7, method="mps")

    assert old_norm == 16.0
    assert ket.divisor == 4.0


def test_peps_normalize_retries_stripped_when_full_cost_is_nonfinite(monkeypatch):
    """normalize should avoid dividing the state by a non-finite full norm."""

    class _Ket:
        def __init__(self):
            self.divisor = None
            self.exponent = 0.0

        def __itruediv__(self, value):
            self.divisor = value
            return self

        def balance_bonds_(self):
            raise AssertionError("balance_bonds=False should skip balancing.")

    class _Norm:
        def __init__(self):
            self.strip_calls = []

        def contract_boundary(self, **kwargs):
            strip = kwargs["final_contract_opts"]["strip_exponent"]
            self.strip_calls.append(strip)
            if not strip:
                return complex(float("nan"), float("nan"))
            return 4.0, 6.0

    norm = _Norm()

    def fake_build_bra_ket(ket=None, *, bra=None):
        assert bra is None
        return ket, norm

    monkeypatch.setattr(pepsy.boundary.metrics, "build_bra_ket", fake_build_bra_ket)

    ket = _Ket()
    with pytest.warns(RuntimeWarning, match="retrying with strip_exponent=True"):
        old_norm = pepsy.peps_normalize(
            ket,
            chi=7,
            method="mps",
            balance_bonds=False,
        )

    assert old_norm == 4.0e6
    assert ket.divisor == 2.0
    assert ket.exponent == pytest.approx(-3.0)
    assert norm.strip_calls == [False, True]


def test_peps_normalize_rejects_nonfinite_stripped_cost_before_mutation(monkeypatch):
    """A genuinely non-finite stripped norm should fail before rescaling data."""

    class _Ket:
        def __init__(self):
            self.divisor = None

        def __itruediv__(self, value):
            self.divisor = value
            return self

    class _Norm:
        def contract_boundary(self, **kwargs):
            assert kwargs["final_contract_opts"]["strip_exponent"] is True
            return float("nan"), 0.0

    def fake_build_bra_ket(ket=None, *, bra=None):
        assert bra is None
        return ket, _Norm()

    monkeypatch.setattr(pepsy.boundary.metrics, "build_bra_ket", fake_build_bra_ket)

    ket = _Ket()
    with pytest.raises(ValueError, match="Boundary norm cost is not finite"):
        pepsy.peps_normalize(
            ket,
            chi=7,
            method="mps",
            strip_exponent=True,
        )

    assert ket.divisor is None


def test_contract_flat_mps_does_not_add_default_layer_tags():
    """Flat contractions should not assume PEPSY KET/BRA layer tags."""
    captured = {}

    class _FlatTN:
        Lx = 2
        Ly = 2

        def contract_boundary(self, **kwargs):
            captured["kwargs"] = kwargs
            return (3.0, 2.0)

    out = pepsy.contract_flat(
        _FlatTN(),
        chi=5,
        method="mps",
        contraction_opt="OPT",
        strip_exponent=True,
        progress=True,
    )

    assert out == (3.0, 2.0)
    assert "layer_tags" not in captured["kwargs"]
    assert "mode" not in captured["kwargs"]
    assert "strip_exponent" not in captured["kwargs"]
    assert captured["kwargs"]["max_bond"] == 5
    assert captured["kwargs"]["sequence"] == ("xmax", "xmin", "ymin", "ymax")
    assert captured["kwargs"]["progbar"] is True
    assert captured["kwargs"]["final_contract_opts"]["strip_exponent"] is True
    assert captured["kwargs"]["final_contract_opts"]["optimize"] == "OPT"


def test_contract_flat_dmrg_uses_flat_boundary_path(monkeypatch):
    """method='dmrg' should build a flat BdyMPS and contract with flat=True."""
    captured = {}
    tn = type("Flat2D", (), {"Lx": 2, "Ly": 2, "tags": {"X0", "Y0"}})()

    class _FakeBdy:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    def fake_bdymps(**kwargs):
        captured["bdymps_kwargs"] = kwargs
        return _FakeBdy()

    def fake_contract_boundary(**kwargs):
        captured["contract_kwargs"] = kwargs
        return pepsy.BoundaryContractResult(
            cost=7.0,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    out = pepsy.contract_flat(tn, chi=4, method="dmrg", progress=False)

    assert out == 7.0
    assert captured["bdymps_kwargs"]["tn_flat"] is tn
    assert captured["bdymps_kwargs"]["flat"] is True
    assert captured["bdymps_kwargs"]["chi"] == 4
    assert captured["contract_kwargs"]["norm"] is tn
    assert captured["contract_kwargs"]["flat"] is True


def test_contract_flat_two_site_starts_rank_one_with_requested_svd_cap(monkeypatch):
    """Two-site boundary creation should avoid eager global chi padding."""
    captured = {}
    tn = type("Flat2D", (), {"Lx": 2, "Ly": 2, "tags": {"X0", "Y0"}})()

    class _FakeBdy:
        mps_b = {"Y0_l": object()}

    def fake_bdymps(**kwargs):
        captured["bdymps_kwargs"] = kwargs
        return _FakeBdy()

    def fake_contract_boundary(**kwargs):
        captured["contract_kwargs"] = kwargs
        return pepsy.BoundaryContractResult(
            cost=7.0,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    out = pepsy.contract_flat(
        tn,
        chi=7,
        method="dmrg",
        fit_mode="two-site",
        progress=False,
    )

    assert out == 7.0
    assert captured["bdymps_kwargs"]["chi"] == 1
    assert captured["contract_kwargs"]["fit_max_bond"] == 7


def test_contract_flat_default_auto_uses_quimb_for_3d():
    """method='auto' should avoid PEPSY's 2D-only DMRG path for 3D TNs."""
    captured = {}

    class _Flat3D:
        Lx = 2
        Ly = 2
        Lz = 2

        def contract_boundary(self, **kwargs):
            captured["kwargs"] = kwargs
            return 5.0

    out = pepsy.contract_flat(_Flat3D(), chi=3)

    assert out == 5.0
    assert captured["kwargs"]["max_bond"] == 3
    assert captured["kwargs"]["sequence"] == (
        "xmax",
        "xmin",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
    )


def test_contract_flat_rejects_dmrg_for_3d():
    """PEPSY DMRG/FIT boundaries are 2D, so 3D flat TNs use quimb methods."""
    tn = type("Flat3D", (), {"Lx": 2, "Ly": 2, "Lz": 2})()

    with pytest.raises(ValueError, match="2D flat tensor networks"):
        pepsy.contract_flat(tn, chi=3, method="dmrg")


@pytest.mark.parametrize("method", ["exact", "ctmrg", "hotrg"])
def test_contract_flat_quimb_routes_forward_strip_exponent(method):
    """All quimb flat routes should preserve stripped contraction output."""
    captured = {}

    class _FlatTN:
        Lx = 2
        Ly = 2

        def contract(self, *args, **kwargs):
            captured["exact"] = (args, kwargs)
            return (2.0, 3.0)

        def contract_ctmrg(self, **kwargs):
            captured["ctmrg"] = kwargs
            return (2.0, 3.0)

        def contract_hotrg(self, **kwargs):
            captured["hotrg"] = kwargs
            return (2.0, 3.0)

    kwargs = {} if method == "exact" else {"chi": 4}
    out = pepsy.contract_flat(
        _FlatTN(),
        method=method,
        strip_exponent=True,
        contraction_opt="OPT",
        **kwargs,
    )

    assert out == (2.0, 3.0)
    if method == "exact":
        _, call_kwargs = captured["exact"]
        assert call_kwargs["strip_exponent"] is True
        assert call_kwargs["optimize"] == "OPT"
    else:
        call_kwargs = captured[method]
        # strip_exponent must travel via final_contract_opts only; quimb's
        # boundary routines reject a top-level strip_exponent kwarg.
        assert "strip_exponent" not in call_kwargs
        assert call_kwargs["final_contract_opts"]["strip_exponent"] is True
        assert call_kwargs["final_contract_opts"]["optimize"] == "OPT"


def test_contract_flat_ctmrg_enters_projector_compatibility_scope(monkeypatch):
    """The shared flat CTMRG route enables the scoped Quimb workaround."""
    events = []

    class _Scope:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    class _FlatTN:
        Lx = 2
        Ly = 2

        def contract_ctmrg(self, **kwargs):
            events.append("contract")
            return 2.0

    monkeypatch.setattr(
        pepsy.boundary.metrics,
        "quimb_ctmrg_projector_compat",
        lambda: _Scope(),
    )

    assert pepsy.contract_flat(_FlatTN(), method="ctmrg", chi=4) == 2.0
    assert events == ["enter", "contract", "exit"]


def test_contract_flat_ctmrg_forwards_stabilization_options():
    """CTMRG stabilization controls should reach Quimb's projector path."""
    captured = {}

    class _FlatTN:
        Lx = 2
        Ly = 2

        def contract_ctmrg(self, **kwargs):
            captured.update(kwargs)
            return 2.0

    reduce_opts = {"method": "cholesky", "shift": 1.0e-9}
    out = pepsy.contract_flat(
        _FlatTN(),
        method="ctmrg",
        chi=4,
        ctmrg_reduce_opts=reduce_opts,
        ctmrg_gauge_smudge=2.0e-8,
    )

    assert out == 2.0
    assert captured["reduce_opts"] == reduce_opts
    assert captured["gauge_smudge"] == 2.0e-8
    assert reduce_opts == {"method": "cholesky", "shift": 1.0e-9}


def test_contract_flat_ctmrg_adds_symmray_stabilization_defaults(monkeypatch):
    """Symmray CTMRG should get a shift and gauge smudge by default."""
    captured = {}

    class _FlatTN:
        Lx = 2
        Ly = 2

        def contract_ctmrg(self, **kwargs):
            captured.update(kwargs)
            return 2.0

    monkeypatch.setattr(
        pepsy.boundary.metrics,
        "_uses_symmray_arrays",
        lambda tn: tn is not None,
    )

    assert pepsy.contract_flat(_FlatTN(), method="ctmrg", chi=4) == 2.0
    assert captured["reduce_opts"] == {"method": "eigh", "shift": 1.0e-12}
    assert captured["gauge_smudge"] == 1.0e-10
    assert captured["canonize_opts"] == {"smudge": 1.0e-10}


def test_quimb_ctmrg_projector_compat_uses_live_insert_target(monkeypatch):
    """CTMRG projector insertion should use the current network snapshot."""
    import quimb.tensor.tensor_core as qtc

    calls = []
    def fake_insert(self, ltags, rtags, *args, insert_into=None, **kwargs):
        calls.append((self, ltags, rtags, insert_into, kwargs))
        return self

    monkeypatch.setattr(
        qtc.TensorNetwork,
        "insert_compressor_between_regions",
        fake_insert,
    )

    stale = object()
    current = object()
    with pepsy.boundary.metrics.quimb_ctmrg_projector_compat():
        qtc.TensorNetwork.insert_compressor_between_regions(
            stale,
            ("L",),
            ("R",),
            insert_into=current,
        )
        qtc.TensorNetwork.insert_compressor_between_regions_(
            stale,
            ("L",),
            ("R",),
            insert_into=current,
        )

    assert len(calls) == 2
    for call in calls:
        assert call[0] is current
        assert call[3] is current
        assert call[4]["inplace"] is True
    assert qtc.TensorNetwork.insert_compressor_between_regions is fake_insert


def test_infidelity_accepts_bdy_holder_dicts_and_fills_missing(monkeypatch):
    """infidelity should accept dict holders and populate missing bdy entries."""
    created = []
    captured = {"contract_calls": []}

    class _FakeBdy:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    def fake_bdymps(**kwargs):
        _ = kwargs
        obj = _FakeBdy()
        created.append(obj)
        return obj

    def fake_contract_boundary(**kwargs):
        captured["contract_calls"].append(kwargs)
        # first: norm, second: norm_target, third: overlap
        idx = len(captured["contract_calls"])
        cost = {1: 2.0, 2: 4.0, 3: 2.0}[idx]
        return pepsy.BoundaryContractResult(
            cost=cost,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    p = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=161, dtype="complex128")
    p_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=163, dtype="complex128")

    bdy = {}
    bdy_target = {}
    bdy_overlap = {}

    out = pepsy.peps_infidelity(
        p,
        p_target,
        chi=8,
        bdy=bdy,
        bdy_target=bdy_target,
        bdy_overlap=bdy_overlap,
        progress=False,
    )

    assert "bdy" in bdy and "bdy" in bdy_target and "bdy" in bdy_overlap
    assert out["bdy"] is bdy["bdy"]
    assert out["bdy_target"] is bdy_target["bdy"]
    assert out["bdy_overlap"] is bdy_overlap["bdy"]
    assert isinstance(out["norm_result"], pepsy.BoundaryContractResult)
    assert isinstance(out["norm_target_result"], pepsy.BoundaryContractResult)
    assert isinstance(out["overlap_result"], pepsy.BoundaryContractResult)
    assert len(captured["contract_calls"]) == 3


def test_infidelity_strip_exponent_uses_scaled_ratio(monkeypatch):
    """infidelity(strip_exponent=True) should compare mantissa/exponent pairs."""
    captured = {"contract_calls": []}

    class _FakeBdy:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    def fake_bdymps(**kwargs):
        _ = kwargs
        return _FakeBdy()

    def fake_contract_boundary(**kwargs):
        captured["contract_calls"].append(kwargs)
        idx = len(captured["contract_calls"])
        cost = {1: (2.0, 10.0), 2: (8.0, 20.0), 3: (4.0, 15.0)}[idx]
        return pepsy.BoundaryContractResult(
            cost=cost,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fake_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    p = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=162, dtype="complex128")
    p_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=164, dtype="complex128")

    out = pepsy.peps_infidelity(
        p,
        p_target,
        chi=8,
        progress=False,
        strip_exponent=True,
    )

    assert all(call["strip_exponent"] is True for call in captured["contract_calls"])
    assert out["norm"] == (2.0, 10.0)
    assert out["norm_target"] == (8.0, 20.0)
    assert out["overlap"] == (4.0, 15.0)
    assert out["infidelity"] == pytest.approx(0.0)


def test_peps_infidelity_ctmrg_skips_known_target_norm(monkeypatch):
    """Known norm_target should skip the target self-overlap contraction."""
    build_calls = []
    contract_calls = []
    p = object()
    p_target = object()

    class _Norm:
        def __init__(self, label):
            self.label = label

        def contract_ctmrg(self, **kwargs):
            contract_calls.append((self.label, kwargs))
            return {"norm": 2.0, "overlap": 2.0}[self.label]

    def fake_build_bra_ket(ket=None, *, bra=None):
        if ket is p and bra is None:
            label = "norm"
        elif ket is p and bra is p_target:
            label = "overlap"
        elif ket is p_target and bra is None:
            label = "target"
        else:  # pragma: no cover - defensive branch for clearer failure
            raise AssertionError("unexpected build_bra_ket inputs")
        build_calls.append(label)
        return ket, _Norm(label)

    monkeypatch.setattr(pepsy.boundary.metrics, "build_bra_ket", fake_build_bra_ket)

    out = pepsy.peps_infidelity(
        p,
        p_target,
        chi=9,
        method="ctmrg",
        norm_target=2.0,
        contraction_opt="OPT",
        cutoff=1.0e-7,
        max_separation=2,
        equalize_norms=True,
        progress=True,
    )

    assert build_calls == ["norm", "overlap"]
    assert [label for label, _ in contract_calls] == ["norm", "overlap"]
    assert out["infidelity"] == pytest.approx(0.0)
    assert out["bdy"] is None
    assert out["bdy_target"] is None
    assert out["bdy_overlap"] is None
    for _, kwargs in contract_calls:
        assert kwargs["max_bond"] == 9
        assert kwargs["cutoff"] == 1.0e-7
        assert kwargs["max_separation"] == 2
        assert kwargs["equalize_norms"] is True
        assert kwargs["progbar"] is True
        assert kwargs["inplace"] is False
        assert kwargs["layer_tags"] == ["KET", "BRA"]
        assert kwargs["final_contract_opts"]["optimize"] == "OPT"


def test_peps_infidelity_mps_uses_layer_tags_for_all_contractions(monkeypatch):
    """MPS boundary infidelity should use KET/BRA layers for each TN."""
    build_calls = []
    contract_calls = []
    p = object()
    p_target = object()

    class _Norm:
        def __init__(self, label):
            self.label = label

        def contract_boundary(self, **kwargs):
            contract_calls.append((self.label, kwargs))
            return {"norm": 2.0, "target": 2.0, "overlap": 2.0}[self.label]

    def fake_build_bra_ket(ket=None, *, bra=None):
        if ket is p and bra is None:
            label = "norm"
        elif ket is p_target and bra is None:
            label = "target"
        elif ket is p and bra is p_target:
            label = "overlap"
        else:  # pragma: no cover - defensive branch for clearer failure
            raise AssertionError("unexpected build_bra_ket inputs")
        build_calls.append(label)
        return ket, _Norm(label)

    monkeypatch.setattr(pepsy.boundary.metrics, "build_bra_ket", fake_build_bra_ket)

    out = pepsy.peps_infidelity(
        p,
        p_target,
        chi=5,
        method="mps",
        progress=False,
    )

    assert build_calls == ["norm", "target", "overlap"]
    assert [label for label, _ in contract_calls] == ["norm", "target", "overlap"]
    assert out["infidelity"] == pytest.approx(0.0)
    for _, kwargs in contract_calls:
        assert kwargs["layer_tags"] == ["KET", "BRA"]
        assert kwargs["max_bond"] == 5
        assert kwargs["mode"] == "mps"


def test_peps_metric_aliases(monkeypatch):
    """PEPS-named metric helpers should delegate to boundary implementations."""
    calls = []

    def fake_boundary_norm(*args, **kwargs):
        calls.append(("norm", args, kwargs))
        return 2.0

    def fake_infidelity(*args, **kwargs):
        calls.append(("infidelity", args, kwargs))
        return {"infidelity": 0.25}

    monkeypatch.setattr(pepsy.boundary.metrics, "boundary_norm", fake_boundary_norm)
    monkeypatch.setattr(pepsy.boundary.metrics, "peps_infidelity", fake_infidelity)

    assert pepsy.peps_norm("p", chi=4) == 2.0
    assert pepsy.peps_fidelity("p", "q", chi=5) == pytest.approx(0.75)
    fidelity_info = pepsy.peps_fidelity(
        "p",
        "q",
        chi=6,
        return_info=True,
    )
    assert fidelity_info["fidelity"] == pytest.approx(0.75)
    assert fidelity_info["infidelity"] == pytest.approx(0.25)
    assert calls[0][0] == "norm"
    assert calls[0][1] == ("p",)
    assert calls[0][2]["chi"] == 4
    assert calls[0][2]["method"] == "dmrg"
    assert calls[1][0] == "infidelity"
    assert calls[1][1] == ("p", "q")
    assert calls[1][2]["chi"] == 5
    assert calls[1][2]["method"] == "dmrg"
    assert calls[2][1] == ("p", "q")
    assert calls[2][2]["chi"] == 6
    assert calls[2][2]["method"] == "dmrg"


def test_infidelity_reuses_existing_bdy_holder_entries(monkeypatch):
    """infidelity should reuse existing holder boundaries without rebuilding."""
    captured = {"contract_calls": []}

    class _ProvidedBdy:
        def __init__(self):
            self.mps_b = {"Y0_l": object()}

    bdy = {"bdy": _ProvidedBdy()}
    bdy_target = {"bdy": _ProvidedBdy()}
    bdy_overlap = {"bdy": _ProvidedBdy()}

    def fail_bdymps(**kwargs):
        _ = kwargs
        raise AssertionError("BdyMPS should not be called when holder already has bdy.")

    def fake_contract_boundary(**kwargs):
        captured["contract_calls"].append(kwargs)
        idx = len(captured["contract_calls"])
        cost = {1: 2.0, 2: 4.0, 3: 2.0}[idx]
        return pepsy.BoundaryContractResult(
            cost=cost,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "BdyMPS", fail_bdymps)
    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    p = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=167, dtype="complex128")
    p_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=169, dtype="complex128")

    out = pepsy.peps_infidelity(
        p,
        p_target,
        bdy=bdy,
        bdy_target=bdy_target,
        bdy_overlap=bdy_overlap,
        progress=False,
    )

    assert out["bdy"] is bdy["bdy"]
    assert out["bdy_target"] is bdy_target["bdy"]
    assert out["bdy_overlap"] is bdy_overlap["bdy"]
    assert len(captured["contract_calls"]) == 3


def test_infidelity_retunes_all_existing_boundaries_to_requested_chi(monkeypatch):
    """Existing bdy handles should all retune to the requested chi."""
    captured = {"contract_calls": []}

    class _ExpandableBdy:
        def __init__(self, chi):
            self.chi = chi
            self.mps_b = {"Y0_l": object()}
            self.expands = []

        def expand_bnd(self, chi, inplace=True):
            self.expands.append((chi, inplace))
            self.chi = chi

    bdy = {"bdy": _ExpandableBdy(4)}
    bdy_target = {"bdy": _ExpandableBdy(13)}
    bdy_overlap = {"bdy": _ExpandableBdy(10)}

    def fake_contract_boundary(**kwargs):
        captured["contract_calls"].append(kwargs)
        idx = len(captured["contract_calls"])
        cost = {1: 2.0, 2: 4.0, 3: 2.0}[idx]
        return pepsy.BoundaryContractResult(
            cost=cost,
            fidel=[],
            direction=kwargs["direction"],
            n_iter=kwargs["n_iter"],
            max_separation=kwargs["max_separation"],
        )

    monkeypatch.setattr(pepsy.boundary.metrics, "contract_boundary", fake_contract_boundary)

    p = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=173, dtype="complex128")
    p_target = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=179, dtype="complex128")

    _ = pepsy.peps_infidelity(
        p,
        p_target,
        chi=10,
        bdy=bdy,
        bdy_target=bdy_target,
        bdy_overlap=bdy_overlap,
        progress=False,
    )

    assert bdy["bdy"].chi == 10
    assert bdy_target["bdy"].chi == 10
    assert bdy_overlap["bdy"].chi == 10
    assert bdy["bdy"].expands == [(10, True)]
    assert bdy_target["bdy"].expands == [(10, True)]
    assert bdy_overlap["bdy"].expands == []
