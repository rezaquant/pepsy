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


def test_bdymps_avg_mps_norm_matches_manual_mean():
    """avg_mps_norm should equal manual mean over all boundary MPS norms."""
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
    assert bdy.avg_mps_norm == manual_avg
