"""Tests for Symmray-backed symmetric MPS/PEPS helpers."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.optimizers import sym_dmrg as sym_dmrg_mod
from pepsy.operators import gate, gate_simple
from pepsy.optimizers.sym_dmrg import (
    _BlockPairContraction,
    _array_with_blocks_like,
    _tensor_with_data,
)
from pepsy.tensors import symmetric as symmetric_mod
from pepsy.tensors import (
    Fermion,
    FermionLatticeSetup,
    OneDMap,
    SpinfulFermion,
    SpinfulFermionHubbard,
    SymmFermions,
    SymGateStream,
    SymHamiltonian,
    SymMPS,
    SymPEPS,
    default_physical_sectors,
    draw_symmray_blocks,
    draw_symmray_mps,
    draw_symmray_mpo,
    draw_symmray_peps,
    fermi_hubbard_u1u1_gate_stream,
    fermi_hubbard_u1u1_hopping_gate_stream,
    fermi_hubbard_u1u1_interaction_gate_stream,
    fermi_hubbard_u1u1_light_pulse_gate_stream,
    fermion_density_param_gen,
    fermion_hopping_param_gen,
    fermion_interaction_param_gen,
    sector_index_map,
    site_charge_alternating,
    site_charge_from_map,
    site_charge_from_occupations,
    site_charge_uniform,
    symmray_block_summary,
    symmray_mps_summary,
    symmray_mpo_summary,
    symmray_peps_summary,
    symm_operator_from_dense,
)


sr = pytest.importorskip("symmray")


@pytest.mark.parametrize(
    ("symmetry", "expected_model", "expected_charge"),
    [
        ("U1", "fermi_hubbard", 4),
        ("U1U1", "fermi_hubbard_u1u1", (2, 2)),
    ],
)
def test_spinful_fermion_helper_bundles_symmetry_aware_building_blocks(
    symmetry, expected_model, expected_charge
):
    """The convenience helper should cover the native U1 and U1U1 workflows."""
    fermions = SpinfulFermion(symmetry=symmetry)

    assert SpinfulFermionHubbard is SpinfulFermion
    assert not hasattr(fermions, "t")
    assert not hasattr(fermions, "U")
    assert not hasattr(fermions, "V")
    assert not hasattr(fermions, "mu")
    with pytest.raises(TypeError, match="unexpected keyword argument 't'"):
        SpinfulFermion(symmetry=symmetry, t=1.0)
    assert isinstance(SymmFermions.spinful(symmetry=symmetry), SpinfulFermion)
    assert pepsy.SpinfulFermion is SpinfulFermion
    assert pepsy.SymmFermions is SymmFermions
    occupations = fermions.half_filled_occupations(4)
    assert fermions.model == expected_model
    assert fermions.total_charge(occupations) == expected_charge
    assert fermions.physical_sectors == default_physical_sectors(model=expected_model)
    assert np.allclose(
        fermions.dense_operator("pair_create")
        @ fermions.dense_operator("pair_annihilate"),
        fermions.dense_operator("doublon"),
    )

    pair_create = fermions.observable("pair_create")
    assert pair_create is fermions.observable("pair_create")
    assert tuple(pair_create.shape) == (4, 4)
    assert fermions.operator_charge("pair_create") == fermions.pair_charge

    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    assert fermions.edge_coloring_layers(edges) == (
        ((0, 1), (2, 3)),
        ((1, 2), (3, 0)),
    )
    stream = fermions.strang_gate_stream(
        edges, 0.02, sites=range(4), t=1.0, U=3.0
    )
    assert isinstance(stream, SymGateStream)
    assert stream.order == 2
    assert len(stream) == 16
    assert all(len(where) == 2 for _, where in stream[4:12])

    ham = fermions.hamiltonian(
        ((0, 1), (1, 2)), t=1.0, U=3.0, mu=0.1
    )
    assert isinstance(ham, SymHamiltonian)
    assert ham.model == expected_model
    assert ham.symmetry == symmetry


def test_spinful_fermion_compatibility_constructors_enforce_spinful_space():
    """Legacy spellings must not accidentally construct a spinless helper."""
    assert SpinfulFermionHubbard is SpinfulFermion
    assert SpinfulFermion(symmetry="U1").spinful
    assert SymmFermions.spinful(symmetry="U1").spinful

    with pytest.raises(TypeError, match="always uses spinful=True"):
        SpinfulFermion(spinful=False)
    with pytest.raises(TypeError, match="always uses spinful=True"):
        SymmFermions.spinful(spinful=False)


@pytest.mark.parametrize("symmetry", ["U1", "Z2"])
def test_unified_fermion_helper_supports_spinless_native_workflow(symmetry):
    """The unified helper should expose the spinless native t-V workflow."""
    fermions = Fermion(spinful=False, symmetry=symmetry)

    assert fermions.model == "fermi_hubbard_spinless"
    assert fermions.physical_sectors == default_physical_sectors(symmetry, 2)
    np.testing.assert_allclose(
        fermions.dense_operator("n"),
        fermions.dense_operator("number"),
    )
    assert fermions.operator_charge("create") == (1 if symmetry == "Z2" else 1)
    assert type(fermions.operator("number")).__name__.endswith("FermionicArray")

    density = fermions.gate("density", 0.1, V=2.0)
    assert type(density).__name__.endswith("FermionicArray")
    assert density.to_dense().shape == (2, 2, 2, 2)

    stream = fermions.gate_stream(
        [(0, 1), (1, 2)],
        0.01,
        sites=range(3),
        order=2,
        t=1.0,
        V=2.0,
        mu=0.3,
    )
    assert isinstance(stream, SymGateStream)
    assert stream.order == 2
    assert all(len(where) == 2 for _, where in stream if isinstance(where, tuple))

    fourth = fermions.gate_stream(
        [(0, 1), (1, 2)],
        0.01,
        sites=range(3),
        order=4,
        t=1.0,
        V=2.0,
        mu=0.3,
    )
    assert fourth.order == 4
    assert len(fourth) == 3 * len(stream)

    ham = fermions.hamiltonian(((0, 1), (1, 2)), t=1.0, V=2.0, mu=0.3)
    assert ham.model == "fermi_hubbard_spinless"
    assert ham.symmetry == symmetry
    assert set(fermions.local_terms(
        ((0, 1), (1, 2)), t=1.0, V=2.0, mu=0.3
    )) == {(0, 1), (1, 2)}


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
def test_unified_fermion_complex64_gates_preserve_native_dtype(
    spinful, symmetry
):
    """Native complex64 gate streams must not rely on optimizer downcasts."""
    fermion = Fermion(spinful=spinful, symmetry=symmetry, dtype="complex64")
    params = {"t": 0.7, "mu": 0.13, "V": 0.2}
    if spinful:
        params["U"] = 1.2
    stream = list(
        fermion.gate_stream(
            ((0, 1), (1, 2)),
            0.01,
            sites=range(3),
            order=2,
            **params,
        )
    )
    direct = (
        fermion.onsite_gate(
            0.01,
            U=1.2 if spinful else None,
            mu=0.13,
        ),
        fermion.hopping_gate(0.01, t=0.7),
        fermion.density_gate(0.01, V=0.2),
    )

    for stream_gate, _where in stream:
        assert {
            np.dtype(block.dtype) for block in stream_gate.blocks.values()
        } == {
            np.dtype("complex64")
        }
    for direct_gate in direct:
        assert {
            np.dtype(block.dtype) for block in direct_gate.blocks.values()
        } == {
            np.dtype("complex64")
        }


def test_spinless_fermion_rejects_ignored_hubbard_couplings():
    """A spinless t-V model must never silently discard a Hubbard U."""
    fermion = Fermion(spinful=False, symmetry="U1")
    edges = ((0, 1),)

    with pytest.raises(TypeError, match="spinless fermions"):
        fermion.onsite_term(0, U=2.0)
    with pytest.raises(TypeError, match="spinless fermions"):
        fermion.onsite_gate(0.01, U=2.0)
    with pytest.raises(ValueError, match="Spinless.*doublon"):
        fermion.interaction_term(0, U=2.0)
    with pytest.raises(TypeError, match="spinless fermions"):
        fermion.gate_stream(edges, 0.01, t=1.0, U=2.0)
    with pytest.raises(TypeError, match="spinless fermions"):
        fermion.strang_gate_stream(edges, 0.01, t=1.0, U=2.0)
    with pytest.raises(TypeError, match="spinless fermions"):
        fermion.hamiltonian(edges, t=1.0, U=2.0)


def test_named_fermion_gate_rejects_missing_and_unknown_parameters():
    """The generic gate front door validates every coupling it accepts."""
    fermion = Fermion(spinful=True, symmetry="U1U1")

    with pytest.raises(TypeError, match="requires explicit t"):
        fermion.gate("hopping", 0.01)
    with pytest.raises(TypeError, match="requires explicit t"):
        fermion.hopping_gate(0.01, t=None)
    with pytest.raises(TypeError, match="requires explicit U"):
        fermion.gate("interaction", 0.01)
    with pytest.raises(TypeError, match="requires explicit V"):
        fermion.density_gate(0.01, V=None)
    with pytest.raises(TypeError, match="Unexpected Fermion.gate parameter"):
        fermion.gate("hopping", 0.01, t=1.0, typo=7.0)
    with pytest.raises(TypeError, match="does not accept edge"):
        fermion.gate("hopping", 0.01, t=1.0, edge=(0, 1))
    with pytest.raises(TypeError, match="at most one"):
        fermion.gate("sxx", 0.01, where=(0, 1), edge=(0, 1))
    with pytest.raises(TypeError, match="Unexpected Fermion.param_gate parameter"):
        fermion.param_gate("hopping", (0.01,), typo=7.0)


def test_fermion_streams_include_explicit_spin_fields_in_matching_hamiltonian():
    """Total-U1 Hubbard streams support longitudinal and transverse fields."""
    fermion = Fermion(spinful=True, symmetry="U1")
    edges = ((0, 1),)
    stream = fermion.strang_gate_stream(
        edges,
        0.01,
        t=1.0,
        U=2.0,
        field_x={0: 0.2, 1: -0.1},
        field_z=0.3,
    )
    hamiltonian = stream.hamiltonian

    assert hamiltonian.explicit_terms
    expected = fermion.spin_x_term(0, field=0.2) + fermion.spin_z_term(
        0, field=0.3
    )
    np.testing.assert_allclose(
        hamiltonian.terms[(0,)].to_dense(), expected.to_dense()
    )
    assert hamiltonian.to_mpo(L=2, compress=False).L == 2
    assert len(stream) == 14

    with pytest.raises(ValueError, match="symmetry='U1' or 'Z2'"):
        Fermion(spinful=True, symmetry="U1U1").strang_gate_stream(
            edges, 0.01, t=1.0, U=2.0, field_x=0.1
        )


def test_spinless_z2_streams_include_pairing_in_matching_hamiltonian():
    """Parity-preserving pairing belongs to the spinless Z2 stream only."""
    fermion = Fermion(spinful=False, symmetry="Z2")
    edges = ((0, 1),)
    stream = fermion.strang_gate_stream(
        edges,
        0.01,
        t=1.0,
        pairing=0.2,
        pairing_phase=0.3,
    )
    reference = fermion.hamiltonian(
        edges,
        t=1.0,
        pairing=0.2,
        pairing_phase=0.3,
    )

    assert stream.hamiltonian.explicit_terms
    np.testing.assert_allclose(
        stream.hamiltonian.terms[(0, 1)].to_dense(),
        reference.terms[(0, 1)].to_dense(),
    )
    assert reference.to_mpo(L=2, compress=False).L == 2
    assert len(stream) == 10

    with pytest.raises(NotImplementedError, match="spinful=False"):
        Fermion(spinful=True, symmetry="Z2").strang_gate_stream(
            edges, 0.01, t=1.0, U=2.0, pairing=0.2
        )


def test_unified_spinful_fermion_gate_stream_runs_native_mps():
    """The unified spinful model should evolve a charge-conserving MPS."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    state = SymMPS.random(
        3,
        symmetry="U1U1",
        phys_dim=fermion.physical_sectors,
        bond_dim=2,
        fermionic=True,
        site_charge=fermion.half_filled_site_charge(3),
        dtype="complex128",
        seed=1,
    )

    state.apply_gates(
        fermion.gate_stream(
            ((0, 1), (1, 2)), 0.01, sites=range(3), t=0.5, U=2.0
        ),
        method="direct",
        contract="split",
        max_bond=4,
        cutoff=1e-10,
        normalize=True,
    )

    assert np.isfinite(np.real(state.norm()))
    assert state.norm() == pytest.approx(1.0)


def test_ps_to_mps_accepts_fermion_and_builds_native_charge_sector():
    """The public MPS constructor should hide SymMPS for Fermion starts."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    mps = pepsy.ps_to_mps(4, fermion=fermion, seed=17)

    assert mps.L == 4
    assert mps.max_bond() == 1
    assert type(mps).__name__ == "MatrixProductState"
    assert not hasattr(mps, "_pepsy_fermion")
    assert all(
        type(mps[site].data).__name__ == "U1U1FermionicArray"
        for site in range(4)
    )
    assert [mps[site].data.charge for site in range(4)] == [
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
    ]
    with pytest.raises(TypeError, match="unexpected keyword argument 'chi'"):
        pepsy.ps_to_mps(4, fermion=fermion, chi=2)


def test_hrs_to_mps_fermion_custom_occupations_preserve_global_sector():
    """Random symmetric growth should preserve the requested total charge."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    occupations = ((0, 1), (1, 0), (0, 1), (1, 0))
    mps = pepsy.hrs_to_mps(
        4,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=2,
        seed=19,
    )

    assert mps.L == 4
    assert mps.max_bond() <= 2
    assert fermion.total_charge(occupations) == (2, 2)


def test_hrs_to_mps_direct_uses_symmray_random_blocks():
    """The direct method should build normalized native random blocks."""
    pytest.importorskip("symmray")
    fermion = Fermion(spinful=True, symmetry="U1U1")
    occupations = ((1, 0), (0, 1), (1, 0), (0, 1))
    mps = pepsy.hrs_to_mps(
        4,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        method="direct",
        subsizes="maximal",
        seed=37,
        dtype="complex128",
    )

    assert mps.max_bond() <= 2
    assert np.real(mps.norm()) == pytest.approx(1.0)
    assert all(
        type(mps[site].data).__name__ == "U1U1FermionicArray"
        for site in range(4)
    )


def test_ps_to_peps_accepts_fermion_coordinate_occupations():
    """The public PEPS constructor should build a native fixed-charge seed."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    occupations = {
        (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
        for x in range(2)
        for y in range(3)
    }

    peps = pepsy.ps_to_peps(
        (2, 3),
        fermion=fermion,
        occupations=occupations,
        seed=19,
        dtype="complex128",
    )

    assert (peps.Lx, peps.Ly) == (2, 3)
    assert peps.max_bond() == 1
    assert all(
        type(peps[x, y].data).__name__ == "U1U1FermionicArray"
        for x, y in occupations
    )
    assert [
        peps[x, y].data.charge
        for x, y in ((0, 0), (0, 1), (1, 0))
    ] == [(1, 0), (0, 1), (0, 1)]
    with pytest.raises(TypeError, match="unexpected keyword argument 'chi'"):
        pepsy.ps_to_peps((2, 3), fermion=fermion, chi=2)


def test_ps_to_peps_accepts_periodic_fermion_state():
    """The native fermionic PEPS constructor should preserve cyclic bonds."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    peps = pepsy.ps_to_peps(
        (3, 3),
        fermion=fermion,
        cyclic=True,
        seed=23,
        dtype="complex128",
    )

    assert (peps.Lx, peps.Ly) == (3, 3)
    assert peps.max_bond() == 1
    assert set(peps[0, 0].inds) & set(peps[0, 2].inds)
    assert set(peps[0, 0].inds) & set(peps[2, 0].inds)
    assert all(
        type(peps[x, y].data).__name__ == "U1U1FermionicArray"
        for x in range(3)
        for y in range(3)
    )


def test_hrs_to_mps_accepts_fermion_sector_and_grows_entanglement():
    """The Haar-random constructor should use chi for symmetric growth."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    occupations = ((1, 0), (0, 1), (1, 0), (0, 1))
    mps = pepsy.hrs_to_mps(
        4,
        fermion=fermion,
        occupations=occupations,
        chi=2,
        random_rounds=20,
        seed=31,
        dtype="complex128",
    )

    assert mps.L == 4
    assert 1 < mps.max_bond() <= 2
    assert all(
        type(mps[site].data).__name__ == "U1U1FermionicArray"
        for site in range(4)
    )
    assert fermion.total_charge(occupations) == (2, 2)
    assert pepsy.hrps_to_mps is pepsy.hrs_to_mps


def test_hrs_to_peps_accepts_fermion_sector_and_chi():
    """The Haar-random PEPS constructor should build native symmetric PEPS."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    occupations = {
        (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
        for x in range(2)
        for y in range(2)
    }
    peps = pepsy.hrs_to_peps(
        (2, 2),
        fermion=fermion,
        occupations=occupations,
        chi=2,
        method="direct",
        subsizes="maximal",
        seed=32,
        dtype="complex128",
        normalize=True,
    )

    assert (peps.Lx, peps.Ly) == (2, 2)
    assert 1 < peps.max_bond() <= 2
    assert np.real((peps.H & peps).contract(all)) == pytest.approx(1.0)
    assert all(
        type(peps[x, y].data).__name__ == "U1U1FermionicArray"
        for x, y in occupations
    )
    assert pepsy.hrps_to_peps is pepsy.hrs_to_peps


def test_hrs_to_peps_skips_global_norm_by_default_for_vmc(monkeypatch):
    """The VMC-safe default avoids the unrelated CPU boundary-norm contraction."""
    fermion = Fermion(spinful=True, symmetry="Z2")
    occupations = {
        (x, y): 1 if (x + y) % 2 == 0 else 0
        for x in range(2)
        for y in range(2)
    }

    def unexpected_normalize(_):
        raise AssertionError("global PEPS normalization must be skipped")

    monkeypatch.setattr(SymPEPS, "normalize", unexpected_normalize)
    peps = pepsy.hrs_to_peps(
        (2, 2),
        fermion=fermion,
        occupations=occupations,
        chi=3,
        seed=33,
    )

    assert (peps.Lx, peps.Ly) == (2, 2)
    assert 1 < peps.max_bond() <= 3


@pytest.mark.parametrize(
    ("symmetry", "expected_charge", "expected_occupation"),
    [
        ("U1U1", (3, 3), (1, 0)),
        ("U1", 6, 1),
        ("Z2", 0, 1),
    ],
)
def test_lattice_half_filling_prepares_explicit_peps_metadata(
    symmetry, expected_charge, expected_occupation
):
    """Lattice setup normalizes occupations without building terms or gates."""
    fermion = Fermion(spinful=True, symmetry=symmetry)

    setup = fermion.lattice_half_filling(3, 2, pattern="checkerboard")

    assert isinstance(setup, FermionLatticeSetup)
    assert pepsy.FermionLatticeSetup is FermionLatticeSetup
    assert (setup.Lx, setup.Ly) == (3, 2)
    assert setup.sites == tuple(
        (x, y) for x in range(3) for y in range(2)
    )
    assert len(setup.edges) == 7
    assert setup.target_charge == expected_charge
    assert setup.target_particles == 6
    assert setup.occupations[(0, 0)] == expected_occupation
    assert setup.site_charge((0, 0)) == expected_occupation
    assert setup.spin_occupations[(0, 0)] == (1, 0)
    assert setup.spin_occupations[(0, 1)] == (0, 1)


def test_symdmrg_fermionic_state_accepts_raw_fermion_constructor_output():
    """SymDMRG should restore native tensors without a SymMPS wrapper input."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    mps = pepsy.ps_to_mps(3, fermion=fermion, seed=23)
    mpo = fermion.hamiltonian(((0, 1), (1, 2)), t=0.5, U=2.0).to_mpo(
        L=3,
        max_bond=8,
        compress=True,
    )

    optimizer = pepsy.SymDMRG2(
        mpo,
        mps,
        chi=2,
        compute_initial_energy=False,
    )
    native = optimizer.fermionic_state(fermion=fermion)

    assert native.L == 3
    assert type(native[0].data).__name__ == "U1U1FermionicArray"


def _randomized_block_tensor(tensor, seed):
    """Keep a tensor's Symmray metadata while replacing its block values."""
    rng = np.random.default_rng(seed)
    blocks = {}
    for sector, block in tensor.data.blocks.items():
        values = rng.normal(size=block.shape) + 1j * rng.normal(size=block.shape)
        blocks[sector] = np.asarray(values, dtype=block.dtype)
    data = _array_with_blocks_like(tensor.data, blocks)
    return _tensor_with_data(tensor, data)


def test_symdmrg_compiled_fanout_plan_matches_blockwise_fh_matvec(monkeypatch):
    """The bosonized FH path reuses one right matrix through fanout GEMM."""
    monkeypatch.setattr(sym_dmrg_mod, "_SECTOR_OPERATOR_MAX_BYTES", 32 * 1024**2)
    mapper = OneDMap(3, 2, mode="snake")
    edges = tuple(qtn.edges_2d_square(3, 2, cyclic=True))
    mpo = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1", "U1U1", edges, t=1.0, U=8.0
    ).to_mpo(mapper=mapper, compress=True, cutoff=1e-12)
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        6,
        bond_dim=4,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)] * 3),
        seed=3,
        dtype="complex128",
    )
    optimizer = pepsy.SymDMRG2(
        mpo,
        state,
        bond_dims=[4],
        cutoffs=[1e-10],
        compute_initial_energy=False,
    )

    assert type(list(state.tn)[0].data).__name__ == "U1U1FermionicArray"
    assert type(optimizer.state[0].data).__name__ == "U1U1Array"
    theta = optimizer.two_site_theta(2)
    assert type(theta.data).__name__ == "U1U1Array"
    problem, _ = optimizer._get_projected_problem(2, theta)
    contraction = problem.right_contraction
    assert type(contraction.left.data).__name__ == "U1U1Array"

    theta_input = qtn.Tensor(
        data=_array_with_blocks_like(
            problem.theta_input_data_template,
            theta.data.blocks,
        ),
        inds=problem.theta_input_inds,
        tags=theta.tags,
    )
    contraction.apply(theta_input)  # Direct reference call that compiles the plan.
    random_theta = _randomized_block_tensor(theta_input, 13)
    reference = _BlockPairContraction(
        optimizer,
        contraction.left,
        contraction.right_inds,
    ).apply(random_theta)
    got = contraction.apply(random_theta)

    assert contraction.compiled_block_plan_uses == 1
    assert contraction.compiled_block_plan_fanout_eligible_groups > 0
    assert contraction.compiled_block_plan_fanout_groups > 0
    assert contraction.compiled_block_plan_fanout_output_blocks > 1
    assert contraction.compiled_block_plan_fanout_static_bytes > 0
    assert contraction.compiled_block_plan_fanout_predicted_matmul_savings > 0
    assert contraction.compiled_block_plan_fanout_matmul_calls > 0
    assert any(
        len({row_stop - row_start for row_start, row_stop in row_slices}) > 1
        for _, _, row_slices, _ in contraction.compiled_block_plan_fanouts
    )
    for sector, expected in reference.data.blocks.items():
        np.testing.assert_allclose(
            got.data.blocks[sector], expected, atol=1e-12, rtol=1e-12
        )

    # A compatible layout must reuse the same compiled fanout plan.
    fanout_calls_before = contraction.compiled_block_plan_fanout_matmul_calls
    repeated = contraction.apply(random_theta)
    assert contraction.compiled_block_plan_uses == 2
    assert contraction.compiled_block_plan_fanout_matmul_calls == (
        fanout_calls_before + contraction.compiled_block_plan_fanout_groups
    )
    for sector, expected in got.data.blocks.items():
        np.testing.assert_allclose(
            repeated.data.blocks[sector], expected, atol=1e-12, rtol=1e-12
        )

    # This window's equivalent composed map changes the accumulation order by
    # slightly more than the strict validation tolerance, so it must retain
    # the streamed plan. This is the guard against numerical regressions.
    randomized_theta = _randomized_block_tensor(theta, 29)
    fallback = problem.apply(randomized_theta)
    assert problem.sector_operator is None
    assert problem.sector_operator_disabled_reason == "validation_mismatch"
    fallback_input = qtn.Tensor(
        data=_array_with_blocks_like(
            problem.theta_input_data_template,
            randomized_theta.data.blocks,
        ),
        inds=problem.theta_input_inds,
        tags=randomized_theta.tags,
    )
    direct_right = _BlockPairContraction(
        optimizer,
        problem.right_contraction.left,
        problem.right_contraction.right_inds,
    ).apply(fallback_input)
    direct_out = _BlockPairContraction(
        optimizer,
        problem.left_contraction.left,
        problem.left_contraction.right_inds,
    ).apply(direct_right)
    if tuple(direct_out.inds) != tuple(randomized_theta.inds):
        direct_out = direct_out.transpose(*randomized_theta.inds)
    for sector, got_block in fallback.data.blocks.items():
        expected = direct_out.data.blocks.get(
            sector,
            np.zeros_like(got_block),
        )
        np.testing.assert_allclose(
            got_block, expected, atol=1e-12, rtol=1e-12
        )

    # A smaller right-first FH window validates the same bounded sector map,
    # then later applications take the cached dense sector operator.
    cached_theta = optimizer.two_site_theta(0)
    cached_problem, _ = optimizer._get_projected_problem(0, cached_theta)
    cached_random_theta = _randomized_block_tensor(cached_theta, 31)
    cached_reference = cached_problem.apply(cached_random_theta)
    assert cached_problem.sector_operator is not None
    assert cached_problem.sector_operator_bytes > 0
    assert cached_problem.sector_operator_block_count > 0
    operator_cached = cached_problem.apply(cached_random_theta)
    assert cached_problem.sector_operator_uses == 1
    for sector, expected in cached_reference.data.blocks.items():
        np.testing.assert_allclose(
            operator_cached.data.blocks[sector], expected, atol=1e-12, rtol=1e-12
        )


def test_symdmrg_sector_operator_default_bypasses_layout_build(monkeypatch):
    """The default streamed matvec does not pay experimental cache setup."""
    mapper = OneDMap(3, 2, mode="snake")
    edges = tuple(qtn.edges_2d_square(3, 2, cyclic=True))
    mpo = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1", "U1U1", edges, t=1.0, U=8.0
    ).to_mpo(mapper=mapper, compress=True, cutoff=1e-12)
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        6,
        bond_dim=4,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)] * 3),
        seed=3,
        dtype="complex128",
    )
    optimizer = pepsy.SymDMRG2(
        mpo,
        state,
        bond_dims=[4],
        cutoffs=[1e-10],
        compute_initial_energy=False,
    )
    theta = optimizer.two_site_theta(0)
    problem, _ = optimizer._get_projected_problem(0, theta)
    monkeypatch.setattr(
        problem,
        "_sector_operator_layout",
        lambda: pytest.fail("default matvec must not inspect the sector cache layout"),
    )

    result = problem.apply(_randomized_block_tensor(theta, 31))

    assert result.data.blocks
    assert problem.sector_operator is None
    assert problem.sector_operator_disabled_reason == "disabled"


def test_native_fermionic_compiled_plan_preserves_phases_and_dummy_modes():
    """Direct native contractions compile only with phase-stable metadata."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=4,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)] * 2),
        seed=7,
        dtype="complex128",
    )
    left, right = list(state.tn)[1:3]
    # Exercise an incoming lazy phase in addition to the tensors' dummy modes.
    right = _tensor_with_data(
        right,
        right.data.phase_sector(next(iter(right.data.blocks))),
    )
    contraction = _BlockPairContraction(None, left, right.inds)
    contraction.apply(right)  # Direct reference call that compiles the plan.
    assert contraction.compiled_block_plan_fermionic
    assert contraction.compiled_block_plan_fanout_groups == 0

    random_right = _randomized_block_tensor(right, 17)
    reference = _BlockPairContraction(None, left, right.inds).apply(random_right)
    got = contraction.apply(random_right)

    assert got.data.dummy_modes == reference.data.dummy_modes
    assert got.data.phases == reference.data.phases
    expected_data = reference.data.phase_sync()
    got_data = got.data.phase_sync()
    for sector, expected in expected_data.blocks.items():
        np.testing.assert_allclose(
            got_data.blocks[sector], expected, atol=1e-12, rtol=1e-12
        )

    # This pair makes the static left array larger, exercising Symmray's
    # opposite dual-index phase-flip branch during plan compilation.
    flip_left, flip_right = list(state.tn)[2:4]
    flip_right = _tensor_with_data(
        flip_right,
        flip_right.data.phase_sector(next(iter(flip_right.data.blocks))),
    )
    flip_plan = _BlockPairContraction(None, flip_left, flip_right.inds)
    flip_plan.apply(flip_right)
    random_flip_right = _randomized_block_tensor(flip_right, 19)
    flip_reference = _BlockPairContraction(
        None, flip_left, flip_right.inds
    ).apply(random_flip_right)
    flip_got = flip_plan.apply(random_flip_right)
    assert flip_got.data.dummy_modes == flip_reference.data.dummy_modes
    assert flip_got.data.phases == flip_reference.data.phases
    for sector, expected in flip_reference.data.phase_sync().blocks.items():
        np.testing.assert_allclose(
            flip_got.data.phase_sync().blocks[sector],
            expected,
            atol=1e-12,
            rtol=1e-12,
        )

    phase_changed = _tensor_with_data(
        random_right,
        random_right.data.phase_sector(next(iter(random_right.data.blocks))),
    )
    builds_before = contraction.compiled_block_plan_builds
    contraction.apply(phase_changed)
    assert contraction.compiled_block_plan_builds == builds_before + 1


@pytest.mark.parametrize("symmetry", ["Z2", "Z2Z2"])
def test_unified_spinful_fermion_supports_symmray_parity_symmetries(symmetry):
    """Spinful parity symmetries should use native charges and gates."""
    fermion = Fermion(spinful=True, symmetry=symmetry)

    assert fermion.model == (
        "fermi_hubbard" if symmetry == "Z2" else "fermi_hubbard_u1u1"
    )
    assert fermion.physical_sectors == default_physical_sectors(symmetry, 4)
    assert fermion.operator_charge("create_up") == (
        1 if symmetry == "Z2" else (0, 1)
    )
    assert fermion.operator_charge("create_down") == (
        1 if symmetry == "Z2" else (1, 0)
    )
    assert fermion.pair_charge == (0 if symmetry == "Z2" else (1, 1))
    assert fermion.pair_annihilation_charge == (
        0 if symmetry == "Z2" else (1, 1)
    )
    assert type(fermion.observable("number")).__name__.endswith("FermionicArray")
    assert type(fermion.hopping_gate(0.01, t=0.5)).__name__.endswith("FermionicArray")
    assert type(fermion.interaction_gate(0.01, U=2.0)).__name__.endswith("FermionicArray")


def test_z2z2_lattice_half_filling_keeps_flavor_occupations():
    """Z2Z2 site metadata must preserve up/down parity separately."""
    fermion = Fermion(spinful=True, symmetry="Z2Z2")

    setup = fermion.lattice_half_filling(2, 2)

    assert all(
        isinstance(occupation, tuple) and len(occupation) == 2
        for occupation in setup.occupations.values()
    )
    assert setup.target_charge == (0, 0)
    assert setup.site_charge((0, 0)) == (1, 0)


def test_unified_spinful_fermion_hamiltonian_keeps_explicit_mu_parameter():
    """Explicit spinful chemical potentials must reach Symmray terms."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    hamiltonian = fermion.hamiltonian(
        ((0, 1),), t=1.0, U=3.0, mu=(0.2, 0.4)
    )

    assert hamiltonian.parameters["mu"] == (0.2, 0.4)
    dense = hamiltonian.terms[(0, 1)].to_dense()
    assert dense[0, 0, 0, 0] == pytest.approx(0.0)
    assert dense[1, 0, 1, 0] == pytest.approx(-0.2)
    assert dense[2, 0, 2, 0] == pytest.approx(-0.4)
    assert dense[0, 1, 0, 1] == pytest.approx(-0.2)
    assert dense[0, 2, 0, 2] == pytest.approx(-0.4)
    assert dense[3, 0, 3, 0] == pytest.approx(3.0 - 0.2 - 0.4)
    assert dense[0, 3, 0, 3] == pytest.approx(3.0 - 0.2 - 0.4)


def test_fermion_exposes_explicit_native_operator_terms():
    """Named and generic APIs return the unexponentiated fermion terms."""
    fermion = Fermion(spinful=True, symmetry="U1U1")

    hopping = fermion.hopping_operator()
    reference_hopping = fermion.hamiltonian(
        ((0, 1),), t=1.7, U=0.0, mu=0.0, V=0.0
    ).terms[(0, 1)]
    np.testing.assert_allclose(
        hopping.to_dense(), -reference_hopping.to_dense() / 1.7
    )

    spin_up = fermion.hopping_operator(spin="up")
    explicit_spin_up = fermion.operator_term(
        [
            (1.0, ((0, "create_up"), (1, "annihilate_up"))),
            (1.0, ((1, "create_up"), (0, "annihilate_up"))),
        ],
        sites=(0, 1),
    )
    np.testing.assert_allclose(spin_up.to_dense(), explicit_spin_up.to_dense())

    np.testing.assert_allclose(
        np.diag(fermion.interaction_operator().to_dense()),
        (0.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(
        np.diag(fermion.chemical_potential_operator().to_dense()),
        (0.0, 1.0, 1.0, 2.0),
    )
    np.testing.assert_allclose(
        np.diag(fermion.onsite_term(0, U=3.0, mu=0.4).to_dense()),
        (0.0, -0.4, -0.4, 3.0 - 0.8),
    )

    density = fermion.density_operator()
    reference_density = fermion.hamiltonian(
        ((0, 1),), t=0.0, U=0.0, mu=0.0, V=0.6
    ).terms[(0, 1)]
    np.testing.assert_allclose(
        0.6 * density.to_dense(), reference_density.to_dense()
    )

    forward_up = fermion.operator_term(
        [(1.0, ((0, "create_up"), (1, "annihilate_up")))],
        sites=(0, 1),
        add_hc=True,
    )
    np.testing.assert_allclose(
        forward_up.to_dense(), fermion.hopping_operator(spin="up").to_dense()
    )

    eta_pair = fermion.eta_pair_operator()
    explicit_eta_pair = fermion.operator_term(
        [
            (1.0, ((0, "pair_create"), (1, "pair_annihilate"))),
            (1.0, ((1, "pair_create"), (0, "pair_annihilate"))),
        ],
        sites=(0, 1),
        charge=(0, 0),
    )
    assert eta_pair.charge == (0, 0)
    np.testing.assert_allclose(eta_pair.to_dense(), explicit_eta_pair.to_dense())

    with pytest.raises(ValueError, match="self-conjugate operator charge"):
        fermion.operator_term(
            [(1.0, ((0, "create_up"),))],
            sites=(0,),
            add_hc=True,
        )

    with pytest.raises(ValueError, match="require spinful fermions"):
        Fermion(spinful=False, symmetry="U1").eta_pair_operator()

    explicit_terms = {
        (0, 1): -1.7 * hopping,
        0: 3.0 * fermion.interaction_operator(),
        1: 3.0 * fermion.interaction_operator(),
    }
    explicit_hamiltonian = fermion.hamiltonian(explicit_terms)
    assert explicit_hamiltonian.explicit_terms
    assert set(explicit_hamiltonian.terms) == {(0, 1), (0,), (1,)}
    explicit_mpo = explicit_hamiltonian.to_mpo(L=2, compress=False)
    reference_mpo = fermion.hamiltonian(
        ((0, 1),), t=1.7, U=3.0, mu=0.0, V=0.0
    ).to_mpo(L=2, compress=False)
    np.testing.assert_allclose(
        _mpo_to_dense_matrix(explicit_mpo, 2),
        _mpo_to_dense_matrix(reference_mpo, 2),
    )


def test_fermion_spin_operator_algebra_and_native_correlators():
    """Native spin helpers should match local algebra and stay symmetric."""
    fermion = Fermion(spinful=True, symmetry="U1")
    sx = fermion.dense_operator("sx")
    sy = fermion.dense_operator("sy")
    sz = fermion.dense_operator("sz")

    np.testing.assert_allclose(sx @ sy - sy @ sx, 1j * sz)
    np.testing.assert_allclose(
        (sx @ sx + sy @ sy + sz @ sz)[1:3, 1:3],
        0.75 * np.eye(2),
    )
    np.testing.assert_allclose(
        fermion.observable("sx").to_dense(),
        sx,
    )
    np.testing.assert_allclose(
        fermion.observable("sy").to_dense(),
        sy,
    )
    assert fermion.spin_x_operator() is fermion.observable("sx")
    assert fermion.spin_y_operator() is fermion.observable("sy")
    assert fermion.spin_z_operator() is fermion.observable("sz")

    for operator in (
        fermion.spin_z_correlator(),
        fermion.spin_x_correlator(),
        fermion.spin_y_correlator(),
        fermion.xy_exchange_operator(),
        fermion.heisenberg_operator(),
    ):
        assert type(operator).__name__ == "U1FermionicArray"
        assert operator.charge == 0


@pytest.mark.parametrize("symmetry", ["U1", "Z2"])
def test_fermion_spin_gates_are_native_unitary_and_parameterized(symmetry):
    """Spin gates should exponentiate natively for total-charge symmetries."""
    fermion = Fermion(spinful=True, symmetry=symmetry)

    def as_matrix(operator):
        dense = np.asarray(operator.to_dense())
        dim = int(np.sqrt(dense.size))
        return dense.reshape(dim, dim)

    gates = (
        fermion.sx_gate({3: 0.13}, site=3),
        fermion.sy_gate(lambda site: 0.17 + 0.01 * site, site=3),
        fermion.sz_gate(0.11),
        fermion.sxx_gate(0.07),
        fermion.syy_gate({(3, 4): 0.09}, edge=(3, 4)),
        fermion.szz_gate(0.05),
        fermion.xy_gate(0.03),
        fermion.heisenberg_gate(0.02),
    )
    for gate_ in gates:
        matrix = as_matrix(gate_)
        np.testing.assert_allclose(
            matrix.conj().T @ matrix,
            np.eye(matrix.shape[0]),
            atol=1e-12,
        )

    imaginary = as_matrix(fermion.syy_gate(0.2, imaginary=True))
    np.testing.assert_allclose(imaginary, imaginary.conj().T, atol=1e-12)
    assert np.all(np.abs(np.linalg.eigvalsh(imaginary)) > 0.0)


def test_fermion_spin_flip_restrictions_are_explicit_under_u1u1():
    """Separate spin charges must reject inhomogeneous spin-flip gates."""
    fermion = Fermion(spinful=True, symmetry="U1U1")

    with pytest.raises(ValueError, match="symmetry='U1' or 'Z2'"):
        fermion.observable("sx")
    with pytest.raises(ValueError, match="symmetry='U1' or 'Z2'"):
        fermion.observable("sy")
    with pytest.raises(ValueError, match="symmetry='U1' or 'Z2'"):
        fermion.syy_gate(0.1)

    assert fermion.observable("sz").charge == (0, 0)
    assert fermion.szz_gate(0.1).charge == (0, 0)
    assert fermion.xy_gate(0.1).charge == (0, 0)
    assert fermion.heisenberg_gate(0.1).charge == (0, 0)
    assert fermion.observable("s_plus").charge != (0, 0)
    with pytest.raises(ValueError, match="charge-neutral"):
        fermion.operator_gate("s_plus", 0.1)


def test_fermion_spin_gates_preserve_torch_backend():
    """The generic operator gate should use the configured block backend."""
    torch = pytest.importorskip("torch")
    fermion = Fermion(
        spinful=True,
        symmetry="U1",
        to_backend=pepsy.backend_torch(dtype=torch.complex128),
    )
    gate_ = fermion.sy_gate(0.1)
    assert gate_.backend == "torch"


def test_fermion_fields_and_pairing_preserve_jax_backend():
    """The extended stream terms keep their configured JAX block backend."""
    jnp = pytest.importorskip("jax.numpy")

    spinful = Fermion(
        spinful=True,
        symmetry="U1",
        to_backend=pepsy.backend_jax(dtype=jnp.complex64),
    )
    spinful_ham = spinful.hamiltonian(
        ((0, 1),), t=1.0, U=2.0, field_z=0.2
    )
    assert all(term.backend == "jax" for term in spinful_ham.terms.values())

    spinless = Fermion(
        spinful=False,
        symmetry="Z2",
        to_backend=pepsy.backend_jax(dtype=jnp.complex64),
    )
    pairing_ham = spinless.hamiltonian(((0, 1),), t=1.0, pairing=0.2)
    assert all(term.backend == "jax" for term in pairing_ham.terms.values())


def test_mps_energy_uses_explicit_fermion_terms_natively():
    """An explicit one- plus two-site SymHamiltonian stays on native MPS terms."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    ham = fermion.hamiltonian(
        {
            (0, 1): -fermion.hopping_operator(),
            0: 2.0 * fermion.interaction_operator(),
            1: 2.0 * fermion.interaction_operator(),
        }
    )
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=2,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)]),
        seed=17,
        dtype="complex128",
    )

    optimizer = pepsy.MpsEnergyOptimizer(
        state,
        terms=ham,
        energy_per_site=False,
        real=False,
    )

    assert optimizer._can_use_native_local_terms(ham, state)
    assert np.isfinite(np.real(optimizer.energy().energy))


def test_fermion_explicit_coordinate_terms_preserve_peps_locations():
    """Coordinate-site terms remain usable by PEPS and mapped MPO workflows."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    edges = (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
    )
    sites = tuple(sorted({site for edge in edges for site in edge}))
    terms = {edge: -fermion.hopping_operator() for edge in edges}
    terms |= {site: fermion.onsite_term(site, U=3.0, mu=0.2) for site in sites}

    hamiltonian = fermion.hamiltonian(terms)

    assert set(hamiltonian.terms) == set(edges) | set(sites)
    assert hamiltonian.terms[(0, 0)].shape == (4, 4)
    assert hamiltonian.terms[edges[0]].shape == (4, 4, 4, 4)

    mpo = hamiltonian.to_mpo(
        mapper=OneDMap(2, 2, mode="snake"),
        compress=False,
    )
    assert mpo.L == 4


def test_symhamiltonian_and_fermion_build_pepo_accept_mapper():
    """Hamiltonian and model shorthands share the native PEPO route."""
    pytest.importorskip("symmray")
    fermion = Fermion(spinful=True, symmetry="U1U1")
    left = (0, 0)
    right = (1, 1)
    term = fermion.operator_term(
        [(1.0, ((left, "double"), (right, "annihilate_up")))],
        sites=(left, right),
        label="symhamiltonian_pepo",
    )
    hamiltonian = fermion.hamiltonian({(left, right): term})
    mapper = OneDMap(2, 2, mode="snake-row-major")

    pepo_from_hamiltonian = hamiltonian.to_pepo(
        2,
        2,
        mapper=mapper,
        max_bond=16,
        compress=False,
    )
    pepo_from_model = fermion.build_pepo(
        hamiltonian=hamiltonian,
        Lx=2,
        Ly=2,
        mapper=mapper,
        max_bond=16,
        compress=False,
    )

    for pepo in (pepo_from_hamiltonian, pepo_from_model):
        assert pepo.Lx == 2
        assert pepo.Ly == 2
        assert list(pepo)[-1].data.charge == term.charge
        assert all(
            type(tensor.data).__name__.endswith("FermionicArray")
            for tensor in pepo
        )


@pytest.mark.parametrize("symmetry", ["U1U1", "U1", "Z2"])
def test_mixed_native_charges_return_explicit_mpo_and_pepo_sectors(symmetry):
    """Mixed native operators decompose into homogeneous charge sectors."""
    pytest.importorskip("symmray")
    fermion = Fermion(spinful=True, symmetry=symmetry)
    left = (0, 0)
    middle = (0, 1)
    right = (1, 1)
    neutral = fermion.hopping_operator()
    charged = fermion.operator_term(
        [(1.0, ((middle, "double"), (right, "annihilate_up")))],
        sites=(middle, right),
        label="mixed_charge_sector",
    )
    hamiltonian = fermion.hamiltonian(
        {
            (left, middle): neutral,
            (middle, right): charged,
        }
    )
    mapper = OneDMap(2, 2, mode="snake-row-major")

    mpo_sectors = hamiltonian.to_mpo(
        mapper=mapper,
        fermionic=True,
        charge_sectors=True,
        compress=False,
    )
    pepo_sectors = hamiltonian.to_pepo(
        2,
        2,
        mapper=mapper,
        fermionic=True,
        charge_sectors=True,
        compress=False,
    )

    assert set(mpo_sectors) == {fermion.zero_charge, charged.charge}
    assert set(pepo_sectors) == {fermion.zero_charge, charged.charge}
    for charge, mpo in mpo_sectors.items():
        assert all(type(tensor.data).__name__.endswith("FermionicArray") for tensor in mpo)
        assert list(mpo)[-1].data.charge == charge
    for pepo in pepo_sectors.values():
        assert pepo.Lx == 2
        assert pepo.Ly == 2
        assert all(
            type(tensor.data).__name__.endswith("FermionicArray")
            for tensor in pepo
        )


@pytest.mark.parametrize("symmetry", ["U1U1", "U1", "Z2"])
def test_fermion_to_pepo_builds_native_coordinate_terms(symmetry):
    """Fermion.to_pepo preserves native grading for supported symmetries."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    left = (0, 1)
    right = (2, 2)
    hopping = fermion.operator_term(
        [(1.0, ((left, "create_up"), (right, "annihilate_up")))],
        sites=(left, right),
        add_hc=True,
    )

    pepo = fermion.to_pepo(
        {(left, right): hopping},
        Lx=3,
        Ly=3,
        max_bond=16,
        compress=False,
    )

    assert pepo.Lx == 3
    assert pepo.Ly == 3
    assert set(pepo.outer_inds()) == {
        f"k{x},{y}" for x in range(3) for y in range(3)
    } | {
        f"b{x},{y}" for x in range(3) for y in range(3)
    }
    assert all(type(tensor.data).__name__.endswith("FermionicArray") for tensor in pepo)


@pytest.mark.parametrize("symmetry", ["U1U1", "U1", "Z2"])
def test_single_native_pepo_term_uses_local_operator_schmidt_rank(symmetry):
    """A disposable one-/two-site PEPO has no generic channel inflation."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    mapper = OneDMap(2, 1, mode="snake")

    hopping = fermion.to_pepo(
        {((0, 0), (1, 0)): fermion.hopping_operator()},
        Lx=2,
        Ly=1,
        mapper=mapper,
        max_bond=None,
        cutoff=0.0,
        compress=False,
    )
    onsite = fermion.to_pepo(
        {((0, 0),): fermion.onsite_term((0, 0), U=8.0)},
        Lx=2,
        Ly=1,
        mapper=mapper,
        max_bond=None,
        cutoff=0.0,
        compress=False,
    )

    assert hopping.max_bond() == 4
    assert onsite.max_bond() == 1
    assert hopping.pepsy_compression_report["operator_schmidt_bond"] == 4
    assert onsite.pepsy_compression_report["operator_schmidt_bond"] == 1
    assert hopping.pepsy_compression_report["direct_local"] is True
    assert onsite.pepsy_compression_report["direct_local"] is True
    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in (*hopping, *onsite)
    )


def test_fermion_to_pepo_native_result_supports_reverse_simple_update():
    """Native PEPO output can take an adjoint gate through operator SU."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    left = (0, 0)
    right = (0, 1)
    term = fermion.operator_term(
        [(1.0, ((left, "create_up"), (right, "annihilate_up")))],
        sites=(left, right),
        add_hc=True,
    )
    pepo = fermion.to_pepo(
        {(left, right): term},
        Lx=2,
        Ly=2,
        max_bond=16,
        compress=False,
    )
    gauges = {}
    pepo.gauge_all_simple_(gauges=gauges, progbar=False)

    out = gate_simple(
        pepo,
        fermion.hopping_gate(0.001, t=1.0).H,
        where=(left, right),
        gauges=gauges,
        max_bond=16,
        cutoff=1e-10,
        contract="split",
        renorm=False,
        inplace=False,
    )

    assert out.max_bond() <= 16
    assert all(type(tensor.data).__name__.endswith("FermionicArray") for tensor in out)
    assert len(gauges) > 0


def test_native_fermion_pepo_reverse_simple_update_matches_state_evolution():
    """Graded operator SU must realize G.H @ O @ G, not a dense sandwich."""
    fermion = Fermion(spinful=True, symmetry="U1")
    left = (0, 0)
    right = (1, 0)
    where = (left, right)
    mapper = OneDMap(2, 2, mode="snake")

    state = pepsy.ps_to_peps(
        (2, 2),
        fermion=fermion,
        occupations={(x, y): 1 for x in range(2) for y in range(2)},
        seed=3,
        dtype="complex128",
        cyclic=False,
    )
    for x in range(2):
        for y in range(2):
            sign = -1 if (x + y) % 2 == 0 else 1
            tensor = state[x, y]
            (sector, block), = tensor.data.blocks.items()
            tensor.data.blocks[sector] = (
                np.asarray([1.0, sign], dtype=np.complex128)
                .reshape(block.shape)
                / np.sqrt(2.0)
            )

    operator = fermion.to_pepo(
        {where: fermion.eta_pair_operator()},
        Lx=2,
        Ly=2,
        mapper=mapper,
        max_bond=64,
        cutoff=0.0,
        compress=False,
        cyclic=False,
    )
    forward_gate = fermion.hopping_gate(0.15, t=1.0)
    evolved_state = gate(
        state,
        forward_gate,
        where=where,
        contract="split",
        max_bond=64,
        cutoff=0.0,
        inplace=False,
    )

    def expectation(psi, pepo):
        applied = pepo.apply(psi, contract=True, compress=False)
        return complex(np.asarray((psi.H & applied).contract(all)).item())

    reference = expectation(evolved_state, operator)
    assert abs(reference) > 1.0e-3

    explicit_operator = operator.copy()
    explicit_gauges = {}
    explicit_operator.gauge_all_simple_(
        gauges=explicit_gauges, progbar=False
    )
    explicit_operator = gate_simple(
        explicit_operator,
        forward_gate.H,
        where=where,
        which="upper",
        gauges=explicit_gauges,
        renorm=False,
        smudge=1.0e-12,
        max_bond=64,
        cutoff=0.0,
        contract="split",
        inplace=False,
    )
    gate_simple(
        explicit_operator,
        forward_gate.T,
        where=where,
        which="lower",
        gauges=explicit_gauges,
        renorm=False,
        smudge=1.0e-12,
        max_bond=64,
        cutoff=0.0,
        contract="split",
        inplace=True,
    )
    explicit_operator.gauge_simple_insert(explicit_gauges)

    gauges = {}
    operator.gauge_all_simple_(gauges=gauges, progbar=False)
    reverse_evolved = gate_simple(
        operator,
        forward_gate.H,
        where=where,
        gauges=gauges,
        renorm=False,
        smudge=1.0e-12,
        max_bond=64,
        cutoff=0.0,
        contract="split",
        inplace=False,
    )
    reverse_evolved.gauge_simple_insert(gauges)

    np.testing.assert_allclose(
        expectation(state, reverse_evolved),
        reference,
        atol=1.0e-8,
        rtol=1.0e-8,
    )
    np.testing.assert_allclose(
        expectation(state, explicit_operator),
        reference,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


@pytest.mark.parametrize(
    ("symmetry", "gate_name", "operator_name"),
    [
        ("U1U1", "hopping", "eta"),
        ("U1U1", "heisenberg", "number_up"),
        ("U1", "hopping", "eta"),
        ("U1", "heisenberg", "number_up"),
        ("U1", "sxx", "number_up"),
        ("Z2", "hopping", "eta"),
        ("Z2", "heisenberg", "number_up"),
        ("Z2", "sxx", "number_up"),
    ],
)
def test_native_fermion_pepo_reverse_simple_update_unitary_families(
    symmetry, gate_name, operator_name
):
    """Several native unitaries must match the exact graded local sandwich."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    where = ((0, 0), (1, 0))
    mapper = OneDMap(2, 1, mode="snake")
    if operator_name == "eta":
        operator = fermion.eta_pair_operator()
    else:
        operator = fermion.operator_term(
            [(1.0, ((where[0], "number_up"),))],
            sites=where,
        )

    if gate_name == "hopping":
        forward_gate = fermion.hopping_gate(0.15, t=1.0)
    elif gate_name == "heisenberg":
        forward_gate = fermion.heisenberg_gate(0.11)
    else:
        forward_gate = fermion.sxx_gate(0.13)

    gate_matrix = forward_gate.fuse((0, 1), (2, 3))
    operator_matrix = operator.fuse((0, 1), (2, 3))
    exact_local = (
        gate_matrix.H @ operator_matrix @ gate_matrix
    ).reshape((4, 4, 4, 4))
    pepo_opts = {
        "Lx": 2,
        "Ly": 1,
        "mapper": mapper,
        "max_bond": 256,
        "cutoff": 0.0,
        "compress": False,
        "cyclic": False,
    }
    evolved = fermion.to_pepo({where: operator}, **pepo_opts)
    exact = fermion.to_pepo({where: exact_local}, **pepo_opts)

    gauges = {}
    evolved.gauge_all_simple_(gauges=gauges, progbar=False)
    gate_simple(
        evolved,
        forward_gate.H,
        where=where,
        gauges=gauges,
        renorm=False,
        smudge=1.0e-12,
        max_bond=256,
        cutoff=0.0,
        contract="split",
        inplace=True,
    )
    evolved.gauge_simple_insert(gauges)

    def hilbert_schmidt(left, right):
        return complex(
            np.asarray((left.H & right).contract(all)).item()
        )

    evolved_norm = hilbert_schmidt(evolved, evolved)
    exact_norm = hilbert_schmidt(exact, exact)
    overlap = hilbert_schmidt(evolved, exact)
    relative_distance = abs(
        evolved_norm + exact_norm - 2.0 * overlap.real
    ) / max(abs(evolved_norm), abs(exact_norm))
    assert relative_distance < 1.0e-10


@pytest.mark.parametrize("symmetry", ["U1U1", "U1", "Z2"])
def test_fermion_to_pepo_supports_charged_odd_native_terms(symmetry):
    """Charged odd terms retain their native charge and dummy mode."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    left = (0, 1)
    right = (2, 2)
    charged = fermion.operator_term(
        [(1.0, ((left, "double"), (right, "annihilate_up")))],
        sites=(left, right),
        label="charged_pepo",
    )

    pepo = fermion.to_pepo(
        {(left, right): charged},
        Lx=3,
        Ly=3,
        max_bond=16,
        compress=False,
    )
    tensors = list(pepo)

    assert charged.charge != fermion.zero_charge
    assert tensors[-1].data.charge == charged.charge
    assert tensors[-1].data.label is not None
    assert tensors[-1].data.dummy_modes
    assert all(type(tensor.data).__name__.endswith("FermionicArray") for tensor in tensors)
    with pytest.raises(ValueError, match="fermionic=True"):
        fermion.to_mpo(
            {(1, 8): charged},
            L=9,
            fermionic=False,
        )


@pytest.mark.parametrize("symmetry", ["U1U1", "U1", "Z2"])
def test_charged_native_pepo_supports_reverse_simple_update(symmetry):
    """Neutral reverse evolution preserves a charged PEPO operator sector."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    left = (0, 0)
    right = (0, 1)
    charged = fermion.operator_term(
        [(1.0, ((left, "double"), (right, "annihilate_up")))],
        sites=(left, right),
        label="charged_pepo",
    )
    pepo = fermion.to_pepo(
        {(left, right): charged},
        Lx=2,
        Ly=2,
        max_bond=16,
        compress=False,
    )
    gauges = {}
    pepo.gauge_all_simple_(gauges=gauges, progbar=False)

    out = gate_simple(
        pepo,
        fermion.hopping_gate(0.001, t=1.0).H,
        where=(left, right),
        gauges=gauges,
        max_bond=16,
        cutoff=1e-10,
        contract="split",
        renorm=False,
        inplace=False,
    )

    tensors = list(out)
    assert tensors[-1].data.charge == charged.charge
    assert tensors[-1].data.dummy_modes
    assert out.max_bond() <= 16
    assert len(gauges) > 0


def test_unified_fermion_peps_energy_accepts_boundary_chi():
    """The shared Fermion Hamiltonian can use SymPEPS boundary measurement."""
    peps = SymPEPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        2,
        bond_dim=2,
        site_charge=site_charge_from_occupations(
            {
                (0, 0): (1, 0),
                (0, 1): (0, 1),
                (1, 0): (1, 0),
                (1, 1): (0, 1),
            }
        ),
        seed=17,
        dtype="complex128",
    )
    fermion = Fermion(symmetry="U1U1")
    ham = fermion.hamiltonian(peps.edges, t=1.0, U=2.0, mu=(0.1, 0.2))

    exact = peps.energy(ham)
    boundary = peps.energy(ham, chi=8)

    assert np.isfinite(exact)
    assert boundary == pytest.approx(exact, rel=1e-7, abs=1e-7)


def test_unified_fermion_gates_and_hamiltonian_preserve_backend():
    """The high-level Fermion helper applies its backend to every block."""
    torch = pytest.importorskip("torch")
    fermion = Fermion(
        symmetry="U1U1",
        to_backend=pepsy.backend_torch(dtype=torch.complex128),
    )

    values = (
        fermion.onsite_gate(0.01, U=8.0),
        fermion.hopping_gate(0.01, t=1.0),
        fermion.density_gate(0.01, V=0.2),
        fermion.hamiltonian(((0, 1),), t=1.0, U=8.0).terms[(0, 1)],
    )
    assert all(value.backend == "torch" for value in values)


def test_spinful_interaction_gate_has_exact_doublon_phase():
    """The onsite gate should only phase the doubly occupied basis state."""
    theta = 0.2 * 3.0
    gate = Fermion(spinful=True, symmetry="U1U1").gate(
        "interaction",
        0.2,
        U=3.0,
    )
    expected = np.diag([1.0, 1.0, 1.0, np.exp(-1j * theta)])
    np.testing.assert_allclose(gate.to_dense(), expected)
    imaginary = Fermion(spinful=True, symmetry="U1U1").interaction_gate(
        0.2,
        U=3.0,
        imaginary=True,
    )
    np.testing.assert_allclose(
        imaginary.to_dense(),
        np.diag([1.0, 1.0, 1.0, np.exp(-theta)]),
    )


def test_fermion_generic_exponential_matches_named_interaction_gate():
    """Generic neutral monomials should share the native exponentiator."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    generic = fermion.exponential(
        [(3.0, ((0, "number_up"), (0, "number_down")))],
        0.2,
        sites=(0,),
    )

    np.testing.assert_allclose(
        generic.to_dense(),
        fermion.interaction_gate(0.2, U=3.0).to_dense(),
    )
    assert type(generic).__name__ == "U1U1FermionicArray"


def test_fermion_hopping_gate_matches_native_hamiltonian_exponential():
    """Native hopping imaginary time must lower the matching term energy."""
    fermion = Fermion(spinful=True, symmetry="U1U1")
    ham = fermion.hamiltonian({(0, 1): -fermion.hopping_operator()})
    named = fermion.hopping_gate(0.01, t=1.0, imaginary=True)
    reference = ham.trotter_gates(0.01, imaginary=True)[0][0]

    fourth = ham.trotter_gates(0.01, imaginary=True, order=4)
    assert fourth.order == 4
    assert len(fourth) == 3 * len(ham.trotter_gates(0.01, imaginary=True, order=2))

    np.testing.assert_allclose(named.to_dense(), reference.to_dense())

    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=1,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)]),
        seed=3,
        dtype="complex128",
    )
    before = pepsy.MpsEnergyOptimizer(
        state.tn,
        terms=ham,
        normalized=True,
        energy_per_site=False,
        real=True,
    ).energy().energy
    out = pepsy.MpsOptimizer(
        state.tn.copy(),
        [(named, (0, 1))],
        chi=8,
        mode="mpo",
        inplace=True,
    ).run(
        progbar=False,
        cutoff=0.0,
        non_unitary=True,
        normalize_final=False,
    )
    after = pepsy.MpsEnergyOptimizer(
        out,
        terms=ham,
        normalized=True,
        energy_per_site=False,
        real=True,
    ).energy().energy

    assert before == pytest.approx(0.0)
    assert after < -1.0e-4


def _native_mps_dense_amplitudes(state):
    """Read amplitudes from a dense fixed-charge native MPS contraction."""
    dense = state.to_dense()
    configs = []
    for config_map in dense.indices[0]._subinfo.extents.values():
        for config, multiplicity in config_map.items():
            configs.extend([config] * multiplicity)
    values = next(iter(dense.blocks.values()))[:, 0]
    assert len(configs) == len(values)
    return dict(zip(configs, np.asarray(values)))


@pytest.mark.parametrize(
    ("spinful", "initial", "moved"),
    [
        (False, (1, 1, 0), (0, 1, 1)),
        (True, ((1, 0), (0, 1), (0, 0)), ((0, 0), (0, 1), (1, 0))),
    ],
)
@pytest.mark.parametrize("imaginary", [False, True])
def test_native_hopping_gate_has_correct_long_range_parity_sign(
    spinful, initial, moved, imaginary
):
    """Native gates retain the Jordan-Wigner sign across an occupied site."""
    fermion = Fermion(
        spinful=spinful,
        symmetry="U1U1" if spinful else "U1",
    )
    dt = 0.13
    state = pepsy.ps_to_mps(
        len(initial),
        fermion=fermion,
        occupations=initial,
        seed=123,
        dtype="complex128",
    )
    out = pepsy.MpsOptimizer(
        state,
        [(fermion.hopping_gate(dt, t=1.3, imaginary=imaginary), (0, 2))],
        chi=4,
        mode="mpo",
        inplace=True,
    ).run(
        progbar=False,
        cutoff=1e-12,
        non_unitary=imaginary,
        normalize_every=False,
        normalize_final=False,
    )
    amplitudes = _native_mps_dense_amplitudes(out)
    ratio = amplitudes[tuple(moved)] / amplitudes[tuple(initial)]
    intermediate_parity = sum(
        sum(charge) if isinstance(charge, tuple) else charge
        for charge in initial[1:-1]
    ) % 2
    sign = (-1) ** intermediate_parity
    angle = 1.3 * dt
    expected = sign * (np.tanh(angle) if imaginary else 1j * np.tan(angle))
    np.testing.assert_allclose(ratio, expected, atol=1e-11, rtol=1e-11)


def test_fermion_parameter_generators_preserve_torch_autodiff():
    """Autoray-style parameter generators must preserve a Torch gradient."""
    torch = pytest.importorskip("torch")
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    gate = fermion_interaction_param_gen((theta,), symmetry="U1U1")
    real_block_sum = sum(block.real.sum() for block in gate.blocks.values())
    gradient = torch.autograd.grad(real_block_sum, theta)[0]
    assert gradient is not None
    assert torch.isfinite(gradient)

    density = fermion_density_param_gen((theta,), symmetry="U1")
    density_sum = sum(block.real.sum() for block in density.blocks.values())
    density_gradient = torch.autograd.grad(density_sum, theta)[0]
    assert torch.isfinite(density_gradient)

    hopping = fermion_hopping_param_gen((theta,), symmetry="U1")
    hopping_sum = sum(block.real.sum() for block in hopping.blocks.values())
    hopping_gradient = torch.autograd.grad(hopping_sum, theta)[0]
    assert torch.isfinite(hopping_gradient)

    spinful_hopping = fermion_hopping_param_gen(
        (theta,),
        spinful=True,
        symmetry="U1U1",
    )
    spinful_hopping_sum = sum(
        block.real.sum() for block in spinful_hopping.blocks.values()
    )
    spinful_hopping_gradient = torch.autograd.grad(
        spinful_hopping_sum,
        theta,
        retain_graph=True,
    )[0]
    assert torch.isfinite(spinful_hopping_gradient)


def test_fermion_parameter_generators_preserve_jax_autodiff():
    """All advertised native parameter generators should trace through JAX."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)

    generators = (
        ("interaction", lambda x: fermion_interaction_param_gen((x,), symmetry="U1U1")),
        ("density", lambda x: fermion_density_param_gen((x,), symmetry="U1")),
        ("hopping", lambda x: fermion_hopping_param_gen((x,), symmetry="U1")),
        (
            "spinful-hopping",
            lambda x: fermion_hopping_param_gen(
                (x,), spinful=True, symmetry="U1U1"
            ),
        ),
    )

    for _name, generator in generators:
        def loss(theta, generator=generator):
            gate = generator(theta)
            return sum(block.real.sum() for block in gate.blocks.values())

        gradient = jax.grad(loss)(jnp.asarray(0.2, dtype=jnp.float64))
        assert np.isfinite(float(gradient))


def test_symmray_block_summary_and_schematic_for_z2_gate():
    """Symmray gate helpers should expose and draw block-sector structure."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=1,
        dtype="complex128",
    )
    rz_gate = state.operator_from_dense(pepsy.rz(0.1), charge=0, sites=1)

    summary = symmray_block_summary(rz_gate)

    assert summary["shape"] == (2, 2)
    assert summary["num_blocks"] == 2
    assert summary["stored_size"] == 2
    assert summary["dense_size"] == 4
    assert [block["sector"] for block in summary["blocks"]] == [(0, 0), (1, 1)]
    assert [block["shape"] for block in summary["blocks"]] == [(1, 1), (1, 1)]
    assert summary["indices"][0]["direction"] == "out"
    assert summary["indices"][1]["direction"] == "in"

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_blocks(
        rz_gate,
        title="Z2 RZ gate",
        return_summary=True,
    )
    assert drawing is not None
    assert drawn_summary["blocks"] == summary["blocks"]


def test_symmray_mps_summary_and_schematic_for_z2_chain():
    """Symmray MPS drawings should expose site blocks and bond-sector metadata."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=2,
        dtype="complex128",
    )

    summary = symmray_mps_summary(state.tn)

    assert summary["num_sites"] == 4
    assert summary["max_bond_dim"] == 2
    assert summary["max_bond_sectors"] == 2
    assert summary["total_stored_size"] == 12
    assert summary["total_dense_size"] == 24
    assert summary["charge_total"] == summary["total_charge"]
    assert summary["Q_total"] == summary["total_parity"]
    assert summary["tensors"][0]["site"] == 0
    assert summary["tensors"][0]["physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["physical"]["direction"] in {"in", "out"}
    assert summary["tensors"][1]["left_bond"]["between"] == (0, 1)
    assert summary["bonds"][0]["chargemap"] == {0: 1, 1: 1}
    assert summary["bonds"][0]["left_direction"] in {"in", "out"}
    assert summary["bonds"][0]["right_direction"] in {"in", "out"}

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_mps(
        state.tn,
        title="Z2 MPS",
        return_summary=True,
    )
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawn_summary["bonds"] == summary["bonds"]

    compat_drawing, compat_summary = draw_symmray_peps(
        state.tn,
        return_summary=True,
    )
    assert hasattr(compat_drawing, "fig")
    assert compat_summary["bonds"] == summary["bonds"]


def test_symmray_mpo_summary_and_schematic_for_z2_operator():
    """Symmray MPO drawings should expose upper/lower physical legs."""
    ham = SymHamiltonian.from_edges(
        "tfim",
        "Z2",
        [(0, 1)],
        jx=-1.0,
        hz=-0.5,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    summary = symmray_mpo_summary(mpo)

    assert summary["num_sites"] == 2
    assert summary["symmetry"] == "Z2"
    assert summary["fermionic_ordering"]["network_kind"] == "mpo"
    assert summary["max_bond_dim"] == 5
    assert summary["max_bond_sectors"] == 2
    assert summary["tensors"][0]["upper_ind"] == mpo.upper_ind(0)
    assert summary["tensors"][0]["lower_ind"] == mpo.lower_ind(0)
    assert summary["tensors"][0]["upper_physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["lower_physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["lower_physical"]["direction"] in {"in", "out"}
    assert summary["bonds"][0]["between"] == (0, 1)
    assert summary["bonds"][0]["left_direction"] in {"in", "out"}
    assert summary["bonds"][0]["right_direction"] in {"in", "out"}

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_mpo(
        mpo,
        title="Z2 MPO",
        show_phys_labels=True,
        return_summary=True,
    )
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawn_summary["bonds"] == summary["bonds"]

    compat_drawing, compat_summary = draw_symmray_peps(mpo, return_summary=True)
    assert hasattr(compat_drawing, "fig")
    assert compat_summary["bonds"] == summary["bonds"]


def test_symmray_mps_mpo_schematics_accept_onedmap_layout():
    """1D Symmray chain drawings should optionally render on a OneDMap grid."""
    pytest.importorskip("matplotlib")
    mapper = OneDMap(2, 2, mode="snake")
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=20,
        dtype="complex128",
    )

    mps_drawing, mps_summary = draw_symmray_mps(
        state.tn,
        mapper=mapper,
        max_sites=3,
        show_bond_labels=True,
        show_phys_labels=True,
        show_diagnostics=True,
        return_summary=True,
    )

    assert hasattr(mps_drawing, "fig")
    assert mps_summary["num_sites"] == 4
    assert mps_drawing.ax.get_aspect() == 1.0
    assert any("+1 sites hidden" in text.get_text() for text in mps_drawing.ax.texts)

    ham = SymHamiltonian.from_edges(
        "tfim",
        "Z2",
        [(0, 1), (1, 2), (2, 3)],
        jx=-1.0,
        hz=-0.5,
    )
    mpo = ham.to_mpo(L=4, compress=False)

    mpo_drawing, mpo_summary = draw_symmray_mpo(
        mpo,
        mapper=mapper,
        show_bond_labels=True,
        show_phys_labels=True,
        show_diagnostics=True,
        return_summary=True,
    )
    compat_drawing = draw_symmray_peps(mpo, mapper=mapper)

    assert hasattr(mpo_drawing, "fig")
    assert hasattr(compat_drawing, "fig")
    assert mpo_summary["num_sites"] == 4
    assert mpo_drawing.ax.get_aspect() == 1.0

    with pytest.raises(ValueError, match="does not match network length"):
        draw_symmray_mps(state.tn, mapper=OneDMap(3, 1))


def test_symmray_peps_summary_and_schematic_for_z2_grid():
    """Symmray PEPS drawings should expose grid bonds and block-sector metadata."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations({(i, j): 0 for i in range(2) for j in range(2)}),
        bond_dim=2,
        seed=3,
        dtype="complex128",
    )

    summary = symmray_peps_summary(state)

    assert summary["Lx"] == 2
    assert summary["Ly"] == 2
    assert summary["num_sites"] == 4
    assert len(summary["bonds"]) == 4
    assert summary["max_bond_dim"] == 2
    assert summary["max_bond_sectors"] == 2
    assert summary["total_stored_size"] == 16
    assert summary["total_dense_size"] == 32
    assert summary["charge_total"] == summary["total_charge"]
    assert summary["Q_total"] == summary["total_parity"]
    assert summary["tensors"][0]["site"] == (0, 0)
    assert summary["tensors"][0]["physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["physical"]["direction"] in {"in", "out"}
    assert summary["tensors"][0]["bonds"]["right"]["between"] == ((0, 0), (0, 1))
    assert summary["tensors"][0]["bonds"]["down"]["between"] == ((0, 0), (1, 0))
    assert summary["bonds"][0]["site_a_direction"] in {"in", "out"}
    assert summary["bonds"][0]["site_b_direction"] in {"in", "out"}

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_peps(
        state,
        title="Z2 PEPS",
        return_summary=True,
    )
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawn_summary["bonds"] == summary["bonds"]

    node_drawing = draw_symmray_peps(state, charge_in_node=True)
    assert hasattr(node_drawing, "fig")


def test_symmray_peps_schematic_hides_auxiliary_bonds_and_sectors_by_default():
    """PEPS drawings should keep routed/multibond debug structure opt-in."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations({(i, j): 0 for i in range(2) for j in range(2)}),
        bond_dim=2,
        seed=31,
        dtype="complex128",
    )

    tn = state.peps.copy()
    top_ind = next(iter(tn[(0, 0)].bonds(tn[(0, 1)])))
    bottom_ind = next(iter(tn[(1, 0)].bonds(tn[(1, 1)])))
    tn.reindex_({bottom_ind: top_ind})
    state.peps = tn

    summary = symmray_peps_summary(state)
    assert len(summary["bonds"]) > len(state.edges)
    assert summary["num_extra_bonds"] > 0

    pytest.importorskip("matplotlib")
    drawing = draw_symmray_peps(state, show_bond_labels=True, show_tensor_labels=False)
    labels = [text.get_text() for text in drawing.ax.texts]
    bond_labels = [label for label in labels if label.startswith("$e_{")]

    assert len(bond_labels) == len(state.edges)
    assert not any("$q_e:$" in label for label in labels)

    debug_drawing = draw_symmray_peps(
        state,
        show_bond_labels=True,
        show_bond_sectors=True,
        show_extra_bonds=True,
        show_tensor_labels=False,
    )
    debug_labels = [text.get_text() for text in debug_drawing.ax.texts]
    debug_bond_labels = [label for label in debug_labels if label.startswith("$e_{")]

    assert len(debug_bond_labels) == len(summary["bonds"])
    assert any("$q_e:$" in label for label in debug_labels)


def test_symmray_peps_schematic_shows_spinful_charge_labels_inside_nodes():
    """Spin-resolved PEPS charges should render as white Q/Sz node labels."""
    pytest.importorskip("matplotlib")
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge_from_occupations(
            {
                (0, 0): (1, 0),
                (0, 1): (0, 1),
                (1, 0): (1, 0),
                (1, 1): (0, 1),
            }
        ),
        bond_dim=2,
        seed=30,
        dtype="complex128",
    )

    drawing = draw_symmray_peps(state, show_tensor_labels=False)
    labels = [text.get_text() for text in drawing.ax.texts]
    node_texts = [text for text in drawing.ax.texts if "$Q=" in text.get_text()]

    assert any("$Q=1$" in label and "$S_z=+1/2$" in label for label in labels)
    assert any("$Q=1$" in label and "$S_z=-1/2$" in label for label in labels)
    assert node_texts
    assert all(text.get_color() == (1.0, 1.0, 1.0, 1.0) for text in node_texts)


def test_symmetric_constructors_apply_to_backend_to_symmray_blocks():
    """Symmetric state/Hamiltonian constructors should backend-map stored blocks."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(dtype=torch.complex128)
    site_charge = site_charge_from_occupations(
        {
            (0, 0): (1, 0),
            (0, 1): (0, 1),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
        }
    )

    state = SymPEPS.random(
        2,
        2,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge,
        bond_dim=2,
        seed=34,
        dtype="complex128",
        to_backend=to_backend,
    )

    tensor = next(iter(state.peps.tensor_map.values()))
    block = next(iter(tensor.data.blocks.values()))
    assert tensor.data.backend == "torch"
    assert isinstance(block, torch.Tensor)
    assert block.dtype == torch.complex128
    summary = symmray_peps_summary(state)
    assert summary["total_stored_size"] > 0

    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        state.edges,
        t=1.0,
        U=4.0,
        mu=0.0,
        to_backend=to_backend,
    )
    term = next(iter(ham.terms.values()))
    term_block = next(iter(term.blocks.values()))
    assert term.backend == "torch"
    assert isinstance(term_block, torch.Tensor)
    assert term_block.dtype == torch.complex128


def test_symmetric_as_scalar_handles_backend_scalars_before_numpy_conversion():
    """Backend scalar conversion should not require NumPy array coercion."""

    class BackendScalar:
        shape = ()

        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 1.25

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("NumPy conversion should not be used.")

    class BackendVector:
        shape = (2,)

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("Non-scalar backend arrays should be returned.")

    vector = BackendVector()

    assert symmetric_mod._as_scalar(BackendScalar()) == pytest.approx(1.25)
    assert symmetric_mod._as_scalar(vector) is vector


def test_symmetric_to_backend_copy_preserves_original_blocks():
    """to_backend(..., inplace=False) should convert a copied wrapper only."""
    torch = pytest.importorskip("torch")
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0, 0, 0, 0]),
        bond_dim=2,
        seed=35,
        dtype="complex128",
    )

    converted = state.to_backend(
        pepsy.backend_torch(dtype=torch.complex128),
        inplace=False,
    )
    original_tensor = next(iter(state.mps.tensor_map.values()))
    converted_tensor = next(iter(converted.mps.tensor_map.values()))
    original_block = next(iter(original_tensor.data.blocks.values()))
    converted_block = next(iter(converted_tensor.data.blocks.values()))

    assert converted is not state
    assert not isinstance(original_block, torch.Tensor)
    assert isinstance(converted_block, torch.Tensor)


def test_symmetric_state_uses_psi_with_network_compatibility_alias():
    """SymMPS should prefer psi/mps naming while keeping network compatibility."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=4,
        dtype="complex128",
    )

    assert state.psi is state.tn
    assert state.network is state.psi
    assert state.mps is state.psi

    wrapped_from_psi = SymMPS(
        psi=state.psi.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )
    wrapped_from_network = SymMPS(
        network=state.psi.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )
    wrapped_from_mps = SymMPS(
        mps=state.psi.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )

    assert wrapped_from_psi.tn is wrapped_from_psi.psi
    assert wrapped_from_network.network is wrapped_from_network.psi
    assert wrapped_from_mps.mps is wrapped_from_mps.psi

    replacement = state.psi.copy()
    wrapped_from_psi.network = replacement
    assert wrapped_from_psi.psi is replacement

    with pytest.raises(TypeError, match="exactly one"):
        SymMPS(
            psi=state.psi,
            network=state.psi,
            symmetry=state.symmetry,
            edges=state.edges,
        )

    with pytest.raises(TypeError, match="only valid"):
        SymMPS(
            peps=state.psi,
            symmetry=state.symmetry,
            edges=state.edges,
        )


def test_sympeps_accepts_peps_constructor_alias():
    """SymPEPS should expose peps as the shape-specific wrapped state name."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations({(i, j): 0 for i in range(2) for j in range(2)}),
        bond_dim=2,
        seed=5,
        dtype="complex128",
    )

    assert state.peps is state.psi
    assert state.network is state.psi

    wrapped = SymPEPS(
        peps=state.peps.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )

    assert wrapped.peps is wrapped.psi
    assert wrapped.tn is wrapped.psi

    replacement = state.peps.copy()
    wrapped.peps = replacement
    assert wrapped.psi is replacement

    with pytest.raises(TypeError, match="only valid"):
        SymPEPS(
            mps=state.peps,
            symmetry=state.symmetry,
            edges=state.edges,
        )


def _square_lattice_edges(Lx, Ly):
    """Return nearest-neighbor square-lattice edges in row-major MPS order."""
    edges = []

    def site(x, y):
        return x * Ly + y

    for x in range(Lx):
        for y in range(Ly):
            if x + 1 < Lx:
                edges.append((site(x, y), site(x + 1, y)))
            if y + 1 < Ly:
                edges.append((site(x, y), site(x, y + 1)))
    return tuple(edges)


def _square_lattice_coordinate_edges(Lx, Ly):
    """Return nearest-neighbor square-lattice edges as PEPS coordinates."""
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            if y + 1 < Ly:
                edges.append(((x, y), (x, y + 1)))
            if x + 1 < Lx:
                edges.append(((x, y), (x + 1, y)))
    return tuple(edges)


def _xy_u1_hamiltonian(edges):
    """Build the U(1)-symmetric XY Hamiltonian from an explicit dense term."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    xy_term_dense = 0.5 * (np.kron(sx, sx) + np.kron(sy, sy))
    xy_term = symm_operator_from_dense(
        xy_term_dense,
        {0: 1, 1: 1},
        symmetry="U1",
        charge=0,
        sites=2,
    )
    return SymHamiltonian(
        model="xy",
        symmetry="U1",
        edges=tuple(edges),
        terms={edge: xy_term for edge in edges},
        parameters={"J": 1.0},
    )


def _all_tensor_data_symmray(tn):
    """Return whether every tensor stores Symmray block-sparse data."""
    return all(
        hasattr(tensor.data, "blocks") and hasattr(tensor.data, "indices")
        for tensor in tn.tensors
    )


def _finite_double_layer_norm(tn):
    """Return whether the direct double-layer norm contraction is finite."""
    norm = (tn.H & tn).contract(all, optimize="auto-hq")
    return np.isfinite(np.real(norm))


def test_symmps_random_unitary_evolution_grows_product_state():
    """Random-unitary initialization should avoid raw random block filling."""
    state = SymMPS.random_unitary_evolution(
        4,
        symmetry="U1U1",
        fermionic=True,
        phys_dim=4,
        bond_dim=4,
        site_charge=site_charge_from_occupations(
            [(1, 0), (0, 1), (1, 0), (0, 1)]
        ),
        seed=123,
        dtype="complex128",
        rounds=20,
    )

    assert state.tn.max_bond() > 1
    assert state.tn.max_bond() <= 4
    assert state.overall_charge() == (2, 2)
    assert _all_tensor_data_symmray(state.tn)
    assert _finite_double_layer_norm(state.tn)
    assert np.real(state.norm()) == pytest.approx(1.0)

    model_state = SymMPS.random_unitary_for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=4,
        site_charge=site_charge_from_occupations(
            [(1, 0), (0, 1), (1, 0), (0, 1)]
        ),
        seed=123,
        dtype="complex128",
        rounds=20,
    )

    assert model_state.model == "fermi_hubbard_u1u1"
    assert model_state.symmetry == "U1U1"
    assert model_state.fermionic
    assert model_state.tn.max_bond() > 1
    assert model_state.overall_charge() == (2, 2)


def _raw_mps_norm(mps):
    """Return an MPS norm with PEPSY's stored exponent removed."""
    raw = mps.copy()
    if hasattr(raw, "exponent"):
        raw.exponent = 0.0
    return raw.norm()


def _build_3x3_symmray_mps_case(name):
    """Build an explicit 3x3 state, Hamiltonian, and gate stream."""
    edges = _square_lattice_edges(3, 3)

    if name == "itf_z2":
        state = SymMPS.random(
            9,
            symmetry="Z2",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations([0] * 9),
            bond_dim=4,
            seed=41,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "itf",
            "Z2",
            edges,
            jx=-1.0,
            hz=-0.5,
        )
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, False, 0

    if name == "xy_u1":
        occupations = [1, 0, 1, 0, 1, 0, 1, 0, 1]
        state = SymMPS.random(
            9,
            symmetry="U1",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations(occupations),
            bond_dim=4,
            seed=42,
            dtype="complex128",
        )
        hamiltonian = _xy_u1_hamiltonian(edges)
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, False, sum(occupations)

    if name == "fermi_hubbard_u1":
        occupations = [1] * 9
        state = SymMPS.random(
            9,
            symmetry="U1",
            phys_dim={0: 1, 1: 2, 2: 1},
            fermionic=True,
            site_charge=site_charge_from_occupations(occupations),
            bond_dim=4,
            seed=43,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "fermi_hubbard",
            "U1",
            edges,
            t=1.0,
            U=2.0,
            mu=0.1,
        )
        gates = hamiltonian.gate_stream(0.0005, imaginary=True)
        return state, hamiltonian, gates, True, sum(occupations)

    raise ValueError(f"Unknown symmetric MPS case {name!r}.")


def _build_3x3_symmray_peps_case(name, *, edges=None):
    """Build an explicit 3x3 PEPS state, Hamiltonian, and gate stream."""
    edges = _square_lattice_coordinate_edges(3, 3) if edges is None else tuple(edges)

    if name == "itf_z2":
        charges = {(i, j): 0 for i in range(3) for j in range(3)}
        state = SymPEPS.random(
            3,
            3,
            symmetry="Z2",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations(charges),
            bond_dim=2,
            seed=51,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "itf",
            "Z2",
            edges,
            jx=-1.0,
            hz=-0.5,
        )
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, 0

    if name == "xy_u1":
        charges = {(i, j): (i + j) % 2 for i in range(3) for j in range(3)}
        state = SymPEPS.random(
            3,
            3,
            symmetry="U1",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations(charges),
            bond_dim=2,
            seed=52,
            dtype="complex128",
        )
        hamiltonian = _xy_u1_hamiltonian(edges)
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, sum(charges.values())

    if name == "fermi_hubbard_u1":
        charges = {(i, j): 1 for i in range(3) for j in range(3)}
        state = SymPEPS.random(
            3,
            3,
            symmetry="U1",
            phys_dim={0: 1, 1: 2, 2: 1},
            fermionic=True,
            site_charge=site_charge_from_occupations(charges),
            bond_dim=2,
            seed=53,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "fermi_hubbard",
            "U1",
            edges,
            t=1.0,
            U=2.0,
            mu=0.1,
        )
        gates = hamiltonian.gate_stream(0.0005, imaginary=True)
        return state, hamiltonian, gates, sum(charges.values())

    raise ValueError(f"Unknown symmetric PEPS case {name!r}.")


def test_sector_and_charge_helpers_make_total_charge_explicit():
    """Physical sectors and local charges should be easy to inspect."""
    assert default_physical_sectors("U1", 4) == {0: 1, 1: 2, 2: 1}
    assert default_physical_sectors(model="fermi_hubbard") == {0: 1, 1: 2, 2: 1}
    assert default_physical_sectors(model="fermi_hubbard_u1u1") == {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1,
    }
    assert sector_index_map({0: 1, 1: 2, 2: 1}) == {0: 0, 1: 1, 2: 1, 3: 2}

    occupations = [1, 0, 1, 0]
    state = SymMPS.random(
        4,
        symmetry="U1",
        bond_dim=2,
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations(occupations),
        seed=31,
        dtype="complex128",
    )

    assert state.phys_sectors == {0: 1, 1: 1}
    assert state.site_charges() == {0: 1, 1: 0, 2: 1, 3: 0}
    assert state.overall_charge() == 2
    assert state.overall_parity() == 0
    assert site_charge_uniform(1)("anything") == 1
    assert site_charge_alternating(even=0, odd=1)((1, 2)) == 1
    assert site_charge_from_map({(0, 0): 1}, default=0)((1, 1)) == 0


def test_symmps_heisenberg_builds_energy_and_imaginary_step():
    """SymMPS should build U(1) Heisenberg terms and evolve in place."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=1,
        dtype="complex128",
    )

    ham = state.build_hamiltonian()

    assert isinstance(ham, SymHamiltonian)
    assert state.symmetry == "U1"
    assert not state.fermionic
    assert len(ham.terms) == 3
    assert all(term.shape == (2, 2, 2, 2) for term in ham.terms.values())
    assert state.tn.L == 4

    energy_before = state.energy(ham)
    state.ground_state(dt=0.01, steps=1, hamiltonian=ham, max_bond=4)
    energy_after = state.energy(ham)

    assert np.isfinite(np.real(energy_before))
    assert np.isfinite(np.real(energy_after))
    assert state.tn.max_bond() <= 4
    assert state.norm() == pytest.approx(1.0)


@pytest.mark.parametrize("model", ["heisenberg", "fermi_hubbard"])
def test_symmps_mps_optimizer_simple_update_preserves_symmray_data(model):
    """Simple update should preserve Symmray tensor data under default settings."""
    state = SymMPS.for_model(
        model,
        3,
        bond_dim=2,
        seed=18,
        dtype="complex128",
    )
    if model == "fermi_hubbard":
        hamiltonian = state.build_hamiltonian(t=1.0, U=2.0, mu=0.1)
    else:
        hamiltonian = state.build_hamiltonian()
    gates = hamiltonian.gate_stream(0.001, imaginary=True)

    optimizer = pepsy.MpsOptimizer(
        state.tn.copy(),
        gates,
        chi=4,
        mode="su",
    )
    out = optimizer.run(progbar=False, cutoff=1.0e-10)

    assert _all_tensor_data_symmray(out)
    assert out.max_bond() <= 4
    assert len(optimizer.gauges) == out.L - 1
    assert optimizer.p_ungauged is not None
    assert np.isfinite(np.real(optimizer.p_ungauged.norm()))


def test_symmps_measures_dense_generic_observables():
    """SymMPS.measure should convert dense local operators to Symmray arrays."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=32,
        dtype="complex128",
    )
    z_op = np.diag([1.0, -1.0])
    zz_op = np.diag([1.0, -1.0, -1.0, 1.0])
    z_sym = symm_operator_from_dense(
        z_op,
        state.phys_sectors,
        symmetry=state.symmetry,
        charge=0,
    )

    measured_dense = state.measure(z_op, where=1, contraction_opt="auto-hq")
    measured_sym = state.measure(z_sym, where=1, contraction_opt="auto-hq")
    measured_zz = state.measure(zz_op, where=(1, 2), contraction_opt="auto-hq")

    assert measured_dense == pytest.approx(measured_sym)
    assert np.isfinite(np.real(measured_zz))


def test_symmps_fermi_hubbard_defaults_to_fermionic_u1():
    """Fermi-Hubbard convenience defaults should use U(1) fermionic tensors."""
    state = SymMPS.for_model(
        "fermi_hubbard",
        3,
        bond_dim=2,
        seed=2,
        dtype="complex128",
    )

    ham = state.build_hamiltonian(t=1.0, U=4.0, mu=0.5)
    gates = state.trotter_gates(0.01, hamiltonian=ham, imaginary=True)

    assert state.symmetry == "U1"
    assert state.fermionic
    assert len(ham.terms) == 2
    assert all(term.shape == (4, 4, 4, 4) for term in ham.terms.values())
    assert isinstance(gates, SymGateStream)
    assert len(gates) == 2

    evolved = state.time_evolve(
        0.01,
        steps=1,
        hamiltonian=ham,
        imaginary=True,
        max_bond=4,
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.max_bond() <= 4
    assert evolved.norm() == pytest.approx(1.0)


def test_fermi_hubbard_u1u1_preset_uses_spin_resolved_fermionic_tensors():
    """U1U1 preset should expose spin-resolved spinful Hubbard sectors."""
    sectors = default_physical_sectors(model="fermi_hubbard_u1u1")
    mps = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=2,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1), (1, 0), (0, 1)]),
        seed=3,
        dtype="complex128",
    )
    ham_mps = mps.build_hamiltonian(t=1.0, U=4.0, mu=0.0)
    evolved_mps = mps.time_evolve(
        0.001,
        steps=1,
        hamiltonian=ham_mps,
        imaginary=True,
        max_bond=4,
        inplace=False,
    )

    assert mps.symmetry == "U1U1"
    assert mps.fermionic
    assert mps.phys_sectors == sectors
    assert mps.overall_charge() == (2, 2)
    assert all(type(term).__name__ == "U1U1FermionicArray" for term in ham_mps.terms.values())
    mps_ordering = mps.fermionic_ordering()
    assert mps_ordering["enabled"] is True
    assert mps_ordering["network_kind"] == "mps"
    assert mps_ordering["methods_reference"]["doi"] == "10.1103/PhysRevResearch.7.023193"
    assert mps_ordering["site_order"] == (0, 1, 2, 3)
    assert mps_ordering["edge_order"] == mps.edges
    assert mps_ordering["edges"][0]["edge"] == (0, 1)
    assert mps_ordering["edges"][0]["edge_order"] == 0
    assert mps_ordering["edges"][0]["index_directions"][0]["site"] == 0
    assert mps_ordering["edges"][0]["index_directions"][0]["direction"] in {"in", "out"}
    assert evolved_mps.overall_charge() == (2, 2)
    assert evolved_mps.tn.max_bond() <= 4
    assert evolved_mps.norm() == pytest.approx(1.0)

    peps_charges = {
        (0, 0): (1, 0),
        (0, 1): (0, 1),
        (1, 0): (1, 0),
        (1, 1): (0, 1),
    }
    peps = SymPEPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        2,
        bond_dim=2,
        site_charge=site_charge_from_occupations(peps_charges),
        seed=4,
        dtype="complex128",
    )
    ham_peps = peps.build_hamiltonian(t=1.0, U=4.0, mu=0.0)
    evolved_peps = peps.time_evolve(
        0.001,
        steps=1,
        hamiltonian=ham_peps,
        imaginary=True,
        max_bond=4,
        method="gate",
        inplace=False,
    )

    assert peps.symmetry == "U1U1"
    assert peps.fermionic
    assert peps.phys_sectors == sectors
    assert peps.overall_charge() == (2, 2)
    assert all(type(term).__name__ == "U1U1FermionicArray" for term in ham_peps.terms.values())
    peps_ordering = peps.fermionic_ordering()
    assert peps_ordering["enabled"] is True
    assert peps_ordering["network_kind"] == "peps"
    assert peps_ordering["methods_reference"]["doi"] == "10.1103/PhysRevResearch.7.023193"
    assert peps_ordering["site_order"] == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert peps_ordering["edge_order"] == peps.edges
    peps_first_edge = peps.edges[0]
    peps_first_record = next(
        record for record in peps_ordering["edges"]
        if record["edge"] == peps_first_edge
    )
    assert peps_first_record["edge_order"] == 0
    assert tuple(item["site"] for item in peps_first_record["index_directions"]) == peps_first_edge
    assert peps_first_record["index_directions"][0]["direction"] in {"in", "out"}
    assert evolved_peps.overall_charge() == (2, 2)
    assert evolved_peps.tn.max_bond() <= 4


def test_fermi_hubbard_u1u1_light_pulse_stream_uses_paper_schedule():
    """The native fermion pulse stream should expose the merged Strang layers."""
    edges = ((0, 1), (1, 2))
    stream = fermi_hubbard_u1u1_light_pulse_gate_stream(
        edges,
        sites=(0, 1, 2),
        t=1.0,
        U=8.0,
        omega=4 * np.pi / 3,
        pulse_steps=2,
        relaxation_steps=2,
    )

    assert isinstance(stream, SymGateStream)
    assert stream.dt == pytest.approx(0.375)
    assert stream.order == 2
    assert len(stream) == 5 * 3 + 4 * 2
    assert [where for _, where in stream[:3]] == [0, 1, 2]
    assert [where for _, where in stream[3:5]] == [(0, 1), (1, 2)]
    assert all(type(gate_i).__name__ == "U1U1FermionicArray" for gate_i, _ in stream)


def test_fermi_hubbard_u1u1_streams_run_mps_optimizer_and_direct_gate():
    """Onsite and Peierls hopping layers should replay through SymMPS paths."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        3,
        bond_dim=2,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1), (1, 0)]),
        seed=51,
        dtype="complex128",
    )
    interaction = fermi_hubbard_u1u1_interaction_gate_stream(
        range(3),
        0.001,
        U=2.0,
    )
    hopping = fermi_hubbard_u1u1_hopping_gate_stream(
        ((0, 1), (1, 2)),
        0.001,
        t=0.5,
        peierls_angle=0.3,
    )
    stream = fermi_hubbard_u1u1_gate_stream(
        ((0, 1), (1, 2)),
        0.001,
        sites=range(3),
        t=0.5,
        U=2.0,
        peierls_angle=0.3,
    )

    direct = state.copy().apply_gates(interaction, method="direct", inplace=False)
    opt = pepsy.MpsOptimizer(state.tn.copy(), stream, chi=4, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-10)

    assert len(interaction) == 3
    assert len(hopping) == 2
    assert len(stream) == 2 * len(interaction) + 2 * len(hopping)
    assert direct.tn.L == 3
    assert out.L == 3
    assert out.max_bond() <= 4
    assert np.isfinite(np.real((out.H & out).contract(all, optimize="auto-hq")))


def test_fermi_hubbard_u1u1_light_pulse_stream_runs_sympeps_simple_update():
    """The same native fermion pulse stream should feed PEPS simple update."""
    site_charge = site_charge_from_occupations(
        {
            (0, 0): (1, 0),
            (0, 1): (0, 1),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
        }
    )
    state = SymPEPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        2,
        bond_dim=2,
        site_charge=site_charge,
        seed=52,
        dtype="complex128",
    )
    stream = fermi_hubbard_u1u1_light_pulse_gate_stream(
        state.edges,
        sites=state.sites,
        t=0.2,
        U=1.0,
        omega=4 * np.pi / 3,
        pulse_steps=2,
    )
    gauges = {}

    out = state.copy().apply_gates(
        stream,
        method="simple",
        gauges=gauges,
        max_bond=4,
        cutoff=1e-10,
    )

    assert out.tn.max_bond() <= 4
    assert len(gauges) > 0
    assert np.isfinite(np.real(out.norm()))


def test_fermi_hubbard_u1u1_hamiltonian_builds_mpo_energy_path():
    """The FH MPO should match adjacent two-site Symmray term energy."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)]),
        seed=11,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1)],
        t=1.0,
        U=8.0,
        mu=0.2,
        V=0.3,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_native_fermionic_mps_rejects_bosonic_mpo_without_opt_in():
    """MPO re-encoding must be explicit for native fermionic MPS states."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=2,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)]),
        seed=12,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1)],
        t=1.0,
        U=8.0,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    with pytest.raises(ValueError, match="encoding mismatch"):
        pepsy.MpsEnergyOptimizer(
            state,
            mpo,
            energy_per_site=False,
        ).energy()


def test_native_fermionic_mpo_energy_is_factorized_on_l12(monkeypatch):
    """Native MPO energy remains exact without forming a global operator."""
    L = 12
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        L,
        bond_dim=4,
        site_charge=site_charge_from_occupations(
            [(1, 0) if site % 2 == 0 else (0, 1) for site in range(L)]
        ),
        seed=222,
        dtype="complex128",
    )
    fermion = Fermion(spinful=True, symmetry="U1U1")
    hamiltonian = fermion.hamiltonian(
        [(site, site + 1) for site in range(L - 1)],
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    mpo = fermion.to_mpo(
        hamiltonian=hamiltonian,
        L=L,
        compress=False,
    )
    monkeypatch.setattr(
        mpo,
        "to_dense",
        lambda: pytest.fail("native MPO energy must remain factorized"),
    )

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        hamiltonian.terms,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_native_mpo_energy_reuses_paths_and_supports_controlled_compression():
    """Native MPO path caching and bounded compression preserve energy accuracy."""
    L = 12
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        L,
        bond_dim=8,
        site_charge=site_charge_from_occupations(
            [(1, 0) if site % 2 == 0 else (0, 1) for site in range(L)]
        ),
        seed=123,
        dtype="complex128",
    )
    fermion = Fermion(spinful=True, symmetry="U1U1")
    hamiltonian = fermion.hamiltonian(
        [(site, site + 1) for site in range(L - 1)],
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    mpo = fermion.to_mpo(hamiltonian=hamiltonian, L=L, compress=False)
    optimizer = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    )

    exact = optimizer.energy().energy
    path_optimizer = optimizer._native_mpo_path_optimizer
    assert path_optimizer is not None
    assert path_optimizer.last_opt is not None

    repeated = optimizer.energy().energy
    assert optimizer._native_mpo_path_optimizer is path_optimizer
    assert complex(repeated) == pytest.approx(complex(exact))

    compressed = optimizer.energy(
        native_mpo_compression={
            "max_bond": 64,
            "cutoff": 1e-12,
            "method": "svd",
        }
    ).energy
    assert complex(compressed) == pytest.approx(complex(exact), abs=1e-10)
    compressed_estimate = optimizer.energy(
        native_mpo_compression={"max_bond": 64, "cutoff": 1e-12}
    )
    assert compressed_estimate.metadata["native_mpo_compression"]["max_bond"] == 64

    with pytest.raises(ValueError, match="requires an explicit max_bond"):
        optimizer.energy(native_mpo_compression={"cutoff": 1e-12})
    with pytest.raises(ValueError, match="requires an explicit max_bond"):
        optimizer.energy(native_mpo_compression={})
@pytest.mark.parametrize(
    ("spinful", "symmetry", "model"),
    [
        (False, "U1", "fermi_hubbard_spinless"),
        (False, "Z2", "fermi_hubbard_spinless"),
        (True, "U1", "fermi_hubbard"),
        (True, "U1U1", "fermi_hubbard_u1u1"),
    ],
)
def test_native_factorized_mpo_matches_terms_and_jw_reference(
    spinful, symmetry, model
):
    """Native MPO energies agree with terms and the separate JW operator oracle."""
    L = 4
    edges = [(site, site + 1) for site in range(L - 1)]
    occupations = (
        [1] * L
        if symmetry == "U1" and spinful
        else ([(1, 0) if site % 2 == 0 else (0, 1) for site in range(L)]
              if spinful
              else [1, 0, 1, 0])
    )
    state = SymMPS.for_model(
        model,
        L,
        symmetry=symmetry,
        bond_dim=4,
        site_charge=site_charge_from_occupations(occupations),
        seed=111,
        dtype="complex128",
    )
    fermion = Fermion(spinful=spinful, symmetry=symmetry)
    if spinful:
        hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)
    else:
        hamiltonian = fermion.hamiltonian(edges, t=1.0, V=0.4, mu=0.1)

    native_mpo = fermion.to_mpo(
        hamiltonian=hamiltonian,
        L=L,
        compress=False,
    )
    jw_mpo = hamiltonian.to_mpo(
        L=L,
        fermionic=False,
        compress=False,
    )
    native_energy = pepsy.MpsEnergyOptimizer(
        state,
        native_mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        hamiltonian.terms,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert complex(native_energy) == pytest.approx(complex(term_energy))
    np.testing.assert_allclose(
        _mpo_to_dense_matrix(native_mpo, L),
        _mpo_to_dense_matrix(jw_mpo, L),
        atol=1e-10,
    )


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_native_mpo_noncontiguous_hopping_matches_native_local_term(symmetry):
    """Native graded MPOs preserve signs across skipped chain sites."""
    L = 5
    fermion = Fermion(spinful=True, symmetry=symmetry)
    occupations = (
        [1] * L
        if symmetry == "U1"
        else [(1, 0) if site % 2 == 0 else (0, 1) for site in range(L)]
    )
    state = SymMPS.random_unitary_evolution(
        L,
        symmetry=symmetry,
        fermionic=True,
        phys_dim=default_physical_sectors(symmetry, 4),
        site_charge=site_charge_from_occupations(occupations),
        bond_dim=4,
        seed=333,
        dtype="complex128",
        rounds=2,
        stall_rounds=1,
    )

    for edge in ((0, 2), (0, 4), (1, 4)):
        term = fermion.hopping_operator()
        hamiltonian = fermion.hamiltonian({edge: term})
        native_mpo = hamiltonian.to_mpo(L=L, fermionic=True, compress=False)
        native_mpo_value = pepsy.MpsEnergyOptimizer(
            state,
            native_mpo,
            energy_per_site=False,
            real=False,
        ).energy().energy
        local_term_value = pepsy.MpsEnergyOptimizer(
            state,
            {edge: term},
            energy_per_site=False,
            real=False,
        ).energy().energy

        assert complex(native_mpo_value) == pytest.approx(
            complex(local_term_value), abs=1e-10
        )


def test_fermi_hubbard_u1u1_mpo_energy_matches_high_bond_fermionic_mps():
    """High-bond fermionic FH MPS energy should use the direct MPO."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=16,
        site_charge=site_charge_from_occupations(
            [(1, 0), (0, 1), (1, 0), (0, 1)]
        ),
        seed=222,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1)],
        t=1.0,
        U=0.0,
        mu=0.0,
    )
    mpo = ham.to_mpo(L=4, compress=True, max_bond=48, cutoff=1e-12)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert not hasattr(mpo, "_pepsy_source_terms")
    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


@pytest.mark.parametrize(
    "model,symmetry,site_charges,params",
    [
        ("tfim", "Z2", [0, 0], {"jx": -1.0, "hz": -0.5}),
        ("heisenberg", "U1", [1, 0], {}),
    ],
)
def test_symmetric_hamiltonian_to_mpo_supports_spin_models(
    model,
    symmetry,
    site_charges,
    params,
):
    """Generic SymHamiltonian MPOs should support non-fermionic Z2/U1 terms."""
    state = SymMPS.for_model(
        model,
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations(site_charges),
        seed=21,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(model, symmetry, [(0, 1)], **params)
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert mpo.L == 2
    assert type(mpo[0].data).__name__.startswith(symmetry)
    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_spinless_fermi_hubbard_u1_hamiltonian_builds_mpo_energy_path():
    """Spinless FH U1 MPOs should preserve the fermionic contraction signs."""
    state = SymMPS.for_model(
        "fermi_hubbard_spinless",
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations([1, 0]),
        seed=23,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [(0, 1)],
        t=1.0,
        V=0.5,
        mu=0.1,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = state.energy(
        hamiltonian=ham,
        normalized=True,
        contraction_opt="auto-hq",
    )

    assert type(mpo[0].data).__name__ == "U1Array"
    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_spinless_fermi_hubbard_u1_hamiltonian_mpo_matches_shifted_edges():
    """Spinless FH hopping signs should not depend on the bond position."""
    state = SymMPS.for_model(
        "fermi_hubbard_spinless",
        4,
        bond_dim=3,
        site_charge=site_charge_from_occupations([1, 0, 1, 0]),
        seed=222,
        dtype="complex128",
    )

    for edge in [(0, 1), (1, 2), (2, 3)]:
        ham = SymHamiltonian.from_edges(
            "fermi_hubbard_spinless",
            "U1",
            [edge],
            t=1.0,
            V=0.0,
            mu=0.0,
        )
        mpo = ham.to_mpo(L=4, compress=False)
        mpo_energy = pepsy.MpsEnergyOptimizer(
            state,
            mpo,
            energy_per_site=False,
            real=False,
            allow_encoding_conversion=True,
        ).energy().energy
        term_energy = state.energy(
            hamiltonian=ham,
            normalized=True,
            contraction_opt="auto-hq",
        )

        assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_spinless_fermi_hubbard_u1_hamiltonian_mpo_compresses_long_range():
    """Spinless FH long-range MPOs should insert parity strings and compress."""
    state = SymMPS.for_model(
        "fermi_hubbard_spinless",
        3,
        bond_dim=3,
        site_charge=site_charge_from_occupations([1, 0, 0]),
        seed=25,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [(0, 2)],
        t=1.0,
        V=0.0,
        mu=0.0,
    )
    mpo = ham.to_mpo(L=3, compress=False)
    mpo_compressed = ham.to_mpo(L=3, compress=True, max_bond=16, cutoff=1e-12)

    energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    energy_compressed = pepsy.MpsEnergyOptimizer(
        state,
        mpo_compressed,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert mpo.max_bond() >= 3
    assert mpo_compressed.max_bond() <= mpo.max_bond()
    assert complex(energy) == pytest.approx(complex(term_energy))
    assert complex(energy_compressed) == pytest.approx(complex(energy))


def test_symhamiltonian_mpo_compression_report_warns_for_soft_bond_cap():
    """A tied Symmray cutoff must not silently exceed the requested cap."""
    lx, ly = 4, 3
    length = lx * ly
    edges = []
    for x in range(lx):
        for y in range(ly):
            site = x * ly + y
            edges.append((site, ((x + 1) % lx) * ly + y))
            edges.append((site, x * ly + ((y + 1) % ly)))
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        edges,
        t=1.0,
        U=8.0,
    )

    with pytest.warns(RuntimeWarning, match="requested max_bond=16"):
        mpo = ham.to_mpo(
            L=length,
            compress=True,
            max_bond=16,
            cutoff=1e-12,
        )

    report = mpo.pepsy_compression_report
    assert report["compressed"] is True
    assert report["cutoff"] == 1e-12
    assert report["requested_max_bond"] == 16
    assert report["raw_max_bond"] > report["final_max_bond"]
    assert report["final_max_bond"] == mpo.max_bond()
    assert report["rank_reduced"] is True
    assert report["cap_bound"] is True
    assert report["max_bond_exceeded"] is True
    assert report["final_max_bond"] > report["requested_max_bond"]


def test_spinless_fermi_hubbard_u1_hamiltonian_mpo_maps_2d_long_range_edge():
    """Spinless FH coordinate edges should match their mapped flat edge."""
    mapper = OneDMap(2, 2, mode="snake")
    idx2coo, coo2idx = mapper.build()
    edge_2d = ((0, 0), (1, 0))
    occupations = {
        (0, 0): 1,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 0,
    }
    state = SymMPS.for_model(
        "fermi_hubbard_spinless",
        4,
        bond_dim=3,
        site_charge=site_charge_from_occupations(
            [occupations[idx2coo[i]] for i in range(4)]
        ),
        seed=26,
        dtype="complex128",
    )
    params = {"t": 1.0, "V": 0.0, "mu": 0.0}
    ham_2d = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [edge_2d],
        **params,
    )
    flat_edge = tuple(coo2idx[site] for site in edge_2d)
    ham_flat = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [flat_edge],
        **params,
    )

    mpo_from_mapper = ham_2d.to_mpo(mapper=mapper, compress=False)
    mpo_from_flat = ham_flat.to_mpo(L=4, compress=False)

    def energy(mpo):
        return pepsy.MpsEnergyOptimizer(
            state,
            mpo,
            energy_per_site=False,
            real=False,
            allow_encoding_conversion=True,
        ).energy().energy

    assert abs(coo2idx[edge_2d[0]] - coo2idx[edge_2d[1]]) > 1
    assert complex(energy(mpo_from_mapper)) == pytest.approx(complex(energy(mpo_from_flat)))


def test_spinful_fermi_hubbard_total_u1_mpo_builds_energy_path():
    """Spinful FH total-U1 MPOs should use the same JW path as U1U1."""
    state = SymMPS.for_model(
        "fermi_hubbard",
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations([1, 1]),
        seed=11,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard",
        "U1",
        [(0, 1)],
        t=1.0,
        U=4.0,
        mu=0.1,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_handles_long_range_string():
    """Long-range mapped FH terms should include a parity string and compress."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        3,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 0), (0, 1)]),
        seed=12,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 2)],
        t=1.0,
        U=0.0,
        mu=0.0,
    )
    mpo = ham.to_mpo(L=3, compress=False)
    mpo_compressed = ham.to_mpo(L=3, compress=True, max_bond=16, cutoff=1e-12)

    energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    energy_compressed = pepsy.MpsEnergyOptimizer(
        state,
        mpo_compressed,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert mpo.max_bond() >= 3
    assert mpo_compressed.max_bond() <= mpo.max_bond()
    assert complex(energy) == pytest.approx(complex(term_energy))
    assert complex(energy_compressed) == pytest.approx(complex(energy))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_supports_long_range_density_v():
    """Spinful U1U1 FH MPOs should support neutral NN density channels."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=4,
        site_charge=site_charge_from_occupations(
            [(1, 0), (0, 0), (1, 1), (0, 1)]
        ),
        seed=1234,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 2), (1, 3)],
        t=0.0,
        U=0.0,
        mu=0.0,
        V=0.7,
    )
    mpo = ham.to_mpo(L=4, compress=False)
    mpo_compressed = ham.to_mpo(L=4, compress=True, max_bond=16, cutoff=1e-12)

    energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    energy_compressed = pepsy.MpsEnergyOptimizer(
        state,
        mpo_compressed,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert mpo.max_bond() >= 3
    assert complex(energy) == pytest.approx(complex(term_energy))
    assert complex(energy_compressed) == pytest.approx(complex(energy))


@pytest.mark.parametrize(
    "hopping,site_charges",
    [
        ((1.0, 0.0), [(0, 1), (0, 0)]),
        ((1.0, 0.0), [(1, 1), (1, 0)]),
        ((0.0, 1.0), [(1, 0), (0, 0)]),
        ((0.0, 1.0), [(1, 1), (0, 1)]),
        ((1.0, 1.0), [(1, 0), (0, 1)]),
        ((1.0, 1.0), [(0, 0), (1, 1)]),
        ((1.0, 1.0), [(1, 1), (0, 0)]),
    ],
)
def test_fermi_hubbard_u1u1_hamiltonian_mpo_matches_two_site_sectors(
    hopping,
    site_charges,
):
    """The U1U1 FH MPO should use JW signs on the two-site boundary case."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations(site_charges),
        seed=123,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1)],
        t=hopping,
        U=0.0,
        mu=0.0,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_matches_shifted_edges():
    """U1U1 FH MPO hopping channels need prefix parity on shifted bonds."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1), (1, 0), (0, 1)]),
        seed=222,
        dtype="complex128",
    )

    for hopping in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        for edge in [(0, 1), (1, 2), (2, 3)]:
            ham = SymHamiltonian.from_edges(
                "fermi_hubbard_u1u1",
                "U1U1",
                [edge],
                t=hopping,
                U=0.0,
                mu=0.0,
            )
            mpo = ham.to_mpo(L=4, compress=False)
            mpo_energy = pepsy.MpsEnergyOptimizer(
                state,
                mpo,
                energy_per_site=False,
                real=False,
                allow_encoding_conversion=True,
            ).energy().energy
            term_energy = state.energy(
                hamiltonian=ham,
                normalized=True,
                contraction_opt="auto-hq",
            )

            assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_builds_pbc_wrap_edge():
    """PBC wrap edges should build a finite long-string U1U1 FH MPO."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1), (1, 0), (0, 1)]),
        seed=222,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        t=1.0,
        U=4.0,
        mu=0.0,
        V=0.25,
    )
    mpo = ham.to_mpo(L=4, compress=False)
    energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy
    term_energy = pepsy.MpsEnergyOptimizer(
        state,
        ham.terms,
        energy_per_site=False,
        real=False,
        allow_encoding_conversion=True,
    ).energy().energy

    assert mpo.L == 4
    assert mpo.max_bond() >= 4
    assert np.isfinite(complex(energy))
    assert complex(energy) == pytest.approx(complex(term_energy))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_maps_2d_edges_with_onedmap():
    """Coordinate FH edges should align with an explicit OneDMap chain path."""
    mapper = OneDMap(2, 2, mode="snake")
    idx2coo, coo2idx = mapper.build()
    edges_2d = (
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
    )
    occupations = {
        (0, 0): (1, 0),
        (0, 1): (0, 1),
        (1, 0): (0, 1),
        (1, 1): (1, 0),
    }
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=3,
        site_charge=site_charge_from_occupations(
            [occupations[idx2coo[i]] for i in range(4)]
        ),
        seed=13,
        dtype="complex128",
    )
    params = {"t": 1.0, "U": 4.0, "mu": 0.25, "V": 0.4}
    ham_2d = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        edges_2d,
        **params,
    )
    flat_edges = tuple((coo2idx[left], coo2idx[right]) for left, right in edges_2d)
    ham_flat = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        flat_edges,
        **params,
    )

    mpo_from_mapper = ham_2d.to_mpo(mapper=mapper, compress=False)
    mpo_from_maps = ham_2d.to_mpo(idx2coo=idx2coo, coo2idx=coo2idx, compress=False)
    mpo_from_flat = ham_flat.to_mpo(L=4, compress=False)

    def energy(mpo):
        return pepsy.MpsEnergyOptimizer(
            state,
            mpo,
            energy_per_site=False,
            real=False,
            allow_encoding_conversion=True,
        ).energy().energy

    energy_flat = energy(mpo_from_flat)
    assert mpo_from_mapper.L == 4
    assert mpo_from_maps.L == 4
    assert complex(energy(mpo_from_mapper)) == pytest.approx(complex(energy_flat))
    assert complex(energy(mpo_from_maps)) == pytest.approx(complex(energy_flat))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_requires_mapper_for_2d_edges():
    """Coordinate FH edges need an explicit chain path for fermionic strings."""
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [((0, 0), (1, 0))],
        t=1.0,
        U=0.0,
        mu=0.0,
    )

    with pytest.raises(ValueError, match="requires mapper=OneDMap"):
        ham.to_mpo()


def test_symmps_gate_stream_runs_mps_optimizer_mpo_heisenberg():
    """Symmray U(1) gates should run through MpsOptimizer(mode='mpo')."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=7,
        dtype="complex128",
    )
    ham = state.build_hamiltonian()
    gates = ham.gate_stream(0.01)

    opt = pepsy.MpsOptimizer(state.tn.copy(), gates, chi=4, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-10)

    assert out.L == 4
    assert out.max_bond() <= 4
    assert np.isfinite(np.real((out.H & out).contract(all, optimize="auto-hq")))


def test_u1u1_fermionic_mps_optimizer_two_site_fit_stays_native():
    """Two-site FIT must preserve spin-resolved sectors and graded metadata."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=2,
        site_charge=site_charge_from_occupations(
            [(1, 0), (0, 1), (1, 0), (0, 1)]
        ),
        seed=205,
        dtype="complex128",
    )
    gates = state.build_hamiltonian(
        t=1.0,
        U=4.0,
        mu=0.0,
    ).gate_stream(0.001)

    optimizer = pepsy.MpsOptimizer(
        state.tn.copy(),
        gates,
        chi=16,
        mode="dmrg",
    )
    out = optimizer.run(
        progbar=False,
        n_iter=2,
        cutoff=1.0e-10,
        fit_block_size=2,
        fit_sweep_sequence="RL",
    )

    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in out.tensors
    )
    reference = pepsy.MpsOptimizer(
        state.tn.copy(),
        gates,
        chi=16,
        mode="mpo",
    ).run(progbar=False, cutoff=0.0)

    assert out.max_bond() <= 16
    assert _finite_double_layer_norm(out)
    assert float(
        np.real(pepsy.tn_fidelity(out, reference, contraction_opt="auto-hq"))
    ) == pytest.approx(1.0, abs=1.0e-9)


def test_symmps_mps_optimizer_handles_spinful_fermi_hubbard_dims():
    """MpsOptimizer MPO mode should accept 4-state Fermi-Hubbard gates."""
    state = SymMPS.for_model(
        "fermi_hubbard",
        3,
        bond_dim=2,
        seed=8,
        dtype="complex128",
    )
    ham = state.build_hamiltonian(t=1.0, U=2.0, mu=0.1)

    evolved = state.time_evolve_mps_optimizer(
        0.005,
        steps=1,
        hamiltonian=ham,
        imaginary=True,
        chi=4,
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.L == 3
    assert evolved.tn.max_bond() <= 4
    assert np.isfinite(np.real(evolved.norm()))
    raw = evolved.tn.copy()
    raw.exponent = 0
    raw_norm = (raw.H & raw).contract(all, optimize="auto-hq")
    assert np.isfinite(np.real(raw_norm))
    assert np.real(raw_norm) > 0.0


def test_symmray_mpo_real_time_fermion_stream_preserves_norm_without_truncation():
    """All Symmray MPO gates must use the graded auto-swap application path."""
    torch = pytest.importorskip("torch")
    import quimb.tensor as qtn

    backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        to_backend=backend,
    )
    lx, ly = 2, 3
    occupations = {
        (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
        for x in range(lx)
        for y in range(ly)
    }
    mapper = pepsy.OneDMap(lx, ly, mode="snake")
    idx2coo, coo2idx = mapper.build()
    state = pepsy.ps_to_mps(
        lx * ly,
        fermion=fermion,
        occupations=tuple(occupations[idx2coo[index]] for index in range(lx * ly)),
        seed=101,
        dtype="complex128",
        to_backend=backend,
    )
    state.normalize()

    half_dt = 0.005 / 2.0
    interaction = fermion.interaction_gate(half_dt, U=8.0, imaginary=False)
    hopping = fermion.hopping_gate(half_dt, t=1.0, imaginary=False)
    edges = tuple(qtn.edges_2d_square(lx, ly, cyclic=False))
    layers = fermion.edge_coloring_layers(edges)
    coordinate_stream = (
        [(interaction, site) for site in occupations]
        + [(hopping, edge) for layer in layers for edge in layer]
        + [(hopping, edge) for layer in reversed(layers) for edge in reversed(layer)]
        + [(interaction, site) for site in occupations]
    )
    stream = [
        (
            gate,
            tuple(coo2idx[site] for site in where)
            if isinstance(where, tuple) and len(where) == 2 and isinstance(where[0], tuple)
            else coo2idx[where],
        )
        for gate, where in coordinate_stream
    ]

    out = pepsy.MpsOptimizer(state, stream * 2, chi=64, mode="mpo", inplace=True).run(
        progbar=False,
        cutoff=0.0,
        non_unitary=False,
    )

    assert all(getattr(tensor.data, "backend", None) == "torch" for tensor in out.tensors)
    assert float(np.real(pepsy.to_float(out.norm()))) == pytest.approx(1.0, abs=1.0e-10)


def test_symmps_mps_optimizer_handles_cuda_torch_blocks():
    """Symmray canonicalization should not coerce CUDA torch blocks via NumPy."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA torch device is not available.")

    to_backend = pepsy.backend_torch(device="cuda:0", dtype=torch.complex128)
    state = SymMPS.for_model(
        "fermi_hubbard",
        3,
        bond_dim=2,
        seed=8,
        dtype="complex128",
        to_backend=to_backend,
    )
    ham = state.build_hamiltonian(
        t=1.0,
        U=2.0,
        mu=0.1,
        to_backend=to_backend,
    )

    evolved = state.time_evolve_mps_optimizer(
        0.005,
        steps=1,
        hamiltonian=ham,
        imaginary=True,
        chi=4,
        inplace=False,
    )

    block = next(iter(next(iter(evolved.tn.tensor_map.values())).data.blocks.values()))
    assert block.device.type == "cuda"
    assert evolved.tn.L == 3


def test_symmps_mps_optimizer_coerces_dense_hamiltonian_terms():
    """Dense custom Hamiltonian terms should not mix NumPy gates into Symmray MPS."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=12,
        dtype="complex128",
    )
    zz_term = np.diag([1.0, -1.0, -1.0, 1.0]).reshape(2, 2, 2, 2)
    dense_hamiltonian = {(i, i + 1): zz_term for i in range(3)}

    ham = state.require_hamiltonian(hamiltonian=dense_hamiltonian)
    assert all(type(term).__module__.split(".")[0] == "symmray" for term in ham.terms.values())

    evolved = state.time_evolve_mps_optimizer(
        0.01,
        steps=1,
        hamiltonian=dense_hamiltonian,
        chi=4,
        mode="mpo",
        run_kwargs={"progbar": False},
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.L == 4
    assert evolved.tn.max_bond() <= 4
    assert np.isfinite(np.real(evolved.norm()))


@pytest.mark.parametrize("case_name", ["itf_z2", "xy_u1", "fermi_hubbard_u1"])
@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "perm", "svd", "exact"])
def test_symmps_mps_optimizer_3x3_streams_cover_supported_modes(case_name, mode):
    """Explicit 3x3 Symmray streams should run through supported MPS modes."""
    state, hamiltonian, gates, non_unitary, expected_charge = _build_3x3_symmray_mps_case(
        case_name
    )

    assert len(hamiltonian.edges) == 12
    assert len(gates) == 12
    assert state.L == 9
    assert state.overall_charge() == expected_charge
    valid_gate_shapes = {(2, 2, 2, 2), (4, 4, 4, 4)}
    assert all(term.shape in valid_gate_shapes for term in hamiltonian.terms.values())
    assert all(gate.shape in valid_gate_shapes for gate, _ in gates)

    opt = pepsy.MpsOptimizer(state.tn.copy(), gates, chi=4, mode=mode)
    run_kwargs = {
        "progbar": False,
        "cutoff": 1.0e-10,
        "n_iter": 4,
    }
    if non_unitary and mode != "exact":
        run_kwargs.update(
            {
                "non_unitary": True,
                "normalize_every": 1,
                "normalize_final": True,
            }
        )

    out = opt.run(**run_kwargs)

    assert _all_tensor_data_symmray(out)
    assert _finite_double_layer_norm(out)

    if mode == "exact":
        assert len(out.tensors) == 1
    else:
        assert out.L == 9
        assert out.max_bond() <= 4

    if non_unitary and mode != "exact":
        events = opt.get_normalizations()
        raw_norm = _raw_mps_norm(out)
        assert len(events) == len(gates)
        assert all(event["method"] == "canonical_center" for event in events)
        assert all(event["reason"] == "compression" for event in events)
        assert all(len(event["sites"]) == 1 for event in events)
        assert all(np.isfinite(event["log10_scale"]) for event in events)
        assert np.isfinite(np.real(raw_norm))
        assert np.real(raw_norm) > 0.0
    else:
        assert opt.get_normalizations() == []


def test_symmps_mps_optimizer_symmray_dmrg_grows_with_native_two_site_fit():
    """Native two-site FIT should replace dense-style Symmray bond padding."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=46,
        dtype="complex128",
    )
    nearest_ham = SymHamiltonian.from_edges(
        "itf",
        "Z2",
        [(0, 1)],
        jx=-1.0,
        hz=-0.5,
    )
    nonlocal_ham = SymHamiltonian.from_edges(
        "itf",
        "Z2",
        [(0, 2)],
        jx=-1.0,
        hz=-0.5,
    )

    nearest_gates = nearest_ham.gate_stream(0.001)
    nonlocal_gates = nonlocal_ham.gate_stream(0.001)

    for mode, gates in [
        ("mpo", nearest_gates),
        ("mpo", nonlocal_gates),
        ("svd", nonlocal_gates),
    ]:
        out = pepsy.MpsOptimizer(
            state.tn.copy(),
            gates,
            chi=2,
            mode=mode,
        ).run(progbar=False, cutoff=1.0e-10)
        assert out.L == 4
        assert out.max_bond() <= 2

    grown = pepsy.MpsOptimizer(
        state.tn.copy(),
        nearest_gates,
        chi=4,
        mode="dmrg",
    ).run(progbar=False, fit_block_size=2)
    assert _all_tensor_data_symmray(grown)
    assert grown.max_bond() <= 4
    assert _finite_double_layer_norm(grown)

    with pytest.raises(ValueError, match="fit_target_strategy='layered'"):
        pepsy.MpsOptimizer(
            state.tn.copy(),
            nearest_gates,
            chi=4,
            mode="dmrg",
        ).run(progbar=False, fit_target_strategy="layered")

    with pytest.raises(ValueError, match="fit_block_size=2"):
        pepsy.MpsOptimizer(
            state.tn.copy(),
            nonlocal_gates,
            chi=4,
            mode="dmrg",
        ).run(progbar=False, fit_block_size=1)


@pytest.mark.parametrize(
    ("model", "symmetry", "site_charge", "hamiltonian_kwargs"),
    [
        (
            "heisenberg",
            "U1",
            site_charge_from_occupations([1, 0, 1]),
            {"j": 1.0},
        ),
        (
            "fermi_hubbard_u1u1",
            "U1U1",
            site_charge_from_occupations([(1, 0), (0, 1), (1, 0)]),
            {"t": 1.0, "U": 2.0, "mu": 0.0},
        ),
    ],
)
@pytest.mark.parametrize(
    ("mode", "expected_blocks"),
    [
        ("dmrg1", [1, 1, 1]),
        ("dmrg2", [2, 2, 1]),
        ("dmrg3", [3, 3, 2]),
    ],
)
def test_symmps_mps_optimizer_dmrg_aliases_preserve_native_rank_schedule(
    model,
    symmetry,
    site_charge,
    hamiltonian_kwargs,
    mode,
    expected_blocks,
):
    """Native U1/U1U1 aliases follow their schedules without densifying."""
    state = SymMPS.random_unitary_for_model(
        model,
        3,
        bond_dim=4,
        site_charge=site_charge,
        seed=123,
        dtype="complex128",
        rounds=8,
    )
    hamiltonian = SymHamiltonian.from_edges(
        model,
        symmetry,
        [(0, 2)],
        **hamiltonian_kwargs,
    )
    optimizer = pepsy.MpsOptimizer(
        state.tn.copy(),
        hamiltonian.gate_stream(0.001),
        chi=4,
        mode=mode,
    )

    out = optimizer.run(
        progbar=False,
        n_iter=3,
        fit_rtol=None,
        timing=True,
    )

    actual_blocks = [
        record["block_size"]
        for record in optimizer.get_run_timing()["fit_steps"]
    ]
    assert actual_blocks == expected_blocks
    assert out.max_bond() <= 4
    assert _all_tensor_data_symmray(out)
    assert _finite_double_layer_norm(out)


@pytest.mark.parametrize("mode", ["mpo", "svd"])
def test_symmps_mps_optimizer_symmray_auto_swap_runs(mode):
    """Symmray auto-swap fallbacks should run without extra diagnostics."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=47,
        dtype="complex128",
    )
    hamiltonian = SymHamiltonian.from_edges(
        "itf",
        "Z2",
        [(0, 2)],
        jx=-1.0,
        hz=-0.5,
    )

    opt = pepsy.MpsOptimizer(
        state.tn.copy(),
        hamiltonian.gate_stream(0.001),
        chi=2,
        mode=mode,
    )
    out = opt.run(
        progbar=False,
        cutoff=1.0e-10,
    )

    assert out.L == 4
    assert out.max_bond() <= 2


def test_sympeps_tfim_builds_z2_terms_and_step():
    """SymPEPS should build Z2 TFIM terms on a square grid."""
    state = SymPEPS.for_model(
        "itf",
        2,
        2,
        bond_dim=2,
        seed=3,
        dtype="complex128",
    )

    ham = state.build_hamiltonian(jx=-1.0, hz=-0.5)

    assert state.symmetry == "Z2"
    assert not state.fermionic
    assert state.Lx == 2
    assert state.Ly == 2
    assert len(ham.terms) == 4
    assert all(term.shape == (2, 2, 2, 2) for term in ham.terms.values())

    state.time_evolve(
        0.005,
        steps=1,
        hamiltonian=ham,
        imaginary=False,
        max_bond=4,
        normalize=False,
    )

    assert state.tn.max_bond() <= 4
    assert np.isfinite(np.real(state.norm()))


def test_sympeps_measures_dense_generic_observables_and_parity():
    """SymPEPS.measure should use quimb PEPS boundary contraction."""
    charges = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        bond_dim=2,
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations(charges),
        seed=33,
        dtype="complex128",
    )
    z_op = np.diag([1.0, -1.0])

    bdy = {}
    measured = state.measure(
        z_op,
        where=(0, 0),
        contraction_opt="auto-hq",
        chi=4,
        bdy=bdy,
        progress=False,
    )
    exact = pepsy.measure_obs(
        state.tn,
        state.operator_from_dense(z_op),
        where=(0, 0),
        ind_id=state.site_ind_id,
        contraction_opt="auto-hq",
    )

    assert state.site_charges() == charges
    assert state.overall_parity() == 1
    assert "plaquette_envs" in bdy
    assert "plaquette_map" in bdy
    assert measured == pytest.approx(exact)


def test_sympeps_measure_requires_chi_without_boundary_holder():
    """Quimb PEPS boundary measurement should make chi selection explicit."""
    state = SymPEPS.for_model("itf", 2, 2, bond_dim=2, seed=34, dtype="complex128")
    z_op = np.diag([1.0, -1.0])

    with pytest.raises(ValueError, match="Provide chi"):
        state.measure(z_op, where=(0, 0), progress=False)


def test_sympeps_measure_delegates_to_quimb_boundary_modes():
    """Quimb MPS/projector boundary modes and CTMRG should accept Symmray data."""
    state = SymPEPS.for_model("itf", 3, 3, bond_dim=2, seed=44, dtype="complex128")
    z_op = np.diag([1.0, -1.0])
    z_sym = state.operator_from_dense(z_op)

    direct_quimb = state.tn.compute_local_expectation(
        {(1, 1): z_sym},
        max_bond=8,
        normalized=True,
        mode="mps",
        contract_optimize="auto-hq",
    )
    wrapped_mps = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="mps",
        contraction_opt="auto-hq",
        progress=False,
    )
    wrapped_projector = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="projector",
        contraction_opt="auto-hq",
        progress=False,
    )
    wrapped_ctmrg_alias = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="ctmrg",
        contraction_opt="auto-hq",
        progress=False,
    )

    norm = state.tn.make_norm()
    exact_norm = norm.contract(all, optimize="auto-hq")
    ctmrg_norm = norm.contract_ctmrg(
        max_bond=8,
        mode="projector",
        final_contract=True,
        final_contract_opts={"optimize": "auto-hq"},
        progbar=False,
    )

    assert wrapped_mps == pytest.approx(direct_quimb)
    assert wrapped_projector == pytest.approx(direct_quimb)
    assert wrapped_ctmrg_alias == pytest.approx(direct_quimb)
    assert ctmrg_norm == pytest.approx(exact_norm)


def test_native_fermionic_ctmrg_matches_exact_on_small_double_layer():
    """Native fermionic CTMRG should remain finite on a small U1U1 PEPS."""
    site_charge = site_charge_from_occupations(
        {
            (0, 0): (1, 0),
            (0, 1): (0, 1),
            (1, 0): (0, 1),
            (1, 1): (1, 0),
            (2, 0): (1, 0),
            (2, 1): (0, 1),
        }
    )
    state = SymPEPS.random(
        3,
        2,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge,
        bond_dim=2,
        seed=74,
        dtype="complex128",
    )
    norm = state.tn.make_norm()
    exact = norm.contract(all, optimize="auto-hq")
    ctmrg = pepsy.contract_flat(
        norm,
        chi=2,
        method="ctmrg",
        progress=False,
        cutoff=1.0e-10,
    )

    assert np.isfinite(ctmrg)
    assert ctmrg == pytest.approx(exact, rel=1.0e-6, abs=1.0e-12)


def test_native_ctmrg_rejects_nonfinite_squared_environment_before_eigh():
    """Non-finite native environment blocks must stop before factorization."""
    import quimb.tensor.tensor_core as qtc

    site_charge = site_charge_from_occupations(
        {
            (0, 0): (1, 0),
            (0, 1): (0, 1),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
        }
    )
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge,
        bond_dim=2,
        seed=75,
        dtype="complex128",
    )
    environment = next(iter(state.peps.tensor_map.values())).data.copy()
    sector = next(iter(environment.blocks))
    environment.set_block(
        sector,
        np.full_like(environment.get_block(sector), np.nan),
    )

    with pepsy.boundary.metrics.quimb_ctmrg_projector_compat():
        with pytest.raises(FloatingPointError, match="squared environment"):
            qtc.squared_op_to_reduced_factor(
                environment,
                environment.shape[0],
                environment.shape[0],
            )


def test_sympeps_gate_stream_runs_pepsy_gate_and_gate_simple():
    """SymPEPS gate streams should work with PEPSY gate wrappers."""
    state = SymPEPS.for_model(
        "heisenberg",
        2,
        2,
        bond_dim=2,
        seed=9,
        dtype="complex128",
    )
    ham = state.build_hamiltonian()
    gates = ham.gate_stream(0.005)

    out_gate = state.copy().apply_gates(
        gates,
        method="gate",
        max_bond=4,
        cutoff=1e-10,
    )
    gauges = {}
    out_simple = state.copy().apply_gates(
        gates,
        method="simple",
        gauges=gauges,
        max_bond=4,
        cutoff=1e-10,
    )

    assert out_gate.tn.max_bond() <= 4
    assert out_simple.tn.max_bond() <= 4
    assert len(gauges) > 0
    assert np.isfinite(np.real(out_gate.norm()))
    assert np.isfinite(np.real(out_simple.norm()))


def test_sympeps_gate_stream_runs_loop_cluster_method():
    """Dense SymPEPS wrappers should expose the loop-cluster update path."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    state = SymPEPS(
        peps=qtn.PEPS.rand(
            2,
            2,
            bond_dim=2,
            phys_dim=2,
            dtype="complex128",
            seed=19,
        ),
        symmetry="dense",
        edges=tuple(qtn.edges_2d_square(2, 2)),
        site_ind_id="k{},{}",
    )
    gates = (
        (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=complex,
            ),
            ((0, 0), (0, 1)),
        ),
    )
    gauges = {}

    out = state.copy().apply_gates(
        gates,
        method="loop_cluster",
        gauges=gauges,
        max_bond=2,
        gate_kwargs={
            "max_loop_size": 0,
            "regauge_opts": {"max_iterations": 4, "tol": 0.0},
            "als_opts": {"max_iterations": 8, "rcond": 1e-11},
        },
    )

    assert out.tn.max_bond() <= 2
    assert out.gauges is gauges
    assert len(gauges) > 0
    assert np.isfinite(np.real(out.norm()))


def test_sympeps_loop_cluster_rejects_block_sparse_tensors_cleanly():
    """Symmray block-sparse PEPS needs a later symmetry-aware reduced update."""
    state = SymPEPS.for_model(
        "heisenberg",
        2,
        2,
        bond_dim=2,
        seed=19,
        dtype="complex128",
    )
    gates = state.build_hamiltonian().gate_stream(0.002)

    with pytest.raises(NotImplementedError, match="dense PEPS tensor arrays"):
        state.copy().apply_gates(
            gates[:1],
            method="loop_cluster",
            gauges={},
            gate_kwargs={
                "regauge_opts": {"max_iterations": 1, "tol": 0.0},
            },
        )


@pytest.mark.parametrize("case_name", ["itf_z2", "xy_u1", "fermi_hubbard_u1"])
@pytest.mark.parametrize("method", ["gate", "simple"])
def test_sympeps_gate_wrappers_3x3_streams_cover_symmetries(case_name, method):
    """PEPSY gate wrappers should handle 3x3 Symmray PEPS gate streams."""
    state, hamiltonian, gates, expected_charge = _build_3x3_symmray_peps_case(
        case_name
    )

    assert len(hamiltonian.edges) == 12
    assert len(gates) == 12
    assert state.Lx == 3
    assert state.Ly == 3
    assert state.overall_charge() == expected_charge

    if method == "gate":
        out = gate(
            state.tn.copy(),
            gates,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
    else:
        gauges = {}
        out = gate_simple(
            state.tn.copy(),
            gates,
            gauges=gauges,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
        assert len(gauges) == len(gates)

    assert out.Lx == 3
    assert out.Ly == 3
    assert out.max_bond() <= 4
    assert _all_tensor_data_symmray(out)


@pytest.mark.parametrize("case_name", ["itf_z2", "xy_u1", "fermi_hubbard_u1"])
@pytest.mark.parametrize("method", ["gate", "simple"])
def test_sympeps_gate_wrappers_route_nonlocal_symmray_swaps(case_name, method):
    """Internal routed SWAPs should be Symmray arrays for nonlocal PEPS gates."""
    nonlocal_edge = (((0, 0), (2, 2)),)
    state, _, gates, _ = _build_3x3_symmray_peps_case(
        case_name,
        edges=nonlocal_edge,
    )

    if method == "gate":
        out = gate(
            state.tn.copy(),
            gates,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
    else:
        gauges = {}
        out = gate_simple(
            state.tn.copy(),
            gates,
            gauges=gauges,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
        assert len(gauges) > 0

    assert out.max_bond() <= 4
    assert _all_tensor_data_symmray(out)


@pytest.mark.parametrize("method", ["gate", "simple"])
def test_sympeps_gate_wrappers_route_nonlocal_u1u1_swaps(method):
    """Product-symmetry PEPS gates should route through neutral SWAP sectors."""
    charges = {(i, j): (1, 1) for i in range(3) for j in range(3)}
    state = SymPEPS.random(
        3,
        3,
        symmetry="U1U1",
        phys_dim={(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        fermionic=True,
        site_charge=site_charge_from_occupations(charges),
        bond_dim=2,
        seed=54,
        dtype="complex128",
    )
    hamiltonian = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [((0, 0), (2, 2))],
        t=1.0,
        U=2.0,
        mu=0.1,
    )
    gates = hamiltonian.gate_stream(0.0005, imaginary=True)

    if method == "gate":
        out = gate(
            state.tn.copy(),
            gates,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
    else:
        out = gate_simple(
            state.tn.copy(),
            gates,
            gauges={},
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )

    assert out.max_bond() <= 4
    assert _all_tensor_data_symmray(out)


@pytest.mark.parametrize("symmetry", ["U1", "U1U1", "Z2"])
def test_sympeps_gate_simple_explicit_reduce_split_fermion_symmetries(symmetry):
    """Explicit reduce-split should work for native fermionic PEPS sectors."""
    if symmetry == "U1U1":
        phys_dim = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        charges = {(i, j): (1, 1) for i in range(3) for j in range(3)}
        model = "fermi_hubbard_u1u1"
    elif symmetry == "Z2":
        phys_dim = {0: 2, 1: 2}
        charges = {(i, j): 1 for i in range(3) for j in range(3)}
        model = "fermi_hubbard"
    else:
        phys_dim = {0: 1, 1: 2, 2: 1}
        charges = {(i, j): 1 for i in range(3) for j in range(3)}
        model = "fermi_hubbard"

    state = SymPEPS.random(
        3,
        3,
        symmetry=symmetry,
        phys_dim=phys_dim,
        fermionic=True,
        site_charge=site_charge_from_occupations(charges),
        bond_dim=2,
        seed=55,
        dtype="complex128",
    )
    hamiltonian = SymHamiltonian.from_edges(
        model,
        symmetry,
        [((0, 0), (2, 2))],
        t=1.0,
        U=2.0,
        mu=0.1,
    )

    out = gate_simple(
        state.tn.copy(),
        hamiltonian.gate_stream(0.0005, imaginary=True),
        gauges={},
        max_bond=4,
        cutoff=1.0e-10,
        contract="reduce-split",
        inplace=False,
    )

    assert out.max_bond() <= 4
    assert _all_tensor_data_symmray(out)


def test_sympeps_gate_method_preserves_pepsy_gate_contract_default(monkeypatch):
    """SymPEPS method='gate' should not override pepsy.gate's default."""
    state = SymPEPS.for_model(
        "heisenberg",
        2,
        2,
        bond_dim=2,
        seed=12,
        dtype="complex128",
    )
    calls = []

    def _fake_gate(tn, gates, **kwargs):
        calls.append((gates, kwargs.copy()))
        return tn

    monkeypatch.setattr("pepsy.operators.gate", _fake_gate)

    out = state.copy().apply_gates(
        ((np.eye(2, dtype=np.complex128), ((0, 0),)),),
        method="gate",
    )

    assert out.tn is not None
    assert "contract" not in calls[0][1]


def test_sympeps_raw_pepsy_gate_functions_accept_symmray_streams():
    """The plain gate functions should accept a SymGateStream directly."""
    state = SymPEPS.for_model("itf", 2, 2, bond_dim=2, seed=10, dtype="complex128")
    gates = state.build_hamiltonian().gate_stream(0.005)
    gauges = {}

    out_gate = gate(state.tn.copy(), gates, max_bond=4, cutoff=1e-10, inplace=False)
    out_simple = gate_simple(
        state.tn.copy(),
        gates,
        gauges=gauges,
        max_bond=4,
        cutoff=1e-10,
        inplace=False,
    )

    assert out_gate.max_bond() <= 4
    assert out_simple.max_bond() <= 4
    assert len(gauges) > 0


def test_symmetric_classes_are_top_level_lazy_exports():
    """Top-level pepsy exports should resolve to the tensor namespace classes."""
    assert pepsy.SymHamiltonian is SymHamiltonian
    assert pepsy.SymGateStream is SymGateStream
    assert pepsy.SymMPS is SymMPS
    assert pepsy.SymPEPS is SymPEPS
    assert pepsy.default_physical_sectors is default_physical_sectors
    assert pepsy.symm_operator_from_dense is symm_operator_from_dense


def _dense_jw_fermi_hubbard(L, edges, *, t=1.0, U=4.0, mu=0.3):
    """Independent dense spinful Fermi-Hubbard Hamiltonian via Jordan-Wigner.

    Built from scratch (2L spin-orbitals, explicit Z strings) with no reference
    to Symmray, so it is an independent oracle for the U1U1 MPO spectrum.
    """
    n = 2 * L
    eye = np.eye(2)
    zed = np.array([[1.0, 0.0], [0.0, -1.0]])
    lower = np.array([[0.0, 1.0], [0.0, 0.0]])

    def annihilate(mode):
        mats = [zed] * mode + [lower] + [eye] * (n - mode - 1)
        out = mats[0]
        for mat in mats[1:]:
            out = np.kron(out, mat)
        return out

    def mode(site, spin):
        return 2 * site + spin  # spin 0 = up, 1 = down

    ham = np.zeros((2 ** n, 2 ** n))
    for i, j in edges:
        for spin in (0, 1):
            hop = annihilate(mode(i, spin)).conj().T @ annihilate(mode(j, spin))
            ham += -t * (hop + hop.conj().T)
    for site in range(L):
        num_up = annihilate(mode(site, 0)).conj().T @ annihilate(mode(site, 0))
        num_dn = annihilate(mode(site, 1)).conj().T @ annihilate(mode(site, 1))
        ham += U * (num_up @ num_dn) - mu * (num_up + num_dn)
    return ham


def _mpo_to_dense_matrix(mpo, L):
    tensor = mpo.copy().contract(all)
    fused = tensor.to_dense(
        [f"k{i}" for i in range(L)], [f"b{i}" for i in range(L)]
    )
    if hasattr(fused, "to_dense"):
        fused = fused.to_dense()
    return np.asarray(fused)


@pytest.mark.parametrize(
    "L,edges",
    [
        (3, [(0, 1), (1, 2)]),          # nearest-neighbour chain
        (3, [(0, 2)]),                  # single long-range (parity string)
        (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),  # periodic wrap -> crossing term
    ],
)
def test_fermi_hubbard_u1u1_mpo_matches_dense_jw_spectrum(L, edges):
    """The U1U1 FH MPO spectrum must match an independent dense JW Hamiltonian."""
    t, U, mu = 1.0, 4.0, 0.3
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1", "U1U1", edges, t=t, U=U, mu=mu
    )
    mpo = ham.to_mpo(L=L, compress=False)

    matrix = _mpo_to_dense_matrix(mpo, L)
    assert np.max(np.abs(matrix - matrix.conj().T)) < 1e-10  # Hermitian

    reference = _dense_jw_fermi_hubbard(L, edges, t=t, U=U, mu=mu)
    spec_mpo = np.sort(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).real)
    spec_ref = np.sort(np.linalg.eigvalsh(reference).real)
    np.testing.assert_allclose(spec_mpo, spec_ref, atol=1e-9)


@pytest.mark.parametrize(
    "L,edges",
    [
        (2, [(0, 1)]),
        (3, [(0, 2)]),
        (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
    ],
)
def test_fermi_hubbard_total_u1_mpo_matches_dense_jw_spectrum(L, edges):
    """The total-U1 spinful FH MPO should match dense JW exactly."""
    t, U, mu = 1.0, 4.0, 0.3
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard", "U1", edges, t=t, U=U, mu=mu
    )
    mpo = ham.to_mpo(L=L, compress=False)

    matrix = _mpo_to_dense_matrix(mpo, L)
    assert np.max(np.abs(matrix - matrix.conj().T)) < 1e-10

    reference = _dense_jw_fermi_hubbard(L, edges, t=t, U=U, mu=mu)
    spec_mpo = np.sort(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).real)
    spec_ref = np.sort(np.linalg.eigvalsh(reference).real)
    np.testing.assert_allclose(spec_mpo, spec_ref, atol=1e-9)


def test_fermi_hubbard_u1u1_mpo_matches_terms_on_2d_snake_lattice():
    """MPO expectation must match the term energy on the notebook's 2D snake map.

    The 4x3 periodic square lattice mapped through a snake OneDMap produces many
    crossing long-range hopping channels -- exactly the case that earlier sign
    conventions got wrong while single-edge tests still passed.
    """
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    Lx, Ly = 4, 3
    L = Lx * Ly
    mapper = OneDMap(Lx, Ly, mode="snake")
    _, coo2idx = mapper.build()
    half = {
        (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
        for x in range(Lx)
        for y in range(Ly)
    }
    site_charge = site_charge_from_occupations(
        {coo2idx[c]: q for c, q in half.items()}
    )

    for cyclic in (False, True):
        edges = [
            (coo2idx[a], coo2idx[b])
            for a, b in qtn.edges_2d_square(Lx, Ly, cyclic=cyclic)
        ]
        state = SymMPS.random(
            L,
            symmetry="U1U1",
            fermionic=True,
            phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
            site_charge=site_charge,
            bond_dim=8,
            seed=13,
            dtype="complex128",
        )
        ham = SymHamiltonian.from_edges(
            "fermi_hubbard_u1u1", "U1U1", edges, t=1.0, U=8.0, mu=0.0
        )
        mpo = ham.to_mpo(L=L, compress=False)

        mpo_energy = pepsy.MpsEnergyOptimizer(
            state,
            mpo,
            energy_per_site=False,
            real=False,
            allow_encoding_conversion=True,
            contraction_opt="greedy",
        ).energy().energy
        term_energy = pepsy.MpsEnergyOptimizer(
            state,
            ham.terms,
            energy_per_site=False,
            real=False,
            allow_encoding_conversion=True,
            contraction_opt="greedy",
        ).energy().energy

        assert complex(mpo_energy) == pytest.approx(complex(term_energy))
