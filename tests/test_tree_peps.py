"""Focused tests for the initial tree-embedded PEPS state API."""

import numpy as np
import pytest

from pepsy.optimizers import (
    TreePeps,
    TreePepsGeometry,
    TreePepsLayoutFinder,
    TreePepsPlan,
    TreePepo,
    TreePepsOptimizer,
)
from pepsy.optimizers.tree_peps._compression import iter_tree_compression_order

pytestmark = pytest.mark.smoke


def test_tree_peps_plan_keeps_coordinate_and_logical_views():
    plan = TreePepsPlan.from_shape((2, 3), order="row-major")

    assert isinstance(plan, TreePepsGeometry)
    assert plan.coordinates[1] == (0, 1)
    assert plan.logical_site((1, 2)) == 5
    assert len(plan.tree_edges) == plan.size - 1
    assert plan.max_degree == 3
    assert plan.max_tensor_rank == 4
    assert plan.is_branching
    assert not plan.is_mps_topology
    assert all(edge in plan.lattice_edges for edge in plan.tree_edges)


def test_tree_peps_plan_traversal_seeds_build_legal_trees():
    snake = TreePepsPlan.from_shape((4, 4))
    row_major = TreePepsPlan.from_shape((4, 4), tree_order="row-major")
    col_major = TreePepsPlan.from_shape((4, 4), tree_order="col-major")
    hilbert = TreePepsPlan.from_shape((4, 4), tree_order="hilbert")
    inside_out = TreePepsPlan.from_shape((4, 4), tree_order="inside-out")

    assert snake.max_degree == 3
    assert hilbert.max_degree == 3
    assert row_major.max_degree == 3
    assert col_major.max_degree == 3
    assert inside_out.coordinate(inside_out.root) == (1, 1)
    assert inside_out.max_degree == 3
    assert len({
        plan.tree_edges
        for plan in (snake, row_major, col_major, hilbert, inside_out)
    }) == 5
    for plan in (row_major, col_major, hilbert, inside_out):
        assert len(plan.tree_edges) == plan.size - 1
        assert set(plan.tree_edges).issubset(set(plan.lattice_edges))
        assert plan.is_connected(range(plan.size))


@pytest.mark.parametrize("mode", ("span-up", "span-down", "span-out", "span-middle"))
def test_tree_peps_span_map_modes_are_canonical_and_bounded(mode):
    plan = TreePepsPlan.from_shape((4, 4), map_mode=mode)

    assert plan.map_mode == mode
    assert plan.tree_order == mode
    assert len(plan.tree_edges) == plan.size - 1
    assert plan.max_degree <= (4 if mode == "span-middle" else 3)
    assert plan.is_connected(range(plan.size))


def test_tree_peps_span_out_alias_and_state_operator_views():
    alias = TreePepsPlan.from_shape((3, 3), map_mode="inside-out")
    state = TreePeps.from_plan(alias)
    operator = TreePepo.identity(alias)

    assert alias.map_mode == "span-out"
    assert state.map_mode == "span-out"
    assert operator.map_mode == "span-out"


def test_tree_peps_span_map_modes_generalize_to_3d():
    for mode in ("span-up", "span-down", "span-out", "span-middle"):
        plan = TreePepsPlan.from_shape((3, 3, 2), map_mode=mode)
        assert plan.map_mode == mode
        assert plan.max_degree <= (4 if mode == "span-middle" else 3)
        assert plan.is_connected(range(plan.size))


def test_tree_peps_span_middle_is_central_backbone_with_axial_chains():
    """span-middle has a horizontal backbone and one vertical chain per site."""
    plan = TreePepsPlan.from_shape((5, 5), map_mode="span-middle")
    middle = (plan.shape[0] - 1) // 2

    def edge(coord0, coord1):
        return tuple(sorted((plan.logical_site(coord0), plan.logical_site(coord1))))

    expected = {
        edge((middle, y), (middle, y + 1))
        for y in range(plan.shape[1] - 1)
    }
    expected.update(
        edge((x, y), (x + 1, y))
        for y in range(plan.shape[1])
        for x in range(plan.shape[0] - 1)
    )
    assert set(plan.tree_edges) == expected
    assert plan.max_degree == 4
    for y in range(1, plan.shape[1] - 1):
        assert len(plan.neighbors((middle, y))) == 4
    for x in range(plan.shape[0]):
        for y in range(plan.shape[1]):
            if x != middle and 0 < x < plan.shape[0] - 1:
                assert len(plan.neighbors((x, y))) == 2

    state = TreePeps.from_plan(plan)
    assert state.max_virtual_degree == 4
    assert state.max_rank == 5
    assert state.validate()


def test_tree_peps_layout_finder_allows_span_middle_degree_four():
    base = TreePepsPlan.from_shape((4, 4))
    finder = TreePepsLayoutFinder(
        base,
        seed_modes=("span-middle",),
        max_iter=0,
    )

    assert finder.max_virtual_degree == 4
    middle_edges = finder._fixed_seed("span-middle")
    middle_plan = finder._build_plan(middle_edges, tree_order="span-middle")
    assert middle_plan.max_degree == 4


def test_tree_peps_row_and_column_major_are_oriented_combs():
    row_major = TreePepsPlan.from_shape(
        (4, 3), order="row-major", tree_order="row-major"
    )
    col_major = TreePepsPlan.from_shape(
        (4, 3), order="row-major", tree_order="col-major"
    )

    def edge(plan, coord0, coord1):
        return tuple(sorted((plan.logical_site(coord0), plan.logical_site(coord1))))

    expected_row = {
        edge(row_major, (x, y), (x + 1, y))
        for y in range(3)
        for x in range(3)
    }
    expected_row.update(
        edge(row_major, (0, y), (0, y + 1)) for y in range(2)
    )
    expected_col = {
        edge(col_major, (x, y), (x, y + 1))
        for x in range(4)
        for y in range(2)
    }
    expected_col.update(
        edge(col_major, (x, 0), (x + 1, 0)) for x in range(3)
    )

    assert set(row_major.tree_edges) == expected_row
    assert set(col_major.tree_edges) == expected_col
    assert row_major.tree_edges != col_major.tree_edges
    assert row_major.max_degree == col_major.max_degree == 3


def test_tree_peps_has_one_physical_leg_and_dual_tags():
    plan = TreePepsPlan.from_shape((2, 2), topology="path")
    state = TreePeps.from_plan(plan)

    assert state.site_tag(0, 1) == "I0,1"
    assert state.logical_site_tag(1) == "I1"
    assert state.site_ind(0, 1) == "k0,1"
    assert state.site_ind_1d(1) == "k0,1"
    assert state.validate()
    assert state.to_dense().shape == (2, 2, 2, 2)
    assert np.allclose(state.norm(), 1.0)
    assert state.max_virtual_degree <= 3
    assert state.max_rank <= 4
    assert state.rank == state.max_rank
    assert state.max_tensor_rank == state.max_rank
    assert state.nsites == state.nqubits == plan.size
    assert state.root == plan.root
    assert state.top_arity == len(plan.children[plan.root])
    assert state.tensor_rank(0) == 1 + len(plan.neighbors(0))
    assert state.max_bond() == 1
    assert state.bond_report()["n_bonds"] == plan.size - 1
    assert state.to_statevector().shape == (2**plan.size,)


def test_tree_peps_tree_topology_and_batch_readout_match_ttn_names():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.from_plan(plan)
    z = np.diag([1.0, -1.0])

    assert state.topology == "tree"
    assert state.is_branching
    assert not state.is_mps_topology
    assert state.max_rank == 4
    assert state.node_path(0, 5) == state.path(0, 5)
    assert state.tree_distance(0, 5) == len(state.path(0, 5)) - 1
    assert state.parent(1) == 0
    assert state.children(0) == plan.children[0]
    assert state.is_leaf(5)
    assert state.subtree_span((0, 5)) == plan.subtree_span((0, 5))
    values = state.local_expectations({0: z, (0, 1): np.eye(4)})
    assert np.allclose(values[0], 1.0)
    assert np.allclose(values[(0, 1)], 1.0)


def test_tree_peps_normalize_preserves_canonical_tree_metadata():
    state = TreePeps.rand(
        TreePepsPlan.from_shape((2, 2), topology="path"), seed=19
    )
    old_norm = state.normalize()

    assert float(abs(old_norm)) > 1.0
    assert np.allclose(state.norm(), 1.0)
    assert state.validate(check_canonical=True)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float64, 1.0e-12),
        (np.complex128, 1.0e-12),
        (np.float32, 1.0e-6),
        (np.complex64, 1.0e-6),
    ],
)
def test_tree_peps_optimizer_auto_cutoff_matches_dtype_policy(dtype, expected):
    """TreePepsOptimizer resolves its default from the live state dtype."""
    plan = TreePepsPlan.from_shape((2, 2), topology="path")
    state = TreePeps.from_plan(plan, dtype=dtype)
    optimizer = TreePepsOptimizer(state, plan=plan, run=False)

    assert optimizer.cutoff == pytest.approx(expected)


def test_tree_peps_canonical_center_norm_matches_dense_norm():
    state = TreePeps.rand(
        TreePepsPlan.from_shape((2, 3)), bond_dim=2, seed=23, canonicalize=True
    )
    dense_norm = np.linalg.norm(np.asarray(state.to_statevector()))

    assert np.allclose(state.norm(), dense_norm)
    assert np.allclose(state.norm(squared=True), dense_norm**2)


def test_tree_peps_show_and_canonical_info(capsys):
    state = TreePeps.rand(
        TreePepsPlan.from_shape((2, 2), topology="path"), bond_dim=2, seed=11
    )

    assert state.show(color=False) is None
    output = capsys.readouterr().out
    assert "●" in output
    assert "━━━━" in output

    info_c = {}
    state.canonicalize(center=0, info_c=info_c)
    assert info_c["cur_orthog"] == (0, 0)
    assert info_c["canonical_region"] == frozenset({0})
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True)


def test_tree_peps_exponent_survives_quimb_interoperability_and_show(capsys):
    """Dense readout and visualization respect Quimb's stored exponent."""
    state = TreePeps.from_plan(
        TreePepsPlan.from_shape((2, 2), topology="path")
    )
    state.exponent = 2.0
    copied = state.copy()

    assert copied.exponent == 2.0
    assert np.linalg.norm(np.asarray(state.to_dense().data)) == pytest.approx(100.0)
    assert state.show(color=False) is None
    assert "●" in capsys.readouterr().out


def test_tree_peps_exact_readout_canonicalization_and_compression():
    plan = TreePepsPlan.from_shape((2, 2), topology="path")
    state = TreePeps.rand(plan, bond_dim=2, seed=11)
    identity = np.eye(2)

    assert np.allclose(state.local_expectation(identity, 0), 1.0)
    canonical = state.canonize_to(0)
    assert canonical.orthogonality_center == 0
    assert canonical.is_canonical_form()
    assert canonical.validate()
    compressed = canonical.compress_edge(0, 1, max_bond=1)
    assert compressed.orthogonality_center == 1
    assert compressed.is_canonical_form()
    assert compressed.validate()


def test_tree_peps_moves_from_canonical_region_using_left_inds():
    plan = TreePepsPlan.from_shape((2, 3))
    state = TreePeps.rand(plan, bond_dim=2, seed=17)

    state.canonize_subtree([0, 1], inplace=True)
    assert state.canonical_region == frozenset({0, 1})
    assert state.is_subtree_canonical_form()
    assert state.isometry_map()[2] == 1

    state.shift_orthogonality_center(5)

    assert state.orthogonality_center == 5
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True)

    state.compress(center=5, max_bond=1)
    assert state.orthogonality_center == 5
    assert state.is_canonical_form()
    assert state.validate(check_canonical=True)


def test_tree_peps_rank_compression_batches_validation_and_keeps_layout():
    """Full native compression chooses legal branches and validates once."""
    plan = TreePepsPlan.from_shape((2, 3), tree_order="row-major")
    state = TreePeps.rand(plan, bond_dim=2, seed=23, canonicalize=True)
    validate_calls = []
    original_validate = state.validate

    def capture_validate(*args, **kwargs):
        validate_calls.append(kwargs.get("check_canonical", False))
        return original_validate(*args, **kwargs)

    state.validate = capture_validate
    state.compress(max_bond=1, cutoff=0.0, order="rank")

    assert validate_calls == [True]
    assert state.max_bond() <= 1
    assert state.plan_signature == TreePeps.from_plan(plan).plan_signature
    assert state.is_canonical_form()

    depth = TreePeps.rand(plan, bond_dim=2, seed=23, canonicalize=True)
    depth.compress(max_bond=1, cutoff=0.0, order="depth")
    assert depth.max_bond() <= 1
    assert depth.is_canonical_form()


def test_tree_rank_schedule_recomputes_after_each_reduction():
    """Rank ordering must read dimensions changed by the previous SVD."""
    class FakeTensor:
        def __init__(self, inds, dims):
            self.inds = tuple(inds)
            self.dims = dims

        def ind_size(self, ind):
            return self.dims[ind]

    plan = TreePepsPlan.from_shape((2, 3), tree_order="row-major")
    dims = {f"p{site}": 2 for site in range(plan.size)}
    dims.update(
        {
            f"b{min(site0, site1)}_{max(site0, site1)}": 4
            for site0, site1 in plan.tree_edges
        }
    )
    tensors = {
        site: FakeTensor(
            [f"p{site}"]
            + [
                f"b{min(site, neighbor)}_{max(site, neighbor)}"
                for neighbor in plan.neighbors(site)
            ],
            dims,
        )
        for site in range(plan.size)
    }

    def bond(site0, site1):
        return f"b{min(site0, site1)}_{max(site0, site1)}"

    schedule = iter_tree_compression_order(
        plan,
        center=plan.root,
        nodes=range(plan.size),
        order="rank",
        tensor_getter=tensors.__getitem__,
        bond_getter=bond,
    )
    assert next(schedule) == (3, 2)
    # A completed compression can change a bond incident on a remaining
    # branch. The next choice must see that live dimension.
    dims["b0_5"] = 100
    assert next(schedule) == (4, 1)


def test_tree_pepo_rank_and_depth_compression_preserve_exact_operator():
    """Both fixed-topology TreePEPO schedules preserve an uncapped operator."""
    plan = TreePepsPlan.from_shape((2, 3), tree_order="row-major")
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    source = TreePepo.from_terms(
        plan,
        {
            (0, 5): np.kron(x, z),
            (1, 4): np.kron(z, x),
        },
    )
    expected = np.asarray(source.to_dense().data)

    for order in ("rank", "depth"):
        compressed = source.copy()
        compressed.compress(max_bond=None, cutoff=0.0, order=order)
        np.testing.assert_allclose(
            np.asarray(compressed.to_dense().data), expected, atol=1e-10, rtol=1e-10
        )
        assert compressed.is_canonical_form()

        bounded = source.copy()
        bounded.compress(max_bond=1, cutoff=0.0, order=order)
        assert bounded.max_bond() <= 1
        assert bounded.is_canonical_form()

    with pytest.raises(ValueError, match="tree compression order"):
        source.copy().compress(max_bond=1, order="invalid")
    with pytest.raises(ValueError, match="tree compression order"):
        TreePeps.rand(plan, bond_dim=2, seed=24).compress(
            max_bond=1, cutoff=0.0, order="invalid"
        )

def test_tree_peps_supports_three_dimensional_coordinate_tags():
    plan = TreePepsPlan.from_shape((2, 1, 2), topology="path")
    state = TreePeps.from_plan(plan)

    assert state.site_tag(1, 0, 1) == "I1,0,1"
    assert state.site_ind(1, 0, 1) == "k1,0,1"
    assert state.logical_site_tag(plan.logical_site((1, 0, 1))) == "I2"
    assert state.z_tag(1) == "Z1"
    assert state.validate()


def test_tree_peps_plan_rejects_non_lattice_or_cyclic_edges():
    with pytest.raises(ValueError, match="subset of the lattice"):
        TreePepsPlan.from_shape((2, 2), tree_edges=[(0, 1), (1, 2), (0, 2)])

    with pytest.raises(ValueError, match="N - 1"):
        TreePepsPlan.from_shape((2, 2), tree_edges=[(0, 1), (1, 2), (2, 3), (3, 0)])


def test_tree_peps_hard_limits_virtual_degree_to_four():
    assert TreePepsPlan.from_shape((2, 3), max_virtual_degree=4)
    with pytest.raises(ValueError, match="at most 4"):
        TreePepsPlan.from_shape((2, 3), max_virtual_degree=5)


def test_tree_peps_requires_explicit_path_topology_for_non_branching_geometry():
    with pytest.raises(ValueError, match="requires at least one site"):
        TreePepsPlan.from_shape((2, 2))

    path = TreePepsPlan.from_shape((2, 2), topology="path")
    assert path.is_mps_topology
    assert not path.is_branching
    assert path.topology == "path"


def test_tree_peps_layout_finder_returns_a_plan_for_all_consumers():
    gate = np.eye(4, dtype=complex)
    finder = TreePepsLayoutFinder(
        (2, 3),
        interactions=[(gate, (0, 5)), (gate, (1, 4))],
        objective="hybrid",
        seed=7,
        max_iter=8,
    )
    plan = finder.run()

    assert isinstance(plan, TreePepsPlan)
    assert plan.max_degree <= 4
    assert finder.plan is plan
    assert finder.report["tree_edges"] == plan.tree_edges
    state = TreePeps.from_plan(plan)
    pepo = TreePepo.from_operator(plan, gate, support=(0, 5))
    optimizer = TreePepsOptimizer(state, plan=plan, chi=None, cutoff=0.0)
    optimizer.apply(pepo)
    assert optimizer.validate(check_canonical=True) is optimizer


def test_tree_peps_layout_finder_compares_fixed_traversal_seeds():
    finder = TreePepsLayoutFinder(
        (4, 4),
        interactions=[(np.eye(4), ((0, 0), (3, 3)))],
        tree_order="inside-out",
        max_iter=0,
    )

    plan = finder.run(refine=False)

    assert plan.coordinate(plan.root) == (1, 1)
    assert finder.report["seed_modes"] == ("inside-out",)
    assert finder.report["n_candidates"] >= 1
    assert finder.report["selected_seed"] in {"source", "inside-out", "refined"}


def test_tree_peps_layout_finder_accepts_canonical_span_map_mode():
    finder = TreePepsLayoutFinder(
        (4, 4),
        interactions=[],
        map_mode="span-out",
        max_iter=0,
    )

    plan = finder.run(refine=False)

    assert finder.map_mode == "span-out"
    assert plan.map_mode == "span-out"
    assert finder.report["map_mode"] == "span-out"


@pytest.mark.parametrize("coarse_grain", ((1, 2), (2, 2)))
def test_tree_peps_layout_finder_accepts_legacy_coarse_map_modes(coarse_grain):
    finder = TreePepsLayoutFinder(
        (4, 4),
        interactions=[],
        map_mode="coarse-alternate-x",
        coarse_grain=coarse_grain,
        max_iter=0,
    )

    plan = finder.run(refine=False)

    assert finder.seed_modes == ("coarse-alternate-x",)
    assert finder.coarse_grain == coarse_grain
    assert plan.map_mode == "coarse-alternate-x"
    assert plan.coarse_grain == coarse_grain
    assert finder.report["coarse_grain"] == coarse_grain
