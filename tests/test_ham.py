"""Regression tests for :mod:`pepsy.operators.hamiltonians` builders."""

import builtins
import inspect

import numpy as np
import pytest
import quimb
import quimb.tensor as qtn

import pepsy as py
from pepsy._internal.cutoff import dtype_auto_cutoff
from pepsy.tensors.core import OneDMap


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float64, 1.0e-12),
        (np.complex128, 1.0e-12),
        (np.float32, 1.0e-6),
        (np.complex64, 1.0e-6),
        (np.float16, 1.0e-3),
    ],
)
def test_auto_cutoff_dtype_policy_is_shared(dtype, expected):
    """All auto-cutoff users share the MPS precision policy."""
    assert dtype_auto_cutoff(dtype) == expected


def test_ham_tn_accepts_shape_alias_for_1d_2d_and_3d_layouts():
    """The builder's geometry spelling matches the higher-order MPO API."""
    chain = py.ham_tn(shape=4)
    assert (chain.Lx, chain.Ly, chain.Lz) == (4, 1, None)
    assert chain.ndim == 2

    square = py.ham_tn(shape=(2, 3))
    assert (square.Lx, square.Ly, square.Lz) == (2, 3, None)
    assert square.L == 6

    cubic = py.ham_tn(shape=(2, 2, 2))
    assert (cubic.Lx, cubic.Ly, cubic.Lz) == (2, 2, 2)
    assert cubic.ndim == 3

    with pytest.raises(TypeError, match="shape conflicts.*Lx"):
        py.ham_tn(shape=(3, 2), Lx=4, Ly=2)
    with pytest.raises(TypeError, match="shape conflicts.*Lz"):
        py.ham_tn(shape=(2, 2), Lx=2, Ly=2, Lz=1)


def test_build_mpo_single_site_term_works_for_ly1():
    """Single-term MPO build should work for 1D (Ly=1) layouts."""
    builder = py.ham_tn(Lx=2, Ly=1, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")

    mpo = builder.build_mpo(
        [
            ((z_op,), ((0, 0),)),
        ],
        compress_each=False,
    )

    assert mpo.L == 2


def test_to_mpo_is_canonical_and_build_mpo_is_compatibility_alias():
    """The conversion spelling is canonical while the old spelling survives."""
    builder = py.ham_tn(Lx=2, Ly=1, data_type="complex128")
    terms = [((quimb.pauli("Z", dtype="complex128"),), (0,))]

    canonical = builder.to_mpo(terms, compress_each=False)
    with pytest.warns(DeprecationWarning, match="use ham_tn.to_mpo"):
        compatibility = builder.build_mpo(terms, compress_each=False)

    assert np.allclose(canonical.to_dense(), compatibility.to_dense())


def test_to_tree_mpo_compiles_terms_without_chain_mpo():
    """Tree conversion returns the native TreePlan operator directly."""
    from pepsy.optimizers import TreeMPO, TreePlan

    plan = TreePlan.from_order([0, 1, 2, 3])
    builder = py.ham_tn(shape=4, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    operator = builder.to_tree_mpo(
        plan,
        [
            ((z_op,), (0,), 0.5),
            ((z_op, z_op), (1, 3), 1.2),
        ],
        cutoff=0.0,
    )

    assert isinstance(operator, TreeMPO)
    assert len(operator.tree_networks) == 1
    assert not hasattr(operator, "Lx")
    assert np.allclose(
        operator.to_dense(),
        0.5 * np.kron(np.kron(np.kron(z_op, np.eye(2)), np.eye(2)), np.eye(2))
        + 1.2 * np.kron(np.kron(np.kron(np.eye(2), z_op), np.eye(2)), z_op),
    )


def test_to_tree_mpo_resolves_coordinates_using_builder_map():
    """Coordinate terms map through the builder when TreePlan has no lattice map."""
    from pepsy.optimizers import TreePlan

    mapper = OneDMap(2, 2, mode="row-major")
    builder = py.ham_tn(shape=(2, 2), mapper=mapper, data_type="complex128")
    plan = TreePlan.from_order([0, 1, 2, 3])
    z_op = quimb.pauli("Z", dtype="complex128")
    by_coordinate = builder.to_tree_mpo(
        plan,
        [((z_op,), ((1, 0),), 1.0)],
        compress=False,
        cutoff=0.0,
    )
    by_logical = builder.to_tree_mpo(
        plan,
        [((z_op,), (2,), 1.0)],
        compress=False,
        cutoff=0.0,
    )

    assert np.allclose(by_coordinate.to_dense(), by_logical.to_dense())


def test_ham_tn_supports_direct_tree_map_mode_conversion():
    """Tree conversions can derive their native plans from one map mode."""
    from pepsy.optimizers import TreeMPO, TreePEPO

    builder = py.ham_tn(
        shape=(2, 3),
        map_mode="row-major",
        data_type="complex128",
    )
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [
        ((z_op,), ((0, 0),), 0.5),
        ((z_op, z_op), ((0, 1), (1, 1)), 1.2),
    ]

    tree_mpo = builder.to_tree_mpo(
        terms,
        map_mode="coarse-alternate-x",
        compress_opts={"order": "depth"},
    )
    tree_pepo = builder.to_tree_pepo(
        terms,
        map_mode="coarse-alternate-x",
        form="left",
        compress_opts={"order": "depth"},
    )

    assert builder.map_mode == "row-major"
    assert isinstance(tree_mpo, TreeMPO)
    assert isinstance(tree_pepo, TreePEPO)
    assert tree_pepo.plan.order == "coarse-alternate-x"
    assert tree_pepo.plan.tree_order == "coarse-alternate-x"


def test_ham_tn_uses_distinct_canonical_tree_and_peps_map_modes():
    from pepsy.optimizers import TreeMPO, TreePEPO

    builder = py.ham_tn(shape=(4, 4), data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [((z_op, z_op), ((0, 0), (3, 3)), 1.0)]

    tree_mpo = builder.to_tree_mpo(
        terms,
        map_mode="coarse-alternate-x",
        compress=False,
    )
    tree_pepo = builder.to_tree_pepo(
        terms,
        map_mode="span-middle",
        compress=False,
    )

    assert isinstance(tree_mpo, TreeMPO)
    assert isinstance(tree_pepo, TreePEPO)
    assert tree_mpo.map_mode == "coarse-alternate-x"
    assert tree_pepo.map_mode == "span-middle"
    assert tree_pepo.plan.order == "snake"


def test_ham_tn_conversion_modes_select_sequential_or_analytic_builds():
    """All builder conversion families expose the same build-mode choice."""
    from pepsy.optimizers import TreePEPO, TreeMPO

    builder = py.ham_tn(shape=(2, 3), data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [
        ((z_op,), ((0, 0),), 0.5),
        ((z_op, z_op), ((0, 1), (1, 1)), 1.2),
    ]

    mpo_term = builder.to_mpo(
        terms,
        compress="term",
        cutoff=0.0,
    )
    mpo_analytic = builder.to_mpo(
        terms,
        compress="automaton",
        cutoff=0.0,
    )
    pepo_term = builder.to_pepo(
        terms,
        compress="term",
        cutoff=0.0,
    )
    pepo_analytic = builder.to_pepo(
        terms,
        compress="automaton",
        cutoff=0.0,
    )
    tree_mpo_term = builder.to_tree_mpo(
        terms,
        map_mode="coarse-alternate-x",
        compress="term",
        cutoff=0.0,
    )
    tree_mpo_analytic = builder.to_tree_mpo(
        terms,
        map_mode="coarse-alternate-x",
        compress="automaton",
        cutoff=0.0,
    )
    tree_pepo_term = builder.to_tree_pepo(
        terms,
        map_mode="coarse-alternate-x",
        compress="term",
        cutoff=0.0,
    )
    tree_pepo_analytic = builder.to_tree_pepo(
        terms,
        map_mode="coarse-alternate-x",
        compress="automaton",
        cutoff=0.0,
    )

    assert np.allclose(mpo_term.to_dense(), mpo_analytic.to_dense())
    assert np.allclose(pepo_term.to_dense(), pepo_analytic.to_dense())
    assert np.allclose(tree_mpo_term.to_dense(), tree_mpo_analytic.to_dense())
    assert np.allclose(tree_pepo_term.to_dense(), tree_pepo_analytic.to_dense())
    assert isinstance(tree_mpo_term, TreeMPO)
    assert isinstance(tree_pepo_term, TreePEPO)


def test_ham_tn_term_mode_compresses_after_each_term(monkeypatch):
    """Term accumulation applies the bond cap after every addition."""
    calls = []
    original_compress = qtn.MatrixProductOperator.compress

    def capture_compress(mpo, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_compress(mpo, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductOperator, "compress", capture_compress)

    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    terms = [((0,), "X", 0.5), (("ZZ", 1.2), (0, 1)), ((2,), "Y", -0.3)]
    builder.to_mpo(terms, compress="term", chi=2, cutoff=0.0)

    assert len(calls) == len(terms)
    assert all(call["max_bond"] == 2 for call in calls)


@pytest.mark.parametrize("max_bond", [None, False])
def test_ham_tn_automaton_skips_numerical_compression_without_bond_cap(
    monkeypatch, max_bond
):
    """An exact automaton build does not run an unbounded compression sweep."""
    calls = []
    original_compress = qtn.MatrixProductOperator.compress

    def capture_compress(mpo, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_compress(mpo, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductOperator, "compress", capture_compress)

    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    builder.to_mpo(
        [((0,), "X", 0.5), (("ZZ", 1.2), (0, 1))],
        compress="automaton",
        max_bond=max_bond,
        cutoff=0.0,
    )

    assert calls == []


def test_ham_tn_compress_is_canonical_strategy_control():
    """The old separate mode selector is compatibility-only by default."""
    for name in ("to_mpo", "to_pepo", "to_tree_mpo", "to_tree_pepo"):
        parameter = inspect.signature(getattr(py.ham_tn, name)).parameters["mode"]
        assert parameter.default is None


def test_ham_tn_to_builder_defaults_are_term_by_term():
    """All public ``to_*`` builders default to sequential term construction."""
    for name in ("to_mpo", "to_pepo", "to_tree_mpo", "to_tree_pepo"):
        parameter = inspect.signature(getattr(py.ham_tn, name)).parameters["compress"]
        assert parameter.default == "term"


def test_ham_tn_default_compresses_after_each_term(monkeypatch):
    """The omitted compression strategy is the incremental term policy."""
    calls = []
    original_compress = qtn.MatrixProductOperator.compress

    def capture_compress(mpo, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_compress(mpo, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductOperator, "compress", capture_compress)
    builder = py.ham_tn(shape=4, data_type="complex128")
    mpo = builder.to_mpo(
        [((quimb.pauli("X", dtype="complex128"),), (0,), 0.5),
         ((quimb.pauli("Z", dtype="complex128"),), (3,), 0.25)],
        cutoff=0.0,
    )

    assert len(calls) == 2
    assert not hasattr(mpo, "pepsy_automaton")


def test_ham_tn_explicit_true_keeps_automatic_route(monkeypatch):
    """Explicit ``compress=True`` retains the compatibility auto policy."""
    calls = []
    original_compress = qtn.MatrixProductOperator.compress

    def capture_compress(mpo, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_compress(mpo, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductOperator, "compress", capture_compress)
    builder = py.ham_tn(shape=4, data_type="complex128")
    mpo = builder.to_mpo(
        [((quimb.pauli("X", dtype="complex128"),), (0,), 0.5),
         ((quimb.pauli("Z", dtype="complex128"),), (3,), 0.25)],
        compress=True,
        cutoff=0.0,
    )

    assert len(calls) == 1
    assert hasattr(mpo, "pepsy_automaton")


def test_ham_tn_tree_automatic_layout_uses_term_supports():
    """Automatic native tree conversion retains its workload-aware finder."""
    builder = py.ham_tn(shape=(3, 3), data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [((z_op, z_op), ((0, 0), (2, 2)), 1.0)]

    tree_mpo = builder.to_tree_mpo(terms, compress="automaton", cutoff=0.0)
    tree_pepo = builder.to_tree_pepo(terms, compress="automaton", cutoff=0.0)

    assert tree_mpo.layout_finder is not None
    assert tree_mpo.validate()
    assert tree_pepo.layout_finder is not None
    assert tree_pepo.validate()
    assert tree_pepo.plan.tree_edges


def test_ham_tn_map_mode_and_backend_overrides_are_per_conversion():
    """Map and backend overrides do not mutate shared builder configuration."""
    builder = py.ham_tn(
        shape=(2, 3),
        map_mode="snake",
        data_type="complex128",
    )
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [((z_op,), ((1, 0),), 0.5)]

    row_major = builder.to_mpo(
        terms,
        map_mode="row-major",
        compress_each=False,
        to_backend=None,
    )
    expected = py.ham_tn(
        shape=(2, 3),
        map_mode="row-major",
        data_type="complex128",
    ).to_mpo(terms, compress_each=False)
    assert np.allclose(row_major.to_dense(), expected.to_dense())

    calls = []

    def convert(array):
        calls.append(array)
        return np.asarray(array, dtype=np.complex64)

    configured = py.ham_tn(
        shape=(2, 3),
        map_mode="snake",
        data_type="complex128",
        to_backend=convert,
    )
    configured.to_tree_mpo(terms, map_mode="row-major", compress=False)
    assert calls
    calls.clear()
    native_tree = configured.to_tree_pepo(
        terms,
        map_mode="row-major",
        tree_order="alternate-x",
        compress=False,
        to_backend=None,
    )
    assert native_tree.validate()
    assert not calls
    assert configured.map_mode == "snake"


def test_tree_operator_show_has_native_tree_and_pepo_lattice_views(capsys):
    """Native tree display is branch-aware while PEPO retains a lattice view."""
    from pepsy.optimizers import TreeMPO, TreePEPO

    builder = py.ham_tn(shape=(2, 3), data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [((z_op, z_op), ((0, 0), (1, 1)), 1.0)]
    tree_mpo = builder.to_tree_mpo(terms, map_mode="coarse-snake", compress=False)
    tree_pepo = builder.to_tree_pepo(
        terms,
        map_mode="snake",
        tree_order="alternate-x",
        compress=False,
    )

    tree_mpo.show()
    tree_mpo_output = capsys.readouterr().out
    assert "physical lattice" not in tree_mpo_output
    assert "◆ q" in tree_mpo_output
    lattice = tree_pepo.ascii_lattice()
    tree_pepo.show(layout="tree")
    topology = tree_pepo.ascii_tree()
    tree_pepo.show(layout="lattice", bond_dims=False, node_ids=True)
    output = capsys.readouterr().out

    assert isinstance(tree_mpo, TreeMPO)
    assert tree_mpo.layout_finder is not None
    assert tree_mpo.copy().layout_finder is tree_mpo.layout_finder
    lattice_mpo = tree_mpo.ascii_lattice()
    assert "physical lattice (2, 3)" in lattice_mpo
    assert "t0: q0 - q4" in lattice_mpo
    assert isinstance(tree_pepo, TreePEPO)
    assert tree_pepo.layout_finder is not None
    assert tree_pepo.copy().layout_finder is tree_pepo.layout_finder
    assert "◆ q" in output
    assert "N" in output
    assert "●" in lattice
    assert "◆ q" in topology


def test_tree_operator_show_color_is_opt_in(capsys):
    """Native tree show keeps plain text default and supports coloured output."""
    from pepsy.optimizers import TreeMPO

    builder = py.ham_tn(shape=4, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    operator = builder.to_tree_mpo(
        [((z_op,), (0,))],
        compress=False,
    )
    assert "\033[" not in operator.ascii_tree()
    operator.show(color=True)
    assert "\033[" in capsys.readouterr().out
    assert isinstance(operator, TreeMPO)


def test_to_tree_pepo_uses_tree_peps_plan_coordinates():
    """TreePEPO conversion honors the logical-coordinate map owned by its plan."""
    from pepsy.optimizers import TreePEPO, TreePepsPlan

    plan = TreePepsPlan.from_shape((2, 3))
    builder = py.ham_tn(shape=(2, 3), data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    operator = builder.to_tree_pepo(
        plan,
        [((z_op,), ((1, 2),), 0.75)],
        cutoff=0.0,
    )

    assert isinstance(operator, TreePEPO)
    logical_site = plan.logical_site((1, 2))
    assert operator.operator_support == (logical_site,)
    assert operator.operator_span == frozenset(plan.subtree_span((logical_site,)))
    assert operator.validate()


def test_tree_operator_arithmetic_matches_dense_products():
    """TreeMPO and TreePEPO arithmetic preserves exact dense semantics."""
    from pepsy.optimizers import TreeMPO, TreePEPO, TreePepsPlan, TreePlan

    z_op = quimb.pauli("Z", dtype="complex128")
    tree_plan = TreePlan.from_order([0, 1, 2, 3])
    tree_builder = py.ham_tn(shape=4, data_type="complex128")
    tree_left = tree_builder.to_tree_mpo(tree_plan, [((z_op,), (0,))], cutoff=0.0)
    tree_right = tree_builder.to_tree_mpo(tree_plan, [((z_op,), (1,), 2.0)], cutoff=0.0)
    assert isinstance(tree_left + tree_right, TreeMPO)
    assert np.allclose(
        (tree_left + tree_right).to_dense(),
        tree_left.to_dense() + tree_right.to_dense(),
    )
    assert np.allclose(
        (tree_left @ tree_right).to_dense(),
        tree_left.to_dense() @ tree_right.to_dense(),
    )
    assert np.allclose(
        (2.0 * tree_left).to_dense(),
        2.0 * tree_left.to_dense(),
    )

    peps_plan = TreePepsPlan.from_shape((2, 3))
    peps_builder = py.ham_tn(shape=(2, 3), data_type="complex128")
    peps_left = peps_builder.to_tree_pepo(peps_plan, [((z_op,), ((0, 0),))], cutoff=0.0)
    peps_right = peps_builder.to_tree_pepo(peps_plan, [((z_op,), ((1, 1),), 2.0)], cutoff=0.0)
    assert isinstance(peps_left + peps_right, TreePEPO)
    assert np.allclose(
        (peps_left + peps_right).to_dense(),
        peps_left.to_dense() + peps_right.to_dense(),
    )
    assert np.allclose(
        (peps_left @ peps_right).to_dense(),
        peps_left.to_dense() @ peps_right.to_dense(),
    )
    assert np.allclose(
        (-peps_left).to_dense(),
        -peps_left.to_dense(),
    )


def test_build_mpo_accepts_mapper_override():
    """build_mpo should allow a one-off mapper override for term placement."""
    builder_default = py.ham_tn(Lx=2, Ly=2, data_type="complex128")
    mapper = OneDMap(2, 2, mode="row-major")
    z_op = quimb.pauli("Z", dtype="complex128")

    ints = [
        ((z_op,), ((0, 0),)),
        ((z_op,), ((1, 1),), 0.5),
    ]

    mpo_from_override = builder_default.build_mpo(
        ints,
        compress_each=False,
        mapper=mapper,
    )

    builder_row_major = py.ham_tn(Lx=2, Ly=2, data_type="complex128", mapper=mapper)
    mpo_from_mapper_builder = builder_row_major.build_mpo(ints, compress_each=False)

    assert mpo_from_override.L == 4
    assert np.allclose(mpo_from_override.to_dense(), mpo_from_mapper_builder.to_dense())


def test_build_mpo_accepts_location_first_pauli_terms():
    """String Pauli terms support location-first and paired spellings."""
    builder = py.ham_tn(Lx=3, Ly=1, data_type="complex128")
    x_op = quimb.pauli("X", dtype="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")

    pauli_mpo = builder.build_mpo(
        [
            ((0,), "x", 0.5),
            (("zz", 1.2), (0, 1)),
        ],
        compress_each=False,
    )
    matrix_mpo = builder.build_mpo(
        [
            ((x_op,), (0,), 0.5),
            ((z_op, z_op), (0, 1), 1.2),
        ],
        compress_each=False,
    )

    assert np.allclose(pauli_mpo.to_dense(), matrix_mpo.to_dense())


def test_build_mpo_accepts_bare_2d_pauli_coordinate_with_mapper():
    """A one-site 2D coordinate can be used without an extra nesting level."""
    mapper = OneDMap(2, 2, mode="row-major")
    builder = py.ham_tn(Lx=2, Ly=2, mapper=mapper, data_type="complex128")
    x_op = quimb.pauli("X", dtype="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")

    pauli_mpo = builder.build_mpo(
        [
            ((0, 0), "X", 0.5),
            (("ZZ", 1.2), ((0, 0), (1, 0))),
        ],
        compress_each=False,
    )
    matrix_mpo = builder.build_mpo(
        [
            ((x_op,), ((0, 0),), 0.5),
            ((z_op, z_op), ((0, 0), (1, 0)), 1.2),
        ],
        compress_each=False,
    )

    assert np.allclose(pauli_mpo.to_dense(), matrix_mpo.to_dense())


def test_build_mpo_resolves_auto_cutoff_options(monkeypatch):
    """Auto cutoff options match the MPS dtype policy and rsum2 convention."""
    calls = []
    original_compress = qtn.MatrixProductOperator.compress

    def capture_compress(mpo, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_compress(mpo, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductOperator, "compress", capture_compress)

    builder = py.ham_tn(Lx=3, Ly=1, data_type="complex64")
    builder.build_mpo(
        [((0,), "X", 0.5), (("ZZ", 1.2), (0, 1))],
        cutoff="auto",
        cutoff_mode="auto",
        compress_each=False,
    )

    assert calls[-1]["cutoff"] == 1.0e-6
    assert calls[-1]["cutoff_mode"] == "rsum2"


def test_ham_builder_converts_generic_mpo_to_configured_backend():
    """A builder-level backend converter is applied to every MPO tensor."""
    torch = pytest.importorskip("torch")
    to_backend = py.backend_torch(dtype=torch.complex128)
    builder = py.ham_tn(
        Lx=3,
        Ly=1,
        cutoff="auto",
        cutoff_mode="auto",
        to_backend=to_backend,
    )
    assert builder.data_type == np.dtype("complex128")

    mpo = builder.build_mpo(
        [((0,), "Y", 0.5), (("ZZ", 1.2), (0, 1))],
        chi=2,
        form="left",
        method="svd",
        compress_opts={"renorm": False},
    )

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in mpo)
    assert mpo.max_bond() <= 2


def test_ham_builder_automaton_preserves_shared_structure_on_backend():
    """Automaton sharing happens before backend conversion and compression."""
    torch = pytest.importorskip("torch")
    to_backend = py.backend_torch(dtype=torch.complex128)
    builder = py.ham_tn(Lx=6, Ly=1, to_backend=to_backend)
    terms = [
        ((site,), "X", 0.5)
        for site in range(builder.L)
    ] + [
        (("ZZ", 1.2), (site, site + 1))
        for site in range(builder.L - 1)
    ]

    mpo = builder.build_mpo(
        terms,
        mode="automaton",
        chi=4,
        cutoff=0.0,
    )

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in mpo)
    assert mpo.pepsy_automaton.bond_dimensions == (3, 3, 3, 3, 3)
    assert mpo.max_bond() <= 4


def test_build_mpo_automaton_mode_matches_term_mode():
    """The finite-state automaton mode returns the same operator."""
    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    terms = [((0,), "X", 0.5), (("ZZ", 1.2), (0, 1)), ((2,), "Y", -0.3)]

    term_mpo = builder.build_mpo(terms, compress_each=False, cutoff=0.0)
    automaton_mpo = builder.build_mpo(
        terms,
        mode="automaton",
        compress_each=False,
        cutoff=0.0,
    )

    assert hasattr(automaton_mpo, "pepsy_automaton")
    assert np.allclose(term_mpo.to_dense(), automaton_mpo.to_dense())


def test_build_mpo_automaton_coalesces_duplicate_product_terms():
    """Equivalent product paths are summed before structural compilation."""
    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    terms = [
        (("ZZ", 1.2), (0, 3)),
        ((0, 3), "ZZ", 0.8),
    ]

    term_mpo = builder.build_mpo(terms, compress_each=False, cutoff=0.0)
    automaton_mpo = builder.build_mpo(
        terms,
        mode="automaton",
        compress_each=False,
        cutoff=0.0,
    )

    assert automaton_mpo.pepsy_automaton.bond_dimensions == (3, 3, 3)
    assert np.allclose(term_mpo.to_dense(), automaton_mpo.to_dense())


def test_build_mpo_automaton_removes_identity_factors():
    """Identity factors do not create unnecessary automaton channels."""
    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    identity = np.eye(2, dtype="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [((identity, z_op), (0, 3), 1.5)]

    term_mpo = builder.build_mpo(terms, compress_each=False, cutoff=0.0)
    automaton_mpo = builder.build_mpo(
        terms,
        mode="automaton",
        compress_each=False,
        cutoff=0.0,
    )

    assert automaton_mpo.pepsy_automaton.bond_dimensions == (2, 2, 2)
    assert np.allclose(term_mpo.to_dense(), automaton_mpo.to_dense())


def test_build_mpo_auto_falls_back_before_wide_automaton(monkeypatch):
    """Auto mode avoids allocating a structurally wide exact automaton."""
    builder = py.ham_tn(Lx=8, Ly=1, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [
        (
            (np.diag([1.0, 1.0 + index / 100.0]), z_op),
            (0, builder.L - 1),
            1.0,
        )
        for index in range(65)
    ]

    def fail_compile(*args, **kwargs):
        raise AssertionError("auto mode should select term accumulation here")

    monkeypatch.setattr(builder, "_compile_automaton", fail_compile)
    mpo = builder.build_mpo(
        terms,
        mode="auto",
        max_bond=1,
        compress_each=False,
        cutoff=0.0,
    )

    assert mpo.max_bond() <= 1
    assert not hasattr(mpo, "pepsy_automaton")


@pytest.mark.parametrize("mode", ["automaton", "auto"])
def test_build_mpo_automaton_modes_compress_final_mpo_to_chi(monkeypatch, mode):
    """Both automaton spellings apply chi to the final compiled MPO."""
    calls = []
    original_compress = qtn.MatrixProductOperator.compress

    def capture_compress(mpo, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_compress(mpo, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductOperator, "compress", capture_compress)

    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    mpo = builder.build_mpo(
        [((0,), "X", 0.5), (("ZZ", 1.2), (0, 1)), ((2,), "Y", -0.3)],
        mode=mode,
        chi=2,
        cutoff=0.0,
        cutoff_mode="auto",
    )

    assert len(calls) == 1
    assert calls[0]["max_bond"] == 2
    assert calls[0]["cutoff_mode"] == "rsum2"
    assert mpo.max_bond() <= 2


def test_build_mpo_and_pepo_accept_native_fermion_terms_with_mapper():
    """Hamiltonian builders forward OneDMap and native fermion terms together."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    left = (0, 0)
    right = (2, 1)
    term = fermion.operator_term(
        [(1.0, ((left, "double"), (right, "annihilate_up")))],
        sites=(left, right),
        label="ham_builder_charged",
    )
    mapper = OneDMap(3, 2, mode="snake-row-major")
    builder = py.ham_tn(
        Lx=3,
        Ly=2,
        mapper=mapper,
        data_type="complex128",
    )

    mpo = builder.build_mpo(
        {(left, right): term},
        fermion=fermion,
        fermionic=True,
        compress_each=False,
    )
    pepo = builder.build_pepo(
        {(left, right): term},
        fermion=fermion,
        fermionic=True,
        compress_each=False,
    )

    assert mpo.L == 6
    assert pepo.Lx == 3
    assert pepo.Ly == 2
    assert list(pepo)[-1].data.charge == term.charge
    assert all(type(tensor.data).__name__.endswith("FermionicArray") for tensor in pepo)


def test_build_mpo_and_pepo_return_mixed_charge_sectors():
    """Hamiltonian builders expose mixed native charges explicitly."""
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    left = (0, 0)
    middle = (0, 1)
    right = (1, 1)
    neutral = fermion.hopping_operator()
    charged = fermion.operator_term(
        [(1.0, ((middle, "double"), (right, "annihilate_up")))],
        sites=(middle, right),
        label="ham_builder_mixed_charge",
    )
    terms = {(left, middle): neutral, (middle, right): charged}
    mapper = OneDMap(2, 2, mode="snake-row-major")
    builder = py.ham_tn(Lx=2, Ly=2, mapper=mapper, data_type="complex128")

    mpo_sectors = builder.build_mpo(
        terms,
        fermion=fermion,
        fermionic=True,
        charge_sectors=True,
        compress_each=False,
    )
    pepo_sectors = builder.build_pepo(
        terms,
        fermion=fermion,
        fermionic=True,
        charge_sectors=True,
        compress_each=False,
    )

    assert set(mpo_sectors) == {fermion.zero_charge, charged.charge}
    assert set(pepo_sectors) == {fermion.zero_charge, charged.charge}
    assert all(mpo.L == 4 for mpo in mpo_sectors.values())
    assert all(pepo.Lx == 2 and pepo.Ly == 2 for pepo in pepo_sectors.values())


def test_build_mpo_uses_canonical_ops_sites_coeff_order():
    """build_mpo should accept the canonical (ops, sites, coeff) term order."""
    builder = py.ham_tn(Lx=2, Ly=2, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    x_op = quimb.pauli("X", dtype="complex128")

    ints = [
        ((z_op,), ((0, 0),), 0.5),
        ((z_op, z_op), ((0, 0), (1, 0)), 1.0),
        ((x_op,), ((1, 1),), -0.25),
    ]

    mpo = builder.build_mpo(ints, compress_each=False)

    assert mpo.L == 4


def test_build_mpo_rejects_legacy_sites_ops_order():
    """build_mpo should reject the old (sites, ops, coeff) term order."""
    builder = py.ham_tn(Lx=2, Ly=2, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")

    ints_legacy = [
        (((0, 0),), (z_op,), 0.5),
    ]

    with pytest.raises(TypeError, match="integer chain indices or 2D coordinates"):
        builder.build_mpo(ints_legacy, compress_each=False)


def test_build_itf_lattice_ly1_has_chain_edges():
    """Ly=1 ITF lattice should reduce to a nearest-neighbor 1D chain."""
    out = py.ham_tn.build_itf_lattice(
        Lx=5,
        Ly=1,
        lattice="square",
        J=1.0,
        field=0.5,
        return_edges=True,
    )

    assert out["builder"].Lx == 5
    assert out["builder"].Ly == 1
    assert out["mpo"].L == 5
    assert out["pepo"] is None
    assert all(y == 0 for _, y in out["one_d_to_two_d"].values())

    expected = {frozenset((i, i + 1)) for i in range(4)}
    got = {frozenset(edge) for edge in out["edges_1d"]}
    assert got == expected


def test_build_itf_lattice_ly1_cyclic_drops_degenerate_edges():
    """Ly=1 with cyclic=True should ignore degenerate singleton periodic edges."""
    with pytest.warns(UserWarning, match="Dropped 5 degenerate generated edge"):
        out = py.ham_tn.build_itf_lattice(
            Lx=5,
            Ly=1,
            lattice="square",
            cyclic=True,
            J=1.0,
            field=0.5,
            return_edges=True,
        )

    assert out["mpo"].L == 5
    assert out["pepo"] is None

    expected = {frozenset((i, i + 1)) for i in range(4)}
    expected.add(frozenset((0, 4)))
    got = {frozenset(edge) for edge in out["edges_1d"]}
    assert got == expected


def test_build_itf_lattice_can_return_pepo_explicitly():
    """build_itf_lattice should only include the PEPO when requested."""
    out = py.ham_tn.build_itf_lattice(
        Lx=3,
        Ly=1,
        lattice="square",
        J=1.0,
        field=0.5,
        return_pepo=True,
    )

    assert out["mpo"].L == 3
    assert out["pepo"].Lx == 3
    assert out["pepo"].Ly == 1


def test_build_itf_lattice_show_returns_schematic_drawing():
    """build_itf_lattice(show=True) should include a schematic MPO drawing."""
    pytest.importorskip("matplotlib")
    out = py.ham_tn.build_itf_lattice(
        Lx=2,
        Ly=2,
        lattice="square",
        J=1.0,
        field=0.5,
        show=True,
    )

    assert out["mpo"].L == 4
    assert out["pepo"] is None
    assert hasattr(out["drawing"], "fig")
    assert hasattr(out["drawing"], "ax")
    assert "Square ITF MPO" in out["drawing"].ax.get_title()


def test_build_itf_lattice_accepts_mapper_instance():
    """build_itf_lattice should accept a preconfigured OneDMap instance."""
    mapper = OneDMap(2, 2, mode="snake-row-major")
    out = py.ham_tn.build_itf_lattice(
        Lx=2,
        Ly=2,
        lattice="square",
        J=1.0,
        field=0.5,
        mapper=mapper,
        return_edges=True,
    )

    assert out["builder"].mapper is mapper
    assert out["builder"].map_mode == "snake-row-major"
    assert out["one_d_to_lattice"] == mapper.build()[0]
    assert out["mpo"].L == 4
    assert out["pepo"] is None


def test_build_itf_lattice_allows_non_snake_mapper_for_default_mpo():
    """build_itf_lattice should allow non-snake mappers when only MPO is requested."""
    mapper = OneDMap(2, 2, mode="row-major")
    out = py.ham_tn.build_itf_lattice(
        Lx=2,
        Ly=2,
        lattice="square",
        J=1.0,
        field=0.5,
        mapper=mapper,
        return_edges=True,
    )

    assert out["builder"].mapper is mapper
    assert out["mpo"].L == 4
    assert out["pepo"] is None


def test_build_itf_lattice_rejects_non_snake_mapper_for_pepo():
    """build_itf_lattice should reject non-snake mappers when PEPO is requested."""
    mapper = OneDMap(2, 2, mode="row-major")

    with pytest.raises(NotImplementedError, match="snake-style 2D mapping"):
        py.ham_tn.build_itf_lattice(
            Lx=2,
            Ly=2,
            lattice="square",
            J=1.0,
            field=0.5,
            mapper=mapper,
            return_edges=True,
            return_pepo=True,
        )


def test_ham_tn_rejects_mapper_shape_mismatch():
    """ham_tn should fail clearly when mapper shape does not match builder shape."""
    with pytest.raises(ValueError, match="mapper shape"):
        py.ham_tn(Lx=2, Ly=3, data_type="complex128", mapper=OneDMap(2, 2, mode="snake"))


def test_map_builder_snake_2d_matches_expected_layout():
    """Default 2D snake traversal should match the legacy ham_tn mapping."""
    map_, map_inv = OneDMap.build(3, 2, mode="snake")
    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
        4: (2, 0),
        5: (2, 1),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_supports_3d_snake():
    """3D snake mode should return consistent 1D<->3D maps."""
    map_, map_inv = OneDMap.build(2, 2, Lz=2, mode="snake")

    assert len(map_) == 8
    assert len(map_inv) == 8
    assert all(len(coord) == 3 for coord in map_.values())
    assert all(map_inv[coord] == idx for idx, coord in map_.items())


def test_map_builder_supports_inside_out_mode():
    """inside-out should be a complete deterministic center-to-edge order."""
    map_, map_inv = OneDMap.build(4, 4, mode="center-out")

    assert map_[0] == (1, 1)
    assert {index: map_[index] for index in range(1, 4)} == {
        1: (1, 2),
        2: (2, 1),
        3: (2, 2),
    }
    assert len(map_) == len(map_inv) == 16
    assert set(map_.values()) == {(x, y) for x in range(4) for y in range(4)}


def test_map_builder_supports_inside_out_3d_mode():
    """inside-out should also cover all sites in a 3D lattice."""
    map_, map_inv = OneDMap.build(3, 3, Lz=3, mode="inside-out")

    assert map_[0] == (1, 1, 1)
    assert len(map_) == len(map_inv) == 27
    assert set(map_.values()) == {
        (x, y, z) for x in range(3) for y in range(3) for z in range(3)
    }


def test_map_builder_supports_row_major_snake_mode():
    """snake-row-major should snake along x within each y row."""
    map_, map_inv = OneDMap.build(3, 2, mode="snake-row-major")
    expected = {
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (2, 1),
        4: (1, 1),
        5: (0, 1),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_supports_folded_snake_mode():
    """folded-snake should alternate opposite periodic columns."""
    map_, map_inv = OneDMap.build(3, 2, mode="folded-snake")
    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (2, 1),
        3: (2, 0),
        4: (1, 0),
        5: (1, 1),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_folded_snake_reduces_6x6_torus_bandwidth():
    """Folded periodic snake should avoid the length-35 torus wrap edge."""
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    edges = tuple(qtn.edges_2d_square(6, 6, cyclic=True))
    _, snake = OneDMap(6, 6, mode="snake").build()
    _, folded = OneDMap(6, 6, mode="folded-snake").build()

    snake_lengths = [abs(snake[a] - snake[b]) for a, b in edges]
    folded_lengths = [abs(folded[a] - folded[b]) for a, b in edges]

    assert max(snake_lengths) == 35
    assert max(folded_lengths) == 12
    assert sum(folded_lengths) == sum(snake_lengths)


def test_map_builder_supports_hilbert_mode():
    """hilbert mode should follow the standard 4x4 Hilbert traversal."""
    map_, map_inv = OneDMap.build(4, 4, mode="hilbert")
    expected = {
        0: (0, 0),
        1: (1, 0),
        2: (1, 1),
        3: (0, 1),
        4: (0, 2),
        5: (0, 3),
        6: (1, 3),
        7: (1, 2),
        8: (2, 2),
        9: (2, 3),
        10: (3, 3),
        11: (3, 2),
        12: (3, 1),
        13: (2, 1),
        14: (2, 0),
        15: (3, 0),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}
    assert all(
        abs(map_[idx][0] - map_[idx + 1][0]) + abs(map_[idx][1] - map_[idx + 1][1]) == 1
        for idx in range(len(map_) - 1)
    )


def test_map_builder_supports_hilbert_row_major_mode():
    """hilbert-row-major should expose the transposed Hilbert orientation."""
    map_, map_inv = OneDMap.build(4, 4, mode="hilbert-row")
    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
        4: (2, 0),
        5: (3, 0),
        6: (3, 1),
        7: (2, 1),
        8: (2, 2),
        9: (3, 2),
        10: (3, 3),
        11: (2, 3),
        12: (1, 3),
        13: (1, 2),
        14: (0, 2),
        15: (0, 3),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_hilbert_rejects_unsupported_shapes():
    """hilbert mode should still reject 3D usage clearly."""
    with pytest.raises(NotImplementedError, match="only for 2D lattices"):
        OneDMap.build(2, 2, Lz=2, mode="hilbert")


def test_map_builder_supports_rectangular_hilbert_mode():
    """hilbert mode should cover arbitrary rectangles without cropping gaps."""
    map_, map_inv = OneDMap.build(3, 5, mode="hilbert")

    assert len(map_) == 15
    assert len(map_inv) == 15
    assert set(map_.values()) == {(x, y) for x in range(3) for y in range(5)}
    assert all(map_inv[coord] == idx for idx, coord in map_.items())
    assert map_[0] == (0, 0)
    assert all(
        abs(map_[idx][0] - map_[idx + 1][0])
        + abs(map_[idx][1] - map_[idx + 1][1])
        == 1
        for idx in range(len(map_) - 1)
    )


def test_map_builder_supports_rectangular_hilbert_row_major_mode():
    """Row-major Hilbert should preserve the transposed rectangular orientation."""
    map_, map_inv = OneDMap.build(3, 5, mode="hilbert-row-major")

    assert len(map_) == 15
    assert len(map_inv) == 15
    assert set(map_.values()) == {(x, y) for x in range(3) for y in range(5)}
    assert all(map_inv[coord] == idx for idx, coord in map_.items())
    assert map_[0] == (0, 0)
    assert map_ != OneDMap.build(3, 5, mode="hilbert")[0]
    assert all(
        abs(map_[idx][0] - map_[idx + 1][0])
        + abs(map_[idx][1] - map_[idx + 1][1])
        == 1
        for idx in range(len(map_) - 1)
    )


@pytest.mark.parametrize("shape", [(1, 7), (2, 5), (5, 2), (5, 4), (8, 3)])
@pytest.mark.parametrize("mode", ["hilbert", "hilbert-row-major"])
def test_map_builder_hilbert_corner_shapes_are_bijective(shape, mode):
    """Generalized Hilbert maps stay bounded and complete at corner shapes."""
    Lx, Ly = shape
    map_, map_inv = OneDMap.build(Lx, Ly, mode=mode)

    assert len(map_) == Lx * Ly
    assert set(map_.values()) == {(x, y) for x in range(Lx) for y in range(Ly)}
    assert map_inv == {coord: idx for idx, coord in map_.items()}
    steps = [
        abs(map_[idx][0] - map_[idx + 1][0])
        + abs(map_[idx][1] - map_[idx + 1][1])
        for idx in range(len(map_) - 1)
    ]
    # A rectangular Hilbert traversal is orthogonal except for the single
    # parity-forced diagonal allowed by the generalized construction.
    assert max(steps, default=0) <= 2
    assert steps.count(2) <= 1


def test_ham_tn_accepts_builtin_mapping_mode_string():
    """ham_tn should accept a OneDMap instance from core builder."""
    builder = py.ham_tn(
        Lx=2,
        Ly=2,
        data_type="complex128",
        mapper=OneDMap(2, 2, mode="row-major"),
    )
    assert builder.map == {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def test_ham_tn_normalizes_map_mode_aliases_via_onedmap():
    """ham_tn should store the normalized OneDMap mode and mapping helper."""
    builder = py.ham_tn(
        Lx=4,
        Ly=4,
        data_type="complex128",
        mapper=OneDMap(4, 4, mode="hilbert-row"),
    )

    assert isinstance(builder.mapper, OneDMap)
    assert builder.map_mode == "hilbert-row-major"
    assert builder.mapper.mode == "hilbert-row-major"
    assert builder.map == builder.mapper.build()[0]


def test_ham_tn_supports_3d_mapping_and_mpo_terms():
    """3D builders should accept (x, y, z) coordinates in interaction terms."""
    builder = py.ham_tn(
        Lx=2,
        Ly=2,
        Lz=2,
        data_type="complex128",
        mapper=OneDMap(2, 2, Lz=2, mode="snake"),
    )
    z_op = quimb.pauli("Z", dtype="complex128")

    assert builder.L == 8
    assert all(len(coord) == 3 for coord in builder.map.values())
    assert builder.map_site((1, 1, 1)) == builder.map_inv[(1, 1, 1)]

    mpo = builder.build_mpo(
        [
            ((z_op,), ((0, 0, 0),)),
            ((z_op, z_op), ((0, 0, 0), (1, 0, 0)), 0.5),
        ],
        compress_each=False,
    )
    assert mpo.L == 8


def test_ham_tn_snake_row_major_supports_pepo_conversion():
    """snake-row-major should remain eligible for 2D MPO->PEPO conversion."""
    builder = py.ham_tn(
        Lx=4,
        Ly=4,
        data_type="complex128",
        mapper=OneDMap(4, 4, mode="snake-row-major"),
    )
    pepo, coord_to_chain = builder.mpo_itf(J=1.0, field=0.5, as_pepo=True)

    assert pepo.Lx == 4
    assert pepo.Ly == 4
    assert coord_to_chain == builder.map_inv


def test_ham_tn_hilbert_mode_rejects_pepo_conversion():
    """Hilbert mode should be available for mapping, but not for PEPO builds."""
    builder = py.ham_tn(
        Lx=4,
        Ly=4,
        data_type="complex128",
        mapper=OneDMap(4, 4, mode="hilbert"),
    )

    with pytest.raises(NotImplementedError, match="snake-style 2D mapping"):
        builder.mpo_itf(J=1.0, field=0.5, as_pepo=True)


def test_ham_tn_row_major_mode_rejects_pepo_conversion():
    """Non-snake maps should fail clearly when PEPO conversion is requested."""
    builder = py.ham_tn(
        Lx=2,
        Ly=2,
        data_type="complex128",
        mapper=OneDMap(2, 2, mode="row-major"),
    )

    with pytest.raises(NotImplementedError, match="snake-style 2D mapping"):
        builder.mpo_itf(J=1.0, field=0.5, as_pepo=True)


def test_ham_tn_3d_rejects_pepo_conversion():
    """3D builders should raise a clear error for 2D-only PEPO conversion."""
    builder = py.ham_tn(Lx=2, Ly=2, Lz=2, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    mpo = builder.build_mpo([((z_op,), ((0, 0, 0),))], compress_each=False)

    with pytest.raises(NotImplementedError, match="only available for 2D builders"):
        builder.mpo_to_pepo(mpo)


def test_mpo_itf_works_for_2d_builder():
    """mpo_itf should build the default 2D square-lattice ITF MPO."""
    builder = py.ham_tn(Lx=3, Ly=2, data_type="complex128")
    mpo, coord_to_chain = builder.mpo_itf(J=1.0, field=0.5)

    assert mpo.L == 6
    assert coord_to_chain == builder.map_inv


def test_mpo_itf_works_for_3d_builder_and_blocks_pepo():
    """mpo_itf should support 3D builders and reject PEPO output there."""
    builder = py.ham_tn(Lx=2, Ly=2, Lz=2, data_type="complex128")
    mpo, coord_to_chain = builder.mpo_itf(J=1.0, field=0.5)

    assert mpo.L == 8
    assert coord_to_chain == builder.map_inv
    assert all(len(coord) == 3 for coord in coord_to_chain)

    with pytest.raises(NotImplementedError, match="only available for 2D builders"):
        builder.mpo_itf(as_pepo=True)


def test_map_builder_supports_col_major_mode():
    """col_major mode should enumerate x fastest within each y row."""
    map_, map_inv = OneDMap.build(2, 3, mode="col_major")
    expected = {
        0: (0, 0),
        1: (1, 0),
        2: (0, 1),
        3: (1, 1),
        4: (0, 2),
        5: (1, 2),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_supports_instance_style_build():
    """OneDMap should support object-style construction then build()."""
    mapper = OneDMap(3, 2, mode="row-major")

    assert mapper.shape == (3, 2)
    assert mapper.mode == "row-major"

    map_, map_inv = mapper.build()
    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
        4: (2, 0),
        5: (2, 1),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_instance_style_can_override_mode_per_call():
    """Instance build/show calls should allow temporary mode overrides."""
    mapper = OneDMap(3, 2, mode="row-major")

    map_, _ = mapper.build(mode="snake")
    assert map_[0] == (0, 0)
    assert map_[1] == (0, 1)
    assert map_[2] == (1, 1)


def test_map_builder_finder_composes_mps_site_order_with_lattice_coords():
    """Finder mode should map optimized MPS positions back to coordinates."""
    base_idx2coo, _ = OneDMap(2, 2, mode="snake").build()
    plan = {"site_order": (2, 0, 3, 1)}
    mapper = OneDMap(2, 2, mode="finder", finder=plan)

    idx2coo, coo2idx = mapper.build()

    assert idx2coo == {
        position: base_idx2coo[logical_site]
        for position, logical_site in enumerate(plan["site_order"])
    }
    assert coo2idx == {coord: position for position, coord in idx2coo.items()}


def test_map_builder_finder_mode_runs_mps_layout_finder():
    """Finder mode should accept a gate stream without touching an MPS state."""
    gates = [
        (quimb.CNOT(), (0, 3)),
        (quimb.CNOT(), (3, 1)),
    ]
    mapper = OneDMap(
        2,
        2,
        mode="finder",
        gates=gates,
        layout_kwargs={"order": "input"},
    )

    idx2coo, _ = mapper.build()

    assert idx2coo == OneDMap(2, 2, mode="snake").build()[0]


def test_map_builder_finder_mode_requires_layout_source():
    """Finder mode should fail clearly when no stream or plan is supplied."""
    with pytest.raises(ValueError, match="mode='finder'"):
        OneDMap(2, 2, mode="finder").build()


def test_map_builder_instance_style_supports_3d_build():
    """Instance-style build() should preserve the 3D mapping modes."""
    mapper = OneDMap(2, 2, Lz=2, mode="col-major")
    map_, map_inv = mapper.build()

    assert mapper.shape == (2, 2, 2)
    assert len(map_) == 8
    assert all(len(coord) == 3 for coord in map_.values())
    assert all(map_inv[coord] == idx for idx, coord in map_.items())


def test_map_builder_show_returns_schematic_drawing():
    """show() should now return a schematic drawing object for 2D maps."""
    pytest.importorskip("matplotlib")
    drawing = OneDMap.show(2, 2, mode="snake")
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawing.ax.get_title() == "OneDMap snake (2x2)"


def test_map_builder_instance_show_returns_schematic_drawing():
    """Instance-style show() should return a drawing and honor override kwargs."""
    pytest.importorskip("matplotlib")
    mapper = OneDMap(2, 2, mode="row-major")
    drawing = mapper.show(mode="snake", title="Instance Mapper")

    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawing.ax.get_title() == "Instance Mapper"


def test_map_builder_show_reports_missing_matplotlib(monkeypatch):
    """show() should report the missing plotting package directly."""
    real_import = builtins.__import__

    def import_without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)

    with pytest.raises(ImportError, match="matplotlib"):
        OneDMap.show(2, 2, mode="snake")


def test_mpo_schematic_reports_missing_matplotlib(monkeypatch):
    """MPO schematic rendering should report the missing plotting package directly."""
    real_import = builtins.__import__

    def import_without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    class DummyMPO:
        L = 4

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)

    builder = py.ham_tn(Lx=2, Ly=2)
    with pytest.raises(ImportError, match="matplotlib"):
        builder._show_mpo_schematic_2d(DummyMPO(), [((0, 0), (1, 0))])


def test_map_builder_show_rejects_3d():
    """show() should fail clearly for 3D maps until a schematic 3D view exists."""
    with pytest.raises(NotImplementedError, match="only available for 2D lattices"):
        OneDMap.show(2, 2, Lz=2, mode="snake")
