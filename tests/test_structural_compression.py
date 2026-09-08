"""Regression tests for builder-side structural operator compression."""

import importlib
import sys
import types

import numpy as np
import quimb
import quimb.tensor as qtn

import pepsy as py
from pepsy.operators._structural_compression import _factor_columns


def test_column_factorization_is_exact_and_reduces_delinearized_channels():
    """The structural factorization only removes roundoff-safe dependence."""
    matrix = np.array(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
        dtype="complex128",
    )

    basis, transfer, changed = _factor_columns(matrix, method="auto")

    assert changed
    assert basis.shape == (2, 1)
    assert np.allclose(basis @ transfer, matrix, atol=0.0, rtol=1.0e-14)


def test_column_factorization_checks_composed_transfer_residual():
    """Tiny pivots must not amplify a valid intermediate QR residual."""
    matrix = np.array(
        [
            [0.0 - 1.29e-33j, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0 - 1.29e-33j, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0 - 1.29e-33j, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, -1.57e-16, 0.0, 0.0],
            [0.0, 0.0, -0.575, 0.0, 1.06e-33j, 0.0],
        ],
        dtype="complex128",
    )

    basis, transfer, _changed = _factor_columns(matrix, method="auto")

    assert np.allclose(basis @ transfer, matrix, rtol=1.0e-12, atol=1.0e-14)


def test_mpo_sweep_preserves_operator_for_both_edge_orientations():
    """The forward and reverse bond reductions preserve a raw MPO exactly."""
    rng = np.random.default_rng(4)
    left = np.empty((3, 2, 2), dtype="complex128")
    vector = rng.normal(size=3) + 1j * rng.normal(size=3)
    for upper in range(2):
        for lower in range(2):
            left[:, upper, lower] = (upper + lower + 1) * vector
    right_dependent = rng.normal(size=(3, 2, 2))
    right_dependent[1] = 2.0 * right_dependent[0]
    right_dependent[2] = 0.0
    right_full = rng.normal(size=(3, 2, 2))
    structural = importlib.import_module(
        "pepsy.operators._structural_compression"
    )._structural_compress_mpo

    forward = qtn.MatrixProductOperator(
        [left, right_dependent],
        shape="lrud",
    )
    forward_reference = forward.to_dense()
    forward_report = structural(forward)

    reverse = qtn.MatrixProductOperator([left, right_full], shape="lrud")
    reverse_reference = reverse.to_dense()
    reverse_report = structural(reverse)

    assert forward_report["changed"]
    assert reverse_report["changed"]
    assert forward.max_bond() == 1
    assert reverse.max_bond() == 1
    assert np.allclose(
        forward.to_dense(),
        forward_reference,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.allclose(
        reverse.to_dense(),
        reverse_reference,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_to_mpo_reduces_exact_automaton_boundary_dependencies():
    """Builder MPOs structurally reduce channels before optional numerical SVD."""
    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    terms = [
        (("ZZ", 1.2), (0, 3)),
        ((0, 3), "ZZ", 0.8),
    ]

    mpo = builder.to_mpo(
        terms,
        mode="automaton",
        compress=False,
        cutoff=0.0,
    )
    reference = builder.to_mpo(
        terms,
        mode="term",
        compress=False,
        cutoff=0.0,
    )

    assert min(mpo.bond_sizes()) < max(mpo.pepsy_automaton.bond_dimensions)
    assert np.allclose(mpo.to_dense(), reference.to_dense())

    records = builder._normalize_automaton_terms(
        tuple(terms),
        phys_dim=2,
        dtype=np.dtype("complex128"),
    )
    raw_mpo, _ = builder._build_mpo_from_automaton(records, phys_dim=2)
    assert np.allclose(mpo.to_dense(), raw_mpo.to_dense())


def test_term_compression_preconditions_each_sequential_svd(monkeypatch):
    """Sequential chain compression reduces channels before every SVD."""
    ham_module = importlib.import_module("pepsy.operators.hamiltonians")
    events = []
    structural = ham_module._structural_compress_mpo
    compress = qtn.MatrixProductOperator.compress

    def capture_structural(mpo, *args, **kwargs):
        events.append("structural")
        return structural(mpo, *args, **kwargs)

    def capture_compress(mpo, *args, **kwargs):
        events.append("compress")
        return compress(mpo, *args, **kwargs)

    monkeypatch.setattr(ham_module, "_structural_compress_mpo", capture_structural)
    monkeypatch.setattr(qtn.MatrixProductOperator, "compress", capture_compress)

    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    builder.to_mpo(
        [((0,), "X", 0.5), (("ZZ", 1.2), (0, 1)), ((2,), "Y", -0.3)],
        compress="term",
        chi=2,
        cutoff=0.0,
    )

    assert events[:6] == [
        "structural", "compress",
        "structural", "compress",
        "structural", "compress",
    ]


def test_term_builder_progress_bar_reports_chi_and_transient_peak(monkeypatch):
    """Term progress follows the MPS style and exposes direct-sum growth."""
    bars = []

    class FakeProgressBar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.postfixes = []
            self.updates = []
            self.closed = False
            bars.append(self)

        def set_postfix(self, postfix):
            self.postfixes.append(dict(postfix))

        def update(self, value):
            self.updates.append(value)

        def close(self):
            self.closed = True

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = FakeProgressBar
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    builder = py.ham_tn(Lx=4, Ly=1, data_type="complex128")
    builder.to_mpo(
        [((0,), "X", 0.5), (("ZZ", 1.2), (0, 1)), ((2,), "X", -0.3)],
        compress="term",
        max_bond=1,
        cutoff=0.0,
        progbar=True,
    )

    assert len(bars) == 1
    bar = bars[0]
    assert bar.kwargs == {
        "total": 3,
        "desc": "mpo-term",
        "leave": True,
        "position": 0,
        "ascii": True,
        "colour": "#2ca02c",
    }
    assert bar.updates == [1, 1, 1]
    assert bar.closed
    assert any(postfix["chi"] == "1/1" for postfix in bar.postfixes)
    assert any("peak" in postfix for postfix in bar.postfixes)


def test_tree_builder_progress_bars_use_the_same_mps_style(monkeypatch):
    """Tree term builders expose matching progress controls and postfixes."""
    bars = []

    class FakeProgressBar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.postfixes = []
            self.updates = []
            self.closed = False
            bars.append(self)

        def set_postfix(self, postfix):
            self.postfixes.append(dict(postfix))

        def update(self, value):
            self.updates.append(value)

        def close(self):
            self.closed = True

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = FakeProgressBar
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    z_op = quimb.pauli("Z", dtype="complex128")
    tree_builder = py.ham_tn(shape=4, data_type="complex128")
    tree_builder.to_tree_mpo(
        py.TreePlan.from_order([0, 1, 2, 3]),
        [((z_op, z_op), (0, 3), 1.2), ((z_op, z_op), (0, 2), 0.8)],
        mode="term",
        max_bond=2,
        cutoff=0.0,
        progbar=True,
    )

    pepo_builder = py.ham_tn(shape=(3, 3), data_type="complex128")
    pepo_builder.to_tree_pepo(
        [
            ((z_op, z_op), ((0, 0), (2, 2)), 1.2),
            ((z_op, z_op), ((0, 1), (2, 1)), 0.8),
        ],
        map_mode="span-up",
        mode="term",
        max_bond=2,
        cutoff=0.0,
        progbar=True,
    )

    assert [bar.kwargs["desc"] for bar in bars] == [
        "tree-mpo-term",
        "tree-pepo-term",
    ]
    assert all(
        bar.kwargs["leave"]
        and bar.kwargs["position"] == 0
        and bar.kwargs["ascii"]
        and bar.kwargs["colour"] == "#2ca02c"
        for bar in bars
    )
    assert all(bar.kwargs["total"] == 2 for bar in bars)
    assert all(bar.updates == [1, 1] and bar.closed for bar in bars)
    assert all(
        any("chi" in postfix for postfix in bar.postfixes)
        for bar in bars
    )


def test_tree_builders_use_structural_compression_without_a_bond_cap():
    """TreeMPO and TreePEPO builders keep the exact operator after reduction."""
    from pepsy.optimizers import TreeMPO, TreePEPO

    z_op = quimb.pauli("Z", dtype="complex128")
    terms = [
        ((z_op, z_op), (0, 3), 1.2),
        ((0, 3), "ZZ", 0.8),
    ]
    builder = py.ham_tn(shape=4, data_type="complex128")
    tree_plan = py.TreePlan.from_order([0, 1, 2, 3])

    tree_mpo = builder.to_tree_mpo(
        tree_plan,
        terms,
        compress=False,
        cutoff=0.0,
    )
    tree_mpo_reference = builder.to_tree_mpo(
        tree_plan,
        terms,
        mode="term",
        compress=False,
        cutoff=0.0,
    )

    assert tree_mpo.max_bond() <= 2
    assert (
        tree_mpo.tree_networks[0].pepsy_tree_operator_bond
        == tree_mpo.max_bond()
    )
    assert np.allclose(tree_mpo.to_dense(), tree_mpo_reference.to_dense())
    normalized_tree_terms = builder._normalize_tree_terms(
        tree_plan,
        terms,
        phys_dim=2,
        dtype=np.dtype("complex128"),
    )
    raw_tree_mpo = TreeMPO.from_terms(
        tree_plan,
        normalized_tree_terms,
        cutoff=0.0,
        dtype=np.dtype("complex128"),
        compress=False,
    )
    assert np.allclose(tree_mpo.to_dense(), raw_tree_mpo.to_dense())

    pepo_builder = py.ham_tn(shape=(3, 3), data_type="complex128")
    pepo_terms = [
        ((z_op, z_op), ((0, 0), (2, 2)), 1.2),
        ((z_op, z_op), ((0, 1), (2, 1)), 0.8),
    ]
    tree_pepo = pepo_builder.to_tree_pepo(
        pepo_terms,
        map_mode="span-up",
        compress=False,
        cutoff=0.0,
    )
    tree_pepo_reference = pepo_builder.to_tree_pepo(
        pepo_terms,
        map_mode="span-up",
        mode="term",
        compress=False,
        cutoff=0.0,
    )

    assert np.allclose(tree_pepo.to_dense(), tree_pepo_reference.to_dense())
    normalized_pepo_terms = pepo_builder._normalize_tree_terms(
        tree_pepo.plan,
        pepo_terms,
        phys_dim=2,
        dtype=np.dtype("complex128"),
    )
    raw_tree_pepo = TreePEPO.from_terms(
        tree_pepo.plan,
        normalized_pepo_terms,
        dims=2,
        dtype=np.dtype("complex128"),
    )
    assert np.allclose(tree_pepo.to_dense(), raw_tree_pepo.to_dense())


def test_tree_pepo_compress_validates_once_after_the_full_sweep(monkeypatch):
    """TreePEPO does not re-run its quadratic validation for every edge."""
    from pepsy.optimizers import TreePEPO, TreePepsPlan

    z_op = quimb.pauli("Z", dtype="complex128")
    builder = py.ham_tn(shape=(3, 3), data_type="complex128")
    terms = [
        ((z_op, z_op), ((0, 0), (2, 2)), 1.2),
        ((z_op, z_op), ((0, 1), (2, 1)), 0.8),
    ]
    plan = TreePepsPlan.from_shape((3, 3), map_mode="span-up")
    normalized = builder._normalize_tree_terms(
        plan,
        terms,
        phys_dim=2,
        dtype=np.dtype("complex128"),
    )
    operator = TreePEPO.from_terms(
        plan,
        normalized,
        dims=2,
        dtype=np.dtype("complex128"),
    )

    calls = []
    original_validate = TreePEPO.validate

    def capture_validate(target, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_validate(target, *args, **kwargs)

    monkeypatch.setattr(TreePEPO, "validate", capture_validate)
    operator.compress(max_bond=2, cutoff=1e-12)

    assert len(calls) == 1
    assert all(call.get("check_canonical") for call in calls)
    assert operator.validate()
