"""Basic public API smoke tests for the pepsy package."""

import pepsy


def test_package_version_available():
    """Package exposes a non-empty version string."""
    assert isinstance(pepsy.__version__, str)
    assert pepsy.__version__


def test_core_symbols_exported():
    """Top-level package exports expected boundary API symbols."""
    assert "BdyMPS" in pepsy.__all__
    assert "CompBdy" in pepsy.__all__
    assert "ContractBoundary" in pepsy.__all__
    assert "prepare_boundary_inputs" in pepsy.__all__
    assert "add_diagonalu_tags" in pepsy.__all__
    assert "make_numpy_array_caster" in pepsy.__all__
