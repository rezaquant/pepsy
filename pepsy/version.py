"""Repository version helpers."""

from pathlib import Path


def get_version() -> str:
    """Return version from package ``VERSION`` or repository-root ``VERSION``."""
    package_version = Path(__file__).with_name("VERSION")
    if package_version.exists():
        return package_version.read_text(encoding="utf-8").strip()

    repo_version = Path(__file__).resolve().parent.parent / "VERSION"
    if repo_version.exists():
        return repo_version.read_text(encoding="utf-8").strip()

    return "0.0.0"


__version__ = get_version()
