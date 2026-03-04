"""Repository version helpers."""

from pathlib import Path


def get_version() -> str:
    """Return the repository version from the ``VERSION`` file."""
    version_file = Path(__file__).with_name("VERSION")
    return version_file.read_text(encoding="utf-8").strip()


__version__ = get_version()
