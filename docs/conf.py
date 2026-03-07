"""Sphinx configuration for pepsy docs."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Avoid numba cache-path issues when importing quimb-dependent modules in docs builds.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp")

project = "pepsy"
author = "pepsy contributors"

from pepsy.version import __version__  # noqa: E402

release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
]

autoclass_content = "both"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", ".DS_Store"]

nbsphinx_execute = "never"

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"pepsy {version}"
html_logo = "_static/pepsy-icon.svg"

html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rezaquant/pepsy",
            "icon": "fa-brands fa-github",
        }
    ],
}
