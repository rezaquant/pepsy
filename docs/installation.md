# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e .[torch]
pip install -e .[jax]
pip install -e .[docs]
```

Build docs locally:

```bash
sphinx-build -W -b html docs docs/_build/html
```
