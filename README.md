# Pepsy Library

`pepsy` is a Python package for boundary-MPS based PEPS norm contraction and related DMRG fitting workflows.

Current package version: `0.0.0` (from `pepsy/VERSION`).

## Package Layout
- `pepsy/`: installable library code
  - `boundary_states.py`: boundary state initialization (`BdyMPS`)
  - `boundary_sweeps.py`: sweep/contraction runner (`CompBdy`)
  - `boundary_norm.py`: input preparation + contraction (`prepare_boundary_inputs`, `ContractBoundary`)
  - `dmrg_fit.py`, `dmrg_helpers.py`, `linalg_registrations.py`
- `example/`: example notebooks
- `tests/`: package tests

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Optional backends:
# pip install -e .[torch]
# pip install -e .[jax]
```

## Quick Usage
```python
import pepsy
from pepsy import BdyMPS, CompBdy, ContractBoundary, prepare_boundary_inputs

print(pepsy.__version__)
```

## Notes
- `.gitattributes` marks notebooks as binary to avoid noisy diffs.
- `.gitignore` excludes checkpoints, caches, `cash/`, and `nohup.out`.
