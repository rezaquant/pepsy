# Pepsy Library

Code and notebooks for density matrix renormalization group (DMRG) and related
tensor-network simulations, with boundary-MPS based PEPS norm contraction.

Current repository version: `0.0.0` (see `VERSION`).

## Layout
- `pepsy/`: installable package.
  - `boundary_states.py`: boundary-state initialization (`BdyMPS`).
  - `boundary_sweeps.py`: boundary sweep/contraction driver (`CompBdy`).
  - `boundary_norm.py`: high-level norm preparation and contraction API.
  - `dmrg_fit.py`, `dmrg_helpers.py`: fitting and backend helper utilities.
- top-level `boundary_*.py`, `dmrg_fit.py`, `dmrg_helpers.py`, `tn_norm.py`:
  compatibility wrappers that re-export from `pepsy.*`.
- `peps_norm_.ipynb`, `peps_boundary_states.ipynb`: active notebooks.
- `cash/`: local contraction cache/artifacts.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Notes
- `.gitattributes` treats notebooks as binary to avoid noisy diffs; use `nbdiff` or screenshots for review.
- `.gitignore` excludes checkpoints, caches, `cash/`, and `nohup.out`. Keep transient data there or outside the repo.
- Large generated data should stay out of version control or use Git LFS if required.
