# DMRG Experiments

Code and notebooks for density matrix renormalization group (DMRG) and related tensor-network simulations. Scripts implement DMRG/cooling routines, circuit utilities, and experiment helpers; notebooks capture exploratory runs and diagnostics.

## Layout
- `dmrg_fit.py`, `boundary_sweeps.py`, `algo_cooling.py`: core DMRG and boundary/cooling routines.
- `dmrg.py`, `svd.py`: runnable scripts for DMRG workflows and SVD-based variants.
- `circuits.py`: circuit construction and Qiskit helpers.
- `quf.py`, `register_.py`: utilities (partitioning, contractions, registration) used across scripts.
- `dmrg*.ipynb`, `mps_*.ipynb`, `plot.ipynb`, `prac.ipynb`: experiments and visualizations.
- `cash/`: local cache/artifacts (ignored).
- `store/`, `z2_exact/`: reference data and helper modules.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For GPU JAX/PyTorch, install vendor wheels as needed.
```

## Notes
- `.gitattributes` treats notebooks as binary to avoid noisy diffs; use `nbdiff` or screenshots for review.
- `.gitignore` excludes checkpoints, caches, `cash/`, and `nohup.out`. Keep transient data there or outside the repo.
- Large generated data should stay out of version control or use Git LFS if required.
