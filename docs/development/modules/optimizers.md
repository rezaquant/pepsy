# pepsy.optimizers

This package contains the high-level optimizers that sit above Pepsy's tensor,
operator, boundary, and solver layers. The core package remains `pepsy`;
`pepsy_examples` is for examples and external testing, and `tc_gauge` is an
important downstream time-compression consumer that depends on Pepsy behavior.

## Layout

- `mps/`: MPS gate-stream optimization.
  - `optimizer.py`: `MpsOptimizer`.
  - `compression.py`: extraction target for compression backends.
  - `normalization.py`: extraction target for non-unitary normalization logic.
  - `diagnostics.py`: extraction target for fidelity/progress records.
- `mpo/`: MPO gate-stream optimization.
  - `optimizer.py`: `MpoOptimizer`.
  - `targets.py`: extraction target for gate-pair and DMRG target builders.
  - `compression.py`: extraction target for compression backends.
- `peps/`: PEPS/PEPO gate-stream optimization.
  - `optimizer.py`: `PepsOptimizer`.
  - `gates.py`: extraction target for gate routing and target application.
  - `warmstart.py`: extraction target for warm-start construction.
  - `routing.py`: extraction target for sweep/global backend routing.
  - `diagnostics.py`: extraction target for infidelity/progress records.
- `sweep/`: local PEPS slice optimization.
  - `optimizer.py`: `SweepOptimizer`.
  - `environments.py`: Quimb MPS boundary store and engine selection helpers.
  - `local_objective.py`: extraction target for local objective assembly.
  - `traces.py`: extraction target for sweep traces and progress summaries.
- `stabilizer_tn/`: `StabilizerMpsSimulator` / `MpsStabOptimizer`, `STNState`,
  and typed STN diagnostic records for the Stim-tableau plus coefficient-MPS
  simulator. See `../plans/stabilizer_tn.md` for its implementation record and
  `docs/howto/stabilizer_tn_magic.md` for exact cooling, greedy checkpoints,
  and immediate versus deferred MAST injection.
- `planning.py`: non-executing physical-versus-stabilizer and
  MPS-versus-tree circuit advice using measured frame supports and explicit
  chi-scaled work proxies.
- `qmera/`: schedule-first qMERA local-energy objectives, parameter
  dictionaries, compiled lightcone contractions, schematics, and
  Symmray-native fermion helpers.
- `global_opt.py`: whole-network variational optimization helpers.

## PEPS optimizer stack

`PepsOptimizer` is the outer gate-stream driver. It builds exact two-site
targets, compresses warm starts to the requested PEPS bond dimension, and then
optionally refines with `SweepOptimizer` or `GlobalOptimizer`.
It exposes `boundary_engine` and `boundary_options` so sweep cleanup can use
the same boundary implementation choices as `SweepOptimizer` directly.

`SweepOptimizer` is the local PEPS slice optimizer. It keeps two environment
stores:

- `bdy` for the trial norm `<state|state>`.
- `bdy_overlap` for the overlap `<target|state>`.

The current default store is `pepsy.boundary.states.BdyMPS`, whose `mps_b`
dictionary contains reusable boundary MPS entries keyed as `Y{i}_l`,
`Y{i}_r`, `X{i}_l`, and `X{i}_r`. `SweepOptimizer` selects a row or column,
attaches the needed left/right environments, optimizes the packed local slice,
and then advances the boundary for the next slice with
`pepsy.boundary.sweeps.CompBdy`.

## Boundary engines

The default dense Pepsy boundary engine is:

```text
build_bra_ket(...) -> BdyMPS(...) -> CompBdy.move_bdy/move_step_bdy(...)
```

That path uses local FIT/DMRG-style boundary updates and works well for the
dense backends it was designed around. It is less suitable for Symmray-backed
networks, where Quimb's native boundary contraction and environment routines
can preserve backend semantics better.

`SweepOptimizer` also supports `boundary_engine="quimb-mps"` (or `"auto"` for
Symmray-looking inputs). This builds local row/column environments with
Quimb's `compute_x_environments(...)` and `compute_y_environments(...)`, while
scalar sweep-time normalization and infidelity use Quimb's
`contract_boundary(...)` through Pepsy's public `method="mps"` metric helpers.
`PepsOptimizer(boundary_engine="auto")` keeps this same policy when it delegates
to sweep cleanup.

## MPS gate-stream optimizer

`MpsOptimizer` defaults to `direct` compression. Other replay modes include
`dmrg`, `swap`, `perm`, `svd`, `mix`, `su`, and `exact`; `mpo` remains a
compatibility alias for `direct`. For repeated evolution on a graph
with a useful one-dimensional layout, call `opt.apply_layout("quality")` once.
The MPS then stays in the selected physical order across `run()` calls and
logical readout goes through `opt.logical_order`, `opt.remap_sample(...)`, or
`opt.to_dense()`.

The persistent reorder is free only for a product MPS (`p.max_bond() == 1`).
For an entangled initial state, the default is to raise; an explicit
`allow_lossy_reorder=True` opts into one reorder using the caller's cutoff.
The deprecated `run(use_layout_finder=True)` compatibility path still performs
the old temporary reorder and swap-back and should not be used for iterated
time evolution.

Canonical metadata in `opt.info_c["cur_orthog"]` is part of the numerical state.
Local expectation and norm diagnostics should move from this tracked range,
not rescan or contract the full MPS. Any target MPS copy needs isolated
metadata. Exact mode intentionally has no canonical cache; switching back to
an MPS mode rebuilds and canonicalizes the state.

## Import style

Use clean class imports at API boundaries:

```python
from pepsy.optimizers import MpsOptimizer, PepsOptimizer, QMeraBuilder, SweepOptimizer
```

Use implementation leaves when a test or internal change needs module globals:

```python
from pepsy.optimizers.sweep.optimizer import SweepOptimizer
```

## Editing notes

- Prefer package namespaces such as `pepsy.boundary`, `pepsy.optimizers`, and
  `pepsy.tensors`; do not revive removed root-level flat modules.
- Add focused tests near the optimizer or boundary behavior being changed.
- For Symmray behavior, keep optional dependencies optional with
  `pytest.importorskip(...)`.
