# `pepsy.boundary.sweeps`

`CompBdy(..., fit_mode="two-site")` fits the complete boundary interval with
`FIT.run_gate(block_size=2)` and explicitly retains two-site updates for every
sweep until its stopping criterion is met. It does not inherit the circuit
solver's block-to-one-site warm-up defaults. Configure it with:

- `fit_max_bond`: required for rank growth beyond the current boundary bond;
  omission safely caps direct `CompBdy` use at the current bond.
- `fit_sweep_sequence="RL"`: alternating sweep directions.
- `fit_cutoff` and `fit_cutoff_mode`: native SVD truncation policy.
- `fit_min_iter`, `fit_rtol`, and `fit_patience`: adaptive stopping policy.
- `fit_timing=True`: include elapsed and per-sweep/site timing records in each
  public `BoundaryFitDiagnostic`; add `fit_timing_sync_device=True` only when
  kernel-complete accelerator profiling is required.

The implementation builds the fixed environment once per sweep and updates
the moving environment after each pair. Thus it does not turn a linear cached
boundary sweep into a full environment rebuild at every bond.

`CompBdy.fit_diagnostics` is reset by each public run/move call. Convergence
metadata is always cheap and available; detailed timers are opt-in.

> API details are maintained as handwritten Markdown in this page.
