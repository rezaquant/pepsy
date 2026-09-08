# Leaning on quimb (and cotengra / autoray)

pepsy already builds on the `quimb` ecosystem — `quimb`, `cotengra`, and
`autoray` are hard dependencies (`pyproject.toml`). The BP workstream should
**wrap quimb where possible** rather than reimplement message passing, gauging,
and contraction from scratch.

## What quimb already provides

- `quimb.tensor.TensorNetwork` — the network object pepsy already uses (pepsy's
  `build_bra_ket` returns quimb-tagged networks; `BdyMPS` consumes them).
- **Belief propagation** — `quimb.tensor` ships BP variants (dense and "loopy"
  1-norm / 2-norm message passing) plus BP-based **gauging** of tensor networks
  (the Tindall–Fishman gauge). These cover the BP fixed point, the gauge, and
  basic environment extraction.
- **Contraction** — `cotengra` path optimization, which pepsy already wraps via
  `build_optimizer(...)` / `contraction_opt="auto-hq"`. Reuse this for the small
  loop-excitation sub-networks in the loop expansion.
- **Backends** — `autoray` dispatch lets the same code run on numpy / torch /
  jax / cupy, and is exactly how `symmray` arrays plug in (see `symmray.md`).

## Verified integration map

| pepsy need | quimb surface | notes |
| --- | --- | --- |
| Build double-layer network | `build_bra_ket` → `qtn.TensorNetwork` | already done |
| BP fixed point | `quimb.tensor.belief_propagation.{L1BP,HV1BP,D1BP,D2BP}` | pepsy filters constructor and `run` options against the installed signatures |
| BP gauge | quimb BP gauging helpers (`gauge_all`-family) | doubles as simple-update gauge / initializer |
| BP environments / RDMs | quimb message → local env contraction | may need a thin pepsy adapter |
| Sub-network contraction | `cotengra` via pepsy `build_optimizer` | do **not** add seed kwargs (tests assert absence) |
| Generalized loops | `TensorNetwork.gen_gloops(**gloop_opts)` | capability-checked and forwarded by scalar and 2-norm loop-cluster APIs |
| Periodic lattice bonds | `qtn.LatticeBondMap` | keeps length-two periodic directions' wrap bonds distinct |
| Long-range MPO gate | `MatrixProductOperator.gate_sandwich_with_auto_swap` | explicit opt-in `pepsy.gate_mpo_auto_swap`; no dense fallback |
| Backend-native random data | `autoray.random.array` | used for FIT warm starts with a NumPy fallback for older Autoray |

The optional surfaces above are detected at execution time. Missing newer
Quimb or Autoray capabilities either keep the existing path or raise a focused
error at the explicit opt-in call; existing defaults are not changed. The
regular `gate` and `gate_simple` paths forward `dagger` and `transpose` only
to the user gate, never to internal routing SWAPs.

## Integration guidelines

- **Adapters, not forks:** accept pepsy-tagged networks, return pepsy result
  dataclasses (`BPResult`), so users never have to context-switch APIs.
- **Version guarding:** quimb is already required; if BP needs a newer quimb,
  guard the import and document the minimum version. Keep behavior graceful if
  an older quimb lacks a feature.
- **Reuse pepsy's optimizer builders** for contraction; don't introduce parallel
  contraction config.
- For MPS gate streams, treat Quimb's `info`/`cur_orthog` metadata as part of
  the algorithm state. Reuse a known canonical range for local expectations
  and one-site norms; do not replace it with a full-network norm contraction.
  When building an uncapped diagnostic target from `p.copy()`, use a separate
  info dictionary so the target cannot corrupt the live optimizer's center.
- A persistent site layout is a bookkeeping permutation over an MPS. It can be
  relabelled without SVD only for `p.max_bond() == 1`; otherwise make the
  one-time reorder explicit and caller-controlled. Keep logical readout as an
  axis/sample remap rather than restoring the physical MPS order every step.
- Exact MPS replay is a separate contracted-TensorNetwork path. It does not
  consume canonical metadata, and returning to an MPS backend requires
  rebuilding and canonicalizing an MPS first.
- **Decision to record (M1):** how much BP logic lives in pepsy vs. is delegated
  to quimb. Write the outcome into `history/` and into `../plans/project.md` §7.

## References

- quimb docs: https://quimb.readthedocs.io/ (tensor + belief propagation).
- cotengra: https://cotengra.readthedocs.io/.
- autoray: https://github.com/jcmgray/autoray.

## 2026-08-31 compatibility audit

- Installed development stack: Quimb `1.15.1.dev37+gdf03dbe79`
  (`df03dbe7989fe19eeb78ca78ea19a87b44da631a`), Autoray
  `0.11.1.dev1+gc56f64427` (`c56f644279f560e93d884ddfb2d7b0b60032382f`),
  Cotengra `0.8.3.dev6+g08fe1a3a1` (`08fe1a3a1398feb4ef667cf7009dc7a47bcdbb81`),
  and Symmray `0.3.1` (`1eaa48c9bdc2d128abed936dbe06a131105ab2e0`).
- Probes confirmed the current Quimb surfaces for SDC compression, seeded
  SRC/FIT, gate transforms, BP constructor/run options, generalized-loop
  options, `LatticeBondMap`, and MPO auto-swap. The upstream `safe_inverse`
  now handles a one-dimensional Symmray `BlockVector` directly.
- The `SimpleUpdateGen` regression now accepts Quimb's fixed long-range gate
  behavior while retaining an exact compatibility assertion for older Quimb
  releases. The safe-inverse workaround is installed only when a behavior
  probe shows that the installed Quimb build needs it.
- Loop-series resummation now forwards Quimb's newer required `num_tensors`
  argument while ignoring it on older Quimb builds. Cluster BP option probes
  use the concrete `d1bp`/`d2bp` method names rather than Pepsy's `1norm`/
  `2norm` labels.
- Tree energy optimizers now initialize their Torch linalg policy, and native
  CPU complex64 QR bypasses unrelated process-global Autoray registrations
  when an earlier autodiff run installed a stabilized rule.
- Focused validation: the pre-fix compatibility run was 141 passed, 1 skipped,
  and 1 stale expectation; after the fixes, the focused compatibility set was
  164 passed and 1 skipped, with Ruff clean. Generalized-loop options are now
  capability-checked with a focused error on older Quimb signatures. The full
  headless suite passed with 3100 passed and 39 skipped.

## 2026-09-02 tree SDC/SRC audit

- The installed Quimb build is `1.15.1.dev39+g369d09b9d`. Its concrete 1D
  compressors expose both `sdc` and seeded `src`, while the arbitrary-geometry
  compressor does not expose either environment algorithm.
- `TreePeps` path operator application now adapts the separate PEPO and state
  layers into a plain temporary `TensorNetwork` with shared site tags, then
  reinstalls Quimb's one-tensor-per-site result into the geometry-owning
  wrapper. `compression_layout="fused"` retains the earlier fused path.
  Branching `TreePeps` and `TreeOptimizer` retain their native tree sweep;
  `src` uses only the local dense `svd:rand` split there.
- This is an API/dispatch integration, not a generalized implementation of
  the paper's CBC algorithm. CBC needs projected Cholesky environments and a
  distinct leaves-to-root / root-to-leaves precomputation, so it remains an
  explicit future compression method rather than an alias.

## 2026-09-02 TreeFIT environment-cache audit

- Installed versions probed in the active Pepsy environment: Quimb
  `1.15.1.dev39+g369d09b9d`, Autoray `0.11.1.dev1+gc56f64427`, Cotengra
  `0.8.3.dev6+g08fe1a3a1`, and Symmray `0.3.2.dev6+ga17699db6`.
- API probes confirmed `TensorNetwork.contract(..., output_inds=...,
  strip_exponent=...)`, `TensorNetwork.norm(..., strip_exponent=...)`, and
  `tensor_split(..., method=..., cutoff_mode=..., bond_ind=...)` are available.
  The installed `quimb.tensor.tn1d.compress` exposes concrete `sdc`, `src`,
  `fit`, and `zipup` functions with explicit `seed` support where applicable.
- Decision: adopt Quimb's stripped-exponent and concrete 1D compressor surfaces
  for their existing MPS/TreePeps path integrations; defer applying a 1D
  compressor to arbitrary trees. TreeFIT keeps its own directed branch
  messages because a branching tree has no single 1D sweep boundary.
- TreeFIT was checked against fresh direct contractions for one-, two-, and
  three-node effective blocks and every directed message. Cache invalidation
  was checked after both local tensor updates and orthogonality-centre path
  movement, including effective blocks that depend on a changed exterior
  branch message. Fused and correctly tagged layered targets were checked,
  including multiple target bonds across one tree edge and stripped exponents.
- Focused validation after the cache fix: `499 passed, 4 skipped` across the
  Tree/TreePeps suites; `217 passed` public API/MPS FIT checks; Ruff,
  compilation, and `git diff --check` clean. A full-suite attempt remains
  subject to the repository's known macOS Matplotlib `_macosx` abort in an
  unrelated Hamiltonian drawing test; the isolated headless test passes.

## 2026-09-03 tree operator conversion and display audit

- The active environment reports Quimb `1.15.1.dev39+g369d09b9d`. The concrete
  `MatrixProductOperator.show` signature is `show(max_width=None)` and renders
  the chain's bond dimensions together with its canonical-direction markers.
  Quimb's generic operator surface does not provide a branched equivalent.
- Decision: adopt the same plain-text visual vocabulary for Pepsy's native
  `TreeMPO` and `TreePEPO` surfaces, with root-first branches, physical-site
  labels, and live bond dimensions. `ascii_tree()` returns the drawing and
  `.show()` prints it. `TreePEPO.show()` defaults to a Quimb-like coordinate
  schematic that leaves removed lattice edges as gaps; `layout="tree"` keeps
  the explicit root-first topology view. Coloring is opt-in so the returned
  drawing stays copy/paste-friendly.
- `ham_tn.to_mpo`, `to_tree_mpo`, and `to_tree_pepo` now share builder-level
  dtype, cutoff, and bond defaults while accepting per-conversion `map_mode`
  overrides. Native tree conversion is direct and does not create an
  intermediate chain MPO. The native tree compressors remain SVD-based and
  expose their geometry-specific options (`order` for `TreeMPO`, `form` /
  `center` / `reduced` for `TreePEPO`) rather than pretending arbitrary-geometry
  Quimb networks support the 1D `sdc`/`src` algorithms.
- The conversion strategy is explicit across the `to_*` builder surface:
  chain `to_mpo`/`to_pepo` retain `term`, `automaton`, and `auto`, with
  `analytic` as an automaton alias; native tree conversions support
  `mode="term"` for per-term compression and `mode="analytic"` for one final
  compression after native direct-sum assembly.
- Focused validation: 213 headless tests passed across Hamiltonian conversion,
  TreeMPO, TreePEPS, and public API suites; the updated `tn_stab.ipynb` also
  executes end to end. Documentation build was not completed because the
  active environment is missing the optional `autoapi` Sphinx extension.

## 2026-09-04 TreePeps / TreePEPO compression audit

- The active environment reports Quimb `1.15.1.dev39+g369d09b9d`, Autoray
  `0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev6+g08fe1a3a1`, and Symmray
  `0.3.2.dev6+ga17699db6`. Probes reconfirmed that generic arbitrary-geometry
  compression exposes local `compress_between`/`canonize_between` only; the
  environment compressors remain a path-only integration.
- Native `TreePeps` and `TreePEPO` compression now use a fixed-topology,
  live-rank leaf schedule by default. The scheduler scores current physical
  and virtual dimensions after each legal leaf reduction, with a deterministic
  `order="depth"` compatibility schedule. This is layout-aware through the
  retained `TreePepsPlan` paths and never relayouts an entangled state.
- Full TreePeps sweeps now defer the expensive whole-network validation until
  the final canonicality check, matching the already-batched TreePEPO path;
  standalone edge operations still validate by default. The public TreePEPO
  application, composition, Hamiltonian builder, and TreePeps optimizer
  surfaces forward the order selection.
- Focused validation after the change: the TreePeps state/optimizer suite
  passed `90` tests with one existing Quimb warning; Python compilation was
  clean. The implementation remains SVD-based on arbitrary trees; Quimb's
  path-only SDC/SRC/ZipUp and the paper's projected-Cholesky CBC algorithm are
  unchanged.
