# Tree FIT execution improvements — 2026-09-08

The initial implementation and audits below retain their historical defaults.
See [the adoption follow-up](#default-adoption-follow-up) for the current
TreeOptimizer default.

Implements the first three priorities from the [mode review](tree_modes_review.md):
less tree travel, faster native environments, and an exact one-node local solve.
The affected production paths are `pepsy.fitting.tree` and
`pepsy.optimizers.tree.optimizer`. Public options are documented in the
[tree API](../../api/optimizers/tree.md).

## Execution and compatibility

- `fit_traversal="depth-first"` / `TreeFIT(traversal="depth-first")` visits
  branches together using an iterative DFS from the active region's medial
  node. Multi-node blocks are anchored at their shallowest node. It preserves
  the block set, reverses the order for inward traversal, and caches orders
  within the fixed-region run. The default `"depth"` retains the previous
  update order. Finite-bond results can change with ordering.
- `fit_environment_strategy="native-blockwise"` /
  `TreeFIT(environment_strategy="native-blockwise")` supplies a local
  Cotengra implementation pair to Quimb's message/effective-tensor
  contractions. Native Symmray `tensordot(mode="blockwise")` avoids charge
  block fusion/unfusion while retaining graded permutations and phases.
  The contraction plan, QR/SVD policy, and global backend registration are
  unchanged. Dense/mixed target-state inputs are rejected for this option;
  missing public upstream capabilities raise explicitly. Default is
  `"default"`. Blockwise is not expected to win for every sector structure.
- `fit_single_node_fast_path=True` automatically skips the disposable guess
  and repeated passes for a truly one-node active region. It canonicalizes
  the exterior and projects once, records the resulting norm, optionally
  scans once, and reports `single_node_exact`. It preserves the child-seed
  draw used by compressed guesses, keeping later measurement RNG behavior.
  `fit_single_node_fast_path=False` restores iterative replay. Standalone
  `run_gate` enables it; complete-tree `run`/`run_eff` retain fixed iteration
  semantics by default. For arbitrary targets, exact means optimal in the
  fixed exterior, not global fidelity one. A one-node gate on the original
  state, including a nonunitary gate, is represented exactly.

## Measurements

Single-thread CPU, complex128, fresh optimizer per replay, construction
excluded; one warmup and median of four timed replays. Profiling runs were
separate from timed runs. Dense input: normalized 12-qubit balanced binary
tree, initial bond dimension four, six seeded nonlocal gates (four two-qubit
and two three-qubit), chi four, cutoff `1e-12`. Dense fidelity uses an
independent statevector gate application.

| Dense mode | Depth (ms) | Depth-first (ms) | Depth fidelity | Depth-first fidelity |
| --- | ---: | ---: | ---: | ---: |
| DMRG2 | 630.6 | 413.1 | 0.574999 | 0.575245 |
| DMRG3 | 598.3 | 456.2 | 0.575257 | 0.575143 |

Depth-first reduces replay time by 34% and 24% here. The DMRG3 fidelity
decreases slightly, illustrating why this remains opt-in. Lossless replay
tests separately agree with direct application to numerical tolerance.

Native input: four-site spinful U1U1 paired occupations, two real-time hopping
gates, chi eight, cutoff zero; reference is graded direct chi 64.

| Native mode | Traversal | Default environments (ms) | Blockwise (ms) |
| --- | --- | ---: | ---: |
| DMRG2 | depth | 216.0 | 155.5 |
| DMRG2 | depth-first | 211.3 | 156.1 |
| DMRG3 | depth | 193.9 | 145.1 |
| DMRG3 | depth-first | 189.7 | 150.9 |

All native fidelities were `0.997589127406466` within floating-point error.
Blockwise reduced depth-ordered replay time by 28% (DMRG2) and 25% (DMRG3).
Combining the options is not automatically better: the small native tree has
little branch travel and the timings include run-to-run noise.

Ten one-qubit X gates on the dense 12-qubit input took 112.7 ms with repeated
DMRG2 FIT and 76.0 ms with the automatic local solve (33% less time). Both
had fidelity one. Separate regressions cover a nonunitary root gate with an
explicit network exponent, native nonunitary local gates on an entangled
state, and arbitrary-target projection with a fixed exterior.

The remaining dominant costs are local environment construction, canonical
movement, and guess setup. In separate cProfile runs, dense DMRG2 effective
tensors (including their messages) consume about 41% after depth-first,
canonical preparation 22%, and guess setup 15%. Native DMRG3 blockwise
effective tensors consume about 44%, guess setup 15%, and factorization 13%.
Nested message percentages must not be added to effective-tensor time.
Larger chi or arity can instead make effective-tensor memory and QR/SVD
arithmetic dominant. These measurements establish neither GPU performance
nor universal fidelity improvement.

## Upstream audit

Rechecked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The required
[Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
again returned an error; actual installed source/signature and numerical
probes supplement it.

Installed versions: Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`,
Symmray `0.3.2.dev7+gd63bb4e3f`, Torch `2.6.0+cu124`.

Actual public API probes:

- `AbelianArray.tensordot(self, other, axes=2, mode="auto", preserve_array=False)`
  accepts `auto`, `fused`, and `blockwise`.
- `FermionicArray.tensordot(self, other, axes=2, preserve_array=False, **kwargs)`
  performs graded preparation and dummy-mode handling around the Abelian
  contraction, forwarding `mode`. Public `symmray.tensordot` additionally
  handles scalar fallback. The implementation does not bypass this wrapper.
- `ContractionTree.get_contractor(self, order=None, prefer_einsum=False,
  strip_exponent=False, check_zero=False, implementation=None, autojit=False,
  progbar=False)` accepts an `(einsum, tensordot)` implementation pair.
  `array_contract_expression(..., **kwargs)` and
  `quimb.tensor.tensor_contract(..., **contract_opts)` forward it. Installed
  contractor code retains native Autoray transpose and the planned order.
- A native two-tensor contraction with the implementation pair succeeded;
  full native FIT then agreed with default contractions on NumPy and Torch
  blocks. Autoray's Symmray tensordot registration was identical before and
  after both policies. The existing dense/Quimb contraction regressions pass.

**Adopt:** exact one-node projection and supported public contraction APIs.
**Prototype:** opt-in branch traversal and native blockwise execution,
capability-gated rather than version-selected; retain prior defaults while
broader workload data is gathered. **Compatibility shim:** none added.
**Defer:** automatic selection between fused and blockwise per message,
odd-parity FIT, chain-only algorithms, larger-chi/GPU tuning, and further
operator/guess setup reduction. No installed dependencies were edited.

## Validation

The focused tree/FIT/Quimb/public-API run passed **602 tests**, including
17 execution-policy regressions in `tests/test_tree_fit_priorities.py`.
The tree stabilizer, TreePeps, TreePEPO, TreeMPO, sampler, and energy suites
passed **188 tests**, for **790 passing tests total**. Repository Ruff, the
tree skill validator, skill catalog validator, and `git diff --check` passed.
The full repository suite was not rerun because this change is confined to
tree FIT and its optimizer integration; shared tree consumers were checked
explicitly instead. Benchmark harnesses and profiles remain under `/tmp`.

## DMRG/direct timing audit (2026-09-08)

A reported Torch table showed direct at 12.759 s and DMRG1/2/3 at
89.363/107.652/145.302 s. The producing script, device, dtype, and iteration
options were not supplied, so those exact measurements were not reproduced.
The current default permits four iterations with two directional passes each,
plus a disposable SRC guess. Named DMRG2/3 include larger-block factorization.

A separate CPU probe used a balanced 12-qubit tree, initial random bond 8
(seed 3), twelve long-range random two-qubit unitaries (NumPy RNG seed 4,
QR of complex Gaussian 4x4 matrices, supports `(i, (i+5) % 12)`), cap 8,
complex128, one Torch thread, and SRC guess seed 13. The untruncated reference
used direct with cap 64 and zero cutoff. Median replay times over three
sequential runs, without profiling or fidelity contractions inside the timer:

| Mode | Traversal | Seconds | Final exact fidelity |
| --- | --- | ---: | ---: |
| direct | depth | 0.0328 | 0.258763168 |
| dmrg1 | depth | 0.3799 | 0.255465384 |
| dmrg1 | depth-first | 0.2464 | 0.259151942 |
| dmrg2 | depth-first | 0.2507 | 0.262097701 |
| dmrg3 | depth-first | 0.2932 | 0.263445759 |

All DMRG fits consumed four iterations; block traces were `(1,1,1,1)`,
`(2,2,1,1)`, and `(3,3,2,1)` respectively. Automatic complex128 rtol was
1e-9. A separate depth-first DMRG1 run with rtol=1e-5 still used all four
iterations and returned the same fidelity; its small timing difference is
not evidence of tolerance-driven savings.

A cProfile run of depth-ordered DMRG1 took 0.865 s: 0.679 s in `run_gate`,
including 672 block updates, 0.346 s preparing canonical centers and 0.302 s
building effective blocks. Profiling adds substantial Python overhead; these
times must not be compared directly with the unprofiled medians above.
Depth-first reduced unprofiled DMRG1 time by about 35% in this small probe,
but traversal changes finite-sweep results. Keep it opt-in pending the user's
actual workload. This establishes a traversal bottleneck and substantial
scheduled work, not a correctness failure or a general GPU speed claim.

Rechecked the required Quimb, Autoray, Cotengra, and Symmray sources; the
Symmray Abelian HTML page remained unavailable. Installed versions were
Quimb 1.15.1.dev39+g369d09b9d, Autoray 0.11.1.dev1+gc56f64427,
Cotengra 0.8.3.dev6+g08fe1a3a1, and Symmray 0.3.2.dev6+ga17699db6.
Probed actual `tensor_contract` and `tensor_split` signatures; no upstream
API or numerical dispatch change was needed. Classification: retain the
existing depth-first prototype; defer default changes and larger-rank/GPU
claims until the reported workload is available. No runtime code changed.

Validation: all 51 incremental-environment and execution-policy tests passed;
whitespace checks passed. The full repository suite was not rerun for this
documentation-only audit.

## Default adoption follow-up

At the user's request, TreeOptimizer now defaults to
`fit_traversal="depth-first"` for generic `dmrg` and named `dmrg1/2/3`.
Automatic initial guesses remain SRC for dense states and direct for native
fermionic states. Explicit traversal/guess choices and each mode's iteration,
rank-growth, and block-transition schedules remain authoritative. Copies
preserve the selected policy. Standalone TreeFIT retains its `"depth"`
default, so shared callers such as TreePeps do not implicitly change order.
Finite-sweep results can change; explicit `fit_traversal="depth"` restores
the former ordering.

Rechecked the required upstream sources and actual public `tensor_contract`
and `tensor_split` signatures in the active environment. Installed versions
remain Quimb 1.15.1.dev39+g369d09b9d, Autoray 0.11.1.dev1+gc56f64427,
Cotengra 0.8.3.dev6+g08fe1a3a1, and Symmray 0.3.2.dev6+ga17699db6.
Symmray's Abelian HTML page remained unavailable. Classification: adopt the
already validated traversal at the optimizer API boundary; no numerical
driver, dependency, or compatibility shim change.

The lossless replay regression now exercises all four modes with omitted
traversal and initialization options, checks the actual SRC/DFS diagnostics,
and validates the final canonical state. Explicit legacy traversal and alias
normalization are covered through optimizer copies; native automatic-guess
regressions retain their direct path.

Validation: 54 focused tests passed; the broader tree/FIT/sampler/trajectory
selection passed 617 tests, skipped 4, and deselected the independently
reproduced upstream DM complex64 failure recorded in the SRC audit. Ruff,
skill catalog validation, and whitespace checks passed. The full repository
suite was not run; this default change is confined to TreeOptimizer and its
replay callers, with standalone TreeFIT's default preserved.
