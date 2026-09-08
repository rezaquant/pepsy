# Tree FIT execution improvements — 2026-09-08

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
