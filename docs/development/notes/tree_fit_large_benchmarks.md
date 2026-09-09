# Tree fitting: larger performance audit (2026-09-08)

These measurements preceded the subsequent adoption of depth-first as the
TreeOptimizer default. References below to unchanged or default traversal
describe the benchmarked revision; the runtime default change is recorded
in [the execution follow-up](tree_fit_execution.md#default-adoption-follow-up).

This audit compares existing execution options without changing production
code or defaults. It follows the user's report of DMRG1/2/3 taking
89/108/145 seconds versus direct at 12.8 seconds. The original producing
script and device were not available, so these are independent workloads.

## Method

All timed replays use Torch CPU, one Torch thread, complex64 unless explicitly
marked complex128, zero cutoff, and a fixed bond cap. CUDA is unavailable on
this host. Optimizer construction, input preparation, exact reference work,
and fidelity contractions are excluded from replay time. Every replay starts
with a fresh optimizer and the same input. Repeated results use one warmup
followed by three timed replays; report their median. Screening runs use one
timed replay and are marked separately. No other benchmark runs concurrently.

Inputs are entangled random states on `TreePlan.from_order(range(n),
structure="balanced")`, using the plan's default arities. Edge dimensions are
`min(chi, 2**min(component_qubits, n-component_qubits))`. Independent complex
Gaussian node arrays use NumPy seed 91, scaled before canonicalization.
Prepare and normalize the input in complex128, then convert to the timed
dtype. This avoids overflow/underflow in synthetic-state preparation.
Input norms are independently checked by full tree overlap contraction.

Two-qubit gates are QR factors of complex Gaussian 4x4 matrices (seed 92),
with supports `(i % n, (i + n//2 + 1) % n)`. The three-qubit replay uses 8x8
gates and supports `(i, i+n//3, i+2*(n//3)) % n`. SRC uses compression seed
13; FIT guesses use seed 13 and the optimizer uses seed 14. Each configuration
receives identical gates. The fixed states/gates are workload probes, not a
statistical distribution of all circuits.

Full replay reference states are obtained by applying the gates directly to
the initial dense statevector. Large single-gate references remain exact
layered operator/state networks: contract the full bra/ket tree bottom-up,
without compressing the reference or materializing a statevector. This
independent overlap routine was checked against an eight-qubit dense result.
The large-tree overlap contractions use complex128 even for complex64 replay.
Fidelities are normalized global overlaps, not cumulative norm proxies.

`d1` denotes `mode="dmrg1"`; `dfs` is `fit_traversal="depth-first"`.
One-/two-iteration one-site fitting means `mode="dmrg", fit_block_size=1`,
`fit_n_iter=1/2`, and `fit_min_iter=1` for the one-iteration case. These are
explicit budgets, not a silent change to named DMRG1's rank-growth policy.
Each iteration contains two directional passes. Unless stated otherwise,
the initial guess is SRC and the default automatic tolerance is retained.

## Repeated measurements

Times are median seconds. Fidelity digits help compare the methods in this
fixed workload; they do not establish general accuracy guarantees. The first
16-qubit two-site replay used complex64 dense reference contractions; later
dense references and the large-tree references use complex128 contractions.

### 24 qubits, cap 64, 48 two-qubit gates, product input

This starts at `|0...0>` on the same balanced layout and uses the full 2^24
statevector as reference. Initial bond dimension is one; every final output
reaches bond 64. It exercises growth during circuit replay, not only
compression of a saturated input.

| Configuration | Seconds | Exact fidelity |
| --- | ---: | ---: |
| direct | 12.3329 | 0.242875562 |
| dm | 2.7104 | 0.242848105 |
| src | 0.4890 | 0.049488172 |
| DMRG1, depth, SRC guess | 35.7384 | 0.243264306 |
| DMRG1, DFS, SRC guess | 10.9217 | 0.242752243 |
| One-site FIT, DFS, SRC guess, 2 iterations | 5.3785 | 0.237771610 |

Both named DMRG1 traversals use `(2,2,1,1)` on 13 gates and `(1,1,1,1)`
on 35 gates. DFS reduces runtime by 69.4% with a small fidelity change.
The explicit two-iteration one-site configuration uses `(1,1)` on every
gate, saving more time while reducing fidelity. The named mode's growth
policy is deliberately not changed by this audit.

### 16 qubits, cap 32, 32 two-qubit gates

| Configuration | Seconds | Exact fidelity |
| --- | ---: | ---: |
| direct | 0.2697 | 0.483115941 |
| dm | 0.2126 | 0.483115584 |
| src | 0.1200 | 0.136155218 |
| DMRG1, depth, SRC guess | 2.3913 | 0.478975475 |
| DMRG1, DFS, SRC guess | 1.0430 | 0.473384023 |
| One-site FIT, DFS, SRC guess, 1 iteration | 0.4907 | 0.431148529 |
| One-site FIT, DFS, direct guess, 1 iteration | 0.6702 | 0.483116746 |
| DMRG2, DFS, SRC guess | 1.6345 | 0.483115792 |
| DMRG3, DFS, SRC guess | 2.9325 | 0.483115464 |

DFS saves time but slightly reduces fidelity here. A direct guess followed
by one refinement iteration reaches essentially the direct result, but still
costs more than simply running direct or DM. This is an example where fitting
is not the best time/accuracy choice.

### 24 qubits, cap 64, 16 two-qubit gates, entangled input

The exact reference contains all 2^24 amplitudes; these are full replay
fidelities, not estimates from local norm loss.

| Configuration | Seconds | Exact fidelity |
| --- | ---: | ---: |
| direct | 5.3335 | 0.562596223 |
| dm | 1.1189 | 0.562558217 |
| src | 0.1924 | 0.209724509 |
| DMRG1, DFS, SRC guess | 3.7569 | 0.560562552 |
| One-site FIT, DFS, SRC guess, 1 iteration | 1.3321 | 0.540931298 |
| One-site FIT, DFS, SRC guess, 2 iterations | 2.1594 | 0.553560849 |
| One-site FIT, DFS, direct guess, 1 iteration | 6.5300 | 0.563588334 |

DM is the strongest time/accuracy choice in this replay, subject to the
precision qualification below. The direct guess slightly improves fidelity
over direct alone, but its preparation makes it expensive. SRC-only losses
accumulate substantially over multiple gates.

### 32 qubits, cap 200, one long-range two-qubit gate

| Configuration | Seconds | Exact fidelity |
| --- | ---: | ---: |
| direct | 13.3545 | 0.999581592 |
| src | 0.2144 | 0.997553702 |
| DMRG1, DFS, SRC guess | 4.5260 | 0.999570105 |
| One-site FIT, DFS, SRC guess, 1 iteration | 1.9706 | 0.999512485 |
| One-site FIT, DFS, SRC guess, 2 iterations | 3.2593 | 0.999557964 |

Direct's timed range was 13.253–16.010 s; DFS DMRG1's was 4.522–4.580 s.
The reported speedup therefore uses medians, not the slowest direct run.
Default-tolerance DMRG1 stops after three iterations in this case. Lowering
its maximum budget from four to three adds no savings because it already
stops at three. One-/two-iteration settings impose an actual accuracy tradeoff.

Additional single-run screening, not repeated medians:

| Configuration | Seconds | Exact fidelity |
| --- | ---: | ---: |
| dm, complex64 | 3.0547 | 0.997190229 |
| zipup | 0.4726 | 0.988462276 |
| sdc | 5.5491 | 0.999248052 |
| DMRG1, depth, SRC guess | 22.4637 | 0.999570139 |
| One-site FIT, DFS, direct guess, 1 iteration | 17.1452 | 0.999581594 |
| DMRG1, DFS, direct guess | 17.0196 | 0.999581594 |
| DMRG1, DFS, SDC guess | 10.1245 | 0.999570589 |
| DMRG2, DFS, SRC guess | 73.3780 | 0.999581594 |
| DMRG3, DFS, SRC guess | >180, stopped | Not measured |

DMRG3 was explicitly terminated after exceeding three minutes. No final
fidelity is attributed to that incomplete run. DMRG2 uses `(2,2,1,1)` blocks;
its improvement over DMRG1 is small compared with its factorization cost.

### 64 qubits, cap 64, one long-range two-qubit gate

| Configuration | Seconds | Exact fidelity |
| --- | ---: | ---: |
| direct | 0.9826 | 0.768776435 |
| dm | 0.6040 | 0.768776394 |
| src | 0.0592 | 0.519925367 |
| DMRG1, depth, SRC guess | 3.2475 | 0.766128042 |
| DMRG1, DFS, SRC guess | 0.7547 | 0.766064487 |
| One-site FIT, DFS, SRC guess, 1 iteration | 0.2715 | 0.743921000 |
| One-site FIT, DFS, SRC guess, 2 iterations | 0.4325 | 0.758333663 |

This is a substantially harder compression than the cap-200 case. SRC alone
is fast but inaccurate; a small fixed number of refinement iterations may
be insufficient. Qubit count alone does not predict runtime: bond dimensions,
tree arity, active support, and local block sizes also matter.

### 16 qubits, cap 32, 16 three-qubit gates spanning branches

| Configuration | Seconds | Exact fidelity |
| --- | ---: | ---: |
| direct | 0.2609 | 0.415872041 |
| dm | 0.2426 | 0.415871977 |
| src | 0.1010 | 0.088601484 |
| DMRG1, DFS, SRC guess | 1.0286 | 0.413948762 |
| One-site FIT, DFS, SRC guess, 1 iteration | 0.3976 | 0.374739575 |
| One-site FIT, DFS, direct guess, 1 iteration | 0.5792 | 0.415871996 |
| DMRG2, DFS, SRC guess | 1.3992 | 0.415872092 |

## Where DMRG1 spends time

A separate instrumented cap-200 run kept the same guess and 60 local updates
for both traversal orders. The timings below overlap: effective-block time
includes message construction and must not be added to it.

| Counter or timer | Depth | DFS |
| --- | ---: | ---: |
| Total replay, seconds | 22.692 | 4.515 |
| Initial guess, seconds | 0.231 | 0.198 |
| Canonical preparation, seconds | 16.686 | 2.754 |
| Effective blocks including messages, seconds | 5.744 | 1.543 |
| Message calls, seconds | 5.015 | 0.821 |
| Canonical edge calls | 271 | 91 |
| Local block updates | 60 | 60 |
| Cache misses (messages and effective blocks) | 379 | 199 |

DFS avoids revisiting distant branches and preserves more useful messages.
This explains the large speedup without reducing the number of local solves.
Even with DFS, center preparation remains a major optimization opportunity.
These counters do not establish that every canonical edge call performs QR.

## DM precision check

At cap 200, complex64 DM differed appreciably from direct: fidelity
0.997190229 versus 0.999581592 and output squared norm 0.980800310 versus
0.999561977. This is actual output-state behavior, measured independently of
the cumulative diagnostic proxy.

Repeating the same seeded input/gate construction in complex128 produced:

| Mode, complex128 | Seconds, one screening run | Fidelity | Squared norm |
| --- | ---: | ---: | ---: |
| direct | 36.7463 | 0.999582102675278 | 0.999582102675270 |
| dm | 9.8694 | 0.999582102675273 | 0.999582102675270 |
| src | 0.5278 | 0.997564694944067 | 0.997564694944064 |

Thus DM is precision-sensitive on this input; complex128 restores agreement
with direct. Do not claim complex64 DM is universally accuracy-equivalent
to direct. The eigendecomposition route's numerical conditioning merits a
separate correctness investigation; this audit changes no driver or default.

## CPU SVD backend experiment

An explicit `TorchLinalgConfig(cpu_svd="torch", stabilized=False)` screening
run with both Torch and BLAS thread pools limited to one reproduced the
cap-200 direct baseline at 13.304 s and fidelity 0.999581591978139.
Switching only the supported CPU SVD choice to `scipy_gesdd` gave a repeated
median of 9.257 s (9.247–9.300 s, one warmup and three timings), with fidelity
0.999581592417037. This is about 31% faster than the repeated ordinary direct
baseline in this workload. It does not beat the SRC-plus-one-site FIT options
for runtime, but preserves direct-level fidelity.

```python
TorchLinalgConfig(cpu_svd="scipy_gesdd", stabilized=False).register()
```

This configures process-wide dispatch and was tested in isolated benchmark
processes for forward CPU simulation. It does not change the CUDA SVD driver,
and no GPU claim follows from it. Other CPU BLAS/LAPACK installations can have
different performance. No installed dependency or user notebook was modified.

## Recommended options

Recommended first change for dense DMRG replay:

```python
engine = TreeOptimizer(
    gates, state=state, tree=plan, chi=chi,
    mode="dmrg1",
    fit_traversal="depth-first",
    fit_init_strategy="auto",  # SRC for dense trees
    run=False,
)
engine.run()
```

This retains the named mode's rank-growth and iteration policy. For an
explicit faster approximation, use `mode="dmrg", fit_block_size=1,
fit_n_iter=2, fit_traversal="depth-first"` with the SRC guess. One iteration
also needs `fit_min_iter=1`. These budgets sacrifice refinement; the harder
cap-64 case shows that loss can be significant.

The measurements support the following choices, not one universal winner:

- Use DFS for DMRG when speed matters, and recheck fidelity because ordering
  can change finite-sweep results. It is the clearest optimization here.
- At high bond dimensions, keep SRC as the initial guess. A direct or SDC
  guess can dominate the entire run before fitting begins.
- SRC alone is the fastest screened option, but its error can accumulate
  severely across a replay. Validate it against a tighter reference.
- For smaller or easy targets, direct or DM can dominate all fitting options
  in time/accuracy. Do not assume DMRG must be faster because it is variational.
- Use DM with a precision check. Complex128 restored direct-equivalent
  results in the difficult cap-200 case; complex64 did not.
- Avoid DMRG2/3 as a default speed choice for saturated large bonds. Select
  larger blocks when their rank growth or fidelity gain justifies the cost.
- A direct guess plus one refinement iteration can be useful on smaller
  targets, but the crossover is workload-dependent; do not hard-code a chi
  threshold from these few examples.
- For CPU direct/SVD-heavy simulation, benchmark the supported
  `TorchLinalgConfig(cpu_svd="scipy_gesdd")` option on the actual workload.

## Scope and validation

The runtime implementation is unchanged. This audit exercises dense Torch
paths only; it does not establish native Symmray, GPU, multi-rank MPI, or
full-circuit fidelity at 32/64 qubits. A large qubit count and a single-gate
compression result must not be presented as an entire large-circuit replay.
The companion execution audit records the installed dependency versions and
the upstream API review. Benchmark harnesses and raw JSONL files are temporary
artifacts outside the package repository.

All 51 incremental-environment and execution-policy regressions passed after
the measurements. Whitespace checks passed. The full repository suite was
not rerun for this documentation-only audit. No runtime defaults changed.
