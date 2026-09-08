# TreeOptimizer mode review and bottlenecks — 2026-09-08

Reviewed the current working tree, including the preceding MPS-parity work.
The review covers ordinary gate modes, explicit chain-MPO replay, shared
TreeFIT, and dense/native numerical paths. Performance measurements below
are CPU measurements, not predictions for GPU or large-chi workloads.

## Changes made

- FIT canonical preparation stops at the first active-block node. If the
  current center is already in the block, its exterior is already isometric;
  moving the center internally before replacing that block is redundant.
  Only actually changed path tensors invalidate cached messages. Local
  factorization still determines the final center.
- Cache each block-size/direction traversal order within the fixed-region
  run. This preserves update order and avoids repeating medial-node/path
  calculations. No persistent all-pairs geometry cache was added.
- Cache compression-hook capabilities for the current function, one entry
  per optimizer. Replacing an integration hook triggers reinspection; legacy
  signatures still work. Public optimizer copies do not inherit this cache.
- Three-node factorization now peels a newly exposed middle node when the
  caller explicitly requests an endpoint center. Previously that valid
  `fit_block(..., center=endpoint)` case raised `KeyError` during installation.
- Default `fit_init_strategy="auto"` preserves dense SRC initialization and
  chooses graded direct compression for native fermionic states. Previously
  native DMRG with the default SRC guess could fail at its first compression.
  Explicit unsupported native SRC/DM requests still raise; the requested
  output compression method is never silently changed.
- `guess-direct` now actually applies/compresses the operator on a private
  copy, as its name implies. Plain `direct` keeps the current state as the
  initial guess. The exact target remains separate from either guess.
- Corrected stale skill guidance claiming ordinary `auto` replay still uses
  the dedicated two-site kernel. Ordinary gates use the TreeMPO subtree route;
  explicit low-level APIs retain the dedicated kernel.

## Dense CPU comparison

Twelve qubits, balanced binary tree, initial random tree bond dimension four,
`chi=4`, complex128 NumPy, cutoff `1e-12`, one BLAS thread. The input state was
normalized. Six fixed seeded gates comprised four nonlocal two-qubit gates
and two three-qubit gates. The table gives the median of three unprofiled
replays after warm-up, with a fresh optimizer per replay. Constructor/layout
work is excluded; gate factorization on cache misses is included. Repeated
replay on the same optimizer can reuse its existing gate-factor cache.

Fidelity is the normalized overlap against an independently evolved dense
state after all six gates. It is not the optimizer's norm-loss proxy.

| Mode | Six-gate replay (ms) | Fidelity |
| --- | ---: | ---: |
| auto | 46.03 | 0.569768 |
| direct | 46.06 | 0.569768 |
| dm | 45.96 | 0.569768 |
| sdc | 46.23 | 0.569768 |
| src | 54.59 | 0.569768 |
| zipup | 28.20 | 0.030977 |
| mpo | 46.22 | 0.569768 |
| submpo | 28.93 | 0.571985 |
| tree_mpo_direct | 46.10 | 0.569768 |
| tree_mpo_dm | 45.83 | 0.569768 |
| dmrg | 635.43 | 0.574999 |
| dmrg1 | 638.13 | 0.577043 |
| dmrg2 | 637.05 | 0.574999 |
| dmrg3 | 598.46 | 0.575257 |

`submpo` uses prebuilt chain MPOs; their construction is excluded, unlike
ordinary gate-to-TreeMPO construction. Its factorization/truncation order
also differs, so it is not an interchangeable route with identical costs.
Tree `sdc` currently uses the deterministic edge-SVD sweep; it is not Quimb's
independent chain SDC algorithm. Tiny tensors do not benefit from SRC here.

Zipup is fast in this case but much less accurate at this restrictive cap:
its intermediate truncations see unvisited, noncanonical environments. The
lossless check below distinguishes this approximation cost from an exact
contraction error. This single circuit is not a general mode ranking.

All fourteen mode spellings were also checked on six qubits with `chi=64`
and cutoff zero, against dense evolution and canonical metadata validation.
Every fidelity was one within `2.3e-16`.

## Measured hot paths

The percentages below are cumulative cProfile envelopes from separate runs.
Profiling increases absolute runtime; nested envelopes must not be added.

| Route | Main measured costs |
| --- | --- |
| direct | Subtree compression 39%; QR message routing 16%; TreeMPO construction 22%. Compression includes its QR/SVD wrappers. |
| zipup | TreeMPO construction 35%; the remaining major work is local message contraction and outgoing splits. |
| dense dmrg2 | Effective-block/environment contraction 44%; center preparation 32%; warm guess 10%; block factorization 4%. |
| dense dmrg3 | Effective-block/environment contraction 43%; center preparation 24%; warm guess 10%; block factorization 10%. |
| native dmrg2 | Effective-block/environment contraction 61%; center preparation 12%; warm guess 11%; block factorization 5%. |
| native dmrg3 | Effective-block/environment contraction 58%; center preparation 7%; warm guess 12%; block factorization 11%. |

Native self-time is concentrated in block fusion/unfusion, sector alignment,
transpose, and blockwise contractions. Native sparsity reduces arithmetic
but does not eliminate this metadata work. Copying complete optimizer
histories is no longer in the FIT warm-start path and is not a dominant
runtime cost in either profile.

The hot-path changes reduced total `canonize_edge_` calls from 1623 to 1457
for dmrg2, and 1279 to 1041 for dmrg3. Traversal-order construction fell from
48 to 24/36 calls respectively. Compression-hook signature inspection fell
from once per edge to once per unchanged hook. Before/after dense output
vectors for direct, SRC, zipup, and DMRG1/2/3 were bit-identical in this
seeded run. Median dmrg2 time fell from 664 to 637 ms and dmrg3 from 641 to
598 ms. These modest timing differences include run-to-run noise; even the
unchanged zipup control varied by about 3%. The reduced operation counts are
stronger evidence than a universal speedup claim.

## Native capability and accuracy checks

A separate four-site spinful U1U1 complex128 test used two hopping gates,
`chi=8`, cutoff zero, and a graded direct `chi=64` reference. Initial
occupations were paired/even. Medians of two warmed unprofiled replays:

| Mode | Two-gate replay (ms) | Native reference fidelity |
| --- | ---: | ---: |
| direct | 31.24 | 0.997589 |
| zipup | 21.21 | 0.944393 |
| dmrg2, automatic graded guess | 210.78 | 0.997589 |
| dmrg3, automatic graded guess | 186.96 | 0.997589 |

Native DM and SRC compression remain explicitly unsupported. Native DMRG
now works with automatic initialization; requesting a native SRC guess
explicitly still fails. Odd-parity fermionic TreeFIT remains unsupported.
The native exact-state regression additionally verifies every local update
through three/two/one-node refinement, preventing compensating phase errors
from hiding behind a terminal-only fidelity check.

## Next improvements, in priority order

The first three priorities below were subsequently implemented in the
[FIT execution follow-up](tree_fit_execution.md), with explicit traversal
and native contraction options and an automatic one-node local solve.

1. Prototype a tree traversal that reduces center travel and message
   invalidations, especially between branches. Current depth-ordered passes
   revisit paths repeatedly. Changing update order can change truncated
   results and convergence, so it needs an explicit policy and fidelity
   comparison rather than a silent reorder.
2. Reduce native local-contraction fusion/unfusion and sector-metadata work
   using supported upstream primitives. Preserve graded ordering and charge
   metadata; dense conversion is not an optimization for this path.
3. Explore an exact one-node shortcut for FIT replay with equivalent norm and
   diagnostic semantics. There is no bond-growth variational problem for a
   truly one-node active region. This review's timing circuit contained only
   two-/three-qubit gates, so that potential benefit is not measured here.
4. Reduce operator/guess setup on small updates: preserve bounded reuse of
   existing gate factors, investigate subtree-only operator construction,
   and avoid rebuilding a temporary optimizer around every compressed guess.
   The existing guess setup is approximately 10% of the measured FIT runtime.
5. Offer bounded diagnostic retention for very long circuits. FIT diagnostics
   currently accumulate independently of `record_history`; this is a memory
   scaling concern, distinct from the measured contraction bottleneck.

At larger chi or higher tree degree, the number and dimensions of exposed
block legs grow rapidly. Effective-tensor memory and QR/SVD arithmetic may
then dominate. Measure the actual circuit/backend before changing threads,
arity, block size, or compression policy. A fast small-chi CPU result does
not establish a fast GPU or high-chi route.

## Compatibility audit and validation

Rechecked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The required
[Abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
again returned an error. Installed-source and numerical probes supplement it.

Installed versions: Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`,
Symmray `0.3.2.dev7+gd63bb4e3f`, Torch `2.6.0+cu124`.

Actual probes included Tensor/Network `copy`, `tensor_contract`, `Tensor.split`,
`MatrixProductOperator.from_dense(A, dims=2, sites=None, L=None, ...,
**split_opts)`, `cotengra.array_contract`, native `phase_flip` and QR/SVD,
plus NumPy/Torch/Symmray `tensordot`/QR/SVD dispatch. Quimb's `svd`, `svd:eig`,
`svd:rand`, and `qr` registry entries resolved to the expected drivers. No
native QR policy, dependency, installed package, or global dispatch changed.

**Adopt:** fewer lossless center moves, existing native direct compression
for automatic native guesses, and unchanged supported upstream contractions.
**Defer:** chain-only algorithms, odd-parity FIT, and reordered traversals.
**Compatibility shim:** retain the existing local legacy compression-hook
signature adapter with bounded capability caching; no new upstream shim.
**Prototype:** none installed in production.

Validation: **585 passed** in the tree/FIT/Quimb/public-API focused run and
**188 passed** across tree stabilizer, TreePeps, TreePEPO, TreeMPO, tree
sampling, and tree energy suites: **773 tests total**. Repository Ruff,
tree skill/catalog validation, and `git diff --check` passed. The full
repository suite was not rerun; no unrelated subsystem changes were made.
Temporary profiling harnesses, NumPy reference vectors, and cProfile data
remain under `/tmp`, outside the package repository.
