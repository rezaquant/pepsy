# `pepsy.operators` inventory

This is the operator-layer ownership and API-tier map. It complements the
[unified exponential guide](../../api/operators/exponentials.md) and the
[operator API plan](../plans/operator_api.md).

The rule is simple: callers should import from `pepsy.operators`; the named
family facades below are the preferred advanced discovery surfaces. Direct
imports from `pepsy.operators.mpo` and `pepsy.operators.cluster` remain
supported compatibility paths, but are not the preferred style for new
examples.

The facade now keeps canonical exports and compatibility exports in separate
internal groups. The split is intentionally organizational: both groups still
resolve from the same public namespace, while the tests enforce that aliases
point to the canonical objects rather than creating parallel implementations.

## API tiers

### Canonical construction API

These are the names to use in new code.

| Area | Canonical names | Owner |
| --- | --- | --- |
| Higher-order MPO | `MPOBasis`, `MPOParameter`, `MPOProductTerm`, `MPOLocalOperatorTerm`, `FirstDegreeMPO`, `MPOBlock`, `MPOBlockPlan`, `MPOChargeValidationReport`, `CompiledMPOExp`, `exp_mpo` | `operators.mpo_higher_order` (semantic implementation: `operators.mpo_semantic`; basis implementation: `operators.mpo_basis`; structural plan implementation: `operators.mpo_block_plan`) |
| MPO exponential metadata | `MPOPhysicalSpace`, `MPOBraiding`, `MPOCompressionReport`, `MPONumericalCompressionReport`, `MPODifferentiableCompressionReport`, `MPOAdaptiveCompressionReport` | `operators.mpo_higher_order` (space implementation in `operators.mpo_space`) |
| Shared report summary | `OperatorReportInfo` and each concrete report's `.api_info` | `operators.diagnostics` |
| Ordered MPO cluster products | `MPOClusterFactor`, `MPOClusterExpansionReport`, `MPOClusterProductExpansion`, `MPOGraphClusterProductExpansion`, `CompiledMPOClusterProduct`, `exp_mpo_cluster`, `exp_mpo_cluster_product` | `operators.mpo_product` |
| PEPO active results | `ActivePEPOBlocks`, `GraphActivePEPOBlocks` | `operators.pepo_cluster` (implementation: `operators.pepo_active`) |
| Square-lattice PEPO exponential | `PauliPEPOTerm`, `PauliPEPOBasis`, `CompiledPEPOExp` | `operators.pepo_cluster` (implementation: `operators.pepo_basis`) |
| Dense/graph PEPO clusters | `ClusterExpansionPlan`, `GraphClusterExpansionPlan`, `ClusterExpansionReport`, `ClusterLattice`, `ConnectedClusterShape`, `GraphConnectedClusterShape` | `operators.pepo_cluster` (implementation: `operators.pepo_dense`; planner boundary: `operators.pepo_geometry`) |
| PEPO model adapters | `ClusterModelAdapter`, `adapt_cluster_model`, `build_cluster_expansion_pepo`, `build_model_cluster_expansion_pepo`, `build_itf_cluster_expansion_pepo`, `build_real_time_cluster_expansion_pepo`, `build_graph_cluster_expansion_pepo` | `operators.pepo_cluster` |
| Ordered PEPO products | `PEPOClusterFactor`, `PEPOClusterProductExpansion`, `CompiledPEPOClusterProduct` | `operators.pepo_cluster` (implementation: `operators.pepo_product`) |
| PEPO composition | `compose_pepo_layers`, `compose_cluster_expansion_pepo` | `operators.pepo_cluster` |
| Pauli operator algebra | `PauliMPO`, `decompose_pauli`, `PauliCompressionReport`, `PauliBondCompressionReport` | `operators.pauli_mpo` |
| Exact MPO structural layer | `MPOAutomaton`, `MPOChannel`, `MPOTransition` | `operators.mpo_automaton` |
| Native tree operators | `TreeMPO`, `TreePEPO`, `TreeSubPEPO`, `ham_tn.to_tree_mpo`, `ham_tn.to_tree_pepo` | `optimizers.tree`, `optimizers.tree_peps`, `operators.hamiltonians` |
| Elementary gates/builders | gate constructors (`x`, `rx`, `cnot`, etc.), `gate`, `build_mpo_from_gates`, `build_pepo_from_gates` | `operators.gates` |

The API lifecycle is:

```text
terms/model → basis or plan → exp/compile_exp → semantic or active result
             → explicit to_mpo()/to_pepo()/to_tensor_network()
```

Compiled objects cache topology and plans only. Parameter values, step values,
and autodiff graphs are rebuilt on evaluation.

### Three separate exponential families

Pepsy uses three different meanings of “cluster” or “order”; they must not be
collapsed into one API:

1. **Higher-order MPO:** the SciPost history/virtual-level construction for
   `exp(step * H)` on a chain. Its `order` is a Taylor/history order.
2. **MPO cluster expansion:** connected interval or graph residuals assembled
   into one MPO. Its `cluster_size` is a spatial support cutoff.
3. **PEPO cluster expansion:** connected spatial residuals factorized into
   square-lattice or graph virtual channels. Its `order` is a local cluster
   cutoff.

The ordered product `exp(A) @ exp(B) @ exp(C)` belongs to the second or third
family according to the output representation. It is a *joint* cluster
expansion: each local cluster first forms the ordered target
`exp(A_S) @ exp(B_S) @ exp(C_S)`, subtracts lower connected contributions, and
then inserts the residual into one MPO/PEPO topology. It is not a sequence of
three independently truncated full-lattice layers. The PEPO implementation
records this invariant as `cache_info["joint_cluster_residual"]`.

The public facades are:

- `operators.mpo_higher_order` — paper-style higher-order MPOs;
- `operators.mpo_product` — connected/joint MPO cluster products;
- `operators.mpo_cluster` — compatibility facade for the MPO product family;
- `operators.mpo_semantic` — semantic/history MPO implementation;
- `operators.pepo_cluster` — connected/joint PEPO cluster expansions;
- `operators.pepo_dense` — dense/graph PEPO implementation;
- `operators.pepo_geometry` — shared PEPO geometry/planner boundary;

The older `operators.mpo` and `operators.cluster` modules remain importable
compatibility facades; they no longer own the large implementation sources.

Every construction/compression report keeps its detailed algorithm-specific
fields and also exposes `.api_info`. That common summary has the keys
`family`, `algorithm`, `representation`, `order`, `factor_count`,
`truncated`, and `differentiable`. It is intended for logging and cross-family
comparison; detailed residuals, ranks, cutoffs, and errors remain on the
concrete report.

### Advanced public construction API

These names are public and documented, but most users should reach them
through the higher-level basis or plan objects first:

- `MPOLevelToken`, `MPOLevel`: symbolic higher-order MPO history metadata.
- `MPOBlock`, `MPOBlockPlan`, `MPOChargeValidationReport`: backend-neutral
  virtual-state/local-block structure and charge validation for compiled MPOs.
- `MPOAutomaton`, `MPOChannel`, `MPOTransition`: exact channel/path assembly.
- `ClusterLattice` and connected-cluster shape records: geometry planning
  independent of tensor values.
- `ClusterInternalSymmetry`: dense cluster charge validation and metadata.
- `FirstDegreeMPO.product`, `power`, `commutator`, and exact compression:
  semantic MPO algebra rather than exponential selection.

Advanced does not mean unstable. It means that the caller is selecting a
construction layer directly and must understand its invariants.

### Compatibility API

These names remain available for existing callers but should not be used in
new examples:

| Compatibility name | Canonical replacement | Notes |
| --- | --- | --- |
| `CompiledMPOEvolution` | `CompiledMPOExp` | Direct class alias |
| `compile_evolution(...)` | `compile_exp(...)` | `MPOBasis` method |
| `time_evolution(...)` | `exp(...)` | Present on the MPO and Pauli/PEPO compatibility surfaces |
| `evolution_mpo(...)` | `exp(...)` | `MPOBasis` method |
| `dt=` | `step=` | Compatibility keyword for the exponential scalar |
| `evaluate(...)` | `exp(...)` | Retained on compiled/basis APIs |
| `MPOClusterBasisExpansion` | `MPOClusterProductExpansion` | Historical class alias |
| `CompiledMPOClusterExp` | `CompiledMPOClusterProduct` | Historical class alias |
| `ClusterBasisExpansion` | `MPOClusterProductExpansion` | Historical class alias |
| `ClusterExpansionBasis` | `MPOClusterProductExpansion` | Historical class alias |
| `ClusterExpBasis` | `MPOClusterProductExpansion` | Historical class alias |
| `MPOClusterExpansion` | `MPOClusterProductExpansion` | Historical class alias |
| `mode="algorithm4"` / `mode="paper_algorithm4"` | `mode="folded"` | Historical mode spellings |
| `mode="optimal"` / `mode="paper_optimal"` | `mode="exact"` | Historical mode spellings |
| `mode="approximate"` / `mode="paper_approximate"` | `mode="hybrid"` | Historical mode spellings |

`ham_tn` is a stable, older Hamiltonian-builder spelling documented in its
own API page. It should remain available while a future naming review decides
whether a clearer builder name is worth a migration path.

### Internal implementation

The following are not part of the supported caller API:

- Any leading-underscore helper in `operators.mpo`, `operators.cluster`,
  `operators.mpo_cluster`, `operators.pauli_mpo`, or `operators.mpo_automaton`.
- `_mpo_sparse.SparseVirtualTensor` and its conversion helpers.
- Backend-specific array normalization, scatter, SVD, embedding, residual,
  and factorization helpers.
- Cache dictionaries and private execution plans stored on basis/result
  objects.

Internal code may be split or rewritten when the public contracts and focused
tests remain unchanged.

## Normalization boundary decision

The first refactoring audit did not introduce a forced universal term parser.
MPO term normalization is already shared by `MPOBasis`, `exp_mpo`, and the MPO
cluster builders through `operators.mpo`. PEPO normalization is deliberately
separate because it interprets support through square-lattice directions,
connected cluster geometry, active virtual sectors, and ordered factor
semantics. Combining those parsers would make the API look uniform while
weakening validation and obscuring which geometry is being compiled.

The next safe extraction target is therefore shared metadata/diagnostic
vocabulary, not a synthetic cross-representation term class. The executable
guard for this decision is [`test_operator_api_contract.py`](../../../tests/test_operator_api_contract.py).

## Construction-family map

| User goal | Use | Do not substitute silently |
| --- | --- | --- |
| Higher-order `exp(step * H)` on a chain | `MPOBasis.exp` or `exp_mpo` | An MPO cluster expansion; its order has a different meaning |
| Repeated MPO evaluations | `MPOBasis.compile_exp` | Caching value-dependent MPOs or autodiff graphs |
| Connected spatial PEPO residuals | `ClusterExpansionPlan` / `GraphClusterExpansionPlan` | Snake-chain MPO intervals when graph geometry matters |
| Differentiable fixed-channel PEPO | `PauliPEPOBasis` | Dense adaptive SVD fitting, which is not the same autodiff contract |
| Ordered `exp(A) @ exp(B) @ ...` PEPO | `PEPOClusterProductExpansion` | `exp(A + B + ...)` or independent full-lattice factor multiplication |
| Ordered local MPO clusters | `MPOClusterProductExpansion.from_factors` | Multiplying separately truncated full-lattice MPOs |
| Pauli sparse algebra/trace | `PauliMPO` | Treating it as a second unrelated exponential engine |
| PEPS/PEPO contraction corrections | `pepsy.bp` loop/cluster APIs | Operator cluster expansion; BP acts on contractions/environments |

## Example smoke set

The five small scripts in
[`examples/operators`](https://github.com/quantinuum-dev/pepsy/tree/develop/examples/operators)
are the canonical smoke examples:

1. [`higher_order_mpo.py`](../../../examples/operators/higher_order_mpo.py)
   — compile and materialize a higher-order MPO.
2. [`fixed_channel_pepo.py`](../../../examples/operators/fixed_channel_pepo.py)
   — evaluate sparse fixed-channel PEPO blocks and materialize explicitly.
3. [`dense_cluster_pepo.py`](../../../examples/operators/dense_cluster_pepo.py)
   — build a finite dense connected-cluster PEPO.
4. [`ordered_pepo_product.py`](../../../examples/operators/ordered_pepo_product.py)
   — build an ordered `exp(A) @ exp(B)` product through one joint topology.
5. [`ordered_mpo_product.py`](../../../examples/operators/ordered_mpo_product.py)
   — build the MPO analogue through the joint local-residual path.

They intentionally use small systems, public namespace imports, and no
notebook output. They are examples of API selection, not performance
benchmarks.

## Review rule for future additions

Before adding an operator symbol, answer four questions in the pull request
or design note:

1. Is it a new user concept, or an implementation detail of an existing
   basis/plan?
2. Which representation does it return: semantic, active, or materialized?
3. Which order does it control: history, cluster, or factor count?
4. Does it belong in `operators`, a more specific subpackage, or
   `experimental`?

If the answer is only “it makes an existing call shorter,” prefer a method or
documented compatibility alias over another top-level export.
