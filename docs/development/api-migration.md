# API migration guide

Pepsy is keeping the current public names during the 0.x compatibility
window. The entries below are deprecated compatibility paths, not removals.
New code and documentation should use the canonical import on the right.

## Deprecated aliases

| Deprecated import | Canonical import |
| --- | --- |
| `pepsy.tensors.backend_cupy` | `pepsy.backends.backend_cupy` |
| `pepsy.tensors.backend_jax` | `pepsy.backends.backend_jax` |
| `pepsy.tensors.backend_numpy` | `pepsy.backends.backend_numpy` |
| `pepsy.tensors.backend_torch` | `pepsy.backends.backend_torch` |
| `pepsy.tensors.build_backend` | `pepsy.backends.build_backend` |
| `pepsy.tensors.get_default_array_backend` | `pepsy.backends.get_default_array_backend` |
| `pepsy.tensors.get_default_grad_backend` | `pepsy.backends.get_default_grad_backend` |
| `pepsy.tensors.get_torch_linalg_config` | `pepsy.backends.get_torch_linalg_config` |
| `pepsy.tensors.register_jax_linalg` | `pepsy.backends.register_jax_linalg` |
| `pepsy.tensors.register_torch_linalg` | `pepsy.backends.register_torch_linalg` |
| `pepsy.tensors.reset_default_backends` | `pepsy.backends.reset_default_backends` |
| `pepsy.tensors.reset_linalg_registrations` | `pepsy.backends.reset_linalg_registrations` |
| `pepsy.tensors.set_default_array_backend` | `pepsy.backends.set_default_array_backend` |
| `pepsy.tensors.set_default_grad_backend` | `pepsy.backends.set_default_grad_backend` |
| `pepsy.tensors.TorchLinalgConfig` | `pepsy.backends.TorchLinalgConfig` |
| `pepsy.tensors.build_contraction` | `pepsy.tensors.build_optimizer` |
| `pepsy.tensors.SpinfulFermionHubbard` | `pepsy.tensors.SpinfulFermion` |
| `pepsy.tensors.hrps_to_mps` | `pepsy.tensors.hrs_to_mps` |
| `pepsy.tensors.hrps_to_peps` | `pepsy.tensors.hrs_to_peps` |
| `pepsy.tensors.hrps_to_ttn` | `pepsy.tensors.hrs_to_ttn` |
| `pepsy.boundary.normalize` | `pepsy.boundary.peps_normalize` |
| `pepsy.boundary.infidelity` | `pepsy.boundary.peps_infidelity` |
| `pepsy.optimizers.QMeraParametricEnergyOptimizer` | `pepsy.optimizers.QMeraEnergyOptimizer` |
| `pepsy.optimizers.MpsStabOptimizer` | `pepsy.optimizers.StabilizerMpsSimulator` |
| `pepsy.optimizers.stabilizer_tn.StabilizerMps` | `pepsy.optimizers.stabilizer_tn.MpsStabOptimizer` |
| `pepsy.experimental.mera` | `pepsy.experimental.qmera` |
| `pepsy.optimizers.mera` | `pepsy.optimizers.qmera` |
| `ham_tn.build_mpo(...)` | `ham_tn.to_mpo(...)` |
| `ham_tn.build_pepo(...)` | `ham_tn.to_pepo(...)` |

These aliases emit `DeprecationWarning` when resolved. Applications can make
the transition visible in CI with:

```bash
python -W error::DeprecationWarning -m pytest
```

## Removal policy

No alias is scheduled for removal from the current 0.x line. Before a planned
breaking release, maintainers should review warning usage, publish release
notes with the table above, and remove only aliases that have completed the
deprecation window. The root-level compatibility facade is governed by
[`api-manifest.txt`](api-manifest.txt) and remains unchanged until that
review.

## Tree operator conversion

The canonical model-facing conversion methods are `ham_tn.to_mpo(...)`,
`ham_tn.to_pepo(...)`, `ham_tn.to_tree_mpo(...)`, and
`ham_tn.to_tree_pepo(...)`. The first two return Quimb chain/lattice operator
networks; the latter two factor directly over a supplied `TreePlan` or
`TreePepsPlan` without creating a chain MPO. `to_treempo` and
`to_treepepsmpo` are short compatibility aliases for the native tree methods.

`ham_tn.build_mpo(...)` and `ham_tn.build_pepo(...)` remain available as
deprecated wrappers and emit `DeprecationWarning`. The lower-level native
routes `TreePlan.build_tree_operator(...)` and
`Fermion.build_tree_operator(...)` remain available when a model object is
already in hand.
