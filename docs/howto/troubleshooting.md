# How-To: Troubleshooting

## `ValueError: Input network must contain X* and Y* lattice tags`

Cause:

- `prepare_boundary_inputs` expects lattice tags for shape inference.

Fix:

- Ensure the network is a tagged PEPS-like object with `X{i}` and `Y{j}` tags.

## `Provided bra must have internal index names disjoint from ket`

Cause:

- You passed a custom `bra` whose internal indices collide with `ket`.

Fix:

- Reindex bra internal indices before calling `prepare_boundary_inputs`.

## Slow runtime

Common causes:

- large `chi`
- high `n_iter`
- difficult contraction geometry

Practical actions:

1. Reduce `chi` and `n_iter` first to baseline runtime.
2. Use `fidel_=True` to see where quality drops.
3. Increase only the parameter that improves that bottleneck.

## Fidelity list is empty

Cause:

- `fidel_=False` during `ContractBoundary` call.

Fix:

- Set `fidel_=True`.
