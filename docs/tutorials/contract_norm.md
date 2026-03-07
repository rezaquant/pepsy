# Tutorial: Contract a PEPS Norm

This tutorial covers a full `prepare -> boundary init -> sweep contract` pipeline.

## Workflow overview

```text
PEPS ket
  -> prepare_boundary_inputs(ket, bra?)
      -> tagged ket + double-layer norm TN
  -> BdyMPS(...)
      -> boundary dictionary mps_b
  -> ContractBoundary(norm, mps_b, ...)
      -> BoundaryContractResult(cost, fidel, ...)
```

## Step 1: prepare inputs

```python
import pepsy
import quimb.tensor as qtn

ket = qtn.PEPS.rand(Lx=4, Ly=4, bond_dim=2, seed=7, dtype="complex128")
ket_tagged, norm = pepsy.prepare_boundary_inputs(ket=ket)
```

## Step 2: initialize boundaries

```python
bdy = pepsy.BdyMPS(
    tn_flat=ket_tagged,
    tn_double=norm,
    chi=64,
    single_layer=False,
)
```

## Step 3: contract

```python
res = pepsy.ContractBoundary(
    norm=norm,
    mps_boundaries=bdy.mps_b,
    direction="y",
    n_iter=2,
    max_separation=0,
    fidel_=True,
)

print("cost:", res.cost)
print("fidel entries:", len(res.fidel))
```

## Notes on parameters

- `chi`: higher means potentially better accuracy, higher runtime/memory.
- `n_iter`: more local fit sweeps per boundary update.
- `direction`: `y`, `y_left`, `y_right`, `x`, `x_left`, `x_right`.
- `max_separation`: currently `0` or `1`.

## Next steps

- See [fidelity diagnostics](fidelity_diagnostics.md) to interpret `res.fidel`.
- See [how-to tuning](../howto/choose_parameters.md) for practical defaults.
