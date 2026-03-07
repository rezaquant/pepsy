# Quickstart

```python
import pepsy
import quimb.tensor as qtn

ket = qtn.PEPS.rand(Lx=3, Ly=3, bond_dim=2, seed=1, dtype="complex128")
ket_tagged, norm = pepsy.prepare_boundary_inputs(ket=ket)

bdy = pepsy.BdyMPS(
    tn_flat=ket_tagged,
    tn_double=norm,
    chi=32,
    single_layer=False,
)

res = pepsy.ContractBoundary(
    norm=norm,
    mps_boundaries=bdy.mps_b,
    direction="y",
    n_iter=2,
    fidel_=True,
)

print("cost:", res.cost)
print("fidelity history:", res.fidel)
```

```{note}
`ContractBoundary` returns `BoundaryContractResult` directly.
Use `res.cost` and `res.fidel` for outputs and diagnostics.
```
