# Tutorial: Fidelity Diagnostics

`ContractBoundary` returns:

- `res.cost`: final contracted scalar
- `res.fidel`: per-step fidelity values collected during sweeps

Use `res.fidel` to understand where approximation error accumulates.

## Basic interpretation

If values are close to `1.0`, local fit quality is strong at those steps.
Lower values identify harder regions.

## Split left/right products

For `direction="y"` and `max_separation=0`, a common split is `Ly // 2`:

```python
import numpy as np

split = ket.Ly // 2
f_left = np.prod(res.fidel[:split]) if split > 0 else 1.0
f_right = np.prod(res.fidel[split:]) if split < len(res.fidel) else 1.0

print("left product:", f_left)
print("right product:", f_right)
```

For `direction="x"`, use `split = ket.Lx // 2`.

## What to change if fidelity drops

1. Increase `chi`.
2. Increase `n_iter`.
3. Try `dmrg_run="global"` for better local solves.
4. Compare `direction="y"` vs `"x"` and choose the stabler one.

## Pitfall

A near-`1.0` fidelity at first step and lower values later is common; it does not
necessarily indicate a one-side bug. Later steps usually carry larger effective
environments and truncation pressure.
