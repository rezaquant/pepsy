# How-To: Choose Parameters

This page gives pragmatic defaults for stable runs.

## Good starting point

```python
res = pepsy.ContractBoundary(
    norm=norm,
    mps_boundaries=bdy.mps_b,
    direction="y",
    n_iter=2,
    max_separation=0,
    fidel_=True,
)
```

With:

- `chi = 32` for quick prototyping
- `chi = 64` for moderate accuracy
- `chi >= 128` for harder instances (costly)

## When to raise `chi`

Raise `chi` if:

- `res.fidel` values drop strongly below your acceptable range
- `cost` is unstable across reruns with nearby settings

## When to raise `n_iter`

Raise `n_iter` if:

- per-step fidelity improves over iterations
- runtime budget allows more local optimization sweeps

## Direction choice

Try both:

- `direction="y"`
- `direction="x"`

Keep the one with better fidelity profile and runtime for your lattice shape.
