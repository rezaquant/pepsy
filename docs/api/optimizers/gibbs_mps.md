# GibbsMps

`GibbsMps` prepares a finite-temperature Gibbs state with purification and
ordinary MPS gate replay:

```python
from pepsy import bell_to_mps
from pepsy.optimizers import GibbsMps

terms = [
    (("ZZ", 1.0), (0, 1)),
    (("X", 0.25), 0),
]

gibbs = GibbsMps(terms, shape=8)
gibbs.prepare(beta=0.4, dt=0.01, chi=64, progress=True)

purification = gibbs.mps
rho = gibbs.to_mpo()                    # normalized, Tr(rho) = 1
rho_raw = gibbs.to_mpo(normalized=False)
Z = gibbs.partition_function()
log_Z = gibbs.log_partition_function()  # natural log, safe for large scales
```

The reusable identity purification is also available directly as
`pepsy.bell_to_mps(L, phys_dim=2, normalized=True, to_backend=...)`. It is a
Quimb `MatrixProductState`, so it can be inspected, converted with
`apply_to_arrays`, or passed to other MPS workflows before any thermal gates
are applied.

The internal MPS has `2 * L` sites in the order
`(physical_0, ancilla_0, physical_1, ancilla_1, ...)`. Physical site `i` is
therefore MPS site `2 * i`, and every Hamiltonian gate acts only on those even
sites. The initial state is a product of Bell pairs

\[
|I\rangle = \bigotimes_i |\Phi_i\rangle,
\qquad
|\Psi_\beta\rangle =
(e^{-\beta H/2} \otimes I)|I\rangle.
\]

Tracing the odd ancilla sites produces a positive operator proportional to
`exp(-beta * H)`. `to_mpo(normalized=True)` divides by its represented trace;
`partition_function()` returns the physical `Tr(exp(-beta * H))`, including the
`d**L` factor when normalized Bell pairs are used. For large or low-temperature
systems, use `log_partition_function()` to keep the partition-function scale in
log-space rather than exponentiating it. Internally, ancilla tracing uses
Quimb's native partial-trace reducer, which preserves Pepsy's MPS exponent in
MPO metadata. If `contract_opts` are supplied, GibbsMps uses equivalent native
tag-wise `strip_exponent=True` contractions because those options are not part
of Quimb's public partial-trace signature. Neither route densifies the
purification.

## Terms and lattice layouts

Terms use the same forms as `MPOBasis.from_terms`, including direct 1D terms:

```python
terms = [
    (("ZZ", J), (i, i + 1)) for i in range(L - 1)
]
terms += [(("X", h), i) for i in range(L)]
```

Regular lattice terms can keep their natural coordinates. If `shape` is
omitted, `GibbsMps` infers the location dimension and the smallest enclosing
2D/3D shape from the term locations, then creates a `OneDMap` using
`map_mode` (defaulting to `"snake"`):

```python
import quimb.tensor as qtn

edges = qtn.edges_2d_square(Lx, Ly, cyclic=True)
sites = sorted({site for edge in edges for site in edge})
terms = [(("zz", J), (u, v)) for (u, v) in edges]
terms += [(("x", h), site) for site in sites]

gibbs = GibbsMps(terms, map_mode="snake")  # infers shape=(Lx, Ly)
# Other supported traversals can be selected, for example:
# gibbs = GibbsMps(terms, map_mode="row-major")
```

The same inference distinguishes flat integer locations as a 1D chain and
coordinate tuples such as `(x, y)` or `(x, y, z)` as lattice sites. All terms
in one Hamiltonian must use the same location dimension. Supply `shape`
explicitly when the terms do not mention every site or when the intended
lattice extends beyond the largest coordinate.

Alternatively, regular lattice terms can use explicit `shape` with
`map_mode`, or an explicit `OneDMap`:

```python
gibbs = GibbsMps(
    terms,
    shape=(Lx, Ly),
    map_mode="snake",
)
```

The first implementation accepts one-site and two-site terms, including
long-range two-site couplings. General operators acting on more than two
sites and explicit string operators across a gap are intentionally rejected;
they need a multi-site gate application route that is not yet part of this
API.

## Imaginary-time stepping and compression

`prepare(beta, n_steps=N)` applies `N` Trotter steps to imaginary time
`beta / 2`. The default `trotter_order=2` uses Quimb's
`LocalHamGen.get_trotter_gates`, which automatically groups non-overlapping
Hamiltonian edges into commuting layers and emits the symmetric palindromic
schedule. `trotter_order=1` and `trotter_order=4` are also available. If `dt`
is supplied instead, `N` is chosen by ceiling so the actual step does not
exceed the requested value. With neither argument, one Trotter step is used.

For example, the graph ordering and fusion controls can be made explicit:

```python
gibbs.prepare(
    beta=0.4,
    n_steps=8,
    trotter_order=2,
    trotter_ordering="sort",       # or None, "random", or edge layers
    trotter_fuse_adjacent=True,
    trotter_alternate=True,
)
```

The Hamiltonian terms are first combined by logical edge. One-site terms on
connected sites are lifted into the incident edges with an equal partition, so
the represented Hamiltonian is unchanged. A one-site term on a site with no
incident edge is kept as an exact one-site exponential; no artificial graph
edge is introduced. The generated Quimb metadata is available as
`gibbs.trotter_gates` (`frac`, `layer`, `step`, and logical `where`) and
`gibbs.trotter_layers`. The executable `gibbs.gates` stream contains the same
gates with logical sites mapped to the even physical positions of the
purification. For Hamiltonians with disconnected one-site terms, the
executable stream additionally contains those exact one-site gates, which do
not have a Quimb edge-layer record.

The graph-layer ordering is resolved once per preparation and passed explicitly
to Quimb's scheduler. Stable automatic orderings are cached by the Gibbs
object, while `trotter_ordering="random"` creates a fresh ordering for each
preparation. Thus `trotter_layers` describes the exact schedule that is
replayed, without a second graph-coloring pass.

The default replay mode is `mode="direct"`, which is appropriate for the
non-unitary gates and the interleaved physical/ancilla layout. Other ordinary
open-boundary `MpsOptimizer` compression modes can be selected. `GibbsMps`
always sets `non_unitary=True`; unitary stabilization and unitary overlap
diagnostics are not used. `to_backend` is forwarded through term compilation,
Bell-pair construction, generated exponentials, MPS replay, and ancilla
tracing.

The `mode="mpo"` spelling remains accepted as a compatibility alias for the
same Quimb direct compressor. `mode="dmrg"`, `mode="svd"`, and other ordinary
open-boundary MPS modes can be selected when a different compression path is
needed.

Common compression controls can be passed directly:

```python
gibbs.prepare(
    beta=0.4,
    n_steps=8,
    mode="dmrg",             # opt-in FIT; default is "direct"
    chi=128,
    contraction_opt="auto",
    n_iter=4,
    normalize_every=True,
    normalize_final=True,
)
```

The remaining `MpsOptimizer` constructor options and run options are available
through `optimizer_kwargs={...}` and `run_kwargs={...}`. Normalization is
non-unitary scale bookkeeping: the optimizer stores removed scale in its
exponent, so `to_mpo(normalized=False)` still represents the same raw operator.

The returned `GibbsMps` object stores the live optimizer in `gibbs.optimizer`,
the prepared gate stream in `gibbs.gates`, and the requested temperature in
`gibbs.beta`. Calling `prepare` again starts from fresh Bell pairs.
