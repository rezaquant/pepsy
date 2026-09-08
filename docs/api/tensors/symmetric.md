# Symmetric Tensor States

## Choosing sectors and charges

Physical sectors are charge maps: ``{charge: sector_size}``. For example,
``{0: 1, 1: 1}`` is a two-state U(1) or Z2 local space, while
``{0: 1, 1: 2, 2: 1}`` is the spinful Fermi-Hubbard local space with total
particle-number U(1). For spin-resolved particle-number sectors use
``model="fermi_hubbard_u1u1"``, whose local charges are ``(n_up, n_down)``.

```python
import pepsy as py

spinless = py.default_physical_sectors("U1", 2)
spinful = py.default_physical_sectors(model="fermi_hubbard")
spinful_spin_resolved = py.default_physical_sectors(model="fermi_hubbard_u1u1")

psi = py.SymMPS.random(
    6,
    symmetry="U1",
    phys_dim=spinless,
    bond_dim={0: 2, 1: 2},
    site_charge=py.site_charge_from_occupations([1, 0, 1, 0, 1, 0]),
)
```

The local ``site_charge`` pattern fixes the global sector represented by the
state. For U(1), ``psi.overall_charge()`` is the sum of local charges. For
``U1U1``, it is a pair such as ``(N_up, N_down)``. For Z2,
``psi.overall_parity()`` is the charge sum modulo two.

```python
even = py.site_charge_uniform(0)
checkerboard = py.site_charge_alternating(even=0, odd=1)
custom = py.site_charge_from_map({(0, 0): 1, (0, 1): 0}, default=0)

peps = py.SymPEPS.random(
    2,
    2,
    symmetry="Z2",
    phys_dim={0: 1, 1: 1},
    site_charge=custom,
)

peps.site_charges()
peps.overall_parity()
```

```python
half_filled_4x4 = py.site_charge_from_occupations(
    [(1, 0), (0, 1)] * 8,
)

fh = py.SymMPS.for_model(
    "fermi_hubbard_u1u1",
    16,
    bond_dim=4,
    site_charge=half_filled_4x4,
)

fh.overall_charge()  # (8, 8)
```

For a user-facing MPS constructor, ``ps_to_mps`` accepts the same ``Fermion``
model and returns the underlying native fermionic MPS directly; callers do not
need to instantiate ``SymMPS``:

```python
fh = py.Fermion(spinful=True, symmetry="U1U1")
psi = py.ps_to_mps(8, fermion=fh, seed=7)

# Override the product-state charge pattern when needed.
psi = py.ps_to_mps(
    8,
    fermion=fh,
    occupations=fh.half_filled_occupations(8),
)
```

``ps_to_mps`` always creates a bond-one, charge-fixed product MPS. For a
random charge-preserving MPS with a requested bond dimension, use
``hrs_to_mps``:

```python
psi = py.hrs_to_mps(
    8,
    fermion=fh,
    occupations=fh.half_filled_occupations(8),
    chi=4,
    seed=7,
)
```

The random-unitary growth fills only symmetry-compatible sectors and keeps the
same global charge. The historical alias ``hrps_to_mps`` points to the same
constructor. For direct Symmray block filling, use ``method="direct"``:

```python
psi = py.hrs_to_mps(
    8,
    fermion=fh,
    occupations=fh.half_filled_occupations(8),
    chi=4,
    method="direct",
    subsizes="maximal",
    seed=7,
)
```

The direct method uses Symmray's ``MPS_fermionic_rand`` constructor, then
right-canonicalizes and normalizes the MPS. It is useful when direct random
block filling is preferred; ``method="unitary"`` remains the default.

The matching PEPS constructor is ``ps_to_peps``. Pass ``(Lx, Ly)`` (or one
integer for a square lattice), a coordinate occupation mapping, and the same
``Fermion`` object:

```python
occupations = {
    (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
    for x in range(Lx)
    for y in range(Ly)
}
seed = py.ps_to_peps(
    (Lx, Ly),
    fermion=fh,
    occupations=occupations,
    seed=7,
)
```

``ps_to_peps`` always returns a bond-one, charge-fixed product PEPS. For a
random Symmray block-filled PEPS with a requested bond dimension, use
``hrs_to_peps``:

```python
psi = py.hrs_to_peps(
    (Lx, Ly),
    fermion=fh,
    occupations=occupations,
    chi=4,
    method="direct",
    subsizes="maximal",
    seed=7,
)
```

The direct method uses Symmray's ``PEPS_fermionic_rand`` constructor. It skips
global normalization by default, avoiding an expensive CPU boundary-MPS
contraction; this is safe for NetKet VMC because a global wavefunction scalar
cancels from sampling and local-energy ratios. Pass ``normalize=True`` only
when a globally normalized PEPS is explicitly required. A unitary PEPS-growth
method is not implemented yet. Both constructors return the underlying PEPS
with native fermionic Symmray tensors; use ``SymPEPS`` only when wrapper
methods or stored Hamiltonian metadata are needed.

For the corresponding full operator identity, use ``id_to_pepo`` with the
same model:

```python
identity = py.id_to_pepo(
    (Lx, Ly),
    fermion=fh,
    cyclic=True,
)
```

This returns a native graded PEPO containing every local charge sector and
repairs periodic bond orientations. ``occupations`` and ``site_charge`` are
intentionally rejected here: they select a product-state sector and would
make the result something other than the full identity.

## Unified native fermion helper

``Fermion`` is the model-facing helper for both one-mode spinless fermions
and four-state spinful Hubbard sites. It owns the local space, symmetry, and
optional backend conversion; physical couplings are passed explicitly when
constructing a term, gate, or stream. ``spinful`` selects the local space;
``symmetry`` selects the conserved charge group. Spinless helpers support
``U1`` and ``Z2``; spinful helpers support ``U1``, ``Z2``, ``U1U1``, and
``Z2Z2``.

```python
spinless = py.Fermion(spinful=False, symmetry="U1")

spinful = py.Fermion(spinful=True, symmetry="U1U1")

spinless.operator("number")
spinful.operator("n_up")
spinful.hopping_operator(spin="up")
spinful.interaction_operator()
spinful.chemical_potential_operator()
spinful.onsite_gate(dt=0.01, site=0, U=8.0)
spinful.gate("interaction", dt=0.01, U=8.0)
spinless.gate_stream(edges, dt=0.01, order=2, t=1.0, V=0.5, mu=0.0)
spinful.local_terms(edges, t=1.0, U=8.0)  # native terms for optimization
spinful.strang_gate_stream(
    edges, dt=0.01, t=1.0, U=8.0, field_z=0.2
)
pairing = py.Fermion(spinful=False, symmetry="Z2")
pairing.strang_gate_stream(edges, dt=0.01, t=1.0, pairing=0.2)
```

The gate-stream helpers accept `order=1`, `2`, or `4`. Order four is an
opt-in Suzuki-Yoshida composition of three symmetric second-order steps and
includes a negative middle substep; the existing order-one and order-two
streams are unchanged. For example, use
`spinless.gate_stream(edges, dt=0.01, order=4, t=1.0, V=0.5)` when the
additional gate applications are worthwhile.

The bare ``*_operator`` methods return explicit native fermionic operators,
not exponentiated gates or model coefficients. For example,
``spinful.hopping_operator(spin="up")`` returns
``c_a^dagger c_b + c_b^dagger c_a`` for canonical two-site locations,
``spinful.interaction_operator()`` returns ``n_up n_down``,
``spinful.chemical_potential_operator()`` returns ``n_up + n_down``, and
``spinful.density_operator()`` returns the bare nearest-neighbor density
product. Use ``hopping_term(edge, t=...)``, ``interaction_term(site, U=...)``,
``chemical_potential_term(site, mu=...)``, ``onsite_term(site, ...)``, or
``density_term(edge, V=...)`` when a location-aware weighted term is desired.

For arbitrary sums of local fermion monomials, use ``operator_term``. The
site order is the order of the returned operator and should match the ``where``
argument used for measurement:

```python
hopping_up = spinful.operator_term([
    (-t, ((i, "create_up"), (j, "annihilate_up"))),
    (-t, ((j, "create_up"), (i, "annihilate_up"))),
], sites=(i, j))

double_occupancy = spinful.operator_term(
    [(U, ((i, "number_up"), (i, "number_down")))],
    sites=(i,),
)

# Add the correctly ordered fermionic Hermitian conjugate automatically.
hopping_up = spinful.operator_term(
    [(1.0, ((0, "create_up"), (1, "annihilate_up")))],
    sites=(0, 1),
    add_hc=True,
)

# Built-in convenience for Delta_0^dagger Delta_1 + h.c.
eta_pair = spinful.eta_pair_operator()

# A complete explicit Hamiltonian can be wrapped for MPO/DMRG conversion.
terms = {
    edge: -t * spinful.hopping_operator()
    for edge in edges
}
terms.update({site: U * spinful.interaction_operator() for site in sites})
terms.update({site: -mu * spinful.chemical_potential_operator() for site in sites})
ham_dmrg = spinful.hamiltonian(terms)
H_mpo = ham_dmrg.to_mpo(L=L)
```

For the all-pairs staggered eta structure factor, use the dedicated compact
native builder instead of expanding one two-site term for every pair:

```python
eta_mpo = spinful.eta_pair_structure_factor_mpo(
    L,
    signs=tuple((-1) ** (site % 2) for site in range(L)),
    normalization=1.0 / L,
)
```

This represents ``(1 / L) * (F^dagger F - sum_i Delta_i^dagger Delta_i)``
with ``F = sum_i signs[i] * Delta_i``. It has four finite-state MPO channels
independent of the number of pairs and remains a native graded MPO for the
MPS evaluator. ``compress=True`` is optional; it is not needed to obtain the
compact bond dimension.

For a term mapping, scalar site keys are normalized to ``(site,)``. The
``SymHamiltonian`` returned by ``fermion.hamiltonian(terms)`` retains the
locations, symmetry, and fermionic ordering metadata needed by ``to_mpo``;
passing the raw dictionary directly to ``to_mpo`` would lose that context.

For a PEPS mapping, use coordinate edges together with flat coordinate site
keys. The nested edge keys disambiguate a coordinate such as ``(x, y)`` from
an MPS edge ``(i, j)``, and the coordinate keys remain available to PEPS
expectation routines:

```python
edges = tuple(peps.edges)  # ((x0, y0), (x1, y1))
sites = tuple(peps.sites)
terms = {edge: -t * fermion.hopping_operator() for edge in edges}
terms |= {site: fermion.onsite_term(site, U=U, mu=mu) for site in sites}
ham_peps = fermion.hamiltonian(terms)
```

Available local names include ``create_up``, ``annihilate_up``,
``number_up``, ``create_down``, ``annihilate_down``, ``number_down``,
``double``, ``pair_create``, ``pair_annihilate``, ``number``, and ``sz``
for spinful fermions; spinless helpers provide ``create``, ``annihilate``,
``number``, and ``parity``. All returned operators retain the selected
Symmray Abelian symmetry and fermionic grading.

The site-layout ``local_terms(edges, t=..., U=..., mu=...)`` mapping is keyed
by edges. Its onsite
Hubbard and chemical-potential pieces are divided by each site's coordination
inside the incident edge tensors, so summing the mapping still includes each
one-site contribution exactly once. If those pieces should be visibly
separate, build the edge terms with ``U=0`` and ``mu=0`` and add one-site
terms such as ``U * spinful.observable("double")`` under keys ``(site,)``.

``onsite_gate`` is the complete local Hubbard term, ``U n_up n_down - mu n``
for spinful sites and ``-mu n`` for spinless sites. ``density_gate`` supplies
the optional nearest-neighbor ``V n_i n_j`` term (total density for spinful
sites). The same ``U``, ``V``, and ``mu`` parameters are included in
``hamiltonian`` and in the returned gate stream; ``stream.hamiltonian`` gives
the corresponding native term container.

For a finite PEPS, request a boundary-controlled energy estimate with the
same measurement options used for local observables:

```python
energy = peps.energy(hamiltonian, chi=64)
energy = peps.energy(hamiltonian, measure_kwargs={"mode": "mps"})
```

Without ``chi`` or measurement options, ``SymPEPS.energy`` retains its exact
doubled-network contraction path. ``energy_density`` accepts the same options.

For qMERA, keep the same ``Fermion`` parameters but request the explicit
two-state mode layout. This returns qMERA ``LocalTerm`` objects rather than
four-state site tensors, matching ``QMeraGeometry(site_modes=("up", "down"))``:

```python
from pepsy.optimizers.qmera import QMeraGeometry

geometry = QMeraGeometry(shape=3, site_modes=("up", "down"))
qmera_terms = spinful.local_terms(geometry, layout="qmera")
# Equivalent shorthand:
qmera_terms = spinful.qmera_terms(geometry)
```

The spinful interaction gate is the exact native Symmray operator
``exp(-i dt U n_up n_down)``. Its local diagonal is
``(1, 1, 1, exp(-i dt U))`` in Symmray's ordered Hubbard basis. Pass
``imaginary=True`` to use the corresponding ``exp(-dt U n_up n_down)`` gate.

``param_gate(name, params)`` and the module-level fermion gate generators
(``fermion_interaction_param_gen``, ``fermion_density_param_gen``, and
``fermion_hopping_param_gen``) follow the Quimb parameter-generator
convention: ``params[0]`` is kept on its original Autoray backend, so Torch
and JAX gradients are not routed through NumPy.

For a custom neutral local Hamiltonian, ``exponential`` accepts terms such as
``(coefficient, ((site, "create_up"), (site, "annihilate_down")))`` and builds
the native Symmray gate directly. ``imaginary=True`` changes the evolution to
``exp(-dt H)``. The local exponential must be neutral so it remains in one
conserved charge sector.

``SpinfulFermion`` and ``SpinfulFermionHubbard`` remain compatibility
constructors that fix ``spinful=True``. ``SymmFermions.spinless(...)`` and
``SymmFermions.spinful(...)`` are factory-style alternatives with the same
local-space guarantees.

Automatic streams and edge-built Hamiltonians also accept ``field_x``,
``field_y``, and ``field_z``. Transverse fields mix up/down occupation and
therefore require spinful total-``U1`` or ``Z2`` symmetry; a longitudinal
``field_z`` also works with spin-resolved ``U1U1``/``Z2Z2``. The ``pairing``
and ``pairing_phase`` options describe a parity-preserving spinless pairing
term and require ``Fermion(spinful=False, symmetry="Z2")``.

## Native spinful-fermion helper

``SpinfulFermion`` bundles the local spinful operators, half-filled product
charge pattern, native Symmray observables, and gate construction for the
total-number ``U1``/``Z2`` or spin-resolved ``U1U1``/``Z2Z2`` formulation. It keeps the
simulation in the direct fermionic representation -- it does not construct a
qubit or Jordan-Wigner circuit. It is not limited to Hamiltonian construction:
the same object supplies local operators, charged observables, and evolution
gates.

```python
fermions = py.SpinfulFermion(
    symmetry="U1U1",  # use "U1" to conserve only total particle number
)

site_charge = fermions.half_filled_site_charge(L=16)
number_up = fermions.observable("number_up")
pair_create = fermions.observable("pair_create")
hamiltonian = fermions.hamiltonian(edges, t=1.0, U=8.0)

# A symmetric, edge-coloured second-order step.  Each colour contains
# vertex-disjoint hopping bonds; forward then reverse colours make the hopping
# product formula second order as well.
gates = fermions.strang_gate_stream(
    edges, dt=0.01, sites=range(16), t=1.0, U=8.0
)
```

``py.SymmFermions`` is the companion namespace for future symmetric-fermion
helpers. ``py.SymmFermions.spinful(...)`` returns the same ``SpinfulFermion``
object when a workflow prefers a model-family entry point.

``pair_create`` and ``pair_annihilate`` carry charge ``+2`` and ``-2`` for
``U1``, ``0`` for ``Z2``, and ``(1, 1)`` for ``U1U1``/``Z2Z2``. Pass those
charges to ``SymMPS.measure`` when evaluating a charged pairing correlator.

For explicit rectangular PEPS workflows, ``Fermion.lattice_half_filling``
prepares only the lattice and charge metadata. It does not build a PEPS,
Hamiltonian, or gate stream, so those steps remain visible in the calling
notebook:

```python
setup = fermion.lattice_half_filling(
    Lx,
    Ly,
    pattern="checkerboard",
    cyclic=True,  # periodic physical Hamiltonian edges
)

sites = setup.sites
edges = setup.edges
occupations = setup.occupations

terms = {edge: -t * fermion.hopping_operator() for edge in edges}
terms |= {site: fermion.onsite_term(site, U=U, mu=mu) for site in sites}
ham = fermion.hamiltonian(terms)

gate_stream = fermion.strang_gate_stream(
    edges,
    dt=0.01,
    sites=sites,
    imaginary=True,
    t=t,
    U=U,
    mu=mu,
)
```

The ``cyclic`` option controls only the physical edge list in ``setup``. Native
fermionic ``ps_to_peps(..., cyclic=True)`` also supports periodic Symmray PEPS
bonds. An OBC MPS or PEPS can still represent a PBC Hamiltonian by using
``cyclic=True`` for the physical edge list and ``cyclic=False`` for the state;
nonlocal wrap-around gates are then routed through the open state network.

For ``U1U1``, ``setup.occupations`` contains ``(n_up, n_down)`` pairs. For
spinful ``U1`` and ``Z2``, it contains scalar total occupations while
``setup.spin_occupations`` retains the up/down checkerboard pattern for
initial walker configurations.

## Random-unitary MPS starts

For DMRG-style random starts, prefer ``SymMPS.random_unitary_evolution`` or the
model-aware ``SymMPS.random_unitary_for_model`` over raw block-filled
``SymMPS.random``. The unitary path starts from the requested product charge
sector, applies charge-preserving two-site random unitary layers, splits back
to the requested bond dimension, canonicalizes, and normalizes. This gives a
well-conditioned random MPS in the same spirit as TeNPy's random-unitary MPS
initialization.

```python
half_filled_6x6 = py.site_charge_from_occupations(
    [(1, 0), (0, 1)] * 18,
)

dmrg_init = py.SymMPS.random_unitary_for_model(
    "fermi_hubbard_u1u1",
    36,
    bond_dim=32,
    site_charge=half_filled_6x6,
    seed=7,
    dtype="complex128",
)

dmrg_init.overall_charge()  # (18, 18)
```

``SymMPS.random`` remains available when a workflow intentionally wants the
lower-level raw random block fill, for example as an initialization stress
test.

For direct fermionic Fermi-Hubbard examples, use Gao et al., "Fermionic tensor
network contraction for arbitrary geometries", Phys. Rev. Research 7, 023193
(2025), https://doi.org/10.1103/PhysRevResearch.7.023193 as the primary
methods reference. Pepsy should keep fermionic parity, additional Abelian
symmetries such as ``U1`` or ``U1U1``, and leg-order metadata in Symmray-backed
arrays, while relying on quimb/cotengra graph optimizers for contraction
ordering.

Use ``state.fermionic_ordering()`` when a workflow needs the package-level
record of the graph and local order data carried by a symmetric state:

```python
ordering = fh.fermionic_ordering()

ordering["enabled"]       # True for direct fermionic states
ordering["site_order"]    # the site labels in tensor-network order
ordering["edge_order"]    # the stored graph edge order
ordering["edges"][0]["index_directions"]
```

The same record is available as ``summary["fermionic_ordering"]`` from
``symmray_mps_summary`` and ``symmray_peps_summary``.

## Backend conversion

Pass ``to_backend=`` to build Symmray blocks directly on a chosen array
backend. The callable is applied to each stored dense block, preserving the
Symmray charge maps and sparse block structure.

```python
import torch

to_backend = py.backend_torch(dtype=torch.complex128)
peps_site_charge = py.site_charge_from_occupations(
    {
        (i, j): (1, 0) if (i + j) % 2 == 0 else (0, 1)
        for i in range(3)
        for j in range(3)
    }
)

peps = py.SymPEPS.random(
    3,
    3,
    symmetry="U1U1",
    fermionic=True,
    phys_dim=py.default_physical_sectors(model="fermi_hubbard_u1u1"),
    site_charge=peps_site_charge,
    bond_dim=4,
    to_backend=to_backend,
)

ham = py.SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    peps.edges,
    t=1.0,
    U=8.0,
    mu=0.0,
    to_backend=to_backend,
)

# Existing wrappers can be converted too.
peps_torch = peps.to_backend(to_backend, inplace=False)
ham_torch = ham.to_backend(to_backend, inplace=False)
```

## Symmetric MPO mapping

Symmetric Hamiltonians can be flattened into an MPS-chain MPO with
``SymHamiltonian.to_mpo(...)``. Coordinate edges from a 2D or 3D lattice can be
mapped with an explicit ``OneDMap`` path. The mapping is required for
coordinate edges because it defines where nonlocal chain channels, including
fermionic parity strings, sit.

Current support is:

- generic charge-neutral rank-4 two-site terms, including ``tfim``/``Z2`` and
  ``heisenberg``/``U1``;
- spinless Fermi-Hubbard ``model="fermi_hubbard_spinless"`` with ``U1`` or
  ``Z2`` symmetry, hopping, density interaction, and chemical-potential terms
  (pairing ``delta != 0`` is not implemented yet);
- spinful Fermi-Hubbard ``model="fermi_hubbard_u1u1"`` with
  ``symmetry="U1U1"``, hopping, onsite interaction, nearest-neighbor density
  interaction, and chemical-potential terms.
- native graded MPO conversion for arbitrary homogeneous-charge one- or multi-site
  ``FermionicArray`` terms, including non-contiguous support;

Spinful total-particle-number ``model="fermi_hubbard"`` with ``symmetry="U1"``
still raises ``NotImplementedError``; use ``model="fermi_hubbard_u1u1"`` when
an MPO is required for spinful Fermi-Hubbard.

For spinful ``U1U1`` Fermi-Hubbard, the local physical space has four spinful
Hubbard states with charges ``(n_up, n_down)``. Onsite terms use
``U * n_up * n_down - mu_up * n_up - mu_down * n_down``. ``mu`` may be a scalar
or ``(mu_up, mu_down)``. Hopping terms use
``-t_sigma c^dagger_i_sigma c_j_sigma`` plus the reverse direction, and ``t``
may be a scalar or ``(t_up, t_down)``. Nearest-neighbor density terms use
``V * (n_up + n_down)_i * (n_up + n_down)_j`` on each supplied edge; ``V`` may
be a scalar, edge mapping, or edge callable.

For fermionic models, non-adjacent mapped hopping edges insert the dense
fermionic parity operator on every intermediate chain site, so a 2D
nearest-neighbor edge can become a nonlocal MPS-chain term without dropping
fermionic signs.

- The returned object is a ``quimb.tensor.MatrixProductOperator`` whose tensor
  data are Symmray block-sparse arrays. Its physical index families default to
  ``k{}`` and ``b{}``, matching Pepsy's MPS/MPO conventions.
- ``to_mpo`` assembles the symmetry-preserving MPO first. The default
  ``compress=True`` then calls quimb MPO compression; pass ``compress=False``
  when you want to inspect or compare the uncompressed assembled MPO exactly.

```python
mapper = py.OneDMap(Lx=4, Ly=4, mode="snake")
idx2coo, coo2idx = mapper.build()

ham = py.SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    square_lattice_edges,
    t=1.0,
    U=8.0,
    V=0.25,
    mu=0.0,
)

mpo = ham.to_mpo(mapper=mapper)

# Equivalent when a workflow already stores the maps explicitly:
mpo = ham.to_mpo(idx2coo=idx2coo, coo2idx=coo2idx)
```

For the common edge-built Fermi-Hubbard path, the model helper is equivalent
and keeps the selected symmetry attached to the construction:

```python
fermion = py.Fermion(spinful=True, symmetry="U1U1")
mpo = fermion.build_mpo(
    [(0, 1), (1, 2)], L=3, t=1.0, U=8.0, mu=0.0, max_bond=16
)
```

This returns the native graded MPO by default. For the explicit
Jordan-Wigner-compatible convention, pass ``fermionic=False``:

```python
mpo = fermion.build_mpo(
    [(0, 1), (1, 2)], L=3, t=1.0, U=8.0, mu=0.0, max_bond=16,
    fermionic=False,
)
```

Native fermionic gate streams from the same ``Fermion`` model can be passed to
``MpoOptimizer``; the optimizer preserves the graded Symmray tensors and their
charge blocks during replay and compression. ``fermionic=False`` remains the
explicit Jordan-Wigner compatibility choice for ``SymHamiltonian.to_mpo``.

For a coordinate-keyed native operator on a 2D lattice, ``Fermion.to_pepo``
provides the corresponding PEPO embedding:

```python
left, right = (0, 1), (2, 2)
fermion = py.Fermion(spinful=True, symmetry="U1U1")
term = fermion.operator_term(
    [(1.0, ((left, "create_up"), (right, "annihilate_up")))],
    sites=(left, right),
    add_hc=True,
)
pepo = fermion.to_pepo(
    {(left, right): term},
    Lx=3,
    Ly=3,
    fermionic=True,
    max_bond=16,
)
assert all(type(tensor.data).__name__.endswith("FermionicArray") for tensor in pepo)
```

``to_pepo`` preserves native Symmray grading and supports the spinful
``U1U1``, ``U1``, and ``Z2`` paths as well as spinless ``U1`` and ``Z2``.
Its current PEPO representation uses the selected snake-style MPO ordering
for fermionic channels; the added transverse lattice bonds are dimension one
unless periodic PEPO bonds are requested with ``cyclic=True``. Terms should
be homogeneous in charge for one MPO/PEPO. Native ``fermionic=True``
construction supports both neutral and nonzero charges by carrying the
operator charge at the open MPO/PEPO boundary. For a mixed-charge collection,
pass ``charge_sectors=True`` to receive ``{charge: PEPO}`` (or ``{charge:
MPO}`` from ``to_mpo``). For odd-parity terms, pass ``label=`` to
``operator_term`` so the native dummy-mode phase metadata is retained. The
Jordan--Wigner compatibility path remains neutral-only. For a one-site
coordinate key, use ``((x, y),)`` rather than ``(x, y)``.

The same native PEPO route is available from an existing Hamiltonian or as a
model-facing shorthand:

```python
pepo = hamiltonian.to_pepo(
    Lx=3,
    Ly=3,
    mapper=py.OneDMap(3, 3, mode="snake-row-major"),
    fermionic=True,
)
pepo = fermion.build_pepo(
    {(left, right): hopping},
    Lx=3,
    Ly=3,
    mapper=py.OneDMap(3, 3, mode="snake-row-major"),
    fermionic=True,
)
```

For Hamiltonian-builder workflows, ``ham_tn.build_pepo(..., fermion=fermion,
fermionic=True, mapper=...)`` is equivalent.

Pass ``to_backend=`` to ``Fermion.to_mpo``, ``Fermion.to_pepo``, or
``Fermion.build_mpo`` when the returned Symmray blocks should use a selected
array backend. Native MPO
assembly, replay, and exact energy measurement are supported. Native MPO
energy applies the operator sitewise as a factorized graded MPO-MPS
contraction, so its cost is controlled by the MPS and MPO bond dimensions.

``Fermion.build_mpo(...)`` is the canonical native construction and defaults
to graded Symmray tensors. Passing ``fermionic=False`` selects the explicit
Jordan--Wigner compatibility MPO path. ``Fermion.to_mpo(...)`` remains a
compatibility alias.

For periodic square lattices encoded as long-range edges in an OBC MPS/MPO,
``mode="folded-snake"`` alternates opposite columns before snaking. On a 6 by
6 torus this lowers the longest nearest-neighbor chain separation from 35
with the standard snake path to 12, while preserving the same total edge
separation. This is often a better starting ordering for fixed-``chi`` MPS
DMRG benchmarks with PBC lattice physics.

For a Hamiltonian whose edges are already integer MPS-chain sites, pass
``L=...`` instead of a coordinate mapper:

```python
ham = py.SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    [(0, 2)],
    t=(1.0, 0.8),
    U=4.0,
    mu=(0.1, 0.2),
)

mpo = ham.to_mpo(L=3, compress=True, max_bond=16, cutoff=1e-12)
```

### MPO compression diagnostics

Every MPO built by ``SymHamiltonian.to_mpo`` carries a construction record in
``mpo.pepsy_compression_report``. It records the raw and final maximum bond,
the requested cap, whether compression reduced the represented rank, and
whether Symmray returned a bond larger than the requested cap:

```python
report = mpo.pepsy_compression_report
print(report["raw_max_bond"], report["final_max_bond"])
if report["max_bond_exceeded"]:
    print("The requested max_bond was a soft cap for this compression.")
```

Pepsy emits a ``RuntimeWarning`` in the latter case. This can occur when a
positive cutoff leaves singular values tied at the selected threshold. The
report's ``rank_reduced`` flag says that the MPO bond rank changed; it is not
an error estimate. For a hard numerical cap, use ``cutoff=0.0`` and verify the
returned ``final_max_bond``. When the cap binds, also check convergence against
a larger cap because the compressed MPO can represent an approximation to the
original Hamiltonian.

``to_mpo`` also accepts ``to_backend=`` to map each stored Symmray block to an
array backend, and ``dtype=`` to choose the dense local operator dtype used
before conversion to block-sparse arrays. The implementation is validated by
checking that supported adjacent MPO energies agree with the local two-site
``SymHamiltonian`` energy path, that compressed long-range MPOs preserve the
uncompressed energy, and that ``OneDMap`` coordinate edges match equivalent
flat integer edges.

## Time evolution

Hamiltonians produce a canonical bundled gate stream, so the same stream can be
used by the PEPSY gate wrappers and MPS optimizer.

```python
psi = py.SymMPS.for_model("heisenberg", 8, bond_dim=4)
ham = psi.build_hamiltonian()
gates = ham.gate_stream(0.01)

psi.time_evolve_mps_optimizer(0.01, hamiltonian=ham, chi=16, mode="direct")
```

For direct spinful Fermi-Hubbard dynamics, Pepsy also exposes native fermionic
``U1U1`` stream builders. These keep the four-state local fermion space and do
not map the system to qubits or spins. The light-pulse helper follows the
paper-style real-time schedule with onsite interaction half-layers, Peierls
hopping layers, and optional field-off relaxation steps.

```python
peps_charge = py.site_charge_from_occupations(
    {
        (i, j): (1, 0) if (i + j) % 2 == 0 else (0, 1)
        for i in range(6)
        for j in range(6)
    }
)
peps = py.SymPEPS.for_model(
    "fermi_hubbard_u1u1",
    6,
    6,
    bond_dim=4,
    site_charge=peps_charge,
    dtype="complex128",
)

edges = peps.edges
sites = peps.sites

pulse = py.fermi_hubbard_u1u1_light_pulse_gate_stream(
    edges,
    sites=sites,
    t=1.0,
    U=8.0,
    omega=4 * np.pi / 3,
    pulse_steps=2,
    relaxation_steps=2,
)
```

For the **Jordan-Wigner spin picture** -- the bosonic representation used by
``SymHamiltonian.to_mpo(model="fermi_hubbard_u1u1")`` and ``SymDMRG2`` -- Pepsy
also exposes matching *bosonic* gate streams. The Jordan-Wigner parity string is
written explicitly into the two-site hopping operator, so these gates act on a
bosonic Jordan-Wigner MPS without fermionic swap phases. Hopping bonds must be
nearest-neighbour in the chain; use ``to_mpo`` for long-range Jordan-Wigner
strings.

```python
edges = [(i, i + 1) for i in range(7)]
jw_step = py.fermi_hubbard_u1u1_jw_gate_stream(
    edges,
    dt=0.05,
    t=1.0,
    U=8.0,
    order=2,
)
```

The Jordan-Wigner stream also accepts `order=4`. It keeps the same
nearest-neighbour restriction and composes the validated order-two
Jordan-Wigner gates, so no new parity-string convention is introduced.

For a **single, controlled Jordan-Wigner conversion**, build the gates and the
MPO from the *same* ``SymHamiltonian`` and ordering. ``jw_trotter_gates`` reads
the same terms, site ordering (``mapper``), local operators, and parity-string
convention as ``to_mpo``, so an energy from the MPO and a time evolution driven
by the gates agree by construction. Non-nearest-neighbour mapped bonds raise
(their string spans intervening sites); use ``to_mpo`` for those, and
``ham.jw_bond_layout(mapper=...)`` to see which bonds are nearest-neighbour
versus long-range under a given ordering.

```python
ham = py.SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1", "U1U1", [(i, i + 1) for i in range(7)], t=1.0, U=8.0
)
mpo = ham.to_mpo(L=8)                 # energy / SymDMRG2
gates = ham.jw_trotter_gates(0.05, order=2)   # consistent time-evolution gates
# gates4 = ham.jw_trotter_gates(0.05, order=4)  # optional higher order
```

``ham.jw_energy(state)`` reads the energy of a bosonic (``fermionic=False``)
``SymMPS`` back from the same conversion -- a sum of local Jordan-Wigner term
expectations via the state's symmetry-aware ``measure`` -- so imaginary-time
evolution driven by ``jw_trotter_gates`` converges to the ``SymDMRG2`` ground
energy.

**Implementation status (long-range / 2D).** The Jordan-Wigner *gate* path is
currently **nearest-neighbour only**. A long-range hop (every perpendicular bond
of a snaked 2D lattice) has a parity string spanning the sites between its
endpoints, so it is not a two-site gate; ``jw_trotter_gates`` and
``fermi_hubbard_u1u1_jw_*_gate_stream`` raise on such bonds, and
``jw_bond_layout`` reports them. The exact long-range gate has a validated
parity-sector sub-MPO form,

```text
exp(scale * H_ij) = (I_int / 2) x (G+ + G-)  +  (S_int / 2) x (G+ - G-),
```

with ``G+- = exp(+- scale * H_hop)`` the adjacent two-site gates and
``S_int = prod_k P_k`` the intervening parity string (constant bond dimension,
linear in span). Landing it on a symmetric MPS needs a Symmray-aware multi-site
MPO gate application (upstream ``gate_nonlocal`` / MPS-addition are not yet
Symmray-compatible), so it is deferred. For 2D today: order the lattice with
``OneDMap(..., mode="folded-snake")`` to maximize nearest-neighbour bonds, use
``to_mpo`` for the residual long-range terms.

When the circuit gate stream should choose the 1D path, use the MPS layout
finder as a mapping mode. This remains a coordinate/index operation only: it
does not allocate an MPS or perform replay, SVD, or truncation.

```python
mapper = py.OneDMap(
    Lx=6,
    Ly=6,
    mode="finder",
    gate_stream=gates,
    layout_kwargs={"objective": "compression", "order": "quality"},
)
idx2coo, coo2idx = mapper.build()
```

The finder assumes the stream uses compact logical labels in
`range(Lx * Ly)`. Set `finder_base_mode="row-major"` (or another regular
mode) when those labels come from a different initial traversal. A previously
computed MPS layout plan may be passed as `finder=plan`.

For a flattened MPS path, feed the same canonical bundled stream to
``MpsOptimizer``:

```python
mapper = py.OneDMap(Lx=6, Ly=6, mode="folded-snake")
idx2coo, coo2idx = mapper.build()
flat_edges = tuple((coo2idx[a], coo2idx[b]) for a, b in square_lattice_edges)

psi = py.SymMPS.for_model(
    "fermi_hubbard_u1u1",
    36,
    bond_dim=8,
    site_charge=py.site_charge_from_occupations([(1, 0), (0, 1)] * 18),
    dtype="complex128",
)

pulse = py.fermi_hubbard_u1u1_light_pulse_gate_stream(
    flat_edges,
    sites=range(36),
    t=1.0,
    U=8.0,
    relaxation_steps=2,
)

opt = py.MpsOptimizer(psi.tn, pulse, chi=64, mode="direct", inplace=True)
psi_t = opt.run(progbar=True, cutoff=1e-10)
```

For a PEPS path, apply the same stream through routed simple update:

```python
gauges = {}
peps.apply_gates(
    pulse,
    method="simple",
    gauges=gauges,
    max_bond=8,
    cutoff=1e-10,
)
```

For Symmray-backed MPS gate streams, ``mode="swap"``, ``mode="perm"``, and
``mode="svd"`` use
quimb's block-aware auto-swap split path for nonlocal 1D gate streams such as a
row-major square lattice. ``mode="direct"`` uses its usual sub-MPO compression for
nearest-neighbor gates and falls back to the same Symmray auto-swap path for
nonlocal gates, because the current quimb/Symmray sub-MPO path mixes in dense
helper tensors. ``mode="exact"`` is useful as a small-system reference.
``mode="fit"`` (the alias of ``mode="dmrg"``) defaults to
``fit_block_size=2`` and can grow visited bonds without dense padding: the
effective two-site tensor is split by Symmray's native block SVD, preserving
U1/U1xU1 charges, dual legs, fermionic dummy modes, and graded phases.
``fit_block_size=3`` is also available for a three-site effective tensor and
performs two native block SVD splits with the same metadata preservation.
`fit_target_strategy="auto"` deliberately selects the native routed-MPS target
for these arrays; the lazy layered target optimization is currently limited to
ordinary NumPy/Torch/CuPy tensors because graded gate-layer tagging needs a
separate phase-contract validation. Explicit `fit_target_strategy="layered"`
therefore fails early instead of densifying or guessing fermionic metadata.
Fixed-rank ``fit_block_size=1`` requires the needed native bond capacity to
exist already.

```python
peps = py.SymPEPS.for_model("itf", 4, 4, bond_dim=2)
ham = peps.build_hamiltonian(jx=-1.0, hz=-0.5)
gauges = {}

peps.time_evolve(
    0.005,
    hamiltonian=ham,
    method="simple",
    gauges=gauges,
    max_bond=8,
)
```

## Measuring observables

Use ``state.measure(obs, where=...)`` for local one- or two-site observables on
both MPS and PEPS. Dense operators are converted to Symmray arrays using the
state's physical sectors. Use ``charge=0`` for symmetry-preserving observables
such as number or ``Z``; use nonzero operator charge for charge-changing
operators.

MPS measurements use direct tensor-network contraction. PEPS measurements use
quimb's PEPS plaquette-environment boundary contraction, so pass ``chi`` or
reusable plaquette-environment holders. Dense operators are converted to
Symmray arrays before calling quimb because quimb's Symmray PEPS gate path
expects block-sparse operators.

```python
import numpy as np

n_op = np.diag([0.0, 1.0])
zz_op = np.diag([1.0, -1.0, -1.0, 1.0])

psi.measure(n_op, where=3)
psi.measure(zz_op, where=(2, 3))

bdy_obs = {}
peps.measure(
    n_op,
    where=(1, 2),
    chi=32,
    bdy=bdy_obs,
    mode="mps",  # or "projector"
)
```

## Inspecting Symmray blocks

Use ``symmray_block_summary`` to inspect the charge sectors and stored block
dimensions of any Symmray-backed local operator or tensor. Use
``draw_symmray_blocks`` for a lightweight ``quimb.schematic`` drawing of the
same information, including the stored-vs-dense entry count.

For whole MPS states, use ``symmray_mps_summary`` and ``draw_symmray_mps``. For
MPOs, use ``symmray_mpo_summary`` and ``draw_symmray_mpo``. For PEPS, use
``symmray_peps_summary`` and ``draw_symmray_peps``. These expose the scientific
structure that is usually hidden in a dense drawing: each site tensor's block
count, physical charge sectors, virtual-bond sector maps, and aggregate
block-sparse storage density. The default schematics follow the compact quimb
style with tensor nodes, physical legs, and bond arrows; extended bond/physical
labels and diagnostics are opt-in.
MPS and MPO drawings also accept ``mapper=OneDMap(...)`` to place the 1D chain
on its 2D lattice path. The mapped view keeps the charge-sector labels but uses
site-colored nodes and quieter gray bonds instead of left/right region shading.
For backwards-compatible notebook use, ``draw_symmray_peps`` also dispatches to
the MPS/MPO drawers when the input is a 1D MPS or MPO object.

In the detailed drawing mode, ``T_i`` is the site tensor, ``B`` is the number of
stored block sectors, and ``e_i`` is a virtual bond. PEPS node circles show
compact white charge labels by default: spin-resolved two-component charges use
total charge ``Q`` and spin projection ``S_z=(N_up-N_down)/2``, while other
charges use the raw Symmray charge ``q`` and total particle number ``N`` where
available. Set ``charge_in_node=False`` to move the raw charge back outside the
node with the tensor label. Bond labels include the two local index
orientations, for example ``out->in``, so the charge-flow convention is visible
on the same line as the bond dimension. Use ``show_bond_sectors=True`` to add
compact ``q_e`` virtual-bond sector maps, and ``show_phys_labels=True`` with
``show_leg_chargemaps=True`` to show ``q_p`` physical-sector maps. For PEPS
wrappers, the overview draws one primary shared index per configured
``SymPEPS.edges`` entry; set ``show_extra_bonds=True`` to debug all non-lattice
or multibond shared indices introduced by routing/gauges. Diagnostics include
both ``charge_total`` and ``Q_total``; for ``Z2`` states ``Q_total`` is reduced
modulo two. Colored block tiles are available with ``show_blocks=True`` for a
focused block-sector view, but the overview diagrams leave them off by default
and show ``B`` instead.
The MPS/PEPS summary dictionaries also expose ``charge_total`` and ``Q_total``
alongside the legacy ``total_charge`` key.

```python
psi = py.SymMPS.for_model(
    "itf",
    4,
    bond_dim=2,
    site_charge=py.site_charge_from_occupations([0] * 4),
)
rz_gate = psi.operator_from_dense(py.rz(0.1), charge=0, sites=1)

summary = py.symmray_block_summary(rz_gate)
summary["blocks"]

gate_drawing = py.draw_symmray_blocks(rz_gate, title="Z2 RZ gate")
display(gate_drawing.fig)

mps_summary = py.symmray_mps_summary(psi.tn)
mps_summary["bonds"]

mps_drawing = py.draw_symmray_mps(psi.tn, title="Symmray ITF MPS")
display(mps_drawing.fig)

detailed_mps_drawing = py.draw_symmray_mps(
    psi.tn,
    title="Symmray ITF MPS with dimensions",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(detailed_mps_drawing.fig)

ham = py.SymHamiltonian.from_edges("tfim", "Z2", [(0, 1), (1, 2), (2, 3)])
mpo = ham.to_mpo(L=4)
mpo_summary = py.symmray_mpo_summary(mpo)
mpo_summary["bonds"]

mpo_drawing = py.draw_symmray_mpo(
    mpo,
    title="Symmray ITF MPO",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(mpo_drawing.fig)

mapper = py.OneDMap(Lx=2, Ly=2, mode="snake")
mapped_mps_drawing = py.draw_symmray_mps(
    psi.tn,
    mapper=mapper,
    title="Symmray ITF MPS on OneDMap",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(mapped_mps_drawing.fig)

mapped_mpo_drawing = py.draw_symmray_mpo(
    mpo,
    mapper=mapper,
    title="Symmray ITF MPO on OneDMap",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(mapped_mpo_drawing.fig)

peps = py.SymPEPS.for_model("itf", 3, 3, bond_dim=2)
peps_summary = py.symmray_peps_summary(peps)
peps_summary["bonds"]

peps_drawing = py.draw_symmray_peps(
    peps,
    title="Symmray ITF PEPS",
    show_bond_labels=True,
    show_diagnostics=True,
)
display(peps_drawing.fig)

sector_debug = py.draw_symmray_peps(
    peps,
    title="Symmray ITF PEPS bond sectors",
    show_bond_labels=True,
    show_bond_sectors=True,
    show_extra_bonds=True,
    show_diagnostics=True,
)
display(sector_debug.fig)
```

At the end of a Symmray ``MpsOptimizer`` notebook, pass the optimized chain
directly:

```python
opt = py.MpsOptimizer(psi.tn.copy(), gates, chi=8, mode="direct")
opt.run(progbar=False)

py.draw_symmray_mps(
    opt.p,
    title="Final Symmray ITF MPS",
    center="middle",
    show_bond_labels=True,
    show_diagnostics=True,
)
```


> API details are maintained as handwritten Markdown in this page.
