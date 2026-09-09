# pepsy.tensors

This package contains Pepsy's tensor-network construction, mapping,
contraction, validation, observable, backend, and symmetric-state helpers.
Other packages should import these helpers through `pepsy.tensors` or the
top-level `pepsy` exports rather than old flat modules.

## Modules

- `core.py`: main implementations for constructors, `OneDMap`, backend
  defaults, contraction optimizers, observables, and dense TN utilities.
- `constructors.py`: facade for product-state, identity, Haar-random, MPS,
  MPO, PEPS, and PEPO constructors.
- `contractions.py`: facade for contraction optimizers, `tn_norm`,
  `tn_fidelity`, and alignment helpers.
- `maps.py`: facade for `OneDMap`.
- `observables.py`: facade for observable and MPO expectation helpers.
- `mps_transfer.py`: repeating-cell and site-selected local transfer actions,
  dense and bosonic Symmray sector adapters, backend-preserving Arnoldi,
  transfer gaps, momenta, degeneracy, and correlation lengths. Local windows
  default to bulk estimates from a private left-canonicalized open-MPS copy,
  with optional right canonicalization. Supplied-gauge windows require
  `canonicalize=None, allow_local=True`; caller input is preserved. Small unresolved
  gaps are distinct from numerical peripheral modes, and Arnoldi can grow
  its basis within an explicit memory cap.
- `symmetric.py`: Symmray-backed `SymMPS`, `SymPEPS`, symmetric Hamiltonian,
  gate-stream, charge-sector, and dense-operator conversion helpers.
- `validation.py`: shared PEPS tag and physical-index validation helpers.

Many leaf modules are intentionally thin facades over `core.py`; keep that
structure unless a change has a strong reason to split implementation.

## Main responsibilities

`OneDMap` maps regular 2D or 3D lattice coordinates onto a 1D path. Supported
modes include `snake`, `snake-row-major`, `row-major`, `col-major`,
`alternate-x`, `alternate-y`, `alternate-z`, `hilbert`,
`hilbert-row-major`, and `diag`. The x/y/z alternating modes are available
for 3D lattices, with x/y alternating within z layers and z as the inner
alternating direction. The additional `finder` mode composes an MPS
gate-stream layout permutation with a base lattice mode. It analyzes only
gate supports and does not construct, replay, or truncate an MPS:

```python
mapper = py.OneDMap(
    6,
    6,
    mode="finder",
    gate_stream=gates,
    layout_kwargs={"objective": "compression", "order": "quality"},
)
idx2coo, coo2idx = mapper.build()
```

The gate stream's logical labels must be the compact integers `0..Lx*Ly-1`.
Use `finder_base_mode="row-major"` when those labels were originally assigned
by a different regular traversal. An existing MPS layout plan can be supplied
with `finder=plan` instead of `gate_stream=`.

Constructors create common tensor-network states and operators:

- `bell_to_mps` (interleaved physical/ancilla Bell-pair purifications),
  `ps_to_mps` (bond-one product states), `ps_to_ttn`, `ps_to_peps`,
  `ps_to_3dpeps`
- `ps_to_mpo`, `ps_to_pepo`
- `id_to_mpo`, `id_to_pepo`
- `haar_random_state`, `random_haar_qubit`
- `hrs_to_mps`, `hrs_to_ttn`, and `hrs_to_peps` are the canonical random-state
  constructors. The historical `hrps_to_*` spellings remain as deprecated
  compatibility aliases.

Contraction helpers include:

- `build_optimizer(...)` and `build_compressed_optimizer(...)`; the former
  `build_contraction(...)` name remains as a deprecated compatibility alias.
- `contract_hypercompressed_tn(...)`
- `contract_hypercompressed_tn_batch(...)` — torch-only batched amplitudes
  ``<x|psi>`` for many int64 configs via `torch.vmap`, reusing one fixed
  compressed contraction tree (one-hot selection; requires `cutoff=0.0`)
- `tn_norm(...)`, `tn_fidelity(...)`, and `tns_align(...)`

Backend helpers manage package-wide defaults and optional linalg shims:

- `set_default_array_backend(...)` / `get_default_array_backend()`
- `set_default_grad_backend(...)` / `get_default_grad_backend()`
- `TorchLinalgConfig` / `get_torch_linalg_config()` — the single structured
  process-global Torch SVD/QR policy. One `config.register()` call installs
  native or relative-regularized SVD autodiff, the matching QR policy, CPU or
  CUDA driver choices, optional fallbacks, and (when requested) Quimb's raw
  Symmray split drivers. `register_torch_linalg(...)` remains a compatibility
  constructor for this class.
- `reset_default_backends()`
- torch and JAX linalg/stop-gradient registrations. Use
  `register_torch_linalg(...)` as the canonical public setup. Its explicit
  `quimb_split_drivers=True` option also configures Quimb's raw Symmray-block
  split paths; `PepsEnergyOptimizer` enables that option automatically.
  Native thin SVD/QR is the default, while `stabilized=True` installs the
  truncation-safe, relative-regularized SVD rules. The stabilized Torch SVD
  falls back to SciPy `gesvd` on CPU forward-driver failures. The QR/LQ split
  driver uses `phase(0)=1`, preserving rank-deficient dense and Symmray
  reconstructions exactly, and a scale-relative regularized VJP for nonzero
  singular pivots rather than dropping the full block gradient.

## Tag and index conventions

PEPS-like networks should carry lattice and site tags:

- `X{i}` for the x coordinate.
- `Y{j}` for the y coordinate.
- `I...` for site identity tags such as `I0,1`.

Physical outer indices conventionally use `k...` for ket legs and `b...` for
bra or operator-output legs. The boundary and optimizer packages depend on
these conventions for shape inference and layer construction.

## Symmetric tensors

`symmetric.py` provides Symmray-backed convenience wrappers and charge-sector
helpers. Symmray remains optional. Code and tests that depend on it should
import lazily or use `pytest.importorskip("symmray")`.

For spinful Fermi-Hubbard states, the named model presets are:

- `fermi_hubbard`: total particle-number `U1` sectors.
- `fermi_hubbard_u1u1`: spin-resolved `U1U1` sectors with charges
  `(N_up, N_down)`.

Use Gao et al., "Fermionic tensor network contraction for arbitrary
geometries", Phys. Rev. Research 7, 023193 (2025),
https://doi.org/10.1103/PhysRevResearch.7.023193 as the main methods
reference for Pepsy/Symmray Fermi-Hubbard examples. The relevant design cue is
to preserve Symmray fermionic parity, symmetry, and leg-order metadata through
gate application, measurement, boundary contraction, and any future arbitrary
graph lattice wrappers.

`SymMPS.fermionic_ordering()` and `SymPEPS.fermionic_ordering()` expose the
package-level record of site order, edge order, local index directions, and the
methods reference. The same record is included in `symmray_mps_summary(...)`
and `symmray_peps_summary(...)` under the `fermionic_ordering` key.

`SymHamiltonian.to_mpo(...)` builds quimb `MatrixProductOperator` objects from
Symmray block-sparse tensors. It supports generic charge-neutral rank-4
two-site terms such as `tfim`/`Z2` and `heisenberg`/`U1`, spinless
Fermi-Hubbard with `U1` or `Z2` symmetry and `delta=0`, and the specialized
spin-resolved Fermi-Hubbard path for `model="fermi_hubbard_u1u1"` with
`symmetry="U1U1"`. Spinful total-`U1` Fermi-Hubbard
(`model="fermi_hubbard"`) intentionally raises `NotImplementedError` until the
degenerate total-charge MPO convention is implemented.

For spinful `U1U1`, the builder uses the four-state local basis with charges
`(n_up, n_down)`, handles onsite `U`/`mu` terms, spin-dependent scalar-or-pair
`t` and `mu` parameters, and creates hopping channels for both spin species.
Spinless and spinful fermionic MPO paths insert fermionic parity on
intermediate MPS-chain sites for non-adjacent mapped hopping terms, so
coordinate-lattice edges preserve signs after flattening through `OneDMap`.

Coordinate edges require `mapper=OneDMap(...)` or explicit `idx2coo`/`coo2idx`
maps when calling `to_mpo(...)`; already-flat integer edges can pass `L=...`.
The method supports optional MPO compression, physical index IDs, `dtype=`, and
`to_backend=` block conversion. Focused coverage lives in
`tests/test_symmetric_tensors.py::test_*hamiltonian*mpo*`.

When changing symmetric behavior, check both dense compatibility and Symmray
routing through PEPS optimizers and boundary contraction paths.

## Editing notes

- Preserve public exports in `src/pepsy/tensors/__init__.py` and top-level
  `src/pepsy/__init__.py` when adding or removing public symbols.
- Do not reintroduce old flat modules such as `pepsy.core`.
- Avoid changing default contraction optimizers, mapper ordering, numerical
  tolerances, or backend coercion unless the task is specifically about that
  behavior.
- Focused validation usually starts with:

```sh
pytest -q tests/test_tensor_constructors.py tests/test_ham.py
```
