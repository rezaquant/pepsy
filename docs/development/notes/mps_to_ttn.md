# MPS-to-TTN conversion: design and upstream audit

Audit date: 2026-09-08. Implementation: `tensors/conversions.py`; public
exports: `pepsy.tensors.mps_to_ttn` and `pepsy.mps_to_ttn`.

## Algorithms

Exact conversion collects subtree messages in postorder. Each message has
the child TTN bond indices and the original MPS virtual indices crossing
that subtree. Contract child messages locally, QR with child/physical legs
on the left and crossing MPS legs on the right, retain Q as the node tensor,
and pass R upward. The root contracts its messages to the root tensor.
This preserves the full coefficient state without any rank tolerance.
Q tensors already carry inward isometry metadata; the root owns the
canonical center. Quimb's extracted input exponent is copied separately.

For finite chi, retain the source MPS plus previously applied adjoint
isometries as an explicit residual network. At each nonroot node, contract
its reduced density matrix, including the residual environment, and retain
the leading eigenvectors. Add their adjoint to the residual and store the
isometry in the output TTN. At the root contract the remaining coefficients.
This avoids first materializing exact TTN tensors. It is a sequential
projection, not a globally optimal variational fit. Eigenvalue degeneracy
at a truncation boundary can change the chosen basis across backends.

The retained dimension is bounded by chi and the product of original MPS
bond dimensions crossing the subtree. Previously installed projections
are on one side of each later cut and cannot increase this rank bound.
Before forming a density matrix, private factors are scaled by their
largest entry. This does not affect the eigenvectors and avoids squaring
an extreme input amplitude. Neither the residual nor output is normalized.

The default `greedy` contraction policy is deterministic and avoids spawning
path-search workers. Users can pass another optimizer. Every contraction
path's maximum intermediate and output size is checked before evaluation;
the local density-matrix/identity allocation is also guarded. This bounds
individual planned arrays, not total process or decomposition workspace.

## Installed APIs and upstream decisions

Python 3.12.11 in the available `cloudspace` environment. The repository's
historical `~/envs/py312/bin/activate` path is absent. NumPy/Numba imports
require a writable `NUMBA_CACHE_DIR` under `/tmp` in the sandbox.

| Dependency | Installed version | Decision |
| --- | --- | --- |
| Quimb | `1.15.1.dev46+g0ad529894` | Adopt existing public contraction, tensor split, and metadata APIs. |
| Autoray | `0.11.1.dev1+gc56f64427` | Adopt `get_namespace(like=array)` for dtype/device-aware identities and native `linalg.eigh` dispatch. |
| Cotengra | `0.8.3.dev7+g1d7fd333f` | Adopt `ContractionTree.max_size()` for the resource guard; retain opt-in path-optimizer selection. |
| Symmray | `0.3.2.dev7+gd63bb4e3f` | Defer graded/block-sparse conversion; reject such input before factorization. |
| Torch | `2.11.0` | Use existing dispatch without registering or replacing process-wide linalg rules. |

Inspected sources:

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html):
  new 1D compression/gating features do not replace this explicit geometry
  conversion. Defer them here; no upstream monkeypatch or compatibility shim.
- [Autoray repository](https://github.com/jcmgray/autoray) and
  [namespace API](https://autoray.readthedocs.io/en/latest/autoapi/autoray/autoray/index.html):
  array creation inherits the sample's device and dtype.
- [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
  [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html): use
  contraction trees directly, without vendoring path-search internals.
- [Symmray repository](https://github.com/jcmgray/symmray): explicit graded
  handling remains necessary. The prescribed
  [Abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
  could not be retrieved by the browser after two attempts; it is not treated
  as evidence of compatibility. Native conversion is deferred.

Local probes confirmed:

```text
Tensor.split(T, left_inds, *, method='auto', absorb='auto', max_bond=None,
             cutoff=1e-10, ..., bond_ind=None, right_inds=None, **kwargs)
qr_stabilized(x, absorb=1, stabilized=True, **kwargs)
TensorNetwork.contraction_tree(self, optimize=None, output_inds=None, **kwargs)
TensorNetwork.contract(..., output_inds=None, optimize=None,
                       preserve_tensor=False, ...)
autoray.get_namespace(like=None, device=None, dtype=None, submodule=None)
```

The exact path explicitly selects `method='qr', stabilized=False` and never
selects an SVD/cutoff route. NumPy `linalg.qr/eigh` dispatch resolves to
NumPy; Torch resolves to its native `linalg_qr/linalg_eigh`. No dispatch
table is modified. `TensorNetwork.make_reduced_density_matrix` is absent
in this Quimb version, so the density matrix is constructed using public
conjugation, reindexing, and contraction operations.

Finite-cap conversion explicitly checks for the Autoray namespace API and
raises a clear capability error if it is absent. No version-string-based
dispatch, dependency upgrade, or package-global compatibility patch is added.

## Validation

`tests/test_mps_to_ttn.py` covers exact coefficient reconstruction for
contiguous and noncontiguous partitions, a physical root, custom physical
indices/tags, global exponent/phase, tiny Schmidt values, zero/single-site
states, cyclic MPS, input isolation, resource limits, and capped gauge
invariance. A crossing-Bell-pair example is exact with a tree cap below the
MPS bond dimension. A random example has nonzero conversion error and obeys
the cap without renormalization. Backend checks cover dtype/device and
replay handoff; GPU checks require explicit device access in this environment.

The API/constructor checks also expose a pre-existing installed metadata
mismatch (`pepsy 0.4.0` versus checkout `0.4.1`). This converter does not
change package version metadata. Default Cotengra worker processes in
existing TreeFIT tests cannot inspect their PIDs inside the process sandbox;
those tests require execution with process access, independently of this
converter's default serial path planning.

The final review added explicit malformed-root/child validation before
tensor work and comments explaining the QR identity, bra/ket index handling,
adjoint projections, and exponent ownership. Regression coverage includes
nonbinary trees, non-qubit physical dimensions, independent single-site
output storage, and the namespace-capability guard.

Validation: all 32 converter tests passed across NumPy, Torch CPU/CUDA,
CuPy, and JAX, including both GPU cap choices. Five selected existing
TreeFIT regressions passed with process access. The public API, constructor,
contraction-dependency, and namespace-capability run passed 40 tests.
Repository-wide Ruff passed. Full repository tests were not run; the earlier
broader API/layout/constructor run had 70 passes, two sandbox GPU skips,
and the installed-version mismatch described above.
