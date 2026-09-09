# Tree SRC, SDC, and zipup algorithm audit

Date: 2026-09-08.

## Implemented algorithms

`optimizers/tree/compression.py` owns the shared dense successive compressor.
Inputs are groups of original target tensors at each tree node, an inward
peel order, and a final hub. TreeMPO application passes the separate operator
and state tensors directly. State-only and two-node edge compression use
the same engine. All exterior state branches are first made isometric toward
the active region, so their boundary bonds are valid output legs.

For a directed cut `u -> v`, SRC contracts the original target component on
the `u` side with independent local Gaussian output sketches. A common sample
index is retained as a hyperindex throughout the contraction. This produces
an environment with dimensions `(samples, original cut bonds...)`; it does
not apply randomized SVD to a local canonical core. Complex sketches use the
live array dtype/device and the supplied compression seed. Scalar environment
rescaling leaves sampled ranges unchanged and controls multiplicative drift.
The batch size is `chi`; projection can use fewer columns when the original
cut dimension is smaller. `chi=None` uses a full-cut rank bound. Nonzero SRC
cutoff is explicitly warned about and ignored, matching its fixed-rank nature.

SDC instead contracts each node with low-rank factors received from the
other neighboring branches, then retains a deterministic truncated-SVD factor
on the outgoing cut. Its environments are bounded by `chi` and the requested
cutoff convention. Direct SVD replaces Gram-matrix factorization here without
changing the low-rank-environment algorithm; it avoids squared conditioning
and the installed NumPy complex64 `svd:eig` compilation problem below.

Two iterative traversals construct directed environments without recursion.
The final inward pass contracts each node's original target layers, projected
child messages, and complementary environment. QR extracts the retained
isometry Q. Contracting Q-dagger with the original local target and child
messages produces the next exact projected message. The hub is the final
projected target, preserving its norm and scale; every other retained node
has an actual inward isometry and matching `left_inds` metadata.

On a path this construction reduces to Quimb's successive environment
algorithms. On a branching tree it is their hierarchical extension: nested
child subspaces are selected using directed sketches of the original
complementary components. We do not claim a published chain error bound
automatically applies to this extension. There is no Cholesky compressor here.

Zipup was already a genuine streamed tree algorithm: it contracts one node's
operator/state layers with incoming child messages and immediately truncates
the outgoing message. It has no complementary-environment pass. The new
regressions explicitly forbid the full-target routing and final canonical
compression paths while checking its state and isometries. It remains
different from both direct compression and SRC/SDC.

Edge histories record dimensions for all three algorithms; local singular
spectra of approximate environments are not reported as global discarded
weights. Norm-survival diagnostics retain their existing separate semantics.
TreeFIT's SRC/SDC guesses use the actual environment algorithms; its later
local variational refinements use ordinary direct SVD.

SRC/SDC currently reject native symmetry arrays rather than replacing the
algorithm or densifying the state. This also removes the former native SDC
alias to direct compression. Native direct and zipup retain their existing
graded routes. CuPy dispatch is supported by the dense operations but GPU
hardware has not been validated.

## Upstream compatibility audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
again returned a retrieval error.

Installed versions: Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev6+g08fe1a3a1`, and Symmray
`0.3.2.dev6+ga17699db6`.

Inspected installed `tensor_network_1d_compress_src`,
`tensor_network_1d_compress_sdc`, and their low-rank-environment projection
sweep, including signatures and handling of `max_bond`, `cutoff`, `seed`,
`project_opts`, and `compress_opts`. The new implementation composes public
`Tensor.split`, `Tensor.isel`, `Tensor.H`, and `tensor_contract`, with explicit
output indices on sample hyperedges. Random arrays use Pepsy's existing
Autoray-backed random helper with explicit dtype, preserving its compatibility
fallback. No upstream code is vendored and no installed package is edited.

Classification: **adopt** successive complementary-environment compression
for the tree geometry; **adopt** stable direct SVD for deterministic factors;
**defer** native graded sketches, GPU benchmarking, and Cholesky compression.

The broader suite independently exposed an existing Quimb/Numba complex64
DM failure: `svd_via_eig_truncated_numpy` cannot compile its full-result branch
because two returns mix complex64/complex128 factors. An isolated public
`Tensor(np.eye(3, dtype='complex64')).split(method='svd:eig', ...)` reproduces
the failure at both zero and positive cutoff, without invoking Pepsy's tree
compressor. The existing tree DM cutoff test reproduces it alone. This change
does not alter DM dispatch or patch that unrelated upstream driver.

## Validation

- Direct algorithm comparison against Quimb on a five-site path, including
  identical SRC sketches and deterministic SDC output.
- Layered, branched MPS-like trees: NumPy, Torch, JAX; complex64/complex128;
  finite caps and lossless caps; a physical root; canonical/isometry metadata;
  exact target projection identities; and seeded reproducibility.
- Partial gate spans, ternary roots, weak branches, zero targets, state-only
  compression, and public two-node compression.
- Full-target routing and local randomized SVD are forbidden in dedicated
  tests, so passing a mode-name-only implementation is insufficient.

No universal speedup or GPU performance claim is made. Directed environments
remain bounded in sample/factor rank, but runtime and memory also depend on
original operator/state cut dimensions and tree arity.

Final checks (counts overlap): the dedicated algorithm suite passed 19 tests;
the broader tree optimizer, TreeFIT, trajectory, sampler, API, and package
layout selection passed 650 tests and skipped 4, with the one independently
reproduced upstream DM complex64 failure described above. The default smoke
suite passed 128 tests. Ruff, skill/catalog validation, and whitespace checks
passed. The full repository suite was not run: validation covered the changed
tree subsystem and its callers, and the known upstream DM failure remains
unresolved.
