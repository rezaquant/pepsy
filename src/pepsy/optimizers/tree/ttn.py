"""A first-class tree-tensor-network state class :class:`TreeTensorNetwork`.

This is the tree analogue of ``quimb``'s :class:`quimb.tensor.MatrixProductState`:
a thin, geometry-owning subclass of ``quimb``'s arbitrary-geometry vector class
:class:`quimb.tensor.TensorNetworkGenVector`.  It carries a rooted
:class:`~pepsy.optimizers.tree.TreePlan` (internal nodes of any arity, not just
binary), a configurable physical-index /
site-tag / node-tag naming scheme, and deterministic tree-edge bond names, so
that higher-level code (notably :class:`~pepsy.optimizers.tree.TreeOptimizer`)
can talk in *node ids* and *qubit labels*. Arbitrary-geometry readout and
low-level tensor operations remain Quimb-backed, while canonicalization,
whole-tree compression, and canonical metadata are owned by this Tree class.

Layout of a tree state
----------------------
* every node of the plan (leaf **and** internal) is one tensor, tagged with the
  structural node tag ``node_tag_id.format(nid)`` (default ``"N{}"``);
* physical-site tensors carry the ``quimb`` site tag
  ``site_tag_id.format(q)`` (default ``"I{}"``) and the physical index
  ``site_ind_id.format(q)`` (default ``"k{}"``) for qubit ``q``; these are
  structural leaves by default;
* the default auto-built plan uses ``top_arity=3`` for the conventional binary
  TTN with three virtual bonds entering the top tensor. Other internal nodes
  then have two child bonds plus one parent bond, so every tensor remains rank
  three; explicit plans can use another root arity;
* a plan may designate one additional ``root_qubit`` carried by the top tensor.
  A binary root then has exactly two child bonds plus this physical leg. Other
  internal nodes remain ancillary bond carriers. This class supplies the
  tree-specific ``local_expectation`` path for both leaf and root sites;
* adjacent nodes ``a`` and ``b`` share the deterministic virtual bond index
  ``_tb{lo}_{hi}`` with ``lo, hi = sorted((a, b))``.

Because the geometry (``_plan``) and naming (``_node_tag_id``) are declared in
:attr:`TreeTensorNetwork._EXTRA_PROPS` they survive ``.copy()`` and every
``quimb`` view/selection operation, exactly like ``site_ind_id`` does for an
MPS.
"""

from __future__ import annotations

import re
import time

import autoray as ar
import numpy as np
import quimb.tensor as qtn
from quimb.tensor.decomp import qr_stabilized as _quimb_qr_stabilized
from quimb.tensor.tensor_core import TensorNetwork
from numbers import Integral

from ...backends import to_float
from .layout import TreePlan, _DEFAULT_TOP_ARITY

__all__ = ["TreeTensorNetwork"]


def _normalize_compression_mode(mode):
    """Normalize the local tree-bond compression decomposition mode."""

    mode = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "svd": "direct",
        "eigh": "dm",
        "density_matrix": "dm",
        "densitymatrix": "dm",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"direct", "dm", "sdc", "src"}:
        raise ValueError(
            "compression_mode must be 'direct', 'dm', 'sdc', or 'src'."
        )
    return mode


def _compression_method(mode):
    """Return the dense Quimb split method for a compression mode."""

    mode = _normalize_compression_mode(mode)
    if mode == "dm":
        return "svd:eig"
    if mode in {"src", "sdc"}:
        raise ValueError("SRC/SDC require complementary environments, not a local split driver")
    return "svd"


def _native_rank_safe_qr(array, backend):
    """Factor a finite complex64 block with safe backend QR."""
    if backend == "torch":
        import torch as xp  # pylint: disable=import-outside-toplevel
    elif backend == "cupy":
        import cupy as xp  # pylint: disable=import-outside-toplevel
    else:  # pragma: no cover - only native GPU backends call this helper
        raise ValueError(f"unsupported native QR backend {backend!r}")

    rows, cols = array.shape
    rank_limit = min(rows, cols)
    r_factor = array.clone()
    q_factor = xp.eye(rows, dtype=array.dtype)

    for index in range(rank_limit):
        vector = r_factor[index:, index]
        magnitude = vector.abs()
        scale = magnitude.amax()
        # Construct the reflector from a max-normalized vector. This avoids
        # the subnormal norm division that makes Torch's complex64 QR fail.
        valid_scale = xp.isfinite(scale) & (scale > 0)
        safe_scale = xp.where(valid_scale, scale, xp.ones_like(scale))
        normalized = vector / safe_scale
        norm = xp.sqrt(xp.sum(normalized.abs() ** 2))
        valid_norm = valid_scale & xp.isfinite(norm) & (norm > 0)
        safe_norm = xp.where(valid_norm, norm, xp.ones_like(norm))
        first = normalized[0]
        first_abs = first.abs()
        safe_first_abs = xp.where(
            first_abs > 0, first_abs, xp.ones_like(first_abs),
        )
        phase = xp.where(
            first_abs > 0,
            first / safe_first_abs,
            xp.ones_like(first),
        )
        alpha = -phase * safe_norm
        reflector = normalized.clone()
        reflector[0] = reflector[0] - alpha
        reflector_norm = xp.sum(reflector.conj() * reflector).real
        valid_reflector = (
            valid_norm
            & xp.isfinite(reflector_norm)
            & (reflector_norm > 0)
        )
        safe_reflector_norm = xp.where(
            valid_reflector, reflector_norm, xp.ones_like(reflector_norm),
        )
        beta = xp.where(
            valid_reflector,
            2 / safe_reflector_norm,
            xp.zeros_like(reflector_norm),
        )

        trailing = r_factor[index:, index:]
        r_factor[index:, index:] = trailing - reflector[:, None] * (
            beta * (reflector.conj() @ trailing)
        )[None, :]

        q_trailing = q_factor[:, index:]
        q_factor[:, index:] = q_trailing - (q_trailing @ reflector)[:, None] * (
            beta * reflector.conj()
        )[None, :]

    return q_factor[:, :rank_limit], r_factor[:rank_limit, :]


def _native_factors_finite(factors, backend):
    """Check native QR factors without converting GPU arrays to NumPy."""
    if factors is None:
        return False
    for factor in factors:
        if factor is None:
            continue
        if backend == "torch":
            finite = factor.isfinite().all().item()
        elif backend == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            finite = cp.isfinite(factor).all().item()
        else:
            finite = np.isfinite(factor).all()
        if not bool(finite):
            return False
    return True


def _native_cast_complex128(array, backend):
    """Promote one native GPU block without moving it off device."""
    if backend == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        return array.to(dtype=torch.complex128)
    if backend == "cupy":
        import cupy as cp  # pylint: disable=import-outside-toplevel

        return array.astype(cp.complex128, copy=False)
    raise ValueError(f"unsupported native GPU backend {backend!r}")


def _native_cast_like(array, reference, backend):
    """Cast one native factor back to the original device and dtype."""
    if backend == "torch":
        return array.to(dtype=reference.dtype, device=reference.device)
    if backend == "cupy":
        return array.astype(reference.dtype, copy=False)
    raise ValueError(f"unsupported native GPU backend {backend!r}")


def _native_cast_factors(factors, reference, backend):
    """Cast non-empty QR factors back to the original native dtype."""
    return tuple(
        None if factor is None else _native_cast_like(factor, reference, backend)
        for factor in factors
    )


def _native_qr_exponent_span(array, backend):
    """Return the stored nonzero magnitude exponent span of one native block."""
    if backend == "torch":
        magnitude = array.detach().abs()
        nonzero = magnitude[magnitude > 0]
        if nonzero.numel() == 0:
            return None
        minimum = float(nonzero.amin().item())
        maximum = float(magnitude.amax().item())
    elif backend == "cupy":
        magnitude = ar.do("abs", array)
        nonzero = magnitude[magnitude > 0]
        if nonzero.size == 0:
            return None
        minimum = float(nonzero.min().item())
        maximum = float(magnitude.max().item())
    else:
        return None
    if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum <= 0:
        return None
    _, maximum_exponent = np.frexp(maximum)
    _, minimum_exponent = np.frexp(minimum)
    return int(maximum_exponent - minimum_exponent)


def _native_qr_block_scaled(array, **kwargs):
    """QR one native charge block with a reversible scaling fallback.

    Torch's complex64 QR can return NaNs for a rank-deficient block even when
    its largest entry is moderate, if other entries in the same block are
    many orders of magnitude smaller. It can also return finite factors while
    losing those tiny entries. Healthy blocks keep Torch's native QR path
    unchanged; small or extremely wide-range blocks use a reversible
    power-of-two scaling, with a native rank-safe fallback for structural rank
    deficiency.
    """
    opts = dict(kwargs)
    opts.pop("method", None)
    opts.pop("fn", None)

    def native_qr(
        x, qr_opts, *, rank_safe=False, allow_failure=False,
    ):
        """Run one native backend QR block without composed dispatch."""
        absorb = qr_opts.get("absorb", "right")
        left_like = absorb in {
            -1, "left", "Us,VH", "lfactor", "Us",
        }
        qr_kwargs = {
            key: value for key, value in qr_opts.items()
            if key not in {"absorb", "stabilized"}
        }
        if left_like:
            x = ar.do("transpose", x, (1, 0))
        try:
            if (
                backend == "torch"
                and getattr(getattr(x, "device", None), "type", "cpu") == "cpu"
            ):
                # A previous autodiff run can leave a stabilized real/complex
                # QR rule in Autoray's process-global Torch namespace. This
                # helper explicitly requests the native complex64 path, so do
                # not let an unrelated registration change its forward rule.
                import torch  # pylint: disable=import-outside-toplevel

                registered_qr = ar.get_lib_fn("torch", "linalg.qr")
                if registered_qr is not torch.linalg.qr:
                    q, r = torch.linalg.qr(x, **qr_kwargs)
                else:
                    q, r = ar.do("linalg.qr", x, **qr_kwargs)
            else:
                q, r = ar.do("linalg.qr", x, **qr_kwargs)
        except Exception:
            if not allow_failure:
                raise
            q, r = None, None
        if rank_safe and not _native_factors_finite((q, r), backend):
            # Native GPU QR can still emit NaNs for finite, structurally
            # rank-deficient blocks after scaling. Use a backend-native
            # fallback that avoids division by subnormal norms and completes
            # the orthonormal basis explicitly for zero sectors.
            q, r = _native_rank_safe_qr(x, backend)
        if q is None or r is None:
            return None
        if left_like:
            left = ar.do("transpose", r, (1, 0))
            right = ar.do("transpose", q, (1, 0))
            if absorb in {-1, "left", "Us,VH"}:
                return left, None, right
            return left, None, None
        if absorb in {"lorthog", "U", 10}:
            return q, None, None
        if absorb in {"rfactor", "sVH", 11}:
            return None, None, r
        return q, None, r

    try:
        backend = ar.infer_backend(array)
    except (AttributeError, TypeError):
        backend = None

    # ``array_split`` calls this once per native charge block.  For the
    # Torch path, bypassing quimb's composed linalg wrapper removes a Python
    # dispatch layer from every block while retaining exactly the same
    # reduced QR and the same ``stabilized=False`` policy.
    use_torch_qr = backend == "torch"
    use_native_gpu_qr = backend == "cupy" or (
        backend == "torch"
        and getattr(getattr(array, "device", None), "type", None) == "cuda"
    )
    if ar.get_dtype_name(array) != "complex64":
        if use_torch_qr or use_native_gpu_qr:
            return native_qr(array, opts)
        return _quimb_qr_stabilized(array, **opts)

    if use_native_gpu_qr:
        # Keep the normal requested-dtype GPU QR path unchanged. Only a
        # genuinely nonfinite result pays for the same-device double retry.
        direct = native_qr(array, opts, allow_failure=True)
        if _native_factors_finite(direct, backend):
            return direct

        high = _native_cast_complex128(array, backend)
        high_result = native_qr(high, opts, allow_failure=True)
        if _native_factors_finite(high_result, backend):
            return _native_cast_factors(high_result, array, backend)

        if backend == "torch":
            block_max = float(array.detach().abs().amax().item())
        else:
            block_max = to_float(ar.do("max", ar.do("abs", array)))
    elif use_torch_qr:
        block_max = float(array.detach().abs().amax().item())
        # Very small blocks have historically needed the scaled path even
        # when the current Torch build happens to return finite QR factors.
        # For moderate blocks, make the same choice when stored entries span
        # enough binary exponents that a finite direct QR can still erase the
        # smallest charge sector. The direct call is retained in that case so
        # healthy blocks keep the one-call fast path and diagnostics can show
        # the protective retry.
        direct = None
        if not (
            np.isfinite(block_max)
            and block_max != 0.0
            and block_max < 2.0**-8
        ):
            direct = native_qr(array, opts)
            exponent_span = _native_qr_exponent_span(array, backend)
            needs_scale = exponent_span is not None and exponent_span >= 64
            if _native_factors_finite(direct, backend) and not needs_scale:
                return direct
    else:
        block_max = to_float(ar.do("max", ar.do("abs", array)))
    if not np.isfinite(block_max) or block_max == 0.0:
        # Preserve the original failure behaviour for non-finite input, while
        # allowing genuinely empty structural sectors through unchanged.
        if use_torch_qr or use_native_gpu_qr:
            return direct
        return _quimb_qr_stabilized(array, **opts)

    # Normalize the exceptional finite block before retrying QR. A threshold
    # based only on ``block_max`` is insufficient: the failing native block
    # has max magnitude ~9e-3 but also contains entries ~1e-32 and structural
    # zeros. Use a power of two so the scaling is exactly reversible in the
    # complex64 representation.
    _, exponent = np.frexp(block_max)
    scale = float(np.ldexp(1.0, -int(exponent)))
    if use_native_gpu_qr:
        high_scaled = native_qr(high * scale, opts, allow_failure=True)
        if _native_factors_finite(high_scaled, backend):
            left, singular_values, right = _native_cast_factors(
                high_scaled, array, backend,
            )
        else:
            fallback = native_qr(
                array * scale,
                opts,
                rank_safe=True,
                allow_failure=True,
            )
            if not _native_factors_finite(fallback, backend):
                raise RuntimeError(
                    "native GPU QR failed in requested and complex128 "
                    "dtypes, including the rank-safe fallback."
                )
            left, singular_values, right = fallback
    elif use_torch_qr:
        fallback = native_qr(
            array * scale,
            opts,
            rank_safe=True,
            allow_failure=True,
        )
        if not _native_factors_finite(fallback, backend):
            raise RuntimeError(
                "native Torch QR failed in requested dtype and its "
                "rank-safe fallback."
            )
        left, singular_values, right = fallback
    else:
        left, singular_values, right = _quimb_qr_stabilized(
            array * scale, **opts,
        )

    # ``absorb='left'`` is the LQ orientation: the left factor carries the
    # scale. All other QR orientations carry it in the right factor.
    absorb = opts.get("absorb", "right")
    if absorb in {-1, "left", "Us,VH", "lfactor", "Us"}:
        if left is not None:
            left = left / scale
    elif right is not None:
        right = right / scale
    return left, singular_values, right


try:  # quimb renamed/removed this generic-vector base across releases.
    from quimb.tensor import TensorNetworkGenVector
except ImportError:  # pragma: no cover - exercised with older quimb releases
    class TensorNetworkGenVector(TensorNetwork):
        """Small compatibility base for quimb versions without GenVector."""

        @property
        def nsites(self):
            return len(getattr(self, "_sites", ()))

        @property
        def sites(self):
            return tuple(getattr(self, "_sites", ()))

        @property
        def site_ind_id(self):
            return self._site_ind_id

        @property
        def site_tag_id(self):
            return self._site_tag_id

        def site_ind(self, site):
            return self._site_ind_id.format(site)

        def site_tag(self, site):
            return self._site_tag_id.format(site)

        def local_expectation(self, operator, where, *, optimize="auto-hq", **kwargs):
            """Evaluate a local expectation using a generic TN contraction."""
            if isinstance(where, Integral):
                where = (int(where),)
            else:
                where = tuple(where)
            operated = qtn.tensor_network_gate_inds(
                self,
                operator,
                [self.site_ind(site) for site in where],
                contract=False,
                inplace=False,
                tags=[],
            )
            numerator = (self.H | operated).contract(all, optimize=optimize)
            denominator = (self.H | self).contract(all, optimize=optimize)
            return numerator / denominator


def _bond_index(a, b):
    """Return the deterministic virtual-bond index name for edge ``(a, b)``."""
    lo, hi = (a, b) if a < b else (b, a)
    return f"_tb{lo}_{hi}"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _is_symmray_array(value):
    """Whether ``value`` is a native Symmray block-sparse array."""
    try:
        return ar.infer_backend(value) == "symmray"
    except (AttributeError, TypeError):
        return hasattr(value, "blocks") and hasattr(value, "indices")


def _native_qr_options_for_tensor(tensor):
    """Return the centralized lossless-QR options for one tensor."""
    return {"stabilized": False} if _is_symmray_array(tensor.data) else {}


def _native_qr_split_tensor(tensor, **kwargs):
    """Split one tree tensor using the native graded QR policy."""
    kwargs.update(_native_qr_options_for_tensor(tensor))
    # QR is a lossless gauge move. Keep it independent from the optimizer's
    # truncating SVD cutoff even though Quimb's shared split signature exposes
    # a nonzero cutoff default.
    kwargs.setdefault("cutoff", 0.0)
    if _is_symmray_array(tensor.data):
        kwargs.setdefault("fn", _native_qr_block_scaled)
    kwargs.setdefault("method", "qr")
    return tensor.split(**kwargs)


def _is_native_mpo(value):
    """Return whether an MPO visibly contains native Symmray tensors."""
    marker = getattr(value, "pepsy_tree_native", None)
    if marker is not None:
        return bool(marker)
    tensors = getattr(value, "tensors", None)
    if tensors is None:
        return None
    try:
        return any(_is_symmray_array(tensor.data) for tensor in tensors)
    except (AttributeError, TypeError):
        return None


def _contract_two_tensors(left, right, *, shared_ind=None):
    """Contract two tensors along one ordinary shared index cheaply.

    Tree routing and native reduced compression repeatedly contract adjacent
    tensors along one virtual edge. Dispatching that ordinary operation
    directly to the active backend avoids rebuilding a generic Quimb/Cotengra
    expression while retaining Symmray's graded fermionic ``tensordot``.
    Unusual hyperedges fall back to Quimb's general contraction path.
    """
    if shared_ind is None:
        shared = tuple(qtn.bonds(left, right))
        if len(shared) != 1:
            return qtn.tensor_contract(left, right)
        shared_ind = shared[0]
    elif shared_ind not in left.inds or shared_ind not in right.inds:
        return qtn.tensor_contract(left, right)

    left_rest = tuple(ind for ind in left.inds if ind != shared_ind)
    right_rest = tuple(ind for ind in right.inds if ind != shared_ind)
    if set(left_rest) & set(right_rest):
        return qtn.tensor_contract(left, right)

    axes = ((left.inds.index(shared_ind),), (right.inds.index(shared_ind),))
    try:
        if _is_symmray_array(left.data):
            # A single shared tree bond is the hot native operation.  Fused
            # Symmray contraction is a good general default, but on Torch CPU
            # it first builds larger fused block views for a contraction that
            # is already one-leg local.  Blockwise dispatch keeps the charge
            # sectors small and avoids that temporary workspace.  CUDA/Torch
            # and other backends retain the fused path, which usually wins by
            # reducing the number of small kernel launches.
            mode = "fused"
            if getattr(left.data, "backend", None) == "torch":
                blocks = getattr(left.data, "blocks", None)
                sample = next(iter(blocks.values()), None) if blocks else None
                if not bool(getattr(sample, "is_cuda", False)):
                    mode = "blockwise"
            data = left.data.tensordot(
                right.data,
                axes=axes,
                mode=mode,
                preserve_array=True,
            )
        else:
            data = ar.do(
                "tensordot", left.data, right.data, axes=axes,
            )
    except (AttributeError, NotImplementedError, TypeError):
        return qtn.tensor_contract(left, right)

    return qtn.Tensor(
        data=data,
        inds=left_rest + right_rest,
        tags=left.tags | right.tags,
    )


def _visible_len(s):
    """Length of ``s`` ignoring any ANSI colour escape sequences."""
    return len(_ANSI_RE.sub("", s))


def _ascii_place(s, width, col):
    """Return ``s`` padded to ``width`` with its centre aligned at column ``col``.

    ANSI-colour aware: the padding is computed from the *visible* length so that
    embedded colour escapes never shift the drawing.
    """
    vis = _visible_len(s)
    start = col - (vis - 1) // 2
    start = max(0, min(start, width - vis))
    return " " * start + s + " " * (width - start - vis)


# Depth-cycled palette for the internal-node markers, a distinct leaf colour,
# and a dim style for the bond numbers / connector lines so the coloured nodes
# stand out.  256-colour SGR codes (rendered by VS Code notebooks and modern
# terminals); on a mono terminal they degrade to plain text.
_LAYER_COLORS = (
    "\x1b[1;38;5;213m",  # root  -> bright magenta / pink
    "\x1b[1;38;5;45m",   #          bright cyan
    "\x1b[1;38;5;82m",   #          bright green
    "\x1b[1;38;5;220m",  #          gold
    "\x1b[1;38;5;208m",  #          orange
    "\x1b[1;38;5;99m",   #          violet
)
_LEAF_COLOR = "\x1b[1;38;5;39m"   # leaves (◆) -> strong blue
_DIM_STYLE = "\x1b[38;5;244m"     # bond dims + connectors -> grey
_RESET = "\x1b[0m"


def _color(s, code, enable):
    """Wrap ``s`` in the SGR ``code`` when ``enable`` is set, else return ``s``."""
    return f"{code}{s}{_RESET}" if enable and code else s



class TreeTensorNetwork(TensorNetworkGenVector):
    """A rooted tree-tensor-network state over physical qubit nodes.

    Subclasses :class:`quimb.tensor.TensorNetworkGenVector`, so it *is* a
    ``quimb`` tensor network: all of ``quimb``'s arbitrary-geometry methods
    (``canonize_around``, ``canonize_between``, ``compress_between``,
    ``gate_inds``, ``to_dense``, ``copy`` ...) work directly. This class provides
    the tree-specific ``local_expectation`` implementation and owns the
    geometry/node/site/index naming glue on top of a :class:`TreePlan`.

    Prefer the builders :meth:`from_plan`, :meth:`from_order`, and :meth:`rand`
    over calling the constructor with raw tensors.

    Parameters
    ----------
    ts : sequence of quimb.tensor.Tensor or quimb.tensor.TensorNetwork
        The tensors of the network, or an existing network to cast/copy.
    plan : TreePlan
        The rooted tree structure (internal nodes may have any arity).  Required
        unless ``ts`` is an existing tensor network being copied/cast (then it
        is taken from ``ts``).
    sites : sequence, optional
        The site (qubit) labels.  Defaults to ``range(plan.n)``.
    site_tag_id, site_ind_id, node_tag_id : str
        Format strings for the ``quimb`` site tag, the physical index, and the
        structural node tag.  Defaults ``"I{}"``, ``"k{}"``, ``"N{}"``.
    """

    _EXTRA_PROPS = (
        "_sites",
        "_site_tag_id",
        "_site_ind_id",
        "_plan",
        "_node_tag_id",
        "_canonical_region",
        "_symmetry",
        "_fermionic",
        "_physical_sectors",
        "_work_bond_counter",
    )

    def __init__(self, ts=(), *, plan=None, sites=None, site_tag_id="I{}",
                 site_ind_id="k{}", node_tag_id="N{}", symmetry=None,
                 fermionic=False, physical_sectors=None, **tn_opts):
        # Copy / cast path: quimb's base ``__init__`` copies ``_EXTRA_PROPS``
        # (``_plan``, ``_node_tag_id``, ...) straight off ``ts``; returning here
        # avoids clobbering them with the fresh-construction defaults below.
        if isinstance(ts, TensorNetwork):
            super().__init__(ts, **tn_opts)
            if isinstance(ts, TreeTensorNetwork):
                if plan is not None and (
                    plan.root != ts.plan.root
                    or plan.children != ts.plan.children
                    or plan.qubit_of_leaf != ts.plan.qubit_of_leaf
                    or plan.root_qubit != ts.plan.root_qubit
                ):
                    raise ValueError(
                        "plan does not match the TreeTensorNetwork being copied."
                    )
                self._fermionic_norm_cache = None
                self._fermionic_norm_cache_version = 0
                self._fermionic_norm_cache_value_version = None
                self._work_bond_counter = getattr(ts, "_work_bond_counter", 0)
                return
            if plan is None:
                raise TypeError(
                    "casting a plain TensorNetwork to TreeTensorNetwork requires "
                    "an explicit TreePlan."
                )
            self._plan = plan
            self._sites = tuple(range(plan.n)) if sites is None else tuple(sites)
            self._site_tag_id = site_tag_id
            self._site_ind_id = site_ind_id
            self._node_tag_id = node_tag_id
            self._canonical_region = None
            self._symmetry = symmetry
            self._fermionic = bool(fermionic)
            self._physical_sectors = physical_sectors
            self._fermionic_norm_cache = None
            self._fermionic_norm_cache_version = 0
            self._fermionic_norm_cache_value_version = None
            self._work_bond_counter = 0
            self.validate()
            return
        super().__init__(ts, **tn_opts)
        if plan is None:
            raise ValueError(
                "TreeTensorNetwork requires a TreePlan (pass plan=...)."
            )
        self._plan = plan
        self._sites = tuple(range(plan.n)) if sites is None else tuple(sites)
        self._site_tag_id = site_tag_id
        self._site_ind_id = site_ind_id
        self._node_tag_id = node_tag_id
        self._symmetry = symmetry
        self._fermionic = bool(fermionic)
        self._physical_sectors = physical_sectors
        self._fermionic_norm_cache = None
        self._fermionic_norm_cache_version = 0
        self._fermionic_norm_cache_value_version = None
        # Frozenset of node ids forming the canonicalised subtree (``None`` if
        # unknown); a one-node region is exactly an orthogonality centre.
        # Tracked here -- surviving ``.copy()`` via ``_EXTRA_PROPS`` -- so the
        # canonical form is a property of the *state*, not of any one driver.
        self._canonical_region = None
        self._work_bond_counter = 0

    def _new_work_bond(self, kind, *nodes):
        """Return a unique private bond label for a native decomposition.

        Native QR/SVD factors are short-lived, but a routed operator bond can
        become a live tree edge before the next hop.  A state-owned counter is
        cheaper than generating a UUID for every factor and, because it is an
        extra copied property, remains collision-free across TTN branches.
        """
        counter = int(getattr(self, "_work_bond_counter", 0))
        self._work_bond_counter = counter + 1
        suffix = "_".join(str(node) for node in nodes)
        if suffix:
            suffix = "_" + suffix
        return f"_pepsy_{kind}_{counter}{suffix}"

    # -- mutation / canonical metadata --------------------------------------

    def invalidate_canonical_form(self):
        """Forget the tracked canonical region after an unmanaged mutation.

        Quimb exposes mutating tensor-network methods and tensors themselves
        can be modified in place. The TTN cannot intercept every such mutation,
        so direct callers should use this method after changing tensor data
        outside the state-aware wrappers below.
        """
        self._canonical_region = None
        self._invalidate_norm_cache()
        return self

    def _invalidate_norm_cache(self):
        """Invalidate the cached native-fermion norm denominator."""
        self._fermionic_norm_cache = None
        self._fermionic_norm_cache_value_version = None
        self._fermionic_norm_cache_version = (
            getattr(self, "_fermionic_norm_cache_version", 0) + 1
        )
        return self

    def _invalidate_after_mutation(self, result):
        self._canonical_region = None
        self._invalidate_norm_cache()
        return result

    def gate_inds_(self, *args, **kwargs):
        """Apply a Quimb gate and invalidate canonical metadata."""
        return self._invalidate_after_mutation(
            super().gate_inds_(*args, **kwargs)
        )

    def canonize_between(self, *args, **kwargs):
        """Canonicalize through Quimb and invalidate the tracked centre."""
        return self._invalidate_after_mutation(
            super().canonize_between(*args, **kwargs)
        )

    def compress_between(self, *args, **kwargs):
        """Compress through Quimb and invalidate the tracked centre."""
        return self._invalidate_after_mutation(
            super().compress_between(*args, **kwargs)
        )

    def canonize_around_(self, *args, **kwargs):
        """Canonicalize around tags through Quimb and invalidate metadata."""
        return self._invalidate_after_mutation(
            super().canonize_around_(*args, **kwargs)
        )

    # -- geometry / naming ----------------------------------------------------

    @property
    def plan(self):
        """The :class:`TreePlan` describing the tree structure."""
        return self._plan

    @property
    def map_mode(self):
        """Canonical geometric label for the tree's leaf layout."""

        return self.plan.map_mode

    @property
    def top_arity(self):
        """Number of virtual child bonds entering the structural root."""
        return self._plan.top_arity

    @property
    def max_virtual_degree(self):
        """Largest number of virtual bonds incident on any live tensor."""
        return self._plan.max_virtual_degree()

    @property
    def max_tensor_rank(self):
        """Largest virtual/physical leg count in the live tree."""
        return self._plan.max_tensor_rank()

    def is_binary(self, *, allow_ternary_root=True):
        """Whether the TTN is binary below an optional ternary top tensor."""
        return self._plan.is_binary(allow_ternary_root=allow_ternary_root)

    @property
    def node_tag_id(self):
        """Format string for the structural node tag (e.g. ``"N{}"``)."""
        return self._node_tag_id

    @property
    def symmetry(self):
        """Native Symmray symmetry label, or ``None`` for dense tensors."""
        return self._symmetry

    @property
    def fermionic(self):
        """Whether the live tensor data uses Symmray's fermionic arrays."""
        return self._fermionic

    def _fermionic_norm_squared(self):
        """Return the exact native-fermion norm squared with caching.

        A known centre uses Symmray's graded one-tensor contraction. If the
        gauge is unknown, contract the complete doubled network so Quimb's
        fermionic contraction machinery retains all graded boundary phases.
        The result is cached until a state mutation invalidates it.
        """
        if not self.fermionic:
            raise TypeError("fermionic norm readout requires a fermionic TTN.")

        cache = getattr(self, "_fermionic_norm_cache", None)
        version = getattr(self, "_fermionic_norm_cache_version", 0)
        if (
            cache is not None
            and getattr(self, "_fermionic_norm_cache_value_version", None)
            == version
        ):
            return cache

        center = self.orthogonality_center
        if center is not None:
            value = self._fermionic_center_norm_squared(center)
        else:
            value = (self.H | self).contract(all, optimize="auto")
        self._fermionic_norm_cache = value
        self._fermionic_norm_cache_value_version = version
        return value

    def _fermionic_center_norm_squared(self, center=None):
        """Read a canonical center norm with Symmray's graded conjugation.

        ``Tensor.H`` only conjugates the data. For a fermionic tensor, the
        one-tensor network conjugation also applies parity phase flips on its
        outer legs. Those flips are the graded identity supplied by the
        canonical exterior, so this remains a one-tensor readout while
        retaining the correct fermionic norm. If no single center is known,
        use the complete doubled-network contraction instead.
        """
        if not self.fermionic:
            raise TypeError("fermionic center norms require a fermionic TTN.")
        if center is None:
            center = self.orthogonality_center
        if center is None:
            return self._fermionic_norm_squared()
        tensor = self.node_tensor(center).copy()
        singleton = qtn.TensorNetwork([tensor])
        return (singleton.H & singleton).contract(all, optimize="auto")

    def _fermionic_local_expectation(
        self, operator, where, *, optimize, normalized,
    ):
        """Evaluate a native observable with the complete graded exterior."""
        profile_sink = getattr(self, "_profile_sink", None)
        profile_started = time.perf_counter() if profile_sink is not None else None
        try:
            inds = [self.site_ind(site) for site in where]
            operated = qtn.tensor_network_gate_inds(
                self,
                operator,
                inds,
                contract=False,
                tags=[],
                info=None,
                inplace=False,
            )
            numerator = (self.H | operated).contract(
                all,
                optimize=optimize,
            )
            if not normalized:
                return numerator
            denominator = self._fermionic_norm_squared()
            return numerator / denominator
        finally:
            if profile_started is not None:
                profile_sink.append({
                    "kind": "native_observable",
                    "support": tuple(where),
                    "seconds": time.perf_counter() - profile_started,
                })

    def _restore_readout_region(self, region):
        """Restore a dense readout's tracked canonical region."""
        if region is None:
            return
        if len(region) == 1:
            self.shift_orthogonality_center(next(iter(region)))
        else:
            self.canonize_subtree_(region)

    @property
    def physical_sectors(self):
        """Native Symmray physical-sector map when one was supplied."""
        return self._physical_sectors

    @property
    def root(self):
        """The root node id of the tree."""
        return self._plan.root

    def local_expectation(
        self, operator, where, *, max_bond=None, optimize="auto",
        normalized=True, **kwargs,
    ):
        """Evaluate a local observable with a backend-specific exact path.

        Dense/nonfermionic trees use the canonical target physical node or minimal
        Steiner subtree and cancel its ordinary isometric exterior. Native
        fermionic trees keep the Symmray operator structured and contract the
        complete doubled tree so graded boundary phases are never discarded.

        ``max_bond`` and extra keyword arguments are accepted for Quimb API
        compatibility. This exact tree contraction does not truncate.
        """
        preserve_gauge = bool(kwargs.pop("_preserve_gauge", True))
        _ = max_bond, kwargs
        if isinstance(where, Integral):
            where = (int(where),)
        else:
            where = tuple(int(site) for site in where)
        if not where or len(set(where)) != len(where):
            raise ValueError("where must contain distinct tree sites.")
        if any(site not in self.plan.node_of_qubit for site in where):
            raise ValueError(f"site(s) {where!r} are outside this tree state.")

        if not self.fermionic and preserve_gauge:
            original_region = self.canonical_region
            if original_region is None:
                # An unknown dense gauge cannot be reconstructed after a
                # target canonicalisation, so perform the readout on an
                # independent copy. Known regions use the cheaper round-trip
                # restoration below.
                work = self.copy()
                return work.local_expectation(
                    operator,
                    where,
                    max_bond=max_bond,
                    optimize=optimize,
                    normalized=normalized,
                    _preserve_gauge=False,
                )
        else:
            original_region = None

        site_nodes = [self.plan.node_of_qubit[site] for site in where]
        phys = [self.site_ind(site) for site in where]
        op = operator
        if self.symmetry is not None and not _is_symmray_array(op):
            raise TypeError(
                "native Symmray TTNs require a native Symmray observable; "
                "use Fermion.observable(...) or another Symmray operator."
            )
        if self.fermionic:
            expected_rank = 2 * len(where)
            if len(ar.shape(op)) != expected_rank:
                raise ValueError(
                    f"a {len(where)}-site Symmray observable must have "
                    f"rank {expected_rank}."
                )
            return self._fermionic_local_expectation(
                op,
                where,
                optimize=optimize,
                normalized=normalized,
            )
        if len(where) == 1:
            self.shift_orthogonality_center(site_nodes[0])
            tensor = self.node_tensor(site_nodes[0])
            physical = phys[0]
            if not _is_symmray_array(op):
                dim = int(tensor.shape[tensor.inds.index(physical)])
                op = ar.do("reshape", op, (dim, dim))
            elif len(ar.shape(op)) != 2:
                raise ValueError("a one-site Symmray observable must be rank two.")
            gate = qtn.Tensor(op, inds=(physical + "*", physical))
            bra = tensor.H.reindex_({physical: physical + "*"})
            numerator = qtn.tensor_contract(bra, gate, tensor, output_inds=[])
            denominator = qtn.tensor_contract(tensor.H, tensor, output_inds=[])
            result = numerator / denominator if normalized else numerator
            if preserve_gauge:
                self._restore_readout_region(original_region)
            return result

        span = self.steiner_nodes(site_nodes)
        if self.orthogonality_center not in span:
            self.shift_orthogonality_center(site_nodes[0])

        internal = {
            self.bond(node, neighbor)
            for node in span
            for neighbor in self.neighbors(node)
            if neighbor in span
        }
        ket = qtn.TensorNetwork([
            self.node_tensor(node).copy() for node in span
        ])
        if not _is_symmray_array(op):
            dims = [
                int(self.node_tensor(node).shape[
                    self.node_tensor(node).inds.index(physical)
                ])
                for node, physical in zip(site_nodes, phys)
            ]
            op = ar.do("reshape", op, tuple(dims + dims))
        elif len(ar.shape(op)) != 2 * len(where):
            raise ValueError(
                "a multi-site Symmray observable must have one output and "
                "one input leg per site."
            )
        gate = qtn.Tensor(op, inds=[p + "*" for p in phys] + phys)
        internal_map = {index: qtn.rand_uuid() for index in internal}
        bra_num = ket.H.reindex({
            **internal_map,
            **{physical: physical + "*" for physical in phys},
        })
        numerator = (bra_num & gate & ket).contract(
            output_inds=[], optimize=optimize,
        )
        if not normalized:
            if preserve_gauge:
                self._restore_readout_region(original_region)
            return numerator
        bra_den = ket.H.reindex(internal_map)
        denominator = (bra_den & ket).contract(
            output_inds=[], optimize=optimize,
        )
        result = numerator / denominator
        if preserve_gauge:
            self._restore_readout_region(original_region)
        return result

    def local_expectations(self, terms, *, optimize="auto", normalized=True):
        """Evaluate many local observables, reusing the path and the norm.

        ``terms`` maps each ``where`` (an int site or a tuple of sites) to its
        operator. Every term is delegated to :meth:`local_expectation` using a
        *shared* ``optimize`` handle, so a reusable contraction optimiser (e.g.
        :func:`pepsy.build_optimizer`) caches one contraction path per
        contraction topology instead of re-planning for every term. For a native
        fermionic tree the graded norm denominator is memoized (see
        :meth:`_fermionic_norm_squared`), so ``normalized=True`` computes it once
        across the whole batch rather than per term. The per-term graded
        contraction is unchanged, so each value matches the corresponding
        :meth:`local_expectation` call exactly.

        Returns a ``{where: value}`` dict following the iteration order of
        ``terms``.
        """
        profile_sink = getattr(self, "_profile_sink", None)
        profile_started = time.perf_counter() if profile_sink is not None else None
        results = {}
        try:
            for where, operator in terms.items():
                if isinstance(where, Integral):
                    support = (int(where),)
                else:
                    support = tuple(int(site) for site in where)
                results[where] = self.local_expectation(
                    operator, support, optimize=optimize, normalized=normalized,
                )
            return results
        finally:
            if profile_started is not None:
                profile_sink.append({
                    "kind": "observable_batch",
                    "count": len(terms),
                    "seconds": time.perf_counter() - profile_started,
                })

    def expectation_mpo_exact(
        self, mpo, where, *, normalized=True, optimize="auto",
    ):
        """Contract ``<psi|MPO|psi>`` without applying or compressing the TTN.

        The tree, a private ket view, and the MPO remain separate tensor
        networks. The lower physical legs are connected to fresh copies of
        the ket physical legs, while the upper physical legs connect to the
        bra, and the complete doubled network is contracted. A native
        :class:`TreeMPO` uses its own ``expectation`` method for tree-native
        contraction.

        ``mpo`` must expose Quimb's regular MPO site interface. Its active site
        labels must match ``where``. For a native fermionic TTN the MPO must
        contain native Symmray tensors, so the graded contraction rules remain
        attached to the operator data.
        """
        if isinstance(where, Integral):
            where = (int(where),)
        else:
            where = tuple(int(site) for site in where)
        if not where or len(set(where)) != len(where):
            raise ValueError("where must contain distinct tree sites.")
        if any(site not in self.plan.node_of_qubit for site in where):
            raise ValueError(f"site(s) {where!r} are outside this tree state.")

        if hasattr(mpo, "expectation") and hasattr(mpo, "tree_networks"):
            all_sites = tuple(sorted(self.plan.node_of_qubit))
            if tuple(sorted(where)) != all_sites:
                raise ValueError(
                    "a TreeMPO must be evaluated on all tree sites so its "
                    "identity legs remain explicit."
                )
            return mpo.expectation(
                self,
                normalized=normalized,
                optimize=optimize,
            )

        required = (
            "gen_sites_present", "site_tag", "upper_ind_id", "lower_ind_id",
            "tag_map", "tensor_map", "copy",
        )
        if not all(hasattr(mpo, name) for name in required):
            raise TypeError(
                "expectation_mpo_exact requires a regular Quimb MPO with "
                "site, physical-index, tensor-map, and copy interfaces."
            )
        try:
            present = tuple(mpo.gen_sites_present())
        except Exception as exc:
            raise TypeError(
                "could not inspect the MPO's active site labels for an "
                "exact tree contraction."
            ) from exc
        if set(present) != set(where):
            raise ValueError(
                "MPO active sites must match the declared support: "
                f"MPO has {present!r}, where is {where!r}."
            )

        mpo_native = _is_native_mpo(mpo)
        if mpo_native is not None and bool(mpo_native) != bool(self.fermionic):
            if self.fermionic:
                raise TypeError(
                    "native fermionic TreeTensorNetwork requires a native "
                    "Symmray MPO for exact graded contraction."
                )
            raise TypeError(
                "a native Symmray MPO cannot be exactly contracted with an "
                "ordinary dense TreeTensorNetwork."
            )

        upper_id = mpo.upper_ind_id
        lower_id = mpo.lower_ind_id
        ket = self.copy()
        mpo_work = mpo.copy()
        ket_reindex = {}
        mpo_reindex = {}
        for site in where:
            physical = self.site_ind(site)
            upper = upper_id.format(site)
            lower = lower_id.format(site)
            try:
                tids = tuple(mpo_work.tag_map[mpo_work.site_tag(site)])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"MPO has no unique tensor for active site {site!r}."
                ) from exc
            if len(tids) != 1:
                raise ValueError(
                    f"MPO site {site!r} must resolve to one tensor; "
                    f"got {len(tids)}."
                )
            op_tensor = mpo_work.tensor_map[tids[0]]
            if upper not in op_tensor.inds or lower not in op_tensor.inds:
                raise ValueError(
                    f"MPO site {site!r} does not contain expected physical "
                    f"indices {upper!r} and {lower!r}."
                )
            fresh = qtn.rand_uuid()
            ket_reindex[physical] = fresh
            mpo_reindex[upper] = physical
            mpo_reindex[lower] = fresh

        # This is the key orientation: bra <- MPO upper, MPO lower -> ket.
        # Every physical index then appears exactly twice, while MPO virtual
        # bonds stay internal to the separate structured MPO network.
        ket.reindex_(ket_reindex)
        mpo_work.reindex_(mpo_reindex)
        numerator = (self.H | mpo_work | ket).contract(
            all,
            optimize=optimize,
        )
        if not normalized:
            return numerator
        denominator = (self.H | self).contract(all, optimize=optimize)
        return numerator / denominator

    def expectation_mpo(
        self, mpo, where, *, max_bond=None, cutoff=0.0,
        normalized=True, optimize="auto", warn_on_truncation=True,
        return_diagnostics=False,
    ):
        """Evaluate a structured MPO expectation without changing this TTN.

        This is a convenience wrapper around
        :meth:`TreeOptimizer.expectation_mpo`.  The MPO is routed once over
        the tree, preserving native Symmray blocks and avoiding a dense
        operator on the full support.  The transformed-state bond cap defaults
        to this TTN's current maximum bond; pass ``max_bond`` explicitly when
        a larger measurement workspace is acceptable.
        ``warn_on_truncation=True`` reports when that workspace actually
        truncates the private transformed ket. ``return_diagnostics=True``
        returns the value together with the per-expectation compression report.
        """
        from .optimizer import TreeOptimizer

        current_bond = int(self.max_bond())
        engine = TreeOptimizer(
            None,
            n=self.nqubits,
            tree=self.plan,
            state=self,
            chi=current_bond if max_bond is None else max_bond,
            cutoff=0.0,
            run=False,
        )
        return engine.expectation_mpo(
            mpo,
            where,
            max_bond=max_bond,
            cutoff=cutoff,
            normalized=normalized,
            optimize=optimize,
            warn_on_truncation=warn_on_truncation,
            return_diagnostics=return_diagnostics,
        )

    @property
    def orthogonality_center(self):
        """Node id of the tracked orthogonality centre (``None`` if unknown).

        This is the tree analogue of an MPS canonical centre: when it is a node
        ``c`` every *other* tensor is an isometry whose legs point toward ``c``
        (``absorb="right"`` convention), so the whole state norm collapses onto
        the single centre tensor under Symmray's graded singleton contraction.
        It is the one-node special case of
        :attr:`canonical_region`; it is updated in place by
        :meth:`shift_orthogonality_center` and :meth:`canonize_around_node_`,
        and -- being derived from a field declared in :attr:`_EXTRA_PROPS` -- it
        survives ``.copy()`` and every ``quimb`` view/selection, so any holder
        of the state (the :class:`~pepsy.optimizers.tree.TreeOptimizer`, a
        sampler, a direct user) reads a single consistent centre rather than
        tracking its own.  It reads ``None`` whenever the canonicalised region
        spans more than one node (an honest "no single centre").
        """
        reg = self.canonical_region
        if reg is not None and len(reg) == 1:
            return next(iter(reg))
        return None

    @orthogonality_center.setter
    def orthogonality_center(self, value):
        if value is None:
            self._canonical_region = None
            return
        if value not in self._plan.children:
            raise ValueError(f"{value!r} is not a node of the tree.")
        self._canonical_region = frozenset({value})

    @property
    def canonical_region(self):
        """Frozenset of node ids forming the canonicalised subtree (``None`` if unknown).

        The range / subtree generalisation of :attr:`orthogonality_center`: when
        it is a connected node set ``R`` every tensor *outside* ``R`` is an
        isometry whose legs point inward toward ``R`` (``absorb="right"``
        convention), so the entire state norm is carried by the region tensors
        -- contracting just the region against its graded conjugate gives the
        squared norm, exactly as the single centre tensor does for a one-node
        region.
        It is updated in place by :meth:`canonize_subtree_` (and its qubit-level
        entry point :meth:`canonize_around_qubits_`) and, being declared in
        :attr:`_EXTRA_PROPS`, survives ``.copy()`` and every ``quimb`` view.
        """
        return getattr(self, "_canonical_region", None)

    @canonical_region.setter
    def canonical_region(self, value):
        if value is None:
            self._canonical_region = None
            return
        self._canonical_region = self._validated_region(value)

    def _with_center(self, nid):
        """Set the tracked centre and return ``self`` (builder convenience)."""
        self._canonical_region = frozenset({nid})
        return self

    @property
    def nqubits(self):
        """Number of physical qubits (an alias of :attr:`nsites`)."""
        return self._plan.n

    def node_tag(self, nid):
        """Return the structural tag of node ``nid``."""
        return self._node_tag_id.format(nid)

    def node_tid(self, nid):
        """Return the tensor id of node ``nid`` via a self-healing cache.

        ``quimb`` mints a fresh tensor identity whenever a tensor is rebuilt
        (e.g. ``gate_inds_`` on a leaf), so a stale cache entry simply misses the
        ``tensor_map`` membership check and is recomputed from the tag map.
        The cache lives in ``__dict__`` (not ``_EXTRA_PROPS``) so a freshly
        copied network starts with an empty, independent cache.
        """
        cache = self.__dict__.get("_node_tid_cache")
        if cache is None:
            cache = self.__dict__["_node_tid_cache"] = {}
        tid = cache.get(nid)
        if tid is not None and tid in self.tensor_map:
            return tid
        tid = next(iter(self.tag_map[self.node_tag(nid)]))
        cache[nid] = tid
        return tid

    def node_tensor(self, nid):
        """Return the live :class:`quimb.tensor.Tensor` for node ``nid``."""
        return self.tensor_map[self.node_tid(nid)]

    def bond(self, a, b):
        """Return the live shared virtual-bond index for adjacent nodes."""
        if b not in self.neighbors(a):
            raise ValueError(f"nodes {a} and {b} are not adjacent in the tree.")
        shared = qtn.bonds(self.node_tensor(a), self.node_tensor(b))
        if len(shared) != 1:
            raise ValueError(
                f"nodes {a} and {b} must share exactly one bond; "
                f"found {sorted(shared)}."
            )
        return next(iter(shared))

    def isometry_direction(self, nid):
        """Return the neighbour proven by ``left_inds`` to receive node ``nid``.

        A tree tensor is an isometry toward exactly one adjacent node when its
        ``left_inds`` contain every leg except that shared tree bond. ``None``
        means no usable local proof is currently recorded. This is a derived
        view of the live tensor metadata, not separately tracked state.
        """
        if nid not in self._plan.children:
            raise ValueError(f"{nid!r} is not a node of the tree.")
        tensor = self.node_tensor(nid)
        if tensor.left_inds is None:
            return None
        left_inds = set(tensor.left_inds)
        right_inds = [
            index for index in tensor.inds if index not in left_inds
        ]
        if len(right_inds) != 1:
            return None
        toward_bond = right_inds[0]
        for neighbour in self.neighbors(nid):
            if toward_bond == self.bond(nid, neighbour):
                return neighbour
        return None

    def isometry_map(self):
        """Return ``{node: toward_node_or_None}`` from live ``left_inds``."""
        return {
            nid: self.isometry_direction(nid)
            for nid in self._plan.nodes()
        }

    def _set_isometry_metadata_from_region(self, region):
        """Record orientations for a state already proven canonical.

        This changes metadata only and therefore must be called solely by
        constructors or kernels that independently establish the stated
        canonical region.
        """
        region = self._validated_region(region)
        for nid in self._plan.nodes():
            tensor = self.node_tensor(nid)
            if nid in region:
                tensor.modify(left_inds=None)
                continue
            toward = self._toward_region(nid, region)
            bond = self.bond(nid, toward)
            tensor.modify(
                left_inds=tuple(
                    index for index in tensor.inds if index != bond
                )
            )
        return self

    def can_skip_canonize(self, a, b, *, absorb="right"):
        """Whether local metadata proves edge canonicalisation is redundant.

        With ``absorb="right"`` node ``a`` must already be isometric toward
        ``b``; the ``"left"`` orientation is symmetric. For native fermionic
        tensors the same local proof is accepted only when the live data is a
        Symmray fermionic array with aligned charge maps. This deliberately
        checks structure, not numerical isometry: the metadata is written by
        the native graded QR path, while malformed or unknown metadata falls
        back to explicit graded QR.
        """
        if absorb not in {"right", "left"}:
            raise ValueError("absorb must be 'right' or 'left'.")
        bond = self.bond(a, b)  # validate the requested tree edge
        if absorb == "right":
            node = a
        else:
            node = b
        tensor = self.node_tensor(node)
        if tensor.left_inds is None:
            return False
        if set(tensor.left_inds) != set(tensor.inds) - {bond}:
            return False
        if not self.fermionic:
            return True

        # Native graded arrays carry the phase/charge convention in their
        # index metadata. Do not infer it from dense data or trust arbitrary
        # ``left_inds`` supplied by a caller: only the native array contract
        # is eligible for this QR-free move.
        data = tensor.data
        if not getattr(data, "fermionic", False):
            return False
        indices = getattr(data, "indices", None)
        duals = getattr(data, "duals", None)
        charges = getattr(data, "charges", None)
        if (
            indices is None
            or duals is None
            or charges is None
            or len(indices) != len(tensor.inds)
            or len(duals) != len(tensor.inds)
            or len(charges) != len(tensor.inds)
        ):
            return False
        check_aligned = getattr(data, "check_chargemaps_aligned", None)
        if check_aligned is None:
            return False
        try:
            check_aligned()
        except (AttributeError, TypeError, ValueError, KeyError):
            return False
        return True

    def validate_isometry_metadata(self, region=None):
        """Validate local ``left_inds`` against a canonical region.

        Every tensor outside ``region`` must point along its unique next edge
        toward that region. Tensors inside the region need not be isometric.
        When no explicit or tracked region exists, only malformed non-``None``
        metadata is rejected. Returns ``self`` when valid.
        """
        if region is None:
            region = self.canonical_region
        else:
            region = self._validated_region(region)

        for nid in self._plan.nodes():
            tensor = self.node_tensor(nid)
            direction = self.isometry_direction(nid)
            if tensor.left_inds is not None and direction is None:
                raise ValueError(
                    f"tree node {nid} has left_inds that do not identify "
                    "exactly one adjacent isometry direction."
                )
            if (
                not self.fermionic
                and region is not None
                and nid not in region
            ):
                expected = self._toward_region(nid, region)
                if direction != expected:
                    raise ValueError(
                        f"tree node {nid} must be isometric toward node "
                        f"{expected}, but left_inds point toward {direction}."
                    )
        return self

    def validate(self, *, check_canonical=False, tol=1e-9):
        """Validate the live network against its :class:`TreePlan`.

        The check is intentionally structural by default and therefore cheap
        enough for construction and resource-preflight paths.  It verifies that
        every planned node has exactly one tensor, every planned physical site
        (a leaf or the optional root site) owns exactly one physical index,
        every plan edge has exactly one live bond, and there are no extra
        tensors, outer legs, or malformed shared indices. Pass
        ``check_canonical=True`` to additionally verify the tracked canonical
        region; that part performs tensor contractions and is more expensive.

        Raises
        ------
        ValueError
            If the plan and live tensor network disagree.  The message names
            the first invariant that failed.  Returns ``self`` when valid.
        """
        plan_nodes = tuple(self._plan.nodes())
        plan_node_set = set(plan_nodes)
        node_tids = {}
        for nid in plan_nodes:
            tag = self.node_tag(nid)
            tids = set(self.tag_map.get(tag, ()))
            if len(tids) != 1:
                raise ValueError(
                    f"tree node {nid} must have exactly one tensor tagged "
                    f"{tag!r}; found {len(tids)}."
                )
            node_tids[nid] = next(iter(tids))

        live_tids = set(self.tensor_map)
        if set(node_tids.values()) != live_tids:
            extra = live_tids - set(node_tids.values())
            missing = set(node_tids.values()) - live_tids
            raise ValueError(
                "TreeTensorNetwork tensor set disagrees with TreePlan "
                f"(extra={sorted(extra)!r}, missing={sorted(missing)!r})."
            )

        physical_inds = {self.site_ind(q) for q in range(self._plan.n)}
        owners = {ind: [] for ind in physical_inds}
        for nid, tid in node_tids.items():
            tensor = self.tensor_map[tid]
            node_phys = physical_inds.intersection(tensor.inds)
            q = self._plan.qubit_of_node.get(nid)
            if q is not None:
                expected = self.site_ind(q)
                if expected not in tensor.inds:
                    raise ValueError(
                        f"tree node {nid} (qubit {q}) is missing physical "
                        f"index {expected!r}."
                    )
                if len(node_phys) != 1 or expected not in node_phys:
                    raise ValueError(
                        f"tree node {nid} must own only physical index "
                        f"{expected!r}; found {sorted(node_phys)!r}."
                    )
                if self.site_tag(q) not in tensor.tags:
                    raise ValueError(
                        f"tree node {nid} is missing site tag "
                        f"{self.site_tag(q)!r}."
                    )
            elif node_phys:
                raise ValueError(
                    f"internal node {nid} unexpectedly carries physical "
                    f"indices {sorted(node_phys)!r}."
                )
            for ind in node_phys:
                owners[ind].append(nid)

        for ind, ind_owners in owners.items():
            if len(ind_owners) != 1:
                raise ValueError(
                    f"physical index {ind!r} must belong to one planned node; "
                    f"found nodes {ind_owners!r}."
                )
        unexpected_outer = set(self.outer_inds()) - physical_inds
        if unexpected_outer:
            raise ValueError(
                "live tree has unregistered outer indices "
                f"{sorted(unexpected_outer)!r}; represent a top physical leg "
                "with TreePlan(root_qubit=...)."
            )

        expected_edges = {
            frozenset((parent, child))
            for parent, children in self._plan.children.items()
            for child in children
        }
        edge_bonds = {}
        for parent, children in self._plan.children.items():
            for child in children:
                shared = qtn.bonds(
                    self.tensor_map[node_tids[parent]],
                    self.tensor_map[node_tids[child]],
                )
                if len(shared) != 1:
                    raise ValueError(
                        f"tree edge ({parent}, {child}) must have exactly one "
                        f"live bond; found {sorted(shared)!r}."
                    )
                edge_bonds[frozenset((parent, child))] = next(iter(shared))

        actual_inner = set(self.inner_inds())
        if actual_inner != set(edge_bonds.values()):
            raise ValueError(
                "live internal indices do not match the TreePlan edges "
                f"(extra={sorted(actual_inner - set(edge_bonds.values()))!r}, "
                f"missing={sorted(set(edge_bonds.values()) - actual_inner)!r})."
            )
        tid_to_node = {tid: nid for nid, tid in node_tids.items()}
        for ind in actual_inner:
            tids = set(self.ind_map.get(ind, ()))
            if len(tids) != 2:
                raise ValueError(
                    f"internal index {ind!r} must occur on two tensors; "
                    f"found {len(tids)}."
                )
            owners = frozenset(tid_to_node[tid] for tid in tids)
            if owners not in expected_edges:
                raise ValueError(
                    f"internal index {ind!r} joins non-adjacent nodes "
                    f"{sorted(owners)!r}."
                )
            if int(self.ind_size(ind)) < 1:
                raise ValueError(
                    f"internal index {ind!r} has invalid dimension "
                    f"{self.ind_size(ind)!r}."
                )

        region = self.canonical_region
        if region is not None:
            if not set(region).issubset(plan_node_set):
                raise ValueError(
                    f"canonical region contains unknown nodes "
                    f"{sorted(set(region) - plan_node_set)!r}."
                )
            if self._validated_region(region) != region:
                raise ValueError("canonical region is not a connected subtree.")
            if check_canonical:
                self.validate_isometry_metadata(region)
                if not self.is_subtree_canonical_form(region, tol=tol):
                    raise ValueError(
                        "tracked canonical region failed the isometry check."
                    )
        return self

    # -- plan delegators ------------------------------------------------------

    def is_leaf(self, nid):
        """Whether ``nid`` is a structural leaf (has no children)."""
        return self._plan.is_leaf(nid)

    def parent(self, nid):
        """Return the parent node id of ``nid`` (``None`` at the root)."""
        return self._plan.parent.get(nid)

    def children(self, nid):
        """Return the child node ids of ``nid`` (empty for a leaf)."""
        return self._plan.children[nid]

    def neighbors(self, nid):
        """Return the adjacent node ids of ``nid`` (children plus parent)."""
        nbrs = list(self._plan.children[nid])
        up = self._plan.parent.get(nid)
        if up is not None:
            nbrs.append(up)
        return nbrs

    def node_path(self, a, b):
        """Return the inclusive node-id path from node ``a`` to node ``b``."""
        return self._plan.node_path(a, b)

    def leaf_of_qubit(self, q):
        """Return the leaf node id carrying qubit ``q``."""
        return self._plan.leaf_of_qubit[q]

    def node_of_qubit(self, q):
        """Return the leaf or physical root node carrying qubit ``q``."""
        return self._plan.node_of_qubit[q]

    def qubit_of_leaf(self, nid):
        """Return the qubit label carried by leaf node ``nid``."""
        return self._plan.qubit_of_leaf[nid]

    def qubit_of_node(self, nid):
        """Return the qubit carried by ``nid``, or ``None`` for a virtual node."""
        return self._plan.qubit_of_node.get(nid)

    def tree_distance(self, qa, qb):
        """Return the site-node path length between qubits ``qa`` and ``qb``."""
        return self._plan.tree_distance(qa, qb)

    def steiner_nodes(self, nodes):
        """Return the node set of the minimal subtree spanning ``nodes``.

        The tree has a unique path between any two nodes, so the union of the
        paths from ``nodes[0]`` to every other node is exactly the minimal
        connected subtree (Steiner tree) that contains all of them.
        """
        nodes = list(nodes)
        if not nodes:
            raise ValueError("need at least one node to span a subtree.")
        for node in nodes:
            if node not in self._plan.children:
                raise ValueError(f"{node!r} is not a node of the tree.")
        root_node = nodes[0]
        span = set()
        for node in nodes:
            span.update(self._plan.node_path(root_node, node))
        return span

    def subtree_span(self, nodes):
        """Return the node set of the minimal connected subtree spanning ``nodes``.

        Alias-like generalisation retained for callers that work directly with
        structural nodes: the union of the unique tree paths from ``nodes[0]``
        to every other node is the minimal connected subtree containing them all.
        """
        nodes = list(nodes)
        if not nodes:
            raise ValueError("need at least one node to span a subtree.")
        for nid in nodes:
            if nid not in self._plan.children:
                raise ValueError(f"{nid!r} is not a node of the tree.")
        anchor = nodes[0]
        span = set()
        for nid in nodes:
            span.update(self._plan.node_path(anchor, nid))
        return span

    def _validated_region(self, nodes, *, span=False):
        """Return a frozenset for the canonical region defined by ``nodes``.

        Validates every id is a real node.  With ``span=True`` the result is the
        minimal connected subtree spanning ``nodes`` (via :meth:`subtree_span`);
        with ``span=False`` the ``nodes`` must *already* form a connected
        subtree, else a :class:`ValueError` names the missing linking nodes.
        """
        want = set(nodes)
        if not want:
            raise ValueError("a canonical region needs at least one node.")
        span_set = self.subtree_span(want)
        if span:
            return frozenset(span_set)
        if span_set != want:
            missing = sorted(span_set - want)
            raise ValueError(
                f"nodes {sorted(want)} are not a connected subtree; the minimal "
                f"spanning subtree also needs {missing} (pass span=True to "
                f"include them)."
            )
        return frozenset(want)

    def _toward_region(self, nid, region):
        """Return the neighbour of ``nid`` that steps toward the subtree ``region``."""
        best = None
        for r in region:
            p = self._plan.node_path(nid, r)
            if best is None or len(p) < len(best):
                best = p
        return best[1]

    def _native_qr_options(self, tensor=None):
        """Return the QR options needed by native graded tree tensors.

        Symmray's stabilized QR phase-normalizes every diagonal of ``R``.
        A structural-zero diagonal therefore creates ``0 / |0|`` and can
        produce a NaN in complex64.  Dense tensors retain Quimb's default
        phase convention; network-level canonicalisation uses ``fermionic``
        because it does not expose the individual tensors here.
        """
        native = self.fermionic if tensor is None else _is_symmray_array(
            tensor.data
        )
        return {"stabilized": False} if native else {}

    def _native_qr_split(self, tensor, **kwargs):
        """Perform a QR split with the native graded zero-sector safeguard."""
        return _native_qr_split_tensor(tensor, **kwargs)

    def _record_native_compression_route(
        self, route, *, edge, before_bond, reduction_hint, reduction_proven,
    ):
        """Record which native compression decomposition was used.

        Route records are attached to the existing opt-in profile sink, so
        ordinary replay does not allocate diagnostic dictionaries or perform
        any timing work.  The records deliberately describe decomposition
        selection rather than timing: the surrounding ``edge_compress`` event
        remains the authoritative duration measurement.
        """
        profile_sink = getattr(self, "_profile_sink", None)
        if profile_sink is None:
            return
        profile_sink.append({
            "kind": "native_compression_route",
            "route": route,
            "edge": tuple(edge),
            "before_bond": int(before_bond),
            "reduced": reduction_hint,
            "reduction_proven": bool(reduction_proven),
            "seconds": 0.0,
        })

    # -- edge-level canonical / compression helpers ---------------------------

    def _fermionic_canonize_edge_(self, a, b, absorb):
        """Move a native graded centre across one edge by explicit QR."""
        if absorb == "right":
            isometric_node, reduced_node = a, b
        elif absorb == "left":
            isometric_node, reduced_node = b, a
        else:
            raise ValueError("absorb must be 'right' or 'left'.")

        isometric = self.node_tensor(isometric_node)
        reduced = self.node_tensor(reduced_node)
        bond = self.bond(isometric_node, reduced_node)
        left_inds = [index for index in isometric.inds if index != bond]
        kept, carry = self._native_qr_split(
            isometric,
            left_inds=left_inds,
            right_inds=(bond,),
            absorb="right",
            cutoff=0.0,
            get="tensors",
            bond_ind=self._new_work_bond(
                "canon", isometric_node, reduced_node,
            ),
        )
        merged = _contract_two_tensors(carry, reduced, shared_ind=bond)
        isometric.modify(
            data=kept.data,
            inds=kept.inds,
            left_inds=kept.left_inds,
        )
        reduced.modify(
            data=merged.data,
            inds=merged.inds,
            left_inds=None,
        )
        return self

    def _fermionic_compress_edge_(
        self, a, b, *, max_bond, cutoff, cutoff_mode, absorb, reduced=True,
        reduction_proven=False,
    ):
        """Compress one native graded tree cut with a reduced graded SVD.

        Native Symmray tensors cannot use Quimb's generic compression helper:
        its QR phase convention is not safe for structural zero sectors in
        complex64.  We nevertheless retain the same reduction that makes the
        dense path fast.  A proven one-sided isometry first gets a native QR
        reduction of the active endpoint, then only the small graded core is
        SVD'd; otherwise both endpoints are QR-reduced before the core SVD.
        The complete two-node graded SVD remains the conservative fallback.
        """
        if absorb == "right":
            isometric_node, reduced_node = a, b
        elif absorb == "left":
            isometric_node, reduced_node = b, a
        else:
            raise ValueError("absorb must be 'right' or 'left'.")

        reduction_hint = reduced
        isometric = self.node_tensor(isometric_node)
        reduced_tensor = self.node_tensor(reduced_node)
        bond = self.bond(isometric_node, reduced_node)
        before_bond = int(self.ind_size(bond))
        left_inds = [index for index in isometric.inds if index != bond]

        # ``reduced="left"`` is the metadata value emitted by the optimizer
        # when ``reduced_node`` is already isometric towards ``isometric``.
        # In that case the environment is an exact identity and decomposing
        # the complete two-node tensor is unnecessary.  Validate the proof at
        # this low-level boundary as well, since direct TTN callers can pass
        # arbitrary reduction hints.
        if (
            reduction_hint == "left"
            and (
                reduction_proven
                or self.can_skip_canonize(
                    isometric_node, reduced_node, absorb="left"
                )
            )
        ):
            # The destination endpoint is already an isometry toward the
            # active endpoint, so only the active endpoint needs reducing.
            # QR it first: after fusing its external legs, a central Tree
            # tensor can be thousands by thousands even though its shared
            # bond is only O(chi).  The QR leaves a core whose right dimension
            # is the live bond, avoiding a full SVD of that large matrix.
            reduced_bond = self._new_work_bond(
                "compress_qr", isometric_node, reduced_node,
            )
            isometric_q, isometric_r = self._native_qr_split(
                isometric,
                left_inds=left_inds,
                right_inds=(bond,),
                absorb="right",
                cutoff=0.0,
                get="tensors",
                bond_ind=reduced_bond,
            )
            compressed_bond = self._new_work_bond(
                "compress_svd", isometric_node, reduced_node,
            )
            core_left, core_right = isometric_r.split(
                left_inds=(reduced_bond,),
                method="svd",
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb="right",
                get="tensors",
                bond_ind=compressed_bond,
            )
            kept = _contract_two_tensors(isometric_q, core_left)
            merged = _contract_two_tensors(core_right, reduced_tensor)
            kept.reindex_({compressed_bond: bond})
            merged.reindex_({compressed_bond: bond})
            isometric.modify(
                data=kept.data,
                inds=kept.inds,
                left_inds=left_inds,
            )
            reduced_tensor.modify(
                data=merged.data,
                inds=merged.inds,
                left_inds=None,
            )
            self._record_native_compression_route(
                "one_sided_left",
                edge=(a, b),
                before_bond=before_bond,
                reduction_hint=reduction_hint,
                reduction_proven=reduction_proven,
            )
            return self

        if (
            reduction_hint == "right"
            and (
                reduction_proven
                or self.can_skip_canonize(
                    isometric_node, reduced_node, absorb="right"
                )
            )
        ):
            # Mirror Quimb's ``reduced="right"`` branch: the active endpoint
            # is split directly while the already-left-isometric endpoint is
            # reused. Native SVD handles the charge blocks independently, so
            # no complete two-node contraction is formed.
            right_inds = [
                index for index in reduced_tensor.inds if index != bond
            ]
            compressed_bond = self._new_work_bond(
                "compress_right", isometric_node, reduced_node,
            )
            core_left, core_right = reduced_tensor.split(
                left_inds=(bond,),
                right_inds=right_inds,
                method="svd",
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb="right",
                get="tensors",
                bond_ind=compressed_bond,
            )
            kept = _contract_two_tensors(isometric, core_left)
            kept.reindex_({compressed_bond: bond})
            core_right.reindex_({compressed_bond: bond})
            isometric.modify(
                data=kept.data,
                inds=kept.inds,
                left_inds=left_inds,
            )
            reduced_tensor.modify(
                data=core_right.data,
                inds=core_right.inds,
                left_inds=None,
            )
            self._record_native_compression_route(
                "one_sided_right",
                edge=(a, b),
                before_bond=before_bond,
                reduction_hint=reduction_hint,
                reduction_proven=reduction_proven,
            )
            return self

        if reduction_hint is True:
            # Mirror ``qtn.tensor_compress_bond(reduced=True)`` while routing
            # both QR decompositions through the native policy above.  This
            # keeps the expensive SVD on the reduced core and avoids the
            # O((Dl * d) x (Dr * d)) full two-node matrix in the common case.
            right_inds = [
                index for index in reduced_tensor.inds if index != bond
            ]
            left_bond = self._new_work_bond(
                "compress_left", isometric_node, reduced_node,
            )
            right_bond = self._new_work_bond(
                "compress_right_qr", isometric_node, reduced_node,
            )
            isometric_q, isometric_r = self._native_qr_split(
                isometric,
                left_inds=left_inds,
                right_inds=(bond,),
                absorb="right",
                cutoff=0.0,
                get="tensors",
                bond_ind=left_bond,
            )
            reduced_l, reduced_q = self._native_qr_split(
                reduced_tensor,
                left_inds=(bond,),
                right_inds=right_inds,
                absorb="left",
                cutoff=0.0,
                get="tensors",
                bond_ind=right_bond,
            )
            core = _contract_two_tensors(isometric_r, reduced_l)
            core_left, core_right = core.split(
                left_inds=(left_bond,),
                method="svd",
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb="right",
                get="tensors",
                bond_ind=bond,
            )
            isometric_compressed = _contract_two_tensors(
                isometric_q, core_left,
            )
            reduced_compressed = _contract_two_tensors(
                core_right, reduced_q,
            )
            isometric.modify(
                data=isometric_compressed.data,
                inds=isometric_compressed.inds,
                left_inds=left_inds,
            )
            reduced_tensor.modify(
                data=reduced_compressed.data,
                inds=reduced_compressed.inds,
                left_inds=None,
            )
            self._record_native_compression_route(
                "two_sided_reduced",
                edge=(a, b),
                before_bond=before_bond,
                reduction_hint=reduction_hint,
                reduction_proven=reduction_proven,
            )
            return self

        # Keep the old complete graded split as a compatibility fallback for
        # direct callers that provide an unrecognised reduction hint.
        theta = _contract_two_tensors(isometric, reduced_tensor, shared_ind=bond)
        kept, remainder = theta.split(
            left_inds=left_inds,
            method="svd",
            max_bond=max_bond,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            absorb="right",
            get="tensors",
            bond_ind=bond,
        )
        isometric.modify(
            data=kept.data,
            inds=kept.inds,
            left_inds=kept.left_inds,
        )
        reduced_tensor.modify(
            data=remainder.data,
            inds=remainder.inds,
            left_inds=None,
        )
        self._record_native_compression_route(
            "full_svd_fallback",
            edge=(a, b),
            before_bond=before_bond,
            reduction_hint=reduction_hint,
            reduction_proven=reduction_proven,
        )
        return self

    def _track_edge_center(self, a, b, absorb, *, previous=None):
        """Update the tracked centre after a gauge move across edge ``a -> b``.

        A single ``absorb="right"`` move makes ``a`` isometric and pushes the
        centre onto ``b``; it therefore advances a centre sitting on ``a`` to
        ``b`` (and symmetrically for ``"left"``).  Any other prior centre is no
        longer the global centre after a lone edge move, so it is set to
        ``None`` (unknown) rather than left lying about the canonical form.
        """
        cur = self.orthogonality_center if previous is None else previous
        if absorb == "right" and cur == a:
            self._canonical_region = frozenset({b})
        elif absorb == "left" and cur == b:
            self._canonical_region = frozenset({a})
        else:
            self._canonical_region = None

    def canonize_edge_(
        self, a, b, absorb="right", *, _isometry_proven=False,
    ):
        """Canonicalise across the tree edge ``a -> b`` in place.

        Dense/nonfermionic trees delegate to Quimb's ``canonize_between``;
        native fermionic trees use the explicit graded QR helper above.
        ``absorb="right"`` leaves node ``a`` isometric and pushes the tracked
        orthogonality centre onto node ``b``. Edges whose live metadata proves
        the required isometry are metadata-only moves for both dense and
        native graded arrays.
        """
        if (
            _isometry_proven
            or self.can_skip_canonize(a, b, absorb=absorb)
        ):
            # The local QR is already represented by the tensor's proven
            # ``left_inds``. Keep centre bookkeeping honest, but do not touch
            # tensor data or invalidate the norm cache.
            previous = self.orthogonality_center
            source, target = (
                (a, b) if absorb == "right" else (b, a)
            )
            if previous == source:
                self._canonical_region = frozenset({target})
            elif previous not in (None, target):
                self._canonical_region = None
            return self
        previous = self.orthogonality_center
        if self.fermionic:
            self._fermionic_canonize_edge_(a, b, absorb)
        else:
            self.canonize_between(
                self.node_tag(a),
                self.node_tag(b),
                absorb=absorb,
                method="qr",
                cutoff=0.0,
            )
        self._invalidate_norm_cache()
        self._track_edge_center(a, b, absorb, previous=previous)
        return self

    def compress_edge_(
        self,
        a,
        b,
        *,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        absorb="right",
        reduced=True,
        compression_mode="direct",
        compression_seed=None,
        _reduction_proven=False,
    ):
        """Compress the tree edge ``a -> b`` in place.

        Dense/nonfermionic trees delegate to Quimb's ``compress_between``.
        Native fermionic trees use the same reduced-core decomposition while
        routing every lossless factorization through the zero-sector-safe
        native QR helper. The tracked :attr:`orthogonality_center` advances as
        for :meth:`canonize_edge_`.

        ``cutoff_mode`` selects Quimb's singular-value cutoff convention. The
        default ``"rsum2"`` matches :class:`TreeOptimizer` and Quimb's
        open-boundary MPS gate-application default; use ``"rel"`` explicitly
        for a relative largest-singular-value threshold.
        ``reduced`` selects the dense Quimb reduction and the corresponding
        native graded reduction. Quimb's one-sided ``"left"`` mode is exact
        when node ``b`` is already isometric on its non-shared legs; native
        trees use the same proof, with the zero-sector-safe QR policy retained
        for the two-sided reduced path.
        ``compression_mode="direct"`` uses the standard SVD, while
        ``compression_mode="dm"`` uses Quimb's density-matrix-equivalent
        ``svd:eig`` decomposition on the local canonical core. ``"sdc"``
        and ``"src"`` construct deterministic or random complementary
        environments for this two-node region and then apply QR projectors.
        Their whole-tree versions are available through :meth:`compress`.
        """
        compression_mode = _normalize_compression_mode(compression_mode)
        if compression_mode in {"src", "sdc"}:
            if absorb not in {"left", "right"}:
                raise ValueError("successive edge compression requires left or right absorption")
            hub = b if absorb == "right" else a
            source = a if absorb == "right" else b
            return self._compress_successive_region(
                [(source, hub)], hub, max_bond=max_bond, cutoff=cutoff,
                cutoff_mode=cutoff_mode, method=compression_mode, seed=compression_seed,
            )
        if compression_mode == "dm" and self.fermionic:
            raise NotImplementedError(
                "compression_mode='dm' is currently available for dense "
                "tree tensors only; use compression_mode='direct' for "
                "native fermionic trees."
            )

        bond = self.bond(a, b)
        before_bond = int(self.ind_size(bond))
        if cutoff == 0.0 and (
            max_bond is None or before_bond <= int(max_bond)
        ):
            # No singular value can be removed, so a QR gauge move is exact
            # for both dense and native graded trees. This also protects direct
            # TreeTensorNetwork callers that do not go through the optimizer's
            # diagnostic-aware wrapper.
            return self.canonize_edge_(a, b, absorb=absorb)
        previous = self.orthogonality_center
        if self.fermionic:
            self._fermionic_compress_edge_(
                a,
                b,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb=absorb,
                reduced=reduced,
                reduction_proven=_reduction_proven,
            )
        else:
            self.compress_between(
                self.node_tag(a),
                self.node_tag(b),
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb=absorb,
                reduced=reduced,
                method=_compression_method(compression_mode),
            )
        self._invalidate_norm_cache()
        self._track_edge_center(a, b, absorb, previous=previous)
        return self

    def _compress_successive_region(self, order, hub, *, max_bond, cutoff,
                                    cutoff_mode, method, seed):
        from .compression import successive_tree_compress

        nodes = {hub, *(u for u, _ in order)}
        if self.fermionic:
            raise NotImplementedError(
                f"tree {method} environments require dense tensors; use direct or zipup"
            )
        self.canonize_subtree_(nodes)
        local = {u: [self.tensor_map[self.node_tid(u)].copy()] for u in nodes}
        result, _ = successive_tree_compress(
            local, order, hub, method=method, max_bond=max_bond,
            cutoff=cutoff, cutoff_mode=cutoff_mode, seed=seed,
        )
        for u, tensor in result.items():
            self.tensor_map[self.node_tid(u)].modify(
                data=tensor.data, inds=tensor.inds,
                left_inds=None if u == hub else tensor.left_inds,
            )
        self._canonical_region = frozenset({hub})
        self._invalidate_norm_cache()
        return self

    def compress(
        self,
        *,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode="rsum2",
        center=None,
        reduced=True,
        compression_mode="direct",
        compression_seed=None,
    ):
        """Compress the complete tree with a centre-oriented SVD sweep.

        This is the high-level Tree analogue of an MPS ``compress`` call. The
        tree has no left-to-right direction, so ``center`` selects the final
        canonical node and every edge is compressed from the outer leaves
        inward along its unique geodesic to that node. ``max_bond`` and
        ``cutoff`` are applied by :meth:`compress_edge_` on each edge.

        Canonicalization itself is always lossless QR. A truncating sweep uses
        native edge SVDs and then records the selected node as the canonical
        centre. Thus direct ``TreeTensorNetwork`` users get the same metadata
        guarantees as :class:`TreeOptimizer`, without needing an optimizer or
        a separate ``info_c`` mapping.

        Parameters
        ----------
        max_bond : int, optional
            Maximum virtual bond dimension. ``None`` keeps every retained
            singular direction.
        cutoff : float, optional
            Singular-value cutoff. ``0.0`` requests a lossless QR gauge move
            whenever no finite ``max_bond`` requires an SVD.
        cutoff_mode : str, optional
            Quimb cutoff convention, normally ``"rsum2"`` or ``"rel"``.
        center : int, optional
            TreePlan node at which to leave the final canonical centre. By
            default, the current centre is retained, or the plan root is used
            when no centre is known.
        reduced : bool, optional
            Use the reduced two-sided edge compression path where available.
        compression_mode : {"direct", "dm", "sdc", "src"}, optional
            ``direct``/``dm`` use canonical edge compression. ``sdc``/``src``
            use deterministic/random complementary environments and a
            successive projector sweep on the original target.
        compression_seed : int, optional
            Seed for ``compression_mode="src"``.

        Returns
        -------
        TreeTensorNetwork
            ``self``, after compression and canonical metadata validation.
        """
        if cutoff is None:
            cutoff = 1e-10
        cutoff = float(cutoff)
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative.")
        if max_bond is not None:
            max_bond = int(max_bond)
            if max_bond < 1:
                raise ValueError("max_bond must be at least one.")

        if center is None:
            center = self.orthogonality_center
            if center is None:
                center = self._plan.root
        if center not in self._plan.children:
            raise ValueError(f"{center!r} is not a node of the tree.")

        compression_mode = _normalize_compression_mode(compression_mode)
        if compression_mode in {"src", "sdc"}:
            order = sorted(
                ((u, self._plan.node_path(u, center)[1])
                 for u in self._plan.nodes() if u != center),
                key=lambda edge: -len(self._plan.node_path(edge[0], center)),
            )
            return self._compress_successive_region(
                order, center, max_bond=max_bond, cutoff=cutoff,
                cutoff_mode=cutoff_mode, method=compression_mode, seed=compression_seed,
            )

        # Establish a known centre once. The subsequent post-order traversal
        # then compresses each edge exactly after all of its outward branches
        # have been reduced, so the final sweep has a valid tree-canonical
        # gauge without a second full canonicalization pass.
        self.shift_orthogonality_center(center)
        order = sorted(
            (node for node in self._plan.nodes() if node != center),
            key=lambda node: (
                -len(self._plan.node_path(node, center)),
                int(node),
            ),
        )
        for node in order:
            neighbor = self._plan.node_path(node, center)[1]
            self.compress_edge_(
                node,
                neighbor,
                max_bond=max_bond,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                absorb="right",
                reduced=reduced,
                compression_mode=compression_mode,
                compression_seed=compression_seed,
            )

        # ``compress_edge_`` conservatively clears the global centre when the
        # prior centre is not the source endpoint. The traversal above has
        # nevertheless established the defining outward isometries, so record
        # the proven final region explicitly and verify it before returning.
        self._canonical_region = frozenset({center})
        self._set_isometry_metadata_from_region({center})
        return self.validate(check_canonical=True)

    def canonize_around_node_(self, nid):
        """Canonicalise the whole tree around node ``nid`` and track it as centre.

        Every non-``nid`` tensor becomes an isometry pointing toward ``nid``, so
        the state norm collapses onto the ``nid`` tensor; :attr:`orthogonality_center`
        is set to ``nid``.  This is the O(N) "establish a centre from scratch"
        path -- prefer :meth:`shift_orthogonality_center` for an incremental move
        from a *known* centre.  It is the one-node special case of
        :meth:`canonize_subtree_`.
        """
        if nid not in self._plan.children:
            raise ValueError(f"{nid!r} is not a node of the tree.")
        return self.canonize_subtree_({nid})

    def canonize_subtree_(self, nodes, *, span=False, absorb="right"):
        """Canonicalise the tree around the connected subtree ``nodes`` in place.

        The range / subtree generalisation of :meth:`canonize_around_node_`:
        every tensor *outside* the subtree becomes an isometry pointing inward
        toward it (inward QR gauging via
        :meth:`quimb.tensor.TensorNetwork.canonize_around`), so the entire state
        norm is carried by the subtree tensors -- contracting just the subtree
        against its conjugate reproduces the squared norm.  The tracked
        :attr:`canonical_region` is set to the subtree (and
        :attr:`orthogonality_center` reads that node iff the subtree is a single
        node).

        ``nodes`` must form a connected subtree; pass ``span=True`` to auto-expand
        to the minimal connected subtree that spans them.  Returns ``self``.
        """
        region = self._validated_region(nodes, span=span)
        tags = [self.node_tag(n) for n in region]
        canonize_opts = {
            "method": "qr",
            "cutoff": 0.0,
        }
        canonize_opts.update(self._native_qr_options())
        self.canonize_around_(
            tags,
            which="any",
            absorb=absorb,
            **canonize_opts,
        )
        self._canonical_region = region
        return self

    def canonize_around_qubits_(self, qubits, *, absorb="right"):
        """Canonicalise around the minimal subtree spanning ``qubits`` in place.

        The qubit-level "range canonicalisation" entry point: given a set of
        qubit labels, gauge every tensor outside the minimal connected subtree
        that spans those qubits' physical nodes to point inward, so the reduced state on
        those qubits is captured by that subtree.  Equivalent to
        ``canonize_subtree_(nodes_of(qubits), span=True)``.  Returns ``self``.
        """
        if isinstance(qubits, Integral):
            qubits = (qubits,)
        site_nodes = [self.node_of_qubit(q) for q in qubits]
        return self.canonize_subtree_(site_nodes, span=True, absorb=absorb)

    def _recover_center_from_region(self, region, target, *, absorb="right"):
        """Recover one centre by peeling a tracked canonical region.

        A multi-node canonical region already has every exterior branch
        pointing inward. Peeling the region's leaves toward ``target`` with
        lossless QR therefore establishes a single centre without touching
        tensors outside the region.
        """
        region = set(region)
        if target not in region:
            raise ValueError("target centre must lie inside the canonical region.")
        if absorb not in {"right", "left"}:
            raise ValueError("absorb must be 'right' or 'left'.")

        remaining = set(region)
        while len(remaining) > 1:
            candidates = [
                node for node in remaining
                if node != target
                and sum(
                    neighbour in remaining
                    for neighbour in self.neighbors(node)
                ) == 1
            ]
            if not candidates:
                raise ValueError(
                    "canonical region is not a connected tree containing target."
                )
            # The region is a tree, so any non-target leaf can be peeled.
            # Sorting keeps the QR sequence deterministic across set order.
            node = min(candidates)
            neighbour = next(
                neighbour
                for neighbour in self.neighbors(node)
                if neighbour in remaining
            )
            if absorb == "right":
                a, b = node, neighbour
            else:
                a, b = neighbour, node
            proof = self.can_skip_canonize(a, b, absorb=absorb)
            self.canonize_edge_(
                a, b, absorb=absorb, _isometry_proven=proof,
            )
            remaining.remove(node)

        self._canonical_region = frozenset({target})
        return self

    def shift_orthogonality_center(
        self, new, *, absorb="right", _skip_validate=False,
    ):
        """Move the tracked orthogonality centre to node ``new`` in place.

        The tree analogue of :meth:`quimb.tensor.MatrixProductState.shift_orthogonality_center`:
        the centre is walked to ``new`` along the *unique tree geodesic* from the
        current centre, canonicalising one edge at a time with a lossless QR
        (:meth:`quimb.tensor.TensorNetwork.canonize_between`).  Only the tensors
        on that path are touched, so a nearby move is O(path length), not O(N).

        * If the centre is already ``new`` this is a no-op (idempotent).
        * If the centre is unknown but a multi-node canonical region is tracked,
          only that region is QR-canonicalised first. Otherwise it falls back
          once to the O(N) :meth:`canonize_around_node_`.

        Returns ``self`` so moves can be chained.

        ``_skip_validate`` is an internal hot-path control for local fitting
        engines that validate once after a complete sweep. The default keeps
        the historical validation behavior of this state class.
        """
        del _skip_validate  # TreeTensorNetwork does not validate per movement.
        if new not in self._plan.children:
            raise ValueError(f"{new!r} is not a node of the tree.")
        cur = self.orthogonality_center
        if cur == new:
            return self
        if cur is None:
            region = self.canonical_region
            if region:
                if new in region:
                    return self._recover_center_from_region(
                        region, new, absorb=absorb
                    )
                entry = min(
                    region,
                    key=lambda node: len(self._plan.node_path(node, new)),
                )
                self._recover_center_from_region(
                    region, entry, absorb=absorb
                )
                cur = entry
            else:
                return self.canonize_around_node_(new)
        path = self._plan.node_path(cur, new)
        for u, v in zip(path, path[1:]):
            if absorb == "right":
                a, b = u, v
            elif absorb == "left":
                # ``absorb='left'`` centres the first tag. Reverse the edge
                # orientation so that the target ``v`` receives the factor.
                a, b = v, u
            else:
                raise ValueError("absorb must be 'right' or 'left'.")
            self.canonize_edge_(a, b, absorb=absorb)
        self._canonical_region = frozenset({new})
        return self

    def is_canonical_form(self, center=None, *, tol=1e-9):
        """Return whether the tree is in canonical form about ``center``.

        Checks the defining property directly: every node other than ``center``
        must be an isometry when all its legs *except* the one pointing toward
        ``center`` are treated as inputs (i.e. ``T @ T^dagger`` over those legs is
        the identity on the toward-centre bond).  ``center`` defaults to the
        tracked :attr:`orthogonality_center`; an unknown centre returns ``False``.
        Primarily a diagnostic / test aid; it is the one-node case of
        :meth:`is_subtree_canonical_form`.
        """
        if center is None:
            center = self.orthogonality_center
        if center is None:
            return False
        return self.is_subtree_canonical_form({center}, tol=tol)

    def is_subtree_canonical_form(self, nodes=None, *, span=False, tol=1e-9):
        """Return whether the tree is canonical about the subtree ``nodes``.

        Checks the defining property directly: every tensor *outside* the
        subtree must be an isometry when all its legs *except* the one pointing
        inward toward the subtree are treated as inputs (``T @ T^dagger`` over
        those legs is the identity on the inward bond).  ``nodes`` defaults to
        the tracked :attr:`canonical_region`; an unknown region returns
        ``False``.  Pass ``span=True`` to test against the minimal connected
        subtree spanning ``nodes``.  Primarily a diagnostic / test aid.
        """
        if nodes is None:
            region = self.canonical_region
            if region is None:
                return False
        else:
            region = self._validated_region(nodes, span=span)
        for nid in self._plan.nodes():
            if nid in region:
                continue
            toward = self._toward_region(nid, region)
            t = self.node_tensor(nid)
            bond = next(iter(qtn.bonds(t, self.node_tensor(toward))))
            if self.fermionic:
                # A singleton TensorNetwork.H applies the parity phase flips
                # on all outer legs. The contraction order must also follow
                # the dual orientation of the open bond, as required by
                # Symmray's graded matrix product.
                singleton = qtn.TensorNetwork([t.copy()])
                tc = next(iter(singleton.H.tensor_map.values()))
                tc = tc.reindex({bond: bond + "*"})
                bond_dual = t.data.indices[t.inds.index(bond)].dual
                if bond_dual:
                    output_inds = [bond, bond + "*"]
                    prod = qtn.tensor_contract(t, tc, output_inds=output_inds)
                else:
                    output_inds = [bond + "*", bond]
                    prod = qtn.tensor_contract(tc, t, output_inds=output_inds)
            else:
                tc = t.H.reindex({bond: bond + "*"})
                output_inds = [bond, bond + "*"]
                prod = qtn.tensor_contract(t, tc, output_inds=output_inds)
            d = int(prod.shape[0])
            data = prod.data
            # Keep the diagnostic on the live backend. In particular,
            # ``ar.to_numpy`` cannot move a CUDA tensor to the host, while a
            # scalar reduction can be transferred safely and cheaply.
            if hasattr(data, "to_dense"):
                data = data.to_dense()
            identity = ar.do("eye", d, like=data)
            close = ar.do(
                "allclose",
                data,
                identity,
                atol=float(tol),
                rtol=1.0e-5,
            )
            if not bool(to_float(close, real=True)):
                return False
        return True

    def cap_qubit_(self, q, vec):
        """Contract qubit ``q`` with ``vec`` and remove that site in place.

        This is the tree counterpart of an MPS physical-index cap. The capped
        leaf is absorbed into its parent; if that creates a virtual-only unary
        parent, the parent and its remaining child are fused. A physical root
        qubit is contracted directly on the root without changing the tree
        edges. Remaining qubit labels are compacted above ``q`` so subsequent
        stream entries retain MPS-style positional semantics.
        """
        q = int(q)
        self._invalidate_norm_cache()
        if q not in self._plan.node_of_qubit:
            raise ValueError(f"cap qubit {q} is outside the tree state.")
        if self._plan.n <= 1:
            raise ValueError("cannot cap the only qubit in a tree state.")
        site_node = self._plan.node_of_qubit[q]
        like = self.node_tensor(site_node).data
        try:
            compatible = (
                ar.infer_backend(vec) == ar.infer_backend(like)
                and ar.get_dtype_name(vec) == ar.get_dtype_name(like)
                and str(getattr(vec, "device", None))
                == str(getattr(like, "device", None))
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            compatible = False
        if not compatible:
            vec = ar.do("array", vec, like=like)
        vec = ar.do("reshape", vec, (-1,))
        phys = self.site_ind(q)
        if int(ar.shape(vec)[0]) != int(self.ind_size(phys)):
            raise ValueError(
                f"cap vector length {ar.shape(vec)[0]} does not match the "
                f"physical dimension {self.ind_size(phys)} of qubit {q}."
            )

        if q == self._plan.root_qubit:
            self.shift_orthogonality_center(self._plan.root)
            root_t = self.node_tensor(self._plan.root)
            cap_t = qtn.Tensor(vec, inds=(phys,))
            merged = qtn.tensor_contract(root_t.copy(), cap_t)
            tags = set(root_t.tags)
            tags.discard(self.site_tag(q))
            root_t.modify(data=merged.data, inds=merged.inds, tags=tags)

            old_n = self._plan.n
            temp_inds = {
                old: f"_ttn_cap_ind_{old}" for old in range(q + 1, old_n)
            }
            temp_tags = {
                old: f"_ttn_cap_tag_{old}" for old in range(q + 1, old_n)
            }
            if temp_inds:
                self.reindex_({
                    self.site_ind(old): temp
                    for old, temp in temp_inds.items()
                })
            if temp_tags:
                self.retag_({
                    self.site_tag(old): temp
                    for old, temp in temp_tags.items()
                })
            if temp_inds:
                self.reindex_({
                    temp: self.site_ind(old - 1)
                    for old, temp in temp_inds.items()
                })
            if temp_tags:
                self.retag_({
                    temp: self.site_tag(old - 1)
                    for old, temp in temp_tags.items()
                })

            self._plan = self._plan.remove_qubit(q)
            self._sites = tuple(range(self._plan.n))
            self.__dict__.pop("_node_tid_cache", None)
            self._canonical_region = frozenset({self._plan.root})
            self.validate()
            return self

        leaf = self._plan.leaf_of_qubit[q]
        parent = self._plan.parent.get(leaf)
        if parent is None:
            raise ValueError("cannot cap the root leaf of a multi-qubit tree.")

        # Put the absorbing parent at the centre before contraction, so the
        # remaining state stays canonical without a full-tree rescan.
        self.shift_orthogonality_center(parent)
        self._canonical_region = None
        leaf_t = self.node_tensor(leaf).copy()
        parent_t = self.node_tensor(parent).copy()
        cap_t = qtn.Tensor(vec, inds=(phys,))
        leaf_message = qtn.tensor_contract(leaf_t, cap_t)
        merged = qtn.tensor_contract(parent_t, leaf_message)

        collapse = (
            len(self._plan.children[parent]) == 2
            and not (
                parent == self._plan.root
                and self._plan.root_qubit is not None
            )
        )
        child = None
        tags = set(parent_t.tags)
        if collapse:
            child = next(c for c in self._plan.children[parent] if c != leaf)
            child_t = self.node_tensor(child).copy()
            merged = qtn.tensor_contract(merged, child_t)
            tags.update(set(child_t.tags) - {self.node_tag(child)})
        tags.discard(self.node_tag(leaf))
        tags.add(self.node_tag(parent))

        self.delete(self.node_tag(leaf))
        if child is not None:
            self.delete(self.node_tag(child))
        live_parent = self.node_tensor(parent)
        live_parent.modify(data=merged.data, inds=merged.inds, tags=tags)

        # Compact physical indices and site tags with temporary names to avoid
        # collisions when k(q+1) is renamed to the removed kq.
        old_n = self._plan.n
        temp_inds = {
            old: f"_ttn_cap_ind_{old}" for old in range(q + 1, old_n)
        }
        temp_tags = {
            old: f"_ttn_cap_tag_{old}" for old in range(q + 1, old_n)
        }
        if temp_inds:
            self.reindex_({self.site_ind(old): temp for old, temp in temp_inds.items()})
        if temp_tags:
            self.retag_({self.site_tag(old): temp for old, temp in temp_tags.items()})
        if temp_inds:
            self.reindex_({temp: self.site_ind(old - 1) for old, temp in temp_inds.items()})
        if temp_tags:
            self.retag_({temp: self.site_tag(old - 1) for old, temp in temp_tags.items()})

        self._plan = self._plan.remove_qubit(q)
        self._sites = tuple(range(self._plan.n))
        self.__dict__.pop("_node_tid_cache", None)
        self._canonical_region = frozenset({parent})
        self.validate()
        return self

    # -- dense read-out -------------------------------------------------------

    def to_statevector(self, order=None):
        """Return the dense statevector in qubit order.

        Parameters
        ----------
        order : sequence of int, optional
            Qubit order of the flattened output (default ``range(nqubits)``).
        """
        if order is None:
            order = range(self._plan.n)
        out_inds = [self.site_ind(q) for q in order]
        return ar.to_numpy(self.to_dense(out_inds)).reshape(-1)

    # -- ascii drawing --------------------------------------------------------

    def _bond_dim(self, a, b):
        """Return the virtual-bond dimension of the tree edge ``(a, b)`` (1 if absent)."""
        try:
            ix = self.bond(a, b)
        except ValueError:
            return 1
        return int(self.ind_size(ix)) if ix in self.ind_map else 1

    def ascii_tree(self, *, bond_dims=True, node_ids=False, color=False):
        """Return a top-down ASCII drawing of the tree, drawn root-first.

        The tree analogue of a ``quimb`` MPS ``show``: the root sits at the top
        and structural leaves at the bottom. Physical nodes are labelled
        ``q{q}``; the optional root site appears beside the top marker. When
        ``bond_dims`` is true every edge is annotated with the dimension of the
        virtual bond joining a node to its parent (so growing entanglement
        shows up as growing numbers on the branches)::

                   ●
              ┌────┴────┐
              1         1
              ●         ●
            ┌─┴──┐    ┌─┴──┐
            1    1    1    1
            ◆    ◆    ◆    ◆
            q0   q1   q2   q3

        Parameters
        ----------
        bond_dims : bool
            Annotate each branch with its virtual-bond dimension (default True).
        node_ids : bool
            Also print the structural node id next to each ``●`` (default False).
        color : bool
            Colour the ``●`` markers by tree depth (a per-layer palette), the
            ``◆`` leaves in a distinct colour, and dim the bond numbers and
            connector lines with ANSI SGR codes so each layer reads clearly
            (default False).  The drawing stays width-correct because padding
            ignores the colour escapes.
        """
        plan = self._plan
        gap = 3  # blank columns between sibling subtrees

        def render(nid, depth):
            """Return ``(lines, root_col, width)`` for the subtree rooted at ``nid``."""
            if plan.is_leaf(nid):
                label = f"q{plan.qubit_of_node[nid]}"
                w = max(1, len(label))
                col = (w - 1) // 2
                marker = _color("◆", _LEAF_COLOR, color)
                lbl = _color(label, _LEAF_COLOR, color)
                return [_ascii_place(marker, w, col),
                        _ascii_place(lbl, w, col)], col, w

            dot = f"●{nid}" if node_ids else "●"
            if nid in plan.qubit_of_node:
                dot += f" q{plan.qubit_of_node[nid]}"
            marker = _color(dot, _LAYER_COLORS[depth % len(_LAYER_COLORS)], color)
            blocks = []
            for child in plan.children[nid]:
                lines, col, w = render(child, depth + 1)
                if bond_dims:
                    d = str(self._bond_dim(child, nid))
                    if len(d) > w:  # widen so a fat bond number still fits
                        lp = (len(d) - w) // 2
                        lines = [
                            " " * lp + ln + " " * (len(d) - w - lp)
                            for ln in lines
                        ]
                        col, w = col + lp, len(d)
                    lines = [_ascii_place(_color(d, _DIM_STYLE, color), w, col)] + lines
                blocks.append((lines, col, w))

            # lay the child subtrees side by side, recording each root column
            offsets, cur = [], 0
            for _, _, w in blocks:
                offsets.append(cur)
                cur += w + gap
            total_w = cur - gap
            child_cols = [off + col for off, (_, col, _) in zip(offsets, blocks)]
            pcol = (child_cols[0] + child_cols[-1]) // 2

            # connector row joining the parent tick to each child stem
            conn = [" "] * total_w
            for i in range(child_cols[0], child_cols[-1] + 1):
                conn[i] = "─"
            for j, cc in enumerate(child_cols):
                conn[cc] = (
                    "┌" if j == 0
                    else "┐" if j == len(child_cols) - 1
                    else "┬"
                )
            conn[pcol] = {
                " ": "┴", "─": "┴", "┌": "├", "┐": "┤", "┬": "┼",
            }[conn[pcol]]
            raw = "".join(conn)
            drawn = raw.rstrip()  # keep trailing pad outside the colour escape
            conn_line = _color(drawn, _DIM_STYLE, color) + raw[len(drawn):]

            # stack the child blocks under parent marker + connector rows
            height = max(len(lines) for lines, _, _ in blocks)
            body = []
            for r in range(height):
                row = []
                for k, (lines, _, w) in enumerate(blocks):
                    row.append(lines[r] if r < len(lines) else " " * w)
                    if k != len(blocks) - 1:
                        row.append(" " * gap)
                body.append("".join(row))

            lines = [
                _ascii_place(marker, total_w, pcol),
                conn_line,
            ] + body
            return lines, pcol, total_w

        lines, _, _ = render(plan.root, 0)
        return "\n".join(line.rstrip() for line in lines)

    def show(self, *, bond_dims=True, node_ids=False, color=True):
        """Print the top-down ASCII drawing of the tree (see :meth:`ascii_tree`).

        The tree analogue of a ``quimb`` MPS ``show``: the root is at the top,
        structural leaves are at the bottom, physical nodes are labelled with
        their qubits, and every edge is annotated with its current virtual-bond
        dimension. By default the markers are coloured by tree layer; pass
        ``color=False`` for plain text.
        """
        print(self.ascii_tree(bond_dims=bond_dims, node_ids=node_ids, color=color))


    # -- builders -------------------------------------------------------------

    @classmethod
    def from_plan(cls, plan, *, dtype=complex, phys_dim=2, site_tag_id="I{}",
                  site_ind_id="k{}", node_tag_id="N{}"):
        """Build the product state ``|0...0>`` on the geometry ``plan``.

        Every virtual bond starts at dimension 1, so the state is trivially in
        canonical form (each tensor is an isometry) with the root as the
        orthogonality centre.
        """
        tensors = []
        for nid in plan.nodes():
            inds = []
            shape = []
            tags = [node_tag_id.format(nid)]
            q = plan.qubit_of_node.get(nid)
            if q is not None:
                inds.append(site_ind_id.format(q))
                shape.append(int(phys_dim))
                tags.append(site_tag_id.format(q))
            for child in plan.children[nid]:
                inds.append(_bond_index(nid, child))
                shape.append(1)
            up = plan.parent.get(nid)
            if up is not None:
                inds.append(_bond_index(nid, up))
                shape.append(1)
            if q is not None:
                data = np.zeros(shape, dtype=dtype)
                data[tuple([0] * len(shape))] = 1.0  # |0>
            else:
                data = np.ones(shape, dtype=dtype)
            tensors.append(qtn.Tensor(data, inds=inds, tags=tags))
        ttn = cls(
            tensors,
            plan=plan,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
        )._with_center(plan.root)
        return ttn._set_isometry_metadata_from_region({plan.root}).validate()

    @classmethod
    def from_symmray_plan(
        cls,
        plan,
        *,
        symmetry,
        physical_sectors,
        leaf_charges,
        bond_dim=1,
        fermionic=False,
        seed=None,
        dtype="complex128",
        site_tag_id="I{}",
        site_ind_id="k{}",
        node_tag_id="N{}",
        subsizes="maximal",
    ):
        """Build a native Symmray TTN on ``plan``.

        Symmray supplies the block-sparse / fermionic tensor construction and
        all subsequent QR/SVD/contraction primitives. This method supplies
        only the tree geometry: leaf nodes and the optional physical root
        receive physical legs, while other internal tree nodes remain virtual
        tensors with neutral charge.
        """
        try:
            import symmray as sr
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TreeTensorNetwork.from_symmray_plan requires the optional "
                "dependency 'symmray'."
            ) from exc

        bond_dim = int(bond_dim)
        if bond_dim < 1:
            raise ValueError("bond_dim must be a positive integer.")
        leaf_charges = dict(leaf_charges)
        expected = set(range(plan.n))
        if set(leaf_charges) != expected:
            raise ValueError(
                "leaf_charges must map every qubit label 0 .. n - 1."
            )

        def zero_charge(value):
            if isinstance(value, tuple):
                return tuple(0 for _ in value)
            return 0

        neutral = zero_charge(next(iter(leaf_charges.values())))
        edges = [
            (parent, child)
            for parent, children in plan.children.items()
            for child in children
        ]
        constructor = (
            sr.TN_fermionic_from_edges_rand
            if fermionic
            else sr.TN_abelian_from_edges_rand
        )

        if edges:
            base = constructor(
                symmetry,
                edges,
                bond_dim=bond_dim,
                phys_dim=None,
                seed=seed,
                dtype=dtype,
                site_tag_id=node_tag_id,
                site_charge=lambda nid: (
                    leaf_charges[plan.qubit_of_node[nid]]
                    if nid in plan.qubit_of_node
                    else neutral
                ),
                subsizes=subsizes,
            )
        else:
            base = qtn.TensorNetwork()

        for nid in plan.nodes():
            if edges:
                tensor = base[node_tag_id.format(nid)]
                bond_indices = list(tensor.data.indices)
                bond_duals = list(tensor.data.duals)
                inds = list(tensor.inds)
            else:
                tensor = None
                bond_indices = []
                bond_duals = []
                inds = []

            q = plan.qubit_of_node.get(nid)
            if q is not None:
                physical_index = sr.utils.rand_index(
                    symmetry,
                    physical_sectors,
                    dual=False,
                    subsizes=subsizes,
                    seed=None if seed is None else int(seed) + nid,
                )
                data = sr.utils.get_rand(
                    symmetry,
                    shape=[*bond_indices, physical_index],
                    duals=[*bond_duals, False],
                    charge=leaf_charges[q],
                    fermionic=fermionic,
                    label=nid,
                    seed=None if seed is None else int(seed) + nid,
                    dtype=dtype,
                    subsizes=subsizes,
                )
                tags = (node_tag_id.format(nid), site_tag_id.format(q))
                inds.append(site_ind_id.format(q))
                if tensor is None:
                    base |= qtn.Tensor(data=data, inds=inds, tags=tags)
                else:
                    tensor.modify(data=data, inds=inds, tags=tags)

        ttn = cls(
            base,
            plan=plan,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
            symmetry=symmetry,
            fermionic=fermionic,
            physical_sectors=physical_sectors,
        )
        # The generic Symmray constructor creates a valid charge-conserving
        # tree. Quimb's native block-aware QR establishes the canonical root
        # and accumulates product-state normalization there as well.
        ttn.canonize_around_node_(plan.root)
        return ttn._with_center(plan.root).validate()

    @classmethod
    def from_order(cls, order, *, weights=None, structure="quality",
                   max_arity=2, community_frac=0.35, star_frac=0.75,
                   dtype=complex, site_tag_id="I{}", site_ind_id="k{}",
                   node_tag_id="N{}", root_qubit=None,
                   top_arity=_DEFAULT_TOP_ARITY, map_mode=None):
        """Build a product state on a tree partitioned from ``order``.

        Convenience wrapper that first builds a :class:`TreePlan` with
        :meth:`TreePlan.from_order` and then :meth:`from_plan`.  ``max_arity``
        and ``structure`` control the tree shape (see
        :meth:`TreePlan.from_order`). The default is a binary tree below a
        three-virtual-leg top tensor when there are at least three leaves;
        pass ``top_arity=None`` or ``top_arity=2`` to use a binary root.
        """
        plan = TreePlan.from_order(
            order, weights=weights, structure=structure,
            max_arity=max_arity, community_frac=community_frac,
            star_frac=star_frac, root_qubit=root_qubit,
            top_arity=top_arity,
            map_mode=map_mode,
        )
        return cls.from_plan(
            plan,
            dtype=dtype,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
        )

    @classmethod
    def rand(cls, plan, D=4, *, phys_dim=2, dtype=complex, seed=None,
             canonicalize=True, site_tag_id="I{}", site_ind_id="k{}",
             node_tag_id="N{}"):
        """Build a random tree state with virtual bond dimension ``D``.

        Every virtual (tree-edge) bond is given dimension ``D`` and every leaf a
        physical dimension ``phys_dim``.  Useful for tests and benchmarks.  When
        ``canonicalize`` is true the state is canonicalised around the root.
        """
        rng = np.random.default_rng(seed)
        is_complex = np.iscomplexobj(np.zeros(1, dtype=dtype))

        def _rand(shape):
            arr = rng.standard_normal(shape)
            if is_complex:
                arr = arr + 1j * rng.standard_normal(shape)
            return arr.astype(dtype)

        tensors = []
        for nid in plan.nodes():
            inds = []
            shape = []
            tags = [node_tag_id.format(nid)]
            q = plan.qubit_of_node.get(nid)
            if q is not None:
                inds.append(site_ind_id.format(q))
                shape.append(phys_dim)
                tags.append(site_tag_id.format(q))
            for child in plan.children[nid]:
                inds.append(_bond_index(nid, child))
                shape.append(D)
            up = plan.parent.get(nid)
            if up is not None:
                inds.append(_bond_index(nid, up))
                shape.append(D)
            tensors.append(qtn.Tensor(_rand(shape), inds=inds, tags=tags))
        ttn = cls(
            tensors,
            plan=plan,
            site_tag_id=site_tag_id,
            site_ind_id=site_ind_id,
            node_tag_id=node_tag_id,
        )
        if canonicalize:
            ttn.canonize_around_node_(plan.root)
        return ttn.validate()

    # -- repr -----------------------------------------------------------------

    def __repr__(self):
        return (
            f"{type(self).__name__}(nqubits={self.nqubits}, "
            f"ntensors={self.num_tensors}, max_bond={self.max_bond()})"
        )
