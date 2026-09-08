"""Hamiltonian builders for dense operators and MPOs."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

from numbers import Integral

import autoray as ar
import numpy as np
import quimb
import quimb.tensor as qtn
from .._internal.cutoff import dtype_auto_cutoff
from .._internal.formatting import (
    ansi_wrap,
    coerce_integral_tuple,
    is_integral_tuple,
    is_xy_site,
    is_xy_sublattice_site,
    resolve_color_mode,
)
from ..tensors.core import OneDMap
from ..tensors.bonds import new_native_bond
from ._structural_compression import (
    _structural_compress_mpo,
    _structural_compress_tree,
)
from .mpo_automaton import MPOAutomaton

__all__ = [
    "ham_tn",
]

_DENSE_MPO_COMPRESS_KEYS = frozenset({
    "absorb",
    "bra",
    "bond_ind",
    "get",
    "info",
    "ltags",
    "matrix_svals",
    "method",
    "renorm",
    "right_inds",
    "rtags",
    "stags",
})


class _InheritBackend:
    """Signature-friendly sentinel for inheriting the builder backend."""

    def __repr__(self):
        return "inherit"


_DEFAULT_BACKEND = _InheritBackend()


class _InheritMaxBond:
    """Signature-friendly sentinel for inheriting the builder bond cap."""

    def __repr__(self):
        return "inherit"


_DEFAULT_MAX_BOND = _InheritMaxBond()


class _DefaultTermCompression(str):
    """Signature-friendly sentinel for the term-by-term builder default."""

    def __new__(cls):
        return super().__new__(cls, "term")


_DEFAULT_COMPRESSION = _DefaultTermCompression()


def _normalize_map_mode_name(mode):
    """Normalize a public geometric mapping spelling for tree dispatch."""
    if not isinstance(mode, str):
        raise TypeError("map_mode must be a string or None.")
    return mode.strip().lower().replace("_", "-")


def _tree_base_map_mode(mode):
    """Return the regular ``OneDMap`` mode behind a tree coarse preset."""
    mode = _normalize_map_mode_name(mode)
    if mode.startswith("coarse-"):
        mode = mode[len("coarse-"):]
    return mode


def _normalize_build_mode(mode, *, tree=False):
    """Normalize the public term-building/compression strategy."""
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    mode = mode.strip().lower().replace("-", "_")
    if tree:
        aliases = {
            "term": "term",
            "terms": "term",
            "sequential": "term",
            "analytic": "analytic",
            "automaton": "analytic",
            "automata": "analytic",
            "auto": "auto",
            "direct": "analytic",
            "direct_sum": "analytic",
        }
        try:
            return aliases[mode]
        except KeyError as exc:
            raise ValueError(
                "mode must be 'term' or 'analytic' (aliases: 'auto' and "
                "'automaton')."
            ) from exc
    aliases = {
        "term": "term",
        "terms": "term",
        "sequential": "term",
        "automaton": "automaton",
        "automata": "automaton",
        "analytic": "automaton",
        "auto": "auto",
    }
    return aliases.get(mode, mode)


def _resolve_compression_request(compress, mode=None, *, tree=False):
    """Resolve the canonical ``compress=`` conversion API.

    ``compress`` is normally a boolean.  The string forms are a compact
    strategy spelling: ``compress="term"`` selects sequential terms and
    ``compress="automaton"`` selects the finite-state/analytic route.
    ``mode=`` remains a compatibility alias for callers using the older
    separate strategy keyword.
    """
    if compress is _DEFAULT_COMPRESSION:
        # Keep ``mode=`` usable when the public ``compress`` argument is
        # omitted, while making the new public default explicitly term-wise.
        compress = True
        if mode is None:
            mode = "term"
    elif isinstance(compress, str):
        mode = compress
        compress = True
    elif compress is None:
        compress = True
    elif not isinstance(compress, (bool, np.bool_)):
        raise TypeError(
            "compress must be a bool, None, or the strategy string "
            "'term'/'automaton'/'auto'."
        )
    if mode is None:
        # A boolean is deliberately only the compression switch.  The
        # explicit boolean route remains workload-aware for compatibility.
        mode = "auto"
    return _normalize_build_mode(mode, tree=tree), bool(compress)


def _normalize_optional_max_bond(value, *, name="max_bond"):
    """Normalize an optional operator bond cap.

    ``None`` and ``False`` explicitly disable numerical bond compression.
    The caller uses ``_DEFAULT_MAX_BOND`` when it wants to inherit the
    builder-level cap instead.
    """
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, None, or False.")
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, None, or False.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value


def _resolve_max_bond(value, default):
    """Resolve a conversion bond cap, preserving explicit no-cap values."""
    if value is _DEFAULT_MAX_BOND:
        value = default
    return _normalize_optional_max_bond(value)


_OPERATOR_PROGBAR_COLOR = "#2ca02c"


def _make_operator_progress(progbar, total, *, desc):
    """Create an MPS-style operator-construction progress bar."""
    if not progbar:
        return None
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    return tqdm(
        total=total,
        desc=desc,
        leave=True,
        position=0,
        ascii=True,
        colour=_OPERATOR_PROGBAR_COLOR,
    )


def _operator_max_bond(operator):
    """Read an operator's current maximum virtual bond safely."""
    try:
        return int(operator.max_bond())
    except (AttributeError, TypeError, ValueError):
        return None


def _set_operator_progress_postfix(progress_bar, operator, *, cap, peak=None):
    """Show current/capped ``chi`` and any pre-compression peak."""
    if progress_bar is None:
        return
    current = _operator_max_bond(operator)
    postfix = {
        "chi": (
            f"{current}/{cap}"
            if current is not None and cap is not None
            else current if current is not None else "?"
        ),
    }
    if peak is not None and current is not None and peak > current:
        postfix["peak"] = peak
    progress_bar.set_postfix(postfix)


def _advance_operator_progress(progress_bar, operator, *, cap, peak=None):
    """Update an operator-construction progress bar after one work unit."""
    if progress_bar is None:
        return
    _set_operator_progress_postfix(
        progress_bar,
        operator,
        cap=cap,
        peak=peak,
    )
    progress_bar.update(1)


class ham_tn:
    """Build MPO Hamiltonians from local terms on a mapped lattice.

    Parameters
    ----------
    shape : int or tuple[int, ...], optional
        Lattice shape. An integer or one-element tuple denotes a 1D chain and
        is normalized to ``(L, 1)``; two- and three-element tuples denote 2D
        and 3D lattices. This is an alias for ``Lx``/``Ly``/``Lz`` and cannot
        disagree with dimensions supplied using those names.
    Lx : int, optional
        Number of lattice sites along x. Required with ``Ly`` when ``shape``
        is omitted.
    Ly : int, optional
        Number of lattice sites along y. Use ``Ly=1`` for a 1D chain when
        using the legacy dimension spelling.
    Lz : int | None, default=None
        Optional number of lattice sites along z. When provided, terms can
        use 3D coordinates ``(x, y, z)`` and the 1D mapping is built in 3D.
    max_bond : int | None | False, default=256
        Compression cap used after term additions or final assembly. ``None``
        and ``False`` disable numerical compression for conversions.
    cutoff : float | {"auto"}, default="auto"
        Compression cutoff used by the selected numerical compression sweep.
        ``"auto"`` selects the same dtype-aware cutoff policy as
        :class:`MpsOptimizer`.
    cutoff_mode : str | {"auto"} | None, default=None
        Singular-value cutoff mode used after each term addition. ``"auto"``
        resolves to Pepsy's ordinary ``"rsum2"`` policy. ``None`` preserves
        Quimb's default cutoff mode.
    chi : int | None, default=None
        Compatibility alias for ``max_bond``.
    data_type : str | numpy.dtype | None, default=None
        Dtype used for identity MPO tensors and operators. When omitted,
        ``to_backend`` is probed for its target dtype; without a converter,
        ``float64`` is used.
    to_backend : callable | None, default=None
        Optional array converter, such as ``pepsy.backend_torch(...)`` or
        ``pepsy.backend_jax(...)``. Local MPO tensors are placed on this
        backend before term addition and compression, and are converted once
        more at the return boundary for safety.
    mapper : pepsy.tensors.core.OneDMap | None, default=None
        Optional preconfigured lattice mapper. When omitted, a default
        ``OneDMap(Lx, Ly, Lz=Lz, mode="snake")`` is constructed.
    map_mode : str | None, default=None
        Shorthand for constructing the stored ``OneDMap`` with a named
        traversal. This cannot be combined with ``mapper``.

    Attributes
    ----------
    map : dict[int, tuple[int, int] | tuple[int, int, int]]
        Mapping from 1D chain index to lattice coordinate.
    map_inv : dict[tuple[int, int] | tuple[int, int, int], int]
        Inverse mapping from lattice coordinate to 1D index.
    mapper : pepsy.tensors.core.OneDMap
        Stored mapping helper instance used to build ``map`` and ``map_inv``.
    map_mode : str
        Canonical name of the stored ``OneDMap`` traversal.
    """

    @staticmethod
    def _coalesce_dim_names(*, Lx=None, Ly=None, Lz=None, L_x=None, L_y=None, L_z=None):
        if Lx is None:
            Lx = L_x
        elif L_x is not None and L_x != Lx:
            raise TypeError("Got both Lx and L_x with different values.")

        if Ly is None:
            Ly = L_y
        elif L_y is not None and L_y != Ly:
            raise TypeError("Got both Ly and L_y with different values.")

        if Lz is None:
            Lz = L_z
        elif L_z is not None and L_z != Lz:
            raise TypeError("Got both Lz and L_z with different values.")

        return Lx, Ly, Lz

    @staticmethod
    def _normalize_shape(shape):
        """Normalize the public ``shape`` alias to ``(Lx, Ly, Lz)``."""
        if isinstance(shape, Integral):
            values = (int(shape), 1)
        else:
            try:
                values = tuple(shape)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "shape must be an integer or a 1D/2D/3D shape tuple."
                ) from exc
            if len(values) == 1:
                values = (values[0], 1)
            elif len(values) not in (2, 3):
                raise ValueError(
                    "shape must contain one, two, or three dimensions."
                )
        if not all(isinstance(value, Integral) for value in values):
            raise TypeError("shape must contain only integer dimensions.")
        values = tuple(int(value) for value in values)
        if any(value < 1 for value in values):
            raise ValueError("shape dimensions must be >= 1.")
        if len(values) == 2:
            return values[0], values[1], None
        return values

    @staticmethod
    def _infer_backend_dtype(to_backend):
        """Infer a NumPy dtype from an array converter's target dtype."""
        try:
            sample = to_backend(np.empty(1, dtype=np.float64))
            dtype_name = ar.get_dtype_name(sample)
            return np.dtype(dtype_name)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise TypeError(
                "to_backend must return an array with an inferable dtype; "
                "pass data_type explicitly if the converter does not expose one."
            ) from exc

    def __init__(
        self,
        Lx=None,
        Ly=None,
        Lz=None,
        *,
        shape=None,
        L_x=None,
        L_y=None,
        L_z=None,
        max_bond=256,
        chi=None,
        cutoff="auto",
        cutoff_mode=None,
        data_type=None,
        to_backend=None,
        mapper=None,
        map_mode=None,
    ):
        Lx, Ly, Lz = self._coalesce_dim_names(Lx=Lx, Ly=Ly, Lz=Lz, L_x=L_x, L_y=L_y, L_z=L_z)
        if shape is not None:
            shape_lx, shape_ly, shape_lz = self._normalize_shape(shape)
            supplied = (Lx, Ly, Lz)
            normalized_shape = (shape_lx, shape_ly, shape_lz)
            for name, old, new in zip(("Lx", "Ly", "Lz"), supplied, normalized_shape):
                if old is not None and old != new:
                    raise TypeError(
                        f"shape conflicts with supplied {name}: {shape!r} vs {old!r}."
                    )
            Lx, Ly, Lz = normalized_shape
        if not isinstance(Lx, Integral) or not isinstance(Ly, Integral):
            raise TypeError("shape or both Lx and Ly must be supplied as integers.")
        if Lx < 1 or Ly < 1:
            raise ValueError("Lx and Ly must be >= 1.")
        if Lz is not None:
            if not isinstance(Lz, Integral):
                raise TypeError("Lz must be an integer or None.")
            if int(Lz) < 1:
                raise ValueError("Lz must be >= 1 when provided.")

        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.Lz = None if Lz is None else int(Lz)
        self.L_x, self.L_y, self.L_z = self.Lx, self.Ly, self.Lz
        self.ndim = 2 if self.L_z is None else 3
        self.L = self.L_x * self.L_y if self.L_z is None else self.L_x * self.L_y * self.L_z
        if self.L < 2:
            dims_str = "Lx * Ly" if self.L_z is None else "Lx * Ly * Lz"
            raise ValueError(f"MPO construction requires {dims_str} >= 2.")

        if chi is not None:
            chi_limit = _normalize_optional_max_bond(chi, name="chi")
            if (
                max_bond not in (None, False, 256)
                and _normalize_optional_max_bond(max_bond) != chi_limit
            ):
                raise TypeError("Pass only one of max_bond and chi, or use equal values.")
            max_bond = chi
        max_bond = _normalize_optional_max_bond(max_bond)
        if isinstance(cutoff, str):
            if cutoff.strip().lower() != "auto":
                raise ValueError("cutoff must be 'auto' or a non-negative number.")
            cutoff = "auto"
        else:
            cutoff = float(cutoff)
            if cutoff < 0.0:
                raise ValueError("cutoff must be >= 0.")
        if cutoff_mode is not None and not isinstance(cutoff_mode, str):
            raise TypeError("cutoff_mode must be a string or None.")
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")

        self.max_bond = max_bond
        self.cutoff = cutoff
        self.cutoff_mode = cutoff_mode
        self._data_type_explicit = data_type is not None
        self.data_type = (
            self._infer_backend_dtype(to_backend)
            if data_type is None and to_backend is not None
            else np.dtype("float64") if data_type is None else np.dtype(data_type)
        )
        self.to_backend = to_backend
        if mapper is not None and map_mode is not None:
            raise TypeError("Pass only one of mapper and map_mode.")
        if mapper is None:
            mapper = OneDMap(
                self.Lx,
                self.Ly,
                Lz=self.Lz,
                mode="snake" if map_mode is None else map_mode,
            )
        elif not isinstance(mapper, OneDMap):
            raise TypeError("mapper must be a pepsy.tensors.core.OneDMap instance or None.")

        if mapper.shape != ((self.L_x, self.L_y) if self.L_z is None else (self.L_x, self.L_y, self.L_z)):
            raise ValueError(
                f"mapper shape {mapper.shape} does not match builder shape "
                f"{(self.L_x, self.L_y) if self.L_z is None else (self.L_x, self.L_y, self.L_z)}."
            )

        self.mapper = mapper
        self.map_mode = self.mapper.mode
        self.map, self.map_inv = self.mapper.build()

    @classmethod
    def build_itf_lattice(
        cls,
        *,
        Lx=None,
        Ly=None,
        Lz=None,
        L_x=None,
        L_y=None,
        L_z=None,
        lattice="square",
        edges=None,
        cyclic=False,
        J=1.0,
        field=1.0,
        max_bond=256,
        chi=None,
        cutoff="auto",
        data_type=None,
        to_backend=None,
        mapper=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
        edge_kwargs=None,
        show=False,
        return_edges=True,
        return_mpo=True,
        return_pepo=False,
        return_builder=True,
    ):
        """Construct a builder and ITF Hamiltonian in one call.

        This is a convenience wrapper around ``ham_tn(...).build_itf(...)`` so
        callers can pass lattice size and model parameters directly.

        Parameters
        ----------
        Lx, Ly : int
            Lattice dimensions used to build the internal ``ham_tn`` builder.
        Lz : int | None, default=None
            Optional z dimension. When provided, the builder accepts 3D site
            coordinates ``(x, y, z)`` in direct MPO term construction.
        lattice, edges, cyclic, J, field, compress_each, cycle_peps, cycle_bond_dim, \
        edge_kwargs, show, return_edges, return_mpo, return_pepo
            Forwarded directly to :meth:`build_itf`.
        max_bond, chi, cutoff, data_type
            Used to construct the internal builder instance.
        to_backend : callable | None, default=None
            Optional array converter stored on the internal builder.
        mapper : pepsy.tensors.core.OneDMap | None, default=None
            Optional mapper forwarded to the internal builder. When omitted,
            the default snake-style mapper is used.
        return_mpo : bool, default=True
            If True, include the constructed MPO in the returned payload.
        return_pepo : bool, default=False
            If True, include the constructed PEPO in the returned payload.
            PEPO construction is opt-in and remains restricted to snake-style
            2D mappings.
        return_builder : bool, default=True
            Deprecated compatibility argument. Output is always a dict and
            always includes the constructed builder.

        Returns
        -------
        dict
            Dictionary with named outputs and mappings:
            optional ``mpo``, optional ``pepo``, optional ``edges``/``drawing``,
            optional ``edges_1d`` (when ``edges`` available),
            ``builder``, ``one_d_to_lattice``, and ``lattice_to_one_d``.
            Legacy aliases ``one_d_to_two_d`` and ``two_d_to_one_d`` are also
            provided for compatibility.
        """
        Lx, Ly, Lz = cls._coalesce_dim_names(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            L_x=L_x,
            L_y=L_y,
            L_z=L_z,
        )
        builder = cls(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            max_bond=max_bond,
            chi=chi,
            cutoff=cutoff,
            data_type=data_type,
            to_backend=to_backend,
            mapper=mapper,
        )
        out = builder.build_itf(
            lattice=lattice,
            edges=edges,
            cyclic=cyclic,
            J=J,
            field=field,
            compress_each=compress_each,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            edge_kwargs=edge_kwargs,
            show=show,
            return_edges=return_edges,
            return_mpo=return_mpo,
            return_pepo=return_pepo,
        )
        _ = return_builder  # accepted for backward compatibility
        payload = {
            "mpo": out[0],
            "pepo": out[1],
            "edges": None,
            "edges_1d": None,
            "drawing": None,
            "builder": builder,
            "one_d_to_lattice": dict(builder.map),
            "lattice_to_one_d": dict(builder.map_inv),
            "one_d_to_two_d": dict(builder.map),
            "two_d_to_one_d": dict(builder.map_inv),
        }
        if return_edges and show:
            payload["edges"] = out[2]
            payload["drawing"] = out[3]
        elif return_edges:
            payload["edges"] = out[2]
        elif show:
            payload["drawing"] = out[2]

        if payload["edges"] is not None:
            map_inv = payload["lattice_to_one_d"]
            payload["edges_1d"] = tuple(
                (map_inv[tuple(site0)], map_inv[tuple(site1)])
                for site0, site1 in payload["edges"]
            )
        return payload

    def _coord_dims(self):
        return self.ndim

    def _coord_label(self):
        return "(x, y)" if self.ndim == 2 else "(x, y, z)"

    def _coord_bounds_label(self):
        if self.ndim == 2:
            return f"(Lx={self.Lx}, Ly={self.Ly})"
        return f"(Lx={self.Lx}, Ly={self.Ly}, Lz={self.Lz})"

    def _coerce_coord(self, site):
        try:
            return coerce_integral_tuple(site, length=self._coord_dims(), name="coordinate")
        except TypeError as exc:
            raise TypeError(
                f"Invalid coordinate: {site!r}. Expected {self._coord_label()} for this builder."
            ) from exc

    def map_site(self, site):
        """Map site spec to 1D chain index.

        ``site`` can be either an integer chain index or a coordinate tuple
        ``(x, y)`` (2D) or ``(x, y, z)`` (3D).
        """
        if isinstance(site, Integral):
            index = int(site)
            if index < 0 or index >= self.L:
                raise ValueError(f"Site index {index} is outside [0, {self.L - 1}].")
            return index

        coord = self._coerce_coord(site)
        if coord not in self.map_inv:
            raise ValueError(
                f"Coordinate {coord} is outside lattice bounds {self._coord_bounds_label()}."
            )
        return self.map_inv[coord]

    def _mapper_for_mode(self, map_mode):
        """Build a per-conversion regular mapper without mutating the builder."""
        if map_mode is None:
            return self.mapper
        return OneDMap(
            self.Lx,
            self.Ly,
            Lz=self.Lz,
            mode=map_mode,
        )

    def _tree_mapper_for_mode(self, map_mode):
        """Build the coordinate mapper paired with a tree layout mode."""
        return self._mapper_for_mode(_tree_base_map_mode(map_mode))

    @staticmethod
    def _resolve_cutoff(value, dtype):
        """Resolve a numeric or dtype-aware truncation cutoff."""
        if isinstance(value, str) and value.strip().lower() == "auto":
            return dtype_auto_cutoff(dtype)
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cutoff must be 'auto' or a non-negative number."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("cutoff must be 'auto' or a non-negative number.")
        return value

    @staticmethod
    def _resolve_cutoff_mode(value):
        """Resolve ``cutoff_mode='auto'`` to Pepsy's default convention."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("cutoff_mode must be a string or None.")
        if value.strip().lower() == "auto":
            return "rsum2"
        return value

    def _mapped_chain_edges_2d(self, *, require_local=False):
        self._require_2d("_mapped_chain_edges_2d")
        chain_edges = set()
        for idx in range(self.L - 1):
            site0 = self.map[idx]
            site1 = self.map[idx + 1]
            if abs(site0[0] - site1[0]) + abs(site0[1] - site1[1]) != 1:
                if require_local:
                    raise NotImplementedError(
                        "PEPO conversion requires a 2D mapping whose consecutive chain "
                        "sites remain nearest neighbours on the lattice. "
                        f"mapper.mode={self.map_mode!r} introduces non-local chain steps."
                    )
                continue
            chain_edges.add(frozenset((site0, site1)))
        return chain_edges

    def _require_snake_style_map(self, method_name):
        """Restrict PEPO-style lattice wiring to serpentine 2D traversals."""
        self._require_2d(method_name)
        mode_norm = OneDMap._normalize_mode(self.map_mode)
        if mode_norm not in {"snake", "snake-row-major"}:
            raise NotImplementedError(
                f"{method_name} requires a snake-style 2D mapping. "
                f"Supported PEPO mapper modes are 'snake' and 'snake-row-major'; "
                f"got mapper.mode={self.map_mode!r}."
            )

    @staticmethod
    def _site_tensor(op, site, L):
        if site == 0 or site == L - 1:
            return op[None, :, :]
        return op[None, None, :, :]

    @staticmethod
    def _as_matrix(op):
        data = getattr(op, "data", op)
        if hasattr(data, "to_dense"):
            data = data.to_dense()
        return np.asarray(ar.to_numpy(data))

    def _coerce_op(self, op, *, phys_dim, dtype):
        if callable(op) and not hasattr(op, "shape"):
            op = op()
        if isinstance(op, str):
            label = op.strip().upper()
            if phys_dim != 2 or len(label) != 1 or label not in "IXYZ":
                raise ValueError(
                    "Pauli labels require phys_dim=2 and must be one of I, X, Y, Z."
                )
            if label == "I":
                op = np.eye(2, dtype=dtype)
            else:
                op = quimb.pauli(label, dtype=dtype)
        arr = self._as_matrix(op)
        if arr.shape != (phys_dim, phys_dim):
            raise ValueError(
                f"Operator must have shape ({phys_dim}, {phys_dim}), got {arr.shape}."
            )
        if np.iscomplexobj(arr) and not np.issubdtype(np.dtype(dtype), np.complexfloating):
            if np.allclose(arr.imag, 0.0):
                arr = arr.real
            else:
                raise ValueError(
                    "Complex-valued operator requires complex data_type "
                    f"(got {np.dtype(dtype)})."
                )
        return np.asarray(arr, dtype=dtype)

    @staticmethod
    def _is_coord_site(site, *, n_dims):
        return is_integral_tuple(site, length=n_dims)

    @staticmethod
    def _is_pauli_payload(value):
        """Return whether ``value`` is a compact Pauli label or word."""
        if isinstance(value, str):
            labels = tuple(value)
        elif isinstance(value, (tuple, list)):
            labels = tuple(value)
        else:
            return False
        return bool(labels) and all(
            isinstance(label, str)
            and len(label) == 1
            and label.upper() in "IXYZ"
            for label in labels
        )

    def _parse_term(self, term):
        if not isinstance(term, (tuple, list)):
            raise TypeError(
                "Each term must be tuple/list: (ops, sites[, coeff]), "
                "(sites, paulis, coeff), or ((paulis, coeff), sites)."
            )
        if len(term) not in (2, 3):
            raise ValueError(
                "Each term must be (ops, sites[, coeff]), "
                "(sites, paulis, coeff), or ((paulis, coeff), sites)."
            )

        if (
            len(term) == 2
            and isinstance(term[0], (tuple, list))
            and len(term[0]) == 2
            and self._is_pauli_payload(term[0][0])
            and np.isscalar(term[0][1])
        ):
            # Convenience form: (("ZZ", J), location).
            ops, coeff = term[0]
            sites = term[1]
        elif len(term) == 3 and self._is_pauli_payload(term[1]):
            # Convenience form: (location, "ZZ", J).
            sites, ops, coeff = term
        else:
            # Existing explicit local-operator form: (ops, sites[, coeff]).
            ops, sites = term[0], term[1]
            coeff = term[2] if len(term) == 3 else 1.0

        if not np.isscalar(coeff):
            raise TypeError("coeff must be a scalar.")

        if isinstance(ops, str):
            ops = tuple(ops)
        elif isinstance(ops, np.ndarray):
            ops = (ops,)
        elif isinstance(ops, (tuple, list)):
            ops = tuple(ops)
        elif hasattr(ops, "shape"):
            ops = (ops,)
        else:
            try:
                ops = tuple(ops)
            except TypeError:
                ops = (ops,)

        if isinstance(sites, Integral):
            sites = (int(sites),)
        elif not isinstance(sites, (tuple, list)):
            raise TypeError(
                f"sites must be an integer or tuple/list of {self._coord_label()} coordinates."
            )
        else:
            sites = tuple(sites)

        # Permit a bare 2D/3D coordinate for a one-site Pauli term, while
        # retaining (0, 1) as two 1D chain sites for a two-site term.
        if (
            len(ops) == 1
            and len(sites) == self._coord_dims()
            and all(isinstance(site, Integral) for site in sites)
        ):
            sites = (sites,)

        if len(sites) != len(ops):
            raise ValueError("sites and ops lengths must match.")
        if len(sites) not in (1, 2):
            raise ValueError("Only 1-site and 2-site terms are supported.")
        if not all(
            isinstance(site, Integral)
            or ham_tn._is_coord_site(site, n_dims=self._coord_dims())
            for site in sites
        ):
            raise TypeError(
                f"Sites must be integer chain indices or {self._coord_dims()}D "
                f"coordinates. Use terms like (({self._coord_label()}),) or a "
                "two-site pair."
            )

        return sites, ops, coeff

    def _require_2d(self, method_name):
        if self.ndim != 2:
            raise NotImplementedError(
                f"{method_name} is currently only available for 2D builders "
                f"(initialize ham_tn with Lz=None)."
            )

    def _term_to_mpo(self, term, *, phys_dim, dtype):
        sites, ops, coeff = self._parse_term(term)
        chain_sites = tuple(self.map_site(site) for site in sites)
        if len(set(chain_sites)) != len(chain_sites):
            raise ValueError("Duplicate sites in one term are not supported.")

        mpo_term = self._identity_mpo_with_swapped_phys_inds(
            phys_dim=phys_dim,
            dtype=dtype,
        )

        for n, (site, op) in enumerate(zip(chain_sites, ops)):
            op_arr = self._coerce_op(op, phys_dim=phys_dim, dtype=dtype)
            if n == 0:
                op_arr = coeff * op_arr
            mpo_term[site].modify(data=self._site_tensor(op_arr, site, self.L))

        return mpo_term

    def _swap_mpo_phys_inds_(self, mpo):
        """Swap MPO physical index families from ``(k, b)`` to ``(b, k)``."""
        mpo.reindex_({f"k{i}": f"l{i}" for i in range(self.L)})
        mpo.reindex_({f"b{i}": f"k{i}" for i in range(self.L)})
        mpo.reindex_({f"l{i}": f"b{i}" for i in range(self.L)})
        return mpo

    def _identity_mpo_with_swapped_phys_inds(self, *, phys_dim, dtype):
        """Build identity MPO and immediately swap physical index families."""
        mpo = qtn.MPO_identity(
            self.L,
            phys_dim=phys_dim,
            dtype=dtype,
        )
        self._swap_mpo_phys_inds_(mpo)
        return mpo

    def _zero_mpo(self, *, phys_dim, dtype):
        mpo = self._identity_mpo_with_swapped_phys_inds(
            phys_dim=phys_dim,
            dtype=dtype,
        )
        for tensor in mpo:
            tensor.modify(data=np.zeros_like(tensor.data, dtype=dtype))
        return mpo

    @staticmethod
    def _apply_to_backend(tn, to_backend):
        """Place dense tensor-network arrays on a backend safely."""
        if to_backend is None:
            return tn
        for tensor in tn:
            data = tensor.data
            if isinstance(data, np.ndarray) and not data.flags.writeable:
                tensor.modify(data=np.array(data, copy=True))
        tn.apply_to_arrays(to_backend)
        return tn

    @staticmethod
    def _operator_signature(operator):
        """Return an exact, hashable signature for a dense local operator."""
        array = np.ascontiguousarray(np.asarray(operator))
        return array.shape, array.dtype.str, array.tobytes()

    @staticmethod
    def _is_zero_scalar(value):
        try:
            return bool(value == 0)
        except (TypeError, ValueError):
            return False

    def _normalize_automaton_terms(
        self,
        ints,
        *,
        phys_dim,
        dtype,
    ):
        """Canonicalize local terms before finite-state compilation."""
        onsite = {}
        duplicate_terms = {}
        identity = np.eye(phys_dim, dtype=dtype)

        for term in ints:
            sites, ops, coeff = self._parse_term(term)
            chain_ops = [
                (self.map_site(site), self._coerce_op(op, phys_dim=phys_dim, dtype=dtype))
                for site, op in zip(sites, ops)
            ]
            chain_sites = tuple(site for site, _op in chain_ops)
            if len(set(chain_sites)) != len(chain_sites):
                raise ValueError("Duplicate sites in one term are not supported.")
            chain_ops.sort(key=lambda site_op: site_op[0])
            non_identity = [
                (site, op)
                for site, op in chain_ops
                if not np.array_equal(op, identity)
            ]

            if not non_identity:
                # A product of onsite identities is a global scalar identity.
                onsite[0] = onsite.get(0, np.zeros_like(identity)) + coeff * identity
                continue

            if len(non_identity) == 1:
                site, op = non_identity[0]
                onsite[site] = onsite.get(site, np.zeros_like(identity)) + coeff * op
                continue

            mapped_sites = tuple(site for site, _op in non_identity)
            mapped_ops = tuple(op for _site, op in non_identity)
            key = (
                mapped_sites,
                tuple(self._operator_signature(op) for op in mapped_ops),
            )
            if key in duplicate_terms:
                old_sites, old_ops, old_coeff = duplicate_terms[key]
                duplicate_terms[key] = (old_sites, old_ops, old_coeff + coeff)
            else:
                duplicate_terms[key] = (mapped_sites, mapped_ops, coeff)

        records = []
        for site in sorted(onsite):
            operator = np.asarray(onsite[site], dtype=dtype)
            if not np.array_equal(operator, np.zeros_like(operator)):
                records.append(((site,), (operator,), 1.0))

        for sites, ops, coeff in duplicate_terms.values():
            if not self._is_zero_scalar(coeff):
                records.append((sites, ops, coeff))

        if not records:
            # Keep the zero Hamiltonian representable by the automaton API.
            records.append(((0,), (np.zeros_like(identity),), 1.0))
        return tuple(records)

    def _compile_automaton(self, records, *, phys_dim):
        """Compile canonical local terms into a trimmed shared automaton.

        The structural pass intentionally stays NumPy-backed so that equal
        local operators can be fingerprinted and shared even when the final
        MPO is requested on another array backend.  ``to_backend`` is applied
        immediately after materializing the MPO, before any numerical
        compression.
        """
        automaton = MPOAutomaton.from_product_terms(
            self.L,
            records,
            phys_dim=phys_dim,
            share_channels=True,
        )
        return automaton.trim()

    def _estimate_automaton_bond_dimensions(self, records, *, phys_dim, dtype):
        """Estimate structural automaton bond dimensions without allocating it."""
        identity = np.eye(phys_dim, dtype=dtype)
        prefixes = [set() for _ in range(max(self.L - 1, 0))]

        for sites, ops, _coeff in records:
            if len(sites) < 2:
                continue
            support = dict(zip(sites, ops))
            prefix = []
            for site in range(sites[0], sites[-1]):
                operator = support.get(site, identity)
                prefix.append(self._operator_signature(operator))
                prefixes[site].add((sites[0], tuple(prefix)))

        return tuple(2 + len(cut_prefixes) for cut_prefixes in prefixes)

    def _build_mpo_from_automaton(
        self,
        records,
        *,
        phys_dim,
    ):
        """Compile canonical local terms through the finite-state MPO builder."""
        automaton = self._compile_automaton(
            records,
            phys_dim=phys_dim,
        )
        mpo = automaton.to_mpo()
        self._swap_mpo_phys_inds_(mpo)
        return mpo, automaton

    def to_mpo(
        self,
        ints=None,
        *,
        phys_dim=2,
        max_bond=_DEFAULT_MAX_BOND,
        chi=None,
        cutoff=None,
        data_type=None,
        compress=_DEFAULT_COMPRESSION,
        compress_each=None,
        cutoff_mode=None,
        form=None,
        create_bond=False,
        compress_opts=None,
        mode=None,
        to_backend=_DEFAULT_BACKEND,
        mapper=None,
        map_mode=None,
        fermion=None,
        edges=None,
        fermionic=None,
        charge_sectors=False,
        progbar=False,
        **model_params,
    ):
        """Convert user interactions to a one-dimensional MPO.

        Parameters
        ----------
        ints : sequence | Mapping | pepsy.Fermion | None
            Sequence of terms. Supported term formats:
            - ``((op,), (coord,))``
            - ``((op1, op2), (coord1, coord2))``
            - ``((op,), (coord,), coeff)``
            - ``((op1, op2), (coord1, coord2), coeff)``
            - ``(location, paulis, coeff)`` such as ``((10,), "X", h)``
            - ``((paulis, coeff), location)`` such as ``(("ZZ", J), (10, 11))``
            The existing matrix form uses ``(ops, coords, coeff)``. Pauli
            forms use one label per location and accept integer chain sites or
            lattice coordinates. A bare 2D/3D coordinate is accepted for a
            one-site Pauli term; use ``((x, y),)`` when you want to make the
            support unambiguous.
        phys_dim : int, default=2
            On-site physical dimension.
        max_bond : int | None | False, default=inherit
            MPO compression max bond. An omitted value inherits the builder
            default; ``None`` or ``False`` disables numerical compression.
        chi : int | None | False, default=None
            Alias for ``max_bond``. Supplying both is allowed only when they
            have the same value.
        cutoff : float | {"auto"} | None, default=None
            MPO compression cutoff. Uses instance default when None. ``"auto"``
            selects a dtype-aware cutoff: ``1e-3`` for 16-bit data,
            ``1e-6`` for 32-bit/complex64 data, and ``1e-12`` otherwise.
        data_type : str | numpy.dtype | None, default=None
            Operator/MPO dtype. Uses instance default when None.
        compress : bool | {"term", "automaton", "auto"}, default="term"
            Whether to compress the result. The default ``"term"`` performs
            sequential accumulation and compresses after every added term.
            These numerical compressions run only when an effective bond cap
            is set. ``True`` and ``"auto"`` select the workload-aware
            automatic route for compatibility; ``"automaton"`` forces the
            shared finite-state route. Automaton assembly is compressed once
            after the complete sum.
        compress_each : bool | None, default=None
            Deprecated compatibility spelling for the old API. ``True``
            compresses after each sequential term; ``False`` compresses once
            after the complete sum. Prefer the strategy-bearing
            ``compress=`` spelling.
        cutoff_mode : str | {"auto"} | None, default=None
            Cutoff mode forwarded to Quimb. ``"auto"`` resolves to ``"rsum2"``;
            None preserves Quimb's default.
        form : {None, "left", "right", "flat"} | int, default=None
            Quimb MPO compression form. ``"left"`` is the usual left-to-right
            sweep; an integer selects the orthogonality center.
        create_bond : bool, default=False
            Forwarded to Quimb when compression needs to create an absent bond.
        compress_opts : mapping | None, default=None
            Additional keywords forwarded to Quimb's compression, such as
            ``method``, ``absorb``, ``renorm``, or ``info``. The explicit
            ``chi``, ``max_bond``, ``cutoff``, ``cutoff_mode``, ``form``, and
            ``create_bond`` arguments take precedence.
        mode : str | None, default=None
            Deprecated compatibility alias for selecting the build strategy.
            Prefer ``compress="term"``, ``compress="automaton"``, or
            ``compress="auto"``.
        to_backend : callable | None, default=None
            Optional per-build array converter. When omitted, the converter
            configured on the builder is used; passing ``None`` explicitly
            disables that converter for this build. Generic MPO tensors are
            placed on this backend before addition and compression. Native
            fermion construction forwards it directly to the native builder.
        progbar : bool, default=False
            Show an MPS-style ``tqdm`` progress bar. Term-by-term builds
            advance once per term and report the current ``chi`` together
            with any temporary pre-compression peak bond.
        mapper : pepsy.tensors.core.OneDMap | None, default=None
            Optional mapper override used only for this MPO build. When
            omitted, the builder's configured mapper is used.
        map_mode : str | None, default=None
            One-off ``OneDMap`` traversal override. This is shorthand for
            ``mapper=OneDMap(..., mode=map_mode)`` and cannot be combined
            with ``mapper``.
        fermion : pepsy.Fermion | None, default=None
            Optional native fermion model. When supplied, ``ints`` (or the
            explicit ``edges`` alias) is passed to ``fermion.build_mpo`` and
            the returned Symmray MPO keeps the model's U1/U1U1 symmetry.
        edges : sequence | None, default=None
            Explicit edge alias for the ``fermion=...`` form. For example,
            ``builder.to_mpo(fermion=f, edges=edges, t=..., U=...)``.
        fermionic : bool | None, default=None
            Native graded encoding flag for the fermion-model form. ``None``
            and ``False`` select the Jordan-Wigner-compatible MPO builder;
            ``True`` selects the native graded ``Fermion.build_mpo(...)``.
        charge_sectors : bool, default=False
            When native construction is enabled, return one MPO per operator
            charge as ``{charge: mpo}`` instead of requiring one homogeneous
            charge for the whole collection.
        **model_params
            Explicit fermion couplings such as ``t``, ``U``/``V``, and ``mu``.

        Returns
        -------
        qtn.MatrixProductOperator
            Built Hamiltonian MPO.
        """
        if (
            fermion is None
            and hasattr(ints, "build_mpo")
            and hasattr(ints, "hamiltonian")
        ):
            fermion = ints
            ints = None

        to_backend = self.to_backend if to_backend is _DEFAULT_BACKEND else to_backend
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        if mapper is not None and map_mode is not None:
            raise TypeError("Pass only one of mapper and map_mode.")
        if map_mode is not None:
            mapper = self._mapper_for_mode(map_mode)

        if chi is not None:
            chi_limit = _normalize_optional_max_bond(chi, name="chi")
            if (
                max_bond is not _DEFAULT_MAX_BOND
                and max_bond not in (None, False)
                and _normalize_optional_max_bond(max_bond) != chi_limit
            ):
                raise TypeError("Pass only one of max_bond and chi, or use equal values.")
            max_bond = chi

        if compress_opts is None:
            compress_extra = {}
        elif not isinstance(compress_opts, Mapping):
            raise TypeError("compress_opts must be a mapping or None.")
        else:
            compress_extra = dict(compress_opts)

        if fermion is None:
            for key in tuple(model_params):
                if key not in _DENSE_MPO_COMPRESS_KEYS:
                    continue
                if key in compress_extra:
                    raise TypeError(f"compression option {key!r} supplied twice.")
                compress_extra[key] = model_params.pop(key)

        if "chi" in compress_extra:
            chi_extra = compress_extra.pop("chi")
            chi_extra_limit = _normalize_optional_max_bond(chi_extra, name="chi")
            if (
                chi is not None
                and _normalize_optional_max_bond(chi, name="chi") != chi_extra_limit
            ):
                raise TypeError("conflicting chi values supplied.")
            if (
                max_bond is not _DEFAULT_MAX_BOND
                and max_bond not in (None, False)
                and _normalize_optional_max_bond(max_bond) != chi_extra_limit
            ):
                raise TypeError("conflicting max_bond and chi values supplied.")
            max_bond = chi_extra

        for name in ("max_bond", "cutoff", "cutoff_mode", "form", "create_bond"):
            if name not in compress_extra:
                continue
            value = compress_extra.pop(name)
            current = {
                "max_bond": max_bond,
                "cutoff": cutoff,
                "cutoff_mode": cutoff_mode,
                "form": form,
                "create_bond": create_bond,
            }[name]
            if name == "max_bond":
                if (
                    current is not _DEFAULT_MAX_BOND
                    and current not in (None, False)
                    and _normalize_optional_max_bond(current)
                    != _normalize_optional_max_bond(value)
                ):
                    raise TypeError(f"conflicting {name} values supplied.")
            elif current is not None and current is not False and current != value:
                raise TypeError(f"conflicting {name} values supplied.")
            if name == "max_bond":
                max_bond = value
            elif name == "cutoff":
                cutoff = value
            elif name == "cutoff_mode":
                cutoff_mode = value
            elif name == "form":
                form = value
            else:
                create_bond = value

        legacy_compress_each = compress_each is not None
        if (
            legacy_compress_each
            and compress is not _DEFAULT_COMPRESSION
            and compress not in (None, True)
        ):
            raise TypeError("Pass only one of compress and compress_each.")
        if legacy_compress_each:
            compress_each = bool(compress_each)
            compression_enabled = True
            mode = _normalize_build_mode("term" if mode is None else mode)
        else:
            mode, compression_enabled = _resolve_compression_request(
                compress,
                mode,
            )
            # Automatic and automaton assembly are compressed once, after the
            # complete sum. Only the explicit term strategy is incremental.
            compress_each = compression_enabled and mode == "term"
        if mode not in {"term", "automaton", "auto"}:
            raise ValueError(
                "mode must be 'term', 'automaton', 'analytic', or 'auto'."
            )
        mode_auto = mode == "auto"

        if fermion is not None:
            if mode != "term" or form is not None or create_bond or compress_extra:
                raise TypeError(
                    "mode='automaton', form, create_bond, and extra compress_opts "
                    "are only valid for dense local-term MPO construction."
                )
            if edges is not None:
                if ints is not None:
                    raise TypeError(
                        "Pass fermion terms through either ints or edges, not both."
                    )
                ints = edges
            if ints is None:
                raise ValueError(
                    "Fermion MPO construction requires terms or an edge sequence."
                )
            if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
                raise ValueError("phys_dim must be an integer >= 1.")
            if not hasattr(fermion, "build_mpo"):
                raise TypeError(
                    "fermion must provide the Fermion.build_mpo interface."
                )
            dtype = self.data_type if data_type is None else np.dtype(data_type)
            max_bond_use = _resolve_max_bond(max_bond, self.max_bond)
            compression_active = compression_enabled and max_bond_use is not None
            cutoff_use = self._resolve_cutoff(
                self.cutoff if cutoff is None else cutoff,
                dtype,
            )
            mapper_use = self.mapper if mapper is None else mapper
            fermionic_use = False if fermionic is None else bool(fermionic)
            if charge_sectors and not fermionic_use:
                raise ValueError("charge_sectors=True requires fermionic=True.")
            return fermion.build_mpo(
                ints,
                L=self.L,
                mapper=mapper_use,
                max_bond=max_bond_use,
                cutoff=cutoff_use,
                compress=(
                    compression_active
                    if not legacy_compress_each
                    else compression_active and compress_each
                ),
                dtype=dtype,
                fermionic=fermionic_use,
                charge_sectors=charge_sectors,
                to_backend=to_backend,
                **model_params,
            )

        if edges is not None or model_params or fermionic is not None or charge_sectors:
            raise TypeError(
                "edges, fermion model parameters, and fermionic encoding are "
                "only valid with fermion=... ."
            )
        if ints is None:
            raise ValueError("ints must be provided.")
        if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
            raise ValueError("phys_dim must be an integer >= 1.")

        dtype = (
            self._infer_backend_dtype(to_backend)
            if data_type is None and to_backend is not None and not self._data_type_explicit
            else self.data_type if data_type is None else np.dtype(data_type)
        )
        max_bond = _resolve_max_bond(max_bond, self.max_bond)
        compression_enabled = compression_enabled and max_bond is not None
        if not legacy_compress_each:
            # ``auto`` can become the term route after the structural width
            # estimate, so derive the schedule from the selected route.
            compress_each = compression_enabled and mode == "term"
        else:
            compress_each = compression_enabled and compress_each
        cutoff = self._resolve_cutoff(
            self.cutoff if cutoff is None else cutoff,
            dtype,
        )
        cutoff_mode = self._resolve_cutoff_mode(
            self.cutoff_mode if cutoff_mode is None else cutoff_mode,
        )
        if cutoff < 0.0:
            raise ValueError("cutoff must be >= 0.")

        builder = self
        if mapper is not None:
            builder = ham_tn(
                Lx=self.Lx,
                Ly=self.Ly,
                Lz=self.Lz,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                cutoff_mode=self.cutoff_mode,
                data_type=self.data_type,
                to_backend=to_backend,
                mapper=mapper,
            )

        compress_options = dict(compress_extra)
        compress_options.update({
            "max_bond": max_bond,
            "cutoff": cutoff,
        })
        if cutoff_mode is not None:
            compress_options["cutoff_mode"] = cutoff_mode
        if form is not None:
            compress_options["form"] = form
        if create_bond:
            compress_options["create_bond"] = create_bond

        automaton_records = None
        automaton = None
        structural_applied = False
        progress_bar = None
        if mode in {"automaton", "auto"}:
            automaton_records = builder._normalize_automaton_terms(
                tuple(ints),
                phys_dim=phys_dim,
                dtype=dtype,
            )
            if mode_auto:
                # Avoid materializing a very wide exact automaton when the
                # term-by-term route will be substantially smaller.  The
                # estimate is conservative for repeated prefixes, so common
                # nearest-neighbour structures still select the automaton.
                auto_bond_limit = max(64, 4 * (max_bond or 256))
                estimated_bonds = builder._estimate_automaton_bond_dimensions(
                    automaton_records,
                    phys_dim=phys_dim,
                    dtype=dtype,
                )
                if max(estimated_bonds, default=1) > auto_bond_limit:
                    mode = "term"
                else:
                    mode = "automaton"

            if mode == "automaton":
                progress_bar = _make_operator_progress(
                    progbar,
                    1,
                    desc="mpo",
                )
                mpo_total, automaton = builder._build_mpo_from_automaton(
                    automaton_records,
                    phys_dim=phys_dim,
                )
                if to_backend is not None:
                    builder._apply_to_backend(mpo_total, to_backend)
                elif automaton is not None:
                    # Remove exact boundary dependencies before the optional
                    # numerical sweep. This is the dense analogue of
                    # deparallelization/delinearization and leaves the
                    # public builder API and automaton route unchanged.
                    _structural_compress_mpo(
                        mpo_total,
                        method="auto",
                    )
                    structural_applied = True
                peak_bond = _operator_max_bond(mpo_total)
                if compression_enabled:
                    mpo_total.compress(**compress_options)
                _advance_operator_progress(
                    progress_bar,
                    mpo_total,
                    cap=max_bond,
                    peak=peak_bond,
                )
                if progress_bar is not None:
                    progress_bar.close()

        if mode == "term":
            term_iter = (
                tuple((ops, sites, coeff) for sites, ops, coeff in automaton_records)
                if automaton_records is not None
                else ints
            )
            progress_bar = _make_operator_progress(
                progbar,
                len(term_iter) if hasattr(term_iter, "__len__") else None,
                desc="mpo-term",
            )
            mpo_total = builder._zero_mpo(phys_dim=phys_dim, dtype=dtype)
            if to_backend is not None:
                builder._apply_to_backend(mpo_total, to_backend)
            peak_bond = None
            for term in term_iter:
                mpo_term = builder._term_to_mpo(term, phys_dim=phys_dim, dtype=dtype)
                if to_backend is not None:
                    builder._apply_to_backend(mpo_term, to_backend)
                mpo_total = mpo_total + mpo_term
                peak_bond = _operator_max_bond(mpo_total)
                if compression_enabled and compress_each:
                    if to_backend is None:
                        # Sequential direct sums can recreate parallel
                        # boundary channels at every addition. Reduce them
                        # before paying for this term's numerical SVD.
                        _structural_compress_mpo(
                            mpo_total,
                            method="auto",
                        )
                        structural_applied = True
                    mpo_total.compress(**compress_options)
                _advance_operator_progress(
                    progress_bar,
                    mpo_total,
                    cap=max_bond,
                    peak=peak_bond,
                )

            if compression_enabled and not compress_each:
                if to_backend is None:
                    _structural_compress_mpo(
                        mpo_total,
                        method="auto",
                    )
                    structural_applied = True
                mpo_total.compress(**compress_options)
            _set_operator_progress_postfix(
                progress_bar,
                mpo_total,
                cap=max_bond,
                peak=peak_bond,
            )
            if progress_bar is not None:
                progress_bar.close()
        if to_backend is not None:
            # Keep the return boundary explicit in case a backend operation
            # materialized an intermediate NumPy array.
            builder._apply_to_backend(mpo_total, to_backend)
        elif not structural_applied:
            # The explicit term route can also leave exact dependencies after
            # its additions. This is a no-op for an already reduced network.
            _structural_compress_mpo(mpo_total, method="auto")
        return mpo_total

    def build_mpo(self, ints=None, **kwargs):
        """Compatibility alias for :meth:`to_mpo`.

        ``to_mpo`` describes the conversion boundary more accurately and is
        the canonical spelling for new code.  Keep this wrapper rather than a
        direct assignment so callers receive a useful migration warning.
        """
        warnings.warn(
            "ham_tn.build_mpo is deprecated; use ham_tn.to_mpo instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_mpo(ints, **kwargs)

    def _add_missing_lattice_bonds_(self, pepo):
        """Add rank-1 bonds for lattice neighbours not already used by the 1D path."""
        self._require_2d("_add_missing_lattice_bonds_")
        chain_edges = self._mapped_chain_edges_2d(require_local=True)

        for x in range(self.L_x):
            for y in range(self.L_y):
                if x + 1 < self.L_x:
                    edge = frozenset(((x, y), (x + 1, y)))
                    if edge not in chain_edges:
                        new_native_bond(
                            pepo[f"I{x},{y}"],
                            pepo[f"I{x + 1},{y}"],
                            size=1,
                        )
                if y + 1 < self.L_y:
                    edge = frozenset(((x, y), (x, y + 1)))
                    if edge not in chain_edges:
                        new_native_bond(
                            pepo[f"I{x},{y}"],
                            pepo[f"I{x},{y + 1}"],
                            size=1,
                        )
        return pepo

    def _add_cycle_bonds_(self, pepo, *, bond_dim=1):
        """Optionally add periodic bonds in x and y directions."""
        self._require_2d("_add_cycle_bonds_")
        if not isinstance(bond_dim, Integral) or bond_dim < 1:
            raise ValueError("bond_dim must be an integer >= 1.")

        if self.L_x > 1:
            for y in range(self.L_y):
                left = pepo[f"I{self.L_x - 1},{y}"]
                right = pepo[f"I0,{y}"]
                if not qtn.bonds(left, right):
                    new_native_bond(left, right, size=int(bond_dim))

        if self.L_y > 1:
            for x in range(self.L_x):
                top = pepo[f"I{x},{self.L_y - 1}"]
                bottom = pepo[f"I{x},0"]
                if not qtn.bonds(top, bottom):
                    new_native_bond(top, bottom, size=int(bond_dim))

        return pepo

    def mpo_to_pepo(
        self,
        mpo,
        *,
        cycle_peps=False,
        cycle_bond_dim=1,
        inplace=False,
    ):
        """Convert a snake-style ordered MPO into a 2D PEPO with lattice tags/indices.

        Parameters
        ----------
        mpo : qtn.MatrixProductOperator
            Input MPO with chain length ``L_x * L_y``.
            PEPO conversion is currently restricted to snake-style 2D maps:
            ``"snake"`` and ``"snake-row-major"``.
        cycle_peps : bool, default=False
            If True, add periodic bonds along x and y boundaries.
        cycle_bond_dim : int, default=1
            Bond dimension used when ``cycle_peps=True``.
        inplace : bool, default=False
            If True, modify ``mpo`` in place.

        Returns
        -------
        qtn.PEPO
            Converted PEPO object with site tags ``I{x},{y}`` and physical
            index ids ``k{x},{y}``, ``b{x},{y}``.
        """
        self._require_snake_style_map("mpo_to_pepo")
        if getattr(mpo, "L", None) != self.L:
            raise ValueError(
                f"MPO length mismatch: expected {self.L}, got {getattr(mpo, 'L', None)}."
            )

        pepo = mpo if inplace else mpo.copy()

        for chain_idx, tensor in enumerate(pepo):
            x, y = self.map[chain_idx]
            tensor.modify(tags=[f"I{x},{y}", f"X{x}", f"Y{y}"])
            upper_ind = pepo.upper_ind(chain_idx)
            lower_ind = pepo.lower_ind(chain_idx)
            tensor.reindex_(
                {
                    upper_ind: f"k{x},{y}",
                    lower_ind: f"b{x},{y}",
                }
            )

        self._add_missing_lattice_bonds_(pepo)

        pepo.view_as_(
            qtn.PEPO,
            Lx=self.L_x,
            Ly=self.L_y,
            site_tag_id="I{},{}",
            x_tag_id="X{}",
            y_tag_id="Y{}",
            upper_ind_id="k{},{}",
            lower_ind_id="b{},{}",
        )

        if cycle_peps:
            self._add_cycle_bonds_(pepo, bond_dim=cycle_bond_dim)

        return pepo

    def to_pepo(
        self,
        ints=None,
        *,
        phys_dim=2,
        max_bond=_DEFAULT_MAX_BOND,
        cutoff=None,
        data_type=None,
        compress=_DEFAULT_COMPRESSION,
        compress_each=None,
        mode=None,
        cycle_peps=False,
        cycle_bond_dim=1,
        mapper=None,
        map_mode=None,
        fermion=None,
        edges=None,
        fermionic=None,
        charge_sectors=False,
        to_backend=_DEFAULT_BACKEND,
        progbar=False,
        **model_params,
    ):
        """Convert interactions or a native fermion model to a PEPO.

        The ``fermion=...``/``edges=...`` form mirrors :meth:`to_mpo` and
        forwards ``mapper=OneDMap(...)`` and ``fermionic=True`` to the native
        fermion MPO builder before converting the result to a PEPO.
        With ``charge_sectors=True``, return ``{charge: pepo}`` for a mixed
        native operator. ``map_mode`` is a shorthand for a one-off regular
        ``OneDMap`` override and cannot be combined with ``mapper``. The
        per-call ``to_backend`` override follows :meth:`to_mpo`, including
        explicit ``None`` to keep the result on NumPy/Symmray arrays.
        ``compress`` and ``mode`` follow :meth:`to_mpo`; omitted ``compress``
        defaults to ``"term"``. An omitted ``max_bond`` inherits the builder
        cap; ``max_bond=None`` or ``False`` disables numerical compression.
        The compatibility spelling
        ``compress_each=`` remains accepted but is not used by new code.
        ``progbar=True`` forwards the MPS-style construction progress bar to
        :meth:`to_mpo`.
        """
        self._require_2d("to_pepo")
        mpo = self.to_mpo(
            ints,
            phys_dim=phys_dim,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            compress=compress,
            compress_each=compress_each,
            mode=mode,
            progbar=progbar,
            mapper=mapper,
            map_mode=map_mode,
            fermion=fermion,
            edges=edges,
            fermionic=fermionic,
            charge_sectors=charge_sectors,
            to_backend=to_backend,
            **model_params,
        )
        if isinstance(mpo, Mapping):
            return {
                charge: self.mpo_to_pepo(
                    sector_mpo,
                    cycle_peps=cycle_peps,
                    cycle_bond_dim=cycle_bond_dim,
                    inplace=False,
                )
                for charge, sector_mpo in mpo.items()
            }
        return self.mpo_to_pepo(
            mpo,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            inplace=False,
        )

    def build_pepo(self, ints=None, **kwargs):
        """Compatibility alias for :meth:`to_pepo`."""
        warnings.warn(
            "ham_tn.build_pepo is deprecated; use ham_tn.to_pepo instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_pepo(ints, **kwargs)

    def _tree_site(self, plan, site, *, mapper=None):
        """Resolve a Hamiltonian site against a native tree plan."""
        if hasattr(plan, "coord_to_one_d"):
            # TreePepsPlan owns the coordinate-to-logical-site mapping.  Do
            # not silently replace it with this builder's OneDMap ordering.
            resolved = plan.resolve_site(site)
        else:
            # TreePlan sites are logical qubit ids.  Coordinate terms use the
            # conversion mapper because a plain TreePlan has no lattice map.
            mapper = self.mapper if mapper is None else mapper
            if isinstance(site, Integral):
                resolved = int(site)
            else:
                coord = self._coerce_coord(site)
                map_inv = self.map_inv if mapper is self.mapper else mapper.build()[1]
                if coord not in map_inv:
                    raise ValueError(
                        f"Coordinate {coord} is outside lattice bounds "
                        f"{self._coord_bounds_label()}."
                    )
                resolved = map_inv[coord]
        if resolved < 0 or resolved >= self.L:
            raise ValueError(
                f"tree site {resolved} is outside the builder range [0, {self.L - 1}]."
            )
        return int(resolved)

    def _normalize_tree_terms(self, plan, ints, *, phys_dim, dtype, mapper=None):
        """Convert the public Hamiltonian term spellings to dense tree terms."""
        if isinstance(ints, Mapping):
            normalized = {}
            for support, operator in ints.items():
                raw_support = (support,) if isinstance(support, Integral) else tuple(support)
                mapped_support = tuple(
                    self._tree_site(plan, site, mapper=mapper) for site in raw_support
                )
                if len(set(mapped_support)) != len(mapped_support):
                    raise ValueError("Duplicate sites in one tree term are not supported.")
                array = self._as_matrix(operator)
                matrix_shape = (phys_dim ** len(mapped_support),) * 2
                tensor_shape = (phys_dim,) * (2 * len(mapped_support))
                if array.shape == matrix_shape and len(mapped_support) > 1:
                    array = array.reshape(tensor_shape)
                elif array.shape not in {
                    matrix_shape,
                    tensor_shape,
                }:
                    raise ValueError(
                        f"tree term on support {mapped_support!r} has shape {array.shape}, "
                        f"expected {matrix_shape} or {tensor_shape}."
                    )
                key = mapped_support[0] if len(mapped_support) == 1 else mapped_support
                normalized[key] = (
                    np.asarray(array, dtype=dtype)
                    if key not in normalized
                    else normalized[key] + np.asarray(array, dtype=dtype)
                )
            if not normalized:
                raise ValueError("tree term mapping must not be empty.")
            return normalized

        try:
            terms = tuple(ints)
        except TypeError as exc:
            raise TypeError("tree terms must be a sequence or mapping.") from exc
        if not terms:
            raise ValueError("tree terms must not be empty.")

        normalized = {}
        for term in terms:
            sites, operators, coeff = self._parse_term(term)
            mapped_support = tuple(
                self._tree_site(plan, site, mapper=mapper) for site in sites
            )
            if len(set(mapped_support)) != len(mapped_support):
                raise ValueError("Duplicate sites in one tree term are not supported.")
            local_ops = tuple(
                self._coerce_op(operator, phys_dim=phys_dim, dtype=dtype)
                for operator in operators
            )
            dense = local_ops[0]
            for operator in local_ops[1:]:
                dense = np.kron(dense, operator)
            dense = coeff * dense
            if np.iscomplexobj(dense) and not np.issubdtype(
                np.dtype(dtype), np.complexfloating
            ) and not np.allclose(np.imag(dense), 0.0):
                raise ValueError(
                    "Complex-valued tree term requires complex data_type "
                    f"(got {np.dtype(dtype)})."
                )
            if len(mapped_support) > 1:
                dense = dense.reshape((phys_dim,) * (2 * len(mapped_support)))
            dense = np.asarray(dense, dtype=dtype)
            key = mapped_support[0] if len(mapped_support) == 1 else mapped_support
            normalized[key] = (
                dense if key not in normalized else normalized[key] + dense
            )
        return normalized

    @staticmethod
    def _apply_tree_backend(operator, to_backend):
        """Convert all native tree-operator arrays to the requested backend."""
        if to_backend is None:
            return operator
        networks = getattr(operator, "tree_networks", (operator,))
        for network in networks:
            network.apply_to_arrays(to_backend)
        return operator

    def _validate_tree_plan(self, plan, *, tree_peps=False):
        """Validate a tree plan against this builder's site count and shape."""
        expected_size = self.L
        actual_size = getattr(plan, "size", None) if tree_peps else getattr(plan, "n", None)
        if actual_size != expected_size:
            name = "TreePepsPlan" if tree_peps else "TreePlan"
            raise ValueError(
                f"{name} has {actual_size} physical sites, expected {expected_size}."
            )
        if tree_peps:
            expected_shape = (self.Lx, self.Ly) if self.Lz is None else (
                self.Lx, self.Ly, self.Lz
            )
            if tuple(plan.shape) != expected_shape:
                raise ValueError(
                    f"TreePepsPlan shape {tuple(plan.shape)!r} does not match "
                    f"builder shape {expected_shape!r}."
                )
        return plan

    def _tree_plan_from_map_mode(self, map_mode):
        """Build a default binary TreePlan for a geometric map mode."""
        from ..optimizers.tree.layout import TreeLayoutFinder, TreePlan

        mode = self.map_mode if map_mode is None else map_mode
        mapper = self._tree_mapper_for_mode(mode)
        _, map_inv = mapper.build()

        def site(*coordinate):
            return map_inv[tuple(coordinate)]

        if self.Lz is None:
            order = TreeLayoutFinder.lattice_order(
                self.Lx,
                self.Ly,
                mode=mode,
                site=site,
            )
        else:
            order = TreeLayoutFinder.lattice_order(
                self.Lx,
                self.Ly,
                Lz=self.Lz,
                mode=mode,
                site=site,
            )
        # TreePlan's defaults are the intended simple path: quality layout,
        # binary internal nodes, and the standard top arity.
        mode_name = _normalize_map_mode_name(mode)
        tree_map_mode = mode_name if mode_name.startswith("coarse-") else None
        return TreePlan.from_order(order, map_mode=tree_map_mode), mapper

    def _raw_tree_term_supports(self, ints):
        """Return public term supports without touching their operator data."""
        if isinstance(ints, Mapping):
            entries = ints.keys()
            supports = []
            for support in entries:
                support = (support,) if isinstance(support, Integral) else tuple(support)
                if not support:
                    raise ValueError("tree term supports must not be empty.")
                supports.append(support)
            return tuple(supports)
        try:
            return tuple(
                tuple(sites)
                for sites, _ops, _coeff in (
                    self._parse_term(term) for term in tuple(ints)
                )
            )
        except TypeError as exc:
            raise TypeError("tree terms must be a sequence or mapping.") from exc

    def _auto_tree_mpo_plan(self, ints, *, max_bond):
        """Choose a TreePlan from the operator support hypergraph.

        This is a bounded workload-aware search, not a claim of global
        optimality.  The selected finder is retained by ``TreeMPO`` so the
        layout can be inspected and reused by later tree algorithms.
        """
        from ..optimizers.tree.layout import TreeLayoutFinder

        base_plan, mapper = self._tree_plan_from_map_mode(None)
        supports = tuple(
            tuple(self._tree_site(base_plan, site, mapper=mapper) for site in support)
            for support in self._raw_tree_term_supports(ints)
        )
        shape = (self.Lx, self.Ly) if self.Lz is None else (
            self.Lx, self.Ly, self.Lz
        )
        map_inv = mapper.build()[1]

        def lattice_site(*coordinate):
            return map_inv[tuple(coordinate)]

        finder = TreeLayoutFinder(
            supports=supports,
            n=self.L,
            objective="hypergraph",
            max_arity=(2, 3, 4),
            chi=max_bond,
            lattice_shape=shape,
            lattice_site=lattice_site,
        )
        plan = finder.run(
            chi=max_bond,
            refine=None,
            topology_refine=None,
            search=None,
        )
        return plan, mapper, finder

    def _tree_layout_finder_for_operator(self, plan, supports, *, mapper):
        """Build display/search metadata matching a TreeMPO's lattice labels."""
        from ..optimizers.tree.layout import TreeLayoutFinder

        map_inv = mapper.build()[1]

        def lattice_site(*coordinate):
            return map_inv[tuple(coordinate)]

        shape = (self.Lx, self.Ly) if self.Lz is None else (
            self.Lx, self.Ly, self.Lz
        )
        order = plan.map_mode or mapper.mode
        return TreeLayoutFinder(
            supports=tuple(supports),
            n=plan.n,
            order=order,
            lattice_shape=shape,
            lattice_site=lattice_site,
            root_qubit=getattr(plan, "root_qubit", None),
        )

    def _tree_peps_layout_finder_for_operator(self, plan, supports):
        """Build layout metadata matching a native ``TreePepo`` plan."""
        from ..optimizers.tree_peps.layout import TreePepsLayoutFinder

        return TreePepsLayoutFinder(
            plan,
            supports=tuple(supports),
            max_virtual_degree=plan.max_virtual_degree,
            tree_order=plan.tree_order,
        )

    def _auto_tree_pepo_plan(self, ints, *, max_bond):
        """Choose a TreePepsPlan from the term-support workload."""
        from ..optimizers.tree_peps.layout import TreePepsLayoutFinder
        from ..optimizers.tree_peps.plan import TreePepsPlan

        shape = (self.Lx, self.Ly) if self.Lz is None else (
            self.Lx, self.Ly, self.Lz
        )
        # The builder's map remains the logical coordinate convention.  The
        # finder is then free to change only the retained physical tree.
        source_plan = TreePepsPlan.from_shape(shape, order=self.map_mode)
        supports = tuple(
            tuple(source_plan.resolve_site(site) for site in support)
            for support in self._raw_tree_term_supports(ints)
        )
        finder = TreePepsLayoutFinder(
            source_plan,
            supports=supports,
            objective="hybrid",
            max_virtual_degree=None,
            # Keep the implicit conversion responsive.  The public finder
            # still exposes larger ``max_iter``/refinement budgets when a
            # deeper layout search is wanted.
            max_iter=min(16, max(8, (self.L - 1) // 2)),
        )
        plan = finder.run(refine=True)
        return plan, finder

    def to_tree_mpo(
        self,
        plan=None,
        ints=None,
        *,
        map_mode=None,
        phys_dim=2,
        max_bond=_DEFAULT_MAX_BOND,
        chi=None,
        cutoff=None,
        data_type=None,
        mode=None,
        compress=_DEFAULT_COMPRESSION,
        compress_opts=None,
        compress_order="rank",
        fermion=None,
        edges=None,
        fermionic=None,
        charge_sectors=False,
        to_backend=_DEFAULT_BACKEND,
        progbar=False,
        **model_params,
    ):
        """Convert interactions directly to a native :class:`TreeMPO`.

        ``plan`` owns the tree geometry when supplied. Otherwise ``plan`` may
        be omitted and the first positional argument is interpreted as the
        terms; the automatic route chooses a workload-aware ``TreePlan`` from
        their support graph, while explicit ``map_mode`` builds a default
        binary plan from a geometric traversal. Ordinary dense terms are
        factorized directly
        over their TreePlan Steiner subtrees; no chain MPO is built or
        attached. When ``fermion`` is supplied, its native
        ``build_tree_operator`` route is used instead.

        ``compress="term"`` (the default) builds one native operator per term
        and compresses after every addition. ``True`` and ``compress="auto"``
        select the workload-aware native state-diagram route for compatibility.
        Automaton assembly is compressed once after the complete term
        collection; term assembly is compressed after every added term. These
        numerical compressions run only when an effective bond cap is set.
        When the automatic route has no explicit ``plan`` or ``map_mode``, its
        layout finder also adapts the TreePlan to the term supports.
        ``compress="automaton"`` forces that full native assembly.
        ``compress=False`` disables numerical compression, and
        ``max_bond=None`` or ``False`` does the same.

        ``mode=`` remains a compatibility alias for the older separate
        strategy keyword; new code should put the strategy in ``compress=``.

        ``compress_opts`` currently accepts the native TreeMPO compression
        option ``order`` (``"rank"`` or ``"depth"``). The common
        ``max_bond`` and ``cutoff`` defaults are inherited from ``ham_tn``.
        ``progbar=True`` shows MPS-style term-construction progress with the
        current ``chi`` and requested cap.
        """
        from ..optimizers.tree import TreeMPO, build_tree_operator
        from ..tensors.symmetric import SymHamiltonian

        from ..optimizers.tree.layout import TreePlan

        if plan is not None and not isinstance(plan, TreePlan):
            if ints is not None:
                raise TypeError("Pass either a TreePlan or terms, not both as positional arguments.")
            ints = plan
            plan = None
        default_term = compress is _DEFAULT_COMPRESSION and mode is None
        build_mode, compress = _resolve_compression_request(
            compress,
            mode,
            tree=True,
        )
        if chi is not None:
            chi_limit = _normalize_optional_max_bond(chi, name="chi")
            if (
                max_bond is not _DEFAULT_MAX_BOND
                and max_bond not in (None, False)
                and _normalize_optional_max_bond(max_bond) != chi_limit
            ):
                raise TypeError("Pass only one of max_bond and chi, or use equal values.")
            max_bond = chi
        backend = self.to_backend if to_backend is _DEFAULT_BACKEND else to_backend
        dtype = (
            self._infer_backend_dtype(backend)
            if data_type is None and backend is not None and not self._data_type_explicit
            else self.data_type if data_type is None else np.dtype(data_type)
        )
        max_bond = _resolve_max_bond(max_bond, self.max_bond)
        compression_enabled = compress and max_bond is not None
        layout_max_bond = max_bond if max_bond is not None else 256
        cutoff = self._resolve_cutoff(
            self.cutoff if cutoff is None else cutoff,
            dtype,
        )

        layout_finder = None
        if plan is None:
            if (
                build_mode in {"analytic", "auto"}
                and map_mode is None
                and ints is not None
                and fermion is None
                and not isinstance(ints, SymHamiltonian)
            ):
                if not isinstance(ints, Mapping):
                    ints = tuple(ints)
                plan, mapper_use, layout_finder = self._auto_tree_mpo_plan(
                    ints,
                    max_bond=layout_max_bond,
                )
            else:
                plan, mapper_use = self._tree_plan_from_map_mode(map_mode)
        else:
            mapper_use = (
                self._tree_mapper_for_mode(map_mode)
                if map_mode is not None else self.mapper
            )
        self._validate_tree_plan(plan)

        if compress_opts is None:
            compress_extra = {}
        elif not isinstance(compress_opts, Mapping):
            raise TypeError("compress_opts must be a mapping or None.")
        else:
            compress_extra = dict(compress_opts)
        if "order" in compress_extra:
            value = compress_extra.pop("order")
            if compress_order != "rank" and compress_order != value:
                raise TypeError("conflicting compression order values supplied.")
            compress_order = value
        if compress_extra:
            names = ", ".join(sorted(compress_extra))
            raise TypeError(
                "TreeMPO compression supports only order='rank' or 'depth'; "
                f"got unsupported options: {names}."
            )
        if compress_order not in {"rank", "depth"}:
            raise ValueError("compress_order must be 'rank' or 'depth'.")

        # Native fermion and SymHamiltonian builders do not expose the dense
        # sequential factorization route. Keep their omitted-argument
        # behavior analytic while the dense public default is term-wise.
        if default_term and (fermion is not None or isinstance(ints, SymHamiltonian)):
            build_mode = "analytic"

        if fermion is not None:
            if build_mode == "term":
                raise ValueError(
                    "mode='term' is only available for dense tree terms; "
                    "native fermion construction uses its analytic builder."
                )
            if edges is not None:
                if ints is not None:
                    raise TypeError("Pass fermion terms through either ints or edges, not both.")
                ints = edges
            if ints is None and not isinstance(ints, SymHamiltonian):
                raise ValueError("TreeMPO construction requires terms or an edge sequence.")
            if model_params and isinstance(ints, SymHamiltonian):
                names = ", ".join(sorted(model_params))
                raise TypeError(
                    "Model parameters cannot be supplied with an existing "
                    f"SymHamiltonian: {names}."
                )
            fermionic_use = True if fermionic is None else bool(fermionic)
            operator = fermion.build_tree_operator(
                ints,
                tree=plan,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compression_enabled,
                dtype=dtype,
                fermionic=fermionic_use,
                charge_sectors=charge_sectors,
                to_backend=backend,
                **model_params,
            )
            supports = (
                ints.terms.keys()
                if isinstance(ints, SymHamiltonian)
                else ints
            )
            operator.layout_finder = self._tree_layout_finder_for_operator(
                plan,
                supports,
                mapper=mapper_use,
            )
            return operator

        if edges is not None or model_params or fermionic is not None or charge_sectors:
            raise TypeError(
                "edges, fermion model parameters, fermionic encoding, and "
                "charge_sectors are only valid with fermion=... ."
            )
        if ints is None:
            raise ValueError("ints must be provided.")
        if isinstance(ints, SymHamiltonian):
            if build_mode == "term":
                raise ValueError(
                    "mode='term' is only available for dense tree terms; "
                    "SymHamiltonian construction uses its analytic builder."
                )
            operator = build_tree_operator(
                plan,
                ints,
                max_bond=max_bond,
                cutoff=cutoff,
                compress=compression_enabled,
                dtype=dtype,
                fermionic=True,
                to_backend=backend,
            )
            operator.layout_finder = self._tree_layout_finder_for_operator(
                plan,
                ints.terms,
                mapper=mapper_use,
            )
            return operator
        if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
            raise ValueError("phys_dim must be an integer >= 1.")
        terms = self._normalize_tree_terms(
            plan,
            ints,
            phys_dim=int(phys_dim),
            dtype=dtype,
            mapper=mapper_use,
        )
        if layout_finder is None:
            layout_finder = self._tree_layout_finder_for_operator(
                plan,
                terms,
                mapper=mapper_use,
            )
        compress_options = {
            "max_bond": max_bond,
            "cutoff": cutoff,
            "order": compress_order,
        }
        if build_mode in {"analytic", "auto"}:
            progress_bar = _make_operator_progress(
                progbar,
                1,
                desc="tree-mpo",
            )
            operator = TreeMPO.from_terms(
                plan,
                terms,
                cutoff=cutoff,
                dtype=dtype,
                max_bond=max_bond,
                compress=False,
                layout_finder=layout_finder,
            )
            self._apply_tree_backend(operator, backend)
            if compression_enabled:
                operator.compress(**compress_options)
            elif backend is None:
                _structural_compress_tree(
                    operator,
                    root=operator.plan.root,
                    parent=operator.plan.parent,
                    children=operator.plan.children,
                    nodes=operator.plan.nodes(),
                    tensor_getter=operator.node_tensor,
                    bond_getter=operator.bond,
                    method="auto",
                )
            _advance_operator_progress(
                progress_bar,
                operator,
                cap=max_bond,
                peak=_operator_max_bond(operator),
            )
            if progress_bar is not None:
                progress_bar.close()
            return operator

        operator = None
        progress_bar = _make_operator_progress(
            progbar,
            len(terms),
            desc="tree-mpo-term",
        )
        peak_bond = None
        for support, term in terms.items():
            term_operator = TreeMPO.from_terms(
                plan,
                {support: term},
                cutoff=cutoff,
                dtype=dtype,
                max_bond=max_bond,
                compress=False,
                layout_finder=layout_finder,
            )
            self._apply_tree_backend(term_operator, backend)
            if operator is None:
                operator = term_operator
                peak_bond = _operator_max_bond(operator)
                if compression_enabled:
                    operator.compress(**compress_options)
            else:
                operator = operator.add_TreeMPO(
                    term_operator,
                    compress=compression_enabled,
                    **compress_options,
                )
                peak_bond = _operator_max_bond(operator)
            _advance_operator_progress(
                progress_bar,
                operator,
                cap=max_bond,
                peak=peak_bond,
            )
        if backend is None and not compression_enabled:
            _structural_compress_tree(
                operator,
                root=operator.plan.root,
                parent=operator.plan.parent,
                children=operator.plan.children,
                nodes=operator.plan.nodes(),
                tensor_getter=operator.node_tensor,
                bond_getter=operator.bond,
                method="auto",
            )
        _set_operator_progress_postfix(
            progress_bar,
            operator,
            cap=max_bond,
            peak=peak_bond,
        )
        if progress_bar is not None:
            progress_bar.close()
        return operator

    to_treempo = to_tree_mpo

    def to_tree_pepo(
        self,
        plan=None,
        ints=None,
        *,
        map_mode=None,
        tree_order=None,
        phys_dim=2,
        max_bond=_DEFAULT_MAX_BOND,
        chi=None,
        cutoff=None,
        cutoff_mode=None,
        data_type=None,
        mode=None,
        compress=_DEFAULT_COMPRESSION,
        form=None,
        center=None,
        reduced=True,
        compress_opts=None,
        to_backend=_DEFAULT_BACKEND,
        progbar=False,
    ):
        """Convert interactions directly to a native :class:`TreePEPO`.

        ``plan`` may be omitted and the first positional argument is then
        interpreted as the terms. For the canonical PEPO API, ``map_mode`` is
        one ``span-*`` string: ``span-up``, ``span-down``, ``span-out``, or
        ``span-middle``. It selects the retained physical spanning tree while
        the builder's ordinary map remains the logical site order. The old
        generic map spellings remain accepted as a compatibility route, with
        ``tree_order`` available there when the two views must differ. The
        returned operator is a complete ``TreePepsPlan`` network. Local terms
        are factorized over their minimal tree spans and never pass through a
        chain MPO or a full dense lattice operator.

        ``compress="term"`` (the default) adds one factorized term at a time
        and compresses after each addition. ``True`` and
        ``compress="auto"`` select the workload-aware native state-diagram
        route for compatibility. Automaton assembly is compressed once after
        all terms are assembled; term assembly is compressed after every
        added term. These numerical compressions run only when an effective
        bond cap is set. When the automatic route has no explicit ``plan``,
        retained-tree ``map_mode``, or ``tree_order``, its layout finder adapts
        the spanning tree to the term supports. ``compress="automaton"``
        forces the same full native assembly. ``compress=False`` disables
        numerical compression, and ``max_bond=None`` or ``False`` does the
        same.

        ``mode=`` remains a compatibility alias for the older separate
        strategy keyword; new code should put the strategy in ``compress=``.

        ``compress_opts`` accepts the native PEPO options ``form``, ``center``,
        ``reduced``, and ``order`` (``"rank"`` or ``"depth"``). The common
        ``max_bond``, ``cutoff``, and ``cutoff_mode`` defaults are inherited
        from ``ham_tn``.
        ``progbar=True`` shows MPS-style term-construction progress with the
        current ``chi`` and requested cap.
        """
        from ..optimizers.tree_peps import TreePepo

        from ..optimizers.tree_peps.plan import TreePepsPlan

        if plan is not None and not isinstance(plan, TreePepsPlan):
            if ints is not None:
                raise TypeError("Pass either a TreePepsPlan or terms, not both as positional arguments.")
            ints = plan
            plan = None
        build_mode, compress = _resolve_compression_request(
            compress,
            mode,
            tree=True,
        )
        if chi is not None:
            chi_limit = _normalize_optional_max_bond(chi, name="chi")
            if (
                max_bond is not _DEFAULT_MAX_BOND
                and max_bond not in (None, False)
                and _normalize_optional_max_bond(max_bond) != chi_limit
            ):
                raise TypeError("Pass only one of max_bond and chi, or use equal values.")
            max_bond = chi
        backend = self.to_backend if to_backend is _DEFAULT_BACKEND else to_backend
        dtype = (
            self._infer_backend_dtype(backend)
            if data_type is None and backend is not None and not self._data_type_explicit
            else self.data_type if data_type is None else np.dtype(data_type)
        )
        max_bond = _resolve_max_bond(max_bond, self.max_bond)
        compression_enabled = compress and max_bond is not None
        layout_max_bond = max_bond if max_bond is not None else 256
        cutoff = self._resolve_cutoff(
            self.cutoff if cutoff is None else cutoff,
            dtype,
        )
        if ints is None:
            raise ValueError("ints must be provided.")

        if plan is None:
            from ..optimizers.tree_peps.plan import _normalize_span_mode

            shape = (self.Lx, self.Ly) if self.Lz is None else (
                self.Lx, self.Ly, self.Lz
            )
            if (
                build_mode in {"analytic", "auto"}
                and map_mode is None
                and tree_order is None
            ):
                if not isinstance(ints, Mapping):
                    ints = tuple(ints)
                plan, _planning_finder = self._auto_tree_pepo_plan(
                    ints,
                    max_bond=layout_max_bond,
                )
            else:
                requested_mode = self.map_mode if map_mode is None else map_mode
                span_mode = _normalize_span_mode(requested_mode)
                if span_mode is not None:
                    if tree_order is not None:
                        raise TypeError(
                            "map_mode='span-*' and tree_order cannot both be "
                            "supplied; use one retained-tree mode"
                        )
                    # A PEPO's logical ids retain the builder's chain map, while
                    # its virtual geometry is selected independently by span-*.
                    plan = TreePepsPlan.from_shape(
                        shape,
                        order=self.map_mode,
                        map_mode=span_mode,
                    )
                else:
                    # Compatibility route for the pre-span API: one generic
                    # map_mode still controls both logical and retained orders.
                    order = requested_mode
                    tree_order_use = order if tree_order is None else tree_order
                    plan = TreePepsPlan.from_shape(
                        shape,
                        order=order,
                        tree_order=tree_order_use,
                    )
        self._validate_tree_plan(plan, tree_peps=True)
        if compress_opts is None:
            compress_extra = {}
        elif not isinstance(compress_opts, Mapping):
            raise TypeError("compress_opts must be a mapping or None.")
        else:
            compress_extra = dict(compress_opts)
        for name, current in (("form", form), ("center", center)):
            if name not in compress_extra:
                continue
            value = compress_extra.pop(name)
            if current is not None and current != value:
                raise TypeError(f"conflicting {name} values supplied.")
            if name == "form":
                form = value
            else:
                center = value
        if "reduced" in compress_extra:
            reduced = compress_extra.pop("reduced")
        compress_order = compress_extra.pop("order", "rank")
        if compress_order not in {"rank", "depth"}:
            raise ValueError("TreePEPO compression order must be 'rank' or 'depth'.")
        if compress_extra:
            names = ", ".join(sorted(compress_extra))
            raise TypeError(
                "TreePEPO compression received unsupported options: "
                f"{names}."
            )
        cutoff_mode = self._resolve_cutoff_mode(
            (self.cutoff_mode if cutoff_mode is None else cutoff_mode)
            or "rsum2",
        )
        if not isinstance(phys_dim, Integral) or int(phys_dim) < 1:
            raise ValueError("phys_dim must be an integer >= 1.")
        terms = self._normalize_tree_terms(
            plan,
            ints,
            phys_dim=int(phys_dim),
            dtype=dtype,
        )
        layout_finder = self._tree_peps_layout_finder_for_operator(
            plan,
            terms,
        )
        compress_options = {
            "max_bond": max_bond,
            "cutoff": cutoff,
            "cutoff_mode": cutoff_mode,
            "reduced": reduced,
            "order": compress_order,
        }
        if form is not None:
            compress_options["form"] = form
        if center is not None:
            compress_options["center"] = center
        if build_mode in {"analytic", "auto"}:
            progress_bar = _make_operator_progress(
                progbar,
                1,
                desc="tree-pepo",
            )
            operator = TreePepo.from_terms(
                plan,
                terms,
                dims=int(phys_dim),
                dtype=dtype,
                layout_finder=layout_finder,
            )
            self._apply_tree_backend(operator, backend)
            if compression_enabled:
                operator.compress(**compress_options)
            elif backend is None:
                _structural_compress_tree(
                    operator,
                    root=operator.plan.root,
                    parent=operator.plan.parent,
                    children=operator.plan.children,
                    nodes=operator.sites,
                    tensor_getter=operator.node_tensor,
                    bond_getter=operator.bond,
                    method="auto",
                )
            _advance_operator_progress(
                progress_bar,
                operator,
                cap=max_bond,
                peak=_operator_max_bond(operator),
            )
            if progress_bar is not None:
                progress_bar.close()
            return operator

        operator = None
        progress_bar = _make_operator_progress(
            progbar,
            len(terms),
            desc="tree-pepo-term",
        )
        peak_bond = None
        for support, term in terms.items():
            term_operator = TreePepo.from_terms(
                plan,
                {support: term},
                dims=int(phys_dim),
                dtype=dtype,
                layout_finder=layout_finder,
            )
            self._apply_tree_backend(term_operator, backend)
            if operator is None:
                operator = term_operator
                peak_bond = _operator_max_bond(operator)
                if compression_enabled:
                    operator.compress(**compress_options)
            else:
                operator = operator.add_operator(
                    term_operator,
                    compress=compression_enabled,
                    _validate=False,
                    **compress_options,
                )
                peak_bond = _operator_max_bond(operator)
            _advance_operator_progress(
                progress_bar,
                operator,
                cap=max_bond,
                peak=peak_bond,
            )
        if backend is None and not compression_enabled:
            _structural_compress_tree(
                operator,
                root=operator.plan.root,
                parent=operator.plan.parent,
                children=operator.plan.children,
                nodes=operator.sites,
                tensor_getter=operator.node_tensor,
                bond_getter=operator.bond,
                method="auto",
            )
            operator.validate()
        _set_operator_progress_postfix(
            progress_bar,
            operator,
            cap=max_bond,
            peak=peak_bond,
        )
        if progress_bar is not None:
            progress_bar.close()
        return operator

    to_treepepsmpo = to_tree_pepo

    def mpo_itf(
        self,
        J=1.0,
        field=1.0,
        *,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        as_pepo=False,
        cycle_peps=False,
        cycle_bond_dim=1,
    ):
        """Build transverse-field Ising MPO on the builder lattice.

        Hamiltonian:
        ``H = J * sum_<ij> Z_i Z_j + field * sum_i X_i``

        For 2D builders this uses square-lattice nearest-neighbour edges.
        For 3D builders (``L_z`` provided) this uses cubic-lattice nearest-
        neighbour edges.

        Returns
        -------
        tuple
            ``(op, coord_to_chain_map)`` where ``op`` is MPO by default and
            PEPO when ``as_pepo=True`` (2D only).
        """
        dtype = self.data_type if data_type is None else np.dtype(data_type)

        if self.ndim == 2:
            edges = tuple(qtn.edges_2d_square(self.L_x, self.L_y, cyclic=False))
        else:
            edges = tuple(qtn.edges_3d_cubic(self.L_x, self.L_y, self.L_z, cyclic=False))

        ints = self._itf_ints_from_edges(edges, J=J, field=field, dtype=dtype)
        mpo = self.to_mpo(
            ints,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=dtype,
            compress_each=compress_each,
        )

        if as_pepo:
            self._require_2d("mpo_itf(as_pepo=True)")
            pepo = self.mpo_to_pepo(
                mpo,
                cycle_peps=cycle_peps,
                cycle_bond_dim=cycle_bond_dim,
                inplace=False,
            )
            return pepo, dict(self.map_inv)
        return mpo, dict(self.map_inv)

    def _itf_ints_from_edges(
        self,
        edges,
        *,
        J,
        field,
        dtype,
    ):
        z_op = np.asarray(quimb.pauli("Z", dtype=dtype), dtype=dtype)
        x_op = np.asarray(quimb.pauli("X", dtype=dtype), dtype=dtype)

        sites = sorted({site for edge in edges for site in edge})
        ints = [((z_op, z_op), edge, J) for edge in edges]
        ints.extend((((x_op,), (site,), field) for site in sites))
        return ints

    def build_itf_from_edges(
        self,
        edges,
        J=1.0,
        field=1.0,
        *,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
        return_mpo=True,
        return_pepo=False,
    ):
        """Build ITF Hamiltonian MPO and optionally PEPO from an edge list.

        Accepts any quimb geometry edge list, e.g.::

            qtn.edges_2d_square(Lx, Ly, cyclic=False)
            qtn.edges_2d_triangular(Lx, Ly, cyclic=False)

        Each edge must be ``((x0, y0), (x1, y1))``.  Sites are inferred as
        the union of all edge endpoints.

        Hamiltonian:
        ``H = J * sum_{edges} Z_i Z_j + field * sum_{sites} X_i``

        Parameters
        ----------
        edges : iterable of ((int, int), (int, int))
            Nearest-neighbour edge list from a quimb geometry function.
        J : float, default=1.0
            ZZ coupling strength.
        field : float, default=1.0
            Transverse-field (X) strength.
        return_mpo : bool, default=True
            If True, build and return the MPO.
        return_pepo : bool, default=False
            If True, also convert the MPO to a PEPO. This requires a
            snake-style 2D mapping.

        Returns
        -------
        tuple
            ``(H_mpo, H_pepo)`` where either entry can be ``None`` when not
            requested.
        """
        dtype = self.data_type if data_type is None else np.dtype(data_type)
        edges = [tuple(edge) for edge in edges]
        if not edges:
            raise ValueError("edges must not be empty.")

        if not return_mpo and not return_pepo:
            return None, None

        ints = self._itf_ints_from_edges(edges, J=J, field=field, dtype=dtype)

        mpo = self.to_mpo(
            ints,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=dtype,
            compress_each=compress_each,
        )
        pepo = None
        if return_pepo:
            pepo = self.mpo_to_pepo(
                mpo,
                cycle_peps=cycle_peps,
                cycle_bond_dim=cycle_bond_dim,
                inplace=False,
            )
        if not return_mpo:
            mpo = None
        return mpo, pepo

    def plot_lattice_snake(
        self,
        edges,
        *,
        ax=None,
        title=None,
        show_chain_index=True,
        edge_color="0.72",
        snake_color="tab:red",
        node_color="tab:blue",
        edge_alpha=0.9,
        snake_alpha=0.95,
        node_size=165,
        edge_linewidth=1.6,
        snake_linewidth=2.6,
        node_edge_color="white",
        node_edge_width=1.2,
        snake_arrows=True,
        snake_cmap="plasma",
        show_legend=True,
        site_positions=None,
        invert_y=None,
        print_output=True,
        color="auto",
    ):
        """Render lattice geometry and mapped traversal as ASCII text.

        Parameters
        ----------
        edges : iterable of ((int, int), (int, int))
            Lattice edge list.
        ax : object | None, default=None
            Kept for API compatibility. Ignored by ASCII renderer.
        title : str | None, default=None
            Header title shown above the ASCII preview.
        show_chain_index : bool, default=True
            If True, append the chain-index listing for the active map.
        snake_arrows : bool, default=True
            If True, mark mapped-path direction on traversed links.
        show_legend : bool, default=True
            If True, append a one-line symbol legend.
        site_positions : Mapping[(int, int), (float, float)] | None, default=None
            Kept for API compatibility. Ignored by ASCII renderer.
        invert_y : bool | None, default=None
            Kept for API compatibility. Ignored by ASCII renderer.
        print_output : bool, default=True
            If True, print the rendered text.
        color : bool | {"auto"}, default="auto"
            Enable ANSI color styling. ``"auto"`` enables colors when stdout
            is a TTY.

        Returns
        -------
        str
            Multiline ASCII rendering of lattice edges and mapped traversal.
        """
        self._require_2d("plot_lattice_snake")
        _ = (
            edge_color,
            snake_color,
            node_color,
            edge_alpha,
            snake_alpha,
            node_size,
            edge_linewidth,
            snake_linewidth,
            node_edge_color,
            node_edge_width,
            snake_cmap,
            site_positions,
            invert_y,
        )

        color_enabled = resolve_color_mode(color)

        if ax is not None:
            warnings.warn(
                "plot_lattice_snake now returns ASCII text; argument 'ax' is ignored.",
                UserWarning,
                stacklevel=2,
            )

        edges = [tuple(edge) for edge in edges]
        if not edges:
            raise ValueError("edges must not be empty for plotting.")

        edge_keys = set()
        for idx, edge in enumerate(edges):
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise TypeError(
                    f"Edge at position {idx} must be ((x0, y0), (x1, y1)), got {edge!r}."
                )
            site0 = self._coerce_coord(edge[0])
            site1 = self._coerce_coord(edge[1])
            edge_keys.add(frozenset((site0, site1)))

        snake_sites = [self.map[idx] for idx in range(self.L)]
        snake_dir = {}
        for site0, site1 in zip(snake_sites[:-1], snake_sites[1:]):
            dx = site1[0] - site0[0]
            dy = site1[1] - site0[1]
            if dx == 1 and dy == 0:
                token = "r"
            elif dx == -1 and dy == 0:
                token = "l"
            elif dx == 0 and dy == 1:
                token = "u"
            elif dx == 0 and dy == -1:
                token = "d"
            elif dx == 1 and dy == -1:
                token = "dr"
            elif dx == -1 and dy == 1:
                token = "ul"
            elif dx == 1 and dy == 1:
                token = "ur"
            elif dx == -1 and dy == -1:
                token = "dl"
            else:
                token = "."
            snake_dir[frozenset((site0, site1))] = token

        if title is None:
            title = f"Lattice Geometry + {self.map_mode} Path ({self.L_x}x{self.L_y})"

        row_count = max(1, 2 * self.L_y - 1)
        row_width = 4 * self.L_x - 3
        canvas = [[" "] * row_width for _ in range(row_count)]
        layer = [["empty"] * row_width for _ in range(row_count)]

        node_to_rc = {
            (x, y): (2 * (self.L_y - 1 - y), 4 * x)
            for x in range(self.L_x)
            for y in range(self.L_y)
        }
        for (x, y), (row, col) in node_to_rc.items():
            if 0 <= row < row_count and 0 <= col < row_width:
                _ = (x, y)
                canvas[row][col] = "●"
                layer[row][col] = "node"

        def _put(row, col, char, layer_name):
            if not (0 <= row < row_count and 0 <= col < row_width):
                return
            current = canvas[row][col]
            if current == "●":
                return
            char_priority = {
                " ": 0,
                "-": 1,
                "_": 1,
                "|": 1,
                "/": 1,
                "\\": 1,
                "*": 1,
                ">": 2,
                "<": 2,
                "^": 2,
                "v": 2,
            }
            layer_priority = {
                "empty": 0,
                "lattice": 1,
                "snake": 2,
                "node": 3,
            }
            current_layer = layer[row][col]
            new_lp = layer_priority.get(layer_name, 1)
            old_lp = layer_priority.get(current_layer, 0)
            new_cp = char_priority.get(char, 1)
            old_cp = char_priority.get(current, 0)
            if (new_lp > old_lp) or (new_lp == old_lp and new_cp >= old_cp):
                canvas[row][col] = char
                layer[row][col] = layer_name

        periodic_edges_summary = set()
        for edge_key in edge_keys:
            site0, site1 = tuple(edge_key)
            if site0 not in node_to_rc or site1 not in node_to_rc:
                continue
            row0, col0 = node_to_rc[site0]
            row1, col1 = node_to_rc[site1]
            token = snake_dir.get(edge_key)
            x0, y0 = site0
            x1, y1 = site1

            wrap_x = self.L_x > 2 and {x0, x1} == {0, self.L_x - 1}
            wrap_y = self.L_y > 2 and {y0, y1} == {0, self.L_y - 1}
            far_jump = abs(x1 - x0) > 1 or abs(y1 - y0) > 1
            is_cyclic_edge = bool(wrap_x or wrap_y or far_jump)
            if is_cyclic_edge:
                periodic_edges_summary.add(tuple(sorted((site0, site1))))

            if row0 == row1 and abs(col0 - col1) == 4:
                col_left = min(col0, col1)
                if token in {"r", "l"}:
                    if snake_arrows:
                        connector = "__>" if token == "r" else "<__"
                    else:
                        connector = "___"
                    conn_layer = "snake"
                else:
                    connector = "___"
                    conn_layer = "cyclic" if is_cyclic_edge else "lattice"
                for i, ch in enumerate(connector):
                    _put(row0, col_left + 1 + i, ch, conn_layer)
                continue

            if col0 == col1 and abs(row0 - row1) == 2:
                row_mid = (row0 + row1) // 2
                if token in {"u", "d"}:
                    if snake_arrows:
                        symbol = "^" if token == "u" else "v"
                    else:
                        symbol = "|"
                    conn_layer = "snake"
                else:
                    symbol = "|"
                    conn_layer = "cyclic" if is_cyclic_edge else "lattice"
                _put(row_mid, col0, symbol, conn_layer)
                continue

            if abs(row0 - row1) == 2 and abs(col0 - col1) == 4:
                row_mid = (row0 + row1) // 2
                col_mid = (col0 + col1) // 2
                slope = (row1 - row0) * (col1 - col0)
                base = "\\" if slope > 0 else "/"
                if token in {"dr", "ur", "dl", "ul"}:
                    if snake_arrows and token in {"dr", "ur"}:
                        symbol = ">"
                    elif snake_arrows and token in {"dl", "ul"}:
                        symbol = "<"
                    else:
                        symbol = base
                    conn_layer = "snake"
                else:
                    symbol = base
                    conn_layer = "cyclic" if is_cyclic_edge else "lattice"
                _put(row_mid, col_mid, symbol, conn_layer)
                continue

            # Non-local periodic connections are summarized separately below.
            if is_cyclic_edge:
                continue

            row_mid = (row0 + row1) // 2
            col_mid = (col0 + col1) // 2
            _put(row_mid, col_mid, "_", "snake" if token is not None else "lattice")

        def _style_row(chars, layers):
            if not color_enabled:
                return "".join(chars)
            codes = {
                "node": "1;37",
                "lattice": "36",
                "cyclic": "1;35",
                "snake": "90",
            }
            out = []
            for ch, layer_name in zip(chars, layers):
                if ch == " ":
                    out.append(ch)
                    continue
                if layer_name == "snake" and ch in {">", "<", "^", "v"}:
                    code = "1;31"
                else:
                    code = codes.get(layer_name)
                out.append(ansi_wrap(ch, code, True) if code else ch)
            return "".join(out)

        lines = [title]
        for row in range(row_count):
            if row % 2 == 0:
                y = self.L_y - 1 - (row // 2)
                prefix = f"Y{y:<2} "
            else:
                prefix = "    "
            if color_enabled and row % 2 == 0:
                prefix = ansi_wrap(prefix, "1;33", True)
            lines.append(prefix + _style_row(canvas[row], layer[row]))

        x_line = "    " + "   ".join(f"X{x}" for x in range(self.L_x))
        if color_enabled:
            x_line = ansi_wrap(x_line, "1;33", True)
        lines.append(x_line)

        if show_legend:
            legend = "legend: ● node  _|/\\ edge  > < ^ v snake direction"
            if color_enabled:
                legend = (
                    "legend: "
                    + ansi_wrap("●", "1;37", True)
                    + " node  "
                    + ansi_wrap("___|/\\", "36", True)
                    + " lattice  "
                    + ansi_wrap("___|", "90", True)
                    + " snake  "
                    + ansi_wrap("> < ^ v", "1;31", True)
                    + " direction  "
                    + ansi_wrap("~~~", "1;35", True)
                    + " cyclic"
                )
            lines.append(legend)

        if periodic_edges_summary:
            header = "cyclic edges (wrap connections):"
            if color_enabled:
                header = ansi_wrap(header, "1;35", True)
            lines.append(header)
            for site0, site1 in sorted(periodic_edges_summary):
                edge_line = f"  {site0} <-> {site1}"
                if color_enabled:
                    edge_line = ansi_wrap(edge_line, "1;35", True)
                lines.append(edge_line)

        if show_chain_index:
            lines.append(
                "snake: "
                + ", ".join(f"{idx}:{coord}" for idx, coord in enumerate(snake_sites))
            )

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text

    @staticmethod
    def _mpo_chain_bond_dims(mpo):
        """Return MPO bond dimensions between consecutive chain tensors."""
        dims = []
        for idx in range(mpo.L - 1):
            left = mpo[idx]
            right = mpo[idx + 1]
            shared = tuple(set(left.inds) & set(right.inds))
            if not shared:
                dims.append(1)
                continue
            dims.append(max(int(left.ind_size(ix)) for ix in shared))
        return dims

    def _show_mpo_schematic_2d(
        self,
        mpo,
        edges,
        *,
        title=None,
        site_positions=None,
        ax=None,
        figsize=None,
    ):
        """Render a schematic MPO-on-lattice view with chain bond dimensions."""
        self._require_2d("_show_mpo_schematic_2d")
        if mpo is None:
            raise ValueError("An MPO is required to render the ITF schematic.")
        if getattr(mpo, "L", None) != self.L:
            raise ValueError(
                f"MPO length mismatch: expected {self.L}, got {getattr(mpo, 'L', None)}."
            )

        try:
            from matplotlib import colormaps
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Schematic plotting requires matplotlib to be available."
            ) from exc

        try:
            from quimb import schematic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Schematic plotting requires quimb.schematic to be available."
            ) from exc

        positions = (
            {tuple(site): (float(xy[0]), float(xy[1])) for site, xy in site_positions.items()}
            if site_positions is not None
            else {
                (x, y): (float(x), float(y))
                for x in range(self.L_x)
                for y in range(self.L_y)
            }
        )
        if title is None:
            title = f"ITF MPO ({self.map_mode}, max bond={int(mpo.max_bond())})"
        if figsize is None:
            xs = [xy[0] for xy in positions.values()]
            ys = [xy[1] for xy in positions.values()]
            figsize = (
                max(5.0, 1.35 * (max(xs) - min(xs) + 1.0)),
                max(4.2, 1.35 * (max(ys) - min(ys) + 1.0)),
            )

        presets = {
            "lattice": {
                "color": (0.80, 0.82, 0.86, 1.0),
                "linewidth": 1.8,
            },
            "node": {
                "facecolor": schematic.get_color("blue"),
                "edgecolor": "white",
                "linewidth": 1.2,
                "radius": 0.18,
            },
        }
        drawing = schematic.Drawing(presets=presets, ax=ax, figsize=figsize)

        for site0, site1 in edges:
            drawing.line(positions[tuple(site0)], positions[tuple(site1)], preset="lattice")

        coords = [self.map[idx] for idx in range(self.L)]
        bond_dims = self._mpo_chain_bond_dims(mpo)
        cmap = colormaps.get_cmap("plasma")
        dim_min = min(bond_dims, default=1)
        dim_max = max(bond_dims, default=1)
        dim_span = max(1, dim_max - dim_min)

        for idx, (site0, site1) in enumerate(zip(coords[:-1], coords[1:])):
            dim = bond_dims[idx]
            frac = (dim - dim_min) / dim_span
            color = cmap(frac)
            width = 2.2 + 2.4 * frac
            pos0 = positions[tuple(site0)]
            pos1 = positions[tuple(site1)]
            drawing.line(pos0, pos1, color=color, linewidth=width, zorder=3)
            drawing.arrowhead(pos0, pos1, color=color, center=0.58, width=0.10)

            mid_x = 0.5 * (pos0[0] + pos1[0])
            mid_y = 0.5 * (pos0[1] + pos1[1])
            dx = pos1[0] - pos0[0]
            dy = pos1[1] - pos0[1]
            norm = max((dx * dx + dy * dy) ** 0.5, 1e-12)
            off_x = -0.08 * dy / norm
            off_y = 0.08 * dx / norm
            drawing.ax.text(
                mid_x + off_x,
                mid_y + off_y,
                str(dim),
                fontsize=8,
                ha="center",
                va="center",
                color=(0.18, 0.20, 0.24, 1.0),
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": (1.0, 1.0, 1.0, 0.88),
                    "edgecolor": color,
                    "linewidth": 0.8,
                },
                zorder=5,
            )

        for coord, chain_idx in self.map_inv.items():
            pos = positions[tuple(coord)]
            drawing.circle(pos, preset="node", zorder=4)
            drawing.ax.text(
                pos[0],
                pos[1],
                str(chain_idx),
                fontsize=9,
                color="white",
                ha="center",
                va="center",
                zorder=6,
            )

        xs = [xy[0] for xy in positions.values()]
        ys = [xy[1] for xy in positions.values()]
        pad_x = max(0.35, 0.08 * (max(xs) - min(xs) + 1.0))
        pad_y = max(0.35, 0.08 * (max(ys) - min(ys) + 1.0))
        drawing.ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        drawing.ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        drawing.ax.set_aspect("equal")
        drawing.ax.set_title(title)
        drawing.ax.axis("off")
        return drawing

    def build_itf(
        self,
        lattice="square",
        *,
        edges=None,
        cyclic=False,
        J=1.0,
        field=1.0,
        max_bond=None,
        cutoff=None,
        data_type=None,
        compress_each=True,
        cycle_peps=False,
        cycle_bond_dim=1,
        edge_kwargs=None,
        show=False,
        return_edges=False,
        return_mpo=True,
        return_pepo=False,
    ):
        """Build ITF Hamiltonian from a named quimb 2D lattice generator.

        This wraps :meth:`build_itf_from_edges` by generating edges from
        ``qtn.edges_2d_<lattice>``. For example:

        - ``lattice="square"`` -> ``qtn.edges_2d_square``
        - ``lattice="triangular"`` -> ``qtn.edges_2d_triangular``
        - ``lattice="hexagonal"`` -> ``qtn.edges_2d_hexagonal``

        You can also pass a callable as ``lattice`` or provide ``edges``
        directly to bypass name-based generation.

        Notes
        -----
        Quimb lattices such as ``hexagonal`` and ``kagome`` use site labels
        of form ``(x, y, sublattice)``. These are remapped internally to an
        expanded rectangular grid ``(x, y * n_sub + offset[sublattice])`` so
        that MPO/PEPO construction can proceed on a supported 2D mapped layout. For
        plotting, a geometric embedding is used so these lattices look like
        their expected physical connectivity.

        Parameters
        ----------
        lattice : str | callable, default="square"
            Lattice name suffix or edge-builder callable.
        edges : iterable | None, default=None
            Optional explicit edge list. If provided, ``lattice`` is ignored.
        cyclic : bool, default=False
            Passed to quimb edge generators when available.
        edge_kwargs : dict | None, default=None
            Extra kwargs forwarded to the edge generator.
        show : bool, default=False
            If True, include a schematic drawing of the MPO path on the lattice.
        return_edges : bool, default=False
            If True, include ``edges`` in the return tuple.
        return_mpo : bool, default=True
            If True, build and return the MPO.
        return_pepo : bool, default=False
            If True, also build and return the PEPO.

        Returns
        -------
        tuple
            ``(H_mpo, H_pepo)`` by default, or
            ``(H_mpo, H_pepo, edges)`` when ``return_edges=True``, or
            includes the schematic drawing when ``show=True``.
        """
        self._require_2d("build_itf")

        def _sublattice_display_positions(site_map_):
            labels_local = sorted({site[2] for site in site_map_}, key=repr)
            n_sub_local = len(labels_local)
            sqrt3 = float(np.sqrt(3.0))

            if n_sub_local == 2:
                offset_seq = [
                    (0.00, 0.00),
                    (0.50, sqrt3 / 6.0),
                ]
            elif n_sub_local == 3:
                offset_seq = [
                    (0.00, 0.00),
                    (0.50, 0.00),
                    (0.25, sqrt3 / 4.0),
                ]
            else:
                offset_seq = [
                    (
                        0.35 * np.cos(2.0 * np.pi * k / max(n_sub_local, 1)),
                        0.35 * np.sin(2.0 * np.pi * k / max(n_sub_local, 1)),
                    )
                    for k in range(n_sub_local)
                ]

            label_to_off = {
                lab: offset_seq[idx]
                for idx, lab in enumerate(labels_local)
            }

            out = {}
            for site_raw, site_rect in site_map_.items():
                x_raw, y_raw, lab = site_raw
                base_x = float(x_raw) + 0.5 * float(y_raw)
                base_y = (sqrt3 / 2.0) * float(y_raw)
                off_x, off_y = label_to_off[lab]
                out[site_rect] = (base_x + off_x, base_y + off_y)
            return out

        if edge_kwargs is None:
            edge_kwargs = {}

        lattice_for_title = "custom"
        if edges is None:
            if callable(lattice):
                edge_builder = lattice
                lattice_for_title = "custom"
            elif isinstance(lattice, str):
                lattice_name = lattice.strip().lower()
                if lattice_name.startswith("edges_2d_"):
                    lattice_name = lattice_name.replace("edges_2d_", "", 1)
                builder_name = f"edges_2d_{lattice_name}"
                edge_builder = getattr(qtn, builder_name, None)
                if edge_builder is None or not callable(edge_builder):
                    available = sorted(
                        name.replace("edges_2d_", "", 1)
                        for name in dir(qtn)
                        if name.startswith("edges_2d_") and callable(getattr(qtn, name))
                    )
                    raise ValueError(
                        f"Unknown lattice '{lattice}'. Available 2D generators: {available}"
                    )
                lattice_for_title = lattice_name
            else:
                raise TypeError("lattice must be a string, callable, or None when edges given.")

            try:
                edges_raw = edge_builder(self.L_x, self.L_y, cyclic=cyclic, **edge_kwargs)
            except TypeError:
                edges_raw = edge_builder(self.L_x, self.L_y, **edge_kwargs)
            edges_use = list(edges_raw)
        else:
            edges_use = list(edges)

        if not edges_use:
            raise ValueError("edges must not be empty.")

        # Some quimb 2D periodic generators emit degenerate singleton/self-loop
        # edges when a lattice dimension is 1 (e.g. Ly=1 with cyclic=True).
        # Drop those generated artifacts so 1D reductions still build cleanly.
        if edges is None:
            filtered_edges = []
            dropped = 0
            for edge in edges_use:
                if isinstance(edge, (tuple, list)) and len(edge) == 1:
                    dropped += 1
                    continue
                if (
                    isinstance(edge, (tuple, list))
                    and len(edge) == 2
                    and edge[0] == edge[1]
                ):
                    dropped += 1
                    continue
                filtered_edges.append(edge)
            if dropped:
                warnings.warn(
                    f"Dropped {dropped} degenerate generated edge(s) for "
                    f"shape ({self.L_x}, {self.L_y}) with cyclic={cyclic}.",
                    UserWarning,
                    stacklevel=2,
                )
            edges_use = filtered_edges

            if not edges_use:
                raise ValueError(
                    "Generated edges are empty after filtering degenerate periodic edges."
                )

        edges_use = [tuple(edge) for edge in edges_use]
        raw_sites = {
            tuple(site) if isinstance(site, (tuple, list)) else site
            for edge in edges_use
            for site in edge
        }
        raw_sites = sorted(raw_sites, key=repr)

        builder_use = self
        plot_site_positions = None
        if all(is_xy_site(site) for site in raw_sites):
            site_map = {site: (int(site[0]), int(site[1])) for site in raw_sites}
        elif all(is_xy_sublattice_site(site) for site in raw_sites):
            labels = sorted({site[2] for site in raw_sites}, key=repr)
            n_sub = len(labels)
            label_to_off = {lab: off for off, lab in enumerate(labels)}
            site_map = {
                site: (
                    int(site[0]),
                    int(site[1]) * n_sub + label_to_off[site[2]],
                )
                for site in raw_sites
            }
            plot_site_positions = _sublattice_display_positions(site_map)
            if n_sub > 1:
                builder_use = ham_tn(
                    L_x=self.L_x,
                    L_y=self.L_y * n_sub,
                    max_bond=self.max_bond,
                    cutoff=self.cutoff,
                    data_type=self.data_type,
                    to_backend=self.to_backend,
                    mapper=OneDMap(self.L_x, self.L_y * n_sub, mode=self.map_mode),
                )
                warnings.warn(
                    "Detected sublattice-labelled sites (e.g. kagome/hexagonal). "
                    f"Remapping to rectangular grid with shape "
                    f"({builder_use.L_x}, {builder_use.L_y}) for MPO/PEPO build.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            sample = raw_sites[0]
            raise TypeError(
                "Unsupported site format from edges. Expected (x, y) or "
                f"(x, y, sublattice), got sample site {sample!r}."
            )

        edges_norm = []
        for idx, edge in enumerate(edges_use):
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise TypeError(
                    f"Edge at position {idx} must be ((x0, y0), (x1, y1)), got {edge!r}."
                )
            site0_raw = tuple(edge[0]) if isinstance(edge[0], (tuple, list)) else edge[0]
            site1_raw = tuple(edge[1]) if isinstance(edge[1], (tuple, list)) else edge[1]
            if site0_raw not in site_map or site1_raw not in site_map:
                raise ValueError(
                    f"Edge at position {idx} references site not present in normalized map."
                )
            site0 = site_map[site0_raw]
            site1 = site_map[site1_raw]

            if site0 == site1:
                raise ValueError(f"Edge at position {idx} has identical endpoints {site0}.")

            for site in (site0, site1):
                x_val, y_val = site
                if not (0 <= x_val < builder_use.L_x and 0 <= y_val < builder_use.L_y):
                    raise ValueError(
                        f"Edge at position {idx} has out-of-bounds site {site} "
                        f"for shape ({builder_use.L_x}, {builder_use.L_y})."
                    )
            edges_norm.append((site0, site1))

        h_mpo, h_pepo = builder_use.build_itf_from_edges(
            edges_norm,
            J=J,
            field=field,
            max_bond=max_bond,
            cutoff=cutoff,
            data_type=data_type,
            compress_each=compress_each,
            cycle_peps=cycle_peps,
            cycle_bond_dim=cycle_bond_dim,
            return_mpo=(return_mpo or show),
            return_pepo=return_pepo,
        )

        drawing = None
        if show:
            drawing = builder_use._show_mpo_schematic_2d(
                h_mpo,
                edges_norm,
                title=f"{lattice_for_title.capitalize()} ITF MPO ({builder_use.map_mode})",
                site_positions=plot_site_positions,
            )

        if not return_mpo:
            h_mpo = None

        if return_edges and show:
            return h_mpo, h_pepo, tuple(edges_norm), drawing
        if return_edges:
            return h_mpo, h_pepo, tuple(edges_norm)
        if show:
            return h_mpo, h_pepo, drawing
        return h_mpo, h_pepo
