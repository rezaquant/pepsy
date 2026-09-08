"""MPO optimization helpers centered on :class:`MpoOptimizer`.

:class:`MpoOptimizer` replays a queue of one- or multi-site gates
``[(gate, where), ...]`` against an MPO ``O`` of length ``L`` with two
physical index families (``ind_id_k``, ``ind_id_b``). Each bundled entry
specifies what acts on the ket and bra legs:

* ``gate``                       → apply ``gate`` on ket and ``gate†`` on bra
  (default "unitary conjugation" semantics ``G O G†`` in the MPO API);
* ``(gate,)`` or ``(gate, None)`` → apply ``gate`` on ket only;
* ``(None, B)``                  → apply ``B†`` on bra only;
* ``(G, B)``                     → apply ``G`` on ket and ``B†`` on bra.

Raw dense payloads follow the existing tensor-index orientation: the four
actions above are respectively ``G.T @ O @ G.conj()``, ``G.T @ O``,
``O @ B.conj()``, and ``G.T @ O @ B.conj()``. To evolve by a conventional
matrix ``A O A†``, supply ``A.T``. Channel events instead accept standard
Kraus matrices and deterministically form ``sum K O K†``.

Three execution backends are supported, all returning the same kind of MPO
but differing in *how* local gate updates are compressed back to bond ``chi``:

* ``mode="dmrg"`` — fit a target MPO with :class:`pepsy.fitting.local.FIT`
  inside a local window ``[xmin, xmax]``; supports batching consecutive
  two-site gates via ``k_2q_batch``;
* ``mode="dmrg1"``, ``"dmrg2"``, or ``"dmrg3"`` — named DMRG schedules
  sharing the same local MPO/FIT kernel: two-site growth for exactly two
  sweeps (``dmrg1``) or a configurable warm-up (``dmrg2``), three-site
  warm-up plus a two-site transition (``dmrg3``), then one-site refinement.
  ``dmrg1`` latches into its
  one-site phase after the attainable MPO bond ranks are saturated. Generic
  ``dmrg`` follows the same adaptive block-to-one-site handoff;
* ``mode="svd"``  — apply the gate with ``reduce-split`` then canonicalize +
  left-compress to ``chi``;
* ``mode="direct"`` (default; ``"mpo"``/``"quimb"`` aliases) — use Quimb's native dagger-aware
  ``gate_sandwich_with_auto_swap`` path for bare/default two-site dense gates,
  and :func:`pepsy.operators.gates.gate_nonlocal_opt` for explicit ket/bra
  pairs and multi-site layers.
  The bare Quimb compressor names (for example ``"src"``, ``"srcmps"``,
  and ``"zipup"``) and their ``"quimb-"`` / ``"mpo-"`` spellings are
  accepted as mode aliases. Use ``"quimb-fit"`` or ``"mpo-fit"`` for
  Quimb's FIT compressor because bare ``"fit"`` remains the historical DMRG
  alias. Symmray MPOs use the block-aware SVD path instead.

The class also tracks a running "normalized-norm" proxy
``sqrt(<O|O> / <O0|O0>)`` that equals ``1`` for purely unitary two-sided
evolution (useful as a quick sanity signal). The absolute MPO norm is never
silently normalized: for an identity MPO on ``L`` qubits,
``<O|O> = 2**L`` and ``norm = sqrt(2**L)``.
"""

from __future__ import annotations

import math
import threading
import time
import warnings
import weakref
from copy import deepcopy
from functools import wraps
from dataclasses import dataclass, field
from numbers import Integral

import autoray as ar
import numpy as np

from ..._internal.random import backend_random_array
from ..._internal.quimb import require_quimb_1d_compression_method
from ..._internal.validation import normalize_integer_tuple
from ...backends import (
    backend_infer,
    infer_backend_converter_from_sample,
    infer_backend_signature,
)
from ...tensors.core import tn_fidelity, tn_norm
from ...fitting.local import FIT
from ...operators.gates import _normalize_gate_entries, gate as apply_gate, gate_nonlocal_opt

__all__ = ["MpoChannelEvent", "MpoOptimizer"]


def _replay_policy(method):
    """Scope optional validation and immutable geometry caches to one replay."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        check = kwargs.get("finite_check", False)
        legacy = kwargs.get("fit_finite_check")
        if legacy is not None:
            if "finite_check" in kwargs and check != legacy:
                raise ValueError("finite_check and fit_finite_check disagree.")
            check = legacy
        if check not in (None, False, True) and not callable(check):
            raise TypeError("finite_check must be bool, callable, or None.")
        kwargs["finite_check"] = check
        previous_check = self._finite_check_enabled
        previous_cache = self._replay_rank_cache
        previous_kinds = self._replay_array_kinds
        self._finite_check_enabled = check not in (None, False)
        self._replay_rank_cache = {}
        self._replay_array_kinds = weakref.WeakKeyDictionary()
        try:
            if self._finite_check_enabled:
                warnings.warn(
                    "MpoOptimizer finite_check is enabled: this optional diagnostic "
                    "is off by default. Tensor/norm checks add work and can "
                    "synchronize devices; use finite_check=False to avoid it.",
                    RuntimeWarning, stacklevel=2,
                )
            return method(self, *args, **kwargs)
        finally:
            self._finite_check_enabled = previous_check
            self._replay_rank_cache = previous_cache
            self._replay_array_kinds = previous_kinds
    return wrapped


# Keep this list aligned with MpsOptimizer's Quimb compression surface. These
# methods are MPO-applicable because ``gate_nonlocal_opt`` compresses the
# selected physical layer as a one-dimensional sub-MPO. MPS-only modes such as
# ``mix``, ``su``, ``perm``, and ``swap`` deliberately remain state-specific.
_MPO_COMPRESSION_METHODS = frozenset(
    {
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "sdc",
        "sdc-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    }
)
_FIT_INIT_STRATEGIES = frozenset(
    {"auto", "direct", "random", "random_expand", "svd_guess"}
    | {f"guess_{method}" for method in _MPO_COMPRESSION_METHODS}
)
_DEFAULT_FIT_INIT_STRATEGY = "guess_src"
_MPO_METHODS_IGNORE_CUTOFF_MODE = frozenset({"src", "srcmps"})
_MPO_METHODS_IGNORE_CUTOFF = frozenset({"src", "srcmps"})
_MPO_METHODS_USE_SEED = frozenset(
    {
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-oversample",
    }
)

# Keep the MPO run-level timing schema aligned with MpsOptimizer. These are
# compatibility totals and named subsets, not an additive partition.
_FIT_TIMING_PHASES = (
    "canonicalization_seconds",
    "sweep_preparation_canonicalization_seconds",
    "moving_canonicalization_seconds",
    "fixed_environment_seconds",
    "effective_seconds",
    "svd_seconds",
    "writeback_seconds",
    "environment_seconds",
    "moving_environment_seconds",
    "non_site_elapsed_seconds",
    "sweep_overhead_seconds",
)


def _summarize_fit_timing(records):
    """Summarize detailed FIT sweep timing without discarding raw records."""
    records = tuple(records)
    fit_indices = {
        int(record["fit_index"])
        for record in records
        if "fit_index" in record
    }
    return {
        "calls": len(fit_indices),
        "sweeps": len(records),
        "site_updates": sum(
            int(record.get("site_count", len(record.get("site_timings", ()))))
            for record in records
        ),
        "elapsed_seconds": sum(
            float(record.get("elapsed_seconds", 0.0)) for record in records
        ),
        **{
            phase: sum(float(record.get(phase, 0.0)) for record in records)
            for phase in _FIT_TIMING_PHASES
        },
    }


@dataclass(frozen=True)
class MpoChannelEvent:
    """A deterministic local channel-sum event for :class:`MpoOptimizer`.

    ``kraus`` contains local operator matrices ``K_i`` and the event applies
    ``sum_i weights[i] * K_i O K_i†``. ``semantics='sum'`` is the only
    executable MPO semantics: an MPO represents the deterministic channel
    sum. ``semantics='sample'`` is retained as an explicit declaration and is
    rejected at replay time because sampled branches belong to the MPS
    trajectory API.
    """

    kraus: tuple
    where: tuple
    weights: tuple | None = None
    labels: tuple | None = None
    semantics: str = "sum"

    def __post_init__(self):
        kraus = tuple(self.kraus)
        where = normalize_integer_tuple(self.where, name="where")
        if not kraus:
            raise ValueError("MpoChannelEvent needs at least one Kraus operator.")
        if len(where) not in {1, 2}:
            raise ValueError("MpoChannelEvent supports one- or two-site channels.")
        if len(set(where)) != len(where):
            raise ValueError("MpoChannelEvent support sites must be distinct.")
        matrices = []
        for operator in kraus:
            matrix = np.asarray(ar.to_numpy(operator))
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Kraus operators must be square matrices.")
            matrices.append(matrix)
        dimension = matrices[0].shape[0]
        if any(matrix.shape != (dimension, dimension) for matrix in matrices):
            raise ValueError("All Kraus operators must have the same dimension.")
        if len(where) == 2:
            local_dimension = int(round(np.sqrt(dimension)))
            if local_dimension * local_dimension != dimension:
                raise ValueError(
                    "Two-site channel matrices must have dimension d**2 by d**2."
                )
        semantics = str(self.semantics).strip().lower().replace("-", "_")
        if semantics not in {"sum", "sample"}:
            raise ValueError("channel semantics must be 'sum' or 'sample'.")
        weights = (
            tuple(1.0 for _ in matrices)
            if self.weights is None
            else tuple(float(weight) for weight in self.weights)
        )
        if len(weights) != len(matrices) or any(
            not math.isfinite(weight) or weight < 0.0 for weight in weights
        ):
            raise ValueError("channel weights must be finite and non-negative.")
        labels = (
            tuple(str(index) for index in range(len(matrices)))
            if self.labels is None
            else tuple(str(label) for label in self.labels)
        )
        if len(labels) != len(matrices) or len(set(labels)) != len(labels):
            raise ValueError("channel labels must be unique and match kraus operators.")
        object.__setattr__(self, "kraus", tuple(matrices))
        object.__setattr__(self, "where", where)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "semantics", semantics)

    @classmethod
    def from_channel(cls, channel, where, *, semantics="sum"):
        """Adapt a public ``TrajectoryChannel`` to deterministic MPO semantics."""
        outcomes = tuple(getattr(channel, "outcomes", ()))
        if not outcomes:
            raise TypeError("channel must expose non-empty outcomes.")
        weights = tuple(
            1.0
            if getattr(channel, "mode", "kraus") == "kraus"
            else float(outcome.probability)
            for outcome in outcomes
        )
        return cls(
            tuple(outcome.gate for outcome in outcomes),
            where,
            weights=weights,
            labels=tuple(outcome.label for outcome in outcomes),
            semantics=semantics,
        )


@dataclass(frozen=True)
class _MpoStreamPlan:
    """Immutable MPO stream metadata with a thread-safe payload cache."""

    entries: tuple
    gates: tuple
    where: tuple
    event_types: tuple[str, ...]
    arities: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    _prepared_cache: dict = field(default_factory=dict, compare=False, repr=False)
    _cache_lock: object = field(
        default_factory=threading.RLock,
        compare=False,
        repr=False,
    )

    def get_or_create(self, key, source, factory):
        """Return a prepared gate payload, invalidating stale source identities."""
        with self._cache_lock:
            cached = self._prepared_cache.get(key)
            if cached is not None and cached[0] is source:
                return cached[1]
            value = factory()
            self._prepared_cache[key] = (source, value)
            return value


def _normalize_gate_queue(gates):
    """Return ``(gate_list, where_list)`` from canonical bundled stream input."""
    if isinstance(gates, MpoChannelEvent):
        return [gates], [gates.where]
    if isinstance(gates, (tuple, list)) and any(
        isinstance(entry, MpoChannelEvent) for entry in gates
    ):
        entries = []
        for entry in gates:
            if isinstance(entry, MpoChannelEvent):
                entries.append((entry, entry.where))
            else:
                entries.extend(
                    _normalize_gate_entries((entry,), where=None, allow_empty=False)
                )
    else:
        entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
    if not entries:
        return [], []
    gate_list, where_list = zip(*entries)

    def normalize_where(where):
        if isinstance(where, Integral):
            return (int(where),)
        if isinstance(where, list):
            return tuple(where)
        return where

    return list(gate_list), [normalize_where(where) for where in where_list]


def _prepare_mpo_stream(gates):
    """Compile immutable MPO stream metadata once at queue boundaries."""
    gate_list, where_list = _normalize_gate_queue(gates)
    entries = tuple(zip(gate_list, where_list))
    event_types = []
    arities = []
    spans = []
    for gate, where in entries:
        if isinstance(gate, MpoChannelEvent):
            event_types.append(f"channel_{gate.semantics}")
            if len(where) not in {1, 2}:
                raise ValueError(
                    "MPO channel events must touch one or two sites."
                )
        else:
            event_types.append("gate")
        arity = len(where)
        if arity < 1:
            raise ValueError("Each MPO stream entry must touch at least one site.")
        arities.append(arity)
        spans.append((min(where), max(where)))
    return _MpoStreamPlan(
        entries=entries,
        gates=tuple(gate_list),
        where=tuple(where_list),
        event_types=tuple(event_types),
        arities=tuple(arities),
        spans=tuple(spans),
    )


class MpoOptimizer:
    """High-level wrapper for MPO gate sweeps.

    Parameters
    ----------
    mpo : qtn.MatrixProductOperator
        Initial MPO ``O``.  Used as the starting point for the queued
        evolution.  By default a copy is taken (see ``inplace``).
    gates : sequence | int | None, optional
        Canonical bundled gate stream ``[(gate, where), ...]``.  ``gate``
        encodes the (ket, bra) action per entry, with each side optionally
        ``None``:

        * ``G``            → apply ``G`` on ket and ``G†`` on bra (the MPO
          API's ``G O G†`` shorthand);
        * ``(G,)``         → ket-only shorthand for ``(G, None)``;
        * ``(G, None)``    → apply ``G`` on ket only;
        * ``(None, B)``    → apply ``B†`` on bra only;
        * ``(G, B)``       → apply ``G`` on ket and ``B†`` on bra.

        A :class:`MpoChannelEvent` is also accepted as a stream entry. It
        applies a deterministic weighted Kraus sum and records trace-
        preservation diagnostics separately from Hilbert-Schmidt norm
        survival. Sampled branch semantics are intentionally rejected here.

        For backward compatibility, passing a bare ``int`` is treated as
        ``chi`` with an empty gate queue.
    chi : int
        Working bond dimension used by all compression backends.
    mode : str, default="direct"
        Compression algorithm for operator evolution. ``"mpo"`` and
        ``"quimb"`` remain silent aliases for ``"direct"``. Bare
        Quimb compression methods such as ``"src"`` are accepted as aliases
        for ``"quimb-src"``. The dense Quimb path supports arbitrary
        one-dimensional gate supports; ``"svd"`` and ``"dmrg"`` retain
        their one- and two-site update paths.
    ind_id_k : str, default="k{}"
        Site-index format string for the ket physical leg family.
    ind_id_b : str, default="b{}"
        Site-index format string for the bra physical leg family.
    contraction_opt : object | None, optional
        Contraction path optimizer keyword used by :func:`tn_norm` and
        :class:`pepsy.fitting.local.FIT`.  Defaults to ``"auto-hq"``.
    inplace : bool, default=False
        When ``True`` mutate ``mpo`` directly; otherwise operate on a copy
        and leave the input untouched.

    Attributes
    ----------
    p : qtn.MatrixProductOperator
        Current MPO state (after construction and after each :meth:`run`).
    G, where : list
        Parsed gate-tensor list and corresponding site-coordinate list.
    losses : list[float]
        Running history of the normalized-norm proxy
        ``sqrt(<O|O> / <O0|O0>)`` appended at sampled steps during a run.
    info_c : dict
        Cached canonicalization metadata (``cur_orthog`` tracks the current
        orthogonality center / span).
    """

    _DMRG_MODE_ALIASES = {"dmrg1": 2, "dmrg2": 2, "dmrg3": 3}
    _ALLOWED_MODES = frozenset(
        {
            "dmrg",
            "dmrg1",
            "dmrg2",
            "dmrg3",
            "svd",
            "mpo",
            "quimb",
        }
        | _MPO_COMPRESSION_METHODS
        | {f"mpo-{method}" for method in _MPO_COMPRESSION_METHODS}
        | {f"quimb-{method}" for method in _MPO_COMPRESSION_METHODS}
    )

    @staticmethod
    def _is_symmray_array(value):
        """Return whether ``value`` is a Symmray block-sparse array."""
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @staticmethod
    def _is_fermionic_array(value):
        """Return whether ``value`` carries Symmray graded-array metadata."""
        return bool(getattr(value, "fermionic", False)) or (
            "fermionicarray" in type(value).__name__.lower()
        )

    @classmethod
    def _has_symmray_data(cls, tn):
        """Return whether any tensor in ``tn`` stores Symmray data."""
        return any(
            cls._is_symmray_array(getattr(tensor, "data", None))
            for tensor in getattr(tn, "tensors", ())
        )

    @staticmethod
    def _resolve_cutoff(value, p):
        """Resolve a numeric or dtype-aware truncation cutoff."""
        if value == "auto":
            dtype_names = [
                str(getattr(tensor.data, "dtype", "")).lower()
                for tensor in p.tensors
            ]
            if any("16" in dtype for dtype in dtype_names):
                return 1.0e-3
            if any("32" in dtype or "complex64" in dtype for dtype in dtype_names):
                return 1.0e-6
            return 1.0e-12
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
    def _resolve_fit_max_span(value, k_2q_batch):
        """Resolve the maximum inclusive spatial span of a DMRG batch."""
        if value is None:
            return None
        if value == "auto":
            return max(3, 2 * int(k_2q_batch) + 1)
        if not isinstance(value, Integral) or int(value) < 2:
            raise ValueError(
                "fit_max_span must be 'auto', None, or an integer >= 2."
            )
        return int(value)

    @classmethod
    def _normalize_mode(cls, mode):
        """Lower-case and validate ``mode`` against :attr:`_ALLOWED_MODES`."""
        mode_norm = str(mode).strip().lower()
        if mode_norm in {"mpo", "quimb"}:
            mode_norm = "direct"
        # Keep one maintained DMRG/FIT implementation while retaining the
        # requested named schedule separately in ``_dmrg_mode_alias``.
        if mode_norm == "fit" or mode_norm in cls._DMRG_MODE_ALIASES:
            mode_norm = "dmrg"
        elif mode_norm in _MPO_COMPRESSION_METHODS:
            mode_norm = f"quimb-{mode_norm}"
        if mode_norm not in cls._ALLOWED_MODES:
            supported = ", ".join(sorted(cls._ALLOWED_MODES))
            raise ValueError(f"Unknown mode: {mode}. Supported modes: {supported}")
        return mode_norm

    @classmethod
    def _is_mpo_mode(cls, mode):
        """Return whether ``mode`` selects Quimb sub-MPO compression."""
        mode_norm = str(mode).strip().lower()
        return (
            mode_norm in {"mpo", "quimb"}
            or mode_norm in _MPO_COMPRESSION_METHODS - {"fit"}
            or mode_norm.startswith(("mpo-", "quimb-"))
        )

    @classmethod
    def _mode_mpo_method(cls, mode):
        """Return the Quimb compressor encoded by an MPO mode name."""
        mode_norm = str(mode).strip().lower()
        if mode_norm in {"mpo", "quimb"}:
            return "direct"
        if mode_norm in _MPO_COMPRESSION_METHODS - {"fit"}:
            return cls._normalize_submpo_method(mode_norm)
        for prefix in ("quimb-", "mpo-"):
            if mode_norm.startswith(prefix):
                return cls._normalize_submpo_method(mode_norm[len(prefix) :])
        return "direct"

    def _resolve_mpo_method(self, method):
        """Resolve an explicit compressor or the current MPO mode."""
        if method is None:
            return self._mode_mpo_method(self.mode)
        return self._normalize_submpo_method(method)

    @classmethod
    def _normalize_submpo_method(cls, method):
        """Validate and normalize a Quimb sub-MPO compression method."""
        method_norm = str(method).strip().lower()
        if method_norm not in _MPO_COMPRESSION_METHODS:
            raise ValueError(f"Unknown subMPO method: {method}")
        require_quimb_1d_compression_method(method_norm)
        return method_norm

    @staticmethod
    def _submpo_compress_options(method, *, cutoff, cutoff_mode, max_bond, seed):
        """Build compressor options compatible with the selected Quimb method."""
        options = {
            "max_bond": max_bond,
            "cutoff": 0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else cutoff,
        }
        if cutoff_mode is not None and method not in _MPO_METHODS_IGNORE_CUTOFF_MODE:
            options["cutoff_mode"] = cutoff_mode
        if seed is not None and method in _MPO_METHODS_USE_SEED:
            options["seed"] = int(seed)
        return options

    @staticmethod
    def _validate_fit_init_strategy(strategy):
        """Normalize the FIT initial-guess construction policy."""
        strategy = str(strategy).strip().lower()
        if strategy.startswith("guess-"):
            strategy = "guess_" + strategy[len("guess-") :]
        strategy = {"mpo": "svd_guess"}.get(strategy, strategy)
        if strategy not in _FIT_INIT_STRATEGIES:
            raise ValueError(
                "fit_init_strategy must be one of 'auto', 'direct', "
                "'random', 'random_expand', or 'guess-<method>'."
            )
        return strategy

    @classmethod
    def _dmrg_alias_block_size(cls, mode):
        """Return the native FIT block size selected by a named mode."""
        return cls._DMRG_MODE_ALIASES.get(str(mode).strip().lower())

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mpo,
        gates=None,
        chi=None,
        mode="direct",
        ind_id_k="k{}",
        ind_id_b="b{}",
        contraction_opt=None,
        inplace=False,
    ):
        # Allow the shorthand ``MpoOptimizer(mpo, chi)``: bare int second arg
        # is interpreted as chi with an empty gate queue.
        if chi is None:
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use MpoOptimizer(mpo, gates, chi) "
                    "or MpoOptimizer(mpo, chi) for an empty gate queue."
                )

        self.inplace = bool(inplace)
        self._finite_check_enabled = False
        self._replay_rank_cache = None
        self._replay_array_kinds = None
        # Work on a copy by default so the user's input MPO stays unchanged.
        self.p = mpo if self.inplace else mpo.copy()
        self._stream_plan = _prepare_mpo_stream(gates)
        self.G = list(self._stream_plan.gates)
        self.where = list(self._stream_plan.where)
        if not isinstance(chi, Integral) or int(chi) < 1:
            raise ValueError("chi must be a positive integer.")
        self.chi = int(chi)
        mode_name = str(mode).strip().lower()
        self._dmrg_mode_alias = (
            mode_name if mode_name in self._DMRG_MODE_ALIASES else None
        )
        self.mode = self._normalize_mode(mode)
        self.ind_id_k = str(ind_id_k)
        self.ind_id_b = str(ind_id_b)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt

        # Reference norm used to normalize the compatibility norm proxy. The
        # structured norm ledger below tracks compression survival separately.
        self.norm_mpo = self._measure_norm(self.p)
        self._reference_log_norm = self._log_norm_from_measurement(self.norm_mpo)
        self.info_c = {}
        self.losses = [1.0]
        self.norm_events = []
        self._norm_log_survival = 0.0
        self.fit_diagnostics = []
        self._last_dmrg_fit_diagnostics = None
        self.last_run_timing = None
        self._timing_state = None
        self.last_run_status = "not_run"
        self.last_run_error = None
        self.last_run_fallback = None
        self.channel_events = []
        self.fallback_events = []
        self.trace_events = []
        self._dmrg1_one_site_locked = False
        self.logical_order = list(range(int(self.p.L)))
        self._persistent_layout_plan = None
        self.last_layout_plan = None
        self.backend = None
        self.backend_dtype = None
        self.backend_device = None
        self.array_backend = None
        self.backend_info()
        self._init_canonicalization()

    @staticmethod
    def _backend_info_for(p):
        """Return lightweight backend metadata for the supplied MPO."""
        try:
            return backend_infer(p)
        except (TypeError, ValueError) as exc:
            # MPO gate splits can legitimately leave real and complex
            # tensors together (for example, a native fermion MPO with real
            # diagonal sites). Only relax dtype agreement when every tensor
            # still has the same backend/device (and Symmray block backend).
            # Do not turn an actual mixed-backend state into a misleading
            # representative-backend report.
            tensors = tuple(getattr(p, "tensors", ()))
            if not tensors:
                raise exc
            signatures = tuple(
                infer_backend_signature(tensor.data)
                for tensor in tensors
            )
            first = signatures[0]
            same_backend = all(
                candidate[0] == first[0]
                and candidate[2] == first[2]
                and candidate[3:] == first[3:]
                for candidate in signatures[1:]
            )
            dtype_relaxation_allowed = first[0] in {"numpy", "symmray"}
            if not same_backend or not dtype_relaxation_allowed:
                raise exc
            return backend_infer(tensors[0].data)

    def backend_info(self):
        """Return the state-derived backend, dtype, and device metadata."""
        info = self._backend_info_for(self.p)
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        return info

    @staticmethod
    def _state_backend_like_for(p):
        """Return a representative raw array from an MPO's tensor data."""
        for tensor in getattr(p, "tensors", ()):
            return getattr(tensor, "data", None)
        return None

    def _state_backend_like(self):
        """Return a representative raw array from the live MPO."""
        return self._state_backend_like_for(self.p)

    def _to_state_backend(self, array):
        """Return ``array`` on the backend, dtype, and device of ``self.p``."""
        like = self._state_backend_like()
        if like is None:
            return np.asarray(ar.to_numpy(array), dtype=complex)

        target_signature = infer_backend_signature(like)
        source_signature = infer_backend_signature(array)
        if source_signature == target_signature:
            return array
        if self._is_symmray_array(array) and self._is_symmray_array(like):
            # Native Symmray payloads already carry charge and dual metadata.
            # A generic Autoray cast cannot safely recreate that structure.
            return array
        if target_signature[0] == "symmray" and source_signature[0] != "symmray":
            raise TypeError(
                "Cannot convert a dense gate/operator payload into a native "
                "Symmray MPO without charge and fermionic metadata. Build the "
                "payload as a Symmray array on the target U1/U1U1 backend."
            )

        converter = infer_backend_converter_from_sample(like)
        if converter is not None:
            return converter(array)
        if target_signature[0] == "numpy":
            return ar.to_numpy(array)
        # Keep the Autoray fallback for optional/custom dense backends.
        return ar.do("array", array, like=like)

    def to_backend(self, array):
        """Return ``array`` on the backend currently owned by ``self.p``.

        Already-compatible arrays are returned by identity. The converter is
        inferred from the live MPO, so replacing the state with :meth:`set_mpo`
        automatically changes the target backend without stale converter state.
        Numeric stream payloads remain explicit: call this helper before
        passing a gate or operator to :meth:`set_gates` or :meth:`run`.
        """
        return self._to_state_backend(array)

    @property
    def gate_stream(self):
        """Return the immutable compiled MPO gate/event stream."""
        return self._stream_plan.entries

    @property
    def stream_plan(self):
        """Return compiled stream metadata and cache statistics source."""
        return self._stream_plan

    def compile_gate_stream(self):
        """Return a reusable summary of the current compiled gate stream."""
        return {
            "entries": self._stream_plan.entries,
            "event_types": self._stream_plan.event_types,
            "arities": self._stream_plan.arities,
            "spans": self._stream_plan.spans,
            "length": len(self._stream_plan.entries),
            "prepared_cache_size": len(self._stream_plan._prepared_cache),
        }

    def clear_gate_cache(self):
        """Clear prepared gate payloads while retaining the compiled stream."""
        with self._stream_plan._cache_lock:
            self._stream_plan._prepared_cache.clear()
        return self

    def gate_stream_layout(self, *, sites=None, L=None, order="quality", **kwargs):
        """Find a physical MPO order that reduces long-range gate spans."""
        if any(event_type != "gate" for event_type in self._stream_plan.event_types):
            raise ValueError(
                "MPO layout planning currently requires an ordinary gate stream; "
                "plan channel streams separately."
            )
        from ..mps.layout import MpsGateStreamLayoutFinder

        finder = MpsGateStreamLayoutFinder(self.gate_stream, sites=sites, L=L)
        return finder.run(order=order, **kwargs)

    def _validate_layout_plan(self, plan):
        """Validate a layout plan against the MPO site positions."""
        length = int(self.p.L)
        site_order = tuple(plan.get("site_order", plan.get("qubit_inds", ())))
        if site_order != tuple(int(site) for site in site_order):
            raise ValueError("MPO layout sites must be integer positions.")
        if set(site_order) != set(range(length)) or len(site_order) != length:
            raise ValueError("MPO layout must be a permutation of range(mpo.L).")
        site_map = plan.get("site_map", plan.get("layout"))
        expected = {site: position for position, site in enumerate(site_order)}
        if not isinstance(site_map, dict) or dict(site_map) != expected:
            raise ValueError("MPO layout site_map must match site_order.")

    def _resolve_layout_plan(self, plan_or_order, layout_kwargs=None):
        """Resolve an explicit, finder-generated, or position layout plan."""
        if isinstance(plan_or_order, dict):
            plan = dict(plan_or_order)
        elif isinstance(plan_or_order, str):
            plan = self.gate_stream_layout(
                order=plan_or_order,
                **dict(layout_kwargs or {}),
            )
        else:
            site_order = tuple(int(site) for site in plan_or_order)
            plan = {
                "kind": "mpo_gate_stream_layout",
                "selected_order": "explicit",
                "site_order": site_order,
                "qubit_inds": site_order,
                "site_map": {
                    site: position for position, site in enumerate(site_order)
                },
            }
        self._validate_layout_plan(plan)
        return plan

    def apply_layout(
        self,
        plan_or_order="quality",
        *,
        cutoff=0.0,
        cutoff_mode="rsum2",
        allow_lossy_reorder=False,
        layout_kwargs=None,
    ):
        """Install a persistent physical MPO order using exact SWAP conjugations.

        The gate stream remains expressed in logical site labels; subsequent
        replay maps those labels through the installed ``site_map``. Reordering
        an already entangled MPO can require bond growth, so lossy compression
        during this one-time operation is opt-in.
        """
        if self._persistent_layout_plan is not None:
            raise ValueError("an MPO layout is already installed on this optimizer.")
        plan = self._resolve_layout_plan(plan_or_order, layout_kwargs)
        if self.p.max_bond() > 1 and not allow_lossy_reorder:
            raise ValueError(
                "reordering an entangled MPO requires allow_lossy_reorder=True."
            )
        import quimb as qu

        cutoff = self._resolve_cutoff(cutoff, self.p)
        current_order = list(range(int(self.p.L)))
        for target_position, logical_site in enumerate(plan["site_order"]):
            current_position = current_order.index(int(logical_site))
            while current_position > target_position:
                left = current_position - 1
                expected = self._canonical_norm_measurement(self.p)
                swap = qu.swap()
                self._apply_gate_pair(
                    self.p,
                    swap,
                    (left, current_position),
                    bra_gate=swap,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract="reduce-split",
                    inplace=True,
                )
                self.canonize_mpo(self.p, (left, current_position))
                for site in range(current_position, left, -1):
                    self.p.right_canonize_site(site, bra=None)
                self.p.left_compress(
                    start=left,
                    stop=current_position,
                    max_bond=self.chi,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                )
                self.info_c["cur_orthog"] = (current_position, current_position)
                observed = self._canonical_norm_measurement(
                    self.p, center=current_position
                )
                self._record_norm_event(
                    "layout_swap",
                    expected_norm=expected,
                    observed_norm=observed,
                    target_norm=expected,
                    where=(left, current_position),
                    unitary=self._unitary_norm_guard_supported(self.p),
                )
                current_order[left], current_order[current_position] = (
                    current_order[current_position],
                    current_order[left],
                )
                current_position -= 1
        self._persistent_layout_plan = plan
        self.logical_order = list(plan["site_order"])
        self.last_layout_plan = plan
        return self

    def _execution_stream(self):
        """Return the compiled stream mapped into the installed MPO order."""
        if self._persistent_layout_plan is None:
            return list(self.G), list(self.where)
        site_map = self._persistent_layout_plan["site_map"]
        gates = []
        wheres = []
        for gate, where in self._stream_plan.entries:
            mapped = tuple(site_map[int(site)] for site in where)
            if isinstance(gate, MpoChannelEvent):
                gate = MpoChannelEvent(
                    gate.kraus,
                    mapped,
                    weights=gate.weights,
                    labels=gate.labels,
                    semantics=gate.semantics,
                )
            gates.append(gate)
            wheres.append(mapped)
        return gates, wheres

    @staticmethod
    def channel_event(channel, where, *, semantics="sum"):
        """Create an MPO channel event from ``TrajectoryChannel``-like input."""
        return MpoChannelEvent.from_channel(channel, where, semantics=semantics)

    @staticmethod
    def kraus_event(kraus, where, *, labels=None, weights=None, semantics="sum"):
        """Create a deterministic or explicitly rejected sampled Kraus event."""
        return MpoChannelEvent(
            tuple(kraus),
            where,
            labels=labels,
            weights=weights,
            semantics=semantics,
        )

    def _current_orthog(self, p=None, *, info=None):
        """Return cached ``(min_site, max_site)`` orthogonality span.

        Accepts cached entries shaped as ``"calc"`` / ``None`` (recompute),
        ``int`` (single site), or 1- and 2-tuples.  The canonical form
        returned span is always a 2-tuple with ``min <= max``. Only the live
        MPO uses ``self.info_c``; disposable targets have independent metadata.
        """
        state = self.p if p is None else p
        if info is None:
            info = self.info_c if state is self.p else {}
        cur = info.get("cur_orthog", "calc")
        if cur == "calc" or cur is None:
            lo, hi = state.calc_current_orthog_center()
            cur = (int(lo), int(hi))
        elif isinstance(cur, Integral):
            cur = (int(cur), int(cur))
        elif len(cur) == 1:
            cur = (int(cur[0]), int(cur[0]))
        elif len(cur) == 2:
            cur = (int(min(cur)), int(max(cur)))
        else:
            raise ValueError("cur_orthog must be an int, (int,), or (int, int).")

        info["cur_orthog"] = cur
        return cur

    def _init_canonicalization(self):
        """Put ``self.p`` into mixed-canonical form with center at ``L // 2``."""
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    def _prepare_dmrg_state(self, fit_block_size=1):
        """Prepare the MPO for the selected local FIT update.

        The historical dense one-site FIT path has fixed bond dimensions and
        still needs the old eager expansion. Native Symmray paths cannot use
        that dense padding, and native two- and three-site FIT splits discover
        the required bond sectors themselves, so native paths never pre-expand
        their multi-sector bonds.
        """
        if int(fit_block_size) != 1 or self._has_symmray_data(self.p):
            return
        if self.p.max_bond() < self.chi:
            self.p.expand_bond_dimension(self.chi, inplace=True)
            self._init_canonicalization()

    def set_mpo(self, mpo):
        """Assign a new MPO and reset canonicalization metadata."""
        if self._replay_rank_cache is not None:
            self._replay_rank_cache.clear()
        self.p = mpo if self.inplace else mpo.copy()
        if not isinstance(self.chi, Integral) or self.chi < 1:
            raise ValueError("chi must be a positive integer.")
        self.norm_mpo = self._measure_norm(self.p)
        self._reference_log_norm = self._log_norm_from_measurement(self.norm_mpo)
        self.losses = [1.0]
        self.norm_events = []
        self._norm_log_survival = 0.0
        self.fit_diagnostics = []
        self._last_dmrg_fit_diagnostics = None
        self.last_run_timing = None
        self._timing_state = None
        self.last_run_status = "not_run"
        self.last_run_error = None
        self.last_run_fallback = None
        self.channel_events = []
        self.fallback_events = []
        self.trace_events = []
        self._dmrg1_one_site_locked = False
        self.logical_order = list(range(int(self.p.L)))
        self._persistent_layout_plan = None
        self.last_layout_plan = None
        self.backend_info()
        self._init_canonicalization()
        return self

    def set_mode(self, mode):
        """Set execution mode."""
        if self._replay_rank_cache is not None:
            self._replay_rank_cache.clear()
        old_mode = self.mode
        old_alias = self._dmrg_mode_alias
        mode_name = str(mode).strip().lower()
        new_alias = (
            mode_name if mode_name in self._DMRG_MODE_ALIASES else None
        )
        new_mode = self._normalize_mode(mode)
        self._dmrg_mode_alias = new_alias
        self.mode = new_mode
        if old_mode != new_mode or old_alias != new_alias:
            self._dmrg1_one_site_locked = False
        return self

    def set_gates(self, gates):
        """Replace the current gate queue with canonical bundled entries."""
        self._stream_plan = _prepare_mpo_stream(gates)
        self.G = list(self._stream_plan.gates)
        self.where = list(self._stream_plan.where)
        return self

    def add_gates(self, gates):
        """Append canonical bundled entries to the existing gate queue."""
        new_plan = _prepare_mpo_stream(gates)
        self._stream_plan = _prepare_mpo_stream(
            self._stream_plan.entries + new_plan.entries
        )
        self.G = list(self._stream_plan.gates)
        self.where = list(self._stream_plan.where)
        return self

    @staticmethod
    def _normalize_fidelity_samples(fidelity_samples):
        """Validate and normalize fidelity-sample count."""
        samples = int(fidelity_samples)
        if samples < 0:
            raise ValueError("fidelity_samples must be >= 0.")
        return samples

    @staticmethod
    def _sampling_steps(total_steps, fidelity_samples):
        """Return the set of gate-step indices at which to sample the norm proxy.

        Indices are 1-based gate counts (``1 ≤ step ≤ total_steps``).  The
        final step is always included so the run history ends with a fresh
        measurement; up to ``fidelity_samples`` extra interior points are
        spread linearly across the remaining range.
        """
        if total_steps <= 0:
            return set()

        samples = MpoOptimizer._normalize_fidelity_samples(fidelity_samples)
        sample_steps = set()

        if total_steps > 1 and samples > 0:
            interior_count = min(samples, total_steps - 1)
            for step in np.linspace(1, total_steps - 1, num=interior_count, dtype=int):
                sample_steps.add(int(step))

        sample_steps.add(total_steps)
        return sample_steps

    @staticmethod
    def _real_float(value):
        """Convert backend scalar/tensor-like values to Python float (real part)."""
        real_value = ar.do("real", value)
        item = getattr(real_value, "item", None)
        if callable(item):
            try:
                real_value = item()
            except TypeError:
                pass
        return float(real_value)

    @staticmethod
    def _log_norm_from_measurement(norm_val):
        """Return the natural log of an MPO norm measurement.

        ``tn_norm(..., strip_exponent=True)`` returns a mantissa/exponent pair
        for the squared norm. Keeping the logarithm here avoids reconstructing
        huge or tiny norms merely to form a relative ratio.
        """
        mantissa, exponent = norm_val
        mantissa = float(abs(mantissa))
        exponent = float(exponent)
        if mantissa == 0.0:
            return -math.inf
        if not math.isfinite(mantissa) or not math.isfinite(exponent):
            return math.nan
        return 0.5 * (math.log(mantissa) + exponent * math.log(10.0))

    @staticmethod
    def _is_norm_measurement(value):
        """Return whether ``value`` is a ``(mantissa, exponent)`` pair."""
        return isinstance(value, tuple) and len(value) == 2

    @staticmethod
    def _log_norm_value_from_measurement(norm_val):
        """Return the natural log of a Frobenius norm measurement."""
        mantissa, exponent = norm_val
        mantissa = float(abs(mantissa))
        exponent = float(exponent)
        if mantissa == 0.0:
            return -math.inf
        if not math.isfinite(mantissa) or not math.isfinite(exponent):
            return math.nan
        return math.log(mantissa) + exponent * math.log(10.0)

    @classmethod
    def _norm_measurement_to_value(cls, norm_val):
        """Reconstruct a finite-safe Frobenius norm from its scaled pair."""
        return cls._exp_from_log(cls._log_norm_value_from_measurement(norm_val))

    @classmethod
    def _norm_measurement_from_squared(cls, norm_val):
        """Convert a scaled squared norm into a scaled Frobenius norm."""
        mantissa, exponent = norm_val
        mantissa = float(abs(mantissa))
        exponent = float(exponent)
        if mantissa == 0.0:
            return 0.0, 0.0
        return math.sqrt(mantissa), 0.5 * exponent

    @staticmethod
    def _exp_from_log(value):
        """Exponentiate a log value while preserving useful overflow semantics."""
        value = float(value)
        if math.isnan(value):
            return math.nan
        if value == -math.inf:
            return 0.0
        if value >= math.log(np.finfo(float).max):
            return math.inf
        if value <= math.log(np.nextafter(0.0, 1.0)):
            return 0.0
        return float(math.exp(value))

    @staticmethod
    def _network_exponent(p):
        """Return a Quimb network exponent as a safe float."""
        try:
            exponent = float(getattr(p, "exponent", 0.0) or 0.0)
        except (TypeError, ValueError):
            exponent = 0.0
        return exponent

    def _canonical_norm_measurement(self, p, center=None):
        """Return ``(mantissa, exponent)`` for the represented MPO norm.

        Dense MPOs use the norm of a tracked canonical center tensor and keep
        Quimb's network exponent separate. Native Symmray or fallback paths
        contract the full squared norm with ``strip_exponent=True``. In both
        cases the returned value is a scaled Frobenius norm, so event ratios
        never need to reconstruct the physical norm first.
        """
        try:
            if self._has_symmray_data(p):
                return self._norm_measurement_from_squared(self._measure_norm(p))
            if center is None:
                # Native block-sparse MPOs can make
                # ``calc_current_orthog_center`` fall through an
                # allclose-to-identity check that densifies a very large
                # virtual tensor. The optimizer already maintains a valid
                # center after every compression, so prefer that cache for the
                # live state and only discover a center for disposable MPOs.
                if p is self.p:
                    center = self.info_c.get("cur_orthog")
                if center in (None, "calc"):
                    center = p.calc_current_orthog_center()
            if isinstance(center, Integral):
                site = int(center)
                span = (site, site)
            else:
                span = tuple(int(value) for value in center)
                site = int(span[-1])
                span = (min(span), max(span))
            if span[0] != span[1]:
                p.canonize([site], cur_orthog=span)
            norm = self._real_float(ar.do("abs", p[site].norm()))
            measurement = (norm, self._network_exponent(p))
            if p is self.p:
                self.info_c["cur_orthog"] = (site, site)
            return measurement
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return self._norm_measurement_from_squared(self._measure_norm(p))

    def _canonical_norm_value(self, p, center=None):
        """Return the represented MPO norm from a canonical center when possible."""
        return self._norm_measurement_to_value(
            self._canonical_norm_measurement(p, center=center)
        )

    def _append_norm_proxy_sample(self, p):
        """Append current normalized MPO norm and return it."""
        measurement = self._canonical_norm_measurement(p)
        log_norm = self._log_norm_value_from_measurement(measurement)
        norm_val = self._exp_from_log(log_norm - self._reference_log_norm)
        self.losses.append(norm_val)
        return norm_val

    def _measure_norm(self, p):
        """Return ``(mantissa, exponent)`` such that ``<O|O> = mantissa * 10**exponent``.

        Working in log-mantissa form keeps the proxy stable for very long
        gate streams where ``<O|O>`` can over- or under-flow.
        """
        mantissa, exponent = tn_norm(p, contraction_opt=self.contraction_opt, strip_exponent=True)
        return self._real_float(mantissa), float(exponent)

    def _normalize_norm(self, norm_val):
        """Convert a ``(mantissa, exponent)`` squared-norm into the relative MPO norm.

        Returns ``sqrt(<O|O>) / sqrt(<O0|O0>)`` — i.e. the ratio of actual
        MPO norms relative to the construction-time reference ``norm_mpo``.
        For purely unitary two-sided evolution this stays equal to ``1``.
        """
        log_norm = self._log_norm_from_measurement(norm_val)
        return self._exp_from_log(log_norm - self._reference_log_norm)

    @staticmethod
    def _fidelity_ratio_from_norms(observed_norm, expected_norm):
        """Return raw and clipped fidelity measured from two norms."""
        observed_norm = float(abs(observed_norm))
        expected_norm = float(abs(expected_norm))
        if (
            expected_norm <= 0.0
            or not math.isfinite(expected_norm)
            or not math.isfinite(observed_norm)
        ):
            return None, None
        raw = (observed_norm / expected_norm) ** 2
        return raw, min(1.0, max(0.0, raw))

    @classmethod
    def _fidelity_ratio_from_measurements(cls, observed_norm, expected_norm):
        """Return raw and clipped fidelity from two scaled norm pairs."""
        observed_log = cls._log_norm_value_from_measurement(observed_norm)
        expected_log = cls._log_norm_value_from_measurement(expected_norm)
        if expected_log == -math.inf or not math.isfinite(expected_log):
            return None, None
        if observed_log == -math.inf:
            return 0.0, 0.0
        if not math.isfinite(observed_log):
            return None, None
        raw = cls._exp_from_log(2.0 * (observed_log - expected_log))
        return raw, min(1.0, max(0.0, raw))

    def _unitary_norm_overshoot_tolerance(self):
        """Return a dtype-aware tolerance for retained-norm overshoots."""
        backend = self.backend_info()
        dtype = str(backend.get("dtype", "")).lower()
        if backend.get("backend") == "symmray":
            # Graded block contractions can accumulate a little more
            # roundoff than dense canonical-center measurements, even when
            # the exact block-sparse norm is used.
            return 1.0e-5
        if "32" in dtype or "complex64" in dtype:
            return max(1.0e-6, 128.0 * np.finfo(np.float32).eps)
        return 1.0e-6

    def _unitary_norm_guard_supported(self, p):
        """Return whether local center norms support the overshoot guard."""
        # Symmray states are measured through the exact block-sparse network
        # norm in ``_canonical_norm_value`` rather than a sector-normalized
        # center tensor, so the same consistency check is valid for both
        # dense and native MPOs.
        return True

    @staticmethod
    def _norm_squared_value(value):
        """Return a finite-safe squared MPO norm for diagnostics."""
        value = float(abs(value))
        if not math.isfinite(value):
            return value
        return float(value * value)

    def _record_norm_event(
        self,
        kind,
        *,
        expected_norm,
        observed_norm,
        where=(),
        target_norm=None,
        unitary=False,
        physical_boundary=False,
        renormalized=None,
    ):
        """Record automatic MPO compression norm survival for one segment.

        The expected norm is measured from the disposable target before FIT or
        direct compression. A physical norm change therefore does not appear
        as compression infidelity; only the retained/expected norm ratio is
        accumulated. Norm fields retain their absolute MPO scale, including
        the ``2**L`` squared norm of an identity MPO.
        """
        expected_measurement = (
            expected_norm if self._is_norm_measurement(expected_norm) else None
        )
        observed_measurement = (
            observed_norm if self._is_norm_measurement(observed_norm) else None
        )
        target_measurement = (
            target_norm if self._is_norm_measurement(target_norm) else None
        )
        if expected_measurement is not None and observed_measurement is not None:
            raw, survival = self._fidelity_ratio_from_measurements(
                observed_measurement,
                expected_measurement,
            )
        else:
            raw, survival = self._fidelity_ratio_from_norms(
                observed_norm,
                expected_norm,
            )
        expected_value = (
            self._norm_measurement_to_value(expected_measurement)
            if expected_measurement is not None
            else float(abs(expected_norm))
        )
        observed_value = (
            self._norm_measurement_to_value(observed_measurement)
            if observed_measurement is not None
            else float(abs(observed_norm))
        )
        target_value = (
            self._norm_measurement_to_value(target_measurement)
            if target_measurement is not None
            else None if target_norm is None else float(abs(target_norm))
        )
        if self._finite_check_enabled and unitary and raw is not None:
            overshoot_tolerance = self._unitary_norm_overshoot_tolerance()
            if raw > 1.0 + overshoot_tolerance:
                raise FloatingPointError(
                    "Retained unitary-compression norm exceeds its expected "
                    f"norm (squared ratio={raw:.6g}, "
                    f"tolerance={overshoot_tolerance:.3g}); "
                    "canonical projection metadata is inconsistent."
                )
        if survival is not None:
            if survival == 0.0:
                self._norm_log_survival = -math.inf
            elif math.isfinite(self._norm_log_survival):
                self._norm_log_survival += math.log(survival)
            cumulative_fidelity = self._cumulative_fidelity()
            cumulative_infidelity = self._cumulative_infidelity()
        else:
            cumulative_fidelity = None
            cumulative_infidelity = None

        event = {
            "kind": str(kind),
            "where": tuple(int(site) for site in where),
            "valid": raw is not None,
            "expected_norm": (
                None if raw is None else expected_value
            ),
            "expected_norm_sq": (
                None
                if raw is None
                else self._norm_squared_value(expected_value)
            ),
            "observed_norm": (
                None if raw is None else observed_value
            ),
            "observed_norm_sq": (
                None
                if raw is None
                else self._norm_squared_value(observed_value)
            ),
            "target_norm": (
                target_value
            ),
            "target_norm_sq": (
                None if target_value is None else self._norm_squared_value(target_value)
            ),
            "expected_norm_mantissa": (
                None
                if expected_measurement is None
                else float(abs(expected_measurement[0]))
            ),
            "expected_norm_exponent": (
                None
                if expected_measurement is None
                else float(expected_measurement[1])
            ),
            "observed_norm_mantissa": (
                None
                if observed_measurement is None
                else float(abs(observed_measurement[0]))
            ),
            "observed_norm_exponent": (
                None
                if observed_measurement is None
                else float(observed_measurement[1])
            ),
            "target_norm_mantissa": (
                None
                if target_measurement is None
                else float(abs(target_measurement[0]))
            ),
            "target_norm_exponent": (
                None
                if target_measurement is None
                else float(target_measurement[1])
            ),
            "fidelity_raw": None if raw is None else float(raw),
            # These are compression fidelities measured from norms. The
            # metric name intentionally does not repeat its measurement source.
            "local_fidelity": None if survival is None else float(survival),
            "local_infidelity": (
                None if survival is None else float(1.0 - survival)
            ),
            "branch_probability": None,
            "physical_boundary": bool(physical_boundary),
            "renormalized": (
                None if renormalized is None else bool(renormalized)
            ),
            "cumulative_fidelity": cumulative_fidelity,
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": cumulative_fidelity,
            "cumulative_compression_infidelity": cumulative_infidelity,
        }
        self.norm_events.append(event)
        return event

    def _cumulative_fidelity(self):
        """Return cumulative fidelity measured from retained norms."""
        return self._exp_from_log(self._norm_log_survival)

    def _cumulative_infidelity(self):
        """Return cumulative infidelity using stable ``expm1``."""
        if self._norm_log_survival == -math.inf:
            return 1.0
        if not math.isfinite(self._norm_log_survival):
            return math.nan
        return float(-math.expm1(self._norm_log_survival))

    def norm_diagnostics(self):
        """Return compression fidelity and separately scaled MPO diagnostics.

        ``norm`` and ``state_norm`` are the represented Frobenius norm
        ``sqrt(<O|O>)`` rather than a unit-normalized progress value. Their
        squared companions therefore retain the physical Hilbert-space scale;
        for an identity MPO on ``L`` qubits, ``norm_sq == 2**L``. The local and
        cumulative fidelity fields are ratios of expected and observed norms,
        so this scale cancels without being discarded from the diagnostics.
        """
        valid = [event for event in self.norm_events if event["valid"]]
        cumulative_fidelity = (
            None if not valid else self._cumulative_fidelity()
        )
        cumulative_infidelity = (
            None if cumulative_fidelity is None else self._cumulative_infidelity()
        )
        current = None if not valid else valid[-1]
        state_norm = self._canonical_norm_value(self.p)
        segment_survivals = [float(event["local_fidelity"]) for event in valid]
        segment_infidelities = [
            float(event["local_infidelity"]) for event in valid
        ]
        if segment_survivals and any(value <= 0.0 for value in segment_survivals):
            geometric_survival = 0.0
        elif segment_survivals:
            geometric_survival = float(
                math.exp(
                    sum(math.log(value) for value in segment_survivals)
                    / len(segment_survivals)
                )
            )
        else:
            geometric_survival = None
        physical = [
            event for event in valid if event.get("physical_boundary", False)
        ]
        return {
            "tracking": True,
            "norm_tracking": True,
            "truncation_tracking": None,
            "events": len(self.norm_events),
            "completed_events": len(valid),
            "completed_segments": len(valid),
            "segments_including_current": len(valid),
            "completed_segment_norms": [
                float(max(0.0, value) ** 0.5) for value in segment_survivals
            ],
            "completed_segment_infidelities": segment_infidelities,
            "current_valid": current is not None,
            "current_event": None if current is None else deepcopy(current),
            "current_fidelity": (
                None if current is None else current["local_fidelity"]
            ),
            "current_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "local_fidelity": (
                None if current is None else current["local_fidelity"]
            ),
            "local_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "cumulative_fidelity": cumulative_fidelity,
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": cumulative_fidelity,
            "cumulative_compression_infidelity": cumulative_infidelity,
            "norm_survival": cumulative_fidelity,
            "fidelity": cumulative_fidelity,
            "infidelity": cumulative_infidelity,
            "norm": state_norm,
            "state_norm": state_norm,
            "norm_sq": self._norm_squared_value(state_norm),
            "state_norm_sq": self._norm_squared_value(state_norm),
            "cumulative_norm": (
                None
                if cumulative_fidelity is None
                else float(cumulative_fidelity ** 0.5)
            ),
            "total_survival_proxy": cumulative_fidelity,
            "total_infidelity_proxy": cumulative_infidelity,
            "total_norm_proxy": (
                None
                if cumulative_fidelity is None
                else float(cumulative_fidelity ** 0.5)
            ),
            "geometric_mean_survival": geometric_survival,
            "geometric_mean_norm": (
                None
                if geometric_survival is None
                else float(geometric_survival ** 0.5)
            ),
            "mean_segment_infidelity": (
                None
                if not segment_infidelities
                else float(sum(segment_infidelities) / len(segment_infidelities))
            ),
            "segment_infidelities": [
                event["local_infidelity"] for event in valid
            ],
            "max_segment_infidelity": (
                None
                if not valid
                else max(event["local_infidelity"] for event in valid)
            ),
            "current_event_kind": None if current is None else current["kind"],
            "current_segment_norm": (
                None
                if current is None
                else float(max(0.0, current["local_fidelity"]) ** 0.5)
            ),
            "current_segment_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "physical_boundary_events": len(physical),
            "physical_boundary_infidelities": [
                event["local_infidelity"] for event in physical
            ],
        }

    def get_trace_events(self):
        """Return channel trace-preservation records."""
        return deepcopy(self.trace_events)

    def channel_diagnostics(self):
        """Return channel-sum and trace-preservation diagnostics."""
        events = self.get_trace_events()
        return {
            "events": len(events),
            "channel_events": deepcopy(self.channel_events),
            "fallback_events": deepcopy(self.fallback_events),
            "trace_events": events,
            "trace_preserving": all(
                event["channel_trace_preserving"] for event in events
            ),
            "max_trace_preservation_residual": (
                0.0
                if not events
                else max(event["trace_preservation_residual"] for event in events)
            ),
        }

    @staticmethod
    def _prepare_gate_tensor(gate, n_sites):
        """Reorder a gate tensor into ``(input, output)`` ket-index order.

        Quimb gates are stored as ``(output, input)`` matrices (or tensors
        with all output axes followed by all input axes). ``apply_gate``
        below expects the opposite ordering, so we transpose accordingly.
        """
        # Native Symmray gates already carry explicit dual metadata describing
        # output and input legs. Transposing them as if they were dense Quimb
        # gates changes the charge sectors (and can make a U1U1 split ask for
        # impossible virtual charges). Keep their graded/block structure intact.
        if MpoOptimizer._is_symmray_array(gate):
            return gate

        shape = tuple(getattr(gate, "shape", ()))
        if len(shape) == 2:
            if int(shape[0]) != int(shape[1]):
                raise ValueError(
                    f"{n_sites}-site gate matrix must be square, got {shape}."
                )
            return ar.do("transpose", gate, (1, 0))
        if len(shape) == 2 * n_sites:
            return ar.do(
                "transpose",
                gate,
                tuple(range(n_sites, 2 * n_sites)) + tuple(range(n_sites)),
            )
        raise ValueError(
            f"{n_sites}-site gate must have a square matrix or "
            f"{2 * n_sites}-axis tensor shape, got {shape}."
        )

    @classmethod
    def _symmray_physical_map(cls, p, site, ind_id):
        """Return the dense charge list for one live MPO physical index."""
        ind = ind_id.format(site)
        tensor = next(
            tensor
            for tensor in getattr(p, "tensors", ())
            if ind in getattr(tensor, "inds", ())
        )
        axis = tensor.inds.index(ind)
        chargemap = tensor.data.indices[axis].chargemap
        return [charge for charge, size in chargemap.items() for _ in range(int(size))]

    @staticmethod
    def _charge_parity(charge):
        """Return fermion parity inferred from a scalar or product charge."""
        if isinstance(charge, tuple):
            return sum(int(part) for part in charge) % 2
        return int(charge) % 2

    @classmethod
    def _fermionic_gate_to_bosonic(cls, p, gate, where, ind_id):
        """Convert an even native gate for a non-graded Symmray MPO.

        ``SymHamiltonian.to_mpo`` deliberately returns a bosonic Symmray MPO
        with the Jordan-Wigner parity convention already encoded in its local
        channels. Native ``Fermion`` gates use Symmray's graded tensor-product
        convention instead. For an even two-site gate, changing conventions
        amounts to the endpoint crossing phase on the input ket sectors.
        """
        if not cls._is_fermionic_array(gate) or not cls._has_symmray_data(p):
            return gate

        sample = next(
            tensor.data
            for tensor in getattr(p, "tensors", ())
            if cls._is_symmray_array(getattr(tensor, "data", None))
        )
        if cls._is_fermionic_array(sample):
            return gate

        try:
            dense = gate.to_dense()
        except AttributeError:
            dense = gate
        dense = np.asarray(ar.to_numpy(dense))
        where = tuple(where)
        physical_maps = [
            cls._symmray_physical_map(p, site, ind_id) for site in where
        ]
        n_sites = len(where)
        if n_sites == 2:
            left_odd = np.array(
                [cls._charge_parity(charge) for charge in physical_maps[0]],
                dtype=bool,
            )
            right_odd = np.array(
                [cls._charge_parity(charge) for charge in physical_maps[1]],
                dtype=bool,
            )
            crossing = np.ones(
                (len(left_odd), len(right_odd)),
                dtype=dense.dtype,
            )
            crossing[np.ix_(left_odd, right_odd)] = -1
            dense = dense * crossing[None, None, :, :]

        import symmray.utils as sr_utils  # pylint: disable=import-outside-toplevel

        sample_charge = next(iter(physical_maps[0]), 0)
        zero = (
            tuple(0 for _ in sample_charge)
            if isinstance(sample_charge, tuple)
            else 0
        )
        return sr_utils.from_dense(
            dense,
            symmetry=getattr(sample, "symmetry", None),
            index_maps=physical_maps * 2,
            duals=(False,) * n_sites + (True,) * n_sites,
            fermionic=False,
            charge=zero,
        )

    @classmethod
    def _prepare_gate_pair(cls, gate, n_sites, bra_gate=None, *, p=None, where=None,
                           ind_id="k{}"):
        """Return ``(g_k, g_b)`` ready to be fed to :func:`apply_gate`.

        ``gate`` becomes ``g_k`` (acts on the ket index family with the
        ket-ordering convention) and ``bra_gate`` becomes
        ``g_b = conj(prepare(bra_gate))``, which when applied to the bra
        family realises ``B† O`` on that side.  Passing ``None`` skips that
        side; at least one of the two must be provided.
        """
        if gate is None and bra_gate is None:
            raise ValueError("At least one of ket gate or bra gate must be provided.")

        if p is not None and where is not None:
            gate = (
                None
                if gate is None
                else cls._fermionic_gate_to_bosonic(p, gate, where, ind_id)
            )
            bra_gate = (
                None
                if bra_gate is None
                else cls._fermionic_gate_to_bosonic(p, bra_gate, where, ind_id)
            )

        g_k = None if gate is None else cls._prepare_gate_tensor(gate, n_sites)
        if bra_gate is None:
            g_b = None
        else:
            g_b = ar.do("conj", cls._prepare_gate_tensor(bra_gate, n_sites))
        return g_k, g_b

    @classmethod
    def _prepare_nonlocal_gate_pair(
        cls, gate, n_sites, bra_gate=None, *, p=None, where=None, ind_id="k{}"
    ):
        """Prepare gates for :func:`gate_nonlocal_opt`.

        ``gate_nonlocal_opt`` uses a different dense gate convention for the
        lower MPO layer than :func:`apply_gate`. The ket payload has the same
        input/output reordering as the regular gate path, but the lower
        payload must be the elementwise-conjugated raw bra operator. Reusing
        :meth:`_prepare_gate_pair` here silently transposes that bra payload
        a second time and gives incorrect results for nonsymmetric complex
        gates.
        """
        if gate is None and bra_gate is None:
            raise ValueError("At least one of ket gate or bra gate must be provided.")

        if p is not None and where is not None:
            gate = (
                None
                if gate is None
                else cls._fermionic_gate_to_bosonic(p, gate, where, ind_id)
            )
            bra_gate = (
                None
                if bra_gate is None
                else cls._fermionic_gate_to_bosonic(p, bra_gate, where, ind_id)
            )

        # The upper layer needs the input/output reordering performed by
        # ``_prepare_gate_tensor``. For the lower layer, ``from_dense``
        # expects ``conj(B)`` directly, not ``conj(B.T)``.
        g_k = None if gate is None else cls._prepare_gate_tensor(gate, n_sites)
        g_b = None if bra_gate is None else ar.do("conj", bra_gate)
        return g_k, g_b

    @classmethod
    def _materialize_split_gate(cls, p, where):
        """Contract lazy native split-gate tensors back into site tensors.

        Quimb's ``split-gate`` application is deliberately lazy: for a
        non-local gate it temporarily leaves an extra tensor at each endpoint
        carrying the site tag. That is useful for general tensor-network
        workflows, but ``MatrixProductOperator.right_canonize_site`` expects
        exactly one tensor per site. Native Symmray arrays cannot use the
        dense swap-gate fallback, so materialize the endpoint tensors before
        the MPO canonicalization/compression sweep.
        """
        if not cls._has_symmray_data(p):
            return p

        for site in set(where):
            tag = f"I{site}"
            if len(p.tag_map.get(tag, ())) > 1:
                p.contract_tags(
                    tag,
                    preserve_tensor=True,
                    inplace=True,
                )
        return p

    def _build_mpo_target(
        self,
        p,
        gate,
        where,
        bra_gate,
        cutoff,
        cutoff_mode="rsum2",
    ):
        """Build an uncapped regular MPO target for one gate pair."""
        p_g = p.copy()
        if len(where) == 1:
            self._apply_gate_pair(
                p_g,
                gate,
                where,
                bra_gate=bra_gate,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                contract=True,
                inplace=True,
            )
            return p_g

        return self._compress_mpo_gate_pair(
            p_g,
            gate,
            where,
            bra_gate,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            layer_order="upper_lower",
            max_bond=None,
        )

    def _can_use_native_gate_sandwich(self, p, gate, bra_gate, where, method):
        """Return whether Quimb's native two-site MPO sandwich is applicable."""
        return (
            method == "direct"
            and not self._has_symmray_data(p)
            and len(where) == 2
            and gate is not None
            and bra_gate is gate
            and hasattr(p, "gate_sandwich_with_auto_swap")
        )

    def _apply_native_gate_sandwich(
        self,
        p,
        gate,
        where,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Apply a bare MPO gate through Quimb's dagger-aware auto-swap path.

        Pepsy's public gate convention stores dense gates in output/input
        order, while the existing MPO replay contract applies the equivalent
        transposed operator to the represented dense matrix. Passing the
        conjugated payload with Quimb's ``dagger=True`` preserves that public
        convention while letting Quimb handle both physical layers, swaps,
        canonical-center updates, and the bra conjugation itself.
        """
        return p.gate_sandwich_with_auto_swap(
            ar.do("conj", gate),
            tuple(where),
            dagger=True,
            info=self.info_c,
            swap_back=True,
            strip_exponent=False,
            contract="split",
            inplace=True,
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )

    def _compress_mpo_gate_pair(
        self,
        p,
        gate,
        where,
        bra_gate,
        *,
        cutoff,
        cutoff_mode="rsum2",
        layer_order="upper_lower",
        max_bond=None,
        method="direct",
        seed=None,
    ):
        """Apply and compress both physical MPO layers in a chosen order."""
        method = self._normalize_submpo_method(method)
        g_k, g_b = self._prepare_nonlocal_gate_pair(
            gate,
            len(where),
            bra_gate=bra_gate,
            p=p,
            where=where,
            ind_id=self.ind_id_k,
        )
        prepared = {"upper": g_k, "lower": g_b}
        if layer_order == "lower_upper":
            layers = ("lower", "upper")
        elif layer_order == "upper_lower":
            layers = ("upper", "lower")
        else:
            raise ValueError(
                "layer_order must be 'lower_upper' or 'upper_lower'."
            )
        for which in layers:
            payload = prepared[which]
            if payload is None:
                continue
            compress_options = self._submpo_compress_options(
                method,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                max_bond=max_bond,
                seed=seed,
            )
            p = gate_nonlocal_opt(
                p,
                payload,
                where,
                which=which,
                method=method,
                info={},
                inplace=True,
                ind_id_k=self.ind_id_k,
                ind_id_b=self.ind_id_b,
                **compress_options,
            )
        return p

    def _build_mpo_fit_guess(
        self,
        p,
        batch_G,
        batch_where,
        *,
        cutoff,
        cutoff_mode="rsum2",
        layer_order="lower_upper",
        method="src",
        seed=None,
    ):
        """Build a capped compressed MPO replay for a local FIT guess.

        The replay is deliberately isolated from both the exact FIT target and
        the live MPO. ``method`` therefore controls only the disposable warm
        start, while FIT still receives the uncapped target built by the DMRG
        target path.
        """
        active = [site for where in batch_where for site in where]
        guess = self._copy_working_state(p, (min(active), max(active)))
        active_sites = []
        for index, (G_i, where_i) in enumerate(zip(batch_G, batch_where)):
            gate, bra_gate, where = self._parse_gate_entry(G_i, where_i)
            active_sites.extend(where)
            if len(where) == 1:
                self._apply_gate_pair(
                    guess,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
            else:
                guess = self._compress_mpo_gate_pair(
                    guess,
                    gate,
                    where,
                    bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    layer_order=layer_order,
                    max_bond=self.chi,
                    method=method,
                    seed=None if seed is None else int(seed) + index,
                )
        if active_sites:
            xmin, xmax = min(active_sites), max(active_sites)
            guess.canonize([xmax], cur_orthog=(xmin, xmax))
        return guess

    @staticmethod
    def _fit_random_data(data, shape, *, strength, rng):
        """Generate deterministic random data on the tensor's backend."""
        dtype_name = str(getattr(data, "dtype", "float64")).lower()
        if "complex64" in dtype_name:
            dtype = np.complex64
        elif "complex" in dtype_name:
            dtype = np.complex128
        elif "float32" in dtype_name:
            dtype = np.float32
        else:
            dtype = np.float64
        return backend_random_array(
            shape,
            like=data,
            dtype=dtype,
            scale=float(strength),
            rng=rng,
        )

    def _build_randomized_fit_guess(
        self,
        p,
        where,
        *,
        block_size,
        rand_strength,
        expand=True,
        seed=0,
    ):
        """Prepare a dense MPO FIT guess with deterministic random data.

        The exact target is always constructed from the unmodified live MPO.
        Random data is restricted to the disposable guess: ``random``
        perturbs existing active tensors, while ``random_expand`` also adds
        directions on active bonds below their physical/``chi`` ceiling.
        Native Symmray data is left to its block-aware FIT warm start because
        dense random padding would destroy charge-sector metadata.
        """
        info = {
            "enabled": False,
            "rand_strength": float(rand_strength),
            "bonds": [],
            "sites": [],
            "expanded": bool(expand),
            "reason": None,
        }
        if int(block_size) not in {2, 3}:
            info["reason"] = "one_site_fit"
            return p, info
        if self._has_symmray_data(p) or any(
            self._is_fermionic_array(tensor.data) for tensor in p
        ):
            info["reason"] = "native_sector_growth"
            return p, info
        if float(rand_strength) == 0.0:
            info["reason"] = "disabled"
            return p, info

        xmin, xmax = min(where), max(where)
        guess = p.copy()
        rng = np.random.default_rng(int(seed))
        bonds = []
        if expand:
            target_sizes = FIT._active_bond_rank_targets(  # pylint: disable=protected-access
                p,
                xmin,
                xmax,
                self.chi,
            )
            if target_sizes is None:
                info["reason"] = "no_active_rank_targets"
                return p, info
            for site, target_size in zip(range(xmin, xmax), target_sizes):
                current_size = int(p.bond_size(site, site + 1))
                target_size = int(target_size)
                if current_size < target_size:
                    bonds.append((site, current_size, target_size))
            if not bonds:
                info["reason"] = "already_at_target"
                return p, info

            import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

            by_target = {}
            for site, current_size, target_size in bonds:
                by_target.setdefault(target_size, []).append(
                    (site, current_size, target_size)
                )
            for target_size, target_bonds in by_target.items():
                bond_inds = [
                    guess.bond(site, site + 1)
                    for site, _, _ in target_bonds
                ]
                qtn.TensorNetwork.expand_bond_dimension(
                    guess,
                    target_size,
                    mode="zeros",
                    inds_to_expand=bond_inds,
                    inplace=True,
                )
                for site, current_size, _ in target_bonds:
                    bond = guess.bond(site, site + 1)
                    for tensor in guess.tensors:
                        if bond not in tensor.inds:
                            continue
                        axis = tensor.inds.index(bond)
                        old_slices = [slice(None)] * tensor.ndim
                        old_slices[axis] = slice(0, current_size)
                        old_data = tensor.data[tuple(old_slices)]
                        random_shape = list(tensor.shape)
                        random_shape[axis] = target_size - current_size
                        random_data = self._fit_random_data(
                            tensor.data,
                            random_shape,
                            strength=rand_strength,
                            rng=rng,
                        )
                        tensor.modify(
                            data=ar.do(
                                "concatenate",
                                (old_data, random_data),
                                axis=axis,
                            )
                        )
        else:
            for site in range(xmin, xmax + 1):
                tensor = guess[site]
                random_data = self._fit_random_data(
                    tensor.data,
                    tensor.shape,
                    strength=rand_strength,
                    rng=rng,
                )
                tensor.modify(data=ar.do("add", tensor.data, random_data))
                info["sites"].append(int(site))

        guess.canonize([xmax], cur_orthog=(xmin, xmax))
        info["enabled"] = True
        info["bonds"] = [
            {
                "bond": int(site),
                "current_rank": int(current_size),
                "target_rank": int(target_size),
                "new_rank": int(guess.bond_size(site, site + 1)),
            }
            for site, current_size, target_size in bonds
        ]
        return guess, info

    def _prepare_fit_initial_guess(
        self,
        p,
        gates,
        wheres,
        *,
        block_size,
        strategy,
        fit_mpo_guess,
        rand_strength,
        seed,
        cutoff,
        cutoff_mode,
        layer_order="lower_upper",
    ):
        """Select a disposable FIT initial guess for an MPO update."""
        requested_strategy = self._validate_fit_init_strategy(strategy)
        random_info = {
            "enabled": False,
            "rand_strength": float(rand_strength),
            "bonds": [],
            "sites": [],
            "expanded": False,
            "reason": "direct",
        }
        result = {
            "fit_guess": p,
            "strategy": "direct",
            "requested_strategy": requested_strategy,
            "guess_method": None,
            "guess_used": False,
            "svd_guess_used": False,
            "guess_backend": None,
            "random_initialization": random_info,
        }

        # Symmray/fermionic MPOs must retain their native sectors. The native
        # FIT kernel already grows and splits those sectors safely, so a dense
        # Quimb source or random guess is not a valid substitution here.
        if self._has_symmray_data(p) or any(
            self._is_fermionic_array(tensor.data) for tensor in p
        ):
            random_info["reason"] = "native_sector_growth"
            return result

        if requested_strategy == "auto":
            selected_strategy = _DEFAULT_FIT_INIT_STRATEGY
        else:
            selected_strategy = requested_strategy

        # Preserve the established legacy switch: disabling ``fit_mpo_guess``
        # disables the implicit default source guess for named schedules, but
        # never overrides an explicit fit_init_strategy choice.
        is_named_window = self._dmrg_mode_alias in {"dmrg1", "dmrg2", "dmrg3"}
        raw_strategy = str(strategy).strip().lower()
        if (
            not fit_mpo_guess
            and is_named_window
            and raw_strategy in {"auto", _DEFAULT_FIT_INIT_STRATEGY, "guess-src"}
        ):
            selected_strategy = "direct"

        if selected_strategy == "svd_guess":
            guess_method = "direct"
        elif selected_strategy.startswith("guess_"):
            guess_method = selected_strategy[len("guess_") :]
        else:
            guess_method = None

        if guess_method is not None:
            fit_guess = self._build_mpo_fit_guess(
                p,
                gates,
                wheres,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                layer_order=layer_order,
                method=guess_method,
                seed=seed,
            )
            result.update(
                fit_guess=fit_guess,
                strategy=selected_strategy,
                guess_method=guess_method,
                guess_used=True,
                svd_guess_used=True,
                guess_backend=f"quimb-{guess_method}",
            )
            random_info["reason"] = selected_strategy
            return result

        if selected_strategy in {"random", "random_expand"}:
            start = min(site for where in wheres for site in where)
            stop = max(site for where in wheres for site in where)
            fit_guess, random_info = self._build_randomized_fit_guess(
                p,
                (start, stop),
                block_size=block_size,
                rand_strength=rand_strength,
                expand=selected_strategy == "random_expand",
                seed=int(seed),
            )
            result["fit_guess"] = fit_guess
            result["strategy"] = selected_strategy
            result["random_initialization"] = random_info

        return result

    @staticmethod
    def _parse_gate_entry(G_i, where_i):
        """Decompose one stream entry into ``(ket_gate, bra_gate, where)``.

        Acceptable shapes for ``G_i`` mirror the constructor convention:

        * bare ``G``        → ``(G, G)``  ("unitary conjugation" default);
        * ``(G,)``          → ``(G, None)`` (ket-only);
        * ``(G, B)``        → explicit pair, either side may be ``None``.

        At least one of the two sides must be non-``None``.
        """
        where_norm = tuple(where_i)
        if not where_norm:
            raise ValueError("Each gate location must contain at least one site.")

        if isinstance(G_i, (tuple, list)):
            if len(G_i) == 1:
                # Explicit ket-only shorthand: (G,) -> (G, None)
                gate, bra_gate = G_i[0], None
            elif len(G_i) == 2:
                gate, bra_gate = G_i
            else:
                raise ValueError("Each MPO gate entry must be G, (G,), or (G, B).")
        else:
            # Default MPO evolution applies U on ket and U† on bra.
            gate, bra_gate = G_i, G_i

        if gate is None and bra_gate is None:
            raise ValueError("Each gate entry must provide at least one of G or B.")
        return gate, bra_gate, where_norm

    def _apply_gate_pair(
        self,
        p,
        gate,
        where,
        bra_gate=None,
        *,
        cutoff,
        cutoff_mode="rsum2",
        contract,
        inplace=True,
    ):
        """Apply the (ket, bra) gate pair onto ``p`` using :func:`apply_gate`.

        Each side is applied independently with its own ``ind_id_*`` so the
        two index families stay decoupled.
        """
        n_sites = len(where)
        if p is self.p and n_sites == 1 and not self._is_unitary_gate_pair(gate, bra_gate):
            # Non-unitary left/right multiplication invalidates an off-center
            # isometry. Move the center before acting, preserving the absolute
            # operator scale; unitary one-site updates keep their old center.
            self.canonize_mpo(p, where)
        needs_state_adaptation = self._has_symmray_data(p) or any(
            self._is_fermionic_array(operator)
            for operator in (gate, bra_gate)
            if operator is not None
        )
        if needs_state_adaptation:
            g_k, g_b = self._prepare_gate_pair(
                gate,
                n_sites,
                bra_gate=bra_gate,
                p=p,
                where=where,
                ind_id=self.ind_id_k,
            )
        else:
            key = (
                "pair",
                id(gate),
                id(bra_gate),
                n_sites,
                self.ind_id_k,
                self.ind_id_b,
            )
            source = gate if gate is not None else bra_gate
            g_k, g_b = self._stream_plan.get_or_create(
                key,
                source,
                lambda: self._prepare_gate_pair(
                    gate,
                    n_sites,
                    bra_gate=bra_gate,
                ),
            )

        if g_k is not None:
            apply_gate(
                p,
                g_k,
                where,
                ind_id=self.ind_id_k,
                contract=contract,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=inplace,
            )
        if g_b is not None:
            apply_gate(
                p,
                g_b,
                where,
                ind_id=self.ind_id_b,
                contract=contract,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=inplace,
            )

        if contract == "split-gate":
            self._materialize_split_gate(p, where)

    def _build_dmrg_target(
        self,
        p,
        gate,
        where,
        bra_gate,
        cutoff,
        cutoff_mode="rsum2",
        *,
        target_cutoff=None,
        target_strategy="auto",
    ):
        """Return ``p`` with one two-site gate pair applied via ``split-gate``.

        The result is the *target* MPO that the local FIT update will fit
        back to bond dimension ``chi`` inside the gate window.
        """
        if target_cutoff is None:
            target_cutoff = cutoff
        target_strategy = self._validate_fit_target_strategy(target_strategy)
        if target_strategy == "auto":
            target_strategy = (
                "mpo" if self._has_symmray_data(p) else "layered"
            )
        if target_strategy == "layered" and self._has_symmray_data(p):
            raise ValueError(
                "fit_target_strategy='layered' is only available for dense MPOs; "
                "use 'auto' or 'mpo' for native Symmray data."
            )
        if target_strategy == "mpo" and not self._has_symmray_data(p):
            return self._build_mpo_target(
                p,
                gate,
                where,
                bra_gate,
                target_cutoff,
                cutoff_mode,
            )
        p_g = p.copy()
        self._apply_gate_pair(
            p_g,
            gate,
            where,
            bra_gate=bra_gate,
            cutoff=target_cutoff,
            cutoff_mode=cutoff_mode,
            contract="split-gate",
            inplace=True,
        )
        return p_g

    @staticmethod
    def _collect_dmrg_batch(
        G_seq,
        where_seq,
        start_idx,
        k_2q_batch,
        *,
        max_span=None,
    ):
        """Greedily collect up to ``k_2q_batch`` consecutive two-site gates.

        Any one-site gates encountered along the way are folded into the
        same batch so they are applied together inside a single FIT window.
        Returns ``(batch_G, batch_where, n_two_qubit_in_batch, next_idx)``.
        """
        batch_G = []
        batch_where = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(G_seq) and two_qubit_in_batch < k_2q_batch:
            where = tuple(where_seq[idx])
            gate = G_seq[idx]
            if isinstance(gate, MpoChannelEvent):
                break
            if max_span is not None and batch_where:
                sites = [site for previous in batch_where for site in previous]
                sites.extend(where)
                proposed_span = max(sites) - min(sites) + 1
                if proposed_span > int(max_span):
                    break
            if len(where) == 1:
                batch_where.append(where)
                batch_G.append(gate)
            elif len(where) == 2:
                batch_where.append(where)
                batch_G.append(gate)
                two_qubit_in_batch += 1
            else:
                raise ValueError("Each gate location must have one or two sites.")
            idx += 1

        return batch_G, batch_where, two_qubit_in_batch, idx

    def _build_dmrg_batch_target(
        self,
        p,
        batch_G,
        batch_where,
        cutoff,
        cutoff_mode="rsum2",
        *,
        target_cutoff=None,
        target_strategy="auto",
    ):
        """Apply a collected DMRG batch onto a copy of ``p``.

        Used to materialise the local target MPO for a batched FIT update.
        Two-site gates are split (``split-gate``) and one-site gates are
        contracted directly.
        """
        if target_cutoff is None:
            target_cutoff = cutoff
        target_strategy = self._validate_fit_target_strategy(target_strategy)
        if target_strategy == "auto":
            target_strategy = "mpo" if self._has_symmray_data(p) else "layered"
        if target_strategy == "layered" and self._has_symmray_data(p):
            raise ValueError(
                "fit_target_strategy='layered' is only available for dense MPOs; "
                "use 'auto' or 'mpo' for native Symmray data."
            )
        p_g = p.copy()
        for G_i, where_i in zip(batch_G, batch_where):
            gate, bra_gate, where = self._parse_gate_entry(G_i, where_i)
            if (
                target_strategy == "mpo"
                and not self._has_symmray_data(p_g)
                and len(where) == 2
            ):
                p_g = self._build_mpo_target(
                    p_g,
                    gate,
                    where,
                    bra_gate,
                    target_cutoff,
                    cutoff_mode,
                )
                continue
            contract = True if len(where) == 1 else "split-gate"
            self._apply_gate_pair(
                p_g,
                gate,
                where,
                bra_gate=bra_gate,
                cutoff=target_cutoff,
                cutoff_mode=cutoff_mode,
                contract=contract,
                inplace=True,
            )
        return p_g

    @staticmethod
    def _validate_fit_target_strategy(strategy):
        """Normalize the MPO FIT target representation policy."""
        strategy = str(strategy).strip().lower()
        if strategy not in {"auto", "layered", "mpo"}:
            raise ValueError(
                "fit_target_strategy must be 'auto', 'layered', or 'mpo'."
            )
        return strategy

    @staticmethod
    def _validate_fit_mpo_guess_order(order):
        """Normalize the layer order used by an MPO FIT initial guess."""
        order = str(order).strip().lower().replace("-", "_")
        aliases = {
            "lower_upper": "lower_upper",
            "lower_then_upper": "lower_upper",
            "bra_ket": "lower_upper",
            "bra_then_ket": "lower_upper",
            "upper_lower": "upper_lower",
            "upper_then_lower": "upper_lower",
            "ket_bra": "upper_lower",
            "ket_then_bra": "upper_lower",
        }
        try:
            return aliases[order]
        except KeyError as exc:
            raise ValueError(
                "fit_mpo_guess_order must select bra/lower then ket/upper "
                "('lower_upper') or ket/upper then bra/lower ('upper_lower')."
            ) from exc

    @staticmethod
    def _is_unitary_gate(gate):
        """Return whether ``gate`` is a numerically unitary operator.

        This is deliberately a small-gate check. It lets the norm ledger use
        the already canonical live MPO as the expected-norm measurement for
        the common ``U O U†`` path, avoiding a second target contraction. If
        the gate cannot be inspected cheaply, return ``False`` so explicit
        target construction remains the conservative choice.
        """
        if gate is None:
            return False
        try:
            dense = gate.to_dense()
        except AttributeError:
            dense = gate
        try:
            dense = np.asarray(ar.to_numpy(dense))
        except (TypeError, ValueError):
            return False
        if dense.ndim == 4:
            dense = dense.reshape(
                int(np.prod(dense.shape[:2])),
                int(np.prod(dense.shape[2:])),
            )
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            return False
        try:
            gram = dense.conj().T @ dense
            return bool(
                np.allclose(
                    gram,
                    np.eye(dense.shape[0], dtype=dense.dtype),
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False

    @classmethod
    def _is_unitary_gate_pair(cls, gate, bra_gate):
        """Return whether a ket/bra gate pair preserves the MPO norm.

        Left or right multiplication by a unitary preserves the
        Hilbert--Schmidt norm, as does the usual ``U O U†`` pair.  In
        particular, an explicit ket-only ``(U, None)`` entry should not force
        construction and contraction of a disposable target MPO merely for
        norm diagnostics.
        """
        ket_unitary = gate is None or cls._is_unitary_gate(gate)
        bra_unitary = bra_gate is None or cls._is_unitary_gate(bra_gate)
        return ket_unitary and bra_unitary

    def _expected_target_norm(
        self,
        p,
        gate,
        where,
        bra_gate,
        *,
        target=None,
        target_cutoff=0.0,
        cutoff_mode="rsum2",
        target_strategy="auto",
    ):
        """Measure the expected post-gate norm as a scaled pair.

        A bare unitary gate (the default MPO API meaning) preserves the MPO
        Hilbert-Schmidt norm before compression. In that case the live
        canonical center is the exact expected norm. Explicit ket/bra pairs
        and non-unitary gates use the materialized target, so physical norm
        changes are still separated from truncation loss.
        """
        if self._is_unitary_gate_pair(gate, bra_gate):
            return self._canonical_norm_measurement(p)
        if target is None:
            target = self._build_dmrg_target(
                p,
                gate,
                where,
                bra_gate,
                cutoff=0.0,
                cutoff_mode=cutoff_mode,
                target_cutoff=target_cutoff,
                target_strategy=target_strategy,
            )
        return self._canonical_norm_measurement(target)

    def _expected_batch_target_norm(
        self,
        p,
        batch_G,
        batch_where,
        *,
        target,
        cutoff_mode="rsum2",
    ):
        """Measure a batched target norm as a scaled pair."""
        all_unitary = True
        for G_i, where_i in zip(batch_G, batch_where):
            gate, bra_gate, _ = self._parse_gate_entry(G_i, where_i)
            if not self._is_unitary_gate_pair(gate, bra_gate):
                all_unitary = False
                break
        if all_unitary:
            return self._canonical_norm_measurement(p)
        return self._canonical_norm_measurement(target)

    def _rank_targets(self, p, start, stop):
        """Cache immutable MPO geometry while keeping exterior ranks live."""
        if stop <= start:
            return ()
        cache = self._replay_rank_cache
        key = (int(p.L), int(self.chi))
        dims = None if cache is None else cache.get(key)
        if dims is None:
            dims = tuple(
                int(p.ind_size(p.upper_ind(site))) * int(p.ind_size(p.lower_ind(site)))
                for site in range(p.L)
            )
            if cache is not None:
                cache[key] = dims
        left = int(p.bond_size(start - 1, start)) if start else 1
        right = int(p.bond_size(stop, stop + 1)) if stop + 1 < p.L else 1
        left_caps = []
        for site in range(start, stop):
            left = min(self.chi, left * dims[site])
            left_caps.append(left)
        targets = list(left_caps)
        for site in range(stop, start, -1):
            right = min(self.chi, right * dims[site])
            targets[site - start - 1] = min(targets[site - start - 1], right)
        return tuple(targets)

    def _bonds_at_rank_targets(self, p, start, stop):
        return all(
            int(p.bond_size(site, site + 1)) >= target
            for site, target in zip(range(start, stop), self._rank_targets(p, start, stop))
        )

    def _resolve_dmrg_fit_block_size(self, p, xmin, xmax, requested):
        """Resolve the live native FIT block for one MPO target window.

        ``dmrg1`` starts directly in its one-site phase when all bonds in a
        non-adjacent active window already have their attainable rank. This
        mirrors the MPS schedule and avoids repeating growth decompositions
        after the MPO has reached its local bond ceiling. If a backend does
        not expose the rank helpers for an MPO, retaining the requested block
        is the safe fallback.
        """
        active = min(int(requested), int(xmax) - int(xmin) + 1)
        if (
            self._dmrg_mode_alias != "dmrg1"
            or active != 2
        ):
            return active
        if self._dmrg1_one_site_locked:
            return 1
        if int(xmax) - int(xmin) < 2:
            return active
        try:
            at_target = self._bonds_at_rank_targets(
                p,
                int(xmin),
                int(xmax),
            )
        except (AttributeError, TypeError, ValueError):
            at_target = False
        return 1 if at_target else active

    def _dmrg1_all_bonds_at_rank_targets(self):
        """Return whether every MPO bond has reached its attainable ceiling."""
        if self._dmrg_mode_alias != "dmrg1":
            return False
        length = int(getattr(self.p, "L", 0))
        if length <= 1:
            return True
        try:
            targets = self._rank_targets(
                self.p,
                0,
                length - 1,
            )
            return all(
                int(self.p.bond_size(site, site + 1)) >= int(target)
                for site, target in enumerate(targets)
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def _maybe_lock_dmrg1_one_site_phase(self):
        """Latch DMRG1 into one-site updates after full-chain saturation."""
        if self._dmrg_mode_alias != "dmrg1":
            return False
        if not self._dmrg1_one_site_locked and self._dmrg1_all_bonds_at_rank_targets():
            self._dmrg1_one_site_locked = True
        return self._dmrg1_one_site_locked

    def _validate_dmrg1_iteration_budget(self, p, xmin, xmax, *, n_iter, block_size):
        """Require two growth sweeps plus refinement for uncapped DMRG1."""
        if self._dmrg_mode_alias != "dmrg1" or int(block_size) != 2 or int(n_iter) >= 3:
            return
        if int(xmax) - int(xmin) < 2:
            return
        try:
            at_target = self._bonds_at_rank_targets(
                p,
                int(xmin),
                int(xmax),
            )
        except (AttributeError, TypeError, ValueError):
            at_target = False
        if at_target or int(n_iter) >= 3:
            return
        raise ValueError(
            "mode='dmrg1' requires n_iter >= 3 for an under-capacity "
            "window: two two-site growth sweeps and at least one "
            "one-site refinement sweep."
        )

    def _fit_overlap_diagnostics(self, target, fitted):
        """Return an optional direct overlap readout against a FIT target."""
        contraction_opt = self.contraction_opt
        if contraction_opt is None or (
            isinstance(contraction_opt, str)
            and contraction_opt.strip().lower() in {"auto", "auto-hq"}
        ):
            contraction_opt = "greedy"
        try:
            overlap = tn_fidelity(
                target.copy(),
                fitted.copy(),
                contraction_opt=contraction_opt,
            )
            overlap = float(ar.do("real", overlap))
            if not math.isfinite(overlap):
                raise ValueError("FIT overlap is non-finite.")
            return {
                "fit_overlap_fidelity": overlap,
                "fit_overlap_infidelity": max(0.0, 1.0 - overlap),
                "fit_overlap_error": None,
            }
        except Exception as exc:  # diagnostic only; FIT result remains valid
            return {
                "fit_overlap_fidelity": None,
                "fit_overlap_infidelity": None,
                "fit_overlap_error": f"{type(exc).__name__}: {exc}",
            }

    def _record_fit_diagnostics(
        self,
        fit,
        *,
        where,
        block_size,
        step,
        mpo_fit_guess_used=False,
        mpo_fit_guess_order=None,
        fit_initialization=None,
        fit_overlap=None,
        fit_overlap_diagnostics=False,
    ):
        """Store a compact diagnostic record for the latest MPO FIT call."""
        fit_initialization = dict(fit_initialization or {})
        fit_overlap = dict(fit_overlap or {})
        record = {
            "step": int(step),
            "where": tuple(int(site) for site in where),
            "block_size": int(block_size),
            "iterations": int(getattr(fit, "iterations_run", 0)),
            "converged": bool(getattr(fit, "converged", False)),
            "convergence_reason": getattr(fit, "convergence_reason", None),
            "relative_change": getattr(fit, "last_relative_change", None),
            "center_site": getattr(fit, "final_center_site", None),
            "direction": getattr(fit, "final_direction", None),
            "final_norm": getattr(fit, "final_norm", None),
            "adaptive_sweeps": int(getattr(fit, "adaptive_sweeps_run", 0)),
            "one_site_refinement_sweeps": int(
                getattr(fit, "one_site_sweeps_run", 0)
            ),
            "mpo_fit_guess_used": bool(mpo_fit_guess_used),
            "mpo_fit_guess_order": mpo_fit_guess_order,
            "svd_guess_used": bool(
                fit_initialization.get("svd_guess_used", False)
            ),
            "fit_init_strategy": fit_initialization.get(
                "strategy", "direct"
            ),
            "fit_init_strategy_requested": fit_initialization.get(
                "requested_strategy", "direct"
            ),
            "guess_used": bool(fit_initialization.get("guess_used", False)),
            "guess_method": fit_initialization.get("guess_method"),
            "guess_backend": fit_initialization.get("guess_backend"),
            "random_initialization": fit_initialization.get(
                "random_initialization",
                {
                    "enabled": False,
                    "reason": "direct",
                },
            ),
            "fit_overlap_diagnostics": bool(fit_overlap_diagnostics),
            "fit_overlap_fidelity": fit_overlap.get("fit_overlap_fidelity"),
            "fit_overlap_infidelity": fit_overlap.get("fit_overlap_infidelity"),
            "fit_overlap_error": fit_overlap.get("fit_overlap_error"),
        }
        timing_records = getattr(fit, "_pepsy_timing_records", None)
        if timing_records is None:
            take_timing_records = getattr(fit, "_take_timing_records", None)
            timing_records = (
                take_timing_records() if callable(take_timing_records) else []
            )
        record["timing"] = timing_records
        self.fit_diagnostics.append(record)
        self._last_dmrg_fit_diagnostics = record
        return record

    def _run_fit_gate(self, fit, **kwargs):
        """Run FIT and transfer its opt-in timing records to the MPO run."""
        fit._finite_check_warning_handled = self._finite_check_enabled
        if self._timing_state is None:
            return fit.run_gate(**kwargs)

        kwargs.setdefault("timing", True)
        kwargs.setdefault(
            "timing_sync_device",
            bool(self._timing_state.get("sync_device", False)),
        )
        fit_index = int(self._timing_state["fit_call_count"])
        self._timing_state["fit_call_count"] += 1
        try:
            return self._timed_call("dmrg.fit", fit.run_gate, **kwargs)
        finally:
            take_timing_records = getattr(fit, "_take_timing_records", None)
            records = take_timing_records() if callable(take_timing_records) else []
            fit._pepsy_timing_records = records
            for timing_record in records:
                timing_record["fit_index"] = fit_index
                timing_record["record_index"] = len(
                    self._timing_state["fit_steps"]
                )
                self._timing_state["fit_steps"].append(timing_record)

    @staticmethod
    def _finite_mpo(p):
        """Return whether every dense or block-sparse MPO array is finite."""
        for tensor in getattr(p, "tensors", ()):
            data = getattr(tensor, "data", None)
            arrays = getattr(data, "blocks", None)
            if isinstance(arrays, dict):
                arrays = arrays.values()
            elif arrays is None:
                arrays = (data,)
            for array in arrays:
                try:
                    finite = ar.do("all", ar.do("isfinite", array))
                    if not bool(ar.to_numpy(finite)):
                        return False
                except (TypeError, ValueError, AttributeError):
                    try:
                        if not np.all(np.isfinite(np.asarray(array))):
                            return False
                    except (TypeError, ValueError):
                        return False
        return True

    def _check_finite(self, finite_check, p=None):
        """Run the requested MPO finite-data check after a replay step."""
        if finite_check in (None, False):
            return
        state = self.p if p is None else p
        if callable(finite_check):
            valid = bool(finite_check(state))
        else:
            valid = self._finite_mpo(state)
        if not valid:
            raise FloatingPointError(
                "MPO replay produced non-finite tensor data."
            )

    @staticmethod
    def _measure_trace(p):
        """Return the real trace of an MPO, with a scalar backend fallback."""
        value = p.trace()
        return float(np.real(ar.to_numpy(value)))

    @staticmethod
    def _relative_trace_residual(value, reference):
        """Return a scale-aware absolute trace residual."""
        scale = max(1.0, abs(float(reference)))
        return abs(float(value) - float(reference)) / scale

    def _build_channel_target(self, p, event, *, cutoff_mode="rsum2"):
        """Build the exact deterministic channel sum before output compression."""
        target = None
        for operator, weight in zip(event.kraus, event.weights):
            if weight == 0.0:
                continue
            # ``_build_dmrg_target`` consumes dense Quimb gate payloads in
            # output/input order and internally transposes them to the
            # represented ket operator. Channel events, however, document
            # their Kraus matrices in the ordinary linear-algebra convention
            # ``K O K.H``. Transpose the payload once so the two conventions
            # meet at the exact target.
            channel_operator = operator.T
            branch = self._build_dmrg_target(
                p,
                channel_operator,
                event.where,
                channel_operator,
                cutoff=0.0,
                cutoff_mode=cutoff_mode,
                target_cutoff=0.0,
                # A channel sum is assembled with MPO addition, which needs
                # one tensor per site. Materialize each disposable branch
                # before adding it, even on dense backends where ordinary FIT
                # targets can remain layered.
                target_strategy="mpo",
            )
            if weight != 1.0:
                branch.multiply_(weight)
            if target is None:
                target = branch
            else:
                target.add_MPO_(branch)
        if target is None:
            raise ValueError("channel event has no nonzero outcome weight.")
        return target

    def _record_trace_event(self, p_before, target, p_after, event):
        """Record channel trace preservation separately from norm survival."""
        input_trace = self._measure_trace(p_before)
        target_trace = self._measure_trace(target)
        observed_trace = self._measure_trace(p_after)
        completeness = sum(
            float(weight) * matrix.conj().T @ matrix
            for matrix, weight in zip(event.kraus, event.weights)
        )
        identity = np.eye(completeness.shape[0], dtype=completeness.dtype)
        completeness_residual = float(
            np.linalg.norm(completeness - identity)
            / max(1.0, np.linalg.norm(identity))
        )
        target_residual = self._relative_trace_residual(target_trace, input_trace)
        compression_residual = self._relative_trace_residual(
            observed_trace, target_trace
        )
        record = {
            "kind": "channel_sum",
            "where": tuple(event.where),
            "semantics": event.semantics,
            "labels": tuple(event.labels),
            "weights": tuple(float(weight) for weight in event.weights),
            "input_trace": input_trace,
            "target_trace": target_trace,
            "observed_trace": observed_trace,
            "channel_completeness_residual": completeness_residual,
            "channel_trace_preserving": bool(completeness_residual <= 1.0e-10),
            "target_trace_residual": target_residual,
            "compression_trace_residual": compression_residual,
            "trace_preservation_residual": max(target_residual, compression_residual),
        }
        self.trace_events.append(record)
        return record

    def _run_channel_sum_event(
        self,
        p,
        event,
        *,
        cutoff,
        cutoff_mode,
        backend,
        fit_runner=None,
        fit_block_size=2,
        step=None,
        transactional_steps=True,
        fit_fallback=None,
    ):
        """Apply and compress one deterministic channel event."""
        if event.semantics == "sample":
            raise ValueError(
                "MPO channel events require semantics='sum'; sampled branches "
                "belong to MpsOptimizer trajectory replay."
            )
        snapshot = (
            self._capture_run_state(p)
            if backend == "dmrg" and (transactional_steps or fit_fallback is not None)
            else None
        )
        # Target construction works on branch copies, so the live ``p`` is
        # still the pre-channel state when trace diagnostics are evaluated.
        # FIT and direct compression are intentionally allowed to mutate their
        # input MPO in place. Keep an immutable pre-channel witness so the
        # trace-preservation report compares input -> target -> retained output
        # rather than accidentally comparing the output to itself.
        p_before = p.copy()
        target = self._timed_call(
            "dmrg.target" if backend == "dmrg" else "channel.target",
            self._build_channel_target,
            p,
            event,
            cutoff_mode=cutoff_mode,
        )
        expected_norm = self._timed_call(
            "norm.expected",
            self._canonical_norm_measurement,
            target,
        )
        fit = None
        if backend == "dmrg" and max(event.where) > min(event.where):
            xmin, xmax = min(event.where), max(event.where)
            self.canonize_mpo(p, (xmin, xmax))
            fit = FIT(
                target,
                p=p,
                cutoffs=cutoff,
                contraction_opt=self.contraction_opt,
                retag=False,
                range_int=[xmin, xmax],
                inplace=True,
                copy_target=False,
            )
            active_block_size = min(int(fit_block_size), xmax - xmin + 1)
            try:
                fit_runner(fit, active_block_size, event.where)
            except Exception as exc:
                if snapshot is None:
                    raise
                self._restore_run_state(snapshot)
                self._record_fit_failure(
                    exc,
                    where=event.where,
                    block_size=active_block_size,
                    step=step or 0,
                )
                if fit_fallback is None:
                    raise
                fallback_backend = "mpo" if fit_fallback == "mpo" else "svd"
                p_after = self._run_channel_sum_event(
                    self.p,
                    event,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    backend=fallback_backend,
                    transactional_steps=False,
                )
                self.fallback_events.append(
                    {
                        "kind": "fit_fallback",
                        "where": tuple(int(site) for site in event.where),
                        "backend": fit_fallback,
                        "step": int(step or 0),
                    }
                )
                self.last_run_fallback = fit_fallback
                return p_after
            p_after = fit.p
            final_center = fit.final_center_site
            if final_center is None:
                final_center = xmax
            observed_norm = self._timed_call(
                "norm.observed",
                self._canonical_norm_measurement,
                p_after,
                final_center,
            )
            self.info_c["cur_orthog"] = (int(final_center), int(final_center))
            self._record_fit_diagnostics(
                fit,
                where=event.where,
                block_size=active_block_size,
                step=step or 0,
            )
        else:
            p_after = target
            p_after.compress(
                form="left",
                max_bond=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            self.p = p_after
            self._init_canonicalization()
            observed_norm = self._timed_call(
                "norm.observed",
                self._canonical_norm_measurement,
                p_after,
            )
        self.p = p_after
        self._timed_call(
            "norm.record",
            self._record_norm_event,
            "channel_sum",
            expected_norm=expected_norm,
            observed_norm=observed_norm,
            target_norm=expected_norm,
            where=event.where,
        )
        trace_record = self._record_trace_event(p_before, target, p_after, event)
        self.channel_events.append(
            {
                "kind": "channel_sum",
                "where": tuple(event.where),
                "outcomes": len(event.kraus),
                "backend": backend,
                "trace_preserving": trace_record["channel_trace_preserving"],
                "trace_preservation_residual": trace_record[
                    "trace_preservation_residual"
                ],
            }
        )
        return p_after

    def _copy_working_state(self, p, where=None):
        """Own mutable dense arrays in a FIT window, preserving isometry tags.

        Exterior arrays are shared read-only. Native/unknown arrays retain
        full deep-copy isolation; Torch copies use differentiable clone().
        """
        cache = self._replay_array_kinds
        if cache is not None and p in cache:
            dense = cache[p]
        else:
            dense = all(
                not self._is_symmray_array(t.data)
                and ar.infer_backend(t.data) in {"numpy", "torch", "jax", "cupy"}
                for t in p
            )
            if cache is not None:
                cache[p] = dense
        if not dense:
            return p.copy(deep=True)
        result = p.copy()
        start, stop = (0, p.L - 1) if where is None else (min(where), max(where))
        for site in range(start, stop + 1):
            tensor = result[site]
            data = tensor.data
            copied = data.clone() if ar.infer_backend(data) == "torch" else ar.do("copy", data)
            tensor.modify(data=copied, left_inds=tensor.left_inds)
        if cache is not None:
            cache[result] = dense
        return result

    def _capture_run_state(self, p=None, *, where=None, copy_state=True):
        """Capture optimizer state needed for atomic DMRG recovery."""
        state = self.p if p is None else p
        return {
            "p": self._copy_working_state(state, where) if copy_state else state,
            "info_c": deepcopy(self.info_c),
            "losses": list(self.losses),
            "norm_events": list(self.norm_events),
            "norm_log_survival": self._norm_log_survival,
            # Committed records are append-only; only their list containers
            # need snapshots. Public accessors keep defensive deep copies.
            "fit_diagnostics": list(self.fit_diagnostics),
            "last_fit": self._last_dmrg_fit_diagnostics,
            "channel_events": list(self.channel_events),
            "fallback_events": list(self.fallback_events),
            "trace_events": list(self.trace_events),
            "dmrg1_one_site_locked": bool(self._dmrg1_one_site_locked),
        }

    def _restore_run_state(self, snapshot):
        """Restore an atomic replay snapshot."""
        self.p = snapshot["p"]
        self.info_c = snapshot["info_c"]
        self.losses = snapshot["losses"]
        self.norm_events = snapshot["norm_events"]
        self._norm_log_survival = snapshot["norm_log_survival"]
        self.fit_diagnostics = snapshot["fit_diagnostics"]
        self._last_dmrg_fit_diagnostics = snapshot["last_fit"]
        self.channel_events = snapshot["channel_events"]
        self.fallback_events = snapshot["fallback_events"]
        self.trace_events = snapshot["trace_events"]
        self._dmrg1_one_site_locked = snapshot["dmrg1_one_site_locked"]

    def _record_fit_failure(self, exc, *, where, block_size, step):
        """Retain a failed FIT attempt in the public per-update history."""
        record = {
            "step": int(step),
            "where": tuple(int(site) for site in where),
            "block_size": int(block_size),
            "iterations": 0,
            "converged": False,
            "convergence_reason": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        self.fit_diagnostics.append(record)
        self._last_dmrg_fit_diagnostics = record
        return record

    def _run_dmrg(
        self,
        G_seq,
        where_seq,
        n_iter,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        k_2q_batch=1,
        fidelity_samples=10,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        fit_max_span=None,
        fit_three_site_sweeps=1,
        target_cutoff=0.0,
        adaptive_block_sweeps=None,
        adaptive_until_rank=False,
        single_pair_fast_path=False,
        fit_min_iter=None,
        fit_rtol=None,
        fit_patience=1,
        finite_check=False,
        timing=False,
        timing_sync_device=False,
        collect_split_diagnostics=False,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_mpo_guess_order="lower_upper",
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_overlap_diagnostics=False,
        transactional_steps=True,
        fit_fallback=None,
    ):
        """Sweep the gate stream with local DMRG-style FIT compression.

        One-site gates are applied exactly; each two-site gate (or batch of
        ``k_2q_batch`` consecutive ones) is fitted by :class:`FIT` back to
        bond ``chi`` inside the gate window ``[xmin, xmax]``. ``fit_block_size``
        selects the one-, two-, or three-site native SVD update, and its
        cutoff policy is forwarded explicitly to both SVD splits and the
        requested sweep sequence. Named DMRG modes additionally pass FIT's
        adaptive block schedule so the larger block is followed by fixed-rank
        one-site refinement.
        """
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")
        if not isinstance(fit_block_size, Integral) or int(fit_block_size) not in {
            1,
            2,
            3,
        }:
            raise ValueError("fit_block_size must be 1, 2, or 3.")
        fit_block_size = int(fit_block_size)
        fit_target_strategy = self._validate_fit_target_strategy(fit_target_strategy)
        fit_mpo_guess = bool(fit_mpo_guess)
        fit_mpo_guess_order = self._validate_fit_mpo_guess_order(
            fit_mpo_guess_order
        )
        fit_init_strategy = self._validate_fit_init_strategy(fit_init_strategy)
        fit_init_rand_strength = float(fit_init_rand_strength)
        if not np.isfinite(fit_init_rand_strength) or fit_init_rand_strength < 0.0:
            raise ValueError(
                "fit_init_rand_strength must be finite and non-negative."
            )
        if not isinstance(fit_init_seed, Integral) or isinstance(fit_init_seed, bool):
            raise ValueError("fit_init_seed must be an integer.")
        fit_init_seed = int(fit_init_seed)
        if fit_init_seed < 0:
            raise ValueError("fit_init_seed must be non-negative.")
        fit_overlap_diagnostics = bool(fit_overlap_diagnostics)
        transactional_steps = bool(transactional_steps)

        p = self.p
        self._maybe_lock_dmrg1_one_site_phase()
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="dmrg_mpo",
                leave=True,
                position=0,
                colour="CYAN",
            )

        def run_local_fit(fit, active_block_size, where=()):
            """Run FIT while keeping generic and named schedules separate."""
            where = tuple(where)
            fit_kwargs = {
                "n_iter": n_iter,
                "verbose": False,
                "block_size": active_block_size,
                "sweep_sequence": fit_sweep_sequence,
                "max_bond": self.chi,
                "cutoff": cutoff,
                "cutoff_mode": cutoff_mode,
                "three_site_sweeps": fit_three_site_sweeps,
                # Match the named MPS schedule while fitting both operator
                # physical legs: three-site warm-up, two-site transition,
                # then one-site refinement. Generic DMRG keeps its handoff.
                "adaptive_block_sweeps": adaptive_block_sweeps,
                "two_site_transition_sweeps": 1 if self._dmrg_mode_alias == "dmrg3" else 0,
                "min_iter": fit_min_iter,
                "rtol": fit_rtol,
                "patience": fit_patience,
                "finite_check": finite_check,
                "timing": timing,
                "timing_sync_device": timing_sync_device,
                "collect_split_diagnostics": collect_split_diagnostics,
            }
            if adaptive_block_sweeps is not None:
                fit_kwargs.update(
                    adaptive_block_sweeps=adaptive_block_sweeps,
                    adaptive_until_rank=(
                        adaptive_until_rank
                        and not (
                            self._dmrg_mode_alias is None
                            and active_block_size in {2, 3}
                            and where
                            and max(where) - min(where) + 1 > active_block_size
                        )
                    ),
                )
            fit_kwargs["single_pair_fast_path"] = bool(single_pair_fast_path)
            if (
                self._dmrg_mode_alias == "dmrg2"
                and active_block_size == 2
                and max(where) == min(where) + 1
            ):
                fit_kwargs["single_pair_fast_path"] = True
            self._run_fit_gate(fit, **fit_kwargs)

        def run_local_fit_transactional(
            fit,
            active_block_size,
            *,
            step_start,
            step_end,
            where,
        ):
            """Run one FIT update with optional per-update recovery/fallback."""
            nonlocal p
            snapshot = (
                self._capture_run_state(p, where=where, copy_state=fit.p is p)
                if transactional_steps or fit_fallback is not None
                else None
            )
            try:
                run_local_fit(fit, active_block_size, where)
            except Exception as exc:
                if snapshot is None:
                    self._record_fit_failure(
                        exc,
                        where=where,
                        block_size=active_block_size,
                        step=step_start + 1,
                    )
                    raise
                self._restore_run_state(snapshot)
                p = self.p
                # Restoring the per-step snapshot also restores diagnostic
                # history, so append the failed attempt after the rollback.
                self._record_fit_failure(
                    exc,
                    where=where,
                    block_size=active_block_size,
                    step=step_start + 1,
                )
                if fit_fallback is None:
                    raise
                direct_runner = self._run_mpo if (
                    fit_fallback == "mpo" and not self._has_symmray_data(p)
                ) else self._run_svd
                direct_kwargs = dict(
                    progbar=False,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    fidelity_samples=0,
                    finite_check=finite_check,
                    fit_target_strategy=fit_target_strategy,
                )
                if fit_fallback == "mpo" and not self._has_symmray_data(p):
                    direct_kwargs["method"] = "direct"
                direct_runner(
                    G_seq[step_start:step_end],
                    where_seq[step_start:step_end],
                    **direct_kwargs,
                )
                p = self.p
                self.fallback_events.append(
                    {
                        "kind": "fit_fallback",
                        "where": tuple(int(site) for site in where),
                        "backend": fit_fallback,
                        "step": int(step_start + 1),
                    }
                )
                self.last_run_fallback = fit_fallback
                return p, None
            p = fit.p
            self.p = p
            return p, fit

        idx = 0
        while idx < len(G_seq):
            if isinstance(G_seq[idx], MpoChannelEvent):
                event = G_seq[idx]
                if len(event.where) == 2:
                    two_qubit_count += 1
                p = self._run_channel_sum_event(
                    p,
                    event,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    backend="dmrg",
                    fit_runner=run_local_fit,
                    fit_block_size=fit_block_size,
                    step=idx + 1,
                    transactional_steps=transactional_steps,
                    fit_fallback=fit_fallback,
                )
                idx += 1
                self._check_finite(finite_check, p)
                norm_proxy = (
                    self._append_norm_proxy_sample(p)
                    if idx in sample_steps
                    else norm_proxy
                )
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "2q": two_qubit_count,
                            "~F": self._cumulative_fidelity(),
                            "norm": norm_proxy,
                            "bnd": p.max_bond(),
                        }
                    )
                    pbar.update(1)
                continue
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
                idx += 1
                advanced = 1
            elif n_sites == 2:
                if k_2q_batch == 1:
                    two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    self.canonize_mpo(p, (xmin, xmax))
                    p_g = self._timed_call(
                        "dmrg.target",
                        self._build_dmrg_target,
                        p,
                        gate,
                        where,
                        bra_gate,
                        cutoff,
                        cutoff_mode,
                        target_cutoff=target_cutoff,
                        target_strategy=fit_target_strategy,
                    )

                    expected_norm = self._timed_call(
                        "norm.expected",
                        self._expected_target_norm,
                        p,
                        gate,
                        where,
                        bra_gate,
                        target=p_g,
                        target_cutoff=target_cutoff,
                        target_strategy=fit_target_strategy,
                        cutoff_mode=cutoff_mode,
                    )
                    requested_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    self._validate_dmrg1_iteration_budget(
                        p,
                        xmin,
                        xmax,
                        n_iter=n_iter,
                        block_size=requested_fit_block_size,
                    )
                    active_fit_block_size = self._resolve_dmrg_fit_block_size(
                        p,
                        xmin,
                        xmax,
                        fit_block_size,
                    )
                    fit_initialization = self._timed_call(
                        "dmrg.fit_guess",
                        self._prepare_fit_initial_guess,
                        p,
                        [G_seq[idx]],
                        [where],
                        block_size=active_fit_block_size,
                        strategy=fit_init_strategy,
                        fit_mpo_guess=fit_mpo_guess,
                        rand_strength=fit_init_rand_strength,
                        seed=(
                            int(fit_init_seed)
                            + 1000003 * int(idx)
                            + 1009 * int(xmin)
                            + int(xmax)
                        ),
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        layer_order=fit_mpo_guess_order,
                    )
                    fit_guess = fit_initialization["fit_guess"]
                    fit = FIT(
                        p_g,
                        p=fit_guess,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                        inplace=True,
                        copy_target=False,
                    )
                    p, fit_result = run_local_fit_transactional(
                        fit,
                        active_fit_block_size,
                        step_start=idx,
                        step_end=idx + 1,
                        where=where,
                    )
                    if fit_result is None:
                        idx += 1
                        advanced = 1
                    else:
                        final_center = fit.final_center_site
                        if final_center is None:
                            final_center = p.calc_current_orthog_center()[-1]
                        observed_norm = self._timed_call(
                            "norm.observed",
                            self._canonical_norm_measurement,
                            p,
                            final_center,
                        )
                        self.info_c["cur_orthog"] = (
                            int(final_center),
                            int(final_center),
                        )
                        self._timed_call(
                            "norm.record",
                            self._record_norm_event,
                            "dmrg_compression",
                            expected_norm=expected_norm,
                            observed_norm=observed_norm,
                            target_norm=expected_norm,
                            where=(xmin, xmax),
                            unitary=(
                                self._is_unitary_gate_pair(gate, bra_gate)
                                and self._unitary_norm_guard_supported(p)
                            ),
                        )
                        fit_overlap = (
                            self._fit_overlap_diagnostics(p_g, fit.p)
                            if fit_overlap_diagnostics
                            else {}
                        )
                        self._record_fit_diagnostics(
                            fit,
                            where=(xmin, xmax),
                            block_size=active_fit_block_size,
                            step=idx + 1,
                            mpo_fit_guess_used=fit_initialization[
                                "svd_guess_used"
                            ],
                            mpo_fit_guess_order=(
                                fit_mpo_guess_order
                                if fit_initialization["svd_guess_used"]
                                else None
                            ),
                            fit_initialization=fit_initialization,
                            fit_overlap=fit_overlap,
                            fit_overlap_diagnostics=fit_overlap_diagnostics,
                        )
                        self._maybe_lock_dmrg1_one_site_phase()
                        self._last_dmrg_fit_diagnostics[
                            "dmrg1_one_site_locked"
                        ] = bool(self._dmrg1_one_site_locked)
                        idx += 1
                        advanced = 1
                    if fit_result is None:
                        self._maybe_lock_dmrg1_one_site_phase()
                        if self._last_dmrg_fit_diagnostics is not None:
                            self._last_dmrg_fit_diagnostics[
                                "dmrg1_one_site_locked"
                            ] = bool(self._dmrg1_one_site_locked)
                else:
                    batch_G, batch_where, two_qubit_in_batch, next_idx = (
                        self._collect_dmrg_batch(
                            G_seq,
                            where_seq,
                            idx,
                            k_2q_batch,
                            max_span=fit_max_span,
                        )
                    )
                    if two_qubit_in_batch < 1:
                        raise RuntimeError("DMRG batch unexpectedly contains no two-qubit gates.")

                    two_qubit_count += two_qubit_in_batch
                    batch_span_sites = [site for where_i in batch_where for site in where_i]
                    xmin, xmax = min(batch_span_sites), max(batch_span_sites)
                    self.canonize_mpo(p, (xmin, xmax))
                    p_g = self._timed_call(
                        "dmrg.target",
                        self._build_dmrg_batch_target,
                        p,
                        batch_G,
                        batch_where,
                        cutoff,
                        cutoff_mode,
                        target_cutoff=target_cutoff,
                        target_strategy=fit_target_strategy,
                    )

                    expected_norm = self._timed_call(
                        "norm.expected",
                        self._expected_batch_target_norm,
                        p,
                        batch_G,
                        batch_where,
                        target=p_g,
                        cutoff_mode=cutoff_mode,
                    )
                    requested_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    self._validate_dmrg1_iteration_budget(
                        p,
                        xmin,
                        xmax,
                        n_iter=n_iter,
                        block_size=requested_fit_block_size,
                    )
                    active_fit_block_size = self._resolve_dmrg_fit_block_size(
                        p,
                        xmin,
                        xmax,
                        fit_block_size,
                    )
                    fit_initialization = self._timed_call(
                        "dmrg.fit_guess",
                        self._prepare_fit_initial_guess,
                        p,
                        batch_G,
                        batch_where,
                        block_size=active_fit_block_size,
                        strategy=fit_init_strategy,
                        fit_mpo_guess=fit_mpo_guess,
                        rand_strength=fit_init_rand_strength,
                        seed=(
                            int(fit_init_seed)
                            + 1000003 * int(idx)
                            + 1009 * int(xmin)
                            + int(xmax)
                        ),
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        layer_order=fit_mpo_guess_order,
                    )
                    fit_guess = fit_initialization["fit_guess"]
                    fit = FIT(
                        p_g,
                        p=fit_guess,
                        cutoffs=cutoff,
                        contraction_opt=self.contraction_opt,
                        retag=False,
                        range_int=[xmin, xmax],
                        inplace=True,
                        copy_target=False,
                    )
                    p, fit_result = run_local_fit_transactional(
                        fit,
                        active_fit_block_size,
                        step_start=idx,
                        step_end=next_idx,
                        where=(xmin, xmax),
                    )
                    if fit_result is None:
                        advanced = next_idx - idx
                        idx = next_idx
                    else:
                        final_center = fit.final_center_site
                        if final_center is None:
                            final_center = p.calc_current_orthog_center()[-1]
                        observed_norm = self._timed_call(
                            "norm.observed",
                            self._canonical_norm_measurement,
                            p,
                            final_center,
                        )
                        self.info_c["cur_orthog"] = (
                            int(final_center),
                            int(final_center),
                        )
                        self._timed_call(
                            "norm.record",
                            self._record_norm_event,
                            "dmrg_compression",
                            expected_norm=expected_norm,
                            observed_norm=observed_norm,
                            target_norm=expected_norm,
                            where=(xmin, xmax),
                            unitary=(
                                all(
                                    self._is_unitary_gate_pair(
                                        *self._parse_gate_entry(gate_i, where_i)[:2]
                                    )
                                    for gate_i, where_i in zip(batch_G, batch_where)
                                )
                                and self._unitary_norm_guard_supported(p)
                            ),
                        )
                        fit_overlap = (
                            self._fit_overlap_diagnostics(p_g, fit.p)
                            if fit_overlap_diagnostics
                            else {}
                        )
                        self._record_fit_diagnostics(
                            fit,
                            where=(xmin, xmax),
                            block_size=active_fit_block_size,
                            step=idx + 1,
                            mpo_fit_guess_used=fit_initialization[
                                "svd_guess_used"
                            ],
                            mpo_fit_guess_order=(
                                fit_mpo_guess_order
                                if fit_initialization["svd_guess_used"]
                                else None
                            ),
                            fit_initialization=fit_initialization,
                            fit_overlap=fit_overlap,
                            fit_overlap_diagnostics=fit_overlap_diagnostics,
                        )
                        self._maybe_lock_dmrg1_one_site_phase()
                        self._last_dmrg_fit_diagnostics[
                            "dmrg1_one_site_locked"
                        ] = bool(self._dmrg1_one_site_locked)
                        advanced = next_idx - idx
                        idx = next_idx
                    if fit_result is None:
                        self._maybe_lock_dmrg1_one_site_phase()
                        if self._last_dmrg_fit_diagnostics is not None:
                            self._last_dmrg_fit_diagnostics[
                                "dmrg1_one_site_locked"
                            ] = bool(self._dmrg1_one_site_locked)
            else:
                raise ValueError("Each gate location must have one or two sites.")

            self._check_finite(finite_check, p)
            # A batched update represents several gate steps, so its endpoint
            # can skip all linearly selected sample indices. Preserve the
            # historical per-update trace in that case while keeping
            # ``fidelity_samples=0`` opt-out semantics.
            if idx in sample_steps or (
                fidelity_samples > 0 and advanced > 1 and idx < len(G_seq)
            ):
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._cumulative_fidelity(),
                    "norm": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        self.p = p

    def _run_svd(
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        fidelity_samples=10,
        finite_check=False,
        fit_target_strategy="auto",
    ):
        """Sweep the gate stream with local ``reduce-split`` + left-compress.

        Two-site updates use ``apply_gate(..., contract='reduce-split')``
        followed by a canonicalise + left-compress sweep across the gate
        window down to bond ``chi``.
        """
        p = self.p
        two_qubit_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="svd",
                leave=True,
                position=0,
                colour="CYAN",
            )

        idx = 0
        while idx < len(G_seq):
            if isinstance(G_seq[idx], MpoChannelEvent):
                event = G_seq[idx]
                if len(event.where) == 2:
                    two_qubit_count += 1
                p = self._run_channel_sum_event(
                    p,
                    event,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    backend="svd",
                )
                idx += 1
                self._check_finite(finite_check, p)
                if idx in sample_steps:
                    norm_proxy = self._append_norm_proxy_sample(p)
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "2q": two_qubit_count,
                            "~F": self._cumulative_fidelity(),
                            "norm": norm_proxy,
                            "bnd": p.max_bond(),
                        }
                    )
                    pbar.update(1)
                continue
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
            elif n_sites == 2:
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                target = None
                if not self._is_unitary_gate_pair(gate, bra_gate):
                    target = self._build_dmrg_target(
                        p,
                        gate,
                        where,
                        bra_gate,
                        cutoff=0.0,
                        cutoff_mode=cutoff_mode,
                        target_cutoff=0.0,
                        target_strategy=fit_target_strategy,
                    )
                expected_norm = self._timed_call(
                    "norm.expected",
                    self._expected_target_norm,
                    p,
                    gate,
                    where,
                    bra_gate,
                    target=target,
                    target_cutoff=0.0,
                    target_strategy=fit_target_strategy,
                    cutoff_mode=cutoff_mode,
                )
                contract = (
                    "split-gate"
                    if self._has_symmray_data(p) and (xmax - xmin > 1)
                    else "reduce-split"
                )

                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=contract,
                    inplace=True,
                )

                self.canonize_mpo(p, (xmin, xmax))
                for i in range(xmax, xmin, -1):
                    p.right_canonize_site(i, bra=None)
                p.left_compress(
                    start=xmin,
                    stop=xmax,
                    max_bond=self.chi,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                )
                # ``left_compress`` leaves the active window's right edge as
                # the canonical center. Supplying it explicitly avoids a
                # native block-sparse center discovery that may densify a
                # large virtual tensor.
                observed_norm = self._timed_call(
                    "norm.observed",
                    self._canonical_norm_measurement,
                    p,
                    center=xmax,
                )
                self.info_c["cur_orthog"] = (xmax, xmax)
                self._timed_call(
                    "norm.record",
                    self._record_norm_event,
                    "svd_compression",
                    expected_norm=expected_norm,
                    observed_norm=observed_norm,
                    target_norm=expected_norm,
                    where=(xmin, xmax),
                    unitary=(
                        self._is_unitary_gate_pair(gate, bra_gate)
                        and self._unitary_norm_guard_supported(p)
                    ),
                )
            else:
                raise ValueError("Each gate location must have one or two sites.")

            self._check_finite(finite_check, p)
            idx += 1
            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._cumulative_fidelity(),
                    "norm": norm_proxy,
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p


    def _run_mpo(
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        fidelity_samples=10,
        finite_check=False,
        fit_target_strategy="auto",
        method=None,
        compression_seed=None,
    ):
        """Sweep the gate stream with :func:`gate_nonlocal_opt` compression.

        Multi-site gates are routed through ``gate_nonlocal_opt`` independently
        on the upper (ket) and lower (bra) MPO families using the selected
        Quimb compressor. One-site gates are applied directly via
        :meth:`_apply_gate_pair`.
        """
        method = self._resolve_mpo_method(method)
        p = self.p
        two_qubit_count = 0
        nonlocal_count = 0
        sample_steps = self._sampling_steps(len(G_seq), fidelity_samples)
        norm_proxy = self.losses[-1]

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc=method,
                leave=True,
                position=0,
                colour="GREEN",
            )

        idx = 0
        while idx < len(G_seq):
            if isinstance(G_seq[idx], MpoChannelEvent):
                event = G_seq[idx]
                if len(event.where) == 2:
                    two_qubit_count += 1
                p = self._run_channel_sum_event(
                    p,
                    event,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    backend="mpo",
                )
                idx += 1
                self.p = p
                self._check_finite(finite_check, p)
                if idx in sample_steps:
                    norm_proxy = self._append_norm_proxy_sample(p)
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "2q": two_qubit_count,
                            "~F": self._cumulative_fidelity(),
                            "norm": norm_proxy,
                            "bnd": p.max_bond(),
                        }
                    )
                    pbar.update(1)
                continue
            gate, bra_gate, where = self._parse_gate_entry(G_seq[idx], where_seq[idx])
            n_sites = len(where)
            if n_sites == 1:
                self._apply_gate_pair(
                    p,
                    gate,
                    where,
                    bra_gate=bra_gate,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    contract=True,
                    inplace=True,
                )
            elif n_sites >= 2:
                if n_sites == 2:
                    two_qubit_count += 1
                nonlocal_count += 1
                target = None
                unitary_pair = self._is_unitary_gate_pair(gate, bra_gate)
                if not unitary_pair:
                    target = self._timed_call(
                        "norm.target",
                        self._build_dmrg_target,
                        p,
                        gate,
                        where,
                        bra_gate,
                        cutoff=0.0,
                        cutoff_mode=cutoff_mode,
                        target_cutoff=0.0,
                        target_strategy=fit_target_strategy,
                    )
                expected_norm = self._timed_call(
                    "norm.expected",
                    self._expected_target_norm,
                    p,
                    gate,
                    where,
                    bra_gate,
                    target=target,
                    target_cutoff=0.0,
                    target_strategy=fit_target_strategy,
                    cutoff_mode=cutoff_mode,
                )
                if self._can_use_native_gate_sandwich(
                    p, gate, bra_gate, where, method
                ):
                    p = self._timed_call(
                        "mpo.gate_sandwich",
                        self._apply_native_gate_sandwich,
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                else:
                    g_k, g_b = self._prepare_nonlocal_gate_pair(
                        gate,
                        n_sites,
                        bra_gate=bra_gate,
                        p=p,
                        where=where,
                        ind_id=self.ind_id_k,
                    )
                    if g_k is not None:
                        compress_options = self._submpo_compress_options(
                            method,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            max_bond=self.chi,
                            seed=(
                                None
                                if compression_seed is None
                                else int(compression_seed) + int(idx)
                            ),
                        )
                        p = gate_nonlocal_opt(
                            p, g_k, where,
                            which="upper", method=method,
                            info=self.info_c, inplace=True,
                            ind_id_k=self.ind_id_k, ind_id_b=self.ind_id_b,
                            **compress_options,
                        )
                    if g_b is not None:
                        compress_options = self._submpo_compress_options(
                            method,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            max_bond=self.chi,
                            seed=(
                                None
                                if compression_seed is None
                                else int(compression_seed) + int(idx)
                            ),
                        )
                        p = gate_nonlocal_opt(
                            p, g_b, where,
                            which="lower", method=method,
                            info=self.info_c, inplace=True,
                            ind_id_k=self.ind_id_k, ind_id_b=self.ind_id_b,
                            **compress_options,
                        )
                self.p = p
                observed_center = self.info_c.get("cur_orthog")
                observed_norm = self._timed_call(
                    "norm.observed",
                    self._canonical_norm_measurement,
                    p,
                    center=observed_center,
                )
                self._timed_call(
                    "norm.record",
                    self._record_norm_event,
                    "mpo_compression",
                    expected_norm=expected_norm,
                    observed_norm=observed_norm,
                    target_norm=expected_norm,
                    where=where,
                    unitary=(
                        unitary_pair
                        and self._unitary_norm_guard_supported(p)
                    ),
                )

            self._check_finite(finite_check, p)
            idx += 1
            if idx in sample_steps:
                norm_proxy = self._append_norm_proxy_sample(self.p)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "nonlocal": nonlocal_count,
                    "~F": self._cumulative_fidelity(),
                    "norm": norm_proxy,
                    "bnd": self.p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = p


    @_replay_policy
    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=8,
        *,
        mode=None,
        compression_seed=None,
        submpo_method=None,
        progbar=False,
        cutoff="auto",
        cutoff_mode="auto",
        fidelity_samples=10,
        k_2q_batch=1,
        fit_block_size=2,
        fit_sweep_sequence="RL",
        fit_max_span="auto",
        fit_three_site_sweeps=1,
        fit_adaptive_sweeps=2,
        fit_single_pair_fast_path=False,
        target_cutoff=0.0,
        fit_min_iter=2,
        fit_rtol="auto",
        fit_patience=2,
        finite_check=False,
        fit_finite_check=None,
        timing=False,
        timing_sync_device=False,
        fit_collect_split_diagnostics=False,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_mpo_guess_order="lower_upper",
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_overlap_diagnostics=False,
        layout=None,
        layout_order="quality",
        layout_kwargs=None,
        layout_allow_lossy_reorder=False,
        atomic=True,
        transactional_steps=True,
        fit_fallback=None,
    ):
        """Run queued gates on both MPO index families.

        Parameters
        ----------
        n_iter : int, default=8
            Inner iterations for DMRG ``FIT`` updates on two-site gates. With
            adaptive block fitting, the initial block phase hands the
            remaining sweeps to one-site refinement. Ignored by ``svd`` mode.
        mode : {"direct", "dmrg", "dmrg1", "dmrg2", "dmrg3", "svd", "quimb-<method>"} | None, default=None
            Optional mode override for this run. Bare Quimb method names
            such as ``"src"`` are accepted as aliases for ``"quimb-src"``.
        compression_seed : int | None, default=None
            Deterministic seed forwarded to randomized Quimb compression modes
            such as ``"src"`` and ``"fit"``. The gate position is mixed into
            the per-update seed.
        submpo_method : str | None, default=None
            Optional Quimb compression override for an MPO-mode run. This is
            the MPO analogue of `MpsOptimizer`'s ``submpo_method``; for
            example, ``mode="direct", submpo_method="src"`` is equivalent to
            ``mode="src"``.
        progbar : bool, default=False
            Show tqdm progress bar.
        cutoff : float | {"auto"}, default="auto"
            Dtype-aware truncation cutoff used in gating and compression.
        cutoff_mode : str, default="auto"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``. Auto uses ``rsum1`` for dense
            density-matrix compression and ``rsum2`` for SVD-based routes.
        fidelity_samples : int, default=10
            ``svd`` mode only: number of intermediate norm-proxy samples.
            A final sample is always recorded at the end of the run.
        k_2q_batch : int, default=1
            ``dmrg`` mode only: number of sequential two-qubit gates to batch
            into one local FIT update. The FIT window uses the batch-wide
            ``[xmin, xmax]`` from all gate locations in the batch.
        fit_block_size : {1, 2, 3}, default=2
            ``dmrg`` mode only: number of neighboring MPO tensors optimized
            by each native FIT update. Two- and three-site updates grow only
            bonds reached by their SVD splits; one-site FIT retains the legacy
            fixed-rank compatibility path.
        fit_sweep_sequence : str, default="RL"
            ``dmrg`` mode only: cyclic FIT sweep directions. ``"R"`` and
            ``"L"`` select one direction, while ``"RL"`` alternates them.
        fit_max_span : int | {"auto"} | None, default="auto"
            ``dmrg`` mode only: maximum inclusive spatial span of a batched
            FIT target. The first long-range gate is always retained.
        fit_three_site_sweeps : int, default=1
            Legacy ``dmrg`` spelling for initial three-site warm-up sweeps
            when ``fit_block_size=3``. Named modes use
            ``fit_adaptive_sweeps``.
        fit_adaptive_sweeps : int, default=2
            Number of initial two- or three-site warm-up sweeps before
            one-site refinement. The generic ``dmrg`` schedule and
            ``dmrg2``/``dmrg3`` use the value; ``dmrg1`` always uses two
            sweeps. The value is clipped to ``n_iter``.
        fit_single_pair_fast_path : bool, default=False
            Stop an adjacent two-site window after its single exact
            variational update. ``dmrg2`` retains the MPS-compatible automatic
            adjacent-pair shortcut; set this to ``True`` to request it for
            every DMRG schedule.
        target_cutoff : float, default=0.0
            ``dmrg`` mode only: cutoff used while constructing the target
            MPO. The output FIT SVD remains controlled by ``cutoff``.
        fit_min_iter : int | None, default=2
            Minimum FIT sweeps before ``fit_rtol`` can stop a DMRG update.
        fit_rtol : float | {"auto"} | None, default="auto"
            Dtype-aware retained-center norm tolerance for early FIT stopping.
            ``None`` preserves fixed ``n_iter`` behavior.
        fit_patience : int, default=2
            Number of stable FIT norm samples required by ``fit_rtol``.
        finite_check : bool | callable, default=False
            Opt-in finite-data and norm-overshoot diagnostics across replay
            modes, including empty runs. Warns once per enabled run. A
            callable receives the MPO and must return a truthy value.
        fit_finite_check : bool | callable | None, default=None
            Compatibility alias for ``finite_check``; conflicting values
            raise an error.
        timing : bool, default=False
            Record wall-clock and FIT sweep timing in ``last_run_timing``.
        timing_sync_device : bool, default=False
            Request backend synchronization for FIT timing measurements.
        fit_collect_split_diagnostics : bool, default=False
            Retain native FIT split metadata in the fit diagnostics.
        fit_target_strategy : {"auto", "layered", "mpo"}, default="auto"
            Dense MPO targets use lazy layered gate tensors by default;
            native Symmray targets use the block-aware MPO representation.
        fit_mpo_guess : bool, default=True
            Legacy compatibility switch for the implicit ``"guess-src"``
            initial-guess policy in named DMRG windows. This does not replace
            the exact FIT target or live MPO. Set ``fit_init_strategy``
            explicitly to control all windows.
        fit_mpo_guess_order : {"lower_upper", "upper_lower"}, default="lower_upper"
            Layer order for the isolated MPO guess. ``"lower_upper"`` means
            bra then ket; ``"upper_lower"`` means ket then bra. In this API
            the lower layer is bra and the upper layer is ket. Aliases
            ``"bra_ket"`` and ``"ket_bra"`` are accepted.
        fit_init_strategy : {"auto", "direct", "random", "random_expand", "guess-<method>"}, default="guess-src"
            Select the disposable FIT initial guess. ``"direct"`` uses the
            current MPO, ``"random"`` perturbs existing active tensors,
            ``"random_expand"`` also seeds newly expanded active bonds, and
            ``"guess-<method>"`` uses a Quimb compressor such as
            ``"guess-src"`` on an isolated MPO replay. ``"auto"`` selects
            ``"guess-src"``. Native Symmray/fermionic MPOs retain their
            sector-preserving direct FIT warm start.
        fit_init_rand_strength : float, default=0.0
            Noise scale used by the explicit ``"random"`` and
            ``"random_expand"`` initial-guess strategies.
        fit_init_seed : int, default=0
            Deterministic seed for randomized FIT guesses and Quimb source
            compression methods used by ``fit_init_strategy``.
        fit_overlap_diagnostics : bool, default=False
            Contract each successful fitted MPO against its disposable exact
            target and record target-overlap fidelity. This adds one extra
            tensor-network contraction per FIT update.
        layout : mapping | sequence | str | None, default=None
            Optional persistent logical-to-physical layout. A string selects a
            gate-stream layout order; a mapping or sequence supplies an
            explicit order.
        layout_order : str, default="quality"
            Finder order used when ``layout=True`` or ``layout`` is omitted.
        layout_kwargs : mapping | None, default=None
            Extra layout-finder options.
        layout_allow_lossy_reorder : bool, default=False
            Allow truncation while installing a layout on an entangled MPO.
        atomic : bool, default=True
            Restore the optimizer state if replay fails. With ``inplace=True``
            the optimizer is restored, but external references to the original
            MPO may already have observed in-place changes.
        transactional_steps : bool, default=True
            Snapshot each DMRG gate or batch before FIT so ``atomic=False`` can
            still preserve all completed updates when one local update fails.
        fit_fallback : {None, "direct", "svd"}, default=None
            If DMRG FIT fails, restore the pre-run state and replay the complete
            stream through the selected direct compression backend.

        Returns
        -------
        qtn.MatrixProductOperator
            Updated MPO after replaying the queued gate stream.

        Notes
        -----
        Native Symmray direct/Quimb modes use block-aware SVD compression;
        DMRG modes retain native FIT. Evolution preserves the operator's
        absolute Hilbert--Schmidt scale and never normalizes it as a state.
        """
        if mode is not None:
            self.set_mode(mode)

        fit_finite_check = finite_check

        if layout is not None and layout is not False:
            requested_layout = layout_order if layout is True else layout
            self.apply_layout(
                requested_layout,
                cutoff=cutoff,
                allow_lossy_reorder=layout_allow_lossy_reorder,
                layout_kwargs=layout_kwargs,
            )

        cutoff = self._resolve_cutoff(cutoff, self.p)
        if cutoff_mode is None or cutoff_mode == "auto":
            method = self._resolve_mpo_method(submpo_method)
            cutoff_mode = "rsum1" if (
                self._is_mpo_mode(self.mode)
                and method == "dm"
                and not self._has_symmray_data(self.p)
            ) else "rsum2"
        timing = bool(timing)
        timing_sync_device = bool(timing_sync_device)

        G_seq, where_seq = self._execution_stream()
        if not G_seq:
            self.last_run_status = "complete"
            self.last_run_error = None
            self.last_run_fallback = None
            self.last_run_timing = None
            if timing:
                self._start_run_timing(event_count=0, sync_device=timing_sync_device)
            try:
                self._check_finite(finite_check)
            except Exception as exc:
                self.last_run_status = "failed"
                self.last_run_error = f"{type(exc).__name__}: {exc}"
                self._finish_run_timing("failed")
                raise
            if timing:
                self._finish_run_timing("complete")
            return self.p

        if not isinstance(fit_patience, Integral) or int(fit_patience) < 1:
            raise ValueError("fit_patience must be a positive integer.")
        if fit_min_iter is not None and (
            not isinstance(fit_min_iter, Integral) or int(fit_min_iter) < 1
        ):
            raise ValueError("fit_min_iter must be a positive integer or None.")
        if fit_rtol == "auto":
            dtype = str(self.backend_dtype).lower()
            fit_rtol = 1e-3 if "16" in dtype else (
                1e-5 if "32" in dtype or "complex64" in dtype else 1e-9
            )
        if fit_rtol is not None:
            fit_rtol = float(fit_rtol)
            if not math.isfinite(fit_rtol) or fit_rtol < 0.0:
                raise ValueError("fit_rtol must be a finite non-negative number or None.")
        if self.mode == "dmrg" and (
            not isinstance(k_2q_batch, Integral) or int(k_2q_batch) < 1
        ):
            raise ValueError("k_2q_batch must be >= 1.")
        if self.mode == "dmrg" and (
            not isinstance(fit_adaptive_sweeps, Integral)
            or int(fit_adaptive_sweeps) < 1
        ):
            raise ValueError("fit_adaptive_sweeps must be a positive integer.")
        if isinstance(k_2q_batch, Integral):
            k_2q_batch = int(k_2q_batch)
        if fit_fallback == "direct":
            fit_fallback = "mpo"  # compatibility record spelling
        if fit_fallback not in {None, "mpo", "svd"}:
            raise ValueError("fit_fallback must be None, 'direct', 'mpo', or 'svd'.")
        fit_target_strategy = self._validate_fit_target_strategy(fit_target_strategy)
        fit_mpo_guess = bool(fit_mpo_guess)
        fit_mpo_guess_order = self._validate_fit_mpo_guess_order(
            fit_mpo_guess_order
        )
        fit_init_strategy = self._validate_fit_init_strategy(fit_init_strategy)
        fit_init_rand_strength = float(fit_init_rand_strength)
        if not np.isfinite(fit_init_rand_strength) or fit_init_rand_strength < 0.0:
            raise ValueError(
                "fit_init_rand_strength must be finite and non-negative."
            )
        if not isinstance(fit_init_seed, Integral) or isinstance(fit_init_seed, bool):
            raise ValueError("fit_init_seed must be an integer.")
        fit_init_seed = int(fit_init_seed)
        if fit_init_seed < 0:
            raise ValueError("fit_init_seed must be non-negative.")
        fit_overlap_diagnostics = bool(fit_overlap_diagnostics)
        if compression_seed is not None:
            if not isinstance(compression_seed, Integral) or isinstance(
                compression_seed, bool
            ):
                raise ValueError("compression_seed must be an integer or None.")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError("compression_seed must be non-negative.")
        mpo_method_override = None
        if submpo_method is not None:
            mpo_method_override = self._normalize_submpo_method(submpo_method)
        atomic = bool(atomic)
        if fit_finite_check not in (None, False, True) and not callable(
            fit_finite_check
        ):
            raise TypeError("fit_finite_check must be bool, callable, or None.")

        snapshot = self._capture_run_state() if atomic or fit_fallback else None
        fallback_event_count = len(self.fallback_events)
        self.last_run_status = "running"
        self.last_run_error = None
        self.last_run_fallback = None
        self.last_run_timing = None
        if timing:
            self._start_run_timing(
                event_count=len(G_seq),
                sync_device=timing_sync_device,
            )

        if self.mode == "dmrg":
            dmrg_alias = self._dmrg_mode_alias
            if dmrg_alias is not None:
                # Named modes are readable schedule aliases, not separate
                # compression backends. Their block size is authoritative;
                # callers tune the warm-up length with fit_adaptive_sweeps.
                fit_block_size = self._dmrg_alias_block_size(dmrg_alias)
            if not isinstance(fit_block_size, Integral) or int(fit_block_size) not in {
                1,
                2,
                3,
            }:
                raise ValueError("fit_block_size must be 1, 2, or 3.")
            if (
                not isinstance(fit_three_site_sweeps, Integral)
                or int(fit_three_site_sweeps) < 1
            ):
                raise ValueError("fit_three_site_sweeps must be a positive integer.")
            if (
                dmrg_alias is None
                and int(fit_block_size) != 3
                and int(fit_three_site_sweeps) != 1
            ):
                raise ValueError(
                    "fit_three_site_sweeps is only configurable when "
                    "fit_block_size=3."
                )
            if dmrg_alias is not None:
                if (
                    not isinstance(n_iter, Integral)
                    or int(n_iter) < 1
                ):
                    raise ValueError("n_iter must be a positive integer.")
                if (
                    not isinstance(fit_adaptive_sweeps, Integral)
                    or int(fit_adaptive_sweeps) < 1
                ):
                    raise ValueError(
                        "fit_adaptive_sweeps must be a positive integer."
                    )
                adaptive_block_sweeps = min(
                    2 if dmrg_alias == "dmrg1" else int(fit_adaptive_sweeps),
                    int(n_iter),
                ) if int(n_iter) >= 2 else None
                adaptive_until_rank = False
                # ``fit_three_site_sweeps`` is the legacy generic-DMRG
                # spelling. Named modes use the common adaptive schedule.
                fit_three_site_sweeps = 1
                single_pair_fast_path = bool(fit_single_pair_fast_path)
            else:
                # Match the generic MPS DMRG policy: block FIT first grows
                # attainable bond spaces, then hands the remaining sweep
                # budget to one-site refinement. Long-range windows opt out
                # of the rank-until-ready phase below because their terminal
                # canonical handoff is fixed by the active span.
                adaptive_block_sweeps = (
                    min(int(fit_adaptive_sweeps), int(n_iter))
                    if int(fit_block_size) in {2, 3} and int(n_iter) >= 2
                    else None
                )
                adaptive_until_rank = (
                    int(fit_block_size) in {2, 3}
                    and int(n_iter) >= 2
                )
                single_pair_fast_path = bool(fit_single_pair_fast_path)
            fit_max_span = self._resolve_fit_max_span(
                fit_max_span,
                k_2q_batch,
            )
            target_cutoff = float(target_cutoff)
            if not np.isfinite(target_cutoff) or target_cutoff < 0.0:
                raise ValueError(
                    "target_cutoff must be a finite non-negative number."
                )
            # Native Symmray FIT uses block-aware target construction and
            # native SVD splits. It must not take Quimb's eager bond-padding
            # route, but it otherwise follows the same variational DMRG path
            # as dense MPOs. ``mode='mpo'`` remains the direct SVD path.
            try:
                self._timed_call(
                    "dmrg.prepare",
                    self._prepare_dmrg_state,
                    fit_block_size=fit_block_size,
                )
                self._timed_call(
                    "dmrg.replay",
                    self._run_dmrg,
                    G_seq,
                    where_seq,
                    n_iter=n_iter,
                    progbar=progbar,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    k_2q_batch=k_2q_batch,
                    fidelity_samples=fidelity_samples,
                    fit_block_size=fit_block_size,
                    fit_sweep_sequence=fit_sweep_sequence,
                    fit_max_span=fit_max_span,
                    fit_three_site_sweeps=int(fit_three_site_sweeps),
                    target_cutoff=target_cutoff,
                    adaptive_block_sweeps=adaptive_block_sweeps,
                    adaptive_until_rank=adaptive_until_rank,
                    single_pair_fast_path=single_pair_fast_path,
                    fit_min_iter=None if fit_min_iter is None else int(fit_min_iter),
                    fit_rtol=fit_rtol,
                    fit_patience=int(fit_patience),
                    finite_check=fit_finite_check,
                    timing=timing,
                    timing_sync_device=timing_sync_device,
                    collect_split_diagnostics=bool(fit_collect_split_diagnostics),
                    fit_target_strategy=fit_target_strategy,
                    fit_mpo_guess=fit_mpo_guess,
                    fit_mpo_guess_order=fit_mpo_guess_order,
                    fit_init_strategy=fit_init_strategy,
                    fit_init_rand_strength=fit_init_rand_strength,
                    fit_init_seed=fit_init_seed,
                    fit_overlap_diagnostics=fit_overlap_diagnostics,
                    transactional_steps=transactional_steps,
                    fit_fallback=fit_fallback,
                )
            except Exception as exc:
                self.last_run_status = "failed"
                self.last_run_error = f"{type(exc).__name__}: {exc}"
                failed_fit_records = []
                if snapshot is not None:
                    failed_fit_records = deepcopy(
                        [
                            record
                            for record in self.fit_diagnostics[
                                len(snapshot["fit_diagnostics"]):
                            ]
                            if record.get("convergence_reason") == "failed"
                        ]
                    )
                if fit_fallback is not None:
                    if snapshot is None:
                        self._finish_run_timing("failed")
                        raise RuntimeError(
                            "fit_fallback requires atomic replay state."
                        ) from exc
                    self._restore_run_state(snapshot)
                    self.fit_diagnostics.extend(failed_fit_records)
                    if failed_fit_records:
                        self._last_dmrg_fit_diagnostics = failed_fit_records[-1]
                    self.last_run_fallback = fit_fallback
                    try:
                        direct_runner = self._run_mpo if (
                            fit_fallback == "mpo"
                            and not self._has_symmray_data(self.p)
                        ) else self._run_svd
                        self._timed_call(
                            "fallback.replay",
                            direct_runner,
                            G_seq,
                            where_seq,
                            progbar=progbar,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            fidelity_samples=fidelity_samples,
                            finite_check=fit_finite_check,
                        )
                    except Exception:
                        self._restore_run_state(snapshot)
                        self._finish_run_timing("failed")
                        raise
                    self.fallback_events.append(
                        {
                            "kind": "run_fallback",
                            "backend": fit_fallback,
                            "step": None,
                        }
                    )
                    self.last_run_status = "fallback"
                else:
                    if snapshot is not None:
                        self._restore_run_state(snapshot)
                        self.fit_diagnostics.extend(failed_fit_records)
                        if failed_fit_records:
                            self._last_dmrg_fit_diagnostics = failed_fit_records[-1]
                    self._finish_run_timing("failed")
                    raise
            if self.last_run_status == "running":
                self.last_run_status = (
                    "fallback"
                    if len(self.fallback_events) > fallback_event_count
                    else "complete"
                )
            self._finish_run_timing(self.last_run_status)
            return self.p

        if self.mode == "svd":
            try:
                self._timed_call(
                    "svd.replay",
                    self._run_svd,
                    G_seq,
                    where_seq,
                    progbar=progbar,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    fidelity_samples=fidelity_samples,
                    finite_check=fit_finite_check,
                    fit_target_strategy=fit_target_strategy,
                )
            except Exception as exc:
                self.last_run_status = "failed"
                self.last_run_error = f"{type(exc).__name__}: {exc}"
                if snapshot is not None:
                    self._restore_run_state(snapshot)
                self._finish_run_timing("failed")
                raise
            self.last_run_status = "complete"
            self._finish_run_timing(self.last_run_status)
            return self.p

        if self._is_mpo_mode(self.mode):
            # ``gate_nonlocal_opt`` creates a dense auxiliary sub-MPO and its
            # generic compression currently loses multi-sector Symmray bond
            # metadata. Reuse the block-aware local SVD route for these MPOs.
            if self._has_symmray_data(self.p):
                try:
                    self._timed_call(
                        "svd.replay",
                        self._run_svd,
                        G_seq,
                        where_seq,
                        progbar=progbar,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        fidelity_samples=fidelity_samples,
                        finite_check=fit_finite_check,
                        fit_target_strategy=fit_target_strategy,
                    )
                except Exception as exc:
                    self.last_run_status = "failed"
                    self.last_run_error = f"{type(exc).__name__}: {exc}"
                    if snapshot is not None:
                        self._restore_run_state(snapshot)
                    self._finish_run_timing("failed")
                    raise
                self.last_run_status = "complete"
                self._finish_run_timing(self.last_run_status)
                return self.p
            try:
                self._timed_call(
                    f"{mpo_method_override or self._mode_mpo_method(self.mode)}.replay",
                    self._run_mpo,
                    G_seq,
                    where_seq,
                    progbar=progbar,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    fidelity_samples=fidelity_samples,
                    finite_check=fit_finite_check,
                    fit_target_strategy=fit_target_strategy,
                    method=(
                        mpo_method_override
                        if mpo_method_override is not None
                        else self._mode_mpo_method(self.mode)
                    ),
                    compression_seed=compression_seed,
                )
            except Exception as exc:
                self.last_run_status = "failed"
                self.last_run_error = f"{type(exc).__name__}: {exc}"
                if snapshot is not None:
                    self._restore_run_state(snapshot)
                self._finish_run_timing("failed")
                raise
            self.last_run_status = "complete"
            self._finish_run_timing(self.last_run_status)
            return self.p

        supported = ", ".join(sorted(self._ALLOWED_MODES))
        raise ValueError(f"Unknown mode: {self.mode}. Supported modes: {supported}")

    def canonize_mpo(self, p, where):
        """Update canonical form around a one- or two-site gate span.

        ``where`` may be an int, a 1-tuple ``(site,)``, or a 2-tuple
        ``(xmin, xmax)``.  Integers and singletons collapse to a single-site
        orthogonality center.
        """
        if isinstance(where, Integral):
            site = int(where)
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 1:
            site = int(where[0])
            where_canon = [site]
            target_orthog = (site, site)
        elif len(where) == 2:
            xmin, xmax = min(int(where[0]), int(where[1])), max(int(where[0]), int(where[1]))
            where_canon = [xmin, xmax]
            target_orthog = (xmin, xmax)
        else:
            raise ValueError("where must be an int, (int,), or (int, int).")

        info = self.info_c if p is self.p else {}
        p.canonize(where_canon, cur_orthog=self._current_orthog(p, info=info), info=info)
        info["cur_orthog"] = target_orthog

    def sync_canonicalization(self, site=None):
        """Repair ``info_c`` after direct canonicalization of the live MPO.

        Low-level access through :attr:`p` can move the MPO orthogonality
        centre without updating the optimizer-owned ``info_c`` dictionary.
        Discover the actual live centre, move it to a single site, and bind
        the resulting metadata back to this optimizer before replay resumes.
        """
        if self._replay_rank_cache is not None:
            self._replay_rank_cache.clear()
        if not hasattr(self.p, "calc_current_orthog_center"):
            raise TypeError("the live MPO does not expose canonical metadata.")

        current = self.p.calc_current_orthog_center()
        if isinstance(current, Integral):
            current = (int(current), int(current))
        else:
            try:
                current = tuple(int(value) for value in current)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "the live MPO returned an invalid orthogonality centre."
                ) from exc
            if len(current) == 1:
                current = (current[0], current[0])
            elif len(current) == 2:
                current = (min(current), max(current))
            else:
                raise ValueError(
                    "the live MPO returned an invalid orthogonality span."
                )

        if site is None:
            site = current[1]
        site = int(site)
        if not 0 <= site < int(self.p.L):
            raise ValueError(
                f"site must lie in [0, {int(self.p.L)}), got {site}."
            )

        self.p.canonize([site], cur_orthog=current)
        self.info_c["cur_orthog"] = (site, site)
        return self.info_c["cur_orthog"]

    def get_fidelities(self):
        """Return the legacy normalized-MPO-norm history.

        This compatibility accessor is not the compression-fidelity ledger;
        use :meth:`norm_diagnostics` or :meth:`get_norm_events` for that.
        """
        return self.losses

    def get_norm_events(self):
        """Return a defensive copy of automatic MPO norm events."""
        return deepcopy(self.norm_events)

    def get_fit_diagnostics(self):
        """Return a defensive copy of the latest MPO FIT diagnostic record."""
        return deepcopy(self._last_dmrg_fit_diagnostics)

    def get_fit_history(self):
        """Return defensive copies of all FIT records from the latest replay."""
        return deepcopy(self.fit_diagnostics)

    def _start_run_timing(self, *, event_count, sync_device=False):
        """Start the opt-in MPO replay timing collector."""
        if self._timing_state is not None:
            raise RuntimeError("an MPO timing collection is already active.")
        self._timing_state = {
            "stages": {},
            "fit_steps": [],
            "fit_call_count": 0,
            "sync_device": bool(sync_device),
            "synchronizer": (
                FIT._make_backend_synchronizer(self.p)
                if sync_device
                else None
            ),
            "event_count": int(event_count),
            "started": time.perf_counter(),
        }
        self._sync_timing_device()

    def _finish_run_timing(self, status):
        """Finish the opt-in timing collector and publish its MPS-shaped record."""
        timing_state = self._timing_state
        if timing_state is None:
            self.last_run_timing = None
            return

        self._sync_timing_device()
        try:
            final_bond = int(self.p.max_bond())
        except (AttributeError, TypeError, ValueError):
            final_bond = None
        backend = self.backend_info()
        self.last_run_timing = {
            "status": str(status),
            "mode": self.mode,
            "mode_alias": self._dmrg_mode_alias,
            "event_count": timing_state["event_count"],
            "elapsed_seconds": float(
                time.perf_counter() - timing_state["started"]
            ),
            "final_bond": final_bond,
            "chi": int(self.chi),
            "backend": backend["backend"],
            "backend_dtype": backend["dtype"],
            "backend_device": backend["device"],
            "timing_sync_device": bool(timing_state["sync_device"]),
            "stages": timing_state["stages"],
            "fit_steps": timing_state["fit_steps"],
            "fit_totals": _summarize_fit_timing(timing_state["fit_steps"]),
            # Retain the compact pre-MPS-schema count for callers that used
            # the original MPO timing record.
            "fit_calls": len(self.fit_diagnostics),
            "fallback": self.last_run_fallback,
            "fit_diagnostics": (
                None
                if self._last_dmrg_fit_diagnostics is None
                else dict(self._last_dmrg_fit_diagnostics)
            ),
            "mix_summary": None,
        }
        self._timing_state = None

    def _record_timing_stage(self, name, elapsed):
        """Accumulate one inclusive replay stage measurement."""
        if self._timing_state is None:
            return
        stage = self._timing_state["stages"].setdefault(
            str(name),
            {"calls": 0, "elapsed_seconds": 0.0},
        )
        stage["calls"] += 1
        stage["elapsed_seconds"] += float(elapsed)

    def _sync_timing_device(self, value=None):
        """Apply an accelerator barrier only during synchronized profiling."""
        if self._timing_state is None:
            return
        synchronizer = self._timing_state.get("synchronizer")
        if synchronizer is not None:
            target = self.p if value is None else value
            synchronizer.synchronize(target, fallback=self.p)

    def _timed_call(self, name, function, *args, **kwargs):
        """Call ``function`` and time it only during an opt-in run."""
        if self._timing_state is None:
            return function(*args, **kwargs)

        self._sync_timing_device()
        started = time.perf_counter()
        try:
            result = function(*args, **kwargs)
        except BaseException:
            self._sync_timing_device()
            self._record_timing_stage(name, time.perf_counter() - started)
            raise
        self._sync_timing_device(result)
        self._record_timing_stage(name, time.perf_counter() - started)
        return result

    def get_run_timing(self):
        """Return the latest opt-in MPO replay timing record."""
        return deepcopy(self.last_run_timing)

    def compress(self, *, cutoff=1e-12, cutoff_mode="rsum2"):
        """Compress the current MPO to ``chi`` while preserving its backend.

        Symmray MPOs use Quimb's local block-aware compression path. This is
        also useful when the optimizer is constructed with an empty gate
        queue and the caller only wants a bond-dimension reduction.
        """
        cutoff = self._resolve_cutoff(cutoff, self.p)
        before = self._canonical_norm_measurement(self.p)
        self.p.compress(
            form="left",
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        self._init_canonicalization()
        after = self._canonical_norm_measurement(self.p)
        self._record_norm_event(
            "manual_compression",
            expected_norm=before,
            observed_norm=after,
            target_norm=before,
            where=tuple(self.info_c.get("cur_orthog", ())),
            unitary=self._unitary_norm_guard_supported(self.p),
        )
        self._append_norm_proxy_sample(self.p)
        return self.p
