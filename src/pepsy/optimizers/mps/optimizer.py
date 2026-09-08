"""MPS optimization helpers centered on :class:`MpsOptimizer`.

:class:`MpsOptimizer` replays a canonical bundled gate stream
``[(gate, where), ...]`` against an MPS, using one of several compression
backends. ``mode="perm"`` uses a lazy permutation swap network: non-local
two-site gates swap the right endpoint next to the left endpoint, apply the
gate, and leave the resulting physical ordering in place. The current
physical-site-to-logical-site ordering is available as ``optimizer.qubits``.
For repeated layout-aware evolution, :meth:`MpsOptimizer.apply_layout`
installs a persistent position-to-logical mapping and never performs a
swap-back; logical readout is available through ``logical_order``,
``remap_sample``, and ``to_dense``.
Bare Quimb method names such as ``mode="src"`` and ``mode="zipup"`` are
accepted and normalized internally to ``"quimb-<method>"``. The explicit
``mode="quimb-direct"`` spelling (with ``"quimb"`` as its direct alias and
the legacy ``"mpo-<method>"`` / ``"mpo"`` spellings retained) also accepts
explicit sub-MPO events of the form
``("submpo", mpo, where)`` or
``{"kind": "submpo", "mpo": mpo, "where": where}``.  In every mode the stream
may also carry *control events* that are state operations rather than gates.
MPO compression modes apply sub-MPO events with ``gate_with_submpo_``;
DMRG keeps multi-site sub-MPOs as tagged lazy FIT target layers and uses the
same SRC warm-up policy as ordinary DMRG targets.  The stream may also carry:

``mode="su"`` is the simple-update backend. It keeps the MPS core and its
bond gauges separate, initializes missing gauges with
``p.gauge_all_simple_(gauges=..., progbar=False)``, and applies each gate with
``pepsy.gate_simple(..., renorm=True)``. The simple-update core is not
canonicalized.

* ``("measure", pauli, where[, outcome])`` — projectively measure a Pauli
  observable, collapse the MPS onto a sampled (or forced ``outcome``)
  eigenvalue, and append ``(pauli, where, outcome, prob)`` to
  :attr:`MpsOptimizer.measurements`.
* ``("cap", where, vec[, absorb])`` — contract site ``where``'s physical index
  with ``vec`` (e.g. ``[1, 1]``) and absorb the result into the ``absorb``
  (``"left"``/``"right"``) neighbour, shortening the MPS by one site.
* ``("reset", where[, basis])`` — mid-circuit reset of qubit(s) to the ``+1``
  eigenstate of ``basis`` (default ``"Z"``); the MPS length is unchanged.
* ``("measure_reset", basis, where[, outcome])`` — measure each target in
  ``basis``, record the outcome(s), then reset to the ``+1`` eigenstate.

Control events split the stream into gate/subMPO segments run through the
active mode and are applied directly to the state between segments, so the same
stream works in every mode. The default gate path assumes a norm-preserving
stream. Compressed DMRG/FIT, mixed, MPO, swap/permutation, and SVD modes can
restore the raw unitary working norm, preventing deep low-precision
underflow. Non-unitary streams should use ``non_unitary=True``; when
``normalize_every`` is enabled this moves the orthogonality center to one site
after every replay step, normalizes that center tensor, and accumulates the
removed scale into ``p.exponent``. Quimb includes that exponent in ``p.norm()``,
so ``p.norm()`` still reports the represented state norm; inspect a copy with
``exponent=0`` to see the rescaled data norm.

On dense MPS states, a multi-site Pauli measurement is represented as a
bond-two windowed sub-MPO for ``(I + m P) / 2``. DMRG modes attach that operator
to an exact lazy target and use the regular FIT schedule and SRC warm-start to
compress the post-measurement state. Native Symmray and fermionic states keep
their dense projector fallback so charge and dummy-mode metadata are preserved.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
import math
import threading
import time
import types
import warnings
import autoray as ar
import numpy as np
import quimb
import quimb.tensor as qtn

from ...backends import (
    backend_infer,
    backend_signatures_compatible,
    infer_backend_converter_from_sample,
    infer_backend_signature,
)
from ...fitting.local import FIT
from ..._internal.cutoff import dtype_auto_cutoff
from ..._internal.random import backend_random_array
from ..._internal.quimb import (
    quimb_1d_compression_method_available as _quimb_compression_method_available,  # noqa: F401
    quimb_1d_compression_method_supports_seed as _quimb_compression_method_supports_seed,
    require_quimb_1d_compression_method as _require_quimb_compression_method,
)
from ...tensors.core import tn_fidelity
from ...operators.gates import (
    _normalize_gate_entries,
    gate as apply_gate,
    gate_simple as apply_gate_simple,
)
from ...operators import primitives as _gate_primitives
from .layout import (
    MpsGateStreamLayoutFinder,
    _normalize_layout_support,
    _unique_ordered,
)

__all__ = [
    "MpsOptimizer",
    "guess",
    "is_submpo_event",
    "normalize_submpo_where",
    "submpo_event_parts",
    "svd_guess",
]


_SUBMPO_EVENT_NAMES = frozenset({"submpo", "mpo"})
_MISSING = object()
_NORM_INCLUDES_EXPONENT_CACHE = {}
_SHOT_DEFAULT_MAX_BRANCHES = 128
_SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS = 0.1
_DEFAULT_FIT_INIT_STRATEGY = "guess_src"
_DEFAULT_CUTOFF_MODE = "rsum2"
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
_MPO_METHODS_NEED_INTERIOR_WORKAROUND = frozenset(
    {
        "zipup-first",
        "zipup-oversample",
        "fit-zipup",
        "fit-projector",
    }
)
# Keep these method groups separate because they answer different questions:
# ``IGNORE_CUTOFF`` describes methods whose rank is fixed by ``max_bond``;
# ``USE_SEED`` describes methods whose randomized initial projection can be
# replayed; and ``NEED_INTERIOR_WORKAROUND`` describes Quimb wrappers that
# otherwise try to permute a partitioned, non-full-chain site-tag sequence.
# Combining them would make a valid option for one method family leak into a
# different family (for example, forwarding a seed as a contraction option).
_QUIMB_SEED_LOCK = threading.RLock()
_FIT_INIT_STRATEGIES = frozenset(
    {"auto", "direct", "random", "random_expand", "svd_guess"}
    | {f"guess_{method}" for method in _MPO_COMPRESSION_METHODS}
)
# This export-oriented list intentionally contains compatibility totals and
# their named subsets. It is not an additive partition of elapsed FIT time.
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


@dataclass(frozen=True)
class _MpsStreamPlan:
    """Immutable stream metadata with a private backend payload cache.

    ``entries`` and ``event_types`` are the backend-neutral portion of the
    plan. The cache is deliberately the only mutable part: it stores converted
    read-only payloads keyed by backend signature and keeps a strong reference
    to the source payload so object-id reuse cannot return a stale conversion.
    """

    entries: tuple
    event_types: tuple[str, ...]
    has_trajectory_events: bool
    trajectory_plan: object = field(default=None, compare=False, repr=False)
    _backend_cache: dict = field(default_factory=dict, compare=False, repr=False)
    _backend_cache_lock: object = field(
        default_factory=threading.RLock,
        compare=False,
        repr=False,
    )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_backend_cache"] = {}
        state.pop("_backend_cache_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        object.__setattr__(self, "_backend_cache_lock", threading.RLock())

    def get_or_create_backend_payload(self, key, source, factory):
        """Return a cached backend payload, creating it exactly once."""
        with self._backend_cache_lock:
            cached = self._backend_cache.get(key)
            if cached is not None and cached[0] is source:
                return cached[1]
            converted = factory()
            self._backend_cache[key] = (source, converted)
            return converted


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


class _DeprecatedOptionDefault:
    """Readable signature sentinel for compatibility-only keyword values."""

    def __repr__(self):
        return "<deprecated>"


_DEPRECATED_OPTION = _DeprecatedOptionDefault()


def _array_backend_signature(array):
    """Return comparable backend / dtype / device metadata for an array."""
    return infer_backend_signature(array)


def _normalize_event_name(name):
    """Normalize a stream event name for matching."""
    return str(name).replace("-", "_").strip().lower()


def _normalize_submpo_where(where):
    """Normalize sub-MPO support sites to a non-empty tuple of 1D ints."""
    if isinstance(where, Integral):
        return (int(where),)
    if (
        isinstance(where, (tuple, list))
        and len(where) > 0
        and all(isinstance(site, Integral) for site in where)
    ):
        return tuple(int(site) for site in where)
    raise ValueError(
        "subMPO event where must be a non-empty sequence of 1D sites."
    )


def normalize_submpo_where(where):
    """Return canonical 1D support sites for a sub-MPO stream event."""

    return _normalize_submpo_where(where)


def _submpo_event_parts(entry):
    """Return ``(mpo, where)`` if ``entry`` is a sub-MPO event, else ``None``."""
    if (
        isinstance(entry, tuple)
        and len(entry) == 3
        and isinstance(entry[0], str)
        and _normalize_event_name(entry[0]) in _SUBMPO_EVENT_NAMES
    ):
        return entry[1], entry[2]

    if not isinstance(entry, Mapping):
        return None

    kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
    if kind is _MISSING or _normalize_event_name(kind) not in _SUBMPO_EVENT_NAMES:
        return None

    mpo = entry.get(
        "mpo",
        entry.get("submpo", entry.get("operator", entry.get("payload", _MISSING))),
    )
    where = entry.get("where", entry.get("sites", _MISSING))
    if mpo is _MISSING or where is _MISSING:
        raise ValueError(
            "subMPO stream event mappings must contain 'mpo' and 'where'."
        )
    return mpo, where


def submpo_event_parts(entry, *, normalize_where=False):
    """Return ``(mpo, where)`` for a public sub-MPO stream event.

    Returns ``None`` when ``entry`` is not an explicit sub-MPO event. Mapping
    events must contain both an MPO payload and support sites, matching the
    accepted :class:`MpsOptimizer` stream contract.
    """

    parts = _submpo_event_parts(entry)
    if parts is None:
        return None
    mpo, where = parts
    if normalize_where:
        where = _normalize_submpo_where(where)
    return mpo, where


def _is_submpo_event(entry):
    """Return whether ``entry`` is an explicit sub-MPO stream event."""
    return submpo_event_parts(entry) is not None


def is_submpo_event(entry):
    """Return whether ``entry`` is an explicit sub-MPO stream event."""

    return submpo_event_parts(entry) is not None


_CONTROL_EVENT_NAMES = frozenset(
    {"measure", "cap", "reset", "measure_reset", "conditional"}
)
_CONDITIONAL_EVENT_ALIASES = frozenset(
    {"if", "conditional", "condition", "feed_forward", "feedforward"}
)
_MEASURE_RESET_ALIASES = {
    "measure_reset": None,
    "mr": None,
    "mreset": None,
    "measure_and_reset": None,
}
_MEASURE_RESET_AXIS_ALIASES = {
    "mrx": "X",
    "mry": "Y",
    "mrz": "Z",
}
_RESET_AXIS_ALIASES = {
    "reset_x": "X",
    "reset_y": "Y",
    "reset_z": "Z",
}
_RESET_FLIP_AXES = {
    "X": "Z",
    "Y": "X",
    "Z": "X",
}

_PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

_SYMBOLIC_ONE_QUBIT_GATES = {
    "h": _gate_primitives.h,
    "hadamard": _gate_primitives.hadamard,
    "x": _gate_primitives.x,
    "y": _gate_primitives.y,
    "z": _gate_primitives.z,
    "s": _gate_primitives.s,
    "sdg": _gate_primitives.sdg,
    "sdag": _gate_primitives.sdg,
    "t": _gate_primitives.t,
    "tdg": _gate_primitives.tdg,
}
_SYMBOLIC_TWO_QUBIT_GATES = {
    "cnot": _gate_primitives.cnot,
    "cx": _gate_primitives.cx,
    "cy": _gate_primitives.cy,
    "cz": _gate_primitives.cz,
    "swap": _gate_primitives.swap,
    "iswap": _gate_primitives.iswap,
}
_SYMBOLIC_ONE_QUBIT_ROTATIONS = {
    "rx": _gate_primitives.rx,
    "ry": _gate_primitives.ry,
    "rz": _gate_primitives.rz,
}
_SYMBOLIC_TWO_QUBIT_ROTATIONS = {
    "rxx": _gate_primitives.rxx,
    "ryy": _gate_primitives.ryy,
    "rzz": _gate_primitives.rzz,
}
_SYMBOLIC_GATE_NAMES = frozenset(
    {
        *_SYMBOLIC_ONE_QUBIT_GATES,
        *_SYMBOLIC_TWO_QUBIT_GATES,
        *_SYMBOLIC_ONE_QUBIT_ROTATIONS,
        *_SYMBOLIC_TWO_QUBIT_ROTATIONS,
        "sqrt_x",
        "sqrt_x_dag",
        "rot",
    }
)


def _normalize_control_where(where, *, single=False):
    """Return canonical support sites for a control (measure/cap/reset) event."""
    if isinstance(where, Integral):
        sites = (int(where),)
    elif (
        isinstance(where, (tuple, list))
        and len(where) > 0
        and all(isinstance(site, Integral) for site in where)
    ):
        sites = tuple(int(site) for site in where)
    else:
        raise ValueError(
            "control event where must be an int or non-empty sequence of ints."
        )
    if single and len(sites) != 1:
        raise ValueError("cap event where must reference exactly one site.")
    return sites


def _normalize_absorb(absorb):
    """Validate and normalize a cap absorption direction."""
    direction = str(absorb).strip().lower()
    if direction not in {"left", "right"}:
        raise ValueError("cap absorb direction must be 'left' or 'right'.")
    return direction


def _canonical_control_name(name):
    """Return ``(canonical_name, default_axis)`` for a control event name."""
    name = _normalize_event_name(name)
    if name in _CONTROL_EVENT_NAMES:
        return name, None
    if name in _MEASURE_RESET_ALIASES:
        return "measure_reset", _MEASURE_RESET_ALIASES[name]
    if name in _MEASURE_RESET_AXIS_ALIASES:
        return "measure_reset", _MEASURE_RESET_AXIS_ALIASES[name]
    if name in _RESET_AXIS_ALIASES:
        return "reset", _RESET_AXIS_ALIASES[name]
    return None


def _is_axis_string(value):
    """Return whether ``value`` is a non-empty X/Y/Z Pauli-basis string."""
    if not isinstance(value, str):
        return False
    axes = [c for c in value.upper() if not c.isspace()]
    return bool(axes) and all(axis in _RESET_FLIP_AXES for axis in axes)


def _normalize_control_axes(pauli, where, *, event):
    """Return one X/Y/Z axis per site for reset-like controls."""
    axes = [c for c in str(pauli).upper() if not c.isspace()]
    if not axes:
        raise ValueError(f"{event} basis must contain at least one Pauli axis.")
    invalid = [axis for axis in axes if axis not in _RESET_FLIP_AXES]
    if invalid:
        raise ValueError(
            f"{event} basis must use only X, Y, or Z axes, got {pauli!r}."
        )
    if len(axes) == 1 and len(where) > 1:
        axes = axes * len(where)
    if len(axes) != len(where):
        raise ValueError(
            f"{event} basis {pauli!r} has {len(axes)} axis/axes but where "
            f"{where!r} has {len(where)} site(s)."
        )
    return tuple(axes)


def _normalize_control_outcomes(outcome, where, *, event):
    """Return one optional forced outcome per site."""
    if outcome is None:
        return (None,) * len(where)
    if isinstance(outcome, Integral):
        return (int(outcome),) * len(where)
    if isinstance(outcome, (tuple, list)):
        if len(outcome) != len(where):
            raise ValueError(
                f"{event} outcome sequence has length {len(outcome)} but where "
                f"{where!r} has {len(where)} site(s)."
            )
        return tuple(None if value is None else int(value) for value in outcome)
    raise ValueError(
        f"{event} outcome must be an int, None, or a sequence matching where."
    )


def _parse_reset_tuple(entry, default_axis):
    """Return reset payload and support for tuple-form reset aliases."""
    if len(entry) < 2:
        raise ValueError("reset event must be ('reset', where[, basis]).")
    if default_axis is not None:
        where = _normalize_control_where(entry[1])
        if len(entry) > 2:
            raise ValueError(f"{entry[0]!r} does not accept an explicit basis.")
        basis = default_axis
    elif len(entry) >= 3 and _is_axis_string(entry[1]):
        basis = entry[1]
        where = _normalize_control_where(entry[2])
    else:
        where = _normalize_control_where(entry[1])
        basis = entry[2] if len(entry) >= 3 else "Z"
    return "reset", {"axes": _normalize_control_axes(basis, where, event="reset")}, where


def _parse_measure_reset_tuple(entry, default_axis):
    """Return measure-reset payload and support for tuple-form events."""
    if default_axis is None:
        if len(entry) < 3:
            raise ValueError(
                "measure_reset event must be "
                "('measure_reset', basis, where[, outcome])."
            )
        basis = entry[1]
        where = _normalize_control_where(entry[2])
        outcome = entry[3] if len(entry) > 3 else None
    else:
        if len(entry) < 2:
            raise ValueError(f"{entry[0]!r} event must specify where.")
        basis = default_axis
        where = _normalize_control_where(entry[1])
        outcome = entry[2] if len(entry) > 2 else None
    return (
        "measure_reset",
        {
            "axes": _normalize_control_axes(basis, where, event="measure_reset"),
            "outcomes": _normalize_control_outcomes(
                outcome, where, event="measure_reset"
            ),
        },
        where,
    )


def _parse_control_tuple(name, entry, default_axis=None):
    """Return ``(name, payload, where)`` for a tuple-form control event."""
    if name == "measure":
        if len(entry) < 3:
            raise ValueError(
                "measure event must be ('measure', pauli, where[, outcome])."
            )
        pauli = str(entry[1])
        where = _normalize_control_where(entry[2])
        outcome = None if len(entry) <= 3 or entry[3] is None else int(entry[3])
        return "measure", {"pauli": pauli, "outcome": outcome}, where
    if name == "cap":
        if len(entry) < 3:
            raise ValueError("cap event must be ('cap', where, vec[, absorb]).")
        where = _normalize_control_where(entry[1], single=True)
        vec = np.asarray(ar.to_numpy(entry[2]), dtype=complex).ravel()
        absorb = _normalize_absorb(entry[3]) if len(entry) > 3 else "left"
        return "cap", {"vec": vec, "absorb": absorb}, where
    if name == "reset":
        return _parse_reset_tuple(entry, default_axis)
    if name == "measure_reset":
        return _parse_measure_reset_tuple(entry, default_axis)
    raise ValueError(f"Unknown control event {name!r}.")


def _parse_control_mapping(name, entry, default_axis=None):
    """Return ``(name, payload, where)`` for a mapping-form control event."""
    if name == "measure":
        pauli = entry.get("pauli", entry.get("observable", _MISSING))
        where = entry.get("where", entry.get("sites", _MISSING))
        if pauli is _MISSING or where is _MISSING:
            raise ValueError("measure event mapping needs 'pauli' and 'where'.")
        outcome = entry.get("outcome", None)
        return (
            "measure",
            {"pauli": str(pauli), "outcome": None if outcome is None else int(outcome)},
            _normalize_control_where(where),
        )
    if name == "cap":
        where = entry.get("where", entry.get("site", _MISSING))
        vec = entry.get("vec", entry.get("vector", _MISSING))
        if where is _MISSING or vec is _MISSING:
            raise ValueError("cap event mapping needs 'where' and 'vec'.")
        absorb = _normalize_absorb(entry.get("absorb", "left"))
        return (
            "cap",
            {
                "vec": np.asarray(vec, dtype=complex).ravel(),
                "absorb": absorb,
                "compact_labels": bool(entry.get("compact_labels", True)),
            },
            _normalize_control_where(where, single=True),
        )
    if name == "reset":
        where = entry.get("where", entry.get("sites", _MISSING))
        if where is _MISSING:
            raise ValueError("reset event mapping needs 'where'.")
        where = _normalize_control_where(where)
        basis = entry.get("basis", entry.get("pauli", default_axis or "Z"))
        return (
            "reset",
            {"axes": _normalize_control_axes(basis, where, event="reset")},
            where,
        )
    if name == "measure_reset":
        where = entry.get("where", entry.get("sites", _MISSING))
        if where is _MISSING:
            raise ValueError("measure_reset event mapping needs 'where'.")
        where = _normalize_control_where(where)
        basis = entry.get(
            "basis",
            entry.get("pauli", entry.get("observable", default_axis)),
        )
        if basis is None:
            raise ValueError("measure_reset event mapping needs 'basis' or 'pauli'.")
        outcome = entry.get("outcome", None)
        return (
            "measure_reset",
            {
                "axes": _normalize_control_axes(
                    basis, where, event="measure_reset"
                ),
                "outcomes": _normalize_control_outcomes(
                    outcome, where, event="measure_reset"
                ),
            },
            where,
        )
    raise ValueError(f"Unknown control event {name!r}.")


def _conditional_support(action):
    """Return the support of one auditable feed-forward action."""
    parts = _submpo_event_parts(action)
    if parts is not None:
        return _normalize_control_where(parts[1])
    if isinstance(action, Mapping):
        where = action.get("where", action.get("sites", _MISSING))
        if where is _MISSING:
            raise ValueError("conditional action mappings must contain 'where'.")
        return _normalize_control_where(where)
    if not isinstance(action, (tuple, list)) or not action:
        raise ValueError("conditional action must be one gate stream entry.")
    head = action[0]
    if not isinstance(head, str):
        if len(action) != 2:
            raise ValueError("conditional matrix action must be (matrix, where).")
        return _normalize_control_where(action[1])
    name = _normalize_event_name(head)
    if name in {"cnot", "cx", "cy", "cz", "swap"}:
        if len(action) != 3:
            raise ValueError(f"conditional {head!r} action needs two targets.")
        return _normalize_control_where(action[1:])
    if name in {"h", "s", "sdg", "sdag", "sqrt_x", "sqrt_x_dag", "x", "y", "z", "t", "tdg"}:
        if len(action) != 2:
            raise ValueError(f"conditional {head!r} action needs one target.")
        return _normalize_control_where(action[1])
    if name in {"rx", "ry", "rz"}:
        if len(action) != 3:
            raise ValueError(f"conditional {head!r} action needs angle and target.")
        return _normalize_control_where(action[2])
    if name in {"rxx", "ryy", "rzz"}:
        if len(action) != 4:
            raise ValueError(f"conditional {head!r} action needs angle and targets.")
        return _normalize_control_where(action[2:])
    if name == "rot":
        if len(action) != 4:
            raise ValueError("conditional 'rot' action needs angle, axes, and targets.")
        return _normalize_control_where(action[3])
    if name == "measure":
        if len(action) < 3:
            raise ValueError("conditional 'measure' action needs pauli and targets.")
        return _normalize_control_where(action[2])
    if name == "reset" or name in _RESET_AXIS_ALIASES:
        return _parse_reset_tuple(action[1:], _RESET_AXIS_ALIASES.get(name))[2]
    if name in _MEASURE_RESET_ALIASES or name in _MEASURE_RESET_AXIS_ALIASES:
        return _parse_measure_reset_tuple(
            action[1:], _MEASURE_RESET_AXIS_ALIASES.get(name)
        )[2]
    if name == "cap":
        if len(action) < 3:
            raise ValueError("conditional 'cap' action needs where and vector.")
        return _normalize_control_where(action[1], single=True)
    if name in _CONDITIONAL_EVENT_ALIASES:
        return _conditional_event_parts(action)[2]
    raise ValueError(f"Unsupported conditional action {action!r}.")


def _normalize_condition_bit(value):
    """Normalize a feed-forward predicate to a classical bit."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return int(value)
    raise ValueError("conditional value/bit must be 0 or 1.")


def _conditional_event_parts(entry):
    """Return ``(name, payload, where)`` for a classical conditional event.

    Tuple form is ``("if", record, bit, action)``. ``record`` follows Stim's
    convention: negative values are offsets from the current measurement
    record (``-1`` is the latest result), while nonnegative values are
    absolute indices. Mapping form accepts ``record``, ``value``/``bit`` and
    ``then``/``action``. Conditions use computational bits: measurement +1 is
    bit 0 and measurement -1 is bit 1.
    """
    if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
        name = _normalize_event_name(entry[0])
        if name not in _CONDITIONAL_EVENT_ALIASES:
            return None
        if len(entry) != 4:
            raise ValueError(
                'conditional event must be ("if", record, bit, action).'
            )
        record, bit, action = entry[1:]
    elif isinstance(entry, Mapping):
        raw_name = entry.get(
            "kind", entry.get("type", entry.get("event", _MISSING))
        )
        if (
            raw_name is _MISSING
            or _normalize_event_name(raw_name) not in _CONDITIONAL_EVENT_ALIASES
        ):
            return None
        if "record" not in entry:
            raise ValueError("conditional event mapping needs 'record'.")
        record = entry["record"]
        if "value" in entry or "bit" in entry:
            bit = entry.get("value", entry.get("bit"))
        elif "outcome" in entry:
            outcome = int(entry["outcome"])
            if outcome not in (-1, 1):
                raise ValueError("conditional outcome must be +1 or -1.")
            bit = int(outcome < 0)
        else:
            raise ValueError("conditional event mapping needs 'value' or 'bit'.")
        action = entry.get(
            "then", entry.get("action", entry.get("gate", _MISSING))
        )
        if action is _MISSING:
            raise ValueError("conditional event mapping needs 'then' or 'action'.")
    else:
        return None
    if isinstance(record, (bool, np.bool_)) or not isinstance(record, Integral):
        raise TypeError("conditional record must be an integer index or offset.")
    return "conditional", {
        "record": int(record),
        "bit": _normalize_condition_bit(bit),
        "action": action,
    }, _conditional_support(action)


def conditional_event_parts(entry):
    """Public parser for ``if``/feed-forward stream events."""
    return _conditional_event_parts(entry)


def _resolve_conditional(payload, measurement_count):
    """Resolve a normalized conditional against the recorded measurements."""
    record = int(payload["record"])
    index = record if record >= 0 else int(measurement_count) + record
    if index < 0 or index >= int(measurement_count):
        raise ValueError(
            f"conditional record {record} is unavailable after "
            f"{measurement_count} measurement(s)."
        )
    return index, int(payload["bit"])


def _control_event_parts(entry):
    """Return ``(name, payload, where)`` for a control event, else ``None``.

    Control events extend the gate stream with state operations that are not
    plain gates: Pauli measurements, physical-index caps that shorten the MPS,
    and mid-circuit resets. Tuple forms are
    ``("measure", pauli, where[, outcome])``, ``("cap", where, vec[, absorb])``,
    ``("reset", where[, basis])``, and
    ``("measure_reset", basis, where[, outcome])``; equivalent mapping forms use a
    ``"kind"``/``"type"``/``"event"`` selector.
    """
    conditional = _conditional_event_parts(entry)
    if conditional is not None:
        return conditional
    if (
        isinstance(entry, tuple)
        and len(entry) >= 1
        and isinstance(entry[0], str)
    ):
        parsed = _canonical_control_name(entry[0])
        if parsed is not None:
            name, default_axis = parsed
            return _parse_control_tuple(name, entry, default_axis)
    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
        if kind is not _MISSING:
            parsed = _canonical_control_name(kind)
            if parsed is not None:
                name, default_axis = parsed
                return _parse_control_mapping(name, entry, default_axis)
    return None


def _is_control_event(entry):
    """Return whether ``entry`` is a measure/cap/reset control event."""
    return _control_event_parts(entry) is not None


def _normalize_gate_where(where):
    """Return canonical one-/two-site gate locations for MPS replay."""
    if isinstance(where, Integral):
        return (int(where),)
    if isinstance(where, list):
        return tuple(where)
    return where


def _normalize_gate_queue(gates):
    """Return ``(payloads, wheres, event_types)`` from bundled stream input."""
    submpo_parts = _submpo_event_parts(gates)
    if submpo_parts is not None:
        mpo, where = submpo_parts
        return [mpo], [_normalize_submpo_where(where)], ["submpo"]

    control_parts = _control_event_parts(gates)
    if control_parts is not None:
        name, payload, where = control_parts
        return [payload], [where], [name]

    if isinstance(gates, (tuple, list)) and any(
        _is_submpo_event(entry) or _is_control_event(entry) for entry in gates
    ):
        payloads = []
        wheres = []
        event_types = []
        for entry in gates:
            submpo_parts = _submpo_event_parts(entry)
            if submpo_parts is not None:
                mpo, where = submpo_parts
                payloads.append(mpo)
                wheres.append(_normalize_submpo_where(where))
                event_types.append("submpo")
                continue
            control_parts = _control_event_parts(entry)
            if control_parts is not None:
                name, payload, where = control_parts
                payloads.append(payload)
                wheres.append(where)
                event_types.append(name)
                continue
            gate_entries = _normalize_gate_entries(
                (entry,),
                where=None,
                allow_empty=False,
            )
            gate, where = gate_entries[0]
            payloads.append(gate)
            wheres.append(_normalize_gate_where(where))
            event_types.append("gate")
        return payloads, wheres, event_types

    entries = _normalize_gate_entries(gates, where=None, allow_empty=True)
    if not entries:
        return [], [], []
    gate_list, where_list = zip(*entries)
    return (
        list(gate_list),
        [_normalize_gate_where(w) for w in where_list],
        ["gate"] * len(gate_list),
    )


def _symbolic_targets(values, *, name, arity):
    """Normalize positional symbolic gate targets."""
    if len(values) == arity:
        targets = values
    elif len(values) == 1 and isinstance(values[0], (tuple, list)):
        targets = values[0]
    else:
        raise ValueError(
            f"{name!r} gate expects {arity} target sites, got {len(values)}."
        )
    if len(targets) != arity or not all(isinstance(site, Integral) for site in targets):
        raise TypeError(f"{name!r} gate targets must be integer site indices.")
    return tuple(int(site) for site in targets)


def _symbolic_rotation_gate(theta, paulis):
    """Build ``exp(-i * theta * P / 2)`` for a Pauli string ``P``."""
    axes = [axis for axis in str(paulis).upper() if not axis.isspace()]
    if not axes or any(axis not in _PAULI_1Q for axis in axes):
        raise ValueError(
            f"rot Pauli axes must be a non-empty string of I, X, Y, or Z, "
            f"got {paulis!r}."
        )
    pauli = _PAULI_1Q[axes[0]]
    for axis in axes[1:]:
        pauli = np.kron(pauli, _PAULI_1Q[axis])
    theta = float(theta)
    dimension = pauli.shape[0]
    return (
        np.cos(theta / 2.0) * np.eye(dimension, dtype=complex)
        - 1j * np.sin(theta / 2.0) * pauli
    )


def _symbolic_gate_entry(entry):
    """Return ``(gate, where)`` for a named gate entry, or ``None``.

    The grammar mirrors the named stream accepted by ``MpsStabOptimizer``:
    fixed gates use ``(name, site[, site])``, rotations use
    ``(name, angle, site[, site])``, and ``rot`` uses
    ``("rot", angle, paulis, sites)``. Unknown names are left untouched so
    control and stochastic stream parsers can handle them normally.
    """
    if not isinstance(entry, (tuple, list)) or not entry:
        return None
    name = entry[0]
    if not isinstance(name, str):
        return None
    name = _normalize_event_name(name)

    if name in _SYMBOLIC_ONE_QUBIT_GATES:
        if len(entry) != 2:
            raise ValueError(f"{name!r} gate expects one target site.")
        where = _symbolic_targets((entry[1],), name=name, arity=1)
        return _SYMBOLIC_ONE_QUBIT_GATES[name](), where[0]

    if name in {"sqrt_x", "sqrt_x_dag"}:
        if len(entry) != 2:
            raise ValueError(f"{name!r} gate expects one target site.")
        where = _symbolic_targets((entry[1],), name=name, arity=1)
        theta = np.pi / 2.0 if name == "sqrt_x" else -np.pi / 2.0
        return _gate_primitives.rx(theta), where[0]

    if name in _SYMBOLIC_TWO_QUBIT_GATES:
        where = _symbolic_targets(entry[1:], name=name, arity=2)
        return _SYMBOLIC_TWO_QUBIT_GATES[name](), where

    if name in _SYMBOLIC_ONE_QUBIT_ROTATIONS:
        if len(entry) != 3:
            raise ValueError(f"{name!r} gate expects an angle and one target site.")
        where = _symbolic_targets((entry[2],), name=name, arity=1)
        return _SYMBOLIC_ONE_QUBIT_ROTATIONS[name](entry[1]), where[0]

    if name in _SYMBOLIC_TWO_QUBIT_ROTATIONS:
        if len(entry) == 4:
            where = _symbolic_targets(entry[2:], name=name, arity=2)
        elif len(entry) == 3:
            where = _symbolic_targets((entry[2],), name=name, arity=2)
        else:
            raise ValueError(f"{name!r} gate expects an angle and two target sites.")
        return _SYMBOLIC_TWO_QUBIT_ROTATIONS[name](entry[1]), where

    if name == "rot":
        if len(entry) != 4:
            raise ValueError("'rot' gate expects angle, Pauli axes, and target sites.")
        where = _symbolic_targets((entry[3],), name=name, arity=len(
            [axis for axis in str(entry[2]).upper() if not axis.isspace()]
        ))
        return _symbolic_rotation_gate(entry[1], entry[2]), where

    return None


def _resolve_symbolic_gate_entry(entry, converter):
    """Resolve one named gate while preserving non-gate stream events."""
    if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
        name = _normalize_event_name(entry[0])
        if name in _CONDITIONAL_EVENT_ALIASES and len(entry) == 4:
            action = _resolve_symbolic_gate_entry(entry[3], converter)
            if action is not entry[3]:
                resolved = list(entry)
                resolved[3] = action
                return tuple(resolved) if isinstance(entry, tuple) else resolved
        symbolic = _symbolic_gate_entry(entry)
        if symbolic is None:
            return entry
        gate, where = symbolic
        if converter is not None:
            gate = converter(gate)
        return gate, where

    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
        if kind is not _MISSING and _normalize_event_name(kind) in _CONDITIONAL_EVENT_ALIASES:
            for key in ("then", "action", "gate"):
                if key in entry:
                    action = _resolve_symbolic_gate_entry(entry[key], converter)
                    if action is not entry[key]:
                        resolved = dict(entry)
                        resolved[key] = action
                        return resolved
        return entry

    return entry


def _contains_symbolic_gate(entry):
    """Return whether an entry (including a conditional action) is named."""
    if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
        name = _normalize_event_name(entry[0])
        if name in _SYMBOLIC_GATE_NAMES:
            return True
        return (
            name in _CONDITIONAL_EVENT_ALIASES
            and len(entry) == 4
            and _contains_symbolic_gate(entry[3])
        )
    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event", _MISSING)))
        if kind is _MISSING or _normalize_event_name(kind) not in _CONDITIONAL_EVENT_ALIASES:
            return False
        return any(
            key in entry and _contains_symbolic_gate(entry[key])
            for key in ("then", "action", "gate")
        )
    return False


def _resolve_symbolic_gate_stream(entries, *, to_backend=None, backend_sample=None):
    """Resolve named gate entries and optionally place them on a backend."""
    converter = to_backend
    if (
        converter is None
        and backend_sample is not None
        and any(_contains_symbolic_gate(entry) for entry in entries)
    ):
        converter = infer_backend_converter_from_sample(backend_sample)

    resolved = []
    changed = False
    for entry in entries:
        item = _resolve_symbolic_gate_entry(entry, converter)
        resolved.append(item)
        changed |= item is not entry
    return tuple(resolved), changed


def _prepare_gate_stream(gates, *, to_backend=None, backend_sample=None):
    """Compile a stream snapshot and identify trajectory-aware entries.

    The noise module owns the stochastic-entry grammar. Import it lazily here
    so the ordinary MPS optimizer does not create an import cycle at module
    load time. Keeping the raw stream is important: trajectory runners need to
    see the original events, while the single-state path still uses the
    normalized ``G`` / ``where`` / ``event_types`` representation below.
    """
    from ..noise import (  # pylint: disable=import-outside-toplevel
        TrajectoryEvent,
        compile_trajectory_stream,
        _leakage_event_parts,
    )

    trajectory_plan = compile_trajectory_stream(gates)
    entries, changed = _resolve_symbolic_gate_stream(
        trajectory_plan.entries,
        to_backend=to_backend,
        backend_sample=backend_sample,
    )
    if changed:
        # Symbolic gates are ordinary entries, so resolution cannot change
        # any trajectory/control boundary. Replace only the immutable payload
        # tuple and retain the one compilation pass and its metadata.
        trajectory_plan = replace(trajectory_plan, entries=entries)
    event_types = []
    has_trajectory_events = False
    for entry in entries:
        if isinstance(entry, TrajectoryEvent):
            event_types.append("trajectory")
            has_trajectory_events = True
        elif _leakage_event_parts(entry) is not None:
            event_types.append("leakage")
            has_trajectory_events = True
        elif _submpo_event_parts(entry) is not None:
            event_types.append("submpo")
        else:
            control_parts = _control_event_parts(entry)
            event_types.append("gate" if control_parts is None else control_parts[0])
    return _MpsStreamPlan(
        entries=entries,
        event_types=tuple(event_types),
        has_trajectory_events=has_trajectory_events,
        trajectory_plan=trajectory_plan,
    )


def _is_interior_submpo_span(p, where):
    """Return whether ``where`` omits one or more end sites of ``p``."""
    return min(where) > 0 or max(where) < int(p.L) - 1


def _run_seeded_quimb(random_seed, function, *args, **kwargs):
    """Run a Quimb randomized operation with an isolated reproducibility seed.

    Newer Quimb compressors accept ``seed`` directly and use an autoray random
    generator that matches the tensor backend. Older releases use Quimb's
    process-global random generator instead, so retain that fallback without
    leaking a ``seed`` option into unrelated contraction calls. The lock keeps
    the fallback deterministic when optimizers run concurrently.
    """
    if random_seed is None:
        return function(*args, **kwargs)
    method = kwargs.get("method")
    if _quimb_compression_method_supports_seed(method):
        kwargs.setdefault("seed", int(random_seed))
        return function(*args, **kwargs)
    with _QUIMB_SEED_LOCK:
        quimb.seed_rand(int(random_seed))
        return function(*args, **kwargs)


def _apply_submpo_with_interior_workaround_impl(
    p,
    submpo,
    where,
    *,
    chi,
    method,
    cutoff,
    cutoff_mode,
    info=None,
    inplace_mpo=False,
    optimize=None,
    seed=None,
):
    """Apply selected Quimb methods without nested sub-MPO tag permutation.

    Quimb's oversampled zip-up and ``fit-{zipup,projector}`` wrappers call a
    second compression dispatcher internally. When the input is a partitioned
    interior sub-MPO, that nested call defaults to permuting full-chain MPS
    labels and can look for a missing ``I0`` tag. Keep the local sub-MPO
    partition, but reproduce the documented wrapper stages with
    ``permute_arrays=False`` at every level.
    """
    si, sf = min(where), max(where)
    p.canonicalize_((si, sf), info=info)
    p.gate_with_op_lazy_(
        submpo,
        transpose=False,
        inplace_op=inplace_mpo,
    )
    site_tags = [p.site_tag(site) for site in range(si, sf + 1)]
    _, subp = p.partition(site_tags, which="any", inplace=True)

    common = {
        "site_tags": site_tags,
        "max_bond": chi,
        "cutoff": 0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else cutoff,
        "permute_arrays": False,
        "inplace": True,
    }
    if cutoff_mode is not None:
        common["cutoff_mode"] = cutoff_mode
    if optimize is not None:
        common["optimize"] = optimize

    if method in {"zipup-first", "zipup-oversample"}:
        # Quimb's native default is max_bond_oversample = 2 * max_bond.
        qtn.tensor_network_1d_compress(
            subp,
            method="zipup",
            max_bond=2 * chi,
            cutoff=cutoff,
            site_tags=site_tags,
            canonize=True,
            sweep_reverse=True,
            permute_arrays=False,
            optimize=optimize or "auto-hq",
            inplace=True,
        )
        qtn.tensor_network_1d_compress(
            subp,
            method="direct",
            canonize=False,
            **common,
        )
    else:
        # Quimb's fit-* wrappers use an isolated non-random guess, then an
        # eight-sweep one-site FIT with no fitting cutoff.
        guess_method = method.removeprefix("fit-")
        qtn.tensor_network_1d_compress(
            subp,
            method="fit",
            max_bond=chi,
            cutoff=0.0,
            bsz=1,
            max_iterations=8,
            tn_fit={
                "method": guess_method,
                "cutoff": cutoff,
                "canonize": guess_method != "projector",
                "permute_arrays": False,
            },
            **{
                key: value
                for key, value in common.items()
                if key not in {"max_bond", "cutoff", "cutoff_mode"}
            },
        )

    p |= subp
    if info is not None:
        info["cur_orthog"] = (si, si)
    return p


def _apply_submpo_with_interior_workaround(
    p,
    submpo,
    where,
    *,
    chi,
    method,
    cutoff,
    cutoff_mode,
    info=None,
    inplace_mpo=False,
    optimize=None,
    seed=None,
):
    """Apply selected Quimb methods with local tags and optional seeding."""
    seed = seed if method in _MPO_METHODS_USE_SEED else None
    return _run_seeded_quimb(
        seed,
        _apply_submpo_with_interior_workaround_impl,
        p,
        submpo,
        where,
        chi=chi,
        method=method,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        info=info,
        inplace_mpo=inplace_mpo,
        optimize=optimize,
        seed=seed,
    )


def _apply_dense_gate_with_method(
    p,
    gate,
    where,
    *,
    dims,
    chi,
    method,
    cutoff,
    cutoff_mode,
    info=None,
    inplace_mpo=True,
    optimize=None,
    seed=None,
):
    """Apply a dense gate using native Quimb or the interior workaround."""
    if dims is None:
        dims = tuple(p.phys_dim(site) for site in where)
    if not (
        method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
        and _is_interior_submpo_span(p, where)
    ):
        opts = {
            "dims": dims,
            "method": method,
            "max_bond": chi,
            "info": {} if info is None else info,
        }
        if cutoff is not None:
            opts["cutoff"] = (
                0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else cutoff
            )
        if cutoff_mode is not None and method not in _MPO_METHODS_IGNORE_CUTOFF_MODE:
            opts["cutoff_mode"] = cutoff_mode
        if optimize is not None:
            opts["optimize"] = optimize
        if method == "fit-projector":
            # Simple-update gauging can divide by zero on exact product-state
            # bonds. The projector fit remains valid without that optional
            # pre-gauge and Quimb's own implementation supports this path.
            opts["canonize"] = False
        quimb_seed = seed if method in _MPO_METHODS_USE_SEED else None
        return _run_seeded_quimb(quimb_seed, p.gate_nonlocal_, gate, where, **opts)

    submpo = qtn.MatrixProductOperator.from_dense(
        gate,
        dims=dims,
        sites=where,
        L=p.L,
    )
    return _apply_submpo_with_interior_workaround(
        p,
        submpo,
        where,
        chi=chi,
        method=method,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        info=info,
        inplace_mpo=inplace_mpo,
        optimize=optimize,
        seed=seed,
    )


def guess(
    p,
    gate,
    where,
    *,
    chi,
    method="zipup",
    dims=None,
    cutoff=0.0,
    cutoff_mode=None,
    info=None,
    inplace=False,
    seed=None,
):
    """Build a disposable compressed MPS guess for a non-local gate.

    This is deliberately a thin wrapper around Quimb's native operation. The
    exact target and the live MPS remain separate; by default only a deep copy
    is modified. ``seed`` is forwarded only to Quimb methods that support
    randomized initialization or sketching.
    """
    method = str(method).strip().lower()
    if method not in _MPO_COMPRESSION_METHODS:
        raise ValueError(f"Unknown compression guess method: {method}")
    _require_quimb_compression_method(method)
    guess = p if inplace else p.copy(deep=True)
    _apply_dense_gate_with_method(
        guess,
        gate,
        where,
        dims=dims,
        chi=chi,
        method=method,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        info=info,
        seed=seed,
    )
    return guess


def svd_guess(p, gate, where, *, chi, **kwargs):
    """Compatibility wrapper for ``guess(..., method="direct")``."""
    return guess(p, gate, where, chi=chi, method="direct", **kwargs)


class MpsOptimizer:  # pylint: disable=too-many-instance-attributes
    """High-level wrapper for MPS gate-sweep objectives.

    Parameters
    ----------
    p : qtn.MatrixProductState
        Initial MPS state.
    gates : sequence[object] | None, optional
        Canonical bundled gate stream ``((gate, where), ...)`` (outer list/tuple
        accepted). If omitted, start with an empty queue and use
        :meth:`set_gates` or :meth:`add_gates` before ``run``. Each ``gate`` is
        applied on the ket family only (state evolution), using :func:`pepsy.operators.gates.gate`.
        Named entries matching the stabilizer stream grammar are also accepted,
        for example ``("H", 0)`` and ``("rzz", theta, 0, 1)``. They are
        materialized as ordinary gate matrices before stream validation.
        ``where`` supports one- or two-site locations in 1D/2D/3D forms.
        For a bare Quimb method such as ``mode="src"`` or the qualified
        ``mode="quimb-<method>"`` (with aliases ``"quimb"`` and legacy
        ``"mpo-<method>"`` / ``"mpo"``), entries may
        also have the explicit sub-MPO form
        ``("submpo", mpo, where)`` or mapping form
        ``{"kind": "submpo", "mpo": mpo, "where": where}``, with a 1D
        support ``where``. :meth:`submpo_event` builds the tuple form.
        In any mode the stream may also carry control events
        ``("measure", pauli, where[, outcome])``,
        ``("cap", where, vec[, absorb])``, ``("reset", where[, basis])``, and
        ``("measure_reset", basis, where[, outcome])`` (built by
        :meth:`measure_event`, :meth:`cap_event`, :meth:`reset_event`, and
        :meth:`measure_reset_event`). Classical feed-forward events use
        ``("if", record, bit, action)`` or the equivalent mapping form and
        apply ``action`` only when the referenced measurement has the
        requested computational bit (``+1 -> 0``, ``-1 -> 1``); negative
        records are offsets from the latest measurement. A
        ``cap`` event shortens the MPS, so later event site labels refer to the
        shortened chain. Stream-local trajectory events and stochastic entries
        are also accepted. They are replayed through the shot runner when
        :meth:`run` is called, so state-dependent channels, measurements, and
        feed-forward actions are sampled independently per trajectory.
        Numeric gate and sub-MPO payloads must already match the MPS backend
        and device. Named gate entries are generated internally and use
        ``to_backend`` (or the initial-state backend when it is omitted).
        Mismatches in user-supplied numeric payloads are rejected with
        preparation guidance; use an explicit converter before constructing
        the optimizer or calling :meth:`set_gates`.
    chi : int
        Positive target/max bond dimension used by compressed modes. Mixed mode
        requires the initial MPS to have ``max_bond() <= chi`` and keeps its
        committed DMRG/MPO results at or below this limit.
    mode : {"fit", "dmrg", "dmrg1", "dmrg2", "dmrg3", "<quimb-method>", "quimb-<method>", "quimb", "mpo-<method>", "mpo", "mix", "swap", "perm", "svd", "su", "exact"}, default="dmrg"
        Optimization backend. ``"fit"`` is the clear alias of the historical
        ``"dmrg"`` spelling. ``"dmrg1"`` uses at most two two-site growth
        sweeps, then one-site refinement; once every bond reaches its
        attainable physical/``chi`` ceiling, it latches one-site updates
        for the rest of the replay. An already-capped window starts
        directly with one-site sweeps. ``"dmrg2"`` uses two-site updates
        for the required warm-up (two sweeps by default), then one-site
        refinement. ``"dmrg3"`` follows the same fixed warm-up policy with
        three-site updates before one-site refinement. ``"mix"`` defaults to
        a direct/MPO warm-up on under-capacity active bonds, followed by
        transactional one-site DMRG/FIT; explicit ``fit_block_size=2`` or
        ``3`` opts into mixed block-FIT transactions.
    contraction_opt : object | None, default="auto-hq"
        Canonical contraction path optimizer keyword.
    ind_id : str, default="k{}"
        Format string for site index labels used by exact gate application.
        Use "k{},{}" when gate sites are 2D coordinates like ``(i, j)``.
    inplace : bool, default=False
        Whether to optimize the provided input state object directly. If
        ``False``, a copy is made and the original input remains unchanged.
    The input state and gate stream are snapshotted at construction. Repeated
    ``run(shots=...)`` calls therefore restart every trajectory from the same
    initial state rather than continuing from an earlier ensemble.
    gauges : dict | None, default=None
        Simple-update bond gauges used only by ``mode="su"``. The dictionary
        is mutated in place and is exposed as :attr:`gauges`. If omitted, the
        optimizer initializes it with ``p.gauge_all_simple_(...)`` before the
        first simple-update gate.
    to_backend : callable | None, default=None
        Optional converter for named gate entries. For example,
        ``to_backend=pepsy.backend_torch(dtype=torch.complex64)`` converts
        the matrices generated for ``("H", 0)`` and ``("rzz", theta, 0, 1)``
        before they cross the stream boundary. When omitted, the converter is
        inferred from the initial MPS for named gates. Existing numeric gate
        and sub-MPO payloads retain the strict explicit-preparation contract.
    Attributes
    ----------
    measurements : list[tuple]
        Results of ``("measure", ...)`` control events, appended in order as
        ``(pauli, where, outcome, prob)`` where ``outcome`` is ``+1``/``-1`` and
        ``prob`` is the Born probability of that outcome before collapse.
        Mid-circuit ``reset`` measurements are not recorded here.
    normalizations : list[dict]
        Automatic normalization events recorded during :meth:`run`. Each entry
        stores the 1-based gate step, removed local squared scale,
        orthogonality span, tensor sites that were rescaled, and resulting
        base-10 ``p.exponent``. The raw tensor data are rescaled; the
        represented norm remains available through ``p.norm()`` because quimb
        applies ``p.exponent``.
    norm_events : list[dict]
        Automatic norm-survival records for compressed gates and physical
        projective/Kraus boundaries. Physical branch probabilities are stored
        on their event but are not multiplied into compression infidelity.
    quality_checks : list[dict]
        Optional finite-data and canonical-gauge health records from
        ``run(quality_check_every=...)``.
    last_run_timing : dict | None
        Most recent opt-in replay timing record from ``run(timing=True)``.
        The record contains total replay time, inclusive stage totals, and,
        for mixed mode, the final ``last_mix_summary``. Use
        :meth:`get_run_timing` for a copy.
    gauges : dict
        Simple-update bond gauges. In ``mode="su"``, ``p`` is the gauged core
        and the physical state is recovered with ``p.gauge_simple_insert(gauges)``.
    p_ungauged : qtn.MatrixProductState | None
        In ``mode="su"``, an automatically refreshed physical-state copy with
        the current simple-update gauges inserted. ``p`` remains the core used
        for continued simple-update evolution.
    logical_order : list[int]
        Persistent-layout mapping from physical MPS position to logical site.
        The list is identity until :meth:`apply_layout` is called.
    """

    _DMRG_MODE_ALIASES = {"dmrg1": 1, "dmrg2": 2, "dmrg3": 3}
    _ALLOWED_MODES = frozenset(
        {
            "dmrg",
            "dmrg1",
            "dmrg2",
            "dmrg3",
            "quimb",
            "mpo",
            "mix",
            "swap",
            "perm",
            "svd",
            "su",
            "exact",
        }
        | _MPO_COMPRESSION_METHODS
        | {f"mpo-{method}" for method in _MPO_COMPRESSION_METHODS}
        | {f"quimb-{method}" for method in _MPO_COMPRESSION_METHODS}
    )
    LayoutFinder = MpsGateStreamLayoutFinder
    _ALLOWED_SUBMPO_METHODS = _MPO_COMPRESSION_METHODS
    _PROGBAR_COLORS = {
        "dmrg": "#1f77b4",
        "mpo": "#2ca02c",
        "mix": "#17becf",
        "swap": "#ff7f0e",
        "perm": "#8c564b",
        "svd": "#d62728",
        "su": "#e377c2",
        "exact": "#9467bd",
    }

    @classmethod
    def _normalize_mode(cls, mode):
        """Validate and normalize execution mode."""
        mode_norm = str(mode).strip().lower()
        # ``fit`` names the algorithm while ``dmrg`` preserves the historical
        # mode spelling. DMRG1/2/3 are readable block-size aliases that share
        # the same implementation and are normalized to ``dmrg``. The alias
        # is recorded by the constructor before this function runs, so the
        # shared implementation can still select the requested schedule.
        if mode_norm == "fit" or mode_norm in cls._DMRG_MODE_ALIASES:
            mode_norm = "dmrg"
        elif mode_norm in _MPO_COMPRESSION_METHODS:
            # Bare Quimb method names are the user-facing spelling. Keep the
            # qualified form as the canonical internal mode so old
            # ``quimb-*`` and ``mpo-*`` aliases continue to behave identically.
            mode_norm = f"quimb-{mode_norm}"
        if mode_norm not in cls._ALLOWED_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        return mode_norm

    @classmethod
    def _is_mpo_mode(cls, mode):
        """Return whether ``mode`` selects Quimb compression."""
        mode_norm = str(mode).strip().lower()
        return (
            mode_norm in {"mpo", "quimb"}
            or mode_norm in _MPO_COMPRESSION_METHODS - {"fit"}
            or mode_norm.startswith(("mpo-", "quimb-"))
        )

    @classmethod
    def _progress_mode_name(cls, mode):
        """Return the short active mode name shown by a replay progress bar."""
        mode_norm = str(mode).strip().lower()
        if cls._is_mpo_mode(mode_norm):
            return cls._mode_mpo_method(mode_norm)
        return mode_norm

    @classmethod
    def _mode_mpo_method(cls, mode):
        """Return the compression method encoded by a Quimb mode name."""
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
        """Resolve an explicit method or the method encoded by ``self.mode``."""
        if method is None:
            return self._mode_mpo_method(self.mode)
        return self._normalize_submpo_method(method)

    @classmethod
    def _dmrg_alias_block_size(cls, mode):
        """Return the fixed block size requested by a DMRG mode alias."""
        return cls._DMRG_MODE_ALIASES.get(str(mode).strip().lower())

    @staticmethod
    def _effective_max_bond(p=None):
        """Return a numeric maximum bond, treating product-state ``None`` as 1."""
        value = p.max_bond() if p is not None else None
        return 1 if value is None else int(value)

    @classmethod
    def _normalize_submpo_method(cls, method):
        """Validate and normalize the sub-MPO compression method."""
        method_norm = str(method).strip().lower()
        if method_norm not in cls._ALLOWED_SUBMPO_METHODS:
            raise ValueError(f"Unknown subMPO method: {method}")
        _require_quimb_compression_method(method_norm)
        return method_norm

    def _submpo_compress_opts(
        self,
        method,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Return compression options for a sub-MPO method."""
        opts = {}
        # ``cutoff`` controls discarded singular weight for ordinary methods.
        # SRC/SRCMPS are rank-controlled randomized projections, so Quimb
        # intentionally ignores a singular-value cutoff for those methods.
        opts["cutoff"] = (
            0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else cutoff
        )
        if (
            cutoff_mode is not None
            and method not in _MPO_METHODS_IGNORE_CUTOFF_MODE
        ):
            opts["cutoff_mode"] = cutoff_mode
        if method == "fit-projector":
            # The optional simple-update pre-gauge is singular on exact
            # product-state bonds. Match the dense-gate path and let the
            # projector fit run without that gauge.
            opts["canonize"] = False
        if method == "direct":
            return opts
        # Quimb's ``auto`` paths already choose their own contraction tree.
        # Only forward an explicit optimizer so this wrapper does not turn a
        # harmless default into an unsupported nested contraction option.
        optimize = self.contraction_opt
        if optimize is None:
            return opts
        if isinstance(optimize, str) and optimize.strip().lower() in {
            "auto",
            "auto-hq",
        }:
            return opts
        opts["optimize"] = optimize
        return opts

    @staticmethod
    def submpo_event(mpo, where):
        """Return a canonical explicit sub-MPO stream event.

        The returned entry can be placed directly inside the ``gates`` stream
        for the Quimb compression mode family. ``where`` is restricted to 1D integer MPS
        sites.
        """

        return ("submpo", mpo, _normalize_submpo_where(where))

    @staticmethod
    def submpo_event_parts(entry, *, normalize_where=False):
        """Return ``(mpo, where)`` when ``entry`` is a sub-MPO event."""

        return submpo_event_parts(entry, normalize_where=normalize_where)

    @staticmethod
    def is_submpo_event(entry):
        """Return whether ``entry`` is an explicit sub-MPO stream event."""

        return is_submpo_event(entry)

    @staticmethod
    def measure_event(pauli, where, outcome=None):
        """Return a canonical Pauli-measurement stream event.

        Collapses the MPS onto a sampled (or forced ``outcome``) eigenvalue of
        the Pauli observable ``pauli`` on ``where`` and appends the result to
        :attr:`measurements`. ``pauli`` is a string such as ``"Z"`` or ``"ZZ"``
        with one axis per site in ``where``.
        """
        where = _normalize_control_where(where)
        if outcome is None:
            return ("measure", str(pauli), where)
        return ("measure", str(pauli), where, int(outcome))

    @staticmethod
    def cap_event(where, vec, absorb="left"):
        """Return a canonical cap stream event.

        Contracts the physical index of site ``where`` with ``vec`` (e.g.
        ``[1, 1]``) and absorbs the resulting matrix into the ``absorb``
        neighbour (``"left"`` or ``"right"``), shortening the MPS by one site.
        """
        (site,) = _normalize_control_where(where, single=True)
        return ("cap", site, np.asarray(vec, dtype=complex).ravel(), _normalize_absorb(absorb))

    @staticmethod
    def reset_event(where, basis="Z"):
        """Return a canonical mid-circuit reset stream event.

        Resets qubit(s) ``where`` to the ``+1`` eigenstate of ``basis`` by a
        measurement collapse followed by a conditional anticommuting Pauli flip.
        The MPS length is unchanged and the internal measurements are not
        recorded. The legacy ``basis="Z"`` form returns ``("reset", where)``.
        """
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(basis, where, event="reset")
        if all(axis == "Z" for axis in axes):
            return ("reset", where)
        return ("reset", where, "".join(axes))

    @staticmethod
    def measure_reset_event(pauli, where, outcome=None):
        """Return a canonical measure-then-reset stream event.

        Each target is measured in the corresponding single-site Pauli basis,
        the outcome is appended to :attr:`measurements`, and the target is then
        reset to the ``+1`` eigenstate of that basis. A one-character ``pauli``
        is broadcast across multiple sites.
        """
        where = _normalize_control_where(where)
        axes = _normalize_control_axes(pauli, where, event="measure_reset")
        if outcome is None:
            return ("measure_reset", "".join(axes), where)
        outcomes = _normalize_control_outcomes(
            outcome, where, event="measure_reset"
        )
        if len(outcomes) == 1:
            return ("measure_reset", "".join(axes), where, outcomes[0])
        return ("measure_reset", "".join(axes), where, outcomes)

    @staticmethod
    def control_event_parts(entry):
        """Return ``(name, payload, where)`` when ``entry`` is a control event."""

        return _control_event_parts(entry)

    @staticmethod
    def is_control_event(entry):
        """Return whether ``entry`` is a measure/cap/reset/MR control event."""

        return _is_control_event(entry)

    @classmethod
    def gate_stream_layout(  # pylint: disable=too-many-locals
        cls,
        gate_stream,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        order="quality",
        objective="locality",
        refine_passes=8,
        refine_numba=True,
        spectral_dense_max=512,
        recursive_dense_max=1024,
        nevergrad_budget=64,
        nevergrad_seed=0,
        nevergrad_optimizer="OnePlusOne",
        kahypar_config_path=None,
        kahypar_seed=0,
        from_scratch=False,
        weight_fn=None,
        weight_mode="auto",
        schmidt_max_dim=4,
        max_operator_qubits=8,
    ):
        """Find a good 1D MPS layout for a bundled gate stream.

        The layout depends only on the stream supports, not on MPS tensor
        values.  The returned plan includes the optimized site order,
        old-site to new-position map, original stream locations, and internal
        mapped locations. It does not mutate or return a replacement gate
        stream.

        Parameters
        ----------
        gate_stream
            Canonical bundled stream accepted by :class:`MpsOptimizer`,
            including explicit sub-MPO events.
        sites : sequence[hashable] | None
            Complete logical site labels to arrange. If omitted, sites are
            inferred from first use in ``gate_stream`` unless ``L`` is given.
        L : int | None
            Convenience for ``sites=range(L)``.
        lattice_shape : pair of int, optional
            The ``(Lx, Ly)`` shape used by named geometric orders such as
            ``"snake"``, ``"folded-snake"``, and ``"hilbert"``. The product
            must equal the number of MPS sites.
        lattice_site : callable, optional
            Optional ``(x, y) -> logical_site`` mapper for named geometric
            orders. The default is ``x * Ly + y``.
        objective : {"locality", "compression"}
            ``"locality"`` minimizes support span and cut congestion using
            event weights. ``"compression"`` ranks layouts by operator-
            Schmidt load over the MPS cuts, with path span as a tie-breaker.
        order : str
            One of ``"quality"``/``"auto"``/``"best"``, ``"recursive"``,
            ``"input"``, ``"degree"``, ``"bfs"``, ``"spectral"``,
            ``"nevergrad"``, ``"kahypar"``, the geometric lattice presets
            ``"row-major"``, ``"col-major"``, ``"snake"``,
            ``"folded-snake"``, and ``"hilbert"``, or the ``"*_refined"``
            variants. Geometric presets require ``lattice_shape``.
        refine_passes : int
            Number of greedy adjacent-swap improvement passes.
        refine_numba : bool
            Use the optional numba polish kernel when numba is installed.
        spectral_dense_max : int
            Maximum site count for dense spectral ordering. ``"auto"`` falls
            back to non-spectral candidates above this size.
        recursive_dense_max : int
            Maximum site count for dense recursive spectral bisection.
        nevergrad_budget : int
            Black-box optimization budget for optional nevergrad candidates.
        nevergrad_seed : int | None
            NumPy seed used while constructing the optional nevergrad candidate.
        nevergrad_optimizer : str
            Name of the nevergrad optimizer class to use.
        from_scratch : bool
            Omit the original site order from the searched candidates and from
            Nevergrad inoculation. The original order remains available in
            ``input_stats`` as a comparison baseline.
        kahypar_config_path : path-like | None
            KaHyPar ``.ini`` config path. If omitted, ``PEPSY_KAHYPAR_CONFIG``
            is used. KaHyPar is skipped unless a config is supplied.
        kahypar_seed : int
            Seed forwarded to KaHyPar recursive bisection.
        weight_fn : callable | None
            Optional ``weight_fn(payload, support, event_type)`` override for
            per-event layout weights.
        weight_mode : {"auto", "count", "angle", "operator_schmidt"}
            Built-in event weighting heuristic. ``"auto"`` uses angle metadata
            when present, otherwise a cheap two-site operator-Schmidt proxy for
            small dense gates, falling back to count weights.
        schmidt_max_dim : int
            Maximum local dimension for the optional operator-Schmidt proxy.
        max_operator_qubits : int | None
            Maximum support size for exact dense rank probes in the
            compression objective. Larger or opaque operators use a
            conservative operator-space rank bound and are marked as bounded
            in the returned diagnostics.

        Returns
        -------
        dict
            Layout plan with ``qubit_inds``/``site_order``, ``layout``/
            ``site_map``, original ``where``, internal ``mapped_where``,
            ``stats``, and ``candidate_scores``.
        """

        finder = cls.LayoutFinder(
            gate_stream,
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
        )
        return finder.run(
            order=order,
            objective=objective,
            refine_passes=refine_passes,
            refine_numba=refine_numba,
            spectral_dense_max=spectral_dense_max,
            recursive_dense_max=recursive_dense_max,
            nevergrad_budget=nevergrad_budget,
            nevergrad_seed=nevergrad_seed,
            nevergrad_optimizer=nevergrad_optimizer,
            kahypar_config_path=kahypar_config_path,
            kahypar_seed=kahypar_seed,
            from_scratch=from_scratch,
            weight_fn=weight_fn,
            weight_mode=weight_mode,
            schmidt_max_dim=schmidt_max_dim,
            max_operator_qubits=max_operator_qubits,
        )

    @classmethod
    def find_gate_stream_layout(cls, gate_stream, **kwargs):
        """Alias for :meth:`gate_stream_layout`."""

        return cls.gate_stream_layout(gate_stream, **kwargs)

    def layout_finder(
        self,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
    ):
        """Return a layout finder for the currently queued gate stream."""

        return type(self).LayoutFinder.from_optimizer(
            self,
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
        )

    def current_gate_stream_layout(
        self,
        *,
        sites=None,
        L=None,
        lattice_shape=None,
        lattice_site=None,
        **kwargs,
    ):
        """Find a layout for the optimizer's currently queued gate stream."""

        return self.layout_finder(
            sites=sites,
            L=L,
            lattice_shape=lattice_shape,
            lattice_site=lattice_site,
        ).run(**kwargs)

    @staticmethod
    def _split_layout_finder_kwargs(layout_kwargs):
        """Split finder-construction options from per-run layout options."""
        kwargs = {} if layout_kwargs is None else dict(layout_kwargs)
        finder_kwargs = {}
        for name in ("lattice_shape", "lattice_site"):
            if name in kwargs:
                finder_kwargs[name] = kwargs.pop(name)
        return finder_kwargs, kwargs

    def select_layout_for_compression(
        self,
        *,
        sites=None,
        L=None,
        layout_kwargs=None,
        pilot_candidates=4,
        pilot_steps=None,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        run_kwargs=None,
    ):
        """Select an MPS layout using a bounded, state-aware pilot replay.

        The finder first produces cheap static candidates with
        ``objective="compression"``. The best ``pilot_candidates`` are then
        replayed on independent copies of the current MPS using the real
        execution mode, ``chi``, cutoff, and backend. The returned plan is
        non-mutating and contains ``pilot`` diagnostics for every candidate.

        This method is intentionally separate from :meth:`run`: layout
        selection can be expensive and should be explicit in production
        workflows. ``pilot_steps`` limits the replay prefix while preserving
        the original optimizer and gate queue.
        """
        if self.mode == "exact":
            raise ValueError(
                "compression layout pilots require an MPS compression mode, "
                "not mode='exact'."
            )
        if self._persistent_layout_plan is not None:
            raise ValueError(
                "compression layout pilots require an optimizer without a "
                "persistent layout; create the pilot before apply_layout()."
            )
        try:
            pilot_candidates = int(pilot_candidates)
        except (TypeError, ValueError) as exc:
            raise ValueError("pilot_candidates must be a positive integer.") from exc
        if pilot_candidates < 1:
            raise ValueError("pilot_candidates must be a positive integer.")
        if pilot_steps is not None:
            try:
                pilot_steps = int(pilot_steps)
            except (TypeError, ValueError) as exc:
                raise ValueError("pilot_steps must be a positive integer or None.") from exc
            if pilot_steps < 1:
                raise ValueError("pilot_steps must be a positive integer or None.")

        kwargs = dict(layout_kwargs or {})
        finder_kwargs, kwargs = self._split_layout_finder_kwargs(kwargs)
        finder = self.layout_finder(
            sites=sites,
            L=L,
            **finder_kwargs,
        )
        kwargs["objective"] = "compression"
        static_plan = finder.run(**kwargs)
        candidates = dict(static_plan.get("candidate_plans", {}))
        if not candidates:
            candidates = {static_plan["selected_order"]: static_plan}
        ranked_names = sorted(
            candidates,
            key=lambda name: candidates[name]["stats"].get(
                "compression_score", candidates[name]["stats"].get("score", 0.0)
            ),
        )[:pilot_candidates]

        base_run_kwargs = dict(run_kwargs or {})
        base_run_kwargs.setdefault("progbar", False)
        base_run_kwargs.setdefault("layout_report", False)
        base_run_kwargs.setdefault("cutoff", cutoff)
        base_run_kwargs.setdefault("cutoff_mode", cutoff_mode)
        pilot_reports = {}
        successful = []
        for name in ranked_names:
            trial = self.copy()
            if pilot_steps is not None:
                trial.G = trial.G[:pilot_steps]
                trial.where = trial.where[:pilot_steps]
                trial.event_types = trial.event_types[:pilot_steps]
            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    trial.run(layout=candidates[name], **base_run_kwargs)
                elapsed = time.perf_counter() - started
                final_bond = int(trial.p.max_bond())
                report = {
                    "status": "ok",
                    "elapsed_seconds": float(elapsed),
                    "final_bond": final_bond,
                    "pilot_steps": len(trial.G),
                }
                successful.append((final_bond, elapsed, name))
            except Exception as exc:  # pragma: no cover - backend-specific
                report = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "pilot_steps": len(trial.G),
                }
            pilot_reports[name] = report

        if not successful:
            raise RuntimeError(
                "All MPS compression layout pilot candidates failed. "
                f"Diagnostics: {pilot_reports!r}"
            )
        selected_name = min(successful)[-1]
        selected = dict(candidates[selected_name])
        selected["selected_order"] = selected_name
        selected["pilot"] = {
            "objective": "compression",
            "pilot_candidates": tuple(ranked_names),
            "selected_order": selected_name,
            "reports": pilot_reports,
        }
        selected["candidate_plans"] = candidates
        return selected

    def plot_layout(
        self,
        plan=None,
        *,
        sites=None,
        L=None,
        layout_kwargs=None,
        **plot_kwargs,
    ):
        """Plot the current gate-stream layout and selected MPS order.

        This is a convenience wrapper around
        :meth:`MpsGateStreamLayoutFinder.plot`. It returns ``(fig, ax)`` and
        does not mutate the optimizer or install the plotted layout. When
        ``plan`` is omitted, the finder computes its default quality plan;
        pass ``layout_kwargs`` to customize that search.
        """
        finder_kwargs, run_kwargs = self._split_layout_finder_kwargs(layout_kwargs)
        finder = self.layout_finder(
            sites=sites,
            L=L,
            **finder_kwargs,
        )
        if plan is None:
            plan = finder.run(**run_kwargs)
        return finder.plot(plan, **plot_kwargs)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        p,
        gates=None,
        chi=None,
        mode="dmrg",
        contraction_opt="auto-hq",
        ind_id="k{}",
        inplace=False,
        gauges=None,
        _capture_initial=True,
        to_backend=None,
    ):
        if chi is None:
            if isinstance(gates, Integral):
                chi = int(gates)
                gates = []
            else:
                raise TypeError(
                    "chi must be provided. Use MpsOptimizer(p, gates, chi) "
                    "or MpsOptimizer(p, chi) for an empty gate queue."
                )
        if not isinstance(chi, Integral) or int(chi) < 1:
            raise ValueError("chi must be a positive integer.")

        self.inplace = bool(inplace)
        self.p = self._install_represented_norm(p if self.inplace else p.copy())
        # Dynamic cap streams shorten the live MPS during replay. Keep a
        # small structural ledger separate from norm/compression diagnostics so
        # callers can inspect the effective register length without inferring
        # it from tensor tags or a private event queue.
        self._initial_mps_length = int(getattr(self.p, "L", 0))
        self._mps_length_history = [self._initial_mps_length]
        self.cap_history = []
        # ``L_eff`` is an operation-active support ledger, not a dense-state
        # or Schmidt-rank measurement. A product MPS starts at zero; replay
        # events activate their current site interval and caps remap/remove
        # that support as the register shrinks.
        self._effective_active_positions = set()
        self._effective_length_history = [0]
        self._effective_site_history = [()]
        self._effective_event_history = []
        self._initial_p = self.p.copy() if _capture_initial else None
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        self._symbolic_gate_to_backend = to_backend
        # A normal optimizer owns its stream cache. Shot-created optimizers
        # explicitly opt into sharing the immutable plan cache after
        # construction; initialize both fields before installing the plan so
        # the constructor follows the same path as set_gates/add_gates.
        self._shared_backend_cache = False
        self._backend_cache_plan = None
        plan = _prepare_gate_stream(
            gates,
            to_backend=to_backend,
            backend_sample=self._state_backend_like_for(self.p),
        )
        normalized_queue = self._normalize_stream_plan_queue(plan)
        self._validate_normalized_gate_queue(normalized_queue)
        self._install_stream_plan(plan, normalized_queue=normalized_queue)
        self.chi = int(chi)
        mode_name = str(mode).strip().lower()
        self._dmrg_mode_alias = (
            mode_name if mode_name in self._DMRG_MODE_ALIASES else None
        )
        self._dmrg_mode_block_size = self._dmrg_alias_block_size(mode)
        self.mode = self._normalize_mode(mode)
        self._validate_canonical_boundary(self.p, self.mode)
        self.contraction_opt = "auto-hq" if contraction_opt is None else contraction_opt
        self.ind_id = str(ind_id)
        if gauges is not None and not isinstance(gauges, dict):
            raise TypeError("gauges must be a mutable dictionary or None.")
        self.gauges = {} if gauges is None else gauges
        self.p_ungauged = None
        self._su_gauges_supplied = gauges is not None
        self._su_gauges_ready = False
        self._su_gauges_state = None
        self._su_force_regauge = False

        self.info_c = {}
        # Physical MPS position -> logical site. ``perm`` mode updates this
        # lazily as non-local gates leave their swap network in place.
        self.qubits = list(range(int(getattr(self.p, "L", 0))))
        # Persistent layout position -> logical site. Unlike ``qubits`` this
        # mapping is installed once by ``apply_layout`` and is never restored.
        self.logical_order = list(self.qubits)
        self._persistent_layout_plan = None
        self.layout_plan = None
        self.normalizations = []
        self.norm_events = []
        self._norm_log_survival = 0.0
        self.quality_checks = []
        self.last_layout_plan = self._persistent_layout_plan
        self.scheduled_layout_plan = None
        self.scheduled_site_order = None
        self.mix_history = []
        self.last_mix_summary = None
        self.last_run_timing = None
        self._timing_state = None
        self._fit_copy_policy_cache = None
        # Optional diagnostics, disabled for normal replay in every mode.
        # run(finite_check=True) enables them only for that replay and warns.
        self._finite_check_enabled = False
        self._mix_dmrg_disabled_reason = None
        self._mix_dmrg_failed_sweep = None
        self._last_dmrg_fit_diagnostics = None
        self._dmrg1_one_site_locked = False
        # Native Symmray one-site writeback can retain duplicate fermionic
        # dummy modes when a product-state DMRG1 warm-up has just saturated
        # its bonds. Keep that narrow initialization case on two-site FIT;
        # non-product native states retain the documented DMRG1 schedule.
        self._dmrg1_native_product_two_site = (
            self._dmrg_mode_alias == "dmrg1"
            and self._is_native_fermionic_product_state(self.p)
        )
        self.measurements = []
        self._rng = np.random.default_rng()
        self._unitary_previous_norm = None
        self.backend = None
        self.backend_dtype = None
        self.backend_device = None
        self.array_backend = None
        self.backend_info()
        self._init_canonicalization()

    def _info_for_state(self, p, info=None):
        """Return canonical metadata owned by ``p``.

        ``info_c`` describes the live optimizer state only. Diagnostic and
        target-building paths frequently work on MPS copies, for which using
        that dictionary would make a temporary state's center look like the
        live state's center. Such copies get an isolated metadata dictionary.
        """
        if info is not None:
            return info
        return self.info_c if p is self.p else {}

    def _current_orthog(self, p=None, *, info=None):
        """Return cached ``(min_site, max_site)`` orthogonality span.

        Cached entries may be ``"calc"`` / ``None`` (recompute), an ``int``,
        or a 1- or 2-tuple. The stored form is always a 2-tuple with
        ``min <= max``.
        """
        state = self.p if p is None else p
        state_info = self._info_for_state(state, info)
        cur = state_info.get("cur_orthog", "calc")
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

        state_info["cur_orthog"] = cur
        return cur

    def _record_orthog_span(self, p, where, *, info=None):
        """Record a span known to remain canonical after a state update."""
        state_info = self._info_for_state(p, info)
        state_info["cur_orthog"] = self._normalize_span(where)
        return state_info["cur_orthog"]

    def _format_ind(self, site):
        """Format a site id using ``self.ind_id``."""
        if isinstance(site, (tuple, list)):
            return self.ind_id.format(*site)
        return self.ind_id.format(site)

    @staticmethod
    def _infer_gate_dims(gate, where):
        """Infer physical dimensions from an explicit rank-2n gate tensor."""
        shape = getattr(gate, "shape", None)
        if shape is None:
            return None
        try:
            shape = tuple(int(d) for d in shape)
        except (TypeError, ValueError):
            return None
        nsites = len(where)
        if len(shape) != 2 * nsites:
            return None
        dims_in = shape[:nsites]
        dims_out = shape[nsites:]
        if dims_in != dims_out:
            return None
        return dims_in

    @staticmethod
    def _is_symmray_array(value):
        """Return whether ``value`` looks like a Symmray block-sparse array."""
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @classmethod
    def _has_symmray_data(cls, tn):
        """Return whether any tensor data in ``tn`` is Symmray-backed."""
        return any(
            cls._is_symmray_array(tensor.data)
            for tensor in getattr(tn, "tensors", ())
        )

    @classmethod
    def _is_native_fermionic_product_state(cls, p):
        """Return whether ``p`` is a native Symmray fermionic product MPS."""
        if not cls._has_symmray_data(p):
            return False
        is_fermionic = getattr(p, "isfermionic", None)
        if not callable(is_fermionic) or not is_fermionic():
            return False
        try:
            return cls._effective_max_bond(p) <= 1
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _mps_data_is_finite(p):
        """Return whether tensor data contains only finite values.

        All dense tensors or symmetry blocks are reduced to scalar booleans on
        their live backend, combined there, and copied to the host once. In
        particular, this neither materializes CuPy/Torch tensors on the host
        nor synchronizes once per MPS site.
        """

        def iter_arrays(data):
            """Yield dense leaves from one dense or block-sparse array."""
            blocks = getattr(data, "blocks", None)
            if blocks is not None:
                if isinstance(blocks, Mapping):
                    blocks = blocks.values()
                try:
                    for block in blocks:
                        yield from iter_arrays(block)
                    return
                except TypeError:
                    pass
            yield data

        checks = []
        for tensor in getattr(p, "tensors", ()):
            for data in iter_arrays(tensor.data):
                try:
                    checks.append(ar.do("all", ar.do("isfinite", data)))
                    continue
                except Exception:
                    pass

                try:
                    if not bool(np.all(np.isfinite(np.asarray(data)))):
                        return False
                except Exception:
                    return False

        if checks:
            try:
                combined = checks[0]
                for check in checks[1:]:
                    combined = ar.do("logical_and", combined, check)
                if not bool(ar.to_numpy(combined)):
                    return False
            except Exception:
                # Unknown backends may not implement scalar logical-and. The
                # supported NumPy/Torch/CuPy path above always has one host
                # conversion; retain a conservative compatibility fallback.
                if not all(bool(ar.to_numpy(check)) for check in checks):
                    return False
        exponent = getattr(p, "exponent", 0.0)
        try:
            return bool(np.isfinite(float(exponent)))
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _is_nearest_neighbor_1d(where):
        """Return whether an integer two-site location is adjacent in MPS order."""
        if len(where) != 2:
            return True
        site0, site1 = where
        if not isinstance(site0, Integral) or not isinstance(site1, Integral):
            return True
        return abs(int(site0) - int(site1)) == 1

    def _validate_symmray_mode_support(self):
        """Fail early for Symmray/MPS combinations with known bad paths."""
        # Block FIT grows only charge sectors generated by the effective target
        # and uses Symmray's native block SVD, so DMRG no longer needs Quimb's
        # dense-style global padding. Other supported modes already dispatch
        # through block-aware gate and split implementations.
        return

    @staticmethod
    def _symmray_structural_zero_cutoff(p, cutoff, cutoff_mode):
        """Turn an exact native split into exact-zero pruning.

        Symmray deliberately keeps every singular direction when
        ``cutoff == 0``. Routed fermionic swaps can consequently retain
        structural zero sectors whose duplicate like-dual dummy modes are not
        valid inputs to a later partial environment contraction. The smallest
        positive value representable by the block's real dtype, interpreted
        as an absolute cutoff, removes only exact zeros: Symmray keeps values
        greater than or equal to the cutoff, so every representable nonzero
        singular value is retained. No tensor is flattened or converted.
        """
        cutoff = float(cutoff)
        if cutoff != 0.0 or not p.isfermionic():
            return cutoff, cutoff_mode

        dtype_name = str(getattr(p, "dtype", "float64")).lower()
        if "bfloat16" in dtype_name:
            # NumPy has no portable bfloat16 scalar. bfloat16 has no
            # subnormal range, so its smallest positive normal is exact here.
            structural_cutoff = 1.1754943508222875e-38
        elif "16" in dtype_name:
            structural_cutoff = np.nextafter(
                np.float16(0.0), np.float16(1.0)
            ).item()
        elif "32" in dtype_name or "complex64" in dtype_name:
            structural_cutoff = np.nextafter(
                np.float32(0.0), np.float32(1.0)
            ).item()
        elif "longdouble" in dtype_name or "float128" in dtype_name:
            structural_cutoff = np.nextafter(
                np.longdouble(0.0), np.longdouble(1.0)
            ).item()
        else:
            structural_cutoff = np.nextafter(
                np.float64(0.0), np.float64(1.0)
            ).item()
        # Preserve extended-precision scalars: coercing the smallest positive
        # ``longdouble`` to Python's binary64 ``float`` can turn it back into
        # zero and silently disable structural-zero pruning.
        return structural_cutoff, "abs"

    @staticmethod
    def _native_needs_safe_qr(p):
        """Return whether native QR needs the low-precision phase guard."""
        dtype_name = str(getattr(p, "dtype", "")).lower()
        return "complex64" in dtype_name or "float32" in dtype_name

    @staticmethod
    def _native_canonize_bond(p, left, right):
        """Canonize one native bond without phase-normalizing QR diagonals."""
        qtn.tensor_canonize_bond(
            p[left],
            p[right],
            stabilized=False,
        )

    def _native_canonicalize_pair(self, p, where, *, info=None):
        """Safely move a native MPS center around a target pair."""
        if info is None:
            info = {}
        i, j = min(where), max(where)
        current = info.get("cur_orthog")
        if current == "calc":
            current = None

        if current is None:
            current = p.calc_current_orthog_center()

        if current is None:
            for site in range(0, i):
                self._native_canonize_bond(p, site, site + 1)
            for site in range(p.L - 1, j, -1):
                self._native_canonize_bond(p, site, site - 1)
            info["cur_orthog"] = (i, j)
            return

        if isinstance(current, Integral):
            cmin = cmax = int(current)
        else:
            cmin, cmax = min(current), max(current)

        if i > cmin:
            for site in range(cmin, i):
                self._native_canonize_bond(p, site, site + 1)
        else:
            i = min(j, cmin)

        if j < cmax:
            for site in range(cmax, j, -1):
                self._native_canonize_bond(p, site, site - 1)
        else:
            j = max(i, cmax)

        info["cur_orthog"] = (i, j)

    def _native_swap_site_to(self, p, site, target, *, info, compress_opts):
        """Move one native site with safe per-bond canonicalization."""
        if site == target:
            return
        if site < target:
            sites = range(site, target)
            absorb = "right"
        else:
            sites = range(site - 1, target - 1, -1)
            absorb = "left"
        swap_opts = dict(compress_opts)
        swap_opts.setdefault("absorb", absorb)
        for left in sites:
            right = left + 1
            self._native_canonicalize_pair(
                p,
                (left, right),
                info=info,
            )
            p.swap_sites_with_compress_(
                left,
                right,
                info=info,
                **swap_opts,
            )

    def _native_gate_with_auto_swap(
        self,
        p,
        gate,
        where,
        *,
        info,
        swap_back,
        **compress_opts,
    ):
        """Apply a native gate while avoiding unsafe complex64 QR phases."""
        i, j = where
        if i > j:
            i, j = j, i
            final_gate_where = (i + 1, i)
            absorb = "left"
        else:
            final_gate_where = (i, i + 1)
            absorb = "right"

        gate_opts = dict(compress_opts)
        gate_opts.setdefault("absorb", absorb)
        need_to_swap = i + 1 != j
        if need_to_swap:
            self._native_swap_site_to(
                p,
                j,
                i + 1,
                info=info,
                compress_opts=compress_opts,
            )

        self._native_canonicalize_pair(p, (i, i + 1), info=info)
        p.gate_split_(gate, where=final_gate_where, **gate_opts)
        info["cur_orthog"] = (i + 1, i + 1)

        if need_to_swap and swap_back:
            self._native_swap_site_to(
                p,
                i + 1,
                j,
                info=info,
                compress_opts=compress_opts,
            )

        return p

    def _apply_symmray_auto_swap_gate(
        self,
        p,
        gate,
        where,
        *,
        cutoff,
        cutoff_mode,
        max_bond=None,
        info=None,
        swap_back=True,
        method=None,
        seed=None,
    ):
        """Apply a Symmray two-site gate through quimb's block-aware swaps."""
        cutoff, cutoff_mode = self._symmray_structural_zero_cutoff(
            p,
            cutoff,
            cutoff_mode,
        )
        compress_opts = {
            "cutoff": cutoff,
            "cutoff_mode": cutoff_mode,
        }
        if max_bond is not None:
            compress_opts["max_bond"] = max_bond
        if method is not None:
            compress_opts["method"] = method
        if seed is not None:
            compress_opts["seed"] = seed
        if info is None:
            info = self.info_c
        if not self._native_needs_safe_qr(p):
            p.gate_with_auto_swap_(
                gate,
                where,
                info=info,
                swap_back=swap_back,
                **compress_opts,
            )
            return p
        return self._native_gate_with_auto_swap(
            p,
            gate,
            where,
            info=info,
            swap_back=swap_back,
            **compress_opts,
        )

    def _warm_start_native_fermionic_fit(
        self,
        p,
        gates,
        wheres,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Open missing native charge sectors without copying ``p_target``.

        Dense FIT uses a disposable randomized guess when dense rank growth is
        needed. Symmray's graded tensors need Quimb's native auto-swap/SVD
        route to create compatible charge blocks, so retain this native-only
        preparation. It mutates
        the current working MPS through the gate algebra; it never transfers
        tensors from the exact target network into ``fit.p``.
        """
        if not (self._has_symmray_data(p) and p.isfermionic()):
            return False
        sites = tuple(site for where in wheres for site in where)
        if max(sites) - min(sites) <= 1:
            return False
        for gate, where in zip(gates, wheres):
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
            else:
                self._apply_symmray_auto_swap_gate(
                    p,
                    gate,
                    where,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    max_bond=self.chi,
                    info=self.info_c,
                )
        return True

    def _build_symmray_auto_swap_target(
        self,
        p,
        gate,
        where,
        cutoff,
        cutoff_mode,
        *,
        copy=True,
        info=None,
    ):
        """Build an un-chi-capped target using Symmray-aware swap routing."""
        p_target = p.copy() if copy else p
        self._apply_symmray_auto_swap_gate(
            p_target,
            gate,
            where,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            info={} if info is None else info,
        )
        return p_target

    @staticmethod
    def _validate_fit_target_strategy(strategy):
        """Normalize the exact FIT target representation policy."""
        strategy = str(strategy).strip().lower()
        if strategy not in {"auto", "layered", "mps"}:
            raise ValueError(
                "fit_target_strategy must be 'auto', 'layered', or 'mps'."
            )
        return strategy

    @staticmethod
    def _validate_fit_init_strategy(strategy):
        """Normalize the FIT initial-guess construction policy."""
        strategy = str(strategy).strip().lower()
        if strategy.startswith("guess-"):
            # ``quimb-<method>`` is the mode spelling; accept the matching
            # hyphenated form for the FIT policy as a readable alias while
            # retaining the historical ``guess_<method>`` name.
            strategy = "guess_" + strategy[len("guess-") :]
        strategy = {"mpo": "svd_guess"}.get(strategy, strategy)
        if strategy not in _FIT_INIT_STRATEGIES:
            raise ValueError(
                "fit_init_strategy must be one of 'auto', 'direct', "
                "'random', 'random_expand', or 'guess-<method>'."
            )
        return strategy

    def _apply_layered_target_gate(
        self,
        target,
        gate,
        where,
        *,
        cutoff,
        cutoff_mode,
    ):
        """Append an exact spatially split gate to a disposable FIT target.

        The gate itself is SVD-factorized across its two sites, but it is not
        contracted into the MPS and no state bond is truncated. FIT can then
        contract this paper-style layered target lazily, avoiding the rapidly
        growing intermediate MPS ranks produced by repeated direct gates.
        """
        if len(where) == 1:
            return self._apply_gate(
                target,
                gate,
                where,
                contract=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=True,
            )

        if len(where) != 2:
            raise ValueError("A layered FIT target gate must act on one or two sites.")
        if self._has_symmray_data(target) or target.isfermionic():
            raise ValueError(
                "fit_target_strategy='layered' is not available for Symmray/"
                "fermionic data; use 'auto' or 'mps' for native graded routing."
            )

        sites = tuple(int(site) for site in where)
        inds = tuple(self._format_ind(site) for site in sites)
        self._timed_call(
            "gate.apply",
            qtn.tensor_network_gate_inds,
            target,
            gate,
            inds,
            contract="split-gate",
            inplace=True,
            method="svd",
            absorb="both",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        # Quimb intentionally leaves lazy gate tensors untagged. Distinct
        # endpoint tags let FIT select each half exactly once, including when
        # several sequential gates share a physical index.
        for site, index in zip(sites, inds):
            tids = tuple(target.ind_map[index])
            if len(tids) != 1:
                raise ValueError(
                    f"Layered FIT target index {index!r} is not uniquely owned."
                )
            target.tensor_map[tids[0]].add_tag(target.site_tag_id.format(site))
        return target

    @staticmethod
    def _align_layered_submpo_tags(submpo, target, where):
        """Copy and align an explicit sub-MPO with the target MPS site tags.

        FIT permits multiple target tensors per site, but every target tensor
        must carry exactly one site tag. Explicit sub-MPO events can arrive
        with a different site-tag formatter (or after a persistent layout
        remap), so align the operator copy before attaching it lazily.
        """
        submpo = submpo.copy()
        site_tag = getattr(submpo, "site_tag", None)
        if not callable(site_tag):
            raise TypeError(
                "Layered DMRG sub-MPO targets require a site-tagged operator."
            )

        active_sites = tuple(sorted({int(site) for site in where}))
        target_tags = set()
        for site in active_sites:
            old_tag = site_tag(site)
            new_tag = target.site_tag(site)
            target_tags.add(new_tag)
            if old_tag != new_tag:
                submpo.retag_({old_tag: new_tag})

        for tensor in submpo.tensors:
            tensor_site_tags = tuple(tag for tag in tensor.tags if tag in target_tags)
            if len(tensor_site_tags) != 1:
                raise ValueError(
                    "Each layered DMRG sub-MPO tensor must carry exactly one "
                    f"target site tag, got {tuple(tensor.tags)!r}."
                )
        return submpo

    def _build_submpo_fit_target(
        self,
        p,
        submpo,
        where,
        target_cutoff,
        cutoff_mode,
        *,
        target_strategy,
    ):
        """Build an exact layered or materialized target for a sub-MPO event."""
        start, stop = min(where), max(where)
        target = p.copy()
        target.canonicalize_((start, stop), info={})

        # Target representation and FIT initial guess are independent knobs:
        # the target must preserve the exact operator action, while the guess
        # only affects how quickly the variational solve finds that target.
        # Layering is safe for dense data because each operator tensor keeps
        # its site tag; native graded data must stay on the materialized route
        # so charge sectors and dummy-mode metadata are not discarded.
        layered_supported = not (
            self._has_symmray_data(target)
            or target.isfermionic()
            or submpo.isfermionic()
        )
        if target_strategy == "layered" and layered_supported:
            aligned_submpo = self._align_layered_submpo_tags(
                submpo,
                target,
                where,
            )
            target.gate_with_op_lazy_(
                aligned_submpo,
                inplace=True,
                inplace_op=False,
            )
            return target, "layered"

        if target_strategy == "layered" and not layered_supported:
            raise ValueError(
                "Layered DMRG sub-MPO targets are not available for "
                "Symmray/fermionic data; use fit_target_strategy='auto' or 'mps'."
            )

        target.gate_with_submpo_(
            submpo,
            where=where,
            method="direct",
            max_bond=None,
            cutoff=target_cutoff,
            cutoff_mode=cutoff_mode,
            info={},
            inplace_mpo=False,
        )
        return target, "mps"

    def _prepare_submpo_fit_initial_guess(
        self,
        p,
        submpo,
        where,
        *,
        block_size,
        strategy,
        fit_mpo_guess,
        rand_strength,
        seed,
        cutoff,
        cutoff_mode,
    ):
        """Build a disposable FIT guess from an explicit sub-MPO event."""
        requested_strategy = self._validate_fit_init_strategy(strategy)
        info = {
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
            "random_initialization": info,
        }
        if self._has_symmray_data(p) or p.isfermionic():
            info["reason"] = (
                "native_sector_growth"
                if int(block_size) in {2, 3}
                else "native_one_site_fit"
            )
            return result

        start, stop = self._normalize_span(where)
        needs_growth = requested_strategy in {"random", "random_expand"} and not (
            FIT._active_bonds_at_rank_targets(p, start, stop, self.chi)  # pylint: disable=protected-access
        )
        if requested_strategy == "auto":
            selected_strategy = _DEFAULT_FIT_INIT_STRATEGY
        else:
            selected_strategy = (
                requested_strategy
                if requested_strategy.startswith("guess_")
                or requested_strategy == "svd_guess"
                else requested_strategy if needs_growth else "direct"
            )
        if (
            not fit_mpo_guess
            and requested_strategy in {"auto", _DEFAULT_FIT_INIT_STRATEGY}
        ):
            selected_strategy = "direct"

        if selected_strategy == "svd_guess":
            guess_method = "direct"
        elif selected_strategy.startswith("guess_"):
            guess_method = selected_strategy[len("guess_") :]
        else:
            guess_method = None

        if guess_method is not None:
            # An explicit ``guess-*`` request is a warm-start policy, not a
            # request to grow rank. Apply it even after the active bonds reach
            # their attainable size; otherwise the one-site phase would use a
            # different initial state from the growth phase.
            fit_guess = p.copy()
            opts = self._submpo_compress_opts(
                guess_method,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            if (
                guess_method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
                and _is_interior_submpo_span(fit_guess, where)
            ):
                _apply_submpo_with_interior_workaround(
                    fit_guess,
                    submpo,
                    where,
                    chi=self.chi,
                    method=guess_method,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    info={},
                    inplace_mpo=False,
                    optimize=opts.get("optimize"),
                    seed=seed,
                )
            else:
                quimb_seed = seed if guess_method in _MPO_METHODS_USE_SEED else None
                _run_seeded_quimb(
                    quimb_seed,
                    fit_guess.gate_with_submpo_,
                    submpo,
                    where=where,
                    method=guess_method,
                    max_bond=self.chi,
                    info={},
                    inplace_mpo=False,
                    **opts,
                )
            result["fit_guess"] = fit_guess
            result["strategy"] = (
                selected_strategy
                if selected_strategy != "svd_guess"
                else "svd_guess"
            )
            result["guess_method"] = guess_method
            result["guess_used"] = True
            result["svd_guess_used"] = True
            info["reason"] = selected_strategy
            return result

        fit_guess, random_info = (
            self._build_randomized_fit_guess(
                p,
                (start, stop),
                block_size=block_size,
                rand_strength=rand_strength,
                expand=selected_strategy == "random_expand",
                seed=int(seed),
            )
            if selected_strategy in {"random", "random_expand"}
            else (p, info)
        )
        if selected_strategy == "direct":
            info["reason"] = "already_at_target"
        result["fit_guess"] = fit_guess
        result["strategy"] = selected_strategy
        result["random_initialization"] = random_info
        return result

    def _apply_gate(self, p, gate, where, **kwargs):
        """Apply a gate using this optimizer's physical-index convention."""
        kwargs.setdefault("ind_id", self.ind_id)
        if self._timing_state is None:
            return apply_gate(p, gate, where, **kwargs)
        return self._timed_call("gate.apply", apply_gate, p, gate, where, **kwargs)

    def _init_canonicalization(self):
        """Initialize canonical form and orthogonality center."""
        if self.mode in {"exact", "su"}:
            # Exact and simple-update evolution do not use canonical metadata.
            self.info_c = {}
            return
        self._validate_canonical_boundary(self.p, self.mode)
        center = self.p.L // 2
        self.info_c = {}
        self.p.canonicalize_([center], cur_orthog="calc", info=self.info_c)
        self._current_orthog(self.p)

    @staticmethod
    def _validate_canonical_boundary(p, mode):
        """Reject periodic states from open-chain canonical MPS modes.

        A periodic MPS cannot be reduced to an exact single-site mixed-
        canonical center: the omitted loop environment is not an identity.
        FIT already requires an open guess, and every optimizer mode that uses
        ``info_c`` and one-center norms must enforce the same boundary contract.
        Exact and simple-update modes do not consume canonical metadata.
        """
        if mode not in {"exact", "su"} and bool(getattr(p, "cyclic", False)):
            raise ValueError(
                "MpsOptimizer canonical modes require an open-boundary MPS; "
                "cyclic MPS data do not have an exact one-tensor canonical norm."
            )

    def _prepare_su_state(self):
        """Prepare the MPS core and bond gauges for simple-update replay."""
        if self._su_gauges_ready and self._su_gauges_state is self.p:
            return

        inner_inds = tuple(self.p.inner_inds())
        missing_gauges = any(index not in self.gauges for index in inner_inds)
        if (
            self._su_force_regauge
            or not self._su_gauges_supplied
            or missing_gauges
        ):
            self.p.gauge_all_simple_(gauges=self.gauges, progbar=False)

        self._su_gauges_ready = True
        self._su_gauges_state = self.p
        self._su_force_regauge = False

    def _refresh_su_physical_state(self):
        """Store a physical copy of the SU core with its gauges inserted."""
        physical = self.p.copy()
        physical.gauge_simple_insert(self.gauges)
        self.p_ungauged = physical
        return physical

    def _prepare_dmrg_state(self):
        """Prepare DMRG without globally padding every MPS bond.

        Two- and three-site FIT discover rank on visited bonds through native
        SVD splits. One-site compatibility runs expand only their active gate
        range immediately before fitting. Avoiding eager global padding removes
        an ``O(L * chi**2)`` memory cost on long, initially low-rank states.
        """
        self._ensure_tracked_center()

    def _prepare_mix_dmrg_state(self, where):
        """Ensure the active bonds can support a mixed-mode DMRG update.

        FIT only optimizes the interval spanned by the gate. Expanding every
        bond in a long MPS would waste ``O(L * chi**2)`` memory, so only the
        active internal indices are padded. Native Symmray callers are routed
        through MPO while an active bond is still short, avoiding Quimb's
        dense-style expansion path.
        """
        if self.chi <= 1 or getattr(self.p, "L", 0) <= 1:
            return

        xmin, xmax = min(where), max(where)
        if xmin == xmax:
            return
        target_sizes = self._mix_target_bond_dimensions()
        bonds_to_expand = [
            site
            for site in range(xmin, xmax)
            if int(self.p.bond_size(site, site + 1)) < target_sizes[site]
        ]
        if bonds_to_expand:
            if self._has_symmray_data(self.p):
                raise ValueError(
                    "One-site FIT cannot pad native Symmray bonds safely; use "
                    "fit_block_size=2 or 3 so the native block SVD grows only "
                    "charge sectors present in the effective target."
                )
            by_target = {}
            for site in bonds_to_expand:
                by_target.setdefault(target_sizes[site], []).append(site)
            for target, sites in by_target.items():
                bond_inds = [self.p.bond(site, site + 1) for site in sites]
                # MatrixProductState overrides this method without exposing
                # ``inds_to_expand``. Calling the public TensorNetwork method
                # retains the MPS object while selecting only these bonds.
                qtn.TensorNetwork.expand_bond_dimension(
                    self.p,
                    int(target),
                    inds_to_expand=bond_inds,
                    inplace=True,
                )
            self._init_canonicalization()

    def _prepare_fit_window(self, where, *, block_size):
        """Prepare a FIT window without pre-expanding adaptive updates.

        Native two- and three-site FIT updates receive the current MPS bond
        dimensions unchanged. Their direction-aware SVD splits grow only the
        visited bonds, up to ``chi``. The one-site compatibility path is the
        sole FIT path that may pre-size active bonds before fitting.
        """
        if int(block_size) == 1:
            self._prepare_mix_dmrg_state(where)

    def _dmrg_fit_block_size(self, p, where, requested_block_size):
        """Resolve the live DMRG block size for an active window.

        DMRG1 uses two-site updates only during its bounded warm-up. Once the
        optimizer has latched the full-chain one-site phase, or the active
        bonds already have no attainable rank growth left, fixed-rank one-site
        sweeps are the correct and cheaper update.
        """
        xmin, xmax = self._normalize_span(where)
        active_block_size = min(
            int(requested_block_size),
            xmax - xmin + 1,
        )
        if self._dmrg_mode_alias == "dmrg1" and active_block_size == 2:
            if self._dmrg1_native_product_two_site:
                return active_block_size
            if self._dmrg1_one_site_locked:
                return 1
            if (
                xmax - xmin >= 2
                and FIT._active_bonds_at_rank_targets(  # pylint: disable=protected-access
                    p,
                    xmin,
                    xmax,
                    self.chi,
                )
            ):
                return 1
        return active_block_size

    def _dmrg1_all_bonds_at_rank_targets(self):
        """Return whether every MPS bond has reached its physical ceiling."""
        if self._dmrg_mode_alias != "dmrg1":
            return False
        target_sizes = self._mix_target_bond_dimensions()
        if not target_sizes:
            return True
        return all(
            int(self.p.bond_size(site, site + 1)) >= int(target)
            for site, target in enumerate(target_sizes)
        )

    def _maybe_lock_dmrg1_one_site_phase(self):
        """Latch DMRG1 into one-site updates after full-chain saturation."""
        if self._dmrg_mode_alias != "dmrg1":
            return False
        if not self._dmrg1_one_site_locked and self._dmrg1_all_bonds_at_rank_targets():
            self._dmrg1_one_site_locked = True
        return self._dmrg1_one_site_locked

    def _validate_dmrg1_iteration_budget(self, p, where, *, n_iter, block_size):
        """Require two growth sweeps plus refinement for uncapped DMRG1."""
        if self._dmrg_mode_alias != "dmrg1" or int(block_size) != 2:
            return
        xmin, xmax = self._normalize_span(where)
        if xmax - xmin < 2:
            return
        if FIT._active_bonds_at_rank_targets(  # pylint: disable=protected-access
            p,
            xmin,
            xmax,
            self.chi,
        ):
            return
        if int(n_iter) < 3:
            raise ValueError(
                "mode='dmrg1' requires n_iter >= 3 for an under-capacity "
                "window: two two-site growth sweeps and at least one "
                "one-site refinement sweep."
            )

    def set_p(self, p):
        """Assign a new state and reset state-dependent optimizer metadata.

        Replacing the represented state starts a new working-norm interval. In
        particular, a retained norm from the previous state must never become
        the unitary compression target for the replacement.
        """
        # Reject an incompatible caller-owned object before ``inplace=True``
        # can install any optimizer-local methods or metadata on it.
        self._validate_canonical_boundary(p, self.mode)
        new_p = self._install_represented_norm(p if self.inplace else p.copy())
        # Validate before replacing the live state so a mixed-backend input
        # cannot leave this optimizer half-updated after a failed assignment.
        self._state_backend_info_for(new_p)
        self._validate_normalized_gate_queue(
            (self.G, self.where, self.event_types),
            state=new_p,
        )
        self.p = new_p
        if self._fit_copy_policy_cache is not None:
            self._fit_copy_policy_cache.clear()
        self._initial_p = self.p.copy()
        self._initial_mps_length = int(getattr(self.p, "L", 0))
        self._mps_length_history = [self._initial_mps_length]
        self.cap_history = []
        self._effective_active_positions = set()
        self._effective_length_history = [0]
        self._effective_site_history = [()]
        self._effective_event_history = []
        self._unitary_previous_norm = None
        self.norm_events = []
        self._norm_log_survival = 0.0
        self._dmrg1_one_site_locked = False
        self._dmrg1_native_product_two_site = (
            self._dmrg_mode_alias == "dmrg1"
            and self._is_native_fermionic_product_state(self.p)
        )
        self.qubits = list(range(int(getattr(self.p, "L", 0))))
        self.logical_order = list(self.qubits)
        self._persistent_layout_plan = None
        self.layout_plan = None
        self.last_layout_plan = None
        self._su_gauges_supplied = False
        self._su_gauges_ready = False
        self._su_gauges_state = None
        self._su_force_regauge = self.mode == "su"
        self.p_ungauged = None
        self.backend_info()
        self._init_canonicalization()

    def sync_canonicalization(self, site=None):
        """Re-establish tracked canonical metadata after external MPS access.

        Quimb's canonical readout helpers, such as
        ``local_expectation_canonical``, move the live MPS orthogonality centre
        in place.  Internal Pepsy paths pass ``info_c`` and stay synchronized,
        but direct calls through :attr:`p` cannot update this optimizer's
        metadata.  Call this method before resuming a canonical-mode replay
        after such an external mutation.  It performs an explicit centre
        discovery, canonicalizes to a single site, and records the resulting
        ``info_c['cur_orthog']``.

        Post-run diagnostic readout should normally use ``p.copy()`` instead;
        this method is the recovery path when the live state was intentionally
        inspected or modified.

        Parameters
        ----------
        site : int, optional
            Site at which to leave the one-site canonical centre.  If omitted,
            the upper endpoint of Quimb's discovered canonical span is used.

        Returns
        -------
        tuple[int, int]
            The synchronized one-site canonical span.
        """
        if self.mode in {"exact", "su"}:
            raise ValueError(
                "sync_canonicalization requires a canonical MPS mode; "
                f"mode={self.mode!r} does not track info_c."
            )
        if not hasattr(self.p, "calc_current_orthog_center"):
            raise TypeError("the live state does not expose MPS canonical metadata.")

        current = self._normalize_span(self.p.calc_current_orthog_center())
        if site is None:
            site = current[1]
        site = int(site)
        if not 0 <= site < int(self.p.L):
            raise ValueError(
                f"site must lie in [0, {int(self.p.L)}), got {site}."
            )

        self.p.canonize(
            [site],
            cur_orthog=current,
            info=self.info_c,
        )
        self.info_c["cur_orthog"] = (site, site)
        return self.info_c["cur_orthog"]

    def normalize(self, eps=1e-15, insert=None):
        """Normalize current ``self.p`` in-place.

        Parameters
        ----------
        eps : float, default=1e-15
            Precision used by Quimb's general normalization path in exact and
            simple-update modes. Canonical open-MPS modes use their tracked
            one-site center directly.
        insert : int | None, default=None
            Optional site where the normalization factor is inserted.

        Returns
        -------
        float | complex
            Previous raw ``self.p.H @ self.p`` value. Canonical open-MPS modes
            derive it from the tracked center; exact and simple-update modes
            use Quimb's general normalization implementation. The removed norm
            factor is accumulated into ``self.p.exponent`` when present, so
            ``self.p.norm()`` continues to report the represented norm while
            the raw data norm becomes one.
        """
        track_canonical_center = self.mode not in {"exact", "su"}
        if track_canonical_center:
            previous_span = self._current_orthog(self.p)
            if insert is None:
                # Preserve an authoritative singleton. For a broad center,
                # choose the right edge once and collapse directly to it.
                insert_site = int(previous_span[1])
            else:
                insert_site = int(insert) % self.p.L
            if previous_span == (insert_site, insert_site):
                scale = self.p[insert_site].norm()
            else:
                scale = self._canonical_span_norm(
                    self.p,
                    (insert_site, insert_site),
                )
            scale_abs = ar.do("abs", scale)
            scale_float = self._real_float(scale_abs)
            if scale_float == 0.0 or (
                self._finite_check_enabled and not math.isfinite(scale_float)
            ):
                raise FloatingPointError(
                    "Cannot normalize an MPS with a zero or non-finite "
                    "canonical-center norm."
                )
            old_norm = scale_abs * scale_abs
            self.p[insert_site].modify(
                data=self.p[insert_site].data / scale
            )
            self._accumulate_exponent(self.p, scale)
            self._record_orthog_span(self.p, (insert_site, insert_site))
        else:
            # Exact/SU states do not have a tracked one-site center. Preserve
            # Quimb's general and cyclic normalization implementation there.
            normalize = getattr(self.p, "normalize", None)
            if callable(normalize):
                old_norm = normalize(eps=eps, insert=insert)
            else:
                # Exact replay stores a contracted TensorNetwork, which does
                # not expose the MPS ``normalize`` helper. Scale the network
                # directly while retaining the same previous-norm contract.
                scale = self._real_float(self.p.norm())
                if scale == 0.0 or (
                    self._finite_check_enabled and not math.isfinite(scale)
                ):
                    raise FloatingPointError(
                        "Cannot normalize an exact state with a zero or "
                        "non-finite norm."
                    )
                old_norm = scale * scale
                self.p.multiply(1.0 / scale, inplace=True)
            self._accumulate_exponent(self.p, old_norm**0.5)
        # ``normalize`` preserves the represented physical state through the
        # exponent, but changes the raw center norm used by unitary compression
        # stabilization. Rebase that scalar on the next run.
        self._invalidate_unitary_norm_baseline()
        return old_norm

    def _copy_impl(self, *, capture_initial):
        """Copy optimizer state, optionally retaining a shot-replay template."""
        copied = type(self)(
            self.p.copy(),
            gates=[],
            chi=self.chi,
            mode=self.mode,
            contraction_opt=self.contraction_opt,
            ind_id=self.ind_id,
            inplace=True,
            gauges=deepcopy(self.gauges),
            _capture_initial=False,
            to_backend=self._symbolic_gate_to_backend,
        )
        copied._dmrg_mode_block_size = self._dmrg_mode_block_size
        copied._dmrg_mode_alias = self._dmrg_mode_alias
        copied._dmrg1_native_product_two_site = (
            self._dmrg1_native_product_two_site
        )
        # ``MatrixProductState.copy()`` does not promise to preserve the
        # physical orthogonality centre. The constructor canonicalizes the
        # copied state, so its freshly initialized ``info_c`` is authoritative
        # here. Overwriting it with the source cache can claim that site 0 is
        # canonical while the copied tensors are centered at site ``L // 2``;
        # a subsequent projective replay can then lose the branch norm.
        if copied.mode not in {"exact", "su"}:
            copied.info_c["cur_orthog"] = tuple(
                int(site) for site in copied.p.calc_current_orthog_center()
            )
        else:
            copied.info_c = deepcopy(self.info_c)
        copied.inplace = self.inplace
        # Coalesced trajectory branches are copied immediately before an
        # ordinary replay and never become nested shot runners. Avoid a
        # second MPS copy for that internal path; public ``copy()`` retains
        # the constructor-state template needed by ``run(shots=...)``.
        copied._initial_p = self.p.copy() if capture_initial else None
        copied._initial_mps_length = self._initial_mps_length
        copied._mps_length_history = list(self._mps_length_history)
        copied.cap_history = deepcopy(self.cap_history)
        copied._effective_active_positions = set(self._effective_active_positions)
        copied._effective_length_history = list(self._effective_length_history)
        copied._effective_site_history = deepcopy(self._effective_site_history)
        copied._effective_event_history = deepcopy(self._effective_event_history)
        copied._stream_plan = self._stream_plan
        copied._gate_stream = tuple(self._gate_stream)
        copied._has_trajectory_events = self._has_trajectory_events
        copied._shared_backend_cache = self._shared_backend_cache
        copied._backend_cache_plan = self._backend_cache_plan
        copied.G = list(self.G)
        copied.where = list(self.where)
        copied.event_types = list(self.event_types)
        copied.qubits = list(self.qubits)
        copied.logical_order = list(self.logical_order)
        copied._persistent_layout_plan = deepcopy(self._persistent_layout_plan)
        copied.layout_plan = deepcopy(self.layout_plan)
        copied.last_layout_plan = deepcopy(self.last_layout_plan)
        copied.scheduled_layout_plan = deepcopy(self.scheduled_layout_plan)
        copied.scheduled_site_order = deepcopy(self.scheduled_site_order)
        copied.normalizations = deepcopy(self.normalizations)
        copied.norm_events = deepcopy(self.norm_events)
        copied._norm_log_survival = self._norm_log_survival
        copied.quality_checks = deepcopy(self.quality_checks)
        copied.mix_history = deepcopy(self.mix_history)
        copied.last_mix_summary = deepcopy(self.last_mix_summary)
        copied.last_run_timing = deepcopy(self.last_run_timing)
        copied._fit_copy_policy_cache = None
        copied._finite_check_enabled = False
        copied._mix_dmrg_disabled_reason = self._mix_dmrg_disabled_reason
        copied._mix_dmrg_failed_sweep = self._mix_dmrg_failed_sweep
        copied._dmrg1_one_site_locked = self._dmrg1_one_site_locked
        copied._last_dmrg_fit_diagnostics = deepcopy(
            self._last_dmrg_fit_diagnostics
        )
        copied.measurements = deepcopy(self.measurements)
        copied._unitary_previous_norm = self._unitary_previous_norm
        copied._trajectory_diagnostics = deepcopy(
            getattr(self, "_trajectory_diagnostics", None)
        )
        copied._su_gauges_supplied = True
        copied._su_gauges_ready = self._su_gauges_ready
        copied._su_gauges_state = copied.p if self._su_gauges_ready else None
        copied._su_force_regauge = self._su_force_regauge
        copied.p_ungauged = (
            self.p_ungauged.copy() if self.p_ungauged is not None else None
        )
        copied._rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        return copied

    def _copy_for_trajectory_branch(self):
        """Return a branch copy without an unused nested-shot snapshot."""
        return self._copy_impl(capture_initial=False)

    def copy(self) -> "MpsOptimizer":
        """Return an independent optimizer copy at its current MPS state.

        The copied optimizer owns a deep copy of the represented MPS and an
        independent canonical-centre cache. Queue entries are intentionally
        retained (without copying immutable gate payloads), so callers can
        continue a partially prepared replay independently. The copy also
        snapshots its current state for a later ``run(shots=...)`` call.
        """
        return self._copy_impl(capture_initial=True)

    def set_mode(self, mode):
        """Switch optimization mode while preserving the represented state."""
        old_mode = self.mode
        old_dmrg_alias = self._dmrg_mode_alias
        mode_name = str(mode).strip().lower()
        new_dmrg_alias = (
            mode_name if mode_name in self._DMRG_MODE_ALIASES else None
        )
        new_dmrg_block_size = self._dmrg_alias_block_size(mode)
        new_mode = self._normalize_mode(mode)
        if new_mode == "exact" and self._persistent_layout_plan is not None:
            raise ValueError(
                "cannot switch a persistent-layout optimizer to mode='exact'; "
                "read out the logical state or create a new optimizer."
            )
        self._validate_canonical_boundary(self.p, new_mode)
        if old_mode == "su" and new_mode != "su":
            if self._su_gauges_ready:
                self.p.gauge_simple_insert(self.gauges)
                self.p_ungauged = self.p.copy()
            self._su_gauges_supplied = False
            self._su_gauges_ready = False
            self._su_gauges_state = None
            self._su_force_regauge = True
        if old_mode == "perm" and new_mode != "perm":
            # Other modes interpret integer ``where`` values as physical MPS
            # positions, so restore the logical ordering before switching.
            self._restore_permutation()
        elif old_mode != "perm" and new_mode == "perm":
            if self._persistent_layout_plan is not None:
                raise ValueError(
                    "cannot switch a persistent layout into mode='perm'; "
                    "use the persistent layout mapping for replay instead."
                )
            self.qubits = list(range(int(getattr(self.p, "L", 0))))
            self.logical_order = list(self.qubits)
        if new_mode == "exact":
            # Exact contractions do not consume canonical metadata. Discard
            # the MPS-only cache so it cannot be mistaken for the contracted
            # TensorNetwork's state.
            self.info_c = {}
        self.mode = new_mode
        self._dmrg_mode_alias = new_dmrg_alias
        self._dmrg_mode_block_size = new_dmrg_block_size
        self._dmrg1_native_product_two_site = (
            self._dmrg_mode_alias == "dmrg1"
            and self._is_native_fermionic_product_state(self.p)
        )
        if old_mode != new_mode or old_dmrg_alias != new_dmrg_alias:
            self._dmrg1_one_site_locked = False
        if self.mode == "su":
            self.info_c = {}
            self.p_ungauged = None
            if old_mode != "su":
                self._su_gauges_ready = False
                self._su_gauges_state = None
                self._su_force_regauge = True
        elif old_mode == "su":
            self._init_canonicalization()
        if old_mode == "exact" and self.mode != "exact":
            # Exact mode stores a fully contracted TensorNetwork, so rebuild an
            # MPS before recreating canonical metadata for an MPS mode.
            self._ensure_mps_state()
            self._init_canonicalization()
        return self

    def _restore_permutation(self):
        """Restore logical site order after a lazy-permutation replay."""
        if self._persistent_layout_plan is not None:
            raise ValueError(
                "persistent layouts are intentionally not restored; use "
                "to_dense(logical_order=True) or remap_sample(...) for readout."
            )
        target = tuple(range(int(getattr(self.p, "L", 0))))
        current = tuple(self.qubits)
        if current != target:
            self._reorder_mps_to_logical_order(target, current_order=current)
        self.qubits = list(target)
        self.logical_order = list(target)

    def restore_qubit_order(self):
        """Restore ``p`` to logical site order and return the managed state."""
        self._restore_permutation()
        return self.p

    def _logical_to_physical_where(self, where):
        """Map logical site locations to current physical MPS positions."""
        if self._persistent_layout_plan is None and self.mode != "perm":
            return tuple(int(site) for site in where)
        order = self.logical_order if self._persistent_layout_plan is not None else self.qubits
        try:
            return tuple(order.index(int(site)) for site in where)
        except ValueError as exc:
            raise ValueError(
                f"logical site in {where!r} is not present in the current "
                f"permutation {order!r}."
            ) from exc

    def _record_permutation_move(self, where):
        """Record the no-swap-back movement made by a two-site gate."""
        i, j = sorted(map(int, where))
        moved = self.qubits.pop(j)
        self.qubits.insert(i + 1, moved)
        self.logical_order = list(self.qubits)

    def _update_permutation_after_cap(self, logical_site, physical_site):
        """Remove a capped logical site and renumber the shortened chain."""
        logical_site = int(logical_site)
        physical_site = int(physical_site)
        if self.qubits[physical_site] != logical_site:
            raise ValueError(
                "cap permutation bookkeeping lost the logical site mapping."
            )
        remaining = [
            logical
            for physical, logical in enumerate(self.qubits)
            if physical != physical_site
        ]
        self.qubits = [
            logical if logical < logical_site else logical - 1
            for logical in remaining
        ]
        self.logical_order = list(self.qubits)

    def logical_site(self, position):
        """Return the logical site currently stored at physical ``position``."""
        position = int(position)
        if not 0 <= position < len(self.logical_order):
            raise IndexError(
                f"physical position {position} is outside the MPS range "
                f"[0, {len(self.logical_order)})."
            )
        return int(self.logical_order[position])

    def position(self, site):
        """Return the physical position currently holding logical ``site``."""
        site = int(site)
        try:
            return int(self.logical_order.index(site))
        except ValueError as exc:
            raise ValueError(
                f"logical site {site} is not present in the current order "
                f"{self.logical_order!r}."
            ) from exc

    def remap_sample(self, config):
        """Remap a physical-order sample/configuration into logical order.

        ``config`` can be a length-``L`` vector or a batch with ``L`` as its
        final dimension. The returned NumPy array has logical site ``i`` at
        index ``i``.
        """
        if isinstance(config, Mapping):
            return {
                self.logical_site(position): value
                for position, value in config.items()
            }
        config = np.asarray(ar.to_numpy(config))
        if config.ndim == 0 or config.shape[-1] != len(self.logical_order):
            raise ValueError(
                "sample configuration must have MPS length as its final "
                f"dimension, got shape {config.shape}."
            )
        logical = np.empty_like(config)
        logical[..., np.asarray(self.logical_order, dtype=int)] = config
        return logical

    def to_dense(self, logical_order=True, **kwargs):
        """Return the statevector with optional logical-site axis ordering.

        With ``logical_order=True`` (the default), axes are ordered by logical
        site labels even when the managed MPS is stored in a persistent layout.
        ``logical_order=False`` returns the underlying physical MPS ordering.
        """
        if not hasattr(self.p, "L"):
            # Exact mode stores a contracted TensorNetwork rather than an MPS,
            # so its output indices must be supplied explicitly to Quimb.
            inds = (
                [self._format_ind(site) for site in range(len(self.logical_order))]
                if logical_order
                else list(self.p.outer_inds())
            )
            return self.p.to_dense(inds, **kwargs)
        if not logical_order or self.logical_order == list(range(self.p.L)):
            return self.p.to_dense(**kwargs)
        logical_inds = [self.p.site_ind(self.position(site)) for site in range(self.p.L)]
        return self.p.to_dense(logical_inds, **kwargs)

    @property
    def gate_stream(self):
        """Return the snapshotted raw stream, including trajectory events."""
        return self._gate_stream

    @property
    def allocated_length(self) -> int:
        """Return the current allocated MPS register length."""
        return int(getattr(self.p, "L", self._mps_length_history[-1]))

    @property
    def effective_active_sites(self) -> tuple[int, ...]:
        """Return current operation-active MPS positions."""
        L = self.allocated_length
        return tuple(
            int(position)
            for position in sorted(self._effective_active_positions)
            if 0 <= position < L
        )

    def _record_effective_event(self, where, *, event_type="gate"):
        """Record active support without inspecting tensor ranks or Schmidt values."""
        if event_type == "cap":
            raise ValueError("cap support must be remapped with _apply_effective_cap")
        positions = tuple(int(site) for site in where)
        if positions:
            left, right = min(positions), max(positions)
            self._effective_active_positions.update(range(left, right + 1))
        active = self.effective_active_sites
        self._effective_length_history.append(len(active))
        self._effective_site_history.append(active)
        self._effective_event_history.append(
            {
                "event_type": str(event_type),
                "where": tuple(positions),
                "L_eff": int(len(active)),
                "active_sites": active,
            }
        )

    def _apply_effective_cap(self, position):
        """Remap operation-active positions after a structural cap."""
        position = int(position)
        self._effective_active_positions = {
            active_position - (active_position > position)
            for active_position in self._effective_active_positions
            if active_position != position
        }
        active = self.effective_active_sites
        self._effective_length_history.append(len(active))
        self._effective_site_history.append(active)
        self._effective_event_history.append(
            {
                "event_type": "cap",
                "where": (position,),
                "L_eff": int(len(active)),
                "active_sites": active,
            }
        )

    @property
    def L_eff(self) -> int:
        """Return operation-active support length, with product start equal to zero.

        This is intentionally a lightweight replay ledger. It estimates the
        MPS support made active by the gate stream and reduced by caps; it does
        not compute Schmidt values, bond entropies, or dense-state ranks.
        """
        return len(self.effective_active_sites)

    @property
    def effective_mps_length(self) -> int:
        """Descriptive alias for :attr:`L_eff`."""
        return self.L_eff

    def mps_length_diagnostics(self):
        """Return allocated-register and operation-active length diagnostics.

        ``length_history`` is the allocated MPS register after construction
        and successful caps. ``L_eff`` and ``effective_length_history`` are a
        separate operation-active support ledger: they start at zero, grow
        when gate sites/intervals are replayed, and shrink when caps remove
        those positions. No Schmidt-rank or SVD inspection is performed.
        """
        history = tuple(int(length) for length in self._mps_length_history)
        effective_history = tuple(
            int(length) for length in self._effective_length_history
        )
        return {
            "initial_length": int(self._initial_mps_length),
            "peak_length": int(max(history, default=0)),
            "minimum_length": int(min(history, default=0)),
            "L_eff": int(self.L_eff),
            "effective_length": int(self.L_eff),
            "removed_sites": int(self._initial_mps_length - self.allocated_length),
            "caps": len(self.cap_history),
            "length_history": history,
            "cap_events": deepcopy(self.cap_history),
            "allocated_length": int(self.allocated_length),
            "allocated_length_history": history,
            "initial_effective_length": 0,
            "peak_effective_length": int(max(effective_history, default=0)),
            "minimum_effective_length": int(min(effective_history, default=0)),
            "effective_length_history": effective_history,
            "L_eff_history": effective_history,
            "effective_active_sites": self.effective_active_sites,
            "effective_site_history": deepcopy(self._effective_site_history),
            "effective_event_history": deepcopy(self._effective_event_history),
            "effective_length_model": "active-operation-envelope",
        }

    @property
    def has_trajectory_events(self):
        """Whether this optimizer owns a stream requiring shot replay."""
        return bool(self._has_trajectory_events)

    @staticmethod
    def _normalize_stream_plan_queue(plan):
        """Normalize a plan once for validation and single-state replay."""
        if plan.has_trajectory_events:
            return [], [], []
        return _normalize_gate_queue(plan.entries)

    def _validate_normalized_gate_queue(self, normalized_queue, *, state=None):
        """Validate an already-normalized queue without normalizing again."""
        gates, _wheres, event_types = normalized_queue
        self._validate_gate_stream_backend(gates, event_types, state=state)

    def _install_stream_plan(self, plan, *, normalized_queue=None):
        """Install a compiled plan and its already-normalized replay queue."""
        if not isinstance(plan, _MpsStreamPlan):
            raise TypeError("plan must be an internal MPS stream plan.")
        if normalized_queue is None:
            normalized_queue = self._normalize_stream_plan_queue(plan)
        self._stream_plan = plan
        self._gate_stream = plan.entries
        self._has_trajectory_events = plan.has_trajectory_events
        if not self._shared_backend_cache:
            self._backend_cache_plan = plan
        if self._has_trajectory_events:
            # Stochastic and stateful leakage entries are consumed by the shot
            # runner, which lowers each sampled branch into an ordinary stream.
            # They cannot be normalized into the single-state queue here.
            self.G, self.where, self.event_types = [], [], []
        else:
            self.G, self.where, self.event_types = normalized_queue

    def _shot_factory(self):
        """Build fresh optimizers from this instance's initial state."""
        template = self._initial_p
        mode = self._dmrg_mode_alias or self.mode
        stream = self._gate_stream
        constructor = {
            "chi": self.chi,
            "mode": mode,
            "contraction_opt": self.contraction_opt,
            "ind_id": self.ind_id,
            "to_backend": self._symbolic_gate_to_backend,
        }

        def make_optimizer():
            options = dict(constructor)
            options["gauges"] = deepcopy(self.gauges)
            options["inplace"] = True
            options["_capture_initial"] = False
            optimizer = type(self)(template.copy(), [], **options)
            optimizer._stream_plan = self._stream_plan
            optimizer._gate_stream = stream
            optimizer._has_trajectory_events = self._has_trajectory_events
            optimizer._shared_backend_cache = True
            optimizer._backend_cache_plan = self._backend_cache_plan
            if self._has_trajectory_events:
                optimizer.G, optimizer.where, optimizer.event_types = [], [], []
            else:
                optimizer.G = list(self.G)
                optimizer.where = list(self.where)
                optimizer.event_types = list(self.event_types)
            if self._persistent_layout_plan is not None:
                # The shot template is already in the frozen physical order.
                # Install only the logical mapping on the child; calling
                # apply_layout again would reorder the state a second time.
                optimizer._persistent_layout_plan = deepcopy(
                    self._persistent_layout_plan
                )
                optimizer.layout_plan = deepcopy(self.layout_plan)
                optimizer.last_layout_plan = deepcopy(self.last_layout_plan)
                optimizer.logical_order = list(self.logical_order)
                optimizer.qubits = list(self.qubits)
            optimizer.scheduled_layout_plan = deepcopy(self.scheduled_layout_plan)
            optimizer.scheduled_site_order = deepcopy(self.scheduled_site_order)
            return optimizer

        return make_optimizer

    @staticmethod
    def _shot_runner_requested(
        shots,
        *,
        has_trajectory_events,
        error_model,
        strategy,
        run_kwargs,
        max_branches,
        importance_sampling,
        max_branch_factor,
        parallel_workers,
        parallel_backend,
        auto_max_expected_faults,
        retain,
        mpi=None,
        workers="auto",
        checkpoint_path=None,
        observable=None,
    ):
        """Return whether ``run`` needs the multi-shot trajectory machinery."""
        if mpi is not None and mpi is not False:
            return True
        if checkpoint_path is not None or observable is not None:
            return True
        if has_trajectory_events or error_model is not None:
            return True
        if isinstance(shots, bool) or not isinstance(shots, Integral):
            return True
        if int(shots) != 1:
            return True
        if workers not in {None, "auto"}:
            return True
        return any(
            (
                strategy != "auto",
                run_kwargs is not None,
                max_branches != _SHOT_DEFAULT_MAX_BRANCHES,
                importance_sampling is not None,
                max_branch_factor is not None,
                parallel_workers != 1,
                parallel_backend != "thread",
                auto_max_expected_faults
                != _SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS,
                retain != "all",
            )
        )

    def _validate_shot_compatibility(self, error_model=None):
        """Reject known-invalid mode and trajectory combinations early."""
        from ..noise import (  # pylint: disable=import-outside-toplevel
            _has_unforced_branching_control,
            _leakage_event_parts,
        )

        entries = self._gate_stream
        controls = tuple(
            entry for entry in entries if self.control_event_parts(entry) is not None
        )
        has_leakage = any(_leakage_event_parts(entry) is not None for entry in entries)
        has_submpo = any(_is_submpo_event(entry) for entry in entries)
        if has_submpo and not self._is_mpo_mode(self.mode):
            raise ValueError(
                "shot replay of sub-MPO events requires an MPO mode; "
                f"mode={self.mode!r} cannot consume sub-MPO payloads."
            )
        if self.mode == "mix" and (controls or has_leakage):
            raise ValueError(
                "mode='mix' is unitary-only and cannot replay controls or leakage."
            )
        if self.mode == "su" and (controls or has_leakage):
            raise ValueError(
                "mode='su' supports gate-only shot replay; controls and leakage "
                "require a canonical MPS mode."
            )
        if error_model is not None and _has_unforced_branching_control(entries):
            raise ValueError(
                "error_model shot replay cannot combine unforced controls; "
                "use stream-local trajectory events instead."
            )

    def _run_shots(
        self,
        shots,
        *,
        error_model=None,
        seed=None,
        run_kwargs=None,
        strategy="auto",
        max_branches=_SHOT_DEFAULT_MAX_BRANCHES,
        auto_max_expected_faults=_SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        retain="all",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=True,
        checkpoint_id=None,
    ):
        """Replay this stream as an independent or coalesced shot ensemble."""
        self._validate_shot_compatibility(error_model=error_model)
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
            raise ValueError("shots must be a nonnegative integer.")
        if self.mode == "perm" and self.logical_order != list(
            range(int(getattr(self.p, "L", 0)))
        ):
            raise ValueError(
                "shot replay does not support a permuted live MPS; "
                "create a fresh optimizer before running shots."
            )
        if error_model is not None and self._has_trajectory_events:
            raise ValueError(
                "do not combine stream-local trajectory events with error_model; "
                "use one noise representation per gate stream."
            )

        from ..noise import (  # pylint: disable=import-outside-toplevel
            NoisyResult,
            run_noisy_shots,
            run_trajectory_shots,
        )

        mpi_enabled = mpi is not None and mpi is not False
        if not mpi_enabled and any(
            value is not None
            for value in (observable, checkpoint_path)
        ):
            raise ValueError(
                "observable and checkpoint options require mpi=True or an MPI communicator."
            )
        if not mpi_enabled and (
            resume
            or checkpoint_keep != 2
            or checkpoint_sync is not True
            or collect_diagnostics is not True
            or checkpoint_id is not None
        ):
            raise ValueError(
                "MPI checkpoint options require mpi=True or an MPI communicator."
            )

        if workers not in {None, "auto"}:
            if (
                isinstance(workers, bool)
                or not isinstance(workers, Integral)
                or workers < 1
            ):
                raise ValueError("workers must be a positive integer or 'auto'.")
        if workers in {None, "auto"} and parallel_workers != 1:
            workers = parallel_workers
        if mpi_enabled:
            from ..mpi import MPIShotRunner  # pylint: disable=import-outside-toplevel

            if strategy == "auto":
                strategy = "independent"
            child_kwargs = dict(run_kwargs or {})
            if progress not in {False, "never"}:
                child_kwargs["progbar"] = False
            communicator = None if mpi is True else mpi
            mpi_gates = (
                self._stream_plan.trajectory_plan
                if self._has_trajectory_events
                else self._gate_stream
            )
            runner = MPIShotRunner(
                self._shot_factory(),
                mpi_gates,
                comm=communicator,
            )
            return runner.run(
                shots,
                seed=seed,
                error_model=error_model,
                run_kwargs=child_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=importance_sampling,
                auto_max_expected_faults=auto_max_expected_faults,
                retain=retain,
                local_workers=workers,
                local_backend="auto",
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
                progress=progress,
            )

        if workers in {None, "auto"}:
            from ..mpi import _resolve_local_workers  # pylint: disable=import-outside-toplevel

            workers = _resolve_local_workers(workers, shots=shots)
        from ..mpi import (  # pylint: disable=import-outside-toplevel
            _make_progress_bar,
            _validate_progress,
        )
        progress_strategy = strategy
        if workers > 1 and strategy == "auto":
            from ..noise import _resolve_auto_parallel_strategy

            progress_strategy = _resolve_auto_parallel_strategy(
                self._stream_plan.entries,
                shots,
                error_model=error_model,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                auto_max_expected_faults=auto_max_expected_faults,
            )

        progress_mode = _validate_progress(progress)
        progress_bar = _make_progress_bar(
            progress_mode,
            shots,
            desc="shots",
        ) if workers > 1 and progress_strategy == "independent" else None
        child_kwargs = dict(run_kwargs or {})
        if workers > 1 and progress_mode != "never":
            child_kwargs["progbar"] = False

        def update_progress(delta):
            if progress_bar is not None:
                progress_bar.update(int(delta))

        common = {
            "seed": seed,
            "run_kwargs": child_kwargs,
            "strategy": strategy,
            "max_branches": max_branches,
            "importance_sampling": importance_sampling,
            "max_branch_factor": max_branch_factor,
            "parallel_workers": workers,
            "parallel_backend": parallel_backend,
            "retain": retain,
        }
        try:
            if error_model is None:
                shot_gates = (
                    self._stream_plan.entries
                    if workers > 1
                    else self._stream_plan.trajectory_plan
                )
                raw = run_trajectory_shots(
                    self._shot_factory(),
                    shot_gates,
                    shots,
                    _progress=update_progress if progress_bar is not None else None,
                    **common,
                )
            else:
                raw = run_noisy_shots(
                    self._shot_factory(),
                    self._gate_stream,
                    error_model,
                    shots,
                    auto_max_expected_faults=auto_max_expected_faults,
                    _progress=update_progress if progress_bar is not None else None,
                    **common,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return NoisyResult(raw)

    def set_gates(self, gates):
        """Replace the current gate list.

        After calling this, ``run(...)`` applies only this new list
        (unless you call :meth:`add_gates` before running).
        """
        plan = _prepare_gate_stream(
            gates,
            to_backend=self._symbolic_gate_to_backend,
            backend_sample=self._state_backend_like(),
        )
        normalized_queue = self._normalize_stream_plan_queue(plan)
        self._validate_normalized_gate_queue(normalized_queue)
        self._shared_backend_cache = False
        self._install_stream_plan(plan, normalized_queue=normalized_queue)
        self.scheduled_layout_plan = None
        self.scheduled_site_order = None
        return self

    def set_gate_schedule(self, schedule, *, reorder_product_state=True):
        """Install a precompiled lifetime-aware gate-stream schedule.

        ``schedule`` is intentionally duck-typed so Pepsy does not depend on
        Tensy's scheduler package.  It must provide ``stream`` and may provide
        ``site_order`` / ``layout_plan`` as returned by Tensy's
        ``schedule_gate_stream``.  The stream already contains physical MPS
        positions and cap events shifted after each removal, so no layout
        replay is requested here (layout replay and cap replay are separate
        operations).

        A non-identity initial layout can be installed exactly only for a
        product input MPS.  For an entangled input, the caller must first
        supply the state in ``schedule.site_order`` or use an explicitly
        controlled lossy reorder.
        """
        if self.mode == "perm":
            raise ValueError(
                "scheduled streams use fixed physical positions after caps; "
                "mode='perm' cannot apply its own lazy permutation on top."
            )
        if self._persistent_layout_plan is not None:
            raise ValueError(
                "install a scheduled stream on a fresh optimizer; persistent "
                "layouts and dynamic cap positions are separate operations."
            )
        stream = getattr(schedule, "stream", None)
        if stream is None:
            raise TypeError("schedule must provide a compiled 'stream'.")
        site_order = tuple(
            getattr(schedule, "site_order", tuple(range(int(self.p.L))))
        )
        if len(site_order) != int(self.p.L) or set(site_order) != set(range(int(self.p.L))):
            raise ValueError(
                "schedule.site_order must be a permutation of the current MPS sites."
            )
        identity = tuple(range(int(self.p.L)))
        if site_order != identity:
            if not reorder_product_state:
                raise ValueError(
                    "schedule has a non-identity site_order; set "
                    "reorder_product_state=True or provide the state in that order."
                )
            if self._effective_max_bond(self.p) != 1:
                raise ValueError(
                    "installing a scheduled non-identity layout requires a "
                    "product input MPS; reorder it explicitly before replay."
                )
            self._relabel_product_mps(site_order, current_order=identity)
            # Shot replay must start from the reordered product state, not the
            # pre-schedule order captured by the constructor.
            self._initial_p = self.p.copy()

        self.set_gates(stream)
        self.scheduled_site_order = site_order
        layout_plan = getattr(schedule, "layout_plan", None)
        self.scheduled_layout_plan = deepcopy(layout_plan)
        return self

    def add_gates(self, gates):
        """Append gates to the existing gate list.

        This preserves previously queued gates and extends them with
        new ones.
        """
        new_plan = _prepare_gate_stream(
            gates,
            to_backend=self._symbolic_gate_to_backend,
            backend_sample=self._state_backend_like(),
        )
        plan = _prepare_gate_stream(
            self._gate_stream + new_plan.entries,
            to_backend=self._symbolic_gate_to_backend,
            backend_sample=self._state_backend_like(),
        )
        normalized_queue = self._normalize_stream_plan_queue(plan)
        self._validate_normalized_gate_queue(normalized_queue)
        self._install_stream_plan(plan, normalized_queue=normalized_queue)
        self.scheduled_layout_plan = None
        self.scheduled_site_order = None
        return self

    @staticmethod
    def _layout_request_enabled(layout):
        return layout is not None and layout is not False

    @staticmethod
    def _coalesce_layout_request(use_layout_finder, layout):
        """Resolve the explicit layout-finder keyword and compatibility alias."""
        primary = use_layout_finder
        alias = layout
        if (
            primary is not None
            and primary is not False
            and alias is not None
            and alias is not False
        ):
            raise ValueError(
                "Specify only one of use_layout_finder=... or layout=...."
            )
        if primary is not None and primary is not False:
            return primary
        return alias

    def _resolve_run_layout(self, layout, layout_order, layout_kwargs):
        """Return ``(finder, plan)`` for a run-time layout request."""
        self.last_layout_plan = None
        if not self._layout_request_enabled(layout):
            return None, None
        if self.mode == "exact":
            raise ValueError("layout-aware replay requires an MPS mode, not exact.")

        if isinstance(layout, Mapping):
            plan = dict(layout)
            finder = self.layout_finder()
        else:
            order = layout_order
            if isinstance(layout, str):
                order = layout
            finder_kwargs, kwargs = self._split_layout_finder_kwargs(layout_kwargs)
            finder = self.layout_finder(**finder_kwargs)
            plan = finder.run(order=order, **kwargs)

        self._validate_layout_plan_for_mps(plan)
        self.last_layout_plan = plan
        return finder, plan

    def _validate_layout_plan_for_mps(self, plan):
        """Validate that a layout plan can be used by this MPS."""
        L = int(getattr(self.p, "L", 0))
        original_order = tuple(range(L))
        site_order = tuple(plan.get("site_order", plan.get("qubit_inds", ())))
        if set(site_order) != set(original_order):
            raise ValueError(
                "layout-aware MpsOptimizer replay currently requires a "
                "permutation of integer MPS sites range(L)."
            )
        if len(site_order) != L:
            raise ValueError("layout site_order length must match p.L.")
        site_map = plan.get("site_map", plan.get("layout"))
        if not isinstance(site_map, Mapping):
            raise ValueError("layout plan must contain a site_map/layout mapping.")
        if set(site_map) != set(original_order):
            raise ValueError("layout site_map keys must match range(p.L).")
        if set(site_map.values()) != set(original_order):
            raise ValueError("layout site_map values must be a permutation of range(p.L).")
        expected_map = {site: position for position, site in enumerate(site_order)}
        if dict(site_map) != expected_map:
            raise ValueError(
                "layout site_map must map each logical site to its position in "
                "site_order."
            )

    def _explicit_layout_plan(self, site_order):
        """Build the standard layout-plan mapping from an explicit site order."""
        site_order = tuple(int(site) for site in site_order)
        site_map = {site: position for position, site in enumerate(site_order)}
        return {
            "kind": "mps_gate_stream_layout",
            "selected_order": "explicit",
            "qubit_inds": site_order,
            "site_order": site_order,
            "order": site_order,
            "layout": site_map,
            "site_map": site_map,
            "inverse_site_map": {
                position: site for site, position in site_map.items()
            },
        }

    def _resolve_layout_plan_argument(self, plan_or_order, layout_kwargs=None):
        """Resolve a persistent-layout argument without touching the MPS."""
        if isinstance(plan_or_order, Mapping):
            plan = dict(plan_or_order)
        elif isinstance(plan_or_order, str):
            finder_kwargs, kwargs = self._split_layout_finder_kwargs(layout_kwargs)
            plan = self.layout_finder(**finder_kwargs).run(
                order=plan_or_order,
                **kwargs,
            )
        else:
            try:
                plan = self._explicit_layout_plan(plan_or_order)
            except TypeError as exc:
                raise TypeError(
                    "plan_or_order must be a layout mapping, an order name, "
                    "or a permutation of logical sites."
                ) from exc
        self._validate_layout_plan_for_mps(plan)
        return plan

    @staticmethod
    def _product_site_vector(p, physical_site):
        """Extract one local vector from a bond-one MPS tensor."""
        tensor = p[p.site_tag(int(physical_site))]
        physical_ind = p.site_ind(int(physical_site))
        try:
            physical_axis = tensor.inds.index(physical_ind)
        except ValueError as exc:  # pragma: no cover - defensive quimb guard
            raise ValueError(
                "product-state relabeling could not locate a physical site index."
            ) from exc

        if any(
            int(size) != 1
            for axis, size in enumerate(tensor.shape)
            if axis != physical_axis
        ):
            raise ValueError(
                "product-state relabeling requires every virtual dimension to "
                "be one."
            )
        axes = [axis for axis in range(tensor.ndim) if axis != physical_axis]
        axes.append(physical_axis)
        data = ar.do("transpose", tensor.data, tuple(axes))
        return data.reshape(-1)

    def _relabel_product_mps(self, target_order, *, current_order):
        """Rebuild a bond-one MPS in a new site order without SVD swaps."""
        p = self.p
        if getattr(p, "cyclic", False):
            raise ValueError(
                "persistent layout relabeling currently requires an open-boundary MPS."
            )

        vectors = {
            logical_site: self._product_site_vector(p, physical_site)
            for physical_site, logical_site in enumerate(current_order)
        }
        arrays = [vectors[logical_site] for logical_site in target_order]
        new_p = qtn.MPS_product_state(
            arrays,
            site_ind_id=p.site_ind_id,
            site_tag_id=p.site_tag_id,
        )
        if hasattr(p, "exponent") and hasattr(new_p, "exponent"):
            new_p.exponent = p.exponent
        self.p = self._install_represented_norm(new_p)
        self.info_c = {}
        self._init_canonicalization()

    def apply_layout(
        self,
        plan_or_order="quality",
        *,
        cutoff=None,
        cutoff_mode="rsum2",
        allow_lossy_reorder=False,
        layout_kwargs=None,
        layout_report=True,
    ):
        """Install a layout permanently and return this optimizer.

        Parameters
        ----------
        plan_or_order : mapping | str | sequence, default="quality"
            A plan returned by :meth:`gate_stream_layout`, a finder order name,
            or an explicit position-to-logical-site permutation.
        cutoff : float | None, default=None
            Cutoff for the one-time reorder of an initially entangled MPS.
            ``None`` uses ``1e-12``. This value is never used for product-state
            relabeling and is never used to restore the original order.
        cutoff_mode : str, default="rsum2"
            Cutoff mode for the optional one-time entangled-state reorder.
        allow_lossy_reorder : bool, default=False
            Allow the one-time reorder when ``p.max_bond() > 1``. If false,
            entangled initial states raise before mutation.
        layout_kwargs : mapping | None, default=None
            Extra keyword arguments passed to the layout finder for string
            ``plan_or_order`` values.
        layout_report : bool, default=True
            Print the usual layout summary when a finder plan is selected.

        Notes
        -----
        The installed ``logical_order`` maps physical MPS positions to logical
        site labels. Subsequent :meth:`run` calls reuse this map and do not
        reorder the MPS back to logical order. Use :meth:`to_dense` or
        :meth:`remap_sample` for logical-order readout.
        """
        if self.mode == "exact":
            raise ValueError("persistent layouts require an MPS execution mode, not exact.")
        if self.mode == "perm":
            raise ValueError(
                "persistent layouts cannot be combined with mode='perm'; choose one."
            )
        if any(event_type == "cap" for event_type in self.event_types):
            raise ValueError(
                "persistent layouts are not supported with cap control events "
                "because cap changes the MPS length."
            )

        plan = self._resolve_layout_plan_argument(plan_or_order, layout_kwargs)
        target_order = tuple(plan["site_order"])
        current_order = tuple(self.logical_order)

        if self._persistent_layout_plan is not None:
            if target_order != current_order:
                raise ValueError(
                    "a persistent layout is already installed; use the existing "
                    "logical_order or create a new optimizer for another layout."
                )
            return self

        identity = tuple(range(int(getattr(self.p, "L", 0))))
        if current_order != identity:
            raise ValueError(
                "cannot install a persistent layout while the MPS already has "
                "a lazy permutation; restore it or create a new optimizer."
            )

        if target_order != current_order:
            if self._effective_max_bond(self.p) == 1:
                self._relabel_product_mps(target_order, current_order=current_order)
            elif not allow_lossy_reorder:
                raise ValueError(
                    "persistent layout requires an initially product MPS "
                    "(p.max_bond() == 1); got max_bond={} . Set "
                    "allow_lossy_reorder=True to pay a one-time reorder cost, "
                    "or apply the layout before entangling the state.".format(
                        self.p.max_bond()
                    )
                )
            else:
                reorder_cutoff = 1e-12 if cutoff is None else float(cutoff)
                if reorder_cutoff < 0.0:
                    raise ValueError("cutoff must be non-negative.")
                self._reorder_mps_to_logical_order(
                    target_order,
                    current_order=current_order,
                    cutoff=reorder_cutoff,
                    cutoff_mode=cutoff_mode,
                )

        self.logical_order = list(target_order)
        self.qubits = list(target_order)
        self._persistent_layout_plan = plan
        self.layout_plan = plan
        self.last_layout_plan = plan
        # Shot replay starts from the configured template. Once a persistent
        # layout is installed, that template must include the one-time reorder
        # so every fresh child can reuse the frozen physical arrangement.
        self._initial_p = self.p.copy()
        if target_order != current_order:
            # Exact product relabeling preserves the norm, while an explicitly
            # lossy entangled reorder can change it. Re-establish the raw
            # unitary baseline in either case instead of trusting metadata from
            # the pre-layout tensor representation.
            self._invalidate_unitary_norm_baseline()
        if layout_report:
            report = self._layout_report_text(plan)
            if report:
                print(report)
        return self

    def _reorder_mps_to_logical_order(
        self,
        target_order,
        *,
        current_order=None,
        cutoff=0.0,
        cutoff_mode="abs",
    ):
        """Physically permute MPS site contents into ``target_order``."""
        target = list(target_order)
        current = (
            list(range(int(getattr(self.p, "L", 0))))
            if current_order is None
            else list(current_order)
        )
        if set(target) != set(current) or len(target) != len(current):
            raise ValueError("target_order must be a permutation of current_order.")

        for target_pos, logical_site in enumerate(target):
            current_pos = current.index(logical_site)
            if current_pos == target_pos:
                continue
            if self._has_symmray_data(self.p) and self._native_needs_safe_qr(self.p):
                self._native_swap_site_to(
                    self.p,
                    current_pos,
                    target_pos,
                    info=self.info_c,
                    compress_opts={
                        "method": "svd",
                        "cutoff": cutoff,
                        "cutoff_mode": cutoff_mode,
                    },
                )
            else:
                self.p.swap_site_to_(
                    current_pos,
                    target_pos,
                    info=self.info_c,
                    method="svd",
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                )
            moved = current.pop(current_pos)
            current.insert(target_pos, moved)

        self._current_orthog(self.p)
        return tuple(current)

    def _normalize_visible_mps_order(self):
        """Make cached visible MPS order match canonical site order."""
        L = int(getattr(self.p, "L", 0))
        site_inds = [self.p.site_ind(site) for site in range(L)]
        outer_inds = getattr(self.p, "_outer_inds", None)
        if outer_inds is not None:
            outer_set = set(outer_inds)
            ordered_outer = [ind for ind in site_inds if ind in outer_set]
            ordered_outer.extend(ind for ind in outer_inds if ind not in site_inds)
            self.p._outer_inds = type(outer_inds)(ordered_outer)

        tid_to_site = self.p._get_tid_to_site_map()
        if tid_to_site:
            ordered_tensors = {}
            for site in range(L):
                for tid, mapped_site in tid_to_site.items():
                    if mapped_site == site:
                        ordered_tensors[tid] = self.p.tensor_map[tid]
            for tid, tensor in self.p.tensor_map.items():
                ordered_tensors.setdefault(tid, tensor)
            self.p.tensor_map.clear()
            self.p.tensor_map.update(ordered_tensors)

    @staticmethod
    def _copy_submpo_for_layout(submpo, site_map, support):
        """Return a copied sub-MPO with site labels remapped by ``site_map``."""
        support = _unique_ordered(support)
        if not support:
            return submpo

        mpo = submpo.copy()
        token = f"_pepsy_layout_{id(mpo)}"
        reindex_to_temp = {}
        reindex_to_final = {}
        retag_to_temp = {}
        retag_to_final = {}

        for count, old_site in enumerate(support):
            new_site = site_map[old_site]
            if old_site == new_site:
                continue

            for kind in ("upper_ind", "lower_ind"):
                ind_fn = getattr(mpo, kind, None)
                if ind_fn is None:
                    continue
                old_ind = ind_fn(old_site)
                new_ind = ind_fn(new_site)
                tmp_ind = f"{token}_{count}_{kind}"
                reindex_to_temp[old_ind] = tmp_ind
                reindex_to_final[tmp_ind] = new_ind

            site_tag = getattr(mpo, "site_tag", None)
            if site_tag is not None:
                old_tag = site_tag(old_site)
                new_tag = site_tag(new_site)
                tmp_tag = f"{token}_{count}_tag"
                retag_to_temp[old_tag] = tmp_tag
                retag_to_final[tmp_tag] = new_tag

        if reindex_to_temp:
            mpo.reindex_(reindex_to_temp)
            mpo.reindex_(reindex_to_final)
        if retag_to_temp:
            mpo.retag_(retag_to_temp)
            mpo.retag_(retag_to_final)
        return mpo

    def _layout_run_sequences(self, G_seq, where_seq, event_seq, plan):
        """Return run-local payloads and mapped locations for ``plan``."""
        site_map = plan.get("site_map", plan.get("layout"))
        mapped_G = []
        mapped_where = []
        for payload, where, event_type in zip(G_seq, where_seq, event_seq):
            support = _normalize_layout_support(where)
            mapped = tuple(site_map[site] for site in support)
            if event_type == "submpo":
                payload = self._copy_submpo_for_layout(payload, site_map, support)
            mapped_G.append(payload)
            mapped_where.append(mapped)
        return mapped_G, mapped_where

    @staticmethod
    def _format_layout_value(value):
        """Format one layout diagnostic value compactly."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"

    @classmethod
    def _format_layout_reduction(cls, before, after):
        """Format ``before -> after`` with a percent decrease when meaningful."""
        before = float(before or 0.0)
        after = float(after or 0.0)
        text = f"{cls._format_layout_value(before)} -> {cls._format_layout_value(after)}"
        if before > 0.0:
            reduction = 100.0 * (before - after) / before
            text += f" ({reduction:.1f}% lower)"
        return text

    @classmethod
    def _layout_report_text(cls, plan):
        """Return a concise human-readable layout improvement report."""
        stats = plan.get("stats", {})
        input_stats = plan.get("input_stats", {})
        if not input_stats:
            return None
        selected = plan.get("selected_order", "<unknown>")
        site_order = plan.get("site_order", plan.get("qubit_inds", ()))
        weight_mode = plan.get("weight_mode", "count")
        objective = plan.get("objective", "locality")
        lines = [
            (
                "MpsOptimizer layout finder: "
                f"order={selected}, sites={len(site_order)}, "
                f"events={stats.get('num_events', input_stats.get('num_events', 0))}, "
                f"weight_mode={weight_mode}, objective={objective}"
            ),
            (
                "  long-range events: "
                + cls._format_layout_reduction(
                    input_stats.get("long_range_events", 0),
                    stats.get("long_range_events", 0),
                )
                + " | weighted: "
                + cls._format_layout_reduction(
                    input_stats.get("weighted_long_range_events", 0.0),
                    stats.get("weighted_long_range_events", 0.0),
                )
            ),
            (
                "  event span max/mean: "
                + cls._format_layout_value(input_stats.get("max_event_span", 0))
                + "/"
                + cls._format_layout_value(input_stats.get("weighted_mean_event_span", 0.0))
                + " -> "
                + cls._format_layout_value(stats.get("max_event_span", 0))
                + "/"
                + cls._format_layout_value(stats.get("weighted_mean_event_span", 0.0))
            ),
            (
                "  score: "
                + cls._format_layout_reduction(
                    input_stats.get("loss", input_stats.get("score", 0.0)),
                    stats.get("loss", stats.get("score", 0.0)),
                )
                + " | graph span: "
                + cls._format_layout_reduction(
                    input_stats.get("weighted_total_span", input_stats.get("total_span", 0.0)),
                    stats.get("weighted_total_span", stats.get("total_span", 0.0)),
                )
                + " | cut L2: "
                + cls._format_layout_reduction(
                    input_stats.get("weighted_cut_congestion_l2", 0.0),
                    stats.get("weighted_cut_congestion_l2", 0.0),
                )
            ),
        ]
        if objective == "compression":
            lines.append(
                "  operator cut load max/total: "
                + cls._format_layout_value(
                    stats.get("max_operator_cut_load", 0.0)
                )
                + "/"
                + cls._format_layout_value(
                    stats.get("total_operator_cut_load", 0.0)
                )
                + " | bounded cut probes: "
                + cls._format_layout_value(stats.get("rank_bounded_cuts", 0))
            )
        return "\n".join(lines)

    def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        n_iter=8,
        progbar=False,
        cutoff="auto",
        cutoff_mode="auto",
        mode=None,
        k_2q_batch=1,
        non_unitary=False,
        _trajectory_non_unitary=False,
        normalize_every=False,
        normalize_final=False,
        normalize_eps=1e-15,
        submpo_method=None,
        compression_seed=None,
        use_layout_finder=False,
        layout_order="quality",
        layout_kwargs=None,
        layout=None,
        layout_report=True,
        measure_renormalize=True,
        seed=None,
        mix_strict=False,
        mix_fit_min_iter=_DEPRECATED_OPTION,
        mix_fit_rtol=_DEPRECATED_OPTION,
        mix_fit_patience=_DEPRECATED_OPTION,
        mix_sticky_nonfinite=False,
        *,
        fit_min_iter=2,
        fit_rtol="auto",
        fit_patience=2,
        fit_block_size=None,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        fit_layer_size=None,
        fit_max_span="auto",
        fit_three_site_sweeps=_DEPRECATED_OPTION,
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        fit_stabilize_unitary=_DEPRECATED_OPTION,
        timing=False,
        timing_sync_device=False,
        quality_check_every=False,
        quality_check_repair=True,
        shots=1,
        error_model=None,
        strategy="auto",
        run_kwargs=None,
        max_branches=_SHOT_DEFAULT_MAX_BRANCHES,
        auto_max_expected_faults=_SHOT_DEFAULT_AUTO_MAX_EXPECTED_FAULTS,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        mpi=None,
        workers="auto",
        progress="auto",
        observable=None,
        chunk_size=None,
        checkpoint_path=None,
        resume=False,
        checkpoint_keep=2,
        checkpoint_sync=True,
        collect_diagnostics=True,
        checkpoint_id=None,
        retain="all",
    ):
        """Run the currently queued gates.

        Parameters
        ----------
        n_iter : int, default=8
            Inner iterations for DMRG local fits. In ``dmrg`` and ``mix``
            modes this is the maximum number of sweeps when adaptive FIT
            stopping is enabled; pass ``fit_rtol=None`` for fixed
            iterations. Adaptive rank-growing windows require at least two
            sweeps. An under-capacity non-adjacent ``dmrg1`` window requires
            ``n_iter >= 3`` so its two fixed growth sweeps leave room for
            one-site refinement. Once all attainable full-chain bond ceilings
            are reached, ``dmrg1`` stays in the one-site phase. The adjacent
            two-site exact fast path is exempt.
            Ignored by ``mpo``/``swap``/``svd``/``exact``.
        progbar : bool, default=False
            Show per-mode progress bars.
        cutoff : float | {"auto"}, default="auto"
            Truncation cutoff used in gate application and local fitting.
            The default ``"auto"`` selects a conservative dtype-aware value;
            pass an explicit number to preserve a fixed cutoff.
        cutoff_mode : str | None | {"auto"}, default="auto"
            Truncation mode forwarded to ``tensor_network_gate_inds`` and
            ``tensor_network_1d_compress``. ``"auto"`` (and compatibility
            value ``None``) uses ``"rsum2"`` for Pepsy's ordinary compression
            paths while preserving Quimb's method-specific native default for
            the MPO path, notably ``"rsum1"`` for ``method="dm"``. Pass a
            string to override it.
        mode : {"fit", "dmrg", "dmrg1", "dmrg2", "dmrg3", "<quimb-method>", "quimb-<method>", "quimb", "mpo-<method>", "mpo", "mix", "swap", "perm", "svd", "su", "exact"} | None, default=None
            Optional mode override for this run. If supplied, updates
            ``self.mode`` before execution.
        k_2q_batch : int, default=1
            DMRG and mixed modes: number of contiguous two-qubit gates to batch
            into one local FIT update. In mixed mode, a failed batch is replayed
            through MPO as one transaction. Standalone one-site gates use the
            exact direct/MPO path; an ordinary DMRG target block can also absorb
            intervening one-site gates before its shared FIT compression.
        non_unitary : bool, default=False
            Convenience flag for non-unitary gate streams. Normalization is
            only available when this is ``True``. This physical scale control
            is separate from unitary FIT working-norm stabilization. If
            enabled, local tensor scale control moves the
            orthogonality center to one site and normalizes it after every
            replay step. In DMRG, a step containing a multi-gate batch is
            normalized once after that batch. The removed scale is accumulated
            in ``p.exponent``.
        normalize_every : int | bool | None, default=False
            Enable one-site normalization after every replay step for a
            non-unitary stream. Use ``True`` (or any positive integer); use
            ``False`` or ``None`` to leave tensor scales untouched. Integer
            values are accepted as a boolean-style convenience and do not
            select an interval.
        normalize_final : bool, default=False
            Normalize a trailing state if the final replay step was not
            already normalized. Requires ``non_unitary=True``.
        normalize_eps : float, default=1e-15
            Numerical threshold used by the final normalization path.
        submpo_method : str | None, default=None
            Optional compression-method override for the MPO family. If
            omitted, a bare Quimb method such as ``mode="src"`` or the
            qualified ``mode="quimb-<method>"`` selects the method;
            ``mode="quimb"`` selects ``"direct"``. The opt-in
            ``"sdc"`` and ``"sdc-oversample"`` methods require a Quimb build
            that provides those compressors; they never replace an existing
            default. The legacy
            ``mode="mpo-<method>"`` / ``mode="mpo"`` spellings remain valid.
            The method
            is forwarded to Quimb for both dense gates and explicit sub-MPO
            stream events.
        compression_seed : int | None, default=None
            Explicit seed forwarded to randomized Quimb compression methods
            such as ``src``, ``srcmps``, and randomized FIT variants. ``None``
            preserves Quimb's backend-global random state.
        use_layout_finder : bool | str | Mapping, default=False
            Deprecated compatibility path. If enabled, call
            :meth:`layout_finder`, temporarily replay the stream in the
            selected 1D site order, then restore the MPS to original site
            order. Use :meth:`apply_layout` for repeated evolution. ``True``
            uses ``layout_order``; a string is used as the order name; a
            mapping is treated as a precomputed layout plan.
        layout_order : str, default="quality"
            Order passed to :meth:`layout_finder().run` when
            ``use_layout_finder=True``.
        layout_kwargs : Mapping | None, default=None
            Extra keyword arguments forwarded to ``layout_finder().run``.
        layout : bool | str | Mapping | None, default=None
            Compatibility alias for ``use_layout_finder``.
        layout_report : bool, default=True
            Print a concise before/after layout summary when layout-aware
            replay is used.
        measure_renormalize : bool, default=True
            Whether ``("measure", ...)`` and ``("reset", ...)`` control events
            renormalize the MPS to unit norm after the projective collapse. The
            outcome's Born probability is still recorded in
            :attr:`measurements`. The layout finder works with measure/reset
            control events (recorded sites always use the logical labels) but
            not with ``cap`` events, which change the MPS length.
        seed : int | None, default=None
            If given, reseed the internal RNG used to sample ``measure``/
            ``reset`` outcomes before running, for reproducible collapses.
        mix_strict : bool, default=False
            In ``mode="mix"``, restore the committed state and re-raise an
            ordinary DMRG trial exception instead of falling back to MPO.
        mix_fit_min_iter, mix_fit_rtol, mix_fit_patience : optional
            Deprecated compatibility aliases for ``fit_min_iter``,
            ``fit_rtol``, and ``fit_patience``. New code should use the
            mode-neutral keyword-only names below.
        mix_sticky_nonfinite : bool, default=False
            After a mixed DMRG trial produces NaN or Inf, use MPO for the
            remainder of this :meth:`run` call instead of retrying DMRG on
            every subsequent gate.
        fit_min_iter : int, default=2
            Minimum FIT sweeps before adaptive convergence can stop in
            ``dmrg`` or ``mix`` mode. Values above ``n_iter`` are clamped to
            ``n_iter``.
        fit_rtol : {"auto"} | float | None, default="auto"
            Relative tolerance for DMRG FIT early stopping. ``"auto"``
            selects a dtype-aware tolerance: ``1e-3`` for 16-bit data,
            ``1e-5`` for 32-bit/``complex64`` data, and ``1e-9`` for higher
            precision data. Early stopping compares changes in the retained
            canonical-center norm ``A``. ``None`` disables early stopping and
            restores fixed ``n_iter`` behavior.
        fit_patience : int, default=2
            Number of same-phase sweep-norm samples in the convergence window.
            The default of two stops after one stable comparison between two
            one-site sweeps. Rank-adaptive DMRG still performs its minimum
            adaptive warm-up before this criterion can stop a run.
        fit_block_size : {1, 2, 3} | None, default=None
            Number of neighboring MPS tensors optimized by each FIT update.
            ``None`` selects two-site FIT for ordinary DMRG and one-site FIT
            for ``mode="mix"``. In mixed mode, the one-site default first
            uses direct/MPO updates to warm under-capacity active bonds, then
            hands later eligible gates to transactional DMRG1.
            Two-site FIT is recommended: it forms both physical legs and the
            two outer virtual legs, then uses a native SVD on the middle bond,
            allowing active bonds to grow up to ``chi``. One-site FIT is kept
            for compatibility with the original fixed-rank update. Three-site
            FIT forms a three-site wavefunction and performs two native,
            direction-aware SVD splits. If the active gate span contains only
            two sites, it automatically falls back to the two-site update.
            Two- and three-site FIT never pre-expand the MPS; only bonds
            visited by their native splits can grow.
        fit_adaptive_sweeps : int, default=2
            Generic ``mode="dmrg"``: minimum number of initial two- or
            three-site sweeps used to adapt the active bond spaces. Generic
            rank-adaptive DMRG continues block sweeps until every active bond
            reaches its physical ceiling; rank stagnation never triggers the
            transition. Long-range windows use the corresponding fixed block
            handoff so their terminal canonical center remains authoritative
            for unitary norm tracking. Otherwise, if a ceiling is not reached,
            the block phase uses all requested sweeps, and remaining sweeps use
            fixed-rank one-site FIT. For named ``mode="dmrg1"``, the two-site phase is fixed at
            two sweeps and this value does not extend it; after that, remaining
            sweeps use one-site FIT and the phase latches once all full-chain
            attainable ceilings are reached. For ``mode="dmrg2"`` and
            ``mode="dmrg3"``, this sets the required two- or three-site
            warm-up length. The value is clipped to ``n_iter`` and ignored
            for ``fit_block_size=1``; the default is two sweeps. Mixed mode's
            one-site path does not use this value as a two-site warm-up.
        fit_sweep_sequence : str, default="RL"
            Cyclic FIT sweep directions. ``"R"`` is left-to-right, ``"L"``
            is right-to-left, and ``"RL"`` alternates. Alternating sweeps avoid
            favoring one canonical direction.
        fit_layer_size : int | None, default=None
            Clear alias for ``k_2q_batch``: the number of sequential two-site
            circuit gates absorbed into one paper-style target block. This is
            independent of ``fit_block_size``, which controls the local
            variational wavefunction tensor.
        fit_max_span : int | {"auto"} | None, default="auto"
            Maximum inclusive spatial span of a batched FIT target. ``"auto"``
            keeps ordinary local layers together while splitting disjoint
            gates before they create an unnecessarily wide active window.
            ``None`` restores unrestricted gate-count batching.
        fit_three_site_sweeps : int, deprecated
            Compatibility alias for ``fit_adaptive_sweeps``. New code should
            use the common adaptive-sweep control.
        target_cutoff : float, default=0.0
            Cutoff used only while constructing the pre-FIT gate target.
            Keeping this at zero separates exact target construction from the
            output truncation controlled by ``cutoff``.
        fit_target_strategy : {"auto", "layered", "mps"}, default="auto"
            Exact target representation. ``"layered"`` keeps ordinary dense
            gates as lazily contracted operator-Schmidt tensors, avoiding
            intermediate target-MPS rank growth. ``"mps"`` materializes the
            traditional routed target. ``"auto"`` selects layered targets
            for NumPy/Torch/CuPy and the native MPS route for Symmray.
            Explicit multi-site sub-MPO events in DMRG use the same layered
            representation when the backend supports lazy FIT targets.
        fit_mpo_guess : bool, default=True
            Legacy compatibility switch for the named DMRG1/DMRG3 default
            ``"guess-src"`` policy. New code should use
            ``fit_init_strategy`` explicitly. Dense DMRG uses the disposable
            SRC guess in both the expansion and one-site/reached-chi phases.
            Native Symmray and fermionic routes use a disposable
            sector-preserving randomized guess for the ``guess-src`` policy;
            this does not replace the exact FIT target or live MPS.
        fit_init_strategy : {"auto", "direct", "random", "random_expand", "guess-<method>"}, default="guess-src"
            Select the disposable FIT initial guess. ``"direct"`` uses the
            current MPS, ``"random"`` perturbs existing tensors without
            changing bond dimensions, ``"random_expand"`` adds seeded
            directions on under-capacity active bonds, and
            ``"guess-<method>"`` uses the corresponding Quimb compression
            method on an isolated copy. For native Symmray/fermionic states,
            ``"guess-src"`` instead uses Symmray's sector-preserving
            randomized SVD on an isolated copy; other Quimb guess methods
            retain the native direct fallback. ``"auto"`` and the default
            select ``"guess-src"`` in both expansion and reached-chi phases.
            On native Symmray/fermionic states this is the sector-preserving
            randomized guess. The underscore spelling ``"guess_<method>"``
            remains accepted as a compatibility alias.
        fit_init_rand_strength : float, default=0.0
            For dense two- and three-site FIT growth windows that are below
            their attainable physical/``chi`` bond ceilings, seed a
            disposable copy of the current MPS with random entries on only
            those active bonds. The exact FIT target is still built from the
            unmodified current MPS. The default ``fit_init_strategy`` is
            ``"guess-src"``, so this strength is unused unless a random
            strategy is selected explicitly. Set it to a positive value to
            enable random initialization. Native Symmray and fermionic
            routes ignore it.
        fit_init_seed : int, default=0
            Deterministic seed for ``"random"`` and ``"random_expand"`` FIT
            guesses and randomized Quimb methods selected through
            ``"guess-<method>"``. Native ``guess-src`` uses the same seed for
            Symmray randomized SVD. The underscore spelling is also accepted.
            The gate position is mixed into the
            per-window stream so repeated runs are reproducible without
            sharing a global RNG.
        fit_single_pair_fast_path : bool, default=False
            Stop an adjacent two-site FIT after its single exact variational
            update. This structural convergence is independent of ``rtol``;
            enable it when deliberately choosing the one-update fast path.
        fit_overlap_diagnostics : bool, default=False
            Contract the final fitted MPS against the disposable exact FIT
            target and report target-overlap fidelity. This adds an extra
            tensor-network contraction after each successful DMRG FIT update;
            when disabled, the overlap fields remain ``None`` while the
            ordinary FIT convergence metadata is still collected.
        stabilize_unitary : bool, default=False
            By default, retain the raw norm change after each unitary FIT or
            mixed/MPO/swap/permutation/SVD compression so norm loss remains
            observable. Set this to ``True`` to restore the working norm for
            numerical scale control; the discarded scale is not stored in
            ``p.exponent``. Pass ``non_unitary=True`` for norm-changing
            streams.
        fit_stabilize_unitary : optional
            Deprecated compatibility alias for ``stabilize_unitary``.
        finite_check : bool, default=False
            Optional diagnostic tensor and scalar-norm validation, not
            required for normal optimization. Leave False to avoid the extra
            checks and possible accelerator synchronization. False disables
            FIT finite scans, non-finite convergence-norm checks, mixed-mode
            commit validation, and unitary norm-consistency validation.
            True also checks the final tensor data in every replay mode and
            emits one performance warning per replay, shared by all nested
            FIT calls. Shot workers inherit this flag
            unless ``run_kwargs`` overrides it. Norm calculations needed for
            convergence/accounting and explicit quality/overlap diagnostics
            remain independent, as do input validation and zero-divisor guards.
        timing : bool, default=False
            Record wall-clock replay timing in :attr:`last_run_timing` without
            printing. Timing is fully opt-in: disabled runs retain the normal
            path without profiling clocks or records. Enabled records include
            inclusive stage totals
            for gate preparation, canonicalization, gate application, FIT,
            normalization, control-event measurement, and the active mode
            replay. Mixed-mode records also include the
            final :attr:`last_mix_summary`. Profiling does not enable FIT SVD
            split diagnostics.
        timing_sync_device : bool, default=False
            When timing is enabled, synchronize supported CUDA/CuPy/JAX work
            at timing boundaries so reported values include device kernels.
            The accelerator route is resolved once; CPU data needs no barrier.
            Leave disabled for the lowest-overhead timing run.
        quality_check_every : int | bool | None, default=False
            If set, periodically check finite tensor data and canonical-gauge
            coverage after replay steps. ``True`` checks every step.
        quality_check_repair : bool, default=True
            Re-canonicalize the live MPS when a periodic gauge check detects
            missing canonical coverage.
        shots : int, default=1
            Number of trajectories to replay. The default preserves the
            single-state return value for ordinary streams. A stream-local
            noisy stream, an explicit ``shots != 1``, or shot-runner options
            dispatches to the trajectory result facade.
        error_model : PauliErrorModel | None, default=None
            Optional legacy Pauli error model for a clean gate stream. This
            selects the Pauli shot runner and cannot be combined with
            stream-local trajectory or leakage entries.
        strategy : {"auto", "independent", "coalesced"}, default="auto"
            Shot representation strategy. ``"auto"`` shares deterministic
            prefixes when the branch count remains bounded and otherwise
            restarts with independent trajectories.
        run_kwargs : mapping | None, default=None
            Keyword arguments forwarded to each fresh optimizer's ordinary
            single-trajectory ``run`` call.
        max_branches : int | None, default=128
            Safety cap for coalesced trajectory replay.
        auto_max_expected_faults : float, default=0.1
            Expected-fault threshold used by automatic legacy Pauli replay.
        importance_sampling : ImportanceSamplingPolicy | None, default=None
            Optional proposal policy for trajectory events.
        max_branch_factor : int | None, default=None
            Optional per-event branch-growth cap for coalesced replay.
        parallel_workers : int, default=1
            Number of workers for explicit parallel shot execution.
        parallel_backend : {"thread", "gpu", "serial"}, default="thread"
            Backend used for explicit parallel shot execution.
        mpi : bool | MPI communicator | None, default=None
            Run the shot ensemble collectively over MPI. ``True`` uses
            ``MPI.COMM_WORLD``; an explicit communicator can be supplied.
        workers : int | "auto" | None, default="auto"
            Local shot workers. ``"auto"`` uses the process CPU allowance and
            divides it across MPI ranks sharing a host. Use ``1`` to force
            serial local execution.
        progress : {"auto", True, False}, default="auto"
            Show one aggregate rank-zero shot progress bar for MPI runs.
        observable : callable, optional
            Observable evaluated during checkpointed MPI shot reduction.
        chunk_size : int, optional
            Number of shots processed in one checkpoint/reduction chunk.
        checkpoint_path : path-like, optional
            Destination for resumable MPI checkpoints.
        resume : bool, default=False
            Resume from ``checkpoint_path`` when a compatible checkpoint exists.
        checkpoint_keep : int, default=2
            Number of completed checkpoints retained on disk.
        checkpoint_sync : bool, default=True
            Synchronize checkpoint writes across MPI ranks.
        collect_diagnostics : bool, default=True
            Collect bounded-memory diagnostic summaries during reduction.
        checkpoint_id : str, optional
            Stable identifier used to distinguish checkpoint streams.
            These options require ``mpi=True`` or an explicit MPI communicator.
        retain : {"all", "final", "none"}, default="all"
            Result retention policy for shot replay. ``"all"`` retains final
            states and replay metadata, ``"final"`` retains final states only,
            and ``"none"`` retains no optimizer states.

        Returns
        -------
        qtn.TensorNetwork | NoisyResult | MPIShotResult
            The updated ``self.p`` state for a single ordinary replay, or a
            stable noisy/MPI result facade when shot replay is selected.
        """
        if not isinstance(finite_check, (bool, np.bool_)):
            raise TypeError("finite_check must be a boolean.")
        finite_check = bool(finite_check)
        if self._shot_runner_requested(
            shots,
            has_trajectory_events=self._has_trajectory_events,
            error_model=error_model,
            strategy=strategy,
            run_kwargs=run_kwargs,
            max_branches=max_branches,
            importance_sampling=importance_sampling,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
            auto_max_expected_faults=auto_max_expected_faults,
            retain=retain,
            mpi=mpi,
            workers=workers,
            checkpoint_path=checkpoint_path,
            observable=observable,
        ):
            if mode is not None:
                self.set_mode(mode)
            run_kwargs = dict(run_kwargs or {})
            run_kwargs.setdefault("finite_check", finite_check)
            return self._run_shots(
                shots,
                error_model=error_model,
                seed=seed,
                run_kwargs=run_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                auto_max_expected_faults=auto_max_expected_faults,
                importance_sampling=importance_sampling,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
                retain=retain,
                mpi=mpi,
                workers=workers,
                progress=progress,
                observable=observable,
                chunk_size=chunk_size,
                checkpoint_path=checkpoint_path,
                resume=resume,
                checkpoint_keep=checkpoint_keep,
                checkpoint_sync=checkpoint_sync,
                collect_diagnostics=collect_diagnostics,
                checkpoint_id=checkpoint_id,
            )

        timing = bool(timing)
        timing_sync_device = bool(timing_sync_device)
        if mode is not None:
            self.set_mode(mode)
        cutoff = self._resolve_cutoff(cutoff)
        quality_check_every = self._resolve_quality_check_every(
            quality_check_every
        )
        quality_check_repair = bool(quality_check_repair)
        self.quality_checks = []
        # Mixed mode is intentionally a direct/MPO warm-up followed by
        # one-site DMRG. Keep the ordinary DMRG default at two-site FIT, while
        # allowing callers to opt into mixed two- or three-site transactions
        # explicitly with ``fit_block_size=2`` or ``3``.
        if fit_block_size is None:
            fit_block_size = 1 if self.mode == "mix" else 2

        # ``auto`` (and the legacy ``None``) uses Pepsy's relative squared
        # weight policy for ordinary paths. MPO paths retain Quimb's native
        # method default when auto is selected, notably rsum1 for ``dm``.
        mpo_cutoff_mode = self._resolve_cutoff_mode(
            cutoff_mode,
            preserve_mpo_default=self._is_mpo_mode(self.mode),
        )
        cutoff_mode = self._resolve_cutoff_mode(cutoff_mode)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.last_layout_plan = self._persistent_layout_plan
        G_seq = list(self.G)
        where_seq = list(self.where)
        event_seq = list(self.event_types)
        if self.mode == "mix":
            self.mix_history = []
            self.last_mix_summary = None
            self._mix_dmrg_disabled_reason = None
            self._mix_dmrg_failed_sweep = None
        if not G_seq:
            def run_empty():
                if self.mode == "su":
                    self._prepare_su_state()
                    self._refresh_su_physical_state()
                return self.p

            return self._run_with_fit_copy_policy(
                run_empty,
                enabled=timing,
                finite_check=finite_check,
                event_count=0,
                sync_device=timing_sync_device,
            )
        self._validate_symmray_mode_support()
        self._validate_event_stream_for_run(G_seq, where_seq, event_seq)
        has_control = any(
            event_type in _CONTROL_EVENT_NAMES for event_type in event_seq
        )
        if self.mode == "su" and has_control:
            raise ValueError(
                "mode='su' supports gate-only streams; control events require "
                "a canonical MPS mode."
            )
        has_cap = any(event_type == "cap" for event_type in event_seq)
        layout_request = self._coalesce_layout_request(use_layout_finder, layout)
        persistent_layout_active = self._persistent_layout_plan is not None
        if self.mode == "su" and (
            persistent_layout_active or self._layout_request_enabled(layout_request)
        ):
            raise ValueError(
                "mode='su' does not support layout replay because its gauges "
                "belong to the current MPS site order."
            )
        if self.mode == "perm" and (
            persistent_layout_active or self._layout_request_enabled(layout_request)
        ):
            raise ValueError(
                "mode='perm' keeps a lazy logical-to-physical permutation; "
                "use either the perm mode or a persistent/transient layout, "
                "not both."
            )
        if has_cap and (
            persistent_layout_active or self._layout_request_enabled(layout_request)
        ):
            raise ValueError(
                "layout replay is not supported together with cap control events "
                "because cap changes the MPS length; run cap streams without a "
                "layout. measure/reset control events support layouts."
            )
        # Preserve the logical (pre-layout) event locations so control-event
        # bookkeeping (e.g. recorded measurement sites) always refers to the
        # user's site labels even when the run replays in a layout order.
        logical_where_seq = list(where_seq)
        if persistent_layout_active:
            if self._layout_request_enabled(layout_request):
                raise ValueError(
                    "a persistent layout is already installed; call run() without "
                    "use_layout_finder/layout arguments."
                )
            layout_plan = self._persistent_layout_plan
            self.last_layout_plan = layout_plan
        else:
            if self._layout_request_enabled(layout_request):
                warnings.warn(
                    "use_layout_finder/layout performs a temporary reorder and "
                    "swap-back; call apply_layout(...) for a persistent layout.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _, layout_plan = self._resolve_run_layout(
                layout_request,
                layout_order,
                layout_kwargs,
            )
        layout_current_order = None
        if layout_plan is not None:
            if layout_report and not persistent_layout_active:
                report = self._layout_report_text(layout_plan)
                if report:
                    print(report)
            layout_order_tuple = tuple(layout_plan["site_order"])
            G_seq, where_seq = self._layout_run_sequences(
                G_seq,
                where_seq,
                event_seq,
                layout_plan,
            )
            if not persistent_layout_active:
                layout_current_order = self._reorder_mps_to_logical_order(
                    layout_order_tuple
                )

        non_unitary = bool(non_unitary)
        if non_unitary or has_control:
            # Non-unitary and control-event runs can change the represented
            # norm without a unitary compression, so the next unitary stream
            # must establish a fresh local norm reference.
            self._unitary_previous_norm = None
        if not non_unitary:
            if normalize_every is not None and normalize_every is not False:
                raise ValueError("normalize_every requires non_unitary=True.")
            if normalize_final:
                raise ValueError("normalize_final requires non_unitary=True.")
        # This is the public-to-backend policy boundary. Validate the stream
        # and layout first, then resolve sentinels once so every mode receives
        # the same numeric cutoff and normalization contract. In particular,
        # ``None`` means Pepsy's default for ordinary paths but remains
        # observable as an omission for Quimb's method-specific MPO defaults.
        normalize_every = self._normalize_every_interval(
            normalize_every,
            non_unitary=non_unitary,
        )
        fit_min_iter = self._resolve_legacy_fit_option(
            canonical_name="fit_min_iter",
            canonical_value=fit_min_iter,
            canonical_default=2,
            legacy_name="mix_fit_min_iter",
            legacy_value=mix_fit_min_iter,
        )
        fit_rtol = self._resolve_legacy_fit_option(
            canonical_name="fit_rtol",
            canonical_value=fit_rtol,
            canonical_default="auto",
            legacy_name="mix_fit_rtol",
            legacy_value=mix_fit_rtol,
        )
        fit_patience = self._resolve_legacy_fit_option(
            canonical_name="fit_patience",
            canonical_value=fit_patience,
            canonical_default=2,
            legacy_name="mix_fit_patience",
            legacy_value=mix_fit_patience,
        )
        stabilize_unitary = self._resolve_legacy_fit_option(
            canonical_name="stabilize_unitary",
            canonical_value=stabilize_unitary,
            canonical_default=False,
            legacy_name="fit_stabilize_unitary",
            legacy_value=fit_stabilize_unitary,
        )
        fit_overlap_diagnostics = bool(fit_overlap_diagnostics)
        if stabilize_unitary and not non_unitary:
            # Each stabilized unitary stream needs a fresh raw-norm reference.
            # Mixed mode then carries this working value across its trials.
            self._unitary_previous_norm = None
        if compression_seed is not None:
            if not isinstance(compression_seed, Integral) or isinstance(
                compression_seed, bool
            ):
                raise ValueError("compression_seed must be an integer or None.")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError("compression_seed must be non-negative.")
        if self.mode in {"dmrg", "mix"}:
            if self._dmrg_mode_block_size is not None:
                # A named DMRG mode is an explicit block-size choice. Use the
                # generic ``mode='dmrg'`` spelling when custom per-run block
                # sizes are needed.
                alias_block_size = (
                    2 if self._dmrg_mode_block_size == 1
                    else self._dmrg_mode_block_size
                )
                if (
                    isinstance(fit_block_size, Integral)
                    and int(fit_block_size)
                    not in {1, 2, alias_block_size}
                ):
                    raise ValueError(
                        f"mode='dmrg{self._dmrg_mode_block_size}' fixes "
                        "fit_block_size; use mode='dmrg' for a custom value."
                    )
                fit_block_size = alias_block_size
            if self.mode == "mix" and non_unitary and not _trajectory_non_unitary:
                raise ValueError("mode='mix' is only for unitary gate streams.")
            if not isinstance(n_iter, Integral) or int(n_iter) < 1:
                raise ValueError("n_iter must be a positive integer.")
            if not isinstance(k_2q_batch, Integral) or k_2q_batch < 1:
                raise ValueError("k_2q_batch must be a positive integer.")
            if fit_layer_size is not None:
                if (
                    not isinstance(fit_layer_size, Integral)
                    or int(fit_layer_size) < 1
                ):
                    raise ValueError("fit_layer_size must be a positive integer or None.")
                if int(k_2q_batch) != 1 and int(k_2q_batch) != int(fit_layer_size):
                    raise ValueError(
                        "fit_layer_size and k_2q_batch specify different target "
                        "layer sizes; pass only one or make them equal."
                    )
                k_2q_batch = int(fit_layer_size)
            fit_max_span = self._resolve_fit_max_span(
                fit_max_span,
                k_2q_batch,
            )
            if (
                not isinstance(fit_block_size, Integral)
                or int(fit_block_size) not in {1, 2, 3}
            ):
                raise ValueError("fit_block_size must be 1, 2, or 3.")
            fit_block_size = int(fit_block_size)
            fit_adaptive_sweeps = self._resolve_legacy_fit_option(
                canonical_name="fit_adaptive_sweeps",
                canonical_value=fit_adaptive_sweeps,
                canonical_default=2,
                legacy_name="fit_three_site_sweeps",
                legacy_value=fit_three_site_sweeps,
            )
            if (
                not isinstance(fit_adaptive_sweeps, Integral)
                or int(fit_adaptive_sweeps) < 1
            ):
                raise ValueError("fit_adaptive_sweeps must be a positive integer.")
            fit_adaptive_sweeps = min(int(fit_adaptive_sweeps), int(n_iter))
            fit_sweep_sequence = FIT._validate_sweep_sequence(
                fit_sweep_sequence
            )
            target_cutoff = float(target_cutoff)
            if not np.isfinite(target_cutoff) or target_cutoff < 0.0:
                raise ValueError(
                    "target_cutoff must be a finite non-negative number."
                )
            fit_target_strategy = self._validate_fit_target_strategy(
                fit_target_strategy
            )
            fit_init_strategy = self._validate_fit_init_strategy(
                fit_init_strategy
            )
            try:
                fit_init_rand_strength = float(fit_init_rand_strength)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "fit_init_rand_strength must be finite and non-negative."
                ) from exc
            if not np.isfinite(fit_init_rand_strength) or fit_init_rand_strength < 0.0:
                raise ValueError(
                    "fit_init_rand_strength must be finite and non-negative."
                )
            if not isinstance(fit_init_seed, Integral) or isinstance(
                fit_init_seed, bool
            ):
                raise ValueError("fit_init_seed must be an integer.")
            fit_init_seed = int(fit_init_seed)
            if fit_init_seed < 0:
                raise ValueError("fit_init_seed must be non-negative.")
            if fit_target_strategy == "layered" and (
                self._has_symmray_data(self.p) or self.p.isfermionic()
            ):
                raise ValueError(
                    "fit_target_strategy='layered' is not available for "
                    "Symmray/fermionic MPS data; use 'auto' or 'mps'."
                )
            if (
                not isinstance(fit_min_iter, Integral)
                or int(fit_min_iter) < 1
            ):
                raise ValueError("fit_min_iter must be a positive integer.")
            if (
                not isinstance(fit_patience, Integral)
                or int(fit_patience) < 1
            ):
                raise ValueError("fit_patience must be a positive integer.")
            if self.mode == "dmrg" and non_unitary and fit_rtol == "auto":
                # Preserve the historical fixed-sweep behavior for
                # non-unitary DMRG unless the caller supplies an explicit
                # tolerance. Mixed mode is unitary-only and keeps adaptive
                # stopping by default.
                fit_rtol = None
            else:
                fit_rtol = self._resolve_fit_rtol(fit_rtol)
            current_max_bond = self.p.max_bond()
            if (
                self.mode == "mix"
                and current_max_bond is not None
                and current_max_bond > self.chi
            ):
                raise ValueError(
                    "mode='mix' requires the initial MPS max bond to be <= chi; "
                    "compress the state first or increase chi."
                )
        if normalize_every is not None and self.mode == "exact":
            raise ValueError(
                "automatic normalization uses MPS canonicalization and is not "
                "available in exact mode."
            )

        submpo_method = self._resolve_mpo_method(submpo_method)

        mode_kwargs = dict(
            n_iter=n_iter,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            mpo_cutoff_mode=mpo_cutoff_mode,
            k_2q_batch=k_2q_batch,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            mix_strict=bool(mix_strict),
            fit_min_iter=int(fit_min_iter),
            fit_rtol=fit_rtol,
            fit_patience=int(fit_patience),
            mix_sticky_nonfinite=bool(mix_sticky_nonfinite),
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            fit_max_span=fit_max_span,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_mpo_guess=bool(fit_mpo_guess),
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=bool(fit_single_pair_fast_path),
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=bool(stabilize_unitary),
            quality_check_every=quality_check_every,
            quality_check_repair=quality_check_repair,
        )

        if has_control:
            try:
                return self._run_with_fit_copy_policy(
                    lambda: self._run_segmented(
                        G_seq,
                        where_seq,
                        event_seq,
                        logical_where_seq=logical_where_seq,
                        progbar=progbar,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        measure_renormalize=measure_renormalize,
                        where_is_physical=persistent_layout_active,
                        mode_kwargs=mode_kwargs,
                    ),
                    enabled=timing,
                    finite_check=finite_check,
                    event_count=len(G_seq),
                    sync_device=timing_sync_device,
                )
            finally:
                if layout_current_order is not None:
                    self._reorder_mps_to_logical_order(
                        tuple(range(int(getattr(self.p, "L", 0)))),
                        current_order=layout_current_order,
                    )
                    self._normalize_visible_mps_order()

        try:
            return self._run_with_fit_copy_policy(
                lambda: self._execute_mode(
                    G_seq,
                    where_seq,
                    event_seq,
                    logical_where_seq=logical_where_seq,
                    progbar=progbar,
                    **mode_kwargs,
                ),
                enabled=timing,
                finite_check=finite_check,
                event_count=len(G_seq),
                sync_device=timing_sync_device,
            )
        finally:
            if layout_current_order is not None:
                self._reorder_mps_to_logical_order(
                    tuple(range(int(getattr(self.p, "L", 0)))),
                    current_order=layout_current_order,
                )
                self._normalize_visible_mps_order()

    def _run_with_fit_copy_policy(self, executor, *, finite_check=False, **timing_options):
        """Scope copy capabilities and runtime validation to one replay.

        State replacement and control boundaries may change the MPS object,
        but validated replay preserves its array backend. Classify each
        network/array type once, lazily, and release the cache on all exits.
        Calls outside replay always inspect their actual input state for
        copy capabilities. Runtime non-finite validation is opt-in in all modes.
        """
        previous_cache = self._fit_copy_policy_cache
        previous_finite_check = self._finite_check_enabled
        self._fit_copy_policy_cache = {}
        self._finite_check_enabled = bool(finite_check)
        try:
            if finite_check:
                # Diagnostic validation is off by default. Warn once for this
                # replay; owned FIT instances suppress only their duplicate.
                warnings.warn(
                    "MpsOptimizer finite_check is enabled: this optional "
                    "diagnostic is off by default and is not required for "
                    "normal optimization. It adds tensor/norm checks and can "
                    "synchronize devices; use finite_check=False to avoid "
                    "this overhead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                def checked_executor():
                    result = executor()
                    if not self._mps_data_is_finite(self.p):
                        raise FloatingPointError("Replay produced non-finite MPS data.")
                    return result

                return self._run_with_timing(checked_executor, **timing_options)
            return self._run_with_timing(executor, **timing_options)
        finally:
            self._fit_copy_policy_cache = previous_cache
            self._finite_check_enabled = previous_finite_check

    def _run_with_timing(
        self,
        executor,
        *,
        enabled,
        event_count,
        sync_device=False,
    ):
        """Execute one replay segment and optionally retain wall-clock timing."""
        if not enabled:
            return executor()

        previous_timing_state = self._timing_state
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
        }
        self._sync_timing_device()
        started = time.perf_counter()
        status = "complete"
        try:
            return executor()
        except BaseException:
            status = "failed"
            raise
        finally:
            self._sync_timing_device()
            try:
                final_bond = int(self.p.max_bond())
            except (AttributeError, TypeError, ValueError):
                final_bond = None
            timing_state = self._timing_state
            self.last_run_timing = {
                "status": status,
                "mode": self.mode,
                "mode_alias": self._dmrg_mode_alias,
                "event_count": int(event_count),
                "elapsed_seconds": float(time.perf_counter() - started),
                "final_bond": final_bond,
                "chi": int(self.chi),
                "backend": self.backend,
                "backend_dtype": self.backend_dtype,
                "backend_device": self.backend_device,
                "timing_sync_device": bool(sync_device),
                # The completed timing state has no remaining writer. Transfer
                # its containers directly and keep the defensive copy at the
                # public ``get_run_timing`` boundary.
                "stages": timing_state["stages"],
                "fit_steps": timing_state["fit_steps"],
                "fit_totals": _summarize_fit_timing(
                    timing_state["fit_steps"]
                ),
                # FIT diagnostics are a flat scalar record. Copy the mapping
                # without adding an internal deep-copy cost; the public
                # ``get_run_timing`` boundary owns the defensive deep copy.
                "fit_diagnostics": (
                    None
                    if self._last_dmrg_fit_diagnostics is None
                    else dict(self._last_dmrg_fit_diagnostics)
                ),
                "mix_summary": (
                    deepcopy(self.last_mix_summary)
                    if self.mode == "mix"
                    else None
                ),
            }
            self._timing_state = previous_timing_state

    def _record_timing_stage(self, name, elapsed):
        """Accumulate one completed inclusive stage measurement."""
        stage = self._timing_state["stages"].setdefault(
            str(name),
            {"calls": 0, "elapsed_seconds": 0.0},
        )
        stage["calls"] += 1
        stage["elapsed_seconds"] += elapsed

    def _sync_timing_device(self, value=None):
        """Apply an accelerator barrier only for synchronized profiling."""
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

        # Torch and CuPy synchronize their device/stream globally. JAX work is
        # tied to returned arrays, so wait on the actual stage result rather
        # than an unrelated already-ready tensor from the live MPS.
        self._sync_timing_device(result)
        self._record_timing_stage(name, time.perf_counter() - started)
        return result

    def get_run_timing(self):
        """Return the most recent opt-in replay and stage timing record."""
        return deepcopy(self.last_run_timing)

    def get_fit_diagnostics(self):
        """Return a copy of the latest DMRG/FIT convergence diagnostics.

        The result is ``None`` before a DMRG/FIT update has completed and for
        replay modes that do not use FIT. The returned dictionary is
        independent of the optimizer's internal diagnostic state. Successful
        FIT updates include the ordinary convergence metadata. The optional
        ``fit_overlap_fidelity`` and ``fit_overlap_infidelity`` fields are
        populated only when ``run(fit_overlap_diagnostics=True)`` requests a
        direct contraction of the fitted MPS with the disposable exact FIT
        target. Those fields are target-overlap diagnostics and are
        intentionally separate from norm-survival fields such as
        ``cumulative_fidelity``. If the optional contraction is unavailable
        for a backend, both values are ``None`` and ``fit_overlap_error``
        records the diagnostic failure without rejecting the successful FIT
        update.
        """
        return deepcopy(self._last_dmrg_fit_diagnostics)

    def _fit_overlap_diagnostics(self, target, fitted):
        """Return the optional direct FIT-target overlap readout.

        This is deliberately separate from the automatic norm ledger.  The
        norm ledger is available for every compression backend and only reads
        the retained canonical centre.  This contraction compares the final
        FIT MPS with the disposable exact DMRG target, so it is a genuine
        target-state overlap but is specific to DMRG and costs an additional
        contraction.
        """
        # DMRG already paid for the target construction. Keep this diagnostic
        # contraction deterministic and local: the high-level ``auto-hq``
        # optimizer can create a multiprocessing pool, which is unnecessary
        # for a one-window overlap and unavailable in restricted runtimes.
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
                raise ValueError("FIT target overlap is non-finite")
        except Exception as exc:  # diagnostic only; FIT result remains valid
            return {
                "fit_overlap_fidelity": None,
                "fit_overlap_infidelity": None,
                "fit_overlap_error": f"{type(exc).__name__}: {exc}",
            }
        overlap = min(1.0, max(0.0, overlap))
        return {
            "fit_overlap_fidelity": overlap,
            "fit_overlap_infidelity": float(1.0 - overlap),
            "fit_overlap_error": None,
        }

    def _execute_mode(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        n_iter,
        progbar,
        cutoff,
        cutoff_mode,
        mpo_cutoff_mode=None,
        k_2q_batch,
        normalize_every,
        normalize_final,
        normalize_eps,
        non_unitary,
        submpo_method,
        compression_seed=None,
        mix_strict=False,
        fit_min_iter=2,
        fit_rtol=None,
        fit_patience=2,
        mix_sticky_nonfinite=False,
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        fit_max_span="auto",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        quality_check_every=None,
        quality_check_repair=True,
    ):
        """Dispatch a gate/subMPO segment to the active mode backend.

        This is the mode-specific core of :meth:`run`; ``G_seq``/``where_seq``/
        ``event_seq`` must contain only ``"gate"``/``"submpo"`` events. Control
        events (measure/cap/reset) are handled by :meth:`_run_segmented`.
        """
        # The stream-install boundary already normalized and validated these
        # payloads. Do not repeat that work for each control-delimited segment.
        # Dispatch directly to the mode implementations, which share the same
        # validated stream but have different state contracts:
        # DMRG owns local variational targets, MPO owns Quimb compression,
        # swap/perm own endpoint movement, and exact/SU deliberately bypass
        # canonical-center bookkeeping. Keeping the branches here prevents a
        # control-event caller from accidentally selecting a gate-only kernel.
        if self.mode == "dmrg":
            self._timed_call("dmrg.prepare", self._prepare_dmrg_state)
            self._timed_call(
                "dmrg.replay",
                self._run_dmrg,
                G_seq,
                where_seq,
                event_seq=event_seq,
                n_iter=n_iter,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                k_2q_batch=k_2q_batch,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                fit_min_iter=fit_min_iter,
                fit_rtol=fit_rtol,
                fit_patience=fit_patience,
                fit_finite_check=finite_check,
                fit_block_size=fit_block_size,
                fit_adaptive_sweeps=fit_adaptive_sweeps,
                fit_sweep_sequence=fit_sweep_sequence,
                fit_max_span=fit_max_span,
                target_cutoff=target_cutoff,
                fit_target_strategy=fit_target_strategy,
                fit_mpo_guess=fit_mpo_guess,
                fit_init_strategy=fit_init_strategy,
                fit_init_rand_strength=fit_init_rand_strength,
                fit_init_seed=fit_init_seed,
                fit_single_pair_fast_path=fit_single_pair_fast_path,
                finite_check=finite_check,
                fit_overlap_diagnostics=fit_overlap_diagnostics,
                stabilize_unitary=stabilize_unitary,
                quality_check_every=quality_check_every,
                quality_check_repair=quality_check_repair,
            )
            return self.p

        if self.mode == "mix":
            self._timed_call(
                "mix.replay",
                self._run_mix,
                G_seq,
                where_seq,
                event_seq,
                logical_where_seq=logical_where_seq,
                n_iter=n_iter,
                k_2q_batch=k_2q_batch,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                submpo_method=submpo_method,
                compression_seed=compression_seed,
                mix_strict=mix_strict,
                fit_min_iter=fit_min_iter,
                fit_rtol=fit_rtol,
                fit_patience=fit_patience,
                sticky_nonfinite=mix_sticky_nonfinite,
                fit_block_size=fit_block_size,
                fit_adaptive_sweeps=fit_adaptive_sweeps,
                fit_sweep_sequence=fit_sweep_sequence,
                fit_max_span=fit_max_span,
                target_cutoff=target_cutoff,
                fit_target_strategy=fit_target_strategy,
                fit_init_strategy=fit_init_strategy,
                fit_init_rand_strength=fit_init_rand_strength,
                fit_init_seed=fit_init_seed,
                fit_single_pair_fast_path=fit_single_pair_fast_path,
                finite_check=finite_check,
                fit_overlap_diagnostics=fit_overlap_diagnostics,
                stabilize_unitary=stabilize_unitary,
                non_unitary=non_unitary,
                quality_check_every=quality_check_every,
                quality_check_repair=quality_check_repair,
            )
            return self.p

        if self._is_mpo_mode(self.mode):
            # Report the selected compression method rather than the internal
            # implementation family. For example, ``mode="mpo"`` and
            # ``mode="direct"`` both report ``direct.replay``.
            replay_name = self._mode_mpo_method(self.mode)
            self._timed_call(
                f"{replay_name}.replay",
                self._run_mpo,
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=mpo_cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                submpo_method=submpo_method,
                compression_seed=compression_seed,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "swap":
            self._timed_call(
                "swap.replay",
                self._run_swap,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "perm":
            self._timed_call(
                "perm.replay",
                self._run_perm,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "svd":
            self._timed_call(
                "svd.replay",
                self._run_svd,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=normalize_every,
                normalize_final=normalize_final,
                normalize_eps=normalize_eps,
                non_unitary=non_unitary,
                stabilize_unitary=stabilize_unitary,
            )
            return self.p

        if self.mode == "su":
            self._timed_call(
                "su.replay",
                self._run_su,
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return self.p

        if self.mode == "exact":
            self._timed_call(
                "exact.replay",
                self._run_exact,
                G_seq,
                where_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return self.p

        raise ValueError(f"Unknown mode: {self.mode}")

    def _run_segmented(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        progbar,
        cutoff,
        cutoff_mode,
        measure_renormalize,
        where_is_physical=False,
        mode_kwargs,
    ):
        """Replay a stream containing measure/cap/reset control events.

        Consecutive ``"gate"``/``"submpo"`` events are grouped into segments run
        through :meth:`_execute_mode` (using the active mode), while control
        events are applied directly to ``self.p`` between segments so the same
        stream works in every mode. ``cap`` events change the MPS length, so
        later event site labels refer to the shortened chain.

        ``where_seq`` holds the execution locations (already mapped into the
        active layout order when a layout is used); ``logical_where_seq`` holds
        the matching user-facing locations for bookkeeping such as recorded
        measurement sites. When no layout is active the two are identical.
        ``where_is_physical`` prevents persistent-layout locations from being
        mapped a second time by the control-event dispatcher.
        """
        if logical_where_seq is None:
            logical_where_seq = where_seq
        seg_G = []
        seg_where = []
        seg_logical_where = []
        seg_event = []

        def flush():
            # A control event is a state boundary: all preceding gates must be
            # committed before its expectation/probability is evaluated, and
            # all following gates must see the collapsed/reset/capped state.
            # Therefore segments are intentionally never allowed to cross a
            # control event, even when the active mode could batch the gates.
            if seg_G:
                self._execute_mode(
                    list(seg_G),
                    list(seg_where),
                    list(seg_event),
                    logical_where_seq=list(seg_logical_where),
                    progbar=progbar,
                    **mode_kwargs,
                )
                seg_G.clear()
                seg_where.clear()
                seg_logical_where.clear()
                seg_event.clear()

        for payload, where, logical_where, event_type in zip(
            G_seq, where_seq, logical_where_seq, event_seq
        ):
            if event_type in _CONTROL_EVENT_NAMES:
                flush()
                # ``where`` may already be physical when a persistent layout
                # is active. ``record_where`` remains logical so measurements
                # and feed-forward records use the user's labels.
                self._apply_control_event(
                    event_type,
                    payload,
                    where,
                    record_where=logical_where,
                    where_is_physical=where_is_physical,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    measure_renormalize=measure_renormalize,
                    mode_kwargs=mode_kwargs,
                )
            else:
                seg_G.append(payload)
                seg_where.append(where)
                seg_logical_where.append(logical_where)
                seg_event.append(event_type)

        flush()
        return self.p

    # ------------------------------------------------------------------ #
    # Control events (measure / cap / reset)
    # ------------------------------------------------------------------ #
    def _apply_control_event(self, *args, **kwargs):
        """Apply one control event, optionally recording its stage time."""
        if self._timing_state is None:
            return self._apply_control_event_impl(*args, **kwargs)
        name = args[0] if args else kwargs.get("name", "unknown")
        return self._timed_call(
            f"control.{name}",
            self._apply_control_event_impl,
            *args,
            **kwargs,
        )

    def _apply_control_event_impl(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        name,
        payload,
        where,
        *,
        record_where=None,
        where_is_physical=False,
        cutoff,
        cutoff_mode,
        measure_renormalize,
        mode_kwargs=None,
    ):
        """Apply one measure/cap/reset control event to ``self.p``."""
        if record_where is None:
            record_where = where
        self._ensure_mps_state()
        self._ensure_tracked_center()
        execution_where = (
            tuple(int(site) for site in where)
            if where_is_physical
            else self._logical_to_physical_where(where)
        )
        if name == "conditional":
            record_index, expected = _resolve_conditional(
                payload, len(self.measurements)
            )
            record = self.measurements[record_index]
            outcome = int(getattr(record, "outcome", record[2]))
            if int(outcome < 0) != expected:
                return
            action_payloads, action_wheres, action_types = _normalize_gate_queue(
                (payload["action"],)
            )
            if len(action_payloads) != 1:
                raise ValueError(
                    "conditional action must normalize to exactly one stream entry."
                )
            action_where = action_wheres[0]
            action_type = action_types[0]
            if action_type in _CONTROL_EVENT_NAMES:
                self._apply_control_event(
                    action_type,
                    action_payloads[0],
                    action_where,
                    record_where=action_where,
                    where_is_physical=where_is_physical,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    measure_renormalize=measure_renormalize,
                    mode_kwargs=mode_kwargs,
                )
            else:
                physical_where = (
                    tuple(int(site) for site in action_where)
                    if where_is_physical
                    else self._logical_to_physical_where(action_where)
                )
                self._execute_mode(
                    [action_payloads[0]],
                    [physical_where],
                    [action_type],
                    logical_where_seq=[action_where],
                    progbar=False,
                    n_iter=1,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    k_2q_batch=1,
                    normalize_every=None,
                    normalize_final=False,
                    normalize_eps=1e-12,
                    non_unitary=False,
                    submpo_method="direct",
                    finite_check=mode_kwargs.get("finite_check", False),
                    fit_overlap_diagnostics=mode_kwargs.get(
                        "fit_overlap_diagnostics",
                        False,
                    ),
                )
            return
        if name == "measure":
            self._apply_measure_event(
                payload["pauli"],
                execution_where,
                payload.get("outcome"),
                record_where=record_where,
                renormalize=measure_renormalize,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                mode_kwargs=mode_kwargs,
            )
        elif name == "cap":
            logical_site = int(where[0])
            physical_site = int(execution_where[0])
            self._apply_cap_event(
                execution_where,
                payload["vec"],
                payload.get("absorb", "left"),
            )
            self._apply_effective_cap(physical_site)
            self._update_permutation_after_cap(logical_site, physical_site)
        elif name == "reset":
            self._apply_reset_event(
                execution_where,
                payload.get("axes"),
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
        elif name == "measure_reset":
            self._apply_measure_reset_event(
                payload["axes"],
                execution_where,
                payload["outcomes"],
                record_where=record_where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                mode_kwargs=mode_kwargs,
            )
        else:  # pragma: no cover - guarded by parsing
            raise ValueError(f"Unknown control event {name!r}.")

        if name != "cap":
            self._record_effective_event(execution_where, event_type=name)

    def _ensure_mps_state(self):
        """Ensure ``self.p`` is a :class:`qtn.MatrixProductState`.

        ``mode="exact"`` fully contracts the state into a single dense tensor;
        control events operate on MPS structure, so rebuild an MPS from the
        physical indices (in ``self.ind_id`` order) when needed.
        """
        p = self.p
        if isinstance(p, qtn.MatrixProductState):
            return p
        outer = set(p.outer_inds())
        ordered = []
        site = 0
        while True:
            ind = self._format_ind(site)
            if ind not in outer:
                break
            ordered.append(ind)
            site += 1
        if len(ordered) != len(outer):
            raise ValueError(
                "cannot rebuild an MPS for a control event: physical indices "
                "are not the standard 1D site-index family."
            )
        dense = p.contract(all, output_inds=ordered, optimize=self.contraction_opt)
        arr = np.asarray(ar.to_numpy(dense.data if hasattr(dense, "data") else dense))
        mps = qtn.MatrixProductState.from_dense(arr, [d for d in arr.shape])
        self.p = self._install_represented_norm(mps)
        # Freshly rebuilt: mark the centre as unknown so the next control event
        # establishes a tracked orthogonality centre (never via a blind scan).
        self.info_c["cur_orthog"] = None
        return self.p

    def _ensure_tracked_center(self):
        """Guarantee ``info_c['cur_orthog']`` is a concrete tracked centre.

        Control events always move the orthogonality centre explicitly rather
        than rescanning with ``calc_current_orthog_center``. When the centre is
        unknown (e.g. a freshly rebuilt exact-mode state, or an ``exact``-mode
        run that never canonicalized), establish one by canonicalizing to site
        ``0`` with a full-span ``cur_orthog`` and record it.
        """
        cur = self.info_c.get("cur_orthog")
        if cur not in (None, "calc"):
            return
        L = int(getattr(self.p, "L", 0))
        if L <= 0:
            return
        self.p.canonize(
            [0],
            cur_orthog=(0, max(0, L - 1)),
            info=self.info_c,
        )
        self.info_c["cur_orthog"] = (0, 0)

    def _state_backend_like(self):
        """Return a representative backend array from ``self.p`` tensor data."""
        return self._state_backend_like_for(self.p)

    @staticmethod
    def _state_backend_info_for(state):
        """Validate and describe the common backend of an MPS-like state."""
        return backend_infer(state)

    def backend_info(self):
        """Return the state-derived backend, dtype, and device diagnostics."""
        info = self._state_backend_info_for(self.p)
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        return info

    @staticmethod
    def _state_backend_like_for(state):
        """Return a representative raw array from an MPS-like state."""
        for tensor in getattr(state, "tensors", ()):
            return tensor.data
        return None

    @staticmethod
    def _backend_mismatch_hint(target_signature):
        """Return concise guidance for preparing a stream payload."""
        if target_signature[0] == "symmray":
            return (
                "Native Symmray states require native Symmray gates with the "
                "matching charge and fermionic metadata; a dense gate cannot "
                "be made safe by a generic cast."
            )
        return (
            "Convert the payload yourself with the same converter used for the "
            "MPS, for example `gate = to_backend(gate)`, before passing it to "
            "MpsOptimizer or set_gates."
        )

    @staticmethod
    def _backend_signatures_compatible(source_signature, target_signature):
        """Return whether two payloads can be used without backend transfer."""
        if source_signature[:1] == target_signature[:1] == ("symmray",):
            # A real native gate is safe to apply to a complex native state:
            # the contraction promotes the result while preserving charge and
            # fermionic metadata. Requiring an exact dtype here rejected
            # ordinary imaginary-time streams whose gates are real-valued.
            return (
                source_signature[0] == target_signature[0]
                and source_signature[2:] == target_signature[2:]
                and np.can_cast(
                    np.dtype(source_signature[1]),
                    np.dtype(target_signature[1]),
                    casting="safe",
                )
            )
        return backend_signatures_compatible(source_signature, target_signature)

    def _validate_gate_stream_backend(
        self,
        gates,
        event_types,
        *,
        state=None,
        path_prefix="stream",
    ):
        """Require user-supplied gate payloads to match the live MPS backend.

        Backend conversion is intentionally an explicit caller operation. This
        validation runs at every public stream/state boundary, while the
        execution path only receives payloads that are already compatible.
        Control events are excluded because their internal dense operators are
        deliberately created and mapped by the optimizer itself. Conditional
        gate actions are checked recursively.
        """
        if not gates:
            return
        if len(gates) != len(event_types):
            raise ValueError(
                "MpsOptimizer backend validation requires payloads and event "
                "types to have the same length."
            )
        state = self.p if state is None else state
        like = self._state_backend_like_for(state)
        if like is None:
            return
        target_signature = _array_backend_signature(like)
        mismatches = []
        for index, (payload, event_type) in enumerate(zip(gates, event_types)):
            path = f"{path_prefix}[{index}]"
            if event_type == "gate":
                source_signature = _array_backend_signature(payload)
                if not self._backend_signatures_compatible(
                    source_signature, target_signature
                ):
                    mismatches.append((path, "gate", source_signature))
            elif event_type == "submpo":
                tensors = tuple(getattr(payload, "tensors", ()))
                source_signatures = {
                    _array_backend_signature(tensor.data) for tensor in tensors
                }
                for source_signature in sorted(
                    (
                        source_signature
                        for source_signature in source_signatures
                        if not self._backend_signatures_compatible(
                            source_signature, target_signature
                        )
                    ),
                    key=repr,
                ):
                    mismatches.append((path, "sub-MPO", source_signature))
            elif event_type == "conditional":
                action = payload.get("action")
                action_gates, _, action_types = _normalize_gate_queue((action,))
                self._validate_gate_stream_backend(
                    action_gates,
                    action_types,
                    state=state,
                    path_prefix=f"{path}.action",
                )

        if not mismatches:
            return
        details = "; ".join(
            f"{path} ({kind}) has {source!r}"
            for path, kind, source in mismatches[:8]
        )
        if len(mismatches) > 8:
            details += f"; ... and {len(mismatches) - 8} more"
        raise TypeError(
            "MpsOptimizer requires every gate and sub-MPO payload to match the "
            f"MPS backend/device and required dtype {target_signature!r} "
            "before use; "
            f"{details}. {self._backend_mismatch_hint(target_signature)}"
        )

    def _to_state_backend(self, array):
        """Return ``array`` cast to the backend and dtype owned by ``self.p``."""
        like = self._state_backend_like()
        if like is None:
            return np.asarray(ar.to_numpy(array), dtype=complex)
        target_signature = _array_backend_signature(like)
        source_signature = _array_backend_signature(array)
        if source_signature == target_signature:
            return array
        if self._is_symmray_array(array) and self._is_symmray_array(like):
            # Symmray arrays deliberately do not implement Autoray's generic
            # ``array(..., like=symmray_array)`` constructor. Their outer
            # object has no scalar dtype either, so the generic dtype fast
            # path below cannot establish compatibility. Native Symmray gates
            # already carry their own block backend and must pass through as
            # graded arrays rather than being rebuilt as dense payloads.
            return array
        if target_signature[0] == "symmray" and source_signature[0] != "symmray":
            raise TypeError(
                "Cannot convert a dense gate/operator payload into a native "
                "Symmray MPS without charge and fermionic metadata. Build the "
                "payload as a Symmray array on the target U1/U1U1 backend."
            )
        converter = infer_backend_converter_from_sample(like)
        if converter is not None:
            return converter(array)
        if target_signature[0] == "numpy":
            return ar.to_numpy(array)
        # Keep the old Autoray fallback for optional/custom dense backends.
        return ar.do("array", array, like=like)

    def to_backend(self, array):
        """Return ``array`` on the backend currently owned by ``self.p``.

        Already-compatible arrays are returned by identity. This public helper
        is intentionally state-derived so replacing the MPS with :meth:`set_p`
        automatically changes the target backend without stale converter state.
        """
        return self._to_state_backend(array)

    def _pauli_operator(self, pauli, where):
        """Return the dense Pauli operator (numpy) for ``pauli`` on ``where``."""
        chars = [c for c in str(pauli).upper() if not c.isspace()]
        if len(chars) != len(where):
            raise ValueError(
                f"pauli string {pauli!r} has {len(chars)} axes but where {where!r} "
                f"has {len(where)} site(s)."
            )
        try:
            op = _PAULI_1Q[chars[0]]
            for axis in chars[1:]:
                op = np.kron(op, _PAULI_1Q[axis])
        except KeyError as exc:  # pragma: no cover - guarded by dict lookup
            raise ValueError(f"unknown Pauli axis in {pauli!r}.") from exc
        return op

    def _build_pauli_projector_submpo(self, pauli, where, outcome):
        """Build ``(I + outcome * P) / 2`` as a bond-two windowed sub-MPO.

        The dense projector is retained for native Symmray/fermionic states,
        where a dense MPO cannot carry the target charge and dummy-mode
        metadata. Dense MPS states use the two product branches directly:
        ``0.5 * I`` and ``0.5 * outcome * P``.
        """
        if len(where) < 2:
            return None
        if self._has_symmray_data(self.p) or self.p.isfermionic():
            return None

        chars = [c for c in str(pauli).upper() if not c.isspace()]
        sites = tuple(int(site) for site in where)
        if len(chars) != len(sites):
            raise ValueError(
                f"pauli string {pauli!r} has {len(chars)} axes but where "
                f"{where!r} has {len(sites)} site(s)."
            )
        if len(set(sites)) != len(sites):
            raise ValueError("measurement sites must be unique.")
        if any(axis not in _PAULI_1Q for axis in chars):
            raise ValueError(f"unknown Pauli axis in {pauli!r}.")

        axes_by_site = dict(zip(sites, chars))
        span = tuple(range(min(sites), max(sites) + 1))
        dtype_name = str(self.backend_dtype).lower()
        dtype = np.complex64 if "complex64" in dtype_name else np.complex128
        identity = np.eye(2, dtype=dtype)
        arrays = []

        for position, site in enumerate(span):
            local = np.asarray(
                _PAULI_1Q[axes_by_site.get(site, "I")],
                dtype=dtype,
            )
            if position == 0:
                tensor = np.zeros((2, 2, 2), dtype=dtype)
                tensor[0] = identity
                tensor[1] = local
            elif position == len(span) - 1:
                tensor = np.zeros((2, 2, 2), dtype=dtype)
                tensor[0] = 0.5 * identity
                tensor[1] = 0.5 * int(outcome) * local
            else:
                tensor = np.zeros((2, 2, 2, 2), dtype=dtype)
                tensor[0, 0] = identity
                tensor[1, 1] = local
            arrays.append(self._to_state_backend(tensor))

        submpo = qtn.MatrixProductOperator(
            arrays,
            sites=span,
            L=int(self.p.L),
            shape="lrud",
            upper_ind_id=self.ind_id,
            lower_ind_id="b{}",
            site_tag_id="I{}",
        )
        return submpo, span

    def _apply_submpo_with_method(
        self,
        p,
        submpo,
        where,
        *,
        method,
        cutoff,
        cutoff_mode,
        info=None,
        seed=None,
    ):
        """Apply and compress a sub-MPO using the selected Quimb method."""
        if info is None:
            info = self._info_for_state(p)
        method = self._normalize_submpo_method(method)
        compress_opts = self._submpo_compress_opts(
            method,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        # The direct API is preferred for a full-chain or ordinary local
        # payload. A partitioned interior payload needs the local workaround
        # only for wrappers whose nested call assumes every chain site has a
        # matching tag; keeping the partition local avoids both tag failures
        # and unnecessary full-chain contraction work.
        if (
            method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
            and _is_interior_submpo_span(p, where)
        ):
            _apply_submpo_with_interior_workaround(
                p,
                submpo,
                where,
                chi=self.chi,
                method=method,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                info=info,
                inplace_mpo=False,
                seed=seed,
                **{
                    key: value
                    for key, value in compress_opts.items()
                    if key == "optimize"
                },
            )
        else:
            _run_seeded_quimb(
                seed,
                p.gate_with_submpo_,
                submpo,
                where=where,
                method=method,
                max_bond=self.chi,
                info=info,
                inplace_mpo=False,
                **compress_opts,
            )
        return p

    def _apply_dense_operator(self, p, op, where, *, max_bond, cutoff, cutoff_mode, info=None):
        """Apply a dense operator ``op`` on ``where`` sites of MPS ``p`` in place.

        ``info`` is the canonicalization tracking dict; it defaults to
        ``self.info_c`` for operations on ``self.p`` and should be an isolated
        dict when acting on a throwaway copy so the tracked centre is preserved.
        """
        if info is None:
            info = self._info_for_state(p)
        where = tuple(int(site) for site in where)
        op_b = self._to_state_backend(op)
        if len(where) == 1:
            self._apply_gate(
                p,
                op_b,
                where,
                contract=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                inplace=True,
            )
        else:
            p.gate_nonlocal_(
                op_b,
                where,
                dims=None,
                max_bond=max_bond,
                info=info,
                method="direct",
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
        return p

    def _state_expectation(self, pauli, where):
        """Return the normalized expectation ``<P> = Re <psi|P|psi> / <psi|psi>``.

        For MPS implementations exposing ``local_expectation_canonical``, move
        the tracked orthogonality centre around the support and contract only
        the local reduced density matrix. This keeps the cost proportional to
        the support span rather than the full chain. The fallback preserves
        compatibility with older Quimb versions without that method.
        """
        p = self.p
        op = self._to_state_backend(self._pauli_operator(pauli, where))
        local_expectation = getattr(p, "local_expectation_canonical", None)
        if callable(local_expectation):
            return self._real_float(
                local_expectation(
                    op,
                    tuple(int(site) for site in where),
                    normalized=True,
                    info=self.info_c,
                    optimize=self.contraction_opt,
                )
            )

        # Compatibility path for older Quimb releases without local MPS
        # expectation support.
        p_op = p.copy()
        self._apply_dense_operator(
            p_op, op, where, max_bond=None, cutoff=0.0, cutoff_mode="abs", info={}
        )
        overlap = (p.H & p_op).contract(
            all, output_inds=(), optimize=self.contraction_opt
        )
        norm_sq = (p.H & p).contract(
            all, output_inds=(), optimize=self.contraction_opt
        )
        norm_val = self._real_float(norm_sq)
        if norm_val == 0.0:
            return 0.0
        return self._real_float(overlap) / norm_val

    def _recanonize_center(self, site, *, renormalize):
        """Move the orthogonality centre to ``site`` and track it exactly.

        Canonicalizes from the currently tracked centre (never a blind scan) so
        ``site`` becomes a single-site orthogonality centre, records it in
        ``info_c``, and, when ``renormalize`` is set, rescales that centre tensor
        to unit norm (its Frobenius norm equals the represented state norm).
        """
        site = int(site)
        self.p.canonize(
            [site],
            cur_orthog=self._current_orthog(self.p),
            info=self.info_c,
        )
        self.info_c["cur_orthog"] = (site, site)
        if not renormalize:
            return
        center = self.p[self.p.site_tag(site)]
        norm = self._real_float(center.norm())
        if norm > 0.0:
            center.modify(data=center.data / norm)
        if hasattr(self.p, "exponent"):
            self.p.exponent = 0.0

    def _finish_measurement_center(self, site, *, renormalize):
        """Track and optionally normalize a post-measurement center."""
        site = int(site)
        current = self._current_orthog(self.p)
        if current != (site, site):
            self.p.canonize(
                [site],
                cur_orthog=current,
                info=self.info_c,
            )
        self.info_c["cur_orthog"] = (site, site)
        if not renormalize:
            return
        center = self.p[self.p.site_tag(site)]
        norm = self._real_float(center.norm())
        if norm > 0.0:
            center.modify(data=center.data / norm)
        if hasattr(self.p, "exponent"):
            self.p.exponent = 0.0

    def _apply_measure_event(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        pauli,
        where,
        outcome,
        *,
        record_where=None,
        renormalize,
        cutoff,
        cutoff_mode,
        norm_kind="measure",
        mode_kwargs=None,
    ):
        """Measure Pauli ``pauli`` on ``where``, collapse, and record the result.

        ``where`` is the execution location; ``record_where`` (defaulting to
        ``where``) is the user-facing location stored in :attr:`measurements`.
        """
        if record_where is None:
            record_where = where
        # Compute the physical Born probability before constructing or applying
        # any localizer. The localizer is a Clifford change of Pauli frame; it
        # can make the same observable look simpler, but its post-frame
        # expectation is not the probability of the original branch.
        exp = self._state_expectation(pauli, where)
        p_plus = min(max(0.5 * (1.0 + exp), 0.0), 1.0)
        if outcome is None:
            m = 1 if self._rng.random() < p_plus else -1
        else:
            m = 1 if int(outcome) >= 0 else -1
        prob = p_plus if m > 0 else (1.0 - p_plus)
        if outcome is not None and prob < 1e-12:
            raise ValueError(
                f"forced measure outcome {outcome} has ~0 probability ({prob:.2e})."
            )
        # Move the orthogonality centre to the (anchor) collapse site so the
        # projector acts at the centre and truncation/renormalization stay
        # local and exactly tracked.
        anchor = min(int(site) for site in where)
        self.canonize_mps(self.p, anchor)
        input_norm = self._real_float(ar.do("abs", self.p.norm()))

        collapse_center = None
        projector_submpo = self._build_pauli_projector_submpo(
            pauli,
            where,
            m,
        )
        if projector_submpo is not None:
            # Dense multi-site projectors stay as a bond-two MPO. DMRG receives
            # it as a lazy exact target; other MPS modes use their selected
            # Quimb compression method directly. This keeps target formation
            # separate from output compression and avoids a dense 2**k matrix.
            submpo, span = projector_submpo
            if self.mode == "dmrg" and mode_kwargs is not None:
                projected_norm, collapse_center = self._run_dmrg_measurement(
                    submpo,
                    span,
                    n_iter=mode_kwargs["n_iter"],
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    fit_min_iter=mode_kwargs["fit_min_iter"],
                    fit_rtol=mode_kwargs["fit_rtol"],
                    fit_patience=mode_kwargs["fit_patience"],
                    fit_block_size=mode_kwargs["fit_block_size"],
                    fit_adaptive_sweeps=mode_kwargs["fit_adaptive_sweeps"],
                    fit_sweep_sequence=mode_kwargs["fit_sweep_sequence"],
                    target_cutoff=mode_kwargs["target_cutoff"],
                    fit_target_strategy=mode_kwargs["fit_target_strategy"],
                    fit_mpo_guess=mode_kwargs["fit_mpo_guess"],
                    fit_init_strategy=mode_kwargs["fit_init_strategy"],
                    fit_init_rand_strength=mode_kwargs["fit_init_rand_strength"],
                    fit_init_seed=mode_kwargs["fit_init_seed"],
                    fit_single_pair_fast_path=mode_kwargs[
                        "fit_single_pair_fast_path"
                    ],
                    finite_check=mode_kwargs.get("finite_check", False),
                    fit_overlap_diagnostics=mode_kwargs[
                        "fit_overlap_diagnostics"
                    ],
                    measurement_index=len(self.measurements),
                )
            else:
                method = (
                    self._mode_mpo_method(self.mode)
                    if self._is_mpo_mode(self.mode)
                    else "direct"
                )
                method_cutoff_mode = cutoff_mode
                if mode_kwargs is not None and self._is_mpo_mode(self.mode):
                    method_cutoff_mode = mode_kwargs.get(
                        "mpo_cutoff_mode",
                        cutoff_mode,
                    )
                self._apply_submpo_with_method(
                    self.p,
                    submpo,
                    span,
                    method=method,
                    cutoff=cutoff,
                    cutoff_mode=method_cutoff_mode,
                    info=self.info_c,
                )
                projected_norm = self._real_float(
                    ar.do("abs", self.p.norm())
                )
        else:
            op = self._pauli_operator(pauli, where)
            dim = op.shape[0]
            projector = 0.5 * (np.eye(dim, dtype=complex) + m * op)
            self._apply_dense_operator(
                self.p,
                projector,
                where,
                max_bond=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            projected_norm = self._real_float(ar.do("abs", self.p.norm()))
        self._record_norm_event(
            norm_kind,
            # ``prob`` is the physical branch factor, while ``projected_norm``
            # is the norm after the selected approximate compression route.
            # The norm event records both effects without counting the branch
            # probability as compression infidelity.
            expected_norm=input_norm * math.sqrt(float(prob)),
            observed_norm=projected_norm,
            where=where,
            branch_probability=prob,
            physical_boundary=True,
            renormalized=renormalize,
        )
        self._finish_measurement_center(
            anchor if collapse_center is None else collapse_center,
            renormalize=renormalize,
        )
        self.measurements.append(
            (str(pauli), tuple(int(site) for site in record_where), int(m), float(prob))
        )
        return m

    def _apply_basis_flip(self, q, axis, *, cutoff, cutoff_mode):
        """Flip the ``-axis`` eigenstate at site ``q`` to the ``+axis`` eigenstate."""
        flip_axis = _RESET_FLIP_AXES[axis]
        self._apply_dense_operator(
            self.p,
            _PAULI_1Q[flip_axis],
            (q,),
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        # A single-site gate at the centre keeps the centre at q.
        self.info_c["cur_orthog"] = (q, q)

    def _apply_reset_event(self, where, axes=None, *, cutoff, cutoff_mode):
        """Reset each qubit in ``where`` to the requested + Pauli eigenstate."""
        if axes is None:
            axes = ("Z",) * len(where)
        for site, axis in zip(where, axes):
            q = int(site)
            exp = self._state_expectation(axis, (q,))
            p_plus = min(max(0.5 * (1.0 + exp), 0.0), 1.0)
            m = 1 if self._rng.random() < p_plus else -1
            projector = 0.5 * (
                np.eye(2, dtype=complex) + m * _PAULI_1Q[axis]
            )
            # Centre at q, collapse, renormalize, and (if needed) flip |1> -> |0>,
            # keeping the tracked centre at q throughout.
            self.canonize_mps(self.p, q)
            input_norm = self._real_float(ar.do("abs", self.p.norm()))
            self._apply_dense_operator(
                self.p,
                projector,
                (q,),
                max_bond=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            projected_norm = self._real_float(ar.do("abs", self.p.norm()))
            branch_probability = p_plus if m > 0 else 1.0 - p_plus
            self._record_norm_event(
                "reset",
                expected_norm=input_norm * math.sqrt(float(branch_probability)),
                observed_norm=projected_norm,
                where=(q,),
                branch_probability=branch_probability,
                physical_boundary=True,
                renormalized=True,
            )
            self._recanonize_center(q, renormalize=True)
            if m < 0:
                self._apply_basis_flip(
                    q, axis, cutoff=cutoff, cutoff_mode=cutoff_mode
                )
        return self.p

    def _apply_measure_reset_event(  # pylint: disable=too-many-arguments
        self,
        axes,
        where,
        outcomes,
        *,
        record_where,
        cutoff,
        cutoff_mode,
        mode_kwargs=None,
    ):
        """Measure each target, record it, then reset it to the + Pauli eigenstate."""
        record_sites = tuple(int(site) for site in record_where)
        for axis, site, record_site, outcome in zip(
            axes, where, record_sites, outcomes
        ):
            q = int(site)
            m = self._apply_measure_event(
                axis,
                (q,),
                outcome,
                record_where=(record_site,),
                renormalize=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                norm_kind="measure_reset",
                mode_kwargs=mode_kwargs,
            )
            if m < 0:
                self._apply_basis_flip(
                    q, axis, cutoff=cutoff, cutoff_mode=cutoff_mode
                )
        return self.p

    def _apply_cap_event(self, where, vec, absorb):
        """Contract site ``where``'s physical index with ``vec`` and shorten the MPS."""
        (q,) = (int(site) for site in where)
        p = self.p
        L = int(p.L)
        if not 0 <= q < L:
            raise ValueError(
                f"cap site {q} is outside the MPS range [0, {L})."
            )
        if L <= 1:
            raise ValueError("cannot cap the only site of a length-1 MPS.")

        # Cap vectors are state contractions rather than operator payloads.
        # The stream parser historically normalized them to complex dtype,
        # which made a real Torch MPS fail at the contraction boundary even
        # for the ordinary real vectors used to sum or project a binary leg.
        # Preserve genuinely complex caps, but discard an exactly-zero
        # imaginary part so the vector follows the live MPS backend/dtype.
        vec_arr = np.asarray(vec).ravel()
        if np.iscomplexobj(vec_arr) and np.all(np.imag(vec_arr) == 0):
            vec_arr = np.asarray(np.real(vec_arr))
        phys_ind = p.site_ind(q)
        phys_dim = p.ind_size(phys_ind)
        if vec_arr.shape[0] != phys_dim:
            raise ValueError(
                f"cap vector length {vec_arr.shape[0]} does not match the "
                f"physical dimension {phys_dim} of site {q}."
            )

        site_ind_id = p.site_ind_id
        site_tag_id = p.site_tag_id
        if absorb == "left":
            neighbour = q - 1 if q > 0 else q + 1
        else:
            neighbour = q + 1 if q < L - 1 else q - 1

        # Move the orthogonality centre onto the absorbing neighbour first: the
        # capped site is then an isometry adjacent to the centre, so merging it
        # in leaves the centre exactly on the (renumbered) neighbour. This keeps
        # the tracked centre exact without any rescan.
        self.canonize_mps(p, neighbour)
        new_center = neighbour if neighbour < q else neighbour - 1

        cap_tensor = qtn.Tensor(self._to_state_backend(vec_arr), inds=(phys_ind,))
        site_tensor = p[p.site_tag(q)]
        neighbour_tensor = p[p.site_tag(neighbour)]
        merged = qtn.tensor_contract(site_tensor, cap_tensor, neighbour_tensor)

        p.delete(p.site_tag(q))
        p.delete(p.site_tag(neighbour))
        merged.modify(tags=(p.site_tag(neighbour),))
        p |= merged

        # Renumber every site above the removed one down by one position.
        temp_reindex = {}
        temp_retag = {}
        for old in range(q + 1, L):
            temp_reindex[site_ind_id.format(old)] = f"__pepsy_cap_k{old - 1}"
            temp_retag[site_tag_id.format(old)] = f"__pepsy_cap_I{old - 1}"
        if temp_reindex:
            p.reindex_(temp_reindex)
        if temp_retag:
            p.retag_(temp_retag)
        final_reindex = {
            f"__pepsy_cap_k{i}": site_ind_id.format(i) for i in range(q, L - 1)
        }
        final_retag = {
            f"__pepsy_cap_I{i}": site_tag_id.format(i) for i in range(q, L - 1)
        }
        if final_reindex:
            p.reindex_(final_reindex)
        if final_retag:
            p.retag_(final_retag)

        capped = p.view_as_(
            qtn.MatrixProductState,
            L=L - 1,
            cyclic=False,
            site_ind_id=site_ind_id,
            site_tag_id=site_tag_id,
        )
        self.p = self._install_represented_norm(capped)
        self.info_c["cur_orthog"] = (new_center, new_center)
        self._mps_length_history.append(int(self.p.L))
        self.cap_history.append(
            {
                "physical_site": int(q),
                "old_length": int(L),
                "new_length": int(self.p.L),
                "absorb": str(absorb),
            }
        )
        return self.p

    def _validate_event_stream_for_run(self, G_seq, where_seq, event_seq):
        """Validate queued event metadata before replay."""
        if not (len(G_seq) == len(where_seq) == len(event_seq)):
            raise ValueError(
                "MpsOptimizer event stream metadata is inconsistent: "
                "payloads, wheres, and event types must have the same length."
            )

        unknown = sorted(set(event_seq) - {"gate", "submpo"} - _CONTROL_EVENT_NAMES)
        if unknown:
            raise ValueError(f"Unknown MPS stream event type(s): {unknown!r}.")

        has_submpo = any(event_type == "submpo" for event_type in event_seq)
        if has_submpo and not (
            self._is_mpo_mode(self.mode) or self.mode == "dmrg"
        ):
            raise ValueError(
                "subMPO stream events require an MPO or DMRG mode."
            )

        has_cap = any(event_type == "cap" for event_type in event_seq)
        if not has_submpo:
            return

        # ``cap`` events shorten the MPS mid-stream, so a static site-range
        # check against the initial length is unreliable; those events are
        # validated dynamically as they are applied.
        L = int(getattr(self.p, "L", 0))
        for step, (where, event_type) in enumerate(
            zip(where_seq, event_seq),
            start=1,
        ):
            if event_type != "submpo":
                continue
            if len(set(where)) != len(where):
                raise ValueError(
                    f"subMPO event at step {step} has repeated site(s): {where!r}."
                )
            if has_cap:
                continue
            out_of_range = [site for site in where if site < 0 or site >= L]
            if out_of_range:
                raise ValueError(
                    f"subMPO event at step {step} references site(s) outside "
                    f"the MPS range [0, {L}): {out_of_range!r}."
                )

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

    def _start_unitary_norm_tracking(self, p):
        """Initialize scalar working-norm tracking for a unitary stream."""
        if self._unitary_previous_norm is not None:
            return
        # The live MPS already has a tracked orthogonality span. Move its
        # right edge to a one-site centre and read the raw centre norm instead
        # of contracting the full doubled MPS network once at stream start.
        current_span = self._current_orthog(p)
        current_norm = self._real_float(
            self._canonical_span_norm(p, current_span)
        )
        self._unitary_previous_norm = current_norm

    def _invalidate_unitary_norm_baseline(self):
        """Forget raw-norm scalars after an out-of-stream state rescaling.

        The next unitary compressed run establishes a fresh raw center norm
        before applying a gate.
        """
        self._unitary_previous_norm = None

    @staticmethod
    def _fidelity_ratio_from_norms(observed_norm, expected_norm, *, finite_check=False):
        """Return raw and clipped fidelity measured from two norms."""
        observed_norm = float(abs(observed_norm))
        expected_norm = float(abs(expected_norm))
        if (
            expected_norm <= 0.0
            or observed_norm < 0.0
            or (finite_check and (
                not np.isfinite(expected_norm) or not np.isfinite(observed_norm)
            ))
        ):
            return None, None
        raw = (observed_norm / expected_norm) ** 2
        return raw, min(max(raw, 0.0), 1.0)

    def _unitary_norm_overshoot_tolerance(self):
        """Return the dtype-aware tolerance for small norm overshoots.

        The norm ratio is evaluated from the retained canonical-center tensor.
        For ``float32``/``complex64`` data, the SVD and canonicalization
        roundoff can accumulate over a gate stream even when the projection is
        otherwise healthy. Keep the historical tolerance for higher precision,
        while allowing a bounded multiple of float32 machine epsilon for the
        low-precision path. The raw ratio is still retained in the event and
        the fidelity contribution remains clipped at one.
        """
        dtype = str(self.backend_dtype).lower()
        if "32" in dtype or "complex64" in dtype:
            return max(1.0e-6, 128.0 * np.finfo(np.float32).eps)
        return 1.0e-6

    def _record_norm_event(
        self,
        kind,
        *,
        expected_norm,
        observed_norm,
        where=(),
        branch_probability=None,
        physical_boundary=False,
        renormalized=None,
    ):
        """Record automatic norm survival without treating physical loss as error.

        ``expected_norm`` is the norm of the exact physical target before
        compression. For a unitary update it is the pre-compression norm; for
        a Kraus/projective branch it includes the branch's Born probability.
        Only the observed/expected norm ratio contributes to the cumulative
        compression survival product.
        """
        raw, survival = self._fidelity_ratio_from_norms(
            observed_norm, expected_norm, finite_check=self._finite_check_enabled
        )
        if (
            self._finite_check_enabled
            and kind == "unitary_compression"
            and raw is not None
        ):
            overshoot_tolerance = self._unitary_norm_overshoot_tolerance()
            if raw > 1.0 + overshoot_tolerance:
                raise FloatingPointError(
                    "Retained unitary-compression norm exceeds its expected norm "
                    f"(squared ratio={raw:.6g}, "
                    f"tolerance={overshoot_tolerance:.3g}); "
                    "canonical projection metadata is inconsistent."
                )
        event = {
            "kind": str(kind),
            "where": tuple(int(site) for site in where),
            "valid": raw is not None,
            "expected_norm": None if raw is None else float(abs(expected_norm)),
            "expected_norm_sq": None if raw is None else float(abs(expected_norm) ** 2),
            "observed_norm": None if raw is None else float(abs(observed_norm)),
            "observed_norm_sq": None if raw is None else float(abs(observed_norm) ** 2),
            "fidelity_raw": None if raw is None else float(raw),
            # These are fidelity/infidelity values measured from norms. The
            # metric name intentionally does not repeat its measurement source.
            "local_fidelity": (
                None if survival is None else float(survival)
            ),
            "local_infidelity": (
                None if survival is None else float(1.0 - survival)
            ),
            "branch_probability": (
                None
                if branch_probability is None
                else float(branch_probability)
            ),
            "physical_boundary": bool(physical_boundary),
            "renormalized": (
                None if renormalized is None else bool(renormalized)
            ),
        }
        if survival is not None:
            if survival == 0.0:
                self._norm_log_survival = -np.inf
            elif self._norm_log_survival != -np.inf:
                self._norm_log_survival += math.log(survival)
            cumulative = (
                0.0
                if self._norm_log_survival == -np.inf
                else float(math.exp(self._norm_log_survival))
            )
            cumulative_infidelity = (
                1.0
                if self._norm_log_survival == -np.inf
                else float(-math.expm1(self._norm_log_survival))
            )
            event["cumulative_fidelity"] = cumulative
            event["cumulative_infidelity"] = cumulative_infidelity
            event["cumulative_compression_fidelity"] = cumulative
            event["cumulative_compression_infidelity"] = cumulative_infidelity
        else:
            event["cumulative_fidelity"] = None
            event["cumulative_infidelity"] = None
            event["cumulative_compression_fidelity"] = None
            event["cumulative_compression_infidelity"] = None
        self.norm_events.append(event)
        if physical_boundary:
            self._invalidate_unitary_norm_baseline()
        return event

    def norm_diagnostics(self):
        """Return automatic norm-based compression diagnostics.

        ``local_fidelity`` and ``cumulative_fidelity`` are fidelities measured
        from retained canonical-centre norms. They are compression-survival
        proxies, not directional overlaps with an independently supplied
        target state. DMRG target overlap, when available, is reported
        separately by :meth:`get_fit_diagnostics`.
        Born probabilities for stochastic branches remain in ``norm_events``
        and do not reduce cumulative compression fidelity.

        ``state_norm`` and ``norm`` are the live represented MPS norm.
        ``cumulative_norm`` is instead the square root of
        ``cumulative_fidelity``. The latter is a retained-compression proxy,
        not a second reading of the live state norm.
        """
        valid = [event for event in self.norm_events if event.get("valid")]
        physical = [
            event for event in valid if event.get("physical_boundary")
        ]
        if not valid:
            survival = None
            infidelity = None
        elif self._norm_log_survival == -np.inf:
            survival = 0.0
            infidelity = 1.0
        else:
            survival = float(math.exp(self._norm_log_survival))
            infidelity = float(-math.expm1(self._norm_log_survival))
        current = valid[-1] if valid else None
        state_norm = self._real_float(ar.do("abs", self.p.norm()))
        event_survivals = [float(event["local_fidelity"]) for event in valid]
        event_infidelities = [
            float(event["local_infidelity"]) for event in valid
        ]
        if event_survivals and any(value <= 0.0 for value in event_survivals):
            geometric_survival = 0.0
        elif event_survivals:
            geometric_survival = float(
                math.exp(sum(math.log(value) for value in event_survivals)
                         / len(event_survivals))
            )
        else:
            geometric_survival = None
        return {
            "tracking": True,
            "norm_tracking": True,
            # MpsOptimizer does not maintain Tree-style per-edge spectrum
            # probes; its canonical path ledger is the available diagnostic.
            "truncation_tracking": None,
            "current_valid": current is not None,
            "events": len(self.norm_events),
            "completed_events": len(valid),
            "completed_segments": len(valid),
            "segments_including_current": len(valid),
            "completed_segment_norms": [
                float(max(0.0, value) ** 0.5) for value in event_survivals
            ],
            "completed_segment_infidelities": event_infidelities,
            # Provenance alias: this is the cumulative fidelity obtained from
            # norm survival, not the live state norm below.
            "norm_survival": survival,
            "local_fidelity": (
                None if current is None else current.get("local_fidelity")
            ),
            "local_infidelity": (
                None if current is None else current.get("local_infidelity")
            ),
            "cumulative_fidelity": survival,
            "cumulative_infidelity": infidelity,
            # Explicit compression aliases retained for callers that want to
            # emphasize what the cumulative fidelity measures.
            "cumulative_compression_fidelity": survival,
            "cumulative_compression_infidelity": infidelity,
            "fidelity": survival,
            "infidelity": infidelity,
            # ``norm`` is the represented live MPS norm. The retained-norm
            # proxy is deliberately separate as ``cumulative_norm``.
            "norm": state_norm,
            "state_norm": state_norm,
            "cumulative_norm": (
                None if survival is None else float(survival**0.5)
            ),
            "total_survival_proxy": survival,
            "total_infidelity_proxy": infidelity,
            "total_norm_proxy": None if survival is None else float(survival**0.5),
            "geometric_mean_survival": geometric_survival,
            "geometric_mean_norm": (
                None
                if geometric_survival is None
                else float(geometric_survival**0.5)
            ),
            "mean_segment_infidelity": (
                None
                if not event_infidelities
                else float(sum(event_infidelities) / len(event_infidelities))
            ),
            "max_segment_infidelity": (
                None if not event_infidelities else float(max(event_infidelities))
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
            "current_fidelity": (
                None if current is None else current["local_fidelity"]
            ),
            "current_infidelity": (
                None if current is None else current["local_infidelity"]
            ),
            "physical_boundary_events": len(physical),
            "physical_boundary_infidelities": [
                event["local_infidelity"]
                for event in physical
            ],
            "completed_projector_infidelities": [
                event["local_infidelity"] for event in physical
            ],
            "completed_nonunitary_infidelities": [
                event["local_infidelity"] for event in physical
            ],
            "completed_combined_infidelities": event_infidelities,
        }

    @staticmethod
    def _accumulate_exponent(p, scale):
        """Accumulate an extracted multiplicative ``scale`` into ``p.exponent``."""
        if hasattr(p, "exponent"):
            p.exponent = p.exponent + ar.do("log10", ar.do("abs", scale))

    @staticmethod
    def _class_norm_includes_exponent(p):
        """Return whether the installed quimb ``norm`` already uses exponent."""
        if not hasattr(p, "exponent"):
            return False

        exponent_orig = p.exponent
        try:
            p.exponent = 0.0
            norm0 = type(p).norm(p)
            p.exponent = 1.0
            norm1 = type(p).norm(p)
        except Exception:
            return False
        finally:
            p.exponent = exponent_orig

        denom = ar.do("abs", norm0)
        try:
            if MpsOptimizer._real_float(denom) == 0.0:
                return False
            ratio = MpsOptimizer._real_float(ar.do("abs", norm1) / denom)
        except Exception:
            return False
        return abs(ratio - 10.0) < 1.0e-8

    @staticmethod
    def _install_represented_norm(p):
        """Make ``p.norm()`` include PEPSY's accumulated base-10 exponent.

        Some quimb versions apply ``TensorNetwork.exponent`` in MPS ``norm``
        already, while others ignore it. PEPSY uses exponent to keep
        non-unitary working data normalized while preserving the represented
        state scale, so optimizer-managed states get a small instance-local
        wrapper only when the installed quimb needs one.
        """
        if (
            (not hasattr(p, "norm"))
            or (not hasattr(p, "exponent"))
            or getattr(p, "_pepsy_norm_includes_exponent", False)
        ):
            return p

        norm_cache_key = type(p)
        norm_includes_exponent = _NORM_INCLUDES_EXPONENT_CACHE.get(
            norm_cache_key,
            _MISSING,
        )
        if norm_includes_exponent is _MISSING:
            norm_includes_exponent = MpsOptimizer._class_norm_includes_exponent(p)
            _NORM_INCLUDES_EXPONENT_CACHE[norm_cache_key] = norm_includes_exponent

        if norm_includes_exponent:
            p._pepsy_norm_includes_exponent = True
            return p

        def _norm_with_exponent(self, output_inds=None, squared=False, **contract_opts):
            raw_norm = type(self).norm(
                self,
                output_inds=output_inds,
                squared=squared,
                **contract_opts,
            )
            exponent = getattr(self, "exponent", 0.0)
            if exponent == 0:
                return raw_norm
            scale_power = 2 * exponent if squared else exponent
            return raw_norm * (10**scale_power)

        p.norm = types.MethodType(_norm_with_exponent, p)
        p._pepsy_norm_includes_exponent = True
        return p

    @staticmethod
    def _normalize_span(where):
        """Return ``(xmin, xmax)`` for an int, singleton, or two-site span."""
        if isinstance(where, Integral):
            site = int(where)
            return site, site
        if len(where) == 1:
            site = int(where[0])
            return site, site
        if len(where) == 2:
            site0, site1 = int(where[0]), int(where[1])
            return min(site0, site1), max(site0, site1)
        raise ValueError("where must be an int, (int,), or (int, int).")

    def _canonical_span_norm(self, p, where, *, fallback=True):
        """Return the raw norm from a single-site orthogonality center.

        The active span is deliberately canonicalized to one site rather than
        contracted as an open multi-site block. Once the MPS is mixed
        canonical around that site, the center tensor's Frobenius norm is the
        represented norm of the raw working data and does not include
        ``p.exponent``. ``p`` can be a target copy, so cached optimizer metadata
        is used as a hint but is never updated for copies.
        """
        requested_span = self._normalize_span(where)
        state_info = self._info_for_state(p)
        cached = state_info.get("cur_orthog", "calc")
        if cached in ("calc", None):
            if fallback:
                current_span = requested_span
            else:
                current_span = self._normalize_span(p.calc_current_orthog_center())
        else:
            current_span = self._normalize_span(cached)

        # A gate can enlarge the non-canonical region from the previous center
        # to its support. Treat that union as the known current span, allowing
        # Quimb to move either boundary without a center rescan.
        current_span = (
            min(current_span[0], requested_span[0]),
            max(current_span[1], requested_span[1]),
        )
        center = int(requested_span[1])
        if current_span != (center, center):
            p.canonize(
                [center],
                cur_orthog=current_span,
                info=state_info,
            )

        state_info["cur_orthog"] = (center, center)
        return p[center].norm()

    def _retained_center_norm_impl(self, p, where):
        """Return ``(norm, center)`` from the cheapest valid MPS center.

        Quimb's compressed gate paths normally leave a one-site
        orthogonality center and record it in ``info_c``. Its tensor norm is
        already the complete raw MPS norm, so moving that center to the edge
        of the gate span would be redundant. Only collapse a genuinely broad
        cached span, for which no single center tensor is yet authoritative.
        """
        current_span = self._current_orthog(p)
        if current_span[0] == current_span[1]:
            center = int(current_span[0])
            return p[center].norm(), center

        norm = self._canonical_span_norm(p, where)
        return norm, int(self._normalize_span(where)[1])

    def _retained_center_norm(self, p, where):
        """Measure a retained center norm with opt-in timing only."""
        if self._timing_state is None:
            return self._retained_center_norm_impl(p, where)
        return self._timed_call(
            "stabilize.norm",
            self._retained_center_norm_impl,
            p,
            where,
        )

    def _build_norm_target(
        self,
        p,
        gate,
        where,
        cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="mps",
        copy=True,
        info=None,
    ):
        """Build an exact pre-output-compression target.

        ``layered`` stores dense gates as small lazy tensors rather than
        repeatedly SVD-compressing an ever-growing target MPS. ``mps`` keeps
        the legacy routed target and remains the native-safe Symmray route.
        """
        target_strategy = self._validate_fit_target_strategy(target_strategy)
        if target_strategy == "auto":
            target_strategy = (
                "mps"
                if self._has_symmray_data(p) or p.isfermionic()
                else "layered"
            )

        p_target = p.copy() if copy else p
        target_info = {} if info is None else info
        if len(where) == 1:
            self._apply_layered_target_gate(
                p_target,
                gate,
                where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            return p_target

        if target_strategy == "layered":
            return self._apply_layered_target_gate(
                p_target,
                gate,
                where,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )

        if self._has_symmray_data(p_target):
            # Keep one native tensor per MPS site. A lazy ``split-gate`` target
            # is useful for one-site contractions but leaves extra gate tensors
            # carrying overlapping site tags, which does not define a unique
            # two-site middle bond. Auto-swap uses Symmray's graded split path
            # and remains uncapped because no ``max_bond`` is supplied.
            return self._build_symmray_auto_swap_target(
                p_target,
                gate,
                where,
                cutoff,
                cutoff_mode,
                copy=False,
                info=target_info,
            )

        p_target.gate_nonlocal_(
            gate,
            where,
            dims=self._infer_gate_dims(gate, where),
            max_bond=None,
            info=target_info,
            method="direct",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        return p_target

    @staticmethod
    def _build_lazy_submpo_target(p, submpo, where, *, copy=True):
        """Build an exact lazy target by attaching a sub-MPO to ``p``."""
        p_target = p.copy() if copy else p
        p_target.gate_with_submpo_(
            submpo,
            where=where,
            method="lazy",
            inplace_mpo=False,
        )
        return p_target

    def _fit_window_copy_supported(self, p):
        """Inspect array capabilities once before sharing exterior data."""
        if self._has_symmray_data(p) or p.isfermionic():
            return False
        return all(
            ar.infer_backend(t.data) in {"numpy", "torch", "jax"}
            for t in p.tensors
        )

    def _copy_fit_window_state(self, p, where):
        """Copy active FIT data, retaining read-only exterior arrays.

        Quimb copies tensor metadata independently and canonicalization
        replaces arrays rather than writing through them. FIT and the local
        compressor update only the endpoint span. Own every array in that
        span so an in-place failure cannot corrupt the source or rollback.
        Native/unknown array types retain the conservative full deep copy.
        """
        cache = self._fit_copy_policy_cache
        if cache is None:
            supported = self._fit_window_copy_supported(p)
        else:
            key = (type(p), type(p[0].data))
            if key not in cache:
                cache[key] = self._fit_window_copy_supported(p)
            supported = cache[key]
        if not supported:
            return p.copy(deep=True)
        start, stop = min(where), max(where)
        copied = p.copy()
        for site in range(start, stop + 1):
            tensor = copied[site]
            # Autoray's Torch ``copy`` detaches the autograd graph. Clone
            # directly so a disposable FIT state retains parameter gradients.
            data = tensor.data
            data = data.clone() if ar.infer_backend(data) == "torch" else ar.do("copy", data)
            tensor.modify(
                data=data, left_inds=tensor.left_inds
            )
        return copied

    def _build_compression_fit_guess(
        self,
        p,
        gate,
        where,
        *,
        method,
        cutoff,
        cutoff_mode,
        seed=None,
    ):
        """Build a disposable Quimb-compressed guess from one gate."""
        return guess(
            self._copy_fit_window_state(p, where),
            gate,
            where,
            inplace=True,
            method=method,
            dims=self._infer_gate_dims(gate, where),
            chi=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            seed=seed,
        )

    def _build_compression_submpo_fit_guess(
        self,
        p,
        submpo,
        where,
        *,
        method,
        cutoff,
        cutoff_mode,
        seed=None,
    ):
        """Build a disposable compressed FIT guess from a sub-MPO."""
        guess_mps = self._copy_fit_window_state(p, where)
        self._apply_submpo_with_method(
            guess_mps,
            submpo,
            where,
            method=method,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            info={},
            seed=seed,
        )
        return guess_mps

    def _build_compression_batch_fit_guess(
        self,
        p,
        gates,
        wheres,
        *,
        method,
        cutoff,
        cutoff_mode,
        seed=None,
    ):
        """Build a disposable Quimb-compressed guess for a gate batch."""
        guess_mps = self._copy_fit_window_state(
            p, tuple(site for where in wheres for site in where)
        )
        for i, (gate, where) in enumerate(zip(gates, wheres)):
            guess(
                guess_mps,
                gate,
                where,
                method=method,
                dims=self._infer_gate_dims(gate, where),
                chi=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                seed=None if seed is None else int(seed) + i,
                inplace=True,
            )
        return guess_mps

    def _native_src_fit_guess_enabled(
        self,
        strategy,
        fit_mpo_guess,
    ):
        """Return whether the native Symmray SRC-style guess is requested."""
        requested_strategy = self._validate_fit_init_strategy(strategy)
        if requested_strategy == "auto":
            requested_strategy = _DEFAULT_FIT_INIT_STRATEGY
        if requested_strategy != "guess_src":
            return False
        if (
            not fit_mpo_guess
            and self._dmrg_mode_alias in {"dmrg1", "dmrg3"}
            and str(strategy).strip().lower() in {"auto", _DEFAULT_FIT_INIT_STRATEGY}
        ):
            return False
        return True

    def _build_native_randomized_fit_guess(
        self,
        p,
        gates,
        wheres,
        *,
        cutoff,
        cutoff_mode,
        seed,
    ):
        """Build a native Symmray SRC-style FIT guess.

        Quimb's dense SRC compressor cannot preserve Symmray charge sectors or
        fermionic dummy-mode metadata. Symmray does expose a native randomized
        truncated SVD, however, so apply the gate sequence on a disposable
        native MPS using ``svd:rand`` at every two-site split. This provides the
        same randomized compressed warm-start role without constructing a dense
        gate, MPO, or random dense tensor.
        """
        if not self._has_symmray_data(p):
            raise TypeError(
                "native randomized FIT guesses require native Symmray data."
            )

        guess_mps = p.copy(deep=True)
        guess_info = {}
        for index, (gate, where) in enumerate(zip(gates, wheres)):
            where = tuple(int(site) for site in where)
            if len(where) == 1:
                self._apply_gate(
                    guess_mps,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
            elif len(where) == 2:
                self._apply_symmray_auto_swap_gate(
                    guess_mps,
                    gate,
                    where,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    max_bond=self.chi,
                    info=guess_info,
                    method="svd:rand",
                    seed=int(seed) + index,
                )
            else:
                raise ValueError(
                    "Native randomized FIT guesses support one- or two-site "
                    "gates only."
                )

        return guess_mps, {
            "backend": "symmray",
            "method": "svd:rand",
            "seed": int(seed),
            "gate_count": len(gates),
        }

    @staticmethod
    def _fit_random_data(data, shape, *, strength, rng):
        """Generate deterministic random data on ``data``'s backend."""
        dtype_name = str(getattr(data, "dtype", "float64"))
        if "complex64" in dtype_name:
            random_dtype = np.complex64
        elif "complex" in dtype_name:
            random_dtype = np.complex128
        elif "float32" in dtype_name:
            random_dtype = np.float32
        else:
            random_dtype = np.float64
        return backend_random_array(
            shape,
            like=data,
            dtype=random_dtype,
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
        """Prepare a dense FIT guess with deterministic random initialization.

        The exact gate target remains built from the unmodified current MPS.
        ``expand=False`` perturbs only existing tensors, while ``expand=True``
        also adds random directions on active bonds below their physical/``chi``
        ceiling. Native Symmray and fermionic states retain their graded
        sector-growth path and are never padded with dense random data.
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
        if self._has_symmray_data(p) or p.isfermionic():
            info["reason"] = "native_sector_growth"
            return p, info
        if float(rand_strength) == 0.0:
            info["reason"] = "disabled"
            return p, info

        xmin, xmax = self._normalize_span(where)
        guess = p.copy(deep=True)
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

        guess_info = {}
        self.canonize_mps(guess, (xmin, xmax), info=guess_info)
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
        submpo=False,
        native_source=None,
    ):
        """Select the disposable FIT guess without changing the live MPS."""
        requested_strategy = self._validate_fit_init_strategy(strategy)
        info = {
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
            "native_randomized_guess_used": False,
            "random_initialization": info,
        }
        if self._has_symmray_data(p) or p.isfermionic():
            if (
                not submpo
                and native_source is not None
                and self._native_src_fit_guess_enabled(
                    requested_strategy,
                    fit_mpo_guess,
                )
            ):
                fit_guess, native_info = self._build_native_randomized_fit_guess(
                    native_source,
                    gates,
                    wheres,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    seed=seed,
                )
                info.update(native_info)
                info["reason"] = "native_src"
                result["fit_guess"] = fit_guess
                result["strategy"] = "guess_src"
                result["guess_method"] = "src"
                result["guess_used"] = True
                result["svd_guess_used"] = True
                result["guess_backend"] = "symmray-svd:rand"
                result["native_randomized_guess_used"] = True
                return result
            info["reason"] = (
                "native_sector_growth"
                if int(block_size) in {2, 3}
                else "native_one_site_fit"
            )
            result["strategy"] = "direct"
            return result

        start, stop = self._normalize_span(wheres[0] if len(wheres) == 1 else (
            min(site for where in wheres for site in where),
            max(site for where in wheres for site in where),
        ))
        needs_growth = requested_strategy in {"random", "random_expand"} and not (
            FIT._active_bonds_at_rank_targets(p, start, stop, self.chi)  # pylint: disable=protected-access
        )
        is_named_svd_window = len(gates) == 1 and self._dmrg_mode_alias in {
            "dmrg1",
            "dmrg3",
        }
        if requested_strategy == "auto":
            selected_strategy = _DEFAULT_FIT_INIT_STRATEGY
        else:
            # An explicit Quimb guess is a warm-start policy, not only a rank
            # enrichment policy. Keep applying it after the active bonds have
            # reached their attainable rank so the one-site FIT phase receives
            # the same SRC-prepared state as the expansion phase.
            selected_strategy = (
                requested_strategy
                if requested_strategy.startswith("guess_")
                or requested_strategy == "svd_guess"
                else requested_strategy if needs_growth else "direct"
            )
        if (
            not fit_mpo_guess
            and is_named_svd_window
            and requested_strategy in {"auto", _DEFAULT_FIT_INIT_STRATEGY}
        ):
            selected_strategy = "direct"

        if selected_strategy == "svd_guess":
            guess_method = "direct"
        elif selected_strategy.startswith("guess_"):
            guess_method = selected_strategy[len("guess_") :]
        else:
            guess_method = None

        if guess_method is not None:
            if len(gates) == 1:
                if submpo:
                    fit_guess = self._build_compression_submpo_fit_guess(
                        p,
                        gates[0],
                        wheres[0],
                        method=guess_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        seed=seed,
                    )
                else:
                    fit_guess = self._build_compression_fit_guess(
                        p,
                        gates[0],
                        wheres[0],
                        method=guess_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        seed=seed,
                    )
            else:
                fit_guess = self._build_compression_batch_fit_guess(
                    p,
                    gates,
                    wheres,
                    method=guess_method,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    seed=seed,
                )
            result["fit_guess"] = fit_guess
            result["strategy"] = "svd_guess"
            if selected_strategy != "svd_guess":
                result["strategy"] = selected_strategy
            result["guess_method"] = guess_method
            result["guess_used"] = True
            result["svd_guess_used"] = True
            info["reason"] = selected_strategy
            return result

        fit_guess, random_info = self._build_randomized_fit_guess(
            p,
            (start, stop),
            block_size=block_size,
            rand_strength=rand_strength,
            expand=selected_strategy == "random_expand",
            seed=int(seed),
        ) if selected_strategy in {"random", "random_expand"} else (p, info)
        if selected_strategy == "direct":
            info["reason"] = "already_at_target"
        result["fit_guess"] = fit_guess
        result["strategy"] = selected_strategy
        result["random_initialization"] = random_info
        return result

    @staticmethod
    def _normalize_every_interval(normalize_every, non_unitary=False):
        """Return whether non-unitary local scale control is enabled.

        Normalization is only meaningful for non-unitary streams. Callers
        validate explicit normalization requests before this helper is reached.
        """
        if not non_unitary:
            return None
        if normalize_every is None or normalize_every is False:
            return None
        if normalize_every is True:
            return True
        if not isinstance(normalize_every, Integral):
            raise TypeError("normalize_every must be a positive integer, bool, or None.")

        interval = int(normalize_every)
        if interval < 1:
            raise ValueError("normalize_every must be >= 1 when enabled.")
        return True

    @staticmethod
    def _accumulate_exponent_log10(p, log10_scale):
        """Accumulate an extracted base-10 log scale into ``p.exponent``."""
        if hasattr(p, "exponent"):
            p.exponent = p.exponent + log10_scale

    @staticmethod
    def _event_old_norm_from_log10(log10_old_norm):
        """Return a float old-norm value from its base-10 log when possible."""
        max_log10 = np.log10(np.finfo(float).max)
        if log10_old_norm > max_log10:
            return np.inf
        if log10_old_norm < -max_log10:
            return 0.0
        return float(10.0**log10_old_norm)

    def _normalize_orthog_tensors(
        self,
        p,
        where,
        *,
        step,
        reason,
        canonicalize=False,
    ):
        """Compatibility wrapper for the one-site center normalizer."""
        _ = canonicalize
        return self._normalize_canonical_center(
            p,
            where,
            step=step,
            reason=reason,
        )

    def _normalize_in_canonical_range(self, p, where, *, step, eps=1e-15):
        """Canonicalize ``where`` and apply one-site scale control."""
        _ = eps
        return self._normalize_canonical_center(
            p,
            where,
            step=step,
            reason="final",
        )

    def _normalize_canonical_center(self, p, where, *, step, reason):
        """Normalize a center and optionally accumulate normalization time."""
        if self._timing_state is None:
            return self._normalize_canonical_center_impl(
                p,
                where,
                step=step,
                reason=reason,
            )
        return self._timed_call(
            "normalization",
            self._normalize_canonical_center_impl,
            p,
            where,
            step=step,
            reason=reason,
        )

    def _normalize_canonical_center_impl(self, p, where, *, step, reason):
        """Normalize one canonical center and preserve its scale in exponent.

        Reuse a tracked singleton center whenever it lies inside ``where``.
        Its Frobenius norm already equals the raw working-MPS norm, so moving
        it to a fixed endpoint would add a redundant QR sweep. A genuinely
        broad center is collapsed to the right edge before normalization.
        """
        span = self._normalize_span(where)
        current_span = self._current_orthog(p)
        if (
            current_span[0] == current_span[1]
            and span[0] <= current_span[0] <= span[1]
        ):
            center = int(current_span[0])
            scale = p[center].norm()
        else:
            scale = self._canonical_span_norm(p, span)
            center = int(span[1])
        scale_float = self._real_float(ar.do("abs", scale))
        if scale_float == 0.0 or (
            self._finite_check_enabled and not np.isfinite(scale_float)
        ):
            return None

        p[center].modify(data=p[center].data / scale)
        log10_scale = self._real_float(ar.do("log10", ar.do("abs", scale)))
        self._accumulate_exponent_log10(p, log10_scale)
        self._record_orthog_span(p, (center, center))

        event = {
            "step": int(step),
            "old_norm": self._event_old_norm_from_log10(2.0 * log10_scale),
            "span": span,
            "insert": center,
            "sites": (center,),
            "scales": (scale_float,),
            "log10_scale": log10_scale,
            "log10_scales": (log10_scale,),
            "reason": str(reason),
            "method": "canonical_center",
            "exponent": self._real_float(getattr(p, "exponent", 0.0)),
        }
        self.normalizations.append(event)
        return event

    def _maybe_normalize_after_step(
        self,
        p,
        *,
        step,
        where,
        normalize_every,
        reason,
    ):
        """Apply one-site scale control after an enabled replay step."""
        if normalize_every is None:
            return None
        return self._normalize_canonical_center(
            p,
            where,
            step=step,
            reason=reason,
        )

    def _maybe_normalize_final(
        self,
        p,
        *,
        step,
        last_normalized_step,
        where,
        normalize_every,
        normalize_final,
        normalize_eps,
    ):
        """Optionally normalize at run end if local scale control was active."""
        if (
            normalize_every is not None
            and normalize_final
            and step > 0
            and last_normalized_step != step
        ):
            return self._normalize_in_canonical_range(
                p,
                where,
                step=step,
                eps=normalize_eps,
            )
        return None

    @staticmethod
    def _format_progress_scalar(value):
        """Format displayed progress scalar with stable precision."""
        return f"{MpsOptimizer._real_float(value):.6f}"

    def _cumulative_fidelity(self):
        """Return displayed cumulative fidelity measured from norms.

        This is not the live MPS norm and is not a target-state overlap. It is
        the product of the local squared canonical-centre norm-survival ratios
        accumulated in ``_norm_log_survival``.
        """
        if self._norm_log_survival == -np.inf:
            return 0.0
        return float(math.exp(self._norm_log_survival))

    @staticmethod
    def _collect_dmrg_batch(
        G_seq,
        where_seq,
        start_idx,
        k_2q_batch,
        *,
        max_span=None,
    ):
        """Collect a DMRG batch with an optional spatial-span cap."""
        batch_G = []
        batch_where = []
        two_qubit_in_batch = 0
        idx = start_idx

        while idx < len(G_seq) and two_qubit_in_batch < k_2q_batch:
            where = where_seq[idx]
            gate = G_seq[idx]
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
        target_cutoff,
        cutoff_mode="rsum2",
        *,
        target_strategy="mps",
    ):
        """Apply a DMRG target block without output-compression truncation."""
        p_g = p.copy()
        for gate, where in zip(batch_G, batch_where):
            if len(where) == 1:
                self._apply_layered_target_gate(
                    p_g,
                    gate,
                    where,
                    cutoff=target_cutoff,
                    cutoff_mode=cutoff_mode,
                )
            else:
                p_g = self._build_norm_target(
                    p_g,
                    gate,
                    where,
                    target_cutoff,
                    cutoff_mode,
                    target_strategy=target_strategy,
                    copy=False,
                )
        return p_g

    def _stabilize_unitary_compression_state(
        self,
        p,
        where,
        target_norm,
        *,
        current_norm=None,
        center_site=None,
        restore=True,
    ):
        """Record compression norm survival and optionally restore its scale.

        The removed scale is deliberately *not* accumulated into ``exponent``
        for unitary evolution: it is approximation loss, not physical
        non-unitary evolution. ``restore=False`` preserves the historical
        un-stabilized output while still recording the norm change.
        """
        span = self._normalize_span(where)
        if current_norm is None or center_site is None:
            current_norm = self._canonical_span_norm(p, span)
            center = int(span[1])
        else:
            center = int(center_site)
            if not span[0] <= center <= span[1]:
                raise ValueError(
                    f"FIT center {center} is outside active span {span}."
                )
        current_float = self._real_float(ar.do("abs", current_norm))
        target_float = self._real_float(ar.do("abs", target_norm))
        if (
            current_float == 0.0
            or target_float == 0.0
            or (self._finite_check_enabled and (
                not np.isfinite(current_float) or not np.isfinite(target_float)
            ))
        ):
            raise FloatingPointError(
                "Cannot stabilize a unitary FIT state with a zero or non-finite norm."
            )
        self._record_norm_event(
            "unitary_compression",
            expected_norm=target_float,
            observed_norm=current_float,
            where=span,
        )
        if restore:
            p[center].modify(data=p[center].data * (target_norm / current_norm))
        else:
            self._unitary_previous_norm = current_float
        self._record_orthog_span(p, (center, center))
        if restore:
            self._unitary_previous_norm = target_float

    def _stabilize_unitary_fit_state(self, *args, **kwargs):
        """Compatibility wrapper for the generalized compression stabilizer."""
        return self._stabilize_unitary_compression_state(*args, **kwargs)

    def _run_mix_mpo_step(
        self,
        gate,
        where,
        event_type,
        *,
        step,
        cutoff,
        cutoff_mode,
        submpo_method,
        compression_seed=None,
        stabilize_unitary,
    ):
        """Apply one mixed-mode step through the MPO backend."""
        self._run_mpo(
            [gate],
            [where],
            [event_type],
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=None,
            normalize_final=False,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            stabilize_unitary=stabilize_unitary,
        )

    def _run_mix_mpo_batch(
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        steps,
        cutoff,
        cutoff_mode,
        submpo_method,
        compression_seed=None,
        stabilize_unitary,
    ):
        """Apply a mixed-mode fallback batch through the MPO backend."""
        self._run_mpo(
            G_seq,
            where_seq,
            event_seq,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=None,
            normalize_final=False,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            stabilize_unitary=stabilize_unitary,
        )
        active_where = (
            min(site for where in where_seq for site in where),
            max(site for where in where_seq for site in where),
        )
        self._validate_mix_norm(active_where, operation="MPO batch")

    def _run_mix_dmrg(self, *args, fit_block_size, **kwargs):
        """Run mixed DMRG with a stable named FIT schedule when available.

        Mixed mode may enter DMRG after an MPO warm-up with a two- or
        three-site FIT window.  Keep that transaction on the corresponding
        fixed schedule: the generic rank-adaptive schedule can leave a
        long-range window with a stale represented norm after a short sweep
        budget, which the unitary invariant must (correctly) reject.
        """
        requested_block_size = int(fit_block_size)
        schedule_alias = {
            1: ("dmrg1", 1),
            2: ("dmrg2", 2),
            3: ("dmrg3", 3),
        }.get(requested_block_size)
        old_dmrg_alias = self._dmrg_mode_alias
        old_dmrg_block_size = self._dmrg_mode_block_size
        if schedule_alias is not None:
            self._dmrg_mode_alias, fit_block_size = schedule_alias
            self._dmrg_mode_block_size = fit_block_size
        kwargs["fit_block_size"] = fit_block_size
        try:
            return self._run_dmrg(*args, **kwargs)
        finally:
            self._dmrg_mode_alias = old_dmrg_alias
            self._dmrg_mode_block_size = old_dmrg_block_size

    def _run_mix_dmrg_step(
        self,
        gate,
        where,
        *,
        step,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        cutoff,
        cutoff_mode,
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
    ):
        """Apply one mixed-mode step through the DMRG backend."""
        self._run_mix_dmrg(
            [gate],
            [where],
            n_iter=n_iter,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=1,
            normalize_every=None,
            normalize_final=False,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_finite_check=finite_check,
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=stabilize_unitary,
        )
        self._validate_mix_norm(where, operation="DMRG step")

    def _run_mix_dmrg_batch(
        self,
        G_seq,
        where_seq,
        *,
        steps,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        cutoff,
        cutoff_mode,
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
    ):
        """Apply a contiguous two-site batch through the DMRG backend."""
        self._run_mix_dmrg(
            G_seq,
            where_seq,
            n_iter=n_iter,
            progbar=False,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=len(G_seq),
            normalize_every=None,
            normalize_final=False,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_finite_check=finite_check,
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=stabilize_unitary,
        )
        active_where = (
            min(site for where in where_seq for site in where),
            max(site for where in where_seq for site in where),
        )
        self._validate_mix_norm(active_where, operation="DMRG batch")
        if self._effective_max_bond(self.p) > int(self.chi):
            raise RuntimeError(
                "DMRG batch exceeded the mixed-mode chi bond limit."
            )

    def _collect_mix_dmrg_batch(
        self,
        G_seq,
        where_seq,
        start_idx,
        k_2q_batch,
        *,
        target_sizes=None,
        allow_short=False,
        max_span=None,
    ):
        """Collect contiguous DMRG-ready gates for one mixed transaction."""
        batch_G = []
        batch_where = []
        idx = start_idx
        while (
            idx < len(G_seq)
            and len(batch_G) < int(k_2q_batch)
            and len(where_seq[idx]) == 2
            and (
                allow_short
                or not self._mix_active_bond_is_short(
                    where_seq[idx], target_sizes=target_sizes
                )
            )
        ):
            if max_span is not None and batch_where:
                sites = [site for previous in batch_where for site in previous]
                sites.extend(where_seq[idx])
                proposed_span = max(sites) - min(sites) + 1
                if proposed_span > int(max_span):
                    break
            batch_G.append(G_seq[idx])
            batch_where.append(where_seq[idx])
            idx += 1
        return batch_G, batch_where, idx

    def _mix_state_snapshot(self):
        """Capture mutable optimizer state before a trial mixed update."""
        return {
            "p": self.p,
            "p_exponent": getattr(self.p, "exponent", None),
            "info_c": deepcopy(self.info_c),
            "unitary_previous_norm": self._unitary_previous_norm,
            "norm_log_survival": self._norm_log_survival,
            "lengths": {
                "normalizations": len(self.normalizations),
                "norm_events": len(self.norm_events),
            },
        }

    @staticmethod
    def _copy_mix_tensor_data(data):
        """Copy one tensor payload without changing its array backend."""
        clone = getattr(data, "clone", None)
        if callable(clone):
            return clone()
        copy_data = getattr(data, "copy", None)
        if callable(copy_data):
            return copy_data()
        return deepcopy(data)

    def _mix_transaction_sites(self, where, info):
        """Return the tensors that a transactional trial can modify.

        FIT canonicalizes from the tracked center to the active window before
        optimizing it.  Copy that connecting interval, rather than the whole
        MPS.  If the center is unknown, retain the old full-copy safety rule.
        """
        L = int(getattr(self.p, "L", 0))
        if L <= 0:
            return ()
        active = (min(where), max(where))
        current = info.get("cur_orthog") if isinstance(info, dict) else None
        if (
            not isinstance(current, (tuple, list))
            or len(current) != 2
            or not all(isinstance(site, Integral) for site in current)
        ):
            return tuple(range(L))
        start = min(active[0], int(current[0]))
        stop = max(active[1], int(current[1]))
        return tuple(range(max(0, start), min(L - 1, stop) + 1))

    def _copy_mix_trial(self, committed_p, where, info):
        """Make a shallow MPS copy with an isolated mutable trial window."""
        sites = self._mix_transaction_sites(where, info)
        try:
            trial_p = committed_p.copy(deep=False)
        except (AttributeError, TypeError, ValueError):
            return committed_p.copy(deep=True), tuple(range(int(committed_p.L)))
        for site in sites:
            tensor = trial_p[site]
            tensor.modify(
                data=self._copy_mix_tensor_data(tensor.data),
                left_inds=tensor.left_inds,
            )
        return trial_p, sites

    def _validate_mix_norm(self, where, *, operation):
        """Validate mixed-mode commit norms only when finite_check is enabled.

        Mixed replay already leaves a tracked canonical center after each
        compression. Reading that center's Frobenius norm is sufficient for
        the normal health check and avoids scanning every tensor payload on
        every transaction. Full tensor-data checks remain available through
        ``quality_check_every``.
        """
        if not self._finite_check_enabled:
            return None
        try:
            retained_norm, _ = self._retained_center_norm(self.p, where)
            norm_value = self._real_float(ar.do("abs", retained_norm))
            exponent = float(getattr(self.p, "exponent", 0.0))
        except Exception as exc:
            raise FloatingPointError(
                f"{operation} retained norm could not be validated: {exc}"
            ) from exc
        if (
            norm_value <= 0.0
            or not np.isfinite(norm_value)
            or not np.isfinite(exponent)
        ):
            raise FloatingPointError(
                f"{operation} produced a zero or non-finite retained norm."
            )
        return norm_value

    def _restore_mix_state(self, snapshot):
        """Restore a mixed-mode transaction without changing caller identity."""
        self.p = snapshot["p"]
        if snapshot["p_exponent"] is not None:
            self.p.exponent = snapshot["p_exponent"]
        self.info_c = snapshot["info_c"]
        self._unitary_previous_norm = snapshot["unitary_previous_norm"]
        self._norm_log_survival = snapshot["norm_log_survival"]
        for attr, length in snapshot["lengths"].items():
            del getattr(self, attr)[length:]

    def _commit_mix_trial(self, committed_p, trial_p, *, sites=None):
        """Commit a successful trial while honoring ``inplace=True``."""
        if not self.inplace:
            self.p = trial_p
            return
        if trial_p is not committed_p:
            if len(committed_p.tensors) != len(trial_p.tensors):
                raise RuntimeError(
                    "mixed-mode trial changed the number of MPS tensors; "
                    "cannot preserve inplace object identity."
                )
            # DMRG can legitimately replace virtual bonds, especially when a
            # non-nearest gate is routed through a trial MPS. Preserve the
            # caller-owned network object, but adopt the valid trial graph
            # rather than rejecting its fresh bond labels. Take detached
            # snapshots first because the shallow trial copy can still share
            # tensors outside its transaction window.
            requested_sites = (
                tuple(range(len(committed_p.tensors)))
                if sites is None
                else tuple(sites)
            )
            changed_sites = set()
            for site in range(len(committed_p.tensors)):
                committed_tensor = committed_p[site]
                trial_tensor = trial_p[site]
                if committed_p.site_ind(site) != trial_p.site_ind(site):
                    raise RuntimeError(
                        "mixed-mode trial changed a physical MPS index; "
                        "cannot preserve inplace object identity."
                    )
                if committed_tensor.tags != trial_tensor.tags:
                    raise RuntimeError(
                        "mixed-mode trial changed MPS tensor tags; "
                        "cannot preserve inplace object identity."
                    )
                if committed_tensor.inds != trial_tensor.inds:
                    changed_sites.add(site)
            commit_sites = tuple(sorted(set(requested_sites) | changed_sites))
            trial_records = []
            for site in commit_sites:
                trial_tensor = trial_p[site]
                trial_records.append(
                    (
                        self._copy_mix_tensor_data(trial_tensor.data),
                        tuple(trial_tensor.inds),
                        trial_tensor.tags,
                        trial_tensor.left_inds,
                    )
                )
            # Include every changed endpoint in addition to the transaction
            # window so both sides of a newly labelled virtual bond are
            # updated. Unchanged tensors outside that window are not copied.
            for site, (data, inds, tags, left_inds) in zip(
                commit_sites,
                trial_records,
            ):
                committed_p[site].modify(
                    data=data,
                    inds=inds,
                    tags=tags,
                    left_inds=left_inds,
                )
            reset_cached = getattr(committed_p, "reset_cached_properties", None)
            if callable(reset_cached):
                reset_cached()
            committed_p.exponent = trial_p.exponent
        self.p = committed_p

    @staticmethod
    def _resolve_legacy_fit_option(
        *,
        canonical_name,
        canonical_value,
        canonical_default,
        legacy_name,
        legacy_value,
    ):
        """Resolve one deprecated FIT option without silently mixing policies.

        Canonical controls have readable defaults in the public signature.
        A supplied legacy value can replace that default, preserving old call
        sites. If the caller also selects a different non-default canonical
        value, fail early instead of guessing which convergence or
        stabilization policy they intended.
        """
        if legacy_value is _DEPRECATED_OPTION:
            return canonical_value
        warnings.warn(
            f"{legacy_name} is deprecated; use {canonical_name} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if (
            canonical_value != canonical_default
            and canonical_value != legacy_value
        ):
            raise ValueError(
                f"{canonical_name} and deprecated {legacy_name} specify "
                "different values; pass only the canonical option."
            )
        return legacy_value

    def _resolve_fit_rtol(self, value):
        """Return a validated dtype-aware FIT stopping tolerance."""
        if value == "auto":
            dtype = str(self.backend_dtype).lower()
            if "16" in dtype:
                return 1e-3
            if "32" in dtype or "complex64" in dtype:
                return 1e-5
            return 1e-9
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fit_rtol must be 'auto', a non-negative number, or None."
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "fit_rtol must be 'auto', a non-negative number, or None."
            )
        return value

    def _resolve_cutoff(self, value):
        """Return a validated truncation cutoff, including ``"auto"``."""
        if value == "auto":
            return dtype_auto_cutoff(self.backend_dtype)
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
    def _resolve_cutoff_mode(value, *, preserve_mpo_default=False):
        """Resolve ``cutoff_mode='auto'`` without leaking it to Quimb.

        Ordinary Pepsy compression uses relative discarded squared weight,
        which is the natural state-fidelity metric. Quimb MPO methods retain
        their own defaults when requested through ``auto``; this matters for
        the density-matrix compressor, whose native default is ``rsum1``.
        ``None`` remains a compatibility spelling for the same policy.
        """
        if value is None or (
            isinstance(value, str) and value.strip().lower() == "auto"
        ):
            if preserve_mpo_default:
                return None
            return _DEFAULT_CUTOFF_MODE
        return value

    @staticmethod
    def _resolve_fit_max_span(value, k_2q_batch):
        """Resolve the maximum inclusive spatial span of a FIT batch."""
        if value is None:
            return None
        if value == "auto":
            # A local layer of ``k`` neighboring two-site gates spans at most
            # roughly ``2 * k`` sites. Long-range first gates are always kept,
            # even when their individual span exceeds this soft cap.
            return max(3, 2 * int(k_2q_batch) + 1)
        if not isinstance(value, Integral) or int(value) < 2:
            raise ValueError(
                "fit_max_span must be 'auto', None, or an integer >= 2."
            )
        return int(value)

    @staticmethod
    def _resolve_quality_check_every(value):
        """Resolve the optional replay quality-check interval."""
        if value is None or value is False or value == 0:
            return None
        if value is True:
            return 1
        if not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(
                "quality_check_every must be None, bool, or a positive integer."
            )
        return int(value)

    def _run_quality_check(self, step, where, *, repair):
        """Record finite-data and canonical-gauge health at a replay step."""
        p = self.p
        finite = bool(self._mps_data_is_finite(p))
        if not finite:
            record = {
                "step": int(step),
                "where": tuple(where),
                "finite": False,
                "canonical_ok": False,
                "repaired": False,
            }
            self.quality_checks.append(record)
            raise FloatingPointError(
                f"quality check at step {step} found non-finite MPS data."
            )

        span = self._current_orthog(p)
        record = {
            "step": int(step),
            "where": tuple(where),
            "finite": True,
            "orthog_span": tuple(span),
            "max_bond": int(p.max_bond()),
            "repaired": False,
        }
        try:
            left_count, right_count = p.count_canonized()
            expected_canonical_sites = max(int(p.L) - 1, 0)
            canonical_ok = (
                int(left_count) + int(right_count)
                >= expected_canonical_sites
            )
            record.update(
                {
                    "canonical_left": int(left_count),
                    "canonical_right": int(right_count),
                    "expected_canonical_sites": expected_canonical_sites,
                    "canonical_ok": bool(canonical_ok),
                }
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            record.update(
                {
                    "canonical_ok": None,
                    "canonical_error": f"{type(exc).__name__}: {exc}",
                }
            )
            canonical_ok = True

        if not canonical_ok and repair:
            self.canonize_mps(p, span[1])
            self._record_orthog_span(p, (span[1], span[1]))
            record["repaired"] = True
            left_count, right_count = p.count_canonized()
            record["canonical_left_after"] = int(left_count)
            record["canonical_right_after"] = int(right_count)
            record["canonical_ok"] = (
                int(left_count) + int(right_count)
                >= max(int(p.L) - 1, 0)
            )
        self.quality_checks.append(record)
        return record

    def _maybe_run_quality_check(self, step, where, every, *, repair):
        """Run a periodic quality check when the requested interval is due."""
        if every is None or int(step) % int(every):
            return None
        return self._timed_call(
            "quality.check",
            self._run_quality_check,
            step,
            where,
            repair=repair,
        )

    @staticmethod
    def _mix_error_is_nonfinite(exc):
        """Return whether an exception reports NaN or infinite numerics."""
        if isinstance(exc, FloatingPointError):
            return True
        if "linalg" in type(exc).__name__.casefold():
            return True
        message = str(exc).casefold()
        return any(
            marker in message
            for marker in ("nan", "infs", "non-finite", "nonfinite", "infinite")
        )

    def _mix_target_bond_dimensions(self):
        """Return each bond's ``chi``-capped physical rank ceiling."""
        L = int(getattr(self.p, "L", 0))
        if L <= 1:
            return []
        dims = []
        for site in range(L):
            try:
                dim = int(self.p.phys_dim(site))
            except (AttributeError, TypeError, ValueError):
                dim = int(self.p.ind_size(self._format_ind(site)))
            dims.append(dim)

        left_caps = []
        rank = 1
        for site in range(L - 1):
            rank = min(int(self.chi), rank * dims[site])
            left_caps.append(rank)

        right_caps = [1] * (L - 1)
        rank = 1
        for site in range(L - 1, 0, -1):
            rank = min(int(self.chi), rank * dims[site])
            right_caps[site - 1] = rank
        return [
            min(int(self.chi), left, right)
            for left, right in zip(left_caps, right_caps)
        ]

    def _mix_active_bond_is_short(self, where, *, target_sizes=None):
        """Return whether an active bond is below its attainable target."""
        if self.chi <= 1 or len(where) < 2:
            return False
        xmin, xmax = min(where), max(where)
        if target_sizes is None:
            target_sizes = self._mix_target_bond_dimensions()
        return any(
            int(self.p.bond_size(site, site + 1)) < target_sizes[site]
            for site in range(xmin, xmax)
        )

    def _run_mix(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        logical_where_seq=None,
        n_iter,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        sticky_nonfinite,
        k_2q_batch=1,
        fit_max_span=None,
        mix_strict=False,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        submpo_method="direct",
        compression_seed=None,
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        non_unitary=False,
        quality_check_every=None,
        quality_check_repair=True,
    ):
        """Apply transactional FIT with an MPO fallback.

        Block FIT grows active bonds directly. Mixed mode's one-site default
        uses direct/MPO updates as a rank warm-up, then hands later eligible
        gates to one-site DMRG/FIT. Explicit block sizes 2 and 3 retain their
        corresponding mixed block-FIT schedules. Non-unitary trajectory
        branches use the explicit MPO fallback because mixed FIT is defined
        only for unitary working-norm updates.
        """
        mix_started = (
            time.perf_counter() if self._timing_state is not None else None
        )
        if non_unitary:
            # Mixed FIT's transactional contract is unitary. A selected Kraus
            # branch is still a valid noisy gate, so keep the requested mode
            # but use its physical MPO compression backend for this step.
            self._run_mpo(
                G_seq,
                where_seq,
                event_seq,
                progbar=progbar,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                normalize_every=None,
                normalize_final=False,
                non_unitary=True,
                submpo_method=submpo_method,
                compression_seed=compression_seed,
                stabilize_unitary=False,
            )
            return self.p
        if any(event_type == "submpo" for event_type in event_seq):
            raise ValueError("mode='mix' currently supports gate streams only.")

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="mix",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["mix"],
            )

        if logical_where_seq is None:
            logical_where_seq = where_seq
        if len(logical_where_seq) != len(where_seq):
            raise ValueError("logical and execution gate streams must have equal length.")

        target_sizes = self._mix_target_bond_dimensions()
        target_bond = max(target_sizes, default=1)
        mix_step_offset = len(self.mix_history)
        mpo_steps = sum(event["backend"] == "mpo" for event in self.mix_history)
        dmrg_steps = sum(event["backend"] == "dmrg" for event in self.mix_history)
        fallback_steps = sum(
            event.get("reason", "").startswith("dmrg_fallback")
            for event in self.mix_history
        )

        def append_entries(entries):
            self.mix_history.extend(entries)
            if pbar is not None:
                final = entries[-1]
                postfix = {
                    "backend": final["backend"],
                    "mpo": mpo_steps,
                    "dmrg": dmrg_steps,
                    "fallback": fallback_steps,
                    "bond": f"{final['end_bond']}/{self.chi}",
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                }
                pbar.set_postfix(postfix)
                pbar.update(len(entries))

        mpo_state_needs_check = False
        mpo_state_check_where = None

        def check_pending_mpo_state():
            """Validate one completed contiguous MPO warm-up block."""
            nonlocal mpo_state_needs_check, mpo_state_check_where
            if not mpo_state_needs_check:
                return
            self._validate_mix_norm(
                mpo_state_check_where,
                operation="MPO warm-up",
            )
            mpo_state_needs_check = False
            mpo_state_check_where = None

        idx = 0
        try:
            while idx < len(G_seq):
                gate = G_seq[idx]
                where = where_seq[idx]
                event_type = event_seq[idx]
                logical_where = logical_where_seq[idx]
                if len(where) not in {1, 2}:
                    raise ValueError("Each gate location must have one or two sites.")

                step = mix_step_offset + idx + 1
                start_bond = self._effective_max_bond(self.p)
                active_bond_is_short = self._mix_active_bond_is_short(
                    where, target_sizes=target_sizes
                )
                # Block FIT can grow the active bonds itself. The mixed
                # one-site path uses direct/MPO warm-up first so that the
                # subsequent one-site FIT has the required bond support.
                needs_rank_warmup = fit_block_size == 1 and (
                    start_bond < target_bond or active_bond_is_short
                )
                use_mpo = (
                    len(where) == 1
                    or self._mix_dmrg_disabled_reason is not None
                    or needs_rank_warmup
                )
                if use_mpo:
                    self._run_mix_mpo_step(
                        gate,
                        where,
                        event_type,
                        step=step,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        submpo_method=submpo_method,
                        compression_seed=compression_seed,
                        stabilize_unitary=stabilize_unitary,
                    )
                    mpo_steps += 1
                    mpo_state_needs_check = True
                    mpo_state_check_where = where
                    if len(where) == 1:
                        reason = "one_site_exact"
                    elif self._mix_dmrg_disabled_reason is not None:
                        reason = "dmrg_disabled_nonfinite"
                    elif start_bond < target_bond:
                        reason = "bond_below_target"
                    else:
                        reason = "active_bond_below_target"
                    entry = {
                        "step": int(step),
                        "where": tuple(logical_where),
                        "execution_where": tuple(where),
                        "start_bond": start_bond,
                        "target_bond": int(target_bond),
                        "backend": "mpo",
                        "reason": reason,
                        "end_bond": self._effective_max_bond(self.p),
                    }
                    if self._mix_dmrg_disabled_reason is not None:
                        entry["dmrg_disabled_reason"] = (
                            self._mix_dmrg_disabled_reason
                        )
                        entry["failed_sweep"] = self._mix_dmrg_failed_sweep
                    append_entries([entry])
                    idx += 1
                    self._maybe_run_quality_check(
                        step,
                        where,
                        quality_check_every,
                        repair=quality_check_repair,
                    )
                    continue

                check_pending_mpo_state()
                batch_G, batch_where, next_idx = self._collect_mix_dmrg_batch(
                    G_seq,
                    where_seq,
                    idx,
                    k_2q_batch,
                    target_sizes=target_sizes,
                    allow_short=fit_block_size in {2, 3},
                    max_span=fit_max_span,
                )
                batch_steps = [
                    mix_step_offset + position + 1
                    for position in range(idx, next_idx)
                ]
                batch_logical_where = logical_where_seq[idx:next_idx]
                if len(batch_where) == 1:
                    active_where = batch_where[0]
                else:
                    xmin = min(min(where_i) for where_i in batch_where)
                    xmax = max(max(where_i) for where_i in batch_where)
                    active_where = (xmin, xmax)
                snapshot = self._mix_state_snapshot()
                committed_p = snapshot["p"]
                self._last_dmrg_fit_diagnostics = None
                try:
                    # DMRG/FIT can mutate its input before it raises or
                    # produces invalid data. Isolate the canonicalization
                    # path and active window, while sharing untouched MPS
                    # tensors with the committed state. Keep the committed
                    # state as the MPO fallback target.
                    trial_p, transaction_sites = self._copy_mix_trial(
                        committed_p,
                        active_where,
                        snapshot["info_c"],
                    )
                    trial_p = self._install_represented_norm(trial_p)
                    self.p = trial_p
                    self.info_c = deepcopy(snapshot["info_c"])
                    self._prepare_fit_window(
                        active_where,
                        block_size=fit_block_size,
                    )
                    self._run_mix_dmrg_batch(
                        batch_G,
                        batch_where,
                        steps=batch_steps,
                        n_iter=n_iter,
                        fit_min_iter=fit_min_iter,
                        fit_rtol=fit_rtol,
                        fit_patience=fit_patience,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        fit_block_size=fit_block_size,
                        fit_adaptive_sweeps=fit_adaptive_sweeps,
                        fit_sweep_sequence=fit_sweep_sequence,
                        target_cutoff=target_cutoff,
                        fit_target_strategy=fit_target_strategy,
                        fit_init_strategy=fit_init_strategy,
                        fit_init_rand_strength=fit_init_rand_strength,
                        fit_init_seed=fit_init_seed,
                        fit_single_pair_fast_path=fit_single_pair_fast_path,
                        finite_check=finite_check,
                        fit_overlap_diagnostics=fit_overlap_diagnostics,
                        stabilize_unitary=stabilize_unitary,
                    )
                    fit_diagnostics = deepcopy(
                        self._last_dmrg_fit_diagnostics or {}
                    )
                    self._commit_mix_trial(
                        committed_p,
                        self.p,
                        sites=transaction_sites,
                    )
                except Exception as exc:  # fallback is the point of mix mode
                    self._restore_mix_state(snapshot)
                    if mix_strict:
                        raise
                    fit_diagnostics = deepcopy(
                        self._last_dmrg_fit_diagnostics or {}
                    )
                    if sticky_nonfinite and self._mix_error_is_nonfinite(exc):
                        self._mix_dmrg_disabled_reason = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        self._mix_dmrg_failed_sweep = getattr(
                            exc,
                            "fit_iteration",
                            fit_diagnostics.get("iterations") or None,
                        )
                    fallback_trial, fallback_sites = self._copy_mix_trial(
                        committed_p,
                        active_where,
                        snapshot["info_c"],
                    )
                    fallback_trial = self._install_represented_norm(fallback_trial)
                    try:
                        self.p = fallback_trial
                        self.info_c = deepcopy(snapshot["info_c"])
                        self._run_mix_mpo_batch(
                            batch_G,
                            batch_where,
                            event_seq[idx:next_idx],
                            steps=batch_steps,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            submpo_method=submpo_method,
                            compression_seed=compression_seed,
                            stabilize_unitary=stabilize_unitary,
                        )
                        self._commit_mix_trial(
                            committed_p,
                            self.p,
                            sites=fallback_sites,
                        )
                        mpo_state_needs_check = False
                    except BaseException:
                        self._restore_mix_state(snapshot)
                        raise
                    mpo_steps += len(batch_G)
                    fallback_steps += len(batch_G)
                    final_bond = self._effective_max_bond(self.p)
                    entries = []
                    for offset, (step_i, where_i, logical_i) in enumerate(
                        zip(batch_steps, batch_where, batch_logical_where)
                    ):
                        entries.append(
                            {
                                "step": int(step_i),
                                "where": tuple(logical_i),
                                "execution_where": tuple(where_i),
                                "start_bond": start_bond,
                                "target_bond": int(target_bond),
                                "backend": "mpo",
                                "reason": (
                                    "dmrg_fallback"
                                    if offset == 0
                                    else "dmrg_fallback_batch"
                                ),
                                "fallback_error": f"{type(exc).__name__}: {exc}",
                                "fit_iterations": fit_diagnostics.get(
                                    "iterations", 0
                                ),
                                "fit_converged": fit_diagnostics.get(
                                    "converged", False
                                ),
                                "fit_relative_change": fit_diagnostics.get(
                                    "relative_change"
                                ),
                                "dmrg_disabled": (
                                    self._mix_dmrg_disabled_reason is not None
                                ),
                                "failed_sweep": self._mix_dmrg_failed_sweep,
                                "end_bond": final_bond,
                            }
                        )
                    append_entries(entries)
                    idx = next_idx
                    self._maybe_run_quality_check(
                        batch_steps[-1],
                        active_where,
                        quality_check_every,
                        repair=quality_check_repair,
                    )
                    continue
                except BaseException:
                    self._restore_mix_state(snapshot)
                    raise

                dmrg_steps += len(batch_G)
                final_bond = self._effective_max_bond(self.p)
                entries = []
                for offset, (step_i, where_i, logical_i) in enumerate(
                    zip(batch_steps, batch_where, batch_logical_where)
                ):
                    entries.append(
                        {
                            "step": int(step_i),
                            "where": tuple(logical_i),
                            "execution_where": tuple(where_i),
                            "start_bond": start_bond,
                            "target_bond": int(target_bond),
                            "backend": "dmrg",
                            "reason": (
                                "bond_at_target"
                                if offset == 0 and start_bond >= target_bond
                                else "dmrg_batch"
                            ),
                            "fit_iterations": fit_diagnostics.get("iterations"),
                            "fit_converged": fit_diagnostics.get("converged"),
                            "fit_relative_change": fit_diagnostics.get(
                                "relative_change"
                            ),
                            "end_bond": final_bond,
                        }
                    )
                append_entries(entries)
                idx = next_idx
                self._maybe_run_quality_check(
                    batch_steps[-1],
                    active_where,
                    quality_check_every,
                    repair=quality_check_repair,
                )
            check_pending_mpo_state()
        finally:
            if pbar is not None:
                pbar.close()

        self.last_mix_summary = {
            "elapsed_seconds": (
                None
                if mix_started is None
                else float(time.perf_counter() - mix_started)
            ),
            "mpo_steps": int(mpo_steps),
            "dmrg_steps": int(dmrg_steps),
            "fallback_steps": int(fallback_steps),
            "final_bond": self._effective_max_bond(self.p),
            "chi": int(self.chi),
            "target_bond": int(target_bond),
            "dmrg_disabled": self._mix_dmrg_disabled_reason is not None,
            "dmrg_disabled_reason": self._mix_dmrg_disabled_reason,
            "failed_sweep": self._mix_dmrg_failed_sweep,
        }

    def _run_fit_gate(self, fit, **kwargs):
        """Run the gate-restricted FIT solver.

        ``FIT.run_gate`` is the MpsOptimizer DMRG kernel. It is the
        gate-window specialization of ``FIT.run_eff``: both reuse cached
        environments, but ``run_gate`` keeps the variational update inside
        the interval touched by the current gate or batch. Calling
        ``run_eff`` here would refit the complete MPS after every gate and
        would no longer implement local DMRG-style compression.
        """
        kwargs.setdefault(
            "two_site_transition_sweeps", 1 if self._dmrg_mode_alias == "dmrg3" else 0
        )
        # FIT is owned by this optimizer and discarded after the call. The
        # active replay already warned about diagnostics; keep every check
        # enabled without emitting another warning for each gate or segment.
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
            # FIT is optimizer-owned and discarded after this call. Move its
            # records into the run-level collector instead of copying every
            # nested per-site dictionary multiple times.
            for record in fit._take_timing_records():
                record["fit_index"] = fit_index
                record["record_index"] = len(self._timing_state["fit_steps"])
                self._timing_state["fit_steps"].append(record)

    def _run_dmrg_measurement(
        self,
        submpo,
        where,
        *,
        n_iter,
        cutoff,
        cutoff_mode,
        fit_min_iter,
        fit_rtol,
        fit_patience,
        fit_block_size,
        fit_adaptive_sweeps,
        fit_sweep_sequence,
        target_cutoff,
        fit_target_strategy,
        fit_mpo_guess,
        fit_init_strategy,
        fit_init_rand_strength,
        fit_init_seed,
        fit_single_pair_fast_path,
        measurement_index,
        finite_check=False,
        fit_overlap_diagnostics=False,
    ):
        """Apply a multi-site projective measurement through DMRG FIT.

        The unnormalized post-measurement state is first attached as a lazy
        sub-MPO target. FIT then compresses that target on the measurement
        span, using the same block schedule and SRC warm-start policy as an
        ordinary DMRG gate. The target remains unnormalized so the caller can
        record the Born-branch norm before renormalizing the live state.
        """
        p = self.p
        # ``where`` is the full support of the Pauli string for the sub-MPO,
        # while the DMRG/FIT window is represented by its endpoint span.
        where_sites = tuple(int(site) for site in where)
        span = (min(where_sites), max(where_sites))
        requested_block_size = min(
            int(fit_block_size),
            span[1] - span[0] + 1,
        )
        self._validate_dmrg1_iteration_budget(
            p,
            span,
            n_iter=n_iter,
            block_size=requested_block_size,
        )
        self._prepare_fit_window(span, block_size=fit_block_size)
        self.canonize_mps(p, span)
        state_snapshot = self._copy_fit_window_state(p, span)
        info_snapshot = dict(self.info_c)

        fit = None
        fit_error = None
        fit_initialization = {
            "strategy": "direct",
            "requested_strategy": fit_init_strategy,
            "guess_method": None,
            "guess_used": False,
            "svd_guess_used": False,
            "random_initialization": {
                "enabled": False,
                "reason": "direct",
            },
        }
        active_fit_block_size = self._dmrg_fit_block_size(
            p,
            span,
            fit_block_size,
        )
        adaptive_sweeps = (
            2 if self._dmrg_mode_alias == "dmrg1" else int(fit_adaptive_sweeps)
        )
        adaptive_rank_schedule = self._dmrg_mode_alias not in {
            "dmrg1",
            "dmrg2",
            "dmrg3",
        } and not (
            self._dmrg_mode_alias is None
            and active_fit_block_size in {2, 3}
            and span[1] - span[0] + 1 > active_fit_block_size
        )
        target_strategy = self._validate_fit_target_strategy(fit_target_strategy)
        _ = target_cutoff  # lazy targets are kept exact until FIT compression

        try:
            # Keep the projected state unnormalized throughout FIT. The final
            # center norm is the branch-amplitude measurement used by the
            # caller; renormalization belongs only to the post-collapse finish
            # so a DMRG approximation cannot replace the physical branch
            # probability with a post-localizer value.
            p_target = self._timed_call(
                "dmrg.target",
                self._build_lazy_submpo_target,
                p,
                submpo,
                span,
            )
            fit_initialization = self._timed_call(
                "dmrg.fit_guess",
                self._prepare_fit_initial_guess,
                p,
                (submpo,),
                (span,),
                block_size=active_fit_block_size,
                strategy=fit_init_strategy,
                fit_mpo_guess=fit_mpo_guess,
                rand_strength=fit_init_rand_strength,
                seed=(
                    int(fit_init_seed)
                    + 1000003 * int(measurement_index)
                    + 1009 * int(span[0])
                    + int(span[1])
                ),
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                submpo=True,
            )
            fit = FIT(
                p_target,
                p=fit_initialization["fit_guess"],
                cutoffs=cutoff,
                contraction_opt=self.contraction_opt,
                retag=False,
                range_int=[span[0], span[1]],
                inplace=True,
                copy_target=False,
            )
            self._run_fit_gate(
                fit,
                n_iter=n_iter,
                verbose=False,
                min_iter=fit_min_iter,
                rtol=fit_rtol,
                patience=fit_patience,
                finite_check=finite_check,
                block_size=active_fit_block_size,
                sweep_sequence=fit_sweep_sequence,
                max_bond=self.chi,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                single_pair_fast_path=fit_single_pair_fast_path,
                adaptive_block_sweeps=adaptive_sweeps,
                adaptive_until_rank=adaptive_rank_schedule,
                final_one_site_sweeps=0,
                collect_split_diagnostics=False,
            )
        except Exception as exc:  # retain the direct compressed fallback
            # FIT failures are transactional for this multi-site window.
            # Restore both tensor data and canonical metadata before using the
            # direct MPO fallback; otherwise partial variational writeback
            # could be silently combined with the fallback target.
            fit_error = exc

        fit_norm = None if fit is None else fit.final_norm
        fit_center = None if fit is None else fit.final_center_site
        if fit_error is not None:
            self.p = self._install_represented_norm(state_snapshot)
            self.info_c = info_snapshot
            try:
                self._apply_submpo_with_method(
                    self.p,
                    submpo,
                    span,
                    method="direct",
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    info=self.info_c,
                )
            except Exception:
                raise fit_error.with_traceback(fit_error.__traceback__)
            current_span = self._current_orthog(self.p)
            center = (
                int(current_span[0])
                if current_span[0] == current_span[1]
                else int(span[1])
            )
            if current_span[0] != current_span[1]:
                self.canonize_mps(self.p, center)
            projected_norm = self._real_float(
                ar.do("abs", self.p[center].norm())
            )
            self._last_dmrg_fit_diagnostics = {
                "iterations": 0 if fit is None else int(fit.iterations_run),
                "converged": False if fit is None else bool(fit.converged),
                "convergence_reason": (
                    None if fit is None else fit.convergence_reason
                ),
                "relative_change": (
                    None if fit is None else fit.last_relative_change
                ),
                "center_site": center,
                "block_size": int(active_fit_block_size),
                "adaptive_sweeps": 0 if fit is None else int(fit.adaptive_sweeps_run),
                "one_site_refinement_sweeps": (
                    0 if fit is None else int(fit.one_site_sweeps_run)
                ),
                "native_fermionic_warm_start": False,
                "mpo_fit_guess_used": False,
                "svd_guess_used": False,
                "guess_used": False,
                "guess_method": None,
                "fit_init_strategy": "direct",
                "fit_init_strategy_requested": fit_init_strategy,
                "random_initialization": fit_initialization.get(
                    "random_initialization"
                ),
                "target_strategy": target_strategy,
                "fit_overlap_diagnostics": bool(fit_overlap_diagnostics),
                "target_representation": "lazy_submpo",
                "backend": "mpo",
                "fallback": True,
                "fallback_reason": "fit_exception",
                "fallback_error": f"{type(fit_error).__name__}: {fit_error}",
                "fit_overlap_fidelity": None,
                "fit_overlap_infidelity": None,
                "fit_overlap_error": None,
            }
            return projected_norm, center

        self.p = self._install_represented_norm(fit.p)
        center = int(fit_center) if fit_center is not None else int(span[1])
        if fit_center is None:
            self.canonize_mps(self.p, center)
        self._record_orthog_span(self.p, (center, center))
        projected_norm = (
            self._real_float(ar.do("abs", fit_norm))
            if fit_norm is not None
            else self._real_float(ar.do("abs", self.p[center].norm()))
        )
        fit_overlap = (
            self._fit_overlap_diagnostics(p_target, fit.p)
            if fit_overlap_diagnostics
            else {}
        )
        self._last_dmrg_fit_diagnostics = {
            "iterations": int(fit.iterations_run),
            "converged": bool(fit.converged),
            "convergence_reason": fit.convergence_reason,
            "relative_change": fit.last_relative_change,
            "center_site": center,
            "block_size": int(active_fit_block_size),
            "adaptive_sweeps": int(fit.adaptive_sweeps_run),
            "one_site_refinement_sweeps": int(fit.one_site_sweeps_run),
            "native_fermionic_warm_start": False,
            "mpo_fit_guess_used": bool(fit_initialization["svd_guess_used"]),
            "svd_guess_used": bool(fit_initialization["svd_guess_used"]),
            "guess_used": bool(fit_initialization["guess_used"]),
            "guess_method": fit_initialization["guess_method"],
            "fit_init_strategy": fit_initialization["strategy"],
            "fit_init_strategy_requested": fit_initialization[
                "requested_strategy"
            ],
            "random_initialization": fit_initialization[
                "random_initialization"
            ],
            "target_strategy": target_strategy,
            "fit_overlap_diagnostics": bool(fit_overlap_diagnostics),
            "target_representation": "lazy_submpo",
            "fit_overlap_fidelity": fit_overlap.get("fit_overlap_fidelity"),
            "fit_overlap_infidelity": fit_overlap.get("fit_overlap_infidelity"),
            "fit_overlap_error": fit_overlap.get("fit_overlap_error"),
            "backend": "fit",
            "fallback": False,
        }
        self._maybe_lock_dmrg1_one_site_phase()
        return projected_norm, center

    def _run_dmrg(
        self,
        G_seq,
        where_seq,
        n_iter,
        event_seq=None,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        k_2q_batch=1,
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        fit_min_iter=None,
        fit_rtol=None,
        fit_patience=1,
        fit_finite_check=None,
        fit_block_size=2,
        fit_adaptive_sweeps=2,
        fit_sweep_sequence="RL",
        fit_max_span=None,
        target_cutoff=0.0,
        fit_target_strategy="auto",
        fit_mpo_guess=True,
        fit_init_strategy=_DEFAULT_FIT_INIT_STRATEGY,
        fit_init_rand_strength=0.0,
        fit_init_seed=0,
        fit_single_pair_fast_path=False,
        finite_check=False,
        fit_overlap_diagnostics=False,
        stabilize_unitary=False,
        quality_check_every=None,
        quality_check_repair=True,
    ):
        """Apply gates with local DMRG-style fitting."""
        if event_seq is None:
            event_seq = ("gate",) * len(G_seq)
        if len(event_seq) != len(G_seq):
            raise ValueError("DMRG event metadata must match the gate stream length.")
        if k_2q_batch < 1:
            raise ValueError("k_2q_batch must be >= 1.")
        fit_target_strategy = self._validate_fit_target_strategy(
            fit_target_strategy
        )
        if fit_target_strategy == "auto":
            # Layered targets retain exact operator factors without building a
            # growing target MPS, but they rely on ordinary dense site tags.
            # Native Symmray/fermionic states therefore stay on their graded
            # materialized route, where charge and dummy-mode metadata survive.
            fit_target_strategy = (
                "mps"
                if self._has_symmray_data(self.p) or self.p.isfermionic()
                else "layered"
            )

        # ``dmrg1`` has a bounded two-site warm-up: exactly two sweeps for an
        # under-capacity window, then one-site refinement. It latches into the
        # one-site phase once every full-chain bond reaches its attainable
        # physical/chi ceiling. ``dmrg2`` and ``dmrg3`` retain their configured
        # fixed warm-ups. Generic ``dmrg`` remains rank-adaptive for local
        # windows and uses the fixed canonical handoff for long-range windows.
        adaptive_rank_schedule = self._dmrg_mode_alias not in {
            "dmrg1",
            "dmrg2",
            "dmrg3",
        }
        adaptive_sweeps = (
            2 if self._dmrg_mode_alias == "dmrg1" else int(fit_adaptive_sweeps)
        )

        self._last_dmrg_fit_diagnostics = None
        p = self.p
        self._maybe_lock_dmrg1_one_site_phase()
        two_qubit_count = 0
        submpo_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)

        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress_mode = self._dmrg_mode_alias or "dmrg"
            pbar = tqdm(
                total=len(G_seq),
                desc=progress_mode,
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["dmrg"],
            )
        else:
            pbar = None

        def use_single_pair_fast_path(xmin, xmax, active_block_size):
            """Apply the named DMRG2 adjacent-pair schedule exception."""
            return bool(fit_single_pair_fast_path) or (
                self._dmrg_mode_alias == "dmrg2"
                and int(xmax) == int(xmin) + 1
                and int(active_block_size) == 2
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            where = where_seq[idx]
            gate = G_seq[idx]
            event_type = event_seq[idx]
            if len(where) == 1:
                if event_type == "submpo":
                    # A one-site sub-MPO does not need a variational window;
                    # keep this small compatibility path on the direct MPO
                    # application route.
                    self._run_mpo(
                        [gate],
                        [where],
                        [event_type],
                        progbar=False,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        normalize_every=None,
                        normalize_final=False,
                        non_unitary=non_unitary,
                        stabilize_unitary=stabilize_unitary,
                    )
                    p = self.p
                    compressed = True
                else:
                    self._apply_gate(
                        p,
                        gate,
                        where,
                        contract=True,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        inplace=True,
                    )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                is_submpo = event_type == "submpo"
                if not is_submpo and len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                if is_submpo and len(where) < 2:
                    raise ValueError(
                        "DMRG sub-MPO events require at least two support sites."
                    )

                if is_submpo or k_2q_batch == 1:
                    if is_submpo:
                        submpo_count += 1
                    else:
                        two_qubit_count += 1
                    xmin, xmax = sorted(where)
                    active_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    active_single_pair_fast_path = use_single_pair_fast_path(
                        xmin,
                        xmax,
                        active_fit_block_size,
                    )
                    self._validate_dmrg1_iteration_budget(
                        p,
                        (xmin, xmax),
                        n_iter=n_iter,
                        block_size=active_fit_block_size,
                    )
                    self._prepare_fit_window(
                        (xmin, xmax),
                        block_size=fit_block_size,
                    )
                    self.canonize_mps(p, (xmin, xmax))
                    unitary_target_norm = self._unitary_previous_norm

                    # Keep a transaction for an unexpected FIT exception.
                    # Normal low-rank long-range starts are repaired directly
                    # below by randomized initialization of the disposable FIT
                    # guess; norm loss is never silently converted into an MPO
                    # result.
                    fit_state_snapshot = (
                        self._copy_fit_window_state(p, (xmin, xmax))
                        if xmax - xmin > 1 and self.mode != "mix"
                        else None
                    )
                    fit_info_snapshot = (
                        dict(self.info_c)
                        if fit_state_snapshot is not None
                        else None
                    )
                    native_fit_guess_source = None
                    if (
                        not is_submpo
                        and self._has_symmray_data(p)
                        and self._native_src_fit_guess_enabled(
                            fit_init_strategy,
                            fit_mpo_guess,
                        )
                    ):
                        native_fit_guess_source = (
                            fit_state_snapshot
                            if fit_state_snapshot is not None
                            else p.copy(deep=True)
                        )

                    if is_submpo:
                        # An explicit sub-MPO is already the operator target;
                        # keep it as a lazy layer rather than densifying it or
                        # applying it to the live MPS before FIT.
                        p_g, active_target_strategy = self._timed_call(
                            "dmrg.target",
                            self._build_submpo_fit_target,
                            p,
                            gate,
                            where,
                            target_cutoff,
                            cutoff_mode,
                            target_strategy=fit_target_strategy,
                        )
                        native_fermionic_warm_start = False
                    else:
                        # Ordinary gates use the same target policy, but the
                        # native fermionic warm start is allowed to open charge
                        # sectors before FIT. That warm start is a disposable
                        # preparation step and never substitutes for ``p_g``.
                        active_target_strategy = fit_target_strategy
                        p_g = self._timed_call(
                            "dmrg.target",
                            self._build_norm_target,
                            p,
                            gate,
                            where,
                            target_cutoff,
                            cutoff_mode,
                            target_strategy=fit_target_strategy,
                        )
                        native_fermionic_warm_start = self._timed_call(
                            "dmrg.native_warm_start",
                            self._warm_start_native_fermionic_fit,
                            p,
                            (gate,),
                            (where,),
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                        )
                    active_fit_block_size = self._dmrg_fit_block_size(
                        p,
                        (xmin, xmax),
                        fit_block_size,
                    )
                    fit_guess_seed = (
                        int(fit_init_seed)
                        + 1000003 * int(idx)
                        + 1009 * int(xmin)
                        + int(xmax)
                    )
                    if is_submpo:
                        fit_initialization = self._timed_call(
                            "dmrg.fit_guess",
                            self._prepare_submpo_fit_initial_guess,
                            p,
                            gate,
                            where,
                            block_size=active_fit_block_size,
                            strategy=fit_init_strategy,
                            fit_mpo_guess=fit_mpo_guess,
                            rand_strength=fit_init_rand_strength,
                            seed=fit_guess_seed,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                        )
                    else:
                        fit_initialization = self._timed_call(
                            "dmrg.fit_guess",
                            self._prepare_fit_initial_guess,
                            p,
                            (gate,),
                            (where,),
                            block_size=active_fit_block_size,
                            strategy=fit_init_strategy,
                            fit_mpo_guess=fit_mpo_guess,
                            rand_strength=fit_init_rand_strength,
                            seed=fit_guess_seed,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            native_source=native_fit_guess_source,
                        )
                    fit_guess = fit_initialization["fit_guess"]
                    svd_guess_used = fit_initialization["svd_guess_used"]
                    mpo_fit_guess_used = svd_guess_used
                    random_initialization = fit_initialization[
                        "random_initialization"
                    ]
                    active_adaptive_sweeps = adaptive_sweeps
                    # A rank-adaptive sweep can finish a long-range window
                    # before its terminal one-site canonicalization has
                    # completed.  The local FIT norm then no longer describes
                    # the whole represented state, and the unitary norm guard
                    # correctly rejects the next gate.  Use the fixed
                    # block schedule for that window; it retains the same
                    # randomized FIT initialization while guaranteeing the
                    # canonical handoff. Named modes already use this path.
                    active_adaptive_rank_schedule = (
                        adaptive_rank_schedule
                        and not (
                            self._dmrg_mode_alias is None
                            and active_fit_block_size in {2, 3}
                            and xmax - xmin + 1 > active_fit_block_size
                        )
                    )
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
                    # Apply the selected one-, two-, or three-site FIT update to
                    # this gate window. ``run_gate`` reuses environments on
                    # both sides while leaving the rest of the MPS fixed.
                    fit_error = None
                    try:
                        self._run_fit_gate(
                            fit,
                            n_iter=n_iter,
                            verbose=False,
                            min_iter=fit_min_iter,
                            rtol=fit_rtol,
                            patience=fit_patience,
                            finite_check=fit_finite_check,
                            block_size=active_fit_block_size,
                            sweep_sequence=fit_sweep_sequence,
                            max_bond=self.chi,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            single_pair_fast_path=active_single_pair_fast_path,
                            adaptive_block_sweeps=active_adaptive_sweeps,
                            adaptive_until_rank=active_adaptive_rank_schedule,
                            final_one_site_sweeps=0,
                            collect_split_diagnostics=False,
                        )
                    except Exception as exc:
                        fit_error = exc
                    finally:
                        self._last_dmrg_fit_diagnostics = {
                            "iterations": int(fit.iterations_run),
                            "converged": bool(fit.converged),
                            "convergence_reason": fit.convergence_reason,
                            "relative_change": fit.last_relative_change,
                            "center_site": fit.final_center_site,
                            "block_size": int(active_fit_block_size),
                            "adaptive_sweeps": int(fit.adaptive_sweeps_run),
                            "one_site_refinement_sweeps": int(
                                fit.one_site_sweeps_run
                            ),
                            "native_fermionic_warm_start": bool(
                                native_fermionic_warm_start
                            ),
                            "mpo_fit_guess_used": bool(mpo_fit_guess_used),
                            "svd_guess_used": bool(svd_guess_used),
                            "guess_used": bool(fit_initialization["guess_used"]),
                            "guess_method": fit_initialization["guess_method"],
                            "guess_backend": fit_initialization.get(
                                "guess_backend"
                            ),
                            "native_randomized_guess_used": bool(
                                fit_initialization.get(
                                    "native_randomized_guess_used", False
                                )
                            ),
                            "fit_init_strategy": fit_initialization["strategy"],
                            "fit_init_strategy_requested": fit_initialization[
                                "requested_strategy"
                            ],
                            "random_initialization": random_initialization,
                            "target_strategy": active_target_strategy,
                            "fit_overlap_diagnostics": bool(
                                fit_overlap_diagnostics
                            ),
                            # Filled only after FIT succeeds.  This is a
                            # target-overlap diagnostic, not norm survival.
                            "fit_overlap_fidelity": None,
                            "fit_overlap_infidelity": None,
                            "fit_overlap_error": None,
                        }

                    fit_center = fit.final_center_site
                    fit_norm = fit.final_norm
                    fit_fallback_reason = None
                    if fit_error is not None and self.mode != "mix":
                        fit_fallback_reason = "fit_exception"

                    if fit_fallback_reason is not None:
                        if fit_state_snapshot is None:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise RuntimeError(
                                "DMRG FIT requested an MPO fallback without "
                                "a transactional state snapshot."
                            )
                        self.p = self._install_represented_norm(
                            fit_state_snapshot
                        )
                        self.info_c = fit_info_snapshot
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "mpo",
                                "fallback": True,
                                "fallback_reason": fit_fallback_reason,
                                "fit_norm": (
                                    None
                                    if fit_norm is None
                                    else self._real_float(
                                        ar.do("abs", fit_norm)
                                    )
                                ),
                            }
                        )
                        try:
                            self._run_mpo(
                                [gate],
                                [where],
                                [event_type],
                                progbar=False,
                                cutoff=cutoff,
                                cutoff_mode=cutoff_mode,
                                normalize_every=None,
                                normalize_final=False,
                                non_unitary=non_unitary,
                                stabilize_unitary=stabilize_unitary,
                            )
                        except Exception:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise
                        p = self.p
                    else:
                        if fit_error is not None:
                            raise fit_error.with_traceback(
                                fit_error.__traceback__
                            )
                        p = self._install_represented_norm(fit.p)
                        self.p = p
                        self._record_orthog_span(
                            p,
                            (fit_center, fit_center)
                            if fit_center is not None
                            else (xmin, xmax),
                        )
                        if not non_unitary:
                            self._timed_call(
                                "dmrg.stabilize",
                                self._stabilize_unitary_fit_state,
                                p,
                                (xmin, xmax),
                                unitary_target_norm,
                                current_norm=fit_norm,
                                center_site=fit_center,
                                restore=stabilize_unitary,
                            )
                        fit_overlap = (
                            {}
                            if self.mode == "mix" or not fit_overlap_diagnostics
                            else self._fit_overlap_diagnostics(p_g, fit.p)
                        )
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "fit",
                                "fallback": False,
                                "fit_overlap_diagnostics": bool(
                                    fit_overlap_diagnostics
                                ),
                                **fit_overlap,
                            }
                        )
                    self._maybe_lock_dmrg1_one_site_phase()
                    self._last_dmrg_fit_diagnostics[
                        "dmrg1_one_site_locked"
                    ] = bool(self._dmrg1_one_site_locked)
                    idx += 1
                    advanced = 1
                    last_where = (xmin, xmax)
                    compressed = True
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
                    active_fit_block_size = min(
                        fit_block_size,
                        xmax - xmin + 1,
                    )
                    self._validate_dmrg1_iteration_budget(
                        p,
                        (xmin, xmax),
                        n_iter=n_iter,
                        block_size=active_fit_block_size,
                    )
                    self._prepare_fit_window(
                        (xmin, xmax),
                        block_size=fit_block_size,
                    )
                    self.canonize_mps(p, (xmin, xmax))
                    unitary_target_norm = self._unitary_previous_norm
                    fit_state_snapshot = (
                        self._copy_fit_window_state(p, (xmin, xmax))
                        if xmax - xmin > 1 and self.mode != "mix"
                        else None
                    )
                    fit_info_snapshot = (
                        dict(self.info_c)
                        if fit_state_snapshot is not None
                        else None
                    )
                    native_fit_guess_source = None
                    if (
                        self._has_symmray_data(p)
                        and self._native_src_fit_guess_enabled(
                            fit_init_strategy,
                            fit_mpo_guess,
                        )
                    ):
                        native_fit_guess_source = (
                            fit_state_snapshot
                            if fit_state_snapshot is not None
                            else p.copy(deep=True)
                        )
                    p_g = self._timed_call(
                        "dmrg.target",
                        self._build_dmrg_batch_target,
                        p,
                        batch_G,
                        batch_where,
                        target_cutoff,
                        cutoff_mode,
                        target_strategy=fit_target_strategy,
                    )
                    native_fermionic_warm_start = self._timed_call(
                        "dmrg.native_warm_start",
                        self._warm_start_native_fermionic_fit,
                        p,
                        batch_G,
                        batch_where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                    )
                    active_fit_block_size = self._dmrg_fit_block_size(
                        p,
                        (xmin, xmax),
                        fit_block_size,
                    )
                    active_single_pair_fast_path = use_single_pair_fast_path(
                        xmin,
                        xmax,
                        active_fit_block_size,
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
                        native_source=native_fit_guess_source,
                    )
                    fit_guess = fit_initialization["fit_guess"]
                    random_initialization = fit_initialization[
                        "random_initialization"
                    ]
                    active_adaptive_sweeps = adaptive_sweeps
                    active_adaptive_rank_schedule = (
                        adaptive_rank_schedule
                        and not (
                            self._dmrg_mode_alias is None
                            and active_fit_block_size in {2, 3}
                            and xmax - xmin + 1 > active_fit_block_size
                        )
                    )
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
                    fit_error = None
                    try:
                        self._run_fit_gate(
                            fit,
                            n_iter=n_iter,
                            verbose=False,
                            min_iter=fit_min_iter,
                            rtol=fit_rtol,
                            patience=fit_patience,
                            finite_check=fit_finite_check,
                            block_size=active_fit_block_size,
                            sweep_sequence=fit_sweep_sequence,
                            max_bond=self.chi,
                            cutoff=cutoff,
                            cutoff_mode=cutoff_mode,
                            single_pair_fast_path=active_single_pair_fast_path,
                            adaptive_block_sweeps=active_adaptive_sweeps,
                            adaptive_until_rank=active_adaptive_rank_schedule,
                            final_one_site_sweeps=0,
                            collect_split_diagnostics=False,
                        )
                    except Exception as exc:
                        fit_error = exc
                    finally:
                        self._last_dmrg_fit_diagnostics = {
                            "iterations": int(fit.iterations_run),
                            "converged": bool(fit.converged),
                            "convergence_reason": fit.convergence_reason,
                            "relative_change": fit.last_relative_change,
                            "center_site": fit.final_center_site,
                            "block_size": int(active_fit_block_size),
                            "adaptive_sweeps": int(fit.adaptive_sweeps_run),
                            "one_site_refinement_sweeps": int(
                                fit.one_site_sweeps_run
                            ),
                            "native_fermionic_warm_start": bool(
                                native_fermionic_warm_start
                            ),
                            "mpo_fit_guess_used": bool(
                                fit_initialization["svd_guess_used"]
                            ),
                            "svd_guess_used": bool(
                                fit_initialization["svd_guess_used"]
                            ),
                            "guess_used": bool(fit_initialization["guess_used"]),
                            "guess_method": fit_initialization["guess_method"],
                            "guess_backend": fit_initialization.get(
                                "guess_backend"
                            ),
                            "native_randomized_guess_used": bool(
                                fit_initialization.get(
                                    "native_randomized_guess_used", False
                                )
                            ),
                            "fit_init_strategy": fit_initialization["strategy"],
                            "fit_init_strategy_requested": fit_initialization[
                                "requested_strategy"
                            ],
                            "random_initialization": random_initialization,
                            "target_strategy": fit_target_strategy,
                            "fit_overlap_diagnostics": bool(
                                fit_overlap_diagnostics
                            ),
                            # Filled only after FIT succeeds.  This is a
                            # target-overlap diagnostic, not norm survival.
                            "fit_overlap_fidelity": None,
                            "fit_overlap_infidelity": None,
                            "fit_overlap_error": None,
                        }

                    fit_center = fit.final_center_site
                    fit_norm = fit.final_norm
                    fit_fallback_reason = None
                    if fit_error is not None and self.mode != "mix":
                        fit_fallback_reason = "fit_exception"

                    if fit_fallback_reason is not None:
                        if fit_state_snapshot is None:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise RuntimeError(
                                "DMRG FIT requested an MPO fallback without "
                                "a transactional state snapshot."
                            )
                        self.p = self._install_represented_norm(
                            fit_state_snapshot
                        )
                        self.info_c = fit_info_snapshot
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "mpo",
                                "fallback": True,
                                "fallback_reason": fit_fallback_reason,
                                "fit_norm": (
                                    None
                                    if fit_norm is None
                                    else self._real_float(
                                        ar.do("abs", fit_norm)
                                    )
                                ),
                            }
                        )
                        try:
                            self._run_mpo(
                                batch_G,
                                batch_where,
                                ["gate"] * len(batch_G),
                                progbar=False,
                                cutoff=cutoff,
                                cutoff_mode=cutoff_mode,
                                normalize_every=None,
                                normalize_final=False,
                                non_unitary=non_unitary,
                                stabilize_unitary=stabilize_unitary,
                            )
                        except Exception:
                            if fit_error is not None:
                                raise fit_error.with_traceback(
                                    fit_error.__traceback__
                                )
                            raise
                        p = self.p
                    else:
                        if fit_error is not None:
                            raise fit_error.with_traceback(
                                fit_error.__traceback__
                            )
                        p = self._install_represented_norm(fit.p)
                        self.p = p
                        self._record_orthog_span(
                            p,
                            (fit_center, fit_center)
                            if fit_center is not None
                            else (xmin, xmax),
                        )
                        if not non_unitary:
                            self._timed_call(
                                "dmrg.stabilize",
                                self._stabilize_unitary_fit_state,
                                p,
                                (xmin, xmax),
                                unitary_target_norm,
                                current_norm=fit_norm,
                                center_site=fit_center,
                                restore=stabilize_unitary,
                            )
                        fit_overlap = (
                            {}
                            if self.mode == "mix" or not fit_overlap_diagnostics
                            else self._fit_overlap_diagnostics(p_g, fit.p)
                        )
                        self._last_dmrg_fit_diagnostics.update(
                            {
                                "backend": "fit",
                                "fallback": False,
                                "fit_overlap_diagnostics": bool(
                                    fit_overlap_diagnostics
                                ),
                                **fit_overlap,
                            }
                        )
                    self._maybe_lock_dmrg1_one_site_phase()
                    self._last_dmrg_fit_diagnostics[
                        "dmrg1_one_site_locked"
                    ] = bool(self._dmrg1_one_site_locked)
                    advanced = next_idx - idx
                    idx = next_idx
                    last_where = (xmin, xmax)
                    compressed = True

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._maybe_run_quality_check(
                idx,
                last_where,
                quality_check_every,
                repair=quality_check_repair,
            )

            self._record_effective_event(last_where, event_type="gate")

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                if submpo_count:
                    postfix["mpo"] = submpo_count
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_su(
        self,
        G_seq,
        where_seq,
        event_seq,
        *,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
    ):
        """Apply a gate stream with simple-update bond gauges.

        ``self.p`` remains the simple-update core and ``self.gauges`` stores
        the external bond factors. This path intentionally does not
        canonicalize the MPS.
        """
        self._prepare_su_state()
        p = self.p

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="su",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["su"],
            )

        try:
            for step, (gate, where, event_type) in enumerate(
                zip(G_seq, where_seq, event_seq),
                start=1,
            ):
                if event_type != "gate":
                    raise ValueError(
                        "mode='su' supports gate-only streams; subMPO events "
                        "are not supported."
                    )
                if len(where) not in {1, 2}:
                    raise ValueError(
                        "Each simple-update gate location must have one or two sites."
                    )

                p = apply_gate_simple(
                    p,
                    gate,
                    where,
                    gauges=self.gauges,
                    ind_id=self.ind_id,
                    renorm=True,
                    max_bond=self.chi,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if pbar is not None:
                    pbar.set_postfix(
                        {"bnd": p.max_bond(), "gauges": len(self.gauges)}
                    )
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        self.p = p
        self.info_c = {}
        self._su_gauges_ready = True
        self._su_gauges_state = self.p
        self._refresh_su_physical_state()

    def _run_mpo(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        event_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        submpo_method="direct",
        compression_seed=None,
        stabilize_unitary=False,
    ):
        """Apply gates with MPO-style nonlocal compression.

        Uses :meth:`qtn.MatrixProductState.gate_nonlocal_` for two-qubit gates.
        """
        p = self.p
        mpo_method = self._normalize_submpo_method(submpo_method)
        # ``mpo`` is the internal backend name; expose the selected Quimb
        # compressor in timing records instead (``direct``, ``src``, etc.).
        timing_name = mpo_method
        gate_cutoff_mode = (
            _DEFAULT_CUTOFF_MODE
            if cutoff_mode is None
            else cutoff_mode
        )
        # ``mpo_cutoff_mode`` is intentionally resolved by ``run`` before
        # entering this backend: ``None`` means keep Quimb's native default
        # for methods such as ``dm``. One-site gates do not invoke that MPO
        # compressor, so they use the ordinary Pepsy cutoff policy here.
        mpo_compress_opts = self._submpo_compress_opts(
            mpo_method,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        mpo_optimize = mpo_compress_opts.get("optimize")
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        submpo_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            progress_mode = self._progress_mode_name(self.mode)
            pbar = tqdm(
                total=len(G_seq),
                desc=progress_mode,
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS.get(
                    progress_mode,
                    self._PROGBAR_COLORS["mpo"],
                ),
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            unitary_target_norm = (
                self._unitary_previous_norm if not non_unitary else None
            )
            where = where_seq[idx]
            gate = G_seq[idx]
            event_type = event_seq[idx]
            if event_type == "submpo":
                # The payload is already an exact operator representation.
                # Apply it through ``gate_with_submpo_`` so the selected
                # compressor sees the original MPO; do not densify it merely
                # to reuse the ordinary gate branch.
                submpo_count += 1
                xmin, xmax = min(where), max(where)
                if (
                    mpo_method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
                    and _is_interior_submpo_span(p, where)
                ):
                    _apply_submpo_with_interior_workaround(
                        p,
                        gate,
                        where,
                        chi=self.chi,
                        method=mpo_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        info=self.info_c,
                        inplace_mpo=False,
                        optimize=mpo_optimize,
                        seed=compression_seed,
                    )
                else:
                    self.canonize_mps(p, (xmin, xmax))
                    submpo_opts = dict(mpo_compress_opts)
                    if mpo_method in _MPO_METHODS_USE_SEED:
                        submpo_opts["seed"] = compression_seed
                    _run_seeded_quimb(
                        None,
                        p.gate_with_submpo_,
                        gate,
                        where=where,
                        method=mpo_method,
                        max_bond=self.chi,
                        info=self.info_c,
                        inplace_mpo=False,
                        **submpo_opts,
                    )

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        f"{timing_name}.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )
            elif len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=gate_cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = self.backend == "symmray"
                self.canonize_mps(p, (xmin, xmax))
                if use_symmray_auto_swap:
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=gate_cutoff_mode,
                        max_bond=self.chi,
                    )
                else:
                    _apply_dense_gate_with_method(
                        p,
                        gate,
                        where,
                        dims=self._infer_gate_dims(gate, where),
                        chi=self.chi,
                        method=mpo_method,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        info=self.info_c,
                        optimize=mpo_optimize,
                        seed=compression_seed,
                    )
                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        f"{timing_name}.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._record_effective_event(where, event_type=event_type)

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                if submpo_count:
                    postfix["mpo"] = submpo_count
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_swap(self, *args, **kwargs):
        """Apply gates with swap-network compression, swapping back."""
        return self._run_swap_network(*args, swap_back=True, mode_name="swap", **kwargs)

    def _run_perm(self, *args, **kwargs):
        """Apply gates with lazy swap-network compression."""
        return self._run_swap_network(*args, swap_back=False, mode_name="perm", **kwargs)

    def _run_swap_network(  # pylint: disable=too-many-locals,too-many-arguments
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        stabilize_unitary=False,
        *,
        swap_back,
        mode_name,
    ):
        """Apply gates with swap-network compression for nonlocal 2-site gates.

        Uses in-place ``gate_with_auto_swap_`` for two-site gates. When
        ``swap_back`` is false, ``where_seq`` is interpreted as logical sites,
        the current ``self.qubits`` mapping translates them to physical sites,
        and the right endpoint remains at the left endpoint's neighbour.
        """
        # ``swap_back`` is the semantic switch: ``swap`` restores the input
        # logical order after each nonlocal gate, while ``perm`` leaves the
        # physical order changed and updates ``self.qubits`` for later gates
        # and logical readout.
        p = self.p
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc=mode_name,
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS[mode_name],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            unitary_target_norm = (
                self._unitary_previous_norm if not non_unitary else None
            )
            logical_where = where_seq[idx]
            where = (
                tuple(int(site) for site in logical_where)
                if swap_back
                else self._logical_to_physical_where(logical_where)
            )
            gate = G_seq[idx]
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1
                xmin, xmax = sorted(where)
                self.canonize_mps(p, (xmin, xmax))

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                if self._has_symmray_data(p) and self._native_needs_safe_qr(p):
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        max_bond=self.chi,
                        info=self.info_c,
                        swap_back=swap_back,
                    )
                else:
                    p.gate_with_auto_swap_(
                        gate,
                        where,
                        info=self.info_c,
                        max_bond=self.chi,
                        swap_back=swap_back,
                        **compress_opts,
                    )
                if not swap_back:
                    self._record_permutation_move(where)

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        f"{mode_name}.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._record_effective_event(last_where, event_type="gate")

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_svd(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        normalize_every=None,
        normalize_final=True,
        normalize_eps=1e-15,
        non_unitary=False,
        stabilize_unitary=False,
    ):
        """Apply gates with local SVD compression for nonlocal 2-site gates.

        Two-site gates are applied with ``contract="reduce-split"`` then
        compressed on the local span to ``max_bond=self.chi``. Symmray-backed
        MPS data use quimb's block-aware auto-swap split path by default as a
        conservative choice for block-sparse edge cases.
        """
        # SVD mode deliberately exposes the local gate-split algorithm. It is
        # useful as a transparent reference, whereas MPO/DMRG modes retain a
        # full operator target and can choose more global compression schemes.
        p = self.p
        stabilize_unitary = bool(stabilize_unitary) and not non_unitary
        if not non_unitary and self._unitary_previous_norm is None:
            self._start_unitary_norm_tracking(p)
        two_qubit_count = 0
        last_where = self._current_orthog(p)
        last_normalized_step = None

        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="svd",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["svd"],
            )

        idx = 0
        while idx < len(G_seq):
            compressed = False
            unitary_target_norm = (
                self._unitary_previous_norm if not non_unitary else None
            )
            where = where_seq[idx]
            gate = G_seq[idx]
            if len(where) == 1:
                self._apply_gate(
                    p,
                    gate,
                    where,
                    contract=True,
                    cutoff=cutoff,
                    cutoff_mode=cutoff_mode,
                    inplace=True,
                )
                if non_unitary:
                    self.canonize_mps(p, where)
                idx += 1
                advanced = 1
                last_where = where
            else:
                if len(where) != 2:
                    raise ValueError("Each gate location must have one or two sites.")
                two_qubit_count += 1

                compress_opts = {"cutoff": cutoff, "cutoff_mode": cutoff_mode}
                xmin, xmax = sorted(where)
                use_symmray_auto_swap = self.backend == "symmray"
                self.canonize_mps(p, (xmin, xmax))
                if use_symmray_auto_swap:
                    self._apply_symmray_auto_swap_gate(
                        p,
                        gate,
                        where,
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        max_bond=self.chi,
                    )
                else:
                    self._apply_gate(
                        p,
                        gate,
                        where,
                        contract="reduce-split",
                        cutoff=cutoff,
                        cutoff_mode=cutoff_mode,
                        inplace=True,
                    )
                    self.canonize_mps(p, (xmin, xmax))

                    for i in range(xmax, xmin, -1):
                        p.right_canonize_site(i, bra=None)
                    p.left_compress(
                        start=xmin,
                        stop=xmax,
                        max_bond=self.chi,
                        **compress_opts,
                    )
                    # ``right_canonize_site`` plus left-to-right compression
                    # leaves the complete raw norm at the right endpoint.
                    self._record_orthog_span(p, (xmax, xmax))

                idx += 1
                advanced = 1
                last_where = (xmin, xmax)
                compressed = True
                if not non_unitary:
                    approx_norm, approx_center = self._retained_center_norm(
                        p, (xmin, xmax)
                    )
                    self._timed_call(
                        "svd.stabilize",
                        self._stabilize_unitary_compression_state,
                        p,
                        (xmin, xmax),
                        unitary_target_norm,
                        current_norm=approx_norm,
                        center_site=approx_center,
                        restore=stabilize_unitary,
                    )

            event = self._maybe_normalize_after_step(
                p,
                step=idx,
                where=last_where,
                normalize_every=normalize_every,
                reason="compression" if compressed else "step",
            )
            if event is not None:
                last_normalized_step = idx

            self._record_effective_event(last_where, event_type="gate")

            if pbar is not None:
                postfix = {
                    "2q": two_qubit_count,
                    "~F": self._format_progress_scalar(
                        self._cumulative_fidelity()
                    ),
                    "bnd": p.max_bond(),
                }
                pbar.set_postfix(postfix)
                pbar.update(advanced)

        if pbar is not None:
            pbar.close()

        event = self._maybe_normalize_final(
            p,
            step=idx,
            last_normalized_step=last_normalized_step,
            where=last_where,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
        )
        if event is not None:
            last_normalized_step = idx

        self.p = self._install_represented_norm(p)

    def _run_exact(  # pylint: disable=too-many-locals
        self,
        G_seq,
        where_seq,
        progbar=False,
        cutoff=1e-12,
        cutoff_mode="rsum2",
    ):
        """Apply gates exactly using in-place ``contract=True`` application.

        Progress bar counts all gates for consistency with other modes.
        """
        self.p = self.p.contract(all, optimize="auto-hq")
        self.p = self._install_represented_norm(qtn.TensorNetwork([self.p]))
        self.info_c = {}
        p = self.p
        two_qubit_count = 0
        pbar = None
        if progbar:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=len(G_seq),
                desc="exact",
                leave=True,
                position=0,
                ascii=True,
                colour=self._PROGBAR_COLORS["exact"],
            )

        for gate, where in zip(G_seq, where_seq):
            if len(where) not in (1, 2):
                raise ValueError("Each gate location must have one or two sites.")

            inds = [self._format_ind(site) for site in where]
            qtn.tensor_network_gate_inds(
                p,
                gate,
                inds,
                contract=True,
                info=None,
                inplace=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
            )
            self._record_effective_event(where, event_type="gate")

            if len(where) == 1:
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "2q": two_qubit_count,
                            "~F": self._format_progress_scalar(
                                self._cumulative_fidelity()
                            ),
                            "bnd": "inf",
                        }
                    )
                    pbar.update(1)
                continue

            two_qubit_count += 1
            if pbar is not None:
                pbar.set_postfix(
                    {
                        "2q": two_qubit_count,
                        "~F": self._format_progress_scalar(
                            self._cumulative_fidelity()
                        ),
                        "bnd": "inf",
                    }
                )
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.p = self._install_represented_norm(p)

    def canonize_mps(self, p, where, *, info=None):
        """Update canonical form and optionally accumulate its wall time."""
        if self._timing_state is None:
            return self._canonize_mps_impl(p, where, info=info)
        return self._timed_call(
            "canonicalize",
            self._canonize_mps_impl,
            p,
            where,
            info=info,
        )

    def _canonize_mps_impl(self, p, where, *, info=None):
        """Update canonical form around a one- or two-site gate span.

        ``where`` may be an int, a 1-tuple ``(site,)``, or a 2-tuple
        ``(xmin, xmax)``. Integers and singletons collapse to a single-site
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
            site0, site1 = int(where[0]), int(where[1])
            xmin, xmax = min(site0, site1), max(site0, site1)
            where_canon = [xmin, xmax]
            target_orthog = (xmin, xmax)
        else:
            raise ValueError("where must be an int, (int,), or (int, int).")

        state_info = self._info_for_state(p, info)
        p.canonize(
            where_canon,
            cur_orthog=self._current_orthog(p, info=state_info),
            info=state_info,
        )
        # Preserve the fitting-window semantics expected by gate updates.
        state_info["cur_orthog"] = target_orthog
        return target_orthog

    def get_quality_checks(self):
        """Return periodic finite-data and canonical-gauge check records."""
        return deepcopy(self.quality_checks)

    def get_normalizations(self):
        """Return automatic normalization events recorded during ``run``.

        Each event contains the 1-based ``step``, removed local ``old_norm``,
        active ``span``, rescaled ``sites``, per-tensor ``scales``, total
        ``log10_scale``, event ``reason``, and resulting base-10 ``exponent``.
        """
        return deepcopy(self.normalizations)

    def get_norm_events(self):
        """Return a defensive copy of automatic norm-survival events."""
        return deepcopy(self.norm_events)
