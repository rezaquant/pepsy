"""``MpsStabOptimizer``: an ``MpsOptimizer``-style gate-stream simulator for STN.

Analogous to :class:`pepsy.MpsOptimizer`, but the state is a *stabilizer tensor
network*: a stim tableau (basis ``B(S, D)``) times a coefficient MPS ``|nu>``
(see :class:`pepsy.optimizers.stabilizer_tn.STNState`).  A gate stream is replayed against
the state, routing each entry to the cheap update path:

* **Clifford gates** update only the tableau (the coefficient MPS ``p`` is
  unchanged, free).
* **Non-Clifford rotations** (single- or multi-qubit Pauli exponentials) update
  only ``p`` via ``exp(-i theta/2 * A) -> exp(-i theta/2 * C^dagger A C)``.
  When a separable stabilizer coefficient site permits the constructive exact
  cooling identity, the update is reduced to a one-site rotation and a
  tableau-only controlled-Pauli cascade; otherwise it is applied as an exact
  bond-dim-2 MPO with optional ``chi`` truncation.
* **Explicit gate matrices** are classified: Clifford matrices go to the
  tableau; non-Clifford single-qubit *unitaries* are ZYZ-decomposed into
  rotations; other few-qubit matrices (unitary **or** non-unitary) are
  Pauli-decomposed ``G = sum_a c_a P_a`` and applied to ``p`` as
  ``M = C^dagger G C = sum_a c_a (C^dagger P_a C)``.  Sparse frame Pauli sums
  are applied as exact low-bond sub-MPOs; dense sums fall back to a compressed
  sum of signed Pauli-string branches.  Non-unitary ``G`` is represented
  without renormalization, so the coefficient norm tracks ``|G|psi>|``.
* **Sub-MPO events** apply a user MPO directly to ``p`` (interpreted in the
  *coefficient* frame; any MPO, unitary or not), matching the ``MpsOptimizer``
  sub-MPO contract.  A *physical*-frame few-qubit operator should instead be
  supplied as a dense ``(matrix, where)`` entry.

Supported gate-stream entry forms::

    ("h", q) ("s", q) ("sdg", q) ("x"|"y"|"z", q)          # 1q Clifford
    ("cnot"|"cx", c, t) ("cz", a, b) ("cy", a, b) ("swap", a, b)
    ("rx"|"ry"|"rz", theta, q)                              # 1q non-Clifford
    ("rxx"|"ryy"|"rzz", theta, a, b)                        # 2q Pauli rotations
    ("rot", theta, "XZ...", where)                          # general Pauli exp
    ("t", q) ("tdg", q)                                     # T / T-dagger
    (matrix, where)                                         # bounded few-qubit gate
    ("submpo", mpo, where)  / {"kind": "submpo", ...}       # coeff-frame sub-MPO
    ("measure", pauli, where[, outcome[, absorb_basis]])   # Pauli measurement
    ("reset", where[, basis])                               # reset qubit(s) to +basis
    ("measure_reset", basis, where[, outcome[, absorb_basis]])
                                                              # measure, record, reset
    ("cap", where, vec[, absorb])                            # guarded dense physical cap
    ("disentangle"[, {"sweeps": ..., "bonds": ..., "tol": ...}])
                                                              # Clifford gauge sweep
"""

from __future__ import annotations

import math
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from typing import List, Optional

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends import (
    backend_infer,
    backend_signatures_compatible,
    infer_backend_converter_from_sample,
    infer_backend_signature,
)
from ...fitting.local import FIT
from ..._internal.random import backend_random_array
from .._fidelity import (
    fidelity_from_log,
    infidelity_from_log,
    log_fidelity_from_norms,
)
from ..mps.layout import MpsGateStreamLayoutFinder
from ..mps.optimizer import (
    _MPO_COMPRESSION_METHODS,
    _MPO_METHODS_IGNORE_CUTOFF,
    _MPO_METHODS_NEED_INTERIOR_WORKAROUND,
    _MPO_METHODS_USE_SEED,
    _apply_submpo_with_interior_workaround,
    _run_seeded_quimb,
    _resolve_conditional,
    conditional_event_parts,
    is_submpo_event,
    submpo_event_parts,
)
from .operators import (
    pauli_combo_submpo,
    pauli_decomposition,
    pauli_matrix,
    pauli_sum_submpo,
    single_qubit_combo_matrix,
    single_qubit_rotation_matrix,
)
from .dense import _as_gate_matrix, _is_unitary, _tableau_from_exact_unitary
from .paulis import (
    _resolve_measurement_disentangle,
    hermitian_pauli_terms,
    pauli_string,
)
from .records import (
    DeferredInjectionRecord,
    DeferredInjectionReport,
    DeferredProjectionRecord,
    ImmediateInjectionReport,
    ImmediateProjectionRecord,
    MeasurementRecord,
    NormEventRecord,
    StabilizerMpsSettingsAdvice,
    StabilizerMpsRunResult,
    StreamAnalysisRecord,
)
from .settings import (
    DEFAULT_MPS_STAB_MAX_PAULI_DECOMPOSITION_QUBITS,
)
from .stn_state import STNState, _tableau_gate_stream, _validate_bits

__all__ = [
    "DeferredInjectionRecord",
    "DeferredInjectionReport",
    "DeferredProjectionRecord",
    "ImmediateInjectionReport",
    "ImmediateProjectionRecord",
    "MeasurementRecord",
    "MpsStabOptimizer",
    "NormEventRecord",
    "StabilizerMpsSettingsAdvice",
    "StabilizerMpsSimulator",
    "StabilizerMpsRunResult",
    "StreamAnalysisRecord",
    "run_stabilizer_mps_stream",
]

_CLIFFORD_NAMES = {
    "h", "x", "y", "z", "s", "sdg", "sdag", "sqrt_x", "sqrt_x_dag",
    "cnot", "cx", "cy", "cz", "swap",
}
_ROTATION_AXES = {"rx": "X", "ry": "Y", "rz": "Z"}
_ROTATION_AXES_2Q = {"rxx": "X", "ryy": "Y", "rzz": "Z"}
_RESET_FLIP_CLIFFORDS = {"X": "z", "Y": "x", "Z": "x"}
_RESET_AXIS_ALIASES = {"reset_x": "X", "reset_y": "Y", "reset_z": "Z"}
_MR_ALIASES = {"measure_reset", "mr", "mreset", "measure_and_reset"}
_MR_AXIS_ALIASES = {"mrx": "X", "mry": "Y", "mrz": "Z"}
_MAX_PAULI_SUM_SUBMPO_TERMS = 4
_FIT_INIT_STRATEGIES = frozenset(
    {"auto", "direct", "random", "random_expand", "svd_guess"}
    | {f"guess_{method}" for method in _MPO_COMPRESSION_METHODS}
)

_TABLEAU_ANSI = {
    "header": "1;36",
    "section": "1;35",
    "destabilizer": "32",
    "stabilizer": "33",
    "muted": "2",
}


def _tableau_color(text, style, enabled):
    """Apply one small ANSI style used by the compact tableau display."""
    if not enabled:
        return str(text)
    return f"\033[{_TABLEAU_ANSI[style]}m{text}\033[0m"


def _format_tableau_pauli(pauli, *, compact=True):
    """Format a Stim Pauli string densely or by its non-identity support."""
    text = str(pauli)
    if not compact:
        return text
    sign, body = text[0], text[1:]
    support = [
        f"{axis}@{site}"
        for site, axis in enumerate(body)
        if axis != "_"
    ]
    return sign + ("I" if not support else " ".join(support))

# Single-qubit Clifford matrices used to localize a signed Pauli string onto one
# qubit for the basis-updating measurement (H, S-dagger, CNOT).
_H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG_MAT = np.array([[1, 0], [0, -1j]], dtype=complex)
_CNOT_MAT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_S_MAT = np.array([[1, 0], [0, 1j]], dtype=complex)

# Populated lazily by ``_two_qubit_clifford_representatives``.  There are 20
# two-qubit Cliffords modulo a Clifford acting independently on each *output*
# qubit.  Such output-local Cliffords leave the Schmidt spectrum invariant, so
# testing one representative per coset finds the same best entanglement score
# as testing all 11,520 two-qubit Cliffords.
_TWO_Q_CLIFFORD_REPS = None


def _normalize_event_name(name):
    """Normalize a named stream event for matching."""
    return str(name).replace("-", "_").strip().lower()


def _normalize_sites(where):
    """Return ``where`` as a non-empty tuple of integer qubit indices."""
    if isinstance(where, Integral):
        return (int(where),)
    try:
        sites = tuple(int(site) for site in where)
    except TypeError as exc:
        raise TypeError("where must be an integer or a sequence of integers.") from exc
    if not sites:
        raise ValueError("where must contain at least one qubit.")
    return sites


def _normalize_measurement_order(order, *, count, targets=None):
    """Normalize a batch measurement order without touching the MPS."""
    if isinstance(order, str) or order is None:
        key = "min_span" if order is None else _normalize_event_name(order)
        if key in {"auto", "span", "min_span", "shortest"}:
            return "min_span"
        if key in {"input", "given", "original"}:
            return "input"
        raise ValueError(
            "measurement order must be 'min_span', 'input', or an explicit "
            "permutation of the batch entries."
        )
    try:
        requested = tuple(int(index) for index in order)
    except TypeError as exc:
        raise TypeError(
            "measurement order must be a supported string or an entry permutation."
        ) from exc
    if len(requested) != int(count) or len(set(requested)) != int(count):
        raise ValueError(
            "an explicit measurement order must be a permutation of the batch."
        )
    if set(requested) == set(range(int(count))):
        return requested
    if targets is not None and len(set(targets)) == int(count):
        target_to_index = {int(target): index for index, target in enumerate(targets)}
        if set(requested) == set(target_to_index):
            return tuple(target_to_index[target] for target in requested)
    raise ValueError(
        "an explicit measurement order must contain batch indices or each "
        "target qubit exactly once."
    )


def _unique_ordered(items):
    """Return items with duplicates removed while preserving first occurrence."""
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


def _layout_angle_weight(theta):
    """Bound an angle-derived layout weight to a simple non-negative scalar."""
    try:
        angle = abs(float(theta))
    except (TypeError, ValueError):
        return 1.0
    return min(1.0, max(0.0, angle)) if np.isfinite(angle) else 1.0


def _operator_schmidt_tail_weight(theta):
    """Return the non-leading Schmidt-weight fraction of a Pauli rotation.

    A Pauli rotation has two operator-Schmidt branches, ``I`` and ``P``.
    The returned value is zero for a product operator and reaches one half
    for the maximally balanced two-branch case.  The layout event keeps a
    unit baseline separately so weak rotations still contribute locality
    pressure while strongly operator-entangling rotations receive priority.
    """
    try:
        theta = float(theta)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(theta):
        return 0.0
    weights = np.asarray(
        [np.cos(theta / 2.0) ** 2, np.sin(theta / 2.0) ** 2],
        dtype=float,
    )
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float((total - weights.max()) / total)


def _dense_operator_schmidt_layout_weight(gate, n_qubits):
    """Return a baseline-plus-tail weight for a small dense operator.

    The two-qubit case is evaluated exactly from the operator-Schmidt
    singular values. Wider matrices retain the unit baseline and are handled
    by their frame supports, avoiding a potentially large dense reshape in
    the static layout pre-pass.
    """
    if int(n_qubits) != 2:
        return 1.0
    try:
        array = np.asarray(ar.to_numpy(gate))
        if array.shape != (4, 4):
            return 1.0
        reshaped = array.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
        singular_values = np.linalg.svd(
            reshaped.reshape(4, 4),
            compute_uv=False,
        )
        weights = np.abs(singular_values) ** 2
        total = float(weights.sum())
        if total <= 0.0:
            return 1.0
        tail = (total - float(weights.max())) / total
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return 1.0
    return 1.0 + float(tail)


def _submpo_operator_layout_weight(mpo):
    """Return an MPO-bond-rank proxy for a coefficient-frame sub-MPO."""
    try:
        max_bond = int(mpo.max_bond())
    except (AttributeError, TypeError, ValueError):
        return 1.0
    if max_bond < 1:
        return 1.0
    # A bond-two Pauli-rotation MPO has one unit of operator cut load. Wider
    # MPOs receive proportionally more priority in the weighted interaction
    # graph, while rank-one operators retain the locality baseline.
    return max(1.0, float(np.log2(max_bond)))


def _is_axis_string(value):
    """Return whether ``value`` is a non-empty X/Y/Z Pauli-basis string."""
    if not isinstance(value, str):
        return False
    axes = [axis for axis in value.upper() if not axis.isspace()]
    return bool(axes) and all(axis in _RESET_FLIP_CLIFFORDS for axis in axes)


def _normalize_pauli_axes(pauli, where, *, event):
    """Return one X/Y/Z axis per site for reset-like events."""
    axes = [axis for axis in str(pauli).upper() if not axis.isspace()]
    if not axes:
        raise ValueError(f"{event} basis must contain at least one Pauli axis.")
    invalid = [axis for axis in axes if axis not in _RESET_FLIP_CLIFFORDS]
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


def _normalize_outcomes(outcome, where, *, event):
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


def _parse_reset_args(params, *, default_axis=None):
    """Parse ``reset`` stream parameters into ``(axes, where)``."""
    if not params:
        raise ValueError('"reset" expects where, optionally with a basis.')
    if default_axis is not None:
        if len(params) != 1:
            raise ValueError("basis-specific reset aliases accept only where.")
        where = _normalize_sites(params[0])
        basis = default_axis
    elif len(params) >= 2 and _is_axis_string(params[0]):
        if len(params) != 2:
            raise ValueError('"reset" accepts only basis and where.')
        basis = params[0]
        where = _normalize_sites(params[1])
    else:
        if len(params) > 2:
            raise ValueError('"reset" accepts where and optional basis only.')
        where = _normalize_sites(params[0])
        basis = params[1] if len(params) == 2 else "Z"
    return _normalize_pauli_axes(basis, where, event="reset"), where


def _parse_measure_reset_args(params, *, default_axis=None):
    """Parse MR stream parameters into ``(axes, where, outcomes, absorb_basis)``."""
    if default_axis is None:
        if len(params) < 2:
            raise ValueError(
                '"measure_reset" expects basis, where, optional outcome, '
                "and optional absorb_basis."
            )
        basis = params[0]
        where = _normalize_sites(params[1])
        outcome = params[2] if len(params) > 2 else None
        absorb = bool(params[3]) if len(params) > 3 else False
        if len(params) > 4:
            raise ValueError('"measure_reset" accepts at most four arguments.')
    else:
        if not params:
            raise ValueError("basis-specific MR aliases expect where.")
        where = _normalize_sites(params[0])
        basis = default_axis
        outcome = params[1] if len(params) > 1 else None
        absorb = bool(params[2]) if len(params) > 2 else True
        if len(params) > 3:
            raise ValueError("basis-specific MR aliases accept at most three arguments.")
    axes = _normalize_pauli_axes(basis, where, event="measure_reset")
    outcomes = _normalize_outcomes(outcome, where, event="measure_reset")
    return axes, where, outcomes, absorb


def _normalize_absorb(absorb):
    """Validate and normalize a cap absorption direction."""
    direction = str(absorb).strip().lower()
    if direction not in {"left", "right"}:
        raise ValueError("cap absorb direction must be 'left' or 'right'.")
    return direction


def _cnot_matrix(control: int, target: int) -> np.ndarray:
    """Return the big-endian two-qubit CNOT matrix for local sites 0 and 1."""
    gate = np.zeros((4, 4), dtype=complex)
    for x in range(4):
        bits = [(x >> 1) & 1, x & 1]
        bits[target] ^= bits[control]
        y = (bits[0] << 1) | bits[1]
        gate[y, x] = 1.0
    return gate


def _two_qubit_tableau_unitary(tableau) -> np.ndarray:
    """Synthesize an exact NumPy unitary for a two-qubit stim tableau.

    ``Tableau.to_unitary_matrix`` currently returns ``complex64``.  The local
    gauge sweep can run repeatedly, so replay its elimination circuit (H, S,
    and CX only) using the exact double-precision matrices instead.
    """
    unitary = np.eye(4, dtype=complex)
    for instruction in tableau.to_circuit("elimination"):
        name = instruction.name
        targets = [target.value for target in instruction.targets_copy()]
        if name == "H":
            for target in targets:
                gate = np.kron(_H_MAT, _I2) if target == 0 else np.kron(_I2, _H_MAT)
                unitary = gate @ unitary
        elif name == "S":
            for target in targets:
                gate = np.kron(_S_MAT, _I2) if target == 0 else np.kron(_I2, _S_MAT)
                unitary = gate @ unitary
        elif name == "CX":
            if len(targets) % 2:
                raise ValueError("stim emitted a CX instruction with an odd target count.")
            for control, target in zip(targets[::2], targets[1::2]):
                unitary = _cnot_matrix(control, target) @ unitary
        else:  # pragma: no cover - stim's documented elimination basis is H/S/CX
            raise ValueError(f"Unsupported tableau-elimination gate {name!r}.")
    return unitary


_I2 = np.eye(2, dtype=complex)


def _two_qubit_clifford_representatives():
    """Return 20 ``(stim.Tableau, unitary)`` entanglement representatives.

    The representatives are left cosets of the local-Clifford subgroup.  If
    ``D`` is a representative and ``L`` is local, ``L D`` has the same
    Schmidt spectrum across the two sites as ``D``.  This keeps a sweep small
    enough to use at every selected MPS bond while retaining the complete
    two-qubit Clifford search space for the chosen objective.
    """
    global _TWO_Q_CLIFFORD_REPS
    if _TWO_Q_CLIFFORD_REPS is not None:
        return _TWO_Q_CLIFFORD_REPS

    import stim

    one_qubit = tuple(stim.Tableau.iter_all(1))
    local = []
    for first in one_qubit:
        for second in one_qubit:
            tableau = stim.Tableau(2)
            tableau.append(first, [0])
            tableau.append(second, [1])
            local.append(tableau)

    unseen = {str(tableau): tableau for tableau in stim.Tableau.iter_all(2)}
    identity = stim.Tableau(2)
    representatives = []
    while unseen:
        # Keep I first: a bond that cannot improve avoids needless gate work.
        tableau = unseen.pop(str(identity), None)
        if tableau is None:
            _, tableau = unseen.popitem()
        representatives.append((tableau, _two_qubit_tableau_unitary(tableau)))
        # ``D.then(L)`` is the circuit D followed by local L, i.e. L D.
        for local_tableau in local:
            unseen.pop(str(tableau.then(local_tableau)), None)

    if len(representatives) != 20:  # pragma: no cover - guards stim API changes
        raise RuntimeError(
            "Expected 20 two-qubit Clifford local-equivalence representatives, "
            f"got {len(representatives)}."
        )
    _TWO_Q_CLIFFORD_REPS = tuple(representatives)
    return _TWO_Q_CLIFFORD_REPS


def _localizing_clifford(terms, n, *, site_position=None):
    """Return ``(ops, v_tableau, pivot)`` for a Clifford ``V`` with ``V M V^dag = +/-Z_k``.

    ``terms`` maps ``site -> 'X'/'Y'/'Z'`` (the support of the signed Pauli ``M``
    on the coefficient qubits).  ``ops`` is a list of ``(name, targets)`` gates
    applied to ``|nu>`` in order (``'h'``, ``'sdg'``, ``'cnot'``); ``v_tableau``
    is the matching :class:`stim.Tableau`; ``pivot`` is the target qubit ``k``.
    Single-qubit axes are rotated to ``Z`` (``X`` via ``H``; ``Y`` via ``S^dag``
    then ``H``) and a CNOT ladder (control ``j``, target ``k``) merges every
    ``Z_j`` onto the pivot ``Z_k``.
    """
    import stim

    if site_position is None:
        site_position = int
    support = sorted(terms, key=lambda site: (site_position(site), int(site)))
    # Pivot = median of the support: the CNOT ladder swaps every other support
    # site next to the pivot, so the median minimises the total MPS swap distance
    # (sum_j |j - pivot|) versus using an endpoint.
    pivot = support[len(support) // 2]
    ops = []
    for j in support:
        axis = terms[j]
        if axis == "X":
            ops.append(("h", (j,)))
        elif axis == "Y":
            ops.append(("sdg", (j,)))  # S^dag then H maps Y -> Z
            ops.append(("h", (j,)))
        # 'Z' needs no single-qubit rotation
    # Merge nearest support sites first so each swap+split spans the shortest gap.
    pivot_pos = site_position(pivot)
    for j in sorted(
        (s for s in support if s != pivot),
        key=lambda s: (abs(site_position(s) - pivot_pos), site_position(s), int(s)),
    ):
        ops.append(("cnot", (j, pivot)))  # control j, target pivot: merge Z_j -> Z_k
    vsim = stim.TableauSimulator()
    vsim.set_num_qubits(n)
    for name, targ in ops:
        getattr(vsim, "s_dag" if name == "sdg" else name)(*targ)
    v_tableau = vsim.current_inverse_tableau().inverse()
    return ops, v_tableau, pivot


class MpsStabOptimizer:
    """Replay a gate stream against a stabilizer + MPS (STN) state.

    Parameters
    ----------
    state : STNState | int | qtn.MatrixProductState
        An existing :class:`STNState`, or an integer number of qubits (a fresh
        ``|0...0>`` state is created). Passing a qubit MPS directly wraps it
        with the identity tableau, so the initial representation is
        ``|psi> = I |p>`` in the ordinary computational basis.
    gates : stream | None
        Optional initial gate stream (see module docstring for entry forms).
    chi : int | None
        Maximum bond dimension for ``|nu>`` truncation.  ``None`` keeps the
        evolution exact (no truncation).
    cutoff : float
        Singular-value cutoff used when truncating ``|nu>``.
    operator_tol : float | None
        Absolute tolerance for pruning Pauli coefficients when decomposing an
        explicit dense operator. ``None`` chooses a matrix-scale-relative
        tolerance from the operator dtype. This is independent of ``cutoff``.
    max_pauli_decomposition_qubits : int | None
        Maximum qubit count for the fallback dense-matrix Pauli decomposition.
        The default, ``2``, bounds its ``4**k`` candidate-term cost. ``None``
        disables the guard. Clifford matrices and one-qubit unitary matrices
        use their specialized paths and do not consume this budget.
    max_pauli_terms : int | None
        Maximum number of retained Pauli terms for a general dense matrix.
        This second guard makes explicit three- or four-qubit opt-ins safe by
        bounding the width of the coefficient-frame operator sum.
    max_dense_cap_qubits : int | None
        Maximum register size for a length-shortening physical ``cap`` event.
        ``cap`` contracts the dense physical state and rebuilds an identity-frame
        coefficient MPS, so this guard keeps the exponential fallback explicit.
        ``None`` opts out of the guard.
    stabilize_unitary : bool
        If ``False`` (default), retain the raw norm loss after each unitary
        compression so it remains visible in the live coefficient MPS. If
        ``True``, restore the pre-compression working norm after recording the
        same local and cumulative compression fidelities. Stabilization changes
        only the live scale; it never erases the diagnostic ledger.
    exact_cooling : bool
        If ``True`` (default), recognize the constructive exact-cooling case
        before a multi-site Pauli rotation. A usable separable stabilizer site
        keeps the coefficient MPS bond unchanged by moving the controlled-Pauli
        part of the update into the Clifford tableau.
    seed : int | None
        Seed for the random-number generator used by measurement sampling.
    dtype : str
        Coefficient-state dtype (used when creating a state from ``n``).
    to_backend : callable | None
        Optional array converter (e.g. ``pepsy.backend_torch(...)`` /
        ``pepsy.backend_cupy(...)`` / ``pepsy.backend_jax(...)``).  When given,
        the coefficient MPS ``|nu>`` and every gate/MPO applied to it are placed
        on that backend, so the heavy MPS contractions run on GPU/torch/JAX.  The
        stim tableau (classical Clifford tracking) stays on the CPU.
    inplace : bool
        If ``True`` (default) mutate the provided ``state``; otherwise operate
        on a copy.
    layout : str | mapping | sequence | None
        Optional static STN frame layout to install after queuing ``gates`` and
        before replay. ``"auto"`` dry-runs the queued tableau/frame supports and
        chooses a coefficient-MPS order; a sequence is interpreted as an
        explicit position-to-logical-site order. Layout installation is exact
        only while the coefficient MPS has ``max_bond() == 1``.
    layout_kwargs : mapping | None
        Extra keyword arguments forwarded to :meth:`current_frame_layout` for
        string layout requests.
    layout_report : bool
        Print a concise before/after frame-layout report when a finder plan is
        installed.
    mode : {"direct", "dm", "zipup", "src", "fit-*", "dmrg", "dmrg1", "dmrg2", "dmrg3", "svd", "swap", "perm", "exact"}
        Compression backend for coefficient-MPS updates. Native compression
        names are used directly, for example ``"direct"``, ``"zipup"``, or
        ``"src"``; the ``"*-first"`` and ``"*-oversample"`` variants are
        available as well. The DMRG modes use local FIT on the coefficient
        target; ``fit_init_strategy`` controls their disposable initial guess.
        On dense backends, DMRG retains the exact coefficient sub-MPO as a
        tagged lazy FIT target layer after canonicalizing the active MPS window;
        Symmray and fermionic routes use the materialized backend-safe target.
        Historical ``"quimb-*"`` and ``"mpo-*"`` spellings remain accepted as
        deprecated aliases.
        ``"svd"``, ``"swap"``, and ``"perm"`` remain compatibility aliases
        for the historical direct coefficient-MPO path. ``"exact"`` forces
        ``chi=None`` and keeps the coefficient MPS lossless up to ``cutoff``.
        Clifford gates remain tableau-only in every mode.
    fit_init_strategy : {"auto", "direct", "random", "random_expand", "guess-<method>"}
        Disposable FIT initialization for dense DMRG windows. The default
        ``"guess-src"`` selects the SRC warm-up before active bonds reach their
        attainable ``chi`` ceilings and continues to prepare the fixed-rank
        one-site phase after expansion. ``"auto"`` resolves to the same
        ``"guess-src"`` policy.
        ``"guess_<method>"`` remains accepted as a compatibility spelling;
        ``"svd_guess"`` is an alias for ``"guess-direct"``. Native Symmray
        and fermionic paths retain their direct sector-aware initialization.
    fit_init_rand_strength : float
        Perturbation strength for ``"random"`` and ``"random_expand"``.
    fit_init_seed : int
        Deterministic seed for randomized FIT guesses.
    compression_seed : int | None
        Seed forwarded to randomized native compression methods. This is kept
        separate from ``seed``, which controls STN measurement sampling.

    Attributes
    ----------
    state : STNState
        The evolving stabilizer tensor-network state.
    infidelities : list[float]
        Cumulative ``infidelity`` samples from compressed updates. Dense
        non-unitary entries use the local ``G^dagger G`` target norm; projective
        boundaries are additionally represented in :attr:`norm_events`.
    norm_events : list[NormEventRecord]
        Segment-boundary snapshots made immediately before projective
        measurement/reset normalization. These preserve the pre-collapse
        truncation proxy separately from the Born branch probability.
    bond_history : list[int]
        ``|nu>`` max bond dimension after each applied entry.
    exact_cooling_events : list[dict]
        Successful constructive exact-cooling updates. Greedy Clifford cooling
        remains the explicit :meth:`disentangle_cliffords` operation.
    immediate_projection_events : list[ImmediateProjectionRecord]
        Per-gadget projection diagnostics from the most recent
        :meth:`run_with_injection` call.
    last_immediate_injection_report : ImmediateInjectionReport | None
        Aggregate projection timing and peak-bond diagnostics from the most
        recent :meth:`run_with_injection` call.
    deferred_projection_events : list[DeferredProjectionRecord]
        Per-ancilla diagnostics from the most recent deferred magic-state
        injection run, in the order the magic register was projected.
    last_deferred_injection_report : DeferredInjectionReport | None
        Aggregate timing and peak-bond diagnostics from the most recent
        :meth:`run_with_deferred_injection` call.
    measurements : list[MeasurementRecord]
        Recorded ``(pauli, where, outcome)`` for each measurement performed.
    stim_plan : pepsy.StimCircuitPlan | None
        Compiled source circuit retained by :meth:`from_stim`; otherwise
        ``None``.
    stim_sample : pepsy.StimNoiseSample | None
        Raw sampled source trajectory retained by :meth:`from_stim`; otherwise
        ``None``. Its fault and herald records are unchanged by any
        ``stream_transform`` supplied to that constructor.
    """

    def __init__(
        self,
        state,
        gates=None,
        *,
        chi: Optional[int] = None,
        cutoff: float = 1e-12,
        operator_tol: Optional[float] = None,
        max_pauli_decomposition_qubits: Optional[int] = (
            DEFAULT_MPS_STAB_MAX_PAULI_DECOMPOSITION_QUBITS
        ),
        max_pauli_terms: Optional[int] = 256,
        max_dense_cap_qubits: Optional[int] = 10,
        exact_cooling: bool = True,
        stabilize_unitary: bool = False,
        seed: Optional[int] = None,
        dtype: str = "complex128",
        to_backend=None,
        inplace: bool = True,
        layout=None,
        layout_kwargs=None,
        layout_report: bool = True,
        mode: str = "direct",
        fit_init_strategy: str = "guess-src",
        fit_init_rand_strength: float = 1.0e-1,
        fit_init_seed: int = 0,
        compression_seed: Optional[int] = None,
    ):
        if isinstance(state, STNState):
            self.state = state if inplace else state.copy()
        elif isinstance(state, Integral):
            self.state = STNState(int(state), dtype=dtype)
        elif isinstance(state, qtn.MatrixProductState):
            import stim

            p = state if inplace else state.copy()
            sim = stim.TableauSimulator()
            sim.set_num_qubits(int(p.L))
            self.state = STNState.from_tableau_and_state(sim, p, dtype=dtype)
        else:
            raise TypeError(
                "state must be an STNState, an integer qubit count, or a "
                "qubit MatrixProductState."
            )

        self.mode = self._normalize_mode(mode)
        self.chi = None if chi is None else int(chi)
        if self.mode == "exact":
            self.chi = None
        self.fit_init_strategy = self._normalize_fit_init_strategy(
            fit_init_strategy
        )
        try:
            self.fit_init_rand_strength = float(fit_init_rand_strength)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fit_init_rand_strength must be finite and non-negative."
            ) from exc
        if (
            not np.isfinite(self.fit_init_rand_strength)
            or self.fit_init_rand_strength < 0.0
        ):
            raise ValueError(
                "fit_init_rand_strength must be finite and non-negative."
            )
        if isinstance(fit_init_seed, bool) or not isinstance(fit_init_seed, Integral):
            raise ValueError("fit_init_seed must be a non-negative integer.")
        self.fit_init_seed = int(fit_init_seed)
        if self.fit_init_seed < 0:
            raise ValueError("fit_init_seed must be a non-negative integer.")
        if compression_seed is not None:
            if isinstance(compression_seed, bool) or not isinstance(
                compression_seed, Integral
            ):
                raise ValueError("compression_seed must be a non-negative integer or None.")
            compression_seed = int(compression_seed)
            if compression_seed < 0:
                raise ValueError(
                    "compression_seed must be a non-negative integer or None."
                )
        self.compression_seed = compression_seed
        self.cutoff = float(cutoff)
        if operator_tol is not None:
            operator_tol = float(operator_tol)
            if not np.isfinite(operator_tol) or operator_tol < 0.0:
                raise ValueError(
                    "operator_tol must be finite and nonnegative, "
                    f"got {operator_tol!r}."
                )
        self.operator_tol = operator_tol
        if max_pauli_decomposition_qubits is not None:
            if (
                isinstance(max_pauli_decomposition_qubits, bool)
                or not isinstance(max_pauli_decomposition_qubits, Integral)
            ):
                raise TypeError(
                    "max_pauli_decomposition_qubits must be an integer or None."
                )
            max_pauli_decomposition_qubits = int(max_pauli_decomposition_qubits)
            if max_pauli_decomposition_qubits < 0:
                raise ValueError(
                    "max_pauli_decomposition_qubits must be nonnegative or None, "
                    f"got {max_pauli_decomposition_qubits}."
                )
        self.max_pauli_decomposition_qubits = max_pauli_decomposition_qubits
        if max_pauli_terms is not None:
            if (
                isinstance(max_pauli_terms, bool)
                or not isinstance(max_pauli_terms, Integral)
            ):
                raise TypeError("max_pauli_terms must be an integer or None.")
            max_pauli_terms = int(max_pauli_terms)
            if max_pauli_terms < 1:
                raise ValueError("max_pauli_terms must be positive or None.")
        self.max_pauli_terms = max_pauli_terms
        if max_dense_cap_qubits is not None:
            if (
                isinstance(max_dense_cap_qubits, bool)
                or not isinstance(max_dense_cap_qubits, Integral)
            ):
                raise TypeError("max_dense_cap_qubits must be an integer or None.")
            max_dense_cap_qubits = int(max_dense_cap_qubits)
            if max_dense_cap_qubits < 0:
                raise ValueError(
                    "max_dense_cap_qubits must be nonnegative or None, "
                    f"got {max_dense_cap_qubits}."
                )
        self.max_dense_cap_qubits = max_dense_cap_qubits
        self.exact_cooling = bool(exact_cooling)
        self.stabilize_unitary = bool(stabilize_unitary)
        self._infidelity_valid = True
        self._current_infidelity = 0.0
        self._compression_segment_log_survival = 0.0
        self._norm_segment_open = False
        self._norm_log_survival = 0.0
        self.dtype = self.state.dtype
        self._rng = np.random.default_rng(seed)
        self._initial_mps_length = int(self.state.n)
        self._mps_length_history = [self._initial_mps_length]
        self.cap_history = []
        self.logical_order = list(range(self.state.n))
        self._logical_to_mps = {q: q for q in self.logical_order}
        self.layout_plan = None
        self.last_layout_plan = None
        self.last_measurement_schedule = ()

        self.to_backend = to_backend
        self._explicit_backend_converter = to_backend
        self._bk_cache: dict = {}
        self._backend_conversion_warnings = set()
        self._backend_signature = None
        self._backend_converter = None
        self._clifford_rot_cache: dict = {}
        self._localizer_cache: dict = {}
        if to_backend is not None:
            # Place the coefficient MPS |nu> on the requested backend. User
            # stream gates/MPOs must be prepared explicitly; _bk* conversion
            # remains for internal generated operators and dtype alignment.
            self.state.p.apply_to_arrays(to_backend)
        if self._backend_signature is None:
            self.backend_info()

        self._queue: List[object] = []
        self._gate_stream = ()
        self._trajectory_plan = None
        self._has_trajectory_events = False
        self._last_run_timing = None
        self._last_fit_diagnostics = None
        self._dmrg1_one_site_locked = False
        self._quality_checks: list[dict] = []
        self.infidelities: List[float] = []
        self._nonunitary_infidelities: List[float] = []
        # Per-update norm ratios are separate from projective boundary events.
        # ``norm_events`` remains the historical boundary ledger; this list
        # makes the ordinary-MPS local-vs-cumulative API available for every
        # compressed unitary coefficient update as well.
        self._compression_norm_events: List[dict] = []
        self.norm_events: List[NormEventRecord] = []
        self.bond_history: List[int] = [self.state.max_bond()]
        self.exact_cooling_events: List[dict] = []
        self.immediate_projection_events: List[ImmediateProjectionRecord] = []
        self.last_immediate_injection_report = None
        self._last_injection_projection_event = None
        self.deferred_projection_events: List[DeferredProjectionRecord] = []
        self.last_deferred_injection_report = None
        self.measurements: List[MeasurementRecord] = []
        # These are populated by ``from_stim``. Keeping the raw compiled
        # trajectory separate from the queued stream makes optional caller-side
        # stream transforms explicit and reproducible.
        self.stim_plan = None
        self.stim_sample = None
        if gates is not None:
            self.add_gates(gates)
        if layout is not None and layout is not False:
            self.apply_layout(
                layout,
                layout_kwargs=layout_kwargs,
                layout_report=layout_report,
            )
        # Shot replay, like ``MpsOptimizer``, always starts from the state that
        # existed when this optimizer was constructed. Capture it after a
        # static layout has been installed so logical labels remain valid.
        self._initial_state = self.state.copy()

    # ------------------------------------------------------------------ #
    # Initial-state constructors (product / GHZ / user tableau+MPS)
    # ------------------------------------------------------------------ #
    _DMRG_MODES = frozenset({"dmrg", "dmrg1", "dmrg2", "dmrg3"})
    _CANONICAL_MPO_MODES = frozenset(_MPO_COMPRESSION_METHODS)
    _LEGACY_MODE_NAMES = frozenset({"quimb", "mpo"})
    _LEGACY_MODE_PREFIXES = ("quimb-", "mpo-")
    _ALLOWED_MODES = frozenset(
        _DMRG_MODES
        | _CANONICAL_MPO_MODES
        | {"svd", "swap", "perm", "exact"}
        | _LEGACY_MODE_NAMES
        | {
            f"{prefix}{method}"
            for prefix in _LEGACY_MODE_PREFIXES
            for method in _MPO_COMPRESSION_METHODS
        }
    )

    @classmethod
    def _normalize_mode(cls, mode):
        """Validate and normalize the coefficient-MPS compression mode."""
        mode = str(mode).strip().lower()
        if mode == "fit":
            mode = "dmrg"
        if mode not in cls._ALLOWED_MODES:
            allowed = (
                "a native method such as direct, zipup, or src; "
                "dmrg*, svd, swap, perm, or exact"
            )
            raise ValueError(
                f"Unknown MpsStabOptimizer mode {mode!r}; choose one of {allowed}."
            )
        if mode in cls._LEGACY_MODE_NAMES or mode.startswith(cls._LEGACY_MODE_PREFIXES):
            warnings.warn(
                f"mode={mode!r} is deprecated; use the bare native compression "
                "method name (for example mode='src' or mode='direct').",
                DeprecationWarning,
                stacklevel=3,
            )
        return mode

    @classmethod
    def _is_quimb_mode(cls, mode):
        """Return whether ``mode`` selects native Quimb compression."""
        mode = str(mode).strip().lower()
        return (
            mode in cls._CANONICAL_MPO_MODES
            or mode in cls._LEGACY_MODE_NAMES
            or mode.startswith(cls._LEGACY_MODE_PREFIXES)
        )

    @classmethod
    def _mode_quimb_method(cls, mode):
        """Return the Quimb compression method encoded by ``mode``."""
        mode = str(mode).strip().lower()
        if mode in cls._CANONICAL_MPO_MODES:
            return mode
        if mode in cls._LEGACY_MODE_NAMES:
            return "direct"
        for prefix in cls._LEGACY_MODE_PREFIXES:
            if mode.startswith(prefix):
                return cls._normalize_quimb_method(mode[len(prefix) :])
        return "direct"

    @classmethod
    def _normalize_quimb_method(cls, method):
        """Validate and normalize a native Quimb compression method."""
        method = str(method).strip().lower()
        if method not in _MPO_COMPRESSION_METHODS:
            allowed = ", ".join(sorted(_MPO_COMPRESSION_METHODS))
            raise ValueError(f"Unknown Quimb compression method {method!r}; choose one of {allowed}.")
        return method

    @classmethod
    def _normalize_fit_init_strategy(cls, strategy):
        """Validate and normalize the disposable DMRG FIT initialization."""
        strategy = str(strategy).strip().lower()
        if strategy.startswith("guess-"):
            strategy = "guess_" + strategy[len("guess-") :]
        if strategy not in _FIT_INIT_STRATEGIES:
            allowed = "auto, direct, random, random_expand, guess-<method>"
            raise ValueError(
                "fit_init_strategy must be one of "
                f"{allowed}; got {strategy!r}."
            )
        return strategy

    @staticmethod
    def _resolved_fit_init_strategy(strategy):
        """Resolve strategy aliases to the canonical FIT warm-start policy."""
        if strategy == "auto":
            return "guess_src"
        if strategy == "svd_guess":
            return "guess_direct"
        return strategy

    @classmethod
    def from_bits(cls, bits, **kwargs) -> "MpsStabOptimizer":
        """Start from a computational-basis product state (``bits`` = str or 0/1 seq)."""
        dtype = kwargs.pop("dtype", "complex128")
        return cls(STNState.from_bits(bits, dtype=dtype), **kwargs)

    @classmethod
    def ghz(cls, n: int, **kwargs) -> "MpsStabOptimizer":
        """Start from the ``n``-qubit GHZ state (a stabilizer state, chi=1)."""
        dtype = kwargs.pop("dtype", "complex128")
        return cls(STNState.ghz(n, dtype=dtype), **kwargs)

    @classmethod
    def from_tableau_and_state(cls, sim, p, **kwargs) -> "MpsStabOptimizer":
        """Start from a user stim tableau ``sim`` and coefficient MPS ``p``."""
        dtype = kwargs.pop("dtype", "complex128")
        return cls(STNState.from_tableau_and_state(sim, p, dtype=dtype), **kwargs)

    # Backward-compatible alias.
    from_tableau_and_nu = from_tableau_and_state

    @classmethod
    def from_mps(cls, p, **kwargs) -> "MpsStabOptimizer":
        """Start from a qubit MPS in the ordinary computational basis."""
        return cls(p, **kwargs)

    @classmethod
    def from_stim(
        cls, circuit, *, seed: Optional[int] = None, stream_transform=None, **kwargs
    ) -> "MpsStabOptimizer":
        """Build one STN trajectory directly from a Stim circuit.

        The circuit is compiled once by :func:`pepsy.compile_stim_circuit`, then
        every native Stim stochastic Pauli channel is sampled once using
        ``seed``. The same seed initializes the later measurement sampler on
        the returned STN (with an independent random-number generator). The
        inferred Stim qubit count creates the initial ``|0...0>`` STN state,
        and the resulting physical Pepsy stream is queued for a later
        :meth:`run`.

        ``stream_transform``, when supplied, receives the immutable sampled
        gate-stream tuple and must return the replacement Pepsy gate stream.
        It is useful for an external circuit producer to insert physical
        non-Stim gates or to omit a terminal readout while preserving Pepsy's
        Stim parsing and noise sampling. The raw :attr:`stim_plan` and
        :attr:`stim_sample` remain available on the returned simulator for
        reproducibility and fault/herald inspection.

        ``state`` and ``gates`` are intentionally not accepted in ``kwargs``:
        this constructor derives the initial register and queued stream from
        the Stim circuit. Use :meth:`from_tableau_and_state` or the regular
        constructor for a non-default initial STN state.
        """
        if "state" in kwargs or "gates" in kwargs:
            raise TypeError(
                "MpsStabOptimizer.from_stim derives state and gates from the "
                "Stim circuit; use stream_transform for stream edits."
            )
        if stream_transform is not None and not callable(stream_transform):
            raise TypeError("stream_transform must be callable or None.")

        # Local imports keep Stim optional and avoid coupling the STN core to
        # the generic trajectory module during ordinary construction.
        from ..noise import (
            _stream_on_optimizer_backend,
            compile_stim_circuit,
            sample_stim_circuit,
        )

        plan = compile_stim_circuit(circuit)
        sample = sample_stim_circuit(plan, seed=seed)
        gates = (
            sample.gate_stream
            if stream_transform is None
            else stream_transform(sample.gate_stream)
        )
        # Stim emits NumPy matrices. Convert this library-generated stream once
        # before it crosses the strict user-stream boundary.
        optimizer = cls(plan.num_qubits, seed=seed, **kwargs)
        optimizer.set_gates(_stream_on_optimizer_backend(gates, optimizer))
        optimizer.stim_plan = plan
        optimizer.stim_sample = sample
        return optimizer

    # ------------------------------------------------------------------ #
    # Properties / queue management
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return self.state.n

    @property
    def L_eff(self) -> int:
        """Return the current effective coefficient-MPS register length."""
        return int(self.state.n)

    @property
    def effective_mps_length(self) -> int:
        """Descriptive alias for :attr:`L_eff`."""
        return self.L_eff

    def mps_length_diagnostics(self):
        """Return the active-register length and dynamic-cap ledger."""
        history = tuple(int(length) for length in self._mps_length_history)
        return {
            "initial_length": int(self._initial_mps_length),
            "peak_length": int(max(history, default=0)),
            "minimum_length": int(min(history, default=0)),
            "L_eff": int(self.L_eff),
            "effective_length": int(self.L_eff),
            "removed_sites": int(self._initial_mps_length - self.L_eff),
            "caps": len(self.cap_history),
            "length_history": history,
            "cap_events": deepcopy(self.cap_history),
        }

    @property
    def p(self):
        """The coefficient MPS (the paper's ``|nu>``), matching ``MpsOptimizer.p``."""
        return self.state.p

    @property
    def nu(self):
        """Alias for :attr:`p`."""
        return self.state.p

    def set_gates(self, gates) -> "MpsStabOptimizer":
        """Replace the queued gate stream."""
        entries = self._as_entries(gates)
        self._install_stream_plan(entries)
        return self

    def add_gates(self, gates) -> "MpsStabOptimizer":
        """Append to the queued gate stream."""
        entries = list(self._queue) + self._as_entries(gates)
        self._install_stream_plan(entries)
        return self

    def _install_stream_plan(self, entries) -> None:
        """Install the backend-neutral trajectory plan for a queued stream."""
        from ..noise import (  # pylint: disable=import-outside-toplevel
            compile_trajectory_stream,
        )

        plan = compile_trajectory_stream(tuple(entries))
        self._validate_gate_stream_backend(plan.entries)
        self._trajectory_plan = plan
        self._gate_stream = tuple(plan.entries)
        self._has_trajectory_events = bool(
            plan.has_trajectory_events or plan.has_leakage
        )
        self._queue = list(plan.entries)

    def _validate_gate_stream_backend(self, entries, *, path_prefix="stream"):
        """Require user matrix and MPO entries to match the live MPS backend.

        The check runs once while a compiled stream is installed. Stateful
        trajectory and leakage entries are intentionally skipped: the shared
        runner selects their matrix outcomes and prepares those generated
        payloads on the optimizer backend before installing the replay stream.
        """
        from ..noise import TrajectoryEvent, _leakage_event_parts

        if self._backend_signature is None:
            self.backend_info()
        target_signature = self._backend_signature
        mismatches = []

        def check_entry(entry, path):
            if isinstance(entry, TrajectoryEvent) or _leakage_event_parts(entry):
                return

            conditional = conditional_event_parts(entry)
            if conditional is not None:
                action = conditional[1]["action"]
                check_entry(action, f"{path}.action")
                return

            submpo = submpo_event_parts(entry, normalize_where=True)
            if submpo is not None:
                mpo, _where = submpo
                for tensor_index, tensor in enumerate(
                    getattr(mpo, "tensors", ())
                ):
                    source_signature = infer_backend_signature(tensor.data)
                    if not backend_signatures_compatible(
                        source_signature, target_signature
                    ):
                        mismatches.append(
                            (
                                f"{path}.tensor[{tensor_index}]",
                                "sub-MPO",
                                source_signature,
                            )
                        )
                return

            if (
                isinstance(entry, (tuple, list))
                and len(entry) == 2
                and not isinstance(entry[0], str)
            ):
                source_signature = infer_backend_signature(entry[0])
                if not backend_signatures_compatible(
                    source_signature, target_signature
                ):
                    mismatches.append((path, "gate", source_signature))

        for index, entry in enumerate(entries):
            check_entry(entry, f"{path_prefix}[{index}]")

        if not mismatches:
            return
        details = "; ".join(
            f"{path} ({kind}) has {source!r}"
            for path, kind, source in mismatches[:8]
        )
        if len(mismatches) > 8:
            details += f"; ... and {len(mismatches) - 8} more"
        raise TypeError(
            "MpsStabOptimizer requires every gate and sub-MPO payload to "
            f"match the coefficient-MPS backend/device and required dtype "
            f"{target_signature!r} "
            f"before use; {details}. Prepare each payload explicitly with "
            "the live backend converter before passing it to the gate stream."
        )

    def gate_stream(self):
        """Return the immutable, compiled stream owned by this optimizer."""
        return self._gate_stream

    @property
    def has_trajectory_events(self) -> bool:
        """Whether this optimizer's stream requires the shot runner."""
        return bool(self._has_trajectory_events)

    @staticmethod
    def submpo_event(mpo, where):
        """Return the canonical coefficient-frame sub-MPO event."""
        return ("submpo", mpo, _normalize_sites(where))

    @staticmethod
    def submpo_event_parts(entry, *, normalize_where=False):
        """Return ``(mpo, where)`` for a sub-MPO event."""
        return submpo_event_parts(entry, normalize_where=normalize_where)

    @staticmethod
    def is_submpo_event(entry):
        """Return whether ``entry`` is a sub-MPO event."""
        return is_submpo_event(entry)

    @staticmethod
    def measure_event(
        pauli, where, outcome=None, absorb_basis=None, *, disentangle=None
    ):
        """Build a canonical Pauli-measurement event shared with MPS."""
        where = _normalize_sites(where)
        if absorb_basis is not None or disentangle is not None:
            absorb_basis = _resolve_measurement_disentangle(
                absorb_basis,
                disentangle,
                default=False,
            )
        entry = ("measure", str(pauli), where)
        if absorb_basis is None:
            if outcome is not None:
                entry += (int(outcome),)
        else:
            entry += (None if outcome is None else int(outcome), bool(absorb_basis))
        return entry

    @staticmethod
    def cap_event(where, vec, absorb="left"):
        """Build a canonical physical cap event."""
        sites = _normalize_sites(where)
        if len(sites) != 1:
            raise ValueError("cap event where must reference exactly one site.")
        return (
            "cap",
            int(sites[0]),
            np.asarray(vec, dtype=complex).ravel(),
            _normalize_absorb(absorb),
        )

    @staticmethod
    def reset_event(where, basis="Z"):
        """Build a canonical reset event."""
        where = _normalize_sites(where)
        axes = _normalize_pauli_axes(basis, where, event="reset")
        if all(axis == "Z" for axis in axes):
            return ("reset", where)
        return ("reset", where, "".join(axes))

    @staticmethod
    def measure_reset_event(
        pauli, where, outcome=None, absorb_basis=None, *, disentangle=None
    ):
        """Build a canonical measure-then-reset event."""
        where = _normalize_sites(where)
        axes = _normalize_pauli_axes(pauli, where, event="measure_reset")
        outcomes = _normalize_outcomes(outcome, where, event="measure_reset")
        if absorb_basis is not None or disentangle is not None:
            absorb_basis = _resolve_measurement_disentangle(
                absorb_basis,
                disentangle,
                default=False,
            )
        entry = ("measure_reset", "".join(axes), where)
        value = None if outcome is None else (
            outcomes[0] if len(outcomes) == 1 else outcomes
        )
        if absorb_basis is None:
            if outcome is not None:
                entry += (value,)
        else:
            entry += (value, bool(absorb_basis))
        return entry

    @staticmethod
    def _as_entries(gates) -> List[object]:
        """Normalize a stream into a list of entries."""
        if gates is None:
            return []
        # Keep the import local: noise is an optional higher-level facade and
        # imports ``MpsOptimizer`` itself.
        from ..noise import (  # pylint: disable=import-outside-toplevel
            TrajectoryEvent,
            TrajectoryStreamPlan,
        )

        if isinstance(gates, TrajectoryStreamPlan):
            return list(gates.entries)
        if isinstance(gates, TrajectoryEvent):
            return [gates]
        # A single sub-MPO event (tuple/mapping) is one entry.
        if is_submpo_event(gates):
            return [gates]
        # A single (matrix, where) or named entry vs a list of entries.
        if isinstance(gates, Mapping):
            return [gates]
        if isinstance(gates, (list, tuple)):
            # Heuristic: a list/tuple whose first element is itself an entry
            # (tuple/list/mapping/ndarray-with-where) is a *stream*; otherwise a
            # single named/matrix entry.
            if len(gates) > 0 and _looks_like_stream(gates):
                return list(gates)
            return [gates]
        raise TypeError(f"Unsupported gate stream: {gates!r}")

    def _stream_entry_sites(self, entry) -> Optional[set[int]]:
        """Return physical logical sites touched by one stream entry.

        ``None`` means the entry shape is too opaque for a magic-ancilla
        scheduler to prove that the reserved pool is untouched.
        """
        parts = submpo_event_parts(entry, normalize_where=True)
        if parts is not None:
            _mpo, where = parts
            return set(_normalize_sites(where))
        conditional = conditional_event_parts(entry)
        if conditional is not None:
            return set(_normalize_sites(conditional[2]))

        if not (isinstance(entry, (list, tuple)) and entry):
            return None
        head = entry[0]
        if not isinstance(head, str):
            if len(entry) != 2:
                return None
            return set(_normalize_sites(entry[1]))

        name = _normalize_event_name(head)
        if name == "disentangle":
            if len(entry) <= 1:
                return set(range(self.n))
            if len(entry) > 2:
                return None
            option = entry[1]
            if isinstance(option, Mapping):
                bonds = option.get("bonds")
            elif isinstance(option, Integral):
                bonds = None
            else:
                return None
            if bonds is None:
                return set(range(self.n))
            sites = set()
            for bond in self._disentangle_bonds(bonds, self.n):
                sites.update((int(bond), int(bond) + 1))
            return sites
        if name in _CLIFFORD_NAMES:
            try:
                return {int(site) for site in entry[1:]}
            except (TypeError, ValueError):
                return None
        if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name in (
            "rot", "t", "tdg",
        ):
            try:
                _theta, where, _axes = self._rotation_spec(name, entry[1:])
            except (TypeError, ValueError, IndexError):
                return None
            return set(where)
        if name == "measure":
            if len(entry) < 3:
                return None
            return set(_normalize_sites(entry[2]))
        if name == "reset" or name in _RESET_AXIS_ALIASES:
            try:
                _axes, where = _parse_reset_args(
                    entry[1:],
                    default_axis=_RESET_AXIS_ALIASES.get(name),
                )
            except (TypeError, ValueError, IndexError):
                return None
            return set(where)
        if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
            try:
                _axes, where, _outcomes, _absorb = _parse_measure_reset_args(
                    entry[1:],
                    default_axis=_MR_AXIS_ALIASES.get(name),
                )
            except (TypeError, ValueError, IndexError):
                return None
            return set(where)
        if name == "cap":
            if len(entry) < 3:
                return None
            return set(_normalize_sites(entry[1]))
        return None

    def _validate_magic_ancilla_pool(
        self,
        ancillas,
        *,
        require_nonempty: bool,
    ) -> tuple[int, ...]:
        """Normalize and validate a reserved magic-ancilla pool."""
        try:
            pool = tuple(int(ancilla) for ancilla in ancillas)
        except TypeError as exc:
            raise TypeError("ancillas must be a sequence of qubit indices.") from exc
        if require_nonempty and not pool:
            raise ValueError("magic injection needs at least one ancilla qubit.")
        if len(set(pool)) != len(pool):
            raise ValueError(f"ancillas must be unique, got {pool!r}.")
        invalid = [ancilla for ancilla in pool if not 0 <= ancilla < self.n]
        if invalid:
            raise ValueError(
                f"ancilla index/indices {tuple(invalid)!r} outside qubit range "
                f"[0, {self.n})."
            )
        return pool

    def _assert_magic_ancillas_clean(self, pool, *, tol: float = 1e-9) -> None:
        """Require each reserved magic ancilla to be a clean physical ``|0>``."""
        for ancilla in pool:
            z_exp = self.expectation("Z", ancilla)
            if abs(z_exp - 1.0) > tol:
                raise ValueError(
                    f"magic ancilla {ancilla} must start clean in physical |0> "
                    f"(expected <Z>=+1, got {z_exp:.6g})."
                )

    def _validate_magic_stream_protection(
        self,
        entries,
        specs,
        pool,
        *,
        mode: str,
    ) -> None:
        """Reject stream entries that would act on reserved magic ancillas."""
        pool_set = set(pool)
        if not pool_set:
            return
        for entry, spec in zip(entries, specs):
            if spec is not None:
                data, _phi = spec
                if not 0 <= data < self.n:
                    raise ValueError(
                        f"{mode} injection target qubit {data} is outside "
                        f"qubit range [0, {self.n})."
                    )
                if data in pool_set:
                    raise ValueError(
                        f"{mode} injection target qubit {data} is in the "
                        f"reserved ancilla pool {pool}."
                    )
                continue
            sites = self._stream_entry_sites(entry)
            if sites is None:
                raise ValueError(
                    f"{mode} injection cannot verify that stream entry {entry!r} "
                    "leaves the reserved magic-ancilla pool untouched."
                )
            touched = sorted(pool_set.intersection(sites))
            if touched:
                raise ValueError(
                    f"{mode} injection reserved ancilla(s) {tuple(touched)!r} "
                    f"are touched by ordinary stream entry {entry!r}."
                )

    @classmethod
    def _analysis_matrix_kind(cls, entry) -> str:
        """Classify a dense matrix entry for stream-level advice."""
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            return "opaque"
        try:
            where = _normalize_sites(entry[1])
            gate = _as_gate_matrix(entry[0], len(where))
        except (TypeError, ValueError, IndexError):
            return "opaque"
        if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
            return "opaque"
        dim = gate.shape[0]
        nq = int(round(math.log2(dim))) if dim > 0 else -1
        if nq < 0 or 2 ** nq != dim or len(where) != nq:
            return "opaque"
        if not _is_unitary(gate):
            return "nonunitary_matrix"
        try:
            is_clifford = _tableau_from_exact_unitary(gate) is not None
        except ImportError:
            return "nonclifford_matrix"
        return "clifford_matrix" if is_clifford else "nonclifford_matrix"

    @classmethod
    def _analysis_entry_kind(cls, entry) -> str:
        """Classify one Pepsy stream entry for whole-stream advice."""
        if submpo_event_parts(entry, normalize_where=True) is not None:
            return "submpo"
        if not (isinstance(entry, (list, tuple)) and entry):
            return "opaque"
        head = entry[0]
        if not isinstance(head, str):
            return cls._analysis_matrix_kind(entry)

        name = _normalize_event_name(head)
        if name in _CLIFFORD_NAMES:
            return "clifford"
        if name == "disentangle":
            return "control"
        try:
            if cls._injectable_rz(entry) is not None:
                return "injectable"
        except (IndexError, TypeError, ValueError):
            return "opaque"
        if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name == "rot":
            try:
                theta = float(entry[1])
            except (IndexError, TypeError, ValueError):
                return "opaque"
            return "clifford" if cls._is_clifford_angle(theta) else "nonclifford"
        if name == "measure":
            return "measure"
        if name == "reset" or name in _RESET_AXIS_ALIASES:
            return "reset"
        if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
            return "measure_reset"
        if name == "cap":
            return "cap"
        return "opaque"

    @classmethod
    def _progress_entry_part(cls, entry) -> str:
        """Return a compact progress-bar label for one stream entry."""
        kind = cls._analysis_entry_kind(entry)
        return {
            "injectable": "T",
            "measure": "measurement",
            "measure_reset": "measure_reset",
            "clifford_matrix": "clifford",
            "nonclifford_matrix": "nonclifford",
            "nonunitary_matrix": "nonunitary",
        }.get(kind, kind)

    @classmethod
    def _analysis_entry_sites(cls, entry, n_qubits: Optional[int]) -> Optional[set[int]]:
        """Return touched physical sites for a stream entry, if cheaply known."""
        parts = submpo_event_parts(entry, normalize_where=True)
        if parts is not None:
            _mpo, where = parts
            return set(_normalize_sites(where))

        if not (isinstance(entry, (list, tuple)) and entry):
            return None
        head = entry[0]
        if not isinstance(head, str):
            if len(entry) != 2:
                return None
            return set(_normalize_sites(entry[1]))

        name = _normalize_event_name(head)
        if name == "disentangle":
            if len(entry) <= 1:
                return None if n_qubits is None else set(range(n_qubits))
            if len(entry) > 2:
                return None
            option = entry[1]
            if isinstance(option, Mapping):
                bonds = option.get("bonds")
            elif isinstance(option, Integral):
                bonds = None
            else:
                return None
            if n_qubits is None:
                return None
            sites = set()
            for bond in cls._disentangle_bonds(bonds, n_qubits):
                sites.update((int(bond), int(bond) + 1))
            return sites
        if name in _CLIFFORD_NAMES:
            return {int(site) for site in entry[1:]}
        if name in _ROTATION_AXES:
            return {int(entry[2])}
        if name in _ROTATION_AXES_2Q:
            return {int(entry[2]), int(entry[3])}
        if name in ("t", "tdg"):
            return {int(entry[1])}
        if name == "rot":
            return set(_normalize_sites(entry[3]))
        if name == "measure":
            if len(entry) < 3:
                return None
            return set(_normalize_sites(entry[2]))
        if name == "reset" or name in _RESET_AXIS_ALIASES:
            _axes, where = _parse_reset_args(
                entry[1:],
                default_axis=_RESET_AXIS_ALIASES.get(name),
            )
            return set(where)
        if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
            _axes, where, _outcomes, _absorb = _parse_measure_reset_args(
                entry[1:],
                default_axis=_MR_AXIS_ALIASES.get(name),
            )
            return set(where)
        if name == "cap":
            if len(entry) < 3:
                return None
            return set(_normalize_sites(entry[1]))
        return None

    @classmethod
    def analyze_stream(cls, gates, *, n_qubits: Optional[int] = None) -> StreamAnalysisRecord:
        """Inspect a Pepsy-native gate stream without executing it.

        This is the stream-first companion to the Stim adapter: it accepts the
        same Pepsy entries as :meth:`apply`, counts the design features that
        affect STN settings, and returns a typed mapping-compatible record.
        """
        if n_qubits is not None:
            if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
                raise TypeError("n_qubits must be a nonnegative integer or None.")
            n_qubits = int(n_qubits)
            if n_qubits < 0:
                raise ValueError("n_qubits must be nonnegative.")

        entries = cls._as_entries(gates)
        counts = {
            "clifford": 0,
            "injectable": 0,
            "nonclifford": 0,
            "structural": 0,
            "control": 0,
            "opaque": 0,
            "dense_matrix": 0,
            "unitary_matrix": 0,
            "nonunitary_matrix": 0,
            "submpo": 0,
            "measure": 0,
            "reset": 0,
            "measure_reset": 0,
            "cap": 0,
        }
        touched: set[int] = set()
        unknown_support = 0
        invalid_sites: list[int] = []

        for entry in entries:
            kind = cls._analysis_entry_kind(entry)
            if kind == "clifford" or kind == "clifford_matrix":
                counts["clifford"] += 1
            elif kind == "injectable":
                counts["injectable"] += 1
            elif kind == "nonclifford" or kind == "nonclifford_matrix":
                counts["nonclifford"] += 1
            elif kind in {"measure", "reset", "measure_reset", "cap"}:
                counts["structural"] += 1
            elif kind == "control":
                counts["control"] += 1
            else:
                counts["opaque"] += 1

            if kind in {"clifford_matrix", "nonclifford_matrix", "nonunitary_matrix"}:
                counts["dense_matrix"] += 1
            if kind in {"clifford_matrix", "nonclifford_matrix"}:
                counts["unitary_matrix"] += 1
            if kind == "nonunitary_matrix":
                counts["nonunitary_matrix"] += 1
            if kind == "submpo":
                counts["submpo"] += 1
            if kind == "measure":
                counts["measure"] += 1
            elif kind == "reset":
                counts["reset"] += 1
            elif kind == "measure_reset":
                counts["measure_reset"] += 1
            elif kind == "cap":
                counts["cap"] += 1

            try:
                sites = cls._analysis_entry_sites(entry, n_qubits)
            except (IndexError, TypeError, ValueError):
                sites = None
            if sites is None:
                unknown_support += 1
            else:
                touched.update(sites)
                invalid_sites.extend(site for site in sites if site < 0)

        if invalid_sites:
            raise ValueError(
                f"stream touches negative qubit index/indices "
                f"{tuple(sorted(set(invalid_sites)))!r}."
            )
        max_qubit = max(touched) if touched else None
        if n_qubits is not None and max_qubit is not None and max_qubit >= n_qubits:
            raise ValueError(
                f"stream touches qubit {max_qubit}, outside n_qubits={n_qubits}."
            )
        estimated_qubits = n_qubits if n_qubits is not None else (
            None if max_qubit is None else max_qubit + 1
        )

        warnings = []
        if unknown_support:
            warnings.append(
                f"{unknown_support} stream entry/entries have unknown qubit support."
            )
        if counts["opaque"]:
            warnings.append(
                "Opaque entries cannot be fully priced by the advisor; validate "
                "them with small exact runs."
            )
        if counts["dense_matrix"]:
            warnings.append(
                "Dense matrix entries use classification and possibly Pauli "
                "decomposition; keep them few-qubit or decompose into named gates."
            )
        if counts["nonunitary_matrix"] or counts["submpo"]:
            warnings.append(
                "Non-unitary matrices and coefficient-frame sub-MPOs can change "
                "normalization physically and suspend the unitary norm-loss proxy."
            )
        if counts["nonclifford"] and counts["injectable"]:
            warnings.append(
                "Only the T-family subset is injectable; other non-Clifford work "
                "stays on the direct STN path."
            )
        if counts["cap"]:
            warnings.append(
                "A cap changes the qubit/MPS length and disables static "
                "stream-layout assumptions past the cap."
            )

        nonmagic_work = counts["nonclifford"] + counts["opaque"]
        is_clifford_t_like = (
            counts["injectable"] > 0
            and counts["nonclifford"] == 0
            and counts["opaque"] == 0
        )
        is_clifford_only = (
            counts["injectable"] == 0
            and nonmagic_work == 0
        )

        return StreamAnalysisRecord(
            total_entries=int(len(entries)),
            n_qubits=n_qubits,
            estimated_qubits=estimated_qubits,
            touched_qubits=tuple(sorted(touched)),
            max_qubit=None if max_qubit is None else int(max_qubit),
            clifford_entries=int(counts["clifford"]),
            injectable_entries=int(counts["injectable"]),
            other_nonclifford_entries=int(counts["nonclifford"]),
            structural_entries=int(counts["structural"]),
            control_entries=int(counts["control"]),
            opaque_entries=int(counts["opaque"]),
            dense_matrix_entries=int(counts["dense_matrix"]),
            unitary_matrix_entries=int(counts["unitary_matrix"]),
            nonunitary_matrix_entries=int(counts["nonunitary_matrix"]),
            submpo_entries=int(counts["submpo"]),
            measurement_entries=int(counts["measure"]),
            reset_entries=int(counts["reset"]),
            measure_reset_entries=int(counts["measure_reset"]),
            cap_entries=int(counts["cap"]),
            is_clifford_only=bool(is_clifford_only),
            is_clifford_t_like=bool(is_clifford_t_like),
            warnings=tuple(_unique_ordered(warnings)),
        )

    @classmethod
    def _magic_strategy_entry_kind(cls, entry) -> str:
        """Classify one stream entry for :meth:`recommend_magic_strategy`."""
        if not (isinstance(entry, (list, tuple)) and entry):
            return "opaque"
        if not isinstance(entry[0], str):
            # Stim's compiler emits its ideal Clifford operations as float32
            # matrices. Recognize small unitary matrices without examining the
            # large/opaque operator forms that this advisory API cannot price.
            if len(entry) != 2:
                return "opaque"
            try:
                if cls._injectable_rz(entry) is not None:
                    return "injectable"
                where = _normalize_sites(entry[1])
                gate = _as_gate_matrix(entry[0], len(where))
                dim = gate.shape[0]
                nq = int(round(math.log2(dim)))
                if (
                    gate.ndim != 2
                    or gate.shape != (dim, dim)
                    or len(where) != nq
                    or 2 ** nq != dim
                    or nq > 2
                    or not _is_unitary(gate)
                ):
                    return "opaque"
            except (ImportError, IndexError, TypeError, ValueError, RuntimeError):
                return "nonclifford"
            try:
                is_clifford = _tableau_from_exact_unitary(gate) is not None
            except ImportError:
                return "nonclifford"
            return (
                "clifford"
                if is_clifford
                else "nonclifford"
            )

        name = _normalize_event_name(entry[0])
        if name in _CLIFFORD_NAMES:
            return "clifford"
        if name == "disentangle":
            return "control"
        try:
            if cls._injectable_rz(entry) is not None:
                return "injectable"
        except (IndexError, TypeError, ValueError):
            return "opaque"

        if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name == "rot":
            try:
                theta = float(entry[1])
            except (IndexError, TypeError, ValueError):
                return "opaque"
            return "clifford" if cls._is_clifford_angle(theta) else "nonclifford"
        if name in {
            "measure", "reset", "cap", *(_RESET_AXIS_ALIASES),
            *(_MR_ALIASES), *(_MR_AXIS_ALIASES),
        }:
            return "structural"
        return "opaque"

    @classmethod
    def recommend_magic_strategy(
        cls,
        gates,
        *,
        ancilla_budget: Optional[int] = None,
        prioritize_peak_bond: bool = False,
    ) -> dict:
        """Analyze a gate stream and recommend an explicit STN execution mode.

        The report is advisory: it never rewrites or executes ``gates``. It
        recognizes injectable ``T``/``T-dagger``/non-Clifford ``Rz(k*pi/4)``
        entries using the same criterion as :meth:`with_injection`, counts other
        non-Clifford rotations, and returns a plain-English ``"message"`` plus
        machine-readable counts. Dense matrices and coefficient-frame sub-MPOs
        are reported as ``opaque`` because classifying them here could be costly
        or require changing their execution behavior.

        ``ancilla_budget`` is the number of extra clean ancillas available for
        injection. With no stated budget, immediate injection is the conservative
        recommendation for an injectable stream. Set
        ``prioritize_peak_bond=True`` together with a budget at least equal to
        the injectable-gate count to recommend deferred MAST instead.

        For a :meth:`from_stim` simulator, call the instance convenience method
        :meth:`queued_magic_strategy` before :meth:`run`; it analyzes the queued
        sampled/``stream_transform``-produced Pepsy stream.
        """
        if ancilla_budget is not None:
            if isinstance(ancilla_budget, bool) or not isinstance(ancilla_budget, Integral):
                raise TypeError("ancilla_budget must be a nonnegative integer or None.")
            ancilla_budget = int(ancilla_budget)
            if ancilla_budget < 0:
                raise ValueError("ancilla_budget must be nonnegative.")

        counts = {
            "clifford": 0,
            "injectable": 0,
            "nonclifford": 0,
            "structural": 0,
            "control": 0,
            "opaque": 0,
        }
        for entry in cls._as_entries(gates):
            counts[cls._magic_strategy_entry_kind(entry)] += 1

        injections = counts["injectable"]
        deferred_feasible = (
            None if ancilla_budget is None else ancilla_budget >= injections
        )
        complete_clifford_t = (
            injections > 0
            and counts["nonclifford"] == 0
            and counts["opaque"] == 0
        )
        if injections == 0 or ancilla_budget == 0:
            mode = "direct"
        elif (
            prioritize_peak_bond
            and ancilla_budget is not None
            and ancilla_budget >= injections
        ):
            mode = "deferred"
        else:
            mode = "immediate"

        if injections == 0:
            if counts["nonclifford"]:
                message = (
                    f"The stream has {counts['nonclifford']} non-Clifford rotation(s), "
                    "but none are injectable T-family Rz rotations. Use direct STN "
                    "execution with exact_cooling=True; schedule greedy cooling only at "
                    "explicit checkpoints if the coefficient bond grows."
                )
            else:
                message = (
                    "The stream has no injectable T-family rotations. Use direct STN "
                    "execution; magic injection is not applicable."
                )
        elif ancilla_budget == 0:
            message = (
                f"The stream has {injections} injectable T-family rotation(s), but the "
                "ancilla budget is zero. Use direct STN execution with exact_cooling=True."
            )
        elif mode == "deferred":
            message = (
                f"The stream has {injections} injectable T-family rotation(s). With "
                f"{ancilla_budget} available ancilla(s) and peak bond prioritized, use "
                "deferred MAST: with_deferred_injection(..., "
                "projection_order='middle_out'). It reserves one ancilla per injected "
                "gate and moves basis-updating projections to the end."
            )
        elif complete_clifford_t:
            message = (
                f"The stream is Clifford+T-like with {injections} injectable T-family "
                "rotation(s). Use immediate injection as the default: "
                "with_injection(..., n_ancilla=1). It rewrites every eligible rotation, "
                "measures the ancilla immediately, and reuses it. Deferred MAST is an "
                f"alternative when {injections} fresh ancillas and lower replay-phase "
                "peak bond are worth a final projection phase."
            )
        else:
            message = (
                f"The stream has {injections} injectable T-family rotation(s), "
                f"{counts['nonclifford']} other non-Clifford rotation(s), and "
                f"{counts['opaque']} opaque entry/entries. Use immediate injection for "
                "the eligible subset; the remaining non-Clifford work stays on the "
                "direct STN path with exact_cooling=True."
            )

        return {
            "recommended_mode": mode,
            "message": message,
            "total_entries": int(sum(counts.values())),
            "clifford_entries": int(counts["clifford"]),
            "injectable_entries": int(injections),
            "other_nonclifford_entries": int(counts["nonclifford"]),
            "structural_entries": int(counts["structural"]),
            "control_entries": int(counts["control"]),
            "opaque_entries": int(counts["opaque"]),
            "is_clifford_t_like": bool(complete_clifford_t),
            "exact_cooling_recommended": True,
            "immediate_ancillas_required": 1 if injections else 0,
            "deferred_ancillas_required": int(injections),
            "ancilla_budget": ancilla_budget,
            "deferred_feasible": deferred_feasible,
            "prioritize_peak_bond": bool(prioritize_peak_bond),
        }

    def queued_magic_strategy(self, **kwargs) -> dict:
        """Recommend a mode for the currently queued Pepsy gate stream.

        This is particularly useful immediately after :meth:`from_stim` and an
        optional ``stream_transform``. Call it before :meth:`run`, because that
        method consumes successfully executed queue entries.
        """
        return type(self).recommend_magic_strategy(self._queue, **kwargs)

    @classmethod
    def recommend_settings(
        cls,
        gates,
        *,
        n_qubits: Optional[int] = None,
        ancilla_budget: Optional[int] = None,
        prioritize_peak_bond: bool = False,
        goal: str = "run",
    ) -> StabilizerMpsSettingsAdvice:
        """Recommend STN settings from a Pepsy stream design.

        The returned advice is intentionally non-executing. It keeps the Pepsy
        stream as the primary interface, uses :meth:`analyze_stream` for facts,
        and calls :meth:`recommend_magic_strategy` for the direct/immediate/
        deferred injection choice.
        """
        normalized_goal = _normalize_event_name(goal)
        if normalized_goal not in {"validate", "run", "benchmark"}:
            raise ValueError(
                "goal must be one of 'validate', 'run', or 'benchmark', "
                f"got {goal!r}."
            )

        analysis = cls.analyze_stream(gates, n_qubits=n_qubits)
        magic = cls.recommend_magic_strategy(
            gates,
            ancilla_budget=ancilla_budget,
            prioritize_peak_bond=prioritize_peak_bond,
        )
        mode = magic["recommended_mode"]
        execution_method = {
            "direct": "apply",
            "immediate": "with_injection",
            "deferred": "with_deferred_injection",
        }[mode]

        nonclifford_pressure = (
            analysis.injectable_entries
            + analysis.other_nonclifford_entries
            + analysis.opaque_entries
        )
        settings = {
            "chi": None,
            "cutoff": 1e-12,
            "exact_cooling": True,
            "stabilize_unitary": False,
        }
        if normalized_goal != "validate" and nonclifford_pressure:
            settings["chi"] = 64
        if (
            mode in {"direct", "immediate", "deferred"}
            and normalized_goal != "validate"
            and nonclifford_pressure
            and not analysis.cap_entries
        ):
            settings["layout"] = "auto"
            settings["layout_report"] = False

        warnings = list(analysis.warnings)
        if settings["chi"] is not None:
            warnings.append(
                "chi=64 is a starting cap, not a convergence claim; sweep chi "
                "for production accuracy."
            )
        elif (
            normalized_goal != "validate"
            and nonclifford_pressure
            and analysis.estimated_qubits is not None
            and analysis.estimated_qubits > 16
        ):
            warnings.append(
                "Exact chi=None can become expensive for larger non-Clifford "
                "streams; use it first as a correctness reference."
            )
        if mode == "immediate" and prioritize_peak_bond and not magic["deferred_feasible"]:
            warnings.append(
                "Deferred MAST was requested by priority, but it needs one fresh "
                "ancilla per injectable gate."
            )
        if normalized_goal == "benchmark":
            warnings.append(
                "Benchmark direct, immediate, and deferred modes separately before "
                "drawing performance conclusions."
            )

        disentangle_recommended = (
            normalized_goal != "validate"
            and settings["chi"] is not None
            and (
                analysis.other_nonclifford_entries
                + analysis.opaque_entries
                + analysis.submpo_entries
            )
            >= 4
        )
        if disentangle_recommended:
            warnings.append(
                "Consider explicit disentangle checkpoints after sizeable "
                "non-Clifford blocks, not after every gate."
            )

        message_parts = [
            f"Use {execution_method} for {normalized_goal} mode "
            f"({mode} execution)."
        ]
        if mode == "immediate":
            message_parts.append(
                "Immediate injection uses one reusable clean magic ancilla by default."
            )
        elif mode == "deferred":
            message_parts.append(
                "Deferred MAST reserves one clean ancilla per injectable gate and "
                "moves projections to the end."
            )
        else:
            message_parts.append(
                "Direct execution keeps all non-Clifford work on the coefficient "
                "MPS path."
            )
        message_parts.append(
            "Constructor settings: "
            + ", ".join(f"{key}={value!r}" for key, value in settings.items())
            + "."
        )
        if warnings:
            message_parts.append("Warnings: " + " ".join(warnings))

        return StabilizerMpsSettingsAdvice(
            goal=normalized_goal,
            recommended_mode=mode,
            execution_method=execution_method,
            settings=settings,
            analysis=analysis,
            magic_strategy=magic,
            immediate_ancillas_required=int(magic["immediate_ancillas_required"]),
            deferred_ancillas_required=int(magic["deferred_ancillas_required"]),
            ancilla_budget=magic["ancilla_budget"],
            deferred_feasible=magic["deferred_feasible"],
            disentangle_checkpoints_recommended=bool(disentangle_recommended),
            warnings=tuple(_unique_ordered(warnings)),
            message=" ".join(message_parts),
        )

    def queued_stream_analysis(self, **kwargs) -> StreamAnalysisRecord:
        """Analyze the currently queued Pepsy stream without consuming it."""
        kwargs.setdefault("n_qubits", self.n)
        return type(self).analyze_stream(self._queue, **kwargs)

    def queued_recommend_settings(self, **kwargs) -> StabilizerMpsSettingsAdvice:
        """Recommend settings for the currently queued Pepsy stream."""
        kwargs.setdefault("n_qubits", self.n)
        return type(self).recommend_settings(self._queue, **kwargs)

    def run_queued_stream(self, **kwargs) -> StabilizerMpsRunResult:
        """Replay the currently queued Pepsy stream through the public runner.

        The current simulator is not mutated. This is useful after
        :meth:`from_stim`, where the Stim circuit has already been sampled and
        converted into a Pepsy stream.
        """
        kwargs.setdefault("n_qubits", self.n)
        return run_stabilizer_mps_stream(self._queue, **kwargs)

    @classmethod
    def run_stream(cls, gates, **kwargs) -> StabilizerMpsRunResult:
        """Run one Pepsy STN gate stream and return a typed result record.

        This is the class-level spelling of :func:`run_stabilizer_mps_stream`
        for new code that already works through :class:`MpsStabOptimizer` /
        :class:`StabilizerMpsSimulator`.
        """
        return run_stabilizer_mps_stream(gates, **kwargs)

    @classmethod
    def simulate(cls, gates, **kwargs) -> StabilizerMpsRunResult:
        """Alias for :meth:`run_stream`."""
        return cls.run_stream(gates, **kwargs)

    # ------------------------------------------------------------------ #
    # Static STN frame auto-layout
    # ------------------------------------------------------------------ #
    def _refresh_layout_map(self) -> None:
        """Refresh the logical-coefficient-site -> MPS-position map."""
        self._logical_to_mps = {
            int(logical): int(pos)
            for pos, logical in enumerate(self.logical_order)
        }

    def _layout_is_identity(self) -> bool:
        """Return whether the coefficient MPS is in logical site order."""
        return tuple(self.logical_order) == tuple(range(self.n))

    def _mps_site(self, logical_site: int) -> int:
        """Map a logical coefficient qubit to its current MPS site position."""
        try:
            return self._logical_to_mps[int(logical_site)]
        except KeyError as exc:
            raise ValueError(
                f"coefficient site {logical_site!r} is not present in the "
                f"current STN layout {self.logical_order!r}."
            ) from exc

    def _mps_sites(self, logical_sites) -> tuple[int, ...]:
        """Map logical coefficient support sites to MPS positions."""
        return tuple(self._mps_site(site) for site in logical_sites)

    def _mps_terms(self, logical_terms) -> dict[int, str]:
        """Map a logical coefficient-frame Pauli support to MPS positions."""
        return {
            self._mps_site(site): axis
            for site, axis in logical_terms.items()
        }

    def current_frame_layout(
        self,
        *,
        order="auto",
        refine_passes=8,
        refine_numba=True,
        spectral_dense_max=512,
        recursive_dense_max=1024,
        nevergrad_budget=64,
        nevergrad_seed=0,
        nevergrad_optimizer="OnePlusOne",
        kahypar_config_path=None,
        kahypar_seed=0,
        weight_mode="operator_schmidt",
    ):
        """Find a static MPS order from the queued STN frame supports.

        The pre-pass replays only tableau-changing events on a temporary copy.
        Each expensive coefficient-frame event contributes the support of its
        current ``C^dagger O C`` image.  By default, multi-site events are
        weighted by a baseline locality cost plus an operator-Schmidt
        entanglement proxy, so stronger two-branch rotations and wider
        coefficient-frame operators receive more layout priority.  The
        returned plan is a Pepsy-style layout plan whose ``site_order`` maps
        MPS positions to logical coefficient qubits.  It does not mutate the
        simulator.

        ``weight_mode`` accepts ``"operator_schmidt"`` (the default),
        ``"count"`` for the historical uniform weighting, and ``"angle"`` /
        ``"auto"`` for angle-based weighting.
        """
        records = self._frame_layout_records(
            self._queue,
            weight_mode=weight_mode,
        )
        stream = [
            ("submpo", {"weight": record["weight"]}, record["support"])
            for record in records
        ]
        finder = MpsGateStreamLayoutFinder(stream, L=self.n)

        def weight_fn(payload, _support, _event_type):
            if isinstance(payload, Mapping):
                return float(payload.get("weight", 1.0))
            return 1.0

        plan = finder.run(
            order=order,
            refine_passes=refine_passes,
            refine_numba=refine_numba,
            spectral_dense_max=spectral_dense_max,
            recursive_dense_max=recursive_dense_max,
            nevergrad_budget=nevergrad_budget,
            nevergrad_seed=nevergrad_seed,
            nevergrad_optimizer=nevergrad_optimizer,
            kahypar_config_path=kahypar_config_path,
            kahypar_seed=kahypar_seed,
            weight_fn=weight_fn,
            weight_mode="count",
        )
        plan = dict(plan)
        plan["kind"] = "stn_frame_layout"
        plan["source"] = "queued_frame_supports"
        plan["frame_events"] = tuple(records)
        plan["frame_weight_mode"] = weight_mode
        return plan

    find_frame_layout = current_frame_layout

    def _frame_layout_records(self, entries, *, weight_mode="operator_schmidt"):
        """Return weighted logical frame-support records for a stream."""
        mode = str(weight_mode).replace("-", "_").strip().lower()
        if mode in ("unit", "uniform", "none"):
            mode = "count"
        if mode not in ("count", "angle", "auto", "operator_schmidt"):
            raise ValueError(
                "STN frame layout weight_mode must be 'operator_schmidt', "
                "'count', 'angle', or 'auto'."
            )
        dry = self.copy()
        dry._queue = []
        records = []
        for entry in self._as_entries(entries):
            dry._frame_layout_trace_entry(entry, records, weight_mode=mode)
        return tuple(records)

    def _frame_layout_weight(
        self,
        *,
        weight_mode,
        theta=None,
        coeff=None,
        support_size=None,
        operator_weight=None,
    ):
        """Return the scalar weight used for one frame-layout record."""
        if weight_mode == "operator_schmidt":
            if operator_weight is not None:
                amplitude = 1.0
                if coeff is not None:
                    try:
                        amplitude = max(abs(complex(coeff)), 1.0e-12)
                    except (TypeError, ValueError):
                        amplitude = 1.0
                return max(1.0e-12, float(operator_weight) * amplitude)
            if theta is not None:
                if support_size is not None and int(support_size) < 2:
                    return 1.0
                # Keep a unit locality baseline and add the non-leading
                # operator-Schmidt weight of the I/P rotation branches.
                return 1.0 + _operator_schmidt_tail_weight(theta)
            if coeff is not None:
                try:
                    return max(abs(complex(coeff)), 1.0e-12)
                except (TypeError, ValueError):
                    return 1.0
            return 1.0
        if coeff is not None:
            try:
                return float(abs(complex(coeff)))
            except (TypeError, ValueError):
                return 1.0
        if weight_mode in ("angle", "auto") and theta is not None:
            return _layout_angle_weight(theta)
        return 1.0

    def _frame_layout_add_pauli(
        self,
        pauli,
        where,
        records,
        *,
        kind,
        entry,
        weight_mode,
        theta=None,
        weight=None,
        absorb_basis=False,
    ):
        """Record one current frame image and optionally dry-run its basis update."""
        m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
        terms, _sign = hermitian_pauli_terms(m_pauli)
        support = tuple(sorted(terms))
        if support:
            if weight is None:
                weight = self._frame_layout_weight(
                    weight_mode=weight_mode,
                    theta=theta,
                    support_size=len(support),
                )
            records.append({
                "kind": kind,
                "entry": entry,
                "support": support,
                "weight": float(weight),
                "operator_weight": float(weight),
                "absorbs_basis": bool(absorb_basis),
            })
        if absorb_basis and support:
            _ops, v_tableau, _k = _localizing_clifford(
                terms,
                self.n,
                site_position=self._mps_site,
            )
            self.state.absorb_basis_clifford(v_tableau)

    def _frame_layout_trace_rotation(self, name, params, records, *, entry, weight_mode):
        """Trace a rotation entry for layout without changing ``|p>``."""
        theta, where, axes = self._rotation_spec(name, params)
        phys = pauli_string(axes, where, self.n)
        if self._is_clifford_angle(theta):
            self._apply_clifford_rotation(theta, where, axes)
            return
        m_pauli = self.state.frame_pauli(phys)
        terms, _sign = hermitian_pauli_terms(m_pauli)
        support = tuple(sorted(terms))
        if support:
            weight = self._frame_layout_weight(
                weight_mode=weight_mode,
                theta=theta,
                support_size=len(support),
            )
            records.append({
                "kind": "rotation",
                "entry": entry,
                "support": support,
                "weight": float(weight),
                "operator_weight": float(weight),
                "absorbs_basis": False,
            })

    def _frame_layout_trace_matrix(self, gate, where, records, *, entry, weight_mode):
        """Trace an explicit physical matrix entry for layout."""
        where = _normalize_sites(where)
        gate = _as_gate_matrix(gate, len(where))
        dim = gate.shape[0]
        nq = int(round(math.log2(dim)))
        if 2 ** nq != dim or gate.shape != (dim, dim):
            raise ValueError(f"Gate matrix must be square 2^k x 2^k, got {gate.shape}.")
        if len(where) != nq:
            raise ValueError(f"Gate on {nq} qubit(s) but where={where!r}.")

        dense_operator_weight = (
            _dense_operator_schmidt_layout_weight(gate, nq)
            if weight_mode == "operator_schmidt"
            else None
        )

        tableau = _tableau_from_exact_unitary(gate)
        gate_is_unitary = _is_unitary(gate)
        if tableau is not None:
            self.state.do_tableau(tableau, where)
            return

        if nq == 1 and gate_is_unitary:
            alpha, theta, beta = _zyz_angles(gate)
            q = where[0]
            self._frame_layout_trace_rotation(
                "rz", (beta, q), records, entry=entry, weight_mode=weight_mode
            )
            self._frame_layout_trace_rotation(
                "ry", (theta, q), records, entry=entry, weight_mode=weight_mode
            )
            self._frame_layout_trace_rotation(
                "rz", (alpha, q), records, entry=entry, weight_mode=weight_mode
            )
            return

        limit = self.max_pauli_decomposition_qubits
        if limit is not None and nq > limit:
            raise ValueError(
                f"Pauli decomposition of a {nq}-qubit dense gate would enumerate "
                f"{4**nq} candidate terms, exceeding "
                f"max_pauli_decomposition_qubits={limit}."
            )
        for term_index, (labels, coeff) in enumerate(
            pauli_decomposition(gate, nq, tol=self.operator_tol), start=1
        ):
            if (
                self.max_pauli_terms is not None
                and term_index > self.max_pauli_terms
            ):
                raise ValueError(
                    f"dense gate retained more than max_pauli_terms="
                    f"{self.max_pauli_terms} during layout analysis."
                )
            phys = pauli_string(labels, where, self.n)
            frame_terms, _sign = hermitian_pauli_terms(self.state.frame_pauli(phys))
            support = tuple(sorted(frame_terms))
            if support:
                weight = self._frame_layout_weight(
                    weight_mode=weight_mode,
                    coeff=coeff,
                    operator_weight=dense_operator_weight,
                )
                records.append({
                    "kind": "matrix_branch",
                    "entry": entry,
                    "support": support,
                    "weight": float(weight),
                    "operator_weight": (
                        float(dense_operator_weight)
                        if dense_operator_weight is not None
                        else float(weight)
                    ),
                    "absorbs_basis": False,
                })

    def _frame_layout_trace_entry(self, entry, records, *, weight_mode):
        """Trace one queued entry into weighted frame-support records."""
        conditional = conditional_event_parts(entry)
        if conditional is not None:
            raise ValueError(
                "static STN frame_layout='auto' cannot safely prepass a "
                "branch-dependent feed-forward action; provide an explicit "
                "layout or use the ordinary interaction layout."
            )
        parts = submpo_event_parts(entry, normalize_where=True)
        if parts is not None:
            mpo, where = parts
            support = tuple(sorted(_unique_ordered(where)))
            if support:
                weight = (
                    _submpo_operator_layout_weight(mpo)
                    if weight_mode == "operator_schmidt"
                    else 1.0
                )
                records.append({
                    "kind": "submpo",
                    "entry": entry,
                    "support": support,
                    "weight": float(weight),
                    "operator_weight": float(weight),
                    "absorbs_basis": False,
                })
            return

        if not (isinstance(entry, (list, tuple)) and len(entry) >= 1):
            raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

        head = entry[0]
        if not isinstance(head, str):
            if len(entry) != 2:
                raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
            gate, where = entry
            self._frame_layout_trace_matrix(
                self._gate_to_numpy(gate),
                where,
                records,
                entry=entry,
                weight_mode=weight_mode,
            )
            return

        name = _normalize_event_name(head)
        if name == "disentangle":
            return
        if name in _CLIFFORD_NAMES:
            self.state.apply_clifford(name, *entry[1:])
            return
        if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name in (
            "rot", "t", "tdg",
        ):
            self._frame_layout_trace_rotation(
                name,
                entry[1:],
                records,
                entry=entry,
                weight_mode=weight_mode,
            )
            return
        if name == "measure":
            pauli, where = entry[1], entry[2]
            absorb = bool(entry[4]) if len(entry) > 4 else False
            self._frame_layout_add_pauli(
                pauli,
                where,
                records,
                kind="measure",
                entry=entry,
                weight_mode=weight_mode,
                absorb_basis=absorb,
            )
            return
        if name == "reset" or name in _RESET_AXIS_ALIASES:
            axes, where = _parse_reset_args(
                entry[1:],
                default_axis=_RESET_AXIS_ALIASES.get(name),
            )
            for axis, q in zip(axes, where):
                self._frame_layout_add_pauli(
                    axis,
                    q,
                    records,
                    kind="reset",
                    entry=entry,
                    weight_mode=weight_mode,
                    absorb_basis=True,
                )
            return
        if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
            axes, where, _outcomes, absorb = _parse_measure_reset_args(
                entry[1:],
                default_axis=_MR_AXIS_ALIASES.get(name),
            )
            for axis, q in zip(axes, where):
                self._frame_layout_add_pauli(
                    axis,
                    q,
                    records,
                    kind="measure_reset",
                    entry=entry,
                    weight_mode=weight_mode,
                    absorb_basis=absorb,
                )
            return
        if name == "cap":
            raise ValueError(
                "static STN auto-layout is not supported with cap events, "
                "because cap changes the qubit/MPS length."
            )
        raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")

    def _validate_layout_plan_for_stn(self, plan) -> None:
        """Validate that a layout plan is a full permutation of STN qubits."""
        site_order = tuple(int(site) for site in plan.get("site_order", plan.get("order", ())))
        if len(site_order) != self.n:
            raise ValueError(
                f"layout site_order length must match n={self.n}, got {len(site_order)}."
            )
        if sorted(site_order) != list(range(self.n)):
            raise ValueError(
                f"layout site_order must be a permutation of range({self.n})."
            )
        site_map = plan.get("site_map", plan.get("layout"))
        if site_map is None:
            raise ValueError("layout plan must contain a site_map/layout mapping.")
        expected = {site: pos for pos, site in enumerate(site_order)}
        if {int(k): int(v) for k, v in dict(site_map).items()} != expected:
            raise ValueError(
                "layout site_map must map each logical coefficient site to its "
                "position in site_order."
            )

    def _explicit_layout_plan(self, site_order):
        """Build a minimal STN frame-layout plan from an explicit site order."""
        site_order = tuple(int(site) for site in site_order)
        site_map = {site: position for position, site in enumerate(site_order)}
        return {
            "kind": "stn_frame_layout",
            "selected_order": "explicit",
            "qubit_inds": site_order,
            "site_order": site_order,
            "order": site_order,
            "layout": site_map,
            "site_map": site_map,
            "inverse_site_map": {
                position: site for site, position in site_map.items()
            },
            "stats": {},
            "input_stats": {},
        }

    def _resolve_layout_plan_argument(self, plan_or_order, layout_kwargs=None):
        """Resolve a static STN layout request without mutating the simulator."""
        if isinstance(plan_or_order, Mapping):
            plan = dict(plan_or_order)
        elif isinstance(plan_or_order, str):
            kwargs = {} if layout_kwargs is None else dict(layout_kwargs)
            plan = self.current_frame_layout(order=plan_or_order, **kwargs)
        else:
            try:
                plan = self._explicit_layout_plan(plan_or_order)
            except TypeError as exc:
                raise TypeError(
                    "plan_or_order must be a layout mapping, an order name, "
                    "or a permutation of logical coefficient sites."
                ) from exc
        self._validate_layout_plan_for_stn(plan)
        return plan

    @staticmethod
    def _product_site_vector(p, physical_site):
        """Extract a local vector from an isolated coefficient-MPS site."""
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
                "extracting a local product vector requires every virtual dimension "
                "to be one."
            )
        axes = [axis for axis in range(tensor.ndim) if axis != physical_axis]
        axes.append(physical_axis)
        data = ar.do("transpose", tensor.data, tuple(axes))
        return data.reshape(-1)

    def _relabel_product_mps(self, target_order, *, current_order):
        """Rebuild a bond-one coefficient MPS in a new logical site order."""
        p = self.state.p
        if getattr(p, "cyclic", False):
            raise ValueError(
                "STN static layout relabeling currently requires an open-boundary MPS."
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
        self.state.p = new_p
        self.state.info = {"cur_orthog": None}

    @staticmethod
    def _format_layout_value(value):
        """Format one layout diagnostic value compactly."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3g}"

    @classmethod
    def _format_layout_reduction(cls, before, after):
        """Format a before/after layout diagnostic compactly."""
        text = f"{cls._format_layout_value(before)} -> {cls._format_layout_value(after)}"
        try:
            before = float(before)
            after = float(after)
        except (TypeError, ValueError):
            return text
        if before > 0.0:
            text += f" ({100.0 * (before - after) / before:.1f}% lower)"
        return text

    @classmethod
    def _layout_report_text(cls, plan):
        """Return a concise human-readable STN layout report."""
        stats = plan.get("stats", {})
        input_stats = plan.get("input_stats", {})
        if not input_stats:
            return None
        selected = plan.get("selected_order", "<unknown>")
        site_order = plan.get("site_order", ())
        lines = [
            (
                "MpsStabOptimizer frame layout: "
                f"order={selected}, sites={len(site_order)}, "
                f"events={stats.get('num_events', input_stats.get('num_events', 0))}"
            ),
            (
                "  frame event span max/mean: "
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
                + " | cut L2: "
                + cls._format_layout_reduction(
                    input_stats.get("weighted_cut_congestion_l2", 0.0),
                    stats.get("weighted_cut_congestion_l2", 0.0),
                )
            ),
        ]
        return "\n".join(lines)

    def apply_layout(
        self,
        plan_or_order="auto",
        *,
        layout_kwargs=None,
        layout_report: bool = True,
    ) -> "MpsStabOptimizer":
        """Install a static STN frame layout while ``|p>`` is still product.

        The tableau/physical qubit labels stay unchanged.  Only the coefficient
        MPS tensor order changes, and every future coefficient-frame support is
        mapped through the installed logical-order map.  This keeps the operation
        safe and exact for any state whose coefficient MPS has ``max_bond()==1``
        (including Clifford-entangled stabilizer states), and rejects entangled
        coefficient states before mutation.
        """
        for entry in self._queue:
            if isinstance(entry, (list, tuple)) and entry:
                head = entry[0]
                if isinstance(head, str) and _normalize_event_name(head) == "cap":
                    raise ValueError(
                        "static STN layout cannot be installed for streams with "
                        "cap events, because cap changes the qubit/MPS length."
                    )
        plan = self._resolve_layout_plan_argument(plan_or_order, layout_kwargs)
        target_order = tuple(int(site) for site in plan["site_order"])
        current_order = tuple(self.logical_order)
        if target_order != current_order:
            if int(self.state.max_bond()) != 1:
                raise ValueError(
                    "static STN layout requires a product coefficient MPS "
                    "(state.max_bond() == 1); got max_bond={} . Apply the "
                "layout before non-Clifford evolution entangles |p>.".format(
                        self.state.max_bond()
                    )
                )
            self._relabel_product_mps(target_order, current_order=current_order)
        self.logical_order = list(target_order)
        self._refresh_layout_map()
        self._localizer_cache.clear()
        self.layout_plan = plan
        self.last_layout_plan = plan
        if layout_report:
            report = self._layout_report_text(plan)
            if report:
                print(report)
        return self

    def _apply_layout_from_entries(
        self,
        entries,
        layout,
        *,
        layout_kwargs=None,
        layout_report: bool = True,
    ) -> None:
        """Install a static layout found from ``entries`` without queuing them."""
        if layout is None or layout is False:
            return
        old_queue = self._queue
        self._queue = list(entries)
        try:
            self.apply_layout(
                layout,
                layout_kwargs=layout_kwargs,
                layout_report=layout_report,
            )
        finally:
            self._queue = old_queue

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _shot_factory(self):
        """Build fresh STN optimizers for the shared trajectory runners."""
        template = getattr(self, "_initial_state", self.state)
        logical_order = tuple(self.logical_order)
        layout_plan = deepcopy(self.layout_plan)

        def make_optimizer():
            optimizer = type(self)(
                template.copy(),
                chi=self.chi,
                mode=self.mode,
                cutoff=self.cutoff,
                operator_tol=self.operator_tol,
                max_pauli_decomposition_qubits=self.max_pauli_decomposition_qubits,
                max_pauli_terms=self.max_pauli_terms,
                max_dense_cap_qubits=self.max_dense_cap_qubits,
                exact_cooling=self.exact_cooling,
                stabilize_unitary=self.stabilize_unitary,
                fit_init_strategy=self.fit_init_strategy,
                fit_init_rand_strength=self.fit_init_rand_strength,
                fit_init_seed=self.fit_init_seed,
                compression_seed=self.compression_seed,
                dtype=self.dtype,
                to_backend=self.to_backend,
                inplace=True,
            )
            if logical_order != tuple(range(optimizer.n)):
                optimizer.logical_order = list(logical_order)
                optimizer._refresh_layout_map()
                optimizer.layout_plan = deepcopy(layout_plan)
                optimizer.last_layout_plan = deepcopy(layout_plan)
            return optimizer

        return make_optimizer

    def _run_shots(
        self,
        shots,
        *,
        error_model=None,
        seed=None,
        run_kwargs=None,
        strategy="auto",
        max_branches=128,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        retain="all",
        auto_max_expected_faults=0.1,
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
        """Dispatch noisy replay through the shared MPS/STN runner."""
        from ..noise import (  # pylint: disable=import-outside-toplevel
            NoisyResult,
            run_noisy_shots,
            run_trajectory_shots,
        )

        mpi_enabled = mpi is not None and mpi is not False
        if not mpi_enabled and any(
            value is not None for value in (observable, checkpoint_path)
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
            plan = self._trajectory_plan
            if plan is None and error_model is None:
                from ..noise import compile_trajectory_stream

                plan = compile_trajectory_stream(self._gate_stream)
            runner = MPIShotRunner(
                self._shot_factory(),
                plan if error_model is None else self._gate_stream,
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
                self._gate_stream,
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
                plan = self._trajectory_plan
                if plan is None:
                    from ..noise import compile_trajectory_stream

                    plan = compile_trajectory_stream(self._gate_stream)
                shot_gates = plan.entries if workers > 1 else plan
                raw = run_trajectory_shots(
                    self._shot_factory(),
                    shot_gates,
                    shots,
                    _progress=(
                        update_progress if progress_bar is not None else None
                    ),
                    **common,
                )
            else:
                if self._has_trajectory_events:
                    raise ValueError(
                        "do not combine stream-local trajectory events with "
                        "error_model; use one noise representation per stream."
                    )
                raw = run_noisy_shots(
                    self._shot_factory(),
                    self._gate_stream,
                    error_model,
                    shots,
                    auto_max_expected_faults=auto_max_expected_faults,
                    _progress=(
                        update_progress if progress_bar is not None else None
                    ),
                    **common,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return NoisyResult(raw)

    def _execution_snapshot(self):
        """Capture the mutable replay state for optional atomic recovery."""
        rng_state = deepcopy(self._rng.bit_generator.state)
        return {
            "state": self.state.copy(),
            "infidelities": list(self.infidelities),
            "nonunitary_infidelities": list(self._nonunitary_infidelities),
            "compression_norm_events": deepcopy(self._compression_norm_events),
            "norm_events": deepcopy(self.norm_events),
            "norm_log_survival": self._norm_log_survival,
            "compression_segment_log_survival": (
                self._compression_segment_log_survival
            ),
            "infidelity_valid": self._infidelity_valid,
            "current_infidelity": self._current_infidelity,
            "norm_segment_open": self._norm_segment_open,
            "bond_history": list(self.bond_history),
            "exact_cooling_events": deepcopy(self.exact_cooling_events),
            "measurements": list(self.measurements),
            "immediate_projection_events": deepcopy(self.immediate_projection_events),
            "last_immediate_injection_report": deepcopy(
                self.last_immediate_injection_report
            ),
            "deferred_projection_events": deepcopy(self.deferred_projection_events),
            "last_deferred_injection_report": deepcopy(
                self.last_deferred_injection_report
            ),
            "logical_order": list(self.logical_order),
            "layout_plan": deepcopy(self.layout_plan),
            "last_layout_plan": deepcopy(self.last_layout_plan),
            "rng_state": rng_state,
            "last_fit_diagnostics": deepcopy(self._last_fit_diagnostics),
            "dmrg1_one_site_locked": bool(self._dmrg1_one_site_locked),
        }

    def _restore_execution_snapshot(self, snapshot) -> None:
        """Restore a snapshot made by :meth:`_execution_snapshot`."""
        self.state = snapshot["state"]
        self.infidelities = list(snapshot["infidelities"])
        self._nonunitary_infidelities = list(snapshot["nonunitary_infidelities"])
        self._compression_norm_events = deepcopy(
            snapshot["compression_norm_events"]
        )
        self.norm_events = deepcopy(snapshot["norm_events"])
        self._norm_log_survival = snapshot["norm_log_survival"]
        self._compression_segment_log_survival = snapshot[
            "compression_segment_log_survival"
        ]
        self._infidelity_valid = snapshot["infidelity_valid"]
        self._current_infidelity = snapshot["current_infidelity"]
        self._norm_segment_open = snapshot["norm_segment_open"]
        self.bond_history = list(snapshot["bond_history"])
        self.exact_cooling_events = deepcopy(snapshot["exact_cooling_events"])
        self.measurements = list(snapshot["measurements"])
        self.immediate_projection_events = deepcopy(
            snapshot["immediate_projection_events"]
        )
        self.last_immediate_injection_report = deepcopy(
            snapshot["last_immediate_injection_report"]
        )
        self.deferred_projection_events = deepcopy(
            snapshot["deferred_projection_events"]
        )
        self.last_deferred_injection_report = deepcopy(
            snapshot["last_deferred_injection_report"]
        )
        self.logical_order = list(snapshot["logical_order"])
        self._refresh_layout_map()
        self.layout_plan = deepcopy(snapshot["layout_plan"])
        self.last_layout_plan = deepcopy(snapshot["last_layout_plan"])
        self._last_fit_diagnostics = deepcopy(snapshot["last_fit_diagnostics"])
        self._dmrg1_one_site_locked = bool(snapshot["dmrg1_one_site_locked"])
        self._rng.bit_generator.state = deepcopy(snapshot["rng_state"])
        self.backend_info()

    def run(
        self,
        *,
        progbar: bool = False,
        shots: int = 1,
        error_model=None,
        seed=None,
        run_kwargs=None,
        strategy="auto",
        max_branches=128,
        importance_sampling=None,
        max_branch_factor=None,
        parallel_workers=1,
        parallel_backend="thread",
        retain="all",
        auto_max_expected_faults=0.1,
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
        timing: bool = False,
        transactional: bool = False,
        atomic=None,
    ):
        """Apply all queued gates in order, consuming successful entries.

        If an entry raises, successfully applied entries are removed while the
        failed entry and its remaining suffix stay queued. Retrying therefore
        never replays an already-applied prefix.

        Parameters
        ----------
        progbar : bool
            Show a ``tqdm`` progress bar reporting the current stream part and
            the MPS-compatible ``infidelity`` diagnostic.
        shots, error_model, strategy, ...
            Shot/noise options matching :meth:`MpsOptimizer.run`. They use the
            shared trajectory runner and return :class:`pepsy.NoisyResult`.
        mpi : bool | MPI communicator | None
            Run the shot ensemble collectively over MPI. ``True`` uses
            ``MPI.COMM_WORLD``.
        workers : int | "auto" | None
            Local shot workers. ``"auto"`` divides the process CPU allowance
            across MPI ranks sharing a host.
        progress : {"auto", True, False}
            Show one aggregate rank-zero progress bar for MPI runs.
        timing : bool
            Record a lightweight wall-clock replay record available through
            :meth:`get_run_timing`.
        transactional : bool
            If true, restore the STN state and diagnostics when an entry fails;
            the failed entry and suffix remain queued for retry. This is opt-in
            because a full STN snapshot is intentionally expensive.
        atomic : bool | None
            Alias for ``transactional``.
        """
        if atomic is not None:
            transactional = bool(atomic)
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
            raise ValueError("shots must be a nonnegative integer.")
        if run_kwargs is not None and not isinstance(run_kwargs, Mapping):
            raise TypeError("run_kwargs must be a mapping or None.")
        shot_requested = bool(
            self._has_trajectory_events
            or error_model is not None
            or int(shots) != 1
            or run_kwargs is not None
            or strategy != "auto"
            or max_branches != 128
            or importance_sampling is not None
            or max_branch_factor is not None
            or int(parallel_workers) != 1
            or parallel_backend != "thread"
            or retain != "all"
            or (mpi is not None and mpi is not False)
            or workers not in {None, "auto"}
            or observable is not None
            or checkpoint_path is not None
        )
        if shot_requested:
            started = time.perf_counter()
            result = self._run_shots(
                shots,
                error_model=error_model,
                seed=seed,
                run_kwargs=run_kwargs,
                strategy=strategy,
                max_branches=max_branches,
                importance_sampling=importance_sampling,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
                retain=retain,
                auto_max_expected_faults=auto_max_expected_faults,
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
            self._last_run_timing = {
                "enabled": bool(timing),
                "mode": "trajectory" if error_model is None else "noisy",
                "entries": len(self._gate_stream),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            return result

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        queue = tuple(self._queue)
        completed = 0
        pbar = None
        snapshot = self._execution_snapshot() if transactional else None
        rolled_back = False
        started = time.perf_counter()
        if progbar and queue:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(queue), desc="stab-mps", leave=True, ascii=True)
        try:
            for entry in queue:
                part = self._progress_entry_part(entry)
                self._apply_entry(entry)
                completed += 1
                if pbar is not None:
                    pbar.update(1)
                    diagnostics = self.norm_diagnostics()
                    infidelity = diagnostics["infidelity"]
                    formatted_infidelity = self._format_progress_infidelity(
                        infidelity
                    )
                    pbar.set_postfix(
                        part=part,
                        infidelity=formatted_infidelity,
                    )
        except BaseException:
            if snapshot is not None:
                self._restore_execution_snapshot(snapshot)
                rolled_back = True
            raise
        finally:
            if pbar is not None:
                pbar.close()
            if completed and not rolled_back:
                del self._queue[:completed]
            self._last_run_timing = {
                "enabled": bool(timing),
                "mode": "direct",
                "entries": len(queue),
                "completed": completed,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        return self

    def apply(self, gates, *, progbar: bool = False) -> "MpsStabOptimizer":
        """Convenience: queue ``gates`` and run immediately."""
        return self.set_gates(gates).run(progbar=progbar)

    @staticmethod
    def _format_progress_infidelity(value) -> str:
        """Format the shared MPS/STN progress-bar infidelity field."""
        if value is None:
            return "n/a"
        text = f"{float(value):#.0e}"
        if "e" not in text:
            return text
        mantissa, exponent = text.split("e", 1)
        sign = exponent[0] if exponent[:1] in "+-" else ""
        digits = exponent[1:] if sign else exponent
        digits = digits.lstrip("0") or "0"
        return f"{mantissa}e{sign}{digits}"

    def get_infidelities(self):
        """Return the cumulative ``infidelity`` trace like ``MpsOptimizer``."""
        return self.infidelities

    def get_norm_events(self):
        """Return defensive copies of projective/Kraus norm events."""
        return [
            event.as_dict() if isinstance(event, NormEventRecord) else dict(event)
            for event in self.norm_events
        ]

    def get_compression_norm_events(self):
        """Return per-update retained-norm compression events.

        These are intentionally separate from :meth:`get_norm_events`, whose
        records mark projective/Kraus normalization boundaries. Each event
        here contains both the local ratio for the update and the cumulative
        ratio within the current boundary-aware ledger.
        """
        return deepcopy(self._compression_norm_events)

    def get_normalizations(self):
        """Return explicit normalization records.

        STN keeps projective and Kraus normalization in ``norm_events`` rather
        than exposing MPS tensor-scale records, so this compatibility method is
        intentionally empty.
        """
        return []

    def get_quality_checks(self):
        """Return finite/gauge checks collected by this optimizer."""
        return deepcopy(self._quality_checks)

    def get_fit_diagnostics(self):
        """Return diagnostics for the most recent STN DMRG/FIT update."""
        return deepcopy(self._last_fit_diagnostics)

    def get_run_timing(self):
        """Return the most recent replay timing record."""
        return deepcopy(self._last_run_timing)

    def to_basis_statevector(self, logical_order=True) -> np.ndarray:
        """Return dense ``|nu>`` coefficients, optionally in logical order."""
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        p_dense = np.asarray(to_numpy(self.state.p.to_dense()), dtype=self.dtype)
        if not logical_order or self._layout_is_identity():
            return p_dense.reshape(-1)
        p_dense = p_dense.reshape([2] * self.n)
        axes = [self._mps_site(logical) for logical in range(self.n)]
        return p_dense.transpose(axes).reshape(-1)

    def to_statevector(self, logical_order=True) -> np.ndarray:
        """Return the dense physical statevector ``|psi> = C|nu>``."""
        if not isinstance(logical_order, (bool, np.bool_)):
            raise TypeError("logical_order must be a boolean.")
        basis = self.to_basis_statevector(logical_order=logical_order)
        site_order = None if logical_order else self.logical_order
        return self.state._statevector_from_basis(  # pylint: disable=protected-access
            basis,
            site_order=site_order,
        )

    def to_physical_statevector(self, logical_order=True) -> np.ndarray:
        """Compatibility alias for :meth:`to_statevector`."""
        return self.to_statevector(logical_order=logical_order)

    def to_mps(
        self,
        *,
        mode="exact",
        chi=None,
        cutoff=0.0,
        cutoff_mode="rsum2",
        n_iter=5,
        logical_order=True,
        progbar=False,
        **run_kwargs,
    ):
        """Return an ordinary MPS for the physical state ``|psi> = C|nu>``.

        The tableau is lowered with :meth:`stim.Tableau.to_circuit` and
        replayed as local gates. This keeps the conversion matrix-free: no
        ``2**n`` by ``2**n`` Clifford matrix is formed.

        Parameters
        ----------
        mode : {"exact", "direct", "zipup", "src", "fit-*", "dmrg"}, default="exact"
            ``"exact"`` applies the tableau circuit with unlimited bond and
            zero cutoff. Native MPS compression methods use their bare names,
            matching the ordinary MPS optimizer; ``"dmrg"`` uses its
            variational replay path. Approximate modes require ``chi``.
            Historical ``"quimb-*"`` and ``"mpo-*"`` forms remain accepted as
            deprecated aliases.
        chi : int or None
            Maximum bond dimension for the approximate modes.
        cutoff : float, default=0.0
            Singular-value cutoff for native compression modes and ``"dmrg"``. It is
            intentionally ignored by the lossless ``"exact"`` path.
        cutoff_mode : str, default="rsum2"
            Cutoff convention for the approximate modes.
        n_iter : int, default=5
            DMRG replay sweeps. Ignored by ``"exact"`` and native non-DMRG modes.
        logical_order : bool, default=True
            Return sites in logical qubit order. If false, preserve the
            coefficient MPS's current physical layout and map tableau gates
            into that layout.
        progbar : bool, default=False
            Show the ordinary MPS progress bar for approximate modes.
        **run_kwargs
            Additional keyword arguments forwarded to
            :meth:`pepsy.MpsOptimizer.run` in native and ``"dmrg"`` modes.

        Returns
        -------
        quimb.tensor.MatrixProductState
            A new ordinary MPS. The STN optimizer is never mutated.
        """
        mode = str(mode).strip().lower()
        from ..mps.optimizer import MpsOptimizer  # pylint: disable=import-outside-toplevel

        if mode != "exact" and mode not in MpsOptimizer._ALLOWED_MODES:
            raise ValueError(
                "mode must be 'exact', 'dmrg', or one of the ordinary MPS "
                "compression modes."
            )
        if not isinstance(logical_order, (bool, np.bool_)):
            raise TypeError("logical_order must be a boolean.")
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff must be finite and nonnegative.")
        if mode != "exact":
            if isinstance(chi, (bool, np.bool_)) or not isinstance(chi, Integral):
                raise TypeError("chi must be a positive integer for approximate modes.")
            chi = int(chi)
            if chi < 1:
                raise ValueError("chi must be a positive integer for approximate modes.")

        p = self.state.p.copy()
        current_order = list(self.logical_order)
        if logical_order and current_order != list(range(self.n)):
            if getattr(p, "cyclic", False):
                raise ValueError(
                    "to_mps(logical_order=True) requires an open-boundary MPS."
                )
            for target_pos, logical_site in enumerate(range(self.n)):
                current_pos = current_order.index(logical_site)
                if current_pos == target_pos:
                    continue
                p.swap_site_to_(
                    current_pos,
                    target_pos,
                    method="svd",
                    cutoff=0.0,
                    cutoff_mode="abs",
                )
                moved = current_order.pop(current_pos)
                current_order.insert(target_pos, moved)

        tableau = self.state._sim.current_inverse_tableau().inverse()  # noqa: SLF001
        gate_stream = []
        for gate, where in _tableau_gate_stream(tableau.to_circuit()):
            if not logical_order:
                where = tuple(self._mps_site(site) for site in where)
            gate_stream.append((self._bk(gate), where))

        if mode == "exact":
            for gate, where in gate_stream:
                if len(where) == 1:
                    p.gate_(gate, where[0], contract=True)
                else:
                    p.gate_(
                        gate,
                        where,
                        contract="swap+split",
                        max_bond=None,
                        cutoff=0.0,
                        cutoff_mode="abs",
                    )
            return p

        options = dict(run_kwargs)
        options.setdefault("n_iter", n_iter)
        options.setdefault("progbar", progbar)
        options.setdefault("cutoff", cutoff)
        options.setdefault("cutoff_mode", cutoff_mode)
        options.setdefault("normalize_final", False)
        options.setdefault("stabilize_unitary", False)
        optimizer = MpsOptimizer(
            p,
            gates=gate_stream,
            chi=chi,
            mode=mode,
            inplace=True,
        )
        optimizer.run(**options)
        return optimizer.p

    def to_physical_mps(self, *args, **kwargs):
        """Compatibility alias for :meth:`to_mps`."""
        return self.to_mps(*args, **kwargs)

    def tableau(self):
        """Return a read-only snapshot of the live basis Clifford tableau.

        The returned :class:`stim.Tableau` represents ``C`` in the STN
        factorization ``|psi> = C |nu>``.  Use ``x_output(i)`` and
        ``z_output(i)`` to inspect destabilizer and stabilizer generators.
        """
        return self.state.tableau()

    def ascii_tableau(
        self,
        *,
        compact=True,
        color=False,
        generators=True,
        max_generators=None,
        diagnostics=True,
    ):
        """Return a compact, Pepsy-style text view of the STN tableau.

        Parameters
        ----------
        compact : bool, default=True
            Show Pauli generators by non-identity support, e.g. ``+X@0 Z@3``.
            Set to ``False`` for Stim's full-width strings such as ``+X__Z``.
        color : bool, default=False
            Add ANSI colour styles to the returned text.
        generators : bool, default=True
            Include the destabilizer/stabilizer generator rows.
        max_generators : int or None, default=None
            Limit the number of generator rows shown.  The header and MPS
            summary are still emitted when this is used for large systems.
        diagnostics : bool, default=True
            Include the coefficient-MPS bond, norm, mode, and queue summary.

        Notes
        -----
        This method only reads the Stim tableau and MPS metadata.  It never
        constructs the dense Clifford matrix and does not mutate the state.
        """
        if max_generators is not None:
            if isinstance(max_generators, bool) or not isinstance(max_generators, Integral):
                raise TypeError("max_generators must be a non-negative integer or None.")
            max_generators = int(max_generators)
            if max_generators < 0:
                raise ValueError("max_generators must be non-negative or None.")

        tableau = self.tableau()
        frame = "I" if self.state.is_identity_frame() else "active"
        chi = "inf" if self.chi is None else str(self.chi)
        header = (
            "STN  |psi> = C |nu>"
            f"   n={self.n}   frame={frame}"
            f"   max_bond={self.state.max_bond()}   chi={chi}"
        )
        lines = [_tableau_color(header, "header", color)]

        if diagnostics:
            norm = self.norm()
            lines.append(
                "  "
                f"mode={self.mode}   norm={norm:.6g}"
                f"   queued={len(self._queue)}"
                f"   recorded_steps={max(0, len(self.bond_history) - 1)}"
            )

        if generators:
            lines.append(_tableau_color("tableau generators:", "section", color))
            count = self.n if max_generators is None else min(self.n, max_generators)
            for q in range(count):
                destabilizer = _format_tableau_pauli(
                    tableau.x_output(q), compact=compact
                )
                stabilizer = _format_tableau_pauli(
                    tableau.z_output(q), compact=compact
                )
                d_label = _tableau_color(f"d{q}", "destabilizer", color)
                s_label = _tableau_color(f"s{q}", "stabilizer", color)
                lines.append(
                    f"  {d_label}: {destabilizer}   {s_label}: {stabilizer}"
                )
            if count < self.n:
                omitted = self.n - count
                lines.append(
                    _tableau_color(
                        f"  ... {omitted} generator row(s) omitted",
                        "muted",
                        color,
                    )
                )

        return "\n".join(lines)

    def show(
        self,
        *,
        compact=True,
        generators=True,
        max_generators=None,
        diagnostics=True,
        color=True,
    ):
        """Print the compact STN tableau view.

        This follows Pepsy's ``TreeTensorNetwork.show`` convention: the
        corresponding ``ascii_tableau`` method returns the drawing as text,
        while ``show`` prints it and returns ``None``.
        """
        print(
            self.ascii_tableau(
                compact=compact,
                color=color,
                generators=generators,
                max_generators=max_generators,
                diagnostics=diagnostics,
            )
        )

    def draw(self, *, format="timeline-text"):
        """Return a Stim circuit diagram for the current tableau.

        ``format="timeline-text"`` is the dependency-free text diagram.
        Stim's other diagram formats, including ``"timeline-svg"``, are
        forwarded unchanged.  ``format="circuit"`` returns the underlying
        :class:`stim.Circuit` instead of rendering it.

        The circuit is a decomposition of the Clifford ``C`` only; it does
        not include the coefficient-MPS state ``|nu>`` or non-Clifford events.
        """
        circuit = self.tableau().to_circuit()
        normalized = str(format).strip().lower()
        if normalized in {"circuit", "stim"}:
            return circuit
        diagram = circuit.diagram(normalized)
        # Stim returns a display helper for every diagram format.  Make the
        # dependency-free text form directly useful with string operations;
        # keep richer formats (SVG, matching, ...) as their native helpers.
        return str(diagram) if normalized == "timeline-text" else diagram

    def amplitude(self, bits) -> complex:
        """Amplitude ``<bits|psi>`` for a bitstring (str ``'010'`` or 0/1 seq).

        Qubit 0 is the leftmost bit. Uses the dense reconstruction (small ``n``).
        """
        bits = _validate_bits(bits, expected_length=self.n)
        index = 0
        for bit in bits:
            index = (index << 1) | int(bit)
        return complex(self.to_statevector()[index])

    def probability(self, bits) -> float:
        """Outcome probability ``|<bits|psi>|**2`` (small ``n``)."""
        amp = self.amplitude(bits)
        return float(abs(amp) ** 2)

    def norm(self) -> float:
        """Norm of the coefficient state ``|nu>`` (represented state norm; ~1).

        Computed from :meth:`_norm_squared`, which uses the tracked orthogonality
        centre when available (no full ``<nu|nu>`` contraction) and never mutates
        the state.
        """
        return float(self._norm_squared() ** 0.5)

    def _norm_squared(self) -> float:
        """Return ``<nu|nu>`` (real) without mutating the state.

        When the tracked orthogonality centre is a single site, ``<nu|nu>`` is the
        squared Frobenius norm of that centre tensor; otherwise the full closed
        ``<nu|nu>`` network is contracted.
        """
        cur = self.state.info.get("cur_orthog")
        if isinstance(cur, tuple) and len(cur) == 2 and cur[0] == cur[1]:
            center = self.state.p[self.state.p.site_tag(int(cur[0]))]
            nrm = float(abs(self._to_scalar(center.norm())))
            exponent = float(getattr(self.state.p, "exponent", 0.0))
            return nrm * nrm * (10.0 ** (2.0 * exponent))
        return float(abs(self._to_scalar(self.state.p.H @ self.state.p)))

    def _unitary_infidelity(self) -> Optional[float]:
        """Return cumulative unitary norm loss from the canonical centre."""
        if not self._infidelity_valid:
            return None

        self._canonize_p_single()
        infidelity = min(1.0, max(0.0, 1.0 - self._norm_squared()))
        if not self._norm_segment_open:
            self._current_infidelity = infidelity
        return infidelity

    def _record_compression_norm_event(
        self,
        before_norm_sq: Optional[float],
        after_infidelity: Optional[float],
        *,
        kind: str = "unitary_compression",
    ) -> None:
        """Record one local retained-norm ratio for a compressed update."""
        if (
            before_norm_sq is None
            or after_infidelity is None
            or not np.isfinite(before_norm_sq)
            or before_norm_sq <= 0.0
        ):
            return
        observed_norm_sq = min(1.0, max(0.0, 1.0 - float(after_infidelity)))
        raw = float(np.divide(observed_norm_sq, float(before_norm_sq)))
        local_log_fidelity = log_fidelity_from_norms(
            observed_norm_sq ** 0.5,
            float(before_norm_sq) ** 0.5,
        )
        local_fidelity = min(1.0, max(0.0, fidelity_from_log(local_log_fidelity)))
        local_infidelity = float(1.0 - local_fidelity)
        if local_fidelity == 0.0 or self._compression_segment_log_survival == -math.inf:
            self._compression_segment_log_survival = -math.inf
        else:
            self._compression_segment_log_survival += math.log(local_fidelity)
        segment_log_fidelity = self._compression_segment_log_survival
        cumulative_log_fidelity = (
            -math.inf
            if segment_log_fidelity == -math.inf
            else self._norm_log_survival + segment_log_fidelity
        )
        segment_fidelity = fidelity_from_log(segment_log_fidelity)
        cumulative_fidelity = fidelity_from_log(cumulative_log_fidelity)
        cumulative_infidelity = infidelity_from_log(cumulative_log_fidelity)
        self._current_infidelity = infidelity_from_log(segment_log_fidelity)
        self._norm_segment_open = True
        self._compression_norm_events.append({
            "step": len(self._compression_norm_events) + 1,
            "kind": str(kind),
            "valid": True,
            "expected_norm": float(max(0.0, before_norm_sq) ** 0.5),
            "observed_norm": float(observed_norm_sq ** 0.5),
            "fidelity_raw": float(raw),
            "local_fidelity": local_fidelity,
            "local_infidelity": local_infidelity,
            "segment_fidelity": float(segment_fidelity),
            "segment_infidelity": infidelity_from_log(segment_log_fidelity),
            "cumulative_fidelity": float(cumulative_fidelity),
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": float(cumulative_fidelity),
            "cumulative_compression_infidelity": cumulative_infidelity,
            "stabilized": bool(self.stabilize_unitary),
        })

    def _stabilize_unitary_norm(
        self,
        target_norm_sq: Optional[float],
        observed_infidelity: Optional[float],
    ) -> None:
        """Restore the pre-compression working norm without changing fidelity data."""
        if (
            not self.stabilize_unitary
            or target_norm_sq is None
            or observed_infidelity is None
            or not self._infidelity_valid
        ):
            return
        target_norm = float(max(0.0, target_norm_sq) ** 0.5)
        observed_norm = float(self._norm_squared() ** 0.5)
        if (
            target_norm <= 0.0
            or observed_norm <= 0.0
            or not np.isfinite(target_norm)
            or not np.isfinite(observed_norm)
        ):
            raise FloatingPointError(
                "Cannot stabilize a unitary compression with a zero or "
                "non-finite retained norm."
            )
        if not np.isclose(target_norm, observed_norm, rtol=1e-14, atol=1e-15):
            center_site = self._canonize_p_single()
            center = self.state.p[self.state.p.site_tag(int(center_site))]
            center.modify(data=center.data * (target_norm / observed_norm))
        self.state.info["cur_orthog"] = (
            int(self.state.info["cur_orthog"][0]),
            int(self.state.info["cur_orthog"][1]),
        )

    def _invalidate_infidelity(self) -> None:
        """Stop unitary norm-loss reporting after an unnormalized update."""
        self._infidelity_valid = False
        self._current_infidelity = None
        self._compression_segment_log_survival = 0.0
        self._norm_segment_open = False

    def _reset_infidelity(self) -> None:
        """Start a fresh normalized unitary segment after projection."""
        self._infidelity_valid = True
        self._current_infidelity = 0.0
        self._compression_segment_log_survival = 0.0
        self._norm_segment_open = False

    def _make_norm_event(
        self,
        kind: str,
        *,
        branch_probability: Optional[float] = None,
        projector_branch_probability: Optional[float] = None,
    ) -> Optional[NormEventRecord]:
        """Snapshot the current unitary segment before projective normalization."""
        if branch_probability is not None:
            branch_probability = min(1.0, max(0.0, float(branch_probability)))
        if projector_branch_probability is None:
            projector_branch_probability = branch_probability
        elif projector_branch_probability is not None:
            projector_branch_probability = min(
                1.0, max(0.0, float(projector_branch_probability))
            )

        event = NormEventRecord(
            kind=str(kind),
            valid=bool(self._infidelity_valid),
            branch_probability=branch_probability,
            projector_branch_probability=projector_branch_probability,
        )
        if not self._infidelity_valid:
            return event

        if self._norm_segment_open:
            segment_infidelity = float(self._current_infidelity)
        else:
            infidelity = self._unitary_infidelity()
            if infidelity is None:
                event["valid"] = False
                return event
            segment_infidelity = float(infidelity)
            segment_fidelity = max(0.0, 1.0 - segment_infidelity)
            self._compression_segment_log_survival = (
                -math.inf
                if segment_fidelity == 0.0
                else math.log(segment_fidelity)
            )
            self._norm_segment_open = True
        segment_fidelity = max(0.0, min(1.0, 1.0 - segment_infidelity))
        norm_sq = self._norm_squared()
        event.update(
            pre_norm=float(norm_sq ** 0.5),
            pre_norm_sq=float(norm_sq),
            segment_infidelity=segment_infidelity,
            segment_fidelity=segment_fidelity,
        )
        return event

    def _commit_norm_event(
        self,
        event: Optional[NormEventRecord],
        *,
        projected_norm: Optional[float] = None,
    ) -> None:
        """Record a pre-normalization event after projection succeeded."""
        if event is None:
            return
        if not isinstance(event, NormEventRecord):
            event = NormEventRecord(**dict(event))
        if projected_norm is not None:
            projected_norm = float(projected_norm)
            projected_norm_sq = max(0.0, projected_norm * projected_norm)
            event["projected_norm"] = projected_norm
            event["projected_norm_sq"] = projected_norm_sq
            pre_norm_sq = event.get("pre_norm_sq")
            branch_probability = event.get("projector_branch_probability")
            if (
                event.get("valid")
                and pre_norm_sq is not None
                and branch_probability is not None
            ):
                expected_norm_sq = max(
                    0.0,
                    float(pre_norm_sq) * float(branch_probability),
                )
                event["expected_projected_norm_sq"] = expected_norm_sq
                event["expected_projected_norm"] = float(expected_norm_sq ** 0.5)
                if expected_norm_sq > 0.0:
                    survival_raw = projected_norm_sq / expected_norm_sq
                    survival = min(1.0, max(0.0, survival_raw))
                    event["projector_survival_raw"] = float(survival_raw)
                    event["projector_survival"] = float(survival)
                    event["projector_infidelity"] = float(1.0 - survival)
        post_norm = self.norm()
        event["post_norm"] = post_norm
        event["post_norm_sq"] = float(post_norm * post_norm)
        self.norm_events.append(event)
        # Keep the cumulative product in log space. This is both cheaper than
        # repeatedly contracting the full state and stable for long streams
        # whose survival can underflow in ordinary floating point products.
        if event.get("valid") and event.get("segment_infidelity") is not None:
            survival = max(
                0.0,
                min(1.0, 1.0 - float(event["segment_infidelity"])),
            )
            projector_survival = event.get("projector_survival")
            if projector_survival is not None:
                survival *= max(0.0, min(1.0, float(projector_survival)))
            self._accumulate_norm_survival(survival)
            event["cumulative_fidelity"] = (
                0.0
                if self._norm_log_survival == -math.inf
                else float(math.exp(self._norm_log_survival))
            )
            event["cumulative_infidelity"] = (
                1.0
                if self._norm_log_survival == -math.inf
                else float(-math.expm1(self._norm_log_survival))
            )

    def _accumulate_norm_survival(self, survival: float) -> None:
        """Accumulate one validated norm-survival factor in log space."""
        survival = max(0.0, min(1.0, float(survival)))
        if survival == 0.0:
            self._norm_log_survival = -math.inf
        elif np.isfinite(self._norm_log_survival):
            self._norm_log_survival += math.log(survival)

    def norm_diagnostics(self, *, include_current: bool = True) -> dict:
        """Summarize segmented unitary norm-loss diagnostics.

        The completed segments are the pre-normalization snapshots in
        :attr:`norm_events`. If ``include_current`` is true and the current
        segment has emitted at least one compressed-unitary sample, its current
        norm is also folded into the product/geometric summaries. The returned
        values are compression/norm-survival proxies only; measurement branch
        probabilities are kept in the individual events and are not multiplied
        into the truncation total. Per-update local ratios are available from
        :meth:`get_compression_norm_events`. Dense non-unitary matrix updates
        contribute their ``G^dagger G``-normalized compression loss. These
        values are not target-state overlaps. ``state_norm`` and ``norm`` are
        the live coefficient-MPS norm; ``cumulative_norm`` is the square-root
        retained-compression proxy.
        """
        completed = [
            event
            for event in self.norm_events
            if event.get("valid") and event.get("segment_infidelity") is not None
        ]
        completed_unitary_losses = [
            float(event["segment_infidelity"]) for event in completed
        ]
        completed_projector_losses = [
            float(event["projector_infidelity"])
            for event in completed
            if event.get("projector_infidelity") is not None
        ]
        completed_nonunitary_losses = list(self._nonunitary_infidelities)
        completed_survivals = []
        for event, unitary_loss in zip(completed, completed_unitary_losses):
            unitary_survival = min(1.0, max(0.0, 1.0 - unitary_loss))
            projector_survival = event.get("projector_survival")
            if projector_survival is None:
                projector_survival = 1.0
            completed_survivals.append(
                unitary_survival * min(1.0, max(0.0, float(projector_survival)))
            )
        survivals = list(completed_survivals)
        survivals.extend(
            min(1.0, max(0.0, 1.0 - loss))
            for loss in completed_nonunitary_losses
        )
        current_loss = None
        if (
            include_current
            and self._infidelity_valid
            and self._norm_segment_open
            and self._current_infidelity is not None
        ):
            current_loss = float(self._current_infidelity)
            survivals.append(min(1.0, max(0.0, 1.0 - current_loss)))

        if survivals:
            current_log_survival = 0.0
            if current_loss is not None:
                current_survival = max(0.0, min(1.0, 1.0 - current_loss))
                current_log_survival = (
                    -math.inf
                    if current_survival == 0.0
                    else math.log(current_survival)
                )
            log_survival = self._norm_log_survival + current_log_survival
            total_survival = (
                0.0 if log_survival == -math.inf else float(math.exp(log_survival))
            )
            if any(survival <= 0.0 for survival in survivals):
                geometric_mean_survival = 0.0
            else:
                geometric_mean_survival = float(
                    math.exp(sum(math.log(survival) for survival in survivals)
                             / len(survivals))
                )
            event_losses = [1.0 - survival for survival in completed_survivals]
            event_losses.extend(completed_nonunitary_losses)
            if current_loss is not None:
                event_losses.append(current_loss)
            mean_segment_infidelity = float(sum(event_losses) / len(event_losses))
            max_segment_infidelity = float(max(event_losses))
        else:
            log_survival = None
            total_survival = None
            geometric_mean_survival = None
            mean_segment_infidelity = None
            max_segment_infidelity = None

        current_norm = (
            None if current_loss is None
            else float(max(0.0, 1.0 - current_loss) ** 0.5)
        )
        cumulative_infidelity = (
            None
            if total_survival is None
            else infidelity_from_log(log_survival)
        )
        norm_survival = total_survival
        cumulative_norm = (
            None if total_survival is None else float(total_survival ** 0.5)
        )
        state_norm = float(self.norm())
        latest_compression = (
            self._compression_norm_events[-1]
            if self._compression_norm_events
            else None
        )
        return {
            "tracking": True,
            "norm_tracking": True,
            # MPS-STN has no Tree-style per-edge spectrum tracker. Keep the
            # explicit field present so cross-backend diagnostics can branch
            # on one stable schema without mistaking ``None`` for disabled
            # norm tracking.
            "truncation_tracking": None,
            "current_valid": bool(self._infidelity_valid),
            "events": len(self.norm_events),
            "norm_events_count": len(self.norm_events),
            "completed_events": len(completed),
            "completed_segments": len(completed),
            "segments_including_current": len(survivals),
            "completed_segment_norms": [event["pre_norm"] for event in completed],
            "completed_segment_infidelities": [
                event["segment_infidelity"] for event in completed
            ],
            "completed_projector_infidelities": [
                event["projector_infidelity"] for event in completed
            ],
            "completed_nonunitary_infidelities": completed_nonunitary_losses,
            "completed_combined_infidelities": [
                float(1.0 - survival) for survival in completed_survivals
            ],
            "compression_events": len(self._compression_norm_events),
            "compression_norm_events": self.get_compression_norm_events(),
            "current_segment_norm": current_norm,
            "current_segment_infidelity": current_loss,
            "current_fidelity": (
                None if current_loss is None else float(1.0 - current_loss)
            ),
            "current_infidelity": current_loss,
            # Local means the most recent compressed update, matching the
            # ordinary MPS and Tree ledgers. The current-segment fields above
            # remain available for the boundary-aware STN history.
            "local_fidelity": (
                None
                if latest_compression is None
                else latest_compression["local_fidelity"]
            ),
            "local_infidelity": (
                None
                if latest_compression is None
                else latest_compression["local_infidelity"]
            ),
            "local_norm": (
                None
                if latest_compression is None
                else float(latest_compression["local_fidelity"] ** 0.5)
            ),
            # Provenance alias for the norm-derived cumulative fidelity; it is
            # distinct from the live state norm returned below.
            "norm_survival": norm_survival,
            "cumulative_fidelity": norm_survival,
            "cumulative_infidelity": cumulative_infidelity,
            "cumulative_compression_fidelity": norm_survival,
            "cumulative_compression_infidelity": cumulative_infidelity,
            # MpsOptimizer-compatible public names. ``infidelity`` is the
            # cumulative multiplicative compression infidelity; it never
            # includes stochastic measurement branch probabilities.
            "fidelity": norm_survival,
            "infidelity": cumulative_infidelity,
            # ``norm`` is the live coefficient-MPS norm. The retained
            # compression proxy is separate as ``cumulative_norm``.
            "norm": state_norm,
            "state_norm": state_norm,
            "cumulative_norm": cumulative_norm,
            "total_survival_proxy": norm_survival,
            "total_infidelity_proxy": cumulative_infidelity,
            "total_norm_proxy": cumulative_norm,
            "geometric_mean_survival": geometric_mean_survival,
            "geometric_mean_norm": (
                None if geometric_mean_survival is None
                else float(geometric_mean_survival ** 0.5)
            ),
            "mean_segment_infidelity": mean_segment_infidelity,
            "max_segment_infidelity": max_segment_infidelity,
            "mean_unitary_segment_infidelity": (
                None if not completed_unitary_losses
                else float(
                    sum(completed_unitary_losses) / len(completed_unitary_losses)
                )
            ),
            "max_unitary_segment_infidelity": (
                None if not completed_unitary_losses
                else float(max(completed_unitary_losses))
            ),
            "mean_projector_infidelity": (
                None if not completed_projector_losses
                else float(
                    sum(completed_projector_losses) / len(completed_projector_losses)
                )
            ),
            "max_projector_infidelity": (
                None if not completed_projector_losses
                else float(max(completed_projector_losses))
            ),
            "current_event_kind": (
                None if not self.norm_events else self.norm_events[-1].get("kind")
            ),
        }

    def _require_nonzero_state(self, action: str) -> float:
        """Return the norm squared or reject a normalized zero-state operation."""
        norm_squared = self._norm_squared()
        if not np.isfinite(norm_squared):
            raise ValueError(
                f"Cannot {action}: coefficient state has invalid norm squared "
                f"{norm_squared!r}."
            )
        if norm_squared <= 0.0:
            raise ValueError(
                f"Cannot {action} a zero-norm state; normalized probabilities "
                "and expectation values are undefined."
            )
        return norm_squared

    # ------------------------------------------------------------------ #
    # Canonical-centre tracking for the coefficient MPS ``|nu>``
    # ------------------------------------------------------------------ #
    def _ensure_p_center(self) -> None:
        """Guarantee a concrete tracked orthogonality centre (never a blind scan).

        When the centre is unknown (fresh state, or invalidated by a full
        rebuild such as an operator-sum branch) it is established by a single
        full-span canonicalization to site ``0`` rather than a
        ``calc_current_orthog_center`` rescan.
        """
        info = self.state.info
        if info.get("cur_orthog") not in (None, "calc"):
            return
        p = self.state.p
        L = int(getattr(p, "L", 0))
        if L <= 0:
            return
        p.canonize(
            [0],
            cur_orthog=(0, max(0, L - 1)),
            info=info,
        )
        info["cur_orthog"] = (0, 0)

    def _canonize_p_single(self) -> int:
        """Reduce the tracked centre to a single site and return it."""
        self._ensure_p_center()
        lo, hi = self.state.info["cur_orthog"]
        if lo != hi:
            self._canonize_p(lo)
            return lo
        return lo

    def _canonize_p(self, site) -> int:
        """Move the coefficient-MPS orthogonality centre to ``site`` (tracked)."""
        self._ensure_p_center()
        site = int(site)
        info = self.state.info
        self.state.p.canonize([site], cur_orthog=info["cur_orthog"], info=info)
        info["cur_orthog"] = (site, site)
        return site

    def _renorm_p_at(self, site) -> float:
        """Rescale the canonical centre tensor at ``site`` to unit norm.

        Raises when the centre norm is ~0, which means a projective collapse hit
        a ~0-probability (e.g. forced / post-selected) outcome. Returns the
        represented norm immediately before the normalization.
        """
        center = self.state.p[self.state.p.site_tag(int(site))]
        nrm = float(abs(self._to_scalar(center.norm())))
        if nrm < 1e-12:
            raise ValueError(
                "projective collapse produced a ~0-norm coefficient state; the "
                f"measured/forced outcome has ~0 probability (centre norm={nrm:.2e})."
            )
        exponent = float(getattr(self.state.p, "exponent", 0.0))
        represented_norm = float(nrm * (10.0 ** exponent))
        center.modify(data=center.data / nrm)
        # Quimb stores an additional base-10 network scale separately from the
        # tensors. The centre is now normalized, so that scale must be cleared.
        self.state.p.exponent = 0.0
        return represented_norm

    def pseudo_stabilizer_rank(self, tol: float = 1e-12) -> int:
        """Pseudo-stabilizer rank ``xi_tilde`` = number of non-zero ``nu_i``."""
        return self.state.pseudo_stabilizer_rank(tol=tol)

    @classmethod
    def truncation_convergence(
        cls,
        n,
        gates,
        chi_values=(1, 2, 4, 8, None),
        *,
        observable=None,
        **kwargs,
    ):
        """Replay a stream at several bond caps and report convergence.

        This is a correctness/benchmark helper for QEC and coherent-noise
        studies. Each row is an independent product-state replay, so the
        result directly exposes how ``chi`` changes the final norm, tracked
        compression diagnostics, peak bond, and an optional user observable.
        ``chi=None`` is the lossless reference up to the configured cutoff.
        """
        values = tuple(chi_values)
        if not values:
            raise ValueError("chi_values must contain at least one bond cap.")
        rows = []
        for chi_value in values:
            options = dict(kwargs)
            options["chi"] = chi_value
            optimizer = cls(int(n), gates=gates, **options)
            optimizer.run()
            row = {
                "chi": chi_value,
                "max_bond": int(optimizer.state.max_bond()),
                "norm": float(optimizer.norm()),
                "norm_diagnostics": optimizer.norm_diagnostics(),
            }
            if callable(observable):
                row["observable"] = observable(optimizer)
            rows.append(row)
        return rows

    def copy(self) -> "MpsStabOptimizer":
        """Return an independent copy (state deep-copied; queue/history reset)."""
        copied = MpsStabOptimizer(
            self.state.copy(),
            chi=self.chi,
            mode=self.mode,
            cutoff=self.cutoff,
            operator_tol=self.operator_tol,
            max_pauli_decomposition_qubits=self.max_pauli_decomposition_qubits,
            max_pauli_terms=self.max_pauli_terms,
            max_dense_cap_qubits=self.max_dense_cap_qubits,
            exact_cooling=self.exact_cooling,
            stabilize_unitary=self.stabilize_unitary,
            fit_init_strategy=self.fit_init_strategy,
            fit_init_rand_strength=self.fit_init_rand_strength,
            fit_init_seed=self.fit_init_seed,
            compression_seed=self.compression_seed,
            dtype=self.dtype,
            to_backend=self.to_backend,
        )
        copied._infidelity_valid = self._infidelity_valid
        copied._current_infidelity = self._current_infidelity
        copied._compression_segment_log_survival = (
            self._compression_segment_log_survival
        )
        copied._norm_segment_open = self._norm_segment_open
        copied._norm_log_survival = self._norm_log_survival
        copied.infidelities = list(self.infidelities)
        copied._nonunitary_infidelities = list(self._nonunitary_infidelities)
        copied._compression_norm_events = deepcopy(self._compression_norm_events)
        copied.norm_events = [
            NormEventRecord(**event.as_dict())
            if isinstance(event, NormEventRecord)
            else dict(event)
            for event in self.norm_events
        ]
        copied._localizer_cache = dict(self._localizer_cache)
        copied._initial_mps_length = self._initial_mps_length
        copied._mps_length_history = list(self._mps_length_history)
        copied.cap_history = deepcopy(self.cap_history)
        copied.logical_order = list(self.logical_order)
        copied._refresh_layout_map()
        copied.layout_plan = deepcopy(self.layout_plan)
        copied.last_layout_plan = deepcopy(self.last_layout_plan)
        copied.last_measurement_schedule = deepcopy(self.last_measurement_schedule)
        copied._last_fit_diagnostics = deepcopy(self._last_fit_diagnostics)
        copied._dmrg1_one_site_locked = bool(self._dmrg1_one_site_locked)
        copied._initial_state = copied.state.copy()
        copied._rng.bit_generator.state = deepcopy(self._rng.bit_generator.state)
        return copied

    # ------------------------------------------------------------------ #
    # Clifford gauge disentangling (p -> D p, C -> C D^dagger)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _disentangle_score(singular_values, tol: float) -> tuple[int, float]:
        """Rank/entropy score of one Schmidt spectrum for a local sweep.

        ``tol`` is a *relative singular-value* threshold.  It is used only to
        decide numerical rank; entropy always uses the normalized full spectrum
        so equal-rank candidates still prefer a less-entangled coefficient MPS.
        """
        singular_values = np.abs(np.asarray(singular_values).reshape(-1))
        if singular_values.size == 0 or singular_values.max(initial=0.0) == 0.0:
            return (0, 0.0)
        weights = singular_values**2
        weights /= weights.sum()
        rank = int(np.count_nonzero(singular_values > tol * singular_values.max()))
        entropy = float(-np.sum(weights[weights > 0.0] * np.log(weights[weights > 0.0])))
        return rank, entropy

    def _bond_singular_values(self, bond: int) -> np.ndarray:
        """Canonicalize at ``bond`` and return its Schmidt singular values.

        The canonicalization deliberately updates the live ``cur_orthog``
        tracker.  Candidate evaluation below only reads the resulting two-site
        tensor, so it never copies or mutates the live coefficient state.
        """
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        singular_values = self.state.p.singular_values(
            int(bond) + 1, info=self.state.info
        )
        return np.asarray(to_numpy(singular_values))

    def _candidate_bond_singular_values(self, bond: int, unitary) -> np.ndarray:
        """Return a candidate's central Schmidt values from the local MPS block.

        :meth:`_bond_singular_values` has put the MPS in mixed canonical form,
        hence the two virtual environments are isometric.  Applying a candidate
        only to that two-site tensor and SVDing it gives the exact score for the
        full MPS while avoiding twenty MPS copies and twenty global sweeps.
        """
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        p = self.state.p
        left = p[int(bond)]
        right = p[int(bond) + 1]
        (shared,) = left.bonds(right)
        physical_left = p.site_ind(int(bond))
        physical_right = p.site_ind(int(bond) + 1)
        left_outer = tuple(
            ind for ind in left.inds if ind not in (physical_left, shared)
        )
        right_outer = tuple(
            ind for ind in right.inds if ind not in (physical_right, shared)
        )
        left_data = np.asarray(to_numpy(
            left.transpose(*left_outer, physical_left, shared).data
        ))
        right_data = np.asarray(to_numpy(
            right.transpose(shared, physical_right, *right_outer).data
        ))
        shared_dim = left_data.shape[-1]
        left_dim = left_data.size // (2 * shared_dim)
        right_dim = right_data.size // (2 * shared_dim)
        pair = np.tensordot(left_data, right_data, axes=(-1, 0)).reshape(
            left_dim, 2, 2, right_dim
        )
        transformed = np.einsum(
            "abij,lijr->labr", np.asarray(unitary).reshape(2, 2, 2, 2), pair
        )
        return np.linalg.svd(
            transformed.reshape(2 * left_dim, 2 * right_dim), compute_uv=False
        )

    @staticmethod
    def _disentangle_bonds(bonds, n: int) -> tuple[int, ...]:
        """Validate and normalize a requested ordered sequence of MPS bonds."""
        if bonds is None:
            return tuple(range(n - 1))
        if isinstance(bonds, Integral) and not isinstance(bonds, (bool, np.bool_)):
            bonds = (int(bonds),)
        else:
            try:
                bonds = tuple(int(bond) for bond in bonds)
            except TypeError as exc:
                raise TypeError("bonds must be an integer, iterable of integers, or None.") from exc
        if any(bond < 0 or bond >= n - 1 for bond in bonds):
            raise ValueError(f"bonds must lie in [0, {n - 2}], got {bonds!r}.")
        return bonds

    def disentangle_cliffords(self, sweeps: int = 1, *, bonds=None,
                               tol: Optional[float] = None) -> list[dict]:
        """Reduce coefficient-MPS entanglement using local Clifford gauge moves.

        For each selected adjacent MPS bond, evaluate the 20 two-qubit Clifford
        classes modulo output-local Cliffords using the local Schmidt spectrum.
        If one improves the lexicographic ``(numerical rank, entropy)`` score,
        apply its representative ``D`` to ``|nu>`` and absorb ``D^dagger`` into
        the tableau.  Thus ``|psi> = C|nu>`` is unchanged (up to the explicitly
        requested numerical cutoff) while entanglement moves from ``|nu>`` into
        the free stabilizer frame.

        Parameters
        ----------
        sweeps : int
            Number of ordered left-to-right passes.  A pass stops early when no
            bond improves.  The usual periodic use needs only ``1``.
        bonds : int | iterable[int] | None
            MPS bond(s) to visit, represented by the left site index.  ``None``
            means all adjacent bonds in left-to-right order.
        tol : float | None
            Relative singular-value rank threshold and SVD compression cutoff.
            ``None`` uses this simulator's ``cutoff``.  Set ``tol=0`` for a
            strictly lossless numerical split (which may retain round-off-sized
            singular values rather than lower the stored bond dimension).

        Returns
        -------
        list[dict]
            One compact diagnostic dictionary per accepted local gauge move.
            The operation records one ``bond_history`` point but intentionally
            records no ``infidelities`` sample: it is a representation change,
            not a physical unitary time-evolution step.
        """
        if isinstance(sweeps, (bool, np.bool_)) or not isinstance(sweeps, Integral):
            raise TypeError("sweeps must be a nonnegative integer.")
        sweeps = int(sweeps)
        if sweeps < 0:
            raise ValueError("sweeps must be nonnegative.")
        if tol is None:
            tol = self.cutoff
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and nonnegative.")
        bonds = self._disentangle_bonds(bonds, self.n)
        moves = []
        if sweeps == 0 or not bonds:
            self._record()
            return moves

        import stim

        representatives = _two_qubit_clifford_representatives()
        for sweep in range(sweeps):
            improved = False
            for bond in bonds:
                before_svals = self._bond_singular_values(bond)
                before_score = self._disentangle_score(before_svals, tol)
                best_index = None
                best_score = before_score
                for index, (_, unitary) in enumerate(representatives):
                    score = self._disentangle_score(
                        self._candidate_bond_singular_values(bond, unitary), tol
                    )
                    if score < best_score:
                        best_index = index
                        best_score = score
                if best_index is None:
                    continue

                tableau, unitary = representatives[best_index]
                # The selected rank is no larger than the original rank.  Do
                # not impose ``self.chi`` here: this is a gauge transform, not
                # a physical evolution whose temporary split may be truncated.
                info = self.state.info
                self.state.p.gate_(
                    self._bk(unitary),
                    (bond, bond + 1),
                    contract="swap+split",
                    max_bond=None,
                    cutoff=tol,
                    info=info,
                    cur_orthog=info.get("cur_orthog"),
                )
                full_tableau = stim.Tableau(self.n)
                logical_targets = [
                    self.logical_order[int(bond)],
                    self.logical_order[int(bond) + 1],
                ]
                full_tableau.append(tableau, logical_targets)
                self.state.absorb_basis_clifford(full_tableau)
                moves.append({
                    "sweep": sweep,
                    "bond": bond,
                    "logical_bond": tuple(logical_targets),
                    "candidate": best_index,
                    "score_before": before_score,
                    "score_after": best_score,
                })
                improved = True
            if not improved:
                break
        self._record()
        return moves

    def _disentangle_event(self, params) -> list[dict]:
        """Dispatch ``("disentangle", ...)`` stream options to the public API."""
        if len(params) == 0:
            return self.disentangle_cliffords()
        if len(params) != 1:
            raise ValueError(
                '"disentangle" accepts no options, an integer sweep count, or one mapping.'
            )
        option = params[0]
        if isinstance(option, Integral) and not isinstance(option, (bool, np.bool_)):
            return self.disentangle_cliffords(sweeps=int(option))
        if not isinstance(option, Mapping):
            raise TypeError(
                '"disentangle" options must be an integer sweep count or a mapping.'
            )
        options = dict(option)
        unknown = set(options).difference({"sweeps", "bonds", "tol"})
        if unknown:
            raise ValueError(
                'Unknown "disentangle" options: ' + ", ".join(sorted(map(str, unknown)))
            )
        return self.disentangle_cliffords(**options)

    # ------------------------------------------------------------------ #
    # Backend helpers (place |nu> gates/MPOs on the configured backend)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_native_gate_matrix(gate, n_qubits):
        """Normalize dense-gate shape without converting backend arrays.

        The dense validation/classification helpers intentionally operate on
        NumPy arrays. This shape-only adapter is used at replay time so a
        differentiable backend tensor can reach the native one-qubit path
        unchanged. Non-array inputs fall back to the historical validator.
        """
        n_qubits = int(n_qubits)
        shape = getattr(gate, "shape", None)
        if shape is not None:
            shape = tuple(int(size) for size in shape)
            dimension = 2**n_qubits
            if shape == (dimension, dimension):
                return gate
            expected_shape = (2,) * (2 * n_qubits)
            if shape == expected_shape:
                return gate.reshape(dimension, dimension)
        return _as_gate_matrix(gate, n_qubits)

    @staticmethod
    def _gate_requires_grad(gate) -> bool:
        """Return whether a backend gate advertises an autodiff history."""
        return bool(getattr(gate, "requires_grad", False))

    @staticmethod
    def _backend_scalar_is_exact_zero(value) -> bool:
        """Test a scalar value for an exact zero without touching its graph."""
        detached = getattr(value, "detach", None)
        if callable(detached):
            value = detached()
        try:
            return bool(np.asarray(ar.to_numpy(value)).item() == 0)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _backend_scalar_has_nonzero_grad(value) -> bool:
        """Return whether a Torch scalar has a locally nonzero derivative.

        This is used only to distinguish structurally zero matrix entries from
        trainable entries that happen to evaluate to zero at an endpoint. It
        performs metadata inspection and ``autograd.grad`` queries, never a
        backward mutation of the caller's leaf gradients.
        """
        if not bool(getattr(value, "requires_grad", False)):
            return False
        grad_fn = getattr(value, "grad_fn", None)
        if grad_fn is None:
            # A leaf scalar is itself a trainable coordinate.
            return True
        leaves = []
        pending = [grad_fn]
        visited = set()
        while pending:
            function = pending.pop()
            if function in visited:
                continue
            visited.add(function)
            variable = getattr(function, "variable", None)
            if variable is not None:
                leaves.append(variable)
                continue
            pending.extend(
                parent
                for parent, _multiplicity in getattr(function, "next_functions", ())
                if parent is not None
            )
        if not leaves:
            return True
        try:
            import torch

            gradients = []
            for component in (value.real, value.imag):
                gradients.extend(
                    torch.autograd.grad(
                        component,
                        leaves,
                        allow_unused=True,
                        retain_graph=True,
                    )
                )
            return any(
                gradient is not None
                and bool(torch.any(gradient.detach() != 0).item())
                for gradient in gradients
            )
        except (RuntimeError, TypeError):
            # Be conservative if a backend does not expose a traversable
            # scalar graph: retaining the term is safer than losing a VJP.
            return True

    def _state_backend_like(self):
        """Return a representative live coefficient-MPS array."""
        for tensor in getattr(self.state.p, "tensors", ()):
            return tensor.data
        return None

    def backend_info(self):
        """Return and cache the live coefficient-MPS backend diagnostics."""
        info = backend_infer(self.state.p)
        signature = infer_backend_signature(self._state_backend_like())
        if signature != self._backend_signature:
            self._bk_cache.clear()
            self._backend_signature = signature
            self._backend_converter = (
                self._explicit_backend_converter
                or infer_backend_converter_from_sample(self._state_backend_like())
            )
        self.backend = info["backend"]
        self.backend_dtype = info["dtype"]
        self.backend_device = info["device"]
        self.array_backend = info.get("array_backend", info["backend"])
        # The live array dtype is authoritative when a caller supplies an
        # existing MPS. This keeps generated coefficient operators aligned
        # with the state rather than with STNState's constructor default.
        self.dtype = info["dtype"]
        self.state.dtype = self.dtype
        return info

    def _warn_backend_conversion(self, source_signature, target_signature, *, kind):
        """Warn once for an explicit stream payload conversion."""
        warning_key = (kind, source_signature, target_signature)
        if (
            source_signature[0] != "builtins"
            and warning_key not in self._backend_conversion_warnings
        ):
            self._backend_conversion_warnings.add(warning_key)
            warnings.warn(
                f"MpsStabOptimizer is converting a {kind} payload from "
                f"backend/dtype/device {source_signature!r} to the live "
                f"coefficient-MPS state {target_signature!r}; provide matching "
                f"{kind} payloads to avoid this transfer or cast.",
                UserWarning,
                stacklevel=3,
            )

    def _to_state_backend(self, array, *, warn=False, kind="operator"):
        """Return an array converted to the live coefficient-MPS signature."""
        like = self._state_backend_like()
        if like is None:
            return np.asarray(array, dtype=self.dtype)
        # The full diagnostic validates every MPS tensor and is intentionally
        # exposed through ``backend_info``. Internal rotations use the cached
        # live signature so backend checks do not become an O(n) scan inside
        # every gate/MPO contraction.
        if self._backend_signature is None:
            self.backend_info()
        target_signature = self._backend_signature
        source_signature = infer_backend_signature(array)
        if source_signature == target_signature:
            return array
        if warn:
            self._warn_backend_conversion(source_signature, target_signature, kind=kind)
        if target_signature[0] == "symmray" and source_signature[0] != "symmray":
            raise TypeError(
                "Cannot convert a dense gate/operator payload into a native "
                "Symmray MPS without charge and fermionic metadata. Build the "
                "payload as a Symmray array on the target U1/U1U1 backend."
            )
        converter = self._backend_converter
        if converter is not None:
            return converter(array)
        return ar.do("array", array, like=like)

    def _bk(self, mat):
        """Backend copy of an internally generated gate matrix."""
        arr = np.asarray(mat, dtype=self.dtype)
        return self._to_state_backend(arr)

    def _bk_const(self, tag: str, mat):
        """Backend copy of a *constant* gate matrix, cached by ``tag``."""
        cached = self._bk_cache.get(tag)
        if cached is None:
            cached = self._to_state_backend(np.asarray(mat, dtype=self.dtype))
            self._bk_cache[tag] = cached
        return cached

    def _bk_mpo(self, mpo, *, warn=True):
        """Return a sub-MPO on the live backend without mutating its source."""
        tensors = tuple(getattr(mpo, "tensors", ()))
        if not tensors:
            return mpo
        if self._backend_signature is None:
            self.backend_info()
        target_signature = self._backend_signature
        source_signatures = {
            infer_backend_signature(tensor.data) for tensor in tensors
        }
        if source_signatures and all(
            backend_signatures_compatible(source, target_signature)
            for source in source_signatures
        ):
            if source_signatures == {target_signature}:
                return mpo
            # Backend/device already match; only normalize an execution copy's
            # dtype when the operator implementation requires it.
        for source_signature in source_signatures:
            if (
                not backend_signatures_compatible(source_signature, target_signature)
                and warn
            ):
                self._warn_backend_conversion(
                    source_signature, target_signature, kind="sub-MPO"
                )
        if target_signature[0] == "symmray":
            for source_signature in source_signatures:
                if source_signature[0] != "symmray":
                    raise TypeError(
                        "Cannot convert a dense sub-MPO into a native Symmray "
                        "MPS without charge and fermionic metadata."
                    )
        converted = mpo.copy()
        apply_to_arrays = getattr(converted, "apply_to_arrays", None)
        if not callable(apply_to_arrays):
            raise TypeError(
                "sub-MPO payloads must provide apply_to_arrays() for backend "
                "conversion."
            )
        apply_to_arrays(self._backend_converter or self._to_state_backend)
        return converted

    @staticmethod
    def _to_scalar(x) -> complex:
        """Convert a (possibly backend) 0-d tensor/array to a Python complex."""
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        return complex(np.asarray(to_numpy(x)))

    @staticmethod
    def _gate_to_numpy(gate) -> np.ndarray:
        """Return a NumPy view/copy of a (possibly backend) gate matrix.

        Explicit gate matrices are classified and Pauli-decomposed with stim and
        NumPy, so a torch/cupy/jax array input is first materialized on the CPU.
        """
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        return np.asarray(to_numpy(gate))

    # ------------------------------------------------------------------ #
    # State primitives used by MpsStabSampler
    # ------------------------------------------------------------------ #
    def _sample_rng(self, seed):
        """Return the RNG used by sampler branch operations.

        The sampler owns shot generation and branch bookkeeping. The optimizer
        only supplies this state-local RNG hook so seeded calls preserve the
        historical behavior when sampling through either public API.
        """
        if seed is None:
            return self._rng
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)

    def _condition_computational_bit(
        self,
        terms,
        sign,
        bit: int,
        *,
        probability: Optional[float] = None,
    ) -> Optional[float]:
        """Condition this copied sampler branch on one fixed-frame bit.

        The sampler owns branch allocation and shot bookkeeping. This method
        performs only the coefficient-MPS projector update, leaving the
        tableau frame unchanged. A multi-site Pauli projector is routed
        through ``_evolve_p`` and therefore honors this optimizer's ``chi``,
        ``cutoff``, and DMRG/direct mode.
        """
        outcome = +1 if int(bit) == 0 else -1
        if probability is None:
            probability = self._outcome_probability(
                self._pauli_expectation(terms, sign), outcome
            )
        probability = float(probability)
        if probability <= 1e-12:
            return None
        # Keep the branch probability (Born rule) separate from the
        # projector's retained-norm diagnostic. ``_apply_projector`` passes
        # this event through the direct or DMRG compression path and fills in
        # ``projector_infidelity`` after the projected MPS is renormalized.
        norm_event = self._make_norm_event(
            "sample_measure_projector",
            branch_probability=probability,
            projector_branch_probability=probability,
        )
        self._apply_projector(terms, sign, outcome, norm_event=norm_event)
        return probability

    def _condition_absorbed_bit(
        self,
        pauli,
        where,
        bit: int,
        *,
        probability: Optional[float] = None,
    ) -> Optional[float]:
        """Condition this copied sampler branch with basis absorption enabled.

        This is deliberately a state-operation primitive rather than a public
        sampling method. The sampler owns branching and shot bookkeeping;
        this method owns the one-branch ``C^dagger O C`` localization, tableau
        update, and coefficient-MPS projection. Unlike the fixed-frame path,
        absorption changes ``C`` on this branch, so callers must recompute the
        next frame image from the returned state.
        """
        outcome = +1 if int(bit) == 0 else -1
        m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
        terms, sign = hermitian_pauli_terms(m_pauli)
        if probability is None:
            probability = self._outcome_probability(
                self._pauli_expectation(terms, sign), outcome
            )
        probability = float(probability)
        if probability <= 1e-12:
            return None
        self._absorb_measure(
            m_pauli,
            outcome,
            norm_event_kind="sample_measure_absorb",
        )
        return probability

    # Sampling compatibility delegates
    # ------------------------------------------------------------------ #
    # Sampling is implemented by MpsStabSampler, next to MpsSampler. Keep
    # these optimizer methods as thin compatibility shims for existing users
    # and trajectory/noise result objects that expose optimizer sampling.
    @staticmethod
    def pack_bit_samples(samples) -> np.ndarray:
        """Compatibility delegate for packing raw sampler bit arrays."""
        from ...sampling.stabilizer import MpsStabSampler

        return MpsStabSampler.pack_bit_samples(samples)

    def sample_bits(
        self,
        shots: int = 1,
        *,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
        absorb_basis: Optional[bool] = None,
        disentangle: Optional[bool] = None,
    ) -> np.ndarray:
        """Compatibility delegate to :class:`pepsy.MpsStabSampler`."""
        from ...sampling.stabilizer import MpsStabSampler

        return MpsStabSampler(
            self,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        ).sample_bits(
            shots,
            seed=seed,
            order=order,
            shuffle=shuffle,
            packed=packed,
            basis=basis,
        )

    def sample_basis(self, shots: int = 1, *, basis="Z", **kwargs):
        """Compatibility alias for :meth:`sample_bits`."""
        return self.sample_bits(shots, basis=basis, **kwargs)

    def sample_bitstrings(
        self,
        shots: int = 1,
        *,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
        absorb_basis: Optional[bool] = None,
        disentangle: Optional[bool] = None,
    ) -> np.ndarray:
        """Compatibility alias for :meth:`sample_bits`."""
        return self.sample_bits(
            shots,
            seed=seed,
            order=order,
            shuffle=shuffle,
            packed=packed,
            basis=basis,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        )

    def probability_bits(
        self,
        bits,
        *,
        order=None,
        basis="Z",
        seed=None,
        absorb_basis=None,
        disentangle=None,
    ) -> float:
        """Compatibility delegate for one product-basis probability."""
        from ...sampling.stabilizer import MpsStabSampler

        return MpsStabSampler(
            self,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        ).probability_bits(
            bits,
            order=order,
            basis=basis,
            seed=seed,
        )

    def probability_bits_many(
        self,
        bitstrings,
        *,
        order=None,
        basis="Z",
        seed=None,
        absorb_basis=None,
        disentangle=None,
    ) -> np.ndarray:
        """Compatibility delegate for many product-basis probabilities."""
        from ...sampling.stabilizer import MpsStabSampler

        return MpsStabSampler(
            self,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        ).probability_bits_many(
            bitstrings,
            order=order,
            basis=basis,
            seed=seed,
        )

    def bitstring_probability(
        self,
        bits,
        *,
        order=None,
        basis="Z",
        seed=None,
        absorb_basis=None,
        disentangle=None,
    ) -> float:
        """Compatibility alias for :meth:`probability_bits`."""
        return self.probability_bits(
            bits,
            order=order,
            basis=basis,
            seed=seed,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        )

    def bitstring_probabilities(
        self,
        bitstrings,
        *,
        order=None,
        basis="Z",
        seed=None,
        absorb_basis=None,
        disentangle=None,
    ) -> np.ndarray:
        """Compatibility alias for :meth:`probability_bits_many`."""
        return self.probability_bits_many(
            bitstrings,
            order=order,
            basis=basis,
            seed=seed,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        )

    def iter_sample_bits(
        self,
        shots: int,
        *,
        chunk_size: int,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
        absorb_basis: Optional[bool] = None,
        disentangle: Optional[bool] = None,
    ):
        """Compatibility delegate for chunked bit sampling."""
        from ...sampling.stabilizer import MpsStabSampler

        yield from MpsStabSampler(
            self,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        ).iter_sample_bits(
            shots,
            chunk_size=chunk_size,
            seed=seed,
            order=order,
            shuffle=shuffle,
            packed=packed,
            basis=basis,
        )

    def iter_sample_bitstrings(
        self,
        shots: int,
        *,
        chunk_size: int,
        seed=None,
        order=None,
        shuffle: bool = True,
        packed: bool = False,
        basis="Z",
        absorb_basis: Optional[bool] = None,
        disentangle: Optional[bool] = None,
    ):
        """Compatibility alias for :meth:`iter_sample_bits`."""
        yield from self.iter_sample_bits(
            shots,
            chunk_size=chunk_size,
            seed=seed,
            order=order,
            shuffle=shuffle,
            packed=packed,
            basis=basis,
            absorb_basis=absorb_basis,
            disentangle=disentangle,
        )

    # ------------------------------------------------------------------ #
    # Entry dispatch
    # ------------------------------------------------------------------ #
    def _apply_conditional_entry(self, entry):
        """Apply one feed-forward action when its recorded bit is true."""
        _name, payload, _where = conditional_event_parts(entry)
        index, expected = _resolve_conditional(payload, len(self.measurements))
        record = self.measurements[index]
        outcome = int(getattr(record, "outcome", record[2]))
        if int(outcome < 0) == expected:
            self._apply_entry(payload["action"])
        return self

    @contextmanager
    def _compatible_torch_linalg(self):
        """Use a dtype-compatible Torch QR/SVD policy for one replay entry.

        Torch linalg registrations are process-global.  A preceding real-valued
        autodiff model can therefore leave Pepsy's real stabilized QR rule
        installed while this optimizer is replaying a complex coefficient MPS.
        The real rule must reject such inputs; temporarily switching only the
        policy mode keeps the replay correct and restores the caller's policy
        immediately afterwards.
        """
        sample = self._state_backend_like()
        is_complex_torch = (
            self.backend == "torch"
            and callable(getattr(sample, "is_complex", None))
            and bool(sample.is_complex())
        )
        if not is_complex_torch:
            yield
            return

        from ...backends import get_torch_linalg_config

        active = get_torch_linalg_config()
        if active is None or active.mode == "complex":
            yield
            return

        with replace(active, mode="complex").activated():
            yield

    def _apply_entry(self, entry) -> None:
        with self._compatible_torch_linalg():
            self._apply_entry_unscoped(entry)

    def _apply_entry_unscoped(self, entry) -> None:
        conditional = conditional_event_parts(entry)
        if conditional is not None:
            self._apply_conditional_entry(entry)
            return
        parts = submpo_event_parts(entry, normalize_where=True)
        if parts is not None:
            mpo, where = parts
            # The complete stream was checked at installation. Do not scan or
            # recast this user MPO again while replaying the accepted queue.
            self._apply_submpo(mpo, where, _validate_backend=False)
            return

        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            head = entry[0]
            if isinstance(head, str):
                name = _normalize_event_name(head)
                if name == "disentangle":
                    self._disentangle_event(entry[1:])
                    return
                if name in _CLIFFORD_NAMES:
                    self.state.apply_clifford(name, *entry[1:])
                    self._record()
                    return
                if name in _ROTATION_AXES or name in _ROTATION_AXES_2Q or name in (
                    "rot", "t", "tdg",
                ):
                    self._apply_rotation(name, entry[1:])
                    return
                if name == "measure":
                    # ("measure", pauli, where[, outcome[, absorb_basis]])
                    pauli, where = entry[1], entry[2]
                    outcome = entry[3] if len(entry) > 3 else None
                    absorb = bool(entry[4]) if len(entry) > 4 else False
                    self.measure(pauli, where, outcome=outcome, absorb_basis=absorb)
                    return
                if name == "reset" or name in _RESET_AXIS_ALIASES:
                    # ("reset", where[, basis]) or ("reset_x", where)
                    axes, where = _parse_reset_args(
                        entry[1:],
                        default_axis=_RESET_AXIS_ALIASES.get(name),
                    )
                    self.reset(where, basis="".join(axes))
                    return
                if name in _MR_ALIASES or name in _MR_AXIS_ALIASES:
                    # ("measure_reset", basis, where[, outcome[, absorb_basis]])
                    axes, where, outcomes, absorb = _parse_measure_reset_args(
                        entry[1:],
                        default_axis=_MR_AXIS_ALIASES.get(name),
                    )
                    self.measure_reset(
                        "".join(axes),
                        where,
                        outcome=outcomes,
                        absorb_basis=absorb,
                    )
                    return
                if name == "cap":
                    # ("cap", where, vec[, absorb])
                    if len(entry) < 3:
                        raise ValueError('"cap" expects where and vec.')
                    absorb = _normalize_absorb(entry[3]) if len(entry) > 3 else "left"
                    self.cap(entry[1], entry[2], absorb=absorb)
                    return
                raise ValueError(f"Unknown gate name {head!r} in stream entry {entry!r}.")
            if len(entry) != 2:
                raise ValueError(f"Unsupported gate stream entry: {entry!r}.")
            # matrix form: (gate_tensor, where)
            gate, where = entry
            where = _normalize_sites(where)
            # Keep a trainable backend matrix native. ``_as_gate_matrix`` is
            # deliberately a NumPy/classification helper and would detach a
            # Torch gate before its Pauli coefficients reach ``|nu>``.
            self._apply_matrix(self._as_native_gate_matrix(gate, len(where)), where)
            return

        raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

    # ------------------------------------------------------------------ #
    # Non-Clifford rotations (|nu> update)
    # ------------------------------------------------------------------ #
    def _rotation_spec(self, name, params):
        """Return ``(theta, where, axes)`` for a rotation stream entry."""
        if name in _ROTATION_AXES:
            theta, q = float(params[0]), int(params[1])
            return theta, (q,), [_ROTATION_AXES[name]]
        if name in _ROTATION_AXES_2Q:
            theta, a, b = float(params[0]), int(params[1]), int(params[2])
            axis = _ROTATION_AXES_2Q[name]
            return theta, (a, b), [axis, axis]
        if name in ("t", "tdg"):
            (q,) = params
            theta = math.pi / 4 if name == "t" else -math.pi / 4
            return theta, (int(q),), ["Z"]
        if name == "rot":
            theta, paulis, where = params
            where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
            return float(theta), where, list(str(paulis))
        raise ValueError(f"Unknown rotation {name!r}.")

    @staticmethod
    def _is_clifford_angle(theta: float) -> bool:
        """Return whether ``exp(-i theta/2 P)`` is Clifford (theta a multiple of pi/2)."""
        k = theta / (math.pi / 2)
        return abs(k - round(k)) < 1e-9

    def _apply_clifford_rotation(self, theta, where, axes) -> None:
        """Apply a Clifford Pauli rotation to the tableau without dense matrices.

        The resulting Clifford depends only on the Pauli axes and the angle
        modulo ``2*pi``, so the directly synthesized tableau is cached.
        """
        import stim

        k = int(round(theta / (math.pi / 2))) % 4
        axes = tuple(str(axis).upper() for axis in axes)
        key = (axes, k)
        tableau = self._clifford_rot_cache.get(key)
        if tableau is None:
            circuit = stim.Circuit()
            # Ensure the tableau includes identity and trailing-identity sites.
            circuit.append("I", range(len(axes)))
            support = [q for q, axis in enumerate(axes) if axis != "I"]
            if support and k:
                pivot = support[0]

                # B P B^dagger = product(Z), then a CNOT parity network maps
                # product(Z) to Z on the pivot. Undoing both around Rz gives
                # exp(-i k*pi/4 P), up to the global phase omitted by tableaus.
                for q in support:
                    if axes[q] == "X":
                        circuit.append("H", [q])
                    elif axes[q] == "Y":
                        circuit.append("S_DAG", [q])
                        circuit.append("H", [q])
                for q in support:
                    if q != pivot:
                        circuit.append("CX", [q, pivot])

                circuit.append({1: "S", 2: "Z", 3: "S_DAG"}[k], [pivot])

                for q in reversed(support):
                    if q != pivot:
                        circuit.append("CX", [q, pivot])
                for q in reversed(support):
                    if axes[q] == "X":
                        circuit.append("H", [q])
                    elif axes[q] == "Y":
                        circuit.append("H", [q])
                        circuit.append("S", [q])

            tableau = stim.Tableau.from_circuit(circuit)
            self._clifford_rot_cache[key] = tableau
        self.state.do_tableau(tableau, where)
        self._record()

    @staticmethod
    def _stabilizer_product_eigenstate(vector, *, tol: float = 1e-10):
        """Return the signed Pauli eigenstate of a normalized local vector.

        A site is usable by constructive exact cooling only when it is both
        isolated in the coefficient MPS and exactly a one-qubit stabilizer
        state. Returning ``(axis, sign)`` means ``sign * axis`` has eigenvalue
        ``+1`` on the vector; otherwise return ``None``.
        """
        from autoray import to_numpy  # pylint: disable=import-outside-toplevel

        vec = np.array(to_numpy(vector), dtype=complex, copy=True).reshape(-1)
        if vec.shape != (2,):  # pragma: no cover - guarded by the qubit MPS API
            return None
        norm = float(np.linalg.norm(vec))
        if norm <= tol:
            return None
        vec /= norm
        bloch = {
            axis: float(np.real(np.vdot(vec, pauli_matrix(axis) @ vec)))
            for axis in ("X", "Y", "Z")
        }
        axis = max(bloch, key=lambda key: abs(bloch[key]))
        if abs(abs(bloch[axis]) - 1.0) > tol:
            return None
        if any(abs(bloch[other]) > tol for other in bloch if other != axis):
            return None
        return axis, (1 if bloch[axis] >= 0.0 else -1)

    @staticmethod
    def _exact_cooling_basis_tableau(axis: str, sign: int):
        """Build a one-qubit Clifford mapping ``sign * axis`` to ``+Z``."""
        import stim

        axis = str(axis).upper()
        sign = int(sign)
        sim = stim.TableauSimulator()
        sim.set_num_qubits(1)
        if sign < 0:
            {"X": sim.z, "Y": sim.x, "Z": sim.x}[axis](0)
        if axis == "X":
            sim.h(0)
        elif axis == "Y":
            sim.s_dag(0)
            sim.h(0)
        elif axis != "Z":  # pragma: no cover - internal Pauli validation
            raise ValueError(f"Unknown Pauli axis {axis!r}.")
        return sim.current_inverse_tableau().inverse()

    def _exact_controlled_pauli_tableau(self, pivot, pivot_axis, pivot_sign, terms):
        """Return the Clifford cascade for one constructive exact-cooling step.

        The pivot starts in the ``+1`` eigenspace of
        ``pivot_sign * pivot_axis``. The resulting Clifford applies the Pauli
        string on the remaining support exactly when the local rotation flips
        that eigenvalue.
        """
        import stim

        pivot = int(pivot)
        basis = self._exact_cooling_basis_tableau(pivot_axis, pivot_sign)
        sim = stim.TableauSimulator()
        sim.set_num_qubits(self.n)
        sim.do_tableau(basis, [pivot])
        for target, axis in terms.items():
            if int(target) == pivot:
                continue
            target = int(target)
            axis = str(axis).upper()
            if axis == "X":
                sim.cnot(pivot, target)
            elif axis == "Z":
                sim.cz(pivot, target)
            elif axis == "Y":
                # CY = S(target) CX S-dagger(target), written in execution order.
                sim.s_dag(target)
                sim.cnot(pivot, target)
                sim.s(target)
            else:  # pragma: no cover - internal Pauli validation
                raise ValueError(f"Unknown Pauli axis {axis!r}.")
        sim.do_tableau(basis.inverse(), [pivot])
        return sim.current_inverse_tableau().inverse()

    def _try_exact_cooling(self, theta, terms, sign) -> bool:
        """Apply the constructive exact-cooling identity when its pivot exists.

        For ``M = sign * A_i * Q`` and an isolated stabilizer coefficient site
        ``i`` whose stabilizer anticommutes with ``A_i``,
        ``R_M(theta)|nu> = G R_Ai(sign * theta)|nu>``. ``G`` is a
        controlled-Pauli Clifford, so it is absorbed into the tableau while the
        coefficient MPS receives only the local rotation.
        """
        if not self.exact_cooling or len(terms) < 2:
            return False

        p = self.state.p
        for pivot in sorted(terms, key=lambda site: (self._mps_site(site), int(site))):
            mps_pivot = self._mps_site(pivot)
            try:
                vector = self._product_site_vector(p, mps_pivot)
            except ValueError:
                continue
            stabilizer = self._stabilizer_product_eigenstate(vector)
            if stabilizer is None:
                continue
            pivot_axis, pivot_sign = stabilizer
            rotation_axis = terms[pivot]
            if rotation_axis == pivot_axis:
                continue  # commuting local Pauli: this site cannot be a pivot

            cascade = self._exact_controlled_pauli_tableau(
                pivot, pivot_axis, pivot_sign, terms
            )
            local_rotation = single_qubit_rotation_matrix(
                theta, rotation_axis, sign, self.dtype
            )
            p.gate_(self._bk(local_rotation), mps_pivot, contract=True)
            # ``absorb_basis_clifford(V)`` sends C -> C V-dagger. Here V = G-dagger,
            # hence the physical representation becomes C G R_Ai |nu>.
            self.state.absorb_basis_clifford(cascade.inverse())
            self.exact_cooling_events.append({
                "pivot": int(pivot),
                "mps_site": int(mps_pivot),
                "support": tuple(sorted(int(site) for site in terms)),
                "pivot_stabilizer": f"{'+' if pivot_sign > 0 else '-'}{pivot_axis}",
            })
            self._record()
            return True
        return False

    def _quimb_compress_opts(self, method):
        """Return the Quimb options shared by coefficient-MPO updates."""
        opts = {
            "cutoff": 0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else self.cutoff,
        }
        if method == "fit-projector":
            # Match the ordinary MPS path: projector fitting does not need the
            # optional pre-gauge and is safer on exact product-state bonds.
            opts["canonize"] = False
        return opts

    def _apply_quimb_submpo(self, p, mpo, where, *, method, max_bond, info):
        """Apply one coefficient-frame sub-MPO with the selected Quimb method."""
        method = self._normalize_quimb_method(method)
        requires_chi = {
            "src",
            "src-first",
            "src-oversample",
            "srcmps",
            "srcmps-first",
            "srcmps-oversample",
            "fit-oversample",
        }
        if max_bond is None and method in requires_chi:
            raise ValueError(
                f"MpsStabOptimizer mode {method!r} requires a finite chi."
            )

        opts = self._quimb_compress_opts(method)
        is_interior = min(where) > 0 or max(where) < int(p.L) - 1
        use_workaround = (
            method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
            and is_interior
            and (max_bond is not None or method.startswith("fit-"))
        )
        if use_workaround:
            return _apply_submpo_with_interior_workaround(
                p,
                mpo,
                where,
                chi=max_bond,
                method=method,
                cutoff=self.cutoff,
                cutoff_mode=None,
                info=info,
                inplace_mpo=False,
                seed=self.compression_seed,
            )

        seed = (
            self.compression_seed
            if method in _MPO_METHODS_USE_SEED
            else None
        )
        _run_seeded_quimb(
            seed,
            p.gate_with_submpo_,
            mpo,
            where=where,
            method=method,
            max_bond=max_bond,
            info=info,
            inplace_mpo=False,
            **opts,
        )
        return p

    def _apply_rotation(self, name, params) -> None:
        theta, where, axes = self._rotation_spec(name, params)
        # Validate the complete support before either the tableau or MPS changes.
        phys = pauli_string(axes, where, self.n)
        # Clifford rotations (angle a multiple of pi/2) are free: update the
        # tableau and leave |nu> untouched (paper's "Clifford = free" principle).
        if self._is_clifford_angle(theta):
            self._apply_clifford_rotation(theta, where, axes)
            return
        m_pauli = self.state.frame_pauli(phys)
        terms, sign = hermitian_pauli_terms(m_pauli)
        support = sorted(terms)
        if not support:  # global phase only; no state change
            self._record()
            return
        if self._try_exact_cooling(theta, terms, sign):
            return
        if len(support) == 1:
            q = support[0]
            mps_q = self._mps_site(q)
            umat = single_qubit_rotation_matrix(theta, terms[q], sign, self.dtype)
            # A single-qubit unitary preserves canonical form and the tracked
            # orthogonality centre, so it is applied without touching the tracker.
            self.state.p.gate_(self._bk(umat), mps_q, contract=True)
            self._record()
            return
        # Multi-qubit Pauli rotation: windowed bond-dim-2 sub-MPO applied only on
        # the support span via gate_with_submpo_ (skips identity sites entirely).
        c = np.cos(theta / 2)
        coef = -1j * sign * np.sin(theta / 2)
        mps_terms = self._mps_terms(terms)
        mpo, where = pauli_combo_submpo(c, coef, mps_terms, self.n, dtype=self.dtype)
        self._record(self._evolve_p(self._bk_mpo(mpo, warn=False), where, unitary=True))

    def _evolve_p(
        self,
        mpo,
        where,
        *,
        unitary: bool = False,
        renormalize: bool = False,
        norm_event: Optional[NormEventRecord] = None,
        norm_event_kind: str = "unitary_compression",
    ) -> Optional[float]:
        """Apply a windowed sub-MPO to the coefficient MPS ``p`` on ``where``.

        Only the ``[min(where), max(where)]`` region is canonicalized and
        compressed.  ``max_bond=None`` (exact) is lossless via the cutoff, which
        stops the bond-dim-2 MPO from doubling the bond on every application.
        """
        before_norm_sq = (
            self._norm_squared()
            if unitary and self._infidelity_valid
            else None
        )
        if self.mode in self._DMRG_MODES:
            self._evolve_p_dmrg(mpo, where)
        else:
            p = self.state.p
            self._ensure_p_center()
            method = (
                self._mode_quimb_method(self.mode)
                if self._is_quimb_mode(self.mode)
                else "direct"
            )
            self._apply_quimb_submpo(
                p,
                mpo,
                where,
                method=method,
                max_bond=None if self.mode == "exact" else self.chi,
                info=self.state.info,
            )
        observed_infidelity = self._unitary_infidelity() if unitary else None
        self._record_compression_norm_event(
            before_norm_sq,
            observed_infidelity,
            kind=norm_event_kind,
        )
        infidelity = (
            None
            if not unitary
            else self._current_infidelity
        )
        if unitary:
            self._stabilize_unitary_norm(before_norm_sq, observed_infidelity)
        if renormalize:
            site = self._canonize_p_single()
            projected_norm = self._renorm_p_at(site)
            self._reset_infidelity()
            self._commit_norm_event(norm_event, projected_norm=projected_norm)
        return infidelity

    @staticmethod
    def _fit_random_data(data, shape, *, strength, rng):
        """Generate backend-compatible random data for a disposable FIT guess."""
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

    def _fit_randomized_guess(self, p, where, *, block_size, expand):
        """Build a deterministic dense random FIT guess when rank can grow."""
        if (
            block_size not in {2, 3}
            or self.chi is None
            or self.fit_init_rand_strength == 0.0
            or self.backend == "symmray"
            or p.isfermionic()
        ):
            return p

        start, stop = min(where), max(where)
        try:
            active = FIT._active_bond_rank_targets(  # pylint: disable=protected-access
                p,
                start,
                stop,
                self.chi,
            )
        except (AttributeError, TypeError, ValueError):
            active = None
        if not active:
            return p

        guess = p.copy(deep=True)
        rng = np.random.default_rng(self.fit_init_seed)
        if expand:
            bonds = []
            for site, target_size in zip(range(start, stop), active):
                current_size = int(p.bond_size(site, site + 1))
                target_size = int(target_size)
                if current_size < target_size:
                    bonds.append((site, current_size, target_size))
            if not bonds:
                return p
            for target_size in sorted({target for _, _, target in bonds}):
                inds = [
                    guess.bond(site, site + 1)
                    for site, _, target in bonds
                    if target == target_size
                ]
                qtn.TensorNetwork.expand_bond_dimension(
                    guess,
                    target_size,
                    mode="zeros",
                    inds_to_expand=inds,
                    inplace=True,
                )
                for site, current_size, target in bonds:
                    if target != target_size:
                        continue
                    bond = guess.bond(site, site + 1)
                    for tensor in guess.tensors:
                        if bond not in tensor.inds:
                            continue
                        axis = tensor.inds.index(bond)
                        old_slices = [slice(None)] * tensor.ndim
                        old_slices[axis] = slice(0, current_size)
                        shape = list(tensor.shape)
                        shape[axis] = target_size - current_size
                        random_data = self._fit_random_data(
                            tensor.data,
                            shape,
                            strength=self.fit_init_rand_strength,
                            rng=rng,
                        )
                        tensor.modify(
                            data=ar.do(
                                "concatenate",
                                (
                                    tensor.data[tuple(old_slices)],
                                    random_data,
                                ),
                                axis=axis,
                            )
                        )
        else:
            for site in range(start, stop + 1):
                tensor = guess[site]
                tensor.modify(
                    data=ar.do(
                        "add",
                        tensor.data,
                        self._fit_random_data(
                            tensor.data,
                            tensor.shape,
                            strength=self.fit_init_rand_strength,
                            rng=rng,
                        ),
                    )
                )
        guess_info = {}
        guess.canonize([start, stop], info=guess_info)
        return guess

    def _fit_initial_guess(self, p, mpo, where, *, block_size):
        """Select an isolated FIT guess without changing the live coefficient MPS."""
        strategy = self._resolved_fit_init_strategy(self.fit_init_strategy)
        if (
            mpo is None
            or block_size not in {1, 2, 3}
            or self.chi is None
            or self.backend == "symmray"
            or p.isfermionic()
        ):
            return p

        start, stop = min(where), max(where)
        try:
            at_target = FIT._active_bonds_at_rank_targets(  # pylint: disable=protected-access
                p,
                start,
                stop,
                self.chi,
            )
        except (AttributeError, TypeError, ValueError):
            at_target = True
        # The SRC warm-up is also intentional after rank expansion: ordinary
        # MPS DMRG uses it to prepare the fixed-rank one-site phase. Keep
        # direct/random policies as no-op warm starts once the active bonds
        # are already at their attainable ceilings.
        if at_target and not strategy.startswith("guess_"):
            return p

        if strategy.startswith("guess_"):
            method = strategy[len("guess_") :]
            guess = p.copy(deep=True)
            self._apply_quimb_submpo(
                guess,
                mpo,
                where,
                method=method,
                max_bond=self.chi,
                info={},
            )
            return guess
        if strategy in {"random", "random_expand"}:
            return self._fit_randomized_guess(
                p,
                where,
                block_size=block_size,
                expand=strategy == "random_expand",
            )
        return p

    def _fit_window_at_rank_targets(self, p, start, stop):
        """Return whether all bonds in a DMRG window have reached their cap."""
        if self.chi is None or stop <= start:
            return False
        try:
            return bool(
                FIT._active_bonds_at_rank_targets(  # pylint: disable=protected-access
                    p,
                    int(start),
                    int(stop),
                    self.chi,
                )
            )
        except (AttributeError, TypeError, ValueError):
            # A conservative answer keeps the variational growth phase when a
            # backend cannot expose FIT's rank-ceiling helper.
            return False

    def _dmrg1_all_bonds_at_rank_targets(self, p):
        """Return whether every full-chain coefficient bond is at its ceiling."""
        if self.mode != "dmrg1" or self.chi is None or self.n < 2:
            return False
        return self._fit_window_at_rank_targets(p, 0, self.n - 1)

    def _maybe_lock_dmrg1_one_site_phase(self, p=None):
        """Latch DMRG1 into one-site refinement after full-chain growth."""
        if self.mode != "dmrg1" or self._dmrg1_one_site_locked:
            return
        if p is None:
            p = self.state.p
        if self._dmrg1_all_bonds_at_rank_targets(p):
            self._dmrg1_one_site_locked = True

    @staticmethod
    def _fit_target_is_layered(target):
        """Return whether ``target`` still contains an operator layer."""
        return len(getattr(target, "tensor_map", ())) > int(target.L)

    @staticmethod
    def _align_fit_submpo_tags(mpo, target, where):
        """Copy and align sub-MPO site tags with the fitted MPS tags.

        FIT accepts layered targets with several tensors per site, but every
        target tensor must carry exactly one tag from the fitted MPS site-tag
        family.  Native STN builders already use matching ``I{site}`` tags;
        this small alignment step also keeps user/MPO site-tag formatters
        compatible without mutating the operator retained for the warm start.
        """
        mpo = mpo.copy()
        site_tag = getattr(mpo, "site_tag", None)
        if not callable(site_tag):
            raise TypeError("DMRG layered targets require a site-tagged sub-MPO.")

        active_sites = tuple(sorted({int(site) for site in where}))
        expected_tags = set()
        for site in active_sites:
            old_tag = site_tag(site)
            new_tag = target.site_tag(site)
            expected_tags.add(new_tag)
            if old_tag != new_tag:
                mpo.retag_({old_tag: new_tag})

        for tensor in mpo.tensors:
            tensor_site_tags = tuple(tag for tag in tensor.tags if tag in expected_tags)
            if len(tensor_site_tags) != 1:
                raise ValueError(
                    "Each layered FIT sub-MPO tensor must carry exactly one "
                    f"fitted-MPS site tag, got {tuple(tensor.tags)!r}."
                )
        return mpo

    def _build_dmrg_fit_target(self, mpo, where):
        """Build the DMRG target, retaining a lazy sub-MPO when supported.

        Dense backends keep the coefficient sub-MPO as a separate operator
        layer.  The live MPS is only canonicalized on the active window before
        the layer is attached, so the target does not pay an intermediate
        ``chi``-independent MPS materialization cost.  Symmray and fermionic
        data retain the native materialized target path because their graded
        operator metadata cannot safely use the generic lazy FIT layer.
        """
        start, stop = min(where), max(where)
        target = self.state.p.copy()
        target.canonicalize_((start, stop), info={})

        self.backend_info()
        layered_supported = (
            self.backend != "symmray"
            and not target.isfermionic()
            and not mpo.isfermionic()
        )
        if layered_supported:
            layered_mpo = self._align_fit_submpo_tags(mpo, target, where)
            target.gate_with_op_lazy_(
                layered_mpo,
                inplace=True,
                inplace_op=False,
            )
            return target, "layered"

        # Native graded routes keep a one-tensor-per-site target and use the
        # existing backend-aware sub-MPO contraction path.
        target.gate_with_submpo_(
            mpo,
            where=where,
            max_bond=None,
            cutoff=0.0,
            info={},
            inplace_mpo=False,
        )
        return target, "mps"

    def _fit_coefficient_target(
        self,
        target,
        where,
        *,
        guess_mpo=None,
        target_strategy=None,
    ):
        """Fit a coefficient-MPS target with the selected DMRG schedule."""
        p = self.state.p
        start, stop = min(where), max(where)
        span = stop - start + 1
        if target_strategy is None:
            target_strategy = (
                "layered" if self._fit_target_is_layered(target) else "mps"
            )
        requested_block_size = 3 if self.mode == "dmrg3" else 2
        block_size = min(requested_block_size, span)
        self._maybe_lock_dmrg1_one_site_phase(p)
        if (
            self.mode == "dmrg1"
            and block_size == 2
            and span > 2
            and self._dmrg1_one_site_locked
        ):
            block_size = 1
        elif (
            self.mode == "dmrg1"
            and block_size == 2
            and span > 2
            and self._fit_window_at_rank_targets(p, start, stop)
        ):
            block_size = 1
        # A three-site FIT update cannot be defined on a two-site active
        # window.  Dense physical gates and mapped Pauli rotations commonly
        # have exactly this support, so match the ordinary MPS DMRG behavior
        # and fall back to a two-site update locally rather than rejecting a
        # valid ``dmrg3`` run.
        fit_guess = self._fit_initial_guess(
            p,
            guess_mpo,
            where,
            block_size=block_size,
        )
        fit = FIT(
            target,
            p=fit_guess,
            cutoffs=self.cutoff,
            retag=False,
            range_int=[start, stop],
            inplace=True,
            copy_target=False,
        )
        adjacent_two_site = span == 2 and block_size == 2
        growth_sweeps = (
            0 if block_size == 1 else (1 if adjacent_two_site else 2)
        )
        resolved_fit_init_strategy = self._resolved_fit_init_strategy(
            self.fit_init_strategy
        )
        guess_method = (
            resolved_fit_init_strategy[len("guess_") :]
            if resolved_fit_init_strategy.startswith("guess_")
            else None
        )
        guess_used = bool(
            guess_method is not None
            and guess_mpo is not None
            and block_size in {1, 2, 3}
            and self.chi is not None
            and self.backend != "symmray"
            and not p.isfermionic()
        )
        fit.run_gate(
            # A two-site gate is already the complete local problem. Match
            # MpsOptimizer's structural fast path and spend one FIT update on
            # it; longer windows use two growth sweeps and one-site handoff.
            n_iter=1 if adjacent_two_site else 3,
            block_size=block_size,
            sweep_sequence="RL",
            max_bond=self.chi,
            cutoff=self.cutoff,
            min_iter=1,
            rtol=None,
            patience=1,
            adaptive_block_sweeps=(
                None if block_size == 1 else growth_sweeps
            ),
            adaptive_until_rank=False,
            # The remaining ``n_iter`` sweep after the two growth sweeps is
            # FIT's one-site handoff. Do not add a second explicit refinement
            # sweep on top of that canonical MpsOptimizer schedule.
            final_one_site_sweeps=0,
            single_pair_fast_path=True,
            collect_split_diagnostics=False,
        )
        self.state.p = fit.p
        self._maybe_lock_dmrg1_one_site_phase(self.state.p)
        self._last_fit_diagnostics = {
            "backend": "fit",
            "mode": self.mode,
            "block_size": int(block_size),
            "growth_sweeps": int(getattr(fit, "adaptive_sweeps_run", growth_sweeps)),
            "one_site_refinement_sweeps": int(
                getattr(fit, "one_site_sweeps_run", 0)
            ),
            "iterations": int(getattr(fit, "iterations_run", 0)),
            "dmrg1_one_site_locked": bool(self._dmrg1_one_site_locked),
            "fit_init_strategy": resolved_fit_init_strategy,
            "fit_init_strategy_requested": self.fit_init_strategy,
            "guess_method": guess_method,
            "guess_used": guess_used,
            "target_strategy": target_strategy,
        }
        center = fit.final_center_site
        if center is None:
            center = stop
        self.state.info["cur_orthog"] = (int(center), int(center))

    def _evolve_p_dmrg(self, mpo, where):
        """Build a layered target and compress it with coefficient FIT."""
        target, target_strategy = self._build_dmrg_fit_target(mpo, where)
        self._fit_coefficient_target(
            target,
            where,
            guess_mpo=mpo,
            target_strategy=target_strategy,
        )

    # ------------------------------------------------------------------ #
    # Measurement (Lemma 3; non-unitary |nu> update)
    # ------------------------------------------------------------------ #
    def _phys_pauli(self, pauli, where):
        """Build the physical Pauli string for ``pauli`` on ``where``."""
        where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
        axes = list(str(pauli))
        if len(axes) != len(where):
            raise ValueError(f"Pauli {pauli!r} and where {where!r} have different lengths.")
        return pauli_string(axes, where, self.n)

    @staticmethod
    def _validate_outcome(outcome):
        """Return a forced Pauli outcome, requiring exactly integer +/-1."""
        if outcome is None:
            return None
        if isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, Integral):
            raise ValueError(f"outcome must be exactly +1 or -1, got {outcome!r}.")
        value = int(outcome)
        if value not in (-1, 1):
            raise ValueError(f"outcome must be exactly +1 or -1, got {outcome!r}.")
        return value

    @staticmethod
    def _outcome_probability(expectation, outcome):
        """Return a numerically clipped Pauli-outcome probability."""
        return min(max(0.5 * (1.0 + outcome * expectation), 0.0), 1.0)

    def _frame_terms(self, pauli, where):
        """Return ``({site: axis}, sign)`` for the ``|nu>``-frame image of a Pauli."""
        m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
        return hermitian_pauli_terms(m_pauli)

    def _pauli_expectation(self, terms, sign) -> float:
        """Return ``<p|M|p> / <p|p>`` for the Pauli ``M = sign * prod terms``."""
        p = self.state.p
        den = self._require_nonzero_state("compute an expectation value for")
        if not terms:  # M = sign * I
            return float(sign)
        m_p = p.copy()
        for site, axis in self._mps_terms(terms).items():
            m_p.gate_(self._bk_const("P" + axis, pauli_matrix(axis)), site, contract=True)
        num = self._to_scalar(p.H @ m_p)
        return float(sign * np.real(num / den))

    def expectation(self, pauli, where=None) -> float:
        """Return the expectation ``<psi|O|psi>`` of a Pauli observable (no collapse).

        Two forms:

        * ``expectation("Z", 0)`` / ``expectation("XZ", (0, 2))`` — a Pauli on
          the given qubit(s).
        * ``expectation("IZZ")`` — a full-register Pauli string (``where=None``),
          length ``n`` with ``"I"`` on idle qubits.
        """
        if where is None:
            if len(str(pauli)) != self.n:
                raise ValueError(
                    f"Full-register Pauli string must have length n={self.n}, "
                    f"got {len(str(pauli))}."
                )
            where = tuple(range(self.n))
        terms, sign = self._frame_terms(pauli, where)
        return self._pauli_expectation(terms, sign)

    def expectation_pauli_sum(self, terms) -> float:
        """Return ``<psi|H|psi>`` for ``H = sum_k coeff_k P_k`` (e.g. a Hamiltonian).

        ``terms`` is an iterable of ``(coeff, pauli)`` or ``(coeff, pauli, where)``
        entries; ``pauli``/``where`` follow the :meth:`expectation` conventions.
        """
        self._require_nonzero_state("compute an expectation value for")
        total = 0.0 + 0.0j
        for term in terms:
            coeff, pauli = term[0], term[1]
            where = term[2] if len(term) > 2 else None
            total += complex(coeff) * self.expectation(pauli, where)
        return float(np.real(total))

    def sync_canonicalization(self, site=None):
        """Re-establish coefficient-MPS metadata after external state access.

        Stabilizer measurement and projection paths pass ``state.info`` to
        Quimb's canonical routines. If a caller instead uses a lower-level
        method directly on ``state.p``, call this before resuming evolution.
        Ordinary diagnostic expectations already operate on a private copy and
        do not require synchronization.
        """
        p = self.state.p
        current = tuple(int(x) for x in p.calc_current_orthog_center())
        current = (min(current), max(current))
        if site is None:
            site = current[1]
        site = int(site)
        if not 0 <= site < int(p.L):
            raise ValueError(f"site must lie in [0, {int(p.L)}), got {site}.")
        p.canonize([site], cur_orthog=current, info=self.state.info)
        self.state.info["cur_orthog"] = (site, site)
        return self.state.info["cur_orthog"]

    def sample(self, pauli, where=None, *, shots: int = 1, seed=None):
        """Draw ``shots`` Born-rule outcomes (+/-1) of a Pauli observable.

        Independent samples of the *current* state; the state is **not**
        collapsed (unlike :meth:`measure`).  Useful for shot statistics.
        Returns a length-``shots`` numpy array of +/-1.
        """
        exp = self.expectation(pauli, where)
        p_plus = 0.5 * (1.0 + exp)
        rng = self._rng if seed is None else np.random.default_rng(seed)
        return np.where(rng.random(int(shots)) < p_plus, 1, -1)

    @staticmethod
    def _normalize_measurement_batch(measurements):
        """Normalize independent single-qubit measurements for scheduling."""
        if isinstance(measurements, (str, bytes)):
            raise TypeError(
                "measure_many expects an iterable of (pauli, qubit[, outcome]) "
                "entries."
            )
        try:
            entries = tuple(measurements)
        except TypeError as exc:
            raise TypeError(
                "measure_many expects an iterable of measurement entries."
            ) from exc

        operations = []
        targets = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
                raise ValueError(
                    "measure_many entries must be (pauli, qubit) or "
                    f"(pauli, qubit, outcome), got entry {index}: {entry!r}."
                )
            pauli, where = entry[:2]
            sites = _normalize_sites(where)
            if len(sites) != 1:
                raise ValueError(
                    "measure_many only schedules independent single-qubit "
                    f"measurements, got where={where!r}."
                )
            (axis,) = _normalize_pauli_axes(
                pauli,
                sites,
                event="measure_many",
            )
            outcome = entry[2] if len(entry) == 3 else None
            outcome = MpsStabOptimizer._validate_outcome(outcome)
            operations.append((axis, sites[0], outcome))
            targets.append(sites[0])

        if len(set(targets)) != len(targets):
            raise ValueError(
                "measure_many requires distinct target qubits so operations can "
                "be safely reordered."
            )
        return tuple(operations)

    def _measurement_span_info(self, axis, qubit, *, absorb_basis):
        """Return cheap Tableau/MPS-layout costs for one measurement candidate."""
        terms, _sign = self._frame_terms(axis, qubit)
        frame_support = tuple(sorted(terms))
        mps_support = tuple(sorted(self._mps_site(site) for site in frame_support))
        if not mps_support:
            return {
                "frame_support": frame_support,
                "mps_support": mps_support,
                "span": 0,
                "localizer_distance": 0,
                "pivot": None,
            }

        span = int(max(mps_support) - min(mps_support))
        ordered_support = sorted(
            frame_support,
            key=lambda site: (self._mps_site(site), int(site)),
        )
        pivot = ordered_support[len(ordered_support) // 2]
        pivot_position = self._mps_site(pivot)
        localizer_distance = int(
            sum(
                abs(self._mps_site(site) - pivot_position)
                for site in frame_support
            )
        )
        return {
            "frame_support": frame_support,
            "mps_support": mps_support,
            "span": span,
            "localizer_distance": (
                localizer_distance if absorb_basis else 0
            ),
            "pivot": int(pivot),
        }

    def _run_measurement_batch(
        self,
        operations,
        *,
        order,
        absorb_basis,
        reset=False,
        reset_after=False,
    ):
        """Run a batch using a metadata-only, adaptive span schedule."""
        if reset and reset_after:
            raise ValueError("reset and reset_after are mutually exclusive.")
        operations = tuple(operations)
        targets = tuple(operation[1] for operation in operations)
        normalized_order = _normalize_measurement_order(
            order,
            count=len(operations),
            targets=targets,
        )
        remaining = list(range(len(operations)))
        result = [None] * len(operations)
        schedule = []

        for step in range(len(operations)):
            if normalized_order == "input":
                input_index = remaining.pop(0)
            elif normalized_order == "min_span":
                candidate_info = {
                    index: self._measurement_span_info(
                        operations[index][0],
                        operations[index][1],
                        absorb_basis=absorb_basis,
                    )
                    for index in remaining
                }
                input_index = min(
                    remaining,
                    key=lambda index: (
                        candidate_info[index]["span"],
                        candidate_info[index]["localizer_distance"],
                        len(candidate_info[index]["frame_support"]),
                        index,
                    ),
                )
                remaining.remove(input_index)
            else:
                rank = {
                    index: position
                    for position, index in enumerate(normalized_order)
                }
                input_index = min(remaining, key=rank.__getitem__)
                remaining.remove(input_index)

            axis, qubit, forced = operations[input_index]
            info = (
                candidate_info[input_index]
                if normalized_order == "min_span"
                else self._measurement_span_info(
                    axis,
                    qubit,
                    absorb_basis=absorb_basis,
                )
            )
            if reset:
                m_pauli = self.state.frame_pauli(self._phys_pauli(axis, qubit))
                outcome = self._absorb_measure(
                    m_pauli,
                    None,
                    norm_event_kind="reset",
                )
            else:
                outcome = self.measure(
                    axis,
                    qubit,
                    outcome=forced,
                    absorb_basis=absorb_basis,
                )
            if (reset or reset_after) and outcome < 0:
                self.state.apply_clifford(_RESET_FLIP_CLIFFORDS[axis], qubit)
                self._record()
            result[input_index] = int(outcome)
            schedule.append({
                "order": int(step),
                "input_index": int(input_index),
                "pauli": str(axis),
                "qubit": int(qubit),
                **info,
            })

        self.last_measurement_schedule = tuple(schedule)
        return tuple(result)

    def measure_many(
        self,
        measurements,
        *,
        order="min_span",
        absorb_basis: Optional[bool] = None,
        disentangle: Optional[bool] = None,
    ):
        """Measure independent single-qubit observables in a cheap span order.

        ``order="min_span"`` is metadata-only: it reads current Tableau frame
        supports and the logical-to-MPS layout, but never runs trial MPS
        contractions or truncations.  Outcomes are returned in input order;
        :attr:`last_measurement_schedule` records execution order and costs.
        Use ``order="input"`` to preserve the supplied order, or pass an
        explicit permutation of batch indices/target qubits.
        """
        absorb_basis = _resolve_measurement_disentangle(
            absorb_basis,
            disentangle,
            default=False,
        )
        operations = self._normalize_measurement_batch(measurements)
        result = self._run_measurement_batch(
            operations,
            order=order,
            absorb_basis=absorb_basis,
        )
        return result[0] if len(result) == 1 else result

    def measure(
        self,
        pauli,
        where,
        *,
        outcome: Optional[int] = None,
        absorb_basis: Optional[bool] = None,
        disentangle: Optional[bool] = None,
    ):
        """Measure a Pauli observable, collapse ``|nu>``, and return ``+1``/``-1``.

        Parameters
        ----------
        pauli : str
            Pauli axes, e.g. ``"Z"`` (single qubit) or ``"XZ"`` (multi-qubit).
        where : int | sequence[int]
            Qubit(s) the observable acts on.
        outcome : int | None
            If given (``+1`` or ``-1``), force this outcome (post-selection);
            otherwise sample according to the Born rule.
        absorb_basis : bool
            If ``True``, use the **basis-updating** (canonical Lemma-3) form: a
            Clifford ``V`` localises the frame image ``M = C^dagger O C`` onto a
            single coefficient qubit ``k`` (``V M V^dagger = +/-Z_k``), ``V`` is
            applied to ``|nu>`` and ``V^dagger`` absorbed into the basis ``C``
            (``|psi>`` preserved), and qubit ``k`` is projected to a definite
            computational value.  The measured qubit is thereby disentangled from
            ``|nu>``, so its support/entanglement leaves the coefficient state —
            the key primitive for magic-state injection (see :meth:`inject_t`).
            The default (``False``) keeps the cheaper fixed-basis projector
            ``(I +- M)/2`` applied directly to ``|nu>``.
        disentangle : bool, optional
            User-facing alias for ``absorb_basis``. If both names are supplied,
            they must agree.

        Returns
        -------
        int
            The measured eigenvalue ``+1`` or ``-1``.
        """
        absorb_basis = _resolve_measurement_disentangle(
            absorb_basis,
            disentangle,
            default=False,
        )
        if absorb_basis:
            m_pauli = self.state.frame_pauli(self._phys_pauli(pauli, where))
            m = self._absorb_measure(
                m_pauli,
                outcome,
                norm_event_kind="measure_absorb",
            )
            self.measurements.append(MeasurementRecord(pauli, where, int(m)))
            return m
        terms, sign = self._frame_terms(pauli, where)
        forced = self._validate_outcome(outcome)
        if forced is None:
            p_plus = self._outcome_probability(
                self._pauli_expectation(terms, sign), +1
            )
            m = 1 if self._rng.random() < p_plus else -1
            branch_probability = p_plus if m > 0 else 1.0 - p_plus
        else:
            m = forced
            probability = self._outcome_probability(
                self._pauli_expectation(terms, sign), m
            )
            if probability <= 1e-12:
                raise ValueError(
                    f"forced outcome {m:+d} has ~0 probability ({probability:.2e})."
                )
            branch_probability = probability
        norm_event = (
            self._make_norm_event("measure", branch_probability=branch_probability)
            if terms
            else None
        )
        self._apply_projector(terms, sign, m, norm_event=norm_event)
        self.measurements.append(MeasurementRecord(pauli, where, int(m)))
        return m

    def reset(self, where, basis="Z", *, order="min_span") -> "MpsStabOptimizer":
        """Reset qubit(s) to the ``+1`` eigenstate of ``basis``.

        Each target is measured with the basis-updating path (so it
        disentangles from ``|nu>``); if the outcome is ``-1`` an anticommuting
        Clifford flips it to the ``+1`` eigenstate.  The legacy
        ``basis="Z"`` form returns qubits to ``|0>``.  Available in a gate
        stream as ``("reset", where)`` or ``("reset", where, basis)``.  The
        internal measurements are *not* appended to :attr:`measurements` (a
        reset is an operation, not a recorded readout).  By default, separate
        targets are processed with the metadata-only ``min_span`` scheduler;
        use ``order="input"`` to preserve their supplied order.
        """
        where = _normalize_sites(where)
        axes = _normalize_pauli_axes(basis, where, event="reset")
        if len(set(where)) != len(where):
            raise ValueError(
                "reset requires distinct target qubits so they can be safely "
                "reordered."
            )
        operations = tuple((axis, q, None) for axis, q in zip(axes, where))
        self._run_measurement_batch(
            operations,
            order=order,
            absorb_basis=True,
            reset=True,
        )
        return self

    def reset_many(self, where, basis="Z", *, order="min_span") -> "MpsStabOptimizer":
        """Reset several independent qubits using the metadata-only span scheduler."""
        return self.reset(where, basis=basis, order=order)

    def measure_reset(
        self,
        pauli,
        where,
        *,
        outcome=None,
        absorb_basis: Optional[bool] = None,
        disentangle: Optional[bool] = None,
        order="min_span",
    ):
        """Measure target qubit(s), record outcomes, then reset to ``+pauli``.

        ``pauli`` is one X/Y/Z axis per target, or one axis broadcast across all
        targets.  Unlike :meth:`reset`, the measurement outcomes are appended to
        :attr:`measurements`.  The default uses the fixed-basis projector; pass
        ``disentangle=True`` to use the basis-updating path so each reset target
        leaves the coefficient MPS compactly.  Separate targets are processed
        with the metadata-only ``min_span`` scheduler by default; use
        ``order="input"`` to preserve their supplied order.  Returned outcomes
        remain aligned with the input target order.
        """
        absorb_basis = _resolve_measurement_disentangle(
            absorb_basis,
            disentangle,
            default=False,
        )
        where = _normalize_sites(where)
        axes = _normalize_pauli_axes(pauli, where, event="measure_reset")
        outcomes = _normalize_outcomes(outcome, where, event="measure_reset")
        if len(set(where)) != len(where):
            raise ValueError(
                "measure_reset requires distinct target qubits so they can be "
                "safely reordered."
            )
        operations = tuple(
            (axis, q, forced)
            for axis, q, forced in zip(axes, where, outcomes)
        )
        measured = self._run_measurement_batch(
            operations,
            order=order,
            absorb_basis=absorb_basis,
            reset_after=True,
        )
        return measured[0] if len(measured) == 1 else measured

    def cap(self, where, vec, *, absorb="left") -> "MpsStabOptimizer":
        """Contract one physical qubit with ``vec`` and shorten the simulator.

        With an identity basis frame, the physical leg is contracted directly
        into neighboring MPS tensors, preserving backend/autodiff vectors. A
        non-identity frame is lowered to an ordinary backend-native MPS first,
        so the cap remains backend-native for both constant and trainable data.
        """
        absorb = _normalize_absorb(absorb)
        if not self._layout_is_identity():
            raise ValueError(
                "physical cap is not supported after installing an STN static "
                "layout, because cap changes the logical qubit set and MPS length."
            )
        sites = _normalize_sites(where)
        if len(sites) != 1:
            raise ValueError("cap expects exactly one qubit site.")
        q = int(sites[0])
        n = self.n
        if n <= 1:
            raise ValueError("cannot cap the only qubit of a one-qubit STN state.")
        if not 0 <= q < n:
            raise ValueError(f"cap site {q} is outside the qubit range [0, {n}).")
        limit = self.max_dense_cap_qubits
        if limit is not None and n > limit:
            raise ValueError(
                f"physical cap register has {n} qubits, exceeding "
                f"max_dense_cap_qubits={limit}. Use a structured capped stream "
                "or raise the limit explicitly."
            )

        shape = getattr(vec, "shape", None)
        if shape is None:
            vec_arr = np.asarray(vec, dtype=self.dtype).ravel()
        else:
            vec_arr = vec.reshape(-1)
        if tuple(int(size) for size in getattr(vec_arr, "shape", ())) != (2,):
            raise ValueError(
                "cap vector must have length 2 for a qubit, got shape "
                f"{getattr(vec_arr, 'shape', None)}."
            )

        if self.state.is_identity_frame():
            p = self.state.p
            phys_ind = p.site_ind(q)
            if p.ind_size(phys_ind) != 2:
                raise ValueError(f"cap site {q} does not have physical dimension 2.")
            if absorb == "left":
                neighbour = q - 1 if q > 0 else q + 1
            else:
                neighbour = q + 1 if q < n - 1 else q - 1

            self._canonize_p(neighbour)
            new_center = neighbour if neighbour < q else neighbour - 1
            cap_tensor = qtn.Tensor(
                self._to_state_backend(vec_arr),
                inds=(phys_ind,),
            )
            site_tensor = p[p.site_tag(q)]
            neighbour_tensor = p[p.site_tag(neighbour)]
            merged = qtn.tensor_contract(site_tensor, cap_tensor, neighbour_tensor)

            site_ind_id = p.site_ind_id
            site_tag_id = p.site_tag_id
            p.delete(p.site_tag(q))
            p.delete(p.site_tag(neighbour))
            merged.modify(tags=(p.site_tag(neighbour),))
            p |= merged

            temp_reindex = {
                site_ind_id.format(old): f"__pepsy_cap_k{old - 1}"
                for old in range(q + 1, n)
            }
            temp_retag = {
                site_tag_id.format(old): f"__pepsy_cap_I{old - 1}"
                for old in range(q + 1, n)
            }
            if temp_reindex:
                p.reindex_(temp_reindex)
                p.retag_(temp_retag)
            final_reindex = {
                f"__pepsy_cap_k{i}": site_ind_id.format(i)
                for i in range(q, n - 1)
            }
            final_retag = {
                f"__pepsy_cap_I{i}": site_tag_id.format(i)
                for i in range(q, n - 1)
            }
            if final_reindex:
                p.reindex_(final_reindex)
                p.retag_(final_retag)

            p = p.view_as_(
                qtn.MatrixProductState,
                L=n - 1,
                cyclic=False,
                site_ind_id=site_ind_id,
                site_tag_id=site_tag_id,
            )
            import stim

            tableau = stim.TableauSimulator()
            tableau.set_num_qubits(n - 1)
            self.state = STNState.from_tableau_and_state(
                tableau,
                p,
                dtype=self.dtype,
            )
            self.logical_order = list(range(n - 1))
            self._logical_to_mps = {q: q for q in self.logical_order}
            self.state.info["cur_orthog"] = (new_center, new_center)
            self.backend_info()
            self._localizer_cache.clear()
            self._invalidate_infidelity()
            self._mps_length_history.append(int(self.state.n))
            self.cap_history.append(
                {
                    "physical_site": int(q),
                    "old_length": int(n),
                    "new_length": int(self.state.n),
                    "absorb": str(absorb),
                }
            )
            self._record()
            return self

        # Lower a nontrivial physical frame to an ordinary backend-native MPS
        # before capping. This avoids any dense statevector reconstruction and
        # retains the exact physical semantics of the cap.
        if (
            self.to_backend is None
            and self._gate_requires_grad(vec_arr)
            and self.backend == "numpy"
        ):
            raise ValueError(
                "A trainable physical cap requires a configured native backend "
                "converter (for example to_backend=backend_torch(...))."
            )
        physical = self.to_mps(mode="exact", logical_order=True)
        lowered = MpsStabOptimizer.from_mps(
            physical,
            chi=self.chi,
            cutoff=self.cutoff,
            max_dense_cap_qubits=self.max_dense_cap_qubits,
            to_backend=self.to_backend,
        )
        lowered.cap(q, vec_arr, absorb=absorb)
        self.state = lowered.state
        self.logical_order = list(range(self.state.n))
        self._logical_to_mps = {q: q for q in self.logical_order}
        self.backend_info()
        self._localizer_cache.clear()
        self._invalidate_infidelity()
        self._mps_length_history.append(int(self.state.n))
        self.cap_history.append(
            {
                "physical_site": int(q),
                "old_length": int(n),
                "new_length": int(self.state.n),
                "absorb": str(absorb),
            }
        )
        self._record()
        return self

    def _localizing_clifford_cached(self, terms):
        """Return the cached measurement localizer for ``terms`` and layout."""
        key = (
            tuple(self.logical_order),
            tuple(sorted((int(site), str(axis)) for site, axis in terms.items())),
        )
        cached = self._localizer_cache.get(key)
        if cached is None:
            ops, v_tableau, k = _localizing_clifford(
                terms,
                self.n,
                site_position=self._mps_site,
            )
            cached = (tuple(ops), v_tableau, int(k))
            self._localizer_cache[key] = cached
        return cached

    def _absorb_measure(
        self,
        m_pauli,
        outcome,
        *,
        norm_event_kind: str = "measure_absorb",
    ) -> int:
        """Basis-updating measurement of the frame Pauli ``m_pauli``; returns ``+/-1``.

        ``m_pauli`` is the signed :class:`stim.PauliString` image
        ``M = C^dagger O C`` of the physical observable on the coefficient qubits.
        """
        self._require_nonzero_state("measure")
        terms, sign = hermitian_pauli_terms(m_pauli)
        forced = self._validate_outcome(outcome)
        support = sorted(terms)
        if not support:  # M = +/- I: deterministic, state unchanged
            if forced is not None and forced != sign:
                raise ValueError(
                    f"forced outcome {forced:+d} has zero probability for "
                    f"the deterministic observable (expected {int(sign):+d})."
                )
            self._record()
            return int(sign)
        # Compute the Born probability before applying the localizing Clifford.
        # The localizer is unitary in exact arithmetic, but its coefficient-MPS
        # implementation can truncate at finite chi.  Sampling after that
        # truncation would make the measurement distribution depend on the
        # approximation used to localize the Pauli.
        p_o_plus = self._outcome_probability(
            self._pauli_expectation(terms, sign), +1
        )
        if forced is not None:
            probability = p_o_plus if forced > 0 else 1.0 - p_o_plus
            if probability <= 1e-12:
                raise ValueError(
                    f"forced outcome {forced:+d} has ~0 probability "
                    f"({probability:.2e})."
                )
            m = forced
        else:
            m = 1 if self._rng.random() < p_o_plus else -1
        ops, v_tableau, k = self._localizing_clifford_cached(terms)
        conj_terms, s = hermitian_pauli_terms(v_tableau(m_pauli))  # V M V^dag
        if conj_terms != {k: "Z"}:  # pragma: no cover - localizer invariant
            raise RuntimeError(
                f"localizer produced {conj_terms!r}, expected Z on qubit {k}."
            )
        # Establish a tracked orthogonality centre before the localizer so the
        # whole measurement runs on a canonical coefficient MPS.
        self._ensure_p_center()
        self._apply_localizer_to_p(ops)
        self.state.absorb_basis_clifford(v_tableau)
        # The localizer should preserve the branch probability exactly, but a
        # finite-chi approximation can change the localized state slightly.
        # Keep the physical probability sampled above separate from the
        # post-localizer probability used to isolate the final one-site
        # projector's compression loss.
        zexp = float(np.real(self._to_scalar(
            self.state.p.local_expectation_canonical(
                self._bk_const("PZ", pauli_matrix("Z")), self._mps_site(k),
                normalized=True, info=self.state.info,
            )
        )))
        projector_p_plus = self._outcome_probability(zexp, s)
        branch_probability = p_o_plus if m > 0 else 1.0 - p_o_plus
        projector_branch_probability = (
            projector_p_plus if m > 0 else 1.0 - projector_p_plus
        )
        norm_event = self._make_norm_event(
            norm_event_kind,
            branch_probability=branch_probability,
            projector_branch_probability=projector_branch_probability,
        )
        zval = m * s  # required Z_k eigenvalue (+1 -> |0>, -1 -> |1>)
        self._project_computational_site(
            k,
            0 if zval > 0 else 1,
            norm_event=norm_event,
        )
        self._record()
        return m

    def _cnot_submpo(self, control, target):
        """Build a coefficient-frame sub-MPO for one localizer CNOT."""
        control = int(control)
        target = int(target)
        branches = (
            (0.5, {}),
            (0.5, {control: "Z"}),
            (0.5, {target: "X"}),
            (-0.5, {control: "Z", target: "X"}),
        )
        return pauli_sum_submpo(branches, self.n, dtype=self.dtype)

    def _apply_localizer_to_p(self, ops) -> None:
        """Apply and track the measurement's localizing Clifford on ``|nu>``."""
        for name, targ in ops:
            mps_targ = self._mps_sites(targ)
            if name == "h":
                # Unitary single-qubit Cliffords preserve the tracked centre.
                self.state.p.gate_(
                    self._bk_const("H", _H_MAT), mps_targ[0], contract=True
                )
            elif name == "sdg":
                self.state.p.gate_(
                    self._bk_const("SDG", _SDG_MAT), mps_targ[0], contract=True
                )
            elif name == "cnot":
                mpo, where = self._cnot_submpo(mps_targ[0], mps_targ[1])
                infidelity = self._evolve_p(
                    self._bk_mpo(mpo, warn=False),
                    where,
                    unitary=True,
                    norm_event_kind="measurement_localizer",
                )
                # This is an internal localizer operation. Keep it in the
                # compression ledger without adding an extra public-entry bond
                # sample; the enclosing measurement records the final bond.
                self._record(infidelity, record_bond=False)

    def _project_computational_site(
        self,
        k,
        keep_bit,
        *,
        norm_event: Optional[NormEventRecord] = None,
    ) -> None:
        """Project coefficient site ``k`` onto ``|keep_bit>`` and renormalize ``|nu>``."""
        mps_k = self._mps_site(k)
        proj = np.zeros((2, 2), dtype=self.dtype)
        proj[keep_bit, keep_bit] = 1.0
        # Move the centre to k so the projector acts at the orthogonality centre
        # (keeping the state canonical there) and renormalize that centre tensor.
        self._canonize_p(mps_k)
        self.state.p.gate_(self._bk(proj), mps_k, contract=True, info=self.state.info)
        self.state.info["cur_orthog"] = (int(mps_k), int(mps_k))
        projected_norm = self._renorm_p_at(mps_k)
        self._reset_infidelity()
        self._commit_norm_event(norm_event, projected_norm=projected_norm)

    def _apply_projector(
        self,
        terms,
        sign,
        m,
        *,
        norm_event: Optional[NormEventRecord] = None,
    ) -> None:
        """Apply ``(I + m M)/2`` to ``|nu>`` and renormalize (M = sign * prod terms)."""
        support = sorted(terms)
        if not support:  # M = +/- I: outcome is deterministic, state unchanged
            self._record()
            return
        coef = 0.5 * m * sign
        if len(support) == 1:
            q = support[0]
            mps_q = self._mps_site(q)
            proj = single_qubit_combo_matrix(0.5, coef, terms[q], self.dtype)
            self._canonize_p(mps_q)
            self.state.p.gate_(self._bk(proj), mps_q, contract=True, info=self.state.info)
            self.state.info["cur_orthog"] = (int(mps_q), int(mps_q))
            projected_norm = self._renorm_p_at(mps_q)
            self._reset_infidelity()
            self._commit_norm_event(norm_event, projected_norm=projected_norm)
            self._record()
            return
        mps_terms = self._mps_terms(terms)
        mpo, where = pauli_combo_submpo(0.5, coef, mps_terms, self.n, dtype=self.dtype)
        self._evolve_p(
            self._bk_mpo(mpo, warn=False),
            where,
            renormalize=True,
            norm_event=norm_event,
        )
        self._record()

    # ------------------------------------------------------------------ #
    # Magic-state injection (R1)
    # ------------------------------------------------------------------ #
    def prepare_magic(self, ancilla, *, angle: float = math.pi / 4) -> "MpsStabOptimizer":
        """Prepare the magic state ``|M> = Rz(angle)|+>`` on a fresh ``|0>`` ancilla.

        The default ``angle = pi/4`` gives the ``T`` resource
        ``|A> = (|0> + e^{i pi/4}|1>)/sqrt(2)`` consumed by :meth:`inject_t`; use
        the matching ``angle`` for :meth:`inject_rz`.  The ancilla **must
        currently be** ``|0>`` — freshly initialised, or just returned to ``|0>``
        by :meth:`reset` (so ancillas can be recycled).  Implemented physically as
        a Clifford ``H`` (tableau only) followed by the ``Rz(angle)`` rotation; on
        a decoupled ``|0>`` qubit this keeps ``|nu>`` compact.
        """
        (a,) = self._validate_magic_ancilla_pool(
            [ancilla],
            require_nonempty=True,
        )
        self._assert_magic_ancillas_clean((a,))
        self.state.apply_clifford("h", a)  # |0> -> |+>, Clifford (tableau only)
        self._record()
        self._apply_rotation("rz", (float(angle), a))  # |+> -> Rz(angle)|+> = |M>
        return self

    def inject_rz(self, data, ancilla, phi, *, outcome: Optional[int] = None) -> int:
        """Apply ``Rz(phi)`` to ``data`` by magic-state injection (gate teleportation).

        Generalises :meth:`inject_t`.  ``phi`` must be a multiple of ``pi/4`` so
        the outcome correction ``Rz(2*phi)`` is Clifford.  For an *arbitrary*
        angle there is no scaling benefit to injecting: the resource state
        ``Rz(phi)|+>`` would itself be prepared with a rotation on ``|nu>``, so
        just apply the gate directly (``("rz", phi, q)`` routes to the exact
        rotation path) or compile it to Clifford+T (e.g. gridsynth) and inject
        each ``T``.  The ``ancilla`` must already hold the matching magic state
        ``|M> = Rz(phi)|+>`` (call ``prepare_magic(ancilla, angle=phi)`` first).

        Steps: ``CNOT(control=data, target=ancilla)`` (Clifford, tableau only);
        basis-updating ``Z`` measurement of the ancilla (disentangles it from
        ``|nu>``); if the outcome is ``-1`` apply the Clifford ``Rz(2*phi)``
        correction on ``data``.  The net channel on ``data`` is ``Rz(phi)`` (up to
        a global phase), keeping the non-Clifford cost on the pre-loaded ancilla.

        Returns the ancilla measurement eigenvalue ``+1``/``-1``.
        """
        phi = float(phi)
        k = phi / (math.pi / 4)
        if abs(k - round(k)) > 1e-9:
            raise ValueError(
                "inject_rz requires phi a multiple of pi/4 (so the Rz(2*phi) "
                "correction is Clifford). For an arbitrary angle, apply it "
                "directly as ('rz', phi, q) (exact rotation path) or compile to "
                "Clifford+T (e.g. gridsynth) and inject each T."
            )
        data, ancilla = int(data), int(ancilla)
        if not 0 <= data < self.n:
            raise ValueError(
                f"injection data qubit {data} is outside qubit range [0, {self.n})."
            )
        if not 0 <= ancilla < self.n:
            raise ValueError(
                f"injection ancilla qubit {ancilla} is outside qubit range [0, {self.n})."
            )
        if data == ancilla:
            raise ValueError("injection data and ancilla qubits must be distinct.")
        # CNOT(control=data, target=ancilla): Clifford, tableau only.
        self.state.apply_clifford("cnot", data, ancilla)
        self._record()
        # Measure the ancilla in Z, absorbing it out of |nu>.
        bond_before = self.state.max_bond()
        projection_start = time.perf_counter()
        m = self.measure("Z", ancilla, absorb_basis=True, outcome=outcome)
        if m < 0:  # ancilla collapsed to |1>: outcome-conditioned Rz(2*phi) correction.
            self._apply_rotation("rz", (2.0 * phi, data))
        self._last_injection_projection_event = ImmediateProjectionRecord(
            data=data,
            ancilla=ancilla,
            angle=phi,
            outcome=int(m),
            elapsed_s=float(time.perf_counter() - projection_start),
            bond_before=int(bond_before),
            bond_after=int(self.state.max_bond()),
        )
        return m

    def inject_t(self, data, ancilla, *, outcome: Optional[int] = None) -> int:
        """Apply ``T`` to ``data`` by consuming a magic ancilla (``inject_rz`` at ``pi/4``).

        The ``ancilla`` must already hold ``|A> = T|+>`` (call :meth:`prepare_magic`
        first).  See :meth:`inject_rz` for the gadget; here the correction is the
        Clifford ``S``.  Returns the ancilla measurement eigenvalue ``+1``/``-1``.
        """
        return self.inject_rz(data, ancilla, math.pi / 4, outcome=outcome)

    def inject_tdg(self, data, ancilla, *, outcome: Optional[int] = None) -> int:
        """Apply ``T-dagger`` via injection (``inject_rz`` at ``-pi/4``; correction ``S-dag``).

        The ``ancilla`` must hold ``T-dag|+>`` (``prepare_magic(ancilla, angle=-pi/4)``).
        """
        return self.inject_rz(data, ancilla, -math.pi / 4, outcome=outcome)

    @classmethod
    def _injectable_rz(cls, entry):
        """Return ``(data_qubit, phi)`` if ``entry`` is an injectable ``Z``-rotation.

        Injectable = a diagonal ``T``/``T-dagger``/``Rz(phi)`` gate that is
        *non-Clifford* and has ``phi`` a multiple of ``pi/4`` (so the injection
        correction is Clifford).  Clifford-angle ``Rz`` (multiple of ``pi/2``) is
        left for the free tableau path, and non-``pi/4`` angles for the normal
        rotation path; both return ``None``.  The matrix form also accepts
        Pepsy's public gate constructors, e.g. ``(pepsy.t(), q)`` and
        ``(pepsy.tdg(), q)``.  A global phase is ignored, as it is physically
        irrelevant to the injection gadget.
        """
        if not isinstance(entry, (list, tuple)) or not entry:
            return None

        # Pepsy's gate API represents T and T-dagger as explicit matrices, and
        # ordinary MPS streams commonly use ``(gate, q)`` entries.  Classify a
        # diagonal unitary by its relative phase rather than by object identity
        # so qarrays, rank-4 gate tensors, and globally phased copies work too.
        if not isinstance(entry[0], str):
            if len(entry) != 2:
                return None
            try:
                where = _normalize_sites(entry[1])
                matrix = _as_gate_matrix(entry[0], 1)
            except (TypeError, ValueError, IndexError):
                return None
            if len(where) != 1 or matrix.shape != (2, 2) or not _is_unitary(matrix):
                return None
            scale = max(float(np.max(np.abs(matrix))), 1.0)
            off_diagonal = matrix - np.diag(np.diag(matrix))
            if not np.allclose(off_diagonal, 0.0, rtol=1e-8, atol=1e-9 * scale):
                return None
            diagonal = np.diag(matrix)
            if min(abs(diagonal[0]), abs(diagonal[1])) <= 1e-12:
                return None
            relative_phase = diagonal[1] / diagonal[0]
            k = np.angle(relative_phase) / (math.pi / 4)
            nearest_k = int(round(k))
            if abs(k - nearest_k) > 1e-8 or nearest_k % 2 == 0:
                return None
            return int(where[0]), nearest_k * (math.pi / 4)

        if len(entry) < 2:
            return None
        name = entry[0].strip().lower()
        if name == "t":
            return int(entry[1]), math.pi / 4
        if name == "tdg":
            return int(entry[1]), -math.pi / 4
        if name == "rz":
            phi, q = float(entry[1]), int(entry[2])
            k = phi / (math.pi / 4)
            if abs(k - round(k)) < 1e-9 and not cls._is_clifford_angle(phi):
                return q, phi
        return None

    @staticmethod
    def _magic_prepare_layout_entries(ancilla, phi):
        """Synthetic layout entries for preparing one magic ancilla."""
        return [("h", int(ancilla)), ("rz", float(phi), int(ancilla))]

    @staticmethod
    def _magic_measure_layout_entry(ancilla, outcome=+1):
        """Synthetic layout entry for a basis-updating magic projection."""
        return ("measure", "Z", int(ancilla), int(outcome), True)

    def _nearest_magic_ancilla(self, candidates, data):
        """Choose the candidate ancilla nearest to ``data`` in current MPS order."""
        return min(
            candidates,
            key=lambda ancilla: abs(self._mps_site(ancilla) - self._mps_site(data)),
        )

    def _immediate_injection_layout_entries(
        self,
        entries,
        specs,
        pool,
        *,
        recycle: bool,
        reset_ancillas: bool,
    ) -> list:
        """Return a layout-only stream for the immediate-injection replay."""
        layout_entries = []
        dirty = {a: False for a in pool}
        for entry, spec in zip(entries, specs):
            if spec is None:
                layout_entries.append(entry)
                continue
            data, phi = spec
            clean = [a for a in pool if not dirty[a]]
            if clean:
                ancilla = self._nearest_magic_ancilla(clean, data)
            elif recycle:
                ancilla = self._nearest_magic_ancilla(pool, data)
                layout_entries.append(("reset", ancilla))
            else:
                ancilla = self._nearest_magic_ancilla(pool, data)
            layout_entries.extend(self._magic_prepare_layout_entries(ancilla, phi))
            layout_entries.append(("cnot", int(data), int(ancilla)))
            layout_entries.append(self._magic_measure_layout_entry(ancilla))
            dirty[ancilla] = True
        if reset_ancillas:
            layout_entries.extend(("reset", ancilla) for ancilla, is_dirty in dirty.items() if is_dirty)
        return layout_entries

    def _deferred_injection_layout_entries(
        self,
        entries,
        specs,
        pool,
        outcomes,
        *,
        projection_order,
        reset_ancillas: bool,
    ) -> list:
        """Return a layout-only stream for the deferred-injection replay."""
        layout_entries = []
        pending = []
        injection_index = 0
        for entry, spec in zip(entries, specs):
            if spec is None:
                layout_entries.append(entry)
                continue
            data, phi = spec
            ancilla = pool[injection_index]
            outcome = outcomes[injection_index]
            layout_entries.extend(self._magic_prepare_layout_entries(ancilla, phi))
            layout_entries.append(("cnot", int(data), int(ancilla)))
            if outcome < 0:
                layout_entries.append(("rz", 2.0 * float(phi), int(data)))
            pending.append(DeferredInjectionRecord(
                index=injection_index,
                ancilla=int(ancilla),
                data=int(data),
                angle=float(phi),
                outcome=int(outcome),
            ))
            injection_index += 1

        sequence = self._deferred_projection_sequence(pending, projection_order)
        if sequence is None:
            sequence = list(pending)
        for event in sequence:
            layout_entries.append(
                self._magic_measure_layout_entry(
                    event["ancilla"],
                    outcome=event["outcome"],
                )
            )
            if reset_ancillas and event["outcome"] < 0:
                layout_entries.append(("x", int(event["ancilla"])))
        return layout_entries

    def run_with_injection(
        self,
        gates,
        *,
        ancillas,
        recycle: bool = True,
        reset_ancillas: bool = True,
        progbar: bool = False,
        layout=None,
        layout_kwargs=None,
        layout_report: bool = True,
    ) -> "MpsStabOptimizer":
        """Replay ``gates``, teleporting ``Z``-rotations through magic-state injection.

        Every injectable gate (``("t", q)`` / ``("tdg", q)`` / ``("rz", phi, q)``
        with ``phi`` a non-Clifford multiple of ``pi/4`` — see
        :meth:`_injectable_rz`) is applied by :meth:`inject_rz` using a qubit from
        the reserved ``ancillas`` pool instead of the ``|nu>``-growing rotation
        path; all other entries replay normally.  Because injection measures the
        ancilla out immediately, one ancilla can be **recycled** for the whole
        stream (``reset`` + re-``prepare_magic``), so a pool of size 1 suffices.

        Parameters
        ----------
        gates : stream
            Gate stream (same forms as :meth:`add_gates`).
        ancillas : sequence[int]
            Reserved magic-ancilla qubits, disjoint from the data qubits the
            stream acts on.  Must currently be ``|0>``.
        recycle : bool
            If ``True`` (default), reset+reuse a spent ancilla when the pool is
            exhausted; if ``False``, raise once every pool ancilla is dirty.
        reset_ancillas : bool
            If ``True`` (default), reset every used ancilla back to ``|0>`` at the
            end, so the final state is ``(data result) (x) |0...0>_ancilla``.
        progbar : bool
            Show a ``tqdm`` progress bar.
        layout : str | mapping | sequence | None
            Optional static frame layout installed before replay. ``"auto"``
            uses a synthetic stream containing the magic preparation,
            injection-CNOT, and basis-updating projection supports.

        Returns ``self``.
        """
        pool = self._validate_magic_ancilla_pool(
            ancillas,
            require_nonempty=True,
        )
        entries = self._as_entries(gates)
        specs = [self._injectable_rz(entry) for entry in entries]
        self._validate_magic_stream_protection(
            entries,
            specs,
            pool,
            mode="immediate",
        )
        self._assert_magic_ancillas_clean(pool)
        self._apply_layout_from_entries(
            self._immediate_injection_layout_entries(
                entries,
                specs,
                pool,
                recycle=recycle,
                reset_ancillas=reset_ancillas,
            ),
            layout,
            layout_kwargs=layout_kwargs,
            layout_report=layout_report,
        )
        dirty = {a: False for a in pool}
        self.immediate_projection_events = []
        self.last_immediate_injection_report = None

        pbar = None
        if progbar and entries:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(entries), desc="stab-inject", leave=True, ascii=True)
        for entry, spec in zip(entries, specs):
            if spec is None:
                self._apply_entry(entry)
            else:
                data, phi = spec
                # Prefer the nearest *clean* ancilla to the data qubit (shorter
                # localizer span -> fewer MPS swaps); recycle the nearest dirty
                # one only if no clean ancilla is left.
                clean = [a for a in pool if not dirty[a]]
                if clean:
                    a = self._nearest_magic_ancilla(clean, data)
                elif recycle:
                    a = self._nearest_magic_ancilla(pool, data)
                    self.reset(a)
                else:
                    raise RuntimeError(
                        "magic-ancilla pool exhausted (recycle=False); "
                        "reserve more ancillas or allow recycling."
                    )
                self.prepare_magic(a, angle=phi)
                self.inject_rz(data, a, phi)
                self.immediate_projection_events.append(
                    self._last_injection_projection_event
                )
                dirty[a] = True
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(chi=self.state.max_bond())
        if pbar is not None:
            pbar.close()

        if reset_ancillas:
            for a in pool:
                if dirty[a]:
                    self.reset(a)
        self.last_immediate_injection_report = ImmediateInjectionReport(
            n_injections=len(self.immediate_projection_events),
            projection_elapsed_s=float(sum(
                event["elapsed_s"] for event in self.immediate_projection_events
            )),
            projection_peak_bond=int(max(
                (
                    max(event["bond_before"], event["bond_after"])
                    for event in self.immediate_projection_events
                ),
                default=self.state.max_bond(),
            )),
        )
        return self

    @classmethod
    def with_injection(
        cls, n_data: int, gates, *, n_ancilla: int = 1, **kwargs
    ) -> "MpsStabOptimizer":
        """Build an ``(n_data + n_ancilla)``-qubit simulator and run ``gates`` with injection.

        Data qubits are ``0 .. n_data - 1``; the last ``n_ancilla`` qubits are the
        recyclable magic-ancilla pool.  All ``T``/``T-dagger``/``pi/4``-``Rz`` gates
        in ``gates`` are teleported through :meth:`inject_rz` (see
        :meth:`run_with_injection`), keeping the non-Clifford cost on the ancilla
        pool instead of the coefficient MPS.  Remaining keyword arguments are
        forwarded to the constructor (``chi``, ``cutoff``, ``operator_tol``,
        ``max_pauli_decomposition_qubits``, ``seed``, ...).
        """
        n_data = int(n_data)
        n_ancilla = int(n_ancilla)
        if n_ancilla < 1:
            raise ValueError("with_injection needs n_ancilla >= 1.")
        run_opts = {
            k: kwargs.pop(k)
            for k in (
                "recycle",
                "reset_ancillas",
                "progbar",
                "layout",
                "layout_kwargs",
                "layout_report",
            )
            if k in kwargs
        }
        sim = cls(n_data + n_ancilla, **kwargs)
        sim.run_with_injection(
            gates, ancillas=range(n_data, n_data + n_ancilla), **run_opts
        )
        return sim

    @staticmethod
    def _deferred_injection_outcomes(outcomes, count, rng) -> tuple[int, ...]:
        """Normalize predetermined magic-measurement outcomes."""
        if outcomes is None:
            return tuple(1 if rng.random() < 0.5 else -1 for _ in range(count))
        try:
            values = tuple(outcomes)
        except TypeError as exc:
            raise TypeError("outcomes must be a sequence of +1/-1 values or None.") from exc
        if len(values) != count:
            raise ValueError(
                f"outcomes must contain one value per injectable gate ({count}), "
                f"got {len(values)}."
            )
        return tuple(MpsStabOptimizer._validate_outcome(value) for value in values)

    def _deferred_projection_metrics(self, ancilla) -> tuple[int, int]:
        """Return current coefficient-frame support size and MPS span for ``Z_a``."""
        terms, _sign = self._frame_terms("Z", ancilla)
        positions = [self._mps_site(site) for site in terms]
        if not positions:
            return 0, 0
        return len(positions), int(max(positions) - min(positions))

    def _deferred_projection_sequence(self, pending, projection_order):
        """Return a static order, or the ``min_span`` sentinel, for pending ancillas."""
        pending = tuple(pending)
        ancillas = tuple(event["ancilla"] for event in pending)
        if isinstance(projection_order, str):
            key = projection_order.replace("-", "_").strip().lower()
            if key in ("input", "injection"):
                return list(pending)
            if key in ("middle_out", "middle"):
                ordered = sorted(
                    pending,
                    key=lambda event: (self._mps_site(event["ancilla"]), event["index"]),
                )
                if len(ordered) % 2:
                    centre = len(ordered) // 2
                    result = [ordered[centre]]
                    left, right = centre - 1, centre + 1
                else:
                    result = []
                    left, right = len(ordered) // 2 - 1, len(ordered) // 2
                while left >= 0 or right < len(ordered):
                    if left >= 0:
                        result.append(ordered[left])
                        left -= 1
                    if right < len(ordered):
                        result.append(ordered[right])
                        right += 1
                return result
            if key in ("min_span", "greedy"):
                return None
            raise ValueError(
                "projection_order must be 'middle_out', 'input', 'min_span', "
                "or an explicit permutation of the used ancillas."
            )
        try:
            requested = tuple(int(ancilla) for ancilla in projection_order)
        except TypeError as exc:
            raise TypeError(
                "projection_order must be a supported string or an ancilla sequence."
            ) from exc
        if len(requested) != len(ancillas) or set(requested) != set(ancillas):
            raise ValueError(
                "an explicit projection_order must be a permutation of the used "
                f"ancillas {ancillas!r}, got {requested!r}."
            )
        by_ancilla = {event["ancilla"]: event for event in pending}
        return [by_ancilla[ancilla] for ancilla in requested]

    def run_with_deferred_injection(
        self,
        gates,
        *,
        ancillas,
        outcomes=None,
        projection_order="middle_out",
        reset_ancillas: bool = True,
        progbar: bool = False,
        layout=None,
        layout_kwargs=None,
        layout_report: bool = True,
    ) -> "MpsStabOptimizer":
        """Replay a circuit using MAST-style deferred magic-state projections.

        Each injectable ``T``/``T-dagger``/non-Clifford ``pi/4``-multiple
        ``Rz`` receives a distinct fresh magic ancilla. The gadget CNOT and its
        preselected branch correction are applied during replay, but the ancilla
        ``Z`` projections are delayed until the circuit has completed. Thus the
        coefficient MPS holds only a product magic register while the circuit is
        replayed; the costly basis-updating projections are concentrated at the
        end, where their order can be chosen.

        The supplied circuit must not act on ``ancillas``. Deferred injection is
        restricted to angles whose feed-forward correction is Clifford, exactly
        as :meth:`inject_rz` is.

        Parameters
        ----------
        ancillas : sequence[int]
            Reserved fresh ancillas. Deferred injection does not recycle them:
            there must be at least one ancilla for every injectable entry.
        outcomes : sequence[int] | None
            Optional predetermined ``+1``/``-1`` magic-measurement outcomes in
            injection order. ``None`` samples their uniform outcomes from this
            simulator's RNG before replay.
        projection_order : {"middle_out", "input", "min_span"} | sequence[int]
            End-of-circuit magic-register projection order. ``middle_out`` is
            the MAST-style default for a contiguous register. ``min_span`` is a
            greedy tableau-only planner that repeatedly chooses the current
            ancilla with the shortest coefficient-MPS frame span.
        reset_ancillas : bool
            Reset measured ancillas to ``|0>`` after projection, leaving the
            direct data result tensored with clean ancillas.
        layout : str | mapping | sequence | None
            Optional static frame layout installed before replay. ``"auto"``
            uses a synthetic stream containing magic preparation, branch
            corrections, and final projection supports.
        """
        pool = self._validate_magic_ancilla_pool(
            ancillas,
            require_nonempty=False,
        )
        entries = self._as_entries(gates)
        specs = [self._injectable_rz(entry) for entry in entries]
        n_injections = sum(spec is not None for spec in specs)
        if len(pool) < n_injections:
            raise ValueError(
                "deferred injection needs one fresh ancilla per injectable gate: "
                f"need {n_injections}, got {len(pool)}."
            )
        self._validate_magic_stream_protection(
            entries,
            specs,
            pool,
            mode="deferred",
        )
        self._assert_magic_ancillas_clean(pool)
        selected_outcomes = self._deferred_injection_outcomes(
            outcomes, n_injections, self._rng
        )
        self._apply_layout_from_entries(
            self._deferred_injection_layout_entries(
                entries,
                specs,
                pool,
                selected_outcomes,
                projection_order=projection_order,
                reset_ancillas=reset_ancillas,
            ),
            layout,
            layout_kwargs=layout_kwargs,
            layout_report=layout_report,
        )
        self.deferred_projection_events = []
        self.last_deferred_injection_report = None
        history_start = len(self.bond_history)
        replay_start = time.perf_counter()
        pending = []
        injection_index = 0

        pbar = None
        if progbar and entries:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(total=len(entries), desc="stab-deferred", leave=True, ascii=True)
        try:
            for entry, spec in zip(entries, specs):
                if spec is None:
                    self._apply_entry(entry)
                else:
                    data, phi = spec
                    ancilla = pool[injection_index]
                    outcome = selected_outcomes[injection_index]
                    self.prepare_magic(ancilla, angle=phi)
                    self.state.apply_clifford("cnot", data, ancilla)
                    self._record()
                    if outcome < 0:
                        # The branch correction must occur at the original gate
                        # location so subsequent physical Clifford gates see it.
                        self._apply_rotation("rz", (2.0 * phi, data))
                    pending.append(DeferredInjectionRecord(
                        index=injection_index,
                        ancilla=ancilla,
                        data=int(data),
                        angle=float(phi),
                        outcome=int(outcome),
                    ))
                    injection_index += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(chi=self.state.max_bond())
        finally:
            if pbar is not None:
                pbar.close()

        replay_elapsed = time.perf_counter() - replay_start
        pre_projection_peak = max(self.bond_history[history_start:], default=self.state.max_bond())
        projection_history_start = len(self.bond_history)
        projection_start = time.perf_counter()
        sequence = self._deferred_projection_sequence(pending, projection_order)
        if sequence is None:
            remaining = list(pending)
            sequence = []
            while remaining:
                event = min(
                    remaining,
                    key=lambda candidate: (
                        self._deferred_projection_metrics(candidate["ancilla"])[1],
                        self._deferred_projection_metrics(candidate["ancilla"])[0],
                        candidate["index"],
                    ),
                )
                sequence.append(event)
                remaining.remove(event)

        for order, event in enumerate(sequence):
            ancilla = event["ancilla"]
            support_size, span = self._deferred_projection_metrics(ancilla)
            before_bond = self.state.max_bond()
            self.measure("Z", ancilla, outcome=event["outcome"], absorb_basis=True)
            after_bond = self.state.max_bond()
            if reset_ancillas and event["outcome"] < 0:
                self.state.apply_clifford("x", ancilla)
                self._record()
            self.deferred_projection_events.append(DeferredProjectionRecord(
                index=int(event["index"]),
                ancilla=int(event["ancilla"]),
                data=int(event["data"]),
                angle=float(event["angle"]),
                outcome=int(event["outcome"]),
                order=int(order),
                support_size=int(support_size),
                mps_span=int(span),
                bond_before=int(before_bond),
                bond_after=int(after_bond),
            ))

        projection_elapsed = time.perf_counter() - projection_start
        projection_peak = max(
            self.bond_history[projection_history_start:], default=self.state.max_bond()
        )
        self.last_deferred_injection_report = DeferredInjectionReport(
            n_injections=int(n_injections),
            projection_order=projection_order,
            replay_elapsed_s=float(replay_elapsed),
            projection_elapsed_s=float(projection_elapsed),
            pre_projection_peak_bond=int(pre_projection_peak),
            projection_peak_bond=int(projection_peak),
            peak_bond=int(max(
                self.bond_history[history_start:],
                default=self.state.max_bond(),
            )),
        )
        return self

    @classmethod
    def with_deferred_injection(
        cls, n_data: int, gates, *, n_ancilla: Optional[int] = None, **kwargs
    ) -> "MpsStabOptimizer":
        """Build a simulator and replay ``gates`` with deferred magic projections.

        When ``n_ancilla`` is omitted, allocate exactly one trailing ancilla for
        each injectable gate. Supplying more is allowed; unused ancillas remain
        in ``|0>``. Unlike :meth:`with_injection`, this constructor cannot reuse
        a one-qubit ancilla pool because all projections are intentionally
        delayed until the end of the circuit.
        """
        n_data = int(n_data)
        entries = cls._as_entries(gates)
        required = sum(cls._injectable_rz(entry) is not None for entry in entries)
        if n_ancilla is None:
            n_ancilla = required
        n_ancilla = int(n_ancilla)
        if n_ancilla < required:
            raise ValueError(
                "with_deferred_injection needs at least one ancilla per injectable "
                f"gate: need {required}, got {n_ancilla}."
            )
        if n_ancilla < 0:
            raise ValueError("n_ancilla must be nonnegative.")
        run_opts = {
            key: kwargs.pop(key)
            for key in (
                "outcomes",
                "projection_order",
                "reset_ancillas",
                "progbar",
                "layout",
                "layout_kwargs",
                "layout_report",
            )
            if key in kwargs
        }
        sim = cls(n_data + n_ancilla, **kwargs)
        sim.run_with_deferred_injection(
            entries, ancillas=range(n_data, n_data + n_ancilla), **run_opts
        )
        return sim

    # ------------------------------------------------------------------ #
    # Explicit gate matrices
    # ------------------------------------------------------------------ #
    def _dense_gate_target_norm(self, gate: np.ndarray, where) -> float:
        """Evaluate ``||G|psi>||`` from the local physical ``G^dagger G``.

        The Gram operator is Pauli-decomposed and each physical Pauli is
        evaluated in the coefficient frame. This is the STN analogue of the
        ordinary MPS canonical local-expectation path and avoids copying an
        uncompressed target MPS.
        """
        where = (
            (int(where),)
            if isinstance(where, Integral)
            else tuple(int(site) for site in where)
        )
        k = len(where)
        norm_squared = self._norm_squared()
        if not np.isfinite(norm_squared):
            raise ValueError(
                "Cannot evaluate a non-unitary gate target norm from an invalid "
                f"coefficient norm squared {norm_squared!r}."
            )
        if norm_squared <= 0.0:
            return 0.0
        gate = np.asarray(ar.to_numpy(gate))
        gram = gate.conj().T @ gate
        expectation = 0.0 + 0.0j
        for term_index, (labels, coefficient) in enumerate(
            pauli_decomposition(gram, k, tol=self.operator_tol), start=1
        ):
            if (
                self.max_pauli_terms is not None
                and term_index > self.max_pauli_terms
            ):
                raise ValueError(
                    f"G^dagger G retained more than max_pauli_terms="
                    f"{self.max_pauli_terms}; increase the explicit term budget."
                )
            physical = pauli_string(labels, where, self.n)
            frame_terms, sign = hermitian_pauli_terms(
                self.state.frame_pauli(physical)
            )
            expectation += complex(coefficient) * self._pauli_expectation(
                frame_terms, sign
            )
        target_squared = float(np.real(expectation)) * norm_squared
        if target_squared < 0.0:
            if target_squared > -1.0e-10:
                target_squared = 0.0
            else:
                raise ValueError(
                    "G^dagger G produced a negative target norm squared: "
                    f"{target_squared!r}."
                )
        return float(target_squared ** 0.5)

    def _nonunitary_compression_infidelity(self, target_norm: float) -> float:
        """Record local compression infidelity relative to ``G^dagger G``."""
        target_norm = float(target_norm)
        approx_norm = float(self.norm())
        if target_norm <= 1.0e-15:
            fidelity = 1.0 if approx_norm <= 1.0e-15 else 0.0
            infidelity = 1.0 - fidelity
        else:
            log_fidelity = log_fidelity_from_norms(approx_norm, target_norm)
            fidelity = fidelity_from_log(log_fidelity)
            infidelity = infidelity_from_log(log_fidelity)
        self._nonunitary_infidelities.append(infidelity)
        self._accumulate_norm_survival(fidelity)
        self._invalidate_infidelity()
        return infidelity

    def _apply_matrix(self, gate: np.ndarray, where) -> None:
        where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
        gate = self._as_native_gate_matrix(gate, len(where))
        if self._gate_requires_grad(gate):
            if len(where) != 1:
                raise ValueError(
                    "Autodiff dense gate matrices currently support one qubit; "
                    "decompose a trainable multi-qubit gate into native stream "
                    "events or provide a coefficient-frame sub-MPO."
                )
            self._apply_trainable_one_qubit_gate(gate, where[0])
            return

        gate = _as_gate_matrix(gate, len(where))
        dim = gate.shape[0]
        nq = int(round(math.log2(dim)))
        if 2 ** nq != dim or gate.shape != (dim, dim):
            raise ValueError(f"Gate matrix must be square 2^k x 2^k, got {gate.shape}.")
        if len(where) != nq:
            raise ValueError(f"Gate on {nq} qubit(s) but where={where!r}.")

        # NOTE: stim.Tableau.from_unitary_matrix does NOT verify unitarity, so a
        # non-unitary matrix that happens to be close to a Clifford (e.g. the
        # near-identity weighted "coin" (1-p)I + pX) would be silently accepted
        # as that Clifford and misapplied. Only attempt the tableau route when the
        # gate is actually unitary.
        tableau = _tableau_from_exact_unitary(gate)
        gate_is_unitary = _is_unitary(gate)

        if tableau is not None:  # Clifford -> tableau update
            self.state.do_tableau(tableau, where)
            self._record()
            return

        if nq == 1 and gate_is_unitary:  # non-Clifford 1q unitary -> ZYZ
            alpha, theta, beta = _zyz_angles(gate)
            q = where[0]
            self._apply_rotation("rz", (beta, q))
            self._apply_rotation("ry", (theta, q))
            self._apply_rotation("rz", (alpha, q))
            return

        # General k-qubit gate (any k, unitary or non-unitary): decompose into
        # Paulis and act on the coefficient MPS via the frame map.
        self._apply_dense_gate(gate, where, unitary=gate_is_unitary)

    def _apply_trainable_one_qubit_gate(self, gate, where) -> None:
        """Apply a trainable one-qubit matrix without leaving the backend.

        For ``G = c_I I + c_X X + c_Y Y + c_Z Z``, the tableau frame maps each
        Pauli to a signed Pauli string on ``|nu>``. The frame/sign metadata is
        discrete and remains host-side; the four coefficients stay as backend
        scalars, so the MPS update retains the caller's autodiff graph.
        """
        shape = tuple(int(size) for size in getattr(gate, "shape", ()))
        if shape != (2, 2):
            raise ValueError(
                "A trainable one-qubit dense gate must have shape (2, 2), "
                f"got {shape}."
            )

        g00, g01 = gate[0, 0], gate[0, 1]
        g10, g11 = gate[1, 0], gate[1, 1]
        coefficients = (
            (g00 + g11) / 2,
            (g01 + g10) / 2,
            (g10 - g01) / (2j),
            (g00 - g11) / 2,
        )
        if (
            self._backend_scalar_is_exact_zero(g01)
            and self._backend_scalar_is_exact_zero(g10)
            and not self._backend_scalar_has_nonzero_grad(g01)
            and not self._backend_scalar_has_nonzero_grad(g10)
        ):
            # Preserve the diagonal I/Z coefficients even when one is zero at
            # the current parameter value: its derivative may be nonzero.
            selected = (("I", coefficients[0]), ("Z", coefficients[3]))
        else:
            selected = tuple(zip("IXYZ", coefficients))
        preserve_zero_gradient = any(
            self._backend_scalar_has_nonzero_grad(coefficient)
            and self._backend_scalar_is_exact_zero(coefficient)
            for _axis, coefficient in selected
        )
        mapped = []
        local_site = None
        local_matrix = None
        local_candidate = True
        for axis, coefficient in selected:
            physical = pauli_string((axis,), (int(where),), self.n)
            frame_terms, sign = hermitian_pauli_terms(
                self.state.frame_pauli(physical)
            )
            weight = coefficient * sign
            mapped.append((weight, frame_terms))
            if not local_candidate:
                continue
            if not frame_terms:
                term_site = local_site
                term_matrix = self._bk_const("I2", _I2)
            elif len(frame_terms) == 1:
                term_site, term_axis = next(iter(frame_terms.items()))
                term_site = int(term_site)
                term_matrix = self._bk_const(
                    "P" + str(term_axis).upper(),
                    pauli_matrix(term_axis),
                )
            else:
                local_candidate = False
                continue
            if local_site is None:
                local_site = term_site
            elif term_site != local_site:
                local_candidate = False
                continue
            contribution = weight * term_matrix
            local_matrix = (
                contribution
                if local_matrix is None
                else local_matrix + contribution
            )

        if local_candidate:
            if local_matrix is None:
                # This is possible only for an exactly-zero matrix. Preserve
                # the backend state structure while retaining the operation's
                # scalar graph if one exists.
                self.state.p = 0.0 * self.state.p
            elif local_site is None:
                self.state.p = local_matrix[0, 0] * self.state.p
            else:
                self.state.p.gate_(
                    local_matrix,
                    self._mps_site(local_site),
                    contract=True,
                )
            self._record()
            return

        # Do not coalesce here: ``_coalesce_operator_sum`` intentionally casts
        # weights to Python complex for its constant-gate path. The branch
        # reducer below accepts backend scalar weights and therefore preserves
        # Torch/JAX autodiff.
        self._record(
            self._apply_operator_sum(
                mapped,
                unitary=False,
                preserve_autodiff_zero=preserve_zero_gradient,
            )
        )

    def _apply_dense_gate(
        self, gate: np.ndarray, where, *, unitary: bool = False
    ) -> None:
        """Apply an arbitrary k-qubit gate ``G`` (unitary or not) to ``|psi>``.

        ``G = sum_a c_a P_a`` (Pauli decomposition); on the coefficient MPS this
        is ``M = C^dagger G C = sum_a c_a (C^dagger P_a C)`` where each
        ``C^dagger P_a C`` is a signed Pauli string. Sparse sums are applied as
        one exact low-bond sub-MPO; denser sums use the balanced branch-sum MPS
        reducer. Because ``C M p = G C p = G|psi>`` this is exact up to
        truncation and needs no renormalization, so it also represents
        non-unitary ``G`` (the coefficient-state norm then tracks ``|G|psi>|``).
        """
        where = (int(where),) if isinstance(where, Integral) else tuple(int(w) for w in where)
        k = len(where)
        # Validate support before either the complexity guard or decomposition.
        pauli_string(("I",) * k, where, self.n)
        if k == 4:
            warnings.warn(
                "Applying a 4-qubit dense physical gate requires a 256-term "
                "Pauli decomposition; this is deliberately expensive and is "
                "preferably expressed as smaller supported gates. The default "
                "max_pauli_decomposition_qubits=3 will reject it unless the "
                "limit is raised explicitly.",
                UserWarning,
                stacklevel=3,
            )
        limit = self.max_pauli_decomposition_qubits
        if limit is not None and k > limit:
            raise ValueError(
                f"Pauli decomposition of a {k}-qubit dense gate would enumerate "
                f"{4**k} candidate terms, exceeding "
                f"max_pauli_decomposition_qubits={limit} (at most {4**limit} "
                "terms). Decompose the physical operator into supported gates "
                "or Pauli rotations; use a submpo event only for an operator "
                "already expressed in the coefficient frame; or raise the "
                "limit explicitly."
            )
        target_norm = (
            None
            if unitary or k < 2
            else self._dense_gate_target_norm(gate, where)
        )
        decomp = pauli_decomposition(gate, k, tol=self.operator_tol)
        branches = []  # (weight, {site: axis})
        for term_index, (labels, coeff) in enumerate(decomp, start=1):
            if (
                self.max_pauli_terms is not None
                and term_index > self.max_pauli_terms
            ):
                raise ValueError(
                    f"dense gate retained more than max_pauli_terms="
                    f"{self.max_pauli_terms}; increase the explicit term budget "
                    "or decompose the operator into smaller supported gates."
                )
            phys = pauli_string(labels, where, self.n)
            frame_terms, sign = hermitian_pauli_terms(self.state.frame_pauli(phys))
            branches.append((coeff * sign, frame_terms))
        branches = self._coalesce_operator_sum(branches)
        support = {site for _, sites in branches for site in sites}
        if (
            0 < len(branches) <= _MAX_PAULI_SUM_SUBMPO_TERMS
            and len(support) >= 2
        ):
            self._record(
                self._apply_pauli_sum_submpo(
                    branches, unitary=unitary, target_norm=target_norm
                )
            )
        else:
            self._record(
                self._apply_operator_sum(
                    branches, unitary=unitary, target_norm=target_norm
                )
            )

    def _coalesce_operator_sum(self, branches):
        """Combine equal Pauli-string branches and prune exact/tolerant zeros."""
        accum = {}
        for weight, sites in branches:
            key = tuple(
                sorted((int(site), str(axis).upper()) for site, axis in sites.items())
            )
            accum[key] = accum.get(key, 0.0j) + complex(weight)
        tol = 0.0 if self.operator_tol is None else self.operator_tol
        return tuple(
            (weight, dict(key))
            for key, weight in accum.items()
            if abs(weight) > tol
        )

    def _apply_pauli_sum_submpo(
        self, branches, *, unitary: bool, target_norm: Optional[float] = None
    ) -> Optional[float]:
        """Apply a sparse Pauli-product sum as one exact coefficient-frame MPO."""
        mapped = tuple(
            (weight, self._mps_terms(sites))
            for weight, sites in branches
        )
        mpo, where = pauli_sum_submpo(mapped, self.n, dtype=self.dtype)
        if unitary:
            return self._evolve_p(self._bk_mpo(mpo, warn=False), where, unitary=True)
        self._evolve_p(self._bk_mpo(mpo, warn=False), where)
        if target_norm is None:
            self._invalidate_infidelity()
            return None
        return self._nonunitary_compression_infidelity(target_norm)

    def _apply_operator_sum(
        self,
        branches,
        *,
        unitary: bool,
        target_norm: Optional[float] = None,
        preserve_autodiff_zero: bool = False,
    ) -> Optional[float]:
        """Apply ``M = sum_j w_j (prod_i P_i)`` to the coefficient MPS ``p``.

        Each branch scales a copy of ``p`` by ``w_j`` and applies its
        (bond-preserving) single-qubit Paulis; the branches are summed and
        compressed to ``chi``/``cutoff``. Unitary sums return the cumulative
        norm-loss proxy. Dense non-unitary sums return the retained norm ratio
        when a physical ``G^dagger G`` target norm was supplied.
        """
        p = self.state.p
        branches = tuple(branches)
        before_norm_sq = (
            self._norm_squared()
            if unitary and self._infidelity_valid
            else None
        )
        if not branches or self._norm_squared() <= 0.0:
            self._set_zero_coefficient_state()
            if unitary:
                observed_infidelity = self._unitary_infidelity()
                self._record_compression_norm_event(
                    before_norm_sq,
                    observed_infidelity,
                )
                infidelity = self._current_infidelity
                self._stabilize_unitary_norm(before_norm_sq, observed_infidelity)
                return infidelity
            if target_norm is not None:
                return self._nonunitary_compression_infidelity(target_norm)
            self._invalidate_infidelity()
            return None

        def combine(left, right, max_bond):
            result = left + right
            preserve_zero = (
                preserve_autodiff_zero
                and (max_bond is None or max_bond >= len(branches))
            )
            if not preserve_zero:
                result.compress(max_bond=max_bond, cutoff=self.cutoff)
            return result

        def build(max_bond):
            preserve_zero = (
                preserve_autodiff_zero
                and (max_bond is None or max_bond >= len(branches))
            )
            # Binary-carry accumulation produces a balanced addition tree while
            # retaining only one partial sum per level (O(log(branches)) live
            # partials instead of materializing every branch MPS at once).
            partials = []
            for w, sites in branches:
                branch = p.copy()
                for site, axis in self._mps_terms(sites).items():
                    branch.gate_(self._bk_const("P" + axis, pauli_matrix(axis)), site, contract=True)
                # Scale one tensor instead of using MPS scalar multiplication.
                # The latter routes through Quimb's exponent normalization and
                # can produce NaNs for an exact zero backend coefficient,
                # especially at a trainable probability endpoint.
                first = branch[branch.site_tag(0)]
                first.modify(data=first.data * w)

                level = 0
                while level < len(partials) and partials[level] is not None:
                    if preserve_zero:
                        branch = partials[level] + branch
                    else:
                        branch = combine(partials[level], branch, max_bond)
                    partials[level] = None
                    level += 1
                if level == len(partials):
                    partials.append(branch)
                else:
                    partials[level] = branch

            result = None
            for partial in reversed(partials):
                if partial is None:
                    continue
                result = (
                    partial
                    if result is None
                    else combine(result, partial, max_bond)
                )
            if not preserve_zero:
                result.compress(max_bond=max_bond, cutoff=self.cutoff)
            return result

        if self.mode in self._DMRG_MODES:
            logical_support = {site for _, sites in branches for site in sites}
            self._fit_coefficient_target(
                build(None),
                self._mps_sites(sorted(logical_support)),
            )
        else:
            max_bond = None if self.mode == "exact" else self.chi
            preserve_zero = (
                preserve_autodiff_zero
                and (max_bond is None or max_bond >= len(branches))
            )
            self.state.p = build(max_bond)
            # A padded direct sum has no canonical centre until a later
            # operation requests one. The regular compressed path is already
            # canonical with the centre at site 0.
            self.state.info["cur_orthog"] = (
                None if preserve_zero
                else (0, 0)
            )
        if unitary:
            observed_infidelity = self._unitary_infidelity()
            self._record_compression_norm_event(
                before_norm_sq,
                observed_infidelity,
            )
            infidelity = self._current_infidelity
            self._stabilize_unitary_norm(before_norm_sq, observed_infidelity)
            return infidelity
        if target_norm is not None:
            return self._nonunitary_compression_infidelity(target_norm)
        self._invalidate_infidelity()
        return None

    def _set_zero_coefficient_state(self) -> None:
        """Install a valid, compact zero MPS with the current site structure."""
        p = self.state.p.copy()
        first = p[p.site_tag(0)]
        first.modify(data=first.data * 0)
        p.exponent = 0.0
        p.compress(max_bond=1, cutoff=0.0)
        p.exponent = 0.0
        self.state.p = p
        self.state.info["cur_orthog"] = (0, 0)

    # ------------------------------------------------------------------ #
    # Sub-MPO events (coefficient-frame operator)
    # ------------------------------------------------------------------ #
    def _copy_submpo_for_layout(self, submpo, support):
        """Return a copied sub-MPO with logical site labels mapped to MPS sites."""
        support = _unique_ordered(int(site) for site in support)
        if not support or self._layout_is_identity():
            return submpo

        mpo = submpo.copy()
        token = f"_pepsy_stn_layout_{id(mpo)}"
        reindex_to_temp = {}
        reindex_to_final = {}
        retag_to_temp = {}
        retag_to_final = {}

        for count, logical_site in enumerate(support):
            mps_site = self._mps_site(logical_site)
            if logical_site == mps_site:
                continue

            for kind in ("upper_ind", "lower_ind"):
                ind_fn = getattr(mpo, kind, None)
                if ind_fn is None:
                    continue
                old_ind = ind_fn(logical_site)
                new_ind = ind_fn(mps_site)
                tmp_ind = f"{token}_{count}_{kind}"
                reindex_to_temp[old_ind] = tmp_ind
                reindex_to_final[tmp_ind] = new_ind

            site_tag = getattr(mpo, "site_tag", None)
            if site_tag is not None:
                old_tag = site_tag(logical_site)
                new_tag = site_tag(mps_site)
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

    def _apply_submpo(self, mpo, where, *, _validate_backend=True) -> None:
        """Apply a user MPO to the coefficient MPS ``p`` (coefficient frame).

        The MPO acts directly on ``p`` (any MPO, unitary or not); it is *not*
        conjugated through the basis Clifford.  For a *physical*-frame operator
        use a dense ``(matrix, where)`` entry, which is frame-mapped for you.
        """
        logical_where = _normalize_sites(where)
        mps_where = self._mps_sites(logical_where)
        mapped_mpo = self._copy_submpo_for_layout(mpo, logical_where)
        if _validate_backend:
            mapped_mpo = self._bk_mpo(mapped_mpo)
        self._evolve_p(mapped_mpo, mps_where)
        self._invalidate_infidelity()
        self._record()

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def _record(
        self,
        infidelity: Optional[float] = None,
        *,
        record_bond: bool = True,
    ) -> None:
        """Record a public update and optionally its bond-history sample."""
        if infidelity is not None:
            self._norm_segment_open = True
            self.infidelities.append(float(infidelity))
        if record_bond:
            self.bond_history.append(self.state.max_bond())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MpsStabOptimizer(n={self.n}, chi={self.chi}, mode={self.mode!r}, "
            f"fit_init_strategy={self.fit_init_strategy!r}, "
            f"operator_tol={self.operator_tol}, "
            f"max_pauli_decomposition_qubits="
            f"{self.max_pauli_decomposition_qubits}, "
            f"max_pauli_terms={self.max_pauli_terms}, "
            f"max_dense_cap_qubits={self.max_dense_cap_qubits}, "
            f"queued={len(self._queue)}, current_chi={self.state.max_bond()})"
        )


def _normalize_runner_mode(mode) -> tuple[str, str]:
    """Return ``(requested_mode, actual_or_sentinel_mode)`` for the stream runner."""
    requested = _normalize_event_name(mode)
    aliases = {
        "recommended": "recommended",
        "advice": "recommended",
        "auto": "recommended",
        "direct": "direct",
        "apply": "direct",
        "immediate": "immediate",
        "injection": "immediate",
        "with_injection": "immediate",
        "deferred": "deferred",
        "mast": "deferred",
        "with_deferred_injection": "deferred",
    }
    if requested not in aliases:
        raise ValueError(
            "mode must be 'direct', 'immediate', 'deferred', or "
            f"'recommended', got {mode!r}."
        )
    return requested, aliases[requested]


def _runner_data_qubits(analysis: StreamAnalysisRecord, n_qubits) -> int:
    """Choose the data-qubit count for a run from explicit or inferred input."""
    if n_qubits is not None:
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
            raise TypeError("n_qubits must be a nonnegative integer or None.")
        n_data = int(n_qubits)
        if n_data < 0:
            raise ValueError("n_qubits must be nonnegative.")
        return n_data
    if analysis.estimated_qubits is None:
        raise ValueError(
            "n_qubits is required when the stream has no inferable qubit support."
        )
    return int(analysis.estimated_qubits)


def _runner_constructor_settings(
    advice: StabilizerMpsSettingsAdvice,
    settings,
    *,
    seed,
    mode: str,
) -> dict:
    """Merge advised settings with caller overrides for simulator construction."""
    ctor = dict(advice.settings)
    if settings is not None:
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping or None.")
        ctor.update(dict(settings))
    if seed is not None:
        ctor["seed"] = seed
    return ctor


def _runner_bond_value(value) -> int:
    """Normalize Quimb's one-site ``None`` max-bond convention to bond 1."""
    return 1 if value is None else int(value)


def _runner_collect_result(
    sim: MpsStabOptimizer,
    *,
    mode: str,
    requested_mode: str,
    execution_method: str,
    settings_used: dict,
    run_options: dict,
    analysis: StreamAnalysisRecord,
    advice: StabilizerMpsSettingsAdvice,
    elapsed_s: float,
    replay_elapsed_s: float,
    projection_elapsed_s: float,
    injection_report,
) -> StabilizerMpsRunResult:
    """Build the public typed result record for one completed stream replay."""
    return StabilizerMpsRunResult(
        simulator=sim,
        mode=mode,
        requested_mode=requested_mode,
        execution_method=execution_method,
        settings=settings_used,
        run_options=dict(run_options),
        analysis=analysis,
        advice=advice,
        elapsed_s=float(elapsed_s),
        replay_elapsed_s=float(replay_elapsed_s),
        projection_elapsed_s=float(projection_elapsed_s),
        final_bond=_runner_bond_value(sim.state.max_bond()),
        peak_bond=max(
            (_runner_bond_value(value) for value in sim.bond_history),
            default=_runner_bond_value(sim.state.max_bond()),
        ),
        norm=float(sim.norm()),
        norm_diagnostics=sim.norm_diagnostics(),
        measurements=tuple(sim.measurements),
        norm_events=tuple(sim.norm_events),
        immediate_projection_events=tuple(sim.immediate_projection_events),
        deferred_projection_events=tuple(sim.deferred_projection_events),
        injection_report=injection_report,
        remaining_queue=int(len(sim._queue)),
    )


def run_stabilizer_mps_stream(
    gates,
    *,
    n_qubits: Optional[int] = None,
    mode: str = "direct",
    settings=None,
    advice: Optional[StabilizerMpsSettingsAdvice] = None,
    ancilla_budget: Optional[int] = None,
    prioritize_peak_bond: bool = False,
    goal: str = "validate",
    n_ancilla: Optional[int] = None,
    run_options=None,
    seed: Optional[int] = None,
) -> StabilizerMpsRunResult:
    """Run one Pepsy STN stream explicitly and return a typed result record.

    ``mode`` defaults to ``"direct"``. Use ``mode="recommended"`` only when the
    caller explicitly wants to execute the mode selected by
    :meth:`MpsStabOptimizer.recommend_settings`.
    """
    entries = MpsStabOptimizer._as_entries(gates)
    if advice is None:
        advice = MpsStabOptimizer.recommend_settings(
            entries,
            n_qubits=n_qubits,
            ancilla_budget=ancilla_budget,
            prioritize_peak_bond=prioritize_peak_bond,
            goal=goal,
        )
    elif not isinstance(advice, StabilizerMpsSettingsAdvice):
        raise TypeError("advice must be a StabilizerMpsSettingsAdvice or None.")

    requested_mode, normalized_mode = _normalize_runner_mode(mode)
    actual_mode = advice.recommended_mode if normalized_mode == "recommended" else normalized_mode
    execution_method = {
        "direct": "apply",
        "immediate": "with_injection",
        "deferred": "with_deferred_injection",
    }[actual_mode]
    n_data = _runner_data_qubits(advice.analysis, n_qubits)
    ctor = _runner_constructor_settings(
        advice,
        settings,
        seed=seed,
        mode=actual_mode,
    )
    run_opts = {} if run_options is None else dict(run_options)

    if actual_mode == "direct":
        settings_used = {"n_qubits": n_data, **ctor}
        start = time.perf_counter()
        sim = MpsStabOptimizer(n_data, entries, **ctor)
        sim.run(**run_opts)
        elapsed = time.perf_counter() - start
        return _runner_collect_result(
            sim,
            mode=actual_mode,
            requested_mode=requested_mode,
            execution_method=execution_method,
            settings_used=settings_used,
            run_options=run_opts,
            analysis=advice.analysis,
            advice=advice,
            elapsed_s=elapsed,
            replay_elapsed_s=elapsed,
            projection_elapsed_s=0.0,
            injection_report=None,
        )

    if actual_mode == "immediate":
        if n_ancilla is None:
            n_ancilla = max(1, int(advice.immediate_ancillas_required))
        n_ancilla = int(n_ancilla)
        settings_used = {"n_data": n_data, "n_ancilla": n_ancilla, **ctor}
        kwargs = {**ctor, **run_opts}
        start = time.perf_counter()
        sim = MpsStabOptimizer.with_injection(
            n_data,
            entries,
            n_ancilla=n_ancilla,
            **kwargs,
        )
        elapsed = time.perf_counter() - start
        report = sim.last_immediate_injection_report
        projection = 0.0 if report is None else float(report.projection_elapsed_s)
        return _runner_collect_result(
            sim,
            mode=actual_mode,
            requested_mode=requested_mode,
            execution_method=execution_method,
            settings_used=settings_used,
            run_options=run_opts,
            analysis=advice.analysis,
            advice=advice,
            elapsed_s=elapsed,
            replay_elapsed_s=max(0.0, elapsed - projection),
            projection_elapsed_s=projection,
            injection_report=report,
        )

    if actual_mode == "deferred":
        if n_ancilla is None:
            n_ancilla = int(advice.deferred_ancillas_required)
        n_ancilla = int(n_ancilla)
        settings_used = {"n_data": n_data, "n_ancilla": n_ancilla, **ctor}
        kwargs = {**ctor, **run_opts}
        start = time.perf_counter()
        sim = MpsStabOptimizer.with_deferred_injection(
            n_data,
            entries,
            n_ancilla=n_ancilla,
            **kwargs,
        )
        elapsed = time.perf_counter() - start
        report = sim.last_deferred_injection_report
        replay = elapsed if report is None else float(report.replay_elapsed_s)
        projection = 0.0 if report is None else float(report.projection_elapsed_s)
        return _runner_collect_result(
            sim,
            mode=actual_mode,
            requested_mode=requested_mode,
            execution_method=execution_method,
            settings_used=settings_used,
            run_options=run_opts,
            analysis=advice.analysis,
            advice=advice,
            elapsed_s=elapsed,
            replay_elapsed_s=replay,
            projection_elapsed_s=projection,
            injection_report=report,
        )

    raise AssertionError(f"unreachable mode {actual_mode!r}")  # pragma: no cover


StabilizerMpsSimulator = MpsStabOptimizer


def _looks_like_stream(gates) -> bool:
    """Heuristic: is ``gates`` a stream of entries (vs a single entry)?"""
    first = gates[0]
    if is_submpo_event(first) or isinstance(first, Mapping):
        return True
    if isinstance(first, (list, tuple)):
        return True
    if hasattr(first, "shape") and hasattr(first, "ndim"):
        # First element is an array -> ``gates`` is a single (matrix, where) entry.
        return False
    # First element is a str/number -> ``gates`` is a single named entry.
    return False


def _zyz_angles(gate: np.ndarray):
    """Return ``(alpha, theta, beta)`` with ``U ~ Rz(alpha) Ry(theta) Rz(beta)``.

    Up to a global phase, using the convention ``Rz(a) = exp(-i a/2 Z)`` and
    ``Ry(t) = exp(-i t/2 Y)``.
    """
    u = np.asarray(ar.to_numpy(gate), dtype=complex)
    det = u[0, 0] * u[1, 1] - u[0, 1] * u[1, 0]
    u = u / np.sqrt(det)  # to SU(2) up to a sign (global phase, irrelevant)
    c = abs(u[0, 0])
    s = abs(u[1, 0])
    theta = 2.0 * math.atan2(s, c)
    apb = -np.angle(u[0, 0]) if c > 1e-12 else 0.0
    amb = -np.angle(-u[0, 1]) if s > 1e-12 else 0.0
    alpha = float(apb + amb)
    beta = float(apb - amb)
    return alpha, float(theta), beta
