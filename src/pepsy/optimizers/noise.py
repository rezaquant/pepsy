"""Stochastic gate-stream entries and trajectory replay.

Pepsy's native design is stream-local: users can place stochastic instructions
such as ``("depolarize1", p, q)`` or ``("amplitude_damping", gamma, q)`` exactly
where the hardware schedule says the channel acts. The trajectory runners
sample a *concrete* branch for each shot and replay the resulting ordinary gate
stream with either :class:`MpsOptimizer` or :class:`MpsStabOptimizer`. The older
``PauliErrorModel`` helpers remain convenience macros for inserting uniform
post-gate Pauli faults into a clean deterministic stream.
"""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .mps.optimizer import MpsOptimizer, _resolve_conditional
from .tree.optimizer import TreeOptimizer

__all__ = [
    "CoalescedMeasurementRecord",
    "CoalescedSampleResult",
    "CoalescedTrajectoryLeaf",
    "CoalescedTrajectoryResult",
    "ImportanceSamplingPolicy",
    "NoisyResult",
    "NoisyShotResult",
    "PauliErrorModel",
    "PauliFault",
    "StimCircuitPlan",
    "StimHerald",
    "StimNoiseSample",
    "StimDetector",
    "StimObservable",
    "StimSyndromeRecord",
    "StimObservableRecord",
    "StimShotResult",
    "TrajectoryDiagnostics",
    "TrajectoryMeasurementRecord",
    "CoherentCrosstalkModel",
    "LeakageRecord",
    "TrajectoryChannel",
    "TrajectoryEvent",
    "TrajectoryStreamPlan",
    "TrajectoryOutcome",
    "TrajectoryRecord",
    "TrajectorySample",
    "TrajectoryShotResult",
    "compile_stim_circuit",
    "compile_trajectory_stream",
    "run_coalesced_noisy_shots",
    "run_coalesced_stim_shots",
    "run_coalesced_trajectory_shots",
    "TreeNoisy",
    "sample_coalesced_bits",
    "run_noisy_shots",
    "run_parallel_noisy_shots",
    "run_parallel_stim_shots",
    "run_parallel_trajectory_shots",
    "run_stim_shots",
    "run_trajectory_shots",
    "sample_noisy_gate_stream",
    "sample_noisy_gate_streams",
    "sample_stim_circuit",
    "sample_stim_circuits",
    "sample_trajectory_stream",
]


_PAULI_MATRICES = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_ONE_QUBIT_NAMES = frozenset({"h", "s", "sdg", "x", "y", "z", "t", "tdg"})
_TWO_QUBIT_NAMES = frozenset({"cnot", "cx", "cy", "cz", "swap"})
_ONE_QUBIT_ROTATIONS = frozenset({"rx", "ry", "rz"})
_TWO_QUBIT_ROTATIONS = frozenset({"rxx", "ryy", "rzz"})
_STOCHASTIC_SINGLE_PAULI_NAMES = {
    "x_error": "X",
    "y_error": "Y",
    "z_error": "Z",
}
_STOCHASTIC_EVENT_NAMES = frozenset(
    {
        *(_STOCHASTIC_SINGLE_PAULI_NAMES),
        "depolarize1",
        "depolarize_1",
        "depolarize2",
        "depolarize_2",
        "pauli_channel1",
        "pauli_channel_1",
        "pauli_channel2",
        "pauli_channel_2",
        "amplitude_damping",
    }
)
_LEAKAGE_EVENT_NAMES = frozenset(
    {
        "leak",
        "leakage",
        "leakage_depolarize",
        "leakage_return",
        "leakage_seepage",
        "measure_leakage",
        "measure_leaked",
        "seepage",
        "unleak",
        "leak2depolar",
        "leak_to_depolar",
    }
)
_CONTROL_NAMES = frozenset(
    {
        "measure",
        "reset",
        "measure_reset",
        "mrx",
        "mry",
        "mrz",
        "reset_x",
        "reset_y",
        "reset_z",
        "cap",
        # Classical feed-forward wrappers are controls too. Their selected
        # ordinary gate action is handled after the record is resolved.
        "if",
        "conditional",
        "condition",
        "feed_forward",
        "feedforward",
        "disentangle",
        "submpo",
    }
)
_STIM_NOISE_NAMES = frozenset(
    {
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "II_ERROR",
        "I_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    }
)
_STIM_IGNORED_NAMES = frozenset(
    {
        "DETECTOR",
        "MPAD",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
    }
)
_STIM_SINGLE_MEASUREMENTS = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
}
_STIM_SINGLE_MEASURE_RESETS = {
    "MR": "Z",
    "MRX": "X",
    "MRY": "Y",
}
_STIM_PAIR_MEASUREMENTS = {
    "MXX": "XX",
    "MYY": "YY",
    "MZZ": "ZZ",
}
_STIM_RESETS = {"R": "Z", "RX": "X", "RY": "Y"}
_STIM_PAULI_2_OUTCOMES = tuple(
    (left, right)
    for left in "IXYZ"
    for right in "IXYZ"
    if (left, right) != ("I", "I")
)
_STIM_UNITARY_CACHE: dict[str, np.ndarray] = {}
_AUTO_MAX_EXPECTED_FAULTS = 0.1
_AUTO_MAX_BRANCHES = 128


@dataclass(frozen=True)
class _TrajectorySeedPair:
    """Pre-split channel/optimizer seeds used by serial and parallel runners."""

    channel: Any
    optimizer: Any


def _trajectory_seed_pairs(seed, shots, *, shot_ids=None):
    if isinstance(seed, _TrajectorySeedPair):
        if int(shots) != 1:
            raise ValueError("a pre-split trajectory seed can only run one shot.")
        return (seed,)
    if shot_ids is not None:
        if isinstance(shot_ids, range):
            if shot_ids.step <= 0 or shot_ids.start < 0:
                raise ValueError("shot_ids must be a nonnegative increasing range.")
            if len(shot_ids) != int(shots):
                raise ValueError("shot_ids must contain one entry per shot.")

            def iter_range_seed_pairs():
                for shot_id in shot_ids:
                    if shot_id >= 2**64:
                        raise ValueError("shot_ids must be smaller than 2**64.")
                    spawn_key = (shot_id & 0xFFFFFFFF, shot_id >> 32)
                    child_seed = np.random.SeedSequence(seed, spawn_key=spawn_key)
                    yield _TrajectorySeedPair(*child_seed.spawn(2))

            return iter_range_seed_pairs()
        shot_ids = tuple(int(shot_id) for shot_id in shot_ids)
        if len(shot_ids) != int(shots):
            raise ValueError("shot_ids must contain one entry per shot.")
        if any(shot_id < 0 for shot_id in shot_ids):
            raise ValueError("shot_ids must be nonnegative.")
        if len(set(shot_ids)) != len(shot_ids):
            raise ValueError("shot_ids must be unique.")
        pairs = []
        for shot_id in shot_ids:
            # A shot-specific spawn key makes the trajectory independent of
            # the number of MPI ranks that happen to execute it. Splitting the
            # integer keeps the key within SeedSequence's uint32 contract.
            if shot_id >= 2**64:
                raise ValueError("shot_ids must be smaller than 2**64.")
            spawn_key = (shot_id & 0xFFFFFFFF, shot_id >> 32)
            child_seed = np.random.SeedSequence(seed, spawn_key=spawn_key)
            pairs.append(_TrajectorySeedPair(*child_seed.spawn(2)))
        return tuple(pairs)
    return tuple(
        _TrajectorySeedPair(*child_seed.spawn(2))
        for child_seed in np.random.SeedSequence(seed).spawn(int(shots))
    )


@dataclass(frozen=True)
class PauliFault:
    """One sampled physical Pauli fault.

    ``gate_index`` identifies the entry in the ideal stream after which the
    fault was inserted. For a compiled Stim circuit it is the index in the
    flattened Stim instruction stream. It gives trajectory users an
    inspectable error record even though the replay stream stores the
    corresponding dense gate matrix.
    """

    gate_index: int
    site: int
    pauli: str


@dataclass(frozen=True)
class ImportanceSamplingPolicy:
    """A proposal distribution for unbiased rare-event trajectory sampling.

    ``proposal`` may be either a mapping from event index to outcome
    probabilities, a label-to-probability mapping used for every event, or a
    callable with signature ``(event_index, labels, target_probabilities,
    optimizer)``.  The target probabilities always come from the physical
    channel and the runner records the likelihood ratio ``target/proposal``.

    Proposal distributions must have support wherever the target distribution
    is nonzero.  ``max_likelihood_ratio`` is an optional safety guard against a
    numerically explosive proposal; it raises before applying a sampled branch.
    """

    proposal: Any
    max_likelihood_ratio: float | None = None

    def __post_init__(self):
        if self.max_likelihood_ratio is not None:
            value = float(self.max_likelihood_ratio)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "max_likelihood_ratio must be finite, positive, or None."
                )
            object.__setattr__(self, "max_likelihood_ratio", value)

    def probabilities(
        self,
        event_index,
        labels,
        target_probabilities,
        optimizer=None,
    ) -> np.ndarray:
        """Resolve and validate the proposal probabilities for one event."""
        labels = tuple(str(label) for label in labels)
        target = np.asarray(target_probabilities, dtype=float)
        if target.ndim != 1 or len(target) != len(labels):
            raise ValueError("importance target probabilities do not match outcomes.")
        if (
            not np.all(np.isfinite(target))
            or np.any(target < -1e-12)
            or not np.isclose(float(target.sum()), 1.0, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("importance target probabilities must form a distribution.")
        target = np.maximum(target, 0.0)

        spec = self.proposal
        if callable(spec):
            spec = spec(event_index, labels, target.copy(), optimizer)
        elif isinstance(spec, Mapping):
            label_keys = set(labels)
            if not set(spec).intersection(label_keys):
                event_spec = spec.get(event_index, spec.get("*"))
                if event_spec is None:
                    return target
                spec = event_spec

        if isinstance(spec, Mapping):
            proposal = np.asarray([spec.get(label, 0.0) for label in labels], dtype=float)
        else:
            proposal = np.asarray(spec, dtype=float)
        if proposal.ndim != 1 or len(proposal) != len(labels):
            raise ValueError(
                f"importance proposal for event {event_index} must match outcomes."
            )
        if (
            not np.all(np.isfinite(proposal))
            or np.any(proposal < -1e-12)
            or not np.isclose(float(proposal.sum()), 1.0, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError(
                f"importance proposal for event {event_index} must sum to one."
            )
        proposal = np.maximum(proposal, 0.0)
        if np.any((target > 1e-14) & (proposal <= 1e-14)):
            raise ValueError(
                f"importance proposal for event {event_index} omits a target branch."
            )
        return proposal


def _coerce_importance_policy(policy):
    if policy is None:
        return None
    if isinstance(policy, ImportanceSamplingPolicy):
        return policy
    if callable(policy) or isinstance(policy, Mapping):
        return ImportanceSamplingPolicy(policy)
    raise TypeError(
        "importance_sampling must be an ImportanceSamplingPolicy, mapping, "
        "callable, or None."
    )


@dataclass(frozen=True)
class PauliErrorModel:
    """Independent one-qubit Pauli channel applied after every gate target.

    Parameters are the probabilities of applying the corresponding error. The
    remaining probability is the identity branch, so ``p_x + p_y + p_z`` must
    be at most one. This is the stochastic Pauli-noise subset supported by
    Stim's ``X_ERROR``, ``Y_ERROR``, ``Z_ERROR``, ``DEPOLARIZE1``, and
    ``PAULI_CHANNEL_1`` instructions.

    Use :meth:`depolarizing`, :meth:`bit_flip`, :meth:`phase_flip`, or
    :meth:`bit_phase_flip` for common channels.
    """

    p_x: float = 0.0
    p_y: float = 0.0
    p_z: float = 0.0

    def __post_init__(self):
        values = (self.p_x, self.p_y, self.p_z)
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Pauli error probabilities must be finite and nonnegative.")
        if sum(values) > 1.0 + 1e-12:
            raise ValueError("p_x + p_y + p_z must not exceed one.")

    @property
    def probabilities(self) -> dict[str, float]:
        """Return the full ``I/X/Y/Z`` probability distribution."""
        return {
            "I": max(0.0, 1.0 - self.p_x - self.p_y - self.p_z),
            "X": self.p_x,
            "Y": self.p_y,
            "Z": self.p_z,
        }

    @classmethod
    def depolarizing(cls, probability: float) -> "PauliErrorModel":
        """Return ``I`` with probability ``1-p`` and each Pauli with ``p/3``."""
        probability = _unit_interval_probability(probability, "depolarizing")
        return cls(*(probability / 3.0,) * 3)

    @classmethod
    def bit_flip(cls, probability: float) -> "PauliErrorModel":
        """Return an ``X``-flip channel with probability ``probability``."""
        return cls(p_x=_unit_interval_probability(probability, "bit-flip"))

    @classmethod
    def phase_flip(cls, probability: float) -> "PauliErrorModel":
        """Return a ``Z``-flip channel with probability ``probability``."""
        return cls(p_z=_unit_interval_probability(probability, "phase-flip"))

    @classmethod
    def bit_phase_flip(cls, probability: float) -> "PauliErrorModel":
        """Return a ``Y``-flip channel with probability ``probability``."""
        return cls(p_y=_unit_interval_probability(probability, "bit-phase-flip"))

    def sample(self, rng: np.random.Generator) -> str:
        """Draw one of ``"I"``, ``"X"``, ``"Y"``, or ``"Z"``."""
        probabilities = self.probabilities
        return str(rng.choice(tuple(probabilities), p=tuple(probabilities.values())))

    def sample_gate_stream(self, gates, *, seed: Optional[int] = None):
        """Sample one noisy replay stream from ``gates``.

        Returned entries are ordinary ``(matrix, site)`` Pauli gates and can be
        passed to either optimizer. Ordinary named or matrix actions inside a
        conditional wrapper receive a matching predicate-wrapped fault;
        measurement/reset controls, nested controls, and coefficient-frame
        ``submpo`` events are preserved without an implicit fault.
        """
        stream, _ = _sample_gate_stream(gates, self, np.random.default_rng(seed))
        return stream

    def sample_gate_streams(self, gates, shots: int, *, seed: Optional[int] = None):
        """Sample ``shots`` independent concrete noisy replay streams."""
        return sample_noisy_gate_streams(gates, self, shots, seed=seed)


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Accuracy and execution summary for a noisy gate-stream replay.

    ``max_kraus_probability_residual`` is measured before the branch
    probabilities are normalized for sampling. A small nonzero value is
    expected from finite-MPS truncation; a large value indicates that the
    channel, local contraction, or compression path needs attention.
    ``used_kraus_copy_fallback`` reports whether any local Kraus probability
    could not use the environment contraction fast path.
    """

    shots: int
    branches: int
    coalesced: bool
    stream_events: int = 0
    trajectory_events: int = 0
    measurement_events: int = 0
    leakage_events: int = 0
    max_live_branches: int = 0
    max_kraus_probability_residual: float = 0.0
    used_kraus_copy_fallback: bool = False


@dataclass(frozen=True)
class NoisyShotResult:
    """Result of replaying independent stochastic Pauli-noise trajectories."""

    optimizers: tuple[Any, ...]
    gate_streams: tuple[tuple[object, ...], ...]
    faults: tuple[tuple[PauliFault, ...], ...]
    weights: tuple[float, ...] = ()
    shot_count: int | None = None
    diagnostics: "TrajectoryDiagnostics | None" = None

    @property
    def shots(self) -> int:
        """Number of independently replayed trajectories."""
        return len(self.optimizers) if self.shot_count is None else int(self.shot_count)

    def estimate(self, values) -> float:
        """Estimate a scalar observable from one value per trajectory."""
        return _weighted_estimate(values, self.weights or (1.0,) * self.shots)

    @property
    def effective_sample_size(self) -> float:
        """Return the importance-weight effective sample size."""
        return _effective_sample_size(self.weights or (1.0,) * self.shots)


@dataclass(frozen=True)
class StimHerald:
    """One classical herald bit sampled from a Stim noise instruction."""

    instruction_index: int
    site: int
    value: bool


@dataclass(frozen=True)
class StimDetector:
    """A compiled Stim detector annotation expressed in measurement offsets."""

    instruction_index: int
    detector_index: int
    rec_targets: tuple[tuple[int, bool], ...]
    coordinates: tuple[float, ...]
    measurement_count: int


@dataclass(frozen=True)
class StimObservable:
    """A compiled Stim logical-observable annotation."""

    instruction_index: int
    observable_index: int
    rec_targets: tuple[tuple[int, bool], ...]
    measurement_count: int


@dataclass(frozen=True)
class StimSyndromeRecord:
    """One resolved detector value from a replayed Stim trajectory."""

    instruction_index: int
    detector_index: int
    value: bool
    coordinates: tuple[float, ...] = ()


@dataclass(frozen=True)
class StimObservableRecord:
    """One resolved logical-observable value from a replayed Stim trajectory."""

    instruction_index: int
    observable_index: int
    value: bool


@dataclass(frozen=True)
class _StimPlanOperation:
    """One flattened Stim instruction in a prevalidated sampling plan."""

    instruction_index: int
    name: str
    args: tuple[float, ...]
    targets: tuple[tuple[str, int], ...]
    entries: tuple[object, ...] = ()
    is_noise: bool = False


@dataclass(frozen=True)
class StimCircuitPlan:
    """Reusable compilation of a supported Stim circuit into Pepsy events.

    Build this once with :func:`compile_stim_circuit`, then pass it to
    :func:`sample_stim_circuit`, :func:`sample_stim_circuits`, or
    :func:`run_stim_shots`. Compiling once avoids repeated repeat-block
    expansion and repeated construction of small Clifford matrices.
    """

    num_qubits: int
    operations: tuple[_StimPlanOperation, ...]
    detectors: tuple[StimDetector, ...] = ()
    observables: tuple[StimObservable, ...] = ()


@dataclass(frozen=True)
class StimNoiseSample:
    """One concrete sampled Stim trajectory ready for either MPS optimizer."""

    gate_stream: tuple[object, ...]
    faults: tuple[PauliFault, ...]
    heralds: tuple[StimHerald, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class StimShotResult:
    """Independent optimizer replays of a compiled Stim circuit."""

    optimizers: tuple[Any, ...]
    samples: tuple[StimNoiseSample, ...]
    plan: StimCircuitPlan | None = None

    @property
    def shots(self) -> int:
        """Number of independently replayed trajectories."""
        return len(self.optimizers)

    @property
    def gate_streams(self) -> tuple[tuple[object, ...], ...]:
        """Concrete stream emitted for every trajectory."""
        return tuple(sample.gate_stream for sample in self.samples)

    @property
    def faults(self) -> tuple[tuple[PauliFault, ...], ...]:
        """Physical Pauli faults sampled for every trajectory."""
        return tuple(sample.faults for sample in self.samples)

    @property
    def heralds(self) -> tuple[tuple[StimHerald, ...], ...]:
        """Herald bits sampled for every trajectory, in circuit order."""
        return tuple(sample.heralds for sample in self.samples)

    @property
    def weights(self) -> tuple[float, ...]:
        """Likelihood ratios for importance-sampled Stim trajectories."""
        return tuple(float(sample.weight) for sample in self.samples)

    def estimate(self, values) -> float:
        """Estimate a scalar observable from one value per shot."""
        return _weighted_estimate(values, self.weights)

    @property
    def effective_sample_size(self) -> float:
        """Return the importance-weight effective sample size."""
        return _effective_sample_size(self.weights)

    @property
    def measurements(self) -> tuple[tuple["TrajectoryMeasurementRecord", ...], ...]:
        """Structured mid-circuit measurements for every replayed shot."""
        return tuple(
            _optimizer_measurement_records(optimizer)
            for optimizer in self.optimizers
        )

    @property
    def syndromes(self):
        """Resolved detector records, one tuple per shot."""
        if self.plan is None:
            return tuple(() for _ in self.optimizers)
        return tuple(
            _resolve_stim_annotations(self.plan, optimizer)[0]
            for optimizer in self.optimizers
        )

    @property
    def observables(self):
        """Resolved logical-observable records, one tuple per shot."""
        if self.plan is None:
            return tuple(() for _ in self.optimizers)
        return tuple(
            _resolve_stim_annotations(self.plan, optimizer)[1]
            for optimizer in self.optimizers
        )


@dataclass(frozen=True)
class TrajectoryOutcome:
    """One named outcome of a local stochastic gate channel."""

    label: str
    gate: Any
    probability: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryChannel:
    """A user-defined local channel sampled as quantum trajectories.

    Create a fixed random-unitary mixture with :meth:`mixture`, or an arbitrary
    normalized single-site Kraus channel with :meth:`kraus`. Kraus channels
    sample the outcome from the evolving MPS state, then normalize the selected
    branch before later gates run. This is suitable for channels such as
    amplitude damping that cannot be represented as a fixed Pauli draw.
    """

    outcomes: tuple[TrajectoryOutcome, ...]
    mode: str

    def __post_init__(self):
        if self.mode not in {"mixture", "kraus"}:
            raise ValueError("TrajectoryChannel mode must be 'mixture' or 'kraus'.")
        if not self.outcomes:
            raise ValueError("TrajectoryChannel needs at least one outcome.")
        labels = [outcome.label for outcome in self.outcomes]
        if len(labels) != len(set(labels)):
            raise ValueError("TrajectoryChannel outcome labels must be unique.")
        matrices = tuple(_trajectory_matrix(outcome.gate) for outcome in self.outcomes)
        dim = matrices[0].shape[0]
        if any(matrix.shape != (dim, dim) for matrix in matrices):
            raise ValueError("TrajectoryChannel outcomes must be square matrices of one size.")
        nqubits = _trajectory_num_qubits(dim)
        if nqubits < 1:
            raise ValueError("TrajectoryChannel outcomes must act on at least one qubit.")
        if self.mode == "mixture":
            probabilities = [outcome.probability for outcome in self.outcomes]
            if any(probability is None for probability in probabilities):
                raise ValueError("Every mixture outcome needs an explicit probability.")
            probabilities = np.asarray(probabilities, dtype=float)
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0.0)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-12)
            ):
                raise ValueError("Trajectory mixture probabilities must be nonnegative and sum to one.")
            if not all(_is_unitary_matrix(matrix) for matrix in matrices):
                raise ValueError("Trajectory mixture outcomes must be unitary matrices.")
        else:
            if any(outcome.probability is not None for outcome in self.outcomes):
                raise ValueError("Kraus outcomes infer probabilities from the evolving state.")
            completeness = sum(
                matrix.conj().T @ matrix for matrix in matrices
            )
            if not np.allclose(completeness, np.eye(dim), atol=1e-10, rtol=1e-10):
                raise ValueError("Kraus operators must satisfy sum(K^dagger K) = I.")

    @classmethod
    def mixture(cls, outcomes) -> "TrajectoryChannel":
        """Build a fixed-probability random-unitary channel.

        ``outcomes`` contains ``(label, probability, matrix)`` entries.
        """
        return cls(
            tuple(
                TrajectoryOutcome(str(label), gate, float(probability))
                for label, probability, gate in outcomes
            ),
            "mixture",
        )

    @classmethod
    def kraus(cls, outcomes) -> "TrajectoryChannel":
        """Build a state-dependent channel from ``(label, Kraus_matrix)`` entries."""
        return cls(
            tuple(TrajectoryOutcome(str(label), gate) for label, gate in outcomes),
            "kraus",
        )

    @classmethod
    def depolarizing(cls, probability: float) -> "TrajectoryChannel":
        """Return a one-qubit depolarizing random-unitary channel."""
        probability = _unit_interval_probability(probability, "depolarizing")
        identity = np.eye(2, dtype=complex)
        return cls.mixture(
            (
                ("I", 1.0 - probability, identity),
                ("X", probability / 3.0, _PAULI_MATRICES["X"]),
                ("Y", probability / 3.0, _PAULI_MATRICES["Y"]),
                ("Z", probability / 3.0, _PAULI_MATRICES["Z"]),
            )
        )

    @classmethod
    def amplitude_damping(cls, gamma: float) -> "TrajectoryChannel":
        """Return a normalized single-qubit amplitude-damping Kraus channel."""
        gamma = _unit_interval_probability(gamma, "amplitude-damping")
        return cls.kraus(
            (
                (
                    "no_jump",
                    np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex),
                ),
                (
                    "jump",
                    np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
                ),
            )
        )


@dataclass(frozen=True)
class CoherentCrosstalkModel:
    """Insert coherent nearest-neighbour ``ZZ`` crosstalk into a gate stream.

    Pepsy rotations use ``exp(-i * theta * P / 2)``.  Therefore a physical
    crosstalk channel ``exp(+i * theta_zz * Z Z)`` from the surface-code model
    is emitted as ``("rzz", -2 * theta_zz, q0, q1)``.  With no adjacency map,
    the pair is the two-qubit gate support.  An adjacency mapping can instead
    add spectator-neighbour pairs touching the active gate support.

    ``sign_mode="random_sign"`` emits ``+theta``/``-theta`` with equal
    probability, which is useful for comparing coherent noise with an
    identical Pauli-twirled approximation.
    """

    theta: float
    adjacency: Mapping[int, tuple[int, ...]] | None = None
    sign_mode: str = "fixed"
    gate_names: tuple[str, ...] = tuple(sorted(_TWO_QUBIT_NAMES))

    def __post_init__(self):
        theta = float(self.theta)
        if not np.isfinite(theta):
            raise ValueError("theta must be finite.")
        mode = str(self.sign_mode).strip().lower().replace("-", "_")
        if mode not in {"fixed", "random_sign"}:
            raise ValueError("sign_mode must be 'fixed' or 'random_sign'.")
        names = tuple(str(name).strip().lower().replace("-", "_") for name in self.gate_names)
        if not names:
            raise ValueError("gate_names must contain at least one two-qubit gate.")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "sign_mode", mode)
        object.__setattr__(self, "gate_names", names)

    def _pairs_for_entry(self, entry):
        support = _event_support(entry)
        if support is None or len(support) != 2:
            return ()
        if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
            name = str(entry[0]).strip().lower().replace("-", "_")
            if name not in self.gate_names:
                return ()
        elif not (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and hasattr(entry[0], "shape")
        ):
            return ()
        active = tuple(int(site) for site in support)
        if self.adjacency is None:
            return (tuple(sorted(active)),)
        pairs = set()
        for left in active:
            for right in self.adjacency.get(left, ()):
                right = int(right)
                if right != left:
                    pairs.add(tuple(sorted((left, right))))
        return tuple(sorted(pairs))

    def transform(self, gates, *, seed=None):
        """Return a stream with coherent ``ZZ`` rotations after active gates."""
        rng = np.random.default_rng(seed)
        transformed = []
        for entry in _as_entries(gates):
            transformed.append(entry)
            for left, right in self._pairs_for_entry(entry):
                sign = 1.0
                if self.sign_mode == "random_sign":
                    sign = 1.0 if rng.integers(2) == 0 else -1.0
                transformed.append(("rzz", -2.0 * self.theta * sign, left, right))
        return tuple(transformed)

    apply = transform


@dataclass(frozen=True)
class TrajectoryEvent:
    """A user-defined channel event embedded in an otherwise ordinary gate stream."""

    channel: TrajectoryChannel
    where: Any

    def __post_init__(self):
        if not isinstance(self.channel, TrajectoryChannel):
            raise TypeError("TrajectoryEvent channel must be a TrajectoryChannel.")
        where = _trajectory_where(self.where)
        dimension = _trajectory_matrix(self.channel.outcomes[0].gate).shape[0]
        if len(where) != _trajectory_num_qubits(dimension):
            raise ValueError(
                "TrajectoryEvent support size must match its channel matrix dimension."
            )
        object.__setattr__(self, "where", where)


@dataclass(frozen=True)
class TrajectoryStreamPlan:
    """Normalized, reusable execution plan for a trajectory gate stream.

    The plan is backend-neutral. It owns the parsed event objects and the
    boundaries between ordinary replay segments and stateful events, while a
    concrete optimizer remains responsible for converting gate payloads to its
    array backend. Keeping this object separate from optimizer state lets the
    independent and coalesced runners share one stream parse.
    """

    entries: tuple[object, ...]
    ordinary_segments: tuple[tuple[int, int], ...] = ()
    trajectory_indices: tuple[int, ...] = ()
    control_indices: tuple[int, ...] = ()
    leakage_indices: tuple[int, ...] = ()

    @property
    def has_trajectory_events(self) -> bool:
        """Whether the plan contains state-dependent or mixture channels."""
        return bool(self.trajectory_indices)

    @property
    def has_controls(self) -> bool:
        """Whether the plan contains measurement/control events."""
        return bool(self.control_indices)

    @property
    def has_leakage(self) -> bool:
        """Whether the plan contains stateful leakage events."""
        return bool(self.leakage_indices)


@dataclass(frozen=True)
class TrajectoryRecord:
    """The sampled outcome of one :class:`TrajectoryEvent`."""

    event_index: int
    where: tuple[int, ...]
    label: str
    probability: float
    proposal_probability: float | None = None
    likelihood_ratio: float = 1.0


@dataclass(frozen=True)
class TrajectoryMeasurementRecord:
    """One structured mid-circuit measurement produced during replay.

    ``outcome`` is a Pauli eigenvalue (``+1``/``-1``), while computational-basis
    consumers can map it to ``0``/``1`` with ``bit = (1 - outcome) // 2``.
    ``probability`` is the Born probability when the backend exposes it;
    stabilizer backends may leave it as ``None`` because their native public
    measurement record intentionally stores only the outcome.
    """

    event_index: int | None
    pauli: str
    where: tuple[int, ...]
    outcome: int
    probability: float | None = None
    reset: bool = False
    instruction_index: int | None = None


@dataclass(frozen=True)
class LeakageRecord:
    """One stateful leakage event sampled during a trajectory replay.

    ``measurement`` uses the PECOS/Selene ternary convention for
    ``measure_leaked``: ``0`` and ``1`` are computational-basis bits, while
    ``2`` means the qubit was known to be leaked.
    """

    event_index: int
    kind: str
    site: Optional[int] = None
    probability: Optional[float] = None
    occurred: Optional[bool] = None
    initially_leaked: Optional[bool] = None
    finally_leaked: Optional[bool] = None
    measurement: Optional[int] = None
    branch: Optional[str] = None


@dataclass(frozen=True)
class TrajectorySample:
    """One sampled fixed-mixture stream and its selected channel outcomes."""

    gate_stream: tuple[object, ...]
    records: tuple[TrajectoryRecord, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class TrajectoryShotResult:
    """Independent MPS trajectory replays from a user-defined noisy gate stream.

    ``gate_streams`` record the normal gate events and selected channel matrices
    for each shot. Kraus-branch normalization is reflected in the retained final
    optimizer state rather than encoded as an additional gate-stream event.
    """

    optimizers: tuple[Any, ...]
    gate_streams: tuple[tuple[object, ...], ...]
    records: tuple[tuple[TrajectoryRecord, ...], ...]
    leakage_records: tuple[tuple[LeakageRecord, ...], ...] = ()
    measurement_records: tuple[tuple[TrajectoryMeasurementRecord, ...], ...] = ()
    weights: tuple[float, ...] = ()
    shot_count: int | None = None
    diagnostics: "TrajectoryDiagnostics | None" = None

    @property
    def shots(self) -> int:
        """Number of independently replayed trajectories."""
        return len(self.optimizers) if self.shot_count is None else int(self.shot_count)

    @property
    def measurements(self):
        """Alias for the structured per-shot measurement records."""
        if self.measurement_records:
            return self.measurement_records
        return tuple(_optimizer_measurement_records(optimizer) for optimizer in self.optimizers)

    def estimate(self, values) -> float:
        """Estimate a scalar observable from one value per trajectory."""
        return _weighted_estimate(values, self.weights or (1.0,) * self.shots)

    @property
    def effective_sample_size(self) -> float:
        """Return the importance-weight effective sample size."""
        return _effective_sample_size(self.weights or (1.0,) * self.shots)


@dataclass(frozen=True)
class CoalescedMeasurementRecord:
    """One forced mid-circuit measurement shared by a coalesced leaf.

    ``count`` is held by the containing :class:`CoalescedTrajectoryLeaf`.
    ``reset=True`` flags the internal forced collapse used to implement a bare
    reset; ordinary ``measure`` and ``measure_reset`` records use ``False``.
    """

    event_index: int
    pauli: str
    where: tuple[int, ...]
    outcome: int
    probability: float
    reset: bool = False


@dataclass(frozen=True)
class CoalescedTrajectoryLeaf:
    """One final state shared by ``count`` independent trajectories.

    The optimizer is one representative of all trajectories in this leaf.
    Its ``gate_stream`` is the concrete replay stream used for the selected
    noise and forced control outcomes. Product-state resets use the native
    trace-preserving reset fast path; entangled resets branch on their hidden
    measurement outcome without adding a user-visible measurement record.
    """

    optimizer: Any
    count: int
    gate_stream: tuple[object, ...]
    records: tuple[TrajectoryRecord, ...] = ()
    faults: tuple[PauliFault, ...] = ()
    heralds: tuple[StimHerald, ...] = ()
    measurements: tuple[CoalescedMeasurementRecord, ...] = ()
    leakage_records: tuple[LeakageRecord, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True)
class CoalescedTrajectoryResult:
    """Exact count-coalesced noisy trajectories.

    Instead of retaining one mutable optimizer for every shot, the result
    retains one optimizer per distinct sampled branch and its multiplicity.
    This is most effective when the expected number of non-identity faults is
    small.  The represented shots are still independent draws: branch counts
    are sampled with multinomial/binomial draws at every stochastic event.
    """

    leaves: tuple[CoalescedTrajectoryLeaf, ...]
    plan: StimCircuitPlan | None = None
    shot_count: int | None = None
    diagnostics: "TrajectoryDiagnostics | None" = None

    @property
    def shots(self) -> int:
        """Number of independently sampled trajectories represented."""
        if self.shot_count is not None:
            return int(self.shot_count)
        return sum(leaf.count for leaf in self.leaves)

    @property
    def branches(self) -> int:
        """Number of retained final optimizer states."""
        return len(self.leaves)

    @property
    def optimizers(self) -> tuple[Any, ...]:
        """One representative optimizer per final branch."""
        return tuple(leaf.optimizer for leaf in self.leaves)

    @property
    def counts(self) -> tuple[int, ...]:
        """Number of shots represented by each optimizer in :attr:`optimizers`."""
        return tuple(leaf.count for leaf in self.leaves)

    @property
    def weights(self) -> tuple[float, ...]:
        """Path likelihood ratios, one per retained coalesced leaf."""
        return tuple(float(leaf.weight) for leaf in self.leaves)

    def estimate(self, values) -> float:
        """Estimate a scalar observable from one value per retained leaf."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) != len(self.leaves):
            raise ValueError("coalesced estimates need one value per retained leaf.")
        numerator = sum(
            int(leaf.count) * float(leaf.weight) * float(value)
            for leaf, value in zip(self.leaves, values)
        )
        return float(numerator / self.shots) if self.shots else float("nan")

    @property
    def effective_sample_size(self) -> float:
        """Return the count-aware importance-weight effective sample size."""
        if not self.leaves:
            return 0.0
        first = sum(float(leaf.count) * float(leaf.weight) for leaf in self.leaves)
        second = sum(
            float(leaf.count) * float(leaf.weight) ** 2 for leaf in self.leaves
        )
        return float(first * first / second) if second > 0.0 else 0.0

    @property
    def syndromes(self):
        """Resolved Stim detector records, one tuple per coalesced leaf."""
        if self.plan is None:
            return tuple(() for _ in self.leaves)
        return tuple(
            _resolve_stim_annotations(self.plan, leaf.optimizer)[0]
            for leaf in self.leaves
        )

    @property
    def observables(self):
        """Resolved Stim observable records, one tuple per coalesced leaf."""
        if self.plan is None:
            return tuple(() for _ in self.leaves)
        return tuple(
            _resolve_stim_annotations(self.plan, leaf.optimizer)[1]
            for leaf in self.leaves
        )

    def sample_bits(self, *, seed=None, sampler_kwargs=None, shuffle=True):
        """Sample every leaf ``count`` times without expanding its optimizer state.

        This is a convenience wrapper around :func:`sample_coalesced_bits`.
        The returned sample rows are terminal readout data, not duplicated MPS
        optimizer objects.
        """
        return sample_coalesced_bits(
            self,
            seed=seed,
            sampler_kwargs=sampler_kwargs,
            shuffle=shuffle,
        )


@dataclass(frozen=True)
class NoisyResult:
    """Stable result facade returned by shot-aware optimizer APIs.

    The low-level runners intentionally retain their historical result types:
    independent replay returns a shot-shaped result, while coalesced replay
    returns one leaf per distinct branch. This facade gives the factory-free
    API one predictable surface while keeping the original object available as
    :attr:`raw` (and forwarding specialized attributes to it).
    """

    raw: NoisyShotResult | TrajectoryShotResult | CoalescedTrajectoryResult

    def __post_init__(self):
        if not isinstance(
            self.raw,
            (NoisyShotResult, TrajectoryShotResult, CoalescedTrajectoryResult),
        ):
            raise TypeError("raw must be a supported noisy runner result.")

    @property
    def coalesced(self) -> bool:
        """Whether the result stores count-bearing coalesced leaves."""
        return isinstance(self.raw, CoalescedTrajectoryResult)

    @property
    def shots(self) -> int:
        """Number of independent trajectories represented by the result."""
        return int(self.raw.shots)

    @property
    def branches(self) -> int:
        """Number of retained optimizer states in this representation."""
        return int(self.raw.branches if self.coalesced else len(self.optimizers))

    @property
    def optimizers(self) -> tuple[Any, ...]:
        """One optimizer per retained state or coalesced leaf."""
        return self.raw.optimizers

    @property
    def counts(self) -> tuple[int, ...]:
        """Shot multiplicity for each retained optimizer state."""
        if self.coalesced:
            return self.raw.counts
        return (1,) * len(self.optimizers)

    @property
    def gate_streams(self) -> tuple[tuple[object, ...], ...]:
        """Concrete replay stream for every retained optimizer state."""
        if self.coalesced:
            return tuple(leaf.gate_stream for leaf in self.raw.leaves)
        return self.raw.gate_streams

    @property
    def weights(self) -> tuple[float, ...]:
        """Importance weight for every retained state or leaf."""
        if self.raw.weights:
            return tuple(float(weight) for weight in self.raw.weights)
        return (1.0,) * self.branches

    @property
    def faults(self):
        """Pauli faults, grouped by retained state or coalesced leaf."""
        if self.coalesced:
            return tuple(leaf.faults for leaf in self.raw.leaves)
        return getattr(self.raw, "faults", tuple(() for _ in self.optimizers))

    @property
    def records(self):
        """Trajectory records, grouped by retained state or coalesced leaf."""
        if self.coalesced:
            return tuple(leaf.records for leaf in self.raw.leaves)
        return getattr(self.raw, "records", tuple(() for _ in self.optimizers))

    @property
    def measurements(self):
        """Structured measurement records, grouped by retained state or leaf."""
        if self.coalesced:
            return tuple(leaf.measurements for leaf in self.raw.leaves)
        measurements = getattr(self.raw, "measurements", None)
        if measurements is not None:
            return measurements
        return tuple(
            _optimizer_measurement_records(optimizer)
            for optimizer in self.optimizers
        )

    def estimate(self, values) -> float:
        """Estimate an observable using the result's shot multiplicities."""
        return self.raw.estimate(values)

    @property
    def effective_sample_size(self) -> float:
        """Return the importance-weight effective sample size."""
        return float(self.raw.effective_sample_size)

    @property
    def diagnostics(self) -> TrajectoryDiagnostics | None:
        """Accuracy and execution diagnostics for the replay."""
        return self.raw.diagnostics

    def __getattr__(self, name):
        """Preserve access to result-specific data through :attr:`raw`."""
        return getattr(self.raw, name)


@dataclass(frozen=True)
class CoalescedSampleResult:
    """Terminal bit samples drawn leaf-by-leaf from a coalesced ensemble.

    ``leaf_indices[row]`` identifies which
    :class:`CoalescedTrajectoryLeaf` produced ``configs[row]``. ``probs`` is
    available for ordinary MPS leaves and is ``None`` for STN leaves, whose
    scalable ``sample_bits`` path intentionally returns configurations only.
    ``lengths[row]`` gives the number of surviving sites. When conditional
    caps produce different lengths, shorter rows are padded on the right
    with ``-1``; only ``configs[row, :lengths[row]]`` contains measured bits.
    """

    configs: np.ndarray
    leaf_indices: np.ndarray
    probs: np.ndarray | None = None
    lengths: np.ndarray | None = None

    def __post_init__(self):
        # Keep the existing three-argument constructor useful for uniform
        # batches. Sampling supplies explicit lengths for a ragged register.
        if self.lengths is None:
            object.__setattr__(self, "lengths", np.full(
                self.configs.shape[0], self.configs.shape[1], dtype=np.int64
            ))

    @property
    def shots(self) -> int:
        """Number of terminal samples."""
        return int(self.configs.shape[0])

    @property
    def branches(self) -> int:
        """Number of represented leaves that produced at least one sample."""
        return int(np.unique(self.leaf_indices).size)


def _unit_interval_probability(value, label: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} probability must lie in [0, 1].")
    return value


def _weighted_estimate(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights):
        raise ValueError("weighted estimates need one value and weight per trajectory.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("weighted estimates require finite values and weights.")
    return float(np.dot(values, weights) / len(values)) if len(values) else float("nan")


def _effective_sample_size(weights) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError("effective sample size requires finite one-dimensional weights.")
    denominator = float(np.dot(weights, weights))
    numerator = float(weights.sum()) ** 2
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _trajectory_diagnostic_state(optimizer) -> dict[str, Any]:
    """Return mutable per-optimizer trajectory-quality counters."""
    state = getattr(optimizer, "_trajectory_diagnostics", None)
    if state is None:
        state = {
            "max_kraus_probability_residual": 0.0,
            "used_kraus_copy_fallback": False,
        }
        try:
            optimizer._trajectory_diagnostics = state
        except AttributeError:
            # External optimizer lookalikes can opt out of private bookkeeping.
            return state
    return state


def _record_kraus_probability_diagnostic(
    optimizer,
    *,
    residual: float | None = None,
    used_copy_fallback: bool = False,
):
    """Record quality information without adding work to the hot path."""
    state = _trajectory_diagnostic_state(optimizer)
    if residual is not None:
        state["max_kraus_probability_residual"] = max(
            float(state.get("max_kraus_probability_residual", 0.0)),
            abs(float(residual)),
        )
    state["used_kraus_copy_fallback"] = bool(
        state.get("used_kraus_copy_fallback", False) or used_copy_fallback
    )


def _trajectory_diagnostic_snapshot(optimizer) -> dict[str, Any]:
    """Copy scalar trajectory-quality counters before a state is discarded."""
    info = getattr(optimizer, "_trajectory_diagnostics", None) or {}
    return {
        "max_kraus_probability_residual": float(
            info.get("max_kraus_probability_residual", 0.0)
        ),
        "used_kraus_copy_fallback": bool(
            info.get("used_kraus_copy_fallback", False)
        ),
    }


def _trajectory_diagnostics(
    plan,
    states,
    *,
    shots: int,
    coalesced: bool,
    max_live_branches: int | None = None,
    diagnostic_infos=(),
) -> TrajectoryDiagnostics:
    """Build a lightweight public diagnostics snapshot for a replay result."""
    entries = tuple(getattr(plan, "entries", ()) or ())
    trajectory_events = sum(
        isinstance(entry, TrajectoryEvent) for entry in entries
    )
    measurement_events = 0
    leakage_events = 0
    for entry in entries:
        parts = MpsOptimizer.control_event_parts(entry)
        if parts is not None and parts[0] in {"measure", "measure_reset"}:
            measurement_events += 1
        if _leakage_event_parts(entry) is not None:
            leakage_events += 1

    max_residual = 0.0
    used_fallback = False
    for state in states:
        optimizer = getattr(state, "optimizer", state)
        info = getattr(optimizer, "_trajectory_diagnostics", None) or {}
        max_residual = max(
            max_residual,
            abs(float(info.get("max_kraus_probability_residual", 0.0))),
        )
        used_fallback = bool(
            used_fallback or info.get("used_kraus_copy_fallback", False)
        )
    for info in diagnostic_infos:
        max_residual = max(
            max_residual,
            abs(float(info.get("max_kraus_probability_residual", 0.0))),
        )
        used_fallback = bool(
            used_fallback or info.get("used_kraus_copy_fallback", False)
        )

    branches = len(states)
    return TrajectoryDiagnostics(
        shots=int(shots),
        branches=int(branches),
        coalesced=bool(coalesced),
        stream_events=len(entries),
        trajectory_events=int(trajectory_events),
        measurement_events=int(measurement_events),
        leakage_events=int(leakage_events),
        max_live_branches=(
            int(shots) if not coalesced else int(branches)
            if max_live_branches is None
            else int(max_live_branches)
        ),
        max_kraus_probability_residual=float(max_residual),
        used_kraus_copy_fallback=bool(used_fallback),
    )


def _trajectory_matrix(gate) -> np.ndarray:
    """Convert a channel outcome to a small dense matrix for validation."""
    matrix = np.asarray(gate, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Trajectory channel outcomes must be square matrices.")
    return matrix


def _trajectory_num_qubits(dimension: int) -> int:
    """Return the qubit arity of a square local channel dimension."""
    if dimension < 1:
        raise ValueError("Trajectory channel outcomes cannot have zero dimension.")
    nqubits = int(round(np.log2(dimension)))
    if 2**nqubits != dimension:
        raise ValueError(
            "Trajectory channel outcomes must have a 2**k by 2**k qubit dimension."
        )
    return nqubits


def _trajectory_where(where) -> tuple[int, ...]:
    """Normalize a local trajectory support to logical qubit labels."""
    if isinstance(where, Integral):
        return (int(where),)
    if (
        isinstance(where, (tuple, list))
        and where
        and all(isinstance(site, Integral) for site in where)
    ):
        return tuple(int(site) for site in where)
    raise ValueError("TrajectoryEvent where must be an integer or non-empty integer tuple.")


def _is_unitary_matrix(matrix: np.ndarray, *, atol: float = 1e-10) -> bool:
    """Return whether a dense channel outcome is unitary."""
    return bool(
        np.allclose(
            matrix.conj().T @ matrix,
            np.eye(matrix.shape[0], dtype=matrix.dtype),
            atol=atol,
            rtol=atol,
        )
    )


def _trajectory_real_scalar(value, *, label: str) -> float:
    """Convert a backend scalar expected to be real into a Python float."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    value = complex(value)
    if abs(value.imag) > 1e-9:
        raise ValueError(f"{label} must be real, got {value!r}.")
    return float(value.real)


def _as_entries(gates) -> list[object]:
    """Normalize a single bundled event or iterable gate stream."""
    if gates is None:
        return []
    if isinstance(gates, TrajectoryStreamPlan):
        return list(gates.entries)
    if isinstance(gates, Mapping):
        return [gates]
    if isinstance(gates, (tuple, list)):
        if not gates:
            return []
        first = gates[0]
        if isinstance(first, str) or hasattr(first, "shape"):
            return [gates]
        if isinstance(first, Mapping) or isinstance(first, (tuple, list)):
            return list(gates)
    try:
        return list(gates)
    except TypeError as exc:
        raise TypeError("gates must be a bundled entry or iterable gate stream.") from exc


def _sites(where) -> tuple[int, ...]:
    if isinstance(where, Integral):
        return (int(where),)
    if isinstance(where, (tuple, list)) and where and all(
        isinstance(site, Integral) for site in where
    ):
        return tuple(int(site) for site in where)
    raise ValueError(f"Cannot determine integer gate support from {where!r}.")


def _normalize_stream_name(name) -> str:
    return str(name).strip().lower().replace("-", "_")


def _event_support(entry) -> Optional[tuple[int, ...]]:
    """Return the physical support that should receive independent Pauli noise."""
    if isinstance(entry, Mapping):
        # Mapping forms in the optimizer streams currently represent controls
        # and coefficient-frame sub-MPOs, which must not receive an implicit
        # physical post-gate channel.
        return None
    if not isinstance(entry, (tuple, list)) or not entry:
        raise ValueError(f"Unsupported gate stream entry: {entry!r}.")

    head = entry[0]
    if isinstance(head, str):
        name = _normalize_stream_name(head)
        if name in _STOCHASTIC_EVENT_NAMES:
            return None
        if name in _LEAKAGE_EVENT_NAMES:
            return None
        if name in _CONTROL_NAMES:
            return None
        if name in _ONE_QUBIT_NAMES:
            if len(entry) != 2:
                raise ValueError(f"{head!r} gate requires one target site.")
            return _sites(entry[1])
        if name in _TWO_QUBIT_NAMES:
            if len(entry) != 3:
                raise ValueError(f"{head!r} gate requires two target sites.")
            return _sites((entry[1], entry[2]))
        if name in _ONE_QUBIT_ROTATIONS:
            if len(entry) != 3:
                raise ValueError(f"{head!r} gate requires angle and target site.")
            return _sites(entry[2])
        if name in _TWO_QUBIT_ROTATIONS:
            if len(entry) != 4:
                raise ValueError(f"{head!r} gate requires angle and two target sites.")
            return _sites((entry[2], entry[3]))
        if name == "rot":
            if len(entry) != 4:
                raise ValueError("'rot' gate requires angle, Pauli axes, and target sites.")
            return _sites(entry[3])
        raise ValueError(
            f"Cannot infer a physical support for named gate {head!r}; use an "
            "ordinary (matrix, where) event or sample the stream explicitly."
        )

    if len(entry) != 2:
        raise ValueError(f"Unsupported matrix gate stream entry: {entry!r}.")
    return _sites(entry[1])


def _conditional_pauli_support(entry):
    """Return ``(payload, support)`` for a noisy ordinary conditional action.

    Conditional measurements, resets, control wrappers, and sub-MPO actions do
    not have a well-defined post-gate one-qubit Pauli channel here. Ordinary
    named or matrix gates do, and are handled after the predicate is resolved.
    """
    parts = MpsOptimizer.control_event_parts(entry)
    if parts is None or parts[0] != "conditional":
        return None
    _name, payload, _where = parts
    action = payload["action"]
    if MpsOptimizer.control_event_parts(action) is not None:
        return None
    try:
        support = _event_support(action)
    except ValueError:
        # Let the optimizer report the original unsupported action with its
        # normal, more specific validation path.
        return None
    if support is None:
        return None
    return payload, support


def _conditional_matches(optimizer, payload) -> bool:
    """Resolve one conditional predicate against an optimizer's records."""
    record_index, expected = _resolve_conditional(
        payload, len(getattr(optimizer, "measurements", ()))
    )
    record = optimizer.measurements[record_index]
    outcome = _normalize_optimizer_measurement(record)[2]
    return int(outcome < 0) == expected


def _pauli_matrix(label: str, *, like=None):
    matrix = _PAULI_MATRICES[label].copy()
    if like is None:
        return matrix
    try:
        import autoray as ar

        return ar.do("array", matrix, like=like)
    except Exception:  # pragma: no cover - backend-specific fallback
        return matrix


def _identity_matrix(nqubits: int) -> np.ndarray:
    return np.eye(2 ** int(nqubits), dtype=complex)


def _pauli_product_matrix(label: str) -> np.ndarray:
    label = str(label).upper()
    matrices = []
    for axis in label:
        if axis == "I":
            matrices.append(_identity_matrix(1))
        elif axis in _PAULI_MATRICES:
            matrices.append(_PAULI_MATRICES[axis])
        else:
            raise ValueError(f"Unknown Pauli label {label!r}.")
    out = matrices[0]
    for matrix in matrices[1:]:
        out = np.kron(out, matrix)
    return out


def _mapping_probability(mapping, label: str) -> float:
    keys = (
        label,
        label.lower(),
        tuple(label),
        tuple(axis.lower() for axis in label),
    )
    for key in keys:
        if key in mapping:
            return float(mapping[key])
    return 0.0


def _pauli_channel_probabilities(probabilities, labels, *, event: str) -> list[float]:
    if isinstance(probabilities, Mapping):
        values = [_mapping_probability(probabilities, label) for label in labels]
    else:
        values = [float(value) for value in probabilities]
        if len(values) != len(labels):
            raise ValueError(
                f"{event} needs {len(labels)} probability values, got {len(values)}."
            )
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"{event} probabilities must be finite and nonnegative.")
    total = float(sum(values))
    if total > 1.0 + 1e-12:
        raise ValueError(f"{event} probabilities must sum to at most one.")
    return values


def _trajectory_event_from_stochastic_entry(entry):
    """Lower a Pepsy stochastic stream entry into a trajectory event."""
    if not (isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str)):
        return None
    name = _normalize_stream_name(entry[0])
    if name not in _STOCHASTIC_EVENT_NAMES:
        return None

    if name in _STOCHASTIC_SINGLE_PAULI_NAMES:
        if len(entry) != 3:
            raise ValueError(f"{entry[0]!r} expects probability and target qubit.")
        probability = _unit_interval_probability(entry[1], name)
        axis = _STOCHASTIC_SINGLE_PAULI_NAMES[name]
        channel = TrajectoryChannel.mixture(
            (
                ("I", 1.0 - probability, _identity_matrix(1)),
                (axis, probability, _PAULI_MATRICES[axis]),
            )
        )
        return TrajectoryEvent(channel, entry[2])

    if name in {"depolarize1", "depolarize_1"}:
        if len(entry) != 3:
            raise ValueError(f"{entry[0]!r} expects probability and target qubit.")
        return TrajectoryEvent(TrajectoryChannel.depolarizing(entry[1]), entry[2])

    if name in {"depolarize2", "depolarize_2"}:
        if len(entry) != 4:
            raise ValueError(f"{entry[0]!r} expects probability and two target qubits.")
        probability = _unit_interval_probability(entry[1], name)
        labels = ["".join(pair) for pair in _STIM_PAULI_2_OUTCOMES]
        channel = TrajectoryChannel.mixture(
            [("II", 1.0 - probability, _identity_matrix(2))]
            + [
                (label, probability / 15.0, _pauli_product_matrix(label))
                for label in labels
            ]
        )
        return TrajectoryEvent(channel, (entry[2], entry[3]))

    if name in {"pauli_channel1", "pauli_channel_1"}:
        if len(entry) != 3:
            raise ValueError(f"{entry[0]!r} expects probabilities and target qubit.")
        labels = ("X", "Y", "Z")
        values = _pauli_channel_probabilities(entry[1], labels, event=name)
        channel = TrajectoryChannel.mixture(
            [("I", 1.0 - float(sum(values)), _identity_matrix(1))]
            + [
                (label, probability, _PAULI_MATRICES[label])
                for label, probability in zip(labels, values)
            ]
        )
        return TrajectoryEvent(channel, entry[2])

    if name in {"pauli_channel2", "pauli_channel_2"}:
        if len(entry) != 4:
            raise ValueError(f"{entry[0]!r} expects probabilities and two target qubits.")
        labels = tuple("".join(pair) for pair in _STIM_PAULI_2_OUTCOMES)
        values = _pauli_channel_probabilities(entry[1], labels, event=name)
        channel = TrajectoryChannel.mixture(
            [("II", 1.0 - float(sum(values)), _identity_matrix(2))]
            + [
                (label, probability, _pauli_product_matrix(label))
                for label, probability in zip(labels, values)
            ]
        )
        return TrajectoryEvent(channel, (entry[2], entry[3]))

    if name == "amplitude_damping":
        if len(entry) != 3:
            raise ValueError(f"{entry[0]!r} expects gamma and target qubit.")
        return TrajectoryEvent(TrajectoryChannel.amplitude_damping(entry[1]), entry[2])

    raise AssertionError(f"Unhandled stochastic stream entry {entry!r}.")


def _leakage_event_parts(entry):
    """Return ``(kind, payload, where)`` for a stateful leakage stream entry."""
    if isinstance(entry, Mapping):
        kind = entry.get("kind", entry.get("type", entry.get("event", None)))
        if kind is None:
            return None
        name = _normalize_stream_name(kind)
        if name not in _LEAKAGE_EVENT_NAMES:
            return None
        if name in {"leak2depolar", "leak_to_depolar"}:
            return (
                "leak2depolar",
                {"enabled": bool(entry.get("enabled", entry.get("value", True)))},
                (),
            )
        where = entry.get("where", entry.get("site", entry.get("qubit", None)))
        if where is None:
            raise ValueError(f"{kind!r} leakage event needs a target qubit.")
        (site,) = _trajectory_where(where)
        if name in {"measure_leaked", "measure_leakage"}:
            return "measure_leaked", {}, (site,)
        probability = _unit_interval_probability(
            entry.get("probability", entry.get("p", 0.0)), name
        )
        if name in {"leak", "leakage"}:
            return "leakage", {"probability": probability, "depolarize": False}, (site,)
        if name == "leakage_depolarize":
            return "leakage", {"probability": probability, "depolarize": True}, (site,)
        if name in {"leakage_return", "leakage_seepage", "seepage", "unleak"}:
            return "leakage_return", {"probability": probability}, (site,)
        raise AssertionError(f"Unhandled leakage event {entry!r}.")

    if not (isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str)):
        return None
    name = _normalize_stream_name(entry[0])
    if name not in _LEAKAGE_EVENT_NAMES:
        return None

    if name in {"leak2depolar", "leak_to_depolar"}:
        if len(entry) != 2:
            raise ValueError(f"{entry[0]!r} expects a boolean enabled flag.")
        return "leak2depolar", {"enabled": bool(entry[1])}, ()

    if name in {"measure_leaked", "measure_leakage"}:
        if len(entry) != 2:
            raise ValueError(f"{entry[0]!r} expects a target qubit.")
        (site,) = _trajectory_where(entry[1])
        return "measure_leaked", {}, (site,)

    if len(entry) != 3:
        raise ValueError(f"{entry[0]!r} expects probability and target qubit.")
    probability = _unit_interval_probability(entry[1], name)
    (site,) = _trajectory_where(entry[2])
    if name in {"leak", "leakage"}:
        return "leakage", {"probability": probability, "depolarize": False}, (site,)
    if name == "leakage_depolarize":
        return "leakage", {"probability": probability, "depolarize": True}, (site,)
    if name in {"leakage_return", "leakage_seepage", "seepage", "unleak"}:
        return "leakage_return", {"probability": probability}, (site,)
    raise AssertionError(f"Unhandled leakage event {entry!r}.")


def _contains_stochastic_entries(entries) -> bool:
    return any(_trajectory_event_from_stochastic_entry(entry) is not None for entry in entries)


def _contains_leakage_entries(entries) -> bool:
    return any(_leakage_event_parts(entry) is not None for entry in entries)


def _sample_gate_stream(
    gates,
    error_model: PauliErrorModel,
    rng,
    *,
    proposal_model: PauliErrorModel | None = None,
    return_weight: bool = False,
):
    if proposal_model is not None and not isinstance(proposal_model, PauliErrorModel):
        raise TypeError("proposal_model must be a PauliErrorModel or None.")
    stream = []
    faults = []
    weight = 1.0
    target_probabilities = error_model.probabilities
    proposal_probabilities = (
        target_probabilities
        if proposal_model is None
        else proposal_model.probabilities
    )
    for gate_index, entry in enumerate(_as_entries(gates)):
        if _trajectory_event_from_stochastic_entry(entry) is not None:
            raise ValueError(
                "Stream-local stochastic entries require run_trajectory_shots(...) "
                "or run_coalesced_trajectory_shots(...). PauliErrorModel is a "
                "convenience macro for clean deterministic streams."
            )
        if _leakage_event_parts(entry) is not None:
            raise ValueError(
                "Stateful leakage entries require run_trajectory_shots(...). "
                "PauliErrorModel is a convenience macro for clean deterministic streams."
            )
        stream.append(entry)
        conditional = _conditional_pauli_support(entry)
        if conditional is not None:
            payload, support = conditional
            action = payload["action"]
            like = (
                action[0]
                if isinstance(action, (tuple, list))
                and action
                and not isinstance(action[0], str)
                else None
            )
            for site in support:
                labels = tuple(target_probabilities)
                pauli = str(
                    rng.choice(
                        labels,
                        p=tuple(proposal_probabilities[label] for label in labels),
                    )
                )
                target = float(target_probabilities[pauli])
                proposal = float(proposal_probabilities[pauli])
                if proposal <= 0.0:
                    raise ValueError("proposal_model sampled a zero-probability branch.")
                weight *= target / proposal
                if pauli == "I":
                    continue
                # Keep the fault behind the same predicate. This is valid for
                # ordinary unitary actions because they do not change records.
                stream.append(
                    (
                        "if",
                        payload["record"],
                        payload["bit"],
                        (_pauli_matrix(pauli, like=like), site),
                    )
                )
                faults.append(PauliFault(gate_index=gate_index, site=site, pauli=pauli))
            continue
        support = _event_support(entry)
        if support is None:
            continue
        like = entry[0] if isinstance(entry, (tuple, list)) else None
        for site in support:
            labels = tuple(target_probabilities)
            pauli = str(
                rng.choice(labels, p=tuple(proposal_probabilities[label] for label in labels))
            )
            target = float(target_probabilities[pauli])
            proposal = float(proposal_probabilities[pauli])
            if proposal <= 0.0:
                raise ValueError("proposal_model sampled a zero-probability branch.")
            weight *= target / proposal
            if pauli == "I":
                continue
            stream.append((_pauli_matrix(pauli, like=like), site))
            faults.append(PauliFault(gate_index=gate_index, site=site, pauli=pauli))
    if return_weight:
        return stream, tuple(faults), float(weight)
    return stream, tuple(faults)


def _run_noisy_conditional_shot(
    optimizer,
    entries,
    error_model,
    rng,
    run_kwargs,
    *,
    proposal_model=None,
):
    """Replay a Pauli shot while resolving conditional actions online.

    A static noisy stream cannot know whether a feed-forward action will run,
    so it would either attach metadata to a false branch or apply an
    importance ratio for a channel that should not have been sampled. Ordinary
    nonconditional segments remain batched; only the boundaries around a
    conditional gate are replayed sequentially.
    """
    stream = []
    faults = []
    pending = []
    weight = 1.0

    def flush():
        if pending:
            _run_trajectory_entries(optimizer, tuple(pending), run_kwargs)
            stream.extend(pending)
            pending.clear()

    for gate_index, entry in enumerate(entries):
        conditional = _conditional_pauli_support(entry)
        if conditional is None:
            sampled, local_faults, local_weight = _sample_gate_stream(
                (entry,),
                error_model,
                rng,
                proposal_model=proposal_model,
                return_weight=True,
            )
            pending.extend(sampled)
            faults.extend(
                PauliFault(gate_index, fault.site, fault.pauli)
                for fault in local_faults
            )
            weight *= local_weight
            continue

        payload, _support = conditional
        flush()
        _run_trajectory_entries(optimizer, (entry,), run_kwargs)
        stream.append(entry)
        if not _conditional_matches(optimizer, payload):
            continue

        action = payload["action"]
        sampled, local_faults, local_weight = _sample_gate_stream(
            (action,),
            error_model,
            rng,
            proposal_model=proposal_model,
            return_weight=True,
        )
        # ``sampled[0]`` is the ideal action. The original conditional action
        # has already run, so replay only its sampled post-action faults.
        for fault_entry in sampled[1:]:
            _run_trajectory_entries(optimizer, (fault_entry,), run_kwargs)
            stream.append(fault_entry)
        faults.extend(
            PauliFault(gate_index, fault.site, fault.pauli)
            for fault in local_faults
        )
        weight *= local_weight

    flush()
    return tuple(stream), tuple(faults), float(weight)


def sample_noisy_gate_stream(gates, error_model: PauliErrorModel, *, seed=None):
    """Sample one concrete post-gate Pauli-noise stream.

    This functional form is equivalent to
    ``error_model.sample_gate_stream(gates, seed=seed)``.
    """
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    return error_model.sample_gate_stream(gates, seed=seed)


def sample_noisy_gate_streams(
    gates, error_model: PauliErrorModel, shots: int, *, seed=None
):
    """Sample ``shots`` independent concrete post-gate Pauli-noise streams."""
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    child_seeds = np.random.SeedSequence(seed).spawn(int(shots))
    return [
        _sample_gate_stream(gates, error_model, np.random.default_rng(child_seed))[0]
        for child_seed in child_seeds
    ]


def _validate_strategy(strategy):
    """Normalize an independent/coalesced trajectory-replay strategy."""
    strategy = str(strategy).lower()
    if strategy not in {"independent", "coalesced", "auto"}:
        raise ValueError(
            "strategy must be 'independent', 'coalesced', or 'auto'."
        )
    return strategy


def _validate_retain(retain):
    """Normalize the amount of per-shot state kept in a result."""
    retain = str(retain).strip().lower().replace("-", "_")
    aliases = {"states": "final", "state": "final", "all": "all"}
    retain = aliases.get(retain, retain)
    if retain not in {"all", "final", "none"}:
        raise ValueError("retain must be 'all', 'final', or 'none'.")
    return retain


def _validate_max_branches(max_branches):
    """Validate an optional positive cap for retained coalesced leaves."""
    if max_branches is None:
        return None
    if (
        isinstance(max_branches, bool)
        or not isinstance(max_branches, Integral)
        or max_branches < 1
    ):
        raise ValueError("max_branches must be a positive integer or None.")
    return int(max_branches)


def _validate_max_branch_factor(max_branch_factor):
    """Validate an optional per-event coalesced branching cap."""
    if max_branch_factor is None:
        return None
    if (
        isinstance(max_branch_factor, bool)
        or not isinstance(max_branch_factor, Integral)
        or max_branch_factor < 1
    ):
        raise ValueError("max_branch_factor must be a positive integer or None.")
    return int(max_branch_factor)


def _validate_parallel_workers(parallel_workers):
    if parallel_workers is None:
        return 1
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, Integral)
        or parallel_workers < 1
    ):
        raise ValueError("parallel_workers must be a positive integer or None.")
    return int(parallel_workers)


def _validate_parallel_backend(parallel_backend):
    backend = str(parallel_backend).strip().lower().replace("-", "_")
    if backend not in {"thread", "threads", "gpu", "serial"}:
        raise ValueError(
            "parallel_backend must be 'thread', 'gpu', or 'serial'."
        )
    return backend


def _parallel_map_ordered(
    function,
    values,
    parallel_workers,
    parallel_backend,
    progress=None,
):
    """Apply independent optimizer work in deterministic input order.

    GPU execution deliberately uses threads rather than processes: optimizer
    state and device allocations remain in the caller's process. The caller's
    factory is responsible for selecting the desired device/backend.
    """
    values = tuple(values)
    if parallel_workers <= 1 or parallel_backend == "serial":
        results = []
        for value in values:
            results.append(function(value))
            if progress is not None:
                progress(1)
        return results
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        results = []
        for result in executor.map(function, values):
            results.append(result)
            if progress is not None:
                progress(1)
        return results


def _importance_distribution(policy, event_index, outcomes, target, optimizer=None):
    """Return ``(target, proposal, ratios)`` for one channel event."""
    return _importance_label_distribution(
        policy,
        event_index,
        tuple(outcome.label for outcome in outcomes),
        target,
        optimizer,
    )


def _importance_label_distribution(
    policy, event_index, labels, target, optimizer=None
):
    """Return ``(target, proposal, ratios)`` for arbitrary labelled outcomes."""
    target = np.asarray(target, dtype=float)
    target = target / float(target.sum())
    if policy is None:
        proposal = target.copy()
    else:
        proposal = policy.probabilities(event_index, labels, target, optimizer)
    ratios = np.zeros_like(target)
    nonzero = proposal > 0.0
    ratios[nonzero] = target[nonzero] / proposal[nonzero]
    if policy is not None and policy.max_likelihood_ratio is not None:
        if np.any(ratios > policy.max_likelihood_ratio * (1.0 + 1e-12)):
            raise ValueError(
                f"importance likelihood ratio exceeds max_likelihood_ratio at "
                f"event {event_index}."
            )
    return target, proposal, ratios


def _expected_pauli_faults(entries, error_model):
    """Return lambda, the expected non-identity Pauli faults per shot."""
    targets = 0
    for entry in entries:
        support = _event_support(entry)
        if support is None:
            conditional = _conditional_pauli_support(entry)
            support = None if conditional is None else conditional[1]
        if support is not None:
            targets += len(support)
    return targets * (error_model.p_x + error_model.p_y + error_model.p_z)


def _has_unforced_branching_control(entries):
    """Return whether a stream contains a control that needs count splitting."""
    for entry in entries:
        parts = MpsOptimizer.control_event_parts(entry)
        if parts is None:
            continue
        name, payload, _where = parts
        if name in {"reset", "cap"}:
            return True
        if name == "measure" and payload.get("outcome") is None:
            return True
        if name == "measure_reset" and any(
            outcome is None for outcome in payload.get("outcomes", ())
        ):
            return True
    return False


def _auto_prefers_coalescing(entries, error_model, max_expected_faults):
    """Choose the rare-fault branch only when it is structurally favorable."""
    if _has_unforced_branching_control(entries):
        return False
    return _expected_pauli_faults(entries, error_model) <= max_expected_faults


def _resolve_auto_parallel_strategy(
    gates,
    shots,
    *,
    error_model=None,
    max_branches=None,
    max_branch_factor=None,
    auto_max_expected_faults=0.1,
):
    """Resolve ``auto`` before dispatching work to local workers.

    The decision mirrors the serial runners' representation choice. Keeping
    it here lets optimizer-level worker selection preserve branch shape while
    still using the parallel independent/coalesced implementations.
    """
    max_branches = _validate_max_branches(max_branches)
    max_branch_factor = _validate_max_branch_factor(max_branch_factor)
    if error_model is not None:
        if not isinstance(error_model, PauliErrorModel):
            raise TypeError("error_model must be a PauliErrorModel.")
        threshold = float(auto_max_expected_faults)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("auto_max_expected_faults must be finite and nonnegative.")
        return (
            "coalesced"
            if _auto_prefers_coalescing(_as_entries(gates), error_model, threshold)
            else "independent"
        )
    plan = compile_trajectory_stream(gates)
    return (
        "coalesced"
        if _trajectory_coalescing_fits_cap(
            plan, shots, max_branches, max_branch_factor
        )
        else "independent"
    )


def _trajectory_coalescing_fits_cap(plan, shots, max_branches, max_branch_factor):
    """Conservatively avoid an auto-coalesced run that must restart.

    This is only an upper-bound preflight. It may choose independent replay
    earlier than necessary, but it never drops probability mass and prevents
    the expensive deterministic-prefix restart when a stream has obviously
    more possible leaves than the configured cap.
    """
    if max_branches is None:
        return True
    possible = 1
    for entry in plan.entries:
        event_factor = 1
        if isinstance(entry, TrajectoryEvent):
            event_factor = len(entry.channel.outcomes)
        else:
            parts = MpsOptimizer.control_event_parts(entry)
            if parts is None:
                continue
            name, payload, where = parts
            if name == "measure" and payload.get("outcome") is None:
                event_factor = 2 ** len(where)
            elif name == "measure_reset":
                event_factor = 2 ** sum(
                    outcome is None for outcome in payload.get("outcomes", ())
                )
        if max_branch_factor is not None and event_factor > max_branch_factor:
            return False
        possible *= event_factor
        if min(possible, int(shots)) > max_branches:
            return False
    return min(int(shots), possible) <= max_branches


def _tree_layout_stream(gate_stream):
    """Make a TreeOptimizer-compatible layout stream from noisy entries."""
    layout_stream = []
    for entry in gate_stream:
        trajectory_event = (
            entry
            if isinstance(entry, TrajectoryEvent)
            else _trajectory_event_from_stochastic_entry(entry)
        )
        if trajectory_event is not None:
            outcome = trajectory_event.channel.outcomes[0]
            layout_stream.append(
                (_trajectory_matrix(outcome.gate), trajectory_event.where)
            )
        else:
            layout_stream.append(entry)
    return tuple(layout_stream)


def _tree_shot_factory(initial_tn, gate_stream, tree_settings):
    """Build a fresh TreeOptimizer factory from one initial tree state."""
    if not isinstance(tree_settings, Mapping):
        raise TypeError("tree_settings must be a mapping of TreeOptimizer settings.")
    constructor = dict(tree_settings)
    forbidden = sorted(set(constructor) & {"gates", "tn", "state", "run"})
    if forbidden:
        names = ", ".join(repr(name) for name in forbidden)
        raise TypeError(
            f"tree_settings cannot contain {names}; pass the initial tree state and "
            "gate stream to TreeNoisy(...)."
        )
    accepted = set(inspect.signature(TreeOptimizer.__init__).parameters)
    accepted.difference_update({"self", "gates", "tn", "state", "run"})
    unknown = sorted(set(constructor) - accepted)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise TypeError(f"unknown TreeOptimizer setting(s): {names}.")
    try:
        template = initial_tn.copy()
    except AttributeError as exc:
        raise TypeError(
            "initial_tn must provide copy() like a TreeTensorNetwork or "
            "product quimb MPS."
        ) from exc
    layout_stream = _tree_layout_stream(gate_stream)

    def make_optimizer():
        return TreeOptimizer(
            layout_stream,
            tn=template.copy(),
            run=False,
            **constructor,
        )

    return make_optimizer


class TreeNoisy:
    """Factory-free noisy tree-tensor gate-stream simulator.

    The public methods mirror the MPS noisy replay API, but each shot is constructed
    with :class:`TreeOptimizer`. The initial state may be an entangled
    ``TreeTensorNetwork`` or a product quimb MPS; the latter is mounted on the
    tree selected from ``gate_stream`` and ``tree_settings``. Feed-forward
    events use the same ``("if", record, bit, action)`` contract as MPS replay.
    """

    def __init__(self, initial_tn, gate_stream, *, tree_settings=None, **optimizer_settings):
        if tree_settings is None:
            constructor = {}
        elif not isinstance(tree_settings, Mapping):
            raise TypeError("tree_settings must be a mapping of TreeOptimizer settings.")
        else:
            constructor = dict(tree_settings)
        constructor.update(optimizer_settings)
        self.tree_settings = MappingProxyType(dict(constructor))
        try:
            self._initial_tn = initial_tn.copy()
        except AttributeError as exc:
            raise TypeError(
                "initial_tn must provide copy() like a TreeTensorNetwork or "
                "product quimb MPS."
            ) from exc
        if isinstance(gate_stream, TrajectoryEvent):
            entries = (gate_stream,)
        else:
            entries = tuple(_as_entries(gate_stream))
        self.gate_stream = entries
        self._factory = _tree_shot_factory(
            self._initial_tn,
            self.gate_stream,
            self.tree_settings,
        )

    @property
    def initial_tn(self):
        """Return a defensive copy of the configured initial tree state."""
        return self._initial_tn.copy()

    def run_trajectory(self, shots: int, **runner_kwargs):
        """Run stream-local trajectory noise through the exact shot runner."""
        return NoisyResult(
            run_trajectory_shots(
                self._factory,
                self.gate_stream,
                shots,
                **runner_kwargs,
            )
        )

    def run_noisy(self, error_model: PauliErrorModel, shots: int, **runner_kwargs):
        """Run a clean stream with the legacy :class:`PauliErrorModel`."""
        return NoisyResult(
            run_noisy_shots(
                self._factory,
                self.gate_stream,
                error_model,
                shots,
                **runner_kwargs,
            )
        )

    def run(
        self,
        shots: int,
        *,
        error_model: PauliErrorModel | None = None,
        seed=None,
        run_kwargs: Optional[Mapping[str, Any]] = None,
        strategy: str = "auto",
        max_branches: int | None = _AUTO_MAX_BRANCHES,
        auto_max_expected_faults: float = _AUTO_MAX_EXPECTED_FAULTS,
        importance_sampling=None,
        max_branch_factor: int | None = None,
        parallel_workers: int = 1,
        parallel_backend: str = "thread",
        retain: str = "all",
    ) -> NoisyResult:
        """Run the configured TreeOptimizer gate stream."""
        common = {
            "seed": seed,
            "run_kwargs": run_kwargs,
            "strategy": strategy,
            "max_branches": max_branches,
            "importance_sampling": importance_sampling,
            "max_branch_factor": max_branch_factor,
            "parallel_workers": parallel_workers,
            "parallel_backend": parallel_backend,
            "retain": retain,
        }
        if error_model is None:
            return self.run_trajectory(shots, **common)
        common["auto_max_expected_faults"] = auto_max_expected_faults
        return self.run_noisy(error_model, shots, **common)


def run_noisy_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    error_model: PauliErrorModel,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    strategy: str = "independent",
    max_branches: int | None = _AUTO_MAX_BRANCHES,
    auto_max_expected_faults: float = _AUTO_MAX_EXPECTED_FAULTS,
    importance_sampling=None,
    max_branch_factor: int | None = None,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
    retain: str = "all",
    _shot_ids=None,
    _progress=None,
) -> NoisyShotResult | CoalescedTrajectoryResult:
    """Build and replay independent noisy trajectories with either MPS optimizer.

    ``optimizer_factory`` must create a fresh :class:`MpsOptimizer` or
    :class:`MpsStabOptimizer` for each trajectory. For example::

        result = run_noisy_shots(
            lambda: pepsy.MpsStabOptimizer(8, chi=32), gates,
            PauliErrorModel.depolarizing(1e-3), shots=1_000, seed=7,
        )

    The result retains the final optimizers, concrete streams, and sampled
    faults. ``run_kwargs`` is forwarded unchanged to each optimizer's ``run``.

    Set ``strategy="coalesced"`` to return the exact count-coalesced result
    from :func:`run_coalesced_noisy_shots`. ``strategy="auto"`` chooses that
    representation only when the expected per-shot fault count ``lambda`` is
    at most ``auto_max_expected_faults`` (default ``0.1``) and the stream has
    no unforced mid-circuit control. If live leaves exceed ``max_branches``
    (default ``128``), it restarts as independent trajectories; no sample is
    dropped or approximated. The default stays ``"independent"`` for full
    backward compatibility.
    """
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer per shot.")
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")

    strategy = _validate_strategy(strategy)
    retain = _validate_retain(retain)
    max_branches = _validate_max_branches(max_branches)
    max_branch_factor = _validate_max_branch_factor(max_branch_factor)
    parallel_workers = _validate_parallel_workers(parallel_workers)
    parallel_backend = _validate_parallel_backend(parallel_backend)
    if _shot_ids is not None and strategy != "independent":
        raise ValueError("shot_ids are supported only for independent replay.")
    entries = _as_entries(gates)
    plan = compile_trajectory_stream(entries)
    auto_max_expected_faults = float(auto_max_expected_faults)
    if (
        not np.isfinite(auto_max_expected_faults)
        or auto_max_expected_faults < 0.0
    ):
        raise ValueError("auto_max_expected_faults must be finite and nonnegative.")
    if parallel_workers > 1:
        if strategy == "auto":
            strategy = _resolve_auto_parallel_strategy(
                entries,
                shots,
                error_model=error_model,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                auto_max_expected_faults=auto_max_expected_faults,
            )
            if strategy == "coalesced":
                try:
                    return run_parallel_noisy_shots(
                        optimizer_factory,
                        entries,
                        error_model,
                        shots,
                        seed=seed,
                        run_kwargs=run_kwargs,
                        strategy=strategy,
                        max_branches=max_branches,
                        importance_sampling=importance_sampling,
                        max_branch_factor=max_branch_factor,
                        parallel_workers=parallel_workers,
                        parallel_backend=parallel_backend,
                        retain=retain,
                        _shot_ids=_shot_ids,
                        _progress=_progress,
                    )
                except _CoalescedBranchCapExceeded:
                    strategy = "independent"
        return run_parallel_noisy_shots(
            optimizer_factory,
            entries,
            error_model,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            strategy=strategy,
            max_branches=max_branches,
            importance_sampling=importance_sampling,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
            retain=retain,
            _shot_ids=_shot_ids,
            _progress=_progress,
        )
    if _contains_stochastic_entries(entries):
        raise ValueError(
            "Stream-local stochastic entries require run_trajectory_shots(...) "
            "or run_coalesced_trajectory_shots(...). PauliErrorModel is a "
            "convenience macro for clean deterministic streams."
        )
    if _contains_leakage_entries(entries):
        raise ValueError(
            "Stateful leakage entries require run_trajectory_shots(...). "
            "PauliErrorModel is a convenience macro for clean deterministic streams."
        )
    has_dynamic_conditionals = any(
        _conditional_pauli_support(entry) is not None for entry in entries
    )

    if strategy == "coalesced":
        return run_coalesced_noisy_shots(
            optimizer_factory,
            entries,
            error_model,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            max_branches=max_branches,
            importance_sampling=importance_sampling,
            max_branch_factor=max_branch_factor,
            retain=retain,
        )
    if strategy == "auto" and _auto_prefers_coalescing(
        entries, error_model, auto_max_expected_faults
    ):
        try:
            return run_coalesced_noisy_shots(
                optimizer_factory,
                entries,
                error_model,
                shots,
                seed=seed,
                run_kwargs=run_kwargs,
                max_branches=max_branches,
                importance_sampling=importance_sampling,
                max_branch_factor=max_branch_factor,
                retain=retain,
            )
        except _CoalescedBranchCapExceeded:
            # Restart from fresh optimizers. This changes neither the target
            # distribution nor its independent-trajectory semantics.
            pass

    child_seeds = _trajectory_seed_pairs(seed, shots, shot_ids=_shot_ids)
    optimizers = []
    streams = []
    faults = []
    weights = []
    for child_seed in child_seeds:
        noise_seed, optimizer_seed = child_seed.channel, child_seed.optimizer
        optimizer = optimizer_factory()
        if not hasattr(optimizer, "set_gates") or not hasattr(optimizer, "run"):
            raise TypeError(
                "optimizer_factory must return an optimizer with set_gates(...) and run(...)."
            )
        _seed_trajectory_optimizer(optimizer, optimizer_seed)
        if has_dynamic_conditionals:
            stream, shot_faults, weight = _run_noisy_conditional_shot(
                optimizer,
                entries,
                error_model,
                np.random.default_rng(noise_seed),
                dict(run_kwargs),
                proposal_model=importance_sampling,
            )
        else:
            stream, shot_faults, weight = _sample_gate_stream(
                entries,
                error_model,
                np.random.default_rng(noise_seed),
                proposal_model=importance_sampling,
                return_weight=True,
            )
            # Sampled/library-generated matrices must cross the optimizer
            # backend boundary before strict stream validation, regardless of
            # whether the coefficient state is an MPS or a TTN/STN wrapper.
            replay_stream = _stream_on_optimizer_backend(stream, optimizer)
            optimizer.set_gates(replay_stream)
            optimizer.run(**dict(run_kwargs))
        if retain != "none":
            optimizers.append(optimizer)
            weights.append(weight)
        if retain == "all":
            streams.append(tuple(stream))
            faults.append(shot_faults)

    return NoisyShotResult(
        tuple(optimizers),
        tuple(streams),
        tuple(faults),
        tuple(weights),
        shot_count=int(shots),
        diagnostics=_trajectory_diagnostics(
            plan,
            optimizers,
            shots=int(shots),
            coalesced=False,
        ),
    )


def run_parallel_noisy_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    error_model: PauliErrorModel,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    strategy: str = "independent",
    max_branches: int | None = _AUTO_MAX_BRANCHES,
    max_branch_factor: int | None = None,
    importance_sampling=None,
    parallel_workers: int = 2,
    parallel_backend: str = "thread",
    retain: str = "all",
    _shot_ids=None,
    _progress=None,
) -> NoisyShotResult | CoalescedTrajectoryResult:
    """Run noisy shots in deterministic parallel batches.

    Independent shots receive fixed child seed streams before any worker is
    started, so changing the worker count does not change a shot's random
    stream. Coalesced execution parallelizes independent live leaves while
    retaining one deterministic branch-splitting RNG. ``parallel_backend='gpu'``
    uses threads in the current process; the optimizer factory must select the
    intended GPU backend/device.
    """
    strategy = _validate_strategy(strategy)
    retain = _validate_retain(retain)
    workers = _validate_parallel_workers(parallel_workers)
    backend = _validate_parallel_backend(parallel_backend)
    if strategy == "auto":
        raise ValueError(
            "parallel noisy execution needs strategy='independent' or 'coalesced'."
        )
    if _shot_ids is not None and strategy != "independent":
        raise ValueError("shot_ids are supported only for independent replay.")
    if strategy == "coalesced":
        return run_coalesced_noisy_shots(
            optimizer_factory,
            gates,
            error_model,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            importance_sampling=importance_sampling,
            parallel_workers=workers,
            parallel_backend=backend,
            retain=retain,
        )

    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    entries = _as_entries(gates)
    plan = compile_trajectory_stream(entries)
    child_seeds = _trajectory_seed_pairs(seed, shots, shot_ids=_shot_ids)

    def run_one(child_seed):
        child = (
            child_seed
            if isinstance(child_seed, _TrajectorySeedPair)
            else _TrajectorySeedPair(*child_seed.spawn(2))
        )
        return run_noisy_shots(
            optimizer_factory,
            entries,
            error_model,
            1,
            seed=child,
            run_kwargs=run_kwargs,
            strategy="independent",
            importance_sampling=importance_sampling,
            retain=retain,
        )

    results = _parallel_map_ordered(
        run_one,
        child_seeds,
        workers,
        backend,
        progress=_progress,
    )
    if retain == "none":
        return NoisyShotResult(
            (),
            (),
            (),
            (),
            shot_count=int(shots),
            diagnostics=_trajectory_diagnostics(
                plan, (), shots=int(shots), coalesced=False
            ),
        )
    return NoisyShotResult(
        tuple(result.optimizers[0] for result in results),
        tuple(result.gate_streams[0] for result in results) if retain == "all" else (),
        tuple(result.faults[0] for result in results) if retain == "all" else (),
        tuple(result.weights[0] for result in results),
        shot_count=int(shots),
        diagnostics=_trajectory_diagnostics(
            plan,
            tuple(result.optimizers[0] for result in results),
            shots=int(shots),
            coalesced=False,
        ),
    )


# ---------------------------------------------------------------------------
# User-defined quantum-trajectory channels in ordinary gate streams.
# ---------------------------------------------------------------------------
def _trajectory_entries(gates) -> list[object]:
    """Normalize a stream that may itself be a single trajectory event."""
    if isinstance(gates, TrajectoryStreamPlan):
        return list(gates.entries)
    if isinstance(gates, TrajectoryEvent):
        return [gates]
    entries = []
    for entry in _as_entries(gates):
        trajectory_event = _trajectory_event_from_stochastic_entry(entry)
        entries.append(entry if trajectory_event is None else trajectory_event)
    return entries


def compile_trajectory_stream(gates) -> TrajectoryStreamPlan:
    """Compile a trajectory stream once for repeated shot replay.

    The returned plan is immutable and safe to share between optimizer
    factories. It deliberately does not convert matrices to a device backend;
    that conversion depends on the live optimizer and is cached there.
    """
    if isinstance(gates, TrajectoryStreamPlan):
        return gates
    entries = tuple(_trajectory_entries(gates))
    trajectory_indices = tuple(
        index for index, entry in enumerate(entries) if isinstance(entry, TrajectoryEvent)
    )
    leakage_indices = tuple(
        index for index, entry in enumerate(entries)
        if _leakage_event_parts(entry) is not None
    )
    control_indices = tuple(
        index for index, entry in enumerate(entries)
        if MpsOptimizer.control_event_parts(entry) is not None
    )
    boundaries = set(trajectory_indices) | set(leakage_indices) | set(control_indices)
    ordinary_segments = []
    start = None
    for index in range(len(entries) + 1):
        if index < len(entries) and index not in boundaries:
            if start is None:
                start = index
            continue
        if start is not None:
            ordinary_segments.append((start, index))
            start = None
    return TrajectoryStreamPlan(
        entries=entries,
        ordinary_segments=tuple(ordinary_segments),
        trajectory_indices=trajectory_indices,
        control_indices=control_indices,
        leakage_indices=leakage_indices,
    )


def _entry_from_trajectory_outcome(outcome: TrajectoryOutcome | Any, where):
    """Turn a selected local outcome into a normal bundled matrix gate."""
    support = _trajectory_where(where)
    gate = outcome.gate if isinstance(outcome, TrajectoryOutcome) else outcome
    return (gate, support[0] if len(support) == 1 else support)


def _sample_trajectory_mixture(channel: TrajectoryChannel, rng):
    """Choose a state-independent random-unitary outcome."""
    probabilities = np.asarray(
        [outcome.probability for outcome in channel.outcomes], dtype=float
    )
    index = int(rng.choice(len(channel.outcomes), p=probabilities))
    return channel.outcomes[index], float(probabilities[index])


def sample_trajectory_stream(
    gates, *, seed=None, importance_sampling=None
) -> TrajectorySample:
    """Sample fixed random-unitary events in a user-defined gate stream.

    The input is an ordinary gate stream with either :class:`TrajectoryEvent`
    objects or Pepsy stochastic entries such as ``("x_error", p, q)`` inserted
    wherever a local noisy channel should act. Only fixed-mixture events can be
    sampled without a state; a :meth:`TrajectoryChannel.kraus` event, including
    ``("amplitude_damping", gamma, q)``, needs the evolving state and must use
    :func:`run_trajectory_shots` instead.
    """
    rng = np.random.default_rng(seed)
    policy = _coerce_importance_policy(importance_sampling)
    stream = []
    records = []
    weight = 1.0
    for event_index, entry in enumerate(_trajectory_entries(gates)):
        if _leakage_event_parts(entry) is not None:
            raise ValueError(
                "Stateful leakage entries require run_trajectory_shots(...); "
                "they cannot be sampled from a gate stream alone."
            )
        if not isinstance(entry, TrajectoryEvent):
            stream.append(entry)
            continue
        if entry.channel.mode != "mixture":
            raise ValueError(
                "State-dependent Kraus channels require run_trajectory_shots(...); "
                "they cannot be sampled from a gate stream alone."
            )
        target = np.asarray(
            [outcome.probability for outcome in entry.channel.outcomes], dtype=float
        )
        target, proposal, ratios = _importance_distribution(
            policy, event_index, entry.channel.outcomes, target
        )
        index = int(rng.choice(len(entry.channel.outcomes), p=proposal))
        outcome = entry.channel.outcomes[index]
        probability = float(target[index])
        proposal_probability = float(proposal[index]) if policy is not None else None
        ratio = float(ratios[index])
        weight *= ratio
        stream.append(_entry_from_trajectory_outcome(outcome, entry.where))
        records.append(
            TrajectoryRecord(
                event_index,
                entry.where,
                outcome.label,
                probability,
                proposal_probability,
                ratio,
            )
        )
    return TrajectorySample(tuple(stream), tuple(records), float(weight))


def _check_trajectory_optimizer(optimizer):
    if not hasattr(optimizer, "set_gates") or not hasattr(optimizer, "run"):
        raise TypeError(
            "optimizer_factory must return an optimizer with set_gates(...) and run(...)."
        )


def _seed_trajectory_optimizer(optimizer, seed):
    """Seed every backend RNG used by a trajectory optimizer.

    The noise runner owns the per-shot ``SeedSequence``.  Optimizer-native
    measurement/reset sampling must consume a separate child stream so a
    repeated trajectory run is reproducible without coupling channel draws to
    measurement draws.
    """
    rng = np.random.default_rng(seed)
    if hasattr(optimizer, "_rng"):
        optimizer._rng = rng
    if hasattr(optimizer, "rng"):
        optimizer.rng = rng


def _normalize_optimizer_measurement(raw):
    """Normalize Pepsy MPS, STN, and TTN measurement record shapes."""
    if isinstance(raw, TrajectoryMeasurementRecord):
        return raw.pauli, raw.where, raw.outcome, raw.probability
    pauli = getattr(raw, "pauli", None)
    where = getattr(raw, "where", None)
    outcome = getattr(raw, "outcome", None)
    probability = getattr(raw, "probability", None)
    if pauli is None and isinstance(raw, Mapping):
        pauli = raw.get("pauli")
        where = raw.get("where")
        outcome = raw.get("outcome")
        probability = raw.get("probability")
    if pauli is None:
        try:
            pauli, where, outcome = raw[:3]
            probability = raw[3] if len(raw) > 3 else None
        except (TypeError, IndexError, ValueError) as exc:
            raise TypeError(f"Unsupported optimizer measurement record {raw!r}.") from exc
    if isinstance(where, Integral):
        where = (int(where),)
    else:
        where = tuple(int(site) for site in where)
    probability = None if probability is None else float(probability)
    return str(pauli), where, int(outcome), probability


def _optimizer_measurement_records(optimizer, *, event_index=None):
    """Return normalized measurement records already retained by an optimizer."""
    records = []
    for raw in getattr(optimizer, "measurements", ()):
        pauli, where, outcome, probability = _normalize_optimizer_measurement(raw)
        records.append(
            TrajectoryMeasurementRecord(
                event_index=event_index,
                pauli=pauli,
                where=where,
                outcome=outcome,
                probability=probability,
            )
        )
    return tuple(records)


def _resolve_stim_annotations(plan, optimizer):
    """Resolve compiled ``rec[...]`` annotations against optimizer results."""
    measurements = _optimizer_measurement_records(optimizer)
    bits = tuple(int(record.outcome < 0) for record in measurements)

    def parity(annotation):
        value = 0
        for offset, inverted in annotation.rec_targets:
            index = int(annotation.measurement_count) + int(offset)
            if index < 0 or index >= len(bits):
                raise ValueError(
                    f"Stim annotation at instruction {annotation.instruction_index} "
                    f"references measurement {index}, but replay produced {len(bits)}."
                )
            bit = bits[index]
            value ^= bit ^ int(bool(inverted))
        return bool(value)

    syndromes = tuple(
        StimSyndromeRecord(
            instruction_index=annotation.instruction_index,
            detector_index=annotation.detector_index,
            value=parity(annotation),
            coordinates=annotation.coordinates,
        )
        for annotation in plan.detectors
    )
    observables = tuple(
        StimObservableRecord(
            instruction_index=annotation.instruction_index,
            observable_index=annotation.observable_index,
            value=parity(annotation),
        )
        for annotation in plan.observables
    )
    return syndromes, observables


def _measurement_metadata(indexed_entries):
    """Return ``(event_index, reset)`` metadata for recorded control outcomes."""
    metadata = []
    for event_index, entry in indexed_entries:
        parts = MpsOptimizer.control_event_parts(entry)
        if parts is None:
            continue
        name, _payload, where = parts
        if name == "measure":
            metadata.append((int(event_index), False))
        elif name == "measure_reset":
            metadata.extend((int(event_index), True) for _ in where)
    return metadata


def _capture_optimizer_measurements(
    optimizer,
    start,
    metadata,
    destination,
):
    """Capture newly appended native records with stream event metadata."""
    native = list(getattr(optimizer, "measurements", ()))
    for offset, raw in enumerate(native[int(start):]):
        pauli, where, outcome, probability = _normalize_optimizer_measurement(raw)
        event_index, reset = (
            metadata[offset] if offset < len(metadata) else (None, False)
        )
        destination.append(
            TrajectoryMeasurementRecord(
                event_index=event_index,
                pauli=pauli,
                where=where,
                outcome=outcome,
                probability=probability,
                reset=reset,
            )
        )


def _is_stabilizer_trajectory_optimizer(optimizer) -> bool:
    """Recognize MPS- or tree-backed STN optimizers without importing them."""
    if _is_tree_stabilizer_trajectory_optimizer(optimizer):
        return True
    return _is_mps_stabilizer_trajectory_optimizer(optimizer)


def _is_tree_stabilizer_trajectory_optimizer(optimizer) -> bool:
    """Recognize TreeStabOptimizer through its lightweight protocol marker."""
    return bool(getattr(optimizer, "_is_tree_stabilizer_trajectory_optimizer", False))


def _is_mps_stabilizer_trajectory_optimizer(optimizer) -> bool:
    """Recognize the chain STN optimizer without importing it at setup time."""
    required = all(
        callable(getattr(optimizer, attr, None))
        for attr in (
            "copy",
            "_mps_site",
            "_canonize_p",
            "_renorm_p_at",
            "_make_norm_event",
            "_commit_norm_event",
        )
    )
    return required and callable(getattr(optimizer, "_reset_infidelity", None))


def _trajectory_norm_squared(optimizer) -> float:
    """Read the represented state norm through the optimizer's public API."""
    norm = getattr(optimizer, "norm", None)
    if callable(norm):
        value = _trajectory_real_scalar(norm(), label="trajectory state norm")
    else:
        p = getattr(optimizer, "p", None)
        if p is None or not hasattr(p, "norm"):
            raise TypeError("trajectory optimizer must expose a state norm through norm() or p.norm().")
        value = _trajectory_real_scalar(p.norm(), label="trajectory state norm")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Cannot sample a trajectory channel from a zero- or invalid-norm state.")
    return value * value


def _mps_local_kraus_norm_squared(optimizer, matrix, where):
    """Evaluate ``<psi|K^dagger K|psi>`` without copying the MPS.

    Quimb's environment contraction works for canonical and non-canonical open
    MPS states and returns the normalized local expectation directly. Returning
    ``None`` keeps the conservative copied-state path available for custom MPS
    lookalikes and backends that cannot contract the generated dense operator.
    """
    p = getattr(optimizer, "p", None)
    compute = getattr(p, "compute_local_expectation", None)
    if not callable(compute):
        return None
    try:
        gram = matrix.conj().T @ matrix
        support = tuple(int(site) for site in where)
        # Quimb's environment helper has a known length-one edge case. The
        # represented state is only a two-component vector there, so evaluate
        # the local Gram form directly instead of copying/applying a candidate
        # MPS branch.
        if len(support) == 1 and int(getattr(p, "L", 0)) == 1:
            dense = getattr(p, "to_dense", None)
            if callable(dense):
                vector = np.asarray(dense(), dtype=complex).reshape(-1)
                gram_numpy = np.asarray(gram, dtype=complex)
                denominator = float(np.vdot(vector, vector).real)
                if denominator > 0.0:
                    value = np.vdot(vector, gram_numpy @ vector) / denominator
                    value = _trajectory_real_scalar(
                        value, label="local Kraus probability"
                    )
                    if not np.isfinite(value) or value < -1e-10:
                        raise ValueError(
                            "local Kraus contraction produced an invalid probability."
                        )
                    return max(0.0, value)
        value = compute(
            # ``compute_local_expectation`` accepts scalar keys in some
            # Quimb releases, but the environment implementation used by
            # Pepsy iterates over ``where``. Keep the support tuple even for
            # one-site channels so the fast path works for every MPS length.
            {support: gram},
            normalized=True,
            return_all=True,
            method="envs",
        )
        if isinstance(value, Mapping):
            value = next(iter(value.values()))
        value = _trajectory_real_scalar(value, label="local Kraus probability")
        if not np.isfinite(value) or value < -1e-10:
            raise ValueError("local Kraus contraction produced an invalid probability.")
        return max(0.0, value)
    except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError):
        return None


def _mps_outcome_norm_squared(optimizer, matrix, where) -> float:
    """Evaluate one Kraus branch without mutating an MPS or exact leaf."""
    p = getattr(optimizer, "p", None)
    apply_gate = getattr(optimizer, "_apply_gate", None)
    remap = getattr(optimizer, "_logical_to_physical_where", None)
    if p is None or not callable(apply_gate) or not callable(remap):
        raise TypeError(
            "State-dependent trajectory channels require MpsOptimizer or MpsStabOptimizer."
        )
    matrix = _to_trajectory_backend(matrix, optimizer)
    physical_where = tuple(remap(where))
    local_probability = _mps_local_kraus_norm_squared(
        optimizer, matrix, physical_where
    )
    if local_probability is not None:
        return local_probability * _trajectory_norm_squared(optimizer)
    _record_kraus_probability_diagnostic(optimizer, used_copy_fallback=True)
    candidate = apply_gate(
        p.copy(),
        matrix,
        physical_where[0] if len(physical_where) == 1 else physical_where,
        contract=True,
    )
    value = _trajectory_real_scalar(candidate.norm(), label="Kraus branch norm")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Kraus branch produced an invalid MPS norm.")
    return value * value


def _tree_outcome_norm_squared(optimizer, matrix, where) -> float:
    """Evaluate one Kraus branch on a copied ordinary TTN without mutation."""
    copy = getattr(optimizer, "copy", None)
    if not callable(copy):
        raise TypeError(
            "State-dependent trajectory channels require an optimizer with copy()."
        )
    candidate = copy()
    apply_gate = getattr(candidate, "apply_gate", None)
    if not callable(apply_gate):
        raise TypeError(
            "State-dependent trajectory channels require TreeOptimizer "
            "apply_gate(...)."
        )
    matrix = _to_trajectory_backend(matrix, candidate)
    support = _trajectory_where(where)
    target = support[0] if len(support) == 1 else support
    apply_gate(matrix, target)
    value = _trajectory_real_scalar(candidate.norm(), label="Kraus branch norm")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Kraus branch produced an invalid TTN norm.")
    return value * value


def _stn_outcome_norm_squared(optimizer, matrix, where) -> float:
    """Evaluate one physical Kraus branch in an independent STN frame copy."""
    candidate = optimizer.copy()
    candidate.set_gates([_entry_from_trajectory_outcome(matrix, where)]).run()
    value = _trajectory_real_scalar(candidate.norm(), label="Kraus branch norm")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("Kraus branch produced an invalid STN norm.")
    return value * value


def _to_trajectory_backend(matrix, optimizer):
    """Convert generated NumPy matrices to the live MPS or TTN backend."""
    cache_plan = getattr(optimizer, "_backend_cache_plan", None)
    if isinstance(optimizer, MpsOptimizer) and cache_plan is not None:
        cache_key = (
            "trajectory-gate",
            id(matrix),
            repr(getattr(optimizer, "backend", None)),
            repr(getattr(optimizer, "backend_dtype", None)),
            repr(getattr(optimizer, "backend_device", None)),
        )
        return cache_plan.get_or_create_backend_payload(
            cache_key,
            matrix,
            lambda: _to_trajectory_backend_uncached(matrix, optimizer),
        )
    return _to_trajectory_backend_uncached(matrix, optimizer)


def _to_trajectory_backend_uncached(matrix, optimizer):
    """Convert one generated matrix without consulting the shared cache."""
    converter = getattr(optimizer, "_to_state_backend", None)
    if callable(converter):
        return converter(matrix)
    converter = getattr(optimizer, "_as_state_backend", None)
    if callable(converter):
        return converter(matrix, warn=False)
    return matrix


def _kraus_probabilities(optimizer, channel: TrajectoryChannel, where) -> np.ndarray:
    """Compute normalized state-dependent probabilities for a local channel."""
    base_norm_squared = _trajectory_norm_squared(optimizer)
    if _is_stabilizer_trajectory_optimizer(optimizer):
        # Both STN frontends already expose the exact local Gram-norm path
        # used by their non-unitary dense-gate implementation.  Evaluating
        # ``<psi|K^dagger K|psi>`` through Pauli expectations avoids making a
        # full optimizer copy for every Kraus outcome.  The selected K branch
        # is still replayed once below, where the trajectory state is updated.
        target_norm = getattr(optimizer, "_dense_gate_target_norm", None)
        if callable(target_norm):
            branch_norm_squared = np.asarray(
                [
                    float(target_norm(outcome.gate, where)) ** 2
                    for outcome in channel.outcomes
                ],
                dtype=float,
            )
        else:  # pragma: no cover - compatibility with external STN lookalikes
            branch_norm_squared = np.asarray(
                [
                    _stn_outcome_norm_squared(optimizer, outcome.gate, where)
                    for outcome in channel.outcomes
                ],
                dtype=float,
            )
    elif callable(getattr(optimizer, "apply_gate", None)):
        branch_norm_squared = np.asarray(
            [
                _tree_outcome_norm_squared(optimizer, outcome.gate, where)
                for outcome in channel.outcomes
            ],
            dtype=float,
        )
    else:
        branch_norm_squared = np.asarray(
            [
                _mps_outcome_norm_squared(optimizer, outcome.gate, where)
                for outcome in channel.outcomes
            ],
            dtype=float,
        )
    probabilities = branch_norm_squared / base_norm_squared
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-10):
        raise ValueError("Kraus channel produced invalid trajectory probabilities.")
    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("Kraus channel has no nonzero trajectory outcome for this state.")
    _record_kraus_probability_diagnostic(optimizer, residual=total - 1.0)
    # A complete channel sums to one. Normalize the tiny residual caused by
    # finite-MPS truncation so the shot sampler remains a proper distribution.
    return probabilities / total


def _run_trajectory_entries(
    optimizer,
    entries,
    run_kwargs,
    *,
    non_unitary=False,
    magic_context=None,
):
    """Run one contiguous ordinary-gate segment on its optimizer backend."""
    if not entries:
        return
    if len(entries) > 1 and any(
        MpsOptimizer.control_event_parts(entry) is not None for entry in entries
    ):
        # The general MPS optimizer accepts a control event as a complete
        # stream, but deliberately does not mix it with bundled ordinary gate
        # entries in one ``set_gates`` call.  Keep ordinary runs batched while
        # isolating each control event; STN and Tree backends follow the same
        # segmentation, so measurement records stay aligned across backends.
        pending = []

        def flush_controls_pending():
            nonlocal pending
            if pending:
                _run_trajectory_entries(
                    optimizer,
                    pending,
                    run_kwargs,
                    non_unitary=non_unitary,
                    magic_context=magic_context,
                )
                pending = []

        for entry in entries:
            if MpsOptimizer.control_event_parts(entry) is None:
                pending.append(entry)
            else:
                flush_controls_pending()
                _run_trajectory_entries(
                    optimizer,
                    [entry],
                    run_kwargs,
                    non_unitary=non_unitary,
                    magic_context=magic_context,
                )
        flush_controls_pending()
        return
    if magic_context is not None:
        pending = []

        def flush_magic_pending():
            nonlocal pending
            if pending:
                _run_trajectory_entries(
                    optimizer,
                    pending,
                    run_kwargs,
                    non_unitary=non_unitary,
                )
                pending = []

        for entry in entries:
            if _magic_entry_spec(optimizer, entry) is None:
                pending.append(entry)
                continue
            flush_magic_pending()
            _apply_magic_entry(optimizer, entry, magic_context)
        flush_magic_pending()
        return
    # A one-entry tuple is itself a valid bundled gate, while optimizers expect
    # a *stream* to distinguish it from that single gate. Keep the outer list
    # explicit for branch steps containing exactly one selected outcome.
    shared_cache = bool(getattr(optimizer, "_shared_backend_cache", False))
    shared_plan = getattr(optimizer, "_backend_cache_plan", None)
    replay_entries = list(_stream_on_optimizer_backend(entries, optimizer))
    optimizer.set_gates(replay_entries)
    # ``set_gates`` normally starts a new user-owned stream plan. Shot-created
    # MPS optimizers instead share the constructor plan's backend payload cache
    # across all branches, so restore that association after replacing the
    # short replay queue.
    if shared_cache and shared_plan is not None:
        optimizer._shared_backend_cache = True
        optimizer._backend_cache_plan = shared_plan
    kwargs = dict(run_kwargs)
    if non_unitary and not _is_stabilizer_trajectory_optimizer(optimizer):
        kwargs["non_unitary"] = True
        kwargs["normalize_every"] = False
        kwargs["normalize_final"] = False
        if isinstance(optimizer, MpsOptimizer):
            kwargs["_trajectory_non_unitary"] = True
    optimizer.run(**kwargs)


def _magic_entry_spec(optimizer, entry):
    """Return an injectable ``(data, angle)`` pair when the backend supports it."""
    classifier = getattr(type(optimizer), "_injectable_rz", None)
    if not callable(classifier):
        return None
    return classifier(entry)


def _new_magic_context(optimizer, ancillas, *, recycle=True, reset_ancillas=True):
    """Validate and initialize incremental immediate-injection state."""
    validate = getattr(optimizer, "_validate_magic_ancilla_pool", None)
    if not callable(validate):
        raise TypeError(
            "magic injection requires MpsStabOptimizer or TreeStabOptimizer "
            "with a validated ancilla pool."
        )
    try:
        pool = validate(ancillas, require_nonempty=True)
    except TypeError:
        # TreeStab keeps the non-empty default in its signature, while the MPS
        # frontend exposes the keyword explicitly for deferred injection.
        pool = validate(ancillas)
    assert_clean = getattr(optimizer, "_assert_magic_ancillas_clean", None)
    if callable(assert_clean):
        assert_clean(pool)
    return {
        "pool": tuple(pool),
        "dirty": {int(ancilla): False for ancilla in pool},
        "recycle": bool(recycle),
        "reset_ancillas": bool(reset_ancillas),
    }


def _apply_magic_entry(optimizer, entry, context):
    """Apply one injectable rotation through a prepared/recycled magic ancilla."""
    spec = _magic_entry_spec(optimizer, entry)
    if spec is None:
        return False
    data, angle = spec
    pool = context["pool"]
    dirty = context["dirty"]
    clean = [ancilla for ancilla in pool if not dirty[ancilla]]
    if clean:
        nearest = getattr(optimizer, "_nearest_magic_ancilla", None)
        ancilla = nearest(clean, data) if callable(nearest) else clean[0]
    elif context["recycle"]:
        nearest = getattr(optimizer, "_nearest_magic_ancilla", None)
        ancilla = nearest(pool, data) if callable(nearest) else pool[0]
        optimizer.reset(ancilla)
        dirty[ancilla] = False
    else:
        raise RuntimeError(
            "magic-ancilla pool exhausted (recycle=False); reserve more ancillas."
        )
    optimizer.prepare_magic(ancilla, angle=angle)
    optimizer.inject_rz(data, ancilla, angle)
    dirty[ancilla] = True
    return True


def _finish_magic_context(optimizer, context):
    """Reset dirty immediate-injection ancillas at the end of a trajectory."""
    if not context["reset_ancillas"]:
        return
    for ancilla, dirty in context["dirty"].items():
        if dirty:
            optimizer.reset(ancilla)
            context["dirty"][ancilla] = False


def _normalize_trajectory_branch(optimizer, where, *, norm_event=None):
    """Normalize a selected Kraus branch while keeping MPS metadata valid."""
    if _is_tree_stabilizer_trajectory_optimizer(optimizer):
        normalize = getattr(optimizer, "normalize", None)
        if not callable(normalize):  # pragma: no cover - protocol guard
            raise TypeError(
                "TreeStab trajectory channels require normalize()."
            )
        normalize()
        return
    if _is_mps_stabilizer_trajectory_optimizer(optimizer):
        site = optimizer._mps_site(_trajectory_where(where)[0])
        optimizer._canonize_p(site)
        projected_norm = optimizer._renorm_p_at(site)
        # A selected Kraus outcome is a normalized quantum-trajectory branch:
        # close the preceding unitary segment without counting its Born weight
        # as compression loss, then establish the new unit-norm baseline.
        optimizer._reset_infidelity()
        optimizer._commit_norm_event(norm_event, projected_norm=projected_norm)
        return
    normalize = getattr(optimizer, "normalize", None)
    if not callable(normalize):
        raise TypeError(
            "State-dependent trajectory channels require an optimizer with normalize()."
        )
    if isinstance(norm_event, dict) and "input_norm" in norm_event:
        probability = float(norm_event["branch_probability"])
        projected_norm = optimizer._real_float(optimizer.p.norm())
        optimizer._record_norm_event(
            "trajectory_kraus",
            expected_norm=float(norm_event["input_norm"]) * np.sqrt(probability),
            observed_norm=projected_norm,
            where=_trajectory_where(where),
            branch_probability=probability,
            physical_boundary=True,
            renormalized=True,
        )
    normalize()
    # MpsOptimizer stores removed scale in ``p.exponent`` so norm diagnostics
    # see the represented non-unitary norm. A quantum-trajectory branch is
    # physically renormalized, therefore clear that bookkeeping scale.
    p = optimizer.p
    if hasattr(p, "exponent"):
        p.exponent = 0.0


@dataclass
class _LeakageState:
    """Classical per-shot leakage state carried outside the qubit MPS."""

    leaked: set[int] = field(default_factory=set)
    leak2depolar: bool = False


def _resolve_trajectory_conditional(optimizer, entry):
    """Return the selected concrete action, or None for a false predicate."""
    while True:
        parts = MpsOptimizer.control_event_parts(entry)
        if parts is None or parts[0] != "conditional":
            return entry
        _name, payload, _where = parts
        record_index, expected = _resolve_conditional(
            payload, len(getattr(optimizer, "measurements", ()))
        )
        record = optimizer.measurements[record_index]
        outcome = int(getattr(record, "outcome", record[2]))
        if int(outcome < 0) != expected:
            return None
        entry = payload["action"]


def _apply_trajectory_cap(optimizer, entry, where, state, run_kwargs, shot_stream):
    """Commit a structural cap before remapping classical leakage labels."""
    _run_trajectory_entries(optimizer, (entry,), run_kwargs)
    # Site labels after a cap refer to the shortened logical chain, including
    # in perm mode. A cap removes even a leaked site's placeholder tensor.
    # Update only after successful replay; failed caps retain the old labels.
    (removed,) = where
    state.leaked = {
        site - (site > removed) for site in state.leaked if site != removed
    }
    shot_stream.append(entry)


def _single_leakage_site(where) -> int:
    where = _trajectory_where(where)
    if len(where) != 1:
        raise ValueError("leakage entries act on exactly one qubit.")
    return int(where[0])


def _reset_zero_entry(site: int) -> tuple[str, int]:
    return ("reset", int(site))


def _pauli_gate_entry(axis: str, site: int) -> tuple[object, int]:
    return (_PAULI_MATRICES[str(axis).upper()], int(site))


def _append_optimizer_measurement(optimizer, pauli, where, outcome, probability=1.0):
    """Append a manually forced leakage measurement in the local record format."""
    measurements = getattr(optimizer, "measurements", None)
    if not isinstance(measurements, list):
        return
    where = tuple(int(site) for site in where)
    outcome = int(outcome)
    if _is_stabilizer_trajectory_optimizer(optimizer):
        record_where = where[0] if len(where) == 1 else where
        try:
            from .stabilizer_tn import MeasurementRecord  # pylint: disable=import-outside-toplevel

            measurements.append(MeasurementRecord(str(pauli), record_where, outcome))
        except Exception:  # pragma: no cover - fallback during unusual import states
            measurements.append((str(pauli), record_where, outcome))
    else:
        measurements.append((str(pauli), where, outcome, float(probability)))


def _last_measurement_bit(optimizer) -> int:
    """Return the computational bit of the optimizer's last Pauli measurement."""
    measurements = getattr(optimizer, "measurements", None)
    if not measurements:
        raise ValueError("measure_leaked did not produce a measurement record.")
    record = measurements[-1]
    outcome = getattr(record, "outcome", record[2])
    return 0 if int(outcome) >= 0 else 1


def _run_leakage_entries(optimizer, entries, run_kwargs, shot_stream):
    if not entries:
        return
    _run_trajectory_entries(optimizer, entries, run_kwargs)
    shot_stream.extend(entries)


def _entry_touches_leaked_qubit(entry, state: _LeakageState) -> bool:
    """Return whether an ordinary gate entry should be suppressed by leakage."""
    if not state.leaked:
        return False
    try:
        support = _event_support(entry)
    except ValueError:
        return False
    return support is not None and any(int(site) in state.leaked for site in support)


def _apply_leakage_reset_control(
    optimizer,
    entry,
    state: _LeakageState,
    run_kwargs,
    shot_stream,
):
    """Replay a reset/prep-style control and clear leakage on its targets."""
    parts = MpsOptimizer.control_event_parts(entry)
    if parts is None or parts[0] != "reset":
        return False
    _name, _payload, where = parts
    _run_leakage_entries(optimizer, (entry,), run_kwargs, shot_stream)
    for site in where:
        state.leaked.discard(int(site))
    return True


def _apply_leaked_measurement_control(
    optimizer,
    entry,
    state: _LeakageState,
    event_index,
    run_kwargs,
    shot_stream,
    leakage_records,
):
    """Handle ordinary measure/measure-reset controls that touch leaked qubits."""
    parts = MpsOptimizer.control_event_parts(entry)
    if parts is None:
        return False
    name, payload, where = parts
    where = tuple(int(site) for site in where)
    touched = [site for site in where if site in state.leaked]
    if name not in {"measure", "measure_reset"} or not touched:
        return False
    if len(where) != 1:
        raise NotImplementedError(
            "leakage-aware multi-qubit measurements are not implemented yet; "
            "use single-site measure or measure_reset entries."
        )
    (site,) = where
    if name == "measure":
        pauli = payload["pauli"]
        _append_optimizer_measurement(optimizer, pauli, where, -1)
        leakage_records.append(
            LeakageRecord(
                event_index=event_index,
                kind="measure",
                site=site,
                initially_leaked=True,
                finally_leaked=True,
                measurement=1,
                branch="leaked",
            )
        )
        return True

    axis = payload["axes"][0]
    _append_optimizer_measurement(optimizer, axis, where, -1)
    reset_entry = _reset_zero_entry(site) if axis == "Z" else ("reset", site, axis)
    state.leaked.discard(site)
    _run_leakage_entries(optimizer, (reset_entry,), run_kwargs, shot_stream)
    leakage_records.append(
        LeakageRecord(
            event_index=event_index,
            kind="measure_reset",
            site=site,
            initially_leaked=True,
            finally_leaked=False,
            measurement=1,
            branch="leaked_reset",
        )
    )
    return True


def _apply_depolarized_leakage(optimizer, site, rng, run_kwargs, shot_stream) -> str:
    """Replace one leakage event by a full one-qubit depolarizing draw."""
    axis = str(rng.choice(("I", "X", "Y", "Z")))
    if axis != "I":
        _run_leakage_entries(
            optimizer,
            (_pauli_gate_entry(axis, site),),
            run_kwargs,
            shot_stream,
        )
    return f"depolarize_{axis}"


def _apply_leakage_event(
    optimizer,
    parts,
    state: _LeakageState,
    rng,
    event_index,
    run_kwargs,
    shot_stream,
    leakage_records,
):
    """Sample and apply one Pepsy-native leakage event."""
    kind, payload, where = parts
    if kind == "leak2depolar":
        state.leak2depolar = bool(payload["enabled"])
        leakage_records.append(
            LeakageRecord(
                event_index=event_index,
                kind="leak2depolar",
                branch="enabled" if state.leak2depolar else "disabled",
            )
        )
        return

    site = _single_leakage_site(where)
    initially_leaked = site in state.leaked

    if kind == "measure_leaked":
        if initially_leaked:
            leakage_records.append(
                LeakageRecord(
                    event_index=event_index,
                    kind="measure_leaked",
                    site=site,
                    initially_leaked=True,
                    finally_leaked=True,
                    measurement=2,
                    branch="leaked",
                )
            )
            return
        entry = ("measure", "Z", site)
        _run_leakage_entries(optimizer, (entry,), run_kwargs, shot_stream)
        bit = _last_measurement_bit(optimizer)
        leakage_records.append(
            LeakageRecord(
                event_index=event_index,
                kind="measure_leaked",
                site=site,
                initially_leaked=False,
                finally_leaked=False,
                measurement=bit,
                branch=f"bit_{bit}",
            )
        )
        return

    probability = float(payload["probability"])
    occurred = bool(rng.random() < probability)
    if kind == "leakage":
        depolarize = bool(payload.get("depolarize", False) or state.leak2depolar)
        branch = "none"
        if occurred:
            if depolarize:
                branch = _apply_depolarized_leakage(
                    optimizer, site, rng, run_kwargs, shot_stream
                )
            elif initially_leaked:
                branch = "already_leaked"
            else:
                _run_leakage_entries(
                    optimizer,
                    (_reset_zero_entry(site),),
                    run_kwargs,
                    shot_stream,
                )
                state.leaked.add(site)
                branch = "leaked"
        leakage_records.append(
            LeakageRecord(
                event_index=event_index,
                kind="leakage_depolarize" if depolarize else "leakage",
                site=site,
                probability=probability,
                occurred=occurred,
                initially_leaked=initially_leaked,
                finally_leaked=site in state.leaked,
                branch=branch,
            )
        )
        return

    if kind == "leakage_return":
        occurred = bool(initially_leaked and rng.random() < probability)
        branch = "not_leaked" if not initially_leaked else "still_leaked"
        if occurred:
            state.leaked.discard(site)
            entries: list[object] = [_reset_zero_entry(site)]
            if bool(rng.random() < 0.5):
                entries.append(_pauli_gate_entry("X", site))
                branch = "return_1"
            else:
                branch = "return_0"
            _run_leakage_entries(optimizer, tuple(entries), run_kwargs, shot_stream)
        leakage_records.append(
            LeakageRecord(
                event_index=event_index,
                kind="leakage_return",
                site=site,
                probability=probability,
                occurred=occurred,
                initially_leaked=initially_leaked,
                finally_leaked=site in state.leaked,
                branch=branch,
            )
        )
        return

    raise AssertionError(f"Unhandled leakage event kind {kind!r}.")


def _apply_trajectory_event(
    optimizer, event, rng, event_index, run_kwargs, *, importance_policy=None
):
    """Sample and apply one channel event, returning its inspectable record."""
    channel = event.channel
    if channel.mode == "mixture":
        target = np.asarray(
            [outcome.probability for outcome in channel.outcomes], dtype=float
        )
        non_unitary = False
    else:
        target = _kraus_probabilities(optimizer, channel, event.where)
        non_unitary = True
    target, proposal, ratios = _importance_distribution(
        importance_policy, event_index, channel.outcomes, target, optimizer
    )
    index = int(rng.choice(len(channel.outcomes), p=proposal))
    outcome = channel.outcomes[index]
    probability = float(target[index])
    proposal_probability = float(proposal[index]) if importance_policy is not None else None
    likelihood_ratio = float(ratios[index])
    if non_unitary and _is_mps_stabilizer_trajectory_optimizer(optimizer):
        norm_event = optimizer._make_norm_event(
            "trajectory_kraus", branch_probability=probability
        )
    elif non_unitary and isinstance(optimizer, MpsOptimizer):
        norm_event = {
            "kind": "trajectory_kraus",
            "branch_probability": float(probability),
            "input_norm": optimizer._real_float(optimizer.p.norm()),
        }
    else:
        norm_event = None
    _run_trajectory_entries(
        optimizer,
        [_entry_from_trajectory_outcome(outcome, event.where)],
        run_kwargs,
        non_unitary=non_unitary,
    )
    if non_unitary:
        _normalize_trajectory_branch(optimizer, event.where, norm_event=norm_event)
    return TrajectoryRecord(
        event_index,
        event.where,
        outcome.label,
        probability,
        proposal_probability,
        likelihood_ratio,
    ), likelihood_ratio


@dataclass
class _CoalescedNode:
    """Mutable construction state for one count-coalesced trajectory leaf."""

    optimizer: Any
    count: int
    weight: float = 1.0
    gate_stream: list[object] = field(default_factory=list)
    records: list[TrajectoryRecord] = field(default_factory=list)
    faults: list[PauliFault] = field(default_factory=list)
    heralds: list[StimHerald] = field(default_factory=list)
    measurements: list[CoalescedMeasurementRecord] = field(default_factory=list)
    leakage_records: list[LeakageRecord] = field(default_factory=list)
    leakage_state: _LeakageState = field(default_factory=_LeakageState)


class _CoalescedBranchCapExceeded(RuntimeError):
    """Internal signal used by auto strategy to restart independently."""


def _check_coalesced_optimizer(optimizer):
    """Validate the additional copy contract needed for branch coalescing."""
    _check_trajectory_optimizer(optimizer)
    if not callable(getattr(optimizer, "copy", None)):
        raise TypeError(
            "coalesced trajectory replay requires an optimizer with copy(); "
            "use MpsOptimizer, TreeOptimizer, or MpsStabOptimizer."
        )


def _copy_coalesced_node(node: _CoalescedNode) -> _CoalescedNode:
    """Copy state only at a genuine nonempty stochastic branch."""
    branch_copy = getattr(node.optimizer, "_copy_for_trajectory_branch", None)
    optimizer = branch_copy() if callable(branch_copy) else node.optimizer.copy()
    if optimizer is node.optimizer:
        raise TypeError("optimizer.copy() must return an independent optimizer state.")
    return _CoalescedNode(
        optimizer=optimizer,
        count=node.count,
        weight=node.weight,
        gate_stream=list(node.gate_stream),
        records=list(node.records),
        faults=list(node.faults),
        heralds=list(node.heralds),
        measurements=list(node.measurements),
        leakage_records=list(node.leakage_records),
        leakage_state=_LeakageState(
            leaked=set(node.leakage_state.leaked),
            leak2depolar=bool(node.leakage_state.leak2depolar),
        ),
    )


def _coalesced_inputs(optimizer_factory, shots, run_kwargs):
    """Validate common public coalesced-runner inputs."""
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")
    return int(shots), dict(run_kwargs)


def _initial_coalesced_nodes(optimizer_factory, shots):
    """Create exactly one ideal-prefix optimizer for a nonempty ensemble."""
    if shots == 0:
        return []
    optimizer = optimizer_factory()
    _check_coalesced_optimizer(optimizer)
    return [_CoalescedNode(optimizer=optimizer, count=shots)]


def _coalesced_probabilities(probabilities, *, context):
    """Normalize a categorical distribution with a useful failure message."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(f"{context} needs at least one branch probability.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"{context} has invalid branch probabilities.")
    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError(f"{context} has no nonzero branch probability.")
    return probabilities / total


def _split_coalesced_nodes(
    nodes,
    outcomes,
    probabilities,
    apply,
    rng,
    *,
    context,
    max_branches=None,
    max_branch_factor=None,
    parallel_workers=1,
    parallel_backend="thread",
):
    """Split count-bearing nodes with exact multinomial branch counts."""
    probabilities = _coalesced_probabilities(probabilities, context=context)
    if len(outcomes) != len(probabilities):
        raise ValueError(f"{context} has mismatched outcomes and probabilities.")
    split = []
    for node in nodes:
        counts = rng.multinomial(node.count, probabilities)
        nonempty = [
            (outcome, float(probability), int(count))
            for outcome, probability, count in zip(outcomes, probabilities, counts)
            if int(count) > 0
        ]
        if max_branches is not None and len(split) + len(nonempty) > max_branches:
            raise _CoalescedBranchCapExceeded(
                f"coalesced trajectory branch cap ({max_branches}) exceeded "
                f"while splitting {context}."
            )
        if max_branch_factor is not None and len(nonempty) > max_branch_factor:
            raise _CoalescedBranchCapExceeded(
                f"coalesced per-event branch budget ({max_branch_factor}) exceeded "
                f"while splitting {context}."
            )
        # Clone every child from the *pre-branch* parent. Applying the first
        # outcome before making later copies would incorrectly include that
        # first outcome in every sibling branch.
        children = [
            node if index == 0 else _copy_coalesced_node(node)
            for index in range(len(nonempty))
        ]
        work = tuple(zip(children, nonempty))

        def apply_child(item):
            child, (outcome, probability, count) = item
            child.count = count
            apply(child, outcome, probability)
            return child

        split.extend(
            _parallel_map_ordered(
                apply_child, work, parallel_workers, parallel_backend
            )
        )
    return split


def _run_coalesced_entries(
    nodes,
    indexed_entries,
    run_kwargs,
    rng,
    *,
    max_branches=None,
    max_branch_factor=None,
    parallel_workers=1,
    parallel_backend="thread",
):
    """Replay ordinary segments once per current node, splitting controls exactly."""
    pending = []

    def flush():
        nonlocal pending
        if not pending:
            return
        entries = tuple(entry for _index, entry in pending)
        def run_node(node):
            active = tuple(
                entry
                for entry in entries
                if not _entry_touches_leaked_qubit(
                    entry, node.leakage_state
                )
            )
            _run_trajectory_entries(node.optimizer, active, run_kwargs)
            node.gate_stream.extend(active)
            return node

        nodes[:] = _parallel_map_ordered(
            run_node, nodes, parallel_workers, parallel_backend
        )
        pending = []

    for event_index, entry in indexed_entries:
        leakage_parts = _leakage_event_parts(entry)
        if leakage_parts is not None:
            flush()
            nodes = _coalesced_leakage_event(
                nodes,
                event_index=event_index,
                parts=leakage_parts,
                run_kwargs=run_kwargs,
                rng=rng,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
            )
            continue
        parts = MpsOptimizer.control_event_parts(entry)
        if parts is not None and parts[0] == "conditional":
            flush()
            selected = []
            for node in nodes:
                action = _resolve_trajectory_conditional(node.optimizer, entry)
                if action is None:
                    selected.append(node)
                else:
                    # Selected controls use the same branching/cap/leakage
                    # path as unconditional controls. Retain the concrete
                    # stream once, not both the action and its wrapper.
                    selected.extend(_run_coalesced_entries(
                        [node], [(event_index, action)], run_kwargs, rng,
                        max_branches=max_branches,
                        max_branch_factor=max_branch_factor,
                        parallel_workers=parallel_workers,
                        parallel_backend=parallel_backend,
                    ))
                if max_branches is not None and len(selected) > max_branches:
                    raise _CoalescedBranchCapExceeded(
                        f"coalesced trajectory branch cap ({max_branches}) exceeded "
                        "during conditional replay."
                    )
            nodes = selected
            continue
        if parts is not None and parts[0] == "cap":
            flush()
            for node in nodes:
                _apply_trajectory_cap(
                    node.optimizer, entry, parts[2], node.leakage_state,
                    run_kwargs, node.gate_stream,
                )
            continue
        if parts is None or parts[0] not in {"measure", "reset", "measure_reset"}:
            pending.append((event_index, entry))
            continue
        flush()
        nodes = _coalesced_control_event(
            nodes,
            event_index,
            parts,
            run_kwargs,
            rng,
            entry=entry,
            absorb_basis=_coalesced_control_absorb_basis(entry, parts[0]),
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
        )
    flush()
    return nodes


def _coalesced_control_absorb_basis(entry, name) -> bool:
    """Preserve the optional STN basis-absorbing control-event flag."""
    if isinstance(entry, Mapping):
        return bool(entry.get("absorb_basis", entry.get("absorb", False)))
    if not isinstance(entry, (tuple, list)):
        return False
    head = str(entry[0]).replace("-", "_").lower()
    if name == "measure":
        return bool(entry[4]) if len(entry) > 4 else False
    if name != "measure_reset":
        return False
    if head in {"mrx", "mry", "mrz"}:
        return bool(entry[3]) if len(entry) > 3 else False
    return bool(entry[4]) if len(entry) > 4 else False


def _coalesced_leakage_measure_leaked(
    nodes,
    *,
    event_index,
    site,
    run_kwargs,
    rng,
    max_branches,
    max_branch_factor,
    parallel_workers,
    parallel_backend,
):
    """Replay ``measure_leaked`` while preserving count-bearing branches."""
    result = []
    for node in nodes:
        state = node.leakage_state
        if site in state.leaked:
            node.leakage_records.append(
                LeakageRecord(
                    event_index=event_index,
                    kind="measure_leaked",
                    site=site,
                    initially_leaked=True,
                    finally_leaked=True,
                    measurement=2,
                    branch="leaked",
                )
            )
            result.append(node)
            continue

        p_plus = _coalesced_measurement_probability(node.optimizer, "Z", (site,))

        def apply(child, outcome, probability):
            outcome = int(outcome)
            entry = ("measure", "Z", site, outcome)
            _run_trajectory_entries(child.optimizer, (entry,), run_kwargs)
            child.gate_stream.append(entry)
            child.measurements.append(
                CoalescedMeasurementRecord(
                    event_index=event_index,
                    pauli="Z",
                    where=(site,),
                    outcome=outcome,
                    probability=float(probability),
                )
            )
            bit = 0 if outcome > 0 else 1
            child.leakage_records.append(
                LeakageRecord(
                    event_index=event_index,
                    kind="measure_leaked",
                    site=site,
                    initially_leaked=False,
                    finally_leaked=False,
                    measurement=bit,
                    branch=f"bit_{bit}",
                )
            )

        result.extend(
            _split_coalesced_nodes(
                [node],
                (+1, -1),
                (p_plus, 1.0 - p_plus),
                apply,
                rng,
                context="leakage measurement",
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
            )
        )
    return result


def _coalesced_leakage_event(
    nodes,
    *,
    event_index,
    parts,
    run_kwargs,
    rng,
    max_branches,
    max_branch_factor,
    parallel_workers,
    parallel_backend,
):
    """Replay one stateful leakage event with exact count coalescing."""
    kind, payload, where = parts
    if kind == "leak2depolar":
        enabled = bool(payload["enabled"])
        for node in nodes:
            node.leakage_state.leak2depolar = enabled
            node.leakage_records.append(
                LeakageRecord(
                    event_index=event_index,
                    kind="leak2depolar",
                    branch="enabled" if enabled else "disabled",
                )
            )
        return nodes

    site = _single_leakage_site(where)
    if kind == "measure_leaked":
        return _coalesced_leakage_measure_leaked(
            nodes,
            event_index=event_index,
            site=site,
            run_kwargs=run_kwargs,
            rng=rng,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
        )

    result = []
    for node in nodes:
        initially_leaked = site in node.leakage_state.leaked
        probability = float(payload["probability"])
        if kind == "leakage":
            depolarize = bool(
                payload.get("depolarize", False)
                or node.leakage_state.leak2depolar
            )
            if depolarize:
                labels = ("none", "I", "X", "Y", "Z")
                probabilities = (
                    1.0 - probability,
                    *(probability / 4.0 for _ in range(4)),
                )
            else:
                labels = ("none", "occurred")
                probabilities = (1.0 - probability, probability)

            def apply(child, label, _branch_probability):
                occurred = label != "none"
                branch = "none"
                if occurred and depolarize:
                    branch = f"depolarize_{label}"
                    if label != "I":
                        _run_leakage_entries(
                            child.optimizer,
                            (_pauli_gate_entry(label, site),),
                            run_kwargs,
                            child.gate_stream,
                        )
                elif occurred:
                    if initially_leaked:
                        branch = "already_leaked"
                    else:
                        _run_leakage_entries(
                            child.optimizer,
                            (_reset_zero_entry(site),),
                            run_kwargs,
                            child.gate_stream,
                        )
                        child.leakage_state.leaked.add(site)
                        branch = "leaked"
                child.leakage_records.append(
                    LeakageRecord(
                        event_index=event_index,
                        kind="leakage_depolarize" if depolarize else "leakage",
                        site=site,
                        probability=probability,
                        occurred=occurred,
                        initially_leaked=initially_leaked,
                        finally_leaked=site in child.leakage_state.leaked,
                        branch=branch,
                    )
                )

        elif kind == "leakage_return":
            if not initially_leaked:
                labels = ("not_leaked",)
                probabilities = (1.0,)
            else:
                labels = ("still_leaked", "return_0", "return_1")
                probabilities = (
                    1.0 - probability,
                    probability / 2.0,
                    probability / 2.0,
                )

            def apply(child, label, _branch_probability):
                occurred = label.startswith("return_")
                branch = label
                if occurred:
                    child.leakage_state.leaked.discard(site)
                    _run_leakage_entries(
                        child.optimizer,
                        (_reset_zero_entry(site),),
                        run_kwargs,
                        child.gate_stream,
                    )
                    if label == "return_1":
                        _run_leakage_entries(
                            child.optimizer,
                            (_pauli_gate_entry("X", site),),
                            run_kwargs,
                            child.gate_stream,
                        )
                child.leakage_records.append(
                    LeakageRecord(
                        event_index=event_index,
                        kind="leakage_return",
                        site=site,
                        probability=probability,
                        occurred=occurred,
                        initially_leaked=initially_leaked,
                        finally_leaked=site in child.leakage_state.leaked,
                        branch=branch,
                    )
                )

        else:  # pragma: no cover - parser guards the event names
            raise AssertionError(f"Unhandled leakage event kind {kind!r}.")

        result.extend(
            _split_coalesced_nodes(
                [node],
                labels,
                probabilities,
                apply,
                rng,
                context=f"leakage {kind}",
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
            )
        )
    return result


def _coalesced_measurement_probability(optimizer, pauli, where) -> float:
    """Compute one Born probability without collapsing the node state."""
    where = tuple(int(site) for site in where)
    expectation = getattr(optimizer, "expectation", None)
    if callable(expectation):
        arg = where[0] if len(where) == 1 else where
        value = expectation(pauli, arg)
    else:
        expectation_pauli = getattr(optimizer, "expectation_pauli", None)
        if callable(expectation_pauli):
            arg = where[0] if len(where) == 1 else where
            value = expectation_pauli(pauli, arg)
            return min(max(0.5 * (1.0 + float(value)), 0.0), 1.0)
        mapped = getattr(optimizer, "_logical_to_physical_where", None)
        state_expectation = getattr(optimizer, "_state_expectation", None)
        if not callable(mapped) or not callable(state_expectation):
            raise TypeError(
                "coalesced measurement branching requires MpsOptimizer, "
                "MpsStabOptimizer, or TreeOptimizer expectation support."
            )
        value = state_expectation(pauli, mapped(where))
    return min(max(0.5 * (1.0 + float(value)), 0.0), 1.0)


def _coalesced_reset_needs_branch(optimizer, where) -> bool:
    """Return whether a pure-state reset can leave distinct remote states.

    A reset of a product qubit has the same post-reset pure state for either
    hidden measurement outcome, so it can use the cheap one-leaf backend path.
    For an entangled target, the hidden measurement result selects different
    conditional states of the rest of the network and must be represented by
    separate coalesced leaves.
    """
    where = tuple(int(site) for site in where)
    if len(where) != 1:
        return True
    site = where[0]
    try:
        values = []
        state_expectation = getattr(optimizer, "_state_expectation", None)
        expectation = getattr(optimizer, "expectation", None)
        for axis in ("X", "Y", "Z"):
            if callable(state_expectation):
                value = state_expectation(axis, (site,))
            elif callable(expectation):
                value = expectation(axis, site)
            else:
                return True
            values.append(float(np.real(value)))
        purity = 0.5 * (1.0 + float(np.dot(values, values)))
        # A pure one-qubit reduced state has purity one. A lower purity means
        # the target is entangled with the rest of the network.
        return purity < 1.0 - 1.0e-7
    except (AttributeError, TypeError, ValueError, RuntimeError):
        # Unknown optimizer protocols should take the safe, branching path.
        return True


def _apply_coalesced_measurement(
    nodes,
    *,
    event_index,
    pauli,
    where,
    forced_outcome,
    measure_reset,
    reset,
    absorb_basis,
    run_kwargs,
    rng,
    max_branches=None,
    max_branch_factor=None,
    parallel_workers=1,
    parallel_backend="thread",
):
    """Branch a Pauli collapse, optionally followed by a reset."""
    where = tuple(int(site) for site in where)

    def apply(node, outcome, probability):
        if measure_reset or reset:
            entry = ("measure_reset", pauli, where[0], int(outcome))
        else:
            entry = ("measure", pauli, where, int(outcome))
        if absorb_basis and not reset:
            entry = (*entry, True)
        _run_trajectory_entries(node.optimizer, (entry,), run_kwargs)
        node.gate_stream.append(entry)
        if reset:
            # A bare reset has no user-visible classical result. The equivalent
            # forced measure-reset stream is used only to make its branch exact.
            measurements = getattr(node.optimizer, "measurements", None)
            if isinstance(measurements, list) and measurements:
                measurements.pop()
        node.measurements.append(
            CoalescedMeasurementRecord(
                event_index=event_index,
                pauli=str(pauli),
                where=where,
                outcome=int(outcome),
                probability=float(probability),
                reset=bool(reset),
            )
        )

    if forced_outcome is not None:
        outcome = 1 if int(forced_outcome) >= 0 else -1
        # Applying the forced event validates an impossible postselection. The
        # recorded value is still the Born probability before that collapse.
        checked = []
        for node in nodes:
            p_plus = _coalesced_measurement_probability(node.optimizer, pauli, where)
            checked.append(p_plus if outcome > 0 else 1.0 - p_plus)
        result = []
        for node, branch_probability in zip(nodes, checked):
            apply(node, outcome, branch_probability)
            result.append(node)
        return result

    # The state can differ between nodes, so each node gets its own binomial
    # draw. This is exactly the result of independent per-shot Born draws.
    result = []
    for node in nodes:
        p_plus = _coalesced_measurement_probability(node.optimizer, pauli, where)
        result.extend(
            _split_coalesced_nodes(
                [node],
                (+1, -1),
                (p_plus, 1.0 - p_plus),
                apply,
                rng,
                context="measurement",
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
            )
        )
    return result


def _coalesced_control_event(
    nodes,
    event_index,
    parts,
    run_kwargs,
    rng,
    *,
    entry=None,
    absorb_basis=False,
    max_branches=None,
    max_branch_factor=None,
    parallel_workers=1,
    parallel_backend="thread",
):
    """Branch an unforced measure/reset event one physical collapse at a time."""
    name, payload, where = parts
    where = tuple(int(site) for site in where)
    if name == "reset":
        if entry is None:  # pragma: no cover - defensive protocol guard
            raise ValueError("coalesced reset replay needs the original entry.")
        # A product-state reset has one post-reset pure state and can stay a
        # single leaf. An entangled target needs hidden-outcome branching: the
        # reset is trace preserving, but its pure-state trajectory is not.
        for axis, site in zip(payload["axes"], where):
            direct = []
            branch = []
            for node in nodes:
                if _coalesced_reset_needs_branch(node.optimizer, (site,)):
                    branch.append(node)
                else:
                    direct.append(node)
            if direct:
                def reset_node(node):
                    reset_entry = (
                        ("reset", site)
                        if axis == "Z"
                        else ("reset", site, axis)
                    )
                    _run_trajectory_entries(
                        node.optimizer, (reset_entry,), run_kwargs
                    )
                    node.gate_stream.append(reset_entry)
                    node.leakage_state.leaked.discard(int(site))
                    return node

                direct = _parallel_map_ordered(
                    reset_node, direct, parallel_workers, parallel_backend
                )
            if branch:
                branch = _apply_coalesced_measurement(
                    branch,
                    event_index=event_index,
                    pauli=axis,
                    where=(site,),
                    forced_outcome=None,
                    measure_reset=False,
                    reset=True,
                    absorb_basis=absorb_basis,
                    run_kwargs=run_kwargs,
                    rng=rng,
                    max_branches=max_branches,
                    max_branch_factor=max_branch_factor,
                    parallel_workers=parallel_workers,
                    parallel_backend=parallel_backend,
                )
            nodes = direct + branch
        return nodes
    if name in {"measure", "measure_reset"}:
        leaked_nodes = []
        normal_nodes = []
        if any(int(site) in node.leakage_state.leaked for node in nodes for site in where):
            if len(where) != 1:
                raise NotImplementedError(
                    "coalesced leakage-aware multi-qubit measurements are not "
                    "implemented; use single-site measure or measure_reset entries."
                )
            site = where[0]
            for node in nodes:
                (leaked_nodes if site in node.leakage_state.leaked else normal_nodes).append(
                    node
                )
            for node in leaked_nodes:
                axis = payload["pauli"] if name == "measure" else payload["axes"][0]
                _append_optimizer_measurement(
                    node.optimizer, axis, (site,), -1, probability=1.0
                )
                node.measurements.append(
                    CoalescedMeasurementRecord(
                        event_index=event_index,
                        pauli=str(axis),
                        where=(site,),
                        outcome=-1,
                        probability=1.0,
                        reset=name == "measure_reset",
                    )
                )
                if name == "measure_reset":
                    node.leakage_state.leaked.discard(site)
                    reset_entry = (
                        ("reset", site)
                        if axis == "Z"
                        else ("reset", site, axis)
                    )
                    _run_leakage_entries(
                        node.optimizer,
                        (reset_entry,),
                        run_kwargs,
                        node.gate_stream,
                    )
                    finally_leaked = False
                    branch = "leaked_reset"
                else:
                    finally_leaked = True
                    branch = "leaked"
                node.leakage_records.append(
                    LeakageRecord(
                        event_index=event_index,
                        kind=name,
                        site=site,
                        initially_leaked=True,
                        finally_leaked=finally_leaked,
                        measurement=1,
                        branch=branch,
                    )
                )
            if not normal_nodes:
                return leaked_nodes
            normal_nodes = _coalesced_control_event(
                normal_nodes,
                event_index,
                parts,
                run_kwargs,
                rng,
                entry=entry,
                absorb_basis=absorb_basis,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
            )
            return leaked_nodes + normal_nodes
    if name == "measure":
        return _apply_coalesced_measurement(
            nodes,
            event_index=event_index,
            pauli=payload["pauli"],
            where=where,
            forced_outcome=payload.get("outcome"),
            measure_reset=False,
            reset=False,
            absorb_basis=absorb_basis,
            run_kwargs=run_kwargs,
            rng=rng,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
        )

    axes = tuple(payload["axes"])
    outcomes = payload.get("outcomes", (None,) * len(where))
    for axis, site, outcome in zip(axes, where, outcomes):
        nodes = _apply_coalesced_measurement(
            nodes,
            event_index=event_index,
            pauli=axis,
            where=(site,),
            forced_outcome=outcome if name == "measure_reset" else None,
            measure_reset=True,
            reset=(name == "reset"),
            absorb_basis=absorb_basis,
            run_kwargs=run_kwargs,
            rng=rng,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
        )
    return nodes


def _coalesced_result(nodes, *, plan=None, retain="all") -> CoalescedTrajectoryResult:
    """Freeze construction nodes into the public memory-efficient result."""
    retain = _validate_retain(retain)
    shot_count = sum(int(node.count) for node in nodes)
    diagnostics = _trajectory_diagnostics(
        plan,
        nodes,
        shots=shot_count,
        coalesced=True,
        max_live_branches=len(nodes),
    )
    if retain == "none":
        return CoalescedTrajectoryResult(
            (),
            plan=plan,
            shot_count=shot_count,
            diagnostics=diagnostics,
        )
    return CoalescedTrajectoryResult(
        tuple(
            CoalescedTrajectoryLeaf(
                optimizer=node.optimizer,
                count=node.count,
                gate_stream=tuple(node.gate_stream) if retain == "all" else (),
                records=tuple(node.records) if retain == "all" else (),
                faults=tuple(node.faults) if retain == "all" else (),
                heralds=tuple(node.heralds) if retain == "all" else (),
                measurements=tuple(node.measurements) if retain == "all" else (),
                leakage_records=(
                    tuple(node.leakage_records) if retain == "all" else ()
                ),
                weight=float(node.weight),
            )
            for node in nodes
        ),
        plan=plan,
        shot_count=shot_count,
        diagnostics=diagnostics,
    )


def sample_coalesced_bits(
    result: CoalescedTrajectoryResult,
    *,
    seed=None,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    shuffle: bool = True,
) -> CoalescedSampleResult:
    """Draw ``leaf.count`` terminal bitstrings from every coalesced leaf.

    Ordinary MPS leaves use :class:`pepsy.sampling.MpsSampler`'s batched native
    path, preserving device-local sampling until the final compact NumPy
    result. STN leaves use :meth:`MpsStabOptimizer.sample_bits`, which is
    already a count-coalesced measurement tree. The function never materializes
    one optimizer per trajectory.

    Parameters
    ----------
    result
        A result returned by one of the ``run_coalesced_*`` functions.
    seed
        Optional reproducible seed. Each leaf receives an independent child
        sequence before optional row shuffling.
    sampler_kwargs
        Optional constructor keywords for :class:`MpsSampler`; ``backend``
        defaults to ``"auto"`` so Torch/CuPy leaf states use their native
        batched sampler.
    shuffle
        Shuffle the final rows to remove the leaf-grouped ordering while
        retaining the corresponding ``leaf_indices`` and probabilities.
    """
    if not isinstance(result, CoalescedTrajectoryResult):
        raise TypeError("result must be a CoalescedTrajectoryResult.")
    if sampler_kwargs is None:
        sampler_kwargs = {}
    elif not isinstance(sampler_kwargs, Mapping):
        raise TypeError("sampler_kwargs must be a mapping or None.")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("shuffle must be a boolean.")
    sampler_kwargs = dict(sampler_kwargs)
    sampler_kwargs.setdefault("backend", "auto")

    leaves = result.leaves
    if not leaves:
        return CoalescedSampleResult(
            configs=np.empty((0, 0), dtype=np.int8),
            leaf_indices=np.empty(0, dtype=np.int64),
            probs=np.empty(0, dtype=float),
        )

    from pepsy.sampling import MpsSampler  # pylint: disable=import-outside-toplevel

    child_seeds = np.random.SeedSequence(seed).spawn(len(leaves))
    configs = []
    probs = []
    leaf_indices = []
    lengths = []
    all_have_probs = True
    for leaf_index, (leaf, child_seed) in enumerate(zip(leaves, child_seeds)):
        count = int(leaf.count)
        child_seed = int(child_seed.generate_state(1, dtype=np.uint64)[0])
        if count < 1:
            raise ValueError("coalesced leaf counts must be positive.")
        optimizer = leaf.optimizer
        if _is_stabilizer_trajectory_optimizer(optimizer):
            batch_configs = np.asarray(
                optimizer.sample_bits(count, seed=child_seed), dtype=np.int8
            )
            all_have_probs = False
        else:
            p = getattr(optimizer, "p", None)
            if p is None or not hasattr(p, "L"):
                raise TypeError(
                    "ordinary coalesced terminal sampling requires an MPS-state "
                    "optimizer; mode='exact' leaves are unsupported."
                )
            batch = MpsSampler(p, **sampler_kwargs).sample_batch(
                count, seed=child_seed, to_numpy=True
            )
            batch_configs = np.asarray(batch.configs, dtype=np.int8)
            remap = getattr(optimizer, "remap_sample", None)
            if callable(remap):
                batch_configs = np.asarray(remap(batch_configs), dtype=np.int8)
            probs.append(np.asarray(batch.probs, dtype=float))
        configs.append(batch_configs)
        leaf_indices.append(np.full(count, leaf_index, dtype=np.int64))
        lengths.append(np.full(count, batch_configs.shape[1], dtype=np.int64))

    max_length = max(batch.shape[1] for batch in configs)
    if all(batch.shape[1] == max_length for batch in configs):
        configs = np.concatenate(configs, axis=0)
    else:
        # Rectangular output remains convenient without inventing measured
        # zeros for removed sites. Avoid padded copies of each leaf batch.
        padded = np.full((sum(batch.shape[0] for batch in configs), max_length),
                         -1, dtype=np.int8)
        offset = 0
        for batch in configs:
            stop = offset + batch.shape[0]
            padded[offset:stop, :batch.shape[1]] = batch
            offset = stop
        configs = padded
    leaf_indices = np.concatenate(leaf_indices, axis=0)
    lengths = np.concatenate(lengths, axis=0)
    probabilities = np.concatenate(probs, axis=0) if all_have_probs else None
    if shuffle and len(configs) > 1:
        permutation = np.random.default_rng(seed).permutation(len(configs))
        configs = configs[permutation]
        leaf_indices = leaf_indices[permutation]
        lengths = lengths[permutation]
        if probabilities is not None:
            probabilities = probabilities[permutation]
    return CoalescedSampleResult(configs, leaf_indices, probabilities, lengths)


def run_trajectory_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    strategy: str = "independent",
    max_branches: int | None = _AUTO_MAX_BRANCHES,
    magic_strategy: str = "direct",
    magic_ancillas=None,
    magic_recycle: bool = True,
    magic_reset_ancillas: bool = True,
    magic_projection_order="middle_out",
    importance_sampling=None,
    max_branch_factor: int | None = None,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
    retain: str = "all",
    _shot_ids=None,
    _progress=None,
) -> TrajectoryShotResult | CoalescedTrajectoryResult:
    """Replay user-defined noisy gate-stream trajectories on MPS or tree optimizers.

    Insert :class:`TrajectoryEvent` objects or Pepsy stochastic entries directly
    into an ordinary gate stream. A ``mixture`` selects a known unitary branch
    by its explicit probability. A ``kraus`` channel evaluates all local branch
    norms on the current MPS, TTN, or TreeStab state, samples the conditional
    probability, applies the chosen branch, and normalizes before evolution
    continues. This includes non-Pauli channels such as
    ``("amplitude_damping", gamma, q)`` without forming a density matrix.

    ``optimizer_factory`` must create a fresh :class:`MpsOptimizer`,
    :class:`TreeOptimizer`, :class:`MpsStabOptimizer`, or
    :class:`TreeStabOptimizer` per shot. Gate segments
    between channel events are batched, so a trajectory does not rebuild an
    optimizer for every gate.
    Set ``strategy="coalesced"`` to share deterministic prefixes and retain one
    optimizer per distinct sampled branch. ``strategy="auto"`` tries coalescing
    and restarts independently if ``max_branches`` would be exceeded.

    ``magic_strategy="immediate"`` applies injectable ``T``/``T†``/grid-aligned
    ``Rz`` entries through a supplied ``magic_ancillas`` pool while noisy and
    measurement events continue to replay normally. ``magic_strategy="deferred"``
    supports fixed-mixture streams by sampling their concrete branches first and
    then using the MAST-style deferred projection runner. State-dependent Kraus
    channels remain on the immediate/direct path because their probabilities
    depend on the evolving state.
    """
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer per shot.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")

    strategy = _validate_strategy(strategy)
    retain = _validate_retain(retain)
    max_branches = _validate_max_branches(max_branches)
    max_branch_factor = _validate_max_branch_factor(max_branch_factor)
    parallel_workers = _validate_parallel_workers(parallel_workers)
    parallel_backend = _validate_parallel_backend(parallel_backend)
    if _shot_ids is not None and strategy != "independent":
        raise ValueError("shot_ids are supported only for independent replay.")
    policy = _coerce_importance_policy(importance_sampling)
    plan = compile_trajectory_stream(gates)
    entries = plan.entries
    if parallel_workers > 1:
        if strategy == "auto":
            strategy = _resolve_auto_parallel_strategy(
                plan,
                shots,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
            )
            if str(magic_strategy).strip().lower().replace("-", "_") != "direct":
                strategy = "independent"
            if strategy == "coalesced":
                try:
                    return run_parallel_trajectory_shots(
                        optimizer_factory,
                        plan.entries,
                        shots,
                        seed=seed,
                        run_kwargs=run_kwargs,
                        strategy=strategy,
                        max_branches=max_branches,
                        magic_strategy=magic_strategy,
                        magic_ancillas=magic_ancillas,
                        magic_recycle=magic_recycle,
                        magic_reset_ancillas=magic_reset_ancillas,
                        magic_projection_order=magic_projection_order,
                        importance_sampling=policy,
                        max_branch_factor=max_branch_factor,
                        parallel_workers=parallel_workers,
                        parallel_backend=parallel_backend,
                        retain=retain,
                        _shot_ids=_shot_ids,
                        _progress=_progress,
                    )
                except _CoalescedBranchCapExceeded:
                    strategy = "independent"
        return run_parallel_trajectory_shots(
            optimizer_factory,
            gates,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            strategy=strategy,
            max_branches=max_branches,
            magic_strategy=magic_strategy,
            magic_ancillas=magic_ancillas,
            magic_recycle=magic_recycle,
            magic_reset_ancillas=magic_reset_ancillas,
            magic_projection_order=magic_projection_order,
            importance_sampling=policy,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
            retain=retain,
            _shot_ids=_shot_ids,
            _progress=_progress,
        )
    magic_strategy = str(magic_strategy).strip().lower().replace("-", "_")
    if magic_strategy not in {"direct", "immediate", "deferred"}:
        raise ValueError(
            "magic_strategy must be 'direct', 'immediate', or 'deferred'."
        )
    has_leakage = plan.has_leakage
    if magic_strategy != "direct" and strategy != "independent":
        raise ValueError(
            "magic_strategy='immediate'/'deferred' requires strategy='independent'; "
            "coalesced magic projection needs an explicit ancilla-branch planner."
        )
    if magic_strategy != "direct" and magic_ancillas is None:
        raise ValueError(
            "magic_ancillas is required when magic_strategy is not 'direct'."
        )
    if magic_strategy == "deferred":
        if has_leakage or any(
            isinstance(entry, TrajectoryEvent) and entry.channel.mode != "mixture"
            for entry in entries
        ):
            raise ValueError(
                "magic_strategy='deferred' currently requires fixed-mixture channels "
                "and no stateful leakage; use magic_strategy='immediate' for Kraus streams."
            )
        optimizers = []
        gate_streams = []
        records = []
        leakage_records = []
        measurement_records = []
        weights = []
        diagnostic_infos = []
        for child_seed in _trajectory_seed_pairs(seed, shots, shot_ids=_shot_ids):
            noise_seed, optimizer_seed = child_seed.channel, child_seed.optimizer
            sample = sample_trajectory_stream(
                entries,
                seed=noise_seed,
                importance_sampling=policy,
            )
            optimizer = optimizer_factory()
            _check_trajectory_optimizer(optimizer)
            _seed_trajectory_optimizer(optimizer, optimizer_seed)
            run_deferred = getattr(optimizer, "run_with_deferred_injection", None)
            if not callable(run_deferred):
                raise TypeError(
                    "magic_strategy='deferred' requires an STN optimizer with "
                    "run_with_deferred_injection(...)."
                )
            run_deferred(
                sample.gate_stream,
                ancillas=magic_ancillas,
                projection_order=magic_projection_order,
                reset_ancillas=magic_reset_ancillas,
                **dict(run_kwargs),
            )
            diagnostic_infos.append(_trajectory_diagnostic_snapshot(optimizer))
            if retain != "none":
                optimizers.append(optimizer)
                weights.append(float(sample.weight))
            if retain == "all":
                gate_streams.append(sample.gate_stream)
                records.append(sample.records)
                leakage_records.append(())
                measurement_records.append(_optimizer_measurement_records(optimizer))
        return TrajectoryShotResult(
            tuple(optimizers),
            tuple(gate_streams),
            tuple(records),
            tuple(leakage_records),
            tuple(measurement_records),
            tuple(weights),
            shot_count=int(shots),
            diagnostics=_trajectory_diagnostics(
                plan,
                optimizers,
                shots=int(shots),
                coalesced=False,
                diagnostic_infos=(
                    diagnostic_infos if retain == "none" else ()
                ),
            ),
        )
    if strategy == "coalesced":
        return run_coalesced_trajectory_shots(
            optimizer_factory,
            entries,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            importance_sampling=policy,
            retain=retain,
        )
    if strategy == "auto" and _trajectory_coalescing_fits_cap(
        plan, shots, max_branches, max_branch_factor
    ):
        try:
            return run_coalesced_trajectory_shots(
                optimizer_factory,
                entries,
                shots,
                seed=seed,
                run_kwargs=run_kwargs,
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                importance_sampling=policy,
                retain=retain,
            )
        except _CoalescedBranchCapExceeded:
            pass

    optimizers = []
    gate_streams = []
    records = []
    leakage_records = []
    measurement_records = []
    weights = []
    diagnostic_infos = []
    for child_seed in _trajectory_seed_pairs(seed, shots, shot_ids=_shot_ids):
        channel_seed, optimizer_seed = child_seed.channel, child_seed.optimizer
        optimizer = optimizer_factory()
        _check_trajectory_optimizer(optimizer)
        _seed_trajectory_optimizer(optimizer, optimizer_seed)
        rng = np.random.default_rng(channel_seed)
        leakage_state = _LeakageState()
        pending = []
        shot_stream = []
        shot_records = []
        shot_leakage_records = []
        shot_measurements = []
        shot_weight = 1.0
        magic_context = None
        if magic_strategy == "immediate":
            magic_context = _new_magic_context(
                optimizer,
                magic_ancillas,
                recycle=magic_recycle,
                reset_ancillas=magic_reset_ancillas,
            )

        def flush_pending():
            nonlocal pending
            if not pending:
                return
            indexed = tuple(pending)
            start = len(getattr(optimizer, "measurements", ()))
            _run_trajectory_entries(
                optimizer,
                [entry for _event_index, entry in indexed],
                run_kwargs,
                magic_context=magic_context,
            )
            _capture_optimizer_measurements(
                optimizer,
                start,
                _measurement_metadata(indexed),
                shot_measurements,
            )
            shot_stream.extend(entry for _event_index, entry in indexed)
            pending = []

        for event_index, entry in enumerate(entries):
            if not isinstance(entry, TrajectoryEvent):
                leakage_parts = _leakage_event_parts(entry)
                if leakage_parts is not None:
                    flush_pending()
                    start = len(getattr(optimizer, "measurements", ()))
                    _apply_leakage_event(
                        optimizer,
                        leakage_parts,
                        leakage_state,
                        rng,
                        event_index,
                        run_kwargs,
                        shot_stream,
                        shot_leakage_records,
                    )
                    _capture_optimizer_measurements(
                        optimizer,
                        start,
                        [(event_index, False)],
                        shot_measurements,
                    )
                    continue
                control_parts = MpsOptimizer.control_event_parts(entry)
                if control_parts is not None and control_parts[0] == "conditional":
                    # Resolve after the preceding measurement is committed.
                    # The concrete action then follows ordinary control and
                    # leakage dispatch, including nested conditional caps.
                    flush_pending()
                    entry = _resolve_trajectory_conditional(optimizer, entry)
                    if entry is None:
                        continue
                    control_parts = MpsOptimizer.control_event_parts(entry)
                if control_parts is not None:
                    if control_parts[0] == "cap":
                        flush_pending()
                        _apply_trajectory_cap(
                            optimizer, entry, control_parts[2], leakage_state,
                            run_kwargs, shot_stream,
                        )
                        continue
                    if control_parts[0] == "reset":
                        flush_pending()
                        _apply_leakage_reset_control(
                            optimizer,
                            entry,
                            leakage_state,
                            run_kwargs,
                            shot_stream,
                        )
                        continue
                    if any(int(site) in leakage_state.leaked for site in control_parts[2]):
                        flush_pending()
                        start = len(getattr(optimizer, "measurements", ()))
                        if _apply_leaked_measurement_control(
                            optimizer,
                            entry,
                            leakage_state,
                            event_index,
                            run_kwargs,
                            shot_stream,
                            shot_leakage_records,
                        ):
                            _capture_optimizer_measurements(
                                optimizer,
                                start,
                                [(event_index, control_parts[0] == "measure_reset")],
                                shot_measurements,
                            )
                            continue
                if _entry_touches_leaked_qubit(entry, leakage_state):
                    continue
                pending.append((event_index, entry))
                continue
            flush_pending()
            record, likelihood_ratio = _apply_trajectory_event(
                optimizer,
                entry,
                rng,
                event_index,
                run_kwargs,
                importance_policy=policy,
            )
            shot_records.append(record)
            shot_weight *= float(likelihood_ratio)
            outcome = next(
                outcome
                for outcome in entry.channel.outcomes
                if outcome.label == record.label
            )
            shot_stream.append(_entry_from_trajectory_outcome(outcome, entry.where))
        flush_pending()
        if magic_context is not None:
            _finish_magic_context(optimizer, magic_context)
        diagnostic_infos.append(_trajectory_diagnostic_snapshot(optimizer))
        if retain != "none":
            optimizers.append(optimizer)
            weights.append(float(shot_weight))
        if retain == "all":
            gate_streams.append(tuple(shot_stream))
            records.append(tuple(shot_records))
            leakage_records.append(tuple(shot_leakage_records))
            measurement_records.append(tuple(shot_measurements))
    return TrajectoryShotResult(
        tuple(optimizers),
        tuple(gate_streams),
        tuple(records),
        tuple(leakage_records),
        tuple(measurement_records),
        tuple(weights),
        shot_count=int(shots),
        diagnostics=_trajectory_diagnostics(
            plan,
            optimizers,
            shots=int(shots),
            coalesced=False,
            diagnostic_infos=(
                diagnostic_infos if retain == "none" else ()
            ),
        ),
    )


def run_parallel_trajectory_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    strategy: str = "independent",
    max_branches: int | None = _AUTO_MAX_BRANCHES,
    max_branch_factor: int | None = None,
    importance_sampling=None,
    magic_strategy: str = "direct",
    magic_ancillas=None,
    magic_recycle: bool = True,
    magic_reset_ancillas: bool = True,
    magic_projection_order="middle_out",
    parallel_workers: int = 2,
    parallel_backend: str = "thread",
    retain: str = "all",
    _shot_ids=None,
    _progress=None,
) -> TrajectoryShotResult | CoalescedTrajectoryResult:
    """Run trajectory shots or coalesced leaves in deterministic parallel batches.

    Independent shot seeds are allocated in shot order before dispatch and the
    ordered map restores that order after workers finish. Coalesced execution
    shares the deterministic prefix and parallelizes only independent live
    leaves. Threads are also used for ``parallel_backend='gpu'`` so live
    Torch/CuPy/JAX state stays in one process and on the backend selected by
    ``optimizer_factory``.
    """
    strategy = _validate_strategy(strategy)
    retain = _validate_retain(retain)
    workers = _validate_parallel_workers(parallel_workers)
    backend = _validate_parallel_backend(parallel_backend)
    if strategy == "auto":
        raise ValueError(
            "parallel trajectory execution needs strategy='independent' or "
            "'coalesced'."
        )
    if _shot_ids is not None and strategy != "independent":
        raise ValueError("shot_ids are supported only for independent replay.")
    if (
        strategy == "coalesced"
        and str(magic_strategy).strip().lower().replace("-", "_") != "direct"
    ):
        raise ValueError(
            "parallel coalesced trajectory execution currently supports "
            "magic_strategy='direct' only."
        )
    if strategy == "coalesced":
        return run_coalesced_trajectory_shots(
            optimizer_factory,
            gates,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            importance_sampling=importance_sampling,
            parallel_workers=workers,
            parallel_backend=backend,
            retain=retain,
        )

    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    entries = _as_entries(gates)
    plan = compile_trajectory_stream(entries)
    child_seeds = _trajectory_seed_pairs(seed, shots, shot_ids=_shot_ids)

    def run_one(child_seed):
        child = (
            child_seed
            if isinstance(child_seed, _TrajectorySeedPair)
            else _TrajectorySeedPair(*child_seed.spawn(2))
        )
        return run_trajectory_shots(
            optimizer_factory,
            entries,
            1,
            seed=child,
            run_kwargs=run_kwargs,
            strategy="independent",
            magic_strategy=magic_strategy,
            magic_ancillas=magic_ancillas,
            magic_recycle=magic_recycle,
            magic_reset_ancillas=magic_reset_ancillas,
            magic_projection_order=magic_projection_order,
            importance_sampling=importance_sampling,
            retain=retain,
        )

    results = _parallel_map_ordered(
        run_one,
        child_seeds,
        workers,
        backend,
        progress=_progress,
    )
    if any(not isinstance(result, TrajectoryShotResult) for result in results):
        raise TypeError("parallel independent trajectory workers returned an invalid result.")
    diagnostic_infos = tuple(
        {
            "max_kraus_probability_residual": float(
                result.diagnostics.max_kraus_probability_residual
            ),
            "used_kraus_copy_fallback": bool(
                result.diagnostics.used_kraus_copy_fallback
            ),
        }
        for result in results
        if result.diagnostics is not None
    )
    if retain == "none":
        return TrajectoryShotResult(
            (),
            (),
            (),
            (),
            (),
            (),
            shot_count=int(shots),
            diagnostics=_trajectory_diagnostics(
                plan,
                (),
                shots=int(shots),
                coalesced=False,
                diagnostic_infos=diagnostic_infos,
            ),
        )
    return TrajectoryShotResult(
        tuple(result.optimizers[0] for result in results),
        tuple(result.gate_streams[0] for result in results) if retain == "all" else (),
        tuple(result.records[0] for result in results) if retain == "all" else (),
        tuple(result.leakage_records[0] for result in results) if retain == "all" else (),
        tuple(result.measurement_records[0] for result in results) if retain == "all" else (),
        tuple(result.weights[0] for result in results),
        shot_count=int(shots),
        diagnostics=_trajectory_diagnostics(
            plan,
            tuple(result.optimizers[0] for result in results),
            shots=int(shots),
            coalesced=False,
        ),
    )


def _apply_coalesced_trajectory_outcome(
    node,
    event,
    outcome,
    target_probability,
    proposal_probability,
    likelihood_ratio,
    event_index,
    run_kwargs,
):
    """Apply a previously selected channel outcome to one count-bearing node."""
    non_unitary = event.channel.mode == "kraus"
    if non_unitary and _is_mps_stabilizer_trajectory_optimizer(node.optimizer):
        norm_event = node.optimizer._make_norm_event(
            "trajectory_kraus", branch_probability=target_probability
        )
    elif non_unitary and isinstance(node.optimizer, MpsOptimizer):
        norm_event = {
            "kind": "trajectory_kraus",
            "branch_probability": float(target_probability),
            "input_norm": node.optimizer._real_float(node.optimizer.p.norm()),
        }
    else:
        norm_event = None
    entry = _entry_from_trajectory_outcome(outcome, event.where)
    _run_trajectory_entries(
        node.optimizer, (entry,), run_kwargs, non_unitary=non_unitary
    )
    if non_unitary:
        _normalize_trajectory_branch(
            node.optimizer, event.where, norm_event=norm_event
        )
    node.weight *= float(likelihood_ratio)
    node.gate_stream.append(entry)
    node.records.append(
        TrajectoryRecord(
            event_index,
            event.where,
            outcome.label,
            float(target_probability),
            None if proposal_probability is None else float(proposal_probability),
            float(likelihood_ratio),
        )
    )


def run_coalesced_trajectory_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    max_branches: int | None = None,
    max_branch_factor: int | None = None,
    importance_sampling=None,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
    retain: str = "all",
) -> CoalescedTrajectoryResult:
    """Replay an exact count-coalesced ensemble of quantum trajectories.

    This is the memory- and compute-efficient counterpart to
    :func:`run_trajectory_shots`. A shared deterministic prefix runs once.
    Whenever a channel outcome or an unforced mid-circuit measurement splits
    the ensemble, a multinomial/binomial draw assigns counts to child states;
    an MPS copy is made only for nonempty child branches. Both fixed mixtures
    and state-dependent Kraus channels are supported exactly.

    The returned result has one :class:`CoalescedTrajectoryLeaf` per distinct
    final branch, rather than one independently mutable optimizer per shot.
    Use ``leaf.count`` as that branch's multiplicity. ``max_branches`` is an
    exact safety cap: exceeding it raises before retaining more live leaves.
    """
    shots, run_kwargs = _coalesced_inputs(optimizer_factory, shots, run_kwargs)
    max_branches = _validate_max_branches(max_branches)
    max_branch_factor = _validate_max_branch_factor(max_branch_factor)
    retain = _validate_retain(retain)
    parallel_workers = _validate_parallel_workers(parallel_workers)
    parallel_backend = _validate_parallel_backend(parallel_backend)
    policy = _coerce_importance_policy(importance_sampling)
    plan = compile_trajectory_stream(gates)
    entries = plan.entries
    nodes = _initial_coalesced_nodes(optimizer_factory, shots)
    channel_seed, optimizer_seed = np.random.SeedSequence(seed).spawn(2)
    if nodes:
        _seed_trajectory_optimizer(nodes[0].optimizer, optimizer_seed)
    rng = np.random.default_rng(channel_seed)
    pending = []

    def flush():
        nonlocal nodes, pending
        nodes = _run_coalesced_entries(
            nodes,
            pending,
            run_kwargs,
            rng,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
        )
        pending = []

    for event_index, entry in enumerate(entries):
        if not isinstance(entry, TrajectoryEvent):
            pending.append((event_index, entry))
            continue
        flush()
        if entry.channel.mode == "mixture":
            outcomes = entry.channel.outcomes
            target, proposal, ratios = _importance_distribution(
                policy,
                event_index,
                outcomes,
                [outcome.probability for outcome in outcomes],
            )

            target_by_label = {
                outcome.label: (target[index], proposal[index], ratios[index])
                for index, outcome in enumerate(outcomes)
            }

            def apply(node, outcome, proposal_probability):
                target_probability, _proposal, ratio = target_by_label[outcome.label]
                _apply_coalesced_trajectory_outcome(
                    node,
                    entry,
                    outcome,
                    target_probability,
                    proposal_probability if policy is not None else None,
                    ratio,
                    event_index,
                    run_kwargs,
                )

            nodes = _split_coalesced_nodes(
                nodes,
                outcomes,
                proposal,
                apply,
                rng,
                context="trajectory mixture",
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
            )
        else:
            split = []
            for node in nodes:
                probabilities = _kraus_probabilities(
                    node.optimizer, entry.channel, entry.where
                )
                target, proposal, ratios = _importance_distribution(
                    policy,
                    event_index,
                    entry.channel.outcomes,
                    probabilities,
                    node.optimizer,
                )
                target_by_label = {
                    outcome.label: (target[index], proposal[index], ratios[index])
                    for index, outcome in enumerate(entry.channel.outcomes)
                }

                def apply(child, outcome, proposal_probability):
                    target_probability, _proposal, ratio = target_by_label[outcome.label]
                    _apply_coalesced_trajectory_outcome(
                        child,
                        entry,
                        outcome,
                        target_probability,
                        proposal_probability if policy is not None else None,
                        ratio,
                        event_index,
                        run_kwargs,
                    )

                split.extend(
                    _split_coalesced_nodes(
                        [node],
                        entry.channel.outcomes,
                        proposal,
                        apply,
                        rng,
                        context="trajectory Kraus channel",
                        max_branches=max_branches,
                        max_branch_factor=max_branch_factor,
                        parallel_workers=parallel_workers,
                        parallel_backend=parallel_backend,
                    )
                )
            nodes = split
    flush()
    return _coalesced_result(nodes, plan=plan, retain=retain)


def run_coalesced_noisy_shots(
    optimizer_factory: Callable[[], Any],
    gates,
    error_model: PauliErrorModel,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    max_branches: int | None = None,
    max_branch_factor: int | None = None,
    importance_sampling=None,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
    retain: str = "all",
) -> CoalescedTrajectoryResult:
    """Replay independent Pauli-noise shots using exact count coalescing.

    Each ideal gate is replayed once per live branch. Its independent Pauli
    channels then split the branch counts with multinomial draws. With a small
    total fault rate, the no-error branch therefore carries most shots and is
    simulated just once, on either CPU or GPU. The probability distribution is
    identical to :func:`run_noisy_shots`; only the retained representation is
    different. ``max_branches`` optionally stops replay before retaining more
    than that many live leaves. It raises :class:`RuntimeError` rather than
    dropping samples; ``run_noisy_shots(strategy="auto")`` catches that
    condition and restarts independently.
    """
    if not isinstance(error_model, PauliErrorModel):
        raise TypeError("error_model must be a PauliErrorModel.")
    shots, run_kwargs = _coalesced_inputs(optimizer_factory, shots, run_kwargs)
    max_branches = _validate_max_branches(max_branches)
    max_branch_factor = _validate_max_branch_factor(max_branch_factor)
    retain = _validate_retain(retain)
    parallel_workers = _validate_parallel_workers(parallel_workers)
    parallel_backend = _validate_parallel_backend(parallel_backend)
    entries = _as_entries(gates)
    plan = compile_trajectory_stream(entries)
    if _contains_stochastic_entries(entries):
        raise ValueError(
            "Stream-local stochastic entries require run_coalesced_trajectory_shots(...). "
            "PauliErrorModel is a convenience macro for clean deterministic streams."
        )
    if _contains_leakage_entries(entries):
        raise ValueError(
            "Stateful leakage entries require run_trajectory_shots(...). "
            "PauliErrorModel is a convenience macro for clean deterministic streams."
        )
    if importance_sampling is not None and not isinstance(
        importance_sampling, PauliErrorModel
    ):
        raise TypeError("importance_sampling must be a PauliErrorModel for Pauli noise.")
    nodes = _initial_coalesced_nodes(optimizer_factory, shots)
    channel_seed, optimizer_seed = np.random.SeedSequence(seed).spawn(2)
    if nodes:
        _seed_trajectory_optimizer(nodes[0].optimizer, optimizer_seed)
    rng = np.random.default_rng(channel_seed)
    pending = []

    def flush():
        nonlocal nodes, pending
        nodes = _run_coalesced_entries(
            nodes,
            pending,
            run_kwargs,
            rng,
            max_branches=max_branches,
            max_branch_factor=max_branch_factor,
            parallel_workers=parallel_workers,
            parallel_backend=parallel_backend,
        )
        pending = []

    outcomes = ("I", "X", "Y", "Z")
    target_probabilities = tuple(error_model.probabilities[label] for label in outcomes)
    proposal_model = error_model if importance_sampling is None else importance_sampling
    proposal_probabilities = tuple(
        proposal_model.probabilities[label] for label in outcomes
    )
    for gate_index, entry in enumerate(entries):
        pending.append((gate_index, entry))
        flush()
        conditional = _conditional_pauli_support(entry)
        if conditional is not None:
            payload, support = conditional
            matched = []
            unmatched = []
            for node in nodes:
                (matched if _conditional_matches(node.optimizer, payload) else unmatched).append(
                    node
                )
            for site in support:

                def apply(node, pauli, proposal_probability):
                    if pauli == "I":
                        target = target_probabilities[0]
                    else:
                        target = error_model.probabilities[pauli]
                        fault_entry = (_pauli_matrix(pauli), site)
                        _run_trajectory_entries(
                            node.optimizer, (fault_entry,), run_kwargs
                        )
                        node.gate_stream.append(fault_entry)
                        node.faults.append(PauliFault(gate_index, int(site), pauli))
                    ratio = float(
                        target / proposal_probabilities[outcomes.index(pauli)]
                    )
                    node.weight *= ratio

                available_branches = (
                    None
                    if max_branches is None
                    else max_branches - len(unmatched)
                )
                if available_branches is not None and available_branches < 1:
                    raise _CoalescedBranchCapExceeded(
                        f"coalesced trajectory branch cap ({max_branches}) exceeded "
                        "while applying a conditional Pauli error."
                    )
                split = _split_coalesced_nodes(
                    matched,
                    outcomes,
                    proposal_probabilities,
                    apply,
                    rng,
                    context="conditional Pauli error model",
                    max_branches=available_branches,
                    max_branch_factor=max_branch_factor,
                    parallel_workers=parallel_workers,
                    parallel_backend=parallel_backend,
                )
                matched = split
            nodes = unmatched + matched
            continue
        support = _event_support(entry)
        if support is None:
            continue
        for site in support:

            def apply(node, pauli, proposal_probability):
                if pauli == "I":
                    target = target_probabilities[0]
                else:
                    target = error_model.probabilities[pauli]
                    fault_entry = (_pauli_matrix(pauli), site)
                    _run_trajectory_entries(node.optimizer, (fault_entry,), run_kwargs)
                    node.gate_stream.append(fault_entry)
                    node.faults.append(PauliFault(gate_index, int(site), pauli))
                ratio = float(target / proposal_probabilities[outcomes.index(pauli)])
                node.weight *= ratio

            nodes = _split_coalesced_nodes(
                nodes,
                outcomes,
                proposal_probabilities,
                apply,
                rng,
                context="Pauli error model",
                max_branches=max_branches,
                max_branch_factor=max_branch_factor,
                parallel_workers=parallel_workers,
                parallel_backend=parallel_backend,
            )
    return _coalesced_result(nodes, plan=plan, retain=retain)


# ---------------------------------------------------------------------------
# Stim circuit compilation and complete native noise-channel sampling.
# ---------------------------------------------------------------------------
def _require_stim():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - only without optional stim
        raise ImportError(
            "Stim circuit noise support requires the optional 'stim' package. "
            "Install it with `python -m pip install stim`."
        ) from exc
    return stim


def _coerce_stim_circuit(circuit):
    stim = _require_stim()
    if isinstance(circuit, stim.Circuit):
        return circuit
    if isinstance(circuit, str):
        return stim.Circuit(circuit)
    raise TypeError("circuit must be a stim.Circuit, Stim source string, or StimCircuitPlan.")


def _stim_qubit_targets(instruction, *, allow_inverted_result=False) -> tuple[int, ...]:
    targets = []
    for target in instruction.targets_copy():
        if not target.is_qubit_target:
            raise NotImplementedError(
                f"Stim instruction {instruction!s} has a non-qubit target that "
                "cannot be replayed as an MPS gate stream."
            )
        if target.is_inverted_result_target and not allow_inverted_result:
            raise NotImplementedError(
                f"Stim instruction {instruction!s} has a classical-record "
                "controlled/inverted target that cannot be replayed as an MPS gate stream."
            )
        targets.append(int(target.value))
    return tuple(targets)


def _stim_pauli_targets(instruction) -> tuple[tuple[str, int], ...]:
    terms = []
    for target in instruction.targets_copy():
        if target.is_combiner:
            raise NotImplementedError(
                f"Stim noise instruction {instruction!s} unexpectedly contains a combiner."
            )
        if target.is_x_target:
            axis = "X"
        elif target.is_y_target:
            axis = "Y"
        elif target.is_z_target:
            axis = "Z"
        else:
            raise NotImplementedError(
                f"Stim noise instruction {instruction!s} has a non-Pauli target."
            )
        terms.append((axis, int(target.value)))
    return tuple(terms)


def _stim_record_targets(instruction) -> tuple[tuple[int, bool], ...]:
    """Normalize Stim ``rec[k]`` targets to ``(offset, inverted)`` pairs."""
    targets = []
    for target in instruction.targets_copy():
        if not target.is_measurement_record_target:
            raise NotImplementedError(
                f"Stim annotation {instruction!s} has a non-record target."
            )
        targets.append((int(target.value), bool(target.is_inverted_result_target)))
    if not targets:
        raise ValueError(f"Stim annotation {instruction!s} needs record targets.")
    return tuple(targets)


def _stim_unitary_matrix(name: str) -> np.ndarray:
    """Return a cached small Clifford matrix from Stim's public tableau API."""
    try:
        return _STIM_UNITARY_CACHE[name]
    except KeyError:
        stim = _require_stim()
        matrix = np.asarray(
            stim.Tableau.from_named_gate(name).to_unitary_matrix(endian="big"),
            dtype=np.complex128,
        )
        _STIM_UNITARY_CACHE[name] = matrix
        return matrix


def _compile_stim_measurement(instruction, name: str) -> tuple[object, ...]:
    if name in _STIM_SINGLE_MEASUREMENTS:
        axis = _STIM_SINGLE_MEASUREMENTS[name]
        return tuple(
            ("measure", axis, site)
            for site in _stim_qubit_targets(instruction, allow_inverted_result=True)
        )
    if name in _STIM_SINGLE_MEASURE_RESETS:
        axis = _STIM_SINGLE_MEASURE_RESETS[name]
        return tuple(
            ("measure_reset", axis, site)
            for site in _stim_qubit_targets(instruction, allow_inverted_result=True)
        )
    if name in _STIM_PAIR_MEASUREMENTS:
        targets = _stim_qubit_targets(instruction, allow_inverted_result=True)
        if len(targets) % 2:
            raise ValueError(f"Stim instruction {instruction!s} needs target pairs.")
        axis = _STIM_PAIR_MEASUREMENTS[name]
        return tuple(
            ("measure", axis, targets[offset : offset + 2])
            for offset in range(0, len(targets), 2)
        )
    if name != "MPP":
        raise NotImplementedError(f"Unsupported Stim measurement instruction {instruction!s}.")

    groups = []
    targets = instruction.targets_copy()
    offset = 0
    while offset < len(targets):
        axes = []
        sites = []
        while True:
            target = targets[offset]
            if target.is_x_target:
                axis = "X"
            elif target.is_y_target:
                axis = "Y"
            elif target.is_z_target:
                axis = "Z"
            else:
                raise ValueError(f"Malformed Stim MPP instruction {instruction!s}.")
            axes.append(axis)
            sites.append(int(target.value))
            offset += 1
            if offset == len(targets) or not targets[offset].is_combiner:
                break
            offset += 1
            if offset == len(targets):
                raise ValueError(f"Malformed Stim MPP instruction {instruction!s}.")
        groups.append(("measure", "".join(axes), tuple(sites)))
    return tuple(groups)


def _compile_stim_unitary(instruction, name: str) -> tuple[object, ...]:
    stim = _require_stim()
    if name in {"I", "II"}:
        return ()
    gate_data = stim.gate_data(name)
    if gate_data.is_single_qubit_gate:
        targets = _stim_qubit_targets(instruction)
        matrix = _stim_unitary_matrix(name)
        return tuple((matrix, site) for site in targets)
    if gate_data.is_two_qubit_gate:
        targets = _stim_qubit_targets(instruction)
        if len(targets) % 2:
            raise ValueError(f"Stim instruction {instruction!s} needs target pairs.")
        matrix = _stim_unitary_matrix(name)
        return tuple(
            (matrix, targets[offset : offset + 2])
            for offset in range(0, len(targets), 2)
        )
    raise NotImplementedError(
        f"Stim instruction {instruction!s} is not a one- or two-qubit unitary "
        "supported by both MPS optimizers."
    )


def _compile_stim_classical_control(instruction, name: str):
    """Lower one Stim measurement-record-controlled Pauli gate.

    The supported form is CX/CY/CZ rec[k] q, including an inverted record
    target. It becomes the backend-independent ("if", k, bit, action) event.
    More elaborate classical arithmetic remains intentionally outside this
    compiler so a record cannot be mistaken for a quantum wire.
    """
    targets = instruction.targets_copy()
    records = [target for target in targets if target.is_measurement_record_target]
    qubits = [target for target in targets if target.is_qubit_target]
    if len(targets) != 2 or len(records) != 1 or len(qubits) != 1:
        raise NotImplementedError(
            f"Stim instruction {instruction!s} must contain exactly one "
            "measurement-record control and one qubit target."
        )
    if name not in {"CX", "CY", "CZ"}:
        raise NotImplementedError(
            f"Stim classical-record-controlled gate {name} is not supported."
        )
    record = records[0]
    # Stim's inverted record target means 'apply when the recorded bit is 0'.
    expected_bit = 0 if record.is_inverted_result_target else 1
    axis = {"CX": "X", "CY": "Y", "CZ": "Z"}[name]
    return (
        "if",
        int(record.value),
        expected_bit,
        (_stim_unitary_matrix(axis), int(qubits[0].value)),
    )


def compile_stim_circuit(circuit) -> StimCircuitPlan:
    """Compile a Stim circuit into reusable physical MPS stream operations.

    All native stochastic error instructions are retained for trajectory
    sampling. Clifford one- and two-qubit gates, single/product Pauli
    measurements, and Pauli-basis resets are also translated. Detector and
    observable annotations are retained in the plan and resolved after replay;
    coordinate and tick instructions have no quantum effect. Measurement-record-
    controlled Pauli gates ``CX/CY/CZ rec[k] q`` are lowered to explicit
    feed-forward events. General classical arithmetic and record-to-record
    operations remain rejected.
    """
    if isinstance(circuit, StimCircuitPlan):
        return circuit
    stim_circuit = _coerce_stim_circuit(circuit)
    stim = _require_stim()
    operations = []
    detectors = []
    observables = []
    measurement_count = 0
    for instruction_index, instruction in enumerate(stim_circuit.flattened()):
        name = instruction.name.upper()
        args = tuple(float(value) for value in instruction.gate_args_copy())
        if name == "DETECTOR":
            detectors.append(
                StimDetector(
                    instruction_index=instruction_index,
                    detector_index=len(detectors),
                    rec_targets=_stim_record_targets(instruction),
                    coordinates=args,
                    measurement_count=measurement_count,
                )
            )
            continue
        if name == "OBSERVABLE_INCLUDE":
            if len(args) != 1 or int(args[0]) != args[0] or args[0] < 0:
                raise ValueError(
                    f"Stim OBSERVABLE_INCLUDE needs one nonnegative integer id: {instruction!s}."
                )
            observables.append(
                StimObservable(
                    instruction_index=instruction_index,
                    observable_index=int(args[0]),
                    rec_targets=_stim_record_targets(instruction),
                    measurement_count=measurement_count,
                )
            )
            continue
        if name in _STIM_NOISE_NAMES:
            targets = (
                _stim_pauli_targets(instruction)
                if name in {"E", "ELSE_CORRELATED_ERROR"}
                else tuple(("I", site) for site in _stim_qubit_targets(instruction))
            )
            operations.append(
                _StimPlanOperation(
                    instruction_index, name, args, targets, is_noise=True
                )
            )
            continue
        if name in _STIM_IGNORED_NAMES:
            continue
        if any(
            target.is_measurement_record_target
            for target in instruction.targets_copy()
        ):
            entries = (_compile_stim_classical_control(instruction, name),)
            operations.append(
                _StimPlanOperation(instruction_index, name, args, (), entries)
            )
            continue
        if name in _STIM_SINGLE_MEASUREMENTS or name in _STIM_SINGLE_MEASURE_RESETS:
            entries = _compile_stim_measurement(instruction, name)
        elif name in _STIM_PAIR_MEASUREMENTS or name == "MPP":
            entries = _compile_stim_measurement(instruction, name)
        elif name in _STIM_RESETS:
            entries = tuple(
                ("reset", site, _STIM_RESETS[name])
                for site in _stim_qubit_targets(instruction)
            )
        else:
            gate_data = stim.gate_data(name)
            if not gate_data.is_unitary:
                raise NotImplementedError(
                    f"Stim instruction {instruction!s} is not a supported quantum "
                    "operation for MPS replay."
                )
            entries = _compile_stim_unitary(instruction, name)
        if name in _STIM_SINGLE_MEASUREMENTS or name in _STIM_SINGLE_MEASURE_RESETS:
            measurement_count += len(entries)
        elif name in _STIM_PAIR_MEASUREMENTS or name == "MPP":
            measurement_count += len(entries)
        operations.append(_StimPlanOperation(instruction_index, name, args, (), entries))
    return StimCircuitPlan(
        int(stim_circuit.num_qubits),
        tuple(operations),
        tuple(detectors),
        tuple(observables),
    )


def _sample_label(
    rng,
    labels,
    probabilities,
    *,
    context: str,
    importance_policy=None,
    event_index=None,
    weight_box=None,
) -> str:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError(f"Invalid probability configuration for {context}.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"Invalid probability configuration for {context}.")
    total = float(probabilities.sum())
    if total > 1.0 + 1e-10:
        raise ValueError(f"Probabilities for {context} sum to more than one.")
    identity = max(0.0, 1.0 - total)
    target, proposal, ratios = _importance_label_distribution(
        importance_policy,
        event_index,
        ("I", *labels),
        (identity, *probabilities),
    )
    index = int(rng.choice(len(target), p=proposal))
    if weight_box is not None:
        weight_box[0] *= float(ratios[index])
    return str(("I", *labels)[index])


def _append_pauli_terms(stream, faults, instruction_index, terms):
    for axis, site in terms:
        if axis == "I":
            continue
        stream.append((_pauli_matrix(axis), site))
        faults.append(PauliFault(instruction_index, site, axis))


def _sample_stim_noise_operation(
    op,
    rng,
    stream,
    faults,
    heralds,
    *,
    correlated,
    importance_policy=None,
    weight_box=None,
):
    """Sample one native Stim noise instruction into local physical Paulis."""
    name = op.name
    qubits = tuple(site for _axis, site in op.targets)
    args = op.args
    if name in {"I_ERROR", "II_ERROR"}:
        return False
    if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"Stim {name} needs one probability argument.")
        axis = name[0]
        for site in qubits:
            sampled = _sample_label(
                rng,
                (axis,),
                (args[0],),
                context=name,
                importance_policy=importance_policy,
                event_index=op.instruction_index,
                weight_box=weight_box,
            )
            if sampled == axis:
                _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name == "DEPOLARIZE1":
        if len(args) != 1:
            raise ValueError("Stim DEPOLARIZE1 needs one probability argument.")
        for site in qubits:
            axis = _sample_label(
                rng,
                ("X", "Y", "Z"),
                (args[0] / 3.0,) * 3,
                context=name,
                importance_policy=importance_policy,
                event_index=op.instruction_index,
                weight_box=weight_box,
            )
            _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name == "PAULI_CHANNEL_1":
        if len(args) != 3:
            raise ValueError("Stim PAULI_CHANNEL_1 needs three probability arguments.")
        for site in qubits:
            axis = _sample_label(
                rng,
                ("X", "Y", "Z"),
                args,
                context=name,
                importance_policy=importance_policy,
                event_index=op.instruction_index,
                weight_box=weight_box,
            )
            _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name in {"DEPOLARIZE2", "PAULI_CHANNEL_2"}:
        if len(qubits) % 2:
            raise ValueError(f"Stim {name} needs target pairs.")
        probabilities = (
            (args[0] / 15.0,) * 15 if name == "DEPOLARIZE2" else args
        )
        if len(probabilities) != 15:
            expected = "one" if name == "DEPOLARIZE2" else "fifteen"
            raise ValueError(f"Stim {name} needs {expected} probability argument(s).")
        labels = tuple(left + right for left, right in _STIM_PAULI_2_OUTCOMES)
        for offset in range(0, len(qubits), 2):
            label = _sample_label(
                rng,
                labels,
                probabilities,
                context=name,
                importance_policy=importance_policy,
                event_index=op.instruction_index,
                weight_box=weight_box,
            )
            _append_pauli_terms(
                stream,
                faults,
                op.instruction_index,
                zip(label, qubits[offset : offset + 2]),
            )
        return False
    if name == "E":
        if len(args) != 1:
            raise ValueError("Stim E/CORRELATED_ERROR needs one probability argument.")
        _target, proposal, ratios = _importance_label_distribution(
            importance_policy,
            op.instruction_index,
            ("I", "FAULT"),
            (1.0 - args[0], args[0]),
        )
        index = int(rng.choice(2, p=proposal))
        if weight_box is not None:
            weight_box[0] *= float(ratios[index])
        occurred = index == 1
        if occurred:
            _append_pauli_terms(stream, faults, op.instruction_index, op.targets)
        return occurred
    if name == "ELSE_CORRELATED_ERROR":
        if len(args) != 1:
            raise ValueError("Stim ELSE_CORRELATED_ERROR needs one probability argument.")
        if correlated is None:
            raise ValueError(
                "Stim ELSE_CORRELATED_ERROR must immediately follow E or another "
                "ELSE_CORRELATED_ERROR."
            )
        if correlated:
            occurred = True
        else:
            _target, proposal, ratios = _importance_label_distribution(
                importance_policy,
                op.instruction_index,
                ("I", "FAULT"),
                (1.0 - args[0], args[0]),
            )
            index = int(rng.choice(2, p=proposal))
            if weight_box is not None:
                weight_box[0] *= float(ratios[index])
            occurred = index == 1
        if not correlated and occurred:
            _append_pauli_terms(stream, faults, op.instruction_index, op.targets)
        return occurred
    if name == "HERALDED_ERASE":
        if len(args) != 1:
            raise ValueError("Stim HERALDED_ERASE needs one probability argument.")
        for site in qubits:
            labels = ("NO_HERALD", "HERALD_I", "X", "Y", "Z")
            target, proposal, ratios = _importance_label_distribution(
                importance_policy,
                op.instruction_index,
                labels,
                (1.0 - args[0],) + (args[0] / 4.0,) * 4,
            )
            index = int(rng.choice(len(labels), p=proposal))
            if weight_box is not None:
                weight_box[0] *= float(ratios[index])
            label = labels[index]
            fired = label != "NO_HERALD"
            heralds.append(StimHerald(op.instruction_index, site, fired))
            if label in {"X", "Y", "Z"}:
                axis = label
                _append_pauli_terms(stream, faults, op.instruction_index, ((axis, site),))
        return False
    if name == "HERALDED_PAULI_CHANNEL_1":
        if len(args) != 4:
            raise ValueError(
                "Stim HERALDED_PAULI_CHANNEL_1 needs four probability arguments."
            )
        for site in qubits:
            label = _sample_label(
                rng,
                ("HERALD_I", "X", "Y", "Z"),
                args,
                context=name,
                importance_policy=importance_policy,
                event_index=op.instruction_index,
                weight_box=weight_box,
            )
            fired = label != "I"
            heralds.append(StimHerald(op.instruction_index, site, fired))
            if label == "HERALD_I":
                continue
            _append_pauli_terms(stream, faults, op.instruction_index, ((label, site),))
        return False
    raise AssertionError(f"Unhandled native Stim noise instruction {name!r}.")


def _sample_stim_plan(
    plan: StimCircuitPlan, rng, *, importance_sampling=None
) -> StimNoiseSample:
    policy = _coerce_importance_policy(importance_sampling)
    stream = []
    faults = []
    heralds = []
    weight_box = [1.0]
    correlated = None
    for op in plan.operations:
        if not op.is_noise:
            stream.extend(op.entries)
            correlated = None
            continue
        if op.name == "E":
            correlated = _sample_stim_noise_operation(
                op,
                rng,
                stream,
                faults,
                heralds,
                correlated=None,
                importance_policy=policy,
                weight_box=weight_box,
            )
        elif op.name == "ELSE_CORRELATED_ERROR":
            correlated = _sample_stim_noise_operation(
                op,
                rng,
                stream,
                faults,
                heralds,
                correlated=correlated,
                importance_policy=policy,
                weight_box=weight_box,
            )
        else:
            _sample_stim_noise_operation(
                op,
                rng,
                stream,
                faults,
                heralds,
                correlated=None,
                importance_policy=policy,
                weight_box=weight_box,
            )
            correlated = None
    return StimNoiseSample(tuple(stream), tuple(faults), tuple(heralds), weight_box[0])


def sample_stim_circuit(
    circuit, *, seed=None, importance_sampling=None
) -> StimNoiseSample:
    """Sample every native Stim error channel into one replayable MPS stream.

    ``circuit`` can be a :class:`stim.Circuit`, Stim source text, or a reusable
    :class:`StimCircuitPlan`. The stream contains the compiled ideal operations
    and the sampled local Pauli faults in their original temporal order.
    """
    return _sample_stim_plan(
        compile_stim_circuit(circuit),
        np.random.default_rng(seed),
        importance_sampling=importance_sampling,
    )


def sample_stim_circuits(
    circuit, shots: int, *, seed=None, importance_sampling=None
) -> list[StimNoiseSample]:
    """Sample ``shots`` independent trajectories from a Stim circuit efficiently."""
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    plan = compile_stim_circuit(circuit)
    return [
        _sample_stim_plan(
            plan,
            np.random.default_rng(child_seed),
            importance_sampling=importance_sampling,
        )
        for child_seed in np.random.SeedSequence(seed).spawn(int(shots))
    ]


def _stream_on_optimizer_backend(stream, optimizer):
    """Convert library-generated dense gates to the live MPS/TTN backend."""
    converted = []
    for entry in stream:
        if (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and not isinstance(entry[0], str)
            and hasattr(entry[0], "shape")
        ):
            converted.append((_to_trajectory_backend(entry[0], optimizer), entry[1]))
        else:
            converted.append(entry)
    return tuple(converted)


def run_stim_shots(
    optimizer_factory: Callable[[], Any],
    circuit,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    importance_sampling=None,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
) -> StimShotResult:
    """Sample and replay a Stim circuit on fresh MPS or STN optimizers.

    The circuit is compiled once, then each shot samples only its native Pauli
    channels. This keeps sampling linear in the flattened circuit size and in
    the number of non-identity faults; no density matrix and no dense noisy
    channel are constructed.
    """
    if not callable(optimizer_factory):
        raise TypeError("optimizer_factory must construct a fresh optimizer per shot.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    if run_kwargs is None:
        run_kwargs = {}
    elif not isinstance(run_kwargs, Mapping):
        raise TypeError("run_kwargs must be a mapping or None.")
    workers = _validate_parallel_workers(parallel_workers)
    backend = _validate_parallel_backend(parallel_backend)
    if workers > 1:
        return run_parallel_stim_shots(
            optimizer_factory,
            circuit,
            shots,
            seed=seed,
            run_kwargs=run_kwargs,
            importance_sampling=importance_sampling,
            parallel_workers=workers,
            parallel_backend=backend,
        )

    plan = compile_stim_circuit(circuit)
    optimizers = []
    samples = []
    for child_seed in _trajectory_seed_pairs(seed, shots):
        noise_seed, optimizer_seed = child_seed.channel, child_seed.optimizer
        sample = _sample_stim_plan(
            plan,
            np.random.default_rng(noise_seed),
            importance_sampling=importance_sampling,
        )
        optimizer = optimizer_factory()
        if not hasattr(optimizer, "set_gates") or not hasattr(optimizer, "run"):
            raise TypeError(
                "optimizer_factory must return an optimizer with set_gates(...) and run(...)."
            )
        _seed_trajectory_optimizer(optimizer, optimizer_seed)
        optimizer.set_gates(_stream_on_optimizer_backend(sample.gate_stream, optimizer))
        optimizer.run(**dict(run_kwargs))
        optimizers.append(optimizer)
        samples.append(sample)
    return StimShotResult(tuple(optimizers), tuple(samples), plan)


def run_parallel_stim_shots(
    optimizer_factory: Callable[[], Any],
    circuit,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
    importance_sampling=None,
    parallel_workers: int = 2,
    parallel_backend: str = "thread",
) -> StimShotResult:
    """Run compiled Stim trajectories in deterministic parallel shot order."""
    workers = _validate_parallel_workers(parallel_workers)
    backend = _validate_parallel_backend(parallel_backend)
    plan = compile_stim_circuit(circuit)
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
        raise ValueError("shots must be a nonnegative integer.")
    child_seeds = _trajectory_seed_pairs(seed, shots)

    def run_one(child_seed):
        return run_stim_shots(
            optimizer_factory,
            plan,
            1,
            seed=child_seed,
            run_kwargs=run_kwargs,
            importance_sampling=importance_sampling,
        )

    results = _parallel_map_ordered(run_one, child_seeds, workers, backend)
    return StimShotResult(
        tuple(result.optimizers[0] for result in results),
        tuple(result.samples[0] for result in results),
        plan,
    )


def _stim_categorical_outcomes(labels, probabilities, *, context):
    """Return identity plus labelled Stim outcomes after strict validation."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError(f"Invalid probability configuration for {context}.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
        raise ValueError(f"Invalid probability configuration for {context}.")
    total = float(probabilities.sum())
    if total > 1.0 + 1e-10:
        raise ValueError(f"Probabilities for {context} sum to more than one.")
    return ("I", *labels), (max(0.0, 1.0 - total), *probabilities)


def _coalesced_stim_pauli_terms(node, op, terms, run_kwargs):
    """Apply and record the non-identity Pauli terms of one Stim outcome."""
    terms = tuple((axis, int(site)) for axis, site in terms if axis != "I")
    if not terms:
        return
    entries = tuple((_pauli_matrix(axis), site) for axis, site in terms)
    _run_trajectory_entries(node.optimizer, entries, run_kwargs)
    node.gate_stream.extend(entries)
    node.faults.extend(
        PauliFault(op.instruction_index, site, axis) for axis, site in terms
    )


def _coalesced_stim_single_site_channel(
    nodes, op, site, labels, probabilities, run_kwargs, rng, *, heralded=False
):
    """Branch one independent native Stim channel target."""
    labels, probabilities = _stim_categorical_outcomes(
        labels, probabilities, context=op.name
    )

    def apply(node, label, _probability):
        if heralded:
            fired = label != "I"
            node.heralds.append(StimHerald(op.instruction_index, int(site), fired))
            if label == "HERALD_I":
                return
        if label != "I":
            _coalesced_stim_pauli_terms(node, op, ((label, site),), run_kwargs)

    return _split_coalesced_nodes(
        nodes,
        labels,
        probabilities,
        apply,
        rng,
        context=op.name,
    )


def _coalesced_stim_noise_operation(nodes, op, run_kwargs, rng):
    """Apply one independent native Stim-noise instruction exactly by counts."""
    name = op.name
    args = op.args
    qubits = tuple(site for _axis, site in op.targets)
    if name in {"I_ERROR", "II_ERROR"}:
        return nodes
    if name in {"X_ERROR", "Y_ERROR", "Z_ERROR"}:
        if len(args) != 1:
            raise ValueError(f"Stim {name} needs one probability argument.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes, op, site, (name[0],), args, run_kwargs, rng
            )
        return nodes
    if name == "DEPOLARIZE1":
        if len(args) != 1:
            raise ValueError("Stim DEPOLARIZE1 needs one probability argument.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes, op, site, ("X", "Y", "Z"), (args[0] / 3.0,) * 3,
                run_kwargs, rng,
            )
        return nodes
    if name == "PAULI_CHANNEL_1":
        if len(args) != 3:
            raise ValueError("Stim PAULI_CHANNEL_1 needs three probability arguments.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes, op, site, ("X", "Y", "Z"), args, run_kwargs, rng
            )
        return nodes
    if name in {"DEPOLARIZE2", "PAULI_CHANNEL_2"}:
        if len(qubits) % 2:
            raise ValueError(f"Stim {name} needs target pairs.")
        probabilities = (args[0] / 15.0,) * 15 if name == "DEPOLARIZE2" else args
        if len(probabilities) != 15:
            expected = "one" if name == "DEPOLARIZE2" else "fifteen"
            raise ValueError(f"Stim {name} needs {expected} probability argument(s).")
        labels = tuple(left + right for left, right in _STIM_PAULI_2_OUTCOMES)
        labels, probabilities = _stim_categorical_outcomes(
            labels, probabilities, context=name
        )
        for offset in range(0, len(qubits), 2):
            pair = qubits[offset : offset + 2]

            def apply(node, label, _probability):
                if label != "I":
                    _coalesced_stim_pauli_terms(
                        node, op, zip(label, pair), run_kwargs
                    )

            nodes = _split_coalesced_nodes(
                nodes, labels, probabilities, apply, rng, context=name
            )
        return nodes
    if name == "HERALDED_ERASE":
        if len(args) != 1:
            raise ValueError("Stim HERALDED_ERASE needs one probability argument.")
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes,
                op,
                site,
                ("HERALD_I", "X", "Y", "Z"),
                (args[0] / 4.0,) * 4,
                run_kwargs,
                rng,
                heralded=True,
            )
        return nodes
    if name == "HERALDED_PAULI_CHANNEL_1":
        if len(args) != 4:
            raise ValueError(
                "Stim HERALDED_PAULI_CHANNEL_1 needs four probability arguments."
            )
        for site in qubits:
            nodes = _coalesced_stim_single_site_channel(
                nodes,
                op,
                site,
                ("HERALD_I", "X", "Y", "Z"),
                args,
                run_kwargs,
                rng,
                heralded=True,
            )
        return nodes
    raise AssertionError(f"Unhandled independent Stim noise instruction {name!r}.")


def _coalesced_stim_correlated_chain(nodes, operations, run_kwargs, rng):
    """Sample a contiguous Stim E/ELSE chain with one categorical split."""
    if not operations:
        return nodes
    if operations[0].name != "E":
        raise ValueError(
            "Stim ELSE_CORRELATED_ERROR must immediately follow E or another "
            "ELSE_CORRELATED_ERROR."
        )
    probabilities = []
    survival = 1.0
    for op in operations:
        if len(op.args) != 1:
            raise ValueError(f"Stim {op.name} needs one probability argument.")
        probability = float(op.args[0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probabilities for {op.name} must lie in [0, 1].")
        probabilities.append(survival * probability)
        survival *= 1.0 - probability
    outcomes = (None, *operations)
    probabilities = (survival, *probabilities)

    def apply(node, op, _probability):
        if op is not None:
            _coalesced_stim_pauli_terms(node, op, op.targets, run_kwargs)

    return _split_coalesced_nodes(
        nodes,
        outcomes,
        probabilities,
        apply,
        rng,
        context="Stim correlated-error chain",
    )


def run_coalesced_stim_shots(
    optimizer_factory: Callable[[], Any],
    circuit,
    shots: int,
    *,
    seed=None,
    run_kwargs: Optional[Mapping[str, Any]] = None,
) -> CoalescedTrajectoryResult:
    """Replay all native Stim Pauli-noise channels using exact count coalescing.

    The Stim circuit is compiled once. Ideal segments and no-error branches are
    shared, while native independent, heralded, two-qubit, and correlated
    error instructions split only the affected count-bearing nodes. Compiled
    mid-circuit measurements and resets use the same exact count branching.
    """
    shots, run_kwargs = _coalesced_inputs(optimizer_factory, shots, run_kwargs)
    plan = compile_stim_circuit(circuit)
    nodes = _initial_coalesced_nodes(optimizer_factory, shots)
    channel_seed, optimizer_seed = np.random.SeedSequence(seed).spawn(2)
    if nodes:
        _seed_trajectory_optimizer(nodes[0].optimizer, optimizer_seed)
    rng = np.random.default_rng(channel_seed)
    correlated = []

    for op in plan.operations:
        if not op.is_noise:
            if correlated:
                nodes = _coalesced_stim_correlated_chain(
                    nodes, correlated, run_kwargs, rng
                )
                correlated = []
            nodes = _run_coalesced_entries(
                nodes,
                tuple((op.instruction_index, entry) for entry in op.entries),
                run_kwargs,
                rng,
            )
            continue
        if op.name == "E":
            if correlated:
                nodes = _coalesced_stim_correlated_chain(
                    nodes, correlated, run_kwargs, rng
                )
            correlated = [op]
            continue
        if op.name == "ELSE_CORRELATED_ERROR":
            if not correlated:
                raise ValueError(
                    "Stim ELSE_CORRELATED_ERROR must immediately follow E or another "
                    "ELSE_CORRELATED_ERROR."
                )
            correlated.append(op)
            continue
        if correlated:
            nodes = _coalesced_stim_correlated_chain(
                nodes, correlated, run_kwargs, rng
            )
            correlated = []
        nodes = _coalesced_stim_noise_operation(nodes, op, run_kwargs, rng)
    if correlated:
        nodes = _coalesced_stim_correlated_chain(nodes, correlated, run_kwargs, rng)
    return _coalesced_result(nodes, plan=plan)
