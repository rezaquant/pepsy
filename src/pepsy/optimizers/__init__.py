"""High-level tensor-network optimizers.

The optimizer domains are loaded on demand. Several of them depend on
optional numerical stacks, so importing this namespace stays inexpensive.
"""

from importlib import import_module
import warnings


_SYMBOL_MODULES = {
    "EnergyEstimate": ".energy",
    "MpsEnergyOptimizer": ".energy",
    "PepsEnergyOptimizer": ".energy",
    "TreeEnergyOptimizer": ".energy",
    "GlobalOptimizer": ".global_opt",
    "QMeraBuilder": ".qmera",
    "QMeraDisentanglerSpec": ".qmera",
    "QMeraEnergyOptimizer": ".qmera",
    "QMeraGeometry": ".qmera",
    "QMeraIsometrySpec": ".qmera",
    "QMeraLayoutCandidate": ".qmera",
    "QMeraLayoutFinder": ".qmera",
    "QMeraLayoutReport": ".qmera",
    "QMeraLayoutScore": ".qmera",
    "QMeraParametricEnergyOptimizer": ".qmera",
    "QMeraPrototypeLayout": ".qmera",
    "QMeraScaleSpec": ".qmera",
    "QMeraUnitarySpec": ".qmera",
    "build_qmera_contraction_optimizer": ".qmera",
    "load_qmera_prototype_layout": ".qmera",
    "MpoOptimizer": ".mpo",
    "MpoChannelEvent": ".mpo",
    "GibbsMps": ".mps",
    "MpsOptimizer": ".mps",
    "MPIRankDiagnostics": ".mpi",
    "MPIShotError": ".mpi",
    "MPIShotResult": ".mpi",
    "MPIShotRunner": ".mpi",
    "run_mpi_shots": ".mpi",
    "SimulatorCandidate": ".planning",
    "SimulatorPlan": ".planning",
    "SimulatorPlanner": ".planning",
    "recommend_simulator": ".planning",
    "PepsOptimizer": ".peps",
    "SimpleUpdateGen": ".peps",
    "SymDMRG2": ".sym_dmrg",
    "SweepOptimizer": ".sweep",
    "TreeLayoutFinder": ".tree",
    "TreeMPO": ".tree",
    "TreeOptimizer": ".tree",
    "TreePlan": ".tree",
    "build_tree_operator": ".tree",
    "square_lattice_zigzag": "._layout_orders",
    "TreePeps": ".tree_peps",
    "TreePepsPlan": ".tree_peps",
    "TreePepsGeometry": ".tree_peps",
    "TreePepsLayoutFinder": ".tree_peps",
    "TreePEPO": ".tree_peps",
    "TreeSubPEPO": ".tree_peps",
    "TreePepo": ".tree_peps",
    "TreeSubPepo": ".tree_peps",
    "TreePepsOptimizer": ".tree_peps",
    "TreeStabOptimizer": ".tree_stabilizer",
    "TreeTensorNetwork": ".tree",
    "CoalescedMeasurementRecord": ".noise",
    "CoherentCrosstalkModel": ".noise",
    "CoalescedSampleResult": ".noise",
    "CoalescedTrajectoryLeaf": ".noise",
    "CoalescedTrajectoryResult": ".noise",
    "ImportanceSamplingPolicy": ".noise",
    "LeakageRecord": ".noise",
    "NoisyResult": ".noise",
    "NoisyShotResult": ".noise",
    "PauliErrorModel": ".noise",
    "PauliFault": ".noise",
    "StimCircuitPlan": ".noise",
    "StimDetector": ".noise",
    "StimHerald": ".noise",
    "StimNoiseSample": ".noise",
    "StimObservable": ".noise",
    "StimObservableRecord": ".noise",
    "StimShotResult": ".noise",
    "StimSyndromeRecord": ".noise",
    "TrajectoryChannel": ".noise",
    "TrajectoryDiagnostics": ".noise",
    "TrajectoryEvent": ".noise",
    "TrajectoryStreamPlan": ".noise",
    "TrajectoryOutcome": ".noise",
    "TrajectoryRecord": ".noise",
    "TrajectoryMeasurementRecord": ".noise",
    "TrajectorySample": ".noise",
    "TrajectoryShotResult": ".noise",
    "compile_stim_circuit": ".noise",
    "compile_trajectory_stream": ".noise",
    "run_coalesced_noisy_shots": ".noise",
    "run_coalesced_stim_shots": ".noise",
    "run_coalesced_trajectory_shots": ".noise",
    "TreeNoisy": ".noise",
    "run_noisy_shots": ".noise",
    "run_parallel_noisy_shots": ".noise",
    "run_parallel_stim_shots": ".noise",
    "run_parallel_trajectory_shots": ".noise",
    "run_stim_shots": ".noise",
    "run_trajectory_shots": ".noise",
    "sample_noisy_gate_stream": ".noise",
    "sample_noisy_gate_streams": ".noise",
    "sample_stim_circuit": ".noise",
    "sample_stim_circuits": ".noise",
    "sample_trajectory_stream": ".noise",
    "sample_coalesced_bits": ".noise",
    "DeferredInjectionRecord": ".stabilizer_tn",
    "DeferredInjectionReport": ".stabilizer_tn",
    "DeferredProjectionRecord": ".stabilizer_tn",
    "ImmediateInjectionReport": ".stabilizer_tn",
    "ImmediateProjectionRecord": ".stabilizer_tn",
    "MeasurementRecord": ".stabilizer_tn",
    "MpsStabOptimizer": ".stabilizer_tn",
    "NormEventRecord": ".stabilizer_tn",
    "STNState": ".stabilizer_tn",
    "StabilizerMpsSettingsAdvice": ".stabilizer_tn",
    "StabilizerMpsSimulator": ".stabilizer_tn",
    "StabilizerMpsRunResult": ".stabilizer_tn",
    "StabilizerTreeRunResult": ".stabilizer_tn",
    "StreamAnalysisRecord": ".stabilizer_tn",
    "run_stabilizer_mps_stream": ".stabilizer_tn",
    "run_stabilizer_tree_stream": ".tree_stabilizer",
}

_SUBMODULES = (
    "energy",
    "global_opt",
    "mera",
    "qmera",
    "mpo",
    "mps",
    "noise",
    "mpi",
    "planning",
    "peps",
    "stabilizer_tn",
    "sym_dmrg",
    "sweep",
    "tree",
    "tree_peps",
    "tree_stabilizer",
)

__all__ = [*_SYMBOL_MODULES, *_SUBMODULES]

_DEPRECATED_ALIASES = {
    "QMeraParametricEnergyOptimizer": "QMeraEnergyOptimizer",
    "MpsStabOptimizer": "StabilizerMpsSimulator",
}

def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        canonical = _DEPRECATED_ALIASES.get(name)
        if canonical is not None:
            warnings.warn(
                f"pepsy.optimizers.{name} is a compatibility alias; use "
                f"pepsy.optimizers.{canonical} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
