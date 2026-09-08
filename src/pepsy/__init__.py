"""Tensor-network simulation, contraction, optimization, and sampling."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING

from ._api import _SYMBOL_MODULES

try:
    __version__ = _pkg_version("pepsy")
except PackageNotFoundError:
    __version__ = "0+unknown"

# Canonical package namespaces. Keep advanced domains explicit so callers can
# discover the stable core without scanning every lazy symbol below.
_CORE_MODULES = (
    "backends",
    "boundary",
    "fitting",
    "interop",
    "operators",
    "optimizers",
    "sampling",
    "solvers",
    "tensors",
)
_ADVANCED_MODULES = ("bp", "experimental", "vmc")
_MODULE_EXPORTS = set(_CORE_MODULES) | set(_ADVANCED_MODULES)

# Compatibility facade metadata stays in a private lazy-safe module. Keep
# existing aliases working, but add new functionality to its owning domain
# namespace instead of expanding this root module.

__all__ = ["__version__", *_CORE_MODULES, *_ADVANCED_MODULES, *_SYMBOL_MODULES]


def __getattr__(name):
    """Lazily import public API symbols and new package namespaces."""
    if name in _MODULE_EXPORTS:
        return import_module(f".{name}", __name__)
    if name in _SYMBOL_MODULES:
        module = import_module(_SYMBOL_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'pepsy' has no attribute {name!r}")


if TYPE_CHECKING:
    from .bp import compress_all_gauge, gauge_all, gauge_all_simple, one_norm_bp  # noqa: F401
    from . import backends, boundary, bp, experimental, fitting, interop, operators, optimizers, sampling, solvers, tensors, vmc  # noqa: F401
    from .interop import GuppyConversionError, GuppyGateStream, GuppyMeasurement, guppy_gate_stream  # noqa: F401
    from .backends import (  # noqa: F401
        backend_infer,
        get_default_array_backend,
        get_default_grad_backend,
        get_torch_linalg_config,
        build_backend,
        register_jax_linalg,
        register_torch_linalg,
        reset_linalg_registrations,
        reset_default_backends,
        set_default_array_backend,
        set_default_grad_backend,
        TorchLinalgConfig,
        to_float,
    )
    from .boundary import (  # noqa: F401
        BdyMPS,
        BoundaryContractResult,
        CompBdy,
        boundary_norm,
        build_bra_ket,
        contract_boundary,
        contract_flat,
        infidelity,
        make_numpy_array_caster,
        normalize,
        peps_fidelity,
        peps_infidelity,
        peps_norm,
        peps_normalize,
    )
    from .fitting import FIT  # noqa: F401
    from .operators import (  # noqa: F401
        ActivePEPOBlocks,
        GraphActivePEPOBlocks,
        GraphClusterExpansionPlan,
        ClusterInternalSymmetry,
        ClusterLattice,
        ClusterModelAdapter,
        ConnectedClusterShape,
        GraphConnectedClusterShape,
        ClusterExpansionReport,
        build_cluster_expansion_pepo,
        build_model_cluster_expansion_pepo,
        build_itf_cluster_expansion_pepo,
        build_real_time_cluster_expansion_pepo,
        compose_pepo_layers,
        compose_cluster_expansion_pepo,
        generate_connected_cluster_shapes,
        build_graph_cluster_expansion_pepo,
        adapt_cluster_model,
        ClusterExpansionPlan,
        PauliPEPOTerm,
        PauliPEPOBasis,
        CompiledPEPOExp,
        PEPOClusterFactor,
        PEPOClusterProductExpansion,
        CompiledPEPOClusterProduct,
        build_mpo_from_gates,
        build_pepo_from_gates,
        compress_mpo_product,
        cnot,
        cphase,
        crx,
        cry,
        crz,
        cu1,
        cu2,
        cu3,
        cx,
        cy,
        cz,
        fsim,
        fsimg,
        gate,
        gate_simple,
        h,
        hadamard,
        ham_tn,
        iswap,
        pauli,
        phase,
        renorm_gauge,
        rx,
        rxx,
        ry,
        ryy,
        rz,
        rzz,
        s,
        sdg,
        su4,
        swap,
        t,
        tdg,
        u1,
        u2,
        u3,
        x,
        y,
        z,
    )
    from .optimizers import DeferredInjectionRecord, DeferredInjectionReport, DeferredProjectionRecord, GibbsMps, GlobalOptimizer, ImmediateInjectionReport, ImmediateProjectionRecord, MeasurementRecord, MpoChannelEvent, MpoOptimizer, MpsEnergyOptimizer, MpsOptimizer, MpsStabOptimizer, MPIRankDiagnostics, MPIShotError, MPIShotResult, MPIShotRunner, NormEventRecord, PepsEnergyOptimizer, PepsOptimizer, STNState, StabilizerMpsSettingsAdvice, StabilizerMpsRunResult, StabilizerMpsSimulator, StabilizerTreeRunResult, StreamAnalysisRecord, SimpleUpdateGen, SymDMRG2, SweepOptimizer, TreeEnergyOptimizer, TreeLayoutFinder, TreeMPO, TreeOptimizer, TreePlan, TreePeps, TreePepsGeometry, TreePepsOptimizer, TreePepsPlan, TreePEPO, TreePepo, TreeSubPEPO, TreeSubPepo, TreeStabOptimizer, build_tree_operator, run_mpi_shots, run_stabilizer_mps_stream, run_stabilizer_tree_stream  # noqa: F401
    from .sampling import FermionConfigurationEncoding, MpsDiagonalEstimate, MpsBatchSampleResult, MpsSampleResult, MpsSampler, MpsStabSampler, PEPSSampleResult, PepsSampler, PepsBpSampler, StabilizerMpsSampler, TreeBatchSampleResult, TreeSampleResult, TreeSampler, VecSampler  # noqa: F401
    from .solvers import FDSolver  # noqa: F401
    from .tensors import (  # noqa: F401
        OneDMap,
        build_contraction,
        Fermion,
        FermionLatticeSetup,
        SpinfulFermion,
        SpinfulFermionHubbard,
        SymmFermions,
        SymGateStream,
        SymHamiltonian,
        SymMPS,
        SymPEPS,
        default_physical_sectors,
        draw_symmray_blocks,
        draw_symmray_mps,
        draw_symmray_mpo,
        draw_symmray_peps,
        fermi_hubbard_u1u1_gate_stream,
        fermi_hubbard_u1u1_hopping_gate_stream,
        fermi_hubbard_u1u1_interaction_gate_stream,
        fermi_hubbard_u1u1_light_pulse_gate_stream,
        fermi_hubbard_u1u1_jw_gate_stream,
        fermi_hubbard_u1u1_jw_hopping_gate_stream,
        fermi_hubbard_u1u1_jw_interaction_gate_stream,
        fermion_density_param_gen,
        fermion_hopping_param_gen,
        fermion_interaction_param_gen,
        bell_to_mps,
        expec_mpo,
        haar_random_state,
        hrs_to_mps,
        hrs_to_peps,
        hrs_to_ttn,
        hrps_to_mps,
        hrps_to_peps,
        hrps_to_ttn,
        id_to_mpo,
        id_to_pepo,
        measure_obs,
        mps_to_ttn,
        ps_to_3dpeps,
        ps_to_mpo,
        ps_to_mps,
        ps_to_ttn,
        ps_to_pepo,
        ps_to_peps,
        random_haar_qubit,
        reg_native_svd_jax,
        reg_native_svd_torch,
        reg_complex_qr_torch,
        reg_complex_svd_jax,
        reg_complex_svd_torch,
        reg_real_qr_torch,
        reg_real_svd_jax,
        reg_real_svd_torch,
        reg_rel_svd_jax,
        reg_rel_svd_torch,
        sector_index_map,
        site_charge_alternating,
        site_charge_from_map,
        site_charge_from_occupations,
        site_charge_uniform,
        symmray_block_summary,
        symmray_mps_summary,
        symmray_mpo_summary,
        symmray_peps_summary,
        symm_operator_from_dense,
        tn_fidelity,
        tn_norm,
        tns_align,
    )
