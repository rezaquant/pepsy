"""Tensor-network maps, constructors, contractions, and validation.

The tensor domains are exposed through a lazy compatibility namespace.  The
implementation is split into focused modules, while existing imports from
``pepsy.tensors`` continue to resolve unchanged.
"""

from importlib import import_module
import warnings


_SYMBOL_MODULES = {}


def _register(module, *names):
    for name in names:
        _SYMBOL_MODULES[name] = module


_register(
    ".maps",
    "OneDMap",
)
_register(
    ".symmetric",
    "FermionLatticeSetup",
    "SymGateStream",
    "SymHamiltonian",
    "SymMPS",
    "SymPEPS",
    "default_physical_sectors",
    "draw_symmray_blocks",
    "draw_symmray_mps",
    "draw_symmray_mpo",
    "draw_symmray_peps",
    "fermi_hubbard_u1u1_gate_stream",
    "fermi_hubbard_u1u1_hopping_gate_stream",
    "fermi_hubbard_u1u1_interaction_gate_stream",
    "fermi_hubbard_u1u1_light_pulse_gate_stream",
    "fermi_hubbard_u1u1_jw_gate_stream",
    "fermi_hubbard_u1u1_jw_hopping_gate_stream",
    "fermi_hubbard_u1u1_jw_interaction_gate_stream",
    "fermion_density_param_gen",
    "fermion_hopping_param_gen",
    "fermion_interaction_param_gen",
    "sector_index_map",
    "site_charge_alternating",
    "site_charge_from_map",
    "site_charge_from_occupations",
    "site_charge_uniform",
    "symmray_block_summary",
    "symmray_mps_summary",
    "symmray_mpo_summary",
    "symmray_peps_summary",
    "symm_operator_from_dense",
)
_register(
    ".symm_fermions",
    "Fermion",
    "SpinfulFermion",
    "SpinfulFermionHubbard",
    "SymmFermions",
)
_register(
    ".constructors",
    "add_cycle",
    "bell_to_mps",
    "expec_mpo",
    "haar_random_state",
    "hrs_to_mps",
    "hrs_to_peps",
    "hrs_to_ttn",
    "hrps_to_mps",
    "hrps_to_peps",
    "hrps_to_ttn",
    "id_to_mpo",
    "id_to_pepo",
    "ps_to_3dpeps",
    "ps_to_mpo",
    "ps_to_mps",
    "ps_to_ttn",
    "ps_to_pepo",
    "ps_to_peps",
    "random_haar_qubit",
    "tns_align",
)
_register(
    ".contractions",
    "build_contraction",
    "build_compressed_optimizer",
    "build_optimizer",
    "contract_hypercompressed_tn",
    "contract_hypercompressed_tn_batch",
    "tn_norm",
)
_register(".observables", "measure_obs", "tn_fidelity")
_register(".conversions", "mps_to_ttn")
_register(".validation", "validate_tensor_network_tags")
_register(
    "..backends.config",
    "backend_cupy",
    "backend_jax",
    "backend_numpy",
    "backend_torch",
    "build_backend",
    "get_default_array_backend",
    "get_default_grad_backend",
    "get_torch_linalg_config",
    "register_jax_linalg",
    "register_torch_linalg",
    "reset_linalg_registrations",
    "reset_default_backends",
    "set_default_array_backend",
    "set_default_grad_backend",
    "reg_complex_qr_torch",
    "reg_native_svd_jax",
    "reg_native_svd_torch",
    "reg_complex_svd_jax",
    "reg_complex_svd_torch",
    "reg_real_qr_torch",
    "reg_real_svd_jax",
    "reg_real_svd_torch",
    "reg_rel_svd_jax",
    "reg_rel_svd_torch",
    "reg_stop_gradient_torch",
    "stop_grad",
    "TorchLinalgConfig",
)

_SUBMODULES = (
    "constructors",
    "conversions",
    "contractions",
    "maps",
    "observables",
    "symmetric",
    "symm_fermions",
    "validation",
    "core",
)

_BACKEND_COMPATIBILITY_ALIASES = frozenset(
    {
        "backend_cupy",
        "backend_jax",
        "backend_numpy",
        "backend_torch",
        "build_backend",
        "get_default_array_backend",
        "get_default_grad_backend",
        "get_torch_linalg_config",
        "register_jax_linalg",
        "register_torch_linalg",
        "reset_default_backends",
        "reset_linalg_registrations",
        "set_default_array_backend",
        "set_default_grad_backend",
        "TorchLinalgConfig",
    }
)

_DEPRECATED_ALIASES = {
    **{name: f"pepsy.backends.{name}" for name in _BACKEND_COMPATIBILITY_ALIASES},
    "build_contraction": "pepsy.tensors.build_optimizer",
    "SpinfulFermionHubbard": "pepsy.tensors.SpinfulFermion",
    "hrps_to_mps": "pepsy.tensors.hrs_to_mps",
    "hrps_to_peps": "pepsy.tensors.hrs_to_peps",
    "hrps_to_ttn": "pepsy.tensors.hrs_to_ttn",
}

__all__ = [*_SYMBOL_MODULES, *_SUBMODULES]


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        canonical = _DEPRECATED_ALIASES.get(name)
        if canonical is not None:
            warnings.warn(
                f"pepsy.tensors.{name} is a compatibility alias; use {canonical} instead.",
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
