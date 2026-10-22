"""Derived quantum metrics for battery materials screening in Q-Lithos.

IMPORTANT SCIENTIFIC GOVERNANCE & PROXY DISCLAIMER:
---------------------------------------------------
All indicators derived in this module are quantum-mechanical molecular proxies
computed from gas-phase / cluster potential energy surfaces (PES).
They provide rapid comparative trends for small battery electrolyte/electrode fragments.

* NEVER call these metrics true ion mobility, bulk ionic conductivity,
  true thermal stability, or macroscopic battery cycle life.
* Every metric here is explicitly named and labeled as a PROXY.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# Physical constants
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.509474


@dataclass
class BatteryMaterialMetrics:
    """Quantitative proxy metrics derived from quantum potential energy surface calculations."""

    molecule_name: str
    equilibrium_bond_length_proxy_angstrom: float
    binding_energy_proxy_hartree: float
    binding_energy_proxy_ev: float
    binding_energy_proxy_kcal_mol: float
    harmonic_force_constant_proxy_ha_per_ang2: float
    energy_gap_proxy_ev: float
    average_vqe_error_proxy_mha: float
    max_vqe_error_proxy_mha: float

    def to_dict(self) -> Dict[str, float | str]:
        """Convert metrics to a dictionary for display and serialization."""
        return {
            "molecule_name": self.molecule_name,
            "equilibrium_bond_length_proxy_angstrom": round(self.equilibrium_bond_length_proxy_angstrom, 4),
            "binding_energy_proxy_hartree": round(self.binding_energy_proxy_hartree, 6),
            "binding_energy_proxy_ev": round(self.binding_energy_proxy_ev, 4),
            "binding_energy_proxy_kcal_mol": round(self.binding_energy_proxy_kcal_mol, 2),
            "harmonic_force_constant_proxy_ha_per_ang2": round(self.harmonic_force_constant_proxy_ha_per_ang2, 4),
            "energy_gap_proxy_ev": round(self.energy_gap_proxy_ev, 4),
            "average_vqe_error_proxy_mha": round(self.average_vqe_error_proxy_mha, 4),
            "max_vqe_error_proxy_mha": round(self.max_vqe_error_proxy_mha, 4),
        }


def derive_battery_metrics(
    molecule_name: str,
    distances: List[float],
    vqe_energies: List[float],
    exact_energies: List[float],
) -> BatteryMaterialMetrics:
    """Derive quantum proxy metrics from potential energy curves.

    Parameters
    ----------
    molecule_name : str
        Identifier of the molecule (e.g. 'LiH', 'NaH', 'NaF', 'Li2').
    distances : List[float]
        Interatomic distances sampled across the scan (Angstroms).
    vqe_energies : List[float]
        Total ground state energies computed via VQE (Hartree).
    exact_energies : List[float]
        Benchmark energies computed via exact diagonalization (Hartree).

    Returns
    -------
    BatteryMaterialMetrics
        Derived metrics with explicit 'proxy' labeling across all attributes.

    Raises
    -------
    ValueError
        If input arrays are empty or lengths do not match.
    """
    if len(distances) < 3 or len(vqe_energies) != len(distances) or len(exact_energies) != len(distances):
        raise ValueError("Requires at least 3 sampled distances with matching energy arrays.")

    r_arr = np.array(distances, dtype=np.float64)
    vqe_arr = np.array(vqe_energies, dtype=np.float64)
    exact_arr = np.array(exact_energies, dtype=np.float64)

    # 1. Equilibrium Bond Length Proxy (R_eq)
    # Find minimum index and perform local parabolic interpolation for sub-grid precision
    min_idx = int(np.argmin(vqe_arr))
    if 0 < min_idx < len(r_arr) - 1:
        # Fit 2nd order polynomial over 3 neighboring points: E(R) = a*R^2 + b*R + c
        pts_r = r_arr[min_idx - 1 : min_idx + 2]
        pts_e = vqe_arr[min_idx - 1 : min_idx + 2]
        poly = np.polyfit(pts_r, pts_e, 2)
        a, b, c = poly[0], poly[1], poly[2]
        if a > 0:  # Valid local minimum
            r_eq_proxy = float(-b / (2.0 * a))
            e_min_proxy = float(a * r_eq_proxy**2 + b * r_eq_proxy + c)
            k_force_proxy = float(2.0 * a)  # d^2E/dR^2 in Ha / A^2
        else:
            r_eq_proxy = float(r_arr[min_idx])
            e_min_proxy = float(vqe_arr[min_idx])
            k_force_proxy = 0.0
    else:
        r_eq_proxy = float(r_arr[min_idx])
        e_min_proxy = float(vqe_arr[min_idx])
        k_force_proxy = 0.0

    # 2. Binding / Dissociation Energy Proxy (D_e)
    # Computed as the depth of the potential well: E(asymptotic dissociated) - E(equilibrium)
    e_dissociated = float(vqe_arr[-1])
    binding_energy_ha = max(0.0, e_dissociated - e_min_proxy)
    binding_energy_ev = binding_energy_ha * HARTREE_TO_EV
    binding_energy_kcal_mol = binding_energy_ha * HARTREE_TO_KCAL_MOL

    # 3. Energy-Gap Proxy
    # Serves as a cluster-level proxy related to electronic rigidity / dissociation resistance
    energy_gap_proxy_ev = binding_energy_ev

    # 4. VQE Error Statistics (VQE vs Exact in mHa)
    errors_mha = np.abs(vqe_arr - exact_arr) * 1000.0
    avg_error_mha = float(np.mean(errors_mha))
    max_error_mha = float(np.max(errors_mha))

    return BatteryMaterialMetrics(
        molecule_name=molecule_name,
        equilibrium_bond_length_proxy_angstrom=r_eq_proxy,
        binding_energy_proxy_hartree=binding_energy_ha,
        binding_energy_proxy_ev=binding_energy_ev,
        binding_energy_proxy_kcal_mol=binding_energy_kcal_mol,
        harmonic_force_constant_proxy_ha_per_ang2=k_force_proxy,
        energy_gap_proxy_ev=energy_gap_proxy_ev,
        average_vqe_error_proxy_mha=avg_error_mha,
        max_vqe_error_proxy_mha=max_error_mha,
    )
