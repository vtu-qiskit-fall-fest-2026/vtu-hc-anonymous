"""Derived quantum metrics for battery materials screening in Q-Lithos.

SCIENTIFIC GOVERNANCE & PROXY DISCLAIMER
-----------------------------------------
All indicators derived in this module are quantum-mechanical molecular proxies
computed from gas-phase / cluster potential energy surfaces (PES).
They provide rapid comparative trends for small battery electrolyte/electrode fragments.

NEVER call these metrics:
  * true ion mobility
  * bulk ionic conductivity
  * true thermal stability
  * macroscopic battery cycle life
  * battery voltage or capacity

Every metric is explicitly named and labeled as a PROXY.

Physical constants (NIST 2018 CODATA):
  1 Hartree = 27.211386245988 eV
  1 Hartree = 627.509474 kcal/mol
  1 kcal/mol = 4.184 kJ/mol
  Chemical accuracy threshold: 1 kcal/mol ≈ 1.5936 mHa
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# Physical constants (NIST 2018 CODATA)
HARTREE_TO_EV: float = 27.211386245988
HARTREE_TO_KCAL_MOL: float = 627.509474
KCAL_TO_KJ: float = 4.184

# Chemical accuracy threshold: 1 kcal/mol expressed in milli-Hartree
# 1 kcal/mol = 1/627.509474 Ha = 1.5936 mHa (to 4 significant figures)
CHEMICAL_ACCURACY_KCAL_MOL: float = 1.0
CHEMICAL_ACCURACY_MHA: float = 1000.0 / HARTREE_TO_KCAL_MOL  # ≈ 1.5936 mHa


@dataclass
class BatteryMaterialMetrics:
    """Quantitative proxy metrics derived from quantum potential energy surface calculations.

    All attributes whose names contain 'proxy' are molecular-level descriptors
    computed from gas-phase diatomic PES calculations.  They must not be
    interpreted as macroscopic battery performance indicators.

    Attributes
    ----------
    molecule_name : str
        Molecule identifier.
    equilibrium_bond_length_proxy_angstrom : float
        Bond length at the potential energy minimum (Å), estimated by parabolic
        interpolation.  This is a gas-phase equilibrium geometry; it does NOT
        represent bulk crystal packing density.
    dissociation_energy_proxy_hartree : float
        Potential well depth:  E(R_dissociated) - E(R_eq) in Hartree.
        This is the gas-phase bond dissociation energy of the diatomic molecule.
        It does NOT represent electrochemical stability, ionic conductivity,
        or battery voltage.
    dissociation_energy_proxy_ev : float
        Same quantity in electronvolts.
    dissociation_energy_proxy_kcal_mol : float
        Same quantity in kcal/mol.
    harmonic_force_constant_proxy_ha_per_ang2 : float
        Local potential well curvature at equilibrium: d²E/dR² in Ha/Å².
        Proportional to the harmonic vibrational frequency of the isolated
        molecule.  It does NOT measure mechanical rigidity of a bulk material.
    average_vqe_error_proxy_mha : float
        Mean absolute VQE–exact error across all scan points in mHa.
    max_vqe_error_proxy_mha : float
        Maximum VQE–exact error across all scan points in mHa.
    chemical_accuracy_achieved : bool
        True only if ALL scan points have error < CHEMICAL_ACCURACY_MHA (≈ 1.5936 mHa).
    energy_gap_proxy_ev : None
        Intentionally always None.  The dissociation well depth is NOT an
        electronic energy gap (HOMO–LUMO gap).  These are different physical
        quantities requiring different methods.  A HOMO–LUMO gap requires an
        orbital energy calculation, not a PES scan.
    """

    molecule_name: str

    # Geometry proxy
    equilibrium_bond_length_proxy_angstrom: float

    # Dissociation energy proxy (formerly mislabeled as "binding energy proxy")
    # Renamed for scientific accuracy: this is D_e (well depth), not binding energy
    # in the solid-state sense.
    dissociation_energy_proxy_hartree: float
    dissociation_energy_proxy_ev: float
    dissociation_energy_proxy_kcal_mol: float

    # Mechanical proxy
    harmonic_force_constant_proxy_ha_per_ang2: float

    # VQE accuracy metrics
    average_vqe_error_proxy_mha: float
    max_vqe_error_proxy_mha: float
    chemical_accuracy_achieved: bool

    # Intentionally None: dissociation well depth is NOT an electronic gap
    energy_gap_proxy_ev: Optional[float] = None

    # Legacy aliases kept for backward cache compatibility
    @property
    def binding_energy_proxy_hartree(self) -> float:
        """Legacy alias for dissociation_energy_proxy_hartree."""
        return self.dissociation_energy_proxy_hartree

    @property
    def binding_energy_proxy_ev(self) -> float:
        """Legacy alias for dissociation_energy_proxy_ev."""
        return self.dissociation_energy_proxy_ev

    @property
    def binding_energy_proxy_kcal_mol(self) -> float:
        """Legacy alias for dissociation_energy_proxy_kcal_mol."""
        return self.dissociation_energy_proxy_kcal_mol

    def to_dict(self) -> Dict:
        """Convert metrics to a dictionary for display and serialization.

        Note: 'energy_gap_proxy_ev' is intentionally omitted because it is
        always None and should never be used as a metric.
        """
        return {
            "molecule_name": self.molecule_name,
            "equilibrium_bond_length_proxy_angstrom": round(
                self.equilibrium_bond_length_proxy_angstrom, 4
            ),
            "dissociation_energy_proxy_hartree": round(
                self.dissociation_energy_proxy_hartree, 6
            ),
            "dissociation_energy_proxy_ev": round(self.dissociation_energy_proxy_ev, 4),
            "dissociation_energy_proxy_kcal_mol": round(
                self.dissociation_energy_proxy_kcal_mol, 2
            ),
            "harmonic_force_constant_proxy_ha_per_ang2": round(
                self.harmonic_force_constant_proxy_ha_per_ang2, 4
            ),
            "average_vqe_error_proxy_mha": round(self.average_vqe_error_proxy_mha, 5),
            "max_vqe_error_proxy_mha": round(self.max_vqe_error_proxy_mha, 5),
            "chemical_accuracy_achieved": self.chemical_accuracy_achieved,
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
        The last point is taken as the dissociated-limit reference.
    vqe_energies : List[float]
        Total ground state energies computed via VQE (Hartree).
    exact_energies : List[float]
        Benchmark energies computed via exact diagonalization (Hartree).

    Returns
    -------
    BatteryMaterialMetrics
        Derived metrics with explicit 'proxy' labeling.

    Raises
    ------
    ValueError
        If input arrays have fewer than 3 points or lengths do not match.
    """
    if len(distances) < 3:
        raise ValueError(
            f"Requires at least 3 sampled distances; got {len(distances)}."
        )
    if len(vqe_energies) != len(distances) or len(exact_energies) != len(distances):
        raise ValueError(
            "Lengths of distances, vqe_energies, and exact_energies must match."
        )

    r_arr = np.array(distances, dtype=np.float64)
    vqe_arr = np.array(vqe_energies, dtype=np.float64)
    exact_arr = np.array(exact_energies, dtype=np.float64)

    # -----------------------------------------------------------------------
    # 1. Equilibrium Bond Length Proxy (R_eq)
    #    Find minimum of VQE PES; refine with local parabolic interpolation
    # -----------------------------------------------------------------------
    min_idx = int(np.argmin(vqe_arr))
    if 0 < min_idx < len(r_arr) - 1:
        # Fit 2nd-order polynomial over 3 neighboring points: E(R) = a*R² + b*R + c
        pts_r = r_arr[min_idx - 1 : min_idx + 2]
        pts_e = vqe_arr[min_idx - 1 : min_idx + 2]
        poly = np.polyfit(pts_r, pts_e, 2)
        a, b, _ = poly[0], poly[1], poly[2]
        if a > 0:  # Valid local minimum (upward-opening parabola)
            r_eq_proxy = float(-b / (2.0 * a))
            e_min_proxy = float(np.polyval(poly, r_eq_proxy))
            k_force_proxy = float(2.0 * a)  # d²E/dR² in Ha/Å²
        else:
            r_eq_proxy = float(r_arr[min_idx])
            e_min_proxy = float(vqe_arr[min_idx])
            k_force_proxy = 0.0
    else:
        r_eq_proxy = float(r_arr[min_idx])
        e_min_proxy = float(vqe_arr[min_idx])
        k_force_proxy = 0.0

    # -----------------------------------------------------------------------
    # 2. Dissociation Energy Proxy (D_e)
    #    Depth of the PES well: E(asymptotic/dissociated limit) - E(minimum)
    #    We use the last (largest-distance) point as the dissociation limit.
    #    This is a gas-phase molecular dissociation energy — NOT bulk cohesive
    #    energy, electrochemical stability, or battery voltage.
    # -----------------------------------------------------------------------
    e_dissociated = float(vqe_arr[-1])
    dissociation_ha = max(0.0, e_dissociated - e_min_proxy)
    dissociation_ev = dissociation_ha * HARTREE_TO_EV
    dissociation_kcal = dissociation_ha * HARTREE_TO_KCAL_MOL

    # -----------------------------------------------------------------------
    # 3. VQE Accuracy Statistics
    #    VQE error relative to exact diagonalization (mHa)
    #    Chemical accuracy: ALL points must be < 1 kcal/mol ≈ 1.5936 mHa
    # -----------------------------------------------------------------------
    errors_mha = np.abs(vqe_arr - exact_arr) * 1000.0
    avg_error_mha = float(np.mean(errors_mha))
    max_error_mha = float(np.max(errors_mha))
    # Chemical accuracy is achieved only when EVERY point is within threshold
    chemical_accuracy = bool(max_error_mha < CHEMICAL_ACCURACY_MHA)

    return BatteryMaterialMetrics(
        molecule_name=molecule_name,
        equilibrium_bond_length_proxy_angstrom=r_eq_proxy,
        dissociation_energy_proxy_hartree=dissociation_ha,
        dissociation_energy_proxy_ev=dissociation_ev,
        dissociation_energy_proxy_kcal_mol=dissociation_kcal,
        harmonic_force_constant_proxy_ha_per_ang2=k_force_proxy,
        energy_gap_proxy_ev=None,  # Intentionally never set
        average_vqe_error_proxy_mha=avg_error_mha,
        max_vqe_error_proxy_mha=max_error_mha,
        chemical_accuracy_achieved=chemical_accuracy,
    )
