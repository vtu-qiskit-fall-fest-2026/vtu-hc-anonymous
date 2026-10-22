"""Molecular definitions for battery materials simulation in Q-Lithos.

This module defines small molecular fragments relevant to solid-state and sodium-ion
battery systems:
- LiH: Lithium Hydride (passivation & solid electrolyte interface layer)
- NaH: Sodium Hydride (carrier dynamics in sodium solid-state batteries)
- NaF: Sodium Fluoride (dense SEI passivation in sodium batteries)
- Li2: Dilithium (elementary lithium metal nucleation & dendrite seed)

Each molecule defines:
- Atomic geometry parameterized by interatomic distance R (Angstroms)
- Equilibrium bond distance
- Scan distances (8-12 points)
- Active space specification (active electrons, active spatial orbitals)
- Maximum qubit budget (<= 8 qubits)
- Battery research context (category, known experimental properties)

SCIENTIFIC CATEGORY SYSTEM
---------------------------
Three categories are explicitly distinguished for scientific integrity:

  CALCULATION_TEST_CASE:
    A molecule selected primarily because it is tractable for the quantum
    chemistry workflow. Its battery relevance is incidental.

  BATTERY_RELEVANT_MOLECULE:
    A molecule known from the scientific literature to appear in battery
    chemistry contexts (e.g., as an SEI component, electrolyte decomposition
    product, or electrode interface species). Its performance has not been
    measured in this application.

  BATTERY_PERFORMANCE_MEASURED:
    A material whose electrochemical performance (capacity, cycle life,
    voltage, etc.) has been experimentally characterized and published.
    None of the current candidates reach this category because the present
    quantum chemistry calculations are on gas-phase diatomic fragments, not
    bulk electrode/electrolyte materials.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


MAX_QUBIT_BUDGET = 8

# Physical constants used in unit annotations
HARTREE_TO_EV = 27.211386245988  # NIST 2018 CODATA
HARTREE_TO_KCAL_MOL = 627.509474  # 1 Ha = 627.509474 kcal/mol


class BatteryCategory:
    """Scientific category labels for molecular candidates."""
    CALCULATION_TEST_CASE = "Quantum Chemistry Test Case"
    BATTERY_RELEVANT_MOLECULE = "Battery-Relevant Molecule (literature)"
    BATTERY_PERFORMANCE_MEASURED = "Battery Material (experimentally characterized)"


@dataclass(frozen=True)
class MoleculeDefinition:
    """Specification of a battery molecule fragment.

    Attributes
    ----------
    name : str
        Short identifier used as a lookup key (e.g. 'LiH').
    formula : str
        Chemical formula for display (e.g. 'LiH', 'Li₂').
    battery_role : str
        Brief description of the molecule's role in battery systems.
    description : str
        Longer plain-language description of battery relevance.
    battery_category : str
        One of the BatteryCategory constants.  Distinguishes a test-case
        molecule from a battery-relevant molecule from a material with
        measured battery performance.
    atom_symbols : tuple[str, str]
        Element symbols for the two atoms.
    equilibrium_distance : float
        Literature / experimental equilibrium bond length in Angstroms.
    literature_source : str
        Citation for the equilibrium distance.
    scan_distances : List[float]
        Bond lengths sampled in the potential energy scan (Angstroms).
    num_active_electrons : int
        Number of electrons in the active space.
    num_active_orbitals : int
        Number of spatial orbitals in the active space.
    max_qubits : int
        Maximum qubit budget (must not exceed MAX_QUBIT_BUDGET = 8).
    charge : int
        Total molecular charge (0 = neutral).
    spin_multiplicity : int
        2S + 1 (1 = singlet, 2 = doublet, 3 = triplet).
    basis_set : str
        Gaussian basis set used for the electronic structure calculation.
    known_properties : dict
        Experimentally known or literature-cited properties relevant to
        battery research.  Values are (value, unit, source) tuples or None.
    battery_caveats : str
        Explicit statement of what the quantum calculation CANNOT establish
        about battery performance for this molecule.
    """

    name: str
    formula: str
    battery_role: str
    description: str
    battery_category: str
    atom_symbols: tuple
    equilibrium_distance: float  # Angstroms
    literature_source: str = "NIST Diatomic Spectroscopic Database"
    scan_distances: List[float] = field(default_factory=list)
    num_active_electrons: int = 2
    num_active_orbitals: int = 2  # Spatial orbitals
    max_qubits: int = MAX_QUBIT_BUDGET
    charge: int = 0
    spin_multiplicity: int = 1
    basis_set: str = "STO-3G"
    known_properties: dict = field(default_factory=dict)
    battery_caveats: str = ""

    def build_atom_string(self, distance: float) -> str:
        """Generate PySCF-compatible atom geometry string for a given bond distance.

        Parameters
        ----------
        distance : float
            Interatomic separation in Angstroms.

        Returns
        -------
        str
            Semicolon-separated atomic coordinate string, e.g.
            'Li 0 0 0; H 0 0 1.595'.
        """
        sym1, sym2 = self.atom_symbols
        return f"{sym1} 0.0 0.0 0.0; {sym2} 0.0 0.0 {distance:.6f}"

    @property
    def qubit_count_expected(self) -> int:
        """Expected qubit count after ParityMapper 2-qubit reduction.

        For a (2e, 2o) active space with parity mapping + 2-qubit reduction:
        2 * num_active_orbitals spin-orbitals - 2 = 2 qubits for all current
        candidates.
        """
        return 2 * self.num_active_orbitals - 2

    @property
    def category_label(self) -> str:
        """Short display label for the battery category."""
        labels = {
            BatteryCategory.CALCULATION_TEST_CASE: "Test Case",
            BatteryCategory.BATTERY_RELEVANT_MOLECULE: "Battery-Relevant",
            BatteryCategory.BATTERY_PERFORMANCE_MEASURED: "Experimentally Measured",
        }
        return labels.get(self.battery_category, self.battery_category)


# ---------------------------------------------------------------------------
# MOLECULE LIBRARY
# 8-12 bond distances sampled across compression, equilibrium, and dissociation
# ---------------------------------------------------------------------------
MOLECULES: Dict[str, MoleculeDefinition] = {
    "LiH": MoleculeDefinition(
        name="LiH",
        formula="LiH",
        battery_role="Solid Electrolyte Interphase (SEI) Passivation Byproduct",
        battery_category=BatteryCategory.BATTERY_RELEVANT_MOLECULE,
        description=(
            "Lithium hydride (LiH) is an established decomposition byproduct found in the "
            "solid electrolyte interphase (SEI) of lithium-metal and lithium-ion batteries. "
            "It forms at the anode surface when organic electrolytes or fluorinated binders "
            "react with metallic lithium. Its bond strength provides a molecular-level proxy "
            "for the energetic stability of this SEI component under repeated cycling. "
            "Note: this calculation models an isolated gas-phase LiH molecule, not the bulk "
            "polycrystalline LiH found in real SEI layers."
        ),
        battery_caveats=(
            "The gas-phase dissociation energy of LiH is NOT the same as the mechanical "
            "stability of an SEI layer, the electrochemical window, or the ionic conductivity "
            "of the interphase. Bulk LiH has an experimentally measured electrochemical "
            "stability window, but that quantity requires periodic solid-state calculations, "
            "not gas-phase diatomic VQE."
        ),
        atom_symbols=("Li", "H"),
        equilibrium_distance=1.5957,
        literature_source="Huber & Herzberg, Constants of Diatomic Molecules (1979); NIST Webbook",
        scan_distances=[1.0, 1.2, 1.4, 1.595, 1.8, 2.0, 2.3, 2.6, 3.0, 3.4],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
        charge=0,
        spin_multiplicity=1,
        basis_set="STO-3G",
        known_properties={
            "equilibrium_bond_length": (1.5957, "Å", "NIST/Huber & Herzberg 1979"),
            "dissociation_energy": (2.515, "eV", "Huber & Herzberg 1979; D₀ = 2.429 eV"),
            "vibrational_frequency": (1405.65, "cm⁻¹", "NIST Webbook"),
        },
    ),
    "NaH": MoleculeDefinition(
        name="NaH",
        formula="NaH",
        battery_role="Sodium-Ion Carrier / Sodium-Metal Anode Interface Species",
        battery_category=BatteryCategory.BATTERY_RELEVANT_MOLECULE,
        description=(
            "Sodium hydride (NaH) appears at sodium-metal anode interfaces in sodium-ion "
            "and sodium-metal battery systems. It can form as a parasitic decomposition "
            "product of ether and carbonate electrolytes contacting the Na anode. "
            "Understanding its molecular bond characteristics provides a basis for comparing "
            "Na-ion detachment energetics across candidate electrolyte families. "
            "This calculation is a gas-phase diatomic model only."
        ),
        battery_caveats=(
            "Gas-phase NaH dissociation energy does not predict sodium-ion mobility, "
            "ionic conductivity, or the electrochemical stability of NaH in a real electrolyte. "
            "Bulk NaH adopts a rock-salt crystal structure; its solid-state properties differ "
            "substantially from those of the isolated molecule modeled here."
        ),
        atom_symbols=("Na", "H"),
        equilibrium_distance=1.8873,
        literature_source="Huber & Herzberg, Constants of Diatomic Molecules (1979); NIST Webbook",
        scan_distances=[1.2, 1.4, 1.6, 1.887, 2.1, 2.4, 2.7, 3.0, 3.4, 3.8],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
        charge=0,
        spin_multiplicity=1,
        basis_set="STO-3G",
        known_properties={
            "equilibrium_bond_length": (1.8873, "Å", "NIST/Huber & Herzberg 1979"),
            "dissociation_energy": (1.90, "eV", "Huber & Herzberg 1979 (estimated D₀)"),
            "vibrational_frequency": (1172.2, "cm⁻¹", "NIST Webbook"),
        },
    ),
    "NaF": MoleculeDefinition(
        name="NaF",
        formula="NaF",
        battery_role="Fluorinated SEI Passivation Layer Component",
        battery_category=BatteryCategory.BATTERY_RELEVANT_MOLECULE,
        description=(
            "Sodium fluoride (NaF) forms a mechanically robust, ionically resistive passivation "
            "layer on sodium anodes when fluorinated additives or solvents are decomposed. "
            "NaF-rich SEI layers are known to suppress parasitic electrolyte oxidation and can "
            "improve coulombic efficiency in sodium-metal batteries. "
            "The gas-phase bond strength of NaF provides a molecular baseline for comparing "
            "Na–F bond cohesion across fluorinated SEI species. "
            "This model is a gas-phase diatomic approximation."
        ),
        battery_caveats=(
            "The gas-phase dissociation energy of NaF cannot be interpreted as the mechanical "
            "strength of a NaF SEI coating, the ionic conductivity of the passivation layer, "
            "or the electrochemical stability window. NaF in a real battery exists in a "
            "polycrystalline or amorphous solid environment with substantially different "
            "electronic structure from the isolated molecule computed here."
        ),
        atom_symbols=("Na", "F"),
        equilibrium_distance=1.9260,
        literature_source="Huber & Herzberg, Constants of Diatomic Molecules (1979); NIST Webbook",
        scan_distances=[1.3, 1.5, 1.7, 1.926, 2.2, 2.5, 2.8, 3.2, 3.6, 4.0],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
        charge=0,
        spin_multiplicity=1,
        basis_set="STO-3G",
        known_properties={
            "equilibrium_bond_length": (1.9260, "Å", "NIST/Huber & Herzberg 1979"),
            "dissociation_energy": (4.99, "eV", "Huber & Herzberg 1979 (D₀ = 4.86 eV)"),
            "vibrational_frequency": (536.0, "cm⁻¹", "NIST Webbook"),
        },
    ),
    "Li2": MoleculeDefinition(
        name="Li2",
        formula="Li₂",
        battery_role="Lithium Metal Nucleation Seed / Dendrite Precursor Fragment",
        battery_category=BatteryCategory.CALCULATION_TEST_CASE,
        description=(
            "Diatomic lithium (Li₂) serves primarily as a well-characterized quantum chemistry "
            "benchmark system. While it represents the smallest lithium–lithium bonding unit, "
            "its relevance to battery dendrite formation is indirect: real lithium dendrites "
            "grow as bulk metallic lithium, not as diatomic molecules. "
            "Li₂ is included here as a computationally tractable test of the VQE pipeline "
            "for a homonuclear system, where the exact diagonalization reference is readily "
            "available. Its multi-reference character at stretched geometries makes it a "
            "useful benchmark for UCCSD accuracy."
        ),
        battery_caveats=(
            "Li₂ is categorized as a quantum chemistry test case, not a battery material. "
            "The gas-phase Li–Li dissociation energy does not predict lithium plating kinetics, "
            "dendrite nucleation barriers, or metal anode Coulombic efficiency. "
            "These quantities require periodic DFT or force-field-based molecular dynamics "
            "on bulk lithium metal slabs, which are beyond the scope of this prototype."
        ),
        atom_symbols=("Li", "Li"),
        equilibrium_distance=2.6731,
        literature_source="Huber & Herzberg, Constants of Diatomic Molecules (1979); NIST Webbook",
        scan_distances=[1.8, 2.1, 2.4, 2.673, 3.0, 3.3, 3.7, 4.1, 4.6, 5.0],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
        charge=0,
        spin_multiplicity=1,
        basis_set="STO-3G",
        known_properties={
            "equilibrium_bond_length": (2.6731, "Å", "NIST/Huber & Herzberg 1979"),
            "dissociation_energy": (1.046, "eV", "Huber & Herzberg 1979 (D₀ = 1.034 eV)"),
            "vibrational_frequency": (351.43, "cm⁻¹", "NIST Webbook"),
        },
    ),
}


def get_molecule(name: str) -> MoleculeDefinition:
    """Retrieve molecule definition by name.

    Parameters
    ----------
    name : str
        Molecule key (e.g. 'LiH', 'NaH', 'NaF', 'Li2').

    Returns
    -------
    MoleculeDefinition
        The corresponding molecule definition.

    Raises
    ------
    KeyError
        If molecule name is not registered.
    """
    if name not in MOLECULES:
        available = list(MOLECULES.keys())
        raise KeyError(f"Unknown molecule '{name}'. Available: {available}")
    return MOLECULES[name]
