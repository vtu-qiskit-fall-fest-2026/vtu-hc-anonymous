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
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


MAX_QUBIT_BUDGET = 8


@dataclass(frozen=True)
class MoleculeDefinition:
    """Specification of a battery molecule fragment."""

    name: str
    formula: str
    battery_role: str
    description: str
    atom_symbols: tuple[str, str]
    equilibrium_distance: float  # Angstroms
    scan_distances: List[float] = field(default_factory=list)
    num_active_electrons: int = 2
    num_active_orbitals: int = 2  # Spatial orbitals
    max_qubits: int = MAX_QUBIT_BUDGET

    def build_atom_string(self, distance: float) -> str:
        """Generate PySCF-compatible atom geometry string for a given bond distance (Angstroms).

        Parameters
        ----------
        distance : float
            Interatomic separation in Angstroms.

        Returns
        -------
        str
            Semicolon-separated atomic coordinate string, e.g. 'Li 0 0 0; H 0 0 1.595'.
        """
        sym1, sym2 = self.atom_symbols
        return f"{sym1} 0.0 0.0 0.0; {sym2} 0.0 0.0 {distance:.6f}"


# 8-12 bond distances sampled across compression, equilibrium, and dissociation regimes
MOLECULES: Dict[str, MoleculeDefinition] = {
    "LiH": MoleculeDefinition(
        name="LiH",
        formula="LiH",
        battery_role="Solid Electrolyte Interphase (SEI) Passivation",
        description=(
            "Lithium hydride is an essential decomposition byproduct in solid-state lithium battery "
            "electrolyte interphases. Simulating its dissociation and binding characteristics gives "
            "direct insight into SEI interfacial stability and ion transport resistance."
        ),
        atom_symbols=("Li", "H"),
        equilibrium_distance=1.595,
        scan_distances=[1.0, 1.2, 1.4, 1.595, 1.8, 2.0, 2.3, 2.6, 3.0, 3.4],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
    ),
    "NaH": MoleculeDefinition(
        name="NaH",
        formula="NaH",
        battery_role="Sodium-Ion Hydride Carrier / Anode Interface",
        description=(
            "Sodium hydride plays a pivotal role in room-temperature sodium metal batteries and hydride "
            "solid electrolytes. Measuring its potential energy profile informs sodium-ion detachment barriers."
        ),
        atom_symbols=("Na", "H"),
        equilibrium_distance=1.887,
        scan_distances=[1.2, 1.4, 1.6, 1.887, 2.1, 2.4, 2.7, 3.0, 3.4, 3.8],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
    ),
    "NaF": MoleculeDefinition(
        name="NaF",
        formula="NaF",
        battery_role="Fluorinated SEI Mechanically Stable Boundary",
        description=(
            "Sodium fluoride forms a dense, mechanically robust passivation layer on sodium anodes that "
            "suppresses runaway parasitic electrolyte degradation. Understanding its bond strength provides "
            "proxies for interfacial integrity under high charging rates."
        ),
        atom_symbols=("Na", "F"),
        equilibrium_distance=1.926,
        scan_distances=[1.3, 1.5, 1.7, 1.926, 2.2, 2.5, 2.8, 3.2, 3.6, 4.0],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
    ),
    "Li2": MoleculeDefinition(
        name="Li2",
        formula="Li2",
        battery_role="Lithium Metal Nucleation & Dendrite Seed Fragment",
        description=(
            "Diatomic lithium represents the fundamental nucleus for metallic lithium plating on solid "
            "electrolytes. Simulating its cohesion provides an energetic baseline for dendrite nucleation tendencies."
        ),
        atom_symbols=("Li", "Li"),
        equilibrium_distance=2.673,
        scan_distances=[1.8, 2.1, 2.4, 2.673, 3.0, 3.3, 3.7, 4.1, 4.6, 5.0],
        num_active_electrons=2,
        num_active_orbitals=2,
        max_qubits=MAX_QUBIT_BUDGET,
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
    -------
    KeyError
        If molecule name is not registered.
    """
    if name not in MOLECULES:
        available = list(MOLECULES.keys())
        raise KeyError(f"Unknown molecule '{name}'. Available: {available}")
    return MOLECULES[name]
