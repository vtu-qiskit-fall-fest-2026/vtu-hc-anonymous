"""Tests for molecule definitions and constraints."""

from molecules import MOLECULES, get_molecule, MAX_QUBIT_BUDGET


def test_registered_molecules():
    """Verify all 4 required molecules exist."""
    expected = {"LiH", "NaH", "NaF", "Li2"}
    assert expected.issubset(set(MOLECULES.keys())), f"Missing molecules! Registered: {list(MOLECULES.keys())}"


def test_molecule_specifications():
    """Verify properties of each registered battery material."""
    for name in ["LiH", "NaH", "NaF", "Li2"]:
        mol = get_molecule(name)
        assert mol.name == name
        assert mol.equilibrium_distance > 0.5
        assert len(mol.atom_symbols) == 2
        # Verify 8-12 scan distances per requirement
        assert 8 <= len(mol.scan_distances) <= 12, f"{name} has {len(mol.scan_distances)} points (expected 8-12)"
        # Verify qubit budget does not exceed 8
        assert mol.max_qubits <= MAX_QUBIT_BUDGET


def test_build_atom_string():
    """Verify geometry formatting for PySCF driver."""
    mol = get_molecule("LiH")
    atom_str = mol.build_atom_string(1.6)
    assert "Li 0.0 0.0 0.0" in atom_str
    assert "H 0.0 0.0 1.6" in atom_str
    assert ";" in atom_str
