"""Tests for Q-Lithos VQE Simulation Engine."""

import pytest
from vqe_engine import run_vqe_simulation


def test_vqe_matches_exact_within_5_mha_for_lih():
    """Verify that VQE total ground state energy matches exact classical diagonalization

    within the strict 5 mHa threshold for LiH at equilibrium distance (1.595 A).
    """
    result = run_vqe_simulation(molecule_name="LiH", distance=1.595, maxiter=50)

    # Core assertion mandated by specification: error < 5 mHa
    assert (
        result.error_mha < 5.0
    ), f"VQE error ({result.error_mha:.4f} mHa) exceeds the 5.0 mHa threshold for LiH!"

    # Verify physical consistency
    assert result.converged, "VQE optimizer did not report convergence."
    assert result.qubit_count <= 8, f"Qubit count ({result.qubit_count}) exceeds max 8 qubits!"
    assert result.iterations > 0, "No iterations recorded."
    assert len(result.convergence_history) > 0, "Convergence history is empty."
    assert len(result.optimal_parameters) > 0, "Optimal parameters list is empty."

    # Energy should be in expected physical range for LiH (~ -8.5 to -8.6 Ha)
    assert -9.0 < result.vqe_energy < -8.0, f"Unreasonable LiH energy: {result.vqe_energy}"
    assert -9.0 < result.exact_energy < -8.0, f"Unreasonable exact LiH energy: {result.exact_energy}"


def test_vqe_qubit_limit_guard():
    """Verify that attempting to configure an excessive active space raises ValueError."""
    from molecules import MoleculeDefinition, MOLECULES, BatteryCategory

    # Temporarily create a synthetic molecule that would exceed 8 qubits
    oversized = MoleculeDefinition(
        name="OversizedMol",
        formula="X2",
        battery_role="Test",
        battery_category=BatteryCategory.CALCULATION_TEST_CASE,
        description="Test",
        atom_symbols=("Li", "H"),
        equilibrium_distance=1.5,
        num_active_electrons=8,
        num_active_orbitals=8,  # 16 spin orbitals -> >8 qubits
        max_qubits=8,
    )
    MOLECULES["OversizedMol"] = oversized

    try:
        with pytest.raises(Exception):
            run_vqe_simulation("OversizedMol", 1.5)
    finally:
        del MOLECULES["OversizedMol"]
