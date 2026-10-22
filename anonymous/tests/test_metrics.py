"""Tests for quantitative proxy metrics and compliance labeling."""

from metrics import derive_battery_metrics, BatteryMaterialMetrics


def test_derive_battery_metrics():
    """Verify proxy metrics derivation and explicit proxy labeling."""
    distances = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.4]
    # Sample harmonic well around 1.6 A
    exact_energies = [
        -8.531675, -8.549896, -8.553768, -8.545237, -8.525905,
        -8.499926, -8.453338, -8.402500, -8.333326, -8.265908
    ]
    # Synthetic VQE with tiny error
    vqe_energies = [e + 0.000001 for e in exact_energies]

    metrics = derive_battery_metrics("LiH", distances, vqe_energies, exact_energies)

    assert isinstance(metrics, BatteryMaterialMetrics)
    # Check that equilibrium bond length is near ~1.4 A
    assert 1.2 < metrics.equilibrium_bond_length_proxy_angstrom < 1.6
    # Check binding energy is positive
    assert metrics.binding_energy_proxy_hartree > 0
    assert metrics.binding_energy_proxy_ev > 0
    assert metrics.binding_energy_proxy_kcal_mol > 0
    # Check error is tiny
    assert metrics.average_vqe_error_proxy_mha < 0.01

    # Strict check: all attribute names representing indicators contain 'proxy'
    metrics_dict = metrics.to_dict()
    for key in metrics_dict:
        if key != "molecule_name":
            assert "proxy" in key, f"Metric key '{key}' does NOT contain 'proxy'!"
