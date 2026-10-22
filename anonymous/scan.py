"""Bond-Length Potential Energy Curve Scan Module for Q-Lithos.

Performs parameterized bond-distance scans across 8-12 points, executing real hybrid
quantum VQE and exact classical diagonalization calculations at every geometry.
Results are cached as JSON so the Streamlit dashboard loads instantly without waiting,
while supporting on-demand recomputation.
"""

from __future__ import annotations
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from molecules import MOLECULES, MoleculeDefinition, get_molecule
from vqe_engine import VQEResult, run_vqe_simulation
from metrics import BatteryMaterialMetrics, derive_battery_metrics


CACHE_DIR = Path(__file__).resolve().parent / "cache"


@dataclass
class ScanPoint:
    """A single evaluation point along the potential energy curve."""

    distance: float  # Angstroms
    vqe_energy: float  # Hartree
    exact_energy: float  # Hartree
    error_mha: float  # milli-Hartree
    qubit_count: int
    converged: bool
    iterations: int
    computation_time_seconds: float
    convergence_history: List[tuple[int, float]] = field(default_factory=list)


@dataclass
class ScanResult:
    """Full bond-length potential energy surface scan results."""

    molecule_name: str
    points: List[ScanPoint]
    metrics: BatteryMaterialMetrics
    backend_info: str
    total_scan_time_seconds: float
    timestamp: float = field(default_factory=time.time)

    @property
    def distances(self) -> List[float]:
        return [p.distance for p in self.points]

    @property
    def vqe_energies(self) -> List[float]:
        return [p.vqe_energy for p in self.points]

    @property
    def exact_energies(self) -> List[float]:
        return [p.exact_energy for p in self.points]

    @property
    def errors_mha(self) -> List[float]:
        return [p.error_mha for p in self.points]

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to a dictionary for JSON caching."""
        return {
            "molecule_name": self.molecule_name,
            "points": [asdict(p) for p in self.points],
            "metrics": self.metrics.to_dict(),
            "backend_info": self.backend_info,
            "total_scan_time_seconds": self.total_scan_time_seconds,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScanResult:
        """Reconstruct ScanResult from dictionary."""
        points = [
            ScanPoint(
                distance=float(p["distance"]),
                vqe_energy=float(p["vqe_energy"]),
                exact_energy=float(p["exact_energy"]),
                error_mha=float(p["error_mha"]),
                qubit_count=int(p["qubit_count"]),
                converged=bool(p["converged"]),
                iterations=int(p["iterations"]),
                computation_time_seconds=float(p.get("computation_time_seconds", 0.0)),
                convergence_history=[(int(h[0]), float(h[1])) for h in p.get("convergence_history", [])],
            )
            for p in data["points"]
        ]
        m_dict = data["metrics"]
        metrics = BatteryMaterialMetrics(
            molecule_name=str(m_dict["molecule_name"]),
            equilibrium_bond_length_proxy_angstrom=float(m_dict["equilibrium_bond_length_proxy_angstrom"]),
            binding_energy_proxy_hartree=float(m_dict["binding_energy_proxy_hartree"]),
            binding_energy_proxy_ev=float(m_dict["binding_energy_proxy_ev"]),
            binding_energy_proxy_kcal_mol=float(m_dict["binding_energy_proxy_kcal_mol"]),
            harmonic_force_constant_proxy_ha_per_ang2=float(m_dict["harmonic_force_constant_proxy_ha_per_ang2"]),
            energy_gap_proxy_ev=float(m_dict["energy_gap_proxy_ev"]),
            average_vqe_error_proxy_mha=float(m_dict["average_vqe_error_proxy_mha"]),
            max_vqe_error_proxy_mha=float(m_dict["max_vqe_error_proxy_mha"]),
        )
        return cls(
            molecule_name=data["molecule_name"],
            points=points,
            metrics=metrics,
            backend_info=data.get("backend_info", "Aer Simulator"),
            total_scan_time_seconds=float(data.get("total_scan_time_seconds", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )


def get_cache_path(molecule_name: str) -> Path:
    """Return JSON cache file path for a given molecule."""
    return CACHE_DIR / f"{molecule_name}_scan.json"


def run_bond_length_scan(
    molecule_name: str,
    distances: Optional[List[float]] = None,
    recompute: bool = False,
    maxiter: int = 50,
    backend_mode: str = "Aer Simulator",
    point_callback: Optional[Callable[[int, int, ScanPoint], None]] = None,
) -> ScanResult:
    """Run or load a bond-length potential energy scan for a molecule.

    Parameters
    ----------
    molecule_name : str
        Molecule name ('LiH', 'NaH', 'NaF', 'Li2').
    distances : Optional[List[float]]
        Optional custom list of distances. Defaults to standard 8-12 points.
    recompute : bool
        If True, ignores existing cache and runs full simulation.
    backend_mode : str
        Quantum simulator or hardware backend mode.
    point_callback : Optional[Callable[[int, int, ScanPoint], None]]
        Optional callback called as each distance point finishes: (index, total, point).

    Returns
    -------
    ScanResult
        Complete scan points and derived battery proxy metrics.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = get_cache_path(molecule_name)

    # 1. Check Cache
    if not recompute and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ScanResult.from_dict(data)
        except Exception:
            # If cache is corrupted, proceed to recompute
            pass

    # 2. Setup scan distances
    mol_def = get_molecule(molecule_name)
    scan_pts = distances if distances is not None else mol_def.scan_distances
    total_points = len(scan_pts)
    points: List[ScanPoint] = []
    start_time = time.time()
    last_backend = "Aer Simulator"

    # 3. Execute Quantum Simulation Across Potential Energy Surface
    for idx, r in enumerate(scan_pts):
        vqe_res = run_vqe_simulation(
            molecule_name=molecule_name,
            distance=r,
            maxiter=maxiter,
            backend_mode=backend_mode,
        )
        last_backend = vqe_res.backend_info
        point = ScanPoint(
            distance=r,
            vqe_energy=vqe_res.vqe_energy,
            exact_energy=vqe_res.exact_energy,
            error_mha=vqe_res.error_mha,
            qubit_count=vqe_res.qubit_count,
            converged=vqe_res.converged,
            iterations=vqe_res.iterations,
            computation_time_seconds=vqe_res.computation_time_seconds,
            convergence_history=vqe_res.convergence_history,
        )
        points.append(point)

        if point_callback:
            point_callback(idx + 1, total_points, point)

    total_time = time.time() - start_time

    # 4. Derive Quantitative Battery Material Proxy Metrics
    metrics = derive_battery_metrics(
        molecule_name=molecule_name,
        distances=[p.distance for p in points],
        vqe_energies=[p.vqe_energy for p in points],
        exact_energies=[p.exact_energy for p in points],
    )

    result = ScanResult(
        molecule_name=molecule_name,
        points=points,
        metrics=metrics,
        backend_info=last_backend,
        total_scan_time_seconds=total_time,
    )

    # 5. Persist to Cache
    temp_cache = cache_file.with_suffix(".tmp")
    with open(temp_cache, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    temp_cache.replace(cache_file)

    return result


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "LiH"
    mols = list(MOLECULES.keys()) if target in ("--all", "all") else [target]

    print(f"=== Q-Lithos Bond-Length Scanning Engine ===")
    print(f"Molecules to compute: {mols}")

    for mol_name in mols:
        print(f"\nComputing PES curve for {mol_name}...")
        res = run_bond_length_scan(
            mol_name,
            recompute=True,
            point_callback=lambda i, tot, pt: print(
                f"  [{i}/{tot}] R={pt.distance:.3f} A -> VQE: {pt.vqe_energy:.6f} Ha | "
                f"Exact: {pt.exact_energy:.6f} Ha | Error: {pt.error_mha:.4f} mHa ({pt.iterations} iters)"
            ),
        )
        print(f"Done in {res.total_scan_time_seconds:.1f} s")
        print(f"Equilibrium R proxy: {res.metrics.equilibrium_bond_length_proxy_angstrom:.3f} A")
        print(f"Binding energy proxy: {res.metrics.binding_energy_proxy_ev:.3f} eV ({res.metrics.binding_energy_proxy_kcal_mol:.1f} kcal/mol)")
        print(f"Average VQE error proxy: {res.metrics.average_vqe_error_proxy_mha:.4f} mHa (Max: {res.metrics.max_vqe_error_proxy_mha:.4f} mHa)")
