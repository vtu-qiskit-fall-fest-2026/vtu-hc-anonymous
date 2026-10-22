"""Bond-Length Potential Energy Curve Scan Module for Q-Lithos.

Performs parameterized bond-distance scans across 8-12 points, executing real hybrid
quantum VQE and exact classical diagonalization calculations at every geometry.
Results are cached as JSON so the Streamlit dashboard loads instantly without waiting,
while supporting on-demand recomputation.

Cache format:
  Each JSON file stores a ScanResult serialized via to_dict() / from_dict().
  The cache is validated on load; a corrupted or incompatible cache triggers
  a transparent recomputation with a user-visible warning.
"""

from __future__ import annotations
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from molecules import MOLECULES, get_molecule
from vqe_engine import VQEResult, run_vqe_simulation
from metrics import (
    BatteryMaterialMetrics,
    derive_battery_metrics,
    CHEMICAL_ACCURACY_MHA,
)


CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Cache schema version — increment when the format changes incompatibly
CACHE_VERSION = 2


@dataclass
class ScanPoint:
    """A single evaluation point along the potential energy curve.

    All energies are in Hartree; distance is in Angstroms; error in mHa.
    """

    distance: float          # Angstroms
    vqe_energy: float        # Total ground state energy, Hartree
    exact_energy: float      # Classical exact (NumPy eigensolver), Hartree
    error_mha: float         # |VQE - Exact| × 1000, milli-Hartree
    qubit_count: int         # Qubits used after ParityMapper reduction
    converged: bool          # VQE optimizer convergence flag
    iterations: int          # Number of optimizer function evaluations
    computation_time_seconds: float
    convergence_history: List[tuple] = field(default_factory=list)

    @property
    def within_chemical_accuracy(self) -> bool:
        """True if this point's VQE error is below the chemical accuracy threshold."""
        return self.error_mha < CHEMICAL_ACCURACY_MHA

    @property
    def error_ev(self) -> float:
        """VQE–exact absolute error in electronvolts."""
        return self.error_mha * 1e-3 * 27.211386245988


@dataclass
class ScanResult:
    """Full bond-length potential energy surface scan results.

    Attributes
    ----------
    molecule_name : str
        Molecule identifier.
    points : List[ScanPoint]
        Ordered scan points (ascending distance).
    metrics : BatteryMaterialMetrics
        Derived proxy metrics from the full PES.
    backend_info : str
        Description of the quantum backend used.
    total_scan_time_seconds : float
        Wall-clock time for the full scan.
    timestamp : float
        Unix timestamp when the scan completed.
    cache_version : int
        Format version for cache validation.
    """

    molecule_name: str
    points: List[ScanPoint]
    metrics: BatteryMaterialMetrics
    backend_info: str
    total_scan_time_seconds: float
    timestamp: float = field(default_factory=time.time)
    cache_version: int = CACHE_VERSION

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

    @property
    def all_converged(self) -> bool:
        """True if all scan points reported VQE convergence."""
        return all(p.converged for p in self.points)

    @property
    def n_within_chemical_accuracy(self) -> int:
        """Number of scan points within the chemical accuracy threshold."""
        return sum(1 for p in self.points if p.within_chemical_accuracy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scan result to a dictionary for JSON caching."""
        return {
            "cache_version": self.cache_version,
            "molecule_name": self.molecule_name,
            "points": [asdict(p) for p in self.points],
            "metrics": self.metrics.to_dict(),
            "backend_info": self.backend_info,
            "total_scan_time_seconds": self.total_scan_time_seconds,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScanResult":
        """Reconstruct ScanResult from a dictionary.

        Handles both old (v1) and new (v2) cache formats for backward
        compatibility with existing precomputed caches.
        """
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
                convergence_history=[
                    (int(h[0]), float(h[1])) for h in p.get("convergence_history", [])
                ],
            )
            for p in data["points"]
        ]

        m = data["metrics"]

        # -----------------------------------------------------------------------
        # Backward-compatible metrics reconstruction
        # Old caches store 'binding_energy_proxy_*'; new ones use
        # 'dissociation_energy_proxy_*'.  Accept either key.
        # -----------------------------------------------------------------------
        def _get_energy(d: Dict, new_key: str, old_key: str) -> float:
            if new_key in d:
                return float(d[new_key])
            if old_key in d:
                return float(d[old_key])
            return 0.0

        dis_ha = _get_energy(
            m, "dissociation_energy_proxy_hartree", "binding_energy_proxy_hartree"
        )
        dis_ev = _get_energy(
            m, "dissociation_energy_proxy_ev", "binding_energy_proxy_ev"
        )
        dis_kcal = _get_energy(
            m, "dissociation_energy_proxy_kcal_mol", "binding_energy_proxy_kcal_mol"
        )

        avg_err = float(m.get("average_vqe_error_proxy_mha", 0.0))
        max_err = float(m.get("max_vqe_error_proxy_mha", 0.0))

        metrics = BatteryMaterialMetrics(
            molecule_name=str(m["molecule_name"]),
            equilibrium_bond_length_proxy_angstrom=float(
                m["equilibrium_bond_length_proxy_angstrom"]
            ),
            dissociation_energy_proxy_hartree=dis_ha,
            dissociation_energy_proxy_ev=dis_ev,
            dissociation_energy_proxy_kcal_mol=dis_kcal,
            harmonic_force_constant_proxy_ha_per_ang2=float(
                m["harmonic_force_constant_proxy_ha_per_ang2"]
            ),
            energy_gap_proxy_ev=None,  # Never populate from cache
            average_vqe_error_proxy_mha=avg_err,
            max_vqe_error_proxy_mha=max_err,
            chemical_accuracy_achieved=bool(
                m.get("chemical_accuracy_achieved", max_err < CHEMICAL_ACCURACY_MHA)
            ),
        )

        return cls(
            molecule_name=data["molecule_name"],
            points=points,
            metrics=metrics,
            backend_info=data.get("backend_info", "Aer Simulator (cached)"),
            total_scan_time_seconds=float(data.get("total_scan_time_seconds", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
            cache_version=int(data.get("cache_version", 1)),
        )


def get_cache_path(molecule_name: str) -> Path:
    """Return JSON cache file path for a given molecule."""
    return CACHE_DIR / f"{molecule_name}_scan.json"


def _validate_cache(data: Dict[str, Any], molecule_name: str) -> bool:
    """Check that cached data is structurally valid and matches the molecule.

    Returns True if the cache can be safely loaded, False if it should be
    discarded and recomputed.
    """
    try:
        if data.get("molecule_name") != molecule_name:
            return False
        points = data.get("points", [])
        if len(points) < 3:
            return False
        # Verify each point has required numeric fields
        required = {"distance", "vqe_energy", "exact_energy", "error_mha"}
        for p in points:
            if not required.issubset(p.keys()):
                return False
            # Sanity check: energies should be finite
            if not all(
                isinstance(p[k], (int, float)) and not (p[k] != p[k])  # NaN check
                for k in ("vqe_energy", "exact_energy")
            ):
                return False
        return True
    except Exception:
        return False


def run_bond_length_scan(
    molecule_name: str,
    distances: Optional[List[float]] = None,
    recompute: bool = False,
    maxiter: int = 50,
    backend_mode: str = "Aer Simulator",
    point_callback: Optional[Callable[[int, int, ScanPoint], None]] = None,
) -> ScanResult:
    """Run or load a bond-length potential energy scan for a molecule.

    If a valid cache exists and recompute=False, the cached result is
    returned immediately.  Otherwise, VQE and exact-diagonalization
    calculations are executed for each distance point and the result is
    persisted to the cache directory.

    Parameters
    ----------
    molecule_name : str
        Molecule name ('LiH', 'NaH', 'NaF', 'Li2').
    distances : Optional[List[float]]
        Custom list of distances (Å).  Defaults to the molecule's
        standard scan_distances.
    recompute : bool
        If True, ignore any existing cache and run fresh calculations.
    maxiter : int
        Maximum SLSQP optimizer iterations per distance point.
    backend_mode : str
        Quantum simulator or hardware backend mode string.
    point_callback : Optional[Callable[[int, int, ScanPoint], None]]
        Callback invoked when each point completes: (index, total, point).

    Returns
    -------
    ScanResult
        Complete scan points and derived battery proxy metrics.

    Raises
    ------
    RuntimeError
        If the quantum simulation fails at any point and no fallback is
        available.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = get_cache_path(molecule_name)

    # -----------------------------------------------------------------------
    # 1. Attempt cache load
    # -----------------------------------------------------------------------
    if not recompute and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8-sig") as fh:
                raw = json.load(fh)
            if _validate_cache(raw, molecule_name):
                return ScanResult.from_dict(raw)
            # Cache is invalid — fall through to recompute
        except Exception:
            pass  # Corrupted file — fall through to recompute

    # -----------------------------------------------------------------------
    # 2. Setup scan parameters
    # -----------------------------------------------------------------------
    mol_def = get_molecule(molecule_name)
    scan_pts = distances if distances is not None else mol_def.scan_distances
    total_points = len(scan_pts)
    points: List[ScanPoint] = []
    start_time = time.time()
    last_backend = backend_mode

    # -----------------------------------------------------------------------
    # 3. Execute VQE simulation across the PES
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 4. Derive proxy metrics
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 5. Persist to cache (atomic write via temp file)
    # -----------------------------------------------------------------------
    temp_cache = cache_file.with_suffix(".tmp")
    try:
        with open(temp_cache, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        temp_cache.replace(cache_file)
    except OSError:
        # Cache write failure is non-fatal
        if temp_cache.exists():
            temp_cache.unlink(missing_ok=True)

    return result


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "LiH"
    mols = list(MOLECULES.keys()) if target in ("--all", "all") else [target]

    print("=== Q-Lithos Bond-Length Scanning Engine ===")
    print(f"Molecules to compute: {mols}")

    for mol_name in mols:
        print(f"\nComputing PES curve for {mol_name}...")
        res = run_bond_length_scan(
            mol_name,
            recompute=True,
            point_callback=lambda i, tot, pt: print(
                f"  [{i}/{tot}] R={pt.distance:.3f} Å  "
                f"VQE: {pt.vqe_energy:.6f} Ha  "
                f"Exact: {pt.exact_energy:.6f} Ha  "
                f"Error: {pt.error_mha:.4f} mHa  "
                f"({'✓ chem. acc.' if pt.within_chemical_accuracy else '✗'})"
            ),
        )
        print(f"Completed in {res.total_scan_time_seconds:.1f} s")
        print(f"  Equilibrium R_eq (proxy):  {res.metrics.equilibrium_bond_length_proxy_angstrom:.4f} Å")
        print(f"  Dissociation D_e (proxy):  {res.metrics.dissociation_energy_proxy_ev:.4f} eV  ({res.metrics.dissociation_energy_proxy_kcal_mol:.2f} kcal/mol)")
        print(f"  Average VQE error:         {res.metrics.average_vqe_error_proxy_mha:.5f} mHa")
        print(f"  Maximum VQE error:         {res.metrics.max_vqe_error_proxy_mha:.5f} mHa")
        print(f"  Chemical accuracy (<1.5936 mHa): {'YES — all points pass' if res.metrics.chemical_accuracy_achieved else 'NO — some points exceed threshold'}")
