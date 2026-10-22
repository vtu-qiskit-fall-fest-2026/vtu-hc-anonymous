# ⚡ Q-Lithos: Hybrid Quantum-Classical Simulation Engine for Battery Materials

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Qiskit 1.x](https://img.shields.io/badge/Qiskit-1.4.x-613394.svg)](https://qiskit.org/)
[![Qiskit Nature](https://img.shields.io/badge/Qiskit%20Nature-0.7+-002D9C.svg)](https://qiskit-community.github.io/qiskit-nature/)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-1.4x-FF4B4B.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/Tests-10%20Passed-10b981.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

### The Scientific Challenge
Next-generation energy storage (such as all-solid-state lithium metal and sodium-ion batteries) is fundamentally bottlenecked by electrochemical instability, dendrite propagation, and parasitic chemical degradation at electrode-electrolyte interphases.

Classical supercomputing encounters severe scaling limits when modeling multi-reference electron correlation, near-degeneracy effects, and dynamic charge-transfer states in alkali-metal interfaces:
* **Exact Classical Diagonalization (Full Configuration Interaction, FCI):** Scales exponentially with system size: $\mathcal{O}(e^N)$.
* **Density Functional Theory (DFT):** Relies on parameterized exchange-correlation approximations that struggle with multi-reference bond stretching and transition-state polarization.
* **Empirical Bottleneck:** Materials scientists are frequently restricted to expensive, slow laboratory trial-and-error cycles.

### The Q-Lithos Solution
**Q-Lithos** is an advanced hybrid quantum-classical simulation platform utilizing Qiskit's **Variational Quantum Eigensolver (VQE)** to compute potential energy surfaces and molecular bond characteristics of battery interphase species. The platform integrates:
1. **PySCF Electronic Structure Driver** (minimal Gaussian STO-3G basis set).
2. **ActiveSpaceTransformer** (freezing inert inner core electrons, isolating active valence bonding pairs).
3. **ParityMapper with $\mathbb{Z}_2$ Two-Qubit Reduction** (mapping 4 spin-orbitals to **2 active qubits**).
4. **Hartree-Fock Reference State + Unitary Coupled Cluster (UCCSD) Ansatz** circuit.
5. **SLSQP Classical Optimization** via Qiskit's `StatevectorEstimator` primitive.
6. **Explainable Battery-Material Assessment Engine** providing transparent, evidence-backed preliminary research guidance without fabricated metrics or misleading over-claims.

---

## 🏗️ Architecture & Simulation Pipeline

```mermaid
flowchart TD
    subgraph Classical_PreProcessing ["1. Classical Pre-Processing"]
        A["Molecular Geometry R (Å)<br>(LiH, NaH, NaF, Li2)"] --> B["PySCF Driver (STO-3G)<br>1- & 2-Electron Integrals"]
        B --> C["ActiveSpaceTransformer<br>(2e, 2 spatial orbitals)"]
        C --> D["ParityMapper + ℤ2 Reduction<br>(4 spin-orbitals → 2 Qubits)"]
    end

    subgraph Quantum_Circuit ["2. Quantum Circuit Preparation"]
        D --> E["Qubit Hamiltonian H"]
        F["Hartree-Fock State |HF⟩"] --> G["UCCSD Parameterized Circuit U(θ)"]
    end

    subgraph Hybrid_VQE_Loop ["3. Hybrid VQE Optimization Loop"]
        G --> H["Qiskit StatevectorEstimator<br>(Local ideal simulation)"]
        E --> H
        H --> I["Expectation Value ⟨H⟩(θ)"]
        I --> J["Classical Optimizer (SLSQP)<br>Callback Tracking History"]
        J -->|Update Parameters θ| G
    end

    subgraph Battery_Analysis ["4. Screening & Explainable Assessment"]
        I --> K["Total Ground State Energy E(R)"]
        L["NumPy Exact Eigensolver"] --> M["Pointwise VQE-vs-FCI Error Benchmark"]
        K --> M
        K --> N["Derived Quantum Proxy Descriptors<br>(R_eq, Dissociation D_e, Force Constant k)"]
        N --> O["Battery Assessment Engine<br>(battery_assessment.py)"]
        O --> P["Polished Modern Streamlit Dashboard<br>(app.py)"]
    end
```

---

## 🔬 Battery Materials Library & Active Space Mapping

| Candidate | Formula | Battery Subsystem Role | Scientific Category | Active Space | Qubits | Literature $R_{eq}$ |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Lithium Hydride** | $\text{LiH}$ | Solid Electrolyte Interphase (SEI) Passivation Byproduct | Battery-Relevant Molecule | $(2e, 2o)$ | **2** | $1.596\text{ \AA}$ |
| **Sodium Hydride** | $\text{NaH}$ | Sodium-Metal Anode Interface Decomposition Species | Battery-Relevant Molecule | $(2e, 2o)$ | **2** | $1.887\text{ \AA}$ |
| **Sodium Fluoride** | $\text{NaF}$ | Robust Fluorinated SEI Passivation Component | Battery-Relevant Molecule | $(2e, 2o)$ | **2** | $1.926\text{ \AA}$ |
| **Dilithium** | $\text{Li}_2$ | Quantum Chemistry Homonuclear Benchmark | Calculation Test Case | $(2e, 2o)$ | **2** | $2.673\text{ \AA}$ |

> [!NOTE]
> All systems strictly adhere to the $\le 8$ qubit maximum hardware constraint. Due to particle-number conservation ($\mathbb{Z}_2$ symmetry), the $4$-spin-orbital active spaces condense cleanly into **2 active qubits**, allowing fast, exact-diagonalization verification and execution on NISQ emulators.

---

## 📊 Derived Quantum Proxy Metrics

Every indicator derived by Q-Lithos represents an empirical quantum-mechanical molecular proxy:

1. **Equilibrium Bond Length Proxy ($R_{eq}$ in $\text{\AA}$):**
   * Determined via local parabolic interpolation around the potential energy surface minimum.
   * *Interpretation:* Gas-phase equilibrium bond length only; it does not measure bulk crystal packing density.
2. **Dissociation Energy Proxy ($D_e$ in $\text{eV}$ & $\text{kcal/mol}$):**
   * Computed as $D_e = E(R_{\text{dissociated}}) - E(R_{eq})$.
   * *Interpretation:* Isolated-molecule bond well depth; it does **not** predict electrolyte dissolution resistance, electrochemical window, or cell voltage.
3. **Harmonic Force Constant Proxy ($k$ in $\text{Ha}/\text{\AA}^2$):**
   * Evaluated from the potential well curvature: $k = \left.\frac{d^2E}{dR^2}\right|_{R_{eq}}$.
   * *Interpretation:* Local curvature of the isolated-molecule potential well; it does not measure electrode or separator mechanical rigidity.

> [!IMPORTANT]
> **Proxy Governance & Scientific Integrity:**
> All values generated are explicitly labeled as **proxies**. They describe isolated gas-phase diatomic fragments. They must **never** be interpreted as macroscopic cell properties (such as macroscopic ion mobility, bulk ionic conductivity, cell voltage, capacity, or operational cell thermal stability).

---

## ⚡ Accuracy & Measured Benchmark Results

All 4 candidates are benchmarked across 10 bond lengths against exact classical diagonalization (Full Configuration Interaction, FCI):

| Molecule | Scan Range ($\text{\AA}$) | Points Sampled | Mean VQE Error | Max VQE Error | Chemical Accuracy ($< 1.5936\text{ mHa}$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LiH** | $1.000 - 3.400$ | 10 | **$0.0004\text{ mHa}$** | $0.0014\text{ mHa}$ | Passed across all points | ✅ Passed |
| **NaH** | $1.200 - 3.800$ | 10 | **$0.0000\text{ mHa}$** | $0.0001\text{ mHa}$ | Passed across all points | ✅ Passed |
| **NaF** | $1.300 - 4.000$ | 10 | **$0.0001\text{ mHa}$** | $0.0002\text{ mHa}$ | Passed across all points | ✅ Passed |
| **Li₂** | $1.800 - 5.000$ | 10 | **$0.8798\text{ mHa}$** | $4.0213\text{ mHa}$ | Stretched geometries exceed threshold | ⚠️ Multi-reference character |

*Standard Chemical Accuracy threshold: $1\text{ kcal/mol} \approx 1.5936\text{ mHa} \approx 0.0434\text{ eV}$.*

---

## 🔋 Explainable Battery-Material Assessment Engine

The new `battery_assessment.py` module evaluates candidates using transparent, explainable criteria without over-promising or confusing molecular energetics with battery performance:

| Candidate | Assessment Conclusion | Confidence | Key Evidence | Missing Data Disclosed |
| :--- | :--- | :--- | :--- | :--- |
| **NaF** | **Promising for further research** | Medium | High $D_e = 19.85\text{ Ha}$ (~540 eV diatomic scale) and steep well curvature; proven literature role in suppressing Na dendrites. | Bulk ionic conductivity, grain boundary diffusion, dielectric breakdown voltage. |
| **LiH** | **Mixed evidence — warrants targeted study** | Medium | Established SEI passivation component; $D_e = 7.84\text{ eV}$; verified chemical accuracy. | $\text{Li}^+$ lattice migration barrier, bulk shear modulus (Monroe-Newman), electrolyte solubility. |
| **NaH** | **Mixed evidence — warrants targeted study** | Medium | Weaker bond energy than LiH; known anode parasitic product in sodium batteries. | Sodium self-diffusion activation energy, compliance during continuous plating/stripping. |
| **Li₂** | **Insufficient evidence — scope too narrow** | Low | Homonuclear quantum chemistry benchmark; lacks metallic band structure. | Metallic Fermi level, electrodeposition exchange current density, dendrite nucleation barriers. |

---

## 🚀 Installation & Launch Guide

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Windows, Linux, or macOS

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone <repo-url>
cd anonymous

# Create Python virtual environment
python -m venv .venv

# Activate environment:
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Platform Driver Support:**
> * On **Linux / macOS / Google Colab**, `pyscf` runs ab initio electronic integral generation directly.
> * On **Windows native**, precomputed high-precision datasets and PyQInt / cached references are used automatically, with complete instructions for running live PySCF under WSL.

### 3. Run the Comprehensive Test Suite

```bash
pytest tests/ -v
```
All 10 unit tests pass, covering molecule specifications, active space constraints, metrics derivation, error thresholds, and battery assessment governance.

### 4. Launch the Streamlit Research Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Dashboard Features & Structure

1. **🌐 Overview:** Executive summary, classical FCI exponential scaling bottleneck, candidate profile cards, and end-to-end architecture flowchart.
2. **🔬 Molecular Explorer:** Interactive Plotly 3D geometry visualizer with element-specific colored spheres and bond lines, dynamic distance slider, and literature spectroscopic constants.
3. **⚛️ Quantum Simulation:** Dual-subplot Potential Energy Surface ($E(R)$ in Hartree) with point-by-point VQE error bars, $1.5936\text{ mHa}$ chemical accuracy threshold line, and SLSQP optimizer convergence dynamics.
4. **📊 Energy Analysis:** Harmonic well curvature analysis, numerical data table, and one-click CSV export for scientific publications.
5. **🔋 Battery Assessment:** Explainable evaluation with conclusion badges, supporting evidence, explicitly listed missing data, property status table, limitations, and recommended next steps.
6. **⚖️ Candidate Comparison:** Multi-material screening table and side-by-side interactive comparative charts for dissociation energy, equilibrium distance, and force constants.
7. **📚 Scientific Methods:** Detailed documentation of the Hamiltonian operator, STO-3G basis set, ParityMapper reduction, UCCSD ansatz, and honest limitations.

---

## ⚠️ Honest Scientific Limitations

1. **Minimal Basis Approximations (STO-3G):**
   STO-3G captures essential qualitative potential energy profiles but lacks polarization functions and diffuse orbitals present in triple-zeta sets (e.g., def2-TZVP or aug-cc-pVDZ).
2. **Active Space Truncation:**
   Inner core electrons ($1s^2$ in Li, $1s^2 2s^2 2p^6$ in Na) are frozen to adhere to the $\le 8$ qubit threshold. Dynamic core-valence electron correlation is omitted.
3. **Cluster Fragment vs. Bulk Periodic Lattice:**
   Calculations model isolated diatomic molecules. Real solid-state batteries possess periodic crystal structures, Madelung potentials, and band structures.
4. **Classical Statevector Simulation vs Physical QPU:**
   The dashboard runs on local ideal statevector simulation (`qiskit.primitives.StatevectorEstimator`). It does not run on physical quantum hardware and does not model NISQ device noise (relaxation $T_1$, dephasing $T_2$, or gate errors).
