"""Q-Lithos: Hybrid Quantum-Classical Simulation Engine for Battery Materials.

A modern, scientifically rigorous quantum chemistry and battery-material research dashboard.
Built with Qiskit 1.x, Plotly, and Streamlit.
"""

from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Q-Lithos core modules
from molecules import MOLECULES, get_molecule, MAX_QUBIT_BUDGET, MoleculeDefinition
from scan import run_bond_length_scan, get_cache_path, ScanResult, ScanPoint
from metrics import BatteryMaterialMetrics, HARTREE_TO_EV, HARTREE_TO_KCAL_MOL, CHEMICAL_ACCURACY_MHA
from battery_assessment import (
    assess_candidate,
    AssessmentConclusion,
    ConfidenceLevel,
    CandidateAssessment,
)

# Load environment configuration (.env)
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="Q-Lithos | Quantum Battery Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS: Dark Navy & Cyan/Teal Scientific Research Dashboard
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Container */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    
    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(10, 15, 30, 0.95), rgba(15, 28, 50, 0.95));
        border: 1px solid rgba(34, 211, 238, 0.25);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), 0 0 20px 0 rgba(34, 211, 238, 0.08);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #22d3ee 0%, #14b8a6 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.5;
        margin-bottom: 14px;
    }
    .context-pill {
        display: inline-block;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.25);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .context-pill-amber {
        display: inline-block;
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(13, 27, 46, 0.85), rgba(18, 35, 60, 0.85));
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: rgba(34, 211, 238, 0.4);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 4px;
        font-family: 'SF Mono', 'Courier New', monospace;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 3px;
    }

    /* Status Badges */
    .status-badge-pass {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .status-badge-fail {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .status-badge-warn {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.06);
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 16px 0;
        color: #fde68a;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Scientific Card */
    .sci-card {
        background: rgba(13, 27, 46, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .sci-header {
        font-weight: 700;
        color: #38bdf8;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    /* Conclusion Pills */
    .conc-promising {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .conc-mixed {
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .conc-insufficient {
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid #94a3b8;
        color: #cbd5e1;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS & CONFIGURATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Q-Lithos Engine")
    st.caption("Hybrid Quantum-Classical Simulation for Battery Materials")

    # Molecule selection
    selected_mol_name = st.selectbox(
        "Select Battery Molecule Fragment",
        options=list(MOLECULES.keys()),
        index=0,
        help="Small molecular fragments representative of solid-state and sodium-ion battery interfaces.",
    )
    mol_def = get_molecule(selected_mol_name)

    # Category and Role Pill

# Expandable molecule information — LEFT SIDEBAR ONLY
with st.sidebar.expander(
    f"🧪 {mol_def.formula} — Molecule Details",
    expanded=False
):
    st.markdown("### Battery Role")
    st.write(mol_def.battery_role)

    category = getattr(mol_def, "battery_category", None)
    if category is not None:
        category_label = getattr(category, "value", str(category))
        st.caption(f"Category: {category_label}")

    st.markdown("### Chemical Description")
    st.write(mol_def.description)

    caveats = getattr(mol_def, "battery_caveats", None)
    if caveats:
        st.markdown("### Scientific Limitations")
        st.info(caveats)
        # Always-visible recompute button in LEFT SIDEBAR

    # Quantum Hardware Constraint HUD
    st.markdown("#### ⚛️ Quantum Execution Profile")
    st.info(
        "**Backend Mode:** Local ideal statevector simulator (`qiskit.primitives.StatevectorEstimator`).\n\n"
        "Classical simulation of quantum algorithms; not physical quantum hardware execution."
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.metric("Active Qubits", f"{mol_def.qubit_count_expected} Qubits")
    with col_b2:
        st.metric("Hardware Budget", f"{MAX_QUBIT_BUDGET} Qubits")

    st.caption(
        f"Active Space: {mol_def.num_active_electrons}e in {mol_def.num_active_orbitals} spatial orbitals. "
        "Mapped via ParityMapper with 2-qubit reduction (4 spin-orbitals → 2 qubits)."
    )

    # Recomputation & Cache Management
    
# Always-visible recompute section in LEFT SIDEBAR
st.sidebar.markdown("---")

cache_path = get_cache_path(selected_mol_name)
has_cache = cache_path.exists()

if has_cache:
    st.sidebar.caption("⚡ Cache status: **Validated dataset ready**")
else:
    st.sidebar.caption("⚠️ Cache status: **Not yet computed**")

recompute_btn = st.sidebar.button(
    "🔄 Recompute on VQE Engine",
    use_container_width=True
)


# ---------------------------------------------------------------------------
# DATA LOADING / EXECUTION CONTROLLER
# ---------------------------------------------------------------------------
progress_placeholder = st.empty()
scan_result: Optional[ScanResult] = None

if recompute_btn:
    with progress_placeholder.container():
        st.info(f"⏳ Running quantum VQE simulation across 10 bond lengths for {selected_mol_name}...")
        prog_bar = st.progress(0)
        status_text = st.empty()

        def on_step(curr, total, pt: ScanPoint):
            prog_bar.progress(curr / total)
            status_text.write(
                f"Sampling R = {pt.distance:.3f} Å | VQE: {pt.vqe_energy:.6f} Ha | Exact: {pt.exact_energy:.6f} Ha (Error: {pt.error_mha:.4f} mHa)"
            )

        try:
            scan_result = run_bond_length_scan(
                molecule_name=selected_mol_name,
                recompute=True,
                backend_mode="Aer Simulator",
                point_callback=on_step,
            )
            time.sleep(0.4)
        except Exception as exc:
            st.error(f"VQE Simulation Engine reported: {exc}")
            # Fall back to cached data if available
            if has_cache:
                st.warning("Reverting to precomputed benchmark dataset.")
                scan_result = run_bond_length_scan(selected_mol_name, recompute=False)
    progress_placeholder.empty()
else:
    # Load from cache (or compute if absent)
    if not has_cache:
        with st.spinner(f"Computing quantum potential energy surface for {selected_mol_name}..."):
            scan_result = run_bond_length_scan(selected_mol_name, recompute=True)
    else:
        scan_result = run_bond_length_scan(selected_mol_name, recompute=False)

metrics = scan_result.metrics
assessment: CandidateAssessment = assess_candidate(selected_mol_name, metrics, mol_def)

# ---------------------------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">Q-Lithos: Hybrid Quantum-Classical Simulation Engine</div>
        <div class="hero-subtitle">
            First-principles quantum chemistry for battery materials screening: resolving multi-reference electron
            correlation in solid-state and sodium-ion molecular interphases via Qiskit Variational Quantum Eigensolver (VQE).
        </div>
        <div>
            <span class="context-pill">⚡ Exponential Scaling: FCI scales as O(e^N)</span>
            <span class="context-pill">🔬 Pipeline: Qiskit 1.x VQE + UCCSD + ParityMapper</span>
            <span class="context-pill">🛡️ Strict Budget: {mol_def.qubit_count_expected} of {MAX_QUBIT_BUDGET} Qubits Used</span>
            <span class="context-pill-amber">⚠️ Scope: Gas-Phase Molecular Proxies (Not Cell Voltage)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOP KPI ROW: ACCURACY & PROXY DESCRIPTORS
# ---------------------------------------------------------------------------
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    avg_err = metrics.average_vqe_error_proxy_mha
    max_err = metrics.max_vqe_error_proxy_mha
    is_chem_acc = metrics.chemical_accuracy_achieved

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">VQE Accuracy (vs Exact FCI)</div>
            <div class="metric-value" style="color: {'#34d399' if is_chem_acc else '#f59e0b'};">
                {avg_err:.4f} <span style="font-size: 0.95rem; color: #94a3b8;">mHa</span>
            </div>
            <div class="metric-sub">Worst-case deviation: {max_err:.4f} mHa</div>
            <div style="margin-top: 10px;">
                <span class="{'status-badge-pass' if is_chem_acc else 'status-badge-warn'}">
                    {'✓ Passed Chemical Accuracy (< 1.6 mHa)' if is_chem_acc else '⚠️ Points Exceed 1.6 mHa'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi2:
    r_eq = metrics.equilibrium_bond_length_proxy_angstrom
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Equilibrium Distance (Proxy)</div>
            <div class="metric-value" style="color: #38bdf8;">
                {r_eq:.3f} <span style="font-size: 0.95rem; color: #94a3b8;">Å</span>
            </div>
            <div class="metric-sub">Literature equilibrium: ~{mol_def.equilibrium_distance:.3f} Å</div>
            <div style="margin-top: 10px;">
                <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Interatomic Packing Proxy
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi3:
    d_ev = metrics.dissociation_energy_proxy_ev
    d_kcal = metrics.dissociation_energy_proxy_kcal_mol
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Dissociation Well Depth (D_e)</div>
            <div class="metric-value" style="color: #a855f7;">
                {d_ev:.3f} <span style="font-size: 0.95rem; color: #94a3b8;">eV</span>
            </div>
            <div class="metric-sub">{d_kcal:.1f} kcal/mol ({metrics.dissociation_energy_proxy_hartree:.5f} Ha)</div>
            <div style="margin-top: 10px;">
                <span style="background: rgba(168, 85, 247, 0.15); color: #c084fc; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Molecular Bond Cohesion Proxy
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi4:
    k_force = metrics.harmonic_force_constant_proxy_ha_per_ang2
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Force Constant (Curvature)</div>
            <div class="metric-value" style="color: #f43f5e;">
                {k_force:.3f} <span style="font-size: 0.95rem; color: #94a3b8;">Ha/Å²</span>
            </div>
            <div class="metric-sub">Parabolic well curvature d²E/dR²</div>
            <div style="margin-top: 10px;">
                <span style="background: rgba(244, 63, 94, 0.15); color: #fb7185; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Local Rigidity Proxy
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------------------------
tab_overview, tab_explorer, tab_quantum, tab_energy, tab_battery, tab_compare, tab_methods = st.tabs([
    "🌐 Overview",
    "🔬 Molecular Explorer",
    "⚛️ Quantum Simulation",
    "📊 Energy Analysis",
    "🔋 Battery Assessment",
    "⚖️ Candidate Comparison",
    "📚 Scientific Methods",
])

# ---------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Next-Generation Battery Materials & Quantum Chemical Simulation")
    
    col_ov1, col_ov2 = st.columns([3, 2])
    with col_ov1:
        st.markdown(
            r"""
            ### The Classical Supercomputing Bottleneck
            All-solid-state lithium metal and high-energy sodium-ion batteries face severe bottlenecks at 
            **solid electrolyte interphases (SEI)** and **electrode interfaces**. 
            
            Accurately predicting atomic-scale passivation, decomposition, and bond dissociation requires 
            resolving strong electron correlation:
            * **Full Configuration Interaction (FCI):** Scales exponentially with system size $\mathcal{O}(e^N)$, 
              rendering exact classical diagonalization intractable beyond minimal active spaces.
            * **Density Functional Theory (DFT):** Relies on parameterized exchange-correlation functionals 
              that often mischaracterize multi-reference states, charge-transfer barriers, and stretched dissociation geometries.
            
            ### The Q-Lithos Hybrid Solution
            **Q-Lithos** applies Qiskit's **Variational Quantum Eigensolver (VQE)** to molecular fragments 
            representative of battery interphases. By employing active-space isolation and $\mathbb{Z}_2$ symmetry 
            reduction via ParityMapper, the simulation condenses the valence bonding problem into **2 qubits**, 
            demonstrating high-fidelity NISQ-compatible quantum chemistry.
            """
        )
    with col_ov2:
        st.markdown(
            f"""
            <div class="sci-card">
                <div class="sci-header">🎯 Active Candidate Profile</div>
                <table style="width:100%; font-size: 0.88rem; color: #cbd5e1;">
                    <tr><td style="color:#94a3b8; padding:4px 0;">Chemical Formula:</td><td><strong>{mol_def.formula}</strong></td></tr>
                    <tr><td style="color:#94a3b8; padding:4px 0;">Subsystem Role:</td><td>{mol_def.battery_role}</td></tr>
                    <tr><td style="color:#94a3b8; padding:4px 0;">Scientific Category:</td><td>{mol_def.category_label}</td></tr>
                    <tr><td style="color:#94a3b8; padding:4px 0;">Literature Bond Length:</td><td>{mol_def.equilibrium_distance:.3f} Å</td></tr>
                    <tr><td style="color:#94a3b8; padding:4px 0;">Active Space:</td><td>({mol_def.num_active_electrons}e, {mol_def.num_active_orbitals}o) → 2 Qubits</td></tr>
                    <tr><td style="color:#94a3b8; padding:4px 0;">Backend Primitive:</td><td>{scan_result.backend_info}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Architecture Overview Cards
    st.markdown("#### 🏗️ Simulation Pipeline Architecture")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown(
            """
            <div class="sci-card">
                <div style="font-weight:700; color:#38bdf8; font-size:0.9rem;">1. Classical Driver</div>
                <p style="font-size:0.8rem; color:#cbd5e1; margin-top:6px;">
                    PySCF Driver with Gaussian STO-3G basis evaluates 1- and 2-electron Coulomb & exchange integrals at geometry R.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2:
        st.markdown(
            """
            <div class="sci-card">
                <div style="font-weight:700; color:#2dd4bf; font-size:0.9rem;">2. Active Space & Qubits</div>
                <p style="font-size:0.8rem; color:#cbd5e1; margin-top:6px;">
                    Core electrons are frozen. ParityMapper with 2-qubit reduction maps 4 spin-orbitals directly to 2 qubits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p3:
        st.markdown(
            """
            <div class="sci-card">
                <div style="font-weight:700; color:#a855f7; font-size:0.9rem;">3. VQE Hybrid Loop</div>
                <p style="font-size:0.8rem; color:#cbd5e1; margin-top:6px;">
                    Hartree-Fock state + UCCSD ansatz executed on StatevectorEstimator; parameterized energy minimized via SLSQP.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p4:
        st.markdown(
            """
            <div class="sci-card">
                <div style="font-weight:700; color:#f43f5e; font-size:0.9rem;">4. Battery Assessment</div>
                <p style="font-size:0.8rem; color:#cbd5e1; margin-top:6px;">
                    Derives molecular proxies, validates against classical FCI benchmarks, and produces explainable research guidance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Core Scientific Proxy Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ SCIENTIFIC GOVERNANCE NOTICE:</strong><br>
            All quantum chemistry indicators derived by Q-Lithos represent <em>first-principles gas-phase molecular proxies</em>. 
            Gas-phase bond dissociation energy (D_e) is an isolated-molecule well depth; it is <strong>NOT</strong> battery cell voltage, 
            solid-state ionic conductivity, or cycle lifetime. Never confuse molecular total energy with battery capacity.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TAB 2: MOLECULAR EXPLORER (3D GEOMETRY & PROPERTIES)
# ---------------------------------------------------------------------------
with tab_explorer:
    st.subheader(f"Molecular Geometry & Atomic Configuration: {mol_def.formula}")
    st.caption("Interactive 3D geometry visualization and literature spectroscopic constants.")

    col_3d, col_info = st.columns([3, 2])

    with col_3d:
        # Atomic coordinates and visual styling
        atom1, atom2 = mol_def.atom_symbols
        # Element radii (scaled for visualization) and color scheme
        element_meta = {
            "H": {"color": "#f8fafc", "size": 16, "name": "Hydrogen", "Z": 1, "mass": 1.008},
            "Li": {"color": "#818cf8", "size": 26, "name": "Lithium", "Z": 3, "mass": 6.94},
            "Na": {"color": "#fb923c", "size": 32, "name": "Sodium", "Z": 11, "mass": 22.99},
            "F": {"color": "#34d399", "size": 22, "name": "Fluorine", "Z": 9, "mass": 18.998},
        }

        # Dynamic bond distance slider
        dist_slider = st.slider(
            "Inspect Interatomic Separation Distance R (Å)",
            min_value=0.8,
            max_value=4.5,
            value=float(round(metrics.equilibrium_bond_length_proxy_angstrom, 3)),
            step=0.05,
            help="Interactively modify interatomic bond length along the z-axis.",
        )

        fig_3d = go.Figure()

        # Bond line connecting atoms
        fig_3d.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, dist_slider],
                mode="lines",
                line=dict(color="#64748b", width=10),
                hoverinfo="none",
                name="Covalent Bond",
            )
        )

        # Atom 1 at origin
        meta1 = element_meta.get(atom1, {"color": "#94a3b8", "size": 20, "name": atom1, "Z": 0, "mass": 0})
        fig_3d.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers+text",
                marker=dict(size=meta1["size"], color=meta1["color"], opacity=0.95),
                text=[f"<b>{atom1}</b>"],
                textposition="top center",
                textfont=dict(color="#f8fafc", size=14),
                hovertemplate=f"Atom: {meta1['name']} ({atom1})<br>Coordinates: (0.0, 0.0, 0.0) Å<br>Z = {meta1['Z']}<extra></extra>",
                name=f"{meta1['name']} (Origin)",
            )
        )

        # Atom 2 at (0, 0, R)
        meta2 = element_meta.get(atom2, {"color": "#94a3b8", "size": 20, "name": atom2, "Z": 0, "mass": 0})
        fig_3d.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[dist_slider],
                mode="markers+text",
                marker=dict(size=meta2["size"], color=meta2["color"], opacity=0.95),
                text=[f"<b>{atom2}</b>"],
                textposition="top center",
                textfont=dict(color="#f8fafc", size=14),
                hovertemplate=f"Atom: {meta2['name']} ({atom2})<br>Coordinates: (0.0, 0.0, {dist_slider:.3f}) Å<br>Z = {meta2['Z']}<extra></extra>",
                name=f"{meta2['name']} (Offset)",
            )
        )

        fig_3d.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False, backgroundcolor="rgba(13,27,46,0.3)"),
                yaxis=dict(range=[-2.5, 2.5], title="", showticklabels=False, backgroundcolor="rgba(13,27,46,0.3)"),
                zaxis=dict(range=[-1.0, 5.0], title="Separation Axis (Å)", backgroundcolor="rgba(13,27,46,0.3)"),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=460,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_info:
        st.markdown("#### 🔬 Spectroscopic & Chemical Metadata")
        st.markdown(
            f"""
            - **Formula:** `{mol_def.formula}`
            - **Constituent Atoms:** {atom1} (Z={meta1['Z']}) + {atom2} (Z={meta2['Z']})
            - **Equilibrium Bond Length (Lit):** `{mol_def.equilibrium_distance:.4f} Å`
            - **Literature Source:** *{mol_def.literature_source}*
            - **Charge:** {mol_def.charge} | **Spin Multiplicity:** {mol_def.spin_multiplicity} (Singlet)
            - **Gaussian Basis Set:** `{mol_def.basis_set}` (Minimal Gaussian)
            - **Active Space Partitioning:** `{mol_def.num_active_electrons} electrons` across `{mol_def.num_active_orbitals} spatial orbitals`
            - **Fermion Mapping:** `ParityMapper` with $\\mathbb{{Z}}_2$ reduction $\\implies$ **2 Qubits**
            """
        )

        st.markdown("#### 📚 Experimental Literature Values")
        if mol_def.known_properties:
            lit_table = [
                {"Property": k.replace("_", " ").title(), "Value": f"{v[0]} {v[1]}", "Citation": v[2]}
                for k, v in mol_def.known_properties.items()
            ]
            st.dataframe(pd.DataFrame(lit_table), use_container_width=True, hide_index=True)
        else:
            st.info("No literature constants registered.")

# ---------------------------------------------------------------------------
# TAB 3: QUANTUM SIMULATION (PES & CONVERGENCE)
# ---------------------------------------------------------------------------
with tab_quantum:
    st.subheader(f"Quantum Variational Simulation: {mol_def.formula}")
    st.caption(
        "Variational Quantum Eigensolver (VQE with UCCSD) vs Classical Exact Diagonalization (Full CI reference) "
        "across sampled interatomic bond lengths."
    )

    df_points = pd.DataFrame([
        {
            "Distance (Å)": p.distance,
            "VQE Energy (Ha)": p.vqe_energy,
            "Exact Energy (Ha)": p.exact_energy,
            "Error (mHa)": p.error_mha,
            "Error (eV)": p.error_ev,
            "Iterations": p.iterations,
            "Qubits": p.qubit_count,
            "Chemical Accuracy": "Passed" if p.within_chemical_accuracy else "Exceeded",
        }
        for p in scan_result.points
    ])

    # Plotly Subplots for PES and Point Error
    fig_pes = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            f"Total Ground State Energy E(R) for {mol_def.formula} (Hartree)",
            "Pointwise VQE Deviation from Classical FCI Reference (mHa)",
        ),
    )

    # Upper Subplot: VQE & Exact Energy Traces
    fig_pes.add_trace(
        go.Scatter(
            x=df_points["Distance (Å)"],
            y=df_points["VQE Energy (Ha)"],
            mode="lines+markers",
            name="VQE (Statevector + UCCSD)",
            line=dict(color="#22d3ee", width=3),
            marker=dict(size=8, color="#22d3ee", symbol="circle"),
            hovertemplate="Distance: %{x:.3f} Å<br>VQE Energy: %{y:.6f} Ha<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig_pes.add_trace(
        go.Scatter(
            x=df_points["Distance (Å)"],
            y=df_points["Exact Energy (Ha)"],
            mode="lines",
            name="Exact Diagonalization (FCI)",
            line=dict(color="#fbbf24", width=2, dash="dash"),
            hovertemplate="Distance: %{x:.3f} Å<br>Exact Energy: %{y:.6f} Ha<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Equilibrium annotation
    fig_pes.add_annotation(
        x=metrics.equilibrium_bond_length_proxy_angstrom,
        y=float(df_points["VQE Energy (Ha)"].min()),
        text=f"R_eq ~ {metrics.equilibrium_bond_length_proxy_angstrom:.3f} Å",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor="#34d399",
        font=dict(color="#34d399", size=11, family="monospace"),
        ax=45,
        ay=-40,
        row=1,
        col=1,
    )

    # Lower Subplot: Error Bar Chart
    colors_error = [
        "#34d399" if err < CHEMICAL_ACCURACY_MHA else "#f87171"
        for err in df_points["Error (mHa)"]
    ]
    fig_pes.add_trace(
        go.Bar(
            x=df_points["Distance (Å)"],
            y=df_points["Error (mHa)"],
            name="VQE Error (mHa)",
            marker=dict(color=colors_error),
            hovertemplate="Distance: %{x:.3f} Å<br>Error: %{y:.4f} mHa<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Chemical accuracy threshold line at 1.5936 mHa (~1 kcal/mol)
    fig_pes.add_hline(
        y=CHEMICAL_ACCURACY_MHA,
        line=dict(color="#ef4444", width=1.5, dash="dot"),
        annotation_text="Chemical Accuracy Threshold (1.5936 mHa = 1 kcal/mol)",
        annotation_position="top right",
        annotation_font=dict(size=9, color="#f87171"),
        row=2,
        col=1,
    )

    fig_pes.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.4)",
    )
    fig_pes.update_xaxes(title_text="Interatomic Bond Distance R (Å)", row=2, col=1)
    fig_pes.update_yaxes(title_text="Energy (Ha)", row=1, col=1)
    fig_pes.update_yaxes(title_text="Error (mHa)", row=2, col=1)

    st.plotly_chart(fig_pes, use_container_width=True)

    # -----------------------------------------------------------------------
    # SLSQP Optimizer Convergence History
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎯 VQE Classical Optimizer Convergence Dynamics")
    st.caption("Step-by-step energy minimization trajectory during SLSQP classical optimization.")

    available_distances = [p.distance for p in scan_result.points]
    default_dist_idx = int(np.argmin([abs(d - mol_def.equilibrium_distance) for d in available_distances]))

    sel_dist = st.selectbox(
        "Select Bond Distance (Å) to Inspect SLSQP Trajectory",
        options=available_distances,
        index=default_dist_idx,
        format_func=lambda d: f"{d:.3f} Å {'(Near Equilibrium)' if abs(d - mol_def.equilibrium_distance) < 0.05 else ''}",
    )

    chosen_pt = next(p for p in scan_result.points if abs(p.distance - sel_dist) < 1e-5)
    hist = chosen_pt.convergence_history

    if hist:
        steps = [h[0] for h in hist]
        vals = [h[1] for h in hist]

        fig_conv = go.Figure()
        fig_conv.add_trace(
            go.Scatter(
                x=steps,
                y=vals,
                mode="lines+markers",
                line=dict(color="#a855f7", width=3),
                marker=dict(size=6, color="#c084fc"),
                name="Optimizer Step Value",
                hovertemplate="Evaluation Step %{x}: %{y:.8f} Ha<extra></extra>",
            )
        )

        fig_conv.update_layout(
            template="plotly_dark",
            title=f"SLSQP Parameter Optimization Trajectory at R = {chosen_pt.distance:.3f} Å ({len(hist)} evaluations)",
            xaxis_title="Function Evaluation Count",
            yaxis_title="Hamiltonian Expectation Value (Hartree)",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,46,0.4)",
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("Total Function Evaluations", chosen_pt.iterations)
        with col_c2:
            st.metric("Initial Step Energy", f"{vals[0]:.6f} Ha")
        with col_c3:
            st.metric("Net Energy Decrement ΔE", f"{(vals[0] - vals[-1]):.6f} Ha")

        st.caption(
            "Note: The optimizer step trajectory records the expectation value of the electronic qubit Hamiltonian. "
            "Final total energies include the classical nuclear repulsion constant."
        )
    else:
        st.info("No detailed step history recorded for this point.")

# ---------------------------------------------------------------------------
# TAB 4: ENERGY ANALYSIS & NUMERICAL TABLES
# ---------------------------------------------------------------------------
with tab_energy:
    st.subheader(f"Energy Decomposition & Numerical Analysis: {mol_def.formula}")
    st.caption("Point-by-point numerical records, harmonic well curvature, and data export.")

    col_en1, col_en2 = st.columns(2)
    with col_en1:
        st.markdown("#### 📐 Harmonic Well Parameters")
        st.markdown(
            f"""
            - **Equilibrium Minimum $R_{{eq}}$ (Proxy):** `{metrics.equilibrium_bond_length_proxy_angstrom:.4f} Å`
            - **Dissociation Limit Reference $R_{{dissoc}}$:** `{scan_result.points[-1].distance:.3f} Å`
            - **Well Depth $D_e$ (Hartree):** `{metrics.dissociation_energy_proxy_hartree:.6f} Ha`
            - **Well Depth $D_e$ (eV):** `{metrics.dissociation_energy_proxy_ev:.4f} eV`
            - **Well Depth $D_e$ (kcal/mol):** `{metrics.dissociation_energy_proxy_kcal_mol:.2f} kcal/mol`
            - **Harmonic Force Constant $k$:** `{metrics.harmonic_force_constant_proxy_ha_per_ang2:.4f} Ha/Å²`
            """
        )
    with col_en2:
        st.markdown("#### 🔬 Accuracy Metrics Summary")
        st.markdown(
            f"""
            - **Points Evaluated:** `{len(scan_result.points)} points`
            - **Points within Chemical Accuracy (<1.6 mHa):** `{scan_result.n_within_chemical_accuracy} / {len(scan_result.points)}`
            - **Mean Absolute Error:** `{metrics.average_vqe_error_proxy_mha:.5f} mHa`
            - **Maximum Absolute Error:** `{metrics.max_vqe_error_proxy_mha:.5f} mHa`
            - **Total Scan Wall Time:** `{scan_result.total_scan_time_seconds:.2f} s`
            """
        )

    st.markdown("#### 📋 Full Numerical Simulation Dataset")
    formatted_df = df_points.copy()
    formatted_df["VQE Energy (Ha)"] = formatted_df["VQE Energy (Ha)"].map(lambda x: f"{x:.8f}")
    formatted_df["Exact Energy (Ha)"] = formatted_df["Exact Energy (Ha)"].map(lambda x: f"{x:.8f}")
    formatted_df["Error (mHa)"] = formatted_df["Error (mHa)"].map(lambda x: f"{x:.5f}")
    formatted_df["Error (eV)"] = formatted_df["Error (eV)"].map(lambda x: f"{x:.7f}")
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    # CSV Download Button
    csv_bytes = df_points.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {selected_mol_name} Numerical PES Data (CSV)",
        data=csv_bytes,
        file_name=f"{selected_mol_name}_q_lithos_pes.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# TAB 5: BATTERY ASSESSMENT
# ---------------------------------------------------------------------------
with tab_battery:
    st.subheader(f"Explainable Battery Material Assessment: {mol_def.formula}")
    st.caption("Evidence-backed evaluation grounded strictly in computational quantum results and established literature.")

    # Conclusion Banner
    conc_badge_classes = {
        AssessmentConclusion.PROMISING: "conc-promising",
        AssessmentConclusion.MIXED_EVIDENCE: "conc-mixed",
        AssessmentConclusion.LOW_PRIORITY: "conc-mixed",
        AssessmentConclusion.INSUFFICIENT_EVIDENCE: "conc-insufficient",
    }
    badge_class = conc_badge_classes.get(assessment.conclusion, "conc-mixed")

    st.markdown(
        f"""
        <div style="background: rgba(13, 27, 46, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">
                Assessment Conclusion
            </div>
            <div class="{badge_class}">
                {assessment.conclusion.value}
            </div>
            <div style="margin-top: 12px; font-size: 0.95rem; color: #cbd5e1;">
                <strong>Confidence Level:</strong> <span style="color: #38bdf8;">{assessment.confidence.value}</span> — {assessment.confidence_rationale}
            </div>
            <div style="margin-top: 10px; font-size: 0.92rem; color: #e2e8f0; line-height: 1.5;">
                {assessment.summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two columns: Supporting Evidence vs Missing Information
    col_ev, col_mis = st.columns(2)

    with col_ev:
        st.markdown("#### ✅ Supporting Evidence & Calculated Results")
        for ev in assessment.supporting_evidence:
            st.markdown(f"- {ev}")

    with col_mis:
        st.markdown("#### ❓ Missing Information (Beyond Gas-Phase Diatomic Scope)")
        for mis in assessment.missing_information:
            st.markdown(f"- {mis}")

    # Complete Property Breakdown Table
    st.markdown("---")
    st.markdown("#### 🔬 Comprehensive Property Availability & Status")
    prop_records = [
        {
            "Property": p.name,
            "Category": p.category,
            "Availability Status": p.status,
            "Value": f"{p.value} {p.unit}" if p.value is not None else "UNAVAILABLE",
            "Scientific Role / Assessment Notes": p.notes,
        }
        for p in assessment.property_breakdown
    ]
    st.dataframe(pd.DataFrame(prop_records), use_container_width=True, hide_index=True)

    # Scientific Limitations & Recommended Next Steps
    col_lim, col_nxt = st.columns(2)
    with col_lim:
        st.markdown("#### ⚠️ Specific Candidate Limitations")
        for lim in assessment.scientific_limitations:
            st.markdown(f"- {lim}")
    with col_nxt:
        st.markdown("#### 🚀 Recommended Next Steps")
        for nxt in assessment.recommended_next_steps:
            st.markdown(f"- {nxt}")

# ---------------------------------------------------------------------------
# TAB 6: CANDIDATE COMPARISON
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Multi-Candidate Battery Material Screening")
    st.caption("Comparative screening across all 4 battery candidates currently registered in the Q-Lithos library.")

    # Compile comparative data across all candidates
    compare_rows = []
    for mname in MOLECULES:
        try:
            mres = run_bond_length_scan(mname, recompute=False)
            mass = assess_candidate(mname, mres.metrics, MOLECULES[mname])
            compare_rows.append({
                "Candidate": mname,
                "Formula": MOLECULES[mname].formula,
                "Battery Subsystem Role": MOLECULES[mname].battery_role,
                "Category": MOLECULES[mname].category_label,
                "Conclusion": mass.conclusion.value,
                "R_eq (Å) [Proxy]": round(mres.metrics.equilibrium_bond_length_proxy_angstrom, 3),
                "Dissociation D_e (eV) [Proxy]": round(mres.metrics.dissociation_energy_proxy_ev, 3),
                "Force Constant k (Ha/Å²)": round(mres.metrics.harmonic_force_constant_proxy_ha_per_ang2, 3),
                "Avg VQE Error (mHa)": round(mres.metrics.average_vqe_error_proxy_mha, 4),
                "Chem. Accuracy": "Passed" if mres.metrics.chemical_accuracy_achieved else "Exceeded",
            })
        except Exception:
            pass

    if compare_rows:
        df_comp = pd.DataFrame(compare_rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        col_cp1, col_cp2 = st.columns(2)

        # Bar chart: Dissociation Energy Proxies
        with col_cp1:
            fig_bar_de = go.Figure()
            fig_bar_de.add_trace(
                go.Bar(
                    x=df_comp["Candidate"],
                    y=df_comp["Dissociation D_e (eV) [Proxy]"],
                    marker=dict(color=["#38bdf8", "#34d399", "#a855f7", "#f43f5e"]),
                    text=df_comp["Dissociation D_e (eV) [Proxy]"].map(lambda x: f"{x:.2f} eV"),
                    textposition="auto",
                )
            )
            fig_bar_de.update_layout(
                template="plotly_dark",
                title="Molecular Dissociation Energy D_e Proxy (eV)",
                xaxis_title="Battery Fragment",
                yaxis_title="Dissociation Energy (eV)",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,27,46,0.4)",
            )
            st.plotly_chart(fig_bar_de, use_container_width=True)

        # Bar chart: Equilibrium Bond Length Proxies
        with col_cp2:
            fig_bar_req = go.Figure()
            fig_bar_req.add_trace(
                go.Bar(
                    x=df_comp["Candidate"],
                    y=df_comp["R_eq (Å) [Proxy]"],
                    marker=dict(color=["#06b6d4", "#10b981", "#8b5cf6", "#ec4899"]),
                    text=df_comp["R_eq (Å) [Proxy]"].map(lambda x: f"{x:.3f} Å"),
                    textposition="auto",
                )
            )
            fig_bar_req.update_layout(
                template="plotly_dark",
                title="Equilibrium Bond Distance R_eq Proxy (Å)",
                xaxis_title="Battery Fragment",
                yaxis_title="Distance (Å)",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,27,46,0.4)",
            )
            st.plotly_chart(fig_bar_req, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 7: SCIENTIFIC METHODS & LIMITATIONS
# ---------------------------------------------------------------------------
with tab_methods:
    st.subheader("Scientific Methodology, Algorithmic Foundation & Limitations")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### ⚛️ Quantum Algorithm Pipeline")
        st.markdown(
            f"""
            1. **Electronic Structure Driver:**
               Uses `PySCFDriver` to compute 1- and 2-electron Coulomb & exchange integrals in second quantization:
               $$H = \\sum_{{pq}} h_{{pq}} a_p^\\dagger a_q + \\frac{{1}}{{2}} \\sum_{{pqrs}} g_{{pqrs}} a_p^\\dagger a_q^\\dagger a_s a_r + E_{{nuc}}$$
            2. **Active Space Reduction:**
               `ActiveSpaceTransformer` isolates the active valence bonding electrons ({mol_def.num_active_electrons}e) and spatial orbitals ({mol_def.num_active_orbitals}o), freezing inner core electrons.
            3. **Fermion-to-Qubit Transformation:**
               `ParityMapper` exploits particle-number parity symmetry $\\mathbb{{Z}}_2$ to eliminate two spin-orbitals, reducing the 4 spin-orbital space down to **2 active qubits**.
            4. **Ansatz Circuit:**
               Unitary Coupled Cluster Singles & Doubles (`UCCSD`):
               $$|\\Psi(\\vec{{\\theta}})\\rangle = \\exp(T(\\vec{{\\theta}}) - T^\\dagger(\\vec{{\\theta}})) |\\text{{HF}}\\rangle$$
            5. **Classical Optimization:**
               Sequential Least Squares Programming (`SLSQP`) optimizes $\\vec{{\\theta}}$ to minimize $\\langle H \\rangle$.
            """
        )

    with col_m2:
        st.markdown("#### ⚠️ Honest Scientific Limitations")
        st.markdown(
            """
            1. **Gas-Phase Diatomic Model vs Bulk Crystal Lattice:**
               Real battery components (such as solid electrolytes, cathodes, or composite SEIs) possess long-range 
               periodic Madelung potentials and band structures. An isolated diatomic molecule cannot predict 
               bulk transport, lattice migration barriers, or cell voltage.
            2. **Minimal Basis Approximation (STO-3G):**
               STO-3G is a minimal Gaussian representation. While computationally efficient, it lacks polarization 
               and diffuse basis functions needed to capture long-range dispersion and anion polarizability (e.g., F⁻, H⁻).
            3. **Classical Simulation of Quantum Algorithms:**
               This dashboard currently uses ideal local statevector simulation (`StatevectorEstimator`). 
               It does not execute on physical quantum hardware or simulate NISQ device noise (relaxation $T_1$, 
               dephasing $T_2$, or gate errors).
            4. **Zero-Point Vibrational Energy (ZPVE):**
               The reported curves represent electronic potential energy surfaces $E(R)$. Real experimental 
               dissociation requires zero-point vibrational correction: $D_0 = D_e - \\frac{1}{2}h\\nu$.
            """
        )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 12px 0;">
        <strong>Q-Lithos: Hybrid Quantum-Classical Simulation Engine</strong> • Developed with Qiskit 1.x & Streamlit<br>
        All derived indicators are explicitly labeled as quantum-mechanical molecular proxies.
    </div>
    """,
    unsafe_allow_html=True,
)
