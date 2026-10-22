"""Q-Lithos: Hybrid Quantum-Classical Simulation Engine for Battery Materials.

Interactive Streamlit Dashboard for Hackathon Presentation.
"""

from __future__ import annotations
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Q-Lithos core modules
from molecules import MOLECULES, get_molecule, MAX_QUBIT_BUDGET
from scan import run_bond_length_scan, get_cache_path, ScanResult
from metrics import BatteryMaterialMetrics, HARTREE_TO_EV, HARTREE_TO_KCAL_MOL

# Load environment configuration (.env)
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="Q-Lithos | Quantum Battery Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished hackathon presentation styling
st.markdown(
    """
    <style>
    /* Global styles */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 28, 48, 0.95), rgba(28, 44, 76, 0.95));
        border: 1px solid rgba(86, 156, 255, 0.25);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4ef2bb, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 16px;
    }
    .context-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .accuracy-badge-pass {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
        border: 1px solid #10b981;
        color: #34d399;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 2px;
    }
    .disclaimer-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        color: #fde68a;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
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

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.15); margin-bottom: 16px;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem;">{mol_def.formula} • {mol_def.battery_role}</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">{mol_def.description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quantum Backend Configuration
    st.markdown("#### ⚛️ Quantum Execution Backend")
    backend_choice = st.radio(
        "Simulation Mode",
        options=["Aer Simulator (Local)", "IBM Quantum (Hardware/Cloud)"],
        index=0,
        help="Select between local Qiskit Aer Statevector simulator or live IBM Quantum backend.",
    )

    ibm_token_set = bool(os.getenv("IBMQ_TOKEN") or os.getenv("IBM_QUANTUM_TOKEN"))
    if backend_choice.startswith("IBM"):
        if ibm_token_set:
            st.success("🟢 IBM Quantum Token detected in .env")
        else:
            st.warning("⚠️ No IBM Quantum Token found in .env. Will fall back to local Aer simulator.")

    # Quantum Budget Indicators
    st.markdown("#### 🛡️ Hardware Constraints")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.metric("Qubits Used", "2 Qubits")
    with col_b2:
        st.metric("Max Budget", f"{MAX_QUBIT_BUDGET} Qubits")

    st.caption(
        "Active Space: 2 electrons in 2 spatial orbitals. "
        "Mapped via ParityMapper with 2-qubit reduction (4 spin-orbitals → 2 qubits)."
    )

    # Recompute Button
    st.markdown("---")
    cache_path = get_cache_path(selected_mol_name)
    has_cache = cache_path.exists()

    if has_cache:
        st.caption("⚡ Cache status: **Precomputed dataset ready**")
    else:
        st.caption("⚠️ Cache status: **Not yet computed**")

    recompute_btn = st.button("🔄 Recompute on Quantum Engine", use_container_width=True)

# ---------------------------------------------------------
# DATA LOADING / EXECUTION
# ---------------------------------------------------------
progress_placeholder = st.empty()
scan_result: Optional[ScanResult] = None

if recompute_btn:
    with progress_placeholder.container():
        st.info(f"⏳ Running real quantum VQE simulation across 10 bond lengths for {selected_mol_name}...")
        prog_bar = st.progress(0)
        status_text = st.empty()

        def on_step(curr, total, pt):
            prog_bar.progress(curr / total)
            status_text.write(
                f"Sampling R = {pt.distance:.3f} Å | VQE: {pt.vqe_energy:.6f} Ha | Exact: {pt.exact_energy:.6f} Ha (Error: {pt.error_mha:.4f} mHa)"
            )

        scan_result = run_bond_length_scan(
            molecule_name=selected_mol_name,
            recompute=True,
            backend_mode="Aer Simulator" if "Aer" in backend_choice else "IBM Quantum",
            point_callback=on_step,
        )
        time.sleep(0.5)
    progress_placeholder.empty()
else:
    # Load from cache (or compute if cache absent)
    if not has_cache:
        with st.spinner(f"Computing first-time quantum potential energy curve for {selected_mol_name}..."):
            scan_result = run_bond_length_scan(selected_mol_name, recompute=True)
    else:
        scan_result = run_bond_length_scan(selected_mol_name, recompute=False)

metrics = scan_result.metrics

# ---------------------------------------------------------
# HERO BANNER & HACKATHON CONTEXT
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">Q-Lithos: Hybrid Quantum-Classical Simulation Engine</div>
        <div class="hero-subtitle">
            Accelerating battery materials screening by resolving strong electron interactions in solid-state and sodium-ion molecular fragments via VQE.
        </div>
        <div>
            <span class="context-pill">⚡ Problem: Classical Supercomputer Scaling Limits</span>
            <span class="context-pill">🔬 Solution: Qiskit VQE + Active Space Truncation</span>
            <span class="context-pill">📊 Impact: Quantitative Quantum Proxy Metrics</span>
            <span class="context-pill">🎯 Molecule: {mol_def.formula} ({mol_def.battery_role})</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# TOP KPI ACCURACY & PROXY METRICS ROW
# ---------------------------------------------------------
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

# KPI 1: Accuracy Badge
with col_kpi1:
    avg_err = metrics.average_vqe_error_proxy_mha
    max_err = metrics.max_vqe_error_proxy_mha
    is_chemical_accuracy = avg_err < 1.6  # 1.6 mHa = 1 kcal/mol standard

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">VQE Accuracy Badge (vs Exact)</div>
            <div class="metric-value" style="color: {'#34d399' if is_chemical_accuracy else '#f59e0b'};">
                {avg_err:.4f} <span style="font-size: 1rem; color: #94a3b8;">mHa</span>
            </div>
            <div class="metric-sub">Max point error: {max_err:.4f} mHa</div>
            <div style="margin-top: 8px;">
                <span style="background: {'rgba(16, 185, 129, 0.2)' if is_chemical_accuracy else 'rgba(245, 158, 11, 0.2)'}; 
                             color: {'#34d399' if is_chemical_accuracy else '#f59e0b'}; 
                             padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;">
                    {'✓ CHEMICAL ACCURACY (< 1.6 mHa)' if is_chemical_accuracy else 'APPROACHING TARGET'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# KPI 2: Equilibrium Bond Length Proxy
with col_kpi2:
    r_eq = metrics.equilibrium_bond_length_proxy_angstrom
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Equilibrium Distance (Proxy)</div>
            <div class="metric-value" style="color: #38bdf8;">
                {r_eq:.3f} <span style="font-size: 1rem; color: #94a3b8;">Å</span>
            </div>
            <div class="metric-sub">Literature equilibrium: ~{mol_def.equilibrium_distance:.3f} Å</div>
            <div style="margin-top: 8px;">
                <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Interfacial Packing Proxy
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# KPI 3: Binding Energy Proxy
with col_kpi3:
    b_ev = metrics.binding_energy_proxy_ev
    b_kcal = metrics.binding_energy_proxy_kcal_mol
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Binding Energy (Proxy)</div>
            <div class="metric-value" style="color: #a855f7;">
                {b_ev:.3f} <span style="font-size: 1rem; color: #94a3b8;">eV</span>
            </div>
            <div class="metric-sub">{b_kcal:.1f} kcal/mol ({metrics.binding_energy_proxy_hartree:.5f} Ha)</div>
            <div style="margin-top: 8px;">
                <span style="background: rgba(168, 85, 247, 0.15); color: #c084fc; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Interfacial Cohesion Proxy
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# KPI 4: Harmonic Force Constant Proxy
with col_kpi4:
    k_force = metrics.harmonic_force_constant_proxy_ha_per_ang2
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Force Constant (Proxy)</div>
            <div class="metric-value" style="color: #f43f5e;">
                {k_force:.3f} <span style="font-size: 1rem; color: #94a3b8;">Ha/Å²</span>
            </div>
            <div class="metric-sub">Curvature d²E/dR² at equilibrium</div>
            <div style="margin-top: 8px;">
                <span style="background: rgba(244, 63, 94, 0.15); color: #fb7185; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">
                    Mechanical Rigidity Proxy
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# INTERACTIVE TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Potential Energy Surface (VQE vs Exact)",
    "🎯 VQE Optimizer Convergence",
    "🧪 Battery Material Proxy Metrics",
    "⚛️ Quantum Circuit & Pipeline Architecture",
])

# ---------------------------------------------------------
# TAB 1: POTENTIAL ENERGY SURFACE
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Potential Energy Curve: {mol_def.formula} ({mol_def.battery_role})")
    st.caption(
        "Direct comparison of Variational Quantum Eigensolver (VQE with UCCSD) against classical "
        "exact diagonalization across 10 bond lengths."
    )

    df_points = pd.DataFrame([
        {
            "Distance (Å)": p.distance,
            "VQE Energy (Ha)": p.vqe_energy,
            "Exact Energy (Ha)": p.exact_energy,
            "Error (mHa)": p.error_mha,
            "Iterations": p.iterations,
            "Qubits": p.qubit_count,
        }
        for p in scan_result.points
    ])

    # Plotly Potential Energy Surface Chart
    fig_pes = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.75, 0.25],
        subplot_titles=("Total Ground State Energy (Hartree)", "Absolute Error: VQE vs Exact (mHa)"),
    )

    # VQE trace
    fig_pes.add_trace(
        go.Scatter(
            x=df_points["Distance (Å)"],
            y=df_points["VQE Energy (Ha)"],
            mode="lines+markers",
            name="VQE (Aer Statevector + UCCSD)",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=8, color="#38bdf8", symbol="circle"),
            hovertemplate="Distance: %{x:.3f} Å<br>VQE Energy: %{y:.6f} Ha<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Exact Reference trace
    fig_pes.add_trace(
        go.Scatter(
            x=df_points["Distance (Å)"],
            y=df_points["Exact Energy (Ha)"],
            mode="lines",
            name="Exact Diagonalization",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            hovertemplate="Distance: %{x:.3f} Å<br>Exact Energy: %{y:.6f} Ha<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Equilibrium annotation
    fig_pes.add_annotation(
        x=metrics.equilibrium_bond_length_proxy_angstrom,
        y=float(df_points["VQE Energy (Ha)"].min()),
        text=f"R_eq (proxy) ~ {metrics.equilibrium_bond_length_proxy_angstrom:.3f} Å",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor="#34d399",
        font=dict(color="#34d399", size=11, family="monospace"),
        ax=40,
        ay=-35,
        row=1,
        col=1,
    )

    # Error bar trace (lower subplot)
    fig_pes.add_trace(
        go.Bar(
            x=df_points["Distance (Å)"],
            y=df_points["Error (mHa)"],
            name="VQE Error (mHa)",
            marker=dict(
                color=df_points["Error (mHa)"],
                colorscale="Viridis",
                showscale=False,
            ),
            hovertemplate="Distance: %{x:.3f} Å<br>Error: %{y:.4f} mHa<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 1.6 mHa chemical accuracy threshold line
    fig_pes.add_hline(
        y=1.6,
        line=dict(color="#ef4444", width=1, dash="dot"),
        annotation_text="Chemical Accuracy Threshold (1.6 mHa = 1 kcal/mol)",
        annotation_position="top right",
        annotation_font=dict(size=9, color="#ef4444"),
        row=2,
        col=1,
    )

    fig_pes.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig_pes.update_xaxes(title_text="Bond Distance R (Å)", row=2, col=1)

    st.plotly_chart(fig_pes, use_container_width=True)

    # Detailed Numerical Data Table
    with st.expander("📋 View Real Numerical Simulation Data Table", expanded=False):
        formatted_df = df_points.copy()
        formatted_df["VQE Energy (Ha)"] = formatted_df["VQE Energy (Ha)"].map(lambda x: f"{x:.8f}")
        formatted_df["Exact Energy (Ha)"] = formatted_df["Exact Energy (Ha)"].map(lambda x: f"{x:.8f}")
        formatted_df["Error (mHa)"] = formatted_df["Error (mHa)"].map(lambda x: f"{x:.5f}")
        st.dataframe(formatted_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: VQE CONVERGENCE DYNAMICS
# ---------------------------------------------------------
with tab2:
    st.subheader(f"VQE Optimization Convergence Traces for {mol_def.formula}")
    st.caption("Step-by-step energy trajectory tracked via the VQE callback during SLSQP classical optimization.")

    # Distance selector for convergence plot
    available_distances = [p.distance for p in scan_result.points]
    default_dist_idx = int(np.argmin([abs(d - mol_def.equilibrium_distance) for d in available_distances]))

    sel_dist = st.selectbox(
        "Select Bond Distance (Å) to Inspect Convergence",
        options=available_distances,
        index=default_dist_idx,
        format_func=lambda d: f"{d:.3f} Å {'(Near Equilibrium)' if abs(d - mol_def.equilibrium_distance) < 0.05 else ''}",
    )

    # Find the corresponding point
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
                name="VQE Step Energy",
                hovertemplate="Step %{x}: %{y:.8f} Ha<extra></extra>",
            )
        )
        # Add target line
        fig_conv.add_hline(
            y=chosen_pt.exact_energy,
            line=dict(color="#f59e0b", width=2, dash="dash"),
            annotation_text=f"Exact Energy ({chosen_pt.exact_energy:.6f} Ha)",
            annotation_position="bottom right",
            annotation_font=dict(color="#f59e0b", size=10),
        )

        fig_conv.update_layout(
            template="plotly_dark",
            title=f"SLSQP Energy Minimization at R = {chosen_pt.distance:.3f} Å ({len(hist)} steps)",
            xaxis_title="Function Evaluation Step",
            yaxis_title="Total Energy (Hartree)",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("Total Iterations", chosen_pt.iterations)
        with col_c2:
            st.metric("Initial Step Energy", f"{vals[0]:.6f} Ha")
        with col_c3:
            st.metric("Energy Decrement ΔE", f"{(vals[0] - vals[-1]):.6f} Ha")
    else:
        st.info("No detailed step history recorded for this point.")

# ---------------------------------------------------------
# TAB 3: BATTERY MATERIAL PROXY METRICS & COMPARISON
# ---------------------------------------------------------
with tab3:
    st.subheader("Quantitative Quantum Battery Indicators (Proxy Metrics)")

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ MANDATORY SCIENTIFIC PROXY GOVERNANCE:</strong><br>
            All quantities derived below represent <em>first-principles quantum-mechanical molecular proxies</em>
            evaluated on small gas-phase cluster fragments. They provide fundamental chemical bonding baselines.
            They are <strong>NEVER</strong> to be confused with macroscopic cell properties such as bulk ionic conductivity,
            macroscopic ion mobility, operational thermal run-away stability, or multi-year cycle life.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("#### 📊 Derived Proxy Metrics Table")
        metrics_table_data = [
            ("Equilibrium Bond Length (Proxy)", f"{metrics.equilibrium_bond_length_proxy_angstrom:.4f} Å", "Interatomic packing density proxy"),
            ("Binding / Dissociation Energy (Proxy)", f"{metrics.binding_energy_proxy_ev:.4f} eV", f"{metrics.binding_energy_proxy_kcal_mol:.1f} kcal/mol"),
            ("Energy-Gap Proxy", f"{metrics.energy_gap_proxy_ev:.4f} eV", "Cluster dissociation resistance proxy"),
            ("Harmonic Force Constant (Proxy)", f"{metrics.harmonic_force_constant_proxy_ha_per_ang2:.4f} Ha/Å²", "Interfacial mechanical rigidity proxy"),
            ("Average VQE Error (Proxy)", f"{metrics.average_vqe_error_proxy_mha:.5f} mHa", "VQE vs Exact diagonalization accuracy"),
            ("Maximum VQE Error (Proxy)", f"{metrics.max_vqe_error_proxy_mha:.5f} mHa", "Worst-case deviation along the scan"),
        ]
        df_metric_table = pd.DataFrame(metrics_table_data, columns=["Metric (Proxy)", "Value", "Physical Context"])
        st.dataframe(df_metric_table, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("#### 🔬 Material Role in Next-Gen Batteries")
        st.markdown(
            f"""
            **Molecule:** `{mol_def.formula}` ({mol_def.name})  
            **Battery Subsystem:** `{mol_def.battery_role}`  

            **Chemical Context:**  
            {mol_def.description}

            **Engineering Interpretation:**
            - **Binding Energy Proxy ({metrics.binding_energy_proxy_ev:.2f} eV):** Measures the energetic threshold required to pull the constituent atoms apart. High binding proxies correlate with passivation layers that resist spontaneous chemical dissolution into liquid or polymer electrolytes.
            - **Equilibrium Distance Proxy ({metrics.equilibrium_bond_length_proxy_angstrom:.3f} Å):** Corresponds to the minimum of the potential well. In solid-state interphases, compact bond distances generally indicate high packing density, which can suppress void formation during volume cycling.
            """
        )

    # Multi-molecule benchmark comparison
    st.markdown("---")
    st.markdown("#### 🌐 Multi-Material Screening Comparison")
    st.caption("Comparative screening across all 4 battery materials currently registered in the Q-Lithos library.")

    # Load all cached scan results
    all_metrics = []
    for mname in MOLECULES:
        try:
            mres = run_bond_length_scan(mname, recompute=False)
            all_metrics.append({
                "Molecule": mname,
                "Role": MOLECULES[mname].battery_role,
                "Equilibrium Dist (Å) [Proxy]": mres.metrics.equilibrium_bond_length_proxy_angstrom,
                "Binding Energy (eV) [Proxy]": mres.metrics.binding_energy_proxy_ev,
                "Force Constant (Ha/Å²) [Proxy]": mres.metrics.harmonic_force_constant_proxy_ha_per_ang2,
                "Avg Error (mHa)": mres.metrics.average_vqe_error_proxy_mha,
            })
        except Exception:
            pass

    if all_metrics:
        df_compare = pd.DataFrame(all_metrics)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # Bar chart comparing Binding Energy Proxies
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=df_compare["Molecule"],
                y=df_compare["Binding Energy (eV) [Proxy]"],
                marker=dict(color=["#38bdf8", "#34d399", "#a855f7", "#f43f5e"]),
                text=df_compare["Binding Energy (eV) [Proxy]"].map(lambda x: f"{x:.2f} eV"),
                textposition="auto",
            )
        )
        fig_bar.update_layout(
            template="plotly_dark",
            title="Binding Energy Proxy Comparison (eV)",
            xaxis_title="Battery Material Fragment",
            yaxis_title="Binding Energy Proxy (eV)",
            height=360,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: QUANTUM CIRCUIT & ARCHITECTURE
# ---------------------------------------------------------
with tab4:
    st.subheader("Quantum-Classical Hybrid Engine Architecture")
    st.markdown(
        """
        Q-Lithos leverages the **Variational Quantum Eigensolver (VQE)** algorithm tailored for chemical
        Hamiltonians through active-space selection and parity reduction.
        """
    )

    st.markdown(
        """
        ```mermaid
        flowchart LR
            A[Atomic Coordinates R] --> B[PySCF Driver STO-3G]
            B --> C[Active Space Transformer]
            C --> D[Parity Mapper + 2-Qubit Reduction]
            D --> E[Qubit Hamiltonian H]
            E --> F[VQE with SLSQP]
            G[Hartree-Fock State] --> H[UCCSD Ansatz Circuit]
            H --> F
            F --> I[Aer Statevector / IBM Quantum Backend]
            I --> F
            F --> J[Interpreted Ground State Energy]
            J --> K[Battery Proxy Metrics & PES Curve]
        ```
        """
    )

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("#### 🔧 Quantum Algorithm Configuration")
        st.markdown(
            f"""
            - **Driver:** `PySCFDriver` with minimal `sto-3g` Gaussian basis set
            - **Active Space:** {mol_def.num_active_electrons} electrons in {mol_def.num_active_orbitals} spatial orbitals
            - **Fermion-to-Qubit Mapping:** `ParityMapper` exploiting $\\mathbb{{Z}}_2$ particle-number conservation for two-qubit reduction
            - **Active Qubits:** **2 Qubits** (Strictly within the $\\le 8$ qubit threshold)
            - **Reference State:** `HartreeFock` configuration
            - **Ansatz:** `UCCSD` (Unitary Coupled Cluster Singles & Doubles) parameterized circuit
            - **Classical Optimizer:** `SLSQP` (Sequential Least Squares Programming)
            - **Quantum Primitive:** `StatevectorEstimator` / `Aer Estimator`
            """
        )

    with col_a2:
        st.markdown("#### ⚠️ Honest Limitations & Scaling Roadmap")
        st.markdown(
            """
            - **Basis Set Limitations:** Minimal STO-3G basis captures core energetic qualitative trends but underestimates polarizability compared to triple-zeta (cc-pVTZ) or augmented sets.
            - **Active Space Truncation:** Freezing inner core orbitals dramatically reduces qubit requirements ($N_{qubits} \\le 8$) while capturing valence bonding, but ignores core-valence electron correlation.
            - **Molecular Proxy Scope:** Calculations model gas-phase diatomic fragments; in bulk solid electrolytes, periodic boundary conditions (PBC) and crystalline Madelung potentials shift absolute energies.
            - **NISQ Noise:** Real quantum hardware suffers from gate infidelities and readout errors; zero-noise extrapolation (ZNE) and readout error mitigation are necessary when deploying to IBM Quantum hardware.
            """
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
        Q-Lithos • Hybrid Quantum-Classical Simulation Engine for Battery Materials • Built with Qiskit 1.x & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
