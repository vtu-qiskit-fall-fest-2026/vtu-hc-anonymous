"""Unit tests for the battery assessment module in Q-Lithos."""

import pytest
from molecules import MOLECULES, get_molecule, BatteryCategory
from scan import run_bond_length_scan
from battery_assessment import (
    assess_candidate,
    AssessmentConclusion,
    ConfidenceLevel,
    CandidateAssessment,
)


def test_assess_all_candidates():
    """Verify that all four registered molecules produce complete, valid assessments."""
    for mol_name in ["LiH", "NaH", "NaF", "Li2"]:
        scan_res = run_bond_length_scan(mol_name, recompute=False)
        mol_def = get_molecule(mol_name)
        assessment = assess_candidate(mol_name, scan_res.metrics, mol_def)

        assert isinstance(assessment, CandidateAssessment)
        assert assessment.molecule_name == mol_name
        assert assessment.formula == mol_def.formula
        assert isinstance(assessment.conclusion, AssessmentConclusion)
        assert isinstance(assessment.confidence, ConfidenceLevel)
        assert len(assessment.supporting_evidence) > 0
        assert len(assessment.missing_information) > 0
        assert len(assessment.property_breakdown) > 0
        assert len(assessment.scientific_limitations) > 0
        assert len(assessment.recommended_next_steps) > 0


def test_assessment_governance_rules():
    """Verify that conclusions and categories match scientific integrity standards."""
    # LiH should be mixed evidence (known SEI component, but gas phase)
    res_lih = run_bond_length_scan("LiH", recompute=False)
    ass_lih = assess_candidate("LiH", res_lih.metrics, get_molecule("LiH"))
    assert ass_lih.conclusion == AssessmentConclusion.MIXED_EVIDENCE
    assert ass_lih.confidence == ConfidenceLevel.MEDIUM

    # NaF should be promising (widely known fluorinated SEI component)
    res_naf = run_bond_length_scan("NaF", recompute=False)
    ass_naf = assess_candidate("NaF", res_naf.metrics, get_molecule("NaF"))
    assert ass_naf.conclusion == AssessmentConclusion.PROMISING
    assert ass_naf.confidence == ConfidenceLevel.MEDIUM

    # Li2 is a quantum test case, not a battery material
    res_li2 = run_bond_length_scan("Li2", recompute=False)
    ass_li2 = assess_candidate("Li2", res_li2.metrics, get_molecule("Li2"))
    assert ass_li2.conclusion == AssessmentConclusion.INSUFFICIENT_EVIDENCE
    assert ass_li2.confidence == ConfidenceLevel.LOW
    assert "benchmark" in ass_li2.summary.lower()


def test_missing_data_is_explicitly_labeled():
    """Ensure that bulk electrochemical properties are explicitly flagged as unavailable."""
    res = run_bond_length_scan("LiH", recompute=False)
    ass = assess_candidate("LiH", res.metrics, get_molecule("LiH"))

    unavailable_props = [p for p in ass.property_breakdown if p.status == "Unavailable"]
    assert len(unavailable_props) >= 4

    prop_names = [p.name for p in unavailable_props]
    assert "Electrochemical Working Voltage" in prop_names
    assert "Bulk Ionic Conductivity" in prop_names
    assert "Electrochemical Stability Window (ESW)" in prop_names
    assert "Cycle Life / Coulombic Efficiency" in prop_names
