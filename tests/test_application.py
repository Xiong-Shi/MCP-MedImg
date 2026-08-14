"""Tests for Layer 4: Application Semantics Layer"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from mcp_medimg.application import (
    InspectionRouter,
    LesionEncoder,
    VerificationRouter,
    VerificationStateMachine,
)
from mcp_medimg.security import PolicyEngine


class TestInspectionRouter:
    def setup_method(self):
        self.router = InspectionRouter()

    def test_lung_cancer_screening(self):
        assert self.router.route("CT", "CHEST", "screening") == "LUNG_CANCER_SCREENING"

    def test_nodule_followup(self):
        assert self.router.route("CT", "CHEST", "followup") == "NODULE_FOLLOWUP"

    def test_breast_screening(self):
        assert self.router.route("MG", "BREAST", "screening") == "BREAST_SCREENING"

    def test_bone_age(self):
        assert self.router.route("XR", "HAND", "bone_age") == "BONE_AGE_ASSESSMENT"

    def test_general_fallback(self):
        assert self.router.route("MR", "BRAIN", "screening") == "GENERAL"

    def test_supported_types(self):
        types = self.router.get_supported_types()
        assert "LUNG_CANCER_SCREENING" in types
        assert "BREAST_SCREENING" in types


class TestLesionEncoder:
    def setup_method(self):
        self.encoder = LesionEncoder()

    def test_lung_rads_1(self):
        assert self.encoder.encode_rads(4.0, "solid") == "Lung-RADS 1"

    def test_lung_rads_2(self):
        assert self.encoder.encode_rads(8.0, "solid") == "Lung-RADS 2"

    def test_lung_rads_3(self):
        assert self.encoder.encode_rads(8.0, "subsolid") == "Lung-RADS 3"

    def test_lung_rads_4a(self):
        assert self.encoder.encode_rads(12.0, "solid") == "Lung-RADS 4A"

    def test_lung_rads_4b(self):
        assert self.encoder.encode_rads(20.0, "solid") == "Lung-RADS 4B"

    def test_birads_2(self):
        assert self.encoder.encode_birads("round", "circumscribed") == "BI-RADS 2"

    def test_birads_4(self):
        assert self.encoder.encode_birads("irregular", "spiculated") == "BI-RADS 4"

    def test_birads_3(self):
        assert self.encoder.encode_birads("oval", "obscured") == "BI-RADS 3"

    def test_encode_rads_from_findings(self):
        findings = [
            {"type": "lung_nodule", "size_mm": 8.0, "nodule_type": "solid"},
            {"type": "lung_nodule", "size_mm": 20.0, "nodule_type": "solid"},
        ]
        encoded = self.encoder.encode_rads_from_findings(findings)
        assert encoded[0]["rads_category"] == "Lung-RADS 2"
        assert encoded[1]["rads_category"] == "Lung-RADS 4B"

    def test_encode_birads_from_findings(self):
        findings = [
            {
                "type": "breast_lesion",
                "mass_shape": "round",
                "mass_margin": "circumscribed",
                "calcification": "none",
            },
            {
                "type": "breast_lesion",
                "mass_shape": "irregular",
                "mass_margin": "spiculated",
                "calcification": "suspicious",
            },
        ]
        encoded = self.encoder.encode_birads_from_findings(findings)
        assert encoded[0]["rads_category"] == "BI-RADS 2"
        assert encoded[1]["rads_category"] == "BI-RADS 4"


class TestVerificationRouter:
    def setup_method(self):
        policy_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "policy.json"
        )
        self.policy = PolicyEngine(policy_file=policy_path)
        self.router = VerificationRouter(policy_engine=self.policy)

    def test_stat_priority_auto(self):
        result = self.router.determine_mode("LUNG_CANCER_SCREENING", "STAT")
        assert result["mode"] == "AUTO"
        assert "STAT" in result["reason"]

    def test_mandatory_for_lung_screening(self):
        result = self.router.determine_mode("LUNG_CANCER_SCREENING", "ROUTINE")
        assert result["mode"] == "MANDATORY"

    def test_auto_for_followup(self):
        result = self.router.determine_mode("NODULE_FOLLOWUP", "ROUTINE")
        assert result["mode"] == "AUTO"

    def test_advisory_for_bone_age(self):
        result = self.router.determine_mode("BONE_AGE_ASSESSMENT", "ROUTINE")
        assert result["mode"] == "ADVISORY"

    def test_new_model_mandatory(self):
        self.router.register_deployment("new-model-v1")
        result = self.router.determine_mode(
            "GENERAL", "ROUTINE", model_id="new-model-v1"
        )
        assert result["mode"] == "MANDATORY"
        assert "warmup" in result["reason"]

    def test_warmup_completion(self):
        self.router.register_deployment("test-model")
        # Process 55 cases (warmup is 50)
        for _ in range(55):
            self.router.record_case("test-model")
        result = self.router.determine_mode("GENERAL", "ROUTINE", model_id="test-model")
        # After warmup, should not be MANDATORY due to new deployment
        assert result["mode"] != "MANDATORY" or "warmup" not in result["reason"]

    def test_get_workflow_mandatory(self):
        wf = self.router.get_workflow("MANDATORY")
        assert "radiologist_review" in wf["required_actions"]
        assert wf["timeout_seconds"] == 300

    def test_get_workflow_auto(self):
        wf = self.router.get_workflow("AUTO")
        assert len(wf["required_actions"]) == 0

    def test_get_workflow_advisory(self):
        wf = self.router.get_workflow("ADVISORY")
        assert wf["timeout_seconds"] == 86400


class TestVerificationStateMachine:
    def setup_method(self):
        self.vsm = VerificationStateMachine("ver_001", timeout_seconds=300)

    def test_initial_state(self):
        assert self.vsm.state == "PENDING"

    def test_valid_transition(self):
        self.vsm.transition_to("UNDER_REVIEW")
        assert self.vsm.state == "UNDER_REVIEW"

    def test_full_workflow(self):
        self.vsm.transition_to("UNDER_REVIEW")
        self.vsm.transition_to("REVIEWED")
        self.vsm.transition_to("CONFIRMED")
        self.vsm.transition_to("FINALIZED")
        assert self.vsm.state == "FINALIZED"
        assert self.vsm.is_terminal()

    def test_modify_workflow(self):
        self.vsm.transition_to("UNDER_REVIEW")
        self.vsm.transition_to("REVIEWED")
        self.vsm.transition_to("MODIFIED")
        self.vsm.transition_to("CONFIRMED")
        self.vsm.transition_to("FINALIZED")
        assert self.vsm.state == "FINALIZED"

    def test_invalid_transition(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            self.vsm.transition_to("FINALIZED")

    def test_invalid_state(self):
        with pytest.raises(ValueError, match="Invalid state"):
            self.vsm.transition_to("INVALID_STATE")

    def test_history_tracking(self):
        self.vsm.transition_to("UNDER_REVIEW", reason="Radiologist assigned")
        history = self.vsm.get_history()
        assert len(history) == 2
        assert history[1]["reason"] == "Radiologist assigned"

    def test_timeout_escalation(self):
        self.vsm.transition_to("UNDER_REVIEW")
        self.vsm.transition_to("TIMEOUT")
        assert self.vsm.state == "TIMEOUT"
        assert self.vsm.is_terminal()
        history = self.vsm.get_history()
        assert any(h.get("state") == "ESCALATED" for h in history)
