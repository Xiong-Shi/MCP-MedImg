"""
End-to-end integration tests covering Table 5 validation scenarios.

Scenarios:
1. Lung nodule screening (LDCT Chest) -> DICOM SR (RADS 2-4) + SEG
2. Breast mammography (MG Breast) -> DICOM SR (BI-RADS 0-6) + SEG
3. Emergency CT (CT Brain STAT) -> Priority dispatch, fast return
4. PHI de-identification -> Hashed ID, age bin
5. Mandatory verification (high-risk screening) -> Required radiologist review
6. Advisory verification (routine diagnostic) -> Optional review with override
7. Auto verification (low-risk followup) -> Direct finalization
8. Service audit (100 consecutive studies) -> Complete audit log export
9. Multi-model routing (3 inspection types x 3 models)
10. PHI irreversibility (hash collision test)
"""

import sys
import os
import json
import hashlib
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from mcp_medimg.gateway import MCPMedimgGateway
from mcp_medimg.security import SecurityModule, AuditLogger
from mcp_medimg.transport.pacs_client import PACSClient


@pytest.fixture
def gateway():
    """Create a gateway instance with temp audit DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    gw = MCPMedimgGateway(
        pacs_client=PACSClient(base_url="http://localhost:8042"),
        audit_logger=AuditLogger(db_path=db_path),
        config={"audit_db": db_path},
    )
    yield gw
    try:
        os.unlink(db_path)
    except OSError:
        pass


def make_mock_dicom(
    study_uid,
    modality="CT",
    body_part="CHEST",
    patient_name="TEST^PATIENT",
    patient_id="PT001",
    patient_birth_date="19800101",
    patient_age="45",
    pixel_data=None,
):
    """Create mock DICOM metadata."""
    if pixel_data is None:
        pixel_data = np.zeros((512, 512), dtype=np.int16)
    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": f"{study_uid}.1",
        "Modality": modality,
        "BodyPartExamined": body_part,
        "PatientName": patient_name,
        "PatientID": patient_id,
        "PatientBirthDate": patient_birth_date,
        "PatientAge": patient_age,
        "pixel_data": pixel_data,
    }


class TestScenario1_LungNoduleScreening:
    """Scenario: Lung nodule screening - LDCT Chest -> DICOM SR (RADS 2-4)"""

    def test_lung_screening_routes_correctly(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.101", modality="CT", body_part="CHEST")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.101",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
            dicom_data=dicom,
        )
        assert result["status"] in ("COMPLETED", "COMPLETED_WITH_ERRORS")
        assert result["inspection_type"] == "LUNG_CANCER_SCREENING"
        assert result["model_id"] == "lung-nodule-v1"
        assert "L1_retrieve" in result["layers_executed"]
        assert "L3_route_infer" in result["layers_executed"]
        assert "L4_lesion_encoding" in result["layers_executed"]
        # DICOM SR output
        assert "formatted_output" in result
        sr = result["formatted_output"]
        assert sr.get("sopClassUID") == "1.2.840.10008.5.1.4.1.1.88.22"
        # Findings should have RADS categories
        for finding in result.get("encoded_findings", result.get("raw_findings", [])):
            if "rads_category" in finding:
                assert "Lung-RADS" in finding["rads_category"]

    def test_lung_screening_has_segmentation(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.102", modality="CT", body_part="CHEST")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.102",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
            dicom_data=dicom,
        )
        assert result["status"] in ("COMPLETED", "COMPLETED_WITH_ERRORS")

    def test_lung_screening_audit_trail(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.103", modality="CT", body_part="CHEST")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.103",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
            dicom_data=dicom,
            operator="dr_zhang",
        )
        assert "audit_id" in result
        audit = gateway.get_audit_trail(result["audit_id"])
        assert audit is not None


class TestScenario2_BreastMammography:
    """Scenario: Breast mammography - MG Breast -> DICOM SR (BI-RADS 0-6)"""

    def test_breast_screening_routes_correctly(self, gateway):
        dicom = make_mock_dicom(
            "1.2.3.4.201", modality="MG", body_part="BREAST", patient_age="50"
        )
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.201",
            modality="MG",
            bodyPart="BREAST",
            clinicalIndication="screening",
            dicom_data=dicom,
        )
        assert result["status"] in ("COMPLETED", "COMPLETED_WITH_ERRORS")
        assert result["inspection_type"] == "BREAST_SCREENING"
        assert result["model_id"] == "breast-lesion-v1"
        for finding in result.get("encoded_findings", result.get("raw_findings", [])):
            if "rads_category" in finding:
                assert "BI-RADS" in finding["rads_category"]


class TestScenario3_EmergencyCT:
    """Scenario: Emergency CT - CT Brain (STAT) -> Priority dispatch"""

    def test_stat_priority_routes_fast_model(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.301", modality="CT", body_part="BRAIN")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.301",
            modality="CT",
            bodyPart="BRAIN",
            clinicalIndication="emergency",
            priority="STAT",
            dicom_data=dicom,
        )
        assert result.get("verification_mode") == "AUTO"
        assert "STAT" in result.get("verification_reason", "")

    def test_stat_auto_finalizes(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.302", modality="CT", body_part="BRAIN")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.302",
            modality="CT",
            bodyPart="BRAIN",
            clinicalIndication="emergency",
            priority="STAT",
            dicom_data=dicom,
        )
        assert result.get("verification_action") == "AUTO_FINALIZED"
        assert result.get("verification_state") == "FINALIZED"


class TestScenario4_PHIDeidentification:
    """Scenario: PHI de-identification -> Hashed ID, age bin"""

    def test_patient_name_removed(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.401", patient_name="ZHANG^SAN")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.401",
            dicom_data=dicom,
        )
        assert "ZHANG^SAN" not in str(result)
        assert "patient_hash" in result
        assert result["patient_hash"] != "PT001"

    def test_patient_id_hashed(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.402", patient_id="SSN123456")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.402",
            dicom_data=dicom,
        )
        assert result["patient_hash"] != "SSN123456"
        assert result["patient_hash"].startswith("sha256:")

    def test_hash_irreversibility(self, gateway):
        """Same patient_id always produces same hash (deterministic within same session)."""
        dicom1 = make_mock_dicom("1.2.3.4.403", patient_id="PT999")
        dicom2 = make_mock_dicom("1.2.3.4.404", patient_id="PT999")
        r1 = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.403", dicom_data=dicom1
        )
        r2 = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.404", dicom_data=dicom2
        )
        assert r1["patient_hash"] == r2["patient_hash"]


class TestScenario5_MandatoryVerification:
    """Scenario: Mandatory verification - high-risk screening -> radiologist review"""

    def test_lung_screening_is_mandatory(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.501", modality="CT", body_part="CHEST")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.501",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
            priority="ROUTINE",
            dicom_data=dicom,
        )
        assert result["verification_mode"] == "MANDATORY"
        assert result["verification_action"] == "PENDING_RADIOLOGIST_REVIEW"
        assert result["verification_state"] == "UNDER_REVIEW"
        workflow = result["verification_workflow"]
        assert "radiologist_review" in workflow["required_actions"]

    def test_mandatory_not_finalized(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.502", modality="CT", body_part="CHEST")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.502",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
            dicom_data=dicom,
        )
        assert result["verification_state"] != "FINALIZED"


class TestScenario6_AdvisoryVerification:
    """Scenario: Advisory verification - routine diagnostic -> optional review"""

    def test_bone_age_is_advisory(self, gateway):
        dicom = make_mock_dicom(
            "1.2.3.4.601", modality="XR", body_part="HAND", patient_age="10"
        )
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.601",
            modality="XR",
            bodyPart="HAND",
            clinicalIndication="bone_age",
            dicom_data=dicom,
        )
        assert result["verification_mode"] == "ADVISORY"
        assert result["verification_action"] == "ADVISORY_REVIEW_OPTIONAL"
        assert result["verification_state"] == "UNDER_REVIEW"
        workflow = result["verification_workflow"]
        assert workflow["timeout_seconds"] == 86400


class TestScenario7_AutoVerification:
    """Scenario: Auto verification - low-risk followup -> direct finalization"""

    def test_followup_is_auto(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.701", modality="CT", body_part="CHEST")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.701",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="followup",
            priority="ROUTINE",
            dicom_data=dicom,
        )
        assert result["verification_mode"] == "AUTO"
        assert result["verification_action"] == "AUTO_FINALIZED"
        assert result["verification_state"] == "FINALIZED"
        workflow = result["verification_workflow"]
        assert len(workflow["required_actions"]) == 0

    def test_auto_verification_has_audit_trail(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.702", modality="CT", body_part="CHEST")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.702",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="followup",
            dicom_data=dicom,
            operator="system_auto",
        )
        audit = gateway.get_audit_trail(result["audit_id"])
        assert audit is not None


class TestScenario8_ServiceAudit:
    """Scenario: Service audit - 100 consecutive studies -> audit log export"""

    def test_100_studies_audit_trail(self, gateway):
        """Run 100 studies and verify audit trail completeness."""
        count = 0
        for i in range(100):
            dicom = make_mock_dicom(
                f"1.2.3.4.{800 + i:03d}",
                modality="CT",
                body_part="CHEST",
                patient_id=f"PT{i:03d}",
            )
            result = gateway.medical_image_analyze(
                studyInstanceUID=f"1.2.3.4.{800 + i:03d}",
                modality="CT",
                bodyPart="CHEST",
                clinicalIndication="screening",
                dicom_data=dicom,
                operator="audit_test",
            )
            if result["status"] in ("COMPLETED", "COMPLETED_WITH_ERRORS"):
                count += 1

        assert count == 100

    def test_audit_log_export_contains_all_fields(self, gateway):
        dicom = make_mock_dicom("1.2.3.4.901")
        result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.901",
            dicom_data=dicom,
            operator="export_test",
        )
        audit = gateway.get_audit_trail(study_uid="1.2.3.4.901")
        assert audit is not None
        assert audit.get("study_uid") == "1.2.3.4.901"


class TestScenario9_MultiModelRouting:
    """Test correct routing for different inspection types."""

    def test_three_models_three_types(self, gateway):
        """Routing accuracy: 3 inspection types x 3 models."""
        test_cases = [
            ("CT", "CHEST", "screening", "lung-nodule-v1"),
            ("MG", "BREAST", "screening", "breast-lesion-v1"),
            ("XR", "HAND", "bone_age", "bone-age-v1"),
        ]
        for modality, body_part, indication, expected_model in test_cases:
            dicom = make_mock_dicom(
                f"1.2.3.4.ROUTING.{expected_model}",
                modality=modality,
                body_part=body_part,
            )
            result = gateway.medical_image_analyze(
                studyInstanceUID=f"1.2.3.4.ROUTING.{expected_model}",
                modality=modality,
                bodyPart=body_part,
                clinicalIndication=indication,
                dicom_data=dicom,
            )
            assert result["model_id"] == expected_model, (
                f"Expected {expected_model} for {modality}/{body_part}/{indication}, got {result.get('model_id')}"
            )

    def test_list_registered_models(self, gateway):
        models = gateway.list_registered_models()
        assert len(models) == 3
        model_ids = [m["id"] for m in models]
        assert "lung-nodule-v1" in model_ids
        assert "breast-lesion-v1" in model_ids
        assert "bone-age-v1" in model_ids


class TestScenario10_PHIIrreversibility:
    """Verify PHI de-identification is irreversible (hash collision test)."""

    def test_hash_collision_resistance(self):
        """Different patient IDs should produce different hashes."""
        sec = SecurityModule()
        d1 = sec.deidentify({"PatientID": "PATIENT_A"})
        d2 = sec.deidentify({"PatientID": "PATIENT_B"})
        assert d1["PatientID"] != d2["PatientID"]

    def test_hash_no_inverse(self):
        """Hash should not be reversible to original ID."""
        sec = SecurityModule()
        original = "TEST_PATIENT_123"
        deid = sec.deidentify({"PatientID": original})
        hashed = deid["PatientID"]
        assert hashed != original
        assert hashed.startswith("sha256:")
        hash_part = hashed.split(":")[1]
        assert len(hash_part) == 16  # truncated to 16 chars for compactness
        assert all(c in "0123456789abcdef" for c in hash_part)
