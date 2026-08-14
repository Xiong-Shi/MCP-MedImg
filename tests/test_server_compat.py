"""
Tests for MCP Server, dicom-mcp compatibility, stateless deployment,
and historical comparison (Table 5 remaining scenarios).
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from mcp_medimg.gateway import MCPMedimgGateway
from mcp_medimg.security import SecurityModule, AuditLogger
from mcp_medimg.transport.pacs_client import PACSClient
from mcp_medimg.compat import DicomMCPCompatibleClient
from mcp_medimg.stateless_verifier import StatelessVerifier
from mcp_medimg.application import HistoricalComparator


@pytest.fixture
def gateway():
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
    patient_age="45",
    pixel_data=None,
):
    if pixel_data is None:
        pixel_data = np.zeros((512, 512), dtype=np.int16)
    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": f"{study_uid}.1",
        "Modality": modality,
        "BodyPartExamined": body_part,
        "PatientName": patient_name,
        "PatientID": patient_id,
        "PatientAge": patient_age,
        "pixel_data": pixel_data,
    }


# ============================================================================
# MCP Server Tests
# ============================================================================


class TestMCPServer:
    """Test MCP Server Tools, Resources, and Prompts."""

    def test_server_importable(self):
        from mcp_medimg import server

        assert server is not None

    def test_server_has_stdio_transport(self):
        from mcp_medimg import server

        assert hasattr(server, "run_stdio")

    def test_server_has_http_transport(self):
        from mcp_medimg import server

        assert hasattr(server, "run_http")

    def test_medical_image_analyze_tool_registered(self):
        from mcp_medimg import server

        # FastMCP stores tools internally; verify the function exists
        assert hasattr(server, "medical_image_analyze")
        assert hasattr(server, "query_audit_trail")
        assert hasattr(server, "list_models")

    def test_tool_returns_json_string(self, gateway, monkeypatch):
        from mcp_medimg import server

        monkeypatch.setattr(server, "_gateway", gateway)

        result = server.medical_image_analyze(
            studyInstanceUID="1.2.3.4.SERVER.1",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["status"] in ("COMPLETED", "COMPLETED_WITH_ERRORS")
        assert parsed["inspection_type"] == "LUNG_CANCER_SCREENING"

    def test_list_models_tool(self, gateway, monkeypatch):
        from mcp_medimg import server

        monkeypatch.setattr(server, "_gateway", gateway)

        result = server.list_models()
        parsed = json.loads(result)
        assert len(parsed) == 3
        assert "lung-nodule-v1" in [m["id"] for m in parsed]

    def test_query_audit_trail_tool(self, gateway, monkeypatch):
        from mcp_medimg import server

        monkeypatch.setattr(server, "_gateway", gateway)

        # First run an analysis
        dicom = make_mock_dicom("1.2.3.4.SERVER.AUDIT")
        gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.SERVER.AUDIT",
            dicom_data=dicom,
        )

        result = server.query_audit_trail(study_uid="1.2.3.4.SERVER.AUDIT")
        parsed = json.loads(result)
        assert parsed is not None

    def test_lung_screening_prompt(self):
        from mcp_medimg import server

        prompt = server.lung_screening_protocol(
            studyInstanceUID="1.2.3.4.PROMPT",
            operator="dr_zhang",
        )
        assert "1.2.3.4.PROMPT" in prompt
        assert "dr_zhang" in prompt
        assert "lung" in prompt.lower() or "Lung" in prompt
        assert "MANDATORY" in prompt

    def test_breast_screening_prompt(self):
        from mcp_medimg import server

        prompt = server.breast_screening_protocol(
            studyInstanceUID="1.2.3.4.BREAST",
            operator="dr_li",
        )
        assert "1.2.3.4.BREAST" in prompt
        assert "BI-RADS" in prompt

    def test_protocol_consistency(self):
        """Verify Tool/Resource/Prompt definitions comply with MCP specifications."""
        from mcp_medimg import server

        # Tools are decorated with @mcp.tool()
        assert callable(server.medical_image_analyze)
        assert callable(server.query_audit_trail)
        assert callable(server.list_models)
        # Prompts are decorated with @mcp.prompt()
        assert callable(server.lung_screening_protocol)
        assert callable(server.breast_screening_protocol)


# ============================================================================
# dicom-mcp Compatibility Tests
# ============================================================================


class TestDicomMCPCompatibility:
    """Test Table 5 scenario: dicom-mcp compatibility."""

    def test_wrapper_initializes_native(self):
        """Wrapper should initialize with native DICOMweb when dicom-mcp not available."""
        client = DicomMCPCompatibleClient(
            pacs_url="http://localhost:8042",
            use_dicom_mcp=False,
        )
        assert client.using_dicom_mcp is False
        assert client._native_client is not None

    def test_wrapper_falls_back_gracefully(self):
        """When dicom-mcp requested but not installed, should fall back to native."""
        client = DicomMCPCompatibleClient(
            pacs_url="http://localhost:8042",
            use_dicom_mcp=True,  # Request dicom-mcp
        )
        # dicom-mcp is not installed, should fall back
        assert client.using_dicom_mcp is False
        assert client._native_client is not None

    def test_wrapper_query_with_security(self):
        """Wrapped query should apply PHI de-identification and audit logging."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            client = DicomMCPCompatibleClient(
                pacs_url="http://localhost:8042",
                use_dicom_mcp=False,
                audit_logger=AuditLogger(db_path=db_path),
            )
            result = client.query(
                study_uid="1.2.3.4.DICOMMCP.1",
                modality="CT",
                operator="dr_smith",
            )
            assert result["security_applied"] is True
            assert result["deidentified"] is True
            assert result["audit_id"] is not None
            assert "dr_smith" not in json.dumps(
                result
            )  # operator name in audit, not in results
        finally:
            os.unlink(db_path)

    def test_wrapper_authorization_enforced(self):
        """Unauthorized operators should be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            client = DicomMCPCompatibleClient(
                pacs_url="http://localhost:8042",
                use_dicom_mcp=False,
                audit_logger=AuditLogger(db_path=db_path),
            )
            with pytest.raises(PermissionError):
                client.query(
                    study_uid="1.2.3.4.DICOMMCP.2",
                    operator="unauthorized_user",
                )
        finally:
            os.unlink(db_path)

    def test_wrapper_store_with_deidentification(self):
        """Store should de-identify DICOM data before storage."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            client = DicomMCPCompatibleClient(
                pacs_url="http://localhost:8042",
                audit_logger=AuditLogger(db_path=db_path),
            )
            result = client.store(
                dicom_object={
                    "StudyInstanceUID": "1.2.3.4.STORE.1",
                    "PatientID": "REAL_PATIENT_ID",
                    "PatientName": "ZHANG^SAN",
                },
                operator="dr_smith",
            )
            assert result["status"] == "STORED"
            assert result["deidentified"] is True
            assert result["audit_id"] is not None
        finally:
            os.unlink(db_path)

    def test_comparison_native_vs_wrapped(self):
        """Demonstrate the value of MCP-Medimg wrapping over raw dicom-mcp.

        Paper: 'When using native dicom-mcp, the output is a raw DICOM data
        structure without security or audit. Wrapped by MCP-Medimg, the same
        data access is augmented with operator authorization, PHI hashing,
        and complete audit logging.'
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # Native (no security)
            native = PACSClient(base_url="http://localhost:8042")
            # Wrapped (with security)
            wrapped = DicomMCPCompatibleClient(
                pacs_url="http://localhost:8042",
                audit_logger=AuditLogger(db_path=db_path),
            )

            # Native has no security attributes
            assert not hasattr(native, "security")
            assert not hasattr(native, "audit")

            # Wrapped has security and audit
            assert hasattr(wrapped, "security")
            assert hasattr(wrapped, "audit")
            assert wrapped.security is not None
            assert wrapped.audit is not None
        finally:
            os.unlink(db_path)


# ============================================================================
# Stateless Deployment Tests
# ============================================================================


class TestStatelessDeployment:
    """Test Table 6: Stateless deployment compliance."""

    def test_stateless_verification_all_checks(self, gateway):
        """Run all stateless compliance checks."""
        verifier = StatelessVerifier(gateway)
        result = verifier.verify_all()
        assert result["overall"] in ("PASS", "PARTIAL")
        assert result["passed"] > 0
        for check_name, check_result in result["checks"].items():
            assert "passed" in check_result
            assert "reason" in check_result

    def test_no_session_state(self, gateway):
        """Verify no persistent session state between calls."""
        verifier = StatelessVerifier(gateway)
        check = verifier._check_no_session_state()
        assert check["passed"] is True

    def test_client_session_id(self, gateway):
        """Verify client-provided session IDs are supported."""
        verifier = StatelessVerifier(gateway)
        check = verifier._check_client_session_id()
        assert check["passed"] is True

    def test_idempotent_tools(self, gateway):
        """Verify identical inputs produce identical clinical decisions."""
        verifier = StatelessVerifier(gateway)
        check = verifier._check_idempotent_tools()
        assert check["passed"] is True

    def test_no_sticky_routing(self, gateway):
        """Verify no sticky routing requirement."""
        verifier = StatelessVerifier(gateway)
        check = verifier._check_no_sticky_routing()
        assert check["passed"] is True

    def test_horizontal_scaling(self, gateway):
        """Verify horizontal scaling compatibility."""
        verifier = StatelessVerifier(gateway)
        check = verifier._check_horizontal_scaling()
        assert check["passed"] is True

    def test_transport_compliance(self, gateway):
        """Verify MCP transport protocol compliance."""
        verifier = StatelessVerifier(gateway)
        check = verifier._check_transport_compliance()
        assert check["passed"] is True


# ============================================================================
# Historical Comparison Tests
# ============================================================================


class TestHistoricalComparison:
    """Test Table 5 scenario: Historical comparison + VDT."""

    def setup_method(self):
        self.comparator = HistoricalComparator()

    def test_compare_with_growth(self):
        """Compare two studies where nodule has grown."""
        prior = [
            {
                "type": "lung_nodule",
                "size_mm": 6.0,
                "rads_category": "Lung-RADS 2",
                "nodule_type": "solid",
                "location": {"x": 200, "y": 200, "z": 5, "slice": 5},
            }
        ]
        current = [
            {
                "type": "lung_nodule",
                "size_mm": 10.0,
                "rads_category": "Lung-RADS 4A",
                "nodule_type": "solid",
                "location": {"x": 202, "y": 198, "z": 5, "slice": 5},
            }
        ]

        result = self.comparator.compare_studies(current, prior, "20260701", "20250101")

        assert result["days_between_studies"] == 546
        assert len(result["matched_findings"]) == 1
        match = result["matched_findings"][0]
        assert match["size_change_mm"] == 4.0
        assert match["size_change_percent"] == pytest.approx(66.7, rel=1)
        assert match["volume_doubling_time_days"] is not None
        assert match["volume_doubling_time_days"] > 0
        assert match["growth_assessment"] in (
            "RAPID_GROWTH",
            "MODERATE_GROWTH",
            "SLOW_GROWTH",
        )

    def test_compare_stable_nodule(self):
        """Compare two studies where nodule is stable."""
        prior = [
            {
                "type": "lung_nodule",
                "size_mm": 5.0,
                "rads_category": "Lung-RADS 2",
                "location": {"x": 100, "y": 100, "z": 3},
            }
        ]
        current = [
            {
                "type": "lung_nodule",
                "size_mm": 5.0,
                "rads_category": "Lung-RADS 2",
                "location": {"x": 100, "y": 100, "z": 3},
            }
        ]

        result = self.comparator.compare_studies(current, prior, "20260701", "20250701")

        match = result["matched_findings"][0]
        assert match["size_change_mm"] == 0.0
        assert match["growth_assessment"] == "STABLE"
        assert match["volume_doubling_time_days"] is None

    def test_new_nodule_detection(self):
        """Detect new nodules in current study not present in prior."""
        prior = []
        current = [
            {
                "type": "lung_nodule",
                "size_mm": 8.0,
                "rads_category": "Lung-RADS 2",
                "location": {"x": 150, "y": 150, "z": 4},
            }
        ]

        result = self.comparator.compare_studies(current, prior, "20260701", "20250701")

        assert len(result["new_findings"]) == 1
        assert len(result["matched_findings"]) == 0
        assert "1 new finding" in result["summary"]

    def test_vdt_calculation(self):
        """Test Volume Doubling Time calculation (Schwartz formula)."""
        # VDT = (t * ln2) / (3 * ln(d2/d1))
        # t=365 days, d1=6mm, d2=12mm
        # VDT = (365 * 0.693) / (3 * ln(2)) = 365 * 0.693 / (3 * 0.693) = 365/3 = 121.7
        vdt = self.comparator._calculate_vdt(6.0, 12.0, 365)
        assert vdt is not None
        assert vdt == pytest.approx(121.7, rel=0.1)

    def test_vdt_no_growth(self):
        """VDT should be None when no growth."""
        vdt = self.comparator._calculate_vdt(8.0, 8.0, 365)
        assert vdt is None

    def test_vdt_shrinkage(self):
        """VDT should be None when nodule shrinks."""
        vdt = self.comparator._calculate_vdt(10.0, 5.0, 365)
        assert vdt is None

    def test_followup_recommendation_rapid_growth(self):
        """Rapid growth should trigger short-interval follow-up."""
        rec = self.comparator._recommend_followup("Lung-RADS 3", 300, 3.0)
        assert "3 months" in rec
        assert "rapid" in rec.lower()

    def test_followup_recommendation_rads4b(self):
        """Lung-RADS 4B should trigger immediate tissue diagnosis."""
        rec = self.comparator._recommend_followup("Lung-RADS 4B", None, None)
        assert "tissue diagnosis" in rec.lower()

    def test_end_to_end_historical_comparison(self, gateway):
        """End-to-end: run two analyses and compare results."""
        # Run prior study
        dicom_prior = make_mock_dicom(
            "1.2.3.4.PRIOR", patient_id="HIST_PT001", patient_age="44"
        )
        prior_result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.PRIOR",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
            dicom_data=dicom_prior,
        )

        # Run current study
        dicom_current = make_mock_dicom(
            "1.2.3.4.CURRENT", patient_id="HIST_PT001", patient_age="45"
        )
        current_result = gateway.medical_image_analyze(
            studyInstanceUID="1.2.3.4.CURRENT",
            modality="CT",
            bodyPart="CHEST",
            clinicalIndication="screening",
            dicom_data=dicom_current,
        )

        # Compare
        comparison = self.comparator.compare_studies(
            current_findings=current_result.get("raw_findings", []),
            prior_findings=prior_result.get("raw_findings", []),
            current_study_date="20260701",
            prior_study_date="20250701",
        )

        assert comparison["days_between_studies"] == 365
        assert "summary" in comparison
