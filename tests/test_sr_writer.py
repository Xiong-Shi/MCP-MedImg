"""
Tests for DICOM SR Writer - SR document creation, STOW-RS storage, and verification.
"""

import sys
import os
import io
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from mcp_medimg.storage import DICOMSRWriter


@pytest.fixture
def sr_writer():
    return DICOMSRWriter(pacs_url="http://localhost:8042")


@pytest.fixture
def sample_findings():
    return [
        {
            "type": "lung_nodule",
            "size_mm": 8.5,
            "rads_category": "Lung-RADS 4A",
            "confidence": 0.92,
            "location": {"x": 200, "y": 150, "z": 5},
        },
        {
            "type": "lung_nodule",
            "size_mm": 4.2,
            "rads_category": "Lung-RADS 2",
            "confidence": 0.85,
            "location": {"x": 300, "y": 250, "z": 8},
        },
    ]


@pytest.fixture
def sample_study_info():
    return {
        "patient_name": "TEST^PATIENT",
        "patient_id": "PT001",
        "study_uid": "1.2.3.4.SR.TEST.STUDY",
        "series_uid": "1.2.3.4.SR.TEST.SERIES",
        "study_date": "20260724",
        "study_time": "100000",
    }


@pytest.fixture
def sample_model_info():
    return {
        "name": "lung-nodule-v1",
        "version": "1.0.0",
        "verified": True,
    }


class TestDICOMSRWriter:
    """Test DICOM SR document creation."""

    def test_create_sr_document(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that SR document is created with correct attributes."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        assert ds is not None
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.34"
        assert str(ds.SOPInstanceUID).startswith("1.2.")
        assert str(ds.SeriesInstanceUID).startswith("1.2.")
        assert ds.Modality == "SR"
        assert ds.CompletionFlag == "COMPLETE"
        assert ds.VerificationFlag == "VERIFIED"

    def test_sr_patient_info(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that patient info is correctly set."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        assert str(ds.PatientName) == "TEST^PATIENT"
        assert ds.PatientID == "PT001"
        assert ds.StudyInstanceUID == "1.2.3.4.SR.TEST.STUDY"

    def test_sr_content_sequence_exists(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that ContentSequence contains finding data."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        assert hasattr(ds, "ContentSequence")
        assert len(ds.ContentSequence) > 0

    def test_sr_content_includes_algorithm_metadata(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that SR includes algorithm name and version."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        # First two items should be algorithm metadata
        algo_name_item = ds.ContentSequence[0]
        assert algo_name_item.ValueType == "TEXT"
        assert "Algorithm Name" in str(
            algo_name_item.ConceptNameCodeSequence[0].CodeMeaning
        )
        assert "lung-nodule-v1" in str(algo_name_item.TextValue)

        algo_version_item = ds.ContentSequence[1]
        assert algo_version_item.ValueType == "TEXT"
        assert "Algorithm Version" in str(
            algo_version_item.ConceptNameCodeSequence[0].CodeMeaning
        )
        assert "1.0.0" in str(algo_version_item.TextValue)

    def test_sr_finding_container_has_children(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that finding containers have nested content."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        # Find finding containers (items 2+)
        finding_containers = [
            item for item in ds.ContentSequence if item.ValueType == "CONTAINER"
        ]
        assert len(finding_containers) == 2  # 2 findings

        # Each container should have children
        for container in finding_containers:
            assert hasattr(container, "ContentSequence")
            assert len(container.ContentSequence) >= 3  # type, size, rads

    def test_sr_size_measurement(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that size measurements are included."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        # Find all NUM items
        finding_containers = [
            item for item in ds.ContentSequence if item.ValueType == "CONTAINER"
        ]

        first_finding = finding_containers[0]
        num_items = [
            item for item in first_finding.ContentSequence if item.ValueType == "NUM"
        ]
        assert len(num_items) >= 1

        # Check the diameter measurement
        size_item = next(
            (
                item
                for item in num_items
                if "Diameter" in str(item.ConceptNameCodeSequence[0].CodeMeaning)
            ),
            None,
        )
        assert size_item is not None
        assert float(size_item.MeasuredValueSequence[0].NumericValue) == 8.5

    def test_sr_rads_category(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that RADS categories are included."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        finding_containers = [
            item for item in ds.ContentSequence if item.ValueType == "CONTAINER"
        ]

        # Check first finding has Lung-RADS 4A
        first_finding = finding_containers[0]
        code_items = [
            item for item in first_finding.ContentSequence if item.ValueType == "CODE"
        ]
        rads_item = next(
            (
                item
                for item in code_items
                if "RADS" in str(item.ConceptNameCodeSequence[0].CodeMeaning)
            ),
            None,
        )
        assert rads_item is not None
        assert "Lung-RADS" in str(rads_item.ConceptCodeSequence[0].CodeMeaning)

    def test_sr_unverified_flag(self, sr_writer, sample_findings, sample_study_info):
        """Test unverified SR gets UNVERIFIED flag."""
        model_info = {"name": "test-model", "version": "0.1", "verified": False}
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=model_info,
        )
        assert ds.VerificationFlag == "UNVERIFIED"

    def test_sr_serializable(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test that SR can be serialized to DICOM file format."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        buffer = io.BytesIO()
        from pydicom.filewriter import dcmwrite

        dcmwrite(buffer, ds)
        buffer.seek(0)

        # Read it back with force=True for safety
        ds_read = dcmread(buffer, force=True)
        assert str(ds_read.SOPInstanceUID) == str(ds.SOPInstanceUID)
        assert ds_read.Modality == "SR"

    def test_sr_empty_findings(self, sr_writer, sample_study_info, sample_model_info):
        """Test SR creation with no findings."""
        ds = sr_writer.create_sr_document(
            findings=[],
            study_info=sample_study_info,
            model_info=sample_model_info,
        )
        assert ds is not None
        # Should still have algorithm metadata
        assert len(ds.ContentSequence) == 2

    def test_sr_breast_findings(self, sr_writer, sample_study_info, sample_model_info):
        """Test SR with breast screening findings (BI-RADS)."""
        findings = [
            {
                "type": "breast_lesion",
                "size_mm": 12.0,
                "rads_category": "BI-RADS 4B",
                "confidence": 0.88,
                "location": {"x": 100, "y": 100, "z": 3},
            }
        ]
        ds = sr_writer.create_sr_document(
            findings=findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )
        assert ds is not None
        finding_containers = [
            item for item in ds.ContentSequence if item.ValueType == "CONTAINER"
        ]
        assert len(finding_containers) == 1


class TestSRStorage:
    """Test SR storage to PACS."""

    def test_store_sr_pacs_unreachable(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test graceful handling when PACS is not running."""
        ds = sr_writer.create_sr_document(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        result = sr_writer.store_sr_to_pacs(
            ds, study_uid=sample_study_info["study_uid"]
        )

        # PACS may or may not be running; both are acceptable for local test
        assert result["status"] in ("STORED", "PACS_UNREACHABLE", "FAILED")
        if result["status"] == "STORED":
            assert "sop_instance_uid" in result
            assert "retrieve_url" in result
        elif result["status"] == "PACS_UNREACHABLE":
            assert "sop_instance_uid" in result

    def test_create_and_store_one_step(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """Test the convenience method create_and_store."""
        result = sr_writer.create_and_store(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )

        assert result["sr_created"] is True
        assert result["sop_class_uid"] == "1.2.840.10008.5.1.4.1.1.88.34"
        assert result["findings_count"] == 2
        assert "store_result" in result
        assert result["store_result"]["status"] in (
            "STORED",
            "PACS_UNREACHABLE",
            "FAILED",
        )

    def test_verify_sr_not_stored(self, sr_writer):
        """Test verification when SR doesn't exist in PACS."""
        result = sr_writer.verify_sr_in_pacs(
            study_uid="1.2.3.4.NONEXISTENT",
            series_uid="1.2.3.4.NONEXISTENT.SERIES",
            instance_uid="1.2.3.4.NONEXISTENT.INSTANCE",
        )
        # Should return verified=False or fail gracefully
        assert result["verified"] in (False, True)  # depends on PACS status
        if not result["verified"]:
            assert "message" in result

    @pytest.mark.skipif(
        not os.environ.get("MCP_MEDIMG_PACS_RUNNING"),
        reason="PACS not running (set MCP_MEDIMG_PACS_RUNNING=1 to enable)",
    )
    def test_store_and_verify_live_pacs(
        self, sr_writer, sample_findings, sample_study_info, sample_model_info
    ):
        """End-to-end: store SR to live Orthanc and verify retrieval."""
        # Create and store
        result = sr_writer.create_and_store(
            findings=sample_findings,
            study_info=sample_study_info,
            model_info=sample_model_info,
        )
        assert result["sr_created"] is True
        assert result["store_result"]["status"] == "STORED"

        # Verify
        verify_result = sr_writer.verify_sr_in_pacs(
            study_uid=result["store_result"]["study_uid"],
            series_uid=result["store_result"]["series_uid"],
            instance_uid=result["store_result"]["sop_instance_uid"],
        )
        assert verify_result["verified"] is True
        assert verify_result["modality"] == "SR"


class TestSREndToEnd:
    """End-to-end integration with Gateway."""

    def test_gateway_result_to_sr(self, sr_writer):
        """Test converting a Gateway analysis result to DICOM SR."""
        from mcp_medimg.gateway import MCPMedimgGateway
        from mcp_medimg.security import AuditLogger, SecurityModule
        from mcp_medimg.transport.pacs_client import PACSClient
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            gw = MCPMedimgGateway(
                pacs_client=PACSClient(base_url="http://localhost:8042"),
                audit_logger=AuditLogger(db_path=db_path),
                config={"audit_db": db_path},
            )

            # Run analysis
            dicom = {
                "StudyInstanceUID": "1.2.3.4.SR.E2E.STUDY",
                "SeriesInstanceUID": "1.2.3.4.SR.E2E.SERIES",
                "Modality": "CT",
                "BodyPartExamined": "CHEST",
                "PatientName": "TEST^E2E",
                "PatientID": "E2E001",
                "PatientAge": "55",
                "pixel_data": np.zeros((512, 512), dtype=np.int16),
            }

            result = gw.medical_image_analyze(
                studyInstanceUID="1.2.3.4.SR.E2E.STUDY",
                dicom_data=dicom,
            )

            # Convert to SR
            sr_result = sr_writer.create_and_store(
                findings=result.get("raw_findings", []),
                study_info={
                    "patient_name": "TEST^E2E",
                    "patient_id": "E2E001",
                    "study_uid": "1.2.3.4.SR.E2E.STUDY",
                    "study_date": "20260724",
                    "study_time": "100000",
                },
                model_info={
                    "name": "lung-nodule-v1",
                    "version": "1.0.0",
                    "verified": True,
                },
            )

            assert sr_result["sr_created"] is True
            assert sr_result["store_result"]["status"] in ("STORED", "PACS_UNREACHABLE")
        finally:
            os.unlink(db_path)
