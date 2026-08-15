"""Basic functionality tests for local 2060 SUPER"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from mcp_medimg.transport.pacs_client import PACSClient
from mcp_medimg.security.security_module import SecurityModule
from mcp_medimg.security.audit_logger import AuditLogger


def _pacs_available():
    try:
        return PACSClient().test_connection()
    except Exception:
        return False


class TestPACSClient:
    @pytest.mark.skipif(not _pacs_available(), reason="Orthanc not running")
    def test_connection(self):
        client = PACSClient()
        assert client.test_connection(), (
            "Orthanc not running! Start with: docker-compose up -d"
        )

    @pytest.mark.skipif(not _pacs_available(), reason="Orthanc not running")
    def test_query_studies(self):
        client = PACSClient()
        studies = client.query_studies()
        assert isinstance(studies, list)
        print(f"Found {len(studies)} studies in PACS")


class TestSecurityModule:
    def test_patient_name_removed(self):
        sec = SecurityModule(level="patient")
        data = {"PatientName": "John Doe", "PatientID": "P12345"}
        result = sec.deidentify(data)
        assert result["PatientName"] == "ANONYMIZED"

    def test_patient_id_hashed(self):
        sec = SecurityModule(level="patient")
        data = {"PatientID": "P12345"}
        result = sec.deidentify(data)
        assert result["PatientID"].startswith("sha256:")
        assert result["PatientID"] != "P12345"


class TestAuditLogger:
    def test_log_creation(self):
        logger = AuditLogger(db_path="reports/test_audit.db")
        audit_id = logger.log_operation(
            operation="test_analysis",
            operator="test_user",
            patient_hash="sha256:abc",
            study_uid="1.2.3",
            result="SUCCESS",
        )
        assert audit_id.startswith("AUDIT_")

    def test_audit_count(self):
        logger = AuditLogger(db_path="reports/test_audit.db")
        initial = logger.get_audit_count()
        logger.log_operation(operation="count_test")
        final = logger.get_audit_count()
        assert final > initial

    def test_regulatory_tags_storage(self):
        logger = AuditLogger(db_path="reports/test_audit.db")
        tags = ["FDA_TPLC", "NMPA_Grade2", "EU_AI_Act"]
        logger.log_operation(
            operation="regulatory_test",
            regulatory_tags=tags,
        )
        latest = logger.get_latest_log()
        assert latest["regulatory_tags"] == tags

    def test_regulatory_tags_default_empty(self):
        logger = AuditLogger(db_path="reports/test_audit.db")
        logger.log_operation(operation="no_tags_test")
        latest = logger.get_latest_log()
        assert latest["regulatory_tags"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
