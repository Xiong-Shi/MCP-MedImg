"""
Dummy (no-op) layer implementations for ablation experiments.

When a layer is disabled via MCPMedimgGateway's enable_* flags,
these classes provide pass-through behavior so the pipeline still
runs end-to-end, allowing per-layer latency measurement (Table 6).
"""

import hashlib
from datetime import datetime


class DummySecurityModule:
    """Pass-through de-identification - returns data unchanged."""

    def deidentify(self, dicom_data: dict) -> dict:
        # Minimal: hash PatientID so downstream still gets a non-empty patient_hash
        pid = dicom_data.get("PatientID", "unknown")
        if pid != "unknown":
            hashed = hashlib.sha256(str(pid).encode()).hexdigest()[:16]
            data = dict(dicom_data)
            data["PatientID"] = hashed
            return data
        return dict(dicom_data)


class DummyAuditLogger:
    """No-op audit logger - generates IDs but stores nothing."""

    def __init__(self, db_path: str = ""):
        self.db_path = db_path

    def log_start(self, request: dict) -> str:
        token = hashlib.sha256(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        return f"AUDIT_DUMMY_{datetime.now().strftime('%Y%m%d%H%M%S')}_{token}"

    def log_complete(
        self, audit_id: str, ai_result: dict, verification: dict, study_uid: str = ""
    ) -> None:
        pass

    def log_operation(self, **kwargs) -> str:
        return "AUDIT_DUMMY_OP"

    def get_latest_log(self, audit_id: str = None) -> dict:
        return {"audit_id": audit_id or "DUMMY", "note": "audit layer disabled"}

    def get_audit_count(self) -> int:
        return 0


class DummyPolicyEngine:
    """No-op policy engine - always returns ADVISORY mode."""

    def determine_verification_mode(
        self, inspection_type: str, priority: str, model_id: str = None
    ) -> dict:
        return {"mode": "ADVISORY", "reason": "Policy layer disabled"}

    def get_workflow(self, mode: str) -> dict:
        return {
            "timeout_seconds": 300,
            "steps": ["UNDER_REVIEW", "REVIEWED", "FINALIZED"],
        }


class DummyAIRouter:
    """Pass-through AI router - no model selection."""

    def route(self, modality, bodyPart, clinicalIndication, priority):
        return None

    def infer(self, model_id, study_data):
        return {
            "result": {"findings": [], "segmentation_mask": None},
            "error": None,
        }


class DummyResultFormatter:
    """Pass-through formatter - returns findings as-is."""

    def to_dicom_sr(self, findings, study_info, model_info):
        return {"findings": findings, "format": "raw_passthrough"}

    def to_fhir_observation(self, findings, study_info):
        return {"findings": findings, "format": "raw_passthrough"}


class DummyInspectionRouter:
    """Pass-through inspection router - returns GENERIC."""

    def route(self, modality, bodyPart, clinicalIndication):
        return "GENERIC"


class DummyLesionEncoder:
    """Pass-through lesion encoder - returns findings unchanged."""

    def encode_rads_from_findings(self, findings):
        return findings

    def encode_birads_from_findings(self, findings):
        return findings


class DummyVerificationRouter:
    """Pass-through verification - always ADVISORY."""

    def determine_mode(self, inspection_type, priority, model_id=None):
        return {"mode": "ADVISORY", "reason": "Verification layer disabled"}

    def get_workflow(self, mode):
        return {
            "timeout_seconds": 300,
            "steps": ["UNDER_REVIEW", "REVIEWED", "FINALIZED"],
        }
