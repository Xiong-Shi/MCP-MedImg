"""
MCP-Medimg Gateway: Orchestrates all four layers as a unified MCP Server.

Workflow: L1 (Retrieve) -> L2 (De-identify + Audit) -> L4 (Inspection + Verification mode)
          -> L3 (Route + Infer + Format) -> L4 (Encode + Apply verification)
          -> L2 (Audit Complete) -> L1 (Store Result)

Note: L4 verification mode is determined before L3 routing so that even
when no matching model exists, the verification workflow is still applied.
This supports STAT priority auto-verification regardless of model availability.
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Optional

from mcp_medimg.transport.pacs_client import PACSClient
from mcp_medimg.security import (
    SecurityModule,
    AuditLogger,
    PolicyEngine,
    InputSanitizer,
)
from mcp_medimg.orchestration import (
    ModelRegistry,
    AIRouter,
    VersionManager,
    ResultFormatter,
    MockNNUNetEngine,
)
from mcp_medimg.application import (
    InspectionRouter,
    LesionEncoder,
    VerificationRouter,
    VerificationStateMachine,
    HistoricalComparator,
)
from mcp_medimg.dummy_layers import (
    DummySecurityModule,
    DummyAuditLogger,
    DummyPolicyEngine,
    DummyAIRouter,
    DummyResultFormatter,
    DummyInspectionRouter,
    DummyLesionEncoder,
    DummyVerificationRouter,
)

logger = logging.getLogger(__name__)


class MCPMedimgGateway:
    """Unified gateway that orchestrates L1-L4 in sequence."""

    def __init__(
        self,
        pacs_client: Optional[PACSClient] = None,
        security_module: Optional[SecurityModule] = None,
        audit_logger: Optional[AuditLogger] = None,
        policy_engine: Optional[PolicyEngine] = None,
        model_registry: Optional[ModelRegistry] = None,
        version_manager: Optional[VersionManager] = None,
        result_formatter: Optional[ResultFormatter] = None,
        inspection_router: Optional[InspectionRouter] = None,
        lesion_encoder: Optional[LesionEncoder] = None,
        verification_router: Optional[VerificationRouter] = None,
        config: Optional[dict] = None,
        enable_security: bool = True,
        enable_orchestration: bool = True,
        enable_semantics: bool = True,
        enable_ai: bool = True,
    ):
        """Unified gateway that orchestrates L1-L4 in sequence.

        Ablation parameters (enable_security/orchestration/semantics/ai)
        support layer-by-layer latency measurement for Table 6.
        When a layer is disabled, its process step becomes a no-op pass-through.
        """
        self.config = config or {}
        self._layer_flags = {
            "security": enable_security,
            "orchestration": enable_orchestration,
            "semantics": enable_semantics,
            "ai": enable_ai,
        }
        pacs_url = self.config.get("pacs_api_url", "http://localhost:8042")

        # L1: Transport (always enabled - baseline)
        self.pacs = pacs_client or PACSClient(base_url=pacs_url)

        # L2: Security
        if enable_security:
            self.security = security_module or SecurityModule()
            self.audit = audit_logger or AuditLogger(
                db_path=self.config.get("audit_db", "audit.db")
            )
            self.policy = policy_engine or PolicyEngine()
            self.sanitizer = InputSanitizer()
        else:
            # Dummy pass-through modules for ablation
            self.security = DummySecurityModule()
            self.audit = DummyAuditLogger()
            self.policy = DummyPolicyEngine()
            self.sanitizer = InputSanitizer()

        # L3: Orchestration
        if enable_orchestration:
            self.registry = model_registry or self._build_default_registry()
            self.router = AIRouter(self.registry)
            self.version_mgr = version_manager or VersionManager()
            self.formatter = result_formatter or ResultFormatter()
        else:
            self.registry = None
            self.router = DummyAIRouter()
            self.version_mgr = None
            self.formatter = DummyResultFormatter()

        # L4: Application Semantics
        if enable_semantics:
            self.inspection = inspection_router or InspectionRouter()
            self.lesion_encoder = lesion_encoder or LesionEncoder()
            self.verification = verification_router or VerificationRouter(
                policy_engine=self.policy if enable_security else PolicyEngine()
            )
        else:
            self.inspection = DummyInspectionRouter()
            self.lesion_encoder = DummyLesionEncoder()
            self.verification = DummyVerificationRouter()

        self._verification_states: dict = {}

    def _build_default_registry(self) -> ModelRegistry:
        registry = ModelRegistry()

        lung_engine = MockNNUNetEngine(modality="CT", task="lung_nodule")
        registry.register(
            "lung-nodule-v1",
            {
                "modality": ["CT"],
                "bodyPart": ["CHEST"],
                "indications": ["screening", "followup"],
                "performance": {"AUC": 0.92, "Sensitivity": 0.88},
                "regulatory": ["FDA_510k", "NMPA_Grade2"],
                "version": "v1.0",
                "inference_time_ms": 200,
                "engine": lung_engine,
                "output_format": "DICOM_SR",
            },
        )

        breast_engine = MockNNUNetEngine(modality="MG", task="breast_lesion")
        registry.register(
            "breast-lesion-v1",
            {
                "modality": ["MG"],
                "bodyPart": ["BREAST"],
                "indications": ["screening"],
                "performance": {"AUC": 0.89},
                "regulatory": ["CE"],
                "version": "v1.0",
                "inference_time_ms": 300,
                "engine": breast_engine,
                "output_format": "DICOM_SR",
            },
        )

        bone_engine = MockNNUNetEngine(modality="XR", task="bone_age")
        registry.register(
            "bone-age-v1",
            {
                "modality": ["XR"],
                "bodyPart": ["HAND"],
                "indications": ["bone_age"],
                "performance": {"AUC": 0.91},
                "regulatory": ["FDA_510k"],
                "version": "v1.0",
                "inference_time_ms": 150,
                "engine": bone_engine,
                "output_format": "DICOM_SR",
            },
        )

        return registry

    def medical_image_analyze(
        self,
        studyInstanceUID: str,
        seriesInstanceUID: str = "",
        modality: str = "CT",
        bodyPart: str = "CHEST",
        clinicalIndication: str = "screening",
        priority: str = "ROUTINE",
        aiModels: Optional[list] = None,
        outputFormat: str = "DICOM_SR",
        verificationMode: str = "ADVISORY",
        operator: str = "system",
        dicom_data: Optional[dict] = None,
    ) -> dict:
        """Analyze a DICOM study with AI-assisted diagnosis."""
        result = {
            "studyInstanceUID": studyInstanceUID,
            "status": "PROCESSING",
            "errors": [],
            "layers_executed": [],
        }

        # === L1: Retrieve study data ===
        try:
            if dicom_data is None:
                studies = self.pacs.query_studies(modality)
                if studies:
                    dicom_data = studies[0]
                else:
                    dicom_data = {
                        "StudyInstanceUID": studyInstanceUID,
                        "SeriesInstanceUID": seriesInstanceUID,
                        "Modality": modality,
                        "BodyPartExamined": bodyPart,
                    }
            result["layers_executed"].append("L1_retrieve")
        except Exception as e:
            result["errors"].append({"layer": "L1", "error": str(e)})
            result["status"] = "FAILED"
            return result

        # === L2: De-identify + audit start ===
        try:
            deidentified = self.security.deidentify(dicom_data)
            audit_id = self.audit.log_start(
                {
                    "operator": operator,
                    "patient_id": dicom_data.get("PatientID", "unknown"),
                    "study_uid": studyInstanceUID,
                    "dicom_data": dicom_data,
                }
            )
            result["audit_id"] = audit_id
            result["patient_hash"] = deidentified.get("PatientID", "unknown")
            result["layers_executed"].append("L2_deidentify_audit_start")
        except Exception as e:
            result["errors"].append({"layer": "L2", "error": str(e)})
            result["status"] = "FAILED"
            return result

        # === L4: Determine inspection type ===
        inspection_type = self.inspection.route(modality, bodyPart, clinicalIndication)
        result["inspection_type"] = inspection_type

        # === L4: Determine verification mode (BEFORE L3, always set) ===
        try:
            ver_result = self.verification.determine_mode(
                inspection_type,
                priority,
                model_id=aiModels[0] if aiModels else None,
            )
            result["verification_mode"] = ver_result["mode"]
            result["verification_reason"] = ver_result["reason"]
            result["layers_executed"].append("L4_verification_mode")
        except Exception as e:
            result["errors"].append({"layer": "L4_verify_mode", "error": str(e)})
            ver_result = {"mode": "ADVISORY", "reason": "Default"}

        # === L3: Route to AI model ===
        formatted = {"findings": []}
        ai_findings = []
        model = None
        model_id = None

        if not self._layer_flags["orchestration"]:
            # Orchestration disabled - skip AI routing entirely
            result["layers_executed"].append("L3_skipped")
        else:
            try:
                if aiModels:
                    model_id = aiModels[0]
                    model = self.registry.get_model(model_id)
                else:
                    model = self.router.route(
                        modality, bodyPart, clinicalIndication, priority
                    )
                    model_id = model["id"] if model else None

                if not model_id or not model:
                    result["errors"].append(
                        {
                            "layer": "L3",
                            "error": f"No suitable model for {modality}/{bodyPart}/{clinicalIndication}",
                        }
                    )
                    # Continue to L4 - do NOT return early
                else:
                    result["model_id"] = model_id
                    result["model_version"] = model["metadata"].get(
                        "version", "unknown"
                    )

                    # L3: Run inference
                    study_data = {
                        "study_uid": studyInstanceUID,
                        "modality": modality,
                        "body_part": bodyPart,
                        "pixel_data": dicom_data.get("pixel_data"),
                        "patient_age": dicom_data.get("PatientAge"),
                    }
                    infer_result = self.router.infer(model_id, study_data)

                    if infer_result["error"]:
                        result["errors"].append(
                            {
                                "layer": "L3_inference",
                                "error": infer_result["error"]["message"],
                            }
                        )
                    else:
                        ai_findings = infer_result["result"].get("findings", [])
                        result["layers_executed"].append("L3_route_infer")

                        # L3: Format results
                        study_info = {
                            "study_uid": studyInstanceUID,
                            "patient_hash": result["patient_hash"],
                        }
                        model_info = {
                            "name": model_id,
                            "version": result["model_version"],
                            "verified": True,
                        }

                        if outputFormat == "DICOM_SR":
                            formatted = self.formatter.to_dicom_sr(
                                ai_findings, study_info, model_info
                            )
                        elif outputFormat == "FHIR":
                            formatted = self.formatter.to_fhir_observation(
                                ai_findings, study_info
                            )
                        else:
                            formatted = {"findings": ai_findings}

                        result["formatted_output"] = formatted
                        result["raw_findings"] = ai_findings
                        result["layers_executed"].append("L3_format")

            except Exception as e:
                result["errors"].append({"layer": "L3", "error": str(e)})

        # === L4: Encode lesions with RADS ===
        try:
            if inspection_type in ("LUNG_CANCER_SCREENING", "NODULE_FOLLOWUP"):
                encoded = self.lesion_encoder.encode_rads_from_findings(ai_findings)
            elif inspection_type == "BREAST_SCREENING":
                encoded = self.lesion_encoder.encode_birads_from_findings(ai_findings)
            else:
                encoded = ai_findings

            result["encoded_findings"] = encoded
            result["layers_executed"].append("L4_lesion_encoding")
        except Exception as e:
            result["errors"].append({"layer": "L4_encode", "error": str(e)})

        # === L4: Apply verification workflow ===
        try:
            vsm = VerificationStateMachine(
                audit_id,
                timeout_seconds=self.verification.get_workflow(ver_result["mode"])[
                    "timeout_seconds"
                ],
            )
            self._verification_states[audit_id] = vsm

            workflow = self.verification.get_workflow(ver_result["mode"])
            result["verification_workflow"] = workflow

            if ver_result["mode"] == "AUTO":
                vsm.transition_to("UNDER_REVIEW")
                vsm.transition_to("REVIEWED")
                vsm.transition_to("CONFIRMED")
                vsm.transition_to("FINALIZED")
                result["verification_action"] = "AUTO_FINALIZED"
            elif ver_result["mode"] == "MANDATORY":
                vsm.transition_to("UNDER_REVIEW")
                result["verification_action"] = "PENDING_RADIOLOGIST_REVIEW"
            else:  # ADVISORY
                vsm.transition_to("UNDER_REVIEW")
                result["verification_action"] = "ADVISORY_REVIEW_OPTIONAL"

            result["verification_state"] = vsm.state
            result["verification_history"] = vsm.get_history()
            result["layers_executed"].append("L4_verification")

        except Exception as e:
            result["errors"].append({"layer": "L4_verify", "error": str(e)})

        # === L2: Audit complete ===
        try:
            self.audit.log_complete(
                audit_id,
                {
                    "model_version": result.get("model_version"),
                    "output": formatted,
                },
                {
                    "mode": result.get("verification_mode", "N/A"),
                    "action": result.get("verification_action", "N/A"),
                    "reason": result.get("verification_reason", ""),
                    "finalized_by": operator
                    if result.get("verification_action") == "AUTO_FINALIZED"
                    else None,
                },
                study_uid=studyInstanceUID,
            )
            result["layers_executed"].append("L2_audit_complete")
        except Exception as e:
            result["errors"].append({"layer": "L2_audit", "error": str(e)})

        # === L1: Store result (mock) ===
        result["store_status"] = "stored_mock"
        result["layers_executed"].append("L1_store")

        if result["errors"]:
            result["status"] = "COMPLETED_WITH_ERRORS"
        else:
            result["status"] = "COMPLETED"

        result["completed_at"] = datetime.now().isoformat()
        return result

    def get_audit_trail(
        self, audit_id: str = None, study_uid: str = None
    ) -> Optional[dict]:
        """Retrieve the complete audit trail for a given analysis.
        Prefers study_uid for lookup; falls back to audit_id."""
        if study_uid:
            return self.audit.get_latest_log(study_uid)
        # If only audit_id is provided, get latest log (fallback)
        return self.audit.get_latest_log(audit_id)

    def list_registered_models(self) -> list:
        return [
            {
                "id": mid,
                "version": m.get("version"),
                "modality": m.get("modality"),
                "bodyPart": m.get("bodyPart"),
            }
            for mid, m in self.registry.models.items()
        ]

    def compare_studies(
        self,
        current_findings: list,
        prior_findings: list,
        current_study_date: str,
        prior_study_date: str,
    ) -> dict:
        """Compare findings between two studies using HistoricalComparator.

        Supports longitudinal analysis with VDT calculation (Schwartz formula)
        and follow-up recommendations based on Lung-RADS criteria.

        Args:
            current_findings: Findings from current study
            prior_findings: Findings from prior study
            current_study_date: Current study date (YYYYMMDD)
            prior_study_date: Prior study date (YYYYMMDD)

        Returns:
            Comparison result with matched nodules, VDT, and recommendations
        """
        comparator = HistoricalComparator()
        return comparator.compare_studies(
            current_findings=current_findings,
            prior_findings=prior_findings,
            current_study_date=current_study_date,
            prior_study_date=prior_study_date,
        )
