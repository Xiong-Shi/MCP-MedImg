"""Verification router implementing conditional verification mechanism"""

from datetime import datetime


class VerificationRouter:
    def __init__(self, policy_engine=None):
        self.policy_engine = policy_engine
        self.deployment_tracker = {}  # model_id -> {case_count, deploy_time}

    def determine_mode(
        self,
        inspection_type: str,
        priority: str,
        model_version: str = None,
        model_id: str = None,
    ) -> dict:
        """Determine verification mode based on context

        Returns dict with 'mode' and 'reason' fields.
        Modes: MANDATORY, ADVISORY, AUTO
        """
        # STAT priority: AUTO for speed
        if priority == "STAT":
            return {"mode": "AUTO", "reason": "STAT priority auto-verifies for speed"}

        # New model deployment: MANDATORY for first N cases
        if model_id and self._is_new_model_deployment(model_id):
            return {
                "mode": "MANDATORY",
                "reason": f"New model deployment warmup period (first N cases)",
            }

        # Look up policy
        if self.policy_engine:
            mode = self.policy_engine.get_inspection_type_mode(inspection_type)
            if mode:
                return {
                    "mode": mode,
                    "reason": f"Institutional policy for {inspection_type}",
                }

        return {"mode": "ADVISORY", "reason": "Default advisory mode"}

    def _is_new_model_deployment(self, model_id: str, warmup_cases: int = 50) -> bool:
        """Check if model is in warmup period"""
        info = self.deployment_tracker.get(model_id)
        if info is None:
            return False
        return info.get("case_count", 0) < warmup_cases

    def register_deployment(self, model_id: str) -> None:
        """Register a new model deployment for tracking"""
        self.deployment_tracker[model_id] = {
            "case_count": 0,
            "deploy_time": datetime.now().isoformat(),
        }

    def record_case(self, model_id: str) -> None:
        """Record that a case was processed"""
        if model_id not in self.deployment_tracker:
            self.register_deployment(model_id)
        self.deployment_tracker[model_id]["case_count"] += 1

    def get_workflow(self, mode: str) -> dict:
        """Get verification workflow steps for a given mode"""
        workflows = {
            "MANDATORY": {
                "steps": [
                    "AI inference",
                    "Radiologist review (required)",
                    "Radiologist approval/rejection",
                    "Result finalization",
                ],
                "required_actions": ["radiologist_review", "radiologist_approval"],
                "timeout_seconds": 300,
            },
            "ADVISORY": {
                "steps": [
                    "AI inference",
                    "Radiologist review (optional)",
                    "Auto-finalize if no review within 24h",
                ],
                "required_actions": [],
                "timeout_seconds": 86400,
            },
            "AUTO": {
                "steps": ["AI inference", "Auto-finalization"],
                "required_actions": [],
                "timeout_seconds": 0,
            },
        }
        return workflows.get(mode, workflows["ADVISORY"])
