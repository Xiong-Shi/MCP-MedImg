"""Policy engine for RBAC and verification rule enforcement"""

import json
from pathlib import Path


class PolicyEngine:
    def __init__(self, policy_file: str = None):
        if policy_file is None:
            policy_file = str(
                Path(__file__).parent.parent.parent.parent / "configs" / "policy.json"
            )
        with open(policy_file) as f:
            self.policy = json.load(f)

    def authorize(self, operator: str, operation: str, resource: dict = None) -> bool:
        roles = self._get_user_roles(operator)
        for role in roles:
            if operation in self.policy.get("permissions", {}).get(role, []):
                return True
        return False

    def get_verification_mode(
        self, modality: str, body_part: str, indication: str
    ) -> str:
        for rule in self.policy.get("verification_rules", []):
            if (
                rule.get("modality") == modality
                and rule.get("body_part") == body_part
                and indication in rule.get("indication", "")
            ):
                return rule["mode"]
        return "ADVISORY"

    def get_inspection_type_mode(self, inspection_type: str) -> str:
        """Look up verification mode by inspection type directly"""
        for rule in self.policy.get("verification_rules", []):
            if rule.get("inspection_type") == inspection_type:
                return rule["mode"]
        return "ADVISORY"

    def _get_user_roles(self, operator: str) -> list:
        return self.policy.get("users", {}).get(operator, ["guest"])
