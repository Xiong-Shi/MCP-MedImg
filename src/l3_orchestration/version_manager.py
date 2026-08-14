"""Version manager with canary release and A/B testing support"""


class VersionManager:
    def __init__(self):
        self.models = {}  # model_name -> {version: metadata}
        self.deploy_times = {}  # model_id -> deployment timestamp

    def register_version(
        self, model_name: str, version: str, status: str = "stable", **kwargs
    ) -> None:
        """Register a model version"""
        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][version] = {"status": status, **kwargs}

    def canary_release(
        self, model_name: str, new_version: str, canary_percentage: int = 10
    ) -> None:
        """Release new version to a percentage of traffic"""
        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][new_version] = {
            "status": "canary",
            "canary_percentage": canary_percentage,
        }

    def get_version_for_request(self, model_name: str, patient_hash: str) -> str:
        """Determine version using consistent hashing"""
        versions = self.models.get(model_name, {})
        if not versions:
            return None
        if len(versions) == 1:
            return list(versions.keys())[0]

        # Consistent hashing for canary routing
        hash_val = int(patient_hash[:8], 16) % 100
        for ver, meta in versions.items():
            if meta.get("status") == "canary":
                canary_pct = meta.get("canary_percentage", 0)
                if hash_val < canary_pct:
                    return ver
        # Default to stable
        for ver, meta in versions.items():
            if meta.get("status") == "stable":
                return ver
        return list(versions.keys())[0]

    def is_new_deployment(
        self, model_name: str, version: str, warmup_cases: int = 50
    ) -> bool:
        """Check if model is in warmup period (first N cases)"""
        key = f"{model_name}:{version}"
        count = self.deploy_times.get(key, {}).get("case_count", 0)
        return count < warmup_cases

    def record_case(self, model_name: str, version: str) -> None:
        """Record that a case was processed by this model version"""
        key = f"{model_name}:{version}"
        if key not in self.deploy_times:
            self.deploy_times[key] = {"case_count": 0}
        self.deploy_times[key]["case_count"] += 1
