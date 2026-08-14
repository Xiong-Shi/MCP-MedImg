"""Model registry following KServe Open Inference Protocol pattern"""


class ModelRegistry:
    def __init__(self):
        self.models = {}  # model_id -> metadata

    def register(self, model_id: str, metadata: dict) -> None:
        """Register a new AI model with full metadata"""
        required_fields = [
            "modality",
            "bodyPart",
            "indications",
            "performance",
            "regulatory",
        ]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required field: {field}")

        # Generate unique resource URI
        metadata["uri"] = f"models://registry/{model_id}"
        self.models[model_id] = metadata

    def discover(
        self, modality: str = None, body_part: str = None, indication: str = None
    ) -> list:
        """Discover models matching criteria"""
        results = []
        for model_id, metadata in self.models.items():
            if modality and modality not in metadata.get("modality", []):
                continue
            if body_part and body_part not in metadata.get("bodyPart", []):
                continue
            if indication and indication not in metadata.get("indications", []):
                continue
            results.append({"id": model_id, "metadata": metadata})
        return results

    def get_model(self, model_id: str) -> dict:
        """Get model metadata by ID"""
        return self.models.get(model_id)

    def list_models(self) -> list:
        """List all registered models"""
        return [{"id": mid, "metadata": meta} for mid, meta in self.models.items()]

    def unregister(self, model_id: str) -> bool:
        """Remove a model from registry"""
        if model_id in self.models:
            del self.models[model_id]
            return True
        return False
