"""AI router for inspection-type-driven model selection"""


class AIRouter:
    def __init__(self, registry):
        self.registry = registry

    def route(
        self, modality: str, body_part: str, indication: str, priority: str = "ROUTINE"
    ) -> dict:
        """Route to optimal AI model based on inspection type"""
        candidates = self.registry.discover(
            modality=modality, body_part=body_part, indication=indication
        )
        if not candidates:
            return None

        # Prioritize by performance (AUC)
        candidates.sort(
            key=lambda x: x["metadata"]["performance"].get("AUC", 0), reverse=True
        )

        # For STAT priority, favor speed over accuracy
        if priority == "STAT":
            candidates.sort(key=lambda x: x["metadata"].get("inference_time_ms", 1000))

        return candidates[0]

    def discover(
        self, modality: str = None, body_part: str = None, indication: str = None
    ) -> list:
        """Discover available models for given criteria"""
        return self.registry.discover(
            modality=modality, body_part=body_part, indication=indication
        )

    def infer(self, model_id: str, study_data: dict, parameters: dict = None) -> dict:
        """Route inference request to model engine"""
        model = self.registry.get_model(model_id)
        if not model:
            return {
                "result": None,
                "error": {
                    "code": "MODEL_UNAVAILABLE",
                    "message": f"Model {model_id} not found",
                    "retryable": False,
                },
            }

        # Get inference engine from model metadata
        engine = model.get("engine")
        if engine is None:
            return {
                "result": None,
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": "No inference engine bound to model",
                    "retryable": False,
                },
            }

        try:
            result = engine.infer(study_data, parameters or {})
            return {
                "result": result,
                "outputFormat": model.get("output_format", "DICOM_SR"),
                "model_version": model.get("version", "unknown"),
                "error": None,
            }
        except Exception as e:
            return {
                "result": None,
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": str(e),
                    "retryable": True,
                },
            }
