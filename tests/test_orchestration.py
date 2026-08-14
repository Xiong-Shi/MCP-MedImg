"""Tests for Layer 3: AI Orchestration Layer"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from mcp_medimg.orchestration import (
    ModelRegistry,
    AIRouter,
    VersionManager,
    ResultFormatter,
    MockNNUNetEngine,
)


class TestModelRegistry:
    def setup_method(self):
        self.registry = ModelRegistry()
        self.mock_engine = MockNNUNetEngine(modality="CT", task="lung_nodule")
        self.registry.register(
            "lung-nodule-v1",
            {
                "modality": ["CT"],
                "bodyPart": ["CHEST"],
                "indications": ["screening", "followup"],
                "performance": {"AUC": 0.92, "Sensitivity": 0.88},
                "regulatory": ["FDA_510k", "NMPA_Grade2"],
                "version": "v1.0",
                "inference_time_ms": 200,
                "engine": self.mock_engine,
                "output_format": "DICOM_SR",
            },
        )

    def test_register_and_discover(self):
        results = self.registry.discover(
            modality="CT", body_part="CHEST", indication="screening"
        )
        assert len(results) == 1
        assert results[0]["id"] == "lung-nodule-v1"

    def test_discover_no_match(self):
        results = self.registry.discover(modality="MR", body_part="BRAIN")
        assert len(results) == 0

    def test_missing_required_field(self):
        with pytest.raises(ValueError, match="Missing required field"):
            self.registry.register("bad-model", {"modality": ["CT"]})

    def test_get_model(self):
        model = self.registry.get_model("lung-nodule-v1")
        assert model is not None
        assert model["version"] == "v1.0"

    def test_unregister(self):
        assert self.registry.unregister("lung-nodule-v1") is True
        assert self.registry.get_model("lung-nodule-v1") is None


class TestAIRouter:
    def setup_method(self):
        self.registry = ModelRegistry()
        self.lung_engine = MockNNUNetEngine(modality="CT", task="lung_nodule")
        self.breast_engine = MockNNUNetEngine(modality="MG", task="breast_lesion")

        self.registry.register(
            "lung-nodule-v1",
            {
                "modality": ["CT"],
                "bodyPart": ["CHEST"],
                "indications": ["screening", "followup"],
                "performance": {"AUC": 0.92},
                "regulatory": ["FDA_510k"],
                "version": "v1.0",
                "inference_time_ms": 200,
                "engine": self.lung_engine,
            },
        )
        self.registry.register(
            "lung-nodule-v2",
            {
                "modality": ["CT"],
                "bodyPart": ["CHEST"],
                "indications": ["screening"],
                "performance": {"AUC": 0.95},
                "regulatory": ["FDA_510k"],
                "version": "v2.0",
                "inference_time_ms": 150,
                "engine": self.lung_engine,
            },
        )
        self.registry.register(
            "breast-lesion-v1",
            {
                "modality": ["MG"],
                "bodyPart": ["BREAST"],
                "indications": ["screening"],
                "performance": {"AUC": 0.89},
                "regulatory": ["CE"],
                "version": "v1.0",
                "inference_time_ms": 300,
                "engine": self.breast_engine,
            },
        )
        self.router = AIRouter(self.registry)

    def test_route_lung(self):
        result = self.router.route("CT", "CHEST", "screening")
        assert result["id"] == "lung-nodule-v2"

    def test_route_breast(self):
        result = self.router.route("MG", "BREAST", "screening")
        assert result["id"] == "breast-lesion-v1"

    def test_route_no_match(self):
        result = self.router.route("MR", "BRAIN", "screening")
        assert result is None

    def test_stat_prefers_speed(self):
        result = self.router.route("CT", "CHEST", "screening", priority="STAT")
        assert result["id"] == "lung-nodule-v2"

    def test_infer_success(self):
        study_data = {
            "pixel_data": np.zeros((512, 512), dtype=np.int16),
            "study_uid": "1.2.3.4.5",
        }
        result = self.router.infer("lung-nodule-v1", study_data)
        assert result["error"] is None
        assert "findings" in result["result"]
        assert result["model_version"] == "v1.0"

    def test_infer_model_unavailable(self):
        result = self.router.infer("nonexistent-model", {})
        assert result["error"]["code"] == "MODEL_UNAVAILABLE"


class TestVersionManager:
    def setup_method(self):
        self.vm = VersionManager()
        self.vm.register_version("lung-nodule", "v1.0", status="stable")
        self.vm.canary_release("lung-nodule", "v2.0", canary_percentage=30)

    def test_canary_release(self):
        versions = self.vm.models["lung-nodule"]
        assert "v2.0" in versions
        assert versions["v2.0"]["status"] == "canary"
        assert versions["v2.0"]["canary_percentage"] == 30

    def test_consistent_hashing(self):
        # Same patient hash should always get same version
        v1 = self.vm.get_version_for_request("lung-nodule", "aabbccdd11223344")
        v2 = self.vm.get_version_for_request("lung-nodule", "aabbccdd11223344")
        assert v1 == v2

    def test_canary_distribution(self):
        # With 30% canary, some patients should get canary
        canary_count = 0
        for i in range(100):
            patient_hash = f"{i:08x}"
            version = self.vm.get_version_for_request("lung-nodule", patient_hash)
            if version == "v2.0":
                canary_count += 1
        assert 15 < canary_count < 45  # Roughly 30%

    def test_warmup_detection(self):
        self.vm.register_version("new-model", "v1.0")
        assert self.vm.is_new_deployment("new-model", "v1.0") is True
        # Process 60 cases
        for _ in range(60):
            self.vm.record_case("new-model", "v1.0")
        assert self.vm.is_new_deployment("new-model", "v1.0") is False


class TestResultFormatter:
    def setup_method(self):
        self.formatter = ResultFormatter()
        self.findings = [
            {
                "type": "lung_nodule",
                "size_mm": 8.5,
                "rads_category": "Lung-RADS 2",
                "confidence": 0.92,
            },
        ]
        self.study_info = {"study_uid": "1.2.3.4.5", "patient_hash": "sha256:abc123"}
        self.model_info = {
            "name": "lung-nodule-v1",
            "version": "v1.0",
            "verified": True,
        }

    def test_to_dicom_sr(self):
        sr = self.formatter.to_dicom_sr(self.findings, self.study_info, self.model_info)
        assert sr["sopClassUID"] == "1.2.840.10008.5.1.4.1.1.88.22"
        assert sr["verificationFlag"] == "VERIFIED"
        assert len(sr["findings"]) == 1
        assert sr["modelInfo"]["name"] == "lung-nodule-v1"

    def test_to_dicom_seg(self):
        mask = np.zeros((512, 512), dtype=np.uint8)
        seg = self.formatter.to_dicom_seg(
            mask,
            [{"study_uid": "1.2.3"}],
            {"algorithm_name": "nnU-Net", "algorithm_version": "v2"},
        )
        assert seg["sopClassUID"] == "1.2.840.10008.5.1.4.1.1.66.4"
        assert seg["algorithmName"] == "nnU-Net"

    def test_to_fhir_observation(self):
        obs = self.formatter.to_fhir_observation(self.findings, self.study_info)
        assert len(obs) == 1
        assert obs[0]["resourceType"] == "Observation"
        assert obs[0]["status"] == "final"

    def test_hash_result(self):
        h = self.formatter.hash_result(self.findings[0])
        assert h.startswith("sha256:")
        # Same input produces same hash
        h2 = self.formatter.hash_result(self.findings[0])
        assert h == h2


class TestMockNNUNetEngine:
    def setup_method(self):
        self.engine = MockNNUNetEngine(modality="CT", task="lung_nodule")

    def test_deterministic_output(self):
        study_data = {
            "study_uid": "1.2.3.4.5",
            "pixel_data": np.zeros((512, 512), dtype=np.int16),
        }
        r1 = self.engine.infer(study_data)
        r2 = self.engine.infer(study_data)
        # Same study_uid should produce same findings
        assert r1["findings"] == r2["findings"]

    def test_lung_nodule_findings(self):
        study_data = {
            "study_uid": "9.8.7.6.5",
            "pixel_data": np.zeros((512, 512), dtype=np.int16),
        }
        result = self.engine.infer(study_data)
        for finding in result["findings"]:
            assert finding["type"] == "lung_nodule"
            assert "rads_category" in finding
            assert "size_mm" in finding
            assert "confidence" in finding

    def test_breast_lesion_engine(self):
        engine = MockNNUNetEngine(modality="MG", task="breast_lesion")
        study_data = {"study_uid": "2.2.2.2.2"}
        result = engine.infer(study_data)
        for finding in result["findings"]:
            assert finding["type"] == "breast_lesion"
            assert "rads_category" in finding

    def test_bone_age_engine(self):
        engine = MockNNUNetEngine(modality="XR", task="bone_age")
        study_data = {"study_uid": "3.3.3.3.3", "patient_age": 10}
        result = engine.infer(study_data)
        assert len(result["findings"]) == 1
        assert result["findings"][0]["type"] == "bone_age_assessment"
        assert "bone_age" in result["findings"][0]

    def test_segmentation_mask(self):
        study_data = {
            "study_uid": "4.4.4.4.4",
            "pixel_data": np.zeros((512, 512), dtype=np.int16),
        }
        result = self.engine.infer(study_data)
        mask = result["segmentation_mask"]
        assert mask.shape == (512, 512)
        # If there are findings, mask should have non-zero pixels
        if len(result["findings"]) > 0:
            assert mask.sum() > 0

    def test_health_check(self):
        assert self.engine.health_check() is True

    def test_rads_recommendations(self):
        # Ensure all RADS categories have recommendations
        study_data = {
            "study_uid": "5.5.5.5.5",
            "pixel_data": np.zeros((512, 512), dtype=np.int16),
        }
        result = self.engine.infer(study_data)
        for finding in result["findings"]:
            assert finding.get("recommendation") is not None
            assert len(finding["recommendation"]) > 0
