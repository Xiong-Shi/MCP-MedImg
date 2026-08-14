"""Mock nnU-Net inference engine for local validation

Simulates AI inference without requiring GPU or actual model weights.
Produces realistic detection results based on image properties
to enable full protocol-level validation.
"""

import hashlib
import random
import numpy as np
from typing import Dict, List, Tuple


class MockNNUNetEngine:
    """Mock nnU-Net engine that simulates segmentation and detection.

    Uses deterministic pseudo-random generation based on study UID
    to produce consistent results for the same input, enabling
    reproducible validation scenarios.
    """

    def __init__(self, modality: str = "CT", task: str = "lung_nodule"):
        self.modality = modality
        self.task = task
        self.is_loaded = True
        self.model_version = "mock-nnunet-v2.2"

    def infer(self, study_data: dict, parameters: dict = None) -> dict:
        """Run mock inference on study data.

        Args:
            study_data: Dict containing pixel_data (ndarray) and metadata
            parameters: Optional inference parameters (threshold, etc.)

        Returns:
            Dict with segmentation mask, findings, and metadata
        """
        params = parameters or {}
        threshold = params.get("threshold", 0.5)

        # Use study UID as seed for deterministic results
        study_uid = study_data.get("study_uid", "default")
        seed = int(hashlib.sha256(study_uid.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        pixel_data = study_data.get("pixel_data")
        if pixel_data is None:
            # Generate a synthetic 512x512 image if not provided
            pixel_data = np.zeros((512, 512), dtype=np.int16)

        # Simulate nodule detection based on task
        findings = self._generate_findings(rng, study_data)
        segmentation_mask = self._generate_segmentation(pixel_data, findings, params)

        return {
            "findings": findings,
            "segmentation_mask": segmentation_mask,
            "model_version": self.model_version,
            "inference_time_ms": rng.randint(150, 400),
            "threshold_used": threshold,
            "task": self.task,
            "modality": self.modality,
        }

    def _generate_findings(self, rng: random.Random, study_data: dict) -> List[Dict]:
        """Generate realistic findings based on task type"""
        findings = []

        if self.task == "lung_nodule":
            # Simulate 0-3 nodules
            num_nodules = rng.randint(0, 3)
            for i in range(num_nodules):
                size_mm = round(rng.uniform(3.0, 25.0), 1)
                nodule_type = rng.choice(["solid", "subsolid", "ground_glass"])
                confidence = round(rng.uniform(0.65, 0.98), 2)

                # Lung-RADS classification
                if size_mm < 6:
                    rads = "Lung-RADS 1"
                elif size_mm < 10:
                    if nodule_type == "solid":
                        rads = "Lung-RADS 2"
                    else:
                        rads = "Lung-RADS 3"
                elif size_mm < 15:
                    rads = "Lung-RADS 4A"
                else:
                    rads = "Lung-RADS 4B"

                findings.append(
                    {
                        "type": "lung_nodule",
                        "location": {
                            "slice": rng.randint(0, 9),
                            "x": rng.randint(100, 400),
                            "y": rng.randint(100, 400),
                            "z": rng.randint(0, 9),
                        },
                        "size_mm": size_mm,
                        "nodule_type": nodule_type,
                        "confidence": confidence,
                        "rads_category": rads,
                        "recommendation": self._get_rads_recommendation(rads),
                    }
                )

        elif self.task == "breast_lesion":
            num_findings = rng.randint(0, 2)
            for i in range(num_findings):
                mass_shape = rng.choice(["round", "oval", "irregular"])
                mass_margin = rng.choice(
                    ["circumscribed", "obscured", "spiculated", "indistinct"]
                )
                calcification = rng.choice(["none", "benign", "suspicious"])

                if mass_shape == "round" and mass_margin == "circumscribed":
                    birads = "BI-RADS 2"
                elif mass_shape == "irregular" or mass_margin == "spiculated":
                    birads = "BI-RADS 4"
                elif calcification == "suspicious":
                    birads = "BI-RADS 4"
                else:
                    birads = "BI-RADS 3"

                findings.append(
                    {
                        "type": "breast_lesion",
                        "location": {
                            "x": rng.randint(100, 400),
                            "y": rng.randint(100, 400),
                        },
                        "size_mm": round(rng.uniform(5.0, 30.0), 1),
                        "mass_shape": mass_shape,
                        "mass_margin": mass_margin,
                        "calcification": calcification,
                        "confidence": round(rng.uniform(0.60, 0.95), 2),
                        "rads_category": birads,
                        "recommendation": self._get_birads_recommendation(birads),
                    }
                )

        elif self.task == "bone_age":
            # Simulate bone age assessment
            chronological_age = study_data.get("patient_age", 10)
            bone_age = chronological_age + rng.uniform(-1.5, 1.5)
            findings.append(
                {
                    "type": "bone_age_assessment",
                    "chronological_age": chronological_age,
                    "bone_age": round(bone_age, 1),
                    "confidence": round(rng.uniform(0.70, 0.92), 2),
                    "discrepancy_months": round(
                        abs(bone_age - chronological_age) * 12, 1
                    ),
                    "rads_category": None,
                    "recommendation": "Routine follow-up"
                    if abs(bone_age - chronological_age) < 1.0
                    else "Endocrinology referral recommended",
                }
            )

        return findings

    def _generate_segmentation(
        self, pixel_data: np.ndarray, findings: list, params: dict
    ) -> np.ndarray:
        """Generate segmentation mask from findings"""
        if hasattr(pixel_data, "shape"):
            shape = pixel_data.shape
        else:
            shape = (512, 512)

        mask = np.zeros(shape, dtype=np.uint8)

        for i, finding in enumerate(findings):
            loc = finding.get("location", {})
            x = loc.get("x", 256)
            y = loc.get("y", 256)

            # Size in pixels (approximate)
            size_mm = finding.get("size_mm", 10)
            radius_px = max(2, int(size_mm / 2))

            if len(shape) == 2:
                y_grid, x_grid = np.ogrid[: shape[0], : shape[1]]
                circle = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius_px**2
                mask[circle] = i + 1
            elif len(shape) == 3:
                z = loc.get("z", 0)
                y_grid, x_grid = np.ogrid[: shape[1], : shape[2]]
                circle = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius_px**2
                mask[z][circle] = i + 1

        return mask

    def _get_rads_recommendation(self, rads: str) -> str:
        recommendations = {
            "Lung-RADS 1": "Routine annual screening",
            "Lung-RADS 2": "Routine annual screening",
            "Lung-RADS 3": "6-month follow-up LDCT",
            "Lung-RADS 4A": "3-month follow-up LDCT or PET/CT",
            "Lung-RADS 4B": "Tissue diagnosis recommended",
        }
        return recommendations.get(rads, "Clinical correlation recommended")

    def _get_birads_recommendation(self, birads: str) -> str:
        recommendations = {
            "BI-RADS 2": "Routine screening",
            "BI-RADS 3": "6-month follow-up",
            "BI-RADS 4": "Biopsy recommended",
        }
        return recommendations.get(birads, "Clinical correlation recommended")

    def health_check(self) -> bool:
        """Return model health status"""
        return self.is_loaded
