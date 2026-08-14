"""nnU-Net v2 inference engine adapter for real GPU inference.

Wraps nnU-Net's NIfTI-based inference pipeline and converts outputs
to the finding dict format expected by the Gateway layer.

This adapter provides the same interface as MockNNUNetEngine, enabling
seamless substitution for cloud validation scenarios.
"""

import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class nnUNetEngine:
    """Real nnU-Net inference engine adapter.

    Wraps nnUNetv2_predict functionality and exposes an .infer()
    interface compatible with MockNNUNetEngine.

    Expected directory structure after training:
        nnUNet_results/
            Dataset501_LungNodule/
                nnUNetTrainer__nnUNetPlans__3d_fullres/
                    fold_0/
                        checkpoint_final.pth
                        plans.json
                        dataset.json
    """

    def __init__(
        self,
        model_name: str = "nnunet_lung",
        model_path: Optional[str] = None,
        dataset_id: int = 501,
        device: str = "cuda:0",
        modality: str = "CT",
        task: str = "lung_nodule",
    ):
        self.model_name = model_name
        self.model_path = model_path  # Path to nnUNet_results/Dataset{did}/...
        self.dataset_id = dataset_id
        self.device = device
        self.modality = modality
        self.task = task
        self.is_loaded = False
        self.model_version = "nnunet-v2-trained"

        # nnU-Net Python API (imported lazily to avoid dependency during local validation)
        self._nnunet_predictor = None
        self._nnunet_available = False

    def _load_model(self) -> bool:
        """Load nnU-Net model weights and configuration.

        Returns True if model loaded successfully, False otherwise.
        """
        try:
            # Lazy import nnU-Net modules (only available on cloud with GPU)
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            from nnunetv2.paths import nnUNet_results
            import torch

            # Determine fold and configuration
            fold = 0  # Can be extended to ensemble multiple folds
            use_folds = (fold,)

            # Initialize predictor
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )

            # Load checkpoint
            if self.model_path:
                model_dir = Path(self.model_path)
            else:
                # Auto-detect from nnUNet_results environment variable
                model_dir = (
                    Path(nnUNet_results) / f"Dataset{self.dataset_id}_LungNodule"
                )

            predictor.initialize_from_trained_model_folder(
                str(model_dir),
                use_folds,
                checkpoint_name="checkpoint_final.pth",
            )

            self._nnunet_predictor = predictor
            self._nnunet_available = True
            self.is_loaded = True
            logger.info(f"nnU-Net model loaded from {model_dir}")
            return True

        except ImportError as e:
            logger.warning(
                f"nnU-Net not available (expected during local validation): {e}"
            )
            self._nnunet_available = False
            return False
        except Exception as e:
            logger.error(f"Failed to load nnU-Net model: {e}")
            logger.debug(traceback.format_exc())
            return False

    def infer(self, study_data: dict, parameters: dict = None) -> dict:
        """Run real nnU-Net inference on study data.

        Compatible interface with MockNNUNetEngine.infer().

        Args:
            study_data: Dict containing pixel_data (ndarray) and metadata.
                Required keys: pixel_data (np.ndarray), study_uid (str)
            parameters: Optional inference parameters.
                threshold: float (default 0.5)
                save_mask: bool (default False)

        Returns:
            Dict with segmentation mask, findings, and metadata.
            Format matches MockNNUNetEngine output for seamless substitution.
        """
        params = parameters or {}
        threshold = params.get("threshold", 0.5)
        save_mask = params.get("save_mask", False)

        # Lazy load on first inference
        if not self.is_loaded:
            if not self._load_model():
                # Fallback: return empty result with error info
                return {
                    "findings": [],
                    "segmentation_mask": None,
                    "model_version": self.model_version,
                    "inference_time_ms": 0,
                    "threshold_used": threshold,
                    "task": self.task,
                    "modality": self.modality,
                    "error": "nnU-Net model not loaded (local validation mode or missing weights)",
                }

        pixel_data = study_data.get("pixel_data")
        study_uid = study_data.get("study_uid", "unknown")

        if pixel_data is None:
            return {
                "findings": [],
                "segmentation_mask": None,
                "model_version": self.model_version,
                "inference_time_ms": 0,
                "threshold_used": threshold,
                "task": self.task,
                "modality": self.modality,
                "error": "No pixel_data provided",
            }

        try:
            start_time = time.perf_counter()

            # Convert pixel_data to nnU-Net expected format
            # nnU-Net expects NIfTI files, but we receive raw pixel arrays
            # This adapter converts ndarray -> temporary NIfTI -> nnU-Net inference
            import tempfile
            import nibabel as nib
            import SimpleITK as sitk

            # Create temporary NIfTI from pixel_data
            # Assume pixel_data is shape (H, W, D) for 3D or (H, W) for 2D
            if isinstance(pixel_data, np.ndarray):
                if pixel_data.ndim == 2:
                    # Add depth dimension for 2D data
                    pixel_data = np.expand_dims(pixel_data, axis=-1)

                # Create NIfTI image
                nifti_img = nib.Nifti1Image(pixel_data, np.eye(4))

                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = Path(tmpdir) / f"{study_uid}_0000.nii.gz"
                    output_dir = Path(tmpdir) / "output"
                    output_dir.mkdir()

                    nib.save(nifti_img, str(input_path))

                    # Run nnU-Net inference
                    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

                    # predictor already initialized in _load_model
                    self._nnunet_predictor.predict_from_files(
                        [[str(input_path)]],
                        str(output_dir),
                        save_probabilities=False,
                        overwrite=True,
                        num_processes_preprocessing=2,
                        num_processes_segmentation_export=2,
                    )

                    # Read segmentation output
                    output_files = list(output_dir.glob("*.nii.gz"))
                    if not output_files:
                        raise RuntimeError("nnU-Net produced no output")

                    seg_nifti = nib.load(str(output_files[0]))
                    segmentation = seg_nifti.get_fdata()

                    # Convert segmentation to findings
                    findings = self._segmentation_to_findings(
                        segmentation, pixel_data, threshold
                    )

                    inference_time_ms = int((time.perf_counter() - start_time) * 1000)

                    result = {
                        "findings": findings,
                        "segmentation_mask": segmentation.astype(np.uint8)
                        if save_mask
                        else None,
                        "model_version": self.model_version,
                        "inference_time_ms": inference_time_ms,
                        "threshold_used": threshold,
                        "task": self.task,
                        "modality": self.modality,
                    }

                    logger.info(
                        f"nnU-Net inference complete: {len(findings)} findings, "
                        f"{inference_time_ms}ms"
                    )
                    return result

        except Exception as e:
            logger.error(f"nnU-Net inference failed: {e}")
            logger.debug(traceback.format_exc())
            return {
                "findings": [],
                "segmentation_mask": None,
                "model_version": self.model_version,
                "inference_time_ms": int((time.perf_counter() - start_time) * 1000),
                "threshold_used": threshold,
                "task": self.task,
                "modality": self.modality,
                "error": str(e),
            }

    def _segmentation_to_findings(
        self,
        segmentation: np.ndarray,
        pixel_data: np.ndarray,
        threshold: float,
    ) -> List[Dict]:
        """Convert nnU-Net segmentation mask to finding dicts.

        Each connected component in the segmentation mask becomes one finding.
        """
        from scipy import ndimage

        findings = []
        label_id = 1  # Foreground label in binary segmentation

        # Find connected components
        labeled_array, num_features = ndimage.label(segmentation > 0)

        for component_id in range(1, num_features + 1):
            component_mask = labeled_array == component_id
            if not component_mask.any():
                continue

            # Compute component properties
            voxel_count = component_mask.sum()
            coords = np.argwhere(component_mask)
            centroid = coords.mean(axis=0)

            # Estimate physical size (mm)
            # Assuming typical CT voxel spacing ~0.5mm isotropic
            voxel_volume_mm3 = 0.5 * 0.5 * 0.5  # mm^3
            volume_mm3 = voxel_count * voxel_volume_mm3

            # Approximate diameter from volume (sphere assumption)
            import math

            diameter_mm = 2 * ((3 * volume_mm3) / (4 * math.pi)) ** (1 / 3)

            # Lung-RADS classification based on size
            if diameter_mm < 6:
                rads = "Lung-RADS 1"
            elif diameter_mm < 10:
                rads = "Lung-RADS 2"
            elif diameter_mm < 15:
                rads = "Lung-RADS 4A"
            else:
                rads = "Lung-RADS 4B"

            confidence = min(0.98, 0.65 + (volume_mm3 / 1000) * 0.1)

            findings.append(
                {
                    "type": "lung_nodule",
                    "location": {
                        "slice": int(centroid[2]) if len(centroid) > 2 else 0,
                        "x": int(centroid[1]),
                        "y": int(centroid[0]),
                        "z": int(centroid[2]) if len(centroid) > 2 else 0,
                    },
                    "size_mm": round(diameter_mm, 1),
                    "nodule_type": "solid",  # nnU-Net output doesn't distinguish types
                    "confidence": round(confidence, 2),
                    "rads_category": rads,
                    "recommendation": self._get_rads_recommendation(rads),
                    "voxel_count": int(voxel_count),
                }
            )

        return findings

    @staticmethod
    def _get_rads_recommendation(rads: str) -> str:
        recommendations = {
            "Lung-RADS 1": "Routine annual screening",
            "Lung-RADS 2": "Routine annual screening",
            "Lung-RADS 3": "6-month follow-up LDCT",
            "Lung-RADS 4A": "3-month follow-up LDCT or PET/CT",
            "Lung-RADS 4B": "Tissue diagnosis recommended",
        }
        return recommendations.get(rads, "Clinical correlation recommended")

    def health_check(self) -> bool:
        """Return model health status."""
        return self.is_loaded
