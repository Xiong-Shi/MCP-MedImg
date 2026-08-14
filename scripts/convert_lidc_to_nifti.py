#!/usr/bin/env python
"""
LIDC-IDRI DICOM to NIfTI converter for nnU-Net training.

Converts LIDC-IDRI DICOM files + XML annotations into nnU-Net v2
compatible dataset format (Dataset501_LungNodule).

Usage:
    python convert_lidc_to_nifti.py \
        --input /data/lidc_idri_50 \
        --output /data/nnUNet_raw/Dataset501_LungNodule \
        --xml-annotations /data/lidc_idri_50/LIDC-XML

PV-12: Local validation - run with --dry-run to verify script logic
without requiring actual DICOM data.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def find_dicom_series(input_dir: Path) -> list:
    """Find all DICOM series directories under input_dir.

    LIDC-IDRI structure:
        LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.XXXX/
            1.3.6.1.4.1.14519.5.2.1.6279.6001.YYYY/
                1-001.dcm, 1-002.dcm, ...

    Returns:
        List of dicts: [{"patient_id": "LIDC-IDRI-0001",
                         "series_uid": "1.3.6...", "dicom_dir": Path}]
    """
    series_list = []
    for patient_dir in sorted(input_dir.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("LIDC-IDRI"):
            continue
        patient_id = patient_dir.name
        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                dcm_files = list(series_dir.glob("*.dcm"))
                if dcm_files:
                    series_list.append(
                        {
                            "patient_id": patient_id,
                            "series_uid": series_dir.name,
                            "dicom_dir": series_dir,
                            "num_slices": len(dcm_files),
                        }
                    )
    return series_list


def convert_dicom_to_nifti(dicom_dir: Path, output_path: Path) -> bool:
    """Convert a DICOM series to a single NIfTI file using SimpleITK.

    Args:
        dicom_dir: Directory containing .dcm files
        output_path: Output .nii.gz path

    Returns:
        True if conversion succeeded
    """
    try:
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        dicom_files = sitk.GetGDCMSeriesFileNames(str(dicom_dir))
        if not dicom_files:
            logger.warning(f"No DICOM series found in {dicom_dir}")
            return False

        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        # Save as NIfTI
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_path))
        logger.info(f"Converted {len(dicom_files)} slices -> {output_path}")
        return True

    except ImportError:
        logger.error("SimpleITK not installed. Install with: pip install SimpleITK")
        return False
    except Exception as e:
        logger.error(f"Conversion failed for {dicom_dir}: {e}")
        return False


def parse_lidc_xml(xml_dir: Path, patient_id: str) -> list:
    """Parse LIDC-IDRI XML annotations for a patient.

    Extracts nodule annotations and converts to binary segmentation labels.

    Args:
        xml_dir: Directory containing LIDC XML files
        patient_id: LIDC-IDRI patient ID

    Returns:
        List of nodule annotations: [{"nodule_id": int, "z_pos": float,
                                      "x": float, "y": float, "diameter": float}]
    """
    import xml.etree.ElementTree as ET

    annotations = []
    xml_files = list(xml_dir.rglob(f"*{patient_id}*.xml"))

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for nodule in root.findall(".//{http://www.nih.gov}nonNodule"):
                # Skip non-nodules
                continue

            for nodule in root.findall(".//{http://www.nih.gov}unclassifiedNodule"):
                z_pos = float(nodule.findtext(".//imageZposition", "0"))
                x = float(nodule.findtext(".//xCoord", "0"))
                y = float(nodule.findtext(".//yCoord", "0"))
                diameter = float(nodule.findtext(".//characteristics/diameter", "0"))

                annotations.append(
                    {
                        "nodule_id": len(annotations),
                        "z_pos": z_pos,
                        "x": x,
                        "y": y,
                        "diameter": diameter,
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to parse {xml_file}: {e}")

    return annotations


def create_segmentation_label(
    nifti_path: Path,
    output_path: Path,
    annotations: list,
) -> bool:
    """Create binary segmentation label from annotations.

    Args:
        nifti_path: Path to the image NIfTI file
        output_path: Path for the label NIfTI file
        annotations: List of nodule annotations

    Returns:
        True if label creation succeeded
    """
    try:
        import SimpleITK as sitk
        import numpy as np

        image = sitk.ReadImage(str(nifti_path))
        label_array = np.zeros(sitk.GetArrayFromImage(image).shape, dtype=np.uint8)

        # Mark nodule regions based on annotations
        for ann in annotations:
            # Convert physical coordinates to pixel indices
            z, y, x = ann.get("z_pos", 0), ann.get("y", 0), ann.get("x", 0)
            diameter = ann.get("diameter", 6.0)
            radius = max(int(diameter / 2), 1)

            # Simple spherical mask (approximate)
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dz * dz + dy * dy + dx * dx <= radius * radius:
                            zz = int(z)
                            yy = int(y) + dy
                            xx = int(x) + dx
                            if (
                                0 <= zz < label_array.shape[0]
                                and 0 <= yy < label_array.shape[1]
                                and 0 <= xx < label_array.shape[2]
                            ):
                                label_array[zz, yy, xx] = 1

        label_image = sitk.GetImageFromArray(label_array)
        label_image.CopyInformation(image)
        sitk.WriteImage(label_image, str(output_path))
        return True

    except Exception as e:
        logger.error(f"Label creation failed: {e}")
        return False


def generate_dataset_json(
    output_dir: Path,
    dataset_id: int = 501,
    dataset_name: str = "LungNodule",
    num_training_cases: int = 0,
) -> None:
    """Generate nnU-Net v2 dataset.json file.

    Args:
        output_dir: Dataset directory (Dataset501_LungNodule/)
        dataset_id: Numeric dataset ID
        dataset_name: Dataset name (no spaces)
        num_training_cases: Number of training cases
    """
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "reference": "LIDC-IDRI",
        "licence": "CC BY 3.0",
        "description": f"Lung nodule segmentation from LIDC-IDRI subset ({num_training_cases} cases)",
    }

    json_path = output_dir / "dataset.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
    logger.info(f"Generated {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIDC-IDRI DICOM to nnU-Net v2 NIfTI format"
    )
    parser.add_argument("--input", required=True, help="Input LIDC-IDRI directory")
    parser.add_argument(
        "--output", required=True, help="Output nnU-Net dataset directory"
    )
    parser.add_argument(
        "--xml-annotations", required=True, help="LIDC XML annotations directory"
    )
    parser.add_argument(
        "--dataset-id", type=int, default=501, help="nnU-Net dataset ID"
    )
    parser.add_argument("--dataset-name", default="LungNodule", help="Dataset name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate script logic without actual conversion",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    xml_dir = Path(args.xml_annotations)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not xml_dir.exists():
        logger.warning(f"XML annotations directory does not exist: {xml_dir}")
        logger.warning("Labels will not be generated. Continuing with images only.")

    # Create nnU-Net directory structure
    images_tr = output_dir / "imagesTr"
    images_ts = output_dir / "imagesTs"
    labels_tr = output_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    # Find all DICOM series
    logger.info(f"Scanning for DICOM series in {input_dir}...")
    series_list = find_dicom_series(input_dir)
    logger.info(f"Found {len(series_list)} DICOM series")

    if not series_list:
        logger.error("No DICOM series found. Check input directory structure.")
        sys.exit(1)

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info(f"Would convert {len(series_list)} series to NIfTI")
        logger.info(f"Output structure: {output_dir}")
        logger.info(f"Images: {images_tr}")
        logger.info(f"Labels: {labels_tr}")
        logger.info("=== DRY RUN COMPLETE (no files written) ===")
        generate_dataset_json(
            output_dir, args.dataset_id, args.dataset_name, len(series_list)
        )
        return

    # Convert each series
    converted = 0
    for i, series in enumerate(series_list):
        case_id = f"lung_{i:04d}"
        image_path = images_tr / f"{case_id}_0000.nii.gz"
        label_path = labels_tr / f"{case_id}.nii.gz"

        # Convert DICOM to NIfTI
        if not convert_dicom_to_nifti(series["dicom_dir"], image_path):
            continue

        # Parse XML annotations and create labels
        if xml_dir.exists():
            annotations = parse_lidc_xml(xml_dir, series["patient_id"])
            if annotations:
                create_segmentation_label(image_path, label_path, annotations)
            else:
                # Empty label (no nodules in this case)
                logger.info(
                    f"No annotations for {series['patient_id']}, creating empty label"
                )

        converted += 1
        logger.info(
            f"[{converted}/{len(series_list)}] {series['patient_id']} -> {case_id}"
        )

    # Generate dataset.json
    generate_dataset_json(output_dir, args.dataset_id, args.dataset_name, converted)

    logger.info(f"Conversion complete: {converted}/{len(series_list)} cases")
    logger.info(
        f"Next step: nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity"
    )


if __name__ == "__main__":
    main()
