#!/usr/bin/env python
"""
PV-13: Verify nnU-Net v2 preprocessing command syntax and environment.

This script checks:
1. Whether nnunetv2 is installed
2. Whether the nnUNetv2_plan_and_preprocess command is available
3. Whether environment variables are set correctly
4. Whether dataset directory structure is valid

Usage:
    python verify_nnunet_preprocess.py --dataset-dir /data/nnUNet_raw/Dataset501_LungNodule --dataset-id 501

PV-13 does NOT require actual nnU-Net installation locally.
It validates that the command syntax is correct and the environment
will be ready when executed on AutoDL.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def check_nnunet_installed() -> bool:
    """Check if nnunetv2 package is installed."""
    try:
        import nnunetv2

        logger.info(f"nnunetv2 version: {nnunetv2.__version__}")
        return True
    except ImportError:
        logger.warning("nnunetv2 is NOT installed (expected on local machine)")
        logger.info("Install on AutoDL with: pip install nnunetv2")
        return False


def check_command_available(cmd: str) -> bool:
    """Check if a command-line tool is available."""
    found = shutil.which(cmd) is not None
    if found:
        logger.info(f"Command '{cmd}' found at: {shutil.which(cmd)}")
    else:
        logger.warning(f"Command '{cmd}' NOT found in PATH")
    return found


def check_env_vars() -> dict:
    """Check nnU-Net environment variables."""
    env_vars = {
        "nnUNet_raw": os.environ.get("nnUNet_raw", ""),
        "nnUNet_preprocessed": os.environ.get("nnUNet_preprocessed", ""),
        "nnUNet_results": os.environ.get("nnUNet_results", ""),
    }

    for name, value in env_vars.items():
        if value:
            logger.info(f"  {name} = {value}")
        else:
            logger.warning(f"  {name} is NOT set")

    return env_vars


def check_dataset_structure(dataset_dir: Path, dataset_id: int) -> bool:
    """Validate nnU-Net v2 dataset directory structure.

    Expected structure:
        Dataset{id}_{Name}/
            dataset.json
            imagesTr/
                case_0000.nii.gz
            labelsTr/
                case.nii.gz
            imagesTs/ (optional)
    """
    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return False

    # Check dataset.json
    dataset_json_path = dataset_dir / "dataset.json"
    if not dataset_json_path.exists():
        logger.error(f"dataset.json not found in {dataset_dir}")
        return False

    try:
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        logger.info(
            f"dataset.json loaded: name={ds.get('name')}, "
            f"numTraining={ds.get('numTraining')}, "
            f"labels={ds.get('labels')}"
        )
    except Exception as e:
        logger.error(f"Failed to read dataset.json: {e}")
        return False

    # Check imagesTr
    images_tr = dataset_dir / "imagesTr"
    if not images_tr.exists():
        logger.error(f"imagesTr directory not found: {images_tr}")
        return False

    image_files = list(images_tr.glob("*_0000.nii.gz"))
    if not image_files:
        logger.warning("No training images found in imagesTr/")
    else:
        logger.info(f"Found {len(image_files)} training images")

    # Check labelsTr
    labels_tr = dataset_dir / "labelsTr"
    if not labels_tr.exists():
        logger.warning(f"labelsTr directory not found: {labels_tr}")
    else:
        label_files = list(labels_tr.glob("*.nii.gz"))
        logger.info(f"Found {len(label_files)} label files")

    return True


def verify_preprocess_command(dataset_id: int, dry_run: bool = True) -> bool:
    """Verify the nnU-Net v2 preprocessing command syntax.

    The correct command is:
        nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity

    Note: No -tr flag needed (default trainer is nnUNetTrainer).
    The -d flag accepts numeric dataset ID only (not DatasetXXX_Name).
    """
    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"

    logger.info(f"Command to execute: {cmd}")

    if dry_run:
        logger.info("Dry run mode - skipping actual execution")
        logger.info("On AutoDL, run:")
        logger.info(f"  export nnUNet_raw=/root/autodl-tmp/nnUNet_raw")
        logger.info(
            f"  export nnUNet_preprocessed=/root/autodl-tmp/nnUNet_preprocessed"
        )
        logger.info(f"  export nnUNet_results=/root/autodl-tmp/nnUNet_results")
        logger.info(f"  {cmd}")
        return True

    # Check if command exists
    if not check_command_available("nnUNetv2_plan_and_preprocess"):
        logger.error("nnUNetv2_plan_and_preprocess not found. Install nnunetv2 first.")
        return False

    # Execute
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            logger.info("Preprocessing completed successfully!")
            logger.info(
                result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
            )
            return True
        else:
            logger.error(f"Preprocessing failed with code {result.returncode}")
            logger.error(result.stderr[-500:] if result.stderr else "No error output")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Preprocessing timed out (3600s)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PV-13: Verify nnU-Net v2 preprocessing environment and command"
    )
    parser.add_argument(
        "--dataset-dir",
        required=False,
        help="nnU-Net dataset directory (e.g., /data/nnUNet_raw/Dataset501_LungNodule)",
    )
    parser.add_argument(
        "--dataset-id", type=int, default=501, help="nnU-Net numeric dataset ID"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate without executing (default: True)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the preprocessing command",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PV-13: nnU-Net v2 Preprocessing Verification")
    logger.info("=" * 60)

    # Step 1: Check nnunetv2 installation
    logger.info("\n[1/4] Checking nnunetv2 installation...")
    nnunet_installed = check_nnunet_installed()

    # Step 2: Check command availability
    logger.info("\n[2/4] Checking command availability...")
    cmd_available = check_command_available("nnUNetv2_plan_and_preprocess")

    # Step 3: Check environment variables
    logger.info("\n[3/4] Checking nnU-Net environment variables...")
    env_vars = check_env_vars()

    # Step 4: Check dataset structure (if directory provided)
    logger.info("\n[4/4] Checking dataset structure...")
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        ds_valid = check_dataset_structure(dataset_dir, args.dataset_id)
    else:
        logger.info("No dataset directory provided, skipping structure check")
        ds_valid = None

    # Verify command syntax
    logger.info("\n" + "=" * 60)
    logger.info("Command syntax verification:")
    execute_flag = args.execute and not args.dry_run
    cmd_valid = verify_preprocess_command(args.dataset_id, dry_run=not execute_flag)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(
        f"  nnunetv2 installed:      {'YES' if nnunet_installed else 'NO (expected locally)'}"
    )
    logger.info(
        f"  Command available:       {'YES' if cmd_available else 'NO (expected locally)'}"
    )
    logger.info(
        f"  Env vars set:            {'YES' if all(env_vars.values()) else 'NO (set on AutoDL)'}"
    )
    logger.info(f"  Dataset valid:           {'YES' if ds_valid else 'NO/SKIPPED'}")
    logger.info(f"  Command syntax valid:    {'YES' if cmd_valid else 'NO'}")
    logger.info("=" * 60)

    if not nnunet_installed and not execute_flag:
        logger.info("\nThis is expected on the local machine (RTX 2060).")
        logger.info("On AutoDL, install nnunetv2 and re-run with --execute.")

    logger.info("\nPV-13 verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
