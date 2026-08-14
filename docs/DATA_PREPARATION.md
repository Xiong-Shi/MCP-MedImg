# Data Preparation Guide

## LIDC-IDRI Dataset

### Source
- **Origin**: The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)
- **Citation**: Armato III, S.G., et al. "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans." Medical Physics, 2011, 38(2): 915-931.
- **Download**: [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

### Preprocessing Steps

1. **DICOM to NIfTI Conversion**
   ```bash
   python scripts/convert_lidc_to_nifti.py \
       --input_dir /path/to/LIDC-IDRI \
       --output_dir data/preprocessed/lidc
   ```

2. **HU Normalization & Resampling**
   ```bash
   # Using nnU-Net v2 auto-preprocess
   nnUNetv2_plan_and_preprocess -d 001 \
       --verify_dataset_integrity
   ```

3. **Quality Check**
   ```bash
   python scripts/verify_nnunet_preprocess.py \
       --dataset_id 001
   ```

### Statistics (200 cases used in paper)
- Mean dimensions: 512 x 512 x 400 (varies significantly)
- Voxel spacing: resampled to 1.0 x 1.0 x 1.0 mm
- HU range: [-1024, 3071] (clip to [-1000, 400] for lung window)

## MSD Heart Dataset

### Source
- **Origin**: Medical Segmentation Decathlon (Task 02: Heart)
- **Citation**: Simpson, A.L., et al. "A large annotated medical image dataset for the development and evaluation of segmentation algorithms." arXiv:1902.09063, 2019.
- **Download**: [Medical Decathlon](http://medicaldecathlon.com/)

### Preprocessing Steps

1. **Auto-preprocess with nnU-Net v2**
   ```bash
   nnUNetv2_plan_and_preprocess -d 002 \
       --verify_dataset_integrity
   ```

2. **Training (5-fold cross-validation)**
   ```bash
   # Run each fold serially to avoid OOM
   for fold in {0..4}; do
       nnUNetv2_train 002 3d_fullres $fold \
           -tr nnUNetTrainer
   done
   ```

### Statistics
- 20 training cases (MRI, cine sequence)
- Mean dimensions: ~320 x 320 x 100
- Ground truth: Left Atrium + Myocardium segmentation

## Data Availability Statement

The reference implementation code, protocol specifications, and reproducibility scripts are available in this repository. The public datasets used (LIDC-IDRI and MSD Task02 Heart) are available via The Cancer Imaging Archive and the Medical Segmentation Decathlon, respectively. Preprocessed data, benchmark logs, and trained model weights are deposited at Zenodo (DOI to be assigned). Due to patient privacy regulations and the MCP-38 threat model, raw audit logs containing hashed PHI are retained by the corresponding author and can be made available to editors and reviewers upon reasonable request under a signed data use agreement.
