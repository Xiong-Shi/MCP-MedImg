# Reproducibility Guide

## Hardware Environment

All experiments were conducted on AutoDL cloud instances with the following specifications:

| Component | Instance 1 (497机) | Instance 2 (505机) |
|-----------|-------------------|-------------------|
| GPU | NVIDIA GeForce RTX 4090D (24GB) | NVIDIA GeForce RTX 4090D (24GB) |
| CPU | AMD EPYC | AMD EPYC |
| RAM | 30GB | 30GB |
| CUDA | 12.1 | 12.1 |
| PyTorch | 2.3.0 | 2.3.0 |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

## Software Versions

```
nnU-Net v2: latest from GitHub (Aug 2026)
pydicom: 2.4.4
highdicom: 0.22.0
Orthanc: 1.12.3 (with DICOMweb plugin)
MCP SDK: 1.0.0
Python: 3.10.12
```

## Experiment Reproduction Steps

### Table 16: Unit Tests (134 passing, 1 skipped)

```bash
# No GPU required, Mock engine used
pytest tests/ -v --tb=short
```

**Expected output**: 134 passed, 1 skipped, 0 failed

### Table 18: LIDC Inference Latency (n=200)

**Prerequisites**: LIDC-IDRI dataset downloaded and preprocessed

```bash
# Download dataset (if not present)
python scripts/prepare_lidc_data.py

# Run inference benchmark
bash scripts/reproduce_lidc_benchmark.sh
```

**Expected results**:
- Load time: 1158.1 +/- 650.5 ms
- Preprocess time: 869.3 +/- 549.2 ms
- Inference time: 3373.6 +/- 3514.7 ms
- End-to-end: 5401.4 +/- 4677.7 ms

### Table 19 & 22: Heart 5-Fold Validation

**Prerequisites**: MSD Heart dataset + trained 5-fold ensemble weights

```bash
# Download pretrained weights
bash scripts/download_pretrained_weights.sh

# Run benchmark
bash scripts/reproduce_heart_benchmark.sh
```

**Expected results**:
- Fold Dice: [0.9381, 0.9323, 0.9259, 0.9410, 0.9250]
- Mean Dice: 0.9325 +/- 0.0064
- End-to-end latency: 32.75s

### Table 20 & 21: Protocol Overhead Decomposition

**Prerequisites**: Orthanc running with DICOMweb, LIDC data uploaded

```bash
# Start Orthanc
docker-compose -f config/orthanc/docker-compose.yml up -d

# Upload DICOM to Orthanc
python scripts/upload_to_orthanc.py

# Run protocol overhead experiment
bash scripts/reproduce_protocol_overhead.sh
```

**Expected results**:
- L1 Transport: 566.35ms (99.8%)
- L2 Security: 0.36ms (0.1%)
- L3 Orchestration: 0.44ms (0.1%)
- L4 Semantics: 0.32ms (0.1%)
- Total overhead: 567.47ms
- G0+overhead vs G4 error: 3.41%
- Spearman rho: 0.872 (p < 0.001)

## Data Preparation

### LIDC-IDRI

1. Download from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
2. Convert DICOM to NIfTI: `python scripts/convert_lidc_to_nifti.py`
3. Preprocess (HU normalization, resampling): nnU-Net v2 auto-preprocess

### MSD Heart

1. Download from [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
2. Preprocess with nnU-Net v2: `nnUNetv2_plan_and_preprocess -d 002`

## Known Issues & Limitations

1. **cuDNN benchmark**: Must set `torch.backends.cudnn.benchmark = False` to avoid OOM on RTX 4090D
2. **Orthanc restart**: Instance reboot requires manual Orthanc restart (not systemd-managed)
3. **GPU memory**: 512^3 volumes may OOM with batch_size > 2 on 24GB VRAM
4. **WADO-RS latency**: L1 transport latency depends on network conditions; use localhost for reproducibility

## Contact

For reproduction issues, please open a GitHub issue or contact the corresponding author.
