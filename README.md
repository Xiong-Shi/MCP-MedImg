# MCP-MedImg: Enabling Standardized PACS-AI Integration through the Model Context Protocol

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21941869.svg)](https://doi.org/10.5281/zenodo.21941869)

> Reference implementation for the paper:  
> "MCP-MedImg: Enabling Standardized PACS-AI Integration through the Model Context Protocol"
>
> **Author**: Xiong Shi (ORCID: [0009-0006-2745-491X](https://orcid.org/0009-0006-2745-491X))

## Architecture

MCP-MedImg extends the Model Context Protocol (MCP) with a four-layer protocol stack specifically designed for medical imaging workflows:

- **L1 Transport**: DICOMweb / DICOM C-MOVE / dicom-mcp compatibility wrapper
- **L2 Security & Compliance**: Hierarchical PHI de-identification + 14-column cryptographic audit trail
- **L3 AI Orchestration**: Model registry, inspection-type routing, version management, result formatting
- **L4 Application Semantics**: Lesion encoding (RADS/BI-RADS), conditional verification state machine

## Quick Start

### Prerequisites
- Ubuntu 22.04 (recommended) / macOS / Windows WSL2
- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.1+ (optional; Mock engine available for CPU-only environments)

### 1. Clone & Install
```bash
git clone https://github.com/Xiong-Shi/MCP-MedImg.git
cd MCP-MedImg
pip install -r requirements.txt
```

### 2. Start Orthanc PACS
```bash
docker-compose -f config/orthanc/docker-compose.yml up -d
```

### 3. Run Unit Tests (Protocol Validation)
```bash
pytest tests/ -v --tb=short
# Expected: 132 passing, 3 skipped across 10 validation scenarios
```

### 4. Run Benchmarks
```bash
bash scripts/reproduce_lidc_benchmark.sh      # Table 18: LIDC-IDRI inference latency (n=200)
bash scripts/reproduce_heart_benchmark.sh     # Table 19, 22: MSD Heart 5-fold validation
bash scripts/reproduce_protocol_overhead.sh   # Table 20, 21: Protocol overhead decomposition (n=98)
```

## Datasets & Model Weights

Due to file size limits, pretrained model weights and full preprocessed datasets are available via separate download:

| Asset | Size | Description |
|-------|------|-------------|
| nnU-Net 5-fold ensemble (MSD Heart) | ~1.2 GB | Trained weights, mean Dice 0.9324±0.0062 |
| LIDC-IDRI preprocessed (n=200) | ~8 GB | HU-normalized, resampled CT volumes (inference uses randomly initialized model, not trained weights) |
| MSD Heart preprocessed (n=20) | ~1 GB | nnU-Net v2 auto-preprocessed |
| Benchmark logs | ~50 MB | Table 18-21 original timing logs |

Download instructions:
```bash
bash scripts/download_pretrained_weights.sh
```

## Key Experimental Results

| Metric | Value | Details |
|--------|-------|---------|
| MSD Heart 5-fold Dice | **0.9324 ± 0.0062** | nnU-Net v2, 1000 epochs per fold |
| LIDC Inference (n=200) | **5.40s** mean end-to-end | RTX 4090D, 22.58M params |
| Protocol Overhead | **567.47ms** (99.8% L1 Transport) | n=98, 3.41% measurement error |
| Unit Tests | **132 passing, 3 skipped** | 10 validation scenarios |

**Analysis Figures**: Detailed analysis charts (LIDC bimodal distribution, PROTO error robustness) are available in the Zenodo dataset:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21941869.svg)](https://doi.org/10.5281/zenodo.21941869)

This implementation addresses the MCP-38 threat taxonomy for medical imaging:
- Hierarchical PHI de-identification (Patient / Department / System level)
- Cryptographic audit trails with SHA-256 hashing
- Containerized sandbox execution
- Input sanitization against indirect prompt injection

## Reproducibility

All experimental results from the paper can be reproduced via the scripts in `scripts/`.

| Paper Table/Figure | Reproduce Script | Expected Runtime |
|-------------------|------------------|------------------|
| Table 16 (134 tests) | `pytest tests/` | ~5 min (CPU) |
| Table 18 (LIDC, n=200) | `scripts/reproduce_lidc_benchmark.sh` | ~20 min (RTX 4090D) |
| Table 19 (Heart, n=20) | `scripts/reproduce_heart_benchmark.sh` | ~12 min (RTX 4090D) |
| Table 20-21 (Overhead) | `scripts/reproduce_protocol_overhead.sh` | ~15 min (RTX 4090D) |
| Table 22 (Dice) | Included in Heart script | -- |

## Citation

If you use this code, please cite:
```bibtex
@article{mcp_medimg_2026,
  title={MCP-MedImg: Enabling Standardized PACS-AI Integration through the Model Context Protocol},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file.

## Disclaimer

This is a research prototype for protocol validation and reproducibility. **Not for clinical use** without appropriate regulatory approval (FDA/NMPA/CE).
