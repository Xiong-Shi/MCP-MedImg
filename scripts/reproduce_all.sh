#!/bin/bash
# MCP-MedImg: One-click full reproduction script
# Reproduces all experimental results from the paper

set -e

echo "=========================================="
echo "MCP-MedImg Full Reproduction Pipeline"
echo "=========================================="
echo ""

# Check prerequisites
echo "[1/4] Checking prerequisites..."
python --version || { echo "Python not found"; exit 1; }
pip --version || { echo "pip not found"; exit 1; }
docker --version || { echo "Docker not found (optional)"; }

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install -r requirements.txt

# Run unit tests
echo ""
echo "[3/4] Running unit tests (Table 16)..."
pytest tests/ -v --tb=short

# Run benchmarks
echo ""
echo "[4/4] Running benchmarks..."
echo ""
echo "  a) LIDC inference benchmark (Table 18)..."
bash scripts/reproduce_lidc_benchmark.sh || echo "  (Skipped - requires GPU + dataset)"
echo ""
echo "  b) Heart inference benchmark (Table 19, 22)..."
bash scripts/reproduce_heart_benchmark.sh || echo "  (Skipped - requires GPU + dataset + weights)"
echo ""
echo "  c) Protocol overhead decomposition (Table 20, 21)..."
bash scripts/reproduce_protocol_overhead.sh || echo "  (Skipped - requires Orthanc + dataset)"

echo ""
echo "=========================================="
echo "Reproduction complete!"
echo "Check docs/experiments/ for raw results."
echo "=========================================="
