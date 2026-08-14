#!/bin/bash
# Download pretrained model weights from Zenodo
# Replace ZENODO_RECORD_ID with actual record ID after upload

ZENODO_RECORD="10.5281/zenodo.xxxxxxx"  # TODO: Update after Zenodo upload

echo "Downloading pretrained model weights..."
echo "Zenodo DOI: ${ZENODO_RECORD}"
echo ""

mkdir -p models/weights

# Download LIDC model weights
echo "[1/2] LIDC model weights..."
wget -O models/weights/lidc_weights.tar.gz \
    "https://zenodo.org/record/xxxxxxx/files/lidc_weights.tar.gz" \
    || echo "LIDC weights not available yet (placeholder)"

# Download Heart 5-fold ensemble
echo "[2/2] Heart 5-fold ensemble weights..."
wget -O models/weights/heart_ensemble.tar.gz \
    "https://zenodo.org/record/xxxxxxx/files/heart_ensemble.tar.gz" \
    || echo "Heart weights not available yet (placeholder)"

# Extract if downloaded
if [ -f "models/weights/lidc_weights.tar.gz" ]; then
    tar xzf models/weights/lidc_weights.tar.gz -C models/
    rm models/weights/lidc_weights.tar.gz
fi

if [ -f "models/weights/heart_ensemble.tar.gz" ]; then
    tar xzf models/weights/heart_ensemble.tar.gz -C models/
    rm models/weights/heart_ensemble.tar.gz
fi

echo ""
echo "Download complete. Check models/ directory."
