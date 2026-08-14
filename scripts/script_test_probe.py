#!/usr/bin/env python3
"""Quick test of probe module."""
import sys
sys.path.insert(0, '/root/autodl-tmp/MCP-Medimg/scripts')

# Test import
from protocol_overhead_probe import ProtocolOverheadProbe, ProbeResult
print("Probe module imported successfully")

# Test data creation
import nibabel as nib
import numpy as np
import os

os.makedirs('/tmp/test_data', exist_ok=True)
for i in range(3):
    img = nib.Nifti1Image(np.random.rand(64, 64, 64).astype(np.float32), np.eye(4))
    nib.save(img, f'/tmp/test_data/test_{i:03d}.nii.gz')
print(f"Created 3 test NIfTI files")

# Test G0 run
probe = ProtocolOverheadProbe(
    group="G0",
    data_dir="/tmp/test_data",
    output_dir="/tmp/test_results"
)
results = probe.run_experiment(limit=2)
print(f"G0 test completed: {len(results)} results")
if results:
    print(f"First result: load={results[0].baseline_load_ms:.2f}ms, inference={results[0].baseline_inference_ms:.2f}ms")

# Test save
output = probe.save_results(results)
print(f"Results saved to: {output}")

# Verify JSON
import json
with open(output) as f:
    data = json.load(f)
print(f"Summary: {data['summary']}")
print("All tests passed!")
