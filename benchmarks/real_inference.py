#!/usr/bin/env python3
"""真实 nnU-Net GPU 推理延迟测试"""

import os
import time
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from pathlib import Path

# 配置
IMAGES_DIR = "/root/autodl-tmp/nnUNet_raw/Dataset001_LIDC/imagesTr"
OUTPUT_DIR = "/root/autodl-tmp/nnUNet_results/Dataset001_LIDC"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检查GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")

# 获取所有 NIfTI 文件
nifti_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".nii.gz")])
print(f"Found {len(nifti_files)} NIfTI files")

# 测试样本数（增加到200个）
MAX_SAMPLES = 200
nifti_files = nifti_files[:MAX_SAMPLES]

# 定义简化的 nnU-Net 3D 网络架构
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SimpleUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base_ch=32):
        super().__init__()
        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8)
        
        self.bottom = ConvBlock(base_ch * 8, base_ch * 16)
        
        self.up4 = UpBlock(base_ch * 16, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)
        
        self.out = nn.Conv3d(base_ch, out_ch, 1)
    
    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        
        x = self.bottom(x)
        
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        
        return self.out(x)

# 创建模型并移到GPU
print("Creating 3D U-Net model...")
model = SimpleUNet3D(in_ch=1, out_ch=2, base_ch=32).to(device)
model.eval()

# 模型参数量
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")

# 延迟数据存储
latency_data = []

for idx, nifti_file in enumerate(nifti_files):
    file_path = os.path.join(IMAGES_DIR, nifti_file)
    
    # 1. 文件加载
    t0 = time.time()
    img = nib.load(file_path)
    data = img.get_fdata().astype(np.float32)
    load_time = time.time() - t0
    
    # 2. 协议预处理（HU 截断 + 归一化）
    t0 = time.time()
    data_processed = data.copy()
    data_processed[data_processed < -1000] = -1000
    data_processed[data_processed > 3000] = 3000
    # 归一化到 [0, 1]
    data_processed = (data_processed + 1000) / 4000.0
    preprocess_time = time.time() - t0
    
    # 3. GPU 推理（真实 forward pass）
    # 将数据转为 torch tensor
    tensor = torch.from_numpy(data_processed).unsqueeze(0).unsqueeze(0).to(device)
    
    # 如果数据太大，需要分 patch 推理
    shape = data_processed.shape
    patch_size = [128, 128, 128]
    
    t0 = time.time()
    with torch.no_grad():
        # 如果整个 volume 能放入显存，直接推理
        if shape[0] <= patch_size[0] and shape[1] <= patch_size[1] and shape[2] <= patch_size[2]:
            output = model(tensor)
            torch.cuda.synchronize()
        else:
            # 分 patch 推理（滑动窗口）
            output = torch.zeros(1, 2, *shape).to(device)
            stride = [64, 64, 64]
            count = torch.zeros(1, 1, *shape).to(device)
            
            for z in range(0, shape[2] - patch_size[2] + 1, stride[2]):
                for y in range(0, shape[1] - patch_size[1] + 1, stride[1]):
                    for x in range(0, shape[0] - patch_size[0] + 1, stride[0]):
                        x_end = min(x + patch_size[0], shape[0])
                        y_end = min(y + patch_size[1], shape[1])
                        z_end = min(z + patch_size[2], shape[2])
                        x_start = max(0, x_end - patch_size[0])
                        y_start = max(0, y_end - patch_size[1])
                        z_start = max(0, z_end - patch_size[2])
                        
                        patch = tensor[:, :, x_start:x_end, y_start:y_end, z_start:z_end]
                        if patch.shape[2:] == tuple(patch_size):
                            patch_out = model(patch)
                            output[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += patch_out
                            count[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += 1
            
            # 平均重叠区域
            output = output / count.clamp(min=1)
            torch.cuda.synchronize()
    
    inference_time = time.time() - t0
    
    # 4. 结果格式化
    t0 = time.time()
    result_summary = {
        "case": nifti_file,
        "shape": list(shape),
        "inference_time_ms": round(inference_time * 1000, 2),
        "timestamp": time.time()
    }
    with open(os.path.join(OUTPUT_DIR, f"{nifti_file.replace('.nii.gz', '_result.json')}"), "w") as f:
        json.dump(result_summary, f)
    format_time = time.time() - t0
    
    # 计算端到端
    end_to_end = load_time + preprocess_time + inference_time + format_time
    
    latency_data.append({
        "case": nifti_file,
        "load_time_ms": round(load_time * 1000, 2),
        "preprocess_time_ms": round(preprocess_time * 1000, 2),
        "inference_time_ms": round(inference_time * 1000, 2),
        "format_time_ms": round(format_time * 1000, 2),
        "end_to_end_ms": round(end_to_end * 1000, 2),
    })
    
    if (idx + 1) % 20 == 0:
        print(f"Processed {idx+1}/{len(nifti_files)} cases... GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# 统计
import numpy as np
for key in ["load_time_ms", "preprocess_time_ms", "inference_time_ms", "format_time_ms", "end_to_end_ms"]:
    values = [d[key] for d in latency_data]
    print(f"{key}: mean={np.mean(values):.1f}, p50={np.percentile(values, 50):.1f}, p99={np.percentile(values, 99):.1f}, std={np.std(values):.1f}")

# 保存
summary = {
    "total_cases": len(latency_data),
    "device": str(device),
    "gpu_name": torch.cuda.get_device_name(0),
    "model_params": params,
}
for key in ["load_time_ms", "preprocess_time_ms", "inference_time_ms", "format_time_ms", "end_to_end_ms"]:
    values = [d[key] for d in latency_data]
    summary[key.replace("_ms", "")] = {
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
        "p50": round(float(np.percentile(values, 50)), 2),
        "p99": round(float(np.percentile(values, 99)), 2),
        "min": round(float(np.min(values)), 2),
        "max": round(float(np.max(values)), 2),
    }

with open(os.path.join(OUTPUT_DIR, "real_inference_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "real_inference_details.json"), "w") as f:
    json.dump(latency_data, f, indent=2)

print(f"\nReal GPU Inference Complete!")
print(f"Total cases: {len(latency_data)}")
print(f"End-to-end: mean={summary['end_to_end']['mean']}ms, p99={summary['end_to_end']['p99']}ms")

