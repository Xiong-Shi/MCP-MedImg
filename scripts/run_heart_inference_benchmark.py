#!/usr/bin/env python3
"""SSH连接实例1并运行MSD Heart端到端推理延迟补测"""

import paramiko
import time
import json
import os

# SSH配置
HOST = "connect.westc.seetacloud.com"
PORT = 13162
USER = "root"
PASSWORD = "yJHEdcqLQ4FR"

REMOTE_WORKDIR = "/root/autodl-tmp/MCP-Medimg"
REMOTE_DATA = "/root/autodl-tmp/nnUNet_raw/Dataset002_Heart"
REMOTE_RESULTS = "/root/autodl-tmp/nnUNet_results/Dataset002_Heart"

print("=" * 60)
print("MSD Heart 端到端推理延迟补测")
print(f"目标: {HOST}:{PORT}")
print(f"工作目录: {REMOTE_WORKDIR}")
print("=" * 60)

# 连接SSH
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD)

# 检查环境
print("\n[1/5] 检查环境...")
stdin, stdout, stderr = client.exec_command(
    "python -c \"import nnunetv2; print('nnUNet OK')\" 2>&1"
)
print(stdout.read().decode().strip())

stdin, stdout, stderr = client.exec_command(
    f"ls {REMOTE_RESULTS}/nnUNetTrainer__nnUNetPlans__3d_fullres/ 2>&1"
)
result_folders = stdout.read().decode().strip()
print(f"结果文件夹: {result_folders[:200]}...")

stdin, stdout, stderr = client.exec_command(
    f"ls {REMOTE_DATA}/imagesTs/ 2>/dev/null | wc -l"
)
test_cases = int(stdout.read().decode().strip())
print(f"测试集案例数: {test_cases}")

stdin, stdout, stderr = client.exec_command(
    f"ls {REMOTE_DATA}/imagesTr/ 2>/dev/null | wc -l"
)
train_cases = int(stdout.read().decode().strip())
print(f"训练集案例数: {train_cases}")

# 编写推理脚本
script_content = '''#!/usr/bin/env python3
"""MSD Heart 端到端推理延迟测量"""

import os
import time
import json
import numpy as np
import torch
from pathlib import Path

os.environ["nnUNet_raw"] = "/root/autodl-tmp/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/root/autodl-tmp/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/root/autodl-tmp/nnUNet_results"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 初始化预测器（5-fold ensemble）
print("\\n[INIT] Loading 5-fold ensemble predictor...")
init_start = time.perf_counter()
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=device,
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)
predictor.initialize_from_trained_model_folder(
    "/root/autodl-tmp/nnUNet_results/Dataset002_Heart/nnUNetTrainer__nnUNetPlans__3d_fullres",
    use_folds=(0, 1, 2, 3, 4),
    checkpoint_name="checkpoint_final.pth"
)
init_time = (time.perf_counter() - init_start) * 1000
print(f"[INIT] Model loading time: {init_time:.2f} ms")

# 获取测试集文件
test_dir = Path("/root/autodl-tmp/nnUNet_raw/Dataset002_Heart/imagesTs")
test_files = sorted(test_dir.glob("*.nii.gz"))
print(f"\\nFound {len(test_files)} test files")

# 端到端推理
results = []
total_start = time.perf_counter()

for i, f in enumerate(test_files):
    case_id = f.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    print(f"\\n[{i+1}/{len(test_files)}] Processing {case_id}...")
    
    # Stage 1: File loading
    t0 = time.perf_counter()
    image, props = SimpleITKIO().read_images([str(f)])
    load_time = (time.perf_counter() - t0) * 1000
    
    # Stage 2: Preprocessing + Inference + Postprocessing
    t1 = time.perf_counter()
    predictor.predict_single_npy_array(
        input_image=image,
        image_properties=props,
        segmentation_previous_stage=None,
        output_file=str(f.parent.parent / "predictions" / f"{case_id}.nii.gz"),
        save_or_return_seg=True
    )
    inference_time = (time.perf_counter() - t1) * 1000
    
    # Stage 3: Result formatting (save to disk)
    t2 = time.perf_counter()
    output_path = f.parent.parent / "predictions" / f"{case_id}.nii.gz"
    format_time = (time.perf_counter() - t2) * 1000
    
    total_time = load_time + inference_time + format_time
    
    result = {
        "case_id": case_id,
        "load_time_ms": round(load_time, 2),
        "preprocess_inference_time_ms": round(inference_time, 2),
        "format_time_ms": round(format_time, 2),
        "end_to_end_ms": round(total_time, 2)
    }
    results.append(result)
    print(f"  Load: {load_time:.2f}ms | Inference: {inference_time:.2f}ms | Format: {format_time:.2f}ms | Total: {total_time:.2f}ms")

total_elapsed = (time.perf_counter() - total_start) * 1000

# 计算统计值
times = [r["end_to_end_ms"] for r in results]
load_times = [r["load_time_ms"] for r in results]
inf_times = [r["preprocess_inference_time_ms"] for r in results]
fmt_times = [r["format_time_ms"] for r in results]

summary = {
    "experiment": "EXP-HEART-003",
    "name": "MSD Heart End-to-End Inference Latency",
    "dataset": "MSD Task02 Heart",
    "cases": len(results),
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0),
    "model_loading_ms": round(init_time, 2),
    "total_elapsed_ms": round(total_elapsed, 2),
    "load_time_ms": {
        "mean": round(np.mean(load_times), 2),
        "std": round(np.std(load_times), 2),
        "p50": round(np.percentile(load_times, 50), 2),
        "p99": round(np.percentile(load_times, 99), 2),
        "min": round(np.min(load_times), 2),
        "max": round(np.max(load_times), 2)
    },
    "preprocess_inference_time_ms": {
        "mean": round(np.mean(inf_times), 2),
        "std": round(np.std(inf_times), 2),
        "p50": round(np.percentile(inf_times, 50), 2),
        "p99": round(np.percentile(inf_times, 99), 2),
        "min": round(np.min(inf_times), 2),
        "max": round(np.max(inf_times), 2)
    },
    "format_time_ms": {
        "mean": round(np.mean(fmt_times), 2),
        "std": round(np.std(fmt_times), 2),
        "p50": round(np.percentile(fmt_times, 50), 2),
        "p99": round(np.percentile(fmt_times, 99), 2),
        "min": round(np.min(fmt_times), 2),
        "max": round(np.max(fmt_times), 2)
    },
    "end_to_end_ms": {
        "mean": round(np.mean(times), 2),
        "std": round(np.std(times), 2),
        "p50": round(np.percentile(times, 50), 2),
        "p99": round(np.percentile(times, 99), 2),
        "min": round(np.min(times), 2),
        "max": round(np.max(times), 2)
    },
    "per_case_results": results
}

# 保存结果
out_dir = Path("/root/autodl-tmp/MCP-Medimg/benchmark_results")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "heart_real_inference_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Cases processed: {len(results)}")
print(f"Model loading: {init_time:.2f} ms")
print(f"End-to-end mean: {summary['end_to_end_ms']['mean']:.2f} ms")
print(f"End-to-end P50:  {summary['end_to_end_ms']['p50']:.2f} ms")
print(f"End-to-end P99:  {summary['end_to_end_ms']['p99']:.2f} ms")
print(f"Load mean:       {summary['load_time_ms']['mean']:.2f} ms")
print(f"Inference mean:  {summary['preprocess_inference_time_ms']['mean']:.2f} ms")
print(f"Format mean:     {summary['format_time_ms']['mean']:.2f} ms")
print(f"\\nSaved to: {out_dir / 'heart_real_inference_summary.json'}")
'''

# 写入脚本到远程
print("\n[2/5] 上传推理脚本...")
sftp = client.open_sftp()
script_path = f"{REMOTE_WORKDIR}/benchmark_heart_inference.py"
with sftp.file(script_path, "w") as f:
    f.write(script_content)
sftp.close()
print(f"Script written to {script_path}")

# 创建预测输出目录
stdin, stdout, stderr = client.exec_command(f"mkdir -p {REMOTE_DATA}/predictions")

# 运行推理
print("\n[3/5] 开始端到端推理测量...")
print("预计耗时: 20-40分钟 (20 cases × ~60s)")
print("=" * 60)

stdin, stdout, stderr = client.exec_command(
    f"cd {REMOTE_WORKDIR} && python benchmark_heart_inference.py 2>&1", timeout=3600
)

# 实时输出
for line in iter(stdout.readline, ""):
    print(line, end="")

err = stderr.read().decode()
if err:
    print(f"\\n[ERROR] {err}")

print("\n[4/5] 检查输出文件...")
stdin, stdout, stderr = client.exec_command(
    f"ls -lh {REMOTE_WORKDIR}/benchmark_results/heart_real_inference_summary.json 2>&1"
)
print(stdout.read().decode())

# 下载结果
print("\n[5/5] 下载结果到本地...")
sftp = client.open_sftp()
remote_file = f"{REMOTE_WORKDIR}/benchmark_results/heart_real_inference_summary.json"
local_file = "D:/MCP-Medimg/benchmarks/heart_real_inference_summary.json"
os.makedirs(os.path.dirname(local_file), exist_ok=True)
sftp.get(remote_file, local_file)
sftp.close()
print(f"Downloaded to: {local_file}")

client.close()
print("\n" + "=" * 60)
print("补测完成!")
print("=" * 60)
