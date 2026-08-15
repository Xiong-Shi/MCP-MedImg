#!/usr/bin/env python
"""SSH into AutoDL instance2 and export LIDC model weights."""

import paramiko
import os
import stat
from pathlib import Path

HOST = "connect.westc.seetacloud.com"
PORT = 56436
USER = "root"
PWD = "BfXZj7Pg6JMq"
LOCAL_OUT = Path("D:/MCP-MedImg/models/instance2_backup")
LOCAL_OUT.mkdir(parents=True, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {HOST}:{PORT} ...")
ssh.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
print("Connected.")

sftp = ssh.open_sftp()

# Step 1: Find all .pth / .pt / checkpoint files related to LIDC
SEARCH_DIRS = [
    "/root/autodl-tmp",
    "/root",
    "/root/autodl-tmp/MCP-Medimg",
]

weights_found = []

for d in SEARCH_DIRS:
    try:
        stdin, stdout, stderr = ssh.exec_command(
            f"find {d} -type f \\( -name '*.pth' -o -name '*.pt' -o -name '*checkpoint*' -o -name '*model*' \\) 2>/dev/null"
        )
        for line in stdout:
            line = line.strip()
            if line:
                weights_found.append(line)
    except Exception as e:
        print(f"  Search error in {d}: {e}")

print(f"\nFound {len(weights_found)} potential weight/model files:")
for i, f in enumerate(weights_found[:50], 1):
    print(f"  {i}. {f}")
if len(weights_found) > 50:
    print(f"  ... and {len(weights_found) - 50} more")

# Step 2: Also check for any LIDC-specific directories
print("\n--- Checking LIDC-related directories ---")
stdin, stdout, stderr = ssh.exec_command(
    "find /root/autodl-tmp -type d -name '*LIDC*' 2>/dev/null"
)
lidc_dirs = [l.strip() for l in stdout if l.strip()]
print(f"LIDC dirs found: {lidc_dirs}")

# Step 3: Check for nnUNet results directories
stdin, stdout, stderr = ssh.exec_command(
    "find /root/autodl-tmp -type d -name 'nnUNet_results' 2>/dev/null"
)
nnunet_dirs = [l.strip() for l in stdout if l.strip()]
print(f"nnUNet results dirs: {nnunet_dirs}")

for d in nnunet_dirs:
    stdin, stdout, stderr = ssh.exec_command(f"ls -laR {d}")
    print(f"\n--- Contents of {d} ---")
    for line in stdout:
        print(line.strip())

# Step 4: Check for any training logs that indicate model save location
print("\n--- Checking training logs ---")
stdin, stdout, stderr = ssh.exec_command(
    "find /root/autodl-tmp -name '*train*log*' -o -name '*training*.log' 2>/dev/null"
)
logs = [l.strip() for l in stdout if l.strip()]
for log in logs[:10]:
    print(f"  Log: {log}")

# Step 5: Check for any saved model info or JSON files in benchmarks
stdin, stdout, stderr = ssh.exec_command(
    "find /root/autodl-tmp -name '*.json' | head -20"
)
jsons = [l.strip() for l in stdout if l.strip()]
print(f"\nJSON files found ({len(jsons)}):")
for j in jsons[:20]:
    print(f"  {j}")

# Step 6: If weights exist, download them
if weights_found:
    print(f"\n--- Downloading {len(weights_found)} files ---")
    for remote_path in weights_found:
        try:
            filename = os.path.basename(remote_path)
            local_path = LOCAL_OUT / filename
            # Handle duplicate filenames by adding prefix
            if local_path.exists():
                dirname = (
                    os.path.dirname(remote_path).replace("/", "_").replace("\\", "_")
                )
                local_path = LOCAL_OUT / f"{dirname}_{filename}"

            size = sftp.stat(remote_path).st_size
            print(f"  {remote_path} -> {local_path} ({size / 1024 / 1024:.1f} MB)")
            sftp.get(remote_path, str(local_path))
        except Exception as e:
            print(f"    ERROR: {e}")
else:
    print("\nNo weight files found on instance2.")

sftp.close()
ssh.close()
print("\nDone.")
