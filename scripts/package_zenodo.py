#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package zenodo_data into a clean zip for Zenodo upload."""

import os
import zipfile
import hashlib

SRC_DIR = r"D:\MCP-MedImg\zenodo_data"
OUTPUT = r"D:\MCP-MedImg\mcp-medimg-v2.0.0.zip"

EXCLUDE_DIRS = {"__pycache__", ".git", ".dumate"}
EXCLUDE_EXTS = {".pyc", ".tmp"}


def should_exclude(rel_path):
    parts = rel_path.split(os.sep)
    # Exclude any path containing __pycache__ or .git
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    # Exclude by extension
    _, ext = os.path.splitext(rel_path)
    if ext.lower() in EXCLUDE_EXTS:
        return True
    return False


count = 0
total_size = 0
with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SRC_DIR):
        # Remove excluded dirs from traversal
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, SRC_DIR)

            if should_exclude(rel_path):
                continue

            zf.write(full_path, rel_path)
            count += 1
            total_size += os.path.getsize(full_path)

# Calculate SHA256
sha = hashlib.sha256()
with open(OUTPUT, "rb") as f:
    while True:
        chunk = f.read(8192)
        if not chunk:
            break
        sha.update(chunk)

print(f"ZIP created: {OUTPUT}")
print(f"Files: {count}")
print(f"Total uncompressed size: {total_size / 1024 / 1024 / 1024:.2f} GB")
print(f"ZIP size: {os.path.getsize(OUTPUT) / 1024 / 1024 / 1024:.2f} GB")
print(f"SHA256: {sha.hexdigest()}")
