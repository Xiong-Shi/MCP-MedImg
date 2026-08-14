#!/usr/bin/env python3
"""
MCP-MedImg Protocol Overhead Benchmark Runner (EXP-PROTO-001)

Orchestrates the 5-group comparative experiment and aggregates results.

Usage:
    python protocol_overhead_benchmark.py --data-dir /path/to/lidc/nifti --output-dir ./results [--model-path /path/to/model.pt] [--limit 200]

Groups:
    G0: baseline_raw        - Pure nnU-Net inference, no protocol
    G1: dicom_mcp_only      - L1 (Transport): DICOM query/fetch/convert
    G2: l1_l2              - L1 + L2 (Security): +PHI de-identify + audit log
    G3: l1_l2_l3           - L1 + L2 + L3 (Orchestration): +routing + DICOM SR
    G4: full_stack         - L1 + L2 + L3 + L4 (Semantics): +inspection + verification + state machine

Author: MCP-MedImg Team
Date: 2026-08-11
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time


# Experiment groups configuration
GROUPS = {
    "G0": {
        "name": "baseline_raw",
        "description": "Pure nnU-Net inference: direct NIfTI load -> ndarray -> GPU -> output. No DICOM, no network, no protocol.",
        "layers": [],
    },
    "G1": {
        "name": "dicom_mcp_only",
        "description": "L1 (Transport): DICOM query/fetch via dicom-mcp + conversion to ndarray -> inference.",
        "layers": ["L1"],
    },
    "G2": {
        "name": "l1_l2",
        "description": "L1 + L2 (Security): G1 + PHI de-identification + audit logging (start/complete).",
        "layers": ["L1", "L2"],
    },
    "G3": {
        "name": "l1_l2_l3",
        "description": "L1 + L2 + L3 (Orchestration): G2 + ModelRegistry routing + ResultFormatter + DICOM SR generation.",
        "layers": ["L1", "L2", "L3"],
    },
    "G4": {
        "name": "full_stack",
        "description": "Full stack (L1+L2+L3+L4): Complete MCP-MedImg protocol with InspectionRouter + VerificationRouter + state machine.",
        "layers": ["L1", "L2", "L3", "L4"],
    },
}


def run_group(
    group: str,
    data_dir: Path,
    model_path: Optional[Path],
    output_dir: Path,
    limit: Optional[int] = None,
    probe_script: Path = Path("protocol_overhead_probe.py"),
) -> Path:
    """Run a single experimental group."""
    
    print(f"\n{'='*60}")
    print(f"Running Group {group}: {GROUPS[group]['name']}")
    print(f"Description: {GROUPS[group]['description']}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        sys.executable,
        str(probe_script),
        "--group", group,
        "--data-dir", str(data_dir),
        "--output", str(output_dir),
    ]
    
    if model_path:
        cmd.extend(["--model-path", str(model_path)])
    if limit:
        cmd.extend(["--limit", str(limit)])
        
    # Run probe
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 4,  # 4 hour timeout per group
        )
        elapsed = time.time() - start_time
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: Group {group} failed with code {result.returncode}")
            print(result.stderr)
            return None
            
        print(f"Group {group} completed in {elapsed:.1f}s")
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Group {group} timed out after 4 hours")
        return None
    except Exception as e:
        print(f"ERROR: Group {group} failed: {e}")
        return None
        
    # Find output file
    output_file = output_dir / f"probe_results_{group}.json"
    if output_file.exists():
        print(f"Results saved to: {output_file}")
        return output_file
    else:
        print(f"WARNING: Expected output file not found: {output_file}")
        return None


def aggregate_results(
    group_results: Dict[str, Path],
    output_dir: Path,
) -> Path:
    """Aggregate all group results into a single comparison report."""
    
    print(f"\n{'='*60}")
    print("Aggregating results across all groups...")
    print(f"{'='*60}\n")
    
    all_data = {}
    
    for group, path in group_results.items():
        if path is None:
            print(f"WARNING: No results for group {group}, skipping")
            continue
            
        with open(path) as f:
            data = json.load(f)
            
        all_data[group] = {
            "config": data["config"],
            "summary": data["summary"],
            "timestamp": data["timestamp"],
        }
        
    # Compute protocol overhead comparison
    comparison = compute_overhead_comparison(all_data)
    
    # Build final report
    report = {
        "experiment": "EXP-PROTO-001",
        "title": "MCP-MedImg Protocol Overhead Decomposition",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": GROUPS,
        "data": all_data,
        "comparison": comparison,
    }
    
    # Save report
    report_path = output_dir / "protocol_overhead_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Also generate markdown summary
    md_path = output_dir / "protocol_overhead_report.md"
    generate_markdown_report(report, md_path)
    
    print(f"\nAggregation complete!")
    print(f"JSON report: {report_path}")
    print(f"Markdown report: {md_path}")
    
    return report_path


def compute_overhead_comparison(all_data: Dict) -> Dict:
    """Compute protocol overhead comparison across groups."""
    
    comparison = {}
    
    # Extract baseline (G0) if available
    g0_summary = all_data.get("G0", {}).get("summary", {})
    g0_mean = g0_summary.get("baseline_total_ms", {}).get("mean", 0)
    
    for group in ["G1", "G2", "G3", "G4"]:
        if group not in all_data:
            continue
            
        summary = all_data[group]["summary"]
        
        # Extract total latency for this group
        # For G1: l1_total_ms + baseline_inference_ms (approximate)
        # For G2+: similar
        total_key = f"l{len(GROUPS[group]['layers'])}_total_ms"
        group_total = summary.get("protocol_overhead_ms", {}).get("mean", 0)
        
        # If baseline available, compute overhead
        if g0_mean > 0:
            overhead_ms = group_total
            overhead_pct = (overhead_ms / g0_mean) * 100
            
            comparison[group] = {
                "absolute_overhead_ms": overhead_ms,
                "relative_overhead_pct": overhead_pct,
                "baseline_ms": g0_mean,
                "group_total_ms": group_total,
            }
            
    return comparison


def generate_markdown_report(report: Dict, output_path: Path):
    """Generate human-readable markdown report."""
    
    with open(output_path, 'w') as f:
        f.write("# MCP-MedImg Protocol Overhead Decomposition Report\n\n")
        f.write(f"**Experiment**: EXP-PROTO-001  \n")
        f.write(f"**Date**: {report['timestamp']}  \n")
        f.write(f"**Dataset**: LIDC-IDRI (200 CT volumes)  \n\n")
        
        # Group descriptions
        f.write("## Experimental Groups\n\n")
        for group, config in GROUPS.items():
            f.write(f"### {group}: {config['name']}\n")
            f.write(f"- **Layers**: {', '.join(config['layers']) if config['layers'] else 'None'}\n")
            f.write(f"- **Description**: {config['description']}\n\n")
            
        # Results summary
        f.write("## Results Summary\n\n")
        f.write("| Group | Layers | Mean Latency (ms) | Protocol Overhead (ms) | Overhead % |\n")
        f.write("|-------|--------|-------------------|------------------------|------------|\n")
        
        for group in ["G0", "G1", "G2", "G3", "G4"]:
            if group not in report["data"]:
                continue
                
            summary = report["data"][group]["summary"]
            layers = GROUPS[group]["layers"]
            
            if group == "G0":
                mean_latency = summary.get("baseline_total_ms", {}).get("mean", 0)
                f.write(f"| {group} | Baseline | {mean_latency:.1f} | 0 | 0% |\n")
            else:
                mean_overhead = summary.get("protocol_overhead_ms", {}).get("mean", 0)
                mean_pct = summary.get("protocol_overhead_pct", {}).get("mean", 0)
                
                # Estimate total latency (baseline + overhead)
                g0_mean = report["data"].get("G0", {}).get("summary", {}).get("baseline_total_ms", {}).get("mean", 0)
                total_latency = g0_mean + mean_overhead
                
                f.write(f"| {group} | {', '.join(layers)} | {total_latency:.1f} | {mean_overhead:.1f} | {mean_pct:.2f}% |\n")
                
        # Detailed per-layer breakdown
        f.write("\n## Per-Layer Breakdown\n\n")
        f.write("*Based on G4 (full stack) results*\n\n")
        
        if "G4" in report["data"]:
            g4_summary = report["data"]["G4"]["summary"]
            
            layer_times = [
                ("L1 (Transport)", "l1_total_ms"),
                ("L2 (Security)", "l2_total_ms"),
                ("L3 (Orchestration)", "l3_total_ms"),
                ("L4 (Semantics)", "l4_total_ms"),
            ]
            
            f.write("| Layer | Mean (ms) | Median (ms) | P99 (ms) | % of Total |\n")
            f.write("|-------|-----------|-------------|----------|------------|\n")
            
            total_protocol = 0
            for layer_name, key in layer_times:
                stats = g4_summary.get(key, {})
                mean = stats.get("mean", 0)
                median = stats.get("median", 0)
                p99 = stats.get("p99", 0)
                total_protocol += mean
                
                f.write(f"| {layer_name} | {mean:.2f} | {median:.2f} | {p99:.2f} | - |\n")
                
        # Key findings
        f.write("\n## Key Findings\n\n")
        
        comparison = report.get("comparison", {})
        if comparison:
            f.write("### Protocol Overhead Summary\n\n")
            for group, data in comparison.items():
                f.write(f"- **{group}** ({GROUPS[group]['name']}): ")
                f.write(f"{data['relative_overhead_pct']:.2f}% overhead ")
                f.write(f"({data['absolute_overhead_ms']:.1f} ms)\n")
                
        f.write("\n## Conclusion\n\n")
        f.write("The protocol overhead decomposition experiment quantifies the latency contribution ")
        f.write("of each layer in the MCP-MedImg protocol stack. Results demonstrate that...\n")
        f.write("*(To be completed after data analysis)*\n")


def main():
    parser = argparse.ArgumentParser(description="MCP-MedImg Protocol Overhead Benchmark")
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Directory containing LIDC-IDRI NIfTI data")
    parser.add_argument("--model-path", type=Path, default=None,
                       help="Path to pre-trained model (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path("./protocol_overhead_results"),
                       help="Output directory for all results")
    parser.add_argument("--limit", type=int, default=200,
                       help="Number of volumes to process per group (default: 200)")
    parser.add_argument("--groups", nargs="+", choices=["G0", "G1", "G2", "G3", "G4"],
                       default=["G0", "G1", "G2", "G3", "G4"],
                       help="Groups to run (default: all)")
    parser.add_argument("--probe-script", type=Path, default=Path("protocol_overhead_probe.py"),
                       help="Path to probe script")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip groups that already have result files")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_dir.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)
        
    if not args.probe_script.exists():
        print(f"ERROR: Probe script not found: {args.probe_script}")
        sys.exit(1)
        
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"MCP-MedImg Protocol Overhead Benchmark")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Volumes per group: {args.limit}")
    print(f"Groups: {', '.join(args.groups)}")
    print(f"{'='*60}\n")
    
    # Run each group
    group_results = {}
    
    for group in args.groups:
        # Check if result already exists
        result_file = args.output_dir / f"probe_results_{group}.json"
        if args.skip_existing and result_file.exists():
            print(f"Skipping {group} (result exists: {result_file})")
            group_results[group] = result_file
            continue
            
        result = run_group(
            group=group,
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            limit=args.limit,
            probe_script=args.probe_script,
        )
        group_results[group] = result
        
    # Aggregate results
    successful_groups = {k: v for k, v in group_results.items() if v is not None}
    
    if successful_groups:
        report_path = aggregate_results(successful_groups, args.output_dir)
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Report: {report_path}")
        print(f"Successful groups: {len(successful_groups)}/{len(args.groups)}")
    else:
        print("\nNo groups completed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
