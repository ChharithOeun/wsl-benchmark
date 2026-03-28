"""
Format benchmark results as table or JSON.
"""

import json as _json


def format_table(data):
    """Return a human-readable ASCII table of benchmark results."""
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("  WSL-BENCHMARK RESULTS")
    lines.append("=" * 64)
    lines.append(f"  Device  : {data.get('device', 'unknown')}")
    lines.append(f"  Platform: {data.get('platform', 'unknown')}")
    lines.append(f"  Python  : {data.get('python_version', 'unknown')}")
    tv = data.get("torch_version")
    lines.append(f"  PyTorch : {tv if tv else 'not installed'}")
    lines.append(f"  Size    : {data.get('size', '?')}x{data.get('size', '?')}")
    lines.append(f"  Warmup  : {data.get('warmup', '?')}  Runs: {data.get('runs', '?')}")
    lines.append("-" * 64)
    header = f"  {'Op':<12} {'Backend':<8} {'Size':<16} {'Median ms':>10} {'Stddev ms':>10}"
    lines.append(header)
    lines.append("-" * 64)
    for r in data.get("results", []):
        if r.get("error") and r.get("median_ms") is None:
            lines.append(f"  {r['op']:<12} {'ERROR':<8} {r.get('size','?'):<16} {'N/A':>10} {'N/A':>10}  [{r['error'][:30]}]")
        else:
            med = f"{r['median_ms']:.3f}" if r.get("median_ms") is not None else "N/A"
            std = f"{r['stddev_ms']:.3f}" if r.get("stddev_ms") is not None else "N/A"
            row = f"  {r['op']:<12} {r.get('backend','?'):<8} {r.get('size','?'):<16} {med:>10} {std:>10}"
            if "gb_per_s" in r and r["gb_per_s"]:
                row += f"  ({r['gb_per_s']} GB/s)"
            lines.append(row)
    lines.append("=" * 64)
    lines.append("")
    return "\n".join(lines)


def format_json(data):
    """Return results as pretty-printed JSON string."""
    return _json.dumps(data, indent=2)
