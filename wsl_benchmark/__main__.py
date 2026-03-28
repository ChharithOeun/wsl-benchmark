"""
CLI entry point: python -m wsl_benchmark

Examples:
    python -m wsl_benchmark
    python -m wsl_benchmark --json
    python -m wsl_benchmark --ops matmul,conv --size 1024 --runs 5
    python -m wsl_benchmark --warmup 2 --runs 20 --json > results.json
    wsl-benchmark  (if installed via pip)
"""

import argparse
import json
import sys

from wsl_benchmark.runner import run_benchmark
from wsl_benchmark.report import format_table, format_json


def main():
    parser = argparse.ArgumentParser(
        prog="wsl-benchmark",
        description=(
            "Benchmark GPU vs CPU performance for PyTorch/JAX workloads. "
            "Supports DirectML (Windows), ROCm (Linux/WSL2), CUDA, MPS, CPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  wsl-benchmark\n"
            "  wsl-benchmark --json\n"
            "  wsl-benchmark --ops matmul,conv --size 1024\n"
            "  wsl-benchmark --warmup 2 --runs 20 --json > results.json\n"
        ),
    )
    parser.add_argument(
        "--ops",
        default="matmul,conv,fft,bandwidth",
        help="Comma-separated ops to benchmark (default: matmul,conv,fft,bandwidth)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Matrix/tensor dimension N for NxN ops (default: 1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before timing (default: 3)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Timed iterations per op (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of table",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    # Validate args
    try:
        ops = [op.strip() for op in args.ops.split(",") if op.strip()]
        if not ops:
            print("[ERROR] --ops cannot be empty", file=sys.stderr)
            sys.exit(1)
        valid_ops = {"matmul", "conv", "fft", "bandwidth"}
        bad = [op for op in ops if op not in valid_ops]
        if bad:
            print(f"[ERROR] Unknown ops: {bad}. Valid: {sorted(valid_ops)}", file=sys.stderr)
            sys.exit(1)
        if args.size < 64 or args.size > 8192:
            print("[ERROR] --size must be between 64 and 8192", file=sys.stderr)
            sys.exit(1)
        if args.warmup < 0 or args.warmup > 100:
            print("[ERROR] --warmup must be between 0 and 100", file=sys.stderr)
            sys.exit(1)
        if args.runs < 1 or args.runs > 1000:
            print("[ERROR] --runs must be between 1 and 1000", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Invalid arguments: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        results = run_benchmark(
            ops=ops,
            size=args.size,
            warmup=args.warmup,
            runs=args.runs,
        )
    except Exception as exc:
        print(f"[ERROR] Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        print(format_json(results))
    else:
        print(format_table(results))

    sys.exit(0)


if __name__ == "__main__":
    main()
