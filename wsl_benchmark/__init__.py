"""
wsl-benchmark -- GPU vs CPU performance benchmarking for PyTorch and JAX.

Cross-platform: Windows (native + WSL2), Linux, macOS.
Optimized for AMD RX 5700 XT (gfx1010) via DirectML and ROCm.
Degrades gracefully to CPU-only if no GPU is detected.

Quick start:
    python -m wsl_benchmark
    python -m wsl_benchmark --json
    python -m wsl_benchmark --ops matmul,conv --size 1024 --runs 5
"""

__version__ = "0.1.0"
__author__ = "ChharithOeun"
__license__ = "MIT"

from wsl_benchmark.runner import run_benchmark, get_results

__all__ = [
    "run_benchmark",
    "get_results",
    "__version__",
    "__author__",
]
