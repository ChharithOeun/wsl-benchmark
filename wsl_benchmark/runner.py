"""
Core benchmark engine for wsl-benchmark.

IMPORTANT: HSA_OVERRIDE_GFX_VERSION is set before any torch import.
This is required for AMD gfx1010 (RX 5700 XT) on ROCm.
"""

import os
import sys
import time
import statistics
import platform

# -- AMD ROCm env fix BEFORE torch import ------------------------------------
# gfx1010 (RX 5700 XT, RX 5700, RX 5600 XT) is not officially supported.
# HSA_OVERRIDE_GFX_VERSION=10.3.0 makes ROCm treat it as gfx1030.
# Must be set BEFORE importing torch or jax.
if not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# -- Try importing torch (optional) ------------------------------------------
try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_OK = False

# -- Try importing numpy (always available as fallback) ----------------------
try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    np = None  # type: ignore
    _NUMPY_OK = False


def _detect_device():
    """Return the best available torch device string, or 'cpu' if torch not available."""
    if not _TORCH_OK:
        return "cpu"
    # Try DirectML (Windows AMD/Intel/NVIDIA via DirectX 12)
    try:
        import torch_directml
        return str(torch_directml.device())
    except (ImportError, Exception):
        pass
    # Try ROCm / CUDA
    if torch.cuda.is_available():
        return "cuda"
    # Try MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _time_op(fn, warmup=3, runs=10):
    """
    Time a callable fn() and return median + stddev in milliseconds.
    Returns (median_ms, stddev_ms, timings_list).
    """
    # Warmup
    for _ in range(warmup):
        fn()
    # Timed runs
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    if len(timings) == 1:
        return timings[0], 0.0, timings
    return statistics.median(timings), statistics.stdev(timings), timings


def _sync_device(device_str):
    """Synchronize GPU after each op for accurate timing."""
    if not _TORCH_OK:
        return
    if "cuda" in device_str or "privateuseone" in device_str:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _benchmark_matmul(size, device_str, warmup, runs):
    """NxN matrix multiply benchmark."""
    if _TORCH_OK:
        try:
            a = torch.randn(size, size, device=device_str)
            b = torch.randn(size, size, device=device_str)
            def fn():
                _ = torch.mm(a, b)
                _sync_device(device_str)
            med, std, _ = _time_op(fn, warmup, runs)
            return {"op": "matmul", "backend": "torch", "device": device_str,
                    "size": f"{size}x{size}", "median_ms": round(med, 3),
                    "stddev_ms": round(std, 3), "error": None}
        except Exception as exc:
            pass
    if _NUMPY_OK:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        def fn():
            np.dot(a, b)
        med, std, _ = _time_op(fn, warmup, runs)
        return {"op": "matmul", "backend": "numpy", "device": "cpu",
                "size": f"{size}x{size}", "median_ms": round(med, 3),
                "stddev_ms": round(std, 3), "error": None}
    return {"op": "matmul", "backend": "none", "device": "none",
            "size": f"{size}x{size}", "median_ms": None,
            "stddev_ms": None, "error": "no backend available"}


def _benchmark_conv(size, device_str, warmup, runs):
    """2D convolution benchmark (batch=8, channels=64, kernel=3)."""
    if not _TORCH_OK:
        return {"op": "conv", "backend": "none", "device": "none",
                "size": f"8x64x{size}x{size}", "median_ms": None,
                "stddev_ms": None, "error": "torch not installed"}
    try:
        N = min(size, 256)  # cap spatial dim for conv to avoid OOM
        x = torch.randn(8, 64, N, N, device=device_str)
        w = torch.randn(64, 64, 3, 3, device=device_str)
        import torch.nn.functional as F
        def fn():
            _ = F.conv2d(x, w, padding=1)
            _sync_device(device_str)
        med, std, _ = _time_op(fn, warmup, runs)
        return {"op": "conv", "backend": "torch", "device": device_str,
                "size": f"8x64x{N}x{N}", "median_ms": round(med, 3),
                "stddev_ms": round(std, 3), "error": None}
    except Exception as exc:
        return {"op": "conv", "backend": "torch", "device": device_str,
                "size": f"8x64x{size}x{size}", "median_ms": None,
                "stddev_ms": None, "error": str(exc)[:100]}


def _benchmark_fft(size, device_str, warmup, runs):
    """2D FFT benchmark."""
    if _TORCH_OK:
        try:
            x = torch.randn(size, size, device=device_str)
            def fn():
                _ = torch.fft.fft2(x)
                _sync_device(device_str)
            med, std, _ = _time_op(fn, warmup, runs)
            return {"op": "fft", "backend": "torch", "device": device_str,
                    "size": f"{size}x{size}", "median_ms": round(med, 3),
                    "stddev_ms": round(std, 3), "error": None}
        except Exception as exc:
            pass
    if _NUMPY_OK:
        x = np.random.randn(size, size).astype(np.float32)
        def fn():
            np.fft.fft2(x)
        med, std, _ = _time_op(fn, warmup, runs)
        return {"op": "fft", "backend": "numpy", "device": "cpu",
                "size": f"{size}x{size}", "median_ms": round(med, 3),
                "stddev_ms": round(std, 3), "error": None}
    return {"op": "fft", "backend": "none", "device": "none",
            "size": f"{size}x{size}", "median_ms": None,
            "stddev_ms": None, "error": "no backend"}


def _benchmark_bandwidth(size, device_str, warmup, runs):
    """Memory bandwidth: copy large tensor (4*size^2 bytes)."""
    if not _TORCH_OK:
        return {"op": "bandwidth", "backend": "none", "device": "none",
                "size": f"{size}x{size}", "median_ms": None,
                "stddev_ms": None, "error": "torch not installed"}
    try:
        src = torch.randn(size, size, device=device_str)
        def fn():
            _ = src.clone()
            _sync_device(device_str)
        med, std, _ = _time_op(fn, warmup, runs)
        bytes_copied = src.nelement() * src.element_size()
        gb_per_s = (bytes_copied / 1e9) / (med / 1000.0) if med > 0 else 0
        return {"op": "bandwidth", "backend": "torch", "device": device_str,
                "size": f"{size}x{size}", "median_ms": round(med, 3),
                "stddev_ms": round(std, 3),
                "gb_per_s": round(gb_per_s, 2), "error": None}
    except Exception as exc:
        return {"op": "bandwidth", "backend": "torch", "device": device_str,
                "size": f"{size}x{size}", "median_ms": None,
                "stddev_ms": None, "error": str(exc)[:100]}


_OP_MAP = {
    "matmul": _benchmark_matmul,
    "conv": _benchmark_conv,
    "fft": _benchmark_fft,
    "bandwidth": _benchmark_bandwidth,
}


def run_benchmark(ops=None, size=1024, warmup=3, runs=10):
    """
    Run the specified benchmark ops.

    Args:
        ops: list of op names (matmul, conv, fft, bandwidth). Default: all.
        size: NxN dimension for matrix ops. Default: 1024.
        warmup: warmup iterations before timing. Default: 3.
        runs: timed iterations per op. Default: 10.

    Returns:
        dict with keys: device, backend, platform, torch_version, results (list)
    """
    if ops is None:
        ops = ["matmul", "conv", "fft", "bandwidth"]

    device_str = _detect_device()
    torch_ver = torch.__version__ if _TORCH_OK else None
    numpy_ver = np.__version__ if _NUMPY_OK else None

    results_list = []
    for op in ops:
        fn = _OP_MAP.get(op)
        if fn is None:
            continue
        result = fn(size, device_str, warmup, runs)
        results_list.append(result)

    return {
        "device": device_str,
        "torch_version": torch_ver,
        "numpy_version": numpy_ver,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "size": size,
        "warmup": warmup,
        "runs": runs,
        "results": results_list,
    }


def get_results(ops=None, size=1024, warmup=3, runs=10):
    """Run benchmark and return formatted table string."""
    from wsl_benchmark.report import format_table
    data = run_benchmark(ops=ops, size=size, warmup=warmup, runs=runs)
    return format_table(data)
