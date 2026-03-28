"""
GPU detection helper. Uses gpu_doctor if installed, else minimal detection.
"""

import os
import sys


def get_device_info():
    """
    Return a dict describing the best available compute device.
    Tries gpu_doctor first, then falls back to minimal detection.
    """
    try:
        import gpu_doctor
        device = gpu_doctor.get_best_device()
        return {"source": "gpu_doctor", "device": str(device)}
    except ImportError:
        pass
    # Minimal fallback
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return {"source": "torch.cuda", "device": "cuda", "name": name}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"source": "torch.mps", "device": "mps", "name": "Apple MPS"}
    except ImportError:
        pass
    try:
        import torch_directml
        dml = torch_directml.device()
        return {"source": "torch_directml", "device": str(dml), "name": "DirectML"}
    except ImportError:
        pass
    return {"source": "cpu_fallback", "device": "cpu", "name": "CPU"}
