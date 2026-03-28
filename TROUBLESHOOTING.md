# Troubleshooting

## No GPU detected (CPU-only mode)

wsl-benchmark always falls back to CPU if no GPU is found. This is expected.
To use a GPU:

```bash
# Windows (DirectML -- works with AMD, Intel, NVIDIA via DirectX 12):
pip install torch-directml
python -m wsl_benchmark

# Linux / WSL2 (ROCm for AMD):
# Install ROCm-enabled torch from https://rocm.docs.amd.com/
python -m wsl_benchmark

# NVIDIA (CUDA):
pip install torch  # standard PyPI torch includes CUDA support
python -m wsl_benchmark
```

## AMD RX 5700 XT not detected (gfx1010)

The RX 5700 XT uses gfx1010 architecture which is not officially supported by ROCm.
wsl-benchmark automatically sets HSA_OVERRIDE_GFX_VERSION=10.3.0 before importing torch.
If it still fails:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python -m wsl_benchmark
```

## Out of memory

Reduce matrix size:
```bash
python -m wsl_benchmark --size 256   # safe for 4GB VRAM
python -m wsl_benchmark --size 128   # safe for 2GB VRAM
```

## WSL2 disk full

If running inside WSL2 and you get disk errors, use wsl-disk-doctor:
https://github.com/ChharithOeun/wsl-disk-doctor

## Conv op fails on DirectML

DirectML has limited conv support. Use --ops matmul,fft,bandwidth to skip conv:
```bash
python -m wsl_benchmark --ops matmul,fft,bandwidth
```

## Installation fails (externally-managed-environment)

```bash
pip install wsl-benchmark --break-system-packages
# OR
python -m venv venv && source venv/bin/activate
pip install wsl-benchmark
```
