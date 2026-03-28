# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.0] - 2026-03-28

### Added

- Initial release of wsl-benchmark
- Core benchmark engine: matmul, conv, fft, bandwidth ops
- Auto-detects best device: DirectML, ROCm/CUDA, MPS, CPU
- Sets HSA_OVERRIDE_GFX_VERSION=10.3.0 before torch import for AMD gfx1010 (RX 5700 XT)
- Graceful CPU-only fallback when no GPU detected
- JSON output mode (--json) for scripting and CI integration
- Cross-platform CI: Ubuntu/Windows/macOS x Python 3.9-3.12
- AI issue responder: 6 pattern categories
- Auto-changelog workflow
- PyPI publish workflow (on GitHub Release)

[0.1.0]: https://github.com/ChharithOeun/wsl-benchmark/releases/tag/v0.1.0
