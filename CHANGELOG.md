# Changelog

## [Unreleased] - 2026-04-19

### Added

- feat: initial release wsl-benchmark v0.1.0

### Fixed

- fix: correct Buy Me a Coffee link to buymeacoffee.com/chharith
- fix: token scope + size limit (security vuln scan)
- fix: ci.yml -- replace broken inline python with pytest, fix install step

### Changed

- docs: add Contributing section to README
- docs: add Buy Me A Coffee link to README

### Other

- Add neon banner


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
