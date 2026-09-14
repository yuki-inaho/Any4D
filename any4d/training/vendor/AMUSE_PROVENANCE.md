# Vendored AMUSE

- Repository: https://github.com/kjeiun/amuse
- Revision: `48922743b32f33f919ab54edde3dbad0d0ce2dc7`
- Source: `src/optim/AMUSE.py`
- Source SHA256: `84fd3fbbc99e1718cf1c821ceff3369439f48e6fbd8ecc3a2b83afa5d82eea1f`
- License: Apache License 2.0, copied unchanged to `AMUSE_LICENSE` from the same
  revision (`LICENSE` in the repository root).
- Vendored: 2026-09-14 by downloading the raw file from the pinned revision.
- Modifications: none. `any4d/training/vendor/amuse.py` is a byte-identical
  copy; `verify_vendored_amuse()` in `any4d/training/geometry_runtime.py`
  re-checks the digest before the optimizer is constructed.
