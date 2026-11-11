# Changelog

## Unreleased

Requires Python 3.12 or later again (1.4.0 required 3.13), so that packages tested on Python 3.12, such as delaunay-watershed and foambryo, can depend on iorec. No code change was needed: the source uses no construct newer than Python 3.10. The CI matrix runs Python 3.12, 3.13 and 3.14; ruff, black and mypy target Python 3.12.

## 1.4.0

Requires Python 3.13 or later. Project URLs now point to GitHub. Adds the `py.typed` marker, SPDX license headers, a CITATION.cff, a CONTRIBUTING guide and a CI workflow. Type annotations use the Python 3.13 syntax and `mypy --strict` passes.

## 1.3.0

Load and save .rec (3D, 2D), Surface Evolver, PLY, OBJ and 3MF meshes with two material labels per triangle; `iorec-convert` command; optional validation on read.

## 1.1.0

First release on PyPI.
