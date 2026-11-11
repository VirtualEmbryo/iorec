# Changelog

## Unreleased

Requires Python 3.12 or later again (1.4.0 required 3.13), so that packages tested on Python 3.12, such as delaunay-watershed and foambryo, can depend on iorec. No code change was needed: the source uses no construct newer than Python 3.10. The CI matrix runs Python 3.12, 3.13 and 3.14; ruff, black and mypy target Python 3.12.

Metadata appendix version 2, written by default: one `# key: value` line per key, the value in JSON (number, string, boolean, sequence or mapping, nested), read back with its type, arrays as tuples. Any key can now carry a typed value without iorec knowing it. Writing refuses NaN and infinite numbers, values JSON cannot hold (such as `None`) and keys with whitespace, `:` or `#`; non-ASCII strings are allowed. `appendix_version=1` on `save_rec`/`save_2d_rec`/`write_rec_appendix` writes version 1, byte for byte as before. iorec 1.1.0 to 1.4.0 read the geometry of version-2 files exactly but see none of their fields.

Reading version 1 is unchanged. `load_rec_appendix`, `load_rec_with_metadata` and `load_2d_rec_with_metadata` take `metadata_types=`, a key -> type map (`str`, `int`, `float`, `bool`, `tuple[X, ...]` or a function of the text) applied to version-1 files only, so a caller can type its own version-1 keys. `convert` keeps the appendix version of its source. An appendix of a version other than 1 or 2 now raises `ValueError` instead of being read as version 1.

## 1.4.0

Requires Python 3.13 or later. Project URLs now point to GitHub. Adds the `py.typed` marker, SPDX license headers, a CITATION.cff, a CONTRIBUTING guide and a CI workflow. Type annotations use the Python 3.13 syntax and `mypy --strict` passes.

## 1.3.0

Load and save .rec (3D, 2D), Surface Evolver, PLY, OBJ and 3MF meshes with two material labels per triangle; `iorec-convert` command; optional validation on read.

## 1.1.0

First release on PyPI.
