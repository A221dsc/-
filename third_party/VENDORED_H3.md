# Vendored H3 C library

- Version: **4.1.0**
- Source: PyPI source distribution `h3-4.1.0.tar.gz`
  (`https://pypi.org/project/h3/4.1.0/`), which includes the H3 C core under
  `src/h3lib`.
- SHA-256 of the downloaded sdist:
  `45455d284fafc38716ad9480cc37004def78eac14c3dccee24122377ebbf36e9`
- Vendored subset: `CMakeLists.txt`, `cmake/`, `LICENSE`, `README.md`,
  `VERSION`, `src/h3lib/include/`, `src/h3lib/lib/`.
- License: Apache License 2.0 (see `h3-4.1.0/LICENSE`).
- The upstream CMake project was not used directly on this host because the
  repository path contains Chinese characters and MinGW Make cannot stat the
  generated absolute source prerequisites. `scripts/build_h3_p0.sh` compiles
  the same C sources with relative paths. The upstream CMake files are kept so
  the sources can be built normally on an ASCII path or on Linux/macOS.

The H3 version is pinned deliberately. All H3-P0 tests and diagnostics must use
this tree; do not mix a system H3 installation into the same run.
