# FrogML

FrogML is a small Python package with a CUDA extension.

## Development setup

The build needs a CUDA 12.0 or newer toolkit and a host C++ compiler supported
by the installed CUDA toolkit. The project pins CMake 3.28.3 and will use that
version even if your system CMake differs. On Linux, CUDA 12.0 supports GCC 12;
use a compatible compiler version if you have a different toolkit. These
commands select GCC 12:

```sh
export CC=gcc-12
export CXX=g++-12
uv sync --locked
```

`uv sync` creates the Python 3.12.3 environment, installs the locked Python
dependencies, and builds FrogML in editable mode. To check the package import:

```sh
uv run --locked python -c 'import frogml; print(frogml.add_scalar)'
```

Build a wheel with:

```sh
uv build --wheel
```

The default build includes CUDA targets 70, 75, 80, 86, 89, and 90 so the same
wheel can run on a range of NVIDIA GPUs. Change `CMAKE_CUDA_ARCHITECTURES` when
configuring CMake directly if you need a different target set. Building does
not require a GPU, but running the CUDA function does.

When changing Python dependencies, run `uv lock` and commit the updated
`uv.lock`. Normal setup should use `uv sync --locked` so it fails instead of
silently changing the dependency resolution.
