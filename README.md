# FrogML

FrogML is a small Python package with a CUDA extension.

## Requirements

Building from source requires a CUDA 12.0 or newer toolkit and a host C++
compiler supported by that toolkit. A CUDA-capable GPU is only required to run
the extension, not to build it. On Linux, CUDA 12.0 supports GCC 12; if needed,
select a compatible compiler before installing:

```sh
export CC=gcc-12
export CXX=g++-12
```

## Install for development

Use any Python version supported by the package (`>=3.9`) and activate a
virtual environment, then run:

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

The editable install builds the CUDA extension and installs the test dependency.
The build uses the CMake version declared in `pyproject.toml`. After native code
or CMake changes, rebuild with:

```sh
python -m pip install --force-reinstall --no-deps -e .
```

If you use `uv`, the equivalent initial setup is:

```sh
uv sync --extra dev
uv run pytest
```

After changing C++/CUDA sources or CMake configuration, explicitly reinstall
FrogML to rebuild the native extension before testing:

```sh
uv sync --extra benchmark --reinstall-package frogml
uv run pytest
```

A plain `uv sync` may leave an already-installed editable extension untouched.
A committed `uv.lock` is not required; `uv` resolves dependencies from
`pyproject.toml` when setting up the environment.

Build a wheel with either `uv build --wheel` or `python -m pip wheel . --no-deps`.

The default build includes CUDA targets 70, 75, 80, 86, 89, and 90 so the same
wheel can run on a range of NVIDIA GPUs. Change `CMAKE_CUDA_ARCHITECTURES` when
configuring CMake directly if you need a different target set.
