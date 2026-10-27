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

## Comparing kernel experiments

Install the benchmark dependencies with `python -m pip install -e ".[dev,benchmark]"`,
then run:

```sh
python benchmarks/benchmark_silhouette_configurations.py --samples 5000 --clusters 5
```

This runs every registered configuration on the same input, checks scores against
the baseline, and prints per-stage CUDA event timings. It writes the individual
runs and summary to `build/silhouette-configuration-timings.json`; use
`--warmups`, `--repeats`, `--seed`, and `--timing-output` to adjust the run.
Totals include kernel launches only, not allocation or data transfer. Timings for
all configurations use the same stage names, even when their kernels differ.

To try a new kernel alongside the existing one:

1. Add the kernel and a launch policy to
   `src/cpp/src/silhouette_experiments.cuh`. The `AArrayRowLaunch` example
   shows how to record a launch with CUDA events. Match the corresponding
   baseline policy's arguments and output contract; for example, the A stage
   produces unnormalized sums and is followed by normalization.
2. In `src/cpp/src/silhouette_bench.cu`, define a configuration using
   `SilhouetteBenchmarkConfiguration` and replace the desired launch policy.
   Give it a unique `name` and add it to `BenchmarkConfigurations`.
3. Rebuild the extension and rerun the command above. The production path
   remains unchanged until you choose to promote an experiment.

The `a_row` example trades atomic additions for a per-sample loop; its speed
depends on the workload and GPU. Score comparisons use a tolerance because
floating-point accumulation order can differ between kernels.
