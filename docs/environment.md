# The environments

What you need depends on what you want to do. Everything here is Linux x86-64; the order of use is in
`docs/reproduce.md`.

| | file | for |
|---|---|---|
| render | `requirements-figures.txt` | drawing the manuscript figures from `04_results` |
| everything, exactly | `environment.lock.txt` + `requirements.lock.txt`, + 3 components by hand | re-deriving, building models, simulating gait, bit for bit |
| everything, readable | `environment.yml` + the same 3 components | the same packages, solved afresh |
| R | `environment-r.yml` | re-fitting the figures' mixed models (`docs/r_environment.md`) |
| AddBiomechanics | `tools/addbiomechanics/*.yml` | re-scaling the gait donors' bodies (`tools/addbiomechanics/README.md`) |

## Rendering the figures

```bash
python -m pip install -r requirements-figures.txt
python figures/render_all.py
```

NumPy, pandas, matplotlib, SciPy (Figures 1 and 7) and Pillow (Figures 1 and 3), pinned to the versions
the manuscript figures were rendered with. No figure script needs anything else. The figure
scripts read small tables and images out of `04_results` and draw them: no OpenSim, no GPU, no VTK, no
R and no simulation tree.

## Everything else

### The exact environment (use this to reproduce)

```bash
conda create -p <prefix> --file environment.lock.txt          # or -n comak-knee-oa
<prefix>/bin/python -m pip install --no-deps -r requirements.lock.txt
conda activate <prefix>
```

The two lock files were exported from the environment the paper's runs used. `environment.lock.txt`
lists every conda package file with its md5, so conda installs exactly those builds, solves nothing,
and refuses a file whose md5 has changed. `requirements.lock.txt` lists every package that environment
got from PyPI, at its exact version; `--no-deps` stops pip from resolving anything. The packages come
from conda-forge, plus MKL from the `anaconda` channel, CUDA 11.8 from `nvidia` and Torch from
`pytorch`. About 12 minutes with every file downloaded, and 6.4 GB.

Left out of the locks, because they are not installed from an index (each file's header names them):
- OpenSim with the JAM solver, and the four pinned libraries (components 1 and 2 below);
- pyKneeSPM, for the surface permutation tests of Figure 5 only (component 3);
- `sqlite-jobqueue`, a work-queue package in the paper's environment that this repository does not use
  (`pipeline/run_queue.py` uses Python's built-in `sqlite3`);
- the pip records of three packages that conda installed over (`vtk`, `pillow`, `attrs`); the conda lock
  installs the files the paper ran with.

**Harmless messages from pykeops.** NSM imports `pykeops` for an optional Sinkhorn (EMD) metric that the
pipeline never computes, and pykeops compiles on import. On a GPU node the first import logs
`cannot find -lnvrtc` and `collect2: error: ld returned 1 exit status` (twice) and writes a cache to
`~/.cache/keops2.2.3`; on a CPU node every command prints
`[KeOps] Warning : cuda was detected, but driver API could not be initialized`. Every pipeline command
also prints `schedulefree not found, skipping import` (from a pinned library). The command carries on, and
no result depends on either.

### The readable spec

```bash
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml
conda activate comak-knee-oa
```

`environment.yml` names the same versions, pins the numerical libraries (the BLAS, numpy, VTK) to the
paper's builds, and lets the solver choose everything else. `CONDA_CHANNEL_PRIORITY=flexible` is needed
because MKL comes from the `anaconda` channel, which strict channel priority excludes once conda-forge
offers an MKL of its own (a different binary). `micromamba` and `mamba` accept the same file.

An environment built from this file passed `pins check --strict` and reproduced two published
simulations bit for bit, but its solver chose other builds for most non-numerical packages, so the lock
is the record. The numerical pins are the ones that
matter: without them, numpy and the solver ran on OpenBLAS 0.3.34 and simulations moved by about as
much as a change of CPU type (`docs/reproduce.md`).

### Then three components by hand

**1. OpenSim with the JAM solver, built from source.** The simulations run on a fork of opensim-core
that contains COMAK and the cartilage contact model. The `opensim` package on PyPI and conda-forge is
stock OpenSim and does not contain them: an environment with it imports cleanly and cannot run a
simulation.

| | |
|---|---|
| source | the JAM fork of opensim-core, commit `c4b81decaac5bcaeaeea1a2943826ecd6b82ad7e`, tag `paper1-v3` on `github.com/gattia/opensim-core` |
| the paper's library | `libosimJAM.so`, 92,902,520 bytes, md5 `705cca876f52b54538537800a3bf021a`, built with GCC 13.3.0 (Ubuntu 24.04), CMake 3.28, `RelWithDebInfo` |

Build prerequisites (Ubuntu package names): a C/C++ compiler (`build-essential`), `cmake`, `git`,
`curl`, and for SWIG `autoconf`, `automake`, `libtool`, `bison` and `libpcre2-dev`. Simbody needs a
BLAS/LAPACK to link against (`libopenblas-dev`). OpenSim's dependency build downloads its other
dependencies, so it needs network access. Python and NumPy come from the activated environment.

```bash
# in the activated environment; JAM=<a build folder outside the repository>
git clone https://github.com/gattia/opensim-core.git $JAM/opensim-core
git -C $JAM/opensim-core checkout paper1-v3      # = c4b81decaac5bcaeaeea1a2943826ecd6b82ad7e

# SWIG 4.1.1 for the Python bindings
curl -L https://github.com/swig/swig/archive/refs/tags/v4.1.1.tar.gz | tar xz -C $JAM
(cd $JAM/swig-4.1.1 && sh autogen.sh && ./configure --prefix=$JAM/swig --disable-ccache && make && make install)

# dependencies, then OpenSim
cmake -S $JAM/opensim-core/dependencies -B $JAM/deps_build -DCMAKE_INSTALL_PREFIX=$JAM/deps_install \
    -DSUPERBUILD_ezc3d=ON -DOPENSIM_WITH_CASADI=OFF -DOPENSIM_WITH_TROPTER=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build $JAM/deps_build -j8
cmake -S $JAM/opensim-core -B $JAM/build -DCMAKE_INSTALL_PREFIX=$JAM/install \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOPENSIM_DEPENDENCIES_DIR=$JAM/deps_install \
    -DBUILD_PYTHON_WRAPPING=ON -DOPENSIM_PYTHON_CONDA=ON -DBUILD_JAVA_WRAPPING=OFF \
    -DOPENSIM_C3D_PARSER=ezc3d -DOPENSIM_WITH_CASADI=OFF -DOPENSIM_WITH_TROPTER=OFF \
    -DOPENSIM_INSTALL_UNIX_FHS=OFF -DSWIG_EXECUTABLE=$JAM/swig/bin/swig \
    -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python -DPython3_ROOT_DIR=$CONDA_PREFIX
cmake --build $JAM/build -j8 && cmake --install $JAM/build

# the Python bindings (from a copy: pip writes build files next to setup.py),
# and the library path every activation needs
cp -r $JAM/install/sdk/Python $JAM/opensim_python && python -m pip install --no-deps $JAM/opensim_python
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/opensim_jam.sh <<EOF
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:$JAM/install/sdk/lib:$JAM/install/sdk/Simbody/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF
```

Pin `Python3_EXECUTABLE`: a later re-configure can otherwise pick up the system Python and fail. The
library path is set by the environment's activation scripts, so run everything after `conda activate`,
or through `conda run`, which runs them too. The environment's `python` called directly does not, and
cannot import `opensim`.

**Which BLAS the solver uses.** Simbody links `libopenblas.so.0`. What matters is the library loaded at
run time: the activation line above puts the environment's own `lib/` first, so `libopenblas.so.0`
resolves to the environment's OpenBLAS 0.3.21, the one the paper's runs loaded. Keep that order, and
build Simbody against OpenBLAS (not MKL or a generic `libblas.so.3`), or the solver calls a different
BLAS. `pins check --strict` compares the md5 of every BLAS library the process loads, so it tells you.

Your library will not have the paper's md5 even from the same commit: the compiler, its version and the
build paths all change the binary. Accept your build explicitly, by file and md5. The check prints the
acceptance, and every run record lists the md5 of the JAM library the simulation loaded:

```bash
PYTHONPATH=src python -m comak_knee_oa.pins check --strict --accept libosimJAM.so=<md5 of your build>
export COMAK_PINS_ACCEPT=libosimJAM.so=<md5 of your build>      # the same, for jobs and the slurm/ scripts
```

Simulations are bit-identical only with the paper's binary, environment and CPU type
(`docs/reproduce.md`); with a build of your own, judge a re-run against the cross-CPU differences
(`docs/reproduce.md`, *4. Checking a re-run against the released one*).

**2. The pinned libraries: NSM, pymskt, nsosim, pycomak.** The pipeline runs from exports of exact
commits, put first on `PYTHONPATH`, rather than from whatever is installed. `src/comak_knee_oa/pins.py`
names the commits (`docs/pinned_versions.md`):

| library | commit | source |
|---|---|---|
| NSM | `bb2c6a3b17bd2cdb6b3378ea11528763e2698bf2` | `github.com/gattia/nsm` |
| pymskt | `88b947e3075d270f345bf7f8b545ead7daefd125` | `github.com/gattia/pymskt`, `main` |
| nsosim | `29037fbfd71db9d5b3f2874214cff4eab2a79dbf` | `github.com/gattia/nsosim`, tag `paper1-v3` |
| pycomak | `bc1a59fed07529840a8b4f267161b249c6414c4f` | `github.com/gattia/pycomak`, tag `paper1-v3` |

Clone each, then export and check, from the repository root (`docs/reproduce.md`, *1. Prerequisites*):

```bash
(cd src && python -m comak_knee_oa.pins export --dest <dir> --repo NSM=<checkout> --repo pymskt=<checkout> \
    --repo nsosim=<checkout> --repo pycomak=<checkout>)
export PYTHONPATH="$(cd src && python -m comak_knee_oa.pins pythonpath --dest <dir>):$PWD/src"
python -m comak_knee_oa.pins check --strict
```

`check --strict` compares the four commits as imported, the md5 of the JAM library, and the md5 of every
BLAS and LAPACK library the process loads (numpy's MKL, SciPy's own OpenBLAS, the solver's OpenBLAS) with
the paper's. Two of the libraries matter in ways that do not raise: the shape models do not load
correctly under a newer NSM, and an older pymskt meshes differently.

**3. pyKneeSPM**, only for the surface permutation tests (*The SPM packages* below).

### What the files do and do not guarantee

The lock reproduces the paper's simulations bit for bit. It was tested by recreating it from a fresh
package cache, adding the JAM build and the pinned libraries, and re-running two published simulations of
one knee on the CPU models their run records name; both were bit-identical.

Not tested in it: the GPU model build (in the paper's own environment it is byte-identical apart from the
run name), and Torch on a GPU. Not covered by either file: another operating system or architecture; the
JAM build, which is compiled from source (above); and pyKneeSPM.

## The SPM packages

The surface statistics of Figure 5 and Extended Data Figure 2 need `pyKneeSPM`. Only
`pipeline/derive_figure5_data.py spm` imports it, so it is installed by hand: the `maps` stage and every
figure script read the cluster maps that `spm` wrote, and run without it. `spm1d==0.4.33`, which pyKneeSPM
and `derive_figure2_data.py` both import, is in both environment files.

```bash
pip install --no-deps "git+https://github.com/gattia/pyKneeSPM.git@f8ac9fa0092d52a6a401e0489e46b1d8502a1e15"
```

This commit draws independent permutations and takes a seed for the permutation null
(`docs/pinned_versions.md`). An older pyKneeSPM installs and runs but has no `seed` parameter, so it
produces a different null and different p-values; `_import_spm()` in `derive_figure5_data.py` checks for
the parameter and refuses to start without it.
