# Pinned versions

The code versions behind the paper's results. The machine-readable pins are in
`src/comak_knee_oa/pins.py`, enforced as in *Enforcing the pins*. Every run record names what actually
ran, so where a record and this page disagree, the record wins.

## Current pins (v3)

| | commit or version | tag | source |
|---|---|---|---|
| `nsosim` | `29037fbfd71db9d5b3f2874214cff4eab2a79dbf` | `paper1-v3` | `github.com/gattia/nsosim` |
| `pycomak` | `bc1a59fed07529840a8b4f267161b249c6414c4f` | `paper1-v3` | `github.com/gattia/pycomak` |
| `opensim-core`, JAM fork | `c4b81decaac5bcaeaeea1a2943826ecd6b82ad7e` | `paper1-v3` | `github.com/gattia/opensim-core` |
| `NSM` | `bb2c6a3b17bd2cdb6b3378ea11528763e2698bf2` | | `github.com/gattia/nsm`, `main` |
| `pymskt` | `88b947e3075d270f345bf7f8b545ead7daefd125` | | `github.com/gattia/pymskt`, `main` |
| `pyKneeSPM` | `f8ac9fa0092d52a6a401e0489e46b1d8502a1e15` | | `github.com/gattia/pyKneeSPM`, `main` |
| `spm1d` | 0.4.33 | | PyPI |

This repository's own commit, with the library commits, is recorded in every model build's and every
simulation's `run_record.json` at the moment it runs. The pressure transfer's per-knee `_provenance.json`
records the library commits it used.

**The solver library**, identified by its bytes: `libosimJAM.so`, 92,902,520 B, md5
`705cca876f52b54538537800a3bf021a`, built from the JAM fork's commit above (`docs/environment.md`).

**The BLAS** of the paper's environment, which moves simulation results by about as much as a change of
CPU type:
- numpy 2.0.2 on MKL 2022.1.0 (`anaconda`, build `hc2b9512_224`), through conda-forge
  `libblas 3.9.0=16_linux64_mkl`;
- the solver's Simbody on conda-forge OpenBLAS 0.3.21 (`pthreads_h78a6416_3`);
- SciPy 1.13.1 on the OpenBLAS 0.3.27 its wheel ships.

`environment.lock.txt` and `requirements.lock.txt` recreate that environment exactly
(`docs/environment.md`), and `PYTHONPATH=src python -m comak_knee_oa.pins check --strict` compares the md5 of every BLAS
library a process loads with `pins.BLAS`.

**Settings the nsosim and pycomak pins expose.** Each is switched by a config key: the check that
AddBiomechanics scaled the COMAK skeleton (`scaling.allow_generic_mismatch`), the per-body mass rule
(`scaling.mass_rule`, required; `docs/scaling_comak_model.md`), and the fat pad's clearance rays
(`fatpad_clearance_rays`). The shipped configs set `volume`, `dilation_normals` and a COMAK window pad
(`comak_timing.comak_window_pad_s`) of 0.005 s for the measured gaits and 0.0 s for the reference gait.

## Enforcing the pins

`src/comak_knee_oa/pins.py` holds the commits a run must use (NSM, pymskt, nsosim, pycomak) and the
identity of the solver library. A launcher does three things, from the repository root:

```bash
# 1. Export each pinned commit into a directory nobody edits (read-only, with a PINNED_COMMIT marker)
(cd src && python -m comak_knee_oa.pins export --dest <dir> --repo NSM=<checkout> --repo pymskt=<checkout> \
    --repo nsosim=<checkout> --repo pycomak=<checkout>)
# 2. Put the exports first on the path
export PYTHONPATH="$(cd src && python -m comak_knee_oa.pins pythonpath --dest <dir>):$PWD/src"
# 3. Refuse to start on any difference
python -m comak_knee_oa.pins check --strict
```

Exports rather than installed checkouts, because a checkout is shared: switching its branch changes the
code under every job that imports it next. Run records name each library's commit, including exports
(read from `PINNED_COMMIT`); a checkout with uncommitted changes records
`<sha>-dirty-<hash of the diff>`, so two different uncommitted states never look alike. Every run record
also lists the OpenSim libraries the process loaded, with their md5.

A JAM library built elsewhere has another md5 even from the same commit; accept it by name and md5
(`--accept libosimJAM.so=<md5>`, or `COMAK_PINS_ACCEPT` for jobs).

## Why the pyKneeSPM commit matters

`pyKneeSPM@f8ac9fa` draws the permutations of the correlation null independently, and takes a seed for
every permutation class (threaded through `MonteCarlo*` and `SingleStatisticSPM`). Without the seed, two
runs of the same cell on identical data give different cluster p-values (0.0059 and 0.0048 in one test).
An older pyKneeSPM installs and runs, so `derive_figure5_data.py::_import_spm()` checks for the `seed`
parameter and refuses to start without it. The seeds themselves are described in
`docs/data_provenance.md` (*Surface permutation tests*).

## Exact reproduction and CPU type

COMAK is sensitive to CPU microarchitecture: the same input on a different CPU type moves knee rotations
by about 1° and contact pressures by up to a few percent. This is floating-point rounding, deterministic
per CPU type. When a comparison exceeds tolerance, check which CPU produced each result before concluding
that the code changed (`docs/reproduce.md` has the measured differences).
