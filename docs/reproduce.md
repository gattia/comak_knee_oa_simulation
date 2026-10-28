# Reproducing the results, end to end

This page runs the whole pipeline in order, from scaling the gait donors' bodies to the figures. Every step is
one command from this repository. Run them by hand on one machine, or hand them to any job scheduler: a job is
just one of these commands, and nothing about a result depends on how it was launched.

To redraw the figures or recheck the statistics only, you do not need any of this; see *Three ways in* in
`README.md`.

## What reproduces bit for bit, and what does not

Read this first, so a difference you see is not a surprise.

"Same environment" below means the exact environment (`environment.lock.txt`, `docs/environment.md`) with the
paper's build of the JAM solver; `PYTHONPATH=src python -m comak_knee_oa.pins check --strict` verifies both.

| step | repeat on the same hardware, same environment | repeat on other hardware |
|---|---|---|
| AddBiomechanics skeleton | byte-identical (md5 in section 3.1) | not measured |
| AddBiomechanics scaling of a donor | **not reproducible.** Its first inverse kinematics uses random restarts, with no seed; repeat runs move the knee scale factor `s_wa` by up to about 1 % (section 4). Contact pressure moves about 1.5 times as much as `s_wa`, the same for every knee of that donor | same |
| model build from the saved shape fits | identical apart from the run name, with the same GPU type and CPU type | ligament attachments move by micrometres (GPU type); wrap surfaces by up to 0.15 mm (CPU type); a few meniscus and fat-pad surfaces re-triangulate (≤ 0.005 mm) |
| gait simulation | **bit-identical** on the same CPU model, given the same model file and the config the run recorded (`run_knee.py reproduce` uses the recorded config by default) | differs between Intel Ice Lake and Broadwell (35 simulations). Largest difference over stance, per simulation: median 1.0° in knee adduction and rotation and the three patellar rotations (1.3° in knee flexion), 1.0 mm in translations, 0.2 % in tibial stance-mean pressure; at most 5.7° (8.3°), 5.1 mm and 3.2 %. About 4 % of simulations flip the patella-repositioning decision (whether the forward simulation moved the patella before COMAK, `forsim_results/total_patella_update.json`). Haswell and Broadwell gave identical results for the one knee tested. Other pairs were not measured |
| pressure summaries, transfer to the reference surface, matrices, quality control, coverage audit | byte-identical | not measured |
| figure statistics | byte-identical | the ROC classifiers (`derive_figure7_data.py`) depend on the CPU type (AUROC ≤ 0.002; the paper's were derived on Intel Broadwell). R model refits agree to ≤ 6e-4 relative |

Consequences:
- **The environment matters as much as the CPU type.** The same knee, model and CPU in an environment built from an
  unpinned spec (numpy and the solver on OpenBLAS 0.3.34 instead of MKL and OpenBLAS 0.3.21) moved by up to 1.5°,
  1.1 mm and 0.4 % in tibiofemoral pressure, and the model build changed in its last digits. Use the lock
  (`environment.yml` pins the same numerical libraries).
- **The vendored AddBiomechanics output is the input of record**, not a new AddBiomechanics run. To reproduce a
  published simulation, use the vendored body.
- **A rebuilt model moves a simulation as much as a CPU change does** (median 0.7°, 0.3 % pressure). Compare a
  rebuilt knee with a published one at group level, not knee by knee.
- **To reproduce a simulation bit for bit**, run it in the exact environment, on the CPU model its
  `run_record.json` names (`environment.cpu_model`), from the same model file, with the config the run recorded.
  `run_knee.py reproduce` does the last part for you (section 4).

## 1. Prerequisites

**Hardware.** An NVIDIA GPU for model building (the paper's models were built on an RTX 2080 Ti; a build uses
about 2 GB of host memory); CPUs for everything else. One simulation needs one core and at most 4 GB.

**Software.** Full instructions are in `docs/environment.md`. In short:

```bash
conda create -n comak-knee-oa --file environment.lock.txt           # the pipeline, as the paper ran it
conda run -n comak-knee-oa python -m pip install --no-deps -r requirements.lock.txt
conda env create -f environment-r.yml                               # comak-r: the figures' mixed models
conda env create -f tools/addbiomechanics/environment.yml           # ab_engine: only to re-run AddBiomechanics
conda env create -f tools/addbiomechanics/opensim_environment.yml   # ab_opensim: likewise
```

`environment.yml` is the same pipeline environment as a readable spec; it is not the record (`docs/environment.md`).

plus three things no package index supplies:

1. **OpenSim with the JAM solver**, built from source and installed into `comak-knee-oa`
   (`docs/environment.md`, *1. OpenSim with the JAM solver*).
2. **The pinned libraries.** NSM, pymskt, nsosim and pycomak run from exports of exact commits, put first on the
   path. The commits are in `src/comak_knee_oa/pins.py`:

   ```bash
   conda activate comak-knee-oa
   LIBS=<a folder outside the repository>      # exports are read-only and kept per commit
   (cd src && python -m comak_knee_oa.pins export --dest "$LIBS" \
       --repo NSM=<NSM checkout> --repo pymskt=<pymskt checkout> \
       --repo nsosim=<nsosim checkout> --repo pycomak=<pycomak checkout>)
   export PYTHONPATH="$(cd src && python -m comak_knee_oa.pins pythonpath --dest "$LIBS"):$PWD/src"
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   python -m comak_knee_oa.pins check --strict
   ```

   `check --strict` compares the four library commits, the md5 of the loaded JAM library and the md5 of every BLAS
   library the process loads with the paper's. A JAM library you built yourself has another md5 even from the same
   commit; accept it by name and md5 (`--accept libosimJAM.so=<md5>`, or `COMAK_PINS_ACCEPT` for jobs), and read
   the simulation results against the hardware table above.
3. **The AddBiomechanics engine**, only to re-run AddBiomechanics: `tools/addbiomechanics/README.md`.

Every command below runs in `comak-knee-oa` with these variables set, from the repository root.

OpenSim appends its log to `opensim.log` in the current directory, here the repository root. Git ignores the file,
but every command that loads a model adds to it, every job of a cluster run included. The solver's own logs stay in
each simulation's results folder (`logs/`). To keep the checkout clean, run the commands from a work folder of your
own, with the repository's paths in full:

```bash
REPO=<the repository>; cd <a work folder>
python $REPO/pipeline/run_knee.py simulate --config $REPO/config/reference_gait.json --knee <knee>
```

The scripts find the repository from their own location and the data from `COMAK_DATA_ROOT`; only the setup above
(`cd src`, `$PWD/src`) is written for the repository root, so set the variables there first.

## 2. Data

The pipeline finds its data in `$COMAK_DATA_ROOT`, else in the path in `data_root.txt`, else in `<repo>/data`.

**Never write into the downloaded data root.** Create a run root of your own. It links the downloaded inputs
read-only and holds everything the run writes:

```bash
python -m comak_knee_oa.new_root create --root <run root> --run-name <name> \
    --baseline <downloaded root> --link-addbiomechanics
python -m comak_knee_oa.new_root link --root <run root> 01_models/OARSI_May.22.2026_Full_run   # saved shape fits
python -m comak_knee_oa.new_root link --root <run root> 03_derived/pressure_on_reference         # transfer cache
python -m comak_knee_oa.new_root link --root <run root> 03_derived/thickness_merged              # thickness change
python -m comak_knee_oa.new_root link --root <run root> 03_derived/thickness
python -m comak_knee_oa.new_root link --root <run root> 03_derived/region_labels
python -m comak_knee_oa.new_root link --root <run root> 03_derived/qc/thickness_visit_exclusions.csv   # see below
export COMAK_DATA_ROOT=<run root>
export COMAK_READ_ONLY_ROOTS=<downloaded root>
```

- `_run.json` in the run root declares the run: its name (configs write `{run}`, and outputs land in
  `01_models/<name>/`, `02_simulations/<gait>/<knee>/comak_<name>/`), the gait donors it simulates
  (`measured_gaits`), and the roots it must never write. Every pipeline command reads it.
- Any write that would land in a read-only root, directly or through a link, raises an error.
- `--link-addbiomechanics` uses the downloaded AddBiomechanics bodies (the input of record). Leave it out if you
  will re-run AddBiomechanics (section 3.1) and vendor your own, or if your copy of the data has no gait donor
  folders (below); the root then still declares the nine donors, so later steps report their arms as missing.
- `--measured-gaits <donor>[,<donor>…]` declares fewer donors, for a test run. `--link-addbiomechanics` still
  links every donor folder of the baseline; that is harmless.
- The linked trees are inputs this pipeline does not regenerate (*Frozen inputs* below).
- `thickness_visit_exclusions.csv` lists two visits whose thickness shape fits failed. No step writes it; the
  analyses that read thickness change refuse to run without it. Link only that file: the rest of
  `03_derived/qc/` is this run's own quality control (section 3.7).

**Gait data.**
- **The reference gait** (overground walking of one healthy adult; the reference arm) is public. It comes from
  opensim-jam's [`jam-resources`](https://github.com/opensim-jam-org/jam-resources), folder
  `models/knee_healthy/experimental_data/motion_analysis/`: `overground_17.trc` (markers), `overground_17_grf.mot`
  (ground reaction forces), `overground_17_ext_loads.xml` (how the forces are applied) and `static1.trc`. Our copies
  are byte-identical to that repository's commit `73306a0`. They ship in the data root at
  `00_inputs/reference/data/reference_data/motion_analysis/`, where `config/reference_gait.json` points. To take them
  from the source instead, clone `jam-resources` and copy those four files there.
- **The nine measured donors' motion data are not part of the public release** (they are not open data). Without
  them you can run the reference-gait arm and everything downstream of the released simulations, but not
  AddBiomechanics or the measured-gait simulations of those donors.
- **Your own gait data** can be used in their place. `docs/scaling_comak_model.md` takes a subject from a walking
  trial (a marker `.trc` and a force-plate `.mot`) to simulations: add your markers to the skeleton, run
  AddBiomechanics locally on your files (section 3.1, `--markers`/`--grf`), write the subject's config, and declare
  the subject in your run root.

## 3. The stages, in order

`<knee>` is a cohort knee such as `9003316_RIGHT` (the list is `00_inputs/cohort.json`); `<donor>` is a gait donor
such as `Subject_119`; a *gait config* is `config/reference_gait.json` or `config/measured_gait/<donor>.json`.

| step | unit | runs on | time per unit |
|---|---|---|---|
| 3.1 AddBiomechanics | donor | 8 cores, 32 GB | 30 min |
| 3.2 model build | knee | GPU + 4 cores, 8 GB (peak about 2 GB) | 6 min (RTX 2080 Ti) |
| 3.3 synthetic knees | knee | GPU + 4 cores, 8 GB | 8 min |
| 3.4 simulation | knee × gait | 1 core, ≤ 4 GB | 31 min median (p90 46 min; up to 4 h when a sweep falls back) |
| 3.5 pressure summaries and transfer | gait | CPU | minutes per gait |
| 3.6 matrices | gait | CPU | about 10 min |
| 3.7 quality control | all | CPU | about 75 min, shardable |
| 3.8 coverage audit | all | CPU | seconds |
| 3.9 figure statistics | figure | CPU | minutes; the surface permutation tests (`derive_figure5_data.py spm`) about 5 h on 20 cores |

The full run is 756 knees × 10 gaits plus 90 synthetic simulations: about 4,000 core-hours, 80 GPU-hours and
1.9 TB on disk (estimated from a 37-knee test run).

### 3.1 AddBiomechanics (optional; needs the donors' raw trials, or your own)

Skeleton (once; seconds), then one run per donor. Set up the engine, its two environments and the marker model in a
work folder outside the repository, as in `tools/addbiomechanics/README.md`, which also says why each piece exists.

```bash
W=<a work folder outside the repository>      # holds AddBiomechanics/, marker_model/, the skeleton and ab_runs/
python pipeline/prepare_addbiomechanics_skeleton.py --model $W/marker_model/full_body_healthy_knee_armless_trc.osim \
    --out $W/comak_ab_skeleton.osim --ab-geometry $W/AddBiomechanics/server/data/Geometry   # md5 be9a614e…
python pipeline/run_addbiomechanics.py stage --donor <donor> --skeleton $W/comak_ab_skeleton.osim \
    --zips '<raw trial zips>' --out $W/ab_runs/<donor>
#   your own subject instead: --markers <trial.trc> --grf <trial_grf.mot> in place of --zips
#   (first write config/measured_gait/<subject>.json; docs/scaling_comak_model.md, step 6)
OMP_NUM_THREADS=8 <ab_engine>/bin/python pipeline/run_addbiomechanics.py run $W/ab_runs/<donor> \
    --engine $W/AddBiomechanics/server/engine --python <ab_engine>/bin/python --opensim-prefix <ab_opensim>
python pipeline/run_addbiomechanics.py vendor $W/ab_runs/<donor> \
    --dest "$COMAK_DATA_ROOT/00_inputs/gait/addbiomechanics/<donor>"
```

Check `$W/ab_runs/<donor>/_results.json` before vendoring: AddBiomechanics can drop a whole trial segment in its
force-plate check. AddBiomechanics cuts each trial into segments of about 20 s and numbers them from 0; every donor's
config uses segment 3 (`markers_segment_3`, the fourth, 60–80 s). In the paper's run it dropped one donor's
segment 3 (`docs/scaling_comak_model.md`, step 4).

To compare your run with the vendored body (section 4): `python pipeline/run_addbiomechanics.py knee-factor
<downloaded root>/00_inputs/gait/addbiomechanics/<donor> $W/ab_runs/<donor>`.

### 3.2 Model build (GPU)

```bash
python pipeline/run_knee.py rebuild --config config/reconstruction.json --knee <knee>
```

Builds the knee's OpenSim model from its saved shape-model fit (`01_models/OARSI_May.22.2026_Full_run`) into
`01_models/<name>/<knee>/`. `--check` lists what it would do. Fitting the shape models again
(`run_knee.py reconstruct`) is possible but is not how the paper's models were made; see *Frozen inputs*.

### 3.3 Synthetic knees (Figure 3, `figure3_shape_pressure.py`)

```bash
python pipeline/run_synthetic.py axis --config config/synthetic_shape_axis.json --verify   # seconds, CPU
python pipeline/run_synthetic.py build --config config/synthetic_shape_axis.json --label <label>   # GPU
python pipeline/run_synthetic.py simulate --config config/synthetic_shape_axis.json --label <label> --gait <donor>
python pipeline/run_synthetic.py cells --config config/synthetic_shape_axis.json      # the label × gait table
```

`slurm/synthetic_build.sbatch` and `slurm/synthetic_simulate.sbatch` are job arrays over that table. Run all
synthetic simulations on one CPU type: they are a 10-point sweep per gait, so CPU differences do not average out.

### 3.4 Simulation (CPU)

```bash
python pipeline/run_knee.py simulate --config <gait config> --knee <knee> --check   # inputs present? nothing runs
python pipeline/run_knee.py simulate --config <gait config> --knee <knee>
python pipeline/run_knee.py verify   --config <gait config> --knee <knee>           # every solver output on disk
```

Measured-gait simulations scale the knee's model to the donor's AddBiomechanics body first. Each simulation writes
`run_record.json` (code commits, host, CPU, library md5s, the config used). **Count a simulation as done only when
`verify` passes**: a solver stage can fail inside a run that otherwise ends normally. One knee in the cohort
(`9916274_RIGHT`) has never converged in any gait.

### 3.5 Pressure on the shared reference surface

Per gait config, after its simulations:

```bash
python pipeline/derive_pressure.py stats    --config <gait config>   # per-vertex pressure over stance, per simulation
python pipeline/derive_pressure.py transfer --config <gait config>   # onto the 20,000-vertex reference surface
```

Both take `--check`; `stats` also takes `--knee`, `--n-shards` and `--shard`. The transfer reuses the linked cache
(`03_derived/pressure_on_reference`: per-knee transforms and shape walks, which do not depend on gait), and refuses
an existing output it has no provenance record for.

### 3.6 Matrices

```bash
python pipeline/combine_reference_data.py matrices --config <gait config>
```

One archive per gait: knees × reference vertices for pressure and thickness change, with validity masks. A knee
enters only with all four fields (pressure and thickness change, femur and tibia).

### 3.7 Quality control

```bash
python pipeline/derive_qc.py spatial                                         # reads the matrices
python pipeline/derive_qc.py timeseries --mode extract --n-shards 8 --shard <0..7>   # reads every joint_mechanics.h5
python pipeline/derive_qc.py timeseries --mode merge --n-shards 8
python pipeline/derive_qc.py timeseries --mode flag
python pipeline/derive_qc.py unify
```

Builds the one table of simulations whose contact pressure is a numerical artifact (bad in either detector).
Every figure that reads pressure takes its exclusions from it. The thresholds are absolute, so re-run this after any
re-simulation.

### 3.8 Coverage audit

```bash
python pipeline/audit_coverage.py --verbose
```

Every derived layer against what it should cover. Exit 0 = clean; 1 = a layer is short; 3 = a derived file no
layer checks. Run it before trusting any count: a layer built before a cohort change does not fail, it just covers
fewer knees.

A run of some knees only (a test of one knee, say) exits 1 by design: layers whose reference is the whole cohort
are short (for one knee, `step1_geometry` is short by 755 of 756), and the audit cannot be told which knees a run
meant to cover. Read the per-layer counts instead: every layer downstream of the simulations should cover the knees you ran.

### 3.9 Figures

Each figure has a derive step (`pipeline/derive_figure*.py`, writing `04_results/`), sometimes an R model
(`Rscript figures/figure*_*.R`, in `comak-r`, with `COMAK_DATA_ROOT` set), and a render (`figures/figure*.py`,
writing `figures/output/<run name>/`).
The chain for every figure, in order, is in `figures/README.md`. Then:

```bash
python figures/render_all.py
```

Derive the ROC figure (`derive_figure7_data.py`) on the CPU type you want to compare with (the paper: Intel
Broadwell); its `meta.json` records the machine. The surface permutation tests are seeded and run as 40
independent cells (`derive_figure5_data.py spm --cell <0..39>`, then `maps`).

On a test run of a few gaits, `derive_figure2_data.py` refuses before writing anything: its per-knee ensemble keeps
a knee only if it was simulated in at least five gait conditions.

The cohort table (`derive_table_ed1_cohort.py`) also needs the OAI clinical files, which are participant-level data
under the OAI data-use agreement and are not in the data root. Download "Complete Data - ASCII" from the OAI, then
set `OAI_CLINICAL_ROOT` to it, or put the path in `oai_clinical_root.txt` (see `oai_clinical_root.txt.example`).
Each checkout needs this, since the file is not tracked; without it the script stops with an error.

## 4. Checking a re-run against the released one

- **One simulation, end to end:**

  ```bash
  python pipeline/run_knee.py reproduce --knee <knee> \
      --published <downloaded root>/02_simulations/<gait>/<knee dir>/comak_<run>
  python pipeline/run_knee.py reproduce --knee <knee> --config <gait config>   # the same, found in your baseline
  ```

  It re-simulates the knee in your run root, from your root's build of its model (3.2), with the config the
  published run recorded in its `run_record.json` (`--use-config` overrides), and compares (next item). It passes
  when the result is bit-identical, or, on another CPU type, within the cross-CPU differences.
  - **Check a published simulation with `reproduce`, not with a plain `simulate`.** `simulate` always uses the
    shipped config; `reproduce` uses the values the published run recorded in its `run_record.json`, prints each
    one that differs from the shipped file, and runs with the recorded one. The released simulations recorded the
    shipped values (window pad 0.0 s for the reference gait, 0.005 s for the measured gaits).
  - It writes to `02_simulations/_reproduction/<gait folder>/<knee dir>/comak`, the gait folder being the one the
    published simulations live in (`reference`, `Subject_121`, …). Nothing is written to the published root.
  - It needs the published simulation in your copy of the data. `--published` takes any results folder you have;
    `--config` looks in your baseline, at `<downloaded root>/02_simulations/<gait folder>/<knee dir>/comak_<run>`.
    Without `02_simulations` in your copy, `reproduce` stops with "no published results folder at …": there is
    nothing to reproduce against.
- **Any two simulations:** `python -m comak_knee_oa.compare sims <published results dir> <re-run results dir>`.
  It first checks the three signal outputs (`comak/_force.sto`, the inverse dynamics, `joint_mechanics.h5`) for
  exact equality, and names any difference in CPU model or native-library md5 between the two run records (a
  library one record does not list is reported as not recorded, not as a difference). Then it
  reports what the paper reads: knee and patellar kinematics (degrees and millimetres, largest difference over
  stance) and each contact's stance-mean and peak pressure and force (percent). These are judged against the
  largest differences measured between two CPU types (`CROSS_CPU_ENVELOPE` in `src/comak_knee_oa/compare.py`, from
  35 simulations on Ice Lake and Broadwell). A run whose patella-repositioning decision differs is reported apart.
  Element-wise tolerances cannot make this judgement: a triangle at the edge of a contact patch switching on is a
  100 % change in one value.
- **Models:** `python -m comak_knee_oa.compare models <published model dir> <your model dir>`, each
  `01_models/<run>/<knee dir>`. It compares file by file with each run name masked, and links by the file they
  reach. `run_record.json` is expected to differ. A file that differs only in its numbers is reported with the
  largest absolute and relative difference. Built on the same GPU and CPU type in the same environment, nothing
  else differs.
- **AddBiomechanics:** `python pipeline/run_addbiomechanics.py knee-factor <vendored folder> <your run folder>`
  prints the knee scale factor `s_wa` of each and their difference; it gives no verdict. AddBiomechanics is not
  deterministic, so expect a difference: repeat runs differ by about 1 % in `s_wa`
  (`tools/addbiomechanics/README.md`).
- **Counts:** quote the number of simulations that pass `verify`, never a scheduler's "done".

## 5. Running at scale

Any scheduler works. What matters:
- every job runs one command from section 3, from one committed checkout, with the pinned libraries and the thread
  variables above;
- a simulation waits for its knee's model; a synthetic simulation for its synthetic knee;
- record each job's host and CPU (the run records do);
- re-check every finished simulation with `verify`.

`pipeline/run_queue.py` is an optional worker loop for the simulations of one gait config. `--seed` fills a
SQLite work list with the cohort's knees; each worker (`--config <gait config> --queue <db>`, one per scheduler
job, say) claims one knee at a time, atomically, runs `simulate`, and records the knee as `done` only when the
`verify` check finds every solver output, otherwise under the failure's own name (a solver budget,
`incomplete_output`, `error`). `--status` counts knees by state, and `--requeue` returns knees left `running` by a
killed worker. It submits no jobs and holds no science: delete the database and every result can still be run knee
by knee.

Resource requests the paper's jobs used: simulation 1 core, 6 GB, up to 16 h; model build 1 GPU, 4 cores, 8 GB;
AddBiomechanics 8 cores, 32 GB, 4 h.

## Frozen inputs (not regenerated by this pipeline)

These come with the data, made by earlier steps of the project. `docs/data_provenance.md` says where each came
from.
- MRI segmentations and surface meshes (Osteoarthritis Initiative).
- The shape-model fits of each knee (`01_models/OARSI_May.22.2026_Full_run`). Shape-model fitting is stochastic,
  so refitting gives slightly different anatomy; the saved fits are the record.
- The reference model and its one-time preparation (wrap surfaces, ligament attachments, mean surfaces).
- Cartilage thickness and its change over time (`03_derived/thickness`, `thickness_merged`; the analyses read
  `thickness_merged`. `pipeline/derive_thickness.py` can recompute the maps and rates into `03_derived/thickness`
  from the released thickness shape-model fits, but nothing rebuilds `thickness_merged`), the two excluded
  thickness visits (`03_derived/qc/thickness_visit_exclusions.csv`), the per-knee
  transforms and shape walks of the transfer (`03_derived/pressure_on_reference`), and the anatomical region labels
  (`03_derived/region_labels`).
- `03_derived/qc/chronic_knees_manual.csv` is the release's hand-curated list for the `v3` run (one knee). A new
  run root starts without one; add knees after inspecting that run's simulations.
