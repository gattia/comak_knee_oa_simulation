# comak_knee_oa_simulation

Code for **"Neural shape models enable cohort-scale simulation of knee cartilage mechanics"**
(Gatti et al.; medRxiv preprint, link to be added).

The repository contains the whole pipeline: MRI segmentations to subject-specific knee models built
with neural shape models, gait simulation with COMAK (OpenSim-JAM), contact pressure and cartilage
thickness change on a shared reference surface, and the statistics and rendering behind every
manuscript figure.

## Citation

```
To be added with the preprint link.
```

## Requirements

All environments are Linux x86-64. What you need depends on how far back you start
([`docs/environment.md`](docs/environment.md) has the details):

| to | install |
|---|---|
| redraw the figures | `requirements-figures.txt` (numpy, pandas, matplotlib, SciPy, Pillow) |
| re-derive the statistics, build models, simulate | `environment.lock.txt` + `requirements.lock.txt` (exact) or `environment.yml` (readable spec), plus OpenSim with the JAM solver built from source and the four pinned libraries (NSM, pymskt, nsosim, pycomak) |
| re-fit the mixed models | `environment-r.yml` ([`docs/r_environment.md`](docs/r_environment.md)) |
| re-run AddBiomechanics | `tools/addbiomechanics/*.yml` ([`tools/addbiomechanics/README.md`](tools/addbiomechanics/README.md)) |

An NVIDIA GPU is needed only to build knee models (`run_knee.py rebuild | reconstruct`,
`run_synthetic.py build`), for the pressure transfer's shape walks (`derive_pressure.py walk`) and for
the optional recomputation of the thickness maps (`derive_thickness.py decode | interpolate`; the
thickness change the analyses read is a released input, see *Three ways in*, (c)). Simulations,
statistics and figures run on CPUs.

The `opensim` package on PyPI and conda-forge does not contain COMAK or the cartilage contact model.
The simulations need the OpenSim-JAM fork, built from source into the pipeline environment
(`docs/environment.md`, *1. OpenSim with the JAM solver*).

## Data

The derived data (segmentations, 3D models, simulation results and `04_results`, the tables the
figures read) are released on publication on Hugging Face: `<URL to be added on publication>`.

| stage | contents | size |
|---|---|---:|
| `00_inputs` | segmentations, meshes, shape-model latents, gait data, reference model | 27 GB |
| `01_models` | per-knee reconstructed anatomy and OpenSim knee models | 110 GB |
| `02_simulations` | COMAK gait simulations, per knee and gait | 1.5 TB |
| `03_derived` | cartilage thickness and contact pressure on the shared reference surface | 52 GB |
| `04_results` | the statistics the figures read | 2.8 GB |

The code finds the data root in this order (`src/comak_knee_oa/paths.py`):

1. `$COMAK_DATA_ROOT`;
2. `data_root.txt` at the repository root, one line, gitignored (see `data_root.txt.example`);
3. `<repo>/data`.

The root must already exist; the code never creates it. The vendored inputs carry a `_manifest.json`
with the SHA256 of every file (`PYTHONPATH=src python -m comak_knee_oa.manifest verify inputs` re-hashes
them). Each model build and each simulation writes a `run_record.json` naming its inputs, options, config,
machine and the commits of this repository and the pinned libraries; the pressure transfer writes a
`_provenance.json` per knee with its inputs, options and library commits.

Other sources:
- MRI: the Osteoarthritis Initiative, [nda.nih.gov/oai](https://nda.nih.gov/oai). The cohort table
  (Extended Data Table 1) also needs the OAI clinical files (`oai_clinical_root.txt.example`).
- The healthy reference gait is public:
  [opensim-jam-org/jam-resources](https://github.com/opensim-jam-org/jam-resources)
  (`docs/reproduce.md`, *2. Data*, names the four files).
- The nine measured gait donors' raw motion data are not shared. Their AddBiomechanics outputs
  and the simulations made with them are in the release.

**Bringing your own gait data.** A walking trial (marker `.trc` and force-plate `.mot`) is scaled with
AddBiomechanics run locally on the COMAK skeleton (`pipeline/run_addbiomechanics.py stage --donor <subject>
--skeleton <skeleton.osim> --markers <trial.trc> --grf <trial_grf.mot> --out <run folder>`, then `run`
and `vendor`), given a config in
`config/measured_gait/`, and simulated like any donor. The steps are in
[`docs/scaling_comak_model.md`](docs/scaling_comak_model.md) and the engine setup in
[`tools/addbiomechanics/README.md`](tools/addbiomechanics/README.md);
`tests/test_addbiomechanics_own_data.py` checks that your own files stage the same run folder as a
donor's trial.

## Three ways in

**(a) Redraw the figures** from `04_results` alone (no GPU, no OpenSim, no R; Python 3.9 to 3.11, for
which every pinned package has a wheel):

```bash
python -m pip install -r requirements-figures.txt
echo /path/to/data > data_root.txt
python figures/render_all.py                 # every figure, into figures/output/v3/ (the run the data root declares)
python figures/figure3_shape_pressure.py     # or one
```

**(b) Redo the statistics.** Each figure has a derivation (`pipeline/derive_figure*.py`, writing
`04_results/<figure>/`), for some an R model, then the render.

Derivations and the R models write into the data root, so give them a run root of your own first: a
folder that links the download's `00_inputs`, `01_models`, `02_simulations` and `03_derived` and holds an
empty `04_results/`; `new_root` makes it. The download itself is declared read-only and the code refuses
to write into it. In the pipeline environment:

```bash
export PYTHONPATH=$PWD/src
DL=<downloaded root>; RUN=<a run root of your own>
python -m comak_knee_oa.new_root create --root $RUN --run-name v3 --baseline $DL --link-addbiomechanics
for p in $DL/01_models/* $DL/02_simulations/* $DL/03_derived/*; do
    python -m comak_knee_oa.new_root link --root $RUN "${p#$DL/}"
done
export COMAK_DATA_ROOT=$RUN

python pipeline/derive_figure4_data.py
Rscript figures/figure4_decile_dose_response.R     # in the comak-r environment
python figures/figure4_decile_dose_response.py
```

`--run-name v3` makes the run root read the released simulations under their own names
(`02_simulations/<gait>/<knee>/comak_v3/`). Figures 4, 5, 7 and 8 derive from `03_derived` (with `00_inputs` and `01_models`). Figures 2, 3
and 6, and Figure 1's panels, read `02_simulations` directly; Extended Data Figure 4 reads
`01_models`. [`figures/README.md`](figures/README.md) lists every chain.

**(c) Re-run the pipeline.** [`docs/reproduce.md`](docs/reproduce.md) gives every step in order, with
hardware and times, and says what reproduces bit for bit. For one knee, in a run root of your own
(`(cd src && python -m comak_knee_oa.new_root create …)`, `docs/reproduce.md` *2. Data*):

```bash
K=9003316_RIGHT
python pipeline/run_knee.py rebuild   --config config/reconstruction.json --knee $K          # GPU
python pipeline/run_knee.py simulate  --config config/measured_gait/Subject_102.json --knee $K
python pipeline/run_knee.py verify    --config config/measured_gait/Subject_102.json --knee $K
python pipeline/run_knee.py reproduce --config config/measured_gait/Subject_102.json --knee $K
```

`rebuild` builds the model from the knee's saved shape-model fit (`01_models/OARSI_May.22.2026_Full_run`)
with the shipped `config/reconstruction.json`; that is how the paper's models in `01_models/v3/` were
built. `reconstruct` fits the shape models again, which is stochastic. `reproduce` re-simulates a knee
and compares it with the released result. Figure 3's synthetic knees run through
`pipeline/run_synthetic.py axis | build | simulate` (job arrays in `slurm/`). To spread the simulations
of one gait over many workers, `pipeline/run_queue.py` claims knees one at a time from a SQLite work
list and runs `simulate` and `verify` on each (`docs/reproduce.md`, *5. Running at scale*).

Not every stage is re-run from raw inputs. The shape-model fits, cartilage thickness and its change
over time (`03_derived/thickness_merged`, which every analysis reads), the transfer's per-knee
transforms and the region labels are released inputs; `docs/reproduce.md`, *Frozen inputs*, lists them.
`pipeline/derive_thickness.py` recomputes thickness maps and thinning rates from the released
thickness shape-model fits (`00_inputs/latents_thickness`) into `03_derived/thickness/`. The analyses
read `03_derived/thickness_merged/` instead, a released tree that joins those outputs (every knee at the
baseline visit) with the follow-up-visit outputs of an earlier derivation by the same method (its
`_merge_report.json` records each file's source); no script here writes it.

**Run names.** Names you meet in the data and the configs:
- `v3`: the paper's run. Its models are in `01_models/v3/`, its simulations in
  `02_simulations/<gait>/<knee>/comak_v3/`, its quality-control keys are `<gait>_v3`, and configs write
  it as `{run}` (the data root's `_run.json` names it).
- `OARSI_May.22.2026_Full_run`: the earlier run that made the saved shape-model fits `v3` builds from.
  The prefix comes from the cohort sample's file name (`sample_used_for_OARSI_analysis…`,
  `pipeline/vendor_sources.json`); OARSI is the Osteoarthritis Research Society International.
- `v2`: the code's name for that earlier run's data root, which declares no run name;
  `src/comak_knee_oa/cohorts.py` can still read it (`V2_LAYOUT`, `MEASURED_V2`).
- `paper1-v3`: the git tag on the nsosim, pycomak and OpenSim-JAM commits the `v3` run used.
- `default`: the reference gait's condition key in the derived trees and the quality-control table;
  its simulations are under `02_simulations/reference/`.
- `Subject_121` and `RSubject_121` in `config/measured_gait/`: two different donors. `Subject_121` is
  the participant the donor-selection design chose and the ninth donor of `v3`; `RSubject_121` is the
  ninth donor of the `v2` run. Both configs are kept so either data root can be read; a new run root
  simulates `Subject_121` unless it declares otherwise (`src/comak_knee_oa/cohorts.py`).

## Repository layout

```
src/comak_knee_oa/      library: data paths, manifests, pins, configs, run records, comparisons
pipeline/               the stages: AddBiomechanics, model build, simulation, derivations, an optional work queue
figures/                one entry point per manuscript figure, the R models, render_all.py
config/                 run configs: model build, reference gait, measured gaits, synthetic axis
slurm/                  job-array scripts for the synthetic knees
tools/addbiomechanics/  environments and wrappers for running AddBiomechanics locally
tests/                  pytest suite
docs/                   environment, reproduction, data provenance, pinned versions, scaling
```

## Figures and tables

| manuscript item | script | output (`figures/output/v3/`) |
|---|---|---|
| Figure 1 | `figures/figure1_pipeline_overview.py` | `figure1_pipeline_overview.png` (+ `.pdf`) |
| Figure 2 | `figures/figure2_compartment_mechanics.py` | `figure2_compartment_mechanics.png` |
| Figure 3 | `figures/figure3_shape_pressure.py` | `figure3_shape_pressure.png` |
| Figure 4 | `figures/figure4_decile_dose_response.py` | `figure4_decile_dose_response.png` |
| Figure 5 | `figures/figure5_spatial_colocalization.py` | `figure5_spatial_colocalization.png` |
| Figure 6 | `figures/figure6_loading_determinants.py` | `figure6_loading_determinants.png` |
| Figure 7 | `figures/figure7_baseline_roc.py` | `figure7_baseline_roc.png` |
| Figure 8 | `figures/figure8_pressure_roi_biomarker.py` | `figure8_pressure_roi_biomarker.png` |
| Extended Data Fig. 1 | `figures/figure_ed1_latent_interpolation.py` | `figure_ed1_latent_interpolation.png` (+ `.pdf`) |
| Extended Data Fig. 2 | `figures/figure5_spatial_colocalization.py` | `figure_ed2_spatial_colocalization_max.png` |
| Extended Data Fig. 3 | `figures/figure8_pressure_roi_biomarker.py` | `figure_ed3_pressure_roi_biomarker_tibia.png` |
| Extended Data Fig. 4 | `figures/figure_ed4_fatpad.py` | `figure_ed4_fatpad.png` (+ `.pdf`) |
| Extended Data Fig. 5 | `figures/figure_ed5_cartilage_regions.py` | `figure_ed5_cartilage_regions.png` |
| Extended Data Table 1 | `pipeline/derive_table_ed1_cohort.py` | `04_results/table_ed1_cohort/` |

Mixed models in R: Figure 2 `figures/figure2_mixed_model.R`, Figure 3 `figures/figure3_mixed_model.R`,
Figure 4 `figures/figure4_decile_dose_response.R`, Figure 6 `figures/figure6_mundlak_model.R`.

## Companion libraries

| library | role | repository |
|---|---|---|
| nsosim | OpenSim knee models from neural shape models | [github.com/gattia/nsosim](https://github.com/gattia/nsosim), tag `paper1-v3` |
| pycomak | COMAK, ForSim and joint mechanics from Python | [github.com/gattia/pycomak](https://github.com/gattia/pycomak), tag `paper1-v3` |
| OpenSim-JAM fork | the contact solver, built from source | [github.com/gattia/opensim-core](https://github.com/gattia/opensim-core), tag `paper1-v3` |
| NSM | neural shape models | [github.com/gattia/nsm](https://github.com/gattia/nsm) |
| pymskt | mesh handling, cartilage thickness | [github.com/gattia/pymskt](https://github.com/gattia/pymskt) |
| pyKneeSPM | surface statistical parametric mapping (Figure 5) | [github.com/gattia/pyKneeSPM](https://github.com/gattia/pyKneeSPM) |

The exact commits are in `src/comak_knee_oa/pins.py` and [`docs/pinned_versions.md`](docs/pinned_versions.md);
`PYTHONPATH=src python -m comak_knee_oa.pins check --strict` confirms that an installation uses them.

## License

MIT; see `LICENSE`.

## Contact

The corresponding author, Anthony A. Gatti (aagatti@stanford.edu).
