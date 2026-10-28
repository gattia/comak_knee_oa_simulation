# Figures

One entry point per manuscript figure. Each runs with no arguments, reads its tables and images from
`04_results/<figure name>/`, and writes into `figures/output/` (or, on a data root that declares a run
in `_run.json`, into `figures/output/<run>/`; `$COMAK_FIGURE_OUTPUT_DIR` overrides both).

```bash
python figures/figure3_shape_pressure.py   # one figure
python figures/render_all.py               # all of them (--list to see them, or a name to filter)
```

`render_all.py` runs every `figures/figure*.py`. It reports a figure whose `04_results` input is
absent (the data is not downloaded, or the derivation has not run) separately from one that raised,
and only the second fails the command. Rendering needs `04_results` only: no simulation tree, no
GPU, no OpenSim and no R.

## Derive, model, render

A figure that needs derived data has up to three steps:

| step | script | reads | writes |
|---|---|---|---|
| derive | `pipeline/derive_figure*.py` | `00_inputs` .. `03_derived` | `04_results/<figure>/` |
| model (some figures) | `figures/figure*_*.R` | `04_results/<figure>/` | `04_results/<figure>/` |
| render | `figures/figure*.py` | `04_results/<figure>/` | `figures/output/` |

The released `04_results` already holds every derived table and fitted model, so R is needed only to
re-fit (`docs/r_environment.md`). A render whose input is missing prints the command that builds it.

| manuscript item | derive | model | render |
|---|---|---|---|
| Figure 1 | `derive_figure1_panels.py` (needs OpenSim; also reads Figures 2–4's results) | | `figure1_pipeline_overview.py` |
| Figure 2 | `derive_figure2_data.py` (reads `joint_mechanics.h5`) | `figure2_mixed_model.R` | `figure2_compartment_mechanics.py` |
| Figure 3 | `derive_figure3_data.py`, `derive_figure3_geometry.py` (the 90 synthetic simulations) | `figure3_mixed_model.R` | `figure3_shape_pressure.py` |
| Figure 4 | `derive_figure4_data.py` (combined matrices) | `figure4_decile_dose_response.R` | `figure4_decile_dose_response.py` |
| Figure 5, Extended Data Fig. 2 | `derive_figure5_data.py spm`, then `maps` | | `figure5_spatial_colocalization.py` |
| Figure 6 | `derive_figure6_data.py`, `derive_figure6_gait_waveforms.py` (inverse dynamics) | `figure6_mundlak_model.R` | `figure6_loading_determinants.py` |
| Figure 7 | `derive_figure7_data.py` | | `figure7_baseline_roc.py` |
| Figure 8, Extended Data Fig. 3 | `derive_figure8_data.py` | | `figure8_pressure_roi_biomarker.py` |
| Extended Data Fig. 1 | none (a schematic) | | `figure_ed1_latent_interpolation.py` |
| Extended Data Fig. 4 | `derive_figure_ed4_fatpad.py` (one knee's built model) | | `figure_ed4_fatpad.py` |
| Extended Data Fig. 5 | `derive_figure_ed5_data.py` (after `derive_figure8_data.py`) | | `figure_ed5_cartilage_regions.py` |
| Extended Data Table 1 | `derive_table_ed1_cohort.py` writes the finished table | | |

Derivations are in `pipeline/`, models and renders in `figures/`. Notes per chain:

- **Figure 3.** `derive_figure3_data.py` reads the synthetic simulations
  (`02_simulations/synthetic_shape_axis/`, made by `pipeline/run_synthetic.py`); `derive_figure3_geometry.py`
  writes `shape_metrics.csv` and the `morph/*.png` anatomy renders. The render reads its shape-axis window
  from the model's `mixed_model_meta.json`, so it cannot plot a range the model was not fitted on. The
  synthetic knees are labelled `fig5_…`, the figure's number in an earlier draft; the labels are part of the
  released tree.
- **Figure 5.** `spm` runs the surface permutation tests as 40 independent, seeded cells
  (`--cell <0..39>`, about 5 h in total; needs pyKneeSPM, `docs/environment.md`); `maps` applies the
  quality control and writes the surfaces, renders and cluster tables. The render reads only what `maps`
  wrote, so re-rendering cannot change the statistics.
- **Figure 6.** The panel reports a correlation across the nine measured gait conditions and does not
  read the Mundlak model, so it renders without R. The model is the reported inference: it keeps every
  simulation and separates the between-condition from the within-condition effect.
- **Figure 7.** 44 feature configurations, each fitted under each of the ten gait conditions (the paper's
  run: 3 h 9 min on ten workers, `--workers`, with 64 GB requested; peak memory was not recorded). Each condition's cells are computed from that condition's own
  data. The classifiers depend on the CPU type (`docs/reproduce.md`).
- **Extended Data Table 1** needs a local OAI "Complete Data - ASCII" download (`$OAI_CLINICAL_ROOT` or
  `oai_clinical_root.txt`); only aggregates are written, to `04_results/table_ed1_cohort/`. It re-derives
  the group definitions (baseline KL grade and OAI's adjudicated incident OA within 48 months) from the OAI files and raises if `00_inputs/groups.json` violates it.

## Each figure's n is its own

Populations differ between figures for stated reasons; do not carry one figure's n to another.
- Figure 2 reads `joint_mechanics.h5` directly and needs only a stance waveform, so knees without
  follow-up MRI count (`coverage.json`, `ensemble_spm_{bone}.json`). Its per-knee ensemble keeps a knee
  simulated in at least five gait conditions.
- Figure 5 counts per gait arm, each dropping its own flagged simulations; the thickness-change maps apply
  no pressure quality control (`population.json`, `n_knees` in each cluster table).
- Figure 6's unit of observation is the gait condition (n = 9): adduction moment and body mass belong
  to the donor, not the knee. Knee and simulation counts are in `derivation_meta.json` and `gait_counts.csv`.
- Figure 7 is a two-group classifier, so the mild-OA group is absent by design (counts per configuration
  in `auroc.csv`).
- Figure 8 reads transferred pressure and per-visit thickness, so its population follows follow-up MRI
  availability and differs between its two intervals (`meta_{bone}.json`).

## Layout and conventions

```
figures/
  shared/palette.py   colours, group labels, outcome-axis wording
  shared/panels.py    panel letters
  shared/qc.py        the shared pressure-outlier table and inclusion manifests
  shared/results.py   locating 04_results and the output folder; require() for missing inputs
  assets/             vendored images (assets/figure1/README.md lists them)
  output/             renders (gitignored)
```

- Colours, labels, panel letters and quality control come from `shared/`, never a local copy.
- Paths resolve through `comak_knee_oa.paths`; `tests/test_no_absolute_paths.py` enforces it.
- A missing input goes through `shared.results.require(..., produced_by=...)`, so the error names the
  command that builds it.
- Every figure that filters simulations writes an inclusion manifest, from the code path that did the
  filtering.
- Every random draw is seeded, including library defaults (for example scikit-learn's randomized `PCA`
  solver), so a re-derivation on the same CPU type reproduces the paper's numbers to the digit (Figure 7's
  classifiers move at the third decimal between CPU types, `docs/reproduce.md`).
- The gait-cycle key of Figure 2 (`assets/gait_stance_silhouettes.png`) was keyed from an AI-generated image.
