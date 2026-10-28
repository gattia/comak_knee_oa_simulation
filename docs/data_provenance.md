# Data provenance

Where the inputs come from, which segmentation produced which tissue, and what the pipeline does and
does not regenerate.

## Segmentations and meshes

Bone/cartilage and the menisci come from two different segmentation networks, which the mesh file
names do not show:

| tissue | mesh files | segmentation label map |
|---|---|---|
| bone + cartilage (femur, tibia, patella) | `{subj}_{side}_{femur,tibia,patella}.vtk`, `*_cart.vtk` | `SAG_3D_DESS_{SIDE}_orig_med_lat_tib_cart_no_correction-label.nrrd` |
| menisci | `{subj}_{side}_{med,lat}_men.vtk` | `SAG_3D_DESS_{SIDE}_bones_cart_men_fib-label.nrrd` (label 10 = medial, 9 = lateral) |

`bones_cart_men_fib` also contains bone and cartilage; those labels are not used. Bone and cartilage
come from the `orig_med_lat_tib_cart_no_correction` network, which does not segment the menisci. The
longitudinal shape-model fits used for cartilage thickness rest on the same
`orig_med_lat_tib_cart_no_correction` segmentation at every visit, so simulation geometry and thickness
measurement share one segmentation.

| meshing parameter | bone | cartilage and menisci |
|---|---|---|
| Gaussian smoothing variance | 0.5 mm | 0.15625 mm |
| decimation | 50 % | 50 % |
| medial–lateral crop | 80 % | 80 % |
| resampling | 20,000 clusters | 20,000 clusters |

## Gait conditions

Ten gait conditions are simulated on every knee:

| | source | protocol | speed | stances per trial |
|---|---|---|---:|---:|
| reference | published reference dataset, one healthy adult | overground | 1.291 m/s | 1 |
| nine measured | gait-retraining trial, knee osteoarthritis | treadmill | 0.688–1.350 m/s | 16–21 |

The protocol is determined from the marker data. During foot-flat the foot does not move relative to
the surface it stands on, so a stationary foot marker means stationary ground and a moving one a belt at
that speed: 0.009 m/s for the reference, 0.688 m/s and above for the nine. `pipeline/describe_gait_conditions.py`
reproduces this, and recovers the trials' own recorded treadmill speeds to about 0.012 m/s.

The reference lies inside the range of the nine on stride time, cadence, stance duration, stance fraction
and speed, and outside it on one measure, peak load: 1.30 body weights against 0.95–1.13. Speed accounts
for roughly a sixth of that difference; protocol and population account for the rest and cannot be
separated, because every treadmill condition comes from the trial and the only overground one does not.
The gait cycle therefore has the same shape across conditions and differs in magnitude.

Body weight is taken from the force data: over a complete stride the mean total vertical ground reaction
force equals body weight. Loads are reported against it, so a mass recorded in the wrong units cannot
rescale a comparison.

## Reference surface

Subjects are compared on one 20,000-vertex cartilage surface decoded from the shape model's zero latent.
It depends on no subject, so vertex *i* is the same anatomical location for every subject and visit.

Each subject is related to it by a latent-space walk between the subject's fit and the zero latent, which
places every reference vertex on that subject's anatomy. Thickness and contact pressure are both carried
onto the reference surface through this correspondence. The correspondence is a function of the
subject's shape-model fit, so the fits are part of the data root and hashed in the stage manifests.

## Coordinate systems

Pressure is simulated on the subject's anatomy in one shape model; the reference surface belongs to
another (the femur models differ). A per-subject transform bridges them through the MRI coordinate
space. Pressure values belong to the simulation, but pressure on the reference surface also depends on
the fit, because the transform and the landing points derive from it.

Left knees are mirrored to right for analysis. Two mirroring conventions are involved, a volume
reflection when the segmentations were meshed and a reflection about the femur centroid in the
shape-model fits; a per-subject correction composes them rather than assuming a plain sign flip on x.

## What the pipeline derives

From the released inputs this repository produces:
- the knee models (`run_knee.py rebuild`), simulations (`run_knee.py simulate`) and synthetic knees
  (`run_synthetic.py`; built and simulated with the cohort's `config/reconstruction.json` and each donor's
  config, and `run_synthetic.py axis --verify` re-derives the shape axis from `01_models` and checks it);
- pressure on the reference surface (`derive_pressure.py`) and the combined matrices
  (`combine_reference_data.py matrices`, including `combined_data_shape.npz`);
- the quality-control table `03_derived/qc/pressure_outliers_unified_<run>.csv` (`derive_qc.py`), which
  every figure reading pressure takes its exclusions from, through `figures/shared/qc.py`;
- every `04_results` table and model (`figures/README.md`).

## Frozen inputs

These are released with the data and not regenerated by this pipeline:
- the MRI segmentations and surface meshes (Osteoarthritis Initiative);
- each knee's shape-model fits (`01_models/OARSI_May.22.2026_Full_run`). Fitting is stochastic, so a refit
  gives slightly different anatomy; the saved fits are the record;
- the reference model and its one-time preparation (wrap surfaces, ligament attachments, mean surfaces).
  Of its two full-body models, `comak_models/May.22.2026_Full_run/` is the `base_model` every knee model is
  built from (`config/reconstruction.json`); from `comak_models/current/` the gait configs read only
  `smith2019_reserve_actuators.xml`. Their `.osim` files differ;
- cartilage thickness and its change over time (`03_derived/thickness`, `03_derived/thickness_merged`) and
  the two thickness visits whose shape fits failed (`03_derived/qc/thickness_visit_exclusions.csv`);
- the transfer's per-knee transforms and shape walks (`03_derived/pressure_on_reference`);
- the anatomical region labels on the reference surface (`03_derived/region_labels`);
  `combine_reference_data.py labels` regenerates them only in a root that does not link them;
- `03_derived/qc/chronic_knees_manual.csv` is the release's hand-curated list for the `v3` run (one knee):
  knees dropped entirely, because a knee whose simulation never converges produces no pressure and no
  detector can flag it. A new run root starts without one; add knees after inspecting that run's
  simulations;
- the nine donors' AddBiomechanics outputs (`00_inputs/gait/addbiomechanics/<donor>/`, an md5 per file).
  AddBiomechanics is not deterministic, so these are the input of record (`docs/scaling_comak_model.md`).

## Surface permutation tests (Figure 5, Extended Data Figure 2)

`derive_figure5_data.py spm` runs 40 independent cells. Each cell's permutation null is seeded at
`SPM_SEED + cell index` (`SPM_SEED = 20260829`, 10,000 permutations), recorded in `spm/_seeds.json` and
as `permutation_seed` in every cell summary. A seed per cell, not one shared draw, because each cell's
cluster-extent threshold is the 95th percentile of its own null and the figure reads the cells as
independent tests; the cell order in `spm_cells()` therefore fixes every p-value. With the pinned
pyKneeSPM (`docs/pinned_versions.md`) two runs of all 40 cells were identical, map bytes included, also
on different compute nodes. The seed fixes which null is drawn, not how far a cluster near the threshold
could move under another seed.

## Sources

- MRI: Osteoarthritis Initiative, [nda.nih.gov/oai](https://nda.nih.gov/oai).
- Reference gait: [opensim-jam `jam-resources`](https://github.com/opensim-jam-org/jam-resources),
  `models/knee_healthy/experimental_data/motion_analysis/` (`overground_17.trc`, `overground_17_grf.mot`,
  `overground_17_ext_loads.xml`, `static1.trc`), byte-identical to its commit `73306a0`.
- Measured gait: AddBiomechanics outputs of nine donors from a gait-retraining trial; the raw motion data
  are not shared.
