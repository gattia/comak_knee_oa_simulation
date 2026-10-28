# Scaling the COMAK model to a new gait subject

How the measured-gait models get their body scaling. Scale factors come from AddBiomechanics (AB), run
on **the COMAK model itself** (knee stripped to a hinge, your markers added). AB's per-segment factors are then
ratios against COMAK's own segments, which is what `nsosim` assumes when it applies them.

AB must scale this model, not AB's stock `ArmlessRajagopal` skeleton. That skeleton carries the same marker
names, so applying its factors to COMAK raises nothing, and on the paper's donors it would make the knee about 3 %
smaller and contact pressure about 6 % higher, the same for every knee of a donor. `nsosim.scaling` refuses that mix-up (step 5).

## 1. Base model

`00_inputs/reference/data/reference_data/comak_models/<version>/full_body_healthy_knee.osim`. Use the
version named by `base_model` in `config/reconstruction.json`. The knee is stripped in step 3, so only the
non-knee segments matter.

## 2. Add your marker set

AB matches your `.trc` marker names against the model's markers, so the model needs a marker on the right body
for every marker in your protocol.
- Place markers on the COMAK segments (OpenSim GUI, or the API). Do not copy coordinates from another
  model: segment lengths differ.
- Use the exact `.trc` names. Leave markers you are unsure of `fixed=false`, so AB can refine them.
- Remove the arms if your protocol has no arm markers.
- Worked example, for the marker set of the gait-retraining trial whose donors the paper uses: the script
  `build_smith2019_armless_trc.py` (`smith2019` is the OpenSim-JAM full-body knee model the COMAK model
  derives from) in [nsosim](https://github.com/gattia/nsosim) (under `scripts/`, at the pinned
  commit). It removes the arms, renames and mirrors COMAK markers, and saves
  `full_body_healthy_knee_armless_trc.osim` (in nsosim's `tests/fixtures/osim_models/`).

## 3. Strip the knee and make it loadable by AB

```bash
# in the pipeline environment (comak-knee-oa), activated or through `conda run`
python pipeline/prepare_addbiomechanics_skeleton.py \
    --model <work dir>/my_markers.osim --out <work dir>/my_ab_skeleton.osim \
    --ab-geometry <work dir>/AddBiomechanics/server/data/Geometry
```

AB does not use your file as written. It rewrites the joints (`rationalizeJoints` in nimblephysics, AB's physics engine), it can load meshes
only from its own Geometry folder, and it records each body's scale only in that body's meshes. Each step below
fixes something that otherwise breaks without an error. Every step except the meshes is exact.

| step | without it |
|---|---|
| Strip the COMAK knee (`nsosim.knee_assembly.strip_comak_knee`), leaving a `knee_r` hinge | – |
| Rebuild that hinge from the input knee | `strip_comak_knee` alone puts the tibia 12 mm posterior, makes positive flexion a hyperextension, and gives no range limits |
| Flatten `MultiplierFunction`s | – (exactness; AB does not apply them) |
| Give `knee_l`/`pf_l` explicit offset frames | AB's rewrite drops the left knee's 0.374 m femur offset: the left tibia sits at the hip |
| Rewrite slope −1 rotations as +1 about the negated axis | AB's rewrite flips the left knee's sign: it bends backwards |
| Linearise 2-knot splines | AB fails to load ("Invalid axis order for Euler joint") |
| Attach AB foot meshes (`r_foot`, `r_bofoot`, left) | AB finds no foot contact points; the dynamics fit crashes (`vector::_M_default_append`) |
| Attach AB femur/tibia meshes to `femur_r`/`tibia_r` | AB records no scale for them (they lost their meshes with the knee bodies) |
| Attach AB skull/jaw, fitted onto COMAK's own skull | AB measures the generic as 0.58 m tall; the height prior and its first scaling bounds go wrong |

It then verifies the result against the **input** model (every body and marker, at rest and flexed) and names the
model `<input name>_ab_skeleton`. Run it in the activated environment or through `conda run`: both set
OpenSim's library path, and the environment's `python` called directly does not.

**The paper's skeleton** is rebuilt byte for byte by this script from nsosim's
`tests/fixtures/osim_models/full_body_healthy_knee_armless_trc.osim`, at the pinned nsosim commit (md5
`be9a614e5a1f89bae5278ac688bfdc5a`; the exact commands are in `tools/addbiomechanics/README.md`).

**A new skeleton** (another marker set) should be checked with AB's own loader before use: after AB's
`rationalizeJoints`, every marker must match OpenSim's within 0.001 mm at rest and flexed, and AB should measure the
generic's height (≈1.56 m for COMAK) and find 5 + 5 foot contact points. These checks are not scripted in this
repository.

The paper's skeleton passes both checks. It was also tested end to end on a synthetic subject:
- truth: COMAK scaled by known factors, moved through gait and range of motion;
- AB's marker RMSE was 2.2 mm;
- the knee factor was within 0.33 % of truth;
- knee IK was within 1.7° RMS;
- the scaled COMAK model's markers, at the true pose, were within 4.3 mm RMS of the true markers.

With AB's stock Rajagopal skeleton the same test gives an AB marker RMSE of 19 mm, and, once its factors and markers
are applied to COMAK, 37 mm RMS between the scaled COMAK model's markers and the true ones.

## 4. Run AddBiomechanics

**Locally (how the paper's models were made).** `pipeline/run_addbiomechanics.py` runs the AddBiomechanics engine (a
checkout of `keenon/AddBiomechanics` at a pinned commit, `server/engine`). Its two environments, the
`opensim-cmd` wrapper it needs and the engine pin are in `tools/addbiomechanics/` (its README has the setup):

```bash
python pipeline/run_addbiomechanics.py stage --donor Subject_119 --skeleton <work dir>/my_ab_skeleton.osim \
    --zips '<provider zips>/OA_GaitRetrainingAddB-*.zip' --out <work dir>/runs/Subject_119
# or, for your own subject's walking trial (a marker .trc and a force-plate .mot):
python pipeline/run_addbiomechanics.py stage --donor <subject> --skeleton <work dir>/my_ab_skeleton.osim \
    --markers <your trial>.trc --grf <your trial>_grf.mot --out <work dir>/runs/<subject>
OMP_NUM_THREADS=8 <ab_engine>/bin/python pipeline/run_addbiomechanics.py run <work dir>/runs/Subject_119 \
    --engine <work dir>/AddBiomechanics/server/engine --python <ab_engine>/bin/python --opensim-prefix <ab_opensim>
python pipeline/run_addbiomechanics.py vendor <work dir>/runs/Subject_119 \
    --dest <data root>/00_inputs/gait/addbiomechanics/Subject_119
python pipeline/run_addbiomechanics.py knee-factor <data root>/00_inputs/gait/addbiomechanics/Subject_119
```

`<work dir>` is a folder outside the repository; `<ab_engine>` and `<ab_opensim>` are the prefixes of the two
environments in `tools/addbiomechanics/`. `knee-factor` prints the knee scale factor `s_wa` (step 6).

- The subject entries (mass, height, sex, age) are read from the `addbiomechanics` section of `config/measured_gait/<donor>.json`:
  the recorded values, not an estimate. Dynamics is on. **For your own subject, write that config first** (step 6.1):
  `stage` reads it.
- Both forms of `stage` build the same run folder; the stage record names the zip member or the file used.
- The trial is staged as `trials/markers/`, so the outputs are named `markers_segment_<n>` like the website's.
- `run` writes `ab_run_record.json`: engine commit (and whether its tree was dirty), nimblephysics and `opensim-cmd`
  versions, the exact command, host, CPU and the inputs' md5s. It refuses to start on an engine commit,
  nimblephysics or `opensim-cmd` other than the paper's, unless `--allow-unpinned` (recorded). `vendor` copies
  `Models/`, `MarkerData/`, `ID/`, `IK/` and the results archive, with an md5 for every file.
- The engine writes every scaled model by running `opensim-cmd run-tool`, and imports `opensim` for a pass that is
  switched off. `run` supplies both from `tools/addbiomechanics/` (`bin/opensim-cmd`, `stub/opensim.py`).
- **AB is not run-to-run deterministic.** Its initial inverse kinematics uses random restarts
  (`initialIKRestarts = 150` in the engine) and nimblephysics threads internally; no seed is exposed. Repeat runs on
  identical inputs differ by about 1 % in `s_wa` (`tools/addbiomechanics/README.md`). Single factors on axes no marker constrains moved up to 7 %
  (the talus of Subject_121, in one of its two repeats, which were made from a fresh installation). Runs differ
  even on one CPU type with one thread. Through the contact model, ±0.2 % in `s_wa` is about ±0.3 % in contact
  pressure, the same for every knee of that donor. So **the vendored output (md5 per file) is the input of record**:
  re-running AB gives a slightly different body. To reproduce a result, start from the vendored files, not from a
  new AB run.
- **AB's own quality checks can drop a whole segment.** If the centre of pressure falls more than 1 cm outside the
  foot on average, AB marks every frame of that segment "missing GRF" and writes no `ID/` files for it. Check
  `_results.json` (`dynamicsStatus` per segment) before pointing a config at a segment. In the paper's run this
  happened to Subject_167's segment 3, whose mean centre-of-pressure distance exceeded the 1 cm limit by only
  0.04 mm. Its force files came from an earlier AB run of the same trial, as that donor's `ab_vendor_supplement.json`
  records.

**On the website**, the equivalent is:

1. On [app.addbiomechanics.org](https://app.addbiomechanics.org), create a subject. Set:
   - mass, height and sex;
   - skeleton = **Custom** (upload `my_ab_skeleton.osim`).
   - Dynamics on or off changes the per-body masses that `nsosim` later transfers (step 6). With dynamics on, the
     foot meshes from step 3 are required.
2. Upload the trials: a static trial if you have one, plus walking `.trc` and `.mot` GRF files.
3. Process, then download the results zip.

Missing `Geometry/*.vtp.ply` warnings in AB's log are harmless; they are display meshes.

## 5. Vendor the download; the skeleton is checked for you

Unzip to `<run root>/00_inputs/gait/addbiomechanics/<subject>/` and keep AB's `Models/` folder intact. `Models/unscaled_generic.osim`
is AB's copy of the skeleton it scaled.

`nsosim.scaling.scale_comak_model` compares it with the model being scaled before doing anything. It checks:
- femur hip-to-knee and tibia knee-to-ankle, as joint-centre vectors;
- hip half-width;
- a tolerance of 1 %.

If the generic is another skeleton (e.g. `ArmlessRajagopal`, `walker_knee_r`), or if it is missing, scaling stops
with an error that lists both sets of landmarks.

Pass `allow_generic_mismatch=True` only when the mismatch is intended, e.g. reproducing models built on that
generic. The override and both sets of landmarks are written to the scaling report's
`ab_generic_check`.

## 6. Use it in the pipeline

1. Add `config/measured_gait/<subject>.json`, copying an existing one (e.g. `Subject_121.json`). Change:
   - `run.gait` to `<subject>`, and the subject folder in `run.results_dir` and `model.scaled_model`;
   - `addbiomechanics`: your subject's recorded `mass_kg`, `height_m`, `sex`, `age_years`, and `trial` (a label for
     the walking trial; with `--markers`/`--grf` it is only recorded);
   - `measured_gait.body_model`: `00_inputs/gait/addbiomechanics/<subject>/Models/match_markers_and_physics.osim`;
   - `measured_gait.markers`, `grf`, `external_loads`: AB's files for the segment you simulate:
     `MarkerData/markers_segment_<n>.trc`, `ID/markers_segment_<n>_grf.mot`,
     `ID/markers_segment_<n>_external_forces.xml`. AB cuts a trial into ~20 s segments numbered from 0; pick one
     whose `_results.json` entry shows dynamics finished;
   - `measured_gait.stance_index`: which stance in that segment to simulate, counted from 0. A stance is a stretch of
     at least `min_stance_frames` frames where `grf_vertical_column` (the right foot, `ground_force_calcn_r_vy`)
     exceeds `force_threshold_N`. Every knee is a right knee in the model, so use a right-foot stance.
   - Keep `scaling.mass_rule: "volume"` and `comak_timing.comak_window_pad_s: 0.005`: both are required, and
     `run_knee.py simulate --check` refuses a config without them. The other sections (`comak_timing`, `timeouts`,
     `comak_optimizer`, `muscle_weights`, `model_settling`) are the paper's solver settings; leave them.
   - Then run AB (step 4) and vendor its output into your run root (`--dest
     <run root>/00_inputs/gait/addbiomechanics/<subject>`).
   - Declare the subject in the run root: `PYTHONPATH=src python -m comak_knee_oa.new_root create --root <run root> --run-name
     <name> --baseline <downloaded root> --measured-gaits <subject>[,<subject 2>…]` (without
     `--link-addbiomechanics`). A root's subject list is fixed when the root is made; make a new root to add one.
2. `nsosim.scaling` reads the per-segment factors from `body_model`:
   - each body's mesh scale divided by the same mesh's scale in AB's `Models/unscaled_generic.osim`. AB writes
     (generic mesh scale) × (body scale), and COMAK's generic mesh scales are not 1 (femur 0.925);
   - non-knee bodies take AB's factors;
   - every knee body, including the patella (which the stripped skeleton lacks), takes one isotropic factor: the
     mean of the femur and tibia long-axis factors (`scale_factors.py`);
   - COMAK bodies that AB's skeleton lacks and that are not knee bodies (the arms, if you removed them) keep
     reference size;
   - AB's marker set is copied onto the scaled model as is. That is correct only because AB fitted it on this same
     skeleton.
3. Masses (`scale_comak_model(..., mass_rule=...)`). The total is AB's physics-pass total (the sum of `body_model`'s
   masses) unless `subject_mass` is passed.
   - `mass_rule="volume"` (the paper's models): OpenSim's volume rule on the COMAK model's own masses,
     m_i = base_i · (sx·sy·sz)_i · c, normalised to the total. AB's per-body masses are this rule on AB's
     generic, so the bodies AB modelled keep AB's ratios; the bodies AB's skeleton lacks take their share out of the
     same total (arms: factor 1; knee sub-bodies, patella and menisci: `s_wa³`).
     - **Bodies no marker observes** borrow a neighbour's factors **for mass and inertia only**. AB returns factors
       for them that nothing constrains: with the COMAK skeleton the head comes back at 1.4–1.9× its volume. The
       list comes from AB's marker set (`nsosim.scaling.derive_mass_proxies`): an observed child if there is one,
       otherwise the nearest observed ancestor. For the paper's donors' 41-marker set: head from torso, talus and toes from calcaneus,
       left patella from left femur.
     - Their geometry and joint frames keep AB's factors, so AB's marker fit is unchanged.
     - The report records `unobserved_bodies` and `mass_proxies`, and per proxied body its proxy and AB's own volume
       factor. `mass_proxies={}` turns this off.
   - `mass_rule="as_run"` (the library default): AB's per-body masses are imported for the bodies AB modelled and the
     total is renormalised. Because AB's masses already sum to the total, the bodies AB does not model are counted on
     top, and every body shrinks (by about 15 %). It reproduces models built with that rule.
   - Pass `recorded_mass_kg` (the mass typed into AB) to have the report record AB's total minus it, a sanity check.
   - The scaling report's `per_body_mass_audit` records every body's base mass, factor and final mass.
   - In the pipeline, a measured-gait config must name the rule (`scaling.mass_rule`). A missing key is an error,
     never a silent fall-back to the library default.
4. Simulate, with the run root as `COMAK_DATA_ROOT`: `python pipeline/run_knee.py simulate --config
   config/measured_gait/<subject>.json --knee <knee>` (after `--check`; then `verify`), and the later steps of
   `docs/reproduce.md` section 3 with the same config.
