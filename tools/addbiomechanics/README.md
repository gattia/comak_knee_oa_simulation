# Running AddBiomechanics locally

The measured-gait simulations scale every knee's body to a gait donor. The donor's body comes from
[AddBiomechanics](https://github.com/keenon/AddBiomechanics) (AB), run on this machine rather than on the website,
so the exact engine is recorded. This folder holds what the engine needs beyond its own source;
`pipeline/run_addbiomechanics.py` uses it. How the body is then applied is in `docs/scaling_comak_model.md`.

> AB is not deterministic. Its first inverse-kinematics step uses random restarts and exposes no seed. Repeat
> runs on identical inputs, per donor: Subject_119, seven runs (the vendored body among them), knee scale factor
> `s_wa` SD 0.25 %, range 0.77 %, the six repeats −0.56 % to +0.21 % from the vendored body; Subject_121, two
> repeats, −0.38 % and +0.56 % from the vendored body, 0.94 % apart. Contact pressure moves about 1.5 times as much
> as `s_wa`, the same for every knee of that donor. So the vendored AB output in the data root
> (`00_inputs/gait/addbiomechanics/<donor>/`, an md5 for every file) is the input of record. Re-running AB gives a
> slightly different body; to reproduce a published simulation, start from the vendored files.

## What is here

| file | what it is | why |
|---|---|---|
| `environment.yml` | the engine's environment (`ab_engine`) | exported from the environment that scaled the paper's donors: nimblephysics 0.10.52.1, numpy 1.24.4 |
| `opensim_environment.yml` | stock OpenSim 4.5.2 (`ab_opensim`) | the engine writes every scaled model it outputs, `match_markers_and_physics.osim` included, by running `opensim-cmd run-tool`. Separate environment: its Python (3.11) is not the engine's (3.9) |
| `bin/opensim-cmd` | a three-line wrapper | gives the engine that one program, from the environment named by `AB_OPENSIM_PREFIX`, without putting the whole environment on `PATH` |
| `stub/opensim.py` | an empty `opensim` module | the engine imports `opensim` at module level for its Moco pass, which is off here. Any use raises, so nothing can run on the stand-in by accident |

The engine source itself is not copied here. It is pinned:

| | |
|---|---|
| repository | `https://github.com/keenon/AddBiomechanics` |
| commit | `8fe69cd8be3786a13e0cb90f4451fb14d7e3cb5c` |
| nimblephysics | `0.10.52.1` |
| `opensim-cmd --version` | `4.5.1-2025-04-01-5cf2dacc8` (the paper's) or `4.5.2-2025-05-02-fcdbeacb2` |

`run_addbiomechanics.py run` checks the commit, nimblephysics and `opensim-cmd`, and refuses to start on a difference (`--allow-unpinned` overrides
it, and the run record lists every difference). Why two OpenSim builds: *The OpenSim build* below.

## Setting up (once)

Everything AB reads and writes lives in a work folder outside the repository, `$W` below; the commands themselves
run from the repository root. (The engine imports `stub/opensim.py`, so Python may leave a `__pycache__/` there,
and OpenSim appends its log to `opensim.log` in the current directory; git ignores both. To keep the log out of the
checkout, see `docs/reproduce.md`, *Prerequisites*.)

```bash
W=<a work folder outside the repository>

# 1. The engine, at the pinned commit (about 650 MB)
mkdir -p $W/AddBiomechanics
git -C $W/AddBiomechanics init -q
git -C $W/AddBiomechanics remote add origin https://github.com/keenon/AddBiomechanics.git
git -C $W/AddBiomechanics fetch --depth 1 origin 8fe69cd8be3786a13e0cb90f4451fb14d7e3cb5c
git -C $W/AddBiomechanics checkout -q FETCH_HEAD

# 2. The two environments
conda env create -f tools/addbiomechanics/environment.yml            # ab_engine (pulls torch's CUDA wheels, ~4 GB)
conda env create -f tools/addbiomechanics/opensim_environment.yml    # ab_opensim
```

## The custom skeleton

AB scales the COMAK model itself, with its knee stripped to a hinge and the gait protocol's markers added
(`docs/scaling_comak_model.md` explains why each change is needed). The paper's skeleton is built from nsosim's
marker model and is reproducible byte for byte:

```bash
# in the pipeline environment (comak-knee-oa, activated), with the pinned libraries on PYTHONPATH
# (docs/reproduce.md, "Prerequisites")
mkdir -p $W/marker_model
cp <nsosim at its pin>/tests/fixtures/osim_models/full_body_healthy_knee_armless_trc.osim $W/marker_model/
ln -s "$COMAK_DATA_ROOT/00_inputs/reference/data/reference_data/comak_models/current/Geometry" $W/marker_model/Geometry
python pipeline/prepare_addbiomechanics_skeleton.py \
    --model $W/marker_model/full_body_healthy_knee_armless_trc.osim \
    --out $W/comak_ab_skeleton.osim \
    --ab-geometry $W/AddBiomechanics/server/data/Geometry
md5sum $W/comak_ab_skeleton.osim    # be9a614e5a1f89bae5278ac688bfdc5a, the skeleton the paper's donors were fitted on
```

The `Geometry` link is needed because nsosim does not track the marker model's meshes. They are the 163 meshes of
the reference model, identical file for file.

## Running one donor

```bash
python pipeline/run_addbiomechanics.py stage --donor Subject_119 --skeleton $W/comak_ab_skeleton.osim \
    --zips '<provider zips>/OA_GaitRetrainingAddB-*.zip' --out $W/runs/Subject_119
OMP_NUM_THREADS=8 <ab_engine>/bin/python pipeline/run_addbiomechanics.py run $W/runs/Subject_119 \
    --engine $W/AddBiomechanics/server/engine --python <ab_engine>/bin/python --opensim-prefix <ab_opensim>
python pipeline/run_addbiomechanics.py vendor $W/runs/Subject_119 \
    --dest "$COMAK_DATA_ROOT/00_inputs/gait/addbiomechanics/Subject_119"

# the knee scale factor of the vendored body and of your run, and their difference
python pipeline/run_addbiomechanics.py knee-factor <downloaded root>/00_inputs/gait/addbiomechanics/Subject_119 \
    $W/runs/Subject_119
```

- `<ab_engine>` and `<ab_opensim>` are the two environments' prefixes (`conda env list` shows them).
- `stage` reads the donor's recorded mass, height, sex and age from `config/measured_gait/<donor>.json`
  (`addbiomechanics`) and its baseline trial from the provider's zips. The zips are the donors' raw motion data,
  which this project does not redistribute.
- `run` takes about 30 minutes on 8 cores. The paper's runs requested 32 GB; the engine was sampled at about 2 GB.
  It writes `ab_run_record.json`: the engine commit, nimblephysics and `opensim-cmd` versions, the command, host,
  CPU and every input's md5.
- `vendor` refuses to write into an existing folder.
- Check `_results.json` (`dynamicsStatus` per segment) before using a segment. AB cuts a trial into segments of
  about 20 s, numbered from 0 (`markers_segment_0`, …); every donor's config uses segment 3, the fourth (60–80 s).
  AB's own quality check can drop a whole segment (`docs/scaling_comak_model.md`, step 4).
- `knee-factor` (pipeline environment) prints `s_wa`, the knee scale factor the simulations apply: the mean of the
  femur and tibia long-axis factors, each the body's mesh scale over the same mesh's scale in
  `unscaled_generic.osim`. It prints the numbers, not a verdict; given the repeat runs above, a new run is
  ordinarily within about 1 % of the vendored body (`docs/reproduce.md`, section 4).

## The OpenSim build

The OpenSim team's conda channel replaced the paper's build with a newer one under the same package name and
version (`opensim-4.5.2-py311np125`): the paper's package (md5 `e245992db710df25418456eddefd3850`) reports
`4.5.1-2025-04-01-5cf2dacc8`, and the one `opensim_environment.yml` installs now (md5
`dd944e018b3d998bb321a1e6e8b30add`) reports `4.5.2-2025-05-02-fcdbeacb2`. The paper's build cannot be installed
by name.

The engine runs `opensim-cmd` twice per donor, to write the scaled model after its kinematics and its dynamics
passes, and for nothing else. Both calls of a Subject_119 run, captured and replayed on each build, gave
byte-identical scaled models. So both builds are accepted, and the run record names the one used.
