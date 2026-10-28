#!/bin/bash
# Build Figure 3's synthetic knees on GPU, one array task per knee.
#
# Submit from the repository root (paths below are relative to it):
#   mkdir -p logs
#   PINNED_PYTHONPATH="$(cd src && python -m comak_knee_oa.pins pythonpath --dest <dir>)" \
#     CONDA_ENV=<env> sbatch --array=0-9 slurm/synthetic_build.sbatch
#
# `python pipeline/run_synthetic.py labels --config ...` lists index -> knee.
# CONDA_ENV defaults to comak-knee-oa (docs/environment.md).
# PINNED_PYTHONPATH must put all four pinned library exports first -- NSM,
# nsosim, pycomak and pymskt, as `pins pythonpath` prints them -- because
# the job refuses to start unless `pins check --strict` passes, and the run record
# reads each version from its export. Make the exports once with
#   (cd src && python -m comak_knee_oa.pins export --dest <dir> --repo NSM=<checkout> \
#       --repo nsosim=<checkout> --repo pycomak=<checkout> --repo pymskt=<checkout>)
# With a JAM build of your own, also export COMAK_PINS_ACCEPT=libosimJAM.so=<its md5>
# (docs/environment.md); sbatch passes the environment through.
# Memory: a build peaks at about 2 GB of host memory (8 GB requested).
#SBATCH --job-name=synth_build
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%A_%a.log

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
CONFIG="${CONFIG:-config/synthetic_shape_axis.json}"

# Activation hooks read unset variables, so -u only after activating.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-comak-knee-oa}"
set -u
export PYTHONPATH="${PINNED_PYTHONPATH:?set PINNED_PYTHONPATH to the pinned library exports}${PYTHONPATH:+:$PYTHONPATH}"
# Refuse to run on any library commit or solver build other than the pinned ones.
(cd src && python -m comak_knee_oa.pins check --strict)

# Rows are "index<TAB>label"; select by index so library chatter on stdout
# cannot shift which knee a task builds.
LABEL=$(python pipeline/run_synthetic.py labels --config "$CONFIG" \
    | awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" '$1 == i {print $2}')
[[ -n "$LABEL" && "$LABEL" == fig5_* ]] || { echo "no label for task ${SLURM_ARRAY_TASK_ID}: '${LABEL}'" >&2; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID} -> ${LABEL} on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python pipeline/run_synthetic.py build --config "$CONFIG" --label "$LABEL"
