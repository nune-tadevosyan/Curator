#!/bin/bash
#SBATCH -A llmservice_nemo_speechlm
#SBATCH -p batch
#SBATCH --job-name=granary-v2-postprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH -t 4:00:00
#SBATCH --output=/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/Curator/logs/%A_%a_postprocess.out
#SBATCH --error=/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/Curator/logs/%A_%a_postprocess.err
#SBATCH --container-image=/lustre/fsw/llmservice_nemo_speechlm/users/nkoluguri/containers/curator-nightly-lhotse.sqsh
#SBATCH --container-mounts=/lustre/fsw/convai_convaird_nemo-speech:/lustre/fsw/convai_convaird_nemo-speech,/lustre/fsw/llmservice_nemo_speechlm:/lustre/fsw/llmservice_nemo_speechlm

# Usage:
#   # 1. Generate the list of input subdirectories:
#   ls -d /path/to/results_large_scale_6/*/ > dirs.txt
#
#   # 2. Submit — the array range is set automatically:
#   N=$(($(wc -l < dirs.txt) - 1))
#   sbatch --array=0-${N}%32 run_array.sh dirs.txt /path/to/output_dir
#
#   %32 caps concurrent jobs to 32 at a time (remove or raise if your cluster allows more).

set -euo pipefail

CURATOR_DIR="/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/Curator"
FASTTEXT_MODEL="/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/postprocess/fleurs/cache/lid.176.ftz"

DIRS_FILE="${1:?Usage: sbatch --array=0-N run_array.sh <dirs.txt> <output_dir>}"
OUTPUT_DIR="${2:?}"

# Pick this task's input directory from the list (1-indexed in the file)
INPUT_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${DIRS_FILE}")
INPUT_DIR="${INPUT_DIR%/}"  # strip trailing slash if present

if [[ -z "${INPUT_DIR}" ]]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID}: no directory assigned, exiting."
    exit 0
fi

echo "Array task  : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX}"
echo "Input dir   : ${INPUT_DIR}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Node        : $(hostname)"
echo "Started     : $(date)"

export PYTHONPATH="${CURATOR_DIR}:${PYTHONPATH:-}"

cd "${CURATOR_DIR}"
python tutorials/audio/granary_v2_postprocessing/pipeline.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --fasttext_model "${FASTTEXT_MODEL}"

echo "Finished : $(date)"
