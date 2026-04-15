#!/bin/bash
#SBATCH -A llmservice_nemo_speechlm
#SBATCH -p batch
#SBATCH --job-name=granary-v2-postprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH -t 4:00:00
#SBATCH --output=/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/Curator/logs/%j_postprocess.out
#SBATCH --error=/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/Curator/logs/%j_postprocess.err
#SBATCH --container-image=/lustre/fsw/llmservice_nemo_speechlm/users/nkoluguri/containers/curator-nightly-lhotse.sqsh
#SBATCH --container-mounts=/lustre/fsw/convai_convaird_nemo-speech:/lustre/fsw/convai_convaird_nemo-speech,/lustre/fsw/llmservice_nemo_speechlm:/lustre/fsw/llmservice_nemo_speechlm

set -euo pipefail

CURATOR_DIR="/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/Curator"
FASTTEXT_MODEL="/lustre/fsw/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/postprocess/fleurs/cache/lid.176.ftz"
INPUT_DIR="${1:?Usage: sbatch run.sh <input_dir> <output_dir> [extra pipeline args]}"
OUTPUT_DIR="${2:?}"
shift 2
EXTRA_ARGS=("$@")   # e.g. --manifest /path/to/shard_0.jsonl

echo "Input dir    : ${INPUT_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Node         : $(hostname)"
echo "Started      : $(date)"

export PYTHONPATH="${CURATOR_DIR}:${PYTHONPATH:-}"

cd "${CURATOR_DIR}"
python tutorials/audio/granary_v2_postprocessing/pipeline.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --fasttext_model "${FASTTEXT_MODEL}" \
    "${EXTRA_ARGS[@]}"

echo "Finished : $(date)"
