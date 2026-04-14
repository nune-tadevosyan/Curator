#!/bin/bash
#SBATCH -A llmservice_nemo_speechlm
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH --job-name=granary-v2-postprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH -t 4:00:00
#SBATCH --output=/lustre/fsw/portfolios/convai/users/ntadevosyan/projects/granary-v2-asr/pipeline/logs/%j_postprocess.out
#SBATCH --error=/lustre/fsw/portfolios/convai/users/ntadevosyan/projects/granary-v2-asr/pipeline/logs/%j_postprocess.err

set -euo pipefail

CURATOR_DIR="/lustre/fs11/portfolios/convai/projects/convai_convaird_nemo-speech/users/ntadevosyan/projects/granary-v2-asr/curator/Curator"
INPUT_CONFIG="${1:?Usage: sbatch run.sh <input_config.yaml> <output_dir>}"
OUTPUT_DIR="${2:?}"

echo "Input config : ${INPUT_CONFIG}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Node         : $(hostname)"
echo "Started      : $(date)"

cd "${CURATOR_DIR}"
uv run python tutorials/audio/granary_v2_postprocessing/pipeline.py \
    --input_config "${INPUT_CONFIG}" \
    --output_dir "${OUTPUT_DIR}"

echo "Finished : $(date)"
