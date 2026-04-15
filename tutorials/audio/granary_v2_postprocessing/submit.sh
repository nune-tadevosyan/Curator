#!/bin/bash
# Submit Slurm jobs for the postprocessing pipeline.
# Manifests are grouped into chunks of MANIFESTS_PER_JOB so the number of
# submitted jobs stays manageable even for large datasets.
#
# Usage:
#   bash submit.sh <output_dir> <input_dir_1> [input_dir_2 ...]
#
# Tune chunk size (default 8) via environment variable:
#   MANIFESTS_PER_JOB=16 bash submit.sh <output_dir> <input_dir>
#
# Multiple input dirs are submitted as sequential waves:
# wave N+1 starts only after every job in wave N finishes (afterany).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"

# Number of manifests processed per job. Raise for small manifests,
# lower for large ones. Each job uses 64 CPUs (set in run.sh).
MANIFESTS_PER_JOB="${MANIFESTS_PER_JOB:-128}"

OUTPUT_DIR="${1:?Usage: bash submit.sh <output_dir> <input_dir_1> [input_dir_2 ...]}"
shift
INPUT_DIRS=("$@")

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
    echo "Error: at least one input_dir is required." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PREV_WAVE_IDS=()

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    mapfile -t MANIFESTS < <(find "${INPUT_DIR}" -name "*.jsonl" | sort)

    if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
        echo "Warning: no *.jsonl found under ${INPUT_DIR}, skipping." >&2
        continue
    fi

    DEPEND_FLAG=""
    if [[ ${#PREV_WAVE_IDS[@]} -gt 0 ]]; then
        DEP_LIST=$(IFS=:; echo "${PREV_WAVE_IDS[*]}")
        DEPEND_FLAG="--dependency=afterany:${DEP_LIST}"
    fi

    N_JOBS=$(( (${#MANIFESTS[@]} + MANIFESTS_PER_JOB - 1) / MANIFESTS_PER_JOB ))
    echo "Wave: ${INPUT_DIR}"
    echo "  Manifests : ${#MANIFESTS[@]}  |  per job : ${MANIFESTS_PER_JOB}  |  jobs : ${N_JOBS}"
    [[ -n "${DEPEND_FLAG}" ]] && echo "  Depends on: ${PREV_WAVE_IDS[*]}"

    CURRENT_WAVE_IDS=()
    chunk=()

    for MANIFEST in "${MANIFESTS[@]}"; do
        chunk+=("${MANIFEST}")

        if [[ ${#chunk[@]} -eq ${MANIFESTS_PER_JOB} ]]; then
            JOB_ID=$(sbatch \
                ${DEPEND_FLAG} \
                --parsable \
                "${RUN_SCRIPT}" "${INPUT_DIR}" "${OUTPUT_DIR}" --manifests "${chunk[@]}")
            CURRENT_WAVE_IDS+=("${JOB_ID}")
            echo "  ${JOB_ID}  ←  ${#chunk[@]} manifests"
            chunk=()
        fi
    done

    # Submit any remaining manifests
    if [[ ${#chunk[@]} -gt 0 ]]; then
        JOB_ID=$(sbatch \
            ${DEPEND_FLAG} \
            --parsable \
            "${RUN_SCRIPT}" "${INPUT_DIR}" "${OUTPUT_DIR}" --manifests "${chunk[@]}")
        CURRENT_WAVE_IDS+=("${JOB_ID}")
        echo "  ${JOB_ID}  ←  ${#chunk[@]} manifests (last chunk)"
    fi

    PREV_WAVE_IDS=("${CURRENT_WAVE_IDS[@]}")
    echo ""
done

echo "All waves submitted."
echo "Monitor : squeue -u ${USER}"
echo "Job IDs : ${PREV_WAVE_IDS[*]}"
