#!/bin/bash
# Submit one independent Slurm job per benchmark subdirectory.
# Multiple input dirs are submitted as sequential waves:
# wave N+1 starts only after every job in wave N finishes (afterany).
#
# Usage:
#   bash submit.sh <output_dir> <input_dir_1> [input_dir_2 ...]
#
# Examples:
#   # Single wave — one job per subdir of results_large_scale_6
#   bash submit.sh /path/to/output /path/to/results_large_scale_6
#
#   # Two sequential waves
#   bash submit.sh /path/to/output /path/to/results_batch_1 /path/to/results_batch_2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"

OUTPUT_DIR="${1:?Usage: bash submit.sh <output_dir> <input_dir_1> [input_dir_2 ...]}"
shift
INPUT_DIRS=("$@")

if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
    echo "Error: at least one input_dir is required." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PREV_WAVE_IDS=()   # job IDs from the previous wave, used to build the dependency

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    # Find immediate subdirectories (one per benchmark).
    # Falls back to the directory itself if there are no subdirs.
    mapfile -t SUBDIRS < <(find "${INPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
    if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
        SUBDIRS=("${INPUT_DIR}")
    fi

    # Build --dependency flag from all job IDs in the previous wave.
    DEPEND_FLAG=""
    if [[ ${#PREV_WAVE_IDS[@]} -gt 0 ]]; then
        DEP_LIST=$(IFS=:; echo "${PREV_WAVE_IDS[*]}")
        DEPEND_FLAG="--dependency=afterany:${DEP_LIST}"
    fi

    echo "Wave: ${INPUT_DIR}  (${#SUBDIRS[@]} jobs)"
    [[ -n "${DEPEND_FLAG}" ]] && echo "  Depends on jobs: ${PREV_WAVE_IDS[*]}"

    CURRENT_WAVE_IDS=()
    for SUBDIR in "${SUBDIRS[@]}"; do
        JOB_ID=$(sbatch \
            ${DEPEND_FLAG} \
            --parsable \
            "${RUN_SCRIPT}" "${SUBDIR}" "${OUTPUT_DIR}")
        CURRENT_WAVE_IDS+=("${JOB_ID}")
        echo "  ${JOB_ID}  ←  $(basename "${SUBDIR}")"
    done

    PREV_WAVE_IDS=("${CURRENT_WAVE_IDS[@]}")
    echo ""
done

echo "All waves submitted."
echo "Monitor: squeue -u ${USER}"
echo "Job IDs: ${PREV_WAVE_IDS[*]}"
