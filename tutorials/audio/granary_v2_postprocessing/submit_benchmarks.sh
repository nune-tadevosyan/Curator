#!/bin/bash
# Submit postprocessing jobs per benchmark with configurable per-benchmark chunk sizes.
#
# Scans top-level subdirectories of <input_dir> and submits each benchmark
# independently via submit.sh. Per-benchmark MANIFESTS_PER_JOB overrides are
# defined in BENCHMARK_CHUNKS below — all others get the DEFAULT (128).
#
# Usage:
#   bash submit_benchmarks.sh <output_dir> <input_dir>
#
# Override defaults via env:
#   DEFAULT_MANIFESTS_PER_JOB=64 CPUS_PER_JOB=32 bash submit_benchmarks.sh <output_dir> <input_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit.sh"

OUTPUT_DIR="${1:?Usage: bash submit_benchmarks.sh <output_dir> <input_dir>}"
INPUT_DIR="${2:?}"

# --------------------------------------------------------------------------
# Per-benchmark chunk size overrides.
# Add/edit entries here: ["benchmark_name"]=N
# --------------------------------------------------------------------------
declare -A BENCHMARK_CHUNKS=(
    ["ytc"]=8
)

DEFAULT_MANIFESTS_PER_JOB="${DEFAULT_MANIFESTS_PER_JOB:-128}"
export CPUS_PER_JOB="${CPUS_PER_JOB:-32}"

# --------------------------------------------------------------------------

mapfile -t BENCH_DIRS < <(find "${INPUT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#BENCH_DIRS[@]} -eq 0 ]]; then
    echo "Error: no subdirectories found under ${INPUT_DIR}" >&2
    exit 1
fi

echo "Output dir : ${OUTPUT_DIR}"
echo "Input dir  : ${INPUT_DIR}"
echo "Benchmarks : ${#BENCH_DIRS[@]}"
echo "CPUs/job   : ${CPUS_PER_JOB}"
echo ""

for BENCH_DIR in "${BENCH_DIRS[@]}"; do
    BENCH_NAME=$(basename "${BENCH_DIR}")

    if [[ -v BENCHMARK_CHUNKS["${BENCH_NAME}"] ]]; then
        CHUNKS="${BENCHMARK_CHUNKS[${BENCH_NAME}]}"
    else
        CHUNKS="${DEFAULT_MANIFESTS_PER_JOB}"
    fi

    echo ">>> ${BENCH_NAME}  (MANIFESTS_PER_JOB=${CHUNKS})"
    # INPUT_ROOT tells submit.sh (and the Slurm job) to use the original root
    # dir as the path anchor, so output mirrors full hierarchy: ytc/en9/manifest.jsonl
    MANIFESTS_PER_JOB="${CHUNKS}" INPUT_ROOT="${INPUT_DIR}" bash "${SUBMIT_SCRIPT}" "${OUTPUT_DIR}" "${BENCH_DIR}"
done
