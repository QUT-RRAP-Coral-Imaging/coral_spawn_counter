#!/bin/bash -l

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-Corals/cslic/coral_spawn_counter}"
PREDICT_JOB_SCRIPT="${PREDICT_JOB_SCRIPT:-${REPO_ROOT}/scripts/predict_multi_model_hpc.bash}"
MODEL_WEIGHTS_PATH="${MODEL_WEIGHTS_PATH:-${1:-}}"
DATASET_CONFIGS_CSV="${DATASET_CONFIGS_CSV:-${2:-}}"
DATASET_CONFIGS_FILE="${DATASET_CONFIGS_FILE:-}"
EXTRA_QSUB_VARS="${EXTRA_QSUB_VARS:-}"

usage() {
	echo "Submit one qsub job per dataset using one model weights file."
	echo
	echo "Usage:"
	echo "  $0 <model_weights_path> <dataset_config_csv>"
	echo "  MODEL_WEIGHTS_PATH=/path/best.pt DATASET_CONFIGS_CSV='/path/ds1.json,/path/ds2.json' $0"
	echo "  MODEL_WEIGHTS_PATH=/path/best.pt DATASET_CONFIGS_FILE=/path/datasets.txt $0"
	echo
	echo "Environment variables:"
	echo "  REPO_ROOT          Default: ${REPO_ROOT}"
	echo "  PREDICT_JOB_SCRIPT Default: ${PREDICT_JOB_SCRIPT}"
	echo "  DATASET_CONFIGS_CSV Comma-separated config JSON paths"
	echo "  DATASET_CONFIGS_FILE Newline-separated config JSON paths (# comments allowed)"
	echo "  EXTRA_QSUB_VARS    Extra vars appended to qsub -v, e.g. 'APPEND_MODEL_TO_SAVE_DIR=0'"
}

trim() {
	local value="$1"
	value="${value#${value%%[![:space:]]*}}"
	value="${value%${value##*[![:space:]]}}"
	printf '%s' "$value"
}

if [ -z "$MODEL_WEIGHTS_PATH" ]; then
	echo "MODEL_WEIGHTS_PATH is required."
	usage
	exit 1
fi

if [ ! -f "$MODEL_WEIGHTS_PATH" ]; then
	echo "Missing model weights: $MODEL_WEIGHTS_PATH"
	exit 1
fi

if [ ! -f "$PREDICT_JOB_SCRIPT" ]; then
	echo "Missing predict job script: $PREDICT_JOB_SCRIPT"
	exit 1
fi

if ! command -v qsub >/dev/null 2>&1; then
	echo "qsub not found in PATH"
	exit 1
fi

mapfile -t DATASET_CONFIGS < <(
	{
		if [ -n "$DATASET_CONFIGS_CSV" ]; then
			printf '%s\n' "$DATASET_CONFIGS_CSV" | tr ',' '\n'
		fi
		if [ -n "$DATASET_CONFIGS_FILE" ]; then
			grep -vE '^\s*($|#)' "$DATASET_CONFIGS_FILE" || true
		fi
	} | sed 's/^\s*//;s/\s*$//' | awk 'NF'
)

if [ "${#DATASET_CONFIGS[@]}" -eq 0 ]; then
	echo "No dataset configs provided. Set DATASET_CONFIGS_CSV or DATASET_CONFIGS_FILE."
	usage
	exit 1
fi

for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
	dataset_cfg="$(trim "$dataset_cfg")"
	if [ -z "$dataset_cfg" ]; then
		continue
	fi
	if [ ! -f "$dataset_cfg" ]; then
		echo "Missing dataset config: $dataset_cfg"
		exit 1
	fi
done

echo "Submitting ${#DATASET_CONFIGS[@]} qsub jobs"
echo "Model: $MODEL_WEIGHTS_PATH"
echo "Predict script: $PREDICT_JOB_SCRIPT"

submitted=0
for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
	dataset_cfg="$(trim "$dataset_cfg")"
	dataset_stem="$(basename "${dataset_cfg%.json}")"
	job_name="coral_${dataset_stem}"
	job_name="$(printf '%s' "$job_name" | tr -cs '[:alnum:]_' '_' | cut -c1-60)"

	qsub_vars="MODEL_WEIGHTS_CSV=${MODEL_WEIGHTS_PATH},BASE_CONFIG=${dataset_cfg},REPO_ROOT=${REPO_ROOT}"
	if [ -n "$EXTRA_QSUB_VARS" ]; then
		qsub_vars="${qsub_vars},${EXTRA_QSUB_VARS}"
	fi

	job_id="$(qsub -N "$job_name" -v "$qsub_vars" "$PREDICT_JOB_SCRIPT")"
	echo "submitted job_id=${job_id} dataset=${dataset_cfg}"
	submitted=$((submitted + 1))
done

echo "Submitted ${submitted} jobs successfully"
