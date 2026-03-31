#!/bin/bash -l

#PBS -N coral_predict_multi_model
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB:gpu_id=A100
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"
pwd

REPO_ROOT="${REPO_ROOT:-Corals/cslic/coral_spawn_counter}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/data_yaml_files/prediction/spawn_predictor_20251227_aken_LAR01.json}"
MODEL_WEIGHTS_CSV="${MODEL_WEIGHTS_CSV:-${1:-}}"
RUN_CONFIG_DIR="${RUN_CONFIG_DIR:-${REPO_ROOT}/data_yaml_files/prediction/generated_multi_model}"
APPEND_MODEL_TO_SAVE_DIR="${APPEND_MODEL_TO_SAVE_DIR:-0}"

# Optional per-job config overrides (avoid editing JSON for each dataset)
IMG_DIR_OVERRIDE="${IMG_DIR_OVERRIDE:-}"
SAVE_DIR_OVERRIDE="${SAVE_DIR_OVERRIDE:-}"
CSLICS_UUID_OVERRIDE="${CSLICS_UUID_OVERRIDE:-}"
SUBMERSION_TIME_OVERRIDE="${SUBMERSION_TIME_OVERRIDE:-}"

PLOT_PATTERN="${PLOT_PATTERN:-*_det.json}"
PLOT_ROLLING_WINDOW="${PLOT_ROLLING_WINDOW:-50}"
PLOT_CONFIDENCE_THRESHOLD="${PLOT_CONFIDENCE_THRESHOLD:-0.0}"
PLOT_TIMESTAMP_SOURCE="${PLOT_TIMESTAMP_SOURCE:-filename}"
PLOT_SCALE_FACTOR="${PLOT_SCALE_FACTOR:-1.0}"
PLOT_TITLE_PREFIX="${PLOT_TITLE_PREFIX:-Temporal detections}"
PLOT_DIVIDER_HOURS="${PLOT_DIVIDER_HOURS:-}"
PLOT_DIVIDER_TIME="${PLOT_DIVIDER_TIME:-}"

if [ -f /home/${USER}/miniforge3/bin/activate ]; then
	source /home/${USER}/miniforge3/bin/activate cgras
elif [ -f /home/${USER}/mambaforge/bin/activate ]; then
	source /home/${USER}/mambaforge/bin/activate cgras
else
	echo "Could not find conda activate script for user ${USER}"
	exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
	echo "Missing base config: $BASE_CONFIG"
	exit 1
fi

if [ -z "$MODEL_WEIGHTS_CSV" ]; then
	echo "MODEL_WEIGHTS_CSV is empty. Provide a comma-separated list of model files."
	echo "PBS note: inline shell env vars are not always exported to jobs."
	echo "Use one of these:"
	echo "  qsub -v MODEL_WEIGHTS_CSV='/path/model_a.pt,/path/model_b.pt' ${REPO_ROOT}/scripts/predict_multi_model_hpc.bash"
	echo "  qsub ${REPO_ROOT}/scripts/predict_multi_model_hpc.bash -- '/path/model_a.pt,/path/model_b.pt'"
	exit 1
fi

IFS=',' read -r -a MODEL_WEIGHTS <<< "$MODEL_WEIGHTS_CSV"
if [ "${#MODEL_WEIGHTS[@]}" -eq 0 ]; then
	echo "No model paths parsed from MODEL_WEIGHTS_CSV"
	exit 1
fi

for i in "${!MODEL_WEIGHTS[@]}"; do
	w="${MODEL_WEIGHTS[$i]}"
	w="${w#${w%%[![:space:]]*}}"
	w="${w%${w##*[![:space:]]}}"
	MODEL_WEIGHTS[$i]="$w"

	if [ -z "$w" ]; then
		echo "Empty model path detected in MODEL_WEIGHTS_CSV"
		exit 1
	fi
	if [ ! -f "$w" ]; then
		echo "Missing model weights: $w"
		exit 1
	fi
done

mkdir -p "$RUN_CONFIG_DIR"

which python
nvidia-smi || true

echo "[Stage 1/2] Running predictions for ${#MODEL_WEIGHTS[@]} model(s)..."
for WEIGHTS_PATH in "${MODEL_WEIGHTS[@]}"; do
	WEIGHTS_STEM="$(basename "${WEIGHTS_PATH%.*}")"
	GENERATED_CONFIG="${RUN_CONFIG_DIR}/$(basename "${BASE_CONFIG%.json}")__${WEIGHTS_STEM}.json"

	python - "$BASE_CONFIG" "$GENERATED_CONFIG" "$WEIGHTS_PATH" "$APPEND_MODEL_TO_SAVE_DIR" "$IMG_DIR_OVERRIDE" "$SAVE_DIR_OVERRIDE" "$CSLICS_UUID_OVERRIDE" "$SUBMERSION_TIME_OVERRIDE" <<'PY'
import json
import sys
from pathlib import Path

base_config_path = Path(sys.argv[1])
generated_config_path = Path(sys.argv[2])
weights_path = sys.argv[3]
append_model_to_save_dir = sys.argv[4] == "1"
img_dir_override = sys.argv[5]
save_dir_override = sys.argv[6]
cslics_uuid_override = sys.argv[7]
submersion_time_override = sys.argv[8]

with base_config_path.open("r") as f:
		cfg = json.load(f)

model_id = Path(weights_path).stem
cfg["surface_weights_path"] = weights_path
cfg["subsurface_weights_path"] = weights_path

if img_dir_override:
	cfg["img_dir"] = img_dir_override
if cslics_uuid_override:
	cfg["cslics_uuid"] = cslics_uuid_override
if submersion_time_override:
	cfg["submersion_time"] = submersion_time_override

if save_dir_override:
	cfg["save_dir"] = save_dir_override
else:
	img_dir_path = Path(cfg["img_dir"])
	cslics_uuid = str(cfg["cslics_uuid"])
	try:
		idx = img_dir_path.parts.index(cslics_uuid)
		cfg["save_dir"] = str(Path(*img_dir_path.parts[:idx]))
	except ValueError:
		# Fallback: keep existing save_dir when cslics_uuid is not in img_dir path
		pass

if append_model_to_save_dir:
		cfg["save_dir"] = str(Path(cfg["save_dir"]) / model_id)

generated_config_path.parent.mkdir(parents=True, exist_ok=True)
with generated_config_path.open("w") as f:
		json.dump(cfg, f, indent=2)

print(f"generated_config={generated_config_path}")
print(f"save_dir={cfg['save_dir']}")
print(f"cslics_uuid={cfg['cslics_uuid']}")
print(f"model_id={model_id}")
PY

	echo "Running predictor for model: ${WEIGHTS_STEM}"
	python -u ${REPO_ROOT}/coral_spawn_counter/coral_spawn_predictor.py --config "$GENERATED_CONFIG"
done

echo "[Stage 2/2] Plotting temporal counts for each model output..."
for WEIGHTS_PATH in "${MODEL_WEIGHTS[@]}"; do
	WEIGHTS_STEM="$(basename "${WEIGHTS_PATH%.*}")"
	GENERATED_CONFIG="${RUN_CONFIG_DIR}/$(basename "${BASE_CONFIG%.json}")__${WEIGHTS_STEM}.json"

	mapfile -t PLOT_META < <(python - "$GENERATED_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
with config_path.open("r") as f:
		cfg = json.load(f)

save_dir = Path(cfg["save_dir"])
cslics_uuid = cfg["cslics_uuid"]
surface_model_id = Path(cfg["surface_weights_path"]).stem
subsurface_model_id = Path(cfg["subsurface_weights_path"]).stem
mode = cfg.get("processing_mode", "both")

base = save_dir / cslics_uuid
labels_dirs = []
if mode in ("surface", "both"):
		labels_dirs.append(base / surface_model_id / "detections_text")
if mode in ("subsurface", "both"):
		labels_dirs.append(base / subsurface_model_id / "detections_text")

seen = set()
for d in labels_dirs:
		ds = str(d)
		if ds in seen:
				continue
		seen.add(ds)
		print(ds)
PY
	)

	if [ "${#PLOT_META[@]}" -eq 0 ]; then
		echo "No labels directories discovered for ${WEIGHTS_STEM}; skipping plot"
		continue
	fi

	for LABELS_DIR in "${PLOT_META[@]}"; do
		if [ ! -d "$LABELS_DIR" ]; then
			echo "Labels directory missing; skipping: $LABELS_DIR"
			continue
		fi

		OUT_PLOT="${LABELS_DIR}/temporal_label_counts_${WEIGHTS_STEM}.png"
		OUT_CSV="${LABELS_DIR}/temporal_label_counts_${WEIGHTS_STEM}.csv"

		PLOT_CMD=(
			python -u ${REPO_ROOT}/coral_spawn_counter/plot_label_counts_temporal.py
			--labels-dir "$LABELS_DIR"
			--pattern "$PLOT_PATTERN"
			--rolling-window "$PLOT_ROLLING_WINDOW"
			--confidence-threshold "$PLOT_CONFIDENCE_THRESHOLD"
			--timestamp-source "$PLOT_TIMESTAMP_SOURCE"
			--scale-factor "$PLOT_SCALE_FACTOR"
			--output-plot "$OUT_PLOT"
			--output-csv "$OUT_CSV"
			--title "${PLOT_TITLE_PREFIX}: ${WEIGHTS_STEM}"
		)

		if [ -n "$PLOT_DIVIDER_HOURS" ]; then
			PLOT_CMD+=(--divider-hours "$PLOT_DIVIDER_HOURS")
		fi
		if [ -n "$PLOT_DIVIDER_TIME" ]; then
			PLOT_CMD+=(--divider-time "$PLOT_DIVIDER_TIME")
		fi

		echo "Plotting labels in: $LABELS_DIR"
		"${PLOT_CMD[@]}"
	done
done

conda deactivate
echo "multi-model prediction + plotting job done"
