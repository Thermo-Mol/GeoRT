#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIF_DIR="${1:-}"
CHECKPOINT="${2:-${SCRIPT_DIR}/checkpoint_best.pt}"
OUTPUT_DIR="${3:-${SCRIPT_DIR}/prediction_outputs}"

if [[ -z "${CIF_DIR}" ]]; then
  echo "Usage: bash run_export.sh /path/to/cif_directory [checkpoint] [output_dir]"
  echo "Example: bash run_export.sh demo_cifs"
  exit 1
fi

python "${SCRIPT_DIR}/export_predictions.py" \
  --checkpoint "${CHECKPOINT}" \
  --source-xlsx "${SCRIPT_DIR}/data/demo_prediction_input.xlsx" \
  --data-dir "${SCRIPT_DIR}/data" \
  --descriptors-xlsx descriptors_example.xlsx \
  --mof-chem-csv mof_local_chemical_descriptors_example.csv \
  --gas-xlsx gas.xlsx \
  --cif-dir "${CIF_DIR}" \
  --split-mode per_gas_mof_random \
  --split-ratios 0.7 0.2 0.1 \
  --target-component both \
  --target-mode direct \
  --target-transform log1p \
  --target-scale-mode global \
  --output-dir "${OUTPUT_DIR}" \
  --model-type transformer \
  --hidden-dim 192 \
  --dropout 0.15 \
  --transformer-layers 4 \
  --attention-heads 8 \
  --ffn-dim 768 \
  --batch-size 512 \
  --random-state 42
