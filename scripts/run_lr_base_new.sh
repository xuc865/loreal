#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/runs}"
DATASET="${DATASET:-oxford_flowers}"
RES="${RES:-96}"
SEED="${SEED:-1}"
SHOTS="${SHOTS:-16}"
CONFIG="${CONFIG:-vit_b16_ep50.yaml}"
DATASET_CONFIG_FILE="${DATASET_CONFIG_FILE:-configs/datasets/${DATASET}.yaml}"
BASE_CONFIG_FILE="${BASE_CONFIG_FILE:-configs/trainers/CoOp/${CONFIG}}"
LOREAL_CONFIG_FILE="${LOREAL_CONFIG_FILE:-configs/trainers/CoOp_LOREAL/${CONFIG}}"
LOAD_EPOCH="${LOAD_EPOCH:-50}"
FORCE="${FORCE:-False}"
RESET_LOREAL="${RESET_LOREAL:-0}"
LOREAL_DIM="${LOREAL_DIM:-32}"
LOREAL_TCE="${LOREAL_TCE:-0.0}"
LOREAL_BPD="${LOREAL_BPD:-0.0}"
LOREAL_COEF1="${LOREAL_COEF1:-1.0}"
LOREAL_COEF2="${LOREAL_COEF2:-2.0}"
LOREAL_GATE_INIT="${LOREAL_GATE_INIT:-0.0}"
LOREAL_LOGIT_BLEND="${LOREAL_LOGIT_BLEND:-1.0}"
LOREAL_BLEND_TRAIN="${LOREAL_BLEND_TRAIN:-False}"
LOREAL_ADAPTIVE_BLEND="${LOREAL_ADAPTIVE_BLEND:-False}"
LOREAL_ADAPTIVE_MAX_BASE="${LOREAL_ADAPTIVE_MAX_BASE:-0.5}"
LOREAL_ADAPTIVE_POWER="${LOREAL_ADAPTIVE_POWER:-2.0}"
LOREAL_PROMPT_ORDER="${LOREAL_PROMPT_ORDER:-attr_ctx_cls}"
LOREAL_N_ATT="${LOREAL_N_ATT:-2}"
LOREAL_ATT1_TEXT="${LOREAL_ATT1_TEXT:-color}"
LOREAL_ATT2_TEXT="${LOREAL_ATT2_TEXT:-shape}"
LOREAL_ATT3_TEXT="${LOREAL_ATT3_TEXT:-size}"
LOREAL_ATT4_TEXT="${LOREAL_ATT4_TEXT:-structure}"
LOREAL_ATT5_TEXT="${LOREAL_ATT5_TEXT:-outline}"
PYTHON="${PYTHON:-/mnt/workspace/wxc/miniconda3/envs/lamp/bin/python}"

# Keep broken or incompatible packages from ~/.local out of the conda env.
export PYTHONNOUSERSITE=1

BASE_TRAINER="CoOp"
LOREAL_TRAINER="CoOp_LOREAL"
SEVERITY="0"

for required_file in "${DATASET_CONFIG_FILE}" "${BASE_CONFIG_FILE}" "${LOREAL_CONFIG_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required config file: ${required_file}" >&2
    echo "Set DATASET, DATASET_CONFIG_FILE, CONFIG, BASE_CONFIG_FILE, or LOREAL_CONFIG_FILE as needed." >&2
    exit 2
  fi
done

RUN_ROOT="${OUTPUT_ROOT}/output/${LOREAL_TRAINER}/base2new/train_base/${DATASET}"
STAGE1_DIR="${RUN_ROOT}/${LOREAL_TRAINER}_stage1_students_pretraining_first/${CONFIG}/seed${SEED}"
STAGE1_EVAL_DIR="${RUN_ROOT}/${LOREAL_TRAINER}_baseline_hr_train_lr_new_test/${RES}/${CONFIG}/seed${SEED}"
STAGE2_DIR="${RUN_ROOT}/${LOREAL_TRAINER}_stage2_students_pretraining_second/${RES}/${CONFIG}/seed${SEED}"
STAGE2_EVAL_DIR="${RUN_ROOT}/${LOREAL_TRAINER}_baseline_lr_new_test/${RES}/${CONFIG}/seed${SEED}"
STAGE3_DIR="${RUN_ROOT}/${LOREAL_TRAINER}_stage3_students_sd/${RES}/${CONFIG}/seed${SEED}"
STAGE4_DIR="${RUN_ROOT}/${LOREAL_TRAINER}_stage4_students_new_test/${RES}/${CONFIG}/seed${SEED}"

LOG_ROOT="${OUTPUT_ROOT}/logs/${LOREAL_TRAINER}/${DATASET}/res${RES}/seed${SEED}"
SUMMARY="${LOG_ROOT}/lr_base_new_summary.txt"
mkdir -p "${LOG_ROOT}"

if [[ "${RESET_LOREAL}" == "1" ]]; then
  rm -rf "${STAGE3_DIR}" "${STAGE4_DIR}"
  rm -f "${LOG_ROOT}/loreal_lr_base_stage3.log" "${LOG_ROOT}/loreal_lr_new_stage4.log"
fi

BASE_OPTS=(
  DATASET.NUM_SHOTS "${SHOTS}"
  TRAINER.MODAL base2novel
  TEST.SPLIT val
  LOREAL.SEVERITY "${SEVERITY}"
)

LOREAL_OPTS=(
  TRAINER.ATPROMPT.USE_ATPROMPT True
  TRAINER.ATPROMPT.ATT_NUM 5
  TRAINER.ATPROMPT.N_ATT1 "${LOREAL_N_ATT}"
  TRAINER.ATPROMPT.N_ATT2 "${LOREAL_N_ATT}"
  TRAINER.ATPROMPT.N_ATT3 "${LOREAL_N_ATT}"
  TRAINER.ATPROMPT.N_ATT4 "${LOREAL_N_ATT}"
  TRAINER.ATPROMPT.N_ATT5 "${LOREAL_N_ATT}"
  TRAINER.ATPROMPT.ATT1_TEXT "${LOREAL_ATT1_TEXT}"
  TRAINER.ATPROMPT.ATT2_TEXT "${LOREAL_ATT2_TEXT}"
  TRAINER.ATPROMPT.ATT3_TEXT "${LOREAL_ATT3_TEXT}"
  TRAINER.ATPROMPT.ATT4_TEXT "${LOREAL_ATT4_TEXT}"
  TRAINER.ATPROMPT.ATT5_TEXT "${LOREAL_ATT5_TEXT}"
  TRAINER.PROMPTKD.KD_WEIGHT 1.0
  LOREAL.DIM "${LOREAL_DIM}"
  LOREAL.TEMP 1.0
  LOREAL.COEF1 "${LOREAL_COEF1}"
  LOREAL.COEF2 "${LOREAL_COEF2}"
  LOREAL.COEF_TCE "${LOREAL_TCE}"
  LOREAL.COEF_BPD "${LOREAL_BPD}"
  LOREAL.GATE_INIT "${LOREAL_GATE_INIT}"
  LOREAL.LOGIT_BLEND "${LOREAL_LOGIT_BLEND}"
  LOREAL.BLEND_TRAIN "${LOREAL_BLEND_TRAIN}"
  LOREAL.ADAPTIVE_BLEND "${LOREAL_ADAPTIVE_BLEND}"
  LOREAL.ADAPTIVE_MAX_BASE "${LOREAL_ADAPTIVE_MAX_BASE}"
  LOREAL.ADAPTIVE_POWER "${LOREAL_ADAPTIVE_POWER}"
  LOREAL.PROMPT_ORDER "${LOREAL_PROMPT_ORDER}"
  LOREAL.STAGE1_DIR "${STAGE1_DIR}"
  LOREAL.STAGE2_DIR "${STAGE2_DIR}"
)

run_and_log() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  echo
  echo "========== ${name} =========="
  printf '+'
  printf ' %q' "$@"
  echo
  "$@" 2>&1 | tee "${log_file}"
}

extract_acc() {
  local label="$1"
  local log_file="$2"
  local line
  line="$(grep -E "accuracy: [0-9.]+%" "${log_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    printf "%-28s %s\n" "${label}" "MISSING"
    return 0
  fi
  local acc total correct f1
  acc="$(sed -n 's/.*accuracy: \([0-9.]*%\).*/\1/p' <<< "${line}")"
  total="$(sed -n 's/.*total: \([0-9,]*\).*/\1/p' <<< "${line}")"
  correct="$(sed -n 's/.*correct: \([0-9,]*\).*/\1/p' <<< "${line}")"
  f1="$(sed -n 's/.*macro_f1: \([0-9.]*%\).*/\1/p' <<< "${line}")"
  printf "%-28s accuracy=%-8s macro_f1=%-8s correct=%s/%s\n" "${label}" "${acc}" "${f1}" "${correct}" "${total}"
}

{
  echo "LOREAL low-resolution base/new experiment"
  echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "dataset=${DATASET}"
  echo "dataset_config_file=${DATASET_CONFIG_FILE}"
  echo "base_config_file=${BASE_CONFIG_FILE}"
  echo "loreal_config_file=${LOREAL_CONFIG_FILE}"
  echo "resolution=${RES}"
  echo "seed=${SEED}"
  echo "shots=${SHOTS}"
  echo "loreal_dim=${LOREAL_DIM}"
  echo "loreal_tce=${LOREAL_TCE}"
  echo "loreal_bpd=${LOREAL_BPD}"
  echo "loreal_coef1=${LOREAL_COEF1}"
  echo "loreal_coef2=${LOREAL_COEF2}"
  echo "loreal_gate_init=${LOREAL_GATE_INIT}"
  echo "loreal_logit_blend=${LOREAL_LOGIT_BLEND}"
  echo "loreal_blend_train=${LOREAL_BLEND_TRAIN}"
  echo "loreal_adaptive_blend=${LOREAL_ADAPTIVE_BLEND}"
  echo "loreal_adaptive_max_base=${LOREAL_ADAPTIVE_MAX_BASE}"
  echo "loreal_adaptive_power=${LOREAL_ADAPTIVE_POWER}"
  echo "loreal_prompt_order=${LOREAL_PROMPT_ORDER}"
  echo "loreal_n_att=${LOREAL_N_ATT}"
  echo "loreal_attr_texts=${LOREAL_ATT1_TEXT},${LOREAL_ATT2_TEXT},${LOREAL_ATT3_TEXT},${LOREAL_ATT4_TEXT},${LOREAL_ATT5_TEXT}"
  echo "baseline_lr_train=CoOp stage2 uses batch['niimg']; LOREAL.OSIZE=${RES}"
  echo "baseline_lr_base_test=stage2 after_train uses batch['niimg']; LOREAL.TOSIZE=${RES}"
  echo "baseline_lr_new_test=eval-only stage2 checkpoint uses batch['niimg']; LOREAL.TOSIZE=${RES}"
  echo "baseline_hr_train_lr_base_test=stage1 trains with batch['niimg']; LOREAL.OSIZE=224; after_train tests with LOREAL.TOSIZE=${RES}"
  echo "baseline_hr_train_lr_new_test=eval-only stage1 checkpoint uses batch['niimg']; LOREAL.TOSIZE=${RES}"
  echo "loreal_lr_train=stage3 student uses batch['niimg']; LOREAL.OSIZE=${RES}; teacher uses batch['img'] at 224"
  echo "loreal_lr_base_test=stage3 after_train uses batch['niimg']; LOREAL.TOSIZE=${RES}"
  echo "loreal_lr_new_test=stage4 eval-only uses batch['niimg']; LOREAL.TOSIZE=${RES}"
  echo
} > "${SUMMARY}"

# Stage 1: standard-resolution CoOp teacher. Required by LOREAL stage 3.
# Its final test is also the 224-trained baseline tested on LR base classes.
run_and_log stage1_standard_teacher \
  "${PYTHON}" train.py --root "${DATA_ROOT}" --seed "${SEED}" --trainer "${BASE_TRAINER}" \
  --dataset-config-file "${DATASET_CONFIG_FILE}" \
  --config-file "${BASE_CONFIG_FILE}" \
  --output-dir "${STAGE1_DIR}" \
  "${BASE_OPTS[@]}" DATASET.SUBSAMPLE_CLASSES base TRAINER.LEVEL 0 \
  LOREAL.OSIZE 224 LOREAL.TOSIZE "${RES}" LOREAL.FORCE "${FORCE}" LOREAL.STAGE 1

# Stage 1 eval: evaluate the 224-trained baseline on LR new classes.
run_and_log baseline_hr_train_lr_new_eval \
  "${PYTHON}" train.py --root "${DATA_ROOT}" --seed "${SEED}" --trainer "${BASE_TRAINER}" \
  --dataset-config-file "${DATASET_CONFIG_FILE}" \
  --config-file "${BASE_CONFIG_FILE}" \
  --output-dir "${STAGE1_EVAL_DIR}" --model-dir "${STAGE1_DIR}" --load-epoch "${LOAD_EPOCH}" --eval-only \
  "${BASE_OPTS[@]}" DATASET.SUBSAMPLE_CLASSES new TRAINER.LEVEL 1 INPUT.SIZE "${RES}" \
  LOREAL.OSIZE 224 LOREAL.TOSIZE "${RES}" LOREAL.FORCE True LOREAL.STAGE 4

# Stage 2: low-resolution CoOp baseline. Its final test is the baseline LR base result.
run_and_log baseline_lr_base_stage2 \
  "${PYTHON}" train.py --root "${DATA_ROOT}" --seed "${SEED}" --trainer "${BASE_TRAINER}" \
  --dataset-config-file "${DATASET_CONFIG_FILE}" \
  --config-file "${BASE_CONFIG_FILE}" \
  --output-dir "${STAGE2_DIR}" \
  "${BASE_OPTS[@]}" DATASET.SUBSAMPLE_CLASSES base TRAINER.LEVEL 0 \
  INPUT.SIZE "${RES}" \
  LOREAL.OSIZE "${RES}" LOREAL.TOSIZE "${RES}" LOREAL.FORCE "${FORCE}" LOREAL.STAGE 2

# Stage 2 eval: evaluate the same low-resolution CoOp baseline on new classes.
run_and_log baseline_lr_new_eval \
  "${PYTHON}" train.py --root "${DATA_ROOT}" --seed "${SEED}" --trainer "${BASE_TRAINER}" \
  --dataset-config-file "${DATASET_CONFIG_FILE}" \
  --config-file "${BASE_CONFIG_FILE}" \
  --output-dir "${STAGE2_EVAL_DIR}" --model-dir "${STAGE2_DIR}" --load-epoch "${LOAD_EPOCH}" --eval-only \
  "${BASE_OPTS[@]}" DATASET.SUBSAMPLE_CLASSES new TRAINER.LEVEL 1 INPUT.SIZE "${RES}" \
  LOREAL.OSIZE 224 LOREAL.TOSIZE "${RES}" LOREAL.FORCE True LOREAL.STAGE 4

# Stage 3: LOREAL self-distillation. Its final test is the LOREAL LR base result.
run_and_log loreal_lr_base_stage3 \
  "${PYTHON}" train.py --root "${DATA_ROOT}" --seed "${SEED}" --trainer "${LOREAL_TRAINER}" \
  --dataset-config-file "${DATASET_CONFIG_FILE}" \
  --config-file "${LOREAL_CONFIG_FILE}" \
  --output-dir "${STAGE3_DIR}" \
  "${BASE_OPTS[@]}" DATASET.SUBSAMPLE_CLASSES base TRAINER.LEVEL 0 \
  LOREAL.OSIZE "${RES}" LOREAL.TOSIZE "${RES}" LOREAL.FORCE "${FORCE}" LOREAL.STAGE 3 \
  "${LOREAL_OPTS[@]}"

# Stage 4: evaluate the distilled low-resolution LOREAL model on new classes.
run_and_log loreal_lr_new_stage4 \
  "${PYTHON}" train.py --root "${DATA_ROOT}" --seed "${SEED}" --trainer "${LOREAL_TRAINER}" \
  --dataset-config-file "${DATASET_CONFIG_FILE}" \
  --config-file "${LOREAL_CONFIG_FILE}" \
  --output-dir "${STAGE4_DIR}" --model-dir "${STAGE3_DIR}" --load-epoch "${LOAD_EPOCH}" --eval-only \
  "${BASE_OPTS[@]}" DATASET.SUBSAMPLE_CLASSES new TRAINER.LEVEL 1 INPUT.SIZE "${RES}" \
  LOREAL.OSIZE 224 LOREAL.TOSIZE "${RES}" LOREAL.FORCE True LOREAL.STAGE 4 \
  "${LOREAL_OPTS[@]}"

{
  echo "Key low-resolution results"
  echo "--------------------------"
  extract_acc "LOREAL LR base" "${LOG_ROOT}/loreal_lr_base_stage3.log"
  extract_acc "LOREAL LR new" "${LOG_ROOT}/loreal_lr_new_stage4.log"
  echo
  echo "Logs: ${LOG_ROOT}"
  echo "Outputs: ${RUN_ROOT}"
} | tee -a "${SUMMARY}"

echo
echo "Summary written to ${SUMMARY}"
