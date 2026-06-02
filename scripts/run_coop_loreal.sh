#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the complete CoOp + LOREAL pipeline.

Required:
  --data-root PATH          Dataset root containing folders such as imagenet/, oxford_pets/.

Common options:
  --output-root PATH        Output prefix. Default: .
  --datasets LIST           Comma-separated dataset keys. Default: oxford_pets
  --resolutions LIST        Comma-separated LR sizes. Paper uses 96,144,192. Default: 96,144,192
  --seeds LIST              Comma-separated seeds. Paper reports 3 runs. Default: 1
  --shots N                 Few-shot samples per class. Default: 16
  --stage NAME              all, loreal, stage1, stage1_eval, stage2, stage2_eval, stage3, stage4. Default: all
  --force BOOL              Force retraining if checkpoints exist. Default: False
  --dry-run                 Print commands without running them.

LOREAL attribute options:
  --attributes LIST         Five comma-separated robust attributes.
                            Default: color,shape,size,structure,outline
  --attr-tokens N           Learnable tokens per attribute, M in the paper. Default: 2
  --meta-dim N              Meta-net hidden dimension, Ds in the paper. Default: 32
  --lambda-hld X            HLD coefficient lambda1. Default: 1.0
  --lambda-lld X            LLD coefficient lambda2. Default: 2.0

Examples:
  bash scripts/run_coop_loreal.sh --data-root /data/TIP-data --datasets oxford_pets --resolutions 96 --seeds 1 --stage loreal
  bash scripts/run_coop_loreal.sh --data-root /data/TIP-data --datasets oxford_pets,stanford_cars --seeds 1,2,3
EOF
}

DATA_ROOT=""
OUTPUT_ROOT="."
DATASETS="oxford_pets"
RESOLUTIONS="96,144,192"
SEEDS="1"
SHOTS="16"
STAGE="all"
FORCE="False"
DRY_RUN="False"
ATTRIBUTES="color,shape,size,structure,outline"
ATTR_TOKENS="2"
META_DIM="32"
LAMBDA_HLD="1.0"
LAMBDA_LLD="2.0"
SEVERITY="0"
CONFIG="vit_b16_ep50.yaml"
LOAD_EPOCH="50"
BASE_TRAINER="CoOp"
LOREAL_TRAINER="CoOp_LOREAL"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --resolutions) RESOLUTIONS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="True"; shift ;;
    --attributes) ATTRIBUTES="$2"; shift 2 ;;
    --attr-tokens) ATTR_TOKENS="$2"; shift 2 ;;
    --meta-dim) META_DIM="$2"; shift 2 ;;
    --lambda-hld) LAMBDA_HLD="$2"; shift 2 ;;
    --lambda-lld) LAMBDA_LLD="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$DATA_ROOT" ]]; then
  echo "Missing --data-root" >&2
  usage
  exit 2
fi

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
IFS=',' read -r -a RES_ARR <<< "$RESOLUTIONS"
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a ATTR_ARR <<< "$ATTRIBUTES"

case "$STAGE" in
  all|loreal|stage1|stage1_eval|stage2|stage2_eval|stage3|stage4) ;;
  *)
    echo "Unknown --stage: $STAGE" >&2
    usage
    exit 2
    ;;
esac

if [[ "${#ATTR_ARR[@]}" -ne 5 ]]; then
  echo "--attributes must contain exactly five attributes." >&2
  echo "The current CoOp_LOREAL config follows the paper and defines ATT1..ATT5 only." >&2
  exit 2
fi

should_run() {
  local candidate="$1"
  if [[ "$STAGE" == "all" || "$STAGE" == "$candidate" ]]; then
    return 0
  fi
  if [[ "$STAGE" == "loreal" ]]; then
    [[ "$candidate" == "stage1" || "$candidate" == "stage2" || "$candidate" == "stage3" || "$candidate" == "stage4" ]]
    return
  fi
  return 1
}

run_cmd() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  if [[ "$DRY_RUN" != "True" ]]; then
    "$@"
  fi
}

stage_dir() {
  local dataset="$1"
  local stage="$2"
  local res="$3"
  local seed="$4"

  case "$stage" in
    stage1)
      echo "${OUTPUT_ROOT}/output/${LOREAL_TRAINER}/base2new/train_base/${dataset}/${LOREAL_TRAINER}_stage1_students_pretraining_first/${CONFIG}/seed${seed}"
      ;;
    stage1_eval)
      echo "${OUTPUT_ROOT}/output/${LOREAL_TRAINER}/base2new/train_base/${dataset}/${LOREAL_TRAINER}_stage1.5_students_pretraining_feval/${CONFIG}/seed${seed}"
      ;;
    stage2)
      echo "${OUTPUT_ROOT}/output/${LOREAL_TRAINER}/base2new/train_base/${dataset}/${LOREAL_TRAINER}_stage2_students_pretraining_second/${res}/${CONFIG}/seed${seed}"
      ;;
    stage2_eval)
      echo "${OUTPUT_ROOT}/output/${LOREAL_TRAINER}/base2new/train_base/${dataset}/${LOREAL_TRAINER}_stage2.5_students_pretraining_seval/${res}/${CONFIG}/seed${seed}"
      ;;
    stage3)
      echo "${OUTPUT_ROOT}/output/${LOREAL_TRAINER}/base2new/train_base/${dataset}/${LOREAL_TRAINER}_stage3_students_sd/${res}/${CONFIG}/seed${seed}"
      ;;
    stage4)
      echo "${OUTPUT_ROOT}/output/${LOREAL_TRAINER}/base2new/train_base/${dataset}/${LOREAL_TRAINER}_stage4_students_new_test/${res}/${CONFIG}/seed${seed}"
      ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
  esac
}

LOREAL_OPTS=()
build_loreal_opts() {
  LOREAL_OPTS=()

  LOREAL_OPTS+=(TRAINER.ATPROMPT.USE_ATPROMPT True)
  LOREAL_OPTS+=(TRAINER.ATPROMPT.ATT_NUM "${#ATTR_ARR[@]}")
  for i in "${!ATTR_ARR[@]}"; do
    local idx=$((i + 1))
    LOREAL_OPTS+=("TRAINER.ATPROMPT.N_ATT${idx}" "$ATTR_TOKENS")
    LOREAL_OPTS+=("TRAINER.ATPROMPT.ATT${idx}_TEXT" "${ATTR_ARR[$i]}")
  done
  LOREAL_OPTS+=(LOREAL.DIM "$META_DIM")
  LOREAL_OPTS+=(LOREAL.COEF1 "$LAMBDA_HLD")
  LOREAL_OPTS+=(LOREAL.COEF2 "$LAMBDA_LLD")
  LOREAL_OPTS+=(LOREAL.TEMP 1.0)
  LOREAL_OPTS+=(TRAINER.PROMPTKD.KD_WEIGHT 1.0)
}

for seed in "${SEED_ARR[@]}"; do
  for res in "${RES_ARR[@]}"; do
    for dataset in "${DATASET_ARR[@]}"; do
      dataset_cfg="configs/datasets/${dataset}.yaml"
      if [[ ! -f "$dataset_cfg" ]]; then
        echo "Dataset config not found: $dataset_cfg" >&2
        exit 2
      fi

      stage1_dir="$(stage_dir "$dataset" stage1 "$res" "$seed")"
      stage1_eval_dir="$(stage_dir "$dataset" stage1_eval "$res" "$seed")"
      stage2_dir="$(stage_dir "$dataset" stage2 "$res" "$seed")"
      stage2_eval_dir="$(stage_dir "$dataset" stage2_eval "$res" "$seed")"
      stage3_dir="$(stage_dir "$dataset" stage3 "$res" "$seed")"
      stage4_dir="$(stage_dir "$dataset" stage4 "$res" "$seed")"

      # Stage 1 in the paper: pretrain the standard-resolution CoOp student.
      if should_run stage1; then
        run_cmd python train.py --root "$DATA_ROOT" --seed "$seed" --trainer "$BASE_TRAINER" \
          --dataset-config-file "$dataset_cfg" \
          --config-file "configs/trainers/${BASE_TRAINER}/${CONFIG}" \
          --output-dir "$stage1_dir" \
          DATASET.NUM_SHOTS "$SHOTS" TRAINER.MODAL base2novel DATASET.SUBSAMPLE_CLASSES base \
          TEST.SPLIT val TRAINER.LEVEL 0 \
          LOREAL.SEVERITY "$SEVERITY" LOREAL.OSIZE 224 LOREAL.TOSIZE "$res" LOREAL.FORCE "$FORCE" LOREAL.KAIDANN 1
      fi

      # Optional baseline evaluation of the stage-1 student on new LR classes.
      if should_run stage1_eval; then
        run_cmd python train.py --root "$DATA_ROOT" --seed "$seed" --trainer "$BASE_TRAINER" \
          --dataset-config-file "$dataset_cfg" \
          --config-file "configs/trainers/${BASE_TRAINER}/${CONFIG}" \
          --output-dir "$stage1_eval_dir" --model-dir "$stage1_dir" --load-epoch "$LOAD_EPOCH" --eval-only \
          DATASET.NUM_SHOTS "$SHOTS" TRAINER.MODAL base2novel DATASET.SUBSAMPLE_CLASSES new \
          TEST.SPLIT val TRAINER.LEVEL 1 INPUT.SIZE "$res" \
          LOREAL.SEVERITY "$SEVERITY" LOREAL.OSIZE 224 LOREAL.TOSIZE "$res" LOREAL.FORCE True LOREAL.KAIDANN 4
      fi

      # Stage 2 in the paper: pretrain the low-resolution CoOp student.
      if should_run stage2; then
        run_cmd python train.py --root "$DATA_ROOT" --seed "$seed" --trainer "$BASE_TRAINER" \
          --dataset-config-file "$dataset_cfg" \
          --config-file "configs/trainers/${BASE_TRAINER}/${CONFIG}" \
          --output-dir "$stage2_dir" \
          DATASET.NUM_SHOTS "$SHOTS" TRAINER.MODAL base2novel DATASET.SUBSAMPLE_CLASSES base \
          TEST.SPLIT val TRAINER.LEVEL 0 \
          LOREAL.SEVERITY "$SEVERITY" LOREAL.OSIZE "$res" LOREAL.TOSIZE "$res" LOREAL.FORCE "$FORCE" LOREAL.KAIDANN 2
      fi

      # Optional baseline evaluation of the stage-2 LR student on new LR classes.
      if should_run stage2_eval; then
        run_cmd python train.py --root "$DATA_ROOT" --seed "$seed" --trainer "$BASE_TRAINER" \
          --dataset-config-file "$dataset_cfg" \
          --config-file "configs/trainers/${BASE_TRAINER}/${CONFIG}" \
          --output-dir "$stage2_eval_dir" --model-dir "$stage2_dir" --load-epoch "$LOAD_EPOCH" --eval-only \
          DATASET.NUM_SHOTS "$SHOTS" TRAINER.MODAL base2novel DATASET.SUBSAMPLE_CLASSES new \
          TEST.SPLIT val TRAINER.LEVEL 1 INPUT.SIZE "$res" \
          LOREAL.SEVERITY "$SEVERITY" LOREAL.OSIZE 224 LOREAL.TOSIZE "$res" LOREAL.FORCE True LOREAL.KAIDANN 4
      fi

      # Stage 3 in the paper: LOREAL self-distillation.
      # It loads stage 1 and stage 2 checkpoints, freezes CoOp prompts/backbone,
      # and trains only the attribute meta-nets with CE + HLD + LLD.
      if should_run stage3; then
        build_loreal_opts
        run_cmd python train.py --root "$DATA_ROOT" --seed "$seed" --trainer "$LOREAL_TRAINER" \
          --dataset-config-file "$dataset_cfg" \
          --config-file "configs/trainers/${LOREAL_TRAINER}/${CONFIG}" \
          --output-dir "$stage3_dir" \
          DATASET.NUM_SHOTS "$SHOTS" TRAINER.MODAL base2novel DATASET.SUBSAMPLE_CLASSES base \
          TEST.SPLIT val TRAINER.LEVEL 0 \
          LOREAL.SEVERITY "$SEVERITY" LOREAL.OSIZE "$res" LOREAL.TOSIZE "$res" LOREAL.FORCE "$FORCE" LOREAL.KAIDANN 3 \
          LOREAL.STAGE1_DIR "$stage1_dir" LOREAL.STAGE2_DIR "$stage2_dir" \
          "${LOREAL_OPTS[@]}"
      fi

      # Stage 4 in the paper: evaluate the distilled LR student on new LR classes.
      if should_run stage4; then
        build_loreal_opts
        run_cmd python train.py --root "$DATA_ROOT" --seed "$seed" --trainer "$LOREAL_TRAINER" \
          --dataset-config-file "$dataset_cfg" \
          --config-file "configs/trainers/${LOREAL_TRAINER}/${CONFIG}" \
          --output-dir "$stage4_dir" --model-dir "$stage3_dir" --load-epoch "$LOAD_EPOCH" --eval-only \
          DATASET.NUM_SHOTS "$SHOTS" TRAINER.MODAL base2novel DATASET.SUBSAMPLE_CLASSES new \
          TEST.SPLIT val TRAINER.LEVEL 1 INPUT.SIZE "$res" \
          LOREAL.SEVERITY "$SEVERITY" LOREAL.OSIZE 224 LOREAL.TOSIZE "$res" LOREAL.FORCE True LOREAL.KAIDANN 4 \
          "${LOREAL_OPTS[@]}"
      fi
    done
  done
done
