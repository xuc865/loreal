DATA="PATH/TIP-data"
OUT_PREFIX="PATH"
# conda create --name PATH --clone PATH

BASE_METHOD=$1
METHOD=$2 # "" 
CONFIG="vit_b16_c2_ep20_batch32_2ctx.yaml"
SEVERITY=0
LOAD_EPOCH=0
COMMAND=$4
if [ "$METHOD" == "MaPLe_REDIS" ] || [ "$METHOD" == "MaPLe" ]; then
    CONFIG="vit_b16_c2_ep20_batch32_2ctx.yaml"
    LOAD_EPOCH=20
elif [ "$METHOD" = "MultiModalAdapter_REDIS" ] || [ "$METHOD" == "MultiModalAdapter" ]; then
    CONFIG="vit_b16_ep5.yaml"
    LOAD_EPOCH=5
elif [ "$METHOD" = "CoOp_REDIS" ] || [ "$METHOD" == "CoOp" ]; then
    CONFIG="vit_b16_ep50.yaml" 
    LOAD_EPOCH=50
elif [ "$METHOD" = "ZeroshotCLIP_REDIS" ] || [ "$METHOD" == "ZeroshotCLIP" ]; then
    CONFIG="vit_b16_ep50.yaml" 
    LOAD_EPOCH=-1
elif [ "$METHOD" = "TCP_REDIS" ] || [ "$METHOD" == "TCP" ]; then
    CONFIG="vit_b16_ep100_ctxv1.yaml" 
    LOAD_EPOCH=50
elif [ "$METHOD" = "MMRL_REDIS" ] || [ "$METHOD" == "MMRL" ]; then
    CONFIG="vit_b16.yaml" 
    LOAD_EPOCH=10
elif [ "$METHOD" = "PromptSRC_REDIS" ] || [ "$METHOD" == "PromptSRC" ]; then
    CONFIG="vit_b16_c2_ep20_batch4_4+4ctx.yaml" 
    LOAD_EPOCH=10
elif [ "$METHOD" = "PromptKD_REDIS" ] || [ "$METHOD" == "PromptKD" ]; then
    CONFIG="vit_b16_c2_ep20_batch8_4+4ctx.yaml" 
    LOAD_EPOCH=20
elif [ "$METHOD" = "REDIS" ]; then
    CONFIG="vit_b16_c2_ep20_batch8_4+4ctx.yaml" 
    LOAD_EPOCH=20
fi  

SEED=2 
DATASETS=("stanford_cars" "fgvc_aircraft")  #"dtd" "oxford_pets" "caltech101" "sun397" "eurosat" "food101" "oxford_flowers" "ucf101" 
#  

OSI=224
for TOSI in 96 128 160 192; do # 48 64 80  112 144 176  208 224

    for DATASET in "${DATASETS[@]}"; do 

        # stage 1: two students pretraining - the first one
        python train.py  --root $DATA --seed $SEED --trainer ${BASE_METHOD} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file "configs/trainers/${BASE_METHOD}/${CONFIG}" \
        --output-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage1_students_pretraining_first/${CONFIG}/seed${SEED}"  \
        DATASET.NUM_SHOTS 16 \
        TRAINER.MODAL base2novel \
        DATASET.SUBSAMPLE_CLASSES base \
        TEST.SPLIT val \
        TRAINER.LEVEL 0 \
        POW.SEVERITY $SEVERITY \
        POW.OSIZE 224 \
        POW.TOSIZE  $TOSI \
        POW.FORCE False \
        POW.STAGE 1


        # stage 1.5: need a test for baseline new-test
        python train.py  --root $DATA --seed $SEED --trainer ${BASE_METHOD} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file "configs/trainers/${BASE_METHOD}/${CONFIG}" \
        --output-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage1.5_students_pretraining_feval/${CONFIG}/seed${SEED}"  \
        --model-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage1_students_pretraining_first/${CONFIG}/seed${SEED}" \
        --load-epoch $LOAD_EPOCH \
        --eval-only \
        DATASET.NUM_SHOTS 16 \
        TRAINER.MODAL base2novel \
        DATASET.SUBSAMPLE_CLASSES new \
        TEST.SPLIT val \
        TRAINER.LEVEL 1 \
        POW.FORCE True  \
        INPUT.SIZE $TOSI \
        POW.SEVERITY $SEVERITY \
        POW.OSIZE 224 \
        POW.TOSIZE $TOSI \
        POW.STAGE 4


        # stage 2: two students pretraining - the second one
        python train.py  --root $DATA --seed $SEED --trainer ${BASE_METHOD} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file "configs/trainers/${BASE_METHOD}/${CONFIG}" \
        --output-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage2_students_pretraining_second/${TOSI}/${CONFIG}/seed${SEED}"  \
        DATASET.NUM_SHOTS 16 \
        TRAINER.MODAL base2novel \
        DATASET.SUBSAMPLE_CLASSES base \
        TEST.SPLIT val \
        TRAINER.LEVEL 0 \
        POW.SEVERITY $SEVERITY \
        POW.OSIZE $TOSI \
        POW.TOSIZE $TOSI \
        POW.FORCE False \
        POW.STAGE 2


        # stage 2.5: need a test for baseline second new-test
        python train.py  --root $DATA --seed $SEED --trainer ${BASE_METHOD} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file "configs/trainers/${BASE_METHOD}/${CONFIG}" \
        --output-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage2.5_students_pretraining_seval/${TOSI}/${CONFIG}/seed${SEED}"  \
        --model-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage2_students_pretraining_second/${TOSI}/${CONFIG}/seed${SEED}" \
        --load-epoch $LOAD_EPOCH \
        --eval-only \
        DATASET.NUM_SHOTS 16 \
        TRAINER.MODAL base2novel \
        DATASET.SUBSAMPLE_CLASSES new \
        TEST.SPLIT val \
        TRAINER.LEVEL 1 \
        POW.FORCE True  \
        INPUT.SIZE $TOSI \
        POW.SEVERITY $SEVERITY \
        POW.OSIZE 224 \
        POW.TOSIZE $TOSI \
        POW.STAGE 4

        # stage 3: two students self-distillation and base test
        python train.py  --root $DATA --seed $SEED --trainer ${METHOD} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file "configs/trainers/${METHOD}/${CONFIG}" \
        --output-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage3_students_sd/${TOSI}/${CONFIG}/seed${SEED}"  \
        DATASET.NUM_SHOTS 16 \
        TRAINER.MODAL base2novel \
        DATASET.SUBSAMPLE_CLASSES base \
        TEST.SPLIT val \
        TRAINER.LEVEL 0 \
        POW.SEVERITY $SEVERITY \
        POW.OSIZE $TOSI \
        POW.TOSIZE $TOSI \
        POW.FORCE $3 \
        POW.STAGE 3
 
        # stage 4: new test 
        python train.py  --root $DATA --seed $SEED --trainer ${METHOD} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file "configs/trainers/${METHOD}/${CONFIG}" \
        --output-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage4_students_new_test/${TOSI}/${CONFIG}/seed${SEED}"  \
        --model-dir "${OUT_PREFIX}/output/${METHOD}/base2new/train_base/${DATASET}/${METHOD}_stage3_students_sd/${TOSI}/${CONFIG}/seed${SEED}" \
        --load-epoch $LOAD_EPOCH \
        --eval-only \
        DATASET.NUM_SHOTS 16 \
        TRAINER.MODAL base2novel \
        DATASET.SUBSAMPLE_CLASSES new \
        TEST.SPLIT val \
        TRAINER.LEVEL 1 \
        POW.FORCE True  \
        INPUT.SIZE $TOSI \
        POW.SEVERITY $SEVERITY \
        POW.OSIZE 224 \
        POW.TOSIZE $TOSI \
        POW.STAGE 4
    
    done
done


RESET_LOREAL=1 LOREAL_LOGIT_BLEND=0.7 LOREAL_GATE_INIT=0.05 bash scripts/run_oxfordflowers_lr_base_new.sh



