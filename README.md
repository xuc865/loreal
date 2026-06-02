# CoOp + LOREAL

This repository currently provides example code for applying LOREAL on top of CoOp. The documentation, unified script, and configs describe that example: first train two CoOp students at different resolutions, then run `CoOp_REDIS` to add LOREAL's attribute meta-net self-distillation on top of CoOp prompts.

## CoOp Method Mapping

The CoOp baseline trainer is `trainers/coop.py`. The LOREAL-on-CoOp trainer is `trainers/coop_redis.py`. Their configs are `configs/trainers/CoOp/vit_b16_ep50.yaml` and `configs/trainers/CoOp_REDIS/vit_b16_ep50.yaml`.

| Paper concept | Code location |
| --- | --- |
| CoOp base prompt `P0` | `PromptLearner.ctx` |
| Five attribute slots `S_k [A_k]` | `TRAINER.ATPROMPT.ATT1_TEXT` to `ATT5_TEXT` |
| Attribute meta-net `S_k = M_k(f_v)` | `PromptLearner.metanets` |
| Cross-resolution bridge | `CustomCLIP.forward(..., student_visual=...)` |
| LLD, Eq. (7) | `CoOp_REDIS.low_level_distillation` |
| HLD, Eq. (8) | `CoOp_REDIS.forward_backward` |
| Final loss `CE + lambda1 * HLD + lambda2 * LLD` | `POW.COEF1` and `POW.COEF2` |

The CoOp + LOREAL pipeline has four paper-aligned stages:

1. `stage1`: pretrain the standard-resolution student with `CoOp`.
2. `stage2`: pretrain the low-resolution student with `CoOp`.
3. `stage3`: run `CoOp_REDIS`, load both CoOp students, freeze CLIP and CoOp prompts, and train only the shared LOREAL attribute meta-nets.
4. `stage4`: evaluate the distilled low-resolution CoOp student on new classes with `CoOp_REDIS`.

## Environment

Python 3.8 or 3.9 is recommended. Install PyTorch first according to your CUDA version.

```bash
cd /Users/wxc/Documents/codes/LOREAL-main

# Install torch/torchvision for your CUDA version first.
# Example for CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

If additional packages are missing at runtime, install the dependencies from `Dassl.pytorch/requirements.txt` first. The root `requirements.txt` only contains lightweight CLIP/CoOp-side requirements.

## Datasets

Put all datasets under one root, for example `/data/TIP-data`, and pass it with `--data-root /data/TIP-data`.

See `docs/DATASETS.md` for download links. The table below lists the directory names and key files that the current dataset loaders read.

| Script key | Expected layout |
| --- | --- |
| `imagenet` | `$DATA/imagenet/classnames.txt`, plus either `$DATA/imagenet/images/train,val` or `$DATA/imagenet/train,val` |
| `caltech101` | `$DATA/caltech-101/101_ObjectCategories`, `split_zhou_Caltech101.json` |
| `oxford_pets` | `$DATA/oxford_pets/images`, `annotations`, `split_zhou_OxfordPets.json` |
| `stanford_cars` | `$DATA/stanford_cars/cars_train`, `cars_test`, `devkit`, `cars_test_annos_withlabels.mat`, `split_zhou_StanfordCars.json` |
| `oxford_flowers` | `$DATA/oxford_flowers/jpg`, `imagelabels.mat`, `cat_to_name.json`, `split_zhou_OxfordFlowers.json` |
| `food101` | `$DATA/food-101/images`, `meta`, `split_zhou_Food101.json` |
| `fgvc_aircraft` | `$DATA/fgvc_aircraft/images`, `variants.txt`, `images_variant_train/val/test.txt` |
| `sun397` | `$DATA/sun397/SUN397`, `split_zhou_SUN397.json` |
| `dtd` | `$DATA/dtd/images`, `labels`, `imdb`, `split_zhou_DescribableTextures.json` |
| `eurosat` | `$DATA/eurosat/2750`, `split_zhou_EuroSAT.json` |
| `ucf101` | `$DATA/ucf101/UCF-101-midframes`, `split_zhou_UCF101.json` |
| `imagenetv2` | `$DATA/imagenetv2/imagenetv2-matched-frequency-format-val`, `classnames.txt` |
| `imagenet_sketch` | `$DATA/imagenet-sketch/images` or `$DATA/imagenet-sketch/sketch`, `classnames.txt` |
| `imagenet_a` | `$DATA/imagenet-adversarial/imagenet-a`, `classnames.txt` |
| `imagenet_r` | `$DATA/imagenet-rendition/imagenet-r`, `classnames.txt` |

ImageNet-derived datasets reuse ImageNet's `classnames.txt`. If a dataset already exists somewhere else, create a symlink under `$DATA`:

```bash
ln -s /real/path/to/imagenet /data/TIP-data/imagenet
```

## Unified CoOp Script

The unified script runs the CoOp example: `stage1/stage2` use `CoOp`, while `stage3/stage4` use `CoOp_REDIS`. Start with a dry run to inspect the commands and output directories:

```bash
bash scripts/run_coop_loreal.sh \
  --data-root /data/TIP-data \
  --datasets oxford_pets \
  --resolutions 96 \
  --seeds 1 \
  --stage loreal \
  --dry-run
```

Run a minimal CoOp + LOREAL experiment:

```bash
bash scripts/run_coop_loreal.sh \
  --data-root /data/TIP-data \
  --output-root /data/LOREAL-runs \
  --datasets oxford_pets \
  --resolutions 96 \
  --seeds 1 \
  --stage loreal
```

Run the common CoOp paper setting with 11 datasets, 3 low-resolution sizes, and 3 seeds:

```bash
bash scripts/run_coop_loreal.sh \
  --data-root /data/TIP-data \
  --output-root /data/LOREAL-runs \
  --datasets imagenet,caltech101,oxford_pets,stanford_cars,oxford_flowers,food101,fgvc_aircraft,sun397,dtd,eurosat,ucf101 \
  --resolutions 96,144,192 \
  --seeds 1,2,3 \
  --shots 16 \
  --stage all
```

Stage selection:

| `--stage` | Meaning |
| --- | --- |
| `all` | Run CoOp stage1, stage1_eval, CoOp stage2, stage2_eval, CoOp_REDIS stage3, and stage4 |
| `loreal` | Run the main paper path: CoOp stage1, CoOp stage2, CoOp_REDIS stage3, and stage4 |
| `stage1` | Train the standard-resolution CoOp student |
| `stage2` | Train the low-resolution CoOp student |
| `stage3` | Run CoOp_REDIS distillation; requires existing stage1/stage2 CoOp checkpoints |
| `stage4` | Evaluate the CoOp_REDIS distilled result; requires an existing stage3 checkpoint |

## CoOp_REDIS Config

`CoOp_REDIS` is the LOREAL-on-CoOp trainer in this repository. The default setup follows the paper's five-attribute form:

```bash
--attributes color,shape,size,structure,outline
--attr-tokens 2
--meta-dim 32
--lambda-hld 1.0
--lambda-lld 2.0
```

Config mapping:

| Option | Config key |
| --- | --- |
| Attribute text | `TRAINER.ATPROMPT.ATT1_TEXT` to `ATT5_TEXT` |
| Learnable tokens per attribute | `TRAINER.ATPROMPT.N_ATT1` to `N_ATT5` |
| Meta-net hidden dimension | `POW.DIM` |
| HLD coefficient `lambda1` | `POW.COEF1` |
| LLD coefficient `lambda2` | `POW.COEF2` |
| Override stage1 checkpoint path | `POW.STAGE1_DIR` |
| Override stage2 checkpoint path | `POW.STAGE2_DIR` |

The script currently requires exactly 5 attributes because the config and paper implementation are expanded as `ATT1..ATT5`. To use more attributes, extend both `Dassl.pytorch/dassl/config/defaults.py` and `configs/trainers/CoOp_REDIS/vit_b16_ep50.yaml`.

The paper uses dataset-specific attributes generated by an LLM and chosen for low-resolution robustness. The default values are generic attributes that make the CoOp + LOREAL method runnable; for strict reproduction of the CoOp tables, replace them with GPT-4o-generated attributes for each dataset.

## Outputs

The script writes outputs under:

```text
$OUTPUT_ROOT/output/CoOp_REDIS/base2new/train_base/$DATASET/
  CoOp_REDIS_stage1_students_pretraining_first/vit_b16_ep50.yaml/seed$SEED/
  CoOp_REDIS_stage2_students_pretraining_second/$RES/vit_b16_ep50.yaml/seed$SEED/
  CoOp_REDIS_stage3_students_sd/$RES/vit_b16_ep50.yaml/seed$SEED/
  CoOp_REDIS_stage4_students_new_test/$RES/vit_b16_ep50.yaml/seed$SEED/
```

Stage 3 automatically loads `prompt_learner/model.pth.tar-50` from the stage1 and stage2 directories. If checkpoints were moved manually, pass explicit paths to `train.py`:

```bash
POW.STAGE1_DIR /path/to/stage1 POW.STAGE2_DIR /path/to/stage2
```

## Notes

- `scripts/run_coop_loreal.sh` is the recommended entry point for the CoOp example.
- The CoOp + LOREAL path no longer depends on personal hard-coded dataset paths. Use `--data-root` for all dataset locations.
- Root `train.py` registers the local datasets plus the `CoOp` and `CoOp_REDIS` trainers used by this example.
