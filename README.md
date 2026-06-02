# LOREAL

## Mitigating Low-Resolution Challenges in Prompt Learning with Attribute-Driven Self-Distillation

LOREAL is a prompt self-distillation framework for improving the low-resolution robustness of vision-language models. Instead of relying only on class-level prompt tuning, LOREAL excavates resolution-robust attribute semantics and uses them to contextualize prompts with visual information from different resolutions.

This repository currently provides example code for applying LOREAL on top of CoOp. The example follows the paper's core recipe: train two CoOp students at different resolutions, share attribute meta-nets between them, and optimize the low-resolution student with both low-level attribute distillation and high-level prediction distillation.

## Highlights

- Attribute-driven prompts: LOREAL augments the base prompt with attribute slots `S_k [A_k]`, where `A_k` is a robust attribute and `S_k` is generated from visual features.
- Cross-modality meta-nets: each attribute owns a lightweight meta-net `M_k`, mapping image features into learnable attribute prompt contents.
- Dual-student self-distillation: one student receives standard-resolution images, while the other receives low-resolution images. The two students share meta-nets and exchange visual semantics across resolutions.
- Low-Level Distillation (LLD): aligns generated attribute contexts across resolutions.
- High-Level Distillation (HLD): aligns output prediction distributions with KL divergence.
- Low-resolution inference: after distillation, the model uses low-resolution images and the learned meta-nets to build attribute-aware prompts at inference time.

## Method Overview

LOREAL starts from a prompt learning model such as CoOp and inserts attribute-aware prompt slots:

```text
A photo of a [CLASS] with S1 [A1] S2 [A2] ... SK [AK]
```

The learnable attribute contents are not static parameters. Given a visual feature `f_v`, LOREAL generates each attribute context through a meta-net:

```text
S_k = M_k(f_v)
```

During self-distillation, two students are pretrained at different resolutions:

- Student alpha processes standard-resolution images.
- Student beta processes low-resolution images.

The students bridge their visual semantics across resolutions. The standard-resolution branch receives attribute contexts generated from low-resolution visual features, and the low-resolution branch receives attribute contexts generated from standard-resolution visual features. LLD aligns the generated attribute contexts, while HLD aligns the prediction distributions. The final objective is:

```text
L = L_CE + lambda1 * L_HLD + lambda2 * L_LLD
```

## Code Map

| Concept | Implementation |
| --- | --- |
| CoOp baseline trainer | `trainers/coop.py` |
| LOREAL-on-CoOp trainer | `trainers/coop_redis.py` |
| Unified training entry | `train.py` |
| Unified launch script | `scripts/run_coop_loreal.sh` |
| CoOp config | `configs/trainers/CoOp/vit_b16_ep50.yaml` |
| LOREAL config | `configs/trainers/CoOp_REDIS/vit_b16_ep50.yaml` |
| Dataset configs | `configs/datasets/*.yaml` |

Inside `trainers/coop_redis.py`:

| Paper component | Code location |
| --- | --- |
| CoOp base prompt `P0` | `PromptLearner.ctx` |
| Attribute slots `S_k [A_k]` | `PromptLearner` |
| Meta-nets `S_k = M_k(f_v)` | `PromptLearner.metanets` |
| Cross-resolution bridge | `CustomCLIP.forward(..., student_visual=...)` |
| LLD, Eq. (7) | `CoOp_REDIS.low_level_distillation` |
| HLD, Eq. (8) | `CoOp_REDIS.forward_backward` |
| Final objective | `loss_ce + POW.COEF1 * loss_hld + POW.COEF2 * loss_lld` |

## Pipeline

The example pipeline has four stages:

1. `stage1`: pretrain a standard-resolution CoOp student.
2. `stage2`: pretrain a low-resolution CoOp student.
3. `stage3`: run LOREAL self-distillation with `CoOp_REDIS`, loading both pretrained students and training the shared attribute meta-nets.
4. `stage4`: evaluate the distilled low-resolution student on new classes.

The unified script handles all directory wiring and checkpoint paths.

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

If additional packages are missing at runtime, install the dependencies from `Dassl.pytorch/requirements.txt` first. The root `requirements.txt` contains lightweight CLIP/CoOp-side requirements.

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

## Quick Start

Start with a dry run to inspect the commands and output directories:

```bash
bash scripts/run_coop_loreal.sh \
  --data-root /data/TIP-data \
  --datasets oxford_pets \
  --resolutions 96 \
  --seeds 1 \
  --stage loreal \
  --dry-run
```

Run a minimal LOREAL experiment:

```bash
bash scripts/run_coop_loreal.sh \
  --data-root /data/TIP-data \
  --output-root /data/LOREAL-runs \
  --datasets oxford_pets \
  --resolutions 96 \
  --seeds 1 \
  --stage loreal
```

Run the common 11-dataset, 3-resolution, 3-seed setting:

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

## Stage Selection

| `--stage` | Meaning |
| --- | --- |
| `all` | Run stage1, stage1_eval, stage2, stage2_eval, stage3, and stage4 |
| `loreal` | Run the main training and evaluation path: stage1, stage2, stage3, and stage4 |
| `stage1` | Train the standard-resolution student |
| `stage2` | Train the low-resolution student |
| `stage3` | Run LOREAL self-distillation; requires existing stage1/stage2 checkpoints |
| `stage4` | Evaluate the distilled result; requires an existing stage3 checkpoint |

## LOREAL Configuration

The default setup follows the paper's five-attribute form:

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

The paper uses dataset-specific attributes generated by an LLM and chosen for low-resolution robustness. The default values are generic attributes that make the example runnable; for strict reproduction, replace them with GPT-4o-generated attributes for each dataset.

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

## Practical Notes

- `scripts/run_coop_loreal.sh` is the recommended entry point for the provided example.
- Dataset locations are controlled by `--data-root`; no personal hard-coded dataset path is required.
- Root `train.py` registers the local datasets and the trainers used by the launch script.
