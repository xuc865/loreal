<div align="center">

<h1>LOREAL</h1>

<h3>Mitigating Low-Resolution Challenges in Prompt Learning with Attribute-Driven Self-Distillation</h3>

<p>
  <img alt="CVPR 2026 Highlight" src="https://img.shields.io/badge/CVPR%202026-Highlight-dc2626.svg">
  <a href="configs/trainers/CoOp_REDIS/vit_b16_ep50.yaml"><img alt="Trainer" src="https://img.shields.io/badge/trainer-CoOp_REDIS-0ea5e9.svg"></a>
  <a href="scripts/run_coop_loreal.sh"><img alt="Pipeline" src="https://img.shields.io/badge/pipeline-4%20stages-16a34a.svg"></a>
  <a href="docs/DATASETS.md"><img alt="Datasets" src="https://img.shields.io/badge/datasets-11%20benchmarks-7c3aed.svg"></a>
  <img alt="Low resolution" src="https://img.shields.io/badge/focus-low--resolution%20robustness-f97316.svg">
</p>

<img src="docs/loreal-readme-hero.svg" alt="LOREAL: attribute-driven self-distillation for low-resolution prompt learning" width="100%">

<p>
  <b>LOREAL turns fragile class-only prompts into resolution-aware prompts.</b><br>
  It excavates robust attribute semantics, generates visual-conditioned attribute tokens, and distills knowledge between standard-resolution and low-resolution students.
</p>

</div>

## Core Innovation

| Module | What changes | Why it matters |
| --- | --- | --- |
| <img src="https://img.shields.io/badge/01-Attribute%20Prompting-00d4ff.svg"> | Extends `A photo of a [CLASS]` into `S1 [color] S2 [shape] ... SK [attribute]` | The prompt no longer depends only on class names; it carries low-resolution-stable visual cues. |
| <img src="https://img.shields.io/badge/02-Meta--Nets-22c55e.svg"> | Learns `S_k = M_k(f_v)` for every attribute | Attribute tokens are generated from image features instead of being static text parameters. |
| <img src="https://img.shields.io/badge/03-Dual%20Students-a855f7.svg"> | Couples a standard-resolution student with a low-resolution student | The low-resolution branch learns from richer visual semantics without changing inference inputs. |
| <img src="https://img.shields.io/badge/04-LLD%20%2B%20HLD-f97316.svg"> | Aligns both generated attribute contexts and output distributions | The model transfers fine-grained prompt semantics and high-level predictions together. |

## Method at a Glance

```text
Low-resolution image
        |
        v
Visual feature f_v -----> Attribute meta-nets M_k -----> S_k prompt tokens
        |                         |                         |
        |                         v                         v
        |              S1 [color] S2 [shape] ... SK [attribute]
        |                         |
        v                         v
  Student beta  <---- self-distillation ---->  Student alpha
        |                LLD + HLD                 |
        v                                          v
Robust low-resolution prompt learning and inference
```

LOREAL starts from a prompt learning model such as CoOp and inserts attribute-aware prompt slots:

```text
A photo of a [CLASS] with S1 [A1] S2 [A2] ... SK [AK]
```

The learnable attribute contents are not static parameters. Given a visual feature `f_v`, LOREAL generates each attribute context through a meta-net:

```text
S_k = M_k(f_v)
```

During self-distillation, two students are pretrained at different resolutions. The standard-resolution branch receives attribute contexts generated from low-resolution visual features, and the low-resolution branch receives attribute contexts generated from standard-resolution visual features. LLD aligns the generated attribute contexts, while HLD aligns the prediction distributions:

```text
L = L_CE + lambda1 * L_HLD + lambda2 * L_LLD
```

This repository provides runnable code for applying LOREAL on top of CoOp. The example follows the paper's core recipe: train two CoOp students at different resolutions, share attribute meta-nets between them, and optimize the low-resolution student with both low-level attribute distillation and high-level prediction distillation.

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

## Acknowledgements

This repository is built on top of several excellent open-source projects:

- [DPC](https://github.com/jreion/dpc)
- [CoOp](https://github.com/KaiyangZhou/CoOp)
- [ATPrompt](https://github.com/zhengli97/ATPrompt)

We thank the authors and maintainers of these projects for releasing their code.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{wang2026loreal,
  title={LOREAL: Mitigating Low-Resolution Challenges in Vision-Language Models with Attribute-driven Prompt Self-Distillation},
  author={Wang, Xucong and Wang, Pengkun and Zhao, Zhe and Yu, Liheng and Mao, Rui and Wang, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={39152--39163},
  year={2026}
}
```
