import argparse
import importlib
import os
import sys


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DASSL_DIR = os.path.join(REPO_DIR, "Dassl.pytorch")
for path in (REPO_DIR, DASSL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from dassl.config import clean_cfg, get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger


CUSTOM_DATASETS = [
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "imagenet",
    "imagenet_a",
    "imagenet_r",
    "imagenet_sketch",
    "imagenetv2",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
]

CUSTOM_TRAINERS = [
    "coop",
    "coop_loreal",
]


def register_custom_modules():
    for name in CUSTOM_DATASETS:
        importlib.import_module(f"datasets.{name}")

    for name in CUSTOM_TRAINERS:
        importlib.import_module(f"trainers.{name}")


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def setup_cfg(args):
    cfg = get_cfg_default()

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    clean_cfg(cfg, args.trainer)
    cfg.freeze()

    return cfg


def main(args):
    register_custom_modules()

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory from which training resumes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="only positive value enables a fixed seed",
    )
    parser.add_argument("--source-domains", type=str, nargs="+")
    parser.add_argument("--target-domains", type=str, nargs="+")
    parser.add_argument("--transforms", type=str, nargs="+")
    parser.add_argument("--config-file", type=str, default="")
    parser.add_argument("--dataset-config-file", type=str, default="")
    parser.add_argument("--trainer", type=str, default="")
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--load-epoch", type=int)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    main(parser.parse_args())
