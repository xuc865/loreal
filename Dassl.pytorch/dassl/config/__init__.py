from .defaults import _C as cfg_default


def get_cfg_default():
    return cfg_default.clone()


def clean_cfg(cfg, trainer):
    """Remove unused trainers (configs).

    Aim: Only show relevant information when calling print(cfg).

    Args:
        cfg (_C): cfg instance.
        trainer (str): trainer name.
    """
    common_keys = {"NAME", "MODAL", "LEVEL", "ATPROMPT", "PROMPTKD"}
    trainer_key = trainer.upper() if trainer else ""
    dependency_keys = {
        "COOP_REDIS": {"COOP", "ATPROMPT", "PROMPTKD"},
    }

    keys = list(cfg.TRAINER.keys())
    for key in keys:
        if key in common_keys or key == trainer_key or key in dependency_keys.get(trainer_key, set()):
            continue
        cfg.TRAINER.pop(key, None)
