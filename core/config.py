"""Config loading + CLI overrides, and the two names that changed with the method.

The objective block used to be called `meanflow:` and checkpoint metadata
`meanflow_meta`; both are read under either name so configs and weights produced before
the rename keep working.
"""

from __future__ import annotations

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        print("[CONFIG] Loaded config keys:")
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply --set key.subkey=value overrides to a config dict.

    Supports nested dotted keys, and YAML-typed values so lists/bools/ints/floats
    parse automatically, e.g.:
        experiment.phase=test
        distributed.gpus=[0,1]
        data.max_len=20
        model.resume_path=/path/to/ckpt.pt
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set override must be key=value, got: {item!r}")
        key_str, val_str = item.split("=", 1)
        keys = key_str.strip().split(".")
        val = yaml.safe_load(val_str)

        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = val

    return config


SNAP_META_KEY = "snap_meta"
LEGACY_META_KEY = "meanflow_meta"


def snap_cfg(config: dict) -> dict:
    """The `snap:` objective block, accepting the legacy `meanflow:` name."""
    return config.get("snap") or config.get("meanflow") or {}


def ckpt_meta(ckpt: dict):
    """Config-identity metadata stored in a checkpoint, under either key."""
    return ckpt.get(SNAP_META_KEY) or ckpt.get(LEGACY_META_KEY)
