"""YAML config with dotted CLI overrides (``--set train.lr=5e-5``)."""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(d):
        if isinstance(d, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in d.items()})
        if isinstance(d, list):
            return [Cfg.wrap(v) for v in d]
        return d

    def to_dict(self):
        return json.loads(json.dumps(self))

    @property
    def work(self) -> Path:
        p = Path(self.work_dir)
        p = p if p.is_absolute() else ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_root(self) -> Path:
        p = Path(self.data.root)
        return p if p.is_absolute() else ROOT / p


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def _merge(base, over):
    for k, v in over.items():
        base[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def _read(path):
    """Read a yaml config; a top-level `_base: other.yaml` key inherits from that file."""
    path = Path(path)
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("_base", None)
    return _merge(_read(path.parent / base), d) if base else d


def load(path=None, overrides=()):
    d = _read(path or ROOT / "configs/default.yaml")
    for o in overrides:
        k, v = o.split("=", 1)
        _set(d, k, v)
    return Cfg.wrap(d)


def cli(description=None, extra=None):
    """Parse --config/--set (plus script-specific args), seed everything, return (cfg, args)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    if extra:
        extra(p)
    args = p.parse_args()
    cfg = load(args.config, args.set)
    seed_all(cfg.seed)
    with open(cfg.work / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return cfg, args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clone(cfg):
    return Cfg.wrap(copy.deepcopy(cfg.to_dict()))
