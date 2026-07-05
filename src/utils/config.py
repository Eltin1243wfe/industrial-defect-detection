"""
Just a thin wrapper around yaml loading so I can do cfg.training.lr instead of
cfg["training"]["lr"] everywhere. Nothing fancy, saved me a lot of bracket typos.
"""

import random
import yaml
import numpy as np
import torch


class DotDict(dict):
    """dict that also lets you access keys as attributes, recursively"""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(val, dict):
            val = DotDict(val)
        return val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: str) -> DotDict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return DotDict(raw)


def set_seed(seed: int = 42):
    # locking down every source of randomness I know about so runs are reproducible
    # (torch's cudnn benchmark can still cause tiny nondeterminism on GPU, that's fine for us)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
