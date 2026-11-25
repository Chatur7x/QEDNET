"""Reproducibility utilities."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import pennylane as qml

        qml.numpy.random.seed(seed)
    except ImportError:
        pass


def hash_config(config: dict) -> str:
    """Deterministic hash of configuration."""
    import yaml

    yaml_str = yaml.dump(config, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(yaml_str.encode()).hexdigest()[:16]


def get_environment_info() -> dict:
    """Capture environment for reproducibility."""
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
    }

    # Package versions
    for pkg in ["numpy", "scipy", "sklearn", "pennylane", "torch", "xgboost", "shap", "mlflow"]:
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = "not_installed"

    return info
