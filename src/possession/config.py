"""Configuration in config.yaml."""

import os
import sys
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _abs(path):
    """Relative routes"""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))

def load_config(path):
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    paths = cfg["paths"]
    paths["official_repo"] = _abs(paths["official_repo"])
    paths["cache_dir"] = _abs(paths["cache_dir"])
    paths["output_dir"] = _abs(paths["output_dir"])
    paths["data_dir"] = _abs(
        paths.get("data_dir") or os.environ.get("MTAH_DATA_DIR") or "../data")

    if not os.path.isfile(os.path.join(paths["official_repo"], "games.py")):
        raise FileNotFoundError()
    if paths["official_repo"] not in sys.path:
        sys.path.insert(0, paths["official_repo"])
    return cfg
