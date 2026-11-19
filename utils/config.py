"""Load paper configurations and explicitly named model variants."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None, seed=None, variant=None):
    path = Path(path) if path else ROOT / "configs/vcsnet.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if seed is not None:
        config["seed"] = int(seed)
    if variant is not None:
        variants = json.loads((ROOT / "configs/variants.json").read_text(encoding="utf-8"))["variants"]
        if variant not in variants:
            raise ValueError("Unknown model variant: " + variant)
        config["model"].update(variants[variant])
        config["name"] = variant
    return config
