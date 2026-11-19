"""Strict loading of public-format checkpoints and Base initialization."""
import copy
import hashlib
from pathlib import Path
import torch

FORMAT = "vcsnet_v1"


def sha256(path):
    """Compute a fingerprint of a user-supplied file at runtime."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path):
    """Read a trusted local checkpoint without modifying it."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("The checkpoint must contain a model state dictionary.")
    return payload


def load_model_weights(model, payload):
    if payload.get("format") != FORMAT:
        raise ValueError("Expected a vcsnet_v1 checkpoint with the public model names.")
    deployment = payload.get("deployment", {})
    if deployment.get("fold_mbre", False) and model.mbre is not None:
        model.eval()
        model.mbre.switch_to_deploy()
    if deployment.get("remove_caat", False):
        model.caat = None
    model.load_state_dict(payload["model"], strict=True)
    return model


def initialize_from_official(model, path):
    """Initialize shared Base layers from the ShanghaiTech Part A checkpoint."""
    state = read_checkpoint(path)["model"]
    incompatible = model.load_state_dict(state, strict=False)
    allowed = ("vcsm.", "caat.", "mbre.")
    invalid = [key for key in incompatible.missing_keys if not key.startswith(allowed)]
    if invalid or incompatible.unexpected_keys:
        raise RuntimeError("Base initialization mismatch: " + repr((invalid, incompatible.unexpected_keys)))
    return incompatible


def deployment_copy(model):
    deployed = copy.deepcopy(model).eval()
    if deployed.mbre is not None:
        deployed.mbre.switch_to_deploy()
    deployed.caat = None
    return deployed
