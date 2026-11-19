"""Shared metric computation for separate validation and test entry points."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch

from inference import load_model, select_points
from utils.runtime import build_loader
from utils.metrics import summarize
from utils.checkpoint import sha256


@torch.no_grad()
def evaluate_loader(model, loader, config, device, count_only=False):
    model.eval()
    entries = []
    started = time.time()
    for index, (samples, targets) in enumerate(loader):
        outputs = model(samples.to(device))
        points, _ = select_points(outputs, config["evaluation"]["threshold"])[0]
        entries.append({"index": index, "image_id": int(targets[0]["image_id"].item()),
                        "selected_pred_points": points.cpu().reshape(-1, 2).tolist(),
                        "target_points": targets[0]["point"].cpu().reshape(-1, 2).tolist(),
                        "input_height": int(samples.shape[-2]), "input_width": int(samples.shape[-1])})
    if not entries:
        raise RuntimeError("No images were evaluated.")
    if count_only:
        import numpy as np
        errors = [len(entry["selected_pred_points"]) - len(entry["target_points"]) for entry in entries]
        result = {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(np.square(errors))))}
    else:
        result = summarize(entries, config["evaluation"]["point_radius"])
        result["elapsed_seconds"] = time.time() - started
        result["milliseconds_per_image"] = 1000 * result["elapsed_seconds"] / len(entries)
        result["input_shapes"] = sorted({f"{e['input_width']}x{e['input_height']}" for e in entries})
    return result, entries


def run_evaluation(checkpoint, data_root, output_dir, split, config_path=None,
                   device="cuda", val_decision=None):
    """Test reads its own list and applies the selected validation settings."""
    if split not in ("val", "test", "background"):
        raise ValueError("Expected val, test, or background.")
    checkpoint, data_root, output_dir = Path(checkpoint), Path(data_root), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Evaluation output directory already exists.")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    checkpoint_hash = sha256(checkpoint)
    decision = None
    if split == "test":
        if val_decision is None:
            raise ValueError("Test requires the selection.json produced by validation.")
        decision = json.loads(Path(val_decision).read_text(encoding="utf-8"))
        if decision.get("split") != "val" or decision.get("decision_frozen_before_test") is not True:
            raise ValueError("Expected a frozen validation selection.")
        if decision.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Test checkpoint differs from the validation-selected checkpoint.")
    model, config = load_model(checkpoint, config_path, device)
    config = copy.deepcopy(config)
    config["dataset"]["root"] = str(data_root)
    if decision is not None:
        # Threshold, padding and matching radius are carried over from Val.
        config["evaluation"] = decision["evaluation"]
        if config["model"] != decision["model"]:
            raise ValueError("Test model configuration differs from validation.")
        val_path = data_root / config["dataset"]["val_list"]
        if sha256(val_path) != decision["list_sha256"]:
            raise ValueError("Validation list differs from the selected validation run.")
    loader = build_loader(config, split)
    list_path = data_root / config["dataset"][split + "_list"]
    metrics, entries = evaluate_loader(model, loader, config, device)
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation.")
    result = {"checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_hash,
              "split": split, "list_file": list_path.name, "list_sha256": sha256(list_path),
              "evaluation": config["evaluation"], "model": config["model"],
              "threshold": config["evaluation"]["threshold"], "pad_block": config["evaluation"]["pad_block"],
              "radius": config["evaluation"]["point_radius"], "test_accessed": split == "test", "metrics": metrics}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "prediction_cache.json").write_text(json.dumps(entries) + "\n", encoding="utf-8")
    if split == "val":
        selection = {key: result[key] for key in ("split", "checkpoint_sha256", "list_sha256", "evaluation", "model")}
        selection["decision_frozen_before_test"] = True
        (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return result


def evaluation_main(split):
    parser = argparse.ArgumentParser(description=f"Evaluate the {split} split using its own explicit image list.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    if split == "test":
        parser.add_argument("--val-decision", type=Path, required=True,
                            help="selection.json from validation of the chosen checkpoint")
    args = parser.parse_args()
    result = run_evaluation(args.checkpoint, args.data_root, args.output_dir, split,
                            args.config, args.device, getattr(args, "val_decision", None))
    print(json.dumps({key: result["metrics"][key] for key in ("mae", "mse", "rmse", "bias", "precision", "recall", "f1")}, indent=2))
