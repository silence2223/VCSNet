"""Fold MBRE and remove CAAT on a model copy, then verify equivalent predictions."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import load_model
from utils.checkpoint import FORMAT, deployment_copy, sha256


@torch.no_grad()
def verify_deployment(model):
    """Compare original and deployment outputs using synthetic inputs."""
    model.eval()
    deployed = deployment_copy(model)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(6189)
    maximum = {"pred_logits": 0.0, "pred_points": 0.0}
    for shape in ((1, 3, 128, 128), (2, 3, 128, 160)):
        sample = torch.randn(shape, generator=generator, device=device)
        original = model(sample)
        converted = deployed(sample)
        for key in maximum:
            torch.testing.assert_close(original[key], converted[key], atol=2e-5, rtol=2e-5)
            maximum[key] = max(maximum[key], float((original[key] - converted[key]).abs().max()))
    return deployed, {"maximum_absolute_difference": maximum, "all_checks_passed": True,
                      "scope": "synthetic_inputs_only", "formal_dataset_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.verify_only and args.output is None:
        parser.error("Provide --output to save deployment weights, or use --verify-only.")
    if args.output is not None and args.output.exists():
        raise FileExistsError("Deployment output already exists.")
    original_sha = sha256(args.checkpoint)
    model, config = load_model(args.checkpoint, args.config, args.device)
    deployed, report = verify_deployment(model)
    assert sha256(args.checkpoint) == original_sha, "Input checkpoint must remain unchanged."
    report["source_checkpoint_sha256"] = original_sha
    if not args.verify_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": FORMAT, "model": deployed.state_dict(), "config": config,
                    "deployment": {"fold_mbre": True, "remove_caat": True},
                    "source_checkpoint_sha256": original_sha, "verification": report}, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
