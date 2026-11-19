"""Train VCSNet or Base with validation-based checkpoint selection."""
import argparse
import json
import math
from pathlib import Path
import time
import torch

from models import VCSNet
from losses import VCSNetCriterion, weighted_loss
from utils.config import load_config
from utils.runtime import seed_everything, build_loader, build_optimizer
from utils.checkpoint import FORMAT, initialize_from_official, load_model_weights, read_checkpoint
from evaluation import evaluate_loader


def train_one_epoch(model, criterion, loader, optimizer, device, epoch, max_norm=0.1):
    """Run forward, weighted loss, backward, gradient clipping and Adam updates."""
    model.train()
    criterion.train()
    criterion.set_epoch(epoch)
    totals = {}
    batches = 0
    for samples, targets in loader:
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        outputs = model(samples)
        losses = criterion(outputs, targets)
        total = weighted_loss(losses, criterion)
        if not math.isfinite(total.item()):
            raise FloatingPointError("Non-finite training loss.")
        optimizer.zero_grad()
        total.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        values = {"loss": float(total.detach()), "lr": optimizer.param_groups[0]["lr"]}
        values.update({key + "_unscaled": float(value.detach()) for key, value in losses.items()})
        values.update({key: float(value.detach() * criterion.weight_dict[key]) for key, value in losses.items() if key in criterion.weight_dict})
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    if batches == 0:
        raise RuntimeError("Training loader is empty.")
    return {key: value / batches for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=("VCSNet", "Base", "VCSM", "CAAT", "VCSM+CAAT"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-init", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.seed, args.variant)
    config["dataset"]["root"] = str(args.data_root)
    options = config["training"]
    device = torch.device(options["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training requires an available CUDA device.")
    if args.output_dir.exists():
        raise FileExistsError("Training output directory already exists.")
    official = args.official_init or Path(config["initialization"]["official_checkpoint"])
    if not args.resume and not official.is_file():
        raise FileNotFoundError("Official ShanghaiTech Part A initialization weights are required.")
    seed_everything(config["seed"])
    model = VCSNet(config).to(device)
    criterion = VCSNetCriterion(model, config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options["lr_drop"])
    train_loader = build_loader(config, "train")
    val_loader = build_loader(config, "val")
    start_epoch = config["initialization"]["start_epoch"]
    if args.resume:
        payload = read_checkpoint(args.resume)
        if payload.get("deployment"):
            raise ValueError("Deployment checkpoints cannot resume full-model training.")
        load_model_weights(model, payload)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["lr_scheduler"])
        start_epoch = payload["epoch"] + 1
    else:
        initialize_from_official(model, official)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_mae = best_rmse = float("inf")
    best_mae_epoch = best_rmse_epoch = None
    stale = 0
    history = args.output_dir / "history.jsonl"
    started = time.time()
    last_epoch = start_epoch - 1
    completion = "maximum_epoch_reached"
    for epoch in range(start_epoch, options["epochs"]):
        last_epoch = epoch
        stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, options["clip_max_norm"])
        scheduler.step()
        payload = {"format": FORMAT, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(), "epoch": epoch, "config": config}
        torch.save(payload, checkpoint_dir / "latest.pth")
        record = {"epoch": epoch, "train": stats}

        if epoch % options["eval_freq"] == 0 and epoch != 0:
            metrics, _ = evaluate_loader(model, val_loader, config, device, count_only=True)
            record["val"] = metrics
            payload.update({"val_mae": metrics["mae"], "val_rmse": metrics["rmse"]})
            if metrics["mae"] < best_mae:
                best_mae, best_mae_epoch, stale = metrics["mae"], epoch, 0
                torch.save(payload, checkpoint_dir / "best_mae.pth")
            else:
                stale += 1
            if metrics["rmse"] < best_rmse:
                best_rmse, best_rmse_epoch = metrics["rmse"], epoch
                torch.save(payload, checkpoint_dir / "best_rmse.pth")
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"epoch={epoch} loss={stats['loss']:.6f}", flush=True)
        if options["early_stop_patience"] and stale >= options["early_stop_patience"]:
            completion = "early_stopping"
            break
    result = {"last_epoch": last_epoch, "best_mae": best_mae, "best_rmse": best_rmse, "best_mae_epoch": best_mae_epoch,
              "best_rmse_epoch": best_rmse_epoch, "elapsed_seconds": time.time() - started, "completion": completion, "test_accessed": False}
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
