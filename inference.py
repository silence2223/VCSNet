"""Single-image point prediction."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from models import VCSNet
from datasets import image_transform
from utils.config import load_config
from utils.checkpoint import read_checkpoint, load_model_weights


def select_points(outputs, threshold=0.72):
    """Select foreground points with probability strictly above the threshold."""
    scores = outputs["pred_logits"].softmax(dim=-1)[..., 1]
    return [(points[probability > threshold], probability[probability > threshold])
            for points, probability in zip(outputs["pred_points"], scores)]


def load_model(checkpoint, config_path=None, device="cpu"):
    payload = read_checkpoint(checkpoint)
    config = load_config(config_path) if config_path else (payload.get("config") or load_config())
    model = VCSNet(config)
    load_model_weights(model, payload)
    return model.to(device).eval(), config


@torch.no_grad()
def predict_image(model, path, threshold=0.72, device="cpu"):
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("Cannot read image: " + str(path))
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    tensor = image_transform()(rgb).unsqueeze(0).to(device)
    points, scores = select_points(model(tensor), threshold)[0]
    return {"image": str(path), "count": len(points), "points": points.cpu().tolist(), "scores": scores.cpu().tolist(), "threshold": threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Inference output already exists.")
    model, config = load_model(args.checkpoint, args.config, args.device)
    result = predict_image(model, args.image, config["evaluation"]["threshold"], args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Predicted count: ", result["count"])


if __name__ == "__main__":
    main()
