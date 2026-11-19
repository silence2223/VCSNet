"""Image-level counting errors and radius-based point localization metrics."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def max_match(prediction, target, radius):
    """Find a maximum-cardinality one-to-one matching within the given radius."""
    prediction = np.asarray(prediction, float).reshape(-1, 2)
    target = np.asarray(target, float).reshape(-1, 2)
    if len(prediction) == 0 or len(target) == 0:
        return 0
    distance = np.linalg.norm(prediction[:, None, :] - target[None, :, :], axis=2)
    invalid = (min(len(prediction), len(target)) + 1) * (radius + 1.0)
    rows, columns = linear_sum_assignment(np.where(distance <= radius, distance, invalid))
    return int(sum(distance[row, column] <= radius for row, column in zip(rows, columns)))


def summarize(predictions, radius=12.0):
    """Aggregate count, localization, density and background statistics."""
    if not predictions:
        raise ValueError("Cannot summarize an empty evaluation set.")
    per_image = []
    buckets = {}
    for entry in predictions:
        pred = np.asarray(entry["selected_pred_points"], float).reshape(-1, 2)
        target = np.asarray(entry["target_points"], float).reshape(-1, 2)
        tp = max_match(pred, target, radius)
        repeated = sum(int((np.linalg.norm(pred - point, axis=1) <= radius).sum()) >= 2 for point in target) if len(pred) else 0
        error = len(pred) - len(target)
        bucket = "0" if len(target) == 0 else "1-10" if len(target) <= 10 else "11-50" if len(target) <= 50 else "50+"
        row = {"image_id": entry["image_id"], "gt_count": len(target), "pred_count": len(pred), "error": error,
               "absolute_error": abs(error), "under": int(error < 0), "over": int(error > 0),
               "tp": tp, "fp": len(pred) - tp, "fn": len(target) - tp, "repeated_gt": int(repeated), "density_bucket": bucket}
        per_image.append(row)
        buckets.setdefault(bucket, []).append(row)
    errors = np.asarray([row["error"] for row in per_image])
    pred_total = sum(row["pred_count"] for row in per_image)
    gt_total = sum(row["gt_count"] for row in per_image)
    tp = sum(row["tp"] for row in per_image)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    background = buckets.get("0", [])
    return {
        "images": len(per_image), "mae": float(np.mean(np.abs(errors))), "mse": float(np.mean(errors * errors)),
        "rmse": float(np.sqrt(np.mean(errors * errors))), "bias": float(np.mean(errors)),
        "prediction_total": pred_total, "ground_truth_total": gt_total,
        "prediction_to_ground_truth_ratio": pred_total / max(1, gt_total),
        "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "fp": pred_total - tp, "fn": gt_total - tp,
        "under_images": sum(row["under"] for row in per_image), "over_images": sum(row["over"] for row in per_image),
        "repeat_rate": sum(row["repeated_gt"] for row in per_image) / gt_total if gt_total else 0.0,
        "background_images": len(background),
        "background_false_positive_total": sum(row["pred_count"] for row in background),
        "background_false_positive_mean": sum(row["pred_count"] for row in background) / len(background) if background else "not_available",
        "density_buckets": {bucket: {"images": len(rows), "gt_total": sum(row["gt_count"] for row in rows),
                                    "pred_total": sum(row["pred_count"] for row in rows), "mae": sum(row["absolute_error"] for row in rows) / len(rows)} for bucket, rows in sorted(buckets.items())},
        "per_image": per_image,
    }
