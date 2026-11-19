"""Point matching, supervised losses and weighted optimization objective."""
import torch
import torch.nn.functional as F
from torch import nn

from .matcher import HungarianMatcher_Crowd
from .caat import CAATCriterion


class DetectionCriterion(nn.Module):
    """Compute classification cross-entropy and matched-point MSE."""
    def __init__(self, options):
        super().__init__()
        if options["point_loss_mode"] != "official_compat":
            raise ValueError("Unsupported point_loss_mode; expected official_compat.")
        self.matcher = HungarianMatcher_Crowd(options["set_cost_class"], options["set_cost_point"])
        self.weight_dict = {"loss_ce": 1, "loss_points": options["point_loss_coef"]}
        empty_weight = torch.ones(2)
        empty_weight[0] = options["eos_coef"]
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs, targets):
        task_outputs = {"pred_logits": outputs["pred_logits"], "pred_points": outputs["pred_points"]}
        indices = self.matcher(task_outputs, targets)
        count = sum(len(target["labels"]) for target in targets)
        count = torch.as_tensor([count], dtype=torch.float, device=outputs["pred_logits"].device)
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(count)
            world_size = torch.distributed.get_world_size()
        normalizer = torch.clamp(count / world_size, min=1).item()


        batch_index = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_index = torch.cat([source for source, _ in indices])
        target_classes = torch.full(outputs["pred_logits"].shape[:2], 0, dtype=torch.int64, device=outputs["pred_logits"].device)
        matched_classes = torch.cat([target["labels"][destination] for target, (_, destination) in zip(targets, indices)])
        target_classes[(batch_index, source_index)] = matched_classes
        ce = F.cross_entropy(outputs["pred_logits"].transpose(1, 2), target_classes, self.empty_weight)

        predicted_points = outputs["pred_points"][(batch_index, source_index)]
        target_points = torch.cat([target["point"][destination] for target, (_, destination) in zip(targets, indices)], dim=0)
        point = F.mse_loss(predicted_points, target_points, reduction="none").sum() / normalizer

        return {"loss_ce": ce, "loss_point": point}


class VCSNetCriterion(nn.Module):
    """Combine Base supervision, optional CAAT loss and MBRE diagnostics."""
    def __init__(self, model, config):
        super().__init__()
        options = config["loss"]
        self.base_criterion = DetectionCriterion(options)
        if model.caat is not None:
            self.base_criterion = CAATCriterion(
                self.base_criterion,
                loss_coef=options["caat_coef"],
                smooth_l1_beta=options["caat_smooth_l1_beta"],
                local_scales=options["caat_local_scales"],
                positive_weight=options["caat_positive_weight"],
                global_weight=options["caat_global_weight"],
            )
        self.weight_dict = dict(self.base_criterion.weight_dict)

        object.__setattr__(self, "_model", model)

    def set_epoch(self, epoch):
        if hasattr(self.base_criterion, "set_epoch"):
            self.base_criterion.set_epoch(epoch)

    def forward(self, outputs, targets):
        losses = dict(self.base_criterion(outputs, targets))
        enhancer = self._model.mbre
        if enhancer is not None:
            diagnostics = enhancer.last_diagnostics
            reference = outputs["pred_logits"].new_zeros(())
            weights = diagnostics.get("branch_weight_mean", reference.new_zeros(4))
            for i, label in enumerate(("dw3", "dw5", "dilated_dw3", "identity")):
                losses["mbre_branch_weight_" + label] = weights[i]
            losses["mbre_gradient_audit_deferred"] = reference.new_ones(())
            for key in ("residual_mean_abs", "residual_max_abs", "branch_entropy_mean", "branch_occupancy_max"):
                losses["mbre_" + key] = diagnostics.get(key, reference)
            losses["mbre_active"] = reference.new_tensor(float(diagnostics.get("active", False)))
        return losses


def weighted_loss(losses, criterion):
    """Sum loss terms whose keys occur in the configured weight dictionary."""
    return sum(losses[key] * criterion.weight_dict[key] for key in losses if key in criterion.weight_dict)
