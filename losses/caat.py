"""Regional count targets and multi-scale CAAT supervision."""
import torch
import torch.nn.functional as F
from torch import nn

def build_local_count_target(targets, feature_hw, image_hw, device, dtype):
    """Accumulate points on the feature grid; empty targets produce zeros."""
    height, width = (int(feature_hw[0]), int(feature_hw[1]))
    image_h, image_w = (float(image_hw[0]), float(image_hw[1]))
    target = torch.zeros((len(targets), 1, height, width), device=device, dtype=dtype)
    for batch_index, item in enumerate(targets):
        points = item['point'].to(device=device, dtype=dtype)
        if points.numel() == 0:
            continue

        xs = (points[:, 0] / max(image_w, 1.0) * width).floor().long().clamp(0, width - 1)
        ys = (points[:, 1] / max(image_h, 1.0) * height).floor().long().clamp(0, height - 1)
        flat = target[batch_index, 0].view(-1)
        flat.index_add_(0, ys * width + xs, torch.ones_like(xs, dtype=dtype))
    return target

class CAATCriterion(nn.Module):
    """Combine regional mean loss with a 0.25-weighted global term."""

    def __init__(self, base_criterion, loss_coef=0.05, smooth_l1_beta=1.0, local_scales=(1, 3, 7), positive_weight=4.0, global_weight=0.25):
        super().__init__()
        self.base_criterion = base_criterion
        self.loss_coef = float(loss_coef)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.local_scales = tuple((int(value) for value in local_scales))
        if not self.local_scales or any((value <= 0 or value % 2 == 0 for value in self.local_scales)):
            raise ValueError('local_scales must contain positive odd kernels')
        self.positive_weight = float(positive_weight)
        self.global_weight = float(global_weight)
        self.weight_dict = dict(base_criterion.weight_dict)
        self.weight_dict['loss_caat_local_count'] = self.loss_coef

    def set_epoch(self, epoch):
        if hasattr(self.base_criterion, 'set_epoch'):
            self.base_criterion.set_epoch(epoch)

    def forward(self, outputs, targets):
        losses = dict(self.base_criterion(outputs, targets))
        pred = outputs['caat_local_count']
        target = build_local_count_target(targets, pred.shape[-2:], outputs['caat_input_hw'], pred.device, pred.dtype)
        local_terms = []
        for kernel in self.local_scales:
            if kernel == 1:
                pred_local, target_local = (pred, target)
            else:
                pred_local = F.avg_pool2d(pred, kernel, stride=1, padding=kernel // 2) * float(kernel * kernel)
                target_local = F.avg_pool2d(target, kernel, stride=1, padding=kernel // 2) * float(kernel * kernel)
            per_cell = F.smooth_l1_loss(pred_local, target_local, beta=self.smooth_l1_beta, reduction='none')

            weights = 1.0 + self.positive_weight * target_local.clamp(0.0, 1.0)
            local_terms.append((per_cell * weights).sum() / weights.sum().clamp_min(1.0))
        local_loss = torch.stack(local_terms).mean()
        predicted_total = pred.flatten(1).sum(dim=1)
        target_total = target.flatten(1).sum(dim=1)
        normalizer = target_total.clamp_min(1.0).sqrt()
        global_loss = F.smooth_l1_loss(predicted_total / normalizer, target_total / normalizer, beta=self.smooth_l1_beta)

        losses['loss_caat_local_count'] = local_loss + self.global_weight * global_loss
        losses['caat_predicted_count'] = predicted_total.detach().mean()
        losses['caat_target_count'] = target_total.detach().mean()
        return losses

    def __getattr__(self, name):
        if name in {'base_criterion', 'loss_coef', 'smooth_l1_beta', 'local_scales', 'positive_weight', 'global_weight', 'weight_dict'}:
            return super().__getattr__(name)
        return getattr(self.base_criterion, name)
