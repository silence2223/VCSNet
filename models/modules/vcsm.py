"""Visible vegetation evidence and semantically conditioned logit calibration."""
import torch
import torch.nn.functional as F
from torch import nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SUPPORTED_INDICES = ('exg', 'exgr', 'gli', 'vari', 'ngrdi', 'rgbvi')

def _signed_safe_denominator(value, eps):
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return sign * value.abs().clamp_min(eps)

class VisibleIndexBank(nn.Module):
    """Compute visible indices from RGB; VCSNet selects ExGR, NGRDI and RGBVI."""

    def __init__(self, indices=('exgr',), eps=0.0001):
        super().__init__()
        indices = tuple((name.strip().lower() for name in indices if name.strip()))
        unknown = sorted(set(indices) - set(_SUPPORTED_INDICES))
        if not indices:
            raise ValueError('At least one vegetation index is required')
        if unknown:
            raise ValueError('Unsupported vegetation indices: {}'.format(', '.join(unknown)))
        self.indices = indices
        self.eps = eps
        self.register_buffer('mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def out_channels(self):
        return len(self.indices)

    def forward(self, normalized_rgb):

        padding_mask = normalized_rgb.eq(0).all(dim=1, keepdim=True)

        rgb = (normalized_rgb * self.std + self.mean).clamp(0.0, 1.0)
        red, green, blue = rgb.chunk(3, dim=1)
        intensity = (red + green + blue).clamp_min(self.eps)
        red_n = red / intensity
        green_n = green / intensity
        blue_n = blue / intensity

        values = {
            'exg': 2.0 * green_n - red_n - blue_n,
            'exgr': 3.0 * green_n - 2.4 * red_n - blue_n,
            'gli': (2.0 * green - red - blue) / (2.0 * green + red + blue).clamp_min(self.eps),
            'vari': ((green - red) / _signed_safe_denominator(green + red - blue, self.eps)).clamp(-3.0, 3.0) / 3.0,
            'ngrdi': (green - red) / (green + red).clamp_min(self.eps),
            'rgbvi': (green.square() - red * blue) / (green.square() + red * blue).clamp_min(self.eps),
        }
        bank = torch.cat([values[name] for name in self.indices], dim=1)
        return bank.masked_fill(padding_mask, 0.0)


class RelativeVegetationEvidence(nn.Module):
    """Build 12 channels of absolute and relative evidence with training-time dropout."""

    def __init__(self, indices=('exgr', 'ngrdi', 'rgbvi'), sample_drop_probability=0.2, channel_drop_probability=0.1, local_kernels=(3, 7)):
        super().__init__()
        if not 0.0 <= sample_drop_probability < 1.0:
            raise ValueError('sample_drop_probability must lie in [0, 1)')
        if not 0.0 <= channel_drop_probability < 1.0:
            raise ValueError('channel_drop_probability must lie in [0, 1)')
        local_kernels = tuple((int(value) for value in local_kernels))
        if len(local_kernels) != 2:
            raise ValueError('exactly two local kernels are required')
        if any((value < 3 or value % 2 == 0 for value in local_kernels)):
            raise ValueError('local kernels must be odd integers >= 3')
        if local_kernels[0] >= local_kernels[1]:
            raise ValueError('local kernels must be strictly increasing')
        self.index_bank = VisibleIndexBank(indices)
        self.sample_drop_probability = float(sample_drop_probability)
        self.channel_drop_probability = float(channel_drop_probability)
        self.local_kernels = local_kernels

    @property
    def out_channels(self):
        return self.index_bank.out_channels * 4

    def _active_mask(self, indices):

        active = indices.new_ones(indices.shape[0], indices.shape[1], 1, 1)
        if self.training and self.channel_drop_probability > 0.0:
            active = (torch.rand_like(active) >= self.channel_drop_probability).to(dtype=indices.dtype)
        if self.training and self.sample_drop_probability > 0.0:
            sample_keep = (torch.rand(indices.shape[0], 1, 1, 1, device=indices.device, dtype=indices.dtype) >= self.sample_drop_probability).to(dtype=indices.dtype)
            active = active * sample_keep
        return active

    @staticmethod
    def _masked_average(value, valid, kernel_size):
        padding = kernel_size // 2
        numerator = F.avg_pool2d(value * valid, kernel_size=kernel_size, stride=1, padding=padding)
        denominator = F.avg_pool2d(valid, kernel_size=kernel_size, stride=1, padding=padding).clamp_min(1e-06)
        return numerator / denominator

    def forward(self, normalized_rgb, target_size):

        padding_mask = normalized_rgb.eq(0).all(dim=1, keepdim=True)
        valid = (~padding_mask).to(dtype=normalized_rgb.dtype)
        indices = torch.tanh(self.index_bank(normalized_rgb))
        active = self._active_mask(indices)
        indices = indices * active * valid
        indices = F.interpolate(indices, size=target_size, mode='bilinear', align_corners=False)
        valid = F.interpolate(valid, size=target_size, mode='nearest')
        active = active.expand(-1, -1, *target_size)
        channel_valid = valid * active

        local_small = self._masked_average(indices, channel_valid, self.local_kernels[0])
        local_large = self._masked_average(indices, channel_valid, self.local_kernels[1])
        second_moment = self._masked_average(indices.square(), channel_valid, self.local_kernels[0])
        local_deviation = (second_moment - local_small.square()).clamp_min(0.0).add(1e-06).sqrt()

        evidence = torch.cat((indices, torch.tanh(indices - local_small), torch.tanh(indices - local_large), local_deviation), dim=1)
        evidence = evidence * channel_valid.repeat(1, 4, 1, 1)
        active_fraction = active.mean(dim=1, keepdim=True) * valid
        relative_energy = evidence[:, self.index_bank.out_channels:].abs().mean(dim=1, keepdim=True)

        strength = active_fraction * relative_energy
        return (evidence, strength, valid)

class _RelativeDecisionBase(nn.Module):

    def __init__(self, indices=('exgr', 'ngrdi', 'rgbvi'), feature_channels=256, hidden_channels=16, sample_drop_probability=0.2, channel_drop_probability=0.1, local_kernels=(3, 7)):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError('hidden_channels must be positive')
        self.evidence = RelativeVegetationEvidence(indices=indices, sample_drop_probability=sample_drop_probability, channel_drop_probability=channel_drop_probability, local_kernels=local_kernels)
        self.vi_encoder = nn.Sequential(
            nn.Conv2d(self.evidence.out_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels, affine=False),
            nn.SiLU(inplace=True),
        )
        self.rgb_context = nn.Conv2d(feature_channels, hidden_channels, kernel_size=1, bias=False)

    def _condition(self, normalized_rgb, feature):
        """Gate vegetation evidence using enhanced P4 context and the validity mask."""
        evidence, strength, valid = self.evidence(normalized_rgb, feature.shape[-2:])
        vi_embedding = self.vi_encoder(evidence)

        rgb_gate = torch.sigmoid(self.rgb_context(feature))
        condition = vi_embedding * rgb_gate * valid
        return (condition, strength)

class VCSM(_RelativeDecisionBase):
    """Calibrate classification logits with a zero-initialized output layer."""
    integration = 'logit'

    def __init__(self, num_anchor_points=4, delta_max=1.5, **kwargs):
        super().__init__(**kwargs)
        if num_anchor_points <= 0:
            raise ValueError('num_anchor_points must be positive')
        if delta_max <= 0.0:
            raise ValueError('delta_max must be positive')
        hidden_channels = self.rgb_context.out_channels
        self.num_anchor_points = int(num_anchor_points)
        self.delta_max = float(delta_max)
        self.calibrator = nn.Conv2d(hidden_channels, self.num_anchor_points, kernel_size=1, bias=False)

        nn.init.zeros_(self.calibrator.weight)

    def effective_delta(self, raw_delta):
        """Bound each logit shift by 1.5; the class margin changes by twice the shift."""
        return self.delta_max * torch.tanh(raw_delta / self.delta_max)

    def forward(self, normalized_rgb, feature, logits):
        batch, anchors, classes = logits.shape
        if classes != 2:
            raise ValueError('VCSM expects two-class point classification logits')
        height, width = feature.shape[-2:]
        expected = height * width * self.num_anchor_points
        if anchors != expected:
            raise ValueError('anchor count {} does not match feature grid {}'.format(anchors, expected))
        condition, strength = self._condition(normalized_rgb, feature)

        margin = (logits[..., 1] - logits[..., 0]).reshape(batch, height, width, self.num_anchor_points)
        uncertainty = 1.0 - torch.tanh(margin.abs().mean(dim=-1, keepdim=False) / 2.0)

        condition = condition * (0.25 + 0.75 * uncertainty[:, None])
        raw_delta = self.calibrator(condition)
        delta_map = self.effective_delta(raw_delta)
        delta = delta_map.permute(0, 2, 3, 1).reshape(batch, anchors, 1)

        correction = torch.cat((-delta, delta), dim=-1)
        calibrated = logits + correction
        diagnostics = {'strength': strength, 'delta': delta_map, 'uncertainty': uncertainty[:, None]}
        return (calibrated, diagnostics)
