"""Multi-Branch Residual Enhancement and deployment folding."""
import torch
import torch.nn.functional as F
from torch import nn

PW_INIT_SEED = 20260730

def _init_pw(weight):
    """Initialize pointwise weights with a local seed and restore the CPU RNG state."""
    state = torch.get_rng_state()
    try:
        torch.manual_seed(PW_INIT_SEED)
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    finally:
        torch.set_rng_state(state)

def _fold(conv, bn):
    """Fold fixed evaluation-mode batch normalization into a convolution."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = -bn.running_mean * scale + bn.bias
    return (conv.weight * scale[:, None, None, None], bias)

def _dilate3(k):
    """Embed a dilation-2 3x3 kernel in a 5x5 support."""
    out = k.new_zeros(k.shape[0], k.shape[1], 5, 5)
    out[:, :, ::2, ::2] = k
    return out

def _pad3(k):
    """Zero-pad a 3x3 kernel to 5x5."""
    return F.pad(k, (1, 1, 1, 1))

class MBRE(nn.Module):
    """Enhance P4 with a residual initialized to zero."""
    integration = 'shared'
    mode = 'mbre_p4'

    def __init__(self, feature_channels=256, residual_max=0.15):
        super().__init__()
        if int(feature_channels) != 256 or float(residual_max) != 0.15:
            raise ValueError('MBRE freezes channels=256 residual_max=0.15')
        self.residual_max = float(residual_max)
        self.dw3 = nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.dw5 = nn.Conv2d(256, 256, 5, padding=2, groups=256, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.dilated_dw3 = nn.Conv2d(256, 256, 3, padding=2, dilation=2, groups=256, bias=False)
        self.bn_dilated = nn.BatchNorm2d(256)
        self.identity_bn = nn.BatchNorm2d(256)
        self.mixture_logits = nn.Parameter(torch.zeros(4, 256))
        self.pw = nn.Conv2d(256, 256, 1, bias=False)
        self._deploy = False
        self.last_diagnostics = {}
        self._init()

    def _init(self):

        for bn in (self.bn3, self.bn5, self.bn_dilated, self.identity_bn):
            nn.init.zeros_(bn.weight)
            nn.init.zeros_(bn.bias)
        _init_pw(self.pw.weight)

    def _branches(self, x):
        return (self.bn3(self.dw3(x)), self.bn5(self.dw5(x)), self.bn_dilated(self.dilated_dw3(x)), self.identity_bn(x))

    def _forward_raw(self, x):
        if self._deploy:
            raw = self.pw(self.deploy_dw(x))
            weights = x.new_zeros(4, 256)
        else:

            weights = torch.softmax(self.mixture_logits, dim=0)
            mixed = sum((w.view(1, -1, 1, 1) * b for w, b in zip(weights, self._branches(x))))

            raw = self.pw(mixed)
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(0) if not self._deploy else x.new_zeros(256)
        return (raw, weights, entropy)

    def forward(self, features):
        x = features[1]
        raw, weights, entropy = self._forward_raw(x)

        residual = self.residual_max * torch.tanh(raw / self.residual_max)
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(0) if not self._deploy else x.new_zeros(256)
        self.last_diagnostics = {'branch_weights': weights.detach(), 'branch_weight_mean': weights.detach().mean(1), 'branch_weight_std': weights.detach().std(1, unbiased=False), 'branch_entropy_mean': entropy.detach().mean(), 'branch_occupancy_max': weights.detach().max(0).values.mean(), 'residual_mean_abs': residual.detach().abs().mean(), 'residual_max_abs': residual.detach().abs().max(), 'active': bool(torch.any(residual.detach().abs() > 0))}
        return (x + residual, dict(self.last_diagnostics))

    @torch.no_grad()
    def equivalent_depthwise_kernel_bias(self):


        weights = torch.softmax(self.mixture_logits.detach(), dim=0)
        folded = []
        for conv, bn, transform in ((self.dw3, self.bn3, _pad3), (self.dw5, self.bn5, lambda k: k), (self.dilated_dw3, self.bn_dilated, _dilate3)):
            k, b = _fold(conv, bn)
            folded.append((transform(k), b))
        ident = torch.zeros_like(self.dw5.weight)

        ident[:, :, 2, 2] = 1
        scale = self.identity_bn.weight / torch.sqrt(self.identity_bn.running_var + self.identity_bn.eps)
        folded.append((ident * scale[:, None, None, None], -self.identity_bn.running_mean * scale + self.identity_bn.bias))
        dw = sum((w.view(-1, 1, 1, 1) * k for w, (k, _) in zip(weights, folded)))
        db = sum((w * b for w, (_, b) in zip(weights, folded)))
        return (dw, db)

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fold evaluation-mode branches while retaining pointwise mixing and the residual."""
        if self._deploy:
            return self
        if self.training:
            raise RuntimeError('folding requires eval mode')
        k, b = self.equivalent_depthwise_kernel_bias()
        self.deploy_dw = nn.Conv2d(256, 256, 5, padding=2, groups=256, bias=True).to(k)
        self.deploy_dw.weight.copy_(k)
        self.deploy_dw.bias.copy_(b)
        for name in ('dw3', 'bn3', 'dw5', 'bn5', 'dilated_dw3', 'bn_dilated', 'identity_bn', 'mixture_logits'):
            delattr(self, name)
        self._deploy = True
        return self
