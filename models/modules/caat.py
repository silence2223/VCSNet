"""CAAT auxiliary count head operating on P4 before MBRE."""
import torch
import torch.nn.functional as F
from torch import nn

class CAATHead(nn.Module):
    """Predict nonnegative regional counts with a softplus output."""

    def __init__(self, feature_channels=256, hidden_channels=24):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(int(feature_channels), int(hidden_channels), 3, padding=1),
            nn.GroupNorm(1, int(hidden_channels)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(hidden_channels), 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.0)

    def forward(self, p4):
        return F.softplus(self.net(p4))
