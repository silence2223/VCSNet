"""Point regression, classification and four-anchor grids adapted from Tencent P2PNet."""
import numpy as np
import torch
from torch import nn

class RegressionModel(nn.Module):
    """Predict coordinate offsets using conv1, conv2 and the output layer."""

    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)

class ClassificationModel(nn.Module):
    """Predict two-class logits."""

    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)

        self.output_act = nn.Sigmoid()

    def _reshape_logits(self, out):
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, _ = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)
        return out2.contiguous().view(batch_size, -1, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        output_input = out
        logits = self._reshape_logits(self.output(output_input))
        return logits

def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    return anchor_points

def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))
    return all_anchor_points

class AnchorPoints(nn.Module):
    """Generate reference points; VCSNet uses stride 8 and four anchors per cell."""

    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // 2 ** x for x in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        return torch.from_numpy(all_anchor_points.astype(np.float32)).to(image.device)
