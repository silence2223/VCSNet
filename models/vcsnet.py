"""VCSNet with VCSM classification calibration, CAAT supervision and MBRE features."""
from torch import nn

from .backbone import VGGBackbone, FPN
from .point_head import RegressionModel, ClassificationModel, AnchorPoints
from .modules.vcsm import VCSM
from .modules.caat import CAATHead
from .modules.mbre import MBRE


class VCSNet(nn.Module):
    """Connect the backbone, prediction heads and three VCSNet modules."""
    def __init__(self, config):
        super().__init__()
        options = config["model"]
        anchors_per_cell = options["row"] * options["line"]


        enhancer = (MBRE(256, options["mbre_residual_max"])
                    if options["enable_mbre"] else None)
        self.backbone = VGGBackbone()
        self.vcsm = VCSM(
            num_anchor_points=anchors_per_cell,
            delta_max=options["vcsm_delta_max"],
            indices=tuple(options["indices"]),
            feature_channels=256,
            hidden_channels=options["vcsm_hidden_channels"],
            sample_drop_probability=options["vcsm_sample_drop"],
            channel_drop_probability=options["vcsm_channel_drop"],
            local_kernels=tuple(options["vcsm_local_kernels"]),
        ) if options["enable_vcsm"] else None
        self.regression = RegressionModel(256, num_anchor_points=anchors_per_cell)
        self.classification = ClassificationModel(256, num_anchor_points=anchors_per_cell, num_classes=2)
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=options["row"], line=options["line"])
        self.fpn = FPN()
        self.mbre = enhancer
        self.caat = CAATHead(256, options["caat_hidden_channels"]) if options["enable_caat"] else None

    def forward(self, images):
        """Map normalized [B,3,H,W] RGB inputs to point predictions and auxiliary tensors."""

        features = self.backbone(images)
        pyramid = self.fpn([features[1], features[2], features[3]])
        original_p4 = pyramid[1]


        enhanced_p4 = original_p4
        if self.mbre is not None:
            enhanced_p4, _ = self.mbre(pyramid)


        offset = self.regression(enhanced_p4) * 100
        logits = self.classification(enhanced_p4)


        vcsm_state = None
        if self.vcsm is not None:
            logits, vcsm_state = self.vcsm(images, enhanced_p4, logits)
        anchors = self.anchor_points(images).repeat(images.shape[0], 1, 1)
        outputs = {"pred_logits": logits, "pred_points": offset + anchors}
        if vcsm_state is not None:
            outputs.update({
                "vi_decision_strength": vcsm_state["strength"],
                "vi_decision_delta": vcsm_state["delta"],
                "vi_decision_uncertainty": vcsm_state["uncertainty"],
            })


        if self.caat is not None:
            outputs["caat_local_count"] = self.caat(original_p4)
            outputs["caat_feature_hw"] = tuple(original_p4.shape[-2:])
            outputs["caat_input_hw"] = tuple(images.shape[-2:])
        return outputs
