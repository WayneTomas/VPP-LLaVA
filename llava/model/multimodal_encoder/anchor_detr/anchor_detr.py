# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
AnchorDETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from llava.model.multimodal_encoder.anchor_detr.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer
import copy


class AnchorDETR(nn.Module):
    """ This is the AnchorDETR module that performs object detection """

    def __init__(self, backbone, transformer,  num_feature_levels, aux_loss=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                if _ == 0:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                else:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src).unsqueeze(1))
            masks.append(mask)
            assert mask is not None

        srcs = torch.cat(srcs, dim=1)

        out = self.transformer(srcs, masks)
        # print(out[-1].shape)

        return out[-1]


def build(args):
    num_classes = 91

    backbone = build_backbone()
    num_feature_levels = 1
    transformer = build_transformer()
    model = AnchorDETR(
        backbone,
        transformer,
        num_feature_levels=num_feature_levels,
        aux_loss=False
    )

    return model