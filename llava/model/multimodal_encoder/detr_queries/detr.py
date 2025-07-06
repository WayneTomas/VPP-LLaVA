# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch.nn.functional as F
from torch import nn

from llava.model.multimodal_encoder.detr_queries.misc import NestedTensor

from .backbone import build_backbone
from llava.model.multimodal_encoder.detr_queries.transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, train_backbone, train_transformer, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.backbone = backbone

        if self.transformer is not None:
            hidden_dim = transformer.d_model
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        else:
            hidden_dim = backbone.num_channels

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        
        if self.transformer is not None and not train_transformer:
            for m in [self.transformer, self.input_proj]:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.num_channels = hidden_dim

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
        # #######################################################################################
        # written by wayne
        # del the nested_tensor_from_tensor_list surpport
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        # #######################################################################################
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        if self.transformer is not None:
            out = self.transformer(self.input_proj(src), mask, pos[-1], query_embed=self.query_embed.weight)
        else:
            out = [mask.flatten(1), src.flatten(2).permute(2, 0, 1)]
             
        return out[0]


def build_detr(args):
    backbone = build_backbone()
    lr_visu_cnn = 1e-5
    lr_visu_tra = 1e-5
    detr_enc_num = 6
    num_queries = 100
    train_backbone = lr_visu_cnn > 0
    train_transformer = lr_visu_tra > 0
    if detr_enc_num > 0:
        transformer = build_transformer(args)
    else:
        transformer = None

    model = DETR(
        backbone,
        transformer,
        num_queries=num_queries,
        train_backbone=train_backbone,
        train_transformer=train_transformer
    )
    return model
