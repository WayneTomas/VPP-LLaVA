import os

import torch
from torchvision import transforms
import torch.nn.functional as F

from ezcolorlog import root_logger as logger

from .base_encoder import BaseVisionTower, ProcessorWrapper
from llava.model.multimodal_encoder.detr_queries.dataset import train_transforms, test_transforms
from llava.model.multimodal_encoder.detr_queries.dataset.transforms import PIL_TRANSFORMS

from ultralytics import YOLO

class YOLOWorldVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super(YOLOWorldVisionTower, self).__init__(vision_tower, args, delay_load)
        self.is_loaded = False
        self.unfreeze_mm_vision_tower = True
        self.args = args

        # extract image resolution from model name
        if self.vision_tower_name.startswith("yoloworld"):
            self._image_size = None

        if not self.delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.vision_model = "yoloworld"

        if self.vision_tower_name.lower() == "yoloworld":
            model = YOLO('checkpoints/yoloworld/yolov8x-worldv2.pt')
            self.vision_tower = model.model
            self.vision_tower.train()
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        self._hidden_size = None
        self._image_size = None
        self._patch_size = None

        preprocess = []
        for t in test_transforms:
            _args = t.copy()
            preprocess.append(PIL_TRANSFORMS[_args.pop('type')](**_args))

        self.image_processor = ProcessorWrapper(preprocess, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        # self.vision_tower.to('cuda')
        logger.info(f"Loaded YOLO World model: {self.vision_tower_name} with hidden size: {self._hidden_size}, image size: {self._image_size}, patch size: {self._patch_size}\nSetting requires_grad: {self.unfreeze_mm_vision_tower}")
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
            # tokens = self.multi_scale_pooling(image_features)
            
            if self.vision_tower.training:
                # tokens = image_features[-1].view(images.shape[0], 144, 400)
                tokens = self.multi_scale_pooling(image_features)
            else:
                # tokens = image_features[-1][-1].view(images.shape[0], 144, 400)
                tokens = self.multi_scale_pooling(image_features[-1])
            return tokens
    def pool_feature(self, feat, output_size):
        # feat: [B, C, H, W]
        # output_size: (h, w)
        pooled = F.adaptive_avg_pool2d(feat, output_size)
        B, C, H, W = pooled.shape
        tokens = pooled.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        return tokens

    def multi_scale_pooling(self, feats):
        """
        feats: list of 3 tensors with shapes:
        [B, C, 20, 20], [B, C, 40, 40], [B, C, 80, 80]
        returns:
        tokens: [B, total_tokens, C]
        """
        tokens_small = self.pool_feature(feats[0], (4, 4))   # 16 tokens
        tokens_mid   = self.pool_feature(feats[1], (6, 6))   # 36 tokens
        tokens_large = self.pool_feature(feats[2], (8, 8))   # 64 tokens

        tokens = torch.cat([tokens_small, tokens_mid, tokens_large], dim=1)
        return tokens

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2