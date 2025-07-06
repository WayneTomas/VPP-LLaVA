import os

import torch
from torchvision import transforms

from ezcolorlog import root_logger as logger

from .base_encoder import BaseVisionTower, ProcessorWrapper
from llava.model.multimodal_encoder.anchor_detr.anchor_detr import build
from llava.model.multimodal_encoder.anchor_detr.dataset import train_transforms, test_transforms
from llava.model.multimodal_encoder.anchor_detr.dataset.transforms import PIL_TRANSFORMS

class AnchorDETRVisionTower(BaseVisionTower):

    def __init__(self, vision_tower, args, delay_load=False):
        super(AnchorDETRVisionTower, self).__init__(vision_tower, args, delay_load)
        self.is_loaded = False
        self.unfreeze_mm_vision_tower = True
        self.args = args

        # extract image resolution from model name
        if self.vision_tower_name.startswith("anchor_detr"):
            self._image_size = None

        if not self.delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.vision_model = "anchor_detr"

        if self.vision_tower_name.lower() == "anchor_detr":
            self.vision_tower = build(self.args)
            local_path = "llava/model/multimodal_encoder/anchor_detr/checkpoints/AnchorDETR_r101_dc5.pth"
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        if os.path.exists(local_path):
            logger.info(f"Loading `{self.vision_tower_name}` from local path: {local_path}")
            ckpt = torch.load(local_path, map_location=torch.device('cpu'))
        else:
            raise ValueError(f'Vision tower path not exist: {local_path}')
        pretrained_dict = ckpt['model']

        # Adjust the shape of the pre-trained weights to match the new model configuration
        if 'transformer.pattern.weight' in pretrained_dict:
            pretrained_pattern_weight = pretrained_dict['transformer.pattern.weight']
            if pretrained_pattern_weight.shape[0] > self.vision_tower.transformer.pattern.weight.shape[0]:
                # If the pre-trained model has more query patterns, truncate it
                pretrained_dict['transformer.pattern.weight'] = pretrained_pattern_weight[:self.vision_tower.transformer.pattern.weight.shape[0]]
            else:
                # If the pre-trained model has fewer query patterns, initialize the remaining patterns randomly
                new_pattern_weight = torch.zeros(self.vision_tower.transformer.pattern.weight.shape)
                new_pattern_weight[:pretrained_pattern_weight.shape[0]] = pretrained_pattern_weight
                pretrained_dict['transformer.pattern.weight'] = new_pattern_weight

        if 'transformer.position.weight' in pretrained_dict:
            pretrained_position_weight = pretrained_dict['transformer.position.weight']
            if pretrained_position_weight.shape[0] > self.vision_tower.transformer.position.weight.shape[0]:
                # If the pre-trained model has more query positions, truncate it
                pretrained_dict['transformer.position.weight'] = pretrained_position_weight[:self.vision_tower.transformer.position.weight.shape[0]]
            else:
                # If the pre-trained model has fewer query positions, initialize the remaining positions randomly
                new_position_weight = torch.zeros(self.vision_tower.transformer.position.weight.shape)
                new_position_weight[:pretrained_position_weight.shape[0]] = pretrained_position_weight
                pretrained_dict['transformer.position.weight'] = new_position_weight

        # Load the pre-trained weights into the vision tower model
        missing_keys, unexpected_keys = self.vision_tower.load_state_dict(pretrained_dict, strict=False)
        print(f"anchor_detr MISSING KEYS: {missing_keys}")
        print(f"anchor_detr UNEXPECTED KEYS: {unexpected_keys}")
        
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
        logger.info(f"Loaded Anchor-DETR model: {self.vision_tower_name} with hidden size: {self._hidden_size}, image size: {self._image_size}, patch size: {self._patch_size}\nSetting requires_grad: {self.unfreeze_mm_vision_tower}")
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
            return image_features

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2