import os

import torch
from torchvision import transforms

from ezcolorlog import root_logger as logger

from .base_encoder import BaseVisionTower, ProcessorWrapper
from llava.model.multimodal_encoder.detr_queries.detr import build_detr
from llava.model.multimodal_encoder.detr_queries.dataset import train_transforms, test_transforms
from llava.model.multimodal_encoder.detr_queries.dataset.transforms import PIL_TRANSFORMS

class DETRQueriesVisionTower(BaseVisionTower):

    def __init__(self, vision_tower, args, delay_load=False):
        super(DETRQueriesVisionTower, self).__init__(vision_tower, args, delay_load)
        self.is_loaded = False
        self.unfreeze_mm_vision_tower = True
        self.args = args

        # extract image resolution from model name
        if self.vision_tower_name.startswith("detr"):
            self._image_size = None

        if not self.delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.vision_model = "detr"

        if self.vision_tower_name.lower() == "detr":
            self.vision_tower = build_detr(self.args)
            local_path = "llava/model/multimodal_encoder/detr_queries/checkpoints/detr-r101-dc5-a2e86def.pth"
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        if os.path.exists(local_path):
            logger.info(f"Loading `{self.vision_tower_name}` from local path: {local_path}")
            ckpt = torch.load(local_path, map_location=torch.device('cpu'))
        else:
            raise ValueError(f'Vision tower path not exist: {local_path}')
        pretrained_dict = ckpt['model']

        # # Load the pre-trained weights into the vision tower model

        missing_keys, unexpected_keys = self.vision_tower.load_state_dict(pretrained_dict, strict=False)
        print(f"DETR MISSING KESY: {missing_keys}")
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
        logger.info(f"Loaded DETR model: {self.vision_tower_name} with hidden size: {self._hidden_size}, image size: {self._image_size}, patch size: {self._patch_size}\nSetting requires_grad: {self.unfreeze_mm_vision_tower}")
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
            return image_features.squeeze(0)

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_token_len(self):
        return None