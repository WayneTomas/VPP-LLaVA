import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
# from .detr_encoder import DETRVisionTower
# from .dino_encoder import DinoVisionTower
# from .ddetr_encoder import DDETRVisionTower
from .detr_encoder_queries import DETRQueriesVisionTower
from .anchor_detr_encoder import AnchorDETRVisionTower
# from .yoloworld_encoder import YOLOWorldVisionTower


# original version
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

# write by wayne
def build_extra_vision_tower(vision_tower_cfg, **kwargs):
    # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'extra_vision_tower', None))
    vision_tower = 'detr'
    if 'detr' in vision_tower.lower():
        return DETRVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


# write by wayne
# def build_dino_vision_tower(vision_tower_cfg, **kwargs):
#     # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'extra_vision_tower', None))
#     vision_tower = 'facebook/dinov2-large'
#     if 'dinov2' in vision_tower.lower():
#         return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')


# write by wayne
def build_detr_quires_vision_tower(vision_tower_cfg, **kwargs):
    # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'extra_vision_tower', None))
    vision_tower = 'detr'
    if 'detr' in vision_tower.lower():
        return DETRQueriesVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


# write by wayne
def build_anchor_detr_vision_tower(vision_tower_cfg, **kwargs):
    # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'extra_vision_tower', None))
    vision_tower = 'anchor_detr'
    if 'anchor_detr' in vision_tower.lower():
        return AnchorDETRVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

# write by wayne
# def build_yoloworld_vision_tower(vision_tower_cfg, **kwargs):
#     # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'extra_vision_tower', None))
#     vision_tower = 'yoloworld'
#     if 'yoloworld' in vision_tower.lower():
#         return YOLOWorldVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')