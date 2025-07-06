import numpy as np
import random

import torch
from PIL import ImageFilter, ImageEnhance

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision import transforms


class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img


class RandomContrast(object):
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation

    def __call__(self, img):
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, img, target):
        if random.random() < 0.8:
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    img = self.rand_brightness(img)
                elif func_id == 1:
                    img = self.rand_contrast(img)
                elif func_id == 2:
                    img = self.rand_saturation(img)

        return img, target


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.], aug_blur=False):
        self.sigma = sigma
        self.p = 0.5 if aug_blur else 0.

    def __call__(self, input_dict):
        if random.random() < self.p:
            img = input_dict['img']
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            input_dict['img'] = img

        return input_dict


class ToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, img, target):
        for k in self.keys:
            target[k] = torch.as_tensor(target[k])
        return F.to_tensor(img), target


class RandomResize(object):
    def __init__(self, sizes, resize_long_side=True, record_resize_info=False):
        self.sizes = sizes
        self.resize_long_side = resize_long_side
        if resize_long_side:
            self.choose_size = max
        else:
            self.choose_size = min
        self.record_resize_info = record_resize_info

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        h, w = img.height, img.width
        ratio = float(size) / self.choose_size(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        img = F.resize(img, (new_h, new_w))

        return img, target



class Compose(object):
    def __init__(self, transforms):
        self.transforms = list()
        for t in transforms:
            args = t.copy()
            transform = PIL_TRANSFORMS[args.pop('type')](**args)
            self.transforms.append(transform)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class NormalizeAndPad(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640,
                 aug_translate=False, center_place=False, padding=True):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate
        self.center_place = center_place
        self.padding = padding

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        elif self.center_place:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)
        else:
            top = left = 0

        if self.padding:
            out_img = torch.zeros((3, self.size, self.size)).float()
            out_img[:, top:top + h, left:left + w] = img
            out_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
            target['mask'] = out_mask
        else:
            out_img = img

        return out_img, target

PIL_TRANSFORMS = {
    'ToTensor': ToTensor,
    'RandomResize': RandomResize,
    'NormalizeAndPad': NormalizeAndPad,
    'Compose': Compose,
    'ColorJitter': ColorJitter,
    'GaussianBlur': GaussianBlur,
}