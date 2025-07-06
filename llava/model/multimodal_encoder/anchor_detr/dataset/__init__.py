train_transforms = [
    dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640]),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, aug_translate=True)
]


test_transforms = [
    dict(type='RandomResize', sizes=[640], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, center_place=True)
]