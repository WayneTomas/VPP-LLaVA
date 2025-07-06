# Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_GRD_Chatterbox_genixer_revised.json](https://huggingface.co/datasets/wayneicloud/VPP-SFT/blob/main/llava_v1_5_GRD_Chatterbox_genixer_revised.json) and put them in train_json folder, then download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- LLaVA-Pretrain: [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- ReferIt: Due to some limitations, please search and down it by yourself.

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── LLaVA-Pretrain
│   └── images
├── referit (for open-vocabulary test, optional)
│   └── images
└── vg
    ├── VG_100K
    └── VG_100K_2
```