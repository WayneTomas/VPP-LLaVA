# [Visual Position Prompt for MLLM based Visual Grounding](https://arxiv.org/abs/2503.15426)

<!-- <p align="center" width="100%">
</p> -->

<div>
<div align="center">
    <a href='https://scholar.google.com/citations?user=D-27eLIAAAAJ&hl=zh-CN' target='_blank'>Wei Tang<sup>*,1,2</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN' target='_blank'>Yanpeng Sun<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN' target='_blank'>Qinying Gu<sup>&#x2709,2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=L6J2V3sAAAAJ&hl=zh-CN' target='_blank'>Zechao Li<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>Nanjing University of Science and Technology;
    <sup>2</sup>Shanghai Artificial Intelliaence Laboratory&emsp;
    </br>
    <sup>&#x2709</sup> Corresponding Author
    </br>
    <sup>*</sup> This work was done during his internship at Shanghai Artificial Intelliaence Laboratory.
    
</div>
 
 -----------------

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

## 💥 News
-  **7 July, 2025**: :boom::boom: All our VPP-LLaVA code and checkpoints have been released on GitHub and Huggingface, respectively: [[🤗 VPP-LLaVA-7b](https://huggingface.co/wayneicloud/VPP-LLaVA-7b)] and [[🤗 VPP-LLaVA-13b](https://huggingface.co/wayneicloud/VPP-LLaVA-13b)]
-  **27 Apr, 2025**: :boom::boom: Our VPP-SFT dataset and the ablation 150K dataset have been released to Huggingface [[🤗 Huggingface Dataset](https://huggingface.co/datasets/wayneicloud/VPP-SFT/tree/main)]
- **19 Mar, 2025**: :boom::boom:  Our paper "Visual Position Prompt for MLLM based Visual Grounding" has been submitted to IEEE Transactions on Multimedia (TMM).

## 👀 About VPP-LLaVA
Although Multimodal Large Language Models (MLLMs) excel at various image-related tasks, they encounter challenges in precisely aligning coordinates with spatial information within images, particularly in position-aware tasks such as visual grounding. This limitation arises from two key factors. First, MLLMs lack explicit spatial references, making it difficult to associate textual descriptions with precise image locations. Second, their feature extraction processes prioritize global context over fine-grained spatial details, leading to weak localization capability. To address this issue, we introduce VPP-LLaVA, an MLLM equipped with Visual Position Prompt (VPP) to improve its grounding capability.  VPP-LLaVA integrates two complementary mechanisms. The global VPP overlays learnable, axis-like embeddings onto the input image to provide structured spatial cues. The local VPP focuses on fine-grained localization by incorporating position-aware queries, which suggests probable object locations. We also introduce a VPP-SFT dataset with 0.6M samples, consolidating high-quality visual grounding data into a compact format for efficient model training. Training on this dataset with VPP enhances the model's performance, achieving state-of-the-art results on standard grounding benchmarks despite using fewer training samples compared to other MLLMs like MiniGPT-v2, which rely on much larger datasets ($\sim$21M samples). The code and VPP-SFT dataset will be available at https://github.com/WayneTomas/VPP-LLaVA upon acceptance.

## Install
1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/WayneTomas/VPP-LLaVA.git
cd VPP-LLaVA
```

2. Install Package
```Shell
conda create -n vpp-llava python=3.10 -y
conda activate vpp-llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
```

4. Install flash-attention v2
You can install `flash-attention` using the following command:
```bash
pip install flash-attn --no-build-isolation
```
However, if you encounter any issues with this method, we recommend downloading the specific version of the flash-attention wheel file from the Releases page and installing it manually. For example, you can download the flash_attn-2.7.0.post2+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl file and install it using the following command:
```Shell
pip install flash_attn-2.7.0.post2+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

### Quick Start With HuggingFace
<details>
<summary>Example Code</summary>

```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "checkpoints/llava-vpp-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `llava/model/builder.py` and the example code of visual grounding llava/eval/refcoco_all/model_refcoco_loader.py.
</details>

### Visual Instruction Tuning
1. Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_GRD_Chatterbox_genixer_revised.json](https://huggingface.co/datasets/wayneicloud/VPP-SFT/blob/main/llava_v1_5_GRD_Chatterbox_genixer_revised.json), and download the images from constituting datasets:

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

## Acknowledgements
This repo is changed from [LLaVA v1.5](https://github.com/haotian-liu/LLaVA). The repo also benifits form [ChatterBox (AAAI 2025)](https://github.com/sunsmarterjie/ChatterBox) and [Genixer (ECCV 2024)](https://github.com/zhaohengyuan1/Genixer)

Thanks for their wonderful works.

## Cite

```bibtex
@misc{tang2025visualpositionpromptmllm,
      title={Visual Position Prompt for MLLM based Visual Grounding}, 
      author={Wei Tang and Yanpeng Sun and Qinying Gu and Zechao Li},
      year={2025},
      eprint={2503.15426},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.15426}, 
}
```
```
paper link: https://arxiv.org/abs/2503.15426
```
