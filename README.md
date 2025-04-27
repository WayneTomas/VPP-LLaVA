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

![](https://black.readthedocs.io/en/stable/_static/license.svg)

## 💥 News
- **19 Mar, 2025**: :boom::boom:  Our paper "Visual Position Prompt for MLLM based Visual Grounding" has been submitted to IEEE Transactions on Multimedia (TMM).
-  **27 Apr, 2025**: :boom::boom: Our VPP-SFT dataset and the ablation 150K dataset have been released to Huggingface [[🤗 Huggingface Dataset](https://huggingface.co/datasets/wayneicloud/VPP-SFT/tree/main)]

## 👀 About VPP-LLaVA
Although Multimodal Large Language Models (MLLMs) excel at various image-related tasks, they encounter challenges in precisely aligning coordinates with spatial information within images, particularly in position-aware tasks such as visual grounding. This limitation arises from two key factors. First, MLLMs lack explicit spatial references, making it difficult to associate textual descriptions with precise image locations. Second, their feature extraction processes prioritize global context over fine-grained spatial details, leading to weak localization capability. To address this issue, we introduce VPP-LLaVA, an MLLM equipped with Visual Position Prompt (VPP) to improve its grounding capability.  VPP-LLaVA integrates two complementary mechanisms. The global VPP overlays learnable, axis-like embeddings onto the input image to provide structured spatial cues. The local VPP focuses on fine-grained localization by incorporating position-aware queries, which suggests probable object locations. We also introduce a VPP-SFT dataset with 0.6M samples, consolidating high-quality visual grounding data into a compact format for efficient model training. Training on this dataset with VPP enhances the model's performance, achieving state-of-the-art results on standard grounding benchmarks despite using fewer training samples compared to other MLLMs like MiniGPT-v2, which rely on much larger datasets ($\sim$21M samples). The code and VPP-SFT dataset will be available at https://github.com/WayneTomas/VPP-LLaVA upon acceptance.


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
