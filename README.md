# Visual Position Prompt for MLLM based Visual Grounding

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://scholar.google.com/citations?user=D-27eLIAAAAJ&hl=zh-CN' target='_blank'>Wei Tang<sup>*,1</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN' target='_blank'>Yanpeng Sun<sup>1</sup></a>&emsp;
    <a href='[https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN](https://scholar.google.com/citations?hl=zh-CN&user=uWe1d3YAAAAJ&view_op=list_works&sortby=pubdate)' target='_blank'>Nanyang Ye<sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN' target='_blank'>Qinying Gu<sup>2</sup></a>&emsp;
    <a href='https://imag-njust.net/zechaoli/' target='_blank'>Zechao Li<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>Nanjing University of Science and Technology;
    <sup>2</sup>Shanghai Artificial Intelliaence Laboratory&emsp;
    </br>
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWayneTomas%2FTransCP&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Updates
- **x x, 2025**: :boom::boom:  Our paper "Visual Position Prompt for MLLM based Visual Grounding" has been submitted to IEEE Transactions on Multimedia (TMM).


Although Multi-modal Large Language Models (MLLMs) excel at various image-related tasks, they encounter challenges in precisely aligning coordinates with spatial information within images, particularly in position-aware tasks such as visual grounding. To address this issue, we introduce VPP-LLaVA, a novel MLLM equipped with a Visual Position Prompt (VPP) designed to enhance its grounding capability. VPP-LLaVA employs both global and local prompting mechanisms for explicit spatial referencing. The global VPP overlays learnable, axis-like embeddings onto the input image, providing foundational spatial cues. Meanwhile, local VPP function through object queries, suggesting probable object locations and helping the decoder associate spatial details with textual inputs. Experimental results demonstrate that VPP-LLaVA achieves state-of-the-art performance on standard visual grounding benchmarks, notably with a relatively small grounding dataset ($\sim$0.6M samples), compared to other MLLMs like MiniGPT-v2, which utilize much larger datasets ($\sim$21M samples). The code, checkpoints, and the collected supervised fine-tuning dataset will be available at https://github.com/WayneTomas/VPP-LLaVA when the manuscript is accepted.
