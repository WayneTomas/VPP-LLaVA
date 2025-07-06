---
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: test
    path: GroundingSuite-Eval.jsonl
task_categories:
- image-segmentation
---
# for VPP-LLaVA on GSEval inference
## For detailed evaluation on the GSEval benchmark, please refer to the original GSEval GitHub repository for more details. We provide the following scripts here for reference: GSEval_eval.sh, model_GSEval_loader.py and evaluate_grounding.py.

# 🚀 GSEval - A Comprehensive Grounding Evaluation Benchmark

GSEval is a meticulously curated evaluation benchmark consisting of 3,800 images, designed to assess the performance of pixel-level and bounding box-level grounding models. It evaluates how well AI systems can understand and localize objects or regions in images based on natural language descriptions.

##  📊 Results
<div align="center">
<img src="./assets/gseval.png">
</div>

<div align="center">
<img src="./assets/gseval_box.png">
</div>


## 📊 Download GSEval
```
git lfs install
git clone https://huggingface.co/datasets/hustvl/GSEval
```

## 📚 Additional Resources
- **Paper:** [ArXiv](https://arxiv.org/abs/2503.10596)
- **GitHub Repository:** [GitHub - GroundingSuite](https://github.com/hustvl/GroundingSuite)