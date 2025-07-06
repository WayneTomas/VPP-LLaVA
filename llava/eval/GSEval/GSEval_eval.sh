#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="TMM_rebuttal_GSEval_13B"
# SPLIT=("refcoco_val", "refcoco_testA", "refcoco_testB")
DATASET="GSEval"
SPLIT="$DATASET"_test
GQADIR="./llava/eval/GSEval/"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/GSEval/model_GSEval_loader.py \
        --model-path ./checkpoints/13b/llava_detr-queries-llm2e-5grd_2e-4detr_2e-5_vpt_2e-4_ep3_unfreezeVIT \
        --question-file ./llava/eval/GSEval/GroundingSuite-Eval.jsonl \
        --image-folder ./llava/eval/GSEval/ \
        --answers-file ./grd_result_TMM_rebuttal/$DATASET/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./grd_result_TMM_rebuttal/$DATASET/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./grd_result_TMM_rebuttal/$DATASET/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd $GQADIR
# echo "Evaluating..."
# echo "Output file: $output_file"
# echo "Eval dir: $GQADIR"
python evaluate_grounding.py --image_dir ./GSEval --gt_file ./GroundingSuite-Eval.jsonl --pred_file $output_file --model_type llava --output_file infer.result


