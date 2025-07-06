#!/bin/bash

export EXTRA_VISION_TOWER_DELAY_LOAD=False
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="rebuttal_main"
# SPLIT=("refcoco_val", "refcoco_testA", "refcoco_testB")
DATASET="refcoco"
SPLIT="$DATASET"_testB
GQADIR="./llava/eval/refcoco_all"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/eval/refcoco_all/model_refcoco_loader.py \
        --model-path ./checkpoints/llava-vpp-7b \
        --question-file ./llava/eval/refcoco_all/$DATASET/$SPLIT.jsonl \
        --image-folder ./playground/data/ \
        --answers-file ./grd_result/$DATASET/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./grd_result/$DATASET/$SPLIT/$CKPT/merge.jsonl
current_dir=$(pwd)
output_file_abs="$current_dir/"${output_file#./}""

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./grd_result/$DATASET/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd $GQADIR
# echo "Evaluating..."
# echo "Output file: $output_file"
# echo "Eval dir: $GQADIR"
echo "$output_file_abs"
python eval_refcoco.py --src "$output_file_abs"
