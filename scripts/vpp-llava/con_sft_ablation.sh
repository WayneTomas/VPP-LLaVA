#!/bin/bash

export VIT_LR=2e-6
export num_train_epochs=3
export output_dir=./checkpoints/TMM/llava_detr-queries_wo_txtprompt_llm2e-5grd_2e-4detr_2e-5_vpt_2e-4_ep${num_train_epochs}_freezeVIT
export data_path=./playground/data/train_json/llava_ablation_150k.json
export EXTRA_VISION_TOWER_DELAY_LOAD=True
wandb offline

deepspeed --master_port 15581 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b-zero3 \
    --version v1 \
    --data_path ${data_path}\
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_dir}\
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    # --unfreeze_mm_vision_tower True \
    # --mm_vision_tower_lr ${VIT_LR} \
