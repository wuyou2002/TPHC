#!/bin/bash
echo "Start running..."
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch run_clm.py \
    --config_name configs/tinyllama_tphc.json \
    --tokenizer_name TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \
    --dataset_name JeanKaddour/minipile \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --block_size 2048 \
    --attn_implementation sdpa \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.015 \
    --learning_rate 3e-4 \
    --weight_decay 1e-1 \
    --bf16 \
    --dtype bfloat16 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 1 \
    --save_total_limit 1 \
    --save_strategy steps \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 500 \
    --logging_steps 50 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --report_to none \
    --output_dir outputs/tinyllama-tphc4
