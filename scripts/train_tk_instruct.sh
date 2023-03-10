#!/bin/bash
set -x

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_HOME=/usr/local/cuda/
export TRANSFORMERS_CACHE=/home/murphy/.cache/huggingface

port=$(shuf -i25000-30000 -n1)


# t0-3b

# deepspeed --master_port $port src/run_s2s.py \
#     --do_train \
#     --do_predict \
#     --predict_with_generate \
#     --model_name_or_path google/t5-xl-lm-adapt \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --generation_max_length 128 \
#     --max_num_instances_per_task 100 \
#     --max_num_instances_per_eval_task 100 \
#     --add_task_name False \
#     --add_task_definition True \
#     --num_pos_examples 2 \
#     --num_neg_examples 0 \
#     --add_explanation False \
#     --tk_instruct False \
#     --data_dir /home/murphy/pengfei_2022/Tk-Instruct/data/splits/default \
#     --task_dir /home/murphy/pengfei_2022/Tk-Instruct/data/tasks \
#     --output_dir /home/murphy/pengfei_2022/Tk-Instruct/output \
#     --overwrite_output_dir \
#     --cache_dir ./cache/ \
#     --overwrite_cache \
#     --per_device_train_batch_size 3 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-05 \
#     --num_train_epochs 2 \
#     --lr_scheduler_type constant \
#     --warmup_steps 0 \
#     --logging_strategy steps \
#     --logging_steps 100 \
#     --evaluation_strategy no \
#     --save_strategy steps \
#     --save_steps 2500 \
#     --deepspeed ds_configs/stage2.config \
#     --bf16 \
#     --run_name tk-instruct-t0-3b-training


# t5-3b
deepspeed --master_port $port src/run_s2s.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path t5-3b \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir /home/murphy/pengfei_2022/Tk-Instruct/data/splits/default \
    --task_dir /home/murphy/pengfei_2022/Tk-Instruct/data/tasks \
    --output_dir /home/murphy/pengfei_2022/Tk-Instruct/tk-instruct-t5-3b-training \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-05 \
    --num_train_epochs 2 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --run_name tk-instruct-t5-3b-training