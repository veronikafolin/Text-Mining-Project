#!/bin/bash

python classify.py \
  --task_name task1 \
  --model_name_or_path dlicari/lsg16k-Italian-Legal-BERT \
  --dataset_name_local ../datasets/COMMA  \
  --do_train \
  --do_predict \
  --do_eval \
  --max_seq_length 4096 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir classification/output/comma/task1 \
  --overwrite_output_dir \
  --overwrite_cache \
  --max_eval_samples 200 \
  --gradient_checkpointing \
  --log_level error \
  --gradient_accumulation_steps 1 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --save_total_limit 1 \
  --weight_decay 0.01 \
  --label_smoothing_factor 0.1 \
  --max_train_samples 10 \
  --max_eval_samples 10 \
  --max_predict_samples 10 \
  #  --sortish_sampler \
  #  --fp16 \
