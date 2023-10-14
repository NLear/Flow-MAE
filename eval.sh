#! /bin/bash

THIS_DIR=$(dirname "$(readlink -f "$0")")

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  "$THIS_DIR"/eval.py \
  --model_name_or_path "./vit-mae-server/checkpoint-879472" \
  --dataset_name "/root/PycharmProjects/DATA/IDS2018Funetune_demo" \
  --validation_dir "./data/datacon1024/train1024.pkl" \
  --output_dir "./vit-mae-demo-eval-datacon" \
  --overwrite_output_dir \
  --return_entity_level_metrics True \
  --remove_unused_columns False \
  --num_channels 1 \
  --num_attention_heads 1 \
  --hidden_dropout_prob 0.0 \
  --mask_ratio 0 \
  --image_column_name "tcp.payload" \
  --norm_pix_loss \
  --num_labels 2 \
  --do_eval \
  --base_learning_rate 1e-4 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.05 \
  --num_train_epochs 0 \
  --warmup_ratio 0.0 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --logging_strategy steps \
  --logging_steps 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --load_best_model_at_end True \
  --save_total_limit 3 \
  --seed 1337 \
