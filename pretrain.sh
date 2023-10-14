#! /bin/bash

THIS_DIR=$(dirname "$(readlink -f "$0")")

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  "$THIS_DIR"/pretrain.py \
  --dataset_name "pretrain1024" \
  --train_dir "./data/IDS2018_4096/pretrain4096_0.75train.pkl" \
  --validation_dir "./data/IDS2018_4096/pretrain4096_0.75test.pkl" \
  --output_dir "./vit-mae-4096" \
  --overwrite_output_dir \
  --remove_unused_columns False \
  --num_channels 1 \
  --num_attention_heads 4 \
  --hidden_dropout_prob 0.0 \
  --mask_ratio 0.15 \
  --image_column_name "tcp.payload" \
  --norm_pix_loss \
  --do_train \
  --do_eval \
  --base_learning_rate 1e-4 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.05 \
  --num_train_epochs 500 \
  --warmup_ratio 0.0 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --logging_strategy steps \
  --logging_steps 10 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --load_best_model_at_end True \
  --save_total_limit 3 \
  --seed 1234 \
  --fp16

#deepspeed --num_gpus=2 \
#  "$THIS_DIR"/pretrain.py \
#  --deepspeed "$THIS_DIR"/pretrain/ds_config.json \
#  --dataset_name "./data/IDS2018_4096" \
#  --train_dir "./data/IDS2018_4096/pretrain4096_0.75train.pkl" \
#  --validation_dir "./data/IDS2018_4096/pretrain4096_0.75test.pkl" \
#  --output_dir "./vit-mae-4096" \
#  --overwrite_output_dir \
#  --remove_unused_columns False \
#  --num_channels 1 \
#  --num_attention_heads 6 \
#  --hidden_dropout_prob 0.0 \
#  --mask_ratio 0.15 \
#  --image_column_name "tcp.payload" \
#  --norm_pix_loss \
#  --do_train \
#  --do_eval \
#  --base_learning_rate 1e-4 \
#  --lr_scheduler_type "cosine" \
#  --weight_decay 0.05 \
#  --num_train_epochs 500 \
#  --warmup_ratio 0.0 \
#  --per_device_train_batch_size 128 \
#  --per_device_eval_batch_size 128 \
#  --logging_strategy steps \
#  --logging_steps 10 \
#  --evaluation_strategy "epoch" \
#  --save_strategy "epoch" \
#  --load_best_model_at_end True \
#  --save_total_limit 3 \
#  --seed 1337 \
#  --fp16

#deepspeed --num_gpus=2 \
#  "$THIS_DIR"/pretrain.py \
#  --deepspeed "$THIS_DIR"/pretrain/ds_config.json \
#  --dataset_name "/root/PycharmProjects/DATA/IDS2018Pretrain_single" \
#  --output_dir "./vit-mae-demo" \
#  --overwrite_output_dir \
#  --remove_unused_columns False \
#  --num_channels 1 \
#  --mask_ratio 0.15 \
#  --image_column_name "tcp.payload" \
#  --norm_pix_loss \
#  --do_train \
#  --do_eval \
#  --base_learning_rate 1.5e-4 \
#  --lr_scheduler_type "cosine" \
#  --weight_decay 0.05 \
#  --num_train_epochs 100 \
#  --warmup_ratio 0.01 \
#  --per_device_train_batch_size 128 \
#  --per_device_eval_batch_size 32 \
#  --logging_strategy steps \
#  --logging_steps 10 \
#  --evaluation_strategy "epoch" \
#  --save_strategy "epoch" \
#  --load_best_model_at_end True \
#  --save_total_limit 3 \
#  --seed 1337 \
#  --fp16
