#! /bin/bash
dataset_name=${1:-"USTC-TFC-2016"}
dataset_dir=${2:-"/root/PycharmProjects/NLearn/train_test_data/USTC-TFC-2016/datasets"}
test_ratio=${3:-0.1}
THIS_DIR=$(dirname "$(readlink -f "$0")")

#torchrun \
#  --nnodes=1 \
#  --nproc_per_node=2 \
#  "$THIS_DIR"/finetune.py \
#  --model_name_or_path "/home/h/PycharmProjects/FlowTransformer/vit-mae-demo/checkpoint-875264" \
#  --dataset_name "/home/h/PycharmProjects/DATA/IDS2018Pretrain_demo" \
#  --train_dir "/home/h/PycharmProjects/DATA/IDS2018Pretrain_demo/train1024.pkl" \
#  --validation_dir "/home/h/PycharmProjects/DATA/IDS2018Pretrain_demo/test1024.pkl" \
#  --output_dir "./vit-mae-demo-finetune" \
#  --overwrite_output_dir \
#  --remove_unused_columns False \
#  --num_channels 1 \
#  --num_attention_heads 2 \
#  --mask_ratio 0.15 \
#  --image_column_name "tcp.payload" \
#  --norm_pix_loss \
#  --num_labels 10 \
#  --do_train \
#  --do_eval \
#  --base_learning_rate 1e-4 \
#  --lr_scheduler_type "cosine" \
#  --weight_decay 0.05 \
#  --num_train_epochs 25 \
#  --warmup_ratio 0.0 \
#  --per_device_train_batch_size 64 \
#  --per_device_eval_batch_size 64 \
#  --logging_strategy steps \
#  --logging_steps 10 \
#  --evaluation_strategy "epoch" \
#  --save_strategy "epoch" \
#  --load_best_model_at_end True \
#  --save_total_limit 3 \
#  --seed 1234 \
#  --fp16

#  --model_name_or_path "./vit-mae-server/checkpoint-879472" \
#  --model_name_or_path "/mnt/data/PycharmData/vit-mae-server/checkpoint-879472" \

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  "$THIS_DIR"/finetune.py \
  --dataset_name "$dataset_name" \
  --dataset_dir "$dataset_dir" \
  --train_dir "/root/PycharmProjects/FlowTransPkl/data/ISCX-VPN-NonVPN-2016-Service/train.pkl" \
  --validation_dir "/root/PycharmProjects/FlowTransPkl/data/ISCX-VPN-NonVPN-2016-Service/test.pkl" \
  --train_val_split "$test_ratio" \
  --dataloader_num_workers 4 \
  --output_dir "./vit-mae-finetune-$dataset_name" \
  --overwrite_output_dir \
  --model_name_or_path "/mnt/data/PycharmData/vit-mae-server/checkpoint-879472" \
  --return_entity_level_metrics True \
  --remove_unused_columns False \
  --num_channels 1 \
  --num_attention_heads 2 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --mask_ratio 0 \
  --image_column_name "layers_layerData" \
  --norm_pix_loss \
  --do_train \
  --do_eval \
  --base_learning_rate 1e-4 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.08 \
  --num_train_epochs 10 \
  --warmup_ratio 0.0 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --logging_strategy steps \
  --logging_steps 50 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --load_best_model_at_end True \
  --metric_for_best_model "eval_f1" \
  --greater_is_better True \
  --save_total_limit 3 \
  --seed 1337

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
