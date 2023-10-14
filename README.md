# Flow-MAE (Masked AutoEncoder for Traffic Classification)

## overview

This repository houses the codebase that underpins the findings presented in our manuscript titled, `Flow-MAE: Leveraging Masked AutoEncoder for Accurate, Efficient and Robust Malicious Traffic Classification`. We are honored to have our work recognized and currently accepted for publication by *RAID 2023* (The 26th International Symposium on Research in Attacks, Intrusions and Defenses). For those interested in a hands-on experience, the `master` branch provides our prototypical implementation of the `Flow-MAE`. Furthermore, for reproducibility and benchmarking purposes, we also include the dataset and associated benchmarks referenced in our study.

Our methodology draws inspiration from ET-BERT, a model that has garnered attention for its robust classification capabilities in network traffic. ET-BERT has elegantly integrated pre-training strategies from the Natural Language Processing (NLP) domain, specifically leveraging BERT, and further fine-tuned these to achieve commendable traffic classification results. In a parallel vein, our approach is predicated on the utilization of Masked AutoEncoders (MAE), a model heralding from the Computer Vision (CV) domain. We harnessed the inherent strengths of MAE for both pre-training and fine-tuning phases. One salient advantage of the MAE over BERT lies in its capability to process longer input sequences. Moreover, its block embedding layer boasts adaptive learning capabilities, which we postulate will augment the accuracy, efficiency, and robustness in classifying malicious network traffic.

To achieve our objectives, this repository seamlessly integrates elements from the MAE implementation and the widely-respected `Hugging Face transformer`. We acknowledge these foundational components. Our repository is further enhanced by  conceptual and practical advancement in the field.

## Introduction

Malicious traffic classification is crucial for Intrusion Detection Systems (IDS). However, traditional Machine Learning approaches necessitate expert knowledge and a significant amount of well-labeled data. Although recent studies have employed pre-training models from the Natural Language Processing domain, such as ET-BERT, for traffic classification, their effectiveness is impeded by limited input length and fixed Byte Pair Encoding.

To address these challenges, this paper presents Flow-MAE, a pre-training model that employs Masked AutoEncoders (MAE) from the Computer Vision domain to achieve accurate, efficient, and robust malicious network traffic classification. Flow-MAE overcomes these challenges by utilizing burst (a generic representation of network traffic) and patch embedding to accommodate extensive traffic length. Moreover, Flow-MAE introduces a self-supervised pre-training task, the Masked Patch Model, which captures unbiased representations from bursts with varying lengths and patterns.

Experimental results from six datasets reveal that Flow-MAE achieves new state-of-the-art accuracy (>0.99), efficiency (>900 samples/s), and robustness across diverse network traffic types. In comparison to the state-of-the-art ET-BERT, Flow-MAE exhibits improvements in accuracy and speed by 0.41\%-1.93\% and 7.8x-10.3x, respectively, while necessitating only 0.2\% FLOPs and 44\% memory overhead. The efficacy of the core designs is validated through few-shot learning and ablation experiments.

## Repository Overview

### Repository Structure

Here's a breakdown of the structure and the key components of the repository:

- **dataset/**: This directory contains common data operation codes and scripts. It serves as the primary hub for managing and processing data used across various stages of the project.
- **eval.py**: A Python script dedicated to evaluating the performance of trained models.
- **eval.sh**: Shell script facilitating the execution of the evaluation process.
- **finetune.py**: Python script that provides functionalities for fine-tuning the models.
- **finetune.sh**: Shell script facilitating the fine-tuning process.
- **model/**: Contains the architecture and model-related files that serve as the backbone for the `Flow-MAE` system.
- **notebooks/**: Directory for Jupyter notebooks related to data visualization and model interpretability.
- **preprocess/**: This directory hosts scripts and utilities for data preprocessing and transformation.
- **preprocess.py**: Main preprocessing script to get the data ready for training and evaluation.
- **pretrain/**: A directory containing resources and scripts dedicated to the pre-training phase of models.
- **pretrain.py**: The primary script guiding the pre-training phase.
- **pretrain.sh**: Shell script aiding in executing the pre-training process.
- **tools/**: A utility directory storing tools and scripts that aid in various tasks throughout the project.
- **utils/**: Contains utility functions and helper scripts commonly used across the repository.

## Building and Running the Code

### System Requirements

The experiments included in this repository have been tested on a testbed equipped with an i7-12700K CPU (8 P-cores @4.7GHz and 4 E-cores @3.6GHz), 64 GB DDR5 DRAM (@6000MT/s), and two NVIDIA GeForce 3090Ti GPUs (24 GB of GDDR6X memory each). The software environment of the testbed includes Ubuntu 22.04.1 LTS (kernel 5.15.0-50), Python 3.8.13, PyTorch 1.12.0, and CUDA 11.6.

Software prerequisites for the successful operation of the provided code are in the `requirements.txt`. Please ensure that your system is equipped with these applications before proceeding with the use of this repository's content.

```python
numpy==1.21.5
torch==1.12.1
transformers==4.24.0
torchvision==0.13.1
tqdm==4.65.0
datasets==2.12.0
fsspec==2022.11.0
pandas==2.0.3
scikit-learn==1.0.2
evaluate==0.4.0
click==8.0.4
dpkt==1.9.8
psutil==5.9.0
pyspark==3.3.2
scapy==2.4.3
pyarrow==8.0.0
```

### Running Pre-training

```bash
THIS_DIR=$(dirname "$(readlink -f "$0")")

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  "$THIS_DIR"/pretrain.py \
  --dataset_name "pretrain1024" \
  --train_dir [YOUR_TRAIN_DATA_FILE] \
  --validation_dir [YOUR_VALIDATION_DATA_FILE] \
  --output_dir "./vit-mae-output" \
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
```

### Running Fine-tuning

```bash
dataset_name=${1:-"USTC-TFC-2016"}
dataset_dir=${2:-"PATH TO DATASETS"}
test_ratio=${3:-0.1}
THIS_DIR=$(dirname "$(readlink -f "$0")")

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  "$THIS_DIR"/finetune.py \
  --dataset_name "$dataset_name" \
  --dataset_dir "$dataset_dir" \
  --train_val_split "$test_ratio" \
  --dataloader_num_workers 4 \
  --output_dir "./vit-mae-finetune-$dataset_name" \
  --overwrite_output_dir \
  --model_name_or_path "PATH TO PRE-TRANNING checkpoint_DIR" \
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
```

## Contact and Support

We encourage open collaboration and welcome any questions or suggestions regarding the repository. If you have any issues or need further assistance, please feel free to raise an issue in this repository or contact us directly.

## License

This project is licensed under the terms of the MIT license.

## Citation

Link to our paper: https://dl.acm.org/doi/10.1145/3607199.3607206

Link to the repository: https://github.com/NLear/Flow-MAE

Plain text:

`Zijun Hang, Yuliang Lu, Yongjie Wang, and Yi Xie. 2023. Flow-MAE: Leveraging Masked AutoEncoder for Accurate, Efficient and Robust Malicious Traffic Classification. In Proceedings of the 26th International Symposium on Research in Attacks, Intrusions and Defenses (RAID '23). Association for Computing Machinery, New York, NY, USA, 297–314. https://doi.org/10.1145/3607199.3607206`

Biber Tex:

```
@inproceedings{10.1145/3607199.3607206,
author = {Hang, Zijun and Lu, Yuliang and Wang, Yongjie and Xie, Yi},
title = {Flow-MAE: Leveraging Masked AutoEncoder for Accurate, Efficient and Robust Malicious Traffic Classification},
year = {2023},
isbn = {9798400707650},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3607199.3607206},
doi = {10.1145/3607199.3607206},
abstract = {Malicious traffic classification is crucial for Intrusion Detection Systems (IDS). However, traditional Machine Learning approaches necessitate expert knowledge and a significant amount of well-labeled data. Although recent studies have employed pre-training models from the Natural Language Processing domain, such as ET-BERT, for traffic classification, their effectiveness is impeded by limited input length and fixed Byte Pair Encoding. To address these challenges, this paper presents Flow-MAE, a pre-training model that employs Masked AutoEncoders (MAE) from the Computer Vision domain to achieve accurate, efficient, and robust malicious network traffic classification. Flow-MAE overcomes these challenges by utilizing burst (a generic representation of network traffic) and patch embedding to accommodate extensive traffic length. Moreover, Flow-MAE introduces a self-supervised pre-training task, the Masked Patch Model, which captures unbiased representations from bursts with varying lengths and patterns. Experimental results from six datasets reveal that Flow-MAE achieves new state-of-the-art accuracy (&gt;0.99), efficiency (&gt;900 samples/s), and robustness across diverse network traffic types. In comparison to the state-of-the-art ET-BERT, Flow-MAE exhibits improvements in accuracy and speed by 0.41\%-1.93\% and 7.8x-10.3x, respectively, while necessitating only 0.2\% FLOPs and 44\% memory overhead. The efficacy of the core designs is validated through few-shot learning and ablation experiments. The code is publicly available at https://github.com/NLear/Flow-MAE.},
booktitle = {Proceedings of the 26th International Symposium on Research in Attacks, Intrusions and Defenses},
pages = {297–314},
numpages = {18},
keywords = {Pre-training Model, Masked Patch Model, Malicious Traffic Classification, Masked AutoEncoder},
location = {Hong Kong, China},
series = {RAID '23}
}
```



## References

1. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16000-16009).

2. Xinjie Lin, Gang Xiong, Gaopeng Gou, Zhen Li, Junzheng Shi, and Jing Yu. 2022. ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification. In Proceedings of the ACM Web Conference 2022 (WWW '22). Association for Computing Machinery, New York, NY, USA, 633–642. https://doi.org/10.1145/3485447.3512217
