import json
import os
import sys
from pathlib import Path

import numpy as np
import psutil
from pyspark.sql import SparkSession

from dataset.session_dataset import count_labels, merge_dataset, split_train_test_pkl
from preprocess.pyspark.spark_aggregator import split_train_test, save_train_test, under_sample, save_parquet, \
    print_df_label_distribution
from utils.logging_utils import get_logger
from utils.run_cmd import run_cmd

logger = get_logger(__name__)


def init_spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    memory_gb = psutil.virtual_memory().available // 1024 // 1024 // 1024
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.memory", f"{memory_gb}g")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    return spark


def get_label_num(path):
    with open(path) as file:
        id2label = json.load(file)
        num_labels = len(id2label)
        return num_labels


def run_finetune(spark):
    num_labels = get_label_num(dataset_root / "id2label.json")
    run_cmd(f"sh ./finetune.sh {dataset_root} {num_labels} {0.1}")


def run_few_shot(spark):
    ratios = [round(num, 1) for num in np.arange(0.1, 0.2, 0.1)]
    num_labels = get_label_num(dataset_root / "id2label.json")

    for ratio in ratios:
        df = spark.read.parquet(str(dataset_path))
        df = under_sample(df, 2000)
        save_parquet(df, small_path)
        print_df_label_distribution(spark, small_path)

        df = spark.read.parquet(str(small_path))
        train_df, test_df = split_train_test(df, ratio)
        save_train_test(train_df, test_df, dataset_root)
        print_df_label_distribution(spark, train_path)
        print_df_label_distribution(spark, test_path)

        # num_labels = split_train_test(merged_name, test_size=ratio)
        run_cmd(f"sh ./finetune.sh {dataset_root} {num_labels} {ratio}")


# dataset_root = Path("data/Andriod")

dataset_root = Path("./data")
# dataset_root = Path("/root/PycharmProjects/FlowTransformer/data")
dataset_path = dataset_root / "dataset.parquet"
small_path = dataset_root / "dataset2000.parquet"
train_path = dataset_root / "train.parquet"
test_path = dataset_root / "test.parquet"


def run_single_finetune_pkl(train_path: Path, test_path: Path, ratio: float = 0.1):
    labels_cnt = count_labels(train_path)
    with open("results.txt", "a") as f:
        f.write(f"{train_path.as_posix()}, num_labels {len(labels_cnt)} {labels_cnt}, ratio {ratio}\n")

    run_cmd(f"sh ./finetune.sh {train_path.as_posix()} {test_path.as_posix()} {len(labels_cnt)} {ratio}")


def run_finetune_pkl(dataset_dir: Path, ratio: float = 0.1):
    # "USTC-TFC2016", "ISCX-Tor",
    # "IDS2018black", "USTC-TFC2016", "ISCX-Tor"
    datasets = [
        dataset_dir / entry
        for entry in [
            "ISCX-VPN-NonVPN-2016-App"
        ]
    ]

    for dataset in datasets:
        run_single_finetune_pkl(dataset / "train.pkl", dataset / "test.pkl", ratio)


def run_few_shot_pkl():
    ratios = [round(num, 1) for num in np.arange(0.1, 1.0, 0.1)]
    # merge_dataset(dataset_root / "merged.pkl", dataset_root / "train.pkl", dataset_root / "test.pkl")
    for ratio in ratios:
        # run_cmd(f"sh ./finetune.sh {dataset_root} {num_labels} {ratio}")
        run_finetune_pkl(dataset_root, ratio)


if __name__ == "__main__":
    try:
        # spk = init_spark()
        # run_finetune(spark=spk)
        # run_few_shot(spark=spk)
        # run_finetune_pkl()
        # run_few_shot_pkl()
        run_finetune_pkl(dataset_root)
    except Exception as e:
        logger.exception(e)
