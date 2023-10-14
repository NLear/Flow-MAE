import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional, Union, Dict, List, Set

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, ClassLabel, Features, Value, Sequence

from preprocess import FEATURE_COL, LABEL_COL, FEATURE_LEN_COL, FEATURE_LEN_MAX, DATASETS_FILE, LABEL_MAPPING_FILE


def preprocess_function(
        record: Union[List[Dict], Dict],
        feature_col: Optional[str] = FEATURE_COL,
        label_col: Optional[str] = LABEL_COL,
        feature_len_col: Optional[str] = FEATURE_LEN_COL,
        feature_len_max: Optional[int] = FEATURE_LEN_MAX,
) -> dict:
    byte_data = record[feature_col]
    if not isinstance(byte_data, list):
        byte_data = [byte_data]

    np_data_list = [
        np.frombuffer(data, dtype=np.uint8)[:feature_len_max] / 255.0 for data in byte_data
    ]

    padded_data_array = np.zeros((len(np_data_list), 1, feature_len_max), dtype=np.float32)
    for i, np_data in enumerate(np_data_list):
        padded_data_array[i, 0, :len(np_data)] = np_data

    padded_data_tensor = torch.from_numpy(padded_data_array)
    if padded_data_tensor.size(0) == 1:
        padded_data_tensor = padded_data_tensor.squeeze(0)

    record[feature_col] = padded_data_tensor

    return record


def read_parquet_to_datasetdict(train_parquet_path: str, test_parquet_path: str = None) -> DatasetDict:
    if test_parquet_path is None:
        # Load the dataset from the train_parquet_path
        dataset_dict = load_dataset(train_parquet_path)
    else:
        # Load the datasets from Parquet files
        train_dataset = load_dataset(train_parquet_path)["train"]
        test_dataset = load_dataset(test_parquet_path)["train"]
        # Add the datasets to the DatasetDict
        dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset_dict


def get_label_mapping(all_labels: Set):
    # Check if all labels are numeric
    all_numeric = all(isinstance(label, (int, float)) for label in all_labels)

    if not all_numeric:
        # Replace special characters with underscores
        all_labels = {
            re.sub('[^0-9a-zA-Z]+', '_', str(label))
            for label in all_labels
        }

        # Ignore case and sort
        all_labels = sorted(all_labels, key=lambda label: label.lower())
    else:
        # Sort numeric labels
        all_labels = sorted(all_labels)

    label_to_id = {label: i for i, label in enumerate(all_labels)}

    return label_to_id, all_labels


def get_label_mapping_from_dataset_dict(dataset_dict, label_col: Optional[str] = LABEL_COL):
    # Create the label name to ID mapping
    all_labels = set()
    for dataset in dataset_dict.values():
        all_labels.update(set(dataset.unique(label_col)))

    return get_label_mapping(all_labels)


def process_parquet_to_datasetdict(
        train_parquet_path: str, test_size: float = 0,
        test_parquet_path: Optional[str] = None,
        save_path: Optional[str] = None,
        feature_col: Optional[str] = FEATURE_COL,
        label_col: Optional[str] = LABEL_COL,
        feature_len_col: Optional[str] = FEATURE_LEN_COL,
        feature_len_max: Optional[int] = FEATURE_LEN_MAX,
) -> DatasetDict:
    dataset_dict = read_parquet_to_datasetdict(train_parquet_path, test_parquet_path)
    dataset_dict = dataset_dict.map(preprocess_function, batched=True)

    label_to_id, label_list = get_label_mapping_from_dataset_dict(dataset_dict)
    class_label = ClassLabel(num_classes=len(label_list), names=label_list)

    # Create a Features object with all columns from the dataset
    features = Features({
        label_col: class_label,
        feature_col: Sequence(feature=Sequence(
            feature=Value(dtype='float32', id=None), length=feature_len_max, id=None),
            length=1, id=None),
        feature_len_col: Value('int64'),
    })
    dataset_dict = dataset_dict.cast(features)

    # split train test
    if "test" not in dataset_dict and test_size > 0:
        dataset_dict = dataset_dict["train"].train_test_split(test_size=test_size, shuffle=True,
                                                              stratify_by_column=label_col)
    if save_path is None:
        save_path = Path(train_parquet_path).with_name(DATASETS_FILE)
    if save_path.exists():
        shutil.rmtree(save_path)

    dataset_dict.save_to_disk(save_path)
    with open(save_path / LABEL_MAPPING_FILE, "w") as f:
        json.dump(label_to_id, f, indent=4)

    return dataset_dict


def print_label_counts(dataset_dict: DatasetDict, label_col: Optional[str] = LABEL_COL):
    for dataset_name, dataset in dataset_dict.items():
        label_counts = Counter(dataset[label_col])

        print(f"Dataset {dataset_name}:")
        for label_name, count in label_counts.items():
            print(f"Label {label_name}: {count}")
        print()


if __name__ == "__main__":
    train_parquet_path = "./train_test_data/ISCX-VPN-NonVPN-2016-App/dataset.parquet"

    dataset_dict = process_parquet_to_datasetdict(train_parquet_path, feature_len_max=1024, test_size=0.1)

    print(dataset_dict)
    print_label_counts(dataset_dict)

    dataset_dict = DatasetDict.load_from_disk(Path(train_parquet_path).with_name(DATASETS_FILE).as_posix())
    print(dataset_dict)
