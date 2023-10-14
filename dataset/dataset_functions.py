import json
import os
import random
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import as_completed
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Callable, Union, Tuple, Iterable, Optional

import datasets
import math
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset, Value, concatenate_datasets

from .files import GlobFiles
from utils.logging_utils import get_logger
from utils.multiprocess import BoundProcessPoolExecutor

logger = get_logger(__name__)


def dataset_load(root, file_dict):
    sub_dataset = datasets.load_dataset(
        "csv",
        data_dir=str(root),
        data_files=file_dict,
        skip_blank_lines=True,
        on_bad_lines='skip'
    )
    return sub_dataset


def prepare_dataset_single():
    dataset = datasets.load_dataset(
        "csv",
        data_dir="/root/PycharmProjects/DATA/IDS2018json/json_sessions/pcap",
        data_files={
            "train": "*TCP.csv",
        }
    )
    pprint(dataset)
    pprint(dataset['train'].features)
    pprint(dataset['train'][:5])
    dataset.save_to_disk("./tcp")


def prepare_dataset(num_proc=16):
    root = "/root/PycharmProjects/DATA/IDS2018json/json_sessions/"
    tcp_files = GlobFiles(root, "*TCP.csv", 200)
    dataset_dicts: Dict[str: DatasetDict] = {}
    save_root = Path("/mnt/data/IDS2018tcp_datasets")
    if not save_root.exists():
        save_root.mkdir()

    future_dict = {}
    for subdir, files in tcp_files.files.items():
        with BoundProcessPoolExecutor(qsize=num_proc, max_workers=num_proc) as executor:
            file_dict = {}
            for file in files:
                name = re.sub(r"[^.\w]+", "_", file.stem)
                file_dict[name] = str(file)
            # TODO
            #  1. remove dataset_load, use from_csv instead
            #  2. fix path issue
            future_dict[executor.submit(dataset_load, root, file_dict)] = name
            for future in as_completed(future_dict):
                try:
                    name = future_dict[future]
                    result_dict: DatasetDict = future.result()
                    rel_path = str(subdir.relative_to(root)) if str(subdir.relative_to(root)) else "default"
                    dataset_dicts[rel_path] = result_dict
                except Exception as e:
                    print(e)
                    traceback.print_exc(file=sys.stdout)
                    exit(1)

    print(dataset_dicts)
    for save_rel, dataset_dict in dataset_dicts.items():
        dataset_dict.save_to_disk(str(Path(save_root, save_rel)))
    info = {
        "splits": list(dataset_dicts.keys()),
        "data_dicts": {
            rel_path: {
                "pcap": pcap,
                "num_rows": dataset.num_rows,
                "num_streams": len(dataset.unique(column="tcp.stream"))
            }
            for rel_path, dataset_dict in dataset_dicts.items()
            for pcap, dataset in dataset_dict.items()
        }
    }
    with open(save_root.joinpath("dataset_dicts.json"), "w") as f:
        json.dump(info, f)


def test_load_dataset_from_csv():
    dataset = dataset_load(
        root="/root/PycharmProjects/DATA/IDS2018json/json_sessions/Wednesday-21-02-2018-pcap/pcap",
        file_dict={"train": "capPC1-172.31.67.35_TCP.csv"}
    )
    print(dataset)
    print(dataset["train"].features)
    streams = dataset["train"].unique(column=STREAM)
    print(len(streams), max(streams))

    dataset.save_to_disk("/root/PycharmProjects/DATA/IDS2018test")

    merged = dataset["train"].map(
        multiprocess_merge,
        batch_size=0,
        remove_columns=dataset["train"].column_names,
        batched=True,
    )
    print(merged)
    print(merged.features)
    streams = merged.unique(column=STREAM)
    print(len(streams), max(streams))

    filtered = merged.filter(lambda x: x[STREAM] == 0)
    print(filtered[:])

    merged.set_format("pandas")
    new_dataset_df = merged[:]
    frequencies = (
        new_dataset_df[STREAM]
        .value_counts()
        .to_frame()
        .reset_index()
        .rename(columns={"index": STREAM, STREAM: "frequency"})
    )
    print(frequencies)


STREAM = "tcp.stream"
PAYLOAD = "tcp.payload"
PAYLOAD_LENGTH = "payload_len"
SEQ_LENGTH_BYTES = 64
SEQ_LENGTH_HEX = 2 * SEQ_LENGTH_BYTES
SEQ_HEIGHT = 4
MAX_EXAMPLES = 1 * SEQ_HEIGHT

MAX_PIXELS = 1024
MAX_PIXELS_B = 2 * MAX_PIXELS


def multiprocess_merge_continues(
        examples: Dict[str, Iterable[Value]],
        columns_to_keep: List[str] = None,
) -> Dict[str, Iterable[Value]]:
    merged_column: Dict[str, str] = defaultdict(str)
    merged_examples: Dict[str: list] = defaultdict(list)

    for stream, payload in zip(examples[STREAM], examples[PAYLOAD]):
        if len(merged_column[str(stream)]) < MAX_PIXELS_B:
            merged_column[str(stream)] += payload[:]

    samples = random.sample(
        list(merged_column.items()),
        len(merged_column) // 32
    )

    for stream, payloads in samples:
        if len(payloads) > MAX_PIXELS_B:
            payloads = payloads[:MAX_PIXELS_B]
        payload_int = [
            int(payloads[i: i + 2], 16)
            for i in range(0, len(payloads), 2)
        ]

        pay_len = len(payload_int)
        merged_examples[PAYLOAD_LENGTH].append(pay_len)

        # if pay_len < MAX_PIXELS:
        #     payload_int.extend([0] * (MAX_PIXELS - pay_len))
        payload_tensor = torch.as_tensor(payload_int, dtype=torch.uint8)
        merged_examples[PAYLOAD].append(payload_tensor)
        merged_examples[STREAM].append(int(stream))

    return merged_examples


def multiprocess_merge(
        examples: Dict[str, Iterable[Value]],
        columns_to_keep: List[str] = None,
) -> Dict[str, Iterable[Value]]:
    columns = examples.keys()
    columns_to_keep = set(columns_to_keep).intersection(columns)
    columns_to_remove = set(columns).difference(columns_to_keep)

    merged_column: Dict[str, list] = defaultdict(list)
    overflow_to_sample_mapping: Dict[str: int] = {}
    merged_examples: Dict[str: list] = defaultdict(list)

    ex_columns = [examples[col] for col in columns_to_keep]
    for stream, payload in zip(examples[STREAM], examples[PAYLOAD]):
        merged_column[str(stream)].append(payload[:SEQ_LENGTH_HEX])

    samples = random.sample(
        list(merged_column.items()),
        len(merged_column) // 32
    )

    for stream, payloads in samples:
        max_payloads = min(len(payloads), MAX_EXAMPLES)
        for i in range(0, max_payloads, SEQ_HEIGHT):
            p_slice = payloads[i: i + SEQ_HEIGHT]
            if len(p_slice) < 4:
                break

            p_slice_uint8 = []
            for payload in p_slice:
                payload_int = [
                    int(payload[i: i + 2], 16)
                    for i in range(0, len(payload), 2)
                ]
                if len(payload_int) < SEQ_LENGTH_BYTES:
                    payload_int += [0] * (SEQ_LENGTH_BYTES - len(payload_int))
                p_slice_uint8.append(
                    torch.as_tensor(payload_int, dtype=torch.uint8)
                )
            example_pad_len = torch.stack(p_slice_uint8, dim=0)
            if SEQ_HEIGHT - example_pad_len.size(0) > 0:
                pad_height = torch.zeros(
                    (SEQ_HEIGHT - example_pad_len.size(0), SEQ_LENGTH_BYTES),
                    dtype=torch.uint8
                )
                example_pad_len_height = torch.cat((example_pad_len, pad_height), dim=0)
                merged_examples[PAYLOAD].append(torch.unsqueeze(example_pad_len_height, dim=0))
            else:
                merged_examples[PAYLOAD].append(torch.unsqueeze(example_pad_len, dim=0))
            merged_examples[STREAM].append(int(stream))

    # for key, values in examples.items():
    #     if key is STREAM:
    #         merged_examples[str(key)] = [
    #             values[j]
    #             for j in range(overflow_to_sample_mapping[str(i)])
    #             for i in values
    #         ]
    #     elif key is PAYLOAD:
    #         merged_examples[PAYLOAD] = merged_payloads[key]

    return merged_examples


if __name__ == "__main__":
    # test_load_dataset_from_json()
    # prepare_dataset()
    # test_load_from_disk()
    # preprocess_pretrain_dataset()

    # src = "/mnt/data/IDS2018tcp_datasets/"

    # dst = "/mnt/data/IDS2018tcp_payloads_slices"

    # dataset_dicts = DatasetDicts(root=src)
    # dataset_dicts.map(multiprocess_merge)

    exit(0)

    # columns = raw_pcap.column_names
    # columns_to_keep = ["tcp.stream", "tcp.payload"]
    # columns_to_remove = list(set(columns).difference(columns_to_keep))
    # raw_pcap = raw_pcap.map(
    #     merge,
    #     batched=True,
    #     batch_size=4,
    #     remove_columns=columns_to_remove
    # )
