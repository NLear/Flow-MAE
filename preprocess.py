import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Tuple

import torch
from datasets import DatasetDict, Dataset
from torchvision.transforms import Normalize
from transformers import HfArgumentParser

from dataset.dataset_functions import multiprocess_merge, SEQ_LENGTH_BYTES, SEQ_HEIGHT, STREAM, PAYLOAD, \
    multiprocess_merge_continues
from dataset.dataset_dicts import DatasetDicts
from dataset.files import GlobFiles
from preprocess.pcap import copy_folder, FileRecord, extract_tcp_udp, \
    split_seesionns, trim_seesionns, traverse_folder_recursive, rm_small_pcap, json_seesionns
from preprocess.pipeline import CopyStage, Pipeline, TraverseStage
from pretrain.functions import tensor_pad
from utils.arguments import PreprocessArguments, DataTrainingArguments, StageArguments
from utils.file_utils import str2path
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_args() -> Tuple[PreprocessArguments, DataTrainingArguments]:
    parser = HfArgumentParser((PreprocessArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    return args


def tshark_extract_pipeline(pre_args):
    src_root, dst_root, output_root = str2path(
        pre_args.dataset_src_root_dir,
        pre_args.dataset_dst_root_dir,
        pre_args.output_dir,
    )

    assert src_root.is_dir()
    dst_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    stage_args = partial(
        StageArguments,
        output_dir=output_root,
        src_file=output_root.joinpath("src.txt"),
        dst_file=output_root.joinpath("dst.txt"),
        num_workers=pre_args.num_workers
    )

    pipeline_args = [
        (
            CopyStage,
            stage_args(
                name="trim_sessions",
                category="CopyStage",
                src_folder=src_root,
                dst_folder=dst_root.joinpath(pre_args.trim_folder),
                cmd="bash preprocess/trim_length.sh {1} {2} "
                    f"{pre_args.packet_length} "
            )
        ),
        # (
        #     CopyStage,
        #     stage_args(
        #         name="extract_tcp_udp",
        #         category="CopyStage",
        #         src_folder=dst_root.joinpath(pre_args.trim_folder),
        #         dst_folder=dst_root.joinpath(pre_args.tcp_udp_folder),
        #         cmd="bash preprocess/extract.sh ",
        #         num_workers=8
        #     )
        # ),
        # (
        #     TraverseStage,
        #     stage_args(
        #         name="rm_small_pcap",
        #         category="TraverseStage",
        #         src_folder=dst_root.joinpath(pre_args.tcp_udp_folder),
        #         cmd="bash preprocess/remove.sh {1} "
        #             f"{pre_args.min_packet_num} "
        #     )
        # ),
        # (
        #     CopyStage,
        #     stage_args(
        #         name="split_seesionns",
        #         category="CopyStage",
        #         src_folder=dst_root.joinpath(pre_args.trim_folder),
        #         dst_folder=dst_root.joinpath(pre_args.split_session_folder),
        #         file2folder=True,
        #         cmd="bash preprocess/split_sessions.sh {1} {2} "
        #             f"{pre_args.min_packet_num} "
        #             f"{pre_args.min_file_size} "
        #             f"flow "
        #             f"{pre_args.splitcap_path} ",
        #     )
        # ),
        # (
        #     CopyStage,
        #     stage_args(
        #         name="trim_sessions",
        #         category="CopyStage",
        #         src_folder=dst_root.joinpath(pre_args.split_session_folder),
        #         dst_folder=dst_root.joinpath(pre_args.trim_time_folder),
        #         cmd="bash preprocess/trim.sh {1} {2} "
        #             f"{pre_args.time_window} "
        #     )
        # ),
        # (
        #     CopyStage,
        #     stage_args(
        #         name="split_packets",
        #         category="CopyStage",
        #         src_folder=dst_root.joinpath(pre_args.tcp_udp_folder),
        #         dst_folder=dst_root.joinpath(pre_args.split_packets_folder),
        #         cmd="bash preprocess/split_pkt.sh {1} {2} "
        #             f"{pre_args.min_packet_num} "
        #             f"{pre_args.max_packet_num} "
        #             f"{pre_args.packet_num} "
        #     )
        # ),
        # (
        #     TraverseStage,
        #     stage_args(
        #         name="rm_small_pcap",
        #         category="TraverseStage",
        #         src_folder=dst_root.joinpath(pre_args.split_packets_folder),
        #         cmd="bash preprocess/remove.sh {1} "
        #             f"{pre_args.min_packet_num} "
        #     )
        # ),
        (
            CopyStage,
            stage_args(
                name="json_seesionns",
                category="CopyStage",
                src_folder=dst_root.joinpath(pre_args.trim_folder),
                dst_folder=dst_root.joinpath(pre_args.json_folder),
                cmd="bash preprocess/pcap2json.sh {1} {2} "
                    f"{pre_args.min_packet_num} "
                    f"{pre_args.min_file_size} "
            )
        ),
    ]

    pipeline = Pipeline(pipeline_args)
    pipeline.run()


# def csv2datasets_pipeline():
#     tcp_files = GlobFiles(
#         root="/mnt/data/IDS2018all/json_sessions",
#         file_pattern="*TCP.csv",
#         threshold=200
#     )
#     pprint(tcp_files.files)
#
#     dataset = DatasetDicts.from_csv_parallel(tcp_files.files)
#     pprint(dataset.shape)
#
#     dataset = dataset.map_parallel(multiprocess_merge, num_proc=16, batch_size=16,
#                                    columns_to_keep=[STREAM, PAYLOAD])
#     pprint(dataset.shape)
#
#     flatten = dataset.flatten_to_dataset_dict(axis=1)
#     pprint(flatten.shape)
#     flatten.save_to_disk("/mnt/data2/IDS2018Train")
#     json.dump(
#         obj={
#             "shape": flatten.shape,
#             "rows": flatten.num_rows
#         },
#         fp=open("/mnt/data2/IDS2018Train/record.json", "w")
#     )


def csv2datasets_pipeline():
    tcp_files = GlobFiles(
        root="/mnt/data3/FlowTrans/IDS2018_Finetune/json_sessions",
        file_pattern="*TCP.csv",
        threshold=100
    )
    # pprint(tcp_files.files)

    dataset = DatasetDicts.from_csv_parallel(tcp_files.files)
    pprint(dataset.shape)

    dataset = dataset.map_parallel(multiprocess_merge_continues, num_proc=48, batch_size=96,
                                   columns_to_keep=[STREAM, PAYLOAD])
    pprint(dataset.shape)

    flatten = dataset.flatten_to_dataset_dict(axis=1)
    pprint(flatten.shape)
    flatten.save_to_disk("/mnt/data3/FlowTrans/IDS2018_FinetuneData")
    json.dump(
        obj={
            "shape": flatten.shape,
            "rows": flatten.num_rows
        },
        fp=open("/mnt/data3/FlowTrans/IDS2018_FinetuneData/record.json", "w")
    )


def preprocess_images(example):
    example["pixel_values"] = [
        tensor_pad(example[PAYLOAD]).div(255)
    ]
    return example


if __name__ == "__main__":
    t1 = time.time()

    # pre_args = PreprocessArguments()
    # tshark_extract_pipeline(pre_args)

    csv2datasets_pipeline()

    t2 = time.time()
    print('run time: %s sec' % time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))
