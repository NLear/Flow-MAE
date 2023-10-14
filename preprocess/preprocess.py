import os
import sys
from pathlib import Path
from typing import Tuple

from transformers import HfArgumentParser

from pcap import copy_folder, FileRecord, extract_tcp_udp, \
    split_seesionns, trim_seesionns, traverse_folder_recursive, rm_small_pcap, json_seesionns
from utils.arguments import PreprocessArguments, DataTrainingArguments
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_args() -> Tuple["PreprocessArguments", "DataTrainingArguments"]:
    parser = HfArgumentParser((PreprocessArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    return args


def old_main():
    preprocess_args, dataset_args = get_args()
    logger.info(preprocess_args)
    logger.info(dataset_args)

    logger.info(f"data set name {preprocess_args.name}, preprocessing...")

    tcp_udp_folder = Path(preprocess_args.dataset_dst_root_dir).joinpath(
        preprocess_args.tcp_udp_folder
    )
    fr = FileRecord()
    copy_folder(
        preprocess_args.dataset_src_root_dir,
        tcp_udp_folder,
        file2folder=False,
        callback_fn=fr.append
    )
    fr.dump("./folder.txt", "./dst.txt")
    extract_tcp_udp("./folder.txt", "./dst.txt")

    split_session_folder = Path(preprocess_args.dataset_dst_root_dir).joinpath(
        preprocess_args.split_session_folder
    )

    fr = FileRecord()
    copy_folder(
        tcp_udp_folder,
        split_session_folder,
        file2folder=True,
        callback_fn=fr.append,
    )
    fr.dump("./folder.txt", "./dst.txt")
    split_seesionns("./folder.txt", "./dst.txt", preprocess_args.splitcap_path)

    trim_folder = Path(preprocess_args.dataset_dst_root_dir).joinpath(
        preprocess_args.trim_folder
    )
    fr = FileRecord()
    copy_folder(
        split_session_folder,
        trim_folder,
        file2folder=False,
        callback_fn=fr.append,
    )
    fr.dump("./folder.txt", "./dst.txt")
    trim_seesionns("./folder.txt", "./dst.txt")

    fr = FileRecord(is_pairs=False)
    traverse_folder_recursive(
        trim_folder,
        callback_fn=fr.append,
    )
    fr.dump("./files.txt")
    rm_small_pcap("./files.txt")

    json_folder = Path(preprocess_args.dataset_dst_root_dir).joinpath(
        preprocess_args.json_folder
    )
    fr = FileRecord()
    copy_folder(
        trim_folder,
        json_folder,
        file2folder=False,
        callback_fn=fr.append,
    )
    fr.dump("./folder.txt", "./dst.txt")
    json_seesionns("./folder.txt", "./dst.txt")

    fr = FileRecord()
    traverse_folder_recursive(
        json_folder,
        callback_fn=fr.append,
    )
    fr.dump("./files.txt")


if __name__ == "__main__":
    old_main()
