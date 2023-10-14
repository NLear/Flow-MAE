import argparse
import json
import multiprocessing
import re
import shutil
import time
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path

from preprocess.datasets_engine.adaptor import transform_pcap
from preprocess.pyspark.spark_adapter import read_and_fetch_packets


def clean_dirs(*dirs):
    for cur_dir in dirs:
        if cur_dir.exists():
            shutil.rmtree(cur_dir)
        cur_dir.mkdir(parents=True)


class PcapDict(dict):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = Path(root_dir)
        self._load_data()

    def _load_data(self):
        # Iterate through all subdirectories in the root directory
        for label_dir in self.root_dir.iterdir():
            # Skip if it's not a directory
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            label = re.sub('[^0-9a-zA-Z]+', '_', str(label))
            # Iterate through all pcap files in the subdirectory
            pcap_files = [
                pcap_file
                for pcap_file in label_dir.iterdir()
                if pcap_file.name.endswith(".pcap")
            ]

            # Store the label and the list of pcap files
            self[label] = pcap_files

    def __repr__(self):
        return f"PcapDict(root_dir={self.root_dir})"


def save_id2label(target_dir_path, id2label):
    with (target_dir_path / "id2label.json").open("w") as f:
        json.dump(id2label, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser(description="PCAP Preprocessing")
    parser.add_argument(
        "-s",
        "--source",
        default="/mnt/data2/ISCX-VPN-NonVPN-2016/ISCX-VPN-NonVPN-App",
        help="path to the directory containing raw pcap files",
    )
    parser.add_argument(
        "-d",
        "--dest",
        default="train_test_data/ISCX-VPN-NonVPN-2016-App",
        help="path to the directory for persisting preprocessed files",
    )
    parser.add_argument(
        "-n",
        "--njob",
        type=int,
        default=8,
        help="num of executors",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=5,
        help="maximum batch size for processing packets",
    )
    parser.add_argument(
        "--output-batch-size",
        type=int,
        default=5000,
        help="maximum batch for processing packets",
    )
    parser.add_argument(
        "-t"
        "--transform-type",
        choices=["adaptor", "AdaptorSpark"],
        default="AdaptorSpark",
        help="specify the type of transform_pcap to use",
    )
    parser.add_argument(
        "-a",
        "--aggregator",
        choices=["adaptor", "pysparkaggregator"],
        default="pysparkaggregator",
        help="Aggregator type to use, e.g., 'pyspark'"
    )

    args = parser.parse_args()
    return args


def main(args, source, target, njob, max_batch, output_batch_size):
    data_dir_path = Path(args.source)
    target_dir_path = Path(args.target)
    tmp_dir = Path("/tmp/spark_parquet")
    clean_dirs(target_dir_path)
    # clean_dirs(target_dir_path, tmp_dir)

    pcap_dict = PcapDict(str(data_dir_path))

    # 使用 Manager().Queue() 替换 multiprocessing.Queue()
    with multiprocessing.Manager() as manager:
        packet_queue = manager.Queue(maxsize=100)

        # 创建生产者进程
        with ProcessPoolExecutor(max_workers=args.njob) as executor:
            futures = []
            for label, label_path in pcap_dict.items():
                for pcap_path in label_path:
                    future = executor.submit(
                        read_and_fetch_packets,
                        packet_queue, pcap_path,
                        args.output_batch_size, args.max_batch,
                        label, tmp_dir)
                    futures.append(future)

            # 创建消费者进程
            consumer_process = multiprocessing.Process(
                target=transform_pcap,
                args=(packet_queue, len(futures))
            )
            consumer_process.start()

            # 等待生产者进程结束
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to read pcap file: {e}")

            # 通知消费者进程结束
            packet_queue.put(None)

        # 等待消费者进程结束
        consumer_process.join()

    aggregator(tmp_dir, target_dir_path, 1024, 5000)


if __name__ == "__main__":
    # # 可以根据需求自定义这些值
    t1 = time.time()
    args = get_args()
    main(args)

    t2 = time.time()
    print(f"duration: {t2 - t1:.2f} seconds")
