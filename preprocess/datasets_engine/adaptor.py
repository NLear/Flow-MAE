import os
import pyarrow as pa
from datasets import Dataset
from multiprocessing import Queue
from typing import Tuple

from preprocess.factory import register_factory, transform_pcap_factory
from preprocess.pyspark.process_packet import transform_packet


def preprocess_function(packet, label):
    feature, feature_len = transform_packet(packet)
    if feature is None or feature_len is None:
        return None

    return {"x": feature, "feature_len": feature_len, "labels": label}


def transform_pcap(packet_queue: Queue, num_producers: int, output_path: str):
    end_count = 0
    rows = []

    while True:
        item = packet_queue.get()
        if item is None:  # 检测到结束标志
            end_count += 1
            if end_count == num_producers:
                break
        else:
            batch, label = item
            for packet in batch:
                row = preprocess_function(packet, label)
                if row is not None:
                    rows.append(row)

    # Create a Dataset from the list of dictionaries
    dataset = Dataset.from_dict({k: [dic[k] for dic in rows] for k in rows[0]})

    # Save transformed Dataset as Arrow file
    dataset_path = os.path.join(output_path, "dataset")
    dataset.save_to_disk(dataset_path)

    print("Dataset processed and saved.")


@register_factory(transform_pcap_factory, "datasets")
class TransformPcapAdaptor:
    def __call__(self, *args, **kwargs):
        # Your adaptor implementation
        ...


if __name__ == "__main__":
    pass
