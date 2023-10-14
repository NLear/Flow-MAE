import itertools
import os
import sys
from multiprocessing import Queue
from pathlib import Path

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StructType, StructField, LongType, BinaryType, StringType
from scapy.all import PcapReader

from preprocess.factory import BaseAdaptor
from preprocess.pyspark.process_packet import transform_packet

# initialise local spark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def read_and_fetch_packets(packet_queue, pcap_path, output_batch_size, max_batch, label):
    print(f"Reading from file: {pcap_path}")
    packet_reader = PcapReader(str(pcap_path))
    batch_count = 0

    while True:
        batch = list(itertools.islice(packet_reader, output_batch_size))
        if not batch or (max_batch and batch_count >= max_batch):
            packet_queue.put(None)
            break
        packet_queue.put((batch, label))
        batch_count += 1


class AdaptorSpark(BaseAdaptor):
    schema = StructType(
        [
            StructField("x", BinaryType(), True),
            StructField("feature_len", LongType(), True),
            StructField("labels", StringType(), True),
        ]
    )

    def preprocess_function(self, packet, label):
        feature, feature_len = transform_packet(packet)
        if feature is None or feature_len is None:
            return None

        return Row(x=feature, feature_len=feature_len, labels=label)

    def transform_pcap(self, packet_queue: Queue, num_producers: int, output_path: Path):
        # Initialize SparkSession
        spark = (
            SparkSession.builder
            .appName("PCAP Transformation")
            .master("local[*]")
            .config("spark.driver.memory", "16g")
            .getOrCreate()
        )

        end_count = 0
        file_counter = 0  # Initialize file counter

        while True:
            item = packet_queue.get()
            if item is None:  # 检测到结束标志
                end_count += 1
                if end_count == num_producers:
                    break
            else:
                batch, label = item

                # Parallelize the batch of packets and apply packet_to_dict_dpkt
                packets_rdd = spark.sparkContext.parallelize(batch)
                transformed_rdd = packets_rdd.map(
                    lambda packet: self.preprocess_function(packet, label)
                ).filter(lambda x: x is not None)

                # Create a DataFrame from the transformed RDD
                transformed_df = spark.createDataFrame(transformed_rdd, schema=self.schema)
                # Increment file counter
                file_counter += 1
                # Save transformed DataFrame as Parquet file
                file_name = Path(output_path) / f"part-{file_counter:04d}.parquet"
                transformed_df.write.mode("append").parquet(str(file_name))

        print("All files processed and saved.")

    def __call__(self, *args, **kwargs):
        return self.transform_pcap(*args, **kwargs)
