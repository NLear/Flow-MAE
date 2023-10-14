import os
import os
import shutil
import sys
import time
from pathlib import Path

import binascii
import click
import dpkt
import pandas as pd
import psutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, LongType, StringType, \
    BinaryType


COLUMNS = ["label_id", "file_path", "packet", "length"]


def init_spark():
    # initialise local spark
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


def write_batch(output_path, batch_index, pd_frame):
    part_output_path = Path(
        str(output_path.absolute()) + f"_part_{batch_index:04d}.parquet.gzip"
    )
    pd_frame.to_parquet(part_output_path, compression='gzip')


def transform_single_pcap(
        pcap_root: Path = None,
        pcap_path: Path = None,
        target_dir_path: Path = None,
        label_id: int = None,
        output_batch_size: int = 10000,
        max_batch: int = None,
):
    cur_tar_dir = target_dir_path / pcap_path.relative_to(pcap_root)
    cur_tar_dir.mkdir(parents=True, exist_ok=True)
    output_path = cur_tar_dir / (pcap_path.name + ".transformed")
    print("Processing", pcap_path)

    pd_frame = pd.DataFrame(columns=COLUMNS)
    batch_index = 0

    with open(pcap_path.as_posix(), "rb") as pcap_f:
        for index, (ts, buf) in enumerate(dpkt.pcap.Reader(pcap_f)):
            if buf is not None:
                pd_frame.loc[index] = [label_id, pcap_path.as_posix(), buf, len(buf)]

                # write every batch_size packets, by default 10000
                if not pd_frame.empty and len(pd_frame) == output_batch_size:
                    write_batch(output_path, batch_index, pd_frame)
                    batch_index += 1
                    pd_frame = pd.DataFrame(columns=COLUMNS)

                if max_batch is not None and batch_index >= max_batch:
                    break

    # final write
    if not pd_frame.empty:
        write_batch(output_path, batch_index, pd_frame)

    print(output_path, "Done")


def tsv2df(spark, path):
    schema = StructType(
        [
            StructField('label', LongType(), True),
            StructField('text_a', StringType(), True)
        ]
    )

    df = spark.read \
        .options(header=True, delimiter='\t') \
        .schema(schema) \
        .csv(path.as_posix())

    df = df.select(
        col("label").alias("label_id"),
        hexstr_to_bytearray("text_a").alias("packet")
    )
    return df


def match_tsv_pcap(tsv_df, df):
    tsv_df = tsv_df.filter(col("label_id").isin([0]))
    tsv_df = tsv_df.withColumn("template", bytearray_to_hexstr("packet"))
    tsv_df.show()

    df = df.withColumn("packet_str", bytearray_to_hexstr("packet"))
    df.show()

    t1 = time.perf_counter()

    # regexes = tsv_df.select("template").rdd.flatMap(lambda x: x).collect()
    # for regex in tqdm.tqdm(regexes):
    #     filtered_df = df.filter(col("packet_str").contains(regex))
    #     if not filtered_df.isEmpty():
    #         filtered_df.show()
    regex = "fccfcf70700101cfc"
    filtered_df = df.filter(col("packet_str").contains(regex))
    filtered_df.show()
    print(filtered_df.count())
    t2 = time.perf_counter()
    print(t2 - t1)


def get_sorted_paths(path):
    sub_dirs = [entry for entry in path.iterdir() if entry.is_dir()]
    sub_dirs.sort(key=lambda x: x.name.lower())
    return sub_dirs


def clean_dirs(*dirs):
    for cur_dir in dirs:
        if cur_dir.exists():
            shutil.rmtree(cur_dir)
        cur_dir.mkdir(parents=True)


def print_spark_df_count(df, cols=None):
    if cols is None:
        cols = ['label_id']
    df.groupBy(cols).count().sort(cols, ascending=True).show()


def aggregator(spark, source):
    # read data
    schema = StructType(
        [
            StructField("label_id", LongType(), True),
            StructField("file_path", StringType(), True),
            StructField("packet", BinaryType(), True),
            StructField("length", LongType(), True),
        ]
    )

    df = spark.read.schema(schema).parquet(
        f"{source.absolute().as_posix()}/*.parquet.gzip"
    )
    return df


@click.option("-n", "--njob", default=-1, help="num of executors", type=int)
def main(njob):
    tsv_path = Path('/root/PycharmProjects/UER_py/datasets/VPN-app/packet/train_dataset.tsv')
    pcap_root = Path("/root/PycharmProjects/DATA/ISCX-VPN-NonVPN-App")
    transformed_pcap = Path("/root/PycharmProjects/DATA/ISCX-VPN-NonVPN-App.transformed")
    target_dir_path = Path("./data")

    # clean_dirs(transformed_pcap, target_dir_path)

    sorted_path = get_sorted_paths(pcap_root)
    id2path = {label_id: label_path for label_id, label_path in enumerate(sorted_path)}
    id2label = {index: path.name for index, path in id2path.items()}
    label2id = {label: index for index, label in id2label.items()}

    # with (target_dir_path / "id2label.json").open("w") as f:
    #     json.dump(id2label, f, indent=4)

    # Parallel(n_jobs=njob)(
    #     delayed(transform_single_pcap)(
    #         pcap_root, pcap_path, transformed_pcap, label_id
    #     )
    #     for label_id, label_path in id2path.items()
    #     for pcap_path in label_path.rglob("*.pcap")
    # )

    spark = init_spark()

    tsv_df = tsv2df(spark, tsv_path)
    print_spark_df_count(tsv_df)

    df = aggregator(spark, transformed_pcap / "*/*")
    print_spark_df_count(df)

    match_tsv_pcap(tsv_df, df)


@udf(returnType=StringType())
def bytearray_to_hexstr(column):
    return binascii.hexlify(column).decode('utf-8')


@udf(returnType=BinaryType())
def hexstr_to_bytearray(column):
    return bytearray.fromhex(column)


if __name__ == "__main__":
    main(-1)
    # spark = init_spark()
    # df1 = spark.createDataFrame([
    #     ("bob1", "abc."), ("bob2", "abc")], ["col1", "col2"])
    # df2 = spark.createDataFrame([
    #     ("tom1", "dd74s6dcAFabbbcd"), ("tom2", "abc"), ("tom3", "abcddd74s6d")], ["col3", "col4"])
    # regexes = df1.select("col2").rdd.flatMap(lambda x: x).collect()
    # for regex in regexes:
    #     df2.filter(col("col4").contains(regex)).show()
