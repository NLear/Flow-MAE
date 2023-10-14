import itertools
from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, Window
from pyspark.sql import (
    SparkSession
)
from pyspark.sql.functions import (
    lit,
    rand,
    row_number,
    least,
    col, count, when, dense_rank, greatest, first, lower, create_map
)
from pyspark.sql.types import (
    IntegerType
)

from preprocess.factory import BaseAggregator
from preprocess.pyspark.datasets_adaptor import get_label_mapping
from preprocess.pyspark.spark_adapter import AdaptorSpark

DATASET_PARQUET = "dataset.parquet"
TRAIN_PARQUET = "train.parquet"
TEST_PARQUET = "test.parquet"


def save_parquet(df: DataFrame, path: Union[str, Path]):
    """
    Save a PySpark DataFrame as a Parquet file.

    Args:
        df: The PySpark DataFrame to be saved.
        path: The file path as a string or a pathlib.Path object.
    """
    # Ensure path is a string
    if isinstance(path, Path):
        path = str(path)
    output_path = Path(path).absolute().as_uri()
    df.write.mode("overwrite").parquet(output_path)
    print_df_label_distribution(df)


def read_parquet_file(path: Union[str, Path]) -> DataFrame:
    """
    Read a Parquet file using PySpark and return a DataFrame.

    Args:
        path: The file path as a string or a pathlib.Path object.

    Returns:
        A PySpark DataFrame containing the data from the Parquet file.
    """
    # Ensure path is a string
    if isinstance(path, Path):
        path = str(path)

    # Create a SparkSession
    spark = SparkSession.builder \
        .appName("Read Parquet File") \
        .getOrCreate()

    # Read the Parquet file into a DataFrame
    df = spark.read.parquet(path)

    return df


def print_df_label_distribution(input_data: Union[str, Path, DataFrame]) -> None:
    """
    Print the label distribution of the given dataset.

    Args:
        input_data: The path to the Parquet file or a DataFrame containing the dataset.
    """
    if isinstance(input_data, (str, Path)):
        # Create a SparkSession
        spark = (
            SparkSession.builder
            .appName("df_label_distribution")
            .getOrCreate()
        )
        input_data = spark.read.parquet(Path(input_data).absolute().as_uri())

    print(
        input_data.groupby("labels").count()
        .withColumn("labels_lower", lower(col("labels")))
        .orderBy("labels_lower", ascending=[True])
        .drop("labels_lower")
        .toPandas()
    )


def split_train_test(df, test_size, label_col="labels"):
    """
    Split the input DataFrame into train and test sets using stratified sampling.

    Args:
        df: Input DataFrame.
        test_size: Fraction of the dataset to be used as test set.
        label_col: Name of the label column.

    Returns:
        train_df: DataFrame containing the training data.
        test_df: DataFrame containing the test data.
    """
    # Add a random column for sorting within the window
    df = df.withColumn("rand", rand())
    # Create a window partitioned by label_col and ordered by the random column
    window = Window.partitionBy(label_col).orderBy(col("rand"))
    # Assign row numbers to each row within the window
    df = df.withColumn("row_num", row_number().over(window))

    # Calculate the test set size for each label
    label_counts = df.groupBy(label_col).count().withColumnRenamed("count", "label_count")
    df = df.join(label_counts, on=label_col)

    # Assign samples to test or train set based on row number and test set size
    test_df = df.filter(col("row_num") <= col("label_count") * test_size)
    train_df = df.filter(col("row_num") > col("label_count") * test_size)

    # Remove temporary columns
    test_df = test_df.drop("row_num", "rand", "label_count")
    train_df = train_df.drop("row_num", "rand", "label_count")

    return train_df, test_df


def save_train_test(train_df, test_df, target: Union[str, Path]) -> None:
    """
    Process the dataset and save the train and test DataFrames as Parquet files.

    Args:
        target: Directory to save the Parquet files.
        train_df: Train data
        test_df: Test data
    """
    save_parquet(train_df, target / TRAIN_PARQUET)
    save_parquet(test_df, target / TEST_PARQUET)


def under_sample(df, sample_per_label, label_col="labels"):
    """
    Undersample the input DataFrame to balance the class distribution.

    Args:
        df: Input DataFrame.
        sample_per_label: Number of samples to be kept for each label.
        label_col: Name of the label column.

    Returns:
        DataFrame with balanced class distribution.
    """
    df = df.withColumn("rand", rand(seed=9876))
    window = Window.partitionBy(col(label_col)).orderBy(col("rand"))

    return (
        df.select(col("*"), row_number().over(window).alias("row_number"))
        .where(col("row_number") <= sample_per_label)
        .drop("row_number", "rand")
    )


def stratified_sample_train_test(df, sample_per_label, test_size, label_col="labels"):
    """
    Split the input DataFrame into train and test sets using stratified sampling.

    Args:
        df: Input DataFrame.
        test_size: Fraction of the dataset to be used as test set.
        sample_per_label: Number of samples to be kept for each label.
        label_col: Name of the label column.

    Returns:
        train_df: DataFrame containing the training data.
        test_df: DataFrame containing the test data.
    """
    # Add a random column for sorting within the window
    df = df.withColumn("rand", rand())
    # Create a window partitioned by label_col and ordered by the random column
    window = Window.partitionBy(label_col).orderBy(col("rand"))
    # Assign row numbers to each row within the window
    df = df.withColumn("row_num", row_number().over(window))

    # Calculate the test set size for each label
    label_counts = df.groupBy(label_col).count().withColumnRenamed("count", "label_count")
    df = df.join(label_counts, on=label_col)

    # Calculate the sample size for each label
    df = df.withColumn("sample_size", least(col("label_count"), lit(sample_per_label)))

    # Calculate the test set size for each label
    df = df.withColumn("test_size", (col("sample_size") * test_size).cast(IntegerType()))

    # Assign samples to test or train set based on row number, test set size, and sample size
    test_df = df.filter(col("row_num") <= col("test_size"))
    train_df = df.filter((col("row_num") > col("test_size")) & (col("row_num") <= col("sample_size")))

    # Remove temporary columns
    test_df = test_df.drop("row_num", "rand", "label_count", "sample_size", "test_size")
    train_df = train_df.drop("row_num", "rand", "label_count", "sample_size", "test_size")

    return train_df, test_df


def filter_df_by_feature_len(
        df: DataFrame,
        min_feature_len: int = 1024,
        threshold: int = 200,
        small_feature_len: int = 120,
) -> DataFrame:
    """
    过滤出feature_len大于一定值的行，如果分组中满足feature_len条件的数据较少，则使用较小的feature_len

    :param df: 输入的DataFrame
    :param min_feature_len: 默认最小的feature_len
    :param threshold: 较少的数据阈值
    :param small_feature_len: 较小的feature_len
    :return: 过滤后的DataFrame
    """
    # 根据labels进行分组并计算每组中满足feature_len条件的数量
    window_spec = Window.partitionBy("labels")
    df_with_count = df.withColumn(
        "count",
        count(when(col("feature_len") >= min_feature_len, True)).over(window_spec),
    )

    # 过滤数据，如果分组中满足feature_len条件的数据较少，则使用较小的feature_len
    df_filtered = df_with_count.filter(
        (col("count") >= threshold) & (col("feature_len") >= min_feature_len)
        | (col("count") < threshold) & (col("feature_len") >= small_feature_len)
    )

    # 删除"count"列
    df_filtered = df_filtered.drop("count")

    return df_filtered


def filter_df_by_feature_len_list(
        df: DataFrame,
        min_feature_len: int = 1024,
        threshold: int = 500,
        small_feature_len_list=None,
) -> DataFrame:
    """
    过滤出feature_len大于一定值的行，如果分组中满足feature_len条件的数据较少，则尝试不同的small_feature_len

    :param df: 输入的DataFrame
    :param min_feature_len: 默认最小的feature_len
    :param threshold: 较少的数据阈值
    :param small_feature_len_list: 较小的feature_len列表
    :return: 过滤后的DataFrame
    """
    # 根据labels进行分组并计算每组中满足feature_len条件的数量
    if small_feature_len_list is None:
        small_feature_len_list = [700, 600, 500, 400, 140, 50]
    window_spec = Window.partitionBy("labels")
    df_with_count = df.withColumn(
        "count",
        count(when(col("feature_len") >= min_feature_len, True)).over(window_spec),
    )

    # 获取适用于每个分组的最大feature_len
    max_feature_len_expr = greatest(
        *[when(col("count") >= threshold, feature_len) for feature_len in [min_feature_len] + small_feature_len_list])
    group_max_feature_len_df = df_with_count.groupBy("labels").agg(
        first(max_feature_len_expr).alias("max_feature_len")
    )
    # 将max_feature_len添加到原始DataFrame
    df_with_max_feature_len = df.join(group_max_feature_len_df, on="labels", how="inner")

    # 过滤数据，保留满足最大feature_len条件的行
    df_filtered = df_with_max_feature_len.filter(col("feature_len") >= col("max_feature_len"))

    # 删除"count"和"max_feature_len"列
    df_filtered = df_filtered.drop("count", "max_feature_len")

    return df_filtered


def print_diff_labels(origin, filtered_df):
    # 计算原始和过滤后的DataFrame中每个标签的记录数
    origin_counts = origin.groupBy("labels").count().withColumnRenamed("count", "original_count")
    filtered_counts = filtered_df.groupBy("labels").count().withColumnRenamed("count", "filtered_count")

    # 找到被删除的标签
    removed_labels = origin_counts.join(filtered_counts, on="labels", how="left_anti")

    # 输出被删除的标签及其原本的数目
    if removed_labels.count() > 0:
        print("Removed labels:")
        removed_labels.select("labels", "original_count").show()
    else:
        print("No labels were removed")


def reorganize_labels(df: DataFrame, label_column: str = "labels") -> DataFrame:
    """
    将不连续的标签重新组织为连续标签

    :param df: 输入的DataFrame
    :param label_column: 包含标签的列的名称，默认为'labels'
    :return: 带有重新组织的连续标签的DataFrame
    """
    # 定义窗口函数，按标签升序排列
    window_spec = Window.partitionBy().orderBy(col(label_column))

    # 使用dense_rank函数为标签分配连续的整数
    df_reorganized = df.withColumn("new_labels", dense_rank().over(window_spec) - 1)

    # 将原始标签列删除，将新标签列重命名为原始标签列
    df_reorganized = df_reorganized.drop(label_column).withColumnRenamed("new_labels", label_column)

    return df_reorganized


def get_label_mapping_from_dataframe(df):
    all_labels = df.select("labels").distinct().rdd.flatMap(lambda x: x).collect()

    label_to_id, all_labels = get_label_mapping(set(all_labels))

    return label_to_id, all_labels


def add_label_id_column(df, label_to_id):
    label_id_column = "labels"
    label_name_column = "label_name"

    # Create a mapping expression using label_to_id dictionary
    mapping_expr = create_map([lit(x) for x in itertools.chain(*label_to_id.items())])

    # Add a new "labels" column with label IDs using the mapping expression
    df = df.withColumn(label_name_column, mapping_expr[col(label_id_column)])

    return df


class PySparkAggregator(BaseAggregator):
    schema = AdaptorSpark.schema

    def aggregator(self, source, target, min_feature_len: int = None, sample_per_label: int = None,
                   test_ratio: float = None) -> None:
        """
        Aggregate the preprocessed data, split it into train and test sets, and save them as Parquet files.

        Args:
            source: Directory containing the preprocessed data.
            target: Directory to save the Parquet files.
            min_feature_len: Minimum feature length to be kept.
            sample_per_label: Number of samples to be kept for each label.
            test_ratio: 0.1
        """
        spark = (
            SparkSession.builder
            .appName("aggregator")
            .master("local[*]")
            .config("spark.driver.memory", "16g")
            .config("spark.ui.showConsoleProgress", "true")  # Enable console progress reporting
            .getOrCreate()
        )

        # Read all the .parquet files recursively from the subdirectories
        df = spark.read.schema(self.schema).parquet(
            f"{Path(source).absolute().as_uri()}/*.transformed.parquet/*.parquet"
        ).filter((col("labels").isNotNull()) & (col("feature_len") > 0))

        # Filter out rows with feature_len less than min_feature_len
        if min_feature_len:
            origin = df
            df = filter_df_by_feature_len(df, min_feature_len)
            print_diff_labels(origin, df)

        # Under-sample the data
        if sample_per_label:
            df = under_sample(df, sample_per_label)

        # Reorganize labels
        # df = reorganize_labels(df)

        # Add label IDs
        # label_to_id, all_labels = get_label_mapping_from_dataframe(df)
        # df = add_label_id_column(df, label_to_id)

        # split train and test sets
        if test_ratio:
            train_df, test_df = split_train_test(df, test_ratio)
            save_train_test(train_df, test_df, target)
        else:
            save_parquet(df, target / DATASET_PARQUET)
