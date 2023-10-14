import os
import sys

import pyspark
from pyspark.sql import SparkSession


def run_spark():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = SparkSession.builder.appName('sparkdf').getOrCreate()

    # list  of student  data
    data = [["1", "sravan", "IT", 45000],
            ["2", "ojaswi", "CS", 85000],
            ["3", "rohith", "CS", 41000],
            ["4", "sridevi", "IT", 56000],
            ["5", "bobby", "ECE", 45000],
            ["6", "gayatri", "ECE", 49000],
            ["7", "gnanesh", "CS", 45000],
            ["8", "bhanu", "Mech", 21000]
            ]

    # specify column names
    columns = ['ID', 'NAME', 'DEPT', 'FEE']

    # creating a dataframe from the lists of data
    dataframe = spark.createDataFrame(data, columns)

    # display
    dataframe.show()

    df = dataframe.groupBy("DEPT").count().agg({"count": "min"}).first()[0]


if __name__ == "__main__":
    run_spark()
