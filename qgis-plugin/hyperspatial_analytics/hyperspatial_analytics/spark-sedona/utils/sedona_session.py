import os
from pyspark.sql import SparkSession

# BUG FIX: Prevent macOS local XML parsing errors that crash JVM
os.environ["HADOOP_CONF_DIR"] = ""

def get_sedona_session(app_name: str = "HyperSpatialJob",
                       local: bool = True):
    try:
        from sedona.spark import SedonaContext
    except ImportError:
        raise ImportError("apache-sedona is not installed or configured correctly.")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.registrator",
                "org.apache.sedona.core.serde.SedonaKryoRegistrator")
        .config("spark.jars.packages",
                "org.apache.sedona:sedona-spark-shaded-3.4_2.12:1.5.1,"
                "org.datasyslab:geotools-wrapper:1.5.1-28.2")
        .config("spark.jars.repositories",
                "https://artifacts.unidata.ucar.edu/repository/unidata-all")
    )
    if local:
        builder = builder.master("local[*]")
    return SedonaContext.create(builder.getOrCreate())
