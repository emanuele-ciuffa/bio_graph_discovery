
from pyspark.sql import SparkSession

from src.utils.config_handler import Config_handler

import os


def get_spark_session(logger):
    """
    This function initializes the Spark Session.

    :param logger: configured logger.
    :return: spark session.
    """
    try:
        config_handler = Config_handler("config-spark.yml")

        logger.info("Spark configuration is starting...")

        # Spark configuration parameters
        spark_master = config_handler.read_property("spark.master")
        spark_app_name = config_handler.read_property("spark.app_name")
        spark_drive_mermory = config_handler.read_property("spark.driver.memory")
        spark_executor_memory = config_handler.read_property("spark.executor.memory")
        spark_executor_memory_overhead = config_handler.read_property("spark.executor.memoryOverhead")
        spark_python_worker_memory = config_handler.read_property("spark.python.worker.memory")
        spark_driver_host = config_handler.read_property("spark.driver.host")
        spark_shuffle_partition = config_handler.read_property("spark.shuffle.partitions")
        spark_ui_host = config_handler.read_property("spark.ui.host")
        spark_ui_port = config_handler.read_property("spark.ui.port")
        spark_broadcast_timeout = config_handler.read_property("spark.broadcast.timeout")
        spark_auto_broadcast_join_threshold = config_handler.read_property("spark.broadcast.auto.join_threshold")

        graphframe_jar = config_handler.read_property("jars.graphframe")
        neo4j_jar = config_handler.read_property("jars.neo4j")

        checkpoint_dir = config_handler.read_property("spark_checkpoint.path")

        logger.info(f"Configured spark.master: {spark_master}")
        logger.info(f"Configured spark.app_name: {spark_app_name}")
        logger.info(f"Configured spark.driver.memory: {spark_drive_mermory}")
        logger.info(f"Configured spark.executor.memory: {spark_executor_memory}")
        logger.info(f"Configured spark.executor.memoryOverhead: {spark_executor_memory_overhead}")
        logger.info(f"Configured spark.python.worker.memory: {spark_python_worker_memory}")
        logger.info(f"Configured spark.driver.host: {spark_driver_host}")
        logger.info(f"Configured spark.sql.shuffle.partitions: {spark_shuffle_partition}")
        logger.info(f"Configured spark.ui.host: {spark_ui_host}")
        logger.info(f"Configured spark.ui.port: {spark_ui_port}")
        logger.info(f"Configured spark.sql.broadcastTimeout: {spark_broadcast_timeout}")
        logger.info(f"Configured spark.sql.autoBroadcastJoinThreshold: {spark_auto_broadcast_join_threshold}")
        logger.info(f"Graphframe jar: {graphframe_jar}")
        logger.info(f"Neo4j driver jar: {neo4j_jar}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")


        # Initialize Spark Session
        spark = SparkSession.builder \
            .master(spark_master) \
            .appName(spark_app_name) \
            .config("spark.driver.memory", spark_drive_mermory) \
            .config("spark.executor.memory", spark_executor_memory) \
            .config("spark.executor.memoryOverhead", spark_executor_memory_overhead) \
            .config("spark.python.worker.memory", spark_python_worker_memory) \
            .config("spark.driver.host", spark_driver_host) \
            .config("spark.sql.shuffle.partitions", spark_shuffle_partition) \
            .config("spark.ui.host", spark_ui_host) \
            .config("spark.ui.port", spark_ui_port) \
            .config("spark.sql.broadcastTimeout", "600") \
            .config("spark.sql.autoBroadcastJoinThreshold", spark_auto_broadcast_join_threshold) \
            .config("spark.jars", f"{graphframe_jar},{neo4j_jar}") \
            .config("spark.driver.extraClassPath", f"{graphframe_jar};{neo4j_jar}") \
            .config("spark.executor.extraClassPath", f"{graphframe_jar};{neo4j_jar}") \
            .getOrCreate()

        # WARNING: Make sure that there's no conflict (different graphframes lib version into site-package)
        ###### C:\Users\xxx\AppData\Local\anaconda3\envs\spark_tres\Lib\site-packages\pyspark\jars

        # 1. Set path for spark checkpoint (ensure that the folder exists)
        spark.sparkContext.setCheckpointDir(checkpoint_dir)

        # Enable Arrow-based columnar data transfers (useful to transform pyspark dataframe to pandas dataframe)
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

        logger.info("Spark configuration is complete")

        logger.info(f"Spark Version: {spark.version}")

        scala_version = spark.sparkContext._gateway.jvm.scala.util.Properties.versionString()
        logger.info(f"Spark environment is using the following Scala version: {scala_version}")

        logger.info(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")

        return spark

    except Exception as e:
        logger.error(e)
        raise Exception("Error while initializing SparkSession")