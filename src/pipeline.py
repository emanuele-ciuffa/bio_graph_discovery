import os
import sys

from src.data_prep import data_preparation
from src.graph_imp import graph_import
from src.graph_analysis import graph_analysis
from src.link_pred import link_prediction
from src.utils.logging_handler import Logging_handler
from src.utils import spark_initializer

from src.utils.config_handler import Config_handler

import traceback
# Forces Spark workers to use your current Conda Python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


try:
    config_handler = Config_handler("config-common.yml")

    logger = Logging_handler(config_handler).get_logger(module_name="main")

    logger.info("Pipeline is starting...")


    # Read selected organism (species) from config file
    organism = config_handler.read_property("organism")
    logger.info(f"Selected organism: {organism}")

    # Read nodes dataset path from config file
    nodes_path = config_handler.read_property("dataset.graph.nodes.path")
    logger.info(f"Nodes path: {nodes_path}")

    # Read edges dataset path from config file
    edges_path = config_handler.read_property("dataset.graph.edges.path")
    logger.info(f"Edges path: {edges_path}")

    # Read log path from config file
    log_path = config_handler.read_property("log.path")
    logger.info(f"Log path: {log_path}")

    # Read log level from config file
    log_level = config_handler.read_property("log.level")
    logger.info(f"Log level: {log_level}")

    # Instatiate spark session
    spark = spark_initializer.get_spark_session(logger)

    # Read boolean value from config value to determine if data preparation step is enabled
    is_data_prep_enabled = config_handler.read_property("pipeline.enable.data_preparation")

    # Read boolean value from config value to determine if graph import step is enabled
    is_graph_import_enabled = config_handler.read_property("pipeline.enable.graph_import")

    # Read boolean value from config value to determine if graph analysis step is enabled
    is_graph_analysis_enabled = config_handler.read_property("pipeline.enable.graph_analysis")

    # Read boolean value from config value to determine if link link_pred step is enabled
    is_link_prediction_enabled = config_handler.read_property("pipeline.enable.link_prediction")

    logger.info(f"Data preparation step enabled: {is_data_prep_enabled}")
    logger.info(f"Graph import step enabled: {is_graph_import_enabled}")
    logger.info(f"Graph analysis step enabled: {is_graph_analysis_enabled}")
    logger.info(f"Link prediction step enabled: {is_link_prediction_enabled}")


    ##################################
    ###### DATA PREPARATION ##########
    ##################################

    # Check if data preparation is enabled
    if is_data_prep_enabled:

        logger.info("Data preparation is starting...")

        # Execute data preparation
        data_preparation.execute(spark=spark,
                                 common_config_handler=config_handler,
                                 nodes_path=nodes_path,
                                 edges_path=edges_path,
                                 organism=organism)

        logger.info(f"Data preparation is finished.")
    else:
        logger.warning("Data preparation has not been enabled (config parameter is set to False).")

    ##################################
    ###### GRAPH DB IMPORT ###########
    ##################################

    # Check if graph import is enabled
    if is_graph_import_enabled:

        logger.info("Graph import is starting...")

        # Create graph DB and import data
        graph_import.execute(common_config_handler=config_handler,
                             nodes_path=nodes_path,
                             edges_path=edges_path,
                             organism=organism)

        logger.info(f"Graph import is finished.")
    else:
        logger.warning("Graph import has not been enabled (config parameter is set to False).")

    ##################################
    ###### GRAPH_ANALYSIS ############
    ##################################
    if is_graph_analysis_enabled:

        # Analyse Graph (organism will be selected from neo4j logs saved during the import)
        graph_analysis.execute(spark=spark,
                               common_config_handler=config_handler,
                               selected_organism=organism)

        logger.info(f"Graph analysis is finished")
    else:
        logger.warning("Graph analysis has not been enabled (config parameter is set to False).")

    # Stop spark session: next analysis won't use spark
    spark.stop()

    logger.info("Spark has been stopped")

    ###################################
    ###### LINK_PREDICTION ############
    ###################################
    if is_link_prediction_enabled:

        # Perform Link Preidction
        link_prediction.execute(common_config_handler=config_handler,
                                selected_organism=organism)
        logger.info(f"Link prediction is finished")
    else:
        logger.warning("Link Prediction has not been enabled (config parameter is set to False).")

    logger.info("Pipeline has terminated successfully.")

except Exception as e:
    logger.error(f"{e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    logger.error(f"Pipeline execution failed.")