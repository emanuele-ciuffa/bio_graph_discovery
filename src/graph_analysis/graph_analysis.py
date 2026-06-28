import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import traceback

from src.utils.config_handler import Config_handler
from src.utils.logging_handler import Logging_handler
from src.utils import file_utils
from src.utils.neo4j_handler import Neo4j_handler

from graphframes import GraphFrame
from src.graph_analysis import graph_algorithms


def execute(spark, common_config_handler, selected_organism):
    """
    Execute Graph Analysis:
    - Community Detection;
    - Degree Centrality;
    - Connected Components (including largest component);
    - Shortest Path.
    :param spark: spark session.
    :param common_config_handler: common config handler.
    :param selected_organism: selected organism from config file.
    """

    try:
        # Instatiate config handler for retrieving neo4j parameters
        config_handler_neo4j = Config_handler("config-neo4j.yml")

        # Instatiate config handler for retrieving graph analysis parameters
        config_handler_graph_analysis = Config_handler("config-graph_analysis.yml")

        # Instatiate logger
        logger = Logging_handler(common_config_handler).get_logger(module_name="graph_analysis")

        # Reading references to connect to neo4j DB
        uri = config_handler_neo4j.read_property("neo4j.uri")
        user = config_handler_neo4j.read_property("neo4j.user")
        password = config_handler_neo4j.read_property("neo4j.password")

        # Instantiate neo4j_handler
        neo4j_handler = Neo4j_handler(uri, user, password, logger)

        # Get last neo4j log
        last_neo4j_log = neo4j_handler.get_neo4j_last_log(logger=logger)

        # Read organism from last log
        organism = last_neo4j_log.organism

        # Validate that the selected organism matches the one in Neo4j
        if selected_organism != organism:
            logger.error(f"Selected organism is '{selected_organism}' which is not compatible with the current neo4j database.")
            raise ValueError(
                f"Organism mismatch: selected organism '{selected_organism}' "
                f"does not match the organism found in Neo4j '{organism}'."
            )

        logger.info(f"Selected organism (species): '{organism}'")

        # 3. Neo4j Connection Credentials
        common_options = {
            "url": f"{uri}",
            "authentication.basic.username": f"{user}",
            "authentication.basic.password": f"{password}",
        }

        ################ DATA LOADING ##################################################################################

        # Load Vertices (Nodes)
        v_df = spark.read.format("org.neo4j.spark.DataSource") \
            .options(**common_options) \
            .option("query",
                    """
                        MATCH (n) 
                        WHERE labels(n)[0] IN ['chemical', 'protein', 'reaction'] 
                        RETURN n.id AS id, labels(n)[0] AS type, n.name AS name
                    """) \
            .load()

        # Load Edges (Relationships)
        e_df = spark.read.format("org.neo4j.spark.DataSource") \
            .options(**common_options) \
            .option("query",
                    """
                    MATCH (s)-[r]->(d)
                        WHERE labels(s)[0] IN ['chemical', 'protein', 'reaction']
                          AND labels(d)[0] IN ['chemical', 'protein', 'reaction']
                        RETURN s.id AS src,
                               d.id AS dst,
                               r.relationship AS relationship,
                               r.type AS type,
                               r.reaction_id AS reaction_id
                    """) \
            .load()

        '''
        .option("query",
                    """
                        MATCH (s)-[r]->(d) 
                        RETURN s.id AS src, d.id AS dst, type(r) AS relationship
                    """) \
        '''

        # Cache them for performance
        v_df.cache()
        e_df.cache()

        # Ensure IDs are strings (GraphFrames requirement)
        v_gf = v_df.withColumn("id", F.trim(F.col("id")).cast("string"))
        e_gf = e_df.withColumn("src", F.trim(F.col("src")).cast("string")) \
            .withColumn("dst", F.trim(F.col("dst")).cast("string"))

        # Create the Graph
        g = GraphFrame(v_gf, e_gf)


        ########## COMMUNITY DETECTION ######################################################################################

        # Check if Community Detection analysis is enabled from config
        community_detection_enabled = config_handler_graph_analysis.read_property("community_detection.enable")

        if community_detection_enabled:

            logger.info("Starting to process Community Detection...")

            exclude_ontological_relationships = config_handler_graph_analysis.read_property(
                "community_detection.exclude_ontological_relationships")

            if exclude_ontological_relationships:
                logger.info(f"Ontological relationships will be excluded from the community detection process.")
            else:
                logger.info("Ontological relationships will be evaluated in the community detection process.")

            max_iter = config_handler_graph_analysis.read_property(
                "community_detection.label_propagation_algorithm.max_iter")

            logger.info(f"Max number of iteration for Label Propagation Algorithm (Community Detection): {max_iter}")

            # Perform Community Detection by Label Propagation Algorithm
            communities_df = graph_algorithms.run_community_detection(spark=spark,
                                                                      logger=logger,
                                                                      common_options=common_options,
                                                                      max_iter=max_iter,
                                                                      exclude_ontological_edges=exclude_ontological_relationships)

            logger.info("Community Detection completed")

            # Read parameter for Communities path
            communities_path = file_utils.get_name_with_organism(
                file_path=config_handler_graph_analysis.read_property("community_detection.path"),
                organism=organism)

            # Save Community Detection reulst
            communities_pandas_df = communities_df.orderBy(F.desc("label")).toPandas()
            communities_pandas_df.to_csv(communities_path)
            logger.info(f"Community Detection results have been saved to the following path: {communities_path}")

        else:
            logger.warning(
                "Community Detection will not be computed. Set 'community_detection.enable' to True in the related config file")



        ########## DEGREE CENTRALITY (Promiscuity) #####################################################################

        # Check if Largest Component analysis is enabled from config
        degree_centrality_enabled = config_handler_graph_analysis.read_property("degree_centrality.enable")

        if degree_centrality_enabled:
            logger.info("Degree Centrality will be computed")
            degree_df = graph_algorithms.degree_centrality(logger=logger,
                                                           g=g)

            # Read parameter for Graph analysis degree path renaming file name with organism name
            degree_path = file_utils.get_name_with_organism(
                file_path=config_handler_graph_analysis.read_property("degree_centrality.path"),
                organism=organism)

            # Save Degree Centrality to csv
            degree_pandas_df = degree_df.toPandas()
            degree_pandas_df.to_csv(degree_path)
            logger.info(f"Degree Centrality results have been saved to the following path: {degree_path}")

        else:
            logger.warning("Degree Centrality will not be computed. Set 'degree_centrality.enable' to True in the related config file")


        ########## CONNECTED COMPONENTS ######################################################################################

        # Initialize variables for the report
        largest_component_id = None
        largest_component_count = None
        components_count = None

        # Check if Largest Component analysis is enabled from config
        connected_components_enabled = config_handler_graph_analysis.read_property("connected_components.enable")

        if connected_components_enabled:

            logger.info("Connected Components analysis")

            # Read Connected Components (output) path
            connected_components_path = file_utils.get_name_with_organism(
                file_path=config_handler_graph_analysis.read_property("connected_components.path"),
                organism=organism)

            # Compute Connected Components (including Largest Component)
            connected_components_tuple = graph_algorithms.connected_components(spark=spark,
                                                                               logger=logger,
                                                                               larget_component_path=connected_components_path,
                                                                               g=g)

            # Decompose the tuple (largest_component_id, largest_component_count, components_count)
            largest_component_id = connected_components_tuple[0] # largest component id (connected component with the highest number of vertices)
            largest_component_count = connected_components_tuple[1] # number of vertices of the largest component
            components_count = connected_components_tuple[2] # number of connected components

            logger.info(f"Largest Component ID is: {largest_component_id}")

            logger.info(f"Largest Component count is: {largest_component_count}")

            logger.info(f"Number of Connected Components: {components_count}")

            if community_detection_enabled:
                # Join Connected Components and Communities
                connected_components_df = spark.read.csv(connected_components_path, header=True, inferSchema=True)

                # Calculate community count for statistics
                window = Window.partitionBy("community")
                communities_with_count_df = communities_df.withColumn("community_count", F.count("community").over(window))

                component_communities_df = (connected_components_df.join(communities_with_count_df,
                                                                         on="id", how="left")
                                            .select(connected_components_df["id"],
                                                    connected_components_df["type"],
                                                    connected_components_df["name"],
                                                    connected_components_df["component_id"],
                                                    connected_components_df["component_count"],
                                                    communities_with_count_df["community"],
                                                    communities_with_count_df["community_count"])
                                            )
                #component_communities_df.show(100, truncate=False)

                logger.info("Communities output will be overwritten with connected component association")

                # Transform the pyspark dataframe to pandas dataframe
                component_communities_pandas = component_communities_df.toPandas()

                # Cast count fields to int
                component_communities_pandas["community"] = (component_communities_pandas["community"]
                                                             .astype("Int64"))
                component_communities_pandas["component_id"] = (component_communities_pandas["component_id"]
                                                                .astype("Int64"))
                component_communities_pandas["component_count"] = (component_communities_pandas["component_count"]
                                                                   .astype("Int64"))
                component_communities_pandas["community_count"] = (component_communities_pandas["community_count"]
                                                                   .astype("Int64"))

                # Save component_communities_pandas to csv
                component_communities_pandas.to_csv(communities_path)

                logger.info(f"Community Detection results have been saved to the following path: {communities_path}")

        else:
            logger.warning("Connected Components will not be computed. Set 'degree_centrality.enable' to True in the related config file")

        ########## SHORTEST PATH #######################################################################################

        # Check if Shortest Path analysis is enabled from config
        shortest_path_enabled = config_handler_graph_analysis.read_property("shortest_path.enable")

        if shortest_path_enabled:
            # Read target_id from config
            target_name = config_handler_graph_analysis.read_property("shortest_path.target_name")

            logger.info(f"Selected Target name for the Shortest Path: '{target_name}'")

            logger.info("Retrieve target_id")

            # Retrieve node id based on target name (node name)
            target_id = neo4j_handler.get_node_id_by_name(target_name)

            logger.info(f"Target id is {target_id}")

            logger.info(f"Shortest Path will be computed")

            max_iter_sp = config_handler_graph_analysis.read_property("shortest_path.max_iterations")

            # Execute the Shortest Path algorithm
            sp_result_df = graph_algorithms.shortest_path(logger=logger,
                                                          neo4j_handler=neo4j_handler,
                                                          g=g,
                                                          target_id=target_id,
                                                          max_iterations=max_iter_sp)

            # Read parameter for Graph analysis degree path renaming file name with organism name
            shortest_path_path = file_utils.get_name_with_organism(
                file_path=config_handler_graph_analysis.read_property("shortest_path.path"),
                organism=organism).replace("VERTEX", target_id)

            # Save Shortest Path to csv
            shortest_path_pandas_df = sp_result_df.toPandas()
            shortest_path_pandas_df.to_csv(shortest_path_path, sep=";")
            logger.info(
                f"Shortest Path results (for vertex '{target_id}') have been saved to the following path: {shortest_path_path}")

        else:
            logger.warning(
                "Shortest Path will not be computed. Set 'shortest_path.enable' to True in the related config file")


        ########## NODE AND EDGE COUNTING ##############################################################################
        node_count = v_df.count()
        edge_count = e_df.count()

        logger.info(f"Node Count: {node_count}")
        logger.info(f"Edge Count: {edge_count}")


        ########## SAVE REPORT RESULTS ##############################################################################

        # Read parameter for Graph analysis report path renaming file name with organism name
        report_path = file_utils.get_name_with_organism(
            file_path=config_handler_graph_analysis.read_property("report.path"),
            organism=organism)

        logger.info(f"Graph analysis report path: {report_path}")

        # Create the Pandas DataFrame
        df_report = pd.DataFrame([{
            "organism": organism,
            "node_count": node_count,
            "edge_count": edge_count,
            "largest_component_id": largest_component_id,
            "largest_component_count": largest_component_count,
            "connected_component_count": components_count
        }])

        logger.info("Saving results...")

        # Save analysis results
        df_report.to_csv(report_path)

        logger.info(f"Graph analysis report has been saved to the following path: {report_path}")

    except Exception as e:
        logger.error(f"{e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception("Error during graph analysis.")

