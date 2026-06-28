from graphframes import GraphFrame
from pyspark.sql import functions as F
import pandas as pd
from src.utils.neo4j_handler import Neo4j_handler
from src.utils.config_handler import Config_handler


def shortest_path(logger, neo4j_handler, g, target_id, max_iterations):
    """
        Computes weighted shortest paths to a target vertex using
        iterative message passing over a GraphFrame.

        This implementation performs distributed shortest-path relaxation
        by propagating distance updates across graph edges for a fixed
        number of iterations. It is designed as a weighted alternative
        to GraphFrames `shortestPaths()`, which does not support edge weights.

        Implemented algorithm is heuristic, because of fixed iterations.

        Weights will be updated to perform the algorith:
        The algorithm will consider only functional relationships except HAS_TARGET when it shares the reaction_id with the destination node.
        :param logger: logger.
        :param neo4j_handler: neo4j_handler.
        :param g: Graphframes graph.
        :param target_id: target id (destination id).
        :param max_iterations: max number of iterations for shortest path search.
        :return: a dataframe (id, dist, hops, vertices_path, edges_path) for each vertex (id) it will be computed the distance (dist) to reach the target_id.
        'dist': the sum of all weights in the path;
        'hops': the number of hops in the path;
        'vertices_path': the vertices in the path;
        'edges_path': the edges path in the path;
    """

    try:
        # Set value to simulate infinite
        INF = 999999.0

        # Define weights
        FUNCTIONAL_WEIGHT = 1.0
        ONTOLOGICAL_WEIGHT = 2.0

        # Retrieve reaction_ids of reaction nodes having the destination node as target
        reaction_ids_from_target = neo4j_handler.get_reaction_ids_from_target(target_id)

        # Initialize vertices
        v_init = g.vertices.withColumn(
            "dist", F.when(F.col("id") == target_id, 0.0).otherwise(INF)
        ).withColumn(
            "path",
            F.when(F.col("id") == target_id, F.array(F.col("id"))).otherwise(F.array().cast("array<string>"))
        ).withColumn(
            "rels",
            F.array().cast("array<string>")
        ).withColumn(
            "edge_reaction_ids",
            F.array().cast("array<string>")
        ).withColumn(
            "edge_reaction_id", # current edge reaction_id
            F.lit(None).cast("string")
        ).withColumn(
            "prev_edge_reaction_id", # previous edge reaction_id
            F.lit(None).cast("string")
        ).withColumn(
            "distances", #
            F.array().cast("array<double>")
        )

        # Define Weighted Edges removing all ontological relationships (Except the ones linked to destination id)
        weighted_edges = g.edges.withColumn(
            "weight", F.when(F.col("type") == "ontological", ONTOLOGICAL_WEIGHT).otherwise(FUNCTIONAL_WEIGHT)
        ).filter(
            (~(F.col("type") == "ontological"))|
            (F.col("reaction_id").isin(reaction_ids_from_target))
        )

        # This initialization MUST be inside the function so Python knows it's the same local variable
        g_work = GraphFrame(v=v_init, e=weighted_edges)

        # Algorithm iterations
        for i in range(max_iterations):
            # Send message payload (sendToSrc for backward mode / sendToDst for forward mode: WARNING: use backward mode to perform all vertices as source_id)
            messages = g_work.aggregateMessages(
                F.min(F.col("MSG")).alias("msg_struct"),
                sendToSrc=F.when(
                    F.col("dst")["dist"] < INF,
                    F.struct(
                        (F.col("dst")["dist"] + F.col("edge")["weight"]).alias("dist"),
                        F.col("dst")["path"].alias("path"),
                        F.col("edge")["relationship"].alias("edge_rel"),
                        F.col("dst")["rels"].alias("prev_rels"),
                        F.col("edge")["reaction_id"].alias("edge_reaction_id"), # current edge "reaction_id" value

                        # Append to the history array related to edge_reaction_ids
                        F.concat(
                            F.col("dst")["edge_reaction_ids"],
                            F.array(F.col("edge")["reaction_id"])
                        ).alias("edge_reaction_ids"),
                        F.element_at(F.col("dst")["edge_reaction_ids"], -1).alias("prev_edge_reaction_id"), # Select previous edge reaction_id

                        # Append to the history array related to edge_reaction_ids
                        F.concat(
                            F.col("dst")["distances"],
                            F.array(F.col("edge")["weight"])
                        ).alias("distances")
                    )
                ),
                sendToDst=None
            )

            # Enrich vertices with the new path step and save the edge_reaction_id context
            vertices_enriched = (g_work.vertices.join(messages, on="id", how="left") \
                .withColumn("is_better",
                            (F.col("msg_struct")["dist"] < F.col("dist")) & F.col("msg_struct").isNotNull()
                            ) \
                .withColumn("dist",
                            F.when(F.col("is_better"), F.col("msg_struct")["dist"]).otherwise(F.col("dist"))
                            ) \
                .withColumn("path",
                            F.when(F.col("is_better"),
                                   F.concat(F.array(F.col("id")), F.col("msg_struct")["path"]))
                            .otherwise(F.col("path"))
                            ) \
                .withColumn("rels",
                            F.when(F.col("is_better"),
                                   F.concat(F.array(F.col("msg_struct")["edge_rel"]),
                                            F.col("msg_struct")["prev_rels"]))
                            .otherwise(F.col("rels"))
                            ) \
                .withColumn("edge_reaction_ids",
                        F.when(F.col("is_better"), F.col("msg_struct")["edge_reaction_ids"])
                        .otherwise(F.col("edge_reaction_ids"))
                            ) \
                .withColumn("edge_reaction_id",
                            F.when(F.col("is_better"), F.col("msg_struct")["edge_reaction_id"])
                            .otherwise(F.col("edge_reaction_id"))
                            ) \
                .withColumn("prev_edge_reaction_id",
                            F.when(F.col("is_better"), F.col("msg_struct")["prev_edge_reaction_id"])
                            .otherwise(F.col("prev_edge_reaction_id"))
                            ) \
                .withColumn("distances",
                            F.when(F.col("is_better"), F.col("msg_struct")["distances"])
                            .otherwise(F.col("distances"))
                            ) \
                            .drop("msg_struct", "is_better"))

            if i % 4 == 0:
                vertices_enriched = vertices_enriched.localCheckpoint(eager=True)


            g_work = GraphFrame(v=vertices_enriched, e=g_work.edges)

    except Exception as e:
        logger.error(f"Shortest Path analysis failed. Error: {e}")
        raise

    return g_work.vertices.select(F.col("id"),
                                  F.col("dist"),
                                  F.size(F.col("rels")).alias("hops"), # Calculate size of the edges array to get hop count
                                  F.array_join(F.col("path"), ", ").alias("vertices_path"),
                                  F.array_join(F.col("rels"), ", ").alias("edges_path")
                                  ).orderBy("hops")


def run_community_detection(spark, logger, common_options, max_iter, exclude_ontological_edges):
    """
    Perform Community Detection by Label Propagation Algorithm.
    Returns numeric community labels while preserving original string node IDs.

    :param spark: spark session.
    :param logger: logger.
    :param common_options: neo4j common options.
    :param max_iter: number of iterations (Recommended: 5 to 10).
    :param exclude_ontological_edges: flag to determine if ontological edges are excluded or not.
    :return: pyspark dataframe representing (id, name, type, community) for each node. Community is a number.
    """

    # 1. Load Nodes - Keep ID as a string initially
    nodes_query = """
    MATCH (n) 
    WHERE labels(n)[0] IN ['chemical', 'protein', 'reaction']
    RETURN n.id AS id, 
           n.name AS name, 
           labels(n)[0] AS type
    """

    nodes_df = spark.read.format("org.neo4j.spark.DataSource") \
        .options(**common_options) \
        .option("query", nodes_query) \
        .load() \
        .withColumn("id", F.col("id").cast("string"))

    nodes_df = nodes_df.checkpoint()

    # Load Edges - Strict filtering based on node labels to prevent broken references
    edges_query = """
    MATCH (s)-[r]->(t) 
    WHERE labels(s)[0] IN ['chemical', 'protein', 'reaction']
      AND labels(t)[0] IN ['chemical', 'protein', 'reaction']
    RETURN s.id AS src, t.id AS dst, type(r) AS interaction
    """

    edges_df = spark.read.format("org.neo4j.spark.DataSource") \
        .options(**common_options) \
        .option("query", edges_query) \
        .load() \
        .withColumn("src", F.col("src").cast("string")) \
        .withColumn("dst", F.col("dst").cast("string"))

    if exclude_ontological_edges:
        logger.info("Excluding ontological edges...")
        ontological_interactions = ['HAS_AGENT', 'AGENT_OF', 'HAS_TARGET', 'TARGET_OF']
        edges_df = edges_df.filter(~F.col("interaction").isin(ontological_interactions))

    edges_df = edges_df.checkpoint()

    # Create a Master String-to-Numeric ID Mapping (Crucial for GraphX/GraphFrames)
    logger.info("Generating 64-bit numeric IDs for GraphX engine compatibility...")
    string_to_num_map = nodes_df.select("id").distinct() \
        .withColumn("numeric_id", F.monotonically_increasing_id()) \
        .checkpoint()

    # Map Vertex DataFrame to Numeric IDs
    numeric_nodes = nodes_df.join(string_to_num_map, "id") \
        .select(
        F.col("numeric_id").alias("id"),  # GraphFrames requires the vertex key to be named 'id'
        F.col("id").alias("original_string_id"),
        F.col("name"),
        F.col("type")
    ).checkpoint()

    # Map Edge DataFrame to Numeric IDs
    src_map = string_to_num_map.withColumnRenamed("id", "src_str").withColumnRenamed("numeric_id", "src_num")
    dst_map = string_to_num_map.withColumnRenamed("id", "dst_str").withColumnRenamed("numeric_id", "dst_num")

    numeric_edges = edges_df \
        .join(src_map, edges_df.src == src_map.src_str, "inner") \
        .join(dst_map, edges_df.dst == dst_map.dst_str, "inner") \
        .select(
        F.col("src_num").alias("src"),
        F.col("dst_num").alias("dst")
    )

    # Make Edges Undirected
    logger.info("Converting directed numeric edges to undirected...")
    reversed_edges = numeric_edges.withColumnRenamed("src", "tmp") \
        .withColumnRenamed("dst", "src") \
        .withColumnRenamed("tmp", "dst")

    undirected_numeric_edges = numeric_edges.unionByName(reversed_edges).dropDuplicates(["src", "dst"]).checkpoint()

    # Initialize GraphFrame with Pure Numeric Data
    g = GraphFrame(numeric_nodes, undirected_numeric_edges)

    logger.info(f"Starting LPA Community Detection (Iterations: {max_iter})...")

    try:
        # Run LPA execution
        result = g.labelPropagation(maxIter=max_iter)

        # Output original string IDs for nodes, but KEEP the community label as a number
        communities = result.select(
            F.col("original_string_id").alias("id"),
            F.col("name"),
            F.col("type"),
            # 'label' is the numeric community ID. If null, fallback to the node's own internal numeric 'id'
            F.coalesce(F.col("label"), F.col("id")).alias("community")
        )

        communities.cache()
        logger.info(f"Label Propagation Algorithm finished successfully. Processed {communities.count()} nodes.")

    except Exception as e:
        logger.error(f"Label Propagation Algorithm failed. Error: {e}")
        raise

    return communities


def degree_centrality(logger, g):
    """
    Compute degree centrality.
    Degree centrality formula applied for each node: sum(degree) / (number_of_nodes - 1).
    Moreover the function will also compute in-degree and out-degree for each node.
    :param logger: logger.
    :param g: graph.
    :return: dataframe representing for each node: id, name, inDegree, outDegree, type, degree, degree_centrality.
    """
    try:
        # Setting degree centrality denominator
        n_vertices = g.vertices.count()
        degree_centrality_denominator = n_vertices - 1 if n_vertices > 1 else 1

        degree_df = (g.degrees \
            .join(g.vertices, "id") \
            .select("id", "name", "type", "degree")
                     .withColumn("degree_centrality", F.col("degree") / degree_centrality_denominator) \
            .orderBy(F.desc("degree")))

        # Define in_degree and out_degree
        in_df = g.inDegrees
        out_df = g.outDegrees

        # Generate dataframe with in_degree and out_degree
        node_stats_df = g.vertices.select("id", "name") \
            .join(in_df, on="id", how="left") \
            .join(out_df, on="id", how="left") \
            .fillna(0)  # Replace nulls with 0 for nodes with no connections

        degree_enriched_df = (node_stats_df.join(
            degree_df.drop("name"),
            on="id",
            how="inner"
        ).select("id",
                 "name",
                 "inDegree",
                 "outDegree",
                 "type",
                 "degree",
                 "degree_centrality").orderBy(F.desc("degree")))

    except Exception as e:
        logger.error(f"Degree Centrality analysis failed. Error: {e}")
        raise

    return degree_enriched_df



def connected_components(spark, logger, larget_component_path, g):
    """
    Compute Connected Components (including Largest Component).
    If a graph is disconnected, the diameter is technically infinite, that's why we compute the largest component instead.
    Connected components will be saved as csv file.
    - largest_component_id: largest component id;
    - largest_component_count: number of vertices of the largest component;
    - components_count: number of connected components.
    :param spark: spark session.
    :param: logger: logger.
    :param larget_component_path: largest component (output) path.
    :param g: graph dataframe.
    :return: a tuple (largest_component_id, largest_component_count, components_count).
    """
    try:
        # WARNING: Disable spark sql optimizer as workaround for a issue caused by a bug of Graphrames 0.8.3
        spark.conf.set("spark.sql.adaptive.enabled", "false")

        # Get connected connected components (list of nodes associated to the related connecteg.connectedComponents()d component)
        connected_components_df = g.connectedComponents()

        # count connected component (nodes count by connected component)
        counts_df = connected_components_df.groupBy("component").count()

        # Get the top row as a Row object to detect the largest component
        top_row = counts_df.orderBy(F.desc("count")).first()

        # Extract values 'largest_component_id' and 'largest_component_val'
        largest_component_id = top_row["component"]
        largest_component_count = top_row["count"]

        # obtain a dataframe listing all nodes associating component id and the related count
        enriched_components = connected_components_df.join(
            F.broadcast(counts_df),
            on="component",
            how="inner"
        ).select(F.col("id"),
                 F.col("type"),
                 F.col("name"),
                 F.col("component").alias("component_id"),
                 F.col("count").alias("component_count"))

        # Save Largest Component results
        (enriched_components.orderBy(F.desc("component_count"))
         .toPandas()
         .to_csv(larget_component_path, index=False))

        logger.info(f"Largest components have been saved to the following path: {larget_component_path}")

        # number of connected components
        components_count = counts_df.count()

        # Re-enable spark sql optimizer for next analysis
        spark.conf.set("spark.sql.adaptive.enabled", "true")

    except Exception as e:
        logger.error(f"Largest Component analysis failed. Error: {e}")
        raise

    return (largest_component_id, largest_component_count, components_count)
