from src.utils import file_utils, spark_initializer
from src.utils.config_handler import Config_handler
from src.utils.logging_handler import Logging_handler

from src.data_prep import semantic_triplets

import pyspark.sql.functions as F

common_config_handler = Config_handler("config-common.yml")

logger = Logging_handler(common_config_handler).get_logger(module_name="test.data_prep")

logger.info("TESTING Data Preparation results...")

# Instatiate spark session
spark = spark_initializer.get_spark_session(logger)

# Variable to be update with False only in case the single test failed
are_all_tests_ok = True

selected_organism = common_config_handler.read_property("organism")

nodes_path = file_utils.get_name_with_organism(
    common_config_handler.read_property("dataset.graph.nodes.path"),
    selected_organism
)

edges_path = file_utils.get_name_with_organism(
    common_config_handler.read_property("dataset.graph.edges.path"),
    selected_organism
)

# Reading nodes output
df_nodes = spark.read.csv(nodes_path, header=True)

logger.info(f"nodes output dataset path: {nodes_path}")
logger.info(f"edges output dataset path: {edges_path}")

# Reading edges output
df_edges = spark.read.csv(edges_path, header=True)

logger.info("TESTING NODES UNIQUENESS*************************************************************************************")
nodes_rows = df_nodes.count()
logger.info(f"Number of rows in nodes output dataset: {nodes_rows}")

nodes_unique = df_nodes.select("id").distinct().count()
logger.info(f"Distinct node IDs in nodes output dataset: {nodes_unique}")

if nodes_rows == nodes_unique:
    logger.info("OK - generate nodes are unique in nodes output dataset")
else:
    logger.warning("FAIL - generate nodes are NOT unique in nodes output dataset")
    are_all_tests_ok = False

logger.info("TESTING ID MATCH BETWEEN NODES AND EDGES ********************************************************************")

# Combine source_id and target_id into a single column of unique IDs from edges
edge_ids = (df_edges.select("source_id")
            .union(df_edges
                   .select("target_id")
                   ).distinct()
            .withColumnRenamed("source_id", "id"))

# Count node ids
node_ids = df_nodes.select("id")

# Using subtract finds rows in 'edge_ids' that are not in 'df_nodes.id'
edge_missing_ids = edge_ids.subtract(node_ids)

# Evaluate the result
edge_missing_count = edge_missing_ids.count()

# Using subtract finds rows in 'edge_ids' that are not in 'df_nodes.id'
node_missing_ids = node_ids.subtract(edge_ids)

if edge_missing_count == 0:
    logger.info("OK - All IDs in df_edges match with the IDs in df_nodes.")
else:
    logger.warning(f"FAIL: Found {edge_missing_count} IDs in edges that are missing from df_nodes.")
    logger.info("Showing sample missing IDs:")
    edge_missing_ids.show()
    are_all_tests_ok = False

# Evaluate the result
node_missing_count = node_missing_ids.count()

if node_missing_count == 0:
    logger.info("OK - All IDs in df_nodes match with the IDs in df_edges.")
else:
    logger.warning(f"FAIL: Found {node_missing_count} IDs in nodes that are missing from edges.")
    logger.info("Showing sample missing IDs:")
    node_missing_ids.show()
    are_all_tests_ok = False



logger.info("TESTING EDGES EMPTY COLUMNS**********************************************************************************")
# ,source_id,target_id,interaction

# check if 'source_id' columns has empty values
edges_source_null = df_edges.filter((F.col("source_id").isNull()) |
                                    (F.col("source_id") == "null") |
                                    (F.col("source_id") == "")).count()

if edges_source_null == 0:
    logger.info("OK - the column 'source_id' has no NULL values")
else:
    logger.warning(f"FAIL - the column 'source_id' has some NULL values: {edges_source_null} records")
    are_all_tests_ok = False

# chech if 'target_id' columns has empty values
edges_target_null = df_edges.filter((F.col("target_id").isNull()) |
                                    (F.col("target_id") == "null") |
                                    (F.col("target_id") == "")).count()

if edges_target_null == 0:
    logger.info("OK - the column 'target_id' has no NULL values")
else:
    logger.warning(f"FAIL - the column 'target_id' has some NULL values: {edges_target_null} records")
    are_all_tests_ok = False

# check if 'interaction' columns has empty values
edges_interaction_null = df_edges.filter((F.col("interaction").isNull()) |
                                         (F.col("interaction") == "null") |
                                         (F.col("interaction") == "")).count()

if edges_interaction_null == 0:
    logger.info("OK - the column 'interaction' has no NULL values")
else:
    logger.warning(f"FAIL - the column 'interaction' has some NULL values: {edges_interaction_null} records")
    are_all_tests_ok = False

logger.info("TESTING NODES EMPTY COLUMNS**********************************************************************************")
# ,id,name,type,chem_ID,gene_ID
# chem_ID and gene_ID could be empty

# Check if 'id' columns has empty values
nodes_id_null = df_nodes.filter((F.col("id").isNull()) |
                                (F.col("id") == "null") |
                                (F.col("id") == ""))

nodes_id_null_count = nodes_id_null.count()

if nodes_id_null_count == 0:
    logger.info("OK - the column 'id' has no NULL values")
else:
    logger.warning(f"FAIL - the column 'id' has some NULL values: {nodes_id_null_count} records")
    nodes_id_null.show(truncate=False)
    are_all_tests_ok = False

# Check if 'name' columns has empty values
nodes_name_null = df_nodes.filter((F.col("name").isNull()) |
                                  (F.col("name") == "null") |
                                  (F.col("name") == ""))

nodes_name_null_count = nodes_name_null.count()

if nodes_name_null_count == 0:
    logger.info("OK - the column 'name' has no NULL values")
else:
    logger.warning(f"FAIL - the column 'name' has some NULL values: {nodes_name_null_count} records")
    nodes_name_null.show(truncate=False)
    are_all_tests_ok = False

logger.info("TESTING NODES TYPES******************************************************************************************")
# Check if all types are present (types are pre-defined, so they can be less or more)
# I read all node types
node_types = [semantic_triplets.LABEL_CHEMICAL, \
              semantic_triplets.LABEL_CHEMICAL_RECOVERED, \
              semantic_triplets.LABEL_PROTEIN, \
              semantic_triplets.LABEL_PROTEIN_RECOVERED, \
              semantic_triplets.LABEL_REACTION]

# extract distinct values (if LABEL_CHEMICAL_RECOVERED == LABEL_CHEMICAL_RECOVERED and LABEL_PROTEIN == LABEL_PROTEIN_RECOVERED)
unique_node_types = list(dict.fromkeys(node_types))

logger.info("Possible node types are:")
for unique_node_type in unique_node_types:
    logger.info(unique_node_type)

df_node_types = df_nodes.select("type").filter(F.col("type").isin(unique_node_types)).distinct()
df_node_types.show(truncate=False)
if df_node_types.count() == len(unique_node_types):
    logger.info("OK - node types are correct")
else:
    logger.warning("FAIL - node types are not correct")
    are_all_tests_ok = False

print("\nTESTING NODES TYPES MATCHING WITH CHEMICALS AND PROTEINS*****************************************************")
# "chem_ID" column can match only with "chemical" type otherwise it is empty
# "gene_ID" column can match only with "protein" type otherwise it is empty

df_check_node_type = df_nodes.select(
    F.col("id"),
    F.col("type"),
    F.col("chem_ID"),
    F.col("gene_ID"),
).withColumn("is_mapped_chemical",
             F.when(
                 (
                         (F.col("type") == semantic_triplets.LABEL_CHEMICAL) |
                         (F.col("type") == semantic_triplets.LABEL_CHEMICAL_RECOVERED)
                 ) & F.col("chem_ID").isNotNull(), True).otherwise(False)
             ) \
    .withColumn("is_mapped_protein",
                F.when(
                    (
                            (F.col("type") == semantic_triplets.LABEL_PROTEIN) |
                            (F.col("type") == semantic_triplets.LABEL_PROTEIN_RECOVERED)
                    ) & F.col("gene_ID").isNotNull(), True).otherwise(False)
                )

# Check if all chemicals are mapped
unmapped_chemicals = df_check_node_type.filter(
    (
            (F.col("type") == semantic_triplets.LABEL_CHEMICAL) |
            (F.col("type") == semantic_triplets.LABEL_CHEMICAL_RECOVERED)
    ) & (F.col("is_mapped_chemical") == False)
)

unmapped_chemicals_c = unmapped_chemicals.count()

if unmapped_chemicals_c > 0:
    logger.warning(f"FAIL - some chemicals are not mapped: {unmapped_chemicals_c} records")
    unmapped_chemicals.show(truncate=False)
    are_all_tests_ok = False
else:
    logger.info("OK - all chemicals are mapped")

# Check if all proteins are mapped
unmapped_proteins = df_check_node_type.filter(
    (
            (F.col("type") == semantic_triplets.LABEL_PROTEIN) |
            (F.col("type") == semantic_triplets.LABEL_PROTEIN_RECOVERED)
    ) & (F.col("is_mapped_protein") == False)
)
unmapped_proteins_c = unmapped_proteins.count()

if unmapped_proteins_c> 0:
    logger.warning(f"FAIL - some proteins are not mapped: {unmapped_proteins_c} records")
    unmapped_proteins.show(truncate=False)
    are_all_tests_ok = False
else:
    logger.info("OK - all proteins are mapped")


logger.info("UNIQUE AGENT_OF and TARGET_OF FOR EACH TARGET_ID************************************************************")
# Check if all reactions have just on AGENT_OF and TARGET_OF

are_all_reactions_unique = True

# Source_id with just one 'AGENT_OF'
df_edges_targetid_agent = df_edges \
    .filter(F.col("interaction") == 'AGENT_OF') \
    .groupBy("target_id", "interaction") \
    .agg(F.count("*").alias("c")) \
    .orderBy(F.col("c").desc())

if int(df_edges_targetid_agent.select(F.col("c")).first()[0]) > 1:
    logger.warning("FAIL - there are some target_id having more than an interaction 'AGENT_OF")
    are_all_reactions_unique = False
    are_all_tests_ok = False
else:
    logger.info("OK - all target_ids have just one interaction 'AGENT_OF")

# Source_id with just one 'TARGET_OF'
df_edges_targetid_target = df_edges \
    .filter(F.col("interaction") == 'TARGET_OF') \
    .groupBy("target_id", "interaction") \
    .agg(F.count("*").alias("c")) \
    .orderBy(F.col("c").desc())

if int(df_edges_targetid_target.select(F.col("c")).first()[0]) > 1:
    logger.warning("FAIL - there are some target_id having more than an interaction 'TARGET_OF'")
    are_all_reactions_unique = False
    are_all_tests_ok = False
else:
    logger.info("OK - all target_ids have just one interaction 'TARGET_OF'")

if are_all_reactions_unique == True:
    logger.info("OK - all reactions are unique (counting just one 'AGENT_OF' and one 'TARGET_OF')")
else:
    logger.warning("FAIL - not all reactions are unique (counting just one 'AGENT_OF' and one 'TARGET_OF')")


logger.info("FINAL OUTCOME************************************************************************************************")
if are_all_tests_ok == True:
    logger.info("All test succeded =)")
else:
    logger.warning("Some tests failed =(")
