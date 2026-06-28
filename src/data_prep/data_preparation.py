from pyspark.sql.types import StructType, StructField, StringType
import pyspark.sql.functions as F

import pandas as pd

from src.data_prep.dp_report import Dp_report
from src.utils.dataset import Dataset
from src.utils import file_utils
from src.utils.config_handler import Config_handler
from src.utils.logging_handler import Logging_handler
import traceback

from src.data_prep import semantic_triplets, graph_data_generation, validation, mapping


def execute(spark, common_config_handler, nodes_path, edges_path, organism):
    """
    Executing data preparation:
    1 - Filter the configured organism (species).\n
    2 - Filter the configured gene form.
        (If gene form if 'protein': remove all records related 'modified' or 'alternative').\n
    3 - Index generation (dataset observation reference).\n
    4 - Generate triplets from interactions.\n
    5 - Recover unlabelled chemicals and proteins, Mapping IDs for chemicals (MeSH identifier) and proteins (NCBI Gene identifier), Validation (remove unrecognized objects) and Index Filtering to keep only integral reactions (if configured).\n
    6 - Save the output (csv files for nodes, edeges and statistical report) as csv.\n
    :param spark: spark session.
    :param common_config_handler: common config handler.
    :param nodes_path: nodes dataset (output path).
    :param edges_path: edges dataset (output path).
    :param organism: selected organism.
    """

    config_handler = Config_handler("config-data_prep.yml")

    dataset_path = config_handler.read_property("dataset.chemical_protein.path")
    col_names_chemical_protein = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms", "Organism",
                                  "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]

    logger = Logging_handler(common_config_handler).get_logger(module_name="data_preparation")

    try:
        # associate dataset metadata
        mol_prot_dataset = Dataset(dataset_path, col_names_chemical_protein)

        # build dataset peparation report
        dp_report = Dp_report()

        logger.info(f"dataset path: {mol_prot_dataset.path}")
        logger.info(f"dataset cols: {mol_prot_dataset.columns}")

        # Create a schema where every column in your array is a StringType
        schema = StructType([StructField(name, StringType(), True) for name in col_names_chemical_protein])

        df = spark.read.csv(dataset_path, header=False, schema=schema)

        logger.info(f"Dataset count: {df.count()}")

        logger.info("Starting Data Preparation...")


        ########## STEP 1 - FILTER ORGANISM ########################################
        logger.info("Starting filtering organism")
        logger.info(f"Selected organism (species): '{organism}'")

        df_organism = df.filter(F.col("Organism") == organism) # ex: organism = Homo sapiens
        dp_report.selected_organism = organism

        dataset_count_by_organism = df_organism.count()
        dp_report.dataset_count_by_organism = dataset_count_by_organism

        logger.info(f"Dataset filter by organism count: {dataset_count_by_organism}")


        ########## STEP 2 - FILTER PROTEIN ########################################
        logger.info("Starting filtering gene form")
        gene_form = config_handler.read_property("gene_form")

        logger.info(f"Selected gene form: '{gene_form}'")
        dp_report.selected_gene_form = gene_form

        df_organism_gene = df_organism.filter(F.col("GeneForms") == gene_form) # ex: gene_form = protein

        dataset_count_by_gene_form = df_organism_gene.count()
        dp_report.dataset_count_by_gene_form = dataset_count_by_gene_form

        logger.info(f"Dataset filter by organism and gene form count: {dataset_count_by_gene_form}")

        logger.warning("If gene form = protein then variants will be removed")
        df_organism_protein = df_organism_gene.filter(
            ~(F.lower(F.col("Interaction")).contains("modified")) &
            ~(F.lower(F.col("Interaction")).contains("alternative"))
        ) if gene_form == "protein" else df_organism_gene

        if gene_form == "protein":
            logger.info(f"Dataset filter by only canonical proteins count: {df_organism_protein.count()}")


        ########## STEP 3 - INDEX GENERATION (dataset obeservation reference) ########################################
        logger.debug("Selecting fields: 'ChemicalID', 'ChemicalName', 'GeneID', 'GeneSymbol', 'Interaction', 'InteractionActions', 'PubMedIDs' from dataset")

        # Usining F.monotonically_increasing_id() we can define unique IDs but sequential numbers are not perfect
        df_required_fields = (df_organism_protein.coalesce(1) # to force sequential numbers with use coalesce function (no parallelization)
                              .withColumn("index", F.monotonically_increasing_id()) \
                              .select("index",
                                      "ChemicalID",
                                      F.trim(F.col("ChemicalName")).alias("ChemicalName"),
                                      "GeneID",
                                      F.trim(F.col("GeneSymbol")).alias("GeneSymbol"),
                                      "Interaction",
                                      "InteractionActions",
                                      "PubMedIDs"))


        ########## STEP 4 - GENERATE TRIPLETS FROM INTERACTIONS ########################################
        logger.info("Starting generate semantic triplets from interaction descriptions")

        logger.debug(f"Semantic triple will consider the following actions: {semantic_triplets.ACTIONS}")

        df_triplets = semantic_triplets.retrieve_semantic_triplets(df_required_fields)

        # Cache triplets dataframe
        logger.debug("Caching df_triplets dataframe")
        df_triplets.cache() # Mark the DataFrame to be saved in memory

        dataset_triplets_count = df_triplets.count()
        dp_report.dataset_triplets_count = dataset_triplets_count

        logger.debug("df_triplets has been cached")

        logger.info(f"Triplets extraction has generated {dataset_triplets_count} records")


        ########## STEP 5 - RECOVER, MAPPING IDs (chemicals and proteins) AND VALIDATION ########################################

        not_recognized_elements_count = df_triplets.filter(F.col('triple_object_type') == semantic_triplets.LABEL_OTHER).count()
        dp_report.not_recognized_elements_count = not_recognized_elements_count

        logger.warning(f"Discarded records counts: {not_recognized_elements_count}")

        # Attempting to recover rows that have been labelled with 'other' (chemicals and proteins)
        df_recovered_triplets = validation.recover_other_objects(spark=spark,
                                                                 logger=logger,
                                                                 config_handler=config_handler,
                                                                 df_triplets=df_triplets)

        # mapping chemicals (MeSH Identifier association by chem_id)
        df_triplets_with_chemical = mapping.map_chemicals(spark, logger, config_handler, df_recovered_triplets)

        # mapping genes (NCBI Gene Identifier association by gene_id)
        df_mapped_triplets = mapping.map_proteins(spark, logger, config_handler, df_triplets_with_chemical)

        logger.info("Only integral reactions will be considered")
        logger.debug(f"Before IDs Filtering triplets count: {df_mapped_triplets.count()}")

        '''
        FILTERING INDEX KEEPING ONLY TRIPLETS WITHOUT 'other' LABEL TO ACTUALIZE VALIDATION
        '''

        # filter by index in order to keep only integral interactions
        df_validated_triplets = validation.filter_index(logger, df_mapped_triplets)

        logger.debug(f"After IDs Filtering triplets count: {df_mapped_triplets.count()}")

        logger.info(f"Discarding all records in which 'triple_object_type' matches with '{semantic_triplets.LABEL_OTHER}'")

        logger.debug("Caching df_validated_triplets")
        df_validated_triplets.cache()

        dataset_validated_triplets_count = df_validated_triplets.count()
        dp_report.dataset_validated_triplets_count = dataset_validated_triplets_count

        logger.debug("df_validated_triplets has been cached")

        logger.info(f"Validated Triplets count: {dataset_validated_triplets_count} records")

        logger.debug("Unpersist df_triplets")
        df_triplets.unpersist(blocking=True) # Unpersist df_triplets to free memory

        logger.debug("df_triplets has been unpersisted")

        logger.info("Computing max reaction depth")

        # Compute max reaction depth to know what's the max number of nested reactions
        max_reaction_depth = semantic_triplets.get_max_reaction_depth(df_validated_triplets)

        logger.info(f"Max reaction depth is {max_reaction_depth}")

        dp_report.max_reaction_depth = max_reaction_depth

        logger.info("Handle unique IDs to prepare node generation")

        # Handle unique IDs for reactions, chemicals and proteins
        df_triplets_unique_id = graph_data_generation.handle_unique_node_id(logger=logger,
                                                                            df_triplets=df_validated_triplets,
                                                                            max_reaction_depth=max_reaction_depth)

        # Cache dataframe
        logger.debug("Caching df_triplet_unique_id")
        df_triplets_unique_id.cache()

        # Invoking an action (count()) to trigger caching
        df_triplets_unique_id.count()
        logger.debug("df_triplet_unique_id has been cached")

        logger.debug("Unpersist df_validated_triplets")
        df_validated_triplets.unpersist(blocking=True)  # Unpersist df_validated_triplets to free memory

        logger.debug("df_validated_triplets has been unpersisted")

        triplets_save = config_handler.read_property("output.triplets.save")

        if triplets_save:
            logger.info("Saving triplets dataframe")

            # Read triplets path from config
            triplets_path = config_handler.read_property("output.triplets.path")

            # Write df_triplets_unique_id to csv
            df_triplets_unique_id.write.format("csv") \
                .mode("overwrite") \
                .option("header", "true") \
                .option("sep", ";") \
                .save(triplets_path)

            logger.info(f"Triplets dataframe has been written to the following path: {triplets_path}")

        else:
            logger.info("Triplets dataframe will not be saved")
            logger.warning("If you want to save triplets dataframe, set 'triplets.save' property")


        ########## STEP 6 - SAVE DATA PREPARATION OUTPUT ########################################
        logger.info("Compute output datasets for saving data preparation")

        # Define nodes
        df_nodes = graph_data_generation.define_nodes(df_triplets=df_triplets_unique_id)

        # Cache dataframe
        logger.debug("Caching df_nodes")
        df_nodes.cache()

        # Invoke an action (count()) to trigger caching
        df_nodes.count()

        logger.debug("df_nodes has been cached")

        # Define relationships
        relationship_df = graph_data_generation.define_relationships(df_triplets=df_triplets_unique_id,
                                                                     node_df=df_nodes)

        # Save nodes
        logger.info("Saving nodes...")

        # Tranform nodes into Pandas dataframe for saving as a unique csv file
        df_pandas_nodes = df_nodes.toPandas()
        #df_pandas_nodes = df_nodes_reviewed.toPandas()

        current_nodes_output_path = file_utils.get_name_with_organism(nodes_path,
                                                                      organism)
        df_pandas_nodes.to_csv(current_nodes_output_path)

        logger.info(f"Nodes records have been saved to the following path: {current_nodes_output_path}")

        # Save relationships
        logger.info("Saving relationships...")

        # Tranform nodes into Pandas dataframe for saving as a unique csv file
        df_pandas_relationships = relationship_df.toPandas()

        current_edges_output_path = file_utils.get_name_with_organism(edges_path, organism)
        df_pandas_relationships.to_csv(current_edges_output_path)

        logger.info(f"Relationships records have been saved to the following path: {current_edges_output_path}")

        # Compute statistics on nodes
        distinct_chemical_count = df_nodes.filter(
            F.col("type") == semantic_triplets.LABEL_CHEMICAL)\
            .select("id").distinct().count()
        dp_report.distinct_chemical_count = distinct_chemical_count

        distinct_protein_count = df_nodes.filter(
            F.col("type") == semantic_triplets.LABEL_PROTEIN)\
            .select("id").distinct().count()
        dp_report.distinct_protein_count = distinct_protein_count

        distinct_reaction_count = df_nodes.filter(
            F.col("type") == semantic_triplets.LABEL_REACTION)\
            .select("id").distinct().count()
        dp_report.distinct_reaction_count = distinct_reaction_count

        logger.debug("Unpersist df_nodes")
        df_nodes.unpersist(blocking=True)
        logger.debug("df_nodes has been unpersisted")

        logger.info(f"Distinct Chemical counts: {distinct_chemical_count}")
        logger.info(f"Distinct Protein counts: {distinct_protein_count}")
        logger.info(f"Distinct Reaction counts: {distinct_reaction_count}")

        # Tranform statistical report into Pandas dataframe for saving as a unique csv file
        dp_report_df = pd.DataFrame([dp_report.__dict__])
        report_path = config_handler.read_property("output.report.dataprep.path")

        # Save statistical report
        logger.info("Saving Data Preparation Report")

        current_report_path = file_utils.get_name_with_organism(report_path, organism) # Rename report file name with date and hour
        dp_report_df.to_csv(current_report_path)

        logger.info(f"Data Preparation Report has been saved to the following path: {current_report_path}")

        logger.info("Data Preparation has terminated")

    except Exception as e:
        logger.error(f"{e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise Exception("Error during data preparation")

