import pyspark.sql.functions as F

from src.data_prep import semantic_triplets, vocabs


def _recover_chemicals(spark, logger, config_handler, df_triplets):
    """
    This function recovers chemicals by an external dataset.
    It will be assigned 'chemical' as label ('triple_object_type' dataframe column) for each object that matches.
    :param spark: spark session.
    :param logger: configured logger.
    :param config_handler: configured config_handler.
    :param df_triplets: triplet_df: pyspark datataframe containing triplets (subject, predicate, and object).
    :return: updated df_triplets (pyspark dataframe).
    """

    # Read Chemical Vocab
    df_chemical_vocab = vocabs.read_chemical_vocab(spark, logger, config_handler)

    # Broadcast Chemical Vocab
    chemical_look_up = F.broadcast(df_chemical_vocab)

    # Get matches from ChemicalName and generate new id (chemical_recovered_*) and new triple_object_type (chemical)
    recovered_chemical_df = df_triplets.join(
        chemical_look_up,
        df_triplets["object"] == chemical_look_up["chem_name"],
        "left"
    ).withColumn("id",
                 F.when(
                     (F.col("triple_object_type") == semantic_triplets.LABEL_OTHER) & (F.col("chem_name").isNotNull()),
                     F.regexp_replace(F.col("id"), semantic_triplets.LABEL_OTHER + "_",
                                      semantic_triplets.LABEL_CHEMICAL_RECOVERED + "_")
                 ).otherwise(F.col("id"))
                 ) \
        .withColumn("triple_object_type",
                    F.when(
                        (F.col("triple_object_type") == semantic_triplets.LABEL_OTHER) & (
                            F.col("chem_name").isNotNull()),
                        semantic_triplets.LABEL_CHEMICAL)
                    .otherwise(F.col("triple_object_type"))
                    ) \
        .drop("chem_ID")\
        .drop("chem_name")

    return recovered_chemical_df


def _recovered_proteins(df_triplets):
    """
    This function recovers proteins checking the object suffix (if it matches with ' protein').
    It will be assigned 'protein' as label ('triple_object_type' dataframe column) for each object that matches.
    No external dataset is used to assign 'protein' label.
    :param df_triplets: triplet_df: pyspark datataframe containing triplets (subject, predicate, and object).
    :return: updated df_triplets (pyspark dataframe).
    """

    # Updating 'triple_object_type' column with Protein if Object ends with " protein"
    recovered_gene_df = df_triplets \
        .withColumn("triple_object_type",
                    F.when(
                        (F.col("triple_object_type") == semantic_triplets.LABEL_OTHER) &
                        (F.lower(F.trim(F.col("object"))).endswith(" protein")), # Checked for the protein suffix
                         F.lit(semantic_triplets.LABEL_PROTEIN_RECOVERED)
                     ).otherwise(F.col("triple_object_type"))
                    ) \
        .withColumn("id",
                    F.when(
                        (F.lower(F.col("id"))).startswith(semantic_triplets.LABEL_OTHER) &
                        (F.lower(F.trim(F.col("object"))).endswith(" protein")),  # Checked for the suffix here,
                        F.regexp_replace(F.col("id"), semantic_triplets.LABEL_OTHER + "_",
                                         semantic_triplets.LABEL_PROTEIN_RECOVERED + "_")
                    ).otherwise(F.col("id"))
                    )

    return recovered_gene_df


def recover_other_objects(spark, logger, config_handler, df_triplets):
    """
    This function will recover 'other' labeled objects. Recovered records will be updated for "id" and "triple_object_type" columns.\n
    It will attempt to recover unlabeled ("other") from Vocab (for chemicals) and from triplets dataframe itself (for proteins).
    :param: spark session
    :param logger: defined logger from the invoking script.
    :param config_handler: config_handler which has been set with the specific config file path.
    :param df_triplets: triplet_df: pyspark datataframe containing triplets (subject, predicate, and object) that has to be validated.
    :return: df_triplets pyspark dataframe having recovered objects.
    """
    # Recover discarded records

    logger.info("Try to recover unknown chemicals from vocab")

    # Chemical recovery
    df_triplets_recov_chemical = _recover_chemicals(spark=spark,
                                                    logger=logger,
                                                    config_handler=config_handler,
                                                    df_triplets=df_triplets)

    logger.info("Trying to assign 'protein' labels to object having ' protein' suffix")
    logger.warning("Some recovered proteins could be discarded if no match with GeneID will be found during the mapping.")

    # Protein recovery
    df_triplets_recovery = _recovered_proteins(df_triplets=df_triplets_recov_chemical)

    return df_triplets_recovery



def filter_index(logger, df_triplets):
    """
    This function filter triplets to keep only integral interaction: \n
    'if the interaction 'A action_1 [B action_2 C]' has categorized at least one element as "other" then this interaction will be excluded.
    Moreover, all indices that have not at least 3 records or 3 multiples of 3 will be removed.
    :param logger: logger.
    :param df_triplets: pyspark Dataframe containing semantic triplets (subject, predicate, object).
    :return: filtered df_triplets
    """

    # Count how many rows exist per index (we assume that a complete interactions require 3 rows or multiples of 3)
    incomplete_indices = df_triplets \
        .groupBy("index") \
        .count() \
        .filter((F.col("count") % 3) != 0) \
        .select("index")

    # Identify only the IDs that must be removed
    ids_to_remove = df_triplets \
        .filter(F.col("triple_object_type") == semantic_triplets.LABEL_OTHER) \
        .select("index") \
        .union(incomplete_indices) \
        .distinct()

    # Broadcast IDs
    ids_look_up = F.broadcast(ids_to_remove)

    logger.info(f"Indices containing rows labeled as '{semantic_triplets.LABEL_OTHER}' are {ids_to_remove.count()}")

    df_filtered_triplets = df_triplets.join(
        ids_look_up,
        on="index",
        how="left_anti"
    )

    return df_filtered_triplets

