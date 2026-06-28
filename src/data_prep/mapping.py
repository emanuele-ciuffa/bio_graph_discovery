import pyspark.sql.functions as F

from src.data_prep import semantic_triplets, vocabs


def map_chemicals(spark, logger, config_handler, df_triplets):
    """
    This function maps the Chemical IDs for each object having triple_object_type = 'chemical':
    MeSH Identifier association.
    :param spark: spark session.
    :param logger: logger.
    :param config_handler: config_handler which has been set with the specific config file path.
    :param df_triplets: triplets pyspark dataframe.
    :return: mapped chemicals (pyspark dataframe).
    """
    # Read Chemical Vocab
    df_chemical_vocab = vocabs.read_chemical_vocab(spark, logger, config_handler)

    # Broadcast Chemical Vocab Dataframe
    chemical_look_up = F.broadcast(df_chemical_vocab)

    # Map triplets with Chemical IDs from related vocab
    df_mapped_triplets = df_triplets.join(
        chemical_look_up,
        df_triplets["object"] == chemical_look_up["chem_name"],
        "left")\
        .drop("ChemicalName")\
        .filter(~(F.col("chem_ID").isNull() & (
            F.col("triple_object_type") == semantic_triplets.LABEL_CHEMICAL_RECOVERED)))  # Remove chemicals which have not an associated chem_id

    return df_mapped_triplets


def map_proteins(spark, logger, config_handler, df_triplets):
    """
    This function maps the Gene IDs for each object having triple_object_type = 'protein':
    NCBI Gene Identifier association.
    :param spark: spark session.
    :param logger: logger.
    :param config_handler: config_handler which has been set with the specific config file path.
    :param df_triplets: triplets pyspark dataframe.
    :return: mapped proteins (pyspark dataframe).
    """
    # Read Gene Vocab
    df_gene_vocab = vocabs.read_gene_vocab(spark, logger, config_handler)

    # Broadcast Gene Vocab Dataframe
    gene_look_up = F.broadcast(df_gene_vocab)

    # Map triplets with Gene IDs from related vocab
    df_mapped_triplets = df_triplets.join(
        gene_look_up,
        F.regexp_replace(df_triplets["object"], " protein", "") == gene_look_up["gene_symbol"],
        "left") \
        .drop("GeneSymbol") \
        .withColumn("triple_object_type",
                    F.when(
                        (F.col("triple_object_type") == semantic_triplets.LABEL_PROTEIN_RECOVERED) &
                        (F.col("gene_ID").isNull()),
                        semantic_triplets.LABEL_OTHER)
                    .otherwise(F.col("triple_object_type"))
                    )
    # Assign 'other' to recovered 'protein' that have no gene_id in the vocab: this objects will be discarded later

    return df_mapped_triplets