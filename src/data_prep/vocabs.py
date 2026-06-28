def read_chemical_vocab(spark, logger, config_handler):
    """
    This function reads chemical vocab data.
    :param spark: spark session.
    :param logger: configured logger.
    :param config_handler: config_handler which has been set with the specific config file path.
    :return: pyspark dataframe containing chemicals vocab (columns: 'chemi_ID' and 'chem_name').
    """
    chemical_vocab_path = config_handler.read_property("data_recovery.chemicals.path")

    chemical_vocab_df = spark.read.csv(chemical_vocab_path, header=True)\
        .select("ChemicalID", "ChemicalName")\
        .withColumnRenamed("ChemicalID", "chem_ID")\
        .withColumnRenamed("ChemicalName", "chem_name")

    logger.debug(f"chemical_vocab_df count {chemical_vocab_df.count()}")

    logger.info("Trying to recover discarded chemicals from chemical vocab")

    return chemical_vocab_df


def read_gene_vocab(spark, logger, config_handler):
    """
    This function reads gene vocab data.
    :param spark: spark session.
    :param logger: configured logger.
    :param config_handler: config_handler which has been set with the specific config file path.
    :return: pyspark dataframe containing gene vocab (columns: 'gene_ID', 'gene_name' and 'gene_symbol').
    """
    gene_vocab_path = config_handler.read_property("data_recovery.genes.path")

    gene_vocab_df = spark.read.csv(gene_vocab_path, header=True)\
        .select("GeneID", "GeneName", "GeneSymbol")\
        .withColumnRenamed("GeneID", "gene_ID") \
        .withColumnRenamed("GeneName", "gene_name") \
        .withColumnRenamed("GeneSymbol", "gene_symbol")

    logger.debug(f"gene_vocab_df count {gene_vocab_df.count()}")

    logger.info("Trying to recover discarded genes from gene vocab")

    return gene_vocab_df