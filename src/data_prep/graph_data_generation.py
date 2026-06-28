import pyspark.sql.functions as F

from pyspark.sql.window import Window

from src.data_prep import semantic_triplets

_AGENT_OF = "AGENT_OF"
_TARGET_OF = "TARGET_OF"


def _generate_chemical_dict(df_triplets):
    """
    Generate a dataframe (key, ids, ids_row). For each chemical id (key) associate a list containing all chemical ids matching with the same chemical name.
    :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object)
    :return: chemical dictionary pyspark dataframe
    """
    # Filter the dataset to only include rows where triple_object_type is 'chemical'
    df_filtered = df_triplets.filter((F.col("triple_object_type") == semantic_triplets.LABEL_CHEMICAL) |
                                     (F.col("triple_object_type") == semantic_triplets.LABEL_CHEMICAL_RECOVERED))

    # Associate ids to the each key
    df_chemical_dict = df_filtered.withColumn(
        "ids",
        F.array_sort(F.collect_set("id").over(
            Window.partitionBy("object")
        ))
    ).select(
        F.col("id").alias("key"),
        F.col("ids")
    )

    return df_chemical_dict

def _generate_protein_dict(df_triplets):
    """
        Generate a dataframe (key, ids, ids_row). For each protein id (key) associate a list containing all protein ids matching with the same protein name.
        :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object)
        :return: protein dictionary pyspark dataframe
        """
    # Filter the dataset to only include rows where triple_object_type is 'protein'
    df_filtered = df_triplets.filter((F.col("triple_object_type") == semantic_triplets.LABEL_PROTEIN) |
                                     (F.col("triple_object_type") == semantic_triplets.LABEL_PROTEIN_RECOVERED))

    # Associate ids to the each key
    df_protein_dict = df_filtered.withColumn(
        "ids",
        F.array_sort(F.collect_set("id").over(
            Window.partitionBy("object")
        ))
    ).select(
        F.col("id").alias("key"),
        F.col("ids")
    )

    return df_protein_dict


def _get_reaction_unique_id_vocab(df_triplets, reaction_depth):
    """
    This function associate the current reaction id to a new reaction id to identify unique reactions.
    :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object).
    :param reaction_depth: current reaction depth to handle.
    :return: a pyspark dataframe with the association ('current_reaction_id', 'unique_reaction_id').
    """
    # Pivoting rows in order to obtain this structure: 'subject', 'has_agent', 'has_action', 'has_target'
    df_pivot = df_triplets.groupBy("subject") \
        .pivot("predicate", ["has_agent", "has_action", "has_target"]) \
        .agg(F.first("object"))

    # Define regex pattern
    depth_pattern = r"^" + semantic_triplets.REACTION_SUBJ + "\d+_(\d+)" #r"reaction_\d+_(\d+)"

    # Exclude reactions from 'has_agent' and 'has_target' columns
    df_pivot = df_pivot.filter(
        (~F.col("has_agent").startswith(semantic_triplets.LABEL_REACTION)) &
        (~F.col("has_agent").startswith(semantic_triplets.MAIN_ACTION_SUBJ))
    ).filter(
        (~F.col("has_target").startswith(semantic_triplets.LABEL_REACTION)) &
        (~F.col("has_target").startswith(semantic_triplets.MAIN_ACTION_SUBJ))
    ) \
        .withColumn("unique_reaction_id",
                    #F.concat(F.lit("n_" + semantic_triplets.LABEL_REACTION + "_" + str(iter_n) + "_"),
                    F.concat(F.lit("n_" + semantic_triplets.LABEL_REACTION + "_0_"),
                             F.sha2(F.concat_ws("||", "has_agent", "has_action", "has_target"), 256)  # Generate suffix with sha2 function
                             )
                    ).withColumn("depth_reaction",
                                 F.when(
                                     F.col("subject").rlike(depth_pattern),
                                     F.regexp_extract(F.col("subject"), depth_pattern, 1).cast("int")
                                 ).otherwise(
                                     F.when(
                                         F.col("has_agent").rlike(depth_pattern),
                                         F.regexp_extract(F.col("has_agent"), depth_pattern, 1).cast("int")
                                     ).otherwise(
                                         F.when(
                                             F.col("has_target").rlike(depth_pattern),
                                             F.regexp_extract(F.col("has_target"), depth_pattern, 1).cast("int")
                                         ).otherwise(F.lit(None))
                                     )
                                 )
                                 )

    # Excluding the reaction ids previously created
    df_reaction_id_vocab = df_pivot.withColumnRenamed("subject", "current_reaction_id")\
        .filter(~F.col("current_reaction_id").startswith("n_" + semantic_triplets.LABEL_REACTION)) \
        .filter(
        (F.col("depth_reaction").isNull()) |
        (F.col("depth_reaction").startswith(semantic_triplets.MAIN_ACTION_SUBJ)) |
        (F.col("depth_reaction") == reaction_depth)
    ) \
        .select("current_reaction_id",
                "unique_reaction_id")

    return df_reaction_id_vocab


def _update_interaction(df_triplets, df_unique_id_react_vocab):
    """
    This function updates interaction ids for df_triplets with reference to both subject and object.
    :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object).
    :param df_unique_id_react_vocab: pyspark dataframe with the association ('current_reaction_id', 'unique_reaction_id').
    :return: updated df_triplets with new interaction ids
    """
    # Update interaction IDs for subject
    df_triplets_interaction_subj = df_triplets.join(df_unique_id_react_vocab,
                                                    df_triplets.subject == df_unique_id_react_vocab.current_reaction_id,
                                                    "left") \
        .withColumn("subject",
                    F.when(F.col("unique_reaction_id").isNotNull(),
                           F.col("unique_reaction_id"))
                    .otherwise(F.col("subject"))
                    ) \
        .select(df_triplets.columns)


    df_triplets_interaction_obj = df_triplets_interaction_subj.join(df_unique_id_react_vocab,
                                                                    df_triplets_interaction_subj.object == df_unique_id_react_vocab.current_reaction_id,
                                                                    "left") \
        .withColumn("object",
                    F.when((F.col("triple_object_type") == semantic_triplets.LABEL_REACTION) &
                           (F.col("unique_reaction_id").isNotNull()),
                           F.col("unique_reaction_id"))
                    .otherwise(F.col("object"))
                    ) \
        .withColumn("id",
                    F.when((F.col("triple_object_type") == semantic_triplets.LABEL_REACTION) &
                           (F.col("unique_reaction_id").isNotNull()),
                           F.col("unique_reaction_id"))
                    .otherwise(F.col("id"))
                    ) \
        .select(df_triplets_interaction_subj.columns)

    return df_triplets_interaction_obj


def _deduplicate_main_actions(df_triplets):
    """
    This function deduplicate main-action_*. Subjects starting with 'main-action_' that have the same objects for the predicates ('has_agent', 'has_action', 'has_target') will be deduplicated.
    :param df_triplets: df_triplets: pyspark datataframe containing triplets (subject, predicate, and object).
    :return: updated df_triplets (pyspark dataframe)
    """

    # Create a dataframe ensuring that two subjects with the same triples have the exact same signature.
    subject_signatures = df_triplets.filter(
        F.col("subject").startswith(semantic_triplets.MAIN_ACTION_SUBJ)
    ).groupBy("Interaction", "subject") \
        .agg(F.sort_array(F.collect_list(F.struct("predicate", "object"))).alias("signature"))

    # Define a Window to pick one subject name per unique signature per Interaction.
    window_spec = Window.partitionBy("Interaction", "signature").orderBy("subject")

    # Filter to keep only the first occurrence (rank 1)
    df_subjects_to_keep = subject_signatures \
        .withColumn("rank", F.row_number().over(window_spec)) \
        .filter(F.col("rank") == 1) \
        .select(F.col("subject").alias("keep_subj"))

    # Broadcasting df_subjects_to_keep
    subj_to_keep_lookup = F.broadcast(df_subjects_to_keep)

    # Join df_triplets and broadcasted df_subjects_to_keep to keep only the rows for the selected subjects
    df_deduplicated_triplets = df_triplets.join(subj_to_keep_lookup,
                                                df_triplets.subject == subj_to_keep_lookup.keep_subj,
                                                "left") \
        .withColumn("keep",
                    F.when(
                        (~F.col("subject").startswith(semantic_triplets.MAIN_ACTION_SUBJ)) |
                        (F.col("keep_subj").isNotNull()),
                        F.lit(1)).otherwise(F.lit(0))
                    ).filter(F.col("keep") == 1) \
        .select(df_triplets.columns)

    return df_deduplicated_triplets


def handle_unique_node_id(logger, df_triplets, max_reaction_depth):
    """
    This function handles unique ids updating subjects and objects too.
    The unique ID replacement is based on specific level of depth and it is iterative (starting from the deepest til the inner reaction).
    :param logger: configured logger.
    :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object).
    :param max_reaction_depth: max reaction depth (max nested level).
    :return: updated df_triplets with unique ids (reaction, chemical, protein).
    """
    # Assing df_triplets to df_triplets_interact_reviewd in order to handle the loop
    df_triplets_interact_reviewd = df_triplets

    logger.info(f"Due to computed max_reaction_depth the iterations to update reaction ids will be {int(max_reaction_depth) + 1}")

    reaction_depth = max_reaction_depth

    for i in range(max_reaction_depth + 1):

        logger.debug(f"Iterarion #{i}")

        # For each iteration replace only specific level of depth (starting from the deepest til the inner reaction)
        df_unique_id_react_vocab = _get_reaction_unique_id_vocab(df_triplets=df_triplets_interact_reviewd,
                                                                 reaction_depth=reaction_depth)

        # update interactions
        df_triplets_interaction = _update_interaction(df_triplets_interact_reviewd, df_unique_id_react_vocab)

        # assign the resutl to df_triplets_interact_reviewd in order to handle next iteration in the loop
        #Truncate the lineage/DAG to prevent exponential planning time
        df_triplets_interact_reviewd = df_triplets_interaction.localCheckpoint()

        reaction_depth -= 1 # Update reaction_depth for next iteration


    # Generate chemical dictionary for existing ids
    df_chemical_dict = _generate_chemical_dict(df_triplets_interact_reviewd)

    # Updating chemical ids
    df_triplets_chem_updated = df_triplets_interact_reviewd.join(df_chemical_dict,
                                                                df_triplets_interact_reviewd.id == df_chemical_dict.key,
                                                                "left").withColumn("id",
                                                                                   F.when((F.col(
                                                                                       "triple_object_type") == semantic_triplets.LABEL_CHEMICAL) |
                                                                                          (F.col(
                                                                                              "triple_object_type") == semantic_triplets.LABEL_CHEMICAL_RECOVERED),
                                                                                          F.col("ids")[0]
                                                                                          # I take the first id in ids column
                                                                                          )
                                                                                   .otherwise(F.col("id"))) \
        .select(df_triplets_interact_reviewd.columns)  # Selecting only left table columns from the join

    # Generate protein dictionary for existing ids
    df_protein_dict = _generate_protein_dict(df_triplets_interact_reviewd)

    # Updating protein ids
    df_triplets_prot_updated = df_triplets_chem_updated.join(df_protein_dict,
                                                             df_triplets_chem_updated.id == df_protein_dict.key,
                                                             "left").withColumn("id",
                                                                                F.when((F.col(
                                                                                    "triple_object_type") == semantic_triplets.LABEL_PROTEIN) |
                                                                                       (F.col("triple_object_type") == semantic_triplets.LABEL_PROTEIN_RECOVERED),
                                                                                       F.col("ids")[0]  # I take the first id in ids column
                                                                                       )
                                                                                .otherwise(F.col("id"))).select(df_triplets_chem_updated.columns) # Selecting only left table columns from the join

    # Distinct all values to remove duplications
    df_triplets_updated = df_triplets_prot_updated.filter(F.col("id").isNotNull()).distinct()

    # Finally we deduplicate remining 'main-action_*' that have the same objects for the predicates ('has_agent', 'has_action', 'has_target')
    df_nodes_updated = _deduplicate_main_actions(df_triplets_updated)

    return df_nodes_updated


def _replace_with_node_ids(triplet_df, node_df):
    """
    This function will replace names of reactions, chemicals and proteins with their IDs
    :param triplet_df: pyspark datataframe containing triplets (subject, predicate, and object).
    :param node_df: pyspark datataframe containing nodes
    :return: updated pyspark datataframe triplet_df with IDs
    """
    # Broadcast node_df
    lookup = F.broadcast(node_df.select(
        F.col("name").alias("lookup_name"),
        F.col("id").alias("standard_id")
    ))

    # Process Object
    df_obj_replaced = triplet_df.join(
        lookup,
        triplet_df.object == lookup.lookup_name,
        "left"
    ).withColumn("object",
                 F.coalesce(F.col("standard_id"), F.col("object"))
                 ).drop("lookup_name", "standard_id")

    # Process Subject
    df_replaced = df_obj_replaced.join(lookup,
                                       df_obj_replaced.subject == lookup.lookup_name,
                                       "left"
                                       ).withColumn("subject",
                                                    F.coalesce(F.col("standard_id"), F.col("subject"))
                                                    ).drop("lookup_name", "standard_id")

    return df_replaced


def _apply_bidirectional_edges_for_ontologies(edges_df):
    """
    This function apply bidirectional edges between nodes.
    Only ontological will be bidirectional (HAS_AGENT/AGENT_OF, HAS_TARGET/TARGET_OF).
    :param edges_df: unidirectional edges dataframe
    :return: edges_df with bidirectional edges (only for ontological edges).
    """
    # Swap source/target and map the inverse interaction (AGENT_OF -> HAS_AGENT, TARGET_OF -> HAS_TARGET)
    inverse_df = (
        edges_df
        .filter(F.col("relationship").isin("AGENT_OF", "TARGET_OF"))
        .select(
            F.col("target_id").alias("source_id"),  # Swap 1
            F.col("source_id").alias("target_id"),  # Swap 2
            F.when(F.col("relationship") == "AGENT_OF", "HAS_AGENT")
            .otherwise("HAS_TARGET").alias("relationship"),
            F.col("reaction_id").alias("reaction_id")
        )
    )

    # adding inverse edge direction
    bi_edges_df = edges_df.unionByName(inverse_df)

    return bi_edges_df


def define_relationships(df_triplets, node_df):
    """
    This function will define the relationships between nodes.
    :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object)
    :param node_df: pyspark dataframe containing node IDs that will map output source_id and target_id
    :return: pyspark dataframe containing the relationships between nodes (source_id, target_id, interaction, reaction_id)
    """
    # Set ID from generated nodes (relationship output will set source_id and target_id from node ids)
    triplet_id_df = _replace_with_node_ids(df_triplets, node_df)

    # Pivot
    distinct_predicates = ["has_agent", "has_target", "has_action"]
    pivoted = triplet_id_df.groupBy("subject").pivot("predicate", distinct_predicates).agg(F.first("object"))

    # Ensure all expected columns exist to prevent "Column not found" errors
    for col_name in ["has_agent", "has_action", "has_target"]:
        if col_name not in pivoted.columns:
            pivoted = pivoted.withColumn(col_name, F.lit(None))

    # Generate Edges
    edges_df = pivoted.withColumn("edges", F.explode(F.array(
        # Edge Type 1: Semantic Interaction
        F.struct(
            F.col("has_agent").alias("source_id"),
            F.col("has_target").alias("target_id"),
            F.regexp_replace(F.upper(F.col("has_action")), " ", "_").alias("relationship"),
            F.col("subject").alias("reaction_id")
        ),
        # Edge Type 2: AGENT_OF
        F.struct(
            F.col("has_agent").alias("source_id"),
            F.col("subject").alias("target_id"),
            F.lit("AGENT_OF").alias("relationship"),
            F.col("subject").alias("reaction_id")
        ),
        # Edge Type 3: TARGET_OF
        F.struct(
            F.col("has_target").alias("source_id"),
            F.col("subject").alias("target_id"),
            F.lit("TARGET_OF").alias("relationship"),
            F.col("subject").alias("reaction_id")
        )
    ))) \
        .select("edges.*") \
        .filter("source_id IS NOT NULL AND target_id IS NOT NULL AND relationship IS NOT NULL") \
        .distinct()  # distinct all relationships

    # Produce bidirectional edges (only for ontological relationships)
    bi_edges_df = _apply_bidirectional_edges_for_ontologies(edges_df) \
        .withColumn("type", F.when(F.col("relationship").isin(["HAS_AGENT", "AGENT_OF", "HAS_TARGET", "TARGET_OF"]),
                                   "ontological").otherwise("functional")) # Determine if the relationship type is 'ontological' or 'functional'

    return bi_edges_df


def define_nodes(df_triplets):
    """
    This function define the nodes to contemplate (chemicals, proteins, and reactions). \n Nodes will be unique, so nodes will be grouped by name
    :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object)
    :return: pyspark dataframe containing the nodes (id, name, type)
    """

    # Define nodes from subject (only if it is a MAIN ACTION or a REACTION)
    df_subject_nodes = df_triplets.filter(
        (
                (F.col("subject").startswith(semantic_triplets.MAIN_ACTION_SUBJ)) |
                (F.col("subject").contains(semantic_triplets.REACTION_SUBJ))) &
        (F.col("predicate") == "has_action")
    ).select(
        F.col("subject").alias("id"), \
        F.col("subject").alias("name"), \
        F.lit(semantic_triplets.LABEL_REACTION).alias("type"), \
        F.lit(None).alias("chem_ID"), \
        F.lit(None).alias("gene_ID"), \
        F.col("object").alias("action") # select action from filtered dataframe
        )

    # Define nodes from objects
    df_object_nodes = (df_triplets.select(
        F.col("id"), \
        F.col("object").alias("name"), \
        F.col("triple_object_type").alias("type"), \
        F.col("chem_ID"), \
        F.col("gene_ID"), \
        F.lit(None).alias("action") \
        ).filter(F.col("type").isin([ \
        semantic_triplets.LABEL_CHEMICAL, \
        semantic_triplets.LABEL_CHEMICAL_RECOVERED, \
        semantic_triplets.LABEL_PROTEIN, \
        semantic_triplets.LABEL_PROTEIN_RECOVERED, \
        semantic_triplets.LABEL_REACTION \
        ]) \
        )
    )

    # Union between subject nodes (only main actions and reactions) and object nodes
    df_nodes = df_object_nodes.unionByName(df_subject_nodes) \
        .distinct() \
        .filter( # Keep only reactions with action property to remove reaction ID duplicates (only for "reaction" type)
        ~(
                (F.col("type") == semantic_triplets.LABEL_REACTION) &
                (F.col("action").isNull())
        )
    )

    return df_nodes
