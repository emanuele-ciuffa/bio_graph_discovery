from pyspark.sql.types import ArrayType, StructType, StructField, StringType
import pyspark.sql.functions as F
import re

from src.utils.config_handler import Config_handler

# Define the schema for our triplets
_triplet_schema = ArrayType(StructType([
    StructField("subject", StringType(), False),
    StructField("predicate", StringType(), False),
    StructField("object", StringType(), False)
]))

# Define constants (prefixes)
MAIN_ACTION_SUBJ = "main-action_"
REACTION_SUBJ = "reaction_"

# Define constants (labels)
LABEL_CHEMICAL = "chemical"
LABEL_PROTEIN = "protein"
LABEL_REACTION = "reaction"
LABEL_ACTION = "action"
LABEL_OTHER = "other"
LABEL_CHEMICAL_RECOVERED = "chemical" # only used as ID and not actually as label
LABEL_PROTEIN_RECOVERED = "protein"

config_handler = Config_handler("config-actions.yml")

# Possibile actions to consider
ACTIONS = config_handler.read_property("actions")

def _extract_triplet_properties(index, text):
    """
    This function will generate triplets properties ('has_agent', 'has_action', 'has_target').
    Both plain and nested reaction will be extracted.
    For each index will be produced at least three properties (for plain reaction: A action B),
    while for every next interaction associated to the same index other three properties will be generated
    (ex: 'A action [B action C]]' will generate 6 properties with different values: index_x, ('has_agent', 'has_action', 'has_target') and index x, ('has_agent', 'has_action', 'has_target').
    Nested reactions will be identified by a numeric suffix, while first number identifies the index.
    Ex: assuming we have a nested reaction such as 'A action [B action [C action D]]':
    reaction_4_0, reaction_4_1, reaction_4_2 (4 identifies the index, while 0, 1, 2 identify the reaction depth).
    :param index: source dataset index.
    :param text: interaction description.
    :return: triplets properties ('has_agent', 'has_action', 'has_target').
    """
    if not text:
        return []

    triplets = []

    pattern = "|".join(ACTIONS)

    actions_regex = r"((?:which\s+)?(?:" + pattern + r"))"

    current_text = text
    current_subject = f"{MAIN_ACTION_SUBJ}{index}"
    reaction_counter = index

    while True:
        # Case 1: Square bracket at the start --> [A] action B
        inverse_nested_match = re.search(r"^\[(.*)\]\s+" + actions_regex + r"\s+(.*)$", current_text)

        # Case 2: Nested structure --> A action [B]
        nested_match = re.search(r"^(.*?)\s+" + actions_regex + r"\s+\[(.*)\]$", current_text)

        # Case 3: Simple structure --> A action B
        simple_match = re.search(r"^(.*?)\s+" + actions_regex + r"\s+([^\[\]]+)$", current_text)

        if inverse_nested_match:
            nested_content, pred, obj = inverse_nested_match.groups()

            inner_match = re.search(r"^(.*?)\s+" + actions_regex + r"\s+(.*)$", nested_content.strip())
            if inner_match:
                i_subj, i_pred, i_obj = inner_match.groups()
                triplets.append((current_subject, "has_agent", i_subj.strip()))
                triplets.append((current_subject, "has_action", i_pred.strip()))
                triplets.append((current_subject, "has_target", i_obj.strip()))

            # Create the next reaction che having main action as agent
            next_reaction_id = f"{REACTION_SUBJ}{index}_{int(reaction_counter) - int(index)}"
            triplets.append((next_reaction_id, "has_agent", current_subject))
            triplets.append((next_reaction_id, "has_action", pred.strip()))
            triplets.append((next_reaction_id, "has_target", obj.strip()))

            reaction_counter += 1

            break

        elif nested_match:
            subj, pred, nested_content = nested_match.groups()
            #next_reaction_id = "reaction_{0}".format(reaction_counter if index != 0 else 1)
            next_reaction_id = f"{REACTION_SUBJ}{index}_{int(reaction_counter) - int(index)}"

            triplets.append((current_subject, "has_agent", subj.strip()))
            triplets.append((current_subject, "has_action", pred.strip()))
            triplets.append((current_subject, "has_target", next_reaction_id))

            current_text = nested_content.strip()
            current_subject = next_reaction_id

            reaction_counter += 1

            # No break here, because we have to analyse the content into [ ]

        elif simple_match:
            subj, pred, obj = simple_match.groups()
            triplets.append((current_subject, "has_agent", subj.strip()))
            triplets.append((current_subject, "has_action", pred.strip()))
            triplets.append((current_subject, "has_target", obj.strip()))
            break
        else:
            break

    return triplets


def _triplet_object_type(chemical, protein, triplet_predicate, triplet_object):
    """
    This function assing the 'triple_object_type' attribute as label to differentiate object type ('chemical', 'protein', 'reaction', 'action', 'other'). \n
    IMPORTANT NOTE: Sometimes 'ChemicalName' or 'GeneSymbol' could refer to a inner reaction (not the main reaction). \n
    So 'ChemicalName' and 'GeneSymbol' should be recovered, for both main reactions and inner reactions, in order to fully assing the 'triple_object_type' attribute.
    :param chemical: 'ChemicalName' value.
    :param protein: 'GeneSymbol' value.
    :param triplet_predicate: triplet predicate.
    :param triplet_object: triplet object.
    :return: the detected label for a given triplet object.
    """
    if str(triplet_predicate) == "has_action":
        return LABEL_ACTION
    elif REACTION_SUBJ in str(triplet_object) or MAIN_ACTION_SUBJ in str(triplet_object):
        return LABEL_REACTION
    if str(triplet_object) == str(chemical):
        return LABEL_CHEMICAL
    elif str(triplet_object).replace(" protein", "") == protein:
        return LABEL_PROTEIN
    else:
        return LABEL_OTHER

def _get_pred_number(predicate):
    """
    This function returns a specific number (0, 1 or 2) for the given predicate.
    :param predicate: predicate ('has_agent', 'has_action', 'has_target')
    :return: the number for the specified predicate.
    """
    if predicate == "has_agent":
        return 0
    elif predicate == "has_action":
        return 1
    else:
        return 2


def retrieve_semantic_triplets(df):
    """
    Retrieve semantic triplets (Subject, Predicate, Object) exctracting from Interaction descriptions
    :param df: Puspark Dataframe which must include columns 'index' and 'Interaction'
    :return: a Pyspark Dataframe with the following columns: "index", "Interaction", "triplet.subject", "triplet.predicate", "triplet.object", "ChemicalName", "GeneSymbol"
    """
    # Create the UDF wrappers
    extract_triplet_properties_udf = F.udf(_extract_triplet_properties, _triplet_schema)
    triplet_object_type_udf = F.udf(_triplet_object_type)
    get_pred_number = F.udf(_get_pred_number)

    triplets_df = (df.withColumn("triplet_array", extract_triplet_properties_udf(F.col("index"), F.col("Interaction"))) \
                   .withColumn("triplet", \
                               F.explode("triplet_array") \
                               ) \
                   .select("index", "Interaction", "triplet.subject", "triplet.predicate", "triplet.object",
                           "ChemicalName", "GeneSymbol") \
                   .withColumn("triple_object_type", \
                               triplet_object_type_udf(F.col("ChemicalName"), F.col("GeneSymbol"), F.col("predicate"),
                                                       F.col("object")) \
                               )
                   .withColumn("pred_numb", get_pred_number(F.col("predicate")))
                   .withColumn("id",
                               F.when(
                                   (F.col("triple_object_type") == LABEL_REACTION) &  # if the row is a reaction type
                                   (
                                       (
                                               (F.col("subject").startswith(
                                                   MAIN_ACTION_SUBJ)) |  # and (if subject is main-action or reaction)
                                               (F.col("subject").startswith(REACTION_SUBJ))
                                       )
                                   ),
                                   F.when(F.col("subject").startswith(MAIN_ACTION_SUBJ),  # in case i main action
                                          F.when(F.col("Interaction").rlike(r"\[\["), # in case the main reaction starts with a component interacting with [[..]
                                                 F.concat(F.lit(REACTION_SUBJ), F.col("index"), F.lit("_"), F.lit("1"))
                                                 ).otherwise(
                                              F.concat(F.lit(REACTION_SUBJ), F.col("index"), F.lit("_"), F.lit("0"))
                                          )
                                          )
                                   .otherwise(F.concat(F.lit(REACTION_SUBJ),
                                                       # in case is reaction the object will be composed by "reaction_" + [index] + last number (suffix) of subject
                                                       F.col("index"), F.lit("_"),
                                                       F.element_at(F.split(F.col("subject"), "_"), -1).cast(
                                                           "int") + 1))
                               )
                               .otherwise(
                                   F.concat(F.col("triple_object_type"), F.lit("_"), F.col("index"), F.lit("_"), \
                                            F.floor(F.monotonically_increasing_id()).cast("string")) \
                                   )
                               )
                   .withColumn("object", \
                               F.when(
                                   (F.col("triple_object_type") == LABEL_REACTION) & (
                                               F.col("predicate") != "has_agent"),
                                   F.col("id")
                               ).otherwise(F.col("object"))
                               ))

    return triplets_df


def get_max_reaction_depth(df_triplets):
    """
    This function compute the max reaction depth (max nested level) in the dataset.
    ex: considering a dataset with the following records we assume that the max reaction depth is 3 (look at index 2):
    index 1 | A action [B action C]
    index 2 | A action [B action [C action [D action E]]]
    index 3 | A action [B action [C action D]]

    :param df_triplets: pyspark datataframe containing triplets (subject, predicate, and object)
    :return: a number (int) that identifies the max reaction depth (max nested level).
    """
    # Group by 'index', count rows, and sort by count descending
    index_counts = df_triplets.groupBy("index") \
        .count() \
        .orderBy(F.col("count").desc())

    # Extract the specific index value and its count as variables:
    top_row = index_counts.first()

    max_reaction_depth = int((int(top_row['count']) / 3) - 1)

    return max_reaction_depth
