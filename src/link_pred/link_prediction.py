
# ECCConv consider weighted graphs analyzing edge properties: https://graphneural.network/layers/convolution/

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.utils.neo4j_handler import Neo4j_handler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.link_pred.GNN_model import LinkPredictor
from src.link_pred import preprocessing

from src.utils.config_handler import Config_handler
from src.utils.logging_handler import Logging_handler
from src.utils import file_utils

import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

import os
import sys


########################################################################################################################
############################################## DEFINE FUNCTIONS ########################################################
########################################################################################################################

@tf.function
def train_step(model, optimizer, x, a, e_type, e_relationship, pairs, labels, ontological_penalty):
    """
    Executes a single training iteration using a weighted binary cross-entropy loss.
    Ontological relationships are penalized, because the model has to focus on functional predictions.

    :param model: model.
    :param optimizer: optimizer.
    :param x: Input feature tensor (node features).
    :param a: Adjacency matrix or structural representation tensor.
    :param e_type: Tensor identifying edge groups (Ontological [1, 0] vs. Functional [0, 1]).
    :param e_relationship: Tensor representing specific edge relationship IDs or one-hot encodings.
    :param pairs: Tensor containing the specific node pairs/indices being evaluated.
    :param labels: Ground truth binary labels for the given pairs.
    :param ontological_penalty: Weight assigned to ontological relationships in loss.
    :return: The scalar mean weighted loss for the current batch.
    """

    with tf.GradientTape() as tape:
        predictions = model([x, a, e_type, e_relationship, pairs], training=True)
        predictions = tf.squeeze(predictions, axis=-1)
        labels = tf.cast(labels, tf.float32)

        # Get the dynamic count of positive samples from the tensor itself
        num_positives = tf.shape(e_type)[0]

        # Create the negative categories using tf.zeros
        neg_e_type = tf.zeros([num_positives, 2], dtype=tf.float32)

        # Use tf.concat to merge e_type and neg_e_type.
        # preparing the category labels for a training batch that includes both positive samples (real relationships from your data)
        # and negative samples (randomly sampled or "fake" relationships).
        full_e_type = tf.concat([e_type, neg_e_type], axis=0)

        # Define Ontological Relationships to be filtered
        is_ontological = tf.reduce_all(full_e_type == [1., 0.], axis=-1)

        # Calculate the raw loss per sample
        raw_loss = tf.keras.losses.binary_crossentropy(labels, predictions)

        # Define your weights based on categories
        weights = tf.where(is_ontological, ontological_penalty, 1.0) # Penalize Ontological [1, 0] assigning 0.1 to the weight else 1.0

        # Apply the weights
        weighted_loss = raw_loss * weights

        # Compute the mean of the weighted loss
        mean_loss = tf.reduce_mean(weighted_loss)

    # Determine how much each individual weight (parameter) in the model contributed to the total mean_loss
    gradients = tape.gradient(mean_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return mean_loss



def execute(common_config_handler, selected_organism):
    """
    This function perform link prediction by processing neo4j db. A Graph Convolutional Neural Network will be trained.
    Trained model will be used to predict links, by penalizing ontological edges.
    :param common_config_handler: Common configuration handler.
    :param selected_organism: Organism to be used for prediction.
    """

    # Instatiate logger
    logger = Logging_handler(common_config_handler).get_logger(module_name="link_prediction")

    # Instantiate config handler
    config_handler = Config_handler("config-link_prediction.yml")

    # Read 'gpu.enable' property from config file
    is_gpu_enable = config_handler.read_property("gpu.enable")

    logger.info(f"gpu.enable = {is_gpu_enable}")

    if is_gpu_enable != False:
        # Set to prevent CUDA incompatible libs: it will disable GPU
         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Read hyperparameters from config file
    learning_rate = config_handler.read_property("training.learning_rate")
    epochs = config_handler.read_property("training.epochs")
    test_size = config_handler.read_property("training.test_size")
    random_state = config_handler.read_property("training.random_state")
    ontological_penalty = config_handler.read_property("training.ontological_penalty")


    ########################################################################################################################
    ############################################## RETRIVE DATA FROM NEO4J #################################################
    ########################################################################################################################

    # Instantiate neo4j config handler
    neo4j_config_handler = Config_handler("config-neo4j.yml")

    # Reading references to connect to neo4j DB
    NEO4J_URI = neo4j_config_handler.read_property("neo4j.uri")
    NEO4J_USER = neo4j_config_handler.read_property("neo4j.user")
    NEO4J_PASSWORD = neo4j_config_handler.read_property("neo4j.password")

    # Instantiate neo4j handler
    neo4j_handler = Neo4j_handler(uri=NEO4J_URI,
                                  user=NEO4J_USER,
                                  password=NEO4J_PASSWORD,
                                  logger=logger)

    # Get last neo4j log
    last_neo4j_log = neo4j_handler.get_neo4j_last_log(logger=logger)

    # Read organism from last log
    organism = last_neo4j_log.organism

    # Validate that the selected organism matches the one in Neo4j
    if selected_organism != organism:
        logger.error(
            f"Selected organism is '{selected_organism}' which is not compatible with the current neo4j database.")
        raise ValueError(
            f"Organism mismatch: selected organism '{selected_organism}' "
            f"does not match the organism found in Neo4j '{organism}'."
        )

    logger.info(f"Selected organism (species): '{organism}'")

    logger.info("Retrieving graph from neo4j")

    # Get last neo4j log
    last_neo4j_log = neo4j_handler.get_neo4j_last_log(logger=logger)

    # Read organism from last log
    organism = last_neo4j_log.organism

    # Validate that the selected organism matches the one in Neo4j
    if selected_organism != organism:
        logger.error(
            f"Selected organism is '{selected_organism}' which is not compatible with the current neo4j database.")
        raise ValueError(
            f"Organism mismatch: selected organism '{selected_organism}' "
            f"does not match the organism found in Neo4j '{organism}'."
        )

    query = """
    MATCH (s)-[r]->(t)
    WHERE labels(s)[0] IN ['chemical', 'protein', 'reaction']
        AND labels(t)[0] IN ['chemical', 'protein', 'reaction']
    RETURN 
        s.id AS source, 
        labels(s)[0] AS source_label,
        t.id AS target, 
        labels(t)[0] AS target_label,
        r.relationship AS relationship,
        r.type AS type
    """
    edges_df = neo4j_handler.fetch_edges(query)

    logger.info("Graph was correctly retrieved from neo4j")

    # Read flag from config file to determine if ontological relationships will be considered or not
    exclude_ontological_relationships = config_handler.read_property("training.exclude_ontological_relationships")

    if exclude_ontological_relationships:
        logger.info(f"Excluding ontological relationships")
        edges_df = edges_df[edges_df["type"] == "functional"]
        # In case ontological relationships are excluded, this kind of relationships won't be penalized because they don't exist
        ontological_penalty = 1.0
    else:
        logger.info("Ontological relationships will be considered for training and inference")


    ########################################################################################################################
    ############################################## PREPROCESSING########## #################################################
    ########################################################################################################################

    logger.info("Starting pre-processing...")

    # Map and Sort to ensure A and E are aligned
    nodes_s = edges_df[['source', 'source_label']].rename(columns={'source': 'id', 'source_label': 'label'})
    nodes_t = edges_df[['target', 'target_label']].rename(columns={'target': 'id', 'target_label': 'label'})

    # Create a lookup ensuring that every node ID in the graph is linked to its correct label
    labels_lookup = pd.concat([nodes_s, nodes_t]).drop_duplicates('id').set_index('id')['label'].to_dict()

    # Merge source nodes and target nodes
    all_nodes = pd.unique(edges_df[['source', 'target']].values.ravel('K'))

    # Encoded nodes
    node_map = {node_id: i for i, node_id in enumerate(all_nodes)} # ex: 'chemical_24_106': 0, 'protein_13_61': 1, and so on

    # Decoded nodes
    inv_node_map = {i: node_id for node_id, i in node_map.items()} #  0: 'chemical_24_106', 1: 'protein_13_61', and so on

    # Counts all nodes
    num_nodes = len(all_nodes)

    # Sources and Targets will be identified by positional index
    sources = edges_df['source'].map(node_map).values # ex: "chemical_A" becomes 0
    targets = edges_df['target'].map(node_map).values # ex: "protein_A" becomes 3

    # Get raw indices (numerical representation of edge list)
    raw_indices = np.stack([sources, targets], axis=1) # List of couples of adjacent nodes by raw indices (ex: [[12 23] [ 8 27]])

    # Lexsort for SparseTensor requirements
    sort_idx = np.lexsort((raw_indices[:, 1], raw_indices[:, 0])) # ex: [ 0  6 30 21  1 23 15  2  5  ... ]

    # Sorting indices
    indices_sorted = raw_indices[sort_idx] # ex: [[ 0  4] [ 0  6] [ 0 18] [ 1 16] [ 2  4] [ 2  6] ...]

    # Get node Feature Matrix
    X = preprocessing.get_X(labels_lookup=labels_lookup,
                            num_nodes=num_nodes,
                            inv_node_map=inv_node_map)

    # Build Edge Feature Matrix
    E_type = preprocessing.get_E(edges_df=edges_df,
                                     sort_idx=sort_idx,
                                     is_only_type=True)

    logger.debug("E_type:\n", E_type)

    # Build Edge Feature Matrix
    E_relationship = preprocessing.get_E(edges_df=edges_df,
                                         sort_idx=sort_idx,
                                         is_only_type=False)

    logger.debug("E_relationship:\n", E_relationship)

    # First split: Extract the final Test Set
    idx_temp_pos, idx_test_pos, E_temp_type, E_test_type, E_temp_relationship, E_test_relationship = train_test_split(
        indices_sorted,
        E_type,
        E_relationship,
        test_size=test_size,  # e.g., 0.15 (15% for testing)
        random_state=random_state
    )

    # Second split: Extract Validation Set from the remaining data
    idx_train_pos, idx_val_pos, E_train_type, E_val_type, E_train_relationship, E_val_relationship = train_test_split(
        idx_temp_pos,
        E_temp_type,
        E_temp_relationship,
        test_size=test_size,
        random_state=random_state
    )

    # Read validation size from config file
    val_size = config_handler.read_property("training.validation_size")

    training_edges_unique, training_edges_counts = np.unique(E_train_relationship, return_counts=True)
    logger.debug(f"TRAINING: Edge relationships count: {dict(zip(training_edges_unique, training_edges_counts))}")

    val_edges_unique, val_edges_counts = np.unique(E_val_relationship, return_counts=True)
    logger.debug(f"VALIDATION: Edge relationships count: {dict(zip(val_edges_unique, val_edges_counts))}")

    test_edges_unique, test_edges_counts = np.unique(E_test_relationship, return_counts=True)
    logger.debug(f"TEST: Edge relationships count: {dict(zip(test_edges_unique, test_edges_counts))}")

    train_size = 1 - val_size - test_size

    logger.info(f"'val_size' (validation size) set to {val_size}' ({val_size * 100}%): {int(val_edges_counts[0])} interactions")

    logger.info(f"'test_size' (test size) set to {test_size}' ({test_size * 100}%): {int(test_edges_counts[0])} interactions")

    logger.info(f"'training size: {train_size}' ({train_size * 100}%): {int(training_edges_counts[0])} interactions")

    logger.debug(f"Train edges: {len(idx_train_pos)}, Val edges: {len(idx_val_pos)}, Test edges: {len(idx_test_pos)}")

    logger.debug("idx_train_pos:\n", idx_train_pos)

    # Generate Adjacency Matrix
    A_train = preprocessing.generate_adjacency_matrix(num_nodes=num_nodes,
                                                      indices_sorted=idx_train_pos)

    # Create a set of all existing edges for negative sampling check
    full_edge_set = set(zip(sources, targets))

    # Build Training set of node pairs and binary labels
    train_pairs, train_labels = preprocessing.create_dataset(pos_edges=idx_train_pos,
                                                             num_nodes=num_nodes,
                                                             full_edge_set=full_edge_set)

    # Build Validation set of node pairs and binary labels
    val_pairs, val_labels = preprocessing.create_dataset(pos_edges=idx_val_pos,
                                                         num_nodes=num_nodes,
                                                         full_edge_set=full_edge_set)

    # Build Test set of node pairs and binary labels
    test_pairs, test_labels = preprocessing.create_dataset(pos_edges=idx_test_pos,
                                                           num_nodes=num_nodes,
                                                           full_edge_set=full_edge_set)

    logger.debug("train_pairs:", len(train_pairs))
    logger.debug("test_pairs:", len(test_pairs))

    logger.info("Pre-processing has terminated")


    ########################################################################################################################
    ############################################## BUILD MODEL #############################################################
    ########################################################################################################################

    logger.info("Building model...")

    # Build the model
    model = LinkPredictor()

    # We pass a small slice of data (just 1 pair) to trigger the "build" process
    _ = model([X, A_train, E_train_type, E_train_relationship, train_pairs[:1]]) # If we remove this line we can't invoke model.summary()

    # Read model architecture path from config file
    model_architecture_path = file_utils.get_name_with_organism(
        file_path=config_handler.read_property("output.model_architecture_path"),
                                               organism=organism)

    # Save model architecture
    with open(model_architecture_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Define Optimization algorithm
    optimizer = Adam(learning_rate=learning_rate) # set optimizer

    # Define loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    logger.info("Model has been built")


    ########################################################################################################################
    ############################################## TRAIN MODEL #############################################################
    ########################################################################################################################

    logger.info(f"Starting Training: {num_nodes} nodes, {idx_train_pos.shape[0]} training edges for {epochs} epochs...")

    logger.debug("E_train_type:\n", E_train_type)

    # Read parameter 'target_val_auc' to determine the threshold to interrupt the training
    target_val_auc = config_handler.read_property("training.target_val_auc")

    logger.info(f"target_val_auc: {target_val_auc} | In case the validation >= this value, the training will stop")

    for epoch in range(1, epochs + 1):
        # Training Step (Updates weights)
        loss = train_step(model, optimizer, X, A_train, E_train_type, E_train_relationship, train_pairs, train_labels,
                          ontological_penalty)

        # Validation Step (Evaluates without updating weights)
        if epoch % 10 == 0:
            # Predict on Validation pairs.
            val_probs = model([X, A_train, E_train_type, E_train_relationship, val_pairs], training=False)

            # Squeeze probabilities and format labels to calculate metrics
            val_probs_sq = tf.squeeze(val_probs)
            val_labels_float = tf.cast(val_labels, tf.float32)

            # Calculate Validation Loss
            val_loss = loss_fn(val_labels_float, val_probs_sq).numpy()

            # Calculate Validation ROC-AUC
            val_roc = roc_auc_score(val_labels, val_probs.numpy().flatten())

            logger.info(
                f"Epoch {epoch:03d} | Train Loss: {loss.numpy():.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_roc:.4f}")

            # 2. EARLY STOPPING CHECK: Stop if the target AUC is reached
            if val_roc >= target_val_auc:
                logger.info(f" [EARLY STOPPING] Target Validation AUC of {target_val_auc} reached or exceeded! "
                            f"Stopping training early at epoch {epoch} with Val AUC: {val_roc:.4f}")
                break  # Exit the epoch loop immediately

    logger.info("Model has been trained")


    ########################################################################################################################
    ############################################## PREDICT DATA ############################################################
    ########################################################################################################################

    logger.info("Predicting data...")

    '''
    COMMENT-------------------------------------------------------------------------------------------------------------
    X: node feature matrix
    A_train: adjacency matrix
    E_train_type: edge feature (type: functional or ontological) matrix
    E_train_relationship: edge feature (relationship: binds to, inhibts the reaction of, and so on) matrix
    test_pairs: test pairs to be evaluated
    --------------------------------------------------------------------------------------------------------------------
    '''

    # --- Evaluation on Unseen Test Data ---
    test_probs = model([X, A_train, E_train_type, E_train_relationship, test_pairs], training=False).numpy()

    # Create a dataframe with the prediction results
    results = pd.DataFrame({
        "Source": [inv_node_map[p[0]] for p in test_pairs],
        "Target": [inv_node_map[p[1]] for p in test_pairs],
        "Score": test_probs.flatten().round(6),
        "Actual_Exist": ["YES" if l == 1 else "no" for l in test_labels]
    })


    ########################################################################################################################
    ############################################## EVALUATE METRICS ########################################################
    ########################################################################################################################

    logger.info("Evaluating model performance metrics")

    # Flatten probabilities to match label shape
    y_true = test_labels
    y_scores = test_probs.flatten()

    # Compute AUC-ROC
    roc_auc = roc_auc_score(y_true, y_scores)

    # Compute Average Precision (PR-AUC)
    # This is often more descriptive for link link_pred in sparse graphs
    avg_precision = average_precision_score(y_true, y_scores)

    logger.info(f"ROC-AUC Score:     {roc_auc:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")

    # Optional: Get ROC curve values if you want to plot later
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)


    ########################################################################################################################
    ############################################## SAVE RESULTS #######################################################
    ########################################################################################################################


    logger.info("Generating and saving ROC curve plot...")
    
    try:
        # Read output metrics image file path from config file
        output_plot_path = config_handler.read_property("output.metrics_roc_precision_image_path")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- ROC Curve ---
        axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Baseline')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.3)

        # --- PR Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        axes[1].plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AP = {avg_precision:.4f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc='upper right')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
        plt.close()


    except Exception as e:
        logger.error(f"Error: {e}")

    logger.info(f"Metrics plot has been saved to {output_plot_path}")


    # Merge with the original edges_df to get the relationship
    # We use a left join on Source and Target to bring in 'relationship'
    results = results.merge(
        edges_df[['source', 'target', 'relationship']],
        left_on=['Source', 'Target'],
        right_on=['source', 'target'],
        how='left'
    )

    # 3. Clean up and fill missing values for negative samples
    results.drop(columns=['source', 'target'], inplace=True)
    results['relationship'] = results['relationship'].fillna("-")

    print(results.sort_values("Score", ascending=False).head(100).to_string(index=False))

    # Read predictions path from config
    predictions_path = file_utils.get_name_with_organism(
        file_path=config_handler.read_property("output.predictions_path"),
        organism=organism)

    logger.info("Saving prediction results...")

    # Saving predictions results:
    results.sort_values("Score", ascending=False).to_csv(predictions_path, index=False)

    logger.info(f"Prediction results have been written to the following path: {predictions_path}")

    # Create the Pandas DataFrame
    df_report = pd.DataFrame([{
        "organism": organism,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "num_nodes": num_nodes,
        "num_tot_interactions": (training_edges_counts + val_edges_counts + test_edges_counts),
        "num_train_interactions": int(training_edges_counts[0]),
        "num_validation_interactions": int(val_edges_counts[0]),
        "num_test_interactions": int(test_edges_counts[0]),
        "training_set_perc": train_size,
        "validation_set_perc": val_size,
        "test_set_perc": test_size
    }])

    report_path = file_utils.get_name_with_organism(
        file_path=config_handler.read_property("output.report_path"),
        organism=organism)

    logger.info("Saving results...")

    # Save analysis results
    df_report.to_csv(report_path)

    logger.info(f"Link prediction report has been saved to the following path: {report_path}")


