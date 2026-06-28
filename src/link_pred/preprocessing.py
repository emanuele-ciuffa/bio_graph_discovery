import numpy as np
import pandas as pd
import tensorflow as tf


def map_id_with_label(labels_lookup, num_nodes, inv_node_map):
    """
    Maps categorical node labels to one-hot encoded feature vectors, aligned
    by graph index.

    This function iterates through all nodes in the graph by their integer indices,
    retrieves their original ID and label, and assigns a 3-dimensional binary
    vector [chemical, protein, reaction] based on the node type.

    :param labels_lookup: Dictionary mapping original node IDs to their categorical labels.
    :param num_nodes: Total number of nodes in the graph.
    :param inv_node_map: Dictionary mapping internal integer indices back to original node IDs.
    :return: A list of lists, where each sub-list is a 3-bit one-hot encoded vector.
    """
    # Create X (The Node Feature Matrix): define the one-hot encoding for your specific labels
    type_map = {
        'chemical': [1, 0, 0],
        'protein': [0, 1, 0],
        'reaction': [0, 0, 1]
    }

    # Build X with a safety check
    x_list = []
    for i in range(num_nodes):
        node_id = inv_node_map[i]
        label = labels_lookup.get(node_id)  # Returns None if missing

        if label in type_map:
            x_list.append(type_map[label])
        else:
            # If the label is missing or doesn't match type_map, use a default [0,0,0]
            # and print a warning so you can fix your Neo4j data
            print(f"Warning: Node {node_id} has unexpected label: {label}")
            x_list.append([0, 0, 0])

    return x_list


def get_X(labels_lookup, num_nodes, inv_node_map):
    """
    Constructs the node feature matrix X by mapping node metadata to a numerical format.

    This function aligns categorical or descriptive labels with the specific integer
    indices used in the graph's adjacency matrix, ensuring that the model receives
    the correct feature vector for every node.

    :param labels_lookup: A dictionary mapping unique node IDs to their respective
                          labels (e.g., chemical, protein, or reaction types).
    :param num_nodes: The total number of nodes, determining the number of rows in matrix X.
    :param inv_node_map: A mapping used to translate internal graph indices back to
                         original dataset IDs to ensure correct label association.
    :return: A numpy array (float32) of shape (num_nodes, feature_dimension)
             representing the initial features for all nodes.
    """

    # Associate label (chemical, protein, reaction) to node_id
    x_list = map_id_with_label(labels_lookup, num_nodes, inv_node_map)

    X = np.array(x_list, dtype='float32')

    return X

def generate_adjacency_matrix(num_nodes, indices_sorted):
    """
    Constructs a sparse adjacency matrix in TensorFlow format from an edge list.

    In a matrix A, the value at A[i, j] represents an edge from i to j.

    This function creates a SparseTensor where entries are 1.0 at the locations
    specified by indices_sorted and 0.0 elsewhere. Using a sparse representation
    is memory-efficient for large graphs.

    :param num_nodes: The total number of unique nodes in the graph (defines the matrix dimensions).
    :param indices_sorted: A numpy array of shape (E, 2) containing the source and target
                           node indices for all E edges.
    :return: A tf.sparse.SparseTensor of shape (num_nodes, num_nodes) representing the graph.
    """
    # Adjacency Matrix
    A = tf.sparse.SparseTensor(
        indices=indices_sorted.astype('int64'),
        values=np.ones(len(indices_sorted), dtype='float32'),
        dense_shape=(num_nodes, num_nodes)
    )

    return A

def get_E(edges_df, sort_idx, is_only_type):
    """
    Constructs an edge feature matrix by encoding categorical edge types into
    binary feature vectors.

    The function performs a two-level encoding:
    1. A broad categorization (Ontological [1, 0] vs. Functional [0, 1]).
    2. A specific one-hot encoding of the eight individual edge types.
    The resulting vectors are sorted to align with the Adjacency Matrix indices.

    :param edges_df: A pandas DataFrame containing at least an 'relationship' column.
    :param sort_idx: An array of indices used to reorder the edge features, ensuring they align perfectly with the sorted edges in the Adjacency Matrix.
    :param is_only_type: boolean value to define if Edges express only category (functional or ontological relationships)
    :return: A float32 numpy array of shape (num_edges, feature_dimension)
             representing encoded features for every edge in the graph.
    """

    # Dynamically one-hot encode the specific relationships present in the data
    E_relationships = pd.get_dummies(edges_df['relationship'])

    # Define the broad type: Ontological [1, 0] vs. Functional [0, 1]
    edges_df['is_ontological'] = (edges_df['type'] == 'ontological').astype(int)
    edges_df['is_functional'] = (edges_df['type'] == 'functional').astype(int)

    E_types = edges_df[['is_ontological', 'is_functional']].values

    '''
        The number of vectors in E matches with the number of type properties (actions).
        And the number of distinct vectors in E matches with the number of distinct type properties (actions).
        Here an example with a limited number of relationships:

        [0. 0. 0. 0. 0. 1. 0. 0.]			'RESULTS_IN_INCREASED_ACTIVITY_OF'
        [0. 0. 0. 0. 1. 0. 0. 0.]			'TARGET_OF'
        [0. 1. 0. 0. 0. 0. 0. 0.]			'INHIBITS_THE_REACTION'
        [0. 0. 0. 0. 0. 0. 0. 1.]			'AGENT_OF'
        [0. 0. 0. 1. 0. 0. 0. 0.]			'RESULTS_IN_DECREASED_ACTIVITY_OF'
        [0. 0. 0. 0. 0. 0. 1. 0.]			'RESULTS_IN_INCREASED_EXPRESSION_OF'
        [1. 0. 0. 0. 0. 0. 0. 0.]			'AFFECTS_THE_ACTIVITY_OF'
        [0. 0. 1. 0. 0. 0. 0. 0.]			'CO-TREATED_WITH'
    '''

    # In case is_only_type is True we produce Edge property based only on types ('is_ontological' or 'is_functional')
    if is_only_type:
        E = E_types[sort_idx].astype('float32')
    else:
        # Concatenate: Category_Bits which represent 'is_ontological' or 'is_functional' AND Type_Bits which represent edge types (ex: CO-TREATED_WITH, RESULTS_IN_DECREASED_ACTIVITY_OF, and so on)
        E_combined = np.concatenate([E_types, E_relationships.values], axis=1).astype('float32')
        E = E_combined[sort_idx]

    return E

def create_dataset(pos_edges, num_nodes, full_edge_set):
    """
    Generates a balanced dataset for link link_pred by combining existing
    positive edges with an equal number of randomly sampled negative edges.

    The function performs 'Negative Sampling' by picking pairs of nodes that
    do not have a connection in the graph, ensuring the model learns to
    identify both the presence and absence of links.

    :param pos_edges: A numpy array of shape (N, 2) containing node index
                      pairs that represent existing links (positive samples).
    :param num_nodes: The total number of nodes in the graph, used to define
                      the range for random sampling.
    :param full_edge_set: A set containing all existing edge tuples (source, target).
                          This is used to ensure sampled negative edges are
                          truly non-existent in the original graph.
    :return: A tuple (pairs, labels) where:
             - pairs: A numpy array containing both positive and negative edge samples.
             - labels: A float32 numpy array where 1.0 represents a positive edge and 0.0 represents a negative edge.
    """
    neg_pairs = []
    while len(neg_pairs) < len(pos_edges):
        idx = np.random.randint(0, num_nodes, (2,))
        if idx[0] != idx[1] and tuple(idx) not in full_edge_set:
            neg_pairs.append(idx)

    pairs = np.concatenate([pos_edges, np.array(neg_pairs)], axis=0)
    labels = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_pairs))], axis=0).astype('float32')
    return pairs, labels
