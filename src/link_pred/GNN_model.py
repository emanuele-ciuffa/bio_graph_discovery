import tensorflow as tf
from tensorflow.keras import layers, Model
from spektral.layers import ECCConv


from src.utils.config_handler import Config_handler


class LinkPredictor(Model):
    def __init__(self):

        super().__init__()

        # Read parameters from config file
        config_handler = Config_handler("config-link_prediction.yml")
        channels = config_handler.read_property("model.channels")

        # ecc_conv_1 hyperparameters
        ecc_conv_1_kernel_network = config_handler.read_property("model.ecc_conv_1.kernel_network")
        ecc_conv_1_activation = config_handler.read_property("model.ecc_conv_1.activation")

        # ecc_conv_2 hyperparameters
        ecc_conv_2_kernel_network = config_handler.read_property("model.ecc_conv_2.kernel_network")
        ecc_conv_2_activation = config_handler.read_property("model.ecc_conv_2.activation")

        # ecc_conv_3 hyperparameters
        ecc_conv_3_kernel_network = config_handler.read_property("model.ecc_conv_3.kernel_network")
        ecc_conv_3_activation = config_handler.read_property("model.ecc_conv_3.activation")

        dropout_graph = config_handler.read_property("model.dropout.graph")
        dropout_dense = config_handler.read_property("model.dropout.dense")

        #dense_layers
        denselayer1_activation = config_handler.read_property("model.dense_layers.layer_1_activation")
        denselayer2_activation = config_handler.read_property("model.dense_layers.layer_2_activation")
        denselayer3_activation = config_handler.read_property("model.dense_layers.final_activation")


        ########################################################################################################################
        ############################################## MODEL ARCHITECTURE ######################################################
        ########################################################################################################################

        # LAYER 1
        # This is the 1-hop layer. Every node looks at its immediate neighbors.
        self.conv1 = ECCConv(channels=channels,
                             kernel_network=ecc_conv_1_kernel_network,
                             activation=ecc_conv_1_activation)

        # LAYER 2
        # This is the 2-hop layer. Since Conv1 already gave nodes information about their neighbors, Conv2 allows information to travel one step further.
        self.conv2 = ECCConv(channels=channels,
                             kernel_network=ecc_conv_2_kernel_network,
                             activation=ecc_conv_2_activation)

        # LAYER 3
        # This is the 3-hop layer. Since Conv2 already gave nodes information about their neighbors, Conv3 allows information to travel one step further.
        self.conv3 = ECCConv(channels=channels,
                             kernel_network=ecc_conv_3_kernel_network,
                             activation=ecc_conv_3_activation)

        # Layer Norm after GNN aggregation
        self.gnn_norm = layers.LayerNormalization(axis=-1)

        # Dropout layers
        self.dropout_graph = layers.Dropout(dropout_graph)
        self.dropout_dense = layers.Dropout(dropout_dense)

        # LAYER 3: first reasoning layer
        self.dense1 = layers.Dense(channels, activation=denselayer1_activation)

        # LAYER 4: second reasoning layer
        self.dense2 = layers.Dense(channels // 2, activation=denselayer2_activation) # Shrinking number of neurons to keep only most important information

        # FINAL LAYER: Decision Layer computing a Sigmoid output: Probability (0 to 1) that a functional link exists.
        self.dense3 = layers.Dense(1, activation=denselayer3_activation)

    def call(self, inputs):
        """
        Executes the forward pass of the LinkPredictor, transforming the global graph
        structure into local pair-wise link probabilities.

        :param inputs: A tuple/list containing:
            - x: Node feature matrix [N, F].
            - a: Sparse Adjacency matrix [N, N].
            - e_cat: Edge features [E, n_cat] for structural/ontological context.
            - e_type: Edge features [E, n_type] for specific relationship semantics.
            - pair_indices: Tensor of shape [M, 2] identifying node pairs for link_pred.
        :return: A Tensor of shape [M, 1] containing Sigmoid probabilities (0 to 1)
                 indicating the likelihood of a functional link existence.
        """

        x, a, e_cat, e_type, pair_indices = inputs

        # MESSAGE PASSING (Graph Contextualization)
        z = self.conv1([x, a, e_type])
        z = self.conv2([z, a, e_type])
        z = self.conv3([z, a, e_type])
        z = self.gnn_norm(z) # Normalizing z before gathering
        z = self.dropout_graph(z, training=False)  # Apply graph dropout


        # FEATURE EXTRACTION
        # Using tf.gather as a selector to retrieve the 32-dimensional embeddings (z)
        u_idx = tf.cast(pair_indices[:, 0], tf.int32)
        v_idx = tf.cast(pair_indices[:, 1], tf.int32)

        z_u = tf.gather(z, u_idx)  # Embedding of the Potential Source
        z_v = tf.gather(z, v_idx)  # Embedding of the Potential Target

        # DIRECTIONAL CONCATENATION
        combined = tf.concat([z_u, z_v], axis=-1)

        # REASONING
        x_out = self.dense1(combined)
        x_out = self.dropout_dense(x_out, training=False)  # Apply dense dropout
        x_out = self.dense2(x_out)

        # FINAL DECISION: Map refined reasoning to a probability score (0 to 1)
        return self.dense3(x_out)


