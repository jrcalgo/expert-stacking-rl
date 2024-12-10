"""
This module defines a simple ensemble of small MLP and CNN networks.
Each MLP and CNN networks are aggregated using an ensemble stacking or bagging procedure that then feeds into a final voting layer.
Reward performance-based weighting is used to indicate the importance of each model in the ensemble.
During bagging:
    - Whatever algorithm that is implementing MiniArchitectureEnsemble must pass replay buffer data to buffer_bagging().
    - Perform actions, accumulate trajectories, and calculate loss value of each bagged model in the ensemble
    - The aggregated mini networks use the value derived from the loss values x (1-performance weights)
      to update their nn parameters.
During stacking:
    - MiniArchitectureEnsemble directly initializes and uses stacked_aggregate_model as the aggregate meta learner.
    - Perform actions, accumulate trajectories, and calculate loss value of stacked_aggregate_model.
    - The aggregate model uses the entire loss value to update its nn parameters during backpropagation.
    - The aggregated mini networks use the value derived from the same loss value x (1-performance weights)
      to update their nn parameters.
After each training series, save current model parameters to disk, discard old model parameters, randomly select new
model indices for a new model minibatch while retaining the most performant model, and repeat the process...
"""

import numpy as np
import tensorflow as tf
from keras import Input
from tensorflow import keras

#
# def buffer_bagging(models: list[keras.Model], X: tf.Tensor, y: tf.Tensor, performance_weights: np.ndarray):
#     """
#     Buffer data for bagging.
#
#     Args:
#         models (list[keras.Model]): List of models in the ensemble.
#         X (tf.Tensor): Input data.
#         y (tf.Tensor): Target data.
#         performance_weights (np.ndarray): Performance weights for the models.
#     """
#     for i, model in enumerate(models):
#         model.train_on_batch(X, y, sample_weight=performance_weights[i])


def aggregate_expert_outputs(expert_q_values, method='average', weights=None) -> tf.Tensor:
    """
    Aggregates the Q-values from multiple experts using averaging without altering tensor shape.

    Parameters:
    - expert_values: Tensor of shape (batch_size, num_experts, act_dim)
    - method: Aggregation method ('average', 'weighted_average', 'max')
    - weights: Optional Tensor of shape (num_experts,) for weighted averaging

    Returns:
    - aggregated_val: Tensor of shape (batch_size, num_experts, act_dim)
    """
    if method == 'average':
        # Compute the average Q-value for each action across all experts
        aggregated_val = tf.reduce_mean(expert_q_values, axis=1, keepdims=True)  # Shape: [batch_size, 1, act_dim]
    elif method == 'weighted_average':
        if weights is None:
            raise ValueError("Weights must be provided for weighted_average aggregation.")
        # Normalize weights
        weights = tf.cast(weights, dtype=expert_q_values.dtype)
        weights = weights / tf.reduce_sum(weights)
        weights = tf.reshape(weights, [1, -1, 1])  # Shape: [1, num_experts, 1]
        # Apply weights
        weighted_val = expert_q_values * weights  # Shape: [batch_size, num_experts, act_dim]
        mean_val = tf.reduce_sum(weighted_val, axis=1, keepdims=True)  # Shape: [batch_size, 1, act_dim]
        # Broadcast the mean to all experts
        aggregated_val = tf.tile(mean_val, [1, tf.shape(expert_q_values)[1], 1])  # Shape: [batch_size, num_experts, act_dim]
    elif method == 'max':
        # Compute the maximum Q-value for each action across all experts
        max_val = tf.reduce_max(expert_q_values, axis=1, keepdims=True)  # Shape: [batch_size, 1, act_dim]
        # Broadcast the max to all experts
        aggregated_val = tf.tile(max_val, [1, tf.shape(expert_q_values)[1], 1])  # Shape: [batch_size, num_experts, act_dim]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return aggregated_val


def expert_MLP(input_dim: int, output_dim: int, seed: int, activation: list[str]):
    """
    Defines a simple Multi-Layer Perceptron (MLP).

    Args:
        activation:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output.
        seed (int): Random seed for reproducibility.

    Returns:
        keras.Model: The MLP model.
    """
    tf.random.set_seed(seed)
    with tf.device('/GPU:0'):
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation=activation[0]),
            keras.layers.Dense(output_dim)
        ])
    return model


def expert_CNN(input_shape: tuple, output_dim: int, kernel_size: int, stride: int, seed: int, activation: list[str], padding='same'):
    """
    Defines a simple Convolutional Neural Network (CNN).

    Args:
        input_shape (tuple): Shape of the input data (height, width, channels).
        output_dim (int): Dimension of output.
        kernel_size (int): Kernel size for convolutions.
        stride (int): Stride for convolutions.
        seed (int): Random seed for reproducibility.
        padding (str): Padding for convolutions ('same' or 'valid').

    Returns:
        keras.Model: The CNN model.
    """
    tf.random.set_seed(seed)
    with tf.device('/GPU:0'):
        model = keras.Sequential([
            Input(shape=input_shape),
            keras.layers.Conv2D(24, kernel_size, strides=stride, padding=padding, activation=activation[0]),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=stride),
            keras.layers.Flatten(),
            keras.layers.Dense(6, activation=activation[1]),
            keras.layers.Dense(output_dim)
        ])
    return model


def stacking_encoder(input_dim: int, action_dim: int, seed: int, activation: str = 'relu'):
    """
    Defines a simple meta-learning encoder.
    Args:
        input_dim:
        output_dim:
        seed:
        activation:

    Returns:

    """
    tf.random.set_seed(seed)
    with tf.device('/GPU:0'):
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dense(action_dim)
        ])
    return model


class ExpertStackingEncoder(keras.Model):
    """
    An ensemble of probabilistic miniMLP and miniCNN models with varied activations and performance-based weighting.
    """
    def __init__(self,
                 mlp_activations: list[list[str]],
                 cnn_activations: list[list[str]],
                 mlp_input_dim,
                 cnn_input_shape,
                 act_dim,
                 initial_seed=0,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 mlp_count=5,
                 cnn_count=5,
                 mlp_batch_size=3,
                 cnn_batch_size=3,
                 grad_dir: str = "./runtime_models",
                 **kwargs):
        super(ExpertStackingEncoder, self).__init__(**kwargs)
        tf.random.set_seed(initial_seed)
        numpy_rng = np.random.RandomState(initial_seed)

        # Default activations if not provided
        if mlp_activations is None:
            self.mlp_activations = [['relu', 'sigmoid'] for _ in range(mlp_count)]
        else:
            self.mlp_activations = mlp_activations
        if cnn_activations is None:
            self.cnn_activations = [['tanh', 'relu'] for _ in range(cnn_count)]
        else:
            self.cnn_activations = cnn_activations

        assert mlp_batch_size <= mlp_count, "MLP batch size cannot exceed the number of MLP models."
        assert cnn_batch_size <= cnn_count, "CNN batch size cannot exceed the number of CNN models."

        assert len(self.mlp_activations) >= mlp_count, "MLP activations must be specified for each MLP model."
        assert len(self.cnn_activations) >= cnn_count, "CNN activations must be specified for each CNN model."
        for act in self.mlp_activations:
            assert len(act) >= 2, "MLP activations must have 2 functions per model."
        for act in self.cnn_activations:
            assert len(act) >= 3, "CNN activations must have 3 functions per model."

        self.mlp_input_dim = mlp_input_dim
        self.cnn_input_shape = cnn_input_shape
        self.act_dim = act_dim
        self.initial_seed = initial_seed
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mlp_count = mlp_count
        self.cnn_count = cnn_count
        self.expert_count = mlp_count + cnn_count
        self.mlp_batch_size = mlp_batch_size
        self.cnn_batch_size = cnn_batch_size
        self.grad_dir = grad_dir

        # Random seeds for each model
        self.mlp_seeds = numpy_rng.randint(0, 2**32, size=mlp_count)
        self.cnn_seeds = numpy_rng.randint(0, 2**32, size=cnn_count)

        # Initialize expert model blocks
        self.mlps = [expert_MLP(mlp_input_dim, act_dim, self.mlp_seeds.item(i), mlp_activations[i]) for i in range(self.mlp_count)]
        self.cnns = [expert_CNN(cnn_input_shape, act_dim, kernel_size, stride, self.cnn_seeds.item(i), cnn_activations[i], padding)
                     for i in range(self.cnn_count)]
        self.experts = self.mlps + self.cnns

        # Initialize stack encoder
        encoder_input_size: int = act_dim + mlp_input_dim
        self.stack_encoder = stacking_encoder(encoder_input_size, act_dim, initial_seed)

        # self.performance_weights = PerformanceWeights(self.models, grad_dir)

        self.current_expert_minibatch: tf.Tensor | None = None
        self.current_expert_mask: tf.Tensor | None = None

        self.rotate_expert_minibatch()

    def rotate_expert_minibatch(self):
        """
        Set a new minibatch of expert models for the ensemble.
        """
        # Randomly select models for the minibatch
        mlp_model_indices = np.random.choice(self.mlp_count, self.mlp_batch_size, replace=False)
        cnn_model_indices = self.mlp_count + np.random.choice(self.cnn_count, self.cnn_batch_size, replace=False)

        # Set the current expert minibatch and mask
        old_expert_minibatch = self.current_expert_minibatch
        self.current_expert_minibatch = np.concatenate((mlp_model_indices, cnn_model_indices))
        self.current_expert_mask = np.zeros(len(self.experts), dtype=bool)
        self.current_expert_mask[self.current_expert_minibatch] = True

        if np.array_equal(self.current_expert_minibatch, old_expert_minibatch):
            print(f"New expert minibatch: {self.current_expert_minibatch}")
            return True
        return False

    def expert_block(self, mlp_obs, cnn_obs, voting: bool = True):
        """
        Processes observations through all experts and applies the current expert mask.
        """
        # Get MLP outputs
        mlp_outputs = [model(mlp_obs) for model in self.mlps]  # List of [batch_size, act_dim]

        # Get CNN outputs
        cnn_outputs = [model(cnn_obs) for model in self.cnns]  # List of [batch_size, act_dim]

        # Combine all expert outputs
        all_expert_outputs = mlp_outputs + cnn_outputs  # Total experts = num_mlps + num_cnns

        # Stack experts: [batch_size, num_experts, act_dim]
        stacked_experts = tf.stack(all_expert_outputs, axis=1)  # Shape: [batch_size, num_experts, act_dim]

        # Determine batch size dynamically
        batch_size = tf.shape(stacked_experts)[0]

        # Prepare mask
        # self.current_expert_mask: [num_experts, 1]
        mask = tf.cast(self.current_expert_mask, dtype=tf.bool)  # Shape: [num_experts, 1]
        mask = tf.reshape(mask, [1, -1, 1])  # Shape: [1, num_experts, 1]
        mask = tf.tile(mask, [batch_size, 1, 1])  # Shape: [batch_size, num_experts, 1]

        # Broadcast mask to [batch_size, num_experts, act_dim]
        mask = tf.broadcast_to(mask, [batch_size, self.expert_count, self.act_dim])  # Shape: [batch_size, num_experts, act_dim]

        # Apply mask: Zero out predictions of inactive experts
        masked_experts = tf.where(mask, stacked_experts, 1e-6 * tf.ones_like(stacked_experts))  # Shape: [batch_size, num_experts, act_dim]

        # Aggregate expert outputs through voting method
        if voting:
            vote_dist = self.expert_vote(masked_experts)  # Shape: [batch_size, num_experts * act_dim]
            # Flatten masked experts back to [batch_size, num_experts * act_dim]
            masked_preds = tf.reshape(vote_dist, [batch_size, -1])  # Shape: [batch_size, num_experts * act_dim]
        else:
            masked_preds = tf.reshape(masked_experts, [batch_size, -1])  # Shape: [batch_size, num_experts * act_dim]

        return masked_preds

    def expert_vote(self, masked_predictions: tf.Tensor):
        """
        Performs a voting operation on the expert predictions.
        """

        expert_votes = aggregate_expert_outputs(masked_predictions, method='average')  # Shape: [batch_size, act_dim]

        return expert_votes

    def stacking(self, expert_vote_distribution: tf.Tensor, original_obs: tf.Tensor):
        """
        Defines the meta learner neural network that aggregates the expert probability outputs.

        Args:
            expert_vote_distribution (tf.Tensor): The action vote distributions concatenated from the expert block.
            original_obs (tf.Tensor): The original observation tensor.

        Returns:
            tf.Tensor: The output tensor from the meta learner.
        """
        # Concatenate the action vote distributions with the original observations
        votes_and_obs: tf.Tensor = tf.concat([expert_vote_distribution, original_obs], axis=-1)
        encoder_qs = self.stack_encoder(votes_and_obs)

        return encoder_qs

    def forward(self, mlp_obs: tf.Tensor, cnn_obs: tf.Tensor, voting: bool = True):
        """
        Forward pass through the ensemble.

        Args:
            X (tf.Tensor): Input tensor.
            training (bool): Flag indicating if the model is in training mode.

        Returns:
            tuple: (action, probs, selected_model_indices)
        """
        assert self.current_expert_minibatch is not None, "Model minibatch not set."
        assert self.current_expert_mask is not None, "Model mask not set."

        expert_vote = self.expert_block(mlp_obs, cnn_obs, voting=voting)
        flattened_expert_preds = tf.reshape(expert_vote, [mlp_obs.shape[0], -1])

        # Preprocess original observations
        # Flatten CNN observations if necessary
        if cnn_obs is not None:
            flattened_cnn_obs = tf.reshape(cnn_obs, [cnn_obs.shape[0], -1])
            original_obs = tf.concat([mlp_obs, flattened_cnn_obs], axis=1)  # Shape: [batch_size, mlp_input_dim + flattened_cnn_obs_dim]
        else:
            original_obs = mlp_obs  # Shape: [batch_size, mlp_input_dim]

        original_obs = tf.cast(original_obs, tf.float32)

        q_vals = self.stacking(expert_vote, original_obs)

        return q_vals, flattened_expert_preds

    def get_config(self):
        """
        Returns the config of the model.

        This is used for serialization.
        """
        config = super(ExpertStackingEncoder, self).get_config()
        config.update({
            'mlp_activations': self.mlp_activations,
            'cnn_activations': self.cnn_activations,
            'mlp_input_dim': self.mlp_input_dim,
            'cnn_input_shape': self.cnn_input_shape,
            'act_dim': self.act_dim,
            'initial_seed': self.initial_seed,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'mlp_count': self.mlp_count,
            'cnn_count': self.cnn_count,
            'mlp_batch_size': self.mlp_batch_size,
            'cnn_batch_size': self.cnn_batch_size,
            'grad_dir': self.grad_dir,
        })
        return config


# class PerformanceWeights:
#     """
#     Manages performance weights for an ensemble of models.
#     """
#     def __init__(self, models: list, grad_dir: str):
#         """
#         Initialize PerformanceWeights.
#
#         Args:
#             models (list[keras.Model]): List of models in the ensemble.
#             grad_dir (str): Directory to save/load gradients.
#         """
#         super().__init__()
#         assert grad_dir is not None, "grad_dir must be specified."
#         self.models = models
#         self.pmf = np.ones(len(models)) / len(models)
#         self.grad_dir = os.path.join(grad_dir, 'session' + str(os.getpid() * 10))
#         if not os.path.exists(self.grad_dir):
#             os.makedirs(self.grad_dir)
#
#     def save_parameters(self, model_index: int, filename: str):
#         """
#         Save gradients (weights) of a specific model.
#
#         Args:
#             model_index (int): Index of the model in the ensemble.
#             filename (str): Filename to save the gradients.
#         """
#         model = self.models[model_index]
#         new_path = os.path.join(self.grad_dir, filename + '.h5')
#         model.save_weights(new_path)
#         print(f"Saved weights for model {model_index} to {new_path}.")
#
#     def load_parameters(self, model_index: int, filename: str):
#         """
#         Load gradients (weights) into a specific model.
#
#         Args:
#             model_index (int): Index of the model in the ensemble.
#             filename (str): Filename from which to load the gradients.
#         """
#         model = self.models[model_index]
#         new_path = os.path.join(self.grad_dir, filename + '.h5')
#         model.load_weights(new_path)
#         print(f"Loaded weights for model {model_index} from {new_path}.")
#
#     def calculate_performance_weights(self, performance_scores: np.ndarray):
#         """
#         Calculate performance weights based on performance scores.
#
#         Args:
#             performance_scores (np.ndarray): Array of performance scores for each model.
#         """
#         total = np.sum(performance_scores)
#         if total > 0:
#             self.pmf = np.array(performance_scores) / total
#         else:
#             self.pmf = np.ones(len(self.models)) / len(self.models)
