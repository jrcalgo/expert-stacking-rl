import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


def miniMLP(input_dim: int, output_dim: int, seed: int):
    """
    Defines a simple Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output.
        seed (int): Random seed for reproducibility.

    Returns:
        keras.Model: The MLP model.
    """
    tf.random.set_seed(seed)
    model = keras.Sequential([
        keras.layers.Dense(input_dim, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(output_dim)
    ])
    return model


def miniCNN(input_shape: tuple, output_dim: int, kernel_size: int, stride: int, seed: int, padding='same'):
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
    model = keras.Sequential([
        keras.layers.Conv2D(36, kernel_size, strides=stride, padding=padding, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=kernel_size, strides=stride),
        keras.layers.Conv2D(24, kernel_size, strides=stride, padding=padding, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=kernel_size, strides=stride),
        keras.layers.Flatten(),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(output_dim)
    ])
    return model


class PerformanceWeights:
    """
    Manages performance weights for an ensemble of models.
    """
    def __init__(self, models: list, grad_dir: str):
        """
        Initialize PerformanceWeights.

        Args:
            models (list[keras.Model]): List of models in the ensemble.
            grad_dir (str): Directory to save/load gradients.
        """
        super().__init__()
        assert grad_dir is not None, "grad_dir must be specified."
        self.models = models
        self.pmf = np.ones(len(models)) / len(models)
        self.grad_dir = os.path.join(grad_dir, 'session' + str(os.getpid() * 10))
        if not os.path.exists(self.grad_dir):
            os.makedirs(self.grad_dir)

    def save_gradients(self, model_index: int, filename: str):
        """
        Save gradients (weights) of a specific model.

        Args:
            model_index (int): Index of the model in the ensemble.
            filename (str): Filename to save the gradients.
        """
        model = self.models[model_index]
        new_path = os.path.join(self.grad_dir, filename + '.h5')
        model.save_weights(new_path)
        print(f"Saved weights for model {model_index} to {new_path}.")

    def load_gradients(self, model_index: int, filename: str):
        """
        Load gradients (weights) into a specific model.

        Args:
            model_index (int): Index of the model in the ensemble.
            filename (str): Filename from which to load the gradients.
        """
        model = self.models[model_index]
        new_path = os.path.join(self.grad_dir, filename + '.h5')
        model.load_weights(new_path)
        print(f"Loaded weights for model {model_index} from {new_path}.")

    def calculate_performance_weights(self, performance_scores: np.ndarray):
        """
        Calculate performance weights based on performance scores.

        Args:
            performance_scores (np.ndarray): Array of performance scores for each model.
        """
        total = np.sum(performance_scores)
        if total > 0:
            self.pmf = np.array(performance_scores) / total
        else:
            self.pmf = np.ones(len(self.models)) / len(self.models)


class MiniArchitectureEnsemble(tf.Module):
    """
    An ensemble of miniMLP and miniCNN models with performance-based weighting.
    """
    def __init__(self,
                 mlp_input_dim,
                 act_dim,
                 input_shape,
                 initial_seed=0,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 mlp_count=12,
                 cnn_count=6,
                 mlp_batch_size=6,
                 cnn_batch_size=6,
                 grad_dir="./runtime_models"):
        super(MiniArchitectureEnsemble, self).__init__()
        model_count = mlp_count + cnn_count
        assert model_count % 2 == 0 and model_count % 3 == 0, "Model count must be divisible by 2 and 3."
        assert mlp_batch_size <= mlp_count, "MLP batch size cannot exceed the number of MLP models."
        assert cnn_batch_size <= cnn_count, "CNN batch size cannot exceed the number of CNN models."

        self.mlp_count = mlp_count
        self.cnn_count = cnn_count
        self.mlp_batch_size = mlp_batch_size
        self.cnn_batch_size = cnn_batch_size

        mlp_seeds = int(initial_seed + np.random.RandomState(initial_seed).randint(0, 2**32, mlp_count))
        self.mlps = [miniMLP(mlp_input_dim, act_dim, mlp_seeds[i]) for i in range(self.mlp_count)]
        cnn_seeds = int(initial_seed + np.random.RandomState(initial_seed).randint(0, 2**32, cnn_count))
        self.cnns = [miniCNN(input_shape, act_dim, kernel_size, stride, cnn_seeds[i], padding) for i in range(self.cnn_count)]
        self.models = self.mlps + self.cnns

        self.performance_weights = PerformanceWeights(self.models, grad_dir)

    def _ensemble_filter(self, predictions: dict) -> tf.Tensor:
        """
        Construct probability distribution from model predictions.

        Args:
            predictions (dict[int, tf.Tensor]): Dict of prediction tensors from models.

        Returns:
            tf.Tensor: Probability distributions over actions from all models.
        """
        probs_list = []
        for pred in predictions.values():
            probs = tf.nn.softmax(pred, axis=-1)
            probs_list.append(probs)
        probs_tensor = tf.stack(probs_list, axis=0)  # Shape: (num_models, batch_size, num_actions)
        return probs_tensor

    def call_ensemble(self, X, training=False):
        """
        Forward pass through the ensemble.

        Args:
            X (tf.Tensor): Input tensor.
            training (bool): Flag indicating if the model is in training mode.

        Returns:
            tuple: (action, probs, selected_model_indices)
        """
        # Select random indices for MLP and CNN models
        mlp_indices = np.random.choice(self.mlp_count, self.mlp_batch_size, replace=False)
        cnn_indices = self.mlp_count + np.random.choice(self.cnn_count, self.cnn_batch_size, replace=False)
        minibatch_indices = np.concatenate([mlp_indices, cnn_indices])

        # Obtain predictions from the selected models
        preds = {}
        for i in minibatch_indices:
            model = self.models[i]
            preds[i] = model(X, training=training)

        # Compute probabilities using the ensemble filter
        ensemble_probs = self._ensemble_filter(preds)  # Shape: (num_models, batch_size, num_actions)

        # Compute the average probabilities across the models
        avg_probs = tf.reduce_mean(ensemble_probs, axis=0)  # Shape: (batch_size, num_actions)

        # Select the action with the highest average probability for each sample in the batch
        action = tf.argmax(avg_probs, axis=-1).numpy()

        return action, ensemble_probs, preds, minibatch_indices
