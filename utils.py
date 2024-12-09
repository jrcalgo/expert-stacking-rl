import json
import tensorflow as tf
import os
import datetime
import tensorboard

import numpy as np


def load_hyperparams(algorithm: str):
    """
    Load hyperparameters from hyperparams.json.

    Returns:
        dict: Dictionary of hyperparameters.
    """
    algo = algorithm.upper()

    current_dir = os.path.dirname(__file__)
    hyperparams_path = os.path.join(current_dir, 'hyperparams.json')

    ensemble_params = None
    algorithm_params = None
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
        ensemble_params = hyperparams['ExpertStackingEncoder']
        algorithm_params = hyperparams[algo]

    return ensemble_params, algorithm_params


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class TensorBoardLogger:
    def __init__(self, log_dir=None):
        """
        Initializes the TensorBoard logger.

        Args:
            log_dir (str, optional): Directory where to save the logs. If None, defaults to './logs/{current_time}'.
        """
        if log_dir is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join("logs", current_time)
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")

    def log_metrics(self, epoch, rewards, expert_loss, encoder_loss):
        """
        Logs the specified metrics to TensorBoard.

        Args:
            epoch (int): Current epoch number.
            rewards (float): Total rewards for the epoch.
            expert_loss (float): Expert loss for the epoch.
            encoder_loss (float): Encoder loss for the epoch.
        """
        with self.writer.as_default():
            tf.summary.scalar('Rewards per Epoch', rewards, step=epoch)
            tf.summary.scalar('Expert Loss per Epoch', expert_loss, step=epoch)
            tf.summary.scalar('Encoder Loss per Epoch', encoder_loss, step=epoch)
            self.writer.flush()

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()
