import json
import os

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
