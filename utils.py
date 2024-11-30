import json


def load_hyperparams(algorithm: str):
    """
    Load hyperparameters from hyperparams.json.

    Returns:
        dict: Dictionary of hyperparameters.
    """
    algo = algorithm.upper()

    ensemble_params = None
    algorithm_params = None
    with open('hyperparams.json') as f:
        hyperparams = json.load(f)
        ensemble_params = hyperparams['MiniArchEnsemble']
        algorithm_params = hyperparams[algo]

    return ensemble_params, algorithm_params


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
