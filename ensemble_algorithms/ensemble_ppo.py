import torch
import numpy as np

from nn_ensemble import MiniArchitectureEnsemble


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)



class EnsemblePPO:
    def __init__(self):
        
        super().__init__()

    def train_one_epoch(self):
        