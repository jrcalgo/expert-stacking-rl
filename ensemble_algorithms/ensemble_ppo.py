import torch
import numpy as np

from network_ensemble import MiniArchitectureEnsemble


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)



class EnsemblePPO:
    def __init__(self):
        
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train_one_epoch(self):
        