import torch
import torch.nn as nn
import numpy as np

def miniMLP(input_dim, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, output_dim),
        nn.Dropout(dropout)
    )

def miniCNN(input_dim, output_dim, kernel_size, stride, padding, dropout):
    return nn.Sequential(
        nn.Conv2d(input_dim, ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        nn.Conv2d(),
        nn.ReLU(),
        nn.MaxPool2d(),

        nn.Flatten(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, output_dim)
    )


class PerformanceWeights:
    def __init__(self, models: np.ndarray[nn.Sequential]):
        self.models = models

    def calculate_performance_weights(self):
    
    def save_gradients(self):
    
    def load_gradients(self):



class EnsembledNNs(nn.Module):
    def __init__(self, mlp_input_dim, mlp_output_dim, conv_input_dim, conv_output_dim, device: str = "cpu", model_count: int = 20, mlp_dropout: float = .33, conv_dropout: float = .33, model_batch_size: int = 10, epsilon):
        assert model_count % 2 == 0
        assert model_batch_size <= model_count
        self.mlp_count = model_count/2
        self.conv_count = model_count/2
        self.model_batch_size = model_batch_size

        self.mlps = np.array([miniMLP(mlp_input_dim, mlp_output_dim, mlp_dropout).to(device) for _ in range(self.mlp_count)])
        self.convs = np.array([miniCNN(conv_input_dim, conv_output_dim, conv_dropout).to(device) for _ in range(self.conv_count)])
        self.models = np.concatenate(self.mlps + self.convs, dtype=nn.Sequential)

        self.performance_weights = PerformanceWeights(self.models)

    def _ensemble_filter(self, predictions):


        return pred
    
    def forward(self, X):
        minibatch_indices = np.random.randint(0, self.mlp_count+self.conv_count, size=self.model_batch_size)
        minibatch = self.models[minibatch_indices]

        preds: list = []
        for i in minibatch:
            preds.append(self.minibatch[i](X))

        return _ensemble_filter(preds), minibatch_indices

    def step(self, X):

        return action
        