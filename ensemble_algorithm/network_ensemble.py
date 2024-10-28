import os
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
    )


# 
def miniCNN(input_dim, output_dim, kernel_size, stride, padding=1, dropout=.33):
    return nn.Sequential(
        nn.Conv2d(input_dim, 36, kernel_size, stride, padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=kernel_size, stride=stride),

        nn.Conv2d(36, 24, kernel_size, stride, padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=kernel_size, stride=stride),

        nn.Flatten(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, output_dim),
    )


class PerformanceWeights:
    def __init__(self, models: list[nn.Sequential], grad_dir):
        super().__init__()
        self.models_dict = [{model: 0} for model in models]
        self.pmf = np.ones(len(models)) / len(models)

        self.grad_dir = grad_dir + '/session' + str(os.getpid()*10) if grad_dir

    def save_gradients(self, model_index: int, filename: str):
        new_path = os.path.join(self.grad_dir, filename + ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self.models_dict[model_index], new_path)

    def load_gradients(self, model_index: int, filename: str):
        new_path = os.path.join(self.grad_dir, filename + ('.pth' if not filename.__contains__('.pth') else ''))
        self.models_dict[model_index] = self.models_dict[model_index].load_state_dict(torch.load(new_path, weights_only=True))

    def calculate_performance_weights(self, performance_scores):
        total = sum(performance_scores)
        self.pmf = np.array(performance_scores) / total


class MiniArchitectureEnsemble(nn.Module):
    def __init__(self, 
                 mlp_input_dim, 
                 cnn_input_dim,
                 act_dim,
                 device: str = "cpu",
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 mlp_count: int = 12,
                 cnn_count: int = 6,
                 mlp_dropout: float = .33, 
                 cnn_dropout: float = .33,
                 mlp_batch_size: int = 6,
                 cnn_batch_size: int = 3,
                 grad_dir: str = "./runtime_models"):
        
        super(MiniArchitectureEnsemble, self).__init__()
        model_count = mlp_count + cnn_count
        assert model_count % 2 == 0 and model_count % 3 == 0
        assert mlp_batch_size <= model_count
        assert cnn_batch_size <= model_count
        self.mlp_count: int = mlp_count
        self.cnn_count: int = cnn_count
        self.mlp_batch_size: int = mlp_batch_size
        self.cnn_batch_size: int = cnn_batch_size

        self.mlps = [miniMLP(mlp_input_dim, act_dim, mlp_dropout).to(device) for _ in range(self.mlp_count)]
        self.cnns = [miniCNN(cnn_input_dim, act_dim, kernel_size, stride, padding, cnn_dropout).to(device) for _ in range(self.cnn_count)]
        self.models = self.mlps + self.cnns

        self.performance_weights = PerformanceWeights(self.models, grad_dir)

    def _ensemble_filter(self, predictions):
        # probability mass distribution based on model performance.
        # if voted with majority, model ~+1. If not, +0
        # convert new pmf to pdf (nn.Softmax). I suppose the pmf will be nn.Categorical...
        probs = torch.stack([nn.functional.softmax(pred, dim=-1) for pred in predictions])

        performance_probs =
        return probs, performance_probs
    
    def forward(self, X) -> tuple:
        minibatch_indices = np.random.choice(len(self.models), self.model_batch_size, replace=False)
        preds = [self.models[i](X) for i in minibatch_indices]
        probs, performance_probs = self._ensemble_filter(preds)
        action = torch.argmax(probs)

        return action, probs, performance_probs, minibatch_indices

        