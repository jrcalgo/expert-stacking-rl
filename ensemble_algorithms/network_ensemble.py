import keras
import 
import numpy as np

import os


def miniMLP(input_dim, output_dim, dropout):
    """
    Defines a simple Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output.
        dropout (float, optional): Dropout rate. Defaults to 0.33.

    Returns:
        nn.Sequential: The MLP model.
    """

    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, output_dim),
    )


def miniCNN(input_dim, output_dim, kernel_size, stride, padding=1, dropout=.33):
    """
    Defines a simple Convolutional Neural Network (CNN).

    Args:
        input_channels (int): Number of input channels.
        output_dim (int): Dimension of output.
        kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
        stride (int, optional): Stride for convolutions. Defaults to 1.
        padding (int, optional): Padding for convolutions. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.33.

    Returns:
        nn.Sequential: The CNN model.
    """

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
    """
    Manages performance weights for an ensemble of models.
    """
    def __init__(self, models: list[nn.Sequential], grad_dir):
        """
        Initialize PerformanceWeights.

        Args:
            models (List[nn.Module]): List of models in the ensemble.
            grad_dir (str): Directory to save/load gradients.
        """

        super().__init__()
        assert grad_dir is not None, "grad_dir must be specified."
        self.models_dict = [{model: 0} for model in models]
        self.pmf = np.zeros(len(models)) / len(models)

        self.grad_dir = grad_dir + '/session' + str(os.getpid()*10)

    def save_gradients(self, model_index: int, filename: str):
        """
        Save gradients of a specific model.

        Args:
            model_index (int): Index of the model in the ensemble.
            filename (str): Filename to save the gradients.
        """

        model = self.models[model_index]
        new_path = os.path.join(self.grad_dir, filename if filename.endswith('.pth') else filename + '.pth')
        torch.save(model.state_dict(), new_path)
        print(f"Saved gradients for model {model_index} to {new_path}.")

    def load_gradients(self, model_index: int, filename: str):
        """
        Load gradients into a specific model.

        Args:
            model_index (int): Index of the model in the ensemble.
            filename (str): Filename from which to load the gradients.
        """

        model = self.models[model_index]
        new_path = os.path.join(self.grad_dir, filename if filename.endswith('.pth') else filename + '.pth')
        state_dict = torch.load(new_path, map_location='cpu')  # Adjust map_location as needed
        model.load_state_dict(state_dict)
        print(f"Loaded gradients for model {model_index} from {new_path}.")

    def calculate_performance_weights(self, performance_scores: np.ndarray):
        """
        Calculate performance weights based on performance scores.

        Args:
            performance_scores (List[float]): List of performance scores for each model.
        """

        total = sum(performance_scores)
        if total > 0:
            self.pmf = np.array(performance_scores) / total
        else:
            self.pmf = np.ones(len(self.models))/ len(self.models)


class MiniArchitectureEnsemble(nn.Module):
    """
    An ensemble of MiniMLP and MiniCNN models with performance-based weighting.
    """
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
                 cnn_batch_size: int = 6,
                 grad_dir: str = "./runtime_models"):
        
        super(MiniArchitectureEnsemble, self).__init__()
        model_count = mlp_count + cnn_count
        assert model_count % 2 == 0 and model_count % 3 == 0, "Model count must be divisible by 2 and 3."
        assert mlp_batch_size <= model_count, "MLP batch size cannot exceed the number of MLP models."
        assert cnn_batch_size <= model_count, "CNN batch size cannot exceed the number of CNN models."

        self.mlp_count: int = mlp_count
        self.cnn_count: int = cnn_count
        self.mlp_batch_size: int = mlp_batch_size
        self.cnn_batch_size: int = cnn_batch_size
        self.device = device

        self.mlps = [miniMLP(mlp_input_dim, act_dim, mlp_dropout).to(device) for _ in range(self.mlp_count)]
        self.cnns = [miniCNN(cnn_input_dim, act_dim, kernel_size, stride, padding, cnn_dropout).to(device) for _ in range(self.cnn_count)]
        self.models = self.mlps + self.cnns

        self.performance_weights = PerformanceWeights(self.models, grad_dir)

    def _ensemble_filter(self, predictions):
        """
        Apply softmax to each model's predictions and stack them.

        Args:
            predictions (List[torch.Tensor]): List of prediction tensors from models.

        Returns:
            torch.Tensor: Stacked probability tensors.
        """
        
        probs = torch.stack([nn.functional.softmax(pred, dim=-1) for pred in predictions])

        return probs
    
    def forward(self, X: torch.Tensor, select_new_models: bool = False) -> tuple:
        """
        Forward pass through the ensemble.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            tuple: (action, probs, selected_model_indices)
        """

        mlp_indices = np.random.choice(self.mlp_count, self.mlp_batch_size, replace=False)
        cnn_indices = self.mlp_count + np.random.choice(self.cnn_count, self.cnn_batch_size, replace=False)
        minibatch_indices = np.concatenate([mlp_indices, cnn_indices])

        preds = [self.models[i](X) for i in minibatch_indices]
        probs = self._ensemble_filter(preds)
        avg_probs = probs.mean(dim=0)
        action = torch.argmax(avg_probs).item()

        return action, probs, minibatch_indices
        