# training_factory.py

from ensemble_algorithms.ensemble_dqn import EnsembleDQN
from ensemble_algorithms.ensemble_ppo import EnsemblePPO
from gym_environments import create_env


class training_factory:
    def __init__(self, algorithm: str, train_iters: int, epochs: int, seed:int, env_name: str, torch_model):
        """
        Initialize the training factory.

        Args:
            algorithm (str): 'dqn' or 'ppo'.
            epochs (int): Number of epochs to train.
            env_name (str): Name of the Gym environment.
            torch_model: The model ensemble to train.
        """
        self.algorithm = algorithm.upper()
        self.train_iters = train_iters
        self.epochs = epochs
        self.env_name = env_name
        self.torch_model = torch_model

        # Initialize the environment
        self.env = create_env(env_name)
        self.env.seed(seed)  # For reproducibility

        # Initialize the agent based on the algorithm
        if self.algorithm == 'DQN':
            self.agent = EnsembleDQN(self.env.env)
        elif self.algorithm == 'PPO':
            self.agent = EnsemblePPO(self.torch_model, self.env.env)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def train(self):
        """
        Train the model ensemble using the specified algorithm.
        """
        for _ in self.train_iters:
            self.agent.train_n_epochs(self.epochs)

        # Save the trained models
        self.agent.save_model()
        print("Training completed and models saved.")
