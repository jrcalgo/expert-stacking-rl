# training_factory.py

import gymnasium as gym
import numpy as np
from typing import Any, List, Tuple

from ensemble_algorithms.ensemble_dqn import EnsembleDQN


class TrainingFactory:
    """
    Factory class to initialize and train reinforcement learning agents using EnsembleDQN, EnsembleDDQN, or EnsemblePPO.
    """

    def __init__(
            self,
            algorithm_name: str,
            env_name: str,
            env_seed: int,
            train_iters: int,
            max_epochs: int,
            replay_buffer_size: int,
            mlp_activations: List[List[str]] = None,
            cnn_activations: List[List[str]] = None,
            hyperparams: dict = None
    ):
        """
        Initialize the TrainingFactory.

        Args:
            algorithm_name (str): 'DQN', 'DDQN', or 'PPO'.
            env_name (str): Name of the Gym environment.
            env_seed (int): Random seed for reproducibility.
            train_iters (int): Number of training iterations.
            max_epochs (int): Number of epochs per training iteration.
            mlp_activations (List[List[str]]): Activation functions for each MLP model.
            cnn_activations (List[List[str]]): Activation functions for each CNN model.
            replay_buffer_size (int): Size of the replay buffer.
            hyperparams (dict): Dictionary containing hyperparameters.
        """
        self.algorithm = algorithm_name.upper()
        self.train_iters = train_iters
        self.max_epochs = max_epochs
        self.env_name = env_name
        self.seed = env_seed
        self.replay_buffer_size = replay_buffer_size
        self.hyperparams = hyperparams

        if hyperparams['mlp_count'] != 0:
            assert mlp_activations is not None, "MLP activations must be provided."
        if hyperparams['cnn_count'] != 0:
            assert cnn_activations is not None, "CNN activations must be provided."

        # Initialize the environment
        self.env = gym.make(env_name, render_mode="human")

        # Extract observation and action space dimensions
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        # Determine if observation space is discrete or continuous
        if isinstance(obs_space, gym.spaces.Box):
            # Continuous observation space (e.g., images or vectors)
            if len(obs_space.shape) == 1:
                # Vector observations for MLP
                mlp_input_dim = obs_space.shape[0]
                mlp_input_size = obs_space.shape[0]
                # For CNN, reshape vector into a single-channel image (e.g., 1 x mlp_input_size)
                cnn_input_dim = (1, mlp_input_size, 1)  # Example reshape
                cnn_input_size = cnn_input_dim
            else:
                # Image observations for CNN
                mlp_input_dim = np.prod(obs_space.shape)  # Flattened image for MLP
                mlp_input_size = mlp_input_dim
                cnn_input_dim = obs_space.shape
                cnn_input_size = cnn_input_dim
        elif isinstance(obs_space, gym.spaces.Discrete):
            # Discrete observation space (e.g., for environments like FrozenLake)
            mlp_input_dim = 1  # One-hot encoding can be used or scalar representation
            mlp_input_size = 1
            # For CNN, reshape scalar into a single-channel image (e.g., 1 x 1 x 1)
            cnn_input_dim = (1, 1, 1)
            cnn_input_size = cnn_input_dim
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(obs_space)}")

        # Determine action space dimensions
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")

        # Initialize the agent based on the algorithm
        if self.algorithm == 'DQN':
            self.learner = EnsembleDQN(
                env=self.env,
                mlp_input_size=mlp_input_size,
                mlp_input_dim=mlp_input_dim,
                cnn_input_size=cnn_input_size,
                cnn_input_dim=cnn_input_dim,
                mlp_activations=mlp_activations,
                cnn_activations=cnn_activations,
                action_dim=action_dim,
                buffer_size=self.replay_buffer_size,
                mlp_count=self.hyperparams['mlp_count'] if 'mlp_count' in self.hyperparams else 3,
                cnn_count=self.hyperparams['cnn_count'] if 'cnn_count' in self.hyperparams else 3,
                mlp_batch_size=self.hyperparams['mlp_batch_size'] if 'mlp_batch_size' in self.hyperparams else 1,
                cnn_batch_size=self.hyperparams['cnn_batch_size'] if 'cnn_batch_size' in self.hyperparams else 1,
                expert_rotation_freq=self.hyperparams['expert_rotation_freq'] if 'expert_rotation_freq' in self.hyperparams else 16,
            )
        elif self.algorithm == 'DDQN':
            self.learner = EnsembleDDQN(
                env=self.env,
                mlp_input_size=mlp_input_size,
                mlp_input_dim=mlp_input_dim,
                cnn_input_size=cnn_input_size,
                cnn_input_dim=cnn_input_dim,
                mlp_activations=mlp_activations,
                cnn_activations=cnn_activations,
                action_dim=action_dim,
                buffer_size=self.replay_buffer_size,
                mlp_count=self.hyperparams['mlp_count'] if 'mlp_count' in self.hyperparams else 3,
                cnn_count=self.hyperparams['cnn_count'] if 'cnn_count' in self.hyperparams else 3,
                mlp_batch_size=self.hyperparams['mlp_batch_size'] if 'mlp_batch_size' in self.hyperparams else 1,
                cnn_batch_size=self.hyperparams['cnn_batch_size'] if 'cnn_batch_size' in self.hyperparams else 1,
                expert_rotation_freq=self.hyperparams['expert_rotation_freq'] if 'expert_rotation_freq' in self.hyperparams else 16,
            )
        elif self.algorithm == 'PPO':
            self.learner = EnsemblePPO(
                env=self.env,
                mlp_input_size=mlp_input_size,
                mlp_input_dim=mlp_input_dim,
                cnn_input_size=cnn_input_size,
                cnn_input_dim=cnn_input_dim,
                mlp_activations=mlp_activations,
                cnn_activations=cnn_activations,
                action_dim=action_dim,
                buffer_size=self.replay_buffer_size,
                mlp_count=self.hyperparams['mlp_count'] if 'mlp_count' in self.hyperparams else 3,
                cnn_count=self.hyperparams['cnn_count'] if 'cnn_count' in self.hyperparams else 3,
                mlp_batch_size=self.hyperparams['mlp_batch_size'] if 'mlp_batch_size' in self.hyperparams else 1,
                cnn_batch_size=self.hyperparams['cnn_batch_size'] if 'cnn_batch_size' in self.hyperparams else 1,
                expert_rotation_freq=self.hyperparams['expert_rotation_freq'] if 'expert_rotation_freq' in self.hyperparams else 16,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        self.epoch = 0

    def simulate_max_epochs(self):
        """
        Train the agent for the specified number of training iterations and epochs.
        """
        for iter_num in range(1, self.train_iters + 1):
            print(f"\n--- Training Iteration {iter_num}/{self.train_iters} ---")
            for epoch in range(1, self.max_epochs + 1):
                self.epoch += 1
                print(f"Training Epoch {epoch}/{self.max_epochs} (Global Epoch: {self.epoch})")
                # Run one epoch of training
                self.run_training_epoch()
            print(f"Completed Training Iteration {iter_num}/{self.train_iters}")

        # Save the trained models
        save_path = "./trained_models"
        self.learner.save_model(save_path)
        print("Training completed and models saved.")

    def run_training_epoch(self):
        """
        Run a single epoch of training by interacting with the environment.
        """
        # Initialize episode
        obs, info = self.env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Preprocess observations for MLP and CNN
            mlp_obs, cnn_obs = self._preprocess_observation(obs)

            # Select action using the agent's policy
            action = self.learner.step(mlp_obs, cnn_obs)

            # Take action in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            self.learner.store_transition(
                mlp_obs=mlp_obs,
                cnn_obs=cnn_obs,
                act=action,
                rew=reward,
                mlp_next_obs=self._preprocess_mlp(next_obs) if mlp_obs is not None else None,
                cnn_next_obs=self._preprocess_cnn(next_obs) if cnn_obs is not None else None,
                done=done
            )

            total_reward += reward
            step_count += 1
            obs = next_obs

            # Train models according to training frequency
            if self.learner.traj > 0 and self.learner.traj % self.learner._traj_per_epoch == 0:
                if self.learner._batch_size <= self.learner.replay_buffer.size:
                    self.learner.train_models(epoch_num=self.epoch)

        print(f"Episode completed: Total Reward = {total_reward}, Epoch Steps = {step_count}")

    def _preprocess_observation(self, obs: Any) -> Tuple[np.ndarray, Any]:
        """
        Preprocess observations for MLP and CNN inputs.

        Args:
            obs (Any): Raw observation from the environment.

        Returns:
            Tuple[np.ndarray, Any]: Tuple containing preprocessed MLP and CNN observations.
        """
        # Depending on the observation space, preprocess accordingly
        # For vector observations, CNN input might be None or a reshaped version

        mlp_obs, cnn_obs = None, None

        # Extract MLP observation
        if self.hyperparams['mlp_count'] != 0:
            if self.learner._mlp_input_dim is not None:
                mlp_obs = self._preprocess_mlp(obs)
            else:
                mlp_obs = None

        # Extract CNN observation
        if self.hyperparams['cnn_count'] != 0:
            if self.learner._cnn_input_dim is not None:
                cnn_obs = self._preprocess_cnn(obs)
            else:
                cnn_obs = None

        return mlp_obs, cnn_obs

    def _preprocess_mlp(self, obs: Any) -> np.ndarray:
        """
        Preprocess the observation for MLP input.

        Args:
            obs (Any): Raw observation.

        Returns:
            np.ndarray: Preprocessed observation with batch dimension.
        """
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                # Add batch dimension
                return np.expand_dims(obs, axis=0)  # Shape: (1, 8)
            elif obs.ndim == 2:
                # Already has batch dimension
                return obs
            else:
                raise ValueError(f"mlp_obs must have 1 or 2 dimensions, got {obs.ndim}")
        elif isinstance(obs, (int, float)):
            # Convert scalar to array with batch dimension
            return np.array([obs], dtype=np.float32).reshape(1, -1)  # Shape: (1, 1)
        else:
            raise ValueError(f"Unsupported observation type for MLP: {type(obs)}")

    def _preprocess_cnn(self, obs: Any) -> np.ndarray:
        """
        Preprocess the observation for CNN input.

        Args:
            obs (Any): Raw observation.

        Returns:
            np.ndarray: Preprocessed observation for CNN expert.
        """
        if isinstance(obs, np.ndarray):
            # Normalize if necessary and reshape to match CNN input dimensions
            obs = obs.astype(np.float32) / 255.0  # Example normalization
            # Reshape based on expected CNN input size
            expected_shape = self.learner._cnn_input_dim
            if len(expected_shape) == 3:
                obs = np.reshape(obs, expected_shape)
            elif len(expected_shape) == 1:
                obs = obs.reshape(-1, *expected_shape)
            else:
                raise ValueError(f"Unsupported CNN input shape: {expected_shape}")
            return obs
        else:
            raise ValueError(f"Unsupported observation type for CNN: {type(obs)}")
