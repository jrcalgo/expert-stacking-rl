import gymnasium as gym
import numpy as np
from typing import Any, Tuple

from baseline_algorithms.DQN import DQN
from baseline_algorithms.REINFORCE import REINFORCE


class TrainingFactory:
    """
    Factory class to initialize and train a reinforcement learning agent using SimpleDQN.
    """

    def __init__(
            self,
            algorithm_name: str,
            env_name: str,
            env_seed: int,
            train_iters: int,
            max_epochs: int,
            replay_buffer_size: int,
            tensorboard_logging: bool = False
    ):
        """
        Initialize the TrainingFactory.

        Args:
            env_name (str): Name of the Gym environment.
            env_seed (int): Random seed for reproducibility.
            train_iters (int): Number of training iterations.
            max_epochs (int): Number of epochs per training iteration.
            replay_buffer_size (int): Size of the replay buffer.
            hyperparams (dict): Dictionary containing hyperparameters.
        """
        self.algorithm_name = algorithm_name.upper()
        self.train_iters = train_iters
        self.max_epochs = max_epochs
        self.env_name = env_name
        self.seed = env_seed
        self.replay_buffer_size = replay_buffer_size

        # Initialize the environment
        self.env = gym.make(env_name)
        self.env.reset(seed=self.seed)

        # Extract observation and action space dimensions
        obs_space = self.env.observation_space
        action_space = self.env.action_space

        # Determine observation space dimensions
        if isinstance(obs_space, gym.spaces.Box):
            if len(obs_space.shape) == 1:
                # Vector observations for SimpleDQN
                input_dim = obs_space.shape[0]
            else:
                # Flatten image observations
                input_dim = np.prod(obs_space.shape)
        elif isinstance(obs_space, gym.spaces.Discrete):
            # Discrete observation space (e.g., FrozenLake)
            input_dim = 1  # Using scalar representation
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(obs_space)}")

        # Determine action space dimensions
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0]
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")

        # Initialize algorithm learner
        if self.algorithm_name == "DQN":
            self.learner = DQN(
                tensorboard_logging=tensorboard_logging,
                env=self.env,
                input_dim=input_dim,
                action_dim=action_dim,
                buffer_size=self.replay_buffer_size,
            )
        elif self.algorithm_name == "REINFORCE":
            self.learner = REINFORCE(
                tensorboard_logging=tensorboard_logging,
                env=self.env,
                input_dim=input_dim,
                action_dim=action_dim,
                buffer_size=self.replay_buffer_size,
            )

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

    def run_training_epoch(self):
        """
        Run a single epoch of training by interacting with the environment.
        """
        # Initialize episode
        obs, info = self.env.reset(seed=self.seed)
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # Select action using the agent's policy
            action, value = self.learner.step(obs)

            # Take action in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            if self.algorithm_name == "DQN":
                self.learner.store_transition(
                    obs=obs,
                    act=action,
                    rew=reward,
                    next_obs=next_obs,
                    done=done
                )
            elif self.algorithm_name == "REINFORCE":
                self.learner.store_transition(
                    obs=obs,
                    act=action,
                    rew=reward,
                    done=done,
                    val=value
                )

            total_reward += reward
            step_count += 1
            obs = next_obs

            # Train models according to training frequency
            if self.learner.traj > 0 and self.learner.traj % self.learner._traj_per_epoch == 0:
                self.learner.train_models(epoch_num=self.epoch)

        print(f"Episode completed: Total Reward = {total_reward}, Epoch Steps = {step_count}")

        if self.learner.logger is not None:
            self.learner.logger.log_rewards(self.epoch, total_reward)
