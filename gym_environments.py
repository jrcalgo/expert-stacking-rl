import gymnasium as gym
from gymnasium import Env
from typing import Any, Optional, Type
import numpy as np


# Optional: Import specific wrappers based on environment types
# For example, Atari environments often use wrappers from `stable_baselines3`
# or other packages for frame stacking, gray-scaling, etc.
# Here, we'll use some common Gym wrappers for illustration.
class GymEnvironment:
    """
    Base class for Gym environments.
    """
    def __init__(self, env_name: str, model: Any = None):
        """
        Initialize the Gym environment.

        Args:
            env_name (str): The name of the Gym environment.
            model (Any, optional): A model to be associated with the environment. Defaults to None.
        """
        self.env_name = env_name
        self.model = model
        self.env: Optional[Env] = None
        self._initialize_env()

    def _initialize_env(self):
        """
        Initialize the Gym environment and apply any common wrappers.
        """ 
        self.env = gym.make(self.env_name)
        self.env = self._apply_common_wrappers(self.env)
        print(f"Initialized environment: {self.env_name}")

    def _apply_common_wrappers(self, env: Env) -> Env:
        """
        Apply common wrappers to all environments.

        Args:
            env (Env): The Gym environment.

        Returns:
            Env: The wrapped Gym environment.
        """
        # Example: Enable automatic rendering
        # env = gym.wrappers.RecordEpisodeStatistics(env)

        # Example: Make the environment monitorable
        # env = gym.wrappers.Monitor(env, './video', force=True)

        return env

    def reset(self) -> Any:
        """
        Reset the environment to an initial state and return the initial observation.

        Returns:
            Any: The initial observation.
        """
        if self.env is None:
            raise ValueError("Environment not initialized.")
        observation = self.env.reset()
        print(f"Environment {self.env_name} reset.")
        return observation

    def step(self, action: Any) -> tuple:
        """
        Run one timestep of the environment's dynamics.

        Args:
            action (Any): An action provided by the agent.

        Returns:
            tuple: A tuple containing (observation, reward, done, info).
        """
        if self.env is None:
            raise ValueError("Environment not initialized.")
        observation, reward, done, info = self.env.step(action)
        print(f"Step taken in {self.env_name}: Action={action}, Reward={reward}, Done={done}")
        return observation, reward, done, info

    def render(self, mode: str = 'human'):
        """
        Render the environment.

        Args:
            mode (str, optional): The mode to render with. Defaults to 'human'.
        """
        if self.env is None:
            raise ValueError("Environment not initialized.")
        self.env.render(mode=mode)
        print(f"Rendered environment {self.env_name} in mode {mode}.")

    def close(self):
        """
        Close the environment.
        """
        if self.env is not None:
            self.env.close()
            print(f"Closed environment {self.env_name}.")

    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for the environment's random number generator(s).

        Args:
            seed (Optional[int], optional): The seed to use. Defaults to None.
        """
        if self.env is not None:
            self.env.seed(seed)
            print(f"Environment {self.env_name} seeded with {seed}.")


# Specific Environment Classes
class AtariEnv(GymEnvironment):
    """
    Class for Atari Gym environments.
    """
    def _initialize_env(self):
        super()._initialize_env()
        if 'Atari' in self.env_name:
            self.env = self._apply_atari_wrappers(self.env)

    def _apply_atari_wrappers(self, env: Env) -> Env:
        """
        Apply Atari-specific wrappers.

        Args:
            env (Env): The Atari Gym environment.

        Returns:
            Env: The wrapped environment.
        """
        # Example: Apply frame stacking and grayscale
        # These wrappers might come from stable_baselines3 or other packages
        # Here, we'll use Gym's built-in wrappers for illustration

        # env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        # env = gym.wrappers.FrameStack(env, num_stack=4)

        print(f"Applied Atari-specific wrappers to {self.env_name}.")
        return env


class ClassicControlEnv(GymEnvironment):
    """
    Class for Classic Control Gym environments.
    """
    def _initialize_env(self):
        super()._initialize_env()
        if self.env_name in ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']:
            self.env = self._apply_classic_control_wrappers(self.env)

    def _apply_classic_control_wrappers(self, env: Env) -> Env:
        """
        Apply Classic Control-specific wrappers.

        Args:
            env (Env): The Classic Control Gym environment.

        Returns:
            Env: The wrapped environment.
        """
        # Example: Add a time limit wrapper if not already present
        # env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

        print(f"Applied Classic Control-specific wrappers to {self.env_name}.")
        return env


class Box2DEnv(GymEnvironment):
    """
    Class for Box2D Gym environments.
    """
    def _initialize_env(self):
        super()._initialize_env()
        # Example environments: 'LunarLander-v2', 'BipedalWalker-v3'
        box2d_env_names = ['LunarLander-v2', 'BipedalWalker-v3', 'CarRacing-v0']
        if self.env_name in box2d_env_names:
            self.env = self._apply_box2d_wrappers(self.env)

    def _apply_box2d_wrappers(self, env: Env) -> Env:
        """
        Apply Box2D-specific wrappers.

        Args:
            env (Env): The Box2D Gym environment.

        Returns:
            Env: The wrapped environment.
        """
        # Example: Apply frame stacking or other wrappers as needed
        # env = gym.wrappers.FrameStack(env, num_stack=4)

        print(f"Applied Box2D-specific wrappers to {self.env_name}.")
        return env


class ToyTextEnv(GymEnvironment):
    """
    Class for Toy Text Gym environments.
    """
    def _initialize_env(self):
        super()._initialize_env()
        toy_text_env_names = ['FrozenLake-v1', 'Taxi-v3', 'CliffWalking-v0']
        if self.env_name in toy_text_env_names:
            self.env = self._apply_toy_text_wrappers(self.env)

    def _apply_toy_text_wrappers(self, env: Env) -> Env:
        """
        Apply Toy Text-specific wrappers.

        Args:
            env (Env): The Toy Text Gym environment.

        Returns:
            Env: The wrapped environment.
        """
        # Toy Text environments might not require special wrappers, but you can add them if needed
        print(f"Applied Toy Text-specific wrappers to {self.env_name}.")
        return env
    

# Factory Function to Initialize Appropriate Environment Class
def create_env(env_name: str, model: Any = None) -> GymEnvironment:
    """
    Factory function to create an appropriate GymEnvironment subclass instance.

    Args:
        env_name (str): The name of the Gym environment.
        model (Any, optional): A model to be associated with the environment. Defaults to None.

    Returns:
        GymEnvironment: An instance of a GymEnvironment subclass.
    """
    atari_envs = ['Pong-v0', 'Breakout-v0', 'SpaceInvaders-v0', 'Atari']  # Add more Atari envs as needed
    box2d_envs = ['LunarLander-v2', 'BipedalWalker-v3', 'CarRacing-v0']
    classic_control_envs = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
    toy_text_envs = ['FrozenLake-v1', 'Taxi-v3', 'CliffWalking-v0']

    if env_name in atari_envs or any(env_name.startswith(name) for name in atari_envs if name == 'Atari'):
        return AtariEnv(env_name, model)
    elif env_name in classic_control_envs:
        return ClassicControlEnv(env_name, model)
    elif env_name in box2d_envs:
        return Box2DEnv(env_name, model)
    elif env_name in toy_text_envs:
        return ToyTextEnv(env_name, model)
    else:
        # Default to base GymEnvironment if no specific class matches
        print(f"No specific class for {env_name}, using base GymEnvironment.")
        return GymEnvironment(env_name, model)