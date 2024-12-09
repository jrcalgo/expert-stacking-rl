import tensorflow as tf
from keras import Input
from tensorflow import keras
import gymnasium as gym
import numpy as np
from utils import load_hyperparams, combined_shape

# Load hyperparameters
ensemble, hyperparams = load_hyperparams('dqn')


def q_network(input_dim, action_dim, seed: int, activations='relu'):
    tf.random.set_seed(seed)
    with tf.device('/GPU:0'):
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation=activations),
            keras.layers.Dense(24, activation=activations),
            keras.layers.Dense(16, activation=activations),
            keras.layers.Dense(8, activation=activations),
            keras.layers.Dense(action_dim)
        ])
    return model


class DQN(tf.Module):
    def __init__(self,
                 env: gym.Env,
                 input_dim,
                 action_dim,
                 buffer_size=10000,
                 batch_size: int = hyperparams['batch_size'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'],
                 gamma: float = hyperparams['gamma'],
                 epsilon: float = hyperparams['epsilon'],
                 epsilon_min: float = hyperparams['epsilon_min'],
                 epsilon_decay: float = hyperparams['epsilon_decay'],
                 q_lr: float = hyperparams['q_lr'],
                 train_q_iters: int = hyperparams['train_q_iters'],
                 train_update_freq: int = hyperparams['train_update_freq'],
                 target_update_freq: int = hyperparams['target_update_freq'],
                 initial_seed: int = ensemble['initial_seed']
                 ):

        super().__init__()
        self.env = env

        # Hyperparameters
        self._input_dim = input_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._traj_per_epoch = traj_per_epoch
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._q_lr = q_lr
        self._train_q_iters = train_q_iters
        self._train_update_freq = train_update_freq
        self._target_update_freq = target_update_freq

        # Q-network
        self.q_network = q_network(input_dim, action_dim, seed=initial_seed)

        # Target network: same architecture, updated periodically
        self.target_network = q_network(input_dim, action_dim, seed=initial_seed)
        self.update_target_network()

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=q_lr)

        # Replay Buffer
        self.data_ptr, self.path_start, self.max_size, self.capacity = 0, 0, buffer_size, buffer_size
        self.obs_buffer = np.zeros(combined_shape(buffer_size, input_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros(combined_shape(buffer_size, input_dim), dtype=np.float32)
        self.act_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.rew_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(buffer_size, dtype=bool)
        self.traj = 0

        self.epochs = 0

    def store_transition(self, obs, act, rew, next_obs, done):
        assert self.data_ptr <= self.max_size

        if self.data_ptr >= self.capacity:
            self.data_ptr = 0

        idx = self.data_ptr
        self.obs_buffer[idx] = obs
        self.next_obs_buffer[idx] = next_obs
        self.act_buffer[idx] = act
        self.rew_buffer[idx] = rew
        self.terminal_buffer[idx] = done

        self.data_ptr += 1
        self.traj += 1

        if done:
            self.path_start = self.data_ptr

    def get_buffer_batch(self):
        assert self.data_ptr >= self._batch_size, "Not enough samples in buffer"

        # Sample random batch
        batch_indices = np.random.randint(0, self.data_ptr, size=self._batch_size)
        obs = self.obs_buffer[batch_indices]
        next_obs = self.next_obs_buffer[batch_indices]
        acts = self.act_buffer[batch_indices]
        rews = self.rew_buffer[batch_indices]
        dones = self.terminal_buffer[batch_indices]
        return obs, acts, rews, next_obs, dones, batch_indices

    def step(self, obs: np.ndarray):
        # Epsilon-greedy action selection
        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)
        if np.random.rand() < self._epsilon:
            return self.env.action_space.sample()
        else:
            obs_tensor = tf.convert_to_tensor(obs.reshape(1, -1), dtype=tf.float32)
            q_vals = self.q_network(obs_tensor)  # shape [1, action_dim]
            action = tf.argmax(q_vals, axis=-1).numpy()[0]
            return action

    def compute_q_loss(self, obs_tf, acts_tf, rews_tf, next_obs_tf, dones_tf):
        """
        Compute the Mean Squared Error loss for DQN.

        Args:
            obs_tf (tf.Tensor): Current observations [batch_size, input_dim].
            acts_tf (tf.Tensor): Actions taken [batch_size].
            rews_tf (tf.Tensor): Rewards received [batch_size].
            next_obs_tf (tf.Tensor): Next observations [batch_size, input_dim].
            dones_tf (tf.Tensor): Done flags [batch_size].

        Returns:
            loss (tf.Tensor): MSE loss.
        """
        # Current Q-values
        q_vals = self.q_network(obs_tf)  # [batch_size, action_dim]
        acts_one_hot = tf.one_hot(acts_tf, depth=self._action_dim, dtype=tf.float32)
        chosen_q = tf.reduce_sum(q_vals * acts_one_hot, axis=1)  # [batch_size]

        # Target Q-values
        target_q_vals = self.target_network(next_obs_tf)  # [batch_size, action_dim]
        max_next_q = tf.reduce_max(target_q_vals, axis=1)  # [batch_size]
        targets = rews_tf + self._gamma * (1.0 - dones_tf) * max_next_q  # [batch_size]

        # Compute loss
        loss = tf.reduce_mean(tf.square(targets - chosen_q))
        return loss

    def train_models(self, epoch_num: int):
        if self.data_ptr < self._batch_size:
            # Not enough samples to train
            return

        obs, acts, rews, next_obs, dones, batch_idxs = self.get_buffer_batch()

        # Convert to tensors
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        next_obs_tf = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        acts_tf = tf.convert_to_tensor(acts, dtype=tf.int32)
        rews_tf = tf.convert_to_tensor(rews, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)

        for _ in range(self._train_q_iters):
            with tf.GradientTape() as tape:
                loss = self.compute_q_loss(obs_tf, acts_tf, rews_tf, next_obs_tf, dones_tf)
            grads = tape.gradient(loss, self.q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        if epoch_num % self._target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        # Copy weights from Q_network to target_network
        self.target_network.set_weights(self.q_network.get_weights())
