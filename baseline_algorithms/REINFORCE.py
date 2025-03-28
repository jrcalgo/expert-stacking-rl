import os

import tensorflow as tf
from keras import Input
from tensorflow import keras
import gymnasium as gym
import numpy as np
from utils import load_hyperparams, combined_shape, TensorBoardLogger

ensemble, hyperparams = load_hyperparams('reinforce')


def policy_network(input_dim: int, action_dim: int, seed: int, activations='relu'):
    tf.random.set_seed(seed)
    with tf.device('GPU:0'):
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation=activations),
            keras.layers.Dense(64, activation=activations),
            keras.layers.Dense(16, activation=activations),
            keras.layers.Dense(action_dim, activation='softmax')
        ])
    return model


def value_network(input_dim, seed:int, activations='relu'):
    tf.random.set_seed(seed)
    with tf.device('GPU:0'):
        model = keras.Sequential([
            Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation=activations),
            keras.layers.Dense(128, activation=activations),
            keras.layers.Dense(1)
        ])
    return model


def _discount_cumsum(x, discount):
    """
    Calculate discounted cumulative sum of x
    """
    n = len(x)

    if n == 0:
        return np.zeros_like(x)

    y = np.zeros_like(x)
    y[-1] = x[-1]
    for t in reversed(range(n-1)):
        y[t] = x[t] + discount * y[t+1]
    return y


class REINFORCE(tf.Module):
    def __init__(self,
                 tensorboard_logging: bool,
                 env: gym.Env,
                 input_dim,
                 action_dim,
                 buffer_size=100000,
                 traj_per_epoch: int = hyperparams['traj_per_epoch'],
                 gamma: float = hyperparams['gamma'],
                 lam: float = hyperparams['lam'],
                 pi_lr: float = hyperparams['pi_lr'],
                 vf_lr: float = hyperparams['vf_lr'],
                 train_vf_iters: int = hyperparams['train_vf_iters'],
                 initial_seed: int = ensemble['initial_seed']
                 ):

        super().__init__()
        self.env = env

        self._input_dim = input_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._traj_per_epoch = traj_per_epoch
        self._gamma = gamma
        self._lam = lam
        self._pi_lr = pi_lr
        self._vf_lr = vf_lr
        self._train_vf_iters = train_vf_iters
        self._initial_seed = initial_seed

        self.policy_network = policy_network(input_dim, action_dim, seed=initial_seed)
        self.value_network = value_network(input_dim, seed=initial_seed)

        self.pi_optimizer = keras.optimizers.Adam(learning_rate=pi_lr)
        self.vf_optimizer = keras.optimizers.Adam(learning_rate=vf_lr)

        self.data_ptr, self.path_start, self.max_size, self.capacity = 0, 0, buffer_size, buffer_size
        self.obs_buffer = np.zeros(combined_shape(buffer_size, input_dim), dtype=np.float32)
        self.act_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.rew_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(buffer_size, dtype=np.float32)

        self.logp_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.val_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.traj = 0

        self.epoch = 0

        self.logger = None
        if tensorboard_logging:
            self.log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
            self.logger = TensorBoardLogger(log_dir=self.log_dir, filename='REINFORCE' + str(os.getpid()))

    def store_transition(self, obs, act, rew, done, val):
        assert self.data_ptr <= self.max_size, "Buffer overflow"

        if self.data_ptr >= self.capacity:
            self.data_ptr = 0

        idx = self.data_ptr
        self.obs_buffer[idx] = obs
        self.act_buffer[idx] = act
        self.rew_buffer[idx] = rew
        self.terminal_buffer[idx] = done
        self.val_buffer[idx] = val

        self.data_ptr += 1

        if done:
            self.compute_advantage_discount()
            self.path_start = self.data_ptr

    def get_buffer_batch(self):
        assert self.data_ptr >= 1, "Not enough samples in buffer"

        return (self.obs_buffer, self.act_buffer, self.rew_buffer, self.terminal_buffer, self.val_buffer,
                self.adv_buffer, self.ret_buffer)

    def step(self, obs: np.ndarray):
        """
        Select action using the policy network.
        """
        obs = tf.convert_to_tensor(obs.reshape(1, -1), dtype=tf.float32)
        action_probs = self.policy_network(obs)
        action = np.random.choice(self._action_dim, p=action_probs.numpy().flatten())
        value = self.value_network(obs)

        return action, value

    @tf.function
    def compute_pi_loss(self, obs_tf, acts_tf, adv_tf):
        """
        Compute the loss for the policy network.
        """
        acts_one_hot = tf.one_hot(acts_tf, depth=self._action_dim, dtype=tf.float32)

        action_probs = self.policy_network(obs_tf)
        action_probs = tf.cast(action_probs, tf.float32)
        action_probs = tf.reduce_sum(acts_one_hot * action_probs, axis=-1)
        log_probs = tf.math.log(action_probs + 1e-10)

        loss = -tf.reduce_mean(log_probs * adv_tf)

        return loss

    @tf.function
    def compute_vf_loss(self, obs_tf, ret_tf):
        """
        Compute the loss for the value network.
        """
        values = self.value_network(obs_tf)
        loss = tf.reduce_mean(tf.square((values - ret_tf)))

        return loss

    def compute_advantage_discount(self, last_value=0):
        """
        Compute the advantage function using the discounted rewards.
        """
        path = slice(self.path_start, last_value)
        rews = np.append(self.rew_buffer[path], last_value)
        vals = np.append(self.val_buffer[path], last_value)

        deltas = rews[:-1] + self._gamma * vals[1:] - vals[:-1]
        self.adv_buffer[path] = _discount_cumsum(deltas, self._gamma * self._lam)

        # Rewards-to-go, which is the target for value function
        self.ret_buffer[path] = _discount_cumsum(rews, self._gamma)[:-1]

        self.path_start = self.data_ptr

    def train_models(self, epoch_num: int):
        if self.data_ptr < 1:
            return

        obs, acts, rews, dones, vals, advs, rets = self.get_buffer_batch()

        # Convert tensors
        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        acts_tf = tf.convert_to_tensor(acts, dtype=tf.int32)

        adv_tf = tf.convert_to_tensor(advs, dtype=tf.float32)
        ret_tf = tf.convert_to_tensor(rets, dtype=tf.float32)

        adv_mean, adv_std = tf.nn.moments(adv_tf, axes=[0])
        adv_tf = (adv_tf - adv_mean) / (adv_std + 1E-8)

        with tf.GradientTape() as tape:
            loss = self.compute_pi_loss(obs_tf, acts_tf, adv_tf)
            grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(grads, self.policy_network.trainable_variables))

        for _ in range(self._train_vf_iters):
            with tf.GradientTape() as tape:
                loss, grads = self.compute_vf_loss(obs_tf, ret_tf)
                grads = tape.gradient(loss, self.value_network.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(grads, self.value_network.trainable_variables))
        del tape
