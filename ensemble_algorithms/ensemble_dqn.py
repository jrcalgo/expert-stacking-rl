from copy import deepcopy
import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import numpy as np
from utils import load_hyperparams, combined_shape
from expert_stacking_encoder import ExpertStackingEncoder

ensemble, hyperparams = load_hyperparams('dqn')


class SumTree:
    """
    SumTree data structure for Prioritized Experience Replay.
    Each leaf node contains the priority of an experience.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_ptr = 0

    def add(self, priority):
        tree_idx = self.data_ptr + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_ptr += 1
        if self.data_ptr >= self.capacity:
            self.data_ptr = 0

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate the change up
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left]:
                    parent_idx = left
                else:
                    v -= self.tree[left]
                    parent_idx = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, mlp_input_dim, cnn_input_dim, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # Initial max priority

        # Initialize buffers
        self.mlp_obs_buffer = np.zeros(combined_shape(capacity, mlp_input_dim), dtype=np.float32)  # To be defined properly
        self.cnn_obs_buffer = np.zeros(combined_shape(capacity, cnn_input_dim), dtype=np.float32)
        self.act_buffer = np.zeros(combined_shape(capacity), dtype=np.int32)
        self.rew_buffer = np.zeros(combined_shape(capacity), dtype=np.float32)
        self.mlp_next_obs_buffer = np.zeros(combined_shape(capacity, mlp_input_dim), dtype=np.float32)
        self.cnn_next_obs_buffer = np.zeros(combined_shape(capacity, cnn_input_dim), dtype=np.float32)
        self.terminal_buffer = np.zeros(combined_shape(capacity), dtype=bool)

        self.size = 0

    def store_transition(self, mlp_obs, cnn_obs, act, rew, mlp_next_obs, cnn_next_obs, done):
        idx = self.tree.data_ptr
        self.mlp_obs_buffer[idx] = mlp_obs
        self.cnn_obs_buffer[idx] = cnn_obs
        self.act_buffer[idx] = act
        self.rew_buffer[idx] = rew
        self.mlp_next_obs_buffer[idx] = mlp_next_obs
        self.cnn_next_obs_buffer[idx] = cnn_next_obs
        self.terminal_buffer[idx] = done

        # Assign maximum priority to new transition
        self.tree.add(self.max_priority ** self.alpha)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            leaf_idx, priority, data_idx = self.tree.get_leaf(s)
            priorities.append(priority)
            batch.append(data_idx)
            idxs.append(leaf_idx)
        sampling_probabilities = priorities / self.tree.total_priority
        is_weight = np.power(self.capacity * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        return batch, is_weight

    def update_priorities(self, leaf_idxs, priorities):
        for leaf_idx, priority in zip(leaf_idxs, priorities):
            self.tree.update(leaf_idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)


class EnsembleDQN(tf.Module):
    def __init__(self,
                 env: gym.Env,
                 mlp_input_size,
                 mlp_input_dim,
                 cnn_input_size,
                 cnn_input_dim,
                 action_dim,
                 mlp_activations,
                 cnn_activations,
                 buffer_size,
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
                 initial_seed: int = ensemble['initial_seed'],
                 kernel_size: int = ensemble['kernel_size'],
                 stride: int = ensemble['stride'],
                 padding: str = ensemble['padding'],
                 mlp_count: int = ensemble['mlp_count'],
                 cnn_count: int = ensemble['cnn_count'],
                 mlp_batch_size: int = ensemble['mlp_batch_size'],
                 cnn_batch_size: int = ensemble['cnn_batch_size'],
                 expert_rotation_freq: int = ensemble['expert_rotation_freq'],
                 grad_dir: str = ensemble['grad_dir'],
                 alpha: float = 0.6,
                 beta: float = 0.4
                 ):

        super().__init__()
        self.env = env

        # Hyperparameters
        self._mlp_input_size = mlp_input_size
        self._mlp_input_dim = mlp_input_dim
        self._cnn_input_size = cnn_input_size
        self._cnn_input_dim = cnn_input_dim
        self._mlp_activations = mlp_activations
        self._cnn_activations = cnn_activations
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

        self._expert_rotation_freq = expert_rotation_freq

        # PER parameters
        self.alpha = alpha
        self.beta = beta

        # Ensemble Q-model
        self.ensemble_model = ExpertStackingEncoder(
            mlp_activations=mlp_activations,
            cnn_activations=cnn_activations,
            mlp_input_dim=mlp_input_dim,
            cnn_input_dim=cnn_input_dim,
            act_dim=action_dim,
            initial_seed=initial_seed,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            mlp_count=mlp_count,
            cnn_count=cnn_count,
            mlp_batch_size=mlp_batch_size,
            cnn_batch_size=cnn_batch_size,
            grad_dir=grad_dir
        )

        # Target model: same architecture, updated periodically
        self._target_model = deepcopy(self.ensemble_model)
        self.update_target_model()

        # One optimizer for all models inside Q_ensemble_model
        all_train_vars = []
        for m in self.ensemble_model.models:
            all_train_vars.extend(m.trainable_variables)
        all_train_vars.extend(self.ensemble_model.stack_encoder.trainable_variables)
        self._optimizer = keras.optimizers.Adam(learning_rate=q_lr)

        # Initialize Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=self._buffer_size, mlp_input_dim=mlp_input_dim, cnn_input_dim=cnn_input_dim, alpha=self.alpha)

        self.epochs = 0
        self.traj = 0

    def store_transition(self, mlp_obs, cnn_obs, act, rew, mlp_next_obs, cnn_next_obs, done):
        self.replay_buffer.store_transition(mlp_obs, cnn_obs, act, rew, mlp_next_obs, cnn_next_obs, done)
        self.traj += 1

    def get_buffer_batch(self):
        batch_indices, is_weights = self.replay_buffer.sample(self._batch_size, self.beta)
        mlp_obs = self.replay_buffer.mlp_obs_buffer[batch_indices]
        cnn_obs = self.replay_buffer.cnn_obs_buffer[batch_indices]
        acts = self.replay_buffer.act_buffer[batch_indices]
        rews = self.replay_buffer.rew_buffer[batch_indices]
        mlp_next_obs = self.replay_buffer.mlp_next_obs_buffer[batch_indices]
        cnn_next_obs = self.replay_buffer.cnn_next_obs_buffer[batch_indices]
        dones = self.replay_buffer.terminal_buffer[batch_indices]
        return mlp_obs, cnn_obs, acts, rews, mlp_next_obs, cnn_next_obs, dones, batch_indices, is_weights

    def step(self, mlp_obs: tf.Tensor, cnn_obs: tf.Tensor):
        # Epsilon-greedy action selection
        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)
        if np.random.rand() < self._epsilon:
            return self.env.action_space.sample()
        else:
            q_vals, _ = self.ensemble_model.forward(mlp_obs, cnn_obs)  # shape [1, act_dim]
            action = tf.argmax(q_vals, axis=-1).numpy()[0]
            return action

    def compute_q_loss(self, mlp_obs_tf, cnn_obs_tf, acts_tf, rews_tf, mlp_next_obs_tf, cnn_next_obs_tf, dones_tf, is_weights):
        # Compute target values
        # Q'(s', a')
        next_q_vals = self._target_model.forward(mlp_next_obs_tf, cnn_next_obs_tf)[0]
        max_next_q = tf.reduce_max(next_q_vals, axis=-1)
        targets = rews_tf + self._gamma * (1.0 - dones_tf) * max_next_q

        # Compute predicted Q(s,a)
        q_vals = self.ensemble_model.forward(mlp_obs_tf, cnn_obs_tf)[0]
        # Gather the Q-values for the chosen actions
        batch_indices = tf.range(self._batch_size, dtype=tf.int32)
        chosen_q = tf.gather_nd(q_vals, tf.stack([batch_indices, acts_tf], axis=1))

        # Compute TD errors
        td_errors = targets - chosen_q

        # Huber loss (more robust to outliers)
        loss = tf.reduce_mean(is_weights * tf.square(td_errors))  # Using MSE; you can switch to Huber if preferred
        return loss, tf.abs(td_errors)

    def train_models(self, epoch_num: int):
        if self.replay_buffer.size < self._batch_size:
            # Not enough samples to train
            return

        mlp_obs, cnn_obs, acts, rews, mlp_next_obs, cnn_next_obs, dones, batch_leaf_idxs, is_weights = self.get_buffer_batch()

        # Convert to tensors
        mlp_obs_tf = tf.convert_to_tensor(mlp_obs, dtype=tf.float32)
        cnn_obs_tf = tf.convert_to_tensor(cnn_obs, dtype=tf.float32)
        acts_tf = tf.convert_to_tensor(acts, dtype=tf.int32)
        rews_tf = tf.convert_to_tensor(rews, dtype=tf.float32)
        mlp_next_obs_tf = tf.convert_to_tensor(mlp_next_obs, dtype=tf.float32)
        cnn_next_obs_tf = tf.convert_to_tensor(cnn_next_obs, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)
        is_weights_tf = tf.convert_to_tensor(is_weights, dtype=tf.float32)

        for _ in range(self._train_q_iters):
            with tf.GradientTape() as tape:
                loss, td_errors = self.compute_q_loss(
                    mlp_obs_tf, cnn_obs_tf, acts_tf, rews_tf,
                    mlp_next_obs_tf, cnn_next_obs_tf, dones_tf, is_weights_tf
                )

            # Gather trainable variables
            all_train_vars = []
            for m in self.ensemble_model.models:
                all_train_vars.extend(m.trainable_variables)
            all_train_vars.extend(self.ensemble_model.stack_encoder.trainable_variables)

            grads = tape.gradient(loss, all_train_vars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
            self._optimizer.apply_gradients(zip(grads, all_train_vars))

        # Update priorities
        td_errors_np = td_errors.numpy()
        new_priorities = td_errors_np + 1e-6  # Small constant to avoid zero priority
        self.replay_buffer.update_priorities(batch_leaf_idxs, new_priorities)

        if epoch_num % self._train_update_freq == 0:
            self.update_target_model()

        if epoch_num % self._expert_rotation_freq == 0:
            self.ensemble_model.rotate_expert_minibatch()

    def update_target_model(self):
        # Copy weights from Q_model to target_model
        for i, m in enumerate(self.ensemble_model.models):
            self._target_model.models[i].set_weights(m.get_weights())
        self._target_model.stack_encoder.set_weights(self.ensemble_model.stack_encoder.get_weights())