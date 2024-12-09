import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import numpy as np
from utils import load_hyperparams, combined_shape
from expert_stacking_encoder import ExpertStackingEncoder

ensemble, hyperparams = load_hyperparams('ensemble_dqn')


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
                 grad_dir: str = ensemble['grad_dir']
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

        self._mlp_count = mlp_count
        self._cnn_count = cnn_count

        self._expert_rotation_freq = expert_rotation_freq

        # Ensemble Q-model
        self.ensemble_model = ExpertStackingEncoder(
            mlp_activations=mlp_activations,
            cnn_activations=cnn_activations,
            mlp_input_dim=mlp_input_dim,
            cnn_input_shape=cnn_input_dim,
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
        self._target_model = ExpertStackingEncoder(
            mlp_activations=mlp_activations,
            cnn_activations=cnn_activations,
            mlp_input_dim=mlp_input_dim,
            cnn_input_shape=cnn_input_dim if cnn_count > 0 else None,
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

        self.update_target_model()

        # One optimizer for all expert models inside Q_ensemble_model
        self._expert_optimizer = keras.optimizers.Adam(learning_rate=q_lr)
        # One optimizer for the encoder
        self._encoder_optimizer = keras.optimizers.Adam(learning_rate=q_lr)

        # Replay Buffer
        self.data_ptr, self.path_start, self.max_size, self.capacity = 0, 0, buffer_size, buffer_size
        self.mlp_obs_buffer = np.zeros(combined_shape(buffer_size, mlp_input_dim), dtype=np.float32)
        self.cnn_obs_buffer = np.zeros(combined_shape(buffer_size, cnn_input_dim), dtype=np.float32)
        self.act_buffer = np.zeros(combined_shape(buffer_size), dtype=np.int32)
        self.rew_buffer = np.zeros(combined_shape(buffer_size), dtype=np.float32)
        self.mlp_next_obs_buffer = np.zeros(combined_shape(buffer_size, mlp_input_dim), dtype=np.float32)
        self.cnn_next_obs_buffer = np.zeros(combined_shape(buffer_size, cnn_input_dim), dtype=np.float32)
        self.terminal_buffer = np.zeros(combined_shape(buffer_size), dtype=bool)
        self.traj = 0

        self.epochs = 0

    def store_transition(self, mlp_obs, cnn_obs, act, rew, mlp_next_obs, cnn_next_obs, done):
        assert self.data_ptr <= self.max_size

        if self.data_ptr >= self.capacity:
            self.data_ptr = 0

        idx = self.data_ptr
        if self._mlp_count is not None:
            self.mlp_obs_buffer[idx] = mlp_obs
            self.mlp_next_obs_buffer[idx] = mlp_next_obs
        if self._cnn_count is not None:
            self.cnn_obs_buffer[idx] = cnn_obs
            self.cnn_next_obs_buffer[idx] = cnn_next_obs
        self.act_buffer[idx] = act
        self.rew_buffer[idx] = rew
        self.terminal_buffer[idx] = done

        self.data_ptr += 1
        self.traj += 1

        if done:
            self.path_start = self.data_ptr

    def get_buffer_batch(self):
        assert self.data_ptr < self.max_size
        assert self.data_ptr >= self._batch_size

        self.data_ptr, self.path_start = 0, 0

        batch_indices = np.random.uniform(0, self.capacity, self._batch_size).astype(int)
        mlp_obs = self.mlp_obs_buffer[batch_indices]
        cnn_obs = self.cnn_obs_buffer[batch_indices]
        acts = self.act_buffer[batch_indices]
        rews = self.rew_buffer[batch_indices]
        mlp_next_obs = self.mlp_next_obs_buffer[batch_indices]
        cnn_next_obs = self.cnn_next_obs_buffer[batch_indices]
        dones = self.terminal_buffer[batch_indices]
        return mlp_obs, cnn_obs, acts, rews, mlp_next_obs, cnn_next_obs, dones, batch_indices

    def step(self, mlp_obs: tf.Tensor, cnn_obs: tf.Tensor):
        # Epsilon-greedy action selection
        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)
        if np.random.rand() < self._epsilon:
            return self.env.action_space.sample()
        else:
            q_vals, _ = self.ensemble_model.forward(mlp_obs, cnn_obs)  # shape [1, act_dim]
            action = tf.argmax(q_vals, axis=-1).numpy()[0]
            return action

    def compute_q_loss(self, mlp_obs_tf, cnn_obs_tf, acts_tf, rews_tf,
                       mlp_next_obs_tf, cnn_next_obs_tf, dones_tf):
        """
        Compute the Q-loss for the Ensemble DQN with Prioritized Experience Replay.

        Args:
            mlp_obs_tf (tf.Tensor): Current MLP observations [batch_size, mlp_dim].
            cnn_obs_tf (tf.Tensor): Current CNN observations [batch_size, cnn_dim].
            acts_tf (tf.Tensor): Actions taken [batch_size].
            rews_tf (tf.Tensor): Rewards received [batch_size].
            mlp_next_obs_tf (tf.Tensor): Next MLP observations [batch_size, mlp_dim].
            cnn_next_obs_tf (tf.Tensor): Next CNN observations [batch_size, cnn_dim].
            dones_tf (tf.Tensor): Done flags [batch_size].

        Returns:
            encoder_loss (tf.Tensor): Encoder loss.
            expert_loss (tf.Tensor): Cumulative expert loss.
        """

        # acts_one_hot = tf.one_hot(acts_tf, depth=self._action_dim, dtype=tf.float32)
        #
        # if self._mlp_count > 0 and self._cnn_count > 0:
        #     concatenated_obs = tf.concat([mlp_obs_tf, cnn_obs_tf], axis=-1)
        # elif self._mlp_count > 0:
        #     concatenated_obs = mlp_obs_tf
        # elif self._cnn_count > 0:
        #     concatenated_obs = cnn_obs_tf
        # else:
        #     raise ValueError("At least one of mlp_count or cnn_count must be > 0")
        #
        # encoder_input = tf.concat([concatenated_obs, acts_one_hot], axis=-1)
        #
        # encoder_q = self.ensemble_model.stack_encoder(encoder_input)
        # chosen_q = tf.reduce_sum(encoder_q * acts_one_hot, axis=1)
        #
        # actions_identity = tf.eye(self._action_dim, dtype=tf.float32)
        #
        # if self._mlp_count > 0 and self._cnn_count > 0:
        #     concatenated_next_obs = tf.concat([mlp_next_obs_tf, cnn_next_obs_tf], axis=-1)
        # elif self._mlp_count > 0:
        #     concatenated_next_obs = mlp_next_obs_tf
        # elif self._cnn_count > 0:
        #     concatenated_next_obs = cnn_next_obs_tf
        # else:
        #     raise ValueError("At least one of mlp_count or cnn_count must be > 0")
        #
        # concatenated_next_obs_expanded = tf.expand_dims(concatenated_next_obs, 1)
        # concatenated_next_obs_tiled = tf.tile(concatenated_next_obs_expanded, [1, self._action_dim, 1])
        #
        # actions_identity_expanded = tf.expand_dims(actions_identity, 0)
        # actions_identity_tiled = tf.tile(actions_identity_expanded, [tf.shape(concatenated_next_obs_tiled)[0], 1, 1])
        #
        # target_encoder_input = tf.concat([concatenated_next_obs_tiled, actions_identity_tiled], axis=-1)
        # target_encoder_input = tf.reshape(target_encoder_input, [tf.shape(concatenated_next_obs_tiled)[0] * self._action_dim, -1])
        #
        # target_encoder_q = self._target_model.stack_encoder(target_encoder_input)
        # target_encoder_q = tf.reshape(target_encoder_q, [tf.shape(concatenated_next_obs_tiled)[0], self._action_dim, self._action_dim])
        #
        # max_next_q = tf.reduce_max(target_encoder_q, axis=-1)
        # max_next_encoder_q = tf.reduce_max(max_next_q, axis=-1)
        # encoder_targets = rews_tf + (self._gamma * (1.0 - dones_tf)) * max_next_encoder_q
        #
        # td_errors = encoder_targets - chosen_q
        # squared_td_errors = tf.square(td_errors)
        # encoder_loss = tf.reduce_mean(squared_td_errors)
        #
        # expert_targets = tf.zeros_like(rews_tf, dtype=tf.float32)
        # for expert in self.ensemble_model.mlps:
        #     next_expert_q = expert(mlp_next_obs_tf)
        #     max_expert_q = tf.reduce_max(next_expert_q, axis=-1)
        #     expert_targets += rews_tf + (self._gamma * (1.0 - dones_tf)) * max_expert_q
        #
        # for expert in self.ensemble_model.cnns:
        #     next_expert_q = expert(cnn_next_obs_tf)
        #     max_expert_q = tf.reduce_max(next_expert_q, axis=-1)
        #     expert_targets += rews_tf + (self._gamma * (1.0 - dones_tf)) * max_expert_q
        #
        # total_experts = len(self.ensemble_model.mlps) + len(self.ensemble_model.cnns)
        # if total_experts > 0:
        #     expert_targets = expert_targets / total_experts
        #
        # expert_loss = tf.reduce_mean(tf.square(expert_targets - chosen_q))
        #
        # return encoder_loss, expert_loss

        ## Below is the old implementation of the loss function, this one does work but is not as efficient as the
        # Compute target values # Q'(s', a')
        next_q_vals, _ = self._target_model.forward(mlp_next_obs_tf, cnn_next_obs_tf)
        _, next_expert_vals = self._target_model.forward(mlp_next_obs_tf, cnn_next_obs_tf, voting=False)

        # Compute max_a Q'(s', a')
        max_next_q = tf.reduce_max(next_q_vals, axis=-1)
        ensemble_targets = rews_tf + (self._gamma * (1.0 - dones_tf)) * max_next_q

        # Compute latent expert Q'(s', a')
        max_expert_vals = tf.reduce_max(next_expert_vals, axis=-1)
        expert_targets = rews_tf + (self._gamma * (1.0 - dones_tf)) * max_expert_vals

        # Compute encoder target
        encoder_targets = ensemble_targets - expert_targets

        # Compute predicted values
        # Q(s,a)
        q_vals, expert_preds = self.ensemble_model.forward(mlp_obs_tf, cnn_obs_tf, voting=False)

        # Gather the Q-values for the chosen actions
        batch_indices = tf.range(self._batch_size, dtype=tf.int32)
        chosen_ensemble_q = tf.gather_nd(q_vals, tf.stack([batch_indices, acts_tf], axis=1))
        chosen_expert_q = tf.gather_nd(expert_preds, tf.stack([batch_indices, acts_tf], axis=1))
        chosen_encoder_q = chosen_ensemble_q - chosen_expert_q

        # Compute TD errors
        expert_loss = expert_targets - chosen_expert_q
        encoder_loss = encoder_targets - chosen_encoder_q

        # MSE losses
        encoder_loss = tf.reduce_mean(tf.square(encoder_loss))
        expert_loss = tf.reduce_mean(tf.square(expert_loss))
        return encoder_loss, expert_loss

    def train_models(self, epoch_num: int):
        if self.data_ptr < self._batch_size:
            # Not enough samples to train
            return

        mlp_obs, cnn_obs, acts, rews, mlp_next_obs, cnn_next_obs, dones, batch_idxs = self.get_buffer_batch()

        mlp_obs_tf, mlp_next_obs_tf = None, None
        cnn_obs_tf, cnn_next_obs_tf = None, None
        # Convert to tensors
        if self._mlp_count > 0:
            mlp_obs_tf = tf.convert_to_tensor(mlp_obs, dtype=tf.float32)
            mlp_next_obs_tf = tf.convert_to_tensor(mlp_next_obs, dtype=tf.float32)
        if self._cnn_count > 0:
            cnn_obs_tf = tf.convert_to_tensor(cnn_obs, dtype=tf.float32)
            cnn_next_obs_tf = tf.convert_to_tensor(cnn_next_obs, dtype=tf.float32)
        acts_tf = tf.convert_to_tensor(acts, dtype=tf.int32)
        rews_tf = tf.convert_to_tensor(rews, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)

        for _ in range(self._train_q_iters):
            with tf.GradientTape(persistent=True) as tape:
                encoder_loss, expert_loss = self.compute_q_loss(
                    mlp_obs_tf, cnn_obs_tf, acts_tf, rews_tf,
                    mlp_next_obs_tf, cnn_next_obs_tf, dones_tf
                )

            # Gather trainable variables
            expert_train_vars = []
            for m in self.ensemble_model.experts:
                expert_train_vars.extend(m.trainable_variables)

            expert_grads = tape.gradient(expert_loss, expert_train_vars)
            expert_grads, _ = tf.clip_by_global_norm(expert_grads, clip_norm=1.0)
            self._expert_optimizer.apply_gradients(zip(expert_grads, expert_train_vars))

            encoder_train_vars = self.ensemble_model.stack_encoder.trainable_variables
            encoder_grads = tape.gradient(encoder_loss, encoder_train_vars)
            encoder_grads, _ = tf.clip_by_global_norm(encoder_grads, clip_norm=1.0)
            self._encoder_optimizer.apply_gradients(zip(encoder_grads, encoder_train_vars))

            del tape

        if epoch_num % self._target_update_freq == 0:
            self.update_target_model()

        if epoch_num % self._expert_rotation_freq == 0:
            result = self.ensemble_model.rotate_expert_minibatch()

    def update_target_model(self):
        # Copy weights from Q_model to target_model
        for i, m in enumerate(self.ensemble_model.experts):
            self._target_model.experts[i].set_weights(m.get_weights())
        self._target_model.stack_encoder.set_weights(self.ensemble_model.stack_encoder.get_weights())
