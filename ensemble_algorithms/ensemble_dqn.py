import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

import numpy as np

from copy import deepcopy
import threading

from network_ensemble import MiniArchitectureEnsemble


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def load_dqn_params():



hyperparams, ensemble = load_dqn_params()

class EnsembleDQN():
    def __init__(self, 
                 env: gym.Env,
                 mlp_input_size, 
                 mlp_input_dim, 
                 cnn_input_size,
                 cnn_input_dim,
                 action_dim,
                 buffer_size, 
                 batch_size: int = hyperparams['batch_size'], 
                 seed: int = hyperparams['seed'], 
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], 
                 gamma: float = hyperparams['gamma'], 
                 epsilon: float = hyperparams['epsilon'], 
                 epsilon_min: float = hyperparams['epsilon_min'],
                 epsilon_decay: float = hyperparams['epsilon_decay'], 
                 q_lr: float = hyperparams['q_lr'], 
                 train_q_iters: int = hyperparams['train_q_iters'], 
                 train_update_freq: int = hyperparams['train_update_freq'], 
                 target_update_freq: int = hyperparams['target_update_freq'],
                 device: str = ensemble['device'],
                 kernel_size: int = ensemble['kernel_size'],
                 stride: int = ensemble['stride'],
                 padding: int = ensemble['padding'],
                 mlp_count: int = ensemble['mlp_count'],
                 cnn_count: int = ensemble['cnn_count'],
                 mlp_dropout: float = ensemble['mlp_dropout'],
                 cnn_dropout: float = ensemble['cnn_dropout'],
                 mlp_batch_size: int = ensemble['mlp_batch_size'],
                 cnn_batch_size: int = ensemble['cnn_batch_size'],
                 grad_dir: str = "./runtime_models" 
                 ):
        
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = env

        ### Hyperparameters
        self._mlp_input_size = mlp_input_size
        self._mlp_input_dim = mlp_input_dim
        self._cnn_input_size = cnn_input_size
        self._cnn_input_dim = cnn_input_dim
        self._action_dim = action_dim

        self._buffer_size = buffer_size
        
        self._batch_size = batch_size

        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._q_lr = q_lr

        self._traj_per_epoch = traj_per_epoch
        self._train_q_iters = train_q_iters
        self._train_update_freq = train_update_freq
        self._target_update_freq = target_update_freq

        ### NNs and Optimizers
        self._ensemble_model = MiniArchitectureEnsemble(mlp_input_dim, cnn_input_dim, action_dim, device, kernel_size,  
                                                        stride, padding, mlp_count, cnn_count, mlp_dropout, cnn_dropout, 
                                                        mlp_batch_size, cnn_batch_size, grad_dir)
        self._target_model = deepcopy(self._ensemble_model)

        self._model_optimizers = [optim.adam(model.parameters(), lr=q_lr) for model in self._ensemble_model.models]
        self._target_optimizers = [optim.adam(model.parameters(), lr=q_lr) for model in self._target_model.models]

        ### Replay buffer
        self.ptr, self.size, self.max_size = 0, 0, self._buffer_size

        self._mlp_obs_buffer = np.zeros(combined_shape(self._buffer_size, self._mlp_input_dim), dtype=np.float32)
        self._mlp_next_obs_buffer = np.zeroes(combined_shape(self._buffer_size, self._cnn_input_dim), dtype=np.float32)
        self._cnn_obs_buffer = np.zeros(combined_shape(self._buffer_size, self._cnn_input_dim), dtype=np.float32)
        self._cnn_next_obs_buffer = np.zeros(combined_shape(self._buffer_size, self._cnn_input_dim), dtype=np.float32)

        self._act_buffer = np.zeros(combined_shape(self._buffer_size), dtype=np.int32)
        self._rew_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._ret_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._q_val_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._done_buffer = np.zeroes(self._buffer_size, dtype=bool)
    
    def select_action(self, obs: torch.Tensor):
        self._epsilon = self._epsilon - self._epsilon_decay
        if np.random.rand() < self._epsilon:
            # choose 2nd most popular/probable
            return self.env.action_space.sample()
        else:
            # choose most popular/probable/argmax action
            q_val_list = []
            for model in self._ensemble_model.models:
                model.eval()
                with torch.no_grad():
                    q_vals = model.forward

    def step(self, obs: torch.Tensor):


    def train_n_epochs(self, n_epochs):
        # initialize replay buffer
        for epoch in n_epochs:

    def compute_loss(self):
