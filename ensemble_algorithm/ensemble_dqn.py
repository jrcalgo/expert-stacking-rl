import torch
import numpy as np

from network_ensemble import MiniArchitectureEnsemble

class EnsembleDQN():
    def __init__(self, mlp_input_size, 
                 mlp_input_dim, 
                 cnn_input_size,
                 cnn_input_dim,
                 action_dim, 
                 buffer_size, 
                 batch_size, 
                 seed, 
                 traj_per_epoch, 
                 gamma, 
                 epsilon, 
                 epsilon_min,
                 epsilon_decay, 
                 q_lr, 
                 train_q_iters, 
                 train_update_freq, 
                 target_update_freq,
                 device,
                 model_count,
                 mlp_dropout,
                 cnn_dropout,
                 model_batch_size,
                 grad_dir):
        
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

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

        self._ensemble_model = MiniArchitectureEnsemble(mlp_input_dim, cnn_input_dim, action_dim, device, model_count, mlp_dropout, cnn_dropout, model_batch_size, grad_dir)

        ### Replay buffer

        self._obs_buffer
        self._next_obs_buffer
        self._act_buffer
        self._rew_buffer
        self._done_buffer


    def append_trajectory(self):


    def train(self):


    def compute_loss(self):

