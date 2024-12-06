import tensorflow as tf
import gymnasium as gym
from utils import load_hyperparams

ensemble, hyperparams = load_hyperparams('ddqn')


class EnsembleDDQN(tf.Module):
    def __init__(self,
                 env: gym.Env,
                 mlp_input_size,
                 mlp_input_dim,
                 cnn_input_size,
                 cnn_input_dim,
                 mlp_activations,
                 cnn_activations,
                 action_dim,
                 buffer_size,
                 ):
        super().__init__()

    def store_transition(self):

    def get_buffer_batch(self):

    def step(self):

    def compute_q_loss(self):

    def train_models(self, epoch_num):
        