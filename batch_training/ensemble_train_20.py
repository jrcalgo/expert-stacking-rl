from ensemble_algorithms import ensemble_dqn, ensemble_ppo
from gym_environments import create_env

if __name__ == "__main__":
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument("--algorithm", default="DQN",
                        help="The algorithm to train. Options: DQN, PPO")
    parser.add_argument("--epochs", default=10000)
    parser.add_argument("--env_name", default="")
    parser.add_argument("--ensemble-method", defaul="stacking",
                        help="The ensemble method to use. Options: stacking, bagging")
    parser.add_argument("--ensemble-mlp-count", default=10, type=int)
    parser.add_argument("--ensemble-cnn-count", default=10, type=int)