from ensemble_algorithms import ensemble_dqn, ensemble_ppo
from gym_environments import create_env

if __name__ == "__main__":
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument("--algorithm", default="PPO")
    parser.add_argument("--epochs", default=10000)
    parser.add_argument("--env_name", default="")