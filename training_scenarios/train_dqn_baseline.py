from baseline_training_factory import TrainingFactory

if __name__ == "__main__":
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument("--algorithm_name", default="DQN",
                        help="The algorithm to train. Options: DQN")
    parser.add_argument("--env_name", default="LunarLander-v2")
    parser.add_argument("--env_seed", default=0)
    parser.add_argument("--train_iters", default=10000)
    parser.add_argument("--max_epochs", default=100000)
    parser.add_argument("--replay_buffer_size", default=10000)
    args = parser.parse_args()

    # Initialize the training factory
    training = TrainingFactory(args.algorithm_name, args.env_name, args.env_seed, args.train_iters, args.max_epochs,
                               args.replay_buffer_size)
    training.simulate_max_epochs()
