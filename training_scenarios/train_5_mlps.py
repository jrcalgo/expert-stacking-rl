from ensemble_training_factory import TrainingFactory

if __name__ == "__main__":
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument("--algorithm_name", default="REINFORCE",
                        help="The algorithm to train. Options: DQN, REINFORCE")
    parser.add_argument("--env_name", default="LunarLander-v2")
    parser.add_argument("--env_seed", default=0)
    parser.add_argument("--train_iters", default=10000)
    parser.add_argument("--max_epochs", default=100000)
    parser.add_argument("--replay_buffer_size", default=100000)
    parser.add_argument("--mlp_count", default=5)
    parser.add_argument("--cnn_count", default=0)
    parser.add_argument("--mlp_batch_size", default=3)
    parser.add_argument("--cnn_batch_size", default=0)
    parser.add_argument("--expert_rotation_freq", default=10)
    parser.add_argument("--log_tensorboard", default=True)
    args = parser.parse_args()

    mlp_activations = [['relu', 'relu'] for _ in range(5)]

    hyperparams_dict = {
        'mlp_count': args.mlp_count,
        'cnn_count': args.cnn_count,
        'mlp_batch_size': args.mlp_batch_size,
        'cnn_batch_size': args.cnn_batch_size,
        'expert_rotation_freq': args.expert_rotation_freq
    }

    # Initialize the training factory
    training = TrainingFactory(args.algorithm_name, args.env_name, args.env_seed, args.train_iters, args.max_epochs,
                               args.replay_buffer_size, mlp_activations, None, args.log_tensorboard, hyperparams_dict)
    training.simulate_max_epochs()
