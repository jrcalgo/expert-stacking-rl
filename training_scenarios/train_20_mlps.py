from ..ensemble_training_factory import TrainingFactory

if __name__ == "__main__":
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument("--algorithm_name", default="DQN",
                        help="The algorithm to train. Options: DQN, PPO")
    parser.add_argument("--env_name", default="LunarLander-v2")
    parser.add_argument("--env_seed", default=0)
    parser.add_argument("--train_iters", default=10000)
    parser.add_argument("--max_epochs", default=100000)
    parser.add_argument("--replay_buffer_size", default=1000)
    parser.add_argument("--mlp_count", default=20)
    parser.add_argument("--cnn_count", default=0)
    parser.add_argument("--mlp_batch_size", default=18)
    parser.add_argument("--cnn_batch_size", default=0)
    parser.add_argument("--expert_rotation_freq", default=8)
    args = parser.parse_args()

    mlp_activations = [['relu', 'sigmoid'] for _ in range(8)]
    mlp_activations += [['sigmoid', 'relu'] for _ in range(8)]
    mlp_activations += [[['relu', 'relu'], ['sigmoid', 'sigmoid']] for _ in range(2)]

    cnn_activations = [['relu', 'relu', 'relu'] for _ in range(args.cnn_count)]

    hyperparams_dict = {
        'mlp_count': args.mlp_count,
        'cnn_count': args.cnn_count,
        'mlp_batch_size': args.mlp_batch_size,
        'cnn_batch_size': args.cnn_batch_size,
        'expert_rotation_freq': args.expert_rotation_freq
    }

    # Initialize the training factory
    training = TrainingFactory(args.algorithm_name, args.env_name, args.env_seed, args.train_iters, args.max_epochs,
                               args.replay_buffer_size, mlp_activations, cnn_activations, hyperparams_dict)
    training.simulate_max_epochs()
