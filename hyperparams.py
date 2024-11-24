{
    "MiniArchEnsemble": {
        "device": "cpu",
        "kernel_size": 32,
        "stride": 3,
        "padding": 3,
        "mlp_count": 12,
        "cnn_count": 12,
        "mlp_dropout": .33,
        "cnn_dropout": .33,
        "mlp_batch_size": 6,
        "cnn_batch_size": 4,
    },
    "DQN": {
        "batch_size": 32,
        "seed": 42,
        "traj_per_epoch": 16,
        "gamma": .98,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.01,
        "q_lr": .0003,
        "train_q_iters": 80,
        "train_update_freq": 16,
        "target_update_freq": 16,
    },
    "PPO": {

    }
}