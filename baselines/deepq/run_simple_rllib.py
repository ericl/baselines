#!/usr/bin/env python

import multiprocessing
import sys

import ray
from ray.tune import register_trainable, run_experiments

from dqn_agent import DQNRLlibAgent

register_trainable("DQNBaseline", DQNRLlibAgent)

ray.init(num_gpus=1)

run_experiments({
    "baseline-rllib-pong": {
        "run": "DQNBaseline",
        "env": "PongNoFrameskip-v4",
        "resources": {
            "cpu": 1,
            "gpu": 1,
        },
        "config": {
            "num_workers": 0,
            "apex": False,
            "lr": .0001,
            "n_step": 1,
            "gamma": 0.99,
            "sample_batch_size": 4,
            "train_batch_size": 32,
            "force_remote_evaluators": False,
            "learning_starts": 10000,
            "buffer_size": 10000,
            "target_network_update_freq": 1000,
            "timesteps_per_iteration": 1000,
        },
    },
})
