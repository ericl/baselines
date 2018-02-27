#!/usr/bin/env python

import multiprocessing
import sys

import ray
from ray.tune import register_trainable, run_experiments

from dqn_agent import DQNRLlibAgent

register_trainable("DQNBaseline", DQNRLlibAgent)

ray.init()

run_experiments({
    "baseline-rllib-cartpole": {
        "run": "DQNBaseline",
        "env": "CartPole-v0",
        "resources": {
            "cpu": 1,
        },
        "config": {
            "num_workers": 0,
            "env_config": {"cartpole": True},
            "apex": False,
            "lr": .0005,
            "n_step": 1,
            "gamma": 0.99,
            "sample_batch_size": 1,
            "train_batch_size": 32,
            "force_remote_evaluators": False,
            "learning_starts": 1000,
            "buffer_size": 10000,
            "target_network_update_freq": 500,
            "timesteps_per_iteration": 500,
        },
    },
})
