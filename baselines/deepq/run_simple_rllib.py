#!/usr/bin/env python

import multiprocessing
import sys

import ray
from ray.tune import register_trainable, run_experiments

from dqn_rllib_agent import DQNRLlibAgent

register_trainable("DQNBaseline", DQNRLlibAgent)
ray.init()

run_experiments({
    "cartpole": {
        "run": "DQNBaseline",
        "env": "CartPole-v0",
        "resources": {
            "cpu": 1,
        },
        "config": {
            "sample_batch_size": 1,
            "apex": False,
        },
    },
})
