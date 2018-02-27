import os

import numpy as np

from ray.rllib.dqn.dqn_evaluator import adjust_nstep
from ray.rllib.dqn.common.atari_wrappers import wrap_deepmind
from ray.rllib.optimizers.evaluator import Evaluator
from ray.rllib.optimizers.sample_batch import SampleBatch, pack

import tensorflow as tf
import models
from simple import ActWrapper
from build_graph import build_train
from utils import BatchInput


class DQNEvaluator(Evaluator):
    
    def __init__(self, config, env_creator):
        self.config = config
        self.local_timestep = 0
        self.episode_rewards = [0.0]
        self.episode_lengths = [0.0]

        self.env = env_creator(self.config["env_config"])
        self.env = wrap_deepmind(self.env, frame_stack=True, scale=True)
        self.obs = self.env.reset()

        self.sess = tf.Session()
        self.sess.__enter__()

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph
        observation_space_shape = self.env.observation_space.shape
        def make_obs_ph(name):
            return BatchInput(observation_space_shape, name=name)

        q_func = models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=True,
        )
        self.model = q_func

        act, train, self.update_target, debug = build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=self.env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.config["lr"]),
            gamma=self.config["gamma"],
            grad_norm_clipping=10,
            param_noise=False
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': env.action_space.n,
        }

        self.act = ActWrapper(act, act_params)

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(
            schedule_timesteps=int(self.config["exploration_fraction"] * self.config["exploration_max_timesteps"]),
            initial_p=1.0,
            final_p=self.config["exploration_final_eps"])

    def sample(self):
        obs, actions, rewards, new_obs, dones = [], [], [], [], []
        for _ in range(
                self.config["sample_batch_size"] + self.config["n_step"] - 1):
            update_eps = self.exploration.value(self.local_timestep)
            action = self.act(
                np.array(self.obs)[None], update_eps=update_eps)[0]
            new_obs, reward, done, _ = self.env.step(action)
            obs.append(self.obs)
            actions.append(action)
            rewards.append(reward)
            new_obs.append(new_obs)
            dones.append(1.0 if done else 0.0)
            self.obs = new_obs
            self.episode_rewards[-1] += reward
            self.episode_lengths[-1] += 1
            if done:
                self.obs = self.env.reset()
                self.episode_rewards.append(0.0)
                self.episode_lengths.append(0.0)
            self.local_timestep += 1

        # N-step Q adjustments
        if self.config["n_step"] > 1:
            # Adjust for steps lost from truncation
            self.local_timestep -= (self.config["n_step"] - 1)
            adjust_nstep(
                self.config["n_step"], self.config["gamma"],
                obs, actions, rewards, new_obs, dones)

        batch = SampleBatch({
            "obs": obs, "actions": actions, "rewards": rewards,
            "new_obs": new_obs, "dones": dones,
            "weights": np.ones_like(rewards)})
        assert batch.count == self.config["sample_batch_size"]

#        td_errors = self.agent.compute_td_error(batch)
        batch.data["obs"] = [pack(o) for o in batch["obs"]]
        batch.data["new_obs"] = [pack(o) for o in batch["new_obs"]]
#        new_priorities = (
#            np.abs(td_errors) + self.config["prioritized_replay_eps"])
#        batch.data["weights"] = new_priorities

        return batch

    def compute_gradients(self, samples):
        raise NotImplementedError

    def apply_gradients(self, grads):
        raise NotImplementedError

    def compute_apply(self, samples):
        return self.train(
            samples["obs"], samples["actions"], samples["rewards"],
            samples["new_obs"], samples["dones"], samples["weights"])

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def stats(self):
        mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 5)
        mean_100ep_length = round(np.mean(self.episode_lengths[-101:-1]), 5)
        return {
            "mean_100ep_reward": mean_100ep_reward,
            "mean_100ep_length": mean_100ep_length,
            "num_episodes": len(self.episode_rewards),
            "local_timestep": self.local_timestep,
        }

    def update_target(self):
        self.update_target()
