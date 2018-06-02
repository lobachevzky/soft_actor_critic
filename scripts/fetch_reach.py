import argparse

import click
import gym
import numpy as np
import tensorflow as tf
from gym.envs.robotics import FetchReachEnv
from gym.envs.robotics.fetch_env import goal_distance

from environments.hindsight_wrapper import HindsightWrapper, State
from sac.train import HindsightTrainer
from scripts.gym_env import str_to_activation, check_probability

ACHIEVED_GOAL = 'achieved_goal'


class FetchReachHindsightWrapper(HindsightWrapper):
    def __init__(self, env):
        assert isinstance(env.unwrapped, FetchReachEnv)
        super().__init__(env)

    def step(self, action):
        s2, r, t, info = self.env.step(action)
        new_s2 = State(**s2)
        r = info['is_success']
        return new_s2, r, t, info

    def reset(self):
        s2 = self.env.reset()
        return State(**s2)

    def _achieved_goal(self):
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        return goal_distance(achieved_goal,
                             desired_goal) < self.env.unwrapped.distance_threshold

    def _desired_goal(self):
        return self.env.unwrapped.goal.copy()

    @staticmethod
    def vectorize_state(states):
        if isinstance(states, State):
            states = [states]
        return np.stack([np.concatenate([
            state.desired_goal,
            state.observation
        ]) for state in states])


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--activation', default='relu', callback=str_to_activation)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=2e-4, type=float)
@click.option('--buffer-size', default=1e7, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=7e3, type=float)
@click.option('--cheat-prob', default=0, type=float, callback=check_probability)
@click.option('--max-steps', default=200, type=int)
@click.option('--n-goals', default=1, type=int)
@click.option('--geofence', default=.4, type=float)
@click.option('--min-lift-height', default=.02, type=float)
@click.option('--grad-clip', default=4e4, type=float)
@click.option('--fixed-block', is_flag=True)
@click.option('--discrete', is_flag=True)
@click.option('--mimic-dir',  default=None, type=str)
@click.option('--mimic-save-dir',  default=None, type=str)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
def cli(max_steps, discrete, fixed_block,
        min_lift_height, geofence, seed, device_num, buffer_size, activation, n_layers, layer_size,
        learning_rate, reward_scale, cheat_prob, grad_clip, batch_size, num_train_steps,
        mimic_dir, mimic_save_dir, logdir, save_path, load_path, render, n_goals):
    HindsightTrainer(
        env=FetchReachHindsightWrapper(gym.make('FetchReach-v0')),
        seed=seed,
        device_num=device_num,
        n_goals=n_goals,
        buffer_size=buffer_size,
        activation=activation,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        reward_scale=reward_scale,
        grad_clip=grad_clip if grad_clip > 0 else None,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        mimic_dir=mimic_dir,
        mimic_save_dir=mimic_save_dir,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=render)
