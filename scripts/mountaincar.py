import click
import gym
import tensorflow as tf

from environments.hindsight_wrapper import MountaincarHindsightWrapper
from sac.train import HindsightTrainer


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--render', is_flag=True)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
def cli(seed, logdir, save_path, load_path, render, device_num):
    HindsightTrainer(
        env=MountaincarHindsightWrapper(
            gym.make('MountainCarContinuous-v0')),
        seed=seed,
        buffer_size=10**7,
        activation=tf.nn.relu,
        n_layers=3,
        layer_size=256,
        learning_rate=3e-4,
        grad_clip=None,
        reward_scale=1e3,
        batch_size=32,
        num_train_steps=1,
        n_goals=1,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        mimic_dir=None,
        mimic_save_dir=None,
        device_num=device_num,
        render=render)
