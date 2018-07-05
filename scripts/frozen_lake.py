import click
import gym
import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.frozen_lake import FrozenLakeEnv
from sac.networks import MlpAgent
from sac.train import Trainer


def check_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter("Param {} should be between 0 and 1".format(value))
    return value


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--n-layers', default=2, type=int)
@click.option('--layer-size', default=128, type=int)
@click.option('--learning-rate', default=9e-5, type=float)
@click.option('--buffer-size', default=1e4, type=int)
@click.option('--num-train-steps', default=1, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=1., type=float)
@click.option('--grad-clip', default=2e3, type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--map-name', default="4x4", type=str)
@click.option('--max-steps', default=100, type=int)
@click.option('--render', is_flag=True)
def cli(seed, buffer_size, n_layers, layer_size, learning_rate, reward_scale,
        batch_size, num_train_steps, logdir, save_path, load_path, render,
        grad_clip, map_name, max_steps):
    env = TimeLimit(
        env=FrozenLakeEnv(map_name=map_name, is_slippery=False),
        max_episode_steps=max_steps
    )
    Trainer(
        env=env,
        base_agent=MlpAgent,
        seq_len=0,
        device_num=1,
        seed=seed,
        buffer_size=buffer_size,
        activation=tf.nn.relu,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        reward_scale=reward_scale,
        batch_size=batch_size,
        grad_clip=grad_clip,
        num_train_steps=num_train_steps,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=render)


if __name__ == '__main__':
    cli()
