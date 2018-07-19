import click
import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.frozen_lake import FrozenLakeEnv
from sac.networks import MlpAgent, MoEAgent
from sac.train import Trainer


def check_probability(ctx, param, value):
    if not (0 <= value <= 1):
        raise click.BadParameter("Param {} should be between 0 and 1".format(value))
    return value


def parse_double(ctx, param, string):
    a, b = map(int, string.split('x'))
    return a, b


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--n-layers', default=2, type=int)
@click.option('--layer-size', default=128, type=int)
@click.option('--learning-rate', default=1e-4, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=1, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=1, type=float)
@click.option('--entropy-scale', default=5e-6, type=float)
@click.option('--grad-clip', default=2e-2, type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--map-dims', default="4x4", type=str, callback=parse_double)
@click.option('--random-map', is_flag=True)
@click.option('--random-start', is_flag=True)
@click.option('--random-goal', is_flag=True)
@click.option('--is-slippery', is_flag=True)
@click.option('--max-steps', default=100, type=int)
@click.option('--render', is_flag=True)
@click.option('--n-networks', default=None, type=int)
def cli(seed, buffer_size, n_layers, layer_size, learning_rate,
        entropy_scale, reward_scale, batch_size,
        num_train_steps, logdir, save_path, load_path, render, grad_clip, map_dims,
        max_steps, n_networks, random_map, random_start, random_goal, is_slippery):
    env = TimeLimit(
        env=FrozenLakeEnv(
            map_dims=map_dims,
            random_map=random_map,
            random_start=random_start,
            random_goal=random_goal,
            is_slippery=is_slippery),
        max_episode_steps=max_steps)
    kwargs = dict(
        env=env,
        seq_len=0,
        device_num=1,
        seed=seed,
        buffer_size=buffer_size,
        activation=tf.nn.relu,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        entropy_scale=entropy_scale,
        reward_scale=reward_scale,
        batch_size=batch_size,
        grad_clip=grad_clip,
        num_train_steps=num_train_steps,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=render)
    if n_networks:
        Trainer(base_agent=MoEAgent, n_networks=n_networks, **kwargs)
    else:
        Trainer(base_agent=MlpAgent, **kwargs)


if __name__ == '__main__':
    cli()
