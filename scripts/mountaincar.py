import click
import gym
import tensorflow as tf
from gym.envs.classic_control import MountainCarEnv
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import MountaincarHindsightWrapper
from sac.networks import MlpAgent
from sac.train import HindsightTrainer, Trainer


class MountCarEnv(MountainCarEnv):
    def step(self, a):
        s, r, t, i = super().step(a)
        return s, max(0, r), t, i


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--relu', 'activation', flag_value=tf.nn.relu, default=True)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=3e-4, type=float)
@click.option('--buffer-size', default=1e7, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=1e4, type=float)
@click.option('--n-goals', default=1, type=int)
@click.option('--grad-clip', default=2e4, type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
@click.option('--no-hindsight', is_flag=True)
def cli(seed, device_num, buffer_size, activation, n_layers, layer_size, learning_rate,
        reward_scale, grad_clip, batch_size, num_train_steps, logdir, save_path,
        load_path, render, n_goals, no_hindsight):
    env = TimeLimit(MountCarEnv(), max_episode_steps=200)
    kwargs = dict(
        seq_len=None,
        base_agent=MlpAgent,
        seed=seed,
        device_num=device_num,
        buffer_size=buffer_size,
        activation=activation,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        reward_scale=reward_scale,
        grad_clip=grad_clip if grad_clip > 0 else None,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=render)  # because render is handled inside env
    hindsight = not no_hindsight
    if hindsight:
        HindsightTrainer(env=MountaincarHindsightWrapper(env),
                         n_goals=n_goals, **kwargs)
    else:
        Trainer(env=env, **kwargs)


if __name__ == '__main__':
    cli()
