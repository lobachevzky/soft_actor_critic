import click
import gym

from environments.old_hindsight_wrapper import MountaincarHindsightWrapper
from sac.train import HindsightTrainer
import tensorflow as tf


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
@click.option('--mimic-dir', default=None, type=str)
@click.option('--mimic-save-dir', default=None, type=str)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
def cli(seed, device_num, buffer_size, activation, n_layers, layer_size, learning_rate,
        reward_scale, grad_clip, batch_size, num_train_steps, mimic_dir, mimic_save_dir,
        logdir, save_path, load_path, render, n_goals):

    HindsightTrainer(
        env=MountaincarHindsightWrapper(gym.make('MountainCarContinuous-v0')),
        mimic_dir=None,
        mimic_save_dir=None,
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
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=render)


if __name__ == '__main__':
    cli()
