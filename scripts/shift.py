from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import ShiftHindsightWrapper
from environments.shift import ShiftEnv
from sac.networks import MlpAgent, MoEAgent, SACXAgent
from sac.train import ShiftHindsightTrainer, ShiftTrainer
from scripts.pick_and_place import env_wrapper, parse_double, put_in_xml_setter


def parse_coordinate(ctx, param, string):
    if string is None:
        return
    return np.array(list(map(float, string.split(','))))


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--n-networks', default=None, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--mlp', 'agent', flag_value=MlpAgent, default=True)
@click.option('--sacx', 'agent', flag_value=SACXAgent)
@click.option('--relu', 'activation', flag_value=tf.nn.relu, default=True)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=1e-4, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--steps-per-action', default=200, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=7e3, type=float)
@click.option('--entropy-scale', default=1, type=float)
@click.option('--max-steps', default=200, type=int)
@click.option('--n-goals', default=1, type=int)
@click.option('--grad-clip', default=2e4, type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render-freq', default=0, type=int)
@click.option('--render', is_flag=True)
@click.option('--record-freq', type=int, default=0)
@click.option('--record-path', type=Path)
@click.option('--image-dims', type=str, callback=parse_double)
@click.option('--record', is_flag=True)
@click.option('--eval', is_flag=True)
@click.option('--randomize-pose', is_flag=True)
@click.option('--set-xml', multiple=True, callback=put_in_xml_setter)
@click.option('--geofence', default=.25, type=float)
@click.option('--hindsight-geofence', default=None, type=float)
@click.option('--fixed-block', default=None, callback=parse_coordinate)
@click.option('--fixed-goal', default=None, callback=parse_coordinate)
@click.option('--goal-x', default=None, callback=parse_coordinate)
@click.option('--goal-y', default=None, callback=parse_coordinate)
@click.option('--xml-file', type=Path, default='world.xml')
@click.option(
    '--use-dof',
    multiple=True,
    default=[
        'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint', 'wrist_roll_joint',
        'hand_l_proximal_joint', 'hand_r_proximal_joint', 'goal_x', 'goal_y'
    ])
@env_wrapper
def cli(max_steps, seed, device_num, buffer_size, activation, n_layers, layer_size,
        learning_rate, reward_scale, entropy_scale, grad_clip, batch_size,
        num_train_steps, steps_per_action, logdir, save_path, load_path, n_goals, eval,
        temp_path, render_freq, record, record_path, record_freq, image_dims,
        hindsight_geofence, geofence, n_networks, agent, fixed_block, fixed_goal,
        randomize_pose, goal_x, goal_y):
    env = TimeLimit(
        max_episode_steps=max_steps,
        env=ShiftEnv(
            geofence=geofence,
            xml_filepath=temp_path,
            steps_per_action=steps_per_action,
            render_freq=render_freq,
            record=record,
            record_path=record_path,
            record_freq=record_freq,
            image_dimensions=image_dims,
            fixed_block=fixed_block,
            fixed_goal=fixed_goal,
            randomize_pose=randomize_pose,
            goal_x=goal_x,
            goal_y=goal_y,
        ))
    kwargs = dict(
        base_agent=agent,
        seq_len=None,
        seed=seed,
        device_num=device_num,
        buffer_size=buffer_size,
        activation=activation,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        reward_scale=reward_scale,
        entropy_scale=entropy_scale,
        grad_clip=grad_clip if grad_clip > 0 else None,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        logdir=logdir,
        save_path=save_path,
        load_path=load_path,
        render=False,  # because render is handled inside env
        evaluation=eval,
    )
    if n_networks:
        kwargs['base_agent'] = MoEAgent
        kwargs['n_networks'] = n_networks

    if hindsight_geofence:
        env = ShiftHindsightWrapper(env=env, geofence=hindsight_geofence)
        ShiftHindsightTrainer(env=env, n_goals=n_goals, **kwargs)
    else:
        ShiftTrainer(env=env, **kwargs)


if __name__ == '__main__':
    cli()
