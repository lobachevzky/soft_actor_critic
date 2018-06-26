from pathlib import Path

import click
import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.multi_task import MultiTaskEnv
from environments.old_hindsight_wrapper import PickAndPlaceHindsightWrapper
from sac.train import MultiTaskHindsightTrainer
from scripts.pick_and_place import mutate_xml, parse_range, put_in_xml_setter


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--relu', 'activation', flag_value=tf.nn.relu, default=True)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=3e-4, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--steps-per-action', default=200, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=7e3, type=float)
@click.option('--max-steps', default=300, type=int)
@click.option('--n-goals', default=1, type=int)
@click.option('--geofence', default=.01, type=float)
@click.option('--min-lift-height', default=.02, type=float)
@click.option('--grad-clip', default=2e4, type=float)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render-freq', default=0, type=int)
@click.option('--eval', is_flag=True)
@click.option('--no-qvel', 'obs_type', flag_value='no-qvel')
@click.option('--add-base-qvel', 'obs_type', flag_value='base-qvel', default=True)
@click.option('--block-xrange', type=str, default="-.1,.1", callback=parse_range)
@click.option('--block-yrange', type=str, default="-.2,.2", callback=parse_range)
@click.option('--set-xml', multiple=True, callback=put_in_xml_setter)
@click.option('--use-dof', multiple=True, default=['slide_x',
                                                   'slide_y',
                                                   'arm_lift_joint',
                                                   'arm_flex_joint',
                                                   'wrist_roll_joint',
                                                   'hand_l_proximal_joint',
                                                   'hand_r_proximal_joint'])
def cli(max_steps, geofence, min_lift_height, seed, device_num, buffer_size, activation,
        n_layers, layer_size, learning_rate, reward_scale, grad_clip, batch_size,
        num_train_steps, steps_per_action, logdir, save_path, load_path, render_freq,
        n_goals, eval, set_xml, use_dof, block_xrange, block_yrange, obs_type):
    xml_filepath = Path(Path(__file__).parent.parent, 'environments', 'models', 'world.xml')
    with mutate_xml(changes=set_xml, dofs=use_dof, xml_filepath=xml_filepath) as temp_path:
        env = PickAndPlaceHindsightWrapper(
            env=TimeLimit(
                max_episode_steps=max_steps,
                env=MultiTaskEnv(steps_per_action=steps_per_action,
                                 min_lift_height=min_lift_height,
                                 obs_type=obs_type,
                                 geofence=geofence,
                                 render_freq=render_freq,
                                 xml_filepath=temp_path,
                                 block_xrange=block_xrange,
                                 block_yrange=block_yrange,
                                 )))
    MultiTaskHindsightTrainer(
        env=env,
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
        render=False,  # because render is handled inside env
        evaluation=eval,
    )


if __name__ == '__main__':
    cli()
