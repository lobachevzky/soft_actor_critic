import click
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import PickAndPlaceHindsightWrapper
from environments.pick_and_place import PickAndPlaceEnv
from sac.train import HindsightPropagationTrainer, HindsightTrainer, TrajectoryTrainer, \
    DoubleBufferHindsightTrainer, SimpleHindsightTrainer
from scripts.gym_env import check_probability, str_to_activation



@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--activation', default='relu', callback=str_to_activation)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=2e-4, type=float)
@click.option('--buffer-size', default=1e7, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=9e3, type=float)
@click.option('--cheat-prob', default=0, type=float, callback=check_probability)
@click.option('--max-steps', default=500, type=int)
@click.option('--n-goals', default=1, type=int)
@click.option('--geofence', default=.4, type=float)
@click.option('--min-lift-height', default=.02, type=float)
@click.option('--default-reward', default=0, type=float)
@click.option('--grad-clip', default=4e4, type=float)
@click.option('--fixed-block', is_flag=True)
@click.option('--hindsight', 'trainer', flag_value=HindsightTrainer, default=True)
@click.option('--reward-prop', 'trainer', flag_value=HindsightPropagationTrainer)
@click.option('--double-buffer', 'trainer', flag_value=DoubleBufferHindsightTrainer)
@click.option('--simple-hindsight', 'trainer', flag_value=SimpleHindsightTrainer)
@click.option('--discrete', is_flag=True)
@click.option('--mimic-dir',  default=None, type=str)
@click.option('--mimic-save-dir',  default=None, type=str)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
def cli(trainer: TrajectoryTrainer.__class__, default_reward, max_steps, discrete, fixed_block, min_lift_height,
        geofence, seed, buffer_size, activation, n_layers, layer_size, learning_rate,
        reward_scale, cheat_prob, grad_clip, batch_size, num_train_steps,
        mimic_dir, mimic_save_dir, logdir, save_path, load_path, render, n_goals):

    print('Using', trainer.__name__)

    trainer(
        env=PickAndPlaceHindsightWrapper(
            default_reward=default_reward,
            env=TimeLimit(
                max_episode_steps=max_steps,
                env=PickAndPlaceEnv(
                    discrete=discrete,
                    cheat_prob=cheat_prob,
                    fixed_block=fixed_block,
                    min_lift_height=min_lift_height,
                    geofence=geofence))),
        seed=seed,
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
