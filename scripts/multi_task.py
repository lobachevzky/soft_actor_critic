import click
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import MultiTaskHindsightWrapper, HindsightWrapper, PickAndPlaceHindsightWrapper
from environments.multi_task import MultiTaskEnv
from sac.train import HindsightTrainer
from scripts.gym_env import check_probability, str_to_activation


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--activation', default='relu', callback=str_to_activation)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=3e-4, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--steps-per-action', default=200, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=7e3, type=float)
@click.option('--cheat-prob', default=0, type=float, callback=check_probability)
@click.option('--max-steps', default=300, type=int)
@click.option('--n-goals', default=1, type=int)
@click.option('--geofence', default=.1, type=float)
@click.option('--min-lift-height', default=.02, type=float)
@click.option('--grad-clip', default=2e4, type=float)
@click.option('--mimic-dir', default=None, type=str)
@click.option('--mimic-save-dir', default=None, type=str)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render', is_flag=True)
@click.option('--baseline', is_flag=True)
def cli(max_steps, geofence, min_lift_height, seed, device_num,
        buffer_size, activation, n_layers, layer_size, learning_rate, reward_scale,
        cheat_prob, grad_clip, batch_size, num_train_steps, steps_per_action, mimic_dir,
        mimic_save_dir, logdir, save_path, load_path, render, n_goals, baseline):

    wrapper = PickAndPlaceHindsightWrapper if baseline else MultiTaskHindsightWrapper
    HindsightTrainer(
        env=wrapper(
            env=TimeLimit(
                max_episode_steps=max_steps,
                env=MultiTaskEnv(
                    steps_per_action=steps_per_action,
                    geofence=geofence,
                    min_lift_height=min_lift_height,
                    render=render))),
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
        render=False)


if __name__ == '__main__':
    cli()
