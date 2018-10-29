import argparse
from pathlib import Path

import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import HSRHindsightWrapper
from environments.hsr import HSREnv
from sac.networks import MlpAgent
from sac.unsupervised_trainer import UnsupervisedTrainer
from sac.utils import create_sess
from scripts.hsr import (cast_to_int, env_wrapper, parse_space, parse_vector,
                         put_in_xml_setter)


@env_wrapper
def main(worker_n_layers, worker_layer_size, worker_learning_rate, worker_entropy_scale,
         worker_reward_scale, worker_num_train_steps, worker_grad_clip, steps_per_action,
         worker_batch_size, worker_buffer_size, boss_n_layers, boss_layer_size,
         boss_learning_rate, boss_entropy_scale, boss_reward_scale, boss_num_train_steps,
         boss_grad_clip, boss_buffer_size, boss_batch_size, max_steps, min_lift_height,
         geofence, hindsight_geofence, seed, goal_space, block_space, concat_record,
         logdir, save_path, load_path, worker_load_path, render_freq, render, n_goals,
         record, randomize_pose, image_dims, record_freq, record_path, temp_path,
         freeze_worker):
    env = HSRHindsightWrapper(
        geofence=hindsight_geofence or geofence,
        env=TimeLimit(
            max_episode_steps=max_steps,
            env=HSREnv(
                steps_per_action=steps_per_action,
                randomize_pose=randomize_pose,
                min_lift_height=min_lift_height,
                xml_filepath=temp_path,
                block_space=block_space,
                goal_space=goal_space,
                geofence=geofence,
                render=render,
                render_freq=render_freq,
                record=record,
                record_path=record_path,
                record_freq=record_freq,
                record_separate_episodes=concat_record,
                image_dimensions=image_dims,
            )))

    worker_kwargs = dict(
        n_layers=worker_n_layers,
        layer_size=worker_layer_size,
        learning_rate=worker_learning_rate,
        entropy_scale=worker_entropy_scale,
        reward_scale=worker_reward_scale,
        grad_clip=worker_grad_clip,
        num_train_steps=worker_num_train_steps,
        batch_size=worker_batch_size,
        buffer_size=worker_buffer_size,
    )

    boss_kwargs = dict(
        n_layers=boss_n_layers,
        layer_size=boss_layer_size,
        learning_rate=boss_learning_rate,
        entropy_scale=boss_entropy_scale,
        reward_scale=boss_reward_scale,
        grad_clip=boss_grad_clip,
        num_train_steps=boss_num_train_steps,
        batch_size=boss_batch_size,
        buffer_size=boss_buffer_size,
    )

    UnsupervisedTrainer(
        env=env,
        sess=create_sess(),
        n_goals=n_goals,
        seq_len=None,
        base_agent=MlpAgent,
        seed=seed,
        activation=tf.nn.relu,
        worker_load_path=worker_load_path,
        worker_kwargs=worker_kwargs,
        update_worker=not freeze_worker,
        boss_kwargs=boss_kwargs).train(
            load_path=load_path,
            logdir=logdir,
            render=False,
            save_path=save_path,
            save_threshold=None,
        )


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--boss-n-layers', type=int, required=True)
    p.add_argument('--boss-layer-size', type=int, required=True)
    p.add_argument('--boss-buffer-size', type=cast_to_int, required=True)
    p.add_argument('--boss-num-train-steps', type=int, required=True)
    p.add_argument('--boss-batch-size', type=int, required=True)
    p.add_argument('--boss-learning-rate', type=float, required=True)
    p.add_argument('--boss-grad-clip', type=float, required=True)
    scales = p.add_mutually_exclusive_group(required=True)
    scales.add_argument('--boss-reward-scale', type=float, default=1)
    scales.add_argument('--boss-entropy-scale', type=float, default=1)
    p.add_argument('--worker-n-layers', type=int, required=True)
    p.add_argument('--worker-layer-size', type=int, required=True)
    p.add_argument('--worker-buffer-size', type=cast_to_int, required=True)
    p.add_argument('--worker-num-train-steps', type=int, required=True)
    p.add_argument('--worker-batch-size', type=int, required=True)
    p.add_argument('--worker-learning-rate', type=float, required=True)
    p.add_argument('--worker-grad-clip', type=float, required=True)
    p.add_argument('--worker-load-path', type=str, default=None)
    scales = p.add_mutually_exclusive_group(required=True)
    scales.add_argument('--worker-reward-scale', type=float, default=1)
    scales.add_argument('--worker-entropy-scale', type=float, default=1)
    p.add_argument('--freeze-worker', action='store_true')
    p.add_argument('--steps-per-action', type=int, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--n-goals', type=int, default=None)
    p.add_argument('--n-blocks', type=int, required=True)
    p.add_argument('--min-lift-height', type=float, default=None)
    p.add_argument('--goal-space', type=parse_space(dim=3), default=None)
    p.add_argument('--block-space', type=parse_space(dim=4), required=True)
    p.add_argument('--geofence', type=float, required=True)
    p.add_argument('--hindsight-geofence', type=float)
    p.add_argument('--randomize-pose', action='store_true')
    p.add_argument('--logdir', type=str, default=None)
    p.add_argument('--save-path', type=str, default=None)
    p.add_argument('--load-path', type=str, default=None)
    p.add_argument(
        '--image-dims', type=parse_vector(length=2, delim=','), default='800,800')
    p.add_argument('--render', action='store_true')
    p.add_argument('--render-freq', type=int, default=None)
    p.add_argument('--record', action='store_true')
    p.add_argument('--concat-record', action='store_true')
    p.add_argument('--record-freq', type=int, default=None)
    p.add_argument('--record-path', type=int, default=None)
    p.add_argument('--xml-file', type=Path, default='world.xml')
    p.add_argument('--set-xml', type=put_in_xml_setter, action='append', nargs='*')
    p.add_argument('--use-dof', type=str, action='append')
    main(**vars(p.parse_args()))


if __name__ == '__main__':
    cli()
