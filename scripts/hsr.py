# stdlib
import argparse
import re
import tempfile
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import List, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import tensorflow as tf
# third party
from gym import spaces
from gym.spaces import Box
from gym.wrappers import TimeLimit

# first party
from environments.hindsight_wrapper import (HSRHindsightWrapper,
                                            MBHSRHindsightWrapper)
from environments.hsr import HSREnv, MultiBlockHSREnv
from sac.networks import MlpAgent
from sac.train import HindsightTrainer, Trainer
from sac.unsupervised_trainer import UnsupervisedTrainer


def make_box(*tuples: Tuple[float, float]):
    low, high = map(np.array, zip(*[(map(float, m)) for m in tuples]))
    return spaces.Box(low=low, high=high, dtype=np.float32)


def parse_space(dim: int):
    def _parse_space(arg: str):
        regex = re.compile('\((-?[\.\d]+),(-?[\.\d]+)\)')
        matches = regex.findall(arg)
        if len(matches) != dim:
            raise argparse.ArgumentTypeError(f'Arg {arg} must have {dim} substrings '
                                             f'matching pattern {regex}.')
        return make_box(*matches)

    return _parse_space


def parse_vector(length: int, delim: str):
    def _parse_vector(arg: str):
        vector = tuple(map(float, arg.split(delim)))
        if len(vector) != length:
            raise argparse.ArgumentError(f'Arg {arg} must include {length} float values'
                                         f'delimited by "{delim}".')
        return vector

    return _parse_vector


def cast_to_int(arg: str):
    return int(float(arg))


ACTIVATIONS = dict(
    relu=tf.nn.relu, leaky=tf.nn.leaky_relu, elu=tf.nn.elu, selu=tf.nn.selu)


def parse_activation(arg: str):
    return ACTIVATIONS[arg]


def put_in_xml_setter(arg: str):
    setters = [XMLSetter(*v.split(',')) for v in arg]
    mirroring = [XMLSetter(p.replace('_l_', '_r_'), v)
                 for p, v in setters if '_l_' in p] \
                + [XMLSetter(p.replace('_r_', '_l_'), v)
                   for p, v in setters if '_r_' in p]
    return [s._replace(path=s.path) for s in setters + mirroring]


def env_wrapper(func):
    @wraps(func)
    def _wrapper(set_xml, use_dof, n_blocks, goal_space, xml_file, geofence, **kwargs):
        xml_filepath = Path(
            Path(__file__).parent.parent, 'environments', 'models', xml_file).absolute()
        if set_xml is None:
            set_xml = []
        site_size = ' '.join([str(geofence)] * 3)
        path = Path('worldbody', 'body[@name="goal"]', 'site[@name="goal"]', 'size')
        set_xml += [XMLSetter(path=f'./{path}', value=site_size)]
        with mutate_xml(
                changes=set_xml,
                dofs=use_dof,
                n_blocks=n_blocks,
                goal_space=goal_space,
                xml_filepath=xml_filepath) as temp_path:
            return func(
                geofence=geofence, temp_path=temp_path, goal_space=goal_space, **kwargs)

    return _wrapper


XMLSetter = namedtuple('XMLSetter', 'path value')


@contextmanager
def mutate_xml(changes: List[XMLSetter], dofs: List[str], goal_space: Box, n_blocks: int,
               xml_filepath: Path):
    def rel_to_abs(path: Path):
        return Path(xml_filepath.parent, path)

    def mutate_tree(tree: ET.ElementTree):

        worldbody = tree.getroot().find("./worldbody")
        rgba = [
            "0 1 0 1",
            "0 0 1 1",
            "0 1 1 1",
            "1 0 0 1",
            "1 0 1 1",
            "1 1 0 1",
            "1 1 1 1",
        ]

        if worldbody:
            for i in range(n_blocks):
                pos = ' '.join(map(str, goal_space.sample()))
                name = f'block{i}'

                body = ET.SubElement(worldbody, 'body', attrib=dict(name=name, pos=pos))
                ET.SubElement(
                    body,
                    'geom',
                    attrib=dict(
                        name=name,
                        type='box',
                        mass='1',
                        size=".05 .025 .017",
                        rgba=rgba[i],
                        condim='6',
                        solimp="0.99 0.99 "
                        "0.01",
                        solref='0.01 1'))
                ET.SubElement(body, 'freejoint', attrib=dict(name=f'block{i}joint'))

        for change in changes:
            parent = re.sub('/[^/]*$', '', change.path)
            element_to_change = tree.find(parent)
            if isinstance(element_to_change, ET.Element):
                print('setting', change.path, 'to', change.value)
                name = re.search('[^/]*$', change.path)[0]
                element_to_change.set(name, change.value)

        for actuators in tree.iter('actuator'):
            for actuator in list(actuators):
                if actuator.get('joint') not in dofs:
                    print('removing', actuator.get('name'))
                    actuators.remove(actuator)
        for body in tree.iter('body'):
            for joint in body.findall('joint'):
                if not joint.get('name') in dofs:
                    print('removing', joint.get('name'))
                    body.remove(joint)

        parent = Path(temp[xml_filepath].name).parent

        for include_elt in tree.findall('*/include'):
            original_abs_path = rel_to_abs(include_elt.get('file'))
            tmp_abs_path = Path(temp[original_abs_path].name)
            include_elt.set('file', str(tmp_abs_path.relative_to(parent)))

        for compiler in tree.findall('compiler'):
            abs_path = rel_to_abs(compiler.get('meshdir'))
            compiler.set('meshdir', str(abs_path))

        return tree

    included_files = [
        rel_to_abs(e.get('file')) for e in ET.parse(xml_filepath).findall('*/include')
    ]

    temp = {
        path: tempfile.NamedTemporaryFile()
        for path in (included_files + [xml_filepath])
    }
    try:
        for path, f in temp.items():
            tree = ET.parse(path)
            mutate_tree(tree)
            tree.write(f)
            f.flush()

        yield Path(temp[xml_filepath].name)
    finally:
        for f in temp.values():
            f.close()


@env_wrapper
def main(max_steps, min_lift_height, geofence, hindsight_geofence, seed, buffer_size,
         activation, n_layers, layer_size, model_activation, model_n_layers,
         model_layer_size, learning_rate, reward_scale, entropy_scale, goal_space,
         block_space, grad_clip, batch_size, n_train_steps, record_separate_episodes,
         steps_per_action, logdir, save_path, load_path, render, render_freq, n_goals,
         record, randomize_pose, image_dims, record_freq, record_path, temp_path,
         save_threshold, no_random_reset, obs_type, multi_block, unsupervised):
    env = TimeLimit(
        max_episode_steps=max_steps,
        env=(MultiBlockHSREnv if multi_block else HSREnv)(
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
            record_separate_episodes=record_separate_episodes,
            image_dimensions=image_dims,
            no_random_reset=no_random_reset,
            obs_type=obs_type,
        ))

    kwargs = dict(
        seq_len=None,
        base_agent=MlpAgent,
        seed=seed,
        buffer_size=buffer_size,
        activation=activation,
        n_layers=n_layers,
        layer_size=layer_size,
        learning_rate=learning_rate,
        model_activation=model_activation,
        model_n_layers=model_n_layers,
        model_layer_size=model_layer_size,
        reward_scale=reward_scale,
        entropy_scale=entropy_scale,
        grad_clip=grad_clip if grad_clip > 0 else None,
        batch_size=batch_size,
        n_train_steps=n_train_steps)

    if hindsight_geofence:
        trainer = HindsightTrainer(
            env=(MBHSRHindsightWrapper if multi_block else HSRHindsightWrapper)(
                env=env, geofence=hindsight_geofence),
            n_goals=n_goals,
            **kwargs)
    elif unsupervised:
        trainer = UnsupervisedTrainer(
            env=HSRHindsightWrapper(env, geofence=geofence), **kwargs)
    else:
        trainer = Trainer(env=env, **kwargs)
    trainer.train(
        load_path=load_path,
        logdir=logdir,
        render=False,
        save_path=save_path,
        save_threshold=save_threshold,
    )


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument(
        '--activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    p.add_argument('--n-layers', type=int, required=True)
    p.add_argument('--layer-size', type=int, required=True)
    p.add_argument(
        '--model-activation',
        type=parse_activation,
        default=tf.nn.relu,
        choices=ACTIVATIONS.values())
    p.add_argument('--model-n-layers', type=int)
    p.add_argument('--model-layer-size', type=int)
    p.add_argument('--buffer-size', type=cast_to_int, required=True)
    p.add_argument('--n-train-steps', type=int, required=True)
    p.add_argument('--steps-per-action', type=int, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    scales = p.add_mutually_exclusive_group(required=True)
    scales.add_argument('--reward-scale', type=float, default=1)
    scales.add_argument('--entropy-scale', type=float, default=1)
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--n-goals', type=int)
    p.add_argument('--n-blocks', type=int, required=True)
    p.add_argument('--min-lift-height', type=float, default=None)
    p.add_argument('--grad-clip', type=float, required=True)
    p.add_argument('--goal-space', type=parse_space(dim=3), default=None)  # TODO
    p.add_argument('--block-space', type=parse_space(dim=4), required=True)
    p.add_argument('--obs-type', type=str, default=None)
    p.add_argument('--geofence', type=float, required=True)
    p.add_argument('--hindsight-geofence', type=float)
    p.add_argument('--no-random-reset', action='store_true')
    p.add_argument('--randomize-pose', action='store_true')
    p.add_argument('--logdir', type=str, default=None)
    p.add_argument('--load-path', type=str, default=None)
    p.add_argument('--save-path', type=str, default=None)
    p.add_argument('--save-threshold', type=int, default=None)
    p.add_argument(
        '--image-dims', type=parse_vector(length=2, delim=','), default='800,800')
    p.add_argument('--render', action='store_true')
    p.add_argument('--render-freq', type=int, default=None)
    p.add_argument('--record', action='store_true')
    p.add_argument('--record-separate-episodes', action='store_true')
    p.add_argument('--record-freq', type=int, default=None)
    p.add_argument('--record-path', type=Path, default=None)
    p.add_argument('--xml-file', type=Path, default='world.xml')
    p.add_argument('--set-xml', type=put_in_xml_setter, action='append', nargs='*')
    p.add_argument('--use-dof', type=str, action='append')
    p.add_argument('--multi-block', action='store_true')
    p.add_argument('--unsupervised', action='store_true')
    main(**vars(p.parse_args()))


if __name__ == '__main__':
    cli()
