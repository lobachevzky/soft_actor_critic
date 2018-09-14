import argparse
import re
import tempfile
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET

import tensorflow as tf
from gym import spaces
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import MujocoHindsightWrapper
from environments.lift import LiftEnv
from sac.networks import MlpAgent
from sac.train import HindsightTrainer, Trainer


def parse_space(dim: int):
    def _parse_space(arg: str):
        regex = '\(\d+,\d+\)(?:$|,)'
        matches = re.match(regex, arg)
        if len(matches.groups()) != dim:
            raise argparse.ArgumentTypeError(
                f'Arg {arg} must have {dim} substrings '
                f'matching pattern {regex}.')
        tuples = [tuple(map(float, s.split(',')))
                  for s in matches.groups()]
        return spaces.Box(*zip(tuples))

    return _parse_space


def parse_vector(length: int, delim: str):
    def _parse_vector(arg: str):
        vector = tuple(map(float, arg.split(delim)))
        if len(vector) != length:
            raise argparse.ArgumentError(
                f'Arg {arg} must include {length} float values'
                f'delimited by "{delim}".'
            )
        return vector

    return _parse_vector


ACTIVATIONS = dict(
    relu=tf.nn.relu
)


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
    def _wrapper(render, render_freq, set_xml, use_dof, xml_file, **kwargs):
        if render and not render_freq:
            render_freq = 20
        xml_filepath = Path(
            Path(__file__).parent.parent, 'environments', 'models', xml_file).absolute()
        with mutate_xml(
                changes=set_xml, dofs=use_dof, xml_filepath=xml_filepath) as temp_path:
            return func(temp_path=temp_path, render_freq=render_freq, **kwargs)

    return _wrapper


XMLSetter = namedtuple('XMLSetter', 'path value')


@contextmanager
def mutate_xml(changes: List[XMLSetter], dofs: List[str], xml_filepath: Path):
    def rel_to_abs(path: Path):
        return Path(xml_filepath.parent, path)

    def mutate_tree(tree: ET.ElementTree):
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
def cli(max_steps, fixed_block, min_lift_height, geofence, hindsight_geofence, seed,
        device_num, buffer_size, activation, n_layers, layer_size, learning_rate,
        reward_scale, entropy_scale, grad_clip, batch_size, num_train_steps,
        concat_recordings, steps_per_action, logdir, save_path, load_path, render_freq,
        n_goals, block_xrange, seq_len, block_yrange, agent, record, randomize_pose,
        image_dims, record_freq, record_path, temp_path):
    env = TimeLimit(
        max_episode_steps=max_steps,
        env=LiftEnv(
            steps_per_action=steps_per_action,
            fixed_block=fixed_block,
            randomize_pose=randomize_pose,
            min_lift_height=min_lift_height,
            xml_filepath=temp_path,
            block_xrange=block_xrange,
            block_yrange=block_yrange,
            geofence=geofence,
            render_freq=render_freq,
            record=record,
            record_path=record_path,
            record_freq=record_freq,
            concat_recordings=concat_recordings,
            image_dimensions=image_dims,
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
        reward_scale=reward_scale,
        entropy_scale=entropy_scale,
        grad_clip=grad_clip if grad_clip > 0 else None,
        batch_size=batch_size,
        num_train_steps=num_train_steps)

    if hindsight_geofence:
        trainer = HindsightTrainer(
            env=MujocoHindsightWrapper(env=env, geofence=hindsight_geofence),
            n_goals=n_goals,
            **kwargs)
    else:
        trainer = Trainer(env=env, **kwargs)
    trainer.train(
        load_path=load_path,
        logdir=logdir,
        render=False,
        save_path=save_path,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int)
    p.add_argument('--activation', type=str, default='relu',
                   choices=ACTIVATIONS.keys())
    p.add_argument('--n-layers', type=int)
    p.add_argument('--layer-size', type=int)
    p.add_argument('--buffer-size', type=int)
    p.add_argument('--num-train-steps', type=int)
    p.add_argument('--num-train-steps', type=int)
    p.add_argument('--steps-per-action', type=int)
    p.add_argument('--batch-size', type=int)
    scales = p.add_mutually_exclusive_group(required=True)
    scales.add_argument('--reward-scale', type=int, default=1)
    scales.add_argument('--entropy-scale', type=int, default=1)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--n-goals', type=int)
    p.add_argument('--n-goals', type=int)
    p.add_argument('--hindsight-geofence', type=float)
    p.add_argument('--geofence', type=float)
    p.add_argument('--min-lift-height', type=float, default=None)
    p.add_argument('--grad-clip', type=float)
    p.add_argument('--goal-space', type=parse_space(dim=3))
    p.add_argument('--block-space', type=parse_space(dim=3))
    p.add_argument('--logdir', type=str, default=None)
    p.add_argument('--save-path', type=str, default=None)
    p.add_argument('--load-path', type=str, default=None)
    p.add_argument('--image-dims',
                   type=parse_vector(length=2, delim=','),
                   default='800,800')
    p.add_argument('--render', action='store_true')
    p.add_argument('--render-freq', type=int, default=None)
    p.add_argument('--record', action='store_true')
    p.add_argument('--concat-record', action='store_true')
    p.add_argument('--record-freq', type=int, default=None)
    p.add_argument('--record-path', type=int, default=None)
    p.add_argument('--xml-file', type=Path, default='world.xml')
    p.add_argument('--set-xml', type=put_in_xml_setter,
                   action='append', nargs='*')
    p.add_argument('--use-dof', action='append', nargs='*',
                   default=[
                       'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint', 'wrist_roll_joint',
                       'hand_l_proximal_joint', 'hand_r_proximal_joint'
                   ])
    cli(**vars(p.parse_args()))
