import argparse
import re
import tempfile
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET

from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import LiftHindsightWrapper
from environments.lift import LiftEnv
from sac.networks import MlpAgent
from sac.train import HindsightTrainer, Trainer
from scripts.hsr import parse_space, parse_vector, cast_to_int, parse_activation, ACTIVATIONS


def put_in_xml_setter(ctx, param, value: str):
    setters = [XMLSetter(*v.split(',')) for v in value]
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
        if not set_xml:
            set_xml = []
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
def main(max_steps, min_lift_height, geofence, hindsight_geofence, seed,
         buffer_size, activation, n_layers, layer_size, learning_rate,
         reward_scale, entropy_scale, goal_space, block_space, grad_clip, batch_size, num_train_steps,
         concat_record, steps_per_action, logdir, save_path, load_path, render_freq,
         n_goals, record, randomize_pose,
         image_dims, record_freq, record_path, temp_path):
    env = TimeLimit(
        max_episode_steps=max_steps,
        env=LiftEnv(
            steps_per_action=steps_per_action,
            fixed_block=False,
            randomize_pose=randomize_pose,
            min_lift_height=min_lift_height,
            xml_filepath=temp_path,
            block_space=block_space,
            geofence=geofence,
            render_freq=render_freq,
            record=record,
            record_path=record_path,
            record_freq=record_freq,
            concat_recordings=concat_record,
            image_dimensions=image_dims,
        ))

    kwargs = dict(
        seq_len=None,
        base_agent=MlpAgent,
        seed=seed,
        device_num=1,
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
            env=LiftHindsightWrapper(env=env, geofence=hindsight_geofence),
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


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--activation', type=parse_activation,
                   default='relu', choices=ACTIVATIONS.keys())
    p.add_argument('--n-layers', type=int, required=True)
    p.add_argument('--layer-size', type=int, required=True)
    p.add_argument('--buffer-size', type=cast_to_int, required=True)
    p.add_argument('--num-train-steps', type=int, required=True)
    p.add_argument('--steps-per-action', type=int, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    scales = p.add_mutually_exclusive_group(required=True)
    scales.add_argument('--reward-scale', type=float, default=1)
    scales.add_argument('--entropy-scale', type=float, default=1)
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--n-goals', type=int, required=True)
    p.add_argument('--min-lift-height', type=float, default=None, required=True)
    p.add_argument('--grad-clip', type=float, required=True)
    p.add_argument('--goal-space', type=parse_space(dim=3), default=None)  # TODO
    p.add_argument('--block-space', type=parse_space(dim=2), required=True)
    p.add_argument('--geofence', type=float, required=True)
    p.add_argument('--hindsight-geofence', type=float)
    p.add_argument('--randomize-pose', action='store_true')
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
    main(**vars(p.parse_args()))

if __name__ == '__main__':
    cli()

