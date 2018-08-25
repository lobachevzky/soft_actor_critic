import re
import tempfile
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET

import click
import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import LiftHindsightWrapper
from environments.lift import LiftEnv
from sac.networks import LstmAgent, MlpAgent
from sac.train import HindsightTrainer, Trainer
from scripts.gym_env import check_probability


def put_in_xml_setter(ctx, param, value: str):
    setters = [XMLSetter(*v.split(',')) for v in value]
    mirroring = [XMLSetter(p.replace('_l_', '_r_'), v)
                 for p, v in setters if '_l_' in p] \
                + [XMLSetter(p.replace('_r_', '_l_'), v)
                   for p, v in setters if '_r_' in p]
    return [s._replace(path=s.path) for s in setters + mirroring]


def parse_double(ctx, param, string):
    if string is None:
        return
    a, b = map(float, string.split(','))
    return a, b


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


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--relu', 'activation', flag_value=tf.nn.relu, default=True)
@click.option('--mlp', 'agent', flag_value=MlpAgent, default=True)
@click.option('--lstm', 'agent', flag_value=LstmAgent)
@click.option('--seq-len', default=8, type=int)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=18e-5, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--steps-per-action', default=300, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=1, type=float)
@click.option('--entropy-scale', default=15e-5, type=float)
@click.option('--cheat-prob', default=0, type=float, callback=check_probability)
@click.option('--max-steps', default=300, type=int)
@click.option('--n-goals', default=1, type=int)
@click.option('--geofence', default=.1, type=float)
@click.option('--min-lift-height', default=.03, type=float)
@click.option('--grad-clip', default=4e4, type=float)
@click.option('--fixed-block', is_flag=True)
@click.option('--randomize-pose', is_flag=True)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--image-dims', type=str, callback=parse_double)
@click.option('--render-freq', type=int, default=0)
@click.option('--render', is_flag=True)
@click.option('--record-freq', type=int, default=0)
@click.option('--record-path', type=Path)
@click.option('--record', is_flag=True)
@click.option('--hindsight', is_flag=True)
@click.option('--block-xrange', type=str, default="-.1,.1", callback=parse_double)
@click.option('--block-yrange', type=str, default="-.2,.2", callback=parse_double)
@click.option('--xml-file', type=Path, default='world.xml')
@click.option('--set-xml', multiple=True, callback=put_in_xml_setter)
@click.option(
    '--use-dof',
    multiple=True,
    default=[
        'slide_x', 'slide_y', 'arm_lift_joint', 'arm_flex_joint', 'wrist_roll_joint',
        'hand_l_proximal_joint', 'hand_r_proximal_joint'
    ])
@env_wrapper
def cli(max_steps, fixed_block, min_lift_height, geofence, seed, device_num, buffer_size,
        activation, n_layers, layer_size, learning_rate, reward_scale, entropy_scale,
        cheat_prob, grad_clip, batch_size, num_train_steps, steps_per_action, logdir,
        save_path, load_path, render_freq, record_freq, record_path, image_dims, record,
        n_goals, block_xrange, block_yrange, agent, seq_len, hindsight, temp_path,
        randomize_pose):
    env = TimeLimit(
        max_episode_steps=max_steps,
        env=LiftEnv(
            cheat_prob=cheat_prob,
            steps_per_action=steps_per_action,
            fixed_block=fixed_block,
            randomize_pose=randomize_pose,
            min_lift_height=min_lift_height,
            xml_filepath=temp_path,
            block_xrange=block_xrange,
            block_yrange=block_yrange,
            render_freq=render_freq,
            record=record,
            record_path=record_path,
            record_freq=record_freq,
            image_dimensions=image_dims,
        ))
    kwargs = dict(
        seq_len=seq_len,
        base_agent=agent,
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
    )
    if hindsight:
        env = LiftHindsightWrapper(env=env, geofence=geofence)
        trainer = HindsightTrainer(env=env, n_goals=n_goals, **kwargs)
    else:
        trainer = Trainer(env=env, **kwargs)
    trainer.train(
        load_path=load_path,
        logdir=logdir,
        save_path=save_path,
        render=False,  # because render is handled inside env
    )


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


if __name__ == '__main__':
    cli()
