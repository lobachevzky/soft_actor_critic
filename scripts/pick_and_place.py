import tempfile
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import List
from xml.etree import ElementTree as ET

import click
import tensorflow as tf
from gym.wrappers import TimeLimit

from environments.hindsight_wrapper import PickAndPlaceHindsightWrapper
from environments.pick_and_place import PickAndPlaceEnv
from sac.train import HindsightTrainer
from scripts.gym_env import check_probability

XMLSetter = namedtuple('XMLSetter', 'path value')


@contextmanager
def mutate_xml(changes: List[XMLSetter], dofs: List[str], xml_filepath: Path):
    def rel_to_abs(path: Path):
        return Path(xml_filepath.parent, path)

    def mutate_tree(tree: ET.ElementTree):
        for change in changes:
            element_to_change = tree.find('./' + str(change.path.parent))
            if isinstance(element_to_change, ET.Element):
                print('setting', change.path, 'to', change.value)
                element_to_change.set(change.path.name, change.value)

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

    included_files = [rel_to_abs(e.get('file')) for e in
                      ET.parse(xml_filepath).findall('*/include')]

    temp = {path: tempfile.NamedTemporaryFile()
            for path in (included_files + [xml_filepath])}
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


def put_in_xml_setter(ctx, param, value: str):
    setters = [XMLSetter(*v.split(',')) for v in value]
    return [s._replace(path=PurePath(s.path)) for s in setters]


@click.command()
@click.option('--seed', default=0, type=int)
@click.option('--device-num', default=0, type=int)
@click.option('--relu', 'activation', flag_value=tf.nn.relu, default=True)
@click.option('--n-layers', default=3, type=int)
@click.option('--layer-size', default=256, type=int)
@click.option('--learning-rate', default=2e-4, type=float)
@click.option('--buffer-size', default=1e5, type=int)
@click.option('--num-train-steps', default=4, type=int)
@click.option('--steps-per-action', default=300, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('--reward-scale', default=7e3, type=float)
@click.option('--cheat-prob', default=0, type=float, callback=check_probability)
@click.option('--max-steps', default=200, type=int)
@click.option('--n-goals', default=1, type=int)
@click.option('--geofence', default=.4, type=float)
@click.option('--min-lift-height', default=.03, type=float)
@click.option('--grad-clip', default=4e4, type=float)
@click.option('--fixed-block', is_flag=True)
@click.option('--discrete', is_flag=True)
@click.option('--isolate-movements', is_flag=True)
@click.option('--logdir', default=None, type=str)
@click.option('--save-path', default=None, type=str)
@click.option('--load-path', default=None, type=str)
@click.option('--render-freq', type=int, default=0)
@click.option('--no-qvel', 'obs_type', flag_value=None, default=True)
@click.option('--add-qvel', 'obs_type', flag_value='qvel')
@click.option('--add-base-qvel', 'obs_type', flag_value='base-qvel')
@click.option('--add-robot-qvel', 'obs_type', flag_value='robot-qvel')
@click.option('--xml-file', type=Path, default='world.xml')
@click.option('--set-xml', multiple=True, callback=put_in_xml_setter)
@click.option('--use-dof', multiple=True, default=['slide_x',
                                                   'slide_y',
                                                   'arm_lift_joint',
                                                   'arm_flex_joint',
                                                   'wrist_roll_joint',
                                                   'hand_l_proximal_joint',
                                                   'hand_r_proximal_joint'])
def cli(max_steps, discrete, fixed_block, min_lift_height, geofence, seed, device_num,
        buffer_size, activation, n_layers, layer_size, learning_rate, reward_scale,
        cheat_prob, grad_clip, batch_size, num_train_steps, steps_per_action, logdir,
        save_path, load_path, render_freq, n_goals, xml_file, set_xml, use_dof,
        isolate_movements, obs_type):
    xml_filepath = Path(Path(__file__).parent.parent, 'environments', 'models', xml_file)
    with mutate_xml(changes=set_xml, dofs=use_dof, xml_filepath=xml_filepath) as temp_path:
        HindsightTrainer(
            env=PickAndPlaceHindsightWrapper(
                env=TimeLimit(
                    max_episode_steps=max_steps,
                    env=PickAndPlaceEnv(
                        discrete=discrete,
                        cheat_prob=cheat_prob,
                        steps_per_action=steps_per_action,
                        fixed_block=fixed_block,
                        min_lift_height=min_lift_height,
                        geofence=geofence,
                        render_freq=render_freq,
                        xml_filepath=temp_path,
                        isolate_movements=isolate_movements,
                        obs_type=obs_type,
                    ))),
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
            render=False)  # because render is handled inside env


if __name__ == '__main__':
    cli()
