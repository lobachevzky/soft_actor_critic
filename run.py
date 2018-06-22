#! /usr/bin/env python
import sys
from pathlib import Path

from runs.commands.new import cli
from runs.util import RunPath

command = "pick-and-place " \
          "--use-dof=slide_x " \
          "--use-dof=arm_flex_joint " \
          "--use-dof=wrist_roll_joint " \
          "--use-dof=hand_l_proximal_joint " \
          "--use-dof=hand_r_proximal_joint "
cli(db_path=Path('runs.db'),
    root=Path('.runs'),
    dir_names=['tensorboard', 'checkpoints'],
    quiet=False,
    assume_yes=False,
    path=RunPath('4dof/forcerange=200'),
    prefix=f'setopt +o nomatch; EGL={sys.argv[1]} nice',
    command=command,
    description='search xml hyperparams to find good x_slide params',
    flags=[
        f"--logdir=.runs/tensorboard/<path>",
        f"--save-path=.runs/checkpoints/<path>/model.ckpt",
        '--set-xml '
        '\'actuator/position[@name="slide_x"]/gear\' 1|'
        '\'actuator/position[@name="slide_x"]/gear\' 5',
        '--set-xml '
        '\'actuator/position[@name="slide_x"]/kp\' 300|'
        '\'actuator/position[@name="slide_x"]/kp\' 500|'
        '\'actuator/position[@name="slide_x"]/kp\' 700',
        '--set-xml '
        '\'actuator/position[@name="slide_x"]/forcerange\' "-200 200"',
        '--set-xml '
        '\'body/joint[@name="slide_x"]/damping\' 2000|'
        '\'body/joint[@name="slide_x"]/damping\' 2200'
    ])

