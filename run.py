#! /usr/bin/env python
import sys
from pathlib import Path

from runs.commands.new import cli
from runs.util import RunPath
gear = sys.argv[1]

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
    path=RunPath('4dof/gear=' + gear),
    prefix=f'setopt +o nomatch; EGL={sys.argv[2]} nice',
    command=command,
    description='search xml hyperparams to find good x_slide params',
    flags=[
        f"--logdir=.runs/tensorboard/<path>",
        f"--save-path=.runs/checkpoints/<path>/model.ckpt",
        f'--set-xml \'actuator/position[@name="slide_x"]/gear\' {gear}',
        '--set-xml '
        '\'actuator/position[@name="slide_x"]/k\' 3|'
        '\'actuator/position[@name="slide_x"]/k\' 5|'
        '\'actuator/position[@name="slide_x"]/k\' 7',
        '--set-xml '
        '\'actuator/position[@name="slide_x"]/forcerange\' "-100 100"|'
        '\'actuator/position[@name="slide_x"]/forcerange\' "-200 200"|'
        '\'actuator/position[@name="slide_x"]/forcerange\' "-300 300"',
        '--set-xml '
        '\'body/joint[@name="slide_x"]/damping\' 1500|'
        '\'body/joint[@name="slide_x"]/damping\' 2000|'
        '\'body/joint[@name="slide_x"]/damping\' 2500'
    ])

