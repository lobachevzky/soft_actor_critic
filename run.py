#! /usr/bin/env python
import argparse

from runs.main import main

parser = argparse.ArgumentParser()
parser.add_argument('flags', nargs='*', type=str)
args = parser.parse_args()
command = "pick-and-place " \
          "--use-dof=slide_x " \
          "--use-dof=arm_flex_joint " \
          "--use-dof=wrist_roll_joint " \
          "--use-dof=hand_l_proximal_joint " \
          "--use-dof=hand_r_proximal_joint " \
          + ' '.join([' --' + f for f in args.flags])

flags = [f'--flag={f}' for f in
         [
             '--set-xml='
             '\'actuator/position[@joint=\"slide_x\"]/kp\',100|'
             '\'actuator/position[@joint=\"slide_x\"]/kp\',300|'
             '\'actuator/position[@joint=\"slide_x\"]/kp\',500',
             '--set-xml='
             '\'actuator/position[@joint=\"slide_x\"]/gear\',3|'
             '\'actuator/position[@joint=\"slide_x\"]/gear\',5|'
             '\'actuator/position[@joint=\"slide_x\"]/gear\',8',
             '--set-xml='
             '\'body/joint[@name=\"slide_x\"]/damping\',1800|'
             '\'body/joint[@name=\"slide_x\"]/damping\',2000|'
             '\'body/joint[@name=\"slide_x\"]/damping\',2200',
         ]]
main(['new',
      f'4dof/block-movement/',
      command,
      '--description="search xml hyperparams to find good x_slide params"']
     + flags)
