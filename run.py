#! /usr/bin/env python
import argparse

from runs.main import main

parser = argparse.ArgumentParser()
parser.add_argument('forcerange', type=str)
args = parser.parse_args()
r = args.forcerange
command = "pick-and-place " \
          "--use-dof=arm_lift_joint " \
          "--use-dof=arm_flex_joint " \
          "--use-dof=wrist_roll_joint " \
          "--use-dof=hand_l_proximal_joint " \
          "--use-dof=hand_r_proximal_joint " \
          f"--set-xml=\'actuator/position[@joint=\"arm_lift_joint\"]/forcerange\',\'-{r} {r}\'"
flags = [f'--flag={f}' for f in
         [
             '--set-xml='
             '\'actuator/position[@joint=\"arm_lift_joint\"]/kp\',5|'
             '\'actuator/position[@joint=\"arm_lift_joint\"]/kp\',8|'
             '\'actuator/position[@joint=\"arm_lift_joint\"]/kp\',10',
             '--set-xml='
             '\'actuator/position[@joint=\"arm_lift_joint\"]/gear\',1|'
             '\'actuator/position[@joint=\"arm_lift_joint\"]/gear\',3|'
             '\'actuator/position[@joint=\"arm_lift_joint\"]/gear\',5',
             '--set-xml='
             '\'body/joint[@name=\"arm_lift_joint\"]/damping\',100|'
             '\'body/joint[@name=\"arm_lift_joint\"]/damping\',150|'
             '\'body/joint[@name=\"arm_lift_joint\"]/damping\',200',
         ]]
main(['new',
      f'4dof/forcerange={r}/',
      command,
      '--description="search xml hyperparams to find good x_slide params"']
     + flags)
