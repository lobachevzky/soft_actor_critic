#! /usr/bin/env python

from runs.main import main

command = "pick-and-place " \
          "--use-dof=slide_x " \
          "--use-dof=arm_flex_joint " \
          "--use-dof=wrist_roll_joint " \
          "--use-dof=hand_l_proximal_joint " \
          "--use-dof=hand_r_proximal_joint "
flags = [f'--flag=\"{f}\"' for f in
         ['--set-xml '
          '\'actuator/position[@name="slide_x"]/gear\' 1|'
          '\'actuator/position[@name="slide_x"]/gear\' 5',
          '--set-xml '
          '\'actuator/position[@name="slide_x"]/kp\' 300|'
          '\'actuator/position[@name="slide_x"]/kp\' 500|'
          '\'actuator/position[@name="slide_x"]/kp\' 700',
          '--set-xml \'actuator/position[@name="slide_x"]/forcerange\' "-200 200"',
          '--set-xml '
          '\'body/joint[@name="slide_x"]/damping\' 2000|'
          '\'body/joint[@name="slide_x"]/damping\' 2200']]
main(['new',
      '4dof/forcerange=200',
      command,
      '--description="search xml hyperparams to find good x_slide params"']
     + flags)
