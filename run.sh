#! /usr/bin/env bash

runs new 4dof/ 'pick-and-place --dofs x_slide arm_flex_joint wrist_roll_joint
hand_l_proximal_joint hand_r_proximal_joint'\
  --flag="--set-xml 'actuator/position[@name="slide_x"]/forcerange' '-100 100'|\
                    'actuator/position[@name="slide_x"]/forcerange' '-200 200'|\
                    'actuator/position[@name="slide_x"]/forcerange' '-300 300'|"\ 
  --flag="--set-xml 'actuator/position[@name="arm_flex_motor"]/forcerange' '-200 400'"  --description='search xml hyperparams to find good x_slide params'
