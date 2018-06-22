#! /usr/bin/env bash

runs new 4dof/ 'pick-and-place '`
`' --use-dof=x_slide'`
`' --use-dof=arm_flex_joint'`
`' --use-dof=wrist_roll_joint'`
`' --use-dof=hand_l_proximal_joint'`
`' --use-dof=hand_r_proximal_joint'\
  --flag='--set-xml '`
`'actuator/position[@name=\"slide_x\"]/gear 3|'`
`'actuator/position[@name=\"slide_x\"]/gear 5|'`
`'actuator/position[@name=\"slide_x\"]/gear 7' \
  --flag='--set-xml '`
`'actuator/position[@name=\"slide_x\"]/kp 3|'`
`'actuator/position[@name=\"slide_x\"]/kp 5|'`
`'actuator/position[@name=\"slide_x\"]/kp 7'\
  --flag='--set-xml '`
`'actuator/position[@name=\"slide_x\"]/forcerange \"-100 100\"|'`
`'actuator/position[@name=\"slide_x\"]/forcerange \"-200 200\"|'`
`'actuator/position[@name=\"slide_x\"]/forcerange \"-300 300\"'\
  --flag='--set-xml '`
`'body/joint[@name=\"slide_x\"]/damping 1500|'`
`'body/joint[@name=\"slide_x\"]/damping 2000|'`
`'body/joint[@name=\"slide_x\"]/damping 2500'\
  --description='search xml hyperparams to find good x_slide params'
