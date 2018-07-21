#! /usr/bin/env bash

eval $(echo "$1 --render" | sed 's/--logdir\S*//g' | sed 's/--save-path/--load-path/g')
