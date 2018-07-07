#! /usr/bin/env python

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='soft-actor-critic',
    version='0.0.0',
    description='An implementation of Soft Actor Critic',
    long_description=long_description,
    url='https://github.com/chrisgrimm/soft_actor_critic',
    author='Chris Grimm',
    author_email='cgrimm1994@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    entry_points=dict(console_scripts=[
        'gym-env=scripts.gym_env:cli',
        'frozen-lake=scripts.frozen_lake:cli',
        'pick-and-place=scripts.pick_and_place:cli',
        'multi-task=scripts.multi_task:cli',
        'mountaincar=scripts.mountaincar:cli',
        'unsupervised=scripts.unsupervised:cli',
        'mouse-control=scripts.mouse_control:cli',
        'write-rewards=write_rewards:main',
    ]),
    install_requires=[
        'tensorflow>=1.6.0', 'scipy==1.1.0', 'gym==0.10.4', 'pygame==1.9.3', 'click==6.7'
    ])
