#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

# TODO: convert readme to restuctured text
with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'future==0.16.0'
    # TODO: put additional package requirements here
]

setup_requirements = [
    'pytest-runner'
]

test_requirements = [
    'flake8==3.2.1',
    'pytest==3.0.5',
    'pytest-cov==2.4.0'
]

setup(
    name='mmworkbench',
    version='0.0.0',
    description="A Python module for building natural language processing models.",
    long_description=readme + '\n\n' + history,
    author="MindMeld, Inc.",
    author_email='contact@mindmeld.com',
    url='https://github.com/mindmeld/mindmeld',
    packages=[
        'mmworkbench',
    ],
    package_dir={'mmworkbench': 'mmworkbench'},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='mindmeld',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements
)
