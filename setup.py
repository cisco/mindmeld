#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains the specification of the mindmeld package"""
# pylint: disable=locally-disabled,invalid-name
from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'attrs~=18.2',
    'Click~=6.0',
    'click-log==0.1.8',
    'distro~=1.3',
    'elasticsearch5~=5.5',
    'Flask~=0.12',
    'Flask-Cors~=3.0',
    'future~=0.16.0',
    'nltk~=3.2',
    'numpy~=1.15',
    'pandas~=0.22',
    'pip>=9.0.1',
    'py~=1.4',
    'python-dateutil~=2.6',
    'pytz>=2018.5',
    'scipy>=0.9,<2.0',
    'scikit-learn>=0.18.1,<0.20; python_version < "3.7"',
    'scikit-learn>=0.19.2,<0.20; python_version >= "3.7"',
    'requests>=2.20.1,<3.0',
    'tqdm~=4.15',
    'python-crfsuite~=0.9; python_version < "3.7"',
    'python-crfsuite>=0.9.6,<1.0; python_version >= "3.7"',
    'sklearn-crfsuite>=0.3.6,<1.0',
    'immutables~=0.9',
    'tensorflow~=1.2; python_version < "3.7"',
    'tensorflow>=1.13.1,<2.0; python_version >= "3.7"',
    'pyyaml>=5.1.1'
]

setup_requirements = [
    'pytest-runner~=2.11',
    'setuptools>=36'
]

test_requirements = [
    'flake8==3.5.0',
    'pylint==1.6.5',
    'pytest==3.8.0',
    'pytest-cov==2.4.0',
    'pytest-asyncio==0.8.0'
]

setup(
    name='mindmeld',
    version='4.1.4',
    description="A Conversational AI platform.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Cisco Systems, Inc.",
    author_email='contact@mindmeld.com',
    url='https://github.com/cisco/mindmeld',
    packages=[
        'mindmeld',
    ],
    package_dir={'mindmeld': 'mindmeld'},
    entry_points={
        'console_scripts': ['mindmeld=mindmeld.cli:cli']
    },
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords=['mindmeld', 'nlp', 'ai', 'conversational'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'License :: OSI Approved :: Apache Software License'
    ],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements
)
