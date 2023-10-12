# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:50:10 2023

@author: abhay.bhandari
"""

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    Parameters
    ----------
    file_path : str
        DESCRIPTION.

    This function returns the list of requirements
    -------
    List[str]
        DESCRIPTION.

    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author_email='abhay.office0@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)