# It is responsible for creating ML application as package
# This can be shared and used in PyPy

# setup.py checks in how many folders there are init_.py (It 
# will try to consider that as package and try to build it)

from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT_E = "-e ."
# -e shows setup.py is present. We remove this as package

# We get the data from requirements.txt and store them as list 
# by removing /n and -e .
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n','') for req in requirements]

        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)

        return requirements
    
setup(
    name="HolidayPackagePredictor",
    version="0.0.1",
    author="Anubhav",
    author_email="tech.freek257@gmail.com",
    install_requires=get_requirements('requirements.txt')
)