from setuptools import find_packages, setup
from typing import List


HEYPEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> list[str]:
    """This function will return the list of requirements"""

    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HEYPEN_E_DOT in requirements:
            requirements.remove(HEYPEN_E_DOT)
    return requirements
 
 
        
setup(
name="diabetes_prediction",
version="0.1.0",
author="Alex",
author_email="alexgreat@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt"),
description="A package for predicting diabetes using machine learning.",
long_description=open("README.md").read(),
long_description_content_type="text/markdown",
python_requires=">=3.8"
)