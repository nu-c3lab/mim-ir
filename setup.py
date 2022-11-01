from setuptools import setup, find_packages

with open('requirements.txt', 'r') as requirements:
    install_requires = requirements.read().splitlines()

setup(
    name='mim_core',
    version='0.1.0',
    author='C3 Lab',
    author_email='markosterbentz2023@u.northwestern.edu',
    description='A package for the core components of the Mim question answering system.',
    url='https://github.com/nu-c3lab/mim-core',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires= install_requires,
    include_package_data=True
    )
