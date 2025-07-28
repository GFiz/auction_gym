from setuptools import setup, find_packages

setup(
    name='auction_gym',
    version='0.1.0',
    description='A gym environment for RTB auction simulation and agent training',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'numpy',
    ],
) 