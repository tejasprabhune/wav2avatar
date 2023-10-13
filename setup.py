from setuptools import setup, find_packages

setup(
    name='wav2avatar',
    version='0.0.1',
    install_requires=[
        'numpy'
    ],
    packages=find_packages(
        include=['wav2avatar*']
    )
)