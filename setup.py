from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='MaxEntML',
    version='1.0.0',
    description='Library for training machine learning models for maximum entropy closure problem for rarefied gaz dynamics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),

    author='Baptiste Romain, Daniele Biasone',
    author_email='baptiste.romain@onera.fr, daniele.biasone@onera.fr',
    url='',
    license='MIT',

    install_requires=[

    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Maximum Entropy closure',
    ],

)