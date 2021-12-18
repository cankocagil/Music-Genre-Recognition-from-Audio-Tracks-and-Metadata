from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='CS464 Final',
    version='0.1',
    description='CS464 Final Setup File',
    long_description = long_description,
    author='Can Kocagil',
    author_email='can.kocagil@ug.bilkent.edu.tr',
    packages = find_packages(where='src'),
    package_dir = 
        {
            '':  'src'
        },
    install_requires=required,
    py_modules = [ 
        splitext(
            basename(path))[0] for path in glob('src/*.py')
    ],
    classifiers = [
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent"
    ],
)