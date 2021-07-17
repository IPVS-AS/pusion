from setuptools import find_packages, setup

setup(
    name="pusion",
    packages=find_packages(exclude=['tests']),
    version="1.3.5",
    description="A python framework for combining multi-classifier decision outputs of classification problems.",
    author="Admir Obralija",
    license="MIT",
    install_requires=[
        'numpy~=1.20.2',
        'scipy~=1.6.2',
        'scikit-learn~=0.24.1',
        'setuptools~=54.2.0',
        'pandas~=1.2.3',
        'matplotlib~=3.4.1',
        'pickle5~=0.0.11'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite="tests"
)
