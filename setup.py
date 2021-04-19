from setuptools import find_packages, setup

setup(
    name="pusion",
    packages=find_packages(exclude=['tests']),
    version="1.3.4",
    description="A python framework for combining multi-classifier decisions in classification problems",
    author="Admir Obralija",
    license="MIT",
    install_requires=[
        'numpy>=1.20.2',
        'scipy>=1.6.2',
        'sklearn>=0.0',
        'pusion>=1.3.3',
        'scikit-learn>=0.24.1',
        'pandas>=1.2.3',
        'matplotlib>=3.4.1',
        'setuptools>=54.2.0'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite="tests"
)
