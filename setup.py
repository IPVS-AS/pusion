from setuptools import find_packages, setup

setup(
    name="pusion",
    packages=find_packages(exclude=['tests']),
    version="1.3.2",
    description="A python framework for combining multi-classifier decisions in classification problems",
    author="Admir Obralija",
    license="MIT",
    install_requires=['numpy', 'scipy', 'sklearn'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite="tests"
)
