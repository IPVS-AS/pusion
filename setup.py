from setuptools import find_packages, setup

setup(
    name="clunpy",
    packages=find_packages(include=['clunpy']),
    version="1.1.4",
    description="Framework for Classifier Decision Fusion",
    author="Admir Obralija",
    license="MIT",
    install_requires=['numpy', 'scipy', 'sklearn'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite="tests"
)
