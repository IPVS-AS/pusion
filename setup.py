from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pusion",
    version="0.2.0",
    description="Pusion (Python Universal Fusion) is a generic and flexible decision fusion framework written in Python for combining multiple classifier’s decision outcomes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Admir Obralija, Yannick Wilhelm",
    author_email="yannick.wilhelm@gsame.uni-stuttgart.de",
    license="MIT",
    url = "https://github.com/IPVS-AS/pusion",
    project_urls={
        "Bug Tracker": "https://github.com/IPVS-AS/pusion/issues",
        "Documentation": "https://ipvs-as.github.io/pusion/build/html/index.html"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
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
    packages=find_packages(exclude=['tests'], where="pusion"),
    package_dir={"": "pusion"},
    tests_require=['pytest'],
    test_suite="tests"
)