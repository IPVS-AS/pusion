Install Pusion
==============

Requirements
------------
- numpy >= 1.20.2
- scipy >= 1.6.2
- scikit-learn >= 0.24.1
- setuptools >= 54.2.0
- pandas >= 1.2.3
- matplotlib >= 3.4.1

Preparation
-----------
In order to generate a python wheel package for `pusion`, the package `setuptools` needs to be installed in your Python
environment before.

.. code:: bash

    pip install setuptools

After cloning `pusion` to your local disc, enter the project and generate the package using with the following command.

.. code:: bash

    python setup.py bdist_wheel

Once generated, the wheel will be moved to the ``dist/`` directory within project's root.

Installation
------------

The generated wheel can be installed using the ``pip`` command, which also installs all required packages for `pusion`.

.. code:: bash

    pip install dist/pusion-<version>-py3-none-any.whl