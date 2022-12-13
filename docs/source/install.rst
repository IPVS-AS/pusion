Install Pusion
==============

General requirements
--------------------
- Python >= 3.6

Package requirements
--------------------
- numpy >= 1.20.2
- scipy >= 1.6.2
- scikit-learn >= 0.24.1
- setuptools >= 54.2.0
- pandas >= 1.2.3
- matplotlib >= 3.4.1
- torchmetrics >= 0.7.2

Preparation
-----------
To generate the python distribution archives of `pusion`, update the PyPA's build to the latest version. Under `Windows` run in your python environment the following command:

.. code:: bash

    py -m pip install --upgrade build

If you are using `MacOS` or `Unix`, run in your python environment:

.. code:: bash

    python3 -m pip install --upgrade build


After cloning `pusion` from GitHub to your local computer, enter the `pusion` directory where the file `pyproject.toml` is located. Under this directory run the following `build` command. For `Windows` users:

.. code:: bash

    py -m build

For `MacOS` or `Unix` user:

.. code:: bash

    python3 -m build

Once successfully executed, two files are generated in the `dist` subfolder within the project's root folder. The `tar.gz` file is the source distribution and the `whl` file is the built distribution.

Installation
------------

The generated wheel can be installed using the ``pip3`` command, which also installs all required packages for `pusion`.

.. code:: bash

    pip3 install dist/pusion-<version>-py3-none-any.whl