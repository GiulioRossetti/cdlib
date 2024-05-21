****************
Installing CDlib
****************

``CDlib`` *requires* python>=3.8.

To install the latest version of our library, download (or clone) the current project, open a terminal, and run the following commands:

.. code-block:: python

    pip install -r requirements.txt
    pip install -r requirements_optional.txt # (Optional) This might not work in Windows systems due to C-based dependencies.
    pip install .


Alternatively, use pip

.. code-block:: python

    pip install cdlib

or conda

.. code-block:: python

    conda create -n cdlib python=3.9
    conda config --add channels giuliorossetti
    conda config --add channels conda-forge
    conda install cdlib




You can install the development version directly from the GitHub repository with

.. code-block:: python

    pip install git+https://github.com/GiulioRossetti/cdlib.git


=====================
Optional Dependencies
=====================

^^^^^^^^^^^^
PyPi package
^^^^^^^^^^^^

The default installation does not include optional dependencies (e.g., ``graph-tool``) to simplify the installation process. If you need them, you can install them manually or run the following command:

.. code-block:: python

    pip install 'cdlib[C]'

This option, safe for GNU/Linux users, will install all those optional dependencies that require C code compilation.

.. code-block:: python

    pip install 'cdlib[pypi]'

This option will install all those optional dependencies that are not available on conda/conda-forge.

.. code-block:: python

    pip install 'cdlib[all]'

This option will install all optional dependencies accessible with the flag C and pypi.

^^^^^^^^
Advanced
^^^^^^^^

Due to strict requirements, installing a subset of optional dependencies is left outside the previous procedures.

----------
graph-tool
----------

``CDlib`` integrates the support for SBM models offered by ``graph-tool``.
To install it, refer to the official `documentation <https://git.skewed.de/count0/graph-tool/wikis/installation-instructions>`_ and install the conda-forge version of the package (or the deb version if in a Unix system).

------
ASLPAw
------

Since its 2.1.0 release, ``ASLPAw`` relies on ``gmpy2``, whose installation through pip is difficult to automate due to some C dependencies.
To address such an issue, test the following recipe:

.. code-block:: python

    conda install gmpy2
    pip install shuffle_graph>=2.1.0 similarity-index-of-label-graph>=2.0.1 ASLPAw>=2.1.0

If ASLPAw installation fails, please refer to the official ``gmpy2`` `repository <https://gmpy2.readthedocs.io/en/latest/intro.html#installation>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Optional Dependencies (Conda package)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CDlib`` relies on a few packages unavailable through conda: to install them, please use pip.

.. code-block:: python

    pip install pycombo
    pip install GraphRicciCurvature
    conda install gmpy2
    pip install shuffle_graph>=2.1.0 similarity-index-of-label-graph>=2.0.1 ASLPAw>=2.1.0

In case ASLPAw installation fails, please refer to the official ``gmpy2`` repository `repository <https://gmpy2.readthedocs.io/en/latest/intro.html#installation>`_.


