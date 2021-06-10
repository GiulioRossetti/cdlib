****************
Installing CDlib
****************

Before installing ``CDlib``, you need to have setuptools installed.

=============
Quick install
=============

Get ``CDlib`` from the Python Package Index at pypl_.

or install it with

.. code-block:: python

    pip install cdlib

and an attempt will be made to find and install an appropriate version that matches your operating system and Python version.
Please note that ``CDlib`` requires Python>=3.8

You can install the development version with

.. code-block:: python

    pip install git+https://github.com/GiulioRossetti/cdlib.git


=====================
Optional Dependencies
=====================

``CDlib`` relies on a few packages calling C code that can be cumbersome to install on Windows machines: to address such issue, the default installation does not try to install set up such requirements.

Such a choice has been made to allow (even) Windows user to install the library and get access to its core functionalities.

To made available (most of) the optional packages you can either:

- (Windows) manually install the optional packages (versions details are specified in ``requirements_optional.txt``) following the original projects guidelines, or
- (Linux/OSX) run the command:

.. code-block:: python

    pip install cdlib[C]



Such caveat will install everything that can be easily automated under Linux/OSX.

--------
Advanced
--------
**Graph-tool**

The only optional dependency that will remain unsatisfied following the previous procedures will be **graph-tool** (used to add SBM models).
If you need it up and running, refer to the official `documentation <https://git.skewed.de/count0/graph-tool/wikis/installation-instructions>`_  and install the conda-forge version of the package.

**ASLPAw**

Since its 2.1.0 release ``ASLPAw`` relies on ``gmpy2`` whose installation through pip is not easy to automatize due to some C dependencies.
To address such issue test the following recipe:

.. code-block:: bash

    conda install gmpy2
    pip install shuffle_graph>=2.1.0 similarity-index-of-label-graph>=2.0.1 ASLPAw>=2.1.0


In case this does not solve the issue, please refer to the official ``gmpy2`` `installation <https://gmpy2.readthedocs.io/en/latest/intro.html#installation>`_ instructions.


======================
Installing from source
======================

You can install from source by downloading a source archive file (tar.gz or zip) or by checking out the source files from the GitHub source code repository.

``CDlib`` is a pure Python package; you don’t need a compiler to build or install it.

-------------------
Source archive file
-------------------
Download the source (tar.gz or zip file) from pypl_  or get the latest development version from GitHub_

Unpack and change directory to the source directory (it should have the files README.txt and setup.py).

Run python setup.py install to build and install

------
GitHub
------
Clone the ``CDlib`` repostitory (see GitHub_ for options)

.. code-block:: python

    git clone https://github.com/GiulioRossetti/cdlib.git

Change directory to CDlib

Run python setup.py install to build and install

If you don’t have permission to install software on your system, you can install into another directory using the --user, --prefix, or --home flags to setup.py.

For example

.. code-block:: python

    python setup.py install --prefix=/home/username/python

or

.. code-block:: python

    python setup.py install --home=~

or

.. code-block:: python

    python setup.py install --user

If you didn’t install in the standard Python site-packages directory you will need to set your PYTHONPATH variable to the alternate location. See http://docs.python.org/2/install/index.html#search-path for further details.

============
Requirements
============
------
Python
------

To use ``CDlib`` you need Python 3.6 or later.

The easiest way to get Python and most optional packages is to install the Enthought Python distribution “Canopy” or using Anaconda.

There are several other distributions that contain the key packages you need for scientific computing. 


.. _pypl: https://pypi.python.org/pypi/CDlib/
.. _GitHub: https://github.com/GiulioRossetti/CDlib/
