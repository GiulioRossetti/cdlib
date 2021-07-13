# CDlib - Community Discovery Library
[![codecov](https://codecov.io/gh/GiulioRossetti/cdlib/branch/master/graph/badge.svg?token=3YJOEVK02B)](https://codecov.io/gh/GiulioRossetti/cdlib)
[![Build](https://github.com/GiulioRossetti/cdlib/actions/workflows/python-package.yml/badge.svg)](https://github.com/GiulioRossetti/cdlib/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/cdlib/badge/?version=latest)](http://cdlib.readthedocs.io/en/latest/?badge=latest)
[![CodeQL](https://github.com/GiulioRossetti/cdlib/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/GiulioRossetti/cdlib/actions/workflows/codeql-analysis.yml)
[![Updates](https://pyup.io/repos/github/GiulioRossetti/cdlib/shield.svg)](https://pyup.io/repos/github/GiulioRossetti/cdlib/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/GiulioRossetti/nclib.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/GiulioRossetti/nclib/context:python)
[![pyversions](https://img.shields.io/pypi/pyversions/cdlib.svg)](https://badge.fury.io/py/cdlib)
[![PyPI version](https://badge.fury.io/py/cdlib.svg)](https://badge.fury.io/py/cdlib)
[![Anaconda-Server Badge](https://anaconda.org/giuliorossetti/cdlib/badges/version.svg)](https://anaconda.org/giuliorossetti/cdlib)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/cdlib/month)](https://pepy.tech/project/cdlib)
[![Downloads](https://pepy.tech/badge/cdlib)](https://pepy.tech/project/cdlib)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4575156.svg)](https://doi.org/10.5281/zenodo.4575156)
[![SBD++](https://img.shields.io/badge/Available%20on-SoBigData%2B%2B-green)](https://sobigdata.d4science.org/group/sobigdata-gateway/explore?siteId=20371853)


[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/_CDlib_.svg?style=social&label=Follow%20%40_CDlib_)](https://twitter.com/_CDlib_)

<!---
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FGiulioRossetti%2Fcdlib.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FGiulioRossetti%2Fcdlib?ref=badge_shield)
--->

``CDlib`` is a meta-library for community discovery in complex networks: it implements algorithms, clustering fitness functions as well as visualization facilities.


``CDlib`` is designed around the ``networkx`` python library: however, when needed, it takes care to automatically convert (from and to) ``igraph`` object so to provide an abstraction on specific algorithm implementations to the final user.

``CDlib`` provides a standardized input/output facilities for several Community Discovery algorithms: whenever possible, to guarantee literature coherent results, implementations of CD algorithms are inherited from their original projects (acknowledged on the [documentation](https://cdlib.readthedocs.io)).


If you use ``CDlib`` as support to your research consider citing:

> G. Rossetti, L. Milli, R. Cazabet.
> **CDlib: a Python Library to Extract, Compare and Evaluate Communities from Complex Networks.**
> Applied Network Science Journal. 2019. 
> [DOI:10.1007/s41109-019-0165-9]()

## Tutorial and Online Environments

Check out the official [tutorial](https://colab.research.google.com/github/GiulioRossetti/cdlib/blob/master/docs/CDlib.ipynb) to get started!

If you would like to test ``CDlib`` functionalities without installing anything on your machine consider using the preconfigured Jupyter Hub instances offered by [SoBigData++](https://sobigdata.d4science.org/group/sobigdata-gateway/explore?siteId=20371853).

## Installation

``CDlib`` *requires* python>=3.8.

To install the latest version of our library just download (or clone) the current project, open a terminal and run the following commands:

```bash
pip install -r requirements.txt
pip install -r requirements_optional.txt # (Optional) this might not work in Windows systems due to C-based dependencies.
pip install .
```

Alternatively use pip
```bash
pip install cdlib
```

or conda
```bash
conda config --add channels giuliorossetti
conda config --add channels conda-forge
conda install cdlib
```

### Optional Dependencies (pip package)
``CDlib`` relies on a few packages calling C code that can be cumbersome to install on Windows machines: to address such issue, the default installation does not try to install set up such requirements.

Such a choice has been made to allow (even) non *unix user to install the library and get access to its core functionalities. 

To integrate the standard installation with you can either:

- (Windows) manually install the optional packages (versions details are specified in ``requirements_optional.txt``) following the original projects guidelines, or
- (Linux/OSX) run the command:

```bash
pip install cdlib[C]
```

Such caveat will install everything that can be easily automated under Linux/OSX. 

#### (Advanced) 

##### Graph-tool
The only optional dependency that will remain unsatisfied following the previous procedures will be ``graph-tool`` (used to add SBM models). 
If you need it up and running, refer to the official [documentation](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions) and install the conda-forge version of the package.

##### ASLPAw

Since its 2.1.0 release ``ASLPAw`` relies on ``gmpy2`` whose installation through pip is not easy to automatize due to some C dependencies.
To address such issue test the following recipe:

```bash
conda install gmpy2 
pip install shuffle_graph>=2.1.0 similarity-index-of-label-graph>=2.0.1 ASLPAw>=2.1.0
```

In case this does not solve the issue, please refer to the official ``gmpy2`` [installation](https://gmpy2.readthedocs.io/en/latest/intro.html#installation) instructions.

### Optional Dependencies (Conda package)

``CDlib`` relies on a few packages not available through conda: to install it please use pip:

```bash
pip install pycombo
pip install GraphRicciCurvature

conda install gmpy2 
pip install shuffle_graph>=2.1.0 similarity-index-of-label-graph>=2.0.1 ASLPAw>=2.1.0
```

In case ASLPAw installation fails, please refer to the official ``gmpy2`` [installation](https://gmpy2.readthedocs.io/en/latest/intro.html#installation) instructions.


## Collaborate with us!

``CDlib`` is an active project, any contribution is welcome!

If you like to include your model in CDlib feel free to fork the project, open an issue and contact us.

### How to contribute to this project?

Contributing is good, doing it correctly is better! Check out our [rules](https://github.com/GiulioRossetti/cdlib/blob/master/.github/CONTRIBUTING.md), issue a proper [pull request](https://github.com/GiulioRossetti/cdlib/blob/master/.github/PULL_REQUEST_TEMPLATE.md) /[bug report](https://github.com/GiulioRossetti/cdlib/blob/master/.github/ISSUE_TEMPLATE/bug_report.md) / [feature request](https://github.com/GiulioRossetti/cdlib/blob/master/.github/ISSUE_TEMPLATE/feature_request.md).

We are a welcoming community... just follow the [Code of Conduct](https://github.com/GiulioRossetti/cdlib/blob/master/.github/CODE_OF_CONDUCT.md).
