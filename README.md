# CDlib - Community Discovery Library
[![Coverage Status](https://coveralls.io/repos/github/GiulioRossetti/cdlib/badge.svg?branch=master)](https://coveralls.io/github/GiulioRossetti/cdlib?branch=master)
[![Build Status](https://travis-ci.org/GiulioRossetti/cdlib.svg?branch=master)](https://travis-ci.org/GiulioRossetti/cdlib)
[![Documentation Status](https://readthedocs.org/projects/cdlib/badge/?version=latest)](http://cdlib.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/GiulioRossetti/cdlib/shield.svg)](https://pyup.io/repos/github/GiulioRossetti/cdlib/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/GiulioRossetti/nclib.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/GiulioRossetti/nclib/context:python)
[![pyversions](https://img.shields.io/pypi/pyversions/cdlib.svg)](https://badge.fury.io/py/cdlib)
[![PyPI version](https://badge.fury.io/py/cdlib.svg)](https://badge.fury.io/py/cdlib)
[![DOI](https://zenodo.org/badge/159944561.svg)](https://zenodo.org/badge/latestdoi/159944561)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FGiulioRossetti%2Fcdlib.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FGiulioRossetti%2Fcdlib?ref=badge_shield)


``CDlib`` provides implementations of several community discovery algorithms.
Moreover, it implements a wide set of partition evaluation measures as well as predefined visualization facilities.

``CDlib`` is designed around the ``networkx`` python library: however, when needed, it takes care to authomatically convert (from and to) ``igraph`` object so to provide an abstraction on specific algorithm implementations to the final user.

This library provides a standardized input/output for several existing Community Discovery algorithms: however, the implementations of all CD algorithms are inherited from existing projects
The original CD projects embedded in `CDlib` are acknowledged on the documentation website.

Check out the [tutorial](https://colab.research.google.com/github/KDDComplexNetworkAnalysis/CNA_Tutorials/blob/master/CDlib.ipynb) to get started!

## Installation

``CDlib`` *requires* python>=3.7.

To install the library just download (or clone) the current project and copy the ndlib folder in the root of your application.

Alternatively use pip:
```bash
pip install cdlib
```

## Collaborate with us!

``CDlib`` is an active project, any contribution is welcome!

If you like to include your model in CDlib feel free to fork the project, open an issue and contact us.

### How to contribute to this project?

Contributing is good, doing it correctly is better! Check out our [rules](https://github.com/GiulioRossetti/cdlib/blob/master/.github/CONTRIBUTING.md), issue a proper [pull request](https://github.com/GiulioRossetti/cdlib/blob/master/.github/PULL_REQUEST_TEMPLATE.md) /[bug report](https://github.com/GiulioRossetti/cdlib/blob/master/.github/ISSUE_TEMPLATE/bug_report.md) / [feature request](https://github.com/GiulioRossetti/cdlib/blob/master/.github/ISSUE_TEMPLATE/feature_request.md).

We are a welcoming community... just follow the [Code of Conduct](https://github.com/GiulioRossetti/cdlib/blob/master/.github/CODE_OF_CONDUCT.md).


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FGiulioRossetti%2Fcdlib.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FGiulioRossetti%2Fcdlib?ref=badge_large)
