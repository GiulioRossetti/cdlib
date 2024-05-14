.. CDlib documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |date| date::

CDlib - Community Detection Library
===================================

``CDlib`` is a Python software package that allows extracting, comparing, and evaluating communities from complex networks.

The library provides a standardized input/output for several Community Detection algorithms.
The implementations of all CD algorithms are inherited from existing projects; each acknowledged in the dedicated method reference page.

If you want to test ``CDlib`` functionalities without installing it on your machine, consider using the preconfigured Jupyter Hub instances offered by the EU funded `SoBigData`_ research infrastructure.

If you use ``CDlib`` in your research please cite the following paper:

   G. Rossetti, L. Milli, R. Cazabet.
   **CDlib: a Python Library to Extract, Compare and Evaluate Communities from Complex Networks.**
   Applied Network Science Journal. 2019. DOI:10.1007/s41109-019-0165-9

================ =================== ==================  ==========  ===============
   **Date**      **Python Versions**   **Main Author**   **GitHub**      **pypl**
|date|                 3.8-3.9       `Giulio Rossetti`_  `Source`_   `Distribution`_
================ =================== ==================  ==========  ===============


^^^^^^^^^^^^^^
CDlib Dev Team
^^^^^^^^^^^^^^

======================= ============================
**Name**                **Contribution**
`Giulio Rossetti`_      Library Design/Documentation
`Letizia Milli`_        Community Models Integration
`Rémy Cazabet`_         Visualization
`Salvatore Citraro`_    Community Models Integration
`Andrea Failla`_        Community Models Integration
======================= ============================


.. toctree::
   :maxdepth: 1
   :hidden:

   overview.rst
   installing.rst
   tutorial.rst
   reference/reference.rst
   bibliography.rst


.. _`Giulio Rossetti`: http://giuliorossetti.github.io
.. _`Letizia Milli`: https://github.com/letiziam
.. _`Salvatore Citraro`: https://github.com/dsalvaz
.. _`Rémy Cazabet`: http://cazabetremy.fr
.. _`Andrea Failla`: http://andreafailla.github.io
.. _`Source`: https://github.com/GiulioRossetti/CDlib
.. _`Distribution`: https://pypi.python.org/pypi/CDlib
.. _`SoBigData`: https://sobigdata.d4science.org/group/sobigdata-gateway/explore?siteId=20371853