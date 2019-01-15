*********
Reference
*********

``NClib`` composes of several modules, each one fulfilling a different task related to community detection.

^^^^^^^^^^^^^^^
nclib.community
^^^^^^^^^^^^^^^

This module contains the community detection algorithms.
All algorithms can be instantiated on both ``networkx`` and ``igraph`` data structures: the conversion among the two formats (whenever needed) happens automatically in the background.

.. toctree::
    :maxdepth: 1

    algorithms.rst

^^^^^^^^^^^^^^
nclib.ensemble
^^^^^^^^^^^^^^

This module contains a few facilities to design algorithm(s) pooling and optimal partition identification.

.. toctree::
    :maxdepth: 1

    ensemble.rst

^^^^^^^^^^^^^^^^
nclib.evaluation
^^^^^^^^^^^^^^^^

This module expose several fitness functions and partition comparison metrics to evaluate the outputs of the implemented algorithms.

.. toctree::
    :maxdepth: 1

    evaluation.rst

^^^^^^^^^^^^^^^
nclib.readwrite
^^^^^^^^^^^^^^^

This module expose input/output facilities to save/load analytical results

.. toctree::
    :maxdepth: 1

    readwrite.rst

^^^^^^^^^
nclib.viz
^^^^^^^^^

This module expose some predefined visual facilities to facilitate the inspection of the computed communities and their properties.

.. toctree::
    :maxdepth: 1

    viz.rst


^^^^^^^^^^^
nclib.utils
^^^^^^^^^^^

This module provides additional functionalities tailored to handle/check/transform ``nclib`` data structures.

.. toctree::
    :maxdepth: 1

    utils.rst