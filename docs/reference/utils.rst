*********
Utilities
*********

``cdlib`` exposes a few utilities to manipulate graph objects generated with ``igraph`` and ``networkx``.


.. automodule:: cdlib.utils


^^^^^^^^^^^^^^^^^^^^
Graph Transformation
^^^^^^^^^^^^^^^^^^^^

Transform ``igraph`` to/from ``networkx`` objects.

.. autosummary::
    :toctree: generated/

    convert_graph_formats

^^^^^^^^^^^^^^^^^^
Identifier mapping
^^^^^^^^^^^^^^^^^^

Remapping of graph nodes. It is often a good idea to limit memory usage and to use progressive integers as node labels.
``cdlib`` automatically - and transparently - makes the conversion for the user; however, this step can be costly: for such reason, the library also exposes facilities to directly pre/post process the network/community data.

.. autosummary::
    :toctree: generated/

    nx_node_integer_mapping
    remap_node_communities

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Global Seeding for Reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cdlib`` provides a utility to globally set the random seed across its algorithms and dependencies:

.. code-block:: python
    import cdlib

    # Set seed for reproducibility
    cdlib.seed(42)

    # All community detection algorithms will now default to use this seed
    from cdlib import algorithms
    import networkx as nx

    G = nx.karate_club_graph()
    communities = algorithms.leiden(G)

    # Reset the seed to the default value
    cdlib.reset_seed()

Using a temporary fixed seed in a context manager:

.. code-block:: python

    from cdlib import fixed_seed

    with fixed_seed(123):
        communities = algorithms.leiden(G)
    # Seed automatically restored


