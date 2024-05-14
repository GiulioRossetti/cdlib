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