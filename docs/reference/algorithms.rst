==========================
Static Community Discovery
==========================

``CDlib`` collects implementations of several Community Discovery algorithms.

To maintain the library organization as clean and resilient to changes as possible, the exposed algorithms are grouped following a simple rationale:

- Those searching for a *crisp* partition of the node-set;
- Those searching for an *overlapping* clustering of the node-set;
- Those that search for a *fuzzy* partition of the node-set;
- Those that cluster *edges*;
- Those that are designed to partition *bipartite* networks;
- Those that are designed to cluster *feature-rich* (node attributed) networks;
- Those that search for *antichains* in DAG (directed acyclic graphs).


.. toctree::
   :maxdepth: 1

   cd_algorithms/node_clustering.rst
   cd_algorithms/edge_clustering.rst


----------------
Ensemble Methods
----------------

``CDlib`` implements basilar ensemble facilities to simplify the design of complex analytical pipelines requiring the instantiation of several community discovery algorithms.

Learn how to (i) pool multiple algorithms on the same network, (ii) perform fitness-driven methods' parameter grid search, and (iii) combine the two in a few lines of code.


.. toctree::
   :maxdepth: 1

   ensemble.rst
