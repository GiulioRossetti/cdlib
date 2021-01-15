******************************
Community Discovery algorithms
******************************

``CDlib`` collects implementations of several Community Discovery algorithms.

To maintain the library organization as clean and resilient as possible the approaches are grouped following a simple, two level, rationale:

1. The first distinction is made on the object clustered, thus separating **Node Clustering** and **Edge Clustering** algorithms;
2. The second distinction is made on the specific kind of partition each one of them generates: **Crisp**, **Overlapping** or **Fuzzy**.

This documentation follows the same rationale.

.. toctree::
   :maxdepth: 1

   cd_algorithms/node_clustering.rst
   cd_algorithms/edge_clustering.rst

Finally ``CDlib`` implements also time-aware algorithms (often referred as Dynamic Community Discovery approaches).

.. toctree::
   :maxdepth: 1

   cd_algorithms/temporal_clustering.rst
