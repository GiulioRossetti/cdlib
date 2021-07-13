==============================
Community Discovery algorithms
==============================

``CDlib`` collects implementations of several Community Discovery algorithms.

To maintain the library organization as clean and resilient to changes as possible, the exposed algorithms are grouped following a simple rationale:

1. Algorithms designed for static networks, and
2. Algorithms designed for dynamic networks.

Moreover, within each category, ``CDlib`` groups together approaches sharing the same set of high-level characteristics.

In particular, static algorithms are organized into:

- Those searching for a *crisp* partition of the node-set;
- Those searching for an *overlapping* clustering of the node-set;
- Those that search for a *fuzzy* partition of the node-set;
- Those that cluster *edges*;
- Those that are designed to partition *bipartite* networks;
- Those that are designed to cluster *feature-rich* (node attributed) networks;
- Those that search for *antichains* in DAG (directed acyclic graphs).

Dynamic algorithms, conversely, are organized to resemble the taxonomy proposed in [Rossetti18]_

- Instant Optimal,
- Temporal Trade-off

This documentation follows the same rationale.

.. toctree::
   :maxdepth: 1

   cd_algorithms/node_clustering.rst
   cd_algorithms/temporal_clustering.rst


----------------
Ensemble Methods
----------------

``CDlib`` implements basilar ensemble facilities to simplify the design of complex analytical pipelines requiring the instantiation of several community discovery algorithms.

Learn how to (i) pool multiple algorithms on the same network, (ii) perform fitness-driven methods' parameter grid search, and (iii) combine the two in few lines of code.


.. toctree::
   :maxdepth: 1

   ensemble.rst

-------
Summary
-------

If you need a summary on the available algorithms and their properties (accepted graph types, community characteristics, computational complexity) refer to:

.. toctree::
   :maxdepth: 1

   cd_algorithms/algorithms.rst


.. [Rossetti18] Rossetti, Giulio, and Rémy Cazabet. "Community discovery in dynamic networks: a survey." ACM Computing Surveys (CSUR) 51.2 (2018): 1-37.