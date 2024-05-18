==========================
Static Community Discovery
==========================

``CDlib`` collects implementations of several Community Discovery algorithms.

To maintain the library organization as clean and resilient to changes as possible, the exposed algorithms are grouped as:

.. toctree::
   :maxdepth: 1

   cd_algorithms/node_clustering.rst
   cd_algorithms/edge_clustering.rst

Moreover, node clustering algorithms are further divided to take into account the type of partition they search for:

- *Crisp* partition (i.e., hard clustering)
- *Overlapping* clustering (i.e., a node can belong to multiple communities);
- *Fuzzy* partition (i.e., soft clustering);
- *Bipartite* clustering (i.e., clustering of bipartite networks).
- *Feature-rich* (node attributed) clustering (i.e., clustering of attributed networks leveraging both topology and node features).
- *Antichains* clustering in DAG (directed acyclic graphs).

For each algorithm, the documentation provides a brief description, the list of parameters, and the reference to the original paper.

----------------
Ensemble Methods
----------------

``CDlib`` implements basilar ensemble facilities to simplify the design of complex analytical pipelines requiring the instantiation of several community discovery algorithms.

Learn how to (i) pool multiple algorithms on the same network, (ii) perform fitness-driven methods' parameter grid search, and (iii) combine the two in a few lines of code.


.. toctree::
   :maxdepth: 1

   ensemble.rst
