*********************
Fuzzy Node Clustering
*********************

^^^^^^^^
Overview
^^^^^^^^

.. currentmodule:: cdlib
.. autoclass:: FuzzyNodeClustering
    :members:
    :inherited-members:


^^^^^^^
Methods
^^^^^^^

Data transformation and I\O
---------------------------

.. autosummary::

    FuzzyNodeClustering.to_json
    FuzzyNodeClustering.to_node_community_map

Evaluating Node Clustering
--------------------------

.. autosummary::

    FuzzyNodeClustering.link_modularity
    FuzzyNodeClustering.normalized_cut
    FuzzyNodeClustering.internal_edge_density
    FuzzyNodeClustering.average_internal_degree
    FuzzyNodeClustering.fraction_over_median_degree
    FuzzyNodeClustering.expansion
    FuzzyNodeClustering.cut_ratio
    FuzzyNodeClustering.edges_inside
    FuzzyNodeClustering.conductance
    FuzzyNodeClustering.max_odf
    FuzzyNodeClustering.avg_odf
    FuzzyNodeClustering.flake_odf
    FuzzyNodeClustering.triangle_participation_ratio
    FuzzyNodeClustering.newman_girvan_modularity
    FuzzyNodeClustering.erdos_renyi_modularity
    FuzzyNodeClustering.modularity_density
    FuzzyNodeClustering.z_modularity
    FuzzyNodeClustering.surprise
    FuzzyNodeClustering.significance

