*******************
Temporal Clustering
*******************

TemporalClustering models communities that evolves as time goes by.

Each temporal community clustering observation is a Clustering object, thus it inherits all properties of its specific concrete class.


^^^^^^^^
Overview
^^^^^^^^

.. currentmodule:: cdlib
.. autoclass:: TemporalClustering
    :members:
    :inherited-members:

^^^^^^^
Methods
^^^^^^^

Data transformation and I\O
---------------------------

.. autosummary::

    TemporalClustering.to_json
    TemporalClustering.get_observation_ids
    TemporalClustering.get_clustering_at
    TemporalClustering.add_clustering
    TemporalClustering.get_community

Evaluating Node Clustering
--------------------------

.. autosummary::

    TemporalClustering.clustering_stability_trend


