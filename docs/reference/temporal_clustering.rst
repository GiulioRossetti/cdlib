===========================
Dynamic Community Discovery
===========================

Algorithms falling in this category generate communities that evolve as time goes by.

Dynamic algorithms are organized to resemble the taxonomy proposed in [Rossetti18]_

- Instant Optimal,
- Temporal Trade-off

For all details on the available methods to extract and manipulate dynamic communities, please refer to the ``TemporalClustering`` documentation.


.. automodule:: cdlib.algorithms

^^^^^^^^^^^^^^^
Instant Optimal
^^^^^^^^^^^^^^^

This first class of approaches is derived directly from applying static community discovery methods to the dynamic case.
A succession of steps is used to model network evolution, and an optimal partition is identified for each.
Dynamic communities are defined from these optimal partitions by specifying relations that connect topologies found in different, possibly consecutive, instants.

``cdlib`` implements a templating approach to transform every static community discovery algorithm into a dynamic one following a standard *Two-Stage* approach:

- Identify: detect static communities on each step of evolution;
- Match: align the communities found at step t with those found at step t − 1, for each step.

Here is an example of a two-step built on top of Louvain partitions of a dynamic snapshot-sequence graph (where each snapshot is an LFR synthetic graph).

.. code-block:: python

    from cdlib import algorithms
    from cdlib import TemporalClustering
    from networkx.generators.community import LFR_benchmark_graph

    tc = TemporalClustering()
    for t in range(0,10):
        g = LFR_benchmark_graph(n=250, tau1=3, tau2=1.5, mu=0.1, average_degree=5, min_community=20, seed=10)
        coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        tc.add_clustering(coms, t)

For what concerns the second stage (snapshots' node clustering matching), refer to the ``Community Events and LifeCycle`` section of the ``cdlib`` documentation.

^^^^^^^^^^^^^^^^^^
Temporal Trade-Off
^^^^^^^^^^^^^^^^^^

Algorithms belonging to the Temporal Trade-off class process iteratively the evolution of the network.
Moreover, unlike Instant optimal approaches, they consider the network and the communities found in the previous step – or n-previous steps – to identify communities in the current one.
Dynamic Community Discovery algorithms falling into this category can be described by an iterative process:

- Initialization: find communities for the initial state of the network;
- Update: find communities at step t using graph at t and past information for each incoming step.

Currently ``cdlib`` features the following Temporal Trade-off algorithms:

.. autosummary::
    :toctree: generated/

    tiles



.. [Rossetti18] Rossetti, Giulio, and Rémy Cazabet. "Community discovery in dynamic networks: a survey." ACM Computing Surveys (CSUR) 51.2 (2018): 1-37.