===========================
Dynamic Community Discovery
===========================

Algorithms falling in this category generates communities that evolve as time goes by.


.. automodule:: cdlib.algorithms

^^^^^^^^^^^^^^^
Instant Optimal
^^^^^^^^^^^^^^^

This first class of approaches is derived directly from the application of static community discovery methods to the dynamic case.
A succession of steps is used to model network evolution, and for each of them is identified an optimal partition.
Dynamic communities are defined from these optimal partitions by specifying relations that connect topologies found in different, possibly consecutive, instants.

``cdlib`` implements a templating approach to transform every static community discovery algorithm in a dynamic one following a standard *Two-Stage* approach:

- Identify: detect static communities on each step of evolution;
- Match: align the communities found at step t with the ones found at step t − 1, for each step.

Here's an example of a two-step built on top of Louvain partitions of a dynamic snapshot-sequence graph (where each snapshot is an LFR synthetic graph).

.. code-block:: python

    from cdlib import algorithms
    from cdlib import TemporalClustering
    from networkx.generators.community import LFR_benchmark_graph

    tc = TemporalClustering()
    for t in range(0,10):
        g = LFR_benchmark_graph(n=250, tau1=3, tau2=1.5, mu=0.1, average_degree=5, min_community=20, seed=10)
        coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        tc.add_clustering(coms, t)

For what concerns the second stage (snapshots' node clustering matching) it is possible to parametrize the set similarity function as follows (example made with a standard Jaccard similarity):

.. code-block:: python

    jaccard = lambda x, y:  len(set(x) & set(y)) / len(set(x) | set(y))
    matches = tc.community_matching(jaccard, two_sided=True)

For all details on the available methods to extract and manipulate dynamic communities please refer to the ``TemporalClustering`` documentation.

^^^^^^^^^^^^^^^^^^
Temporal Trade-Off
^^^^^^^^^^^^^^^^^^

Algorithms belonging to the Temporal Trade-off class process iteratively the evolution of the network.
Moreover, unlike Instant optimal approaches, they take into account the network and the communities found in the previous step – or n-previous steps – to identify communities in the current one.
Dynamic Community Discovery algorithms falling into this category can be described by an iterative process:

- Initialization: find communities for the initial state of the network;
- Update: for each incoming step, find communities at step t using graph at t and past information.

.. autosummary::
    :toctree: algs/

    tiles

