===============
Node Clustering
===============

Algorithms falling in this category generates communities composed by nodes.
The communities can represent neat, *crisp*, partition as well as *overlapping* or even *fuzzy* ones.


.. automodule:: cdlib.algorithms

^^^^^^^^^^^^^^^^^
Crisp Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *partition* if each node belongs to one and only one community.
Methods in this subclass returns as result a ``NodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    agdl
    async_fluid
    cpm
    der
    eigenvector
    em
    gdmp2
    girvan_newman
    greedy_modularity
    infomap
    label_propagation
    leiden
    louvain
    rber_pots
    rb_pots
    scan
    significance_communities
    spinglass
    surprise_communities
    walktrap


^^^^^^^^^^^^^^^^^^^^^^^
Overlapping Communities
^^^^^^^^^^^^^^^^^^^^^^^

A clustering is said to be *overlapping* if any generic node can be assigned to more than one community.
Methods in this subclass returns as result a ``NodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    angel
    big_clam
    conga
    congo
    demon
    ego_networks
    kclique
    lais2
    lemon
    lfm
    multicom
    node_perception
    overlapping_seed_set_expansion
    slpa


^^^^^^^^^^^^^^^^^
Fuzzy Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *fuzzy* if each node can belongs (with a different degree of likelihood) to more than one community.
Methods in this subclass returns as result a ``FuzzyNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    frc_fgsn

