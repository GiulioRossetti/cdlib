===============
Node Clustering
===============

Algorithms falling in this category generates communities composed by nodes.
The communities can represent neat, *crisp*, partition as well as *overlapping* or even *fuzzy* ones.


.. automodule:: nclib.algorithms

^^^^^^^^^^^^^^^^^
Crisp Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *partition* if each node belongs to one and only one community.
Methods in this subclass returns as result a ``nclib.NodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    girvan_newman
    em
    scan
    gdmp2
    spinglass
    eigenvector
    agdl
    louvain
    leiden
    rb_pots
    rber_pots
    cpm
    significance_communities
    surprise_communities
    greedy_modularity
    infomap
    walktrap
    label_propagation
    async_fluid
    der


^^^^^^^^^^^^^^^^^^^^^^^
Overlapping Communities
^^^^^^^^^^^^^^^^^^^^^^^

A clustering is said to be *overlapping* if any generic node can be assigned to more than one community.
Methods in this subclass returns as result a ``nclib.NodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    ego_networks
    demon
    angel
    node_perception
    overlapping_seed_set_expansion
    kclique
    lfm
    lais2
    congo
    conga
    lemon
    slpa
    multicom
    big_clam



^^^^^^^^^^^^^^^^^
Fuzzy Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *fuzzy* if each node can belongs (with a different degree of likelihood) to more than one community.
Methods in this subclass returns as result a ``nclib.FuzzyNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    frc_fgsn

