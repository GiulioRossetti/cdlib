===============
Node Clustering
===============

Algorithms falling in this category generate communities composed by nodes.
The communities can represent neat, *crisp*, partition as well as *overlapping* or even *fuzzy* ones.

.. note::
    The following lists are aligned to CD methods available in the *GitHub main branch* of `CDlib`_.

    In particular, the current version of the library on pypl (that can be installed through pip) does not include the following algorithms: belief, ga.


.. automodule:: cdlib.algorithms

^^^^^^^^^^^^^^^^^
Crisp Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *partition* if each node belongs to one and only one community.
Methods in this subclass return as result a ``NodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    agdl
    aslpaw
    async_fluid
    belief
    cpm
    chinesewhispers
    der
    edmot
    eigenvector
    em
    ga
    gdmp2
    girvan_newman
    greedy_modularity
    infomap
    label_propagation
    leiden
    louvain
    markov_clustering
    rber_pots
    rb_pots
    scan
    significance_communities
    spinglass
    surprise_communities
    walktrap
    sbm_dl
    sbm_dl_nested


^^^^^^^^^^^^^^^^^^^^^^^
Overlapping Communities
^^^^^^^^^^^^^^^^^^^^^^^

A clustering is said to be *overlapping* if any generic node can be assigned to more than one community.
Methods in this subclass return as result a ``NodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    angel
    big_clam
    conga
    congo
    danmf
    demon
    ego_networks
    egonet_splitter
    kclique
    lais2
    lemon
    lfm
    multicom
    nmnf
    nnsed
    node_perception
    overlapping_seed_set_expansion
    percomvc
    slpa
    wCommunity


^^^^^^^^^^^^^^^^^
Fuzzy Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *fuzzy* if each node can belongs (with a different degree of likelihood) to more than one community.
Methods in this subclass return as result a ``FuzzyNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    frc_fgsn


^^^^^^^^^^^^^^
Node Attribute
^^^^^^^^^^^^^^

Methods in this subclass return as result a ``AttrNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    eva
    ilouvain


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bipartite Graph Communities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods in this subclass return as result a ``BiNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    bimlpa
    CPM_Bipartite
    infomap_bipartite


^^^^^^^^^^^^^^^^^^^^^
Antichain Communities
^^^^^^^^^^^^^^^^^^^^^

Methods in this subclass are designed to extract communities from Directed Acyclic Graphs (DAG) and return as result a ``NodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    siblinarity_antichain


.. _`CDlib`: https://github.com/GiulioRossetti/CDlib