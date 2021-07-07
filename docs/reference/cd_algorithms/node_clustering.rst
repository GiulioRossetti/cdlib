===============
Node Clustering
===============

Algorithms falling in this category generate communities composed by nodes.
The communities can represent neat, *crisp*, partition as well as *overlapping* or even *fuzzy* ones.

.. note::
    The following lists are aligned to CD methods available in the *GitHub main branch* of `CDlib`_.


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
    gemsec
    girvan_newman
    greedy_modularity
    head_tail
    infomap
    kcut
    label_propagation
    leiden
    louvain
    lswl
    lswl_plus
    markov_clustering
    mod_m
    mod_r
    paris
    pycombo
    rber_pots
    rb_pots
    ricci_community
    scan
    significance_communities
    spinglass
    surprise_communities
    walktrap
    sbm_dl
    sbm_dl_nested
    scd
    threshold_clustering


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
    core_expansion
    danmf
    dcs
    demon
    ego_networks
    egonet_splitter
    endntm
    kclique
    lais2
    lemon
    lpam
    lpanni
    lfm
    multicom
    mnmf
    nnsed
    node_perception
    overlapping_seed_set_expansion
    umstmo
    percomvc
    slpa
    symmnmf
    walkscan
    wCommunity


^^^^^^^^^^^^^^^^^
Fuzzy Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *fuzzy* if each node can belongs (with a different degree of likelihood) to more than one community.
Methods in this subclass return as result a ``FuzzyNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    frc_fgsn
    principled_clustering


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
    condor
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