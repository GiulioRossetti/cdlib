===============
Node Clustering
===============

Algorithms falling in this category generate communities composed of nodes.
The communities can represent neat, *crisp*, partitions and *overlapping* or even *fuzzy* ones.

.. note::
    The following lists are aligned to CD methods available in the *GitHub main branch* of `CDlib`_.


.. automodule:: cdlib.algorithms


^^^^^^^^^^^^^^^^^
Crisp Communities
^^^^^^^^^^^^^^^^^

A clustering is considered a *partition* if each node belongs to one and only one community.
As a result, methods in this subclass return a ``NodeClustering`` object instance.


.. autosummary::
    :toctree: ../generated/

    agdl
    async_fluid
    bayan
    belief
    cpm
    der
    eigenvector
    em
    ga
    gdmp2
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
    mcode
    mod_m
    mod_r
    paris
    pycombo
    rber_pots
    rb_pots
    ricci_community
    r_spectral_clustering
    scan
    significance_communities
    spinglass
    surprise_communities
    walktrap
    sbm_dl
    sbm_dl_nested
    spectral
    threshold_clustering


^^^^^^^^^^^^^^^^^^^^^^^
Overlapping Communities
^^^^^^^^^^^^^^^^^^^^^^^

A clustering is said to be *overlapping* if any generic node can be assigned to more than one community.
As a result, methods in this subclass return a ``NodeClustering`` object instance.

.. autosummary::
    :toctree: ../generated/

    aslpaw
    angel
    coach
    conga
    congo
    core_expansion
    dcs
    demon
    dpclus
    ebgc
    ego_networks
    endntm
    kclique
    graph_entropy
    ipca
    lais2
    lemon
    lpam
    lpanni
    lfm
    multicom
    node_perception
    overlapping_seed_set_expansion
    umstmo
    percomvc
    slpa
    walkscan
    wCommunity


^^^^^^^^^^^^^^^^^
Fuzzy Communities
^^^^^^^^^^^^^^^^^

A clustering is *fuzzy* if each node can belong (with a different degree of likelihood) to more than one community.
As a result, methods in this subclass return a ``FuzzyNodeClustering`` object instance.

.. autosummary::
    :toctree: ../generated/

    frc_fgsn
    principled_clustering


^^^^^^^^^^^^^^
Node Attribute
^^^^^^^^^^^^^^

As a result, methods in this subclass return a ``AttrNodeClustering`` object instance.

.. autosummary::
    :toctree: ../generated/

    eva
    ilouvain


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bipartite Graph Communities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a result, methods in this subclass return a ``BiNodeClustering`` object instance.

.. autosummary::
    :toctree: ../generated/

    bimlpa
    condor
    CPM_Bipartite
    infomap_bipartite
    spectral


^^^^^^^^^^^^^^^^^^^^^
Antichain Communities
^^^^^^^^^^^^^^^^^^^^^

Methods in this subclass are designed to extract communities from Directed Acyclic Graphs (DAG) and return. As a result, a ``NodeClustering`` object instance.

.. autosummary::
    :toctree: ../generated/

    siblinarity_antichain


---------------
Edge Clustering
---------------

Algorithms falling in this category generate communities composed of edges.
They return, as a result, a ``EdgeClustering`` object instance.

.. autosummary::
    :toctree: ../generated/

    hierarchical_link_community



.. _`CDlib`: https://github.com/GiulioRossetti/CDlib