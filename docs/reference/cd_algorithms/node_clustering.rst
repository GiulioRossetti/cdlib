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
    aslpaw
    async_fluid
    chinese_whispers
    cpm
    der
    edmot
    eigenvector
    em
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
Methods in this subclass returns as result a ``NodeClustering`` object instance.

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


^^^^^^^^^^^^^^^^^
Fuzzy Communities
^^^^^^^^^^^^^^^^^

A clustering is said to be a *fuzzy* if each node can belongs (with a different degree of likelihood) to more than one community.
Methods in this subclass returns as result a ``FuzzyNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    frc_fgsn


^^^^^^^^^^^^^^
Node Attribute
^^^^^^^^^^^^^^

Methods in this subclass returns as result a ``AttrNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    eva
    ilouvain


^^^^^^^^^^^^^^
Bipartite Node
^^^^^^^^^^^^^^

Methods in this subclass returns as result a ``BiNodeClustering`` object instance.

.. autosummary::
    :toctree: algs/

    bimlpa