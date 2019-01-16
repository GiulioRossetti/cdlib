******************************
Community Discovery algorithms
******************************


.. automodule:: nclib.community

^^^^^^^^^^^^^^^^
Modularity based
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: algs/

    greedy_modularity
    louvain
    leiden
    rb_pots
    rber_pots
    cpm
    significance_communities
    surprise_communities

^^^^^^^^^^^^
Node-Centric
^^^^^^^^^^^^

.. autosummary::
    :toctree: algs/

    ego_networks
    demon
    angel
    node_perception
    overlapping_seed_set_expansion
    lemon

^^^^^^^^^^^^^^^^^
Diffusive Process
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: algs/

    label_propagation
    async_fluid
    slpa
    multicom
    markov_clustering

^^^^^^^^^^^^
Random walks
^^^^^^^^^^^^

.. autosummary::
    :toctree: algs/

    infomap
    walktrap


^^^^^^^^^^
Structural
^^^^^^^^^^

.. autosummary::
    :toctree: algs/

    der
    big_clam
    kclique
    girvan_newman
    em
    lfm
    scan
    hierarchical_link_community
    lais2
    gdmp2
    spinglass
    eigenvector
    congo
    conga
    agdl
