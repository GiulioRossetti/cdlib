==============================
Algorithms Reference Table
==============================

This page provides a compact overview of the algorithms exposed by CDlib.
Complexity values are intentionally high-level summaries, since many methods are heuristic,
output-dependent, or have implementation-specific performance characteristics.

The table is organized by category and links each algorithm to its API reference page.
If an algorithm appears in more than one category in the API, the table uses the most
natural placement and notes shared entries where needed.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Crisp Communities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - agdl
     - Undirected / directed, weighted
     - Super-linear, iterative
     - Crisp
     - AGDL (2012)
     - :doc:`API <generated/cdlib.algorithms.agdl>`
   * - async_fluid
     - Undirected, unweighted
     - Near-linear per iteration
     - Crisp
     - Fluid Communities (2018)
     - :doc:`API <generated/cdlib.algorithms.async_fluid>`
   * - label_propagation
     - Undirected, unweighted
     - Near-linear per iteration
     - Crisp
     - Alias of label_propagation_raghavan
     - :doc:`API <generated/cdlib.algorithms.label_propagation>`
   * - bayan
     - Undirected, weighted
     - Exponential worst-case, exact/heuristic
     - Crisp
     - Bayan (2022)
     - :doc:`API <generated/cdlib.algorithms.bayan>`
   * - belief
     - Undirected, unweighted
     - Iterative, near-linear per pass
     - Crisp
     - Scalable detection by message passing (2014)
     - :doc:`API <generated/cdlib.algorithms.belief>`
   * - cpm
     - Undirected, unweighted
     - Clique-search / exponential worst-case
     - Crisp
     - CPM (2011)
     - :doc:`API <generated/cdlib.algorithms.cpm>`
   * - der
     - Undirected, weighted
     - Spectral / super-linear
     - Crisp
     - Community Detection via Measure Space Embedding (2015)
     - :doc:`API <generated/cdlib.algorithms.der>`
   * - eigenvector
     - Undirected, weighted
     - Spectral, iterative
     - Crisp
     - Eigenvector modularity method (2006)
     - :doc:`API <generated/cdlib.algorithms.eigenvector>`
   * - em
     - Undirected, weighted
     - Iterative, parameter-dependent
     - Crisp
     - Mixture community / EM analysis (2007)
     - :doc:`API <generated/cdlib.algorithms.em>`
   * - ga
     - Undirected, unweighted
     - Heuristic, population-based
     - Crisp
     - GA-Net (2008)
     - :doc:`API <generated/cdlib.algorithms.ga>`
   * - gdmp2
     - Undirected, weighted
     - Near-linear / iterative
     - Crisp
     - Dense subgraph extraction (2012)
     - :doc:`API <generated/cdlib.algorithms.gdmp2>`
   * - girvan_newman
     - Undirected, weighted
     - Cubic / repeated edge-betweenness
     - Crisp
     - Girvan-Newman (2002)
     - :doc:`API <generated/cdlib.algorithms.girvan_newman>`
   * - greedy_modularity
     - Undirected, weighted
     - Roughly O(m log^2 n)
     - Crisp
     - Clauset-Newman-Moore (2004)
     - :doc:`API <generated/cdlib.algorithms.greedy_modularity>`
   * - head_tail
     - Undirected, weighted
     - Local / output-dependent
     - Crisp
     - Head/Tail communities (2010)
     - :doc:`API <generated/cdlib.algorithms.head_tail>`
   * - infomap
     - Directed / undirected, weighted
     - Near-linear / iterative
     - Crisp
     - Infomap (2008)
     - :doc:`API <generated/cdlib.algorithms.infomap>`
   * - kcut
     - Undirected, weighted
     - Heuristic / output-dependent
     - Crisp
     - K-cut family (CDlib implementation)
     - :doc:`API <generated/cdlib.algorithms.kcut>`
   * - label_propagation
     - Undirected, unweighted
     - Near-linear per iteration
     - Crisp
     - Raghavan et al. (2007)
     - :doc:`API <generated/cdlib.algorithms.label_propagation>`
   * - label_propagation_raghavan
     - Undirected, unweighted
     - Near-linear per iteration
     - Crisp
     - Raghavan et al. (2007); alias: label_propagation
     - :doc:`API <generated/cdlib.algorithms.label_propagation_raghavan>`
   * - label_propagation_cordasco_gargano
     - Undirected, unweighted
     - Near-linear per iteration
     - Crisp
     - Cordasco & Gargano label propagation
     - :doc:`API <generated/cdlib.algorithms.label_propagation_cordasco_gargano>`
   * - leiden
     - Directed / undirected, weighted
     - Near-linear average
     - Crisp
     - Leiden (2018)
     - :doc:`API <generated/cdlib.algorithms.leiden>`
   * - louvain
     - Directed / undirected, weighted
     - Near-linear average
     - Crisp
     - Louvain (2008)
     - :doc:`API <generated/cdlib.algorithms.louvain>`
   * - lswl
     - Undirected, unweighted
     - Local / output-dependent
     - Crisp
     - LSWL (2008)
     - :doc:`API <generated/cdlib.algorithms.lswl>`
   * - lswl_plus
     - Undirected, unweighted
     - Local / output-dependent
     - Crisp
     - LSWL+ (2010)
     - :doc:`API <generated/cdlib.algorithms.lswl_plus>`
   * - markov_clustering
     - Undirected, weighted
     - Super-linear, iterative
     - Crisp
     - MCL (2002)
     - :doc:`API <generated/cdlib.algorithms.markov_clustering>`
   * - mcode
     - Undirected, weighted
     - Near-linear / sparse
     - Crisp
     - MCODE (2003)
     - :doc:`API <generated/cdlib.algorithms.mcode>`
   * - mod_m
     - Undirected, weighted
     - Local / output-dependent
     - Crisp
     - Exploring local community structures (2008)
     - :doc:`API <generated/cdlib.algorithms.mod_m>`
   * - mod_r
     - Undirected, weighted
     - Local / output-dependent
     - Crisp
     - Finding local community structure (2005)
     - :doc:`API <generated/cdlib.algorithms.mod_r>`
   * - paris
     - Undirected, weighted
     - Hierarchical / super-linear
     - Crisp
     - Paris (2017)
     - :doc:`API <generated/cdlib.algorithms.paris>`
   * - pycombo
     - Undirected, weighted
     - Heuristic / super-linear
     - Crisp
     - COMBO / PyCombo family
     - :doc:`API <generated/cdlib.algorithms.pycombo>`
   * - rber_pots
     - Directed / undirected, weighted
     - Iterative, super-linear
     - Crisp
     - Reichardt-Bornholdt (2006)
     - :doc:`API <generated/cdlib.algorithms.rber_pots>`
   * - rb_pots
     - Directed / undirected, weighted
     - Iterative, super-linear
     - Crisp
     - Reichardt-Bornholdt / directed modularity (2006, 2008)
     - :doc:`API <generated/cdlib.algorithms.rb_pots>`
   * - ricci_community
     - Undirected, weighted
     - Expensive / transport-based
     - Crisp
     - Ollivier-Ricci community detection
     - :doc:`API <generated/cdlib.algorithms.ricci_community>`
   * - r_spectral_clustering
     - Undirected, weighted
     - Spectral / cubic worst-case
     - Crisp
     - R spectral clustering
     - :doc:`API <generated/cdlib.algorithms.r_spectral_clustering>`
   * - scan
     - Undirected, unweighted
     - Near-linear
     - Crisp
     - SCAN (2007)
     - :doc:`API <generated/cdlib.algorithms.scan>`
   * - significance_communities
     - Undirected, weighted
     - Super-linear, optimization-based
     - Crisp
     - Significant scales in community structure (2013)
     - :doc:`API <generated/cdlib.algorithms.significance_communities>`
   * - spinglass
     - Undirected, weighted
     - Heuristic / exponential worst-case
     - Crisp
     - Statistical mechanics of community detection (2006)
     - :doc:`API <generated/cdlib.algorithms.spinglass>`
   * - spectral
     - Bipartite / undirected, weighted
     - Spectral / cubic worst-case
     - Crisp
     - Spectral clustering family
     - :doc:`API <generated/cdlib.algorithms.spectral>`
   * - surprise_communities
     - Undirected, weighted
     - Super-linear, optimization-based
     - Crisp
     - Asymptotical surprise (2015)
     - :doc:`API <generated/cdlib.algorithms.surprise_communities>`
   * - threshold_clustering
     - Directed, unweighted
     - Near-linear
     - Crisp
     - Threshold clustering
     - :doc:`API <generated/cdlib.algorithms.threshold_clustering>`
   * - walktrap
     - Undirected, weighted
     - Super-linear, iterative
     - Crisp
     - Random walks community detection (2006)
     - :doc:`API <generated/cdlib.algorithms.walktrap>`
   * - sbm_dl
     - Directed / undirected, weighted
     - MCMC / greedy, super-linear
     - Crisp
     - Bayesian SBM inference (2014)
     - :doc:`API <generated/cdlib.algorithms.sbm_dl>`
   * - sbm_dl_nested
     - Directed / undirected, weighted
     - MCMC / greedy, super-linear
     - Crisp
     - Hierarchical SBM inference (2014)
     - :doc:`API <generated/cdlib.algorithms.sbm_dl_nested>`

^^^^^^^^^^^^^^^^^^^^^^^
Overlapping Communities
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - aslpaw
     - Undirected, weighted
     - Iterative, near-linear per pass
     - Overlapping
     - ASLPAw (2014)
     - :doc:`API <generated/cdlib.algorithms.aslpaw>`
   * - angel
     - Undirected, unweighted
     - Local / output-dependent
     - Overlapping
     - ANGEL (2019)
     - :doc:`API <generated/cdlib.algorithms.angel>`
   * - coach
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - COACH
     - :doc:`API <generated/cdlib.algorithms.coach>`
   * - conga
     - Undirected, weighted
     - Super-linear, betweenness-based
     - Overlapping
     - CONGA (2007)
     - :doc:`API <generated/cdlib.algorithms.conga>`
   * - congo
     - Undirected, weighted
     - Super-linear, hierarchical
     - Overlapping
     - CONGO (2008)
     - :doc:`API <generated/cdlib.algorithms.congo>`
   * - core_expansion
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - Core expansion (2020)
     - :doc:`API <generated/cdlib.algorithms.core_expansion>`
   * - dcs
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - Divide and Conquer Strategy
     - :doc:`API <generated/cdlib.algorithms.dcs>`
   * - demon
     - Undirected, weighted
     - Near-linear on ego networks
     - Overlapping
     - DEMON (2012)
     - :doc:`API <generated/cdlib.algorithms.demon>`
   * - dpclus
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - DPCLUS
     - :doc:`API <generated/cdlib.algorithms.dpclus>`
   * - ebgc
     - Undirected, weighted
     - Near-linear / iterative
     - Overlapping
     - EBGC
     - :doc:`API <generated/cdlib.algorithms.ebgc>`
   * - ego_networks
     - Undirected, unweighted
     - O(sum of ego neighborhoods)
     - Overlapping
     - Ego networks
     - :doc:`API <generated/cdlib.algorithms.ego_networks>`
   * - endntm
     - Undirected, weighted
     - Iterative / output-dependent
     - Overlapping
     - EnDNTM
     - :doc:`API <generated/cdlib.algorithms.endntm>`
   * - kclique
     - Undirected, unweighted
     - Exponential in clique size
     - Overlapping
     - Clique percolation (2005)
     - :doc:`API <generated/cdlib.algorithms.kclique>`
   * - graph_entropy
     - Undirected, weighted
     - Super-linear
     - Overlapping
     - Graph entropy clustering
     - :doc:`API <generated/cdlib.algorithms.graph_entropy>`
   * - ipca
     - Undirected, weighted
     - Spectral / super-linear
     - Overlapping
     - IPCA
     - :doc:`API <generated/cdlib.algorithms.ipca>`
   * - lais2
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - Efficient identification of overlapping communities (2005)
     - :doc:`API <generated/cdlib.algorithms.lais2>`
   * - lemon
     - Undirected, weighted
     - Local spectral / output-dependent
     - Overlapping
     - LEMON (2015)
     - :doc:`API <generated/cdlib.algorithms.lemon>`
   * - l1_ppr
     - Undirected, weighted
     - Local push, roughly O(1/epsilon) localized work
     - Overlapping
     - Local Partitioning for Graphs (2006)
     - :doc:`API <generated/cdlib.algorithms.l1_ppr>`
   * - lpam
     - Undirected, weighted
     - Heuristic / super-linear
     - Overlapping
     - Link Partitioning Around Medoids (2021)
     - :doc:`API <generated/cdlib.algorithms.lpam>`
   * - lpanni
     - Undirected, weighted
     - Iterative, near-linear per pass
     - Overlapping
     - LPANNI (2018)
     - :doc:`API <generated/cdlib.algorithms.lpanni>`
   * - lfm
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - LFM (2009)
     - :doc:`API <generated/cdlib.algorithms.lfm>`
   * - multicom
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - Multicom (2018)
     - :doc:`API <generated/cdlib.algorithms.multicom>`
   * - node_perception
     - Undirected, weighted
     - Super-linear
     - Overlapping
     - Node Perception (2015)
     - :doc:`API <generated/cdlib.algorithms.node_perception>`
   * - overlapping_seed_set_expansion
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - Seed Set Expansion (2013)
     - :doc:`API <generated/cdlib.algorithms.overlapping_seed_set_expansion>`
   * - ppr_sweep
     - Undirected, weighted
     - Near-linear / linear-system solve
     - Overlapping
     - PageRank-nibble (2006)
     - :doc:`API <generated/cdlib.algorithms.ppr_sweep>`
   * - hk_sweep
     - Undirected, weighted
     - O(K * m)
     - Overlapping
     - Heat kernel sweep (2009)
     - :doc:`API <generated/cdlib.algorithms.hk_sweep>`
   * - clauset
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - Clauset local modularity (2005)
     - :doc:`API <generated/cdlib.algorithms.clauset>`
   * - umstmo
     - Undirected, weighted
     - O(m log n)
     - Overlapping
     - Union of maximum spanning trees
     - :doc:`API <generated/cdlib.algorithms.umstmo>`
   * - percomvc
     - Undirected, weighted
     - Heuristic / output-dependent
     - Overlapping
     - PercoMCV (2019)
     - :doc:`API <generated/cdlib.algorithms.percomvc>`
   * - slpa
     - Undirected, weighted
     - O(T * m)
     - Overlapping
     - SLPA (2011)
     - :doc:`API <generated/cdlib.algorithms.slpa>`
   * - walkscan
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - WalkSCAN
     - :doc:`API <generated/cdlib.algorithms.walkscan>`
   * - wCommunity
     - Undirected, weighted
     - Local / output-dependent
     - Overlapping
     - Weighted community detection (2010)
     - :doc:`API <generated/cdlib.algorithms.wCommunity>`

^^^^^^^^^^^^^^^^^
Fuzzy Communities
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - frc_fgsn
     - Undirected, weighted
     - Heuristic / super-linear
     - Fuzzy
     - FRC-FGSN (2015)
     - :doc:`API <generated/cdlib.algorithms.frc_fgsn>`
   * - principled_clustering
     - Undirected, weighted
     - Heuristic / super-linear
     - Fuzzy
     - Principled clustering
     - :doc:`API <generated/cdlib.algorithms.principled_clustering>`

^^^^^^^^^^^^^^^^^^^^^^
Attributed Communities
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - eva
     - Attributed, undirected
     - Heuristic / super-linear
     - Attribute-aware
     - EVA (2020)
     - :doc:`API <generated/cdlib.algorithms.eva>`
   * - ilouvain
     - Attributed, undirected
     - Heuristic / super-linear
     - Attribute-aware
     - iLouvain (2015)
     - :doc:`API <generated/cdlib.algorithms.ilouvain>`

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bipartite Graph Communities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - bimlpa
     - Bipartite, weighted
     - Iterative, near-linear per pass
     - Bipartite
     - BiMLPA (2020)
     - :doc:`API <generated/cdlib.algorithms.bimlpa>`
   * - condor
     - Bipartite, weighted
     - Iterative / output-dependent
     - Bipartite
     - CONDOR
     - :doc:`API <generated/cdlib.algorithms.condor>`
   * - CPM_Bipartite
     - Bipartite, weighted
     - Clique-search / exponential worst-case
     - Bipartite
     - Barber (2007)
     - :doc:`API <generated/cdlib.algorithms.CPM_Bipartite>`
   * - infomap_bipartite
     - Bipartite, weighted
     - Near-linear / iterative
     - Bipartite
     - Bipartite Infomap
     - :doc:`API <generated/cdlib.algorithms.infomap_bipartite>`
   * - spectral
     - Bipartite, weighted
     - Spectral / cubic worst-case
     - Bipartite
     - Spectral bipartite clustering
     - :doc:`API <generated/cdlib.algorithms.spectral>`

^^^^^^^^^^^^^^^^^^^^^^^
Antichain Communities
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - siblinarity_antichain
     - DAG, directed
     - Iterative / output-dependent
     - Antichain
     - Siblinarity antichain (2020)
     - :doc:`API <generated/cdlib.algorithms.siblinarity_antichain>`

^^^^^^^^^^^^^^^^^^^
Edge Clustering
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - hierarchical_link_community
     - Undirected, weighted
     - Super-linear, hierarchical
     - Edge
     - Link communities reveal multiscale complexity (2010)
     - :doc:`API <generated/cdlib.algorithms.hierarchical_link_community>`
   * - hierarchical_link_community_w
     - Undirected, weighted
     - Super-linear, hierarchical
     - Edge
     - Weighted link communities
     - :doc:`API <generated/cdlib.algorithms.hierarchical_link_community_w>`
   * - hierarchical_link_community_full
     - Undirected, weighted
     - Super-linear, hierarchical
     - Edge
     - Full link community hierarchy
     - :doc:`API <generated/cdlib.algorithms.hierarchical_link_community_full>`

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dynamic Community Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 12 20 16

   * - Name
     - Network type
     - Complexity
     - Category
     - Reference
     - Docs
   * - tiles
     - Temporal / dynamic, weighted
     - Iterative, O(T * m)
     - Dynamic, temporal trade-off
     - Tiles (2017)
     - :doc:`API <generated/cdlib.algorithms.tiles>`

.. note::

   Some public API names share the same generated reference page, for example
   ``label_propagation`` and the bipartite ``spectral`` entry.
