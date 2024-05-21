***************************
Evaluation and Benchmarking
***************************

The evaluation of Community Discovery algorithms is not an easy task.
``cdlib`` implements two families of evaluation strategies:

- *Internal* evaluation through fitness scores;
- *External* evaluation through partition comparison.

Moreover, ``cdlib`` integrates both standard *synthetic network benchmarks* and *real networks with annotated ground truths*, thus allowing for testing identified communities against ground truths.

Finally, ``cdlib`` also provides a way to generate *rank* clustering results algorithms over a given input graph.


.. note::
    The following lists are aligned to CD evaluation methods available in the *GitHub main branch* of `cdlib`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Internal Evaluation: Fitness scores
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fitness functions allow to summarize the characteristics of a computed set of communities. ``cdlib`` implements the following quality scores:

.. automodule:: cdlib.evaluation

.. autosummary::
    :toctree: generated/

    avg_distance
    avg_embeddedness
    average_internal_degree
    avg_transitivity
    conductance
    cut_ratio
    edges_inside
    expansion
    fraction_over_median_degree
    hub_dominance
    internal_edge_density
    normalized_cut
    max_odf
    avg_odf
    flake_odf
    scaled_density
    significance
    size
    surprise
    triangle_participation_ratio
    purity


Among the fitness function, a well-defined family of measures is the Modularity-based one:

.. autosummary::
    :toctree: generated/

    erdos_renyi_modularity
    link_modularity
    modularity_density
    modularity_overlap
    newman_girvan_modularity
    z_modularity


Some measures will return an instance of ``FitnessResult`` that takes together min/max/mean/std values of the computed index.

.. autosummary::
    :toctree: generated/

    FitnessResult

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
External Evaluation: Partition Comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is often useful to compare different graph partitions to assess their resemblance.
``cdlib`` implements the following partition comparisons scores:

.. autosummary::
    :toctree: generated/

    adjusted_mutual_information
    mi
    rmi
    normalized_mutual_information
    overlapping_normalized_mutual_information_LFK
    overlapping_normalized_mutual_information_MGH
    variation_of_information
    rand_index
    adjusted_rand_index
    omega
    f1
    nf1
    southwood_index
    rogers_tanimoto_index
    sorensen_index
    dice_index
    czekanowski_index
    fowlkes_mallows_index
    jaccard_index
    sample_expected_sim
    overlap_quality
    geometric_accuracy
    classification_error
    ecs



Some measures will return an instance of ``MatchingResult`` that takes together the computed index's mean and standard deviation values.

.. autosummary::
    :toctree: generated/

    MatchingResult


^^^^^^^^^^^^^^^^^^^^
Synthetic Benchmarks
^^^^^^^^^^^^^^^^^^^^

External evaluation scores can be fruitfully used to compare alternative clusterings of the same network and to assess to what extent an identified node clustering matches a known *ground truth* partition.

To facilitate such a standard evaluation task, ``cdlib`` exposes a set of standard synthetic network generators providing topological community ground truth annotations.

In particular, ``cdlib`` make available benchmarks for:

- *static* community discovery;
- *dynamic* community discovery;
- *feature-rich* (i.e., node-attributed) community discovery.

All details can be found on the dedicated page.

.. toctree::
   :maxdepth: 1

   benchmark.rst


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Networks With Annotated Communities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although evaluating a topological partition against an annotated "semantic" one is not among the safest paths to follow [Peel17]_, ``cdlib`` natively integrates well-known medium-size network datasets with ground-truth communities.

Due to the non-negligible sizes of such datasets, we designed a simple API to gather them transparently from a dedicated remote repository.

All details on remote datasets can be found on the dedicated page.

.. toctree::
   :maxdepth: 1

   datasets.rst


.. _`cdlib`: https://github.com/GiulioRossetti/cdlib

.. [Peel17] Peel, Leto, Daniel B. Larremore, and Aaron Clauset. "The ground truth about metadata and community detection in networks." Science Advances 3.5 (2017): e1602548.