**********
Evaluation
**********

The evaluation of Community Discovery algorithms is not an easy task.
``CDlib`` implements two families of evaluation strategies:

- Internal evaluation through quality scores
- External evaluation through partitions comparison

^^^^^^^^^^^^^^^^^
Fitness Functions
^^^^^^^^^^^^^^^^^

Fitness functions allows to summarize the characteristics of a computed set of communities. ``CDlib`` implements the following quality scores:

.. automodule:: cdlib.evaluation


.. autosummary::
    :toctree: eval/

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


Among the fitness function a well-defined family of measures is the Modularity-based one:

.. autosummary::
    :toctree: eval/

    erdos_renyi_modularity
    link_modularity
    modularity_density
    newman_girvan_modularity
    z_modularity


Some measures will return an instance of ``FitnessResult`` that takes together min/max/mean/std values of the computed index.

.. autosummary::
    :toctree: eval/

    FitnessResult

^^^^^^^^^^^^^^^^^^^^^
Partition Comparisons
^^^^^^^^^^^^^^^^^^^^^

It is often useful to compare different graph partition to assess their resemblance (i.e., to perform ground truth testing).
``CDlib`` implements the following partition comparisons scores:

.. autosummary::
    :toctree: eval/

    adjusted_mutual_information
    adjusted_rand_index
    f1
    nf1
    normalized_mutual_information
    omega
    overlapping_normalized_mutual_information_LFK
    overlapping_normalized_mutual_information_MGH
    variation_of_information


Some measures will return an instance of ``MatchingResult`` that takes together mean and standard deviation values of the computed index.

.. autosummary::
    :toctree: eval/

    MatchingResult
