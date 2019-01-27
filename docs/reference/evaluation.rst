**********
Evaluation
**********


^^^^^^^^^^^^^^^^^
Fitness Functions
^^^^^^^^^^^^^^^^^


.. automodule:: nclib.evaluation



.. autosummary::
    :toctree: eval/

    link_modularity
    normalized_cut
    internal_edge_density
    average_internal_degree
    fraction_over_median_degree
    expansion
    cut_ratio
    edges_inside
    conductance
    max_odf
    avg_odf
    flake_odf
    triangle_participation_ratio
    newman_girvan_modularity
    erdos_renyi_modularity
    modularity_density
    z_modularity
    surprise
    significance
    size

Some measures will return an instance of ``FitnessResult`` that takes together min/max/mean/std values of the computed index.

.. autosummary::
    :toctree: eval/

    FitnessResult

^^^^^^^^^^^^^^^^^^^^^
Partition Comparisons
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: eval/

    normalized_mutual_information
    overlapping_normalized_mutual_information
    omega
    f1
    nf1
    adjusted_rand_index
    adjusted_mutual_information
    variation_of_information

Some measures will return an instance of ``MatchingResult`` that takes together mean and standard deviation values of the computed index.

.. autosummary::
    :toctree: eval/

    MatchingResult
