**********
Evaluation
**********


^^^^^^^^^^^^^^^^^
Fitness Functions
^^^^^^^^^^^^^^^^^


.. automodule:: nclib.evaluation



.. autosummary::
    :toctree: eval/

    average_internal_degree
    conductance
    cut_ratio
    edges_inside
    expansion
    fraction_over_median_degree
    internal_edge_density
    link_modularity
    normalized_cut
    max_odf
    avg_odf
    flake_odf
    significance
    size
    surprise
    triangle_participation_ratio


.. autosummary::
    :toctree: eval/

    erdos_renyi_modularity
    newman_girvan_modularity
    modularity_density
    z_modularity


Some measures will return an instance of ``FitnessResult`` that takes together min/max/mean/std values of the computed index.

.. autosummary::
    :toctree: eval/

    FitnessResult

^^^^^^^^^^^^^^^^^^^^^
Partition Comparisons
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: eval/

    adjusted_mutual_information
    adjusted_rand_index
    f1
    nf1
    normalized_mutual_information
    omega
    overlapping_normalized_mutual_information
    variation_of_information


Some measures will return an instance of ``MatchingResult`` that takes together mean and standard deviation values of the computed index.

.. autosummary::
    :toctree: eval/

    MatchingResult
