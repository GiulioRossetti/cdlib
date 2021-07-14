******************
Ranking Algorithms
******************

Let's assume that you ran a set **X** of community discovery algorithms on a given graph **G** and that, for each of the obtained clustering, you computed a set **Y** of fitness scores.

- Is there a way to rank the obtained clusterings by their quality as expressed by **Y**?
- Is it possible to validate the statistical significance of the obtained ranking?
- Can we do the same while comparing different clustering (e.g., using NMI, NF1, ARI, AMI...)?

Don't worry, ``cdlib`` got you covered!

(Yes, we are aware that Community Detection is an ill-posed problem for which `No Free-Lunch`_ can be expected... however, we're not aiming at a general ranking here!)

-------------------------
Ranking by Fitness Scores
-------------------------

.. automodule:: cdlib.evaluation
.. autoclass:: FitnessRanking
    :members:
    :inherited-members:

^^^^^^^
Methods
^^^^^^^

.. autosummary::

    FitnessRanking.rank
    FitnessRanking.topsis
    FitnessRanking.friedman_ranking
    FitnessRanking.bonferroni_post_hoc

--------------------------------
Ranking by Clustering Similarity
--------------------------------

.. automodule:: cdlib.evaluation
.. autoclass:: ComparisonRanking
    :members:
    :inherited-members:

^^^^^^^
Methods
^^^^^^^

.. autosummary::

    ComparisonRanking.rank
    ComparisonRanking.topsis
    ComparisonRanking.friedman_ranking
    ComparisonRanking.bonferroni_post_hoc



.. _`No Free-Lunch`: https://en.wikipedia.org/wiki/No_free_lunch_theorem