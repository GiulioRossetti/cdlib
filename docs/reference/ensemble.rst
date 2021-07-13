================
Ensemble Methods
================

Methods to automate the execution of multiple instances of community detection algorithm(s).


.. automodule:: cdlib.ensemble

---------------------
Configuration Objects
---------------------

Ranges can be specified to automate the execution of a same method while varying (part of) its inputs.

``Parameter`` allows to specify ranges for numeric parameters, while ``BoolParamter`` for boolean ones.

.. autosummary::
    :toctree: generated/

    Parameter
    BoolParameter


----------------------
Multiple Instantiation
----------------------

Two scenarios often arise when applying community discovery algorithms to a graph:

1. the need to compare the results obtained by a give algorithm while varying its parameters
2. the need to compare the multiple algorithms

``cdlib`` allows to do so by leveraging, respectively, ``grid_execution`` and ``pool``.


.. autosummary::
    :toctree: generated/

    grid_execution
    pool

----------------------------
Optimal Configuration Search
----------------------------

In some scenarios it could be helpful delegate to the library the selection of the method parameters to obtain a partition that optimize a given quality function.
``cdlib`` allows to do so using the methods ``grid_search`` and ``random_search``.
Finally, ``pool_grid_filter`` generalizes such approach allowing to obtain the optimal partitions from a pool of different algorithms.

.. autosummary::
    :toctree: generated/

    grid_search
    random_search
    pool_grid_filter
