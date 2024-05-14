================
Ensemble Methods
================

Methods to automate the execution of multiple community detection algorithm(s) instances.


.. automodule:: cdlib.ensemble

---------------------
Configuration Objects
---------------------

Ranges can be specified to automate the execution of the same method while varying (part of) its inputs.

``Parameter`` allows to specify ranges for numeric parameters, while ``BoolParamter`` for boolean ones.

.. autosummary::
    :toctree: generated/

    Parameter
    BoolParameter


----------------------
Multiple Instantiation
----------------------

Two scenarios often arise when applying community discovery algorithms to a graph:

1. the need to compare the results obtained by a given algorithm while varying its parameters
2. the need to compare the multiple algorithms

``cdlib`` allows to do so by leveraging, respectively, ``grid_execution`` and ``pool``.


.. autosummary::
    :toctree: generated/

    grid_execution
    pool

----------------------------
Optimal Configuration Search
----------------------------

In some scenarios, it could be helpful to delegate to the library the selection of the method parameters to obtain a partition that optimizes a given quality function.
``cdlib`` allows to do so using the methods ``grid_search`` and ``random_search``.
Finally, ``pool_grid_filter`` generalizes such an approach, allowing one to obtain the optimal partitions from a pool of different algorithms.

.. autosummary::
    :toctree: generated/

    grid_search
    random_search
    pool_grid_filter