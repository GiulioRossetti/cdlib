****************
Visual Analytics
****************

At the end of the analytical process is it often useful to visualize the obtained results.
``cdlib`` provides a few built-in facilities to ease such task.

^^^^^^^^^^^^^^^^^^^^^
Network Visualization
^^^^^^^^^^^^^^^^^^^^^

Visualizing a graph is always a good idea (if its size is reasonable).


.. automodule:: cdlib.viz


.. autosummary::
    :toctree: generated/

    plot_network_clusters
    plot_community_graph



^^^^^^^^^^^^^^^
Analytics plots
^^^^^^^^^^^^^^^

Community evaluation outputs can be easily used to generate a visual representation of the main partition characteristics.

.. autosummary::
    :toctree: generated/

    plot_sim_matrix
    plot_com_stat
    plot_com_properties_relation
    plot_scoring
