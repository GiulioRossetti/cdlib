****************
Visual Analytics
****************

At the end of the analytical process, it is often useful to visualize the obtained results.
``cdlib`` provides a few built-in facilities to ease such tasks.

^^^^^^^^^^^^^^^^^^^^^
Network Visualization
^^^^^^^^^^^^^^^^^^^^^

Visualizing a graph is always a good idea (if its size is reasonable).


.. automodule:: cdlib.viz


.. autosummary::
    :toctree: generated/

    plot_network_clusters
    plot_network_highlighted_clusters
    plot_community_graph



^^^^^^^^^^^^^^^
Analytics plots
^^^^^^^^^^^^^^^

Community evaluation outputs can be easily used to represent the main partition characteristics visually.

.. autosummary::
    :toctree: generated/

    plot_sim_matrix
    plot_com_stat
    plot_com_properties_relation
    plot_scoring


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Dynamic Community Events plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamic community detection algorithms can be evaluated using the dynamic community events framework. The results can be visualized using the following functions.

.. autosummary::
    :toctree: generated/

    plot_flow
    plot_event_radar
    plot_event_radars
    typicality_distribution