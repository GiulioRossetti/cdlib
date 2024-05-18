==============================
Community Events and LifeCycle
==============================

Community events describe the changes in the community structure of a network over time.
The community structure of a network can change due to the arrival or departure of nodes, the creation or dissolution of communities, or the merging or splitting of communities.

The ``cdlib`` library provides a set of tools to analyze the evolution of communities over time, including the detection of community events and the analysis of community life cycles.

The interface of the library is designed to be as simple as possible, allowing users to easily analyze the evolution of communities in their networks.

Check the ``LifeCycle`` class for more details:

.. toctree::
    :maxdepth: 1

    classes/lifecycle.rst


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clustering with Explicit LifeCycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some dynamic community detection algorithms (e.g., Temporal trade-off ones) provide an explicit representation of the life cycle of communities.
In this case it is not necessary to detect community events as post-processing, as the life cycle of communities is already available.

To analyze such pre-computed events apply the following snippet:

.. code-block:: python

    from cdlib import LifeCycle
    from cdlib import algorithms
    import dynetx as dn

    dg = dn.DynGraph()
    for x in range(10):
        g = nx.erdos_renyi_graph(200, 0.05)
        dg.add_interactions_from(list(g.edges()), t=x)
    coms = algorithms.tiles(dg, 2)

    lc = LifeCycle()
    lc.from_temporal_clustering(coms)
    lc.compute_events_from_explicit_matching()



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clustering without Explicit LifeCycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case the dynamic community detection algorithm does not provide an explicit representation of the life cycle of communities, the library provides a set of tools to detect community events and analyze the life cycle of communities.
In particular, the library allows to identify events following four different strategies:

- **Facets** events definition
- **Greene** events definition
- **Asur** events definition
- **Custom** events definition

The first three strategies are based on the definition of community events proposed in the literature, while the last one allows users to define their own events.

To apply one of the first three strategies, use the following snippet:

.. code-block:: python

    from cdlib import LifeCycle
    from networkx.generators.community import LFR_benchmark_graph

    tc = TemporalClustering()
    for t in range(0, 10):
        g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
        )
        coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        tc.add_clustering(coms, t)

    events = LifeCycle(tc)
    events.compute_events("facets") # or "greene" or "asur"

.. note::
    Each strategy has its parameters that can be specified passing a dictionary to the compute_events method.
    In particular, the ``facets`` strategy requires the specification of the ``min_branch_size`` parameter (default 1), while ``greene`` and `asur`` require the specification of the ``threshold`` parameter (default 0.1).


To define custom events, use the following snippet:

.. code-block:: python

    from cdlib import LifeCycle
    from networkx.generators.community import LFR_benchmark_graph

    tc = TemporalClustering()
    for t in range(0, 10):
        g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
        )
        coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        tc.add_clustering(coms, t)

    events = LifeCycle(tc)
    events.compute_events("custom", threshold=0.5) # or any other custom definition


^^^^^^^^^^^^^^^^^^^^^^^^^^
Analyzing Events and Flows
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the community events have been detected, the library provides a set of tools to analyze them.
Each event is characterized by a set of properties, such as the type of event, the communities involved, the nodes involved, and the time of occurrence.

.. note::

    The library assigns a unique identifier to each community of the form ``t_c`` where ``t`` is the time of occurrence and ``c`` is the community identifier.
    E.g., the community with identifier ``2_3`` is the community with identifier ``3`` at time ``2``.

Each tracking strategy defines a different set of events (e.g., creation, dissolution, merging, splitting).
However, ``cdlib`` generalize the concept of event breaking it down into four components. For each generic temporal community ``t_c`` it provide access to:

- **In flow**: the set of nodes that have entered the community ``t_c`` from clusters of time ``t-1``;
- **Out flow**: the set of nodes that will leave the community ``t_c`` at time ``t+1``;
- **From Events**: the set of events that generate the community observed at ``t`` tha involved clusters at time ``t-1``;
- **To Events**: the set of events community ``t_c`` starts at time ``t`` that will affect clusters at time ``t+1``;

All these information can be summarized in a community temporal-dependency digraph called ``polytree``.

Here an example of how to analyze community events and flows:

.. code-block:: python

    from cdlib import LifeCycle
    from networkx.generators.community import LFR_benchmark_graph

    tc = TemporalClustering()
    for t in range(0, 10):
        g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
        )
        coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        tc.add_clustering(coms, t)

    events = LifeCycle(tc)
    events.compute_events("facets") # or "greene" or "asur"
    event_types = events.get_event_types() # provide the list of available events for the detected method (in this case for 'facets')

    out_flow = events.analyze_flow("1_2", "+") # if the community id is not specified all the communities are considered
    in_flow = events.analyze_flow("1_2", "-")
    events = events.get_event("1_2") # to compute events for all communities use the get_events() method

Each event is characterized by its degree of importance for the actual status of the community.
In particular, ``facets`` events are fuzzy events (more than one can occur at the same time) while ``greene`` and ``asur`` events are crisp events (only one can occur at the same time).

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Visualizing Events and Flows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library provides a set of tools to visualize the events and flows detected in the community structure of a network.

.. note::

    The library uses the ``networkx`` library to represent the community structure of a network and the ``matplotlib`` / ``plotly`` library to visualize it.

Here an example of how to visualize community events, flows and polytree:

.. code-block:: python

    from cdlib import LifeCycle
    from cdlib.viz import (
        plot_flow,
        plot_event_radar,
        plot_event_radars,
        typicality_distribution,
        )
    from networkx.generators.community import LFR_benchmark_graph

    tc = TemporalClustering()
    for t in range(0, 10):
        g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
        )
        coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        tc.add_clustering(coms, t)

    events = LifeCycle(tc)
    events.compute_events("facets") # or "greene" or "asur"

    fig = plot_flow(events)
    fig.show()

    fig = plot_event_radar(events, "1_2", direction="+") # only out events
    fig.show()

    fig = plot_event_radars(events, "1_2") # both in and out events
    fig.show()

    fig = typicality_distribution(events, "+")
    fig.show()

    dg = events.polytree()
    fig = nx.draw_networkx(dg, with_labels=True)
    fig.show()

For a detailed description of the available methods and parameters, check the ``Visual Analytics`` section of the ``cdlib`` reference guide.

^^^^^^^^^^^^^^^^
Validating Flows
^^^^^^^^^^^^^^^^

The library provides a set of tools to statistically validate the observed flows against null models.

Here an example of how to validate the observed flows:

.. code-block:: python

    from cdlib import LifeCycle
    from cdlib.lifecycles.validation import validate_flow, validate_all_flows
    from networkx.generators.community import LFR_benchmark_graph

    tc = TemporalClustering()
    for t in range(0, 10):
        g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
        )
        coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        tc.add_clustering(coms, t)

    events = LifeCycle(tc)
    events.compute_events("facets") # or "greene" or "asur"

    cf = events.flow_null("1_2", "+", iterations=1000)  # validate the out flow of community 1_2. Iterations define the number of randomizations to perform.
    vf = events.all_flows_null("+", iterations=1000) # validate all out flows

Both validation methods return a dictionary keyed by set identifier and valued by mean, std, and p-value of the observed flow against the null model.

.. automodule:: cdlib.lifecycles
    :members:
    :undoc-members:

.. autosummary::
    :toctree: generated/

    flow_null
    all_flows_null

