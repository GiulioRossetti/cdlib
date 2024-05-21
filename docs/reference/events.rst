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
    import networkx as nx

    dg = dn.DynGraph()
    for x in range(10):
        g = nx.erdos_renyi_graph(200, 0.05)
        dg.add_interactions_from(list(g.edges()), t=x)

    coms = algorithms.tiles(dg, 2)

    lc = LifeCycle(coms)
    lc.compute_events_from_explicit_matching()



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clustering without Explicit LifeCycle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case the dynamic community detection algorithm does not provide an explicit representation of the life cycle of communities, the library provides a set of tools to detect community events and analyze the life cycle of communities.
In particular, the library allows to identify events following four different strategies:

- **Facets** events definition [Failla24]_
- **Greene** events definition [Greene2010]_
- **Asur** events definition [Asur2009]_
- **Custom** events definition

The first three strategies are based on the definition of community events proposed in the literature, while the last one allows users to define their own events.

To apply one of the first three strategies, use the following snippet:

.. code-block:: python

    from cdlib import LifeCycle, TemporalClustering, algorithms
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
    In particular, the ``facets`` strategy requires the specification of the ``min_branch_size`` parameter (default 1), while ``greene`` and ``asur`` require the specification of the ``threshold`` parameter (default 0.1).


To define custom events, use the following snippet:

.. code-block:: python

    from cdlib import LifeCycle, TemporalClustering, algorithms
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
    jaccard = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
    events.compute_events_with_custom_matching(jaccard, threshold=0.3, two_sided=True)

In the above snippet, the ``jaccard`` function is used to define the similarity between two communities.
The ``threshold`` parameter is used to define the minimum similarity required to consider two communities one an evolution of the other.
Changing the similarity function and the threshold allows users to define their own matching strategies.

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

    from cdlib import LifeCycle, TemporalClustering, algorithms
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

    ev = events.get_event("1_2") # to compute events for all communities use the get_events() method
    print(ev.out_flow)  # to get the out flow of the community 1_2
    print(ev.in_flow)  # to get the in flow of the community 1_2
    print(ev.from_event)  # to get the from events of the community 1_2
    print(ev.to_event)  # to get the to events of the community 1_2

    out_flow = events.analyze_flow("1_2", "+") # if the community id is not specified all the communities are considered
    in_flow = events.analyze_flow("1_2", "-")

Each event is characterized by its degree of importance for the actual status of the community.
In particular, ``facets`` events are fuzzy events (more than one can occur at the same time) while ``greene`` and ``asur`` events are crisp events (only one can occur at the same time).

.. note::
    Following the ``facets`` terminology,  ``analyze_flow`` and ``analyze_flows`` returns a dictionary describing the flow in terms of its Unicity, Identity and Outflow.
    For a detailed description of such measures refer to [Failla24]_

In addition, if the temporal network comes with attributes associated to the nodes (either dynamically changing or not - i.e., political leanings), the library provides a set of tools to analyze the typicality of the events.

Setting and retreiving node attributes is straightforward:

.. code-block:: python

    from cdlib import LifeCycle, TemporalClustering, algorithms
    from networkx.generators.community import LFR_benchmark_graph

    def random_leaning():
        attrs = {}
        for i in range(250): # 250 nodes
            attrs[i] = {}
            for t in range(10): # 10 time steps
                attrs[i][t] = random.choice(["left", "right"])
        return attrs

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
    events.set_attribute(random_leaning(), "political_leaning")
    attrs = events.get_attribute("political_leaning")

    events.analyze_flow("1_1", "+",  attr="political_leaning") # to analyze the flow of political leaning in the community 1_1

Attributes are stored as a dictionary of dictionaries where the first key is the node id and the second key is the time step.

If such information is available, the ``analyze_flow`` method will integrate in its analysis an evaluation of flow-attribute entropy.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Visualizing Events and Flows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library provides a set of tools to visualize the events and flows detected in the community structure of a network.

.. note::

    The library uses the ``networkx`` library to represent the community structure of a network and the ``matplotlib`` / ``plotly`` library to visualize it.

Here an example of how to visualize community events, flows and polytree:

.. code-block:: python

    from cdlib import LifeCycle, TemporalClustering, algorithms
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

    from cdlib import LifeCycle, TemporalClustering, algorithms
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


.. [Failla24] Andrea Failla, Rémy Cazabet, Giulio Rossetti, Salvatore Citraro . "Redefining Event Types and Group Evolution in Temporal Data.", arXiv preprint arXiv:2403.06771. 2024

.. [Asur2009] Sitaram Asur, Parthasarathy Srinivasan, Ucar Duygu. "An event-based framework for characterizing the evolutionary behavior of interaction graphs." ACM Transactions on Knowledge Discovery from Data (TKDD) 3.4 (2009): 1-36.

.. [Greene2010] Derek Greene, Doyle Donal, Cunningham, Padraig. "Tracking the evolution of communities in dynamic social networks." 2010 international conference on advances in social networks analysis and mining. IEEE, 2010.