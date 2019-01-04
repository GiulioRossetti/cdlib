import matplotlib.pyplot as plt
import networkx as nx
from nclib.utils import convert_graph_formats
from community import induced_graph

COLOR = ['r', 'b', 'g', 'c', 'm', 'y', 'k',
         '0.8', '0.2', '0.6', '0.4', '0.7', '0.3', '0.9', '0.1', '0.5']


def plot_network_clusters(graph, partition, position, figsize=(8, 8), node_size=200, plot_overlaps=False, plot_labels=False):
    """
    Plot a graph with node color coding for communities.

    :param graph: NetworkX/igraph graph
    :param partition: list of list of nodes. List of communities.
    :param position: dictionary
       A dictionary with nodes as keys and positions as values.
       Example: networkx.fruchterman_reingold_layout(G)
    :param figsize: pair of float, default (8, 8)
        Figure size.
    :param node_size: int, default 200
        Node size.
    :param plot_overlaps: bool, default False
        Flag to control if multiple community memberships are plotted.
    :param plot_labels: bool, default False
        Flag to control if node labels are plotted.
    """

    graph = convert_graph_formats(graph, nx.Graph)

    n_communities = min(len(partition), len(COLOR))
    plt.figure(figsize=figsize)
    plt.axis('off')

    fig = nx.draw_networkx_nodes(graph, position, node_size=node_size, node_color='w')
    fig.set_edgecolor('k')
    nx.draw_networkx_edges(graph, position, alpha=.5)
    for i in range(n_communities):
        if len(partition[i]) > 0:
            if plot_overlaps:
                size = (n_communities - i) * node_size
            else:
                size = node_size
            fig = nx.draw_networkx_nodes(graph, position, node_size=size,
                                         nodelist=partition[i], node_color=COLOR[i])
            fig.set_edgecolor('k')
    if plot_labels:
        nx.draw_networkx_labels(graph, position, labels={node: str(node) for node in graph.nodes()})

    return fig


def plot_community_graph(graph, partition, figsize=(8, 8), node_size=200, plot_overlaps=False, plot_labels=False):
    """
        Plot a community-graph with node color coding for communities.

        :param graph: NetworkX/igraph graph
        :param partition: list of list of nodes. List of communities.
        :param figsize: pair of float, default (8, 8)
            Figure size.
        :param node_size: int, default 200
            Node size.
        :param plot_overlaps: bool, default False
            Flag to control if multiple community memberships are plotted.
        :param plot_labels: bool, default False
            Flag to control if node labels are plotted.
        """

    node_to_com = {}
    for cid, com in enumerate(partition):
        for node in com:
            if node not in node_to_com:
                node_to_com[node] = cid
            else:
                # duplicating overlapped node
                alias = "%s_%s" % (node, cid)
                node_to_com[alias] = cid
                edges = [(alias, y) for y in graph.neighbors(node)]
                graph.add_edges_from(edges)

    # handling partial coverage
    s = nx.subgraph(graph, node_to_com.keys())

    # community graph construction
    c_graph = induced_graph(node_to_com, s)
    node_cms = [[node] for node in c_graph.nodes()]
    return plot_network_clusters(c_graph, node_cms, nx.spring_layout(c_graph), figsize=figsize,
                                 node_size=node_size, plot_overlaps=plot_overlaps, plot_labels=plot_labels)



