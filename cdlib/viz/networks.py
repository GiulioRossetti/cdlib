import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from cdlib import NodeClustering
from cdlib.utils import convert_graph_formats
from community import community_louvain

__all__ = ["plot_network_clusters", "plot_community_graph"]

# [r, b, g, c, m, y, k, 0.8, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1, 0.5]
COLOR = (
    (1, 0, 0),
    (0, 0, 1),
    (0, 0.5, 0),
    (0, 0.75, 0.75),
    (0.75, 0, 0.75),
    (0.75, 0.75, 0),
    (0, 0, 0),
    (0.8, 0.8, 0.8),
    (0.2, 0.2, 0.2),
    (0.6, 0.6, 0.6),
    (0.4, 0.4, 0.4),
    (0.7, 0.7, 0.7),
    (0.3, 0.3, 0.3),
    (0.9, 0.9, 0.9),
    (0.1, 0.1, 0.1),
    (0.5, 0.5, 0.5),
)


def __filter(partition: list, top_k: int, min_size: int) -> list:
    if isinstance(min_size, int) and min_size > 0:
        partition = list(filter(lambda nodes: len(nodes) >= min_size, partition))
    if isinstance(top_k, int) and top_k > 0:
        partition = partition[:top_k]
    return partition


def plot_network_clusters(
    graph: object,
    partition: NodeClustering,
    position: dict = None,
    figsize: tuple = (8, 8),
    node_size: int = 200,
    plot_overlaps: bool = False,
    plot_labels: bool = False,
    cmap: object = None,
    top_k: int = None,
    min_size: int = None,
    edge_weights: dict = None,  # Nouveau paramètre pour les poids des arêtes
) -> object:
    """
    Plot a graph with node color coding for communities.

    :param graph: NetworkX/igraph graph
    :param partition: NodeClustering object
    :param position: A dictionary with nodes as keys and positions as values. Example: networkx.fruchterman_reingold_layout(G). By default, uses nx.spring_layout(g)
    :param figsize: the figure size; it is a pair of float, default (8, 8)
    :param node_size: int, default 200
    :param plot_overlaps: bool, default False. Flag to control if multiple algorithms memberships are plotted.
    :param plot_labels: bool, default False. Flag to control if node labels are plotted.
    :param cmap: str or Matplotlib colormap, Colormap(Matplotlib colormap) for mapping intensities of nodes. If set to None, original colormap is used.
    :param top_k: int, Show the top K influential communities. If set to zero or negative value indicates all.
    :param min_size: int, Exclude communities below the specified minimum size.
    :param edge_weights: dict, dictionary containing edge weights

    Example:

    >>> from cdlib import algorithms, viz
    >>> import networkx as nx
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> pos = nx.spring_layout(g)
    >>> viz.plot_network_clusters(g, coms, pos, edge_weights=edge_weights)
    """
    if not isinstance(cmap, (type(None), str, matplotlib.colors.Colormap)):
        raise TypeError(
            "The 'cmap' argument must be NoneType, str or matplotlib.colors.Colormap, "
            "not %s." % (type(cmap).__name__)
        )

    partition = __filter(partition.communities, top_k, min_size)
    graph = convert_graph_formats(graph, nx.Graph)
    if position is None:
        position = nx.spring_layout(graph)

    n_communities = len(partition)
    if n_communities == 0:
        warnings.warn("There are no communities that match the filter criteria.")
        return None

    if cmap is None:
        n_communities = min(n_communities, len(COLOR))
        cmap = matplotlib.colors.ListedColormap(COLOR[:n_communities])
    else:
        cmap = plt.cm.get_cmap(cmap, n_communities)
    _norm = matplotlib.colors.Normalize(vmin=0, vmax=n_communities - 1)
    fontcolors = list(
        map(
            lambda rgb: ".15" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) > 0.408 else "w",
            [cmap(_norm(i))[:3] for i in range(n_communities)],
        )
    )

    plt.figure(figsize=figsize)
    plt.axis("off")

    filtered_nodelist = list(np.concatenate(partition))
    filtered_edgelist = list(
        filter(
            lambda edge: len(np.intersect1d(edge, filtered_nodelist)) == 2,
            graph.edges(),
        )
    )

    fig = nx.draw_networkx_nodes(
        graph, position, node_size=node_size, node_color="w", nodelist=filtered_nodelist
    )
    fig.set_edgecolor("k")
    
    edge_widths = [edge_weights.get(edge, 1.0) for edge in filtered_edgelist]
    # Interpolate edge widths to be between 1 and 10
    min_width = min(edge_widths)
    max_width = max(edge_widths)
    edge_widths = np.interp(edge_widths, (min_width, max_width), (1, 10))
    nx.draw_networkx_edges(graph, position, alpha=0.5, edgelist=filtered_edgelist, width=edge_widths)
    
    edge_labels = {(edge[0], edge[1]): edge_weights.get(edge, "") for edge in filtered_edgelist}
    nx.draw_networkx_edge_labels(
                graph, 
                position, 
                edge_labels=edge_labels, 
                font_color="red", 
                label_pos=0.5,  # Adjust the position of the numerical values
                font_size=8,    # Set the font size of the numerical values
                font_weight='bold',
            )


    if plot_labels:
        nx.draw_networkx_labels(
            graph,
            position,
            font_color=".8",
            labels={node: str(node) for node in filtered_nodelist},
        )

    for i in range(n_communities):
        if len(partition[i]) > 0:
            if plot_overlaps:
                size = (n_communities - i) * node_size
            else:
                size = node_size
            fig = nx.draw_networkx_nodes(
                graph,
                position,
                node_size=size,
                nodelist=partition[i],
                node_color=[cmap(_norm(i))],
            )
            fig.set_edgecolor("k")

    if plot_labels:
        for i in range(n_communities):
            if len(partition[i]) > 0:
                nx.draw_networkx_labels(
                    graph,
                    position,
                    font_color=fontcolors[i],
                    labels={node: str(node) for node in partition[i]},
                )
    return fig

def calculate_cluster_edge_weights(graph, node_to_com):
    cluster_edge_weights = {}

    for edge in graph.edges():
        source, target = edge
        source_com = node_to_com.get(source, None)
        target_com = node_to_com.get(target, None)

        if source_com is not None and target_com is not None and source_com != target_com:
            # Nodes belong to different communities
            cluster_pair = (source_com, target_com)
            
            if cluster_pair not in cluster_edge_weights:
                cluster_edge_weights[cluster_pair] = 1
            else:
                cluster_edge_weights[cluster_pair] += 1

    return cluster_edge_weights
def plot_community_graph(
    graph: object,
    partition: NodeClustering,
    figsize: tuple = (8, 8),
    node_size: int = 200,
    plot_overlaps: bool = False,
    plot_labels: bool = False,
    cmap: object = None,
    top_k: int = None,
    min_size: int = None,
) -> object:
    """
    Plot a algorithms-graph with node color coding for communities.

    :param graph: NetworkX/igraph graph
    :param partition: NodeClustering object
    :param figsize: the figure size; it is a pair of float, default (8, 8)
    :param node_size: int, default 200
    :param plot_overlaps: bool, default False. Flag to control if multiple algorithms memberships are plotted.
    :param plot_labels: bool, default False. Flag to control if node labels are plotted.
    :param cmap: str or Matplotlib colormap, Colormap(Matplotlib colormap) for mapping intensities of nodes. If set to None, original colormap is used..
    :param top_k: int, Show the top K influential communities. If set to zero or negative value indicates all.
    :param min_size: int, Exclude communities below the specified minimum size.

    Example:

    >>> from cdlib import algorithms, viz
    >>> import networkx as nx
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> viz.plot_community_graph(g, coms)
    """

    cms = __filter(partition.communities, top_k, min_size)

    node_to_com = {}
    for cid, com in enumerate(cms):
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

    # algorithms graph construction
    c_graph = community_louvain.induced_graph(node_to_com, s)
    node_cms = [[node] for node in c_graph.nodes()]

    # Calculate edge weights for each cluster
    edge_weights = calculate_cluster_edge_weights(graph, node_to_com)
    #print(edge_weights)

    return plot_network_clusters(
        c_graph,
        NodeClustering(node_cms, None, ""),
        nx.spring_layout(c_graph),
        figsize=figsize,
        node_size=node_size,
        plot_overlaps=plot_overlaps,
        plot_labels=plot_labels,
        cmap=cmap,
        edge_weights=edge_weights  # Pass edge_weights to the plotting function
    )
