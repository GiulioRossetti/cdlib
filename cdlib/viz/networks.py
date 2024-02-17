import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from cdlib import NodeClustering
from cdlib.utils import convert_graph_formats
from community import community_louvain
from typing import Union

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
    node_size: Union[int, dict] = 200, # 200 default value
    plot_overlaps: bool = False,
    plot_labels: bool = False,
    cmap: object = None,
    top_k: int = None,
    min_size: int = None,
    show_edge_widths: bool = False,
    show_edge_weights: bool = False,
    show_node_sizes: bool = False,
) -> object:
    """
    Plot a graph with node color coding for communities.

    :param graph: NetworkX/igraph graph
    :param partition: NodeClustering object
    :param position: A dictionary with nodes as keys and positions as values. Example: networkx.fruchterman_reingold_layout(G). By default, uses nx.spring_layout(g)
    :param figsize: the figure size; it is a pair of float, default (8, 8)
    :param node_size: The size of nodes. It can be an integer or a dictionary mapping nodes to sizes. Default is 200.
    :param plot_overlaps: bool, default False. Flag to control if multiple algorithms memberships are plotted.
    :param plot_labels: bool, default False. Flag to control if node labels are plotted.
    :param cmap: str or Matplotlib colormap, Colormap(Matplotlib colormap) for mapping intensities of nodes. If set to None, original colormap is used.
    :param top_k: int, Show the top K influential communities. If set to zero or negative value indicates all.
    :param min_size: int, Exclude communities below the specified minimum size.
    :param show_edge_widths: Flag to control if edge widths are shown. Default is False.
    :param show_edge_weights: Flag to control if edge weights are shown. Default is False.
    :param show_node_sizes: Flag to control if node sizes are shown. Default is False.

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
    

    # TODO : Réecrire le code de tel sorte à ce qu'on crée le cercle en premier en fonction des poids des liens intra cluster pour eviter la superposition des clusters
    if position is None:
        supergraph = nx.cycle_graph(len(partition))
        superpos = nx.spring_layout(supergraph, scale=1, seed=429)
        
        centers = list(superpos.values())
        position = {}
        for center, comm in zip(centers, partition):
            position.update(nx.spring_layout(nx.subgraph(graph, comm), center=center, seed=1430, k=0.5, weight=5))
        
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
    if isinstance(node_size, int):
        fig = nx.draw_networkx_nodes(
            graph, position, node_size=node_size, node_color="w", nodelist=filtered_nodelist
        )
        fig.set_edgecolor("k")
    
    filtered_edge_widths = [1] * len(filtered_edgelist) 

    if show_edge_widths:
        edge_widths = nx.get_edge_attributes(graph, "weight")
        filtered_edge_widths = [weight for (edge, weight) in edge_widths.items() if edge[0] != edge[1]]
        
        min_weight = min(filtered_edge_widths)
        max_weight = max(filtered_edge_widths)

        filtered_edge_widths = np.interp(filtered_edge_widths, (min_weight, max_weight), (1, 5))
    
    nx.draw_networkx_edges(graph, position, alpha=0.5, edgelist=filtered_edgelist, width=filtered_edge_widths, connectionstyle="arc3,rad=0.1")

    if show_edge_weights:
        edge_weights = nx.get_edge_attributes(graph, "weight")
        filtered_edge_weights = [{edge: weight} for edge, weight in edge_weights.items() if edge[0] != edge[1]] 
        
        for edge_weight in filtered_edge_weights:
            nx.draw_networkx_edge_labels(
                    graph,
                    position,
                    edge_labels=edge_weight,
                    font_color="red",
                    label_pos=0.5,
                    font_size=8,
                    font_weight='bold',
                )

    if plot_labels:
        nx.draw_networkx_labels(
            graph,
            position,
            font_color=".8",
            labels={node: str(node) for node in filtered_nodelist},
        )

    if isinstance(node_size, dict) and show_node_sizes:
        # Extract values from the node_size dictionary
        node_values = list(node_size.values())

        # Interpolate node_size values to be between 200 and 500
        min_node_size = min(node_values) if node_values else 1
        max_node_size = max(node_values) if node_values else 1
        node_size = {key: np.interp(value, (min_node_size, max_node_size), (200, 1000)) for key, value in node_size.items()}
    else:
        node_size = 200
        
    for i in range(n_communities):
        if len(partition[i]) > 0:
            if plot_overlaps:
                if isinstance(node_size, dict):
                    size = (n_communities - i) * node_size[i]  # Use interpolated size from dictionary
                else:
                    size = (n_communities - i) * node_size  # Use fixed size
            else:
                if isinstance(node_size, dict):
                    size = node_size[i]  # Use interpolated size from dictionary
                else:
                    size = node_size  # Use fixed size
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
    """
    Calculate edge weights between different clusters.

    This function calculates the edge weights between nodes belonging to different clusters.
    It iterates through the edges of the graph, identifies the communities of the source and target nodes,
    and increments the edge weight for the corresponding cluster pair.

    :param graph: NetworkX/igraph graph
    :param node_to_com: Dictionary mapping nodes to their community IDs
    """
    cluster_edge_weights = {}

    for edge in graph.edges():
        source, target = edge
        source_com = node_to_com.get(source, None)
        target_com = node_to_com.get(target, None)

        if source_com is not None and target_com is not None and source_com != target_com:
            # Nodes belong to different communities
            cluster_pair = (source_com, target_com)
            
            # Check if edge data is not empty
            edge_data = graph.get_edge_data(source_com, target_com)
            
            # Check if edge data is None, empty or not
            if edge_data is None:
                edge_weight = 0
            elif edge_data == {}:
                edge_weight = 1
            else: # edge_data contains an element 'weight' : int
                edge_weight = edge_data['weight']

            if cluster_pair not in cluster_edge_weights:
                cluster_edge_weights[cluster_pair] = edge_weight
            else:
                cluster_edge_weights[cluster_pair] += edge_weight
                
    cluster_edge_weights_array = [(source, target, weight) for (source, target), weight in cluster_edge_weights.items()]
    graph.add_weighted_edges_from(cluster_edge_weights_array)

def calculate_cluster_sizes(partition: NodeClustering) -> Union[int, dict]:
    """
    Calculate the total weight of all nodes in each cluster.

    :param partition: The partition of the graph into clusters.
    :type partition: NodeClustering
    :return: If all clusters have the same size, return the size as an integer.
             Otherwise, return a dictionary mapping cluster ID to the number of nodes in the cluster.
    :rtype: Union[int, dict]
    """
    cluster_sizes = {}
    unique_values = set()
    
    for cid, com in enumerate(partition.communities):
        total_weight = 0
        
        #print("cluster: ", cid)
        for node in com:
            if 'weight' in partition.graph.nodes[node]:  # If node data contains a 'weight' attribute
                total_weight += partition.graph.nodes[node]['weight']
            else: # If node data is empty
                total_weight += 1  # Default weight is 1
                
        cluster_sizes[cid] = total_weight      
        
    if len(unique_values) == 1:
        return int(unique_values.pop())  # All elements have the same value, return that value as an integer
    else:
        return cluster_sizes  # Elements have different values, return the dictionary

def plot_community_graph(
    graph: object,
    partition: NodeClustering,
    figsize: tuple = (8, 8),
    node_size: Union[int, dict] = 200,
    plot_overlaps: bool = False,
    plot_labels: bool = False,
    cmap: object = None,
    top_k: int = None,
    min_size: int = None,
    show_edge_weights: bool = True,
    show_edge_widths: bool = True,
    show_node_sizes: bool = True,
) -> object:
    """
    Plot a algorithms-graph with node color coding for communities.

    :param graph: NetworkX/igraph graph
    :param partition: NodeClustering object
    :param figsize: the figure size; it is a pair of float, default (8, 8)
    :param node_size: The size of nodes. It can be an integer or a dictionary mapping nodes to sizes. Default is 200.
    :param plot_overlaps: bool, default False. Flag to control if multiple algorithms memberships are plotted.
    :param plot_labels: bool, default False. Flag to control if node labels are plotted.
    :param cmap: str or Matplotlib colormap, Colormap(Matplotlib colormap) for mapping intensities of nodes. If set to None, original colormap is used..
    :param top_k: int, Show the top K influential communities. If set to zero or negative value indicates all.
    :param min_size: int, Exclude communities below the specified minimum size.
    :param show_edge_widths: Flag to control if edge widths are shown. Default is True.
    :param show_edge_weights: Flag to control if edge weights are shown. Default is True.
    :param show_node_sizes: Flag to control if node sizes are shown. Default is True.

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

    # Calculate edge weights and edge widths for each cluster
    if(show_edge_weights or show_edge_widths):
        calculate_cluster_edge_weights(graph, node_to_com)
    
    if show_node_sizes:
        # Calculate cluster sizes for adjusting node sizes
        node_size = calculate_cluster_sizes(partition)

    return plot_network_clusters(
        c_graph,
        NodeClustering(node_cms, None, ""),
        nx.spring_layout(c_graph),
        figsize=figsize,
        node_size=node_size,
        plot_overlaps=plot_overlaps,
        plot_labels=plot_labels,
        cmap=cmap,
        show_edge_weights=show_edge_weights,
        show_edge_widths=show_edge_widths,
        show_node_sizes=show_node_sizes,
    )
