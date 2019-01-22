from nclib.classes.clustering import Clustering
import networkx as nx
import igraph as ig
from collections import defaultdict


class EdgeClustering(Clustering):

    def __init__(self, communities, graph, method_name, method_parameters=None, overlap=False):
        """
        Edge Clustering representation.

        :param communities: list of communities
        :param method_name: algorithms discovery algorithm name
        :param overlap: boolean, whether the partition is overlapping or not
        """
        super().__init__(communities, graph, method_name, method_parameters, overlap)
        
        if graph is not None:
            edge_count = len({eid: None for community in communities for eid in community})
            if isinstance(graph, nx.Graph):
                self.node_coverage = edge_count / graph.number_of_edges()
            elif isinstance(graph, ig.Graph):
                self.node_coverage = edge_count / graph.ecount()
            else:
                raise ValueError("Unsupported Graph type.")

    def __check_graph(self):
        return self.graph is not None

    def to_edge_community_map(self):
        """
        Generate a <edge, list(communities)> representation of the current clustering

        :return: dict of the form <edge, list(communities)>
        """

        edge_to_communities = defaultdict(list)
        for cid, community in enumerate(self.communities):
            for edge in community:
                edge_to_communities[edge].append(cid)

        return edge_to_communities
