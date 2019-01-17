import json
import networkx as nx
import igraph as ig
from collections import defaultdict


class EdgeClustering(object):

    def __init__(self, communities, graph, method_name, method_parameters=None, overlap=False):
        """
        Edge Clustering representation.

        :param communities: list of communities
        :param method_name: community discovery algorithm name
        :param overlap: boolean, whether the partition is overlapping or not
        """
        self.communities = communities
        self.graph = graph
        self.method_name = method_name
        self.overlap = overlap
        self.method_parameters = method_parameters

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

    def to_json(self):
        """
        Generate a JSON representation of the community object

        :return: a JSON formatted string representing the object
        """

        partition = {"communities": self.communities, "algorithm": self.method_name,
                     "params": self.method_parameters, "overlap": self.overlap}

        return json.dumps(partition)

    def to_edge_community_map(self):
        """
        Generate a <node, list(communities)> representation of the current clustering

        :return: dict of the form <node, list(communities)>
        """

        node_to_communities = defaultdict(list)
        for cid, community in enumerate(self.communities):
            for node in community:
                node_to_communities[node].append(cid)

        return node_to_communities
