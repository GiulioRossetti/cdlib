import json
from nclib import evaluation
import networkx as nx
import igraph as ig
from collections import defaultdict


class NodeClustering(object):

    def __init__(self, communities, graph, method_name, method_parameters=None, overlap=False):
        """
        Communities representation.

        :param communities: list of communities
        :param method_name: algorithms discovery algorithm name
        :param overlap: boolean, whether the partition is overlapping or not
        """
        self.communities = communities
        self.graph = graph
        self.method_name = method_name
        self.overlap = overlap
        self.method_parameters = method_parameters

        if graph is not None:
            node_count = len({nid: None for community in communities for nid in community})
            if isinstance(graph, nx.Graph):
                self.node_coverage = node_count / graph.number_of_nodes()
            elif isinstance(graph, ig.Graph):
                self.node_coverage = node_count / graph.vcount()
            else:
                raise ValueError("Unsupported Graph type.")

    def __check_graph(self):
        return self.graph is not None

    def to_json(self):
        """
        Generate a JSON representation of the algorithms object

        :return: a JSON formatted string representing the object
        """

        partition = {"communities": self.communities, "algorithm": self.method_name,
                     "params": self.method_parameters, "overlap": self.overlap}

        return json.dumps(partition)

    def to_node_community_map(self):
        """
        Generate a <node, list(communities)> representation of the current clustering

        :return: dict of the form <node, list(communities)>
        """

        node_to_communities = defaultdict(list)
        for cid, community in enumerate(self.communities):
            for node in community:
                node_to_communities[node].append(cid)

        return node_to_communities

    def link_modularity(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.link_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def normalized_cut(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.normalized_cut(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def internal_edge_density(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.internal_edge_density(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def average_internal_degree(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.average_internal_degree(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def fraction_over_median_degree(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.fraction_over_median_degree(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def expansion(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.expansion(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def cut_ratio(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.cut_ratio(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def edges_inside(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.edges_inside(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def conductance(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.conductance(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def max_odf(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.max_odf(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def avg_odf(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.avg_odf(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def flake_odf(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.flake_odf(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def triangle_participation_ratio(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.triangle_participation_ratio(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def newman_girvan_modularity(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.newman_girvan_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def erdos_renyi_modularity(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.erdos_renyi_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def modularity_density(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.modularity_density(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def z_modularity(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.z_modularity(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def surprise(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.surprise(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def significance(self):
        """

        :return:
        """
        if self.__check_graph():
            return evaluation.significance(self.graph, self)
        else:
            raise ValueError("Graph instance not specified")

    def normalized_mutual_information(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.normalized_mutual_information(self, clustering)

    def overlapping_normalized_mutual_information(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.overlapping_normalized_mutual_information(self, clustering)

    def omega(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.omega(self, clustering)

    def f1(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.f1(self, clustering)

    def nf1(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.nf1(self, clustering)

    def adjusted_rand_index(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.adjusted_rand_index(self, clustering)

    def adjusted_mutual_information(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.adjusted_mutual_information(self, clustering)

    def variation_of_information(self, clustering):
        """

        :param clustering:
        :return:
        """
        return evaluation.variation_of_information(self, clustering)
