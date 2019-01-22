from nclib.classes.node_clustering import NodeClustering
import json


class FuzzyNodeClustering(NodeClustering):

    def __init__(self, communities, allocation_matrix, graph, method_name, method_parameters=None, overlap=False):
        """
        Communities representation.

        :param communities: list of communities
        :param method_name: algorithms discovery algorithm name
        :param overlap: boolean, whether the partition is overlapping or not
        """
        super().__init__(communities, graph, method_name, method_parameters, overlap)
        self.allocation_matrix = allocation_matrix

    def to_json(self):
        """
        Generate a JSON representation of the algorithms object

        :return: a JSON formatted string representing the object
        """

        partition = {"communities": self.communities, "algorithm": self.method_name,
                     "params": self.method_parameters, "overlap": self.overlap, "coverage": self.node_coverage,
                     "allocation_matrix": self.allocation_matrix}

        return json.dumps(partition)
