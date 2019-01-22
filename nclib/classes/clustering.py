import json


class Clustering(object):

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
        self.node_coverage = 1

    def to_json(self):
        """
        Generate a JSON representation of the algorithms object

        :return: a JSON formatted string representing the object
        """

        partition = {"communities": self.communities, "algorithm": self.method_name,
                     "params": self.method_parameters, "overlap": self.overlap, "coverage": self.node_coverage}

        return json.dumps(partition)
