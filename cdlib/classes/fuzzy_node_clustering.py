from cdlib.classes.node_clustering import NodeClustering
import json


class FuzzyNodeClustering(NodeClustering):
    """Fuzzy Node Communities representation.

    :param communities: list of communities
    :param node_allocation: dictionary specifying for each node the allocation of probability toward the communities it is placed in
    :param graph: a networkx/igraph object
    :param method_name: community discovery algorithm name
    :param method_parameters: configuration for the community discovery algorithm used
    :param overlap: boolean, whether the partition is overlapping or not
    """

    def __init__(
        self,
        communities: list,
        node_allocation: dict,
        graph: object,
        method_name: str = "",
        method_parameters: dict = None,
        overlap: bool = False,
    ):
        super().__init__(communities, graph, method_name, method_parameters, overlap)
        self.allocation_matrix = node_allocation

    def to_json(self) -> str:
        """
        Generate a JSON representation of the algorithms object

        :return: a JSON formatted string representing the object
        """

        partition = {
            "communities": self.communities,
            "algorithm": self.method_name,
            "params": self.method_parameters,
            "overlap": self.overlap,
            "coverage": self.node_coverage,
            "allocation_matrix": self.allocation_matrix,
        }

        return json.dumps(partition)
