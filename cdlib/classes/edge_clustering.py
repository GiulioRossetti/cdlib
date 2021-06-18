from cdlib.classes.clustering import Clustering
import networkx as nx

try:
    import igraph as ig
except ModuleNotFoundError:
    ig = None

from collections import defaultdict


class EdgeClustering(Clustering):
    """Edge Clustering representation.

    :param communities: list of communities
    :param graph: a networkx/igraph object
    :param method_name: community discovery algorithm name
    :param method_parameters: configuration for the community discovery algorithm used
    :param overlap: boolean, whether the partition is overlapping or not
    """

    def __init__(
        self,
        communities: list,
        graph: object,
        method_name: str = "",
        method_parameters: dict = None,
        overlap: bool = False,
    ):
        super().__init__(communities, graph, method_name, method_parameters, overlap)

        if graph is not None:
            edge_count = len(
                {eid: None for community in communities for eid in community}
            )
            if isinstance(graph, nx.Graph):
                self.node_coverage = edge_count / graph.number_of_edges()
            elif ig is not None and isinstance(graph, ig.Graph):
                self.node_coverage = edge_count / graph.ecount()
            else:
                raise ValueError("Unsupported Graph type.")

    def to_edge_community_map(self) -> dict:
        """
        Generate a <edge, list(communities)> representation of the current clustering

        :return: dict of the form <edge, list(communities)>
        """

        edge_to_communities = defaultdict(list)
        for cid, community in enumerate(self.communities):
            for edge in community:
                edge_to_communities[edge].append(cid)

        return edge_to_communities
