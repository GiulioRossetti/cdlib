from cdlib.classes.clustering import Clustering


class NamedClustering(Clustering):
    """Node Communities representation.

    :param communities: dict of named communities <community_id, node_list>
    :param graph: a networkx/igraph object
    :param method_name: community discovery algorithm name
    :param method_parameters: configuration for the community discovery algorithm used
    :param overlap: boolean, whether the partition is overlapping or not
    """

    def __init__(
        self,
        communities: list,
        graph: object,
        method_name: str,
        method_parameters: dict = None,
        overlap: bool = False,
    ):
        self.named_communities = communities
        flat_coms = list(communities.values())
        super().__init__(flat_coms, graph, method_name, method_parameters, overlap)
