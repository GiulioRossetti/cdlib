from cdlib.classes.node_clustering import NodeClustering


class BiNodeClustering(NodeClustering):

    """Bipartite Node Communities representation.

      :param left_communities: list of left communities
      :param right_communities: list of right communities
      :param graph: a networkx/igraph object
      :param method_name: community discovery algorithm name
      :param method_parameters: configuration for the community discovery algorithm used
      :param overlap: boolean, whether the partition is overlapping or not
      """

    def __init__(self, left_communities, right_communities, graph, method_name, method_parameters=None, overlap=False):
        self.left_communities = left_communities
        self.right_communities = right_communities
        super().__init__(self.left_communities + self.right_communities, graph, method_name, method_parameters, overlap)
