from cdlib.classes.node_clustering import NodeClustering
from cdlib import evaluation


class AttrNodeClustering(NodeClustering):

    """Attribute Node Communities representation.

      :param communities: list of communities
      :param graph: a networkx/igraph object
      :param method_name: community discovery algorithm name
      :param coms_labels: dictionary specifying for each community the frequency of the attribute values
      :param method_parameters: configuration for the community discovery algorithm used
      :param overlap: boolean, whether the partition is overlapping or not
      """

    def __init__(self, communities, graph, method_name, coms_labels=None, method_parameters=None, overlap=False):
        super().__init__(communities, graph, method_name, method_parameters, overlap)
        self.coms_labels = coms_labels

    def purity(self):

        """Purity is the product of the frequencies of the most frequent labels carried by the nodes within the communities
        :return: FitnessResult object
        """
        res = None
        if self.coms_labels is not None:
            res = evaluation.purity(self.coms_labels)
        return res