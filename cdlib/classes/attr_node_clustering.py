from cdlib.classes.node_clustering import NodeClustering
from cdlib import evaluation
from collections import defaultdict


class AttrNodeClustering(NodeClustering):

    """Attribute Node Communities representation.

    :param communities: list of communities
    :param graph: a networkx/igraph object
    :param method_name: community discovery algorithm name
    :param coms_labels: list of node-labeled communities
    :param count_coms_labels: dictionary specifying for each community the frequency of the attribute values
    :param method_parameters: configuration for the community discovery algorithm used
    :param overlap: boolean, whether the partition is overlapping or not
    """

    def __init__(
        self,
        communities: list,
        graph: object,
        method_name: str = "",
        coms_labels: dict = defaultdict(list),
        count_coms_labels: dict = defaultdict(lambda: defaultdict(dict)),
        method_parameters: dict = None,
        overlap: bool = False,
    ):
        super().__init__(communities, graph, method_name, method_parameters, overlap)
        self.coms_labels = coms_labels
        self.count_coms_labels = count_coms_labels

    def add_coms_labels(self, node_labels, name_attr: str) -> list:
        """Return a list of labeled communities
        :param node_labels: dict 'node_id': 'node_label'
        :return: a list of labeled communities
        """
        for community in self.communities:
            attr_com = [node_labels[node] for node in community]
            self.coms_labels[name_attr].append(attr_com)

    def add_count_coms_labels(self) -> dict:
        """

        :return:
        """
        if self.coms_labels is not None:
            for name_attr, labeled_part in self.coms_labels.items():
                for i, labeled_com in enumerate(labeled_part):
                    label_com_count = {x: labeled_com.count(x) for x in set(labeled_com)}
                    self.count_coms_labels[i][name_attr].update(label_com_count)

    def normalized_entropy(self, logb: int, tot_n_classes: int) -> evaluation.FitnessResult:
        """
        Shannon Entropy
        :param logb:
        :param tot_n_classes:
        :return:
        """
        res = None
        if self.coms_labels is not None:
            res = evaluation.entropy(self.coms_labels, logb, tot_n_classes)
        return res

    def purity(self) -> evaluation.FitnessResult:
        """
        Purity is the product of the frequencies of the most frequent labels carried by the nodes within the communities
        :return: FitnessResult object
        """
        res = None
        if self.count_coms_labels is not None:
            res = evaluation.purity(self.count_coms_labels)
        return res
