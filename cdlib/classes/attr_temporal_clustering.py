from cdlib.classes.temporal_clustering import TemporalClustering
from cdlib.classes import AttrNodeClustering
from collections import defaultdict


class AttrTemporalClustering(TemporalClustering):
    """
        Attributed Temporal Communities representation
    """
    def __init__(self):
        super().__init__()
        #self.named_labeled_communities = {}

    def get_labeled_clustering_at(self, time: object):
        """
        :param time: the time of observation
        :return: a Clustering object
        """
        return self.get_clustering_at(time).named_labeled_communities

    def get_labeled_community(self, cid: str):
        """

        :param cid:
        :return:
        """

        t, _ = cid.split("_")
        labeled_coms = self.get_labeled_clustering_at(int(t))
        return labeled_coms[cid]

    def add_labeled_clustering(self, temp_named_clustering: object, time_node_labels: dict, time: int, name_attrs: list):
        """

        :param temp_named_clustering:
        :param time_node_labels:
        :param time:
        :return:
        """
        named_attr_coms = defaultdict(list)
        attr_coms = AttrNodeClustering(temp_named_clustering.communities, temp_named_clustering.graph)
        for name in name_attrs:
            attr_coms.add_coms_labels(time_node_labels[time][name], name)
            named_attr_coms[name] = attr_coms.coms_labels[name]

        attr_coms.add_count_coms_labels()
        #attr_coms.count_coms_labels

        named_labeled_clustering = defaultdict(lambda: defaultdict())
        for name, labeled_clust in named_attr_coms.items():
            for i, labeled_com in enumerate(labeled_clust):
                named_labeled_clustering[f"{time}_{i}"][name] = labeled_com

        temp_named_clustering.named_labeled_communities = named_labeled_clustering

    def labeled_inflow(self, time_node_labels: dict, name_attrs: list):
        """

        :return:
        """

        labeled_flow_from_past = defaultdict(lambda: defaultdict(list))
        tids = self.get_observation_ids()
        rangeto = [v for v in reversed(range(len(tids))) if v >= min(tids)]

        for tid in rangeto:
            current_clustering = self.clusterings[tid]
            for c_name, nodes in current_clustering.named_communities.items():
                count_flow = 0
                tid_prev = int(c_name.split('_')[0]) - 1
                if tid_prev >= min(tids):
                    prev_clustering = self.clusterings[tid_prev]
                    n_prev_coms = len(prev_clustering.named_communities)

                    for nc in range(n_prev_coms):  # for each com in the previous partition
                        com_prev = str(tid_prev) + '_' + str(nc)
                        for n in nodes:  # for each node in the current com
                            if n in self.get_community(com_prev):
                                for name_attr in name_attrs:
                                    label_com_prev = time_node_labels[tid_prev][name_attr][n]
                                    labeled_flow_from_past[c_name][name_attr].append(label_com_prev)
                                count_flow += 1
                    if count_flow == 0:
                        for name_attr in name_attrs:
                            labeled_flow_from_past[c_name][name_attr] = []

        return labeled_flow_from_past

    def labeled_outflow(self, time_node_labels: dict, name_attrs: list):
        """

        :return:
        """
        labeled_flow_to_future = defaultdict(lambda: defaultdict(list))
        tids = self.get_observation_ids()

        for tid in tids:
            current_clustering = self.clusterings[tid]
            for c_name, nodes in current_clustering.named_communities.items():
                count_flow = 0
                tid_succ = int(c_name.split('_')[0]) + 1
                if tid_succ < len(tids):
                    succ_clustering = self.clusterings[tid_succ]
                    n_succ_coms = len(succ_clustering.named_communities)

                    for nc in range(n_succ_coms):  # for each com in the next partition
                        com_succ = str(tid_succ) + '_' + str(nc)
                        for n in nodes:  # for each node in the current com
                            if n in self.get_community(com_succ):
                                for name_attr in name_attrs:
                                    label_com_succ = time_node_labels[tid_succ][name_attr][n]
                                    labeled_flow_to_future[c_name][name_attr].append(label_com_succ)
                                count_flow += 1
                    if count_flow == 0:
                        labeled_flow_to_future[c_name][name_attr] = []

        return labeled_flow_to_future