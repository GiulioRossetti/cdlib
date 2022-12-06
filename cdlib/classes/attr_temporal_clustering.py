from cdlib.classes.temporal_clustering import TemporalClustering
from collections import defaultdict


class AttrTemporalClustering(TemporalClustering):
    def __init__(self):
        """
        Attributed Temporal Communities representation
        :param labeled_communities: a defaultdict object.
                                    Keys represent communities: their ids are assigned following the pattern {tid}_{cid},
                                    where tid is the time of observation and cid is the community id.
                                    Values are dicts in turns, where a key represents the attribute name, and the values
                                    are lists of nodes labeled with the attribute values they are assigned.
        """

        super().__init__()
        self.labeled_communities = defaultdict(lambda: defaultdict())

    def get_labeled_community(self, cid: str):
        """
        Return the labeled nodes within a given temporal community.

        :param cid: community id of the form {tid}_{cid}, where tid is the time of observation and
               cid is the position of the community within the Clustering object.
        :return: a dict where keys are attribute names and values are the labeled nodes within cid
        """

        labeled_coms = self.labeled_communities[cid]
        return labeled_coms[cid]

    def get_labeled_communities(self, named_clustering: object, time_node_labels: dict, time: list, name_attrs: list):
        """
        Get the labeled communities.

        :param named_clustering: a defaultdict object of NamedClustering object
        :param time_node_labels: the dict of dynamic node labels for each attribute.
                                 The dict is structured as follows:
                                 Keys are temporal ids and values represent a dict of dicts; in this other dict, keys represent
                                 the attribute names: each attribute name allows to access to the dict of the node attribute labels
                                 for that temporal id; in this last dict, keys are node ids and values are labels.
        :param time: list of temporal ids
        :param name_attrs: list of attribute names
        """

        for t in time:
            named_attr_coms = defaultdict(list)
            coms = named_clustering[t].communities
            for i, community in enumerate(coms):
                for name in name_attrs:
                    attr_com = [time_node_labels[t][name][node] for node in community]
                    named_attr_coms[name].append(attr_com)
                    self.labeled_communities[f"{t}_{i}"][name] = attr_com

    def labeled_inflow(self, time_node_labels: dict, name_attrs: list):
        """
        Reconstruct labeled node flows across adjacent observations from the past to the present.
        "New" nodes, i.e., nodes belonging to the present community that do not appear in any past
        adjacent observation, by definition are not considered in the in-flow analysis.

        :param time_node_labels: the dict of dynamic node labels for each attribute.
                                 The dict is structured as follows:
                                 Keys are temporal ids and values represent a dict of dicts; in this other dict, keys represent
                                 the attribute names: each attribute name allows to access to the dict of the node attribute labels
                                 for that temporal id; in this last dict, keys are node ids and values are labels.
        :param name_attrs: list of attribute names
        :return: a defaultdict object.
                 Keys represent communities, their ids are assigned following the pattern {tid}_{cid},
                 where tid is the time of observation and cid is the community id.
                 Values are dicts of lists, where keys are attribute names and values are the list of attribute-labeled nodes
                 that flowed to the keys from the adjacent past observation.
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
        Reconstruct labeled node flows across adjacent observations from the present to the future.
        "Dead" nodes, i.e., nodes belonging to the present communities that do not appear in any future
        adjacent observation, by definition are not considered in the out-flow analysis.

        :param time_node_labels: the dict of dynamic node labels for each attribute.
                                 The dict is structured as follows:
                                 Keys are temporal ids and values represent a dict of dicts; in this other dict, keys represent
                                 the attribute names: each attribute name allows to access to the dict of the node attribute labels
                                 for that temporal id; in this last dict, keys are node ids and values are labels.
        :param name_attrs: list of attribute names
        :return: a defaultdict object.
        :return: Keys represent communities, their ids are assigned following the pattern {tid}_{cid},
                where tid is the time of observation and cid is the community id.
                Values are dicts of lists, where keys are attribute names and values are the lists of attribute-labeled nodes
                that flowed from the keys to the adjacent future observation.
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