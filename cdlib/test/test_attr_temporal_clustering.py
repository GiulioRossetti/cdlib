import unittest

from cdlib import algorithms
from cdlib import TemporalClustering, NamedClustering
from cdlib.classes import AttrTemporalClustering
from cdlib import evaluation
from collections import defaultdict
import networkx as nx
import random
import json

def get_attributed_temporal_network_clustering(same_number_nodes=True):

    tc = AttrTemporalClustering()
    t_attrs = defaultdict(lambda: defaultdict(dict))
    l1 = ["one", "two", "three"]
    l2 = ['1', '2', '3']
    number_of_nodes = [50, 60, 70, 80, 90, 100]
    for t in range(10):
        if same_number_nodes == False:
            g = nx.erdos_renyi_graph(random.sample(number_of_nodes, 1)[0], 0.05)
        else:
            g = nx.erdos_renyi_graph(100, 0.05)
        coms = algorithms.louvain(g)
        # simulating named clustering
        nc = NamedClustering(
            {i: c for i, c in enumerate(coms.communities)}, g, coms.method_name
        )
        for node in g.nodes():
            g.nodes[node]['l1'] = random.choice(l1)
            g.nodes[node]['l2'] = random.choice(l2)
        node_labels_1 = nx.get_node_attributes(g, 'l1')
        node_labels_2 = nx.get_node_attributes(g, 'l2')
        for i, (k, v) in enumerate(node_labels_1.items()):
            t_attrs[t]['l1'][k] = v
            t_attrs[t]['l2'][k] = node_labels_2[i]
        tc.add_clustering(nc, t)

    for t in range(10):
        tc.add_labeled_clustering(tc.get_clustering_at(t), t_attrs, t, ['l1', 'l2'])

    return tc, t_attrs

class AttrTemporalClusteringTests(unittest.TestCase):
    def test_labeled_clustering(self):
        list_attrs = ['l1', 'l2']
        tc, time_node_labels = get_attributed_temporal_network_clustering()
        self.assertIsInstance(time_node_labels, defaultdict)

        tids = tc.get_observation_ids()
        for tid in tids:
            labeled_coms = tc.get_labeled_clustering_at(tid)
            self.assertIsInstance(labeled_coms, dict)

            for c_name in labeled_coms.keys():
                self.assertIsInstance(tc.get_labeled_community(c_name), dict)
                for attr, vals in tc.get_labeled_community(c_name).items():
                    self.assertIn(attr, list_attrs)
                    self.assertIsInstance(vals, list)

    def test_labeled_flows(self):
        list_attrs = ['l1', 'l2']
        tc, time_node_labels = get_attributed_temporal_network_clustering(same_number_nodes=False)
        labeled_past_flow = tc.labeled_inflow(time_node_labels, list_attrs)
        labeled_future_flow = tc.labeled_outflow(time_node_labels, list_attrs)

        tids = tc.get_observation_ids()
        size_clusterings = sum([len(tc.get_clustering_at(tid).named_communities.keys()) for tid in tids])
        size_first_clustering = len(tc.get_clustering_at(tids[0]).named_communities.keys())
        size_last_clustering = len(tc.get_clustering_at(tids[-1]).named_communities.keys())

        self.assertEquals(len(labeled_past_flow), size_clusterings-size_first_clustering)
        self.assertEquals(len(labeled_future_flow), size_clusterings-size_last_clustering)

    def test_flow_coms_contribution(self):
        list_attrs = ['l1', 'l2']
        tc, time_node_labels = get_attributed_temporal_network_clustering(same_number_nodes=False)
        past_flow, _ = tc.inflow()

        for c_name, flow in past_flow.items():
            coms_contrib = list(set(flow))
            for com in coms_contrib:
                labeled_com = tc.get_labeled_community(com)
                self.assertIsInstance(labeled_com, dict)
                for attr, vals in labeled_com.items():
                    self.assertIn(attr, list_attrs)
                    self.assertGreaterEqual(len(vals), 1)



if __name__ == '__main__':
    unittest.main()
