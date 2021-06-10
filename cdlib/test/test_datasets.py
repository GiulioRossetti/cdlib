import unittest
from cdlib import datasets, NodeClustering
import networkx as nx


class DatasetCase(unittest.TestCase):
    def test_net_list(self):
        nets = datasets.available_networks()
        self.assertIsInstance(nets, list)

    def test_gt_list(self):
        nets = datasets.available_ground_truths()
        self.assertIsInstance(nets, list)

    def test_net_remote(self):
        gi = datasets.fetch_network_data(net_type="igraph", net_name="karate_club")
        g = datasets.fetch_network_data(net_type="networkx", net_name="karate_club")
        self.assertEqual(g.number_of_edges(), gi.ecount())

    def test_gt_remote(self):
        gt = datasets.fetch_ground_truth_data("karate_club")
        self.assertIsInstance(gt, NodeClustering)

    def test_combined(self):
        g, gt = datasets.fetch_network_ground_truth("karate_club", net_type="networkx")
        self.assertIsInstance(gt, NodeClustering)
        self.assertIsInstance(g, nx.Graph)

    def test_remote_data(self):
        for net in datasets.available_networks():
            gi = datasets.fetch_network_data(net_type="igraph", net_name=net)
            self.assertIsNotNone(gi)

        for net in datasets.available_ground_truths():
            gi = datasets.fetch_network_ground_truth(net_name=net)
            self.assertIsNotNone(gi)


if __name__ == "__main__":
    unittest.main()
