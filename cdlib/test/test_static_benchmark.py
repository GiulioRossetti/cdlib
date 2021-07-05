import unittest

import cdlib
from cdlib import algorithms
from cdlib import benchmark
import networkx as nx


class STBenchTest(unittest.TestCase):
    def test_LFR(self):
        n = 250
        tau1 = 3
        tau2 = 1.5
        mu = 0.1
        G, coms = benchmark.LFR(n, tau1, tau2, mu, average_degree=5, min_community=20)
        self.assertIsInstance(G, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)
        nodes_coms = sum([len(c) for c in coms.communities])
        self.assertEqual(nodes_coms, G.number_of_nodes())

    def test_xmark(self):
        N = 2000
        gamma = 3
        beta = 2
        m_cat = ["auto", "auto"]
        theta = 0.3
        mu = 0.5
        avg_k = 10
        min_com = 20

        g, coms = benchmark.XMark(
            n=N,
            gamma=gamma,
            beta=beta,
            mu=mu,
            m_cat=m_cat,
            theta=theta,
            avg_k=avg_k,
            min_com=min_com,
            type_attr="categorical",
        )

        set1 = nx.get_node_attributes(g, "label_0")
        set2 = nx.get_node_attributes(g, "label_1")
        self.assertIsInstance(g, nx.Graph)
        self.assertIsInstance(coms, cdlib.NodeClustering)
        self.assertEqual(len(set(set1.values())), len(coms.communities))
        self.assertEqual(len(set(set2.values())), len(coms.communities))


if __name__ == "__main__":
    unittest.main()
