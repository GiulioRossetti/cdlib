import unittest
import networkx as nx
from nclib import viz, community
import os
import matplotlib.pyplot as plt


class NetworkVizTests(unittest.TestCase):

    def test_nx_cluster(self):

        g = nx.karate_club_graph()
        coms = community.louvain(g)
        pos = nx.spring_layout(g)
        viz.plot_network_clusters(g, coms, pos)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

    def test_community_graph(self):

        g = nx.karate_club_graph()
        coms = community.louvain(g)
        viz.plot_community_graph(g, coms)

        plt.savefig("cg.pdf")
        os.remove("cg.pdf")


if __name__ == '__main__':
    unittest.main()
