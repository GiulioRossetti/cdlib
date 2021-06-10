import unittest
import networkx as nx
from cdlib import viz, algorithms
import os
import matplotlib.pyplot as plt


class NetworkVizTests(unittest.TestCase):
    def test_nx_cluster(self):

        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        pos = nx.spring_layout(g)
        viz.plot_network_clusters(g, coms, pos)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

        coms = algorithms.demon(g, 0.25)
        pos = nx.spring_layout(g)
        viz.plot_network_clusters(g, coms, pos, plot_labels=True, plot_overlaps=True)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

    def test_community_graph(self):

        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        viz.plot_community_graph(g, coms)

        plt.savefig("cg.pdf")
        os.remove("cg.pdf")

        coms = algorithms.demon(g, 0.25)
        viz.plot_community_graph(g, coms, plot_overlaps=True, plot_labels=True)

        plt.savefig("cg.pdf")
        os.remove("cg.pdf")
