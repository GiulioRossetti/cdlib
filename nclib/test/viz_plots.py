import unittest
import networkx as nx
from nclib import viz, community, evaluation
import os
import matplotlib.pyplot as plt


class NetworkVizTests(unittest.TestCase):

    def test_heatmap(self):

        g = nx.karate_club_graph()
        coms = community.louvain(g)
        coms2 = community.walktrap(g)

        viz.plot_sim_matrix([coms,coms2],evaluation.adjusted_mutual_information)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")


if __name__ == '__main__':
    unittest.main()
