import unittest
import networkx as nx
from cdlib import viz, algorithms, evaluation, NodeClustering
import os
import matplotlib.pyplot as plt
import numpy as np


class PlotsVizTests(unittest.TestCase):

    def test_plot_sim_matrix(self):

        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.walktrap(g)

        viz.plot_sim_matrix([coms, coms2], evaluation.adjusted_mutual_information)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

    def test_plot_com_stat(self):

        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.walktrap(g)

        viz.plot_com_stat([coms, coms2], evaluation.size)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

        viz.plot_com_stat(coms, evaluation.size)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

    def test_plot_com_properties_relation(self):

        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.walktrap(g)

        viz.plot_com_properties_relation([coms, coms2], evaluation.size, evaluation.internal_edge_density)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

        viz.plot_com_properties_relation(coms, evaluation.size, evaluation.internal_edge_density)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

        viz.plot_com_properties_relation(coms, evaluation.size, evaluation.internal_edge_density,
                                         log_x=True, log_y=True)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

    def test_plot_scoring(self):

        graphs = []
        names = []
        for mu in np.arange(0.1, 0.2, 0.05):
            for j in range(2):
                g = nx.algorithms.community.LFR_benchmark_graph(250, 3, 1.5, mu, average_degree=5, min_community=20,
                                                                seed=10)
                name = "mu:%.2f" % mu
                names.append(name)
                graphs.append(g)

        references = []
        for g in graphs:
            references.append(NodeClustering(communities={frozenset(g.nodes[v]['community']) for v in g}, graph=g,
                                             method_name="reference"))

        algos = [algorithms.crisp_partition.louvain,
                 algorithms.crisp_partition.leiden]

        viz.plot_scoring(graphs, references, names, algos, nbRuns=2)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")


if __name__ == '__main__':
    unittest.main()
