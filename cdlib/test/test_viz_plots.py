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
        coms2 = algorithms.label_propagation(g)

        viz.plot_sim_matrix([coms, coms2], evaluation.adjusted_mutual_information)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

    def test_plot_com_stat(self):

        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.label_propagation(g)

        viz.plot_com_stat([coms, coms2], evaluation.size)

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

        viz.plot_com_stat(coms, evaluation.size)

        plt.savefig("cluster2.pdf")
        os.remove("cluster2.pdf")

    def test_plot_com_properties_relation(self):

        g = nx.karate_club_graph()
        coms = algorithms.louvain(g)
        coms2 = algorithms.label_propagation(g)

        viz.plot_com_properties_relation(
            [coms, coms2], evaluation.size, evaluation.internal_edge_density
        )

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

        viz.plot_com_properties_relation(
            coms, evaluation.size, evaluation.internal_edge_density
        )

        plt.savefig("cluster.pdf")
        os.remove("cluster.pdf")

    # def test_plot_scoring(self):

    #    g1 = nx.generators.community.LFR_benchmark_graph(1000, 3, 1.5, 0.5, min_community=20, average_degree=5)
    #    g2 = nx.generators.community.LFR_benchmark_graph(1000, 3, 1.5, 0.7, min_community=20, average_degree=5)

    #    names = ["g1", "g2"]
    #    graphs = [g1, g2]

    #    algos = [algorithms.crisp_partition.louvain,
    #             algorithms.crisp_partition.label_propagation]

    #    references = []
    #    m = ['Louvain', 'LP']
    #    for i, g in enumerate(graphs):
    #        coms = [g.nodes[v]['community'] for v in g]
    #        coms = [list(c) for c in coms]
    #        print(coms)

    #        references.append(NodeClustering(communities=coms, graph=g, method_name=m[i]))

    #    viz.plot_scoring(graphs, references, names, algos, nbRuns=2)

    #    plt.savefig("cluster.pdf")
    #    os.remove("cluster.pdf")


if __name__ == "__main__":
    unittest.main()
