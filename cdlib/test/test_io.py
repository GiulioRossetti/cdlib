import unittest
from cdlib import algorithms
from cdlib import readwrite
import networkx as nx
import os


class IOTests(unittest.TestCase):

    def test_read_write(self):
        g = nx.karate_club_graph()
        communities = algorithms.louvain(g)

        readwrite.write_community_csv(communities, "coms.csv")
        communities_r = readwrite.read_community_csv("coms.csv", nodetype=int)
        self.assertListEqual(communities.communities, communities_r.communities)
        os.remove("coms.csv")
