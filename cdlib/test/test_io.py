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

        readwrite.write_community_csv(communities, "coms.gzip", compress=True)
        communities_r = readwrite.read_community_csv(
            "coms.gzip", nodetype=int, compress=True
        )
        self.assertListEqual(communities.communities, communities_r.communities)
        os.remove("coms.gzip")

    def test_read_write_json(self):
        g = nx.karate_club_graph()
        communities = algorithms.louvain(g)
        readwrite.write_community_json(communities, "coms.json")
        communities_r = readwrite.read_community_json("coms.json")
        self.assertListEqual(communities.communities, communities_r.communities)
        os.remove("coms.json")

        communities = algorithms.louvain(g)
        readwrite.write_community_json(communities, "coms.gzip", compress=True)
        communities_r = readwrite.read_community_json("coms.gzip", compress=True)
        self.assertListEqual(communities.communities, communities_r.communities)
        os.remove("coms.gzip")

        communities = algorithms.frc_fgsn(g, 1, 0.5, 3)
        readwrite.write_community_json(communities, "coms.json")
        communities_r = readwrite.read_community_json("coms.json")
        self.assertListEqual(communities.communities, communities_r.communities)
        os.remove("coms.json")

        communities = algorithms.hierarchical_link_community(g)
        readwrite.write_community_json(communities, "coms.json")
        communities_r = readwrite.read_community_json("coms.json")
        self.assertListEqual(communities.communities, communities_r.communities)

        with open("coms.json") as f:
            cr = f.read()
        readwrite.read_community_from_json_string(cr)
        os.remove("coms.json")
