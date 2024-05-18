import unittest

from cdlib import readwrite
from cdlib import LifeCycle, TemporalClustering
from cdlib import algorithms
from networkx.generators.community import LFR_benchmark_graph
from cdlib.readwrite import write_lifecycle_json, read_lifecycle_json
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

    def test_events_read_write(self):

        tc = TemporalClustering()
        for t in range(0, 10):
            g = LFR_benchmark_graph(
                n=250,
                tau1=3,
                tau2=1.5,
                mu=0.1,
                average_degree=5,
                min_community=20,
                seed=10,
            )
            coms = algorithms.louvain(g)
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")
        write_lifecycle_json(events, "lifecycle.json")
        e = read_lifecycle_json("lifecycle.json")
        self.assertIsInstance(e, LifeCycle)
        os.remove("lifecycle.json")
