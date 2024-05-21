import unittest
import cdlib
from cdlib import algorithms
from cdlib import LifeCycle
from cdlib import TemporalClustering
from cdlib.lifecycles.algorithms.event_analysis import (
    facets,
    event_weights,
    event as evn,
)
from plotly import graph_objects as go
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import matplotlib.pyplot as plt
import dynetx as dn
import os
from cdlib.viz import (
    plot_flow,
    plot_event_radar,
    plot_event_radars,
    typicality_distribution,
)


class EventTest(unittest.TestCase):
    def test_creation(self):

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
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")

        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)
        c = events.analyze_flow("0_2", "+")
        self.assertIsInstance(c, dict)

        events = LifeCycle(tc)
        events.compute_events("asur")

        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)
        c = events.analyze_flow("0_2", "+")
        self.assertIsInstance(c, dict)

        events = LifeCycle(tc)
        events.compute_events("greene")

        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)

        c = events.analyze_flow("0_2", "+")
        self.assertIsInstance(c, dict)

    def test_custom_matching(self):
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
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        jaccard = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
        events.compute_events_with_custom_matching(jaccard, two_sided=True)
        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)

        events.compute_events_with_custom_matching(
            jaccard, two_sided=False, threshold=0
        )
        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)

    def test_polytree(self):
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
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")
        g = events.polytree()
        self.assertIsInstance(g, nx.DiGraph)

    def test_null_model(self):
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
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")
        cf = events.validate_flow("0_2", "+")
        self.assertIsInstance(cf, dict)

        vf = events.validate_all_flows("+")
        self.assertIsInstance(vf, dict)

    def test_viz(self):

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
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")

        fig = plot_flow(events)
        self.assertIsInstance(fig, go.Figure)

        plot_event_radar(events, "0_2", direction="+")
        plt.savefig("radar.pdf")
        os.remove("radar.pdf")

        plot_event_radars(events, "0_2")
        plt.savefig("radars.pdf")
        os.remove("radars.pdf")

        typicality_distribution(events, "+")
        plt.savefig("td.pdf")
        os.remove("td.pdf")

    def test_explicit(self):

        dg = dn.DynGraph()
        for x in range(10):
            g = nx.erdos_renyi_graph(200, 0.05)
            dg.add_interactions_from(list(g.edges()), t=x)
        coms = algorithms.tiles(dg, 2)

        events = LifeCycle(coms)
        events.compute_events_from_explicit_matching()

        c = events.analyze_flows("+")
        self.assertIsInstance(c, dict)

    def test_node_attributes(self):
        import random

        def random_attributes():
            attrs = {}
            for i in range(250):
                attrs[i] = {}
                for t in range(10):
                    attrs[i][t] = random.choice(["A", "B", "C", "D", "E"])
            return attrs

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
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")
        events.set_attribute(random_attributes(), "fakeattribute")
        attrs = events.get_attribute("fakeattribute")
        self.assertIsInstance(attrs, dict)

        events.analyze_flow("1_1", "+", attr="fakeattribute")
        self.assertIsInstance(attrs, dict)

        ev = events.get_event("1_1")
        a = ev.out_flow  # to get the out flow of the community 1_2
        self.assertIsInstance(a, dict)
        a = ev.in_flow  # to get the in flow of the community 1_2
        self.assertIsInstance(a, dict)
        a = ev.from_event  # to get the from events of the community 1_2
        self.assertIsInstance(a, dict)
        a = ev.to_event  # to get the to events of the community 1_2
        self.assertIsInstance(a, dict)

    def test_marginal(self):
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
            coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
            tc.add_clustering(coms, t)

        events = LifeCycle(tc)
        events.compute_events("facets")

        # marginal tests (not all methods are tested since they are not of use in cdlib -
        # they are invoked for completeness)
        self.assertIsInstance(
            events.cm.slice(0, 5), cdlib.lifecycles.classes.matching.CommunityMatching
        )
        self.assertIsInstance(events.cm.universe_set(), set)
        self.assertIsInstance(list(events.cm.group_iterator()), list)
        self.assertIsInstance(list(events.cm.group_iterator(3)), list)
        events.cm.filter_on_group_size(1, 100)
        events.cm.get_element_membership(1)
        events.cm.get_all_element_memberships()
        events.get_events()
        events.get_event_types()
        ev = events.get_event("1_1")
        ev.get_from_event()
        ev.get_to_event()
        facets((events.cm), "0_2", "+")
        event_weights(events.cm, "0_2", "+")
        evn(events.cm, "0_2", "+")


if __name__ == "__main__":
    unittest.main()
