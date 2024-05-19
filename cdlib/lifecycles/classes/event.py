from cdlib.classes import TemporalClustering
from cdlib.lifecycles.classes.matching import CommunityMatching
from cdlib.lifecycles.algorithms.null_model import flow_null, all_flows_null
from cdlib.lifecycles.algorithms.event_analysis import (
    events_all,
    analyze_all_flows,
    analyze_flow,
)
from cdlib.lifecycles.algorithms.classic_match import *
import networkx as nx
from collections import defaultdict
from typing import Callable
import json


class CommunityEvent(object):
    def __init__(self, com_id):
        """
        Constructor

        :param com_id: community id
        """

        self.com_id = com_id
        self.from_event = {}
        self.to_event = {}
        self.in_flow = {}
        self.out_flow = {}

    def set_from_event(self, from_event: dict):
        """
        Set from event

        :param from_event: from event
        """
        self.from_event = {f: v for f, v in from_event.items() if v > 0}

    def set_to_event(self, to_event: dict):
        """
        Set to event

        :param to_event: to event
        """
        self.to_event = {t: v for t, v in to_event.items() if v > 0}

    def set_in_flow(self, in_flow: dict):
        """
        Set in flow

        :param in_flow: in flow
        """
        self.in_flow = in_flow

    def set_out_flow(self, out_flow: dict):
        """
        Set out flow

        :param out_flow: out flow
        """
        self.out_flow = out_flow

    def get_from_event(self) -> dict:
        """
        Get from event

        :return: from event
        """
        return self.from_event

    def get_to_event(self) -> dict:
        """
        Get to event

        :return: to event
        """
        return self.to_event

    def get_in_flow(self) -> dict:
        """
        Get in flow

        :return: in flow
        """
        return self.in_flow

    def get_out_flow(self) -> dict:
        """
        Get out flow

        :return: out flow
        """
        return self.out_flow

    def to_json(self) -> dict:
        """
        Convert the event to json

        :return: the event as json
        """
        res = {
            "com_id": self.com_id,
            "from_event": self.from_event,
            "to_event": self.to_event,
            "in_flow": {k: list(v) for k, v in self.in_flow.items()},
            "out_flow": {k: list(v) for k, v in self.out_flow.items()},
        }

        return res


class LifeCycle(object):
    """
    Class representing the lifecycle of a temporal clustering.
    It allows to compute the events composing the lifecycle (leveraging different definitions)
    and to analyze them starting from a TemporalClustering object.
    """

    def __init__(self, clustering: TemporalClustering = None):
        """
        Constructor

        :param clustering: a TemporalClustering Object

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        """
        self.clustering = clustering
        self.events = {}
        self.event_types = []
        self.cm = CommunityMatching()
        if clustering is not None:
            self.cm.set_temporal_clustering(self.clustering)
        self.algo = None

    def compute_events_from_explicit_matching(self):
        """
        Compute the events of the lifecycle using the explicit matching (if available)

         :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> from dynetx import DynGraph
        >>> dg = DynGraph()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     dg.add_interactions_from(g, t)
        >>> tc = algorithms.tiles(dg, 10)
        >>> events = LifeCycle(tc)
        >>> events.compute_events_from_explicit_matching()
        """
        if not self.clustering.has_explicit_match():
            raise ValueError("No explicit matching available")

        lifecycle = self.clustering.get_explicit_community_match()

        flows = {
            "+": defaultdict(lambda: defaultdict(set)),
            "-": defaultdict(lambda: defaultdict(set)),
        }
        events = {
            "+": defaultdict(lambda: defaultdict(set)),
            "-": defaultdict(lambda: defaultdict(set)),
        }

        for e in lifecycle:
            xtid = int(e[0].split("_")[0])
            ytid = int(e[1].split("_")[0])
            if xtid < ytid:
                flows["+"][e[0]][e[1]] = set(
                    self.clustering.get_community(e[0])
                ).intersection(set(self.clustering.get_community(e[1])))
            else:
                flows["-"][e[0]][e[1]] = set(
                    self.clustering.get_community(e[0])
                ).intersection(set(self.clustering.get_community(e[1])))

        self.__instantiate_events(flows, events)

    def compute_events_with_custom_matching(
        self,
        method: Callable[[set, set], float],
        two_sided: bool = True,
        threshold: float = 0.2,
    ):
        """
        Compute the events of the lifecycle using a custom matching similarity function


        :param method: a set similarity function with co-domain in [0,1] (e.g., Jaccard)
        :param two_sided: boolean.
                            Whether the match has to be applied only from the past to the future (False)
                            or even from the future to the past (True, default)
        :param threshold: the threshold above which two communities are considered matched

        :Example:

        >>> from cdlib import algorithms
        >>> from cdlib import TemporalClustering, LifeCycle
        >>>  tc = TemporalClustering()
        >>> # build the temporal clustering object
        >>> evts = LifeCycle(tc)
        >>> jaccard = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
        >>> evts.compute_events_with_custom_matching(jaccard, two_sided=True, threshold=0.2)
        """

        self.event_types = ["Merge", "Split", "Continuation"]
        self.algo = "custom"
        lifecycle = []

        for i in range(self.clustering.current_observation - 1):
            c_i = self.clustering.clusterings[i]
            c_j = self.clustering.clusterings[i + 1]
            for name_i, com_i in c_i.named_communities.items():

                best_match = []
                best_score = 0

                for name_j, com_j in c_j.named_communities.items():

                    match = method(com_i, com_j)
                    if match > best_score:
                        best_match = [name_j]
                        best_score = match
                    elif match == best_score:
                        best_match.append(name_j)

                for j in best_match:
                    lifecycle.append((name_i, j, best_score))

        if two_sided:

            for i in range(self.clustering.current_observation - 1, 0, -1):
                c_i = self.clustering.clusterings[i]
                c_j = self.clustering.clusterings[i - 1]

                for name_i, com_i in c_i.named_communities.items():
                    # name_i = f"{self.obs_to_time[i]}_{cid_i}"
                    best_match = []
                    best_score = 0

                    for name_j, com_j in c_j.named_communities.items():
                        # name_j = f"{self.obs_to_time[i-1]}_{cid_j}"

                        match = method(com_i, com_j)
                        if match > best_score:
                            best_match = [name_j]
                            best_score = match
                        elif match == best_score:
                            best_match.append(name_j)

                    for j in best_match:
                        lifecycle.append((j, name_i, best_score))

        flows = {
            "+": defaultdict(lambda: defaultdict(set)),
            "-": defaultdict(lambda: defaultdict(set)),
        }
        events = {
            "+": defaultdict(lambda: defaultdict(set)),
            "-": defaultdict(lambda: defaultdict(set)),
        }

        for e in lifecycle:
            xtid = int(e[0].split("_")[0])
            ytid = int(e[1].split("_")[0])
            if e[2] > threshold:
                if xtid < ytid:
                    flows["+"][e[0]][e[1]] = set(
                        self.clustering.get_community(e[0])
                    ).intersection(set(self.clustering.get_community(e[1])))
                else:
                    flows["-"][e[0]][e[1]] = set(
                        self.clustering.get_community(e[0])
                    ).intersection(set(self.clustering.get_community(e[1])))

                self.__instantiate_events(flows, events)

    def __instantiate_events(self, flows, events):
        for e in flows["-"]:
            if len(flows["-"][e].keys()) == 1:
                events["-"][e] = {"Continuation": 1}
            else:
                events["-"][e] = {"Merge": 1}

        for e in flows["+"]:
            if len(flows["+"][e].keys()) == 1:
                events["+"][e] = {"Continuation": 1}
            else:
                events["+"][e] = {"Split": 1}

        for cid in flows["+"]:
            if cid not in self.events:
                self.events[cid] = CommunityEvent(cid)
            self.events[cid].set_out_flow(flows["+"][cid])

        for cid in flows["-"]:
            if cid not in self.events:
                self.events[cid] = CommunityEvent(cid)
            self.events[cid].set_in_flow(flows["-"][cid])

        from_events = events["-"]
        to_events = events["+"]

        for cid in from_events:
            self.events[cid].set_from_event(
                {k: v for k, v in from_events[cid].items() if v > 0}
            )

        for cid in to_events:
            self.events[cid].set_to_event(
                {k: v for k, v in to_events[cid].items() if v > 0}
            )

    def compute_events(
        self,
        matching_type: str = "facets",
        matching_params: dict = {"min_branch_size": 1, "threshold": 0.5},
    ):
        """
        Compute the events of the lifecycle

        :param matching_type: the type of matching algorithm to use. Options are "facets", "asur", "greene".
        :param matching_params: the parameters of the matching algorithm.
                                Defaults to {"min_branch_size": 1, "threshold": 0.5}.
                                The former parameter is required for "facets", the latter by "asur" and "greene".

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")

        """

        if matching_type == "facets":

            self.algo = "facets"

            self.event_types = [
                "Birth",
                "Accumulation",
                "Growth",
                "Expansion",
                "Continuation",
                "Merge",
                "Offspring",
                "Reorganization",
                "Death",
                "Dispersion",
                "Shrink",
                "Reduction",
                "Continuation",
                "Split",
                "Ancestor",
                "Disassemble",
            ]

            out_flows = self.cm.all_flows(
                "+", min_branch_size=matching_params["min_branch_size"]
            )

            for cid in out_flows:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_out_flow(out_flows[cid])

            in_flows = self.cm.all_flows(
                "-", min_branch_size=matching_params["min_branch_size"]
            )

            for cid in in_flows:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_in_flow(in_flows[cid])

            events = events_all(self.cm)
            from_events = events["-"]
            to_events = events["+"]

            for cid in from_events:
                self.events[cid].set_from_event(
                    {k: v for k, v in from_events[cid].items() if v > 0}
                )

            for cid in to_events:
                self.events[cid].set_to_event(
                    {k: v for k, v in to_events[cid].items() if v > 0}
                )

        elif matching_type == "asur":

            self.algo = "asur"

            self.event_types = ["Merge", "Split", "Continuation", "Birth", "Death"]

            events, flows = events_asur(self.cm, th=matching_params["threshold"])

            c_to_evt = defaultdict(lambda: defaultdict(int))
            c_from_evt = defaultdict(lambda: defaultdict(int))
            for _, v in events.items():

                for e in v:
                    src_tid = int(e["src"].split("_")[0])
                    if "ref_sets" in e:
                        trg_tid = int(e["ref_sets"][0].split("_")[0])
                    else:
                        trg_tid = int(e["ref_set"].split("_")[0])

                    if src_tid < trg_tid:
                        c_to_evt[e["src"]][e["type"]] += 1
                    else:
                        c_from_evt[e["src"]][e["type"]] += 1

            c_from_evt = {
                k: {k2: v2 / sum(v.values()) for k2, v2 in v.items() if v2 > 0}
                for k, v in c_from_evt.items()
            }
            c_to_evt = {
                k: {k2: v2 / sum(v.values()) for k2, v2 in v.items() if v2 > 0}
                for k, v in c_to_evt.items()
            }

            c_from_flow = defaultdict(lambda: defaultdict(list))
            c_to_flow = defaultdict(lambda: defaultdict(list))

            for _, v in flows.items():
                for e in v:
                    src_tid = int(e["src"].split("_")[0])
                    trg_tid = int(e["target"].split("_")[0])

                    if src_tid < trg_tid:
                        c_from_flow[e["src"]][e["target"]] = e["flow"]
                    else:
                        c_to_flow[e["src"]][e["target"]] = e["flow"]

            for cid in c_to_flow:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_in_flow(c_to_flow[cid])

            for cid in c_from_flow:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_out_flow(c_from_flow[cid])

            for cid in c_to_evt:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_to_event(
                    {k: v for k, v in c_to_evt[cid].items() if v > 0}
                )

            for cid in c_from_evt:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_from_event(
                    {k: v for k, v in c_from_evt[cid].items() if v > 0}
                )

        elif matching_type == "greene":

            self.algo = "greene"

            self.event_types = ["Merge"]

            events, flow = event_graph_greene(self.cm, th=matching_params["threshold"])
            c_to_evt = defaultdict(lambda: defaultdict(int))
            c_from_evt = defaultdict(lambda: defaultdict(int))
            for _, v in events.items():

                for e in v:
                    src_tid = int(e["src"].split("_")[0])
                    if "ref_sets" in e:
                        trg_tid = int(e["ref_sets"][0].split("_")[0])
                    else:
                        trg_tid = int(e["ref_set"].split("_")[0])

                    if src_tid < trg_tid:
                        c_to_evt[e["src"]][e["type"]] += 1
                    else:
                        c_from_evt[e["src"]][e["type"]] += 1

            for cid in flow:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_in_flow(flow[cid])

            for cid in c_to_evt:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_to_event(
                    {k: v for k, v in c_to_evt[cid].items() if v > 0}
                )

            for cid in c_from_evt:
                if cid not in self.events:
                    self.events[cid] = CommunityEvent(cid)
                self.events[cid].set_from_event(
                    {k: v for k, v in c_from_evt[cid].items() if v > 0}
                )

        else:
            raise ValueError(f"Unknown matching type {matching_type}")

    def get_event(self, com_id: str) -> CommunityEvent:
        """
        Get the events associated to a community

        :param com_id: the community id
        :return: the events associated to the community

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> evt = events.get_event("0_2")

        """
        return self.events.get(com_id)

    def get_events(self) -> dict:
        """
        Get all the events

        :return: the events

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> evts = events.get_events()
        """
        return self.events

    def get_event_types(self) -> list:
        """
        Get the event types

        :return: the event types

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> evts = events.get_event_types()
        """
        return self.event_types

    def analyze_flows(
        self, direction: str = "+", min_branch_size: int = 1, attr=None
    ) -> dict:
        """
        Analyze the flows of the lifecycle

        :param direction: the temporal direction in which the flows are to be analyzed. Options are "+" and "-".
        :param min_branch_size: the minimum branch size
        :param attr: the attribute to analyze
        :return: the analyzed flows

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> c = events.analyze_flows("+")

        """
        if self.cm is not None:
            return analyze_all_flows(self.cm, direction, min_branch_size, attr)
        else:
            raise ValueError("No temporal clustering set")

    def analyze_flow(
        self, com_id: str, direction: str = "+", min_branch_size: int = 1, attr=None
    ) -> dict:
        """
        Analyze the flow of a community

        :param com_id: the community id
        :param direction: the temporal direction in which the flow is to be analyzed. Options are "+" and "-".
        :param min_branch_size: the minimum branch size
        :param attr: the attribute to analyze
        :return: the analyzed flow

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        """
        if self.cm is not None:
            return analyze_flow(self.cm, com_id, direction, min_branch_size, attr)
        else:
            raise ValueError("No temporal clustering set")

    def set_attribute(self, attr: dict, name: str):
        """
        Set the attributes of the lifecycle

        :param attr: the attributes
        :param name: the name of the attribute

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> import random
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>>
        >>> def random_attributes():
        >>>     attrs = {}
        >>>     for i in range(250):
        >>>        attrs[i] = {}
        >>>        for t in range(10):
        >>>             attrs[i][t] = random.choice(["A", "B", "C", "D", "E"])
        >>>     return attrs
        >>>
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> events.set_attribute(random_attributes(), "fakeattribute")

        """
        if self.cm is not None:
            self.cm.set_attributes(attr, name)
        else:
            raise ValueError("No temporal clustering set")

    def get_attribute(self, name: str) -> dict:
        """
        Get the attributes associated to the nodes

        :param name: the name of the attribute

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> import random
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>>
        >>> def random_attributes():
        >>>     attrs = {}
        >>>     for i in range(250):
        >>>        attrs[i] = {}
        >>>        for t in range(10):
        >>>             attrs[i][t] = random.choice(["A", "B", "C", "D", "E"])
        >>>     return attrs
        >>>
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> events.set_attribute(random_attributes(), "fakeattribute")
        >>> attrs = events.get_attribute("fakeattribute")
        """
        if self.cm is not None:
            return self.cm.get_attributes(name)
        else:
            raise ValueError("No temporal clustering set")

    def polytree(self) -> nx.DiGraph:
        """
        Reconstruct the poly-tree representing communities lifecycles using a provided similarity function.

        :return: a networkx DiGraph object.
                Nodes represent communities, their ids are assigned following the pattern {tid}_{cid},
                where tid is the time of observation and
                cid is the position of the community within the Clustering object.

        :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> g = events.polytree()
        """

        g = nx.DiGraph()
        for e in self.events:
            evt = self.events[e]
            for f in evt.get_in_flow():
                g.add_edge(f, e)
            for t in evt.get_out_flow():
                g.add_edge(e, t)

        return g

    def validate_flow(
        self,
        target: str,
        direction: str,
        min_branch_size: int = 1,
        iterations: int = 1000,
    ) -> dict:
        """
        Compare the flow with a null model. Each branch of each flow is compared with a null branch of the same size.
        The null model is generated by randomly sampling elements from the reference partition *iterations* times.
        The mean and standard deviation of the null model are used to compute a z-score
        for each branch, which is then used to compute a p-value.

        :param target: target set identifier
        :param direction: temporal direction, either "+" (out flow) or "-" (in flow)
        :param min_branch_size: minimum size of a branch to be considered
        :param iterations: number of random draws to be used to generate the null model
        :return:

         :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> cf = events.validate_flow("0_2", "+")
        """
        return flow_null(self.cm, target, direction, min_branch_size, iterations)

    def validate_all_flows(
        self, direction: str, min_branch_size=1, iterations=1000
    ) -> dict:
        """
        Compare all flows with null models. See validate_flow for details.

        :param direction: temporal direction, either "+" (out flow) or "-" (in flow)
        :param min_branch_size: minimum size of a branch to be considered
        :param iterations: number of random draws to be used to generate the null model
        :return: a dictionary keyed by set identifier and valued by mean, std, and p-value

         :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> vf = events.validate_all_flows("+")
        """
        return all_flows_null(self.cm, direction, min_branch_size, iterations)

    def to_json(self) -> dict:
        """
        Convert the lifecycle to json

        :return: the lifecycle as json

         :Example:

        >>> from cdlib import TemporalClustering, LifeCycle
        >>> from cdlib import algorithms
        >>> from networkx.generators.community import LFR_benchmark_graph
        >>> tc = TemporalClustering()
        >>> for t in range(0, 10):
        >>>     g = LFR_benchmark_graph(
        >>>         n=250,
        >>>         tau1=3,
        >>>         tau2=1.5,
        >>>         mu=0.1,
        >>>         average_degree=5,
        >>>         min_community=20,
        >>>         seed=10,
        >>>     )
        >>>     coms = algorithms.louvain(g)  # here any CDlib algorithm can be applied
        >>>     tc.add_clustering(coms, t)
        >>> events = LifeCycle(tc)
        >>> events.compute_events("facets")
        >>> events.to_json()
        """
        res = {
            "algorithm": self.algo,
            "events": {k: v.to_json() for k, v in self.events.items()},
            "event_types": list(self.event_types),
        }

        return res
