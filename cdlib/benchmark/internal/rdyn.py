import networkx as nx
import numpy as np
import random

__author__ = "Giulio Rossetti"
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"
__version__ = "0.2.0"


class RDynV2(object):
    def __init__(
        self,
        size: int = 300,
        iterations: int = 5,
        avg_deg: int = 15,
        sigma: float = 0.6,
        lambdad: int = 1,
        alpha: float = 2.5,
        paction: int = 1,
        prenewal: float = 0.8,
        quality_threshold: float = 0.2,
        new_node: float = 0.0,
        del_node: float = 0.0,
        max_evts: int = 1,
    ):

        # set the network generator parameters
        self.size = size
        self.iterations = iterations
        self.avg_deg = avg_deg
        self.sigma = sigma
        self.lambdad = lambdad
        self.exponent = alpha
        self.paction = paction
        self.renewal = prenewal
        self.new_node = new_node
        self.del_node = del_node
        self.max_evts = max_evts

        # event targets
        self.communities_involved = []

        # initialize communities data structures
        self.communities = {}
        self.node_to_com = [i for i in range(0, size)]
        self.total_coms = 0
        self.performed_community_action = "START\n"
        self.quality_threshold = quality_threshold
        self.exp_node_degs = []

        # initialize the graph
        self.graph = nx.empty_graph(self.size)
        self.stable = 0

        self.it = 0
        self.count = 0

    def execute(self, simplified: bool = True) -> [object, object]:
        """
        :return:
        """
        # generate degree sequence
        self.__compute_degree_sequence()

        # generate community size dist
        exp_com_s = self.__compute_community_size_distribution()

        # assign node to community
        self.__node_to_community_initial_assignement(exp_com_s)

        # main loop (iteration)
        while self.it < self.iterations:

            # community check and event generation
            comp = nx.number_connected_components(self.graph)
            if comp <= len(self.communities):
                if self.__test_communities():
                    self.it += 1
                    yield self.graph, self.communities
                    self.__generate_event(simplified)

            # node removal
            ar = random.random()
            if ar < self.del_node:
                self.__remove_node()

            # node addition
            ar = random.random()
            if ar < self.new_node:
                self.__add_node()

            # get nodes within communities that needs to adjust
            nodes = self.__get_nodes()

            # inner loop (nodes)
            for n in nodes:

                # discard deleted nodes
                if self.node_to_com[n] == -1:
                    continue

                # check for decayed edges
                removal = self.__get_vanished_edges(n)

                # removal phase
                for n1 in removal:
                    r = random.random()

                    # edge renewal phase
                    # check for intra/inter renewal thresholds
                    if (
                        r <= self.renewal
                        and self.node_to_com[n1] == self.node_to_com[n]
                        or r > self.renewal
                        and self.node_to_com[n1] != self.node_to_com[n]
                    ):

                        # Exponential decay
                        timeout = (self.it + 1) + int(random.expovariate(self.lambdad))
                        self.graph.adj[n][n1]["d"] = timeout

                    else:
                        # edge to be removed
                        self.graph.remove_edge(n, n1)

                # expected degree reached
                if self.graph.degree(n) >= self.exp_node_degs[n]:
                    continue

                # decide if the node is active during this iteration
                action = random.random()

                # the node has not yet reached it expected degree and it acts in this round
                if self.graph.degree([n])[n] < self.exp_node_degs[n] and (
                    action <= self.paction or self.it == 0
                ):

                    com_nodes = list(self.communities[self.node_to_com[n]])

                    # probability for intra/inter community edges
                    r = random.random()

                    # check if at least sigma% of the node link are within the community
                    s = self.graph.subgraph(com_nodes)
                    d = s.degree([n])[n]  # Intra community edges

                    if r <= self.sigma and d < len(com_nodes) - 1:
                        self.__new_intra_community_edge(s, n)

                    # inter-community edges
                    elif r > self.sigma:
                        self.__new_inter_community_edge(n)

    def __new_intra_community_edge(self, s, n):

        n_neigh = set(s.neighbors(n))

        random.shuffle(list(n_neigh))
        target = None

        # selecting target node
        candidates = {
            j: (self.exp_node_degs[j] - self.graph.degree(j))
            for j in s.nodes()
            if (self.exp_node_degs[j] - self.graph.degree(j)) > 0 and j != n
        }

        if len(candidates) > 0:
            target = random.sample(list(candidates), 1)[0]

        # Interaction Exponential decay
        timeout = (self.it + 1) + int(random.expovariate(self.lambdad))

        # Edge insertion
        if target is not None and not self.graph.has_edge(n, target) and target != n:
            self.graph.add_edge(n, target, d=timeout)
            self.count += 1

    def __new_inter_community_edge(self, n):
        # randomly identifying a target community
        try:
            cid = random.sample(
                set(self.communities.keys()) - {self.node_to_com[n]}, 1
            )[0]
        except:
            return

        s = self.graph.subgraph(self.communities[cid])

        # check for available nodes within the identified community
        candidates = {
            j: (self.exp_node_degs[j] - self.graph.degree(j))
            for j in s.nodes()
            if (self.exp_node_degs[j] - self.graph.degree(j)) > 0 and j != n
        }

        # PA selection on available community nodes
        if len(candidates) > 0:
            candidatesp = list(
                np.array(list(candidates.values()), dtype="float")
                / sum(list(candidates.values()))
            )
            target = np.random.choice(list(candidates.keys()), 1, candidatesp)[0]

            if self.graph.has_node(target) and not self.graph.has_edge(n, target):
                # Interaction exponential decay
                timeout = (self.it + 1) + int(random.expovariate(self.lambdad))
                self.graph.add_edge(n, target, d=timeout)
                self.count += 1

    def __compute_degree_sequence(self):
        minv = float(self.avg_deg) / (2 ** (1 / (self.exponent - 1)))
        s = [2]
        while not nx.is_graphical(s):
            s = list(
                map(
                    int,
                    nx.utils.powerlaw_sequence(
                        self.graph.number_of_nodes(), self.exponent
                    ),
                )
            )
            x = [int(p + minv) for p in s]
        self.exp_node_degs = sorted(x)

    def __compute_community_size_distribution(self):
        min_node_degree = min(self.exp_node_degs)
        min_size_com = int(min_node_degree * self.sigma)
        s = list(
            map(
                int,
                nx.utils.powerlaw_sequence(
                    int(self.graph.number_of_nodes() / self.avg_deg), self.exponent
                ),
            )
        )
        sizes = sorted([p + min_size_com for p in s])
        while sum(sizes) > self.graph.number_of_nodes():
            for i, cs in enumerate(sizes):
                sizes[i] = cs - 1

        for c, _ in enumerate(sizes):
            self.communities[c] = []
        return sizes

    def __node_to_community_initial_assignement(self, community_sizes):
        degs = [(i, v) for i, v in enumerate(self.exp_node_degs)]
        unassigned = []

        for n in degs:
            nid, nd = n

            assigned = False
            for c, c_size in enumerate(community_sizes):
                c_taken = len(self.communities[c])

                # check if the node can be added to the community
                if float(nd) * self.sigma <= c_size and c_taken < c_size:
                    self.communities[c].append(nid)
                    assigned = True
                    break

            if not assigned:
                unassigned.append(n)

        if len(unassigned) > 0:
            for i in unassigned:
                for cid in self.communities:
                    self.communities[cid].append(i[0])
                    community_sizes[cid] += 1
                    self.exp_node_degs[i[0]] = community_sizes[cid] - 1
                    break

        ntc = []
        for cid, nodes in self.communities.items():
            for n in nodes:
                ntc.append((n, cid))

        for node_com in ntc:
            self.node_to_com[node_com[0]] = node_com[1]

    def __test_communities(self):
        mcond = 0
        for k in self.communities_involved:
            c = self.communities[k]
            if len(c) == 0:
                return False

            s = self.graph.subgraph(c)
            comps = nx.number_connected_components(s)

            if comps > 1:
                cs = nx.connected_components(s)
                i = random.sample(next(cs), 1)[0]
                j = random.sample(next(cs), 1)[0]
                timeout = (self.it + 1) + int(random.expovariate(self.lambdad))
                self.graph.add_edge(i, j, d=timeout)
                self.count += 1
                return False

            score = self.__conductance_test(k, s)
            if score > mcond:
                mcond = score
        if mcond > self.quality_threshold:
            return False
        return True

    def __conductance_test(self, comid, community):
        s_degs = community.degree()
        g_degs = self.graph.degree(community.nodes())

        # Conductance
        edge_across = 2 * sum([g_degs[n] - s_degs[n] for n in community.nodes()])
        c_nodes_total_edges = community.number_of_edges() + (2 * edge_across)

        if edge_across > 0:
            ratio = float(edge_across) / float(c_nodes_total_edges)
            if ratio > self.quality_threshold:
                self.communities_involved.append(comid)
                self.communities_involved = list(set(self.communities_involved))

                for i in community.nodes():
                    nn = list(self.graph.neighbors(i))
                    for j in nn:
                        if j not in community.nodes():
                            self.count += 1
                            self.graph.remove_edge(i, j)
                            continue
            return ratio
        return 0

    def __generate_event(self, simplified=True):

        communities_involved = []
        self.stable += 1

        options = ["M", "S"]

        evt_number = random.sample(range(1, self.max_evts + 1), 1)[0]
        evs = np.random.choice(options, evt_number, p=[0.5, 0.5], replace=True)
        chosen = []

        if len(self.communities) == 1:
            evs = "S"

        self.total_coms = len(self.communities)

        self.performed_community_action = ""

        for p in evs:

            if p == "M":
                # Generate a single merge
                if len(self.communities) == 1:
                    continue
                candidates = list(set(self.communities.keys()) - set(chosen))

                # promote merging of small communities
                cl = [len(v) for c, v in self.communities.items() if c in candidates]
                comd = 1 - np.array(cl, dtype="float") / sum(cl)
                comd /= sum(comd)

                ids = []
                try:
                    ids = np.random.choice(candidates, 2, p=list(comd), replace=False)
                except:
                    continue

                chosen.extend(ids)
                communities_involved.extend([ids[0]])

                for node in self.communities[ids[1]]:
                    self.node_to_com[node] = ids[0]

                self.communities[ids[0]].extend(self.communities[ids[1]])
                del self.communities[ids[1]]

            else:
                # Generate a single splits
                if len(self.communities) == 1:
                    continue

                candidates = list(set(self.communities.keys()) - set(chosen))

                cl = [len(v) for c, v in self.communities.items() if c in candidates]
                comd = np.array(cl, dtype="float") / sum(cl)

                try:
                    ids = np.random.choice(candidates, 1, p=list(comd), replace=False)
                except:
                    continue

                c_nodes = len(self.communities[ids[0]])

                if c_nodes > 6:
                    try:
                        size = random.sample(range(3, c_nodes - 3), 1)[0]
                        first = random.sample(self.communities[ids[0]], size)
                    except:
                        continue

                    cid = max(list(self.communities.keys())) + 1
                    chosen.extend([ids[0], cid])
                    communities_involved.extend([ids[0], cid])

                    # adjusting max degree
                    for node in first:
                        self.node_to_com[node] = cid
                        if self.exp_node_degs[node] > (len(first) - 1) * self.sigma:
                            self.exp_node_degs[node] = int(
                                (len(first) - 1) + (len(first) - 1) * (1 - self.sigma)
                            )

                    self.communities[cid] = first
                    self.communities[ids[0]] = [
                        ci for ci in self.communities[ids[0]] if ci not in first
                    ]

                    # adjusting max degree
                    for node in self.communities[ids[0]]:
                        if (
                            self.exp_node_degs[node]
                            > (len(self.communities[ids[0]]) - 1) * self.sigma
                        ):
                            self.exp_node_degs[node] = int(
                                (len(self.communities[ids[0]]) - 1)
                                + (len(self.communities[ids[0]]) - 1) * (1 - self.sigma)
                            )

        if not simplified:
            self.communities_involved = list(self.communities.keys())
        else:
            self.communities_involved = communities_involved

    def __add_node(self):
        nid = self.size
        self.graph.add_node(nid)
        cid = random.sample(list(self.communities.keys()), 1)[0]
        self.communities[cid].append(nid)
        self.node_to_com.append(cid)
        deg = random.sample(
            range(
                2,
                int(
                    (len(self.communities[cid]) - 1)
                    + (len(self.communities[cid]) - 1) * (1 - self.sigma)
                ),
            ),
            1,
        )[0]
        if deg == 0:
            deg = 1
        self.exp_node_degs.append(deg)
        self.size += 1

    def __remove_node(self):

        com_sel = [c for c, v in self.communities.items() if len(v) > 3]
        if len(com_sel) > 0:
            cid = random.sample(com_sel, 1)[0]
            s = self.graph.subgraph(self.communities[cid])
            sd = dict(s.degree)
            min_value = min(sd.values())
            candidates = [k for k in sd if sd[k] == min_value]
            nid = random.sample(candidates, 1)[0]
            eds = list(self.graph.edges([nid]))
            for e in eds:
                self.count += 1
                self.graph.remove_edge(e[0], e[1])

            self.exp_node_degs[nid] = 0
            self.node_to_com[nid] = -1
            nodes = set(self.communities[cid])
            self.communities[cid] = list(nodes - {nid})
            self.graph.remove_node(nid)

    def __get_nodes(self):
        if len(self.communities_involved) == 0:
            return self.graph.nodes()
        else:
            nodes = {}
            for cid in self.communities_involved:
                for nid in self.communities[cid]:
                    nodes[nid] = None
            return list(nodes.keys())

    def __get_vanished_edges(self, n):
        removal = []
        node_neighbors = nx.all_neighbors(self.graph, n)
        if len(self.communities) >= nx.number_connected_components(self.graph):
            for n1 in node_neighbors:
                delay = self.graph.get_edge_data(n, n1)["d"]
                if delay == self.it:
                    removal.append(n1)
        return removal


# def main():
#     import argparse
#
#     sys.stdout.write("-------------------------------------\n")
#     sys.stdout.write("               {RDyn}                \n")
#     sys.stdout.write("           Graph Generator      \n")
#     sys.stdout.write("     Handling Community Dynamics  \n")
#     sys.stdout.write("-------------------------------------\n")
#     sys.stdout.write("Author: " + __author__ + "\n")
#     sys.stdout.write("Email:  " + __email__ + "\n")
#     sys.stdout.write("------------------------------------\n")
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('nodes', type=int, help='Number of nodes', default=1000)
#     parser.add_argument('iterations', type=int, help='Number of iterations', default=1000)
#     parser.add_argument('simplified', type=bool, help='Simplified execution', default=True)
#     parser.add_argument('-d', '--avg_degree', type=int, help='Average node degree', default=15)
#     parser.add_argument('-s', '--sigma', type=float, help='Sigma', default=0.7)
#     parser.add_argument('-l', '--lbd', type=float, help='Lambda community size distribution', default=1)
#     parser.add_argument('-a', '--alpha', type=int, help='Alpha degree distribution', default=2.5)
#     parser.add_argument('-p', '--prob_action', type=float, help='Probability of node action', default=1)
#     parser.add_argument('-r', '--prob_renewal', type=float, help='Probability of edge renewal', default=0.8)
#     parser.add_argument('-q', '--quality_threshold', type=float, help='Conductance quality threshold', default=0.3)
#     parser.add_argument('-n', '--new_nodes', type=float, help='Probability of node appearance', default=0)
#     parser.add_argument('-j', '--delete_nodes', type=float, help='Probability of node vanishing', default=0)
#     parser.add_argument('-e', '--max_events', type=int, help='Max number of community events for stable iteration', default=1)
#
#     args = parser.parse_args()
#     rdyn = RDynV2(size=args.nodes, iterations=args.iterations, avg_deg=args.avg_degree,
#                 sigma=args.sigma, lambdad=args.lbd, alpha=args.alpha, paction=args.prob_action,
#                 prenewal=args.prob_renewal, quality_threshold=args.quality_threshold,
#                 new_node=args.new_nodes, del_node=args.delete_nodes, max_evts=args.max_events)
#     rdyn.execute(simplified=args.simplified)
