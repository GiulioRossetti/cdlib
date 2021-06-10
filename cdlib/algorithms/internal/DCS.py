import networkx as nx
import numpy as np
from functools import reduce


class Leader_Identification(object):
    def __init__(self, leader_epsilon):
        self.overlap = leader_epsilon

    @staticmethod
    def __inclusion(c1, c2):
        """
        :param c1: node neighbors
        :param c2: node neighbors
        """

        intersection = set(c2) & set(c1)
        smaller_set = min(len(c1), len(c2))
        return len(intersection) / float(smaller_set)

    def __search_leader(self, G, leaders, is_leader, epsilon):
        """
        :param G: G: the networkx graph on which perform DCS
        :param leadrs:list of leaders
        :param is_leader: check if this node satisfies our condition
        :param dlt_nodes: nodes that failed to satisfy our condition in previous iteration
        :param epsilon: the tolerance required in order to merge communities
        """

        com_val = []
        find_edge = []

        node_nbrs = list(G.neighbors(is_leader))

        if len(node_nbrs) > 2:

            for _, leader in enumerate(leaders):
                lead_nbrs = list(G.neighbors(leader))
                find_edge = np.append(find_edge, G.has_edge(is_leader, leader))
                com_val = np.append(com_val, self.__inclusion(node_nbrs, lead_nbrs))

        if len(com_val) > 0:
            return (
                len(com_val),
                len(np.where(np.array(com_val) <= epsilon)[0]),
                len(np.where(find_edge > 0)[0]),
            )
        else:
            if len(find_edge) > 0:
                return len(com_val), 0, len(np.where(find_edge > 0)[0])
            else:
                return len(com_val), 0, 0

    def leader_finding(self, G):
        """
        :param G: G: the networkx graph on which perform DCS
        """

        nod = [
            nx.degree(G, i) + sum(dict(nx.degree(G, G.neighbors(i))).values())
            for i in G.nodes()
        ]
        sort_with_extended_degree = sorted(
            np.column_stack((nod, G.nodes())), key=lambda x: x[0], reverse=True
        )
        leaders = [sort_with_extended_degree[i][1] for i in range(2)]

        for j, k in enumerate(sort_with_extended_degree[2:-1]):

            leader_condition = self.__search_leader(G, leaders, k[1], self.overlap)

            if (
                leader_condition[1] > 0
                and ((leader_condition[0] - leader_condition[1]) <= 2)
                and leader_condition[2] <= 1
            ):
                leaders.append(k[1])

        return leaders


class Merging(object):
    def __init__(self):
        """
        Constructor
        """

    @staticmethod
    def __generalized_inclusion(c1, c2, epsilon):
        """
        :param c1: community
        :param c2: community
        """

        res = None
        intersection = set(c2) & set(c1)
        smaller_set = min(len(c1), len(c2))

        if len(intersection) == 0:
            return None

        if not smaller_set == 0:
            res = float(len(intersection)) / float(smaller_set)

        if res >= epsilon:
            union = set(c2) | set(c1)
            return union

    def merge_communities(self, communities, actual_community, epsilon):

        """
        :param communities: dictionary of communities
        :param actual_community: a community
        :param epsilon: the tolerance required in order to merge communities
        """

        if tuple(actual_community) in communities:
            return communities

        else:

            inserted = False

            for test_community in communities.items():

                union = self.__generalized_inclusion(
                    actual_community, test_community[0], epsilon
                )

                if union is not None:
                    communities.pop(test_community[0])
                    communities[tuple(sorted(union))] = 0
                    inserted = True
                    break

            if not inserted:
                communities[tuple(sorted(actual_community))] = 0

        return communities


class Community(Merging):
    def __init__(self):
        """
        Constructor
        """

        Merging.__init__(self)
        self.all_communities = {}

    def execute(self, G, leader_nodes, epsilon, depth):
        """
        Execute CD algorithm
        :param leader_nodes:
        :param G: the networkx graph on which perform detection
        :param epsilon: the tolerance required in order to merge communities (default 0.5)
        :param depth: the level of depth for neighborhood extraction (1 or 2)
        """

        self.G = G
        self.epsilon = epsilon
        self.depth = depth
        self._communities(leader_nodes, G, nx.density(G), self.depth)

        return self.all_communities

    def _communities(self, leaders, G, graph_density, depth):
        """
        :param leader: leading nodes
        :param G: graph
        :param graph_density: density of the whole graph
        :param depth: neighboorhood order
        """

        for _, i in enumerate(leaders):

            order_neigborhod = 1
            new = []
            community_list = []
            dlt_nodes = []
            ego_nodes = set()

            while order_neigborhod <= depth:

                ego_nodes = self.__without_ego(
                    self.G, ego_nodes, order_neigborhod, dlt_nodes, i
                )
                extract_subgraph = G.subgraph(ego_nodes)
                edges_in_subgraph = extract_subgraph.size()
                nodes_to_be_checked_for_community = list(
                    set(ego_nodes).difference({i}, set(new))
                )
                new.extend(nodes_to_be_checked_for_community)
                outside_cc_edges = (
                    sum(dict(nx.degree(G, ego_nodes)).values()) - 2 * edges_in_subgraph
                )
                with_node_conductance = outside_cc_edges / (
                    (2.0 * edges_in_subgraph) + outside_cc_edges
                )

                for _, nods in enumerate(nodes_to_be_checked_for_community):

                    in_comm = extract_subgraph.degree(nods)
                    out_comm = G.degree(nods) - in_comm
                    outside_edges_of_a_node = edges_in_subgraph - in_comm
                    without_node_cf = (outside_cc_edges - out_comm) / (
                        0.001
                        + (2.0 * outside_edges_of_a_node)
                        + (outside_cc_edges - out_comm)
                    )
                    conductance_score_node = without_node_cf - with_node_conductance
                    subgraph_density_without_node = self.__generalized(
                        outside_edges_of_a_node, len(extract_subgraph) - 1
                    )
                    density_score_node = (
                        nx.density(extract_subgraph)
                        - subgraph_density_without_node
                        + graph_density
                    )

                    """condition for a node to be included in a community"""
                    if conductance_score_node >= 0 and density_score_node >= 0:
                        community_list.append(nods)
                    else:
                        dlt_nodes.append(nods)

                order_neigborhod += 1

            community_list.append(i)

            if len(community_list) <= 6:
                community_list.extend(extract_subgraph.nodes())

            self.all_communities = Merging.merge_communities(
                self, self.all_communities, list(set(community_list)), self.epsilon
            )
        return

    @staticmethod
    def __generalized(d1, d2):
        """
        :param d1: edges
        :param d2: nodes
        """

        if d2 == 1:
            return 0

        return (2.0 * d1) / (d2 * (d2 - 1))

    @staticmethod
    def __without_ego(G, list_nodes, root, dlt_nodes, leader):
        """
        :param G: graph
        :list_nodes: nodes included in the previous iteration
        :root: depth level
        :dlt_nodes: nodes that failed to satisfy our condition in previous iteration
        :leader: root node
        """

        if root == 1:
            nodes = {leader}
            nodes.update(G.neighbors(leader))
            return nodes
        else:
            remove_unimportant_nodes = list(set(list_nodes).difference(set(dlt_nodes)))
            nodes = set()
            nodes.update(remove_unimportant_nodes)
            for n in remove_unimportant_nodes:
                nodes.update(nx.neighbors(G, n))
            return set(nodes).difference(set(dlt_nodes))


class Bridge(Merging):
    def __init__(self, size=None):
        """
        Constructor
        """

        Merging.__init__(self)
        self.size = size

    @staticmethod
    def __assign_to_biggest_module(comm, community_list):
        comm = list(comm)
        comm.pop(np.argmax(community_list))
        community_list.remove(max(community_list))
        return list(reduce(lambda x, y: x + y, comm))

    @staticmethod
    def __delete_edges(graph, nolist, node):

        select_edges = set.intersection(set(nolist), set(nx.neighbors(graph, node)))
        select_edges_to_remove = [
            (j, node)
            for j in select_edges
            if (nx.degree(graph, node) > 1 and nx.degree(graph, j) > 1)
        ]

        graph.remove_edges_from(select_edges_to_remove)
        return graph

    def create_ap_points(self, G):
        """
        this method creates weak articulation points
        :param G: the networkx graph on which perform detection
        """

        for j in G:

            egonet = G.subgraph(nx.all_neighbors(G, j))
            local = nx.connected_components(egonet)
            extract_components = list(local)

            # to avoid whiskers, we verify this condition and it produced good results for >5
            if len(egonet.nodes()) > 5:

                all_communities = {}

                for k in extract_components:
                    for l in k:
                        # small value of epsilon or merging parameter to identiy divserse
                        # second-order neighborhood in the graph
                        all_communities = Merging.merge_communities(
                            self, all_communities, nx.neighbors(G, l), 0.30
                        )

                communities_counter = [len(x) for x in all_communities.keys()]

                if sum(np.bincount(communities_counter)[3:]) > 1:
                    # this process is similar for this function and bridge_function
                    tobedel = self.__assign_to_biggest_module(
                        all_communities.keys(), communities_counter
                    )
                    G = nx.Graph(G)
                    self.__delete_edges(G, tobedel, j)

        return G

    def bridge_function(self, G):
        """
        bridge function to identify weak nodes of the graph
        :param G: the networkx graph on which perform detection
        """

        components = nx.connected_components(G)
        big_mod = {}
        normal_mod = {}
        bigmodcounter = normmodcounter = 0

        for _, i in enumerate(components):
            i = nx.subgraph(G, i)
            for _, j in enumerate(i):

                egonet = i.subgraph(nx.all_neighbors(i, j))
                local = nx.connected_components(egonet)
                conn_comp = list(local)

                if len(conn_comp) > 1:

                    all_communities = {}

                    for k in conn_comp:

                        egos_local_communities = set(k)
                        for l in k:
                            egos_local_communities.update(nx.neighbors(i, l))

                        egos_local_communities.remove(j)

                        # merge two list if they have even a single member in common
                        all_communities = Merging.merge_communities(
                            self, all_communities, list(egos_local_communities), 0
                        )

                    communities_counter = [len(x) for x in all_communities.keys()]

                    if sum(np.bincount(communities_counter)[3:]) > 1:
                        # if big modules identified then remove the edges and assign
                        # the node to the largest component connected to it
                        tobedel = self.__assign_to_biggest_module(
                            all_communities.keys(), communities_counter
                        )
                        i = nx.Graph(i)
                        self.__delete_edges(i, tobedel, j)

            assign_modules = list(nx.connected_components(i))

            # check modules size and assign them
            for i, mod in enumerate(assign_modules):

                if len(mod) > self.size:
                    big_mod[bigmodcounter] = mod
                    bigmodcounter += 1

                else:
                    normal_mod[normmodcounter] = mod
                    normmodcounter += 1

        return big_mod, normal_mod


def main_dcs(G):
    # local centrality to identify meaningful modules of the graph
    # we set 3000 as a module size as discusssed in our paper
    local_centrality = Bridge(size=3000)
    bigmod, normmod = local_centrality.bridge_function(G)

    for _, j in enumerate(bigmod.values()):

        G1 = G.subgraph(j)
        G2 = local_centrality.create_ap_points(G1)
        _, norm_modules = local_centrality.bridge_function(G2)

        for i, k in enumerate(norm_modules.values(), start=len(normmod)):
            normmod[i] = k

    # leader identification and community spreading phase
    communities = {}
    comm_counter = 0

    for j, i in enumerate(normmod.values()):

        # for very small modules, we search for its neighbors and merge them in the communities
        if len(i) <= 9:

            sma = set(i)
            for k in i:
                sma.update(nx.neighbors(G, k))

            communities[comm_counter] = sma
            comm_counter += 1

        else:

            # to avoid local heterogeneity we set epsilon (merging value)
            # of 0.75 for big modules and 0.30 for small modules
            if 10 <= len(i) <= 500:
                heterogeneity = 0.30
            else:
                heterogeneity = 0.75

            extract_G = G.subgraph(i)

            lead = Leader_Identification(leader_epsilon=0.60)
            leaders = lead.leader_finding(extract_G)

            cd = Community()
            comm_list = cd.execute(extract_G, leaders, epsilon=heterogeneity, depth=2)

            for c in comm_list.keys():
                communities[comm_counter] = c
                comm_counter += 1

    cms = []
    for c in communities.values():
        cms.append(list(c))

    return cms
