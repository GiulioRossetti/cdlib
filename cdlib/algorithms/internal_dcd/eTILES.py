import copy
import networkx as nx

__author__ = "Giulio Rossetti"
__contact__ = "giulio.rossetti@gmail.com"
__website__ = "about.giuliorossetti.net"
__license__ = "BSD"


class eTILES(object):
    """
    TILES
    Algorithm for evolutionary community discovery
    ***Explicit removal***
    """

    def __init__(self, dg: object, obs: int = 7):
        """
        Constructor
        :param g: DyNetx graph
        :param obs: observation window
        :param start: starting date
        :param end: ending date
        """
        self.cid = 0
        self.actual_slice = obs
        self.dg = dg
        self.g = nx.Graph()
        self.splits = None
        self.removed = 0
        self.added = 0
        self.obs = obs
        self.communities = {}
        self.mathces = []

    @property
    def new_community_id(self) -> int:
        """
        Return a new community identifier
        :return: new community id
        """
        self.cid += 1
        self.communities[self.cid] = {}
        return self.cid

    def execute(self) -> dict:
        """
        Execute TILES algorithm
        """

        last_break = min(self.dg.temporal_snapshots_ids())
        count = 0

        #################################################
        #                   Main Cycle                  #
        #################################################

        for l in self.dg.stream_interactions():
            e = {}
            action = l[2]
            u = l[0]
            v = l[1]
            dt = int(l[3])

            e["weight"] = 1
            e["u"] = u
            e["v"] = v

            #############################################
            #               Observations                #
            #############################################

            dif = dt - last_break

            if dif >= self.obs:
                last_break = dt
                self.added -= 1
                self.added = 1
                self.removed = 0
                self.print_communities()
                yield self.communities

            if u == v:
                continue

            if action == "-":
                self.remove_edge(e)
                continue

            if not self.g.has_node(u):
                self.g.add_node(u)
                self.g.nodes[u]["c_coms"] = {}

            if not self.g.has_node(v):
                self.g.add_node(v)
                self.g.nodes[v]["c_coms"] = {}

            if self.g.has_edge(u, v):
                w = self.g.adj[u][v]["weight"]
                self.g.adj[u][v]["weight"] = w + e["weight"]
                continue
            else:
                self.g.add_edge(u, v, weight=e["weight"])

            u_n = list(self.g.neighbors(u))
            v_n = list(self.g.neighbors(v))

            #############################################
            #               Evolution                   #
            #############################################

            # new community of peripheral nodes (new nodes)
            if len(u_n) > 1 and len(v_n) > 1:
                common_neighbors = set(u_n) & set(v_n)
                self.common_neighbors_analysis(u, v, common_neighbors)

            count += 1

        #  Last writing
        self.added = 0
        self.removed = 0
        self.print_communities()

        yield self.communities

    def print_communities(self):
        """
        Print the actual communities
        """

        nodes_to_coms = {}
        merge = {}
        coms_to_remove = []
        drop_c = []

        for idc, comk in self.communities.items():

            com = comk.keys()

            if self.communities[idc] is not None:
                if len(com) > 2:
                    key = tuple(sorted(com))

                    # Collision check and merge index build (maintaining the lowest id)
                    if key not in nodes_to_coms:
                        nodes_to_coms[key] = idc
                    else:
                        old_id = nodes_to_coms[key]
                        drop = idc
                        if idc < old_id:
                            drop = old_id
                            nodes_to_coms[key] = idc

                        # merged to remove
                        coms_to_remove.append(drop)
                        if not nodes_to_coms[key] in merge:
                            merge[nodes_to_coms[key]] = [idc]
                        else:
                            merge[nodes_to_coms[key]].append(idc)
                else:
                    drop_c.append(idc)
            else:
                drop_c.append(idc)

        for dc in drop_c:
            self.destroy_community(dc)

        # Community Cleaning
        for comid, c_val in merge.items():
            # maintain minimum community after merge
            c_val.append(comid)
            k = min(c_val)

            c_val.remove(k)
            if self.actual_slice > self.obs:
                for fr in c_val:
                    self.mathces.append(
                        (
                            f"{self.actual_slice-self.obs}_{fr}",
                            f"{self.actual_slice}_{k}",
                            None,
                        )
                    )

        m = 0
        for c in coms_to_remove:
            self.destroy_community(c)
            m += 1

        self.actual_slice += self.obs

    def common_neighbors_analysis(self, u, v, common_neighbors):
        """
        General case in which both the nodes are already present in the net.
        :param u: a node
        :param v: a node
        :param common_neighbors: common neighbors of the two nodes
        """

        # no shared neighbors
        if len(common_neighbors) < 1:
            return

        else:

            shared_coms = set(self.g.nodes[v]["c_coms"].keys()) & set(
                self.g.nodes[u]["c_coms"].keys()
            )
            only_u = set(self.g.nodes[u]["c_coms"].keys()) - set(
                self.g.nodes[v]["c_coms"].keys()
            )
            only_v = set(self.g.nodes[v]["c_coms"].keys()) - set(
                self.g.nodes[u]["c_coms"].keys()
            )

            # community propagation: a community propagates iff at least two of [u, v, z] are central
            propagated = False

            for z in common_neighbors:
                for c in self.g.nodes[z]["c_coms"].keys():
                    if c in only_v:
                        self.add_to_community(u, c)
                        propagated = True

                    if c in only_u:
                        self.add_to_community(v, c)
                        propagated = True

                for c in shared_coms:
                    if c not in self.g.nodes[z]["c_coms"]:
                        self.add_to_community(z, c)
                        propagated = True

            else:
                if not propagated:
                    # new community
                    actual_cid = self.new_community_id
                    self.add_to_community(u, actual_cid)
                    self.add_to_community(v, actual_cid)

                    for z in common_neighbors:
                        self.add_to_community(z, actual_cid)

    def remove_edge(self, e):
        """
        Edge removal procedure
        :param e: edge
        """

        coms_to_change = {}

        self.removed += 1
        u = e["u"]
        v = e["v"]

        if self.g.has_edge(u, v):

            # u and v shared communities
            if (
                len(list(self.g.neighbors(u))) > 1
                and len(list(self.g.neighbors(v))) > 1
            ):
                coms = set(self.g.nodes[u]["c_coms"].keys()) & set(
                    self.g.nodes[v]["c_coms"].keys()
                )

                for c in coms:
                    if c not in coms_to_change:
                        cn = set(self.g.neighbors(u)) & set(self.g.neighbors(v))
                        coms_to_change[c] = [u, v]
                        coms_to_change[c].extend(list(cn))
                    else:
                        cn = set(self.g.neighbors(u)) & set(self.g.neighbors(v))
                        coms_to_change[c].extend(list(cn))
                        coms_to_change[c].extend([u, v])
                        ctc = set(coms_to_change[c])
                        coms_to_change[c] = list(ctc)
            else:
                if len(list(self.g.neighbors(u))) < 2:
                    coms_u = copy.copy(list(self.g.nodes[u]["c_coms"].keys()))
                    for cid in coms_u:
                        self.remove_from_community(u, cid)

                if len(list(self.g.neighbors(v))) < 2:
                    coms_v = copy.copy(list(self.g.nodes[v]["c_coms"].keys()))
                    for cid in coms_v:
                        self.remove_from_community(v, cid)

            self.g.remove_edge(u, v)

        # update of shared communities
        self.update_shared_coms(coms_to_change)

    def destroy_community(self, cid):
        nodes = [x for x in self.communities[cid].keys()]
        for n in nodes:
            self.remove_from_community(n, cid)
        self.communities.pop(cid, None)

    def add_to_community(self, node, cid):

        self.g.nodes[node]["c_coms"][cid] = None
        if cid in self.communities:
            self.communities[cid][node] = None
        else:
            self.communities[cid] = {node: None}

    def remove_from_community(self, node, cid):
        if cid in self.g.nodes[node]["c_coms"]:
            self.g.nodes[node]["c_coms"].pop(cid, None)
            if cid in self.communities and node in self.communities[cid]:
                self.communities[cid].pop(node, None)

    def update_shared_coms(self, coms_to_change):
        # update of shared communities
        for c in coms_to_change:
            if c not in self.communities:
                continue

            c_nodes = self.communities[c].keys()

            if len(c_nodes) > 3:

                sub_c = self.g.subgraph(c_nodes)
                c_components = nx.number_connected_components(sub_c)

                # unbroken community
                if c_components == 1:
                    to_mod = sub_c.subgraph(coms_to_change[c])
                    self.modify_after_removal(to_mod, c)

                # broken community: bigger one maintains the id, the others obtain a new one
                else:
                    new_ids = []

                    first = True
                    components = nx.connected_components(sub_c)
                    for com in components:
                        if first:
                            if len(com) < 3:
                                self.destroy_community(c)
                            else:
                                to_mod = list(set(com) & set(coms_to_change[c]))
                                sub_c = self.g.subgraph(to_mod)
                                self.modify_after_removal(sub_c, c)
                            first = False

                        else:
                            if len(com) > 3:
                                # update the memberships: remove the old ones and add the new one
                                to_mod = list(set(com) & set(coms_to_change[c]))
                                sub_c = self.g.subgraph(to_mod)

                                central = self.centrality_test(sub_c).keys()
                                if len(central) >= 3:
                                    actual_id = self.new_community_id
                                    new_ids.append(actual_id)
                                    for n in central:
                                        self.add_to_community(n, actual_id)

                    # splits
                    if len(new_ids) > 0 and self.actual_slice > 0:
                        for n in new_ids:
                            self.mathces.append(
                                (
                                    f"{self.actual_slice-self.obs}_{c}",
                                    f"{self.actual_slice}_{n}",
                                    None,
                                )
                            )
            else:
                self.destroy_community(c)

    def modify_after_removal(self, sub_c, c):
        """
        Maintain the clustering coefficient invariant after the edge removal phase
        :param sub_c: sub-community to evaluate
        :param c: community id
        """
        central = self.centrality_test(sub_c).keys()

        # in case of previous splits, update for the actual nodes
        remove_node = set(self.communities[c].keys()) - set(sub_c.nodes())

        for rm in remove_node:
            self.remove_from_community(rm, c)

        if len(central) < 3:
            self.destroy_community(c)
        else:
            not_central = set(sub_c.nodes()) - set(central)
            for n in not_central:
                self.remove_from_community(n, c)

    def centrality_test(self, subgraph):
        central = {}

        for u in subgraph.nodes():
            if u not in central:
                cflag = False
                neighbors_u = set(self.g.neighbors(u))
                if len(neighbors_u) > 1:
                    for v in neighbors_u:
                        if u > v:
                            if cflag:
                                break
                            else:
                                neighbors_v = set(self.g.neighbors(v))
                                cn = neighbors_v & neighbors_v
                                if len(cn) > 0:
                                    central[u] = None
                                    central[v] = None
                                    for n in cn:
                                        central[n] = None
                                    cflag = True
        return central

    def get_matches(self):
        return self.mathces
