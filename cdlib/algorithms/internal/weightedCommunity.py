__authors__ = [
    "Marco Cardia <cardiamc@gmail.com>",
    "Francesco Sabiu <fsabiu@gmail.com>",
]


class weightedCommunity(object):
    def __init__(
        self,
        G,
        min_bel_degree,
        threshold_bel_degree,
        weightName="weight",
        save=False,
        outfile_name="weighted_communities.txt",
    ):
        """
        Constructor

        :param G: an igraph.Graph object
        :param min_bel_degree: the tolerance, in terms of beloging degree, required in order to add a node in a community
        :param threshold_bel_degree: the tolerance, in terms of beloging degree, required in order to add a node in a 'NLU' community
        :param weightName: Name of the attribute containing the weights
        """
        self.G = G
        self.N = len(self.G.vs)
        self.L = len(self.G.es)
        self.min_bel_degree = min_bel_degree
        self.threshold_bel_degree = threshold_bel_degree
        self.weightName = weightName
        self.save = save
        self.outfile_name = outfile_name
        self.communities = []
        self.strengths = []
        self.T = 0

        # Labels
        self.G.vs["label"] = ["F"] * self.N

    # Strength of a node
    def strength(self, node):
        strength = 0
        for j in self.G.neighbors(node):
            strength += self.G.es[self.G.get_eid(node, j)][self.weightName]
        return strength

    # Belonging degree
    def belonging_degree(self, node, community):
        strength = 0
        strength = sum(
            [
                self.G.es[self.G.get_eid(node, i)][self.weightName]
                for i in self.G.neighbors(node)
                if i in community
            ]
        )
        return strength / self.strength(node)

    # Modularity of a graph
    def modularity(self):
        import itertools
        import random

        Q_new = 0
        for c in self.communities:
            pair_components = list(itertools.combinations(c, 2))
            pair_components = random.sample(
                pair_components, 1000
            )  # Reducing execution time!
            for pair_c in pair_components:
                e_id = self.G.get_eid(pair_c[0], pair_c[1], error=False)
                weight = self.G.es[e_id][self.weightName] if e_id != -1 else 0
                Q_new += (
                    self.belonging_degree(pair_c[0], c)
                    * self.belonging_degree(pair_c[1], c)
                    * (
                        weight
                        - self.strength(pair_c[0])
                        * self.strength(pair_c[1])
                        / (2 * self.L)
                    )
                )

        Q_new = Q_new / (2 * self.L)

        return Q_new

    def allStrengths(self):
        for node in self.G.vs:
            self.strengths.append(self.strength(node))

        # Updating graph
        self.G.vs["strength"] = self.strengths

    def strongestNotLabeled(self):
        # Getting indices labeled with 'F'
        indices = [i for i in range(len(self.G.vs)) if self.G.vs[i]["label"] == "F"]

        # Getting strengths for such items
        ss = {}
        for i in indices:
            ss[self.strengths[i]] = i

        # Returning (one of) the strongest node index(es)
        return ss[max(ss.keys())]

    def nodesRemotion(self, c, min_bel_degree):
        old_c = 0
        while old_c != len(c):
            old_c = 0
            c_list = [el for el in c]
            for node in c_list:
                old_c = len(c_list)
                if self.belonging_degree(node, c) < min_bel_degree:
                    try:
                        c.remove(node)
                    except:
                        pass
        return c

    def initialCommunityDetection(self):
        c = set()
        # Getting strongest labeled with 'F'
        strongest = self.strongestNotLabeled()

        # Adding strongest to community
        c.add(strongest)
        self.G.vs[strongest]["label"] = "T"

        # Adding strongest's neighbors
        for neighbor in self.G.neighbors(strongest):
            c.add(neighbor)

        # Until convergence
        # if belonging degree of the community nodes is lower than min_bel_degree, we remove them.
        c = self.nodesRemotion(c, self.min_bel_degree)

        # If commuty is empty, add strongest to it
        if len(c) == 0:
            c.add(strongest)

        return c

    def find_initial_community_neighbors(self, c):
        ### Finding initial community neighbors
        c_neighbors = set()

        for node in c:
            for nb in self.G.neighbors(node):
                c_neighbors.add(nb)
        return c_neighbors

    def define_nu_nlu(self, c, c_neighbors):
        """### Find Nu (b > 0.5) and Nlu (0.4 < b < 0.5) starting from the set of neighbors"""
        nu = set()
        nlu = set()

        for nb in c_neighbors:
            # Nu set: neighbor with belonging degree >= min_bel_degree
            if self.belonging_degree(nb, c) >= self.min_bel_degree:
                nu.add(nb)
            # Nlu set neighbor with threshold < belonging degree < min_bel_degree
            elif self.belonging_degree(nb, c) > self.threshold_bel_degree:
                nlu.add(nb)

        return nu, nlu

    def add_nlu_to_community(self, c, nlu):
        while len(nlu) > 0:
            Q = self.modularity()

            candidate = nlu.pop()
            c.add(candidate)
            Q_new = self.modularity()
            if Q_new > Q:
                Q = Q_new
            else:
                c.remove(candidate)
        return c

    def expandCommunity(self, c):
        """##  Expanding the community"""
        c_old = 0
        nu, nlu = set(), set()

        while c_old != len(c):  # While it adds nodes belonging to nlu in the community
            while c_old != len(
                c
            ):  # While it adds nodes belonging to nu in the community
                c_neighbors = self.find_initial_community_neighbors(c)

                # Computing nu and nlu
                nu, nlu = self.define_nu_nlu(c, c_neighbors)

                # Add Nu nodes to initial community and return to neighbors search
                c_old = len(c)
                c = c.union(nu)

                # Asserting size
                if len(c) != len(c) + len(nu) - len(c.intersection(nu)):
                    print("Incompatible sizes")
                    return -1

            # Add the nodes of Nlu to the community only if their presence increases the community modularity Q0
            c_old = len(c)
            c = self.add_nlu_to_community(c, nlu)

        # Mark the community nodes with 'T' label
        self.G.vs["label"] = [
            v["label"] if not v.index in c else "T" for v in self.G.vs
        ]

        # Updating T label counter
        self.T = len([v for v in self.G.vs["label"] if v == "T"])

        return c

    def getCommunities(self):
        return self.communities

    def computeCommunity(self):
        if len(self.communities) == 0:  # Just the 1st time
            # Strenghts
            self.allStrengths()

        # Initial community detection
        c = self.initialCommunityDetection()

        # Community expansion
        c = self.expandCommunity(c)

        self.communities.append(c)

    def computeCommunities(self):
        while self.N != self.T:
            self.computeCommunity()

        # output communities
        if self.save and len(self.getCommunities()) > 0:
            out_file_com = open(self.outfile_name, "w")

            for cid, c in enumerate(self.getCommunities()):
                out_file_com.write("%d\t%s\n" % (cid, str(c)))

            out_file_com.flush()
            out_file_com.close()

        return self.getCommunities()
