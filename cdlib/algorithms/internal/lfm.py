import random

"""
Lancichinetti, Andrea, Santo Fortunato, and János Kertész. 
"Detecting the overlapping and hierarchical algorithms structure in complex networks"
New Journal of Physics 11.3 (2009): 033015.>>
"""


class Community(object):
    def __init__(self, G, alpha=1.0):
        self.g = G
        self.alpha = alpha
        self.nodes = set()
        self.k_in = 0
        self.k_out = 0

    def add_node(self, node):
        neighbors = set(self.g.neighbors(node))
        node_k_in = len(neighbors & self.nodes)
        node_k_out = len(neighbors) - node_k_in
        self.nodes.add(node)
        self.k_in += 2 * node_k_in
        self.k_out = self.k_out + node_k_out - node_k_in

    def remove_vertex(self, node):
        neighbors = set(self.g.neighbors(node))
        community_nodes = self.nodes
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self.nodes.remove(node)
        self.k_in -= 2 * node_k_in
        self.k_out = self.k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):
        neighbors = set(self.g.neighbors(node))
        old_k_in = self.k_in
        old_k_out = self.k_out
        vertex_k_in = len(neighbors & self.nodes)
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self.alpha
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self.alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        neighbors = set(self.g.neighbors(node))
        new_k_in = self.k_in
        new_k_out = self.k_out
        node_k_in = len(neighbors & self.nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self.alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self.alpha
        return new_fitness - old_fitness

    def recalculate(self):
        for vid in self.nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        neighbors = set()
        for node in self.nodes:
            neighbors.update(set(self.g.neighbors(node)) - self.nodes)
        return neighbors

    def get_fitness(self):
        return float(self.k_in) / ((self.k_in + self.k_out) ** self.alpha)


class LFM_nx(object):
    def __init__(self, G, alpha):
        self.g = G
        self.alpha = alpha

    def execute(self):
        communities = []
        node_not_include = list(self.g.nodes.keys())[:]
        while len(node_not_include) != 0:
            c = Community(self.g, self.alpha)
            # randomly select a seed node
            seed = random.choice(node_not_include)
            c.add_node(seed)

            to_be_examined = c.get_neighbors()
            while to_be_examined:
                # largest fitness to be added
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)
                    m[node] = fitness
                to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]

                # stop condition
                if to_be_add[1] < 0.0:
                    break
                c.add_node(to_be_add[0])

                to_be_remove = c.recalculate()
                while to_be_remove is not None:
                    c.remove_vertex(to_be_remove)
                    to_be_remove = c.recalculate()

                to_be_examined = c.get_neighbors()

            for node in c.nodes:
                if node in node_not_include:
                    node_not_include.remove(node)
            communities.append(list(c.nodes))
        return list(communities)
