import random

"""
Lancichinetti, Andrea, Santo Fortunato, and János Kertész. 
"Detecting the overlapping and hierarchical algorithms structure in complex networks"
New Journal of Physics 11.3 (2009): 033015.>>
"""


class Community(object):
    def __init__(self, G, alpha=1.0, weight="weight"):
        self.g = G
        self.alpha = alpha
        self.nodes = set()
        self.k_in = 0.0
        self.k_out = 0.0
        self.weight = weight

    def _get_node_strengths(self, node):
        node_k_in = 0.0
        node_k_out = 0.0
        for neighbor, data in self.g[node].items():
            edge_weight = data.get(self.weight, 1.0)
            if neighbor in self.nodes:
                node_k_in += edge_weight
            else:
                node_k_out += edge_weight
        return node_k_in, node_k_out

    def add_node(self, node):
        node_k_in, node_k_out = self._get_node_strengths(node)
        self.nodes.add(node)
        self.k_in += 2 * node_k_in
        self.k_out += node_k_out - node_k_in

    def remove_vertex(self, node):
        # Note: Strengths must be calculated *before* removing the node
        node_k_in, node_k_out = self._get_node_strengths(node)
        self.nodes.remove(node)
        self.k_in -= 2 * node_k_in
        self.k_out -= node_k_out - node_k_in

    def cal_add_fitness(self, node):
        node_k_in, node_k_out = self._get_node_strengths(node)
        new_k_in = self.k_in + 2 * node_k_in
        new_k_out = self.k_out + node_k_out - node_k_in

        if (self.k_in + self.k_out) == 0:
            old_fitness = 0.0
        else:
            old_fitness = self.k_in / (self.k_in + self.k_out) ** self.alpha

        if (new_k_in + new_k_out) == 0:
            new_fitness = 0.0
        else:
            new_fitness = new_k_in / (new_k_in + new_k_out) ** self.alpha

        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        node_k_in, node_k_out = self._get_node_strengths(node)
        
        current_fitness = self.get_fitness()

        new_k_in = self.k_in - 2 * node_k_in
        new_k_out = self.k_out - (node_k_out - node_k_in)

        if (new_k_in + new_k_out) == 0:
            new_fitness = 0.0
        else:
            new_fitness = new_k_in / (new_k_in + new_k_out) ** self.alpha
        
        return current_fitness - new_fitness

    def recalculate(self):
        for vid in self.nodes:
            fitness_change = self.cal_remove_fitness(vid)
            # If fitness increases after removal, it's a good candidate to remove
            if fitness_change < 0.0:
                return vid
        return None

    def get_neighbors(self):
        neighbors = set()
        for node in self.nodes:
            neighbors.update(set(self.g.neighbors(node)) - self.nodes)
        return neighbors

    def get_fitness(self):
        if (self.k_in + self.k_out) == 0:
            return 0.0
        return self.k_in / ((self.k_in + self.k_out) ** self.alpha)


class LFM_nx(object):
    def __init__(self, G, alpha, weight="weight"):
        self.g = G
        self.alpha = alpha
        self.weight = weight

    def execute(self):
        communities = []
        node_not_include = list(self.g.nodes())[:]
        while len(node_not_include) != 0:
            c = Community(self.g, self.alpha, self.weight)
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
