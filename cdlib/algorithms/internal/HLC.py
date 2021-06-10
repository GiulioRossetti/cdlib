from copy import copy
from heapq import heappush, heappop
from itertools import combinations, chain
from collections import defaultdict


# Hierarchical Link Community
"""
Yong-Yeol Ahn, James P. Bagrow, and Sune Lehmann, 
Link communities reveal multiscale complexity in networks, 
Nature 466, 761 (2010)
"""


def get_sorted_edge(a, b):
    return tuple(sorted([a, b]))


def HLC_read_edge_list_unweighted(g):
    adj_list_dict = defaultdict(set)
    edges = set()

    for e in g.edges():
        ni, nj = e[0], e[1]
        edges.add(get_sorted_edge(ni, nj))
        adj_list_dict[ni].add(nj)
        adj_list_dict[nj].add(ni)
    return dict(adj_list_dict), edges


def HLC_read_edge_list_weighted(g):
    adj = defaultdict(set)
    edges = set()
    ij2wij = {}
    for e in g.edges(data=True):
        ni, nj, wij = e[0], e[1], e[2]["weight"]
        if ni != nj:
            ni, nj = get_sorted_edge(ni, nj)
            edges.add((ni, nj))
            ij2wij[ni, nj] = wij
            adj[ni].add(nj)
            adj[nj].add(ni)
    return dict(adj), edges, ij2wij


class HLC(object):
    @staticmethod
    def get_sorted_pair(a, b):
        return tuple(sorted([a, b]))

    @staticmethod
    def sort_edge_pairs_by_similarity(adj_list_dict):
        def cal_jaccard(left_set, right_set):
            return 1.0 * len(left_set & right_set) / len(left_set | right_set)

        inc_adj_list_dict = dict((n, adj_list_dict[n] | {n}) for n in adj_list_dict)
        min_heap = []
        for vertex in adj_list_dict:
            if len(adj_list_dict[vertex]) > 1:
                for i, j in combinations(adj_list_dict[vertex], 2):
                    edge_pair = HLC.get_sorted_pair(
                        HLC.get_sorted_pair(i, vertex), HLC.get_sorted_pair(j, vertex)
                    )
                    similarity_ratio = cal_jaccard(
                        inc_adj_list_dict[i], inc_adj_list_dict[j]
                    )
                    heappush(min_heap, (1 - similarity_ratio, edge_pair))
        return [heappop(min_heap) for _ in range(len(min_heap))]

    def __init__(self, adj_list_dict, edges):
        self.adj_list_dict = adj_list_dict
        self.edges = edges
        self.density_factor = 2.0 / len(edges)

        self.edge2cid = {}
        self.cid2nodes, self.cid2edges = {}, {}
        self.orig_cid2edge = {}
        self.curr_max_cid = 0
        self.linkage = []  # dendrogram

        def initialize_edges():
            for cid, edge in enumerate(self.edges):
                edge = HLC.get_sorted_pair(*edge)
                self.edge2cid[edge] = cid
                self.cid2edges[cid] = {edge}
                self.orig_cid2edge[cid] = edge
                self.cid2nodes[cid] = set(edge)
            self.curr_max_cid = len(self.edges) - 1

        initialize_edges()  # every edge in its own comm

        self.D = 0.0  # partition density
        self.list_D = [(1.0, 0.0)]  # list of (S_i,D_i) tuples...
        self.best_D = 0.0
        self.best_S = 1.0  # similarity threshold at self.best_D
        self.best_P = None  # best partition, dict: edge -> cid

    def single_linkage(self, threshold=None, w=None, dendro_flag=False):
        def merge_comms(edge1, edge2, S):
            def cal_density(edge_num, vertex_num):
                return (
                    edge_num
                    * (edge_num - vertex_num + 1.0)
                    / ((vertex_num - 2.0) * (vertex_num - 1.0))
                    if vertex_num > 2
                    else 0.0
                )

            if (
                not edge1 or not edge2
            ):  # We'll get (None, None) at the end of clustering
                return
            cid1, cid2 = self.edge2cid[edge1], self.edge2cid[edge2]
            if cid1 == cid2:  # already merged!
                return
            m1, m2 = len(self.cid2edges[cid1]), len(self.cid2edges[cid2])
            n1, n2 = len(self.cid2nodes[cid1]), len(self.cid2nodes[cid2])
            Dc1, Dc2 = cal_density(m1, n1), cal_density(m2, n2)
            if m2 > m1:  # merge smaller into larger
                cid1, cid2 = cid2, cid1

            if dendro_flag:
                self.curr_max_cid += 1
                newcid = self.curr_max_cid
                self.cid2edges[newcid] = self.cid2edges[cid1] | self.cid2edges[cid2]
                self.cid2nodes[newcid] = set()
                for e in chain(self.cid2edges[cid1], self.cid2edges[cid2]):
                    self.cid2nodes[newcid] |= set(e)
                    self.edge2cid[e] = newcid
                del self.cid2edges[cid1], self.cid2nodes[cid1]
                del self.cid2edges[cid2], self.cid2nodes[cid2]
                m, n = len(self.cid2edges[newcid]), len(self.cid2nodes[newcid])

                self.linkage.append((cid1, cid2, S))

            else:
                self.cid2edges[cid1] |= self.cid2edges[cid2]
                for e in self.cid2edges[cid2]:  # move edges,nodes from cid2 to cid1
                    self.cid2nodes[cid1] |= set(e)
                    self.edge2cid[e] = cid1
                del self.cid2edges[cid2], self.cid2nodes[cid2]

                m, n = len(self.cid2edges[cid1]), len(self.cid2nodes[cid1])

            Dc12 = cal_density(m, n)
            self.D += (
                Dc12 - Dc1 - Dc2
            ) * self.density_factor  # update partition density

        if w is None:
            sorted_edge_list = HLC.sort_edge_pairs_by_similarity(self.adj_list_dict)
        else:
            sorted_edge_list = sort_edge_pairs_by_similarity_weighted(
                self.adj_list_dict, w
            )

        prev_similarity = -1
        for oms, eij_eik in chain(sorted_edge_list, [(1.0, (None, None))]):
            cur_similarity = 1 - oms
            if threshold and cur_similarity < threshold:
                break

            if cur_similarity != prev_similarity:  # update list
                if self.D >= self.best_D:  # check PREVIOUS merger, because that's
                    self.best_D = self.D  # the end of the tie
                    self.best_S = cur_similarity
                    self.best_P = copy(self.edge2cid)  # slow...
                self.list_D.append((cur_similarity, self.D))
                prev_similarity = cur_similarity
            merge_comms(eij_eik[0], eij_eik[1], cur_similarity)

        if threshold is not None:
            return self.edge2cid, self.D
        if dendro_flag:
            return (
                self.best_P,
                self.best_S,
                self.best_D,
                self.list_D,
                self.orig_cid2edge,
                self.linkage,
            )
        else:
            return self.best_P, self.best_S, self.best_D, self.list_D


def sort_edge_pairs_by_similarity_weighted(adj_list_dict, edge_weight_dict):
    inc_adj_list_dict = dict((n, adj_list_dict[n] | {n}) for n in adj_list_dict)

    def cal_jaccard(intersect_val, left_val, right_val):
        return intersect_val / (left_val + right_val - intersect_val)

    Aij = copy(edge_weight_dict)
    n2a_sqrd = {}
    for vertex in adj_list_dict:
        Aij[vertex, vertex] = float(
            sum(
                edge_weight_dict[HLC.get_sorted_pair(vertex, i)]
                for i in adj_list_dict[vertex]
            )
        )
        Aij[vertex, vertex] /= len(adj_list_dict[vertex])
        n2a_sqrd[vertex] = sum(
            Aij[HLC.get_sorted_pair(vertex, i)] ** 2 for i in inc_adj_list_dict[vertex]
        )  # includes (n,n)!

    min_heap = []
    for vertex in adj_list_dict:
        if len(adj_list_dict[vertex]) > 1:
            for i, j in combinations(adj_list_dict[vertex], 2):
                edge_pair = HLC.get_sorted_pair(
                    HLC.get_sorted_pair(i, vertex), HLC.get_sorted_pair(j, vertex)
                )
                ai_dot_aj = float(
                    sum(
                        Aij[HLC.get_sorted_pair(i, x)] * Aij[HLC.get_sorted_pair(j, x)]
                        for x in inc_adj_list_dict[i] & inc_adj_list_dict[j]
                    )
                )
                similarity_ratio = cal_jaccard(ai_dot_aj, n2a_sqrd[i], n2a_sqrd[j])
                heappush(min_heap, (1 - similarity_ratio, edge_pair))
    return [heappop(min_heap) for _ in range(len(min_heap))]
