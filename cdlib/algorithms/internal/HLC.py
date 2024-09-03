from copy import copy
from heapq import heappush, heappop
from itertools import combinations, chain
from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage

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


class HLC_full(object):
    def __init__(
        self,
        net,
        weight="weight",
        simthr=None,
        hcmethod=None,
        min_edges=None,
        verbose=False,
        dictio=None,
    ):
        self.edge_counter = 0
        self.clusters = self.edge_clustering(
            net,
            weight=weight,
            simthr=simthr,
            hcmethod=hcmethod,
            min_edges=min_edges,
            verbose=verbose,
            dictio=dictio,
        )

    def edge_clustering(
        self,
        net,
        weight=None,
        simthr=None,
        hcmethod=None,
        min_edges=None,
        verbose=False,
        dictio=None,
    ):
        (
            condensed_dist_vector,
            edge_matrix_len,
            edge_list,
        ) = self.get_edge_similarity_matrix(
            net, weight=weight, simthr=simthr, verbose=verbose, dictio=dictio
        )
        clustering = linkage(condensed_dist_vector, hcmethod)
        final_clusters = self.get_clusters_by_partition_density(
            clustering, edge_matrix_len, edge_list, min_edges=min_edges
        )
        return final_clusters

    def get_edge_similarity_matrix(
        self, net, weight=None, simthr=None, verbose=False, dictio=None
    ):
        node_list = list(net.nodes())
        node_list.sort()
        adj = nx.adjacency_matrix(net, weight=weight, nodelist=node_list)
        adj = (
            adj.toarray()
        )  # This line is  needed as a change in csr format from matrix to array sparse. Diference in dot product if this line is removed!

        if weight == None:
            degree = np.sum(adj, axis=1)
            np.fill_diagonal(adj, 1)
        else:
            degree = adj > 0
            degree = np.sum(degree, axis=1)
            weigth_sum = np.sum(adj, axis=1)
            np.fill_diagonal(
                adj, weigth_sum / degree
            )  # Ahn ecuation 4 in supplementary file

        dotproduct = np.dot(
            adj, adj
        )  # This efficiently calculates the vector products needed for tanimoto coefficient (ai and aj)
        adj = None  # Remove matrix in order to save memory
        edge_dict = {}
        data = []
        col_i = []
        col_j = []  # To save tanimoto similarities as sparse data
        cache = (
            {}
        )  # To save tanimoto similarity by pairs and avoid caclulate it when a pair is repeated
        # k_node, i_neigh, j_neigh they are adj matrix indexes AND node names
        edge_similarities = []
        for (
            k_node
        ) in (
            node_list
        ):  # take node as reference and calculate tanimoto coeff for each pair of edges
            neighbor_list = list(
                net.neighbors(k_node)
            )  # take neighbors to get pairs and compare edges
            neighbor_list.sort()
            if len(neighbor_list) < 2:
                continue  # Skip k nodes that has NOT enough neighbours to perform tanimoto
            while len(neighbor_list) > 0:
                i_neigh = neighbor_list.pop()
                for j_neigh in neighbor_list:
                    if weight != None:
                        sim = self.get_tanimoto_index(
                            i_neigh, j_neigh, dotproduct, cache
                        )
                    else:
                        sim = self.get_jaccard_index(
                            i_neigh, j_neigh, dotproduct, degree, cache
                        )
                    if simthr != None and sim < simthr:
                        continue
                    pair = [
                        self.get_edge_id(k_node, i_neigh, edge_dict),
                        self.get_edge_id(k_node, j_neigh, edge_dict),
                    ]
                    pair.sort()
                    ki_edge_id, kj_edge_id = pair
                    data.append(sim)
                    if verbose:
                        a_pair = "_".join(sorted([dictio[k_node], dictio[i_neigh]]))
                        b_pair = "_".join(sorted([dictio[k_node], dictio[j_neigh]]))
                        ids = sorted([a_pair, b_pair])
                        edge_similarities.append(ids + [str(sim)])
                    col_i.append(ki_edge_id)
                    col_j.append(kj_edge_id)

        if verbose:
            with open("edge_scores.txt", "w") as f:
                for e in edge_similarities:
                    f.write("\t".join(e) + "\n")
        condensed_dist_vector, edge_matrix_len = self.get_distance_condensed_vector(
            data, col_i, col_j
        )
        return condensed_dist_vector, edge_matrix_len, list(edge_dict.keys())

    def get_tanimoto_index(self, i_neigh, j_neigh, dotproduct, cache):
        sim, pair = self.get_sim(i_neigh, j_neigh, cache)
        if sim == None:
            a_i2 = dotproduct[i_neigh, i_neigh]
            a_j2 = dotproduct[j_neigh, j_neigh]
            a_ij = dotproduct[i_neigh, j_neigh]
            sim = a_ij / (a_i2 + a_j2 - a_ij)
            cache[pair] = sim
        return sim

    def get_jaccard_index(self, i_neigh, j_neigh, dotproduct, degree, cache):
        sim, pair = self.get_sim(i_neigh, j_neigh, cache)
        if sim == None:
            a_ij = dotproduct[i_neigh, j_neigh]
            sim = a_ij / min(degree[i_neigh], degree[j_neigh])
            cache[pair] = sim
        return sim

    def get_sim(self, i_neigh, j_neigh, cache):
        pair = [i_neigh, j_neigh]
        pair.sort()
        pair = tuple(pair)
        sim = cache.get(pair)
        return sim, pair

    def get_edge_id(self, a, b, e_dict):
        e = [a, b]
        e.sort()
        edge_id = tuple(e)
        e_index = e_dict.get(edge_id)
        if e_index == None:
            e_index = self.edge_counter
            e_dict[edge_id] = e_index
            self.edge_counter += 1
        return e_index

    def get_distance_condensed_vector(self, data, col_i, col_j):
        edge_matrix_len = (
            max([max(col_i), max(col_j)]) + 1
        )  # Values in col_i and col_j are 0 based indexes so we need to add 1 to get the vector size
        upper_triangle_size = (edge_matrix_len**2 - edge_matrix_len) // 2
        condensed_vector = np.ones(upper_triangle_size)
        for idx, sim in enumerate(
            data
        ):  # m * i + j - ((i + 2) * (i + 1)) / 2 from https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            i = col_i[idx]
            j = col_j[idx]
            v = edge_matrix_len * i + j - ((i + 2) * (i + 1)) // 2
            condensed_vector[v] = 1 - sim
        return condensed_vector, edge_matrix_len

    def get_clusters_by_partition_density(
        self, clustering, edge_len, edge_list, min_edges=None
    ):
        tree = {}  # clust id : [member_ids]
        edges_per_cluster = {}  # clust_id : [ edge_tuples ]
        partial_partition_densities = {}  # clust_id : [ cluster_partition_density ]

        counter = edge_len  # this works as cluster id. This is used by the linkage method to tag the intermediate clusters: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        last_dist = None
        constant = 2 / edge_len
        last_cluster_pool = []
        max_pden = -10000000000
        max_cluster_ids = []
        for a_id, b_id, dist, n_members in clustering:
            dist = round(
                dist, 5
            )  # To make equal similar distances that differs in very low values
            if (
                last_dist != None and dist != last_dist
            ):  # We could have several clusters at the same dist, so we group the merge events to clculate the partition density
                p_den = self.get_pden(
                    last_cluster_pool, partial_partition_densities, constant
                )
                if p_den > max_pden:  # check the best partition density
                    max_pden = p_den
                    max_cluster_ids = last_cluster_pool
            a_id = int(
                a_id
            )  # Linkage method returns member ids as float instead of int
            b_id = int(b_id)
            member_list = self.get_member_list(
                counter, a_id, b_id, edge_len, tree
            )  # members that we merge to build the new agglomerative cluster
            nodes, edges = self.get_nodesNedges_per_cluster(member_list, edge_list)
            edges_per_cluster[counter] = edges
            partial_partition_densities[
                counter
            ] = self.get_cluster_partial_partition_density(member_list, nodes)
            last_cluster_pool = [
                cl_id for cl_id in last_cluster_pool if cl_id not in [a_id, b_id]
            ]  # update clusters removin merged cl ids and adding the new cluters ids
            last_cluster_pool.append(counter)
            last_dist = dist
            counter += 1

        p_den = self.get_pden(
            last_cluster_pool, partial_partition_densities, constant
        )  # update clusters removin merged cl ids and adding the new cluters ids
        if (
            p_den > max_pden
        ):  # check the best partition density on the last distance that not was checked
            max_pden = p_den
            max_cluster_ids = last_cluster_pool

        final_clusters = []
        for cluster_id in max_cluster_ids:
            members = edges_per_cluster[cluster_id]
            if min_edges == None or len(members) >= min_edges:
                final_clusters.append(members)
        return final_clusters

    def add_cluster_members(self, cluster, member_id, n_records, tree):
        if (
            member_id < n_records
        ):  # check if member_id is a cluster with only one member that is a original record. That id is less than n_records is the criteria described in https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            cluster.append(member_id)
        else:  # The id represents the merge of two previous clusters. We obtain the member list from the tree and remove it to merge in the it in the new cluster
            cluster.extend(tree.pop(member_id))

    def get_member_list(self, cluster_id, a_id, b_id, edge_len, tree):
        member_list = []
        self.add_cluster_members(
            member_list, a_id, edge_len, tree
        )  # get cluster members from previous a cluster
        self.add_cluster_members(
            member_list, b_id, edge_len, tree
        )  # get cluster members from previous b cluster
        tree[cluster_id] = member_list
        return member_list

    def get_nodesNedges_per_cluster(self, members, edge_list):
        nodes = []
        edges = []
        for member in members:
            edge = edge_list[member]
            edges.append(edge)
            nodes.extend(edge)  # Add edge nodes to node list
        return list(set(nodes)), edges

    def get_cluster_partial_partition_density(self, edges, nodes):
        n = len(nodes)  # node number
        if n == 2:
            partial_partition_density = 0
        else:
            m = len(edges)  # link number
            # partial_partition_density = (m-(n-1))/(n*(n-1)/(2-(n-1))) #Ahn
            partial_partition_density = (m * (m - n + 1)) / ((n - 2) * (n - 1))  # kalinka
        return partial_partition_density

    def get_pden(self, last_cluster_pool, partial_partition_densities, constant):
        partition_den_sum = sum(
            [partial_partition_densities[cl_id] for cl_id in last_cluster_pool]
        )  # Partition density
        p_den = constant * partition_den_sum
        return p_den
