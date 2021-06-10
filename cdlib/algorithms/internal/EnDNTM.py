import networkx as nx
import random
from networkx.algorithms import community as com
import random
import collections


def __bfs(graph, root, epsilon):
    seen, queue, distance = {root}, collections.deque([root]), 1
    while queue:
        parent = queue.popleft()
        for child in nx.all_neighbors(graph, parent):
            if child not in seen:
                if distance < epsilon:
                    seen.add(child)
                    queue.append(child)
        distance += 1
    return seen


def endntm_evalFuction(graph, clusters_list, etha=0.5):
    mod = com.modularity(graph, clusters_list)
    coverage = nx.algorithms.community.quality.coverage(graph, clusters_list)
    val = (1 - etha) * coverage + etha * mod
    return val


def endntm_find_overlap_cluster(graph, clusters_list, epsilon):

    node_cluster_dic = {}
    cluster_node_dic = {}
    for i, cluster in zip(range(len(clusters_list)), clusters_list):
        for node in cluster:
            node_cluster_dic[node] = i + 1
        cluster_node_dic[i + 1] = set(cluster)

    cluster_no = 0

    # finding Overlapping candidate node
    overlap_cand_dic = {}
    for cluster in clusters_list:
        cluster_no += 1
        for node in cluster:
            neighborhood = __bfs(graph, node, epsilon)
            neighborhood = neighborhood - cluster
            if list(neighborhood) != list():
                overlap_cand_dic[node] = neighborhood

    # filtering overlapping node
    for node in list(overlap_cand_dic.keys()):
        nghbr_clstr_dic = {}
        neighborhood = overlap_cand_dic[node]
        for vertex in list(neighborhood):
            if node_cluster_dic[vertex] not in list(nghbr_clstr_dic):
                nghbr_clstr_dic[node_cluster_dic[vertex]] = {vertex}
            else:
                nghbr_clstr_dic[node_cluster_dic[vertex]].add(vertex)

        # calculating distribution threshold
        disb_threshold = graph.degree(node) / (len(nghbr_clstr_dic) + 1)

        # Generating the overlapping cluster by including overlapping node
        for cid in list(nghbr_clstr_dic):
            if len(nghbr_clstr_dic[cid]) >= disb_threshold:
                cluster_node_dic[cid].add(node)

    overlap_cluster = list(
        cluster_node_dic[cid] for cid in list(cluster_node_dic.keys())
    )
    return overlap_cluster
