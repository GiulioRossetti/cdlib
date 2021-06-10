"""
实现LPANNI [1]。
[1] Meilian L , Zhenlin Z , Zhihe Q , et al.
LPANNI: Overlapping Community Detection Using Label Propagation in Large-Scale Complex Networks[J].
IEEE Transactions on Knowledge and Data Engineering, 2018, PP:1-1.
"""

import networkx as nx


def compute_NI(g: nx.Graph):
    """
    计算网络中每个节点的ni，将结果存储在节点的NI属性中
    :param g: 网络
    :return:None
    """
    max_ni = 0
    min_ni = 2 * g.number_of_nodes()
    for node in g.nodes:
        neighbors = list(g.adj[node].keys())
        triangle_num = 0
        for i in neighbors:
            for j in neighbors:
                if i < j and g.has_edge(i, j):
                    triangle_num += 1
        ni = g.degree[node] + triangle_num
        g.nodes[node]["NI"] = ni
        max_ni = max(max_ni, ni)
        min_ni = min(min_ni, ni)

    for node in g.nodes:
        g.nodes[node]["NI"] = 0.5 + 0.5 * (g.nodes[node]["NI"] - min_ni) / (
            max_ni - min_ni
        )


def compute_sim(g: nx.Graph):
    """
    计算网络中节点u,v的相似度，其中(u,v)∈E
    根据[1]文的考量，将α设置为3.
    (u,v)∈E是因为在LPA算法中只有相邻节点才会互相传播标签
    不存在边的那些节点对计算了也没用
    :param g:网络图
    :return:None
    """
    for edge in g.edges:
        u, v = edge[0], edge[1]
        # p=1
        s = 1.0
        # 开始计算p=2的情况
        for i in list(g.adj[u].keys()):
            if g.has_edge(i, v):
                s += 1 / 2
        # 开始计算p=3的情况
        for i in list(g.adj[u].keys()):
            for j in list(g.adj[i].keys()):
                if g.has_edge(j, v) and (not j == u):
                    s += 1 / 3
        for i in (u, v):
            if g.nodes[i].get("s", -1) == -1:
                g.nodes[i]["s"] = {}
        g.nodes[u]["s"][v] = s
        g.nodes[v]["s"][u] = s

    for edge in g.edges:
        u, v = edge[0], edge[1]
        for i in (u, v):
            if g.nodes[i].get("sim", -1) == -1:
                g.nodes[i]["sim"] = {}
        sim = (
            g.nodes[u]["s"][v]
            / (sum(g.nodes[u]["s"].values()) * sum(g.nodes[v]["s"].values())) ** 0.5
        )
        g.nodes[u]["sim"][v] = sim
        g.nodes[v]["sim"][u] = sim


def compute_NNI(g: nx.Graph):
    """
    计算节点v对节点u的影响力，其中(v,u)∈E
    u节点的属性'NNI'是一个字典，其中包含所有与u相邻的节点V
    对于每一个v∈V，g.nodes[u]['NNI'][v]表示v给u造成的影响力
    :param g:网络图
    :return:None
    """
    for u in g.nodes:
        g.nodes[u]["NNI"] = {}
        sim_max = max(g.nodes[u]["sim"].values())
        for v in list(g.adj[u].keys()):
            g.nodes[u]["NNI"][v] = (
                g.nodes[v]["NI"] * g.nodes[u]["sim"][v] / sim_max
            ) ** 0.5


def LPANNI(g: nx.Graph):
    """
    实现LPANNI对网络进行社区划分
    :param g: 网络图
    :return: None
    """
    compute_NI(g)
    compute_sim(g)
    compute_NNI(g)

    nodes = list(g.nodes)
    v_queue = []
    for node in nodes:
        v_queue.append((node, g.nodes[node]["NI"]))
        g.nodes[node]["L"] = {node: 1}
        g.nodes[node]["dominant"] = 1
        g.nodes[node]["label"] = node
    v_queue = sorted(v_queue, key=lambda v: v[1])
    nodes = [v[0] for v in v_queue]

    # 定义最大迭代次数
    T = 10
    t = 0
    while t < T:
        change = False
        for node in nodes:
            L_Ng = {}
            # 计算邻居们的标签和权重
            for neighbor in list(g.adj[node].keys()):
                c, b = (
                    g.nodes[neighbor]["label"],
                    g.nodes[neighbor]["dominant"] * g.nodes[node]["NNI"][neighbor],
                )
                if L_Ng.get(c, -1) == -1:
                    L_Ng[c] = b
                else:
                    L_Ng[c] += b
            # 除去权重过小的标签
            avg = sum(L_Ng.values()) / len(L_Ng)
            max_dominant = 0
            label = -1
            g.nodes[node]["L"] = {}
            for c in L_Ng.keys():
                if L_Ng[c] >= avg:
                    g.nodes[node]["L"][c] = L_Ng[c]
                    if L_Ng[c] > max_dominant:
                        max_dominant = L_Ng[c]
                        label = c
            sum_dominant = sum(g.nodes[node]["L"].values())

            for c in g.nodes[node]["L"].keys():
                g.nodes[node]["L"][c] /= sum_dominant

            if not g.nodes[node]["label"] == label:
                g.nodes[node]["label"] = label
                change = True
            g.nodes[node]["dominant"] = max_dominant / sum_dominant
        if not change:
            break
        t += 1


class GraphGenerator:
    """
    定义归属系数的阈值
    也就是说，只有一个节点对一个社区的归属系数大于这个阈值时，我们才考虑将这个节点加入这个社区中
    """

    b_threshold = 0.0
    g = nx.Graph()

    def __init__(self, threshold, g):
        self.b_threshold = threshold
        self.g = g

    def get_Overlapping_communities(self) -> list:
        """
        从图中将所有的重叠社区返回
        :return: list
        """
        d = {}
        for node in self.g.nodes:
            L = self.g.nodes[node]["L"]
            for label in L.keys():
                if L[label] > self.b_threshold:
                    # 这个节点属于label 社区
                    if d.get(label, -1) == -1:
                        d[label] = {node}
                    else:
                        d[label].add(node)
        return list(d.values())

    def get_Overlapping_nodes(self) -> set:
        """
        从图中将所有的重叠节点返回
        :return: 所有的重叠节点
        """
        overlapping_nodes = set()
        for node in self.g.nodes:
            L = self.g.nodes[node]["L"]
            count = 0
            for label in L.keys():
                if L[label] > self.b_threshold:
                    count += 1
                    if count >= 2:
                        overlapping_nodes.add(node)
                        break
        return overlapping_nodes
