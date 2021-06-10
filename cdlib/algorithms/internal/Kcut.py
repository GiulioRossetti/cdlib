import numpy as np
from sklearn.cluster import KMeans
import sklearn.preprocessing as pre
import networkx as nx
from collections import defaultdict


def Kcut(data, kmax):
    """
    此算法执行的是Kcut算法  Ruan J , Zhang W . An Efficient Spectral Algorithm for Network Community Discovery and
    Its Applications to Biological and Social Networks[C]// Data Mining, 2007. ICDM 2007. Seventh IEEE International Conference on. IEEE, 2007.
    data:图的邻接矩阵  kmax:k-menas算法k的最大值
    """
    Q = -100.0  # 模块度值
    # 计算社团的所有边
    c = 0.0  # 所有的边数
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if data[i, j] == 1:
                c = c + 1
    c = c / 2
    for k in range(2, 2 * kmax + 1):
        Qk = 0  # 当前社团的模块度
        clusters = []  # 当前的聚类结果,是一个n*1列向量，每一行代表一个样本的类标签
        # 计算图的度
        W = np.zeros((data.shape[0], data.shape[0]))
        D = np.zeros([data.shape[0], data.shape[0]])
        # 计算对角度矩阵
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                W[i, j] = np.linalg.norm(data[i, :] - data[i, :], 2)
            value0 = sum(W[i, :]) / data.shape[0]
            for j in range(data.shape[0]):
                if W[i, j] < value0:
                    W[i, j] = 0
        for i in range(data.shape[0]):
            D[i, i] = np.sum(W[i, :])
        # 计算拉普拉斯矩阵
        L = D - W
        value, vector = np.linalg.eig(L)  # 对拉普拉斯矩阵进行特征值分解
        seq = np.argsort(value)  # 计算特征值从小到大排列的序号
        seq = seq[0:k]
        vector = vector[:, seq]
        vector = np.array(vector)
        for i in range(vector.shape[0]):  # 将向量的单位长度进行单位化
            vector[i, :] = vector[i, :] / (np.linalg.norm(vector[i, :], 2) + 0.00001)
        estimator = KMeans(n_clusters=k)  # k-means聚类
        vector = np.real(vector)
        vector = pre.minmax_scale(vector)
        estimator.fit(vector)
        clusters = estimator.labels_  # 获得最终的聚类标签,类标签为一个行向量
        """计算社团划分结果的模块度"""
        for i in range(k + 1):
            # 获得第i社团的节点
            e = 0  # 两个顶点均在第i个社团
            a = 0  # 至少一个顶点在第i个社团
            node = np.argwhere(clusters == i)
            node = node[:, 0]
            node = np.array(node)
            for m in range(data.shape[0]):
                for n in range(data.shape[0]):
                    if data[m, n] == 1:
                        temp1 = []
                        temp2 = []
                        temp1 = np.argwhere(node == m)
                        temp2 = np.argwhere(node == n)
                        if len(temp1) != 0:
                            a = a + 1
                            if len(temp2) != 0:
                                e = e + 1
            a = a / 2
            e = e / 2
            Qk = Qk + e / c - (a / c) * (a / c)
        if Qk > Q:  # 当前的模块度优于之前的模块度
            Q = Qk
            result = clusters  # 保存社团划分结果

    coms = defaultdict(list)
    for n, c in enumerate(result):
        coms[c].append(n)

    return list(coms.values())


def kcut_exec(g, kmax):
    nodes = list(g.nodes())
    data = nx.to_numpy_array(g, nodelist=nodes)
    result = Kcut(data, kmax)

    communities = []
    for c in result:
        cms = []
        for n in c:
            cms.append(nodes[n])
        communities.append(cms)

    return communities
