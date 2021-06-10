import networkx as nx
import itertools

parent = dict()
rank = dict()


def __make_set(vertice):
    parent[vertice] = vertice
    rank[vertice] = 0


def __find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = __find(parent[vertice])
    return parent[vertice]


def __union(vertice1, vertice2):
    root1 = __find(vertice1)
    root2 = __find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]:
            rank[root2] += 1


def UMSTMO(G):
    G3 = nx.Graph()
    # list of edges in la list T
    T = list(G.edges())

    i = 0
    # joining each edge to its weight
    z2 = []
    while i < len(T):
        # z is number of common neighbors e(i,j)
        z = len(list(nx.common_neighbors(G, T[i][0], T[i][1])))
        # x the number of neighbors i and x1 for j
        x = len(list(G.neighbors(T[i][0])))
        x1 = len(list(G.neighbors(T[i][1])))
        if z > 0:
            # p is the value of jaccard coefficient
            p = z / (x + x1 + z)
            # add weight to the edge
            G[T[i][0]][T[i][1]]["weight"] = p
            # list of edges and their weights
            z2.append([p, T[i][0], T[i][1]])
        else:
            G[T[i][0]][T[i][1]]["weight"] = 0
            z2.append([0, T[i][0], T[i][1]])
        i = i + 1

    for l in G.nodes():
        __make_set(l)

    # sort the list of edges according to their weights
    z2.sort(reverse=True)
    B = []
    eu = []
    i = 0
    # construction of the union of all maximum spanning tree
    for k in z2:
        M = []
        elarge = []
        t8 = 0
        while i < len(z2):
            if z2[i][0] == k[0]:
                elarge.append([z2[i][1], z2[i][2]])
                i = i + 1
                t8 = i
            else:
                i = len(z2)
        i = t8
        for ll in elarge:
            if __find(ll[0]) != __find(ll[1]):
                M.append(ll)
        for kk in M:
            __union(kk[0], kk[1])
        eu.extend(M)
        B.append(elarge)

    tr1 = eu
    for i in tr1:
        G3.add_edge(i[0], i[1])

    l1 = []

    for i in G3.nodes():
        di = G3.degree(i)
        l1.append([di, i])

    l1.sort(reverse=True)
    ll = []
    for k in l1:
        ll.append(k[1])
    re = []
    up = []
    for k in ll:
        b = G3.neighbors(k)
        c = []
        c1 = []

        i = 0
        while i < len(list(b)):
            j = i + 1
            while j < len(list(b)):

                if G.has_edge(b[i], b[j]):
                    c1.extend([b[i], b[j]])
                j = j + 1
            i = i + 1

        c1 = list(set(c1))
        c.append(c1)
        if len(list(b)) == 1:
            if k not in up:
                c.append([b[0]])
                up.append(b[0])
        for k2 in c:
            if len(k2) > 0:
                k2.append(k)
                k2.sort()
                if k2 not in re:
                    re.append(k2)
                for ll2 in k2:
                    up.append(ll2)

    sup = list(set(G.nodes()) - set(up))
    re.sort(key=len, reverse=True)
    res = re

    for k in sup:
        max_val = 0
        ne = G.neighbors(k)
        i = 0
        temp = 0
        while i < len(res):
            aa = len(set(ne).intersection(res[i]))

            if aa >= max_val:
                max_val = aa
                temp = i
                if aa > len(list(ne)) - aa:
                    i = len(res)
            i = i + 1
        if max_val > 0:
            res[temp].append(k)
        else:
            res.append([k])

    r = 0
    while r < len(res):
        j = r + 1
        while j < len(res):
            if len(set(res[r]).intersection(res[j])) / (len(res[j])) >= 0.5:
                res[r] = list(set(res[r]).union(res[j]))
                res.pop(j)
            else:
                j = j + 1
        r = r + 1

    r = 0
    while r < len(res):
        j = r + 1
        while j < len(res):
            if len(set(res[r]).intersection(res[j])) / (len(res[j])) >= 0.33:
                res[r] = list(set(res[r]).union(res[j]))
                res.pop(j)
            else:
                j = j + 1
        r = r + 1

    r = 0
    while r < len(res):
        j = r + 1
        while j < len(res):
            if len(set(res[r]).intersection(res[j])) / (len(res[j])) >= 0.2:
                res[r] = list(set(res[r]).union(res[j]))
                res.pop(j)
            else:
                j = j + 1
        r = r + 1

    return res
