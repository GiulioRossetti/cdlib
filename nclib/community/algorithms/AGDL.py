from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx


#KNN
def knn(k, X):
    neigh = NearestNeighbors(k+1,metric='euclidean', n_jobs=-1).fit(X)
    kneighbors = neigh.kneighbors(X,k+1,)
    distance = np.array(kneighbors[0][:,1:])
    indices = np.array(kneighbors[1][:,1:])
    return distance, indices
###############################################################################################################################################
#KNNGRAPH
def w_matrix(data, distance, indices, Ks, a=10):

    n = len(data)

    weight_matrix = np.zeros([n, n])

    sigma2 = (a / n / Ks) * np.linalg.norm(distance)**2

    if Ks==1:
        for i in range(n):
            #for j in range(n):
            j=indices[i][0]
            weight_matrix[i][j] = np.exp(-1 * (np.linalg.norm(data[i]- data[j])** 2) / sigma2)
    else:
        for i in range(n):
            #for j in range(n):
            for j in indices[i]:
                weight_matrix[i][j] = np.exp(-1 * (np.linalg.norm(data[i]- data[j])** 2) / sigma2)

    return weight_matrix, sigma2

def k0graph(X,distance,indices, a=10):

    W, sigma2 = w_matrix(X, distance, indices, 1, a)

    Vc = []
    n = len(W)

    x,y = np.where(W>0)

    for i in range(len(x)):
        x_index, y_index = -1,-1
        for k in range(len(Vc)):
            if y[i] in Vc[k]:
                y_index = k
            if x[i] in Vc[k]:
                x_index = k


        if x_index < 0 and y_index < 0:
            Vc.append([x[i],y[i]])
        elif x_index >= 0 and y_index < 0:
            Vc[x_index].append(y[i])
        elif x_index < 0 and y_index >=0:
            Vc[y_index].append(x[i])
        elif x_index == y_index:
            continue
        else:
            Vc[x_index].extend(Vc[y_index])
            del Vc[y_index]


    return Vc
###########################################################################################################################################

def getAffinityMaxtrix(Vc,W):
    nc = len(Vc)

    affinity = np.zeros([nc,nc])

    for i in range(nc):
        for j in range(i+1,nc):
            ij = np.ix_(Vc[i],Vc[j])
            ji = np.ix_(Vc[j],Vc[i])

            W_ij, W_ji = W[ij], W[ji]
            Ci, Cj = len(Vc[i]),len(Vc[j])

            ones_i = np.ones((Ci,1))
            ones_j = np.ones((Cj,1))
            affinity[i][j] = (1/Ci**2)*np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i) + (1/Cj**2)*np.transpose(ones_j).dot(W_ji).dot(W_ij).dot(ones_j)
            affinity[j][i] = affinity[i][j]
    return affinity

def getAffinityBtwCluster(C1, C2, W):


    ij = np.ix_(C1, C2)
    ji = np.ix_(C2, C1)

    W_ij, W_ji = W[ij], W[ji]
    Ci, Cj = len(C1), len(C2)

    ones_i = np.ones((Ci, 1))
    ones_j = np.ones((Cj, 1))
    affinity = (1/Ci**2)*np.transpose(ones_i).dot(W_ij).dot(W_ji).dot(ones_i) + (1/Cj**2)*np.transpose(ones_j).dot(W_ji).dot(W_ij).dot(ones_j)
    return affinity[0,0]

def getNeighbor(Vc, Kc, W):
    Ns, As = [], []
    A = getAffinityMaxtrix(Vc, W)

    for i in range(len(A)):
        As.append([x for x in sorted(list(A[i]))[-1 * Kc:] if x > 0])
        n = len(As[i])
        if n==0:
            Ns.append([])
        else:
            Ns.append(A[i].argsort()[-1*n:].tolist())

    return Ns,As


def Agdl(g, targetClusterNum, Ks, Kc, a):

    data = nx.to_numpy_matrix(g)
    distance, indices = knn(Ks, data)

    cluster = k0graph(data, distance, indices,a)
    length = 0
    for i in range(len(cluster)):
        length += len(cluster[i])

    W,sigma = w_matrix(data, distance, indices, Ks,a)
    neighborSet, affinitySet = getNeighbor(cluster, Kc, W)
    currentClusterNum = len(cluster)

    while currentClusterNum > targetClusterNum:

        max_affinity = 0
        max_index1 = 0
        max_index2 = 0
        for i in range(len(neighborSet)):
            if len(neighborSet[i])==0:
                continue
            aff = max(affinitySet[i])
            if aff > max_affinity:
                j = int(neighborSet[i][affinitySet[i].index(aff)])
                max_affinity = aff

                if i < j:
                    max_index1 = i
                    max_index2 = j
                else:
                    max_index1 = j
                    max_index2 = i

        if max_index1 == max_index2:
            break


        #merge two cluster
        cluster[max_index1].extend(cluster[max_index2])
        cluster[max_index2] = []


        if max_index2 in neighborSet[max_index1]:
            p = neighborSet[max_index1].index(max_index2)
            del neighborSet[max_index1][p]
        if max_index1 in neighborSet[max_index2]:
            p = neighborSet[max_index2].index(max_index1)
            del neighborSet[max_index2][p]

        for i in range(len(neighborSet)):
            if i==max_index1 or i==max_index2:
                continue

            if max_index1 in neighborSet[i]:
                aff_update = getAffinityBtwCluster(cluster[i], cluster[max_index1], W)

                p = neighborSet[i].index(max_index1)
                affinitySet[i][p] = aff_update # fix the affinity values

            if max_index2 in neighborSet[i]:

                p = neighborSet[i].index(max_index2)
                del neighborSet[i][p]
                del affinitySet[i][p]

                if max_index1 not in neighborSet[i]:
                    aff_update = getAffinityBtwCluster(cluster[i], cluster[max_index1], W)
                    neighborSet[i].append(max_index1)
                    affinitySet[i].append(aff_update)

        neighborSet[max_index1].extend(neighborSet[max_index2])
        neighborSet[max_index1] = list(set(neighborSet[max_index1]))

        affinitySet[max_index1] = []

        neighborSet[max_index2] = []
        affinitySet[max_index2] = []

        # Fine the Kc-nearest clusters for Cab

        for i in range(len(neighborSet[max_index1])):
            target_index = neighborSet[max_index1][i]
            newAffinity = getAffinityBtwCluster(cluster[target_index], cluster[max_index1], W)
            affinitySet[max_index1].append(newAffinity)


        if len(affinitySet[max_index1]) > Kc:
            index = np.argsort(affinitySet[max_index1])
            new_neighbor = []
            new_affinity = []
            for j in range(Kc):
                new_neighbor.append(neighborSet[max_index1][index[-1*j]])
                new_affinity.append(affinitySet[max_index1][index[-1*j]])

            neighborSet[max_index1] = new_neighbor
            affinitySet[max_index1] = new_affinity

        currentClusterNum = currentClusterNum - 1


    reduced_cluster = []
    for i in range(len(cluster)):
        if len(cluster[i]) != 0:
            reduced_cluster.append(cluster[i])
    length = 0
    for i in range(len(reduced_cluster)):
        length += len(reduced_cluster[i])

    return reduced_cluster


g = nx.karate_club_graph()
coms = Agdl(g,3,2,2)