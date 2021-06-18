import numpy as np


# Step 1: normalize the decision matrix
def norm(x, y):
    """normalization function; x is the array with the
    performances and y is the normalization method.
    For vector input 'v' and for linear 'l'
    """

    if y == "v":
        k = np.array(np.cumsum(x ** 2, 0))
        z = np.array(
            [
                [
                    round(x[i, j] / np.sqrt(k[x.shape[0] - 1, j]), 3)
                    for j in range(x.shape[1])
                ]
                for i in range(x.shape[0])
            ]
        )
        return z
    else:
        yy = []
        k = []
        for i in range(x.shape[1]):
            yy.append(np.amax(x[:, i : i + 1]))
            k = np.array(yy)

        z = np.array(
            [
                [round(x[i, j] / k[j], 3) for j in range(x.shape[1])]
                for i in range(x.shape[0])
            ]
        )
        return z


# Step 2: find the weighted normalized decision matrix
def mul_w(r, t):
    """multiplication of each evaluation by the associate
    weight; r stands for the weights matrix and t for
    the normalized matrix resulting from norm()
    """
    z = np.array(
        [
            [round(t[i, j] * r[j], 3) for j in range(t.shape[1])]
            for i in range(t.shape[0])
        ]
    )
    return z


# Step 3: calculate the ideal and anti-ideal solutions
def zenith_nadir(x, y):
    """zenith and nadir virtual action function; x is the
    weighted normalized decision matrix and y is the
    action used. For min/max input 'm' and for absolute
    input enter 'a'
    """
    if y == "m":
        bb = []
        cc = []
        for i in range(x.shape[1]):
            bb.append(np.amax(x[:, i : i + 1]))
            b = np.array(bb)
            cc.append(np.amin(x[:, i : i + 1]))
            c = np.array(cc)
        return b, c
    else:
        b = np.ones(x.shape[1])
        c = np.zeros(x.shape[1])
        return b, c


# Step 4: determine the distance to the ideal and anti-ideal
# solutions
def distance(x, y, z):
    """calculate the distances to the ideal solution (di+)
    and the anti-ideal solution (di-); x is the result
    of mul_w() and y, z the results of zenith_nadir()
    """
    a = np.array(
        [[(x[i, j] - y[j]) ** 2 for j in range(x.shape[1])] for i in range(x.shape[0])]
    )
    b = np.array(
        [[(x[i, j] - z[j]) ** 2 for j in range(x.shape[1])] for i in range(x.shape[0])]
    )
    return np.sqrt(sum(a, 1)), np.sqrt(sum(b, 1))


# TOPSIS method: it calls the other functions and includes
# step 5
def topsis(matrix, weight, norm_m, id_sol):
    """matrix is the initial decision matrix, weight is
    the weights matrix, norm_m is the normalization
    method, id_sol is the action used, and pl is 'y'
    for plotting the results or any other string for
    not
    """
    z = mul_w(weight, norm(matrix, norm_m))
    s, f = zenith_nadir(z, id_sol)
    p, n = distance(z, s, f)
    final_s = np.array([n[i] / (p[i] + n[i]) for i in range(p.shape[0])])

    return final_s
