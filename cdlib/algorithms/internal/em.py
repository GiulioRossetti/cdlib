import random
import collections

"""
Newman, Mark EJ, and Elizabeth A. Leicht. 
"Mixture algorithms and exploratory analysis in networks." 
Proceedings of the National Academy of Sciences 104.23 (2007): 9564-9569.
"""


class EM_nx(object):
    def __init__(self, g, k, max_iter=100):
        self.g = g
        self.n = len(self.g.nodes)
        self.k = k
        self.pi = []
        self.theta = []
        self.max_iter = max_iter

    def e_step(self, q):
        for i in range(self.n):
            q.append([])
            norm = 0.0
            for g in range(self.k):
                x = self.pi[g]
                for j in self.g.neighbors(i):
                    x *= self.theta[g][j]
                q[i].append(x)
                norm += x
            for g in range(self.k):
                q[i][g] /= norm

    def m_step(self, q):
        for g in range(self.k):
            sum1 = 0.0
            sum3 = 0.0
            for i in range(self.n):
                sum1 += q[i][g]
                sum2 = 0.0
                for j in self.g.neighbors(i):
                    sum2 += q[j][g]
                self.theta[g][i] = sum2  # update theta
                sum3 += q[i][g] * len(list(self.g.neighbors(i)))
            self.pi[g] = sum1 / self.n  # update pi
            for i in range(self.n):
                self.theta[g][i] /= sum3  # norm

    def execute(self):
        # initial parameters
        X = [1.0 + random.random() for i in range(self.k)]
        norm = sum(X)
        self.pi = [x / norm for x in X]

        for i in range(self.k):
            Y = [1.0 + random.random() for j in range(self.n)]
            norm = sum(Y)
            self.theta.append([y / norm for y in Y])

        q_old = []
        for iter_time in range(self.max_iter):
            q = []
            # E-step
            self.e_step(q)
            # M-step
            self.m_step(q)

            if iter_time != 0:
                deltasq = 0.0
                for i in range(self.n):
                    for g in range(self.k):
                        deltasq += (q_old[i][g] - q[i][g]) ** 2
                # print "delta: ", deltasq
                if deltasq < 0.05:
                    # print "iter_time: ", iter_time
                    break

            q_old = []
            for i in range(self.n):
                q_old.append([])
                for g in range(self.k):
                    q_old[i].append(q[i][g])

        communities = collections.defaultdict(lambda: set())
        for i in range(self.n):
            c_id = 0
            cur_max = q[i][0]
            for j in range(1, self.k):
                if q[i][j] > cur_max:
                    cur_max = q[i][j]
                    c_id = j
            communities[c_id].add(i)
        return list(communities.values())
