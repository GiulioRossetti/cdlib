import networkx as nx
import copy
import random
import numpy as np

__all__ = ["XMark_benchmark"]


try:
    from scipy.special import zeta as _zeta

    def zeta(x, q, tolerance):
        return _zeta(x, q)


except ImportError:

    def zeta(x, q, tolerance):
        """The Hurwitz zeta function, or the Riemann zeta function of two
        arguments.
        ``x`` must be greater than one and ``q`` must be positive.
        This function repeatedly computes subsequent partial sums until
        convergence, as decided by ``tolerance``.
        """
        z = 0
        z_prev = -float("inf")
        k = 0
        while abs(z - z_prev) > tolerance:
            z_prev = z
            z += 1 / ((k + q) ** x)
            k += 1
        return z


def _zipf_rv_below(gamma, xmin, threshold, seed):
    """Returns a random value chosen from the bounded Zipf distribution.
    Repeatedly draws values from the Zipf distribution until the
    threshold is met, then returns that value.
    """
    result = nx.utils.zipf_rv(gamma, xmin, seed)
    while result > threshold:
        result = nx.utils.zipf_rv(gamma, xmin, seed)
    return result


def _powerlaw_sequence(gamma, low, high, condition, length, max_iters, seed):
    """Returns a list of numbers obeying a constrained power law distribution.
    ``gamma`` and ``low`` are the parameters for the Zipf distribution.
    ``high`` is the maximum allowed value for values draw from the Zipf
    distribution. For more information, see :func:`_zipf_rv_below`.
    ``condition`` and ``length`` are Boolean-valued functions on
    lists. While generating the list, random values are drawn and
    appended to the list until ``length`` is satisfied by the created
    list. Once ``condition`` is satisfied, the sequence generated in
    this way is returned.
    ``max_iters`` indicates the number of times to generate a list
    satisfying ``length``. If the number of iterations exceeds this
    value, :exc:`~networkx.exception.ExceededMaxIterations` is raised.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    for i in range(max_iters):
        seq = []
        while not length(seq):
            seq.append(_zipf_rv_below(gamma, low, high, seed))
        if condition(seq):
            return seq
    raise nx.ExceededMaxIterations("Could not create power law sequence")


def _generate_min_degree(gamma, average_degree, max_degree, tolerance, max_iters):
    """Returns a minimum degree from the given average degree."""
    min_deg_top = max_degree
    min_deg_bot = 1
    min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
    itrs = 0
    mid_avg_deg = 0
    while abs(mid_avg_deg - average_degree) > tolerance:
        if itrs > max_iters:
            raise nx.ExceededMaxIterations("Could not match average_degree")
        mid_avg_deg = 0
        for x in range(int(min_deg_mid), max_degree + 1):
            mid_avg_deg += (x ** (-gamma + 1)) / zeta(gamma, min_deg_mid, tolerance)
        if mid_avg_deg > average_degree:
            min_deg_top = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        else:
            min_deg_bot = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        itrs += 1
    return round(min_deg_mid)


def _assign_random_labels(seq, labels, lab_imb):
    """For Categorical attributes: assign the purest label of each community.
    Return a list of the purest labels."""
    tot_lab_seq = []
    for lb in labels:
        if lb == "auto":
            card_lab = [i for i in range(1, len(seq) + 1)]
        else:
            card_lab = [i for i in range(1, lb + 1)]
        lab_seq = []
        for i in range(len(seq)):
            if lb == "auto":
                lab_seq.append(i + 1)
            else:
                lab_seq.append(random.randrange(1, lb + 1))

        tot_lab_seq.append(lab_seq)
    return tot_lab_seq


def _assign_random_means(seq, labels, lab_imb, mu):
    """For Categorical attributes: assign the desired mean of each community.
    Return a list of the means."""
    tot_lab_seq = []
    for k, lb in enumerate(labels):
        if lb == "auto":
            dist = [i * 10 for i in range(len(seq))]
            multimodal = [
                random.uniform(dist[i] - 2, dist[i] + 2) for i in range(len(seq))
            ]
        else:
            dist = [i * 10 for i in range(lb)]
            multimodal = [random.uniform(dist[i] - 2, dist[i] + 2) for i in range(lb)]
        lab_seq = []
        for i in range(len(seq)):
            if lb == "auto":
                lab_seq.append(multimodal[i])
            else:
                lab_seq.append(
                    random.choice(multimodal)
                )  # will be the mean of community

        tot_lab_seq.append(lab_seq)
    return tot_lab_seq


def _assign_node_memberships(degree_seq):
    overlap_seq = [2 for el in degree_seq]
    return overlap_seq


def _generate_overlapping_communities(
    degree_seq, community_sizes, overlap_seq, mu, max_iters
):
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        for _ in range(overlap_seq[v]):
            c = random.choice(range(len(community_sizes)))
            s = round(degree_seq[v] * (1 - mu))
            # If the community is large enough, add the node to the chosen
            # community. Otherwise, return it to the list of unaffiliated
            # nodes
            if s < community_sizes[c]:
                result[c].add(v)
            else:
                free.append(v)
            # If the community is too big, remove a node from it.
            if len(result[c]) > community_sizes[c]:
                free.append(result[c].pop())


def _generate_communities(
    degree_seq,
    community_sizes,
    lab_coms,
    mu,
    labels,
    noise,
    std,
    max_iters,
    seed,
    type_attr,
):
    """Returns a list of sets, each of which represents a community.
    ``degree_seq`` is the degree sequence that must be met by the
    graph.
    ``community_sizes`` is the community size distribution that must be
    met by the generated list of sets.
    ``mu`` is a float in the interval [0, 1] indicating the fraction of
    intra-community edges incident to each node.
    ``max_iters`` is the number of times to try to add a node to a
    community. This must be greater than the length of
    ``degree_seq``, otherwise this function will always fail. If
    the number of iterations exceeds this value,
    :exc:`~networkx.exception.ExceededMaxIterations` is raised.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    The communities returned by this are sets of integers in the set {0,
    ..., *n* - 1}, where *n* is the length of ``degree_seq``.
    """
    card_lab = []
    for lb in labels:
        if lb == "auto":  # number of labels equal to the number of communities
            card_lab.append([i for i in range(1, len(community_sizes) + 1)])
        else:
            card_lab.append([i for i in range(1, lb + 1)])

    # This assumes the nodes in the graph will be natural numbers.
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    lab_nodes = [list(range(n)) for i in range(len(card_lab))]
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        c = random.choice(range(len(community_sizes)))
        s = round(degree_seq[v] * (1 - mu))
        # If the community is large enough, add the node to the chosen
        # community. Otherwise, return it to the list of unaffiliated nodes.
        if s < community_sizes[c]:
            result[c].add(v)

            if type_attr == "continuous":
                for j, attr in enumerate(lab_coms):
                    lab_nodes[j][v] = float(np.random.normal(attr[c], std, 1))
            # if categorical
            else:
                for j, attr in enumerate(lab_coms):
                    if random.uniform(0, 1) < 1 - noise:
                        lab_nodes[j][v] = attr[c]
                    else:
                        l = random.choice(card_lab[j])
                        lab_nodes[j][v] = l
        else:
            free.append(v)
        # If the community is too big, remove a node from it.
        if len(result[c]) > community_sizes[c]:
            free.append(result[c].pop())
        if not free:
            return lab_nodes, result
    msg = "Could not assign communities"
    raise nx.ExceededMaxIterations(msg)


def XMark_benchmark(
    n,
    tau1,
    tau2,
    mu,
    labels=2,
    std=0.1,
    noise=0,
    lab_imb=0,
    average_degree=None,
    min_degree=None,
    max_degree=None,
    min_community=None,
    max_community=None,
    tol=1.0e-7,
    max_iters=500,
    seed=None,
    type_attr="categorical",
):

    # Perform some basic parameter validation.
    if not tau1 > 1:
        raise nx.NetworkXError("tau1 must be greater than one")
    if not tau2 > 1:
        raise nx.NetworkXError("tau2 must be greater than one")
    if not 0 <= mu <= 1:
        raise nx.NetworkXError("mu must be in the interval [0, 1]")

    # Validate parameters for generating the degree sequence.
    if max_degree is None:
        max_degree = n
    elif not 0 < max_degree <= n:
        raise nx.NetworkXError("max_degree must be in the interval (0, n]")
    if not ((min_degree is None) ^ (average_degree is None)):
        raise nx.NetworkXError(
            "Must assign exactly one of min_degree and" " average_degree"
        )
    if min_degree is None:
        min_degree = _generate_min_degree(
            tau1, average_degree, max_degree, tol, max_iters
        )

    # Generate a degree sequence with a power law distribution.
    low, high = min_degree, max_degree

    def condition(seq):
        return sum(seq) % 2 == 0

    def length(seq):
        return len(seq) >= n

    deg_seq = _powerlaw_sequence(tau1, low, high, condition, length, max_iters, seed)

    # Validate parameters for generating the community size sequence.
    if min_community is None:
        min_community = min(deg_seq)
    if max_community is None:
        max_community = max(deg_seq)

    # Generate a community size sequence with a power law distribution.
    low, high = min_community, max_community

    def condition(seq):
        return sum(seq) == n

    def length(seq):
        return sum(seq) >= n

    comms = _powerlaw_sequence(tau2, low, high, condition, length, max_iters, seed)

    overlap_seq = _assign_node_memberships(deg_seq)

    overlap_comms = copy.copy(comms)
    overlap_deg_seq = copy.copy(deg_seq)
    cycle = [overlap_comms, overlap_deg_seq]
    if sum(overlap_comms) < sum(overlap_seq):
        add_size = abs(sum(overlap_comms) - sum(overlap_seq))
        for new in range(add_size):
            for cy in cycle:
                pick = random.choice(list(range(len(cy))))
            if cy[pick] < np.median(cy):
                cy[pick] += 1
            else:
                cy[random.choice(list(range(len(cy))))] += 1

    if type_attr == "continuous":
        lab_coms = _assign_random_means(comms, labels, lab_imb, mu)
    else:
        lab_coms = _assign_random_labels(comms, labels, lab_imb)

    # Generate the communities based on the given degree sequence and
    # community sizes.
    max_iters *= 10 * n

    lab_nodes, communities = _generate_communities(
        deg_seq, comms, lab_coms, mu, labels, noise, std, max_iters, seed, type_attr
    )

    # Finally, generate the benchmark graph based on the given
    # communities, joining nodes according to the intra- and
    # inter-community degrees.
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, c in enumerate(communities):

        for u in c:
            while G.degree(u) < round(deg_seq[u] * (1 - mu)):
                v = random.choice(list(c))
                G.add_edge(u, v)

            while G.degree(u) < deg_seq[u]:
                v = random.choice(range(n))
                if v not in c:
                    G.add_edge(u, v)

            for j, lab in enumerate(lab_nodes):
                G.nodes[u]["label_" + str(j)] = lab[u]
            G.nodes[u]["community"] = c

    return G
