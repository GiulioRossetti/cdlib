import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import defaultdict
import math

__all__ = ["detect_belief_communities"]


def get_degree_vector(g: nx.Graph):
    # compute the vector of node degrees
    return sp.csr_matrix(np.array([g.degree()[i] for i in range(g.number_of_nodes())]))


def init_beliefs(q: int, g: nx.Graph):
    # initialize the beliefs

    n_n = g.number_of_nodes()
    b = sp.random(
        n_n, q, density=1, data_rvs=lambda x: np.random.random(x), format="csr"
    )
    b = sp.csr_matrix(b / b.sum(axis=1))

    return b


def init_messages(q: int, g: nx.Graph):
    # initialize the messages

    n_n = g.number_of_nodes()
    messages = sp.dok_matrix((n_n * n_n, q))

    for i in range(q):
        for node in g.nodes:
            for neighbor in g.neighbors(node):
                messages[node * n_n + neighbor, i] = np.random.random()

    sum = messages.sum(axis=1).transpose().tolist()[0]
    norm = sp.diags(sum, format="csr")
    norm[norm.nonzero()] = 1 / norm[norm.nonzero()]
    messages = messages.tocsr(copy=True)

    return norm.dot(messages)


def compute_theta(beliefs: sp.csr_matrix, degrees: sp.csr_matrix):
    # compute_theta(beliefs, degrees)[t,0] returns theta of community t
    return degrees.dot(beliefs)


def update_matrix(g: nx.Graph):
    # compute the matrix needed to update the messages

    n_n = g.number_of_nodes()
    update_mat = sp.dok_matrix((n_n * n_n, n_n * n_n), dtype=np.int8)
    for node in g.nodes:
        for neighbor in g.neighbors(node):
            for node_neighbor in g.neighbors(node):
                if node_neighbor != neighbor:
                    update_mat[node * n_n + neighbor, node_neighbor * n_n + node] = 1

    update_mat = update_mat.tocsr(copy=True)

    return update_mat


def belief_matrix(g: nx.Graph):
    # comupte the matrix needed to calculate the beliefs

    n_n = g.number_of_nodes()
    update_mat = sp.dok_matrix((n_n, n_n * n_n), dtype=np.int8)
    for node in g.nodes:
        for neighbor in g.neighbors(node):
            update_mat[node, neighbor * n_n + node] = 1

    update_mat = update_mat.tocsr(copy=True)

    return update_mat


def get_external_field_beliefs(theta: sp.csr_matrix, g: nx.Graph, beta: float):
    # compute the external field based on theta for the computation of the beliefs

    m = g.number_of_edges()

    degrees = get_degree_vector(g)

    ext_field = -(beta / (2 * m)) * degrees.transpose().dot(theta)

    return ext_field


def external_field_update_matrix(g: nx.Graph, beta: float, q: int):
    # compute the matrix needed to update the external field

    n_n = g.number_of_nodes()

    m = g.number_of_edges()

    ext_field_update_matrix = sp.dok_matrix((n_n * n_n, n_n))

    for i in range(q):
        for node in g.nodes:
            for neighbor in g.neighbors(node):
                ext_field_update_matrix[(node) * n_n + neighbor, neighbor] = -(
                    (beta * g.degree()[node]) / (2 * m)
                )

    ext_field_update_matrix = ext_field_update_matrix.tocsr()

    return ext_field_update_matrix


def get_external_field(theta: sp.csr_matrix, ext_update: sp.csr_matrix):
    # update the external field for the message update

    n_n = ext_update.shape[1]

    ones_vec = sp.csr_matrix(np.ones((n_n, 1)))

    theta_transformed = ones_vec * theta

    ext_field = ext_update * theta_transformed

    return ext_field


def run_bp_community_detection(
    g: nx.Graph,
    q: int = 2,
    beta: float = 1.012,
    max_it: int = 30,
    eps: float = 0.0001,
    reruns_if_not_conv: int = 20,
):
    # run belief propagation community detection with given parameters

    n_n = g.number_of_nodes()

    # building update_matrix, belief_update_matrix, degree_vector
    update_mat = update_matrix(g)
    belief_update_mat = belief_matrix(g)
    degrees = get_degree_vector(g)

    # building external_field_update_matrix
    ext_field_update_matrix = external_field_update_matrix(g=g, beta=beta, q=q)

    converged = False
    runs = 0
    for i in range(reruns_if_not_conv):

        if converged:
            break

        runs = runs + 1

        # initializing beliefs and messages
        beliefs = init_beliefs(q, g)
        messages = init_messages(q, g)

        # building a vector with ones at certain positions, needed for computation
        ones_vector = sp.dok_matrix((n_n * n_n, q))
        ones_vector[messages.nonzero()] = 1
        ones_vector = ones_vector.tocsr(copy=True)

        # print('Run: ', i+1)

        belief_diff = 1

        iterations = 0
        for j in range(max_it):
            iterations = iterations + 1
            if belief_diff < eps:
                converged = True
                break

            theta = compute_theta(beliefs, degrees)

            # update beliefs with updated messages
            new_messages = ones_vector + (messages * (np.exp(beta) - 1))
            new_messages.data = np.log(new_messages.data)

            updated_beliefs = belief_update_mat.dot(new_messages)
            old_beliefs = beliefs.copy()

            beliefs = updated_beliefs + get_external_field_beliefs(
                theta=theta, g=g, beta=beta
            )
            beliefs.data = np.exp(beliefs.data)

            b_sum = beliefs.sum(axis=1).transpose().tolist()[0]
            b_norm = sp.diags(b_sum, format="csr")
            b_norm.data = 1 / b_norm.data
            beliefs = b_norm.dot(beliefs)
            beliefs = beliefs.tocsr(copy=True)

            # update messages

            external_field = get_external_field(
                theta=theta, ext_update=ext_field_update_matrix
            )

            updated_messages = update_mat.dot(new_messages)

            messages = external_field + updated_messages

            messages.data = np.exp(messages.data)

            m_sum = messages.sum(axis=1).transpose().tolist()[0]
            m_norm = sp.diags(m_sum, format="csr")
            m_norm.data = 1 / m_norm.data
            messages = m_norm.dot(messages)
            messages = messages.tocsr(copy=True)

            belief_diff = np.linalg.norm(beliefs.todense() - old_beliefs)

    detected_communities = [np.argmax(beliefs[i, :]) for i in range(beliefs.shape[0])]

    return beliefs, detected_communities, runs, iterations, converged


def compute_opt_beta(q: int, c: int):
    # compute the optimal value for the parameter beta

    return math.log(q / (math.sqrt(c) - 1) + 1)


def detect_belief_communities(
    g: nx.Graph,
    max_it: int = 100,
    eps: float = 0.0001,
    reruns_if_not_conv: int = 5,
    threshold: float = 0.005,
    q_max: int = 7,
):

    from networkx.algorithms.community.quality import modularity

    # determine number of optimal communities and run community detection for a given network
    # The nodes have to be labeled form 0 to n

    modularity_0 = 0
    modularity_1 = threshold
    q = 1
    c = 2 * g.number_of_edges() / g.number_of_nodes()
    partition = ()

    # run belief propagation community detection with increasing number of communities until the modularity of the
    # detected partition does not increase more then given threshold

    while modularity_1 - modularity_0 >= threshold:
        old_partition = partition
        beta = compute_opt_beta(q, c)
        modularity_0 = modularity_1
        partition = run_bp_community_detection(
            g=g,
            q=q,
            beta=beta,
            max_it=max_it,
            eps=eps,
            reruns_if_not_conv=reruns_if_not_conv,
        )

        modularity_1 = modularity(
            g,
            [
                {i for i in range(len(partition[1])) if partition[1][i] == j}
                for j in set(partition[1])
            ],
        )

        if not partition[4]:
            curr_partition = partition
            partition = old_partition
            modularity_1 = modularity_0
            modularity_0 = modularity_1 - threshold

        if q == 1:
            modularity_0 = modularity_1 - threshold

        if q > q_max:
            break

        q = q + 1

    if len(old_partition) != 0:
        return prepare_coms(g, old_partition[1])
    else:
        return prepare_coms(g, curr_partition[1])


def prepare_coms(g, cms):
    ass = list(zip(g.nodes(), cms))

    coms = defaultdict(list)
    for i in ass:
        coms[i[1]].append(i[0])

    return list(coms.values())
