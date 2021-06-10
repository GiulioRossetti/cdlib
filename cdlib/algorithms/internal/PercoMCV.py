#######################################################################################################################
#               PercoMCV code source, implemented by the www.abil.ac.cd team composed by Elie Mayogha,                #
#                       Selain Kasereka, Nathanael Kasoro, Ho Tuong Vinh and Joel Kinganga                            #
#                                                                                                                     #
#       We invite contributor to reuse our code source and cite our paper. We would like to be contacted when         #
#       this code is used, this way will allow us to know the evolution of our proposed algorithm. Injoy              #
#                                   Contact us: contact@abil.ac.cd - University of Kinshasa                           #
#######################################################################################################################

from collections import defaultdict
import networkx as nx

# First step
# computation of k-clique percolation algorithm
# with k = 4


def __k_clique_communities(g, cliques=None):
    if cliques is None:
        cliques = nx.find_cliques(g)
    cliques = [frozenset(c) for c in cliques if len(c) >= 4]

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    perc_graph = nx.Graph()
    perc_graph.add_nodes_from(cliques)
    for clique in cliques:
        for adj_clique in _get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= 3:
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield frozenset.union(*component)


def _get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques


def percoMVC(g):
    # Zachary's Karate club example
    c = list(__k_clique_communities(g))

    m2 = set()
    coms = []
    for com in c:
        coms_list = list(com)
        coms += [coms_list]
        m1 = set(coms_list)
        m2 = m2 | m1

    t = []
    p = 1
    while p <= len(g.nodes):
        t.append(p)
        p += 1
    t = set(t)
    nodn_classes = t - m2

    # Second step
    # Trying to classify unclassified nodes7

    nodn_classes = sorted(nodn_classes)

    for Com in range(len(coms)):
        if len(coms[Com]) > 3:  # Check si la communauté à plus de 3 noeud

            sub = g.subgraph(coms[Com])

            # Calcul de la centralité de vecteur propre
            centrality = nx.eigenvector_centrality(sub)
            vercteur_pr = sorted(
                (round((centrality[node]), 2), node) for node in centrality
            )
            for vect in range(len(vercteur_pr)):
                centralitiness = (
                    vercteur_pr[vect][0] / vercteur_pr[len(vercteur_pr) - 1][0]
                )
                if centralitiness >= 0.99:  # check if the node is 99% central
                    neud_central = vercteur_pr[vect][1]
                    for nod in range(len(nodn_classes)):
                        if g.has_edge(nodn_classes[nod], neud_central):
                            coms[Com] += [nodn_classes[nod]]

    return coms
