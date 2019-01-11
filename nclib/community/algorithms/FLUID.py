from collections import Counter
import random

__all__ = ['asyn_fluidc']

# Optional to fix the random seed
#random.seed(123)

def asyn_fluidc(G, k, max_iter=100):
    """
    Fluid Communities: A Competitive and Highly Scalable Community Detection Algorithm.
    Args:
        - G: Graph to run the algorithm into.
            + type: networkx.Graph
        - k: Number of communities to search.
            + type: int
        - max_iter: Number of maximum iterations allowed.
            + type: int
    Paper: "Fluid Communities: A Competitive, Scalable and Diverse Community Detection Algorithm"
    Ferran Parés, Dario Garcia-Gasulla, Armand Vilalta, Jonatan Moreno, Eduard Ayguadé, Jesús Labarta, Ulises Cortés, Toyotaro Suzumura
    Return:
        - List of communities, where each community is a list of vertex ID.
          Each vertex ID can be either an int or str.
            + type: list(list(int or str))
    """
    # Initialization
    max_density = 1.0
    vertices = list(G)
    random.shuffle(vertices)
    communities = {n: i for i, n in enumerate(vertices[:k])}
    density = {}
    com_to_numvertices = {}
    for vertex in communities.keys():
        com_to_numvertices[communities[vertex]] = 1
        density[communities[vertex]] = max_density
    # Set up control variables and start iterating
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
		# Loop over all vertices in graph in a random order
        vertices = list(G)
        random.shuffle(vertices)
        for vertex in vertices:
            # Updating rule
            com_counter = Counter()
            # Take into account self vertex community
            try:
                com_counter.update({communities[vertex]: density[communities[vertex]]})
            except KeyError:
                pass
            # Gather neighbour vertex communities
            for v in G[vertex]:
                try:
                    com_counter.update({communities[v]: density[communities[v]]})
                except KeyError:
                    continue
            # Check which is the community with highest density
            new_com = -1
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                best_communities = [com for com, freq in com_counter.items()
                               if (max_freq - freq) < 0.0001]
                # If actual vertex com in best communities, it is preserved
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                # If vertex community changes...
                if new_com == -1:
                    # Set flag of non-convergence
                    cont = True
                    # Randomly chose a new community from candidates
                    new_com = random.choice(best_communities)
                    # Update previous community status
                    try:
                        com_to_numvertices[communities[vertex]] -= 1
                        density[communities[vertex]] = max_density / \
                        com_to_numvertices[communities[vertex]]
                    except KeyError:
                        pass
                    # Update new community status
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = max_density / \
                    com_to_numvertices[communities[vertex]]
        # If maximum iterations reached --> output actual results
        if iter_count > max_iter:
            print ('Exiting by max iterations!')
            break
    # Return results by grouping communities as list of vertices
    return create_list_comm(communities,k)

def create_list_comm(orig_dict,k):
    """
    Inverting Python dictionary keys and values: Many to one --> One to many
    Args:
        - orig_dict: Dictionary desired to invert.
            + type: dict
    Return:
        - Inverted dictionary
            + type: dict
    """

    dict_communities = {}
    for i in range(0,k):
        dict_communities[i] = []

    for node in orig_dict:
        dict_communities[orig_dict[node]].append(node)

    list_communities = []
    for com in dict_communities:
        list_communities.append(dict_communities[com])

    return list_communities


