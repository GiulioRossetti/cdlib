import numpy as np
import copy
from collections import defaultdict


def slpa_nx(G, T, r):
    """
    Speaker-Listener Label Propagation Algorithm (SLPA)
    see http://arxiv.org/abs/1109.5720
    """

    # Stage 1: Initialization
    memory = {i: {i: 1} for i in G.nodes()}

    # Stage 2: Evolution
    for t in range(T):

        listeners_order = list(G.nodes())
        np.random.shuffle(listeners_order)

        for listener in listeners_order:
            speakers = G[listener].keys()
            if len(speakers) == 0:
                continue

            labels = defaultdict(int)

            for j, speaker in enumerate(speakers):
                # Speaker Rule
                total = float(sum(memory[speaker].values()))
                labels[
                    list(memory[speaker].keys())[
                        np.random.multinomial(
                            1, [freq / total for freq in memory[speaker].values()]
                        ).argmax()
                    ]
                ] += 1

            # Listener Rule
            accepted_label = max(labels, key=labels.get)

            # Update listener memory
            if accepted_label in memory[listener]:
                memory[listener][accepted_label] += 1
            else:
                memory[listener][accepted_label] = 1

    # Stage 3:
    for node, mem in memory.items():
        its = copy.copy(list(mem.items()))
        for label, freq in its:
            if freq / float(T + 1) < r:
                del mem[label]

    # Find nodes membership
    communities = {}
    for node, mem in memory.items():
        for label in mem.keys():
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = {node}

    # Remove nested communities
    nested_communities = set()
    keys = list(communities.keys())
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i + 1 :]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                nested_communities.add(label0)
            elif comm0.issuperset(comm1):
                nested_communities.add(label1)

    for comm in nested_communities:
        del communities[comm]

    return [list(c) for c in communities.values()]
