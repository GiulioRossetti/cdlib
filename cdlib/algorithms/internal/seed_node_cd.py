from __future__ import annotations

from collections import Counter

import networkx as nx


def _seed_score(graph: nx.Graph) -> dict:
    clustering = nx.clustering(graph)
    degree = dict(graph.degree())
    return {node: degree[node] * clustering[node] for node in graph.nodes()}


def seed_node_communities(graph: nx.Graph) -> list[list]:
    g = graph.copy()
    communities: list[list] = []

    while g.number_of_nodes() > 0:
        score = _seed_score(g)
        if not score:
            break
        max_score = max(score.values())
        candidates = [node for node, value in score.items() if value == max_score]
        seed = max(candidates, key=lambda node: g.degree(node))
        degree = dict(g.degree())
        if degree[seed] < 3 or score[seed] < (max_score / 2.0):
            break

        community = [seed, *list(g.neighbors(seed))]
        communities.append(sorted(set(community)))
        g.remove_nodes_from(community)

    remaining = list(g.nodes())
    if remaining and communities:
        original = graph
        unresolved = set(remaining)
        while unresolved:
            next_unresolved = set()
            assignments = []
            for node in unresolved:
                neighbors = set(original.neighbors(node))
                overlaps = [len(neighbors & set(comm)) for comm in communities]
                if not overlaps:
                    next_unresolved.add(node)
                    continue
                max_overlap = max(overlaps)
                if overlaps.count(max_overlap) > 1:
                    next_unresolved.add(node)
                    continue
                assignments.append((node, overlaps.index(max_overlap)))
            if not assignments:
                break
            for node, idx in assignments:
                communities[idx].append(node)
                unresolved.discard(node)
            if next_unresolved == unresolved:
                break
            unresolved = next_unresolved

        for node in unresolved:
            communities.append([node])

    elif remaining:
        communities = [[node] for node in remaining]

    return [sorted(set(comm)) for comm in communities if len(comm) > 0]
