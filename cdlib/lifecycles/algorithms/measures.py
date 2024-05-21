from collections import Counter
from math import log, e
from typing import Union, Tuple

import numpy as np
import cdlib.lifecycles.algorithms.event_analysis as ea

__all__ = [
    "_normalized_shannon_entropy",
    "facet_unicity",
    "facet_identity",
    "facet_outflow",
    "facet_metadata",
    "purity",
    "event_typicality",
    "stability",
]


def _entropy(labels: list, base=2) -> float:
    """
    computes the Shannon entropy of a list of labels

    :param labels: the list of labels
    :param base: the base of the logarithm
    :return: the set entropy
    """
    n = len(labels)
    counter = Counter(labels)
    probabilities = [count / n for count in counter.values()]

    return -sum(p * log(p, base) for p in probabilities)


def _normalized_shannon_entropy(labels, base=2):
    """
    the normalized Shannon entropy is the Shannon entropy divided by the maximum possible entropy
    (logb(n) where n is the number of labels)

    :param labels: the list of labels
    :param base: the base of the logarithm
    :return: the normalized Shannon entropy
    """

    # Example of problem: 40,40,1 compared with 40,40

    base = e if base is None else base

    ent = _entropy(labels, base)
    max_ent = log(len(list(set(labels))), base)
    # print(ent, max_ent, labels)

    normalized_entropy = ent / max_ent
    return normalized_entropy


def _max_second_difference(labels):
    """
    Function computing the difference between the most frequent attribute value and the
    second most frequent attribute value

    Args:
        labels (_type_): the list of labels

    Returns:
        _type_: _description_
    """
    if len(set(labels)) < 2:
        return 1
    n = len(labels)
    counter = Counter(labels)
    probabilities = [count / n for count in counter.values()]
    max_val = max(probabilities)
    second_largest = sorted(probabilities)[-2]
    return max_val - second_largest


def facet_unicity(labels: list) -> float:
    """
    the unicity facet quantifies the extent to which a target set comes from one (=1) or multiple (->0) flows.
    It is computed as the difference between the largest and the second largest group size
    If the target set is composed of a single group, the unicity facet is 1

    :param labels: the list of group labels
    :return: the unicity facet
    """

    if len(set(labels)) < 2:
        return 1
    else:
        # return gini_index(labels)
        # return normalized_shannon_entropy(labels)
        # return berger_parker_index(labels)
        return _max_second_difference(labels)


def facet_identity(target: set, reference: list) -> float:
    """
    the identity facet quantifies how much the identity of the target set is shared with the reference groups.


    :param target: the target set
    :param reference: the reference sets
    :return: the contribution factor
    """
    w = 0
    persistent = 0
    for r in reference:
        flow = r.intersection(target)
        w += len(flow) * len(flow) / len(r)
        # print(len(flow),len(r),len(target),w)
        persistent += len(flow)
    # denominator=len(target)
    if persistent == 0:
        return 0.0
    denominator = persistent
    w = w / denominator
    return w


def facet_outflow(target: set, reference: list) -> float:
    """
    the outflow facet is the ratio of the number of elements
    in the target set that are not in any of the reference sets

    :param target: the target set
    :param reference: the reference sets
    :return: the difference factor
    """
    try:
        return len(target.difference(set.union(*reference))) / len(target)
    except TypeError:  # if reference is empty
        return 1.0


def facet_metadata(
    target_labels: list, reference_labels: list, base: int = None
) -> Union[float, None]:
    """
    compute the change in attribute entropy between a target set and a reference set

    :param target_labels: the labels of the target set
    :param reference_labels: the labels of the reference sets (a list of lists)
    :param base: the base of the logarithm
    :return: the change in attribute entropy
    """
    base = e if base is None else base
    try:
        target_entropy = _normalized_shannon_entropy(target_labels, base)
    except ZeroDivisionError:
        target_entropy = 0

    reference_entropy = 0
    if len(reference_labels) > 0:
        for labels in reference_labels:
            try:
                reference_entropy += _normalized_shannon_entropy(labels, base)
            except ZeroDivisionError:
                continue

        reference_entropy /= len(reference_labels)
    else:
        return None
    return target_entropy - reference_entropy


def stability(lc: object, direction: str) -> float:
    """
    compute the temporal partition stability.
    The stability is the average of the continue events scores.

    :param lc: the lifecycle object
    :param direction: the temporal direction
    :return: the stability score

    """
    events = ea.events_all(lc)

    res = 0
    if len(events[direction]) == 0:
        return 0
    for group, event in events[direction].items():
        res += event["Continue"]
    return res / len(events[direction])


def purity(labels: list) -> Tuple[str, float]:
    """
    compute the purity of a set of labels. Purity is defined as the relative frequency
    of the most frequent attribute value

    :param labels: the list of labels
    :return: a tuple of the most frequent attribute value and its frequency
    """
    most_common_attribute, freq = Counter(labels).most_common(1)[0]
    return most_common_attribute, freq / len(labels)


def event_typicality(event_scores: dict) -> Tuple[str, float]:
    """
    compute the event's name and its typicality score.
    The typicality score is the highest score among all events scores.

    :param event_scores: a dictionary keyed by event name and valued by the event score
    :return: a tuple of the event name and its typicality score

    """
    highest_score = 0
    event = ""
    for ev, score in event_scores.items():
        if score > highest_score:
            highest_score = score
            event = ev
    return event, highest_score
